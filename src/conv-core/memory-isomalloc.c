/******************************************************************************

A migratable memory allocator.

FIXME: isomalloc is threadsafe, so the isomallocs *don't* need to
be wrapped in CmiMemLock.  (Doesn't hurt, tho')

*****************************************************************************/

/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"

#include "memory-isomalloc.h"

static int memInit=0;

struct CmiIsomallocBlockList_tag {
	/*Prev and next form a circular doubly-linked list of blocks*/
	struct CmiIsomallocBlockList_tag *prev,*next;
	CmiIsomallocBlock block; /*So we can isofree this block*/
	unsigned int userSize; /*Bytes of user data in this block*/
# define MAGIC1 0xcabba7e0
	int magic; /*Magic number (to detect corruption & pad block)*/
	/*actual data of block follows here...*/
};

typedef struct CmiIsomallocBlockList_tag Slot;

/*Convert a slot to a user address*/
static char *Slot_toUser(Slot *s) {return (char *)(s+1);}
static Slot *Slot_fmUser(void *s) {return ((Slot *)s)-1;}

/*The current allocation arena */
CpvStaticDeclare(Slot *,isomalloc_blocklist);

#define ISOMALLOC_PUSH \
	Slot *isomalloc_blocklist=CpvAccess(isomalloc_blocklist);\
	CpvAccess(isomalloc_blocklist)=NULL;\
	rank_holding_CmiMemLock=CmiMyRank();\

#define ISOMALLOC_POP \
	CpvAccess(isomalloc_blocklist)=isomalloc_blocklist;\
	rank_holding_CmiMemLock=-1;\

static void meta_init(char **argv)
{
   CpvInitialize(Slot *,isomalloc_blocklist);
}

static void *meta_malloc(size_t size)
{
	void *ret=NULL;
	if (CpvInitialized(isomalloc_blocklist) && CpvAccess(isomalloc_blocklist)) 
	{ /*Isomalloc a new block and link it in*/
		CmiIsomallocBlock blk;
		Slot *n; /*Newly created slot*/
		ISOMALLOC_PUSH /*Disable isomalloc while inside isomalloc*/
		n=(Slot *)CmiIsomalloc(sizeof(Slot)+size,&blk);
		ISOMALLOC_POP
		n->block=blk;
		n->userSize=size;
#ifndef CMK_OPTIMIZE
		n->magic=MAGIC1;
#endif
		n->prev=isomalloc_blocklist;
		n->next=isomalloc_blocklist->next;
		isomalloc_blocklist->next->prev=n;
		isomalloc_blocklist->next=n;
		ret=Slot_toUser(n);
		CmiPrintf("Isomalloc'd %p: %d\n",ret,size);
	}
	else /*Just use regular malloc*/
		ret=mm_malloc(size);
	return ret;
}

static void meta_free(void *mem)
{
	if (CmiIsomallocInRange(mem)) 
	{ /*Unlink this slot and isofree*/
		Slot *n=Slot_fmUser(mem);
		CmiIsomallocBlock blk=n->block;
		ISOMALLOC_PUSH
#ifndef CMK_OPTIMIZE
		if (n->magic!=MAGIC1) 
			CmiAbort("Heap corruption detected!  Run with ++debug to find out hwere");
#endif
		CmiPrintf("Isofree'd %p\n",mem);
		n->prev->next=n->next;
		n->next->prev=n->prev;
		CmiIsomallocFree(&blk);
		ISOMALLOC_POP
	}
	else /*Just use regular malloc*/
		mm_free(mem);
}

static void *meta_calloc(size_t nelem, size_t size)
{
	void *ret=meta_malloc(nelem*size);
	memset(ret,0,nelem*size);
	return ret;
}

static void meta_cfree(void *mem)
{
	meta_free(mem);
}

static void *meta_realloc(void *oldBuffer, size_t newSize)
{
	void *newBuffer;
	/*Just forget it for regular malloc's:*/
	if (!CmiIsomallocInRange(oldBuffer)) 
		return mm_realloc(oldBuffer,newSize);
	
	newBuffer = meta_malloc(newSize);
	if ( newBuffer && oldBuffer ) {
		/*Preserve old buffer contents*/
		Slot *o=Slot_fmUser(oldBuffer);
		size_t size=o->userSize;
		if (size<newSize) size=newSize;
		if (size > 0)
			memcpy(newBuffer, oldBuffer, size);
	}
	if (oldBuffer)
		meta_free(oldBuffer);
	return newBuffer;
}

static void *meta_memalign(size_t align, size_t size)
{
	return meta_malloc(size);
}

static void *meta_valloc(size_t size)
{
	return meta_malloc(size);
}

#define CMK_MEMORY_HAS_NOMIGRATE
/*Allocate non-migratable memory:*/
void *malloc_nomigrate(size_t size) { 
  void *result;
  CmiMemLock();
  result = mm_malloc(size);
  CmiMemUnlock();
  return result;
}

void free_nomigrate(void *mem)
{
  CmiMemLock();
  mm_free(mem);
  CmiMemUnlock();
}

#define CMK_MEMORY_HAS_ISOMALLOC
/*Build a new blockList.*/
CmiIsomallocBlockList *CmiIsomallocBlockListNew(void)
{
	CmiIsomallocBlockList *ret;
	CmiIsomallocBlock blk;
	ret=(CmiIsomallocBlockList *)CmiIsomalloc(sizeof(*ret),&blk);
	ret->next=ret; /*1-entry circular linked list*/
	ret->prev=ret;
	ret->block=blk;
	ret->userSize=0;
	ret->magic=MAGIC1;
	return ret;
}

/*Make this blockList "active"-- the recipient of incoming
mallocs.  Returns the old blocklist.*/
CmiIsomallocBlockList *CmiIsomallocBlockListActivate(CmiIsomallocBlockList *l)
{
	register Slot **s=&CpvAccess(isomalloc_blocklist);
	CmiIsomallocBlockList *ret=*s;
	*s=l;
	return ret;
}

/*Pup all the blocks in this list.  This amounts to two circular
list traversals.  Because everything's isomalloc'd, we don't even
have to restore the pointers-- they'll be restored automatically!
*/
void CmiIsomallocBlockListPup(pup_er p,CmiIsomallocBlockList **lp)
{
	CmiIsomallocBlock blk;
	int i,nBlocks=0;
	Slot *cur=NULL, *start=*lp;
#ifndef CMK_OPTIMIZE
	if (CpvAccess(isomalloc_blocklist)!=NULL)
		CmiAbort("Called CmiIsomallocBlockListPup while a blockList is active!\n"
			"You should swap out the active blocklist before pupping.\n");
#endif
	/*Count the number of blocks in the list*/
	if (!pup_isUnpacking(p)) {
		nBlocks=1; /*<- Since we have to skip the start block*/
		for (cur=start->next; cur!=start; cur=cur->next) 
			nBlocks++;
		/*Prepare for next trip around list:*/
		cur=start;
	}
	pup_int(p,&nBlocks);
	
	/*Pup each block in the list*/
	for (i=0;i<nBlocks;i++) {
		void *newBlock;
		if (!pup_isUnpacking(p)) {
			blk=cur->block;
			cur=cur->next;
		}
		newBlock=CmiIsomallocPup(p,&blk);
		if (i==0 && pup_isUnpacking(p))
			*lp=(Slot *)newBlock;
	}
	if (pup_isDeleting(p))
		*lp=NULL;
}

/*Delete all the blocks in this list.*/
void CmiIsomallocBlockListFree(CmiIsomallocBlockList *l)
{
	Slot *start=l;
	Slot *cur=start;
	if (cur==NULL) return; /*Already deleted*/
	do {
		Slot *doomed=cur;
		cur=cur->next; /*Have to stash next before deleting cur*/
		CmiIsomallocFree(&doomed->block);
	} while (cur!=start);
}





