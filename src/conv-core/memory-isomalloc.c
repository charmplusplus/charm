/******************************************************************************

A migratable memory allocator.

FIXME: isomalloc is threadsafe, so the isomallocs *don't* need to
be wrapped in CmiMemLock.  (Doesn't hurt, tho')

*****************************************************************************/

/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"

#include "memory-isomalloc.h"

/*The current allocation arena */
CpvStaticDeclare(CmiIsomallocBlockList *,isomalloc_blocklist);

#define ISOMALLOC_PUSH \
	CmiIsomallocBlockList *pushed_blocklist=CpvAccess(isomalloc_blocklist);\
	CpvAccess(isomalloc_blocklist)=NULL;\
	rank_holding_CmiMemLock=CmiMyRank();\

#define ISOMALLOC_POP \
	CpvAccess(isomalloc_blocklist)=pushed_blocklist;\
	rank_holding_CmiMemLock=-1;\

static void meta_init(char **argv)
{
   CpvInitialize(CmiIsomallocBlockList *,isomalloc_blocklist);
}

static void *meta_malloc(size_t size)
{
	void *ret=NULL;
	if (CpvInitialized(isomalloc_blocklist) && CpvAccess(isomalloc_blocklist)) 
	{ /*Isomalloc a new block and link it in*/
		ISOMALLOC_PUSH /*Disable isomalloc while inside isomalloc*/
		ret=CmiIsomallocBlockListMalloc(pushed_blocklist,size);
		ISOMALLOC_POP
	}
	else /*Just use regular malloc*/
		ret=mm_malloc(size);
	return ret;
}

static void meta_free(void *mem)
{
	if (CmiIsomallocInRange(mem)) 
	{ /*Unlink this slot and isofree*/
		ISOMALLOC_PUSH
		CmiIsomallocBlockListFree(mem);
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
		/*Must preserve old buffer contents, so we need the size of the
		  buffer.  SILLY HACK: muck with internals of blocklist header.*/
		size_t size=CmiIsomallocLength(((CmiIsomallocBlockList *)oldBuffer)-1)-
			sizeof(CmiIsomallocBlockList);
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

/*Make this blockList "active"-- the recipient of incoming
mallocs.  Returns the old blocklist.*/
CmiIsomallocBlockList *CmiIsomallocBlockListActivate(CmiIsomallocBlockList *l)
{
	register CmiIsomallocBlockList **s=&CpvAccess(isomalloc_blocklist);
	CmiIsomallocBlockList *ret=*s;
	*s=l;
	return ret;
}





