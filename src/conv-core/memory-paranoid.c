/*
 * Orion's debugging malloc(), olawlor@acm.org, 1/11/2001
 * 
 * This is a special version of malloc() and company for debugging software
 * that is suspected of overrunning or underrunning the boundaries of a
 * malloc buffer, or touching free memory.
 *
 * Detects writes before allocated region, writes after allocated region,
 * double-deletes, uninitialized reads; i.e., most heap-related crashing errors.
 * Includes a "memory_check()" routine which can check the entire heap's
 * consistency on command.
 *
 * This version of malloc() should not be linked into production software,
 * since it increases the time and memory overhead of malloc().
 */

static void memAbort(const char *err, void *ptr)
{
#if 1
/*Parallel version*/
	CmiPrintf("[%d] memory-wary> FATAL HEAP ERROR!  %s (block %p)\n",
		CmiMyPe(),err,ptr);
	CmiAbort("memory-wary> FATAL HEAP ERROR");
#else
/*Simple printf version*/
	fprintf(stderr,"memory-wary> FATAL HEAP ERROR!  %s (block %p)\n",
		err,ptr);
	fflush(stdout);fflush(stderr);
	abort();
#endif
}

/*This is the largest block we reasonably expect anyone to allocate*/
#define MAX_BLOCKSIZE (1024*1024*512)

/*
 * Struct Slot contains all of the information about a malloc buffer except
 * for the contents of its memory.
 */
struct _Slot {
/*Doubly-linked allocated block list*/
	struct _Slot *next;
	struct _Slot *prev;

/*The number of bytes of user data*/
	int  userSize;

/*A magic number field, to verify this is an actual malloc'd buffer*/
#define SLOTMAGIC 0x8402a5f5
#define SLOTMAGIC_VALLOC 0x7402a5f5
#define SLOTMAGIC_FREED 0xDEADBEEF
	int  magic;

/*Padding to detect writes before and after buffer*/
#define PADLEN 32 /*Bytes of padding at start and end of buffer*/
	char pad[PADLEN];
};
typedef struct _Slot	Slot;


#define PADFN(i) (char)(217+(i))
/*Write these padding bytes*/
static void setPad(char *pad) {
	int i;
	for (i=0;i<PADLEN;i++)
		pad[i]=PADFN(i);
}

/*The given area is uninitialized-- fill it as such. */
int memory_fill=-1;
static void fill_uninit(char *loc,int len)
{
	static int fillCount=0;
	int fill=memory_fill;
	char fillChar;
	if (fill==-1) /*Alternate zero and DE fill*/
		fill=(fillCount++)%2;
	if (fill!=0) fillChar=0xDE;
	else fillChar=0;
	memset(loc,fillChar,len);
}


/*Convert a slot to a user address*/
static char *Slot_toUser(Slot *s) {
	return ((char *)s)+sizeof(Slot);
}


/*Head of the circular slot list*/
Slot slot_first={&slot_first,&slot_first};


/********* Heap Checking ***********/

int memory_checkfreq=100; /*Check entire heap every this many malloc/frees*/
int memory_checkphase=0; /*malloc/free counter*/

/*Check these padding bytes*/
static void checkPad(char *pad,char *errMsg,void *ptr) {
	int i;
	for (i=0;i<PADLEN;i++)
		if (pad[i]!=PADFN(i)) {
			fprintf(stderr,"Corrupted data:");
			for (i=0;i<PADLEN;i++) 
				if (pad[i]!=PADFN(i))
					fprintf(stderr," %02x",(unsigned
int)(unsigned char)pad[i]);
				else fprintf(stderr," -");
			fprintf(stderr,"\n");
			memAbort(errMsg,ptr);
		}
}

/*Check if this pointer is "bad"-- not in the heap.*/
static int badPointer(Slot *p) {
	char *c=(char *)p;
	if ((c<(char *)0x1000) || (c+0x1000)<(char *)0x1000)
		return 1;
	return 0;
}

/*Check this slot for consistency*/
static void checkSlot(Slot *s) {
	char *user=Slot_toUser(s);
	if (badPointer(s))
		memAbort("Non-heap pointer passed to checkSlot",user);
	if (s->magic!=SLOTMAGIC && s->magic!=SLOTMAGIC_VALLOC)
		memAbort("Corrupted slot magic number",user);
	if (s->userSize<0)
		memAbort("Corrupted (negative) user size field",user);
	if (s->userSize>=MAX_BLOCKSIZE)
		memAbort("Corrupted (huge) user size field",user);
	if (badPointer(s->prev) || (s->prev->next!=s))
		memAbort("Corrupted back link",user);
	if (badPointer(s->next) || (s->next->prev!=s))
		memAbort("Corrupted forward link",user);

	checkPad(s->pad,"Corruption before start of block",user);
	checkPad(user+s->userSize,"Corruption after block",user);	
}

/*Check the entire heap for consistency*/
void memory_check(void)
{
	Slot *cur=slot_first.next;
	int nBlocks=0, nBytes=0;
	int nMegs,nKb;
	while (cur!=&slot_first) {
		checkSlot(cur);
		nBlocks++;
		nBytes+=cur->userSize;
		cur=cur->next;
	}
	nMegs=nBytes/(1024*1024);
	nKb=(nBytes-(nMegs*1024*1024))/1024;
	CmiPrintf("[%d] Heap checked-- clean. %d blocks / %d.%03d megs\n",
		CmiMyPe(),nBlocks,nMegs,(int)(nKb*1000.0/1024.0)); 
	memory_checkphase=0;
}

/********** Allocation/Free ***********/

/*Write a valid slot to this field*/
static void *setSlot(Slot *s,int userSize) {
	char *user=Slot_toUser(s);

/*Determine if it's time for a heap check*/
	if ((++memory_checkphase)>=memory_checkfreq) memory_check();

/*Splice into the slot list just past the head*/
	s->next=slot_first.next;
	s->prev=&slot_first;
	s->next->prev=s;
	s->prev->next=s;
	
	s->magic=SLOTMAGIC;
	s->userSize=userSize;
	setPad(s->pad); /*Padding before block*/
	fill_uninit(user,s->userSize); /*Block*/	
	setPad(user+s->userSize); /*Padding after block*/
	return (void *)user;
}

/*Delete this slot structure*/
static void freeSlot(Slot *s) {
	checkSlot(s);

/*Splice out of the slot list*/
	s->next->prev=s->prev;
	s->prev->next=s->next;
	s->prev=s->next=(Slot *)0x0F0;

	s->magic=SLOTMAGIC_FREED;
	fill_uninit(Slot_toUser(s),s->userSize);	
	s->userSize=-1;

/*Determine if it's time for a heap check*/
	if ((++memory_checkphase)>=memory_checkfreq) memory_check();
}

/*Convert a user address to a slot*/
static Slot *Slot_fmUser(void *user) {
	char *cu=(char *)user;
	Slot *s=(Slot *)(cu-sizeof(Slot));
	checkSlot(s);
	return s;
}


/********** meta_ routines ***********/

/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"


/*Return the system page size*/
static int meta_getpagesize(void)
{
	static int cache=0;
#if defined(CMK_GETPAGESIZE_AVAILABLE)
	if (cache==0) cache=getpagesize();
#else
	if (cache==0) cache=8192;
#endif
	return cache;
}

static void meta_init(char **argv)
{
  CmiPrintf("[%d] Converse -memory mode: paranoid\n",CmiMyPe());
}

static void *meta_malloc(size_t size)
{
  Slot *s=(Slot *)mm_malloc(sizeof(Slot)+size+PADLEN);
  if (s==NULL) return s;
  return setSlot(s,size);
}

static void meta_free(void *mem)
{
  Slot *s;
  if (mem==NULL) return; /*Legal, but misleading*/
  if (badPointer((Slot *)mem))
    memAbort("Free'd near-NULL block",mem);

  s=((Slot *)mem)-1;
  if (s->magic==SLOTMAGIC_VALLOC)
  { /*Allocated with special alignment*/
    freeSlot(s);
    mm_free(((char *)mem)-meta_getpagesize());
  }
  else if (s->magic==SLOTMAGIC) 
  { /*Ordinary allocated block */
    freeSlot(s);
    mm_free(s);
  }
  else if (s->magic==SLOTMAGIC_FREED)
    memAbort("Free'd block twice",mem);
  else /*Unknown magic number*/
    memAbort("Free'd non-malloc'd block",mem);
}

static void *meta_calloc(size_t nelem, size_t size)
{
  void *area=meta_malloc(nelem*size);
  memset(area,0,nelem*size);
  return area;
}

static void meta_cfree(void *mem)
{
  meta_free(mem);
}

static void *meta_realloc(void *oldBuffer, size_t newSize)
{
  void *newBuffer = meta_malloc(newSize);
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
  /*Allocate a whole extra page for our slot structure*/
  char *alloc=(char *)mm_memalign(align,meta_getpagesize()+size+PADLEN);
  Slot *s=(Slot *)(alloc+meta_getpagesize()-sizeof(Slot));  
  void *user=setSlot(s,size);
  s->magic=SLOTMAGIC_VALLOC;
  return user;  
}
static void *meta_valloc(size_t size)
{
  return meta_memalign(meta_getpagesize(),size);
}
