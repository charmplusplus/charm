/*
 * Orion's debugging malloc(), olawlor@acm.org, 2003/9/7
 * 
 * This is a special version of malloc() and company for debugging software
 * that is suspected of leaking memory.
 *
 * The basic usage is:
 *    1.) Do your startup
 *    2.) Call CmiMemoryMark(), which marks all allocated memory as OK.
 *    3.) Do the work you think is leaking memory.
 *    4.) Call CmiMemorySweep(const char *where), which prints out any mallocs
 *        since the last sweep or mark. The where argument is used to tag sweeps
 *        and is concatenated to the initial print statement for the sweep.
 *
 * This version of malloc() only introduces a small constant amount
 * of slowdown per malloc/free.
 */

#if ! CMK_MEMORY_BUILD_OS
/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"
#endif


typedef struct Slot Slot;
/*
 * Struct Slot contains all of the information about a malloc buffer except
 * for the contents of its memory.
 */
struct Slot {
/*Doubly-linked allocated block list*/
	Slot *next;
	Slot *prev;

/*The number of bytes of user data*/
	int userSize;

/*A magic number field, to verify this is an actual malloc'd buffer*/
#define SLOTMAGIC 0x8402a5f5
#define SLOTMAGIC_VALLOC 0x7402a5f5
#define SLOTMAGIC_FREED 0xDEADBEEF
	int  magic;

/* Controls the number of stack frames to print out */
#define STACK_LEN 9
	void *from[STACK_LEN];
};

/*Convert a slot to a user address*/
static char *Slot_toUser(Slot *s) {
	return ((char *)s)+sizeof(Slot);
}

/*Convert a user address to a slot*/
static Slot *Slot_fmUser(void *user) {
	char *cu=(char *)user;
	Slot *s=(Slot *)(cu-sizeof(Slot));
	return s;
}

static void printSlot(Slot *s) {
	CmiPrintf("[%d] Leaked block of %d bytes at %p:\n",
		CmiMyPe(), s->userSize, Slot_toUser(s));
	CmiBacktracePrint(s->from,STACK_LEN);
}

/********* Heap Checking ***********/

/*Head of the current circular allocated block list*/
Slot slot_first_storage={&slot_first_storage,&slot_first_storage};
Slot *slot_first=&slot_first_storage;

#define CMI_MEMORY_ROUTINES 1

/* Mark all allocated memory as being OK */
void CmiMemoryMark(void) {
	CmiMemLock();
	/* Just make a new linked list of slots */
	slot_first=(Slot *)mm_malloc(sizeof(Slot));
	slot_first->next=slot_first->prev=slot_first;
	CmiMemUnlock();
}

/* Mark this allocated block as being OK */
void CmiMemoryMarkBlock(void *blk) {
	Slot *s=Slot_fmUser(blk);
	CmiMemLock();
	if (s->magic!=SLOTMAGIC) CmiAbort("CmiMemoryMarkBlock called on non-malloc'd block!\n");
	/* Splice us out of the current linked list */
	s->next->prev=s->prev;
	s->prev->next=s->next;
	s->prev=s->next=s; /* put us in our own list */
	CmiMemUnlock();
}

/* Print out all allocated memory */
void CmiMemorySweep(const char *where)
{
	Slot *cur;
	int nBlocks=0,nBytes=0;
	CmiMemLock();
	cur=slot_first->next;
	CmiPrintf("[%d] ------- LEAK CHECK: %s -----\n",CmiMyPe(), where);
	while (cur!=slot_first) {
		printSlot(cur);
		nBlocks++; nBytes+=cur->userSize;
		cur=cur->next;
	}
	if (nBlocks) {
		CmiPrintf("[%d] Total leaked memory: %d blocks, %d bytes\n",
			CmiMyPe(),nBlocks,nBytes);
		/* CmiAbort("Memory leaks detected!\n"); */
	}
	CmiMemUnlock();
	CmiMemoryMark();
}
void CmiMemoryCheck(void) {}

/********** Allocation/Free ***********/

static int memoryTraceDisabled = 0;

/*Write a valid slot to this field*/
static void *setSlot(Slot *s,int userSize) {
	char *user=Slot_toUser(s);

/*Splice into the slot list just past the head*/
	s->next=slot_first->next;
	s->prev=slot_first;
	s->next->prev=s;
	s->prev->next=s;

	s->magic=SLOTMAGIC;
	s->userSize=userSize;
	{
		int n=STACK_LEN;
                if (memoryTraceDisabled==0) {
                  memoryTraceDisabled = 1;
                  CmiBacktraceRecord(s->from,3,&n);
                  memoryTraceDisabled = 0;
                } else {
                  s->from[0] = (void*)10;
                  s->from[1] = (void*)9;
                  s->from[2] = (void*)8;
                  s->from[3] = (void*)7;
                }
	}
	return (void *)user;
}

/*Delete this slot structure*/
static void freeSlot(Slot *s) {
/*Splice out of the slot list*/
	s->next->prev=s->prev;
	s->prev->next=s->next;
	s->prev=s->next=(Slot *)0x0F00;

	s->magic=SLOTMAGIC_FREED;
	s->userSize=-1;
}


/********** meta_ routines ***********/

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

/*Only display startup status messages from processor 0*/
static void status(char *msg) {
  if (CmiMyPe()==0 && !CmiArgGivingUsage()) {
    CmiPrintf("%s",msg);
  }
}
static void meta_init(char **argv)
{
  status("Converse -memory mode: leak");
  status("\n");
}

static void *meta_malloc(size_t size)
{
  Slot *s=(Slot *)mm_malloc(sizeof(Slot)+size);
  if (s==NULL) return s;
  return setSlot(s,size);
}

static void meta_free(void *mem)
{
  Slot *s;
  if (mem==NULL) return; /*Legal, but misleading*/

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
    CmiAbort("Free'd block twice");
  else /*Unknown magic number*/
    CmiAbort("Free'd non-malloc'd block");
}

static void *meta_calloc(size_t nelem, size_t size)
{
  void *area=meta_malloc(nelem*size);
  if (area != NULL) memset(area,0,nelem*size);
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
    if (size>newSize) size=newSize;
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
  char *alloc=(char *)mm_memalign(align,meta_getpagesize()+size);
  Slot *s=(Slot *)(alloc+meta_getpagesize()-sizeof(Slot));
  void *user=setSlot(s,size);
  s->magic=SLOTMAGIC_VALLOC;
  return user;
}
static void *meta_valloc(size_t size)
{
  return meta_memalign(meta_getpagesize(),size);
}
