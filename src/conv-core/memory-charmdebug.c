/*
 * Filippo's charm debug memory module, gioachin@uiuc.edu, 2005/10
 * based on Orion's memory-leak.c
 *
 * This special version of malloc() and company is meant to be used in
 * conjunction with the parallel debugger CharmDebug.
 *
 * Functionalities provided:
 * - detect multiple delete on a pointer
 * - stacktrace for all memory allocated
 * - division of the memory in differnt types of allocations
 * - sweep of the memory searching for leaks
 */

/* Use Gnumalloc as meta-meta malloc fallbacks (mm_*) */
#include "memory-gnu.c"

/*#include "pup_c.h" */

typedef struct _Slot Slot;
typedef struct _SlotStack SlotStack;

/**
 * Struct Slot contains all of the information about a malloc buffer except
 * for the contents of its memory.
 */
struct _Slot {
  /*Doubly-linked allocated block list*/
  Slot *next;
  Slot *prev;
	
  /*The number of bytes of user data*/
  int userSize;

  /*A magic number field, to verify this is an actual malloc'd buffer, and what
    type of allocation it is*/
#define SLOTMAGIC            0x8402a5f5
#define SLOTMAGIC_VALLOC     0x7402a5f5
#define SLOTMAGIC_FREED      0xDEADBEEF
  int  magic;

  /* Controls the number of stack frames to print out. Should be always odd, so
     the total size of this struct becomes multiple of 8 bytes everywhere */
#define STACK_LEN 15
  void *from[STACK_LEN];

  /* Pointer to extra stacktrace, when the user requested more trace */
  void *extraStack;
};

struct _SlotStack {
  /* empty for the moment, to be filled when needed */
};

/*Convert a slot to a user address*/
static char *SlotToUser(Slot *s) {
  return ((char *)s)+sizeof(Slot);
}

/*Convert a user address to a slot*/
static Slot *UserToSlot(void *user) {
  char *cu=(char *)user;
  Slot *s=(Slot *)(cu-sizeof(Slot));
  return s;
}

static void printSlot(Slot *s) {
  CmiPrintf("[%d] Leaked block of %d bytes at %p:\n",
	    CmiMyPe(), s->userSize, SlotToUser(s));
  CmiBacktracePrint(s->from,STACK_LEN);
}

/********* Circural lists of allocated memory *********/

/* Charm objects (chares) */
Slot slot_objects_storage = {&slot_objects_storage, &slot_objects_storage};
Slot *slot_objects = &slot_objects_storage;

/* Charm messages */
Slot slot_messages_storage = {&slot_messages_storage, &slot_messages_storage};
Slot *slot_messages = &slot_messages_storage;

/* Converse CmiAlloc'ed memory */
Slot slot_converse_storage = {&slot_converse_storage, &slot_converse_storage};
Slot *slot_converse = &slot_converse_storage;

/* Memory allocated for system purposes */
Slot slot_system_storage = {&slot_system_storage, &slot_system_storage};
Slot *slot_system = &slot_system_storage;

/* Memory allocated by the user */
Slot slot_user_storage = {&slot_user_storage, &slot_user_storage};
Slot *slot_user = &slot_user_storage;

/* Unknown memory */
Slot slot_unknown_storage = {&slot_unknown_storage, &slot_unknown_storage};
Slot *slot_unknown = &slot_unknown_storage;

#define NUMBER_OF_MEMORY_SLOTS   6
Slot *slot_first[NUMBER_OF_MEMORY_SLOTS] = {&slot_objects_storage,
					    &slot_messages_storage,
					    &slot_converse_storage,
					    &slot_system_storage,
					    &slot_user_storage,
					    &slot_unknown_storage};
char *slot_name[NUMBER_OF_MEMORY_SLOTS] = {"objects", "messages", "converse",
					   "system", "user", "unknown"};

/********* Cpd routines for pupping data to the debugger *********/

int cpd_memory_single_length(int i) {
  int n=0;
  Slot *first = slot_first[i];
  Slot *cur = first->next;
  while (cur != first) {
    n++;
    cur = cur->next;
  }
  return n;
}

int cpd_memory_length(void *lenParam) {
  int i, n=0;
  for (i=0; i<NUMBER_OF_MEMORY_SLOTS; ++i) {
    n += cpd_memory_single_length(i);
  }
  return n;
}

void cpd_memory_single_pup(Slot* list, pup_er p) {
  Slot *cur = list->next;
  /* Stupid hack to avoid sending the memory we just allocated for this packing,
     otherwise the lenghts will mismatch */
  if (list==slot_first[NUMBER_OF_MEMORY_SLOTS-1] && pup_isPacking(p))
    cur = cur->next;
  while (cur != list) {
    int i;
    unsigned int userData = (cur+1);
    CpdListBeginItem(p, 0);
    pup_comment(p, "loc");
    pup_uint(p, &userData);
    pup_comment(p, "size");
    pup_int(p, &cur->userSize);
    pup_comment(p, "stack");
    for (i=0; i<STACK_LEN; ++i) {
      if (cur->from[i] <= 0) break;
      //      if (cur->from[i] > 0) pup_uint(p, (unsigned int*)&cur->from[i]);
      //      else break;
    }
    pup_uints(p, (unsigned int*)cur->from, i);
    cur = cur->next;
  }
}

void cpd_memory_pup(void *itemParam, pup_er p, CpdListItemsRequest *req) {
  int i;
  for (i=0; i<NUMBER_OF_MEMORY_SLOTS; ++i) {
    CpdListBeginItem(p, 0);
    pup_comment(p, "name");
    pup_chars(p, slot_name[i], strlen(slot_name[i]));
    pup_comment(p, "slots");
    pup_syncComment(p, pup_sync_begin_array, 0);
    cpd_memory_single_pup(slot_first[i], p);
    pup_syncComment(p, pup_sync_end_array, 0);
  }
}


/*
/ ********* Heap Checking *********** /

/ *Head of the current circular allocated block list* /
Slot slot_first_storage={&slot_first_storage,&slot_first_storage};
Slot *slot_first=&slot_first_storage;

#define CMI_MEMORY_ROUTINES 1

/ * Mark all allocated memory as being OK * /
void CmiMemoryMark(void) {
	CmiMemLock();
	/ * Just make a new linked list of slots * /
	slot_first=(Slot *)mm_malloc(sizeof(Slot));
	slot_first->next=slot_first->prev=slot_first;
	CmiMemUnlock();
}

/ * Mark this allocated block as being OK * /
void CmiMemoryMarkBlock(void *blk) {
	Slot *s=Slot_fmUser(blk);
	CmiMemLock();
	if (s->magic!=SLOTMAGIC) CmiAbort("CmiMemoryMarkBlock called on non-malloc'd block!\n");
	/ * Splice us out of the current linked list * /
	s->next->prev=s->prev;
	s->prev->next=s->next;
	s->prev=s->next=s; / * put us in our own list * /
	CmiMemUnlock();
}

/ * Print out all allocated memory * /
void CmiMemorySweep(const char *where) {
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
		/ * CmiAbort("Memory leaks detected!\n"); * /
	}
	CmiMemUnlock();
	CmiMemoryMark();
}
void CmiMemoryCheck(void) {}
*/


/********** Allocation/Free ***********/

/* Write a valid slot to this field */
static void *setSlot(Slot *s,int userSize) {
  char *user=SlotToUser(s);
  
  /* Splice into the slot list just past the head */
  s->next=slot_unknown->next;
  s->prev=slot_unknown;
  s->next->prev=s;
  s->prev->next=s;
  
  s->magic=SLOTMAGIC;
  s->userSize=userSize;
  s->extraStack=(SlotStack *)0;
  {
    int i;
    int n=STACK_LEN;
    CmiBacktraceRecord(s->from,3,&n);
    for (i=n; i<STACK_LEN; ++i) s->from[i] = 0;
  }
  return (void *)user;
}

/* Delete this slot structure */
static void freeSlot(Slot *s) {
  /* Splice out of the slot list */
  s->next->prev=s->prev;
  s->prev->next=s->next;
  s->prev=s->next=(Slot *)0;//0x0F00; why was it not 0?
  
  s->magic=SLOTMAGIC_FREED;
  s->userSize=-1;
}

/********** meta_ routines ***********/

/* Return the system page size */
static int meta_getpagesize(void) {
  static int cache=0;
#if defined(CMK_GETPAGESIZE_AVAILABLE)
  if (cache==0) cache=getpagesize();
#else
  if (cache==0) cache=8192;
#endif
  return cache;
}

/* Only display startup status messages from processor 0 */
static void status(char *msg) {
  if (CmiMyPe()==0 && !CmiArgGivingUsage()) {
    CmiPrintf("%s",msg);
  }
}

static void meta_init(char **argv) {
  status("Converse -memory mode: charmdebug\n");
}

static void *meta_malloc(size_t size) {
  Slot *s=(Slot *)mm_malloc(sizeof(Slot)+size);
  if (s==NULL) return s;
  return setSlot(s,size);
}

static void meta_free(void *mem) {
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

static void *meta_calloc(size_t nelem, size_t size) {
  void *area=meta_malloc(nelem*size);
  memset(area,0,nelem*size);
  return area;
}

static void meta_cfree(void *mem) {
  meta_free(mem);
}

static void *meta_realloc(void *oldBuffer, size_t newSize) {
  void *newBuffer = meta_malloc(newSize);
  if ( newBuffer && oldBuffer ) {
    /*Preserve old buffer contents*/
    Slot *o=UserToSlot(oldBuffer);
    size_t size=o->userSize;
    if (size>newSize) size=newSize;
    if (size > 0)
      memcpy(newBuffer, oldBuffer, size);

    meta_free(oldBuffer);
  }
  return newBuffer;
}

static void *meta_memalign(size_t align, size_t size) {
  int overhead = align;
  while (overhead < sizeof(Slot)+sizeof(SlotStack)) overhead += align;
  /* Allocate the required size + the overhead needed to keep the user alignment */

  char *alloc=(char *)mm_memalign(align,overhead+size);
  Slot *s=(Slot *)(alloc+overhead-sizeof(Slot));  
  void *user=setSlot(s,size);
  s->magic=SLOTMAGIC_VALLOC;
  s->extraStack = (SlotStack *)alloc; /* use the extra space as stack */
  return user;  
}

static void *meta_valloc(size_t size) {
  return meta_memalign(meta_getpagesize(),size);
}
