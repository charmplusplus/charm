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

/* Utilities needed by the code */
#include "ckhashtable.h"

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

#define FLAGS_MASK        0xF
#define LEAK_FLAG         0x8
#define UNKNOWN_TYPE      0x0
#define SYSTEM_TYPE       0x1
#define USER_TYPE         0x2
#define CHARE_TYPE        0x3
#define MESSAGE_TYPE      0x4
  /* A magic number field, to verify this is an actual malloc'd buffer, and what
     type of allocation it is. The last 4 bits of the magic number are used to
     define a classification of mallocs. */
#define SLOTMAGIC            0x8402a5e0
#define SLOTMAGIC_VALLOC     0x7402a5e0
#define SLOTMAGIC_FREED      0xDEADBEEF
  int  magic;

  /* Controls the number of stack frames to print out. Should be always odd, so
     the total size of this struct becomes multiple of 8 bytes everywhere */
#define STACK_LEN 15
  void *from[STACK_LEN];

  /* Pointer to extra stacktrace, when the user requested more trace */
  SlotStack *extraStack;
};

struct _SlotStack {
  char *protectedMemory;
  int protectedMemoryLength;
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

static int isLeakSlot(Slot *s) {
  return s->magic & LEAK_FLAG;
}

static void printSlot(Slot *s) {
  CmiPrintf("[%d] Leaked block of %d bytes at %p:\n",
	    CmiMyPe(), s->userSize, SlotToUser(s));
  CmiBacktracePrint(s->from,STACK_LEN);
}

/********* Circural list of allocated memory *********/

/* First memory slot */
Slot slot_first_storage = {&slot_first_storage, &slot_first_storage};
Slot *slot_first = &slot_first_storage;

/********* Cpd routines for pupping data to the debugger *********/

int cpd_memory_length(void *lenParam) {
  int n=0;
  Slot *cur = slot_first->next;
  while (cur != slot_first) {
    n++;
    cur = cur->next;
  }
  return n;
}

void cpd_memory_single_pup(Slot* list, pup_er p) {
  Slot *cur = list->next;
  /* Stupid hack to avoid sending the memory we just allocated for this packing,
     otherwise the lenghts will mismatch */
  if (pup_isPacking(p)) cur = cur->next;
  while (cur != list) {
    int i;
    int flags;
    // FIXME: this is bad since it consider only 32 bit machines!!!
    unsigned int userData = (cur+1);
    CpdListBeginItem(p, 0);
    pup_comment(p, "loc");
    pup_uint(p, &userData);
    pup_comment(p, "size");
    pup_int(p, &cur->userSize);
    pup_comment(p, "flags");
    flags = cur->magic & FLAGS_MASK;
    pup_int(p, &flags);
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
  CpdListBeginItem(p, 0);
  pup_comment(p, "name");
  pup_chars(p, "memory", strlen("memory"));
  pup_comment(p, "slots");
  pup_syncComment(p, pup_sync_begin_array, 0);
  cpd_memory_single_pup(slot_first, p);
  pup_syncComment(p, pup_sync_end_array, 0);
}

void check_memory_leaks(CpdListItemsRequest *);
void cpd_memory_leak(void *iterParam, pup_er p, CpdListItemsRequest *req) {
  if (pup_isSizing(p)) {
    // let's perform the memory leak checking. This is the first step in the
    // packing, where we size, in the second step we pack and we avoid doing
    // this check again.
    check_memory_leaks(req);
  }
  cpd_memory_pup(iterParam, p, req);
}

/********* Heap Checking ***********/

// FIXME: this function assumes that all memory is allocated in slot_unknown!
void check_memory_leaks(CpdListItemsRequest *req) {
  // Step 1)
  // index all memory into a CkHashtable, with a scan of 4 bytes.
  CkHashtable_c table;
  Slot leaking, inProgress;
  Slot *sl, **fnd, *found;
  char *scanner;
  int begin_stack, end_stack;
  int begin_data, end_data;
  int begin_bss, end_bss;
  int growing_dimension = 0;

  // copy all the memory from "slot_first" to "leaking"
  slot_first->next->prev = &leaking;
  slot_first->prev->next = &leaking;
  leaking.prev = slot_first->prev;
  leaking.next = slot_first->next;
  slot_first->next = slot_first;
  slot_first->prev = slot_first;

  table = CkCreateHashtable_pointer(sizeof(char *), 10000);
  for (sl = leaking.next; sl != &leaking; sl = sl->next) {
    // index the i-th memory slot
    char *ptr;
    sl->magic |= LEAK_FLAG;
    for (ptr = ((char*)sl)+sizeof(Slot); ptr < ((char*)sl)+sizeof(Slot)+sl->userSize; ptr+=sizeof(char*)) {
      //printf("memory %p\n",ptr);
      //growing_dimension++;
      //if ((growing_dimension&0xFF) == 0) printf("inserted %d objects\n",growing_dimension);
      char **object = (char**)CkHashtablePut(table, &ptr);
      *object = (char*)sl;
    }
  }

  // Step 2)
  // start the check with the stack and the global data. The stack is found
  // through the current pointer, going up until 16 bits filling (considering
  // the stack grows toward decreasing addresses). The pointers to the global
  // data (segments .data and .bss) are passed in with "req" as the "extra"
  // field, with the structure "begin .data", "end .data", "begin .bss", "end .bss".
  inProgress.prev = &inProgress;
  inProgress.next = &inProgress;
  begin_stack = &table;
  end_stack = begin_stack | 0xFFFFF - sizeof(char*) + 1;
  if (req->extraLen != 4*sizeof(char*)) {
    CmiPrintf("requested for a memory leak check with wrong information! %d bytes\n",req->extraLen);
  }
  if (sizeof(char*) == 4) {
    /* 32 bit addresses */
    begin_data = ntohl(((char**)(req->extra))[0]);
    end_data = ntohl(((char**)(req->extra))[1]) - sizeof(char*) + 1;
    begin_bss = ntohl(((char**)(req->extra))[2]);
    end_bss = ntohl(((char**)(req->extra))[3]) - sizeof(char*) + 1;
  } else {
    /* 64 bit addresses */
    CmiAbort("not ready yet");
    begin_data = ntohl(((char**)(req->extra))[0]);
    end_data = ntohl(((char**)(req->extra))[1]) - sizeof(char*) + 1;
    begin_bss = ntohl(((char**)(req->extra))[2]);
    end_bss = ntohl(((char**)(req->extra))[3]) - sizeof(char*) + 1;
  }
  //printf("scanning stack from %p to %p\n",begin_stack,end_stack);
  for (scanner = begin_stack; scanner < end_stack; scanner+=2) {
    fnd = (Slot**)CkHashtableGet(table, scanner);
    //if (fnd != NULL) printf("scanning stack %p, %d\n",*fnd,isLeakSlot(*fnd));
    if (fnd != NULL && isLeakSlot(*fnd)) {
      found = *fnd;
      /* mark slot as not leak */
      //printf("stack pointing to %p\n",found+1);
      found->magic &= ~LEAK_FLAG;
      /* move the slot a inProgress */
      found->next->prev = found->prev;
      found->prev->next = found->next;
      found->next = inProgress.next;
      found->prev = &inProgress;
      found->next->prev = found;
      found->prev->next = found;
    }
  }
  //printf("scanning data from %p to %p\n",begin_data,end_data);
  for (scanner = begin_data; scanner < end_data; scanner+=2) {
    fnd = (Slot**)CkHashtableGet(table, scanner);
    //if (fnd != NULL) printf("scanning data %p, %d\n",*fnd,isLeakSlot(*fnd));
    if (fnd != NULL && isLeakSlot(*fnd)) {
      found = *fnd;
      /* mark slot as not leak */
      //printf("data pointing to %p\n",found+1);
      found->magic &= ~LEAK_FLAG;
      /* move the slot a inProgress */
      found->next->prev = found->prev;
      found->prev->next = found->next;
      found->next = inProgress.next;
      found->prev = &inProgress;
      found->next->prev = found;
      found->prev->next = found;
    }
  }
  //printf("scanning bss from %p to %p\n",begin_bss,end_bss);
  for (scanner = begin_bss; scanner < end_bss; scanner+=2) {
    //printf("bss: %p %p\n",scanner,*(char**)scanner);
    fnd = (Slot**)CkHashtableGet(table, scanner);
    //if (fnd != NULL) printf("scanning bss %p, %d\n",*fnd,isLeakSlot(*fnd));
    if (fnd != NULL && isLeakSlot(*fnd)) {
      found = *fnd;
      /* mark slot as not leak */
      //printf("bss pointing to %p\n",found+1);
      found->magic &= ~LEAK_FLAG;
      /* move the slot a inProgress */
      found->next->prev = found->prev;
      found->prev->next = found->next;
      found->next = inProgress.next;
      found->prev = &inProgress;
      found->next->prev = found;
      found->prev->next = found;
    }
  }

  // Step 3)
  // continue iteratively to check the memory by sweeping it with the
  // "inProcess" list
  while (inProgress.next != &inProgress) {
    sl = inProgress.next;
    //printf("scanning memory %p\n",sl);
    /* move slot back to the main list (slot_first) */
    sl->next->prev = sl->prev;
    sl->prev->next = sl->next;
    sl->next = slot_first->next;
    sl->prev = slot_first;
    sl->next->prev = sl;
    sl->prev->next = sl;
    /* scan through this memory and pick all the slots which are still leaking
       and add them to the inProgress list */
    if (sl->extraStack != NULL && sl->extraStack->protectedMemory != NULL) mprotect(sl->extraStack->protectedMemory, sl->extraStack->protectedMemoryLength, PROT_READ);
    for (scanner = ((char*)sl)+sizeof(Slot); scanner < ((char*)sl)+sizeof(Slot)+sl->userSize-sizeof(char*)+1; scanner+=sizeof(char*)) {
      fnd = (Slot**)CkHashtableGet(table, scanner);
      //if (fnd != NULL) printf("scanning heap %p, %d\n",*fnd,isLeakSlot(*fnd));
      if (fnd != NULL && isLeakSlot(*fnd)) {
	found = *fnd;
	/* mark slot as not leak */
	//printf("heap pointing to %p\n",found+1);
	found->magic &= ~LEAK_FLAG;
	/* move the slot a inProgress */
	found->next->prev = found->prev;
	found->prev->next = found->next;
	found->next = inProgress.next;
	found->prev = &inProgress;
	found->next->prev = found;
	found->prev->next = found;
      }
    }
    if (sl->extraStack != NULL && sl->extraStack->protectedMemory != NULL) mprotect(sl->extraStack->protectedMemory, sl->extraStack->protectedMemoryLength, PROT_NONE);
  }

  // Step 4)
  // move back all the entries in leaking to slot_first
  if (leaking.next != &leaking) {
    leaking.next->prev = slot_first;
    leaking.prev->next = slot_first->next;
    slot_first->next->prev = leaking.prev;
    slot_first->next = leaking.next;
  }


  // mark all the entries in the leaking list as leak, and put them back
  // into the main list
  /*sl = leaking.next;
  while (sl != &leaking) {
    sl->magic | LEAK_FLAG;
  }
  if (leaking.next != &leaking) {
    slot_first->next->prev = leaking.prev;
    leaking.prev->next = slot_first->next;
    leaking.next->prev = slot_first;
    slot_first->next = leaking.next;
  }  
  */

  CkDeleteHashtable(table);
}

/*

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

static int memoryTraceDisabled = 0;

/* Write a valid slot to this field */
static void *setSlot(Slot *s,int userSize) {
  char *user=SlotToUser(s);
  
  /* Splice into the slot list just past the head */
  s->next=slot_first->next;
  s->prev=slot_first;
  s->next->prev=s;
  s->prev->next=s;
  
  /* Set the last 4 bits of magic to classify the memory together with the magic */
  s->magic=SLOTMAGIC + (memory_status_info>0? USER_TYPE : SYSTEM_TYPE);
  //if (memory_status_info>0) printf("user allocation\n");
  s->userSize=userSize;
  s->extraStack=(SlotStack *)0;
  {
    int i;
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
      s->from[4] = (void*)6;
    }
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
  if ((s->magic&~FLAGS_MASK)==SLOTMAGIC_VALLOC)
  { /*Allocated with special alignment*/
    freeSlot(s);
    mm_free(s->extraStack);
    /*mm_free(((char *)mem)-meta_getpagesize());*/
  }
  else if ((s->magic&~FLAGS_MASK)==SLOTMAGIC) 
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
  s->magic = SLOTMAGIC_VALLOC + (s->magic&0xF);
  s->extraStack = (SlotStack *)alloc; /* use the extra space as stack */
  s->extraStack->protectedMemory = NULL;
  s->extraStack->protectedMemoryLength = 0;
  return user;  
}

static void *meta_valloc(size_t size) {
  return meta_memalign(meta_getpagesize(),size);
}

void setProtection(char* mem, char *ptr, int len, int flag) {
  Slot *sl = (Slot*)(mem-sizeof(Slot));
  if (sl->extraStack == NULL) CmiAbort("Tried to protect memory not memaligned\n");
  if (flag != 0) {
    sl->extraStack->protectedMemory = ptr;
    sl->extraStack->protectedMemoryLength = len;
  } else {
    sl->extraStack->protectedMemory = NULL;
    sl->extraStack->protectedMemoryLength = 0;
  }
}

void setMemoryTypeChare(void *ptr) {
  Slot *sl = UserToSlot(ptr);
  sl->magic = (sl->magic & ~FLAGS_MASK) | CHARE_TYPE;
}

/* The input parameter is the pointer to the envelope, after the CmiChunkHeader */
void setMemoryTypeMessage(void *ptr) {
  void *realptr = (char*)ptr - sizeof(CmiChunkHeader);
  Slot *sl = UserToSlot(realptr);
  if ((sl->magic&~FLAGS_MASK) == SLOTMAGIC || (sl->magic&~FLAGS_MASK) == SLOTMAGIC_VALLOC) {
    sl->magic = (sl->magic & ~FLAGS_MASK) | MESSAGE_TYPE;
  }
}
