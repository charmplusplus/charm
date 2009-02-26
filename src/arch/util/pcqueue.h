/** @file
 * @brief Producer-Consumer Queues
 * @ingroup Machine
 *
 * This queue implementation enables a producer and a consumer to
 * communicate via a queue.  The queues are optimized for this situation,
 * they don't require any operating system locks (they do require 32-bit
 * reads and writes to be atomic.)  Cautions: there can only be one
 * producer, and one consumer.  These queues cannot store null pointers.
 *
 ****************************************************************************/

/**
 * \addtogroup Machine
 * @{
 */

#ifndef __PCQUEUE__
#define __PCQUEUE__


/*****************************************************************************
 * #define CMK_PCQUEUE_LOCK
 * PCQueue doesn't need any lock, the lock here is only
 * for debugging and testing purpose! it only make sense in smp version
 ****************************************************************************/
/*#define CMK_PCQUEUE_LOCK  1 */
#if CMK_SMP && !defined(CMK_PCQUEUE_LOCK)
/*#define PCQUEUE_MULTIQUEUE  1 */
#define CMK_PCQUEUE_PUSH_LOCK 1 
#endif

/* If we are using locks in PCQueue, we disable any other fence operation,
 * otherwise we use the ones provided by converse.h */
#ifdef CMK_PCQUEUE_LOCK
#define PCQueue_CmiMemoryReadFence()
#define PCQueue_CmiMemoryWriteFence()
#define PCQueue_CmiMemoryAtomicIncrement(someInt)  someInt=someInt+1
#define PCQueue_CmiMemoryAtomicDecrement(someInt)  someInt=someInt-1
#else
#define PCQueue_CmiMemoryReadFence                 CmiMemoryReadFence
#define PCQueue_CmiMemoryWriteFence                CmiMemoryWriteFence
#define PCQueue_CmiMemoryAtomicIncrement           CmiMemoryAtomicIncrement
#define PCQueue_CmiMemoryAtomicDecrement           CmiMemoryAtomicDecrement
#endif

#if CMK_SMP
#define CMK_SMP_volatile volatile
#else
#define CMK_SMP_volatile
#endif

#define PCQueueSize 0x100

/** This data type is at least one cache line of padding, used to avoid
 *  cache line thrashing on SMP systems.  On x86, this is just for performance;
 *  on other CPUs, this can affect your choice of fence operations.
 **/
typedef struct CmiMemorySMPSeparation_t {
        unsigned char padding[128];
} CmiMemorySMPSeparation_t;

typedef struct CircQueueStruct
{
  struct CircQueueStruct * CMK_SMP_volatile next;
  int push;
#if CMK_SMP
  CmiMemorySMPSeparation_t pad1;
#endif
  int pull;
#if CMK_SMP
  CmiMemorySMPSeparation_t pad2;
#endif
  char *data[PCQueueSize];
}
*CircQueue;

typedef struct PCQueueStruct
{
  CircQueue head;
#if CMK_SMP
  CmiMemorySMPSeparation_t pad1;
#endif
  CircQueue CMK_SMP_volatile tail;
#if CMK_SMP
  CmiMemorySMPSeparation_t pad2;
#endif
  int  len;
#if defined(CMK_PCQUEUE_LOCK) || defined(CMK_PCQUEUE_PUSH_LOCK)
  CmiNodeLock  lock;
#endif
}
*PCQueue;

/* static CircQueue Cmi_freelist_circqueuestruct = 0;
   static int freeCount = 0; */

#define FreeCircQueueStruct(dg) {\
  CircQueue d;\
  CmiMemLock();\
  d=(dg);\
  d->next = Cmi_freelist_circqueuestruct;\
  Cmi_freelist_circqueuestruct = d;\
  freeCount++;\
  CmiMemUnlock();\
}

#if !XT3_PCQUEUE_HACK
#define MallocCircQueueStruct(dg) {\
  CircQueue d;\
  CmiMemLock();\
  d = Cmi_freelist_circqueuestruct;\
  if (d==(CircQueue)0){\
    d = ((CircQueue)calloc(1, sizeof(struct CircQueueStruct))); \
  }\
  else{\
    freeCount--;\
    Cmi_freelist_circqueuestruct = d->next;\
    }\
  dg = d;\
  CmiMemUnlock();\
}
#else
#define MallocCircQueueStruct(dg) {\
  CircQueue d;\
  CmiMemLock();\
  d = Cmi_freelist_circqueuestruct;\
  if (d==(CircQueue)0){\
    d = ((CircQueue)malloc(sizeof(struct CircQueueStruct))); \
    d = ((CircQueue)memset(d, 0, sizeof(struct CircQueueStruct))); \
  }\
  else{\
    freeCount--;\
    Cmi_freelist_circqueuestruct = d->next;\
    }\
  dg = d;\
  CmiMemUnlock();\
}
#endif

static PCQueue PCQueueCreate(void)
{
  CircQueue circ;
  PCQueue Q;

  /* MallocCircQueueStruct(circ); */
#if !XT3_PCQUEUE_HACK
  circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));
#else
  circ = (CircQueue)malloc(sizeof(struct CircQueueStruct));
  circ = (CircQueue)memset(circ, 0, sizeof(struct CircQueueStruct));
#endif
  Q = (PCQueue)malloc(sizeof(struct PCQueueStruct));
  _MEMCHECK(Q);
  Q->head = circ;
  Q->tail = circ;
  Q->len = 0;
#if defined(CMK_PCQUEUE_LOCK) || defined(CMK_PCQUEUE_PUSH_LOCK)
  Q->lock = CmiCreateLock();
#endif
  return Q;
}

static void PCQueueDestroy(PCQueue Q)
{
  CircQueue circ = Q->head;
  while (circ != Q->tail) {
    free(circ);
    circ = circ->next;
  }
  free(circ);
  free(Q);
}

static int PCQueueEmpty(PCQueue Q)
{
  CircQueue circ = Q->head;
  char *data = circ->data[circ->pull];
  return (data == 0);
}

static int PCQueueLength(PCQueue Q)
{
  return Q->len;
}

static char *PCQueuePop(PCQueue Q)
{
  CircQueue circ; int pull; char *data;

#ifdef CMK_PCQUEUE_LOCK
    if (Q->len == 0) return 0;        /* If atomic increment are used, Q->len is always right */
    CmiLock(Q->lock);
#endif
    circ = Q->head;
    pull = circ->pull;
    data = circ->data[pull];

    PCQueue_CmiMemoryReadFence();

#if XT3_ONLY_PCQUEUE_WORKAROUND
    if (data && (Q->len > 0)) {
#else
    if (data) {
#endif
      circ->pull = (pull + 1);
      circ->data[pull] = 0;
      if (pull == PCQueueSize - 1) { /* just pulled the data from the last slot
                                     of this buffer */
        PCQueue_CmiMemoryReadFence();
        /*while (circ->next == 0);   This instruciton does not seem needed... but it might */
        Q->head = circ-> next; /* next buffer must exist, because "Push"  */
        CmiAssert(Q->head != NULL);

	/* FreeCircQueueStruct(circ); */
        free(circ);

	/* links in the next buffer *before* filling */
                               /* in the last slot. See below. */
      }
      PCQueue_CmiMemoryAtomicDecrement(Q->len);
#ifdef CMK_PCQUEUE_LOCK
      CmiUnlock(Q->lock);
#endif
      return data;
    }
    else { /* queue seems to be empty. The producer may be adding something
              to it, but its ok to report queue is empty. */
#ifdef CMK_PCQUEUE_LOCK
      CmiUnlock(Q->lock);
#endif
      return 0;
    }
}

static void PCQueuePush(PCQueue Q, char *data)
{
  CircQueue circ, circ1; int push;

#if defined(CMK_PCQUEUE_LOCK) || defined(CMK_PCQUEUE_PUSH_LOCK)
  CmiLock(Q->lock);
#endif
  circ1 = Q->tail;
#ifdef PCQUEUE_MULTIQUEUE
  CmiMemoryAtomicFetchAndInc(circ1->push, push);
#else
  push = circ1->push;
  circ1->push = (push + 1);
#endif
#ifdef PCQUEUE_MULTIQUEUE
  while (push >= PCQueueSize) {
    /* this circqueue is full, and we need to wait for the thread writing
 *        the last slot to allocate the new queue */
    PCQueue_CmiMemoryReadFence();
    while (Q->tail == circ1);
    circ1 = Q->tail;
    CmiMemoryAtomicFetchAndInc(circ1->push, push);
  }
#endif

  if (push == (PCQueueSize -1)) { /* last slot is about to be filled */
    /* this way, the next buffer is linked in before data is filled in
       in the last slot of this buffer */

#if !XT3_PCQUEUE_HACK
    circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));
#else
    circ = (CircQueue)malloc(sizeof(struct CircQueueStruct));
    circ = (CircQueue)memset(circ, 0, sizeof(struct CircQueueStruct));
#endif
    /* MallocCircQueueStruct(circ); */

#ifdef PCQUEUE_MULTIQUEUE
    PCQueue_CmiMemoryWriteFence();
#endif

    Q->tail->next = circ;
    Q->tail = circ;
  }
  PCQueue_CmiMemoryWriteFence();
  
  circ1->data[push] = data;
  PCQueue_CmiMemoryAtomicIncrement(Q->len);

#if defined(CMK_PCQUEUE_LOCK) || defined(CMK_PCQUEUE_PUSH_LOCK)
  CmiUnlock(Q->lock);
#endif
}

#endif

/*@}*/
