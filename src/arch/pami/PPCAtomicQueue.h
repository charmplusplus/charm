
#ifndef __PPC_ATOMIC_QUEUE__
#define __PPC_ATOMIC_QUEUE__

#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "pcqueue.h"

#define DEFAULT_SIZE         2048

#define CMI_PPCQ_SUCCESS  0
#define CMI_PPCQ_EAGAIN  -1

/////////////////////////////////////////////////////
// \brief Basic atomic operations should to defined
// ppc_atomic_t : the datatype of the atomic (uint32_t or uint64_t)
// PPC_AtomicStore : store a value to the atomic counter
// PPC_AtomicLoadIncrementBounded : bounded increment
// PPC_AtomicWriteFence : a producer side write fence
// PPC_AtomicReadFence  : consumer side read fence
// PPC_AtomicCounterAllocate : allocate atomic counters
/////////////////////////////////////////////////////

#if CMK_PPC_ATOMIC_DEFAULT_IMPL
#include "default_ppcq.h"
#else
//define new ppc atomics in the pami instance directory
#include "ppc_atomicq_impl.h"
#endif

#if 0
void PPC_AtomicCounterAllocate (void **atomic_mem, size_t  atomic_memsize);
ppc_atomic_type_t PPC_AtomicLoadIncrementBounded (volatile ppc_atomic_t *counter);
void PPC_AtomicStore(volatile ppc_atomic_t *counter, ppc_atomic_type_t val);
void PPC_AtomicReadFence();
void PPC_AtomicWriteFence();
#endif

typedef  void* PPCAtomicQueueElement;

typedef struct _ppcatomicstate {
  volatile ppc_atomic_t Producer;
  volatile ppc_atomic_t UpperBound;
  char pad[32 - 2*sizeof(ppc_atomic_t)];
} PPCAtomicState;

typedef struct _ppcatomicq {
  PPCAtomicState              * _state;
  volatile void * volatile    * _array;
  volatile ppc_atomic_type_t    _consumer;
  int                           _qsize;
  int                           _useOverflowQ;
  PCQueue                       _overflowQ;   //40 byte structure
  char                          _pad[24];     //align to 64 bytes
} PPCAtomicQueue; //should be padded

void PPCAtomicQueueInit      (void            * atomic_mem,
  size_t            atomic_memsize,
  PPCAtomicQueue  * queue,
  int               use_overflow,
  int               nelem)
{
  pami_result_t rc;

  //Verify counter array is 64-byte aligned
#if CMK_BLUEGENEQ
  assert ( (((uintptr_t) atomic_mem) & (0x1F)) == 0 );
  assert (sizeof(PPCAtomicState) == 32); //all counters need to be lined up
  assert (sizeof(PPCAtomicState) <= atomic_memsize);
#endif

  queue->_useOverflowQ = use_overflow;

  int qsize = 2;
  while (qsize < nelem)
    qsize *= 2;
  queue->_qsize = qsize;

  queue->_state = (PPCAtomicState *) atomic_mem;
  queue->_overflowQ = PCQueueCreate();
  queue->_consumer = 0;
  PPC_AtomicStore(&queue->_state->Producer, 0);
  PPC_AtomicStore(&queue->_state->UpperBound, qsize);

  rc = posix_memalign ((void **)&queue->_array,
      128, /* Typical L1 line size for POWER */
      sizeof(PPCAtomicQueueElement) * qsize);

  assert(rc == PAMI_SUCCESS);
  memset((void*)queue->_array, 0, sizeof(PPCAtomicQueueElement)*qsize);
}

int PPCAtomicEnqueue (PPCAtomicQueue          * queue,
                      void                   * element)
{
  //fprintf(stderr,"Insert message %p\n", element);

  register int qsize_1 = queue->_qsize - 1;
  ppc_atomic_type_t index = PPC_AtomicLoadIncrementBounded(&queue->_state->Producer);
  PPC_AtomicWriteFence();
  if (index != CMI_PPC_ATOMIC_FAIL) {
    queue->_array[index & qsize_1] = element;
    return CMI_PPCQ_SUCCESS;
  }

  //We dont want to use the overflow queue
  if (!queue->_useOverflowQ)
    return CMI_PPCQ_EAGAIN; //Q is full, try later

  //No ordering is guaranteed if there is overflow
  PCQueuePush(queue->_overflowQ, element);

  return CMI_PPCQ_SUCCESS;
}

void * PPCAtomicDequeue (PPCAtomicQueue    *queue)
{
  ppc_atomic_type_t head, tail;
  tail = PPC_AQVal(queue->_state->Producer);
  head = queue->_consumer;
  register int qsize_1 = queue->_qsize-1;

  volatile void *e = NULL;
  if (head < tail) {
    e = queue->_array[head & qsize_1];
    if (e == NULL)
      return NULL;

    queue->_array[head & qsize_1] = NULL;
    PPC_AtomicReadFence();

    head ++;
    queue->_consumer = head;

    //Charm++ does not require message ordering
    //So we dont acquire overflow mutex here
    ppc_atomic_type_t n = head + queue->_qsize;

    //Update bound every 16 consumes
    if ((n & 0xF) == 0)
      PPC_AtomicStore(&queue->_state->UpperBound, n);
    return (void*) e;
  }

  //We dont have an overflowQ
  if (!queue->_useOverflowQ)
    return NULL;

  e = PCQueuePop (queue->_overflowQ);
  return (void *) e;
}

int PPCAtomicQueueEmpty (PPCAtomicQueue *queue) {
  return ( (PCQueueLength(queue->_overflowQ) == 0) &&
      (PPC_AQVal(queue->_state->Producer) == queue->_consumer) );
}

//spin block in the PPC atomic queue till there is a message. fail and
//return after n iterations
int PPCAtomicQueueSpinWait (PPCAtomicQueue    * queue,
                            int                n)
{
  if (!PPCAtomicQueueEmpty(queue))
    return 0;  //queue is not empty so return

  ppc_atomic_type_t head, tail;
  head = queue->_consumer;

  size_t i = n;
  do {
    tail = PPC_AQVal(queue->_state->Producer);
    i--;
  }
  //While the queue is empty and i < n
  while (head == tail && i != 0);

  return 0; //fail queue is empty
}

//spin block in the PPC atomic queue till there is a message. fail and
//return after n iterations
int PPCAtomicQueue2QSpinWait (PPCAtomicQueue    * queue0,
                              PPCAtomicQueue    * queue1,
                              int                n)
{
  if (!PPCAtomicQueueEmpty(queue0))
    return 0;  //queue0 is not empty so return

  if (!PPCAtomicQueueEmpty(queue1))
    return 0;  //queue is not empty so return

  ppc_atomic_type_t head0, tail0;
  ppc_atomic_type_t head1, tail1;

  head0 = queue0->_consumer;
  head1 = queue1->_consumer;

  size_t i = n;
  do {
    tail0 = PPC_AQVal(queue0->_state->Producer);
    tail1 = PPC_AQVal(queue1->_state->Producer);
    i --;
  } while (head0==tail0 && head1==tail1 && i!=0);

  return 0;
}

#endif
