
#ifndef __L2_ATOMIC_QUEUE__
#define __L2_ATOMIC_QUEUE__

#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "spi/include/l2/atomic.h"
#include "spi/include/l1p/flush.h"
#include "pcqueue.h"

#define DEFAULT_SIZE         1024
#define L2_ATOMIC_FULL        0x8000000000000000UL
#define L2_ATOMIC_EMPTY       0x8000000000000000UL

typedef  void* L2AtomicQueueElement;

typedef struct _l2atomicstate {
  volatile uint64_t Consumer;	// not used atomically
  volatile uint64_t Producer;
  volatile uint64_t UpperBound;
  volatile uint64_t Flush;	// contents not used
} L2AtomicState;

typedef struct _l2atomicq {
  L2AtomicState               * _l2state;
  volatile void * volatile    * _array;
  PCQueue                       _overflowQ;
  pthread_mutex_t               _overflowMutex;
} L2AtomicQueue;

void L2AtomicQueueInit(void *l2mem, size_t l2memsize, L2AtomicQueue *queue) {
  pami_result_t rc;
  
  //Verify counter array is 64-byte aligned 
  assert( (((uintptr_t) l2mem) & (0x1F)) == 0 );  
  assert (sizeof(L2AtomicState) <= l2memsize);
  
  queue->_l2state = (L2AtomicState *)l2mem;
  pthread_mutex_init(&queue->_overflowMutex, NULL);
  queue->_overflowQ = PCQueueCreate();
  L2_AtomicStore(&queue->_l2state->Consumer, 0);
  L2_AtomicStore(&queue->_l2state->Producer, 0);
  L2_AtomicStore(&queue->_l2state->UpperBound, DEFAULT_SIZE);
  
  rc = posix_memalign ((void **)&queue->_array,
		       64, /*L1 line size for BG/Q */
		       sizeof(L2AtomicQueueElement) * DEFAULT_SIZE);

  assert(rc == PAMI_SUCCESS);
  memset((void*)queue->_array, 0, sizeof(L2AtomicQueueElement)*DEFAULT_SIZE);
}

void L2AtomicEnqueue (L2AtomicQueue          * queue,
		      void                   * element) 
{
  //fprintf(stderr,"Insert message %p\n", element);

  uint64_t index = L2_AtomicLoadIncrementBounded(&queue->_l2state->Producer);
  ppc_msync();
  if (index != L2_ATOMIC_FULL) {
    queue->_array[index & (DEFAULT_SIZE-1)] = element;
    return;
  }
  
  pthread_mutex_lock(&queue->_overflowMutex);
  // must check again to avoid race
  if ((index = L2_AtomicLoadIncrementBounded(&queue->_l2state->Producer)) != L2_ATOMIC_FULL) {
    queue->_array[index & (DEFAULT_SIZE-1)] = element;
  } else {
    PCQueuePush(queue->_overflowQ, element);
  }
  pthread_mutex_unlock(&queue->_overflowMutex);
}

void * L2AtomicDequeue (L2AtomicQueue    *queue)
{
  uint64_t head, tail;
  tail = queue->_l2state->Producer;
  head = queue->_l2state->Consumer;

  volatile void *e = NULL;
  if (head < tail) {    
    e = queue->_array[head & (DEFAULT_SIZE-1)];
    while (e == NULL) 
      e = queue->_array[head & (DEFAULT_SIZE-1)];

    //fprintf(stderr,"Found message %p\n", e);

    queue->_array[head & (DEFAULT_SIZE-1)] = NULL;
    ppc_msync();

    head ++;
    queue->_l2state->Consumer = head;    
    
    if (head == tail) {
      pthread_mutex_lock(&queue->_overflowMutex);      
      if (PCQueueLength(queue->_overflowQ) == 0) {
	uint64_t n = head + DEFAULT_SIZE;
	// is atomic-store needed?
	L2_AtomicStore(&queue->_l2state->UpperBound, n);
      }
      pthread_mutex_unlock(&queue->_overflowMutex);
    }
    return (void*) e;
  }
  
  /* head == tail (head cannot be greater than tail) */
  if (PCQueueLength(queue->_overflowQ) > 0) {
    pthread_mutex_lock(&queue->_overflowMutex);      
    e = PCQueuePop (queue->_overflowQ);    
    if (PCQueueLength(queue->_overflowQ) == 0) {
      uint64_t n = head + DEFAULT_SIZE;
      // is atomic-store needed?
      L2_AtomicStore(&queue->_l2state->UpperBound, n);
    }
    pthread_mutex_unlock(&queue->_overflowMutex);      
    
    return (void *) e;
  }

  return (void *) e;
}

int L2AtomicQueueEmpty (L2AtomicQueue *queue) {
  return ( (PCQueueLength(queue->_overflowQ) == 0) &&
	   (queue->_l2state->Producer == queue->_l2state->Consumer) );
}

#endif
