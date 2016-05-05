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

#include "conv-config.h"

/*****************************************************************************
 * #define CMK_PCQUEUE_LOCK
 * PCQueue doesn't need any lock, the lock here is only
 * for debugging and testing purpose! it only make sense in smp version
 ****************************************************************************/
/*#define CMK_PCQUEUE_LOCK  1 */
#if CMK_SMP && !CMK_PCQUEUE_LOCK
/*#define PCQUEUE_MULTIQUEUE  1 */

#if !CMK_SMP_NO_PCQUEUE_PUSH_LOCK
#define CMK_PCQUEUE_PUSH_LOCK 1
#endif

#endif

#if CMK_SMP
#include <atomic>
#define CMK_SMP_volatile volatile
#else
#define CMK_SMP_volatile
#endif

/* If we are using locks in PCQueue, we disable any other fence operation,
 * otherwise we use the ones provided by converse.h and std::atomic */
#if !CMK_SMP || CMK_PCQUEUE_LOCK
#define PCQueue_CmiMemoryReadFence()
#define PCQueue_CmiMemoryWriteFence()
#define PCQueue_CmiMemoryAtomicIncrement(k, mem) k=k+1
#define PCQueue_CmiMemoryAtomicDecrement(k, mem) k=k-1
#define PCQueue_CmiMemoryAtomicLoad(k, mem)      k
#define PCQueue_CmiMemoryAtomicStore(k, v, mem)  k=v
#else
#define PCQueue_CmiMemoryReadFence               CmiMemoryReadFence
#define PCQueue_CmiMemoryWriteFence              CmiMemoryWriteFence
#define PCQueue_CmiMemoryAtomicIncrement(k, mem) std::atomic_fetch_add_explicit(&k, 1, mem)
#define PCQueue_CmiMemoryAtomicDecrement(k, mem) std::atomic_fetch_sub_explicit(&k, 1, mem)
#define PCQueue_CmiMemoryAtomicLoad(k, mem)      std::atomic_load_explicit(&k, mem)
#define PCQueue_CmiMemoryAtomicStore(k, v, mem)  std::atomic_store_explicit(&k, v, mem);
#endif

#define PCQueueSize 0x100

/**
 * The simple version of pcqueue has dropped the function of being
 * expanded if the queue is full. On one hand, each operation becomes simpler
 * and has fewer memory accesses. On the other hand, the simple pcqueue
 * is only for experimental usage.
 */
#if !USE_SIMPLE_PCQUEUE

typedef struct CircQueueStruct
{
  struct CircQueueStruct * CMK_SMP_volatile next;
  int push;
#if CMK_SMP
  char _pad1[CMI_CACHE_LINE_SIZE - (sizeof(struct CircQueueStruct *) + sizeof(int))]; // align to cache line
#endif
  int pull;
#if CMK_SMP
  char _pad2[CMI_CACHE_LINE_SIZE - sizeof(int)]; // align to cache line
  std::atomic<char *> data[PCQueueSize];
#else
  char *data[PCQueueSize];
#endif
}
*CircQueue;

typedef struct PCQueueStruct
{
  CircQueue head;
#if CMK_SMP
  char _pad1[CMI_CACHE_LINE_SIZE - sizeof(CircQueue)]; // align to cache line
#endif
  CircQueue CMK_SMP_volatile tail;
#if CMK_SMP
  char _pad2[CMI_CACHE_LINE_SIZE - sizeof(CircQueue)]; // align to cache line
  std::atomic<int> len;
#else
  int len;
#endif
#if CMK_PCQUEUE_LOCK || CMK_PCQUEUE_PUSH_LOCK
  CmiNodeLock  lock;
#endif
}
*PCQueue;

static PCQueue PCQueueCreate(void)
{
  CircQueue circ;
  PCQueue Q;

  circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));
  Q = (PCQueue)malloc(sizeof(struct PCQueueStruct));
  _MEMCHECK(Q);
  Q->head = circ;
  Q->tail = circ;
  Q->len = 0;
#if CMK_PCQUEUE_LOCK || CMK_PCQUEUE_PUSH_LOCK
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
  return (PCQueue_CmiMemoryAtomicLoad(Q->len, std::memory_order_acquire) == 0);
}

static int PCQueueLength(PCQueue Q)
{
  return PCQueue_CmiMemoryAtomicLoad(Q->len, std::memory_order_acquire);
}

static char *PCQueueTop(PCQueue Q)
{
  CircQueue circ; int pull; char *data;

    if (PCQueue_CmiMemoryAtomicLoad(Q->len, std::memory_order_relaxed) == 0) return 0;
#if CMK_PCQUEUE_LOCK
    CmiLock(Q->lock);
#endif
    circ = Q->head;
    pull = circ->pull;
    data = PCQueue_CmiMemoryAtomicLoad(circ->data[pull], std::memory_order_acquire);

#if CMK_PCQUEUE_LOCK
      CmiUnlock(Q->lock);
#endif
      return data;
}


static char *PCQueuePop(PCQueue Q)
{
  CircQueue circ; int pull; char *data;

    if (PCQueue_CmiMemoryAtomicLoad(Q->len, std::memory_order_relaxed) == 0) return 0;
#if CMK_PCQUEUE_LOCK
    CmiLock(Q->lock);
#endif
    circ = Q->head;
    pull = circ->pull;
    data = PCQueue_CmiMemoryAtomicLoad(circ->data[pull], std::memory_order_acquire);


    if (data) {
      circ->pull = (pull + 1);
      circ->data[pull] = 0;
      if (pull == PCQueueSize - 1) { /* just pulled the data from the last slot
                                     of this buffer */
        PCQueue_CmiMemoryReadFence();
        /*while (circ->next == 0);   This instruciton does not seem needed... but it might */
        Q->head = circ-> next; /* next buffer must exist, because "Push"  */
        CmiAssert(Q->head != NULL);

        free(circ);

	/* links in the next buffer *before* filling */
                               /* in the last slot. See below. */
      }
      PCQueue_CmiMemoryAtomicDecrement(Q->len, std::memory_order_release);
#if CMK_PCQUEUE_LOCK
      CmiUnlock(Q->lock);
#endif
      return data;
    }
    else { /* queue seems to be empty. The producer may be adding something
              to it, but its ok to report queue is empty. */
#if CMK_PCQUEUE_LOCK
      CmiUnlock(Q->lock);
#endif
      return 0;
    }
}

static void PCQueuePush(PCQueue Q, char *data)
{
  CircQueue circ, circ1; int push;

#if CMK_PCQUEUE_LOCK|| CMK_PCQUEUE_PUSH_LOCK
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

    circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));

#ifdef PCQUEUE_MULTIQUEUE
    PCQueue_CmiMemoryWriteFence();
#endif

    Q->tail->next = circ;
    Q->tail = circ;
  }

  PCQueue_CmiMemoryAtomicStore(circ1->data[push], data, std::memory_order_release);
  PCQueue_CmiMemoryAtomicIncrement(Q->len, std::memory_order_relaxed);

#if CMK_PCQUEUE_LOCK || CMK_PCQUEUE_PUSH_LOCK
  CmiUnlock(Q->lock);
#endif
}

#else

/**
 * The beginning of definitions for simple pcqueue
 */
typedef struct PCQueueStruct
{
  char **head; /*pointing to the first element*/
#if CMK_SMP
  char _pad1[CMI_CACHE_LINE_SIZE - sizeof(char**)]; // align to cache line
#endif

  //char** CMK_SMP_volatile tail; /*pointing to the last element*/
  char** tail; /*pointing to the last element*/
#if CMK_SMP
  char _pad2[CMI_CACHE_LINE_SIZE - sizeof(char**)]; // align to cache line
  std::atomic<int> len;
#else
  int  len;
#endif
#if CMK_SMP
  char _pad3[CMI_CACHE_LINE_SIZE - sizeof(int)]; // align to cache line
#endif

  const char **data;
  const char **bufEnd;

#if CMK_PCQUEUE_LOCK
  CmiNodeLock  lock;
#endif

}
*PCQueue;

static PCQueue PCQueueCreate(void)
{
  PCQueue Q;

  Q = (PCQueue)malloc(sizeof(struct PCQueueStruct));
  Q->data = (const char **)malloc(sizeof(char *)*PCQueueSize);
  memset(Q->data, 0, sizeof(char *)*PCQueueSize);
  _MEMCHECK(Q);
  Q->head = (char **)Q->data;
  Q->tail = (char **)Q->data;
  Q->len = 0;
  Q->bufEnd = Q->data + PCQueueSize;

#if CMK_PCQUEUE_LOCK || CMK_PCQUEUE_PUSH_LOCK
  Q->lock = CmiCreateLock();
#endif

  return Q;
}

static void PCQueueDestroy(PCQueue Q)
{
  free(Q->data);
  free(Q);
}

static int PCQueueEmpty(PCQueue Q)
{
  return (PCQueue_CmiMemoryAtomicLoad(Q->len, std::memory_order_acquire) == 0);
}

static int PCQueueLength(PCQueue Q)
{
  return PCQueue_CmiMemoryAtomicLoad(Q->len, std::memory_order_acquire);
}
static char *PCQueueTop(PCQueue Q)
{

    char *data;

#if CMK_PCQUEUE_LOCK
    CmiLock(Q->lock);
#endif

    data = *(Q->head);
//    if(data == 0) return 0;
     
#if CMK_PCQUEUE_LOCK
      CmiUnlock(Q->lock);
#endif

      return data;
}

static char *PCQueuePop(PCQueue Q)
{

    char *data;

#if CMK_PCQUEUE_LOCK
    CmiLock(Q->lock);
#endif

    data = *(Q->head);
//    if(data == 0) return 0;
     
    PCQueue_CmiMemoryReadFence();    

    if(data){
      *(Q->head) = 0;
      Q->head++;

      if (Q->head == (char **)Q->bufEnd ) { 
	Q->head = (char **)Q->data;
      }
      PCQueue_CmiMemoryAtomicDecrement(Q->len, std::memory_order_release);

    }

#if CMK_PCQUEUE_LOCK
      CmiUnlock(Q->lock);
#endif

      return data;
}
static void PCQueuePush(PCQueue Q, char *data)
{
#if CMK_PCQUEUE_LOCK || CMK_PCQUEUE_PUSH_LOCK
  CmiLock(Q->lock);
#endif

  PCQueue_CmiMemoryWriteFence();

  //CmiAssert(*(Q->tail)==0);

  *(Q->tail) = data;
   Q->tail++;

  if (Q->tail == (char **)Q->bufEnd) { /* last slot is about to be filled */
    /* this way, the next buffer is linked in before data is filled in
       in the last slot of this buffer */
    Q->tail = (char **)Q->data;
  }

#if 0
  if(Q->head == Q->tail && Q->len>0){ /* the whole buffer is fully occupied; len>0 is used to differentiate the case when the queue is empty in which head is also equal to tail*/
       CmiAbort("Simple PCQueue is full!!\n");
/*       char **newdata = (char **)malloc(sizeof(char *)*(Q->len << 1));
       int rsize = Q->data + Q->curSize - Q->head;
       int lsize = Q->tail - Q->data;
       memcpy(newdata, Q->head, sizeof(char *)*rsize);
       memcpy(newdata+rsize, Q->data, sizeof(char *)*lsize);
       free(Q->data);
       Q->data = newdata;
       Q->head = newdata;
       Q->tail = Q->data + Q->len;
*/
  }
#endif

  PCQueue_CmiMemoryAtomicIncrement(Q->len, std::memory_order_release);

#if CMK_PCQUEUE_LOCK || CMK_PCQUEUE_PUSH_LOCK
  CmiUnlock(Q->lock);
#endif
}
#endif

// CMK_LOCKLESS_QUEUE (disabled by default)
#if CMK_LOCKLESS_QUEUE

/*
 * MPSC Queue Design - Justin Miron
 *
 * MPSCQueue-Block
 *   _         DataNodes
 *  |_|->N       _
 *  |1|-------->|_|
 *  |2|-------| |1|
 *  |_|->N    | |2|
 *  |_|->N    |  _
 *  |_|->N    ->|3|
 *              |4|
 *              |_|
 *
 * The queue is designed as a multi level array. This allows us to achieve higher memory bound for a minimal space requirement.
 * It is composed of two blocks the single array of DataNodes and a DataNode is an array of char *.
 *
 * MPSCQueue-Blocks --> DataNode --> char *
 *
 * Memory Allocation: The first push in the first DataNode index of a data node allocates the node.
 * Memory Reclamation: The last push in the last DataNode frees the memory of the DataNode.
 *
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <sched.h>

typedef char** DataNode; //Data nodes are an array of char *

// Queue parameters initialized in init.C
extern int DataNodeSize;
extern int MaxDataNodes;
extern int QueueUpperBound;
extern int DataNodeWrap;
extern int QueueWrap;
extern int messageQueueOverflow;

/* Queue Parameters - All must be 2^n for wrapping to work */
#define NodePoolSize 0x100
#define FreeNodeWrap (NodePoolSize - 1)

void ReportOverflow()
{
  CmiMemoryAtomicIncrement(messageQueueOverflow);
}

/*
 * FreeNodePoolStruct
 * Holds nodes that are no longer in use but are not being freed
 * Acts as a SPSC bounded queue
 */
typedef struct FreeNodePoolStruct
{
  std::atomic_uint push;
  std::atomic_uint pull;
  std::atomic<uintptr_t> nodes[NodePoolSize];
} *FreeNodePool;

/*
 * MPSCQueueStruct
 * Linked list of queues that adds new node when DataNodeStruct is full
 * Insert at tail_node->push and remove at head_node->pull
 */
typedef struct MPSCQueueStruct
{
  std::atomic_uint push;
  char pad1[CMI_CACHE_LINE_SIZE - sizeof(std::atomic_uint)]; // align to cache line
  unsigned int pull;
  char pad2[CMI_CACHE_LINE_SIZE - sizeof(unsigned int)]; // align to cache line
  std::atomic<uintptr_t> *nodes;
  char pad3[CMI_CACHE_LINE_SIZE - sizeof(std::atomic<uintptr_t> *)]; // align to cache line
  FreeNodePool freeNodePool;
} *MPSCQueue;

static unsigned int WrappedDifference(unsigned int push, unsigned int pull)
{
  unsigned int difference;
  if(push < pull)
  {
    difference = (UINT_MAX - pull + push);
  }
  else
  {
    difference = push - pull;
  }
  return difference;
}

static int QueueFull(unsigned int push, unsigned int pull)
{
  int difference = WrappedDifference(push, pull);

  // The use of QueueUpperBound - 2*DataNodeSize is to remove the possiblity of wrapping
  // around the entire queue and pushing to a block that is going to be freed.
  // This removes concurrency issues with wrapping the entire queue and pushing to a block that
  // is being popped.
  if(difference >= (QueueUpperBound - 2*DataNodeSize))
    return 1;
  else
    return 0;
}

/* Creates a DataNode while holds the char* to data */
static DataNode DataNodeCreate(void)
{
  DataNode node = (DataNode)malloc(sizeof(char*)*DataNodeSize);
  int i;
  for(i = 0; i < DataNodeSize; ++i) node[i] = NULL;
  return node;
}

/* Initialize the FreeNodePool as a SPSC queue */
static FreeNodePool FreeNodePoolCreate(void)
{
  FreeNodePool free_q;

  free_q = (FreeNodePool)malloc(sizeof(struct FreeNodePoolStruct));
  std::atomic_store_explicit(&free_q->push, 0u, std::memory_order_relaxed);
  std::atomic_store_explicit(&free_q->pull, 0u, std::memory_order_relaxed);

  int i;
  for(i = 0; i < NodePoolSize; ++i)
    std::atomic_store_explicit(&free_q->nodes[i], (uintptr_t)NULL, std::memory_order_relaxed);

  return free_q;
}

/* Clean up all data on the queue */
static void FreeNodePoolDestroy(FreeNodePool q)
{
  int i;
  for(i = 0; i < NodePoolSize; ++i)
  {
    if((uintptr_t)atomic_load_explicit(&q->nodes[i], std::memory_order_acquire) != (uintptr_t)NULL)
    {
      DataNode n = (DataNode)atomic_load_explicit(&q->nodes[i], std::memory_order_acquire);
      free(n);
    }
  }
  free(q);
}

static DataNode get_free_node(FreeNodePool q)
{
  DataNode node;
  unsigned int push;
  unsigned int pull = std::atomic_load_explicit(&q->pull, std::memory_order_acquire);

  // Claim the next unique pull value if the queue is not empty
  do
  {
    push = std::atomic_load_explicit(&q->push, std::memory_order_acquire);

    if(pull == push) // Pool is empty, need to allocate a new DataNode.
      return DataNodeCreate();

  } while(!std::atomic_compare_exchange_weak_explicit(&q->pull, &pull, (pull + 1) & FreeNodeWrap, std::memory_order_release, std::memory_order_relaxed));

  // If the element is NULL, a producer is still pushing to the slot, but has begun the push operation.
  while((node = (DataNode)atomic_load_explicit(&q->nodes[pull], std::memory_order_acquire)) == NULL);
  std::atomic_store_explicit(&q->nodes[pull], (uintptr_t)NULL, std::memory_order_release); //NULL out spot in queue to indicate available spot.

  return node;
}

static void add_free_node(FreeNodePool q, DataNode available)
{
  unsigned int pull;
  unsigned int push = std::atomic_load_explicit(&q->push, std::memory_order_acquire);

  // Claim a unique push value if the queue is not empty
  do
  {
    // The next push value's element has not been NULL'd yet by a consumer, indicating the queue is full
    if((uintptr_t)atomic_load_explicit(&q->nodes[push], std::memory_order_acquire) != (uintptr_t)NULL)
    {
      free(available);
      return;
    }

  } while(!std::atomic_compare_exchange_weak_explicit(&q->push, &push, (push + 1) & FreeNodeWrap, std::memory_order_release, std::memory_order_relaxed));

  std::atomic_store_explicit(&q->nodes[push], (uintptr_t)available, std::memory_order_release);
}

static MPSCQueue MPSCQueueCreate(void)
{
  /* Initialize the MPSCQueue struct */
  MPSCQueue Q = (MPSCQueue)malloc(sizeof(struct MPSCQueueStruct));
  Q->nodes = (std::atomic<uintptr_t>*)malloc(sizeof(std::atomic<uintptr_t>)*MaxDataNodes);
  Q->freeNodePool = FreeNodePoolCreate();
  Q->pull = 0;
  std::atomic_store_explicit(&Q->push, 0u, std::memory_order_relaxed);

  unsigned int i;
  for(i = 0; i < MaxDataNodes; ++i)
  {
    std::atomic_store_explicit(&Q->nodes[i], (uintptr_t)NULL, std::memory_order_relaxed);
  }

  return Q;
}

static void MPSCQueueDestroy(MPSCQueue Q)
{
  /* Iterate through blocks. Every Datanode in the array must be freed before the node array */
  unsigned int i, j;
  for(i = 0; i < MaxDataNodes; ++i)
    if((uintptr_t)atomic_load_explicit(&Q->nodes[i], std::memory_order_acquire) != (uintptr_t)NULL)
    {
      free((DataNode)atomic_load_explicit(&Q->nodes[i], std::memory_order_acquire));
    }

  FreeNodePoolDestroy(Q->freeNodePool);
  free(Q->nodes);
  free(Q);
}

/* Index of DataNode in the node array */
static inline unsigned int get_node_index(unsigned int value)
{
  return ((value & QueueWrap) / DataNodeSize);
}

/* Gets the DataNode to push to */
static DataNode get_push_node(MPSCQueue Q, unsigned int push_idx)
{
  /* Index in the block: block[block_idx] */
  unsigned int node_idx = get_node_index(push_idx);

  /* If it is the first in the DataNode then create the node otherwise wait for it */
  if((push_idx & DataNodeWrap) == 0)
  {
    DataNode new_node = get_free_node(Q->freeNodePool);
    std::atomic_store_explicit(&Q->nodes[node_idx], (uintptr_t)new_node, std::memory_order_release);
    return new_node;
  }
  else // Wait until the producer with the first element in the DataNode creates the node.
  {
    DataNode node;
    while((node = (DataNode)atomic_load_explicit(&Q->nodes[node_idx], std::memory_order_acquire)) == NULL);
    return node;
  }
}

/* Get index of pop node in the popped block */
static inline DataNode get_pop_node(MPSCQueue Q, unsigned int pull_idx)
{
  unsigned int node_idx = get_node_index(pull_idx);

  return (DataNode)atomic_load_explicit(&Q->nodes[node_idx], std::memory_order_relaxed);
}

/* Check whether or not a node is ready to be freed */
static void check_mem_reclamation(MPSCQueue Q, unsigned int pull_idx, DataNode node)
{
  unsigned int node_idx = get_node_index(pull_idx);

  /* If we are pulling from the end of a node, free the node */
  if((pull_idx & DataNodeWrap) == (DataNodeSize - 1))
  {
    add_free_node(Q->freeNodePool, node);
    std::atomic_store_explicit(&Q->nodes[node_idx], (uintptr_t)NULL, std::memory_order_relaxed);
  }
}

static int MPSCQueueEmpty(MPSCQueue Q)
{
  unsigned int push = std::atomic_load_explicit(&Q->push, std::memory_order_relaxed);
  unsigned int pull = Q->pull;
  return WrappedDifference(push, pull) == 0;
}

static int MPSCQueueLength(MPSCQueue Q)
{
  unsigned int push = std::atomic_load_explicit(&Q->push, std::memory_order_relaxed);
  unsigned int pull = Q->pull;
  return (int)WrappedDifference(push, pull);
}

static char *MPSCQueueTop(MPSCQueue Q)
{
  unsigned int pull = Q->pull;
  unsigned int push = std::atomic_load_explicit(&Q->push, std::memory_order_acquire);

  DataNode node = get_pop_node(Q, pull);

  if(pull == push || node == NULL) return NULL; // Queue is empty

  unsigned int node_pull = pull & DataNodeWrap;

  char * data = node[node_pull];
  return data;
}

static char *MPSCQueuePop(MPSCQueue Q)
{
  unsigned int pull = Q->pull;
  unsigned int push = std::atomic_load_explicit(&Q->push, std::memory_order_acquire);
  if(pull == push) // If the queue is empty
    return NULL;

  DataNode node = get_pop_node(Q, pull);
  if(node == NULL) // If a producer has not finished allocating the block we are attempting to pop from
    return NULL;

  unsigned int node_pull = pull & DataNodeWrap;

  char * data = node[node_pull];
  if(data == NULL) // If a producer has not finished pushing an element we are attempting to pop
    return NULL;

  node[node_pull] = NULL; //NULL the element to indicate it is available again

  Q->pull = (pull + 1);

  check_mem_reclamation(Q, pull, node); //Check if we can free the node

  return data;
}

static void MPSCQueuePush(MPSCQueue Q, char *data)
{
  unsigned int push = std::atomic_fetch_add_explicit(&Q->push, 1u, std::memory_order_release);
  unsigned int pull = Q->pull;

  while(QueueFull(push, pull)) //Block until the push index is available to push to
  {
    ReportOverflow();

    sched_yield();
    pull = Q->pull;
  }

  DataNode node = get_push_node(Q, push);
  node[push & DataNodeWrap] = data;
}


/* BEGINNING OF MPMC CODE */
/*
 * MPMC Queue Design - Justin Miron
 *
 * MPMCQueueBlock
 *   _         MPMCDataNodes
 *  |_|->N       _
 *  |1|-------->|_|
 *  |2|-------| |1|
 *  |_|->N    | |2|
 *  |_|->N    |  _
 *  |_|->N    ->|3|
 *              |4|
 *              |_|
 *
 * The queue is designed as a multi level array. This allows us to achieve higher memory bound for a minimal space requirement.
 * It is composed of two blocks the single MPMCQueue-NodeArray and a MPMCDataNode for each of the indices
 * in the MPMCQueue-NodeArray.
 *
 * MPMCQueue-NodeArray --> MPMCDataNode --> char *
 *
 * Memory Allocation: The first push in the first MPMCDataNode of a MPMCNodeBlock creates the block and the MPMCDataNode.
 * Otherwise the first push in each MPMCDataNode creates the MPMCDataNode.
 *
 * Memory Reclamation: The nodes and blocks each keep a count of the number of pop operations performed on them. A consumer atomically
 * increments this count at the very end after it has acquired the data and no longer uses the node.
 * Once this count hits the size of the node, whichever consumer is the last one using the block (atomically fetches DataNodeSize
 * is responsible for freeing the node.
 *
 *
 */

/*
 * MPMCDataNodeStruct
 * Array buffer that holds the internal data of the queue
 */
typedef struct MPMCDataNodeStruct
{
  std::atomic<uintptr_t> *data;
  std::atomic_uint num_popped;
} *MPMCDataNode;

/*
 * FreeMPMCNodePoolStruct
 * Holds nodes that are no longer in use but are not being freed
 * Acts as a SPSC bounded queue
 */
typedef struct FreeMPMCNodePoolStruct
{
  std::atomic_uint push;
  std::atomic_uint pull;
  std::atomic<uintptr_t> nodes[NodePoolSize];
} *FreeMPMCNodePool;

/*
 * MPMCQueueStruct
 * Linked list of queues that adds new node when MPMCDataNodeStruct is full
 * Insert at tail_node->push and remove at head_node->pull
 */
typedef struct MPMCQueueStruct
{
  std::atomic_uint push;
  char pad1[CMI_CACHE_LINE_SIZE - sizeof(std::atomic_uint)]; // align to cache line
  std::atomic_uint pull;
  char pad2[CMI_CACHE_LINE_SIZE - sizeof(std::atomic_uint)]; // align to cache line
  std::atomic<uintptr_t> *nodes;
  char pad3[CMI_CACHE_LINE_SIZE - sizeof(std::atomic<uintptr_t> *)]; // align to cache line
  FreeMPMCNodePool freeMPMCNodePool;
  char pad4[CMI_CACHE_LINE_SIZE - sizeof(FreeMPMCNodePool)]; // align to cache line
  std::atomic_flag queueOverflowed;

} *MPMCQueue;

/* Creates a MPMCDataNode while holds the char* to data */
static MPMCDataNode MPMCDataNodeCreate(void)
{
  MPMCDataNode node = (MPMCDataNode)malloc(sizeof(struct MPMCDataNodeStruct));
  node->data = (std::atomic<uintptr_t>*)malloc(sizeof(std::atomic<uintptr_t>)*DataNodeSize);
  std::atomic_store_explicit(&node->num_popped, 0u, std::memory_order_relaxed);
  int i;
  for(i = 0; i < DataNodeSize; ++i) std::atomic_store_explicit(&node->data[i], (uintptr_t)NULL, std::memory_order_relaxed);
  return node;
}

/* Initialize the FreeMPMCNodePool as a MPMC queue */
static FreeMPMCNodePool FreeMPMCNodePoolCreate(void)
{
  FreeMPMCNodePool free_q;

  free_q = (FreeMPMCNodePool)malloc(sizeof(struct FreeMPMCNodePoolStruct));
  std::atomic_store_explicit(&free_q->push, 0u, std::memory_order_relaxed);
  std::atomic_store_explicit(&free_q->pull, 0u, std::memory_order_relaxed);

  int i;
  for(i = 0; i < NodePoolSize; ++i)
    std::atomic_store_explicit(&free_q->nodes[i], (uintptr_t)NULL, std::memory_order_relaxed);

  return free_q;
}

/* Clean up all data on the queue */
static void FreeMPMCNodePoolDestroy(FreeMPMCNodePool q)
{
  int i;
  for(i = 0; i < NodePoolSize; ++i)
  {
    if((uintptr_t)atomic_load_explicit(&q->nodes[i], std::memory_order_relaxed) != (uintptr_t)NULL)
    {
      MPMCDataNode n = (MPMCDataNode)atomic_load_explicit(&q->nodes[i], std::memory_order_relaxed);
      free(n->data);
      free(n);
    }
  }
  free(q);
}

static MPMCDataNode mpmc_get_free_node(FreeMPMCNodePool q)
{
  unsigned int push;
  unsigned int pull = std::atomic_load_explicit(&q->pull, std::memory_order_acquire);
  do
  {
    push = std::atomic_load_explicit(&q->push, std::memory_order_acquire); // maybe make relaxed

    if(pull == push) //Pool is empty, need to allocate the data node
      return MPMCDataNodeCreate();

  } while(!std::atomic_compare_exchange_weak_explicit(&q->pull, &pull, (pull + 1) & FreeNodeWrap, std::memory_order_release, std::memory_order_relaxed));

  MPMCDataNode node;
  while((node = (MPMCDataNode)atomic_load_explicit(&q->nodes[pull], std::memory_order_acquire)) == NULL);
  std::atomic_store_explicit(&q->nodes[pull], (uintptr_t)NULL, std::memory_order_release); // NULL the element to indicate the element is available to push to.

  return node;
}

static void mpmc_add_free_node(FreeMPMCNodePool q, MPMCDataNode available)
{
  /* If the queue is full, cannot add available to it */
  unsigned int pull;
  unsigned int push = std::atomic_load_explicit(&q->push, std::memory_order_acquire);
  do
  {
    // If the element is not NULL, either a consumer is still poping from that index, or the queue is full.
    if((uintptr_t)atomic_load_explicit(&q->nodes[push], std::memory_order_acquire) != (uintptr_t)NULL)
    {
      free(available->data);
      free(available);
      return;
    }

  } while(!std::atomic_compare_exchange_weak_explicit(&q->push, &push, (push + 1) & FreeNodeWrap, std::memory_order_release, std::memory_order_relaxed));

  std::atomic_store_explicit(&q->nodes[push], (uintptr_t)available, std::memory_order_release);
}

static MPMCQueue MPMCQueueCreate(void)
{
  /* Initialize the MPMCQueue struct */
  MPMCQueue Q = (MPMCQueue)malloc(sizeof(struct MPMCQueueStruct));
  Q->nodes = (std::atomic<uintptr_t>*)malloc(sizeof(std::atomic<uintptr_t>)*MaxDataNodes);
  Q->freeMPMCNodePool = FreeMPMCNodePoolCreate();
  std::atomic_store_explicit(&Q->pull, 0u, std::memory_order_relaxed);
  std::atomic_store_explicit(&Q->push, 0u, std::memory_order_relaxed);

  unsigned int i;
  for(i = 0; i < MaxDataNodes; ++i)
  {
    std::atomic_store_explicit(&Q->nodes[i], (uintptr_t)NULL, std::memory_order_relaxed);
  }

  return Q;
}

static void MPMCQueueDestroy(MPMCQueue Q)
{
  /* Iterate through blocks. Every MPMCDataNode in the block must be freed, then the node array itself */
  unsigned int i, j;
  for(i = 0; i < MaxDataNodes; ++i)
  {
    if((uintptr_t)atomic_load_explicit(&Q->nodes[i], std::memory_order_relaxed) != (uintptr_t)NULL)
    {
      MPMCDataNode n = (MPMCDataNode)atomic_load_explicit(&Q->nodes[i], std::memory_order_relaxed);
      free(n->data);
      free(n);
    }
  }
  /* Free the Pool structures and the queue itself */
  FreeMPMCNodePoolDestroy(Q->freeMPMCNodePool);
  free(Q->nodes);
  free(Q);
}

/* Gets the MPMCDataNode to push to */
static MPMCDataNode mpmc_get_push_node(MPMCQueue Q, unsigned int push_idx)
{
  unsigned int node_idx = get_node_index(push_idx);

  MPMCDataNode node = (MPMCDataNode)atomic_load(&Q->nodes[node_idx]);

  if(node == NULL)
  {
    // if(node_idx == 0) printf("CREATING NODE IDX == 0, idx: %d\n", push_idx);
    MPMCDataNode new_node = MPMCDataNodeCreate();
    if(std::atomic_compare_exchange_strong(&Q->nodes[node_idx], (uintptr_t *)&node, (uintptr_t)new_node))
      return new_node;
    else
    {
      free(new_node->data);
      free(new_node);
      return (MPMCDataNode)atomic_load(&Q->nodes[node_idx]);
    }
  }
  else
  {
    return node;
  }
}

/* Get index of pop node in the popped block */
static inline MPMCDataNode mpmc_get_pop_node(MPMCQueue Q, unsigned int pull_idx)
{
  unsigned int node_idx = get_node_index(pull_idx);

  return (MPMCDataNode)atomic_load_explicit(&Q->nodes[node_idx], std::memory_order_relaxed);
}

/* After popping we may need to free the memory */
static void mpmc_check_mem_reclamation(MPMCQueue Q, unsigned int pull_idx, MPMCDataNode node)
{
  unsigned int node_idx = get_node_index(pull_idx);

  unsigned int node_popped = std::atomic_fetch_add_explicit(&node->num_popped, 1u, std::memory_order_relaxed);

  // If we have popped all the elements out of a DataNode, it can be freed
  if(node_popped == DataNodeSize - 1)
  {
    mpmc_add_free_node(Q->freeMPMCNodePool, node);
    std::atomic_store_explicit(&Q->nodes[node_idx], (uintptr_t)NULL, std::memory_order_relaxed);
  }
}

static int MPMCQueueEmpty(MPMCQueue Q)
{
  return (atomic_load_explicit(&Q->push, std::memory_order_relaxed) == std::atomic_load_explicit(&Q->pull, std::memory_order_relaxed));
}

static int MPMCQueueLength(MPMCQueue Q)
{
  return std::atomic_load_explicit(&Q->push, std::memory_order_relaxed) - std::atomic_load_explicit(&Q->pull, std::memory_order_relaxed);
}

static char *MPMCQueueTop(MPMCQueue Q)
{
  unsigned int pull = std::atomic_load_explicit(&Q->pull, std::memory_order_acquire); // maybe make relaxed
  unsigned int push = std::atomic_load_explicit(&Q->push, std::memory_order_acquire);
  MPMCDataNode node = mpmc_get_pop_node(Q, pull);

  if(pull == push || node == NULL)
    return NULL;

  unsigned int node_pull = pull & DataNodeWrap;

  char * data;
  while((data = (char *)atomic_load_explicit(&node->data[node_pull], std::memory_order_acquire)) == NULL);

  return data;
}

static char *MPMCQueuePop(MPMCQueue Q)
{
  unsigned int pull;
  unsigned int push;
  MPMCDataNode node;
  do
  {
    pull = std::atomic_load_explicit(&Q->pull, std::memory_order_acquire); // maybe make relaxed
    push = std::atomic_load_explicit(&Q->push, std::memory_order_acquire);

    node = mpmc_get_pop_node(Q, pull);

    if(pull == push || node == NULL) //The next element to pull from is not finished being pushed, or the queue is empty.
      return NULL;

  } while(!std::atomic_compare_exchange_weak_explicit(&Q->pull, &pull, pull + 1, std::memory_order_release, std::memory_order_relaxed));

  unsigned int node_pull = pull & DataNodeWrap;

  char * data;
  while((data = (char *)atomic_load_explicit(&node->data[node_pull], std::memory_order_acquire)) == NULL);
  std::atomic_store_explicit(&node->data[node_pull], (uintptr_t)NULL, std::memory_order_release); //NULL out the element to indicate it is available.

  mpmc_check_mem_reclamation(Q, pull, node); //Check if we can free the data node

  return data;
}

static void MPMCQueuePush(MPMCQueue Q, void *data)
{
  unsigned int push = std::atomic_fetch_add_explicit(&Q->push, 1u, std::memory_order_release);
  unsigned int pull = std::atomic_load_explicit(&Q->pull, std::memory_order_acquire);

  while(QueueFull(push, pull)) //Next value to push to is not available yet, block until available
  {
    ReportOverflow();

    sched_yield();
    pull = std::atomic_load_explicit(&Q->pull, std::memory_order_acquire);
  }

  MPMCDataNode node = mpmc_get_push_node(Q, push & QueueWrap);
  std::atomic_store_explicit(&node->data[push & DataNodeWrap], (uintptr_t)data, std::memory_order_release);
}


#endif /* the endif for "#if CMK_LOCKLESS_QUEUE" */


/* the endif for "ifndef _PCQUEUE_" */
#endif

/*@}*/
