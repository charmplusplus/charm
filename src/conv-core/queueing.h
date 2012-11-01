#ifndef QUEUEING_H
#define QUEUEING_H
/*#define FASTQ*/

/** 
    @file 
    Declarations of queuing data structure functions.
    @ingroup CharmScheduler   

    @addtogroup CharmScheduler
    @{
 */

#include "conv-config.h"

#ifdef __cplusplus
extern "C" {
#endif


/** A memory limit threshold for adaptively scheduling */
extern int schedAdaptMemThresholdMB;

#ifndef CINTBITS
#define CINTBITS ((unsigned int) (sizeof(int)*8))
#endif
#ifndef CLONGBITS
#define CLONGBITS ((unsigned int) (sizeof(CmiInt8)*8))
#endif

/** Stores a variable bit length priority */
typedef struct prio_struct
{
  unsigned short bits;
  unsigned short ints;
  unsigned int data[1];
}
*_prio;

/**
   A double ended queue of void* pointers stored in a circular buffer,
   with internal space for 4 entries
*/
typedef struct deq_struct
{
  /* Note: if head==tail, circ is empty */
  void **bgn; /**< Pointer to first slot in circular buffer */
  void **end; /**< Pointer past last slot in circular buffer */
  void **head; /**< Pointer to first used slot in circular buffer */
  void **tail; /**< Pointer to next available slot in circular buffer */
  void *space[4]; /**< Enough space for the first 4 entries */
}
*_deq;

#ifndef FASTQ
/**
   A bucket in a priority queue which contains a deque(storing the
   void* pointers) and references to other buckets in the hash
   table.
*/
typedef struct prioqelt_struct
{
  struct deq_struct data;
  struct prioqelt_struct *ht_next; /**< Pointer to next bucket in hash table. */
  struct prioqelt_struct **ht_handle; /**< Pointer to pointer that points to me (!) */
  struct prio_struct pri;
}
*_prioqelt;
#else
typedef struct prioqelt_struct
{
  struct deq_struct data;
  struct prioqelt_struct *ht_left; /**< Pointer to left bucket in hash table. */
  struct prioqelt_struct *ht_right; /**< Pointer to right bucket in hash table. */
  struct prioqelt_struct *ht_parent; /**< Pointer to the parent bucket in the hash table */
  struct prioqelt_struct **ht_handle; /**< Pointer to pointer in the hashtable that points to me (!) */
  struct prio_struct pri;
  /*  int deleted; */
}
*_prioqelt;
#endif
/*
#ifndef FASTQ
#define PRIOQ_TABSIZE 1017
#else */
#define PRIOQ_TABSIZE 1017
/*#endif */

/*#ifndef FASTQ*/
/**
   A priority queue, implemented as a heap of prioqelt_struct buckets
   (each bucket represents a single priority value and contains a
   deque of void* pointers)
*/
typedef struct prioq_struct
{
  int heapsize; 
  int heapnext;
  _prioqelt *heap; /**< An array of prioqelt's */
  _prioqelt *hashtab;
  int hash_key_size;
  int hash_entry_size;
}
*_prioq;
/*#else
typedef struct prioq1_struct
{
  int heapsize;
  int heapnext;
  prioqelt1 *heap;
  prioqelt1 hashtab[PRIOQ_TABSIZE];
}
*prioq1;
#endif
*/

/*#ifndef FASTQ*/
/**
   A set of 3 queues: a positive priority prioq_struct, a negative
   priority prioq_struct, and a zero priority deq_struct.
   
   If the user modifies the queue, NULL entries may be present, and
   hence NULL values will be returned by CqsDequeue().
*/
typedef struct Queue_struct
{
  unsigned int length;
  unsigned int maxlen;
  struct deq_struct zeroprio; /**< A double ended queue for zero priority messages */
  struct prioq_struct negprioq; /**< A priority queue for negative priority messages */
  struct prioq_struct posprioq; /**< A priority queue for negative priority messages */
#if CMK_USE_STL_MSGQ
  void *stlQ; /**< An STL-based alternative to charm's msg queues */
#endif
}
*Queue;
/*#else
typedef struct Queue1_struct
{
  unsigned int length;
  unsigned int maxlen;
  struct deq_struct zeroprio;
  struct prioq1_struct negprioq;
  struct prioq1_struct posprioq;
}
*Queue;
#endif
*/

/**
    Initialize a Queue and its three internal queues (for positive,
    negative, and zero priorities)
*/
Queue CqsCreate(void);

/** Delete a Queue */
void CqsDelete(Queue);

/** Enqueue with priority 0 */
void CqsEnqueue(Queue, void *msg);

/** Enqueue behind other elements of priority 0 */
void CqsEnqueueFifo(Queue, void *msg);

/** Enqueue ahead of other elements of priority 0 */
void CqsEnqueueLifo(Queue, void *msg);

/**
    Enqueue something (usually an envelope*) into the queue in a
    manner consistent with the specified strategy and priority.
*/
void CqsEnqueueGeneral(Queue, void *msg, int strategy, 
		       int priobits, unsigned int *prioPtr);

/**
    Produce an array containing all the entries in a Queue
    @return a newly allocated array filled with copies of the (void*)
    elements in the Queue.
    @param [in] q a Queue
    @param [out] resp an array of pointer entries found in the Queue,
    with as many entries as the Queue's length. The caller must
    CmiFree this.
*/
void CqsEnumerateQueue(Queue q, void ***resp);

/**
   Retrieve the highest priority message (one with most negative
   priority)
*/
void CqsDequeue(Queue, void **msgPtr);

unsigned int CqsLength(Queue);
int CqsEmpty(Queue);
int CqsPrioGT_(unsigned int ints1, unsigned int *data1, unsigned int ints2, unsigned int *data2);
int CqsPrioGT(_prio, _prio);

/** Get the priority of the highest priority message in q */
_prio CqsGetPriority(Queue);

void CqsIncreasePriorityForEntryMethod(Queue q, const int entrymethod);

/**
    Remove an occurence of a specified entry from the Queue by setting
    its entry to NULL.

    The size of the Queue will not change, it will now just contain an
    entry for a NULL pointer.
*/
void CqsRemoveSpecific(Queue, const void *msgPtr);

#ifdef ADAPT_SCHED_MEM
void CqsIncreasePriorityForMemCriticalEntries(Queue q);
#endif

#ifdef __cplusplus
}
#endif

/** @} */

#endif
