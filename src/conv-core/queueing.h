/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef QUEUEING_H
#define QUEUEING_H
/*#define FASTQ*/


/** 
   @addtogroup CharmScheduler
   @{
 */


#ifdef __cplusplus
extern "C" {
#endif

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
*prio;

/** A double ended queue of void* pointers stored in a circular buffer, with internal space for 4 entries */
typedef struct deq_struct
{
  /* Note: if head==tail, circ is empty */
  void **bgn; /**< Pointer to first slot in circular buffer */
  void **end; /**< Pointer past last slot in circular buffer */
  void **head; /**< Pointer to first used slot in circular buffer */
  void **tail; /**< Pointer to next available slot in circular buffer */
  void *space[4]; /**< Enough space for the first 4 entries */
}
*deq;

#ifndef FASTQ
/** An bucket in a priority queue which contains a deque(storing the void* pointers) and references to other buckets in the hash table. */
typedef struct prioqelt_struct
{
  struct deq_struct data;
  struct prioqelt_struct *ht_next; /**< Pointer to next bucket in hash table. */
  struct prioqelt_struct **ht_handle; /**< Pointer to pointer that points to me (!) */
  struct prio_struct pri;
}
*prioqelt;
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
*prioqelt;
#endif
/*
#ifndef FASTQ
#define PRIOQ_TABSIZE 1017
#else */
#define PRIOQ_TABSIZE 1017
/*#endif */

/*#ifndef FASTQ*/
/** A priority queue, implemented as a heap of prioqelt_struct buckets (each bucket represents a single priority value and contains a deque of void* pointers) */
typedef struct prioq_struct
{
  int heapsize; 
  int heapnext;
  prioqelt *heap; /**< An array of prioqelt's */
  prioqelt *hashtab;
  int hash_key_size;
  int hash_entry_size;
}
*prioq;
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
/** A set of 3 queues: a positive priority prioq_struct, a negative priority prioq_struct, and a zero priority deq_struct.
    If the user modifies the queue, NULL entries may be present, and hence NULL values will be returned by CqsDequeue().
*/
typedef struct Queue_struct
{
  unsigned int length;
  unsigned int maxlen;
  struct deq_struct zeroprio; /**< A double ended queue for zero priority messages */
  struct prioq_struct negprioq; /**< A priority queue for negative priority messages */
  struct prioq_struct posprioq; /**< A priority queue for negative priority messages */
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

Queue CqsCreate(void);
void CqsDelete(Queue);
void CqsEnqueue(Queue, void *msg);
void CqsEnqueueFifo(Queue, void *msg);
void CqsEnqueueLifo(Queue, void *msg);
void CqsEnqueueGeneral(Queue, void *msg,int strategy, 
	       int priobits, unsigned int *prioPtr);

void CqsEnumerateQueue(Queue q, void ***resp);
void CqsDequeue(Queue, void **msgPtr);

unsigned int CqsLength(Queue);
unsigned int CqsMaxLength(Queue);
int CqsEmpty(Queue);
int CqsPrioGT(prio, prio);
prio CqsGetPriority(Queue);

deq CqsPrioqGetDeq(prioq pq, unsigned int priobits, unsigned int *priodata);
void *CqsPrioqDequeue(prioq pq);
void CqsDeqEnqueueFifo(deq d, void *data);

void CqsIncreasePriorityForEntryMethod(Queue q, const int entrymethod);
void CqsRemoveSpecific(Queue, const void *msgPtr);


#ifdef __cplusplus
};
#endif

/** @} */

#endif
