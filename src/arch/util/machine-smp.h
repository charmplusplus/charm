/** @file
 * \brief structures for the SMP versions
 * \ingroup Machine
 */

/**
 * \addtogroup Machine
 * @{
 */

#ifndef MACHINE_SMP_H
#define MACHINE_SMP_H

/*
CmiIdleLock
CmiState
*/

/***********************************************************
 * SMP Idle Locking
 *   In an SMP system, idle processors need to sleep on a
 * lock so that if a message for them arrives, they can be
 * woken up.
 **********************************************************/

#if CMK_SHARED_VARS_NT_THREADS

typedef struct {
  int hasMessages; /*Is there a message waiting?*/
  volatile int isSleeping; /*Are we asleep in this cond?*/
  HANDLE sem;
} CmiIdleLock;

#elif CMK_SHARED_VARS_POSIX_THREADS_SMP

typedef struct {
  volatile int hasMessages; /*Is there a message waiting?*/
  volatile int isSleeping; /*Are we asleep in this cond?*/
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} CmiIdleLock;

#else        /* non SMP */

typedef struct {
  int hasMessages;
} CmiIdleLock;

#endif


/*#define CMK_SMP_MULTIQ 1*/
#if CMK_SMP_MULTIQ
/* 
 * The value is usually equal to the number of cores on
 * this node for the best possible performance. In such
 * cases, the CMK_PCQUEUE_PUSHLOCk should be disabled.
 *
 * For large fat smp node (say, over 16 cores per node),
 * then this value could be half or quarter of the #cores.
 * Then the CMK_PCQUEUE_PUSHLOCK should be enabled.
 *
 * */
#ifndef MULTIQ_GRPSIZE
#define MULTIQ_GRPSIZE 8
#endif
#endif

/************************************************************
 *
 * Processor state structure
 *
 ************************************************************/

typedef struct CmiStateStruct
{
  int pe, rank;
#if !CMK_SMP_MULTIQ
  CMIQueue recv; 
#else
  CMIQueue recv[MULTIQ_GRPSIZE];
  int myGrpIdx;
  int curPolledIdx;
#endif

  void *localqueue;
  CmiIdleLock idle;
}
*CmiState;

typedef struct CmiNodeStateStruct
{
  CmiNodeLock immSendLock; /* lock for pushing into immediate queues */
  CmiNodeLock immRecvLock; /* lock for processing immediate messages */
  CMIQueue     immQ; 	   /* immediate messages to handle ASAP: 
                              Locks: push(SendLock), pop(RecvLock) */
  CMIQueue     delayedImmQ; /* delayed immediate messages:
                              Locks: push(RecvLock), pop(RecvLock) */
#if CMK_NODE_QUEUE_AVAILABLE
  CmiNodeLock CmiNodeRecvLock;
  CMIQueue     NodeRecv;
#endif
}
CmiNodeState;

#endif

/*@}*/
