
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

/************************************************************
 *
 * Processor state structure
 *
 ************************************************************/

typedef struct CmiStateStruct
{
  int pe, rank;
  PCQueue recv; 
  void *localqueue;
  CmiIdleLock idle;
}
*CmiState;

typedef struct CmiNodeStateStruct
{
  CmiNodeLock immSendLock; /* lock for pushing into immediate queues */
  CmiNodeLock immRecvLock; /* lock for processing immediate messages */
  PCQueue     immQ; 	   /* immediate messages to handle ASAP: 
                              Locks: push(SendLock), pop(RecvLock) */
  PCQueue     delayedImmQ; /* delayed immediate messages:
                              Locks: push(RecvLock), pop(RecvLock) */
#if CMK_NODE_QUEUE_AVAILABLE
  CmiNodeLock CmiNodeRecvLock;
  PCQueue     NodeRecv;
#endif
}
CmiNodeState;

#endif
