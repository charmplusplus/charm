
#ifndef MACHINE_SMP_H
#define MACHINE_SMP_H

/*
CmiIdleLock
CmiState
*/

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

typedef struct CmiStateStruct
{
  int pe, rank;
  PCQueue recv; 
  void *localqueue;
  CmiIdleLock idle;
}
*CmiState;

#endif
