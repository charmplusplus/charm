/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

 /**************************************************************************
 *
 * typedef CthThread
 *
 *   - a first-class thread object.
 *
 * CthThread CthSelf()
 *
 *   - returns the current thread.
 *
 * void CthResume(CthThread t)
 *
 *   - Immediately transfers control to thread t.  Note: normally, the user
 *     of a thread package wouldn't explicitly choose which thread to transfer
 *     to.  Instead, the user would rely upon a "scheduler" to choose the
 *     next thread.  Therefore, this routine is primarily intended for people
 *     who are implementing schedulers, not for end-users.  End-users should
 *     probably call CthSuspend or CthAwaken (see below).
 *
 * CthThread CthCreate(CthVoidFn fn, void *arg, int size)
 *
 *   - Creates a new thread object.  The thread is not given control yet.
 *     The thread is not passed to the scheduler.  When (and if) the thread
 *     eventually receives control, it will begin executing the specified 
 *     function 'fn' with the specified argument.  The 'size' parameter
 *     specifies the stack size, 0 means use the default size.
 *
 * void CthFree(CthThread t)
 *
 *   - Frees thread t.  You may free the currently-executing thread, although
 *     the free will actually be postponed until the thread suspends.
 *
 *
 * In addition to the routines above, the threads package assumes that there
 * will be a "scheduler" of some sort, whose job is to select which threads
 * to execute.  The threads package does not provide a scheduler (although
 * converse may provide one or more schedulers separately).  However, for
 * standardization reasons, it does define an interface to which all schedulers
 * can comply.  A scheduler consists of a pair of functions:
 *
 *   - An awaken-function.  The awaken-function is called to
 *     notify the scheduler that a particular thread needs the CPU.  The
 *     scheduler is expected to respond to this by inserting the thread
 *     into a ready-pool of some sort.
 *
 *   - A choose-next function.  The choose-next function is called to
 *     to ask the scheduler which thread to execute next.
 *
 * The interface to the scheduler is formalized in the following functions:
 *
 * void CthSchedInit()
 *
 *   - you must call this before any of the following functions will work.
 *
 * void CthSuspend()
 *
 *   - The thread calls this function, which in turn calls the scheduler's
 *     choose-next function.  It then resumes whatever thread is returned
 *     by the choose-next function.
 *
 * void CthAwaken(CthThread t)
 *
 *   - The thread-package user calls this function, which in turn calls the
 *     scheduler's awaken-function to awaken thread t.  This probably causes
 *     the thread t to be inserted in the ready-pool.
 *
 * void CthSetStrategy(CthThread t, CthAwkFn awakenfn, CthThFn choosefn)
 *
 *     This specifies the scheduling functions to be used for thread 't'.
 *     The scheduling functions must have the following prototypes:
 *
 *          void awakenfn(CthThread t);
 *          CthThread choosefn();
 *
 *     These functions must be provided on a per-thread basis.  (Eg, if you
 *     CthAwaken a thread X, then X's awakefn will be called.  If a thread Y
 *     calls CthSuspend, then Y's choosefn will be called to pick the next
 *     thread.)  Of course, you may use the same functions for all threads
 *     (the common case), but the specification on a per-thread basis gives
 *     you maximum flexibility in controlling scheduling.
 *
 *     See also: common code, CthSetStrategyDefault.
 *
 * void CthYield()
 *
 *   - simply executes { CthAwaken(CthSelf()); CthSuspend(); }.  This
 *     combination gives up control temporarily, but ensures that control
 *     will eventually return.
 *
 *
 * Note: there are several possible ways to implement threads.   
 *****************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if CMK_MEMORY_PROTECTABLE
#include <malloc.h> /*<- for memalign*/
#endif

#include "converse.h"
#include "qt.h"

#include "conv-trace.h"
#include <sys/types.h>

/**************************** Shared Base Thread Class ***********************/
typedef struct CthThreadBase
{
  /*Start with a message header so threads can be enqueued 
    as messages (e.g., by CthEnqueueNormalThread in convcore.c)
  */
  char cmicore[CmiMsgHeaderSizeBytes];
  
  CmiObjId   tid;        /* globally unique tid */
  CthAwkFn   awakenfn;   /* Insert this thread into the ready queue */
  CthThFn    choosefn;   /* Return the next ready thread */
  CthThread  next; /* Next active thread */
  int        suspendable; /* Can this thread be blocked */
  int        exiting;    /* Is this thread finished */

  char      *data;       /* thread private data */
  int        datasize;   /* size of thread-private data, in bytes */

  int isIsomalloc; /* thread stack was isomalloc'd*/
  void      *stack; /*Pointer to thread stack*/
  int        stacksize; /*Size of thread stack (bytes)*/
} CthThreadBase;

/*Macros to convert between base and specific thread types*/
#define B(t) ((CthThreadBase *)(t))
#define S(t) ((CthThread)(t))

/*********** Thread-local storage *********/

CthCpvStatic(CthThread,  CthCurrent); /*Current thread*/
CthCpvDeclare(char *,    CthData); /*Current thread's private data (externally visible)*/
CthCpvStatic(int,        CthDatasize);

void CthSetThreadID(CthThread th, int a, int b, int c)
{
  B(th)->tid.id[0] = a;
  B(th)->tid.id[1] = b;
  B(th)->tid.id[2] = c;
}

char *CthGetData(CthThread t) { return B(t)->data; }

/* Ensure this thread has at least enough 
room for all the thread-local variables 
initialized so far on this processor.
*/
static void CthFixData(CthThread t)
{
  int datasize = CthCpvAccess(CthDatasize);
  if (B(t)->datasize < datasize) {
    B(t)->datasize = 2*datasize;
    /* Note: realloc(NULL,size) is equivalent to malloc(size) */
    B(t)->data = (char *)realloc(B(t)->data, B(t)->datasize);
  }
}

/*
Allocate another size bytes of thread-local storage,
and return the offset into the thread storage buffer.
 */
int CthRegister(int size)
{
  int datasize=CthCpvAccess(CthDatasize);
  CthThreadBase *th=(CthThreadBase *)CthCpvAccess(CthCurrent);
  int result, align = 1;
  while (size>align) align<<=1;
  datasize = (datasize+align-1) & ~(align-1);
  result = datasize;
  datasize += size;
  CthCpvAccess(CthDatasize) = datasize;
  CthFixData(S(th)); /*Make the current thread have this much storage*/
  CthCpvAccess(CthData) = th->data;
  return result;
}

/*********** Creation and Deletion **********/
CthCpvStatic(int, _defaultStackSize);

static void CthThreadBaseInit(CthThreadBase *th)
{
  static int serialno = 1;
#if CMK_LINUX_PTHREAD_HACK
  /*HACK for LinuxThreads: to support our user-level threads
    library, we use a slightly modified version of libpthread.a
    with user-level threads support enabled via these flags.
  */
  extern int __pthread_find_self_with_pid;
  extern int __pthread_nonstandard_stacks;
  __pthread_find_self_with_pid=1;
  __pthread_nonstandard_stacks=1;
#endif

  th->awakenfn = 0;
  th->choosefn = 0;
  th->next=0;
  th->suspendable = 1;
  th->exiting = 0;

  th->data=0;
  th->datasize=0;
  CthFixData(S(th));

  CthSetStrategyDefault(S(th));

  th->isIsomalloc=0;
  th->stack=NULL;
  th->stacksize=0;

  th->tid.id[0] = CmiMyPe();
  th->tid.id[1] = serialno++;
  th->tid.id[2] = 0;
}

static void *CthAllocateStack(CthThreadBase *th,int *stackSize,int useIsomalloc)
{
  void *ret=NULL;
  if (*stackSize==0) *stackSize=CthCpvAccess(_defaultStackSize);
  th->stacksize=*stackSize;
  if (!useIsomalloc) {
    ret=malloc(*stackSize); 
  } else {
    th->isIsomalloc=1;
    ret=CmiIsomalloc(*stackSize);
  }
  _MEMCHECK(ret);
  th->stack=ret;
  return ret;
}
static void CthThreadBaseFree(CthThreadBase *th)
{
  free(th->data);
  if (th->isIsomalloc) {
	  CmiIsomallocFree(th->stack);
  } 
  else if (th->stack!=NULL) {
	  free(th->stack);
  }
  th->stack=NULL;
}

CpvDeclare(int, _numSwitches); /*Context switch count*/

static void CthBaseInit(char **argv)
{
  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(int,  _defaultStackSize);
  CthCpvAccess(_defaultStackSize)=32768;
  CmiGetArgInt(argv,"+stacksize",&CthCpvAccess(_defaultStackSize));  
  
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(char *, CthData);
  CthCpvInitialize(int,        CthDatasize);
  
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthDatasize)=0;
}

int CthImplemented() { return 1; } 

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void CthPupBase(pup_er p,CthThreadBase *t,int useIsomalloc)
{
#ifndef CMK_OPTIMIZE
	if ((CthThread)t==CthCpvAccess(CthCurrent))
		CmiAbort("CthPupBase: Cannot pack running thread!");
#endif
	/*Really need a pup_functionPtr here:*/
	pup_bytes(p,&t->awakenfn,sizeof(t->awakenfn));
	pup_bytes(p,&t->choosefn,sizeof(t->choosefn));
	pup_bytes(p,&t->next,sizeof(t->next));
	pup_int(p,&t->suspendable);
	pup_int(p,&t->datasize);
	if (pup_isUnpacking(p)) { 
		t->data = (char *) malloc(t->datasize);_MEMCHECK(t->data);
	}
	pup_bytes(p,(void *)t->data,t->datasize);
	pup_int(p,&t->isIsomalloc);
	pup_int(p,&t->stacksize);	
	if (t->isIsomalloc) {
		CmiIsomallocPup(p,&t->stack);
	} else {
		if (useIsomalloc)
			CmiAbort("You must use CthCreateMigratable to use CthPup!\n");
		/*Pup the stack pointer as raw bytes*/
		pup_bytes(p,&t->stack,sizeof(t->stack));
	}
}

static void CthThreadFinished(CthThread t)
{
	B(t)->exiting=1;
	CthSuspend();
}

/************ Scheduler Interface **********/

void CthSetSuspendable(CthThread t, int val) { B(t)->suspendable = val; }
int CthIsSuspendable(CthThread t) { return B(t)->suspendable; }

void CthSetNext(CthThread t, CthThread v) { B(t)->next = v; }
CthThread CthGetNext(CthThread t) { return B(t)->next; }

static void CthNoStrategy(void)
{
  CmiAbort("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
}

void CthSetStrategy(CthThread t, CthAwkFn awkfn, CthThFn chsfn)
{
  B(t)->awakenfn = awkfn;
  B(t)->choosefn = chsfn;
}

static void CthBaseResume(CthThread t)
{
  CpvAccess(_numSwitches)++;
  CthFixData(t); /*Thread-local storage may have changed in other thread.*/
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = B(t)->data;
}

/**
  switch the thread to t
*/
void CthSwitchThread(CthThread t)
{
  CthBaseResume(t);
}

/*
Suspend: finds the next thread to execute, and resumes it
*/
void CthSuspend(void)
{
  CthThread next;
  CthThreadBase *cur=B(CthCpvAccess(CthCurrent));
  
  if (cur->choosefn == 0) CthNoStrategy();
  next = cur->choosefn();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceSuspend();
#endif
  CthResume(next);
}

void CthAwaken(CthThread th)
{
  if (B(th)->awakenfn == 0) CthNoStrategy();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken(th);
#endif
  B(th)->awakenfn(th, CQS_QUEUEING_FIFO, 0, 0);
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthCurrent));
  CthSuspend();
}

void CthAwakenPrio(CthThread th, int s, int pb, unsigned int *prio)
{
  if (B(th)->awakenfn == 0) CthNoStrategy();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken(th);
#endif
  B(th)->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, unsigned int *prio)
{
  CthAwakenPrio(CthCpvAccess(CthCurrent), s, pb, prio);
  CthSuspend();
}


/*************************** Stack-Copying Threads (obsolete) *******************
Basic idea: switch from thread A (currently running) to thread B by copying
A's stack from the system stack area into A's buffer in the heap, then
copy B's stack from its heap buffer onto the system stack.

This allows thread migration, because the system stack is in the same
location on every processor; but the context-switching overhead (especially
for threads with deep stacks) is extremely high.

Written by Josh Yelon around May 1999
*/
#if CMK_THREADS_COPY_STACK

#define SWITCHBUF_SIZE 16384

typedef struct CthProcInfo *CthProcInfo;

typedef struct CthThreadStruct
{
  CthThreadBase base;
  CthVoidFn  startfn;    /* function that thread will execute */
  void      *startarg;   /* argument that start function will be passed */
  qt_t      *savedstack; /* pointer to saved stack */
  int        savedsize;  /* length of saved stack (zero when running) */	
  int        stacklen;   /* length of the allocated savedstack >= savedsize */
  qt_t      *savedptr;   /* stack pointer */
} CthThreadStruct;

CthThread CthPup(pup_er p, CthThread t)
{
    if (pup_isUnpacking(p))
      { t = (CthThread) malloc(sizeof(CthThreadStruct));_MEMCHECK(t);}
    pup_bytes(p, (void*) t, sizeof(CthThreadStruct)); 
    CthPupBase(p,&t->base,0);
    pup_int(p,&t->savedsize);
    if (pup_isUnpacking(p)) {
      t->savedstack = (qt_t*) malloc(t->savedsize);_MEMCHECK(t->savedstack);
    }
    pup_bytes(p, (void*) t->savedstack, t->savedsize);
    
    if (pup_isDeleting(p))
      {CthFree(t);t=0;}

    return t;
}

struct CthProcInfo
{
  qt_t      *stackbase;
  qt_t      *switchbuf_sp;
  qt_t      *switchbuf;
};

CthCpvDeclare(CthProcInfo, CthProc);

static void CthThreadInit(CthThread t, CthVoidFn fn, void *arg)
{
  CthThreadBaseInit(&t->base);
  t->startfn = fn;
  t->startarg = arg;
  t->savedstack = 0;
  t->savedsize = 0;
  t->stacklen = 0;
  t->savedptr = 0;
}

static void CthThreadFree(CthThread t)
{
  CthThreadBaseFree(&t->base);
  if (t->savedstack) free(t->savedstack);
  free(t);
}

void CthFree(t)
CthThread t;
{
  if (t==NULL) return;
  CthProcInfo proc = CthCpvAccess(CthProc);
  if (t != CthSelf()) {
    CthThreadFree(t);
  } else
    t->base.exiting = 1;
}

void CthDummy() { }

void CthInit(char **argv)
{
  CthThread t; CthProcInfo p; qt_t *switchbuf, *sp;

  CthCpvInitialize(CthProcInfo, CthProc);

  CthBaseInit(argv,t);
  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthCpvAccess(CthCurrent)=t;
  CthThreadInit(t,0,0);

  p = (CthProcInfo)malloc(sizeof(struct CthProcInfo));
  _MEMCHECK(p);
  CthCpvAccess(CthProc)=p;
  /* leave some space for current stack frame < 256 bytes */
  sp = (qt_t*)(((size_t)&t) & ~((size_t)0xFF));
  p->stackbase = QT_SP(sp, 0x100);
  switchbuf = (qt_t*)malloc(QT_STKALIGN + SWITCHBUF_SIZE);
  _MEMCHECK(switchbuf);
  switchbuf = (qt_t*)((((size_t)switchbuf)+QT_STKALIGN) & ~(QT_STKALIGN-1));
  p->switchbuf = switchbuf;
  sp = QT_SP(switchbuf, SWITCHBUF_SIZE);
  sp = QT_ARGS(sp,0,0,0,(qt_only_t*)CthDummy);
  p->switchbuf_sp = sp;

}

static void CthOnly(CthThread t, void *dum1, void *dum2)
{
  t->startfn(t->startarg);
  CthThreadFinished(t);
}

static void CthResume1(qt_t *sp, CthProcInfo proc, CthThread t)
{
  int bytes; qt_t *lo, *hi;
  CthThread old = CthCpvAccess(CthCurrent);
  CthBaseResume(t);
  if (old->base.exiting) {
    CthThreadFree(old);
  } else {
#ifdef QT_GROW_DOWN
    lo = sp; hi = proc->stackbase;
#else
    hi = sp; lo = proc->stackbase;
#endif
    bytes = ((size_t)hi)-((size_t)lo);
    if(bytes > old->stacklen) {
      if(old->savedstack) free((void *)old->savedstack);
      old->savedstack = (qt_t*)malloc(bytes);
      _MEMCHECK(old->savedstack);
      old->stacklen = bytes;
    }
    old->savedsize = bytes;
    old->savedptr = sp;
    memcpy(old->savedstack, lo, bytes);
  }
  if (t->savedstack) {
#ifdef QT_GROW_DOWN
    lo = t->savedptr;
#else
    lo = proc->stackbase;
#endif
    memcpy(lo, t->savedstack, t->savedsize);
    t->savedsize=0;
    sp = t->savedptr;
  } else {
    sp = proc->stackbase;
    sp = QT_ARGS(sp,t,0,0,(qt_only_t*)CthOnly);
  }
  QT_ABORT((qt_helper_t*)CthDummy,0,0,sp);
}

void CthResume(t)
CthThread t;
{
  CthProcInfo proc = CthCpvAccess(CthProc);
  QT_BLOCK((qt_helper_t*)CthResume1, proc, t, proc->switchbuf_sp);
}

CthThread CthCreate(CthVoidFn fn,void *arg,int size)
{
  CthThread result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result, fn, arg);
  return result;
}
CthThread CthCreateMigratable(CthVoidFn fn,void *arg,int size)
{
  /*All threads are migratable under stack copying*/
  return CthCreate(fn,arg,size);
}

/**************************************************************************
QuickThreads does not work on Win32-- our stack-shifting large allocas
fail a stack depth check.  Windows NT and 98 provide a user-level thread 
interface called "Fibers", used here.

Written by Sameer Paranjpye around October 2000
*/
#elif  CMK_THREADS_ARE_WIN32_FIBERS

#include <windows.h>
#include <winbase.h>

#ifndef _WIN32_WINNT
#define _WIN32_WINNT  0x0400
#endif

#if(_WIN32_WINNT >= 0x0400)
typedef VOID (WINAPI *PFIBER_START_ROUTINE)(
    LPVOID lpFiberParameter
    );
typedef PFIBER_START_ROUTINE LPFIBER_START_ROUTINE;
#endif

#if(_WIN32_WINNT >= 0x0400)
WINBASEAPI
LPVOID
WINAPI
CreateFiber(
    DWORD dwStackSize,
    LPFIBER_START_ROUTINE lpStartAddress,
    LPVOID lpParameter
    );

WINBASEAPI
VOID
WINAPI
DeleteFiber(
    LPVOID lpFiber
    );

WINBASEAPI
LPVOID
WINAPI
ConvertThreadToFiber(
    LPVOID lpParameter
    );

WINBASEAPI
VOID
WINAPI
SwitchToFiber(
    LPVOID lpFiber
    );

WINBASEAPI
BOOL
WINAPI
SwitchToThread(
    VOID
    );
#endif /* _WIN32_WINNT >= 0x0400 */


struct CthThreadStruct
{
  CthThreadBase base;
  LPVOID     fiber;
};

CthCpvStatic(CthThread,  CthPrevious);

typedef CthThread threadTable[100];
CthCpvStatic(threadTable, exitThreads);
CthCpvStatic(int, 	nExit);

static void CthThreadInit(CthThread t)
{
  CthThreadBaseInit(&t->base);
}

void CthInit(char **argv)
{
  CthThread t;

  CthCpvInitialize(CthThread,  CthPrevious);
  CthCpvInitialize(int,        nExit);
  CthCpvInitialize(threadTable,        exitThreads);

  CthCpvAccess(CthPrevious)=0;
  CthCpvAccess(nExit)=0;

  CthBaseInit(argv);  
  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthCpvAccess(CthCurrent)=t;
  CthThreadInit(t);
  t->fiber = ConvertThreadToFiber(t);
  _MEMCHECK(t->fiber);

}

void CthThreadFree(CthThread old)
{
  CthThreadBaseFree(&old->base);
  if (old->fiber) DeleteFiber((PVOID)old->fiber);
  free(old);
}

static void CthClearThreads()
{
  int i,p,m;
  int n = CthCpvAccess(nExit);
  CthThread tc = CthCpvAccess(CthCurrent);
  CthThread tp = CthCpvAccess(CthPrevious);
  m = n;
  p=0;
  for (i=0; i<m; i++) {
    CthThread t = CthCpvAccess(exitThreads)[i];
    if (t && t != tc && t != tp) {
      CthThreadFree(t);
      CthCpvAccess(nExit) --;
    }
    else {
      if (p != i) CthCpvAccess(exitThreads)[p] = t;
      p++;
    }
  }
  if (m!=p)
  for (i=m; i<n; i++,p++) {
      CthCpvAccess(exitThreads)[p] = CthCpvAccess(exitThreads)[i];
  }
}

void CthFree(CthThread t)
{
  if (t==NULL) return;
  /* store into exiting threads table to avoid delete thread itself */
  CthCpvAccess(exitThreads)[CthCpvAccess(nExit)++] = t;
  if (t==CthCpvAccess(CthCurrent)) 
  {
     t->base.exiting = 1;
  } 
  else 
  {
    CthClearThreads();
/*  was
    if (t->data) free(t->data);
    DeleteFiber(t->fiber);
    free(t);
*/
  }
}

#if 0
void CthFiberBlock(CthThread t)
{
  CthThread tp;
  
  SwitchToFiber(t->fiber);
  tp = CthCpvAccess(CthPrevious);
  if (tp != 0 && tp->killed == 1)
    CthThreadFree(tp);
}
#endif

void CthResume(CthThread t)
{
  CthThread tc;

  tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CthBaseResume(t);
  CthCpvAccess(CthPrevious)=tc;
#if 0
  if (tc->base.exiting) 
  {
    SwitchToFiber(t->fiber);
  } 
  else 
    CthFiberBlock(t);
#endif
  SwitchToFiber(t->fiber);
}

VOID CALLBACK FiberSetUp(PVOID fiberData)
{
  void **ptr = (void **) fiberData;
  qt_userf_t* fn = (qt_userf_t *)ptr[0];
  void *arg = ptr[1];
  CthThread  t = CthSelf();

  CthClearThreads();

  fn(arg);

  CthCpvAccess(exitThreads)[CthCpvAccess(nExit)++] = t;
  CthThreadFinished(t);
}

CthThread CthCreate(CthVoidFn fn, void *arg, int size)
{
  CthThread result; 
  void**    fiberData;
  fiberData = (void**)malloc(2*sizeof(void *));
  fiberData[0] = (void *)fn;
  fiberData[1] = arg;
  
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  result->fiber = CreateFiber(0, FiberSetUp, (PVOID) fiberData);
  if (!result->fiber)
    CmiAbort("CthCreate failed to create fiber!\n");
  
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}
CthThread CthCreateMigratable(CthVoidFn fn,void *arg,int size)
{
  /*Fibers are never migratable, unless we can figure out how to set their stacks*/
  return CthCreate(fn,arg,size);
}

/***************************************************
Use Posix Threads to simulate cooperative user-level
threads.  This version is very portable but inefficient.

Written by Milind Bhandarkar around November 2000
*/
#elif CMK_THREADS_USE_PTHREADS

#include <pthread.h>

struct CthThreadStruct
{
  CthThreadBase base;
  pthread_t  self;
  pthread_cond_t cond;
  pthread_cond_t *creator;
  CthVoidFn  fn;
  void      *arg;
  char       inited;
};


CthCpvStatic(pthread_mutex_t, sched_mutex);

static void CthThreadInit(t)
CthThread t;
{
  CthThreadBaseInit(&t->base);
  t->inited = 0;
  pthread_cond_init(&(t->cond) , (pthread_condattr_t *) 0);
}

void CthInit(char **argv)
{
  CthThread t;

  CthCpvInitialize(pthread_mutex_t, sched_mutex);

  pthread_mutex_init(&CthCpvAccess(sched_mutex), (pthread_mutexattr_t *) 0);
  pthread_mutex_lock(&CthCpvAccess(sched_mutex));
  CthBaseInit(argv); 
  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthCpvAccess(CthCurrent)=t;
  CthThreadInit(t);
  t->self = pthread_self();
}

void CthFree(t)
CthThread t;
{
  if (t==NULL) return;
  if (t==CthCpvAccess(CthCurrent)) {
    t->base.exiting = 1;
  } else {
    CthThreadBaseFree(&t->base);
    free(t);
  }
}

void CthResume(CthThread t)
{
  CthThread tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CthBaseResume(t);
  pthread_cond_signal(&(t->cond));
  if (tc->base.exiting) {
    pthread_mutex_unlock(&CthCpvAccess(sched_mutex));
    pthread_exit(0);
  } else {
    /* pthread_cond_wait might (with low probability) return when the 
      condition variable has not been signaled, guarded with 
      predicate checks */
    do {
    pthread_cond_wait(&(tc->cond), &CthCpvAccess(sched_mutex));
    } while (tc!=CthCpvAccess(CthCurrent)) ;
  }
}

static void *CthOnly(CthThread arg)
{
  arg->inited = 1;
  pthread_mutex_lock(&CthCpvAccess(sched_mutex));
  pthread_cond_signal(arg->creator);
  do {
  pthread_cond_wait(&(arg->cond), &CthCpvAccess(sched_mutex));
  } while (arg!=CthCpvAccess(CthCurrent)) ;
  arg->fn(arg->arg);
  CthThreadFinished(arg);
  return 0;
}

CthThread CthCreate(CthVoidFn fn, void *arg, int size)
{
  CthThread result;
  CthThread self = CthSelf();
  /* size is ignored in this version */
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  result->fn = fn;
  result->arg = arg;
  result->creator = &(self->cond);
  if (0 != pthread_create(&(result->self), (pthread_attr_t *) 0, CthOnly, (void*) result)) 
    CmiAbort("CthCreate failed to created a new pthread\n");
  do {
  pthread_cond_wait(&(self->cond), &CthCpvAccess(sched_mutex));
  } while (result->inited==0);
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}
CthThread CthCreateMigratable(CthVoidFn fn,void *arg,int size)
{
  /*Pthreads are never migratable, unless we can figure out how to set their stacks*/
  return CthCreate(fn,arg,size);
}

/***************************************************************
Use SysV r3 setcontext/getcontext calls instead of
quickthreads.  This works on some architectures (such as
IA64) where quickthreads' setjmp/alloca/longjmp fails.

Written by Gengbin Zheng around April 2001
*/
#elif  CMK_THREADS_USE_CONTEXT

#include <signal.h>
#include <ucontext.h>

struct CthThreadStruct
{
  CthThreadBase base;
  ucontext_t context;
};


static void CthThreadInit(t)
CthThread t;
{
  CthThreadBaseInit(&t->base);
}

void CthInit(char **argv)
{
  CthThread t;

  CthBaseInit(argv);
  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthCpvAccess(CthCurrent)=t;
  CthThreadInit(t);
}

static void CthThreadFree(CthThread t)
{
  CthThreadBaseFree(&t->base);
  free(t);
}

void CthFree(CthThread t)
{
  if (t==NULL) return;
  if (t==CthCpvAccess(CthCurrent)) {
    t->base.exiting = 1;
  } else {
    CthThreadFree(t);
  }
}

void CthResume(t)
CthThread t;
{
  CthThread tc;
  tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CthBaseResume(t);
  if (tc->base.exiting) {
    CthThreadFree(tc);
    setcontext(&t->context);
  } else {
    if (0 != swapcontext(&tc->context, &t->context)) 
      CmiAbort("CthResume: swapcontext failed.\n");
  }
/*This check will mistakenly fail if the thread migrates (changing tc)
  if (tc!=CthCpvAccess(CthCurrent)) { CmiAbort("Stack corrupted?\n"); }
*/
}

void CthStartThread(qt_userf_t fn,void *arg)
{
  fn(arg);
  CthThreadFinished(CthSelf());
}

#define STP_STKALIGN(sp, alignment) \
  ((void *)((((qt_word_t)(sp)) + (alignment) - 1) & ~((alignment)-1)))

static CthThread CthCreateInner(CthVoidFn fn,void *arg,int size,int migratable)
{
  CthThread result;
  char *stack;
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  size += SIGSTKSZ;
  CthAllocateStack(&result->base,&size,migratable);
  stack = result->base.stack;
#if CMK_STACK_GROWDOWN
  stack = stack +  size - MINSIGSTKSZ;
  size = stack - (char *)result->base.stack;
#endif
  if (0 != getcontext(&result->context))
    CmiAbort("CthCreateInner: getcontext failed.\n");
  result->context.uc_stack.ss_sp = stack;
  result->context.uc_stack.ss_size = size;
  result->context.uc_stack.ss_flags = 0;
  result->context.uc_link = 0;
  makecontext(&result->context, (void (*) (void))CthStartThread, 2, fn, arg);
  return result;  
}

CthThread CthCreate(CthVoidFn fn,void *arg,int size)
{
  return CthCreateInner(fn,arg,size,0);
}
CthThread CthCreateMigratable(CthVoidFn fn,void *arg,int size)
{
  return CthCreateInner(fn,arg,size,1);
}

CthThread CthPup(pup_er p, CthThread t)
{
  if (pup_isUnpacking(p)) {
	  t=(CthThread)malloc(sizeof(struct CthThreadStruct));
	  _MEMCHECK(t);
  }
  CthPupBase(p,&t->base,1);
  
  /*Pup the processor context as bytes-- this is not guarenteed to work!*/
  pup_bytes(p,&t->context,sizeof(t->context));
  if (pup_isDeleting(p)) {
	  CthFree(t);
  }
  return 0;
}

#else 
/***************************************************************
Basic qthreads implementation. 

These threads can also add a "protection block" of
inaccessible memory to detect stack overflows, which
would otherwise just trash the heap.

(7/13/2001 creation times on 300MHz AMD K6-3 x86, Linux 2.2.18:
Qt setjmp, without stackprotect: 18.5 us
Qt i386, without stackprotect: 17.9 us
Qt setjmp, with stackprotect: 68.6 us
)

Written by Josh Yelon around 1995
*/

#if defined(CMK_OPTIMIZE) || (!CMK_MEMORY_PROTECTABLE)
#  define CMK_STACKPROTECT 0

#  define CthMemAlign(x,n) 0
#  define CthMemoryProtect(p,l) CmiAbort("Shouldn't call CthMemoryProtect!\n")
#  define CthMemoryUnprotect(p,l) CmiAbort("Shouldn't call CthMemoryUnprotect!\n")
#else
#  define CMK_STACKPROTECT 1

#  include "sys/mman.h"
#  define CthMemAlign(x,n) memalign((x),(n))
#  define CthMemoryProtect(p,l) mprotect(p,l,PROT_NONE)
#  define CthMemoryUnprotect(p,l) mprotect(p,l,PROT_READ | PROT_WRITE)
#endif

struct CthThreadStruct
{
  CthThreadBase base;

  char      *protect;
  int        protlen;

  qt_t      *stack;
  qt_t      *stackp;
};

static CthThread CthThreadInit(void)
{
  CthThread ret=(CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(ret);
  CthThreadBaseInit(&ret->base);
  ret->protect = 0;
  ret->protlen = 0;
  
  return ret;
}

static void CthThreadFree(CthThread t)
{
  CthThreadBaseFree(&t->base);
  if (t->protlen!=0) {
    CthMemoryUnprotect(t->protect, t->protlen);
    free(t->stack);
  }
  free(t);
}

void CthInit(char **argv)
{
  CthThread mainThread;

  CthBaseInit(argv);  
  mainThread=CthThreadInit();
  CthCpvAccess(CthCurrent)=mainThread;
  mainThread->base.suspendable=0; /*Can't suspend main thread (trashes Quickthreads jump buffer)*/
}

void CthFree(CthThread t)
{
  if (t==NULL) return;
  if (t==CthCpvAccess(CthCurrent)) {
    t->base.exiting = 1;
  } else {
    CthThreadFree(t);
  }
}

static void *CthAbortHelp(qt_t *sp, CthThread old, void *null)
{
  CthThreadFree(old);
  return (void *) 0;
}

static void *CthBlockHelp(qt_t *sp, CthThread old, void *null)
{
  old->stackp = sp;
  return (void *) 0;
}

void CthResume(t)
CthThread t;
{
  CthThread tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CthBaseResume(t);
  if (tc->base.exiting) {
    QT_ABORT((qt_helper_t*)CthAbortHelp, tc, 0, t->stackp);
  } else {
    QT_BLOCK((qt_helper_t*)CthBlockHelp, tc, 0, t->stackp);
  }
/*This check will mistakenly fail if the thread migrates (changing tc)
  if (tc!=CthCpvAccess(CthCurrent)) { CmiAbort("Stack corrupted?\n"); }
*/
}

static void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  fn(arg);
  CthThreadFinished(CthSelf());
}

static CthThread CthCreateInner(CthVoidFn fn, void *arg, int size,int isomalloc)
{
  CthThread result; qt_t *stack, *stackbase, *stackp;
  int doProtect=(!isomalloc) && CMK_STACKPROTECT;
  result=CthThreadInit();
  if (doProtect) 
  { /*Can only protect on a page boundary-- allocate an extra page and align stack*/
	  if (size==0) size=CthCpvAccess(_defaultStackSize);
	  size = (size+(CMK_MEMORY_PAGESIZE*2)-1) & ~(CMK_MEMORY_PAGESIZE-1);
	  stack = (qt_t*)CthMemAlign(CMK_MEMORY_PAGESIZE, size);
  } else
	  stack=CthAllocateStack(&result->base,&size,isomalloc);
  stackbase = QT_SP(stack, size);
  stackp = QT_ARGS(stackbase, arg, result, (qt_userf_t *)fn, CthOnly);
  result->stack = stack;
  result->stackp = stackp;
  if (doProtect) {
#ifdef QT_GROW_UP
  /*Stack grows up-- protect highest page of stack*/
    result->protect = ((char*)stack) + size - CMK_MEMORY_PAGESIZE;
#else
  /*Stack grows down-- protect lowest page in stack*/
    result->protect = ((char*)stack);
#endif
    result->protlen = CMK_MEMORY_PAGESIZE;
    CthMemoryProtect(result->protect, result->protlen);
  }
  return result;
}
CthThread CthCreate(CthVoidFn fn, void *arg, int size)
{ return CthCreateInner(fn,arg,size,0);}

CthThread CthCreateMigratable(CthVoidFn fn, void *arg, int size)
{ return CthCreateInner(fn,arg,size,1);}

CthThread CthPup(pup_er p, CthThread t)
{
  if (pup_isUnpacking(p)) {
	  t=CthThreadInit();
  }
  CthPupBase(p,&t->base,1);
  
  /*Pup the stack pointer as bytes-- this works because stack is isomalloc'd*/
  pup_bytes(p,&t->stackp,sizeof(t->stackp));

  /*Don't worry about stack protection on migration*/  

  if (pup_isDeleting(p)) {
	  CthFree(t);
	  return 0;
  }
  return t;
}

void CthTraceResume(CthThread t)
{
  traceResume(&t->base.tid);
}

#endif
