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
  
  CthAwkFn   awakenfn;   /* Insert this thread into the ready queue */
  CthThFn    choosefn;   /* Return the next ready thread */
  CthThread  next; /* Next active thread */
  int        suspendable; /* Can this thread be blocked */
  int        exiting;    /* Is this thread finished */

  char      *data;       /* thread private data */
  int        datasize;   /* size of thread-private data, in bytes */
} CthThreadBase;

/*Macros to convert between base and specific thread types*/
#define B(t) ((CthThreadBase *)(t))
#define S(t) ((CthThread)(t))

static void CthThreadBaseInit(CthThreadBase *th)
{
  th->awakenfn = 0;
  th->choosefn = 0;
  th->next=0;
  th->suspendable = 1;
  th->exiting = 0;

  th->data=0;
  th->datasize=0;

  CthSetStrategyDefault(S(th));
}
static void CthThreadBaseFree(CthThreadBase *th)
{
  free(th->data);
}

CthCpvStatic(int, _defaultStackSize);

CpvDeclare(int, _numSwitches); /*Context switch count*/
CthCpvStatic(CthThread,  CthCurrent); /*Current thread*/
CthCpvDeclare(char *,    CthData); /*Current thread's private data (externally visible)*/
CthCpvStatic(int,        CthDatasize);

static void CthBaseInit(char **argv,CthThread mainThread)
{
  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(int,  _defaultStackSize);
  CthCpvAccess(_defaultStackSize)=32768;
  CmiGetArgInt(argv,"+stacksize",&CthCpvAccess(_defaultStackSize));  
  
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(char *, CthData);
  CthCpvInitialize(int,        CthDatasize);
  
  CthCpvAccess(CthCurrent)=mainThread;
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthDatasize)=0;
}

int CthImplemented() { return 1; } 

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void CthPupBase(pup_er p,CthThreadBase *t)
{
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
}

static void CthThreadFinished(CthThread t)
{
	B(t)->exiting=1;
	CthSuspend();
}

/*********** Thread-local storage *********/

char *CthGetData(CthThread t) { return B(t)->data; }

/* Ensure this thread has at least enough 
room for all the thread-local variables 
initialized so far on this processor.
*/
static void CthFixData(CthThread t)
{
  int datasize = CthCpvAccess(CthDatasize);
  if (B(t)->datasize < datasize) {
    B(t)->datasize = datasize;
    /* Note: realloc(NULL,size) is equivalent to malloc(size) */
    B(t)->data = (char *)realloc(B(t)->data, datasize);
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

#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceResume();
#endif
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
#ifndef CMK_OPTIMIZE
    if (pup_isPacking(p))
    {
      if (t->savedsize == 0)
        CmiAbort("Trying to pack a running thread!!\n");
    }
#endif
    if (pup_isUnpacking(p))
      { t = (CthThread) malloc(sizeof(CthThreadStruct));_MEMCHECK(t);}
    pup_bytes(p, (void*) t, sizeof(CthThreadStruct)); 
    CthPupBase(p,&t->base);
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

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t,0,0);
  CthBaseInit(argv,t);

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

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result, fn, arg);
  return result;
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

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t);
  t->fiber = ConvertThreadToFiber(t);
  _MEMCHECK(t->fiber);
  CthBaseInit(argv,t);

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

/**********************************************************************
Efficient migratable threads.  This approach reserves virtual address space
for every thread's stack, on every processor, across the machine.  This allows
thread migration with quick context switching.

Written by Milind Bhandarkar around August 2000
*/
#elif CMK_THREADS_USE_ISOMALLOC

#include <stdlib.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

typedef struct _slotblock
{
  int startslot;
  int nslots;
} slotblock;

typedef struct _slotset
{
  int maxbuf;
  slotblock *buf;
  int emptyslots;
} slotset;

/*
 * creates a new slotset of nslots entries, starting with all
 * empty slots. The slot numbers are [startslot,startslot+nslot-1]
 */
static slotset *
new_slotset(int startslot, int nslots)
{
  int i;
  slotset *ss = (slotset*) malloc(sizeof(slotset));
  _MEMCHECK(ss);
  ss->maxbuf = 16;
  ss->buf = (slotblock *) malloc(sizeof(slotblock)*ss->maxbuf);
  _MEMCHECK(ss->buf);
  ss->emptyslots = nslots;
  ss->buf[0].startslot = startslot;
  ss->buf[0].nslots = nslots;
  for (i=1; i<ss->maxbuf; i++)
    ss->buf[i].nslots = 0;
  return ss;
}

/*
 * returns new block of empty slots. if it cannot find any, returns (-1).
 */
static int
get_slots(slotset *ss, int nslots)
{
  int i;
  if(ss->emptyslots < nslots)
    return (-1);
  for(i=0;i<(ss->maxbuf);i++)
    if(ss->buf[i].nslots >= nslots)
      return ss->buf[i].startslot;
  return (-1);
}

/* just adds a slotblock to an empty position in the given slotset. */
static void
add_slots(slotset *ss, int sslot, int nslots)
{
  int pos, emptypos = -1;
  if (nslots == 0)
    return;
  for (pos=0; pos < (ss->maxbuf); pos++) {
    if (ss->buf[pos].nslots == 0) {
      emptypos = pos;
      break; /* found empty slotblock */
    }
  }
  if (emptypos == (-1)) /*no empty slotblock found */
  {
    int i;
    int newsize = ss->maxbuf*2;
    slotblock *newbuf = (slotblock *) malloc(sizeof(slotblock)*newsize);
    _MEMCHECK(newbuf);
    for (i=0; i<(ss->maxbuf); i++)
      newbuf[i] = ss->buf[i];
    for (i=ss->maxbuf; i<newsize; i++)
      newbuf[i].nslots  = 0;
    free(ss->buf);
    ss->buf = newbuf;
    emptypos = ss->maxbuf;
    ss->maxbuf = newsize;
  }
  ss->buf[emptypos].startslot = sslot;
  ss->buf[emptypos].nslots = nslots;
  ss->emptyslots += nslots;
  return;
}

/* grab a slotblock with specified range of blocks
 * this is different from get_slots, since it pre-specifies the
 * slots to be grabbed.
 */
static void
grab_slots(slotset *ss, int sslot, int nslots)
{
  int pos, eslot, e;
  eslot = sslot + nslots;
  for (pos=0; pos < (ss->maxbuf); pos++)
  {
    if (ss->buf[pos].nslots == 0)
      continue;
    e = ss->buf[pos].startslot + ss->buf[pos].nslots;
    if(sslot >= ss->buf[pos].startslot && eslot <= e)
    {
      int old_nslots;
      old_nslots = ss->buf[pos].nslots;
      ss->buf[pos].nslots = sslot - ss->buf[pos].startslot;
      ss->emptyslots -= (old_nslots - ss->buf[pos].nslots);
      add_slots(ss, sslot + nslots, old_nslots - ss->buf[pos].nslots - nslots);
      return;
    }
  }
  CmiAbort("requested a non-existent slotblock\n");
}

/*
 * Frees slot by adding it to one of the blocks of empty slots.
 * this slotblock is one which is contiguous with the slots to be freed.
 * if it cannot find such a slotblock, it creates a new slotblock.
 * If the buffer fills up, it adds up extra buffer space.
 */
static void
free_slots(slotset *ss, int sslot, int nslots)
{
  int pos;
  /* eslot is the ending slot of the block to be freed */
  int eslot = sslot + nslots;
  for (pos=0; pos < (ss->maxbuf); pos++)
  {
    int e = ss->buf[pos].startslot + ss->buf[pos].nslots;
    if (ss->buf[pos].nslots == 0)
      continue;
    /* e is the ending slot of pos'th slotblock */
    if (e == sslot) /* append to the current slotblock */
    {
	    ss->buf[pos].nslots += nslots;
      ss->emptyslots += nslots;
	    return;
    }
    if(eslot == ss->buf[pos].startslot) /* prepend to the current slotblock */
    {
	    ss->buf[pos].startslot = sslot;
	    ss->buf[pos].nslots += nslots;
      ss->emptyslots += nslots;
	    return;
    }
  }
  /* if we are here, it means we could not find a slotblock that the */
  /* block to be freed was combined with. */
  add_slots(ss, sslot, nslots);
}

/*
 * destroys slotset
 */
static void
delete_slotset(slotset* ss)
{
  free(ss->buf);
  free(ss);
}

#if CMK_THREADS_DEBUG
static void
print_slots(slotset *ss)
{
  int i;
  CmiPrintf("[%d] maxbuf = %d\n", CmiMyPe(), ss->maxbuf);
  CmiPrintf("[%d] emptyslots = %d\n", CmiMyPe(), ss->emptyslots);
  for(i=0;i<ss->maxbuf;i++) {
    if(ss->buf[i].nslots)
      CmiPrintf("[%d] (%d, %d) \n", CmiMyPe(), ss->buf[i].startslot, 
          ss->buf[i].nslots);
  }
}
#endif

/*
 * this message is used both as a request and a reply message.
 * request:
 *   pe is the processor requesting slots.
 *   t,fn,arg are the thread-details for which these slots are requested.
 * reply:
 *   pe is the processor that responds
 *   slot is the starting slot that it sends
 *   t is the thread for which this slot is used.
 */
typedef struct _slotmsg
{
  char cmicore[CmiMsgHeaderSizeBytes];
  int pe;
  int slot;
  int nslots;
  CthThread t;
  CthVoidFn fn;
  void *arg;
} slotmsg;

struct CthThreadStruct
{
  CthThreadBase base;
  int        slotnum;
  int        nslots;
  int        awakened;
  qt_t      *stack;
  qt_t      *stackp;
};

CpvStaticDeclare(int, reqHdlr);
CpvStaticDeclare(int, reqSpecHdlr);
CpvStaticDeclare(int, respHdlr);
CpvStaticDeclare(int, respSpecHdlr);
CpvStaticDeclare(slotset *, myss);
CpvStaticDeclare(void *, heapbdry);
CpvStaticDeclare(int, zerofd);
CpvStaticDeclare(int, numslots);


/* 
 * this handler responds to a specific request for slotblock
 */
static void
reqspecific(slotmsg *msg)
{
  grab_slots(CpvAccess(myss),msg->slot,msg->nslots);
  CmiSetHandler(msg,CpvAccess(respSpecHdlr));
  CmiSyncSendAndFree(msg->pe, sizeof(slotmsg), msg);
}

/* 
 * this handler is invoked as a response to a specific request for slotblock
 */
static void
respspecific(slotmsg *msg)
{
  CthThread t = msg->t;
  t->slotnum = msg->slot;
  if(t->awakened)
    CthAwaken(t);
  CmiFree(msg);
}

/*
 * this handler is invoked by a request for slot.
 */
static void 
reqslots(slotmsg *msg)
{
  int slot, pe;
  if(msg->pe == CmiMyPe())
    CmiAbort("All stack slots have been exhausted!\n");
  slot = get_slots(CpvAccess(myss),msg->nslots);
  if(slot==(-1))
  {
    CmiSyncSendAndFree((CmiMyPe()+1)%CmiNumPes(), sizeof(slotmsg), msg);
  }
  else
  {
    pe = msg->pe;
    msg->pe = CmiMyPe();
    msg->slot = slot;
    grab_slots(CpvAccess(myss), msg->slot, msg->nslots);
    CmiSetHandler(msg,CpvAccess(respHdlr));
    CmiSyncSendAndFree(pe, sizeof(slotmsg), msg);
  }
}

static void CthStackCreate(CthThread t, CthVoidFn fn, void *arg, 
			   int slotnum, int nslots);

/*
 * this handler is invoked by a response for slots.
 * it sets the slot number and nslots for the thread, and actually awakens it
 * if it was awakened with CthAwaken already.
 */
static void 
respslots(slotmsg *msg)
{
  CthThread t = msg->t;
  if(t->slotnum != (-2))
  {
    CmiError("[%d] requested a slot for a live thread? aborting.\n",CmiMyPe());
    CmiAbort("");
  }
  CthStackCreate(t, msg->fn, msg->arg, msg->slot, msg->nslots);
  if(t->awakened)
    CthAwaken(t);
  CmiFree(msg);
}


static void CthThreadInit(t)
CthThread t;
{
  CthThreadBaseInit(&t->base);
  t->slotnum = (-1);
  t->nslots = 0;
  t->awakened = 0;
}

/*
 * maps the virtual memory associated with slot using mmap
 */
static void *
map_slots(int slot, int nslots)
{
  void *pa;
  char *addr;
  size_t sz = CpvAccess(_defaultStackSize);
  addr = (char*) CpvAccess(heapbdry) + slot*sz;
  pa = mmap((void*) addr, sz*nslots, 
            PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED,
            CpvAccess(zerofd), 0);
  if((pa==((void*)(-1))) || (pa != (void*)addr)) {
    CmiError("mmap call failed to allocate %d bytes at %p.\n", sz*nslots, addr);
    CmiAbort("Exiting\n");
  }
  return pa;
}

/*
 * unmaps the virtual memory associated with slot using munmap
 */
static void
unmap_slots(int slot, int nslots)
{
  size_t sz = CpvAccess(_defaultStackSize);
  char *addr = (char*) CpvAccess(heapbdry) + slot*sz;
  int retval = munmap((void*) addr, sz*nslots);
  if (retval==(-1))
    CmiAbort("munmap call failed to deallocate requested memory.\n");
}
static void *__cur_stack_frame(void)
{
  char __dummy;
  void *top_of_stack=(void *)&__dummy;
  return top_of_stack;
}

void CthInit(char **argv)
{
  CthThread t;
  int i;

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t);
  CthBaseInit(argv,t);

  /*Round stack size up to nearest page size*/
  CpvAccess(_defaultStackSize) = (CpvAccess(_defaultStackSize)
	+CMK_MEMORY_PAGESIZE-1) & ~(CMK_MEMORY_PAGESIZE-1);
#if CMK_THREADS_DEBUG
  CmiPrintf("[%d] Using stacksize of %d\n", CmiMyPe(), CpvAccess(_defaultStackSize));
#endif

  CpvInitialize(void *, heapbdry);
  CpvInitialize(int, zerofd);
  CpvInitialize(int, numslots);
  /*
   * calculate the number of slots according to stacksize
   * divide up into number of available processors
   * and allocate the slotset.
   */
  do {
    void *heap = (void*) malloc(1);
    void *stack = __cur_stack_frame();
    void *stackbdry;
    int stacksize = CpvAccess(_defaultStackSize);
    _MEMCHECK(heap);
#if CMK_THREADS_DEBUG
    CmiPrintf("[%d] heap=%p\tstack=%p\n",CmiMyPe(),heap,stack);
#endif
    /* Align heap to a 1G boundary to leave space to grow */
    /* Align stack to a 256M boundary  to leave space to grow */
#ifdef QT_GROW_UP
    CpvAccess(heapbdry) = (void *)(((size_t)heap-(2<<30))&(~((1<<30)-1)));
    stackbdry = (void *)(((size_t)stack+(1<<28))&(~((1<<28)-1)));
    CpvAccess(numslots) = (((size_t)CpvAccess(heapbdry)-(size_t)stackbdry)/stacksize)
                 / CmiNumPes();
#else
    CpvAccess(heapbdry) = (void *)(((size_t)heap+(2<<30))&(~((1<<30)-1)));
    stackbdry = (void *)(((size_t)stack-(1<<28))&(~((1<<28)-1)));
    CpvAccess(numslots) = (((size_t)stackbdry-(size_t)CpvAccess(heapbdry))/stacksize)
                 / CmiNumPes();
#endif
#if CMK_THREADS_DEBUG
    CmiPrintf("[%d] heapbdry=%p\tstackbdry=%p\n",
              CmiMyPe(),CpvAccess(heapbdry),stackbdry);
    CmiPrintf("[%d] numthreads per pe=%d\n",CmiMyPe(),CpvAccess(numslots));
#endif
    CpvAccess(zerofd) = open("/dev/zero", O_RDWR);
    if(CpvAccess(zerofd)<0)
      CmiAbort("Cannot open /dev/zero. Aborting.\n");
    free(heap);
  } while(0);

  CpvInitialize(slotset *, myss);
  CpvAccess(myss) = new_slotset(CmiMyPe()*CpvAccess(numslots), CpvAccess(numslots));

}

void CthHandlerInit(void)
{
  CpvInitialize(int, reqHdlr);
  CpvInitialize(int, respHdlr);
  CpvInitialize(int, reqSpecHdlr);
  CpvInitialize(int, respSpecHdlr);

  CpvAccess(reqHdlr) = CmiRegisterHandler((CmiHandler)reqslots);
  CpvAccess(respHdlr) = CmiRegisterHandler((CmiHandler)respslots);
  CpvAccess(reqSpecHdlr) = CmiRegisterHandler((CmiHandler)reqspecific);
  CpvAccess(respSpecHdlr) = CmiRegisterHandler((CmiHandler)respspecific);
}

static void
CthFreeKeepSlots(CthThread t)
{
  if (t==CthCpvAccess(CthCurrent)) {
    t->base.exiting = 1;
  } else {
    CthThreadBaseFree(&t->base);
    if(t->slotnum >= 0) unmap_slots(t->slotnum, t->nslots);
    free(t);
  }
}

void CthFree(t)
CthThread t;
{
  if(t->slotnum >= 0) free_slots(CpvAccess(myss), t->slotnum, t->nslots);
  CthFreeKeepSlots(t);
}

static void *
CthAbortHelp(qt_t *sp, CthThread old, void *null)
{
  CthThreadBaseFree(&old->base);  
  if(old->slotnum >= 0)
  {
    free_slots(CpvAccess(myss), old->slotnum, old->nslots);
    unmap_slots(old->slotnum, old->nslots);
  }
  free(old);
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
  /* NOTE: if thread migrated, CthCurrent will not be equal to tc.
  if (tc!=CthCpvAccess(CthCurrent)) { CmiAbort("Stack corrupted?\n"); }
  */
}

static void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  fn(arg);
  CthThreadFinished(CthSelf());
}

static void
CthStackCreate(CthThread t, CthVoidFn fn, void *arg, int slotnum, int nslots)
{
  qt_t *stack, *stackbase, *stackp;
  int size = CpvAccess(_defaultStackSize);
  t->slotnum = slotnum;
  t->nslots = nslots;
  stack = (qt_t*) map_slots(slotnum, nslots);
  _MEMCHECK(stack);
  stackbase = QT_SP(stack, size);
  stackp = QT_ARGS(stackbase, arg, t, (qt_userf_t *)fn, CthOnly);
  t->stack = stack;
  t->stackp = stackp;
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread result; 
  int slotnum, nslots = 1;
  if(size>CpvAccess(_defaultStackSize))
    nslots = (int) (size/CpvAccess(_defaultStackSize)) + 1;
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  slotnum = get_slots(CpvAccess(myss), nslots);
  if(slotnum == (-1)) /* mype does not have any free slots left */
  {
    slotmsg msg;
    result->slotnum = (-2);
    msg.pe = CmiMyPe();
    msg.slot = (-1); /* just to be safe */
    msg.nslots = nslots;
    msg.t = result;
    msg.fn = fn;
    msg.arg = arg;
    CmiSetHandler((void*)&msg, CpvAccess(reqHdlr));
    CmiSyncSend((CmiMyPe()+1)%CmiNumPes(), sizeof(msg), (void*)&msg);
  }
  else
  {
    grab_slots(CpvAccess(myss), slotnum, nslots);
    CthStackCreate(result, fn, arg, slotnum, nslots);
  }
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  qt_t *stackbase,*stack;
  int ssz;
  int i;

#ifndef CMK_OPTIMIZE
  if (pup_isPacking(p))
  {
    if (CthCpvAccess(CthCurrent) == t)
      CmiAbort("Trying to pack a running thread!!\n");
    if(t->slotnum < 0)
      CmiAbort("Trying to pack a thread that is not migratable!!\n");
  }
#endif
  if (pup_isUnpacking(p)) {
    t = (CthThread) malloc(sizeof(struct CthThreadStruct));
    _MEMCHECK(t);
  }
  pup_bytes(p, (void*)t, sizeof(struct CthThreadStruct));
  CthPupBase(p,&t->base);

  if (pup_isUnpacking(p)) {
    int homePe = t->slotnum/CpvAccess(numslots);
    /* 
     * allocate the stack whether I am the homePe or not. Adjust slotset
     * by sending a message later.
     */
    stack = (qt_t*) map_slots(t->slotnum, t->nslots);
    _MEMCHECK(stack);
    if(stack != t->stack)
      CmiAbort("Stack pointers do not match after migration!!\n");  
    if(pup_isUserlevel(p)) 
    {
      if(homePe == CmiMyPe())
      {
        grab_slots(CpvAccess(myss), t->slotnum, t->nslots);
      } else {
        slotmsg *msg = (slotmsg*) CmiAlloc(sizeof(slotmsg));
        _MEMCHECK(msg);
        msg->pe = CmiMyPe();
        msg->slot = t->slotnum;
        t->slotnum = (-2);
        msg->nslots = t->nslots;
        msg->t = t;
        CmiSetHandler(msg, CpvAccess(reqSpecHdlr));
        CmiSyncSendAndFree(homePe, sizeof(slotmsg), msg);
      }
    }
  }

  stackbase = QT_SP(t->stack, CpvAccess(_defaultStackSize));
#ifdef QT_GROW_UP
  ssz = ((char*)(t->stackp)-(char*)(stackbase));
  pup_bytes(p, (void*)stackbase, ssz);
#else
  ssz = ((char*)(stackbase)-(char*)(t->stackp));
  pup_bytes(p, (void*)t->stackp, ssz);
#endif

  if(pup_isDeleting(p))
  {
    CthFreeKeepSlots(t);
    t = 0;
  }
  return t;
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
  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t);
  t->self = pthread_self();
  CthBaseInit(argv,t);
}

void CthFree(t)
CthThread t;
{
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
  char      *stack;
};


static void CthThreadInit(t)
CthThread t;
{
  CthThreadBaseInit(&t->base);
  t->stack=NULL;
}

void CthInit(char **argv)
{
  CthThread t;

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t);
  CthBaseInit(argv,t);
}

static void CthThreadFree(CthThread t)
{
  CthThreadBaseFree(&t->base);
  free(t->stack);
  free(t);
}

void CthFree(CthThread t)
{
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
  if (tc!=CthCpvAccess(CthCurrent)) { CmiAbort("Stack corrupted?\n"); }
}

void CthStartThread(qt_userf_t fn,void *arg)
{
  fn(arg);
  CthThreadFinished(CthSelf());
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread result; char *stack;
  if (size==0) size = CthCpvAccess(_defaultStackSize);
  stack = (char*)malloc(size);
  _MEMCHECK(stack);
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  result->stack = stack;
  getcontext(&result->context);
  result->context.uc_stack.ss_sp = stack;
  result->context.uc_stack.ss_size = size;
  result->context.uc_stack.ss_flags = 0;
  result->context.uc_link = 0;
  makecontext(&result->context, (void (*) (void))CthStartThread, 2, fn, arg);
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}

#else 
/***************************************************************
Non-migratable, stack-on-heap threads implementation. 
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
#if CMK_STACKPROTECT
  char      *protect;
  int        protlen;
#endif
  qt_t      *stack;
  qt_t      *stackp;
};

static CthThread CthThreadInit(void)
{
  CthThread ret=(CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(ret);
  CthThreadBaseInit(&ret->base);
#if CMK_STACKPROTECT
  ret->protect = 0;
  ret->protlen = 0;
#endif
  
  return ret;
}

static void CthThreadFree(CthThread t)
{
  CthThreadBaseFree(&t->base);
#if CMK_STACKPROTECT
  CthMemoryUnprotect(t->protect, t->protlen);
#endif  
  free(t->stack);
  free(t);
}

void CthInit(char **argv)
{
  CthThread mainThread;

  mainThread=CthThreadInit();
  mainThread->base.suspendable=0; /*Can't suspend main thread (trashes Quickthreads jump buffer)*/
  CthBaseInit(argv,mainThread);
}

void CthFree(CthThread t)
{
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
  if (tc!=CthCpvAccess(CthCurrent)) { CmiAbort("Stack corrupted?\n"); }
}

static void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  fn(arg);
  CthThreadFinished(CthSelf());
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread result; qt_t *stack, *stackbase, *stackp;
  if (size==0)
    size = CthCpvAccess(_defaultStackSize);
#if CMK_STACKPROTECT
  size = (size+(CMK_MEMORY_PAGESIZE*2)-1) & ~(CMK_MEMORY_PAGESIZE-1);
  stack = (qt_t*)CthMemAlign(CMK_MEMORY_PAGESIZE, size);
#else
  stack = (qt_t*)malloc(size);
#endif
  _MEMCHECK(stack);
  result=CthThreadInit();
  stackbase = QT_SP(stack, size);
  stackp = QT_ARGS(stackbase, arg, result, (qt_userf_t *)fn, CthOnly);
  result->stack = stack;
  result->stackp = stackp;
#if CMK_STACKPROTECT
#ifdef QT_GROW_UP
  /*Stack grows up-- barrier at high end of stack*/
  result->protect = ((char*)stack) + size - CMK_MEMORY_PAGESIZE;
#else
  /*Stack grows down-- barrier at low end of stack*/
  result->protect = ((char*)stack);
#endif
  result->protlen = CMK_MEMORY_PAGESIZE;
  CthMemoryProtect(result->protect, result->protlen);
#endif
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}
#endif
