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
 * Note: there are several possible ways to implement threads.   No one
 * way works on all machines.  Instead, we provide one implementation which
 * covers a lot of machines, and a second implementation that simply prints
 * "Not Implemented".  We may provide other implementations in the future.
 *
 *****************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if CMK_MEMORY_PROTECTABLE
#include <malloc.h> /*<- for memalign*/
#endif

#include "converse.h"
#ifndef _WIN32
#include "qt.h"
#endif

#if CMK_THREADS_USE_PTHREADS
#include <pthread.h>
#endif

#include "conv-trace.h"
#include <sys/types.h>

static void CthNoStrategy(void);

CpvDeclare(int, _numSwitches);

#if CMK_THREADS_COPY_STACK

#define SWITCHBUF_SIZE 16384

typedef struct CthProcInfo *CthProcInfo;

typedef struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes];
  CthAwkFn  awakenfn;
  CthThFn    choosefn;
  CthVoidFn  startfn;    /* function that thread will execute */
  void      *startarg;   /* argument that start function will be passed */
  int        insched;    /* is this thread in scheduler queue */
  int        killed;     /* thread is marked for death */
  char      *data;       /* thread private data */
  int        datasize;   /* size of thread-private data, in bytes */
  int        suspendable;
  int        Event;
  CthThread  qnext;      /* for cthsetnext and cthgetnext */
  qt_t      *savedstack; /* pointer to saved stack */
  int        savedsize;  /* length of saved stack (zero when running) */
  int        stacklen;   /* length of the allocated savedstack >= savedsize */
  qt_t      *savedptr;   /* stack pointer */
} CthThreadStruct;

char *CthGetData(CthThread t) { return t->data; }

CthThread CthPup(pup_er p, CthThread t)
{
#ifndef CMK_OPTIMIZE
    if (pup_isPacking(p))
    {
      if (t->savedsize == 0)
        CmiAbort("Trying to pack a running thread!!\n");
      if (t->insched)
        CmiAbort("Trying to pack a thread in scheduler queue!!\n");
    }
#endif
    if (pup_isUnpacking(p))
      { t = (CthThread) malloc(sizeof(CthThreadStruct));_MEMCHECK(t);}
    pup_bytes(p, (void*) t, sizeof(CthThreadStruct)); 
    if (pup_isUnpacking(p)) {
      t->data = (char *) malloc(t->datasize);_MEMCHECK(t->data);
      t->savedstack = (qt_t*) malloc(t->savedsize);_MEMCHECK(t->savedstack);
    }
    pup_bytes(p, (void*) t->data, t->datasize);
    pup_bytes(p, (void*) t->savedstack, t->savedsize);
    
    if (pup_isDeleting(p))
      {CthFree(t);t=0;}

    return t;
}

struct CthProcInfo
{
  CthThread  current;
  int        datasize;
  qt_t      *stackbase;
  qt_t      *switchbuf_sp;
  qt_t      *switchbuf;
};

CthCpvDeclare(char *, CthData);
CthCpvDeclare(CthProcInfo, CthProc);

static void CthThreadInit(CthThread t, CthVoidFn fn, void *arg)
{
  t->awakenfn = 0;
  t->choosefn = 0;
  t->startfn = fn;
  t->startarg = arg;
  t->insched = 0;
  t->killed = 0;
  t->data = 0;
  t->datasize = 0;
  t->qnext = 0;
  t->savedstack = 0;
  t->savedsize = 0;
  t->stacklen = 0;
  t->savedptr = 0;
  t->suspendable = 1;
  CthSetStrategyDefault(t);
}

void CthFixData(CthThread t)
{
  CthProcInfo proc = CthCpvAccess(CthProc);
  int datasize = proc->datasize;
  if (t->data == 0) {
    t->datasize = datasize;
    t->data = (char *)malloc(datasize);
    _MEMCHECK(t->data);
    return;
  }
  if (t->datasize != datasize) {
    t->datasize = datasize;
    t->data = (char *)realloc(t->data, datasize);
    return;
  }
}

static void CthFreeNow(CthThread t)
{
  if (t->data) free(t->data);
  if (t->savedstack) free(t->savedstack);
  free(t);
}

void CthFree(t)
CthThread t;
{
  CthProcInfo proc = CthCpvAccess(CthProc);
  if ((t->insched == 0)&&(t != proc->current)) {
    CthFreeNow(t);
    return;
  }
  t->killed = 1;
}

void CthDummy() { }

void CthInit(char **argv)
{
  CthThread t; CthProcInfo p; qt_t *switchbuf, *sp;

  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(char *, CthData);
  CthCpvInitialize(CthProcInfo, CthProc);

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  p = (CthProcInfo)malloc(sizeof(struct CthProcInfo));
  _MEMCHECK(p);
  CthThreadInit(t,0,0);
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthProc)=p;
  /* leave some space for current stack frame < 256 bytes */
  sp = (qt_t*)(((size_t)&t) & ~((size_t)0xFF));
  p->stackbase = QT_SP(sp, 0x100);
  p->current = t;
  p->datasize = 0;
  switchbuf = (qt_t*)malloc(QT_STKALIGN + SWITCHBUF_SIZE);
  _MEMCHECK(switchbuf);
  switchbuf = (qt_t*)((((size_t)switchbuf)+QT_STKALIGN) & ~(QT_STKALIGN-1));
  p->switchbuf = switchbuf;
  sp = QT_SP(switchbuf, SWITCHBUF_SIZE);
  sp = QT_ARGS(sp,0,0,0,(qt_only_t*)CthDummy);
  p->switchbuf_sp = sp;
  CthSetStrategyDefault(t);
}

CthThread CthSelf()
{
  CthThread result = CthCpvAccess(CthProc)->current;
  if (result==0) CmiAbort("BARF!\n");
  return result;
}

static void CthOnly(CthThread t, void *dum1, void *dum2)
{
  t->startfn(t->startarg);
  t->killed = 1;
  CthSuspend();
}

static void CthResume1(qt_t *sp, CthProcInfo proc, CthThread t)
{
  int bytes; qt_t *lo, *hi;
  CthThread old = proc->current;
  if (old->killed) {
    if (old->insched==0) CthFreeNow(old);
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
  CthFixData(t);
  CthCpvAccess(CthData) = t->data;
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
  proc->current = t;
  t->insched = 0;
  QT_ABORT((qt_helper_t*)CthDummy,0,0,sp);
}

void CthResume(t)
CthThread t;
{
  CthProcInfo proc = CthCpvAccess(CthProc);
  CpvAccess(_numSwitches)++;
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

void CthSuspend()
{
  CthThread current, next;
  current = CthCpvAccess(CthProc)->current;
  if(!(current->suspendable))
    CmiAbort("trying to suspend main thread!!\n");
  if (current->choosefn == 0) CthNoStrategy();
  /* Pick a thread, discarding dead ones */
  while (1) {
    next = current->choosefn();
    if (next->killed == 0) break;
    CmiAbort("picked dead thread.\n");
    if (next==current)
      CmiAbort("Current thread dead, cannot pick new thread.\n");
    CthFreeNow(next);
  }
  CthResume(next);
}

void CthAwaken(th)
CthThread th;
{
  if (th->awakenfn == 0) CthNoStrategy();
  if (th->insched) CmiAbort("CthAwaken: thread already awake.\n");
  th->awakenfn(th, CQS_QUEUEING_FIFO, 0, 0);
  th->insched = 1;
}

void CthAwakenPrio(CthThread th, int s, int pb, int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  if (th->insched) CmiAbort("CthAwaken: thread already awake.\n");
  th->awakenfn(th, s, pb, prio);
  th->insched = 1;
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthProc)->current);
  CthSuspend();
}

void CthYieldPrio(int s, int pb, int *prio)
{
  CthAwakenPrio(CthCpvAccess(CthProc)->current, s, pb, prio);
  CthSuspend();
}

int CthRegister(size)
int size;
{
  CthProcInfo proc = CthCpvAccess(CthProc);
  int result;
  proc->datasize = (proc->datasize + 7) & (~7);
  result = proc->datasize;
  proc->datasize += size;
  CthFixData(proc->current);
  CthCpvAccess(CthData) = proc->current->data;
  return result;
}

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


typedef void *(qt_userf_t)(void *pu);

struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes];
  CthAwkFn  awakenfn;
  CthThFn    choosefn;
  int        autoyield_enable;
  int        autoyield_blocks;
  char      *data;
  int        datasize;
  int        suspendable;
  int        killed;
  int        Event;
  CthThread  qnext;
  LPVOID     fiber;
};

char *CthGetData(CthThread t) { return t->data; }

CthCpvDeclare(char *,    CthData);
CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(CthThread,  CthPrevious);
CthCpvStatic(int,        CthExiting);
CthCpvStatic(int,        CthDatasize);

typedef CthThread threadTable[100];
CthCpvStatic(threadTable, exitThreads);
CthCpvStatic(int, 	nExit);

static void CthThreadInit(CthThread t)
{
  t->awakenfn = 0;
  t->choosefn = 0;
  t->data=0;
  t->datasize=0;
  t->killed = 0;
  t->qnext=0;
  t->autoyield_enable = 0;
  t->autoyield_blocks = 0;
  t->suspendable = 1;
}

void CthFixData(CthThread t)
{
  int datasize = CthCpvAccess(CthDatasize);
  
  if (t->data == 0) 
  {
    t->datasize = datasize;
    t->data = (char *)malloc(datasize);
    _MEMCHECK(t->data);
  }
  
  else if (t->datasize != datasize) 
  {
    t->datasize = datasize;
    t->data = (char *)realloc(t->data, datasize);
    _MEMCHECK(t->data);
  }
}

void CthInit(char **argv)
{
  CthThread t;
  LPVOID    fiber;

  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(char *,     CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(CthThread,  CthPrevious);
  CthCpvInitialize(int,        CthDatasize);
  CthCpvInitialize(int,        CthExiting);
  CthCpvInitialize(int,        nExit);
  CthCpvInitialize(threadTable,        exitThreads);

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  
  CthThreadInit(t);
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthPrevious)=0;
  CthCpvAccess(CthCurrent)=t;
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthExiting)=0;
  CthCpvAccess(nExit)=0;
  CthSetStrategyDefault(t);
  fiber = ConvertThreadToFiber(t);
  _MEMCHECK(fiber);
  t->fiber = fiber;
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void *CthAbortHelp(CthThread old)
{
  if (old->data) free(old->data);
  if (old->fiber) DeleteFiber((PVOID)old->fiber);
  free(old);
  return (void *) 0;
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
      CthAbortHelp(t);
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
    CthCpvAccess(CthExiting) = 1;
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
    CthAbortHelp(tp);
}
#endif

void CthResume(CthThread t)
{
  CthThread tc;

  tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CpvAccess(_numSwitches)++;
  CthFixData(t);
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = t->data;
  CthCpvAccess(CthPrevious)=tc;
#if 0
  if (CthCpvAccess(CthExiting)) 
  {
    CthCpvAccess(CthExiting)=0;
    tc->killed = 1;
    SwitchToFiber(t->fiber);
  } 
  else 
    CthFiberBlock(t);
#endif
  SwitchToFiber(t->fiber);
}

#if 0
void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  fn(arg);
  CthCpvAccess(CthExiting) = 1;
CthCpvAccess(exitThreads)[CthCpvAccess(nExit)++] = CthSelf();
  CthSuspend();
}
#endif

VOID CALLBACK FiberSetUp(PVOID fiberData)
{
  void **ptr = (void **) fiberData;
  qt_userf_t* fn = (qt_userf_t *)ptr[0];
  void *arg = ptr[1];
  CthThread  t = CthSelf();

/*was:
  CthOnly((void *)ptr[1], 0, ptr[0]);
*/

  CthClearThreads();

  fn(arg);

  CthCpvAccess(CthExiting) = 1;
  CthCpvAccess(exitThreads)[CthCpvAccess(nExit)++] = t;
  CthSuspend();
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
  
  CthSetStrategyDefault(result);
  return result;
}

void CthSuspend()
{
  CthThread next;

  if(!(CthCpvAccess(CthCurrent)->suspendable))
    CmiAbort("trying to suspend main thread!!\n");
  if (CthCpvAccess(CthCurrent)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(CthCurrent)->choosefn();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceSuspend();
#endif
  CthResume(next);
}

void CthAwaken(CthThread th)
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, CQS_QUEUEING_FIFO, 0, 0);
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthCurrent));
  CthSuspend();
}

void CthAwakenPrio(CthThread th, int s, int pb, unsigned int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, unsigned int *prio)
{
  CthAwakenPrio(CthCpvAccess(CthCurrent), s, pb, prio);
  CthSuspend();
}

int CthRegister(int size)
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  
  CthCpvAccess(CthDatasize) = 
    (CthCpvAccess(CthDatasize)+align-1) & ~(align-1);
  result = CthCpvAccess(CthDatasize);
  CthCpvAccess(CthDatasize) += size;
  CthFixData(CthCpvAccess(CthCurrent));
  CthCpvAccess(CthData) = CthCpvAccess(CthCurrent)->data;
  return result;
}


void CthAutoYield(CthThread t, int flag)
{
  t->autoyield_enable = flag;
}

int CthAutoYielding(CthThread t)
{
  return t->autoyield_enable;
}

void CthAutoYieldBlock()
{
  CthCpvAccess(CthCurrent)->autoyield_blocks ++;
}

void CthAutoYieldUnblock()
{
  CthCpvAccess(CthCurrent)->autoyield_blocks --;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}
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
  {
    if(ss->buf[i].nslots >= nslots)
    {
      int slot = ss->buf[i].startslot;
      ss->buf[i].startslot += nslots;
      ss->buf[i].nslots -= nslots;
      ss->emptyslots -= nslots;
      return slot;
    }
  }
  return (-1);
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
  int pos, emptypos = (-1);
  /* eslot is the ending slot of the block to be freed */
  int eslot = sslot + nslots;
  ss->emptyslots += nslots;
  for (pos=0; pos < (ss->maxbuf); pos++)
  {
    if (ss->buf[pos].nslots == 0 && emptypos==(-1))
    {
      /* find the first empty slot just in case it is required */
      emptypos = pos;
    }
    else
    {
      /* e is the ending slot of pos'th slotblock */
      int e = ss->buf[pos].startslot + ss->buf[pos].nslots;
      if (e == sslot)
      {
	ss->buf[pos].nslots += nslots;
	return;
      }
      if(eslot == ss->buf[pos].startslot)
      {
	ss->buf[pos].startslot = sslot;
	ss->buf[pos].nslots += nslots;
	return;
      }
    }
  }
  /* if we are here, it means we could not find a slotblock that the */
  /* block to be freed was combined with. */
  /* now check if we could not find an empty slotblock */
  if (emptypos == (-1))
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
  return;
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
  char cmicore[CmiMsgHeaderSizeBytes];
  CthAwkFn   awakenfn;
  CthThFn    choosefn;
  int        autoyield_enable;
  int        autoyield_blocks;
  char      *data;
  int        datasize;
  int        suspendable;
  int        Event;
  CthThread  qnext;
  int        slotnum;
  int        nslots;
  int        awakened;
  qt_t      *stack;
  qt_t      *stackp;
};

char *CthGetData(CthThread t) { return t->data; }

#define STACKSIZE (32768)
CpvStaticDeclare(int, _stksize);

CpvStaticDeclare(int, reqHdlr);
CpvStaticDeclare(int, respHdlr);
CpvStaticDeclare(slotset *, myss);
CpvStaticDeclare(void *, heapbdry);
CpvStaticDeclare(int, zerofd);


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
  /* msg will be automatically freed by the scheduler */
}

CthCpvDeclare(char *,    CthData);
CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(int,        CthExiting);
CthCpvStatic(int,        CthDatasize);

static void CthThreadInit(t)
CthThread t;
{
  t->awakenfn = 0;
  t->choosefn = 0;
  t->data=0;
  t->datasize=0;
  t->qnext=0;
  t->autoyield_enable = 0;
  t->autoyield_blocks = 0;
  t->suspendable = 1;
  t->slotnum = (-1);
  t->nslots = 0;
  t->awakened = 0;
}

void CthFixData(t)
CthThread t;
{
  int datasize = CthCpvAccess(CthDatasize);
  if (t->data == 0) {
    t->datasize = datasize;
    t->data = (char *)malloc(datasize);
    _MEMCHECK(t->data);
    return;
  }
  if (t->datasize != datasize) {
    t->datasize = datasize;
    t->data = (char *)realloc(t->data, datasize);
    _MEMCHECK(t->data);
    return;
  }
}

/*
 * maps the virtual memory associated with slot using mmap
 */
static void *
map_slots(int slot, int nslots)
{
  void *pa;
  char *addr;
  size_t sz = CpvAccess(_stksize);
  addr = (char*) CpvAccess(heapbdry) + slot*sz;
  pa = mmap((void*) addr, sz*nslots, 
            PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED,
            CpvAccess(zerofd), 0);
  if(pa== (void*)(-1))
    CmiAbort("mmap call failed to allocate requested memory.\n");
  return pa;
}

/*
 * unmaps the virtual memory associated with slot using munmap
 */
static void
unmap_slots(int slot, int nslots)
{
  size_t sz = CpvAccess(_stksize);
  char *addr = (char*) CpvAccess(heapbdry) + slot*sz;
  int retval = munmap((void*) addr, sz*nslots);
  if (retval==(-1))
    CmiAbort("munmap call failed to deallocate requested memory.\n");
}
static void *__cur_stack_frame(void)
{
  char __dummy;
  return (void*) &__dummy;
}

void CthInit(char **argv)
{
  CthThread t;
  int i, numslots;

  CpvInitialize(int, _stksize);
  CpvAccess(_stksize) = 0;
  CmiGetArgInt(argv,"+stacksize",&CpvAccess(_stksize));
  CpvAccess(_stksize) = CpvAccess(_stksize) ? CpvAccess(_stksize) : STACKSIZE;
  CpvAccess(_stksize) = (CpvAccess(_stksize)+(CMK_MEMORY_PAGESIZE*2)-1) & 
                       ~(CMK_MEMORY_PAGESIZE-1);
#if CMK_THREADS_DEBUG
  CmiPrintf("[%d] Using stacksize of %d\n", CmiMyPe(), CpvAccess(_stksize));
#endif

  CpvInitialize(void *, heapbdry);
  CpvInitialize(int, zerofd);
  /*
   * calculate the number of slots according to stacksize
   * divide up into number of available processors
   * and allocate the slotset.
   */
  do {
    void *heap = (void*) malloc(1);
    void *stack = __cur_stack_frame();
    void *stackbdry;
    int stacksize = CpvAccess(_stksize);
    _MEMCHECK(heap);
#if CMK_THREADS_DEBUG
    CmiPrintf("[%d] heap=%p\tstack=%p\n",CmiMyPe(),heap,stack);
#endif
    /* Align heap to a 1G boundary to leave space to grow */
    /* Align stack to a 256M boundary  to leave space to grow */
#ifdef QT_GROW_UP
    CpvAccess(heapbdry) = (void *)(((size_t)heap-(2<<30))&(~((1<<30)-1)));
    stackbdry = (void *)(((size_t)stack+(1<<28))&(~((1<<28)-1)));
    numslots = (((size_t)CpvAccess(heapbdry)-(size_t)stackbdry)/stacksize)
                 / CmiNumPes();
#else
    CpvAccess(heapbdry) = (void *)(((size_t)heap+(2<<30))&(~((1<<30)-1)));
    stackbdry = (void *)(((size_t)stack-(1<<28))&(~((1<<28)-1)));
    numslots = (((size_t)stackbdry-(size_t)CpvAccess(heapbdry))/stacksize)
                 / CmiNumPes();
#endif
#if CMK_THREADS_DEBUG
    CmiPrintf("[%d] heapbdry=%p\tstackbdry=%p\n",
              CmiMyPe(),CpvAccess(heapbdry),stackbdry);
    CmiPrintf("[%d] numthreads per pe=%d\n",CmiMyPe(),numslots);
#endif
    CpvAccess(zerofd) = open("/dev/zero", O_RDWR);
    if(CpvAccess(zerofd)<0)
      CmiAbort("Cannot open /dev/zero. Aborting.\n");
    free(heap);
  } while(0);

  CpvInitialize(slotset *, myss);
  CpvAccess(myss) = new_slotset(CmiMyPe()*numslots, numslots);

  CpvInitialize(int, reqHdlr);
  CpvInitialize(int, respHdlr);

  CpvAccess(reqHdlr) = CmiRegisterHandler((CmiHandler)reqslots);
  CpvAccess(respHdlr) = CmiRegisterHandler((CmiHandler)respslots);

  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(char *,     CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(int,        CthDatasize);
  CthCpvInitialize(int,        CthExiting);

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t);
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthCurrent)=t;
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthExiting)=0;
  CthSetStrategyDefault(t);
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(CthCurrent)) {
    CthCpvAccess(CthExiting) = 1;
  } else {
    if (t->data) free(t->data);
    if(t->slotnum >= 0)
    {
      free_slots(CpvAccess(myss), t->slotnum, t->nslots);
      unmap_slots(t->slotnum, t->nslots);
    }
    free(t);
  }
}

static void *
CthAbortHelp(qt_t *sp, CthThread old, void *null)
{
  if (old->data) free(old->data);
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
  CthThread tc;
  tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CpvAccess(_numSwitches)++;
  CthFixData(t);
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = t->data;
  if (CthCpvAccess(CthExiting)) {
    CthCpvAccess(CthExiting)=0;
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
  CthCpvAccess(CthExiting) = 1;
  CthSuspend();
}

static void
CthStackCreate(CthThread t, CthVoidFn fn, void *arg, int slotnum, int nslots)
{
  qt_t *stack, *stackbase, *stackp;
  int size = CpvAccess(_stksize);
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
  if(size>CpvAccess(_stksize))
    nslots = (int) (size/CpvAccess(_stksize)) + 1;
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  CthSetStrategyDefault(result);
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
    CthStackCreate(result, fn, arg, slotnum, nslots);
  }
  return result;
}

void CthSuspend()
{
  CthThread next;
#if CMK_WEB_MODE
  void usageStop();
#endif
  if(!(CthCpvAccess(CthCurrent)->suspendable))
    CmiAbort("trying to suspend main thread!!\n");
  if (CthCpvAccess(CthCurrent)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(CthCurrent)->choosefn();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceSuspend();
#endif
#if CMK_WEB_MODE
  usageStop();
#endif
  CthResume(next);
}

void CthAwaken(th)
CthThread th;
{
  if (th->awakenfn == 0) CthNoStrategy();
  th->awakened = 1;
  /* thread stack is not yet allocated. */
  if (th->slotnum == (-2))
    return;
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, CQS_QUEUEING_FIFO, 0, 0);
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthCurrent));
  CthSuspend();
}

void CthAwakenPrio(CthThread th, int s, int pb, unsigned int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  th->awakened = 1;
  if (th->slotnum == (-2))
    return;
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, unsigned int *prio)
{
  CthAwakenPrio(CthCpvAccess(CthCurrent), s, pb, prio);
  CthSuspend();
}

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  CthCpvAccess(CthDatasize) = (CthCpvAccess(CthDatasize)+align-1) & ~(align-1);
  result = CthCpvAccess(CthDatasize);
  CthCpvAccess(CthDatasize) += size;
  CthFixData(CthCpvAccess(CthCurrent));
  CthCpvAccess(CthData) = CthCpvAccess(CthCurrent)->data;
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

  if (pup_isUnpacking(p)) {
    t->data = (char *) malloc(t->datasize);
    _MEMCHECK(t->data);
    stack = (qt_t*) map_slots(t->slotnum, t->nslots);
    _MEMCHECK(stack);
    if(stack != t->stack)
      CmiAbort("Stack pointers do not match after migration!!\n");  
  }
  pup_bytes(p, (void*)t->data, t->datasize);

  stackbase = QT_SP(t->stack, CpvAccess(_stksize));
#ifdef QT_GROW_UP
  ssz = ((char*)(t->stackp)-(char*)(stackbase));
  pup_bytes(p, (void*)stackbase, ssz);
#else
  ssz = ((char*)(stackbase)-(char*)(t->stackp));
  pup_bytes(p, (void*)t->stackp, ssz);
#endif

  if(pup_isDeleting(p))
  {
    CthFree(t);
    t = 0;
  }
  return t;
}

#elif CMK_THREADS_USE_PTHREADS

#define CthMemAlign(x,n) malloc(n)
#define memalign(m, a) valloc(a)

struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes];
  CthAwkFn  awakenfn;
  CthThFn    choosefn;
  pthread_t  self;
  pthread_cond_t cond;
  pthread_cond_t *creator;
  CthVoidFn  fn;
  void      *arg;
  char      *data;
  int        datasize;
  int        suspendable;
  int        Event;
  CthThread  qnext;
  char       inited;
};

char *CthGetData(CthThread t) { return t->data; }

CthCpvDeclare(char *,    CthData);
CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(int,        CthExiting);
CthCpvStatic(int,        CthDatasize);
CthCpvStatic(pthread_mutex_t, sched_mutex);

static void CthThreadInit(t)
CthThread t;
{
  t->awakenfn = 0;
  t->choosefn = 0;
  t->data=0;
  t->datasize=0;
  t->qnext=0;
  t->suspendable = 1;
  t->inited = 0;
  pthread_cond_init(&(t->cond) , (pthread_condattr_t *) 0);
}

void CthFixData(t)
CthThread t;
{
  int datasize = CthCpvAccess(CthDatasize);
  if (t->data == 0) {
    t->datasize = datasize;
    t->data = (char *)malloc(datasize);
    _MEMCHECK(t->data);
    return;
  }
  if (t->datasize != datasize) {
    t->datasize = datasize;
    t->data = (char *)realloc(t->data, datasize);
    return;
  }
}

void CthInit(char **argv)
{
  CthThread t;

  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(char *,     CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(int,        CthDatasize);
  CthCpvInitialize(int,        CthExiting);
  CthCpvInitialize(pthread_mutex_t, sched_mutex);

  pthread_mutex_init(&CthCpvAccess(sched_mutex), (pthread_mutexattr_t *) 0);
  pthread_mutex_lock(&CthCpvAccess(sched_mutex));
  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t);
  t->self = pthread_self();
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthCurrent)=t;
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthExiting)=0;
  CthSetStrategyDefault(t);
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(CthCurrent)) {
    CthCpvAccess(CthExiting) = 1;
  } else {
    if (t->data) free(t->data);
    free(t);
  }
}

void CthResume(CthThread t)
{
  CthThread tc;
  tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CpvAccess(_numSwitches)++;
  CthFixData(t);
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = t->data;
  pthread_cond_signal(&(t->cond));
  if (CthCpvAccess(CthExiting)) {
    CthCpvAccess(CthExiting)=0;
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
  CthCpvAccess(CthExiting) = 1;
  CthSuspend();
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
  CthSetStrategyDefault(result);
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

void CthSuspend()
{
  CthThread t = CthSelf();
  CthThread next;
#if CMK_WEB_MODE
  void usageStop();
#endif
  if(!(t->suspendable))
    CmiAbort("trying to suspend main thread!!\n");
  if (t->choosefn == 0) CthNoStrategy();
  next = t->choosefn();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceSuspend();
#endif
#if CMK_WEB_MODE
  usageStop();
#endif
  CthResume(next);
}

void CthAwaken(th)
CthThread th;
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, CQS_QUEUEING_FIFO, 0, 0);
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthCurrent));
  CthSuspend();
}

void CthAwakenPrio(CthThread th, int s, int pb, unsigned int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, unsigned int *prio)
{
  CthAwakenPrio(CthCpvAccess(CthCurrent), s, pb, prio);
  CthSuspend();
}

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  CthCpvAccess(CthDatasize) = (CthCpvAccess(CthDatasize)+align-1) & ~(align-1);
  result = CthCpvAccess(CthDatasize);
  CthCpvAccess(CthDatasize) += size;
  CthFixData(CthCpvAccess(CthCurrent));
  CthCpvAccess(CthData) = CthCpvAccess(CthCurrent)->data;
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}

#elif  CMK_THREADS_USE_CONTEXT

#include <signal.h>
#include <ucontext.h>

#define STACKSIZE (32768)
static int _stksize = 0;

struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes];
  CthAwkFn  awakenfn;
  CthThFn    choosefn;
  int        autoyield_enable;
  int        autoyield_blocks;
  char      *data;
  int        datasize;
  int        suspendable;
  int        Event;
  CthThread  qnext;
  char      *stack;
  ucontext_t      stackp;
};

char *CthGetData(CthThread t) { return t->data; }

CthCpvDeclare(char *,    CthData);
CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(int,        CthExiting);
CthCpvStatic(int,        CthDatasize);

static void CthThreadInit(t)
CthThread t;
{
  t->awakenfn = 0;
  t->choosefn = 0;
  t->data=0;
  t->datasize=0;
  t->qnext=0;
  t->autoyield_enable = 0;
  t->autoyield_blocks = 0;
  t->suspendable = 1;
}

void CthFixData(t)
CthThread t;
{
  int datasize = CthCpvAccess(CthDatasize);
  if (t->data == 0) {
    t->datasize = datasize;
    t->data = (char *)malloc(datasize);
    _MEMCHECK(t->data);
    return;
  }
  if (t->datasize != datasize) {
    t->datasize = datasize;
    t->data = (char *)realloc(t->data, datasize);
    return;
  }
}

void CthInit(char **argv)
{
  CthThread t;

  CmiGetArgInt(argv,"+stacksize",&_stksize);
  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(char *,     CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(int,        CthDatasize);
  CthCpvInitialize(int,        CthExiting);

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  CthThreadInit(t);
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthCurrent)=t;
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthExiting)=0;
  CthSetStrategyDefault(t);
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(CthCurrent)) {
    CthCpvAccess(CthExiting) = 1;
  } else {
    if (t->data) free(t->data);
    free(t->stack);
    free(t);
  }
}

static void *CthAbortHelp(ucontext_t *sp, CthThread old, void *null)
{
  if (old->data) free(old->data);
  free(old->stack);
  free(old);
  return (void *) 0;
}

void CthResume(t)
CthThread t;
{
  CthThread tc;
  tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CpvAccess(_numSwitches)++;
  CthFixData(t);
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = t->data;
  if (CthCpvAccess(CthExiting)) {
    CthCpvAccess(CthExiting)=0;
    CthAbortHelp(&tc->stackp, tc, 0);
    setcontext(&t->stackp);
  } else {
    if (0 != swapcontext(&tc->stackp, &t->stackp)) 
      CmiAbort("CthResume: swapcontext failed.\n");
  }
  if (tc!=CthCpvAccess(CthCurrent)) { CmiAbort("Stack corrupted?\n"); }
}

void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  fn(arg);
  CthCpvAccess(CthExiting) = 1;
  CthSuspend();
}

#define STP_STKALIGN(sp, alignment) \
  ((void *)((((qt_word_t)(sp)) + (alignment) - 1) & ~((alignment)-1)))

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread result; char *stack, *stackbase;
  ucontext_t *jb, *stackp;
  size = 16384*2;
  stack = (char*)malloc(size);
  _MEMCHECK(stack);
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  stackbase = stack;
  stackbase = STP_STKALIGN(stackbase, 16);
#if CMK_STACK_GROWDOWN
  stackbase = stackbase+size-2048;
#elif CMK_STACK_GROWUP
  stackbase = stackbase+8192;
#else
  #error "Must define stack grow up or down in conv-mach.h!"
  CmiAbort("Must define stack grow up or down in conv-mach.h!\n");
#endif
  getcontext(&result->stackp);
  result->stackp.uc_stack.ss_sp = stackbase;
  result->stackp.uc_stack.ss_size = size - 8192;
  result->stackp.uc_stack.ss_flags = 0;
  result->stackp.uc_link = 0;
  makecontext(&result->stackp, (void (*) (void))CthOnly, 3, arg, result, fn);
  result->stack = stack;
  CthSetStrategyDefault(result);
  return result;
}

void CthSuspend()
{
  CthThread next;
#if CMK_WEB_MODE
  void usageStop();
#endif
  if(!(CthCpvAccess(CthCurrent)->suspendable))
    CmiAbort("trying to suspend main thread!!\n");
  if (CthCpvAccess(CthCurrent)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(CthCurrent)->choosefn();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceSuspend();
#endif
#if CMK_WEB_MODE
  usageStop();
#endif
  CthResume(next);
}

void CthAwaken(th)
CthThread th;
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, CQS_QUEUEING_FIFO, 0, 0);
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthCurrent));
  CthSuspend();
}

void CthAwakenPrio(CthThread th, int s, int pb, unsigned int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, unsigned int *prio)
{
  CthAwakenPrio(CthCpvAccess(CthCurrent), s, pb, prio);
  CthSuspend();
}

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  CthCpvAccess(CthDatasize) = (CthCpvAccess(CthDatasize)+align-1) & ~(align-1);
  result = CthCpvAccess(CthDatasize);
  CthCpvAccess(CthDatasize) += size;
  CthFixData(CthCpvAccess(CthCurrent));
  CthCpvAccess(CthData) = CthCpvAccess(CthCurrent)->data;
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}

#else /*Default, stack-on-heap threads*/

#define STACKSIZE (32768)
static int _stksize = 0;

#ifdef CMK_OPTIMIZE
#  define CMK_STACKPROTECT 0
#else
#  define CMK_STACKPROTECT 1

#  if CMK_MEMORY_PROTECTABLE

#    include "sys/mman.h"
#    define CthMemAlign(x,n) memalign((x),(n))
#    define CthMemoryProtect(p,l) mprotect(p,l,PROT_NONE)
#    define CthMemoryUnprotect(p,l) mprotect(p,l,PROT_READ | PROT_WRITE)

#  else

#    define CthMemAlign(x,n) malloc(n)
#    define CthMemoryProtect(p,l) 
#    define CthMemoryUnprotect(p,l)
#    define memalign(m, a) valloc(a)

#  endif

#endif

struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes];
  CthAwkFn  awakenfn;
  CthThFn    choosefn;
  int        autoyield_enable;
  int        autoyield_blocks;
  char      *data;
  int        datasize;
  int        suspendable;
  int        Event;
  CthThread  qnext;
#if CMK_STACKPROTECT
  char      *protect;
  int        protlen;
#endif
  qt_t      *stack;
  qt_t      *stackp;
};

char *CthGetData(CthThread t) { return t->data; }

CthCpvDeclare(char *,    CthData);
CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(int,        CthExiting);
CthCpvStatic(int,        CthDatasize);

static void CthThreadInit(t)
CthThread t;
{
  t->awakenfn = 0;
  t->choosefn = 0;
  t->data=0;
  t->datasize=0;
  t->qnext=0;
  t->autoyield_enable = 0;
  t->autoyield_blocks = 0;
  t->suspendable = 1;
}

void CthFixData(t)
CthThread t;
{
  int datasize = CthCpvAccess(CthDatasize);
  if (t->data == 0) {
    t->datasize = datasize;
    t->data = (char *)malloc(datasize);
    _MEMCHECK(t->data);
    return;
  }
  if (t->datasize != datasize) {
    t->datasize = datasize;
    t->data = (char *)realloc(t->data, datasize);
    return;
  }
}

void CthInit(char **argv)
{
  CthThread t;

  CmiGetArgInt(argv,"+stacksize",&_stksize);
  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(char *,     CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(int,        CthDatasize);
  CthCpvInitialize(int,        CthExiting);

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
#if CMK_STACKPROTECT
  t->protect = 0;
  t->protlen = 0;
#endif
  CthThreadInit(t);
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthCurrent)=t;
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthExiting)=0;
  CthSetStrategyDefault(t);
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(CthCurrent)) {
    CthCpvAccess(CthExiting) = 1;
  } else {
#if CMK_STACKPROTECT
    CthMemoryUnprotect(t->protect, t->protlen);
#endif
    if (t->data) free(t->data);
    free(t->stack);
    free(t);
  }
}

static void *CthAbortHelp(qt_t *sp, CthThread old, void *null)
{
#if CMK_STACKPROTECT
  CthMemoryUnprotect(old->protect, old->protlen);
#endif
  if (old->data) free(old->data);
  free(old->stack);
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
  CthThread tc;
  tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CpvAccess(_numSwitches)++;
  CthFixData(t);
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = t->data;
  if (CthCpvAccess(CthExiting)) {
    CthCpvAccess(CthExiting)=0;
    QT_ABORT((qt_helper_t*)CthAbortHelp, tc, 0, t->stackp);
  } else {
    QT_BLOCK((qt_helper_t*)CthBlockHelp, tc, 0, t->stackp);
  }
  if (tc!=CthCpvAccess(CthCurrent)) { CmiAbort("Stack corrupted?\n"); }
}

static void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  fn(arg);
  CthCpvAccess(CthExiting) = 1;
  CthSuspend();
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread result; qt_t *stack, *stackbase, *stackp;
  size = (size) ? size : ((_stksize) ? _stksize : STACKSIZE);
#if CMK_STACKPROTECT
  size = (size+(CMK_MEMORY_PAGESIZE*2)-1) & ~(CMK_MEMORY_PAGESIZE-1);
  stack = (qt_t*)CthMemAlign(CMK_MEMORY_PAGESIZE, size);
#else
  stack = (qt_t*)malloc(size);
#endif
  _MEMCHECK(stack);
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  stackbase = QT_SP(stack, size);
  stackp = QT_ARGS(stackbase, arg, result, (qt_userf_t *)fn, CthOnly);
  result->stack = stack;
  result->stackp = stackp;
#if CMK_STACKPROTECT
  if (stack==stackbase) {
    result->protect = ((char*)stack) + size - CMK_MEMORY_PAGESIZE;
    result->protlen = CMK_MEMORY_PAGESIZE;
  } else {
    result->protect = ((char*)stack);
    result->protlen = CMK_MEMORY_PAGESIZE;
  }
  CthMemoryProtect(result->protect, result->protlen);
#endif
  CthSetStrategyDefault(result);
  return result;
}

void CthSuspend()
{
  CthThread next;
#if CMK_WEB_MODE
  void usageStop();
#endif
  if(!(CthCpvAccess(CthCurrent)->suspendable))
    CmiAbort("trying to suspend main thread!!\n");
  if (CthCpvAccess(CthCurrent)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(CthCurrent)->choosefn();
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceSuspend();
#endif
#if CMK_WEB_MODE
  usageStop();
#endif
  CthResume(next);
}

void CthAwaken(th)
CthThread th;
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, CQS_QUEUEING_FIFO, 0, 0);
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthCurrent));
  CthSuspend();
}

void CthAwakenPrio(CthThread th, int s, int pb, unsigned int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, unsigned int *prio)
{
  CthAwakenPrio(CthCpvAccess(CthCurrent), s, pb, prio);
  CthSuspend();
}

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  CthCpvAccess(CthDatasize) = (CthCpvAccess(CthDatasize)+align-1) & ~(align-1);
  result = CthCpvAccess(CthDatasize);
  CthCpvAccess(CthDatasize) += size;
  CthFixData(CthCpvAccess(CthCurrent));
  CthCpvAccess(CthData) = CthCpvAccess(CthCurrent)->data;
  return result;
}

CthThread CthPup(pup_er p, CthThread t)
{
  CmiAbort("CthPup not implemented.\n");
  return 0;
}
#endif

/* Common Functions */

void setEvent(CthThread t, int event) { t->Event = event; }
int getEvent(CthThread t) { return t->Event; }

void CthSetSuspendable(CthThread t, int val) { t->suspendable = val; }
int CthIsSuspendable(CthThread t) { return t->suspendable; }

void CthSetNext(CthThread t, CthThread v) { t->qnext = v; }
CthThread CthGetNext(CthThread t) { return t->qnext; }

static void CthNoStrategy(void)
{
  CmiAbort("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
}

int CthImplemented() { return 1; } 

void CthSetStrategy(CthThread t, CthAwkFn awkfn, CthThFn chsfn)
{
  t->awakenfn = awkfn;
  t->choosefn = chsfn;
}
