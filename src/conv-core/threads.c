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
 
#ifdef WIN32
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#endif

#include "converse.h"
#ifndef  WIN32
#include "qt.h"
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

int CthPackBufSize(CthThread t)
{
#ifndef CMK_OPTIMIZE
  if (t->savedsize == 0)
    CmiAbort("Trying to pack a running thread!!\n");
#endif
  return sizeof(CthThreadStruct) + t->datasize + t->savedsize;
}

void CthPackThread(CthThread t, void *buffer)
{
#ifndef CMK_OPTIMIZE
  if (t->savedsize == 0)
    CmiAbort("Trying to pack a running thread!!\n");
  if (t->insched)
    CmiAbort("Trying to pack a thread in scheduler queue!!\n");
#endif
  memcpy(buffer, (void *)t, sizeof(CthThreadStruct));
  memcpy(((char*)buffer)+sizeof(CthThreadStruct), 
         (void *)t->data, t->datasize);
  free((void *)t->data);
  memcpy(((char*)buffer)+sizeof(CthThreadStruct)+t->datasize, 
         (void *)t->savedstack, t->savedsize);
  free((void *)t->savedstack);
  free(t);
}

CthThread CthUnpackThread(void *buffer)
{
  CthThread t = (CthThread) malloc(sizeof(CthThreadStruct));
  _MEMCHECK(t);
  memcpy((void*) t, buffer, sizeof(CthThreadStruct));
  t->data = (char *) malloc(t->datasize);
  _MEMCHECK(t->data);
  memcpy((void*)t->data, ((char*)buffer)+sizeof(CthThreadStruct),
         t->datasize);
  t->savedstack = (qt_t*) malloc(t->savedsize);
  _MEMCHECK(t->savedstack);
  t->stacklen = t->savedsize;
  memcpy((void*)t->savedstack, 
         ((char*)buffer)+sizeof(CthThreadStruct)+t->datasize, t->savedsize);
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

#define _WIN32_WINNT  0x0400

#include <windows.h>
#include <winbase.h>

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

CthCpvDeclare(char *,    CthData);
CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(CthThread,  CthPrevious);
CthCpvStatic(int,        CthExiting);
CthCpvStatic(int,        CthDatasize);

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

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  
  CthThreadInit(t);
  CthCpvAccess(CthData)=0;
  CthCpvAccess(CthPrevious)=0;
  CthCpvAccess(CthCurrent)=t;
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthExiting)=0;
  CthSetStrategyDefault(t);
  fiber = ConvertThreadToFiber(t);
  t->fiber = fiber;
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

void CthFree(CthThread t)
{
  if (t==CthCpvAccess(CthCurrent)) 
  {
    CthCpvAccess(CthExiting) = 1;
  } 
  else 
  {
    CmiError("Not implemented CthFree.\n");
    exit(1);
  }
}

static void *CthAbortHelp(CthThread old)
{
  if (old->data) free(old->data);
  DeleteFiber(old->fiber);
  free(old);
  return (void *) 0;
}


static void CthFiberBlock(CthThread t)
{
  CthThread tp;
  
  SwitchToFiber(t->fiber);
  tp = CthCpvAccess(CthPrevious);
  if (tp != 0 && tp->killed == 1)
    CthAbortHelp(tp);
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
  CthCpvAccess(CthPrevious)=tc;
  if (CthCpvAccess(CthExiting)) 
  {
    CthCpvAccess(CthExiting)=0;
    tc->killed = 1;
    SwitchToFiber(t->fiber);
  } 
  else 
    CthFiberBlock(t);
  
}

static void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  fn(arg);
  CthCpvAccess(CthExiting) = 1;
  CthSuspend();
}

VOID CALLBACK FiberSetUp(PVOID fiberData)
{
  void **ptr = (void **) fiberData;
  CthOnly((void *)ptr[1], 0, ptr[0]);
}

CthThread CthCreate(CthVoidFn fn, void *arg, int size)
{
  CthThread result; 
  void**    fiberData;
  fiberData = (void *) malloc(2*sizeof(void *));
  fiberData[0] = (void *)fn;
  fiberData[1] = arg;
  
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  result->fiber = CreateFiber(0, FiberSetUp, (PVOID) fiberData);
  
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

void CthAwakenPrio(CthThread th, int s, int pb, int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, int *prio)
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

int CthPackBufSize(CthThread t)
{
  CmiAbort("CthPackBufSize not implemented.\n");
  return 0;
}

void CthPackThread(CthThread t, void *buffer)
{
  CmiAbort("CthPackThread not implemented.\n");
}

CthThread CthUnpackThread(void *buffer)
{
  CmiAbort("CthUnpackThread not implemented.\n");
  return (CthThread) 0;
}

#else

#define STACKSIZE (32768)
static int _stksize = 0;

#if CMK_MEMORY_PROTECTABLE

#include "sys/mman.h"
#define CthMemAlign(x,n) memalign((x),(n))
#define CthMemoryProtect(p,l) mprotect(p,l,PROT_NONE)
#define CthMemoryUnprotect(p,l) mprotect(p,l,PROT_READ | PROT_WRITE)

#else

#define CthMemAlign(x,n) malloc(n)
#define CthMemoryProtect(p,l) 
#define CthMemoryUnprotect(p,l)
#define memalign(m, a) valloc(a)

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
  char      *protect;
  int        protlen;
  qt_t      *stack;
  qt_t      *stackp;
};

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
  int i;

  for(i=0;argv[i];i++) {
    if(strncmp("+stacksize",argv[i],10)==0) {
      if (strlen(argv[i]) > 10) {
        sscanf(argv[i], "+stacksize%d", &_stksize);
      } else {
        if (argv[i+1]) {
          sscanf(argv[i+1], "%d", &_stksize);
        }
      }
    }
  }
  CpvInitialize(int, _numSwitches);
  CpvAccess(_numSwitches) = 0;

  CthCpvInitialize(char *,     CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(int,        CthDatasize);
  CthCpvInitialize(int,        CthExiting);

  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(t);
  t->protect = 0;
  t->protlen = 0;
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
    CmiError("Not implemented CthFree.\n");
    exit(1);
  }
}

static void *CthAbortHelp(qt_t *sp, CthThread old, void *null)
{
  CthMemoryUnprotect(old->protect, old->protlen);
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
  if (tc!=CthCpvAccess(CthCurrent)) { CmiError("Stack corrupted?\n"); exit(1); }
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
  size = (size+(CMK_MEMORY_PAGESIZE*2)-1) & ~(CMK_MEMORY_PAGESIZE-1);
  stack = (qt_t*)CthMemAlign(CMK_MEMORY_PAGESIZE, size);
  _MEMCHECK(stack);
  result = (CthThread)malloc(sizeof(struct CthThreadStruct));
  _MEMCHECK(result);
  CthThreadInit(result);
  stackbase = QT_SP(stack, size);
  stackp = QT_ARGS(stackbase, arg, result, (qt_userf_t *)fn, CthOnly);
  result->stack = stack;
  result->stackp = stackp;
  if (stack==stackbase) {
    result->protect = ((char*)stack) + size - CMK_MEMORY_PAGESIZE;
    result->protlen = CMK_MEMORY_PAGESIZE;
  } else {
    result->protect = ((char*)stack);
    result->protlen = CMK_MEMORY_PAGESIZE;
  }
  CthMemoryProtect(result->protect, result->protlen);
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

void CthAwakenPrio(CthThread th, int s, int pb, int *prio)
{
  if (th->awakenfn == 0) CthNoStrategy();
  CpvAccess(curThread) = th;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceAwaken();
#endif
  th->awakenfn(th, s, pb, prio);
}

void CthYieldPrio(int s, int pb, int *prio)
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

int CthPackBufSize(CthThread t)
{
  CmiAbort("CthPackBufSize not implemented.\n");
  return 0;
}

void CthPackThread(CthThread t, void *buffer)
{
  CmiAbort("CthPackThread not implemented.\n");
}

CthThread CthUnpackThread(void *buffer)
{
  CmiAbort("CthUnpackThread not implemented.\n");
  return (CthThread) 0;
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
