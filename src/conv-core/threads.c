
/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * Revision 1.33  1997/05/05 13:47:14  jyelon
 * Revamped threads package using quickthreads.
 *
 ***************************************************************************
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
 * void CthSetStrategy(CthThread t, CthVoidFn awakenfn, CthThFn choosefn)
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
 
#include "converse.h"
#include "qt.h"

#define STACKSIZE (32768)

struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes];
  CthVoidFn  awakenfn;
  CthThFn    choosefn;
  int        autoyield_enable;
  int        autoyield_blocks;
  char      *data;
  int        datasize;
/** addition for tracing */
  int        Event;
/** End Addition */
  CthThread  qnext;
  qt_t      *stackp;
};

/** addition for tracing */
void setEvent(CthThread t, int event)
{
  t->Event = event;
}

int getEvent(CthThread t)
{
  return t->Event;
}
/** End Addition */

CthCpvDeclare(char *,    CthData);
CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(int,        CthExiting);
CthCpvStatic(int,        CthDatasize);

/** addition for tracing */
CpvDeclare(CthThread, cThread);
/** End Addition */

int CthImplemented()
{ return 1; }

static void CthNoStrategy()
{
  CmiPrintf("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
  exit(1);
}

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
}

void CthFixData(t)
CthThread t;
{
  int datasize = CthCpvAccess(CthDatasize);
  if (t->data == 0) {
    t->datasize = datasize;
    t->data = (char *)malloc(datasize);
    return;
  }
  if (t->datasize != datasize) {
    t->datasize = datasize;
    t->data = (char *)realloc(t->data, datasize);
    return;
  }
}

void CthInit()
{
  CthThread t;

  CthCpvInitialize(char *,     CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(int,        CthDatasize);
  CthCpvInitialize(int,        CthExiting);
  
  t = (CthThread)malloc(sizeof(struct CthThreadStruct));
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
  if (old->data) free(old->data);
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
  CthFixData(t);
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = t->data;
  if (CthCpvAccess(CthExiting)) {
    CthCpvAccess(CthExiting)=0;
    QT_ABORT((qt_helper_t*)CthAbortHelp, tc, 0, t->stackp);
  } else {
    QT_BLOCK((qt_helper_t*)CthBlockHelp, tc, 0, t->stackp);
  }
  if (tc!=CthCpvAccess(CthCurrent)) { CmiError("Fugged up.\n"); exit(1); }
}

static void CthOnly(void *arg, void *vt, qt_userf_t fn)
{
  CthThread t = (CthThread)vt; CthThread next;
  fn(arg);
  CthCpvAccess(CthExiting) = 1;
  CthSuspend();
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread result; char *stack;
  if (size==0) size = STACKSIZE;
  size += QT_STKALIGN;
  result = (CthThread)malloc(sizeof(struct CthThreadStruct)+size);
  if (result==0) { CmiError("Out of memory."); exit(1); }
  stack = ((char*)result)+sizeof(struct CthThreadStruct);
  stack = (char *)QT_STKROUNDUP(((CMK_SIZE_T)stack));
  CthThreadInit(result);
  result->stackp = QT_SP(stack, size - QT_STKALIGN);
  result->stackp = 
    QT_ARGS(result->stackp, arg, result, (qt_userf_t *)fn, CthOnly);
  CthSetStrategyDefault(result);
  return result;
}

void CthSuspend()
{
  CthThread next;
  if (CthCpvAccess(CthCurrent)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(CthCurrent)->choosefn();
  /** addition for tracing */
  trace_end_execute(0, -1, 0);
  /* end addition */
  CthResume(next);
}

void CthAwaken(th)
CthThread th;
{
  if (th->awakenfn == 0) CthNoStrategy();
  /** addition for tracing */
  CpvAccess(cThread) = th;
  trace_creation(0,0,0);
  /* end addition */
  th->awakenfn(th);
}

void CthSetStrategy(t, awkfn, chsfn)
CthThread t;
CthVoidFn awkfn;
CthThFn chsfn;
{
  t->awakenfn = awkfn;
  t->choosefn = chsfn;
}

void CthYield()
{
  CthAwaken(CthCpvAccess(CthCurrent));
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

void CthSetNext(CthThread t, CthThread v)
{
  t->qnext = v;
}

CthThread CthGetNext(CthThread t) 
{
  return t->qnext;
}
