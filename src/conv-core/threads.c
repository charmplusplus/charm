
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
 * $Log$
 * Revision 1.5  1995-09-20 16:36:56  jyelon
 * *** empty log message ***
 *
 * Revision 1.4  1995/09/20  15:39:54  jyelon
 * Still ironing out initial bugs.
 *
 * Revision 1.3  1995/09/20  15:07:44  jyelon
 * *** empty log message ***
 *
 * Revision 1.2  1995/09/20  14:58:12  jyelon
 * Did some work on threads stuff.
 *
 * Revision 1.1  1995/09/20  13:16:33  jyelon
 * Initial revision
 *
 * Revision 2.0  1995/07/05  22:22:26  brunner
 * Separated out megatest, and added RCS headers to all files
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
 *     the free will actually be postponed until the thread exits.
 *
 *
 *     It is the user's job to manage the ready-pool.  This gives the user
 *     maximum control over prioritization and scheduling.  The user must
 *     provide two functions: one that inserts a thread into the ready-pool,
 *     and one that selects a next-thread from the ready-pool (deleting it
 *     from the ready-pool at the same time).  These functions are called
 *     ready_insert_fn and ready_choose_fn respectively:
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
 * void CthYield()
 *
 *   - simply executes { CthAwaken(CthSelf()); CthSuspend(); }.  This
 *     combination gives up control temporarily, but ensures that control
 *     will eventually return.
 *
 *
 * The threads-package makes it possible to associate private data
 * with a thread.  to this end, we provide the following functions:
 *
 * void CthSetVar(CthThread t, void **var, void *val)
 *
 *     Specifies that the variable global variable pointed to by 'var'
 *     should be set to value 'val' whenever thread 't' is executing.
 *     'var' should be of type (void *), or at least should be coercible
 *     to a (void *).  This can be used to associate thread-private data
 *     with thread 't'.
 *
 * it is intended that this function be used as follows:
 *
 *     struct th_info { any thread-private info desired }
 *     struct th_info *curr_info;
 *
 *     ...
 *
 *     t = CthCreate(...);
 *     CthSetVar(t, &curr_info, malloc(sizeof(struct th_info));
 *
 * That way, whenever thread t is executing, it can access its private
 * data by dereferencing curr_info.  We also provide:
 *
 * void *CthGetVar(CthThread t, void **var)
 *
 *     This makes it possible to retrieve values previously stored with
 *     CthSetVar when t is _not_ executing.  Returns the value that 'var' will
 *     be set to when 't' is running.
 *
 *
 *
 * Note: there are several possible ways to implement threads.   No one
 * way works on all machines.  Instead, we provide one implementation which
 * covers a lot of machines, and a second implementation that simply prints
 * "Not Implemented".  We may provide other implementations in the future.
 *
 *****************************************************************************/
 
#include "converse.h"

/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_USE_ALLOCA.
 *
 * This particular implementation of threads works on most machines that
 * support alloca.  So far, all workstations that we have tested can run this
 * version.
 *
 *****************************************************************************/

#ifdef CMK_THREADS_USE_ALLOCA

#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

struct StructCthThread
{
  jmp_buf    jb;
  CthVoidFn  fn;
  void      *arg;
  char      *top;
  CthVoidFn  awakenfn;
  CthThFn    choosefn;
  void     **data_var[10];
  void      *data_val[10];
  int        data_count;
  double     stack[1];
};

static CthThread thread_current;
static CthThread thread_exiting;
static int       thread_growsdown;

#define ABS(a) (((a) > 0)? (a) : -(a) )

static void CthInit()
{
  char *sp1 = alloca(8);
  char *sp2 = alloca(8);
  if (sp2<sp1) thread_growsdown = 1;
  else         thread_growsdown = 0;
  thread_current = (CthThread)malloc(sizeof(struct StructCthThread));
  thread_current->fn=0;
  thread_current->arg=0;
  thread_current->data_count=0;
}

CthThread CthSelf()
{
  if (thread_current==0) CthInit();
  return thread_current;
}

static void CthTransfer(CthThread t)
{    
  char *oldsp, *newsp;
  /* Put the stack pointer in such a position such that   */
  /* the longjmp moves the stack pointer down (this way,  */
  /* the rs6000 doesn't complain about "longjmp to an     */
  /* inactive stack frame"                                */
  oldsp = alloca(0);
  newsp = t->top;
  thread_current->top = oldsp;
  if (thread_growsdown) {
    newsp -= SLACK;
    alloca(oldsp - newsp);
  } else {
    newsp += SLACK;
    alloca(newsp - oldsp);
  }
  thread_current = t;
  longjmp(t->jb, 1);
}

void CthResume(t)
     CthThread t;
{
  int i;
  for (i=0; i<t->data_count; i++)
    *(t->data_var[i]) = t->data_val[i];
  if (t == thread_current) return;
  if ((setjmp(thread_current->jb))==0)
    CthTransfer(t);
  if (thread_exiting) { free(thread_exiting); thread_exiting=0; }
}

CthThread CthCreate(fn, arg, size)
  CthVoidFn fn; void *arg; int size;
{
  CthThread  result; char *oldsp, *newsp; int offs, erralloc;
  if (size==0) size = STACKSIZE;
  if (thread_current == 0) CthInit();
  result = (CthThread)malloc(sizeof(struct StructCthThread) + size);
  oldsp = alloca(0);
  if (thread_growsdown) {
      newsp = ((char *)(result->stack)) + size - SLACK;
      offs = oldsp - newsp;
  } else {
      newsp = ((char *)(result->stack)) + SLACK;
      offs = newsp - oldsp;
  }
  result->fn = fn;
  result->arg = arg;
  result->top = newsp;
  result->awakenfn = 0;
  result->choosefn = 0;
  result->data_count = 0;
  erralloc = (char *)alloca(offs) - newsp;
  if (ABS(erralloc) >= SLACK) 
    { printf("error #83742.\n"); exit(1); }
  if (setjmp(result->jb)) {
    if (thread_exiting) { free(thread_exiting); thread_exiting=0; }
    (thread_current->fn)(thread_current->arg);
    thread_exiting = thread_current;
    CthSuspend();
  }
  else return result;
}

void CthFree(CthThread t)
{
  if (t==thread_current) {
    thread_exiting = t;
  } else free(t);
}

static void CthNoStrategy()
{
  CmiPrintf("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
  exit(1);
}

void CthSuspend()
{
  CthThread next;
  if (thread_current->choosefn == 0) CthNoStrategy();
  next = thread_current->choosefn();
  CthResume(next);
}

void CthAwaken(th)
     CthThread th;
{
  if (th->awakenfn == 0) CthNoStrategy();
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
  CthAwaken(thread_current);
  CthSuspend();
}

void CthSetVar(CthThread thr, void **var, void *val)
{
  int i;
  int data_count = thr->data_count;
  for (i=0; i<data_count; i++)
    if (var == thr->data_var[i])
      { thr->data_val[i]=val; return; }
  if (data_count==10) {
    fprintf(stderr,"CthSetVar: Thread data space full.\n");
    exit(1);
  }
  thr->data_var[data_count]=var;
  thr->data_val[data_count]=val;
  thr->data_count++;
}

void *CthGetVar(CthThread thr, void **var)
{
  int i;
  int data_count = thr->data_count;
  for (i=0; i<data_count; i++)
    if (var == thr->data_var[i])
      return thr->data_val[i];
  return 0;
}

#endif /* CMK_THREADS_USE_ALLOCA */

/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_UNAVAILABLE
 *
 * This particular implementation of threads works on most machines that
 * support alloca.  So far, all workstations that we have tested can run this
 * version.
 *
 *****************************************************************************/

#ifdef CMK_THREADS_UNAVAILABLE

static void CthFail()
{
  CmiPrintf("Threads not currently supported on this hardware platform.\n");
  exit(1);
}
        
CthThread CthSelf()
    { CthFail(); }

void CthResume(t)
    CthThread t;
    { CthFail(); }

CthThread CthCreate(fn, arg, size)
    CthVoidFn fn; void *arg; int size;
    { CthFail(); }

void CthFree(t)
    CthThread t;
    { CthFail(); }

void CthSuspend()
    { CthFail(); }

void CthAwaken(t)
    CthThread t;
    { CthFail(); }

void CthSetStrategy(t, awkfn, chsfn)
    CthVoidFn insfn; CthThFn chsfn;
    { CthFail(); }

void CthYield()
    { CthFail(); }

void CthSetVar(t, var, val)
    CthThread t; void **var; void *val;
    { CthFail(); }

void *CthGetVar(t, var)
    CthThread t; void **var;
    { CthFail(); }

#endif /* CMK_THREADS_UNAVAILABLE */
