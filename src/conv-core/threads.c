
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
 * Revision 1.1  1995-09-20 13:16:33  jyelon
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
 * void CthAwaken(CthThread t)
 *
 *   - puts the specified thread t in the ready-pool.  The "ready-pool"
 *     is a set of threads that are waiting for the CPU to execute them.
 *     Assuming that no thread holds the CPU forever, putting a thread in
 *     the ready-pool (using CthAwaken) ensures that it will receive the
 *     CPU at some point in the future.
 *
 * void CthSuspend()
 *
 *   - immediately transfers control to some thread in the ready-pool.
 *     if there is no thread in the ready pool, this is an error.
 *     Note: If the suspender wishes to receive control again at some point
 *     in the future, it must ensure that somebody eventually awakens it
 *     using CthAwaken.  For example, if a thread is suspending because it
 *     is waiting for a condition to be set, it should store itself in a
 *     queue associated with the condition variable.  When the condition
 *     is set, the threads in the condition's queue should be awakened.
 *     This threads package performs none of the queue manipulation described
 *     above: this is up to you. (Although Converse may provide a separate
 *     conditions+locks package that does the job).
 *
 * void CthExit()
 *
 *   - This behaves exactly like CthSuspend, except that the thread
 *     is freed in the process.
 *
 * void CthYield()
 *
 *   - Places the current thread in the ready-pool (see CthAwaken) and
 *     then transfers control to some other thread (see CthSuspend).  This
 *     combination gives up control temporarily, but ensures that control
 *     will eventually return.
 *
 * CthThread CthCreate(CthVoidFn fn, void *arg, int size,
 *                     CthVoidFn ready_insert_fn,
 *                     CthThFn ready_choose_fn)
 *
 *   - Creates a new thread object.  The thread is not placed in the ready
 *     pool.  When (and if) the thread eventually receives control, it will
 *     begin executing the specified function 'fn' with the specified
 *     argument.  The 'size' parameter specifies the stack size, 0 means
 *     use the default size.
 *
 *     It is the user's job to manage the ready-pool.  This gives the user
 *     maximum control over prioritization and scheduling.  The user must
 *     provide two functions: one that inserts a thread into the ready-pool,
 *     and one that selects a next-thread from the ready-pool (deleting it
 *     from the ready-pool at the same time).  These functions are called
 *     ready_insert_fn and ready_choose_fn respectively:
 *
 *     void ready_insert_fn(CthThread t);
 *     CthThread ready_choose_fn();
 *
 *     These functions must be provided on a per-thread basis.  (Eg, if you
 *     awaken a thread X, then X's ready_insert_fn will be called.  If a
 *     thread Y calls suspend, then Y's ready_choose_fn will be called to
 *     pick the next thread.)  Of course, you may use the same functions
 *     for all threads (the common case), but the specification on a per-
 *     thread basis gives you maximum flexibility in controlling scheduling.
 *
 * void CthSetVar(CthThread t, void **var, void *val)
 *
 *     Specifies that the variable 'var' should be set to 'val' whenever
 *     thread 't' is executing.  This is essentially a means for threads to
 *     have private data associated with them.  Note: each call to CthSetVar
 *     slightly increases the context-switch overhead of switching to 
 *     thread 't'.  Therefore, we recommend that you only store one or two
 *     var/val pairs on each thread.  Of course, val can be a pointer to
 *     a struct, so you can cheaply store any _volume_ of data that you want.
 *     Note: in case you decide to run your program on a shared-memory
 *     machine, var should almost certainly be processor-private (a pointer
 *     to a Cpv variable).
 *
 * void *CthGetVar(CthThread t, void **var)
 *
 *     This makes it possible to retrieve values previously stored with
 *     CthSetVar.  Returns the value that 'var' will be set to when 't' is
 *     running.
 *
 *
 *
 * Note: there are several possible ways to implement threads.   No one
 * way works on all machines.  Instead, we provide one implementation which
 * covers a lot of machines, and a second implementation that simply prints
 * "Not Implemented".  We may provide other implementations in the future.
 *
 *****************************************************************************/
 

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
#include "converse.h"

#define STACKSIZE (32768)
#define SLACK     256

struct CthThread
{
  jmp_buf    jb;
  CthVoidFn  fn;
  void      *arg;
  char      *top;
  CthVoidFn  ready_insert_fn;
  CthThFn    ready_choose_fn;
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
  thread_current = (CthThread)malloc(sizeof(struct CthThread));
  thread_current->fn=0;
  thread_current->arg=0;
  thread_current->data_count=0;
}

CthThread CthSelf()
{
  if (thread_current==0) CthInit();
  return thread_current;
}

CthThread CthCreate(fn, arg, size, insfn, chsfn)
  CthVoidFn fn; void *arg; int size;
  CthVoidFn insfn;
  CthThFn chsfn;
{
  CthThread  result; char *oldsp, *newsp; int offs, erralloc;
  if (size==0) size = STACKSIZE;
  if (thread_current == 0) CthInit();
  result = (CthThread)malloc(sizeof(struct CthThread) + size);
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
  result->data_count = 0;
  result->ready_insert_fn = insfn;
  result->ready_choose_fn = chsfn;
  erralloc = (char *)alloca(offs) - newsp;
  if (ABS(erralloc) >= SLACK) 
    { printf("error #83742.\n"); exit(1); }
  if (setjmp(result->jb)) {
    if (thread_exiting) { free(thread_exiting); thread_exiting=0; }
    (thread_current->fn)(thread_current->arg);
    CthExit();
  }
  else return result;
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

static void CthResume(t)
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

void CthExit()
{
  thread_exiting = thread_current;
  CthSuspend();
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

void CthAwaken(th)
     CthThread th;
{
  th->ready_insert_fn(th);
}

void CthSuspend()
{
  CthThread next = thread_current->ready_choose_fn();
  CthResume(next);
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
        
CthThread CthCreate(fn, arg, size, insfn, chsfn)
    CthVoidFn fn; void *arg; int size; CthVoidFn insfn; CthThFn chsfn;
    { CthFail(); }

void CthYield()
    { CthFail(); }

void CthSuspend()
    { CthFail(); }

void CthExit()
    { CthFail(); }

void CthAwaken(t)
    CthThread t;
    { CthFail(); }

CthThread CthSelf()
    { CthFail(); }

void CthSetVal(t, var, val)
    CthThread t; void **var; void *val;
    { CthFail(); }

void *CthGetVal(t, var)
    CthThread t; void **var;
    { CthFail(); }

#endif /* CMK_THREADS_UNAVAILABLE */
