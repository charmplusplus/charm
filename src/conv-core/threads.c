
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
 * Revision 1.26  1997-01-17 15:49:08  jyelon
 * Made many changes for SMP version.  In particular, memory module now uses
 * CmiMemLock and CmiMemUnlock instead of CmiInterruptsBlock, which no longer
 * exists.  Threads package uses CthCpv to declare all its global vars.
 * Much other restructuring.
 *
 * Revision 1.25  1996/11/23 02:25:34  milind
 * Fixed several subtle bugs in the converse runtime for convex
 * exemplar.
 *
 * Revision 1.24  1996/10/24 20:51:50  milind
 * Removed the additional token after one #endif.
 *
 * Revision 1.23  1996/07/15 21:00:49  jyelon
 * Moved some code into common, changed mach-flags from #ifdef to #if
 *
 * Revision 1.22  1996/07/02 21:01:39  jyelon
 * Added CMK_THREADS_USE_JB_TWEAKING
 *
 * Revision 1.21  1995/10/31 19:53:21  jyelon
 * Added 'CMK_THREADS_USE_ALLOCA_WITH_PRAGMA'
 *
 * Revision 1.20  1995/10/31  19:49:30  jyelon
 * Added 'pragma alloca'
 *
 * Revision 1.19  1995/10/20  17:29:10  jyelon
 * *** empty log message ***
 *
 * Revision 1.18  1995/10/19  18:21:39  jyelon
 * moved CthSetStrategyDefault to convcore.c
 *
 * Revision 1.17  1995/10/19  04:19:47  jyelon
 * A correction to eatstack.
 *
 * Revision 1.16  1995/10/18  22:20:17  jyelon
 * Added 'eatstack' threads implementation.
 *
 * Revision 1.15  1995/10/18  01:58:43  jyelon
 * added ifdef around 'alloca.h'
 *
 * Revision 1.14  1995/10/13  22:33:36  jyelon
 * There was some bizzare test in CthCreate which I think was supposed
 * to check for failure of some kind, but it didn't work.
 *
 * Revision 1.13  1995/10/13  18:14:10  jyelon
 * K&R changes, etc.
 *
 * Revision 1.12  1995/10/10  06:15:23  jyelon
 * Fixed a bug.
 *
 * Revision 1.11  1995/09/30  15:03:33  jyelon
 * Cleared up some confusion about 'private' variables in uniprocessor version.
 *
 * Revision 1.10  1995/09/30  11:48:15  jyelon
 * Fixed a bug (CthSetVar(t,...) failed when t is current thread.)
 *
 * Revision 1.9  1995/09/27  22:23:15  jyelon
 * Many bug-fixes.  Added Cpv macros to threads package.
 *
 * Revision 1.8  1995/09/26  18:30:46  jyelon
 * *** empty log message ***
 *
 * Revision 1.7  1995/09/26  18:26:00  jyelon
 * Added CthSetStrategyDefault, and cleaned up a bit.
 *
 * Revision 1.6  1995/09/20  17:22:14  jyelon
 * Added CthImplemented
 *
 * Revision 1.5  1995/09/20  16:36:56  jyelon
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

/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_UNAVAILABLE
 *
 * This is a set of stubs I can use as a stopgap to get converse to compile
 * on machines to which I haven't yet ported threads.
 *
 *****************************************************************************/

#if CMK_THREADS_UNAVAILABLE

static void CthFail()
{
  CmiPrintf("Threads not currently supported on this hardware platform.\n");
  exit(1);
}

int CthImplemented()
    { return 0; }

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
    CthThread t; CthVoidFn awkfn; CthThFn chsfn;
    { CthFail(); }

void CthYield()
    { CthFail(); }

void CthInit()
    {  }

#endif /* CMK_THREADS_UNAVAILABLE */

/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_USE_ALLOCA.
 *
 * This particular implementation of threads works on most machines that
 * support alloca.
 *
 *****************************************************************************/

#if CMK_THREADS_REQUIRE_ALLOCA_H
#include <alloca.h>
#endif

#if CMK_THREADS_REQUIRE_PRAGMA_ALLOCA
#pragma alloca
#endif

#if CMK_THREADS_USE_ALLOCA
#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

CthCpvDeclare(char *,CthData);

struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes]; /* So we can enqueue them */
  jmp_buf    jb;
  CthVoidFn  fn;
  void      *arg;
  char      *top;
  CthVoidFn  awakenfn;
  CthThFn    choosefn;
  char      *data;
  int        datasize;
  double     stack[1];
};


CthCpvStatic(CthThread,  thread_current);
CthCpvStatic(CthThread,  thread_exiting);
CthCpvStatic(int,        thread_growsdown);
CthCpvStatic(int,        thread_datasize);

#define ABS(a) (((a) > 0)? (a) : -(a) )

int CthImplemented()
    { return 1; }

void CthInit()
{
  CthThread t;
  char *sp1 = (char *)alloca(8);
  char *sp2 = (char *)alloca(8);
  
  CthCpvInitialize(char *,CthData);
  CthCpvInitialize(CthThread,  thread_current);
  CthCpvInitialize(CthThread,  thread_exiting);
  CthCpvInitialize(int,        thread_growsdown);
  CthCpvInitialize(int,        thread_datasize);

  if (sp2<sp1) CthCpvAccess(thread_growsdown) = 1;
  else         CthCpvAccess(thread_growsdown) = 0;
  t = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct));
  t->fn=0;
  t->arg=0;
  t->data=0;
  t->datasize=0;
  t->awakenfn = 0;
  t->choosefn = 0;
  CthCpvAccess(thread_current)=t;
  CthCpvAccess(thread_datasize)=1;
  CthCpvAccess(thread_exiting)=0;
  CthCpvAccess(CthData)=0;
}

CthThread CthSelf()
{
  return CthCpvAccess(thread_current);
}

static void CthTransfer(t)
CthThread t;
{    
  char *oldsp, *newsp;
  /* Put the stack pointer in such a position such that   */
  /* the longjmp moves the stack pointer down (this way,  */
  /* the rs6000 doesn't complain about "longjmp to an     */
  /* inactive stack frame"                                */
  oldsp = (char *)alloca(0);
  newsp = t->top;
  CthCpvAccess(thread_current)->top = oldsp;
  if (CthCpvAccess(thread_growsdown)) {
    newsp -= SLACK;
    alloca(oldsp - newsp);
  } else {
    newsp += SLACK;
    alloca(newsp - oldsp);
  }
  CthCpvAccess(thread_current) = t;
  CthCpvAccess(CthData) = t->data;
  longjmp(t->jb, 1);
}

void CthFixData(t)
CthThread t;
{
  int datasize = CthCpvAccess(thread_datasize);
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

static void CthFreeNow(t)
CthThread t;
{
  if (t->data) free(t->data); 
  CmiFree(t);
}

void CthResume(t)
CthThread t;
{
  int i;
  CthThread tc = CthCpvAccess(thread_current);
  if (t == tc) return;
  CthFixData(t);
  if (setjmp(tc->jb)==0)
    CthTransfer(t);
  if (CthCpvAccess(thread_exiting))
    { CthFreeNow(CthCpvAccess(thread_exiting)); CthCpvAccess(thread_exiting)=0; }
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread  result; char *oldsp, *newsp; int offs, erralloc;
  if (size==0) size = STACKSIZE;
  result = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct) + size);
  oldsp = (char *)alloca(0);
  if (CthCpvAccess(thread_growsdown)) {
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
  result->data = 0;
  result->datasize = 0;
  alloca(offs);
  if (setjmp(result->jb)) {
    if (CthCpvAccess(thread_exiting))
      { CthFreeNow(CthCpvAccess(thread_exiting)); CthCpvAccess(thread_exiting)=0; }
    (CthCpvAccess(thread_current)->fn)(CthCpvAccess(thread_current)->arg);
    CthCpvAccess(thread_exiting) = CthCpvAccess(thread_current);
    CthSuspend();
  }
  else return result;
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(thread_current)) {
    CthCpvAccess(thread_exiting) = t;
  } else CthFreeNow(t);
}

static void CthNoStrategy()
{
  CmiPrintf("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
  exit(1);
}

void CthSuspend()
{
  CthThread next;
  if (CthCpvAccess(thread_current)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(thread_current)->choosefn();
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
  CthAwaken(CthCpvAccess(thread_current));
  CthSuspend();
}

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  CthCpvAccess(thread_datasize) = (CthCpvAccess(thread_datasize)+align-1) & ~(align-1);
  result = CthCpvAccess(thread_datasize);
  CthCpvAccess(thread_datasize) += size;
  CthFixData(CthCpvAccess(thread_current));
  CthCpvAccess(CthData) = CthCpvAccess(thread_current)->data;
  return result;
}

#endif /* CMK_THREADS_USE_ALLOCA */

/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_USE_JB_TWEAKING
 *
 * This threads implementation saves and restores state using setjmp
 * and longjmp, and it creates new states by doing a setjmp and then
 * twiddling the contents of the jmp_buf.  It uses a heuristic to find
 * the places in the jmp_buf it needs to adjust to change the stack
 * pointer.  It sometimes works.  It has the advantage that it doesn't
 * require alloca.
 *
 ****************************************************************************/

#if CMK_THREADS_USE_JB_TWEAKING

#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

CthCpvDeclare(char *,CthData);

typedef struct { jmp_buf jb; } jmpb;

struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes]; /* So we can enqueue them */
  jmp_buf    jb;
  CthVoidFn  fn;
  void      *arg;
  CthVoidFn  awakenfn;
  CthThFn    choosefn;
  char      *data;
  int        datasize;
  double     stack[1];
};


static jmp_buf    thread_launching;
static CthThread  thread_current;
static CthThread  thread_exiting;
static int        thread_growsdown;
static int        thread_datasize;
static int        thread_jb_offsets[10];
static int        thread_jb_count;

#define ABS(a) (((a) > 0)? (a) : -(a) )

int CthImplemented()
    { return 1; }

static void CthInitSub1(jmpb *bufs, int *frames, int n)
{
  double d;
  frames[n] = (int)(&d);
  setjmp(bufs[n].jb);
  if (n==0) return;
  CthInitSub1(bufs, frames, n-1);
}

static void CthInitSub2()
{
  if (setjmp(thread_launching)) {
    (CthCpvAccess(thread_current)->fn)(CthCpvAccess(thread_current)->arg);
    exit(1);
  }
}

void CthInit()
{
  int frames[2];
  jmpb bufs[2];
  int i, j, delta, size, *p0, *p1;

  CthCpvAccess(thread_datasize)=1;
  CthCpvAccess(thread_current)=(CthThread)CmiAlloc(sizeof(struct CthThreadStruct));
  CthCpvAccess(thread_current)->fn=0;
  CthCpvAccess(thread_current)->arg=0;
  CthCpvAccess(thread_current)->data=0;
  CthCpvAccess(thread_current)->datasize=0;
  CthCpvAccess(thread_current)->awakenfn = 0;
  CthCpvAccess(thread_current)->choosefn = 0;

  /* analyze the activation record. */
  CthInitSub1(bufs, frames, 1);
  CthInitSub2();
  CthCpvAccess(thread_growsdown) = (frames[0] < frames[1]);
  size = (sizeof(jmpb)/sizeof(int));
  delta = frames[0]-frames[1];
  p0 = (int *)(bufs+0);
  p1 = (int *)(bufs+1);
  thread_jb_count = 0;
  for (i=0; i<size; i++) {
    if (thread_jb_count==10) goto fail;
    if ((p0[i]-p1[i])==delta) {
      thread_jb_offsets[thread_jb_count++] = i;
      ((int *)(&thread_launching))[i] -= (int)(frames[1]);
    }
  }
  if (thread_jb_count == 0) goto fail;
  return;
fail:
  CmiPrintf("Thread initialization failed.\n");
  exit(1);
}

CthThread CthSelf()
{
  return CthCpvAccess(thread_current);
}

static void CthTransfer(t)
CthThread t;
{
  CthCpvAccess(thread_current) = t;
  CthCpvAccess(CthData) = t->data;
  longjmp(t->jb, 1);
}

void CthFixData(t)
CthThread t;
{
  if (t->data == 0) {
    t->datasize = CthCpvAccess(thread_datasize);
    t->data = (char *)malloc(CthCpvAccess(thread_datasize));
    return;
  }
  if (t->datasize != CthCpvAccess(thread_datasize)) {
    t->datasize = CthCpvAccess(thread_datasize);
    t->data = (char *)realloc(t->data, t->datasize);
    return;
  }
}

static void CthFreeNow(t)
CthThread t;
{
  if (t->data) free(t->data); 
  CmiFree(t);
}

void CthResume(t)
CthThread t;
{
  int i;
  if (t == CthCpvAccess(thread_current)) return;
  CthFixData(t);
  if ((setjmp(CthCpvAccess(thread_current)->jb))==0)
    CthTransfer(t);
  if (CthCpvAccess(thread_exiting))
    { CthFreeNow(CthCpvAccess(thread_exiting)); CthCpvAccess(thread_exiting)=0; }
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread  result; int i, sp;
  if (size==0) size = STACKSIZE;
  result = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct) + size);
  sp = ((int)(result->stack));
  sp += (CthCpvAccess(thread_growsdown)) ? (size - SLACK) : SLACK;
  result->fn = fn;
  result->arg = arg;
  result->awakenfn = 0;
  result->choosefn = 0;
  result->data = 0;
  result->datasize = 0;
  memcpy(&(result->jb), &thread_launching, sizeof(thread_launching));
  for (i=0; i<thread_jb_count; i++)
    ((int *)(&(result->jb)))[thread_jb_offsets[i]] += sp;
  return result;
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(thread_current)) {
    CthCpvAccess(thread_exiting) = t;
  } else CthFreeNow(t);
}

static void CthNoStrategy()
{
  CmiPrintf("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
  exit(1);
}

void CthSuspend()
{
  CthThread next;
  if (CthCpvAccess(thread_current)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(thread_current)->choosefn();
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
  CthAwaken(CthCpvAccess(thread_current));
  CthSuspend();
}

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  CthCpvAccess(thread_datasize) = (CthCpvAccess(thread_datasize)+align-1) & ~(align-1);
  result = CthCpvAccess(thread_datasize);
  CthCpvAccess(thread_datasize) += size;
  CthFixData(CthCpvAccess(thread_current));
  CthCpvAccess(CthData) = CthCpvAccess(thread_current)->data;
  return result;
}

#endif

/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_USE_JB_TWEAKING_EXEMPLAR
 *
 * This threads implementation saves and restores state using setjmp
 * and longjmp, and it creates new states by doing a setjmp and then
 * twiddling the contents of the jmp_buf.  It uses a heuristic to find
 * the places in the jmp_buf it needs to adjust to change the stack
 * pointer.  It sometimes works.  It has the advantage that it doesn't
 * require alloca.
 *
 ****************************************************************************/

#if CMK_THREADS_USE_JB_TWEAKING_EXEMPLAR

#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

CthCpvDeclare(char*,CthData);

typedef struct { jmp_buf jb; } jmpb;

typedef struct CthThreadStruct
{
  char cmicore[CmiMsgHeaderSizeBytes]; /* So we can enqueue them */
  jmp_buf    jb;
  CthVoidFn  fn;
  void      *arg;
  CthVoidFn  awakenfn;
  CthThFn    choosefn;
  char      *data;
  int        datasize;
  double     stack[1];
}ThreadStruct;


typedef int arr10[10];

CthCpvStatic(jmp_buf,thread_launching);
CthCpvStatic(CthThread,thread_current);
CthCpvStatic(CthThread,thread_exiting);
CthCpvStatic(int,thread_growsdown);
CthCpvStatic(int,thread_datasize);
CthCpvStatic(arr10,thread_jb_offsets);
CthCpvStatic(int,thread_jb_count);

#define ABS(a) (((a) > 0)? (a) : -(a) )

int CthImplemented()
    { return 1; }

static void CthInitSub1(jmpb *bufs, int *frames, int n)
{
  double d;
  frames[n] = (int)(&d);
  setjmp(bufs[n].jb);
  if (n==0) return;
  CthInitSub1(bufs, frames, n-1);
}

static void CthInitSub2()
{
  if (setjmp(CthCpvAccess(thread_launching))) {
    (CthCpvAccess(thread_current)->fn)(CthCpvAccess(thread_current)->arg);
    exit(1);
  }
}

void CthInit()
{
  int frames[2];
  jmpb bufs[2];
  int i, j, delta, size, *p0, *p1;

  CthCpvInitialize(char*,CthData);
  CthCpvInitialize(jmp_buf,thread_launching);
  CthCpvInitialize(CthThread,thread_current);
  CthCpvInitialize(CthThread,thread_exiting);
  CthCpvInitialize(int,thread_growsdown);
  CthCpvInitialize(int,thread_datasize);
  CthCpvInitialize(arr10,thread_jb_offsets);
  CthCpvInitialize(int,thread_jb_count);
  
  CthCpvAccess(thread_datasize)=1;
  CthCpvAccess(thread_current)=(CthThread)CmiAlloc(sizeof(struct CthThreadStruct));
  CthCpvAccess(thread_current)->fn=0;
  CthCpvAccess(thread_current)->arg=0;
  CthCpvAccess(thread_current)->data=0;
  CthCpvAccess(thread_current)->datasize=0;
  CthCpvAccess(thread_current)->awakenfn = 0;
  CthCpvAccess(thread_current)->choosefn = 0;

  /* analyze the activation record. */
  CthInitSub1(bufs, frames, 1);
  CthInitSub2();
  CthCpvAccess(thread_growsdown) = (frames[0] < frames[1]);
  size = (sizeof(jmpb)/sizeof(int));
  delta = frames[0]-frames[1];
  p0 = (int *)(bufs+0);
  p1 = (int *)(bufs+1);
  CthCpvAccess(thread_jb_count) = 0;
  for (i=0; i<size; i++) {
    if (CthCpvAccess(thread_jb_count)==10) goto fail;
    if ((p0[i]-p1[i])==delta) {
      CthCpvAccess(thread_jb_offsets)[CthCpvAccess(thread_jb_count)++] = i;
      ((int *)(&CthCpvAccess(thread_launching)))[i] -= (int)(frames[1]);
    }
  }
  if (CthCpvAccess(thread_jb_count) == 0) goto fail;
  return;
fail:
  CmiPrintf("Thread initialization failed.\n");
  exit(1);
}

CthThread CthSelf()
{
  return CthCpvAccess(thread_current);
}

static void CthTransfer(t)
CthThread t;
{
  CthCpvAccess(thread_current) = t;
  CthCpvAccess(CthData) = t->data;
  longjmp(t->jb, 1);
}

void CthFixData(t)
CthThread t;
{
  if (t->data == 0) {
    t->datasize = CthCpvAccess(thread_datasize);
    t->data = (char *)malloc(CthCpvAccess(thread_datasize));
    return;
  }
  if (t->datasize != CthCpvAccess(thread_datasize)) {
    t->datasize = CthCpvAccess(thread_datasize);
    t->data = (char *)realloc(t->data, t->datasize);
    return;
  }
}

static void CthFreeNow(t)
CthThread t;
{
  if (t->data) free(t->data); 
  CmiFree(t);
}

void CthResume(t)
CthThread t;
{
  int i;
  if (t == CthCpvAccess(thread_current)) return;
  CthFixData(t);
  if ((setjmp(CthCpvAccess(thread_current)->jb))==0)
    CthTransfer(t);
  if (CthCpvAccess(thread_exiting))
    { CthFreeNow(CthCpvAccess(thread_exiting)); CthCpvAccess(thread_exiting)=0; }
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread  result; int i, sp;
  if (size==0) size = STACKSIZE;
  result = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct) + size);
  sp = ((int)(result->stack));
  sp += (CthCpvAccess(thread_growsdown)) ? (size - SLACK) : SLACK;
  result->fn = fn;
  result->arg = arg;
  result->awakenfn = 0;
  result->choosefn = 0;
  result->data = 0;
  result->datasize = 0;
  memcpy(&(result->jb), &CthCpvAccess(thread_launching), sizeof(CthCpvAccess(thread_launching)));
  for (i=0; i<CthCpvAccess(thread_jb_count); i++)
    ((int *)(&(result->jb)))[CthCpvAccess(thread_jb_offsets)[i]] += sp;
  return result;
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(thread_current)) {
    CthCpvAccess(thread_exiting) = t;
  } else CthFreeNow(t);
}

static void CthNoStrategy()
{
  CmiPrintf("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
  exit(1);
}

void CthSuspend()
{
  CthThread next;
  if (CthCpvAccess(thread_current)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(thread_current)->choosefn();
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
  CthAwaken(CthCpvAccess(thread_current));
  CthSuspend();
}

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  CthCpvAccess(thread_datasize) = (CthCpvAccess(thread_datasize)+align-1) & ~(align-1);
  result = CthCpvAccess(thread_datasize);
  CthCpvAccess(thread_datasize) += size;
  CthFixData(CthCpvAccess(thread_current));
  CthCpvAccess(CthData) = CthCpvAccess(thread_current)->data;
  return result;
}

#endif

