
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
 * Revision 1.22  1996-07-02 21:01:39  jyelon
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
 * threads: implementation CMK_THREADS_USE_ALLOCA.
 *
 * This particular implementation of threads works on most machines that
 * support alloca.
 *
 * Note that we have NOT used Cpv variables.  This makes it possible to use
 * this version as the core threads package in the uth version.  I'm trying
 * to get around this design... I'd rather use Cpv variables.
 *
 * As a result of the lack of Cpv variables, this version won't work on any
 * shared-memory machine where globals are shared by default.
 *
 *****************************************************************************/

#ifdef CMK_THREADS_USE_ALLOCA_WITH_HEADER_FILE
#include <alloca.h>
#endif

#ifdef CMK_THREADS_USE_ALLOCA_WITH_PRAGMA
#pragma alloca
#endif

#ifdef CMK_THREADS_USE_ALLOCA
#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

char *CthData;

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


static CthThread  thread_current;
static CthThread  thread_exiting;
static int        thread_growsdown;
static int        thread_datasize;

#define ABS(a) (((a) > 0)? (a) : -(a) )

int CthImplemented()
    { return 1; }

void CthInit()
{
  char *sp1 = (char *)alloca(8);
  char *sp2 = (char *)alloca(8);
  if (sp2<sp1) thread_growsdown = 1;
  else         thread_growsdown = 0;
  thread_datasize=1;
  thread_current=
    (CthThread)CmiAlloc(sizeof(struct CthThreadStruct));
  thread_current->fn=0;
  thread_current->arg=0;
  thread_current->data=0;
  thread_current->datasize=0;
  thread_current->awakenfn = 0;
  thread_current->choosefn = 0;
}

CthThread CthSelf()
{
  return thread_current;
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
  thread_current->top = oldsp;
  if (thread_growsdown) {
    newsp -= SLACK;
    alloca(oldsp - newsp);
  } else {
    newsp += SLACK;
    alloca(newsp - oldsp);
  }
  thread_current = t;
  CthData = t->data;
  longjmp(t->jb, 1);
}

void CthFixData(t)
CthThread t;
{
  if (t->data == 0) {
    t->datasize = thread_datasize;
    t->data = (char *)malloc(thread_datasize);
    return;
  }
  if (t->datasize != thread_datasize) {
    t->datasize = thread_datasize;
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
  if (t == thread_current) return;
  CthFixData(t);
  if ((setjmp(thread_current->jb))==0)
    CthTransfer(t);
  if (thread_exiting)
    { CthFreeNow(thread_exiting); thread_exiting=0; }
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread  result; char *oldsp, *newsp; int offs, erralloc;
  if (size==0) size = STACKSIZE;
  result = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct) + size);
  oldsp = (char *)alloca(0);
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
  result->data = 0;
  result->datasize = 0;
  alloca(offs);
  if (setjmp(result->jb)) {
    if (thread_exiting)
      { CthFreeNow(thread_exiting); thread_exiting=0; }
    (thread_current->fn)(thread_current->arg);
    thread_exiting = thread_current;
    CthSuspend();
  }
  else return result;
}

void CthFree(t)
CthThread t;
{
  if (t==thread_current) {
    thread_exiting = t;
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

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  thread_datasize = (thread_datasize+align-1) & ~(align-1);
  result = thread_datasize;
  thread_datasize += size;
  CthFixData(thread_current);
  CthData = thread_current->data;
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

#ifdef CMK_THREADS_USE_JB_TWEAKING

#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

char *CthData;

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
    (thread_current->fn)(thread_current->arg);
    exit(1);
  }
}

void CthInit()
{
  int frames[2];
  jmpb bufs[2];
  int i, j, delta, size, *p0, *p1;

  thread_datasize=1;
  thread_current=(CthThread)CmiAlloc(sizeof(struct CthThreadStruct));
  thread_current->fn=0;
  thread_current->arg=0;
  thread_current->data=0;
  thread_current->datasize=0;
  thread_current->awakenfn = 0;
  thread_current->choosefn = 0;

  /* analyze the activation record. */
  CthInitSub1(bufs, frames, 1);
  CthInitSub2();
  thread_growsdown = (frames[0] < frames[1]);
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
  return thread_current;
}

static void CthTransfer(t)
CthThread t;
{
  thread_current = t;
  CthData = t->data;
  longjmp(t->jb, 1);
}

void CthFixData(t)
CthThread t;
{
  if (t->data == 0) {
    t->datasize = thread_datasize;
    t->data = (char *)malloc(thread_datasize);
    return;
  }
  if (t->datasize != thread_datasize) {
    t->datasize = thread_datasize;
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
  if (t == thread_current) return;
  CthFixData(t);
  if ((setjmp(thread_current->jb))==0)
    CthTransfer(t);
  if (thread_exiting)
    { CthFreeNow(thread_exiting); thread_exiting=0; }
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread  result; int i, sp;
  if (size==0) size = STACKSIZE;
  result = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct) + size);
  sp = ((int)(result->stack));
  sp += (thread_growsdown) ? (size - SLACK) : SLACK;
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
  if (t==thread_current) {
    thread_exiting = t;
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

int CthRegister(size)
int size;
{
  int result;
  int align = 1;
  while (size>align) align<<=1;
  thread_datasize = (thread_datasize+align-1) & ~(align-1);
  result = thread_datasize;
  thread_datasize += size;
  CthFixData(thread_current);
  CthData = thread_current->data;
  return result;
}

#endif CMK_THREADS_USE_JB_TWEAKING

/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_USE_EATSTACK
 *
 * This thing just doesn't work.  Maybe I'll figure out why someday.
 *
 * I got the idea for this from a Dr. Dobb's journal, of all places.
 * Threads are in the stack segment.  The topmost thread in the stack segment
 * is responsible for creating new threads.  Whenever a new thread is needed,
 * it recurses deeply until about 64k of stack space has been used up.  It
 * then saves its state using setjmp, thereby creating a new top thread,
 * and making the old top thread into a regular 64k thread, which goes on
 * the freelist.
 *
 * This implementation has two distinct disadvantages.  One, the number of
 * threads allowed is limited by the size to which your stack will grow.  Two,
 * the size of the thread is fixed at 64k regardless of what you specify,
 * since I haven't written any serious stack-memory allocation code yet.
 *
 * It has the obvious advantage that it doesn't require alloca.
 *
 *****************************************************************************/

#ifdef CMK_THREADS_USE_EATSTACK
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE_STD   65536
#define STACKSIZE_MAIN  (65536*4)

CMK_STATIC_PROTO void CthClipTop CMK_PROTO((int));

struct CthThreadStruct {
  char cmicore[CmiMsgHeaderSizeBytes]; /* So we can enqueue them */
  CthThread  next;
  jmp_buf    ctrl;
  jmp_buf    jb;
  CthVoidFn  fn;
  void      *arg;
  CthVoidFn  awakenfn;
  CthThFn    choosefn;
  char      *data;
  int        datasize;
};

CpvDeclare(char *, CthData);

CpvStaticDeclare(CthThread, CthCurr);
CpvStaticDeclare(CthThread, CthTop);
CpvStaticDeclare(CthThread, CthFreeList);
CpvStaticDeclare(CthThread, CthExiting);
CpvStaticDeclare(int      , CthDatasize);

int CthImplemented()
{
  return 1;
}

static void CthFreeNow(t)
CthThread t;
{
  t->next = CpvAccess(CthFreeList);
  CpvAccess(CthFreeList) = t;
}

static void CthFreeExiting()
{
  CthThread t = CpvAccess(CthExiting);
  if (t && (t!=CpvAccess(CthCurr)))
    { CthFreeNow(t); CpvAccess(CthExiting)=0; }
}

void CthFree(t)
CthThread t;
{
  if (t==CpvAccess(CthCurr))
    CpvAccess(CthExiting) = t;
  else CthFreeNow(t);
}

static void CthSaveLoad(save, load, index)
jmp_buf save; jmp_buf load; int index;
{
  if (setjmp(save)==0) longjmp(load, index);
}

static void CthNoStrategy()
{
  CmiPrintf("Called CthAwaken or CthSuspend before calling CthSetStrategy.\n");
  exit(1);
}
    
void CthSuspend()
{
  CthThread curr = CthSelf();
  CthThread next;
  if (curr->choosefn == 0) CthNoStrategy();
  next = curr->choosefn();
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
  CthAwaken(CthSelf());
  CthSuspend();
}
 
static void CthFixData(t)
CthThread t;
{
  if (t->data == 0) {
    t->datasize = CpvAccess(CthDatasize);
    t->data = (char *)malloc(t->datasize);
    return;
  }
  if (t->datasize != CpvAccess(CthDatasize)) {
    t->datasize = CpvAccess(CthDatasize);
    t->data = (char *)realloc(t->data, t->datasize);
    return;
  }
}

int CthRegister(size)
int size;
{
  int dsize = CpvAccess(CthDatasize);
  CthThread self = CthSelf();
  int result;
  int align = 1;
  while (size>align) align<<=1;
  dsize = (dsize + align-1) & ~(align-1);
  result = dsize;
  dsize += size;
  CpvAccess(CthDatasize) = dsize;
  CthFixData(self);
  CpvAccess(CthData) = self->data;
  return result;
}

static void CthReset(t,fn,arg)
CthThread t; CthVoidFn fn; void *arg;
{
  memcpy(t->jb, t->ctrl, sizeof(t->jb));
  t->fn = fn;
  t->arg = arg;
  t->awakenfn = 0;
  t->choosefn = 0;
  t->data = 0;
  t->datasize = 0;
}

static void CthExtendStack()
{
  CthThread t;
  /* I am top thread, and I have been asked by the current           */
  /* thread to extend the freelist.  I must recurse deeply using     */
  /* CthClipTop, which will truncate me and make me just another     */
  /* unused thread.  I must push my truncated self onto to freelist, */
  /* then go back to current thread (which invoked me).              */
  t = CpvAccess(CthTop);
  t->next = CpvAccess(CthFreeList);
  CpvAccess(CthFreeList) = t;
  CthClipTop(STACKSIZE_STD);
  longjmp(CpvAccess(CthCurr)->jb,1);
}

static void CthCallUserFn()
{
  CthThread t;
  /* I am a thread which has recently been removed from the     */
  /* freelist and returned by CthCreate.  I have just been      */
  /* awakened via CthResume, therefore, I am the current        */
  /* thread.  I must start executing my func.                   */
  t = CpvAccess(CthCurr);
  CthFreeExiting();
  (t->fn)(t->arg);
  CthFree(CthSelf());
  CthSuspend();
  printf("thread ran off end!\n");
  exit(1);
}

static void CthController()
{
  /* I am the top thread, and I have recursed deeply to lower my     */
  /* stack pointer. I break off what remains of the stack segment    */
  /* and store it in CthTop, thereby making myself the second-from-  */
  /* the-top thread, and then I return.  My callee will then take    */
  /* me and push me on the freelist.                                 */
  CpvAccess(CthTop) = (CthThread)malloc(sizeof(struct CthThreadStruct));
  switch(setjmp(CpvAccess(CthTop)->ctrl)) {
  case 0: return;
  case 1: CthExtendStack();
  case 2: CthCallUserFn();
  }
}

static void CthClip_1(size) 
int size;
{ char gap[1024];  CthClipTop(size-1024);  }

static void CthClip_4(size) 
int size;
{ char gap[4096];  CthClipTop(size-4096);  }

static void CthClip_16(size)
int size;
{ char gap[16384]; CthClipTop(size-16384); }

static void CthClip_64(size)
int size;
{ char gap[65536]; CthClipTop(size-65536); }

static void CthClipTop(size)
int size;
{
  if       (size >= 65536) CthClip_64(size);
  else if  (size >= 16384) CthClip_16(size);
  else if  (size >=  4096) CthClip_4(size);
  else if  (size >      0) CthClip_1(size);
  else CthController();
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread new;
  if (CpvAccess(CthFreeList)==0)
    CthSaveLoad(CpvAccess(CthCurr)->jb, CpvAccess(CthTop)->ctrl, 1);
  new = CpvAccess(CthFreeList);
  CpvAccess(CthFreeList) = new->next;
  CthReset(new, fn, arg);
  return new;
}

CthThread CthSelf()
{
  return CpvAccess(CthCurr);
}

void CthResume(t)
CthThread t;
{
  CthThread old;
  old = CpvAccess(CthCurr);
  CthFixData(t);
  CpvAccess(CthCurr) = t;
  CpvAccess(CthData) = t->data;
  CthSaveLoad(old->jb, t->jb, 2);
  CthFreeExiting();
}

void CthInit(argv)
char **argv;
{
  CpvInitialize(CthThread, CthCurr);
  CpvInitialize(CthThread, CthTop);
  CpvInitialize(CthThread, CthFreeList);
  CpvInitialize(CthThread, CthExiting);
  CpvInitialize(int,       CthDatasize);
  CpvInitialize(char *,    CthData);

  CpvAccess(CthCurr) = (CthThread)malloc(sizeof(struct CthThreadStruct));
  CpvAccess(CthTop) = 0;
  CpvAccess(CthFreeList) = 0;
  CpvAccess(CthExiting) = 0;
  CpvAccess(CthDatasize) = 0;
  CpvAccess(CthData) = 0;

  CthReset(CpvAccess(CthCurr),0,0);
  CthClipTop(STACKSIZE_MAIN);
}

#endif /* CMK_THREADS_USE_EATSTACK */


/*****************************************************************************
 *
 * threads: implementation CMK_THREADS_UNAVAILABLE
 *
 * This is a set of stubs I can use as a stopgap to get converse to compile
 * on machines to which I haven't yet ported threads.
 *
 *****************************************************************************/

#ifdef CMK_THREADS_UNAVAILABLE

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

