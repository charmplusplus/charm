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
  int        autoyield_enable;
  int        autoyield_blocks;
  char      *data;
  int        datasize;
  double     stack[1];
};

CthCpvStatic(CthThread,  CthCurrent);
CthCpvStatic(CthThread,  CthExiting);
CthCpvStatic(int,        CthGrowsdown);
CthCpvStatic(int,        CthDatasize);

#define ABS(a) (((a) > 0)? (a) : -(a) )

int CthImplemented()
    { return 1; }

static void CthThreadInit(t)
CthThread t;
{
  t->fn=0;
  t->arg=0;
  t->awakenfn = 0;
  t->choosefn = 0;
  t->data=0;
  t->datasize=0;
  t->autoyield_enable = 0;
  t->autoyield_blocks = 0;
}

void CthInit()
{
  CthThread t;
  char *sp1 = (char *)alloca(8);
  char *sp2 = (char *)alloca(8);
  
  CthCpvInitialize(char *,CthData);
  CthCpvInitialize(CthThread,  CthCurrent);
  CthCpvInitialize(CthThread,  CthExiting);
  CthCpvInitialize(int,        CthGrowsdown);
  CthCpvInitialize(int,        CthDatasize);

  if (sp2<sp1) CthCpvAccess(CthGrowsdown) = 1;
  else         CthCpvAccess(CthGrowsdown) = 0;
  t = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct));
  CthThreadInit(t);
  CthCpvAccess(CthCurrent)=t;
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthExiting)=0;
  CthCpvAccess(CthData)=0;
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
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

static void CthFreeNow(t)
CthThread t;
{
  if (t->data) free(t->data); 
  CmiFree(t);
}

static void CthTransfer(f, t)
CthThread f, t;
{    
  char *oldsp, *newsp;
  /* Put the stack pointer in such a position such that   */
  /* the longjmp moves the stack pointer down (this way,  */
  /* the rs6000 doesn't complain about "longjmp to an     */
  /* inactive stack frame"                                */
  if (setjmp(f->jb)==0) {
    oldsp = (char *)alloca(0);
    newsp = t->top;
    CthCpvAccess(CthCurrent)->top = oldsp;
    if (CthCpvAccess(CthGrowsdown)) {
      newsp -= SLACK;
      alloca(oldsp - newsp);
    } else {
      newsp += SLACK;
      alloca(newsp - oldsp);
    }
    CthCpvAccess(CthCurrent) = t;
    CthCpvAccess(CthData) = t->data;
    longjmp(t->jb, 1);
  }
}

void CthResume(t)
CthThread t;
{
  int i;
  CthThread tc = CthCpvAccess(CthCurrent);
  if (t == tc) return;
  CthFixData(t);
  CthTransfer(tc, t);
  if (CthCpvAccess(CthExiting)) {
    CthFreeNow(CthCpvAccess(CthExiting));
    CthCpvAccess(CthExiting)=0;
  }
}

static void CthInitHold(CthThread f, CthThread t)
{
  if (setjmp(t->jb)==0)
    longjmp(CthCpvAccess(CthCurrent)->jb, 1);
}

static void CthBeginThread(CthThread f, CthThread t)
{
  CthInitHold(f, t);
  (CthCpvAccess(CthCurrent)->fn)(CthCpvAccess(CthCurrent)->arg);
  CthCpvAccess(CthExiting) = CthCpvAccess(CthCurrent);
  CthSuspend();
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread  result; char *oldsp, *newsp; size_t offs; int erralloc;
  if (size==0) size = STACKSIZE;
  result = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct) + size);
  oldsp = (char *)alloca(0);
  if (CthCpvAccess(CthGrowsdown)) {
    newsp = ((char *)(result->stack)) + size - SLACK;
    offs = oldsp - newsp;
  } else {
    newsp = ((char *)(result->stack)) + SLACK;
    offs = newsp - oldsp;
  }
  CthThreadInit(result);
  result->fn = fn;
  result->arg = arg;
  result->top = newsp;
  if (setjmp(CthCpvAccess(CthCurrent)->jb)==0) {
    alloca(offs);
    CthBeginThread(CthCpvAccess(CthCurrent), result);
  }
  return result;
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(CthCurrent)) {
    CthCpvAccess(CthExiting) = t;
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
  if (CthCpvAccess(CthCurrent)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(CthCurrent)->choosefn();
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

CthCpvDeclare(char*,CthData);

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
  int        autoyield_enable;
  int        autoyield_blocks;
  double     stack[1];
};

typedef size_t arr10[10];
CthCpvStatic(jmp_buf,CthLaunching);
CthCpvStatic(CthThread,CthCurrent);
CthCpvStatic(CthThread,CthExiting);
CthCpvStatic(int,CthGrowsdown);
CthCpvStatic(int,CthDatasize);
CthCpvStatic(arr10,Cth_jb_offsets);
CthCpvStatic(int,Cth_jb_count);

#define ABS(a) (((a) > 0)? (a) : -(a) )

int CthImplemented()
    { return 1; }

static void CthInitSub1(jmpb *bufs, size_t *frames, int n)
{
  double d;
  frames[n] = (size_t)(&d);
  setjmp(bufs[n].jb);
  if (n==0) return;
  CthInitSub1(bufs, frames, n-1);
}

static void CthInitSub2()
{
  if (setjmp(CthCpvAccess(CthLaunching))) {
    (CthCpvAccess(CthCurrent)->fn)(CthCpvAccess(CthCurrent)->arg);
    exit(1);
  }
}

void CthInit()
{
  size_t frames[2];
  jmpb bufs[2];
  int i, j, size;
  size_t delta, *p0, *p1;

  CthCpvInitialize(char*,CthData);
  CthCpvInitialize(jmp_buf,CthLaunching);
  CthCpvInitialize(CthThread,CthCurrent);
  CthCpvInitialize(CthThread,CthExiting);
  CthCpvInitialize(int,CthGrowsdown);
  CthCpvInitialize(int,CthDatasize);
  CthCpvInitialize(arr10,Cth_jb_offsets);
  CthCpvInitialize(int,Cth_jb_count);
  
  CthCpvAccess(CthDatasize)=1;
  CthCpvAccess(CthCurrent)=(CthThread)CmiAlloc(sizeof(struct CthThreadStruct));
  CthCpvAccess(CthCurrent)->fn=0;
  CthCpvAccess(CthCurrent)->arg=0;
  CthCpvAccess(CthCurrent)->data=0;
  CthCpvAccess(CthCurrent)->datasize=0;
  CthCpvAccess(CthCurrent)->awakenfn = 0;
  CthCpvAccess(CthCurrent)->choosefn = 0;
  CthCpvAccess(CthCurrent)->autoyield_enable = 0;
  CthCpvAccess(CthCurrent)->autoyield_blocks = 0;

  /* analyze the activation record. */
  CthInitSub1(bufs, frames, 1);
  CthInitSub2();
  CthCpvAccess(CthGrowsdown) = (frames[0] < frames[1]);
  size = (sizeof(jmpb)/sizeof(size_t));
  delta = frames[0]-frames[1];
  p0 = (size_t *)(bufs+0);
  p1 = (size_t *)(bufs+1);
  CthCpvAccess(Cth_jb_count) = 0;
  for (i=0; i<size; i++) {
    if (CthCpvAccess(Cth_jb_count)==10) goto fail;
    if ((p0[i]-p1[i])==delta) {
      CthCpvAccess(Cth_jb_offsets)[CthCpvAccess(Cth_jb_count)++] = i;
      ((size_t *)(&CthCpvAccess(CthLaunching)))[i] -= (size_t)(frames[1]);
    }
  }
  if (CthCpvAccess(Cth_jb_count) == 0) goto fail;
  return;
fail:
  CmiPrintf("Thread initialization failed.\n");
  exit(1);
}

CthThread CthSelf()
{
  return CthCpvAccess(CthCurrent);
}

static void CthTransfer(t)
CthThread t;
{
  CthCpvAccess(CthCurrent) = t;
  CthCpvAccess(CthData) = t->data;
  longjmp(t->jb, 1);
}

void CthFixData(t)
CthThread t;
{
  if (t->data == 0) {
    t->datasize = CthCpvAccess(CthDatasize);
    t->data = (char *)malloc(CthCpvAccess(CthDatasize));
    return;
  }
  if (t->datasize != CthCpvAccess(CthDatasize)) {
    t->datasize = CthCpvAccess(CthDatasize);
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
  if (t == CthCpvAccess(CthCurrent)) return;
  CthFixData(t);
  if ((setjmp(CthCpvAccess(CthCurrent)->jb))==0)
    CthTransfer(t);
  if (CthCpvAccess(CthExiting))
    { CthFreeNow(CthCpvAccess(CthExiting)); CthCpvAccess(CthExiting)=0; }
}

CthThread CthCreate(fn, arg, size)
CthVoidFn fn; void *arg; int size;
{
  CthThread  result; int i, sp;
  if (size==0) size = STACKSIZE;
  result = (CthThread)CmiAlloc(sizeof(struct CthThreadStruct) + size);
  sp = ((int)(result->stack));
  sp += (CthCpvAccess(CthGrowsdown)) ? (size - SLACK) : SLACK;
  result->fn = fn;
  result->arg = arg;
  result->data = 0;
  result->datasize = 0;
  result->awakenfn = 0;
  result->choosefn = 0;
  result->autoyield_enable = 0;
  result->autoyield_blocks = 0;
  memcpy(&(result->jb), &CthCpvAccess(CthLaunching), sizeof(CthCpvAccess(CthLaunching)));
  for (i=0; i<CthCpvAccess(Cth_jb_count); i++)
    ((size_t *)(&(result->jb)))[CthCpvAccess(Cth_jb_offsets)[i]] += sp;
  return result;
}

void CthFree(t)
CthThread t;
{
  if (t==CthCpvAccess(CthCurrent)) {
    CthCpvAccess(CthExiting) = t;
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
  if (CthCpvAccess(CthCurrent)->choosefn == 0) CthNoStrategy();
  next = CthCpvAccess(CthCurrent)->choosefn();
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

int  CthAutoYielding(CthThread t)
{
  return t->autoyield_enable;
}

void CthAutoYieldBlock()
{
  CthCpvAccess(CthCurrent)->autoyield_blocks++;
}

void CthAutoYieldUnblock()
{
  CthCpvAccess(CthCurrent)->autoyield_blocks--;
}

#endif

 This function gets all outstanding messages out of the network, executing
 their handlers if they are for this lang, else enqueing them in the FIFO 
 queue

int
CmiClearNetworkAndCallHandlers(lang)
int lang;
{
  int retval = 0;
  int *msg, *first ;
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang )  {
        if (CpvAccess(CmiLastBuffer)) CmiFree(CpvAccess(CmiLastBuffer));
        CpvAccess(CmiLastBuffer) = msg;
        (CmiGetHandlerFunction(msg))(msg);
        retval = 1;
      } else {
	FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      }
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  
  while ( (msg = CmiGetNonLocal()) != NULL ) {
    if (CmiGetHandler(msg)==lang) {
      if (CpvAccess(CmiLastBuffer)) CmiFree(CpvAccess(CmiLastBuffer));
      CpvAccess(CmiLastBuffer) = msg;
      (CmiGetHandlerFunction(msg))(msg);
      retval = 1;
    } else {
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
    }
  }
  return retval;
}

 
 Same as above function except that it does not execute any handler functions

int
CmiClearNetwork(lang)
int lang;
{
  int retval = 0;
  int *msg, *first ;
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg ;
    do {
      if ( CmiGetHandler(msg)==lang ) 
	  retval = 1;
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  while ( (msg = CmiGetNonLocal()) != NULL ) {
    if (CmiGetHandler(msg)==lang) 
      retval = 1;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  return retval;
}


#if CMK_THREADS_USE_JB_TWEAKING_ORIGINAL

#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

CthCpvDeclare(char*,CthData);

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


static jmp_buf thread_launching;
CthCpvStatic(CthThread, thread_current);
CthCpvStatic(CthThread, thread_exiting);
CthCpvStatic(int, thread_growsdown);
CthCpvStatic(int, thread_datasize);
static int thread_jb_offsets[10];
static int thread_jb_count;

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

#if CMK_THREADS_USE_JB_TWEAKING_ORIGIN

#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>

#define STACKSIZE (32768)
#define SLACK     256

CthCpvDeclare(char*,CthData);

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


static jmp_buf thread_launching;
CthCpvStatic(CthThread, thread_current);
CthCpvStatic(CthThread, thread_exiting);
CthCpvStatic(int, thread_growsdown);
CthCpvStatic(int, thread_datasize);
static size_t thread_jb_offsets[10];
static int thread_jb_count;

#define ABS(a) (((a) > 0)? (a) : -(a) )

int CthImplemented()
    { return 1; }

static void CthInitSub1(jmpb *bufs, size_t *frames, int n)
{
  double d;
  frames[n] = (size_t)(&d);
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
  size_t frames[2];
  jmpb bufs[2];
  int i, j, size;
  size_t delta, *p0, *p1;

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
  size = (sizeof(jmpb)/sizeof(size_t));
  delta = frames[0]-frames[1];
  p0 = (size_t *)(bufs+0);
  p1 = (size_t *)(bufs+1);
  thread_jb_count = 0;
  for (i=0; i<size; i++) {
    if (thread_jb_count==10) goto fail;
    if ((p0[i]-p1[i])==delta) {
      thread_jb_offsets[thread_jb_count++] = i;
      ((size_t *)(&thread_launching))[i] -= (size_t)(frames[1]);
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
    ((size_t *)(&(result->jb)))[thread_jb_offsets[i]] += sp;
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

