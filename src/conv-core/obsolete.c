
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

