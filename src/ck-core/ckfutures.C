
#include "ckdefs.h"
#include "chare.h"
#include "c++interface.h"
#include "ckfutures.top.h"
#include <stdlib.h>

/******************************************************************************
 *
 * The sequential future abstraction.
 *
 *****************************************************************************/

typedef struct Future_s
{
  int ready;
  void *value;
  CthThread waiters;
  int next; 
} Future;

typedef struct
{
  Future *array;
  int max;
  int freelist;
}
FutureState;

CpvStaticDeclare(FutureState, futurestate);

static void addedFutures(int lo, int hi)
{
  int i;
  FutureState *fs = &(CpvAccess(futurestate));
  Future *array = fs->array;

  for (i=lo; i<hi; i++)
    array[i].next = i+1;
  array[hi-1].next = fs->freelist;
  fs->freelist = lo;
}

static int createFuture()
{
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut; int handle, origsize, newsize, i;

  /* if the freelist is empty, allocate more futures. */
  if (fs->freelist == -1) {
    fs->max = fs->max * 2;
    fs->array = (Future*)realloc(fs->array, sizeof(Future)*(fs->max));
    addedFutures(origsize, newsize);
  }
  handle = fs->freelist;
  fut = fs->array + handle;
  fs->freelist = fut->next;
  fut->ready = 0;
  fut->value = 0;
  fut->waiters = 0;
  fut->next = 0;
  return handle;
}

static void destroyFuture(int handle)
{
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = (fs->array)+handle;
  fut->next = fs->freelist;
  fs->freelist = handle;
}

static void *waitFuture(int handle, int destroy)
{
  CthThread self = CthSelf();
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = (fs->array)+handle;
  void *value;

  if (!(fut->ready)) {
    CthSetNext(self, fut->waiters);
    fut->waiters = self;
    CthSuspend();
  }
  value = fut->value;
  if (destroy) destroyFuture(handle);
  return value;
}

static void setFuture(int handle, void *pointer)
{
  CthThread t;
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = (fs->array)+handle;
  fut->ready = 1;
  fut->value = pointer;
  for (t=fut->waiters; t; t=CthGetNext(t))
    CthAwaken(t);
  fut->waiters = 0;
}

extern "C" void futuresModuleInit()
{
  int i; Future *array;
  CpvInitialize(FutureState, futurestate);
  CpvAccess(futurestate).array = (Future *)malloc(10*sizeof(Future));
  CpvAccess(futurestate).max   = 10;
  CpvAccess(futurestate).freelist = -1;
  addedFutures(0,10);
}


/******************************************************************************
 *
 *
 *****************************************************************************/

int futureBocNum;

class FutureInitMsg : public comm_object
{
  public: int x ;
};

extern "C" void *CRemoteCallBranchFn( int ep, void *m, int group, int processor)
{ 
  void * result;
  ENVELOPE *env = ENVELOPE_UPTR(m);
  int i = createFuture();
  SetRefNumber(m,i);
  SetEnv_pe(env, CmiMyPe());
  GeneralSendMsgBranch(ep, m, processor, -1, group);
  result = waitFuture(i, 1);
  return (result);
}

extern "C" void *CRemoteCallFn(int ep, void *m, ChareIDType *ID)
{ 
  void * result;
  ENVELOPE *env = ENVELOPE_UPTR(m);
  int i = createFuture();
  SetRefNumber(m,i);
  SetEnv_pe(env, CmiMyPe());
  SendMsg(ep, m, ID);
  result = waitFuture(i, 1);
  return (result);
}

class FutureBOC: public groupmember
{
public:

  FutureBOC(FutureInitMsg *m) 
  {
  }
  void SetFuture(FutureInitMsg * m)
  {
    int key;
    key = GetRefNumber(m);
    setFuture( key, m);
  }
};

extern "C" void futuresCreateBOC()
{
  FutureInitMsg *message2 = new (MsgIndex(FutureInitMsg)) FutureInitMsg ;
  futureBocNum = new_group (FutureBOC, FutureInitMsg, message2);
}

extern "C" void CSendToFuture(void *m, int processor)
{
  CSendMsgBranch(FutureBOC, SetFuture, FutureInitMsg, m, futureBocNum, processor);
}


#include "ckfutures.bot.h"
