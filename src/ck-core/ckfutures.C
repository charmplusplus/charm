#include "charm++.h"
#include "ck.h"
#include "ckfutures.h"
#include <stdlib.h>

/******************************************************************************
 *
 * The sequential future abstraction.
 *
 *****************************************************************************/

typedef struct Future_s {
  int ready;
  void *value;
  CthThread waiters;
  int next; 
} Future;

typedef struct {
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
  Future *fut; int handle, origsize;

  /* if the freelist is empty, allocate more futures. */
  if (fs->freelist == -1) {
    origsize = fs->max;
    fs->max = fs->max * 2;
    fs->array = (Future*)realloc(fs->array, sizeof(Future)*(fs->max));
    addedFutures(origsize, fs->max);
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

void _futuresModuleInit(void)
{
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

CProxy_FutureBOC fBOC(0);

class FutureInitMsg : public CMessage_FutureInitMsg {
  public: int x ;
};

class  FutureMain : public Chare {
  public:
    FutureMain(CkArgMsg *m) {
      fBOC.ckSetGroupId(CProxy_FutureBOC::ckNew(new FutureInitMsg));
      CkFreeMsg(m);
    }
};

extern "C" 
void *CkRemoteBranchCall(int ep, void *m, int group, int PE)
{ 
  void * result;
  envelope *env = UsrToEnv(m);
  int i = createFuture();
  env->setRef(i);
  CkSendMsgBranch(ep, m, PE, group);
  result = waitFuture(i, 1);
  return (result);
}

extern "C" 
void *CkRemoteNodeBranchCall(int ep, void *m, int group, int node)
{ 
  void * result;
  envelope *env = UsrToEnv(m);
  int i = createFuture();
  env->setRef(i);
  CkSendMsgNodeBranch(ep, m, node, group);
  result = waitFuture(i, 1);
  return (result);
}

extern "C" 
void *CkRemoteCall(int ep, void *m, CkChareID *ID)
{ 
  void * result;
  envelope *env = UsrToEnv(m);
  int i = createFuture();
  env->setRef(i);
  CkSendMsg(ep, m, ID);
  result = waitFuture(i, 1);
  return (result);
}

class FutureBOC: public Group {
public:
  FutureBOC(FutureInitMsg *m) { delete m; }
  void SetFuture(FutureInitMsg * m) { 
    int key;
    key = UsrToEnv((void *)m)->getRef();
    setFuture( key, m);
  }
};

extern "C" 
void CkSendToFuture(int futNum, void *m, int PE)
{
  UsrToEnv(m)->setRef(futNum);
  fBOC.SetFuture((FutureInitMsg *)m,PE);
}

#include "CkFutures.def.h"
