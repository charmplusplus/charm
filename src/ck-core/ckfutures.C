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

static int createFuture(void)
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

extern "C"
void CkReleaseFuture(CkFutureID handle)
{
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = (fs->array)+handle;
  fut->next = fs->freelist;
  fs->freelist = handle;
}

extern "C"
int CkProbeFuture(CkFutureID handle)
{
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = (fs->array)+handle;
  return (fut->ready);
}

extern "C"
void *CkWaitFuture(CkFutureID handle)
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
  return value;
}

extern "C"
void CkWaitVoidFuture(CkFutureID handle)
{
  CkFreeMsg(CkWaitFuture(handle));
}

static void setFuture(CkFutureID handle, void *pointer)
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

CProxy_FutureBOC fBOC(0);

class FutureInitMsg : public CMessage_FutureInitMsg {
  public: int x ;
};

class  FutureMain : public Chare {
  public:
    FutureMain(CkArgMsg *m) {
      fBOC.ckSetGroupId(CProxy_FutureBOC::ckNew(new FutureInitMsg));
      delete m;
    }
};

extern "C" 
CkFutureID CkRemoteBranchCallAsync(int ep, void *m, CkGroupID group, int PE)
{ 
  envelope *env = UsrToEnv(m);
  int i = createFuture();
  env->setRef(i);
  CkSendMsgBranch(ep, m, PE, group);
  return i;
}

extern "C" 
void *CkRemoteBranchCall(int ep, void *m, CkGroupID group, int PE)
{ 
  void * result;
  int i = CkRemoteBranchCallAsync(ep, m, group, PE);
  result = CkWaitFuture(i);
  CkReleaseFuture(i);
  return (result);
}

extern "C" 
CkFutureID CkRemoteNodeBranchCallAsync(int ep, void *m, CkGroupID group, int node)
{ 
  envelope *env = UsrToEnv(m);
  int i = createFuture();
  env->setRef(i);
  CkSendMsgNodeBranch(ep, m, node, group);
  return i;
}

extern "C" 
void *CkRemoteNodeBranchCall(int ep, void *m, CkGroupID group, int node)
{ 
  void * result;
  int i = CkRemoteNodeBranchCallAsync(ep, m, group, node);
  result = CkWaitFuture(i);
  CkReleaseFuture(i);
  return (result);
}

extern "C" 
CkFutureID CkRemoteCallAsync(int ep, void *m, CkChareID *ID)
{ 
  envelope *env = UsrToEnv(m);
  int i = createFuture();
  env->setRef(i);
  CkSendMsg(ep, m, ID);
  return i;
}

extern "C" 
void *CkRemoteCall(int ep, void *m, CkChareID *ID)
{ 
  void * result;
  int i = CkRemoteCallAsync(ep, m, ID);
  result = CkWaitFuture(i);
  CkReleaseFuture(i);
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
void CkSendToFuture(CkFutureID futNum, void *m, int PE)
{
  UsrToEnv(m)->setRef(futNum);
  fBOC.SetFuture((FutureInitMsg *)m,PE);
}

#include "CkFutures.def.h"
