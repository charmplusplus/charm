/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "charm++.h"
#include "ck.h"
#include "ckfutures.h"
#include <stdlib.h>

/******************************************************************************
 *
 * The sequential future abstraction:
 *  A "future" represents a thread of control that has been passed
 * to another processor.  It provides a place for a (local) thread to
 * block and the machinery for resuming control.  Futures are used to
 * implement Charm++'s "[sync]" methods.
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
    _MEMCHECK(fs->array);
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
  fut = (fs->array)+handle;
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
  _MEMCHECK(CpvAccess(futurestate).array);
  CpvAccess(futurestate).max   = 10;
  CpvAccess(futurestate).freelist = -1;
  addedFutures(0,10);
}

CkGroupID _fbocID;

class FutureInitMsg : public CMessage_FutureInitMsg {
  public: int x ;
};

class  FutureMain : public Chare {
  public:
    FutureMain(CkArgMsg *m) {
      _fbocID = CProxy_FutureBOC::ckNew(new FutureInitMsg);
      delete m;
    }
    FutureMain(CkMigrateMessage *m) {}
};

extern "C" 
CkFutureID CkRemoteBranchCallAsync(int ep, void *m, CkGroupID group, int PE)
{ 
  CkFutureID ret=CkCreateAttachedFuture(m);
  CkSendMsgBranch(ep, m, PE, group);
  return ret;
}

extern "C" 
void *CkRemoteBranchCall(int ep, void *m, CkGroupID group, int PE)
{ 
  CkFutureID i = CkRemoteBranchCallAsync(ep, m, group, PE);  
  return CkWaitReleaseFuture(i);
}

extern "C" 
CkFutureID CkRemoteNodeBranchCallAsync(int ep, void *m, CkGroupID group, int node)
{ 
  CkFutureID ret=CkCreateAttachedFuture(m);
  CkSendMsgNodeBranch(ep, m, node, group);
  return ret;
}

extern "C" 
void *CkRemoteNodeBranchCall(int ep, void *m, CkGroupID group, int node)
{ 
  CkFutureID i = CkRemoteNodeBranchCallAsync(ep, m, group, node);
  return CkWaitReleaseFuture(i);
}

extern "C" 
CkFutureID CkRemoteCallAsync(int ep, void *m, const CkChareID *ID)
{ 
  CkFutureID ret=CkCreateAttachedFuture(m);
  CkSendMsg(ep, m, ID);
  return ret;
}

extern "C" 
void *CkRemoteCall(int ep, void *m, const CkChareID *ID)
{ 
  CkFutureID i = CkRemoteCallAsync(ep, m, ID);
  return CkWaitReleaseFuture(i);
}

extern "C" CkFutureID CkCreateAttachedFuture(void *msg)
{
	CkFutureID ret=createFuture();
	UsrToEnv(msg)->setRef(ret);
	return ret;
}
extern "C" void *CkWaitReleaseFuture(CkFutureID futNum)
{
	void *result=CkWaitFuture(futNum);
	CkReleaseFuture(futNum);
	return result;
}

class FutureBOC: public Group {
public:
  FutureBOC(FutureInitMsg *m) { delete m; }
  FutureBOC(CkMigrateMessage *m) {}
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
  CProxy_FutureBOC fBOC(_fbocID);
  fBOC[PE].SetFuture((FutureInitMsg *)m);
}

#include "CkFutures.def.h"
