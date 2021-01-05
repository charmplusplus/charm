/**
\file
\addtogroup CkFutures

To call [sync] entry methods, we need a way to block
the current Converse thread until the called method returns.

A "future" represents a thread of control that has been passed
to another processor.  It provides a place for a (local) thread to
block and the machinery for resuming control based on a remote
event.  Futures are thus used to implement Charm++'s "[sync]" methods.

This "sequential futures abstraction" is a well-studied concept
in remote process control.
*/
/*@{*/
#include "charm++.h"
#include "ck.h"
#include "ckarray.h"
#include "ckfutures.h"
#include <stdlib.h>
#include <limits>
#include <memory>

typedef struct Future_s {
  bool ready;
  void *value;
  std::vector<CthThread> waiters;
  int next;
} Future;

typedef struct {
  std::vector<std::unique_ptr<Future>> array{};
  int max;
  int freelist;
}
FutureState;

class CkSema {
  private:
    CkQ<void*> msgs;
    CkQ<CthThread> waiters;
  public:
    void *wait(void) {
      void *retmsg = msgs.deq();
      if(retmsg==0) {
        waiters.enq(CthSelf());
        CthSuspend();
        retmsg = msgs.deq();
      }
      return retmsg;
    }
    void waitN(int n, void *marray[]) {
      while (1) {
        if(msgs.length()<n) {
          waiters.enq(CthSelf());
          CthSuspend();
          continue;
        }
        for(int i=0;i<n;i++)
          marray[i] = msgs.deq();
        return;
      }
    }
    void signal(void *msg)
    {
      msgs.enq(msg);
      if(!waiters.isEmpty())
        CthAwaken(waiters.deq());
      return;
    }
};

class CkSemaPool {
  private:
    std::vector<CkSema*> pool;
    CkQ<int> freelist;
  public:
    int getNew(void) {
      int idx;
      if(freelist.isEmpty()) {
        idx = pool.size();
        pool.push_back(new CkSema());
      } else {
        idx = freelist.deq();
        pool[idx] = new CkSema();
      }
      return idx;
    }
    void release(int idx) {
      CkSema * sem = pool[idx];
      delete sem;
      freelist.enq(idx);
    }
    void _check(int idx) {
#if CMK_ERROR_CHECKING
      if(pool[idx]==0) {
	      CkAbort("ERROR! operation attempted on invalid semaphore\n");
      }
#endif
    }
    void *wait(int idx) { 
      _check(idx);
      return pool[idx]->wait(); 
    }
    void waitN(int idx, int n, void *marray[]) { 
      _check(idx);
      pool[idx]->waitN(n, marray); 
    }
    void signal(int idx, void *msg) { 
      _check(idx);
      pool[idx]->signal(msg); 
    }
};

CpvStaticDeclare(FutureState, futurestate);
CpvStaticDeclare(CkSemaPool*, semapool);

static void addedFutures(FutureState *fs, int lo, int hi)
{
  for (auto i = lo; i < hi; i++) {
    auto &ptr = fs->array[i];
    if (!ptr) ptr.reset(new Future());
    ptr->next = (i == (hi - 1)) ? fs->freelist : i + 1;
  }

  fs->freelist = lo;
}

static 
inline
int createFuture(void)
{
  FutureState *fs = &(CpvAccess(futurestate));
  int handle, origsize;

  /* if the freelist is empty, allocate more futures. */
  if (fs->freelist == -1) {
    origsize = fs->max;
    fs->max = fs->max * 2;
    fs->array.resize(fs->max);
    addedFutures(fs, origsize, fs->max);
  }
  
  // handle may overflow CMK_REFNUM_TYPE, creating problems when waiting on this future
  CkAssert(fs->freelist <= std::numeric_limits<CMK_REFNUM_TYPE>::max());

  handle = fs->freelist;
  Future* fut = fs->array[handle].get();
  fs->freelist = fut->next;
  fut->ready = false;
  fut->value = 0;
  fut->waiters = {};
  fut->next = 0;
  return handle;
}

CkFuture CkCreateFuture(void)
{
  CkFuture fut;
  fut.id = createFuture();
  fut.pe = CkMyPe();
  return fut;
}

void CkReleaseFutureID(CkFutureID handle)
{
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = fs->array[handle].get();
  fut->next = fs->freelist;
  fs->freelist = handle;
}

int CkProbeFutureID(CkFutureID handle)
{
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = fs->array[handle].get();
  return (int)(fut->ready);
}

void *CkWaitFutureID(CkFutureID handle)
{
  CthThread self = CthSelf();
  FutureState *fs = &(CpvAccess(futurestate));
  Future *fut = fs->array[handle].get();
  void *value;

  if (!(fut->ready)) {
    fut->waiters.push_back(self);
    while (!fut->ready) { CthSuspend(); }
  }

  value = fut->value;
#if CMK_ERROR_CHECKING
  if (value==NULL) 
	CkAbort("ERROR! CkWaitFuture would have to return NULL!\n"
	"This can happen when a thread that calls a sync method "
	"gets a CthAwaken call *before* the sync method returns.");
#endif
  return value;
}

void Future_dropWaiter(Future* future, const CthThread& th) {
  auto& waiting = future->waiters;
  auto it = std::find(waiting.begin(), waiting.end(), th);
  if (it != waiting.end()) {
    waiting.erase(it);
  }
}

std::pair<void*, CkFutureID> CkWaitAnyID(const std::vector<CkFutureID>& handles) {
  auto self = CthSelf();
  auto fs = &(CpvAccess(futurestate));
  const int n = handles.size();
  std::vector<Future*> futures;
  for (auto handle : handles) {
    auto future = fs->array[handle].get();
    futures.push_back(future);
    if (!future->ready) {
      future->waiters.push_back(self);
    }
  }
  do {
    for (int i = 0; i < n; i++) {
      // when a future is ready
      if (futures[i]->ready) {
        // drop ourself from all the futures
        for (auto future: futures) Future_dropWaiter(future, self);
        // then return the value/handle pair
        return std::make_pair(futures[i]->value, handles[i]);
      }
    }
    // suspend until a future becomes ready...
    CthSuspend();
  } while (true);
}

std::vector<void*> CkWaitAllIDs(const std::vector<CkFutureID>& ids) {
  // a more sophisticated solution should be implemented that interleaves
  // waiting on multiple futures... but this works in the meantime
  std::vector<void*> results;
  for (auto id: ids) {
    results.push_back(CkWaitFutureID(id));
  }
  return results;
}

void CkReleaseFuture(CkFuture fut)
{
  CkReleaseFutureID(fut.id);
}

int CkProbeFuture(CkFuture fut)
{
  return CkProbeFutureID(fut.id);
}

void *CkWaitFuture(CkFuture fut)
{
  return CkWaitFutureID(fut.id);
}

void CkWaitVoidFuture(CkFutureID handle)
{
  CkFreeMsg(CkWaitFutureID(handle));
}

static void setFuture(CkFutureID handle, void *pointer)
{
  CthThread t;
  FutureState *fs = &(CpvAccess(futurestate));
  Future* fut = fs->array[handle].get();
  fut->ready = true;
#if CMK_ERROR_CHECKING
  if (pointer==NULL) CkAbort("setFuture called with NULL!");
#endif
  fut->value = pointer;
  for (auto t: fut->waiters) {
    CthAwaken(t);
  }
  fut->waiters = {};
}

void _futuresModuleInit(void)
{
  CpvInitialize(FutureState, futurestate);
  CpvInitialize(CkSemaPool *, semapool);
  auto fs = &(CpvAccess(futurestate));
  fs->array.resize(10);
  fs->max = 10;
  fs->freelist = -1;
  addedFutures(fs, 0, 10);
  CpvAccess(semapool) = new CkSemaPool();
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


CkFutureID CkCreateAttachedFuture(void *msg)
{
  CkFutureID ret=createFuture();
  UsrToEnv(msg)->setRef(ret);
  return ret;
}

CkFutureID CkCreateAttachedFutureSend(void *msg, int ep,
CkArrayID id, CkArrayIndex idx,
void(*fptr)(CkArrayID,CkArrayIndex,void*,int,int),int size)
{
CkFutureID ret=createFuture();
UsrToEnv(msg)->setRef(ret);
#if IGET_FLOWCONTROL
if (TheIGetControlClass.iget_request(ret,msg,ep,id,idx,fptr,size))
#endif
(fptr)(id,idx,msg,ep,0);
return ret;
}


/*
CkFutureID CkCreateAttachedFutureSend(void *msg, int ep, void *obj,void(*fptr)(void*,void*,int,int))
{
  CkFutureID ret=createFuture();
  UsrToEnv(msg)->setRef(ret);
#if IGET_FLOWCONTROL
  if (TheIGetControlClass.iget_request(ret,msg,ep,obj,fptr)) 
#endif
  (fptr)(obj,msg,ep,0);
  return ret;
}
*/
void *CkWaitReleaseFuture(CkFutureID futNum)
{
#if IGET_FLOWCONTROL
  TheIGetControlClass.iget_resend(futNum);
#endif
  void *result=CkWaitFutureID(futNum);
  CkReleaseFutureID(futNum);
#if IGET_FLOWCONTROL
  TheIGetControlClass.iget_free(1);
//  TheIGetControlClass.iget_free(sizeof(result));
#endif
  return result;
}

class FutureBOC: public IrrGroup {
public:
  FutureBOC(void){ }
  FutureBOC(FutureInitMsg *m) { delete m; }
  FutureBOC(CkMigrateMessage *m) { }
  void SetFuture(FutureInitMsg * m) { 
#if CMK_ERROR_CHECKING
    if (m==NULL) CkAbort("FutureBOC::SetFuture called with NULL!");
#endif
    int key;
    key = UsrToEnv((void *)m)->getRef();
    setFuture( key, m);
  }
  void SetSema(FutureInitMsg *m) {
#if CMK_ERROR_CHECKING
    if (m==NULL) CkAbort("FutureBOC::SetSema called with NULL!");
#endif
    int idx;
    idx = UsrToEnv((void *)m)->getRef();
    CpvAccess(semapool)->signal(idx,(void*)m);
  }
};

extern "C" 
void CkSendToFutureID(CkFutureID futNum, void *m, int PE)
{
  UsrToEnv(m)->setRef(futNum);
  CProxy_FutureBOC fBOC(_fbocID);
  fBOC[PE].SetFuture((FutureInitMsg *)m);
}

void  CkSendToFuture(CkFuture fut, void *msg)
{
  CkSendToFutureID(fut.id, msg, fut.pe);
}

CkSemaID CkSemaCreate(void)
{
  CkSemaID id;
  id.pe = CkMyPe();
  id.idx = CpvAccess(semapool)->getNew();
  return id;
}

void *CkSemaWait(CkSemaID id)
{
#if CMK_ERROR_CHECKING
  if(id.pe != CkMyPe()) {
    CkAbort("ERROR: Waiting on nonlocal semaphore! Aborting..\n");
  }
#endif
  return CpvAccess(semapool)->wait(id.idx);
}

void CkSemaWaitN(CkSemaID id, int n, void *marray[])
{
#if CMK_ERROR_CHECKING
  if(id.pe != CkMyPe()) {
    CkAbort("ERROR: Waiting on nonlocal semaphore! Aborting..\n");
  }
#endif
  CpvAccess(semapool)->waitN(id.idx, n, marray);
}

void CkSemaSignal(CkSemaID id, void *m)
{
  UsrToEnv(m)->setRef(id.idx);
  CProxy_FutureBOC fBOC(_fbocID);
  fBOC[id.pe].SetSema((FutureInitMsg *)m);
}

void CkSemaDestroy(CkSemaID id)
{
#if CMK_ERROR_CHECKING
  if(id.pe != CkMyPe()) {
    CkAbort("ERROR: destroying a nonlocal semaphore! Aborting..\n");
  }
#endif
  CpvAccess(semapool)->release(id.idx);
}


/*@}*/
#include "CkFutures.def.h"

