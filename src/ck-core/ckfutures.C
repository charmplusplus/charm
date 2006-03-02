/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
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
    CkVec<CkSema*> pool;
    CkQ<int> freelist;
  public:
    int getNew(void) {
      CkSema *sem = new CkSema();
      int idx;
      if(freelist.isEmpty()) {
        idx = pool.length();
        pool.insertAtEnd(sem);
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
#ifndef CMK_OPTIMIZE
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
#ifndef CMK_OPTIMIZE
  if (value==NULL) 
	CkAbort("ERROR! CkWaitFuture would have to return NULL!\n"
	"This can happen when a thread that calls a sync method "
	"gets a CthAwaken call *before* the sync method returns.");
#endif
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
#ifndef CMK_OPTIMIZE
  if (pointer==NULL) CkAbort("setFuture called with NULL!");
#endif
  fut->value = pointer;
  for (t=fut->waiters; t; t=CthGetNext(t))
    CthAwaken(t);
  fut->waiters = 0;
}

void _futuresModuleInit(void)
{
  CpvInitialize(FutureState, futurestate);
  CpvInitialize(CkSemaPool *, semapool);
  CpvAccess(futurestate).array = (Future *)malloc(10*sizeof(Future));
  _MEMCHECK(CpvAccess(futurestate).array);
  CpvAccess(futurestate).max   = 10;
  CpvAccess(futurestate).freelist = -1;
  addedFutures(0,10);
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

#ifndef _IGET_TOKENQUEUE_
#define _IGET_TOKENQUEUE_ 0
#endif

#if _IGET_TOKENQUEUE_
#ifndef _IGET_TOKEN_ 
#define _IGET_TOKEN_ 100 
#endif
#endif

#if _IGET_TOKENQUEUE_
static int iget_token = _IGET_TOKEN_;
typedef struct iget_token_struct {
  CkFutureID futNum;
  void *m;
  int ep;
  void *obj;     
  void(*fptr)(void*,void*,int,int);
  struct iget_token_struct *next;
} *iget_tokenqueue_entry;

static iget_tokenqueue_entry iget_tokenqueue_head=NULL;
static iget_tokenqueue_entry iget_tokenqueue_end=NULL;

inline void iget_tokenqueue_enqueue(int futNum,void* m,int ep,void *obj,void(*fptr)(void*,void*,int,int)){
  iget_tokenqueue_entry e=new struct iget_token_struct(); 
  e->futNum=futNum; e->m=m; e->ep=ep; e->obj=obj; e->fptr=fptr; e->next=NULL;
  if(iget_tokenqueue_end==NULL){
    iget_tokenqueue_end=iget_tokenqueue_head=e;
  } 
  else{
    iget_tokenqueue_end->next=e;
    iget_tokenqueue_end=e;	   
  }
} 

inline iget_tokenqueue_entry iget_tokenqueue_dequeue() {
  if(iget_tokenqueue_head==NULL) 
    return NULL;
  iget_tokenqueue_entry e = iget_tokenqueue_head;
  iget_tokenqueue_head = iget_tokenqueue_head->next;
  if(iget_tokenqueue_head==NULL) 
    iget_tokenqueue_end=NULL;
  return e;
}
#endif

extern "C" CkFutureID CkCreateAttachedFuture(void *msg)
{
  CkFutureID ret=createFuture();
  UsrToEnv(msg)->setRef(ret);
  return ret;
}

extern "C" CkFutureID CkCreateAttachedFutureStatus(void *msg, int ep, void *obj,void(*fptr)(void*,void*,int,int), int* status)
{
	CkFutureID ret=createFuture();
	UsrToEnv(msg)->setRef(ret);
	*status = 1; 
#if _IGET_TOKENQUEUE_
	if(iget_token>0){
		iget_token--;
	} else {
		iget_tokenqueue_enqueue(ret,msg,ep,obj,fptr);
		*status = 0; // No send will be done this case
	}
#endif
	return ret;
}

extern "C" void *CkWaitReleaseFuture(CkFutureID futNum)
{
	void *result=CkWaitFuture(futNum);
	CkReleaseFuture(futNum);
#if _IGET_TOKENQUEUE_
	iget_tokenqueue_entry e=iget_tokenqueue_dequeue();
	if(e==NULL) 
	  iget_token++;
	else{
	  (*e->fptr)(e->obj,e->m, e->ep, 0);
	  delete e;
	}
#endif
	return result;
}

class FutureBOC: public IrrGroup {
public:
  FutureBOC(void){ }
  FutureBOC(FutureInitMsg *m) { delete m; }
  FutureBOC(CkMigrateMessage *m) { }
  void SetFuture(FutureInitMsg * m) { 
#ifndef CMK_OPTIMIZE
    if (m==NULL) CkAbort("FutureBOC::SetFuture called with NULL!");
#endif
    int key;
    key = UsrToEnv((void *)m)->getRef();
    setFuture( key, m);
  }
  void SetSema(FutureInitMsg *m) {
#ifndef CMK_OPTIMIZE
    if (m==NULL) CkAbort("FutureBOC::SetSema called with NULL!");
#endif
    int idx;
    idx = UsrToEnv((void *)m)->getRef();
    CpvAccess(semapool)->signal(idx,(void*)m);
  }
};

extern "C" 
void CkSendToFuture(CkFutureID futNum, void *m, int PE)
{
  UsrToEnv(m)->setRef(futNum);
  CProxy_FutureBOC fBOC(_fbocID);
  fBOC[PE].SetFuture((FutureInitMsg *)m);
}

extern "C"
CkSemaID CkSemaCreate(void)
{
  CkSemaID id;
  id.pe = CkMyPe();
  id.idx = CpvAccess(semapool)->getNew();
  return id;
}

extern "C"
void *CkSemaWait(CkSemaID id)
{
#ifndef CMK_OPTIMIZE
  if(id.pe != CkMyPe()) {
    CkAbort("ERROR: Waiting on nonlocal semaphore! Aborting..\n");
  }
#endif
  return CpvAccess(semapool)->wait(id.idx);
}

extern "C"
void CkSemaWaitN(CkSemaID id, int n, void *marray[])
{
#ifndef CMK_OPTIMIZE
  if(id.pe != CkMyPe()) {
    CkAbort("ERROR: Waiting on nonlocal semaphore! Aborting..\n");
  }
#endif
  CpvAccess(semapool)->waitN(id.idx, n, marray);
}

extern "C"
void CkSemaSignal(CkSemaID id, void *m)
{
  UsrToEnv(m)->setRef(id.idx);
  CProxy_FutureBOC fBOC(_fbocID);
  fBOC[id.pe].SetSema((FutureInitMsg *)m);
}

extern "C"
void CkSemaDestroy(CkSemaID id)
{
#ifndef CMK_OPTIMIZE
  if(id.pe != CkMyPe()) {
    CkAbort("ERROR: destroying a nonlocal semaphore! Aborting..\n");
  }
#endif
  CpvAccess(semapool)->release(id.idx);
}

/*@}*/
#include "CkFutures.def.h"

