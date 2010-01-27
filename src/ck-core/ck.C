/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
/**
\addtogroup Ck

These routines implement a basic remote-method-invocation system
consisting of chares and groups.  There is no migration. All
the bindings are written to the C language, although most
clients, including the rest of Charm++, are actually C++.
*/
#include "ck.h"
#include "trace.h"
#include "queueing.h"

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
#include "pathHistory.h"
void automaticallySetMessagePriority(envelope *env); // in control point framework.
#endif

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif // CMK_LBDB_ON

#ifndef CMK_CHARE_USE_PTR
CpvDeclare(CkVec<void *>, chare_objs);
CpvDeclare(CkVec<int>, chare_types);
CpvDeclare(CkVec<VidBlock *>, vidblocks);
#endif

#define CK_MSG_SKIP_OR_IMM    (CK_MSG_EXPEDITED | CK_MSG_IMMEDIATE)

VidBlock::VidBlock() { state = UNFILLED; msgQ = new PtrQ(); _MEMCHECK(msgQ); }

int CMessage_CkMessage::__idx=-1;
int CMessage_CkArgMsg::__idx=0;
int CkIndex_Chare::__idx;
int CkIndex_Group::__idx;
int CkIndex_ArrayBase::__idx=-1;

extern int _defaultObjectQ;

//Charm++ virtual functions: declaring these here results in a smaller executable
Chare::Chare(void) {
  thishandle.onPE=CkMyPe();
  thishandle.objPtr=this;
#ifndef CMK_CHARE_USE_PTR
     // for plain chare, objPtr is actually the index to chare obj table
  if (chareIdx >= 0) thishandle.objPtr=(void*)chareIdx;
#endif
#ifdef _FAULT_MLOG_
  mlogData = new ChareMlogData();
  mlogData->objID.type = TypeChare;
  mlogData->objID.data.chare.id = thishandle;
#endif
#if CMK_OBJECT_QUEUE_AVAILABLE
  if (_defaultObjectQ)  CkEnableObjQ();
#endif
}

Chare::Chare(CkMigrateMessage* m) {
  thishandle.onPE=CkMyPe();
  thishandle.objPtr=this;

#if CMK_OBJECT_QUEUE_AVAILABLE
  if (_defaultObjectQ)  CkEnableObjQ();
#endif
}

void Chare::CkEnableObjQ()
{
#if CMK_OBJECT_QUEUE_AVAILABLE
  objQ.create();
#endif
}

Chare::~Chare() {}

void Chare::pup(PUP::er &p)
{
  p(thishandle.onPE);
  thishandle.objPtr=(void *)this;
#ifndef CMK_CHARE_USE_PTR
  p(chareIdx);
  if (chareIdx != -1) thishandle.objPtr=(void*)chareIdx;
#endif
#ifdef _FAULT_MLOG_
  if(p.isUnpacking()){
    mlogData = new ChareMlogData();
  }
  mlogData->pup(p);
#endif
}

int Chare::ckGetChareType() const {
  return -3;
}
char *Chare::ckDebugChareName(void) {
  char buf[100];
  sprintf(buf,"Chare on pe %d at %p",CkMyPe(),this);
  return strdup(buf);
}
int Chare::ckDebugChareID(char *str, int limit) {
  // pure chares for now do not have a valid ID
  str[0] = 0;
  return 1;
}
void Chare::ckDebugPup(PUP::er &p) {
  pup(p);
}

/// This method is called before starting a [threaded] entry method.
void Chare::CkAddThreadListeners(CthThread th, void *msg) {
  CthSetThreadID(th, thishandle.onPE, (int)(((char *)thishandle.objPtr)-(char *)0), 0);
  traceAddThreadListeners(th, UsrToEnv(msg));
}


void CkMessage::ckDebugPup(PUP::er &p,void *msg) {
  p.comment("Bytes");
  int ts=UsrToEnv(msg)->getTotalsize();
  int msgLen=ts-sizeof(envelope);
  if (msgLen>0)
    p((char*)msg,msgLen);
}

IrrGroup::IrrGroup(void) {
  thisgroup = CkpvAccess(_currentGroup);
#ifdef _FAULT_MLOG_
        mlogData->objID.type = TypeGroup;
        mlogData->objID.data.group.id = thisgroup;
        mlogData->objID.data.group.onPE = CkMyPe();
#endif
}

IrrGroup::~IrrGroup() {
  // remove the object pointer
  CmiImmediateLock(CkpvAccess(_groupTableImmLock));
  CkpvAccess(_groupTable)->find(thisgroup).setObj(NULL);
  CmiImmediateUnlock(CkpvAccess(_groupTableImmLock));
}

void IrrGroup::pup(PUP::er &p)
{
  Chare::pup(p);
  p|thisgroup;
}

int IrrGroup::ckGetChareType() const {
  return CkpvAccess(_groupTable)->find(thisgroup).getcIdx();
}

int IrrGroup::ckDebugChareID(char *str, int limit) {
  if (limit<5) return -1;
  str[0] = 1;
  *((int*)&str[1]) = thisgroup.idx;
  return 5;
}

char *IrrGroup::ckDebugChareName() {
  return strdup(_chareTable[ckGetChareType()]->name);
}

void IrrGroup::ckJustMigrated(void)
{
}

void IrrGroup::CkAddThreadListeners(CthThread tid, void *msg) {
  /* FIXME: **CW** not entirely sure what we should do here yet */
}

void Group::CkAddThreadListeners(CthThread th, void *msg) {
  Chare::CkAddThreadListeners(th, msg);
  CthSetThreadID(th, thisgroup.idx, 0, 0);
}

void Group::pup(PUP::er &p)
{
  CkReductionMgr::pup(p);
  reductionInfo.pup(p);
}

/**** Delegation Manager Group */
CkDelegateMgr::~CkDelegateMgr() { }

//Default delegator implementation: do not delegate-- send directly
void CkDelegateMgr::ChareSend(CkDelegateData *pd,int ep,void *m,const CkChareID *c,int onPE)
  { CkSendMsg(ep,m,c); }
void CkDelegateMgr::GroupSend(CkDelegateData *pd,int ep,void *m,int onPE,CkGroupID g)
  { CkSendMsgBranch(ep,m,onPE,g); }
void CkDelegateMgr::GroupBroadcast(CkDelegateData *pd,int ep,void *m,CkGroupID g)
  { CkBroadcastMsgBranch(ep,m,g); }
void CkDelegateMgr::GroupSectionSend(CkDelegateData *pd,int ep,void *m,int nsid,CkSectionID *s)
  { CkSendMsgBranchMulti(ep,m,s->_cookie.aid,s->npes,s->pelist); }
void CkDelegateMgr::NodeGroupSend(CkDelegateData *pd,int ep,void *m,int onNode,CkNodeGroupID g)
  { CkSendMsgNodeBranch(ep,m,onNode,g); }
void CkDelegateMgr::NodeGroupBroadcast(CkDelegateData *pd,int ep,void *m,CkNodeGroupID g)
  { CkBroadcastMsgNodeBranch(ep,m,g); }
void CkDelegateMgr::NodeGroupSectionSend(CkDelegateData *pd,int ep,void *m,int nsid,CkSectionID *s)
  { CkSendMsgNodeBranchMulti(ep,m,s->_cookie.aid,s->npes,s->pelist); }
void CkDelegateMgr::ArrayCreate(CkDelegateData *pd,int ep,void *m,const CkArrayIndexMax &idx,int onPE,CkArrayID a)
{
	CProxyElement_ArrayBase ap(a,idx);
	ap.ckInsert((CkArrayMessage *)m,ep,onPE);
}
void CkDelegateMgr::ArraySend(CkDelegateData *pd,int ep,void *m,const CkArrayIndexMax &idx,CkArrayID a)
{
	CProxyElement_ArrayBase ap(a,idx);
	ap.ckSend((CkArrayMessage *)m,ep);
}
void CkDelegateMgr::ArrayBroadcast(CkDelegateData *pd,int ep,void *m,CkArrayID a)
{
	CProxy_ArrayBase ap(a);
	ap.ckBroadcast((CkArrayMessage *)m,ep);
}

void CkDelegateMgr::ArraySectionSend(CkDelegateData *pd,int ep,void *m, int nsid,CkSectionID *s, int opts)
{
	CmiAbort("ArraySectionSend is not implemented!\n");
/*
	CProxyElement_ArrayBase ap(a,idx);
	ap.ckSend((CkArrayMessage *)m,ep);
*/
}

/*** Proxy <-> delegator communication */
CkDelegateData::~CkDelegateData() {}

CkDelegateData *CkDelegateMgr::DelegatePointerPup(PUP::er &p,CkDelegateData *pd) {
  return pd; // default implementation ignores pup call
}

/** FIXME: make a "CkReferenceHandle<CkDelegateData>" class to avoid
   this tricky manual reference counting business... */

void CProxy::ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr) {
 	if (dPtr) dPtr->ref();
	ckUndelegate();
	delegatedMgr = dTo;
	delegatedPtr = dPtr;
}
void CProxy::ckUndelegate(void) {
	delegatedMgr=NULL;
	if (delegatedPtr) delegatedPtr->unref();
	delegatedPtr=NULL;
}

/// Copy constructor
CProxy::CProxy(const CProxy &src)
    :delegatedMgr(src.delegatedMgr)
{
    delegatedPtr = NULL;
    if(delegatedMgr != NULL && src.delegatedPtr != NULL)
        delegatedPtr = src.delegatedMgr->ckCopyDelegateData(src.delegatedPtr);
}

/// Assignment operator
CProxy& CProxy::operator=(const CProxy &src) {
	CkDelegateData *oldPtr=delegatedPtr;
	ckUndelegate();
	delegatedMgr=src.delegatedMgr;

        if(delegatedMgr != NULL && src.delegatedPtr != NULL)
            delegatedPtr = delegatedMgr->ckCopyDelegateData(src.delegatedPtr);
        else
            delegatedPtr = NULL;

        // subtle: do unref *after* ref, because it's possible oldPtr == delegatedPtr
	if (oldPtr) oldPtr->unref();
	return *this;
}

void CProxy::pup(PUP::er &p) {
      CkGroupID delegatedTo;
      delegatedTo.setZero();
      int isNodeGroup = 0;
      if (!p.isUnpacking()) {
        if (delegatedMgr) {
          delegatedTo = delegatedMgr->CkGetGroupID();
 	  isNodeGroup = delegatedMgr->isNodeGroup();
        }
      }
      p|delegatedTo;
      if (!delegatedTo.isZero()) {
        p|isNodeGroup;
        if (p.isUnpacking()) {
	  if (isNodeGroup)
		delegatedMgr=(CkDelegateMgr *)CkLocalNodeBranch(delegatedTo);
	  else
		delegatedMgr=(CkDelegateMgr *)CkLocalBranch(delegatedTo);
	}

        delegatedPtr = delegatedMgr->DelegatePointerPup(p,delegatedPtr);
	if (p.isUnpacking() && delegatedPtr)
            delegatedPtr->ref();
      }
}

/**** Array sections */
#define CKSECTIONID_CONSTRUCTOR_DEF(index) \
CkSectionID::CkSectionID(const CkArrayID &aid, const CkArrayIndex##index *elems, const int nElems): _nElems(nElems) { \
  _cookie.aid = aid;	\
  _cookie.get_pe() = CkMyPe();	\
  _elems = new CkArrayIndexMax[nElems];	\
  for (int i=0; i<nElems; i++) _elems[i] = elems[i];	\
  pelist = NULL;	\
  npes  = 0;	\
}

CKSECTIONID_CONSTRUCTOR_DEF(1D)
CKSECTIONID_CONSTRUCTOR_DEF(2D)
CKSECTIONID_CONSTRUCTOR_DEF(3D)
CKSECTIONID_CONSTRUCTOR_DEF(4D)
CKSECTIONID_CONSTRUCTOR_DEF(5D)
CKSECTIONID_CONSTRUCTOR_DEF(6D)
CKSECTIONID_CONSTRUCTOR_DEF(Max)

CkSectionID::CkSectionID(const CkGroupID &gid, const int *_pelist, const int _npes): _nElems(0), _elems(NULL), npes(_npes) {
  pelist = new int[npes];
  for (int i=0; i<npes; i++) pelist[i] = _pelist[i];
  _cookie.aid = gid;
}

CkSectionID::CkSectionID(const CkSectionID &sid) {
  int i;
  _cookie = sid._cookie;
  _nElems = sid._nElems;
  if (_nElems > 0) {
    _elems = new CkArrayIndexMax[_nElems];
    for (i=0; i<_nElems; i++) _elems[i] = sid._elems[i];
  } else _elems = NULL;
  npes = sid.npes;
  if (npes > 0) {
    pelist = new int[npes];
    for (i=0; i<npes; ++i) pelist[i] = sid.pelist[i];
  } else pelist = NULL;
}

void CkSectionID::operator=(const CkSectionID &sid) {
  int i;
  _cookie = sid._cookie;
  _nElems = sid._nElems;
  if (_nElems > 0) {
    _elems = new CkArrayIndexMax[_nElems];
    for (i=0; i<_nElems; i++) _elems[i] = sid._elems[i];
  } else _elems = NULL;
  npes = sid.npes;
  if (npes > 0) {
    pelist = new int[npes];
    for (i=0; i<npes; ++i) pelist[i] = sid.pelist[i];
  } else pelist = NULL;
}

void CkSectionID::pup(PUP::er &p) {
    p | _cookie;
    p(_nElems);
    if (_nElems > 0) {
      if (p.isUnpacking()) _elems = new CkArrayIndexMax[_nElems];
      for (int i=0; i< _nElems; i++) p | _elems[i];
      npes = 0;
      pelist = NULL;
    } else {
      // If _nElems is zero, than this section describes processors instead of array elements
      _elems = NULL;
      p(npes);
      if (p.isUnpacking()) pelist = new int[npes];
      p(pelist, npes);
    }
}

/**** Tiny random API routines */

#ifdef CMK_CUDA
void CUDACallbackManager(void *fn) {
  if (fn != NULL) {
    CkCallback *cb = (CkCallback*) fn;
    cb->send();
  }
}

#endif

extern "C"
void CkSetRefNum(void *msg, int ref)
{
  UsrToEnv(msg)->setRef(ref);
}

extern "C"
int CkGetRefNum(void *msg)
{
  return UsrToEnv(msg)->getRef();
}

extern "C"
int CkGetSrcPe(void *msg)
{
  return UsrToEnv(msg)->getSrcPe();
}

extern "C"
int CkGetSrcNode(void *msg)
{
  return CmiNodeOf(CkGetSrcPe(msg));
}

extern "C"
void *CkLocalBranch(CkGroupID gID) {
  return _localBranch(gID);
}

static
void *_ckLocalNodeBranch(CkGroupID groupID) {
  CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
  void *retval = CksvAccess(_nodeGroupTable)->find(groupID).getObj();
  CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
  return retval;
}

extern "C"
void *CkLocalNodeBranch(CkGroupID groupID)
{
  void *retval;
  // we are called in a constructor
  if (CkpvAccess(_currentNodeGroupObj) && CkpvAccess(_currentGroup) == groupID)
    return CkpvAccess(_currentNodeGroupObj);
  while (NULL== (retval=_ckLocalNodeBranch(groupID)))
  { // Nodegroup hasn't finished being created yet-- schedule...
    CsdScheduler(0);
  }
  return retval;
}

extern "C"
void *CkLocalChare(const CkChareID *pCid)
{
	int pe=pCid->onPE;
	if (pe<0) { //A virtual chare ID
		if (pe!=(-(CkMyPe()+1)))
			return NULL;//VID block not on this PE
		VidBlock *v=(VidBlock *)pCid->objPtr;
		return v->getLocalChare();
	}
	else
	{ //An ordinary chare ID
		if (pe!=CkMyPe())
			return NULL;//Chare not on this PE
		return pCid->objPtr;
	}
}

CkpvDeclare(char **,Ck_argv);
extern "C" char **CkGetArgv(void) {
	return CkpvAccess(Ck_argv);
}
extern "C" int CkGetArgc(void) {
	return CmiGetArgc(CkpvAccess(Ck_argv));
}

/******************** Basic support *****************/
extern "C" void CkDeliverMessageFree(int epIdx,void *msg,void *obj)
{
#ifdef _FAULT_MLOG_
        CpvAccess(_currentObj) = (Chare *)obj;
//      printf("[%d] CurrentObj set to %p\n",CkMyPe(),obj);
#endif
  //BIGSIM_OOC DEBUGGING
  //CkPrintf("CkDeliverMessageFree: name of entry fn: %s\n", _entryTable[epIdx]->name);
  //fflush(stdout);
#ifndef CMK_OPTIMIZE
  CpdBeforeEp(epIdx, obj, msg);
#endif
  _entryTable[epIdx]->call(msg, obj);
#ifndef CMK_OPTIMIZE
  CpdAfterEp(epIdx);
#endif
  if (_entryTable[epIdx]->noKeep)
  { /* Method doesn't keep/delete the message, so we have to: */
    _msgTable[_entryTable[epIdx]->msgIdx]->dealloc(msg);
  }
}
extern "C" void CkDeliverMessageReadonly(int epIdx,const void *msg,void *obj)
{
  //BIGSIM_OOC DEBUGGING
  //CkPrintf("CkDeliverMessageReadonly: name of entry fn: %s\n", _entryTable[epIdx]->name);
  //fflush(stdout);

  void *deliverMsg;
#ifdef _FAULT_MLOG_
        CpvAccess(_currentObj) = (Chare *)obj;
#endif
  if (_entryTable[epIdx]->noKeep)
  { /* Deliver a read-only copy of the message */
    deliverMsg=(void *)msg;
  } else
  { /* Method needs a copy of the message to keep/delete */
    void *oldMsg=(void *)msg;
    deliverMsg=CkCopyMsg(&oldMsg);
#ifndef CMK_OPTIMIZE
    if (oldMsg!=msg)
      CkAbort("CkDeliverMessageReadonly: message pack/unpack changed message pointer!");
#endif
  }
#ifndef CMK_OPTIMIZE
  CpdBeforeEp(epIdx, obj, (void*)msg);
#endif
  _entryTable[epIdx]->call(deliverMsg, obj);
#ifndef CMK_OPTIMIZE
  CpdAfterEp(epIdx);
#endif
}

static inline void _invokeEntryNoTrace(int epIdx,envelope *env,void *obj)
{
  register void *msg = EnvToUsr(env);
  _SET_USED(env, 0);
  CkDeliverMessageFree(epIdx,msg,obj);
}

static inline void _invokeEntry(int epIdx,envelope *env,void *obj)
{

#ifndef CMK_OPTIMIZE /* Consider tracing: */
  if (_entryTable[epIdx]->traceEnabled) {
    _TRACE_BEGIN_EXECUTE(env);
    _invokeEntryNoTrace(epIdx,env,obj);
    _TRACE_END_EXECUTE();
  }
  else
#endif
    _invokeEntryNoTrace(epIdx,env,obj);

}

/********************* Creation ********************/

extern "C"
void CkCreateChare(int cIdx, int eIdx, void *msg, CkChareID *pCid, int destPE)
{
  CkAssert(cIdx == _entryTable[eIdx]->chareIdx);
  envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  if(pCid == 0) {
    env->setMsgtype(NewChareMsg);
  } else {
    pCid->onPE = (-(CkMyPe()+1));
    //  pCid->magic = _GETIDX(cIdx);
    pCid->objPtr = (void *) new VidBlock();
    _MEMCHECK(pCid->objPtr);
    env->setMsgtype(NewVChareMsg);
    env->setVidPtr(pCid->objPtr);
#ifndef CMK_CHARE_USE_PTR
    CpvAccess(vidblocks).push_back((VidBlock*)pCid->objPtr);
    int idx = CpvAccess(vidblocks).size()-1;
    pCid->objPtr = (void *)idx;
    env->setVidPtr((void *)idx);
#endif
  }
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _charmHandlerIdx);
  _TRACE_CREATION_1(env);
  CpvAccess(_qd)->create();
  _STATS_RECORD_CREATE_CHARE_1();
  _SET_USED(env, 1);
  if(destPE == CK_PE_ANY)
    env->setForAnyPE(1);
  else
    env->setForAnyPE(0);
  CldEnqueue(destPE, env, _infoIdx);
  _TRACE_CREATION_DONE(1);
}

void CkCreateLocalGroup(CkGroupID groupID, int epIdx, envelope *env)
{
  register int gIdx = _entryTable[epIdx]->chareIdx;
  register void *obj = malloc(_chareTable[gIdx]->size);
  _MEMCHECK(obj);
  setMemoryTypeChare(obj);
  CmiImmediateLock(CkpvAccess(_groupTableImmLock));
  CkpvAccess(_groupTable)->find(groupID).setObj(obj);
  CkpvAccess(_groupTable)->find(groupID).setcIdx(gIdx);
  CkpvAccess(_groupIDTable)->push_back(groupID);
  PtrQ *ptrq = CkpvAccess(_groupTable)->find(groupID).getPending();
  if(ptrq) {
    void *pending;
    while((pending=ptrq->deq())!=0)
      CldEnqueue(CkMyPe(), pending, _infoIdx);
//    delete ptrq;
      CkpvAccess(_groupTable)->find(groupID).clearPending();
  }
  CmiImmediateUnlock(CkpvAccess(_groupTableImmLock));

  CkpvAccess(_currentGroup) = groupID;
  CkpvAccess(_currentGroupRednMgr) = env->getRednMgr();
#ifndef CMK_CHARE_USE_PTR
  ((Chare *)obj)->chareIdx = -1;
#endif
  _invokeEntryNoTrace(epIdx,env,obj); /* can't trace groups: would cause nested begin's */
  _STATS_RECORD_PROCESS_GROUP_1();
}

void CkCreateLocalNodeGroup(CkGroupID groupID, int epIdx, envelope *env)
{
  register int gIdx = _entryTable[epIdx]->chareIdx;
  int objSize=_chareTable[gIdx]->size;
  register void *obj = malloc(objSize);
  _MEMCHECK(obj);
  setMemoryTypeChare(obj);
  CkpvAccess(_currentGroup) = groupID;

// Now that the NodeGroup is created, add it to the table.
//  NodeGroups can be accessed by multiple processors, so
//  this is in the opposite order from groups - invoking the constructor
//  before registering it.
// User may call CkLocalNodeBranch() inside the nodegroup constructor
//  store nodegroup into _currentNodeGroupObj
  CkpvAccess(_currentNodeGroupObj) = obj;
#ifndef CMK_CHARE_USE_PTR
  ((Chare *)obj)->chareIdx = -1;
#endif
  _invokeEntryNoTrace(epIdx,env,obj);
  CkpvAccess(_currentNodeGroupObj) = NULL;
  _STATS_RECORD_PROCESS_NODE_GROUP_1();

  CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
  CksvAccess(_nodeGroupTable)->find(groupID).setObj(obj);
  CksvAccess(_nodeGroupTable)->find(groupID).setcIdx(gIdx);
  CksvAccess(_nodeGroupIDTable).push_back(groupID);

  PtrQ *ptrq = CksvAccess(_nodeGroupTable)->find(groupID).getPending();
  if(ptrq) {
    void *pending;
    while((pending=ptrq->deq())!=0)
      CldNodeEnqueue(CkMyNode(), pending, _infoIdx);
//    delete ptrq;
      CksvAccess(_nodeGroupTable)->find(groupID).clearPending();
  }
  CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
}

void _createGroup(CkGroupID groupID, envelope *env)
{
  _CHECK_USED(env);
  _SET_USED(env, 1);
  register int epIdx = env->getEpIdx();
  int gIdx = _entryTable[epIdx]->chareIdx;
  CkNodeGroupID rednMgr;
  if(_chareTable[gIdx]->isIrr == 0){
		CProxy_CkArrayReductionMgr rednMgrProxy = CProxy_CkArrayReductionMgr::ckNew(0, groupID);
		rednMgr = rednMgrProxy;
//		rednMgrProxy.setAttachedGroup(groupID);
  }else{
	rednMgr.setZero();
  }
  env->setGroupNum(groupID);
  env->setSrcPe(CkMyPe());
  env->setRednMgr(rednMgr);
  env->setGroupEpoch(CkpvAccess(_charmEpoch));

  if(CkNumPes()>1) {
    CkPackMessage(&env);
    CmiSetHandler(env, _bocHandlerIdx);
    _numInitMsgs++;
    CmiSyncBroadcast(env->getTotalsize(), (char *)env);
    CpvAccess(_qd)->create(CkNumPes()-1);
    CkUnpackMessage(&env);
  }
  _STATS_RECORD_CREATE_GROUP_1();
  CkCreateLocalGroup(groupID, epIdx, env);
}

void _createNodeGroup(CkGroupID groupID, envelope *env)
{
  _CHECK_USED(env);
  _SET_USED(env, 1);
  register int epIdx = env->getEpIdx();
  env->setGroupNum(groupID);
  env->setSrcPe(CkMyPe());
  env->setGroupEpoch(CkpvAccess(_charmEpoch));
  if(CkNumNodes()>1) {
    CkPackMessage(&env);
    CmiSetHandler(env, _bocHandlerIdx);
    _numInitMsgs++;
    if (CkpvAccess(_charmEpoch)==0) CksvAccess(_numInitNodeMsgs)++;
    CmiSyncNodeBroadcast(env->getTotalsize(), (char *)env);
    CpvAccess(_qd)->create(CkNumNodes()-1);
    CkUnpackMessage(&env);
  }
  _STATS_RECORD_CREATE_NODE_GROUP_1();
  CkCreateLocalNodeGroup(groupID, epIdx, env);
}

// new _groupCreate

static CkGroupID _groupCreate(envelope *env)
{
  register CkGroupID groupNum;

  // check CkMyPe(). if it is 0 then idx is _numGroups++
  // if not, then something else...
  if(CkMyPe() == 0)
     groupNum.idx = CkpvAccess(_numGroups)++;
  else
     groupNum.idx = _getGroupIdx(CkNumPes(),CkMyPe(),CkpvAccess(_numGroups)++);
  _createGroup(groupNum, env);
  return groupNum;
}

// new _nodeGroupCreate
static CkGroupID _nodeGroupCreate(envelope *env)
{
  register CkGroupID groupNum;
  CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));                // change for proc 0 and other processors
  if(CkMyNode() == 0)				// should this be CkMyPe() or CkMyNode()?
          groupNum.idx = CksvAccess(_numNodeGroups)++;
   else
          groupNum.idx = _getGroupIdx(CkNumNodes(),CkMyNode(),CksvAccess(_numNodeGroups)++);
  CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
  _createNodeGroup(groupNum, env);
  return groupNum;
}

/**** generate the group idx when group is creator pe is not pe0
 **** the 32 bit index has msb set to 1 (+ve indices are used by proc 0)
 **** remaining bits contain the group creator processor number and
 **** the idx number which starts from 1(_numGroups or _numNodeGroups) on each proc ****/

int _getGroupIdx(int numNodes,int myNode,int numGroups)
{
        int idx;
        int x = (int)ceil(log((double)numNodes)/log((double)2));// number of bits needed to store node number
        int n = 32 - (x+1);                                     // number of bits remaining for the index
        idx = (myNode<<n) + numGroups;                          // add number of processors, shift by the no. of bits needed,
                                                                // then add the next available index
	// of course this won't work when int is 8 bytes long on T3E
        //idx |= 0x80000000;                                      // set the most significant bit to 1
	idx = - idx;
								// if int is not 32 bits, wouldn't this be wrong?
        return idx;
}

extern "C"
CkGroupID CkCreateGroup(int cIdx, int eIdx, void *msg)
{
  CkAssert(cIdx == _entryTable[eIdx]->chareIdx);
  register envelope *env = UsrToEnv(msg);
  env->setMsgtype(BocInitMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  _TRACE_CREATION_N(env, CkNumPes());
  CkGroupID gid = _groupCreate(env);
  _TRACE_CREATION_DONE(1);
  return gid;
}

extern "C"
CkGroupID CkCreateNodeGroup(int cIdx, int eIdx, void *msg)
{
  CkAssert(cIdx == _entryTable[eIdx]->chareIdx);
  register envelope *env = UsrToEnv(msg);
  env->setMsgtype(NodeBocInitMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  _TRACE_CREATION_N(env, CkNumNodes());
  CkGroupID gid = _nodeGroupCreate(env);
  _TRACE_CREATION_DONE(1);
  return gid;
}

static inline void *_allocNewChare(envelope *env, int &idx)
{
  int chareIdx = _entryTable[env->getEpIdx()]->chareIdx;
  void *tmp=malloc(_chareTable[chareIdx]->size);
  _MEMCHECK(tmp);
#ifndef CMK_CHARE_USE_PTR
  CpvAccess(chare_objs).push_back(tmp);
  CpvAccess(chare_types).push_back(chareIdx);
  idx = CpvAccess(chare_objs).size()-1;
#endif
  setMemoryTypeChare(tmp);
  return tmp;
}

static void _processNewChareMsg(CkCoreState *ck,envelope *env)
{
  int idx;
  register void *obj = _allocNewChare(env, idx);
#ifndef CMK_CHARE_USE_PTR
  ((Chare *)obj)->chareIdx = idx;
#endif
  _invokeEntry(env->getEpIdx(),env,obj);
}

void CkCreateLocalChare(int epIdx, envelope *env)
{
  env->setEpIdx(epIdx);
  _processNewChareMsg(NULL, env);
}

static void _processNewVChareMsg(CkCoreState *ck,envelope *env)
{
  int idx;
  register void *obj = _allocNewChare(env, idx);
  register CkChareID *pCid = (CkChareID *)
      _allocMsg(FillVidMsg, sizeof(CkChareID));
  pCid->onPE = CkMyPe();
#ifndef CMK_CHARE_USE_PTR
  pCid->objPtr = (void*)idx;
#else
  pCid->objPtr = obj;
#endif
  // pCid->magic = _GETIDX(_entryTable[env->getEpIdx()]->chareIdx);
  register envelope *ret = UsrToEnv(pCid);
  ret->setVidPtr(env->getVidPtr());
  register int srcPe = env->getSrcPe();
  ret->setSrcPe(CkMyPe());
  CmiSetHandler(ret, _charmHandlerIdx);
  CmiSyncSendAndFree(srcPe, ret->getTotalsize(), (char *)ret);
  CpvAccess(_qd)->create();
#ifndef CMK_CHARE_USE_PTR
  ((Chare *)obj)->chareIdx = idx;
#endif
  _invokeEntry(env->getEpIdx(),env,obj);
}

/************** Receive: Chares *************/

static inline void _processForPlainChareMsg(CkCoreState *ck,envelope *env)
{
  register int epIdx = env->getEpIdx();
  register int mainIdx = _chareTable[_entryTable[epIdx]->chareIdx]->mainChareType();
  register void *obj;
  if (mainIdx != -1)  {           // mainchare
    CmiAssert(CkMyPe()==0);
    obj = _mainTable[mainIdx]->getObj();
  }
  else {
#ifndef CMK_CHARE_USE_PTR
    if (_chareTable[_entryTable[epIdx]->chareIdx]->chareType == TypeChare)
      obj = CpvAccess(chare_objs)[(CmiIntPtr)env->getObjPtr()];
    else
      obj = env->getObjPtr();
#else
    obj = env->getObjPtr();
#endif
  }
  _invokeEntry(epIdx,env,obj);
}

static inline void _processForChareMsg(CkCoreState *ck,envelope *env)
{
  register int epIdx = env->getEpIdx();
  register void *obj = env->getObjPtr();
  _invokeEntry(epIdx,env,obj);
}

static inline void _processFillVidMsg(CkCoreState *ck,envelope *env)
{
#ifndef CMK_CHARE_USE_PTR
  register VidBlock *vptr = CpvAccess(vidblocks)[(CmiIntPtr)env->getVidPtr()];
#else
  register VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _CHECK_VALID(vptr, "FillVidMsg: Not a valid VIdPtr\n");
#endif
  register CkChareID *pcid = (CkChareID *) EnvToUsr(env);
  _CHECK_VALID(pcid, "FillVidMsg: Not a valid pCid\n");
  vptr->fill(pcid->onPE, pcid->objPtr);
  CmiFree(env);
}

static inline void _processForVidMsg(CkCoreState *ck,envelope *env)
{
#ifndef CMK_CHARE_USE_PTR
  register VidBlock *vptr = CpvAccess(vidblocks)[(CmiIntPtr)env->getVidPtr()];
#else
  VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _CHECK_VALID(vptr, "ForVidMsg: Not a valid VIdPtr\n");
#endif
  _SET_USED(env, 1);
  vptr->send(env);
}

/************** Receive: Groups ****************/

/**
 This message is sent to this groupID--prepare to
 handle this message by looking up the group,
 and possibly stashing the message.
*/
IrrGroup *_lookupGroup(CkCoreState *ck,envelope *env,const CkGroupID &groupID)
{

	CmiImmediateLock(CkpvAccess(_groupTableImmLock));
	IrrGroup *obj = ck->localBranch(groupID);
	if (obj==NULL) { /* groupmember not yet created: stash message */
		ck->getGroupTable()->find(groupID).enqMsg(env);
	}
	else { /* will be able to process message */
		ck->process();
	}
	CmiImmediateUnlock(CkpvAccess(_groupTableImmLock));
	return obj;
}

static inline void _deliverForBocMsg(CkCoreState *ck,int epIdx,envelope *env,IrrGroup *obj)
{
#if CMK_LBDB_ON
  // if there is a running obj being measured, stop it temporarily
  LDObjHandle objHandle;
  int objstopped = 0;
  LBDatabase *the_lbdb = (LBDatabase *)CkLocalBranch(_lbdb);
  if (the_lbdb->RunningObject(&objHandle)) {
    objstopped = 1;
    the_lbdb->ObjectStop(objHandle);
  }
#endif
  _invokeEntry(epIdx,env,obj);
#if CMK_LBDB_ON
  if (objstopped) the_lbdb->ObjectStart(objHandle);
#endif
  _STATS_RECORD_PROCESS_BRANCH_1();
}

static inline void _processForBocMsg(CkCoreState *ck,envelope *env)
{
  register CkGroupID groupID =  env->getGroupNum();
  register IrrGroup *obj = _lookupGroup(ck,env,env->getGroupNum());
  if(obj) {
    _deliverForBocMsg(ck,env->getEpIdx(),env,obj);
  }
}

static inline void _deliverForNodeBocMsg(CkCoreState *ck,envelope *env,void *obj)
{
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  _processForChareMsg(ck,env);
  _STATS_RECORD_PROCESS_NODE_BRANCH_1();
}

static inline void _deliverForNodeBocMsg(CkCoreState *ck,int epIdx, envelope *env,void *obj)
{
  env->setEpIdx(epIdx);
  _deliverForNodeBocMsg(ck,env, obj);
}

static inline void _processForNodeBocMsg(CkCoreState *ck,envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register void *obj;

  CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
  obj = CksvAccess(_nodeGroupTable)->find(groupID).getObj();
  if(!obj) { // groupmember not yet created
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(env))     // buffer immediate message
      CmiDelayImmediate();
    else
#endif
    CksvAccess(_nodeGroupTable)->find(groupID).enqMsg(env);
    CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
    return;
  }
  CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
#if CMK_IMMEDIATE_MSG
  if (!CmiIsImmediate(env))
#endif
  ck->process();
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  _processForChareMsg(ck,env);
  _STATS_RECORD_PROCESS_NODE_BRANCH_1();
}

void _processBocInitMsg(CkCoreState *ck,envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register int epIdx = env->getEpIdx();
  CkCreateLocalGroup(groupID, epIdx, env);
}

void _processNodeBocInitMsg(CkCoreState *ck,envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register int epIdx = env->getEpIdx();
  CkCreateLocalNodeGroup(groupID, epIdx, env);
}

/************** Receive: Arrays *************/

static void _processArrayEltInitMsg(CkCoreState *ck,envelope *env) {
  CkArray *mgr=(CkArray *)_lookupGroup(ck,env,env->getsetArrayMgr());
  if (mgr) {
    _SET_USED(env, 0);
    mgr->insertElement((CkMessage *)EnvToUsr(env));
  }
}
static void _processArrayEltMsg(CkCoreState *ck,envelope *env) {
  CkArray *mgr=(CkArray *)_lookupGroup(ck,env,env->getsetArrayMgr());
  if (mgr) {
    _SET_USED(env, 0);
    mgr->getLocMgr()->deliverInline((CkMessage *)EnvToUsr(env));
  }
}

//BIGSIM_OOC DEBUGGING
#define TELLMSGTYPE(x) //x

/**
 * This is the main converse-level handler used by all of Charm++.
 *
 * \addtogroup CriticalPathFramework
 */
void _processHandler(void *converseMsg,CkCoreState *ck)
{
  register envelope *env = (envelope *) converseMsg;

//#if CMK_RECORD_REPLAY
  if (ck->watcher!=NULL) {
    if (!ck->watcher->processMessage(env,ck)) return;
  }
//#endif
#ifdef _FAULT_MLOG_
        Chare *obj=NULL;
        CkObjID sender;
        MCount SN;
        MlogEntry *entry=NULL;
        if(env->getMsgtype() == ForBocMsg || env->getMsgtype() == ForNodeBocMsg ||
        env->getMsgtype() == ForArrayEltMsg){
                sender = env->sender;
                SN = env->SN;
                int result = preProcessReceivedMessage(env,&obj,&entry);
                if(result == 0){
                        return;
                }
        }
#endif

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  //  CkPrintf("START\n");
  criticalPath_start(env);
#endif


  switch(env->getMsgtype()) {
// Group support
    case BocInitMsg :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: BocInitMsg\n", CkMyPe());)
      ck->process(); if(env->isPacked()) CkUnpackMessage(&env);
      _processBocInitMsg(ck,env);
      break;
    case NodeBocInitMsg :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: NodeBocInitMsg\n", CkMyPe());)
      ck->process(); if(env->isPacked()) CkUnpackMessage(&env);
      _processNodeBocInitMsg(ck,env);
      break;
    case ForBocMsg :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: ForBocMsg\n", CkMyPe());)
      // QD processing moved inside _processForBocMsg because it is conditional
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForBocMsg(ck,env);
      // stats record moved inside _processForBocMsg because it is conditional
      break;
    case ForNodeBocMsg :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: ForNodeBocMsg\n", CkMyPe());)
      // QD processing moved to _processForNodeBocMsg because it is conditional
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForNodeBocMsg(ck,env);
      // stats record moved to _processForNodeBocMsg because it is conditional
      break;

// Array support
    case ArrayEltInitMsg:
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: ArrayEltInitMsg\n", CkMyPe());)
      if(env->isPacked()) CkUnpackMessage(&env);
      _processArrayEltInitMsg(ck,env);
      break;
    case ForArrayEltMsg:
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: ForArrayEltMsg\n", CkMyPe());)
      if(env->isPacked()) CkUnpackMessage(&env);
      _processArrayEltMsg(ck,env);
      break;

// Chare support
    case NewChareMsg :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: NewChareMsg\n", CkMyPe());)
      ck->process(); if(env->isPacked()) CkUnpackMessage(&env);
      _processNewChareMsg(ck,env);
      _STATS_RECORD_PROCESS_CHARE_1();
      break;
    case NewVChareMsg :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: NewVChareMsg\n", CkMyPe());)
      ck->process(); if(env->isPacked()) CkUnpackMessage(&env);
      _processNewVChareMsg(ck,env);
      _STATS_RECORD_PROCESS_CHARE_1();
      break;
    case ForChareMsg :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: ForChareMsg\n", CkMyPe());)
      ck->process(); if(env->isPacked()) CkUnpackMessage(&env);
      _processForPlainChareMsg(ck,env);
      _STATS_RECORD_PROCESS_MSG_1();
      break;
    case ForVidMsg   :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: ForVidMsg\n", CkMyPe());)
      ck->process();
      _processForVidMsg(ck,env);
      break;
    case FillVidMsg  :
      TELLMSGTYPE(CkPrintf("proc[%d]: _processHandler with msg type: FillVidMsg\n", CkMyPe());)
      ck->process();
      _processFillVidMsg(ck,env);
      break;

    default:
      CmiAbort("Fatal Charm++ Error> Unknown msg-type in _processHandler.\n");
  }
#ifdef _FAULT_MLOG_
        if(obj != NULL){
                postProcessReceivedMessage(obj,sender,SN,entry);
        }
#endif


#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  criticalPath_end();
  //  CkPrintf("STOP\n");
#endif


}


/******************** Message Send **********************/

void _infoFn(void *converseMsg, CldPackFn *pfn, int *len,
             int *queueing, int *priobits, unsigned int **prioptr)
{
  register envelope *env = (envelope *)converseMsg;
  *pfn = (CldPackFn)CkPackMessage;
  *len = env->getTotalsize();
  *queueing = env->getQueueing();
  *priobits = env->getPriobits();
  *prioptr = (unsigned int *) env->getPrioPtr();
}

void CkPackMessage(envelope **pEnv)
{
  register envelope *env = *pEnv;
  if(!env->isPacked() && _msgTable[env->getMsgIdx()]->pack) {
    register void *msg = EnvToUsr(env);
    _TRACE_BEGIN_PACK();
    msg = _msgTable[env->getMsgIdx()]->pack(msg);
    _TRACE_END_PACK();
    env=UsrToEnv(msg);
    env->setPacked(1);
    *pEnv = env;
  }
}

void CkUnpackMessage(envelope **pEnv)
{
  register envelope *env = *pEnv;
  register int msgIdx = env->getMsgIdx();
  if(env->isPacked()) {
    register void *msg = EnvToUsr(env);
    _TRACE_BEGIN_UNPACK();
    msg = _msgTable[msgIdx]->unpack(msg);
    _TRACE_END_UNPACK();
    env=UsrToEnv(msg);
    env->setPacked(0);
    *pEnv = env;
  }
}

//There's no reason for most messages to go through the Cld--
// the PE can never be CLD_ANYWHERE; wasting _infoFn calls.
// Thus these accellerated versions of the Cld calls.

int index_objectQHandler;
int index_tokenHandler;
static int index_skipCldHandler;

static void _skipCldHandler(void *converseMsg)
{
  register envelope *env = (envelope *)(converseMsg);
  CmiSetHandler(converseMsg, CmiGetXHandler(converseMsg));
#if CMK_GRID_QUEUE_AVAILABLE
  if (CmiGridQueueLookupMsg ((char *) converseMsg)) {
    CqsEnqueueGeneral ((Queue) CpvAccess (CsdGridQueue),
		       env, env->getQueueing (), env->getPriobits (),
		       (unsigned int *) env->getPrioPtr ());
  } else {
    CqsEnqueueGeneral ((Queue) CpvAccess (CsdSchedQueue),
		       env, env->getQueueing (), env->getPriobits (),
		       (unsigned int *) env->getPrioPtr ());
  }
#else
  CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	env, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
#endif
}


//static void _skipCldEnqueue(int pe,envelope *env, int infoFn)
// Made non-static to be used by ckmessagelogging
void _skipCldEnqueue(int pe,envelope *env, int infoFn)
{
  if(pe == CkMyPe() ){
    if(!CmiNodeAlive(CkMyPe())){
	printf("[%d] Invalid processor sending itself a message \n",CkMyPe());
//	return;
    }
  }
  if (pe == CkMyPe() && !CmiImmIsRunning()) {
#if CMK_OBJECT_QUEUE_AVAILABLE
    Chare *obj = CkFindObjectPtr(env);
    if (obj && obj->CkGetObjQueue().queue()) {
      _enqObjQueue(obj, env);
    }
    else
#endif
    CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	env, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
  } else {
    CkPackMessage(&env);
    int len=env->getTotalsize();
    CmiSetXHandler(env,CmiGetHandler(env));
#if CMK_OBJECT_QUEUE_AVAILABLE
    CmiSetHandler(env,index_objectQHandler);
#else
    CmiSetHandler(env,index_skipCldHandler);
#endif
    CmiSetInfo(env,infoFn);
    if (pe==CLD_BROADCAST) {
#ifdef _FAULT_MLOG_             
                        CmiSyncBroadcast(len, (char *)env);
#else
 			CmiSyncBroadcastAndFree(len, (char *)env); 
#endif

}
    else if (pe==CLD_BROADCAST_ALL) { 
#ifdef _FAULT_MLOG_             
                        CmiSyncBroadcastAll(len, (char *)env);
#else
                        CmiSyncBroadcastAllAndFree(len, (char *)env);
#endif

}
    else{
#ifdef _FAULT_MLOG_             
                        CmiSyncSend(pe, len, (char *)env);
#else
                        CmiSyncSendAndFree(pe, len, (char *)env);
#endif

		}
  }
}

#if CMK_BLUEGENE_CHARM
#   define  _skipCldEnqueue   CldEnqueue
#endif

// by pass Charm++ priority queue, send as Converse message
static void _noCldEnqueueMulti(int npes, int *pes, envelope *env)
{
  CkPackMessage(&env);
  int len=env->getTotalsize();
  CmiSyncListSendAndFree(npes, pes, len, (char *)env);
}

static void _noCldEnqueue(int pe, envelope *env)
{
/*
  if (pe == CkMyPe()) {
    CmiHandleMessage(env);
  } else
*/
  CkPackMessage(&env);
  int len=env->getTotalsize();
  if (pe==CLD_BROADCAST) { CmiSyncBroadcastAndFree(len, (char *)env); }
  else if (pe==CLD_BROADCAST_ALL) { CmiSyncBroadcastAllAndFree(len, (char *)env); }
  else CmiSyncSendAndFree(pe, len, (char *)env);
}

//static void _noCldNodeEnqueue(int node, envelope *env)
//Made non-static to be used by ckmessagelogging
void _noCldNodeEnqueue(int node, envelope *env)
{
/*
  if (node == CkMyNode()) {
    CmiHandleMessage(env);
  } else {
*/
  CkPackMessage(&env);
  int len=env->getTotalsize();
  if (node==CLD_BROADCAST) { 
#ifdef _FAULT_MLOG_
        CmiSyncNodeBroadcast(len, (char *)env);
#else
	CmiSyncNodeBroadcastAndFree(len, (char *)env); 
#endif
}
  else if (node==CLD_BROADCAST_ALL) { 
#ifdef _FAULT_MLOG_
                CmiSyncNodeBroadcastAll(len, (char *)env);
#else
		CmiSyncNodeBroadcastAllAndFree(len, (char *)env); 
#endif

}
  else {
#ifdef _FAULT_MLOG_
        CmiSyncNodeSend(node, len, (char *)env);
#else
	CmiSyncNodeSendAndFree(node, len, (char *)env);
#endif
  }
}

static inline int _prepareMsg(int eIdx,void *msg,const CkChareID *pCid)
{
  register envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  _SET_USED(env, 1);
  env->setMsgtype(ForChareMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  criticalPath_send(env);
  automaticallySetMessagePriority(env);
#endif
#ifndef CMK_OPTIMIZE
  setMemoryOwnedBy(((char*)env)-sizeof(CmiChunkHeader), 0);
#endif
#if CMK_OBJECT_QUEUE_AVAILABLE
  CmiSetHandler(env, index_objectQHandler);
#else
  CmiSetHandler(env, _charmHandlerIdx);
#endif
  if (pCid->onPE < 0) { //Virtual chare ID (VID)
    register int pe = -(pCid->onPE+1);
    if(pe==CkMyPe()) {
#ifndef CMK_CHARE_USE_PTR
      VidBlock *vblk = CpvAccess(vidblocks)[(CmiIntPtr)pCid->objPtr];
#else
      VidBlock *vblk = (VidBlock *) pCid->objPtr;
#endif
      void *objPtr;
      if (NULL!=(objPtr=vblk->getLocalChare()))
      { //A ready local chare
	env->setObjPtr(objPtr);
	return pe;
      }
      else { //The vidblock is not ready-- forget it
        vblk->send(env);
        return -1;
      }
    } else { //Valid vidblock for another PE:
      env->setMsgtype(ForVidMsg);
      env->setVidPtr(pCid->objPtr);
      return pe;
    }
  }
  else {
    env->setObjPtr(pCid->objPtr);
    return pCid->onPE;
  }
}

static inline int _prepareImmediateMsg(int eIdx,void *msg,const CkChareID *pCid)
{
  int destPE = _prepareMsg(eIdx, msg, pCid);
  if (destPE != -1) {
    register envelope *env = UsrToEnv(msg);
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
    criticalPath_send(env);
    automaticallySetMessagePriority(env);
#endif
    CmiBecomeImmediate(env);
  }
  return destPE;
}

extern "C"
void CkSendMsg(int entryIdx, void *msg,const CkChareID *pCid, int opts)
{
  if (opts & CK_MSG_INLINE) {
    CkSendMsgInline(entryIdx, msg, pCid, opts);
    return;
  }
#ifndef CMK_OPTIMIZE
  if (opts & CK_MSG_IMMEDIATE) {
    CmiAbort("Immediate message is not allowed in Chare!");
  }
#endif
  register envelope *env = UsrToEnv(msg);
  int destPE=_prepareMsg(entryIdx,msg,pCid);
  // Before it traced the creation only if destPE!=-1 (i.e it did not when the
  // VidBlock was not yet filled). The problem is that the creation was never
  // traced later when the VidBlock was filled. One solution is to trace the
  // creation here, the other to trace it in VidBlock->msgDeliver().
  _TRACE_CREATION_1(env);
  if (destPE!=-1) {
    CpvAccess(_qd)->create();
    if (opts & CK_MSG_SKIP_OR_IMM)
      _noCldEnqueue(destPE, env);
    else
      CldEnqueue(destPE, env, _infoIdx);
  }
  _TRACE_CREATION_DONE(1);
}

extern "C"
void CkSendMsgInline(int entryIndex, void *msg, const CkChareID *pCid, int opts)
{
  if (pCid->onPE==CkMyPe())
  {
    if(!CmiNodeAlive(CkMyPe())){
	return;
    }
#ifndef CMK_OPTIMIZE
    //Just in case we need to breakpoint or use the envelope in some way
    _prepareMsg(entryIndex,msg,pCid);
#endif
		//Just directly call the chare (skip QD handling & scheduler)
    register envelope *env = UsrToEnv(msg);
    if (env->isPacked()) CkUnpackMessage(&env);
    _STATS_RECORD_PROCESS_MSG_1();
    _invokeEntryNoTrace(entryIndex,env,pCid->objPtr);
  }
  else {
    //No way to inline a cross-processor message:
    CkSendMsg(entryIndex,msg,pCid,opts&!CK_MSG_INLINE);
  }
}

static inline envelope *_prepareMsgBranch(int eIdx,void *msg,CkGroupID gID,int type)
{
  register envelope *env = UsrToEnv(msg);
  CkNodeGroupID nodeRedMgr;
  _CHECK_USED(env);
  _SET_USED(env, 1);
  env->setMsgtype(type);
  env->setEpIdx(eIdx);
  env->setGroupNum(gID);
  env->setSrcPe(CkMyPe());
#ifndef CMK_OPTIMIZE
  nodeRedMgr.setZero();
  env->setRednMgr(nodeRedMgr);
#endif
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  criticalPath_send(env);
  automaticallySetMessagePriority(env);
#endif
#ifndef CMK_OPTIMIZE
  setMemoryOwnedBy(((char*)env)-sizeof(CmiChunkHeader), 0);
#endif
  CmiSetHandler(env, _charmHandlerIdx);
  return env;
}

static inline envelope *_prepareImmediateMsgBranch(int eIdx,void *msg,CkGroupID gID,int type)
{
  envelope *env = _prepareMsgBranch(eIdx, msg, gID, type);
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  criticalPath_send(env);
  automaticallySetMessagePriority(env);
#endif
  CmiBecomeImmediate(env);
  return env;
}

static inline void _sendMsgBranch(int eIdx, void *msg, CkGroupID gID,
                  int pe=CLD_BROADCAST_ALL, int opts = 0)
{
  int numPes;
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForBocMsg);
#ifdef _FAULT_MLOG_
        sendTicketGroupRequest(env,pe,_infoIdx);
#else
  _TRACE_ONLY(numPes = (pe==CLD_BROADCAST_ALL?CkNumPes():1));
  _TRACE_CREATION_N(env, numPes);
  if (opts & CK_MSG_SKIP_OR_IMM)
    _noCldEnqueue(pe, env);
  else
    _skipCldEnqueue(pe, env, _infoIdx);
  _TRACE_CREATION_DONE(1);
#endif
}

static inline void _sendMsgBranchMulti(int eIdx, void *msg, CkGroupID gID,
                           int npes, int *pes)
{
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForBocMsg);
  _TRACE_CREATION_MULTICAST(env, npes, pes);
  CldEnqueueMulti(npes, pes, env, _infoIdx);
  _TRACE_CREATION_DONE(1); 	// since it only creates one creation event.
}

extern "C"
void CkSendMsgBranchImmediate(int eIdx, void *msg, int destPE, CkGroupID gID)
{
#if CMK_IMMEDIATE_MSG && ! CMK_SMP
  if (destPE==CkMyPe())
  {
    CkSendMsgBranchInline(eIdx, msg, destPE, gID);
    return;
  }
  //Can't inline-- send the usual way
  register envelope *env = UsrToEnv(msg);
  int numPes;
  _TRACE_ONLY(numPes = (destPE==CLD_BROADCAST_ALL?CkNumPes():1));
  env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForBocMsg);
  _TRACE_CREATION_N(env, numPes);
  _noCldEnqueue(destPE, env);
  _STATS_RECORD_SEND_BRANCH_1();
  CkpvAccess(_coreState)->create();
  _TRACE_CREATION_DONE(1);
#else
  // no support for immediate message, send inline
  CkSendMsgBranchInline(eIdx, msg, destPE, gID);
#endif
}

extern "C"
void CkSendMsgBranchInline(int eIdx, void *msg, int destPE, CkGroupID gID, int opts)
{
  if (destPE==CkMyPe())
  {
    if(!CmiNodeAlive(CkMyPe())){
	return;
    }
    IrrGroup *obj=(IrrGroup *)_localBranch(gID);
    if (obj!=NULL)
    { //Just directly call the group:
#ifndef CMK_OPTIMIZE
      envelope *env=_prepareMsgBranch(eIdx,msg,gID,ForBocMsg);
#else
      envelope *env=UsrToEnv(msg);
#endif
      _deliverForBocMsg(CkpvAccess(_coreState),eIdx,env,obj);
      return;
    }
  }
  //Can't inline-- send the usual way, clear CK_MSG_INLINE
  CkSendMsgBranch(eIdx,msg,destPE,gID,opts&!CK_MSG_INLINE);
}

extern "C"
void CkSendMsgBranch(int eIdx, void *msg, int pe, CkGroupID gID, int opts)
{
  if (opts & CK_MSG_INLINE) {
    CkSendMsgBranchInline(eIdx, msg, pe, gID, opts);
    return;
  }
  if (opts & CK_MSG_IMMEDIATE) {
    CkSendMsgBranchImmediate(eIdx,msg,pe,gID);
    return;
  }
  _sendMsgBranch(eIdx, msg, gID, pe, opts);
  _STATS_RECORD_SEND_BRANCH_1();
  CkpvAccess(_coreState)->create();
}

extern "C"
void CkSendMsgBranchMultiImmediate(int eIdx,void *msg,CkGroupID gID,int npes,int *pes)
{
#if CMK_IMMEDIATE_MSG && ! CMK_SMP
  register envelope *env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForBocMsg);
  _TRACE_CREATION_MULTICAST(env, npes, pes);
  _noCldEnqueueMulti(npes, pes, env);
  _TRACE_CREATION_DONE(1);      // since it only creates one creation event.
#else
  _sendMsgBranchMulti(eIdx, msg, gID, npes, pes);
  CpvAccess(_qd)->create(-npes);
#endif
  _STATS_RECORD_SEND_BRANCH_N(npes);
  CpvAccess(_qd)->create(npes);
}

extern "C"
void CkSendMsgBranchMulti(int eIdx,void *msg,CkGroupID gID,int npes,int *pes, int opts)
{
  if (opts & CK_MSG_IMMEDIATE) {
    CkSendMsgBranchMultiImmediate(eIdx,msg,gID,npes,pes);
    return;
  }
    // normal mesg
  _sendMsgBranchMulti(eIdx, msg, gID, npes, pes);
  _STATS_RECORD_SEND_BRANCH_N(npes);
  CpvAccess(_qd)->create(npes);
}

extern "C"
void CkBroadcastMsgBranch(int eIdx, void *msg, CkGroupID gID, int opts)
{
  _sendMsgBranch(eIdx, msg, gID, CLD_BROADCAST_ALL, opts);
  _STATS_RECORD_SEND_BRANCH_N(CkNumPes());
  CpvAccess(_qd)->create(CkNumPes());
}

static inline void _sendMsgNodeBranch(int eIdx, void *msg, CkGroupID gID,
                int node=CLD_BROADCAST_ALL, int opts=0)
{
  int numPes;
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
#ifdef _FAULT_MLOG_
        sendTicketNodeGroupRequest(env,node,_infoIdx);
#else
  _TRACE_ONLY(numPes = (node==CLD_BROADCAST_ALL?CkNumNodes():1));
  _TRACE_CREATION_N(env, numPes);
  if (opts & CK_MSG_SKIP_OR_IMM) {
    _noCldNodeEnqueue(node, env);
    if (opts & CK_MSG_IMMEDIATE) {    // immediate msg is invisible to QD
      CkpvAccess(_coreState)->create(-numPes);
    }
  }
  else
    CldNodeEnqueue(node, env, _infoIdx);
  _TRACE_CREATION_DONE(1);
#endif
}

static inline void _sendMsgNodeBranchMulti(int eIdx, void *msg, CkGroupID gID,
                           int npes, int *nodes)
{
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
  _TRACE_CREATION_N(env, npes);
  for (int i=0; i<npes; i++) {
    CldNodeEnqueue(nodes[i], env, _infoIdx);
  }
  _TRACE_CREATION_DONE(1);  // since it only creates one creation event.
}

extern "C"
void CkSendMsgNodeBranchImmediate(int eIdx, void *msg, int node, CkGroupID gID)
{
#if CMK_IMMEDIATE_MSG
  if (node==CkMyNode())
  {
    CkSendMsgNodeBranchInline(eIdx, msg, node, gID);
    return;
  }
  //Can't inline-- send the usual way
  register envelope *env = UsrToEnv(msg);
  int numPes;
  _TRACE_ONLY(numPes = (node==CLD_BROADCAST_ALL?CkNumNodes():1));
  env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
  _TRACE_CREATION_N(env, numPes);
  _noCldNodeEnqueue(node, env);
  _STATS_RECORD_SEND_BRANCH_1();
  /* immeidate message is invisible to QD */
//  CkpvAccess(_coreState)->create();
  _TRACE_CREATION_DONE(1);
#else
  // no support for immediate message, send inline
  CkSendMsgNodeBranchInline(eIdx, msg, node, gID);
#endif
}

extern "C"
void CkSendMsgNodeBranchInline(int eIdx, void *msg, int node, CkGroupID gID, int opts)
{
  if (node==CkMyNode())
  {
    CmiImmediateLock(CksvAccess(_nodeGroupTableImmLock));
    void *obj = CksvAccess(_nodeGroupTable)->find(gID).getObj();
    CmiImmediateUnlock(CksvAccess(_nodeGroupTableImmLock));
    if (obj!=NULL)
    { //Just directly call the group:
#ifndef CMK_OPTIMIZE
      envelope *env=_prepareMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
#else
      envelope *env=UsrToEnv(msg);
#endif
      _deliverForNodeBocMsg(CkpvAccess(_coreState),eIdx,env,obj);
      return;
    }
  }
  //Can't inline-- send the usual way
  CkSendMsgNodeBranch(eIdx,msg,node,gID,opts&!CK_MSG_INLINE);
}

extern "C"
void CkSendMsgNodeBranch(int eIdx, void *msg, int node, CkGroupID gID, int opts)
{
  if (opts & CK_MSG_INLINE) {
    CkSendMsgNodeBranchInline(eIdx, msg, node, gID, opts);
    return;
  }
  if (opts & CK_MSG_IMMEDIATE) {
    CkSendMsgNodeBranchImmediate(eIdx, msg, node, gID);
    return;
  }
  _sendMsgNodeBranch(eIdx, msg, gID, node, opts);
  _STATS_RECORD_SEND_NODE_BRANCH_1();
  CkpvAccess(_coreState)->create();
}

extern "C"
void CkSendMsgNodeBranchMultiImmediate(int eIdx,void *msg,CkGroupID gID,int npes,int *nodes)
{
#if CMK_IMMEDIATE_MSG && ! CMK_SMP
  register envelope *env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
  _noCldEnqueueMulti(npes, nodes, env);
#else
  _sendMsgNodeBranchMulti(eIdx, msg, gID, npes, nodes);
  CpvAccess(_qd)->create(-npes);
#endif
  _STATS_RECORD_SEND_NODE_BRANCH_N(npes);
  CpvAccess(_qd)->create(npes);
}

extern "C"
void CkSendMsgNodeBranchMulti(int eIdx,void *msg,CkGroupID gID,int npes,int *nodes, int opts)
{
  if (opts & CK_MSG_IMMEDIATE) {
    CkSendMsgNodeBranchMultiImmediate(eIdx,msg,gID,npes,nodes);
    return;
  }
    // normal mesg
  _sendMsgNodeBranchMulti(eIdx, msg, gID, npes, nodes);
  _STATS_RECORD_SEND_NODE_BRANCH_N(npes);
  CpvAccess(_qd)->create(npes);
}

extern "C"
void CkBroadcastMsgNodeBranch(int eIdx, void *msg, CkGroupID gID, int opts)
{
  _sendMsgNodeBranch(eIdx, msg, gID, CLD_BROADCAST_ALL, opts);
  _STATS_RECORD_SEND_NODE_BRANCH_N(CkNumNodes());
  CpvAccess(_qd)->create(CkNumNodes());
}

//Needed by delegation manager:
extern "C"
int CkChareMsgPrep(int eIdx, void *msg,const CkChareID *pCid)
{ return _prepareMsg(eIdx,msg,pCid); }
extern "C"
void CkGroupMsgPrep(int eIdx, void *msg, CkGroupID gID)
{ _prepareMsgBranch(eIdx,msg,gID,ForBocMsg); }
extern "C"
void CkNodeGroupMsgPrep(int eIdx, void *msg, CkGroupID gID)
{ _prepareMsgBranch(eIdx,msg,gID,ForNodeBocMsg); }

void _ckModuleInit(void) {
	index_skipCldHandler = CkRegisterHandler((CmiHandler)_skipCldHandler);
	index_objectQHandler = CkRegisterHandler((CmiHandler)_ObjectQHandler);
	index_tokenHandler = CkRegisterHandler((CmiHandler)_TokenHandler);
	CkpvInitialize(TokenPool*, _tokenPool);
	CkpvAccess(_tokenPool) = new TokenPool;
}


/************** Send: Arrays *************/

extern void CkArrayManagerInsert(int onPe,void *msg);
//extern void CkArrayManagerDeliver(int onPe,void *msg);

static void _prepareOutgoingArrayMsg(envelope *env,int type)
{
  _CHECK_USED(env);
  _SET_USED(env, 1);
  env->setMsgtype(type);
#ifndef CMK_OPTIMIZE
  setMemoryOwnedBy(((char*)env)-sizeof(CmiChunkHeader), 0);
#endif
  CmiSetHandler(env, _charmHandlerIdx);
  CpvAccess(_qd)->create();
}

extern "C"
void CkArrayManagerInsert(int pe,void *msg,CkGroupID aID) {
  register envelope *env = UsrToEnv(msg);
  env->getsetArrayMgr()=aID;
  _prepareOutgoingArrayMsg(env,ArrayEltInitMsg);
  CldEnqueue(pe, env, _infoIdx);
}

extern "C"
void CkArrayManagerDeliver(int pe,void *msg, int opts) {
  register envelope *env = UsrToEnv(msg);
  _prepareOutgoingArrayMsg(env,ForArrayEltMsg);
#ifdef _FAULT_MLOG_
        sendTicketArrayRequest(env,pe,_infoIdx);
#else
  if (opts & CK_MSG_IMMEDIATE)
    CmiBecomeImmediate(env);
  if (opts & CK_MSG_SKIP_OR_IMM)
    _noCldEnqueue(pe, env);
  else
    _skipCldEnqueue(pe, env, _infoIdx);
#endif
}

class ElementDestroyer : public CkLocIterator {
private:
        CkLocMgr *locMgr;
public:
        ElementDestroyer(CkLocMgr* mgr_):locMgr(mgr_){};
        void addLocation(CkLocation &loc) {
	  loc.destroyAll();
        }
};

void CkDeleteChares() {
  int i;
  int numGroups = CkpvAccess(_groupIDTable)->size();

  // delete all array elements
  for(i=0;i<numGroups;i++) {
    IrrGroup *obj = CkpvAccess(_groupTable)->find((*CkpvAccess(_groupIDTable))[i]).getObj();
    if(obj && obj->isLocMgr())  {
      CkLocMgr *mgr = (CkLocMgr*)obj;
      ElementDestroyer destroyer(mgr);
      mgr->iterate(destroyer);
printf("[%d] DELETE!\n", CkMyPe());
    }
  }

  // delete all groups
  CmiImmediateLock(CkpvAccess(_groupTableImmLock));
  for(i=0;i<numGroups;i++) {
    CkGroupID gID = (*CkpvAccess(_groupIDTable))[i];
    IrrGroup *obj = CkpvAccess(_groupTable)->find(gID).getObj();
    if (obj) delete obj;
  }
  CmiImmediateUnlock(CkpvAccess(_groupTableImmLock));

  // delete all node groups
  if (CkMyRank() == 0) {
    int numNodeGroups = CksvAccess(_nodeGroupIDTable).size();
    for(i=0;i<numNodeGroups;i++) {
      CkGroupID gID = CksvAccess(_nodeGroupIDTable)[i];
      IrrGroup *obj = CksvAccess(_nodeGroupTable)->find(gID).getObj();
      if (obj) delete obj;
    }
  }
}

//------------------- Message Watcher (record/replay) ----------------

#include "crc32.h"

CkpvDeclare(int, envelopeEventID);

CkMessageWatcher::~CkMessageWatcher() {}

class CkMessageRecorder : public CkMessageWatcher {
public:
  CkMessageRecorder(FILE *f_) { f=f_; }
  ~CkMessageRecorder() {
    fprintf(f,"-1 -1 -1");
    fclose(f);
  }

private:
  virtual CmiBool process(envelope *env,CkCoreState *ck) {
    if (env->getEvent()) {
      bool wasPacked = env->isPacked();
      if (!wasPacked) CkPackMessage(&env);
      //unsigned int crc = crc32_initial(((unsigned char*)env)+CmiMsgHeaderSizeBytes, env->getTotalsize()-CmiMsgHeaderSizeBytes);
      unsigned int crc1 = crc32_initial(((unsigned char*)env)+CmiMsgHeaderSizeBytes, sizeof(*env)-CmiMsgHeaderSizeBytes);
      unsigned int crc2 = crc32_initial(((unsigned char*)env)+sizeof(*env), env->getTotalsize()-sizeof(*env));
      fprintf(f,"%d %d %d %hhd %x %x\n",env->getSrcPe(),env->getTotalsize(),env->getEvent(), env->getMsgtype()==NodeBocInitMsg || env->getMsgtype()==ForNodeBocMsg, crc1, crc2);
      if (!wasPacked) CkUnpackMessage(&env);
    }
    return CmiTrue;
  }
};

class CkMessageDetailRecorder : public CkMessageWatcher {
public:
  CkMessageDetailRecorder(FILE *f_) {
    f=f_;
    /* The file starts with "x 0" if it is little endian, "0 x" if big endian.
     * The value of 'x' is the pointer size.
     */
    CmiUInt2 little = sizeof(void*);
    fwrite(&little, 2, 1, f);
  }
  ~CkMessageDetailRecorder() {fclose(f);}
private:
  virtual CmiBool process(envelope *env, CkCoreState *ck) {
    bool wasPacked = env->isPacked();
    if (!wasPacked) CkPackMessage(&env);
    CmiUInt4 size = env->getTotalsize();
    fwrite(&size, 4, 1, f);
    fwrite(env, env->getTotalsize(), 1, f);
    if (!wasPacked) CkUnpackMessage(&env);
    return CmiTrue;
  }
};

//#define REPLAYDEBUG(args) ckout<<"["<<CkMyPe()<<"] "<< args <<endl;
#define REPLAYDEBUG(args) /* empty */

extern "C" void CkMessageReplayQuiescence(void *rep, double time);

class CkMessageReplay : public CkMessageWatcher {
  int counter;
	int nextPE, nextSize, nextEvent, nexttype; //Properties of next message we need:
	unsigned int crc1, crc2;
	/// Read the next message we need from the file:
	void getNext(void) {
		if (6!=fscanf(f,"%d%d%d%d%x%x", &nextPE,&nextSize,&nextEvent,&nexttype,&crc1,&crc2)) {
			// CkAbort("CkMessageReplay> Syntax error reading replay file");
			nextPE=nextSize=nextEvent=nexttype=-1; //No destructor->record file just ends in the middle!
		}
		counter++;
	}
	/// If this is the next message we need, advance and return CmiTrue.
	CmiBool isNext(envelope *env) {
		if (nextPE!=env->getSrcPe()) return CmiFalse;
		if (nextEvent!=env->getEvent()) return CmiFalse;
		if (nextSize!=env->getTotalsize())
                {
			CkPrintf("CkMessageReplay> Message size changed during replay org: [%d %d %d] got: [%d %d %d]\n", nextPE, nextEvent, nextSize, env->getSrcPe(), env->getEvent(), env->getTotalsize());
                        return CmiFalse;
                }
		bool wasPacked = env->isPacked();
		if (!wasPacked) CkPackMessage(&env);
		//unsigned int crcnew = crc32_initial(((unsigned char*)env)+CmiMsgHeaderSizeBytes, env->getTotalsize()-CmiMsgHeaderSizeBytes);
		unsigned int crcnew1 = crc32_initial(((unsigned char*)env)+CmiMsgHeaderSizeBytes, sizeof(*env)-CmiMsgHeaderSizeBytes);
		unsigned int crcnew2 = crc32_initial(((unsigned char*)env)+sizeof(*env), env->getTotalsize()-sizeof(*env));
		if (crcnew1 != crc1) {
		  CkPrintf("CkMessageReplay %d> Envelope CRC changed during replay org: [0x%x] got: [0x%x]\n",CkMyPe(),crc1,crcnew1);
		}
        if (crcnew2 != crc2) {
          CkPrintf("CkMessageReplay %d> Message CRC changed during replay org: [0x%x] got: [0x%x]\n",CkMyPe(),crc2,crcnew2);
        }
        if (!wasPacked) CkUnpackMessage(&env);
		return CmiTrue;
	}

	/// This is a (short) list of messages we aren't yet ready for:
	CkQ<envelope *> delayed;

	/// Try to flush out any delayed messages
	void flush(void) {
		int len=delayed.length();
		for (int i=0;i<len;i++) {
			envelope *env=delayed.deq();
			if (isNext(env)) { /* this is the next message: process it */
				REPLAYDEBUG("Dequeueing message: "<<env->getSrcPe()<<" "<<env->getTotalsize()<<" "<<env->getEvent())
				CmiSyncSendAndFree(CkMyPe(),env->getTotalsize(),(char *)env);
				return;
			}
			else /* Not ready yet-- put it back in the
				queue */
			  {
				REPLAYDEBUG("requeueing delayed message: "<<env->getSrcPe()<<" "<<env->getTotalsize()<<" "<<env->getEvent())
				delayed.enq(env);
			  }
		}
	}

public:
	CkMessageReplay(FILE *f_) {
	  counter=0;
	  f=f_;
	  getNext();
	  REPLAYDEBUG("Constructing ckMessageReplay: "<< nextPE <<" "<< nextSize <<" "<<nextEvent);
	  CmiStartQD(CkMessageReplayQuiescence, this);
	}
	~CkMessageReplay() {fclose(f);}

private:
	virtual CmiBool process(envelope *env,CkCoreState *ck) {
	  REPLAYDEBUG("ProcessMessage message: "<<env->getSrcPe()<<" "<<env->getTotalsize()<<" "<<env->getEvent() <<" " <<env->getMsgtype() <<" " <<env->getMsgIdx());
                if (env->getEvent() == 0) return CmiTrue;
		if (isNext(env)) { /* This is the message we were expecting */
			REPLAYDEBUG("Executing message: "<<env->getSrcPe()<<" "<<env->getTotalsize()<<" "<<env->getEvent())
			getNext(); /* Advance over this message */
			flush(); /* try to process queued-up stuff */
			return CmiTrue;
		}
#if CMK_SMP
                else if (env->getMsgtype()==NodeBocInitMsg || env->getMsgtype()==ForNodeBocMsg) {
                         // try next rank, we can't just buffer the msg and left
                         // we need to keep unprocessed msg on the fly
                        int nextpe = CkMyPe()+1;
                        if (nextpe == CkNodeFirst(CkMyNode())+CkMyNodeSize())
                        nextpe = CkNodeFirst(CkMyNode());
                        CmiSyncSendAndFree(nextpe,env->getTotalsize(),(char *)env);
                        return CmiFalse;
                }
#endif
		else /*!isNext(env) */ {
			REPLAYDEBUG("Queueing message: "<<env->getSrcPe()<<" "<<env->getTotalsize()<<" "<<env->getEvent()
				<<" because we wanted "<<nextPE<<" "<<nextSize<<" "<<nextEvent)
			delayed.enq(env);
                        flush();
			return CmiFalse;
		}
	}
};

extern "C" void CkMessageReplayQuiescence(void *rep, double time) {
  CkPrintf("[%d] Quiescence detected\n",CkMyPe());
  CkMessageReplay *replay = (CkMessageReplay*)rep;
  
}

#include "trace-common.h" /* For traceRoot and traceRootBaseLength */

static FILE *openReplayFile(const char *prefix, const char *suffix, const char *permissions) {

	int i;
	char *fName = new char[CkpvAccess(traceRootBaseLength)+strlen(prefix)+strlen(suffix)+7];
	strncpy(fName, CkpvAccess(traceRoot), CkpvAccess(traceRootBaseLength));
	sprintf(fName+CkpvAccess(traceRootBaseLength), "%s%06d%s",prefix,CkMyPe(),suffix);
	FILE *f=fopen(fName,permissions);
	REPLAYDEBUG("openReplayfile "<<fName);
	if (f==NULL) {
		CkPrintf("[%d] Could not open replay file '%s' with permissions '%w'\n",
			CkMyPe(),fName,permissions);
		CkAbort("openReplayFile> Could not open replay file");
	}
	return f;
}

#include "ckliststring.h"
void CkMessageWatcherInit(char **argv,CkCoreState *ck) {
    char *procs = NULL;
	REPLAYDEBUG("CkMessageWaterInit ");
    if (CmiGetArgStringDesc(argv,"+record-detail",&procs,"Record full message content for the specified processors")) {
        CkListString list(procs);
        if (list.includes(CkMyPe())) {
          CpdSetInitializeMemory(1);
          ck->addWatcher(new CkMessageDetailRecorder(openReplayFile("ckreplay_",".detail","w")));
        }
    }
	if (CmiGetArgFlagDesc(argv,"+record","Record message processing order")) {
	    CpdSetInitializeMemory(1);
		ck->addWatcher(new CkMessageRecorder(openReplayFile("ckreplay_",".log","w")));
	}
	if (CmiGetArgFlagDesc(argv,"+replay","Re-play recorded message stream")) {
	    CpdSetInitializeMemory(1);
		ck->addWatcher(new CkMessageReplay(openReplayFile("ckreplay_",".log","r")));
	}
	if (CmiGetArgStringDesc(argv,"+replay-detail",&procs,"Re-play the specified processors from recorded message content")) {
	    CpdSetInitializeMemory(1);
	  /*Nothing yet*/
	}
}

extern "C"
int CkMessageToEpIdx(void *msg) {
        envelope *env=UsrToEnv(msg);
	int ep=env->getEpIdx();
	if (ep==CkIndex_CkArray::recvBroadcast(0))
		return env->getsetArrayBcastEp();
	else
		return ep;
}

extern "C"
int getCharmEnvelopeSize() {
  return sizeof(envelope);
}


#include "CkMarshall.def.h"

