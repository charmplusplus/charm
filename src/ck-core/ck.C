/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"
#include "trace.h"

VidBlock::VidBlock() { state = UNFILLED; msgQ = new PtrQ(); _MEMCHECK(msgQ); }

int CMessage_CkMessage::__idx=-1;
int CMessage_CkArgMsg::__idx=0;
int CkIndex_Chare::__idx;
int CkIndex_Group::__idx;
int CkIndex_ArrayBase::__idx=-1;

//Charm++ virtual functions: declaring these here results in a smaller executable
Chare::Chare(void) {
  thishandle.onPE=CkMyPe();
  thishandle.objPtr=this;
}
Chare::~Chare() {}

void Chare::pup(PUP::er &p)
{
  p(thishandle.onPE);
  thishandle.objPtr=(void *)this;
}

IrrGroup::IrrGroup(void) {
  thisgroup = CkpvAccess(_currentGroup);
}
IrrGroup::~IrrGroup() {}

void IrrGroup::pup(PUP::er &p)
{
  Chare::pup(p);
  p|thisgroup;
}

void IrrGroup::ckJustMigrated(void)
{
}

void Group::pup(PUP::er &p)
{
  CkReductionMgr::pup(p);
  reductionInfo.pup(p);
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
      p|isNodeGroup;
      if (p.isUnpacking()) {
	if (!delegatedTo.isZero()) {
//	  isNodeGroup? ckNodeDelegate(delegatedTo): ckDelegate(delegatedTo);
	  if (isNodeGroup)
		delegatedMgr=(CkDelegateMgr *)CkLocalNodeBranch(delegatedTo);
	  else
		delegatedMgr=(CkDelegateMgr *)CkLocalBranch(delegatedTo);
	}
      }
}

CkDelegateMgr::~CkDelegateMgr() { }

//Default delegator implementation: do not delegate-- send directly
void CkDelegateMgr::ChareSend(int ep,void *m,const CkChareID *c,int onPE)
  { CkSendMsg(ep,m,c); }
void CkDelegateMgr::GroupSend(int ep,void *m,int onPE,CkGroupID g)
  { CkSendMsgBranch(ep,m,onPE,g); }
void CkDelegateMgr::GroupBroadcast(int ep,void *m,CkGroupID g)
  { CkBroadcastMsgBranch(ep,m,g); }
void CkDelegateMgr::NodeGroupSend(int ep,void *m,int onNode,CkNodeGroupID g)
  { CkSendMsgNodeBranch(ep,m,onNode,g); }
void CkDelegateMgr::NodeGroupBroadcast(int ep,void *m,CkNodeGroupID g)
  { CkBroadcastMsgNodeBranch(ep,m,g); }
void CkDelegateMgr::ArrayCreate(int ep,void *m,const CkArrayIndexMax &idx,int onPE,CkArrayID a)
{
	CProxyElement_ArrayBase ap(a,idx);
	ap.ckInsert((CkArrayMessage *)m,ep,onPE);
}
void CkDelegateMgr::ArraySend(int ep,void *m,const CkArrayIndexMax &idx,CkArrayID a)
{
	CProxyElement_ArrayBase ap(a,idx);
	ap.ckSend((CkArrayMessage *)m,ep);
}
void CkDelegateMgr::ArrayBroadcast(int ep,void *m,CkArrayID a)
{
	CProxy_ArrayBase ap(a);
	ap.ckBroadcast((CkArrayMessage *)m,ep);
}

void CkDelegateMgr::ArraySectionSend(int ep,void *m, CkArrayID a,CkSectionCookie &s)
{
	CmiAbort("ArraySectionSend is not implemented!\n");
/*
	CProxyElement_ArrayBase ap(a,idx);
	ap.ckSend((CkArrayMessage *)m,ep);
*/
}

CkSectionID::CkSectionID(const CkArrayID &aid, const CkArrayIndexMax *elems, const int nElems): _nElems(nElems) {
  _cookie.aid = aid;
  _cookie.pe = CkMyPe();
  _elems = new CkArrayIndexMax[nElems];
  for (int i=0; i<nElems; i++) _elems[i] = elems[i];
}

CkSectionID::CkSectionID(const CkSectionID &sid) {
  _cookie = sid._cookie;
  _nElems = sid._nElems;
  _elems = new CkArrayIndexMax[_nElems];
  for (int i=0; i<_nElems; i++) _elems[i] = sid._elems[i];
}

void CkSectionID::operator=(const CkSectionID &sid) {
  _cookie = sid._cookie;
  _nElems = sid._nElems;
  _elems = new CkArrayIndexMax[_nElems];
  for (int i=0; i<_nElems; i++) _elems[i] = sid._elems[i];
}

CkSectionID::~CkSectionID() { delete [] _elems; }

void CkSectionID::pup(PUP::er &p) {
    p | _cookie;
    p(_nElems);
    if (p.isUnpacking()) _elems = new CkArrayIndexMax[_nElems];
    for (int i=0; i< _nElems; i++) p | _elems[i];
}

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

extern "C"
void *CkLocalNodeBranch(CkGroupID groupID)
{
  CmiLock(CksvAccess(_nodeLock));
  void *retval = CksvAccess(_nodeGroupTable)->find(groupID).getObj();
  CmiUnlock(CksvAccess(_nodeLock));
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

CpvDeclare(char **,Ck_argv);
extern "C" char **CkGetArgv(void) {
	return CpvAccess(Ck_argv);
}
extern "C" int CkGetArgc(void) {
	return CmiGetArgc(CpvAccess(Ck_argv));
}

/******************** Basic support *****************/
extern "C" void CkDeliverMessageFree(int epIdx,void *msg,void *obj)
{
  _entryTable[epIdx]->call(msg, obj);
  if (_entryTable[epIdx]->noKeep)
  { /* Method doesn't keep/delete the message, so we have to: */
     CkFreeMsg(msg);
  }
}
extern "C" void CkDeliverMessageReadonly(int epIdx,const void *msg,void *obj)
{
  void *deliverMsg;
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
  _entryTable[epIdx]->call(deliverMsg, obj);
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
  _TRACE_CREATION_1(env);
  if(pCid == 0) {
    env->setMsgtype(NewChareMsg);
  } else {
    pCid->onPE = (-(CkMyPe()+1));
    //  pCid->magic = _GETIDX(cIdx);
    pCid->objPtr = (void *) new VidBlock();
    _MEMCHECK(pCid->objPtr);
    env->setMsgtype(NewVChareMsg);
    env->setVidPtr(pCid->objPtr);
  }
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _charmHandlerIdx);
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
  CkpvAccess(_groupTable)->find(groupID).setObj(obj);
  CkpvAccess(_groupTable)->find(groupID).setDefCtor(_chareTable[gIdx]->defCtor);
  CkpvAccess(_groupTable)->find(groupID).setMigCtor(_chareTable[gIdx]->migCtor);
  CkpvAccess(_groupTable)->find(groupID).setName(_chareTable[gIdx]->name);
  CkpvAccess(_groupIDTable)->push_back(groupID);
  PtrQ *ptrq = CkpvAccess(_groupTable)->find(groupID).getPending();
  if(ptrq) {
    void *pending;
    while((pending=ptrq->deq())!=0)
      CldEnqueue(CkMyPe(), pending, _infoIdx);
    delete ptrq;
  }

  CkpvAccess(_currentGroup) = groupID;
  CkpvAccess(_currentGroupRednMgr) = env->getRednMgr();
  _invokeEntryNoTrace(epIdx,env,obj); /* can't trace groups: would cause nested begin's */
  _STATS_RECORD_PROCESS_GROUP_1();
}

void CkCreateLocalNodeGroup(CkGroupID groupID, int epIdx, envelope *env)
{
  register int gIdx = _entryTable[epIdx]->chareIdx;
  register void *obj = malloc(_chareTable[gIdx]->size);
  _MEMCHECK(obj);
  CmiLock(CksvAccess(_nodeLock));
  CksvAccess(_nodeGroupTable)->find(groupID).setObj(obj);
  CksvAccess(_nodeGroupTable)->find(groupID).setDefCtor(_chareTable[gIdx]->defCtor);
  CksvAccess(_nodeGroupTable)->find(groupID).setMigCtor(_chareTable[gIdx]->migCtor);
  CksvAccess(_nodeGroupTable)->find(groupID).setName(_chareTable[gIdx]->name);
  CksvAccess(_nodeGroupIDTable).push_back(groupID);
  CmiUnlock(CksvAccess(_nodeLock));
  PtrQ *ptrq = CksvAccess(_nodeGroupTable)->find(groupID).getPending();
  if(ptrq) {
    void *pending;
    while((pending=ptrq->deq())!=0)
      CldNodeEnqueue(CkMyNode(), pending, _infoIdx);
    delete ptrq;
  }
  CkpvAccess(_currentGroup) = groupID;
  _invokeEntryNoTrace(epIdx,env,obj);
  _STATS_RECORD_PROCESS_NODE_GROUP_1();
}

void _createGroup(CkGroupID groupID, envelope *env)
{
  _CHECK_USED(env);
  _SET_USED(env, 1);
  register int epIdx = env->getEpIdx();
  register int msgIdx = env->getMsgIdx();
  CkNodeGroupID rednMgr = CProxy_CkArrayReductionMgr::ckNew();
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
  register int msgIdx = env->getMsgIdx();
  env->setGroupNum(groupID);
  env->setSrcPe(CkMyPe());
  env->setGroupEpoch(CkpvAccess(_charmEpoch));
  register void *msg =  EnvToUsr(env);
  if(CkNumNodes()>1) {
    CkPackMessage(&env);
    CmiSetHandler(env, _bocHandlerIdx);
    _numInitMsgs++;
    CksvAccess(_numInitNodeMsgs)++;
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
  CmiLock(CksvAccess(_nodeLock));                // change for proc 0 and other processors
  if(CkMyNode() == 0)				// should this be CkMyPe() or CkMyNode()?
          groupNum.idx = CksvAccess(_numNodeGroups)++;
   else
          groupNum.idx = _getGroupIdx(CkNumNodes(),CkMyNode(),CksvAccess(_numNodeGroups)++);
  CmiUnlock(CksvAccess(_nodeLock));
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
        idx |= 0x80000000;                                      // set the most significant bit to 1
								// if int is not 32 bits, wouldn't this be wrong?
        return idx;
}

extern "C"
CkGroupID CkCreateGroup(int cIdx, int eIdx, void *msg)
{
  CkAssert(cIdx == _entryTable[eIdx]->chareIdx);
  register envelope *env = UsrToEnv(msg);
  _TRACE_CREATION_N(env, CkNumPes());
  env->setMsgtype(BocInitMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  CkGroupID gid = _groupCreate(env);
  _TRACE_CREATION_DONE(CkNumPes());
  return gid;
}

extern "C"
CkGroupID CkCreateNodeGroup(int cIdx, int eIdx, void *msg)
{
  CkAssert(cIdx == _entryTable[eIdx]->chareIdx);
  register envelope *env = UsrToEnv(msg);
  _TRACE_CREATION_N(env, CkNumNodes());
  env->setMsgtype(NodeBocInitMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  CkGroupID gid = _nodeGroupCreate(env);
  _TRACE_CREATION_DONE(CkNumNodes());
  return gid;
}

static inline void *_allocNewChare(envelope *env)
{
  void *tmp=malloc(_chareTable[_entryTable[env->getEpIdx()]->chareIdx]->size);
  _MEMCHECK(tmp);
  return tmp;
}
 
static void _processNewChareMsg(CkCoreState *ck,envelope *env)
{
  register void *obj = _allocNewChare(env);
  _invokeEntry(env->getEpIdx(),env,obj);
}

static void _processNewVChareMsg(CkCoreState *ck,envelope *env)
{
  register void *obj = _allocNewChare(env);
  register CkChareID *pCid = (CkChareID *) 
      _allocMsg(FillVidMsg, sizeof(CkChareID));
  pCid->onPE = CkMyPe();
  pCid->objPtr = obj;
  // pCid->magic = _GETIDX(_entryTable[env->getEpIdx()]->chareIdx);
  register envelope *ret = UsrToEnv(pCid);
  ret->setVidPtr(env->getVidPtr());
  register int srcPe = env->getSrcPe();
  ret->setSrcPe(CkMyPe());
  CmiSetHandler(ret, _charmHandlerIdx);
  CmiSyncSendAndFree(srcPe, ret->getTotalsize(), (char *)ret);
  CpvAccess(_qd)->create();
  _invokeEntry(env->getEpIdx(),env,obj);
}

/************** Message Receive ****************/

static inline void _processForChareMsg(CkCoreState *ck,envelope *env)
{
  register int epIdx = env->getEpIdx();
  register void *obj = env->getObjPtr();
  _invokeEntry(epIdx,env,obj);
}

static inline void _deliverForBocMsg(CkCoreState *ck,int epIdx,envelope *env,IrrGroup *obj)
{
  _invokeEntry(epIdx,env,obj);
  _STATS_RECORD_PROCESS_BRANCH_1();  
}

static inline void _processForBocMsg(CkCoreState *ck,envelope *env)
{
  register CkGroupID groupID =  env->getGroupNum();
  register IrrGroup *obj = ck->localBranch(groupID);
  if(!obj) { // groupmember not yet created
    ck->getGroupTable()->find(groupID).enqMsg(env);
    return;
  }
  ck->process();
  register int epIdx = env->getEpIdx();
  _deliverForBocMsg(ck,epIdx,env,obj);
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
  CmiLock(CksvAccess(_nodeLock));
  obj = CksvAccess(_nodeGroupTable)->find(groupID).getObj();
  if(!obj) { // groupmember not yet created
    CksvAccess(_nodeGroupTable)->find(groupID).enqMsg(env);
    CmiUnlock(CksvAccess(_nodeLock));
    return;
  }
  CmiUnlock(CksvAccess(_nodeLock));
  ck->process();
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  _processForChareMsg(ck,env);
  _STATS_RECORD_PROCESS_NODE_BRANCH_1();
}

static inline void _processFillVidMsg(CkCoreState *ck,envelope *env)
{
  register VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _CHECK_VALID(vptr, "FillVidMsg: Not a valid VIdPtr\n");
  register CkChareID *pcid = (CkChareID *) EnvToUsr(env);
  _CHECK_VALID(pcid, "FillVidMsg: Not a valid pCid\n");
  vptr->fill(pcid->onPE, pcid->objPtr);
  CmiFree(env);
}

static inline void _processForVidMsg(CkCoreState *ck,envelope *env)
{
  VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _CHECK_VALID(vptr, "ForVidMsg: Not a valid VIdPtr\n");
  _SET_USED(env, 1);
  vptr->send(env);
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

/**
 * This is the main converse-level handler used by all of Charm++.
 */
void _processHandler(void *converseMsg,CkCoreState *ck)
{
  register envelope *env = (envelope *) converseMsg;
#if CMK_RECORD_REPLAY
  if (ck->watcher!=NULL) {
    if (!ck->watcher->processMessage(env,ck)) return;
  }
#endif
  switch(env->getMsgtype()) {
    case NewChareMsg :
      ck->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processNewChareMsg(ck,env);
      _STATS_RECORD_PROCESS_CHARE_1();
      break;
    case NewVChareMsg :
      ck->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processNewVChareMsg(ck,env);
      _STATS_RECORD_PROCESS_CHARE_1();
      break;
    case BocInitMsg :
      ck->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processBocInitMsg(ck,env);
      break;
    case NodeBocInitMsg :
      ck->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processNodeBocInitMsg(ck,env);
      break;
    case ForChareMsg :
      ck->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForChareMsg(ck,env);
      _STATS_RECORD_PROCESS_MSG_1();
      break;
    case ForBocMsg :
      // QD processing moved inside _processForBocMsg because it is conditional
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForBocMsg(ck,env);
      // stats record moved inside _processForBocMsg because it is conditional
      break;
    case ForNodeBocMsg :
      // QD processing moved to _processForNodeBocMsg because it is conditional
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForNodeBocMsg(ck,env);
      // stats record moved to _processForNodeBocMsg because it is conditional
      break;
    case ForVidMsg   :
      ck->process();
      _processForVidMsg(ck,env);
      break;
    case FillVidMsg  :
      ck->process();
      _processFillVidMsg(ck,env);
      break;
    default:
      CmiAbort("Internal Error: Unknown-msg-type. Contact Developers.\n");
  }
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
#include "queueing.h"

static int index_skipCldHandler;
static void _skipCldHandler(void *converseMsg)
{
  register envelope *env = (envelope *)(converseMsg);
  CmiSetHandler(converseMsg, CmiGetXHandler(converseMsg));
  CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	env, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
}

static void _skipCldEnqueue(int pe,envelope *env, int infoFn)
{
  if (pe == CkMyPe()) {
    CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	env, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
  } else {
    CkPackMessage(&env);
    int len=env->getTotalsize();
    CmiSetXHandler(env,CmiGetHandler(env));
    CmiSetHandler(env,index_skipCldHandler);
    CmiSetInfo(env,infoFn);
    if (pe==CLD_BROADCAST) { CmiSyncBroadcastAndFree(len, (char *)env); }
    else if (pe==CLD_BROADCAST_ALL) { CmiSyncBroadcastAllAndFree(len, (char *)env); }
    else CmiSyncSendAndFree(pe, len, (char *)env);
  }
}

#if CMK_BLUEGENE_CHARM
#   define  _skipCldEnqueue   CldEnqueue
#endif

static void _noCldEnqueue(int pe, envelope *env)
{
  if (pe == CkMyPe()) {
    CmiHandleMessage(env);
  } else {
    CkPackMessage(&env);
    int len=env->getTotalsize();
    if (pe==CLD_BROADCAST) { CmiSyncBroadcastAndFree(len, (char *)env); }
    else if (pe==CLD_BROADCAST_ALL) { CmiSyncBroadcastAllAndFree(len, (char *)env); }
    else CmiSyncSendAndFree(pe, len, (char *)env);
  }
}

static void _noCldNodeEnqueue(int node, envelope *env)
{
  if (node == CkMyNode()) {
    CmiHandleMessage(env);
  } else {
    CkPackMessage(&env);
    int len=env->getTotalsize();
    if (node==CLD_BROADCAST) { CmiSyncNodeBroadcastAndFree(len, (char *)env); }
    else if (node==CLD_BROADCAST_ALL) { CmiSyncNodeBroadcastAllAndFree(len, (char *)env); }
    else CmiSyncNodeSendAndFree(node, len, (char *)env);
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
  CmiSetHandler(env, _charmHandlerIdx);
  if (pCid->onPE < 0) { //Virtual chare ID (VID)
    register int pe = -(pCid->onPE+1);
    if(pe==CkMyPe()) {
      VidBlock *vblk = (VidBlock *) pCid->objPtr;
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
    CmiSetHandler(env, CpvAccessOther(CmiImmediateMsgHandlerIdx,0));
    CmiSetXHandler(env, _charmHandlerIdx);
  }
  return destPE;
}

extern "C"
void CkSendMsg(int entryIdx, void *msg,const CkChareID *pCid)
{
  register envelope *env = UsrToEnv(msg);
  int destPE=_prepareMsg(entryIdx,msg,pCid);
  if (destPE!=-1) {
    _TRACE_CREATION_1(env);
    CpvAccess(_qd)->create();
    CldEnqueue(destPE, env, _infoIdx);
    _TRACE_CREATION_DONE(1);
  }
}

extern "C"
void CkSendMsgInline(int entryIndex, void *msg, const CkChareID *pCid)
{
  if (pCid->onPE==CkMyPe())
  { //Just directly call the chare (skip QD handling & scheduler)
    register envelope *env = UsrToEnv(msg);
    if (env->isPacked()) CkUnpackMessage(&env);
    _STATS_RECORD_PROCESS_MSG_1();
    _invokeEntryNoTrace(entryIndex,env,pCid->objPtr);
  }
  else {
#if CMK_IMMEDIATE_MSG
    register envelope *env = (envelope *) UsrToEnv(msg);
    if (env->isImmediate()) {
      env->setImmediate(CmiFalse);
      int destPE=_prepareImmediateMsg(entryIndex,msg,pCid);
      // go into VidBlock when destPE is -1
      if (destPE!=-1) {
        _TRACE_CREATION_1(env);
        CpvAccess(_qd)->create();
        _noCldEnqueue(destPE, env);
        _TRACE_CREATION_DONE(1);
      }
    }
    else
#endif
    //No way to inline a cross-processor message:
    CkSendMsg(entryIndex,msg,pCid);
  }
}

static inline envelope *_prepareMsgBranch(int eIdx,void *msg,CkGroupID gID,int type)
{
  register envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  _SET_USED(env, 1);
  env->setMsgtype(type);
  env->setEpIdx(eIdx);
  env->setGroupNum(gID);
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _charmHandlerIdx);
  return env;
}

static inline envelope *_prepareImmediateMsgBranch(int eIdx,void *msg,CkGroupID gID,int type)
{
  envelope *env = _prepareMsgBranch(eIdx, msg, gID, type);
  CmiSetHandler(env, CpvAccessOther(CmiImmediateMsgHandlerIdx,0));
  CmiSetXHandler(env, _charmHandlerIdx);
  return env;
}

static inline void _sendMsgBranch(int eIdx, void *msg, CkGroupID gID,
                           int pe=CLD_BROADCAST_ALL)
{
  int numPes;
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForBocMsg);
  _TRACE_ONLY(numPes = (pe==CLD_BROADCAST_ALL?CkNumPes():1));
  _TRACE_CREATION_N(env, numPes);
  _skipCldEnqueue(pe, env, _infoIdx);
  _TRACE_CREATION_DONE(numPes);
}

static inline void _sendMsgBranchMulti(int eIdx, void *msg, CkGroupID gID,
                           int npes, int *pes)
{
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForBocMsg);
  _TRACE_CREATION_N(env, npes);
  CldEnqueueMulti(npes, pes, env, _infoIdx);
  _TRACE_CREATION_DONE(npes);
}

extern "C"
void CkSendMsgBranch(int eIdx, void *msg, int pe, CkGroupID gID)
{
  _sendMsgBranch(eIdx, msg, gID, pe);
  _STATS_RECORD_SEND_BRANCH_1();
  CpvAccess(_qd)->create();
}

extern "C"
void CkSendMsgBranchInline(int eIdx, void *msg, int destPE, CkGroupID gID)
{
  if (destPE==CkMyPe())
  {
    IrrGroup *obj=(IrrGroup *)_localBranch(gID);
    if (obj!=NULL)
    { //Just directly call the group:
      envelope *env=UsrToEnv(msg);
      _deliverForBocMsg(CkpvAccess(_coreState),eIdx,env,obj);
      return;
    }
#if CMK_IMMEDIATE_MSG
    else {
      envelope *env=UsrToEnv(msg);
      env->setImmediate(CmiFalse);
    }
#endif
  }
  //Can't inline-- send the usual way
#if CMK_IMMEDIATE_MSG
  register envelope *env = UsrToEnv(msg);
  if (env->isImmediate()) {
    int numPes;
    _TRACE_ONLY(numPes = (destPE==CLD_BROADCAST_ALL?CkNumPes():1));
    _TRACE_CREATION_N(env, numPes);
    env->setImmediate(CmiFalse);
    env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForBocMsg);
    _noCldEnqueue(destPE, env);
    _TRACE_CREATION_DONE(numPes);
    _STATS_RECORD_SEND_BRANCH_1();
    CpvAccess(_qd)->create();
  }
  else
#endif
  CkSendMsgBranch(eIdx,msg,destPE,gID);
}

extern "C"
void CkSendMsgBranchMulti(int eIdx,void *msg,int npes,int *pes,CkGroupID gID)
{
  _sendMsgBranchMulti(eIdx, msg, gID, npes, pes);
  _STATS_RECORD_SEND_BRANCH_N(npes);
  CpvAccess(_qd)->create(npes);
}

extern "C"
void CkBroadcastMsgBranch(int eIdx, void *msg, CkGroupID gID)
{
  _sendMsgBranch(eIdx, msg, gID);
  _STATS_RECORD_SEND_BRANCH_N(CkNumPes());
  CpvAccess(_qd)->create(CkNumPes());
}

static inline void _sendMsgNodeBranch(int eIdx, void *msg, CkGroupID gID,
                           int node=CLD_BROADCAST_ALL)
{
  int numPes;
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
  _TRACE_ONLY(numPes = (node==CLD_BROADCAST_ALL?CkNumNodes():1));
  _TRACE_CREATION_N(env, numPes);
  CldNodeEnqueue(node, env, _infoIdx);
  _TRACE_CREATION_DONE(numPes);
}

extern "C"
void CkSendMsgNodeBranch(int eIdx, void *msg, int node, CkGroupID gID)
{
  _sendMsgNodeBranch(eIdx, msg, gID, node);
  _STATS_RECORD_SEND_NODE_BRANCH_1();
  CpvAccess(_qd)->create();
}

extern "C"
void CkSendMsgNodeBranchInline(int eIdx, void *msg, int node, CkGroupID gID)
{
  if (node==CkMyNode()) 
  { 
    CmiLock(CksvAccess(_nodeLock));
    void *obj = CksvAccess(_nodeGroupTable)->find(gID).getObj();
    CmiUnlock(CksvAccess(_nodeLock));
    if (obj!=NULL) 
    { //Just directly call the group:
      envelope *env=UsrToEnv(msg);
      _deliverForNodeBocMsg(CkpvAccess(_coreState),eIdx,env,obj);
      return;
    }
#if CMK_IMMEDIATE_MSG
    else {
      envelope *env=UsrToEnv(msg);
      env->setImmediate(CmiFalse);
    }
#endif
  }
  //Can't inline-- send the usual way
#if CMK_IMMEDIATE_MSG
  register envelope *env = UsrToEnv(msg);
  if (env->isImmediate()) {
    int numPes;
    _TRACE_ONLY(numPes = (node==CLD_BROADCAST_ALL?CkNumNodes():1));
    _TRACE_CREATION_N(env, numPes);
    env->setImmediate(CmiFalse);
    env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
    _noCldNodeEnqueue(node, env);
    _STATS_RECORD_SEND_BRANCH_1();
    CpvAccess(_qd)->create();
    _TRACE_CREATION_DONE(numPes);
  }
  else
#endif
  CkSendMsgNodeBranch(eIdx,msg,node,gID);
}

extern "C"
void CkBroadcastMsgNodeBranch(int eIdx, void *msg, CkGroupID gID)
{
  _sendMsgNodeBranch(eIdx, msg, gID);
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
}

//------------------- Message Watcher (record/replay) ----------------

CkMessageWatcher::~CkMessageWatcher() {}

class CkMessageRecorder : public CkMessageWatcher {
	FILE *f;
public:
	CkMessageRecorder(FILE *f_) :f(f_) {}
	~CkMessageRecorder() {
		fprintf(f,"-1 -1 -1");
		fclose(f);
	}
	
	virtual bool processMessage(envelope *env,CkCoreState *ck) {
		fprintf(f,"%d %d %d\n",env->getSrcPe(),env->getTotalsize(),env->getEvent());
		return true;
	}
};

// #define REPLAYDEBUG(args) ckout<<"["<<CkMyPe()<<"] "<< args <<endl;
#define REPLAYDEBUG(args) /* empty */

class CkMessageReplay : public CkMessageWatcher {
	FILE *f;
	int nextPE, nextSize, nextEvent; //Properties of next message we need:
	/// Read the next message we need from the file:
	void getNext(void) {
		if (3!=fscanf(f,"%d%d%d",&nextPE,&nextSize,&nextEvent)) {
			// CkAbort("CkMessageReplay> Syntax error reading replay file");
			nextPE=nextSize=nextEvent=-1; //No destructor->record file just ends in the middle!
		}
	}
	/// If this is the next message we need, advance and return true.
	bool isNext(envelope *env) {
		if (nextPE!=env->getSrcPe()) return false;
		if (nextEvent!=env->getEvent()) return false;
		if (nextSize!=env->getTotalsize())
			CkAbort("CkMessageReplay> Message size changed during replay");
		return true;
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
			else /* Not ready yet-- put it back in the queue */
				delayed.enq(env);
		}
	}
	
public:
	CkMessageReplay(FILE *f_) :f(f_) { getNext(); }
	~CkMessageReplay() {fclose(f);}
	
	virtual bool processMessage(envelope *env,CkCoreState *ck) {
		if (isNext(env)) { /* This is the message we were expecting */
			REPLAYDEBUG("Executing message: "<<env->getSrcPe()<<" "<<env->getTotalsize()<<" "<<env->getEvent())
			getNext(); /* Advance over this message */
			flush(); /* try to process queued-up stuff */
			return true;
		}
		else /*!isNext(env) */ {
			REPLAYDEBUG("Queueing message: "<<env->getSrcPe()<<" "<<env->getTotalsize()<<" "<<env->getEvent()
				<<" because we wanted "<<nextPE<<" "<<nextSize<<" "<<nextEvent)
			delayed.enq(env);
			return false;
		}
	}
};

static FILE *openReplayFile(const char *permissions) {
	char fName[200];
	sprintf(fName,"ckreplay_%06d.log",CkMyPe());
	FILE *f=fopen(fName,permissions);
	if (f==NULL) {
		CkPrintf("[%d] Could not open replay file '%s' with permissions '%w'\n",
			CkMyPe(),fName,permissions);
		CkAbort("openReplayFile> Could not open replay file");
	}
	return f;
}

void CkMessageWatcherInit(char **argv,CkCoreState *ck) {
	if (CmiGetArgFlagDesc(argv,"+record","Record message processing order"))
		ck->watcher=new CkMessageRecorder(openReplayFile("w"));
	if (CmiGetArgFlagDesc(argv,"+replay","Re-play recorded message stream"))
		ck->watcher=new CkMessageReplay(openReplayFile("r"));
}










#include "CkMarshall.def.h"

