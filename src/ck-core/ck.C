/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"
#include "trace.h"

VidBlock::VidBlock() { state = UNFILLED; msgQ = new PtrQ(); _MEMCHECK(msgQ); }

int CMessage_CkArgMsg::__idx=0;
int CkIndex_Chare::__idx;
int CkIndex_Group::__idx;
int CkIndex_ArrayBase::__idx=-1;

Group::~Group() {}

//Chare virtual functions: declaring these here results in a smaller executable
Chare::~Chare() {}

void Chare::pup(PUP::er &p) 
{
	 p(thishandle.onPE);
	 p(thishandle.magic);
	 thishandle.objPtr=(void *)this;
} 

void Group::pup(PUP::er &p) 
{
	Chare::pup(p);
	p|thisgroup;
} 
CkDelegateMgr::~CkDelegateMgr() { }

//Default delegator implementation: do not delegate-- send directly
void CkDelegateMgr::ChareSend(int ep,void *m,const CkChareID *c)
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
void CkGetChareID(CkChareID *pCid) {
  pCid->onPE = CkMyPe();
  pCid->objPtr = CpvAccess(_currentChare);
  //  pCid->magic = _GETIDX(CpvAccess(_currentChareType));
}

extern "C"
CkGroupID CkGetGroupID(void) {
  return CpvAccess(_currentGroup);
}

extern "C"
CkGroupID CkGetNodeGroupID(void) {
  return CpvAccess(_currentNodeGroup);
}

extern "C"
void *CkLocalBranch(CkGroupID gID) {
  return _localBranch(gID);
}

extern "C"
void *CkLocalNodeBranch(CkGroupID groupID)
{
  CmiLock(_nodeLock);
  void *retval = _nodeGroupTable->find(groupID).getObj();
  CmiUnlock(_nodeLock);
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
}

void _createGroupMember(CkGroupID groupID, int eIdx, void *msg)
{
  register int gIdx = _entryTable[eIdx]->chareIdx;
  register void *obj = malloc(_chareTable[gIdx]->size);
  _MEMCHECK(obj);
  CpvAccess(_groupTable).find(groupID).setObj(obj);
  PtrQ *ptrq = CpvAccess(_groupTable).find(groupID).getPending();
  if(ptrq) {
    void *pending;
    while((pending=ptrq->deq())!=0)
      CldEnqueue(CkMyPe(), pending, _infoIdx);
    delete ptrq;
  }
  register void *prev = CpvAccess(_currentChare);
  register int prevtype = CpvAccess(_currentChareType);
  CpvAccess(_currentChare) = obj;
  CpvAccess(_currentChareType) = gIdx;
  register CkGroupID prevGrp = CpvAccess(_currentGroup);
  CpvAccess(_currentGroup) = groupID;
  _SET_USED(UsrToEnv(msg), 0);
  _entryTable[eIdx]->call(msg, obj);
  CpvAccess(_currentChare) = prev;
  CpvAccess(_currentChareType) = prevtype;
  CpvAccess(_currentGroup) = prevGrp;
  _STATS_RECORD_PROCESS_GROUP_1();
}

void _createNodeGroupMember(CkGroupID groupID, int eIdx, void *msg)
{
  register int gIdx = _entryTable[eIdx]->chareIdx;
  register void *obj = malloc(_chareTable[gIdx]->size);
  _MEMCHECK(obj);
  CmiLock(_nodeLock);
  _nodeGroupTable->find(groupID).setObj(obj);
  CmiUnlock(_nodeLock);
  PtrQ *ptrq = _nodeGroupTable->find(groupID).getPending();
  if(ptrq) {
    void *pending;
    while((pending=ptrq->deq())!=0)
      CldNodeEnqueue(CkMyNode(), pending, _infoIdx);
    delete ptrq;
  }
  register void *prev = CpvAccess(_currentChare);
  register int prevtype = CpvAccess(_currentChareType);
  CpvAccess(_currentChare) = obj;
  CpvAccess(_currentChareType) = gIdx;
  register CkGroupID prevGrp = CpvAccess(_currentNodeGroup);
  CpvAccess(_currentNodeGroup) = groupID;
  _SET_USED(UsrToEnv(msg), 0);
  _entryTable[eIdx]->call(msg, obj);
  CpvAccess(_currentChare) = prev;
  CpvAccess(_currentChareType) = prevtype;
  CpvAccess(_currentNodeGroup) = prevGrp;
  _STATS_RECORD_PROCESS_NODE_GROUP_1();
}

void _createGroup(CkGroupID groupID, envelope *env)
{
  _CHECK_USED(env);
  _SET_USED(env, 1);
  register int epIdx = env->getEpIdx();
  register int msgIdx = env->getMsgIdx();
  env->setGroupNum(groupID);
  env->setSrcPe(CkMyPe());
  register void *msg =  EnvToUsr(env);
  if(CkNumPes()>1) {
    if(!env->isPacked() && _msgTable[msgIdx]->pack) {
      _TRACE_BEGIN_PACK();
      msg = _msgTable[msgIdx]->pack(msg);
      UsrToEnv(msg)->setPacked(1);
      _TRACE_END_PACK();
    }
    env = UsrToEnv(msg);
    CmiSetHandler(env, _bocHandlerIdx);
    _numInitMsgs++;
    CmiSyncBroadcast(env->getTotalsize(), env);
    CpvAccess(_qd)->create(CkNumPes()-1);
    if(env->isPacked() && _msgTable[msgIdx]->unpack) {
      _TRACE_BEGIN_UNPACK();
      msg = _msgTable[msgIdx]->unpack(msg);
      UsrToEnv(msg)->setPacked(0);
      _TRACE_END_UNPACK();
    }
  }
  _STATS_RECORD_CREATE_GROUP_1();
  _createGroupMember(groupID, epIdx, msg);
}

void _createNodeGroup(CkGroupID groupID, envelope *env)
{
  _CHECK_USED(env);
  _SET_USED(env, 1);
  register int epIdx = env->getEpIdx();
  register int msgIdx = env->getMsgIdx();
  env->setGroupNum(groupID);
  env->setSrcPe(CkMyPe());
  register void *msg =  EnvToUsr(env);
  if(CkNumNodes()>1) {
    if(!env->isPacked() && _msgTable[msgIdx]->pack) {
      _TRACE_BEGIN_PACK();
      msg = _msgTable[msgIdx]->pack(msg);
      UsrToEnv(msg)->setPacked(1);
      _TRACE_END_PACK();
    }
    env = UsrToEnv(msg);
    CmiSetHandler(env, _bocHandlerIdx);
    _numInitMsgs++;
    _numInitNodeMsgs++;
    CmiSyncNodeBroadcast(env->getTotalsize(), env);
    CpvAccess(_qd)->create(CkNumNodes()-1);
    if(env->isPacked() && _msgTable[msgIdx]->unpack) {
      _TRACE_BEGIN_UNPACK();
      msg = _msgTable[msgIdx]->unpack(msg);
      UsrToEnv(msg)->setPacked(0);
      _TRACE_END_UNPACK();
    }
  }
  _STATS_RECORD_CREATE_NODE_GROUP_1();
  _createNodeGroupMember(groupID, epIdx, msg);
}

static CkGroupID _groupCreate(envelope *env)
{
  register CkGroupID groupNum;
  groupNum.pe = CkMyPe();
  groupNum.idx = CpvAccess(_numGroups)++;
  _createGroup(groupNum, env);
  return groupNum;
}

static CkGroupID _nodeGroupCreate(envelope *env)
{
  register CkGroupID groupNum;
  groupNum.pe = CkMyNode();
  CmiLock(_nodeLock);
  groupNum.idx = _numNodeGroups++;
  CmiUnlock(_nodeLock);
  _createNodeGroup(groupNum, env);
  return groupNum;
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
  return _groupCreate(env);
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
  return _nodeGroupCreate(env);
}


static inline void *_allocNewChare(envelope *env)
{
  void *tmp=malloc(_chareTable[_entryTable[env->getEpIdx()]->chareIdx]->size);
  _MEMCHECK(tmp);
  return tmp;
}
 
static void _processNewChareMsg(envelope *env)
{
  register void *obj = _allocNewChare(env);
  register void *msg = EnvToUsr(env);
  CpvAccess(_currentChare) = obj;
  CpvAccess(_currentChareType)=_entryTable[env->getEpIdx()]->chareIdx;
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[env->getEpIdx()]->call(msg, obj);
  _TRACE_END_EXECUTE();
}

static void _processNewVChareMsg(envelope *env)
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
  CmiSyncSendAndFree(srcPe, ret->getTotalsize(), ret);
  CpvAccess(_qd)->create();
  CpvAccess(_currentChare) = obj;
  CpvAccess(_currentChareType)=_entryTable[env->getEpIdx()]->chareIdx;
  register void *msg = EnvToUsr(env);
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[env->getEpIdx()]->call(msg, obj);
  _TRACE_END_EXECUTE();
}

/************** Message Receive ****************/

static inline void _processForChareMsg(envelope *env)
{
  register void *msg = EnvToUsr(env);
  register int epIdx = env->getEpIdx();
  register void *obj = env->getObjPtr();
  CpvAccess(_currentChare) = obj;
  CpvAccess(_currentChareType)=_entryTable[epIdx]->chareIdx;
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[epIdx]->call(msg, obj);
  _TRACE_END_EXECUTE();
}

static inline void _processForBocMsg(envelope *env)
{
  register CkGroupID groupID =  env->getGroupNum();
  register void *obj = _localBranch(groupID);
  if(!obj) { // groupmember not yet created
    CpvAccess(_groupTable).find(groupID).enqMsg(env);
    return;
  }
  CpvAccess(_qd)->process();
  CpvAccess(_currentGroup) = groupID;
  register int epIdx = env->getEpIdx();
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[epIdx]->call(EnvToUsr(env), obj);
  _TRACE_END_EXECUTE();
  _STATS_RECORD_PROCESS_BRANCH_1();
}

static inline void _processForNodeBocMsg(envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register void *obj;
  CmiLock(_nodeLock);
  obj = _nodeGroupTable->find(groupID).getObj();
  if(!obj) { // groupmember not yet created
    _nodeGroupTable->find(groupID).enqMsg(env);
    CmiUnlock(_nodeLock);
    return;
  }
  CmiUnlock(_nodeLock);
  CpvAccess(_qd)->process();
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  CpvAccess(_currentNodeGroup) = groupID;
  _processForChareMsg(env);
  _STATS_RECORD_PROCESS_NODE_BRANCH_1();
}

static inline void _processFillVidMsg(envelope *env)
{
  register VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _CHECK_VALID(vptr, "FillVidMsg: Not a valid VIdPtr\n");
  register CkChareID *pcid = (CkChareID *) EnvToUsr(env);
  _CHECK_VALID(pcid, "FillVidMsg: Not a valid pCid\n");
  vptr->fill(pcid->onPE, pcid->objPtr, pcid->magic);
  CmiFree(env);
}

static inline void _processForVidMsg(envelope *env)
{
  VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _CHECK_VALID(vptr, "ForVidMsg: Not a valid VIdPtr\n");
  _SET_USED(env, 1);
  vptr->send(env);
}

void _processBocInitMsg(envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register int epIdx = env->getEpIdx();
  _createGroupMember(groupID, epIdx, EnvToUsr(env));
}

void _processNodeBocInitMsg(envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register int epIdx = env->getEpIdx();
  _createNodeGroupMember(groupID, epIdx, EnvToUsr(env));
}

void _processHandler(void *msg)
{
  register envelope *env = (envelope *) msg;
  switch(env->getMsgtype()) {
    case NewChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processNewChareMsg(env);
      _STATS_RECORD_PROCESS_CHARE_1();
      break;
    case NewVChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processNewVChareMsg(env);
      _STATS_RECORD_PROCESS_CHARE_1();
      break;
    case BocInitMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processBocInitMsg(env);
      break;
    case NodeBocInitMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processNodeBocInitMsg(env);
      break;
    case ForChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForChareMsg(env);
      _STATS_RECORD_PROCESS_MSG_1();
      break;
    case ForBocMsg :
      // QD processing moved inside _processForBocMsg because it is conditional
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForBocMsg(env);
      // stats record moved inside _processForBocMsg because it is conditional
      break;
    case ForNodeBocMsg :
      // QD processing moved to _processForNodeBocMsg because it is conditional
      if(env->isPacked()) CkUnpackMessage(&env);
      _processForNodeBocMsg(env);
      // stats record moved to _processForNodeBocMsg because it is conditional
      break;
    case ForVidMsg   :
      CpvAccess(_qd)->process();
      _processForVidMsg(env);
      break;
    case FillVidMsg  :
      CpvAccess(_qd)->process();
      _processFillVidMsg(env);
      break;
    default:
      CmiAbort("Internal Error: Unknown-msg-type. Contact Developers.\n");
  }
}


/******************** Message Send **********************/

void _infoFn(void *msg, CldPackFn *pfn, int *len,
             int *queueing, int *priobits, unsigned int **prioptr)
{
  register envelope *env = (envelope *)msg;
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
static void _skipCldHandler(void *msg)
{
  register envelope *env = (envelope *)(msg);
  CmiSetHandler(msg, CmiGetXHandler(msg));
  CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	env, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
}

static void _skipCldEnqueue(int pe,envelope *env, int infoFn)
{
  if (pe == CmiMyPe()) {
    CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	env, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
  } else {
    CkPackMessage(&env);
    int len=env->getTotalsize();
    CmiSetXHandler(env,CmiGetHandler(env));
    CmiSetHandler(env,index_skipCldHandler);
    CmiSetInfo(env,infoFn);
    if (pe==CLD_BROADCAST) { CmiSyncBroadcastAndFree(len, env); }
    else if (pe==CLD_BROADCAST_ALL) { CmiSyncBroadcastAllAndFree(len, env); }
    else CmiSyncSendAndFree(pe, len, env);
  }
}

extern "C"
void CkSendMsg(int entryIdx, void *msg,const CkChareID *pCid)
{
  register envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  env->setMsgtype(ForChareMsg);
  env->setEpIdx(entryIdx);
  CmiSetHandler(env, _charmHandlerIdx);
  _SET_USED(env, 1);
  if(pCid->onPE < 0) {
    register int pe = -(pCid->onPE+1);
    if(pe==CkMyPe()) {
      VidBlock *vblk = (VidBlock *) pCid->objPtr;
      vblk->send(env);
    } else {
      env->setMsgtype(ForVidMsg);
      env->setSrcPe(CkMyPe());
      env->setVidPtr(pCid->objPtr);
      _TRACE_CREATION_1(env);
      CpvAccess(_qd)->create();
      CldEnqueue(pe, env, _infoIdx);
    }
  } else {
    env->setSrcPe(CkMyPe());
    env->setObjPtr(pCid->objPtr);
    _TRACE_CREATION_1(env);
    CpvAccess(_qd)->create();
    CldEnqueue(pCid->onPE, env, _infoIdx);
  }
  _STATS_RECORD_SEND_MSG_1();
}

static inline void _sendMsgBranch(int eIdx, void *msg, CkGroupID gID, 
                           int pe=CLD_BROADCAST_ALL)
{
  register envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  _SET_USED(env, 1);
  env->setMsgtype(ForBocMsg);
  env->setEpIdx(eIdx);
  env->setGroupNum(gID);
  env->setSrcPe(CkMyPe());
  if(pe==CLD_BROADCAST_ALL) {
    _TRACE_CREATION_N(env, CkNumPes());
  } else {
    _TRACE_CREATION_1(env);
  }
  CmiSetHandler(env, _charmHandlerIdx);
  _skipCldEnqueue(pe, env, _infoIdx);
}

static inline void _sendMsgBranchMulti(int eIdx, void *msg, CkGroupID gID, 
                           int npes, int *pes)
{
  register envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  _SET_USED(env, 1);
  env->setMsgtype(ForBocMsg);
  env->setEpIdx(eIdx);
  env->setGroupNum(gID);
  env->setSrcPe(CkMyPe());
  _TRACE_CREATION_N(env, npes);
  CmiSetHandler(env, _charmHandlerIdx);
  CldEnqueueMulti(npes, pes, env, _infoIdx);
}

extern "C"
void CkSendMsgBranch(int eIdx, void *msg, int pe, CkGroupID gID)
{
  _sendMsgBranch(eIdx, msg, gID, pe);
  _STATS_RECORD_SEND_BRANCH_1();
  CpvAccess(_qd)->create();
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
  register envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  _SET_USED(env, 1);
  env->setMsgtype(ForNodeBocMsg);
  env->setEpIdx(eIdx);
  env->setGroupNum(gID);
  env->setSrcPe(CkMyPe());
  if(node==CLD_BROADCAST_ALL) {
    _TRACE_CREATION_N(env, CkNumNodes());
  } else {
    _TRACE_CREATION_1(env);
  }
  CmiSetHandler(env, _charmHandlerIdx);
  CldNodeEnqueue(node, env, _infoIdx);
}

extern "C"
void CkSendMsgNodeBranch(int eIdx, void *msg, int node, CkGroupID gID)
{
  _sendMsgNodeBranch(eIdx, msg, gID, node);
  _STATS_RECORD_SEND_NODE_BRANCH_1();
  CpvAccess(_qd)->create();
}

extern "C"
void CkBroadcastMsgNodeBranch(int eIdx, void *msg, CkGroupID gID)
{
  _sendMsgNodeBranch(eIdx, msg, gID);
  _STATS_RECORD_SEND_NODE_BRANCH_N(CkNumNodes());
  CpvAccess(_qd)->create(CkNumNodes());
}

void _ckModuleInit(void) {
	index_skipCldHandler = CmiRegisterHandler((CmiHandler)_skipCldHandler);
}

#include "CkMarshall.def.h"

