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
  ckEnableTracing=CmiTrue; 
}
IrrGroup::~IrrGroup() {}

void IrrGroup::pup(PUP::er &p) 
{
  Chare::pup(p);
  p|thisgroup;
  p|ckEnableTracing;
}

NodeGroup::NodeGroup(void) {
  __nodelock=CmiCreateLock();
}
NodeGroup::~NodeGroup() {
  CmiDestroyLock(__nodelock);
}
void NodeGroup::pup(PUP::er &p)
{
  IrrGroup::pup(p);
}

void Group::pup(PUP::er &p) 
{
  CkReductionMgr::pup(p);
  reductionInfo.pup(p);
}

CkComponent::~CkComponent() {}
void CkComponent::pup(PUP::er &p) {}
CkComponent *IrrGroup::ckLookupComponent(int userIndex)
{
	return NULL;
}
CkComponent *CkComponentID::ckLookup(void) const {
	CkComponent *ret=_localBranch(gid)->ckLookupComponent(index);
#ifndef CMK_OPTIMIZE
	if (ret==NULL) CkAbort("Group returned a NULL component!\n");
#endif
	return ret;
}
void CkComponentID::pup(PUP::er &p)
{
	p|gid;
	p|index;
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

CpvDeclare(char **,Ck_argv);
extern "C" char **CkGetArgv(void) {
	return CpvAccess(Ck_argv);
}
extern "C" int CkGetArgc(void) {
	return CmiGetArgc(CpvAccess(Ck_argv));
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
  CkpvAccess(_groupTable).find(groupID).setObj(obj);
  PtrQ *ptrq = CkpvAccess(_groupTable).find(groupID).getPending();
  if(ptrq) {
    void *pending;
    while((pending=ptrq->deq())!=0)
      CldEnqueue(CkMyPe(), pending, _infoIdx);
    delete ptrq;
  }
  CkpvAccess(_currentGroup) = groupID;
  _SET_USED(UsrToEnv(msg), 0);
  _entryTable[eIdx]->call(msg, obj);
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
  CkpvAccess(_currentGroup) = groupID;
  _SET_USED(UsrToEnv(msg), 0);
  _entryTable[eIdx]->call(msg, obj);
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
    CmiSyncBroadcast(env->getTotalsize(), (char *)env);
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
    CmiSyncNodeBroadcast(env->getTotalsize(), (char *)env);
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
  CmiLock(_nodeLock);                           // change for proc 0 and other processors
  if(CkMyNode() == 0)				// should this be CkMyPe() or CkMyNode()?
          groupNum.idx = _numNodeGroups++;
   else
          groupNum.idx = _getGroupIdx(CkNumNodes(),CkMyNode(),_numNodeGroups++);
  CmiUnlock(_nodeLock);
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
  CmiSyncSendAndFree(srcPe, ret->getTotalsize(), (char *)ret);
  CpvAccess(_qd)->create();
  register void *msg = EnvToUsr(env);
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[env->getEpIdx()]->call(msg, obj);
  _TRACE_END_EXECUTE();
}

/************** Message Receive ****************/

static inline void _deliverForChareMsg(int epIdx,envelope *env,void *obj)
{
  register void *msg = EnvToUsr(env);
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[epIdx]->call(msg, obj);
  _TRACE_END_EXECUTE();
}


static inline void _processForChareMsg(envelope *env)
{
  register int epIdx = env->getEpIdx();
  register void *obj = env->getObjPtr();
  _deliverForChareMsg(epIdx,env,obj);
}

static inline void _deliverForBocMsg(int epIdx,envelope *env,IrrGroup *obj)
{
  CmiBool tracingEnabled=obj->ckTracingEnabled();
  if (tracingEnabled) _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[epIdx]->call(EnvToUsr(env), obj);
  if (tracingEnabled) _TRACE_END_EXECUTE();
  _STATS_RECORD_PROCESS_BRANCH_1();  
}

static inline void _processForBocMsg(envelope *env)
{
  register CkGroupID groupID =  env->getGroupNum();
  register IrrGroup *obj = _localBranch(groupID);
  if(!obj) { // groupmember not yet created
    CkpvAccess(_groupTable).find(groupID).enqMsg(env);
    return;
  }
  CpvAccess(_qd)->process();
  register int epIdx = env->getEpIdx();
  _deliverForBocMsg(epIdx,env,obj);
}

static inline void _deliverForNodeBocMsg(envelope *env,void *obj)
{
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  _processForChareMsg(env);
  _STATS_RECORD_PROCESS_NODE_BRANCH_1();
}

static inline void _deliverForNodeBocMsg(int epIdx, envelope *env,void *obj)
{
  env->setEpIdx(epIdx);
  _deliverForNodeBocMsg(env, obj);
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
  _processForChareMsg(env);
  _STATS_RECORD_PROCESS_NODE_BRANCH_1();
}

static inline void _processFillVidMsg(envelope *env)
{
  register VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _CHECK_VALID(vptr, "FillVidMsg: Not a valid VIdPtr\n");
  register CkChareID *pcid = (CkChareID *) EnvToUsr(env);
  _CHECK_VALID(pcid, "FillVidMsg: Not a valid pCid\n");
  vptr->fill(pcid->onPE, pcid->objPtr);
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

void _processHandler(void *converseMsg)
{
  register envelope *env = (envelope *) converseMsg;
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
    _deliverForChareMsg(entryIndex,env,pCid->objPtr);
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
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForBocMsg);
  if(pe==CLD_BROADCAST_ALL) {
    _TRACE_CREATION_N(env, CkNumPes());
  } else {
    _TRACE_CREATION_1(env);
  }
  _skipCldEnqueue(pe, env, _infoIdx);
}

static inline void _sendMsgBranchMulti(int eIdx, void *msg, CkGroupID gID, 
                           int npes, int *pes)
{
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForBocMsg);
  _TRACE_CREATION_N(env, npes);
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
void CkSendMsgBranchInline(int eIdx, void *msg, int destPE, CkGroupID gID)
{
  if (destPE==CkMyPe()) 
  { 
    IrrGroup *obj=(IrrGroup *)_localBranch(gID);
    if (obj!=NULL) 
    { //Just directly call the group:
      envelope *env=UsrToEnv(msg);
      _deliverForBocMsg(eIdx,env,obj);
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
    env->setImmediate(CmiFalse);
    env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForBocMsg);
    if(destPE==CLD_BROADCAST_ALL) {
      _TRACE_CREATION_N(env, CkNumPes());
    } else {
      _TRACE_CREATION_1(env);
    }
    _noCldEnqueue(destPE, env);
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
  register envelope *env = _prepareMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
  if(node==CLD_BROADCAST_ALL) {
    _TRACE_CREATION_N(env, CkNumNodes());
  } else {
    _TRACE_CREATION_1(env);
  }
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
void CkSendMsgNodeBranchInline(int eIdx, void *msg, int node, CkGroupID gID)
{
  if (node==CkMyNode()) 
  { 
    CmiLock(_nodeLock);
    void *obj = _nodeGroupTable->find(gID).getObj();
    CmiUnlock(_nodeLock);
    if (obj!=NULL) 
    { //Just directly call the group:
      envelope *env=UsrToEnv(msg);
      _deliverForNodeBocMsg(eIdx,env,obj);
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
    env->setImmediate(CmiFalse);
    env = _prepareImmediateMsgBranch(eIdx,msg,gID,ForNodeBocMsg);
    if(node==CLD_BROADCAST_ALL) {
      _TRACE_CREATION_N(env, CkNumNodes());
    } else {
      _TRACE_CREATION_1(env);
    }
    _noCldNodeEnqueue(node, env);
    _STATS_RECORD_SEND_BRANCH_1();
    CpvAccess(_qd)->create();
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

#include "CkMarshall.def.h"

