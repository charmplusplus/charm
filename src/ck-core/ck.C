#include "ck.h"
#include "trace.h"

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
}

extern "C"
CkGroupID CkGetGroupID(void) {
  return CpvAccess(_currentGroup);
}

extern "C"
CkGroupID CkGetNodeGroupID(void) {
  return CpvAccess(_currentNodeGroup);
}

static inline void *_allocNewChare(envelope *env)
{
  return malloc(_chareTable[_entryTable[env->getEpIdx()]->chareIdx]->size);
}
 
static void _processNewChareMsg(envelope *env)
{
  register void *obj = _allocNewChare(env);
  register void *msg = EnvToUsr(env);
  CpvAccess(_currentChare) = obj;
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
  register envelope *ret = UsrToEnv(pCid);
  ret->setVidPtr(env->getVidPtr());
  register int srcPe = env->getSrcPe();
  ret->setSrcPe(CkMyPe());
  CmiSetHandler(ret, _charmHandlerIdx);
  CmiSyncSendAndFree(srcPe, ret->getTotalsize(), ret);
  CpvAccess(_qd)->create();
  CpvAccess(_currentChare) = obj;
  register void *msg = EnvToUsr(env);
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[env->getEpIdx()]->call(msg, obj);
  _TRACE_END_EXECUTE();
}

static inline void _processForChareMsg(envelope *env)
{
  register void *msg = EnvToUsr(env);
  register int epIdx = env->getEpIdx();
  register void *obj = env->getObjPtr();
  CpvAccess(_currentChare) = obj;
  _TRACE_BEGIN_EXECUTE(env);
  _SET_USED(env, 0);
  _entryTable[epIdx]->call(msg, obj);
  _TRACE_END_EXECUTE();
}

static inline void _processForBocMsg(envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register void *obj = CpvAccess(_groupTable)->find(groupID);
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  CpvAccess(_currentGroup) = groupID;
  _processForChareMsg(env);
}

static inline void _processForNodeBocMsg(envelope *env)
{
  register CkGroupID groupID = env->getGroupNum();
  register void *obj;
  CmiLock(_nodeLock);
  obj = _nodeGroupTable->find(groupID);
  CmiUnlock(_nodeLock);
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  CpvAccess(_currentNodeGroup) = groupID;
  _processForChareMsg(env);
}

static inline void _processFillVidMsg(envelope *env)
{
  register VidBlock *vptr = (VidBlock *) env->getVidPtr();
  register CkChareID *pcid = (CkChareID *) EnvToUsr(env);
  vptr->fill(pcid->onPE, pcid->objPtr);
  CmiFree(env);
}

static inline void _processForVidMsg(envelope *env)
{
  VidBlock *vptr = (VidBlock *) env->getVidPtr();
  _SET_USED(env, 1);
  vptr->send(env);
}

static inline void _processDBocReqMsg(envelope *env)
{
  assert(CkMyPe()==0);
  register CkGroupID groupNum;
  groupNum = _numGroups++;
  env->setMsgtype(DBocNumMsg);
  register int srcPe = env->getSrcPe();
  env->setSrcPe(CkMyPe());
  env->setGroupNum(groupNum);
  CmiSyncSendAndFree(srcPe, env->getTotalsize(), env);
  CpvAccess(_qd)->create();
}

static inline void _processDNodeBocReqMsg(envelope *env)
{
  assert(CkMyNode()==0);
  CmiLock(_nodeLock);
  register CkGroupID groupNum = _numNodeGroups++;
  CmiUnlock(_nodeLock);
  env->setMsgtype(DNodeBocNumMsg);
  register int srcNode = CmiNodeOf(env->getSrcPe());
  env->setSrcPe(CkMyPe());
  env->setGroupNum(groupNum);
  CmiSyncNodeSendAndFree(srcNode, env->getTotalsize(), env);
  CpvAccess(_qd)->create();
}

static inline void _processDBocNumMsg(envelope *env)
{
  register envelope *usrenv = (envelope *) env->getUsrMsg();
  register int retEp = env->getRetEp();
  register CkChareID *retChare = (CkChareID *) EnvToUsr(env);
  register CkGroupID groupID = env->getGroupNum();
  _createGroup(groupID, usrenv, retEp, retChare);
}

static inline void _processDNodeBocNumMsg(envelope *env)
{
  register envelope *usrenv = (envelope *) env->getUsrMsg();
  register int retEp = env->getRetEp();
  register CkChareID *retChare = (CkChareID *) EnvToUsr(env);
  register CkGroupID groupID = env->getGroupNum();
  _createNodeGroup(groupID, usrenv, retEp, retChare);
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
  CmiGrabBuffer(&msg);
  register envelope *env = (envelope *) msg;
  switch(env->getMsgtype()) {
    case NewChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processNewChareMsg(env);
#ifndef CMK_OPTIMIZE
      CpvAccess(_myStats)->recordProcessChare();
#endif
      break;
    case NewVChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processNewVChareMsg(env);
#ifndef CMK_OPTIMIZE
      CpvAccess(_myStats)->recordProcessChare();
#endif
      break;
    case BocInitMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processBocInitMsg(env);
      break;    
    case NodeBocInitMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processNodeBocInitMsg(env);
      break;
    case DBocReqMsg:
      CpvAccess(_qd)->process();
      _processDBocReqMsg(env);
      break;
    case DNodeBocReqMsg:
      CpvAccess(_qd)->process();
      _processDNodeBocReqMsg(env);
      break;
    case DBocNumMsg:
      CpvAccess(_qd)->process();
      _processDBocNumMsg(env);
      break;
    case DNodeBocNumMsg:
      CpvAccess(_qd)->process();
      _processDNodeBocNumMsg(env);
      break;
    case ForChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processForChareMsg(env);
#ifndef CMK_OPTIMIZE
      CpvAccess(_myStats)->recordProcessMsg();
#endif
      break;
    case ForBocMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processForBocMsg(env);
#ifndef CMK_OPTIMIZE
      CpvAccess(_myStats)->recordProcessBranch();
#endif
      break;
    case ForNodeBocMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processForNodeBocMsg(env);
#ifndef CMK_OPTIMIZE
      CpvAccess(_myStats)->recordProcessNodeBranch();
#endif
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

void _infoFn(void *msg, CldPackFn *pfn, int *len,
             int *queueing, int *priobits, unsigned int **prioptr)
{
  register envelope *env = (envelope *)msg;
  *pfn = (CldPackFn)_packFn;
  *len = env->getTotalsize();
  *queueing = env->getQueueing();
  *priobits = env->getPriobits();
  *prioptr = (unsigned int *) env->getPrioPtr();
}

void _packFn(void **pEnv)
{
  register envelope *env = *((envelope **)pEnv);
  register int msgIdx = env->getMsgIdx();
  if(!env->isPacked() && _msgTable[msgIdx]->pack) {
    register void *msg = EnvToUsr(env);
    _TRACE_BEGIN_PACK();
    msg = _msgTable[msgIdx]->pack(msg);
    _TRACE_END_PACK();
    UsrToEnv(msg)->setPacked(1);
    *((envelope **)pEnv) = UsrToEnv(msg);
  }
}

void _unpackFn(void **pEnv)
{
  register envelope *env = *((envelope **)pEnv);
  register int msgIdx = env->getMsgIdx();
  if(_msgTable[msgIdx]->unpack) {
    register void *msg = EnvToUsr(env);
    _TRACE_BEGIN_UNPACK();
    msg = _msgTable[msgIdx]->unpack(msg);
    _TRACE_END_UNPACK();
    UsrToEnv(msg)->setPacked(0);
    *((envelope **)pEnv) = UsrToEnv(msg);
  }
}

extern "C"
void CkSendMsg(int entryIdx, void *msg, CkChareID *pCid)
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
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordSendMsg();
#endif
}

extern "C"
void CkCreateChare(int cIdx, int eIdx, void *msg, CkChareID *pCid, int destPE)
{
  assert(cIdx == _entryTable[eIdx]->chareIdx);
  envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  if(pCid == 0) {
    env->setMsgtype(NewChareMsg);
  } else {
    pCid->onPE = (-(CkMyPe()+1));
    pCid->objPtr = (void *) new VidBlock();
    env->setMsgtype(NewVChareMsg);
    env->setVidPtr(pCid->objPtr);
  }
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _charmHandlerIdx);
  _TRACE_CREATION_1(env);
  CpvAccess(_qd)->create();
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordCreateChare();
#endif
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
  CpvAccess(_groupTable)->add(groupID, obj);
  register void *prev = CpvAccess(_currentChare);
  CpvAccess(_currentChare) = obj;
  register int prevGrp = CpvAccess(_currentGroup);
  CpvAccess(_currentGroup) = groupID;
  _SET_USED(UsrToEnv(msg), 0);
  _entryTable[eIdx]->call(msg, obj);
  CpvAccess(_currentChare) = prev;
  CpvAccess(_currentGroup) = prevGrp;
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordProcessGroup();
#endif
}

void _createNodeGroupMember(CkGroupID groupID, int eIdx, void *msg)
{
  register int gIdx = _entryTable[eIdx]->chareIdx;
  register void *obj = malloc(_chareTable[gIdx]->size);
  CmiLock(_nodeLock);
  _nodeGroupTable->add(groupID, obj);
  CmiUnlock(_nodeLock);
  register void *prev = CpvAccess(_currentChare);
  CpvAccess(_currentChare) = obj;
  register int prevGrp = CpvAccess(_currentNodeGroup);
  CpvAccess(_currentNodeGroup) = groupID;
  _SET_USED(UsrToEnv(msg), 0);
  _entryTable[eIdx]->call(msg, obj);
  CpvAccess(_currentChare) = prev;
  CpvAccess(_currentNodeGroup) = prevGrp;
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordProcessNodeGroup();
#endif
}

void _createGroup(CkGroupID groupID, envelope *env, int retEp, CkChareID *retChare)
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
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordCreateGroup();
#endif
  _createGroupMember(groupID, epIdx, msg);
  if(retEp) {
    msg = CkAllocMsg(0, sizeof(int), 0); // 0 is a system msg of size int
    *((int *)msg) = groupID;
    CkSendMsg(retEp, msg, retChare);
  }
}

void _createNodeGroup(CkGroupID groupID, envelope *env, int retEp, CkChareID *retChare)
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
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordCreateNodeGroup();
#endif
  _createNodeGroupMember(groupID, epIdx, msg);
  if(retEp) {
    msg = CkAllocMsg(0, sizeof(int), 0); // 0 is a system msg of size int
    *((int *)msg) = groupID;
    CkSendMsg(retEp, msg, retChare);
  }
}

static CkGroupID _staticGroupCreate(envelope *env, int retEp, CkChareID *retChare)
{
  register CkGroupID groupNum = _numGroups++;
  _createGroup(groupNum, env, retEp, retChare);
  return groupNum;
}

static void _dynamicGroupCreate(envelope *env, int retEp, CkChareID * retChare)
{
  register CkChareID *msg = 
    (CkChareID*) _allocMsg(DBocReqMsg, sizeof(CkChareID));
  if(retChare)
    *msg = *retChare;
  register envelope *newenv = UsrToEnv((void *)msg);
  newenv->setUsrMsg(env);
  newenv->setSrcPe(CkMyPe());
  newenv->setRetEp(retEp);
  CmiSetHandler(newenv, _charmHandlerIdx);
  CmiSyncSendAndFree(0, newenv->getTotalsize(), newenv); 
  CpvAccess(_qd)->create();
}

static CkGroupID _staticNodeGroupCreate(envelope *env, int retEp, CkChareID *retChare)
{
  CmiLock(_nodeLock);
  register CkGroupID groupNum = _numNodeGroups++;
  CmiUnlock(_nodeLock);
  _createNodeGroup(groupNum, env, retEp, retChare);
  return groupNum;
}

static void _dynamicNodeGroupCreate(envelope *env, int retEp, CkChareID * retChare)
{
  register CkChareID *msg = 
    (CkChareID*) _allocMsg(DNodeBocReqMsg, sizeof(CkChareID));
  if(retChare)
    *msg = *retChare;
  register envelope *newenv = UsrToEnv((void *)msg);
  newenv->setUsrMsg(env);
  newenv->setSrcPe(CkMyPe());
  newenv->setRetEp(retEp);
  CmiSetHandler(newenv, _charmHandlerIdx);
  CmiSyncNodeSendAndFree(0, newenv->getTotalsize(), newenv); 
  CpvAccess(_qd)->create();
}

extern "C"
CkGroupID CkCreateGroup(int cIdx, int eIdx, void *msg, int retEp,CkChareID *retChare)
{
  assert(cIdx == _entryTable[eIdx]->chareIdx);
  register envelope *env = UsrToEnv(msg);
  env->setMsgtype(BocInitMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  _TRACE_CREATION_N(env, CkNumPes());
  if(CkMyPe()==0) {
    return _staticGroupCreate(env, retEp, retChare);
  } else {
    _dynamicGroupCreate(env, retEp, retChare);
    return (-1);
  }
}

extern "C"
CkGroupID CkCreateNodeGroup(int cIdx, int eIdx, void *msg, int retEp,CkChareID *retChare)
{
  assert(cIdx == _entryTable[eIdx]->chareIdx);
  register envelope *env = UsrToEnv(msg);
  env->setMsgtype(NodeBocInitMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  _TRACE_CREATION_N(env, CkNumNodes());
  if(CkMyNode()==0) {
    return _staticNodeGroupCreate(env, retEp, retChare);
  } else {
    _dynamicNodeGroupCreate(env, retEp, retChare);
    return (-1);
  }
}

extern "C"
void *CkLocalBranch(CkGroupID groupID)
{
  return CpvAccess(_groupTable)->find(groupID);
}

extern "C"
void *CkLocalNodeBranch(CkGroupID groupID)
{
  CmiLock(_nodeLock);
  void *retval = _nodeGroupTable->find(groupID);
  CmiUnlock(_nodeLock);
  return retval;
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
  CmiSetHandler(env, _charmHandlerIdx);
  CldEnqueue(pe, env, _infoIdx);
}

extern "C"
void CkSendMsgBranch(int eIdx, void *msg, int pe, CkGroupID gID)
{
  _TRACE_CREATION_1(UsrToEnv(msg));
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordSendBranch();
#endif
  _sendMsgBranch(eIdx, msg, gID, pe);
  CpvAccess(_qd)->create();
}

extern "C"
void CkBroadcastMsgBranch(int eIdx, void *msg, CkGroupID gID)
{
  _TRACE_CREATION_N(UsrToEnv(msg), CkNumPes());
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordSendBranch(CkNumPes());
#endif
  _sendMsgBranch(eIdx, msg, gID);
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
  CmiSetHandler(env, _charmHandlerIdx);
  CldNodeEnqueue(node, env, _infoIdx);
}

extern "C"
void CkSendMsgNodeBranch(int eIdx, void *msg, int node, CkGroupID gID)
{
  _TRACE_CREATION_1(UsrToEnv(msg));
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordSendNodeBranch();
#endif
  _sendMsgNodeBranch(eIdx, msg, gID, node);
  CpvAccess(_qd)->create();
}

extern "C"
void CkBroadcastMsgNodeBranch(int eIdx, void *msg, CkGroupID gID)
{
  _TRACE_CREATION_N(UsrToEnv(msg), CkNumNodes());
#ifndef CMK_OPTIMIZE
  CpvAccess(_myStats)->recordSendNodeBranch(CkNumNodes());
#endif
  _sendMsgNodeBranch(eIdx, msg, gID);
  CpvAccess(_qd)->create(CkNumNodes());
}
