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
void CkGetChareID(CkChareID *pCid) {
  pCid->onPE = CkMyPe();
  pCid->objPtr = CpvAccess(_currentChare);
}

extern "C"
int CkGetGroupID(void) {
  return CpvAccess(_currentGroup);
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
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->beginExecute(env);
  _entryTable[env->getEpIdx()]->call(msg, obj);
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->endExecute();
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
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->beginExecute(env);
  _entryTable[env->getEpIdx()]->call(msg, obj);
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->endExecute();
}

static inline void _processForChareMsg(envelope *env)
{
  register void *msg = EnvToUsr(env);
  register int epIdx = env->getEpIdx();
  register void *obj = env->getObjPtr();
  CpvAccess(_currentChare) = obj;
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->beginExecute(env);
  _entryTable[epIdx]->call(msg, obj);
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->endExecute();
}

static inline void _processForBocMsg(envelope *env)
{
  register int groupID = env->getGroupNum();
  register void *obj = CpvAccess(_groupTable)->find(groupID);
  env->setMsgtype(ForChareMsg);
  env->setObjPtr(obj);
  CpvAccess(_currentGroup) = groupID;
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
  vptr->send(env);
}

static inline void _processDBocReqMsg(envelope *env)
{
  assert(CkMyPe()==0);
  register int groupNum = _numGroups++;
  env->setMsgtype(DBocNumMsg);
  register int srcPe = env->getSrcPe();
  env->setSrcPe(CkMyPe());
  env->setGroupNum(groupNum);
  CmiSyncSendAndFree(srcPe, env->getTotalsize(), env);
  CpvAccess(_qd)->create();
}

static inline void _processDBocNumMsg(envelope *env)
{
  register envelope *usrenv = (envelope *) env->getUsrMsg();
  register int retEp = env->getRetEp();
  register CkChareID *retChare = (CkChareID *) EnvToUsr(env);
  register int groupID = env->getGroupNum();
  _createGroup(groupID, usrenv, retEp, retChare);
}

void _processBocInitMsg(envelope *env)
{
  register int groupID = env->getGroupNum();
  register int epIdx = env->getEpIdx();
  _createGroupMember(groupID, epIdx, EnvToUsr(env));
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
      CpvAccess(_myStats)->recordProcessChare();
      break;
    case NewVChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processNewVChareMsg(env);
      CpvAccess(_myStats)->recordProcessChare();
      break;
    case BocInitMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processBocInitMsg(env);
      break;
    case DBocReqMsg:
      CpvAccess(_qd)->process();
      _processDBocReqMsg(env);
      break;
    case DBocNumMsg:
      CpvAccess(_qd)->process();
      _processDBocNumMsg(env);
      break;
    case ForChareMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processForChareMsg(env);
      CpvAccess(_myStats)->recordProcessMsg();
      break;
    case ForBocMsg :
      CpvAccess(_qd)->process();
      if(env->isPacked()) _unpackFn((void **)&env);
      _processForBocMsg(env);
      CpvAccess(_myStats)->recordProcessBranch();
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
    if(CpvAccess(traceOn))
      CpvAccess(_trace)->beginPack();
    msg = _msgTable[msgIdx]->pack(msg);
    if(CpvAccess(traceOn))
      CpvAccess(_trace)->endPack();
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
    if(CpvAccess(traceOn))
      CpvAccess(_trace)->beginUnpack();
    msg = _msgTable[msgIdx]->unpack(msg);
    if(CpvAccess(traceOn))
      CpvAccess(_trace)->endUnpack();
    UsrToEnv(msg)->setPacked(0);
    *((envelope **)pEnv) = UsrToEnv(msg);
  }
}

extern "C"
void CkSendMsg(int entryIdx, void *msg, CkChareID *pCid)
{
  register envelope *env = UsrToEnv(msg);
  env->setMsgtype(ForChareMsg);
  env->setEpIdx(entryIdx);
  CmiSetHandler(env, _charmHandlerIdx);
  if(pCid->onPE < 0) {
    register int pe = -(pCid->onPE+1);
    if(pe==CkMyPe()) {
      VidBlock *vblk = (VidBlock *) pCid->objPtr;
      vblk->send(env);
    } else {
      env->setMsgtype(ForVidMsg);
      env->setSrcPe(CkMyPe());
      env->setVidPtr(pCid->objPtr);
      if(CpvAccess(traceOn))
        CpvAccess(_trace)->creation(env);
      CpvAccess(_qd)->create();
      CldEnqueue(pe, env, _infoIdx);
    }
  } else {
    env->setSrcPe(CkMyPe());
    env->setObjPtr(pCid->objPtr);
    if(CpvAccess(traceOn))
      CpvAccess(_trace)->creation(env);
    CpvAccess(_qd)->create();
    CldEnqueue(pCid->onPE, env, _infoIdx);
  }
  CpvAccess(_myStats)->recordSendMsg();
}

extern "C"
void CkCreateChare(int cIdx, int eIdx, void *msg, CkChareID *pCid, int destPE)
{
  assert(cIdx == _entryTable[eIdx]->chareIdx);
  envelope *env = UsrToEnv(msg);
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
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->creation(env);
  CpvAccess(_qd)->create();
  CpvAccess(_myStats)->recordCreateChare();
  CldEnqueue(destPE, env, _infoIdx);
}

void _createGroupMember(int groupID, int eIdx, void *msg)
{
  register int gIdx = _entryTable[eIdx]->chareIdx;
  register void *obj = malloc(_chareTable[gIdx]->size);
  CpvAccess(_groupTable)->add(groupID, obj);
  register void *prev = CpvAccess(_currentChare);
  CpvAccess(_currentChare) = obj;
  register int prevGrp = CpvAccess(_currentGroup);
  CpvAccess(_currentGroup) = groupID;
  _entryTable[eIdx]->call(msg, obj);
  CpvAccess(_currentChare) = prev;
  CpvAccess(_currentGroup) = prevGrp;
  CpvAccess(_myStats)->recordProcessGroup();
}

void _createGroup(int groupID, envelope *env, int retEp, CkChareID *retChare)
{
  register int epIdx = env->getEpIdx();
  register int msgIdx = env->getMsgIdx();
  env->setGroupNum(groupID);
  env->setSrcPe(CkMyPe());
  register void *msg =  EnvToUsr(env);
  if(CkNumPes()>1) {
    if(!env->isPacked() && _msgTable[msgIdx]->pack) {
      msg = _msgTable[msgIdx]->pack(msg);
      UsrToEnv(msg)->setPacked(1);
    }
    env = UsrToEnv(msg);
    CmiSetHandler(env, _bocHandlerIdx);
    _numInitMsgs++;
    CmiSyncBroadcast(env->getTotalsize(), env);
    CpvAccess(_qd)->create(CkNumPes()-1);
    if(env->isPacked() && _msgTable[msgIdx]->unpack) {
      if(CpvAccess(traceOn))
        CpvAccess(_trace)->beginUnpack();
      msg = _msgTable[msgIdx]->unpack(msg);
      if(CpvAccess(traceOn))
        CpvAccess(_trace)->endUnpack();
      UsrToEnv(msg)->setPacked(0);
    }
  }
  CpvAccess(_myStats)->recordCreateGroup();
  _createGroupMember(groupID, epIdx, msg);
  if(retChare) {
    msg = CkAllocMsg(0, sizeof(int), 0); // 0 is a system msg of size int
    *((int *)msg) = groupID;
    CkSendMsg(retEp, msg, retChare);
  }
}

static int _staticGroupCreate(envelope *env, int retEp, CkChareID *retChare)
{
  register int groupNum = _numGroups++;
  _createGroup(groupNum, env, retEp, retChare);
  return groupNum;
}

static void _dynamicGroupCreate(envelope *env, int retEp, CkChareID * retChare)
{
  register CkChareID *msg = 
    (CkChareID*) _allocMsg(DBocReqMsg, sizeof(CkChareID));
  *msg = *retChare;
  register envelope *newenv = UsrToEnv((void *)msg);
  newenv->setUsrMsg(env);
  newenv->setSrcPe(CkMyPe());
  newenv->setEpIdx(retEp);
  CmiSetHandler(newenv, _charmHandlerIdx);
  CmiSyncSendAndFree(0, newenv->getTotalsize(), newenv); 
  CpvAccess(_qd)->create();
}

extern "C"
int CkCreateGroup(int cIdx, int eIdx, void *msg, int retEp,CkChareID *retChare)
{
  assert(cIdx == _entryTable[eIdx]->chareIdx);
  register envelope *env = UsrToEnv(msg);
  env->setMsgtype(BocInitMsg);
  env->setEpIdx(eIdx);
  env->setSrcPe(CkMyPe());
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->creation(env, CkNumPes());
  if(CkMyPe()==0) {
    return _staticGroupCreate(env, retEp, retChare);
  } else {
    _dynamicGroupCreate(env, retEp, retChare);
    return (-1);
  }
}

extern "C"
void *CkLocalBranch(int groupID)
{
  return CpvAccess(_groupTable)->find(groupID);
}

static inline void _sendMsgBranch(int eIdx, void *msg, int gID, 
                           int pe=CLD_BROADCAST_ALL)
{
  register envelope *env = UsrToEnv(msg);
  env->setMsgtype(ForBocMsg);
  env->setEpIdx(eIdx);
  env->setGroupNum(gID);
  env->setSrcPe(CkMyPe());
  CmiSetHandler(env, _charmHandlerIdx);
  CldEnqueue(pe, env, _infoIdx);
}

extern "C"
void CkSendMsgBranch(int eIdx, void *msg, int pe, int gID)
{
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->creation(UsrToEnv(msg));
  _sendMsgBranch(eIdx, msg, gID, pe);
  CpvAccess(_myStats)->recordSendBranch();
  CpvAccess(_qd)->create();
}

extern "C"
void CkBroadcastMsgBranch(int eIdx, void *msg, int gID)
{
  if(CpvAccess(traceOn))
    CpvAccess(_trace)->creation(UsrToEnv(msg), CkNumPes());
  _sendMsgBranch(eIdx, msg, gID);
  CpvAccess(_myStats)->recordSendBranch(CkNumPes());
  CpvAccess(_qd)->create(CkNumPes());
}
