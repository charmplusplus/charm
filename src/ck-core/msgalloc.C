#include "ck.h"
#if !CMK_RECONVERSE  /* reconverse has no converse-level queueing.h */
#include "queueing.h"
#endif

CkpvDeclare(size_t *, _offsets);

void *CkAllocSysMsg(const CkEntryOptions *opts)
{
  if(opts == NULL)
    return CkpvAccess(_msgPool)->get();

  envelope *env = _allocEnv(ForChareMsg, 0, opts->getPriorityBits(), GroupDepNum{(int)opts->getGroupDepNum()});
  setMemoryTypeMessage(env);
  env->setMsgIdx(0);

  env->setIsVarSysMsg(1);
  // Set the message's queueing type
  env->setQueueing((unsigned char)opts->getQueueing());

  // Copy the priority bytes into the env from the opts
  if (opts->getPriorityPtr() != NULL)
    CmiMemcpy(env->getPrioPtr(), opts->getPriorityPtr(), env->getPrioBytes());

  // Copy the group dependence into the env from the opts
  if(opts->getGroupDepNum() > 0)
    CmiMemcpy(env->getGroupDepPtr(), opts->getGroupDepPtr(), env->getGroupDepSize());

  return EnvToUsr(env);
}

void CkFreeSysMsg(void *m)
{
  CkpvAccess(_msgPool)->put(m);
}

void* CkAllocMsg(int msgIdx, int msgBytes, int prioBits, GroupDepNum groupDepNum)
{
  envelope* env = _allocEnv(ForChareMsg, msgBytes, prioBits, groupDepNum);
  setMemoryTypeMessage(env);

  env->setQueueing(_defaultQueueing);
  env->setMsgIdx(msgIdx);

  return EnvToUsr(env);
}

void* CkAllocBuffer(void *msg, int bufsize)
{
  bufsize = CkMsgAlignLength(bufsize);
  envelope *env = UsrToEnv(msg);
  envelope *packbuf = _allocEnv(env->getMsgtype(), bufsize,
                      env->getPriobits(),
                      GroupDepNum{(int)env->getGroupDepNum()});
  
  int size = packbuf->getTotalsize();
  CmiMemcpy(packbuf, env, sizeof(envelope));
  packbuf->setTotalsize(size);
  packbuf->setPacked(!env->isPacked());
  CmiMemcpy(packbuf->getPrioPtr(), env->getPrioPtr(), packbuf->getPrioBytes());

  return EnvToUsr(packbuf);;
}

#if CMK_ERROR_CHECKING
// The message of the [nokeep] entry method this PE is executing, if any
// (set around the call in ck.C). The runtime owns nokeep messages, marshalled
// parameters included, and frees them when the method returns; a free by the
// method itself is a double free that used to surface, if at all, as a crash
// somewhere else later.
CkpvDeclare(void *, _nokeepMsgInFlight);
CkpvDeclare(int, _nokeepEpInFlight);
#endif

void  CkFreeMsg(void *msg)
{
  if (msg!=NULL) {
#if CMK_ERROR_CHECKING
      if (CkpvInitialized(_nokeepMsgInFlight) && msg == CkpvAccess(_nokeepMsgInFlight))
        CkAbort("Entry method %s is [nokeep] but freed its message. The runtime owns "
                "the message of a nokeep entry method (every marshalled entry method "
                "that is not threaded is nokeep) and frees it when the method returns.",
                _entryTable[CkpvAccess(_nokeepEpInFlight)]->name);
#endif
      CmiFree(UsrToEnv(msg));
  }
}


void* CkCopyMsg(void **pMsg)
{// cannot simply memcpy, because srcMsg could be varsize msg
  void *srcMsg = *pMsg;
  envelope *env = UsrToEnv(srcMsg);
  unsigned char msgidx = env->getMsgIdx();
  if(!env->isPacked() && _msgTable[msgidx]->pack) {
    srcMsg = _msgTable[msgidx]->pack(srcMsg);
    UsrToEnv(srcMsg)->setPacked(1);
  }
  int size = UsrToEnv(srcMsg)->getTotalsize();
  envelope *newenv = (envelope *) CmiAlloc(size);
  CmiMemcpy(newenv, UsrToEnv(srcMsg), size);
  //memcpy(newenv, UsrToEnv(srcMsg), size);
  if(UsrToEnv(srcMsg)->isPacked() && _msgTable[msgidx]->unpack) {
    srcMsg = _msgTable[msgidx]->unpack(srcMsg);
    UsrToEnv(srcMsg)->setPacked(0);
  }
  *pMsg = srcMsg;
  if(newenv->isPacked() && _msgTable[msgidx]->unpack) {
    srcMsg = _msgTable[msgidx]->unpack(EnvToUsr(newenv));
    UsrToEnv(srcMsg)->setPacked(0);
  } else srcMsg = EnvToUsr(newenv);

  setMemoryTypeMessage(newenv);
  return srcMsg;
}

// Copy a PACKED message and unpack only the copy; the source stays packed and at
// the same address. CkCopyMsg instead packs and re-unpacks the source itself, which
// for a custom pack/unpack pair (the manual's idiom: pack deletes its input, unpack
// builds a new object) replaces the source object, so any other holder of the old
// pointer is left with freed memory.
void* CkCopyPackedMsg(const void *packedMsg)
{
  envelope *env = UsrToEnv(packedMsg);
  CkAssert(env->isPacked());
  const int size = env->getTotalsize();
  envelope *newenv = (envelope *) CmiAlloc(size);
  CmiMemcpy(newenv, env, size);
  setMemoryTypeMessage(newenv);
  CkUnpackMessage(&newenv);
  return EnvToUsr(newenv);
}

void* CkReferenceMsg(void* msg)
{
  CmiReference(UsrToEnv(msg));
  return msg;
}

void  CkSetQueueing(void *msg, int strategy)
{
  UsrToEnv(msg)->setQueueing((unsigned char) strategy);
}


void* CkPriorityPtr(void *msg)
{
#if CMK_ERROR_CHECKING
  if (UsrToEnv(msg)->getPriobits() == 0) CkAbort("Trying to access priority bits, but none was allocated");
#endif
  return UsrToEnv(msg)->getPrioPtr();
}

CkMarshallMsg *CkAllocateMarshallMsgNoninline(int size,const CkEntryOptions *opts)
{
	//Allocate the message
	CkMarshallMsg *m=new (size,opts->getPriorityBits(),GroupDepNum{(int)opts->getGroupDepNum()}) CkMarshallMsg;
	//Copy the user's priority data into the message
	envelope *env=UsrToEnv(m);
	setMemoryTypeMessage(env);
	if (opts->getPriorityPtr() != NULL)
		CmiMemcpy(env->getPrioPtr(),opts->getPriorityPtr(),env->getPrioBytes());

	// Copy the group dependence into the env from the opts
	if(opts->getGroupDepNum() > 0)
		CmiMemcpy(env->getGroupDepPtr(), opts->getGroupDepPtr(), env->getGroupDepSize());

	//Set the message's queueing type
	env->setQueueing((unsigned char)opts->getQueueing());
	return m;
}

