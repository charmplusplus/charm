#include "ck.h"
#include "queueing.h"

CkpvDeclare(size_t *, _offsets);

extern "C"
void *CkAllocSysMsg(void)
{
  return CkpvAccess(_msgPool)->get();
}

extern "C"
void CkFreeSysMsg(void *m)
{
  CkpvAccess(_msgPool)->put(m);
}

extern "C"
void* CkAllocMsg(int msgIdx, int msgBytes, int prioBits)
{
  register envelope* env;
  env = _allocEnv(ForChareMsg, msgBytes, prioBits);
  setMemoryTypeMessage(env);

  env->setQueueing(_defaultQueueing);
  env->setMsgIdx(msgIdx);

  return EnvToUsr(env);
}

extern "C"
void* CkAllocBuffer(void *msg, int bufsize)
{
  bufsize = CkMsgAlignLength(bufsize);
  register envelope *env = UsrToEnv(msg);
  register envelope *packbuf;
  packbuf = _allocEnv(env->getMsgtype(), bufsize, 
                      env->getPriobits());
  
  register int size = packbuf->getTotalsize();
  CmiMemcpy(packbuf, env, sizeof(envelope));
  packbuf->setTotalsize(size);
  packbuf->setPacked(!env->isPacked());
  CmiMemcpy(packbuf->getPrioPtr(), env->getPrioPtr(), packbuf->getPrioBytes());

  return EnvToUsr(packbuf);;
}

extern "C"
void  CkFreeMsg(void *msg)
{
  if (msg!=NULL) {
      CmiFree(UsrToEnv(msg));
  }
}


extern "C"
void* CkCopyMsg(void **pMsg)
{// cannot simply memcpy, because srcMsg could be varsize msg
  register void *srcMsg = *pMsg;
  register envelope *env = UsrToEnv(srcMsg);
  register unsigned char msgidx = env->getMsgIdx();
  if(!env->isPacked() && _msgTable[msgidx]->pack) {
    srcMsg = _msgTable[msgidx]->pack(srcMsg);
    UsrToEnv(srcMsg)->setPacked(1);
  }
  register int size = UsrToEnv(srcMsg)->getTotalsize();
  register envelope *newenv = (envelope *) CmiAlloc(size);
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

extern "C"
void  CkSetQueueing(void *msg, int strategy)
{
  UsrToEnv(msg)->setQueueing((unsigned char) strategy);
}


extern "C"
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
	CkMarshallMsg *m=new (size,opts->getPriorityBits())CkMarshallMsg;
	//Copy the user's priority data into the message
	envelope *env=UsrToEnv(m);
	setMemoryTypeMessage(env);
	if (opts->getPriorityPtr() != NULL)
		CmiMemcpy(env->getPrioPtr(),opts->getPriorityPtr(),env->getPrioBytes());
	//Set the message's queueing type
	env->setQueueing((unsigned char)opts->getQueueing());
	return m;
}

