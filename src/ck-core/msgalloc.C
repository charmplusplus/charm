/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"
#include "queueing.h"

#define ELAN_MESSAGE_SIZE 16384
#define SIZEFIELD(m) ((int *)((char *)(m)-2*sizeof(int)))[0]
#define CMI_MSG_TYPE(msg)    ((CmiMsgHeaderBasic *)msg)->type

#if 0
extern Queue localMsgBuf;
#endif

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
#if 0
  register int tsize = sizeof(envelope) + ALIGN(msgBytes) 
    + sizeof(int)*PW(prioBits);

  if(tsize < ELAN_MESSAGE_SIZE) {
    if(!CqsEmpty(localMsgBuf)) {
      //      CkPrintf("Getting message from queue\n");
      CqsDequeue(localMsgBuf, (void **)&env);
    }
    else 
      env = (envelope *)CmiAlloc(ELAN_MESSAGE_SIZE);
    
    env->setMsgtype(ForChareMsg);
    env->setTotalsize(tsize);
    env->setPriobits(prioBits);
    env->setPacked(0);
    _SET_USED(env, 0);
    
   CMI_MSG_TYPE(env) = 1;
  }
  else 
#endif 
    env = _allocEnv(ForChareMsg, msgBytes, prioBits);
  
  env->setQueueing(_defaultQueueing);
  env->setMsgIdx(msgIdx);
  return EnvToUsr(env);
}

extern "C"
void* CkAllocBuffer(void *msg, int bufsize)
{
  bufsize = ALIGN(bufsize);
  register envelope *env = UsrToEnv(msg);
  register envelope *packbuf;

#if 0
  register int tsize = sizeof(envelope) + ALIGN(bufsize) 
    + sizeof(int)*PW(env->getPriobits());
  
  if(tsize < ELAN_MESSAGE_SIZE) {
    if(!CqsEmpty(localMsgBuf)) {
      //      CkPrintf("Getting message from queue\n");
      CqsDequeue(localMsgBuf, (void **)&packbuf);
    }
    else 
      packbuf = (envelope *)CmiAlloc(ELAN_MESSAGE_SIZE);
    
    packbuf->setMsgtype(env->getMsgtype());
    packbuf->setTotalsize(tsize);
    packbuf->setPriobits(env->getPriobits());
    packbuf->setPacked(0);
    _SET_USED(packbuf, 0);
    
    CMI_MSG_TYPE(packbuf) = 1;
  }
  else 
#endif  
    packbuf = _allocEnv(env->getMsgtype(), bufsize, 
			env->getPriobits());
  
  register int size = packbuf->getTotalsize();
  memcpy(packbuf, env, sizeof(envelope));
  packbuf->setTotalsize(size);
  packbuf->setPacked(!env->isPacked());
  memcpy(packbuf->getPrioPtr(), env->getPrioPtr(), packbuf->getPrioBytes());
  return EnvToUsr(packbuf);;
}

extern "C"
void  CkFreeMsg(void *msg)
{
  if (msg!=NULL) {
#if 0
    register envelope *env = UsrToEnv(msg);
    if(SIZEFIELD(env) ==  ELAN_MESSAGE_SIZE) {
      CqsEnqueue(localMsgBuf, env);
      //      CkPrintf("Returning message to queue\n");    
    }
    else
#endif
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
  memcpy(newenv, UsrToEnv(srcMsg), size);
  if(UsrToEnv(srcMsg)->isPacked() && _msgTable[msgidx]->unpack) {
    srcMsg = _msgTable[msgidx]->unpack(srcMsg);
    UsrToEnv(srcMsg)->setPacked(0);
  }
  *pMsg = srcMsg;
  if(newenv->isPacked() && _msgTable[msgidx]->unpack) {
    srcMsg = _msgTable[msgidx]->unpack(EnvToUsr(newenv));
    UsrToEnv(srcMsg)->setPacked(0);
  } else srcMsg = EnvToUsr(newenv);

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
  return UsrToEnv(msg)->getPrioPtr();
}

// This cannot be in the header file because for loop cannot be expanded
// inline by the stupid HP C++ compiler.

MsgPool::MsgPool() 
{ 
  for(int i=0;i<MAXMSGS;i++)
    msgs[i] = _alloc();
  num = MAXMSGS;
}

CkMarshallMsg *CkAllocateMarshallMsgNoninline(int size,const CkEntryOptions *opts)
{
	//Allocate the message
	CkMarshallMsg *m=new (size,opts->getPriorityBits())CkMarshallMsg;
	//Copy the user's priority data into the message
	envelope *env=UsrToEnv(m);
	memcpy(env->getPrioPtr(),opts->getPriorityPtr(),env->getPrioBytes());
	//Set the message's queueing type
	env->setQueueing((unsigned char)opts->getQueueing());
	return m;
}

