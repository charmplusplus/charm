/*
  implements per object message queue - every object maintains its own 
  queue of messages.

  potentially good for:
  1. load balancer to know the number of pending messages waiting in its queue;
  2. improve cache performance by executing all messages belonging to one obj.

  There are two ways to use it:
  1. by default, no queue is created for any object. Using "+objq" runtime
     option will create queue for every object
  2. in a Chare's constructor, call CkUsesObjQ() to turn on the object queue
     for that calling Chare.

  created by Gengbin Zheng, gzheng@uiucedu on 12/29/2003
*/
#include "charm++.h"
#include "envelope.h"
#include "queueing.h"
#include "ckobjQ.h"

CkpvDeclare(TokenPool*, _tokenPool);

extern int index_tokenHandler;

extern CkMigratable * CkArrayMessageObjectPtr(envelope *env);

#define OBJQ_FIFO   1

// turn on object queue
void CkObjectMsgQ::create() { 
  if (!objQ) 
#if OBJQ_FIFO
    objQ = (void *)CdsFifo_Create();
#else
    objQ = (void *)CqsCreate(); 
#endif
}

int CkObjectMsgQ::length() const { 
#if OBJQ_FIFO
  return objQ?CdsFifo_Length((CdsFifo)objQ):0; 
#else
  return objQ?CqsLength((Queue)objQ):0; 
#endif
}

CkObjectMsgQ::~CkObjectMsgQ() 
{
  if (objQ) {
    process();
    // delete objQ
#if OBJQ_FIFO
    CdsFifo_Destroy(objQ);
#else
    CqsDelete((Queue)objQ);
#endif
  }
}

// must be re-entrant
void CkObjectMsgQ::process() 
{
#if CMK_OBJECT_QUEUE_AVAILABLE
  if (objQ == NULL) return;
//  if (inprocessing) return;
  int mlen = length();
  if (mlen == 0) return;

  ObjectToken *tok;
#if OBJQ_FIFO
  tok = (ObjectToken*)CdsFifo_Dequeue(objQ);
#else
  CqsDequeue((Queue)objQ, (void **)&tok);
#endif
  while (tok != NULL) {
    envelope *env = (envelope *)tok->message;
    if (env) {
      // release messages in the queue
      tok->message = NULL;
/*
      CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
      env, env->getQueueing(),env->getPriobits(),
      (unsigned int *)env->getPrioPtr());
*/
      // not be able to call inline, enqueue to the obj msg queue
      CdsFifo_Enqueue(CpvAccess(CsdObjQueue), env);
    }
    else
      CkpvAccess(_tokenPool)->put(tok);
#if OBJQ_FIFO
    tok = (ObjectToken*)CdsFifo_Dequeue(objQ);
#else
    CqsDequeue((Queue)objQ, (void **)&tok);
#endif
  }
#endif
}

// find out the object pointer from a charm message envelope
Chare * CkFindObjectPtr(envelope *env)
{
  Chare *obj = NULL;
  switch(env->getMsgtype()) {
    case BocInitMsg:
    case NodeBocInitMsg:
    case ArrayEltInitMsg:
    case NewChareMsg:
    case NewVChareMsg:
    case ForVidMsg:
    case FillVidMsg:
      break;
    case ForArrayEltMsg:
      obj = CkArrayMessageObjectPtr(env);
      break;
    case ForChareMsg : {
      // FIXME, chare calls CldEnqueue  which bypass the object queue
      obj = (Chare*)env->getObjPtr();
      break;
    }
    case ForBocMsg : {
      obj = _localBranch(env->getGroupNum());
      break;
    }
    case ForNodeBocMsg : {
      obj = (Chare*)(CksvAccess(_nodeGroupTable)->find(env->getGroupNum()).getObj());
      break;
    }
    default:
      CmiAbort("Fatal Charm++ Error> Unknown msg-type in CkFindObjectPtr.\n");
  }
  return obj;
}

#if CMK_OBJECT_QUEUE_AVAILABLE
// insert an envelope msg into object queue
void _enqObjQueue(Chare *obj, envelope *env)
{
    ObjectToken *token = CkpvAccess(_tokenPool)->get();
    CmiAssert(token);
    token->message = env;
    token->objPtr = obj;
  
    CmiSetHandler(token, index_tokenHandler);
    // enq to charm sched queue
    CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	token, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
    // also enq to object queue
#if OBJQ_FIFO
    CdsFifo_Enqueue(obj->CkGetObjQueue().queue(), token);
#else
    CqsEnqueueGeneral((Queue)(obj->CkGetObjQueue().queue()),
  	token, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
#endif
}
#endif


// converseMsg is a real message
void _ObjectQHandler(void *converseMsg)
{
#if CMK_OBJECT_QUEUE_AVAILABLE
  register envelope *env = (envelope *)(converseMsg);
  Chare *obj = CkFindObjectPtr(env);
  // swap handler back
//  CmiSetHandler(env, CmiGetXHandler(env));
  // I can do this because this message is always a charm+ message
  CmiSetHandler(env, _charmHandlerIdx);
  if (obj && obj->CkGetObjQueue().queue()) {  // queue enabled
    _enqObjQueue(obj, env);
  }
  else {   // obj queue not enabled
    CqsEnqueueGeneral((Queue)CpvAccess(CsdSchedQueue),
  	env, env->getQueueing(),env->getPriobits(),
  	(unsigned int *)env->getPrioPtr());
  }
#else
  CmiAbort("Invalide _ObjectQHandler called!");
#endif
}

// normally from sched queue
void _TokenHandler(void *tokenMsg)
{
#if CMK_OBJECT_QUEUE_AVAILABLE
  ObjectToken *token = (ObjectToken*)tokenMsg;
  Chare *obj = token->objPtr;
  void *message = token->message;
  // we are done with this token out of sched queue
  CkpvAccess(_tokenPool)->put(token);
  if (message == NULL) {    // message already consumed
    return;
  }
  CkObjectMsgQ &objQ = obj->CkGetObjQueue();
  objQ.process();
#else
  CmiAbort("Invalide _TokenHandler called!");
#endif
}
