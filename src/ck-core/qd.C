/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"

CpvDeclare(QdState*, _qd);

static inline void _bcastQD1(QdState* state, QdMsg *msg)
{
  msg->setPhase(0);
  state->propagate(msg);
  msg->setPhase(1);
  msg->setCreated(state->getCreated());
  msg->setProcessed(state->getProcessed());
  envelope *env = UsrToEnv((void*)msg);
  CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), env);
  state->markProcessed();
  state->reset();
  state->setStage(1);
}

static inline void _bcastQD2(QdState* state, QdMsg *msg)
{
  msg->setPhase(1);
  state->propagate(msg);
  msg->setPhase(2);
  msg->setDirty(state->isDirty());
  envelope *env = UsrToEnv((void*)msg);
  CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), env);
  state->reset();
  state->setStage(2);
}

static inline void _handlePhase0(QdState *state, QdMsg *msg)
{
  assert(CkMyPe()==0 || state->getStage()==0);
  if(CkMyPe()==0) {
    QdCallback *qdcb = new QdCallback(msg->getEp(), msg->getCid());
    _MEMCHECK(qdcb);
    state->enq(qdcb);
  }
  if(state->getStage()==0)
    _bcastQD1(state, msg);
  else
    CkFreeMsg(msg);
}

static inline void _handlePhase1(QdState *state, QdMsg *msg)
{
  switch(state->getStage()) {
    case 0 :
      assert(CkMyPe()!=0);
      _bcastQD2(state, msg);
      break;
    case 1 :
      state->subtreeCreate(msg->getCreated());
      state->subtreeProcess(msg->getProcessed());
      state->reported();
      if(state->allReported()) {
        if(CkMyPe()==0) {
          if(state->getCCreated()==state->getCProcessed()) {
            _bcastQD2(state, msg);
          } else {
            _bcastQD1(state, msg);
          }
        } else {
          msg->setCreated(state->getCCreated());
          msg->setProcessed(state->getCProcessed());
          envelope *env = UsrToEnv((void*)msg);
          CmiSyncSendAndFree(state->getParent(), 
                             env->getTotalsize(), env);
          state->reset();
          state->setStage(0);
        }
      } else
          CkFreeMsg(msg);
      break;
    default: CmiAbort("Internal QD Error. Contact Developers.!\n");
  }
}

static inline void _handlePhase2(QdState *state, QdMsg *msg)
{
  assert(state->getStage()==2);
  state->subtreeSetDirty(msg->getDirty());
  state->reported();
  if(state->allReported()) {
    if(CkMyPe()==0) {
      if(state->isDirty()) {
        _bcastQD1(state, msg);
      } else {
        QdCallback* cb;
        while(NULL!=(cb=state->deq())) {
          cb->send();
          delete cb;
        }
        state->reset();
        state->setStage(0);
        CkFreeMsg(msg);
      }
    } else {
      msg->setDirty(state->isDirty());
      envelope *env = UsrToEnv((void*)msg);
      CmiSyncSendAndFree(state->getParent(), env->getTotalsize(), env);
      state->reset();
      state->setStage(0);
    }
  } else
    CkFreeMsg(msg);
}

static void _callWhenIdle(QdMsg *msg)
{
  QdState *state = CpvAccess(_qd);
  switch(msg->getPhase()) {
    case 0 : _handlePhase0(state, msg); break;
    case 1 : _handlePhase1(state, msg); break;
    case 2 : _handlePhase2(state, msg); break;
    default: CmiAbort("Internal QD Error. Contact Developers.!\n");
  }
}

void _qdHandler(envelope *env)
{
  register QdMsg *msg = (QdMsg*) EnvToUsr(env);
  CcdCallOnCondition(CcdPROCESSORIDLE, (CcdVoidFn)_callWhenIdle, (void*) msg);
}

extern "C"
void CkStartQD(int eIdx, CkChareID *cid)
{
  register QdMsg *msg = (QdMsg*) CkAllocMsg(0,sizeof(QdMsg),0);
  msg->setPhase(0);
  msg->setEp(eIdx);
  msg->setCid(*cid);
  register envelope *env = UsrToEnv((void *)msg);
  CmiSetHandler(env, _qdHandlerIdx);
  CldEnqueue(0, env, _infoIdx);
}
