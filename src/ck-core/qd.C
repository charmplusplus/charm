/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "ck.h"

#define  DEBUGP(x)   // CmiPrintf x;

#if CMK_BLUEGENE_CHARM
// this is a hack for bgcharm++, I need to figure out a better
// way to do this
#define CmiSyncSendAndFree    CmiFreeSendFn
#endif

CpvDeclare(QdState*, _qd);

static inline void _bcastQD1(QdState* state, QdMsg *msg)
{
  msg->setPhase(0);
  state->propagate(msg);
  msg->setPhase(1);
  DEBUGP(("[%d] State: getCreated:%d getProcessed:%d\n", CmiMyPe(), state->getCreated(), state->getProcessed()));
#if 1
  int created = state->getCreated();
  int processed = state->getProcessed();
#ifdef CMK_CPV_IS_SMP
  if (CmiMyRank()==0) {
    created += CpvAccessOther(_qd, CmiMyNodeSize())->getCreated();
    processed += CpvAccessOther(_qd, CmiMyNodeSize())->getProcessed();
  }
#endif
  msg->setCreated(created);
  msg->setProcessed(processed);
#else
  msg->setCreated(state->getCreated());
  msg->setProcessed(state->getProcessed());
#endif
  envelope *env = UsrToEnv((void*)msg);
  CmiSyncSendAndFree(CmiMyPe(), env->getTotalsize(), (char *)env);
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
  CmiSyncSendAndFree(CmiMyPe(), env->getTotalsize(), (char *)env);
  state->reset();
  state->setStage(2);
}

static inline void _handlePhase0(QdState *state, QdMsg *msg)
{
  CkAssert(CmiMyPe()==0 || state->getStage()==0);
  if(CmiMyPe()==0) {
    QdCallback *qdcb = new QdCallback(msg->getCb());
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
      CkAssert(CmiMyPe()!=0);
      _bcastQD2(state, msg);
      break;
    case 1 :
      DEBUGP(("[%d] msg: getCreated:%d getProcessed:%d\n", CmiMyPe(), msg->getCreated(), msg->getProcessed()));
      state->subtreeCreate(msg->getCreated());
      state->subtreeProcess(msg->getProcessed());
      state->reported();
      if(state->allReported()) {
        if(CmiMyPe()==0) {
          DEBUGP(("ALL: %p getCCreated:%d getCProcessed:%d\n", state, state->getCCreated(), state->getCProcessed()));
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
                             env->getTotalsize(), (char *)env);
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
//  This assertion seems too strong for smp and uth version.
//  CkAssert(state->getStage()==2);
  state->subtreeSetDirty(msg->getDirty());
  state->reported();
  if(state->allReported()) {
    if(CmiMyPe()==0) {
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
      CmiSyncSendAndFree(state->getParent(), env->getTotalsize(), (char *)env);
      state->reset();
      state->setStage(0);
    }
  } else
    CkFreeMsg(msg);
}

static void _callWhenIdle(QdMsg *msg)
{
  DEBUGP(("[%d] callWhenIdle\n", CmiMyPe()));
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
  CcdCallOnCondition(CcdPROCESSOR_STILL_IDLE, (CcdVoidFn)_callWhenIdle, (void*) msg);
}


void CkStartQD(const CkCallback& cb)
{
  register QdMsg *msg = (QdMsg*) CkAllocMsg(0,sizeof(QdMsg),0);
  msg->setPhase(0);
  msg->setCb(cb);
  register envelope *env = UsrToEnv((void *)msg);
  CmiSetHandler(env, _qdHandlerIdx);
#if CMK_BLUEGENE_CHARM
  CmiFreeSendFn(0, env->getTotalsize(), (char *)env);
#else
  CldEnqueue(0, env, _infoIdx);
#endif
}

extern "C"
void CkStartQD(int eIdx, const CkChareID *cid)
{
  CkStartQD(CkCallback(eIdx, *cid));
}
