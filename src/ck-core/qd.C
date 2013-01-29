#define  DEBUGP(x)   //  CmiPrintf x;

#include "ck.h"

extern int _qdCommHandlerIdx;

// a fake QD which just wait for several seconds to triger QD callback
int _dummy_dq = 0;                      /* seconds to wait for */

#if CMK_BIGSIM_CHARM
// this is a hack for bgcharm++, I need to figure out a better
// way to do this
#undef CmiSyncSendAndFree
#define CmiSyncSendAndFree    CmiFreeSendFn
#endif

CpvDeclare(QdState*, _qd);

// called when a node asks children for their counters
// send broadcast msg (phase 0) to children, and report to itself (phase 1)
// stage 1 means the node is waiting for reports from children
static inline void _bcastQD1(QdState* state, QdMsg *msg)
{
  msg->setPhase(0);
  state->propagate(msg);
  msg->setPhase(1);
  DEBUGP(("[%d] _bcastQD1: State: getCreated:%d getProcessed:%d\n", CmiMyPe(), state->getCreated(), state->getProcessed()));
#if ! CMK_SHARED_VARS_UNIPROCESSOR && !CMK_MULTICORE
/*
  QdState *comm_state;
  static int comm_create=0, comm_process=0;
  if (CmiMyRank()==0) {
    comm_state = CpvAccessOther(_qd, CmiMyNodeSize());
    int new_create = comm_state->getCreated();
    int new_process = comm_state->getProcessed();
    // combine counters with comm thread
    CmiAssert(new_create==0);
    CmiAssert(new_create == 0&& new_process==0);
    state->create(new_create-comm_create);
    state->process(new_process-comm_process);
    comm_create = new_create;
    comm_process = new_process;
  }
*/
#endif
  msg->setCreated(state->getCreated());
  msg->setProcessed(state->getProcessed());
  envelope *env = UsrToEnv((void*)msg);
  CmiSyncSendAndFree(CmiMyPe(), env->getTotalsize(), (char *)env);
  state->markProcessed();
  state->reset();
  state->setStage(1);
  DEBUGP(("[%d] _bcastQD1 stage changed to: %d\n", CmiMyPe(), state->getStage()));
}

// final phase to check if the counters become dirty or not
// stage 2 means the node is waiting for children to report their dirty state
static inline void _bcastQD2(QdState* state, QdMsg *msg)
{
  DEBUGP(("[%d] _bcastQD2: \n", CmiMyPe()));
  msg->setPhase(1);
  state->propagate(msg);
  msg->setPhase(2);
  msg->setDirty(state->isDirty());
  envelope *env = UsrToEnv((void*)msg);
  CmiSyncSendAndFree(CmiMyPe(), env->getTotalsize(), (char *)env);
  state->reset();
  state->setStage(2);
  DEBUGP(("[%d] _bcastQD2: stage changed to: %d\n", CmiMyPe(), state->getStage()));
}

static inline void _handlePhase0(QdState *state, QdMsg *msg)
{
  DEBUGP(("[%d] _handlePhase0: stage: %d, msg phase: %d\n", CmiMyPe(), state->getStage(), msg->getPhase()));
  CkAssert(CmiMyPe()==0 || state->getStage()==0);
  if(CmiMyPe()==0) {
    QdCallback *qdcb = new QdCallback(msg->getCb());
    _MEMCHECK(qdcb);
    state->enq(qdcb);		// stores qd callback
  }
  if(state->getStage()==0)
    _bcastQD1(state, msg);        // start asking children for the counters
  else
    CkFreeMsg(msg);               // already in the middle of processing
}

// collecting counters from children
static inline void _handlePhase1(QdState *state, QdMsg *msg)
{
  DEBUGP(("[%d] _handlePhase1: stage: %d, msg phase: %d\n", CmiMyPe(), state->getStage(), msg->getPhase()));
  switch(state->getStage()) {
    case 0 :
      CkAssert(CmiMyPe()!=0);
      _bcastQD2(state, msg);
      break;
    case 1 :
      DEBUGP(("[%d] msg: getCreated:%d getProcessed:%d\n", CmiMyPe(), msg->getCreated(), msg->getProcessed()));
        // add children's counters
      state->subtreeCreate(msg->getCreated());
      state->subtreeProcess(msg->getProcessed());
      state->reported();
      if(state->allReported()) {
        if(CmiMyPe()==0) {
          DEBUGP(("ALL: %p getCCreated:%d getCProcessed:%d\n", state, state->getCCreated(), state->getCProcessed()));
          if(state->getCCreated()==state->getCProcessed()) {
            _bcastQD2(state, msg);    // almost reached, one pass to make sure
          } else {
            _bcastQD1(state, msg);    // not reached, go over again
          }
        } else {
            // report counters to parent
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

// check if counters became dirty and notify parents
static inline void _handlePhase2(QdState *state, QdMsg *msg)
{
//  This assertion seems too strong for smp and uth version.
  DEBUGP(("[%d] _handlePhase2: stage: %d, msg phase: %d \n", CmiMyPe(), state->getStage(), msg->getPhase()));
  CkAssert(state->getStage()==2);
  state->subtreeSetDirty(msg->getDirty());
  state->reported();
  if(state->allReported()) {
    if(CmiMyPe()==0) {
      if(state->isDirty()) {
        _bcastQD1(state, msg);   // dirty, restart again
      } else {             
          // quiescence detected, send callbacks
        DEBUGP(("[%d] quiescence detected at %f.\n", CmiMyPe(), CmiWallTimer()));
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
        // tell parent if the counters on the node is dirty or not
      DEBUGP(("[%d] _handlePhase2 dirty:%d\n", CmiMyPe(), state->isDirty()));
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
  DEBUGP(("[%d] callWhenIdle msg:%p \n", CmiMyPe(), msg));
  QdState *state = CpvAccess(_qd);
  switch(msg->getPhase()) {
    case 0 : _handlePhase0(state, msg); break;
    case 1 : _handlePhase1(state, msg); break;
    case 2 : _handlePhase2(state, msg); break;
    default: CmiAbort("Internal QD Error. Contact Developers.!\n");
  }
}

static void _invokeQD(QdMsg *msg)
{
  QdCallback *cb = new QdCallback(msg->getCb());
  cb->send();
  delete cb;
}

void _qdHandler(envelope *env)
{
  register QdMsg *msg = (QdMsg*) EnvToUsr(env);
  DEBUGP(("[%d] _qdHandler msg:%p \n", CmiMyPe(), msg));
  if (_dummy_dq > 0)
    CcdCallFnAfter((CcdVoidFn)_invokeQD,(void *)msg, _dummy_dq*1000); // in ms
  else
    CcdCallOnCondition(CcdPROCESSOR_STILL_IDLE, (CcdVoidFn)_callWhenIdle, (void*) msg);
}

// when a message is sent or received from an immediate handler from comm
// thread or interrupt handler, the counter is sent to rank 0 of the same node
void _qdCommHandler(envelope *env)
{
  register QdCommMsg *msg = (QdCommMsg*) EnvToUsr(env);
  DEBUGP(("[%d] _qdCommHandler msg:%p \n", CmiMyPe(), msg));
  if (msg->flag == 0)
    CpvAccess(_qd)->create(msg->count);
  else
    CpvAccess(_qd)->process(msg->count);
  CmiFree(env);
}

void QdState::sendCount(int flag, int count)
{
  if (_dummy_dq == 0) {
#if CMK_NET_VERSION && ! CMK_SMP && ! defined(CMK_CPV_IS_SMP)
        if (CmiImmIsRunning())
#else
        if (CmiMyRank() == CmiMyNodeSize())
#endif
        {
          register QdCommMsg *msg = (QdCommMsg*) CkAllocMsg(0,sizeof(QdCommMsg),0);
          msg->flag = flag;
          msg->count = count;
          register envelope *env = UsrToEnv((void *)msg);
          CmiSetHandler(env, _qdCommHandlerIdx);
          CmiFreeSendFn(CmiNodeFirst(CmiMyNode()), env->getTotalsize(), (char *)env);
        }
  }
}

void CkStartQD(const CkCallback& cb)
{
  register QdMsg *msg = (QdMsg*) CkAllocMsg(0,sizeof(QdMsg),0);
  msg->setPhase(0);
  msg->setCb(cb);
  register envelope *env = UsrToEnv((void *)msg);
  CmiSetHandler(env, _qdHandlerIdx);
#if CMK_MEM_CHECKPOINT
  CmiGetRestartPhase(env) = 9999;        // make sure it is always executed
#endif
#if CMK_BIGSIM_CHARM
  CmiFreeSendFn(0, env->getTotalsize(), (char *)env);
#else
  _CldEnqueue(0, env, _infoIdx);
#endif
}

extern "C"
void CkStartQD(int eIdx, const CkChareID *cid)
{
  CkStartQD(CkCallback(eIdx, *cid));
}
