#include "PipeBroadcastStrategy.h"

void propagate_handler(void *message) {
  int instid = CmiGetXHandler(message);
  PipeBroadcastConverse *myStrategy = (PipeBroadcastConverse *)ConvComlibGetStrategy(instid);
  envelope *env = (envelope*)message;
  myStrategy->propagate((char*)message, false, env->getSrcPe(), env->getTotalsize(), &envelope::setSrcPe);
}

void PipeBroadcastStrategy::deliverer(char *msg, int dim) {
  envelope *env = (envelope*)msg;
  ComlibPrintf("isArray = %d\n", (getType() == ARRAY_STRATEGY));

  if (getType() == ARRAY_STRATEGY) {
    // deliver the message to the predefined group "ainfo"
    ComlibPrintf("[%d] deliverer: delivering a finished message\n",CkMyPe());
    ainfo.localBroadcast(env);
  }

  if (getType() == GROUP_STRATEGY) {
    // deliver the message to the predifined group "ginfo"
    CkGroupID gid;
    ginfo.getSourceGroup(gid);
    CkSendMsgBranchInline(env->getEpIdx(), EnvToUsr(env), CkMyPe(), gid);
  }
}

void PipeBroadcastStrategy::commonInit(int _topology, int _pipeSize) {
  converseStrategy = new PipeBroadcastConverse(_topology, _pipeSize, this);
}

PipeBroadcastStrategy::PipeBroadcastStrategy(int _topology, int _pipeSize)
  : CharmStrategy() {
  //isArray = 0;
  commonInit(_topology, _pipeSize);
}

PipeBroadcastStrategy::PipeBroadcastStrategy(int _topology, CkArrayID _aid, int _pipeSize)
  : CharmStrategy() {
  setType(ARRAY_STRATEGY);
  ainfo.setDestinationArray(_aid);
  commonInit(_topology, _pipeSize);
}

PipeBroadcastStrategy::PipeBroadcastStrategy(CkGroupID _gid, int _topology, int _pipeSize)
  : CharmStrategy() {
  setType(GROUP_STRATEGY);
  ginfo.setSourceGroup(_gid);
  commonInit(_topology, _pipeSize);
}

void PipeBroadcastStrategy::insertMessage(CharmMessageHolder *cmsg){
  messageBuf->enq(cmsg);
  doneInserting();
}

// routine for interfacing with converse.
// Require only the converse reserved header if forceSplit is true
void PipeBroadcastStrategy::conversePipeBcast(envelope *env, int totalSize) {
  // set the instance ID to be used by the receiver using the XHandler variable
  CmiSetXHandler(env, myInstanceID);

  if (totalSize > ((PipeBroadcastConverse*)converseStrategy)->getPipeSize()) {
    ((PipeBroadcastConverse*)converseStrategy)->conversePipeBcast((char*)env, totalSize);
  } else {
    // the message fit into the pipe, so send it in a single chunk
    ComlibPrintf("[%d] Propagating message in one single chunk\n",CkMyPe());
    CmiSetHandler(env, propagateHandle);
    env->setSrcPe(CkMyPe());
    ((PipeBroadcastConverse*)converseStrategy)->propagate((char*)env, false, CkMyPe(), totalSize, &envelope::setSrcPe);
  }
}

void PipeBroadcastStrategy::doneInserting(){
  ComlibPrintf("[%d] DoneInserting\n",CkMyPe());
  while (!messageBuf->isEmpty()) {
    CharmMessageHolder *cmsg = messageBuf->deq();
    // modify the Handler to deliver the message to the propagator
    envelope *env = UsrToEnv(cmsg->getCharmMessage());

    conversePipeBcast(env, env->getTotalsize());
  }
}

void PipeBroadcastStrategy::pup(PUP::er &p){
  ComlibPrintf("[%d] PipeBroadcast pupping %s\n",CkMyPe(), (p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"));
  CharmStrategy::pup(p);

  if (p.isUnpacking()) {
    converseStrategy = new PipeBroadcastConverse(0,0,this);
  }
  p | *converseStrategy;

  if (p.isUnpacking()) {
    propagateHandle = CmiRegisterHandler((CmiHandler)propagate_handler);
    messageBuf = new CkQ<CharmMessageHolder *>;
    converseStrategy->setHigherLevel(this);
  }
}

//PUPable_def(PipeBroadcastStrategy);
