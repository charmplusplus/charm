#include "PipeBroadcastStrategy.h"

//PipeBcastHashKey CODE
int PipeBcastHashKey::staticCompare(const void *k1,const void *k2,size_t ){
    return ((const PipeBcastHashKey *)k1)->
                compare(*(const PipeBcastHashKey *)k2);
}

CkHashCode PipeBcastHashKey::staticHash(const void *v,size_t){
    return ((const PipeBcastHashKey *)v)->hash();
}

CpvExtern(CkGroupID, cmgrID);

void propagate_handler(void *message) {
  // call the appropriate function PipeBroadcastStrategy::propagate
  //int instid = ((envelope *)message)->getEpIdx();
  //int instid = ((CkMcastBaseMsg*)(EnvToUsr((envelope*)message)))->_cookie.sInfo.cInfo.instId;
  int instid = CmiGetXHandler(message);
  PipeBroadcastStrategy *myStrategy = (PipeBroadcastStrategy *)CProxy_ComlibManager(CpvAccess(cmgrID)).ckLocalBranch()->getStrategy(instid);
  myStrategy->propagate((envelope *)message, false);
}

void propagate_handler_frag(void *message) {
  int instid = CmiGetXHandler(message);
  PipeBroadcastStrategy *myStrategy = (PipeBroadcastStrategy *)CProxy_ComlibManager(CpvAccess(cmgrID)).ckLocalBranch()->getStrategy(instid);
  myStrategy->propagate((envelope *)message, true);
}


void PipeBroadcastStrategy::propagate(envelope *env, int isFragmented){
  // find destination processors and send
  int destination, tmp, k;
  int num_pes, *dest_pes;
  PipeBcastInfo *info = (PipeBcastInfo*)(((char*)env)+CmiReservedHeaderSize);
  int srcPeNumber = isFragmented ? info->srcPe : env->getSrcPe();
  int totalSendingSize = isFragmented ? info->chunkSize+CmiReservedHeaderSize+sizeof(PipeBcastInfo) : env->getTotalsize();

  switch (topology) {
  case USE_LINEAR:
    if (srcPeNumber == (CmiMyPe()+1)%CmiNumPes()) break;
    destination = (CmiMyPe()+1) % CmiNumPes();
    ComlibPrintf("[%d] Pipebroadcast sending to %d\n",CmiMyPe(), destination);
    CmiSyncSend(destination, totalSendingSize, (char *)env);
    break;
  case USE_HYPERCUBE:
    num_pes=0;
    tmp = srcPeNumber ^ CmiMyPe();
    k = int(log(CmiNumPes()) * log_of_2_inv + 2);
    if (tmp) {
      do {--k;} while (!(tmp>>k));
    }
    ComlibPrintf("[%d] tmp=%d, k=%d\n",CmiMyPe(),tmp,k);
    // now 'k' is the last dimension in the hypercube used for exchange
    if (isFragmented) info->srcPe = CmiMyPe();
    else env->setSrcPe(CmiMyPe());  // where the message is coming from
    dest_pes = (int *)malloc(k*sizeof(int));
    --k;  // next dimension in the cube to be used
    for ( ; k>=0; --k) {
      // add the processor destination at level k if it exist
      dest_pes[num_pes] = CmiMyPe() ^ (1<<k);
      if (dest_pes[num_pes] >= CmiNumPes()) {
	dest_pes[num_pes] &= (-1)<<k;
	if (CmiNumPes()>dest_pes[num_pes]) dest_pes[num_pes] += (CmiMyPe() - (CmiMyPe() & ((-1)<<k))) % (CmiNumPes() - dest_pes[num_pes]);
      }
      if (dest_pes[num_pes] < CmiNumPes()) {
	ComlibPrintf("[%d] PipeBroadcast sending to %d\n",CmiMyPe(), dest_pes[num_pes]);
	++num_pes;
      }
    }
    //CmiSyncListSend(num_pes, dest_pes, env->getTotalsize(), (char *)env);
    for (k=0; k<num_pes; ++k) CmiSyncSend(dest_pes[k], totalSendingSize, (char *)env);
    free(dest_pes);
    break;

    // for other strategies

  default:
    // should NEVER reach here!
    CmiPrintf("Error, topology %d not known\n",topology);
    CkExit();
  }

  // deliver messages to local objects (i.e. send it to ComlibManager)
  deliverer(env, isFragmented);
  //CmiSetHandler(env, CmiGetXHandler(env));
  //CmiSyncSendAndFree(CmiMyPe(), env->getTotalsize(), (char *)env);

}

void PipeBroadcastStrategy::deliverer(envelope *env_frag, int isFragmented) {
  envelope *env;
  int isFinished=0;
  ComlibPrintf("isArrayDestination = %d\n",isArrayDestination);
  if (isArrayDestination) {
    // check if the message is fragmented
    if (isFragmented) {
      // store the fragment in the hash table until completed
      ComlibPrintf("[%d] deliverer: received fragmented message, storing\n",CkMyPe());
      PipeBcastInfo *info = (PipeBcastInfo*)(((char*)env_frag)+CmiReservedHeaderSize);

      PipeBcastHashKey key (info->bcastPe, info->seqNumber);
      PipeBcastHashObj *position = fragments.get(key);

      char *incomingMsg;
      if (position) {
	// the message already exist, add to it
	ComlibPrintf("[%d] adding to an existing message for id %d/%d (%d remaining)\n",CkMyPe(),info->bcastPe,info->seqNumber,position->remaining-1);
	incomingMsg = position->message;
	memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo))*info->chunkNumber), ((char*)env_frag)+CmiReservedHeaderSize+sizeof(PipeBcastInfo), info->chunkSize);

	if (--position->remaining == 0) {  // message completely received
	  isFinished = 1;
	  env = (envelope*)incomingMsg;
	  // delete from the hash table
	  fragments.remove(key);
	}

      } else {
	// the message doesn't exist, create it
	ComlibPrintf("[%d] creating new message of size %d for id %d/%d; chunk=%d chunkSize=%d\n",CkMyPe(),info->messageSize,info->bcastPe,info->seqNumber,info->chunkNumber,info->chunkSize);
	incomingMsg = (char*)CmiAlloc(info->messageSize);
	memcpy (incomingMsg, env_frag, CmiReservedHeaderSize);
	memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo))*info->chunkNumber), ((char*)env_frag)+CmiReservedHeaderSize+sizeof(PipeBcastInfo), info->chunkSize);
	int remaining = (int)ceil((double)info->messageSize/(pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo)))-1;
	if (remaining) {  // more than one chunk (it was not forced to be splitted)
	  PipeBcastHashObj *object = new PipeBcastHashObj(info->messageSize, remaining, incomingMsg);
	  fragments.put(key) = object;
	} else {  // only one chunk, it was forces to be splitted
	  isFinished = 1;
	  env = (envelope*)incomingMsg;
	  // nothing to delete from fragments since nothing has been added
	}
      }
      CmiFree(env_frag);

    } else {  // message not fragmented
      ComlibPrintf("[%d] deliverer: received message in single chunk\n",CkMyPe());
      isFinished = 1;
      env = env_frag;
    }

    if (isFinished) {
      CkArray *dest_array = CkArrayID::CkLocalBranch(destArrayID);
      localDest = new CkVec<CkArrayIndexMax>;
      dest_array->getComlibArrayListener()->getLocalIndices(*localDest);
      void *msg = EnvToUsr(env);
      CkArrayIndexMax idx;
      ArrayElement *elem;
      int ep = env->getsetArrayEp();
      CkUnpackMessage(&env);

      ComlibPrintf("[%d] deliverer: delivering a finished message\n",CkMyPe());
      for (int count = 0; count < localDest->size(); ++count) {
	idx = (*localDest)[count];
	ComlibPrintf("[%d] Sending message to ",CkMyPe());
	if (comm_debug) idx.print();

	CProxyElement_ArrayBase ap(destArrayID, idx);
	elem = ap.ckLocal();
	CkDeliverMessageReadonly (ep, msg, elem);
      }
      delete localDest;
      // the envelope env should be deleted only if the message is delivered
      CmiFree(env);
    }
  }
}

PipeBroadcastStrategy::PipeBroadcastStrategy()
  :topology(USE_HYPERCUBE), pipeSize(DEFAULT_PIPE), Strategy() {
  isArrayDestination = 0;
  commonInit();
}

PipeBroadcastStrategy::PipeBroadcastStrategy(int _topology)
  :topology(_topology), pipeSize(DEFAULT_PIPE), Strategy() {
  isArrayDestination = 0;
  commonInit();
}

PipeBroadcastStrategy::PipeBroadcastStrategy(int _topology, int _pipeSize)
  :topology(_topology), pipeSize(_pipeSize), Strategy() {
  isArrayDestination = 0;
  commonInit();
}

PipeBroadcastStrategy::PipeBroadcastStrategy(int _topology, CkArrayID aid)
  :topology(_topology), destArrayID(aid), pipeSize(DEFAULT_PIPE), Strategy() {
  isArrayDestination = 1;
  CmiPrintf("init: %d %d\n",topology, pipeSize);
  commonInit();
}

PipeBroadcastStrategy::PipeBroadcastStrategy(int _topology, CkArrayID aid, int _pipeSize)
  :topology(_topology), destArrayID(aid), pipeSize(_pipeSize), Strategy() {
  isArrayDestination = 1;
  commonInit();
}

void PipeBroadcastStrategy::commonInit(){
  log_of_2_inv = 1/log(2);
  seqNumber = 0;
}

void PipeBroadcastStrategy::insertMessage(CharmMessageHolder *cmsg){
  ComlibPrintf("[%d] Pipelined Broadcast with strategy %d\n",CkMyPe(),topology);
  messageBuf->enq(cmsg);
  doneInserting();
}

// routine for interfacing with converse.
// Require only the converse reserved header if forceSplit is true
void PipeBroadcastStrategy::conversePipeBcast(envelope *env, int totalSize, int forceSplit) {
  // set the instance ID to be used by the receiver using the XHandler variable
  CmiSetXHandler(env, myInstanceID);

  if (totalSize > pipeSize || forceSplit) {
    ++seqNumber;
    // message doesn't fit into the pipe: split it into chunks and propagate them individually
    ComlibPrintf("[%d] Propagating message in multiple chunks\n",CkMyPe());

    char *sendingMsg;
    char *nextChunk = ((char*)env)+CmiReservedHeaderSize;
    int remaining = totalSize-CmiReservedHeaderSize;
    int reducedPipe = pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo);
    ComlibPrintf("reducedPipe = %d, CmiReservedHeaderSize = %d, sizeof(PipeBcastInfo) = %d\n",reducedPipe,CmiReservedHeaderSize,sizeof(PipeBcastInfo));
    ComlibPrintf("sending %d chunks of size %d, total=%d\n",(int)ceil(((double)totalSize-CmiReservedHeaderSize)/reducedPipe),reducedPipe,remaining);
    for (int i=0; i<(int)ceil(((double)totalSize-CmiReservedHeaderSize)/reducedPipe); ++i) {
      sendingMsg = (char*)CmiAlloc(pipeSize);
      CmiSetHandler(env, propagateHandle_frag);
      memcpy (sendingMsg, env, CmiReservedHeaderSize);
      PipeBcastInfo *info = (PipeBcastInfo*)(sendingMsg+CmiReservedHeaderSize);
      info->srcPe = CmiMyPe();
      info->bcastPe = CmiMyPe();
      info->seqNumber = seqNumber;
      info->chunkNumber = i;
      info->chunkSize = reducedPipe<remaining ? reducedPipe : remaining;
      info->messageSize = totalSize;
      memcpy (sendingMsg+CmiReservedHeaderSize+sizeof(PipeBcastInfo), nextChunk, reducedPipe);

      remaining -= reducedPipe;
      nextChunk += reducedPipe;

      propagate((envelope*)sendingMsg, true);
    }

  } else {
    // the message fit into the pipe, so send it in a single chunk
    ComlibPrintf("[%d] Propagating message in one single chunk\n",CkMyPe());
    CmiSetHandler(env, propagateHandle);
    env->setSrcPe(CmiMyPe());
    //env->setEpIdx(myInstanceID);
    propagate(env, false);
  }
}

void PipeBroadcastStrategy::doneInserting(){
  ComlibPrintf("[%d] DoneInserting\n",CkMyPe());
  while (!messageBuf->isEmpty()) {
    CharmMessageHolder *cmsg = messageBuf->deq();
    // modify the Handler to deliver the message to the propagator
    envelope *env = UsrToEnv(cmsg->getCharmMessage());

    conversePipeBcast(env, env->getTotalsize(), false);
  }
}

void PipeBroadcastStrategy::pup(PUP::er &p){
  ComlibPrintf("[%d] PipeBroadcast pupping %s\n",CkMyPe(), (p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"));
  Strategy::pup(p);
  p|pipeSize;
  p|topology;
  p|seqNumber;
  p|isArrayDestination;
  p|destArrayID;

  if (p.isUnpacking()) {
    log_of_2_inv = 1/log(2);
    messageBuf = new CkQ<CharmMessageHolder *>;
    propagateHandle = CmiRegisterHandler((CmiHandler)propagate_handler);
    propagateHandle_frag = CmiRegisterHandler((CmiHandler)propagate_handler_frag);
  }
  //p|messageBuf;
  //p|fragments;

}

//PUPable_def(PipeBroadcastStrategy);
