#include "pipebroadcaststrategy.h"

//PipeBcastHashKey CODE
int PipeBcastHashKey::staticCompare(const void *k1,const void *k2,size_t ){
    return ((const PipeBcastHashKey *)k1)->
                compare(*(const PipeBcastHashKey *)k2);
}

CkHashCode PipeBcastHashKey::staticHash(const void *v,size_t){
    return ((const PipeBcastHashKey *)v)->hash();
}

void PipeBroadcastConverse::commonInit(){
  log_of_2_inv = 1/log((double)2);
  seqNumber = 0;
}

extern void propagate_handler(void *);

void propagate_handler_frag(void *message) {
  int instid = CmiGetXHandler(message);
  PipeBroadcastConverse *myStrategy = (PipeBroadcastConverse*)ConvComlibGetStrategy(instid);
  //CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()->getStrategy(instid);
  PipeBcastInfo *info = (PipeBcastInfo*)(((char*)message)+CmiReservedHeaderSize);
  myStrategy->propagate((char*)message, true, info->srcPe, info->chunkSize+CmiReservedHeaderSize+sizeof(PipeBcastInfo), NULL);
}

void PipeBroadcastConverse::propagate(char *env, int isFragmented, int srcPeNumber, int totalSendingSize, setFunction setPeNumber){
  // find destination processors and send
  int destination, tmp, k;
  int num_pes, *dest_pes;
  PipeBcastInfo *info = (PipeBcastInfo*)(env+CmiReservedHeaderSize);
  //int srcPeNumber = isFragmented ? info->srcPe : env->getSrcPe();
  //int totalSendingSize = isFragmented ? info->chunkSize+CmiReservedHeaderSize+sizeof(PipeBcastInfo) : env->getTotalsize();

  switch (topology) {
  case USE_LINEAR:
    if (srcPeNumber == (CmiMyPe()+1)%CmiNumPes()) break;
    destination = (CmiMyPe()+1) % CmiNumPes();
    ComlibPrintf("[%d] Pipebroadcast sending to %d\n",CmiMyPe(), destination);
    CmiSyncSend(destination, totalSendingSize, env);
    break;
  case USE_HYPERCUBE:
    tmp = srcPeNumber ^ CmiMyPe();
    k = int(log((double)CmiNumPes()) * log_of_2_inv + 2);
    if (tmp) {
      do {--k;} while (!(tmp>>k));
    }
    ComlibPrintf("[%d] tmp=%d, k=%d\n",CmiMyPe(),tmp,k);
    // now 'k' is the last dimension in the hypercube used for exchange
    if (isFragmented) info->srcPe = CmiMyPe();
    else setPeNumber(env,CmiMyPe());  // where the message is coming from
    dest_pes = (int *)malloc(k*sizeof(int));
    --k;  // next dimension in the cube to be used
    num_pes = HypercubeGetBcastDestinations(k, dest_pes);

    /*
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
    */

    //CmiSyncListSend(num_pes, dest_pes, env->getTotalsize(), (char *)env);
    for (k=0; k<num_pes; ++k) CmiSyncSend(dest_pes[k], totalSendingSize, env);
    free(dest_pes);
    break;

    // for other strategies

  default:
    // should NEVER reach here!
    CmiPrintf("Error, topology %d not known\n",topology);
    CkExit();
  }

  // deliver messages to local objects (i.e. send it to ComlibManager)
  storing(env, isFragmented);
  //CmiSetHandler(env, CmiGetXHandler(env));
  //CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), (char *)env);

}

void PipeBroadcastConverse::storing(char* fragment, int isFragmented) {
  char *complete;
  int isFinished=0;
  //ComlibPrintf("isArray = %d\n", (getType() == ARRAY_STRATEGY));

  // check if the message is fragmented
  if (isFragmented) {
    // store the fragment in the hash table until completed
    ComlibPrintf("[%d] deliverer: received fragmented message, storing\n",CkMyPe());
    PipeBcastInfo *info = (PipeBcastInfo*)(fragment+CmiReservedHeaderSize);

    PipeBcastHashKey key (info->bcastPe, info->seqNumber);
    PipeBcastHashObj *position = fragments.get(key);

    char *incomingMsg;
    if (position) {
      // the message already exist, add to it
      ComlibPrintf("[%d] adding to an existing message for id %d/%d (%d remaining)\n",CmiMyPe(),info->bcastPe,info->seqNumber,position->remaining-1);
      incomingMsg = position->message;
      memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo))*info->chunkNumber), fragment+CmiReservedHeaderSize+sizeof(PipeBcastInfo), info->chunkSize);

      if (--position->remaining == 0) {  // message completely received
	isFinished = 1;
	complete = incomingMsg;
	// delete from the hash table
	fragments.remove(key);
      }

    } else {
      // the message doesn't exist, create it
      ComlibPrintf("[%d] creating new message of size %d for id %d/%d; chunk=%d chunkSize=%d\n",CmiMyPe(),info->messageSize,info->bcastPe,info->seqNumber,info->chunkNumber,info->chunkSize);
      incomingMsg = (char*)CmiAlloc(info->messageSize);
      memcpy (incomingMsg, fragment, CmiReservedHeaderSize);
      memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo))*info->chunkNumber), fragment+CmiReservedHeaderSize+sizeof(PipeBcastInfo), info->chunkSize);
      int remaining = (int)ceil((double)info->messageSize/(pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo)))-1;
      if (remaining) {  // more than one chunk (it was not forced to be splitted)
	PipeBcastHashObj *object = new PipeBcastHashObj(info->messageSize, remaining, incomingMsg);
	fragments.put(key) = object;
      } else {  // only one chunk, it was forces to be splitted
	isFinished = 1;
	complete = incomingMsg;
	// nothing to delete from fragments since nothing has been added
      }
    }
    CmiFree(fragment);

  } else {  // message not fragmented
    ComlibPrintf("[%d] deliverer: received message in single chunk\n",CmiMyPe());
    isFinished = 1;
    complete = fragment;
  }

  if (isFinished) {
    higherLevel->deliverer(complete);
  }
}

void PipeBroadcastConverse::deliverer(char *msg) {
  // TO BE DONE
}

PipeBroadcastConverse::PipeBroadcastConverse(int _topology, int _pipeSize, Strategy *parent) : Strategy(), topology(_topology), pipeSize(_pipeSize) {
  higherLevel = parent;
  seqNumber = 0;
  ComlibPrintf("init: %d %d\n",topology, pipeSize);
}

void PipeBroadcastConverse::insertMessage(MessageHolder *cmsg){
  ComlibPrintf("[%d] Pipelined Broadcast with converse strategy %d\n",CkMyPe(),topology);
  messageBuf->enq(cmsg);
  doneInserting();
}

void PipeBroadcastConverse::doneInserting(){
  ComlibPrintf("[%d] DoneInserting\n",CkMyPe());
  while (!messageBuf->isEmpty()) {
    MessageHolder *cmsg = messageBuf->deq();
    // modify the Handler to deliver the message to the propagator
    char *env = cmsg->getMessage();

    //conversePipeBcast(env, env->getTotalsize(), false);
  }
}

// routine for interfacing with converse.
// Require only the converse reserved header if forceSplit is true
void PipeBroadcastConverse::conversePipeBcast(char *env, int totalSize) {
  // set the instance ID to be used by the receiver using the XHandler variable
  CmiSetXHandler(env, myInstanceID);

  ++seqNumber;
  // message doesn't fit into the pipe: split it into chunks and propagate them individually
  ComlibPrintf("[%d] Propagating message in multiple chunks (totalsize=%d)\n",CmiMyPe(),totalSize);

  char *sendingMsg;
  char *nextChunk = env+CmiReservedHeaderSize;
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

    propagate(sendingMsg, true, CmiMyPe(), totalSize, NULL);
  }
}

void PipeBroadcastConverse::pup(PUP::er &p){
  Strategy::pup(p);
  ComlibPrintf("[%d] initial of Pipeconverse pup %s\n",CmiMyPe(),(p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"));

  p | pipeSize;
  p | topology;
  p | seqNumber;

  ComlibPrintf("[%d] PipeBroadcast converse pupping %s, size=%d, topology=%d\n",CmiMyPe(), (p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"),pipeSize,topology);

  if (p.isUnpacking()) {
    log_of_2_inv = 1/log((double)2);
    messageBuf = new CkQ<MessageHolder *>;
    propagateHandle_frag = CmiRegisterHandler((CmiHandler)propagate_handler_frag);
  }
  //p|(*messageBuf);
  //p|fragments;

}

PUPable_def(PipeBroadcastConverse);
