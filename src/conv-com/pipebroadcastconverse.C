/**
   @addtogroup ComlibConverseStrategy
   @{
   @file
   Implementation of the PipeBroadcastConverse strategy
*/

#include <math.h>
#include "pipebroadcastconverse.h"

inline int log_of_2 (int i) {
  int m;
  for (m=0; i>(1<<m); ++m);
  return m;
}

//PipeBcastHashKey CODE
int PipeBcastHashKey::staticCompare(const void *k1,const void *k2,size_t ){
    return ((const PipeBcastHashKey *)k1)->
                compare(*(const PipeBcastHashKey *)k2);
}

CkHashCode PipeBcastHashKey::staticHash(const void *v,size_t){
    return ((const PipeBcastHashKey *)v)->hash();
}

/*
void PipeBroadcastConverse::commonInit(){
  //log_of_2_inv = 1/log((double)2);
  seqNumber = 0;
}
*/

//extern void propagate_handler(void *);

CkpvDeclare(int, pipeline_handler);
/**
 * Converse handler for messages broadcasted through PipeBroadcastConverse and
 * subclasses when fragmentation is needed. The message in this case has always
 * a PipeBcastInfo structure right after the converse header.
 */
void PipelineFragmentHandler(void *message) {
  int instid = CmiGetStrategy(message);
  PipeBroadcastConverse *myStrategy = (PipeBroadcastConverse*)ConvComlibGetStrategy(instid);
  ComlibPrintf("[%d] PipelineFragmentHandler: %d\n",CkMyPe(),instid);
  //  PipeBcastInfo *info = (PipeBcastInfo*)(((char*)message)+CmiReservedHeaderSize);
  myStrategy->propagate((char*)message, true);//, info->srcPe, info->chunkSize+CmiReservedHeaderSize+sizeof(PipeBcastInfo), NULL);
}

CkpvDeclare(int, pipeline_frag_handler);
/**
 * Converse handler for messages broadcasted through PipeBroadcastConverse and
 * subclasses when no fragmentation is needed (i.e the total size is less than
 * the pipeSize)
 */
void PipelineHandler(void *message) {
  int instid = CmiGetStrategy(message);
  PipeBroadcastConverse *myStrategy = (PipeBroadcastConverse*)ConvComlibGetStrategy(instid);
  ComlibPrintf("[%d] PipelineHandler: %d\n",CkMyPe(),instid);
  //PipeBcastInfo *info = (PipeBcastInfo*)(((char*)message)+CmiReservedHeaderSize);
  myStrategy->propagate((char*)message, false);
}

PipeBroadcastConverse::PipeBroadcastConverse(short _topology, int _pipeSize) : Strategy(), topology(_topology), pipeSize(_pipeSize) {
  seqNumber = 0;
  //messageBuf = new CkQ<MessageHolder *>;
  //if (!parent) propagateHandle_frag = CmiRegisterHandler((CmiHandler)propagate_handler_frag);
  ComlibPrintf("[%d] PipeBroadcastConverse constructor: %d %d\n",CkMyPe(),topology, pipeSize);
  //if (!parent) ComlibPrintf("[%d] registered handler fragmented to %d\n",CkMyPe(),propagateHandle_frag);
}

CmiFragmentHeader *PipeBroadcastConverse::getFragmentHeader(char *msg) {
  return (CmiFragmentHeader*)(msg+CmiReservedHeaderSize);
}

void PipeBroadcastConverse::propagate(char *msg, int isFragmented) {//, int srcPeNumber, int totalSendingSize, setFunction setPeNumber){
  // find destination processors and send
  int destination, tmp, k; //, sizeToSend;
  int num_pes, *dest_pes;
  //int srcPeNumber = isFragmented ? info->srcPe : env->getSrcPe();
  //int totalSendingSize = isFragmented ? info->chunkSize+CmiReservedHeaderSize+sizeof(PipeBcastInfo) : env->getTotalsize();

  // get the information about sourcePe and message size
  int srcPeNumber, totalSendingSize;
  CmiFragmentHeader *frag = NULL;
  PipeBcastInfo *info = NULL;
  if (isFragmented) {
    info = (PipeBcastInfo*)(msg+CmiReservedHeaderSize);
    srcPeNumber = info->srcPe;
    totalSendingSize = info->messageSize;
  } else {
    frag = getFragmentHeader(msg);
    srcPeNumber = frag->senderPe;
    totalSendingSize = frag->msgSize;
  }

  switch (topology) {
  case USE_LINEAR:
    if (srcPeNumber == (CkMyPe()+1)%CkNumPes()) break;
    destination = (CkMyPe()+1) % CkNumPes();
    ComlibPrintf("[%d] Pipebroadcast sending to %d\n",CkMyPe(), destination);
    CmiSyncSend(destination, totalSendingSize, msg);
    break;
  case USE_HYPERCUBE:
    tmp = srcPeNumber ^ CkMyPe();
    k = log_of_2(CkNumPes()) + 2;
    if (tmp) {
      do {--k;} while (!(tmp>>k));
    }
    ComlibPrintf("[%d] tmp=%d, k=%d\n",CkMyPe(),tmp,k);
    // now 'k' is the last dimension in the hypercube used for exchange
    if (isFragmented) info->srcPe = CkMyPe();
    else frag->senderPe = CkMyPe();
    dest_pes = (int *)malloc(k*sizeof(int));
    --k;  // next dimension in the cube to be used
    num_pes = HypercubeGetBcastDestinations(CkMyPe(), CkNumPes(), k, dest_pes);

    /*
    for ( ; k>=0; --k) {
      // add the processor destination at level k if it exist
      dest_pes[num_pes] = CkMyPe() ^ (1<<k);
      if (dest_pes[num_pes] >= CkNumPes()) {
	dest_pes[num_pes] &= (-1)<<k;
	if (CkNumPes()>dest_pes[num_pes]) dest_pes[num_pes] += (CkMyPe() - (CkMyPe() & ((-1)<<k))) % (CkNumPes() - dest_pes[num_pes]);
      }
      if (dest_pes[num_pes] < CkNumPes()) {
	ComlibPrintf("[%d] PipeBroadcast sending to %d\n",CkMyPe(), dest_pes[num_pes]);
	++num_pes;
      }
    }
    */

    //CmiSyncListSend(num_pes, dest_pes, env->getTotalsize(), (char *)env);
#ifdef CMI_COMLIB_WITH_REFERENCE
    for (k=0; k<num_pes; ++k) {
      //ComlibPrintf("[%d] PipeBroadcast sending to %d\n",CkMyPe(), dest_pes[k]);
      CmiReference(msg);
      CmiSyncSendAndFree(dest_pes[k], totalSendingSize, msg);
    }
#else
    CmiSyncListSend(num_pes, dest_pes, totalSendingSize, msg);
#endif
    //sizeToSend = pipeSize<totalSendingSize ? pipeSize : totalSendingSize;
    //for (k=0; k<num_pes; ++k) CmiSyncSend(dest_pes[k], sizeToSend, env);
    free(dest_pes);
    break;

    // for other strategies

  default:
    // should NEVER reach here!
    char error_msg[100];
    sprintf(error_msg, "Error, topology %d not known\n",topology);
    CmiAbort(error_msg);
  }

  // decide what to do after with the message
  if (isFragmented) store(msg);
  else deliver(msg, totalSendingSize);

  // deliver messages to local objects (i.e. send it to ComlibManager)
  //storing(env, isFragmented);
  //CmiSetHandler(env, CmiGetXHandler(env));
  //CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), (char *)env);

}

// this function is called only on fragmented messages
void PipeBroadcastConverse::store(char* fragment) {
  //char *complete;
  //int isFinished=0;
  //int totalDimension;
  //ComlibPrintf("isArray = %d\n", (getType() == ARRAY_STRATEGY));

  // check if the message is fragmented
  //if (isFragmented) {
  // store the fragment in the hash table until completed
  //ComlibPrintf("[%d] deliverer: received fragmented message, storing\n",CkMyPe());
  PipeBcastInfo *info = (PipeBcastInfo*)(fragment+CmiReservedHeaderSize);

  PipeBcastHashKey key (info->bcastPe, info->seqNumber);
  PipeBcastHashObj *position = fragments.get(key);

  char *incomingMsg;
  if (position) {
    // the message already exist, add to it
    ComlibPrintf("[%d] adding to an existing message for id %d/%d (%d remaining)\n",CkMyPe(),info->bcastPe,info->seqNumber,position->remaining-1);
    incomingMsg = position->message;
    memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo))*info->chunkNumber), fragment+CmiReservedHeaderSize+sizeof(PipeBcastInfo), info->chunkSize);

    if (--position->remaining == 0) {  // message completely received
      // the deliver function will take care of deleting the message
      deliver(incomingMsg, position->dimension);
      //isFinished = 1;
      //complete = incomingMsg;
      //totalDimension = position->dimension;
      // delete from the hash table
      fragments.remove(key);
      delete position;
    }

  } else {
    // the message doesn't exist, create it
    ComlibPrintf("[%d] creating new message of size %d for id %d/%d; chunk=%d chunkSize=%d\n",CkMyPe(),info->messageSize,info->bcastPe,info->seqNumber,info->chunkNumber,info->chunkSize);
    incomingMsg = (char*)CmiAlloc(info->messageSize);
    memcpy (incomingMsg, fragment, CmiReservedHeaderSize);
    memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo))*info->chunkNumber), fragment+CmiReservedHeaderSize+sizeof(PipeBcastInfo), info->chunkSize);
    int remaining = (int)ceil((double)info->messageSize/(pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo)))-1;
    CmiAssert(remaining > 0);
    //if (remaining) {  // more than one chunk (it was not forced to be splitted)
    PipeBcastHashObj *object = new PipeBcastHashObj(info->messageSize, remaining, incomingMsg);
    fragments.put(key) = object;
    /*
      } else {  // only one chunk, it was forces to be splitted
      isFinished = 1;
      complete = incomingMsg;
      // nothing to delete from fragments since nothing has been added
      }
    */
  }
  CmiFree(fragment);

  /*
    } else {  // message not fragmented
    ComlibPrintf("[%d] deliverer: received message in single chunk\n",CkMyPe());
    isFinished = 1;
    complete = fragment;
    }
  */

  //if (isFinished) {
  //}
}

void PipeBroadcastConverse::deliver(char *msg, int dimension) {
  //ComlibPrintf("{%d} dest = %d, %d, %x\n",CkMyPe(),destinationHandler, dimension,CmiHandlerToInfo(destinationHandler).hdlr);
  CmiFragmentHeader *info = (CmiFragmentHeader*)(msg+CmiReservedHeaderSize);
  CmiSetHandler(msg, info->destination);
  CmiSyncSendAndFree(CkMyPe(), dimension, msg);
  /*
  if (destinationHandler) {
    CmiSetHandler(msg, destinationHandler);
    CmiSyncSendAndFree(CkMyPe(), dimension, msg);
  } else {
    CmiPrintf("[%d] Pipelined Broadcast: message not delivered since destination not set!");
  }
  */
}

void PipeBroadcastConverse::insertMessage(MessageHolder *cmsg){
  ComlibPrintf("[%d] PipeBroadcastConverse::insertMessage %d\n",CkMyPe(),topology);
  char *msg = cmsg->getMessage();
  int size = cmsg->getSize();
  if (size < pipeSize) {
    // sending message in a single chunk
    CmiSetHandler(msg, CkpvAccess(pipeline_handler));
    CmiFragmentHeader *frag = getFragmentHeader(msg);
    frag->senderPe = CkMyPe();
    frag->msgSize = size;
    propagate(msg, false);

  } else {
    // sending message in multiple chunk: message doesn't fit into the pipe:
    // split it into chunks and propagate them individually
    ++seqNumber;
    ComlibPrintf("[%d] Propagating message in multiple chunks (totalsize=%d)\n",CkMyPe(),size);

    char *sendingMsg;
    char *nextChunk = msg+CmiReservedHeaderSize;
    int remaining = size-CmiReservedHeaderSize;
    int reducedPipe = pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo);
    int sendingMsgSize;
    CmiSetHandler(msg, CkpvAccess(pipeline_frag_handler));

    // send all the chunks one after the other
    for (int i=0; i<(int)ceil(((double)size-CmiReservedHeaderSize)/reducedPipe); ++i) {
      sendingMsgSize = reducedPipe<remaining? pipeSize : remaining+CmiReservedHeaderSize+sizeof(PipeBcastInfo);
      sendingMsg = (char*)CmiAlloc(sendingMsgSize);
      memcpy (sendingMsg, msg, CmiReservedHeaderSize);
      PipeBcastInfo *info = (PipeBcastInfo*)(sendingMsg+CmiReservedHeaderSize);
      info->srcPe = CkMyPe();
      info->bcastPe = CkMyPe();
      info->seqNumber = seqNumber;
      info->chunkNumber = i;
      info->chunkSize = reducedPipe<remaining ? reducedPipe : remaining;
      info->messageSize = size;
      memcpy (sendingMsg+CmiReservedHeaderSize+sizeof(PipeBcastInfo), nextChunk, info->chunkSize);

      remaining -= info->chunkSize;
      nextChunk += info->chunkSize;

      propagate(sendingMsg, true);
    }

  }
  //CmiSetHandler(msg, CsvAccess(pipeBcastPropagateHandle_frag));
  //conversePipeBcast(msg, cmsg->getSize());
  delete cmsg;
}

/*
void PipeBroadcastConverse::doneInserting(){
  ComlibPrintf("[%d] DoneInserting\n",CkMyPe());
  while (!messageBuf->isEmpty()) {
    MessageHolder *cmsg = messageBuf->deq();
    // modify the Handler to deliver the message to the propagator
    char *env = cmsg->getMessage();
    CmiSetHandler(env, CsvAccess(pipeBcastPropagateHandle_frag));
    conversePipeBcast(env, cmsg->getSize());
    delete cmsg;
    //conversePipeBcast(env, env->getTotalsize(), false);
  }
}
*/

/*
// routine for interfacing with converse.
// Require only the converse reserved header if forceSplit is true
void PipeBroadcastConverse::conversePipeBcast(char *msg, int totalSize) {
  // set the instance ID to be used by the receiver using the XHandler variable
  //CmiSetXHandler(env, myInstanceID);

  ++seqNumber;
  // message doesn't fit into the pipe: split it into chunks and propagate them individually
  ComlibPrintf("[%d] Propagating message in multiple chunks (totalsize=%d)\n",CkMyPe(),totalSize);

  char *sendingMsg;
  char *nextChunk = msg+CmiReservedHeaderSize;
  int remaining = totalSize-CmiReservedHeaderSize;
  int reducedPipe = pipeSize-CmiReservedHeaderSize-sizeof(PipeBcastInfo);
  int sendingMsgSize;
  //ComlibPrintf("reducedPipe = %d, CmiReservedHeaderSize = %d, sizeof(PipeBcastInfo) = %d\n",reducedPipe,CmiReservedHeaderSize,sizeof(PipeBcastInfo));
  //ComlibPrintf("sending %d chunks of size %d, total=%d to handle %d\n",(int)ceil(((double)totalSize-CmiReservedHeaderSize)/reducedPipe),reducedPipe,remaining,CsvAccess(pipeBcastPropagateHandle_frag));
  CmiSetHandler(msg, CsvAccess(pipeline_frag_handler));
  //ComlibPrintf("setting env handler to %d\n",CsvAccess(pipeBcastPropagateHandle_frag));

  // send all the chunks one after the other
  for (int i=0; i<(int)ceil(((double)totalSize-CmiReservedHeaderSize)/reducedPipe); ++i) {
    sendingMsgSize = reducedPipe<remaining? pipeSize : remaining+CmiReservedHeaderSize+sizeof(PipeBcastInfo);
    sendingMsg = (char*)CmiAlloc(sendingMsgSize);
    memcpy (sendingMsg, env, CmiReservedHeaderSize);
    PipeBcastInfo *info = (PipeBcastInfo*)(sendingMsg+CmiReservedHeaderSize);
    info->srcPe = CkMyPe();
    info->bcastPe = CkMyPe();
    info->seqNumber = seqNumber;
    info->chunkNumber = i;
    info->chunkSize = reducedPipe<remaining ? reducedPipe : remaining;
    info->messageSize = totalSize;
    memcpy (sendingMsg+CmiReservedHeaderSize+sizeof(PipeBcastInfo), nextChunk, info->chunkSize);

    remaining -= info->chunkSize;
    nextChunk += info->chunkSize;

    propagate(sendingMsg, true);//, CkMyPe(), sendingMsgSize, NULL);
  }
  CmiFree(msg);
}
*/

void PipeBroadcastConverse::pup(PUP::er &p){
  Strategy::pup(p);
  ComlibPrintf("[%d] initial of PipeBroadcastConverse::pup %s\n",CkMyPe(),(p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"));

  p | pipeSize;
  p | topology;
  p | seqNumber;

  //ComlibPrintf("[%d] PipeBroadcast converse pupping %s, size=%d, topology=%d\n",CkMyPe(), (p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"),pipeSize,topology);

  /*
  if (p.isUnpacking()) {
    //log_of_2_inv = 1/log((double)2);
    messageBuf = new CkQ<MessageHolder *>;
    //propagateHandle_frag = CmiRegisterHandler((CmiHandler)propagate_handler_frag);
    ComlibPrintf("[%d] registered handler fragmented to %d\n",CkMyPe(),CsvAccess(pipeBcastPropagateHandle_frag));
  }
  if (p.isPacking()) {
    delete messageBuf;
  }
  //p|(*messageBuf);
  //p|fragments;
  */
}

PUPable_def(PipeBroadcastConverse)

/*@}*/
