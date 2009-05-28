// #ifdef filippo

// #include <math.h>
// #include "pipelinestrategy.h"

// inline int log_of_2 (int i) {
//   int m;
//   for (m=0; i>(1<<m); ++m);
//   return m;
// }

// //PipelineHashKey CODE
// int PipelineHashKey::staticCompare(const void *k1,const void *k2,size_t ){
//     return ((const PipelineHashKey *)k1)->
//                 compare(*(const PipelineHashKey *)k2);
// }

// CkHashCode PipelineHashKey::staticHash(const void *v,size_t){
//     return ((const PipelineHashKey *)v)->hash();
// }

// void PipelineStrategy::commonInit(){
//   //log_of_2_inv = 1/log((double)2);
//   seqNumber = 0;
// }

// //extern void propagate_handler(void *);

// void deliver_handler(void *message) {
//   int instid = CmiGetXHandler(message);
//   PipelineStrategy *myStrategy = (PipelineStrategy*)ConvComlibGetStrategy(instid);
//   ComlibPrintf("[%d] propagate_handler_frag: calling on instid %d %x\n",CkMyPe(),instid,myStrategy);
//   //CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()->getStrategy(instid);
//   PipelineInfo *info = (PipelineInfo*)(((char*)message)+CmiReservedHeaderSize);
//   myStrategy->storing((char*)message);
// }

// void PipelineStrategy::storing(char* fragment) {
//   char *complete;
//   int isFinished=0;
//   int totalDimension;
//   //ComlibPrintf("isArray = %d\n", (getType() == ARRAY_STRATEGY));

//   // store the fragment in the hash table until completed
//   ComlibPrintf("[%d] deliverer: received fragmented message, storing\n",CkMyPe());
//   PipelineInfo *info = (PipelineInfo*)(fragment+CmiReservedHeaderSize);

//   PipelineHashKey key (info->bcastPe, info->seqNumber);
//   PipelineHashObj *position = fragments.get(key);

//   char *incomingMsg;
//   if (position) {
//     // the message already exist, add to it
//     ComlibPrintf("[%d] adding to an existing message for id %d/%d (%d remaining)\n",CkMyPe(),info->bcastPe,info->seqNumber,position->remaining-1);
//     incomingMsg = position->message;
//     memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipelineInfo))*info->chunkNumber), fragment+CmiReservedHeaderSize+sizeof(PipelineInfo), info->chunkSize);

//     if (--position->remaining == 0) {  // message completely received
//       isFinished = 1;
//       complete = incomingMsg;
//       totalDimension = position->dimension;
//       // delete from the hash table
//       fragments.remove(key);
//     }

//   } else {
//     // the message doesn't exist, create it
//     ComlibPrintf("[%d] creating new message of size %d for id %d/%d; chunk=%d chunkSize=%d\n",CkMyPe(),info->messageSize,info->bcastPe,info->seqNumber,info->chunkNumber,info->chunkSize);
//     incomingMsg = (char*)CmiAlloc(info->messageSize);
//     memcpy (incomingMsg, fragment, CmiReservedHeaderSize);
//     memcpy (incomingMsg+CmiReservedHeaderSize+((pipeSize-CmiReservedHeaderSize-sizeof(PipelineInfo))*info->chunkNumber), fragment+CmiReservedHeaderSize+sizeof(PipelineInfo), info->chunkSize);
//     int remaining = (int)ceil((double)info->messageSize/(pipeSize-CmiReservedHeaderSize-sizeof(PipelineInfo)))-1;
//     if (remaining) {  // more than one chunk (it was not forced to be splitted)
//       PipelineHashObj *object = new PipelineHashObj(info->messageSize, remaining, incomingMsg);
//       fragments.put(key) = object;
//     } else {  // only one chunk, it was forces to be splitted
//       isFinished = 1;
//       complete = incomingMsg;
//       // nothing to delete from fragments since nothing has been added
//     }
//   }
//   CmiFree(fragment);

//   if (isFinished) {
//     higherLevel->deliverer(complete, totalDimension);
//   }
// }

// void PipelineStrategy::deliverer(char *msg, int dimension) {
//   ComlibPrintf("{%d} dest = %d, %d, %x\n",CkMyPe(),destinationHandler, dimension,CmiHandlerToInfo(destinationHandler).hdlr);
//   if (destinationHandler) {
//     CmiSetHandler(msg, destinationHandler);
//     CmiSyncSendAndFree(CkMyPe(), dimension, msg);
//   } else {
//     CmiPrintf("[%d] Pipelined Broadcast: message not delivered since destination not set!");
//   }
// }

// PipelineStrategy::PipelineStrategy(int _pipeSize, Strategy *parent) : Strategy(), pipeSize(_pipeSize) {
//   if (parent) higherLevel = parent;
//   else higherLevel = this;
//   seqNumber = 0;
//   messageBuf = new CkQ<MessageHolder *>;
//   //if (!parent) propagateHandle_frag = CmiRegisterHandler((CmiHandler)propagate_handler_frag);
//   ComlibPrintf("init: %d (%x)\n",pipeSize,this);
//   //if (!parent) ComlibPrintf("[%d] registered handler fragmented to %d\n",CkMyPe(),propagateHandle_frag);
// }

// void PipelineStrategy::insertMessage(MessageHolder *cmsg){
//   ComlibPrintf("[%d] Pipelined Broadcast with converse strategy\n",CkMyPe());
//   messageBuf->enq(cmsg);
//   doneInserting();
// }

// void PipelineStrategy::doneInserting(){
//   ComlibPrintf("[%d] DoneInserting\n",CkMyPe());
//   while (!messageBuf->isEmpty()) {
//     MessageHolder *cmsg = messageBuf->deq();
//     // modify the Handler to deliver the message to the propagator
//     char *env = cmsg->getMessage();
//     //CmiSetHandler(env, deliverHandle);
//     conversePipeline(env, cmsg->getSize(), cmsg->dest_proc);
//     delete cmsg;
//     //conversePipeline(env, env->getTotalsize(), false);
//   }
// }

// // routine for interfacing with converse.
// // Require only the converse reserved header if forceSplit is true
// void PipelineStrategy::conversePipeline(char *env, int totalSize, int destination) {
//   // set the instance ID to be used by the receiver using the XHandler variable
//   CmiSetXHandler(env, myInstanceID);

//   ++seqNumber;
//   // message doesn't fit into the pipe: split it into chunks and propagate them individually
//   ComlibPrintf("[%d] Propagating message in multiple chunks (totalsize=%d)\n",CkMyPe(),totalSize);

//   char *sendingMsg;
//   char *nextChunk = env;//+CmiReservedHeaderSize;
//   int remaining = totalSize;//-CmiReservedHeaderSize;
//   int reducedPipe = pipeSize-CmiReservedHeaderSize-sizeof(PipelineInfo);
//   int sendingMsgSize;
//   ComlibPrintf("reducedPipe = %d, CmiReservedHeaderSize = %d, sizeof(PipelineInfo) = %d\n",reducedPipe,CmiReservedHeaderSize,sizeof(PipelineInfo));
//   ComlibPrintf("sending %d chunks of size %d, total=%d to handle %d\n",(int)ceil(((double)totalSize-CmiReservedHeaderSize)/reducedPipe),reducedPipe,remaining,deliverHandle);
//   //CmiSetHandler(env, deliverHandle);
//   ComlibPrintf("setting env handler to %d\n",deliverHandle);
//   for (int i=0; i<(int)ceil(((double)totalSize-CmiReservedHeaderSize)/reducedPipe); ++i) {
//     sendingMsgSize = reducedPipe<remaining? pipeSize : remaining+CmiReservedHeaderSize+sizeof(PipelineInfo);
//     sendingMsg = (char*)CmiAlloc(sendingMsgSize);
//     //memcpy (sendingMsg, env, CmiReservedHeaderSize);
//     CmiSetHandler(sendingMsg, deliverHandle);
//     PipelineInfo *info = (PipelineInfo*)(sendingMsg+CmiReservedHeaderSize);
//     info->srcPe = CkMyPe();
//     info->bcastPe = CkMyPe();
//     info->seqNumber = seqNumber;
//     info->chunkNumber = i;
//     info->chunkSize = reducedPipe<remaining ? reducedPipe : remaining;
//     info->messageSize = totalSize;
//     memcpy (sendingMsg+CmiReservedHeaderSize+sizeof(PipelineInfo), nextChunk, info->chunkSize);

//     remaining -= info->chunkSize;
//     nextChunk += info->chunkSize;

//     //propagate(sendingMsg, true, CkMyPe(), sendingMsgSize, NULL);
//     CmiSyncSendAndFree(destination, sendingMsgSize, sendingMsg);
//   }
//   CmiFree(env);
// }

// void PipelineStrategy::pup(PUP::er &p){
//   Strategy::pup(p);
//   ComlibPrintf("[%d] initial of Pipeconverse pup %s\n",CkMyPe(),(p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"));

//   p | pipeSize;
//   p | seqNumber;

//   ComlibPrintf("[%d] PipeBroadcast converse pupping %s, size=%d\n",CkMyPe(), (p.isPacking()==0)?(p.isUnpacking()?"UnPacking":"sizer"):("Packing"),pipeSize);

//   if (p.isUnpacking()) {
//     //log_of_2_inv = 1/log((double)2);
//     messageBuf = new CkQ<MessageHolder *>;
//     deliverHandle = CmiRegisterHandler((CmiHandler)deliver_handler);
//     //propagateHandle_frag = CmiRegisterHandler((CmiHandler)propagate_handler_frag);
//     //ComlibPrintf("[%d] registered handler to %d\n",CkMyPe(),deliverHandle);
//   }
//   if (p.isPacking()) {
//     delete messageBuf;
//   }
//   //p|(*messageBuf);
//   //p|fragments;

// }

// PUPable_def(PipelineStrategy);

// #endif
