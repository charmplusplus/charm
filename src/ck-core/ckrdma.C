/*
 * Charm Onesided API Utility Functions
 */

#include "charm++.h"
#include "ck.h"
#include "converse.h"
#include "cmirdmautils.h"


#if CMK_SMP
/*readonly*/ extern CProxy_ckcallback_group _ckcallbackgroup;
#endif

/*********************************** Zerocopy Direct API **********************************/

// Get Methods
void CkNcpyBuffer::memcpyGet(CkNcpyBuffer &source) {
  // memcpy the data from the source buffer into the destination buffer
  memcpy((void *)ptr, source.ptr, cnt);
}

#if CMK_USE_CMA
void CkNcpyBuffer::cmaGet(CkNcpyBuffer &source) {
  CmiIssueRgetUsingCMA(source.ptr,
         source.layerInfo,
         source.pe,
         ptr,
         layerInfo,
         pe,
         cnt);
}
#endif

void CkNcpyBuffer::rdmaGet(CkNcpyBuffer &source) {

  int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
  int ackSize = sizeof(CkCallback);

  if(mode == CK_BUFFER_UNREG) {
    // register it because it is required for RGET
    CmiSetRdmaBufferInfo(layerInfo + CmiGetRdmaCommonInfoSize(), ptr, cnt, mode);

    isRegistered = true;
  }

  // Create a general object that can be used across layers and can store the state of the CkNcpyBuffer objects
  int ncpyObjSize = getNcpyOpInfoTotalSize(
                      layerInfoSize,
                      ackSize,
                      layerInfoSize,
                      ackSize);

  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)CmiAlloc(ncpyObjSize);

  setNcpyOpInfo(source.ptr,
                (char *)(source.layerInfo),
                layerInfoSize,
                (char *)(&source.cb),
                ackSize,
                source.cnt,
                source.mode,
                source.isRegistered,
                source.pe,
                source.ref,
                ptr,
                (char *)(layerInfo),
                layerInfoSize,
                (char *)(&cb),
                ackSize,
                cnt,
                mode,
                isRegistered,
                pe,
                ref,
                ncpyOpInfo);

  CmiIssueRget(ncpyOpInfo);
}

// Perform a nocopy get operation into this destination using the passed source
CkNcpyStatus CkNcpyBuffer::get(CkNcpyBuffer &source){
  if(mode == CK_BUFFER_NOREG || source.mode == CK_BUFFER_NOREG) {
    CkAbort("Cannot perform RDMA operations in CK_BUFFER_NOREG mode\n");
  }

  // Check that the source buffer fits into the destination buffer
  CkAssert(source.cnt <= cnt);

  // Check that this object is local when get is called
  CkAssert(CkNodeOf(pe) == CkMyNode());

  CkNcpyMode transferMode = findTransferMode(source.pe, pe);

  //Check if it is a within-process sending
  if(transferMode == CkNcpyMode::MEMCPY) {
    memcpyGet(source);

    //Invoke the source callback
    source.cb.send(sizeof(CkNcpyBuffer), &source);

    //Invoke the destination callback
    cb.send(sizeof(CkNcpyBuffer), this);

    // rdma data transfer complete
    return CkNcpyStatus::complete;

#if CMK_USE_CMA
  } else if(transferMode == CkNcpyMode::CMA) {

    cmaGet(source);

    //Invoke the source callback
    source.cb.send(sizeof(CkNcpyBuffer), &source);

    //Invoke the destination callback
    cb.send(sizeof(CkNcpyBuffer), this);

    // rdma data transfer complete
    return CkNcpyStatus::complete;

#endif
  } else if (transferMode == CkNcpyMode::RDMA) {

    int outstandingRdmaOps = 1; // used by true-RDMA layers

#if CMK_ONESIDED_IMPL
#if CMK_CONVERSE_MPI
    outstandingRdmaOps += 1; // MPI layer invokes CmiDirectAckHandler twice as sender and receiver post separately
#endif
#else
    outstandingRdmaOps += 1; // non-RDMA layers invoke CmiDirectAckHandler twice using regular messages
#endif

    // Create QD to ensure that outstanding rdmaGet call is accounted for
    QdCreate(outstandingRdmaOps);

    rdmaGet(source);

    // rdma data transfer incomplete
    return CkNcpyStatus::incomplete;

  } else {
    CkAbort("Invalid CkNcpyMode");
  }
}

// Put Methods
void CkNcpyBuffer::memcpyPut(CkNcpyBuffer &destination) {
  // memcpy the data from the source buffer into the destination buffer
  memcpy((void *)destination.ptr, ptr, cnt);
}

#if CMK_USE_CMA
void CkNcpyBuffer::cmaPut(CkNcpyBuffer &destination) {
  CmiIssueRputUsingCMA(destination.ptr,
                       destination.layerInfo,
                       destination.pe,
                       ptr,
                       layerInfo,
                       pe,
                       cnt);
}
#endif

void CkNcpyBuffer::rdmaPut(CkNcpyBuffer &destination) {

  int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
  int ackSize = sizeof(CkCallback);

  if(mode == CK_BUFFER_UNREG) {
    // register it because it is required for RPUT
    CmiSetRdmaBufferInfo(layerInfo + CmiGetRdmaCommonInfoSize(), ptr, cnt, mode);

    isRegistered = true;
  }

  // Create a general object that can be used across layers that can store the state of the CkNcpyBuffer objects
  int ncpyObjSize = getNcpyOpInfoTotalSize(
                      layerInfoSize,
                      ackSize,
                      layerInfoSize,
                      ackSize);

  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)CmiAlloc(ncpyObjSize);

  setNcpyOpInfo(ptr,
                (char *)(layerInfo),
                layerInfoSize,
                (char *)(&cb),
                ackSize,
                cnt,
                mode,
                isRegistered,
                pe,
                ref,
                destination.ptr,
                (char *)(destination.layerInfo),
                layerInfoSize,
                (char *)(&destination.cb),
                ackSize,
                destination.cnt,
                destination.mode,
                destination.isRegistered,
                destination.pe,
                destination.ref,
                ncpyOpInfo);

  CmiIssueRput(ncpyOpInfo);
}

// Perform a nocopy put operation into the passed destination using this source
CkNcpyStatus CkNcpyBuffer::put(CkNcpyBuffer &destination){
  if(mode == CK_BUFFER_NOREG || destination.mode == CK_BUFFER_NOREG) {
    CkAbort("Cannot perform RDMA operations in CK_BUFFER_NOREG mode\n");
  }
  // Check that the source buffer fits into the destination buffer
  CkAssert(cnt <= destination.cnt);

  // Check that this object is local when put is called
  CkAssert(CkNodeOf(pe) == CkMyNode());

  CkNcpyMode transferMode = findTransferMode(pe, destination.pe);

  //Check if it is a within-process sending
  if(transferMode == CkNcpyMode::MEMCPY) {
    memcpyPut(destination);

    //Invoke the destination callback
    destination.cb.send(sizeof(CkNcpyBuffer), &destination);

    //Invoke the source callback
    cb.send(sizeof(CkNcpyBuffer), this);

    // rdma data transfer complete
    return CkNcpyStatus::complete;

#if CMK_USE_CMA
  } else if(transferMode == CkNcpyMode::CMA) {
    cmaPut(destination);

    //Invoke the destination callback
    destination.cb.send(sizeof(CkNcpyBuffer), &destination);

    //Invoke the source callback
    cb.send(sizeof(CkNcpyBuffer), this);

    // rdma data transfer complete
    return CkNcpyStatus::complete;

#endif
  } else if (transferMode == CkNcpyMode::RDMA) {

    int outstandingRdmaOps = 1; // used by true-RDMA layers

#if CMK_ONESIDED_IMPL
#if CMK_CONVERSE_MPI
    outstandingRdmaOps += 1; // MPI layer invokes CmiDirectAckHandler twice as sender and receiver post separately
#endif
#else
    outstandingRdmaOps += 1; // non-RDMA layers invoke CmiDirectAckHandler twice using regular messages
#endif

    // Create QD to ensure that outstanding rdmaGet call is accounted for
    QdCreate(outstandingRdmaOps);

    rdmaPut(destination);

    // rdma data transfer incomplete
    return CkNcpyStatus::incomplete;

  } else {
    CkAbort("Invalid CkNcpyMode");
  }
}

// reconstruct the CkNcpyBuffer object for the source
void constructSourceBufferObject(NcpyOperationInfo *info, CkNcpyBuffer &src) {
  src.ptr = info->srcPtr;
  src.pe  = info->srcPe;
  src.cnt = info->srcSize;
  src.ref = info->srcRef;
  src.mode = info->srcMode;
  src.isRegistered = info->isSrcRegistered;
  memcpy((char *)(&src.cb), info->srcAck, info->srcAckSize); // initialize cb
  memcpy((char *)(src.layerInfo), info->srcLayerInfo, info->srcLayerSize); // initialize layerInfo
}

// reconstruct the CkNcpyBuffer object for the destination
void constructDestinationBufferObject(NcpyOperationInfo *info, CkNcpyBuffer &dest) {
  dest.ptr = info->destPtr;
  dest.pe  = info->destPe;
  dest.cnt = info->destSize;
  dest.ref = info->destRef;
  dest.mode = info->destMode;
  dest.isRegistered = info->isDestRegistered;
  memcpy((char *)(&dest.cb), info->destAck, info->destAckSize); // initialize cb
  memcpy((char *)(dest.layerInfo), info->destLayerInfo, info->destLayerSize); // initialize layerInfo
}

void invokeSourceCallback(NcpyOperationInfo *info) {
  CkCallback *srcCb = (CkCallback *)(info->srcAck);
  if(srcCb->requiresMsgConstruction()) {
    if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_SRC_ACK) {
      CkNcpyBuffer src;
      constructSourceBufferObject(info, src);
      //Invoke the sender's callback
      invokeCallback(info->srcAck, info->srcPe, src);
    }
  }
}

void invokeDestinationCallback(NcpyOperationInfo *info) {
  CkCallback *destCb = (CkCallback *)(info->destAck);
  if(destCb->requiresMsgConstruction()) {
    if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_DEST_ACK) {
      CkNcpyBuffer dest;
      constructDestinationBufferObject(info, dest);
      //Invoke the receiver's callback
      invokeCallback(info->destAck, info->destPe, dest);
    }
  }
}

void handleDirectApiCompletion(NcpyOperationInfo *info) {
  invokeSourceCallback(info);
  invokeDestinationCallback(info);

  if(info->freeMe == CMK_FREE_NCPYOPINFO)
    CmiFree(info);
}

// Ack handler function which invokes the callback
void CkRdmaDirectAckHandler(void *ack) {

  // Process QD to mark completion of the outstanding RDMA operation
  QdProcess(1);

  NcpyOperationInfo *info = (NcpyOperationInfo *)(ack);

  CkCallback *srcCb = (CkCallback *)(info->srcAck);
  CkCallback *destCb = (CkCallback *)(info->destAck);

  switch(info->opMode) {
    case CMK_DIRECT_API    : handleDirectApiCompletion(info); // Ncpy Direct API
                             break;
#if CMK_ONESIDED_IMPL
    case CMK_EM_API        : handleEntryMethodApiCompletion(info); // Ncpy EM API invoked through a GET
                             break;

    case CMK_EM_API_REVERSE: handleReverseEntryMethodApiCompletion(info); // Ncpy EM API invoked through a PUT
                             break;
#endif
    default                : CmiAbort("Unknown opMode");
                             break;
  }
}

// Helper methods
void invokeCallback(void *cb, int pe, CkNcpyBuffer &buff) {

#if CMK_SMP
    //call to callbackgroup to call the callback when calling from comm thread
    //this adds one more trip through the scheduler
    _ckcallbackgroup[pe].call(*(CkCallback *)(cb), sizeof(CkNcpyBuffer), (const char *)(&buff));
#else
    //Invoke the destination callback
    ((CkCallback *)(cb))->send(sizeof(CkNcpyBuffer), &buff);
#endif
}

// Returns CkNcpyMode::MEMCPY if both the PEs are the same and memcpy can be used
// Returns CkNcpyMode::CMA if both the PEs are in the same physical node and CMA can be used
// Returns CkNcpyMode::RDMA if RDMA needs to be used
CkNcpyMode findTransferMode(int srcPe, int destPe) {
  if(CmiNodeOf(srcPe)==CmiNodeOf(destPe))
    return CkNcpyMode::MEMCPY;
#if CMK_USE_CMA
  else if(CmiDoesCMAWork() && CmiPeOnSamePhysicalNode(srcPe, destPe))
    return CkNcpyMode::CMA;
#endif
  else
    return CkNcpyMode::RDMA;
}






/*********************************** Zerocopy Entry Method API ****************************/
#if CMK_ONESIDED_IMPL
/*
 * Extract ncpy buffer information from the metadata message, allocate buffers
 * and issue ncpy calls (either memcpy or cma read or rdma get). Main method called on
 * the destination to perform zerocopy operations as a part of the Zerocopy Entry Method
 * API
 */
envelope* CkRdmaIssueRgets(envelope *env){
  // Iterate over the ncpy buffer and either perform the operations
  int numops, bufsize, msgsize;
  bufsize = getRdmaBufSize(env);
  numops = getRdmaNumOps(env);
  msgsize = env->getTotalsize();

  CkNcpyMode ncpyMode = findTransferMode(env->getSrcPe(), CkMyPe());

  int totalMsgSize = CK_ALIGN(msgsize, 16) + bufsize;
  char *ref;
  int layerInfoSize, ncpyObjSize, extraSize;

  if(ncpyMode == CkNcpyMode::RDMA) {

    layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;

    ncpyObjSize = getNcpyOpInfoTotalSize(
                  layerInfoSize,
                  sizeof(CkCallback),
                  layerInfoSize,
                  0);

    extraSize = ncpyObjSize - sizeof(NcpyOperationInfo);

    totalMsgSize += sizeof(NcpyEmInfo) + numops*(sizeof(NcpyEmBufferInfo) + extraSize);
  }

  // Allocate the new message which stores the receiver buffers
  envelope *copyenv = (envelope *)CmiAlloc(totalMsgSize);

  //Copy the metadata message(without the machine specific info) into the buffer
  memcpy(copyenv, env, msgsize);

  /* Set the total size of the message excluding the receiver's machine specific info
   * which is not required when the receiver's entry method executes
   */
  copyenv->setTotalsize(totalMsgSize);

  // Set rdma flag to be false to prevent message handler on the receiver
  // from intercepting it
  copyenv->setRdma(false);

  if(ncpyMode == CkNcpyMode::RDMA) {
    ref = (char *)copyenv + CK_ALIGN(msgsize, 16) + bufsize;

    NcpyEmInfo *ncpyEmInfo = (NcpyEmInfo *)ref;
    ncpyEmInfo->numOps = numops;
    ncpyEmInfo->counter = 0;
    ncpyEmInfo->msg = copyenv;
  }

  char *buf = (char *)copyenv + CK_ALIGN(msgsize, 16);

  CkUnpackMessage(&copyenv);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  up|numops;
  p|numops;

  for(int i=0; i<numops; i++){
    // source buffer
    CkNcpyBuffer source;
    up|source;

    // destination buffer
    CkNcpyBuffer dest((const void *)buf, CK_BUFFER_UNREG);
    dest.cnt = source.cnt;

    // Set the common layerInfo for the destination
    CmiSetRdmaCommonInfo(dest.layerInfo, dest.ptr, dest.cnt);

    if(ncpyMode == CkNcpyMode::MEMCPY) {

      dest.memcpyGet(source);

      // Invoke source callback
      source.cb.send(sizeof(CkNcpyBuffer), &source);

#if CMK_USE_CMA
    } else if(ncpyMode == CkNcpyMode::CMA) {

      dest.cmaGet(source);

      // Invoke source callback
      source.cb.send(sizeof(CkNcpyBuffer), &source);

#endif
    } else if(ncpyMode == CkNcpyMode::RDMA) {

      if(dest.mode == CK_BUFFER_UNREG) {
        // register it because it is required for RGET
        CmiSetRdmaBufferInfo(dest.layerInfo + CmiGetRdmaCommonInfoSize(), dest.ptr, dest.cnt, dest.mode);

        dest.isRegistered = true;
      }

      NcpyEmBufferInfo *ncpyEmBufferInfo = (NcpyEmBufferInfo *)(ref + sizeof(NcpyEmInfo) + i *(sizeof(NcpyEmBufferInfo) + extraSize));
      ncpyEmBufferInfo->index = i;

      NcpyOperationInfo *ncpyOpInfo = &(ncpyEmBufferInfo->ncpyOpInfo);

      setNcpyOpInfo(source.ptr,
                    (char *)(source.layerInfo),
                    layerInfoSize,
                    (char *)(&source.cb),
                    sizeof(CkCallback),
                    source.cnt,
                    source.mode,
                    source.isRegistered,
                    source.pe,
                    source.ref,
                    dest.ptr,
                    (char *)(dest.layerInfo),
                    layerInfoSize,
                    NULL,
                    0,
                    dest.cnt,
                    dest.mode,
                    dest.isRegistered,
                    dest.pe,
                    (char *)(ncpyEmBufferInfo), // destRef
                    ncpyOpInfo);

      // set opMode for entry method API
      ncpyOpInfo->opMode = CMK_EM_API;
      ncpyOpInfo->freeMe = CMK_DONT_FREE_NCPYOPINFO; // Since ncpyOpInfo is a part of the charm message, don't explicitly free it
                                                     // It'll be freed when the message is freed by the RTS after the execution of the entry method
      ncpyOpInfo->refPtr = ncpyEmBufferInfo;

      // Do no launch Rgets here as they could potentially cause a race condition in the SMP mode
      // The race condition is caused when an RGET completes and invokes the CkRdmaDirectAckHandler
      // on the comm. thread as the message is being inside this for loop on the worker thread

    } else {
      CkAbort("Invalid Mode");
    }

    //Update the CkRdmaWrapper pointer of the new message
    source.ptr = buf;

    //Update the pointer
    buf += CK_ALIGN(source.cnt, 16);
    p|source;
  }

  CkPackRdmaPtrs(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);

  if(ncpyMode == CkNcpyMode::MEMCPY || ncpyMode == CkNcpyMode::CMA ) {
    // All operations have completed
    CkPackMessage(&copyenv);
    return copyenv; // to indicate the completion of the gets
    // copyenv represents the new message which consists of the destination buffers

  } else{

    // Launch rgets
    for(int i=0; i<numops; i++){
      NcpyEmBufferInfo *ncpyEmBufferInfo = (NcpyEmBufferInfo *)(ref + sizeof(NcpyEmInfo) + i *(sizeof(NcpyEmBufferInfo) + extraSize));
      NcpyOperationInfo *ncpyOpInfo = &(ncpyEmBufferInfo->ncpyOpInfo);
      CmiIssueRget(ncpyOpInfo);
    }
    return NULL; // to indicate an async operation which may not be complete
  }
}

// Method called to unpack rdma pointers
void CkPackRdmaPtrs(char *msgBuf){
  PUP::toMem p((void *)msgBuf);
  PUP::fromMem up((void *)msgBuf);
  int numops;
  up|numops;
  p|numops;

  // Pack ncpy pointers in CkNcpyBuffer
  for(int i=0; i<numops; i++){
    CkNcpyBuffer w;
    up|w;
    w.ptr = (void *)((char *)w.ptr - (char *)msgBuf);
    p|w;
  }
}

// Method called to unpack rdma pointers
void CkUnpackRdmaPtrs(char *msgBuf){
  PUP::toMem p((void *)msgBuf);
  PUP::fromMem up((void *)msgBuf);
  int numops;
  up|numops;
  p|numops;

  // Unpack ncpy pointers in CkNcpyBuffer
  for(int i=0; i<numops; i++){
    CkNcpyBuffer w;
    up|w;
    w.ptr = (void *)((char *)msgBuf + (size_t)w.ptr);
    p|w;
  }
}

//Determine the number of ncpy ops from the message
int getRdmaNumOps(envelope *env){
  int numops;
  CkUnpackMessage(&env);
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  CkPackMessage(&env);
  return numops;
}

// Get the sum of ncpy buffer sizes using the metadata message
int getRdmaBufSize(envelope *env){
  /*
   * Determine the number of ncpy operations and the sum of all
   * ncpy buffer sizes by iterating over the CkNcpyBuffers in the message
   */
  int numops, bufsize=0;
  CkUnpackMessage(&env);
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  for(int i=0; i<numops; i++){
    CkNcpyBuffer w; up|w;
    bufsize += CK_ALIGN(w.cnt, 16);
  }
  CkPackMessage(&env);
  return bufsize;
}

void handleEntryMethodApiCompletion(NcpyOperationInfo *info) {
  invokeSourceCallback(info);
  if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_DEST_ACK) {
    // Invoke the ackhandler function to update the counter
    CkRdmaEMAckHandler(info->destPe, info->refPtr);
  }
}

void handleReverseEntryMethodApiCompletion(NcpyOperationInfo *info) {
  invokeSourceCallback(info);
  if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_DEST_ACK) {
    // Send a message to the receiver to invoke the ackhandler function to update the counter
    CmiInvokeRemoteAckHandler(info->destPe, info->refPtr);
  }
  if(info->freeMe == CMK_FREE_NCPYOPINFO)
    CmiFree(info);
}

// Ack handler function called when a Zerocopy Entry Method buffer completes
void CkRdmaEMAckHandler(int destPe, void *ack) {

  NcpyEmBufferInfo *emBuffInfo = (NcpyEmBufferInfo *)(ack);

  char *ref = (char *)(emBuffInfo);

  int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
  int ncpyObjSize = getNcpyOpInfoTotalSize(
                    layerInfoSize,
                    sizeof(CkCallback),
                    layerInfoSize,
                    0);


  NcpyEmInfo *ncpyEmInfo = (NcpyEmInfo *)(ref - (emBuffInfo->index) * (sizeof(NcpyEmBufferInfo) + ncpyObjSize - sizeof(NcpyOperationInfo)) - sizeof(NcpyEmInfo));
  ncpyEmInfo->counter++; // A zerocopy get completed, update the counter

#if CMK_REG_REQUIRED
  NcpyOperationInfo *ncpyOpInfo = &(emBuffInfo->ncpyOpInfo);
  // De-register the destination buffer
  CmiDeregisterMem(ncpyOpInfo->destPtr, ncpyOpInfo->destLayerInfo + CmiGetRdmaCommonInfoSize(), ncpyOpInfo->destPe, ncpyOpInfo->destMode);
#endif

  // Check if all rdma operations are complete
  if(ncpyEmInfo->counter == ncpyEmInfo->numOps) {

    // invoke the charm message handler to enqueue the messsage
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRES
    // invoked from the comm thread, so send message to the worker thread
    CmiPushPE(CmiRankOf(destPe), ncpyEmInfo->msg);
#else
    // invoked from the worker thread, process message
    CmiHandleMessage(ncpyEmInfo->msg);
#endif
  }
}
#endif
/* End of CMK_ONESIDED_IMPL */
