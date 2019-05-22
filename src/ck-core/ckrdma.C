/*
 * Charm Onesided API Utility Functions
 */

#include "charm++.h"
#include "ck.h"
#include "converse.h"
#include "cmirdmautils.h"
#include <algorithm>

#if CMK_SMP
/*readonly*/ extern CProxy_ckcallback_group _ckcallbackgroup;
#endif

/*********************************** Zerocopy Direct API **********************************/
// Get Methods
void CkNcpyBuffer::memcpyGet(CkNcpyBuffer &source) {
  // memcpy the data from the source buffer into the destination buffer
  memcpy((void *)ptr, source.ptr, std::min(cnt, source.cnt));
}

#if CMK_USE_CMA
void CkNcpyBuffer::cmaGet(CkNcpyBuffer &source) {
  CmiIssueRgetUsingCMA(source.ptr,
         source.layerInfo,
         source.pe,
         ptr,
         layerInfo,
         pe,
         std::min(cnt, source.cnt));
}
#endif

void CkNcpyBuffer::rdmaGet(CkNcpyBuffer &source) {

  int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
  int ackSize = sizeof(CkCallback);

  if(regMode == CK_BUFFER_UNREG) {
    // register it because it is required for RGET
    CmiSetRdmaBufferInfo(layerInfo + CmiGetRdmaCommonInfoSize(), ptr, cnt, regMode);

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
                source.regMode,
                source.deregMode,
                source.isRegistered,
                source.pe,
                source.ref,
                ptr,
                (char *)(layerInfo),
                layerInfoSize,
                (char *)(&cb),
                ackSize,
                cnt,
                regMode,
                deregMode,
                isRegistered,
                pe,
                ref,
                ncpyOpInfo);

  CmiIssueRget(ncpyOpInfo);
}

// Perform a nocopy get operation into this destination using the passed source
CkNcpyStatus CkNcpyBuffer::get(CkNcpyBuffer &source){
  if(regMode == CK_BUFFER_NOREG || source.regMode == CK_BUFFER_NOREG) {
    CkAbort("Cannot perform RDMA operations in CK_BUFFER_NOREG mode\n");
  }

  // Check that the source buffer fits into the destination buffer
  if(cnt < source.cnt)
    CkAbort("CkNcpyBuffer::get (destination.cnt < source.cnt) Destination buffer is smaller than the source buffer\n");

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
    CkAbort("CkNcpyBuffer::get : Invalid CkNcpyMode");
  }
}

// Put Methods
void CkNcpyBuffer::memcpyPut(CkNcpyBuffer &destination) {
  // memcpy the data from the source buffer into the destination buffer
  memcpy((void *)destination.ptr, ptr, std::min(cnt, destination.cnt));
}

#if CMK_USE_CMA
void CkNcpyBuffer::cmaPut(CkNcpyBuffer &destination) {
  CmiIssueRputUsingCMA(destination.ptr,
                       destination.layerInfo,
                       destination.pe,
                       ptr,
                       layerInfo,
                       pe,
                       std::min(cnt, destination.cnt));
}
#endif

void CkNcpyBuffer::rdmaPut(CkNcpyBuffer &destination) {

  int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
  int ackSize = sizeof(CkCallback);

  if(regMode == CK_BUFFER_UNREG) {
    // register it because it is required for RPUT
    CmiSetRdmaBufferInfo(layerInfo + CmiGetRdmaCommonInfoSize(), ptr, cnt, regMode);

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
                regMode,
                deregMode,
                isRegistered,
                pe,
                ref,
                destination.ptr,
                (char *)(destination.layerInfo),
                layerInfoSize,
                (char *)(&destination.cb),
                ackSize,
                destination.cnt,
                destination.regMode,
                destination.deregMode,
                destination.isRegistered,
                destination.pe,
                destination.ref,
                ncpyOpInfo);

  CmiIssueRput(ncpyOpInfo);
}

// Perform a nocopy put operation into the passed destination using this source
CkNcpyStatus CkNcpyBuffer::put(CkNcpyBuffer &destination){
  if(regMode == CK_BUFFER_NOREG || destination.regMode == CK_BUFFER_NOREG) {
    CkAbort("Cannot perform RDMA operations in CK_BUFFER_NOREG mode\n");
  }
  // Check that the source buffer fits into the destination buffer
  if(destination.cnt < cnt)
    CkAbort("CkNcpyBuffer::put (destination.cnt < source.cnt) Destination buffer is smaller than the source buffer\n");

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
    CkAbort("CkNcpyBuffer::put : Invalid CkNcpyMode");
  }
}

// reconstruct the CkNcpyBuffer object for the source
void constructSourceBufferObject(NcpyOperationInfo *info, CkNcpyBuffer &src) {
  src.ptr = info->srcPtr;
  src.pe  = info->srcPe;
  src.cnt = info->srcSize;
  src.ref = info->srcRef;
  src.regMode = info->srcRegMode;
  src.deregMode = info->srcDeregMode;
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
  dest.regMode = info->destRegMode;
  dest.deregMode = info->destDeregMode;
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
    case CMK_DIRECT_API           : handleDirectApiCompletion(info); // Ncpy Direct API
                                    break;
#if CMK_ONESIDED_IMPL
    case CMK_EM_API               : handleEntryMethodApiCompletion(info); // Ncpy EM API invoked through a GET
                                    break;

    case CMK_EM_API_SRC_ACK_INVOKE: invokeSourceCallback(info);
                                    break;

    case CMK_EM_API_REVERSE       : handleReverseEntryMethodApiCompletion(info); // Ncpy EM API invoked through a PUT
                                    break;

    case CMK_BCAST_EM_API         : handleBcastEntryMethodApiCompletion(info); // Ncpy EM Bcast API
                                    break;

    case CMK_BCAST_EM_API_REVERSE : handleBcastReverseEntryMethodApiCompletion(info); // Ncpy EM Bcast API invoked through a PUT
                                    break;
    case CMK_READONLY_BCAST       : readonlyGetCompleted(info);
                                    break;
#endif
    default                       : CkAbort("CkRdmaDirectAckHandler: Unknown ncpyOpInfo->opMode");
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

void enqueueNcpyMessage(int destPe, void *msg){
  // invoke the charm message handler to enqueue the messsage
#if CMK_SMP && !CMK_ENABLE_ASYNC_PROGRES
  if(destPe == CkMyPe()) // invoked from the same worker thread, call message handler directly
    CmiHandleMessage(msg);
  else                   // invoked from the comm thread, so send message to the worker thread
    CmiPushPE(CmiRankOf(destPe), msg);
#else
  // invoked from the same logical node (process), call message handler directly
  // or invoked from the same worker thread, call message handler directly
  CmiHandleMessage(msg);
#endif
}


#if CMK_ONESIDED_IMPL
/*********************************** Zerocopy Entry Method API ****************************/


/************************* Zerocopy Entry Method API - Utility functions ******************/

void performRgets(char *ref, int numops, int extraSize) {
  // Launch rgets
  for(int i=0; i<numops; i++){
    NcpyEmBufferInfo *ncpyEmBufferInfo = (NcpyEmBufferInfo *)(ref + sizeof(NcpyEmInfo) + i *(sizeof(NcpyEmBufferInfo) + extraSize));
    NcpyOperationInfo *ncpyOpInfo = &(ncpyEmBufferInfo->ncpyOpInfo);
    CmiIssueRget(ncpyOpInfo);
  }
}

// Method called on completion of an Zcpy EM API (Send or Recv, P2P or BCAST)
void CkRdmaEMAckHandler(int destPe, void *ack) {

  if(_topoTree == NULL) CkAbort("CkRdmaIssueRgets:: topo tree has not been calculated \n");
  CmiSpanningTreeInfo &t = *_topoTree;

  NcpyEmBufferInfo *emBuffInfo = (NcpyEmBufferInfo *)(ack);

  char *ref = (char *)(emBuffInfo);

  int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
  int ncpyObjSize = getNcpyOpInfoTotalSize(
                    layerInfoSize,
                    sizeof(CkCallback),
                    layerInfoSize,
                    0);


  NcpyEmInfo *ncpyEmInfo = (NcpyEmInfo *)(ref - (emBuffInfo->index) * (sizeof(NcpyEmBufferInfo) + ncpyObjSize - sizeof(NcpyOperationInfo)) - sizeof(NcpyEmInfo));
  ncpyEmInfo->counter++; // Operation completed, update counter

#if CMK_REG_REQUIRED
  if(ncpyEmInfo->mode == ncpyEmApiMode::P2P_SEND ||
     (ncpyEmInfo->mode == ncpyEmApiMode::BCAST_SEND && t.child_count == 0)) {  // EM P2P Send API or EM BCAST Send API

    NcpyOperationInfo *ncpyOpInfo = &(emBuffInfo->ncpyOpInfo);

    // De-register the destination buffer
    CmiDeregisterMem(ncpyOpInfo->destPtr, ncpyOpInfo->destLayerInfo + CmiGetRdmaCommonInfoSize(), ncpyOpInfo->destPe, ncpyOpInfo->destRegMode);

  } else if(ncpyEmInfo->mode == ncpyEmApiMode::P2P_RECV ||
           (ncpyEmInfo->mode == ncpyEmApiMode::BCAST_RECV && t.child_count == 0)) {  // EM P2P Post API or EM BCAST Post API
    NcpyOperationInfo *ncpyOpInfo = &(emBuffInfo->ncpyOpInfo);

    // De-register only if destDeregMode is CK_BUFFER_DEREG
    if(ncpyOpInfo->destDeregMode == CK_BUFFER_DEREG) {
      CmiDeregisterMem(ncpyOpInfo->destPtr, ncpyOpInfo->destLayerInfo + CmiGetRdmaCommonInfoSize(), ncpyOpInfo->destPe, ncpyOpInfo->destRegMode);
    }
  }
#endif

  if(ncpyEmInfo->counter == ncpyEmInfo->numOps) {
    // All operations have been completed

    switch(ncpyEmInfo->mode) {
      case ncpyEmApiMode::P2P_SEND    : enqueueNcpyMessage(destPe, ncpyEmInfo->msg);
                                        break;

      case ncpyEmApiMode::P2P_RECV    : enqueueNcpyMessage(destPe, ncpyEmInfo->msg);
                                        CmiFree(ncpyEmInfo);
                                        break;

      case ncpyEmApiMode::BCAST_SEND  : processBcastSendEmApiCompletion(ncpyEmInfo, destPe);
                                        break;

      case ncpyEmApiMode::BCAST_RECV  : processBcastRecvEmApiCompletion(ncpyEmInfo, destPe);
                                        break;

      default                         : CmiAbort("Invalid operation mode");
                                        break;
    }
  }
}

void performEmApiMemcpy(CkNcpyBuffer &source, CkNcpyBuffer &dest, ncpyEmApiMode emMode) {
  dest.memcpyGet(source);

  if(emMode == ncpyEmApiMode::P2P_SEND || emMode == ncpyEmApiMode::P2P_RECV) {
    // Invoke source callback
    source.cb.send(sizeof(CkNcpyBuffer), &source);
  } // send a message to the parent to indicate completion
  else if (emMode == ncpyEmApiMode::BCAST_SEND || emMode == ncpyEmApiMode::BCAST_RECV) {
    // Invoke the bcast handler
    CkRdmaEMBcastAckHandler((void *)source.bcastAckInfo);
  }
}

#if CMK_USE_CMA
void performEmApiCmaTransfer(CkNcpyBuffer &source, CkNcpyBuffer &dest, int child_count, ncpyEmApiMode emMode) {
  dest.cmaGet(source);

  if(emMode == ncpyEmApiMode::P2P_SEND || emMode == ncpyEmApiMode::P2P_RECV) {
    // Invoke source callback
    source.cb.send(sizeof(CkNcpyBuffer), &source);
  }
  else if (emMode == ncpyEmApiMode::BCAST_SEND || emMode == ncpyEmApiMode::BCAST_RECV) {
    if(child_count != 0) {
      if(dest.regMode == CK_BUFFER_UNREG) {
        // register it because it is required for RGET performed by child nodes
        CmiSetRdmaBufferInfo(dest.layerInfo + CmiGetRdmaCommonInfoSize(), dest.ptr, dest.cnt, dest.regMode);
        dest.isRegistered = true;
      }
    }
  }
}
#endif

void performEmApiRget(CkNcpyBuffer &source, CkNcpyBuffer &dest, int opIndex, char *ref, int extraSize, ncpyEmApiMode emMode) {

  int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
  if(dest.regMode == CK_BUFFER_UNREG) {
    // register it because it is required for RGET
    CmiSetRdmaBufferInfo(dest.layerInfo + CmiGetRdmaCommonInfoSize(), dest.ptr, dest.cnt, dest.regMode);

    dest.isRegistered = true;
  }

  NcpyEmBufferInfo *ncpyEmBufferInfo = (NcpyEmBufferInfo *)(ref + sizeof(NcpyEmInfo) + opIndex *(sizeof(NcpyEmBufferInfo) + extraSize));
  ncpyEmBufferInfo->index = opIndex;

  NcpyOperationInfo *ncpyOpInfo = &(ncpyEmBufferInfo->ncpyOpInfo);
  setNcpyOpInfo(source.ptr,
                (char *)(source.layerInfo),
                layerInfoSize,
                (char *)(&source.cb),
                sizeof(CkCallback),
                source.cnt,
                source.regMode,
                source.deregMode,
                source.isRegistered,
                source.pe,
                source.ref,
                dest.ptr,
                (char *)(dest.layerInfo),
                layerInfoSize,
                NULL,
                0,
                dest.cnt,
                dest.regMode,
                dest.deregMode,
                dest.isRegistered,
                dest.pe,
                (char *)(ncpyEmBufferInfo), // destRef
                ncpyOpInfo);

  // set opMode
  if(emMode == ncpyEmApiMode::BCAST_SEND || emMode == ncpyEmApiMode::BCAST_RECV)
    ncpyOpInfo->opMode = CMK_BCAST_EM_API;  // mode for bcast
  else if(emMode == ncpyEmApiMode::P2P_SEND || emMode == ncpyEmApiMode::P2P_RECV)
    ncpyOpInfo->opMode = CMK_EM_API;  // mode for p2p
  else
    CmiAbort("Invalid Mode\n");

  ncpyOpInfo->freeMe = CMK_DONT_FREE_NCPYOPINFO; // Since ncpyOpInfo is a part of the charm message, don't explicitly free it
                                                 // It'll be freed when the message is freed by the RTS after the execution of the entry method
  ncpyOpInfo->refPtr = ncpyEmBufferInfo;

  // Do no launch Rgets here as they could potentially cause a race condition in the SMP mode
  // The race condition is caused when an RGET completes and invokes the CkRdmaDirectAckHandler
  // on the comm. thread as the message is being inside this for loop on the worker thread
}

void performEmApiNcpyTransfer(CkNcpyBuffer &source, CkNcpyBuffer &dest, int opIndex, int child_count, char *ref, int extraSize, CkNcpyMode ncpyMode, ncpyEmApiMode emMode){

  switch(ncpyMode) {
    case CkNcpyMode::MEMCPY: performEmApiMemcpy(source, dest, emMode);
                                   break;
#if CMK_USE_CMA
    case CkNcpyMode::CMA   : performEmApiCmaTransfer(source, dest, child_count, emMode);
                                   break;
#endif
    case CkNcpyMode::RDMA  : performEmApiRget(source, dest, opIndex, ref, extraSize, emMode);
                                   break;

    default                      : CkAbort("Invalid Mode");
                                   break;
  }
}


void preprocessRdmaCaseForRgets(int &layerInfoSize, int &ncpyObjSize, int &extraSize, int &totalMsgSize, int &numops) {
    layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;

    ncpyObjSize = getNcpyOpInfoTotalSize(
                  layerInfoSize,
                  sizeof(CkCallback),
                  layerInfoSize,
                  0);

    extraSize = ncpyObjSize - sizeof(NcpyOperationInfo);

    totalMsgSize += sizeof(NcpyEmInfo) + numops*(sizeof(NcpyEmBufferInfo) + extraSize);
}

void setNcpyEmInfo(char *ref, envelope *env, int &msgsize, int &numops, void *forwardMsg, ncpyEmApiMode emMode) {

    NcpyEmInfo *ncpyEmInfo = (NcpyEmInfo *)ref;
    ncpyEmInfo->numOps = numops;
    ncpyEmInfo->counter = 0;
    ncpyEmInfo->msg = env;

    ncpyEmInfo->forwardMsg = forwardMsg; // useful only for BCAST, NULL for P2P
    ncpyEmInfo->pe = CkMyPe();
    ncpyEmInfo->mode = emMode; // P2P or BCAST
}

/* Zerocopy Entry Method API Functions */
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


// Determine the number of ncpy ops and the sum of the ncpy buffer sizes
// from the metadata message
void getRdmaNumopsAndBufsize(envelope *env, int &numops, int &bufsize) {
  numops = 0;
  bufsize = 0;
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  for(int i=0; i<numops; i++){
    CkNcpyBuffer w;
    up|w;
    bufsize += CK_ALIGN(w.cnt, 16);
  }
}

void handleEntryMethodApiCompletion(NcpyOperationInfo *info) {

#if CMK_REG_REQUIRED
  // send a message to the source to de-register and invoke callback
  if(info->srcDeregMode == CK_BUFFER_DEREG)
    CmiInvokeRemoteDeregAckHandler(info->srcPe, info);
  else // Do not de-register source when srcDeregMode != CK_BUFFER_DEREG
#endif
    invokeSourceCallback(info);

  if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_DEST_ACK) {
    // Invoke the ackhandler function to update the counter
    CkRdmaEMAckHandler(info->destPe, info->refPtr);
  }
}

void handleReverseEntryMethodApiCompletion(NcpyOperationInfo *info) {

  if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_DEST_ACK) {
    // Send a message to the receiver to invoke the ackhandler function to update the counter
    CmiInvokeRemoteAckHandler(info->destPe, info->refPtr);
  }

#if CMK_REG_REQUIRED
  // De-register source only when srcDeregMode == CK_BUFFER_DEREG
  if(info->srcDeregMode == CK_BUFFER_DEREG) {
    CmiDeregisterMem(info->srcPtr, info->srcLayerInfo + CmiGetRdmaCommonInfoSize(), info->srcPe, info->srcRegMode);
    info->isSrcRegistered = 0; // Set isSrcRegistered to 0 after de-registration
  }
#endif

  invokeSourceCallback(info);

  if(info->freeMe == CMK_FREE_NCPYOPINFO)
    CmiFree(info);
}

/******************************* Zerocopy P2P EM SEND API Functions ***********************/

/*
 * Extract ncpy buffer information from the metadata message, used passed buffers
 * and issue ncpy calls (either memcpy or cma read or rdma get). Main method called on
 * the destination to perform zerocopy operations as a part of the Zerocopy Entry Method
 * API
 */
envelope* CkRdmaIssueRgets(envelope *env, ncpyEmApiMode emMode, void *forwardMsg){

  int numops=0, bufsize=0, msgsize=0;

  CkUnpackMessage(&env); // Unpack message to access msgBuf inside getRdmaNumopsAndBufsize
  getRdmaNumopsAndBufsize(env, numops, bufsize);
  CkPackMessage(&env); // Pack message to ensure corret copying into copyenv

  msgsize = env->getTotalsize();
  int totalMsgSize = CK_ALIGN(msgsize, 16) + bufsize;
  char *ref;
  int layerInfoSize, ncpyObjSize, extraSize;

  CkNcpyMode ncpyMode = findTransferMode(env->getSrcPe(), CkMyPe());
  if(_topoTree == NULL) CkAbort("CkRdmaIssueRgets:: topo tree has not been calculated \n");
  CmiSpanningTreeInfo &t = *_topoTree;

  layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;

  if(ncpyMode == CkNcpyMode::RDMA) {
    preprocessRdmaCaseForRgets(layerInfoSize, ncpyObjSize, extraSize, totalMsgSize, numops);
  }

  // Allocate the new message which stores the receiver buffers
  envelope *copyenv = (envelope *)CmiAlloc(totalMsgSize);

  //Copy the metadata message(without the machine specific info) into the buffer
  memcpy(copyenv, env, msgsize);

  /* Set the total size of the message excluding the receiver's machine specific info
   * which is not required when the receiver's entry method executes
   */
  copyenv->setTotalsize(totalMsgSize);

  // Make the message a regular message to prevent message handler on the receiver
  // from intercepting it
  CMI_ZC_MSGTYPE(copyenv) = CMK_REG_NO_ZC_MSG;

  if(ncpyMode == CkNcpyMode::RDMA) {
    ref = (char *)copyenv + CK_ALIGN(msgsize, 16) + bufsize;
    setNcpyEmInfo(ref, copyenv, msgsize, numops, forwardMsg, emMode);
  }

  char *buf = (char *)copyenv + CK_ALIGN(msgsize, 16);

  CkUnpackMessage(&copyenv);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  up|numops;
  p|numops;

  // source buffer
  CkNcpyBuffer source;

  for(int i=0; i<numops; i++){
    up|source;

    // destination buffer
    CkNcpyBuffer dest((const void *)buf, source.cnt, CK_BUFFER_UNREG);

    performEmApiNcpyTransfer(source, dest, i, t.child_count, ref, extraSize, ncpyMode, emMode);

    //Update the CkRdmaWrapper pointer of the new message
    source.ptr = buf;

    memcpy(source.layerInfo, dest.layerInfo, layerInfoSize);

    //Update the pointer
    buf += CK_ALIGN(source.cnt, 16);
    p|source;
  }

  // Substitute buffer pointers by their offsets from msgBuf to handle migration
  CkPackRdmaPtrs(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);

  if(emMode == ncpyEmApiMode::P2P_SEND) {
    switch(ncpyMode) {
      case CkNcpyMode::MEMCPY:
      case CkNcpyMode::CMA   :  return copyenv;
                                break;

      case CkNcpyMode::RDMA  :  performRgets(ref, numops, extraSize);
                                break;

      default                :  CmiAbort("Invalid transfer mode\n");
                                break;
    }
  } else if(emMode == ncpyEmApiMode::BCAST_SEND) {
    switch(ncpyMode) {
      case CkNcpyMode::MEMCPY:  CkPackMessage(&copyenv); // Pack message as it will be forwarded to peers
                                forwardMessageToPeerNodes(copyenv, copyenv->getMsgtype());
                                return copyenv;
                                break;

      case CkNcpyMode::CMA   :  CkPackMessage(&copyenv);
                                handleMsgUsingCMAPostCompletionForSendBcast(copyenv, env, source);
                                break;

      case CkNcpyMode::RDMA  :  performRgets(ref, numops, extraSize);
                                break;

      default                :  CmiAbort("Invalid transfer mode\n");
                                break;
    }
  } else {
    CmiAbort("Invalid operation mode\n");
  }
  return NULL;
}


/******************************* Zerocopy P2P EM RECV API Functions ***********************/
/*
 * Extract ncpy buffer information from the metadata message
 * and issue ncpy calls (either memcpy or cma read or rdma get). Main method called on
 * the destination to perform zerocopy operations as a part of the Zerocopy Entry Method
 * API
 */
void CkRdmaIssueRgets(envelope *env, ncpyEmApiMode emMode, void *forwardMsg, int numops, void **arrPtrs, int *arrSizes, CkNcpyBufferPost *postStructs){

  if(emMode == ncpyEmApiMode::BCAST_SEND)
    CkAbort("CkRdmaIssueRgets:: topo tree has not been calculated \n");

  // Iterate over the ncpy buffer and either perform the operations
  int msgsize = env->getTotalsize();

  int refSize = 0;
  char *ref;
  int layerInfoSize, ncpyObjSize, extraSize;

  CkNcpyMode ncpyMode = findTransferMode(env->getSrcPe(), CkMyPe());
  if(_topoTree == NULL) CkAbort("CkRdmaIssueRgets:: topo tree has not been calculated \n");
  CmiSpanningTreeInfo &t = *_topoTree;

  layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;

  if(ncpyMode == CkNcpyMode::RDMA) {
    preprocessRdmaCaseForRgets(layerInfoSize, ncpyObjSize, extraSize, refSize, numops);
    ref = (char *)CmiAlloc(refSize);
  }

  // Make the message a regular message to prevent message handler on the receiver
  // from intercepting it
  if(emMode == ncpyEmApiMode::P2P_RECV)
    CMI_ZC_MSGTYPE(env) = CMK_ZC_P2P_RECV_DONE_MSG;

  if(ncpyMode == CkNcpyMode::RDMA) {
    setNcpyEmInfo(ref, env, msgsize, numops, forwardMsg, emMode);
  }

  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  p|numops;

  // source buffer
  CkNcpyBuffer source;

  for(int i=0; i<numops; i++){
    up|source;

    if(source.cnt < arrSizes[i])
      CkAbort("CkRdmaIssueRgets: Size of the posted buffer > Size of the source buffer\n");

    // destination buffer
    CkNcpyBuffer dest((const void *)arrPtrs[i], arrSizes[i], postStructs[i].regMode, postStructs[i].deregMode);

    performEmApiNcpyTransfer(source, dest, i, t.child_count, ref, extraSize, ncpyMode, emMode);

    //Update the CkRdmaWrapper pointer of the new message
    source.ptr = arrPtrs[i];

    memcpy(source.layerInfo, dest.layerInfo, layerInfoSize);

    p|source;
  }


  if(emMode == ncpyEmApiMode::P2P_RECV) {
    switch(ncpyMode) {
      case CkNcpyMode::MEMCPY:
      case CkNcpyMode::CMA   :  enqueueNcpyMessage(CkMyPe(), env);
                                break;

      case CkNcpyMode::RDMA  :  performRgets(ref, numops, extraSize);
                                break;

      default                :  CmiAbort("Invalid transfer mode\n");
                                break;
    }
  } else if(emMode == ncpyEmApiMode::BCAST_RECV) {
    switch(ncpyMode) {
      case CkNcpyMode::MEMCPY:  handleMsgOnChildPostCompletionForRecvBcast(env);
                                break;

      case CkNcpyMode::CMA   :  if(t.child_count == 0) {
                                  sendAckMsgToParent(env);
                                  handleMsgOnChildPostCompletionForRecvBcast(env);
                                } else {
                                  // Allocate a structure NcpyBcastInterimAckInfo to maintain state for ack handling
                                  NcpyBcastInterimAckInfo *bcastAckInfo = allocateInterimNodeAckObj(env, NULL, CkMyPe());
                                  handleMsgOnInterimPostCompletionForRecvBcast(env, bcastAckInfo, CkMyPe());
                                }
                                break;

      case CkNcpyMode::RDMA  :  performRgets(ref, numops, extraSize);
                                break;

      default                :  CmiAbort("Invalid transfer mode\n");
                                break;
    }
  } else {
    CmiAbort("Invalid operation mode\n");
  }
}

/***************************** Zerocopy Bcast Entry Method API ****************************/

/********************** Zerocopy Bcast Entry Method API - Utility functions ***************/

// Method called on the bcast source to store some information for ack handling
void CkRdmaPrepareBcastMsg(envelope *env) {

  int numops;
  CkUnpackMessage(&env);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);

  up|numops;
  p|numops;

  NcpyBcastRootAckInfo *bcastAckInfo = (NcpyBcastRootAckInfo *)CmiAlloc(sizeof(NcpyBcastRootAckInfo) + numops * sizeof(CkNcpyBuffer));

  CmiSpanningTreeInfo &t = *_topoTree;
  bcastAckInfo->numChildren = t.child_count + 1;
  bcastAckInfo->setCounter(0);
  bcastAckInfo->isRoot  = true;
  bcastAckInfo->numops  = numops;
  bcastAckInfo->pe = CkMyPe();

  for(int i=0; i<numops; i++) {
    CkNcpyBuffer source;
    up|source;

    bcastAckInfo->src[i] = source;

    source.bcastAckInfo = bcastAckInfo;

    p|source;
  }
  CkPackMessage(&env);
}

// Method called to extract the parent bcastAckInfo from the received message for ack handling
// Requires message to be unpacked
const void *getParentBcastAckInfo(void *msg, int &srcPe) {
  int numops;
  CkNcpyBuffer source;
  envelope *env = (envelope *)msg;
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);

  up|numops;
  p|numops;

  CkAssert(numops >= 1);

  up|source;
  p|source;

  srcPe = source.pe;
  return source.bcastAckInfo;
}

// Called only on intermediate nodes
// Allocate a NcpyBcastInterimAckInfo and return it
NcpyBcastInterimAckInfo *allocateInterimNodeAckObj(envelope *myEnv, envelope *myChildEnv, int pe) {
  CmiSpanningTreeInfo &t = *_topoTree;

  // Allocate a NcpyBcastInterimAckInfo object
  NcpyBcastInterimAckInfo *bcastAckInfo = (NcpyBcastInterimAckInfo *)CmiAlloc(sizeof(NcpyBcastInterimAckInfo));

  // Initialize fields of bcastAckInfo
  bcastAckInfo->numChildren = t.child_count;
  bcastAckInfo->counter = 0;
  bcastAckInfo->isRoot = false;
  bcastAckInfo->pe = pe;

  // Recv Bcast API uses myEnv as myChildEnv (and myChildEnv is NULL)
  bcastAckInfo->isRecv = (myChildEnv == NULL);
  bcastAckInfo->isArray = (myEnv->getMsgtype() == ForArrayEltMsg);

  // initialize derived calss NcpyBcastInterimAckInfo fields
  bcastAckInfo->msg = myEnv; // this message will be enqueued after the completion of all operations

  return bcastAckInfo;
}

// Method called on the root node and other intermediate parent nodes on completion of RGET through ZC Bcast
void CkRdmaEMBcastAckHandler(void *ack) {
  NcpyBcastAckInfo *bcastAckInfo = (NcpyBcastAckInfo *)ack;

  // Increment counter to indicate that another child was completed
  // Since incCounter() is equivalent to counter++, it returns the value of 'counter' before incrementing it by 1.
  // For that reason, the comparison is performed with bcastAckInfo->incCounter() + 1 to compare with the updated value
  if(bcastAckInfo->incCounter() + 1 == bcastAckInfo->numChildren) {
    // All child nodes have completed RGETs

    // TODO: replace with a swtich with 3 cases
    if(bcastAckInfo->isRoot) {

      NcpyBcastRootAckInfo *bcastRootAckInfo = (NcpyBcastRootAckInfo *)(bcastAckInfo);
      // invoke the callback with the pointer
      for(int i=0; i<bcastRootAckInfo->numops; i++) {
#if CMK_REG_REQUIRED
        // Deregister source buffer respecting source buffer's dereg mode
        if(bcastRootAckInfo->src[i].deregMode == CK_BUFFER_DEREG)
          bcastRootAckInfo->src[i].deregisterMem();
#endif

        invokeCallback(&(bcastRootAckInfo->src[i].cb),
                       bcastRootAckInfo->pe,
                       bcastRootAckInfo->src[i]);
      }

      CmiFree(bcastRootAckInfo);

    } else {
      CmiSpanningTreeInfo &t = *_topoTree;

      NcpyBcastInterimAckInfo *bcastInterimAckInfo = (NcpyBcastInterimAckInfo *)(bcastAckInfo);

      if(bcastInterimAckInfo->isRecv)  { // bcast post api
        // This node should send a message to its parent
        envelope *myMsg = (envelope *)(bcastInterimAckInfo->msg);

        // deregister using the message
#if CMK_REG_REQUIRED
        deregisterMemFromMsg(myMsg, true);
#endif
        // send a message to the parent to signal completion
        int srcPe;
        CkArray *mgr = NULL;
        envelope *env = (envelope *)bcastInterimAckInfo->msg;
        CkUnpackMessage(&env); // Unpack message before sending it to getParentBcastAckInfo
        char *ref = (char *)(getParentBcastAckInfo(bcastInterimAckInfo->msg, srcPe));
        CkPackMessage(&env);

        NcpyBcastInterimAckInfo *ncpyBcastAck = (NcpyBcastInterimAckInfo *)ref;
        CmiInvokeBcastAckHandler(ncpyBcastAck->origPe, ncpyBcastAck->parentBcastAckInfo);

        CMI_ZC_MSGTYPE(myMsg) = CMK_ZC_BCAST_RECV_DONE_MSG;

        CkUnpackMessage(&myMsg); // DO NOT REMOVE THIS

        if(bcastInterimAckInfo->isArray) {
          myMsg->setMsgtype(ForArrayEltMsg);

          mgr = getArrayMgrFromMsg(myMsg);
          mgr->forwardZCMsgToOtherElems(myMsg);
        }
#if CMK_SMP
        if(CmiMyNodeSize() > 1 && myMsg->getMsgtype() != ForNodeBocMsg) {
          sendRecvDoneMsgToPeers(myMsg, mgr);
        } else {
          if(myMsg->getMsgtype() == ForArrayEltMsg) {
            myMsg->setMsgtype(ForBocMsg);
            myMsg->getsetArrayEp() = mgr->getRecvBroadcastEpIdx();
          }
          enqueueNcpyMessage(bcastAckInfo->pe, myMsg);
        }
#else
        CMI_ZC_MSGTYPE(myMsg) = CMK_ZC_BCAST_RECV_ALL_DONE_MSG;

        if(myMsg->getMsgtype() == ForArrayEltMsg) {
          myMsg->setMsgtype(ForBocMsg);
          myMsg->getsetArrayEp() = mgr->getRecvBroadcastEpIdx();
        }
        enqueueNcpyMessage(bcastAckInfo->pe, myMsg);
#endif
      } else { // bcast send api

        envelope *myMsg = (envelope *)(bcastInterimAckInfo->msg);

        // deregister using the message
#if CMK_REG_REQUIRED
        deregisterMemFromMsg(myMsg, false);
#endif
        // send a message to the parent to signal completion
        envelope *env = (envelope *)bcastInterimAckInfo->msg;
        CkUnpackMessage(&env); // Unpack message before sending it to getParentBcastAckInfo
        sendAckMsgToParent(env);
        CkPackMessage(&env);

        forwardMessageToPeerNodes(myMsg, myMsg->getMsgtype());

        // enquque message to execute EM on the intermediate node
        enqueueNcpyMessage(bcastAckInfo->pe, bcastInterimAckInfo->msg);

        CmiFree(bcastInterimAckInfo);
      }
    }
  }
}

// Called only on intermediate nodes
// Method forwards a message to all the children
void forwardMessageToChildNodes(envelope *myChildrenMsg, UChar msgType) {
#if CMK_SMP && CMK_NODE_QUEUE_AVAILABLE
  if(msgType == ForNodeBocMsg) {
    // Node level forwarding for nodegroup bcasts
    CmiForwardNodeBcastMsg(myChildrenMsg->getTotalsize(), (char *)myChildrenMsg);
  } else
#endif
  // Proc level forwarding
  CmiForwardProcBcastMsg(myChildrenMsg->getTotalsize(), (char *)myChildrenMsg);
}

// Method forwards a message to all the peer nodes
void forwardMessageToPeerNodes(envelope *myMsg, UChar msgType) {
#if CMK_SMP
#if CMK_NODE_QUEUE_AVAILABLE
  if(msgType == ForBocMsg)
#endif // CMK_NODE_QUEUE_AVAILABLE
    CmiForwardMsgToPeers(myMsg->getTotalsize(), (char *)myMsg);
#endif
}

void handleBcastEntryMethodApiCompletion(NcpyOperationInfo *info){
  if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_DEST_ACK) {
    // invoking the entry method
    // Invoke the ackhandler function to update the counter
    CkRdmaEMAckHandler(info->destPe, info->refPtr);
  }
}

void handleBcastReverseEntryMethodApiCompletion(NcpyOperationInfo *info) {
  if(info->ackMode == CMK_SRC_DEST_ACK || info->ackMode == CMK_DEST_ACK) {
    // Invoke the remote ackhandler function
    CmiInvokeRemoteAckHandler(info->destPe, info->refPtr);
  }
  if(info->freeMe == CMK_FREE_NCPYOPINFO)
    CmiFree(info);
}

void deregisterMemFromMsg(envelope *env, bool isRecv) {
  CkUnpackMessage(&env);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  int numops;
  up|numops;
  p|numops;

  CkNcpyBuffer dest;

  for(int i=0; i<numops; i++){
    up|dest;

    // De-register the destination buffer when isRecv is false (i.e. using ZC Bcast Send API) or
    // when isRecv is true, respect deregMode and de-register
    if( (!isRecv) || (isRecv && dest.deregMode == CMK_BUFFER_DEREG) ) {
      CmiDeregisterMem(dest.ptr, (char *)dest.layerInfo + CmiGetRdmaCommonInfoSize(), dest.pe, dest.regMode);
    }

    p|dest;
  }
  CkPackMessage(&env);
}

/****************************** Zerocopy BCAST EM SEND API Functions ***********************/

void handleMsgUsingCMAPostCompletionForSendBcast(envelope *copyenv, envelope *env, CkNcpyBuffer &source) {
  CmiSpanningTreeInfo &t = *_topoTree;

  if(t.child_count == 0) { // child node

    // Send a message to the parent node to signal completion
    CmiInvokeBcastAckHandler(source.pe, (void *)source.bcastAckInfo);

    // Only forwarding is to peer PEs
    forwardMessageToPeerNodes(copyenv, copyenv->getMsgtype());

    // enqueue message to execute EM on the child node
    enqueueNcpyMessage(CkMyPe(), copyenv);

  } else { // intermediate node

    // Allocate a structure NcpyBcastInterimAckInfo to maintain state for ack handling
    NcpyBcastInterimAckInfo *bcastAckInfo = allocateInterimNodeAckObj(copyenv, env, CkMyPe());

    //// Replace parent pointers with my pointers for my children
    CkReplaceSourcePtrsInBcastMsg(env, copyenv, bcastAckInfo, CkMyPe());

    // Send message to children for them to Rget from me
    forwardMessageToChildNodes(env, copyenv->getMsgtype());
  }
}

void processBcastSendEmApiCompletion(NcpyEmInfo *ncpyEmInfo, int destPe) {
  CmiSpanningTreeInfo &t = *_topoTree;
  envelope *myEnv = (envelope *)(ncpyEmInfo->msg);

  if(t.child_count == 0) { // Child Node

    CkUnpackMessage(&myEnv); // Unpack message before sending it to sendAckMsgToParent
    sendAckMsgToParent(myEnv);
    CkPackMessage(&myEnv);

    // Since I am a child node, no more forwarding to any more childing
    // Only forwarding is to peer PEs
    forwardMessageToPeerNodes(myEnv, myEnv->getMsgtype());

    // enquque message to execute EM on the child node
    enqueueNcpyMessage(destPe, myEnv);

  } else { // Intermediate Node

    envelope *myChildEnv = (envelope *)(ncpyEmInfo->forwardMsg);

    // Allocate a structure NcpyBcastInterimAckInfo to maintain state for ack handling
    NcpyBcastInterimAckInfo *bcastAckInfo = allocateInterimNodeAckObj(myEnv, myChildEnv, ncpyEmInfo->pe);

    // Replace parent pointers with my pointers for my children
    CkReplaceSourcePtrsInBcastMsg(myChildEnv, myEnv, bcastAckInfo, ncpyEmInfo->pe);

    // Send message to children for them to Rget from me
    forwardMessageToChildNodes(myChildEnv, myEnv->getMsgtype());
  }
}

// Method called on intermediate nodes after RGET to switch old source pointers with my pointers
void CkReplaceSourcePtrsInBcastMsg(envelope *prevEnv, envelope *env, void *bcastAckInfo, int origPe) {

  int numops;

  CkUnpackMessage(&prevEnv);
  PUP::toMem p_prev((void *)(((CkMarshallMsg *)EnvToUsr(prevEnv))->msgBuf));
  PUP::fromMem up_prev((void *)((CkMarshallMsg *)EnvToUsr(prevEnv))->msgBuf);

  CkUnpackMessage(&env);
  CkUnpackRdmaPtrs((((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);

  up_prev|numops;
  up|numops;

  p|numops;
  p_prev|numops;

  for(int i=0; i<numops; i++){
    // source buffer
    CkNcpyBuffer prev_source, source;

    // unpack from previous message
    up_prev|prev_source;

    // unpack from current message
    up|source;

    const void *bcastAckInfoTemp = source.bcastAckInfo;
    int orig_source_pe = source.pe;

    source.bcastAckInfo = bcastAckInfo;
    source.pe = origPe;

    // pack updated CkNcpyBuffer into previous message
    p_prev|source;

    source.bcastAckInfo = bcastAckInfoTemp;
    source.pe = orig_source_pe;

    // pack back CkNcpyBuffer into current message
    p|source;
  }

  CkPackMessage(&prevEnv);

  CkPackRdmaPtrs((((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  CkPackMessage(&env);
}

/****************************** Zerocopy BCAST EM POST API Functions ***********************/

void processBcastRecvEmApiCompletion(NcpyEmInfo *ncpyEmInfo, int destPe) {
  CmiSpanningTreeInfo &t = *_topoTree;
  envelope *myEnv = (envelope *)(ncpyEmInfo->msg);

  if(t.child_count == 0) {  // Child Node
    // Send message to all peer elements on this PE
    // Send a message to the worker thread
#if CMK_SMP
    CmiInvokeBcastPostAckHandler(destPe, ncpyEmInfo->msg);
#else
    CkRdmaEMBcastPostAckHandler(ncpyEmInfo->msg);
#endif
    CmiFree(ncpyEmInfo); // Allocated in CkRdmaIssueRgets

  } else { // Intermediate Node

    // Send a message to the worker thread
    // NOTE:: ncpyEmInfo is sent instead of ncpyEmInfo->msg
#if CMK_SMP
    CmiInvokeBcastPostAckHandler(destPe, ncpyEmInfo);
#else
    CkRdmaEMBcastPostAckHandler(ncpyEmInfo);
#endif
  }
}

void CkRdmaEMBcastPostAckHandler(void *msg) {

  CmiSpanningTreeInfo &t = *_topoTree;

  // send a message to your parents if you are a child node
  if(t.child_count == 0) {

    // Send message to all peer elements on this PE
    envelope *env = (envelope *)(msg);
    sendAckMsgToParent(env);
    handleMsgOnChildPostCompletionForRecvBcast(env);

  } else if(t.child_count !=0 && t.parent != -1) {

    NcpyEmInfo *ncpyEmInfo = (NcpyEmInfo *)(msg);
    envelope *env = (envelope *)(ncpyEmInfo->msg);

    // Allocate a structure NcpyBcastInterimAckInfo to maintain state for ack handling
    NcpyBcastInterimAckInfo *bcastAckInfo = allocateInterimNodeAckObj(env, NULL, ncpyEmInfo->pe);
    handleMsgOnInterimPostCompletionForRecvBcast(env, bcastAckInfo, ncpyEmInfo->pe);

    CmiFree(ncpyEmInfo); // Allocated in CkRdmaIssueRgets

  } else {
    CmiAbort("parent node reaching CkRdmaEMBcastPostAckHandler\n");
  }

}

void CkReplaceSourcePtrsInBcastMsg(envelope *env, NcpyBcastInterimAckInfo *bcastAckInfo, int origPe) {

  int numops;
  CkUnpackMessage(&env);
  //CkUnpackRdmaPtrs((((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);

  up|numops;
  p|numops;

  // source buffer
  CkNcpyBuffer source;

  for(int i=0; i<numops; i++){
    // unpack from current message
    up|source;

    const void *bcastAckInfoTemp = source.bcastAckInfo;
    int orig_source_pe = source.pe;

    bcastAckInfo->parentBcastAckInfo = (void *)bcastAckInfoTemp;
    bcastAckInfo->origPe = orig_source_pe;

    source.bcastAckInfo = bcastAckInfo;
    source.pe = origPe;

    // pack back CkNcpyBuffer into current message
    p|source;
  }

  // CkPackRdmaPtrs((((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  CkPackMessage(&env);
}

#if CMK_SMP
void updatePeerCounterAndPush(envelope *env) {
  int pe;
  int numops;

  CkUnpackMessage(&env);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);

  up|numops;
  p|numops;

  CkNcpyBuffer source;

  up|source;

  pe = CmiNodeFirst(CmiMyNode());

  void *ref = (void *)source.bcastAckInfo;
  NcpyBcastRecvPeerAckInfo *peerAckInfo = (NcpyBcastRecvPeerAckInfo *)ref;
  source.bcastAckInfo = peerAckInfo->bcastAckInfo;

  p|source;
  CkPackMessage(&env);
  CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_RECV_ALL_DONE_MSG;
  CmiSpanningTreeInfo &t = *_topoTree;
  peerAckInfo->decNumPeers();
  if(peerAckInfo->getNumPeers() == 0) {
    CmiPushPE(CmiRankOf(peerAckInfo->peerParentPe), env);
  }
}

void sendRecvDoneMsgToPeers(envelope *env, CkArray *mgr) {

  CmiSpanningTreeInfo &t = *_topoTree;

  // Allocate a struct for handling peer acks
  NcpyBcastRecvPeerAckInfo *peerAckInfo = new NcpyBcastRecvPeerAckInfo();

  // Find how many peers I have
  peerAckInfo->setNumPeers(CmiMyNodeSize() - 1);
  peerAckInfo->msg = (void *)env;
  peerAckInfo->peerParentPe = CmiMyPe();

  int numops;

  // Replace bcastAckInfo with peerAckInfo
  CkUnpackMessage(&env);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);

  up|numops;
  p|numops;

  CkNcpyBuffer source;

  up|source;

  peerAckInfo->bcastAckInfo = (void *)source.bcastAckInfo;
  source.bcastAckInfo = peerAckInfo;

  p|source;

  CkPackMessage(&env);

  if(env->getMsgtype() == ForArrayEltMsg) {
    env->setMsgtype(ForBocMsg);
    env->getsetArrayEp() = mgr->getRecvBroadcastEpIdx();
  }
  CmiForwardMsgToPeers(env->getTotalsize(), (char *)env);
}
#endif

// Send a message to the parent node to signal completion
void sendAckMsgToParent(envelope *env)  {
  int srcPe;

  // srcPe is passed by reference and written inside the method
  char *ref = (char *)getParentBcastAckInfo(env,srcPe);

  // Invoke BcastAckHandler on the parent node to notify completion
  CmiInvokeBcastAckHandler(srcPe, ref);
}

CkArray* getArrayMgrFromMsg(envelope *env) {
  CkArray *mgr = NULL;
  CkGroupID gId = env->getArrayMgr();
  IrrGroup *obj = _getCkLocalBranchFromGroupID(gId);
  CkAssert(obj!=NULL);
  mgr = (CkArray *)obj;
  return mgr;
}

void handleArrayMsgOnChildPostCompletionForRecvBcast(envelope *env) {
  CkArray *mgr = getArrayMgrFromMsg(env);
  mgr->forwardZCMsgToOtherElems(env);

#if CMK_SMP
  if(CmiMyNodeSize() > 1) {
    sendRecvDoneMsgToPeers(env, mgr);
  } else
#endif
  {
    CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_RECV_ALL_DONE_MSG;
    env->setMsgtype(ForBocMsg);
    env->getsetArrayEp() = mgr->getRecvBroadcastEpIdx();
    CmiHandleMessage(env);
  }
}

void handleGroupMsgOnChildPostCompletionForRecvBcast(envelope *env) {
  CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_RECV_DONE_MSG;
#if CMK_SMP
  if(CmiMyNodeSize() > 1) {
    sendRecvDoneMsgToPeers(env, NULL);
  } else
#endif
  {
    CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_RECV_ALL_DONE_MSG;
    CmiHandleMessage(env);
  }
}

void handleNGMsgOnChildPostCompletionForRecvBcast(envelope *env) {
  CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_RECV_ALL_DONE_MSG;
  CmiHandleMessage(env);
}

void handleMsgOnChildPostCompletionForRecvBcast(envelope *env) {
  switch(env->getMsgtype()) {

    case ForArrayEltMsg : handleArrayMsgOnChildPostCompletionForRecvBcast(env);
                          break;
    case ForBocMsg      : handleGroupMsgOnChildPostCompletionForRecvBcast(env);
                          break;
    case ForNodeBocMsg  : handleNGMsgOnChildPostCompletionForRecvBcast(env);
                          break;
    default             : CmiAbort("Type of message currently not supported\n");
                          break;
  }
}

void handleMsgOnInterimPostCompletionForRecvBcast(envelope *env, NcpyBcastInterimAckInfo *bcastAckInfo, int pe) {
  // Replace parent pointers with my pointers for my children
  CkReplaceSourcePtrsInBcastMsg(env, bcastAckInfo, pe);

  CMI_ZC_MSGTYPE(env) = CMK_ZC_BCAST_RECV_MSG;

  if(env->getMsgtype() == ForArrayEltMsg) {
    CkArray *mgr = getArrayMgrFromMsg(env);
    env->setMsgtype(ForBocMsg);
    env->getsetArrayEp() = mgr->getRecvBroadcastEpIdx();
  }

  // Send message to children for them to Rget from me
  forwardMessageToChildNodes(env, env->getMsgtype());
}


/***************************** Zerocopy Readonly Bcast Support ****************************/

extern int _roRdmaDoneHandlerIdx,_initHandlerIdx;
CksvExtern(int, _numPendingRORdmaTransfers);
extern UInt numZerocopyROops, curROIndex;
extern NcpyROBcastAckInfo *roBcastAckInfo;

void readonlyUpdateNumops() {
  //update numZerocopyROops
  numZerocopyROops++;
}

// Method to allocate an object on the source for de-registration after bcast completes
void readonlyAllocateOnSource() {

  if(_topoTree == NULL) CkAbort("CkRdmaIssueRgets:: topo tree has not been calculated \n");
  CmiSpanningTreeInfo &t = *_topoTree;

  // allocate the buffer to keep track of the completed operations
  roBcastAckInfo = (NcpyROBcastAckInfo *)CmiAlloc(sizeof(NcpyROBcastAckInfo) + numZerocopyROops * sizeof(NcpyROBcastBuffAckInfo));

  roBcastAckInfo->counter = 0;
  roBcastAckInfo->isRoot = (t.parent == -1);
  roBcastAckInfo->numChildren = t.child_count;
  roBcastAckInfo->numops = numZerocopyROops;
}

// Method to initialize the allocated object with each source buffer's information
void readonlyCreateOnSource(CkNcpyBuffer &src) {
  src.bcastAckInfo = roBcastAckInfo;

  NcpyROBcastBuffAckInfo *buffAckInfo = &(roBcastAckInfo->buffAckInfo[curROIndex]);

  buffAckInfo->ptr = src.ptr;
  buffAckInfo->regMode = src.regMode;
  buffAckInfo->pe = src.pe;

  // store the source layer information for de-registration
  memcpy(buffAckInfo->layerInfo, src.layerInfo, CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES);

  curROIndex++;
}

// Method to perform an get for Readonly transfer
// In the case of SMP mode, readonlyGet is invoked on the worker thread (CmiMyRank == 0)
// In the case of Non-SMP mode, readonlyGet is invoked on the same process which receives RODataMsg
void readonlyGet(CkNcpyBuffer &src, CkNcpyBuffer &dest, void *refPtr) {

  CkAssert(CkMyRank() == 0);

  CmiSpanningTreeInfo &t = *_topoTree;

  CkNcpyMode transferMode = findTransferMode(src.pe, dest.pe);
  if(transferMode == CkNcpyMode::MEMCPY) {
    CmiAbort("memcpy: should not happen\n");
  }
#if CMK_USE_CMA
  else if(transferMode == CkNcpyMode::CMA) {
    dest.cmaGet(src);

    // Decrement _numPendingRORdmaTransfers after completion of an Get operation
    CksvAccess(_numPendingRORdmaTransfers)--;

    // Initialize previously allocated structure for ack tracking on intermediate nodes
    if(t.child_count != 0)  // Intermediate Node
      readonlyCreateOnSource(dest);

    // When all pending RO Rdma transfers are complete
    if(CksvAccess(_numPendingRORdmaTransfers) == 0) {

      if(t.child_count != 0) {  // Intermediate Node

        // Send a message to my child nodes
        envelope *env = (envelope *)(refPtr);
        CmiForwardProcBcastMsg(env->getTotalsize(), (char *)env);

      } else { // Child Node

        // deregister dest buffer
        CmiDeregisterMem(dest.ptr, dest.layerInfo + CmiGetRdmaCommonInfoSize(), dest.pe, dest.regMode);

        // Send a message to the parent to signal completion in order to deregister
        envelope *compEnv = _allocEnv(ROChildCompletionMsg);
        compEnv->setSrcPe(CkMyPe());
        CmiSetHandler(compEnv, _roRdmaDoneHandlerIdx);
        CmiSyncSendAndFree(src.pe, compEnv->getTotalsize(), (char *)compEnv);
      }

      // Directly call checkInitDone to notify RO Rdma completion
      checkForInitDone(true);
    }
  }
#endif
  else {


    int layerInfoSize = CMK_COMMON_NOCOPY_DIRECT_BYTES + CMK_NOCOPY_DIRECT_BYTES;
    int ackSize = 0;
    int ncpyObjSize = getNcpyOpInfoTotalSize(
                      layerInfoSize,
                      ackSize,
                      layerInfoSize,
                      ackSize);
    NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)CmiAlloc(ncpyObjSize);
    setNcpyOpInfo(src.ptr,
                  (char *)(src.layerInfo),
                  layerInfoSize,
                  NULL,
                  ackSize,
                  src.cnt,
                  src.regMode,
                  src.deregMode,
                  src.isRegistered,
                  src.pe,
                  src.ref,
                  dest.ptr,
                  (char *)(dest.layerInfo),
                  layerInfoSize,
                  NULL,
                  ackSize,
                  dest.cnt,
                  dest.regMode,
                  dest.deregMode,
                  dest.isRegistered,
                  dest.pe,
                  dest.ref,
                  ncpyOpInfo);

    ncpyOpInfo->opMode = CMK_READONLY_BCAST;
    ncpyOpInfo->refPtr = refPtr;

    // Initialize previously allocated structure for ack tracking on intermediate nodes
    if(t.child_count != 0)
      readonlyCreateOnSource(dest);

    CmiIssueRget(ncpyOpInfo);
  }
}

// Method invoked when an RO RDMA operation is complete
// In the case of SMP mode, readonlyGetCompleted is invoked on the comm. thread
// In the case of Non-SMP mode, readonlyGet is invoked on the same process which receives the RODataMsg
void readonlyGetCompleted(NcpyOperationInfo *ncpyOpInfo) {

  if(_topoTree == NULL) CkAbort("CkRdmaIssueRgets:: topo tree has not been calculated \n");
  CmiSpanningTreeInfo &t = *_topoTree;

  // Lock not needed for SMP mode as no other thread decrements _numPendingRORdmaTransfers
  CksvAccess(_numPendingRORdmaTransfers)--;

  // When all pending RO Rdma transfers are complete
  if(CksvAccess(_numPendingRORdmaTransfers) == 0) {

    if(t.child_count != 0) {  // Intermediate Node

      envelope *env = (envelope *)(ncpyOpInfo->refPtr);

      // Send a message to my child nodes
      CmiForwardProcBcastMsg(env->getTotalsize(), (char *)env);

      //TODO:QD support

    } else {

      // deregister dest buffer
      CmiDeregisterMem(ncpyOpInfo->destPtr, ncpyOpInfo->destLayerInfo + CmiGetRdmaCommonInfoSize(), ncpyOpInfo->destPe, ncpyOpInfo->destRegMode);

      // Send a message to the parent to signal completion in order to deregister
      envelope *compEnv = _allocEnv(ROChildCompletionMsg);
      compEnv->setSrcPe(CkMyPe());
      CmiSetHandler(compEnv, _roRdmaDoneHandlerIdx);
      CmiSyncSendAndFree(ncpyOpInfo->srcPe, compEnv->getTotalsize(), (char *)compEnv);
    }

#if CMK_SMP
    // Send a message to my first node to signal completion
    envelope *sigEnv = _allocEnv(ROPeerCompletionMsg);
    sigEnv->setSrcPe(CkMyPe());
    CmiSetHandler(sigEnv, _roRdmaDoneHandlerIdx);
    CmiSyncSendAndFree(CmiNodeFirst(CmiMyNode()), sigEnv->getTotalsize(), (char *)sigEnv);
#else
    // Directly call checkInitDone to notify RO Rdma completion
    checkForInitDone(true);
#endif

  }

  // Free ncpyOpInfo allocated inside readonlyGet
  CmiFree(ncpyOpInfo);
}
#endif
/* End of CMK_ONESIDED_IMPL */
