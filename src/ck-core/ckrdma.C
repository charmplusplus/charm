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

#if CMK_ONESIDED_IMPL
/* Sender Functions */

/*
 * Add the machine specific information if metadata message
 * sent across network. Do nothing if message is sent to same PE/Node
 */
void CkRdmaPrepareMsg(envelope **env, int pe){
  if(CmiNodeOf(pe)!=CmiMyNode()){
    envelope *prevEnv = *env;
    *env = CkRdmaCreateMetadataMsg(prevEnv, pe);
    CkFreeMsg(EnvToUsr(prevEnv));
  }
#if CMK_SMP && CMK_IMMEDIATE_MSG
  //Reset the immediate bit if it is a within node/pe message
  else
    CmiResetImmediate(*env);
#endif
}

/*
 * Add machine specific information to the msg. This includes information
 * information both common and specific to each rdma parameter
 * Metdata message format: <-env-><-msg-><-migen-><-mispec1-><-mispec2->..<-mispecn->
 * migen: machine info generic to rdma operation
 * mispec: machine info specific to rdma operation for n rdma operations
 */
envelope* CkRdmaCreateMetadataMsg(envelope *env, int pe){

  int numops = getRdmaNumOps(env);
  int msgsize = env->getTotalsize();

  //CmiGetRdmaInfoSize returns the size of machine specific information
  // for numops RDMA operations
  int mdsize = msgsize + CmiGetRdmaInfoSize(numops);

  //pack before copying
  CkPackMessage(&env);

  //Allocate a new metadata message, set machine specific info
  envelope *copyenv = (envelope *)CmiAlloc(mdsize);
  memcpy(copyenv, env, msgsize);
  copyenv->setTotalsize(mdsize);
#if CMK_SMP && CMK_IMMEDIATE_MSG
  CmiBecomeImmediate(copyenv);
#endif

  //Set the generic information
  char *md = (char *)copyenv + msgsize;
  CmiSetRdmaInfo(md, pe, numops);
  md += CmiGetRdmaGenInfoSize();

  //Set the rdma op specific information
  CkUnpackMessage(&copyenv);
  PUP::fromMem up((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  up|numops;
  p|numops;
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w; up|w;
    CkRdmaWrapper *wack = new CkRdmaWrapper(w);
    void *ack = CmiSetRdmaAck(CkHandleRdmaCookie, wack);
    w.callback = (CkCallback *)ack;
    p|w;
    CmiSetRdmaOpInfo(md, w.ptr, w.cnt, w.callback, pe);
    md += CmiGetRdmaOpInfoSize();
  }
  CkPackMessage(&copyenv);

  //return the env with the machine specific information
  return copyenv;
}

/*
 * Method called on sender when ack is received
 * Access the CkRdmaWrapper using the cookie passed and invoke callback
 */
void CkHandleRdmaCookie(void *cookie){
  CkRdmaWrapper *w = (CkRdmaWrapper *)cookie;
  CkCallback *cb= w->callback;
#if CMK_SMP
  //call the callbackgroup on my node's first PE when callback requires to be called from comm thread
  //this adds one more trip through the scheduler
  _ckcallbackgroup[CmiNodeFirst(CmiMyNode())].call(*cb, sizeof(void *), (char*)&w->ptr);
#else
  //direct call to callback when calling from worker thread
  cb->send(sizeof(void *), &w->ptr);
#endif

  delete cb;
  delete (CkRdmaWrapper *)cookie;
}


/* Receiver Functions */

/*
 * Method called when received message is on the same PE/Node
 * This involves a direct copy from the sender's buffer into the receiver's buffer
 * A new message is allocated with size = existing message size + sum of all rdma buffers.
 * The newly allocated message contains the existing marshalled message along with
 * space for the rdma buffers which are copied from the source buffers.
 * The buffer and the message are allocated contiguously to free the buffer
 * when the message gets free.
 */
envelope* CkRdmaCopyMsg(envelope *env){
  int numops, bufsize, msgsize;
  bufsize = getRdmaBufSize(env);
  msgsize = env->getTotalsize();

  CkPackMessage(&env);
  envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + bufsize);
  memcpy(copyenv, env, msgsize);
  copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + bufsize);
  copyenv->setRdma(false);

  char* buf = (char *)copyenv + CK_ALIGN(msgsize, 16);
  CkUnpackMessage(&copyenv);
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf));
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  up|numops;
  p|numops;
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    //Copy the buffer from the source pointer to the message
    memcpy(buf, w.ptr, w.cnt);

    //Invoke callback as it is safe to rewrite into the source buffer
    (w.callback)->send(sizeof(void *), &w.ptr);
    delete w.callback;

    //Update the CkRdmaWrapper pointer of the new message
    w.ptr = buf;

    //Update the pointer
    buf += CK_ALIGN(w.cnt, 16);
    p|w;
  }
  CkPackRdmaPtrs(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  CkPackMessage(&copyenv);
  return copyenv;
}

/*
 * Extract rdma based information from the metadata message,
 * allocate buffers and issue RDMA get call
 */
void CkRdmaIssueRgets(envelope *env){
  /*
   * Determine the buffer size('bufsize') and the message size('msgsize')
   * from the metadata message. 'msgSize' is the metadata message size
   * without the sender's machine specific information.
   */
  int numops, bufsize, msgsize;
  bufsize = getRdmaBufSize(env);
  numops = getRdmaNumOps(env);
  msgsize = env->getTotalsize() - CmiGetRdmaInfoSize(numops);

  /* Allocate the receiver's message, which contains the metadata message sent by the sender
   * (without its machine specific info) of size 'msgSize', the entire receiver's buffer of
   * size 'bufsize', and the receiver's machine specific info of size 'CmiGetRdmaRecvInfoSize(numops)'.
   * The receiver's machine specific info is added to this message to avoid separately freeing it
   * in the machine layer.
   */
  envelope *copyenv = (envelope *)CmiAlloc(CK_ALIGN(msgsize, 16) + bufsize + CmiGetRdmaRecvInfoSize(numops));

  //Copy the metadata message(without the machine specific info) into the buffer
  memcpy(copyenv, env, msgsize);

#if CMK_SMP && CMK_IMMEDIATE_MSG
  CmiResetImmediate(copyenv);
#endif

  //Receiver's machine specific info is at an offset, after the sender md and the receiver's buffer
  char *recv_md = ((char *)copyenv) + CK_ALIGN(msgsize, 16) + bufsize;

  /* Set the total size of the message excluding the receiver's machine specific info
   * which is not required when the receiver's entry method executes
   */
  copyenv->setTotalsize(CK_ALIGN(msgsize, 16) + bufsize);

  CkUnpackMessage(&copyenv);
  CkUpdateRdmaPtrs(copyenv, msgsize, recv_md, ((char *)env) + msgsize);
  CkPackRdmaPtrs(((CkMarshallMsg *)EnvToUsr(copyenv))->msgBuf);
  CkPackMessage(&copyenv);

  // Set rdma to be false to prevent message handler on the receiver
  // from intercepting it
  copyenv->setRdma(false);

  // Free the existing message
  CkFreeMsg(EnvToUsr(env));

  //Call the lower layer API for performing RDMA gets
  CmiIssueRgets(recv_md, copyenv->getSrcPe());
}


/*
 * Method called to update machine specific information for receiver
 * using the metadata message given by the sender along with updating
 * pointers of the CkRdmawrappers
 * - assumes that the msg is unpacked
 */
void CkUpdateRdmaPtrs(envelope *env, int msgsize, char *recv_md, char *src_md){
  PUP::toMem p((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  PUP::fromMem up((void *)(((CkMarshallMsg *)EnvToUsr(env))->msgBuf));
  int numops;
  up|numops;
  p|numops;

  //Use the metadata info to set the machine info for receiver
  //generic info for all RDMA operations
  CmiSetRdmaRecvInfo(recv_md, numops, env, src_md, env->getTotalsize());
  recv_md += CmiGetRdmaGenRecvInfoSize();
  char *buf = ((char *)env) + CK_ALIGN(msgsize, 16);
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    //Set RDMA operation specific info
    CmiSetRdmaRecvOpInfo(recv_md, buf, w.callback, w.cnt, i, src_md);
    recv_md += CmiGetRdmaOpRecvInfoSize();
    w.ptr = buf;
    buf += CK_ALIGN(w.cnt, 16);
    p|w;
  }
}


/*
 * Method called to pack rdma pointers inside CkRdmaWrappers in the message
 * Assumes that msg is unpacked
 */
void CkPackRdmaPtrs(char *msgBuf){
  PUP::toMem p((void *)msgBuf);
  PUP::fromMem up((void *)msgBuf);
  int numops;
  up|numops;
  p|numops;

  //Pack Rdma pointers in CkRdmaWrappers
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    w.ptr = (void *)((char *)w.ptr - (char *)msgBuf);
    p|w;
  }
}


/*
 * Method called to unpack rdma pointers inside CkRdmaWrappers in the message
 * Assumes that msg is unpacked
 */
void CkUnpackRdmaPtrs(char *msgBuf){
  PUP::toMem p((void *)msgBuf);
  PUP::fromMem up((void *)msgBuf);
  int numops;
  up|numops;
  p|numops;

  //Unpack Rdma pointers in CkRdmaWrappers
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w;
    up|w;
    w.ptr = (void *)((char *)msgBuf + (size_t)w.ptr);
    p|w;
  }
}


//Determine the number of rdma ops from the message
int getRdmaNumOps(envelope *env){
  int numops;
  CkUnpackMessage(&env);
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  CkPackMessage(&env);
  return numops;
}

/*
 * Determine the total size of the buffers to be copied
 * This is to be allocated at the end of the message
 */
int getRdmaBufSize(envelope *env){
  /*
   * Determine the number of rdma operations and the sum of all
   * rdma buffer sizes by iterating over the ckrdmawrappers in the message
   */
  int numops, bufsize=0;
  CkUnpackMessage(&env);
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  up|numops;
  for(int i=0; i<numops; i++){
    CkRdmaWrapper w; up|w;
    bufsize += CK_ALIGN(w.cnt, 16);
  }
  CkPackMessage(&env);
  return bufsize;
}

#endif
/* End of CMK_ONESIDED_IMPL */

/* Support for Direct Nocopy API */

// Ack handler function which invokes the callback
void CkRdmaDirectAckHandler(void *ack) {

  // Process QD to mark completion of the outstanding RDMA operation
  QdProcess(1);

  NcpyOperationInfo *info = (NcpyOperationInfo *)(ack);

  CkCallback *srcCb = (CkCallback *)(info->srcAck);
  CkCallback *destCb = (CkCallback *)(info->destAck);

  CkNcpyBuffer src, dest;

  if(srcCb->requiresMsgConstruction()) {
    // reconstruct the CkNcpyBuffer object for the source
    src.ptr = info->srcPtr;
    src.pe  = info->srcPe;
    src.cnt = info->srcSize;
    src.ref = info->srcRef;
    src.mode = info->srcMode;
    src.isRegistered = info->isSrcRegistered;
    memcpy((char *)(&src.cb), srcCb, info->srcAckSize); // initialize cb
    memcpy((char *)(src.layerInfo), info->srcLayerInfo, info->srcLayerSize); // initialize layerInfo
  }

  if(destCb->requiresMsgConstruction()) {
    // reconstruct the CkNcpyBuffer object for the destination
    dest.ptr = info->destPtr;
    dest.pe  = info->destPe;
    dest.cnt = info->destSize;
    dest.ref = info->destRef;
    dest.mode = info->destMode;
    dest.isRegistered = info->isDestRegistered;
    memcpy((char *)(&dest.cb), destCb, info->destAckSize); // initialize cb
    memcpy((char *)(dest.layerInfo), info->destLayerInfo, info->destLayerSize); // initialize layerInfo
  }

  if(info->ackMode == 0 || info->ackMode == 1) {

#if CMK_SMP
    //call the callbackgroup on my node's first PE when callback requires to be called from comm thread
    //this adds one more trip through the scheduler
    _ckcallbackgroup[CmiNodeFirst(CmiMyNode())].call(*srcCb, sizeof(CkNcpyBuffer), (const char *)(&src));
#else
    //Invoke the destination callback
    srcCb->send(sizeof(CkNcpyBuffer), &src);
#endif
  }

  if(info->ackMode == 0 || info->ackMode == 2) {

#if CMK_SMP
    //call the callbackgroup on my node's first PE when callback requires to be called from comm thread
    //this adds one more trip through the scheduler
    _ckcallbackgroup[CmiNodeFirst(CmiMyNode())].call(*destCb, sizeof(CkNcpyBuffer), (const char *)(&dest));
#else
    //Invoke the destination callback
    destCb->send(sizeof(CkNcpyBuffer), &dest);
#endif
  }

  if(info->freeMe)
    CmiFree(info);
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
CkNcpyStatus CkNcpyBuffer::get(CkNcpyBuffer &source, CkNcpyCallbackMode cbMode){
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

    //Invoke the destination callback if cbMode is CB_INVOKE_ALL
    if(cbMode == CkNcpyCallbackMode::CB_INVOKE_ALL)
      cb.send(sizeof(CkNcpyBuffer), this);

    // rdma data transfer complete
    return CkNcpyStatus::complete;

#if CMK_USE_CMA
  } else if(transferMode == CkNcpyMode::CMA) {

    cmaGet(source);

    //Invoke the source callback
    source.cb.send(sizeof(CkNcpyBuffer), &source);

    //Invoke the destination callback if cbMode is CB_INVOKE_ALL
    if(cbMode == CkNcpyCallbackMode::CB_INVOKE_ALL)
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
CkNcpyStatus CkNcpyBuffer::put(CkNcpyBuffer &destination, CkNcpyCallbackMode cbMode){
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

    //Invoke the source callback if cbMode is CB_INVOKE_ALL
    if(cbMode == CkNcpyCallbackMode::CB_INVOKE_ALL)
      cb.send(sizeof(CkNcpyBuffer), this);

    // rdma data transfer complete
    return CkNcpyStatus::complete;

#if CMK_USE_CMA
  } else if(transferMode == CkNcpyMode::CMA) {
    cmaPut(destination);

    //Invoke the destination callback
    destination.cb.send(sizeof(CkNcpyBuffer), &destination);

    //Invoke the source callback if cbMode is CB_INVOKE_ALL
    if(cbMode == CkNcpyCallbackMode::CB_INVOKE_ALL)
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
