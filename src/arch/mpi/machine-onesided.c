/*
  included in machine.c
  Nitin Bhat 03/07/2017
*/

#include "machine-rdma.h"

//Post MPI_Irecvs to receive the RDMA buffers
void LrtsIssueRgets(void *recv, int pe){
  int i;
  CmiMPIRzvRdmaRecvList_t *recvInfo = (CmiMPIRzvRdmaRecvList_t *)recv;
  MPI_Request reqBufferRecv;
  int srcRank = recvInfo->srcRank;

  for(i=0; i<recvInfo->numOps; i++){
    void *buffer = recvInfo->rdmaOp[i].buffer;
    int size = recvInfo->rdmaOp[i].size;
    int srcTag = recvInfo->rdmaOp[i].tag;

    if(MPI_SUCCESS != MPI_Irecv(buffer, size, MPI_BYTE, srcRank, srcTag, charmComm, &reqBufferRecv))
      CmiAbort("LrtsIssueRgets: MPI_Irecv failed!\n");
    recvInfo->rdmaOp[i].req = reqBufferRecv;
  }

  //Add receiver's information to the list to wait on it for completion
  CpvAccess(RdmaRecvQueueLen)++;
  if (CpvAccess(recvRdmaBuffers)==0)
    CpvAccess(recvRdmaBuffers) = recvInfo;
  else
    CpvAccess(endRdmaBuffer)->next = recvInfo;
  CpvAccess(endRdmaBuffer) = recvInfo;
}

// Post MPI_Isend or MPI_Irecv to send/recv the RDMA buffer
// Used by both non-SMP and SMP modes
CmiCommHandle MPISendOrRecvOneBuffer(SMSG_LIST *smsg, int tag){
  int node = smsg->destpe;
  int size = smsg->size;

  char *msg = smsg->msg;
  int dstrank;

#if CMK_MEM_CHECKPOINT || CMK_MESSAGE_LOGGING
  dstrank = petorank[node];
  smsg->dstrank = dstrank;
#else
  dstrank=node;
#endif

  if(smsg->type == ONESIDED_BUFFER_DIRECT_SEND || smsg->type == ONESIDED_BUFFER) {
    if (MPI_SUCCESS != MPI_Isend((void *)msg, size, MPI_BYTE, dstrank, tag, charmComm, &(smsg->req)))
      CmiAbort("LrtsSendBuffer: MPI_Isend failed!\n");
  } else if(smsg->type == ONESIDED_BUFFER_DIRECT_RECV) {
    if (MPI_SUCCESS != MPI_Irecv((void *)msg, size, MPI_BYTE, dstrank, tag, charmComm, &(smsg->req)))
      CmiAbort("LrtsSendBuffer: MPI_Irecv failed!\n");
  } else {
    CmiAbort("Invalid type of smsg\n");
  }

  //Add sender's buffer information to the list to wait on it for completion
  CpvAccess(MsgQueueLen)++;
  if (CpvAccess(sent_msgs)==0)
    CpvAccess(sent_msgs) = smsg;
  else {
    CpvAccess(end_sent)->next = smsg;
  }
  CpvAccess(end_sent) = smsg;
}

//Post MPI_Isend directly for non-smp or through the comm thread for smp mode
void MPIPostOneBuffer(const void *buffer, void *ref, int size, int pe, int tag, int type) {

  int destLocalNode = CmiNodeOf(pe);
  int destRank = CmiGetNodeGlobal(destLocalNode, CmiMyPartition());

  SMSG_LIST *msg_tmp = allocateSmsgList((void *)buffer, destRank, size, 0, type, ref);

#if CMK_SMP
#if MULTI_SENDQUEUE
  PCQueuePush(procState[CmiMyRank()].postMsgBuf,(char *)msg_tmp);
  return;
#else
  if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
    PCQueuePush(postMsgBuf,(char *)msg_tmp);
    return;
  }
#endif
#endif //end of CMK_SMP

  MPISendOrRecvOneBuffer(msg_tmp, tag);
}

/* Support for Nocopy Direct API */

// Perform an RDMA Get call into the local target address from the remote source address
void LrtsIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  int size) {

  // Generate a new tag
  int tag = getNewMPITag();
  SMSG_LIST *msg_tmp;
  MPI_Request reqBufferRecv;

  // Send a message to the source with a tag to have the source post an MPI_ISend
  int postInfoMsgSize = CmiMsgHeaderSizeBytes + sizeof(CmiMPIRzvRdmaPostInfo_t) + srcAckSize;
  char *postInfoMsg = (char *)CmiAlloc(postInfoMsgSize);

  CmiMPIRzvRdmaPostInfo_t *postInfo = (CmiMPIRzvRdmaPostInfo_t *)(postInfoMsg + CmiMsgHeaderSizeBytes);
  postInfo->buffer = (void *)srcAddr;
  postInfo->size = size;
  postInfo->tag = tag;
  postInfo->ackSize = srcAckSize;
  postInfo->tgtPe = tgtPe;
  postInfo->srcPe = srcPe;

  // Copy the source ack so that the remote source can invoke it after completion
  memcpy((char *)(postInfo) + sizeof(CmiMPIRzvRdmaPostInfo_t),
         srcAck,
         srcAckSize);

  // Mark the message type as POST_DIRECT_SEND
  // On receiving a POST_DIRECT_SEND, the MPI rank should post an MPI_Isend
  CMI_MSGTYPE(postInfoMsg) = POST_DIRECT_SEND;

  // Determine the remote rank
  int destLocalNode = CmiNodeOf(srcPe);
  int destRank = CmiGetNodeGlobal(destLocalNode, CmiMyPartition());

#if CMK_SMP
  if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
    EnqueueMsg(postInfoMsg, postInfoMsgSize, destRank, 0, POST_DIRECT_SEND, NULL);
  }
  else
#endif
  {
    msg_tmp = allocateSmsgList(postInfoMsg, destRank, postInfoMsgSize, 0, POST_DIRECT_SEND, NULL);
    MPISendOneMsg(msg_tmp);
  }

  // Create a local object to invoke the acknowledgement on completion of the MPI_Irecv
  CmiMPIRzvRdmaAckInfo_t *tgtAckNew = (CmiMPIRzvRdmaAckInfo_t *)malloc(sizeof(CmiMPIRzvRdmaAckInfo_t) + tgtAckSize);
  tgtAckNew->pe = tgtPe;
  tgtAckNew->tag = tag;
  memcpy((char *)tgtAckNew + sizeof(CmiMPIRzvRdmaAckInfo_t),
         tgtAck,
         tgtAckSize);

  // Post an MPI_Irecv for the target buffer with the tag
  // ONESIDED_BUFFER_DIRECT_RECV indicates that the method should post an irecv
  MPIPostOneBuffer(tgtAddr, tgtAckNew, size, srcPe, tag, ONESIDED_BUFFER_DIRECT_RECV);
}

// Perform an RDMA Put call into the remote target address from the local source address
void LrtsIssueRput(
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  int size) {

  // Generate a new tag
  int tag = getNewMPITag();
  SMSG_LIST *msg_tmp;

  // Send a message to the target with a tag to have the target post a MPI_Irecv
  int postInfoMsgSize = CmiMsgHeaderSizeBytes + sizeof(CmiMPIRzvRdmaPostInfo_t) + tgtAckSize;
  char *postInfoMsg = (char *)CmiAlloc(postInfoMsgSize);

  CmiMPIRzvRdmaPostInfo_t *postInfo = (CmiMPIRzvRdmaPostInfo_t *)(postInfoMsg + CmiMsgHeaderSizeBytes);
  postInfo->buffer = (void *)tgtAddr;
  postInfo->size = size;
  postInfo->tag = tag;
  postInfo->ackSize = tgtAckSize;
  postInfo->srcPe = srcPe;
  postInfo->tgtPe = tgtPe;

  // Copy the target ack so that the remote target can invoke it after completion
  memcpy((char *)(postInfo) + sizeof(CmiMPIRzvRdmaPostInfo_t),
         tgtAck,
         tgtAckSize);

  // Mark the message type as POST_DIRECT_RECV
  // On receiving a POST_DIRECT_RECV, the MPI rank should post an MPI_Irecv
  CMI_MSGTYPE(postInfoMsg) = POST_DIRECT_RECV;

  // Determine the remote rank
  int destLocalNode = CmiNodeOf(tgtPe);
  int destRank = CmiGetNodeGlobal(destLocalNode, CmiMyPartition());

#if CMK_SMP
  if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
    EnqueueMsg(postInfoMsg, postInfoMsgSize, destRank, 0, POST_DIRECT_RECV, NULL);
  }
  else
#endif
  {
    msg_tmp = allocateSmsgList(postInfoMsg, destRank, postInfoMsgSize, 0, POST_DIRECT_RECV, NULL);
    MPISendOneMsg(msg_tmp);
  }

  // Create a local object to invoke the acknowledgement on completion of the MPI_Isend
  CmiMPIRzvRdmaAckInfo_t *srcAckNew = (CmiMPIRzvRdmaAckInfo_t *)malloc(sizeof(CmiMPIRzvRdmaAckInfo_t) + srcAckSize);
  srcAckNew->pe = srcPe;
  srcAckNew->tag = tag;
  memcpy((char *)srcAckNew + sizeof(CmiMPIRzvRdmaAckInfo_t),
         srcAck,
         srcAckSize);

  // Post an MPI_ISend for the source buffer with the tag
  // ONESIDED_BUFFER_DIRECT_SEND indicates that the method should post an isend
  MPIPostOneBuffer(srcAddr, (void *)srcAckNew, size, tgtPe, tag, ONESIDED_BUFFER_DIRECT_SEND);
}

// Method invoked to deregister target memory handle
void LrtsReleaseTargetResource(void *info, int pe){
}

// Method invoked to deregister source memory handle
void LrtsReleaseSourceResource(void *info, int pe){
}
