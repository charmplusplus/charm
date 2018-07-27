/*
  included in machine.C
  Nitin Bhat 03/07/2017
*/

//Post MPI_Irecvs to receive the RDMA buffers
void LrtsIssueRgets(void *recv, int pe){
  int i;
  CmiMPIRzvRdmaRecvList_t *recvInfo = (CmiMPIRzvRdmaRecvList_t *)recv;
  MPI_Request reqBufferRecv;
  int srcPe = recvInfo->srcPe;

  for(i=0; i<recvInfo->numOps; i++){
    void *buffer = recvInfo->rdmaOp[i].buffer;
    int size = recvInfo->rdmaOp[i].size;
    int srcTag = recvInfo->rdmaOp[i].tag;

    MPIPostOneBuffer(buffer, (char *)(&(recvInfo->rdmaOp[i])), size, srcPe, srcTag, ONESIDED_BUFFER_RECV);  
  }
}

// Post MPI_Isend or MPI_Irecv to send/recv the RDMA buffer
// Used by both non-SMP and SMP modes
void MPISendOrRecvOneBuffer(SMSG_LIST *smsg, int tag){
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

  if(smsg->type == ONESIDED_BUFFER_DIRECT_SEND || smsg->type == ONESIDED_BUFFER_SEND) {
    if (MPI_SUCCESS != MPI_Isend((void *)msg, size, MPI_BYTE, dstrank, tag, charmComm, &(smsg->req)))
      CmiAbort("LrtsSendBuffer: MPI_Isend failed!\n");
  } else if(smsg->type == ONESIDED_BUFFER_DIRECT_RECV || smsg->type == ONESIDED_BUFFER_RECV) {
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

  SMSG_LIST *msg_tmp = allocateSmsgList((char *)buffer, destRank, size, 0, type, ref);

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

// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfoMsg) {

  // Generate a new tag
  int tag = getNewMPITag();
  SMSG_LIST *msg_tmp;
  MPI_Request reqBufferRecv;

  // Mark the message type as POST_DIRECT_SEND
  // On receiving a POST_DIRECT_SEND, the MPI rank should post an MPI_Isend
  CMI_MSGTYPE(ncpyOpInfoMsg) = POST_DIRECT_SEND;

  // Send the tag for the receiver MPI rank to post an MPI_Isend
  ncpyOpInfoMsg->tag = tag;

  // Determine the remote rank
  int destLocalNode = CmiNodeOf(ncpyOpInfoMsg->srcPe);
  int destRank = CmiGetNodeGlobal(destLocalNode, CmiMyPartition());

#if CMK_SMP
  if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
    EnqueueMsg(ncpyOpInfoMsg, ncpyOpInfoMsg->ncpyOpInfoSize, destRank, 0, POST_DIRECT_SEND, NULL);
  }
  else
#endif
  {
    msg_tmp = allocateSmsgList((char *)ncpyOpInfoMsg, destRank, ncpyOpInfoMsg->ncpyOpInfoSize, 0, POST_DIRECT_SEND, NULL);
    MPISendOneMsg(msg_tmp);
  }

  // Post an MPI_Irecv for the destination buffer with the tag
  // ONESIDED_BUFFER_DIRECT_RECV indicates that the method should post an irecv
  MPIPostOneBuffer(ncpyOpInfoMsg->destPtr, ncpyOpInfoMsg, ncpyOpInfoMsg->srcSize, ncpyOpInfoMsg->srcPe, tag, ONESIDED_BUFFER_DIRECT_RECV);
}

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfoMsg) {

  // Generate a new tag
  int tag = getNewMPITag();
  SMSG_LIST *msg_tmp;

  // Mark the message type as POST_DIRECT_RECV
  // On receiving a POST_DIRECT_RECV, the MPI rank should post an MPI_Irecv
  CMI_MSGTYPE(ncpyOpInfoMsg) = POST_DIRECT_RECV;

  // Send the tag for the receiver MPI rank to post an MPI_Irecv
  ncpyOpInfoMsg->tag = tag;

  // Determine the remote rank
  int destLocalNode = CmiNodeOf(ncpyOpInfoMsg->destPe);
  int destRank = CmiGetNodeGlobal(destLocalNode, CmiMyPartition());

#if CMK_SMP
  if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
    EnqueueMsg(ncpyOpInfoMsg, ncpyOpInfoMsg->ncpyOpInfoSize, destRank, 0, POST_DIRECT_RECV, NULL);
  }
  else
#endif
  {
    msg_tmp = allocateSmsgList((char *)ncpyOpInfoMsg, destRank, ncpyOpInfoMsg->ncpyOpInfoSize, 0, POST_DIRECT_RECV, NULL);
    MPISendOneMsg(msg_tmp);
  }

  // Post an MPI_ISend for the source buffer with the tag
  // ONESIDED_BUFFER_DIRECT_SEND indicates that the method should post an isend
  MPIPostOneBuffer(ncpyOpInfoMsg->srcPtr, ncpyOpInfoMsg, ncpyOpInfoMsg->srcSize, ncpyOpInfoMsg->destPe, tag, ONESIDED_BUFFER_DIRECT_SEND);
}

// Method invoked to deregister source memory (Empty method to maintain API consistency)
void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode){
}
