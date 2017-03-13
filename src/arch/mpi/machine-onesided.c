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

//Post MPI_Isend to send the RDMA buffer
//Used by both non-SMP and SMP modes
CmiCommHandle MPISendOneRdmaBuffer(SMSG_LIST *smsg){
  int node = smsg->destpe;
  int size = smsg->size;

  CmiMPIRzvRdmaOpInfo_t *rdmaOpInfo = (CmiMPIRzvRdmaOpInfo_t *)smsg->rdmaOpInfo;
  char *msg = smsg->msg;
  int tag = rdmaOpInfo->tag;
  int dstrank;

#if CMK_MEM_CHECKPOINT || CMK_MESSAGE_LOGGING
  dstrank = petorank[node];
  smsg->dstrank = dstrank;
#else
  dstrank=node;
#endif

  if (MPI_SUCCESS != MPI_Isend((void *)msg, size, MPI_BYTE, dstrank, tag, charmComm, &(smsg->req)))
    CmiAbort("LrtsSendBuffer: MPI_Isend failed!\n");

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
void MPIPostRdmaBuffer(const void *buffer, void *ack, int size, int pe, int tag){

  int destLocalNode = CmiNodeOf(pe);
  int destRank = CmiGetNodeGlobal(destLocalNode, CmiMyPartition());

  SMSG_LIST *msg_tmp = (SMSG_LIST *) malloc(sizeof(SMSG_LIST));

  CmiMPIRzvRdmaOpInfo_t *rdmaOpInfo = (CmiMPIRzvRdmaOpInfo_t *) malloc(sizeof(CmiMPIRzvRdmaOpInfo_t));
  rdmaOpInfo->ack = ack;
  rdmaOpInfo->tag = tag;
  msg_tmp->rdmaOpInfo = rdmaOpInfo;
  msg_tmp->msg = (char *)buffer;
  msg_tmp->destpe = destRank;
  msg_tmp->size = size;
  msg_tmp->next = 0;

#if CMK_SMP
#if MULTI_SENDQUEUE
  PCQueuePush(procState[CmiMyRank()].sendMsgBuf,(char *)msg_tmp);
  return;
#else
  if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
    PCQueuePush(sendMsgBuf,(char *)msg_tmp);
    return;
  }
#endif
#endif //end of CMK_SMP

  MPISendOneRdmaBuffer(msg_tmp);
}
