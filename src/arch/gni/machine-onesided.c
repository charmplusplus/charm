/*
   included in machine.c
   Justin Miron
 */

#define DEBUG_ONE_SIDED 0

#if DEBUG_ONE_SIDED
#define MACH_DEBUG(x) x
#else
#define MACH_DEBUG(x)
#endif

void _initOnesided()
{
  gni_return_t status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &rdma_onesided_cqh);
  GNI_RC_CHECK("GNI_CqCreateRDMA", status);

#if MULTI_THREAD_SEND
  rdma_onesided_cq_lock = CmiCreateLock();
#endif
  MACH_DEBUG(CmiPrintf("[%d]_initOneSided: Initialized CQ and lock\n", CmiMyPe()));
}

int checkFourByteAligned(void *recv){
  CmiGNIRzvRdmaRecv_t* recvInfo = (CmiGNIRzvRdmaRecv_t*)recv;
  int i, size;
  for(i = 0; i < recvInfo->numOps; ++i)
  {
    CmiGNIRzvRdmaRecvOp_t * recvOp = &recvInfo->rdmaOp[i];
    uint64_t remote_addr = recvOp->remote_addr;
    uint64_t local_addr = recvOp->local_addr;
    int length = recvOp->size;
    if(((local_addr % 4)==0) && ((remote_addr % 4)==0) && ((length % 4)==0))
      continue;
    MACH_DEBUG(CmiPrintf("[%d][%d][%d] Unaligned, should use PUT\n", CmiMyPe(), CmiMyNode(), CmiMyRank()));
    return 0;
  }
  MACH_DEBUG(CmiPrintf("[%d][%d][%d] Aligned, should use GET\n", CmiMyPe(), CmiMyNode(), CmiMyRank()));
  return 1;
}

void rdma_sendMdBackForPut( CmiGNIRzvRdmaRecv_t* recvInfo, int src_pe){
  int size = LrtsGetRdmaRecvInfoSize(recvInfo->numOps);
  send_smsg_message(&smsg_queue, CmiNodeOf(src_pe), recvInfo, size, RDMA_PUT_MD_TAG, 0, NULL, NONCHARM_SMSG_DONT_FREE, 0);
  MACH_DEBUG(CmiPrintf("[%d]rdma_sendMdBackForPut: Sent md back to %d for PUT\n", CmiMyPe(), src_pe));
}

void  rdma_sendAck (CmiGNIRzvRdmaRecvOp_t* recvOpInfo, int src_pe)
{
  CmiGNIAckOp_t *ack_data = (CmiGNIAckOp_t *)malloc(sizeof(CmiGNIAckOp_t));
  ack_data->ack = (CmiRdmaAck *)recvOpInfo->src_info;
  ack_data->mem_hndl = recvOpInfo->remote_mem_hndl;
  MACH_DEBUG(CmiPrintf("[%d]rdma_sendAck: Sent rdma ack to %d\n", CmiMyPe(), src_pe));
  gni_return_t status = send_smsg_message(&smsg_queue, CmiNodeOf(src_pe), ack_data, sizeof(CmiGNIAckOp_t), RDMA_ACK_TAG, 0, NULL, NONCHARM_SMSG, 1);
#if !CMK_SMSGS_FREE_AFTER_EVENT
  if(status == GNI_RC_SUCCESS) {
    free(ack_data);
  }
#endif
}

void  rdma_sendMsgForPutCompletion (CmiGNIRzvRdmaRecv_t* recvInfo, int destNode)
{
  int size = LrtsGetRdmaRecvInfoSize(recvInfo->numOps);
  gni_return_t status = send_smsg_message(&smsg_queue, destNode, recvInfo, size, RDMA_PUT_DONE_TAG, 0, NULL, NONCHARM_SMSG, 1);
#if !CMK_SMSGS_FREE_AFTER_EVENT
  if(status == GNI_RC_SUCCESS) {
    free(recvInfo);
  }
#endif
  MACH_DEBUG(CmiPrintf("[%d]rdma_sendMsgForPutCompletion: Sent md back to node:%d to indicate PUT completion\n", CmiMyPe(), destNode));
}

gni_return_t post_rdma(uint64_t remote_addr, gni_mem_handle_t remote_mem_hndl,
    uint64_t local_addr, gni_mem_handle_t local_mem_hndl,
    int length, uint64_t post_id, int destNode, int type)
{
  gni_return_t            status;
  gni_post_descriptor_t *pd;
  MallocPostDesc(pd);

  // length, local_addr, and remote_addr must be 4-byte aligned.
  pd->type                = type;
  pd->cq_mode             = GNI_CQMODE_GLOBAL_EVENT;
  pd->dlvr_mode           = GNI_DLVMODE_PERFORMANCE;
  pd->length              = length;
  pd->local_addr          = local_addr;
  pd->local_mem_hndl      = local_mem_hndl;
  pd->remote_addr         = remote_addr;
  pd->remote_mem_hndl     = remote_mem_hndl;
  pd->src_cq_hndl         = rdma_onesided_cqh;
  pd->rdma_mode           = 0;
  pd->amo_cmd             = 0;
  pd->first_operand       = post_id;

  MACH_DEBUG(CmiPrintf("[%d]post_rdma, local_addr: %p, remote_addr: %p, length: %d\n", 
        CmiMyPe(), (void *)local_addr, (void *)remote_addr, length));
  //local_addr, remote_addr and length must be 4-byte aligned if called with GET
  if(type == GNI_POST_RDMA_GET)
    CmiEnforce(((local_addr % 4) == 0) && ((remote_addr) % 4 == 0));

  CMI_GNI_LOCK(rdma_onesided_cq_lock);
  status = GNI_PostRdma(ep_hndl_array[destNode], pd);
  CMI_GNI_UNLOCK(rdma_onesided_cq_lock);
  GNI_RC_CHECK("PostRdma", status);

  return status;
}

void LrtsIssueRputs(void *recv, int node)
{
  CmiGNIRzvRdmaRecv_t* recvInfo = (CmiGNIRzvRdmaRecv_t *)recv;
  gni_return_t status;
  int i;

  MACH_DEBUG(CmiPrintf("Started LrtsIssueRputs, Issued from %d to node:%d\n", CmiMyPe(), node));

  for(i = 0; i < recvInfo->numOps; ++i){
    CmiGNIRzvRdmaRecvOp_t *recvOp = &recvInfo->rdmaOp[i];
    gni_mem_handle_t remote_mem_hndl = recvOp->remote_mem_hndl;
    gni_mem_handle_t local_mem_hndl = recvOp->local_mem_hndl;
    uint64_t remote_addr = recvOp->remote_addr;
    uint64_t buffer = recvOp->local_addr;
    int length = recvOp->size;

    uint64_t opAddress = (uint64_t)(recvOp);

    status = post_rdma(remote_addr, remote_mem_hndl, buffer, local_mem_hndl,
        length, opAddress, node, GNI_POST_RDMA_PUT);
  }
}

void LrtsIssueRgets(void *recv, int pe)
{
  CmiGNIRzvRdmaRecv_t* recvInfo = (CmiGNIRzvRdmaRecv_t*)recv;
  gni_return_t status;
  int i;

  MACH_DEBUG(CmiPrintf("Started LrtsIssueRgets, Issued from %d to %d\n", CmiMyPe(), pe));

  if(checkFourByteAligned(recv)){
    for(i = 0; i < recvInfo->numOps; ++i)
    {
      CmiGNIRzvRdmaRecvOp_t * recvOp = &recvInfo->rdmaOp[i];
      gni_mem_handle_t remote_mem_hndl = recvOp->remote_mem_hndl;
      uint64_t remote_addr = recvOp->remote_addr;
      uint64_t buffer = recvOp->local_addr;
      int length = recvOp->size;
      uint64_t opAddress = (uint64_t)(recvOp);

      /* Register the local buffer with the NIC */
      gni_mem_handle_t local_mem_hndl;

      status = GNI_MemRegister(nic_hndl, buffer, length, NULL,  GNI_MEM_READWRITE, -1, &local_mem_hndl);
      GNI_RC_CHECK("Error! Exceeded Allowed Pinned Memory Limit! GNI_MemRegister on Receiver Buffer (target) Failed before GET", status);

      recvOp->local_mem_hndl = local_mem_hndl;
      status = post_rdma(remote_addr, remote_mem_hndl, buffer, local_mem_hndl,
          length, opAddress, CmiNodeOf(pe), GNI_POST_RDMA_GET);
    }
  }
  //use RPUT because of 4-byte alignment not being conformed
  else{

    //send metadata message back to the sender for performing RPUT
    for(i = 0; i < recvInfo->numOps; ++i)
    {
      CmiGNIRzvRdmaRecvOp_t * recvOp = &recvInfo->rdmaOp[i];
      uint64_t remote_addr = recvOp->remote_addr;
      uint64_t buffer = recvOp->local_addr;
      int length = recvOp->size;

      /* Register the local buffer with the NIC */
      gni_mem_handle_t local_mem_hndl;
      status = GNI_MemRegister(nic_hndl, buffer, length, NULL,  GNI_MEM_READWRITE, -1, &local_mem_hndl);
      GNI_RC_CHECK("Error! Exceeded Allowed Pinned Memory Limit! GNI_MemRegister on Receiver Buffer (target) Failed before PUT", status);

      //Switch local and remote handles and buffers as recvInfo will be sent to the sender for PUT
      recvOp->local_mem_hndl = recvOp->remote_mem_hndl;
      recvOp->remote_mem_hndl = local_mem_hndl;
      recvOp->local_addr = remote_addr;
      recvOp->remote_addr = buffer;
    }

    //send metadata message to receiver to perform a PUT
    rdma_sendMdBackForPut(recvInfo, pe);
  }
}

/*
 * This code largely overlaps with code in PumpLocalTransactions,
 * any changes to PumpLocalTransactions should be reflected here
 */
void PumpOneSidedRDMATransactions(gni_cq_handle_t rdma_cq, CmiNodeLock rdma_cq_lock)
{
  gni_cq_entry_t ev;
  gni_return_t status;
  uint64_t type, inst_id;

  while(1)
  {
    CMI_GNI_LOCK(rdma_cq_lock);
    status = GNI_CqGetEvent(rdma_cq, &ev);
    CMI_GNI_UNLOCK(rdma_cq_lock);
    if(status != GNI_RC_SUCCESS)
    {
      break;
    }

    MACH_DEBUG(CmiPrintf("[%d]PumpOneSidedTransaction: Received GET completion event.\n", CmiMyPe()));
    type = GNI_CQ_GET_TYPE(ev);
    if (type == GNI_CQ_EVENT_TYPE_POST){

      gni_post_descriptor_t   *tmp_pd;
      CMI_GNI_LOCK(rdma_cq_lock);
      status = GNI_GetCompleted(rdma_cq, ev, &tmp_pd);
      CMI_GNI_UNLOCK(rdma_cq_lock);
      GNI_RC_CHECK("GNI_GetCompleted", status);

      if(tmp_pd->type == GNI_POST_RDMA_GET){
        CmiGNIRzvRdmaRecvOp_t * recvOpInfo = (CmiGNIRzvRdmaRecvOp_t *)tmp_pd->first_operand;
        CmiGNIRzvRdmaRecv_t * recvInfo = (CmiGNIRzvRdmaRecv_t *)((char *)recvOpInfo
            - recvOpInfo->opIndex * sizeof(CmiGNIRzvRdmaRecvOp_t)
            - sizeof(CmiGNIRzvRdmaRecv_t));

        // Deregister registered receiver memory used for GET
        status = GNI_MemDeregister(nic_hndl, &(recvOpInfo->local_mem_hndl));
        GNI_RC_CHECK("GNI_MemDeregister on Receiver for GET operation", status);

        rdma_sendAck(recvOpInfo, recvInfo->srcPE);
        recvInfo->comOps++;
        if(recvInfo->comOps == recvInfo->numOps)
        {
          char * msg = (char *)recvInfo->msg;
          int msg_size = CmiGetMsgSize(msg);
          handleOneRecvedMsg(msg_size, msg);
          MACH_DEBUG(CmiPrintf("[%d]PumpOneSidedTransaction: Final Ack sent to %d\n", CmiMyPe(), CMI_DEST_RANK(msg)));

        }
        free(tmp_pd);
      }
      else if(tmp_pd->type == GNI_POST_RDMA_PUT){
        CmiGNIRzvRdmaRecvOp_t * recvOpInfo = (CmiGNIRzvRdmaRecvOp_t *)tmp_pd->first_operand;
        CmiGNIRzvRdmaRecv_t * recvInfo = (CmiGNIRzvRdmaRecv_t *)((char *)recvOpInfo
            - recvOpInfo->opIndex * sizeof(CmiGNIRzvRdmaRecvOp_t)
            - sizeof(CmiGNIRzvRdmaRecv_t));
        recvInfo->comOps++;
        // Deregister registered sender memory used for PUT
        status = GNI_MemDeregister(nic_hndl, &(recvOpInfo->local_mem_hndl));
        GNI_RC_CHECK("GNI_MemDeregister on Sender for PUT operation", status);

        if(recvInfo->comOps == recvInfo->numOps)
        {
          // send message to the receiver to signal PUT completion so that
          // the receiver can call the message handler
          rdma_sendMsgForPutCompletion(recvInfo, recvInfo->destNode);
        }
        //call ack on the sender
        CmiRdmaAck* ack = (CmiRdmaAck *)(recvOpInfo->src_info);
        ack->fnPtr(ack->token);
        //free callback structure, CmiRdmaAck allocated in CmiSetRdmaAck
        free(ack);

        free(tmp_pd);
      }
      else{
        CmiAbort("Invalid POST type");
      }
    }
    else
    {
      CmiAbort("Received message on onesided rdma queue that wasn't a POSTRdma");
    }
  }
}
