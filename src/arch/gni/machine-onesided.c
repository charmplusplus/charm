/*
   included in machine.C
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
    int length, uint64_t post_id, int destNode, int type, unsigned short int mode)
{
  gni_return_t            status;
  gni_post_descriptor_t *pd;
  MallocPostDesc(pd);

  // length, local_addr, and remote_addr must be 4-byte aligned.
  pd->type                = (gni_post_type_t)type;
  pd->cq_mode             = GNI_CQMODE_GLOBAL_EVENT;
  pd->dlvr_mode           = GNI_DLVMODE_PERFORMANCE;
  pd->length              = length;
  pd->local_addr          = local_addr;
  pd->local_mem_hndl      = local_mem_hndl;
  pd->remote_addr         = remote_addr;
  pd->remote_mem_hndl     = remote_mem_hndl;
  pd->src_cq_hndl         = rdma_onesided_cqh;
  pd->rdma_mode           = 0;
  pd->amo_cmd             = (gni_fma_cmd_type_t)0;

  switch(mode) {
    case INDIRECT_SEND                :  // Using entry method api
    case DIRECT_SEND_RECV             :  // Using direct api GET or PUT
    case DIRECT_SEND_RECV_UNALIGNED   :  // Using direct api GET,
                                         // which resulted into a PUT because of alignment
                                         pd->first_operand = mode;
                                         break;
    default                           :  CmiAbort("Invalid case\n");
                                         break;
  }

  pd->second_operand     = post_id;
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
        length, opAddress, node, GNI_POST_RDMA_PUT, INDIRECT_SEND);
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
      GNI_RC_CHECK("Error! Exceeded Allowed Pinned Memory Limit! GNI_MemRegister on Receiver Buffer (destination) Failed before GET", status);

      recvOp->local_mem_hndl = local_mem_hndl;
      status = post_rdma(remote_addr, remote_mem_hndl, buffer, local_mem_hndl,
          length, opAddress, CmiNodeOf(pe), GNI_POST_RDMA_GET, INDIRECT_SEND);
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
      GNI_RC_CHECK("Error! Exceeded Allowed Pinned Memory Limit! GNI_MemRegister on Receiver Buffer (destination) Failed before PUT", status);

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

        if(tmp_pd->first_operand == INDIRECT_SEND) {
          // Invoke the method handler if used for indirect api
          CmiGNIRzvRdmaRecvOp_t * recvOpInfo = (CmiGNIRzvRdmaRecvOp_t *)tmp_pd->second_operand;
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
        } else if (tmp_pd->first_operand == DIRECT_SEND_RECV) {
          // Call the ack handler function if used for direct api
          CmiInvokeNcpyAck((void *)tmp_pd->second_operand);
        } else {
          CmiAbort("Invalid case!\n");
        }
        free(tmp_pd);
      }
      else if(tmp_pd->type == GNI_POST_RDMA_PUT){

        if(tmp_pd->first_operand == INDIRECT_SEND) {
          // Invoke the method handler if used for indirect api
          CmiGNIRzvRdmaRecvOp_t * recvOpInfo = (CmiGNIRzvRdmaRecvOp_t *)tmp_pd->second_operand;
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
        } else if (tmp_pd->first_operand == DIRECT_SEND_RECV) {

          // Call the ack handler function if used for direct api
          CmiInvokeNcpyAck((void *)tmp_pd->second_operand);

        } else if (tmp_pd->first_operand == DIRECT_SEND_RECV_UNALIGNED) {

          // Send the metadata back to the original PE which invoked PUT
          // The original PE then calls the ack handler function
          CmiGNIRzvRdmaDirectInfo_t *putOpInfo = (CmiGNIRzvRdmaDirectInfo_t *)tmp_pd->second_operand;

          // send the ref data to the destination
          gni_return_t status = send_smsg_message(&smsg_queue, CmiNodeOf(putOpInfo->destPe), putOpInfo, sizeof(CmiGNIRzvRdmaDirectInfo_t), RDMA_PUT_DONE_DIRECT_TAG, 0, NULL, NONCHARM_SMSG, 1);
#if !CMK_SMSGS_FREE_AFTER_EVENT
          if(status == GNI_RC_SUCCESS) {
            free(putOpInfo);
          }
#endif
        }  else {
          CmiAbort("Invalid case!\n");
        }
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

/* Support for Nocopy Direct API */

// Set the machine specific information for a nocopy pointer
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode){

  gni_mem_handle_t mem_hndl;
  gni_return_t status = GNI_RC_SUCCESS;

  if(mode == CMK_BUFFER_PREREG && SIZEFIELD(ptr) < BIG_MSG) {
    // Allocation for CMK_BUFFER_PREREG happens through CmiAlloc, which is allocated out of a mempool
    if(IsMemHndlZero(GetMemHndl(ptr))) {
      // register it and get the info
      status = registerMemory(GetMempoolBlockPtr(ptr), GetMempoolsize(ptr), &(GetMemHndl(ptr)), NULL);
      if(status == GNI_RC_SUCCESS) {
        // registration successful, get memory handle
        mem_hndl = GetMemHndl(ptr);
      } else {
        GNI_RC_CHECK("Error! Memory registration failed!", status);
      }
    }
    else {
      // get the handle
      mem_hndl = GetMemHndl(ptr);
    }
  } else {
    status = GNI_MemRegister(nic_hndl, (uint64_t)ptr, (uint64_t)size, NULL, GNI_MEM_READWRITE, -1, &mem_hndl);
    GNI_RC_CHECK("Error! Memory registration failed!", status);
  }
  CmiGNIRzvRdmaPtr_t *rdmaSrc = (CmiGNIRzvRdmaPtr_t *)info;
  rdmaSrc->mem_hndl = mem_hndl;
}

// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  int size) {

  // Remote buffer is registered, perform GET
  CmiAssert(srcAckSize == destAckSize);

  // Register local buffer if it is not registered
  if(*destMode == CMK_BUFFER_UNREG) {
    ((CmiGNIRzvRdmaPtr_t *)destInfo)->mem_hndl = registerDirectMem(destAddr, size, GNI_MEM_READWRITE);
    *destMode = CMK_BUFFER_REG;
  }

  if(*srcMode == CMK_BUFFER_UNREG) {
    // Remote buffer is unregistered, send a message to register it and perform PUT

    int mdMsgSize = sizeof(CmiGNIRzvRdmaReverseOp_t) + destAckSize + srcAckSize;
    CmiGNIRzvRdmaReverseOp_t *regAndPutMsg = (CmiGNIRzvRdmaReverseOp_t *)malloc(mdMsgSize);

    regAndPutMsg->destAddr      = destAddr;
    regAndPutMsg->destPe        = destPe;
    regAndPutMsg->rem_mem_hndl  = ((CmiGNIRzvRdmaPtr_t *)destInfo)->mem_hndl;
    regAndPutMsg->destMode      = *destMode;

    regAndPutMsg->srcAddr       = srcAddr;
    regAndPutMsg->srcPe         = srcPe;
    regAndPutMsg->srcMode       = *srcMode;

    regAndPutMsg->ackSize       = destAckSize;
    regAndPutMsg->size          = size;

    memcpy((char*)regAndPutMsg + sizeof(CmiGNIRzvRdmaReverseOp_t), destAck, destAckSize);
    memcpy((char*)regAndPutMsg + sizeof(CmiGNIRzvRdmaReverseOp_t) + srcAckSize, srcAck, srcAckSize);

    // send all the data to the source to register and perform a put
#if CMK_SMP
    // send the small message to the other node through the comm thread
    buffer_small_msgs(&smsg_queue, regAndPutMsg, mdMsgSize, CmiNodeOf(srcPe), RDMA_REG_AND_PUT_MD_DIRECT_TAG);
#else // non-smp mode
    // send the small message directly
    gni_return_t status = send_smsg_message(&smsg_queue, CmiNodeOf(srcPe), regAndPutMsg, mdMsgSize, RDMA_REG_AND_PUT_MD_DIRECT_TAG, 0, NULL, NONCHARM_SMSG, 1);
    GNI_RC_CHECK("Sending REG & PUT metadata msg failed!", status);
#if !CMK_SMSGS_FREE_AFTER_EVENT
    if(status == GNI_RC_SUCCESS) {
      free(regAndPutMsg);
    }
#endif // end of !CMK_SMSGS_FREE_AFTER_EVENT
#endif // end of CMK_SMP

  } else {
    void *ref = CmiGetNcpyAck(srcAddr, srcAck, srcPe, destAddr, destAck, destPe, srcAckSize);

    uint64_t src_addr = (uint64_t)srcAddr;
    uint64_t dest_addr = (uint64_t)destAddr;
    uint64_t ref_addr = (uint64_t)ref;

    CmiGNIRzvRdmaPtr_t *src_info = (CmiGNIRzvRdmaPtr_t *)srcInfo;
    CmiGNIRzvRdmaPtr_t *dest_info = (CmiGNIRzvRdmaPtr_t *)destInfo;

    //check alignment as Rget in GNI requires 4 byte alignment for src_addr, dest_adder and size
    if(((src_addr % 4)==0) && ((dest_addr % 4)==0) && ((size % 4)==0)) {
      // 4-byte aligned, perform GET
#if CMK_SMP
      // send a message to the comm thread, making it do the GET
      CmiGNIRzvRdmaDirectInfo_t *getOpInfo = (CmiGNIRzvRdmaDirectInfo_t *)malloc(sizeof(CmiGNIRzvRdmaDirectInfo_t));
      getOpInfo->dest_mem_hndl = dest_info->mem_hndl;
      getOpInfo->dest_addr = dest_addr;
      getOpInfo->src_mem_hndl = src_info->mem_hndl;
      getOpInfo->src_addr = src_addr;
      getOpInfo->destPe = CmiMyPe();
      getOpInfo->size = size;
      getOpInfo->ref = ref_addr;

      buffer_small_msgs(&smsg_queue, getOpInfo, sizeof(CmiGNIRzvRdmaDirectInfo_t), CmiNodeOf(srcPe), RDMA_COMM_PERFORM_GET_TAG);
#else // non-smp mode
      // perform GET directly
      gni_return_t status = post_rdma(src_addr, src_info->mem_hndl, dest_addr, dest_info->mem_hndl,
          size, ref_addr, CmiNodeOf(srcPe), GNI_POST_RDMA_GET, DIRECT_SEND_RECV);
#endif // end of !CMK_SMP

    } else {
      // not 4-byte aligned, send md for performing PUT from other node
      // Allocate machine specific metadata
      CmiGNIRzvRdmaDirectInfo_t *putOpInfo = (CmiGNIRzvRdmaDirectInfo_t *)malloc(sizeof(CmiGNIRzvRdmaDirectInfo_t));
      putOpInfo->dest_mem_hndl = dest_info->mem_hndl;
      putOpInfo->dest_addr = dest_addr;
      putOpInfo->src_mem_hndl = src_info->mem_hndl;
      putOpInfo->src_addr = src_addr;
      putOpInfo->destPe = CmiMyPe();
      putOpInfo->size = size;
      putOpInfo->ref = ref_addr;

      // send all the data to the source to perform a put
#if CMK_SMP
      // send the small message to the other node through the comm thread
      buffer_small_msgs(&smsg_queue, putOpInfo, sizeof(CmiGNIRzvRdmaDirectInfo_t), CmiNodeOf(srcPe), RDMA_PUT_MD_DIRECT_TAG);
#else // nonsmp mode
      // send the small message directly
      gni_return_t status = send_smsg_message(&smsg_queue, CmiNodeOf(srcPe), putOpInfo, sizeof(CmiGNIRzvRdmaDirectInfo_t), RDMA_PUT_MD_DIRECT_TAG, 0, NULL, NONCHARM_SMSG, 1);
      GNI_RC_CHECK("Sending PUT metadata msg failed!", status);
#if !CMK_SMSGS_FREE_AFTER_EVENT
      if(status == GNI_RC_SUCCESS) {
        free(putOpInfo);
      }
#endif // end of !CMK_SMSGS_FREE_AFTER_EVENT
#endif // end of CMK_SMP
    }
  }
}

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(
  const void* destAddr,
  void *destInfo,
  void *destAck,
  int destAckSize,
  int destPe,
  unsigned short int *destMode,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  unsigned short int *srcMode,
  int size) {

  // Register local buffer if it is not registered
  if(*srcMode == CMK_BUFFER_UNREG) {
    ((CmiGNIRzvRdmaPtr_t *)srcInfo)->mem_hndl = registerDirectMem(srcAddr, size, GNI_MEM_READ_ONLY);
    *srcMode = CMK_BUFFER_REG;
  }

  if(*destMode == CMK_BUFFER_UNREG) {
    // Remote buffer is unregistered, send a message to register it and perform GET

    int mdMsgSize = sizeof(CmiGNIRzvRdmaReverseOp_t) + destAckSize + srcAckSize;
    CmiGNIRzvRdmaReverseOp_t *regAndGetMsg = (CmiGNIRzvRdmaReverseOp_t *)malloc(mdMsgSize);

    regAndGetMsg->srcAddr       = srcAddr;
    regAndGetMsg->srcPe         = srcPe;
    regAndGetMsg->rem_mem_hndl  = ((CmiGNIRzvRdmaPtr_t *)srcInfo)->mem_hndl;
    regAndGetMsg->srcMode       = *srcMode;

    regAndGetMsg->destAddr      = destAddr;
    regAndGetMsg->destPe        = destPe;
    regAndGetMsg->destMode      = *destMode;

    regAndGetMsg->ackSize       = srcAckSize;
    regAndGetMsg->size          = size;

    memcpy((char*)regAndGetMsg + sizeof(CmiGNIRzvRdmaReverseOp_t), destAck, destAckSize);
    memcpy((char*)regAndGetMsg + sizeof(CmiGNIRzvRdmaReverseOp_t) + srcAckSize, srcAck, srcAckSize);

    // send all the data to the source to register and perform a get
#if CMK_SMP
    // send the small message to the other node through the comm thread
    buffer_small_msgs(&smsg_queue, regAndGetMsg, mdMsgSize, CmiNodeOf(destPe), RDMA_REG_AND_GET_MD_DIRECT_TAG);
#else // non-smp mode
    // send the small message directly
    gni_return_t status = send_smsg_message(&smsg_queue, CmiNodeOf(destPe), regAndGetMsg, mdMsgSize, RDMA_REG_AND_GET_MD_DIRECT_TAG, 0, NULL, NONCHARM_SMSG, 1);
    GNI_RC_CHECK("Sending REG & GET metadata msg failed!", status);
#if !CMK_SMSGS_FREE_AFTER_EVENT
    if(status == GNI_RC_SUCCESS) {
      free(regAndGetMsg);
    }
#endif // end of !CMK_SMSGS_FREE_AFTER_EVENT
#endif // end of CMK_SMP

  } else {

    // Remote buffer is registered, perform PUT
    CmiAssert(srcAckSize == destAckSize);
    void *ref = CmiGetNcpyAck(srcAddr, srcAck, srcPe, destAddr, destAck, destPe, srcAckSize);

    uint64_t dest_addr = (uint64_t)destAddr;
    uint64_t src_addr = (uint64_t)srcAddr;
    uint64_t ref_addr = (uint64_t)ref;

    CmiGNIRzvRdmaPtr_t *dest_info = (CmiGNIRzvRdmaPtr_t *)destInfo;
    CmiGNIRzvRdmaPtr_t *src_info = (CmiGNIRzvRdmaPtr_t *)srcInfo;

#if CMK_SMP
    // send a message to the comm thread, making it do the PUT
    CmiGNIRzvRdmaDirectInfo_t *putOpInfo = (CmiGNIRzvRdmaDirectInfo_t *)malloc(sizeof(CmiGNIRzvRdmaDirectInfo_t));
    putOpInfo->dest_mem_hndl = dest_info->mem_hndl;
    putOpInfo->dest_addr = dest_addr;
    putOpInfo->src_mem_hndl = src_info->mem_hndl;
    putOpInfo->src_addr = src_addr;
    putOpInfo->destPe = CmiMyPe();
    putOpInfo->size = size;
    putOpInfo->ref = ref_addr;

    buffer_small_msgs(&smsg_queue, putOpInfo, sizeof(CmiGNIRzvRdmaDirectInfo_t), CmiNodeOf(destPe), RDMA_COMM_PERFORM_PUT_TAG);
#else // nonsmp mode
    // perform PUT directly
    gni_return_t status = post_rdma(dest_addr, dest_info->mem_hndl, src_addr, src_info->mem_hndl,
        size, ref_addr, CmiNodeOf(destPe), GNI_POST_RDMA_PUT, DIRECT_SEND_RECV);
#endif // end of !CMK_SMP
  }
}

// Register memory and return mem_hndl
gni_mem_handle_t registerDirectMem(const void *ptr, int size, unsigned short int mode) {
  CmiAssert(mode == GNI_MEM_READWRITE || mode == GNI_MEM_READ_ONLY);
  gni_mem_handle_t mem_hndl;
  gni_return_t status = GNI_RC_SUCCESS;
  status = GNI_MemRegister(nic_hndl, (uint64_t)ptr, (uint64_t)size, NULL, mode, -1, &mem_hndl);
  GNI_RC_CHECK("Error! Memory registration failed inside LrtsRegisterMemory!", status);
  return mem_hndl;
}

// Deregister memory mem_hndl
void deregisterDirectMem(gni_mem_handle_t mem_hndl, int pe) {
  gni_return_t status = GNI_RC_SUCCESS;
  status = GNI_MemDeregister(nic_hndl, &mem_hndl);
  GNI_RC_CHECK("GNI_MemDeregister failed!", status);
}

// Method invoked to deregister memory handle
void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode){
  if(mode == CMK_BUFFER_REG || (mode == CMK_BUFFER_PREREG && SIZEFIELD(ptr) >= BIG_MSG)) {
    CmiGNIRzvRdmaPtr_t *destInfo = (CmiGNIRzvRdmaPtr_t *)info;
    deregisterDirectMem(destInfo->mem_hndl, pe);
  }
}

#if CMK_SMP
// Method used by the comm thread to perform GET - called from SendBufferMsg
void _performOneRgetForWorkerThread(MSG_LIST *ptr) {
  CmiGNIRzvRdmaDirectInfo_t *getOpInfo = (CmiGNIRzvRdmaDirectInfo_t *)(ptr->msg);
  post_rdma(getOpInfo->src_addr,
            getOpInfo->src_mem_hndl,
            getOpInfo->dest_addr,
            getOpInfo->dest_mem_hndl,
            getOpInfo->size,
            getOpInfo->ref,
            ptr->destNode,
            GNI_POST_RDMA_GET,
            DIRECT_SEND_RECV);
  free(getOpInfo);
}

// Method used by the comm thread to perform PUT - called from SendBufferMsg
void _performOneRputForWorkerThread(MSG_LIST *ptr) {
  CmiGNIRzvRdmaDirectInfo_t *putOpInfo = (CmiGNIRzvRdmaDirectInfo_t *)(ptr->msg);
  post_rdma(putOpInfo->dest_addr,
            putOpInfo->dest_mem_hndl,
            putOpInfo->src_addr,
            putOpInfo->src_mem_hndl,
            putOpInfo->size,
            putOpInfo->ref,
            ptr->destNode,
            GNI_POST_RDMA_PUT,
            DIRECT_SEND_RECV);
  free(putOpInfo);
}
#endif
