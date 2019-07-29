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

        if (tmp_pd->first_operand == DIRECT_SEND_RECV) {
          // Call the ack handler function if used for direct api
          CmiInvokeNcpyAck((void *)tmp_pd->second_operand);
        } else {
          CmiAbort("Invalid case!\n");
        }
        free(tmp_pd);
      }
      else if(tmp_pd->type == GNI_POST_RDMA_PUT){

        if (tmp_pd->first_operand == DIRECT_SEND_RECV) {
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
void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfo) {

  if(ncpyOpInfo->isSrcRegistered == 0) {
    // Remote buffer is unregistered, send a message to register it and perform PUT

    // set OpMode for reverse operation
    setReverseModeForNcpyOpInfo(ncpyOpInfo);

#if CMK_SMP
    // send the small message to the other node through the comm thread
    buffer_small_msgs(&smsg_queue, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize,
                          CmiNodeOf(ncpyOpInfo->srcPe),
                          RDMA_REG_AND_PUT_MD_DIRECT_TAG);
#else // non-smp mode

    int msgMode = (ncpyOpInfo->freeMe == CMK_FREE_NCPYOPINFO) ? CHARM_SMSG : SMSG_DONT_FREE;

    // send the small message directly
    gni_return_t status = send_smsg_message(&smsg_queue,
                            CmiNodeOf(ncpyOpInfo->srcPe),
                            ncpyOpInfo,
                            ncpyOpInfo->ncpyOpInfoSize,
                            RDMA_REG_AND_PUT_MD_DIRECT_TAG,
                            0, NULL, msgMode, 1);
#if !CMK_SMSGS_FREE_AFTER_EVENT
    if(status == GNI_RC_SUCCESS && ncpyOpInfo->freeMe == CMK_FREE_NCPYOPINFO) {
      CmiFree(ncpyOpInfo);
    }
#endif // end of !CMK_SMSGS_FREE_AFTER_EVENT
#endif // end of CMK_SMP

  } else {
    // Remote buffer is registered, perform GET

    uint64_t src_addr = (uint64_t)(ncpyOpInfo->srcPtr);
    uint64_t dest_addr = (uint64_t)(ncpyOpInfo->destPtr);
    uint64_t length    = (uint64_t)(ncpyOpInfo->srcSize);

    //check alignment as Rget in GNI requires 4 byte alignment for src_addr, dest_adder and size
    if(((src_addr % 4)==0) && ((dest_addr % 4)==0) && ((length % 4)==0)) {
      // 4-byte aligned, perform GET
#if CMK_SMP
      // send a message to the comm thread, making it do the GET
      buffer_small_msgs(&smsg_queue, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize, CmiNodeOf(ncpyOpInfo->srcPe), RDMA_COMM_PERFORM_GET_TAG);
#else // non-smp mode
      // perform GET directly
      gni_return_t status = post_rdma(
                            src_addr,
                            ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
                            dest_addr,
                            ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
                            std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
                            (uint64_t)ncpyOpInfo,
                            CmiNodeOf(ncpyOpInfo->srcPe),
                            GNI_POST_RDMA_GET,
                            DIRECT_SEND_RECV);
#endif // end of !CMK_SMP

    } else {
      // send all the data to the source to perform a put

      // set OpMode for reverse operation
      setReverseModeForNcpyOpInfo(ncpyOpInfo);

#if CMK_SMP
      // send the small message to the other node through the comm thread
      buffer_small_msgs(&smsg_queue, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize, CmiNodeOf(ncpyOpInfo->srcPe), RDMA_PUT_MD_DIRECT_TAG);
#else // nonsmp mode
      // send the small message directly
      int msgMode = (ncpyOpInfo->freeMe == CMK_FREE_NCPYOPINFO) ? CHARM_SMSG : SMSG_DONT_FREE;

      gni_return_t status = send_smsg_message(&smsg_queue, CmiNodeOf(ncpyOpInfo->srcPe), ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize, RDMA_PUT_MD_DIRECT_TAG, 0, NULL, SMSG_DONT_FREE, 1);
#if !CMK_SMSGS_FREE_AFTER_EVENT
      if(status == GNI_RC_SUCCESS && ncpyOpInfo->freeMe == CMK_FREE_NCPYOPINFO) {
        CmiFree(ncpyOpInfo);
      }
#endif // end of !CMK_SMSGS_FREE_AFTER_EVENT
#endif // end of CMK_SMP
    }
  }
}

// Perform an RDMA Put call into the remote destination address from the local source address
void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfo) {
  if(ncpyOpInfo->destRegMode == CMK_BUFFER_UNREG) {
    // Remote buffer is unregistered, send a message to register it and perform GET

    // send all the data to the source to register and perform a get
#if CMK_SMP
    // send the small message to the other node through the comm thread
    buffer_small_msgs(&smsg_queue, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize,
                      CmiNodeOf(ncpyOpInfo->destPe),
                      RDMA_REG_AND_GET_MD_DIRECT_TAG);
#else // non-smp mode

    // send the small message directly
    gni_return_t status = send_smsg_message(&smsg_queue,
                              CmiNodeOf(ncpyOpInfo->destPe),
                              ncpyOpInfo,
                              ncpyOpInfo->ncpyOpInfoSize,
                              RDMA_REG_AND_GET_MD_DIRECT_TAG,
                              0, NULL, CHARM_SMSG, 1);
#if !CMK_SMSGS_FREE_AFTER_EVENT
    if(status == GNI_RC_SUCCESS) {
      CmiFree(ncpyOpInfo);
    }
#endif // end of !CMK_SMSGS_FREE_AFTER_EVENT
#endif // end of CMK_SMP

  } else {
    // Remote buffer is registered, perform PUT

    uint64_t dest_addr = (uint64_t)(ncpyOpInfo->destPtr);
    uint64_t src_addr = (uint64_t)(ncpyOpInfo->srcPtr);

#if CMK_SMP
    // send a message to the comm thread, making it do the PUT
    buffer_small_msgs(&smsg_queue, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize, CmiNodeOf(ncpyOpInfo->destPe), RDMA_COMM_PERFORM_PUT_TAG);
#else // nonsmp mode
    // perform PUT directly
    gni_return_t status = post_rdma(dest_addr,
                          ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
                          src_addr,
                          ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
                          std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
                          (uint64_t)ncpyOpInfo,
                          CmiNodeOf(ncpyOpInfo->destPe),
                          GNI_POST_RDMA_PUT,
                          DIRECT_SEND_RECV);
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
  if(mode != CMK_BUFFER_PREREG && mode != CMK_BUFFER_NOREG) {
    CmiGNIRzvRdmaPtr_t *destInfo = (CmiGNIRzvRdmaPtr_t *)info;
    deregisterDirectMem(destInfo->mem_hndl, pe);
  }
}

#if CMK_SMP
// Method used by the comm thread to perform GET - called from SendBufferMsg
void _performOneRgetForWorkerThread(MSG_LIST *ptr) {
  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(ptr->msg);
  post_rdma((uint64_t)ncpyOpInfo->srcPtr,
            ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
            (uint64_t)ncpyOpInfo->destPtr,
            ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
            std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
            (uint64_t)ncpyOpInfo,
            ptr->destNode,
            GNI_POST_RDMA_GET,
            DIRECT_SEND_RECV);
}

// Method used by the comm thread to perform PUT - called from SendBufferMsg
void _performOneRputForWorkerThread(MSG_LIST *ptr) {
  NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(ptr->msg);
  post_rdma((uint64_t)ncpyOpInfo->destPtr,
            ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
            (uint64_t)ncpyOpInfo->srcPtr,
            ((CmiGNIRzvRdmaPtr_t *)((char *)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize()))->mem_hndl,
            std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
            (uint64_t)ncpyOpInfo,
            ptr->destNode,
            GNI_POST_RDMA_PUT,
            DIRECT_SEND_RECV);
}
#endif


void LrtsInvokeRemoteDeregAckHandler(int pe, NcpyOperationInfo *ncpyOpInfo) {

  if(ncpyOpInfo->opMode == CMK_BCAST_EM_API)
    return;

  // ncpyOpInfo is a part of the received message and can be freed before this send completes
  // for that reason, it is copied into a new message
  NcpyOperationInfo *newNcpyOpInfo = newNcpyOpInfo = (NcpyOperationInfo *)CmiAlloc(ncpyOpInfo->ncpyOpInfoSize);
  memcpy(newNcpyOpInfo, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize);

  resetNcpyOpInfoPointers(newNcpyOpInfo);
  newNcpyOpInfo->freeMe = CMK_FREE_NCPYOPINFO;

#if CMK_SMP
  // send the small message to the other node through the comm thread
  buffer_small_msgs(&smsg_queue, newNcpyOpInfo, newNcpyOpInfo->ncpyOpInfoSize,
                        CmiNodeOf(pe),
                        RDMA_DEREG_AND_ACK_MD_DIRECT_TAG);
#else // non-smp mode

  // send the small message directly
  gni_return_t status = send_smsg_message(&smsg_queue,
                          CmiNodeOf(pe),
                          newNcpyOpInfo,
                          newNcpyOpInfo->ncpyOpInfoSize,
                          RDMA_DEREG_AND_ACK_MD_DIRECT_TAG,
                          0, NULL, CHARM_SMSG, 1);
#if !CMK_SMSGS_FREE_AFTER_EVENT
  if(status == GNI_RC_SUCCESS && newNcpyOpInfo->freeMe == CMK_FREE_NCPYOPINFO) {
    CmiFree(newNcpyOpInfo);
  }
#endif // end of !CMK_SMSGS_FREE_AFTER_EVENT
#endif // end of CMK_SMP
}
