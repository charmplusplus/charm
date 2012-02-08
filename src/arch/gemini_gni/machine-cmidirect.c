/** @file
 * uGNI cmiDirect communication
 * @ingroup Machine
*/

/*
  included in machine.c
  Yanhua Sun, 2/5/2012
*/

//#define     CMI_DIRECT_DEBUG    0
#include "cmidirect.h"

static void printHandle(CmiDirectUserHandle *userHandle, char *s)
{
    CmiPrintf( "[%d]%s(%p)(%p,%p,%p)==>(%p,%p,%p)(%d)(%p,%p)\n", CmiMyPe(), s, userHandle, userHandle->localBuf, userHandle->localMdh.qword1, userHandle->localMdh.qword2, 
        userHandle->remoteBuf, userHandle->remoteMdh.qword1, userHandle->remoteMdh.qword2, userHandle->transSize, userHandle->callbackFnPtr, userHandle->callbackData );
}

/**
 To be called on the receiver to create a handle and return its number
**/
CmiDirectUserHandle CmiDirect_createHandle(int localNode,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData,double initialValue) {

    gni_return_t            status = GNI_RC_SUCCESS;
    CmiDirectUserHandle userHandle;
    userHandle.handle=1; 
    userHandle.localNode=localNode;
    userHandle.remoteNode=_Cmi_mynode;
    userHandle.transSize=recvBufSize;
    userHandle.remoteBuf=recvBuf;
    userHandle.initialValue=initialValue;
    userHandle.callbackFnPtr=callbackFnPtr;
    userHandle.callbackData=callbackData;

    if(recvBufSize <= SMSG_MAX_MSG)
    {
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, userHandle.remoteBuf, recvBufSize, &(userHandle.remoteMdh), &omdh);
    }
    else if(IsMemHndlZero((GetMemHndl(userHandle.remoteBuf)))){
        status = registerMempool(userHandle.remoteBuf);
        userHandle.remoteMdh = GetMemHndl(userHandle.remoteBuf);
    } else
        userHandle.remoteMdh = GetMemHndl(userHandle.remoteBuf);
    if(status != GNI_RC_SUCCESS) {
        userHandle.remoteMdh.qword1 = 0;
        userHandle.remoteMdh.qword2 = 0;
    }

#if CMI_DIRECT_DEBUG
    printHandle(userHandle, "Create Handler");
#endif
    return userHandle;
}

/****
 To be called on the local to attach the local's buffer to this handle
******/

void CmiDirect_assocLocalBuffer(CmiDirectUserHandle *userHandle,void *sendBuf,int sendBufSize) {

    /* one-sided primitives would require registration of memory */
    gni_return_t            status = GNI_RC_SUCCESS;
    
    userHandle->localBuf=sendBuf;
    if(userHandle->transSize <= SMSG_MAX_MSG)
    {
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, userHandle->localBuf, userHandle->transSize, &userHandle->localMdh, &omdh);
    }
    else if(IsMemHndlZero((GetMemHndl(userHandle->localBuf)))){
        status = registerMempool(userHandle->localBuf);
        userHandle->localMdh = GetMemHndl(userHandle->localBuf);
    } else
        userHandle->localMdh = GetMemHndl(userHandle->localBuf);
   
    if(status != GNI_RC_SUCCESS) {
        userHandle->localMdh.qword1 = 0;
        userHandle->localMdh.qword2 = 0;
    }

#if CMI_DIRECT_DEBUG
    printHandle(userHandle, "Associate Handler");
#endif
}

/****
To be called on the local to do the actual data transfer
******/
void CmiDirect_put(CmiDirectUserHandle *userHandle) {

    gni_post_descriptor_t *pd;

#if USE_LRTS_MEMPOOL
    if (userHandle->remoteNode== CmiMyNode()) {
        CmiMemcpy(userHandle->remoteBuf,userHandle->localBuf,userHandle->transSize);
        (*(userHandle->callbackFnPtr))(userHandle->callbackData);
    } else {
        gni_return_t status;
        RDMA_REQUEST        *rdma_request_msg;
        MallocPostDesc(pd);
        if(userHandle->transSize <= LRTS_GNI_RDMA_THRESHOLD)
            pd->type            = GNI_POST_FMA_PUT;
        else
            pd->type            = GNI_POST_RDMA_PUT;
        pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT;
        pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
        pd->length          = userHandle->transSize;
        pd->local_addr      = (uint64_t) (userHandle->localBuf);
        pd->local_mem_hndl  = userHandle->localMdh; 
        pd->remote_addr     = (uint64_t)(userHandle->remoteBuf);
        pd->remote_mem_hndl = userHandle->remoteMdh;
        pd->src_cq_hndl     = 0;
        pd->rdma_mode       = 0;
        pd->first_operand   = (uint64_t) (userHandle->callbackFnPtr);
        pd->second_operand  = (uint64_t) (userHandle->callbackData);
        pd->amo_cmd         = 1;
        if(pd->type == GNI_POST_RDMA_PUT) 
            status = GNI_PostRdma(ep_hndl_array[userHandle->remoteNode], pd);
        else
            status = GNI_PostFma(ep_hndl_array[userHandle->remoteNode],  pd);
        if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
        {
            MallocRdmaRequest(rdma_request_msg);
            rdma_request_msg->destNode = userHandle->remoteNode;
            rdma_request_msg->pd = pd;
#if CMK_SMP
            PCQueuePush(sendRdmaBuf, (char*)rdma_request_msg);
#else
            if(sendRdmaBuf == 0)
            {
                sendRdmaBuf = sendRdmaTail = rdma_request_msg;
            }else{
                sendRdmaTail->next = rdma_request_msg;
                sendRdmaTail =  rdma_request_msg;
            }
#endif
        }else
            GNI_RC_CHECK("CMI_Direct_PUT", status);
    }
#else
    CmiPrintf("Normal Send in CmiDirect Put\n");
    CmiAbort("");
#endif

#if CMI_DIRECT_DEBUG
    printHandle(userHandle, "After Direct_put");
    CmiPrintf("[%d] RDMA put %d,%d bytes addr %p to remoteNode %d:%p \n\n",CmiMyPe(), userHandle->transSize, pd->length, (void*)(pd->local_addr), userHandle->remoteNode, (void*) (pd->remote_addr));
#endif

}

// needs to figure out what is local/remote
void CmiDirect_get(CmiDirectUserHandle *userHandle) {

    gni_post_descriptor_t *pd;

#if USE_LRTS_MEMPOOL
    if (userHandle->remoteNode== CmiMyNode()) {
        CmiMemcpy(userHandle->remoteBuf,userHandle->localBuf,userHandle->transSize);
        (*(userHandle->callbackFnPtr))(userHandle->callbackData);
    } else {
        gni_return_t status;
        RDMA_REQUEST        *rdma_request_msg;
        MallocPostDesc(pd);
        if(userHandle->transSize <= LRTS_GNI_RDMA_THRESHOLD)
            pd->type            = GNI_POST_FMA_GET;
        else
            pd->type            = GNI_POST_RDMA_GET;
        pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT;
        pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
        pd->length          = userHandle->transSize;
        pd->local_addr      = (uint64_t) (userHandle->localBuf);
        pd->local_mem_hndl  = userHandle->localMdh; 
        pd->remote_addr     = (uint64_t)(userHandle->remoteBuf);
        pd->remote_mem_hndl = userHandle->remoteMdh;
        pd->src_cq_hndl     = 0;
        pd->rdma_mode       = 0;
        pd->first_operand   = (uint64_t) (userHandle->callbackFnPtr);
        pd->second_operand  = (uint64_t) (userHandle->callbackData);
        pd->amo_cmd         = 2;
        if(pd->type == GNI_POST_RDMA_GET) 
            status = GNI_PostRdma(ep_hndl_array[userHandle->remoteNode], pd);
        else
            status = GNI_PostFma(ep_hndl_array[userHandle->remoteNode],  pd);
        if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
        {
            MallocRdmaRequest(rdma_request_msg);
            rdma_request_msg->destNode = userHandle->remoteNode;
            rdma_request_msg->pd = pd;
#if CMK_SMP
            PCQueuePush(sendRdmaBuf, (char*)rdma_request_msg);
#else
            if(sendRdmaBuf == 0)
            {
                sendRdmaBuf = sendRdmaTail = rdma_request_msg;
            }else{
                sendRdmaTail->next = rdma_request_msg;
                sendRdmaTail =  rdma_request_msg;
            }
#endif
        }else
            GNI_RC_CHECK("CMI_Direct_GET", status);
    }
#else
    CmiPrintf("Normal Send in CmiDirect Get\n");
    CmiAbort("");
#endif

#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA get %d,%d bytes addr %p to remoteNode %d:%p \n\n",CmiMyPe(), userHandle->transSize, pd->length, (void*)(pd->local_addr), userHandle->remoteNode, (void*) (pd->remote_addr));
#endif


}

/**** up to the user to safely call this */
void CmiDirect_deassocLocalBuffer(CmiDirectUserHandle *userHandle) {


}

/**** up to the user to safely call this */
void CmiDirect_destroyHandle(CmiDirectUserHandle *userHandle) {
    free(userHandle);
}

/**** Should not be called the first time *********/
void CmiDirect_ready(CmiDirectUserHandle *userHandle) {
}

/**** Should not be called the first time *********/
void CmiDirect_readyPollQ(CmiDirectUserHandle *userHandle) {
}

/**** Should not be called the first time *********/
void CmiDirect_readyMark(CmiDirectUserHandle *userHandle) {
}

