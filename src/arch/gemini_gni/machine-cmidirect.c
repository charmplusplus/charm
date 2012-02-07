/** @file
 * uGNI cmiDirect communication
 * @ingroup Machine
*/

/*
  included in machine.c
  Yanhua Sun, 2/5/2012
*/

#define     CMI_DIRECT_DEBUG    1
#include "cmidirect.h"

static void printHandle(CmiDirectUserHandle *userHandle)
{
#if CMI_DIRECT_DEBUG
    CmiPrintf( "[%d] sender (%p, %lld, %lld), remote(%p, %lld, %lld)\n", CmiMyPe(), userHandle->senderBuf, userHandle->senderMdh.qword1, userHandle->senderMdh.qword2, 
        userHandle->recverBuf, userHandle->recverMdh.qword1, userHandle->recverMdh.qword2);
#endif
}

/**
 To be called on the receiver to create a handle and return its number
**/
CmiDirectUserHandle* CmiDirect_createHandle(int senderNode,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData,double initialValue) {

    gni_return_t            status = GNI_RC_SUCCESS;
    CmiDirectUserHandle *userHandle = malloc(sizeof(CmiDirectUserHandle));
    userHandle->handle=1; 
    userHandle->senderNode=senderNode;
    userHandle->recverNode=_Cmi_mynode;
    userHandle->recverBufSize=recvBufSize;
    userHandle->recverBuf=recvBuf;
    userHandle->initialValue=initialValue;
    userHandle->callbackFnPtr=callbackFnPtr;
    userHandle->callbackData=callbackData;

    if(recvBufSize < SMSG_MAX_MSG)
    {
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, userHandle->recverBuf, recvBufSize, &userHandle->recverMdh, &omdh);
    }
    else if(IsMemHndlZero((GetMemHndl(userHandle->recverBuf)))){
        status = registerMempool(userHandle->recverBuf);
        userHandle->recverMdh = GetMemHndl(userHandle->recverBuf);
    } else
        userHandle->recverMdh = GetMemHndl(userHandle->recverBuf);
    if(status != GNI_RC_SUCCESS) {
        userHandle->recverMdh.qword1 = 0;
        userHandle->recverMdh.qword2 = 0;
    }

#if CMI_DIRECT_DEBUG
    printHandle(userHandle);
#endif
    return userHandle;
}

/****
 To be called on the sender to attach the sender's buffer to this handle
******/

void CmiDirect_assocLocalBuffer(CmiDirectUserHandle *userHandle,void *sendBuf,int sendBufSize) {

    /* one-sided primitives would require registration of memory */
    /* with two-sided primitives we just record the sender buf in the handle */
    gni_return_t            status = GNI_RC_SUCCESS;
    userHandle->senderBuf=sendBuf;
 
    if(userHandle->recverBufSize < SMSG_MAX_MSG)
    {
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, userHandle->senderBuf, userHandle->recverBufSize, &userHandle->senderMdh, &omdh);
    }
    else if(IsMemHndlZero((GetMemHndl(userHandle->senderBuf)))){
        status = registerMempool(userHandle->senderBuf);
        userHandle->senderMdh = GetMemHndl(userHandle->senderBuf);
    } else
        userHandle->senderMdh = GetMemHndl(userHandle->senderBuf);

#if CMI_DIRECT_DEBUG
    CmiPrintf("Assciate");
    printHandle(userHandle);
#endif
}

/****
To be called on the sender to do the actual data transfer
******/
void CmiDirect_put(CmiDirectUserHandle *userHandle) {

#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] call put\n", CmiMyPe());
    printHandle(userHandle);
#endif
#if USE_LRTS_MEMPOOL
    if (userHandle->recverNode== CmiMyNode()) {
        CmiMemcpy(userHandle->recverBuf,userHandle->senderBuf,userHandle->recverBufSize);
        (*(userHandle->callbackFnPtr))(userHandle->callbackData);
    } else {
        gni_post_descriptor_t *pd;
        gni_return_t status;
        RDMA_REQUEST        *rdma_request_msg;
        MallocPostDesc(pd);
        if(userHandle->recverBufSize <= 2048)
            pd->type            = GNI_POST_FMA_PUT;
        else
            pd->type            = GNI_POST_RDMA_PUT;
        pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT;
        pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
        pd->length          = userHandle->recverBufSize;
        pd->local_addr      = (uint64_t) (userHandle->senderBuf);
        pd->local_mem_hndl  = userHandle->senderMdh; 
        pd->remote_addr     = (uint64_t)(userHandle->recverBuf);
        pd->remote_mem_hndl = userHandle->recverMdh;
        pd->src_cq_hndl     = 0;
        pd->rdma_mode       = 0;
        pd->first_operand   = (uint64_t) userHandle-> callbackFnPtr;
        pd->second_operand  = (uint64_t) userHandle-> callbackData;
        pd->amo_cmd         = 1;
        if(pd->type == GNI_POST_RDMA_PUT) 
            status = GNI_PostRdma(ep_hndl_array[userHandle->recverNode], pd);
        else
            status = GNI_PostFma(ep_hndl_array[userHandle->recverNode],  pd);
        printDesc(pd);
        if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
        {
            MallocRdmaRequest(rdma_request_msg);
            rdma_request_msg->destNode = userHandle->recverNode;
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
            GNI_RC_CHECK("AFter posting", status);
    }
#else
    CmiPrintf("Normal Send in CmiDirect Put\n");
    CmiAbort("");
#endif

#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA put addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif

}

// needs to figure out what is sender/recver
void CmiDirect_get(CmiDirectUserHandle *userHandle) {
#if USE_LRTS_MEMPOOL
    if (userHandle->recverNode== _Cmi_mynode) {
        CmiMemcpy(userHandle->recverBuf,userHandle->senderBuf,userHandle->recverBufSize);
        (*(userHandle->callbackFnPtr))(userHandle->callbackData);
    } else {
        gni_post_descriptor_t *pd;
        gni_return_t status;
        RDMA_REQUEST        *rdma_request_msg;
        MallocPostDesc(pd);
        if(userHandle->recverBufSize <= 2048)
            pd->type            = GNI_POST_FMA_GET;
        else
            pd->type            = GNI_POST_RDMA_GET;
        pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT;
        pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
        pd->length          = userHandle->recverBufSize;
        pd->local_addr      = (uint64_t) (userHandle->recverBuf);
        pd->local_mem_hndl  = userHandle->recverMdh; 
        pd->remote_addr     = (uint64_t)(userHandle->senderBuf);
        pd->remote_mem_hndl = userHandle->senderMdh;
        pd->src_cq_hndl     = 0;
        pd->rdma_mode       = 0;

        if(pd->type == GNI_POST_RDMA_PUT) 
            status = GNI_PostRdma(ep_hndl_array[userHandle->senderNode], pd);
        else
            status = GNI_PostFma(ep_hndl_array[userHandle->senderNode],  pd);
        if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
        {
            MallocRdmaRequest(rdma_request_msg);
            rdma_request_msg->destNode = userHandle->senderNode;
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
            GNI_RC_CHECK("AFter posting", status);
    }
#else
    CmiPrintf("Normal Send in CmiDirect Put\n");
    CmiAbort("");
#endif

#if CMI_DIRECT_DEBUG
        CmiPrintf("[%d] RDMA put addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif

}

/**** up to the user to safely call this */
void CmiDirect_deassocLocalBuffer(CmiDirectUserHandle *userHandle) {


}

/**** up to the user to safely call this */
void CmiDirect_destroyHandle(CmiDirectUserHandle *userHandle) {
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

