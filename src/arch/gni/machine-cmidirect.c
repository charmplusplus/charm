/** @file
 * uGNI cmiDirect communication
 * @ingroup Machine
*/

/*
  included in machine.c
  Yanhua Sun, 2/5/2012
*/

#define     CMI_DIRECT_DEBUG    0
#include "cmidirect.h"
CmiDirectMemoryHandler CmiDirect_registerMemory(void *buff, int size)
{
    CmiDirectMemoryHandler mem_hndl; 
    gni_return_t        status;
    status = registerMessage(buff, size, 0, &mem_hndl); 
    //MEMORY_REGISTER(onesided_hnd, nic_hndl, buff, size, &mem_hndl, &omdh, status);
    GNI_RC_CHECK("cmidirect register memory fails\n", status);
    return mem_hndl;
}
static void printHandle(CmiDirectUserHandle *userHandle, char *s)
{
    CmiPrintf( "[%d]%s(%p)(%p,%p,%p)==>(%p,%p,%p)(%d)(%p,%p)\n", CmiMyPe(), s, userHandle, userHandle->localBuf, userHandle->localMdh.qword1, userHandle->localMdh.qword2, 
        userHandle->remoteBuf, userHandle->remoteMdh.qword1, userHandle->remoteMdh.qword2, userHandle->transSize, userHandle->callbackFnPtr, userHandle->callbackData );
}

struct infiDirectUserHandle CmiDirect_createHandle_mem(CmiDirectMemoryHandler *mem_hndl, void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData)
{
    gni_return_t            status = GNI_RC_SUCCESS;
    CmiDirectUserHandle userHandle;
    userHandle.handle=1; 
    userHandle.remoteNode= CmiMyNode();
    userHandle.remoteRank = CmiMyRank();
    userHandle.transSize=recvBufSize;
    userHandle.remoteBuf=recvBuf;
    userHandle.callbackFnPtr=callbackFnPtr;
    userHandle.callbackData=callbackData;
    userHandle.remoteMdh = *mem_hndl;
    userHandle.initialValue=0;
#if CMI_DIRECT_DEBUG
    //printHandle(&userHandle, "Create Handler");
#endif
    return userHandle;

}
/**
 To be called on the receiver to create a handle and return its number
**/
CmiDirectUserHandle CmiDirect_createHandle(int localNode,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData,double initialValue) {

    gni_return_t            status = GNI_RC_SUCCESS;
    CmiDirectUserHandle userHandle;
    userHandle.handle=1; 
    userHandle.localNode=localNode;
    userHandle.remoteNode= CmiMyNode();
    userHandle.transSize=recvBufSize;
    userHandle.remoteBuf=recvBuf;
    userHandle.initialValue=initialValue;
    userHandle.callbackFnPtr=callbackFnPtr;
    userHandle.callbackData=callbackData;
    if(recvBufSize <= SMSG_MAX_MSG)
    {
        status = registerMessage(userHandle.remoteBuf, recvBufSize, 0, &userHandle.remoteMdh); 
        //MEMORY_REGISTER(onesided_hnd, nic_hndl, userHandle.remoteBuf, recvBufSize, &(userHandle.remoteMdh), &omdh, status);
    }
    else if(IsMemHndlZero((GetMemHndl(userHandle.remoteBuf)))){
        //status = registerMempool(userHandle.remoteBuf);
        userHandle.remoteMdh = GetMemHndl(userHandle.remoteBuf);
    } else
        userHandle.remoteMdh = GetMemHndl(userHandle.remoteBuf);
    if(status != GNI_RC_SUCCESS) {
        userHandle.remoteMdh.qword1 = 0;
        userHandle.remoteMdh.qword2 = 0;
    }

#if REMOTE_EVENT
    userHandle.ack_index =  IndexPool_getslot(&ackPool, userHandle.remoteBuf, 1);
#endif
#if CMI_DIRECT_DEBUG
    //printHandle(&userHandle, "Create Handler");
#endif
    return userHandle;
}

void CmiDirect_saveHandler(CmiDirectUserHandle* h, void *ptr)
{
    h->remoteHandler = ptr;
}

void CmiDirect_assocLocalBuffer_mem(CmiDirectUserHandle *userHandle, CmiDirectMemoryHandler *mem_hndl, void *sendBuf,int sendBufSize) {
    gni_return_t            status = GNI_RC_SUCCESS;
    
    userHandle->localNode=CmiMyNode();
    userHandle->localBuf=sendBuf;

    userHandle->localMdh = *mem_hndl;
 
#if CMI_DIRECT_DEBUG
    printHandle(userHandle, "Associate Handler");
#endif
}
/****
 To be called on the local to attach the local's buffer to this handle
******/

void CmiDirect_assocLocalBuffer(CmiDirectUserHandle *userHandle,void *sendBuf,int sendBufSize) {

    /* one-sided primitives would require registration of memory */
    gni_return_t            status = GNI_RC_SUCCESS;
    
    userHandle->localNode=CmiMyNode();
    userHandle->localBuf=sendBuf;

    if(userHandle->transSize <= SMSG_MAX_MSG)
    {
        status = registerMessage(userHandle->localBuf, userHandle->transSize, 0, &(userHandle->localMdh)); 
        //MEMORY_REGISTER(onesided_hnd, nic_hndl, userHandle->localBuf, userHandle->transSize, &userHandle->localMdh, &omdh, status);
    }
    else if(IsMemHndlZero((GetMemHndl(userHandle->localBuf)))){
        //status = registerMempool(userHandle->localBuf);
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
        pd->first_operand   = (uint64_t)(userHandle->remoteHandler);
        pd->amo_cmd         = 1;
        pd->cqwrite_value   = DIRECT_SEQ;
#if REMOTE_EVENT
        bufferRdmaMsg(sendRdmaBuf, CmiGetNodeGlobal(userHandle->remoteNode,CmiMyPartition()), pd, userHandle->ack_index); 
#else
        bufferRdmaMsg(sendRdmaBuf, CmiGetNodeGlobal(userHandle->remoteNode,CmiMyPartition()), pd, -1); 
#endif
#if CMI_DIRECT_DEBUG
        printHandle(userHandle, "After Direct_put");
        CmiPrintf("[%d] RDMA put %d,%d bytes addr %p to remoteNode %d:%p \n\n",CmiMyPe(), userHandle->transSize, pd->length, (void*)(pd->local_addr), userHandle->remoteNode, (void*) (pd->remote_addr));
#endif
    }
#else
    CmiPrintf("Normal Send in CmiDirect Put\n");
    CmiAbort("");
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
        pd->cqwrite_value   = DIRECT_SEQ;
#if REMOTE_EVENT
        bufferRdmaMsg(sendRdmaBuf, CmiGetNodeGlobal(userHandle->remoteNode,CmiMyPartition()), pd, userHandle->ack_index); 
#else
        bufferRdmaMsg(sendRdmaBuf, CmiGetNodeGlobal(userHandle->remoteNode,CmiMyPartition()), pd, -1);
#endif
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA get %d,%d bytes addr %p to remoteNode %d:%p \n\n",CmiMyPe(), userHandle->transSize, pd->length, (void*)(pd->local_addr), userHandle->remoteNode, (void*) (pd->remote_addr));
#endif
    }
#else
    CmiPrintf("Normal Send in CmiDirect Get\n");
    CmiAbort("");
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

