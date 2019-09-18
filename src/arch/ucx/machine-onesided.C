/*
 * Copyright (c) 2019, Mellanox Technologies. All rights reserved.
 * See LICENSE in this directory.
 */

#include <ucp/api/ucp.h>


inline void UcxMemMap(UcxRdmaInfo *info, void *ptr, int size)
{
    ucp_mem_map_params_t memParams;
    ucs_status_t status;
    void *rbuf;
    size_t rkeySize;

    memset(&memParams, 0, sizeof(ucp_mem_map_params_t));
    memParams.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    memParams.length     = size;
    memParams.address    = ptr;

    status = ucp_mem_map(ucxCtx.context, &memParams, &(info->memh));
    UCX_CHECK_STATUS(status, "ucp_mem_map");

    status = ucp_rkey_pack(ucxCtx.context, info->memh, &rbuf, &rkeySize);
    UCX_CHECK_STATUS(status, "ucp_rkey_pack");

    CmiEnforce(rkeySize <= UCX_MAX_PACKED_RKEY_SIZE);
    memcpy(info->packedRkey, rbuf, rkeySize);

    ucp_rkey_buffer_release(rbuf);
    UCX_LOG(4, " key packed, size %ld, buf %p memh %d",
            rkeySize, info->packedRkey, info->memh);
}

void UcxRmaReqCompleted(void *request, ucs_status_t status)
{
    UcxRequest *req = (UcxRequest*)request;
    CmiEnforce(status == UCS_OK);

    CmiInvokeNcpyAck(req->ncpyAck);
    ucp_rkey_destroy(req->rkey);
    UCX_REQUEST_FREE(req);
    UCX_LOG(4, "RMA req completed %p", req);
}

void UcxRmaSendCompleted(void *request, ucs_status_t status)
{
    UcxRequest *req = (UcxRequest*)request;
    CmiEnforce(status == UCS_OK);

    UCX_REQUEST_FREE(req);
    UCX_LOG(4, "RMA Send completed %p", req);
}

void UcxRmaSendCompletedAndFree(void *request, ucs_status_t status)
{
    UcxRequest *req = (UcxRequest*)request;
    CmiEnforce(status == UCS_OK);

    CmiFree(req->msgBuf);

    UCX_REQUEST_FREE(req);
    UCX_LOG(4, "RMA Send completed %p", req);
}

void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size,
                           unsigned short int mode)
{
    UcxRdmaInfo *rdmaDest = (UcxRdmaInfo*)info;

    UCX_LOG(4, " %p, size %d", ptr, size);

    UcxMemMap(rdmaDest, (void*)ptr, size);
}

void LrtsDeregisterMem(const void *ptr, void *info, int pe, unsigned short int mode)
{
    ucs_status_t status;
    UcxRdmaInfo *ucxInfo = (UcxRdmaInfo*)info;

    UCX_LOG(4, " %p, PE %d, info %p, memh %d", ptr, pe, ucxInfo, ucxInfo->memh);

    if ((mode != CMK_BUFFER_NOREG) && (ucxInfo->memh)) {
        status = ucp_mem_unmap(ucxCtx.context, ucxInfo->memh);
        UCX_CHECK_STATUS(status, "ucp_mem_unmap");
    }
}

void UcxRmaOp(NcpyOperationInfo *ncpyOpInfo, int op)
{
    UcxRdmaInfo *srcInfo = (UcxRdmaInfo*)((char*)(ncpyOpInfo->srcLayerInfo) + CmiGetRdmaCommonInfoSize());
    UcxRdmaInfo *dstInfo = (UcxRdmaInfo*)((char*)(ncpyOpInfo->destLayerInfo) + CmiGetRdmaCommonInfoSize());
    ucs_status_ptr_t statusReq;
    ucs_status_t status;
    ucp_rkey_h rkey;
    ucp_ep_h ep;

    UCX_LOG(4, "RmaOp: op %d, (srcPE %d destPE %d) (srcSize %d destSize %d) dest rbuf %p, Smemh %d Dmemh %d",
            op, ncpyOpInfo->srcPe, ncpyOpInfo->destPe, ncpyOpInfo->srcSize,
            ncpyOpInfo->destSize, dstInfo->packedRkey, srcInfo->memh, dstInfo->memh);

#if CMK_SMP
    if(CmiMyRank() != CmiMyNodeSize()) { // worker thread

      // Push a UcxPendingRequest to post the GET or PUT operation from the comm thread
      UcxPendingRequest *req = (UcxPendingRequest*)CmiAlloc(sizeof(UcxPendingRequest));
      req->msgBuf = ncpyOpInfo;
      req->dNode  = (op == UCX_RMA_OP_GET) ? CmiNodeOf(ncpyOpInfo->srcPe) : CmiNodeOf(ncpyOpInfo->destPe);
      req->size   = std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize);
      req->op     = op;

      UCX_LOG(3, " --> (PE=%i) enq msg (queue depth=%i), dNode %i, size %i",
              CmiMyPe(), PCQueueLength(ucxCtx.txQueue), req->dNode, req->size);
      PCQueuePush(ucxCtx.txQueue, (char *)req);
      return;
    }
#endif

    if (op == UCX_RMA_OP_PUT) {
        ep     = ucxCtx.eps[CmiNodeOf(ncpyOpInfo->destPe)];
        status = ucp_ep_rkey_unpack(ep, dstInfo->packedRkey, &rkey);
        UCX_CHECK_STATUS(status, "ucp_ep_rkey_unpack");

        statusReq = ucp_put_nb(ep, ncpyOpInfo->srcPtr,
                               std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
                               (uint64_t)ncpyOpInfo->destPtr, rkey,
                               UcxRmaReqCompleted);
    } else {
        CmiEnforce(op == UCX_RMA_OP_GET);

        ep = ucxCtx.eps[CmiNodeOf(ncpyOpInfo->srcPe)];
        status = ucp_ep_rkey_unpack(ep, srcInfo->packedRkey, &rkey);
        UCX_CHECK_STATUS(status, "ucp_ep_rkey_unpack");

        statusReq = ucp_get_nb(ep, (void*)ncpyOpInfo->destPtr,
                               std::min(ncpyOpInfo->srcSize, ncpyOpInfo->destSize),
                               (uint64_t)ncpyOpInfo->srcPtr, rkey,
                               UcxRmaReqCompleted);
    }

    if (!UCS_PTR_IS_PTR(statusReq)) {
        CmiEnforce(UCS_PTR_STATUS(statusReq) == UCS_OK);
        CmiInvokeNcpyAck(ncpyOpInfo);
        ucp_rkey_destroy(rkey);
    } else {
        ((UcxRequest*)statusReq)->ncpyAck = ncpyOpInfo;
        ((UcxRequest*)statusReq)->rkey    = rkey;
    }
}

// Perform an RDMA Get call into the local destination address from the remote source address
void LrtsIssueRget(NcpyOperationInfo *ncpyOpInfo)
{
    UCX_LOG(4, "srcPE %d destPE %d ", ncpyOpInfo->srcPe, ncpyOpInfo->destPe);

    if (ncpyOpInfo->isSrcRegistered != 0) {
        UcxRmaOp(ncpyOpInfo, UCX_RMA_OP_GET);
    } else {
        // Remote buffer is not registered, ask peer to register and send back to me for GET

        UcxSendMsg(CmiNodeOf(ncpyOpInfo->srcPe), ncpyOpInfo->srcPe,
                   ncpyOpInfo->ncpyOpInfoSize, (char*)ncpyOpInfo,
                   UCX_RMA_TAG_REG_AND_SEND_BACK, UcxRmaSendCompleted);

        UCX_LOG(4, "Sending PUT REQ to %d, mem size %d", ncpyOpInfo->srcPe,ncpyOpInfo->srcSize);
    }
}

void LrtsIssueRput(NcpyOperationInfo *ncpyOpInfo)
{
    UCX_LOG(4, "srcPE %d destPE %d, size %d",
            ncpyOpInfo->srcPe, ncpyOpInfo->destPe, ncpyOpInfo->ncpyOpInfoSize);

    // Since UCX doesn't guarantee destination completion with PUT, it is implemented
    // as a reverse operation with the remote destination performing a GET
    UcxSendMsg(CmiNodeOf(ncpyOpInfo->destPe), ncpyOpInfo->destPe,
               ncpyOpInfo->ncpyOpInfoSize, (char*)ncpyOpInfo,
               UCX_RMA_TAG_GET, UcxRmaSendCompleted);
    UCX_LOG(4, "Sending Get REQ to %d, mem size %d", ncpyOpInfo->destPe,ncpyOpInfo->srcSize);
}

void LrtsInvokeRemoteDeregAckHandler(int pe, NcpyOperationInfo *ncpyOpInfo)
{
    NcpyOperationInfo *newNcpyOpInfo;

    if(ncpyOpInfo->opMode == CMK_DIRECT_API) {

        // ncpyOpInfo is not freed
        newNcpyOpInfo = ncpyOpInfo;

    } else if(ncpyOpInfo->opMode == CMK_EM_API || ncpyOpInfo->opMode == CMK_BCAST_EM_API) {

        // ncpyOpInfo is a part of the received message and can be freed before this send completes
        // for that reason, it is copied into a new message
        newNcpyOpInfo = (NcpyOperationInfo *)CmiAlloc(ncpyOpInfo->ncpyOpInfoSize);
        memcpy(newNcpyOpInfo, ncpyOpInfo, ncpyOpInfo->ncpyOpInfoSize);

        resetNcpyOpInfoPointers(newNcpyOpInfo);
        newNcpyOpInfo->freeMe = CMK_FREE_NCPYOPINFO;

    } else {
        CmiAbort("UCX: LrtsInvokeRemoteDeregAckHandler - ncpyOpInfo->opMode is not valid for dereg\n");
    }

    UcxSendMsg(CmiNodeOf(pe), pe,
               newNcpyOpInfo->ncpyOpInfoSize, (char*)newNcpyOpInfo,
               UCX_RMA_TAG_DEREG_AND_ACK, UcxRmaSendCompletedAndFree);
}
