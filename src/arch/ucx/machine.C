/*
 * Copyright (c) 2019, Mellanox Technologies. All rights reserved.
 * See LICENSE in this directory.
 */

#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string>

#include "converse.h"
#include "cmirdmautils.h"
#include "machine.h"
#include "pcqueue.h"
#include "machine-lrts.h"
#include "machine-rdma.h"
#include "machine-common-core.C"

// UCX  headers
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#include <ucs/datastruct/mpool.h>

#if CMK_USE_PMI || CMK_USE_SIMPLEPMI
#include "runtime-pmi.C"
#elif CMK_USE_PMI2
#include "runtime-pmi2.C"
#elif CMK_USE_PMIX
#include "runtime-pmix.C"
#endif

#define CmiSetMsgSize(msg, sz)    ((((CmiMsgHeaderBasic *)msg)->size) = (sz))

#define UCX_MSG_PROBE_THRESH            32768
#define UCX_MSG_NUM_RX_REQS             64
#define UCX_MSG_NUM_RX_REQS_MAX         1024
#define UCX_TAG_MSG_BITS                4
#define UCX_TAG_RMA_BITS                4
#define UCX_TAG_PE_BITS                 32
#define UCX_MSG_TAG_EAGER               UCS_BIT(0)
#define UCX_MSG_TAG_PROBE               UCS_BIT(1)
#define UCX_MSG_TAG_DEVICE              UCS_BIT(2)
#define UCX_RMA_TAG_GET                 UCS_BIT(UCX_TAG_MSG_BITS + 1)
#define UCX_RMA_TAG_REG_AND_SEND_BACK   UCS_BIT(UCX_TAG_MSG_BITS + 2)
#define UCX_RMA_TAG_DEREG_AND_ACK       UCS_BIT(UCX_TAG_MSG_BITS + 3)
#define UCX_MSG_TAG_MASK                UCS_MASK(UCX_TAG_MSG_BITS)
#define UCX_RMA_TAG_MASK                (UCS_MASK(UCX_TAG_RMA_BITS) << UCX_TAG_MSG_BITS)
#define UCX_MSG_TAG_MASK_FULL           0xffffffffffffffffUL

#define UCX_LOG_PRIO 50 // Disabled by default

enum {
    UCX_SEND_OP,        // Regular Send using UcxSendMsg
    UCX_RMA_OP_PUT,     // RMA Put operation using UcxRmaOp
    UCX_RMA_OP_GET,     // RMA Get operation using UcxRmaOp
#if CMK_CUDA
    UCX_DEVICE_SEND_OP, // Device send
    UCX_DEVICE_RECV_OP, // Device recv
#endif
};

enum {
    UCX_AM_ID_SEND
#if CMK_ONESIDED_IMPL
    , UCX_AM_ID_RMA_REG_AND_SEND_BACK,
    UCX_AM_ID_RMA_GET,
    UCX_AM_ID_RMA_TAG_DEREG_AND_ACK
#endif
};

#define UCX_LOG(prio, fmt, ...) \
    do { \
        if (prio >= UCX_LOG_PRIO) { \
            CmiPrintf("UCX:%d-%d:%s> " fmt"\n",CmiMyNode(), CmiMyRank(), __func__, ##__VA_ARGS__); \
        } \
    } while (0)


typedef struct UcxRequest
{
    void           *msgBuf;
    int            idx;
    int            completed;
#if CMK_ONESIDED_IMPL
    void           *ncpyAck;
    ucp_rkey_h     rkey;
#endif
#if CMK_CUDA
    void*          cb;
    DeviceRdmaOp*  device_op;
    DeviceRecvType type;
#endif
} UcxRequest;

typedef struct UcxContext
{
    ucp_context_h     context;
    ucp_worker_h      worker;
    ucp_ep_h          *eps;
    UcxRequest        **rxReqs;
#if CMK_SMP
    PCQueue           txQueue;
#endif
    int               eagerSize;
    int               numRxReqs;
} UcxContext;

#ifdef CMK_SMP
typedef struct UcxPendingRequest
{
    int                     state;
    int                     index;
    void                    *msgBuf;
    int                     size;
    ucp_tag_t               tag;
    int                     dNode;
    int                     op;
    ucp_send_callback_t     cb;
#if CMK_CUDA
    ucp_tag_recv_callback_t recv_cb;
    ucp_tag_t               mask;
    DeviceRdmaOp*           device_op;
    DeviceRecvType          type;
#endif
} UcxPendingRequest;
#endif

static UcxContext ucxCtx;

static void UcxRxReqCompleted(void *request, ucs_status_t status,
                              size_t length, void* user_data);

static void UcxPrepostRxBuffers();

#if CMK_CUDA
CpvDeclare(int, tag_counter);
#endif

static ucs_status_t UcxAmRxDataCb(void *arg, const void *header, size_t header_length,
                                  void *data, size_t length,
                                  const ucp_am_recv_param_t *param);

#if CMK_ONESIDED_IMPL
static ucs_status_t UcxAmRxRmaPutCb(void *arg, const void *header, size_t header_length,
                                    void *data, size_t length,
                                    const ucp_am_recv_param_t *param);
static ucs_status_t UcxAmRxRmaGetCb(void *arg, const void *header, size_t header_length,
                                    void *data, size_t length,
                                    const ucp_am_recv_param_t *param);
static ucs_status_t UcxAmRxRmaDeregCb(void *arg, const void *header, size_t header_length,
                                      void *data, size_t length,
                                      const ucp_am_recv_param_t *param);

#include "machine-onesided.h"
#endif

#define UCX_CHECK_STATUS(_status, _str) \
{ \
    if (UCS_STATUS_IS_ERR(_status)) { \
        CmiAbort("UCX: " _str " failed: %s", ucs_status_string(_status)); \
    } \
}

#define UCX_CHECK_RET(_ret, _str, _cond) \
{ \
    if (_cond) { \
        CmiAbort("UCX: " _str " failed: %d", _ret); \
    } \
}

#define UCX_CHECK_PMI_RET(_ret, _str) UCX_CHECK_RET(_ret, _str, _ret)

#if CMK_CUDA
inline void UcxInvokeRecvHandler(DeviceRdmaOp* op, DeviceRecvType type) {
  switch (type) {
    case DEVICE_RECV_TYPE_CHARM:
      CmiInvokeRecvHandler(op);
      break;
    // TODO: AMPI and Charm4py
    default:
      CmiAbort("Invalid recv type: %d\n", type);
      break;
  }
}
#endif

void UcxRequestInit(void *request)
{
    UcxRequest *req = (UcxRequest*)request;
    req->msgBuf     = NULL;
    req->idx        = -1;
    req->completed  = 0;
#if CMK_CUDA
    req->cb         = NULL;
    req->device_op  = NULL;
#endif
}

static void UcxInitEps(int numNodes, int myId)
{
    size_t addrlen;
    ucp_address_t *address;
    ucs_status_t status;
    ucp_ep_params_t eParams;
    ucp_ep_h ep;
    int i, j, ret, peer, maxkey, maxval, parts, len, partLen;
    char *keys, *addrp, *remoteAddr;

    ret = runtime_get_max_keylen(&maxkey);
    UCX_CHECK_PMI_RET(ret, "UcxInitEps: runtime_get_max_keylen error");
    ret = runtime_get_max_vallen(&maxval);
    UCX_CHECK_PMI_RET(ret, "UcxInitEps: runtime_get_max_vallen error");

    // Reduce maxval value, because with PMI1 it has to fit cmd + key + value
    maxval -= 48;
    CmiEnforce(maxval > 0);

    keys = (char*)CmiAlloc(maxkey);
    CmiEnforce(keys);

    ucxCtx.eps = (ucp_ep_h*)CmiAlloc(sizeof(ucp_ep_h)*numNodes);
    CmiEnforce(ucxCtx.eps);

    status = ucp_worker_get_address(ucxCtx.worker, &address, &addrlen);
    UCX_CHECK_STATUS(status, "UcxInitEps: ucp_worker_get_address error");
    CmiEnforce(addrlen < std::numeric_limits<int>::max()); //address should fit to int

    parts = (addrlen / maxval) + 1;

    // Publish number of address parts at first
    ret = snprintf(keys, maxkey, "UCX-size-%d", myId);
    UCX_CHECK_RET(ret, "UcxInitEps: snprintf error", (ret <= 0));
    ret = runtime_kvs_put(keys, &parts, sizeof(parts));
    UCX_CHECK_PMI_RET(ret, "UcxInitEps: runtime_kvs_put error");

    addrp = (char*)address;
    len   = (int)addrlen;
    for (i = 0; i < parts; ++i) {
        partLen = std::min(maxval, len);
        ret = snprintf(keys, maxkey, "UCX-%d-%d", myId, i);
        UCX_CHECK_RET(ret, "UcxInitEps: snprintf error", (ret <= 0));
        ret = runtime_kvs_put(keys, addrp, partLen);
        UCX_CHECK_PMI_RET(ret, "UcxInitEps: runtime_kvs_put error");
        addrp += partLen;
        len   -= partLen;
    }

    // Ensure that all nodes published their worker addresses
    ret = runtime_barrier();
    UCX_CHECK_PMI_RET(ret, "UcxInitEps: runtime_barrier");

    ucp_worker_release_address(ucxCtx.worker, address);

    for (i = 0; i < numNodes; ++i) {
        peer = (i + myId) % numNodes;

        ret = snprintf(keys, maxkey, "UCX-size-%d", peer);
        UCX_CHECK_RET(ret, "UcxInitEps: snprintf error", (ret <= 0));
        ret = runtime_kvs_get(keys, &parts, sizeof(parts), peer);
        UCX_CHECK_PMI_RET(ret, "UcxInitEps: runtime_kvs_get error");

        remoteAddr = (char*)CmiAlloc(addrlen);
        CmiEnforce(remoteAddr);

        addrp = remoteAddr;
        len   = addrlen;
        for (j = 0; j < parts; ++j) {
            partLen = std::min(maxval, len);
            ret = snprintf(keys, maxkey, "UCX-%d-%d", peer, j);
            UCX_CHECK_RET(ret, "UcxInitEps: snprintf error", (ret <= 0));
            ret = runtime_kvs_get(keys, addrp, partLen, peer);
            UCX_CHECK_PMI_RET(ret, "UcxInitEps: runtime_kvs_get error");
            addrp += maxval;
            len   -= maxval;
        }

        eParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        eParams.address    = (const ucp_address_t*)remoteAddr;

        status = ucp_ep_create(ucxCtx.worker, &eParams, &ucxCtx.eps[peer]);
        UCX_CHECK_STATUS(status, "ucp_ep_create failed");
        UCX_LOG(4, "Connecting to %d (ep %p)", peer, ucxCtx.eps[peer]);
        CmiFree(remoteAddr);
    }

    CmiFree(keys);
}

void UcxSetAmDataHandler(ucp_worker_h worker, uint16_t am_id,
                         ucp_am_recv_callback_t data_cb)
{
    ucp_am_handler_param_t param;
    ucs_status_t status;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB;
    param.id         = am_id;
    param.cb         = data_cb;
    status           = ucp_worker_set_am_recv_handler(worker, &param);
    UCX_CHECK_STATUS(status, "UcxSetAmDataHandler:ucp_worker_set_am_recv_handler"
                             " ucp_worker_get_address error");
}

// Should be called for every node (not PE)
// Only invoked by comm threads
void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID)
{
    ucp_params_t cParams;
    ucp_config_t *config;
    ucp_worker_params_t wParams;
    ucs_status_t status;
    int ret;

    ret = runtime_init(myNodeID, numNodes);
    UCX_CHECK_PMI_RET(ret, "runtime_init");

    status = ucp_config_read("Charm++", NULL, &config);
    UCX_CHECK_STATUS(status, "ucp_config_read");

    // Initialize UCX context
    cParams.field_mask        = UCP_PARAM_FIELD_FEATURES          |
                                UCP_PARAM_FIELD_REQUEST_SIZE      |
                                UCP_PARAM_FIELD_TAG_SENDER_MASK   |
                                UCP_PARAM_FIELD_MT_WORKERS_SHARED |
                                UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    cParams.features          = UCP_FEATURE_AM | UCP_FEATURE_RMA;
    cParams.request_size      = sizeof(UcxRequest);
    cParams.tag_sender_mask   = 0ul;
    cParams.mt_workers_shared = 0;
    cParams.estimated_num_eps = *numNodes;

    status = ucp_init(&cParams, config, &ucxCtx.context);
    ucp_config_release(config);
    UCX_CHECK_STATUS(status, "ucp_init");

    // Create UCP worker
    wParams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wParams.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(ucxCtx.context, &wParams, &ucxCtx.worker);
    UCX_CHECK_STATUS(status, "ucp_worker_create");

    // Initialize Active Message data handlers
    UcxSetAmDataHandler(ucxCtx.worker, UCX_AM_ID_SEND, UcxAmRxDataCb);
#if CMK_ONESIDED_IMPL
    UcxSetAmDataHandler(ucxCtx.worker, UCX_AM_ID_RMA_REG_AND_SEND_BACK, UcxAmRxRmaPutCb);
    UcxSetAmDataHandler(ucxCtx.worker, UCX_AM_ID_RMA_GET, UcxAmRxRmaGetCb);
    UcxSetAmDataHandler(ucxCtx.worker, UCX_AM_ID_RMA_TAG_DEREG_AND_ACK, UcxAmRxRmaDeregCb);
#endif

    UcxInitEps(*numNodes, *myNodeID);

    // Ensure connects completion
    status = ucp_worker_flush(ucxCtx.worker);
    UCX_CHECK_STATUS(status, "ucp_worker_flush");

#if CMK_SMP
    ucxCtx.txQueue = PCQueueCreate();
#endif

    UCX_LOG(5, "Initialized: preposted reqs %d, rndv thresh %d\n",
            ucxCtx.numRxReqs, ucxCtx.eagerSize);

#if CMK_CUDA
    CpvInitialize(int, tag_counter);
    CpvAccess(tag_counter) = 0;
#endif
}

static ucs_status_t UcxAmRxDataCb(void *arg, const void *header, size_t header_length,
                                  void *data, size_t length,
                                  const ucp_am_recv_param_t *param)
{
    void *buf = CmiAlloc(length);

    CmiAssert(header_length == 0); // header is not used

    UCX_LOG(3, "RX AM data (%s), length %zu",
            (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) ? "rndv" : "eager",
            length);

    if (!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
        memcpy(buf, data, length);
        handleOneRecvedMsg(length, (char*)buf);
        return UCS_OK;
    }

    // RNDV request arrived need to initiate receive
    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK  |
                          UCP_OP_ATTR_FIELD_USER_DATA |
                          UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.recv_am   = UcxRxReqCompleted;
    params.user_data    = buf;
    ucs_status_ptr_t sp = ucp_am_recv_data_nbx(ucxCtx.worker, data, buf,
                                               length, &params);
    if (ucs_unlikely(UCS_PTR_IS_ERR(sp))) {
        CmiPrintf("UCX: ucp_am_recv_data_nbx failed with %s\n",
                  ucs_status_string(UCS_PTR_STATUS(sp)));
        CmiFree(buf);
    }

    return UCS_OK;
}

#if CMK_ONESIDED_IMPL
// Register the source buffer and send back to destination to perform GET
static ucs_status_t UcxAmRxRmaPutCb(void *arg, const void *header, size_t header_length,
                                    void *data, size_t length,
                                    const ucp_am_recv_param_t *param)
{
    CmiAssert(header_length == 0); // header is not used
    // Auxilliary RMA messages are always sent with eager
    CmiAssert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));

    NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)CmiAlloc(length);
    memcpy(ncpyOpInfo, data, length);
    UCX_LOG(4, "Got ncpy size %d (meta size %d)", ncpyOpInfo->srcSize, ncpyOpInfo->ncpyOpInfoSize);
    resetNcpyOpInfoPointers(ncpyOpInfo);

    UcxRdmaInfo *info = (UcxRdmaInfo *)(ncpyOpInfo->srcLayerInfo + CmiGetRdmaCommonInfoSize());

    UcxMemMap(info, (void *)ncpyOpInfo->srcPtr, ncpyOpInfo->srcSize);

    ncpyOpInfo->isSrcRegistered = 1;

    ncpyOpInfo->freeMe = CMK_FREE_NCPYOPINFO; // It's a message, not a realy ncpy Obj
    UCX_LOG(4, "Reset ncpy size %d (meta size %d)", ncpyOpInfo->destSize, ncpyOpInfo->ncpyOpInfoSize);

    // send back to destination process to perform GET
    UcxSendMsg(CmiNodeOf(ncpyOpInfo->destPe), ncpyOpInfo->destPe,
               ncpyOpInfo->ncpyOpInfoSize, (char*)ncpyOpInfo, UCX_AM_ID_RMA_GET,
               UCP_AM_SEND_FLAG_EAGER, UcxRmaSendCompletedAndFree);

    return UCS_OK;
}

static ucs_status_t UcxAmRxRmaGetCb(void *arg, const void *header, size_t header_length,
                                    void *data, size_t length,
                                    const ucp_am_recv_param_t *param)
{
    CmiAssert(header_length == 0); // header is not used
    // Auxilliary RMA messages are always sent with eager
    CmiAssert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));

    NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)CmiAlloc(length);
    memcpy(ncpyOpInfo, data, length);

    UCX_LOG(4, "RX Get request %d (meta size %d)", ncpyOpInfo->srcSize, ncpyOpInfo->ncpyOpInfoSize);

    resetNcpyOpInfoPointers(ncpyOpInfo);

    ncpyOpInfo->freeMe = CMK_FREE_NCPYOPINFO; // It's a message, not a real ncpy Obj
    UcxRmaOp(ncpyOpInfo, UCX_RMA_OP_GET);

    return UCS_OK;
}

static ucs_status_t UcxAmRxRmaDeregCb(void *arg, const void *header, size_t header_length,
                                      void *data, size_t length,
                                      const ucp_am_recv_param_t *param)
{
    CmiAssert(header_length == 0); // header is not used
    // Auxilliary RMA messages are always sent with eager
    CmiAssert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));

    NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)CmiAlloc(length);
    memcpy(ncpyOpInfo, data, length);

    UCX_LOG(4, "RX dereg req %d (meta size %d)", ncpyOpInfo->srcSize, ncpyOpInfo->ncpyOpInfoSize);

    resetNcpyOpInfoPointers(ncpyOpInfo);
    ncpyOpInfo->freeMe = CMK_FREE_NCPYOPINFO;

    if(CmiMyNode() == CmiNodeOf(ncpyOpInfo->srcPe)) { // source node
        LrtsDeregisterMem(ncpyOpInfo->srcPtr,
                          ncpyOpInfo->srcLayerInfo + CmiGetRdmaCommonInfoSize(),
                          ncpyOpInfo->srcPe,
                          ncpyOpInfo->srcRegMode);

        ncpyOpInfo->isSrcRegistered = 0; // Set isSrcRegistered to 0 after de-registration

        // Invoke source ack
        if(ncpyOpInfo->opMode != CMK_BCAST_EM_API) {
            ncpyOpInfo->opMode = CMK_EM_API_SRC_ACK_INVOKE;
            CmiInvokeNcpyAck(ncpyOpInfo);
        }

    } else if(CmiMyNode() == CmiNodeOf(ncpyOpInfo->destPe)) { // destination node

        LrtsDeregisterMem(ncpyOpInfo->destPtr,
                          ncpyOpInfo->destLayerInfo + CmiGetRdmaCommonInfoSize(),
                          ncpyOpInfo->destPe,
                          ncpyOpInfo->destRegMode);

        ncpyOpInfo->isDestRegistered = 0; // Set isDestRegistered to 0 after de-registration

        // Invoke destination ack
        ncpyOpInfo->opMode = CMK_EM_API_DEST_ACK_INVOKE;
        CmiInvokeNcpyAck(ncpyOpInfo);

    } else {
        CmiAbort(" Cannot de-register on a different node than the source or destinaton");
    }

    return UCS_OK;
}

#endif

static void UcxRxReqCompleted(void *request, ucs_status_t status,
                              size_t length, void *user_data)
{
    UCX_LOG(3, "status %d len %zu, buf %p\n", status, length, user_data);

    CmiEnforce(user_data != NULL); // user_data points to msg buffer

    if (ucs_unlikely(status != UCS_OK)) {
        CmiPrintf("UCX: AM RNDV receive failed with %s\n", status);
    }

    handleOneRecvedMsg(length, (char*)user_data);
    ucp_request_free(request);
}

static void UcxTxReqCompleted(void *request, ucs_status_t status, void *user_data)
{
    CmiEnforce(status == UCS_OK);
    CmiEnforce(user_data); // user_data points to msg buffer

    UCX_LOG(3, "TX req %p completed, free msg %p", request, user_data);
    CmiFree(user_data);
    ucp_request_free(request);
}

static inline void* UcxSendAm(int destNode, int size, char *msg, unsigned amId,
                              unsigned send_flags, ucp_send_nbx_callback_t cb)
{
    UcxRequest *req;
    ucp_request_param_t params;

    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    params.cb.send      = (ucp_send_nbx_callback_t)cb;
    params.user_data    = msg;

    if (send_flags) {
        params.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
        params.flags         = send_flags;
    }

    req = (UcxRequest*)ucp_am_send_nbx(ucxCtx.eps[destNode], amId, NULL, 0, msg,
                                       size, &params);
    if (!UCS_PTR_IS_PTR(req)) {
        CmiEnforce(!UCS_PTR_IS_ERR(req));
        return NULL;
    }

    return req;
}

inline void* UcxSendMsg(int destNode, int destPE, int size, char *msg,
                        unsigned amId, unsigned send_flags, ucp_send_nbx_callback_t cb)
{
    UCX_LOG(3, "destNode=%i destPE=%i size=%i msg=%p", destNode, destPE, size, msg);

#if CMK_SMP
    UcxPendingRequest *req = (UcxPendingRequest*)CmiAlloc(sizeof(UcxPendingRequest));
    req->msgBuf     = msg;
    req->size       = size;
    req->id         = amId;
    req->dNode      = destNode;
    req->cb         = cb;
    req->op         = UCX_SEND_OP;   // Mark this request as a regular message (UCX_SEND_OP)
    req->send_flags = send_flags;

    UCX_LOG(3, " --> (PE=%i) enq msg (queue depth=%i), dNode %i, size %i",
            CmiMyPe(), PCQueueLength(ucxCtx.txQueue), destNode, size);
    PCQueuePush(ucxCtx.txQueue, (char *)req);
#else
    void *req = UcxSendAm(destNode, size, msg, amId, send_flags, cb);
#endif

    return req;
}

/**
 * In non-SMP mode, this is used to send a message.
 * In CMK_SMP mode, this is called by a worker thread to send a message.
 */
CmiCommHandle LrtsSendFunc(int destNode, int destPE, int size, char *msg, int mode)
{

    void *req;

    CmiSetMsgSize(msg, size);

    req = UcxSendMsg(destNode, destPE, size, msg, UCX_AM_ID_SEND, 0, UcxTxReqCompleted);
    if (req == NULL) {
        /* Request completed in place or error occured */
        UCX_LOG(3, "Sent msg %p (len %d) inline", msg, size);
        CmiFree(msg);
        return NULL;
    }

    return (CmiCommHandle)req;
}

void LrtsPreCommonInit(int everReturn)
{
    UCX_LOG(2, "LrtsPreCommonInit");
}

void LrtsPostCommonInit(int everReturn)
{
    UCX_LOG(2, "LrtsPostCommonInit");
}

#if CMK_SMP
static inline int ProcessTxQueue()
{
    UcxPendingRequest *req;

    req = (UcxPendingRequest*)PCQueuePop(ucxCtx.txQueue);
    if (req) {
        if(req->op == UCX_SEND_OP) { // Regular Message
           void *status_ptr = UcxSendAm(req->dNode, req->size, (char*)req->msgBuf,
                                        req->id, req->send_flags, req->cb);
            if (!UCS_PTR_IS_PTR(status_ptr)) {
                if(req->id != UCX_AM_ID_SEND) {
                    NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(req->msgBuf);
                    if(ncpyOpInfo->freeMe == CMK_FREE_NCPYOPINFO)
                        CmiFree(ncpyOpInfo);
                } else {
                    CmiFree(req->msgBuf);
                }
            }
        }
#if CMK_ONESIDED_IMPL
        else if(req->op == UCX_RMA_OP_GET || req->op == UCX_RMA_OP_PUT) { // RMA Get or Put

            // Post the GET or PUT operation from the comm thread
            UcxRmaOp((NcpyOperationInfo *)(req->msgBuf), req->op);
        }
#endif
#if CMK_CUDA
        else if (req->op == UCX_DEVICE_SEND_OP) { // Send device data
          ucs_status_ptr_t status_ptr;
          status_ptr = ucp_tag_send_nb(ucxCtx.eps[req->dNode], req->msgBuf,
                                       req->size, ucp_dt_make_contig(1),
                                       req->tag, req->cb);
          if (!UCS_PTR_IS_PTR(status_ptr)) {
            // Either send was complete or error
            CmiEnforce(!UCS_PTR_IS_ERR(status_ptr));
            CmiEnforce(UCS_PTR_STATUS(status_ptr) == UCS_OK);
          } else {
            // Callback function will be invoked once send completes
            UcxRequest* store_req = (UcxRequest*)status_ptr;
            store_req->msgBuf = req->msgBuf;
          }
        } else if (req->op == UCX_DEVICE_RECV_OP) { // Recv device data
          ucs_status_ptr_t status_ptr;
          status_ptr = ucp_tag_recv_nb(ucxCtx.worker, req->msgBuf, req->size,
                                       ucp_dt_make_contig(1), req->tag, req->mask,
                                       req->recv_cb);
          CmiEnforce(!UCS_PTR_IS_ERR(status_ptr));

          UcxRequest* ret_req = (UcxRequest*)status_ptr;
          if (ret_req->completed) {
            // Recv was completed immediately
            UcxInvokeRecvHandler(req->device_op, req->type);
            UCX_REQUEST_FREE(ret_req);
          } else {
            // Recv wasn't completed immediately, recv_cb will be invoked
            // sometime later
            ret_req->device_op = req->device_op;
            ret_req->msgBuf = req->msgBuf;
            ret_req->type = req->type;
          }
        }
#endif
        else {
          CmiAbort("[%d][%d][%d] UCX:ProcessTxQueue req->op(%d) is Invalid\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), req->op);
        }
        CmiFree(req);
        return 1;
    }
    return 0;
}
#endif

void LrtsAdvanceCommunication(int whileidle)
{
    ucp_tag_message_h msg;
    ucp_tag_recv_info_t info;
    int cnt;

    do {
       cnt = ucp_worker_progress(ucxCtx.worker);

#if CMK_SMP
       cnt += ProcessTxQueue();
#endif
    } while (cnt);
}

void LrtsDrainResources()
{
    int ret;
    LrtsAdvanceCommunication(0);
    ret = runtime_barrier();
    UCX_CHECK_PMI_RET(ret, "runtime_barrier");
}

void LrtsExit(int exitcode)
{
    int ret;
    int i;
    UcxRequest *req;
    ucs_status_t status;

    UCX_LOG(4, "LrtsExit");

    LrtsAdvanceCommunication(0);

    ucp_worker_destroy(ucxCtx.worker);
    ucp_cleanup(ucxCtx.context);

    CmiFree(ucxCtx.eps);
#if CMK_SMP
    PCQueueDestroy(ucxCtx.txQueue);
#endif

    if(!CharmLibInterOperate || userDrivenMode) {
        ret = runtime_barrier();
        UCX_CHECK_PMI_RET(ret, "runtime_barrier");

        ret = runtime_fini();
        UCX_CHECK_PMI_RET(ret, "runtime_fini");
        if (!userDrivenMode) {
          exit(exitcode);
        }
    }
}

#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl()
{
    if (CmiMyRank() == CmiMyNodeSize()) {
        CommunicationServerThread(0);
    }
}
#endif

// In CMK_SMP, this is called by worker thread
void LrtsPostNonLocal()
{
    UCX_LOG(2, "LrtsPostNonLocal");
}

void LrtsAbort(const char *message)
{
    UCX_LOG(2, "LrtsAbort '%s'", message);
    exit(1);
    CMI_NORETURN_FUNCTION_END
}

void  LrtsNotifyIdle()
{
    UCX_LOG(2, "LrtsNotifyIdle");
}

void  LrtsBeginIdle()
{
    UCX_LOG(2, "LrtsBeginIdle");
}

void  LrtsStillIdle()
{
    UCX_LOG(2, "LrtsStillIdle");
}

void  LrtsBarrier()
{
    int ret;
    ret = runtime_barrier();
    UCX_CHECK_PMI_RET(ret, "runtime_barrier");
}

#if CMK_CUDA
void UcxSendDeviceCompleted(void* request, ucs_status_t status)
{
  CmiEnforce(status == UCS_OK);
  UcxRequest* req = (UcxRequest*)request;

  UCX_REQUEST_FREE(req);
}

void UcxRecvDeviceCompleted(void* request, ucs_status_t status,
                            ucp_tag_recv_info_t* info)
{
  UcxRequest* req = (UcxRequest*)request;

  if (ucs_unlikely(status == UCS_ERR_CANCELED)) return;
  CmiEnforce(status == UCS_OK);

  if (req->msgBuf != NULL) {
    // Invoke recv handler since data transfer is complete
    UcxInvokeRecvHandler(req->device_op, req->type);
    UCX_REQUEST_FREE(req);
  } else {
    // Request was completed immediately
    // Handle recv in the caller
    req->completed = 1;
  }
}

void LrtsSendDevice(int dest_pe, const void*& ptr, size_t size, uint64_t& tag) {
  // FIXME: Is this tag generation OK?
  tag = ((uint64_t)CpvAccess(tag_counter)++ << (UCX_TAG_PE_BITS + UCX_TAG_MSG_BITS)) | (CmiMyPe() << UCX_TAG_MSG_BITS) | UCX_MSG_TAG_DEVICE;
#if CMK_SMP
  UcxPendingRequest* req = (UcxPendingRequest*)CmiAlloc(sizeof(UcxPendingRequest));
  req->msgBuf = (void*)ptr;
  req->size   = size;
  req->tag    = tag;
  req->dNode  = CmiNodeOf(dest_pe);
  req->cb     = UcxSendDeviceCompleted;
  req->op     = UCX_DEVICE_SEND_OP;

  PCQueuePush(ucxCtx.txQueue, (char *)req);
#else
  ucs_status_ptr_t status_ptr;
  status_ptr = ucp_tag_send_nb(ucxCtx.eps[CmiNodeOf(dest_pe)], (void*)ptr, size,
                               ucp_dt_make_contig(1), tag,
                               UcxSendDeviceCompleted);

  if (!UCS_PTR_IS_PTR(status_ptr)) {
    // Either send was complete or error
    CmiEnforce(!UCS_PTR_IS_ERR(status_ptr));
    CmiEnforce(UCS_PTR_STATUS(status_ptr) == UCS_OK);
  } else {
    // Callback function will be invoked once send completes
    UcxRequest* req = (UcxRequest*)status_ptr;
    req->msgBuf = (void*)ptr;
  }
#endif // CMK_SMP
}

void LrtsRecvDevice(DeviceRdmaOp* op, DeviceRecvType type)
{
#if CMK_SMP
  UcxPendingRequest *req = (UcxPendingRequest*)CmiAlloc(sizeof(UcxPendingRequest));
  req->msgBuf    = (void*)op->dest_ptr;
  req->size      = op->size;
  req->tag       = op->tag;
  req->op        = UCX_DEVICE_RECV_OP;
  req->device_op = op;
  req->mask      = UCX_MSG_TAG_MASK_FULL;
  req->recv_cb   = UcxRecvDeviceCompleted;
  req->type      = type;

  PCQueuePush(ucxCtx.txQueue, (char *)req);
#else
  ucs_status_ptr_t status_ptr;
  status_ptr = ucp_tag_recv_nb(ucxCtx.worker, (void*)op->dest_ptr, op->size,
                               ucp_dt_make_contig(1), op->tag,
                               UCX_MSG_TAG_MASK_FULL, UcxRecvDeviceCompleted);
  CmiEnforce(!UCS_PTR_IS_ERR(status_ptr));

  UcxRequest* req = (UcxRequest*)status_ptr;
  if (req->completed) {
    // Recv was completed immediately
    UcxInvokeRecvHandler(op, type);
    UCX_REQUEST_FREE(req);
  } else {
    // Recv wasn't completed immediately, recv_cb will be invoked
    // sometime later
    req->device_op = op;
    req->msgBuf = (void*)op->dest_ptr;
    req->type = type;
  }
#endif // CMK_SMP
}
#endif // CMK_CUDA

#if CMK_ONESIDED_IMPL
#include "machine-onesided.C"
#endif
