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
#include "conv-ccs.h"
#include "ccs-server.h"
#include "ckrescale.h"
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

#if CMK_SHRINK_EXPAND
#include <limits.h>
#include <setjmp.h>
#include <unordered_map>
#include "../../util/coordinator/coord_client.h"

static void UcxCloseEp(ucp_ep_h ep);
static void UcxForceCloseEps(ucp_ep_h *eps, int n);

// FORCE-closing endpoints to departing peers (no disconnect handshake)
// requires PEER error-handling mode, but PEER mode excludes transports
// without fault-detection support -- notably shared memory -- which slows
// intra-node traffic badly (~5x on loopback). Opt in with +ucx_peer_err
// when rescale latency matters more than intra-node throughput.
static bool _ucx_peer_err_mode = false;
static void UcxSetEpErrMode(ucp_ep_params_t &p)
{
    if (_ucx_peer_err_mode) {
        p.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        p.err_mode    = UCP_ERR_HANDLING_MODE_PEER;
    }
}
// shrinkExpandreplyToken now lives in src/util/ckrescale.C, which every build
// links, so it exists for machine layers and for Reconverse alike.
extern int numProcessAfterRestart;
extern char *_shrinkexpand_basedir;
// mynewpe now lives in src/util/ckrescale.C, which every build links.

// Set by ConverseCleanup, read by charm_main after longjmp
extern jmp_buf _shrinkexpand_jmpbuf;
static bool _shrinkexpand_restarting = false;
static int  _shrinkexpand_new_numnodes = 0;
static int  _shrinkexpand_my_node = 0;

// Coordinator state (per-node).
static int   _coord_fd       = -1;
static char *_coord_host     = nullptr;  // owned by argv, do not free
static int   _coord_port     = 0;
static uint32_t _coord_epoch = 0;        // tracked locally; updated on each commit
extern int _rescaleGeneration;           // cluster-global generation (convcore.C)
// Set in LrtsInit when launcher passed +nodeId/+numNodes/+coordinator and PMI
// was skipped. Read in LrtsDrainResources/LrtsExit to route shutdown barriers
// through the coordinator instead of runtime_barrier (which would no-op or
// crash since PMI was never initialized).
static bool  _coord_bootstrap_active = false;
static int   _coord_bootstrap_my_node = -1;
// Mirror of the coordinator's current member list (in current nodeId order).
// Used by UcxReInitEpsFromView to diff old vs new and avoid tearing down eps
// to peers that survived. Updated in LrtsInit and at every successful commit.
static std::vector<coord::Member> _coord_members;
#endif

#define CmiSetMsgSize(msg, sz)    ((((CmiMsgHeaderBasic *)msg)->size) = (sz))

#define UCX_MSG_PROBE_THRESH            32768
#define UCX_MSG_NUM_RX_REQS             64
#define UCX_MSG_NUM_RX_REQS_MAX         1024
// UCX_TAG_MSG_BITS bumped from 4 to 6 to make room for the rescale-time
// control-plane tags below (RECONFIG, BARRIER_ARRIVE, BARRIER_RELEASE).
// RMA tags shift up by 2 bits accordingly; nothing in the codebase places
// data in the formerly-RMA-mask bits 4..5 standalone.
#define UCX_TAG_MSG_BITS                6
#define UCX_TAG_RMA_BITS                4
#define UCX_TAG_PE_BITS                 32
#define UCX_MSG_TAG_EAGER               UCS_BIT(0)
#define UCX_MSG_TAG_PROBE               UCS_BIT(1)
#define UCX_MSG_TAG_DEVICE              UCS_BIT(2)
#if CMK_SHRINK_EXPAND
// Control-plane tags used only at rescale time.
// RECONFIG: chain-broadcast of the new cluster view from PE 0 down the
//   binary tree over OLD-numbering UCX endpoints (pre-ep-rebuild).
// BARRIER_ARRIVE / BARRIER_RELEASE: tree-collective barrier replacing the
//   coordinator BARRIER round-trip; up-reduce then down-broadcast over
//   NEW-numbering UCX endpoints (post-ep-rebuild).
// All three must NOT overlap with EAGER/PROBE/DEVICE in the low
// UCX_TAG_MSG_BITS; bits 3,4,5 are free in that range.
#define UCX_MSG_TAG_RECONFIG            UCS_BIT(3)
#define UCX_MSG_TAG_BARRIER_ARRIVE      UCS_BIT(4)
#define UCX_MSG_TAG_BARRIER_RELEASE     UCS_BIT(5)
// Forward declaration so the newcomer-side LrtsInit (defined earlier in this
// file) can invoke the tree barrier (defined alongside the other rescale
// helpers, much further down).
static void UcxTreeBarrier(int newNodeId, int numNodes);
#endif
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

#if CMK_CUDA
// hybridAPI; used on the newcomer path to overlap CUDA context creation
// with the coordinator INTEGRATE wait.
extern "C" void hapiWarmupDeviceContext(void);
#endif

#define UCX_LOG(prio, fmt, ...) \
    do { \
        if (prio >= UCX_LOG_PRIO) { \
            CmiPrintf("UCX:%d-%d:%s> " fmt"\n",CmiMyNode(), CmiMyRank(), __func__, ##__VA_ARGS__); \
        } \
    } while (0)

#define UCX_REQUEST_FREE(req) \
    do { \
        req->msgBuf    = NULL; \
        req->completed = 0; \
        ucp_request_free(req); \
    } while(0)


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
                              ucp_tag_recv_info_t *info);
static void UcxPrepostRxBuffers();

#if CMK_CUDA
CpvDeclare(int, tag_counter);
#endif

#if CMK_ONESIDED_IMPL
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
        UcxSetEpErrMode(eParams);
        eParams.address    = (const ucp_address_t*)remoteAddr;

        status = ucp_ep_create(ucxCtx.worker, &eParams, &ucxCtx.eps[peer]);
        UCX_CHECK_STATUS(status, "ucp_ep_create failed");
        UCX_LOG(4, "Connecting to %d (ep %p)", peer, ucxCtx.eps[peer]);
        CmiFree(remoteAddr);
    }

    CmiFree(keys);
}

// Should be called for every node (not PE)
// Only invoked by comm threads
void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID)
{
    _ucx_peer_err_mode = CmiGetArgFlagDesc(*argv, "+ucx_peer_err",
        "create UCX endpoints with PEER error-handling mode (enables "
        "force-close of endpoints to departing peers on rescale; may "
        "exclude shared-memory transports)");
#if CMK_SHRINK_EXPAND
    if (_shrinkexpand_restarting) {
        // UCX was already re-initialized by UcxReInitEpsFromView in ConverseCleanup.
        // Just report the new topology to the caller.
        *numNodes = _shrinkexpand_new_numnodes;
        *myNodeID = _shrinkexpand_my_node;
        return;
    }

    // Newcomer process: standalone (no PMI), joins a running cluster via the
    // coordinator. Launched by external manager with:
    //   ./binary +newcomer +coordinator host:port +restart <basedir> ...
    // We init UCX locally, register with the coordinator and block in INTEGRATE
    // until PE 0 commits the rescale, then build endpoints from the returned
    // member list. The +restart flag (injected in charm_main if missing) is what
    // gets us into CkRestartMain's broadcast handler so we receive readonlies.
    if (CmiGetArgFlagDesc(*argv, "+newcomer",
            "Join a running cluster as a newcomer (used by external manager)")) {
        ucp_params_t cParams;
        ucp_config_t *config;
        ucp_worker_params_t wParams;
        ucs_status_t status;

        status = ucp_config_read("Charm++", NULL, &config);
        UCX_CHECK_STATUS(status, "ucp_config_read (newcomer)");

        cParams.field_mask        = UCP_PARAM_FIELD_FEATURES          |
                                    UCP_PARAM_FIELD_REQUEST_SIZE      |
                                    UCP_PARAM_FIELD_TAG_SENDER_MASK   |
                                    UCP_PARAM_FIELD_REQUEST_INIT      |
                                    UCP_PARAM_FIELD_MT_WORKERS_SHARED |
                                    UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        cParams.features          = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
        cParams.request_size      = sizeof(UcxRequest);
        cParams.tag_sender_mask   = 0ul;
        cParams.request_init      = UcxRequestInit;
        cParams.mt_workers_shared = 0;
        // Conservative; reset after INTEGRATE tells us actual count.
        cParams.estimated_num_eps = 1;

        status = ucp_init(&cParams, config, &ucxCtx.context);
        ucp_config_release(config);
        UCX_CHECK_STATUS(status, "ucp_init (newcomer)");

        wParams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        wParams.thread_mode = UCS_THREAD_MODE_SINGLE;
        status = ucp_worker_create(ucxCtx.context, &wParams, &ucxCtx.worker);
        UCX_CHECK_STATUS(status, "ucp_worker_create (newcomer)");

        ucxCtx.numRxReqs = UCX_MSG_NUM_RX_REQS;
        if (CmiGetArgInt(*argv, "+ucx_num_rx_reqs", &ucxCtx.numRxReqs)) {
            if ((ucxCtx.numRxReqs <= 0) || (ucxCtx.numRxReqs > UCX_MSG_NUM_RX_REQS_MAX)) {
                CmiPrintf("UCX: Invalid number of RX reqs: %d\n", ucxCtx.numRxReqs);
                CmiAbort(__func__);
            }
        }

        int thresh = UCX_MSG_PROBE_THRESH;
        CmiGetArgInt(*argv, "+ucx_rndv_thresh", &thresh);
        ucxCtx.eagerSize = std::max(LrtsGetMaxNcpyOperationInfoSize(), thresh);

        char *coordSpec = nullptr;
        CmiGetArgStringDesc(*argv, "+coordinator", &coordSpec,
                            "host:port of the rescale coordinator");
        if (coordSpec == nullptr) {
            CmiAbort("UCX: +newcomer requires +coordinator host:port");
        }
        char *colon = strchr(coordSpec, ':');
        if (colon == nullptr) CmiAbort("UCX: +coordinator must be host:port");
        *colon = '\0';
        _coord_host = coordSpec;
        _coord_port = atoi(colon + 1);
        if (_coord_port <= 0) CmiAbort("UCX: +coordinator port invalid");

        ucp_address_t *myAddr = nullptr;
        size_t myAddrLen = 0;
        status = ucp_worker_get_address(ucxCtx.worker, &myAddr, &myAddrLen);
        UCX_CHECK_STATUS(status, "ucp_worker_get_address (newcomer)");

        _coord_fd = coord::connect_blocking(_coord_host, _coord_port);
        if (_coord_fd < 0) {
            CmiAbort("UCX: failed to connect to coordinator (newcomer)");
        }

        // Phase 1: snapshot. Coordinator returns the current member list
        // immediately so we can build speculative eps + handshake while
        // waiting for COMMIT. Final nodeIds and the post-kill member set
        // arrive later via INTEGRATE.
        coord::ClusterView snapshot;
        if (!coord::register_newcomer(_coord_fd, myAddr, (uint32_t)myAddrLen,
                                      &snapshot)) {
            CmiAbort("UCX: coordinator REGISTER_NEWCOMER failed");
        }
        ucp_worker_release_address(ucxCtx.worker, myAddr);

        // Speculative eps keyed by ucxAddr (the snapshot's nodeIds may not
        // survive compact renumbering, so we can't index by them yet).
        std::unordered_map<std::string, ucp_ep_h> specEps;
        specEps.reserve(snapshot.members.size());
        for (const auto &m : snapshot.members) {
            ucp_ep_params_t eParams;
            eParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
            UcxSetEpErrMode(eParams);
            eParams.address    = (const ucp_address_t *)m.ucxAddr.data();
            ucp_ep_h ep = nullptr;
            status = ucp_ep_create(ucxCtx.worker, &eParams, &ep);
            UCX_CHECK_STATUS(status, "ucp_ep_create (newcomer speculative)");
            specEps[m.ucxAddr] = ep;
        }

        UcxPrepostRxBuffers();
        // Force handshakes to complete now so they overlap with the manager's
        // RTT to PE 0 + PE 0's wait for the LB step to drain.
        status = ucp_worker_flush(ucxCtx.worker);
        UCX_CHECK_STATUS(status, "ucp_worker_flush (newcomer speculative)");

#if CMK_CUDA
        // Warm up the CUDA driver and primary context while blocked below:
        // the INTEGRATE wait typically lasts seconds (until the running job
        // reaches its next load balancing step), whereas hapiInit's context
        // creation costs ~170 ms and would otherwise run after integration,
        // where the init-time lockstep barriers put it on the critical path
        // of every survivor's restore (observed as the dominant term of the
        // GPU expand: 172 ms at 4 nodes growing to 233 ms at 8).
        hapiWarmupDeviceContext();
#endif

        // Phase 2: block until COMMIT pushes the final view.
        coord::ClusterView view;
        if (!coord::await_integrate(_coord_fd, &view)) {
            CmiAbort("UCX: coordinator INTEGRATE (await) failed");
        }
        _coord_epoch = view.epoch;
        _rescaleGeneration = (int)view.epoch;

        const int newNumNodes = (int)view.members.size();
        *numNodes = newNumNodes;
        *myNodeID = (int)view.nodeId;
        _shrinkexpand_my_node      = (int)view.nodeId;
        _shrinkexpand_new_numnodes = newNumNodes;

        // Diff: assign speculative eps to their final nodeId slot, create
        // fresh eps for members not in the snapshot (other newcomers), close
        // any speculative eps to peers that turned out to be killed.
        ucxCtx.eps = (ucp_ep_h *)CmiAlloc(sizeof(ucp_ep_h) * newNumNodes);
        for (int i = 0; i < newNumNodes; i++) ucxCtx.eps[i] = nullptr;

        for (const auto &m : view.members) {
            if ((int)m.nodeId == *myNodeID) continue;
            auto it = specEps.find(m.ucxAddr);
            if (it != specEps.end()) {
                ucxCtx.eps[m.nodeId] = it->second;
                specEps.erase(it);
            } else {
                ucp_ep_params_t eParams;
                eParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
                UcxSetEpErrMode(eParams);
                eParams.address    = (const ucp_address_t *)m.ucxAddr.data();
                status = ucp_ep_create(ucxCtx.worker, &eParams,
                                       &ucxCtx.eps[m.nodeId]);
                UCX_CHECK_STATUS(status, "ucp_ep_create (newcomer post-INTEGRATE)");
            }
        }
        // Anything left in specEps is to a peer that got killed during the
        // wait window — batched force close (the peer is exiting).
        {
            std::vector<ucp_ep_h> toClose;
            toClose.reserve(specEps.size());
            for (auto &kv : specEps) toClose.push_back(kv.second);
            UcxForceCloseEps(toClose.data(), (int)toClose.size());
        }

        _coord_members = view.members;

#if CMK_SMP
        ucxCtx.txQueue = PCQueueCreate();
#endif

#if CMK_CUDA
        CpvInitialize(int, tag_counter);
        CpvAccess(tag_counter) = 0;
#endif

        // Tree-collective barrier over the post-INTEGRATE topology. Newcomer
        // joins as a (typically leaf) node at its assigned new nodeId; sends
        // ARRIVE up to its parent and waits for RELEASE. Pairs with the
        // survivor-side UcxTreeBarrier in ConverseCleanup.
        UcxTreeBarrier(static_cast<int>(view.nodeId), newNumNodes);

        CmiPrintf("Charm> newcomer integrated: nodeId=%u epoch=%u members=%d\n",
                  view.nodeId, _coord_epoch, newNumNodes);
        return;
    }
#endif

    ucp_params_t cParams;
    ucp_config_t *config;
    ucp_worker_params_t wParams;
    ucs_status_t status;
    int ret;

#if CMK_SHRINK_EXPAND
    // Coord-bootstrap mode: when the launcher passes +nodeId, +numNodes, and
    // +coordinator, the coordinator is the only source of truth for cluster
    // wireup — skip PMI entirely. This lets us launch via a thin ssh fan-out
    // (charmrun_ssh) instead of mpirun/prterun, which have the supervised-
    // daemon model that triggers job teardown when any host disappears (spot
    // termination, etc.). The PMI fallback is preserved for backward
    // compatibility with mpirun-launched runs.
    int _arg_nodeId   = -1;
    int _arg_numNodes = -1;
    char *_arg_coord  = nullptr;
    bool _coord_bootstrap_mode = false;
    CmiGetArgIntDesc(*argv, "+nodeId", &_arg_nodeId,
                     "(coord-bootstrap) this rank's node ID, set by launcher");
    CmiGetArgIntDesc(*argv, "+numNodes", &_arg_numNodes,
                     "(coord-bootstrap) total node count, set by launcher");
    // Don't consume +coordinator here — peek so the existing coord-registration
    // block can read it normally. CmiGetArgString without -consume isn't
    // available, so use the standard form; the second read below will then
    // re-extract it from the saved coordSpec_saved buffer.
    char *_coord_peek = nullptr;
    CmiGetArgStringDesc(*argv, "+coordinator", &_coord_peek,
                        "host:port of the rescale coordinator");
    if (_arg_nodeId >= 0 && _arg_numNodes > 0 && _coord_peek != nullptr) {
        _coord_bootstrap_mode = true;
        _arg_coord = _coord_peek;
        *myNodeID  = _arg_nodeId;
        *numNodes  = _arg_numNodes;
    }
    if (!_coord_bootstrap_mode)
#endif
    {
        ret = runtime_init(myNodeID, numNodes);
        UCX_CHECK_PMI_RET(ret, "runtime_init");
    }

    status = ucp_config_read("Charm++", NULL, &config);
    UCX_CHECK_STATUS(status, "ucp_config_read");

    // Initialize UCX context
    cParams.field_mask        = UCP_PARAM_FIELD_FEATURES          |
                                UCP_PARAM_FIELD_REQUEST_SIZE      |
                                UCP_PARAM_FIELD_TAG_SENDER_MASK   |
                                UCP_PARAM_FIELD_REQUEST_INIT      |
                                UCP_PARAM_FIELD_MT_WORKERS_SHARED |
                                UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    cParams.features          = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
    cParams.request_size      = sizeof(UcxRequest);
    cParams.tag_sender_mask   = 0ul;
    cParams.request_init      = UcxRequestInit;
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

    ucxCtx.numRxReqs = UCX_MSG_NUM_RX_REQS;
    if (CmiGetArgInt(*argv, "+ucx_num_rx_reqs", &ucxCtx.numRxReqs)) {
        if ((ucxCtx.numRxReqs <= 0) || (ucxCtx.numRxReqs > UCX_MSG_NUM_RX_REQS_MAX)) {
            CmiPrintf("UCX: Invalid number of RX reqs: %d\n", ucxCtx.numRxReqs);
            CmiAbort(__func__);
        }
    }

    // Eager messages should fit NcpyOperationInfo data.
    // Adjust rendezvous threshold accordingly.
    int thresh = UCX_MSG_PROBE_THRESH;
    CmiGetArgInt(*argv, "+ucx_rndv_thresh", &thresh);
    ucxCtx.eagerSize = std::max(LrtsGetMaxNcpyOperationInfoSize(), thresh);

#if CMK_SHRINK_EXPAND
    if (_coord_bootstrap_mode) {
      // Coord-bootstrap path: the launcher gave us +nodeId/+numNodes and the
      // coordinator already has every rank's UCX address. Register, collect
      // the full member list, build endpoints directly. No PMI KVS exchange,
      // no UcxInitEps.
      _coord_bootstrap_active  = true;
      _coord_bootstrap_my_node = *myNodeID;

      char *colon = strchr(_arg_coord, ':');
      if (colon == nullptr) CmiAbort("UCX: +coordinator must be host:port");
      *colon = '\0';
      _coord_host = _arg_coord;
      _coord_port = atoi(colon + 1);
      if (_coord_port <= 0) CmiAbort("UCX: +coordinator port invalid");

      ucp_address_t *myAddr = nullptr;
      size_t myAddrLen = 0;
      status = ucp_worker_get_address(ucxCtx.worker, &myAddr, &myAddrLen);
      UCX_CHECK_STATUS(status,
                       "ucp_worker_get_address (coord-bootstrap REGISTER)");

      _coord_fd = coord::connect_blocking(_coord_host, _coord_port);
      if (_coord_fd < 0) {
        CmiAbort("UCX: failed to connect to coordinator (coord-bootstrap)");
      }
      coord::ClusterView view;
      if (!coord::register_initial(_coord_fd, (uint32_t)*myNodeID,
                                   myAddr, (uint32_t)myAddrLen, &view)) {
        CmiAbort("UCX: coordinator REGISTER_INITIAL failed (coord-bootstrap)");
      }
      _coord_epoch   = view.epoch;
      _coord_members = view.members;
      ucp_worker_release_address(ucxCtx.worker, myAddr);

      if ((int)view.nodeId != *myNodeID) {
        CmiAbort("UCX: coordinator returned nodeId mismatching launcher rank "
                 "(coord-bootstrap)");
      }
      if ((int)view.members.size() != *numNodes) {
        CmiAbort("UCX: coordinator returned %d members, launcher said %d "
                 "(coord-bootstrap)",
                 (int)view.members.size(), *numNodes);
      }

      ucxCtx.eps = (ucp_ep_h *)CmiAlloc(sizeof(ucp_ep_h) * (*numNodes));
      CmiEnforce(ucxCtx.eps);
      for (int i = 0; i < *numNodes; i++) ucxCtx.eps[i] = nullptr;
      for (const auto &m : view.members) {
        if ((int)m.nodeId == *myNodeID) continue;
        ucp_ep_params_t eParams;
        eParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        UcxSetEpErrMode(eParams);
        eParams.address    = (const ucp_address_t *)m.ucxAddr.data();
        status = ucp_ep_create(ucxCtx.worker, &eParams, &ucxCtx.eps[m.nodeId]);
        UCX_CHECK_STATUS(status, "ucp_ep_create (coord-bootstrap)");
      }

      if (*myNodeID == 0) {
        CmiPrintf("Charm> coordinator registered (coord-bootstrap): "
                  "nodeId=%u epoch=%u members=%zu\n",
                  view.nodeId, view.epoch, view.members.size());
      }
    } else
#endif
    {
      UcxInitEps(*numNodes, *myNodeID);
    }

    UcxPrepostRxBuffers();

    // Ensure connects completion
    status = ucp_worker_flush(ucxCtx.worker);
    UCX_CHECK_STATUS(status, "ucp_worker_flush");

#if CMK_SHRINK_EXPAND
    // PMI bootstrap path still needs to register with the coordinator (so the
    // coord has a complete membership view even on runs that never rescale).
    // The coord-bootstrap branch above already did this.
    if (!_coord_bootstrap_mode) {
      // +coordinator was already consumed from argv by the bootstrap-mode
      // peek above; reuse that value rather than re-parsing.
      char *coordSpec = _coord_peek;
      if (coordSpec == nullptr) {
        CmiAbort("UCX: +coordinator host:port is required (shrink/expand build).");
      }
      char *colon = strchr(coordSpec, ':');
      if (colon == nullptr) CmiAbort("UCX: +coordinator must be host:port");
      *colon = '\0';
      _coord_host = coordSpec;
      _coord_port = atoi(colon + 1);
      if (_coord_port <= 0) CmiAbort("UCX: +coordinator port invalid");

      // Get our local UCX worker address to publish.
      ucp_address_t *myAddr = nullptr;
      size_t myAddrLen = 0;
      status = ucp_worker_get_address(ucxCtx.worker, &myAddr, &myAddrLen);
      UCX_CHECK_STATUS(status, "ucp_worker_get_address (coordinator REGISTER)");

      _coord_fd = coord::connect_blocking(_coord_host, _coord_port);
      if (_coord_fd < 0) {
        CmiAbort("UCX: failed to connect to coordinator");
      }
      coord::ClusterView view;
      if (!coord::register_initial(_coord_fd, (uint32_t)*myNodeID,
                                   myAddr, (uint32_t)myAddrLen, &view)) {
        CmiAbort("UCX: coordinator REGISTER_INITIAL failed");
      }
      _coord_epoch = view.epoch;
      _rescaleGeneration = (int)view.epoch;
      _coord_members = view.members;
      ucp_worker_release_address(ucxCtx.worker, myAddr);

      if ((int)view.nodeId != *myNodeID) {
        CmiAbort("UCX: coordinator returned nodeId mismatching PMI rank");
      }
      if (*myNodeID == 0) {
        CmiPrintf("Charm> coordinator registered: nodeId=%u epoch=%u members=%zu\n",
                  view.nodeId, view.epoch, view.members.size());
      }
    }
#endif

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

static inline UcxRequest* UcxPostRxReqInternal(ucp_tag_t tag, size_t size,
                                               ucp_tag_message_h msg)
{
    void *buf = CmiAlloc(size);
    UcxRequest *req;

    if (tag == UCX_MSG_TAG_EAGER) {
        req = (UcxRequest*)ucp_tag_recv_nb(ucxCtx.worker, buf,
                                           ucxCtx.eagerSize,
                                           ucp_dt_make_contig(1), tag,
                                           UCX_MSG_TAG_MASK,
                                           UcxRxReqCompleted);
    } else {
        CmiEnforce(tag == UCX_MSG_TAG_PROBE);
        req = (UcxRequest*)ucp_tag_msg_recv_nb(ucxCtx.worker, buf, size,
                                               ucp_dt_make_contig(1), msg,
                                               UcxRxReqCompleted);
    }

    CmiEnforce(!UCS_PTR_IS_ERR(req));
    UCX_LOG(3, "Posted RX buf %p size %zu, req %p, tag %zu, comp %d\n",
            req->msgBuf, size, req, tag, req->completed);

    // Request completed immediately
    if (req->completed) {
        if (!(tag & UCX_RMA_TAG_MASK)) {
            handleOneRecvedMsg(size, (char*)buf);
        }
    } else {
        req->msgBuf = buf;
    }

    return req;
}

static inline UcxRequest* UcxPostRxReq(ucp_tag_t tag, size_t size,
                                       ucp_tag_message_h msg)
{
    UcxRequest *req = UcxPostRxReqInternal(tag, size, msg);
    int idx = req->idx;

    do {
        if (req->completed) {
            UCX_REQUEST_FREE(req);

            if (tag & UCX_MSG_TAG_EAGER) {
                req = UcxPostRxReqInternal(UCX_MSG_TAG_EAGER, ucxCtx.eagerSize, NULL);
                req->idx = idx;
                ucxCtx.rxReqs[idx] = req;
            } else {
                return NULL;
            }
        }
        else {
            return req;
        }
    }
    while (1);
}

static inline UcxRequest* UcxHandleRxReq(UcxRequest *request, char *rxBuf,
                                         size_t size, ucp_tag_t tag, int idx)
{
    if (!(tag & UCX_RMA_TAG_MASK)) {
        handleOneRecvedMsg(size, rxBuf);
    }

    UCX_REQUEST_FREE(request);

    if (tag & UCX_MSG_TAG_EAGER) {
        ucxCtx.rxReqs[idx]      = UcxPostRxReq(UCX_MSG_TAG_EAGER,
                                               ucxCtx.eagerSize, NULL);
        ucxCtx.rxReqs[idx]->idx = idx;
        return ucxCtx.rxReqs[idx];
    }

    return NULL;
}

static void UcxRxReqCompleted(void *request, ucs_status_t status,
                              ucp_tag_recv_info_t *info)
{
    UcxRequest *req = (UcxRequest*)request;

    UCX_LOG(3, "status %d len %zu, buf %p, req %p, tag %zu\n",
            status,  info->length, req->msgBuf, request, info->sender_tag);

    if (ucs_unlikely(status == UCS_ERR_CANCELED)) {
        return;
    }

#if CMK_ONESIDED_IMPL
    if (info->sender_tag & UCX_RMA_TAG_REG_AND_SEND_BACK) {

        // Register the source buffer and send back to destination to perform GET

        NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(req->msgBuf);
        UCX_LOG(4, "Got ncpy size %zu (meta size %d)", ncpyOpInfo->srcSize, ncpyOpInfo->ncpyOpInfoSize);
        resetNcpyOpInfoPointers(ncpyOpInfo);

        UcxRdmaInfo *info = (UcxRdmaInfo *)(ncpyOpInfo->srcLayerInfo + CmiGetRdmaCommonInfoSize());

        UcxMemMap(info,
                  (void *)ncpyOpInfo->srcPtr,
                  ncpyOpInfo->srcSize);

        ncpyOpInfo->isSrcRegistered = 1;

        ncpyOpInfo->freeMe = CMK_FREE_NCPYOPINFO; // It's a message, not a realy ncpy Obj
        UCX_LOG(4, "Reset ncpy size %zu (meta size %d)", ncpyOpInfo->destSize, ncpyOpInfo->ncpyOpInfoSize);

        // send back to destination process to perform GET
        UcxSendMsg(CmiNodeOf(ncpyOpInfo->destPe), ncpyOpInfo->destPe,
                   ncpyOpInfo->ncpyOpInfoSize, (char*)ncpyOpInfo,
                   UCX_RMA_TAG_GET, UcxRmaSendCompletedAndFree);

    } else if (info->sender_tag & UCX_RMA_TAG_GET) {
        NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(req->msgBuf);
        resetNcpyOpInfoPointers(ncpyOpInfo);

        ncpyOpInfo->freeMe = CMK_FREE_NCPYOPINFO; // It's a message, not a real ncpy Obj
        UcxRmaOp(ncpyOpInfo, UCX_RMA_OP_GET);

    } else if (info->sender_tag & UCX_RMA_TAG_DEREG_AND_ACK) {
        NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(req->msgBuf);
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
    }
#endif

    if (req->msgBuf != NULL) {
        // Request is not completed immediately
        UcxHandleRxReq(req, (char*)req->msgBuf, info->length, info->sender_tag, req->idx);
    } else {
        req->completed = 1;
    }
}

static void UcxPrepostRxBuffers()
{
    int i;

    ucxCtx.rxReqs = (UcxRequest**)CmiAlloc(sizeof(UcxRequest*) * ucxCtx.numRxReqs);

    for (i = 0; i < ucxCtx.numRxReqs; i++) {
        ucxCtx.rxReqs[i] = UcxPostRxReq(UCX_MSG_TAG_EAGER, ucxCtx.eagerSize, NULL);
        ucxCtx.rxReqs[i]->idx = i;
    }
    UCX_LOG(3, "UCX: preposted %d rx requests", ucxCtx.numRxReqs);
}

void UcxTxReqCompleted(void *request, ucs_status_t status)
{
    UcxRequest *req = (UcxRequest*)request;

    CmiEnforce(status == UCS_OK);
    CmiEnforce(req->msgBuf);

    UCX_LOG(3, "TX req %p completed, free msg %p", req, req->msgBuf);
    CmiFree(req->msgBuf);
    UCX_REQUEST_FREE(req);
}

// tag may carry RMA tag
inline void* UcxSendMsg(int destNode, int destPE, int size, char *msg,
                        ucp_tag_t tag, ucp_send_callback_t cb)
{
    ucp_tag_t sTag;

#if CMK_SHRINK_EXPAND
    // Fail fast on sends to a node that does not exist in the current view
    // (e.g. a stale location entry naming a departed PE after a shrink).
    // Such sends would otherwise index past the endpoint table or target a
    // closed endpoint and vanish silently, starving the cluster.
    if (destNode < 0 || destNode >= _Cmi_numnodes ||
        ucxCtx.eps[destNode] == NULL) {
        CmiAbort("UCX: send to invalid node %d (numnodes=%d, pe=%d, size=%d)",
                 destNode, _Cmi_numnodes, destPE, size);
    }
#endif

    // Combine tag and sTag: sTag defines msg protocol, tag may indicate RMA requests
    sTag  = (size > ucxCtx.eagerSize) ? UCX_MSG_TAG_PROBE : UCX_MSG_TAG_EAGER;

    // Auxilliary messages (which add bits to the tag) should use eager.
    CmiEnforce((tag == 0ul) || (sTag == UCX_MSG_TAG_EAGER));

    sTag |= tag;

    UCX_LOG(3, "destNode=%i destPE=%i size=%i msg=%p, tag=%" PRIu64,
            destNode, destPE, size, msg, tag);
#if CMK_SMP
    UcxPendingRequest *req = (UcxPendingRequest*)CmiAlloc(sizeof(UcxPendingRequest));
    req->msgBuf = msg;
    req->size   = size;
    req->tag    = sTag;
    req->dNode  = destNode;
    req->cb     = cb;
    req->op     = UCX_SEND_OP;   // Mark this request as a regular message (UCX_SEND_OP)

    UCX_LOG(3, " --> (PE=%i) enq msg (queue depth=%i), dNode %i, size %i",
            CmiMyPe(), PCQueueLength(ucxCtx.txQueue), destNode, size);
    PCQueuePush(ucxCtx.txQueue, (char *)req);
#else
    UcxRequest *req;

    req = (UcxRequest*)ucp_tag_send_nb(ucxCtx.eps[destNode], msg, size,
                                       ucp_dt_make_contig(1), sTag, cb);
    if (!UCS_PTR_IS_PTR(req)) {
        CmiEnforce(!UCS_PTR_IS_ERR(req));
        return NULL;
    }

    req->msgBuf = msg;
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

    req = UcxSendMsg(destNode, destPE, size, msg, 0ul, UcxTxReqCompleted);
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
    if (req)
    {
        if(req->op == UCX_SEND_OP) { // Regular Message
            ucs_status_ptr_t status_ptr;
            status_ptr = ucp_tag_send_nb(ucxCtx.eps[req->dNode], req->msgBuf,
                                         req->size, ucp_dt_make_contig(1),
                                         req->tag, req->cb);

            if (!UCS_PTR_IS_PTR(status_ptr)) {
                CmiEnforce(!UCS_PTR_IS_ERR(status_ptr));

                if(req->tag & UCX_RMA_TAG_MASK) {
                    NcpyOperationInfo *ncpyOpInfo = (NcpyOperationInfo *)(req->msgBuf);
                    if(ncpyOpInfo->freeMe == CMK_FREE_NCPYOPINFO)
                        CmiFree(ncpyOpInfo);
                } else {
                    CmiFree(req->msgBuf);
                }
            } else {
                ((UcxRequest*)status_ptr)->msgBuf = req->msgBuf;
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

       // Probe with full tag mask to avoid long traversing thru unexpected
       // queue of eager messages (messages with non-full mask added to the
       // same unexpected queue)
       msg = ucp_tag_probe_nb(ucxCtx.worker, UCX_MSG_TAG_PROBE,
                              UCX_MSG_TAG_MASK_FULL, 1, &info);
       if (msg != NULL) {
           UCX_LOG(3, "Got msg %p, len %zu\n", msg, info.length);
           UcxPostRxReq(UCX_MSG_TAG_PROBE, info.length, msg);
       }

#if CMK_SMP
       cnt += ProcessTxQueue();
#endif
    } while (cnt);
}

void LrtsDrainResources()
{
    int ret;
    LrtsAdvanceCommunication(0);
#if CMK_SHRINK_EXPAND
    if (_coord_bootstrap_active) {
      // No PMI session — use the coordinator's all-ranks barrier instead.
      if (!coord::barrier(_coord_fd, _coord_epoch,
                          (uint32_t)_coord_bootstrap_my_node)) {
        CmiAbort("coord::barrier failed in LrtsDrainResources");
      }
      return;
    }
#endif
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

    for (i = 0; i < ucxCtx.numRxReqs; ++i) {
        req = ucxCtx.rxReqs[i];
        CmiFree(req->msgBuf);
        ucp_request_cancel(ucxCtx.worker, req);
        ucp_request_free(req);
    }

    ucp_worker_destroy(ucxCtx.worker);
    ucp_cleanup(ucxCtx.context);

    CmiFree(ucxCtx.eps);
    CmiFree(ucxCtx.rxReqs);
#if CMK_SMP
    PCQueueDestroy(ucxCtx.txQueue);
#endif

    if(!CharmLibInterOperate || userDrivenMode) {
#if CMK_SHRINK_EXPAND
        if (_coord_bootstrap_active) {
          // No PMI — coord barrier + clean socket close. The coord itself
          // tolerates clients dropping (it's not part of the supervised
          // launch tree), so nothing else needed.
          if (!coord::barrier(_coord_fd, _coord_epoch,
                              (uint32_t)_coord_bootstrap_my_node)) {
            // Best-effort: don't abort at shutdown if peers already left.
            CmiPrintf("Charm> coord::barrier failed at LrtsExit (ignored)\n");
          }
          if (_coord_fd >= 0) ::close(_coord_fd);
          if (!userDrivenMode) {
            exit(exitcode);
          }
        } else
#endif
        {
          ret = runtime_barrier();
          UCX_CHECK_PMI_RET(ret, "runtime_barrier");

          ret = runtime_fini();
          UCX_CHECK_PMI_RET(ret, "runtime_fini");
          if (!userDrivenMode) {
            exit(exitcode);
          }
        }
    }
}

void LrtsCleanup()
{
  int ret;
    int i;
    UcxRequest *req;
    ucs_status_t status;

    UCX_LOG(4, "LrtsExit");

    LrtsAdvanceCommunication(0);

    for (i = 0; i < ucxCtx.numRxReqs; ++i) {
        req = ucxCtx.rxReqs[i];
        CmiFree(req->msgBuf);
        ucp_request_cancel(ucxCtx.worker, req);
        ucp_request_free(req);
    }

    ucp_worker_destroy(ucxCtx.worker);
    ucp_cleanup(ucxCtx.context);

    CmiFree(ucxCtx.eps);
    CmiFree(ucxCtx.rxReqs);
#if CMK_SMP
    PCQueueDestroy(ucxCtx.txQueue);
#endif

    if(!CharmLibInterOperate || userDrivenMode) {
#if CMK_SHRINK_EXPAND
        if (_coord_bootstrap_active) {
          if (!coord::barrier(_coord_fd, _coord_epoch,
                              (uint32_t)_coord_bootstrap_my_node)) {
            CmiPrintf("Charm> coord::barrier failed at LrtsCleanup (ignored)\n");
          }
          if (_coord_fd >= 0) ::close(_coord_fd);
        } else
#endif
        {
          ret = runtime_barrier();
          UCX_CHECK_PMI_RET(ret, "runtime_barrier");

          ret = runtime_fini();
          UCX_CHECK_PMI_RET(ret, "runtime_fini");
        }
    }
}

void LrtsCleanup()
{
  int ret;
    int i;
    UcxRequest *req;
    ucs_status_t status;

    UCX_LOG(4, "LrtsExit");

    LrtsAdvanceCommunication(0);

    for (i = 0; i < ucxCtx.numRxReqs; ++i) {
        req = ucxCtx.rxReqs[i];
        CmiFree(req->msgBuf);
        ucp_request_cancel(ucxCtx.worker, req);
        ucp_request_free(req);
    }

    ucp_worker_destroy(ucxCtx.worker);
    ucp_cleanup(ucxCtx.context);

    CmiFree(ucxCtx.eps);
    CmiFree(ucxCtx.rxReqs);
#if CMK_SMP
    PCQueueDestroy(ucxCtx.txQueue);
#endif

    if(!CharmLibInterOperate || userDrivenMode) {
        ret = runtime_barrier();
        UCX_CHECK_PMI_RET(ret, "runtime_barrier");

        ret = runtime_fini();
        UCX_CHECK_PMI_RET(ret, "runtime_fini");
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


#if CMK_SHRINK_EXPAND
extern char *se_avail_vector;  // populated on PE 0 by ck-ldb/manager.C realloc()
extern std::vector<char> se_avail_snapshot;  // durable copy, same origin

// Close one endpoint cleanly (flush mode) and wait for completion. UCX returns
// a request handle that needs to be progressed to OK before we drop the worker.
static void UcxCloseEp(ucp_ep_h ep)
{
    ucs_status_ptr_t req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
    if (req == NULL) return;
    if (UCS_PTR_IS_ERR(req)) {
        UCX_LOG(50, "ucp_ep_close_nb failed: %s",
                ucs_status_string(UCS_PTR_STATUS(req)));
        return;
    }
    ucs_status_t st;
    do {
        ucp_worker_progress(ucxCtx.worker);
        st = ucp_request_check_status(req);
    } while (st == UCS_INPROGRESS);
    ucp_request_free(req);
}

// Batch-close endpoints to removed members. The peers received DIE and are
// exiting, and all traffic to them was drained before the commit, so FORCE
// mode applies: no flush or disconnect handshake with a peer that may
// already be gone. All closes are initiated first, then progressed
// together, so n closes cost roughly the slowest one rather than the sum.
static void UcxForceCloseEps(ucp_ep_h *eps, int n)
{
    std::vector<ucs_status_ptr_t> reqs;
    reqs.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (eps[i] == nullptr) continue;
        ucs_status_ptr_t req = ucp_ep_close_nb(eps[i], UCP_EP_CLOSE_MODE_FORCE);
        if (UCS_PTR_IS_ERR(req)) {
            // FORCE requires endpoints created with PEER error-handling
            // mode; on transports without it UCX rejects the request
            // inline and the ep is untouched. Fall back to a flush-mode
            // close, still initiated without waiting so all closes
            // progress together below.
            req = ucp_ep_close_nb(eps[i], UCP_EP_CLOSE_MODE_FLUSH);
            if (UCS_PTR_IS_ERR(req)) {
                UCX_LOG(50, "ucp_ep_close_nb failed: %s",
                        ucs_status_string(UCS_PTR_STATUS(req)));
                continue;
            }
        }
        if (req == NULL) continue;
        reqs.push_back(req);
    }
    bool pending = !reqs.empty();
    while (pending) {
        ucp_worker_progress(ucxCtx.worker);
        pending = false;
        for (auto &req : reqs) {
            if (req == nullptr) continue;
            ucs_status_t st = ucp_request_check_status(req);
            if (st == UCS_INPROGRESS) {
                pending = true;
            } else {
                ucp_request_free(req);
                req = nullptr;
            }
        }
    }
}

// Empty no-op callbacks for the control-plane reconfig send/recv. We poll
// for completion via ucp_request_check_status; the callback only fires if
// the request didn't complete inline.
static void UcxReconfigSendCb(void *req, ucs_status_t status) { (void)req; (void)status; }
static void UcxReconfigRecvCb(void *req, ucs_status_t status,
                              ucp_tag_recv_info_t *info)
{
    (void)req; (void)status; (void)info;
}

// Synchronous reconfig send over an existing OLD-numbering UCX endpoint.
// Used by PE 0 (and by each chain forwarder) during ConverseCleanup, when
// regular Charm traffic is already drained. Blocks until UCX reports
// completion.
static void UcxSendReconfigBytes(int destNodeOldId, const uint8_t *data, size_t size)
{
    ucs_status_ptr_t req = ucp_tag_send_nb(ucxCtx.eps[destNodeOldId], data, size,
                                           ucp_dt_make_contig(1),
                                           UCX_MSG_TAG_RECONFIG,
                                           UcxReconfigSendCb);
    if (UCS_PTR_IS_ERR(req)) {
        CmiAbort("UCX: ucp_tag_send_nb(RECONFIG) to nodeId=%d failed: %s",
                 destNodeOldId, ucs_status_string(UCS_PTR_STATUS(req)));
    }
    if (req == NULL) return;  // completed inline
    ucs_status_t st;
    do {
        ucp_worker_progress(ucxCtx.worker);
        st = ucp_request_check_status(req);
    } while (st == UCS_INPROGRESS);
    if (st != UCS_OK) {
        CmiAbort("UCX: RECONFIG send to nodeId=%d errored: %s",
                 destNodeOldId, ucs_status_string(st));
    }
    ucp_request_free(req);
}

// Probe + receive the reconfig broadcast on a survivor. Returns the raw
// payload bytes (delta-shape; see ConverseCleanup for the layout). Tag
// matches strictly on UCX_MSG_TAG_RECONFIG within the MSG tag range so it
// can't collide with eager/probe traffic.
static std::vector<uint8_t> UcxRecvReconfigBytes(ucp_tag_message_h msgHandle,
                                                  size_t length)
{
    std::vector<uint8_t> buf(length);
    ucs_status_ptr_t req = ucp_tag_msg_recv_nb(ucxCtx.worker, buf.data(), length,
                                               ucp_dt_make_contig(1), msgHandle,
                                               UcxReconfigRecvCb);
    if (UCS_PTR_IS_ERR(req)) {
        CmiAbort("UCX: ucp_tag_msg_recv_nb(RECONFIG) failed: %s",
                 ucs_status_string(UCS_PTR_STATUS(req)));
    }
    if (req == NULL) return buf;  // completed inline
    ucs_status_t st;
    do {
        ucp_worker_progress(ucxCtx.worker);
        st = ucp_request_check_status(req);
    } while (st == UCS_INPROGRESS);
    if (st != UCS_OK) {
        CmiAbort("UCX: RECONFIG recv errored: %s", ucs_status_string(st));
    }
    ucp_request_free(req);
    return buf;
}

// Compute this rank's children in the binary-tree broadcast over surviving
// OLD nodeIds. Survivors are conceptually re-indexed 0..S-1 (skipping the
// kill set, preserving relative order), and each survivor index s sends to
// children at 2s+1 and 2s+2. Returns each child's OLD nodeId (or -1 if not
// present). myOldId is assumed to be a survivor — leaves get two -1s.
//
// O(N) per call where N is the cluster size (10s..1000s); runs once per
// participating rank per rescale, well below the cost of even one UCX
// endpoint create. Critical path is log2(S) hops instead of S-1.
static void UcxReconfigTreeChildren(int oldNumNodes, int myOldId,
                                     const std::vector<uint32_t> &killSet,
                                     int *leftChildOldId, int *rightChildOldId)
{
    *leftChildOldId = -1;
    *rightChildOldId = -1;
    std::vector<int> survivors;
    survivors.reserve(oldNumNodes);
    for (int i = 0; i < oldNumNodes; ++i) {
        bool killed = false;
        for (uint32_t k : killSet) {
            if ((int)k == i) { killed = true; break; }
        }
        if (!killed) survivors.push_back(i);
    }
    int myIdx = -1;
    for (int i = 0; i < (int)survivors.size(); ++i) {
        if (survivors[i] == myOldId) { myIdx = i; break; }
    }
    if (myIdx < 0) return;  // caller is not a survivor (shouldn't happen here)
    int li = 2 * myIdx + 1;
    int ri = 2 * myIdx + 2;
    if (li < (int)survivors.size()) *leftChildOldId = survivors[li];
    if (ri < (int)survivors.size()) *rightChildOldId = survivors[ri];
}

// Serialize the reconfig delta payload. NodeId is intentionally omitted —
// every receiver computes its own new nodeId from killSet (new = old - count
// of killed ids < self).
static std::vector<uint8_t> UcxBuildReconfigPayload(
    uint32_t epoch,
    const std::vector<uint32_t> &killSet,
    const std::vector<coord::Member> &added)
{
    std::vector<uint8_t> buf;
    coord::put_u32(buf, epoch);
    coord::put_u32_vec(buf, killSet);
    coord::put_members(buf, added);
    return buf;
}

// Forward the reconfig payload to this rank's binary-tree children over
// existing OLD-numbering UCX endpoints. PE 0 calls this after coord::commit;
// each survivor calls it before rebuilding its eps. No-op for leaves.
static void UcxReconfigTreeForward(int oldNumNodes, int myOldId,
                                    const std::vector<uint32_t> &killSet,
                                    const std::vector<uint8_t> &payload)
{
    int leftChild, rightChild;
    UcxReconfigTreeChildren(oldNumNodes, myOldId, killSet, &leftChild, &rightChild);
    if (leftChild >= 0) {
        UCX_LOG(3, "RECONFIG tree forward: oldId=%d -> oldId=%d (L, size=%zu)",
                myOldId, leftChild, payload.size());
        UcxSendReconfigBytes(leftChild, payload.data(), payload.size());
    }
    if (rightChild >= 0) {
        UCX_LOG(3, "RECONFIG tree forward: oldId=%d -> oldId=%d (R, size=%zu)",
                myOldId, rightChild, payload.size());
        UcxSendReconfigBytes(rightChild, payload.data(), payload.size());
    }
}

// Empty-payload, synchronous tag send over a NEW-numbering UCX endpoint.
// Used by the tree-collective barrier; the payload is implicit in the tag.
static void UcxSendEmptyTagSync(int destNewNodeId, ucp_tag_t tag)
{
    ucs_status_ptr_t req = ucp_tag_send_nb(ucxCtx.eps[destNewNodeId], NULL, 0,
                                           ucp_dt_make_contig(1), tag,
                                           UcxReconfigSendCb);
    if (UCS_PTR_IS_ERR(req)) {
        CmiAbort("UCX: ucp_tag_send_nb(tag=%" PRIx64 ") to nodeId=%d failed: %s",
                 (uint64_t)tag, destNewNodeId,
                 ucs_status_string(UCS_PTR_STATUS(req)));
    }
    if (req == NULL) return;  // completed inline
    ucs_status_t st;
    do {
        ucp_worker_progress(ucxCtx.worker);
        st = ucp_request_check_status(req);
    } while (st == UCS_INPROGRESS);
    if (st != UCS_OK) {
        CmiAbort("UCX: tag-send (tag=%" PRIx64 ") to nodeId=%d errored: %s",
                 (uint64_t)tag, destNewNodeId, ucs_status_string(st));
    }
    ucp_request_free(req);
}

// Probe + drain an empty-payload tag message. Spins on worker_progress until
// matched; the message is consumed (payload discarded since it's empty).
static void UcxRecvEmptyTagSync(ucp_tag_t tag)
{
    ucp_tag_message_h msgh = NULL;
    ucp_tag_recv_info_t info;
    while (msgh == NULL) {
        msgh = ucp_tag_probe_nb(ucxCtx.worker, tag, UCX_MSG_TAG_MASK, 1, &info);
        if (msgh == NULL) ucp_worker_progress(ucxCtx.worker);
    }
    // info.length is 0; pass a small stack buffer just in case.
    char dummy;
    ucs_status_ptr_t req = ucp_tag_msg_recv_nb(ucxCtx.worker, &dummy, 0,
                                               ucp_dt_make_contig(1), msgh,
                                               UcxReconfigRecvCb);
    if (UCS_PTR_IS_ERR(req)) {
        CmiAbort("UCX: ucp_tag_msg_recv_nb(tag=%" PRIx64 ") failed: %s",
                 (uint64_t)tag, ucs_status_string(UCS_PTR_STATUS(req)));
    }
    if (req == NULL) return;  // completed inline
    ucs_status_t st;
    do {
        ucp_worker_progress(ucxCtx.worker);
        st = ucp_request_check_status(req);
    } while (st == UCS_INPROGRESS);
    if (st != UCS_OK) {
        CmiAbort("UCX: tag-recv (tag=%" PRIx64 ") errored: %s",
                 (uint64_t)tag, ucs_status_string(st));
    }
    ucp_request_free(req);
}

// Tree-collective barrier over NEW-numbering UCX endpoints. Replaces the
// coordinator's O(N) TCP star with two O(log N) tree passes over the same
// binary tree shape used by the RECONFIG broadcast (rooted at NEW nodeId 0,
// children of n are 2n+1 and 2n+2, parent is (n-1)/2).
//
// Phase 1 (up-reduce, ARRIVE): each rank waits for ARRIVE from each child,
// then sends ARRIVE to its parent. When PE 0 has received ARRIVE from both
// of its children, every rank has finished UcxReInitEpsFromView.
//
// Phase 2 (down-broadcast, RELEASE): root sends RELEASE to its children;
// each rank forwards RELEASE to its children before returning. When a rank
// returns, it's guaranteed every rank has hit the barrier — same guarantee
// as the old coord-mediated barrier.
//
// Coordinator does zero work for this barrier. Critical path is
// 2 * ceil(log2(N)) UCX hops instead of 2N TCP frames serialized on coord.
static void UcxTreeBarrier(int newNodeId, int numNodes)
{
    int parent = (newNodeId == 0) ? -1 : (newNodeId - 1) / 2;
    int leftChild  = (2 * newNodeId + 1 < numNodes) ? (2 * newNodeId + 1) : -1;
    int rightChild = (2 * newNodeId + 2 < numNodes) ? (2 * newNodeId + 2) : -1;

    // Up-reduce
    if (leftChild  >= 0) UcxRecvEmptyTagSync(UCX_MSG_TAG_BARRIER_ARRIVE);
    if (rightChild >= 0) UcxRecvEmptyTagSync(UCX_MSG_TAG_BARRIER_ARRIVE);
    if (parent >= 0)     UcxSendEmptyTagSync(parent, UCX_MSG_TAG_BARRIER_ARRIVE);

    // Down-broadcast
    if (parent >= 0)     UcxRecvEmptyTagSync(UCX_MSG_TAG_BARRIER_RELEASE);
    if (leftChild  >= 0) UcxSendEmptyTagSync(leftChild,  UCX_MSG_TAG_BARRIER_RELEASE);
    if (rightChild >= 0) UcxSendEmptyTagSync(rightChild, UCX_MSG_TAG_BARRIER_RELEASE);
}

// Coordinator-driven endpoint reconfig. PE 0 already drove COMMIT and pushed
// RECONFIG to other survivors; this just consumes the new view (passed in)
// and updates the eps array.
//
// Diff-based: surviving peers are identified by ucxAddr. Their existing eps
// are reused (and just remapped to the new compactly-renumbered nodeId), so
// no UCX handshake is repeated. Killed peers' eps are closed; newcomers get
// fresh eps. _coord_members is the OLD view we're diffing against.
//
// oldNodeId is this rank's nodeId in the old numbering — needed to skip the
// self slot in ucxCtx.eps (which UcxInitEps left uninitialized).
static void UcxReInitEpsFromView(const coord::ClusterView &view,
                                 int oldNumNodes, int oldNodeId)
{
    ucs_status_t status;

    std::unordered_map<std::string, int> oldByAddr;
    oldByAddr.reserve(_coord_members.size());
    for (const auto &m : _coord_members) {
        if (static_cast<int>(m.nodeId) == oldNodeId) continue;
        oldByAddr.emplace(m.ucxAddr, static_cast<int>(m.nodeId));
    }

    int newNumNodes = static_cast<int>(view.members.size());
    ucp_ep_h *newEps = (ucp_ep_h*)CmiAlloc(sizeof(ucp_ep_h) * newNumNodes);
    CmiEnforce(newEps);
    for (int i = 0; i < newNumNodes; ++i) newEps[i] = nullptr;

    std::vector<bool> oldUsed(oldNumNodes, false);

    for (const auto &m : view.members) {
        if (static_cast<int>(m.nodeId) == static_cast<int>(view.nodeId)) continue;
        auto it = oldByAddr.find(m.ucxAddr);
        if (it != oldByAddr.end()) {
            int oldId = it->second;
            newEps[m.nodeId] = ucxCtx.eps[oldId];
            oldUsed[oldId] = true;
            UCX_LOG(4, "Reusing ep: oldId=%d -> newId=%u (ep %p)",
                    oldId, m.nodeId, newEps[m.nodeId]);
        } else {
            ucp_ep_params_t eParams;
            eParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
            UcxSetEpErrMode(eParams);
            eParams.address    = reinterpret_cast<const ucp_address_t*>(m.ucxAddr.data());
            status = ucp_ep_create(ucxCtx.worker, &eParams, &newEps[m.nodeId]);
            UCX_CHECK_STATUS(status, "ucp_ep_create (reconfig newcomer)");
            UCX_LOG(4, "New ep to nodeId=%u (ep %p)", m.nodeId, newEps[m.nodeId]);
        }
    }

    // Close eps to killed peers (anything in old eps that didn't get reused,
    // skipping the self slot which was never initialized). Batched force
    // close: the peers are exiting and traffic to them was drained before
    // the commit.
    {
        std::vector<ucp_ep_h> toClose;
        for (int i = 0; i < oldNumNodes; ++i) {
            if (i == oldNodeId) continue;
            if (oldUsed[i]) continue;
            toClose.push_back(ucxCtx.eps[i]);
        }
        UcxForceCloseEps(toClose.data(), (int)toClose.size());
    }

    CmiFree(ucxCtx.eps);
    ucxCtx.eps = newEps;

    // Receives are tag-matched and not bound to specific eps; leave the
    // preposted recv pool alone. Just flush so handshakes for newly-added
    // eps complete before the LB step starts driving traffic.
    status = ucp_worker_flush(ucxCtx.worker);
    UCX_CHECK_STATUS(status, "ucp_worker_flush (reconfig)");

    _coord_members = view.members;
}

void ConverseCleanup(void)
{
  MACHSTATE(2,"ConverseCleanup {");

#if CMK_SHRINK_EXPAND
  {
    extern double rescale_t_cleanup_enter;
    extern double rescale_wall_now();
    if (CmiMyPe() == 0) rescale_t_cleanup_enter = rescale_wall_now();
  }
#endif

  CmiBarrier();

#if CMK_USE_SYSVSHM
	CmiExitSysvshm();
#elif CMK_USE_PXSHM
	CmiExitPxshm();
#endif
  ConverseCommonExit();
  CmiNodeBarrier();

  if (get_shrinkexpand_exit()) {
    int oldNumNodes = _Cmi_numnodes;
    int myNode = CmiMyNode();

    // Drain any in-flight messages before we touch the coordinator/eps.
    LrtsAdvanceCommunication(0);

    // PE 0 writes new PE count for charmrun_elastic compatibility (legacy).
    // Coordinator is the source of truth for the new shape; this is a hint.
    if (CmiMyPe() == 0) {
      std::string path = std::string(_shrinkexpand_basedir) + "/numRestartProcs.txt";
      FILE *fp = fopen(path.c_str(), "w");
      if (fp != NULL) {
        fprintf(fp, "%d", numProcessAfterRestart);
        fclose(fp);
      }
    }

    coord::ClusterView view;
    bool gotDie = false;

    if (CmiMyPe() == 0) {
      // PE 0 drives the COMMIT. Build kill set from se_avail_vector
      // (in OLD nodeId space — entries with value 0 are being killed).
      if (se_avail_snapshot.size() < (size_t)oldNumNodes) {
        CmiAbort("UCX: shrink/expand exit on PE 0 with missing avail snapshot");
      }
      std::vector<uint32_t> kills;
      int survivors = 0;
      for (int i = 0; i < oldNumNodes; ++i) {
        if (se_avail_snapshot[i]) survivors++;
        else kills.push_back(static_cast<uint32_t>(i));
      }
      uint32_t take = (numProcessAfterRestart > survivors)
                          ? static_cast<uint32_t>(numProcessAfterRestart - survivors)
                          : 0;
      // For Stage 3 we only support shrink end-to-end; expand requires
      // newcomers being ready (Stage 4). Clamp take by what coordinator has.
      uint32_t pending = 0;
      if (!coord::query_pending(_coord_fd, &pending)) {
        CmiAbort("UCX: coord::query_pending failed");
      }
      if (take > pending) {
        CmiPrintf("Charm> coordinator: requested %u newcomers, only %u available\n",
                  take, pending);
        take = pending;
      }
      if (!coord::commit(_coord_fd, _coord_epoch, kills, take,
                         _coord_members, &view)) {
        CmiAbort("UCX: coord::commit failed");
      }
      CmiPrintf("Charm> coordinator COMMIT: epoch %u->%u, %u kills, %u taken, %zu members\n",
                _coord_epoch, view.epoch, (uint32_t)kills.size(), take, view.members.size());
      {
        extern double rescale_t_commit_done;
        extern double rescale_wall_now();
        rescale_t_commit_done = rescale_wall_now();
      }

      // Tree-broadcast the delta to surviving non-initiator ranks via the
      // existing UCX endpoints (still in OLD numbering at this point). The
      // last `take` entries of view.members are the newly-added newcomers.
      std::vector<coord::Member> added(view.members.end() - take, view.members.end());
      std::vector<uint8_t> reconfigPayload =
          UcxBuildReconfigPayload(view.epoch, kills, added);
      UcxReconfigTreeForward(oldNumNodes, myNode, kills, reconfigPayload);
    } else {
      // Non-initiator: wait for either DIE on TCP (killed ranks) or the UCX
      // tree-broadcast (survivors). Poll both so we don't deadlock either
      // way; the coordinator only pushes DIE on TCP — RECONFIG arrives via
      // UCX from this rank's chain predecessor.
      std::vector<uint8_t> reconfigPayload;
      bool gotReconfig = false;
      while (!gotDie && !gotReconfig) {
        // Non-blocking TCP probe for DIE.
        coord::Frame f;
        bool eof = false;
        if (coord::try_read_frame(_coord_fd, &f, &eof)) {
          if (f.type == coord::DIE) {
            gotDie = true;
            break;
          }
          CmiAbort("UCX: unexpected coord frame type %u during rescale", f.type);
        }
        if (eof) {
          CmiAbort("UCX: coordinator closed connection during rescale");
        }
        // UCX probe for RECONFIG broadcast.
        ucp_tag_recv_info_t info;
        ucp_tag_message_h msgh = ucp_tag_probe_nb(ucxCtx.worker,
                                                   UCX_MSG_TAG_RECONFIG,
                                                   UCX_MSG_TAG_MASK, 1, &info);
        if (msgh != NULL) {
          reconfigPayload = UcxRecvReconfigBytes(msgh, info.length);
          gotReconfig = true;
          break;
        }
        ucp_worker_progress(ucxCtx.worker);
      }

      if (gotReconfig) {
        // Parse the delta payload (no nodeId — we compute our own from the
        // kill set). Then forward to our chain successor BEFORE rebuilding
        // eps so the chain operates on the still-valid OLD ep array.
        const uint8_t *p = reconfigPayload.data();
        const uint8_t *end = p + reconfigPayload.size();
        uint32_t newEpoch = coord::get_u32(p, end);
        std::vector<uint32_t> killSet = coord::get_u32_vec(p, end);
        std::vector<coord::Member> added = coord::get_members(p, end);

        UcxReconfigTreeForward(oldNumNodes, myNode, killSet, reconfigPayload);

        // Build view from delta. Our new nodeId = oldId minus the count of
        // killed old ids strictly less than oldId (compact renumber).
        uint32_t myNewId = static_cast<uint32_t>(myNode);
        for (uint32_t k : killSet) {
          if ((int)k < myNode) myNewId--;
        }
        view.nodeId = myNewId;
        view.epoch = newEpoch;
        view.members = coord::apply_member_delta(_coord_members, killSet, added);
      }
    }

    if (gotDie) {
      // Killed: tear down everything and exit.
      //
      // NOTE: We previously tried `kill(getppid(), SIGTERM)` here to make
      // the local orted/prted exit cleanly so the HNP would stop watching
      // this node (so a later spot-instance termination wouldn't trigger
      // "lost communication" job teardown). It failed: the HNP treats any
      // unexpected daemon exit — including a clean SIGTERM-induced one —
      // as "lost communication" and tears down the entire job
      // *immediately*. So the kill broke normal shrink without buying
      // anything for the spot-loss case. Both ORTE and PRRTE behave this
      // way; the TCP-loss path is not gated by any MCA flag.
      //
      // The correct fix lives outside Charm — either switch to a launcher
      // that doesn't supervise daemons (Slurm srun, or a thin ssh-based
      // launcher with PMIx-direct bootstrap), or accept that a spot loss
      // ends the run and let charmrun_elastic restart the survivors.
      ucp_worker_destroy(ucxCtx.worker);
      ucp_cleanup(ucxCtx.context);
      CmiFree(ucxCtx.eps);
      CmiFree(ucxCtx.rxReqs);
#if CMK_SMP
      PCQueueDestroy(ucxCtx.txQueue);
#endif
      ::close(_coord_fd);
      _exit(0);
    }

    // Survivor: rebuild endpoints against the new view; worker stays alive.
    UcxReInitEpsFromView(view, oldNumNodes, myNode);
    _coord_epoch = view.epoch;
    _rescaleGeneration = (int)view.epoch;
    {
      extern double rescale_t_ep_reinit_done;
      extern double rescale_wall_now();
      if (CmiMyPe() == 0) rescale_t_ep_reinit_done = rescale_wall_now();
    }

    // Tree-collective barrier over the new-numbering UCX topology, replacing
    // the O(N) coordinator TCP star. Survivors and newcomers both call this;
    // up-reduce + down-broadcast over the binary tree (root at new nodeId 0).
    UcxTreeBarrier(static_cast<int>(view.nodeId),
                   static_cast<int>(view.members.size()));
    {
      extern double rescale_t_barrier_done, rescale_t_longjmp;
      extern double rescale_wall_now();
      if (CmiMyPe() == 0) {
        rescale_t_barrier_done = rescale_wall_now();
        rescale_t_longjmp      = rescale_t_barrier_done; // no work between
      }
    }

    _shrinkexpand_restarting = true;
    _shrinkexpand_new_numnodes = static_cast<int>(view.members.size());
    _shrinkexpand_my_node = static_cast<int>(view.nodeId);
    longjmp(_shrinkexpand_jmpbuf, 1);
  } else {
    CmiBarrier();
    ConverseExit();
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
#if CMK_SHRINK_EXPAND
    // Newcomers never call PMI runtime_init, and survivors' PMI world is the
    // pre-shrink set — neither can use the PMI fence to sync with the current
    // (post-rescale) membership. Route through the coordinator instead so all
    // current members participate.
    if (_coord_fd >= 0) {
        if (!coord::barrier(_coord_fd, _coord_epoch,
                            (uint32_t)CmiMyNode())) {
            CmiAbort("UCX: coord::barrier (LrtsBarrier) failed");
        }
        return;
    }
#endif
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

void LrtsSendDevice(int dest_pe, int src_pe, const void*& ptr, size_t size, uint64_t& tag) {
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
