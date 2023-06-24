/** @file
 * OFI LRTS machine layer
 *
 * Copyright (c) 2017, Intel Corporation. All rights reserved.
 * See LICENSE in this directory.
 *
 * Authors: Yohann Burette <yohann.burette@intel.com>
 *          Mikhail Shiryaev <mikhail.shiryaev@intel.com>
 *          Marat Shamshetdinov <marat.shamshetdinov@intel.com>
 * Date:    2017-06-23
 *
 * 10,000ft view:
 *  - Each Charm++ node opens an OFI RDM endpoint
 *  - For small (enough) messages, the sender sends the data directly
 *  - For long messages,
 *      1) the sender sends a OFIRmaHeader describing the data,
 *      2) the receiver retrieves the data with RMA Read,
 *      3) once done, the receiver sends an OFIRmaAck back.
 *  - The tag associated with each communication helps the receiver
 *    parse the data (i.e. short, long or ack).
 *  - The receiver uses a OFILongMsg structure to keep track of an
 *    ongoing long message retrieval.
 *
 * Runtime options:
 *  +ofi_eager_maxsize: (default: 65536) Threshold between buffered and RMA
 *                      paths.
 *  +ofi_cq_entries_count: (default: 8) Maximum number of entries to read from
 *                         the completion queue.
 *  +ofi_use_inject: (default: 1) Whether use buffered send.
 *  +ofi_num_recvs: (default: 8) Number of pre-posted receive buffers.
 *  +ofi_runtime_tcp: (default: off) During the initialization phase, the
 *                    OFI EP names need to be exchanged among all nodes. By
 *                    default, the exchange is done with both PMI and OFI. If
 *                    this flag is set then the exchange is done with PMI only.
 *
 *  Memory pool specific options:
 *  +ofi_mempool_init_size_mb: (default: 8) The initial size of memory pool in MBytes.
 *  +ofi_mempool_expand_size_mb: (default: 4) The size of expanding chunk in MBytes.
 *  +ofi_mempool_max_size_mb: (default: 512) The limit for total size
 *                            of memory pool in MBytes.
 *  +ofi_mempool_lb_size: (default: 1024) The left border size in bytes
 *                        from which the memory pool is used.
 *  +ofi_mempool_rb_size: (default: 67108864) The right border size in bytes
 *                        to which the memory pool is used.
 * @ingroup Machine
 */
/*@{*/

#include <stdio.h>
#include <errno.h>
#include "converse.h"
#include "cmirdmautils.h"
#include <algorithm>

/*Support for ++debug: */
#include <unistd.h> /*For getpid()*/
#include <stdlib.h> /*For sleep()*/

#include "machine.h"

/* TODO: macros regarding redefining locks that will affect pcqueue.h*/
#include "pcqueue.h"

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* TODO: add any that are related */
/* =======End of Definitions of Performance-Specific Macros =======*/


/* =====Beginning of Definitions of Message-Corruption Related Macros=====*/
/* TODO: add any that are related */
/* =====End of Definitions of Message-Corruption Related Macros=====*/


/* =====Beginning of Declarations of Machine Specific Variables===== */
/* TODO: add any that are related */
/* =====End of Declarations of Machine Specific Variables===== */

#include "machine-lrts.h"
#include "machine-common-core.C"

/* Libfabric headers */
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>

#define USE_OFIREQUEST_CACHE 0

/* Definition of OFIRequest + request cache */
#include "request.h"

/* Runtime to exchange EP addresses during LrtsInit() */
#if CMK_USE_PMI || CMK_USE_SIMPLEPMI
#include "runtime-pmi.C"
#elif CMK_USE_PMI2
#include "runtime-pmi2.C"
#elif CMK_USE_PMIX
#include "runtime-pmix.C"
#endif

#define USE_MEMPOOL 0

#if USE_MEMPOOL

#include "mempool.h"
#define MEMPOOL_INIT_SIZE_MB_DEFAULT   8
#define MEMPOOL_EXPAND_SIZE_MB_DEFAULT 4
#define MEMPOOL_MAX_SIZE_MB_DEFAULT    512
#define MEMPOOL_LB_DEFAULT             1024
#define MEMPOOL_RB_DEFAULT             67108864
#define ONE_MB                         1048576

CpvDeclare(mempool_type*, mempool);

#endif /* USE_MEMPOOL */

#define CmiSetMsgSize(msg, sz)  ((((CmiMsgHeaderBasic *)msg)->size) = (sz))

#define CACHELINE_LEN 64

#define OFI_NUM_RECV_REQS_DEFAULT    8
#define OFI_NUM_RECV_REQS_MAX        4096

#define OFI_EAGER_MAXSIZE_DEFAULT    65536
#define OFI_EAGER_MAXSIZE_MAX        1048576

#define OFI_CQ_ENTRIES_COUNT_DEFAULT 8
#define OFI_CQ_ENTRIES_COUNT_MAX     1024

#define OFI_USE_INJECT_DEFAULT       1

#define OFI_KEY_FORMAT_EPNAME "ofi-epname-%i"

#define OFI_OP_SHORT 0x1ULL
#define OFI_OP_LONG  0x2ULL
#define OFI_OP_ACK   0x3ULL
#define OFI_RDMA_DIRECT_REG_AND_PUT 0x4ULL
#define OFI_RDMA_DIRECT_REG_AND_GET 0x5ULL

#define OFI_RDMA_DIRECT_DEREG_AND_ACK 0x6ULL

#define OFI_OP_NAMES 0x8ULL

#define OFI_READ_OP 1
#define OFI_WRITE_OP 2

#define OFI_OP_MASK  0x7ULL

#define MR_ACCESS_PERMISSIONS (FI_REMOTE_READ | FI_READ | FI_RECV | FI_SEND)

static inline int process_completion_queue();

#define ALIGNED_ALLOC(ptr, size)                                        \
  do {                                                                  \
      int pm_ret = posix_memalign((void**)(&ptr), CACHELINE_LEN, size); \
      if (CMI_UNLIKELY((pm_ret != 0) || !ptr))                          \
      {                                                                 \
          CmiAbort("posix_memalign: ret %d", pm_ret);                   \
      }                                                                 \
  } while (0)

#define OFI_RETRY(func)                                 \
    do {                                                \
        intmax_t _ret;                                  \
        do {                                            \
            _ret = func;                                \
            if (CMI_LIKELY(_ret == 0)) break;           \
            if (_ret != -FI_EAGAIN) {                   \
                CmiAbort("OFI_RETRY: ret %jd\n", _ret); \
            }                                           \
            process_completion_queue();                 \
        } while (_ret == -FI_EAGAIN);                   \
    } while (0)

/* OFI_INFO is used to print information messages during LrtsInit() */
#define OFI_INFO(...) \
    if (*myNodeID == 0) CmiPrintf("Charm++>ofi> " __VA_ARGS__)

#define PRINT_THREAD_INFO(message)              \
    MACHSTATE5(2,"thread info: process_idx=%i " \
              "local_thread_idx=%i "            \
              "global_thread_idx=%i "           \
              "local_worker_thread_count=%i "   \
              "is_comm_thread=%i\n",            \
              CmiMyNode(),                      \
              CmiMyRank(),                      \
              CmiMyPe(),                        \
              CmiMyNodeSize(),                  \
              CmiInCommThread());

/**
 * OFI RMA Header
 * Message sent by sender to receiver during RMA Read of long messages.
 *  - nodeNo: Target node number
 *  - src_msg: Address of source msg; Sent back as part of OFIRmaAck
 *  - len: Length of message
 *  - key: Remote key
 *  - mr: Address of memory region; Sent back as part of OFIRmaAck
 */
typedef struct OFIRmaHeader {
    uint64_t src_msg;
    uint64_t len;
    uint64_t key;
    uint64_t mr;
    int      nodeNo;
} OFIRmaHeader;

/**
 * OFI RMA Ack
 * Message sent by receiver to sender during RMA Read of long messages.
 *  - src_msg: Address of source msg; Received as part of OFIRmaHeader
 *  - mr: Address of memory region; Received as part of OFIRmaHeader
 */
typedef struct OFIRmaAck {
    uint64_t src_msg;
    uint64_t mr;
} OFIRmaAck;

/**
 * OFI Long Message
 * Structure stored by the receiver about ongoing RMA Read of long message.
 *  - asm_msg: Assembly buffer where the data is RMA Read into
 *  - nodeNo: Target node number
 *  - rma_ack: OFI Rma Ack sent to sender once all the data has been RMA Read
 *  - completion_count: Number of expected RMA Read completions
 *  - mr: Memory Region where the data is RMA Read into
 */
typedef struct OFILongMsg {
    char                *asm_msg;
    int                 nodeNo;
    struct OFIRmaAck    rma_ack;
    size_t              completion_count;
    struct fid_mr       *mr;
} OFILongMsg;

/**
 * TO_OFI_REQ is used retrieve the request associated with a given fi_context.
 */
#define TO_OFI_REQ(_ptr_context) \
    container_of((_ptr_context), OFIRequest, context)

typedef struct OFIContext {
    /** Endpoint to communicate on */
    struct fid_ep *ep;

    /** Completion queue handle */
    struct fid_cq *cq;

    /**
     * Maximum size for eager messages.
     * RMA Read for larger messages.
     */
    size_t eager_maxsize;

    /**
     * Maximum inject size.
     */
    size_t inject_maxsize;

    /**
     * Maximum number of completion queue entries that
     * can be retrieved by each fi_cq_read() call.
     */
    size_t cq_entries_count;

    /**
     * Whether to use buffered send (aka inject)
     */
    int use_inject;

#if USE_OFIREQUEST_CACHE
    /** OFIRequest allocator */
    request_cache_t *request_cache;
#endif

#if CMK_SMP
    /**
     * Producer/Consumer Queue used in CMK_SMP mode:
     *  - worker thread pushes messages to the queue
     *  - comm thread pops and sends
     * Locking is already done by PCQueue.
     */
    PCQueue send_queue;
#endif

    /** Fabric Domain handle */
    struct fid_fabric *fabric;

    /** Access Domain handle */
    struct fid_domain *domain;

    /** Address vector handle */
    struct fid_av *av;

    /**
     * Maximum size for RMA operations.
     * Multiple RMA operations for larger messages.
     */
    size_t rma_maxsize;

    /**
     * MR mode:
     *  - FI_MR_SCALABLE allows us to register all the memory with our own key,
     *  - FI_MR_BASIC requires us to register the RMA buffers
     *    and to exchange the keys.
     */
    enum fi_mr_mode mr_mode;

    /** Used as unique key value in FI_MR_SCALABLE mode */
    uint64_t mr_counter;

    /** Number of pre-posted receive requests */
    int num_recv_reqs;

    /** Pre-posted receive requests */
    OFIRequest **recv_reqs;

#if USE_MEMPOOL
    size_t mempool_init_size;
    size_t mempool_expand_size;
    size_t mempool_max_size;
    size_t mempool_lb_size;
    size_t mempool_rb_size;
#endif
} OFIContext __attribute__ ((aligned (CACHELINE_LEN)));

static void recv_callback(struct fi_cq_tagged_entry *e, OFIRequest *req);
static int fill_av(int myid, int nnodes, struct fid_ep *ep,
                   struct fid_av *av, struct fid_cq *cq);
static int fill_av_ofi(int myid, int nnodes, struct fid_ep *ep,
                       struct fid_av *av, struct fid_cq *cq);

static OFIContext context;

#include "machine-rdma.h"
#if CMK_ONESIDED_IMPL
#include "machine-onesided.h"
#endif

/* ### Beginning of Machine-startup Related Functions ### */
void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID)
{
    struct fi_info        *providers;
    struct fi_info        *prov;
    struct fi_info        *hints;
    struct fi_domain_attr domain_attr = {0};
    struct fi_tx_attr     tx_attr = { 0 };
    struct fi_cq_attr     cq_attr = { 0 };
    struct fi_av_attr     av_attr = { (enum fi_av_type)0 };
    int                   fi_version;
    size_t                max_header_size;

    int i;
    int ret;

    /**
     * Initialize our runtime environment -- e.g. PMI.
     */
    ret = runtime_init(myNodeID, numNodes);
    if (ret) {
        CmiAbort("OFI::LrtsInit::runtime_init failed");
    }

    /**
     * Hints to filter providers
     * See man fi_getinfo for a list of all filters
     * mode: This OFI machine will pass in context into communication calls
     * ep_type: Reliable datagram operation
     * resource_mgmt: Let the provider manage the resources
     * caps: Capabilities required from the provider. We want to use the
     *       tagged message queue and rma read APIs.
     */
    hints = fi_allocinfo();
    CmiAssert(NULL != hints);
    hints->mode                          = FI_CONTEXT;
    hints->ep_attr->type                 = FI_EP_RDM;
    hints->domain_attr->resource_mgmt    = FI_RM_ENABLED;
    hints->caps                          = FI_TAGGED;
    hints->caps                         |= FI_RMA;
    hints->caps                         |= FI_REMOTE_READ;

    /**
     * FI_VERSION provides binary backward and forward compatibility support
     * Specify the version of OFI this machine is coded to, the provider will
     * select struct layouts that are compatible with this version.
     */
    fi_version = FI_VERSION(1, 0);

    ret = fi_getinfo(fi_version, NULL, NULL, 0ULL, hints, &providers);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_getinfo error");
    }

    if (providers == NULL) {
        CmiAbort("OFI::LrtsInit::No provider found");
    }

    /**
     * Here we elect to use the first provider from the list.
     * Further filtering could be done at this point (e.g. name).
     */
    prov = providers;

    OFI_INFO("provider: %s\n", prov->fabric_attr->prov_name);
    OFI_INFO("control progress: %d\n", prov->domain_attr->control_progress);
    OFI_INFO("data progress: %d\n", prov->domain_attr->data_progress);

    context.inject_maxsize = prov->tx_attr->inject_size;
    OFI_INFO("maximum inject message size: %ld\n", context.inject_maxsize);

    context.eager_maxsize = OFI_EAGER_MAXSIZE_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_eager_maxsize", (int*)&context.eager_maxsize);
    if (context.eager_maxsize > prov->ep_attr->max_msg_size)
        CmiAbort("OFI::LrtsInit::Eager max size > max msg size.");
    if (context.eager_maxsize > OFI_EAGER_MAXSIZE_MAX || context.eager_maxsize < 0)
        CmiAbort("OFI::LrtsInit::Eager max size range error.");
    max_header_size = (sizeof(OFIRmaHeader) >= sizeof(OFIRmaAck)) ? sizeof(OFIRmaHeader) : sizeof(OFIRmaAck);
    if (context.eager_maxsize < max_header_size)
        CmiAbort("OFI::LrtsInit::Eager max size too small to fit headers.");
    OFI_INFO("eager maximum message size: %ld (maximum header size: %ld)\n",
             context.eager_maxsize, max_header_size);

    context.cq_entries_count = OFI_CQ_ENTRIES_COUNT_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_cq_entries_count", (int*)&context.cq_entries_count);
    if (context.cq_entries_count > OFI_CQ_ENTRIES_COUNT_MAX || context.cq_entries_count <= 0)
        CmiAbort("OFI::LrtsInit::Cq entries count range error");
    OFI_INFO("cq entries count: %ld\n", context.cq_entries_count);

    context.use_inject = OFI_USE_INJECT_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_use_inject", &context.use_inject);
    if (context.use_inject < 0)
        CmiAbort("OFI::LrtsInit::Use inject value error");
    OFI_INFO("use inject: %d\n", context.use_inject);

    context.rma_maxsize = prov->ep_attr->max_msg_size;
    context.mr_mode = static_cast<fi_mr_mode>(prov->domain_attr->mr_mode);
    context.mr_counter = 0;

    OFI_INFO("maximum rma size: %ld\n", context.rma_maxsize);
    OFI_INFO("mr mode: 0x%x\n", context.mr_mode);

    if ((context.mr_mode != FI_MR_BASIC) &&
        (context.mr_mode != FI_MR_SCALABLE)) {
        CmiAbort("OFI::LrtsInit::Unsupported MR mode");
    }

    OFI_INFO("use memory pool: %d\n", USE_MEMPOOL);

#if USE_MEMPOOL
    size_t mempool_init_size_mb = MEMPOOL_INIT_SIZE_MB_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_mempool_init_size_mb", (int*)&mempool_init_size_mb);
    context.mempool_init_size = mempool_init_size_mb * ONE_MB;

    size_t mempool_expand_size_mb = MEMPOOL_EXPAND_SIZE_MB_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_mempool_expand_size_mb", (int*)&mempool_expand_size_mb);
    context.mempool_expand_size = mempool_expand_size_mb * ONE_MB;

    size_t mempool_max_size_mb = MEMPOOL_MAX_SIZE_MB_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_mempool_max_size_mb", (int*)&mempool_max_size_mb);
    context.mempool_max_size = mempool_max_size_mb * ONE_MB;

    context.mempool_lb_size = MEMPOOL_LB_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_mempool_lb_size", (int*)&context.mempool_lb_size);

    context.mempool_rb_size = MEMPOOL_RB_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_mempool_rb_size", (int*)&context.mempool_rb_size);

    if (context.mempool_lb_size > context.mempool_rb_size)
        CmiAbort("OFI::LrtsInit::Mempool left border should be less or equal to right border");

    OFI_INFO("mempool init size: %ld\n", context.mempool_init_size);
    OFI_INFO("mempool expand size: %ld\n", context.mempool_expand_size);
    OFI_INFO("mempool max size: %ld\n", context.mempool_max_size);
    OFI_INFO("mempool left border size: %ld\n", context.mempool_lb_size);
    OFI_INFO("mempool right border size: %ld\n", context.mempool_rb_size);
#endif

    /**
     * Open fabric
     * The getinfo struct returns a fabric attribute struct that can be used to
     * instantiate the virtual or physical network. This opens a "fabric
     * provider". See man fi_fabric for details.
     */
    ret = fi_fabric(prov->fabric_attr, &context.fabric, NULL);
    if (ret < 0) {
        fi_freeinfo(providers);
        CmiAbort("OFI::LrtsInit::fi_fabric error");
    }

    /**
     * Create the access domain, which is the physical or virtual network or
     * hardware port/collection of ports.  Returns a domain object that can be
     * used to create endpoints.  See man fi_domain for details.
     */
    ret = fi_domain(context.fabric, prov, &context.domain, NULL);
    if (ret < 0) {
        fi_freeinfo(providers);
        CmiAbort("OFI::LrtsInit::fi_domain error");
    }

    /**
     * Create a transport level communication endpoint.  To use the endpoint,
     * it must be bound to completion counters or event queues and enabled,
     * and the resources consumed by it, such as address vectors, counters,
     * completion queues, etc. See man fi_endpoint for more details.
     */
    ret = fi_endpoint(context.domain, /* In:  Domain object   */
                      prov,           /* In:  Provider        */
                      &context.ep,    /* Out: Endpoint object */
                      NULL);          /* Optional context     */
    if (ret < 0) {
        fi_freeinfo(providers);
        CmiAbort("OFI::LrtsInit::fi_endpoint error");
    }

    /**
     * Create the objects that will be bound to the endpoint.
     * The objects include:
     *     - completion queue for events
     *     - address vector of other endpoint addresses
     */
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    ret = fi_cq_open(context.domain, &cq_attr, &context.cq, NULL);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_cq_open error");
    }

    /**
     * Since the communications happen between Nodes and that each Node
     * has a number (NodeNo), we can use the Address Vector in FI_AV_TABLE
     * mode. The addresses of the Nodes simply need to be inserted in order
     * so that the NodeNo becomes the index in the AV. The advantage being
     * that the fi_addrs are stored by the OFI provider.
     */
    av_attr.type = FI_AV_TABLE;
    ret = fi_av_open(context.domain, &av_attr, &context.av, NULL);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_av_open error");
    }

    /**
     * Bind the CQ and AV to the endpoint object.
     */
    ret = fi_ep_bind(context.ep,
                     (fid_t)context.cq,
                     FI_RECV | FI_TRANSMIT);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_bind EP-CQ error");
    }
    ret = fi_ep_bind(context.ep,
                     (fid_t)context.av,
                     0);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_bind EP-AV error");
    }

    /**
     * Enable the endpoint for communication
     * This commits the bind operations.
     */
    ret = fi_enable(context.ep);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_enable error");
    }

    OFI_INFO("use request cache: %d\n", USE_OFIREQUEST_CACHE);

#if USE_OFIREQUEST_CACHE
    /**
     * Create request cache.
     */
    context.request_cache = create_request_cache();
#endif

    /**
     * Create local receive buffers and pre-post them.
     */
    context.num_recv_reqs = OFI_NUM_RECV_REQS_DEFAULT;
    CmiGetArgInt(*argv, "+ofi_num_recvs", &context.num_recv_reqs);
    if (context.num_recv_reqs > OFI_NUM_RECV_REQS_MAX || context.num_recv_reqs <= 0)
        CmiAbort("OFI::LrtsInit::Num recv reqs range error");
    OFI_INFO("number of pre-allocated recvs: %i\n", context.num_recv_reqs);

    /**
     * Exchange EP names and insert them into the AV.
     */
    if (CmiGetArgFlag(*argv, "+ofi_runtime_tcp")) {
        OFI_INFO("exchanging addresses over TCP\n");
        ret = fill_av(*myNodeID, *numNodes, context.ep,
                       context.av, context.cq);
        if (ret < 0) {
            CmiAbort("OFI::LrtsInit::fill_av");
        }
    } else {
        OFI_INFO("exchanging addresses over OFI\n");
        ret = fill_av_ofi(*myNodeID, *numNodes, context.ep,
                           context.av, context.cq);
        if (ret < 0) {
            CmiAbort("OFI::LrtsInit::fill_av_ofi");
        }
    }

#if CMK_SMP
    /**
     * Initialize send queue.
     */
    context.send_queue = PCQueueCreate();
#endif

    /**
     * Free providers info since it's not needed anymore.
     */
    fi_freeinfo(hints);
    hints = NULL;
    fi_freeinfo(providers);
    providers = NULL;
}

static inline
void prepost_buffers()
{
    OFIRequest **reqs;
    ALIGNED_ALLOC(reqs, sizeof(void*) * context.num_recv_reqs);

    int i;
    for (i = 0; i < context.num_recv_reqs; i++) {
#if USE_OFIREQUEST_CACHE
        reqs[i] = alloc_request(context.request_cache);
#else
        reqs[i] = (OFIRequest *)CmiAlloc(sizeof(OFIRequest));
#endif
        reqs[i]->callback = recv_callback;
        reqs[i]->data.recv_buffer = CmiAlloc(context.eager_maxsize);
        CmiAssert(reqs[i]->data.recv_buffer);

        MACHSTATE2(3, "---> posting recv req %p buf=%p",
                   reqs[i], reqs[i]->data.recv_buffer);

        /* Receive from any node with any tag */
        OFI_RETRY(fi_trecv(context.ep,
                           reqs[i]->data.recv_buffer,
                           context.eager_maxsize,
                           NULL,
                           FI_ADDR_UNSPEC,
                           0,
                           OFI_OP_MASK,
                           &(reqs[i]->context)));
    }
    context.recv_reqs = reqs;
}

static inline
void send_short_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * A short message was sent.
     * Free up resources.
     */
    char *msg;

    MACHSTATE(3, "OFI::send_short_callback {");

    msg = (char *)req->data.short_msg;
    CmiAssert(msg);
    MACHSTATE1(3, "--> msg=%p", msg);

    if(req->freeMe)
      CmiFree(msg);

#if USE_OFIREQUEST_CACHE
    free_request(req);
#else
    CmiFree(req);
#endif

    MACHSTATE(3, "} OFI::send_short_callback done");
}

static inline
void send_rma_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * An OFIRmaHeader was sent.
     * Free up resources.
     */
    OFIRmaHeader *header;

    MACHSTATE(3, "OFI::send_rma_callback {");

    header = req->data.rma_header;
    free(header);

#if USE_OFIREQUEST_CACHE
    free_request(req);
#else
    CmiFree(req);
#endif

    MACHSTATE(3, "} OFI::send_rma_callback done");
}

static inline
void ofi_send(void *buf, size_t buf_size, int addr, uint64_t tag, OFIRequest *req)
{
    if (context.use_inject && buf_size <= context.inject_maxsize)
    {
        /**
         * The message is small enough to be injected.
         * This won't generate any completion, so we can free the msg now.
         */
        MACHSTATE(3, "----> inject");

        OFI_RETRY(fi_tinject(context.ep,
                             buf,
                             buf_size,
                             addr,
                             tag));
        req->callback(NULL, req);
    }
    else
    {
        /* Else, use regular send. */
        OFI_RETRY(fi_tsend(context.ep,
                           buf,
                           buf_size,
                           NULL,
                           addr,
                           tag,
                           &req->context));
    }
}

/**
 * sendMsg is used to send a message.
 * In CMK_SMP mode, this is called by the comm thread.
 */
static inline int sendMsg(OFIRequest *req)
{
    int       ret;
    uint64_t  op;
    char     *buf;
    size_t    len;

    MACHSTATE5(2,
               "OFI::sendMsg destNode=%i destPE=%i size=%i msg=%p mode=%i {",
               req->destNode, req->destPE, req->size, req->data, req->mode);

    if (req->size <= context.eager_maxsize) {
        /**
         * The message is small enough to be sent entirely.
         */
        MACHSTATE(3, "--> eager");

        op = OFI_OP_SHORT;
        buf = (char *)req->data.short_msg;
        len = req->size;
    } else {
        /**
         * The message is too long to be sent directly.
         * Let other side use RMA Read instead by sending an OFIRmaHeader.
         */
        MACHSTATE(3, "--> long");

        op = OFI_OP_LONG;
        buf = (char *)req->data.rma_header;
        len = sizeof(OFIRmaHeader);
    }

    ofi_send(buf, len, req->destNode, op, req);

    MACHSTATE(2, "} OFI::sendMsg");
    return 0;
}

/**
 * In non-SMP mode, this is used to send a message.
 * In CMK_SMP mode, this is called by a worker thread to send a message.
 */
CmiCommHandle LrtsSendFunc(int destNode, int destPE, int size, char *msg, int mode)
{

    int           ret;
    OFIRequest    *req;

    MACHSTATE5(2,
            "OFI::LrtsSendFunc destNode=%i destPE=%i size=%i msg=%p mode=%i {",
            destNode, destPE, size, msg, mode);

    CmiSetMsgSize(msg, size);

#if USE_OFIREQUEST_CACHE
    req = alloc_request(context.request_cache);
#else
    req = (OFIRequest *)CmiAlloc(sizeof(OFIRequest));
#endif
    CmiAssert(req);

    req->destNode = destNode;
    req->destPE   = destPE;
    req->size     = size;
    req->mode     = mode;

    if (size <= context.eager_maxsize) {
        /**
         * The message is small enough to be sent entirely.
         */
        MACHSTATE(3, "--> eager");

        req->callback = send_short_callback;
        req->data.short_msg = msg;
    } else {
        /**
         * The message is too long to be sent directly.
         * Let other side use RMA Read instead by sending an OFIRmaHeader.
         */
        OFIRmaHeader  *rma_header;
        struct fid_mr *mr;
        uint64_t      requested_key = 0;

        MACHSTATE(3, "--> long");

        ALIGNED_ALLOC(rma_header, sizeof(*rma_header));

        if (FI_MR_SCALABLE == context.mr_mode) {
            /**
             *  In FI_MR_SCALABLE mode, we need to specify a unique key when
             *  registering memory. Here we simply increment a counter
             *  atomically.
             */
            requested_key = __sync_fetch_and_add(&(context.mr_counter), 1);
        }

        /* Register new MR to RMA Read from */
        ret = fi_mr_reg(context.domain,        /* In:  domain object */
                        msg,                   /* In:  lower memory address */
                        size,                  /* In:  length */
                        MR_ACCESS_PERMISSIONS, /* In:  access permissions */
                        0ULL,                  /* In:  offset (not used) */
                        requested_key,         /* In:  requested key */
                        0ULL,                  /* In:  flags */
                        &mr,                   /* Out: memregion object */
                        NULL);                 /* In:  context (not used) */

        if (ret) {
            MACHSTATE1(3, "fi_mr_reg error: %d\n", ret);
            CmiAbort("fi_mr_reg error");
        }

        rma_header->nodeNo  = CmiMyNodeGlobal();
        rma_header->src_msg = (uint64_t)msg;
        rma_header->len     = size;
        rma_header->key     = fi_mr_key(mr);
        rma_header->mr      = (uint64_t)mr;

        req->callback        = send_rma_callback;
        req->data.rma_header = rma_header;
    }

#if CMK_SMP
    /* Enqueue message */
    MACHSTATE2(2, " --> (PE=%i) enqueuing message (queue depth=%i)",
               CmiMyPe(), PCQueueLength(context.send_queue));
    PCQueuePush(context.send_queue, (char *)req);
#else
    /* Send directly */
    sendMsg(req);
#endif

    MACHSTATE(2, "} OFI::LrtsSendFunc");

    return (CmiCommHandle)req;
}

static inline
void send_ack_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * An OFIRmaAck was sent (see rma_read_callback()).
     * We are done with the RMA Read operation. Free up the resources.
     */
    OFILongMsg *long_msg;

    MACHSTATE(3, "OFI::send_ack_callback {");

    long_msg = req->data.long_msg;
    CmiAssert(long_msg);

    if (long_msg->mr)
        fi_close((struct fid*)long_msg->mr);

    free(long_msg);

#if USE_OFIREQUEST_CACHE
    free_request(req);
#else
    CmiFree(req);
#endif

    MACHSTATE(3, "} OFI::send_ack_callback done");
}

static inline
void rma_read_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * An RMA Read operation completed.
     */
    OFILongMsg *long_msg;

    MACHSTATE(3, "OFI::rma_read_callback {");

    long_msg = req->data.long_msg;
    CmiAssert(long_msg);
    CmiAssert(long_msg->completion_count > 0);

    long_msg->completion_count--;
    MACHSTATE1(3, "--> completion_count=%ld", long_msg->completion_count);

    if (0 == long_msg->completion_count) {
        /**
         * long_msg can be destroyed in case of fi_tinject,
         * so save pointer to assembly buffer to use it below.
         */
        char* asm_msg = long_msg->asm_msg;

        /**
         *  The message has been RMA Read completely.
         *  Send ACK to notify the other side that we are done.
         *  The resources are freed by send_ack_callback().
         */
        req->callback = send_ack_callback;
        req->data.long_msg = long_msg;

        ofi_send(&long_msg->rma_ack,
                 sizeof long_msg->rma_ack,
                 long_msg->nodeNo,
                 OFI_OP_ACK,
                 req);

        /**
         * Pass received message to upper layer.
         */
        MACHSTATE1(3, "--> Finished receiving msg size=%i", CMI_MSG_SIZE(asm_msg));

        handleOneRecvedMsg(CMI_MSG_SIZE(asm_msg), asm_msg);
    } else {
#if USE_OFIREQUEST_CACHE
      free_request(req);
#else
      CmiFree(req);
#endif
    }

    MACHSTATE(3, "} OFI::rma_read_callback done");
}

static inline
void process_short_recv(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * A short message was received:
     *   - Pass the message to the upper layer,
     *   - Allocate new recv buffer.
     */

    char    *data;
    size_t  msg_size;

    data = (char *)req->data.recv_buffer;
    CmiAssert(data);

    msg_size = CMI_MSG_SIZE(data);
    MACHSTATE2(3, "--> eager msg (e->len=%ld msg_size=%ld)", e->len, msg_size);

    req->data.recv_buffer = CmiAlloc(context.eager_maxsize);
    CmiAssert(req->data.recv_buffer);

    handleOneRecvedMsg(e->len, data);
}

static inline
void process_long_recv(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * An OFIRmaHeader was received:
     *   - Allocate enough space to store the long message,
     *   - Create OFILongMsg to keep track of the data retrieval,
     *   - Issue the RMA Read operation(s) to retrieve the data.
     */

    int ret;
    OFILongMsg *long_msg;
    OFIRequest *rma_req;
    OFIRmaHeader *rma_header;
    struct fid_mr *mr = NULL;
    char *asm_buf;
    int nodeNo;
    uint64_t rbuf;
    size_t len;
    uint64_t rkey;
    uint64_t rmsg;
    uint64_t rmr;
    char *lbuf;
    size_t remaining;
    size_t chunk_size;

    CmiAssert(e->len == sizeof(OFIRmaHeader));

    /**
     * Parse header
     */
    rma_header = req->data.rma_header;

    nodeNo = rma_header->nodeNo;
    rmsg   = rma_header->src_msg;
    len    = rma_header->len;
    rkey   = rma_header->key;
    rmr    = rma_header->mr;

    MACHSTATE2(3, "--> Receiving long msg len=%ld rkey=0x%lx", len, rkey);

    /**
     * Prepare buffer
     */
    asm_buf = (char *)CmiAlloc(len);
    CmiAssert(asm_buf);

    if (FI_MR_BASIC == context.mr_mode) {
        /* Register local MR to read into */
        ret = fi_mr_reg(context.domain,        /* In:  domain object */
                        asm_buf,               /* In:  lower memory address */
                        len,                   /* In:  length */
                        MR_ACCESS_PERMISSIONS, /* In:  access permissions */
                        0ULL,                  /* In:  offset (not used) */
                        0ULL,                  /* In:  requested key (none)*/
                        0ULL,                  /* In:  flags */
                        &mr,                   /* Out: memregion object */
                        NULL);                 /* In:  context (not used) */
        if (ret) {
            MACHSTATE1(3, "fi_mr_reg error: %d\n", ret);
            CmiAbort("fi_mr_reg error");
        }
    }

    /**
     * Save some information about the RMA Read operation(s)
     */
    ALIGNED_ALLOC(long_msg, sizeof(*long_msg));
    long_msg->asm_msg          = asm_buf;
    long_msg->nodeNo           = nodeNo;
    long_msg->rma_ack.src_msg  = rmsg;
    long_msg->rma_ack.mr       = rmr;
    long_msg->completion_count = 0;
    long_msg->mr               = mr;

    /**
     * Issue RMA Read request(s)
     */
    remaining = len;
    lbuf      = asm_buf;
    rbuf      = (FI_MR_SCALABLE == context.mr_mode) ? 0 : rmsg;

    while (remaining > 0) {
        /* Determine size of operation */
        chunk_size = (remaining <= context.rma_maxsize) ? remaining : context.rma_maxsize;

#if USE_OFIREQUEST_CACHE
        rma_req = alloc_request(context.request_cache);
#else
        rma_req = (OFIRequest *)CmiAlloc(sizeof(OFIRequest));
#endif
        CmiAssert(rma_req);
        rma_req->callback = rma_read_callback;
        rma_req->data.long_msg = long_msg;

        /* Increment number of expected completions */
        long_msg->completion_count++;

        MACHSTATE5(3, "---> RMA Read lbuf %p rbuf %p rmsg %p len %ld chunk #%d",
                   lbuf, rbuf, rmsg, chunk_size, long_msg->completion_count);

        OFI_RETRY(fi_read(context.ep,
                          lbuf,
                          chunk_size,
                          (mr) ? fi_mr_desc(mr) : NULL,
                          nodeNo,
                          rbuf,
                          rkey,
                          &rma_req->context));

        remaining  -= chunk_size;
        lbuf       += chunk_size;
        rbuf       += chunk_size;
    }
}

static inline
void process_long_send_ack(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * An OFIRmaAck was received; Close memory region and free original msg.
     */

    struct fid *mr;
    char *msg;

    mr = (struct fid*)req->data.rma_ack->mr;
    CmiAssert(mr);
    fi_close(mr);

    msg = (char *)req->data.rma_ack->src_msg;
    CmiAssert(msg);

    MACHSTATE1(3, "--> Finished sending msg size=%i", CMI_MSG_SIZE(msg));

    CmiFree(msg);
}

static inline
void recv_callback(struct fi_cq_tagged_entry *e, OFIRequest *req)
{
    /**
     * Some data was received:
     *  - the tag tells us what type of message it is; process it
     *  - repost recv request
     */
    MACHSTATE(3, "OFI::recv_callback {");

    switch (e->tag) {
    case OFI_OP_SHORT:
        process_short_recv(e, req);
        break;
    case OFI_OP_LONG:
        process_long_recv(e, req);
        break;
    case OFI_OP_ACK:
        process_long_send_ack(e, req);
        break;
#if CMK_ONESIDED_IMPL
    case OFI_RDMA_DIRECT_REG_AND_PUT:
        process_onesided_reg_and_put(e, req);
        break;
    case OFI_RDMA_DIRECT_REG_AND_GET:
        process_onesided_reg_and_get(e, req);
        break;
    case OFI_RDMA_DIRECT_DEREG_AND_ACK:
        process_onesided_dereg_and_ack(e, req);
        break;
#endif
    default:
        MACHSTATE2(3, "--> unknown operation %x len=%ld", e->tag, e->len);
        CmiAbort("!! Wrong operation !!");
    }

    MACHSTATE2(3, "Reposting recv req %p buf=%p", req, req->data.recv_buffer);
    OFI_RETRY(fi_trecv(context.ep,
                       req->data.recv_buffer,
                       context.eager_maxsize,
                       NULL,
                       FI_ADDR_UNSPEC,
                       0,
                       OFI_OP_MASK,
                       &req->context));

    MACHSTATE(3, "} OFI::recv_callback done");
}

static inline
int process_completion_queue()
{
    int ret;
    struct fi_cq_tagged_entry entries[context.cq_entries_count];
    struct fi_cq_err_entry error;
    OFIRequest *req;

    ret = fi_cq_read(context.cq, entries, context.cq_entries_count);
    if (ret > 0)
    {
        /* One or more completions were found */
        int idx;
        for (idx = 0; idx < ret; idx++)
        {
            struct fi_cq_tagged_entry* e = &(entries[idx]);
            CmiAssert(e->op_context != NULL);

            /* Retrieve request from context */
            req = container_of((e->op_context), OFIRequest, context);

            /* Execute request callback */
            if ((e->flags & FI_SEND) ||
                (e->flags & FI_RECV) ||
                (e->flags & FI_RMA))
            {
                req->callback(e, req);
            }
            else
            {
                MACHSTATE1(3, "Missed event with flags=%x", e->flags);
                CmiAbort("!! Missed an event !!");
            }
        }
    }
    else if (ret == -FI_EAGAIN)
    {
        /* Completion Queue is empty */
        ret = 0;
    }
    else if (ret < 0)
    {
        MACHSTATE1(3, "POLL: Error %d\n", ret);
        CmiPrintf("POLL: Error %d\n", ret);
        if (ret == -FI_EAVAIL)
        {
            MACHSTATE(3, "POLL: error available\n");
            CmiPrintf("POLL: error available\n");
            ret = fi_cq_readerr(context.cq, &error, sizeof(error));
            if (ret < 0)
            {
                CmiAbort("can't retrieve error");
            }
            MACHSTATE2(3, "POLL: error is %d (ret=%d)\n", error.err, ret);
            CmiPrintf("POLL: error is %d (ret=%d)\n", error.err, ret);
            const char* strerror = fi_cq_strerror(context.cq, error.prov_errno, error.err_data, nullptr, 0);
            if (strerror == nullptr)
            {
                CmiAbort("can't retrieve error string");
            }
            MACHSTATE1(3, "POLL: error string is \"%s\"\n", strerror);
            CmiPrintf("POLL: error string is \"%s\"\n", strerror);
        }
        CmiAbort("Polling error");
    }
    return ret;
}

#if CMK_SMP
static inline
int process_send_queue()
{
    OFIRequest *req;
    int ret = 0;
    /**
     * Comm thread sends the next message that is waiting in the send_queue.
     */
    req = (OFIRequest*)PCQueuePop(context.send_queue);
    if (req)
    {
        MACHSTATE2(2, " --> (PE=%i) dequeuing message (queue depth: %i)",
                CmiMyPe(), PCQueueLength(context.send_queue));
        MACHSTATE5(2,
                " --> dequeuing destNode=%i destPE=%i size=%d msg=%p mode=%d",
                req->destNode, req->destPE, req->size, req->data, req->mode);
        sendMsg(req);
        ret = 1;
    }
    return ret;
}
#endif

#if USE_MEMPOOL

void *alloc_mempool_block(size_t *size, mem_handle_t *mem_hndl, int expand_flag)
{
    size_t alloc_size =  expand_flag ? context.mempool_expand_size : context.mempool_init_size;
    if (*size < alloc_size) *size = alloc_size;
    if (*size > context.mempool_max_size)
    {
        CmiPrintf("Error: there is attempt to allocate memory block with size %lld which is greater than the maximum mempool allowed %lld.\n"
                  "Please increase the maximum mempool size by using +ofi-mempool-max-size\n",
                  *size, context.mempool_max_size);
        CmiAbort("alloc_mempool_block");
    }

    void *pool;
    ALIGNED_ALLOC(pool, *size);
    return pool;
}

void free_mempool_block(void *ptr, mem_handle_t mem_hndl)
{
    free(ptr);
}

#endif

void LrtsPreCommonInit(int everReturn)
{
    MACHSTATE(2, "OFI::LrtsPreCommonInit {");

    PRINT_THREAD_INFO();

#if USE_MEMPOOL
    CpvInitialize(mempool_type*, mempool);
    CpvAccess(mempool) = mempool_init(context.mempool_init_size,
                                      alloc_mempool_block,
                                      free_mempool_block,
                                      context.mempool_max_size);
#endif

    if (!CmiMyRank()) prepost_buffers();

    MACHSTATE(2, "} OFI::LrtsPreCommonInit");
}

void LrtsPostCommonInit(int everReturn)
{
    MACHSTATE(2, "OFI::LrtsPostCommonInit {");
    MACHSTATE(2, "} OFI::LrtsPostCommonInit");
}

void LrtsAdvanceCommunication(int whileidle)
{
    int processed_count;
    MACHSTATE(2, "OFI::LrtsAdvanceCommunication {");

    do
    {
        processed_count = 0;
        processed_count += process_completion_queue();
#if CMK_SMP
        processed_count += process_send_queue();
#endif
    } while (processed_count > 0);

    MACHSTATE(2, "} OFI::LrtsAdvanceCommunication done");
}

void LrtsDrainResources() /* used when exiting */
{
    int ret;
    MACHSTATE1(2, "OFI::LrtsDrainResources (PE=%i {", CmiMyPe());
    LrtsAdvanceCommunication(0);
    ret = runtime_barrier();
    if (ret) {
        MACHSTATE1(2, "runtime_barrier() returned %i", ret);
        CmiAbort("OFI::LrtsDrainResources failed");
    }
    MACHSTATE(2, "} OFI::LrtsDrainResources");
}

void* LrtsAlloc(int n_bytes, int header)
{
    void *ptr = NULL;
    size_t size = n_bytes + header;

#if USE_MEMPOOL
    if (size <= context.mempool_lb_size || size >= context.mempool_rb_size)
        ALIGNED_ALLOC(ptr, size);
    else
        ptr = mempool_malloc(CpvAccess(mempool), size, 1);
#else
    ALIGNED_ALLOC(ptr, size);
#endif

    if (!ptr) CmiAbort("LrtsAlloc");
    return ptr;
}

void LrtsFree(void *msg)
{
#if USE_MEMPOOL
    CmiUInt4 size = SIZEFIELD((char*)msg + sizeof(CmiChunkHeader)) + sizeof(CmiChunkHeader);
    if (size <= context.mempool_lb_size || size >= context.mempool_rb_size)
        free(msg);
    else
#if CMK_SMP
        mempool_free_thread(msg);
#else
        mempool_free(CpvAccess(mempool), msg);
#endif /* CMK_SMP */
#else
    free(msg);
#endif /* USE_MEMPOOL */
}

void LrtsExit(int exitcode)
{
    int        ret;
    int        i;
    OFIRequest *req;

    MACHSTATE(2, "OFI::LrtsExit {");

    LrtsAdvanceCommunication(0);

    for (i = 0; i < context.num_recv_reqs; i++)
    {
        req = context.recv_reqs[i];
        ret = fi_cancel((fid_t)context.ep, (void *)&(req->context));
        if (ret < 0) CmiAbort("fi_cancel error");
        CmiFree(req->data.recv_buffer);
#if USE_OFIREQUEST_CACHE
        free_request(req);
#else
        CmiFree(req);
#endif
    }

#if CMK_SMP
    PCQueueDestroy(context.send_queue);
#endif

    if (context.recv_reqs)
        free(context.recv_reqs);
    if (context.av)
        fi_close((struct fid *)(context.av));
    if (context.cq)
        fi_close((struct fid *)(context.cq));
    if (context.ep)
        fi_close((struct fid *)(context.ep));
    if (context.domain)
        fi_close((struct fid *)(context.domain));
    if (context.fabric)
        fi_close((struct fid *)(context.fabric));

#if USE_OFIREQUEST_CACHE
    destroy_request_cache(context.request_cache);
#endif

#if USE_MEMPOOL
    mempool_destroy(CpvAccess(mempool));
#endif

    if(!CharmLibInterOperate || userDrivenMode) {
        ret = runtime_barrier();
        if (ret) {
            MACHSTATE1(2, "runtime_barrier() returned %i", ret);
            CmiAbort("OFI::LrtsExit failed");
        }
        ret = runtime_fini();
        if (ret) {
            MACHSTATE1(2, "runtime_fini() returned %i", ret);
            CmiAbort("OFI::LrtsExit failed");
        }
        if (!userDrivenMode) {
          exit(exitcode);
        }
    }

    MACHSTATE(2, "} OFI::LrtsExit");
}

#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl()
{
    MACHSTATE(2, "OFI::CmiMachineProgressImpl {");
    if (CmiMyRank() == CmiMyNodeSize()) {
        CommunicationServerThread(0);
    } else {
        MACHSTATE(2, "skipping");
    }
    MACHSTATE(2, "} OFI::CmiMachineProgressImpl");
}
#endif

/* In CMK_SMP, this is called by worker thread */
void LrtsPostNonLocal()
{
    MACHSTATE(2, "OFI::LrtsPostNonLocal {");
    MACHSTATE(2, "} OFI::LrtsPostNonLocal");
}

void LrtsAbort(const char *message)
{
    MACHSTATE1(2, "OFI::LrtsAbort '%s' {", message);
    exit(1);
    MACHSTATE(2, "} OFI::LrtsAbort");
    CMI_NORETURN_FUNCTION_END
}

void  LrtsNotifyIdle()
{
    MACHSTATE(2, "OFI::LrtsNotifyIdle {");
    MACHSTATE(2, "} OFI::LrtsNotifyIdle");
}

void  LrtsBeginIdle()
{
    MACHSTATE(2, "OFI::LrtsBeginIdle {");
    MACHSTATE(2, "} OFI::LrtsBeginIdle");
}

void  LrtsStillIdle()
{
    MACHSTATE(2, "OFI::LrtsStillIdle {");
    MACHSTATE(2, "} OFI::LrtsStillIdle");
}

void  LrtsBarrier()
{
    int ret;
    MACHSTATE(2, "OFI::LrtsBarrier {");
    ret = runtime_barrier();
    if (ret) {
        MACHSTATE1(2, "runtime_barrier() returned %i", ret);
        CmiAbort("OFI::LrtsBarrier failed");
    }
    MACHSTATE(2, "} OFI::LrtsBarrier");
}

/* Other assist function */

/**
 * fill_av_ofi() is used during LrtsInit to exchange all the EP names and to
 * insert them into the AV. The exchange is performed using both PMI and OFI.
 * This is used by default. See +ofi_runtime_tcp flag for other option.
 */
static
int fill_av_ofi(int myid,
                int nnodes,
                struct fid_ep *ep,
                struct fid_av *av,
                struct fid_cq *cq)
{
    char                       my_epname[FI_NAME_MAX];
    size_t                     epnamelen;
    int                        max_keylen;
    char                      *key;
    char                      *epnames;
    size_t                     epnameslen;
    struct fi_context         *epnames_contexts;
    struct fi_cq_tagged_entry  e;
    size_t                     nexpectedcomps;
    int                        ret;
    int                        i;

    /**
     * Get our EP name. This will be exchanged with the other nodes.
     */
    epnamelen = sizeof(my_epname);
    ret = fi_getname((fid_t)ep, &my_epname, &epnamelen);
    CmiAssert(FI_NAME_MAX >= epnamelen);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_getname error");
    }

    /**
     * Publish our EP name.
     */
    ret = runtime_get_max_keylen(&max_keylen);
    if (ret) {
        CmiAbort("OFI::LrtsInit::runtime_get_max_keylen error");
    }

    key = (char *)malloc(max_keylen);
    CmiAssert(key);

    ret = snprintf(key, max_keylen, OFI_KEY_FORMAT_EPNAME, myid);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::snprintf error");
    }

    ret = runtime_kvs_put(key, &my_epname, epnamelen);
    if (ret) {
        CmiAbort("OFI::LrtsInit::runtime_kvs_put error");
    }

    /**
     * Allocate buffer which will contain all the EP names ordered by Node Id.
     * Once all the names are exchanged, they will be inserted into the AV.
     */
    epnameslen = FI_NAME_MAX * nnodes;
    epnames = (char *)malloc(epnameslen);
    CmiAssert(epnames);
    memset(epnames, 0, epnameslen);

    if (myid != 0) {
        /**
         * Non-root nodes expect a message which contains the EP names.
         */
        epnames_contexts = (struct fi_context *)malloc(sizeof(struct fi_context));
        CmiAssert(epnames_contexts);

        ret = fi_trecv(ep,
                       epnames,
                       epnameslen,
                       NULL,
                       FI_ADDR_UNSPEC,
                       OFI_OP_NAMES,
                       0ULL,
                       &epnames_contexts[0]);

        /* Reap 1 recv completion */
        nexpectedcomps = 1;
    }

    /**
     * Wait for all the other nodes to publish their EP names.
     */
    ret = runtime_barrier();
    if (ret) {
        CmiAbort("OFI::LrtsInit::runtime_barrier error");
    }

    if (myid == 0) {
        /**
         * Root gathers all the epnames and sends them to the other nodes.
         */

        /* Retrieve all epnames */
        for (i=0; i<nnodes; ++i) {
            memset(key, 0, max_keylen);
            ret = snprintf(key, max_keylen, OFI_KEY_FORMAT_EPNAME, i);
            if (ret < 0) {
                CmiAbort("OFI::LrtsInit::snprintf error");
            }

            ret = runtime_kvs_get(key, epnames+(i*epnamelen), epnamelen, i);
            if (ret) {
                CmiAbort("OFI::LrtsInit::runtime_kvs_get error");
            }
        }

        /* AV insert */
        ret = fi_av_insert(av, epnames, nnodes, NULL, 0, NULL);
        if (ret < 0) {
            CmiAbort("OFI::LrtsInit::fi_av_insert error");
        }

        /* Send epnames to everyone */
        epnames_contexts = (struct fi_context *)malloc(nnodes * sizeof(struct fi_context));
        CmiAssert(epnames_contexts);
        for (i=1; i<nnodes; ++i) {
            ret = fi_tsend(ep,
                           epnames,
                           epnameslen,
                           NULL,
                           i,
                           OFI_OP_NAMES,
                           &epnames_contexts[i]);
            if (ret) {
                CmiAbort("OFI::LrtsInit::fi_tsend error (+ofi_runtime_tcp may be needed)");
            }
        }

        /* Reap 1 send completion per non-root node */
        nexpectedcomps = nnodes - 1;
    }

    while (nexpectedcomps > 0) {
        memset(&e, 0, sizeof e);
        ret = fi_cq_read(cq, &e, 1);
        if (ret > 0) {
            /* A completion was found */
            if (((e.flags & FI_SEND) && (myid != 0)) ||
                ((e.flags & FI_RECV) && (myid == 0))) {
                /* This message was received in error */
                CmiAbort("OFI::LrtsInit::fi_cq_read unexpected completion.");
            }
            nexpectedcomps--;
        } else if(ret == -FI_EAGAIN) {
            /* Completion Queue is empty */
            continue;
        } else if (ret < 0) {
           CmiAbort("OFI::LrtsInit::fi_cq_read error.");
        }
    }

    if (myid != 0) {
        /**
         * Non-root nodes
         */

        /* AV insert */
        ret = fi_av_insert(av, epnames, nnodes, NULL, 0, NULL);
        if (ret < 0) {
            CmiAbort("OFI::LrtsInit::fi_av_insert error");
        }
    }

    free(key);
    free(epnames);
    free(epnames_contexts);

    return 0;
}

/**
 * fill_av() is used during LrtsInit to exchange all the EP names and to insert
 * them into the AV. The exchange is performed using PMI only. Currently
 * enabled only if +ofi_runtime_tcp flag is set.
 */
static
int fill_av(int myid,
            int nnodes,
            struct fid_ep *ep,
            struct fid_av *av,
            struct fid_cq *cq)
{
    char    my_epname[FI_NAME_MAX];
    size_t  epnamelen;
    int     max_keylen;
    char   *key;
    char   *epnames;
    size_t  epnameslen;
    int     ret;
    int     i;

    /**
     * Get our EP name. This will be exchanged with the other nodes.
     */
    epnamelen = sizeof(my_epname);
    ret = fi_getname((fid_t)ep, &my_epname, &epnamelen);
    CmiAssert(FI_NAME_MAX >= epnamelen);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_getname error");
    }

    /**
     * Publish our EP name.
     */
    ret = runtime_get_max_keylen(&max_keylen);
    if (ret) {
        CmiAbort("OFI::LrtsInit::runtime_get_max_keylen error");
    }

    key = (char *)malloc(max_keylen);
    CmiAssert(key);

    ret = snprintf(key, max_keylen, OFI_KEY_FORMAT_EPNAME, myid);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::snprintf error");
    }

    ret = runtime_kvs_put(key, &my_epname, epnamelen);
    if (ret) {
        CmiAbort("OFI::LrtsInit::runtime_kvs_put error");
    }

    /**
     * Allocate buffer which will contain all the EP names ordered by Node Id.
     * Once all the names are exchanged, they will be inserted into the AV.
     */
    epnameslen = FI_NAME_MAX * nnodes;
    epnames = (char *)malloc(epnameslen);
    CmiAssert(epnames);
    memset(epnames, 0, epnameslen);

    /**
     * Wait for all the other nodes to publish their EP names.
     */
    ret = runtime_barrier();
    if (ret) {
        CmiAbort("OFI::LrtsInit::runtime_barrier error");
    }

    /**
     * Retrieve all the EP names in order.
     */
    for (i=0; i<nnodes; ++i) {
        memset(key, 0, max_keylen);
        ret = snprintf(key, max_keylen, OFI_KEY_FORMAT_EPNAME, i);
        if (ret < 0) {
            CmiAbort("OFI::LrtsInit::snprintf error");
        }

        ret = runtime_kvs_get(key, epnames+(i*epnamelen), epnamelen, i);
        if (ret) {
            CmiAbort("OFI::LrtsInit::runtime_kvs_get error");
        }

    }

    /**
     * Insert all the EP names into the AV.
     */
    ret = fi_av_insert(av, epnames, nnodes, NULL, 0, NULL);
    if (ret < 0) {
        CmiAbort("OFI::LrtsInit::fi_av_insert error");
    }

    free(key);
    free(epnames);

    return 0;
}

#if CMK_ONESIDED_IMPL
#include "machine-onesided.C"
#endif
