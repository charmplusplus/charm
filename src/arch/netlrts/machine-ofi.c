/** @file
 * OFI (libfabric) implementation of Converse NET version
 * @ingroup NET
 * contains only OFI specific code for:
 * - CmiMachineInit()
 * - CmiCommunicationInit()
 * - CmiNotifyStillIdle()
 * - CmiNotifyIdle()
 * - DeliverViaNetwork()
 * - CommunicationServerNet()
 * - CmiMachineExit()
 *
 *  Copyright (c) 2016, Intel Corporation
 *  written by Yohann Burette <yohann.burette@intel.com>, Alexey Malkhanov <alexey.malhanov@intel.com>
 *  September 29, 2015
 *  Updated in September 2016
 */

/**
 * @addtogroup NET
 * @{
 */

/**
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
 */

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>

#define OFI_NUM_RECV_REQS 1

#define OFI_MR_KEY 0xCAFE

#define OFI_OP_SHORT 0x1
#define OFI_OP_LONG  0x2
#define OFI_OP_ACK   0x3

struct OFIRequest;
typedef  void (*callback_fn) (struct fi_cq_tagged_entry *, struct OFIRequest *);

static void poll_netlrts();

#ifdef HAVE_BUILTIN_EXPECT
#  define unlikely(x_) __builtin_expect(!!(x_),0)
#  define likely(x_)   __builtin_expect(!!(x_),1)
#else
#  define unlikely(x_) (x_)
#  define likely(x_)   (x_)
#endif

#define OFI_RETRY(func)                    \
    do{                                    \
        ssize_t _ret;                      \
		do{                                \
            _ret = func;                   \
            if (likely(_ret == 0)) break;  \
            if (_ret != -FI_EAGAIN){       \
                CmiAbort("OFI_RETRY error");         \
            }                              \
            poll_netlrts();                \
		}while(_ret == -FI_EAGAIN);        \
    } while(0)

/**
 * OFI RMA Header
 * Message sent by sender to receiver during RMA Read of long messages.
 *  - nodeNo: Target node number
 *  - src_ogm: Address of source OutgoingMsg; Sent back as part of OFIRmaAck
 *  - buf: Address of data at the target
 *  - len: Length of message
 *  - key: Remote key
 */
struct OFIRmaHeader {
    int      nodeNo;
    uint64_t src_ogm;
    uint64_t buf;
    uint64_t len;
    uint64_t key;
};

/**
 * OFI RMA Ack
 * Message sent by receiver to sender during RMA Read of long messages.
 *  - src_ogm: Address of source OutgoingMsg; Received as part of OFIRmaHeader
 */
struct OFIRmaAck {
    uint64_t src_ogm;
};

/**
 * OFI Long Message
 * Structure stored by the receiver about ongoing RMA Read of long message.
 *  - asm_msg: Assembly buffer where the data is RMA Read into
 *  - nodeNo: Target node number
 *  - remote_buf: Address of data at the target
 *  - rma_ack: OFI Rma Ack sent to sender once all the data has been RMA Read
 *  - completion_count: Number of expected RMA Read completions
 *  - mr: Memory Region where the data is RMA Read into
 */
struct OFILongMsg {
    char                *asm_msg;
    int                 nodeNo;
    uint64_t            remote_buf;
    struct OFIRmaAck    rma_ack;
    size_t              completion_count;
    struct fid_mr       *mr;
};

/**
 * OFI Request
 * Structure representing data movement operations.
 *  - context: fi_context
 *  - callback: Request callback called upon completion
 *  - data: Pointer to data associated with the request
 *      - recv_buffer: used when posting a receive buffer
 *      - rma_header: used when an OFIRmaHeader was received or sent
 *      - rma_ack: used when an OFIRmaAck was received
 *      - long_msg: used when an RMA Read operation completed
 *      - ogm: used when a short message was sent
 */
struct OFIRequest {
    struct fi_context   context;
    callback_fn         callback;
    union {
        void                *recv_buffer;
        struct OFIRmaHeader *rma_header;
        struct OFIRmaAck    *rma_ack;
        struct OFILongMsg   *long_msg;
        OutgoingMsg         ogm;
    } data;
};

/**
 * Retrieve the OFI Request associated with the given fi_context.
 */
#define TO_OFI_REQ(_ptr_context) \
    container_of((_ptr_context), struct OFIRequest, context)

#define MIN(a, b) \
    ((a) <= (b)) ? a : b;

struct OFIContext {
    /** Fabric Domain handle */
    struct fid_fabric *fabric;

    /** Access Domain handle */
    struct fid_domain *domain;

    /** Address vector handle */
    struct fid_av *av;

    /** Completion queue handle */
    struct fid_cq *cq;

    /** Memory region handle */
    struct fid_mr *mr;

    /** Endpoint to communicate on */
    struct fid_ep *ep;

    /** Local EP name */
    char my_epname[FI_NAME_MAX];

    /** EP name length */
    size_t epnamelen;

    /** Pre-posted receive requests */
    struct OFIRequest *recv_reqs;

    /**
     * Maximum size for eager messages.
     * RMA Read for larger messages.
     */
    size_t eager_size;

    /**
     * Maximum size for RMA operations.
     * Multiple RMA operations for larger messages.
     */
    size_t max_rma_size;

    /**
     * MR mode:
     *  - FI_MR_SCALABLE allows us to register all the memory with our own key,
     *  - FI_MR_BASIC requires us to register the RMA buffers
     *    and to exchange the keys.
     */
    enum fi_mr_mode mr_mode;

};

static void recv_callback(struct fi_cq_tagged_entry *e, struct OFIRequest *req);
static void rma_read_callback(struct fi_cq_tagged_entry *e, struct OFIRequest *req);

static struct OFIContext *context = NULL;
FILE * pFile;

void CmiMachineInit(char **argv)
{
    struct fi_info               *providers;
    struct fi_info               *prov;
    struct fi_info               *hints;
    struct fi_domain_attr         domain_attr = {0};
    struct fi_tx_attr             tx_attr = { 0 };
    struct fi_cq_attr             cq_attr = { 0 };
    struct fi_av_attr             av_attr = { 0 };
    int                           fi_version;
    int                           min_space;

    int                           i;
    int                           ret;

    MACHSTATE1(3, "PID %d - CmiMachineInit", getpid());

    context = CmiAlloc(sizeof *context);
    CmiAssert(NULL != context);

    /**
     * Hints to filter providers
     * See man fi_getinfo for a list of all filters
     * mode: This OFI machine will pass in context into communication calls
     * ep_type: Reliable datagram operation
     * caps: Capabilities required from the provider. We want to use the
     *       tagged message queue and rma read APIs.
     */
    hints = fi_allocinfo();
    CmiAssert(NULL != hints);
    hints->mode                         = FI_CONTEXT;
    hints->ep_attr->type                = FI_EP_RDM;
    hints->domain_attr->resource_mgmt   = FI_RM_ENABLED;
    hints->caps                         = FI_TAGGED;
    hints->caps                        |= FI_RMA;
    hints->caps                        |= FI_REMOTE_READ;

    /**
     * FI_VERSION provides binary backward and forward compatibility support
     * Specify the version of OFI this machine is coded to, the provider will
     * select struct layouts that are compatible with this version.
     */
    fi_version = FI_VERSION(1, 0);

    ret = fi_getinfo(fi_version, NULL, NULL, 0ULL, hints, &providers);
    if (ret < 0) {
        MACHSTATE1(3, "fi_getinfo error: %d", ret);
        CmiAbort("fi_getinfo error");
    }

    if (providers == NULL) {
        CmiAbort("No message provider found");
    }

    /**
     * Here we elect to use the first provider from the list.
     * Further filtering could be done at this point (e.g. name).
     */
    prov = providers;

    MACHSTATE1(3, "Using provider '%s'", prov->fabric_attr->name);

    //context->eager_size = prov->ep_attr->max_msg_size / 2;
    //context->eager_size = prov->ep_attr->max_msg_size;
    context->eager_size = 65536;

    context->max_rma_size = prov->ep_attr->max_msg_size;
    context->mr_mode = prov->domain_attr->mr_mode;

    MACHSTATE1(3, "Maximum eager message size: %ld", context->eager_size);
    MACHSTATE1(3, "Maximum rma size: %ld", context->max_rma_size);
    MACHSTATE1(3, "MR mode: %d", context->mr_mode);

    if ((context->mr_mode != FI_MR_BASIC) &&
        (context->mr_mode != FI_MR_SCALABLE)) {
        MACHSTATE1(3, "Unsupported MR mode : %d", context->mr_mode);
        CmiAbort("Unsupported MR mode");
    }

    /**
     * Open fabric
     * The getinfo struct returns a fabric attribute struct that can be used to
     * instantiate the virtual or physical network. This opens a "fabric
     * provider". See man fi_fabric for details.
     */
    ret = fi_fabric(prov->fabric_attr, &context->fabric, NULL);
    if (ret < 0) {
        MACHSTATE1(3, "fi_fabric error: %s", fi_strerror(-ret));
        fi_freeinfo(providers);
        CmiAbort("fi_fabric error");
    }

    /**
     * Create the access domain, which is the physical or virtual network or
     * hardware port/collection of ports.  Returns a domain object that can be
     * used to create endpoints.  See man fi_domain for details.
     */
    ret = fi_domain(context->fabric, prov, &context->domain, NULL);
    if (ret < 0) {
        MACHSTATE1(3, "fi_domain error: %s", fi_strerror(-ret));
        fi_freeinfo(providers);
        CmiAbort("fi_domain error");
    }

    /**
     * Create a transport level communication endpoint.  To use the endpoint,
     * it must be bound to completion counters or event queues and enabled,
     * and the resources consumed by it, such as address vectors, counters,
     * completion queues, etc.
     * see man fi_endpoint for more details.
     */
    ret = fi_endpoint(context->domain, /* In:  Domain object   */
                      prov,            /* In:  Provider        */
                      &context->ep,    /* Out: Endpoint object */
                      NULL);           /* Optional context     */
    if (ret < 0) {
        MACHSTATE1(3, "fi_endpoint error: %s", fi_strerror(-ret));
        fi_freeinfo(providers);
        CmiAbort("fi_endpoint error");
    }

    /**
     * Create the objects that will be bound to the endpoint.
     * The objects include:
     *     - completion queue for events
     *     - address vector of other endpoint addresses
     *     - dynamic memory-spanning memory region
     */
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    ret = fi_cq_open(context->domain, &cq_attr, &context->cq, NULL);
    if (ret < 0) {
        MACHSTATE1(3, "fi_cq_open error: %s\n", fi_strerror(-ret));
        CmiAbort("fi_cq_open error");
    }

    /**
     * The remote fi_addr will be stored in the OtherNodeStruct struct.
     * (see machine-dgram.c for details)
     * So, we use the AV in "map" mode.
     */
    /* TODO: update comments */
#if 0
    av_attr.type = FI_AV_TABLE;
#else
    av_attr.type = FI_AV_MAP;
#endif
    ret = fi_av_open(context->domain, &av_attr, &context->av, NULL);
    if (ret < 0) {
        MACHSTATE1(3, "fi_av_open error: %s\n", fi_strerror(-ret));
        CmiAbort("fi_av_open error");
    }

    context->mr = NULL;
    if (FI_MR_SCALABLE == context->mr_mode) {
        /**
         * Create Memory Region.
         * All OFI communication routines require an MR.
         */
        ret = fi_mr_reg(context->domain,
                        0,
                        UINTPTR_MAX,
                        FI_SEND | FI_RECV | FI_REMOTE_READ | FI_READ,
                        0ULL,
                        OFI_MR_KEY,
                        0ULL,
                        &context->mr,
                        NULL);
        if (ret < 0) {
            MACHSTATE1(3, "fi_mr_reg error: %s\n", fi_strerror(-ret));
            CmiAbort("fi_mr_reg error");
        }
    }

    /**
     * Bind the CQ and AV to the endpoint object.
     */
    ret = fi_ep_bind(context->ep,
                     (fid_t)context->cq,
                     FI_RECV | FI_TRANSMIT);
    if (ret < 0) {
        MACHSTATE1(3, "fi_bind EP-CQ error: %s\n", fi_strerror(-ret));
        CmiAbort("fi_bind EP-CQ error");
    }
    ret = fi_ep_bind(context->ep,
                     (fid_t)context->av,
                     0);
    if (ret < 0) {
        MACHSTATE1(3, "fi_bind EP-AV error: %s\n", fi_strerror(-ret));
        CmiAbort("fi_bind EP-AV error");
    }

    /**
     * Enable the endpoint for communication
     * This commits the bind operations.
     */
    ret = fi_enable(context->ep);
    if (ret < 0) {
        MACHSTATE1(3, "fi_enable error: %s\n", fi_strerror(ret));
        CmiAbort("fi_enable error");
    }

    /**
     * Get our EP name. This will be exchanged with the other nodes.
     * See node_addresses_obtain() in machine.c for details.
     */
    context->epnamelen = sizeof(context->my_epname);
    ret = fi_getname((fid_t)context->ep,
                     &context->my_epname,
                     &context->epnamelen);
    CmiAssert(FI_NAME_MAX >= context->epnamelen);
    if (ret < 0) {
        MACHSTATE1(3, "fi_getname error: %s\n", fi_strerror(ret));
        CmiAbort("fi_getname error");
    }

    MACHSTATE1(3, "EP name length: %d", context->epnamelen);

    /**
     * Free providers info since it's not needed anymore.
     */
    fi_freeinfo(hints);
    hints = NULL;
    fi_freeinfo(providers);
    providers = NULL;

    MACHSTATE(3, "} CmiMachineInit");

}

/**
 * Initialize the remote nodes.
 * This function is called in machine-dgram.c.
 */
void fabric_OtherNodes_init(ChNodeinfo *data, OtherNode nodes, int n)
{
    /**
     * Initialize the OtherNode structure for each node.
     * See machine-dgram.c for its definition.
     */

    int ret;
    int namelen;
    char *epnames;
    fi_addr_t *fi_addrs;
    int i;

    MACHSTATE(3, "fabricInitNodes {");

    CmiAssert(data);
    CmiAssert(n > 0);

    /* Assuming all names have the same length */
    namelen = ChMessageInt(data[0].epnamelen);

    epnames = CmiAlloc(n * namelen);
    CmiAssert(epnames);
    fi_addrs = CmiAlloc(n * sizeof(fi_addr_t));
    CmiAssert(fi_addrs);

    /* Extract epnames from data */
    for(i=0; i<n; ++i) {
        memcpy(epnames + (i*namelen), data[i].epname, namelen);
    }

    /* Insert epnames into the AV */
    ret = fi_av_insert(context->av, epnames, n, fi_addrs, 0, NULL);
    if (ret < 0) {
        MACHSTATE1(3, "fi_av_insert error: %d\n", ret);
        CmiAbort("fi_av_insert error");
    }

    /* Store the fi_addrs */
    for(i=0; i<n; ++i) {
        nodes[i].nodeNo  = i;
        nodes[i].fi_addr = fi_addrs[i];
    }

    CmiFree(epnames);
    CmiFree(fi_addrs);

    MACHSTATE(3, "} fabricInitNodes done");
}

void CmiCommunicationInit(char **argv)
{
}

/**
 * CmiRecvQueuesInit() initializes the local receive buffer(s).
 * This function is called by ConverseRunPE() in machine.c.
 */
void CmiRecvQueuesInit(void)
{
    struct OFIRequest   *reqs;
    size_t              block_size;
    int                 ret;
    int                 i;

    MACHSTATE(3, "CmiRecvQueuesInit {");

    reqs = CmiAlloc(sizeof(struct OFIRequest) * OFI_NUM_RECV_REQS);
    CmiAssert(reqs);

    block_size = context->eager_size;

    for(i = 0; i < OFI_NUM_RECV_REQS; i++) {
        /*TODO: why is this still a malloc and not a CmiAlloc? */
        reqs[i].data.recv_buffer = malloc(block_size);
        reqs[i].callback = recv_callback;

        MACHSTATE2(3, "---> posting recv req %p buf=%p",
                   &reqs[i], reqs[i].data.recv_buffer);

        /* Receive from any node with any tag */
        OFI_RETRY(fi_trecv(context->ep,
                           reqs[i].data.recv_buffer,
                           block_size,
                           NULL,
                           FI_ADDR_UNSPEC,
                           0,
                           0xFFFFFFFFFFFFFFFFULL,
                           &reqs[i].context));
    }

    context->recv_reqs      = reqs;

    MACHSTATE(3, "} CmiRecvQueuesInit");
}

void LrtsStillIdle() {}
void LrtsNotifyIdle() {}
void LrtsBeginIdle() {}

/******************
Check the communication server socket and

*****************/
int CheckSocketsReady(int withDelayMs)
{
    int nreadable;
    CMK_PIPE_DECL(withDelayMs);

    CmiStdoutAdd(CMK_PIPE_SUB);
    if (Cmi_charmrun_fd!=-1) CMK_PIPE_ADDREAD(Cmi_charmrun_fd);

    nreadable=CMK_PIPE_CALL();
    ctrlskt_ready_read = 0;
    dataskt_ready_read = 0;
    dataskt_ready_write = 0;

    if (nreadable == 0) {
        MACHSTATE(1,"} CheckSocketsReady (nothing readable)")
        return nreadable;
    }
    if (nreadable==-1) {
        CMK_PIPE_CHECKERR();
        MACHSTATE(2,"} CheckSocketsReady (INTERRUPTED!)")
        return CheckSocketsReady(0);
    }
    CmiStdoutCheck(CMK_PIPE_SUB);
    if (Cmi_charmrun_fd!=-1)
        ctrlskt_ready_read = CMK_PIPE_CHECKREAD(Cmi_charmrun_fd);
    MACHSTATE(1,"} CheckSocketsReady")
    return nreadable;
}


/*** Service the charmrun socket
*************/

static void ServiceCharmrun_nolock()
{
    int again = 1;
    MACHSTATE(2,"ServiceCharmrun_nolock begin {")
    while (again)
    {
        again = 0;
        CheckSocketsReady(0);
        if (ctrlskt_ready_read) {
            ctrl_getone();
            again=1;
        }
        if (CmiStdoutNeedsService()) {
            CmiStdoutService();
        }
    }
    MACHSTATE(2,"} ServiceCharmrun_nolock end")
}


static inline
void process_short_recv(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * A short message was received:
     *   - Copy the data out of the recv buffer,
     *   - Pass the message to the upper layer.
     */

    char    *data;
    char    *asm_msg;
    size_t  msg_size;
    int     drank, spe, seqno, magic, broot;

    data = req->data.recv_buffer;
    CmiAssert(data);

    msg_size = CMI_MSG_SIZE(data);
    MACHSTATE2(3, "--> eager msg (e->len=%ld msg_size=%ld)", e->len, msg_size);
    CmiAssert(msg_size == e->len);

    DgramHeaderBreak(data, drank, spe, magic, seqno, broot);
    MACHSTATE2(3, "--> Received msg (size=%ld seqno=%d)", msg_size, seqno);

    asm_msg = CmiAlloc(msg_size);
    CmiAssert(asm_msg);

    memcpy(asm_msg, data, e->len);

    handleOneRecvedMsg(msg_size, asm_msg);
}

static inline
void process_long_recv(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * An OFIRmaHeader was received:
     *   - Allocate enough space to store the long message,
     *   - Create OFILongMsg to keep track of the data retrieval,
     *   - Issue the RMA Read operation(s) to retrieve the data.
     */

    int ret;
    struct OFILongMsg *long_msg;
    struct OFIRequest *rma_req;
    struct OFIRmaHeader *rma_header;
    struct fid_mr *mr;
    char *asm_buf;
    int nodeNo;
    uint64_t rbuf;
    size_t len;
    uint64_t rkey;
    uint64_t rogm;
    char *lbuf;
    size_t remaining;
    size_t chunk_size;

    CmiAssert(e->len == sizeof(struct OFIRmaHeader));

    /**
     * Parse header
     */
    rma_header = req->data.rma_header;

    nodeNo = rma_header->nodeNo;
    rogm = rma_header->src_ogm;
    rbuf = rma_header->buf;
    len = rma_header->len;
    rkey = rma_header->key;

    MACHSTATE1(3, "--> Receiving long msg len=%ld", len);

    /**
     * Prepare buffer
     */
    asm_buf = CmiAlloc(len);
    CmiAssert(asm_buf);

    if (FI_MR_SCALABLE == context->mr_mode) {
        /* Use global MR */
        mr = context->mr;
    } else {
        /* Register new MR to read into */
        ret = fi_mr_reg(context->domain,
                        asm_buf,
                        len,
                        FI_READ | FI_RECV,
                        0ULL,
                        OFI_MR_KEY,
                        0ULL,
                        &mr,
                        NULL);
        if (ret) {
            MACHSTATE1(3, "fi_mr_reg error: %d\n", ret);
            CmiAbort("fi_mr_reg error");
        }
    }

    /**
     * Save some information about the RMA Read operation(s)
     */
    long_msg = CmiAlloc(sizeof *long_msg);
    CmiAssert(long_msg);
    long_msg->asm_msg          = asm_buf;
    long_msg->nodeNo           = nodeNo;
    long_msg->remote_buf       = rbuf;
    long_msg->rma_ack.src_ogm  = rogm;
    long_msg->completion_count = 0;
    long_msg->mr               = mr;

    /**
     * Issue RMA Read request(s)
     */
    remaining = len;
    lbuf      = asm_buf;
    rbuf      = rbuf;

    while (remaining > 0) {
        /* Determine size of operation */
        chunk_size = MIN(remaining, context->max_rma_size);

        rma_req = CmiAlloc(sizeof *rma_req);
        CmiAssert(rma_req);
        rma_req->callback = rma_read_callback;
        rma_req->data.long_msg = long_msg;

        /* Increment number of expected completions */
        long_msg->completion_count++;

        MACHSTATE5(3, "---> RMA Read lbuf %p rbuf %p ogm %p len %ld chunk #%d",
                   lbuf, rbuf, rogm, chunk_size, long_msg->completion_count);

        OFI_RETRY(fi_read(context->ep,
                          lbuf,
                          chunk_size,
                          fi_mr_desc(mr),
                          nodes[nodeNo].fi_addr,
                          rbuf,
                          rkey,
                          &rma_req->context));

        remaining  -= chunk_size;
        lbuf       += chunk_size;
        rbuf       += chunk_size;
    }
}

static inline
void process_long_send_ack(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * An OFIRmaAck was received:
     *  - Retrieve original OutgoingMsg
     *  - Decrement its refcount
     */
    OutgoingMsg ogm;
    int    drank, spe, seqno, magic, broot;

    ogm = (OutgoingMsg)req->data.rma_ack->src_ogm;
    CmiAssert(ogm);
    CmiAssert(ogm->refcount > 0);

    DgramHeaderBreak(ogm->data, drank, spe, magic, seqno, broot);
    MACHSTATE1(3, "--> Finished sending msg seqno=%d", seqno);

    ogm->refcount--;
    GarbageCollectMsg(ogm);
}


static inline
void recv_callback(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * Some data was received:
     *  - the tag tells us what type of message it is; process it
     *  - repost recv request
     */
    MACHSTATE(3, "recv_callback {");

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
    default:
        MACHSTATE2(3, "--> unknown operation %x len=%ld", e->tag, e->len);
        CmiAbort("!! Wrong operation !!");
    }

    MACHSTATE2(3, "Reposting recv req %p buf=%p", req, req->data.recv_buffer);
    OFI_RETRY(fi_trecv(context->ep,
                       req->data.recv_buffer,
                       context->eager_size,
                       NULL,
                       FI_ADDR_UNSPEC,
                       0,
                       0xFFFFFFFFFFFFFFFFULL,
                       &req->context));

    MACHSTATE(3, "} recv_callback done");
}

static inline
void send_ack_callback(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * An OFIRmaAck was sent (see rma_read_callback()).
     * Free up the resources.
     */
    struct OFILongMsg *long_msg;

    MACHSTATE(3, "send_ack_callback {");

    long_msg = req->data.long_msg;
    CmiAssert(long_msg);

    if (FI_MR_BASIC == context->mr_mode) {
        fi_close((struct fid*)long_msg->mr);
    }

    CmiFree(long_msg);
    CmiFree(req);

    MACHSTATE(3, "} send_ack_callback done");
}

static inline
void rma_read_callback(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * An RMA Read operation completed.
     */
    struct OFILongMsg *long_msg;
    struct OFIRequest *ack_req;
    int                drank, spe, seqno, magic, broot;

    MACHSTATE(3, "rma_read_callback {");

    long_msg = req->data.long_msg;
    CmiAssert(long_msg);
    CmiAssert(long_msg->completion_count > 0);

    long_msg->completion_count--;
    MACHSTATE1(3, "--> completion_count=%ld", long_msg->completion_count);

    if (0 == long_msg->completion_count) {
        /**
         *  The message has been RMA Read completely.
         *  Send ACK to notify the other side that we are done.
         */
        ack_req = CmiAlloc(sizeof *ack_req);
        CmiAssert(ack_req);

        ack_req->callback = send_ack_callback;
        ack_req->data.long_msg = long_msg;

        OFI_RETRY(fi_tsend(context->ep,
                           &long_msg->rma_ack,
                           sizeof long_msg->rma_ack,
                           NULL,
                           nodes[long_msg->nodeNo].fi_addr,
                           OFI_OP_ACK,
                           &ack_req->context));

        /**
         * Pass received message to upper layer.
         */
        DgramHeaderBreak(long_msg->asm_msg, drank, spe, magic, seqno, broot);
        MACHSTATE2(3, "--> Sent Ack node=%d seqno=%d", long_msg->nodeNo, seqno);
        MACHSTATE1(3, "--> Finished receiving msg seqno=%d", seqno);

        handleOneRecvedMsg(CMI_MSG_SIZE(long_msg->asm_msg), long_msg->asm_msg);
    }
    CmiFree(req);

    MACHSTATE(3, "} rma_read_callback done");
}

static void poll_netlrts()
{
    /**
     * Progress routine.
     */
    int ret;
    struct fi_cq_tagged_entry e;
    struct fi_cq_err_entry error;
    struct OFIRequest *req;

    do {
        memset(&e, 0, sizeof e);
        ret = fi_cq_read(context->cq, &e, 1);
        if (ret > 0) {
            /* A completion was found */
            if (NULL != e.op_context) {
                /* Retrieve request from context */
                req = TO_OFI_REQ(e.op_context);

                /* Execute request callback */
                if ((e.flags & FI_SEND) ||
                    (e.flags & FI_RECV) ||
                    (e.flags & FI_RMA)) {
                    req->callback(&e, req);
                } else {
                    MACHSTATE1(3, "Missed event with flags=%x", e.flags);
                    CmiAbort("!! Missed an event !!");
                }
            } else {
                CmiAbort("Error: op_context is NULL...");
            }
        } else if(ret == -FI_EAGAIN) {
            /* Completion Queue is empty */
            continue;
        } else if (ret < 0) {
            MACHSTATE1(3, "POLL: Error %d\n", ret);
            if (ret == -FI_EAVAIL) {
                MACHSTATE(3, "POLL: error available\n");
                ret = fi_cq_readerr(context->cq, (void *)&error, sizeof(error));
                if (ret < 0) {
                    CmiAbort("can't retrieve error");
                }
                MACHSTATE2(3, "POLL: error is %d (ret=%d)\n", error.err, ret);
            }
            CmiAbort("Polling error");
        }
    } while (ret > 0);
}

static inline
void send_short_callback(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * A short message was sent.
     * Free up resources.
     */
    OutgoingMsg ogm;

    MACHSTATE(3, "send_short_callback {");

    ogm = req->data.ogm;
    CmiAssert(ogm);
    ogm->refcount--;
    MACHSTATE2(3, "--> ogm=%p refcount=%d", ogm, ogm->refcount);
    GarbageCollectMsg(ogm);

    CmiFree(req);

    MACHSTATE(3, "} send_short_callback done");
}

static inline
void send_rma_callback(struct fi_cq_tagged_entry *e, struct OFIRequest *req)
{
    /**
     * An OFIRmaHeader was sent.
     * Free up resources.
     */
    struct OFIRmaHeader *header;

    MACHSTATE(3, "send_rma_callback {");

    header = req->data.rma_header;
    CmiFree(header);
    CmiFree(req);

    MACHSTATE(3, "} send_rma_callback done");
}

void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy)
{
    int                 ret;
    int                 seqno;
    struct OFIRequest   *req, *ack_req;
    struct OFIRmaHeader *rma_header;
    struct fid_mr       *mr;
    uint64_t            op;
    void                *buf;
    size_t              len;

    seqno = node->send_next;
    node->send_next = ((seqno+1)&DGRAM_SEQNO_MASK);

    DgramHeaderMake(ogm->data, rank, ogm->src, Cmi_charmrun_pid, seqno, broot);

    MACHSTATE3(3, "DeliverViaNetwork node=%d len=%d seqno=%d {",
               node->nodeNo, ogm->size, seqno);

    MACHSTATE3(3, "--> src=%d dst=%d broot=%d", ogm->src, rank, broot);

    req = CmiAlloc(sizeof *req);
    CmiAssert(req);

    ogm->refcount++;

    if (ogm->size <= context->eager_size) {
        /**
         * The message is small enough to be sent entirely.
         */
        MACHSTATE(3, "--> eager");

        req->callback = send_short_callback;
        req->data.ogm = ogm;

        op = OFI_OP_SHORT;
        buf = ogm->data;
        len = ogm->size;
    } else {
        /**
         * The message is too long to be sent directly.
         * Let other side use RMA Read instead by sending an OFIRmaHeader.
         */
        MACHSTATE(3, "--> long");

        rma_header = CmiAlloc(sizeof *rma_header);
        CmiAssert(rma_header);

        if (FI_MR_SCALABLE == context->mr_mode) {
            /* Use global MR */
            mr = context->mr;
        } else {
            /* Register new MR to RMA Read from */
            ret = fi_mr_reg(context->domain,
                            ogm->data,
                            ogm->size,
                            FI_SEND | FI_REMOTE_READ,
                            0ULL,
                            OFI_MR_KEY,
                            0ULL,
                            &mr,
                            NULL);
            if (ret) {
                MACHSTATE1(3, "fi_mr_reg error: %d\n", ret);
                CmiAbort("fi_mr_reg error");
            }
        }

        rma_header->nodeNo  = CmiMyNode();
        rma_header->src_ogm = ogm;
        rma_header->buf     = (uint64_t)ogm->data;
        rma_header->len     = ogm->size;
        rma_header->key     = fi_mr_key(mr);

        req->callback           = send_rma_callback;
        req->data.rma_header    = rma_header;

        op = OFI_OP_LONG;
        buf = rma_header;
        len = sizeof *rma_header;
    }

    OFI_RETRY(fi_tsend(context->ep,
                       buf,
                       len,
                       NULL,
                       node->fi_addr,
                       op,
                       &req->context));

    MACHSTATE(3, "} DeliverViaNetwork");
}

void MachineExit()
{
    int                  ret;
    int                  i;
    struct OFIRequest   *req;

    MACHSTATE(3, "CmiMachineExit {");

    for(i = 0; i < OFI_NUM_RECV_REQS; i++) {
        req = &context->recv_reqs[i];
        ret = fi_cancel((fid_t)context->ep, (void *)&req->context);
        if (ret < 0) {
            CmiAbort("fi_cancel error");
        }
        free(req->data.recv_buffer);
    }
    if (context->recv_reqs) {
        CmiFree(context->recv_reqs);
    }
    if (context->av)
        fi_close((struct fid *)(context->av));
    if (context->cq)
        fi_close((struct fid *)(context->cq));
    if (context->ep)
        fi_close((struct fid *)(context->ep));
    if (context->mr)
        fi_close((struct fid *)(context->mr));
    if (context->domain)
        fi_close((struct fid *)(context->domain));
    if (context->fabric)
        fi_close((struct fid *)(context->fabric));

    if (context)
        CmiFree(context);
    MACHSTATE(3, "} CmiMachineExit");
}

static void CommunicationServerNet(int sleepTime, int where)
{

    if (COMM_SERVER_FROM_INTERRUPT == where) {
#if CMK_IMMEDIATE_MSG
        CmiHandleImmediate();
#endif
        return;
    }

    ServiceCharmrun_nolock();
    poll_netlrts();
    /* when called by communication thread or in interrupt */
#if CMK_IMMEDIATE_MSG
    if (COMM_SERVER_FROM_SMP == where) {
        CmiHandleImmediate();
    }
#endif
}


/*@}*/
