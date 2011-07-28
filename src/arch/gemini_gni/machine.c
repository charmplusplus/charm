/*****************************************************************************
 * $Source$
 * $Author$  Yanhua Sun
 * $Date$  07-01-2011
 * $Revision$ 
 *****************************************************************************/

/** @file
 * Gemini GNI machine layer
 */
/*@{*/

#include <stdio.h>
#include <errno.h>
#include "converse.h"

#include "gni_pub.h"
#include "mpi.h"
#include "pmi.h"
/*Support for ++debug: */
#if defined(_WIN32) && ! defined(__CYGWIN__)
#include <windows.h>
#include <wincon.h>
#include <sys/types.h>
#include <sys/timeb.h>
static void sleep(int secs) {
    Sleep(1000*secs);
}
#else
#include <unistd.h> /*For getpid()*/
#endif
#include <stdlib.h> /*For sleep()*/

#include "machine.h"

#include "pcqueue.h"

#define DEBUY_PRINT

#ifdef DEBUY_PRINT
#define PRINT_INFO(msg) {fprintf(stdout, "%s\n", msg); fflush(stdout);}
#else
#define PRINT_INFO(msg)
#endif
/* =======Beginning of Definitions of Performance-Specific Macros =======*/
static int useSMSG   = 1;
static int useDynamicSmsg = 0;
static int useMsgq = 0;
static int size, rank;
#define FMA_PER_CORE  1024
#define FMA_BUFFER_SIZE 1024
#define SMSG_PER_MSG    1024
#define SMSG_MAX_CREDIT 16
#define SMSG_BUFFER_SIZE        10240
#define FMA_BTE_THRESHOLD  4096
#define MSGQ_MAXSIZE       4096

#define DEBUG

#ifdef GNI_RC_CHECK
#undef GNI_RC_CHECK
#endif

#ifdef DEBUG
#define GNI_RC_CHECK(msg,rc) do { if(rc != GNI_RC_SUCCESS) {           printf("%s; err=%s\n",msg,gni_err_str[rc]); exit(911); } } while(0)
#else
#define GNI_RC_CHECK(msg,rc)
#endif

static gni_nic_handle_t      nic_hndl;
static unsigned int         *MPID_UGNI_AllAddr;
static gni_smsg_attr_t      *smsg_attr;
static gni_smsg_attr_t      *smsg_attr_vec;
static char                 *smsg_mem_buffer = NULL;
static uint32_t             smsg_memlen;

gni_msgq_attr_t         msgq_attrs;
gni_msgq_handle_t       msgq_handle;
gni_msgq_ep_attr_t      msgq_ep_attrs;
gni_msgq_ep_attr_t      msgq_ep_attrs_size;

#define REMOTE_QUEUE_ENTRIES  1048576
#define LOCAL_QUEUE_ENTRIES 1024
/* SMSG is data message */
#define DATA_TAG        0
/* SMSG is a control message to initialize a BTE */
#define LMSG_INIT_TAG        1 
/* =====Beginning of Declarations of Machine Specific Variables===== */
static int cookie;
static int modes = 0;
static gni_cq_handle_t       rx_cqh = NULL;
static gni_cq_handle_t       remote_bte_cq_hndl = NULL;
static gni_cq_handle_t       tx_cqh = NULL;
static gni_ep_handle_t       *ep_hndl_array;

/* preallocated memory buffer for FMA for short message and control message */
static char            *fma_buffer;
static int              fma_buffer_len_eachcore = FMA_BUFFER_SIZE;
typedef struct {
    gni_mem_handle_t mdh;
    uint64_t addr;
} mdh_addr_t;

static mdh_addr_t            my_mdh_addr;
static mdh_addr_t            *fma_buffer_mdh_addr_base;
static gni_mem_handle_t      fma_buffer_mdh_addr;
static gni_mem_handle_t      my_smsg_mdh_mailbox;
/* =====Beginning of Declarations of Machine Specific Functions===== */

typedef struct msg_list
{
    int destNode;
    int size;
    void *msg;
    struct msg_list *next;

}MSG_LIST;

typedef struct control_msg
{
    int             source;   //source rank
    uint64_t        source_addr;
    gni_mem_handle_t    source_mem_hndl;
    uint64_t            length;
}CONTROL_MSG;

/* LrtsSent is called but message can not be sent by SMSGSend because of mailbox full or no credit */
static MSG_LIST *buffered_smsg_head= 0;
static MSG_LIST *buffered_smsg_tail= 0;
/* SmsgSend return success but message sent is not confirmed by remote side */

static MSG_LIST *buffered_fma_head = 0;
static MSG_LIST *buffered_fma_tail = 0;

static unsigned int get_gni_nic_address(int device_id)
{
    unsigned int address, cpu_id;
    gni_return_t status;
    int i, alps_dev_id=-1,alps_address=-1;
    char *token, *p_ptr;

    p_ptr = getenv("PMI_GNI_DEV_ID");
    if (!p_ptr) {
        status = GNI_CdmGetNicAddress(device_id, &address, &cpu_id);
       
        GNI_RC_CHECK("GNI_CdmGetNicAddress", status);
    } else {
        while ((token = strtok(p_ptr,":")) != NULL) {
            alps_dev_id = atoi(token);
            if (alps_dev_id == device_id) {
                break;
            }
            p_ptr = NULL;
        }
        CmiAssert(alps_dev_id != -1);
        p_ptr = getenv("PMI_GNI_LOC_ADDR");
        CmiAssert(p_ptr != NULL);
        i = 0;
        while ((token = strtok(p_ptr,":")) != NULL) {
            if (i == alps_dev_id) {
                alps_address = atoi(token);
                break;
            }
            p_ptr = NULL;
            ++i;
        }
        CmiAssert(alps_address != -1);
        address = alps_address;
    }
    return address;
}
static void* gather_nic_addresses(void)
{
    unsigned int local_addr,*alladdrs;
    int size,rc;
    size_t addr_len;

    MPI_Comm_size(MPI_COMM_WORLD,&size);

    /*
     * * just assume a single gemini device
     */
    local_addr = get_gni_nic_address(0);

    addr_len = sizeof(unsigned int);

    alladdrs = (unsigned int *)malloc(addr_len * size);
    CmiAssert(alladdrs != NULL);

    MPI_Allgather(&local_addr, addr_len, MPI_BYTE, alladdrs, addr_len, MPI_BYTE, MPI_COMM_WORLD);

    return (void *)alladdrs;

}

static uint8_t get_ptag(void)
{
    char *p_ptr, *token;
    uint8_t ptag;

    p_ptr = getenv("PMI_GNI_PTAG");
    CmiAssert(p_ptr != NULL);
    token = strtok(p_ptr, ":");
    ptag = (uint8_t)atoi(token);
    return ptag;
        
}

static uint32_t get_cookie(void)
{
    uint32_t cookie;
    char *p_ptr, *token;

    p_ptr = getenv("PMI_GNI_COOKIE");
    CmiAssert(p_ptr != NULL);
    token = strtok(p_ptr, ":");
    cookie = (uint32_t)atoi(token);

    return cookie;
}

/*
 * Local side event handler
 *
 */

void LocalEventHandle(gni_cq_entry_t *cq_entry, void *userdata)
{

    int type;

    type = GNI_CQ_GET_TYPE(*cq_entry);

    if(type == GNI_CQ_EVENT_TYPE_SMSG)
    {

    }
}

void RemoteSmsgEventHandle(gni_cq_entry_t *cq_entry, void *userdata)
{
}

void RemoteBteEventHandle(gni_cq_entry_t *cq_entry, void *userdata)
{
}
/* =====Beginning of Definitions of Message-Corruption Related Macros=====*/
/* TODO: add any that are related */
/* =====End of Definitions of Message-Corruption Related Macros=====*/


#include "machine-common.h"
#include "machine-common.c"

/* Network progress function is used to poll the network when for
   messages. This flushes receive buffers on some  implementations*/
#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl() {
}
#endif


/* 
 * The message can be copied to registered memory buffer and be sent
 * This message memory can be registered to network. It depends on which one is cheaper
 *
 * register might be better when the msg is large
 */

static int send_with_fma(int destNode, int size, char *msg)
{
    gni_post_descriptor_t   pd;
    gni_return_t status;
    CONTROL_MSG *control_msg_tmp;
    MSG_LIST *msg_tmp;
    if(buffered_fma_head != 0)
    {
        msg_tmp = (MSG_LIST *)LrtsAlloc(sizeof(MSG_LIST));
        msg_tmp->msg = msg;
        msg_tmp->destNode = destNode;
        msg_tmp ->size = size;
        buffered_smsg_tail->next = msg_tmp;
        buffered_smsg_tail = msg_tmp;
        return 0;
    } else
    {
        pd.type            = GNI_POST_FMA_PUT;
        pd.cq_mode         = GNI_CQMODE_GLOBAL_EVENT | GNI_CQMODE_REMOTE_EVENT;
        pd.dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
        pd.length          = 8;
        pd.remote_addr     = fma_buffer_mdh_addr_base[destNode].addr + fma_buffer_len_eachcore*destNode;
        pd.remote_mem_hndl = fma_buffer_mdh_addr_base[destNode].mdh;
        if(size < FMA_BUFFER_SIZE)
        {
            /* send the message */
            pd.local_addr      = (uint64_t) msg;
        }else
        {
            /* construct a control message and send */
            control_msg_tmp = (CONTROL_MSG *)LrtsAlloc((int)sizeof(CONTROL_MSG));
            GNI_MemRegister(nic_hndl, (uint64_t)msg, 
                size, remote_bte_cq_hndl,
                GNI_MEM_READ_ONLY | GNI_MEM_USE_GART,
                -1, &(control_msg_tmp->source_mem_hndl));
            
            control_msg_tmp->source = _Cmi_mynode;
            control_msg_tmp->source_addr = (uint64_t)msg;
            msg_tmp = (MSG_LIST *)LrtsAlloc(sizeof(MSG_LIST));
            msg_tmp->msg = control_msg_tmp;
            msg_tmp->destNode = destNode;
            msg_tmp ->size = size;
        }
        status = GNI_PostFma(ep_hndl_array[destNode], &pd);
        if(status == GNI_RC_SUCCESS)
            return 1;
        else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE) 
        {
            /* store into buffer fma_list and send later */
            buffered_fma_head = msg_tmp;
            buffered_fma_tail = msg_tmp;
            return 0;
        }
    }
}
static int send_with_smsg(int destNode, int size, char *msg)
{
    gni_return_t status;
    MSG_LIST *msg_tmp;
    CONTROL_MSG *control_msg_tmp;

    /* No mailbox available */
    if(buffered_smsg_head != 0)
    {
        msg_tmp = (MSG_LIST *)LrtsAlloc(sizeof(MSG_LIST));
        msg_tmp->msg = msg;
        msg_tmp->destNode = destNode;
        msg_tmp ->size = size;
        buffered_smsg_tail->next = msg_tmp;
        buffered_smsg_tail = msg_tmp;
        return 0;
    }else
    {
        /* Can use SMSGSend */
        if(size < SMSG_PER_MSG)
        {
            /* send the msg itself */
            msg_tmp = (MSG_LIST *)LrtsAlloc(sizeof(MSG_LIST));
            msg_tmp->msg = msg;
            msg_tmp->destNode = destNode;
            msg_tmp ->size = size;
            status = GNI_SmsgSendWTag(ep_hndl_array[destNode], &(msg_tmp->size), (uint32_t)sizeof(int), msg, (uint32_t)size, NULL, DATA_TAG);
        }else
        {
            /* construct a control message and send */
            control_msg_tmp = (CONTROL_MSG *)LrtsAlloc(sizeof(CONTROL_MSG));
            
            GNI_MemRegister(nic_hndl, (uint64_t)msg, 
                size, remote_bte_cq_hndl,
                GNI_MEM_READ_ONLY | GNI_MEM_USE_GART,
                -1, &(control_msg_tmp->source_mem_hndl));
            
            control_msg_tmp->source = _Cmi_mynode;
            control_msg_tmp->source_addr = (uint64_t)msg;
        
            msg_tmp = (MSG_LIST *)LrtsAlloc(sizeof(MSG_LIST));
            msg_tmp->msg = control_msg_tmp;
            msg_tmp->destNode = destNode;
            msg_tmp ->size = size;
            
            status = GNI_SmsgSendWTag(ep_hndl_array[destNode], 0, 0, control_msg_tmp, sizeof(CONTROL_MSG), NULL, LMSG_INIT_TAG);
        }
        if(status == GNI_RC_SUCCESS)
        {
            return 1;
        }else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE) 
        {
            /* store into buffer smsg_list and send later */
            buffered_smsg_head = msg_tmp;
            buffered_smsg_tail = msg_tmp;
            return 0;
        }
    }
}

static CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode)
{
    PRINT_INFO("Calling LrtsSend")
    if(useSMSG)
    {
        send_with_smsg(destNode, size, msg); 
    }else
    {
        send_with_fma(destNode, size, msg); 
    }
    return 0;
}

static void LrtsPreCommonInit(int everReturn){}
static void LrtsPostCommonInit(int everReturn){}
static void LrtsDrainResources() /* used when exit */
{}

void LrtsPostNonLocal(){}
/* pooling CQ to receive network message */
static void PumpMsgs()
{
    void *header;
    uint8_t             tag_data;
    uint8_t             tag_control;
    gni_return_t status;
    uint64_t source_data, source_control, type, inst_id, data;
    gni_cq_entry_t event_data;
    int msg_nbytes;
    void *msg_data;
    CONTROL_MSG *request_msg;
    gni_post_descriptor_t pd;

    status = GNI_CqGetEvent(rx_cqh, &event_data);

    if(status == GNI_RC_SUCCESS)
    {
        type = GNI_CQ_GET_TYPE(event_data);
        inst_id = GNI_CQ_GET_INST_ID(event_data);
    
    }else
        return;
    /* Check local queue about RDMA_GET */
    if((status = GNI_SmsgGetNextWTag(ep_hndl_array[inst_id], &header, &tag_data)) == GNI_RC_SUCCESS)
    {
        /* copy msg out and then put into queue */
        memcpy(&msg_nbytes, header, sizeof(int));   
        msg_data = CmiAlloc(msg_nbytes);
        handleOneRecvedMsg(msg_nbytes, msg_data);
        GNI_SmsgRelease(ep_hndl_array[inst_id]);
    } else if ((status = GNI_SmsgGetNextWTag(ep_hndl_array[inst_id], &header, &tag_control)) == GNI_RC_SUCCESS)
    {
        /* initial a get to transfer data from the sender side */
        request_msg = (CONTROL_MSG *) header;
        msg_data = CmiAlloc(request_msg->length); //need align checking
        /* register this memory */
        if(request_msg->length <= FMA_BTE_THRESHOLD) 
            pd.type            = GNI_POST_FMA_GET;
        else
            pd.type            = GNI_POST_RDMA_GET;

        pd.cq_mode         = GNI_CQMODE_GLOBAL_EVENT |            GNI_CQMODE_REMOTE_EVENT;
        pd.dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
        pd.length          = request_msg->length;
        pd.local_addr      = (uint64_t) request_msg;
        pd.remote_addr     = request_msg->source_addr;
        pd.remote_mem_hndl = request_msg->source_mem_hndl;
     // rdma specific
        pd.src_cq_hndl     = tx_cqh;
        pd.rdma_mode       = 0;

        if(pd.type == GNI_POST_RDMA_PUT) 
            status = GNI_PostRdma(ep_hndl_array[request_msg->source], &pd);
        else
            status = GNI_PostFma(ep_hndl_array[request_msg->source], &pd);

        if(status = GNI_RC_SUCCESS)
        {
            /* put into receive buffer queue */
        }else
        {
        }

        GNI_SmsgRelease(ep_hndl_array[inst_id]);
    }

}

/* Check whether message send or get is confirmed by remote */
static void ReleaseSentMessages()
{
    gni_cq_entry_t ev;
    gni_return_t status;
    uint64_t type, source, inst_id, data_addr;
    gni_post_descriptor_t *tmp_pd;
    MSG_LIST  *ptr;
    status = GNI_CqGetEvent(tx_cqh, &ev);
    if(status == GNI_RC_SUCCESS)
    {
        type        = GNI_CQ_GET_TYPE(ev);
        inst_id     = GNI_CQ_GET_INST_ID(ev);
        data_addr   = GNI_CQ_GET_DATA(ev);
    }else
        return;

    if(type == GNI_CQ_EVENT_TYPE_POST)
    {
        status = GNI_GetCompleted(tx_cqh, ev, &tmp_pd);
    }
    /* memory leak here , need to realease struct MSG_list */ 
    CmiFree(data_addr);
}

static void SendBufferMsg()
{
    int ret;
    while(buffered_smsg_head != 0)
    {
        if(useSMSG)
        {
            ret = send_with_smsg(buffered_smsg_head->destNode, buffered_smsg_head->size, buffered_smsg_head->msg); 
        }else
        {
            ret = send_with_fma(buffered_smsg_head->destNode, buffered_smsg_head->size, buffered_smsg_head->msg); 
        }
        if(ret == GNI_RC_SUCCESS) 
        {
            buffered_smsg_head = buffered_smsg_head->next;
        }else
            break;
    }
}
static void LrtsAdvanceCommunication()
{
    /*  Receive Msg first */

    //CmiPrintf("Calling Lrts Pump Msg PE:%d\n", CmiMyPe());
    PumpMsgs();
    /* Release Sent Msg */
    //CmiPrintf("Calling Lrts Rlease Msg PE:%d\n", CmiMyPe());
    ReleaseSentMessages();
    //CmiPrintf("Calling Lrts Send Buffmsg PE:%d\n", CmiMyPe());
    /* Send buffered Message */
    SendBufferMsg();
}

void remoteEventHandle(gni_cq_entry_t *event_data, void *context)
{
    gni_return_t status, source_data, source_control;
    uint64_t            source;
    void                *header;
    uint8_t             tag_data;
    uint8_t             tag_control;

    tag_data = DATA_TAG;
    tag_control = LMSG_INIT_TAG;
    /* pool the CQ to check which smsg endpoint to get the data */
    //status = GNI_CqGetEvent(remote_cq_hndl, &event_data);
        
    /* check whether it is data or control information */
    source = GNI_CQ_GET_SOURCE(*event_data);

    if((status = GNI_SmsgGetNextWTag(ep_hndl_array[source], &header, &tag_data)) == GNI_RC_SUCCESS)
    {
        /* copy msg out and then put into queue */

    } else if ((status = GNI_SmsgGetNextWTag(ep_hndl_array[source], &header, &tag_control)) == GNI_RC_SUCCESS)
    {
        /* initial a get to transfer data from the sender side */
    } else
    {

    }

}

/* CDM initialization, register cache, two CQs created, DMA buffer, free list of transaction management structure.  
 * SMSG - all endpoint pairs created, buffer allocated, connection built (static)
 * MSQ */
static void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID)
{
    register int          i;
    int                   rc;
    int                   device_id = 0;
    int                   events_returned;
    int                   test_id;
    unsigned int          local_addr;
    unsigned int          remote_addr;
    gni_cdm_handle_t      cdm_hndl;
    gni_nic_handle_t      nic_hndl;
    gni_return_t          status = GNI_RC_SUCCESS;
    uint32_t              vmdh_index = -1;
    uint64_t              *recv_buffer;
    uint8_t                  ptag;
    int first_spawned;
    int                      size, rank;

    void (*local_event_handler)(gni_cq_entry_t *, void *) = &LocalEventHandle;
    void (*remote_smsg_event_handler)(gni_cq_entry_t *, void *) = &RemoteSmsgEventHandle;
    void (*remote_bte_event_handler)(gni_cq_entry_t *, void *) = &RemoteBteEventHandle;
    
    //useDynamicSmsg = CmiGetArgFlag(argv, "+useDynamicSmsg");
    //useMsgq = CmiGetArgFlag(argv, "+useMsgq");


    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD,myNodeID);
    MPI_Comm_size(MPI_COMM_WORLD,numNodes);


    size = *numNodes;
    rank = *myNodeID;
    
    ptag = get_ptag();
    cookie = get_cookie();
    
    /* Create and attach to the communication  domain */

    status = GNI_CdmCreate(rank, ptag, cookie, modes, &cdm_hndl);
    GNI_RC_CHECK("GNI_CdmCreate", status);
    /* device id The device id is the minor number for the device that is assigned to the device by the system when the device is created. To determine the device number, look in the /dev directory, which contains a list of devices. For a NIC, the device is listed as kgniX, where X is the device number
    0 default */
    status = GNI_CdmAttach(cdm_hndl, device_id, &local_addr, &nic_hndl);
    GNI_RC_CHECK("GNI_CdmAttach", status);
   
    /* create the local completion queue */
    /* adaptive control TODO more option */
    /* the third parameter : The number of events the NIC allows before generating an interrupt. Setting this parameter to zero results in interrupt delivery with every event. When using this parameter, the mode parameter must be set to GNI_CQ_BLOCKING*/
    status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &tx_cqh);
    //status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, &local_event_handler, NULL, &tx_cqh);
    GNI_RC_CHECK("GNI_CqCreate (tx)", status);
    
    /* create the destination completion queue for receiving micro-messages, make this queue considerably larger than the number of transfers */

    status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &rx_cqh);
    status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &remote_bte_cq_hndl);

    GNI_RC_CHECK("Create CQ (rx)", status);

    /* create the endpoints. they need to be bound to allow later CQWrites to them */
    ep_hndl_array = (gni_ep_handle_t*)malloc(size * sizeof(gni_ep_handle_t));
    CmiAssert(ep_hndl_array != NULL);

    MPID_UGNI_AllAddr = (unsigned int *)gather_nic_addresses();
    /* include self */
    for (i=0; i<size; i++) {
        if(i == rank) continue;
        status = GNI_EpCreate(nic_hndl, tx_cqh, &ep_hndl_array[i]);
        GNI_RC_CHECK("GNI_EpCreate ", status);   
        remote_addr = MPID_UGNI_AllAddr[i];
        status = GNI_EpBind(ep_hndl_array[i], remote_addr, i);
        
        GNI_RC_CHECK("GNI_EpBind ", status);   
    }
    /* Allocate a dma buffer to hook the destination cq to */
    
    fma_buffer = (uint64_t *)calloc(FMA_PER_CORE*size, 1);
    CmiAssert(fma_buffer != NULL);

    status = GNI_MemRegister(nic_hndl, (uint64_t)fma_buffer,
        FMA_PER_CORE*size, rx_cqh, 
        GNI_MEM_READWRITE | GNI_MEM_USE_GART, vmdh_index,
        &fma_buffer_mdh_addr);

    GNI_RC_CHECK("Memregister DMA ", status);
    /* Gather up all of the mdh's over the socket network, 
     * this also serves as a barrier */
    fma_buffer_mdh_addr_base = (mdh_addr_t*)malloc(size* sizeof(mdh_addr_t));
    CmiAssert(fma_buffer_mdh_addr_base);

    my_mdh_addr.addr = (uint64_t)recv_buffer;
    my_mdh_addr.mdh = fma_buffer_mdh_addr;

    MPI_Allgather(&my_mdh_addr, sizeof(mdh_addr_t), MPI_BYTE, fma_buffer_mdh_addr_base, sizeof(mdh_addr_t), MPI_BYTE, MPI_COMM_WORLD);
   
    /*  If use SMSG mode (1) register memory buffer as mailbox (2)gather the buffer on remote peers */
    if(useMsgq==0 && useSMSG==1 ){
        PRINT_INFO("Using static SMSG")
        smsg_memlen = SMSG_BUFFER_SIZE ;
        smsg_mem_buffer = (char*)calloc(smsg_memlen, 1);
        status = GNI_MemRegister(nic_hndl, (uint64_t)smsg_mem_buffer,
                                smsg_memlen, rx_cqh,
                                GNI_MEM_READWRITE | GNI_MEM_USE_GART,   
                                vmdh_index,
                                &my_smsg_mdh_mailbox);
       
        GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);

        smsg_attr = (gni_smsg_attr_t *)malloc(size*sizeof(gni_smsg_attr_t));

        if(!useDynamicSmsg){
            for(i=0; i<size; i++)
            {
                smsg_attr[i].msg_type = GNI_SMSG_TYPE_MBOX;
                smsg_attr[i].msg_buffer = smsg_mem_buffer;
                smsg_attr[i].buff_size = smsg_memlen;
                smsg_attr[i].mem_hndl = my_smsg_mdh_mailbox;
                smsg_attr[i].mbox_offset = smsg_memlen/size*rank;
                smsg_attr[i].mbox_maxcredit = SMSG_MAX_CREDIT;
                smsg_attr[i].msg_maxsize = SMSG_PER_MSG;
            }
            smsg_attr_vec = (gni_smsg_attr_t*)malloc(size * sizeof(gni_smsg_attr_t));
            CmiAssert(smsg_attr_vec);
            MPI_Alltoall(smsg_attr, sizeof(gni_smsg_attr_t), MPI_BYTE, smsg_attr_vec, sizeof(gni_smsg_attr_t), MPI_BYTE, MPI_COMM_WORLD);
            for(i=0; i<size; i++)
            {
                if (rank == i) continue;
                /* initialize the smsg channel */
                status = GNI_SmsgInit(ep_hndl_array[i], &smsg_attr[i], 
                    &smsg_attr_vec[i]);

                GNI_RC_CHECK("SMSG Init", status);
            } //end initialization
        } //end static 
        /* do nothing if dynamic connection */
    } //end smsg
    else {

        /* MSGQ is to send and receive short messages for large jobs (exceeding 200,000 ranks). The          performance scales by the node count rather than rank count */
        msgq_attrs.max_msg_sz = MSGQ_MAXSIZE;
        msgq_attrs.smsg_q_sz = 1;
        msgq_attrs.rcv_pool_sz = 1;
        msgq_attrs.num_msgq_eps = 2;
        msgq_attrs.nloc_insts = 8;
        msgq_attrs.modes = 0;
        msgq_attrs.rcv_cq_sz = REMOTE_QUEUE_ENTRIES ;

        status = GNI_MsgqInit(nic_hndl, NULL, NULL, NULL, &msgq_attrs, &msgq_handle);
        GNI_RC_CHECK("MSGQ Init", status);
    } //end MSGQ

    PRINT_INFO("Done with LrtsInit")
}

static void* LrtsAlloc(int n_bytes)
{

    if(n_bytes <= SMSG_PER_MSG)
    {
        return malloc(n_bytes);
    }else if(n_bytes <= FMA_BTE_THRESHOLD)
    {
        return malloc(n_bytes);
    }else 
    {
        return memalign(64, n_bytes);
    }
}

static void  LrtsFree(void *msg)
{
    free(msg);
}


static void LrtsExit()
{
    /* free memory ? */
    MPI_Finalize();
}
/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message) {

    MPI_Abort(MPI_COMM_WORLD, -1);
}

/**************************  TIMER FUNCTIONS **************************/

/************Barrier Related Functions****************/

int CmiBarrier()
{
    int status;
    status = MPI_Barrier(MPI_COMM_WORLD);
    return status;

}
/*@}*/

/*******************************************
 *
 * internal function only to this file 
 */
