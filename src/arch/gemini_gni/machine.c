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
#include <stdlib.h>
#include <errno.h>
#include <malloc.h>
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

//#define  USE_ONESIDED 1

#ifdef USE_ONESIDED
#include "onesided.h"
#endif
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
#define PRINT_INFO(msg) {fprintf(stdout, "[%d] %s\n", CmiMyPe(), msg); fflush(stdout);}
#else
#define PRINT_INFO(msg)
#endif
/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* If SMSG is not used */
#define FMA_PER_CORE  1024
#define FMA_BUFFER_SIZE 1024
/* If SMSG is used */
#define SMSG_PER_MSG    1024
#define SMSG_MAX_CREDIT 16
#define SMSG_BUFFER_SIZE        1024000

#define MSGQ_MAXSIZE       4096
/* large message transfer with FMA or BTE */
#define LRTS_GNI_RDMA_THRESHOLD  16384

#define REMOTE_QUEUE_ENTRIES  1048576
#define LOCAL_QUEUE_ENTRIES 1024
/* SMSG is data message */
#define DATA_TAG        0x38
/* SMSG is a control message to initialize a BTE */
#define LMSG_INIT_TAG        0x39 
#define ACK_TAG         0x37

#define DEBUG
#ifdef GNI_RC_CHECK
#undef GNI_RC_CHECK
#endif
#ifdef DEBUG
#define GNI_RC_CHECK(msg,rc) do { if(rc != GNI_RC_SUCCESS) {           printf("%s; err=%s\n",msg,gni_err_str[rc]); exit(911); } } while(0)
#else
#define GNI_RC_CHECK(msg,rc)
#endif

#ifdef USE_ONESIDED
onesided_hnd_t   onesided_hnd;
onesided_md_t    omdh;
#endif

static int useStaticSMSG   = 1;
static int useStaticMSGQ = 0;
static int useStaticFMA = 0;
static int mysize, myrank;
static gni_nic_handle_t      nic_hndl;

gni_msgq_attr_t         msgq_attrs;
gni_msgq_handle_t       msgq_handle;
gni_msgq_ep_attr_t      msgq_ep_attrs;
gni_msgq_ep_attr_t      msgq_ep_attrs_size;
/* =====Beginning of Declarations of Machine Specific Variables===== */
static int cookie;
static int modes = 0;
static gni_cq_handle_t       rx_cqh = NULL;
static gni_cq_handle_t       remote_bte_cq_hndl = NULL;
static gni_cq_handle_t       tx_cqh = NULL;
static gni_ep_handle_t       *ep_hndl_array;

/* preallocated memory buffer for FMA for short message and control message */
static int              fma_buffer_len_eachcore = FMA_BUFFER_SIZE;
typedef struct {
    gni_mem_handle_t mdh;
    uint64_t addr;
} mdh_addr_t;

static mdh_addr_t            *fma_buffer_mdh_addr_base;
typedef struct msg_list
{
    uint32_t destNode;
    uint32_t size;
    void *msg;
    struct msg_list *next;
    uint8_t tag;
}MSG_LIST;

typedef struct control_msg
{
    uint64_t            source_addr;
    int                 source;               /* source rank */
    int                 length;
    gni_mem_handle_t    source_mem_hndl;
    struct control_msg *next;
}CONTROL_MSG;

/* reuse PendingMsg memory */
static CONTROL_MSG *control_freelist=NULL;

#define FreeControlMsg(d)       \
  d->next = control_freelist;\
  control_freelist = d;\

#define MallocControlMsg(d) \
  d = control_freelist;\
  if (d==0) {d = ((CONTROL_MSG*)malloc(sizeof(CONTROL_MSG)));\
             _MEMCHECK(d);\
  } else control_freelist = d->next;


/* LrtsSent is called but message can not be sent by SMSGSend because of mailbox full or no credit */
static MSG_LIST *buffered_smsg_head= 0;
static MSG_LIST *buffered_smsg_tail= 0;
/* SmsgSend return success but message sent is not confirmed by remote side */

static MSG_LIST *buffered_fma_head = 0;
static MSG_LIST *buffered_fma_tail = 0;

/* functions  */


static void* LrtsAllocRegister(int n_bytes, gni_mem_handle_t* mem_hndl);

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


#include "machine-lrts.h"
#include "machine-common-core.c"

/* Network progress function is used to poll the network when for
   messages. This flushes receive buffers on some  implementations*/
#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl() {
}
#endif


/* 
 * The message can be copied to registered memory buffer and be sent
 * This message memory can be registered to network. It depends on which one is cheaper
 * register might be better when the msg is large
 */

static int send_with_fma(int destNode, int size, char *msg)
{
    gni_post_descriptor_t   pd;
    gni_return_t status;
    CONTROL_MSG *control_msg_tmp;
    MSG_LIST *msg_tmp;
    uint32_t              vmdh_index = -1;
    if(buffered_fma_head != 0)
    {
        msg_tmp = (MSG_LIST *)malloc(sizeof(MSG_LIST));
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
            control_msg_tmp = (CONTROL_MSG *)malloc((int)sizeof(CONTROL_MSG));
            GNI_MemRegister(nic_hndl, (uint64_t)msg, 
                size, remote_bte_cq_hndl,
                GNI_MEM_READ_ONLY | GNI_MEM_USE_GART,
                vmdh_index, &(control_msg_tmp->source_mem_hndl));
            
            control_msg_tmp->source = _Cmi_mynode;
            control_msg_tmp->source_addr = (uint64_t)msg;
            control_msg_tmp->length = size;
        }
        status = GNI_PostFma(ep_hndl_array[destNode], &pd);
        if(status == GNI_RC_SUCCESS)
            return 1;
        else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE) 
        {
            msg_tmp = (MSG_LIST *)malloc(sizeof(MSG_LIST));
            msg_tmp->msg = control_msg_tmp;
            msg_tmp->destNode = destNode;
            msg_tmp ->size = size;
            /* store into buffer fma_list and send later */
            buffered_fma_head = msg_tmp;
            buffered_fma_tail = msg_tmp;
            return 0;
        }
    }
}

static int send_with_smsg(int destNode, int size, char *msg)
{
    gni_return_t        status;
    MSG_LIST            *msg_tmp;
    CONTROL_MSG         *control_msg_tmp;
    uint8_t             tag_data = DATA_TAG;
    uint8_t             tag_control= LMSG_INIT_TAG ;
    uint32_t            vmdh_index = -1;

    /* No mailbox available, buffer this msg and its info */
    if(buffered_smsg_head != 0)
    {
        msg_tmp = (MSG_LIST *)malloc(sizeof(MSG_LIST));
        msg_tmp->destNode = destNode;
        if(size <=SMSG_PER_MSG)
        {
            msg_tmp->msg    = msg;
            msg_tmp->size   = size;
            msg_tmp->tag    = tag_data;
        }else
        {
            MallocControlMsg(control_msg_tmp);

            status = GNI_MemRegister(nic_hndl, (uint64_t)msg, 
                size, rx_cqh,
                GNI_MEM_READ_ONLY | GNI_MEM_USE_GART,
                vmdh_index, &(control_msg_tmp->source_mem_hndl));
            
            GNI_RC_CHECK("MemRegister fails at ", status);
            control_msg_tmp->source         = myrank;
            control_msg_tmp->source_addr    = (uint64_t)msg;
            control_msg_tmp->length         =size; 
            msg_tmp->msg                    = control_msg_tmp;
            msg_tmp->size                   = sizeof(CONTROL_MSG);
            msg_tmp->tag                    = tag_control;
        }
        buffered_smsg_tail->next    = msg_tmp;
        buffered_smsg_tail          = msg_tmp;
        return 0;
    }else
    {
        /* Can use SMSGSend */
        if(size <= SMSG_PER_MSG)
        {
            /* send the msg itself */
            status = GNI_SmsgSendWTag(ep_hndl_array[destNode], &size, sizeof(int), msg, size, 0, tag_data);
            if (status == GNI_RC_SUCCESS)
            {
                CmiFree(msg);
                return 1;
            }
            else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE)
            {
                msg_tmp             = (MSG_LIST *)malloc(sizeof(MSG_LIST));
                msg_tmp->msg        = msg;
                msg_tmp->destNode   = destNode;
                msg_tmp->size       = size;
                msg_tmp->tag        = tag_data;
                /* store into buffer smsg_list and send later */
                buffered_smsg_head  = msg_tmp;
                buffered_smsg_tail  = msg_tmp;
                return 0;
            }
            else
                GNI_RC_CHECK("GNI_SmsgSendWTag", status);
        }else
        {
            /* construct a control message and send */
            //control_msg_tmp = (CONTROL_MSG *)malloc(sizeof(CONTROL_MSG));
            MallocControlMsg(control_msg_tmp);
            
            status = GNI_MemRegister(nic_hndl, (uint64_t)msg, 
                size, rx_cqh,
                GNI_MEM_READ_ONLY | GNI_MEM_USE_GART,
                vmdh_index, &(control_msg_tmp->source_mem_hndl));
            
            GNI_RC_CHECK("MemRegister fails at ", status);
            control_msg_tmp->source         = myrank;
            control_msg_tmp->source_addr    = (uint64_t)msg;
            control_msg_tmp->length         = size; 
            status = GNI_SmsgSendWTag(ep_hndl_array[destNode], 0, 0, control_msg_tmp, sizeof(CONTROL_MSG), 0, tag_control);
            if(status == GNI_RC_SUCCESS)
            {
                FreeControlMsg(control_msg_tmp);
                //CmiPrintf("Control message sent bytes:%d on PE:%d, source=%d, add=%p, mem_hndl=%ld, %ld\n", sizeof(CONTROL_MSG), myrank, control_msg_tmp->source, (void*)control_msg_tmp->source_addr, (control_msg_tmp->source_mem_hndl).qword1, (control_msg_tmp->source_mem_hndl).qword2);
                return 1;
            }else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE) 
            {
                msg_tmp             = (MSG_LIST *)malloc(sizeof(MSG_LIST));
                msg_tmp->msg        = control_msg_tmp;
                msg_tmp->destNode   = destNode;
                msg_tmp ->size      = sizeof(CONTROL_MSG);
                msg_tmp->tag        = tag_control;
                /* store into buffer smsg_list and send later */
                buffered_smsg_head = msg_tmp;
                buffered_smsg_tail = msg_tmp;
                return 0;
            }
            else
                GNI_RC_CHECK("GNI_SmsgSendWTag", status);
        }
    }
}

static CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode)
{
    //PRINT_INFO("Calling LrtsSend")
    if(useStaticSMSG)
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
static void PumpNetworkMsgs()
{
    void *header;
    uint8_t             msg_tag, data_tag, control_tag, ack_tag;
    gni_return_t status;
    uint64_t inst_id;
    gni_cq_entry_t event_data;
    int msg_nbytes;
    void *msg_data;
    gni_mem_handle_t msg_mem_hndl;
    CONTROL_MSG *request_msg;
    gni_post_descriptor_t *pd;

    data_tag        = DATA_TAG;
    control_tag     = LMSG_INIT_TAG;
    ack_tag         = ACK_TAG;
    status = GNI_CqGetEvent(rx_cqh, &event_data);

    if(status == GNI_RC_SUCCESS)
    {
        inst_id = GNI_CQ_GET_INST_ID(event_data);
    }else
        return;
  
    msg_tag = GNI_SMSG_ANY_TAG;
    status = GNI_SmsgGetNextWTag(ep_hndl_array[inst_id], &header, &msg_tag);

    //CmiPrintf("++++## PumpNetwork Small msg is received on PE:%d message tag=%c\n", myrank, msg_tag);
    if(status  == GNI_RC_SUCCESS)
    {
        /* copy msg out and then put into queue */
        if(msg_tag == data_tag)
        {
            msg_nbytes = *(int*)header;
            msg_data = CmiAlloc(msg_nbytes);
            memcpy(msg_data, (char*)header+sizeof(int), msg_nbytes);
            handleOneRecvedMsg(msg_nbytes, msg_data);
        }else if(msg_tag == control_tag)
        {
            /* initial a get to transfer data from the sender side */
            request_msg = (CONTROL_MSG *) header;
            msg_data = LrtsAllocRegister(request_msg->length, &msg_mem_hndl); //need align checking
            pd = (gni_post_descriptor_t*)malloc(sizeof(gni_post_descriptor_t));
            //bzero(&pd, sizeof(pd));
            if(request_msg->length < LRTS_GNI_RDMA_THRESHOLD) 
                pd->type            = GNI_POST_FMA_GET;
            else
                pd->type            = GNI_POST_RDMA_GET;

            pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT |  GNI_CQMODE_REMOTE_EVENT;
            pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
            pd->length          = request_msg->length;
            pd->local_addr      = (uint64_t) msg_data;
            pd->local_mem_hndl  = msg_mem_hndl; 
            pd->remote_addr     = request_msg->source_addr;
            pd->remote_mem_hndl = request_msg->source_mem_hndl;
            pd->src_cq_hndl     = 0;     /* tx_cqh;  */
            pd->rdma_mode       = 0;

           // CmiPrintf("source=%d, addr=%p, handler=%ld,%ld \n", request_msg->source, (void*)pd.remote_addr, (request_msg->source_mem_hndl).qword1, (request_msg->source_mem_hndl).qword2);
            if(pd->type == GNI_POST_RDMA_GET) 
                status = GNI_PostRdma(ep_hndl_array[request_msg->source], pd);
            else
                status = GNI_PostFma(ep_hndl_array[request_msg->source],  pd);
            GNI_RC_CHECK("post ", status);
           
            //CmiPrintf("post status=%s on PE:%d\n", gni_err_str[status], myrank);
            if(status = GNI_RC_SUCCESS)
            {
                /* put into receive buffer queue */
            }else
            {
            }
        }else if(msg_tag == ack_tag){
            /* Get is done, release message . Now put is not used yet*/
            request_msg = (CONTROL_MSG *) header;
            //CmiPrintf("++++## ACK msg is received on PE:%d message size=%d, addr=%p\n", myrank, request_msg->length, (void*)request_msg->source_addr);
            CmiFree((void*)request_msg->source_addr);
            //CmiPrintf("++++## release ACK msg is received on PE:%d message size=%d\n", myrank, request_msg->length);
        }else{
            CmiPrintf("weird tag problem\n");
            CmiAbort("Unknown tag\n");
        }
        GNI_SmsgRelease(ep_hndl_array[inst_id]);
    }else
    {
        //CmiPrintf("Message not ready\n");
        //
    }

}

/* Check whether message send or get is confirmed by remote */
static void PumpLocalTransactions()
{
    gni_cq_entry_t ev;
    gni_return_t status;
    uint64_t type, inst_id, data_addr;
    uint8_t         ack_tag = ACK_TAG;
    gni_post_descriptor_t *tmp_pd;
    MSG_LIST *msg_tmp;
    //gni_post_descriptor_t   ack_pd;
    MSG_LIST  *ptr;
    CONTROL_MSG *ack_msg_tmp;
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
        GNI_RC_CHECK("Local CQ completed ", status);
        //Message is sent, free message , put is not used now
        if(tmp_pd->type == GNI_POST_RDMA_PUT || tmp_pd->type == GNI_POST_FMA_PUT)
        {
            CmiFree((void *)tmp_pd->local_addr);
        }else if(tmp_pd->type == GNI_POST_RDMA_GET || tmp_pd->type == GNI_POST_FMA_GET)
        {
           /* Send an ACK to remote side */
           /*ack_pd.type = GNI_POST_CQWRITE;
           ack_pd.cq_mode = GNI_CQMODE_GLOBAL_EVENT;
           ack_pd.dlvr_mode = GNI_DLVMODE_NO_ADAPT;
           ack_pd.cqwrite_value = tmp_pd->remote_addr&0x0000ffffffffffff;
           ack_pd.remote_mem_hndl = tmp_pd->remote_mem_hndl;
           status = GNI_PostCqWrite(ep_hndl_array[inst_id], &ack_pd);
           GNI_RC_CHECK("Ack Post by CQWrite ", status);
           */
            //CmiPrintf("\nPE:%d Received large message by get , sizefield=%d, length=%d, addr=%p\n", myrank, SIZEFIELD((void*)tmp_pd->local_addr), tmp_pd->length, tmp_pd->remote_addr); 
           //CmiPrintf("\n+PE:%d Received large message by get , sizefield=%d, length=%d, addr=%p\n", myrank, remote_length , tmp_pd->length, (void*)remote_addr); 
           MallocControlMsg(ack_msg_tmp);
           ack_msg_tmp->source = myrank;
           //CmiPrintf("\n++PE:%d Received large message by get , sizefield=%d, length=%d, addr=%p\n", myrank, SIZEFIELD((void*)tmp_pd->local_addr), tmp_pd->length, tmp_pd->remote_addr); 
           //CmiPrintf("\n+++PE:%d Received large message by get , sizefield=%d, length=%d, addr=%p\n", myrank, remote_length , tmp_pd->length, (void*)remote_addr); 
           ack_msg_tmp->source_addr = tmp_pd->remote_addr;
           ack_msg_tmp->length=tmp_pd->length; 
           ack_msg_tmp->source_mem_hndl = tmp_pd->remote_mem_hndl;
           //CmiPrintf("PE:%d sending ACK back addr=%p \n", myrank, ack_msg_tmp->source_addr); 
           
           if(buffered_smsg_head!=0)
           {
               msg_tmp = (MSG_LIST *)malloc(sizeof(MSG_LIST));
               msg_tmp->msg = ack_msg_tmp;
               msg_tmp ->size = sizeof(CONTROL_MSG);
               msg_tmp->tag = ack_tag;
               msg_tmp->destNode = inst_id;
               buffered_smsg_tail->next = msg_tmp;
               buffered_smsg_tail = msg_tmp;
           }else
           {
               //CmiPrintf("PE:%d sending ACK back addr=%p \n", myrank, ack_msg_tmp->source_addr); 
               status = GNI_SmsgSendWTag(ep_hndl_array[inst_id], 0, 0, ack_msg_tmp, sizeof(CONTROL_MSG), 0, ack_tag);
               if(status == GNI_RC_SUCCESS)
               {
                   FreeControlMsg(ack_msg_tmp);
               }else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE)
               {
                   msg_tmp = (MSG_LIST *)malloc(sizeof(MSG_LIST));
                   msg_tmp->msg = ack_msg_tmp;
                   msg_tmp ->size = sizeof(CONTROL_MSG);
                   msg_tmp->tag = ack_tag;
                   msg_tmp->destNode = inst_id;
                   buffered_smsg_head = msg_tmp;
                   buffered_smsg_tail = msg_tmp;
               }
               else
                   GNI_RC_CHECK("GNI_SmsgSendWTag", status);
           }
           handleOneRecvedMsg(SIZEFIELD((void*)tmp_pd->local_addr), (void*)tmp_pd->local_addr); 
        }
    }
}

static void SendBufferMsg()
{
    MSG_LIST *ptr;
    uint8_t tag_data, tag_control, tag_ack;
    gni_return_t status;

    tag_data = DATA_TAG;
    tag_control = LMSG_INIT_TAG;
    tag_ack = ACK_TAG;
    /* can add flow control here to control the number of messages sent before handle message */
    while(buffered_smsg_head != 0)
    {
        if(useStaticSMSG)
        {
            ptr = buffered_smsg_head;
            if(ptr->tag == tag_data)
            {
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], &(ptr->size), (uint32_t)sizeof(int), ptr->msg, ptr->size, 0, tag_data);
                if(status == GNI_RC_SUCCESS)
                    CmiFree(ptr->msg);
            }else if(ptr->tag ==tag_control)
            {
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], 0, 0, ptr->msg, sizeof(CONTROL_MSG), 0, tag_control);
                if(status == GNI_RC_SUCCESS)
                    FreeControlMsg(((CONTROL_MSG)(ptr->msg)));
            }else if (ptr->tag == tag_ack)
            {
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], 0, 0, ptr->msg, sizeof(CONTROL_MSG), 0, tag_ack);
                if(status == GNI_RC_SUCCESS)
                    FreeControlMsg(ptr->msg);
            }
        } else if(useStaticMSGQ)
        {
            CmiPrintf("MSGQ Send not done\n");
        }else
        {
            CmiPrintf("FMA Send not done\n");
        }
        if(status == GNI_RC_SUCCESS)
        {
            buffered_smsg_head= buffered_smsg_head->next;
            free(ptr);
        }else
            break;
    }
}
static void LrtsAdvanceCommunication()
{
    /*  Receive Msg first */
    //CmiPrintf("Calling Lrts Pump Msg PE:%d\n", CmiMyPe());
    PumpNetworkMsgs();
    /* Release Sent Msg */
    //CmiPrintf("Calling Lrts Rlease Msg PE:%d\n", CmiMyPe());
    PumpLocalTransactions();
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

/* if FMA is used to transfer small messages, static allocation*/
static void _init_smallMsgwithFma(int size)
{
    char            *fma_buffer;
    gni_mem_handle_t      fma_buffer_mdh_addr;
    mdh_addr_t            my_mdh_addr;
    gni_return_t status;
    
    fma_buffer = (char *)calloc(FMA_PER_CORE*size, 1);
    CmiAssert(fma_buffer != NULL);

    status = GNI_MemRegister(nic_hndl, (uint64_t)fma_buffer,
        FMA_PER_CORE*size, rx_cqh, 
        GNI_MEM_READWRITE | GNI_MEM_USE_GART, -1,
        &fma_buffer_mdh_addr);

    GNI_RC_CHECK("Memregister DMA ", status);
    /* Gather up all of the mdh's over the socket network, 
     * * this also serves as a barrier */
    fma_buffer_mdh_addr_base = (mdh_addr_t*)malloc(size* sizeof(mdh_addr_t));
    CmiAssert(fma_buffer_mdh_addr_base);

    my_mdh_addr.addr = (uint64_t)fma_buffer;
    my_mdh_addr.mdh = fma_buffer_mdh_addr;

    MPI_Allgather(&my_mdh_addr, sizeof(mdh_addr_t), MPI_BYTE, fma_buffer_mdh_addr_base, sizeof(mdh_addr_t), MPI_BYTE, MPI_COMM_WORLD);

}

static void _init_static_smsg()
{
    gni_smsg_attr_t      *smsg_attr;
    gni_smsg_attr_t      *smsg_attr_vec;
    char                 *smsg_mem_buffer = NULL;
    uint32_t             smsg_memlen;
    gni_mem_handle_t      my_smsg_mdh_mailbox;
    register    int         i;
    gni_return_t status;
    uint32_t              vmdh_index = -1;

    smsg_memlen = SMSG_BUFFER_SIZE ;
    smsg_mem_buffer = (char*)calloc(smsg_memlen, 1);
    _MEMCHECK(smsg_mem_buffer);
    status = GNI_MemRegister(nic_hndl, (uint64_t)smsg_mem_buffer,
                             smsg_memlen, rx_cqh,
                             GNI_MEM_READWRITE | GNI_MEM_USE_GART,   
                             vmdh_index,
                            &my_smsg_mdh_mailbox);

    GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);

    smsg_attr = (gni_smsg_attr_t *)malloc(mysize*sizeof(gni_smsg_attr_t));
    _MEMCHECK(smsg_attr);

    for(i=0; i<mysize; i++)
    {
        smsg_attr[i].msg_type = GNI_SMSG_TYPE_MBOX;
        smsg_attr[i].msg_buffer = smsg_mem_buffer;
        smsg_attr[i].buff_size = smsg_memlen;
        smsg_attr[i].mem_hndl = my_smsg_mdh_mailbox;
        smsg_attr[i].mbox_offset = smsg_memlen/mysize*i;
        smsg_attr[i].mbox_maxcredit = SMSG_MAX_CREDIT;
        smsg_attr[i].msg_maxsize = SMSG_PER_MSG;
    }
    smsg_attr_vec = (gni_smsg_attr_t*)malloc(mysize * sizeof(gni_smsg_attr_t));
    CmiAssert(smsg_attr_vec);
    MPI_Alltoall(smsg_attr, sizeof(gni_smsg_attr_t), MPI_BYTE, smsg_attr_vec, sizeof(gni_smsg_attr_t), MPI_BYTE, MPI_COMM_WORLD);
    for(i=0; i<mysize; i++)
    {
        if (myrank == i) continue;
        /* initialize the smsg channel */
        status = GNI_SmsgInit(ep_hndl_array[i], &smsg_attr[i], &smsg_attr_vec[i]);
        GNI_RC_CHECK("SMSG Init", status);
    } //end initialization
    free(smsg_attr);
    free(smsg_attr_vec);
} 

static void _init_static_msgq()
{
    gni_return_t status;
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


}
static void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID)
{
    register int          i;
    int                   rc;
    int                   device_id = 0;
    unsigned int          remote_addr;
    gni_cdm_handle_t      cdm_hndl;
    gni_return_t          status = GNI_RC_SUCCESS;
    uint32_t              vmdh_index = -1;
    uint8_t               ptag;
    unsigned int         local_addr, *MPID_UGNI_AllAddr;
    
    void (*local_event_handler)(gni_cq_entry_t *, void *)       = &LocalEventHandle;
    void (*remote_smsg_event_handler)(gni_cq_entry_t *, void *) = &RemoteSmsgEventHandle;
    void (*remote_bte_event_handler)(gni_cq_entry_t *, void *)  = &RemoteBteEventHandle;
    
    //useStaticSMSG = CmiGetArgFlag(*argv, "+useStaticSmsg");
    //useStaticMSGQ = CmiGetArgFlag(*argv, "+useStaticMsgQ");


    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD,myNodeID);
    MPI_Comm_size(MPI_COMM_WORLD,numNodes);


    mysize = *numNodes;
    myrank = *myNodeID;
  
    if(myrank == 0)
    {
        printf("Charm++> Running on Gemini (GNI)\n");
    }
#ifdef USE_ONESIDED
    onesided_init(NULL, &onesided_hnd);

    // this is a GNI test, so use the libonesided bypass functionality
    onesided_gni_bypass_get_nih(onesided_hnd, &nic_hndl);
    local_addr = gniGetNicAddress();
#else
    ptag = get_ptag();
    cookie = get_cookie();
    
    //Create and attach to the communication  domain */
    status = GNI_CdmCreate(myrank, ptag, cookie, modes, &cdm_hndl);
    GNI_RC_CHECK("GNI_CdmCreate", status);
    //* device id The device id is the minor number for the device
    //that is assigned to the device by the system when the device is created.
    //To determine the device number, look in the /dev directory, which contains a list of devices. For a NIC, the device is listed as kgniX
    //where X is the device number 0 default 
    status = GNI_CdmAttach(cdm_hndl, device_id, &local_addr, &nic_hndl);
    GNI_RC_CHECK("GNI_CdmAttach", status);
    local_addr = get_gni_nic_address(0);
#endif
    MPID_UGNI_AllAddr = (unsigned int *)malloc(sizeof(unsigned int) * mysize);
    _MEMCHECK(MPID_UGNI_AllAddr);
    MPI_Allgather(&local_addr, sizeof(unsigned int), MPI_BYTE, MPID_UGNI_AllAddr, sizeof(unsigned int), MPI_BYTE, MPI_COMM_WORLD);
    /* create the local completion queue */
    /* the third parameter : The number of events the NIC allows before generating an interrupt. Setting this parameter to zero results in interrupt delivery with every event. When using this parameter, the mode parameter must be set to GNI_CQ_BLOCKING*/
    status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &tx_cqh);
    //status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, &local_event_handler, NULL, &tx_cqh);
    GNI_RC_CHECK("GNI_CqCreate (tx)", status);
    
    /* create the destination completion queue for receiving micro-messages, make this queue considerably larger than the number of transfers */

    status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &rx_cqh);
    GNI_RC_CHECK("Create CQ (rx)", status);
    
    status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &remote_bte_cq_hndl);
    GNI_RC_CHECK("Create BTE CQ", status);

    /* create the endpoints. they need to be bound to allow later CQWrites to them */
    ep_hndl_array = (gni_ep_handle_t*)malloc(mysize * sizeof(gni_ep_handle_t));
    _MEMCHECK(ep_hndl_array);

    for (i=0; i<mysize; i++) {
        if(i == myrank) continue;
        status = GNI_EpCreate(nic_hndl, tx_cqh, &ep_hndl_array[i]);
        GNI_RC_CHECK("GNI_EpCreate ", status);   
        remote_addr = MPID_UGNI_AllAddr[i];
        status = GNI_EpBind(ep_hndl_array[i], remote_addr, i);
        GNI_RC_CHECK("GNI_EpBind ", status);   
    }
    /* Depending on the number of cores in the job, decide different method */
    /* SMSG is fastest but not scale; Msgq is scalable, FMA is own implementation for small message */
    if(useStaticSMSG == 1)
    {
        _init_static_smsg(mysize);
    }else if(useStaticMSGQ == 1)
    {
        _init_static_msgq();
    }else if( useStaticFMA == 1)
    {
        _init_smallMsgwithFma(mysize);
    }
    free(MPID_UGNI_AllAddr);
    PRINT_INFO("\nDone with LrtsInit")
}

#define ALIGNBUF                64

void* LrtsAlloc(int n_bytes, int header)
{
    if(n_bytes <= SMSG_PER_MSG)
    {
        int totalsize = n_bytes+header;
        return malloc(totalsize);
/*
    }else if(n_bytes <= LRTS_GNI_RDMA_THRESHOLD)
    {
        return malloc(n_bytes);
*/
    }else 
    {
        CmiAssert(header <= ALIGNBUF);
        char *res = memalign(ALIGNBUF, n_bytes+ALIGNBUF);
        return res + ALIGNBUF - header;
    }
}

void  LrtsFree(void *msg)
{
    int size = SIZEFIELD((char*)msg+sizeof(CmiChunkHeader));
    if (size <= SMSG_PER_MSG)
      free(msg);
    else
      free((char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);
/*
    CmiFree(msg);
*/
}

static void* LrtsAllocRegister(int n_bytes, gni_mem_handle_t* mem_hndl)
{
    void *ptr;
    gni_return_t status;
    ptr = CmiAlloc(n_bytes);
    _MEMCHECK(ptr);
    status = GNI_MemRegister(nic_hndl, (uint64_t)ptr,
        n_bytes, rx_cqh, 
        GNI_MEM_READWRITE | GNI_MEM_USE_GART, -1,
        mem_hndl);

    GNI_RC_CHECK("GNI_MemRegister in LrtsAlloc", status);
    return ptr;

}

static void LrtsExit()
{
    /* free memory ? */
    MPI_Finalize();
}

void CmiAbort(const char *message) {

    MPI_Abort(MPI_COMM_WORLD, -1);
}

/**************************  TIMER FUNCTIONS **************************/
#if CMK_TIMER_USE_SPECIAL
/* MPI calls are not threadsafe, even the timer on some machines */
static CmiNodeLock  timerLock = 0;
static int _absoluteTime = 0;
static double starttimer = 0;
static int _is_global = 0;

int CmiTimerIsSynchronized() {
    int  flag;
    void *v;

    /*  check if it using synchronized timer */
    if (MPI_SUCCESS != MPI_Attr_get(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &v, &flag))
        printf("MPI_WTIME_IS_GLOBAL not valid!\n");
    if (flag) {
        _is_global = *(int*)v;
        if (_is_global && CmiMyPe() == 0)
            printf("Charm++> MPI timer is synchronized\n");
    }
    return _is_global;
}

int CmiTimerAbsolute() {
    return _absoluteTime;
}

double CmiStartTimer() {
    return 0.0;
}

double CmiInitTime() {
    return starttimer;
}

void CmiTimerInit(char **argv) {
    _absoluteTime = CmiGetArgFlagDesc(argv,"+useAbsoluteTime", "Use system's absolute time as wallclock time.");
    if (_absoluteTime && CmiMyPe() == 0)
        printf("Charm++> absolute MPI timer is used\n");
    _is_global = CmiTimerIsSynchronized();

    if (_is_global) {
        if (CmiMyRank() == 0) {
            double minTimer;
#if CMK_TIMER_USE_XT3_DCLOCK
            starttimer = dclock();
#else
            starttimer = MPI_Wtime();
#endif

            MPI_Allreduce(&starttimer, &minTimer, 1, MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD );
            starttimer = minTimer;
        }
    } else { /* we don't have a synchronous timer, set our own start time */
        CmiBarrier();
        CmiBarrier();
        CmiBarrier();
#if CMK_TIMER_USE_XT3_DCLOCK
        starttimer = dclock();
#else
        starttimer = MPI_Wtime();
#endif
    }

    CmiNodeAllBarrier();          /* for smp */
}

/**
 * Since the timerLock is never created, and is
 * always NULL, then all the if-condition inside
 * the timer functions could be disabled right
 * now in the case of SMP.
 */
double CmiTimer(void) {
    double t;
#if CMK_TIMER_USE_XT3_DCLOCK
    t = dclock();
#else
    t = MPI_Wtime();
#endif
    return _absoluteTime?t: (t-starttimer);
}

double CmiWallTimer(void) {
    double t;
#if CMK_TIMER_USE_XT3_DCLOCK
    t = dclock();
#else
    t = MPI_Wtime();
#endif
    return _absoluteTime? t: (t-starttimer);
}

double CmiCpuTimer(void) {
    double t;
#if CMK_TIMER_USE_XT3_DCLOCK
    t = dclock() - starttimer;
#else
    t = MPI_Wtime() - starttimer;
#endif
    return t;
}

#endif

/************Barrier Related Functions****************/

int CmiBarrier()
{
    int status;
    status = MPI_Barrier(MPI_COMM_WORLD);
    return status;

}
