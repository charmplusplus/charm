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

#define PRINT_SYH  0

#if PRINT_SYH
int         lrts_send_request = 0;
int         lrts_received_msg = 0;
int         lrts_local_done_msg = 0;
#endif

#include "machine.h"

#include "pcqueue.h"
//#define  USE_ONESIDED 1
#ifdef USE_ONESIDED
//onesided implementation is wrong, since no place to restore omdh
#include "onesided.h"
onesided_hnd_t   onesided_hnd;
onesided_md_t    omdh;
#define MEMORY_REGISTER(handler, nic_hndl, msg, size, mem_hndl, myomdh)  omdh. onesided_mem_register(handler, (uint64_t)msg, size, 0, myomdh) 

#define MEMORY_DEREGISTER(handler, nic_hndl, mem_hndl, myomdh) onesided_mem_deregister(handler, myomdh)

#else
uint8_t   onesided_hnd, omdh;
#define  MEMORY_REGISTER(handler, nic_hndl, msg, size, mem_hndl, myomdh) GNI_MemRegister(nic_hndl, (uint64_t)msg,  (uint64_t)size, NULL,  GNI_MEM_READWRITE|GNI_MEM_USE_GART, -1, mem_hndl)

#define  MEMORY_DEREGISTER(handler, nic_hndl, mem_hndl, myomdh)  GNI_MemDeregister(nic_hndl, (mem_hndl))
#endif

#define CmiGetMsgSize(m)  ((CmiMsgHeaderExt*)m)->size
#define CmiSetMsgSize(m,s)  do {((((CmiMsgHeaderExt*)m)->size)=(s));} while(0)

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* If SMSG is not used */
#define FMA_PER_CORE  1024
#define FMA_BUFFER_SIZE 1024
/* If SMSG is used */
#define SMSG_MAX_MSG     1024
#define SMSG_MAX_CREDIT  128

#define MSGQ_MAXSIZE       4096
/* large message transfer with FMA or BTE */
#define LRTS_GNI_RDMA_THRESHOLD  16384

#define REMOTE_QUEUE_ENTRIES  1048576
#define LOCAL_QUEUE_ENTRIES   10240
/* SMSG is data message */
#define DATA_TAG          0x38
/* SMSG is a control message to initialize a BTE */
#define LMSG_INIT_TAG     0x39 
#define ACK_TAG           0x37

#define DEBUG
#ifdef GNI_RC_CHECK
#undef GNI_RC_CHECK
#endif
#ifdef DEBUG
#define GNI_RC_CHECK(msg,rc) do { if(rc != GNI_RC_SUCCESS) {           CmiPrintf("[%d] %s; err=%s\n",CmiMyPe(),msg,gni_err_str[rc]); CmiAbort("GNI_RC_CHECK"); } } while(0)
#else
#define GNI_RC_CHECK(msg,rc)
#endif

#define ALIGN4(x)        (size_t)((~3)&((x)+3)) 

static int useStaticSMSG   = 1;
static int useStaticMSGQ = 0;
static int useStaticFMA = 0;
static int mysize, myrank;
static gni_nic_handle_t      nic_hndl;


static void             **smsg_attr_ptr;
gni_msgq_attr_t         msgq_attrs;
gni_msgq_handle_t       msgq_handle;
gni_msgq_ep_attr_t      msgq_ep_attrs;
gni_msgq_ep_attr_t      msgq_ep_attrs_size;
/* =====Beginning of Declarations of Machine Specific Variables===== */
static int cookie;
static int modes = 0;
static gni_cq_handle_t       rx_cqh = NULL;
static gni_cq_handle_t       rdma_cqh = NULL;
static gni_cq_handle_t       tx_cqh = NULL;
static gni_ep_handle_t       *ep_hndl_array;

typedef    struct  pending_smg
{
    int     inst_id;
    struct  pending_smg *next;
} PENDING_GETNEXT;


/* preallocated memory buffer for FMA for short message and control message */
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

typedef struct  rmda_msg
{
    int                   destNode;
    gni_post_descriptor_t *pd;
    struct  rmda_msg      *next;
}RDMA_REQUEST;

/* reuse PendingMsg memory */
static CONTROL_MSG          *control_freelist=0;
static MSG_LIST             *msglist_freelist=0;
static RDMA_REQUEST         *rdma_freelist = 0;
#define FreeControlMsg(d)       \
  do {  \
  (d)->next = control_freelist;\
  control_freelist = d;\
  } while (0); 

#define MallocControlMsg(d) \
  d = control_freelist;\
  if (d==0) {d = ((CONTROL_MSG*)malloc(sizeof(CONTROL_MSG)));\
             _MEMCHECK(d);\
  } else control_freelist = d->next;


#define FreeMsgList(d)       \
  do {  \
  (d)->next = msglist_freelist;\
  msglist_freelist = d;\
  } while (0); 

#define MallocMsgList(d) \
  d = msglist_freelist;\
  if (d==0) {d = ((MSG_LIST*)malloc(sizeof(MSG_LIST)));\
             _MEMCHECK(d);\
  } else msglist_freelist = d->next;

#define FreeRdmaRequest(d)       \
  do {  \
  (d)->next = rdma_freelist;\
  rdma_freelist = d;\
  } while (0); 

#define MallocRdmaRequest(d) \
  d = rdma_freelist;\
  if (d==0) {d = ((RDMA_REQUEST*)malloc(sizeof(RDMA_REQUEST)));\
             _MEMCHECK(d);\
  } else rdma_freelist = d->next;

/* reuse gni_post_descriptor_t */
static gni_post_descriptor_t *post_freelist=NULL;

#if 1
#define FreePostDesc(d)       \
  do {  \
    (d)->next_descr = post_freelist;\
    post_freelist = d;\
  } while (0); 

#define MallocPostDesc(d) \
  d = post_freelist;\
  if (d==0) { \
     d = ((gni_post_descriptor_t*)malloc(sizeof(gni_post_descriptor_t)));\
     _MEMCHECK(d);\
  } else post_freelist = d->next_descr;
#else

#define FreePostDesc(d)     free(d);
#define MallocPostDesc(d)   d = ((gni_post_descriptor_t*)malloc(sizeof(gni_post_descriptor_t))); _MEMCHECK(d);

#endif

static PENDING_GETNEXT     *pending_smsg_head = 0;
static PENDING_GETNEXT     *pending_smsg_tail = 0;

/* LrtsSent is called but message can not be sent by SMSGSend because of mailbox full or no credit */
static MSG_LIST *buffered_smsg_head= 0;
static MSG_LIST *buffered_smsg_tail= 0;

/* SmsgSend return success but message sent is not confirmed by remote side */

static RDMA_REQUEST  *pending_rdma_head = 0;
static RDMA_REQUEST  *pending_rdma_tail = 0;

static MSG_LIST *buffered_fma_head = 0;
static MSG_LIST *buffered_fma_tail = 0;

/* functions  */

static void
allgather(void *in,void *out, int len)
{
    //PMI_Allgather is out of order
    int i,rc, extend_len;
    int  rank_index;
    char *out_ptr, *out_ref;
    char *in2;

    extend_len = sizeof(int) + len;
    in2 = (char*)malloc(extend_len);

    memcpy(in2, &myrank, sizeof(int));
    memcpy(in2+sizeof(int), in, len);

    out_ptr = (char*)malloc(mysize*extend_len);

    rc = PMI_Allgather(in2, out_ptr, extend_len);
    GNI_RC_CHECK("allgather", rc);

    out_ref = out;

    for(i=0;i<mysize;i++) {
        //rank index 
        memcpy(&rank_index, &(out_ptr[extend_len*i]), sizeof(int));
        //copy to the rank index slot
        memcpy(&out_ref[rank_index*len], &out_ptr[extend_len*i+sizeof(int)], len);
    }

    free(out_ptr);
    free(in2);

}

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

inline
static void delay_send_small_msg(void *msg, int size, int destNode, uint8_t tag)
{
    MSG_LIST        *msg_tmp;
    MallocMsgList(msg_tmp);
    msg_tmp->destNode = destNode;
    msg_tmp->size   = size;
    msg_tmp->msg    = msg;
    msg_tmp->tag    = tag;
    msg_tmp->next   = NULL;
    if (buffered_smsg_tail == NULL) {
      buffered_smsg_head  = buffered_smsg_tail  = msg_tmp;
    }
    else {
      buffered_smsg_tail->next    = msg_tmp;
      buffered_smsg_tail          = msg_tmp;
    }
    // CmiPrintf("[%d] delay_send_small_msg msg to PE %d  tag: 0x%x \n", myrank, destNode, tag);
}

static int send_with_smsg(int destNode, int size, char *msg)
{
    gni_return_t        status  =   GNI_RC_SUCCESS;
    CONTROL_MSG         *control_msg_tmp;
    const uint8_t       tag_data    = DATA_TAG;
    const uint8_t       tag_control = LMSG_INIT_TAG ;
    uint32_t            vmdh_index  = -1;

    CmiSetMsgSize(msg, size);
#if PRINT_SYH
    lrts_send_request++;
    CmiPrintf("LrtsSend PE:%d==>%d, size=%d, messageid:%d\n", myrank, destNode, size, lrts_send_request);
#endif
    /* No mailbox available, buffer this msg and its info */
    if(buffered_smsg_head != 0)
    {
        if(size <=SMSG_MAX_MSG)
        {
            delay_send_small_msg(msg, size, destNode, tag_data);
        }
        else
        {
            MallocControlMsg(control_msg_tmp);
            control_msg_tmp->source_addr    = (uint64_t)msg;
            control_msg_tmp->source         = myrank;
            control_msg_tmp->length         =size; 
            control_msg_tmp->source_mem_hndl.qword1 = 0;
            control_msg_tmp->source_mem_hndl.qword2 = 0;
            delay_send_small_msg((char*)control_msg_tmp, sizeof(CONTROL_MSG), destNode, tag_control);
        }
        return 0;
    }
    else {
        /* Can use SMSGSend */
        if(size <= SMSG_MAX_MSG)
        {
            /* send the msg itself */
            status = GNI_SmsgSendWTag(ep_hndl_array[destNode], NULL, 0, msg, size, 0, tag_data);
            //CmiPrintf("[%d] send_with_smsg sends a data msg to PE %d status: %s\n", myrank, destNode, gni_err_str[status]);
            if (status == GNI_RC_SUCCESS)
            {
                CmiFree(msg);
                return 1;
            }
            else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE)
            {
                //CmiPrintf("[%d] data msg add to send queue\n", myrank);
                delay_send_small_msg(msg, size, destNode, tag_data);
                return 0;
            }
            else
                GNI_RC_CHECK("GNI_SmsgSendWTag", status);
        }else
        {
            /* construct a control message and send */
            //control_msg_tmp = (CONTROL_MSG *)malloc(sizeof(CONTROL_MSG));
            MallocControlMsg(control_msg_tmp);
            control_msg_tmp->source_addr    = (uint64_t)msg;
            control_msg_tmp->source         = myrank;
            control_msg_tmp->length         = size;
            
            status = MEMORY_REGISTER(onesided_hnd, nic_hndl,msg, size, &(control_msg_tmp->source_mem_hndl), &omdh);
            
            if(status == GNI_RC_ERROR_RESOURCE || status == GNI_RC_ERROR_NOMEM)
            {
                control_msg_tmp->source_mem_hndl.qword1 = 0;
                control_msg_tmp->source_mem_hndl.qword2 = 0;
            }else if(status == GNI_RC_SUCCESS)
            {
                status = GNI_SmsgSendWTag(ep_hndl_array[destNode], 0, 0, control_msg_tmp, sizeof(CONTROL_MSG), 0, tag_control);
                //CmiPrintf("[%d] send_with_smsg sends a control msg to PE %d status: %d\n", myrank, destNode, status);
                if(status == GNI_RC_SUCCESS)
                {
                    FreeControlMsg(control_msg_tmp);
                    return 1;
                }
            }
            else
            {
                GNI_RC_CHECK("MemRegister fails at ", status);
            }
            
            // Memory register fails or send fails 
            /* store into buffer smsg_list and send later */
            delay_send_small_msg((char*)control_msg_tmp, sizeof(CONTROL_MSG), destNode, tag_control);
            return 0;
        }
    }
}

static CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode)
{
    if(useStaticSMSG)
    {
        send_with_smsg(destNode, size, msg); 
    }
    else {
        CmiAssert(0);
    }
    return 0;
}

static void LrtsPreCommonInit(int everReturn){}

/* Idle-state related functions: called in non-smp mode */
void CmiNotifyIdleForGemini(void) {
    LrtsAdvanceCommunication();
}

static void LrtsPostCommonInit(int everReturn)
{
#if CMK_SMP
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdleForGemini,NULL);
#endif

}


void LrtsPostNonLocal(){}
/* pooling CQ to receive network message */
static int  processSmsg(uint64_t inst_id);
static void PumpNetworkMsgs()
{
    uint64_t            inst_id;
    PENDING_GETNEXT     *pending_next;
    int                 ret;
    gni_cq_entry_t      event_data;
    gni_return_t        status;
    while(pending_smsg_head != 0)
    {
        pending_next= pending_smsg_head;
        ret = processSmsg(pending_next->inst_id);
        if(ret == 0)
            break;
        else
        {
#if PRINT_SYH
            CmiPrintf("Msg does happen %d from %d\n", myrank, pending_next->inst_id);
#endif
            pending_smsg_head=pending_smsg_head->next;
            free(pending_next);
        }
    }
   
    while (1) {
        status = GNI_CqGetEvent(rx_cqh, &event_data);
        if(status == GNI_RC_SUCCESS)
        {
            inst_id = GNI_CQ_GET_INST_ID(event_data);
            if(GNI_CQ_OVERRUN(event_data))
            {
                CmiPrintf("ERROR in overrun PE:%d\n", myrank);
                CmiAbort("Overrun problem and abort");
            }
        }else if (status == GNI_RC_NOT_DONE)
        {
            return;
        }else
        {
            GNI_RC_CHECK("CQ Get event", status);
        }
        ret = processSmsg(inst_id);
        if (ret == 0) {
           pending_next = (PENDING_GETNEXT*)malloc(sizeof(PENDING_GETNEXT));   
           pending_next->next = 0;
           pending_next->inst_id = inst_id;
           if(pending_smsg_head == 0)
           {
              pending_smsg_head = pending_next;
           }else
               pending_smsg_tail->next =pending_next;
           pending_smsg_tail= pending_next;
        }
    }

}

// 0 means no ready message 1means msg received
static int  processSmsg(uint64_t inst_id)
{
    void                *header;
    uint8_t             msg_tag;
    const uint8_t       data_tag = DATA_TAG;
    const uint8_t       control_tag = LMSG_INIT_TAG;
    const uint8_t       ack_tag = ACK_TAG;
    gni_return_t        status;
    int                 msg_nbytes;
    void                *msg_data;
    gni_mem_handle_t    msg_mem_hndl;
    CONTROL_MSG         *request_msg;
    RDMA_REQUEST        *rdma_request_msg;
    gni_post_descriptor_t *pd;
    PENDING_GETNEXT     *pending_next;
 
    msg_tag = GNI_SMSG_ANY_TAG;
    status = GNI_SmsgGetNextWTag(ep_hndl_array[inst_id], &header, &msg_tag);
#if PRINT_SYH
    CmiPrintf("[%d] PumpNetworkMsgs small msgs is received from PE: %d, tag=0x%x, status=%s\n", myrank, inst_id, msg_tag, gni_err_str[status]);
#endif

    if(status  == GNI_RC_SUCCESS)
    {
#if PRINT_SYH
        lrts_received_msg++;
        CmiPrintf("+++[%d] PumpNetwork data msg is received, messageid:%d\n", myrank, lrts_received_msg);
#endif
        /* copy msg out and then put into queue */
        if(msg_tag == data_tag)
        {
            msg_nbytes = CmiGetMsgSize(header);
            msg_data    = CmiAlloc(msg_nbytes);
            //CmiPrintf("[%d] PumpNetworkMsgs: get datamsg, size: %d msg id:%d\n", myrank, msg_nbytes, GNI_CQ_GET_MSG_ID(event_data));
            memcpy(msg_data, (char*)header, msg_nbytes);
            handleOneRecvedMsg(msg_nbytes, msg_data);
            GNI_SmsgRelease(ep_hndl_array[inst_id]);
        }
        else if(msg_tag == control_tag) 
        {
            //CmiPrintf("[%d] PumpNetwork control msg is received\n", myrank);
            /* initial a get to transfer data from the sender side */
            request_msg = (CONTROL_MSG *) header;
            msg_data = CmiAlloc(request_msg->length);
            _MEMCHECK(msg_data);
           
            status = MEMORY_REGISTER(onesided_hnd, nic_hndl, msg_data, (request_msg->length), &msg_mem_hndl, &omdh);

            if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR) 
            {
                GNI_SmsgRelease(ep_hndl_array[inst_id]);
                GNI_RC_CHECK("Mem Register before post", status);
            }

            //buffer this request and send later
            MallocPostDesc(pd);
            if(request_msg->length < LRTS_GNI_RDMA_THRESHOLD) 
                pd->type            = GNI_POST_FMA_GET;
            else
                pd->type            = GNI_POST_RDMA_GET;

            pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT |  GNI_CQMODE_REMOTE_EVENT;
            pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
            pd->length          = ALIGN4(request_msg->length);
            pd->local_addr      = (uint64_t) msg_data;
            pd->remote_addr     = request_msg->source_addr;
            pd->remote_mem_hndl = request_msg->source_mem_hndl;
            pd->src_cq_hndl     = 0;     /* tx_cqh;  */
            pd->rdma_mode       = 0;

            //memory registration successful
            if(status == GNI_RC_SUCCESS)
            {
                pd->local_mem_hndl  = msg_mem_hndl;
                if(pd->type == GNI_POST_RDMA_GET) 
                    status = GNI_PostRdma(ep_hndl_array[request_msg->source], pd);
                else
                    status = GNI_PostFma(ep_hndl_array[request_msg->source],  pd);
            }else
            {
                pd->local_mem_hndl.qword1  = 0; 
                pd->local_mem_hndl.qword1  = 0; 
            }
            GNI_SmsgRelease(ep_hndl_array[inst_id]);
            if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
            {
                MallocRdmaRequest(rdma_request_msg);
                rdma_request_msg->next = 0;
                rdma_request_msg->destNode = inst_id;
                if(pending_rdma_head == 0)
                {
                    pending_rdma_head = rdma_request_msg;
                }else
                {
                    pending_rdma_tail->next = rdma_request_msg;
                }
                pending_rdma_tail = rdma_request_msg;
                return 1;
            }else
                GNI_RC_CHECK("AFter posting", status);
        }
        else if(msg_tag == ack_tag) {
            /* Get is done, release message . Now put is not used yet*/
            request_msg = (CONTROL_MSG *) header;
            MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &(request_msg->source_mem_hndl), &omdh);
            
            CmiFree((void*)request_msg->source_addr);
            GNI_SmsgRelease(ep_hndl_array[inst_id]);
            SendRdmaMsg();
        }else{
            GNI_SmsgRelease(ep_hndl_array[inst_id]);
            CmiPrintf("weird tag problem\n");
            CmiAbort("Unknown tag\n");
        }
        return 1;
    }else 
    {
        return 0;
    }
}

/* Check whether message send or get is confirmed by remote */
static void PumpLocalTransactions()
{
    gni_cq_entry_t ev;
    gni_return_t status;
    uint64_t type, inst_id;
    uint8_t         ack_tag = ACK_TAG;
    gni_post_descriptor_t *tmp_pd;
    //gni_post_descriptor_t   ack_pd;
    MSG_LIST  *ptr;
    CONTROL_MSG *ack_msg_tmp;

    while (1) 
    {
        status = GNI_CqGetEvent(tx_cqh, &ev);
        if(status == GNI_RC_SUCCESS)
        {
            type        = GNI_CQ_GET_TYPE(ev);
            inst_id     = GNI_CQ_GET_INST_ID(ev);
        }else if (status == GNI_RC_NOT_DONE)
        {
            return;
        }else
        {
            GNI_RC_CHECK("CQ Get event", status);
        }
#if PRINT_SYH
        lrts_local_done_msg++;
        CmiPrintf("*[%d]  PumpLocalTransactions GNI_CQ_GET_TYPE %d. Localdone=%d\n", myrank, GNI_CQ_GET_TYPE(ev), lrts_local_done_msg);
#endif
        if (type == GNI_CQ_EVENT_TYPE_SMSG) {
#if PRINT_SYH
            CmiPrintf("**[%d] PumpLocalTransactions localdone=%d\n", myrank,  lrts_local_done_msg);
#endif
        }
        else if(type == GNI_CQ_EVENT_TYPE_POST)
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
                MallocControlMsg(ack_msg_tmp);
                ack_msg_tmp->source = myrank;
                ack_msg_tmp->source_addr = tmp_pd->remote_addr;
                ack_msg_tmp->length=tmp_pd->length; 
                ack_msg_tmp->source_mem_hndl = tmp_pd->remote_mem_hndl;
                //CmiPrintf("PE:%d sending ACK back addr=%p \n", myrank, ack_msg_tmp->source_addr); 
           
                if(buffered_smsg_head!=0)
                {
                    delay_send_small_msg(ack_msg_tmp, sizeof(CONTROL_MSG), inst_id, ack_tag);
                }else
                {
                    status = GNI_SmsgSendWTag(ep_hndl_array[inst_id], 0, 0, ack_msg_tmp, sizeof(CONTROL_MSG), 0, ack_tag);
                    if(status == GNI_RC_SUCCESS)
                    {
                        FreeControlMsg(ack_msg_tmp);
                    }else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE)
                    {
                        delay_send_small_msg(ack_msg_tmp, sizeof(CONTROL_MSG), inst_id, ack_tag);
                    }
                    else
                        GNI_RC_CHECK("GNI_SmsgSendWTag", status);
                }
                MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &tmp_pd->local_mem_hndl, &omdh);
                
                handleOneRecvedMsg(SIZEFIELD((void*)tmp_pd->local_addr), (void*)tmp_pd->local_addr); 
                SendRdmaMsg(); 
            }
            FreePostDesc(tmp_pd);
        }
    }   /* end of while loop */
}

static int SendRdmaMsg()
{

    RDMA_REQUEST            *ptr;
    gni_post_descriptor_t   *pd;
    gni_return_t            status = GNI_RC_SUCCESS;
    gni_mem_handle_t        msg_mem_hndl;
    while(pending_rdma_head != 0)
    {
        ptr=pending_rdma_head;
        pd = ptr->pd;
        // register memory first
        if( pd->local_mem_hndl.qword1  == 0 && pd->local_mem_hndl.qword2  == 0)
        {
            status = MEMORY_REGISTER(onesided_hnd, nic_hndl, pd->local_addr, pd->length, &(pd->local_mem_hndl), &omdh);
        }
        if(status == GNI_RC_SUCCESS)
        {
            if(pd->type == GNI_POST_RDMA_GET) 
                status = GNI_PostRdma(ep_hndl_array[ptr->destNode], pd);
            else
                status = GNI_PostFma(ep_hndl_array[ptr->destNode],  pd);
            if(status == GNI_RC_SUCCESS)
            {
                pending_rdma_head = pending_rdma_head->next; 
                FreePostDesc(pd);
                FreeRdmaRequest(ptr);
            }
            else
                return 1;
        }else
            return 1;
    } //end while
    return 0;
}

static int SendBufferMsg()
{
    MSG_LIST            *ptr;
    CONTROL_MSG         *control_msg_tmp;
    uint8_t             tag_data, tag_control, tag_ack;
    gni_return_t        status;

    tag_data    = DATA_TAG;
    tag_control = LMSG_INIT_TAG;
    tag_ack     = ACK_TAG;
    /* can add flow control here to control the number of messages sent before handle message */
    while(buffered_smsg_head != 0)
    {
        if(useStaticSMSG)
        {
            ptr = buffered_smsg_head;
            if(ptr->tag == tag_data)
            {
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], NULL, 0, ptr->msg, ptr->size, 0, tag_data);
                //CmiPrintf("[%d] SendBufferMsg sends a data msg to PE %d status: %s\n", myrank, ptr->destNode, gni_err_str[status]);
                if(status == GNI_RC_SUCCESS) {
                    CmiFree(ptr->msg);
                }
            }
            else if(ptr->tag ==tag_control)
            {
                control_msg_tmp = (CONTROL_MSG*)ptr->msg;
                if(control_msg_tmp->source_mem_hndl.qword1 == 0 && control_msg_tmp->source_mem_hndl.qword2 == 0)
                {
                    MEMORY_REGISTER(onesided_hnd, nic_hndl, control_msg_tmp->source_addr, control_msg_tmp->length, &(control_msg_tmp->source_mem_hndl), &omdh);
                    if(status != GNI_RC_SUCCESS)
                        break;
                }
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], 0, 0, ptr->msg, sizeof(CONTROL_MSG), 0, tag_control);
                //CmiPrintf("[%d] SendBufferMsg sends a control msg to PE %d status: %d\n", myrank, ptr->destNode, status);
                if(status == GNI_RC_SUCCESS) {
                    FreeControlMsg((CONTROL_MSG*)(ptr->msg));
                }
            }else if (ptr->tag == tag_ack)
            {
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], 0, 0, ptr->msg, sizeof(CONTROL_MSG), 0, tag_ack);
                //CmiPrintf("[%d] SendBufferMsg sends a tag msg to PE %d status: %d\n", myrank, ptr->destNode, status);
                if(status == GNI_RC_SUCCESS) {
                    FreeControlMsg((CONTROL_MSG*)ptr->msg);
                }
            }
        } else if(useStaticMSGQ)
        {
            CmiAbort("MSGQ Send not done\n");
        }else
        {
            CmiAbort("FMA Send not done\n");
        }
        if(status == GNI_RC_SUCCESS)
        {
            buffered_smsg_head = buffered_smsg_head->next;
            FreeMsgList(ptr);
        }else
            return 0;
    }
    return 1;
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

static void _init_static_smsg()
{
    gni_smsg_attr_t      *smsg_attr;
    gni_smsg_attr_t      *smsg_attr_vec;
    unsigned int         smsg_memlen;
    gni_mem_handle_t     my_smsg_mdh_mailbox;
    register    int      i;
    gni_return_t status;
    uint32_t              vmdh_index = -1;

     smsg_attr = (gni_smsg_attr_t *)malloc(mysize*sizeof(gni_smsg_attr_t));
    _MEMCHECK(smsg_attr);

    smsg_attr_ptr = malloc(sizeof(void*) *mysize);
    for(i=0; i<mysize; i++)
    {
        if(i==myrank)
            continue;
        smsg_attr[i].msg_type = GNI_SMSG_TYPE_MBOX;//GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
        smsg_attr[i].mbox_offset = 0;
        smsg_attr[i].mbox_maxcredit = SMSG_MAX_CREDIT;
        smsg_attr[i].msg_maxsize = SMSG_MAX_MSG;
        status = GNI_SmsgBufferSizeNeeded(&smsg_attr[i], &smsg_memlen);
        GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);

        smsg_attr_ptr[i] = memalign(64, smsg_memlen);
        _MEMCHECK(smsg_attr_ptr[i]);
        bzero(smsg_attr_ptr[i], smsg_memlen);
        status = GNI_MemRegister(nic_hndl, (uint64_t)smsg_attr_ptr[i],
            smsg_memlen, rx_cqh,
            GNI_MEM_READWRITE | GNI_MEM_USE_GART | GNI_MEM_PI_FLUSH,   
            vmdh_index,
            &my_smsg_mdh_mailbox);

        GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);
      
        smsg_attr[i].msg_buffer = smsg_attr_ptr[i];
        smsg_attr[i].buff_size = smsg_memlen;
        smsg_attr[i].mem_hndl = my_smsg_mdh_mailbox;
    }
    smsg_attr_vec = (gni_smsg_attr_t*)malloc(mysize * mysize * sizeof(gni_smsg_attr_t));
    CmiAssert(smsg_attr_vec);
   
    allgather(smsg_attr, smsg_attr_vec,  mysize*sizeof(gni_smsg_attr_t));
    for(i=0; i<mysize; i++)
    {
        if (myrank == i) continue;
        /* initialize the smsg channel */
        status = GNI_SmsgInit(ep_hndl_array[i], &smsg_attr[i], &smsg_attr_vec[i*mysize+myrank]);
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
    register int            i;
    int                     rc;
    int                     device_id = 0;
    unsigned int            remote_addr;
    gni_cdm_handle_t        cdm_hndl;
    gni_return_t            status = GNI_RC_SUCCESS;
    uint32_t                vmdh_index = -1;
    uint8_t                 ptag;
    unsigned int            local_addr, *MPID_UGNI_AllAddr;
    int                     first_spawned;
    int                     physicalID;
    //void (*local_event_handler)(gni_cq_entry_t *, void *)       = &LocalEventHandle;
    //void (*remote_smsg_event_handler)(gni_cq_entry_t *, void *) = &RemoteSmsgEventHandle;
    //void (*remote_bte_event_handler)(gni_cq_entry_t *, void *)  = &RemoteBteEventHandle;
    
    //useStaticSMSG = CmiGetArgFlag(*argv, "+useStaticSmsg");
    //useStaticMSGQ = CmiGetArgFlag(*argv, "+useStaticMsgQ");
    
    status = PMI_Init(&first_spawned);
    GNI_RC_CHECK("PMI_Init", status);

    status = PMI_Get_size(&mysize);
    GNI_RC_CHECK("PMI_Getsize", status);

    status = PMI_Get_rank(&myrank);
    GNI_RC_CHECK("PMI_getrank", status);

    physicalID = CmiPhysicalNodeID(myrank);
    
    printf("Pysical Node ID:%d for PE:%d\n", physicalID, myrank);

    *myNodeID = myrank;
    *numNodes = mysize;
  
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
    allgather(&local_addr, MPID_UGNI_AllAddr, sizeof(unsigned int));
    /* create the local completion queue */
    /* the third parameter : The number of events the NIC allows before generating an interrupt. Setting this parameter to zero results in interrupt delivery with every event. When using this parameter, the mode parameter must be set to GNI_CQ_BLOCKING*/
    status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &tx_cqh);
    GNI_RC_CHECK("GNI_CqCreate (tx)", status);
    
    /* create the destination completion queue for receiving micro-messages, make this queue considerably larger than the number of transfers */

    status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &rx_cqh);
    GNI_RC_CHECK("Create CQ (rx)", status);
    
    //status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &rdma_cqh);
    //GNI_RC_CHECK("Create BTE CQ", status);

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
    }
    free(MPID_UGNI_AllAddr);
}

#define ALIGNBUF                64

void* LrtsAlloc(int n_bytes, int header)
{
    if(n_bytes <= SMSG_MAX_MSG)
    {
        int totalsize = n_bytes+header;
        return malloc(totalsize);
    }else 
    {
        CmiAssert(header <= ALIGNBUF);
        n_bytes = ALIGN4(n_bytes);           /* make sure size if 4 aligned */
        char *res = memalign(ALIGNBUF, n_bytes+ALIGNBUF);
        return res + ALIGNBUF - header;
    }
}

void  LrtsFree(void *msg)
{
    int size = SIZEFIELD((char*)msg+sizeof(CmiChunkHeader));
    if (size <= SMSG_MAX_MSG)
      free(msg);
    else
      free((char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);
}

static void LrtsExit()
{
    /* free memory ? */
    PMI_Finalize();
    exit(0);
}

static void LrtsDrainResources()
{
    while (!SendBufferMsg()) {
      PumpNetworkMsgs();
      PumpLocalTransactions();
    }
}

void CmiAbort(const char *message) {

    CmiPrintf("CmiAbort is calling on PE:%d\n", myrank);
    PMI_Abort(-1, message);
}

/**************************  TIMER FUNCTIONS **************************/
#if CMK_TIMER_USE_SPECIAL
/* MPI calls are not threadsafe, even the timer on some machines */
static CmiNodeLock  timerLock = 0;
static int _absoluteTime = 0;
static double starttimer = 0;
static int _is_global = 0;

int CmiTimerIsSynchronized() {
    return 0;
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
}

/**
 * Since the timerLock is never created, and is
 * always NULL, then all the if-condition inside
 * the timer functions could be disabled right
 * now in the case of SMP.
 */
double CmiTimer(void) {

    return 0;
}

double CmiWallTimer(void) {
    return 0;
}

double CmiCpuTimer(void) {
    return 0;
}

#endif
/************Barrier Related Functions****************/

int CmiBarrier()
{
    int status;
    status = PMI_Barrier();
    return status;

}
