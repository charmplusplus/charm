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
#include <stdint.h>
#include <errno.h>
#include <malloc.h>

#include "gni_pub.h"
#include "pmi.h"

#include "converse.h"

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

#define USE_LRTS_MEMPOOL   1

#if USE_LRTS_MEMPOOL
static size_t _mempool_size = 1024ll*1024*8;
#endif

#define PRINT_SYH  0

#if PRINT_SYH
int         lrts_smsg_success = 0;
int         lrts_send_msg_id = 0;
int         lrts_send_rdma_id = 0;
int         lrts_send_rdma_success = 0;
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
#define  MEMORY_REGISTER(handler, nic_hndl, msg, size, mem_hndl, myomdh) GNI_MemRegister(nic_hndl, (uint64_t)msg,  (uint64_t)size, smsg_rx_cqh,  GNI_MEM_READWRITE, -1, mem_hndl)

#define  MEMORY_DEREGISTER(handler, nic_hndl, mem_hndl, myomdh)  GNI_MemDeregister(nic_hndl, (mem_hndl))
#endif

#define CmiGetMsgSize(m)  ((CmiMsgHeaderExt*)m)->size
#define CmiSetMsgSize(m,s)  ((((CmiMsgHeaderExt*)m)->size)=(s))

#define ALIGNBUF                64

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* If SMSG is not used */
#define FMA_PER_CORE  1024
#define FMA_BUFFER_SIZE 1024
/* If SMSG is used */
static int  SMSG_MAX_MSG;
//static int  log2_SMSG_MAX_MSG;
#define SMSG_MAX_CREDIT  36

#define MSGQ_MAXSIZE       4096
/* large message transfer with FMA or BTE */
#define LRTS_GNI_RDMA_THRESHOLD  4096

#define REMOTE_QUEUE_ENTRIES  1048576
#define LOCAL_QUEUE_ENTRIES   32

#define ACK_TAG           0x30
/* SMSG is data message */
#define SMALL_DATA_TAG          0x31
/* SMSG is a control message to initialize a BTE */
#define MEDIUM_HEAD_TAG         0x32
#define MEDIUM_DATA_TAG         0x33
#define LMSG_INIT_TAG     0x39 

#define DEBUG
#ifdef GNI_RC_CHECK
#undef GNI_RC_CHECK
#endif
#ifdef DEBUG
#define GNI_RC_CHECK(msg,rc) do { if(rc != GNI_RC_SUCCESS) {           CmiPrintf("[%d] %s; err=%s\n",CmiMyPe(),msg,gni_err_str[rc]); CmiAbort("GNI_RC_CHECK"); } } while(0)
#else
#define GNI_RC_CHECK(msg,rc)
#endif

#define ALIGN64(x)       (size_t)((~63)&((x)+63))
#define ALIGN4(x)        (size_t)((~3)&((x)+3)) 

static int Mempool_MaxSize = 1024*1024*128;
static int useStaticSMSG   = 1;
static int useStaticMSGQ = 0;
static int useStaticFMA = 0;
static int mysize, myrank;
static gni_nic_handle_t      nic_hndl;


static void             *smsg_mailbox_base;
gni_msgq_attr_t         msgq_attrs;
gni_msgq_handle_t       msgq_handle;
gni_msgq_ep_attr_t      msgq_ep_attrs;
gni_msgq_ep_attr_t      msgq_ep_attrs_size;

/* preallocated memory buffer for FMA for short message and control message */
typedef struct {
    gni_mem_handle_t mdh;
    uint64_t addr;
} mdh_addr_t;


/* preallocated DMA buffer */
int                     DMA_slots;
uint64_t                DMA_avail_tag = 0;
uint32_t                DMA_incoming_avail_tag = 0;
uint32_t                DMA_outgoing_avail_tag = 0;
void                    *DMA_incoming_base_addr;
void                    *DMA_outgoing_base_addr;
mdh_addr_t              DMA_buffer_base_mdh_addr;
mdh_addr_t              *DMA_buffer_base_mdh_addr_vec;
int                     DMA_buffer_size;
int                     DMA_max_single_msg = 131072;//524288 ;

#define                 DMA_SIZE_PER_SLOT       8192


typedef struct dma_msgid_map
{
    uint64_t     msg_id;
    int     msg_subid
} dma_msgid_map_t;

dma_msgid_map_t         *dma_map_list;

typedef struct msg_trace
{
    uint64_t    msg_id;
    int         done_num;
}msg_trace_t;

msg_trace_t             *pending_msg_list;
/* =====Beginning of Declarations of Machine Specific Variables===== */
static int cookie;
static int modes = 0;
static gni_cq_handle_t       smsg_rx_cqh = NULL;
static gni_cq_handle_t       smsg_tx_cqh = NULL;
static gni_cq_handle_t       post_rx_cqh = NULL;
static gni_cq_handle_t       post_tx_cqh = NULL;
static gni_ep_handle_t       *ep_hndl_array;

typedef    struct  pending_smg
{
    int     inst_id;
    struct  pending_smg *next;
} PENDING_GETNEXT;


typedef struct msg_list
{
    uint32_t destNode;
    uint32_t size;
    void *msg;
    struct msg_list *next;
    uint8_t tag;
}MSG_LIST;

typedef struct medium_msg_list
{
    uint32_t destNode;
    uint32_t msg_id;
    uint32_t msg_subid;
    uint32_t remain_size;
    void *msg;
    struct medium_msg_list *next;
}MEDIUM_MSG_LIST;


typedef struct control_msg
{
    uint64_t            source_addr;
    int                 source;               /* source rank */
    int                 length;
    gni_mem_handle_t    source_mem_hndl;
    struct control_msg *next;
}CONTROL_MSG;

typedef struct medium_msg_control
{
    uint64_t            dma_offset;     //the dma_buffer for this block of msg
    int                 msg_id;         //Id for the total index
    int                 msg_subid;      //offset inside the message id 
}MEDIUM_MSG_CONTROL;

typedef struct  rmda_msg
{
    int                   destNode;
    gni_post_descriptor_t *pd;
    struct  rmda_msg      *next;
}RDMA_REQUEST;

/* reuse PendingMsg memory */
static CONTROL_MSG          *control_freelist=0;
static MSG_LIST             *msglist_freelist=0;
static MSG_LIST             **smsg_msglist_head= 0;
static MSG_LIST             **smsg_msglist_tail= 0;
static MSG_LIST             *smsg_free_head=0;
static MSG_LIST             *smsg_free_tail=0;

/*
#define FreeMsgList(msg_head, msg_tail, free_head, free_tail)       \
    if(free_head == 0)  free_head = free_tail = msg_head;    \
    else   free_tail = free_tail->next;    \
    if( msg_head->next == msg_tail) msg_head =0;   \
    else msg_head= msg_head->next;    

#define MallocMsgList(d, msg_head, msg_tail, free_head, free_tail, msgsize) \
    if(free_head == 0) {d= malloc(msgsize);  \
        if(msg_head == 0)   msg_head =msg_tail = msg_head->next = msg_tail->next = d; \
        else { msg_tail->next = d; d->next = msg_head; msg_tail=d;} \
    }else {d = free_head; free_head = free_head->next; if(free_tail->next == free_head) free_head =0;} \
*/

#define FreeMsgList(d)  \
  (d)->next = msglist_freelist;\
  msglist_freelist = d;



#define MallocMsgList(d) \
  d = msglist_freelist;\
  if (d==0) {d = ((MSG_LIST*)malloc(sizeof(MSG_LIST)));\
             _MEMCHECK(d);\
  } else msglist_freelist = d->next;


#define FreeControlMsg(d)       \
  (d)->next = control_freelist;\
  control_freelist = d;

#define MallocControlMsg(d) \
  d = control_freelist;\
  if (d==0) {d = ((CONTROL_MSG*)malloc(sizeof(CONTROL_MSG)));\
             _MEMCHECK(d);\
  } else control_freelist = d->next;


static RDMA_REQUEST         *rdma_freelist = NULL;

#define FreeControlMsg(d)       \
  (d)->next = control_freelist;\
  control_freelist = d;

#define MallocControlMsg(d) \
  d = control_freelist;\
  if (d==0) {d = ((CONTROL_MSG*)malloc(sizeof(CONTROL_MSG)));\
             _MEMCHECK(d);\
  } else control_freelist = d->next;

#define FreeMediumControlMsg(d)       \
  (d)->next = medium_control_freelist;\
  medium_control_freelist = d;


#define MallocMediumControlMsg(d) \
    d = medium_control_freelist;\
    if (d==0) {d = ((MEDIUM_MSG_CONTROL*)malloc(sizeof(MEDIUM_MSG_CONTROL)));\
    _MEMCHECK(d);\
} else mediumcontrol_freelist = d->next;

#define FreeRdmaRequest(d)       \
  (d)->next = rdma_freelist;\
  rdma_freelist = d;

#define MallocRdmaRequest(d) \
  d = rdma_freelist;\
  if (d==0) {d = ((RDMA_REQUEST*)malloc(sizeof(RDMA_REQUEST)));\
             _MEMCHECK(d);\
  } else rdma_freelist = d->next;

/* reuse gni_post_descriptor_t */
static gni_post_descriptor_t *post_freelist=0;

#if 1
#define FreePostDesc(d)       \
    (d)->next_descr = post_freelist;\
    post_freelist = d;

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
static int      buffered_smsg_counter = 0;

/* SmsgSend return success but message sent is not confirmed by remote side */

static RDMA_REQUEST  *pending_rdma_head = 0;
static RDMA_REQUEST  *pending_rdma_tail = 0;
static MSG_LIST *buffered_fma_head = 0;
static MSG_LIST *buffered_fma_tail = 0;

/* functions  */
#define IsFree(a,ind)  !( a& (1<<(ind) ))
#define SET_BITS(a,ind) a = ( a | (1<<(ind )) )
#define Reset(a,ind) a = ( a & (~(1<<(ind))) )

#include "mempool.c"

#if CMK_PERSISTENT_COMM
#include "machine-persistent.c"
#endif

/* get the upper bound of log 2 */
int mylog2(int size)
{
    int op = size;
    unsigned int ret=0;
    unsigned int mask = 0;
    int i;
    while(op>0)
    {
        op = op >> 1;
        ret++;

    }
    for(i=1; i<ret; i++)
    {
        mask = mask << 1;
        mask +=1;
    }

    ret -= ((size &mask) ? 0:1);
    return ret;
}

static void
allgather(void *in,void *out, int len)
{
    static int *ivec_ptr=NULL,already_called=0,job_size=0;
    int i,rc;
    int my_rank;
    char *tmp_buf,*out_ptr;

    if(!already_called) {

        rc = PMI_Get_size(&job_size);
        CmiAssert(rc == PMI_SUCCESS);
        rc = PMI_Get_rank(&my_rank);
        CmiAssert(rc == PMI_SUCCESS);

        ivec_ptr = (int *)malloc(sizeof(int) * job_size);
        CmiAssert(ivec_ptr != NULL);

        rc = PMI_Allgather(&my_rank,ivec_ptr,sizeof(int));
        CmiAssert(rc == PMI_SUCCESS);

        already_called = 1;

    }

    tmp_buf = (char *)malloc(job_size * len);
    CmiAssert(tmp_buf);

    rc = PMI_Allgather(in,tmp_buf,len);
    CmiAssert(rc == PMI_SUCCESS);

    out_ptr = out;

    for(i=0;i<job_size;i++) {

        memcpy(&out_ptr[len * ivec_ptr[i]],&tmp_buf[i * len],len);

    }

    free(tmp_buf);
}
static void
allgather_2(void *in,void *out, int len)
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
    msg_tmp->next   = 0;
    if (smsg_msglist_head[destNode] == 0) {
        smsg_msglist_head[destNode]  = msg_tmp;
    }
    else {
      smsg_msglist_tail[destNode]->next    = msg_tmp;
    }
    smsg_msglist_tail[destNode]          = msg_tmp;
#if PRINT_SYH
    buffered_smsg_counter++;
#endif
}

// messages smaller than max single SMSG
inline
static void send_small_messages(int destNode, int size, char *msg)
{
    gni_return_t        status  =   GNI_RC_SUCCESS;
    const uint8_t       tag_data    = SMALL_DATA_TAG;
    
    if(smsg_msglist_head[destNode] == 0)
    {
        status = GNI_SmsgSendWTag(ep_hndl_array[destNode], NULL, 0, msg, size, 0, SMALL_DATA_TAG);
        if (status == GNI_RC_SUCCESS)
        {
#if PRINT_SYH
            lrts_smsg_success++;
            if(lrts_smsg_success == lrts_send_msg_id)
                CmiPrintf("GOOD [%d==>%d] sent done%d (msgs=%d)\n", myrank, destNode, lrts_smsg_success, lrts_send_msg_id);
            else
                CmiPrintf("BAD [%d==>%d] sent done%d (msgs=%d)\n", myrank, destNode, lrts_smsg_success, lrts_send_msg_id);
#endif
            CmiFree(msg);
            return;
        }else if (status == GNI_RC_INVALID_PARAM)
        {
            CmiAbort("SmsgSen fails\n");
        }
    }
    //buffer this message when buffer_msg_head!=0 or send fails
    delay_send_small_msg(msg, size, destNode, SMALL_DATA_TAG);
}

// Get first 0 in DMA_tags starting from index
static int get_first_avail_bit(uint64_t DMA_tags, int start_index)
{

    uint64_t         mask = 0x1;
    register    int     i=0;
    while((DMA_tags & mask) && i<DMA_slots) {mask << 1; i++;}

}

static int send_medium_messages(int destNode, int size, char *msg)
{
#if 0
    gni_return_t status = GNI_RC_SUCCESS;
    int first_avail_bit=0;
    uint64_t mask = 0x1;
    MEDIUM_MSG_CONTROL  *medium_msg_control_tmp;
    MEDIUM_MSG_LIST        *msg_tmp;
    int blocksize, remain_size, pos;
    int sub_id = 0;
    remain_size = size;
    pos = 0;  //offset before which data are sent
    /* copy blocks of the message to DMA preallocated buffer and send SMSG */
    //Check whether there is any available DMA buffer
    
    do{
        while((DMA_avail_tag & mask) && first_avail_bit<DMA_slots) {mask << 1; first_avail_bit++;}
        if(first_avail_bit == DMA_slots) //No available DMA, buffer this message
        {
            MallocMediumMsgList(msg_tmp);
            msg_tmp->destNode = destNode;
            msg_tmp->msg_id   = lrts_send_msg_id;
            msg_tmp->msg_subid   = sub_id;
            msg_tmp->size   = remain_size;
            msg_tmp->msg    = msg+pos;
            msg_tmp->next   = NULL;
            break;
        }else
        {
            //copy this part of the message into this DMA buffer
            //TODO optimize here, some data can go with this SMSG
            blocksize = (remain_size>DMA_SIZE_PER_SLOT)?DMA_SIZE_PER_SLOT: remain_size;
            memcpy(DMA_buffer_base_mdh_addr.addr[first_avail_bit], msg+pos, blocksize);
            pos += blocksize;
            remain_size -= blocksize;
            SET_BITS(DMA_avail_tag, first_avail_bit);
           
            MallocMediumControlMsg(medium_msg_control_tmp);
            medium_msg_control_tmp->msg_id = lrts_send_msg_id;
            medium_msg_control_tmp->msg_subid = sub_id;
            if(status == GNI_RC_SUCCESS)
            {
                if(sub_id==0)
                    status = GNI_SmsgSendWTag(ep_hndl_array[destNode], NULL, 0, medium_msg_tmp, sizeof(MEDIUM_MSG_CONTROL), 0, MEDIUM_HEAD_TAG);
                else
                    status = GNI_SmsgSendWTag(ep_hndl_array[destNode], NULL, 0, medium_msg_tmp, sizeof(MEDIUM_MSG_CONTROL), 0, MEDIUM_DATA_TAG);
            }
            //buffer this smsg
            if(status != GNI_RC_SUCCESS)
            {
                delay_send_small_msg(medium_msg_tmp, sizeof(MEDIUM_MSG_CONTROL), destNode, MEDIUM_HEAD_TAG);
            }
            sub_id++;
        }while(remain_size > 0 );

        }
    }
#endif
}

// Large message, send control to receiver, receiver register memory and do a GET 
inline
static int send_large_messages(int destNode, int size, char *msg)
{
#if     USE_LRTS_MEMPOOL
    gni_return_t        status  =   GNI_RC_SUCCESS;
    CONTROL_MSG         *control_msg_tmp;
    uint32_t            vmdh_index  = -1;
    /* construct a control message and send */
    MallocControlMsg(control_msg_tmp);
    control_msg_tmp->source_addr    = (uint64_t)msg;
    control_msg_tmp->source         = myrank;
    control_msg_tmp->length         =ALIGN4(size); //for GET 4 bytes aligned 
    //memcpy( &(control_msg_tmp->source_mem_hndl), GetMemHndl(msg), sizeof(gni_mem_handle_t)) ;
    control_msg_tmp->source_mem_hndl = GetMemHndl(msg);
#if PRINT_SYH
    lrts_send_msg_id++;
    CmiPrintf("Large LrtsSend PE:%d==>%d, size=%d, messageid:%d LMSG\n", myrank, destNode, size, lrts_send_msg_id);
#endif
    status = GNI_SmsgSendWTag(ep_hndl_array[destNode], 0, 0, control_msg_tmp, sizeof(CONTROL_MSG), 0, LMSG_INIT_TAG);
    if(status == GNI_RC_SUCCESS)
    {
#if PRINT_SYH
        lrts_smsg_success++;
        if(lrts_smsg_success == lrts_send_msg_id)
            CmiPrintf("GOOD [%d==>%d] sent done%d (msgs=%d)\n", myrank, destNode, lrts_smsg_success, lrts_send_msg_id);
        else
            CmiPrintf("BAD [%d==>%d] sent done%d (msgs=%d)\n", myrank, destNode, lrts_smsg_success, lrts_send_msg_id);
#endif
        FreeControlMsg(control_msg_tmp);
        return 1;
    }

    delay_send_small_msg((char*)control_msg_tmp, sizeof(CONTROL_MSG), destNode, LMSG_INIT_TAG);
    return 0;
// NOT use mempool, should slow 
#else
    gni_return_t        status  =   GNI_RC_SUCCESS;
    CONTROL_MSG         *control_msg_tmp;
    uint32_t            vmdh_index  = -1;
    /* construct a control message and send */
    MallocControlMsg(control_msg_tmp);
    control_msg_tmp->source_addr    = (uint64_t)msg;
    control_msg_tmp->source         = myrank;
    control_msg_tmp->length         =ALIGN4(size); //for GET 4 bytes aligned 
    control_msg_tmp->source_mem_hndl.qword1 = 0;
    control_msg_tmp->source_mem_hndl.qword2 = 0;
#if PRINT_SYH
    lrts_send_msg_id++;
    CmiPrintf("Large LrtsSend PE:%d==>%d, size=%d, messageid:%d LMSG\n", myrank, destNode, size, lrts_send_msg_id);
#endif
    if(smsg_msglist_head[destNode] == 0)
    {
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl,msg, ALIGN4(size), &(control_msg_tmp->source_mem_hndl), &omdh);
        if(status == GNI_RC_SUCCESS)
        {
            status = GNI_SmsgSendWTag(ep_hndl_array[destNode], 0, 0, control_msg_tmp, sizeof(CONTROL_MSG), 0, LMSG_INIT_TAG);
            if(status == GNI_RC_SUCCESS)
            {
#if PRINT_SYH
                lrts_smsg_success++;
                if(lrts_smsg_success == lrts_send_msg_id)
                    CmiPrintf("GOOD [%d==>%d] sent done%d (msgs=%d)\n", myrank, destNode, lrts_smsg_success, lrts_send_msg_id);
                else
                    CmiPrintf("BAD [%d==>%d] sent done%d (msgs=%d)\n", myrank, destNode, lrts_smsg_success, lrts_send_msg_id);
#endif
                FreeControlMsg(control_msg_tmp);
                return 1;
            }

        } else if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR)
        {
            CmiAbort("Memory registor for large msg\n");
        }
    }
    delay_send_small_msg((char*)control_msg_tmp, sizeof(CONTROL_MSG), destNode, LMSG_INIT_TAG);
    return 0;
#endif
}

static CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode)
{
    CmiSetMsgSize(msg, size);

    if(size <= SMSG_MAX_MSG)
    {
#if PRINT_SYH
    lrts_send_msg_id++;
    CmiPrintf("SMSG LrtsSend PE:%d==>%d, size=%d, messageid:%d\n", myrank, destNode, size, lrts_send_msg_id);
#endif
    send_small_messages(destNode, size, msg);
    }/*else if(size <=DMA_max_single_msg )
    {
        send_medium_messages(destNode, size, msg);
    }*/else
    {
        send_large_messages(destNode, size, msg);
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
static void PumpNetworkRdmaMsgs()
{
    gni_cq_entry_t      event_data;
    gni_return_t        status;
    while( (status = GNI_CqGetEvent(post_rx_cqh, &event_data)) == GNI_RC_SUCCESS);
}

static int SendRdmaMsg();
static void getLargeMsgRequest(void* header, uint64_t inst_id);
static void PumpNetworkSmsg()
{
    uint64_t            inst_id;
    PENDING_GETNEXT     *pending_next;
    int                 ret;
    gni_cq_entry_t      event_data;
    gni_return_t        status;
    void                *header;
    uint8_t             msg_tag;
    int                 msg_nbytes;
    void                *msg_data;
    gni_mem_handle_t    msg_mem_hndl;
 

    while ((status =GNI_CqGetEvent(smsg_rx_cqh, &event_data)) == GNI_RC_SUCCESS)
    {
        inst_id = GNI_CQ_GET_INST_ID(event_data);
        // GetEvent returns success but GetNext return not_done. caused by Smsg out-of-order transfer
#if PRINT_SYH
        CmiPrintf("[%d] PumpNetworkMsgs small msgs is received from PE: %d,  status=%s\n", myrank, inst_id,  gni_err_str[status]);
#endif
        msg_tag = GNI_SMSG_ANY_TAG;
        while( (status = GNI_SmsgGetNextWTag(ep_hndl_array[inst_id], &header, &msg_tag)) == GNI_RC_SUCCESS)
        {
#if PRINT_SYH
            lrts_received_msg++;
            CmiPrintf("+++[%d] PumpNetwork msg is received, messageid:%d tag=%d\n", myrank, lrts_received_msg, msg_tag);
#endif
            /* copy msg out and then put into queue (small message) */
            switch (msg_tag) {
            case SMALL_DATA_TAG:
            {
                msg_nbytes = CmiGetMsgSize(header);
                msg_data    = CmiAlloc(msg_nbytes);
                memcpy(msg_data, (char*)header, msg_nbytes);
                handleOneRecvedMsg(msg_nbytes, msg_data);
                break;
            }
            case LMSG_INIT_TAG:
            {
#if PRINT_SYH
                CmiPrintf("+++[%d] from %d PumpNetwork Rdma Request msg is received, messageid:%d tag=%d\n", myrank, inst_id, lrts_received_msg, msg_tag);
#endif
                getLargeMsgRequest(header, inst_id);
                break;
            }
            case ACK_TAG:
            {
                /* Get is done, release message . Now put is not used yet*/
#if         !USE_LRTS_MEMPOOL
                MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &(((CONTROL_MSG *)header)->source_mem_hndl), &omdh);
#endif
                CmiFree((void*)((CONTROL_MSG *) header)->source_addr);
                SendRdmaMsg();
                break;
            }
            default: {
                CmiPrintf("weird tag problem\n");
                CmiAbort("Unknown tag\n");
            }
            }
            GNI_SmsgRelease(ep_hndl_array[inst_id]);
            msg_tag = GNI_SMSG_ANY_TAG;
        } //endwhile getNext
    }   //end while GetEvent
    if(status == GNI_RC_ERROR_RESOURCE)
    {
        GNI_RC_CHECK("Smsg_rx_cq full", status);
    }
}

static void getLargeMsgRequest(void* header, uint64_t inst_id)
{
#if     USE_LRTS_MEMPOOL
    CONTROL_MSG         *request_msg;
    gni_return_t        status;
    void                *msg_data;
    gni_post_descriptor_t *pd;
    RDMA_REQUEST        *rdma_request_msg;
    gni_mem_handle_t    msg_mem_hndl;
    int source;
    // initial a get to transfer data from the sender side */
    request_msg = (CONTROL_MSG *) header;
    source = request_msg->source;
    msg_data = CmiAlloc(request_msg->length);
    _MEMCHECK(msg_data);
    //memcpy(&msg_mem_hndl, GetMemHndl(msg_data), sizeof(gni_mem_handle_t));
    msg_mem_hndl = GetMemHndl(msg_data);

    MallocPostDesc(pd);
    if(request_msg->length < LRTS_GNI_RDMA_THRESHOLD) 
        pd->type            = GNI_POST_FMA_GET;
    else
        pd->type            = GNI_POST_RDMA_GET;

    pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT |  GNI_CQMODE_REMOTE_EVENT;
    pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
    pd->length          = ALIGN4(request_msg->length);
    pd->local_addr      = (uint64_t) msg_data;
    pd->local_mem_hndl  = msg_mem_hndl;
    pd->remote_addr     = request_msg->source_addr;
    pd->remote_mem_hndl = request_msg->source_mem_hndl;
    pd->src_cq_hndl     = 0;//post_tx_cqh;     /* smsg_tx_cqh;  */
    pd->rdma_mode       = 0;

    if(pd->type == GNI_POST_RDMA_GET) 
        status = GNI_PostRdma(ep_hndl_array[source], pd);
    else
        status = GNI_PostFma(ep_hndl_array[source],  pd);
    if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
    {
        MallocRdmaRequest(rdma_request_msg);
        rdma_request_msg->next = 0;
        rdma_request_msg->destNode = inst_id;
        rdma_request_msg->pd = pd;
        if(pending_rdma_head == 0)
        {
            pending_rdma_head = rdma_request_msg;
        }else
        {
            pending_rdma_tail->next = rdma_request_msg;
        }
        pending_rdma_tail = rdma_request_msg;
    }else
        GNI_RC_CHECK("AFter posting", status);

#else
    CONTROL_MSG         *request_msg;
    gni_return_t        status;
    void                *msg_data;
    gni_post_descriptor_t *pd;
    RDMA_REQUEST        *rdma_request_msg;
    gni_mem_handle_t    msg_mem_hndl;
    int source;
    // initial a get to transfer data from the sender side */
    request_msg = (CONTROL_MSG *) header;
    source = request_msg->source;
    msg_data = CmiAlloc(request_msg->length);
    _MEMCHECK(msg_data);

    status = MEMORY_REGISTER(onesided_hnd, nic_hndl, msg_data, request_msg->length, &msg_mem_hndl, &omdh);

    if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR) 
    {
        GNI_RC_CHECK("Mem Register before post", status);
    }

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
    pd->src_cq_hndl     = 0;//post_tx_cqh;     /* smsg_tx_cqh;  */
    pd->rdma_mode       = 0;

    //memory registration successful
    if(status == GNI_RC_SUCCESS)
    {
        pd->local_mem_hndl  = msg_mem_hndl;
        if(pd->type == GNI_POST_RDMA_GET) 
            status = GNI_PostRdma(ep_hndl_array[source], pd);
        else
            status = GNI_PostFma(ep_hndl_array[source],  pd);
    }else
    {
        pd->local_mem_hndl.qword1  = 0; 
        pd->local_mem_hndl.qword1  = 0; 
    }
    if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
    {
        MallocRdmaRequest(rdma_request_msg);
        rdma_request_msg->next = 0;
        rdma_request_msg->destNode = inst_id;
        rdma_request_msg->pd = pd;
        if(pending_rdma_head == 0)
        {
            pending_rdma_head = rdma_request_msg;
        }else
        {
            pending_rdma_tail->next = rdma_request_msg;
        }
        pending_rdma_tail = rdma_request_msg;
    }else
        GNI_RC_CHECK("AFter posting", status);
#endif
}

/* Check whether message send or get is confirmed by remote */
static void PumpLocalSmsgTransactions()
{
    gni_return_t            status;
    gni_cq_entry_t          ev;
    while ((status = GNI_CqGetEvent(smsg_tx_cqh, &ev)) == GNI_RC_SUCCESS)
    {
#if PRINT_SYH
        lrts_local_done_msg++;
        //CmiPrintf("*[%d]  PumpLocalSmsgTransactions GNI_CQ_GET_TYPE %d. Localdone=%d\n", myrank, GNI_CQ_GET_TYPE(ev), lrts_local_done_msg);
#endif
        if(GNI_CQ_OVERRUN(ev))
        {
            CmiPrintf("Overrun detected in local CQ");
            CmiAbort("Overrun in TX");
        }
    }
    if(status == GNI_RC_ERROR_RESOURCE)
    {
        GNI_RC_CHECK("Smsg_tx_cq full", status);
    }
}

static void PumpLocalRdmaTransactions()
{
    gni_cq_entry_t          ev;
    gni_return_t            status;
    uint64_t                type, inst_id;
    gni_post_descriptor_t   *tmp_pd;
    MSG_LIST                *ptr;
    CONTROL_MSG             *ack_msg_tmp;

    while ( (status = GNI_CqGetEvent(smsg_tx_cqh, &ev)) == GNI_RC_SUCCESS) 
    {
        type        = GNI_CQ_GET_TYPE(ev);
#if PRINT_SYH
        lrts_local_done_msg++;
        CmiPrintf("**[%d] SMSGPumpLocalTransactions localdone=%d (type=%d)\n", myrank,  lrts_local_done_msg, type);
#endif
        /*if(type == GNI_CQ_EVENT_TYPE_SMSG)
        {
        }else */
        if (type == GNI_CQ_EVENT_TYPE_POST)
        {
            inst_id     = GNI_CQ_GET_INST_ID(ev);
#if PRINT_SYH
            CmiPrintf("**[%d] SMSGPumpLocalTransactions localdone=%d\n", myrank,  lrts_local_done_msg);
#endif
            status = GNI_GetCompleted(smsg_tx_cqh, ev, &tmp_pd);
            //CmiPrintf("**[%d] SMSGPumpLocalTransactions local done(type=%d) length=%d, size=%d\n", myrank, type, tmp_pd->length, SIZEFIELD((void*)(tmp_pd->local_addr)) );
            ////Message is sent, free message , put is not used now
            if(tmp_pd->type == GNI_POST_RDMA_PUT || tmp_pd->type == GNI_POST_FMA_PUT)
            {
                CmiFree((void *)tmp_pd->local_addr);
            }else if(tmp_pd->type == GNI_POST_RDMA_GET || tmp_pd->type == GNI_POST_FMA_GET)
            {
                /* Send an ACK to remote side */
                MallocControlMsg(ack_msg_tmp);
                ack_msg_tmp->source             = myrank;
                ack_msg_tmp->source_addr        = tmp_pd->remote_addr;
                ack_msg_tmp->length             = tmp_pd->length; 
                ack_msg_tmp->source_mem_hndl    = tmp_pd->remote_mem_hndl;
#if PRINT_SYH
                lrts_send_msg_id++;
                CmiPrintf("ACK LrtsSend PE:%d==>%d, size=%d, messageid:%d ACK\n", myrank, inst_id, sizeof(CONTROL_MSG), lrts_send_msg_id);
#endif
                if(smsg_msglist_head[inst_id]!=0)
                {
                    delay_send_small_msg(ack_msg_tmp, sizeof(CONTROL_MSG), inst_id, ACK_TAG);
                }else
                {
                    status = GNI_SmsgSendWTag(ep_hndl_array[inst_id], 0, 0, ack_msg_tmp, sizeof(CONTROL_MSG), 0, ACK_TAG);
                    if(status == GNI_RC_SUCCESS)
                    {
#if PRINT_SYH
                        lrts_smsg_success++;
                        CmiPrintf("[%d==>%d] sent ACK done%d\n", myrank, inst_id, lrts_smsg_success);
#endif
                        FreeControlMsg(ack_msg_tmp);
                    }else if(status == GNI_RC_NOT_DONE || status == GNI_RC_ERROR_RESOURCE)
                    {
                        delay_send_small_msg(ack_msg_tmp, sizeof(CONTROL_MSG), inst_id, ACK_TAG);
                    }
                    else
                        GNI_RC_CHECK("GNI_SmsgSendWTag", status);
                }
#if     !USE_LRTS_MEMPOOL
                MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &tmp_pd->local_mem_hndl, &omdh);
#endif
                CmiAssert(SIZEFIELD((void*)(tmp_pd->local_addr)) <= tmp_pd->length);
                //handleOneRecvedMsg(SIZEFIELD((void*)(tmp_pd->local_addr)), (void*)tmp_pd->local_addr); 
                handleOneRecvedMsg(tmp_pd->length, (void*)tmp_pd->local_addr); 
                SendRdmaMsg(); 
            }
            FreePostDesc(tmp_pd);
        }
    }
}

static int SendRdmaMsg()
{
    gni_return_t            status = GNI_RC_SUCCESS;
    gni_mem_handle_t        msg_mem_hndl;

    RDMA_REQUEST *ptr = pending_rdma_head;
    RDMA_REQUEST *prev = NULL;

    while (ptr != NULL)
    {
        gni_post_descriptor_t *pd = ptr->pd;
        // register memory first
        if( pd->local_mem_hndl.qword1 == 0 && pd->local_mem_hndl.qword2 == 0)
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
                RDMA_REQUEST *tmp = ptr;
                if (prev)
                  prev->next = ptr->next;
                else
                  pending_rdma_head = ptr->next;
                ptr = ptr->next;
                FreeRdmaRequest(tmp);
                continue;
            }
        }
        prev = ptr;
        ptr = ptr->next;
    } //end while
    return 0;
}

// return 1 if all messages are sent
static int SendBufferMsg()
{
    MSG_LIST            *ptr;
    CONTROL_MSG         *control_msg_tmp;
    gni_return_t        status;
    int done = 1;
    register    int     i;
    //if( smsg_msglist_head == 0 && buffered_smsg_counter!= 0 ) {CmiPrintf("WRONGWRONG on rank%d, buffermsg=%d, (msgid-succ:%d)\n", myrank, buffered_smsg_counter, (lrts_send_msg_id-lrts_smsg_success)); CmiAbort("sendbuf");}
    /* can add flow control here to control the number of messages sent before handle message */
    for(i=0; i<mysize; i++)
    {
        if(i==myrank || smsg_msglist_head[i]==0) continue;
       
        while(smsg_msglist_head[i]!=0)
        {
        if(useStaticSMSG)
        {
            ptr = smsg_msglist_head[i];
            CmiAssert(ptr!=NULL);
            if(ptr->tag == SMALL_DATA_TAG)
            {
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], NULL, 0, ptr->msg, ptr->size, 0, SMALL_DATA_TAG);
               // CmiPrintf("[%d==>%d] buffer send call(%d)%s\n", myrank, ptr->destNode, lrts_smsg_success, gni_err_str[status] );
                if(status == GNI_RC_SUCCESS) {
#if PRINT_SYH
                    lrts_smsg_success++;
                    if(lrts_smsg_success == lrts_send_msg_id)
                        CmiPrintf("GOOD [%d==>%d] sent done%d (msgs=%d)\n", myrank, ptr->destNode, lrts_smsg_success, lrts_send_msg_id);
                    else
                        CmiPrintf("BAD [%d==>%d] sent done%d (msgs=%d)\n", myrank, ptr->destNode, lrts_smsg_success, lrts_send_msg_id);
#endif
                    CmiFree(ptr->msg);
                }
            }
            else if(ptr->tag == LMSG_INIT_TAG)
            {
                control_msg_tmp = (CONTROL_MSG*)ptr->msg;
#if PRINT_SYH
                CmiPrintf("[%d==>%d] LMSG buffer send call(%d)%s\n", myrank, ptr->destNode, lrts_smsg_success, gni_err_str[status] );
#endif
                if(control_msg_tmp->source_mem_hndl.qword1 == 0 && control_msg_tmp->source_mem_hndl.qword2 == 0)
                {
                    MEMORY_REGISTER(onesided_hnd, nic_hndl, control_msg_tmp->source_addr, control_msg_tmp->length, &(control_msg_tmp->source_mem_hndl), &omdh);
                    if(status != GNI_RC_SUCCESS) {
                        done = 0;
                        break;
                    }
                }
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], 0, 0, ptr->msg, sizeof(CONTROL_MSG), 0, LMSG_INIT_TAG);
                if(status == GNI_RC_SUCCESS) {
#if PRINT_SYH
                    lrts_smsg_success++;
                    CmiPrintf("[%d==>%d] sent LMSG done%d\n", myrank, ptr->destNode, lrts_smsg_success);
#endif
                    FreeControlMsg((CONTROL_MSG*)(ptr->msg));
                }
            }else if (ptr->tag == ACK_TAG)
            {
#if PRINT_SYH
                CmiPrintf("[%d==>%d] ACK buffer send call(%d)%s\n", myrank, ptr->destNode, lrts_smsg_success, gni_err_str[status] );
#endif
                status = GNI_SmsgSendWTag(ep_hndl_array[ptr->destNode], 0, 0, ptr->msg, sizeof(CONTROL_MSG), 0, ACK_TAG);
                if(status == GNI_RC_SUCCESS) {
#if PRINT_SYH
                    lrts_smsg_success++;
                    if(lrts_smsg_success == lrts_send_msg_id)
                        CmiPrintf("GOOD [%d==>%d] sent done%d (msgs=%d)\n", myrank, ptr->destNode, lrts_smsg_success, lrts_send_msg_id);
                    else
                        CmiPrintf("BAD [%d==>%d] sent done%d (msgs=%d)\n", myrank, ptr->destNode, lrts_smsg_success, lrts_send_msg_id);
#endif
                    FreeControlMsg((CONTROL_MSG*)ptr->msg);
                }
            }else
            {
                CmiPrintf("Weird tag\n");
                CmiAbort("should not happen\n");
            }
        } 
        if(status == GNI_RC_SUCCESS)
        {
            smsg_msglist_head[i] = smsg_msglist_head[i]->next;
            FreeMsgList(ptr);
#if PRINT_SYH
            buffered_smsg_counter--;
            if(lrts_smsg_success == lrts_send_msg_id)
                CmiPrintf("GOOD send buff [%d==>%d] send buffer sent done%d (msgs=%d)\n", myrank, ptr->destNode, lrts_smsg_success, lrts_send_msg_id);
            else
                CmiPrintf("BAD send buff [%d==>%d] sent done%d (msgs=%d)\n", myrank, ptr->destNode, lrts_smsg_success, lrts_send_msg_id);
#endif
        }else {
            done = 0;
            break;
        }
        } //end pooling this i-th core
    }   // end pooling for all cores
#if PRINT_SYH
    if(lrts_send_msg_id-lrts_smsg_success !=0)
        CmiPrintf("WRONG [%d buffered msg is empty(%p) (actually=%d)(buffercounter=%d)\n", myrank, smsg_msglist_head[i], (lrts_send_msg_id-lrts_smsg_success, buffered_smsg_counter));
#endif
    return done;
}

static void LrtsAdvanceCommunication()
{
    /*  Receive Msg first */
    //CmiPrintf("Calling Lrts Pump Msg PE:%d\n", CmiMyPe());
    PumpNetworkSmsg();
    //CmiPrintf("Calling Lrts Pump RdmaMsg PE:%d\n", CmiMyPe());
    //PumpNetworkRdmaMsgs();
    /* Release Sent Msg */
    //CmiPrintf("Calling Lrts Rlease Msg PE:%d\n", CmiMyPe());
    //PumpLocalSmsgTransactions();
    //CmiPrintf("Calling Lrts Rlease RdmaMsg PE:%d\n", CmiMyPe());
    PumpLocalRdmaTransactions();
    //CmiPrintf("Calling Lrts Send Buffmsg PE:%d\n", CmiMyPe());
    /* Send buffered Message */
    SendBufferMsg();
    SendRdmaMsg();
}

static void _init_static_smsg()
{
    gni_smsg_attr_t      *smsg_attr;
    gni_smsg_attr_t      remote_smsg_attr;
    gni_smsg_attr_t      *smsg_attr_vec;
    unsigned int         smsg_memlen;
    gni_mem_handle_t     my_smsg_mdh_mailbox;
    int      i;
    gni_return_t status;
    uint32_t              vmdh_index = -1;
    mdh_addr_t            base_infor;
    mdh_addr_t            *base_addr_vec;
    if(mysize <=1024)
    {
        SMSG_MAX_MSG = 1024;
        //log2_SMSG_MAX_MSG = 10;
    }else if (mysize > 1024 && mysize <= 16384)
    {
        SMSG_MAX_MSG = 512;
        //log2_SMSG_MAX_MSG = 9;

    }else {
        SMSG_MAX_MSG = 256;
        //log2_SMSG_MAX_MSG = 8;
    }
    
    smsg_attr = malloc(mysize * sizeof(gni_smsg_attr_t));
    
    smsg_attr[0].msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
    smsg_attr[0].mbox_maxcredit = SMSG_MAX_CREDIT;
    smsg_attr[0].msg_maxsize = SMSG_MAX_MSG;
    status = GNI_SmsgBufferSizeNeeded(&smsg_attr[0], &smsg_memlen);
    GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);
    smsg_mailbox_base = memalign(64, smsg_memlen*(mysize));
    _MEMCHECK(smsg_mailbox_base);
    bzero(smsg_mailbox_base, smsg_memlen*(mysize));
    
    status = GNI_MemRegister(nic_hndl, (uint64_t)smsg_mailbox_base,
            smsg_memlen*(mysize), smsg_rx_cqh,
            GNI_MEM_READWRITE,   
            vmdh_index,
            &my_smsg_mdh_mailbox);

    GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);

#if 1
    base_infor.addr =  (uint64_t)smsg_mailbox_base;
    base_infor.mdh =  my_smsg_mdh_mailbox;
    base_addr_vec = malloc(mysize * sizeof(mdh_addr_t));

    allgather(&base_infor, base_addr_vec,  sizeof(mdh_addr_t));
 
    for(i=0; i<mysize; i++)
    {
        if(i==myrank)
            continue;
        smsg_attr[i].msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
        smsg_attr[i].mbox_maxcredit = SMSG_MAX_CREDIT;
        smsg_attr[i].msg_maxsize = SMSG_MAX_MSG;
        smsg_attr[i].mbox_offset = i*smsg_memlen;
        smsg_attr[i].buff_size = smsg_memlen;
        smsg_attr[i].msg_buffer = smsg_mailbox_base ;
        smsg_attr[i].mem_hndl = my_smsg_mdh_mailbox;
    }

    for(i=0; i<mysize; i++)
    {
        if (myrank == i) continue;

        remote_smsg_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
        remote_smsg_attr.mbox_maxcredit = SMSG_MAX_CREDIT;
        remote_smsg_attr.msg_maxsize = SMSG_MAX_MSG;
        remote_smsg_attr.mbox_offset = myrank*smsg_memlen;
        remote_smsg_attr.buff_size = smsg_memlen;
        remote_smsg_attr.msg_buffer = (void*)base_addr_vec[i].addr;
        remote_smsg_attr.mem_hndl = base_addr_vec[i].mdh;

        /* initialize the smsg channel */
        status = GNI_SmsgInit(ep_hndl_array[i], &smsg_attr[i], &remote_smsg_attr);
        GNI_RC_CHECK("SMSG Init", status);
    } //end initialization

    free(base_addr_vec);
#else
    for(i=0; i<mysize; i++)
    {
        if(i==myrank)
            continue;
        smsg_attr[i].msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
        smsg_attr[i].mbox_maxcredit = SMSG_MAX_CREDIT;
        smsg_attr[i].msg_maxsize = SMSG_MAX_MSG;
        if(i<myrank)
            smsg_attr[i].mbox_offset = i*smsg_memlen;
        else
            smsg_attr[i].mbox_offset = (i-1)*smsg_memlen;

        smsg_attr[i].msg_buffer = smsg_mailbox_base;
        smsg_attr[i].buff_size = smsg_memlen;
        smsg_attr[i].mem_hndl = my_smsg_mdh_mailbox;
#if 0 
        if(i<2)
        {
            CmiPrintf("[assign %d==%d]  offset=%d buf=%p,credit=%d size=%d, bufsize=%d\n", myrank, i, (smsg_attr[i]).mbox_offset, (smsg_attr[i]).msg_buffer, (smsg_attr[i]).mbox_maxcredit, (smsg_attr[i]).msg_maxsize, (smsg_attr[i]).buff_size ); 
#endif 
        }
    }
    smsg_attr_vec = (gni_smsg_attr_t*)malloc(mysize * mysize * sizeof(gni_smsg_attr_t));
    _MEMCHECK(smsg_attr_vec);
 
    if(myrank==0)
        CmiPrintf("total size=%d\n", mysize*mysize*sizeof(gni_smsg_attr_t));
    allgather(smsg_attr, smsg_attr_vec,  mysize*sizeof(gni_smsg_attr_t));
    
    CmiBarrier();
    for(i=0; i<mysize; i++)
    {
        if (myrank == i) continue;
        /* initialize the smsg channel */
        status = GNI_SmsgInit(ep_hndl_array[i], &smsg_attr[i], &smsg_attr_vec[i*mysize+myrank]);
#if 0 
        if(i<2)
        {
            CmiPrintf("[%d==%d] SmsgInit rc=%s, offset=%d buf=%p,credit=%d size=%d, bufsize=%d\n", myrank, i, gni_err_str[status], (smsg_attr_vec[i*mysize+myrank]).mbox_offset, (smsg_attr_vec[i*mysize+myrank]).msg_buffer, (smsg_attr_vec[i*mysize+myrank]).mbox_maxcredit, (smsg_attr_vec[i*mysize+myrank]).msg_maxsize, (smsg_attr_vec[i*mysize+myrank]).buff_size ); 
        }
#endif
        GNI_RC_CHECK("SMSG Init", status);
    } //end initialization
    CmiBarrier();
    free(smsg_attr_vec);
#endif
    free(smsg_attr);
    status = GNI_SmsgSetMaxRetrans(nic_hndl, 4096);
    GNI_RC_CHECK("SmsgSetMaxRetrans Init", status);
    smsg_msglist_head = (MSG_LIST**) malloc(mysize*sizeof(MSG_LIST*));
    memset(smsg_msglist_head, 0, mysize*sizeof(MSG_LIST*));
    smsg_msglist_tail = (MSG_LIST**) malloc(mysize*sizeof(MSG_LIST*));
    memset(smsg_msglist_tail, 0, mysize*sizeof(MSG_LIST*));
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

static void _init_DMA_buffer()
{
    gni_return_t            status = GNI_RC_SUCCESS;
    /*AUTO tuning */
    /* suppose max_smsg is 1024, DMA buffer is split into 2048, 4096, 8192, ... */
    /*  This method might be better for SMP, but it is bad for Nonsmp since msgs are sharing same slots */
    /*
     * DMA_slots = 19-log2_SMSG_MAX_MSG;
    DMA_incoming_avail_tag = malloc(DMA_slots);
    DMA_buffer_size = 2*(DMA_max_single_msg - SMSG_MAX_MSG); 
    DMA_incoming_base_addr =  memalign(ALIGNBUF, DMA_buffer_size);
    DMA_outgoing_base_addr =  memalign(ALIGNBUF, DMA_buffer_size);
    
    status = GNI_MemRegister(nic_hndl, (uint64_t)DMA_incoming_base_addr,
            DMA_buffer_size, smsg_rx_cqh,
            GNI_MEM_READWRITE ,   
            vmdh_index,
            &);
            */
    // one is reserved to avoid deadlock
    DMA_slots           = 17; // each one is 8K  16*8K + 1 slot reserved to avoid deadlock
    DMA_buffer_size     = DMA_max_single_msg + 8192;
    DMA_buffer_base_mdh_addr.addr = (uint64_t)memalign(ALIGNBUF, DMA_buffer_size);
    status = GNI_MemRegister(nic_hndl, DMA_buffer_base_mdh_addr.addr,
        DMA_buffer_size, smsg_rx_cqh,
        GNI_MEM_READWRITE ,   
        -1,
        &(DMA_buffer_base_mdh_addr.mdh));
    GNI_RC_CHECK("GNI_MemRegister", status);
    DMA_buffer_base_mdh_addr_vec = (mdh_addr_t*) malloc(sizeof(mdh_addr_t) * mysize);

    allgather(&DMA_buffer_base_mdh_addr, DMA_buffer_base_mdh_addr_vec, sizeof(mdh_addr_t) );
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
   
    //Mempool_MaxSize = CmiGetArgFlag(*argv, "+useMemorypoolSize");
    //useStaticSMSG = CmiGetArgFlag(*argv, "+useStaticSmsg");
    //useStaticMSGQ = CmiGetArgFlag(*argv, "+useStaticMsgQ");
    
    status = PMI_Init(&first_spawned);
    GNI_RC_CHECK("PMI_Init", status);

    status = PMI_Get_size(&mysize);
    GNI_RC_CHECK("PMI_Getsize", status);

    status = PMI_Get_rank(&myrank);
    GNI_RC_CHECK("PMI_getrank", status);

    //physicalID = CmiPhysicalNodeID(myrank);
    
    //printf("Pysical Node ID:%d for PE:%d\n", physicalID, myrank);

    *myNodeID = myrank;
    *numNodes = mysize;
  
    if(myrank == 0)
    {
        printf("Charm++> Running on Gemini (GNI) using %d  cores\n", mysize);
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
    status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &smsg_tx_cqh);
    GNI_RC_CHECK("GNI_CqCreate (tx)", status);
    
    //status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &post_tx_cqh);
    //GNI_RC_CHECK("GNI_CqCreate post (tx)", status);
    /* create the destination completion queue for receiving micro-messages, make this queue considerably larger than the number of transfers */

    status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &smsg_rx_cqh);
    GNI_RC_CHECK("Create CQ (rx)", status);
    
    //status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &post_rx_cqh);
    //GNI_RC_CHECK("Create Post CQ (rx)", status);
    
    //status = GNI_CqCreate(nic_hndl, REMOTE_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &rdma_cqh);
    //GNI_RC_CHECK("Create BTE CQ", status);

    /* create the endpoints. they need to be bound to allow later CQWrites to them */
    ep_hndl_array = (gni_ep_handle_t*)malloc(mysize * sizeof(gni_ep_handle_t));
    _MEMCHECK(ep_hndl_array);

    for (i=0; i<mysize; i++) {
        if(i == myrank) continue;
        status = GNI_EpCreate(nic_hndl, smsg_tx_cqh, &ep_hndl_array[i]);
        GNI_RC_CHECK("GNI_EpCreate ", status);   
        remote_addr = MPID_UGNI_AllAddr[i];
        status = GNI_EpBind(ep_hndl_array[i], remote_addr, i);
        GNI_RC_CHECK("GNI_EpBind ", status);   
    }
    /* Depending on the number of cores in the job, decide different method */
    /* SMSG is fastest but not scale; Msgq is scalable, FMA is own implementation for small message */
    if(mysize > 1)
    {
        if(useStaticSMSG == 1)
        {
            _init_static_smsg(mysize);
        }else if(useStaticMSGQ == 1)
        {
            _init_static_msgq();
        }
    }
#if     USE_LRTS_MEMPOOL
    init_mempool(_mempool_size);
    //init_mempool(Mempool_MaxSize);
#endif

    /* init DMA buffer for medium message */

    //_init_DMA_buffer();
    
    free(MPID_UGNI_AllAddr);
}


void* LrtsAlloc(int n_bytes, int header)
{
    void *ptr;
#if 0
    CmiPrintf("\n[PE:%d]Alloc Lrts for bytes=%d, head=%d\n", CmiMyPe(), n_bytes, header);
#endif
    if(n_bytes <= SMSG_MAX_MSG)
    {
        int totalsize = n_bytes+header;
        ptr = malloc(totalsize);
    }else 
    {

        CmiAssert(header <= ALIGNBUF);
#if     USE_LRTS_MEMPOOL
        n_bytes = ALIGN64(n_bytes);
        char *res = syh_mempool_malloc(ALIGNBUF+n_bytes);
#else
        n_bytes = ALIGN4(n_bytes);           /* make sure size if 4 aligned */
        char *res = memalign(ALIGNBUF, n_bytes+ALIGNBUF);
#endif
        ptr = res + ALIGNBUF - header;
    }
#if 0 
    CmiPrintf("Done Alloc Lrts for bytes=%d, head=%d\n", n_bytes, header);
#endif
    return ptr;
}

void  LrtsFree(void *msg)
{
    int size = SIZEFIELD((char*)msg+sizeof(CmiChunkHeader));
    if (size <= SMSG_MAX_MSG)
      free(msg);
    else
    {
#if 0
        CmiPrintf("[PE:%d] Free lrts for bytes=%d, ptr=%p\n", CmiMyPe(), size, (char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);
#endif
#if     USE_LRTS_MEMPOOL
        syh_mempool_free((char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);
#else
        free((char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);
#endif
    }
#if 0 
    CmiPrintf("Done Free lrts for bytes=%d\n", size);
#endif
}

static void LrtsExit()
{
    /* free memory ? */
#if     USE_LRTS_MEMPOOL
    kill_allmempool();
#endif
    PMI_Finalize();
    exit(0);
}

static void LrtsDrainResources()
{
    while (!SendBufferMsg()) {
        PumpNetworkSmsg();
        PumpNetworkRdmaMsgs();
        PumpLocalSmsgTransactions();
        PumpLocalRdmaTransactions();
    }
    PMI_Barrier();
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
static int _is_global = 0;
static struct timespec start_ns;

inline int CmiTimerIsSynchronized() {
    return 1;
}

inline int CmiTimerAbsolute() {
    return _absoluteTime;
}

double CmiStartTimer() {
    return 0.0;
}

double CmiInitTime() {
    return (double)(start_ts.tv_sec)+(double)start_ts.tv_nsec/1000000000.0;
}

void CmiTimerInit(char **argv) {
    _absoluteTime = CmiGetArgFlagDesc(argv,"+useAbsoluteTime", "Use system's absolute time as wallclock time.");
    if (_absoluteTime && CmiMyPe() == 0)
        printf("Charm++> absolute  timer is used\n");
    
    _is_global = CmiTimerIsSynchronized();


    if (_is_global) {
        if (CmiMyRank() == 0) {
            clock_gettime(CLOCK_MONOTONIC, &start_ts)
        }
    } else { /* we don't have a synchronous timer, set our own start time */
        CmiBarrier();
        CmiBarrier();
        CmiBarrier();
        clock_gettime(CLOCK_MONOTONIC, &start_ts)
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
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts)
    return _absoluteTime?((double)(now_ts.tv_sec)+(double)now_ts.tv_nsec/1000000000.0)
        : (double)( now_ts.tv_sec - start_ts.tv_sec ) + (((double) now_ts.tv_nsec - (double) start_ts.tv_nsec)  / 1000000000.0);
}

double CmiWallTimer(void) {
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts)
    return _absoluteTime?((double)(now_ts.tv_sec)+(double)now_ts.tv_nsec/1000000000.0)
        : (double)( now_ts.tv_sec - start_ts.tv_sec ) + (((double) now_ts.tv_nsec - (double) start_ts.tv_nsec)  / 1000000000.0);
}

double CmiCpuTimer(void) {
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts)
    return _absoluteTime?((double)(now_ts.tv_sec)+(double)now_ts.tv_nsec/1000000000.0)
        : (double)( now_ts.tv_sec - start_ts.tv_sec ) + (((double) now_ts.tv_nsec - (double) start_ts.tv_nsec)  / 1000000000.0);
}

#endif
/************Barrier Related Functions****************/

int CmiBarrier()
{
    int status;
    status = PMI_Barrier();
    return status;

}



