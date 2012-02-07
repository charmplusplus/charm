
/** @file
 * Gemini GNI machine layer
 *
 * Author:   Yanhua Sun
             Gengbin Zheng
 * Date:   07-01-2011
 *
 */
/*@{*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <malloc.h>
#include <unistd.h>

#include "gni_pub.h"
#include "pmi.h"

#include "converse.h"
#include <time.h>

#define PRINT_SYH  0

// Trace communication thread
#if CMK_TRACE_ENABLED && CMK_SMP_TRACE_COMMTHREAD
#define TRACE_THRESHOLD     0.00005
#define CMI_MPI_TRACE_MOREDETAILED 0
#undef CMI_MPI_TRACE_USEREVENTS
#define CMI_MPI_TRACE_USEREVENTS 1
#else
#undef CMK_SMP_TRACE_COMMTHREAD
#define CMK_SMP_TRACE_COMMTHREAD 0
#endif

#define CMK_TRACE_COMMOVERHEAD 0
#if CMK_TRACE_ENABLED && CMK_TRACE_COMMOVERHEAD
#undef CMI_MPI_TRACE_USEREVENTS
#define CMI_MPI_TRACE_USEREVENTS 1
#else
#undef CMK_TRACE_COMMOVERHEAD
#define CMK_TRACE_COMMOVERHEAD 0
#endif

#if CMI_MPI_TRACE_USEREVENTS && CMK_TRACE_ENABLED && ! CMK_TRACE_IN_CHARM
CpvStaticDeclare(double, projTraceStart);
#define  START_EVENT()  CpvAccess(projTraceStart) = CmiWallTimer();
#define  END_EVENT(x)   traceUserBracketEvent(x, CpvAccess(projTraceStart), CmiWallTimer());
#else
#define  START_EVENT()
#define  END_EVENT(x)
#endif

#define CMI_EXERT_SEND_CAP	0
#define	CMI_EXERT_RECV_CAP	0

#if CMI_EXERT_SEND_CAP
#define SEND_CAP 16
#endif

#if CMI_EXERT_RECV_CAP
#define RECV_CAP 2
#endif



#define             SMSG_ATTR_SIZE      sizeof(gni_smsg_attr_t)

static int useDynamicSMSG  =0;               /* dynamic smsgs setup */

static int avg_smsg_connection = 32;
static int                 *smsg_connected_flag= 0;
static gni_smsg_attr_t     **smsg_attr_vector_local;
static gni_smsg_attr_t     **smsg_attr_vector_remote;
static gni_ep_handle_t     ep_hndl_unbound;
static gni_smsg_attr_t     send_smsg_attr;
static gni_smsg_attr_t     recv_smsg_attr;

typedef struct _dynamic_smsg_mailbox{
   void     *mailbox_base;
   int      size;
   int      offset;
   gni_mem_handle_t  mem_hndl;
   struct      _dynamic_smsg_mailbox  *next;
}dynamic_smsg_mailbox_t;

static dynamic_smsg_mailbox_t  *mailbox_list;

#define REMOTE_EVENT                      0
#define USE_LRTS_MEMPOOL                  1

#if USE_LRTS_MEMPOOL
#if CMK_SMP
#define STEAL_MEMPOOL                     0
#endif

#define oneMB (1024ll*1024)
#if CMK_SMP
static CmiInt8 _mempool_size = 8*oneMB;
#else
static CmiInt8 _mempool_size = 32*oneMB;
#endif
static CmiInt8 _expand_mem =  4*oneMB;
#endif

//Dynamic flow control about memory registration
#define  MAX_BUFF_SEND      512*oneMB
static CmiInt8 buffered_send_msg = 0;

#define BIG_MSG                  16*oneMB
#define ONE_SEG                  4*oneMB
#define BIG_MSG_PIPELINE         4

#if CMK_SMP
#define COMM_THREAD_SEND 1
#endif
int         rdma_id = 0;
#if PRINT_SYH
int         lrts_smsg_success = 0;
int         lrts_send_msg_id = 0;
int         lrts_send_rdma_success = 0;
int         lrts_received_msg = 0;
int         lrts_local_done_msg = 0;
#endif

#include "machine.h"

#include "pcqueue.h"

#include "mempool.h"

#if CMK_PERSISTENT_COMM
#include "machine-persistent.h"
#endif

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
#if REMOTE_EVENT
#define  MEMORY_REGISTER(handler, nic_hndl, msg, size, mem_hndl, myomdh) GNI_MemRegister(nic_hndl, (uint64_t)msg,  (uint64_t)size, smsg_rx_cqh,  GNI_MEM_READWRITE, -1, mem_hndl)
#else
#define  MEMORY_REGISTER(handler, nic_hndl, msg, size, mem_hndl, myomdh) GNI_MemRegister(nic_hndl, (uint64_t)msg,  (uint64_t)size, NULL,  GNI_MEM_READWRITE, -1, mem_hndl)
#endif
#define  MEMORY_DEREGISTER(handler, nic_hndl, mem_hndl, myomdh)  GNI_MemDeregister(nic_hndl, (mem_hndl))
#endif

#define   IncreaseMsgInFlight(x) (((block_header*)(((mempool_header*)((char*)x-ALIGNBUF))->mempool_ptr))->msgs_in_flight)++
#define   DecreaseMsgInFlight(x)   (((block_header*)(((mempool_header*)((char*)x-ALIGNBUF))->mempool_ptr))->msgs_in_flight)-- 
#define   GetMempooladdr(x)  ((mempool_header*)((char*)x-ALIGNBUF))->mempool_ptr
#define   GetMempoolsize(x)  ((block_header*)(((mempool_header*)((char*)x-ALIGNBUF))->mempool_ptr))->size
#define   GetMemHndl(x)  ((block_header*)(((mempool_header*)((char*)x-ALIGNBUF))->mempool_ptr))->mem_hndl
#define   GetMemHndlFromHeader(x) ((block_header*)x)->mem_hndl
#define   NoMsgInFlight(x)  ((block_header*)(((mempool_header*)((char*)x-ALIGNBUF))->mempool_ptr))->msgs_in_flight == 0
#define   IsMemHndlZero(x)  (x.qword1 == 0 && x.qword2 == 0)
#define   SetMemHndlZero(x)  x.qword1 = 0; x.qword2 = 0
#define   NotRegistered(x)  IsMemHndlZero(((block_header*)x)->mem_hndl)

#define CmiGetMsgSize(m)  ((CmiMsgHeaderExt*)m)->size
#define CmiSetMsgSize(m,s)  ((((CmiMsgHeaderExt*)m)->size)=(s))
#define CmiGetMsgSeq(m)  ((CmiMsgHeaderExt*)m)->seq
#define CmiSetMsgSeq(m, s)  ((((CmiMsgHeaderExt*)m)->seq) = (s))

#define ALIGNBUF                64

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* If SMSG is not used */

#define FMA_PER_CORE  1024
#define FMA_BUFFER_SIZE 1024

/* If SMSG is used */
static int  SMSG_MAX_MSG = 1024;
#define SMSG_MAX_CREDIT 72 

#define MSGQ_MAXSIZE       2048
/* large message transfer with FMA or BTE */
#define LRTS_GNI_RDMA_THRESHOLD  1024 

#if CMK_SMP
static int  REMOTE_QUEUE_ENTRIES=163840; 
static int LOCAL_QUEUE_ENTRIES=163840; 
#else
static int  REMOTE_QUEUE_ENTRIES=20480;
static int LOCAL_QUEUE_ENTRIES=20480; 
#endif

#define BIG_MSG_TAG  0x26
#define PUT_DONE_TAG      0x28
#define DIRECT_PUT_DONE_TAG      0x29
#define ACK_TAG           0x30
/* SMSG is data message */
#define SMALL_DATA_TAG          0x31
/* SMSG is a control message to initialize a BTE */
#define MEDIUM_HEAD_TAG         0x32
#define MEDIUM_DATA_TAG         0x33
#define LMSG_INIT_TAG           0x39 
#define VERY_LMSG_INIT_TAG      0x40 
#define VERY_LMSG_TAG           0x41 

#define DEBUG
#ifdef GNI_RC_CHECK
#undef GNI_RC_CHECK
#endif
#ifdef DEBUG
#define GNI_RC_CHECK(msg,rc) do { if(rc != GNI_RC_SUCCESS) {           printf("[%d] %s; err=%s\n",CmiMyPe(),msg,gni_err_str[rc]); CmiAbort("GNI_RC_CHECK"); } } while(0)
#else
#define GNI_RC_CHECK(msg,rc)
#endif

#define ALIGN64(x)       (size_t)((~63)&((x)+63))
//#define ALIGN4(x)        (size_t)((~3)&((x)+3)) 

static int useStaticMSGQ = 0;
static int useStaticFMA = 0;
static int mysize, myrank;
gni_nic_handle_t      nic_hndl;

typedef struct {
    gni_mem_handle_t mdh;
    uint64_t addr;
} mdh_addr_t ;
// this is related to dynamic SMSG

typedef struct mdh_addr_list{
    gni_mem_handle_t mdh;
   void *addr;
    struct mdh_addr_list *next;
}mdh_addr_list_t;

static unsigned int         smsg_memlen;
gni_smsg_attr_t    **smsg_local_attr_vec = 0;
mdh_addr_t          setup_mem;
mdh_addr_t          *smsg_connection_vec = 0;
gni_mem_handle_t    smsg_connection_memhndl;
static int          smsg_expand_slots = 10;
static int          smsg_available_slot = 0;
static void         *smsg_mailbox_mempool = 0;
mdh_addr_list_t     *smsg_dynamic_list = 0;

static void             *smsg_mailbox_base;
gni_msgq_attr_t         msgq_attrs;
gni_msgq_handle_t       msgq_handle;
gni_msgq_ep_attr_t      msgq_ep_attrs;
gni_msgq_ep_attr_t      msgq_ep_attrs_size;

/* =====Beginning of Declarations of Machine Specific Variables===== */
static int cookie;
static int modes = 0;
static gni_cq_handle_t       smsg_rx_cqh = NULL;
static gni_cq_handle_t       smsg_tx_cqh = NULL;
static gni_cq_handle_t       post_rx_cqh = NULL;
static gni_cq_handle_t       post_tx_cqh = NULL;
static gni_ep_handle_t       *ep_hndl_array;
#if CMK_SMP && !COMM_THREAD_SEND
static CmiNodeLock           *ep_lock_array;
static CmiNodeLock           tx_cq_lock; 
static CmiNodeLock           rx_cq_lock; 
#endif
typedef struct msg_list
{
    uint32_t destNode;
    uint32_t size;
    void *msg;
    uint8_t tag;
#if !CMK_SMP
    struct msg_list *next;
#endif
}MSG_LIST;


typedef struct control_msg
{
    uint64_t            source_addr;    /* address from the start of buffer  */
    uint64_t            dest_addr;      /* address from the start of buffer */
    //int                 source;         /* source rank */
    int                 total_length;   /* total length */
    int                 length;         /* length of this packet */
    uint8_t             seq_id;         //big message   0 meaning single message
    gni_mem_handle_t    source_mem_hndl;
    struct control_msg *next;
}CONTROL_MSG;

#ifdef CMK_DIRECT
typedef struct{
    void *recverBuf;
    void (*callbackFnPtr)(void *);
    void *callbackData;
}CMK_DIRECT_HEADER;
#endif
typedef struct  rmda_msg
{
    int                   destNode;
    gni_post_descriptor_t *pd;
#if !CMK_SMP
    struct  rmda_msg      *next;
#endif
}RDMA_REQUEST;

#if CMK_SMP
PCQueue sendRdmaBuf;
typedef struct  msg_list_index
{
    int         next;
    PCQueue     sendSmsgBuf;
} MSG_LIST_INDEX;
#else
RDMA_REQUEST        *sendRdmaBuf;
RDMA_REQUEST        *sendRdmaTail;
typedef struct  msg_list_index
{
    int         next;
    MSG_LIST    *sendSmsgBuf;
    MSG_LIST    *tail;
} MSG_LIST_INDEX;
#endif
/* reuse PendingMsg memory */
static CONTROL_MSG          *control_freelist=0;
static MSG_LIST             *msglist_freelist=0;
static int                  smsg_head_index = -1;
static MSG_LIST_INDEX       *smsg_msglist_index= 0;
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

#if CMK_SMP

#define FreeMsgList(d)   free(d);
#define MallocMsgList(d)  d = ((MSG_LIST*)malloc(sizeof(MSG_LIST)));

#else

#define FreeMsgList(d)  \
  (d)->next = msglist_freelist;\
  msglist_freelist = d;

#define MallocMsgList(d) \
  d = msglist_freelist;\
  if (d==0) {d = ((MSG_LIST*)malloc(sizeof(MSG_LIST)));\
             _MEMCHECK(d);\
  } else msglist_freelist = d->next; \
  d->next =0;
#endif

#if CMK_SMP

#define FreeControlMsg(d)      free(d);
#define MallocControlMsg(d)    d = ((CONTROL_MSG*)malloc(sizeof(CONTROL_MSG)));

#else

#define FreeControlMsg(d)       \
  (d)->next = control_freelist;\
  control_freelist = d;

#define MallocControlMsg(d) \
  d = control_freelist;\
  if (d==0) {d = ((CONTROL_MSG*)malloc(sizeof(CONTROL_MSG)));\
             _MEMCHECK(d);\
  } else control_freelist = d->next;

#endif

static RDMA_REQUEST         *rdma_freelist = NULL;

#define FreeMediumControlMsg(d)       \
  (d)->next = medium_control_freelist;\
  medium_control_freelist = d;


#define MallocMediumControlMsg(d) \
    d = medium_control_freelist;\
    if (d==0) {d = ((MEDIUM_MSG_CONTROL*)malloc(sizeof(MEDIUM_MSG_CONTROL)));\
    _MEMCHECK(d);\
} else mediumcontrol_freelist = d->next;

# if CMK_SMP
#define FreeRdmaRequest(d)       free(d);
#define MallocRdmaRequest(d)     d = ((RDMA_REQUEST*)malloc(sizeof(RDMA_REQUEST)));   
#else

#define FreeRdmaRequest(d)       \
  (d)->next = rdma_freelist;\
  rdma_freelist = d;

#define MallocRdmaRequest(d) \
  d = rdma_freelist;\
  if (d==0) {d = ((RDMA_REQUEST*)malloc(sizeof(RDMA_REQUEST)));\
             _MEMCHECK(d);\
  } else rdma_freelist = d->next; \
    d->next =0;
#endif

/* reuse gni_post_descriptor_t */
static gni_post_descriptor_t *post_freelist=0;

#if !CMK_SMP
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


/* LrtsSent is called but message can not be sent by SMSGSend because of mailbox full or no credit */
static int      buffered_smsg_counter = 0;

/* SmsgSend return success but message sent is not confirmed by remote side */
static MSG_LIST *buffered_fma_head = 0;
static MSG_LIST *buffered_fma_tail = 0;

/* functions  */
#define IsFree(a,ind)  !( a& (1<<(ind) ))
#define SET_BITS(a,ind) a = ( a | (1<<(ind )) )
#define Reset(a,ind) a = ( a & (~(1<<(ind))) )

CpvDeclare(mempool_type*, mempool);

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

static void SendRdmaMsg();
static void PumpNetworkSmsg();
static void PumpLocalRdmaTransactions();
static int SendBufferMsg();


inline 
static gni_return_t registerMempool(void *msg)
{
    gni_return_t status;
    int size = GetMempoolsize(msg);
    void *addr = GetMempooladdr(msg);
    gni_mem_handle_t  *memhndl =   &(GetMemHndl(msg));
   
    mempool_type *mptr = CpvAccess(mempool);
    block_header *current = &(mptr->block_head);
    while(1)
    {
       status = MEMORY_REGISTER(onesided_hnd, nic_hndl, addr, size, memhndl, &omdh);
       //find one slot to de-register
       if(status == GNI_RC_SUCCESS)
       {
           break;
       }
       else if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR)
       {
                CmiAbort("Memory registor for mempool fails\n");
       }
        
       while( current!= NULL && (current->msgs_in_flight>0 || IsMemHndlZero(current->mem_hndl) ))
           current = current->block_next?(block_header *)((char*)mptr+current->block_next):NULL;
       
       if(current == NULL)
       { status = GNI_RC_ERROR_RESOURCE; break;}
       status = MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &(GetMemHndlFromHeader(current)) , &omdh);

       GNI_RC_CHECK("registerMemorypool de-register", status);
       SetMemHndlZero(GetMemHndlFromHeader(current));
    }; 
    return status;
}

inline
static void buffer_small_msgs(void *msg, int size, int destNode, uint8_t tag)
{
    MSG_LIST        *msg_tmp;
    MallocMsgList(msg_tmp);
    msg_tmp->destNode = destNode;
    msg_tmp->size   = size;
    msg_tmp->msg    = msg;
    msg_tmp->tag    = tag;
    //msg_tmp->next   = 0;

#if !CMK_SMP
    if (smsg_msglist_index[destNode].sendSmsgBuf == 0 ) {
        smsg_msglist_index[destNode].next = smsg_head_index;
        smsg_head_index = destNode;
        smsg_msglist_index[destNode].tail = smsg_msglist_index[destNode].sendSmsgBuf = msg_tmp;
    }else
    {
        smsg_msglist_index[destNode].tail->next = msg_tmp;
        smsg_msglist_index[destNode].tail = msg_tmp;
    }
#else
    PCQueuePush(smsg_msglist_index[destNode].sendSmsgBuf, (char*)msg_tmp);
#endif
#if PRINT_SYH
    buffered_smsg_counter++;
#endif
}

inline static void print_smsg_attr(gni_smsg_attr_t     *a)
{
    printf("type=%d\n, credit=%d\n, size=%d\n, buf=%p, offset=%d\n", a->msg_type, a->mbox_maxcredit, a->buff_size, a->msg_buffer, a->mbox_offset);
}

inline
static void setup_smsg_connection(int destNode)
{
    mdh_addr_list_t  *new_entry = 0;
    gni_post_descriptor_t *pd;
    gni_smsg_attr_t      *smsg_attr;
    gni_return_t status = GNI_RC_NOT_DONE;
    RDMA_REQUEST        *rdma_request_msg;
    
    if(smsg_available_slot == smsg_expand_slots)
    {
        new_entry = (mdh_addr_list_t*)malloc(sizeof(mdh_addr_list_t));
        new_entry->addr = memalign(64, smsg_memlen*smsg_expand_slots);
        bzero(new_entry->addr, smsg_memlen*smsg_expand_slots);

        status = GNI_MemRegister(nic_hndl, (uint64_t)new_entry->addr,
            smsg_memlen*smsg_expand_slots, smsg_rx_cqh,
            GNI_MEM_READWRITE,   
            -1,
            &(new_entry->mdh));
        smsg_available_slot = 0; 
        new_entry->next = smsg_dynamic_list;
        smsg_dynamic_list = new_entry;
    }
    smsg_attr = (gni_smsg_attr_t*) malloc (sizeof(gni_smsg_attr_t));
    smsg_attr->msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
    smsg_attr->mbox_maxcredit = SMSG_MAX_CREDIT;
    smsg_attr->msg_maxsize = SMSG_MAX_MSG;
    smsg_attr->mbox_offset = smsg_available_slot * smsg_memlen;
    smsg_attr->buff_size = smsg_memlen;
    smsg_attr->msg_buffer = smsg_dynamic_list->addr;
    smsg_attr->mem_hndl = smsg_dynamic_list->mdh;
    smsg_local_attr_vec[destNode] = smsg_attr;
    smsg_available_slot++;
    MallocPostDesc(pd);
    pd->type            = GNI_POST_FMA_PUT;
    pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT |  GNI_CQMODE_REMOTE_EVENT;
    pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
    pd->length          = sizeof(gni_smsg_attr_t);
    pd->local_addr      = (uint64_t) smsg_attr;
    pd->remote_addr     = (uint64_t)&((((gni_smsg_attr_t*)(smsg_connection_vec[destNode].addr))[myrank]));
    pd->remote_mem_hndl = smsg_connection_vec[destNode].mdh;
    pd->src_cq_hndl     = 0;
    pd->rdma_mode       = 0;
    status = GNI_PostFma(ep_hndl_array[destNode],  pd);
    print_smsg_attr(smsg_attr);
    if(status == GNI_RC_ERROR_RESOURCE )
    {
        MallocRdmaRequest(rdma_request_msg);
        rdma_request_msg->destNode = destNode;
        rdma_request_msg->pd = pd;
        /* buffer this request */
    }
#if PRINT_SYH
    if(status != GNI_RC_SUCCESS)
       printf("[%d=%d] send post FMA %s\n", myrank, destNode, gni_err_str[status]);
    else
        printf("[%d=%d]OK send post FMA \n", myrank, destNode);
#endif
}

/* useDynamicSMSG */
inline 
static void alloc_smsg_attr( gni_smsg_attr_t *local_smsg_attr)
{
    gni_return_t status = GNI_RC_NOT_DONE;

    if(mailbox_list->offset == mailbox_list->size)
    {
        dynamic_smsg_mailbox_t *new_mailbox_entry;
        new_mailbox_entry = (dynamic_smsg_mailbox_t*)malloc(sizeof(dynamic_smsg_mailbox_t));
        new_mailbox_entry->size = smsg_memlen*avg_smsg_connection;
        new_mailbox_entry->mailbox_base = malloc(new_mailbox_entry->size);
        bzero(new_mailbox_entry->mailbox_base, new_mailbox_entry->size);
        new_mailbox_entry->offset = 0;
        
        status = GNI_MemRegister(nic_hndl, (uint64_t)new_mailbox_entry->mailbox_base,
            new_mailbox_entry->size, smsg_rx_cqh,
            GNI_MEM_READWRITE,   
            -1,
            &(new_mailbox_entry->mem_hndl));

        //status = MEMORY_REGISTER(onesided_hnd, nic_hndl, new_mailbox_entry->mailbox_base, new_mailbox_entry->size, &(new_mailbox_entry->mem_hndl), &omdh);
        GNI_RC_CHECK("register", status);
        new_mailbox_entry->next = mailbox_list;
        mailbox_list = new_mailbox_entry;
    }
    local_smsg_attr->msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
    local_smsg_attr->mbox_maxcredit = SMSG_MAX_CREDIT;
    local_smsg_attr->msg_maxsize = SMSG_MAX_MSG;
    local_smsg_attr->mbox_offset = mailbox_list->offset;
    mailbox_list->offset += smsg_memlen;
    local_smsg_attr->buff_size = smsg_memlen;
    local_smsg_attr->msg_buffer = mailbox_list->mailbox_base;
    local_smsg_attr->mem_hndl = mailbox_list->mem_hndl;
}

/* useDynamicSMSG */
inline 
static int connect_to(int destNode)
{
    gni_return_t status = GNI_RC_NOT_DONE;
    CmiAssert(smsg_connected_flag[destNode] == 0);
    CmiAssert (smsg_attr_vector_local[destNode] == NULL);
    smsg_attr_vector_local[destNode] = (gni_smsg_attr_t*) malloc (sizeof(gni_smsg_attr_t));
    alloc_smsg_attr(smsg_attr_vector_local[destNode]);
    smsg_attr_vector_remote[destNode] = (gni_smsg_attr_t*) malloc (sizeof(gni_smsg_attr_t));
            
#if CMK_SMP && !COMM_THREAD_SEND
    //CmiLock(ep_lock_array[destNode]);
    CmiLock(tx_cq_lock);
#endif
    status = GNI_EpPostDataWId (ep_hndl_array[destNode], smsg_attr_vector_local[destNode], sizeof(gni_smsg_attr_t),smsg_attr_vector_remote[destNode] ,sizeof(gni_smsg_attr_t), destNode+mysize);
#if CMK_SMP && !COMM_THREAD_SEND
    //CmiUnlock(ep_lock_array[destNode]);
    CmiUnlock(tx_cq_lock);
#endif
    if (status == GNI_RC_ERROR_RESOURCE) {
      /* possibly destNode is making connection at the same time */
      free(smsg_attr_vector_local[destNode]);
      smsg_attr_vector_local[destNode] = NULL;
      free(smsg_attr_vector_remote[destNode]);
      smsg_attr_vector_remote[destNode] = NULL;
      mailbox_list->offset -= smsg_memlen;
      return 0;
    }
    GNI_RC_CHECK("GNI_Post", status);
    //printf("[%d] setting up %d -> %d\n", myrank, myrank, destNode);
    smsg_connected_flag[destNode] = 1;
    return 1;
}

//inline static gni_return_t send_smsg_message(int destNode, void *header, int size_header, void *msg, int size, uint8_t tag, int inbuff )
inline static gni_return_t send_smsg_message(int destNode, void *msg, int size, uint8_t tag, int inbuff )
{
    unsigned int          remote_address;
    uint32_t              remote_id;
    gni_return_t status = GNI_RC_NOT_DONE;
    gni_smsg_attr_t      *smsg_attr;
    gni_post_descriptor_t *pd;
    gni_post_state_t      post_state;
    char                  *real_data; 

    if (useDynamicSMSG) {
        switch (smsg_connected_flag[destNode]) {
        case 0: 
            connect_to(destNode);         /* continue to case 1 */
        case 1:                           /* pending connection, do nothing */
            status = GNI_RC_NOT_DONE;
            if(inbuff ==0)
                buffer_small_msgs(msg, size, destNode, tag);
            return status;
        }
    }
#if CMK_SMP
    if(PCQueueEmpty(smsg_msglist_index[destNode].sendSmsgBuf) || inbuff==1)
    {
#else
    if(smsg_msglist_index[destNode].sendSmsgBuf == 0 || inbuff==1)
    {
#endif
#if CMK_SMP && !COMM_THREAD_SEND
        CmiLock(tx_cq_lock);
#endif
        status = GNI_SmsgSendWTag(ep_hndl_array[destNode], 0, 0, msg, size, 0, tag);
#if CMK_SMP && !COMM_THREAD_SEND
        CmiUnlock(tx_cq_lock);
#endif
        if(status == GNI_RC_SUCCESS)
        {
#if CMK_SMP_TRACE_COMMTHREAD
            if(tag == SMALL_DATA_TAG || tag == LMSG_INIT_TAG)
            { 
                START_EVENT();
                if ( tag == SMALL_DATA_TAG)
                    real_data = (char*)msg; 
                else 
                    real_data = (char*)(((CONTROL_MSG*)msg)->source_addr);
                TRACE_COMM_CREATION(CpvAccess(projTraceStart), real_data);
            }
#endif
#if PRINT_SYH
            lrts_smsg_success++;
            printf("[%d==>%d] send done%d (msgs=%d)\n", myrank, destNode, lrts_smsg_success, lrts_send_msg_id);
#endif     
            return status;
        }
    }
    if(inbuff ==0)
        buffer_small_msgs(msg, size, destNode, tag);
    return status;
}


inline static CONTROL_MSG* construct_control_msg(int size, char *msg, uint8_t seqno)
{
    /* construct a control message and send */
    CONTROL_MSG         *control_msg_tmp;
    MallocControlMsg(control_msg_tmp);
    control_msg_tmp->source_addr = (uint64_t)msg;
    control_msg_tmp->seq_id    = seqno;
    //control_msg_tmp->source    = myrank;
    control_msg_tmp->total_length = control_msg_tmp->length = ALIGN64(size); //for GET 4 bytes aligned 
#if     USE_LRTS_MEMPOOL
    if(size < BIG_MSG)
    {
        control_msg_tmp->source_mem_hndl = GetMemHndl(msg);
    }
    else
    {
        SetMemHndlZero(control_msg_tmp->source_mem_hndl);
        control_msg_tmp->length = size - (seqno-1)*ONE_SEG;
        if (control_msg_tmp->length > ONE_SEG) control_msg_tmp->length = ONE_SEG;
    }
#else
    SetMemHndlZero(control_msg_tmp->source_mem_hndl);
#endif
    return control_msg_tmp;
}

// Large message, send control to receiver, receiver register memory and do a GET 
inline
static void send_large_messages(int destNode, CONTROL_MSG  *control_msg_tmp)
{
    gni_return_t        status  =   GNI_RC_SUCCESS;
    uint32_t            vmdh_index  = -1;
    int                 size;
    int                 offset = 0;
    uint64_t            source_addr;

    size    =   control_msg_tmp->total_length;
    source_addr = control_msg_tmp->source_addr;

    if(buffered_send_msg >= MAX_BUFF_SEND)
    {
        buffer_small_msgs(control_msg_tmp, sizeof(CONTROL_MSG), destNode, LMSG_INIT_TAG);
        CmiPrintf(" [%d] send_large hit max %lld\n", myrank, buffered_send_msg); 
        return;
    }
#if     USE_LRTS_MEMPOOL
    if( control_msg_tmp->seq_id == 0 ){
        if(IsMemHndlZero(GetMemHndl(source_addr))) //it is in mempool, it is possible to be de-registered by others
        {
            //register the corresponding mempool
            status = registerMempool((void*)source_addr);
            if(status == GNI_RC_SUCCESS)
            {
                control_msg_tmp->source_mem_hndl = GetMemHndl(source_addr);
            }
        }else
        {
            control_msg_tmp->source_mem_hndl = GetMemHndl(source_addr);
            status = GNI_RC_SUCCESS;
        }
    }else    // BIG_MSG
    {
        int offset = ONE_SEG*(control_msg_tmp->seq_id-1);
        source_addr += offset;
        size = control_msg_tmp->length;
//printf("[%d] send_large_messages seq: %d  addr: %p size: %d %d\n", CmiMyPe(), control_msg_tmp->seq_id, source_addr, size, control_msg_tmp->length);
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, source_addr, ALIGN64(size), &(control_msg_tmp->source_mem_hndl), &omdh);
    }

    if(status == GNI_RC_SUCCESS)
    {
        status = send_smsg_message( destNode, control_msg_tmp, sizeof(CONTROL_MSG), LMSG_INIT_TAG, 0);  
        if(status == GNI_RC_SUCCESS)
        {
            buffered_send_msg += ALIGN64(size);
            if(control_msg_tmp->seq_id == 0)
                IncreaseMsgInFlight(source_addr);
            FreeControlMsg(control_msg_tmp);
        }
    } else if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR)
    {
        CmiAbort("Memory registor for large msg\n");
    }else 
    {
        buffer_small_msgs(control_msg_tmp, sizeof(CONTROL_MSG), destNode, LMSG_INIT_TAG);
    }
#else
    status = MEMORY_REGISTER(onesided_hnd, nic_hndl,msg, ALIGN64(size), &(control_msg_tmp->source_mem_hndl), &omdh);
    if(status == GNI_RC_SUCCESS)
    {
        status = send_smsg_message( destNode, control_msg_tmp, sizeof(CONTROL_MSG), LMSG_INIT_TAG, 0);  
        if(status == GNI_RC_SUCCESS)
        {
            FreeControlMsg(control_msg_tmp);
        }
    } else if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR)
    {
        CmiAbort("Memory registor for large msg\n");
    }else 
    {
        buffer_small_msgs(control_msg_tmp, sizeof(CONTROL_MSG), destNode, LMSG_INIT_TAG);
    }
#endif
}

inline void LrtsPrepareEnvelope(char *msg, int size)
{
    CmiSetMsgSize(msg, size);
}

CmiCommHandle LrtsSendFunc(int destNode, int size, char *msg, int mode)
{
    gni_return_t        status  =   GNI_RC_SUCCESS;
    uint8_t tag;
    CONTROL_MSG         *control_msg_tmp;

    LrtsPrepareEnvelope(msg, size);

#if PRINT_SYH
    printf("LrtsSendFn %d==>%d, size=%d\n", myrank, destNode, size);
#endif 
#if CMK_SMP && COMM_THREAD_SEND
    if(size <= SMSG_MAX_MSG)
        buffer_small_msgs(msg, size, destNode, SMALL_DATA_TAG);
    else if (size < BIG_MSG) {
        control_msg_tmp =  construct_control_msg(size, msg, 0);
        buffer_small_msgs(control_msg_tmp, sizeof(CONTROL_MSG), destNode, LMSG_INIT_TAG);
    }
    else {
          CmiSetMsgSeq(msg, 0);
          control_msg_tmp =  construct_control_msg(size, msg, 1);
          buffer_small_msgs(control_msg_tmp, sizeof(CONTROL_MSG), destNode, LMSG_INIT_TAG);
    }
#else //non-smp, smp(worker sending)
    if(size <= SMSG_MAX_MSG)
    {
        status = send_smsg_message( destNode,  msg, size, SMALL_DATA_TAG, 0);  
        if(status == GNI_RC_SUCCESS)
            CmiFree(msg);
    }
    else if (size < BIG_MSG) {
        control_msg_tmp =  construct_control_msg(size, msg, 0);
        send_large_messages(destNode, control_msg_tmp);
    }
    else {
#if     USE_LRTS_MEMPOOL
        CmiSetMsgSeq(msg, 0);
        control_msg_tmp =  construct_control_msg(size, msg, 1);
        send_large_messages(destNode, control_msg_tmp);
#else
        control_msg_tmp =  construct_control_msg(size, msg, 0);
        send_large_messages(destNode, control_msg_tmp);
#endif
    }
#endif
    return 0;
}

static void    PumpDatagramConnection();
static void registerUserTraceEvents() {
#if CMI_MPI_TRACE_USEREVENTS && CMK_TRACE_ENABLED && !CMK_TRACE_IN_CHARM
    traceRegisterUserEvent("setting up connections", 10);
    traceRegisterUserEvent("Receiving small msgs", 20);
    traceRegisterUserEvent("Release local transaction", 30);
    traceRegisterUserEvent("Sending buffered small msgs", 40);
    traceRegisterUserEvent("Sending buffered rdma msgs", 50);
#endif
}

void LrtsPostCommonInit(int everReturn)
{
#if CMI_MPI_TRACE_USEREVENTS && CMK_TRACE_ENABLED && !CMK_TRACE_IN_CHARM
    CpvInitialize(double, projTraceStart);
    /* only PE 0 needs to care about registration (to generate sts file). */
    if (CmiMyPe() == 0) {
        registerMachineUserEventsFunction(&registerUserTraceEvents);
    }
#endif

#if CMK_SMP
    CmiIdleState *s=CmiNotifyGetState();
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,NULL);
    if (useDynamicSMSG)
    CcdCallOnConditionKeep(CcdPERIODIC_10ms, (CcdVoidFn) PumpDatagramConnection, NULL);
#endif
}

/* this is called by worker thread */
void LrtsPostNonLocal(){
#if CMK_SMP
#if !COMM_THREAD_SEND
    if(mysize == 1) return;
    PumpLocalRdmaTransactions();
    SendBufferMsg();
    SendRdmaMsg();
#endif
#endif
}

/* useDynamicSMSG */
static void    PumpDatagramConnection()
{
    uint32_t          remote_address;
    uint32_t          remote_id;
    gni_return_t status;
    gni_post_state_t  post_state;
    uint64_t          datagram_id;
    int i;

   while ((status = GNI_PostDataProbeById(nic_hndl, &datagram_id)) == GNI_RC_SUCCESS)
   {
       if (datagram_id >= mysize) {           /* bound endpoint */
           int pe = datagram_id - mysize;
#if CMK_SMP && !COMM_THREAD_SEND
           CmiLock(tx_cq_lock);
#endif
           status = GNI_EpPostDataTestById( ep_hndl_array[pe], datagram_id, &post_state, &remote_address, &remote_id);
#if CMK_SMP && !COMM_THREAD_SEND
           CmiUnlock(tx_cq_lock);
#endif
       if(status == GNI_RC_SUCCESS && post_state == GNI_POST_COMPLETED)
       {
          CmiAssert(remote_id == pe);
          status = GNI_SmsgInit(ep_hndl_array[pe], smsg_attr_vector_local[pe], smsg_attr_vector_remote[pe]);
          GNI_RC_CHECK("Dynamic SMSG Init", status);
#if PRINT_SYH
          printf("++ Dynamic SMSG setup [%d===>%d] done\n", myrank, pe);
#endif
	  CmiAssert(smsg_connected_flag[pe] == 1);
          smsg_connected_flag[pe] = 2;
       }
     }
     else {         /* unbound ep */
       status = GNI_EpPostDataTestById( ep_hndl_unbound, datagram_id, &post_state, &remote_address, &remote_id);
       if(status == GNI_RC_SUCCESS && post_state == GNI_POST_COMPLETED)
       {
          CmiAssert(remote_id<mysize);
	  CmiAssert(smsg_connected_flag[remote_id] <= 0);
          status = GNI_SmsgInit(ep_hndl_array[remote_id], &send_smsg_attr, &recv_smsg_attr);
          GNI_RC_CHECK("Dynamic SMSG Init", status);
#if PRINT_SYH
          printf("++ Dynamic SMSG setup2 [%d===>%d] done\n", myrank, remote_id);
#endif
          smsg_connected_flag[remote_id] = 2;

          alloc_smsg_attr(&send_smsg_attr);
          status = GNI_EpPostDataWId (ep_hndl_unbound, &send_smsg_attr,  SMSG_ATTR_SIZE, &recv_smsg_attr, SMSG_ATTR_SIZE, myrank);
          GNI_RC_CHECK("post unbound datagram", status);
        }
     }
   }
}

/* pooling CQ to receive network message */
static void PumpNetworkRdmaMsgs()
{
    gni_cq_entry_t      event_data;
    gni_return_t        status;

}

static void getLargeMsgRequest(void* header, uint64_t inst_id);
static void PumpNetworkSmsg()
{
    uint64_t            inst_id;
    int                 ret;
    gni_cq_entry_t      event_data;
    gni_return_t        status;
    void                *header;
    uint8_t             msg_tag;
    int                 msg_nbytes;
    void                *msg_data;
    gni_mem_handle_t    msg_mem_hndl;
    gni_smsg_attr_t     *smsg_attr;
    gni_smsg_attr_t     *remote_smsg_attr;
    int                 init_flag;
    CONTROL_MSG         *control_msg_tmp, *header_tmp;
    uint64_t            source_addr;
#if CMK_SMP && !COMM_THREAD_SEND
    while(1)
    {
        CmiLock(tx_cq_lock);
        status =GNI_CqGetEvent(smsg_rx_cqh, &event_data);
        CmiUnlock(tx_cq_lock);
        if(status != GNI_RC_SUCCESS)
            break;
#else
    while ( (status =GNI_CqGetEvent(smsg_rx_cqh, &event_data)) == GNI_RC_SUCCESS)
    {
#endif
        inst_id = GNI_CQ_GET_INST_ID(event_data);
        // GetEvent returns success but GetNext return not_done. caused by Smsg out-of-order transfer
#if PRINT_SYH
        printf("[%d] PumpNetworkMsgs is received from PE: %d,  status=%s\n", myrank, inst_id,  gni_err_str[status]);
#endif
        if (useDynamicSMSG) {
            /* subtle: smsg may come before connection is setup */
            while (smsg_connected_flag[inst_id] != 2) 
               PumpDatagramConnection();
        }
        msg_tag = GNI_SMSG_ANY_TAG;
#if CMK_SMP && !COMM_THREAD_SEND
        while(1) {
            CmiLock(tx_cq_lock);
            status = GNI_SmsgGetNextWTag(ep_hndl_array[inst_id], &header, &msg_tag);
            CmiUnlock(tx_cq_lock);
            if (status != GNI_RC_SUCCESS)
                break;
#else
        while( (status = GNI_SmsgGetNextWTag(ep_hndl_array[inst_id], &header, &msg_tag)) == GNI_RC_SUCCESS){
#endif
            /* copy msg out and then put into queue (small message) */
            switch (msg_tag) {
            case SMALL_DATA_TAG:
            {
                START_EVENT();
                msg_nbytes = CmiGetMsgSize(header);
                msg_data    = CmiAlloc(msg_nbytes);
                memcpy(msg_data, (char*)header, msg_nbytes);
#if CMK_SMP_TRACE_COMMTHREAD
                TRACE_COMM_CREATION(CpvAccess(projTraceStart), msg_data);
#endif
                handleOneRecvedMsg(msg_nbytes, msg_data);
                break;
            }
            case LMSG_INIT_TAG:
            {
#if PRINT_SYH
                printf("[%d] from %d request for Large msg is received, messageid:%d tag=%d\n", myrank, inst_id, lrts_received_msg, msg_tag);
#endif
                getLargeMsgRequest(header, inst_id);
                break;
            }
            case ACK_TAG:   //msg fit into mempool
            {
                /* Get is done, release message . Now put is not used yet*/
#if ! USE_LRTS_MEMPOOL
                MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &(((CONTROL_MSG *)header)->source_mem_hndl), &omdh);
#else
                DecreaseMsgInFlight( ((void*)((CONTROL_MSG *) header)->source_addr));
#endif
                buffered_send_msg -= ((CONTROL_MSG *) header)->length;
                CmiFree((void*)((CONTROL_MSG *) header)->source_addr);
                //SendRdmaMsg();
                break;
            }
            case BIG_MSG_TAG:  //big msg, de-register, transfer next seg
            {
                header_tmp = (CONTROL_MSG *) header;
                void *msg = (void*)(header_tmp->source_addr);
                int cur_seq = CmiGetMsgSeq(header_tmp->source_addr);
                int offset = ONE_SEG*(cur_seq+1);
                MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &(header_tmp->source_mem_hndl), &omdh);
                buffered_send_msg -= ((CONTROL_MSG *) header)->length;
                int remain_size = CmiGetMsgSize(msg) - header_tmp->length;
                if (remain_size < 0) remain_size = 0;
                CmiSetMsgSize(msg, remain_size);
//printf("[%d] PumpNetworkSmsg: free %p len: %d offset: %d seq: %d msg size: %d %d\n", myrank, header_tmp->source_addr_orig, header_tmp->total_length, offset, cur_seq, CmiGetMsgSize(msg), header_tmp->length);
                if(remain_size <= 0) //transaction done
                {
                    CmiFree(msg);
                }else if (header_tmp->total_length > offset)
                {
                    CmiSetMsgSeq(header_tmp->source_addr, cur_seq+1);
                    MallocControlMsg(control_msg_tmp);
                    control_msg_tmp->source_addr = header_tmp->source_addr;
                    control_msg_tmp->dest_addr    = header_tmp->dest_addr;
                    control_msg_tmp->total_length   = header_tmp->total_length; 
                    //control_msg_tmp->source         = myrank;
                    control_msg_tmp->length         = header_tmp->total_length - offset;
                    if (control_msg_tmp->length >= ONE_SEG) control_msg_tmp->length = ONE_SEG;
                    control_msg_tmp->seq_id         = cur_seq+1+1;
                    //send next seg
                    send_large_messages(inst_id, control_msg_tmp);
                         // pipelining
                    if (header_tmp->seq_id == 1) {
                      int i;
                      for (i=1; i<BIG_MSG_PIPELINE; i++) {
                        int seq = cur_seq+i+2;
                        CmiSetMsgSeq(header_tmp->source_addr, seq-1);
                        control_msg_tmp =  construct_control_msg(header_tmp->total_length, (char *)header_tmp->source_addr, seq);
                        control_msg_tmp->dest_addr = header_tmp->dest_addr;
                        send_large_messages(inst_id, control_msg_tmp);
                        if (header_tmp->total_length <= ONE_SEG*seq) break;
                      }
                    }
                }
                //SendRdmaMsg();
                break;
            }

#if CMK_PERSISTENT_COMM
            case PUT_DONE_TAG: //persistent message
            void *msg = (void *)((CONTROL_MSG *) header)->source_addr;
            int size = ((CONTROL_MSG *) header)->length;
            CmiReference(msg);
            handleOneRecvedMsg(size, msg); 
            break;
#endif
#ifdef CMK_DIRECT
            case DIRECT_PUT_DONE_TAG:  //cmi direct 
            (*(((CMK_DIRECT_HEADER*) header)->callbackFnPtr))(((CMK_DIRECT_HEADER*) header)->callbackData);
           break;
#endif
            default: {
                printf("weird tag problem\n");
                CmiAbort("Unknown tag\n");
                     }
            }
#if CMK_SMP && !COMM_THREAD_SEND
            CmiLock(tx_cq_lock);
#endif
            GNI_SmsgRelease(ep_hndl_array[inst_id]);
#if CMK_SMP && !COMM_THREAD_SEND
            CmiUnlock(tx_cq_lock);
#endif
            msg_tag = GNI_SMSG_ANY_TAG;
        } //endwhile getNext
    }   //end while GetEvent
    if(status == GNI_RC_ERROR_RESOURCE)
    {
        GNI_RC_CHECK("Smsg_rx_cq full", status);
    }
}

static void printDesc(gni_post_descriptor_t *pd)
{
    printf(" addr=%p, ", pd->local_addr); 
    printf(" remote addr=%p, ", pd->remote_addr);
    printf(" local %lld %lld, remote %lld, %lld\n", pd->local_mem_hndl.qword1, pd->local_mem_hndl.qword2, pd->remote_mem_hndl.qword1, pd->remote_mem_hndl.qword2); 
}

// for BIG_MSG called on receiver side for receiving control message
// LMSG_INIT_TAG
static void getLargeMsgRequest(void* header, uint64_t inst_id )
{
#if     USE_LRTS_MEMPOOL
    CONTROL_MSG         *request_msg;
    gni_return_t        status = GNI_RC_SUCCESS;
    void                *msg_data;
    gni_post_descriptor_t *pd;
    RDMA_REQUEST        *rdma_request_msg;
    gni_mem_handle_t    msg_mem_hndl;
    int source, size, transaction_size, offset = 0;

    // initial a get to transfer data from the sender side */
    request_msg = (CONTROL_MSG *) header;
    //source = request_msg->source;
    size = request_msg->total_length; 
//printf("[%d] getLargeMsgRequest seq: %d size: %d\n", CmiMyPe(), request_msg->seq_id , size);
    if(request_msg->seq_id < 2)   {
        msg_data = CmiAlloc(size);
        CmiSetMsgSeq(msg_data, 0);
        _MEMCHECK(msg_data);
    }
    else {
        offset = ONE_SEG*(request_msg->seq_id-1);
        msg_data = (char*)request_msg->dest_addr + offset;
    }
   
    MallocPostDesc(pd);
    pd->cqwrite_value = request_msg->seq_id;
    if( request_msg->seq_id == 0)
    {
        pd->local_mem_hndl= GetMemHndl(msg_data);
        transaction_size = ALIGN64(size);
        if(IsMemHndlZero(pd->local_mem_hndl))
        {
            status = registerMempool((void*)(msg_data));
            if(status == GNI_RC_SUCCESS)
            {
                pd->local_mem_hndl = GetMemHndl(msg_data);
            }
            else
            {
                SetMemHndlZero(pd->local_mem_hndl);
            }
        }
    }
    else{
        transaction_size = ALIGN64(request_msg->length);
        status = MEMORY_REGISTER(onesided_hnd, nic_hndl, msg_data, transaction_size, &(pd->local_mem_hndl), &omdh);
        if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR) 
        {
            GNI_RC_CHECK("Invalid/permission Mem Register in post", status);
        }
    }
    pd->first_operand = ALIGN64(size);                   //  total length

    if(request_msg->total_length < LRTS_GNI_RDMA_THRESHOLD)
        pd->type            = GNI_POST_FMA_GET;
    else
        pd->type            = GNI_POST_RDMA_GET;
#if REMOTE_EVENT
    pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT |  GNI_CQMODE_REMOTE_EVENT;
#else
    pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT;
#endif
    pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
    pd->length          = transaction_size;
    pd->local_addr      = (uint64_t) msg_data;
    pd->remote_addr     = request_msg->source_addr + offset;
    pd->remote_mem_hndl = request_msg->source_mem_hndl;
    pd->src_cq_hndl     = 0;//post_tx_cqh;     /* smsg_tx_cqh;  */
    pd->rdma_mode       = 0;

    //memory registration success
    if(status == GNI_RC_SUCCESS)
    {
       // CmiPrintf(" PE:%d reigster(size=%d)(%s) (%lld, %lld), (%lld, %lld)\n", myrank, pd->length, gni_err_str[status], (pd->local_mem_hndl).qword1, (pd->local_mem_hndl).qword2, (pd->remote_mem_hndl).qword1, (pd->remote_mem_hndl).qword2);
#if CMK_SMP && !COMM_THREAD_SEND
        CmiLock(tx_cq_lock);
#endif
        if(pd->type == GNI_POST_RDMA_GET) 
            status = GNI_PostRdma(ep_hndl_array[inst_id], pd);
        else
            status = GNI_PostFma(ep_hndl_array[inst_id],  pd);
#if CMK_SMP && !COMM_THREAD_SEND
        CmiUnlock(tx_cq_lock);
#endif
         
        if(status == GNI_RC_SUCCESS )
        {
            if(pd->cqwrite_value == 0)
                IncreaseMsgInFlight(msg_data);
        }
    }else
    {
        SetMemHndlZero(pd->local_mem_hndl);
    }
    if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
    {
        MallocRdmaRequest(rdma_request_msg);
        rdma_request_msg->destNode = inst_id;
        rdma_request_msg->pd = pd;
#if CMK_SMP
        PCQueuePush(sendRdmaBuf, (char*)rdma_request_msg);
#else
        if(sendRdmaBuf == 0)
        {
            sendRdmaBuf = sendRdmaTail = rdma_request_msg;
        }else{
            sendRdmaTail->next = rdma_request_msg;
            sendRdmaTail =  rdma_request_msg;
        }
#endif
    }else {
        /* printf("source: %d pd:%p\n", source, pd); */
        GNI_RC_CHECK("AFter posting", status);
    }
#else
    CONTROL_MSG         *request_msg;
    gni_return_t        status;
    void                *msg_data;
    gni_post_descriptor_t *pd;
    RDMA_REQUEST        *rdma_request_msg;
    gni_mem_handle_t    msg_mem_hndl;
    //int source;
    // initial a get to transfer data from the sender side */
    request_msg = (CONTROL_MSG *) header;
    //source = request_msg->source;
    msg_data = CmiAlloc(request_msg->length);
    _MEMCHECK(msg_data);

    status = MEMORY_REGISTER(onesided_hnd, nic_hndl, msg_data, request_msg->length, &msg_mem_hndl, &omdh);

    if (status == GNI_RC_INVALID_PARAM || status == GNI_RC_PERMISSION_ERROR) 
    {
        GNI_RC_CHECK("Invalid/permission Mem Register in post", status);
    }

    MallocPostDesc(pd);
    if(request_msg->length < LRTS_GNI_RDMA_THRESHOLD) 
        pd->type            = GNI_POST_FMA_GET;
    else
        pd->type            = GNI_POST_RDMA_GET;
#if REMOTE_EVENT
    pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT |  GNI_CQMODE_REMOTE_EVENT;
#else
    pd->cq_mode         = GNI_CQMODE_GLOBAL_EVENT;
#endif
    pd->dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
    pd->length          = ALIGN64(request_msg->length);
    pd->local_addr      = (uint64_t) msg_data;
    pd->remote_addr     = request_msg->source_addr;
    pd->remote_mem_hndl = request_msg->source_mem_hndl;
    pd->src_cq_hndl     = 0;//post_tx_cqh;     /* smsg_tx_cqh;  */
    pd->rdma_mode       = 0;

    //memory registration successful
    if(status == GNI_RC_SUCCESS)
    {
        pd->local_mem_hndl  = msg_mem_hndl;
#if CMK_SMP && !COMM_THREAD_SEND
            CmiLock(tx_cq_lock);
#endif
        if(pd->type == GNI_POST_RDMA_GET) 
            status = GNI_PostRdma(ep_hndl_array[inst_id], pd);
        else
            status = GNI_PostFma(ep_hndl_array[inst_id],  pd);
#if CMK_SMP && !COMM_THREAD_SEND
            CmiUnlock(tx_cq_lock);
#endif
    }else
    {
        SetMemHndlZero(pd->local_mem_hndl);
    }
    if(status == GNI_RC_ERROR_RESOURCE|| status == GNI_RC_ERROR_NOMEM )
    {
        MallocRdmaRequest(rdma_request_msg);
        rdma_request_msg->next = 0;
        rdma_request_msg->destNode = inst_id;
        rdma_request_msg->pd = pd;
        PCQueuePush(sendRdmaBuf, (char*)rdma_request_msg);
    }else {
        /* printf("source: %d pd:%p\n", source, pd); */
        GNI_RC_CHECK("AFter posting", status);
    }
#endif
}

static void PumpLocalRdmaTransactions()
{
    gni_cq_entry_t          ev;
    gni_return_t            status;
    uint64_t                type, inst_id;
    gni_post_descriptor_t   *tmp_pd;
    MSG_LIST                *ptr;
    CONTROL_MSG             *ack_msg_tmp;
    uint8_t             msg_tag;
#ifdef CMK_DIRECT
    CMK_DIRECT_HEADER       *cmk_direct_done_msg;
#endif
#if CMK_SMP && !COMM_THREAD_SEND
    while(1) {
        CmiLock(tx_cq_lock);
        status = GNI_CqGetEvent(smsg_tx_cqh, &ev);
        CmiUnlock(tx_cq_lock);
        if(status != GNI_RC_SUCCESS)
            break;
#else
    while( (status = GNI_CqGetEvent(smsg_tx_cqh, &ev)) == GNI_RC_SUCCESS)
    {
#endif
        type = GNI_CQ_GET_TYPE(ev);
        if (type == GNI_CQ_EVENT_TYPE_POST)
        {
            inst_id     = GNI_CQ_GET_INST_ID(ev);
#if PRINT_SYH
            printf("[%d] LocalTransactions localdone=%d\n", myrank,  lrts_local_done_msg);
#endif
            //status = GNI_GetCompleted(post_tx_cqh, ev, &tmp_pd);
#if CMK_SMP && !COMM_THREAD_SEND
            CmiLock(tx_cq_lock);
#endif
            status = GNI_GetCompleted(smsg_tx_cqh, ev, &tmp_pd);
#if CMK_SMP && !COMM_THREAD_SEND
            CmiUnlock(tx_cq_lock);
#endif
#ifdef CMK_DIRECT
            if(tmp_pd->amo_cmd == 1)
            {
                cmk_direct_done_msg = (CMK_DIRECT_HEADER*) malloc(sizeof(CMK_DIRECT_HEADER));
                cmk_direct_done_msg->callbackFnPtr = (void*)( tmp_pd->first_operand);
                cmk_direct_done_msg->recverBuf = (void*)(tmp_pd->remote_addr);
                cmk_direct_done_msg->callbackData = (void*)(tmp_pd->second_operand); 
            }
            else{
                MallocControlMsg(ack_msg_tmp);
                ack_msg_tmp->source_addr = tmp_pd->remote_addr;
                ack_msg_tmp->source_mem_hndl    = tmp_pd->remote_mem_hndl;
            }
#else
            MallocControlMsg(ack_msg_tmp);
            ack_msg_tmp->source_addr = tmp_pd->remote_addr;
            ack_msg_tmp->source_mem_hndl    = tmp_pd->remote_mem_hndl;
#endif
            ////Message is sent, free message , put is not used now
            switch (tmp_pd->type) {
#if CMK_PERSISTENT_COMM
            case GNI_POST_RDMA_PUT:
#if ! USE_LRTS_MEMPOOL
                MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &tmp_pd->local_mem_hndl, &omdh);
#endif
            case GNI_POST_FMA_PUT:
                CmiFree((void *)tmp_pd->local_addr);
                msg_tag = PUT_DONE_TAG;
                break;
#endif
#ifdef CMK_DIRECT
            case GNI_POST_RDMA_PUT:
            case GNI_POST_FMA_PUT:
                //sender ACK to receiver to trigger it is done
                msg_tag = DIRECT_PUT_DONE_TAG;
                break;
#endif
            case GNI_POST_RDMA_GET:
            case GNI_POST_FMA_GET:
#if  ! USE_LRTS_MEMPOOL
                MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &tmp_pd->local_mem_hndl, &omdh);
                msg_tag = ACK_TAG;  
#else
                ack_msg_tmp->seq_id = tmp_pd->cqwrite_value;
                if(ack_msg_tmp->seq_id > 0)      // BIG_MSG
                {
                    msg_tag = BIG_MSG_TAG; 
                    MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &tmp_pd->local_mem_hndl, &omdh);
                    ack_msg_tmp->dest_addr = tmp_pd->local_addr - ONE_SEG*(ack_msg_tmp->seq_id-1);
                    ack_msg_tmp->source_addr -= ONE_SEG*(ack_msg_tmp->seq_id-1);
                } 
                else
                {
                    msg_tag = ACK_TAG;  
                    ack_msg_tmp->dest_addr = tmp_pd->local_addr;
                }
                ack_msg_tmp->length = tmp_pd->length;
                ack_msg_tmp->total_length = tmp_pd->first_operand;     // total size
#endif
                break;
            default:
                CmiPrintf("type=%d\n", tmp_pd->type);
                CmiAbort("PumpLocalRdmaTransactions: unknown type!");
            }
#if CMK_DIRECT
            if(tmp_pd->amo_cmd == 1)
                status = send_smsg_message(inst_id, cmk_direct_done_msg, sizeof(CMK_DIRECT_HEADER), msg_tag, 0); 
            else
#endif
                status = send_smsg_message(inst_id, ack_msg_tmp, sizeof(CONTROL_MSG), msg_tag, 0);  
            if(status == GNI_RC_SUCCESS)
            {
#if CMK_DIRECT
                if(tmp_pd->amo_cmd == 1)
                    free(cmk_direct_done_msg); 
                else
#endif
                    FreeControlMsg(ack_msg_tmp);
            }
#if CMK_PERSISTENT_COMM
            if (tmp_pd->type == GNI_POST_RDMA_GET || tmp_pd->type == GNI_POST_FMA_GET)
#endif
            {
                if( msg_tag == ACK_TAG){    //msg fit in mempool 
#if PRINT_SYH
                    printf("Normal msg transaction PE:%d==>%d\n", myrank, inst_id);
#endif
#if CMK_SMP_TRACE_COMMTHREAD
                    START_EVENT();
#endif
                    CmiAssert(SIZEFIELD((void*)(tmp_pd->local_addr)) <= tmp_pd->length);
                    DecreaseMsgInFlight((void*)tmp_pd->local_addr);
#if CMK_SMP_TRACE_COMMTHREAD
                    TRACE_COMM_CREATION(CpvAccess(projTraceStart), (void*)tmp_pd->local_addr);
#endif
                    handleOneRecvedMsg(tmp_pd->length, (void*)tmp_pd->local_addr); 
                }else {
                  void *msg = (void*)tmp_pd->local_addr-(tmp_pd->cqwrite_value-1)*ONE_SEG;
                  CmiSetMsgSeq(msg, CmiGetMsgSeq(msg)+1);
                  if (tmp_pd->first_operand <= ONE_SEG*CmiGetMsgSeq(msg)) {
#if CMK_SMP_TRACE_COMMTHREAD
                    START_EVENT();
#endif
#if PRINT_SYH
                    printf("Pipeline msg done [%d]\n", myrank);
#endif
#if CMK_SMP_TRACE_COMMTHREAD
                    TRACE_COMM_CREATION(CpvAccess(projTraceStart), msg);
#endif
                    handleOneRecvedMsg(tmp_pd->first_operand, msg); 
                  }
                }
                //SendRdmaMsg();
            }
            FreePostDesc(tmp_pd);
        }
    } //end while
    if(status == GNI_RC_ERROR_RESOURCE)
    {
        GNI_RC_CHECK("Smsg_tx_cq full", status);
    }
}

static void  SendRdmaMsg()
{
    gni_return_t            status = GNI_RC_SUCCESS;
    gni_mem_handle_t        msg_mem_hndl;
    RDMA_REQUEST *ptr = NULL;

#if CMK_SMP
    while (!(PCQueueEmpty(sendRdmaBuf)))
    {
        ptr = (RDMA_REQUEST*)PCQueuePop(sendRdmaBuf);
#else
    while( sendRdmaBuf != 0) {
        ptr = sendRdmaBuf;
#endif
        gni_post_descriptor_t *pd = ptr->pd;
        status = GNI_RC_SUCCESS;
        
        if(pd->cqwrite_value == 0)
        {
            if(IsMemHndlZero((GetMemHndl(pd->local_addr))))
            {
                status = registerMempool((void*)(pd->local_addr));
                if(status == GNI_RC_SUCCESS)
                {
                    pd->local_mem_hndl = GetMemHndl((void*)(pd->local_addr));
                }
            }else
            {
                pd->local_mem_hndl = GetMemHndl((void*)(pd->local_addr));
                status = GNI_RC_SUCCESS;
            }
        }else if( IsMemHndlZero(pd->local_mem_hndl)) //big msg, can not fit into memory pool
        {
            status = MEMORY_REGISTER(onesided_hnd, nic_hndl, pd->local_addr, pd->length, &(pd->local_mem_hndl), &omdh);
        }
        if(status == GNI_RC_SUCCESS)
        {
#if !CMK_SMP
            sendRdmaBuf = sendRdmaBuf->next;
#endif
#if CMK_SMP && !COMM_THREAD_SEND
            CmiLock(tx_cq_lock);
#endif
            if(pd->type == GNI_POST_RDMA_GET || pd->type == GNI_POST_RDMA_PUT) 
                status = GNI_PostRdma(ep_hndl_array[ptr->destNode], pd);
            else
                status = GNI_PostFma(ep_hndl_array[ptr->destNode],  pd);
#if CMK_SMP && !COMM_THREAD_SEND
            CmiUnlock(tx_cq_lock);
#endif
            if(status == GNI_RC_SUCCESS)
            {
                if(pd->cqwrite_value == 0)
                    IncreaseMsgInFlight(((void*)(pd->local_addr)));
                FreeRdmaRequest(ptr);
                continue;
            }
        }else
        {
#if CMK_SMP
            PCQueuePush(sendRdmaBuf, (char*)ptr);
#endif
            break;
        }
    } //end while
}

// return 1 if all messages are sent
static int SendBufferMsg()
{
    MSG_LIST            *ptr, *previous_head, *current_head;
    CONTROL_MSG         *control_msg_tmp;
    gni_return_t        status;
    int done = 1;
    register    int     i, register_size;
    void                *register_addr;
    int                 index_previous = -1;
    int                 index = smsg_head_index;
#if CMI_EXERT_SEND_CAP
    int			sent_cnt = 0;
#endif

#if ! CMK_SMP
    index = smsg_head_index;
#else
    index = 0;
#endif
    //if( smsg_msglist_head == 0 && buffered_smsg_counter!= 0 ) {printf("WRONGWRONG on rank%d, buffermsg=%d, (msgid-succ:%d)\n", myrank, buffered_smsg_counter, (lrts_send_msg_id-lrts_smsg_success)); CmiAbort("sendbuf");}
    /* can add flow control here to control the number of messages sent before handle message */

#if CMK_SMP
    while(index <mysize)
    {
        while(!PCQueueEmpty(smsg_msglist_index[index].sendSmsgBuf))
        {
#else
    while(index != -1)
    {
        while(smsg_msglist_index[index].sendSmsgBuf != 0)
        {
#endif
            if (useDynamicSMSG && smsg_connected_flag[index] != 2) {   
                /* connection not exists yet */
              done = 0;
              break;
            }
#if CMK_SMP
            ptr = (MSG_LIST*)PCQueuePop(smsg_msglist_index[index].sendSmsgBuf);
#else
            ptr = smsg_msglist_index[index].sendSmsgBuf;
#endif
            CmiAssert(ptr!=NULL);
            status = GNI_RC_ERROR_RESOURCE;
            switch(ptr->tag)
            {
            case SMALL_DATA_TAG:
                status = send_smsg_message( ptr->destNode,  ptr->msg, ptr->size, ptr->tag, 1);  
                if(status == GNI_RC_SUCCESS)
                {
                    CmiFree(ptr->msg);
                }
                break;
            case LMSG_INIT_TAG:
                if(buffered_send_msg >= MAX_BUFF_SEND)
                {
                    CmiPrintf(" [%d] send_buff hit max %lld\n", myrank, buffered_send_msg); 
                    done = 0; break;
                }
                control_msg_tmp = (CONTROL_MSG*)ptr->msg;
                register_size = control_msg_tmp->length;
                uint64_t            source_addr;
                source_addr = control_msg_tmp->source_addr;
#if     USE_LRTS_MEMPOOL
                if(control_msg_tmp->seq_id ==0) //fit into memory
                {
                    if(IsMemHndlZero(GetMemHndl(control_msg_tmp->source_addr))) //it is in mempool, it is possible to be de-registered by others
                    {
                        status = registerMempool((void*)(control_msg_tmp->source_addr));
                        if(status == GNI_RC_SUCCESS)
                        {
                            control_msg_tmp->source_mem_hndl = GetMemHndl(control_msg_tmp->source_addr);
                        }
                    }else{
                        control_msg_tmp->source_mem_hndl = GetMemHndl(control_msg_tmp->source_addr);
                        status = GNI_RC_SUCCESS;
                    }
                }
                else if(control_msg_tmp->seq_id >0) //large msg
                {
                    //register_size = control_msg_tmp->length>=ONE_SEG?ONE_SEG:control_msg_tmp->length;
                    int offset = ONE_SEG*(control_msg_tmp->seq_id-1);
                    source_addr += offset;
                    register_addr = (void*) source_addr;
//printf("[%d] send_large_buffer messages seq: %d  addr: %p size: %d %d\n", CmiMyPe(), control_msg_tmp->seq_id, source_addr, register_size, control_msg_tmp->length);
                    status = MEMORY_REGISTER(onesided_hnd, nic_hndl, register_addr, register_size, &(control_msg_tmp->source_mem_hndl), &omdh);
                }
#else
                status = MEMORY_REGISTER(onesided_hnd, nic_hndl, register_addr, register_size, &(control_msg_tmp->source_mem_hndl), &omdh);
#endif
                if(status != GNI_RC_SUCCESS) {
                    done = 0;
                    break;
                }
                status = send_smsg_message( ptr->destNode, ptr->msg, sizeof(CONTROL_MSG), ptr->tag, 1);  
                if(status == GNI_RC_SUCCESS)
                {   
                    buffered_send_msg += ALIGN64(register_size);
                    if(control_msg_tmp->seq_id ==0)
                        IncreaseMsgInFlight(source_addr);
                    FreeControlMsg((CONTROL_MSG*)(ptr->msg));
                }
                break;
            case   ACK_TAG:
            case   BIG_MSG_TAG:
                status = send_smsg_message( ptr->destNode, ptr->msg, sizeof(CONTROL_MSG), ptr->tag, 1);  
                if(status == GNI_RC_SUCCESS)
                {
                    FreeControlMsg((CONTROL_MSG*)ptr->msg);
                }
                break;
#ifdef CMK_DIRECT
            case DIRECT_PUT_DONE_TAG:
                status = send_smsg_message( ptr->destNode, ptr->msg, sizeof(CMK_DIRECT_HEADER), ptr->tag, 1);  
                if(status == GNI_RC_SUCCESS)
                {
                    free((CMK_DIRECT_HEADER*)ptr->msg);
                }
                break;

#endif
            default:
                printf("Weird tag\n");
                CmiAbort("should not happen\n");
            }
            if(status == GNI_RC_SUCCESS)
            {
#if !CMK_SMP
                smsg_msglist_index[index].sendSmsgBuf = smsg_msglist_index[index].sendSmsgBuf->next;
#endif
#if PRINT_SYH
                buffered_smsg_counter--;
                printf("[%d==>%d] buffered smsg sending done\n", myrank, ptr->destNode);
#endif
                FreeMsgList(ptr);
#if CMI_EXERT_SEND_CAP
                sent_cnt++;
                if(sent_cnt == SEND_CAP)
                    break;
#endif
            }else {
#if CMK_SMP
                PCQueuePush(smsg_msglist_index[index].sendSmsgBuf, (char*)ptr);
#endif
                done = 0;
                break;
            } 
        
        } //end while
#if !CMK_SMP
        if(smsg_msglist_index[index].sendSmsgBuf == 0)
        {
            if(index_previous != -1)
                smsg_msglist_index[index_previous].next = smsg_msglist_index[index].next;
            else
                smsg_head_index = smsg_msglist_index[index].next;
        }else
        {
            index_previous = index;
        }
        index = smsg_msglist_index[index].next;
#else
        index++;
#endif

#if CMI_EXERT_SEND_CAP
	if(sent_cnt == SEND_CAP)
		break;
#endif
    }   // end pooling for all cores
    return done;
}

void LrtsAdvanceCommunication(int whileidle)
{
    /*  Receive Msg first */
#if 0
    if(myrank == 0)
    printf("Calling Lrts Pump Msg PE:%d\n", myrank);
#endif
#if CMK_SMP_TRACE_COMMTHREAD
    double startT, endT;
#endif
    //if(mysize == 1) return;
    if (useDynamicSMSG && whileidle)
    {
#if CMK_SMP_TRACE_COMMTHREAD
        startT = CmiWallTimer();
#endif
        PumpDatagramConnection();
#if CMK_SMP_TRACE_COMMTHREAD
        endT = CmiWallTimer();
        if (endT-startT>=TRACE_THRESHOLD) traceUserBracketEvent(10, startT, endT);
#endif
    }

#if CMK_SMP_TRACE_COMMTHREAD
    startT = CmiWallTimer();
#endif
    PumpNetworkSmsg();
#if CMK_SMP_TRACE_COMMTHREAD
    endT = CmiWallTimer();
    if (endT-startT>=TRACE_THRESHOLD) traceUserBracketEvent(20, startT, endT);
#endif

#if CMK_SMP_TRACE_COMMTHREAD
    startT = CmiWallTimer();
#endif
    PumpLocalRdmaTransactions();
#if CMK_SMP_TRACE_COMMTHREAD
    endT = CmiWallTimer();
    if (endT-startT>=TRACE_THRESHOLD) traceUserBracketEvent(30, startT, endT);
#endif
#if 0
    if(myrank == 0)
    printf("Calling Lrts Send Buffmsg PE:%d\n", myrank);
#endif
    /* Send buffered Message */
#if CMK_SMP_TRACE_COMMTHREAD
    startT = CmiWallTimer();
#endif
    SendBufferMsg();
#if CMK_SMP_TRACE_COMMTHREAD
    endT = CmiWallTimer();
    if (endT-startT>=TRACE_THRESHOLD) traceUserBracketEvent(40, startT, endT);
#endif

#if CMK_SMP_TRACE_COMMTHREAD
    startT = CmiWallTimer();
#endif
    SendRdmaMsg();
#if CMK_SMP_TRACE_COMMTHREAD
    endT = CmiWallTimer();
    if (endT-startT>=TRACE_THRESHOLD) traceUserBracketEvent(50, startT, endT);
#endif
    //CmiPrintf("[%d]send buffer=%dM\n", myrank, buffered_send_msg/(1024*1024));
#if 0
    if(myrank == 0)
    printf("done PE:%d\n", myrank);
#endif
}

/* useDynamicSMSG */
static void _init_dynamic_smsg()
{
    gni_return_t status;
    uint32_t     vmdh_index = -1;
    int i;

    smsg_attr_vector_local = (gni_smsg_attr_t**)malloc(mysize * sizeof(gni_smsg_attr_t*));
    smsg_attr_vector_remote = (gni_smsg_attr_t**)malloc(mysize * sizeof(gni_smsg_attr_t*));
    smsg_connected_flag = (int*)malloc(sizeof(int)*mysize);
    for(i=0; i<mysize; i++) {
        smsg_connected_flag[i] = 0;
        smsg_attr_vector_local[i] = NULL;
        smsg_attr_vector_remote[i] = NULL;
    }
    if(mysize <=512)
    {
        SMSG_MAX_MSG = 4096;
    }else if (mysize <= 4096)
    {
        SMSG_MAX_MSG = 4096/mysize * 1024;
    }else if (mysize <= 16384)
    {
        SMSG_MAX_MSG = 512;
    }else {
        SMSG_MAX_MSG = 256;
    }

    send_smsg_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
    send_smsg_attr.mbox_maxcredit = SMSG_MAX_CREDIT;
    send_smsg_attr.msg_maxsize = SMSG_MAX_MSG;
    status = GNI_SmsgBufferSizeNeeded(&send_smsg_attr, &smsg_memlen);
    GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);

    mailbox_list = (dynamic_smsg_mailbox_t*)malloc(sizeof(dynamic_smsg_mailbox_t));
    mailbox_list->size = smsg_memlen*avg_smsg_connection;
    //mailbox_list->mailbox_base = malloc(mailbox_list->size);
    posix_memalign(&mailbox_list->mailbox_base, 64, mailbox_list->size);
    bzero(mailbox_list->mailbox_base, mailbox_list->size);
    mailbox_list->offset = 0;
    mailbox_list->next = 0;
    
    status = GNI_MemRegister(nic_hndl, (uint64_t)(mailbox_list->mailbox_base),
        mailbox_list->size, smsg_rx_cqh,
        GNI_MEM_READWRITE,   
        vmdh_index,
        &(mailbox_list->mem_hndl));
    //status = MEMORY_REGISTER(onesided_hnd, nic_hndl, mailbox_list->mailbox_base, mailbox_list->size, &(mailbox_list->mem_hndl), &omdh);
    GNI_RC_CHECK("MEMORY registration for smsg", status);

    status = GNI_EpCreate(nic_hndl, smsg_tx_cqh, &ep_hndl_unbound);
    GNI_RC_CHECK("Unbound EP", status);
    
    alloc_smsg_attr(&send_smsg_attr);

    status = GNI_EpPostDataWId (ep_hndl_unbound, &send_smsg_attr,  SMSG_ATTR_SIZE, &recv_smsg_attr, SMSG_ATTR_SIZE, myrank);
    GNI_RC_CHECK("post unbound datagram", status);

      /* always pre-connect to proc 0 */
    //if (myrank != 0) connect_to(0);
}

static void _init_static_smsg()
{
    gni_smsg_attr_t      *smsg_attr;
    gni_smsg_attr_t      remote_smsg_attr;
    gni_smsg_attr_t      *smsg_attr_vec;
    gni_mem_handle_t     my_smsg_mdh_mailbox;
    int      ret, i;
    gni_return_t status;
    uint32_t              vmdh_index = -1;
    mdh_addr_t            base_infor;
    mdh_addr_t            *base_addr_vec;
    char *env;

    if(mysize <=512)
    {
        SMSG_MAX_MSG = 1024;
    }else if (mysize <= 4096)
    {
        SMSG_MAX_MSG = 1024;
    }else if (mysize <= 16384)
    {
        SMSG_MAX_MSG = 512;
    }else {
        SMSG_MAX_MSG = 256;
    }
    
    env = getenv("CHARM_UGNI_SMSG_MAX_SIZE");
    if (env) SMSG_MAX_MSG = atoi(env);
    CmiAssert(SMSG_MAX_MSG > 0);

    smsg_attr = malloc(mysize * sizeof(gni_smsg_attr_t));
    
    smsg_attr[0].msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
    smsg_attr[0].mbox_maxcredit = SMSG_MAX_CREDIT;
    smsg_attr[0].msg_maxsize = SMSG_MAX_MSG;
    status = GNI_SmsgBufferSizeNeeded(&smsg_attr[0], &smsg_memlen);
    GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);
    ret = posix_memalign(&smsg_mailbox_base, 64, smsg_memlen*(mysize));
    CmiAssert(ret == 0);
    bzero(smsg_mailbox_base, smsg_memlen*(mysize));
    //if (myrank == 0) printf("Charm++> allocates %.2fMB for SMSG mailbox. \n", smsg_memlen*mysize/1e6);
    
    status = GNI_MemRegister(nic_hndl, (uint64_t)smsg_mailbox_base,
            smsg_memlen*(mysize), smsg_rx_cqh,
            GNI_MEM_READWRITE,   
            vmdh_index,
            &my_smsg_mdh_mailbox);

    GNI_RC_CHECK("GNI_GNI_MemRegister mem buffer", status);

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
    free(smsg_attr);

    status = GNI_SmsgSetMaxRetrans(nic_hndl, 4096);
    GNI_RC_CHECK("SmsgSetMaxRetrans Init", status);
} 

inline
static void _init_smsg()
{
    int i;

     smsg_msglist_index = (MSG_LIST_INDEX*)malloc(mysize*sizeof(MSG_LIST_INDEX));
     for(i =0; i<mysize; i++)
     {
        smsg_msglist_index[i].next = -1;
#if CMK_SMP
        smsg_msglist_index[i].sendSmsgBuf = PCQueueCreate();
#else
        smsg_msglist_index[i].sendSmsgBuf = 0; 
#endif
        
     }
     smsg_head_index = -1;
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

#if CMK_SMP && STEAL_MEMPOOL
void *steal_mempool_block(size_t *size, gni_mem_handle_t *mem_hndl)
{
    void *pool = NULL;
    int i, k;
    // check other ranks
    for (k=0; k<CmiMyNodeSize()+1; k++) {
        i = (CmiMyRank()+k)%CmiMyNodeSize();
        if (i==CmiMyRank()) continue;
        mempool_type *mptr = CpvAccessOther(mempool, i);
        CmiLock(mptr->mempoolLock);
        mempool_block *tail =  (mempool_block *)((char*)mptr + mptr->memblock_tail);
        if ((char*)tail == (char*)mptr) {     /* this is the only memblock */
            CmiUnlock(mptr->mempoolLock);
            continue;
        }
        mempool_header *header = (mempool_header*)((char*)tail + sizeof(mempool_block));
        if (header->size >= *size && header->size == tail->size - sizeof(mempool_block)) {
            /* search in the free list */
          mempool_header *free_header = mptr->freelist_head?(mempool_header*)((char*)mptr+mptr->freelist_head):NULL;
          mempool_header *current = free_header;
          while (current) {
            if (current->next_free == (char*)header-(char*)mptr) break;
            current = current->next_free?(mempool_header*)((char*)mptr + current->next_free):NULL;
          }
          if (current == NULL) {         /* not found in free list */
            CmiUnlock(mptr->mempoolLock);
            continue;
          }
printf("[%d:%d:%d] steal from %d tail: %p size: %d %d %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, tail, header->size, tail->size, sizeof(mempool_block));
            /* search the previous memblock, and remove the tail */
          mempool_block *ptr = (mempool_block *)mptr;
          while (ptr) {
            if (ptr->memblock_next == mptr->memblock_tail) break;
            ptr = ptr->memblock_next?(mempool_block *)((char*)mptr + ptr->memblock_next):NULL;
          }
          CmiAssert(ptr!=NULL);
          ptr->memblock_next = 0;
          mptr->memblock_tail = (char*)ptr - (char*)mptr;

            /* remove memblock from the free list */
          current->next_free = header->next_free;
          if (header == free_header) mptr->freelist_head = header->next_free;

          CmiUnlock(mptr->mempoolLock);

          pool = (void*)tail;
          *mem_hndl = tail->mem_hndl;
          *size = tail->size;
          return pool;
        }
        CmiUnlock(mptr->mempoolLock);
    }

      /* steal failed, deregister and free memblock now */
    int freed = 0;
    for (k=0; k<CmiMyNodeSize()+1; k++) {
        i = (CmiMyRank()+k)%CmiMyNodeSize();
        mempool_type *mptr = CpvAccessOther(mempool, i);
        if (i!=CmiMyRank()) CmiLock(mptr->mempoolLock);

        mempool_block *mempools_head = &(mptr->mempools_head);
        mempool_block *current = mempools_head;
        mempool_block *prev = NULL;

        while (current) {
          int isfree = 0;
          mempool_header *free_header = mptr->freelist_head?(mempool_header*)((char*)mptr+mptr->freelist_head):NULL;
printf("[%d:%d:%d] checking rank: %d ptr: %p size: %d wanted: %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, current, current->size, *size);
          mempool_header *cur = free_header;
          mempool_header *header;
          if (current != mempools_head) {
            header = (mempool_header*)((char*)current + sizeof(mempool_block));
             /* search in free list */
            if (header->size == current->size - sizeof(mempool_block)) {
              cur = free_header;
              while (cur) {
                if (cur->next_free == (char*)header-(char*)mptr) break;
                cur = cur->next_free?(mempool_header*)((char*)mptr + cur->next_free):NULL;
              }
              if (cur != NULL) isfree = 1;
            }
          }
          if (isfree) {
              /* remove from free list */
            cur->next_free = header->next_free;
            if (header == free_header) mptr->freelist_head = header->next_free;
             // deregister
            gni_return_t status = MEMORY_DEREGISTER(onesided_hnd, nic_hndl, &current->mem_hndl, &omdh);
            GNI_RC_CHECK("steal Mempool de-register", status);
            mempool_block *ptr = current;
            current = current->memblock_next?(mempool_block *)((char*)mptr+current->memblock_next):NULL;
            prev->memblock_next = current?(char*)current - (char*)mptr:0;
printf("[%d:%d:%d] free rank: %d ptr: %p size: %d wanted: %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, ptr, ptr->size, *size);
            freed += ptr->size;
            free(ptr);
             // try now
            if (freed > *size) {
              if (pool == NULL) {
                int ret = posix_memalign(&pool, ALIGNBUF, *size);
                CmiAssert(ret == 0);
              }
              status = MEMORY_REGISTER(onesided_hnd, nic_hndl, pool, *size,  mem_hndl, &omdh);
              if (status == GNI_RC_SUCCESS) {
                if (i!=CmiMyRank()) CmiUnlock(mptr->mempoolLock);
printf("[%d:%d:%d] GOT IT rank: %d wanted: %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, *size);
                return pool;
              }
printf("[%d:%d:%d] TRIED but fails: %d wanted: %d %d\n", CmiMyPe(), CmiMyNode(), CmiMyRank(), i, *size, status);
            }
          }
          else {
             prev = current;
             current = current->memblock_next?(mempool_block *)((char*)mptr+current->memblock_next):NULL;
          }
        }

        if (i!=CmiMyRank()) CmiUnlock(mptr->mempoolLock);
    }
      /* still no luck registering pool */
    if (pool) free(pool);
    return NULL;
}
#endif
static long long int total_mempool_size = 0;
static long long int total_mempool_calls = 0;
#if USE_LRTS_MEMPOOL
void *alloc_mempool_block(size_t *size, gni_mem_handle_t *mem_hndl, int expand_flag)
{
    void *pool;
    int ret;

    int default_size =  expand_flag? _expand_mem : _mempool_size;
    if (*size < default_size) *size = default_size;
    total_mempool_size += *size;
    total_mempool_calls += 1;
    ret = posix_memalign(&pool, ALIGNBUF, *size);
    if (ret != 0) {
#if CMK_SMP && STEAL_MEMPOOL
      pool = steal_mempool_block(size, mem_hndl);
      if (pool != NULL) return pool;
#endif
      printf("Charm++> can not allocate memory pool of size %.2fMB. \n", 1.0*(*size)/1024/1024);
      if (ret == ENOMEM)
        CmiAbort("alloc_mempool_block: out of memory.");
      else
        CmiAbort("alloc_mempool_block: posix_memalign failed");
    }
    /*
    gni_return_t status = MEMORY_REGISTER(onesided_hnd, nic_hndl, pool, *size,  mem_hndl, &omdh);
#if CMK_SMP && STEAL_MEMPOOL
    if(expand_flag && status != GNI_RC_SUCCESS) {
      free(pool);
      pool = steal_mempool_block(size, mem_hndl);
      if (pool != NULL) status = GNI_RC_SUCCESS;
    }
#endif
*/
  //  if(status != GNI_RC_SUCCESS)
    {
        //printf("[%d] Charm++> too much registering memory of %d bytes: Please try to use large page (module load craype-hugepages8m) or contact charm++ developer for help.[%lld, %lld]\n", CmiMyPe(), *size, total_mempool_size, total_mempool_calls);
        SetMemHndlZero((*mem_hndl));
    
   }
    //if(expand_flag)
    //    printf("[%d] Alloc more Memory pool of %d bytes: [mempool=%lld, calls=%lld]\n", CmiMyPe(), *size, total_mempool_size, total_mempool_calls);
    return pool;
}

void free_mempool_block(void *ptr, gni_mem_handle_t mem_hndl)
{
    if(!(IsMemHndlZero(mem_hndl)))
    {
        gni_return_t status = GNI_MemDeregister(nic_hndl, &mem_hndl);
        GNI_RC_CHECK("free_mempool_block Mempool de-register", status);
    }
    free(ptr);
}
#endif

void LrtsPreCommonInit(int everReturn){
#if USE_LRTS_MEMPOOL
    CpvInitialize(mempool_type*, mempool);
    CpvAccess(mempool) = mempool_init(_mempool_size, alloc_mempool_block, free_mempool_block);
#endif
}


FILE *debugLog = NULL;
void LrtsInit(int *argc, char ***argv, int *numNodes, int *myNodeID)
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
    char                   *env;

    //void (*local_event_handler)(gni_cq_entry_t *, void *)       = &LocalEventHandle;
    //void (*remote_smsg_event_handler)(gni_cq_entry_t *, void *) = &RemoteSmsgEventHandle;
    //void (*remote_bte_event_handler)(gni_cq_entry_t *, void *)  = &RemoteBteEventHandle;
   
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
  
#if !COMM_THREAD_SEND
    /* Currently, we only consider the case that comm. thread will only recv msgs */
    Cmi_smp_mode_setting = COMM_THREAD_ONLY_RECV;
#endif

    env = getenv("CHARM_UGNI_REMOTE_QUEUE_SIZE");
    if (env) REMOTE_QUEUE_ENTRIES = atoi(env);
    CmiGetArgInt(*argv,"+useRecvQueue", &REMOTE_QUEUE_ENTRIES);

    env = getenv("CHARM_UGNI_LOCAL_QUEUE_SIZE");
    if (env) LOCAL_QUEUE_ENTRIES = atoi(env);
    CmiGetArgInt(*argv,"+useSendQueue", &LOCAL_QUEUE_ENTRIES);

    useDynamicSMSG = CmiGetArgFlag(*argv, "+useDynamicSmsg");
    CmiGetArgIntDesc(*argv, "+smsgConnection", &avg_smsg_connection,"Initial number of SMSGS connection per code");
    if (avg_smsg_connection>mysize) avg_smsg_connection = mysize;
    //useStaticMSGQ = CmiGetArgFlag(*argv, "+useStaticMsgQ");
    
    if(myrank == 0)
    {
        printf("Charm++> Running on Gemini (GNI) with %d processes\n", mysize);
        printf("Charm++> %s SMSG\n", useDynamicSMSG?"Dynamic":"Static");
    }
#ifdef USE_ONESIDED
    onesided_init(NULL, &onesided_hnd);

    // this is a GNI test, so use the libonesided bypass functionality
    onesided_gni_bypass_get_nih(onesided_hnd, &nic_hndl);
    local_addr = gniGetNicAddress();
#else
    ptag = get_ptag();
    cookie = get_cookie();
#if 0
    modes = GNI_CDM_MODE_CQ_NIC_LOCAL_PLACEMENT;
#endif
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
    
    status = GNI_CqCreate(nic_hndl, LOCAL_QUEUE_ENTRIES, 0, GNI_CQ_NOBLOCK, NULL, NULL, &post_tx_cqh);
    GNI_RC_CHECK("GNI_CqCreate post (tx)", status);
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
#if CMK_SMP && !COMM_THREAD_SEND
    tx_cq_lock  = CmiCreateLock();
    rx_cq_lock  = CmiCreateLock();
#endif
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
        if (useDynamicSMSG)
          _init_dynamic_smsg();
        else
          _init_static_smsg();
        _init_smsg();
        PMI_Barrier();
    }

#if     USE_LRTS_MEMPOOL
    char *str;
    str = getenv("CHARM_UGNI_MEMPOOL_SIZE");
    if (str) _mempool_size = CmiReadSize(env);
    if (CmiGetArgStringDesc(*argv,"+useMemorypoolSize",&str,"Set the memory pool size")) 
        _mempool_size = CmiReadSize(str);

    if (myrank==0) printf("Charm++> memory pool size: %1.fMB\n", _mempool_size/1024.0/1024);
#endif

    /* init DMA buffer for medium message */

    //_init_DMA_buffer();
    
    free(MPID_UGNI_AllAddr);
#if CMK_SMP
    sendRdmaBuf = PCQueueCreate();
#else
    sendRdmaBuf = 0;
#endif
}

void* LrtsAlloc(int n_bytes, int header)
{
    void *ptr;
#if 0
    printf("\n[PE:%d]Alloc Lrts for bytes=%d, head=%d %d\n", CmiMyPe(), n_bytes, header, SMSG_MAX_MSG);
#endif
    if(n_bytes <= SMSG_MAX_MSG)
    {
        int totalsize = n_bytes+header;
        ptr = malloc(totalsize);
    }
    else {
        CmiAssert(header+sizeof(mempool_header) <= ALIGNBUF);
#if     USE_LRTS_MEMPOOL
        n_bytes = ALIGN64(n_bytes);
        if(n_bytes <= BIG_MSG)
        {
            char *res = mempool_malloc(CpvAccess(mempool), ALIGNBUF+n_bytes-sizeof(mempool_header), 1);
            mempool_header* mh = ((mempool_header*)(res-sizeof(mempool_header)));
            block_header *bh = (block_header*)(mh->mempool_ptr);
            mem_handle_t hh=  bh->mem_hndl;
            ptr = res - sizeof(mempool_header) + ALIGNBUF - header;
        }else 
        {
            //printf("$$$$ [%d] Large message  %d\n", myrank, n_bytes); 
            char *res = memalign(ALIGNBUF, n_bytes+ALIGNBUF);
            ptr = res + ALIGNBUF - header;

        }
#else
        n_bytes = ALIGN64(n_bytes);           /* make sure size if 4 aligned */
        char *res = memalign(ALIGNBUF, n_bytes+ALIGNBUF);
        ptr = res + ALIGNBUF - header;
#endif
    }
#if 0 
    printf("Done Alloc Lrts for bytes=%d, head=%d\n", n_bytes, header);
#endif
    return ptr;
}

void  LrtsFree(void *msg)
{
    int size = SIZEFIELD((char*)msg+sizeof(CmiChunkHeader));
    if (size <= SMSG_MAX_MSG)
      free(msg);
    else if(size>BIG_MSG)
    {
        free((char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);

    }else
    {
#if 0
        printf("[PE:%d] Free lrts for bytes=%d, ptr=%p\n", CmiMyPe(), size, (char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);
#endif
#if     USE_LRTS_MEMPOOL
#if CMK_SMP
        mempool_free_thread((char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF + sizeof(mempool_header));
#else
        mempool_free(CpvAccess(mempool), (char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF + sizeof(mempool_header));
#endif
#else
        free((char*)msg + sizeof(CmiChunkHeader) - ALIGNBUF);
#endif
    }
#if 0 
    printf("Done Free lrts for bytes=%d\n", size);
#endif
}

void LrtsExit()
{
    /* free memory ? */
#if USE_LRTS_MEMPOOL
    mempool_destroy(CpvAccess(mempool));
#endif
    PMI_Finalize();
    exit(0);
}

void LrtsDrainResources()
{
    if(mysize == 1) return;
    while (!SendBufferMsg()) {
        if (useDynamicSMSG)
            PumpDatagramConnection();
        PumpNetworkSmsg();
        //PumpNetworkRdmaMsgs();
        PumpLocalRdmaTransactions();
        SendRdmaMsg();
    }
    PMI_Barrier();
}

void LrtsAbort(const char *message) {
    printf("CmiAbort is calling on PE:%d\n", myrank);
    CmiPrintStackTrace(0);
    PMI_Abort(-1, message);
}

/**************************  TIMER FUNCTIONS **************************/
#if CMK_TIMER_USE_SPECIAL
/* MPI calls are not threadsafe, even the timer on some machines */
static CmiNodeLock  timerLock = 0;
static int _absoluteTime = 0;
static int _is_global = 0;
static struct timespec start_ts;

inline int CmiTimerIsSynchronized() {
    return 0;
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
            clock_gettime(CLOCK_MONOTONIC, &start_ts);
        }
    } else { /* we don't have a synchronous timer, set our own start time */
        CmiBarrier();
        CmiBarrier();
        CmiBarrier();
        clock_gettime(CLOCK_MONOTONIC, &start_ts);
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
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    return _absoluteTime?((double)(now_ts.tv_sec)+(double)now_ts.tv_nsec/1000000000.0)
        : (double)( now_ts.tv_sec - start_ts.tv_sec ) + (((double) now_ts.tv_nsec - (double) start_ts.tv_nsec)  / 1000000000.0);
}

double CmiWallTimer(void) {
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    return _absoluteTime?((double)(now_ts.tv_sec)+(double)now_ts.tv_nsec/1000000000.0)
        : ( now_ts.tv_sec - start_ts.tv_sec ) + ((now_ts.tv_nsec - start_ts.tv_nsec)  / 1000000000.0);
}

double CmiCpuTimer(void) {
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);
    return _absoluteTime?((double)(now_ts.tv_sec)+(double)now_ts.tv_nsec/1000000000.0)
        : (double)( now_ts.tv_sec - start_ts.tv_sec ) + (((double) now_ts.tv_nsec - (double) start_ts.tv_nsec)  / 1000000000.0);
}

#endif
/************Barrier Related Functions****************/

int CmiBarrier()
{
    int status;

#if CMK_SMP
    /* make sure all ranks reach here, otherwise comm threads may reach barrier ignoring other ranks  */
    CmiNodeAllBarrier();
    if (CmiMyRank() == CmiMyNodeSize())
#else
    if (CmiMyRank() == 0)
#endif
    {
        /**
         *  The call of CmiBarrier is usually before the initialization
         *  of trace module of Charm++, therefore, the START_EVENT
         *  and END_EVENT are disabled here. -Chao Mei
         */
        /*START_EVENT();*/
        status = PMI_Barrier();
        GNI_RC_CHECK("PMI_Barrier", status);
        /*END_EVENT(10);*/
    }
    CmiNodeAllBarrier();
    return status;

}
#if CMK_DIRECT
#include "machine-cmidirect.c"
#endif
#if CMK_PERSISTENT_COMM
#include "machine-persistent.c"
#endif


