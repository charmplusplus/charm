#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>

#include "converse.h"
#include "machine.h"
#include "pcqueue.h"

#include <bpcore/ppc450_inlines.h>
#include "dcmf.h"
#include "dcmf_multisend.h"

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* =======End of Definitions of Performance-Specific Macros =======*/

/* =======Beginning of Definitions of Msg Header Specific Macros =======*/
/* =======End of Definitions of Msg Header Specific Macros =======*/

/* =====Beginning of Definitions of Message-Corruption Related Macros=====*/
#define CMI_MAGIC(msg)                   ((CmiMsgHeaderBasic *)msg)->magic
#define CHARM_MAGIC_NUMBER               126

#if CMK_ERROR_CHECKING
static int checksum_flag = 0;
extern unsigned char computeCheckSum(unsigned char *data, int len);

#define CMI_SET_CHECKSUM(msg, len)      \
        if (checksum_flag)  {   \
          ((CmiMsgHeaderBasic *)msg)->cksum = 0;        \
          ((CmiMsgHeaderBasic *)msg)->cksum = computeCheckSum((unsigned char*)msg, len);        \
        }

#define CMI_CHECK_CHECKSUM(msg, len)    \
        if (checksum_flag)      \
          if (computeCheckSum((unsigned char*)msg, len) != 0)  { \
            printf("\n\n------------------------------\n\nReceiver %d size %d:", CmiMyPe(), len); \
	    { \
	    int count; \
            for(count = 0; count < len; count++) { \
                printf("%2x", msg[count]);                 \
            } \
            }                                             \
            printf("------------------------------\n\n"); \
            CmiAbort("Fatal error: checksum doesn't agree!\n"); \
          }
#else
#define CMI_SET_CHECKSUM(msg, len)
#define CMI_CHECK_CHECKSUM(msg, len)
#endif
/* =====End of Definitions of Message-Corruption Related Macros=====*/


/* =====Beginning of Declarations of Machine Specific Variables===== */
typedef struct ProcState {
    /* PCQueue      sendMsgBuf; */      /* per processor message sending queue */
    CmiNodeLock  recvLock;              /* for cs->recv */
    CmiNodeLock bcastLock;
} ProcState;

static ProcState  *procState;

volatile int msgQueueLen;
volatile int outstanding_recvs;

DCMF_Protocol_t  cmi_dcmf_short_registration __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_eager_registration __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_rzv_registration   __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_multicast_registration   __attribute__((__aligned__(16)));


typedef struct msg_list {
    char              * msg;
//    int                 size;
//    int                 destpe;
    int               * pelist;
//    DCMF_Callback_t     cb;
//    DCQuad              info __attribute__((__aligned__(16)));
    DCMF_Request_t      send __attribute__((__aligned__(16)));
} SMSG_LIST __attribute__((__aligned__(16)));

#define MAX_NUM_SMSGS   64
CpvDeclare(PCQueue, smsg_list_q);
static SMSG_LIST * smsg_allocate();
static void smsg_free (SMSG_LIST *smsg);

/* =====End of Declarations of Machine Specific Variables===== */


/* =====Beginning of Declarations of Machine Specific Functions===== */
/* Utility functions */
char *ALIGN_16(char *p) {
    return((char *)((((unsigned long)p)+0xf)&0xfffffff0));
}

void mysleep (int cycles) { /* approximate sleep command */
    unsigned long long start = DCMF_Timebase();
    unsigned long long end = start + cycles;
    while (start < end)
        start = DCMF_Timebase();
    return;
}
static void SendMsgsUntil(int);

/* ######Begining of Machine-specific RDMA related functions###### */
#define BGP_USE_AM_DIRECT 1
/* #define BGP_USE_RDMA_DIRECT 1 */
/* #define CMI_DIRECT_DEBUG 1 */
#if BGP_USE_AM_DIRECT

DCMF_Protocol_t  cmi_dcmf_direct_registration __attribute__((__aligned__(16)));
/** The receive side of a put implemented in DCMF_Send */

typedef struct {
    void *recverBuf;
    void (*callbackFnPtr)(void *);
    void *callbackData;
    DCMF_Request_t *DCMF_rq_t;
} dcmfDirectMsgHeader;

/* nothing for us to do here */
#if (DCMF_VERSION_MAJOR >= 2)
void direct_send_done_cb(void*nothing, DCMF_Error_t *err)
#else
void direct_send_done_cb(void*nothing)
#endif
{
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA send_done_cb\n", CmiMyPe());
#endif
}

DCMF_Callback_t  directcb;

void     direct_short_pkt_recv (void             * clientdata,
                                const DCQuad     * info,
                                unsigned           count,
                                unsigned           senderrank,
                                const char       * buffer,
                                const unsigned     sndlen) {
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA direct_short_pkt_recv\n", CmiMyPe());
#endif
    dcmfDirectMsgHeader *msgHead=  (dcmfDirectMsgHeader *) info;
    CmiMemcpy(msgHead->recverBuf, buffer, sndlen);
    (*(msgHead->callbackFnPtr))(msgHead->callbackData);
}


#if (DCMF_VERSION_MAJOR >= 2)
typedef void (*cbhdlr) (void *, DCMF_Error_t *);
#else
typedef void (*cbhdlr) (void *);
#endif

DCMF_Request_t * direct_first_pkt_recv_done (void              * clientdata,
        const DCQuad      * info,
        unsigned            count,
        unsigned            senderrank,
        const unsigned      sndlen,
        unsigned          * rcvlen,
        char             ** buffer,
        DCMF_Callback_t   * cb
                                            ) {
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA direct_first_pkt_recv_done\n", CmiMyPe());
#endif
    /* pull the data we need out of the header */
    *rcvlen=sndlen;
    dcmfDirectMsgHeader *msgHead=  (dcmfDirectMsgHeader *) info;
    cb->function= (cbhdlr)msgHead->callbackFnPtr;
    cb->clientdata=msgHead->callbackData;
    *buffer=msgHead->recverBuf;
    return msgHead->DCMF_rq_t;
}
#endif /* end of #if BGP_USE_AM_DIRECT */

#ifdef BGP_USE_RDMA_DIRECT
static struct DCMF_Callback_t dcmf_rdma_cb_ack;

DCMF_Protocol_t  cmi_dcmf_direct_put_registration __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_direct_get_registration __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_direct_rdma_registration __attribute__((__aligned__(16)));
/** The receive side of a DCMF_Put notification implemented in DCMF_Send */

typedef struct {
    void (*callbackFnPtr)(void *);
    void *callbackData;
} dcmfDirectRDMAMsgHeader;

#if (DCMF_VERSION_MAJOR >= 2)
void direct_send_rdma_done_cb(void*nothing, DCMF_Error_t *err)
#else
void direct_send_rdma_done_cb(void*nothing)
#endif
{
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA send_rdma_done_cb result %d\n", CmiMyPe());
#endif


}

DCMF_Callback_t  directcb;

void     direct_short_rdma_pkt_recv (void             * clientdata,
                                     const DCQuad     * info,
                                     unsigned           count,
                                     unsigned           senderrank,
                                     const char       * buffer,
                                     const unsigned     sndlen) {
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA direct_short_rdma_pkt_recv\n", CmiMyPe());
#endif
    dcmfDirectRDMAMsgHeader *msgHead=  (dcmfDirectRDMAMsgHeader *) info;
    (*(msgHead->callbackFnPtr))(msgHead->callbackData);
}

#if (DCMF_VERSION_MAJOR >= 2)
typedef void (*cbhdlr) (void *, DCMF_Error_t *);
#else
typedef void (*cbhdlr) (void *);
#endif

DCMF_Request_t * direct_first_rdma_pkt_recv_done (void              * clientdata,
        const DCQuad      * info,
        unsigned            count,
        unsigned            senderrank,
        const unsigned      sndlen,
        unsigned          * rcvlen,
        char             ** buffer,
        DCMF_Callback_t   * cb
                                                 ) {
    CmiAbort("direct_first_rdma_pkt_recv should not be called");
}
#endif /* end of #if BGP_USE_RDMA_DIRECT */
/* ######End of Machine-specific RDMA related functions###### */


/* ### Beginning of Communication-Op Related Functions ### */
/* The machine-specific send-related function */
#if (DCMF_VERSION_MAJOR >= 2)
static void send_done(void *data, DCMF_Error_t *err);
static void send_multi_done(void *data, DCMF_Error_t *err);
#else
static void send_done(void *data);
static void send_multi_done(void *data);
#endif
static CmiCommHandle MachineSpecificSendForDCMF(int destNode, int size, char *msg, int mode);
#define LrtsSendFunc MachineSpecificSendForDCMF

/* The machine-specific recv-related function (on the receiver side) */
#if (DCMF_VERSION_MAJOR >= 2)
static void recv_done(void *clientdata, DCMF_Error_t * err);
#else
static void recv_done(void *clientdata);
#endif
DCMF_Request_t * first_multi_pkt_recv_done (const DCQuad      * info,
        unsigned            count,
        unsigned            senderrank,
        const unsigned      sndlen,
        unsigned            connid,
        void              * clientdata,
        unsigned          * rcvlen,
        char             ** buffer,
        unsigned          * pw,
        DCMF_Callback_t   * cb
                                           );
DCMF_Request_t * first_pkt_recv_done (void              * clientdata,
                                      const DCQuad      * info,
                                      unsigned            count,
                                      unsigned            senderrank,
                                      const unsigned      sndlen,
                                      unsigned          * rcvlen,
                                      char             ** buffer,
                                      DCMF_Callback_t   * cb
                                     );

/* ### End of Communication-Op Related Functions ### */

/* ### Beginning of Machine-startup Related Functions ### */
static void MachineInitForDCMF(int *argc, char ***argv, int *numNodes, int *myNodeID);
#define LrtsInit MachineInitForDCMF

static void MachinePreCommonInitForDCMF(int everReturn);
static void MachinePostCommonInitForDCMF(int everReturn);
#define LrtsPreCommonInit MachinePreCommonInitForDCMF
#define LrtsPostCommonInit MachinePostCommonInitForDCMF
/* ### End of Machine-startup Related Functions ### */

/* ### Beginning of Machine-running Related Functions ### */
static void AdvanceCommunicationForDCMF();
#define LrtsAdvanceCommunication AdvanceCommunicationForDCMF

static void DrainResourcesForDCMF();
#define LrtsDrainResources DrainResourcesForDCMF

static void MachineExitForDCMF();
#define LrtsExit MachineExitForDCMF

/* ### End of Machine-running Related Functions ### */

/* ### Beginning of Idle-state Related Functions ### */

/* ### End of Idle-state Related Functions ### */

static void MachinePostNonLocalForDCMF();
#define LrtsPostNonLocal MachinePostNonLocalForDCMF

/* =====End of Declarations of Machine Specific Functions===== */

/**
 *  Macros that overwrites the common codes, such as
 *  CMK_SMP_NO_COMMTHD, NETWORK_PROGRESS_PERIOD_DEFAULT,
 *  USE_COMMON_SYNC_P2P, CMK_HAS_SIZE_IN_MSGHDR,
 *  CMK_OFFLOAD_BCAST_PROCESS etc.
 */
#define CMK_OFFLOAD_BCAST_PROCESS 1
#include "machine-common.h"
#include "machine-common.c"

/*######Beginning of functions related with Communication-Op functions ######*/

/* Utility functions */
static inline SMSG_LIST * smsg_allocate() {
    SMSG_LIST *smsg = (SMSG_LIST *)PCQueuePop(CpvAccess(smsg_list_q));
    if (smsg != NULL)
        return smsg;

    void * buf = malloc(sizeof(SMSG_LIST));
    assert(buf!=NULL);
    assert (((unsigned)buf & 0x0f) == 0);

    return (SMSG_LIST *) buf;
}

static inline void smsg_free (SMSG_LIST *smsg) {
    int size = PCQueueLength (CpvAccess(smsg_list_q));
    if (size < MAX_NUM_SMSGS)
        PCQueuePush (CpvAccess(smsg_list_q), (char *) smsg);
    else
        free (smsg);
}

static void SendMsgsUntil(int targetm) {
    while (msgQueueLen>targetm) {
#if CMK_SMP
        DCMF_CriticalSection_enter (0);
#endif

        while (DCMF_Messager_advance()>0);

#if CMK_SMP
        DCMF_CriticalSection_exit (0);
#endif
    }
}

/* Send functions */
/* The callback on sender side */
#if (DCMF_VERSION_MAJOR >= 2)
static void send_done(void *data, DCMF_Error_t *err)
#else
static void send_done(void *data)
#endif
/* send done callback: sets the smsg entry to done */
{
    SMSG_LIST *msg_tmp = (SMSG_LIST *)(data);
    CmiFree(msg_tmp->msg);
    smsg_free (msg_tmp);
    msgQueueLen--;
}

#if (DCMF_VERSION_MAJOR >= 2)
static void send_multi_done(void *data, DCMF_Error_t *err)
#else
static void send_multi_done(void *data)
#endif
/* send done callback: sets the smsg entry to done */
{
    SMSG_LIST *msg_tmp = (SMSG_LIST *)(data);
    CmiFree(msg_tmp->msg);
    free(msg_tmp->pelist);
    smsg_free(msg_tmp);
    msgQueueLen--;
}

/* The machine specific send function */
static CmiCommHandle MachineSpecificSendForDCMF(int destNode, int size, char *msg, int mode) {
    SMSG_LIST *msg_tmp = smsg_allocate(); //(SMSG_LIST *) malloc(sizeof(SMSG_LIST));
    //msg_tmp->destpe = destNode;
    //msg_tmp->size = size;
    msg_tmp->msg = msg;

    DCMF_Callback_t cb;
    DCQuad info;

    cb.function = send_done;
    cb.clientdata = msg_tmp;


#if CMK_ERROR_CHECKING
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    CMI_SET_CHECKSUM(msg, size);
#endif
    CMI_MSG_SIZE(msg) = size;

    //msg_tmp->cb.function = send_done;
    //msg_tmp->cb.clientdata   =   msg_tmp;

    DCMF_Protocol_t *protocol = NULL;

    if (size < 224)
        protocol = &cmi_dcmf_short_registration;
    else if (size < 2048)
        protocol = &cmi_dcmf_eager_registration;
    else
        protocol = &cmi_dcmf_rzv_registration;

#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif

    msgQueueLen ++;
    /*
     * Original one:
     *     DCMF_Send (protocol, &msg_tmp->send, msg_tmp->cb,
                   DCMF_MATCH_CONSISTENCY, msg_tmp->destpe,
                   msg_tmp->size, msg_tmp->msg, &msg_tmp->info, 1);
           Ref:http://dcmf.anl-external.org/docs/mpi:dcmfd/group__SEND.html
     */
    DCMF_Send (protocol, &msg_tmp->send, cb, DCMF_MATCH_CONSISTENCY,
               destNode, size, msg, &info, 0);

#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif

    return 0;
}

#define MAX_MULTICAST 128
DCMF_Opcode_t  CmiOpcodeList [MAX_MULTICAST];

void  machineMulticast(int npes, int *pelist, int size, char* msg) {
    CQdCreate(CpvAccess(cQdState), npes);

    CmiAssert (npes < MAX_MULTICAST);

#if CMK_ERROR_CHECKING
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    CMI_SET_CHECKSUM(msg, size);
#endif

    CMI_MSG_SIZE(msg) = size;

    SMSG_LIST *msg_tmp = smsg_allocate(); //(SMSG_LIST *) malloc(sizeof(SMSG_LIST));

    //msg_tmp->destpe    = -1;      //multicast operation
    //msg_tmp->size      = size * npes; //keep track of #bytes outstanding
    msg_tmp->msg       = msg;
    msg_tmp->pelist    = pelist;

    DCMF_Multicast_t  mcast_info __attribute__((__aligned__(16)));
    DCQuad info;

    mcast_info.registration   = & cmi_dcmf_multicast_registration;
    mcast_info.request        = & msg_tmp->send;
    mcast_info.cb_done.function    =   send_multi_done;
    mcast_info.cb_done.clientdata  =   msg_tmp;
    mcast_info.consistency    =   DCMF_MATCH_CONSISTENCY;
    mcast_info.connection_id  =   CmiMyPe();
    mcast_info.bytes          =   size;
    mcast_info.src            =   msg;
    mcast_info.nranks         =   npes;
    mcast_info.ranks          =   (unsigned *)pelist;
    mcast_info.opcodes        =   CmiOpcodeList;   //static list of MAX_MULTICAST entires with 0 in them
    mcast_info.flags          =   0;
    mcast_info.msginfo        =   &info;
    //mcast_info.count          =   1;
    mcast_info.count          =   0;

#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif
    msgQueueLen++;
    DCMF_Multicast (&mcast_info);

#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif
}

/* Recv functions */
/* The callback on the recv side */
#if (DCMF_VERSION_MAJOR >= 2)
static void recv_done(void *clientdata, DCMF_Error_t * err)
#else
static void recv_done(void *clientdata)
#endif
/* recv done callback: push the recved msg to recv queue */
{

    char *msg = (char *) clientdata;

    /*printf ("NODE[%d] Recv message done with msg rank %d\n", CmiMyNode(), CMI_DEST_RANK(msg));*/
    MACHSTATE3(2,"[%d] recv_done begin with msg %p size=%d { ", CmiMyNode(), msg, CMI_MSG_SIZE(msg));
#if CMK_ERROR_CHECKING
    int sndlen = CMI_MSG_SIZE(msg);
    CMI_CHECK_CHECKSUM(msg, sndlen);
    if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
        CmiAbort("Charm++ Warning: Non Charm++ Message Received. \n");
        return;
    }
#endif

    handleOneRecvedMsg(CMI_MSG_SIZE(msg), msg);

    outstanding_recvs--;
    MACHSTATE(2,"} recv_done end ");
    return;
}

void short_pkt_recv (void             * clientdata,
                     const DCQuad     * info,
                     unsigned           count,
                     unsigned           senderrank,
                     const char       * buffer,
                     const unsigned     sndlen) {
    outstanding_recvs ++;
    int alloc_size = sndlen;

    char * new_buffer = (char *)CmiAlloc(alloc_size);
    CmiMemcpy (new_buffer, buffer, sndlen);

#if (DCMF_VERSION_MAJOR >= 2)
    recv_done (new_buffer, NULL);
#else
    recv_done (new_buffer);
#endif
}

DCMF_Request_t * first_multi_pkt_recv_done (const DCQuad      * info,
        unsigned            count,
        unsigned            senderrank,
        const unsigned      sndlen,
        unsigned            connid,
        void              * clientdata,
        unsigned          * rcvlen,
        char             ** buffer,
        unsigned          * pw,
        DCMF_Callback_t   * cb
                                           ) {
    outstanding_recvs ++;
    int alloc_size = sndlen + sizeof(DCMF_Request_t) + 16;
    /*printf ("%d: Receiving message %d bytes from %d\n", CmiMyPe(), sndlen, senderrank);*/
    /* printf ("Receiving %d bytes\n", sndlen); */
    *rcvlen = sndlen;  /* to avoid malloc(0) which might return NULL */

    *buffer = (char *)CmiAlloc(alloc_size);
    cb->function = recv_done;
    cb->clientdata = *buffer;

    *pw  = 0x7fffffff;
    return (DCMF_Request_t *) ALIGN_16(*buffer + sndlen);
}

DCMF_Request_t * first_pkt_recv_done (void              * clientdata,
                                      const DCQuad      * info,
                                      unsigned            count,
                                      unsigned            senderrank,
                                      const unsigned      sndlen,
                                      unsigned          * rcvlen,
                                      char             ** buffer,
                                      DCMF_Callback_t   * cb
                                     ) {
    outstanding_recvs ++;
    int alloc_size = sndlen + sizeof(DCMF_Request_t) + 16;
    /* printf ("%d: Receiving message %d bytes from %d\n", CmiMyPe(), sndlen, senderrank);*/
    /* printf ("Receiving %d bytes\n", sndlen); */
    *rcvlen = sndlen;  /* to avoid malloc(0) which might return NULL */

    *buffer = (char *)CmiAlloc(alloc_size);
    cb->function = recv_done;
    cb->clientdata = *buffer;

    return (DCMF_Request_t *) ALIGN_16(*buffer + sndlen);
}

#if 0
/* -----------------------------------------
 * Rectangular broadcast implementation
 * -----------------------------------------
 */
unsigned int *ranklist;
BGTsC_t        barrier;
#define MAX_COMM  256
static void * comm_table [MAX_COMM];

typedef struct rectbcast_msg {
    BGTsRC_t           request;
    DCMF_Callback_t    cb;
    char              *msg;
} RectBcastInfo;


static void bcast_done (void *data) {
    RectBcastInfo *rinfo = (RectBcastInfo *) data;
    CmiFree (rinfo->msg);
    free (rinfo);
}

static  void *   getRectBcastRequest (unsigned comm) {
    return comm_table [comm];
}


static  void *  bcast_recv     (unsigned               root,
                                unsigned               comm,
                                const unsigned         sndlen,
                                unsigned             * rcvlen,
                                char                ** rcvbuf,
                                DCMF_Callback_t      * const cb) {

    int alloc_size = sndlen + sizeof(BGTsRC_t) + 16;

    *rcvlen = sndlen;  /* to avoid malloc(0) which might
                                   return NULL */

    *rcvbuf       =  (char *)CmiAlloc(alloc_size);
    cb->function  =   recv_done;
    cb->clientdata = *rcvbuf;

    return (BGTsRC_t *) ALIGN_16 (*rcvbuf + sndlen);

}


extern void bgl_machine_RectBcast (unsigned                 commid,
                                   const char             * sndbuf,
                                   unsigned                 sndlen) {
    RectBcastInfo *rinfo  =   (RectBcastInfo *) malloc (sizeof(RectBcastInfo));
    rinfo->cb.function    =   bcast_done;
    rinfo->cb.clientdata  =   rinfo;

    BGTsRC_AsyncBcast_start (commid, &rinfo->request, &rinfo->cb, sndbuf, sndlen);

}

extern void        bgl_machine_RectBcastInit  (unsigned               commID,
        const BGTsRC_Geometry_t* geometry) {

    CmiAssert (commID < 256);
    CmiAssert (comm_table [commID] == NULL);

    BGTsRC_t *request =  (BGTsRC_t *) malloc (sizeof (BGTsRC_t));
    comm_table [commID] = request;

    BGTsRC_AsyncBcast_init  (request, commID,  geometry);
}

/*--------------------------------------------------------------
 *----- End Rectangular Broadcast Implementation ---------------
 *--------------------------------------------------------------*/
#endif


/*######End of functions related with Communication-Op functions ######*/


/* ######Beginning of functions related with communication progress ###### */
static INLINE_KEYWORD void AdvanceCommunicationForDCMF() {
#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif

    while (DCMF_Messager_advance()>0);
    //DCMF_Messager_advance();

#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif
}
/* ######End of functions related with communication progress ###### */

static void MachinePostNonLocalForDCMF() {
    /* None here */
}

/* Network progress function is used to poll the network when for
   messages. This flushes receive buffers on some  implementations*/
#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl() {
    AdvanceCommunicationForDCMF();
#if CMK_IMMEDIATE_MSG
    CmiHandleImmediate();
#endif
}
#endif

/* ######Beginning of functions related with exiting programs###### */
static void DrainResourcesForDCMF() {
    while (msgQueueLen > 0 || outstanding_recvs > 0) {
        AdvanceCommunicationForDCMF();
    }
}

static void MachineExitForDCMF() {
    DCMF_Messager_finalize();
    exit(EXIT_SUCCESS);
}
/* ######End of functions related with exiting programs###### */


/* ######Beginning of functions related with starting programs###### */
/**
 *  Obtain the number of nodes, my node id, and consuming machine layer
 *  specific arguments
 */
static void MachineInitForDCMF(int *argc, char ***argv, int *numNodes, int *myNodeID) {

    DCMF_Messager_initialize();

#if CMK_SMP
    DCMF_Configure_t  config_in, config_out;
    config_in.thread_level= DCMF_THREAD_MULTIPLE;
    config_in.interrupts  = DCMF_INTERRUPTS_OFF;

    DCMF_Messager_configure(&config_in, &config_out);
    //assert (config_out.thread_level == DCMF_THREAD_MULTIPLE); //not supported in vn mode
#endif

    DCMF_Send_Configuration_t short_config, eager_config, rzv_config;


    short_config.protocol      = DCMF_DEFAULT_SEND_PROTOCOL;
    short_config.cb_recv_short = short_pkt_recv;
    short_config.cb_recv       = first_pkt_recv_done;

#if (DCMF_VERSION_MAJOR >= 3)
    short_config.network  = DCMF_DEFAULT_NETWORK;
#elif (DCMF_VERSION_MAJOR == 2)
    short_config.network  = DCMF_DefaultNetwork;
#endif

    eager_config.protocol      = DCMF_DEFAULT_SEND_PROTOCOL;
    eager_config.cb_recv_short = short_pkt_recv;
    eager_config.cb_recv       = first_pkt_recv_done;
#if (DCMF_VERSION_MAJOR >= 3)
    eager_config.network  = DCMF_DEFAULT_NETWORK;
#elif (DCMF_VERSION_MAJOR == 2)
    eager_config.network  = DCMF_DefaultNetwork;
#endif

#ifdef  OPT_RZV
#warning "Enabling Optimize Rzv"
    rzv_config.protocol        = DCMF_RZV_SEND_PROTOCOL;
#else
    rzv_config.protocol        = DCMF_DEFAULT_SEND_PROTOCOL;
#endif
    rzv_config.cb_recv_short   = short_pkt_recv;
    rzv_config.cb_recv         = first_pkt_recv_done;
#if (DCMF_VERSION_MAJOR >= 3)
    rzv_config.network  = DCMF_DEFAULT_NETWORK;
#elif (DCMF_VERSION_MAJOR == 2)
    rzv_config.network  = DCMF_DefaultNetwork;
#endif

    DCMF_Send_register (&cmi_dcmf_short_registration, &short_config);
    DCMF_Send_register (&cmi_dcmf_eager_registration, &eager_config);
    DCMF_Send_register (&cmi_dcmf_rzv_registration,   &rzv_config);

#ifdef BGP_USE_AM_DIRECT
    DCMF_Send_Configuration_t direct_config;
    direct_config.protocol      = DCMF_DEFAULT_SEND_PROTOCOL;
    direct_config.cb_recv_short = direct_short_pkt_recv;
    direct_config.cb_recv       = direct_first_pkt_recv_done;
#if (DCMF_VERSION_MAJOR >= 3)
    direct_config.network  = DCMF_DEFAULT_NETWORK;
#elif (DCMF_VERSION_MAJOR == 2)
    direct_config.network  = DCMF_DefaultNetwork;
#endif
    DCMF_Send_register (&cmi_dcmf_direct_registration,   &direct_config);
    directcb.function=direct_send_done_cb;
    directcb.clientdata=NULL;
#endif

#ifdef BGP_USE_RDMA_DIRECT
    /* notification protocol */
    DCMF_Send_Configuration_t direct_rdma_config;
    direct_rdma_config.protocol      = DCMF_DEFAULT_SEND_PROTOCOL;
    direct_rdma_config.cb_recv_short = direct_short_rdma_pkt_recv;
    direct_rdma_config.cb_recv       = direct_first_rdma_pkt_recv_done;
#if (DCMF_VERSION_MAJOR >= 3)
    direct_rdma_config.network  = DCMF_DEFAULT_NETWORK;
#elif (DCMF_VERSION_MAJOR == 2)
    direct_rdma_config.network  = DCMF_DefaultNetwork;
#endif
    DCMF_Send_register (&cmi_dcmf_direct_rdma_registration,   &direct_rdma_config);
    directcb.function=direct_send_rdma_done_cb;
    directcb.clientdata=NULL;
    /* put protocol */
    DCMF_Put_Configuration_t put_configuration = { DCMF_DEFAULT_PUT_PROTOCOL };
    DCMF_Put_register (&cmi_dcmf_direct_put_registration, &put_configuration);
    DCMF_Get_Configuration_t get_configuration = { DCMF_DEFAULT_GET_PROTOCOL };
    DCMF_Get_register (&cmi_dcmf_direct_get_registration, &get_configuration);

#endif
    //fprintf(stderr, "Initializing Eager Protocol\n");

    *numNodes = DCMF_Messager_size();
    *myNodeID = DCMF_Messager_rank();

    CmiBarrier();
    CmiBarrier();
    CmiBarrier();

    /* NOTE: the following codes requires #PEs, which is not available
     * until this function finishes. And it allocate O(p) space */
    int totalPEs = _Cmi_mynodesize * (*numNodes);
    DCMF_Multicast_Configuration_t mconfig;
    mconfig.protocol = DCMF_MEMFIFO_DMA_MSEND_PROTOCOL;
    mconfig.cb_recv  = first_multi_pkt_recv_done;
    mconfig.clientdata = NULL;
    mconfig.connectionlist = (void **) malloc (totalPEs * sizeof(unsigned long));
    mconfig.nconnections = totalPEs;
    DCMF_Multicast_register(&cmi_dcmf_multicast_registration, &mconfig);

    int actualNodeSize = _Cmi_mynodesize;
#if !CMK_SMP_NO_COMMTHD
    actualNodeSize++; //considering the extra comm thread
#endif
    int i;
    procState = (ProcState *)CmiAlloc((actualNodeSize) * sizeof(ProcState));
    for (i=0; i<actualNodeSize; i++) {
        /*    procState[i].sendMsgBuf = PCQueueCreate();   */
        procState[i].recvLock = CmiCreateLock();
        procState[i].bcastLock = CmiCreateLock();
    }

    /* checksum flag */
    if (CmiGetArgFlag(*argv,"+checksum")) {
#if CMK_ERROR_CHECKING
        checksum_flag = 1;
        if (*myNodeID == 0) CmiPrintf("Charm++: CheckSum checking enabled! \n");
#else
        if (*myNodeID == 0) CmiPrintf("Charm++: +checksum ignored in optimized version! \n");
#endif
    }

}

static void MachinePreCommonInitForDCMF(int everReturn) {
    CpvInitialize(PCQueue, smsg_list_q);
    CpvAccess(smsg_list_q) = PCQueueCreate();
}

static void MachinePostCommonInitForDCMF(int everReturn) {
#if !CMK_SMP || CMK_SMP_NO_COMMTHD
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
#endif

    CmiBarrier();
}
/* ######End of functions related with starting programs###### */

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message) {
    CmiError("------------- Processor %d Exiting: Called CmiAbort ------------\n"
             "{snd:%d,rcv:%d} Reason: %s\n",CmiMyPe(),
             msgQueueLen, outstanding_recvs, message);

#if 0
    /* Since it's a abort, why bother to drain the resources? The system
     * should clean it self
     */
    /* FIXME: what happens in the SMP mode??? */
    DrainResourcesForDCMF();
#endif
    assert(0);
}


/*********** Beginning of MULTICAST/VECTOR SENDING FUNCTIONS **************/
/*

 * In relations to some flags, some other delivery functions may be needed.
 */

#if !CMK_MULTICAST_LIST_USE_COMMON_CODE

void CmiSyncListSendFn(int npes, int *pes, int size, char *msg) {
    char *copymsg = CopyMsg(msg, size);
    CmiFreeListSendFn(npes, pes, size, copymsg);
}

/* This optimized multicast only helps NAMD when #atoms/CPU is
 * less than 10 according to Sameer Kumar. So it is off in
 * default.
 */
#define OPTIMIZED_MULTICAST  0

#if OPTIMIZED_MULTICAST
#warning "Using Optimized Multicast"
#endif

void CmiFreeListSendFn(int npes, int *pes, int size, char *msg) {
    CmiAssert(npes>=1);
    if (npes==1) {
        CmiFreeSendFn(pes[0], size, msg);
        return;
    }

    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeListSendFn on comm thd on node %d\n", CmiMyNode());
    //printf("%d: In Free List Send Fn\n", CmiMyPe());

    int i;
#if OPTIMIZED_MULTICAST
    int *newpelist = (int *)malloc(sizeof(int)*npes);
    int new_npes = npes;
    memcpy(newpelist, pes, sizeof(int)*npes);
#if CMK_SMP
    new_npes = 0;
    for (i=0; i<npes; i++) {
        if (CmiNodeOf(pes[i]) == CmiMyNode()) {
            CmiSyncSend(pes[i], size, msg);
        } else {
            newpelist[new_npes++] = pes[i];
        }
    }
    if (new_npes == 0) {
        CmiFree(msg);
        return;
    }
#endif

    CMI_SET_BROADCAST_ROOT(msg,0);
#if !CMK_SMP
    CMI_DEST_RANK(msg) = 0;
#else
#error optimized multicast should not be enabled in SMP mode
#endif

    CQdCreate(CpvAccess(cQdState), new_npes);
    machineMulticast (new_npes, newpelist, size, msg);
#else /* non-optimized multicast */

    for (i=0; i<npes-1; i++) {
#if !CMK_SMP
        CmiReference(msg);
        CmiFreeSendFn(pes[i], size, msg);
#else
    CmiSyncSend(pes[i], size, msg);
#endif
    }
    CmiFreeSendFn(pes[npes-1], size, msg);
#endif /* end of #if OPTIMIZED_MULTICAST */
}
#endif /* end of #if !CMK_MULTICAST_LIST_USE_COMMON_CODE */

/*********** End of MULTICAST/VECTOR SENDING FUNCTIONS **************/

/**************************  TIMER FUNCTIONS **************************/

/************Barrier Related Functions****************/
/* Barrier related functions */
/*TODO: does DCMF provide any Barrrier related functions ??? --Chao Mei */
/* Barrier needs to be implemented!!! -Chao Mei */
/* These two barriers are only needed by CmiTimerInit to synchronize all the
   threads. They do not need to provide a general barrier. */
int CmiBarrier() {
    return 0;
}
int CmiBarrierZero() {
    return 0;
}

#include "manytomany.c"

/*********************************************************************************************
This section is for CmiDirect. This is a variant of the  persistent communication in which
the user can transfer data between processors without using Charm++ messages. This lets the user
send and receive data from the middle of his arrays without any copying on either send or receive
side
*********************************************************************************************/


#ifdef BGP_USE_AM_DIRECT

#include "cmidirect.h"

/* We can avoid a receiver side lookup by just sending the whole shebang.
   DCMF header is in units of quad words (16 bytes), so we'd need less than a
   quad word for the handle if we just sent that and did a lookup. Or exactly
   2 quad words for the buffer pointer, callback pointer, callback
   data pointer, and DCMF_Request_t pointer with no lookup.

   Since CmiDirect is generally going to be used for messages which aren't
   tiny, the extra 16 bytes is not likely to impact performance noticably and
   not having to lookup handles in tables simplifies the code enormously.

   EJB   2008/4/2
*/


/**
 To be called on the receiver to create a handle and return its number
**/
struct infiDirectUserHandle CmiDirect_createHandle(int senderNode,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData,double initialValue) {
    /* with two-sided primitives we just bundle the buffer and callback info into the handle so the sender can remind us about it later. */
    struct infiDirectUserHandle userHandle;
    userHandle.handle=1; /* doesn't matter on BG/P*/
    userHandle.senderNode=senderNode;
    userHandle.recverNode=_Cmi_mynode;
    userHandle.recverBufSize=recvBufSize;
    userHandle.recverBuf=recvBuf;
    userHandle.initialValue=initialValue;
    userHandle.callbackFnPtr=callbackFnPtr;
    userHandle.callbackData=callbackData;
    userHandle.DCMF_rq_trecv=(DCMF_Request_t *) ALIGN_16(CmiAlloc(sizeof(DCMF_Request_t)+16));
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA create addr %p %d callback %p callbackdata %p\n",CmiMyPe(),userHandle.recverBuf,userHandle.recverBufSize, userHandle.callbackFnPtr, userHandle.callbackData);
#endif
    return userHandle;
}

/****
 To be called on the sender to attach the sender's buffer to this handle
******/

void CmiDirect_assocLocalBuffer(struct infiDirectUserHandle *userHandle,void *sendBuf,int sendBufSize) {

    /* one-sided primitives would require registration of memory */

    /* with two-sided primitives we just record the sender buf in the handle */
    userHandle->senderBuf=sendBuf;
    CmiAssert(sendBufSize==userHandle->recverBufSize);
    userHandle->DCMF_rq_tsend = (DCMF_Request_t *) ALIGN_16(CmiAlloc(sizeof(DCMF_Request_t)+16));
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA assoc addr %p %d to receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,sendBufSize, userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif

}

/****
To be called on the sender to do the actual data transfer
******/
void CmiDirect_put(struct infiDirectUserHandle *userHandle) {
    /** invoke a DCMF_Send with the direct callback */
    DCMF_Protocol_t *protocol = NULL;
    protocol = &cmi_dcmf_direct_registration;
    /* local copy */
    CmiAssert(userHandle->recverBuf!=NULL);
    CmiAssert(userHandle->senderBuf!=NULL);
    CmiAssert(userHandle->recverBufSize>0);
    if (userHandle->recverNode== _Cmi_mynode) {
#if CMI_DIRECT_DEBUG
        CmiPrintf("[%d] RDMA local put addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif

        CmiMemcpy(userHandle->recverBuf,userHandle->senderBuf,userHandle->recverBufSize);
        (*(userHandle->callbackFnPtr))(userHandle->callbackData);
    } else {
        dcmfDirectMsgHeader msgHead;
        msgHead.recverBuf=userHandle->recverBuf;
        msgHead.callbackFnPtr=userHandle->callbackFnPtr;
        msgHead.callbackData=userHandle->callbackData;
        msgHead.DCMF_rq_t=(DCMF_Request_t *) userHandle->DCMF_rq_trecv;
#if CMK_SMP
        DCMF_CriticalSection_enter (0);
#endif
#if CMI_DIRECT_DEBUG
        CmiPrintf("[%d] RDMA put addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif
        DCMF_Send (protocol,
                   (DCMF_Request_t *) userHandle->DCMF_rq_tsend,
                   directcb, DCMF_MATCH_CONSISTENCY, userHandle->recverNode,
                   userHandle->recverBufSize, userHandle->senderBuf,
                   (struct DCQuad *) &(msgHead), 2);

#if CMK_SMP
        DCMF_CriticalSection_exit (0);
#endif
    }
}

void CmiDirect_get(struct infiDirectUserHandle *userHandle) {
    CmiAbort("Not Implemented, switch to #define BGP_USE_RDMA_DIRECT");
}

/**** up to the user to safely call this */
void CmiDirect_deassocLocalBuffer(struct infiDirectUserHandle *userHandle) {
    CmiAssert(userHandle->senderNode==_Cmi_mynode);
#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif
    CmiFree(userHandle->DCMF_rq_tsend);
#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif

}

/**** up to the user to safely call this */
void CmiDirect_destroyHandle(struct infiDirectUserHandle *userHandle) {
    CmiAssert(userHandle->recverNode==_Cmi_mynode);
#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif
    CmiFree(userHandle->DCMF_rq_trecv);

#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif
}


/**** Should not be called the first time *********/
void CmiDirect_ready(struct infiDirectUserHandle *userHandle) {
    /* no op on BGP */
}

/**** Should not be called the first time *********/
void CmiDirect_readyPollQ(struct infiDirectUserHandle *userHandle) {
    /* no op on BGP */
}

/**** Should not be called the first time *********/
void CmiDirect_readyMark(struct infiDirectUserHandle *userHandle) {
    /* no op on BGP */
}

#endif /* BGP_USE_AM_DIRECT*/

#ifdef BGP_USE_RDMA_DIRECT

#include "cmidirect.h"

/*
   Notification protocol passes callback function and data in a single
   quadword.  This occurs in a message triggered by the sender side ack
   callback and therefore has higher latency than polling, but is guaranteed
   to be semantically correct.  The latency for a single packet that isn't
   hitting charm/converse should be pretty minimal, but you could run into
   sender side progress issues.  The alternative of polling on the out of band
   byte scheme creates correctness issues in that the data really has to be
   out of band and you rely on the buffer being written in order.  It also has
   annoying polling issues.  A third scheme could add a second put to a
   control region to poll upon and force sequential consistency between
   puts. Its not really clear that this would be faster or avoid the progress
   issue since you run into the same issues to enforce that sequential
   consistency.

   EJB   2011/1/20
*/


/* local function to use the ack as our signal to send a remote notify */
static void CmiNotifyRemoteRDMA(void *handle, struct DCMF_Error_t *error) {
    struct infiDirectUserHandle *userHandle= (struct infiDirectUserHandle *) handle;
    dcmfDirectRDMAMsgHeader msgHead;
    msgHead.callbackFnPtr=userHandle->callbackFnPtr;
    msgHead.callbackData=userHandle->callbackData;
#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA notify put addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p \n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif
    DCMF_Result res=DCMF_Send (&cmi_dcmf_direct_rdma_registration,
                               userHandle->DCMF_rq_tsend,
                               directcb, DCMF_MATCH_CONSISTENCY, userHandle->recverNode,
                               sizeof(dcmfDirectRDMAMsgHeader),

                               userHandle->DCMF_notify_buf,
                               (struct DCQuad *) &(msgHead), 1);
//    CmiAssert(res==DCMF_SUCCESS);
#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif
}

/**
 To be called on the receiver to create a handle and return its number
**/


struct infiDirectUserHandle CmiDirect_createHandle(int senderNode,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData,double initialValue) {
    /* one-sided primitives require registration of memory */
    struct infiDirectUserHandle userHandle;
    size_t numbytesRegistered=0;
    DCMF_Result regresult=DCMF_Memregion_create( &userHandle.DCMF_recverMemregion,
                          &numbytesRegistered,
                          recvBufSize,
                          recvBuf,
                          0);
    CmiAssert(numbytesRegistered==recvBufSize);
    CmiAssert(regresult==DCMF_SUCCESS);


    userHandle.handle=1; /* doesn't matter on BG/P*/
    userHandle.senderNode=senderNode;
    userHandle.recverNode=_Cmi_mynode;
    userHandle.recverBufSize=recvBufSize;
    userHandle.recverBuf=recvBuf;
    userHandle.initialValue=initialValue;
    userHandle.callbackFnPtr=callbackFnPtr;
    userHandle.callbackData=callbackData;
    userHandle.DCMF_rq_trecv=(DCMF_Request_t *) ALIGN_16(CmiAlloc(sizeof(DCMF_Request_t)+16));
#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA create addr %p %d callback %p callbackdata %p\n",CmiMyPe(),userHandle.recverBuf,userHandle.recverBufSize, userHandle.callbackFnPtr, userHandle.callbackData);
#endif
    return userHandle;
}

/****
 To be called on the sender to attach the sender's buffer to this handle
******/

void CmiDirect_assocLocalBuffer(struct infiDirectUserHandle *userHandle,void *sendBuf,int sendBufSize) {
    /* one-sided primitives would require registration of memory */
    userHandle->senderBuf=sendBuf;
    CmiAssert(sendBufSize==userHandle->recverBufSize);
    userHandle->DCMF_rq_tsend =(DCMF_Request_t *) ALIGN_16(CmiAlloc(sizeof(DCMF_Request_t)+16));
    size_t numbytesRegistered=0;  // set as return value from create
    userHandle->DCMF_notify_buf=ALIGN_16(CmiAlloc(sizeof(DCMF_Request_t)+32));
    userHandle->DCMF_notify_cb.function=CmiNotifyRemoteRDMA;
    userHandle->DCMF_notify_cb.clientdata=userHandle;
    DCMF_Result regresult=DCMF_Memregion_create( &userHandle->DCMF_senderMemregion,
                          &numbytesRegistered,
                          sendBufSize,
                          sendBuf,
                          0);
    CmiAssert(numbytesRegistered==sendBufSize);
    CmiAssert(regresult==DCMF_SUCCESS);

#if CMI_DIRECT_DEBUG
    CmiPrintf("[%d] RDMA assoc addr %p %d to receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,sendBufSize, userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif

}


/****
To be called on the sender to do the actual data transfer
******/
void CmiDirect_put(struct infiDirectUserHandle *userHandle) {
    /** invoke a DCMF_Put with the direct callback */

    CmiAssert(userHandle->recverBuf!=NULL);
    CmiAssert(userHandle->senderBuf!=NULL);
    CmiAssert(userHandle->recverBufSize>0);
    if (userHandle->recverNode== _Cmi_mynode) {     /* local copy */
#if CMI_DIRECT_DEBUG
        CmiPrintf("[%d] RDMA local put addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif

        CmiMemcpy(userHandle->recverBuf,userHandle->senderBuf,userHandle->recverBufSize);
        (*(userHandle->callbackFnPtr))(userHandle->callbackData);
    } else {
#if CMK_SMP
        DCMF_CriticalSection_enter (0);
#endif
#if CMI_DIRECT_DEBUG
        CmiPrintf("[%d] RDMA put addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif
        DCMF_Result
        Res= DCMF_Put(&cmi_dcmf_direct_put_registration,
                      userHandle->DCMF_rq_tsend,
                      directcb, DCMF_RELAXED_CONSISTENCY,
                      userHandle->recverNode,
                      userHandle->recverBufSize,
                      &userHandle->DCMF_senderMemregion,
                      &userHandle->DCMF_recverMemregion,
                      0, /* offsets are zero */
                      0,
                      userHandle->DCMF_notify_cb
                     );
        CmiAssert(Res==DCMF_SUCCESS);
#if CMK_SMP
        DCMF_CriticalSection_exit (0);
#endif
    }
}

/****
To be called on the receiver to initiate the actual data transfer
******/
void CmiDirect_get(struct infiDirectUserHandle *userHandle) {
    /** invoke a DCMF_Get with the direct callback */

    CmiAssert(userHandle->recverBuf!=NULL);
    CmiAssert(userHandle->senderBuf!=NULL);
    CmiAssert(userHandle->recverBufSize>0);
    if (userHandle->recverNode== _Cmi_mynode) {     /* local copy */
#if CMI_DIRECT_DEBUG
        CmiPrintf("[%d] RDMA local get addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif

        CmiMemcpy(userHandle->senderBuf,userHandle->recverBuf,userHandle->recverBufSize);
        (*(userHandle->callbackFnPtr))(userHandle->callbackData);
    } else {
        struct DCMF_Callback_t done_cb;
        done_cb.function=userHandle->callbackFnPtr;
        done_cb.clientdata=userHandle->callbackData;
#if CMK_SMP
        DCMF_CriticalSection_enter (0);
#endif
#if CMI_DIRECT_DEBUG
        CmiPrintf("[%d] RDMA get addr %p %d to recverNode %d receiver addr %p callback %p callbackdata %p\n",CmiMyPe(),userHandle->senderBuf,userHandle->recverBufSize, userHandle->recverNode,userHandle->recverBuf, userHandle->callbackFnPtr, userHandle->callbackData);
#endif
        DCMF_Result
        Res= DCMF_Get(&cmi_dcmf_direct_get_registration,
                      (DCMF_Request_t *) userHandle->DCMF_rq_tsend,
                      done_cb, DCMF_RELAXED_CONSISTENCY,
                      userHandle->recverNode,
                      userHandle->recverBufSize,
                      & userHandle->DCMF_recverMemregion,
                      & userHandle->DCMF_senderMemregion,
                      0, /* offsets are zero */
                      0
                     );
        CmiAssert(Res==DCMF_SUCCESS);


#if CMK_SMP
        DCMF_CriticalSection_exit (0);
#endif
    }
}

/**** up to the user to safely call this */
void CmiDirect_deassocLocalBuffer(struct infiDirectUserHandle *userHandle) {
    CmiAssert(userHandle->senderNode==_Cmi_mynode);
#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif

    DCMF_Memregion_destroy((DCMF_Memregion_t*) userHandle->DCMF_senderMemregion);
    CmiFree(userHandle->DCMF_notify_buf);
    CmiFree(userHandle->DCMF_rq_tsend);
#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif

}

/**** up to the user to safely call this */
void CmiDirect_destroyHandle(struct infiDirectUserHandle *userHandle) {
    CmiAssert(userHandle->recverNode==_Cmi_mynode);
#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif

    DCMF_Memregion_destroy((DCMF_Memregion_t*) userHandle->DCMF_recverMemregion);
    CmiFree(userHandle->DCMF_rq_trecv);

#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif
}



/**** Should not be called the first time *********/
void CmiDirect_ready(struct infiDirectUserHandle *userHandle) {
    /* no op on BGP */
}

/**** Should not be called the first time *********/
void CmiDirect_readyPollQ(struct infiDirectUserHandle *userHandle) {
    /* no op on BGP */
}

/**** Should not be called the first time *********/
void CmiDirect_readyMark(struct infiDirectUserHandle *userHandle) {
    /* no op on BGP */
}

#endif /* BGP_USE_RDMA_DIRECT*/

/*@}*/

