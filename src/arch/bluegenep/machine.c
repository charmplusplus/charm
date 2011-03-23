
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "machine.h"
#include "converse.h"
#include "pcqueue.h"
#include "assert.h"
#include "malloc.h"

#include <bpcore/ppc450_inlines.h>

#include "dcmf.h"
#include "dcmf_multisend.h"

char *ALIGN_16(char *p) {
    return((char *)((((unsigned long)p)+0xf)&0xfffffff0));
}

#define PROGRESS_PERIOD 1024

//There are two roles of comm thread:
// 1. polling other cores' bcast msg queue
// 2. bcast msg is handled only by comm thd
#define BCASTMSG_ONLY_TO_COMMTHD 0

CpvDeclare(PCQueue, broadcast_q);                 //queue to send broadcast messages
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(PCQueue, node_bcastq);
CsvDeclare(CmiNodeLock, node_bcastLock);
#endif

/*
    To reduce the buffer used in broadcast and distribute the load from
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of
  spanning tree broadcast algorithm.
    This will use the fourth short in message as an indicator of spanning tree
  root.
*/
#if CMK_SMP
#define CMK_BROADCAST_SPANNING_TREE   1
#else
#define CMK_BROADCAST_SPANNING_TREE    1
#define CMK_BROADCAST_HYPERCUBE        0
#endif /* CMK_SMP */

#define BROADCAST_SPANNING_FACTOR     2

//The root of the message infers the type of the message
// 1. root is 0, then it is a normal point-to-point message
// 2. root is larger than 0 (>=1), then it is a broadcast message across all processors (cores)
// 3. root is less than 0 (<=-1), then it is a broadcast message across all nodes
#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_IS_BCAST_ON_CORES(msg) (CMI_BROADCAST_ROOT(msg) > 0)
#define CMI_IS_BCAST_ON_NODES(msg) (CMI_BROADCAST_ROOT(msg) < 0)
#define CMI_GET_CYCLE(msg)               ((CmiMsgHeaderBasic *)msg)->root

#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank
#define CMI_MAGIC(msg)                   ((CmiMsgHeaderBasic *)msg)->magic

/* FIXME: need a random number that everyone agrees ! */
#define CHARM_MAGIC_NUMBER               126

#if !CMK_OPTIMIZE
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
            for(count = 0; count < len; count++) { \
                printf("%2x", msg[count]);                 \
            }                                             \
            printf("------------------------------\n\n"); \
            CmiAbort("Fatal error: checksum doesn't agree!\n"); \
          }
#else
#define CMI_SET_CHECKSUM(msg, len)
#define CMI_CHECK_CHECKSUM(msg, len)
#endif

#define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);

#if CMK_BROADCAST_HYPERCUBE
#  define CMI_SET_CYCLE(msg, cycle)  CMI_GET_CYCLE(msg) = (cycle);
#else
#  define CMI_SET_CYCLE(msg, cycle)
#endif

int               _Cmi_numpes;
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_numnodes;  /* Total number of address spaces */
int                Cmi_nodestart; /* First processor in this address space */
CpvDeclare(void*, CmiLocalQueue);


#if CMK_NODE_QUEUE_AVAILABLE
#define SMP_NODEMESSAGE   (0xFB) // rank of the node message when node queue
// is available
#define NODE_BROADCAST_OTHERS (-1)
#define NODE_BROADCAST_ALL    (-2)
#endif


typedef struct ProcState {
    /* PCQueue      sendMsgBuf; */      /* per processor message sending queue */
    CmiNodeLock  recvLock;              /* for cs->recv */
    CmiNodeLock bcastLock;
} ProcState;

static ProcState  *procState;

#if CMK_SMP && !CMK_MULTICORE
static volatile int commThdExit = 0;
static CmiNodeLock commThdExitLock = 0;
#endif

void ConverseRunPE(int everReturn);
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);

//So far we dont define any comm threads
int Cmi_commthread = 0;

#include "machine-smp.c"
CsvDeclare(CmiNodeState, NodeState);
#include "immediate.c"

void AdvanceCommunications();


#if !CMK_SMP
/************ non SMP **************/
static struct CmiStateStruct Cmi_state;
int _Cmi_mype;
int _Cmi_myrank;

void CmiMemLock(void) {}
void CmiMemUnlock(void) {}

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

//void CmiYield(void) { sleep(0); }

static void CmiStartThreads(char **argv) {
    CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
    _Cmi_mype = Cmi_nodestart;
    _Cmi_myrank = 0;
}
#endif  /* !CMK_SMP */

//int received_immediate;
//int received_broadcast;

/*Add a message to this processor's receive queue, pe is a rank */
void CmiPushPE(int pe,void *msg) {
    CmiState cs = CmiGetStateN(pe);
    MACHSTATE2(3,"Pushing message into rank %d's queue %p{",pe, cs->recv);
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        /**(CmiUInt2 *)msg = pe;*/
        //received_immediate = 1;
        //printf("PushPE: N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        //CMI_DEST_RANK(msg) = pe;
        CmiPushImmediateMsg(msg);
        return;
    }
#endif
#if CMK_SMP
    CmiLock(procState[pe].recvLock);
#endif

    PCQueuePush(cs->recv,(char *)msg);
    //printf("%d: PCQueue length = %d, msg = %x\n", CmiMyPe(), PCQueueLength(cs->recv), msg);

#if CMK_SMP
    CmiUnlock(procState[pe].recvLock);
#endif
    CmiIdleLock_addMessage(&cs->idle);
    MACHSTATE1(3,"} Pushing message into rank %d's queue done",pe);
}

#if CMK_NODE_QUEUE_AVAILABLE
/*Add a message to this processor's receive queue */
static void CmiPushNode(void *msg) {
    MACHSTATE(3,"Pushing message into NodeRecv queue");
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        //printf("PushNode: N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        //CMI_DEST_RANK(msg) = 0;
        CmiPushImmediateMsg(msg);
        return;
    }
#endif
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv,msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    {
        CmiState cs=CmiGetStateN(0);
        CmiIdleLock_addMessage(&cs->idle);
    }
}
#endif /* CMK_NODE_QUEUE_AVAILABLE */

volatile int msgQueueLen;
volatile int outstanding_recvs;

static int Cmi_dim;     /* hypercube dim of network */

static char     **Cmi_argv;
static char     **Cmi_argvcopy;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */

extern void ConverseCommonInit(char **argv);
extern void ConverseCommonExit(void);
extern void CthInit(char **argv);

static void SendMsgsUntil(int);


void SendSpanningChildren(int size, char *msg);
#if CMK_NODE_QUEUE_AVAILABLE
void SendSpanningChildrenNode(int size, char *msg);
#endif
void SendHypercube(int size, char *msg);

DCMF_Protocol_t  cmi_dcmf_short_registration __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_eager_registration __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_rzv_registration   __attribute__((__aligned__(16)));
DCMF_Protocol_t  cmi_dcmf_multicast_registration   __attribute__((__aligned__(16)));


#define BGP_USE_AM_DIRECT 1
//#define BGP_USE_RDMA_DIRECT 1
//#define CMI_DIRECT_DEBUG 1
#ifdef BGP_USE_AM_DIRECT


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


#endif

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


#endif


typedef struct msg_list {
    char              * msg;
    int                 size;
    int                 destpe;
    int               * pelist;
    DCMF_Callback_t     cb;
    DCQuad              info __attribute__((__aligned__(16)));
    DCMF_Request_t      send __attribute__((__aligned__(16)));
} SMSG_LIST __attribute__((__aligned__(16)));

#define MAX_NUM_SMSGS   64
CpvDeclare(PCQueue, smsg_list_q);

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

typedef struct {
    int sleepMs; /*Milliseconds to sleep while idle*/
    int nIdles; /*Number of times we've been idle in a row*/
    CmiState cs; /*Machine state*/
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void) {
    CmiIdleState *s=(CmiIdleState *)CmiAlloc(sizeof(CmiIdleState));
    s->sleepMs=0;
    s->nIdles=0;
    s->cs=CmiGetState();
    return s;
}


#if (DCMF_VERSION_MAJOR >= 2)
static void send_done(void *data, DCMF_Error_t *err) 
#else 
static void send_done(void *data) 
#endif
/* send done callback: sets the smsg entry to done */
{
    SMSG_LIST *msg_tmp = (SMSG_LIST *)(data);
    CmiFree(msg_tmp->msg);
    //free(data);
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


#if (DCMF_VERSION_MAJOR >= 2)
static void recv_done(void *clientdata, DCMF_Error_t * err) 
#else 
static void recv_done(void *clientdata) 
#endif
/* recv done callback: push the recved msg to recv queue */
{

    char *msg = (char *) clientdata;
    int sndlen = ((CmiMsgHeaderBasic *) msg)->size;

    //fprintf (stderr, "%d Recv message done \n", CmiMyPe());

    /* then we do what PumpMsgs used to do:
     * push msg to recv queue */
    int count=0;
    CMI_CHECK_CHECKSUM(msg, sndlen);
    if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
        CmiAbort("Charm++ Warning: Non Charm++ Message Received. \n");
        return;
    }

#if CMK_BROADCAST_SPANNING_TREE | CMK_BROADCAST_HYPERCUBE
    if (CMI_IS_BCAST_ON_CORES(msg) ) {
        int pe = CMI_DEST_RANK(msg);

        //printf ("%d: Receiving bcast message from %d with %d bytes for %d\n", CmiMyPe(), CMI_BROADCAST_ROOT(msg), sndlen, pe);

        char *copymsg;
        copymsg = (char *)CmiAlloc(sndlen);
        CmiMemcpy(copymsg,msg,sndlen);

        //received_broadcast = 1;
#if CMK_SMP
        CmiLock(procState[pe].bcastLock);
        PCQueuePush(CpvAccessOther(broadcast_q, pe), copymsg);
        CmiUnlock(procState[pe].bcastLock);
#else
        PCQueuePush(CpvAccess(broadcast_q), copymsg);
#endif
    }
#endif

#if CMK_NODE_QUEUE_AVAILABLE
#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_IS_BCAST_ON_NODES(msg)) {
        //printf ("%d: Receiving node bcast message from %d with %d bytes for %d\n", CmiMyPe(), CMI_BROADCAST_ROOT(msg), sndlen, CMI_DEST_RANK(msg));
        char *copymsg = (char *)CmiAlloc(sndlen);
        CmiMemcpy(copymsg,msg,sndlen);
        //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
        CmiLock(CsvAccess(node_bcastLock));
        PCQueuePush(CsvAccess(node_bcastq), copymsg);
        CmiUnlock(CsvAccess(node_bcastLock));
        //CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    }
#endif
    if (CMI_DEST_RANK(msg) == SMP_NODEMESSAGE)
        CmiPushNode(msg);
    else
#endif

#if CMK_SMP && !CMK_MULTICORE && BCASTMSG_ONLY_TO_COMMTHD
        if (CMI_DEST_RANK(msg)<_Cmi_mynodesize) { //not the comm thd
            CmiPushPE(CMI_DEST_RANK(msg), (void *)msg);
        }
#else
        CmiPushPE(CMI_DEST_RANK(msg), (void *)msg);
#endif

    outstanding_recvs --;
}


void     short_pkt_recv (void             * clientdata,
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

    //printf ("%d: Receiving message %d bytes from %d\n", CmiMyPe(), sndlen, senderrank);

    /* printf ("Receiving %d bytes\n", sndlen); */
    *rcvlen = sndlen;  /* to avoid malloc(0) which might
                                   return NULL */

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

    //printf ("%d: Receiving message %d bytes from %d\n", CmiMyPe(), sndlen, senderrank);

    /* printf ("Receiving %d bytes\n", sndlen); */
    *rcvlen = sndlen;  /* to avoid malloc(0) which might
                                   return NULL */

    *buffer = (char *)CmiAlloc(alloc_size);
    cb->function = recv_done;
    cb->clientdata = *buffer;

    return (DCMF_Request_t *) ALIGN_16(*buffer + sndlen);
}


#if CMK_NODE_QUEUE_AVAILABLE
void sendBroadcastMessagesNode() {
    if (PCQueueLength(CsvAccess(node_bcastq))==0) return;
    //node broadcast message could be always handled by any cores (including
    //comm thd) on this node
    //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    CmiLock(CsvAccess(node_bcastLock));
    char *msg = PCQueuePop(CsvAccess(node_bcastq));
    CmiUnlock(CsvAccess(node_bcastLock));
    //CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    while (msg) {
#if CMK_BROADCAST_SPANNING_TREE
        //printf("sendBroadcastMessagesNode: node %d rank %d with msg root %d\n", CmiMyNode(), CmiMyRank(), CMI_BROADCAST_ROOT(msg));
        SendSpanningChildrenNode(((CmiMsgHeaderBasic *) msg)->size, msg);
#endif
        CmiFree(msg);
        //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
        CmiLock(CsvAccess(node_bcastLock));
        msg = PCQueuePop(CsvAccess(node_bcastq));
        CmiUnlock(CsvAccess(node_bcastLock));
        //CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    }
}
#endif

void sendBroadcastMessages() {
#if !CMK_MULTICORE && !BCASTMSG_ONLY_TO_COMMTHD
//in the presence of comm thd, and it is not responsible for broadcasting msg,
//the comm thd will help to pull the msg from rank 0 which is predefined as the
//core on a smp node to receive bcast msg.
    int toPullRank = CmiMyRank();
    if (CmiMyRank()==_Cmi_mynodesize) toPullRank = 0; //comm thd only pulls msg from rank 0
#else
    int toPullRank = CmiMyRank();
#endif
    PCQueue toPullQ;

    /*
    if(CmiMyRank()==_Cmi_mynodesize)
      printf("Comm thd on node [%d] is pulling bcast msg\n", CmiMyNode());
    else
      printf("Work thd [%d] on node [%d] is pulling bcast msg\n", CmiMyRank(), CmiMyNode());
    */
#if !CMK_MULTICORE && !BCASTMSG_ONLY_TO_COMMTHD
    toPullQ = CpvAccessOther(broadcast_q, toPullRank);
#else
    toPullQ = CpvAccess(broadcast_q);
#endif

    if (PCQueueLength(toPullQ)==0) return;
#if CMK_SMP
    CmiLock(procState[toPullRank].bcastLock);
#endif

    char *msg = (char *) PCQueuePop(toPullQ);

#if CMK_SMP
    CmiUnlock(procState[toPullRank].bcastLock);
#endif

    while (msg) {

#if CMK_BROADCAST_SPANNING_TREE
        SendSpanningChildren(((CmiMsgHeaderBasic *) msg)->size, msg);
#elif CMK_BROADCAST_HYPERCUBE
        SendHypercube(((CmiMsgHeaderBasic *) msg)->size, msg);
#endif

        CmiFree (msg);

#if CMK_SMP
        CmiLock(procState[toPullRank].bcastLock);
#endif

#if !CMK_MULTICORE && !BCASTMSG_ONLY_TO_COMMTHD
        toPullQ = CpvAccessOther(broadcast_q, toPullRank);
#else
        toPullQ = CpvAccess(broadcast_q);
#endif
        msg = (char *) PCQueuePop(toPullQ);

#if CMK_SMP
        CmiUnlock(procState[toPullRank].bcastLock);
#endif
    }
}

CpvDeclare(unsigned, networkProgressCount);
int  networkProgressPeriod;

#if 0
unsigned int *ranklist;

BGTsC_t        barrier;

// -----------------------------------------
// Rectangular broadcast implementation
// -----------------------------------------

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




//--------------------------------------------------------------
//----- End Rectangular Broadcast Implementation ---------------
//--------------------------------------------------------------
#endif

//approx sleep command
void mysleep (int cycles) {
    unsigned long long start = DCMF_Timebase();
    unsigned long long end = start + cycles;

    while (start < end)
        start = DCMF_Timebase();

    return;
}

static void * test_buf;

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret) {
    int n, i, count;

    //fprintf(stderr, "Initializing Converse Blue Gene/P machine Layer\n");

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

    _Cmi_numnodes = DCMF_Messager_size();
    _Cmi_mynode = DCMF_Messager_rank();

    unsigned rank = DCMF_Messager_rank();
    unsigned size = DCMF_Messager_size();

    CmiBarrier();
    CmiBarrier();
    CmiBarrier();

    /* processor per node */
    _Cmi_mynodesize = 1;
    CmiGetArgInt(argv,"+ppn", &_Cmi_mynodesize);
#if ! CMK_SMP
    if (_Cmi_mynodesize > 1 && _Cmi_mynode == 0)
        CmiAbort("+ppn cannot be used in non SMP version!\n");
#endif

    _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
    Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
    Cmi_argvcopy = CmiCopyArgs(argv);
    Cmi_argv = argv;
    Cmi_startfn = fn;
    Cmi_usrsched = usched;

    //printf ("Starting Charm with %d nodes and %d processors\n", CmiNumNodes(), CmiNumPes());

    DCMF_Multicast_Configuration_t mconfig;
    mconfig.protocol = DCMF_MEMFIFO_DMA_MSEND_PROTOCOL;
    mconfig.cb_recv  = first_multi_pkt_recv_done;
    mconfig.clientdata = NULL;
    mconfig.connectionlist = (void **) malloc (CmiNumPes() * sizeof(unsigned long));
    mconfig.nconnections = CmiNumPes();  
    DCMF_Multicast_register(&cmi_dcmf_multicast_registration, &mconfig);


    /* find dim = log2(numpes), to pretend we are a hypercube */
    for ( Cmi_dim=0,n=_Cmi_numpes; n>1; n/=2 )
        Cmi_dim++ ;


    /* checksum flag */
    if (CmiGetArgFlag(argv,"+checksum")) {
#if !CMK_OPTIMIZE
        checksum_flag = 1;
        if (_Cmi_mynode == 0) CmiPrintf("Charm++: CheckSum checking enabled! \n");
#else
        if (_Cmi_mynode == 0) CmiPrintf("Charm++: +checksum ignored in optimized version! \n");
#endif
    }

    CsvInitialize(CmiNodeState, NodeState);
    CmiNodeStateInit(&CsvAccess(NodeState));

#if CMK_NODE_QUEUE_AVAILABLE
    CsvInitialize(PCQueue, node_bcastq);
    CsvAccess(node_bcastq) = PCQueueCreate();
    CsvInitialize(CmiNodeLock, node_bcastLock);
    CsvAccess(node_bcastLock) = CmiCreateLock();
#endif

    int actualNodeSize = _Cmi_mynodesize;
#if !CMK_MULTICORE
    actualNodeSize++; //considering the extra comm thread
#endif

    procState = (ProcState *)CmiAlloc((actualNodeSize) * sizeof(ProcState));
    for (i=0; i<actualNodeSize; i++) {
        /*    procState[i].sendMsgBuf = PCQueueCreate();   */
        procState[i].recvLock = CmiCreateLock();
        procState[i].bcastLock = CmiCreateLock();
    }

#if CMK_SMP && !CMK_MULTICORE
    commThdExitLock = CmiCreateLock();
#endif

    /* Network progress function is used to poll the network when for
       messages. This flushes receive buffers on some  implementations*/
    networkProgressPeriod = PROGRESS_PERIOD;
    CmiGetArgInt(argv, "+networkProgressPeriod", &networkProgressPeriod);

    //printf ("Starting Threads\n");

    CmiStartThreads(argv);

    ConverseRunPE(initret);
}


int PerrorExit (char *err) {
    fprintf (stderr, "err\n\n");
    exit (-1);
    return -1;
}


void ConverseRunPE(int everReturn) {
    //printf ("ConverseRunPE on rank %d\n", CmiMyPe());

    CmiIdleState *s=CmiNotifyGetState();
    CmiState cs;
    char** CmiMyArgv;
    CmiNodeAllBarrier();

    cs = CmiGetState();
    CpvInitialize(void *,CmiLocalQueue);
    CpvAccess(CmiLocalQueue) = cs->localqueue;

    if (CmiMyRank())
        CmiMyArgv=CmiCopyArgs(Cmi_argvcopy);
    else
        CmiMyArgv=Cmi_argv;

    CthInit(CmiMyArgv);

    /* initialize the network progress counter*/
    /* Network progress function is used to poll the network when for
       messages. This flushes receive buffers on some  implementations*/
    CpvInitialize(int , networkProgressCount);
    CpvAccess(networkProgressCount) = 0;

    CpvInitialize(PCQueue, broadcast_q);
    CpvAccess(broadcast_q) = PCQueueCreate();

    CpvInitialize(PCQueue, smsg_list_q);
    CpvAccess(smsg_list_q) = PCQueueCreate();

    //printf ("Before Converse Common Init\n");
    ConverseCommonInit(CmiMyArgv);

    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);

    CmiBarrier();

    /* Converse initialization finishes, immediate messages can be processed.
       node barrier previously should take care of the node synchronization */
    _immediateReady = 1;

    /* communication thread */
    if (CmiMyRank() == CmiMyNodeSize()) {
        Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
        while (1) CommunicationServer(5);
    } else {
        //printf ("Calling Start Fn and the scheduler \n");

        if (!everReturn) {
            Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
            if (Cmi_usrsched==0) CsdScheduler(-1);
            ConverseExit();
        }
    }
}

#if CMK_SMP
static int inexit = 0;

/* test if all processors recv queues are empty */
static int RecvQueueEmpty() {
    int i;
    for (i=0; i<_Cmi_mynodesize; i++) {
        CmiState cs=CmiGetStateN(i);
        if (!PCQueueEmpty(cs->recv)) return 0;
    }
    return 1;
}

#endif


//extern void DCMF_Messager_dumpTimers();

void ConverseExit(void) {

    while (msgQueueLen > 0 || outstanding_recvs > 0) {
        AdvanceCommunications();
    }

    CmiNodeBarrier();
    ConverseCommonExit();

    //  if(CmiMyPe()%101 == 0)
    //DCMF_Messager_dumpTimers();

    if (CmiMyPe() == 0) {
        printf("End of program\n");
    }

    CmiNodeBarrier();
//  CmiNodeAllBarrier ();

#if CMK_SMP && !CMK_MULTICORE
    //CmiLock(commThdExitLock);
    commThdExit = 1;
    //CmiUnlock(commThdExitLock);

    _bgp_msync();
#endif

    int rank0 = 0;

    if (CmiMyRank() == 0) {
        rank0 = 1;
        //CmiFree(procState);
        DCMF_Messager_finalize();
    }

    CmiNodeBarrier();
//  CmiNodeAllBarrier ();

    if (rank0)
        exit(0);
    else
        pthread_exit(NULL);
}

/* exit() called on any node would abort the whole program */
void CmiAbort(const char * message) {
    CmiError("------------- Processor %d Exiting: Called CmiAbort ------------\n"
             "{snd:%d,rcv:%d} Reason: %s\n",CmiMyPe(),
             msgQueueLen, outstanding_recvs, message);
    //CmiPrintStackTrace(0);

    while (msgQueueLen > 0 || outstanding_recvs > 0) {
        AdvanceCommunications();
    }

    CmiBarrier();
    assert (0);
}

static void CommunicationServer(int sleepTime) {
#if CMK_SMP && !CMK_MULTICORE
    //CmiLock(commThdExitLock);
    if (commThdExit) {
        while (msgQueueLen > 0 || outstanding_recvs > 0) {
            AdvanceCommunications();
        }
        CmiUnlock(commThdExitLock);
        pthread_exit(NULL);
        return;
    }
    //CmiUnlock(commThdExitLock);
#endif
    AdvanceCommunications();

#if CMK_IMMEDIATE_MSG && CMK_SMP
    CmiHandleImmediate();
#endif

    //mysleep(sleepTime);
}

static void CommunicationServerThread(int sleepTime) {
#if CMK_SMP
    CommunicationServer(sleepTime);
#endif
//immediate msgs are handled in AdvanceCommunications
}

#if CMK_NODE_QUEUE_AVAILABLE
char *CmiGetNonLocalNodeQ(void) {
    CmiState cs = CmiGetState();
    char *result = 0;
    CmiIdleLock_checkMessage(&cs->idle);
    if (!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
        MACHSTATE1(3,"CmiGetNonLocalNodeQ begin %d {", CmiMyPe());

        if (CmiTryLock(CsvAccess(NodeState).CmiNodeRecvLock) == 0) {
            //CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
            result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
            CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
        }

        MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
    }
    return result;
}
#endif


void *CmiGetNonLocal() {

    CmiState cs = CmiGetState();

    void *msg = NULL;
    CmiIdleLock_checkMessage(&cs->idle);
    /* although it seems that lock is not needed, I found it crashes very often
       on mpi-smp without lock */

#if !CMK_SMP || CMK_MULTICORE  /*|| !BCASTMSG_ONLY_TO_COMMTHD*/
//ChaoMei changes
    AdvanceCommunications();
#endif

    /*if(CmiMyRank()==0) printf("Got stuck here on proc[%d] node[%d]\n", CmiMyPe(), CmiMyNode());*/

    if (PCQueueLength(cs->recv)==0) return NULL;

#if CMK_SMP
    CmiLock(procState[cs->rank].recvLock);
#endif

    msg =  PCQueuePop(cs->recv);

#if CMK_SMP
    CmiUnlock(procState[cs->rank].recvLock);
#endif

    return msg;
}

static void CmiSendSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        /* CmiBecomeNonImmediate(msg); */
        //printf("In SendSelf, N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        CmiPushImmediateMsg(msg);
        CmiHandleImmediate();
        return;
    }
#endif
    
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}

#if CMK_SMP
static void CmiSendPeer (int rank, int size, char *msg) {
#if CMK_BROADCAST_SPANNING_TREE | CMK_BROADCAST_HYPERCUBE
    if (CMI_BROADCAST_ROOT(msg) != 0) {
        char *copymsg;
        copymsg = (char *)CmiAlloc(size);
        CmiMemcpy(copymsg,msg,size);

        CmiLock(procState[rank].bcastLock);
        PCQueuePush(CpvAccessOther(broadcast_q, rank), copymsg);
        CmiUnlock(procState[rank].bcastLock);
    }
#endif
    
    CmiPushPE (rank, msg);
}
#endif


void machineSend(SMSG_LIST *msg_tmp) {    

    if (msg_tmp->destpe == CmiMyNode())
        CmiAbort("Sending to self\n");

    CmiAssert(msg_tmp->destpe >= 0 && msg_tmp->destpe < CmiNumNodes());
    msg_tmp->cb.function     =   send_done;
    msg_tmp->cb.clientdata   =   msg_tmp;

    DCMF_Protocol_t *protocol = NULL;

    if (msg_tmp->size < 224)
        protocol = &cmi_dcmf_short_registration;
    else if (msg_tmp->size < 2048)
        protocol = &cmi_dcmf_eager_registration;
    else
        protocol = &cmi_dcmf_rzv_registration;


#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif

    msgQueueLen ++;
    DCMF_Send (protocol, &msg_tmp->send, msg_tmp->cb,
               DCMF_MATCH_CONSISTENCY, msg_tmp->destpe,
               msg_tmp->size, msg_tmp->msg, &msg_tmp->info, 1);

/*    
    #if CMK_SMP && !CMK_MULTICORE
    //Adding this advance call here improves the SMP performance
    //a little bit although it is possible that some more bugs are
    //introduced
    DCMF_Messager_advance();
    #endif
*/

#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif
}

#define MAX_MULTICAST 128
DCMF_Opcode_t  CmiOpcodeList [MAX_MULTICAST];

void  machineMulticast(int npes, int *pelist, int size, char* msg){  
  CQdCreate(CpvAccess(cQdState), npes);
  
  CmiAssert (npes < MAX_MULTICAST);

  CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
  ((CmiMsgHeaderBasic *)msg)->size = size;  
  CMI_SET_BROADCAST_ROOT(msg,0);
  CMI_SET_CHECKSUM(msg, size);
  
  SMSG_LIST *msg_tmp = smsg_allocate(); //(SMSG_LIST *) malloc(sizeof(SMSG_LIST));
  
  msg_tmp->destpe    = -1;      //multicast operation
  msg_tmp->size      = size * npes; //keep track of #bytes outstanding
  msg_tmp->msg       = msg;
  msg_tmp->pelist    = pelist;
  
  msgQueueLen ++;
  
  DCMF_Multicast_t  mcast_info __attribute__((__aligned__(16)));
  
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
  mcast_info.msginfo        =   &msg_tmp->info;
  mcast_info.count          =   1;

  DCMF_Multicast (&mcast_info);
}



void CmiGeneralFreeSendN (int node, int rank, int size, char * msg);

/* The general free send function
 * Send is synchronous, and free msg after posted
 */
void  CmiGeneralFreeSend(int destPE, int size, char* msg) {

  if (destPE < 0 || destPE > CmiNumPes ())
    printf ("Sending to %d\n", destPE);

  CmiAssert (destPE >= 0 && destPE < CmiNumPes());

    CmiState cs = CmiGetState();

    if (destPE==cs->pe) {
        CmiSendSelf(msg);
        return;
    }

#if CMK_SMP && !CMK_MULTICORE && BCASTMSG_ONLY_TO_COMMTHD
    //In the presence of comm thd which is responsible for sending bcast msgs, then
    //CmiNodeOf and CmiRankOf may return incorrect information if the destPE is considered
    //as a comm thd.
    if (destPE >= _Cmi_numpes) { //destination is a comm thd
        int nid = destPE - _Cmi_numpes;
        int rid = _Cmi_mynodesize;
        CmiGeneralFreeSendN (nid, rid, size, msg);
    } else {
        CmiGeneralFreeSendN (CmiNodeOf (destPE), CmiRankOf (destPE), size, msg);
    }
#else
    //printf ("%d: Sending Message to %d \n", CmiMyPe(), destPE);
    CmiGeneralFreeSendN (CmiNodeOf (destPE), CmiRankOf (destPE), size, msg);
#endif
}

void CmiGeneralFreeSendN (int node, int rank, int size, char * msg) {

    //printf ("%d, %d: Sending Message to node %d rank %d \n", CmiMyPe(),
    //  CmiMyNode(), node, rank);

#if CMK_SMP
    CMI_DEST_RANK(msg) = rank;
    //CMI_SET_CHECKSUM(msg, size);

    if (node == CmiMyNode()) {
        CmiSendPeer (rank, size, msg);
        return;
    }
#endif

    SMSG_LIST *msg_tmp = smsg_allocate(); //(SMSG_LIST *) malloc(sizeof(SMSG_LIST));
    msg_tmp->destpe = node; //destPE;
    msg_tmp->size = size;
    msg_tmp->msg = msg;

    machineSend(msg_tmp);
}

void CmiSyncSendFn(int destPE, int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg,msg,size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncSendFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeSendFn(destPE,size,copymsg);
}

void CmiFreeSendFn(int destPE, int size, char *msg) {    
    CQdCreate(CpvAccess(cQdState), 1);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeSendFn on comm thd on node %d\n", CmiMyNode());

    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)msg)->size = size;
    CMI_SET_CHECKSUM(msg, size);

    CmiGeneralFreeSend(destPE,size,msg);
}

/* same as CmiSyncSendFn, but don't set broadcast root in msg header */
void CmiSyncSendFn1(int destPE, int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg, msg, size);

    //  asm volatile("sync" ::: "memory");

    CMI_MAGIC(copymsg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)copymsg)->size = size;
    CMI_SET_CHECKSUM(copymsg, size);

    CmiGeneralFreeSend(destPE,size,copymsg);
}

#define NODE_LEVEL_ST_IMPLEMENTATION 1
#if NODE_LEVEL_ST_IMPLEMENTATION
//send msgs to other ranks except the rank specified in the argument
static void CmiSendChildrenPeers(int rank, int size, char *msg) {
    //printf ("%d [%d]: Send children peers except rank %d\n",  CmiMyPe(), CmiMyNode(), CmiMyRank());
    int r=0;

//With comm thd, broadcast msg will be pulled from bcast queue from the comm thd
//And the msg would be finally pushed into other cores on the same node by the
//comm thd. But it's possible a msg has already been pushed into rank 0 in
//recv_done func call. So there's no need to push the bcast msg into the recv
//queue again in the case BCASTMSG_ONLY_TO_COMMTHD is not set

#if !CMK_MULTICORE && !BCASTMSG_ONLY_TO_COMMTHD
    //indicate this is called from comm thread.
    if (rank == _Cmi_mynodesize) r = 1;
#endif

    for (;r<rank; r++) {
        char *copymsg;
        copymsg = (char *)CmiAlloc(size);
        CmiMemcpy(copymsg,msg,size);        
        CmiPushPE (r, copymsg);
    }

    for (r=rank+1; r<_Cmi_mynodesize; r++) {
        char *copymsg;
        copymsg = (char *)CmiAlloc(size);
        CmiMemcpy(copymsg,msg,size);        
        CmiPushPE (r, copymsg);
    }
}

//In this implementation, msgs are first sent out to the comm thd or rank 0 of other nodes
//then send msgs to the other cores on the same node
void SendSpanningChildren(int size, char *msg) {
    CmiState cs = CmiGetState();
    int startpe = CMI_BROADCAST_ROOT(msg)-1;
    int i;

    //printf ("%d [%d]: In Send Spanning Tree with startpe %d\n",  CmiMyPe(), CmiMyNode(), startpe);

    CmiAssert(startpe>=0 && startpe<_Cmi_numpes);

    int startNid = CmiNodeOf(startpe);
    int thisNid, thisRid;
    thisNid = CmiMyNode();
    thisRid = CmiMyRank();

    //printf ("%d [%d/%d]: In Send Spanning Tree with startpe %d\n",  CmiMyPe(), CmiMyNode(), thisRid, startpe);

    //Step1: send to cores that has comm thd on other nodes
    int dist = thisNid - startNid;
    if (dist<0) dist += _Cmi_numnodes;
    for (i=1; i <= BROADCAST_SPANNING_FACTOR; i++) {
        int nid = BROADCAST_SPANNING_FACTOR*dist + i;
        if (nid > _Cmi_numnodes - 1) break;
        nid += startNid;
        nid = nid%_Cmi_numnodes;
        CmiAssert(nid>=0 && nid<_Cmi_numnodes && nid!=thisNid);
#if CMK_SMP && !CMK_MULTICORE && BCASTMSG_ONLY_TO_COMMTHD
        int p = nid + _Cmi_numpes;
#else
        int p = CmiNodeFirst(nid);
#endif
        //printf ("%d [%d]: Sending Spanning Tree Msg to %d\n",  CmiMyPe(), CmiMyNode(), p);
        CmiSyncSendFn1(p, size, msg);
    }

    //Step2: send to other cores (i.e. excluding myself cs->pe) on the same nodes (just a flat send)
    CmiSendChildrenPeers(thisRid, size, msg);

#if !CMK_SMP
#if ENABLE_BROADCAST_THROTTLE
    SendMsgsUntil (0);
#endif
#endif
}
#else
/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg) {
    CmiState cs = CmiGetState();
    int startpe = CMI_BROADCAST_ROOT(msg)-1;
    int i;

    //printf ("%d [%d]: In Send Spanning Tree\n",  CmiMyPe(), CmiMyNode());

    CmiAssert(startpe>=0 && startpe<_Cmi_numpes);
    int dist = cs->pe-startpe;
    if (dist<0) dist+=_Cmi_numpes;
    for (i=1; i <= BROADCAST_SPANNING_FACTOR; i++) {
        int p = BROADCAST_SPANNING_FACTOR*dist + i;
        if (p > _Cmi_numpes - 1) break;
        p += startpe;
        p = p%_Cmi_numpes;
        CmiAssert(p>=0 && p<_Cmi_numpes && p!=cs->pe);

        //printf ("%d [%d]: Sending Spanning Tree Msg to %d\n",  CmiMyPe(), CmiMyNode(), p);
        CmiSyncSendFn1(p, size, msg);
    }    
}
#endif

/* send msg along the hypercube in broadcast. (Sameer) */
void SendHypercube(int size, char *msg) {
    CmiState cs = CmiGetState();
    int curcycle = CMI_GET_CYCLE(msg);
    int i;

    double logp = CmiNumPes();
    logp = log(logp)/log(2.0);
    logp = ceil(logp);

    /*  CmiPrintf("In hypercube\n"); */
    /* assert(startpe>=0 && startpe<_Cmi_numpes); */

    for (i = curcycle; i < logp; i++) {
        int p = cs->pe ^ (1 << i);
        /*   CmiPrintf("p = %d, logp = %5.1f\n", p, logp);*/
        if (p < CmiNumPes()) {
            CMI_SET_CYCLE(msg, i + 1);

            CmiAssert(p>=0 && p<_Cmi_numpes && p!=cs->pe);
            CmiSyncSendFn1(p, size, msg);
        }
    }
}

void CmiSyncBroadcastFn(int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg,msg,size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncBroadcastFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeBroadcastFn(size,copymsg);
}

void CmiFreeBroadcastFn(int size, char *msg) {

    //printf("%d: Calling Broadcast %d\n", CmiMyPe(), size);

    CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE    
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeBroadcastFn on comm thd on node %d\n", CmiMyNode());

    //printf ("%d: Starting Spanning Tree Broadcast of size %d bytes\n", CmiMyPe(), size);

    CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
    SendSpanningChildren(size, msg);
    CmiFree(msg);
#elif CMK_BROADCAST_HYPERCUBE    
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);

    CMI_SET_CYCLE(msg, 0);
    SendHypercube(size, msg);
    CmiFree(msg);
#else
    int i;

    for ( i=cs->pe+1; i<_Cmi_numpes; i++ )
        CmiSyncSendFn(i,size,msg);

    for ( i=0; i<cs->pe; i++ )
        CmiSyncSendFn(i,size,msg);

    CmiFree(msg);
#endif
}

void CmiSyncBroadcastAllFn(int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg,msg,size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncBroadcastAllFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeBroadcastAllFn(size,copymsg);
}

void CmiFreeBroadcastAllFn(int size, char *msg) {

    //printf("%d: Calling All Broadcast %d\n", CmiMyPe(), size);

    CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE

    //printf ("%d: Starting Spanning Tree Broadcast of size %d bytes\n", CmiMyPe(), size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeBroadcastAllFn on comm thd on node %d\n", CmiMyNode());

    CmiSyncSendFn(cs->pe,size,msg);
    
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);

    CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
    SendSpanningChildren(size, msg);
    CmiFree(msg);

#elif CMK_BROADCAST_HYPERCUBE
    CmiSyncSendFn(cs->pe,size,msg);
    
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);

    CMI_SET_CYCLE(msg, 0);
    SendHypercube(size, msg);
    CmiFree(msg);
#else
    int i ;

    DCMF_CriticalSection_enter (0);

    for ( i=0; i<_Cmi_numpes; i++ ) {
        CmiSyncSendFn(i,size,msg);

        if ( (i % 32) == 0 )
            SendMsgsUntil (0);
    }

    DCMF_CriticalSection_exit (0);

    CmiFree(msg);
#endif
}

void AdvanceCommunications() {

#if CMK_SMP
    DCMF_CriticalSection_enter (0);
#endif

    while(DCMF_Messager_advance()>0);
    //DCMF_Messager_advance();

#if CMK_SMP
    DCMF_CriticalSection_exit (0);
#endif

    sendBroadcastMessages();
#if CMK_NODE_QUEUE_AVAILABLE
    sendBroadcastMessagesNode();
#endif

#if CMK_IMMEDIATE_MSG && !CMK_SMP
    CmiHandleImmediate();
#endif
}


static void SendMsgsUntil(int targetm) {

    while (msgQueueLen>targetm) {
      //AdvanceCommunications ();
#if CMK_SMP
      DCMF_CriticalSection_enter (0);
#endif
      
      while(DCMF_Messager_advance()>0);
      //DCMF_Messager_advance();
      
#if CMK_SMP
      DCMF_CriticalSection_exit (0);
#endif
    }
}

void CmiNotifyIdle() {
#if !CMK_SMP || CMK_MULTICORE
    AdvanceCommunications();
#endif
}


/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/************ Recommended routines ***********************/
/************ You dont have to implement these but they are supported
 in the converse syntax and some rare programs may crash. But most
 programs dont need them. *************/

CmiCommHandle CmiAsyncSendFn(int dest, int size, char *msg) {
    CmiAbort("CmiAsyncSendFn not implemented.");
    return (CmiCommHandle) 0;
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg) {
    CmiAbort("CmiAsyncBroadcastFn not implemented.");
    return (CmiCommHandle) 0;
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg) {
    CmiAbort("CmiAsyncBroadcastAllFn not implemented.");
    return (CmiCommHandle) 0;
}

int           CmiAsyncMsgSent(CmiCommHandle handle) {
    CmiAbort("CmiAsyncMsgSent not implemented.");
    return 0;
}
void          CmiReleaseCommHandle(CmiCommHandle handle) {
    CmiAbort("CmiReleaseCommHandle not implemented.");
}


/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/* Optional routines which could use common code which is shared with
   other machine layer implementations. */

/* MULTICAST/VECTOR SENDING FUNCTIONS

 * In relations to some flags, some other delivery functions may be needed.
 */

#if ! CMK_MULTICAST_LIST_USE_COMMON_CODE

void CmiSyncListSendFn(int npes, int *pes, int size, char *msg) {
    char *copymsg;
    copymsg = (char *)CmiAlloc(size);
    CmiMemcpy(copymsg,msg,size);
    CmiFreeListSendFn(npes, pes, size, msg);
}

//#define OPTIMIZED_MULTICAST  0

void CmiFreeListSendFn(int npes, int *pes, int size, char *msg) {
#if CMK_SMP && !CMK_MULTICORE
    //DCMF_CriticalSection_enter (0);
#endif
    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)msg)->size = size;
    CMI_SET_CHECKSUM(msg, size);

    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeListSendFn on comm thd on node %d\n", CmiMyNode());

    //printf("%d: In Free List Send Fn\n", CmiMyPe());
    int new_npes = 0;

    int i, count = 0, my_loc = -1;
    for (i=0; i<npes; i++) {
        //if (pes[i] == CmiMyPe() || CmiNodeOf(pes[i]) == CmiMyNode()) {
        if (CmiNodeOf(pes[i]) == CmiMyNode()) {
            CmiSyncSend(pes[i], size, msg);
            //my_loc = i;
        }
    }

#if OPTIMIZED_MULTICAST
#warning "Using Optimized Multicast"
    if (npes > 1) {    
      int *newpelist = (int *) malloc (sizeof(int) * npes);
      int new_npes = 0;
    
      for(i=0; i<npes; i++) {
	if(CmiNodeOf(pes[i]) == CmiMyNode()) 
	  continue;
	else
	  newpelist[new_npes++] = pes[i];
      }

      if (new_npes >= 1)
	machineMulticast (new_npes, newpelist, size, msg);
      else
	CmiFree (msg);
      return;
    }
#endif

    for (i=0;i<npes;i++) {
        //if (pes[i] == CmiMyPe() || CmiNodeOf(pes[i]) == CmiMyNode());
        if (CmiNodeOf(pes[i]) == CmiMyNode());
        else if (i < npes - 1) {
#if !CMK_SMP /*|| (CMK_SMP && !CMK_MULTICORE)*/
            CmiReference(msg);
            CmiGeneralFreeSend(pes[i], size, msg);
#else
            CmiSyncSend(pes[i], size, msg);
#endif
        }
    }

    //if (npes  && (pes[npes-1] != CmiMyPe() && CmiNodeOf(pes[i]) != CmiMyNode()))
    if (npes  && CmiNodeOf(pes[npes-1]) != CmiMyNode())
        CmiSyncSendAndFree(pes[npes-1], size, msg); //Sameto CmiFreeSendFn
    else
        CmiFree(msg);

    //AdvanceCommunications();
#if CMK_SMP && !CMK_MULTICORE
    //DCMF_CriticalSection_exit (0);
#endif
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int size, char *msg) {
    CmiAbort("CmiAsyncListSendFn not implemented.");
    return (CmiCommHandle) 0;
}
#endif

/** NODE SENDING FUNCTIONS

 * If there is a node queue, and we consider also nodes as entity (tipically in
 * SMP versions), these functions are needed.
 */

#if CMK_NODE_QUEUE_AVAILABLE

void          CmiSyncNodeSendFn(int, int, char *);
CmiCommHandle CmiAsyncNodeSendFn(int, int, char *);
void          CmiFreeNodeSendFn(int, int, char *);

void          CmiSyncNodeBroadcastFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastFn(int, char *);
void          CmiFreeNodeBroadcastFn(int, char *);

void          CmiSyncNodeBroadcastAllFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastAllFn(int, char *);
void          CmiFreeNodeBroadcastAllFn(int, char *);

#endif


#if CMK_SHARED_VARS_POSIX_THREADS_SMP

int CmiMyPe();
int CmiMyRank();
int CmiNodeFirst(int node);
int CmiNodeSize(int node);
int CmiNodeOf(int pe);
int CmiRankOf(int pe);

int CmiMyPe(void) {
    return CmiGetState()->pe;
}

int CmiMyRank(void) {
    return CmiGetState()->rank;
}

int CmiNodeFirst(int node) {
    return node*_Cmi_mynodesize;
}
int CmiNodeSize(int node)  {
    return _Cmi_mynodesize;
}

int CmiNodeOf(int pe)      {
    return (pe/_Cmi_mynodesize);
}
int CmiRankOf(int pe)      {
    return pe%_Cmi_mynodesize;
}


/* optional, these functions are implemented in "machine-smp.c", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier(void);
void CmiNodeAllBarrier(void);
CmiNodeLock CmiCreateLock();
void CmiDestroyLock(CmiNodeLock lock);

#endif

/** IMMEDIATE MESSAGES

 * If immediate messages are supported, the following function is needed. There
 * is an exeption if the machine progress is also defined (see later for this).

 * Moreover, the file "immediate.c" should be included, otherwise all its
 * functions and variables have to be redefined.
*/

#if CMK_CCS_AVAILABLE

#include "immediate.c"

#if ! CMK_MACHINE_PROGRESS_DEFINED /* Hack for some machines */
void CmiProbeImmediateMsg();
#endif

#endif


/** MACHINE PROGRESS DEFINED

 * Some machines (like BlueGene/L) do not have coprocessors, and messages need
 * to be pulled out of the network manually. For this reason the following
 * functions are needed. Notice that the function "CmiProbeImmediateMsg" must
 * not be defined anymore.
 */

#if CMK_MACHINE_PROGRESS_DEFINED



void CmiMachineProgressImpl() {

#if !CMK_SMP
    AdvanceCommunications();
#else
    /*Not implemented yet. Communication server does not seem to be
      thread safe */
#endif
}

#endif

/* Dummy implementation */
extern int CmiBarrier() {
    //Use DCMF barrier later
}

#if CMK_NODE_QUEUE_AVAILABLE
static void CmiSendNodeSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        //printf("SendNodeSelf: N[%d]P[%d]R[%d] received an imm msg with hdl: %p\n", CmiMyNode(), CmiMyPe(), CmiMyRank(), CmiGetHandler(msg));
        CmiPushImmediateMsg(msg);
#if CMK_MULTICORE
        CmiHandleImmediate();
#endif
        return;
    }
#endif    
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv, msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
}

CmiCommHandle CmiAsyncNodeSendFn(int dstNode, int size, char *msg) {
    CmiAbort ("Async Node Send not supported\n");
}

void CmiFreeNodeSendFn(int node, int size, char *msg) {

    CMI_SET_BROADCAST_ROOT(msg,0);
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)msg)->size = size;
    CMI_SET_CHECKSUM(msg, size);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeNodeSendFn on comm thd on node %d\n", CmiMyNode());
    
    CQdCreate(CpvAccess(cQdState), 1);

    if (node == _Cmi_mynode) {
        CmiSendNodeSelf(msg);
    } else {
        CmiGeneralFreeSendN(node, SMP_NODEMESSAGE, size, msg);
    }
}

void CmiSyncNodeSendFn(int p, int s, char *m) {
    char *dupmsg;
    dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncNodeSendFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeNodeSendFn(p, s, dupmsg);
}

CmiCommHandle CmiAsyncNodeBroadcastFn(int s, char *m) {
    return NULL;
}

void SendSpanningChildrenNode(int size, char *msg) {
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;
    //printf("on node %d rank %d, send node spanning children with root %d\n", CmiMyNode(), CmiMyRank(), startnode);
    assert(startnode>=0 && startnode<CmiNumNodes());

    int dist = CmiMyNode()-startnode;
    if (dist<0) dist += CmiNumNodes();
    int i;
    for (i=1; i <= BROADCAST_SPANNING_FACTOR; i++) {
        int nid = BROADCAST_SPANNING_FACTOR*dist + i;
        if (nid > CmiNumNodes() - 1) break;
        nid += startnode;
        nid = nid%CmiNumNodes();
        assert(nid>=0 && nid<CmiNumNodes() && nid!=CmiMyNode());
        char *dupmsg = (char *)CmiAlloc(size);
        CmiMemcpy(dupmsg,msg,size);
        //printf("In SendSpanningChildrenNode, sending bcast msg (root %d) from node %d to node %d\n", startnode, CmiMyNode(), nid);
        CmiGeneralFreeSendN(nid, SMP_NODEMESSAGE, size, dupmsg);
    }
}

/* need */
void CmiFreeNodeBroadcastFn(int s, char *m) {
#if CMK_BROADCAST_SPANNING_TREE
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeNodeBcastFn on comm thd on node %d\n", CmiMyNode());
    
    CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);

    int mynode = CmiMyNode();
    CMI_SET_BROADCAST_ROOT(m, -mynode-1);
    CMI_MAGIC(m) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)m)->size = s;
    CMI_SET_CHECKSUM(m, s);
    //printf("In CmiFreeNodeBroadcastFn, sending bcast msg from root node %d\n", CMI_BROADCAST_ROOT(m));

    SendSpanningChildrenNode(s, m);
#else
    int i;
    for (i=0; i<CmiNumNodes(); i++) {
        if (i==CmiMyNode()) continue;
        char *dupmsg = (char *)CmiAlloc(s);
        CmiMemcpy(dupmsg,m,s);
        CmiFreeNodeSendFn(i, s, dupmsg);
    }
#endif
    CmiFree(m);    
}

void CmiSyncNodeBroadcastFn(int s, char *m) {
    char *dupmsg;
    dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncNodeBcastFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeNodeBroadcastFn(s, dupmsg);
}

/* need */
void CmiFreeNodeBroadcastAllFn(int s, char *m) {
    char *dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    CMI_MAGIC(dupmsg) = CHARM_MAGIC_NUMBER;
    ((CmiMsgHeaderBasic *)dupmsg)->size = s;
    CMI_SET_CHECKSUM(dupmsg, s);

    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiFreeNodeBcastAllFn on comm thd on node %d\n", CmiMyNode());
    
    CQdCreate(CpvAccess(cQdState), 1);
    CmiSendNodeSelf(dupmsg);

    CmiFreeNodeBroadcastFn(s, m);
}

void CmiSyncNodeBroadcastAllFn(int s, char *m) {
    char *dupmsg;
    dupmsg = (char *)CmiAlloc(s);
    CmiMemcpy(dupmsg,m,s);
    //if(CmiMyRank()==CmiMyNodeSize()) printf("CmiSyncNodeBcastAllFn on comm thd on node %d\n", CmiMyNode());
    CmiFreeNodeBroadcastAllFn(s, dupmsg);
}


CmiCommHandle CmiAsyncNodeBroadcastAllFn(int s, char *m) {
    return NULL;
}
#endif //end of CMK_NODE_QUEUE_AVAILABLE

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
void CmiDirect_deassocLocalBuffer(struct infiDirectUserHandle *userHandle)
{
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
void CmiDirect_destroyHandle(struct infiDirectUserHandle *userHandle){
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
static void CmiNotifyRemoteRDMA(void *handle, struct DCMF_Error_t *error)
{
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
void CmiDirect_deassocLocalBuffer(struct infiDirectUserHandle *userHandle)
{
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
void CmiDirect_destroyHandle(struct infiDirectUserHandle *userHandle){
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

