/*
 *created by Chao Mei
 *revised by Yanhua, Gengbin
 */

#if CMK_C_INLINE
#define INLINE_KEYWORD inline
#else
#define INLINE_KEYWORD
#endif

#if MACHINE_DEBUG_LOG
FILE *debugLog = NULL;
#endif

/******* broadcast related  */
#ifndef CMK_BROADCAST_SPANNING_TREE
#define CMK_BROADCAST_SPANNING_TREE    1
#endif

#ifndef CMK_BROADCAST_HYPERCUBE
#define CMK_BROADCAST_HYPERCUBE        0
#endif

#define BROADCAST_SPANNING_FACTOR      4
/* The number of children used when a msg is broadcast inside a node */
#define BROADCAST_SPANNING_INTRA_FACTOR  8

/* Root of broadcast:
 * non-bcast msg: root = 0;
 * proc-level bcast msg: root >=1; (CmiMyPe()+1)
 * node-level bcast msg: root <=-1; (-CmiMyNode()-1)
 */
#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);

/**
 * For some machine layers such as on Active Message framework,
 * the receiver callback is usally executed on an internal
 * thread (i.e. not the flow managed by ours). Therefore, for
 * forwarding broadcast messages, we could have a choice whether
 * to offload such function to the flow we manage such as the
 * communication thread. -
 */

#ifndef CMK_OFFLOAD_BCAST_PROCESS
#define CMK_OFFLOAD_BCAST_PROCESS 0
#endif

#if CMK_OFFLOAD_BCAST_PROCESS
CsvDeclare(PCQueue, procBcastQ);
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(PCQueue, nodeBcastQ);
#endif
#endif

#if CMK_WITH_STATS
static int  MSG_STATISTIC = 0;
int     msg_histogram[22];
static int _cmi_log2(int size)
{
    int ret = 1;
    size = size-1;
    while( (size=size>>1)>0) ret++;
    return ret;
}
#endif

#if CMK_BROADCAST_HYPERCUBE
/* ceil(log2(CmiNumNodes)) except when _Cmi_numnodes is 1, used for hypercube */
static int CmiNodesDim;
#endif
/* ###End of Broadcast related definitions ### */


static void handleOneBcastMsg(int size, char *msg);
static void processBcastQs();

/* Utility functions for forwarding broadcast messages,
 * should not be used in machine-specific implementations
 * except in some special occasions.
 */
static INLINE_KEYWORD void processProcBcastMsg(int size, char *msg);
static INLINE_KEYWORD void processNodeBcastMsg(int size, char *msg);
static void SendSpanningChildrenProc(int size, char *msg);
static void SendHyperCubeProc(int size, char *msg);
#if CMK_NODE_QUEUE_AVAILABLE
static void SendSpanningChildrenNode(int size, char *msg);
static void SendHyperCubeNode(int size, char *msg);
#endif

static void SendSpanningChildren(int size, char *msg, int rankToAssign, int startNode);
static void SendHyperCube(int size,  char *msg, int rankToAssign, int startNode);

#if USE_COMMON_SYNC_BCAST || USE_COMMON_ASYNC_BCAST
#if !CMK_BROADCAST_SPANNING_TREE && !CMK_BROADCAST_HYPERCUBE
#warning "Broadcast function is based on the plain P2P O(P)-message scheme!!!"
#endif
#endif


void CmiSyncBroadcastFn(int size, char *msg);
CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg);
void CmiFreeBroadcastFn(int size, char *msg);

void CmiSyncBroadcastAllFn(int size, char *msg);
CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg);
void CmiFreeBroadcastAllFn(int size, char *msg);

#if CMK_NODE_QUEUE_AVAILABLE
void CmiSyncNodeBroadcastFn(int size, char *msg);
CmiCommHandle CmiAsyncNodeeroadcastFn(int size, char *msg);
void CmiFreeNodeBroadcastFn(int size, char *msg);

void CmiSyncNodeBroadcastAllFn(int size, char *msg);
CmiCommHandle CmiAsyncNodeBroadcastAllFn(int size, char *msg);
void CmiFreeNodeBroadcastAllFn(int size, char *msg);
#endif

/************** Done with Broadcast related */

#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank

#ifndef CMK_HAS_SIZE_IN_MSGHDR
#define CMK_HAS_SIZE_IN_MSGHDR 1
#endif
#if CMK_HAS_SIZE_IN_MSGHDR
#define CMI_MSG_SIZE(msg)  ((CmiMsgHeaderBasic *)msg)->size
#else
#define CMI_MSG_SIZE(msg)  (CmiAbort("Has no msg size in header"))
#endif

#if CMK_NODE_QUEUE_AVAILABLE
/* This value should be larger than the number of cores used
 * per charm smp node. So it's currently set to such a large
 * value.
 */
#define DGRAM_NODEMESSAGE   (0x1FFB)
#endif

/* Node state structure */
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_numnodes;  /* Total number of address spaces */
int               _Cmi_numpes;    /* Total number of processors */

CpvDeclare(void*, CmiLocalQueue);

/* different modes for sending a message */
#define P2P_SYNC      0x1
#define P2P_ASYNC     0x2
#define BCAST_SYNC    0x4
#define BCAST_ASYNC   0x8
#define OUT_OF_BAND   0x10

enum MACHINE_SMP_MODE {
    INVALID_MODE,
#if CMK_BLUEGENEQ
    COMM_THREAD_SEND_RECV = 1,
#else 
    COMM_THREAD_SEND_RECV = 0,
#endif
    COMM_THREAD_ONLY_RECV, /* work threads will do the send */
    COMM_WORK_THREADS_SEND_RECV, /* work and comm threads do the both send/recv */
    COMM_THREAD_NOT_EXIST /* work threads will do both send and recv */
};
/* The default mode of smp charm runtime */
static enum MACHINE_SMP_MODE Cmi_smp_mode_setting = COMM_THREAD_SEND_RECV;


#if CMK_SMP
static volatile int commThdExit = 0;
static CmiNodeLock  commThdExitLock = 0;

/**
 *  The macro defines whether to have a comm thd to offload some
 *  work such as forwarding bcast messages etc. This macro
 *  should be defined before including "machine-smp.c". Note
 *  that whether a machine layer in SMP mode could run w/o comm
 *  thread depends on the support of the underlying
 *  communication library.
 *
 */
#ifndef CMK_SMP_NO_COMMTHD
#define CMK_SMP_NO_COMMTHD 0
#endif

#if CMK_SMP_NO_COMMTHD
int Cmi_commthread = 0;
#else
int Cmi_commthread = 1;
#endif

#endif

/*SHOULD BE MOVED TO MACHINE-SMP.C ??*/
static int Cmi_nodestart;

/*
 * Network progress utility variables. Period controls the rate at
 * which the network poll is called
 */
#ifndef NETWORK_PROGRESS_PERIOD_DEFAULT
#define NETWORK_PROGRESS_PERIOD_DEFAULT 1000
#endif

CpvDeclare(unsigned , networkProgressCount);
int networkProgressPeriod;


/* ===== Beginning of Common Function Declarations ===== */
void CmiAbort(const char *message);
static void PerrorExit(const char *msg);

/* This function handles the msg received as which queue to push into */
static void handleOneRecvedMsg(int size, char *msg);

/* Utility functions for forwarding broadcast messages,
 * should not be used in machine-specific implementations
 * except in some special occasions.
 */
static void SendToPeers(int size, char *msg);


void CmiPushPE(int rank, void *msg);

#if CMK_NODE_QUEUE_AVAILABLE
void CmiPushNode(void *msg);
#endif

/* Functions regarding send ops declared in converse.h */

/* In default, using the common codes for msg sending */
#ifndef USE_COMMON_SYNC_P2P
#define USE_COMMON_SYNC_P2P 1
#endif
#ifndef USE_COMMON_ASYNC_P2P
#define USE_COMMON_ASYNC_P2P 1
#endif
#ifndef USE_COMMON_SYNC_BCAST
#define USE_COMMON_SYNC_BCAST 1
#endif
#ifndef USE_COMMON_ASYNC_BCAST
#define USE_COMMON_ASYNC_BCAST 1
#endif

static void CmiSendSelf(char *msg);

void CmiSyncSendFn(int destPE, int size, char *msg);
CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg);
void CmiFreeSendFn(int destPE, int size, char *msg);

#if CMK_NODE_QUEUE_AVAILABLE
static void CmiSendNodeSelf(char *msg);

void CmiSyncNodeSendFn(int destNode, int size, char *msg);
CmiCommHandle CmiAsyncNodeSendFn(int destNode, int size, char *msg);
void CmiFreeNodeSendFn(int destNode, int size, char *msg);

#endif

/* Functions and variables regarding machine startup */
static char     **Cmi_argv;
static char     **Cmi_argvcopy;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */
void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret);
static void ConverseRunPE(int everReturn);

/* Functions regarding machine running on every proc */
static void AdvanceCommunication(int whenidle);
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);
void ConverseExit(void);

/* Functions providing incoming network messages */
void *CmiGetNonLocal(void);
#if CMK_NODE_QUEUE_AVAILABLE
void *CmiGetNonLocalNodeQ(void);
#endif
/* Utiltiy functions */
static char *CopyMsg(char *msg, int len);

/* ===== End of Common Function Declarations ===== */

#include "machine-smp.c"

/* ===== Beginning of Idle-state Related Declarations =====  */
typedef struct {
    int sleepMs; /*Milliseconds to sleep while idle*/
    int nIdles; /*Number of times we've been idle in a row*/
    CmiState cs; /*Machine state*/
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void);

/**
 *  Generally,
 *
 *  CmiNotifyIdle is used in non-SMP mode when the proc is idle.
 *  When the proc is idle, AdvanceCommunication needs to be
 *  called.
 *
 *  CmiNotifyStillIdle and CmiNotifyBeginIdle are used in SMP mode.
 *
 *  Different layers have choices of registering different callbacks for
 *  idle state.
 */
static void CmiNotifyBeginIdle(CmiIdleState *s);
static void CmiNotifyStillIdle(CmiIdleState *s);
void CmiNotifyIdle(void);
/* ===== End of Idle-state Related Declarations =====  */

/* ===== Beginning of Processor/Node State-related Stuff =====*/
#if !CMK_SMP
/************ non SMP **************/
static struct CmiStateStruct Cmi_state;
int _Cmi_mype;
int _Cmi_myrank;

void CmiMemLock() {}
void CmiMemUnlock() {}

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

void CmiYield(void) {
    sleep(0);
}

static void CmiStartThreads(char **argv) {
    CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
    _Cmi_mype = Cmi_nodestart;
    _Cmi_myrank = 0;
}
#else
/************** SMP *******************/
INLINE_KEYWORD int CmiMyPe() {
    return CmiGetState()->pe;
}
INLINE_KEYWORD int CmiMyRank() {
    return CmiGetState()->rank;
}
INLINE_KEYWORD int CmiNodeFirst(int node) {
    return node*_Cmi_mynodesize;
}
INLINE_KEYWORD int CmiNodeSize(int node) {
    return _Cmi_mynodesize;
}
INLINE_KEYWORD int CmiNodeOf(int pe) {
    return (pe/_Cmi_mynodesize);
}
INLINE_KEYWORD int CmiRankOf(int pe) {
    return pe%_Cmi_mynodesize;
}
#endif
CsvDeclare(CmiNodeState, NodeState);
/* ===== End of Processor/Node State-related Stuff =====*/

#include "machine-broadcast.c"
#include "immediate.c"
#include "machine-commthd-util.c"

/* ===== Beginning of Common Function Definitions ===== */
static void PerrorExit(const char *msg) {
    perror(msg);
    exit(1);
}

/* ##### Beginning of Functions Related with Message Sending OPs ##### */
/*Add a message to this processor's receive queue, pe is a rank */
void CmiPushPE(int rank,void *msg) {
    CmiState cs = CmiGetStateN(rank);
    MACHSTATE2(3,"Pushing message into rank %d's queue %p{",rank, cs->recv);
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        MACHSTATE1(3, "[%p] Push Immediate Message begin{",CmiGetState());
        CMI_DEST_RANK(msg) = rank;
        CmiPushImmediateMsg(msg);
        MACHSTATE1(3, "[%p] Push Immediate Message end}",CmiGetState());
        return;
    }
#endif

    PCQueuePush(cs->recv,(char*)msg);

#if CMK_SHARED_VARS_POSIX_THREADS_SMP
  if (_Cmi_noprocforcommthread)
#endif
    CmiIdleLock_addMessage(&cs->idle);
    MACHSTATE1(3,"} Pushing message into rank %d's queue done",rank);
}

#if CMK_NODE_QUEUE_AVAILABLE
/*Add a message to this processor's receive queue */
void CmiPushNode(void *msg) {
    MACHSTATE(3,"Pushing message into NodeRecv queue");
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        CMI_DEST_RANK(msg) = 0;
        CmiPushImmediateMsg(msg);
        return;
    }
#endif
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv,msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);

#if CMK_SHARED_VARS_POSIX_THREADS_SMP
    if (_Cmi_noprocforcommthread)
#endif
    {
        CmiState cs=CmiGetStateN(0);
        CmiIdleLock_addMessage(&cs->idle);
    }
}
#endif

/* This function handles the msg received as which queue to push into */
static INLINE_KEYWORD void handleOneRecvedMsg(int size, char *msg) {
    int isBcastMsg = 0;
#if CMK_BROADCAST_SPANNING_TREE || CMK_BROADCAST_HYPERCUBE
    isBcastMsg = (CMI_BROADCAST_ROOT(msg)!=0);
#endif

    if (isBcastMsg) {
        handleOneBcastMsg(size, msg);
        return;
    }

#if CMK_NODE_QUEUE_AVAILABLE
    if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE){
        CmiPushNode(msg);
        return;
    }
#endif
    CmiPushPE(CMI_DEST_RANK(msg), msg);

}


static void SendToPeers(int size, char *msg) {
    /* FIXME: now it's just a flat p2p send!! When node size is large,
    * it should also be sent in a tree
    */
    int exceptRank = CMI_DEST_RANK(msg);
    int i;
    for (i=0; i<exceptRank; i++) {
        CmiPushPE(i, CopyMsg(msg, size));
    }
    for (i=exceptRank+1; i<CmiMyNodeSize(); i++) {
        CmiPushPE(i, CopyMsg(msg, size));
    }
}


/* Functions regarding sending operations */
static void CmiSendSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        /* CmiBecomeNonImmediate(msg); */
        CmiPushImmediateMsg(msg);
        CmiHandleImmediate();
        return;
    }
#endif
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}

/* Functions regarding P2P send op */
#if USE_COMMON_SYNC_P2P
void CmiSyncSendFn(int destPE, int size, char *msg) {
    char *dupmsg = CopyMsg(msg, size);
    CmiFreeSendFn(destPE, size, dupmsg);
}

#if CMK_USE_PXSHM
#include "machine-pxshm.c"
#endif
#if CMK_USE_XPMEM
#include "machine-xpmem.c"
#endif

static int refcount = 0;

#if CMK_USE_OOB
CpvExtern(int, _urgentSend);
#endif

/* a wrapper of LrtsSendFunc */
#if CMK_C_INLINE
inline 
#endif
CmiCommHandle CmiSendNetworkFunc(int destPE, int size, char *msg, int mode)
{
        int rank;
        int destNode = CmiNodeOf(destPE); 
#if CMK_USE_PXSHM
        if (CmiValidPxshm(destNode, size)) {
          CmiSendMessagePxshm(msg, size, destNode, &refcount);
          //for (int i=0; i<refcount; i++) CmiReference(msg);
          return 0;
        }
#endif
#if CMK_USE_XPMEM
        if (CmiValidXpmem(destNode, size)) {
          CmiSendMessageXpmem(msg, size, destNode, &refcount);
          //for (int i=0; i<refcount; i++) CmiReference(msg);
          return 0;
        }
#endif
#if CMK_PERSISTENT_COMM
        if (CpvAccess(phs)) {
          if (size > PERSIST_MIN_SIZE) {
            CmiAssert(CpvAccess(curphs) < CpvAccess(phsSize));
            LrtsSendPersistentMsg(CpvAccess(phs)[CpvAccess(curphs)], destNode, size, msg);
            return 0;
          }
        }
#endif

#if CMK_WITH_STATS
if (MSG_STATISTIC)
{
    int ret_log = _cmi_log2(size);
    if(ret_log >21) ret_log = 21;
    msg_histogram[ret_log]++;
}
#endif
#if CMK_USE_OOB
    if (CpvAccess(_urgentSend)) mode |= OUT_OF_BAND;
#endif
    return LrtsSendFunc(CmiNodeOf(destPE), destPE, size, msg, mode);
}

void CmiFreeSendFn(int destPE, int size, char *msg) {
    CMI_SET_BROADCAST_ROOT(msg, 0);
    CQdCreate(CpvAccess(cQdState), 1);
    if (CmiMyPe()==destPE) {
        CmiSendSelf(msg);
#if CMK_PERSISTENT_COMM
        if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
    } 
    else {
        int destNode = CmiNodeOf(destPE);
        int destRank = CmiRankOf(destPE);
#if CMK_SMP
        if (CmiMyNode()==destNode) {
            CmiPushPE(destRank, msg);
#if CMK_PERSISTENT_COMM
            if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
            return;
        }
#endif
        CMI_DEST_RANK(msg) = destRank;
        CmiSendNetworkFunc(destPE, size, msg, P2P_SYNC);
#if CMK_PERSISTENT_COMM
        if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
    }
}
#endif

#if USE_COMMON_ASYNC_P2P
CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg) {
    int destNode = CmiNodeOf(destPE);
    if (destNode == CmiMyNode()) {
        CmiSyncSendFn(destPE,size,msg);
        return 0;
    } else {
#if CMK_WITH_STATS
if (  MSG_STATISTIC)
{
    int ret_log = _cmi_log2(size);
        if(ret_log >21) ret_log = 21;
        msg_histogram[ret_log]++;
}
#endif
        return CmiSendNetworkFunc(destPE, size, msg, P2P_ASYNC);
    }
}
#endif

#if CMK_NODE_QUEUE_AVAILABLE
static void CmiSendNodeSelf(char *msg) {
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        CmiPushImmediateMsg(msg);
        if (!_immRunning) CmiHandleImmediate();
        return;
    }
#endif
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv, msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
}

#if USE_COMMON_ASYNC_P2P
void CmiSyncNodeSendFn(int destNode, int size, char *msg) {
    char *dupmsg = CopyMsg(msg, size);
    CmiFreeNodeSendFn(destNode, size, dupmsg);
}

void CmiFreeNodeSendFn(int destNode, int size, char *msg) {
    CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
    CQdCreate(CpvAccess(cQdState), 1);
    CMI_SET_BROADCAST_ROOT(msg, 0);
    if (destNode == CmiMyNode()) {
        CmiSendNodeSelf(msg);
    } else {
#if CMK_WITH_STATS
if (  MSG_STATISTIC)
{
    int ret_log = _cmi_log2(size);
    if(ret_log >21) ret_log = 21;
    msg_histogram[ret_log]++;
}
#endif
        CmiSendNetworkFunc(CmiNodeFirst(destNode), size, msg, P2P_SYNC);
    }
#if CMK_PERSISTENT_COMM
    if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
}
#endif

#if USE_COMMON_ASYNC_P2P
CmiCommHandle CmiAsyncNodeSendFn(int destNode, int size, char *msg) {
    if (destNode == CmiMyNode()) {
        CmiSyncNodeSendFn(destNode, size, msg);
        return 0;
    } else {
#if CMK_WITH_STATS
if (  MSG_STATISTIC)
{
        int ret_log = _cmi_log2(size);
        if(ret_log >21) ret_log = 21;
        msg_histogram[ret_log]++;
}
#endif
        return CmiSendNetworkFunc(CmiNodeFirst(destNode), size, msg, P2P_ASYNC);
    }
}
#endif
#endif

/* ##### Beginning of Functions Related with Machine Startup ##### */
void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret) {
    int _ii;
    int tmp;
#if CMK_WITH_STATS
    MSG_STATISTIC = CmiGetArgFlag(argv, "+msgstatistic");
#endif
    /* processor per node */
    _Cmi_mynodesize = 1;
    if (!CmiGetArgInt(argv,"+ppn", &_Cmi_mynodesize))
        CmiGetArgInt(argv,"++ppn", &_Cmi_mynodesize);
#if ! CMK_SMP
    if (_Cmi_mynodesize > 1 && _Cmi_mynode == 0)
        CmiAbort("+ppn cannot be used in non SMP version!\n");
#endif

    /* Network progress function is used to poll the network when for
    messages. This flushes receive buffers on some  implementations*/
    networkProgressPeriod = NETWORK_PROGRESS_PERIOD_DEFAULT;
    CmiGetArgInt(argv, "+networkProgressPeriod", &networkProgressPeriod);

    /* _Cmi_mynodesize has to be obtained before LrtsInit
     * because it may be used inside LrtsInit
     */
    /* argv could be changed inside LrtsInit */
    /* Inside this function, the number of nodes and my node id are obtained */
#if CMK_WITH_STATS
if (  MSG_STATISTIC)
{
    for(_ii=0; _ii<22; _ii++)
        msg_histogram[_ii] = 0;
}
#endif

    LrtsInit(&argc, &argv, &_Cmi_numnodes, &_Cmi_mynode);
#if MACHINE_DEBUG_LOG
    char ln[200];
    sprintf(ln,"debugLog.%d", _Cmi_mynode);
    debugLog=fopen(ln,"w");
    if (debugLog == NULL)
    {
        CmiAbort("debug file not open\n");
    }
#endif


	if (_Cmi_mynode==0) {
#if !CMK_SMP 
		printf("Charm++> Running on non-SMP mode\n");
#else
		printf("Charm++> Running on SMP mode, %d worker threads per process\n", _Cmi_mynodesize);
		if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
			printf("Charm++> The comm. thread both sends and receives messages\n");
		} else if (Cmi_smp_mode_setting == COMM_THREAD_ONLY_RECV) {
			printf("Charm++> The comm. thread only receives messages, while work threads send messages\n");
		} else if (Cmi_smp_mode_setting == COMM_WORK_THREADS_SEND_RECV) {
			printf("Charm++> Both  comm. thread and worker thread send and messages\n");
		} else if (Cmi_smp_mode_setting == COMM_THREAD_NOT_EXIST) {
			printf("Charm++> There's no comm. thread. Work threads both send and receive messages\n");
		} else {
			CmiAbort("Charm++> Invalid SMP mode setting\n");
		}
#endif
	}

    _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
    Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
    Cmi_argvcopy = CmiCopyArgs(argv);
    Cmi_argv = argv;
    Cmi_startfn = fn;
    Cmi_usrsched = usched;

#if CMK_USE_PXSHM
    CmiInitPxshm(argv);
#endif
#if CMK_USE_XPMEM
    CmiInitXpmem(argv);
#endif

    /* CmiTimerInit(); */
#if CMK_BROADCAST_HYPERCUBE
    /* CmiNodesDim = ceil(log2(CmiNumNodes)) except when #nodes is 1*/
    tmp = CmiNumNodes()-1;
    CmiNodesDim = 0;
    while (tmp>0) {
        CmiNodesDim++;
        tmp = tmp >> 1;
    }
    if (CmiNumNodes()==1) CmiNodesDim=1;
#endif

    CsvInitialize(CmiNodeState, NodeState);
    CmiNodeStateInit(&CsvAccess(NodeState));
#if CMK_SMP
    commThdExitLock = CmiCreateLock();
#endif

#if CMK_OFFLOAD_BCAST_PROCESS
    /* the actual queues should be created on comm thread considering NUMA in SMP */
    CsvInitialize(PCQueue, procBcastQ);
#if CMK_NODE_QUEUE_AVAILABLE
    CsvInitialize(PCQueue, nodeBcastQ);
#endif
#endif

#if CMK_SMP && CMK_LEVERAGE_COMMTHREAD
    CsvInitialize(PCQueue, notifyCommThdMsgBuffer);
#endif

    CmiStartThreads(argv);

    ConverseRunPE(initret);
}

extern void ConverseCommonInit(char **argv);
extern void CthInit(char **argv);

static void ConverseRunPE(int everReturn) {
    CmiState cs;
    char** CmiMyArgv;

    LrtsPreCommonInit(everReturn);

#if CMK_OFFLOAD_BCAST_PROCESS
    int createQueue = 1;
#if CMK_SMP
#if CMK_SMP_NO_COMMTHD
    /* If there's no comm thread, then the queue is created on rank 0 */
    if (CmiMyRank()) createQueue = 0;
#else
    if (CmiMyRank()<CmiMyNodeSize()) createQueue = 0;
#endif
#endif

    if (createQueue) {
        CsvAccess(procBcastQ) = PCQueueCreate();
#if CMK_NODE_QUEUE_AVAILABLE
        CsvAccess(nodeBcastQ) = PCQueueCreate();
#endif
    }
#endif

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
    CpvInitialize(unsigned , networkProgressCount);
    CpvAccess(networkProgressCount) = 0;

    ConverseCommonInit(CmiMyArgv);

    LrtsPostCommonInit(everReturn);

#if CMK_SMP && CMK_LEVERAGE_COMMTHREAD
    CmiInitNotifyCommThdScheme();
#endif
    /* Converse initialization finishes, immediate messages can be processed.
       node barrier previously should take care of the node synchronization */
    _immediateReady = 1;

    if(CharmLibInterOperate) {
	/* !!! Not considering SMP mode now */
	/* TODO: make interoperability working in SMP!!! */
	Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
	CsdScheduler(-1);
    } else {
      /* communication thread */
      if (CmiMyRank() == CmiMyNodeSize()) {
        Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
        while (1) CommunicationServerThread(5);
      } else { /* worker thread */
        if (!everReturn) {
          Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
          if (Cmi_usrsched==0) CsdScheduler(-1);
          ConverseExit();
        }
      }
    }
}
/* ##### End of Functions Related with Machine Startup ##### */

/* ##### Beginning of Functions Related with Machine Running ##### */
static INLINE_KEYWORD void AdvanceCommunication(int whenidle) {
    int doProcessBcast = 1;

#if CMK_USE_PXSHM
    CommunicationServerPxshm();
#endif
#if CMK_USE_XPMEM
    CommunicationServerXpmem();
#endif

    LrtsAdvanceCommunication(whenidle);

#if CMK_OFFLOAD_BCAST_PROCESS
#if CMK_SMP_NO_COMMTHD
    /*FIXME: only asks rank 0 to process bcast msgs, so perf may suffer*/
    if (CmiMyRank()) doProcessBcast = 0;
#endif
    if (doProcessBcast) processBcastQs();
#endif

#if CMK_IMMEDIATE_MSG
#if !CMK_SMP
    CmiHandleImmediate();
#endif
#if CMK_SMP && CMK_SMP_NO_COMMTHD
    if (CmiMyRank()==0) CmiHandleImmediate();
#endif
#endif
}

extern void ConverseCommonExit();

static void CommunicationServer(int sleepTime) {
#if CMK_SMP 
    AdvanceCommunication(1);

    if (commThdExit == CmiMyNodeSize()) {
        MACHSTATE(2, "CommunicationServer exiting {");
        LrtsDrainResources();
        MACHSTATE(2, "} CommunicationServer EXIT");

        ConverseCommonExit();

#if CMK_USE_PXSHM
        CmiExitPxshm();
#endif
#if CMK_USE_XPMEM
        CmiExitXpmem();
#endif
        LrtsExit();
    }
#endif
}

static void CommunicationServerThread(int sleepTime) {
    CommunicationServer(sleepTime);
#if CMK_IMMEDIATE_MSG
    CmiHandleImmediate();
#endif
}

void ConverseExit(void) {
    int i;
#if !CMK_SMP
    LrtsDrainResources();
#else
	if(Cmi_smp_mode_setting == COMM_THREAD_ONLY_RECV
	   || Cmi_smp_mode_setting == COMM_THREAD_NOT_EXIST)
		LrtsDrainResources();
#endif

    ConverseCommonExit();

#if CMK_WITH_STATS
if (MSG_STATISTIC)
{
    for(i=0; i<22; i++)
    {
        CmiPrintf("[MSG PE:%d]", CmiMyPe());
        if(msg_histogram[i] >0)
            CmiPrintf("(%d:%d) ", 1<<i, msg_histogram[i]);
    }
    CmiPrintf("\n");
}
#endif

#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyPe() == 0) CmiPrintf("End of program\n");
#endif

#if !CMK_SMP  || CMK_SMP_NO_COMMTHD
#if CMK_USE_PXSHM
    CmiExitPxshm();
#endif
#if CMK_USE_XPMEM
    CmiExitXpmem();
#endif
    LrtsExit();
#else
    /* In SMP, the communication thread will exit */
    /* atomic increment */
    CmiLock(commThdExitLock);
    commThdExit++;
    CmiUnlock(commThdExitLock);
    while (1) CmiYield();
#endif
}
/* ##### End of Functions Related with Machine Running ##### */

void CmiAbort(const char *message) {
#if CMK_USE_PXSHM
    CmiExitPxshm();
#endif
#if CMK_USE_XPMEM
    CmiExitXpmem();
#endif
    LrtsAbort(message);
}

/* ##### Beginning of Functions Providing Incoming Network Messages ##### */
void *CmiGetNonLocal(void) {
    CmiState cs = CmiGetState();
    void *msg = NULL;

#if !CMK_SMP || CMK_SMP_NO_COMMTHD
    /**
      * In SMP mode with comm thread, it's possible a normal
      * msg is sent from an immediate msg which is executed
      * on comm thread. In this case, the msg is sent to
      * the network queue of the work thread. Therefore,
      * even there's only one worker thread, the polling of
      * network queue is still required.
      */
    if (CmiNumPes() == 1) return NULL;
#endif

    MACHSTATE2(3, "[%p] CmiGetNonLocal begin %d{", cs, CmiMyPe());
    CmiIdleLock_checkMessage(&cs->idle);
    /* ?????although it seems that lock is not needed, I found it crashes very often
       on mpi-smp without lock */
    msg = PCQueuePop(cs->recv);
#if !CMK_SMP  || CMK_SMP_NO_COMMTHD
    if (!msg) {
       AdvanceCommunication(0);
       msg = PCQueuePop(cs->recv);
    }
#else
//    LrtsPostNonLocal();
#endif

    MACHSTATE3(3,"[%p] CmiGetNonLocal from queue %p with msg %p end }",CmiGetState(),(cs->recv), msg);

    return msg;
}

#if CMK_NODE_QUEUE_AVAILABLE
void *CmiGetNonLocalNodeQ(void) {
    CmiState cs = CmiGetState();
    char *result = 0;
    CmiIdleLock_checkMessage(&cs->idle);
    if (!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
        MACHSTATE1(3,"CmiGetNonLocalNodeQ begin %d {", CmiMyPe());
        CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
        result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
        CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
        MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
    }

    return result;
}
#endif
/* ##### End of Functions Providing Incoming Network Messages ##### */

static CmiIdleState *CmiNotifyGetState(void) {
    CmiIdleState *s=(CmiIdleState *)malloc(sizeof(CmiIdleState));
    s->sleepMs=0;
    s->nIdles=0;
    s->cs=CmiGetState();
    return s;
}

static void CmiNotifyBeginIdle(CmiIdleState *s) {
    s->sleepMs=0;
    s->nIdles=0;
}

/*Number of times to spin before sleeping*/
#define SPINS_BEFORE_SLEEP 20
static void CmiNotifyStillIdle(CmiIdleState *s) {
    MACHSTATE1(2,"still idle (%d) begin {",CmiMyPe())
#if !CMK_SMP
    AdvanceCommunication(1);
#else
    LrtsPostNonLocal();

    if (_Cmi_noprocforcommthread) {
    s->nIdles++;
    if (s->nIdles>SPINS_BEFORE_SLEEP) { /*Start giving some time back to the OS*/
        s->sleepMs+=2;
        if (s->sleepMs>10) s->sleepMs=10;
    }

    if (s->sleepMs>0) {
        MACHSTATE1(2,"idle lock(%d) {",CmiMyPe())
        CmiIdleLock_sleep(&s->cs->idle,s->sleepMs);
        MACHSTATE1(2,"} idle lock(%d)",CmiMyPe())
    }
    }
#endif

    MACHSTATE1(2,"still idle (%d) end {",CmiMyPe())
}

/* usually called in non-smp mode */
void CmiNotifyIdle(void) {
    AdvanceCommunication(1);
    CmiYield();
}

/* Utiltiy functions */
static char *CopyMsg(char *msg, int len) {
    char *copy = (char *)CmiAlloc(len);
#if CMK_ERROR_CHECKING
    if (!copy) {
        CmiAbort("Error: out of memory in machine layer\n");
    }
#endif
    memcpy(copy, msg, len);
    return copy;
}



