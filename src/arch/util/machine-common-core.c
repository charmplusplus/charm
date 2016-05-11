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
CsvDeclare(CMIQueue, procBcastQ);
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(CMIQueue, nodeBcastQ);
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

#if CMK_SMP_TRACE_COMMTHREAD
extern double TraceTimerCommon();
#endif

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

#include <assert.h>

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

// global state, equals local if running one partition
PartitionInfo _partitionInfo;
int _Cmi_mype_global;
int _Cmi_numpes_global;
int _Cmi_mynode_global;
int _Cmi_numnodes_global;
static int _writeToStdout = 1;

// Node state structure, local information for the partition
int               _Cmi_myphysnode_numprocesses;  /* Total number of processes within this node */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_numnodes;  /* Total number of address spaces */
int               _Cmi_numpes;    /* Total number of processors */
int               userDrivenMode; /* Set by CharmInit for interop in user driven mode */
extern int CharmLibInterOperate;

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
volatile int commThdExit = 0;
CmiNodeLock  commThdExitLock = 0;
extern CmiNodeLock  interopCommThdExitLock;

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
#define CMK_SMP_NO_COMMTHD CMK_MULTICORE
#endif

#if CMK_SMP_NO_COMMTHD
int Cmi_commthread = 0;
#else
int Cmi_commthread = 1;
#endif

#endif

/*SHOULD BE MOVED TO MACHINE-SMP.C ??*/
static int Cmi_nodestart = -1;
static int Cmi_nodestartGlobal = -1;

/*
 * Network progress utility variables. Period controls the rate at
 * which the network poll is called
 */
#ifndef NETWORK_PROGRESS_PERIOD_DEFAULT
#define NETWORK_PROGRESS_PERIOD_DEFAULT 1000
#endif

CpvDeclare(unsigned , networkProgressCount);
int networkProgressPeriod;

#if CMK_CCS_AVAILABLE
extern int ccsRunning;
#endif

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
void CommunicationServerThread(int sleepTime);
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

CsvDeclare(CmiNodeState, NodeState);
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
    _Cmi_mype_global = _Cmi_mynode_global;
}

INLINE_KEYWORD int CmiNodeSpan() {
  return 1;
}
#else
/************** SMP *******************/
INLINE_KEYWORD CMIQueue CmiMyRecvQueue() {
    return CmiGetState()->recv;
}

#if CMK_NODE_QUEUE_AVAILABLE
INLINE_KEYWORD CMIQueue CmiMyNodeQueue() {
    return CsvAccess(NodeState).NodeRecv;
}
#endif

INLINE_KEYWORD int CmiMyPe() {
    return CmiGetState()->pe;
}
INLINE_KEYWORD int CmiNodeSpan() {
  return (CmiMyNodeSize() + 1);
}
INLINE_KEYWORD int CmiMyPeGlobal() {
    return CmiGetPeGlobal(CmiGetState()->pe,CmiMyPartition());
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

static int CmiState_hasMessage() {
  CmiState cs = CmiGetState();
  return CmiIdleLock_hasMessage(cs);
}

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

    CMIQueuePush(cs->recv,(char*)msg);

#if CMK_SHARED_VARS_POSIX_THREADS_SMP
  if (_Cmi_sleepOnIdle)
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
    CMIQueuePush(CsvAccess(NodeState).NodeRecv,msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);

#if CMK_SHARED_VARS_POSIX_THREADS_SMP
    if (_Cmi_sleepOnIdle)
#endif
    {
        CmiState cs=CmiGetStateN(0);
        CmiIdleLock_addMessage(&cs->idle);
    }
}
#endif

/* This function handles the msg received as which queue to push into */
static INLINE_KEYWORD void handleOneRecvedMsg(int size, char *msg) {

#if CMK_SMP_TRACE_COMMTHREAD
    TRACE_COMM_CREATION(TraceTimerCommon(), msg);
#endif

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
//inter-partition send
void CmiInterSyncSendFn(int destPE, int partition, int size, char *msg) {
    char *dupmsg = CopyMsg(msg, size);
    CmiInterFreeSendFn(destPE, partition, size, dupmsg);
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

//declaration so that it can be used
CmiCommHandle CmiInterSendNetworkFunc(int destPE, int partition, int size, char *msg, int mode);
//I am changing this function to offload task to a generic function - the one
//that handles sending to any partition
INLINE_KEYWORD CmiCommHandle CmiSendNetworkFunc(int destPE, int size, char *msg, int mode) {
  return CmiInterSendNetworkFunc(destPE, CmiMyPartition(), size, msg, mode);
}
//the generic function that replaces the older one
CmiCommHandle CmiInterSendNetworkFunc(int destPE, int partition, int size, char *msg, int mode)
{
        int rank;
        int destLocalNode = CmiNodeOf(destPE); 
        int destNode = CmiGetNodeGlobal(destLocalNode,partition); 
#if CMK_USE_PXSHM      
        if ((partition == CmiMyPartition()) && CmiValidPxshm(destLocalNode, size)) {
          CmiSendMessagePxshm(msg, size, destLocalNode, &refcount);
          //for (int i=0; i<refcount; i++) CmiReference(msg);
          return 0;
        }
#endif
#if CMK_USE_XPMEM     
        if ((partition == CmiMyPartition()) && CmiValidXpmem(destLocalNode, size)) {
          CmiSendMessageXpmem(msg, size, destLocalNode, &refcount);
          //for (int i=0; i<refcount; i++) CmiReference(msg);
          return 0;
        }
#endif
#if CMK_PERSISTENT_COMM
        if (CpvAccess(phs)) {
          if (size > PERSIST_MIN_SIZE) {
            CmiAssert(CpvAccess(curphs) < CpvAccess(phsSize));
            PersistentSendsTable *slot = (PersistentSendsTable *)CpvAccess(phs)[CpvAccess(curphs)];
            CmiAssert(CmiNodeOf(slot->destPE) == destLocalNode);
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
    return LrtsSendFunc(destNode, CmiGetPeGlobal(destPE,partition), size, msg, mode);
}

//I am changing this function to offload task to a generic function - the one
//that handles sending to any partition
INLINE_KEYWORD void CmiFreeSendFn(int destPE, int size, char *msg) {
    CmiInterFreeSendFn(destPE, CmiMyPartition(), size, msg);
}
//and the generic implementation - I may be in danger of making the frequent
//case slower - two extra comparisons may happen
void CmiInterFreeSendFn(int destPE, int partition, int size, char *msg) {
    CMI_SET_BROADCAST_ROOT(msg, 0);
    CQdCreate(CpvAccess(cQdState), 1);
    if (CmiMyPe()==destPE && partition == CmiMyPartition()) {
        CmiSendSelf(msg);
#if CMK_PERSISTENT_COMM
        if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
    } 
    else {
        int destNode = CmiNodeOf(destPE);
        int destRank = CmiRankOf(destPE);
#if CMK_SMP
        if (CmiMyNode()==destNode && partition == CmiMyPartition()) {
            CmiPushPE(destRank, msg);
#if CMK_PERSISTENT_COMM
            if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
            return;
        }
#endif
        CMI_DEST_RANK(msg) = destRank;
        CmiInterSendNetworkFunc(destPE, partition, size, msg, P2P_SYNC);

#if CMK_PERSISTENT_COMM
        if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
    }
}
#endif

#if USE_COMMON_ASYNC_P2P
//not implementing it for partition
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
    CMIQueuePush(CsvAccess(NodeState).NodeRecv, msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
}

//I think this #if is incorrect - should be SYNC_P2P
#if USE_COMMON_SYNC_P2P
INLINE_KEYWORD void CmiSyncNodeSendFn(int destNode, int size, char *msg) {
    char *dupmsg = CopyMsg(msg, size);
    CmiFreeNodeSendFn(destNode, size, dupmsg);
}
//inter-partition send
void CmiInterSyncNodeSendFn(int destNode, int partition, int size, char *msg) {
    char *dupmsg = CopyMsg(msg, size);
    CmiInterFreeNodeSendFn(destNode, partition, size, dupmsg);
}

//again, offloading the task to a generic function
INLINE_KEYWORD void CmiFreeNodeSendFn(int destNode, int size, char *msg) {
  CmiInterFreeNodeSendFn(destNode, CmiMyPartition(), size, msg);
}
//and the inter-partition function
void CmiInterFreeNodeSendFn(int destNode, int partition, int size, char *msg) {
    CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
    CQdCreate(CpvAccess(cQdState), 1);
    CMI_SET_BROADCAST_ROOT(msg, 0);
    if (destNode == CmiMyNode() && CmiMyPartition() == partition) {
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
        CmiInterSendNetworkFunc(CmiNodeFirst(destNode), partition, size, msg, P2P_SYNC);
    }
#if CMK_PERSISTENT_COMM
    if (CpvAccess(phs)) CpvAccess(curphs)++;
#endif
}
#endif

#if USE_COMMON_ASYNC_P2P
//not implementing it for partition
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

// functions related to partition
#if defined(_WIN32) && ! defined(__CYGWIN__)
  /* strtok is thread safe in VC++ */
#define strtok_r(x,y,z) strtok(x,y)
#endif

#include "TopoManager.h"
extern void createCustomPartitions(int numparts, int *partitionSize, int *nodeMap);
extern void setDefaultPartitionParams();

void create_topoaware_partitions() {
  int i, j, numparts_bak;
  Partition_Type type_bak;
  int *testMap;

  _partitionInfo.nodeMap = (int*)malloc(CmiNumNodesGlobal()*sizeof(int));
  _MEMCHECK(_partitionInfo.nodeMap);

  type_bak = _partitionInfo.type;
  _partitionInfo.type = PARTITION_SINGLETON;

  numparts_bak = _partitionInfo.numPartitions;
  _partitionInfo.numPartitions = 1;

  _partitionInfo.myPartition = 0;

  _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
  
  TopoManager_init();
  if(_partitionInfo.scheme == 100) {
    createCustomPartitions(numparts_bak, _partitionInfo.partitionSize, _partitionInfo.nodeMap);       
  } else {
    TopoManager_createPartitions(_partitionInfo.scheme, numparts_bak, _partitionInfo.nodeMap);
  }
  TopoManager_free();
  
  _partitionInfo.type = type_bak;
  _partitionInfo.numPartitions = numparts_bak;

#if CMK_ERROR_CHECKING
  testMap = (int*)calloc(CmiNumNodesGlobal(), sizeof(int));
  for(i = 0; i < CmiNumNodesGlobal(); i++) {
    assert(_partitionInfo.nodeMap[i] >= 0);
    assert(_partitionInfo.nodeMap[i] < CmiNumNodesGlobal());
    assert(testMap[_partitionInfo.nodeMap[i]] == 0);
    testMap[_partitionInfo.nodeMap[i]] = 1;
  }
  free(testMap);
#endif

  for(i = 0; i < _partitionInfo.numPartitions; i++) {
    int endnode = _partitionInfo.partitionPrefix[i] + _partitionInfo.partitionSize[i];
    for(j = _partitionInfo.partitionPrefix[i]; j < endnode; j++) {
      if(_partitionInfo.nodeMap[j] == CmiMyNodeGlobal()) {
        _Cmi_mynode = j - _partitionInfo.partitionPrefix[i];
        _partitionInfo.myPartition = i;
      }
    }
  }
}

void CmiSetNumPartitions(int nump) {
  _partitionInfo.numPartitions = nump;
}

void CmiSetMasterPartition() {
  if(!CmiMyNodeGlobal() && _partitionInfo.type != PARTITION_DEFAULT) {
    CmiAbort("setMasterPartition used with incompatible option\n");
  }
  _partitionInfo.type = PARTITION_MASTER;
} 

void CmiSetPartitionSizes(char *sizes) {
  int length = strlen(sizes);
  _partitionInfo.partsizes = (char*)malloc((length+1)*sizeof(char));

  if(!CmiMyNodeGlobal() && _partitionInfo.type != PARTITION_DEFAULT) {
    CmiAbort("setPartitionSizes used with incompatible option\n");
  }

  memcpy(_partitionInfo.partsizes, sizes, length*sizeof(char));
  _partitionInfo.partsizes[length] = '\0';
  _partitionInfo.type = PARTITION_PREFIX;
}

void CmiSetPartitionScheme(int scheme) {
  _partitionInfo.scheme = scheme;
  _partitionInfo.isTopoaware = 1;
}

void CmiSetCustomPartitioning() {
  _partitionInfo.scheme = 100;
  _partitionInfo.isTopoaware = 1;
}

static int create_partition_map( char **argv)
{
  char* token, *tptr;
  int i, flag;
  
  _partitionInfo.numPartitions = 1; 
  _partitionInfo.type = PARTITION_DEFAULT;
  _partitionInfo.partsizes = NULL;
  _partitionInfo.scheme = 0;
  _partitionInfo.isTopoaware = 0;

  setDefaultPartitionParams();

  if(!CmiGetArgIntDesc(argv,"+partitions", &_partitionInfo.numPartitions,"number of partitions")) {
    CmiGetArgIntDesc(argv,"+replicas", &_partitionInfo.numPartitions,"number of partitions");
  }

#if CMK_MULTICORE
  if(_partitionInfo.numPartitions != 1) {
    CmiAbort("+partitions other than 1 is not allowed for multicore build\n");
  }
#endif

  _partitionInfo.partitionSize = (int*)calloc(_partitionInfo.numPartitions,sizeof(int));
  _partitionInfo.partitionPrefix = (int*)calloc(_partitionInfo.numPartitions,sizeof(int));
  
  if (CmiGetArgFlagDesc(argv,"+master_partition","assign a process as master partition")) {
    _partitionInfo.type = PARTITION_MASTER;
  }
 
  if (CmiGetArgStringDesc(argv, "+partition_sizes", &_partitionInfo.partsizes, "size of partitions")) {
    if(!CmiMyNodeGlobal() && _partitionInfo.type != PARTITION_DEFAULT) {
      CmiAbort("+partition_sizes used with incompatible option, possibly +master_partition\n");
    }
    _partitionInfo.type = PARTITION_PREFIX;
  }

  if (CmiGetArgFlagDesc(argv,"+partition_topology","topology aware partitions")) {
    _partitionInfo.isTopoaware = 1;
    _partitionInfo.scheme = 1;
  }  

  if (CmiGetArgIntDesc(argv,"+partition_topology_scheme", &_partitionInfo.scheme, "topology aware partitioning scheme")) {
    _partitionInfo.isTopoaware = 1;
  }

  if (CmiGetArgFlagDesc(argv,"+use_custom_partition", "custom partitioning scheme")) {
    _partitionInfo.scheme = 100;
    _partitionInfo.isTopoaware = 1;
  }

  if(_partitionInfo.type == PARTITION_DEFAULT) {
    if((_Cmi_numnodes_global % _partitionInfo.numPartitions) != 0) {
      CmiAbort("Number of partitions does not evenly divide number of processes. Aborting\n");
    }
    _partitionInfo.partitionPrefix[0] = 0;
    _partitionInfo.partitionSize[0] = _Cmi_numnodes_global / _partitionInfo.numPartitions;
    for(i = 1; i < _partitionInfo.numPartitions; i++) {
      _partitionInfo.partitionSize[i] = _partitionInfo.partitionSize[i-1];
      _partitionInfo.partitionPrefix[i] = _partitionInfo.partitionPrefix[i-1] + _partitionInfo.partitionSize[i-1];
    } 
    _partitionInfo.myPartition = _Cmi_mynode_global / _partitionInfo.partitionSize[0];
  } else if(_partitionInfo.type == PARTITION_MASTER) {
    if(((_Cmi_numnodes_global-1) % (_partitionInfo.numPartitions-1)) != 0) {
      CmiAbort("Number of non-master partitions does not evenly divide number of processes minus one. Aborting\n");
    }
    _partitionInfo.partitionSize[0] = 1;
    _partitionInfo.partitionPrefix[0] = 0;
    _partitionInfo.partitionSize[1] = (_Cmi_numnodes_global-1) / (_partitionInfo.numPartitions-1);
    _partitionInfo.partitionPrefix[1] = 1;
    for(i = 2; i < _partitionInfo.numPartitions; i++) {
      _partitionInfo.partitionSize[i] = _partitionInfo.partitionSize[i-1];
      _partitionInfo.partitionPrefix[i] = _partitionInfo.partitionPrefix[i-1] + _partitionInfo.partitionSize[i-1];
    } 
    _partitionInfo.myPartition = 1 + (_Cmi_mynode_global-1) / _partitionInfo.partitionSize[1];
    if(!_Cmi_mynode_global) 
      _partitionInfo.myPartition = 0;
  } else if(_partitionInfo.type == PARTITION_PREFIX) {
    token = strtok_r(_partitionInfo.partsizes, ",", &tptr);
    while (token)
    {
      int i,j;
      int hasdash=0, hascolon=0, hasdot=0;
      int start, end, stride = 1, block = 1, size;
      for (i = 0; i < strlen(token); i++) {
        if (token[i] == '-') hasdash=1;
        else if (token[i] == ':') hascolon=1;
        else if (token[i] == '.') hasdot=1;
      }
      if (hasdash) {
        if (hascolon) {
          if (hasdot) {
            if (sscanf(token, "%d-%d:%d.%d#%d", &start, &end, &stride, &block, &size) != 5)
              printf("Warning: Check the format of \"%s\".\n", token);
          }
          else {
            if (sscanf(token, "%d-%d:%d#%d", &start, &end, &stride, &size) != 4)
              printf("Warning: Check the format of \"%s\".\n", token);
          }
        }
        else {
          if (sscanf(token, "%d-%d#%d", &start, &end, &size) != 3)
            printf("Warning: Check the format of \"%s\".\n", token);
        }
      }
      else {
        if (sscanf(token, "%d#%d", &start, &size) != 2) {
          printf("Warning: Check the format of \"%s\".\n", token);
        }
        end = start;
      }
      if (block > stride) {
        printf("Warning: invalid block size in \"%s\" ignored.\n", token);
        block = 1;
      }
      for (i = start; i <= end; i += stride) {
        for (j = 0; j < block; j++) {
          if (i + j > end) break;
          _partitionInfo.partitionSize[i+j] = size;
        }
      }
      token = strtok_r(NULL, ",", &tptr);
    }
    _partitionInfo.partitionPrefix[0] = 0;
    _partitionInfo.myPartition = 0;
    for(i = 1; i < _partitionInfo.numPartitions; i++) {
      if(_partitionInfo.partitionSize[i-1] <= 0) {
        CmiAbort("Partition size has to be greater than zero.\n");
      }
      _partitionInfo.partitionPrefix[i] = _partitionInfo.partitionPrefix[i-1] + _partitionInfo.partitionSize[i-1];
      if((_Cmi_mynode_global >= _partitionInfo.partitionPrefix[i]) && (_Cmi_mynode_global < (_partitionInfo.partitionPrefix[i] + _partitionInfo.partitionSize[i]))) {
        _partitionInfo.myPartition = i;
      }
    } 
    if(_partitionInfo.partitionSize[i-1] <= 0) {
      CmiAbort("Partition size has to be greater than zero.\n");
    }
  }
  _Cmi_mynode = _Cmi_mynode - _partitionInfo.partitionPrefix[_partitionInfo.myPartition];

  if(_partitionInfo.isTopoaware) {
    create_topoaware_partitions();
  }
}

void CmiCreatePartitions(char **argv) {
  _Cmi_numnodes_global = _Cmi_numnodes;
  _Cmi_mynode_global = _Cmi_mynode;
  _Cmi_numpes_global = _Cmi_numnodes_global * _Cmi_mynodesize;

  if(Cmi_nodestart != -1) {
    Cmi_nodestartGlobal = Cmi_nodestart;
  } else {
    Cmi_nodestartGlobal =  _Cmi_mynode_global * _Cmi_mynodesize;
  }

  //creates partitions, reset _Cmi_mynode to be the new local rank
  if(_partitionInfo.numPartitions > _Cmi_numnodes_global) {
    CmiAbort("Number of partitions requested in greater than the number of nodes\n");
  }
  create_partition_map(argv);
  
  //reset other local variables
  _Cmi_numnodes = CmiMyPartitionSize();
  //mype and numpes will be set following this
}

INLINE_KEYWORD int node_lToGTranslate(int node, int partition) {
  int rank;
  if(_partitionInfo.type == PARTITION_SINGLETON) { 
    return node;
  } else if(_partitionInfo.type == PARTITION_DEFAULT) { 
    rank =  (partition * _partitionInfo.partitionSize[0]) + node;
  } else if(_partitionInfo.type == PARTITION_MASTER) {
    if(partition == 0) {
      CmiAssert(node == 0);
      rank =  0;
    } else {
      rank = 1 + ((partition - 1) * _partitionInfo.partitionSize[1]) + node;
    }
  } else if(_partitionInfo.type == PARTITION_PREFIX) {
    rank = _partitionInfo.partitionPrefix[partition] + node;
  } else {
    CmiAbort("Partition type did not match any of the supported types\n");
  }
  if(_partitionInfo.isTopoaware) {
    return _partitionInfo.nodeMap[rank];
  } else {
    return rank;
  }
}

INLINE_KEYWORD int pe_lToGTranslate(int pe, int partition) {
  if(_partitionInfo.type == PARTITION_SINGLETON) 
    return pe;

  if(pe < CmiPartitionSize(partition)*CmiMyNodeSize()) {
    return node_lToGTranslate(CmiNodeOf(pe),partition)*CmiMyNodeSize() + CmiRankOf(pe);
  }

  return CmiNumPesGlobal() + node_lToGTranslate(pe - CmiPartitionSize(partition)*CmiMyNodeSize(), partition);
}

INLINE_KEYWORD int node_gToLTranslate(int node) {
  CmiAbort("Conversion from global rank to local rank is not supported. Please contact Charm++ developers with the use case.\n");
}

INLINE_KEYWORD int pe_gToLTranslate(int pe) {
  CmiAbort("Conversion from global rank to local rank is not supported. Please contact Charm++ developers with the use case.\n");
}
//end of functions related to partition

extern int quietMode;
extern int quietModeRequested;

/* ##### Beginning of Functions Related with Machine Startup ##### */
void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret) {
    int _ii;
    int tmp;
    //handle output to files for partition if requested
    char *stdoutbase,*stdoutpath;


#if CMK_WITH_STATS
    MSG_STATISTIC = CmiGetArgFlag(argv, "+msgstatistic");
#endif

  if (CmiGetArgFlagDesc(argv,"++quiet","Omit non-error runtime messages")) {
    quietModeRequested = quietMode = 1;
  }

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

#if MACHINE_DEBUG_LOG
    char ln[200];
    sprintf(ln,"debugLog.%d", _Cmi_mynode);
    debugLog=fopen(ln,"w");
    if (debugLog == NULL)
    {
        CmiAbort("debug file not open\n");
    }
#endif

    LrtsInit(&argc, &argv, &_Cmi_numnodes, &_Cmi_mynode);


    if (_Cmi_mynode==0) {
#if !CMK_SMP 
      if (!quietMode) printf("Charm++> Running in non-SMP mode: numPes %d\n",_Cmi_numnodes);
      MACHSTATE1(4,"running nonsmp %d", _Cmi_mynode)
#else

#if !CMK_MULTICORE
      if (!quietMode) printf("Charm++> Running in SMP mode: numNodes %d,  %d worker threads per process\n", CmiNumNodes(),_Cmi_mynodesize);
      if (Cmi_smp_mode_setting == COMM_THREAD_SEND_RECV) {
        if (!quietMode) printf("Charm++> The comm. thread both sends and receives messages\n");
      } else if (Cmi_smp_mode_setting == COMM_THREAD_ONLY_RECV) {
        if (!quietMode) printf("Charm++> The comm. thread only receives messages, while work threads send messages\n");
      } else if (Cmi_smp_mode_setting == COMM_WORK_THREADS_SEND_RECV) {
        if (!quietMode) printf("Charm++> Both  comm. thread and worker thread send and messages\n");
      } else if (Cmi_smp_mode_setting == COMM_THREAD_NOT_EXIST) {
        if (!quietMode) printf("Charm++> There's no comm. thread. Work threads both send and receive messages\n");
      } else {
        CmiAbort("Charm++> Invalid SMP mode setting\n");
      }
#else
      if (!quietMode) printf("Charm++> Running in Multicore mode:  %d threads\n", _Cmi_mynodesize);
#endif
      
#endif
    }

    CmiCreatePartitions(argv);

    _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
    Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
    Cmi_argvcopy = CmiCopyArgs(argv);
    Cmi_argv = argv;
    Cmi_startfn = fn;
    Cmi_usrsched = usched;

    if ( CmiGetArgStringDesc(argv,"+stdout",&stdoutbase,"base filename to redirect partition stdout to") ) {
      stdoutpath = malloc(strlen(stdoutbase) + 30);
      sprintf(stdoutpath, stdoutbase, CmiMyPartition(), CmiMyPartition(), CmiMyPartition());
      if ( ! strcmp(stdoutpath, stdoutbase) ) {
        sprintf(stdoutpath, "%s.%d", stdoutbase, CmiMyPartition());
      }
      if ( CmiMyPartition() == 0 && CmiMyNode() == 0 && !quietMode) {
        printf("Redirecting stdout to files %s through %d\n",stdoutpath,CmiNumPartitions()-1);
      }
      if ( ! freopen(stdoutpath, "a", stdout) ) {
        fprintf(stderr,"Rank %d failed redirecting stdout to file %s: %s\n", CmiMyNodeGlobal(), stdoutpath,
            strerror(errno));
        CmiAbort("Error redirecting stdout to file.");
      }
      _writeToStdout = 0;
      free(stdoutpath);
    }
#if CMK_SMP
    comm_mutex=CmiCreateLock();
#endif

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
    interopCommThdExitLock = CmiCreateLock();
#endif

#if CMK_OFFLOAD_BCAST_PROCESS
    /* the actual queues should be created on comm thread considering NUMA in SMP */
    CsvInitialize(CMIQueue, procBcastQ);
#if CMK_NODE_QUEUE_AVAILABLE
    CsvInitialize(CMIQueue, nodeBcastQ);
#endif
#endif

#if CMK_SMP && CMK_LEVERAGE_COMMTHREAD
    CsvInitialize(CMIQueue, notifyCommThdMsgBuffer);
#endif

    CmiStartThreads(argv);

    ConverseRunPE(initret);
}

extern void ConverseCommonInit(char **argv);
extern void CthInit(char **argv);
static void ConverseRunPE(int everReturn) {
    CmiState cs;
    char** CmiMyArgv;

#if CMK_CCS_AVAILABLE
/**
 * The reason to initialize this variable here:
 * cmiArgDebugFlag is possibly accessed in CmiPrintf/CmiError etc.,
 * therefore, we have to initialize this variable before any calls
 * to those functions (such as CmiPrintf). Otherwise, we may encounter
 * a memory segmentation fault (bad memory access). Note, even
 * testing CpvInitialized(cmiArgDebugFlag) doesn't help to solve
 * this problem because the variable indicating whether cmiArgDebugFlag is
 * initialized or not is not initialized, thus possibly causing another
 * bad memory access. --Chao Mei
 */
  CpvInitialize(int, cmiArgDebugFlag);
  CpvAccess(cmiArgDebugFlag) = 0;
#endif

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
        CsvAccess(procBcastQ) = CMIQueueCreate();
#if CMK_NODE_QUEUE_AVAILABLE
        CsvAccess(nodeBcastQ) = CMIQueueCreate();
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
    
    // register idle events

#if CMK_SMP
    {
      CmiIdleState *sidle=CmiNotifyGetState();
      CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)sidle);
      CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)sidle);
    }
#else
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle, NULL);
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle, NULL);
#endif


    LrtsPostCommonInit(everReturn);

#if CMK_SMP && CMK_LEVERAGE_COMMTHREAD
    CmiInitNotifyCommThdScheme();
#endif
    /* Converse initialization finishes, immediate messages can be processed.
       node barrier previously should take care of the node synchronization */
    _immediateReady = 1;

    /* communication thread */
    if (CmiMyRank() == CmiMyNodeSize()) {
      Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
      if(!CharmLibInterOperate) {
        while (1) CommunicationServerThread(5);
      }
    } else { /* worker thread */
      if (!everReturn) {
        Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
        if (Cmi_usrsched==0) CsdScheduler(-1);
        if(!CharmLibInterOperate) {
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
  
#if CMK_USE_XPMEM
        CmiExitXpmem();
#endif
        CmiNodeAllBarrier();
        LrtsExit();
    }
#endif
}

void CommunicationServerThread(int sleepTime) {
    CommunicationServer(sleepTime);
#if CMK_IMMEDIATE_MSG
    CmiHandleImmediate();
#endif
}

void ConverseExit(void) {
    int i;
    if (quietModeRequested) quietMode = 1;
#if !CMK_SMP || CMK_SMP_NO_COMMTHD
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
    if (CmiMyPe() == 0) 
      CmiPrintf("[Partition %d][Node %d] End of program\n",CmiMyPartition(),CmiMyNode());
#endif

#if !CMK_SMP || CMK_BLUEGENEQ
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
    CmiNodeAllBarrier();
#if CMK_SMP_NO_COMMTHD
#if CMK_USE_XPMEM
    if (CmiMyRank() == 0) CmiExitXpmem();
    CmiNodeAllBarrier();
#endif
    if (CmiMyRank() == 0) LrtsExit();
#endif
    if(CharmLibInterOperate && !userDrivenMode)
      CmiYield();
    else 
      while (1) CmiYield();
#endif
}
/* ##### End of Functions Related with Machine Running ##### */

void CmiAbortHelper(const char *source, const char *message, const char *suggestion,
                    int tellDebugger, int framesToSkip) {
  if (tellDebugger)
    CpdAborting(message);

  if (CmiNumPartitions() == 1) {
    CmiError("------------- Processor %d Exiting: %s ------------\n"
             "Reason: %s\n", CmiMyPe(), source, message);
  } else {
    CmiError("------- Partition %d Processor %d Exiting: %s ------\n"
             "Reason: %s\n", CmiMyPartition(), CmiMyPe(), source, message);
  }

  if (suggestion && suggestion[0])
    CmiError("Suggestion: %s\n", suggestion);

  CmiPrintStackTrace(framesToSkip);

#if CMK_USE_PXSHM
  CmiExitPxshm();
#endif
#if CMK_USE_XPMEM
  CmiExitXpmem();
#endif
  LrtsAbort(message);
}

void CmiAbort(const char *message) {
  CmiAbortHelper("Called CmiAbort", message, NULL, 1, 0);
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
#if CMK_CCS_AVAILABLE
    if (CmiNumPes() == 1 && CmiNumPartitions() == 1 && ccsRunning != 1) return NULL;
#else
    if (CmiNumPes() == 1 && CmiNumPartitions() == 1) return NULL;
#endif
#endif

    MACHSTATE2(3, "[%p] CmiGetNonLocal begin %d{", cs, CmiMyPe());
    CmiIdleLock_checkMessage(&cs->idle);
    /* ?????although it seems that lock is not needed, I found it crashes very often
       on mpi-smp without lock */
    msg = CMIQueuePop(cs->recv);
#if (!CMK_SMP || CMK_SMP_NO_COMMTHD) && !CMK_MULTICORE
    if (!msg) {
       AdvanceCommunication(0);
       msg = CMIQueuePop(cs->recv);
    }
#else
//    LrtsPostNonLocal();
#endif
#if CMK_CCS_AVAILABLE
    if(msg != NULL && CmiNumPes() == 1 && CmiNumPartitions() == 1 )
    {
      ccsRunning = 0;
    }
#endif

    MACHSTATE3(3,"[%p] CmiGetNonLocal from queue %p with msg %p end }",CmiGetState(),(cs->recv), msg);

    return msg;
}

#if CMK_NODE_QUEUE_AVAILABLE
void *CmiGetNonLocalNodeQ(void) {
    CmiState cs = CmiGetState();
    char *result = 0;
    CmiIdleLock_checkMessage(&cs->idle);
    if (!CMIQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
        MACHSTATE1(3,"CmiGetNonLocalNodeQ begin %d {", CmiMyPe());
        CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
        result = (char *) CMIQueuePop(CsvAccess(NodeState).NodeRecv);
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
    if(s!= NULL){
        s->sleepMs=0;
        s->nIdles=0;
    }
    LrtsBeginIdle();
}

/*Number of times to spin before sleeping*/
#define SPINS_BEFORE_SLEEP 20
static void CmiNotifyStillIdle(CmiIdleState *s) {
    MACHSTATE1(2,"still idle (%d) begin {",CmiMyPe())
#if (!CMK_SMP || CMK_SMP_NO_COMMTHD) && !CMK_MULTICORE
    AdvanceCommunication(1);
#else
    LrtsPostNonLocal();

#if CMK_SHARED_VARS_POSIX_THREADS_SMP
    if (_Cmi_sleepOnIdle)
#endif
    {
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
    LrtsStillIdle();
    CsdResetPeriodic();
    MACHSTATE1(2,"still idle (%d) end {",CmiMyPe())
}

/* usually called in non-smp mode */
void CmiNotifyIdle(void) {
    AdvanceCommunication(1);
    CmiYield();
    LrtsNotifyIdle();
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

/************Barrier Related Functions****************/
/* must be called on all ranks including comm thread in SMP */
int CmiBarrier() {
#if CMK_SMP
    /* make sure all ranks reach here, otherwise comm threads may reach barrier ignoring other ranks  */
    CmiNodeAllBarrier();
#endif
#if ( CMK_SMP && !CMK_SMP_NO_COMMTHD)
    if (CmiMyRank() == CmiMyNodeSize())
    {
#else
    if (CmiMyRank() == 0)
    {
#endif
        LrtsBarrier();
    }
#if CMK_SMP
    CmiNodeAllBarrier();
#endif
    return 0;
}


