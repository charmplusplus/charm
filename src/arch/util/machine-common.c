#if CMK_C_INLINE
#define INLINE_KEYWORD inline
#else
#define INLINE_KEYWORD
#endif

/* ###Beginning of Broadcast related definitions ### */
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
#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank
#define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);

#if USE_COMMON_SYNC_BCAST || USE_COMMON_ASYNC_BCAST
#if !CMK_BROADCAST_SPANNING_TREE && !CMK_BROADCAST_HYPERCUBE
#warning "Broadcast function is based on the plain P2P O(P)-message scheme!!!"
#endif
#endif

/**
 * For some machine layers such as on Active Message framework,
 * the receiver callback is usally executed on an internal
 * thread (i.e. not the flow managed by ours). Therefore, for
 * forwarding broadcast messages, we could have a choice whether
 * to offload such function to the flow we manage such as the
 * communication thread. -Chao Mei
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

#if CMK_BROADCAST_HYPERCUBE
/* ceil(log2(CmiNumNodes)) except when _Cmi_numnodes is 1, used for hypercube */
static int CmiNodesDim;
#endif
/* ###End of Broadcast related definitions ### */


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
#define P2P_SYNC 0x1
#define P2P_ASYNC 0x2
#define BCAST_SYNC 0x3
#define BCAST_ASYNC 0x4

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
 *  --Chao Mei
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

static void handleOneBcastMsg(int size, char *msg);
static void processBcastQs();

/* Utility functions for forwarding broadcast messages,
 * should not be used in machine-specific implementations
 * except in some special occasions.
 */
static void processProcBcastMsg(int size, char *msg);
static void processNodeBcastMsg(int size, char *msg);
static void SendSpanningChildrenProc(int size, char *msg);
static void SendHyperCubeProc(int size, char *msg);
#if CMK_NODE_QUEUE_AVAILABLE
static void SendSpanningChildrenNode(int size, char *msg);
static void SendHyperCubeNode(int size, char *msg);
#endif

static void SendSpanningChildren(int size, char *msg, int rankToAssign, int startNode);
static void SendHyperCube(int size,  char *msg, int rankToAssign, int startNode);
/* send to other ranks except me on the same node*/
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

void CmiSyncBroadcastFn(int size, char *msg);
CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg);
void CmiFreeBroadcastFn(int size, char *msg);

void CmiSyncBroadcastAllFn(int size, char *msg);
CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg);
void CmiFreeBroadcastAllFn(int size, char *msg);

#if CMK_NODE_QUEUE_AVAILABLE
static void CmiSendNodeSelf(char *msg);

void CmiSyncNodeSendFn(int destNode, int size, char *msg);
CmiCommHandle CmiAsyncNodeSendFn(int destNode, int size, char *msg);
void CmiFreeNodeSendFn(int destNode, int size, char *msg);

void CmiSyncNodeBroadcastFn(int size, char *msg);
CmiCommHandle CmiAsyncNodeBroadcastFn(int size, char *msg);
void CmiFreeNodeBroadcastFn(int size, char *msg);

void CmiSyncNodeBroadcastAllFn(int size, char *msg);
CmiCommHandle CmiAsyncNodeBroadcastAllFn(int size, char *msg);
void CmiFreeNodeBroadcastAllFn(int size, char *msg);
#endif

/* Functions and variables regarding machine startup */
static char     **Cmi_argv;
static char     **Cmi_argvcopy;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */
void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret);
static void ConverseRunPE(int everReturn);

static void MachineSpecificInit(int argc, char **argv, int *numNodes, int *myNodeID);
/* Used in ConverseRunPE: one is called before ConverseCommonInit;
  * The other is called after ConverseCommonInit as some data structures
  * have been initialized, such as the tracing-relatd stuff --Chao Mei
  */
static void MachineSpecificPreCommonInit(int everReturn);
static void MachineSpecificPostCommonInit(int everReturn);

/* Functions regarding machine running on every proc */
static void AdvanceCommunication();
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);
void ConverseExit(void);

static void MachineSpecificAdvanceCommunication();
static void MachineSpecificDrainResources(); /* used when exit */
static void MachineSpecificExit();

/* Functions providing incoming network messages */
void *CmiGetNonLocal(void);
#if CMK_NODE_QUEUE_AVAILABLE
void *CmiGetNonLocalNodeQ(void);
#endif
void MachineSpecificPostNonLocal(void);

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
INLINE_KEYWORD int CmiMyPe(void) {
    return CmiGetState()->pe;
}
INLINE_KEYWORD int CmiMyRank(void) {
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

#include "immediate.c"

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

    PCQueuePush(cs->recv,msg);
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
    if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
        CmiPushNode(msg);
    else
#endif
        CmiPushPE(CMI_DEST_RANK(msg), msg);

}


/* ##### Beginning of Broadcast-related functions' defitions ##### */
static void handleOneBcastMsg(int size, char *msg) {
    CmiAssert(CMI_BROADCAST_ROOT(msg)!=0);
#if CMK_OFFLOAD_BCAST_PROCESS
    if (CMI_BROADCAST_ROOT(msg)>0) {
        PCQueuePush(CsvAccess(procBcastQ), msg);
    } else {
#if CMK_NODE_QUEUE_AVAILABLE
        PCQueuePush(CsvAccess(nodeBcastQ), msg);
#endif
    }
#else
    if (CMI_BROADCAST_ROOT(msg)>0) {
        processProcBcastMsg(size, msg);
    } else {
#if CMK_NODE_QUEUE_AVAILABLE
        processNodeBcastMsg(size, msg);
#endif
    }
#endif
}

static void processBcastQs() {
#if CMK_OFFLOAD_BCAST_PROCESS
    char *msg;
    do {
        msg = PCQueuePop(CsvAccess(procBcastQ));
        if (!msg) break;
        MACHSTATE2(4, "[%d]: process a proc-level bcast msg %p begin{", CmiMyNode(), msg);
        processProcBcastMsg(CMI_MSG_SIZE(msg), msg);
        MACHSTATE2(4, "[%d]: process a proc-level bcast msg %p end}", CmiMyNode(), msg);
    } while (1);
#if CMK_NODE_QUEUE_AVAILABLE
    do {
        msg = PCQueuePop(CsvAccess(nodeBcastQ));
        if (!msg) break;
        MACHSTATE2(4, "[%d]: process a node-level bcast msg %p begin{", CmiMyNode(), msg);
        processNodeBcastMsg(CMI_MSG_SIZE(msg), msg);
        MACHSTATE2(4, "[%d]: process a node-level bcast msg %p end}", CmiMyNode(), msg);
    } while (1);
#endif
#endif
}

static INLINE_KEYWORD void processProcBcastMsg(int size, char *msg) {
#if CMK_BROADCAST_SPANNING_TREE
    SendSpanningChildrenProc(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    SendHyperCubeProc(size, msg);
#endif

    /* Since this function is only called on intermediate nodes,
     * the rank of this msg should be 0.
     */
    CmiAssert(CMI_DEST_RANK(msg)==0);
    /*CmiPushPE(CMI_DEST_RANK(msg), msg);*/
    CmiPushPE(0, msg);
}

static INLINE_KEYWORD void processNodeBcastMsg(int size, char *msg) {
#if CMK_BROADCAST_SPANNING_TREE
    SendSpanningChildrenNode(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    SendHyperCubeNode(size, msg);
#endif

    /* In SMP mode, this push operation needs to be executed
     * after forwarding broadcast messages. If it is executed
     * earlier, then during the bcast msg forwarding period,
     * the msg could be already freed on the worker thread.
     * As a result, the forwarded message could be wrong!
     * --Chao Mei
     */
    CmiPushNode(msg);
}

static void SendSpanningChildren(int size, char *msg, int rankToAssign, int startNode) {
#if CMK_BROADCAST_SPANNING_TREE
    int i, oldRank;

    oldRank = CMI_DEST_RANK(msg);
    /* doing this is to avoid the multiple assignment in the following for loop */
    CMI_DEST_RANK(msg) = rankToAssign;
    /* first send msgs to other nodes */
    CmiAssert(startNode >=0 &&  startNode<CmiNumNodes());
    for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int nd = CmiMyNode()-startNode;
        if (nd<0) nd+=CmiNumNodes();
        nd = BROADCAST_SPANNING_FACTOR*nd + i;
        if (nd > CmiNumNodes() - 1) break;
        nd += startNode;
        nd = nd%CmiNumNodes();
        CmiAssert(nd>=0 && nd!=CmiMyNode());
#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        CmiMachineSpecificSendFunc(nd, size, msg, P2P_SYNC);
#else
        char *newmsg = CopyMsg(msg, size);
        CmiMachineSpecificSendFunc(nd, size, newmsg, P2P_SYNC);
#endif
    }
    CMI_DEST_RANK(msg) = oldRank;
#endif
}
static void SendHyperCube(int size,  char *msg, int rankToAssign, int startNode) {
#if CMK_BROADCAST_HYPERCUBE
    int i, cnt, tmp, relDist, oldRank;
    const int dims=CmiNodesDim;

    oldRank = CMI_DEST_RANK(msg);
    /* doing this is to avoid the multiple assignment in the following for loop */
    CMI_DEST_RANK(msg) = rankToAssign;

    /* first send msgs to other nodes */
    relDist = CmiMyNode()-startNode;
    if (relDist < 0) relDist += CmiNumNodes();
    cnt=0;
    tmp = relDist;
    /* count how many zeros (in binary format) relDist has */
    for (i=0; i<dims; i++, cnt++) {
        if (tmp & 1 == 1) break;
        tmp = tmp >> 1;
    }

    /*CmiPrintf("ND[%d]: SendHypercube with snd=%d, relDist=%d, cnt=%d\n", CmiMyNode(), startnode, relDist, cnt);*/
    for (i = cnt-1; i >= 0; i--) {
        int nd = relDist + (1 << i);
        if (nd >= CmiNumNodes()) continue;
        nd = (nd+startNode)%CmiNumNodes();
        /*CmiPrintf("ND[%d]: send to node %d\n", CmiMyNode(), nd);*/
        CmiAssert(nd>=0 && nd!=CmiMyNode());
#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        CmiMachineSpecificSendFunc(nd, size, msg, P2P_SYNC);
#else
        char *newmsg = CopyMsg(msg, size);
        CmiMachineSpecificSendFunc(nd, size, newmsg, P2P_SYNC);
#endif
    }
    CMI_DEST_RANK(msg) = oldRank;
#endif
}

static void SendSpanningChildrenProc(int size, char *msg) {
    int startpe = CMI_BROADCAST_ROOT(msg)-1;
    int startnode = CmiNodeOf(startpe);
#if CMK_SMP
    if (startpe > CmiNumPes()) startnode = startpe - CmiNumPes();
#endif
    SendSpanningChildren(size, msg, 0, startnode);
#if CMK_SMP
    /* second send msgs to my peers on this node */
    SendToPeers(size, msg);
#endif
}

/* send msg along the hypercube in broadcast. (Sameer) */
static void SendHyperCubeProc(int size, char *msg) {
    int startpe = CMI_BROADCAST_ROOT(msg)-1;
    int startnode = CmiNodeOf(startpe);
#if CMK_SMP
    if (startpe > CmiNumPes()) startnode = startpe - CmiNumPes();
#endif
    SendHyperCube(size, msg, 0, startnode);
#if CMK_SMP
    /* second send msgs to my peers on this node */
    SendToPeers(size, msg);
#endif
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

#if CMK_NODE_QUEUE_AVAILABLE
static void SendSpanningChildrenNode(int size, char *msg) {
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;
    SendSpanningChildren(size, msg, DGRAM_NODEMESSAGE, startnode);
}
static void SendHyperCubeNode(int size, char *msg) {
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;
    SendHyperCube(size, msg, DGRAM_NODEMESSAGE, startnode);
}
#endif
/*##### End of Broadcast-related functions' defitions #####*/

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

void CmiFreeSendFn(int destPE, int size, char *msg) {
    CMI_SET_BROADCAST_ROOT(msg, 0);
    CQdCreate(CpvAccess(cQdState), 1);
    if (CmiMyPe()==destPE) {
        CmiSendSelf(msg);
    } else {
        int destNode = CmiNodeOf(destPE);
#if CMK_SMP
        if (CmiMyNode()==destNode) {
            CmiPushPE(CmiRankOf(destPE), msg);
            return;
        }
#endif
        CMI_DEST_RANK(msg) = CmiRankOf(destPE);
        CmiMachineSpecificSendFunc(destNode, size, msg, P2P_SYNC);
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
        return CmiMachineSpecificSendFunc(destPE, size, msg, P2P_ASYNC);
    }
}
#endif

#if USE_COMMON_SYNC_BCAST
/* Functions regarding broadcat op that sends to every one else except me */
void CmiSyncBroadcastFn(int size, char *msg) {
    int mype = CmiMyPe();
    int i;

    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
#if CMK_SMP
    /*record the rank to avoid re-sending the msg in  spanning tree or hypercube*/
    CMI_DEST_RANK(msg) = CmiMyRank();
#endif

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT(msg, mype+1);
    SendSpanningChildrenProc(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    CMI_SET_BROADCAST_ROOT(msg, mype+1);
    SendHyperCubeProc(size, msg);
#else
    for ( i=mype+1; i<_Cmi_numpes; i++ )
        CmiSyncSendFn(i, size, msg) ;
    for ( i=0; i<mype; i++ )
        CmiSyncSendFn(i, size, msg) ;
#endif

    /*CmiPrintf("In  SyncBroadcast broadcast\n");*/
}

void CmiFreeBroadcastFn(int size, char *msg) {
    CmiSyncBroadcastFn(size,msg);
    CmiFree(msg);
}
#endif

#if USE_COMMON_ASYNC_BCAST
/* FIXME: should use spanning or hypercube, but luckily async is never used */
CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg) {
    /*CmiPrintf("In  AsyncBroadcast broadcast\n");*/
    CmiAbort("CmiAsyncBroadcastFn should never be called");
    return 0;
}
#endif

/* Functions regarding broadcat op that sends to every one */
void CmiSyncBroadcastAllFn(int size, char *msg) {
    CmiSyncSendFn(CmiMyPe(), size, msg) ;
    CmiSyncBroadcastFn(size, msg);
}

void CmiFreeBroadcastAllFn(int size, char *msg) {
    CmiSendSelf(msg);
    CmiSyncBroadcastFn(size, msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg) {
    CmiSendSelf(CopyMsg(msg, size));
    return CmiAsyncBroadcastFn(size, msg);
}

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
        CmiMachineSpecificSendFunc(destNode, size, msg, P2P_SYNC);
    }
}
#endif

#if USE_COMMON_ASYNC_P2P
CmiCommHandle CmiAsyncNodeSendFn(int destNode, int size, char *msg) {
    if (destNode == CmiMyNode()) {
        CmiSyncNodeSendFn(destNode, size, msg);
        return 0;
    } else {
        return CmiMachineSpecificSendFunc(destNode, size, msg, P2P_ASYNC);
    }
}
#endif

#if USE_COMMON_SYNC_BCAST
void CmiSyncNodeBroadcastFn(int size, char *msg) {
    int mynode = CmiMyNode();
    int i;
    CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT(msg, -CmiMyNode()-1);
    SendSpanningChildrenNode(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    CMI_SET_BROADCAST_ROOT(msg, -CmiMyNode()-1);
    SendHyperCubeNode(size, msg);
#else
    for (i=mynode+1; i<CmiNumNodes(); i++)
        CmiSyncNodeSendFn(i, size, msg);
    for (i=0; i<mynode; i++)
        CmiSyncNodeSendFn(i, size, msg);
#endif
}

void CmiFreeNodeBroadcastFn(int size, char *msg) {
    CmiSyncNodeBroadcastFn(size, msg);
    CmiFree(msg);
}
#endif

#if USE_COMMON_ASYNC_BCAST
CmiCommHandle CmiAsyncNodeBroadcastFn(int size, char *msg) {
    CmiSyncNodeBroadcastFn(size, msg);
    return 0;
}
#endif

void CmiSyncNodeBroadcastAllFn(int size, char *msg) {
    CmiSyncNodeSendFn(CmiMyNode(), size, msg);
    CmiSyncNodeBroadcastFn(size, msg);
}

CmiCommHandle CmiAsyncNodeBroadcastAllFn(int size, char *msg) {
    CmiSendNodeSelf(CopyMsg(msg, size));
    return CmiAsyncNodeBroadcastFn(size, msg);
}

void CmiFreeNodeBroadcastAllFn(int size, char *msg) {
    CmiSyncNodeBroadcastFn(size, msg);
    /* Since it's a node-level msg, the msg could be executed on any other
     * procs on the same node. This means, the push of this msg to the
     * node-level queue could be immediately followed a pop of this msg on
     * other cores on the same node even when this msg has not been sent to
     * other nodes. This is the reason CmiSendNodeSelf must be called after
     * CmiSyncNodeBroadcastFn -Chao Mei
     */
    CmiSendNodeSelf(msg);
}
#endif
/* ##### End of Functions Related with Message Sending OPs ##### */

/* ##### Beginning of Functions Related with Machine Startup ##### */
void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret) {
    int tmp;
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

    /* _Cmi_mynodesize has to be obtained before MachineSpecificInit
     * because it may be used inside MachineSpecificInit
     */
    /* argv could be changed inside MachineSpecificInit */
    /* Inside this function, the number of nodes and my node id are obtained */
    MachineSpecificInit(argc, argv, &_Cmi_numnodes, &_Cmi_mynode);

    _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
    Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
    Cmi_argvcopy = CmiCopyArgs(argv);
    Cmi_argv = argv;
    Cmi_startfn = fn;
    Cmi_usrsched = usched;

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

    CmiStartThreads(argv);
    ConverseRunPE(initret);
}

static void ConverseRunPE(int everReturn) {
    CmiState cs;
    char** CmiMyArgv;

    MachineSpecificPreCommonInit(everReturn);

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

    MachineSpecificPostCommonInit(everReturn);

    /* Converse initialization finishes, immediate messages can be processed.
       node barrier previously should take care of the node synchronization */
    _immediateReady = 1;

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
/* ##### End of Functions Related with Machine Startup ##### */

/* ##### Beginning of Functions Related with Machine Running ##### */
static INLINE_KEYWORD void AdvanceCommunication() {
    int doProcessBcast = 1;

    MachineSpecificAdvanceCommunication();

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

static void CommunicationServer(int sleepTime) {
#if CMK_SMP
    AdvanceCommunication();

    if (commThdExit == CmiMyNodeSize()) {
        MACHSTATE(2, "CommunicationServer exiting {");
        MachineSpecificDrainResources();
        MACHSTATE(2, "} CommunicationServer EXIT");

        ConverseCommonExit();

        MachineSpecificExit();
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
#if !CMK_SMP
    MachineSpecificDrainResources();
#endif

    ConverseCommonExit();

#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyPe() == 0) CmiPrintf("End of program\n");
#endif

#if !CMK_SMP
    MachineSpecificExit();
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


/* ##### Beginning of Functions Providing Incoming Network Messages ##### */
void *CmiGetNonLocal(void) {
    CmiState cs = CmiGetState();
    void *msg = NULL;

    if (CmiNumPes() == 1) return NULL;

    MACHSTATE2(3, "[%p] CmiGetNonLocal begin %d{", cs, CmiMyPe());
    CmiIdleLock_checkMessage(&cs->idle);
    /* ?????although it seems that lock is not needed, I found it crashes very often
       on mpi-smp without lock */
#if !CMK_SMP
    AdvanceCommunication();
#endif

    msg = PCQueuePop(cs->recv);

#if !CMK_SMP
    MachineSpecificPostNonLocal();
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

/* ##### Beginning of Functions Related with Idle-state ##### */
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

#if !CMK_SMP
    AdvanceCommunication();
#endif

    MACHSTATE1(2,"still idle (%d) end {",CmiMyPe())
}

/* usually called in non-smp mode */
void CmiNotifyIdle(void) {
    AdvanceCommunication();
    CmiYield();
}
/* ##### End of Functions Related with Idle-state ##### */

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
/* ===== End of Common Function Definitions ===== */
