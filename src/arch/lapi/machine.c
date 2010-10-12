/*****************************************************************************
LAPI version of machine layer
Based on the template machine layer

Developed by
Filippo Gioachin   03/23/05
Chao Mei 01/28/2010
************************************************************************/

#include <lapi.h>

#include "converse.h"

#include <assert.h>
#include <errno.h>

/*Support for ++debug: */
#include <unistd.h> /*For getpid()*/
#include <stdlib.h> /*For sleep()*/

#include "machine.h"

/* Read the following carefully before building the machine layer on LAPI */
/* =========BEGIN OF EXPLANATION OF MACRO USAGE=============*/
/**
 * 1. non-SMP mode: 
 *   CMK_SMP = 0; 
 *   CMK_PCQUEUE_LOCK = 1; (could be removed if memory fence and atomic ops are used) 
 *  
 *   (The following two could be disabled to reduce the overhead of machine layer) 
 *     ENSURE_MSG_PAIRORDER = 0|1; 
 *     ENABLE_CONVERSE_QD = 0|1; 
 *  
 *   (If ENSURE_MSG_PAIRORDER is 1, then setting DECOUPLE_BCAST_PROCESS to 1
 *   will make the msg seqno increase at step of 1 w/o data race;
 *     DECOUPLE_BCAST_PROCESS = 0|1;
 * =========================================================== 
 * 2. SMP mode without comm thd:
 *    CMK_SMP = 1;
 *    CMK_PCQUEUE_LOCK = 1;
 *    CMK_SMP_NO_COMMTHD = 1;
 *  
 *    ENSURE_MSG_PAIRORDER and ENABLE_CONVERSE_QD have same options as in non-SMP mode;
 *  
 *    DECOUPLE_BCAST_PROCESS has same options as in non-SMP mode;
 * =========================================================== 
 *  3. SMP mode with comm thd:
 *     CMK_SMP = 1;
 *     CMK_PCQUEUE_LOCK = 1;
 *     CMK_SMP_NO_COMMTHD = 0;
 *  
 *     ENSURE_MSG_PAIRORDER and ENABLE_CONVERSE_QD have same options as in non-SMP mode;
 *  
 *     (The following must be set with 1 as bcast msg is dealt with in comm thd!)
 *     DECOUPLE_BCAST_PROCESS = 1;
 *  ===========================================================
 *  
 *  Assumptions we made in different mode:
 *  1. non-SMP:
 *     a) imm msgs are processed when the proc is idle, and periodically. They should
 *        never be processed in the LAPI thread;
 *     b) forwarding msgs could be done on the proc or in the internal LAPI
 *        completion handler threads;
 *  2. SMP w/o comm thd:
 *     a) same with non-SMP a)
 *     b) forwarding bcast msgs could be done on proc whose rank=0;
 *        (enable DECOUPLE_BCAST_PROCESS)  or in internal LAPI completion
 *        handler threads;
 *     c) the destination rank of proc-level bcast msg is always 0;
 *  3. SMP w/ comm thd:
 *     a) imm msgs are processed in comm thd;
 *     b) forwarding bcast msgs is done in comm thd;
 *     c) same with 2 c)
 *  
 */
/* =========END OF EXPLANATION OF MACRO USAGE=============*/

#if CMK_SMP
#define CMK_PCQUEUE_LOCK 1
#else
/** 
 *  In non-smp case: the LAPI completion handler thread will
 *  also access the proc's recv queue (a PCQueue), so the queue
 *  needs to be protected. The number of producers equals the
 *  #completion handler threads, while there's only one consumer
 *  for the queue. Right now, the #completion handler threads is
 *  set to 1, so the atomic operation for PCQueue should be
 *  achieved via memory fence. --Chao Mei
 */

/* Redefine CmiNodeLocks only for PCQueue data structure */
#define CmiNodeLock CmiNodeLock_nonsmp
#undef CmiCreateLock
#undef CmiLock
#undef CmiUnlock
#undef CmiTryLock
#undef CmiDestroyLock
typedef pthread_mutex_t *CmiNodeLock_nonsmp;
CmiNodeLock CmiCreateLock(){
  CmiNodeLock lk = (CmiNodeLock)malloc(sizeof(pthread_mutex_t));  
  pthread_mutex_init(lk,(pthread_mutexattr_t *)0);
  return lk;
}
#define CmiLock(lock) (pthread_mutex_lock(lock))
#define CmiUnlock(lock) (pthread_mutex_unlock(lock))
#define CmiTryLock(lock) (pthread_mutex_trylock(lock))
void CmiDestroyLock(CmiNodeLock lock){
    pthread_mutex_destroy(lock);
    free(lock);
}
#define CMK_PCQUEUE_LOCK 1
#endif
#include "pcqueue.h"

/** 
 *  The converse qd is rarely used in current charm apps, so the
 *  counter for converse qd could be disabled for max
 *  performance. --Chao Mei
 */
#define ENABLE_CONVERSE_QD 1

#if CMK_SMP
/** 
 *  The macro defines whether to have a comm thd to offload some
 *  work such as processing immdiate messages, forwarding
 *  broadcast messages etc. This macro should be defined before
 *  including "machine-smp.c".
 *  --Chao Mei
 */
#define CMK_SMP_NO_COMMTHD 0
#if CMK_SMP_NO_COMMTHD
int Cmi_commthread = 0;
#else
int Cmi_commthread = 1;
#endif

#endif

/** 
 *  Enable this macro will offload the broadcast relay from the
 *  internal completion handler thread. This will make the msg
 *  seqno free of data-race. In SMP mode with comm thread where
 *  comm thread will forward bcast msgs, this macro should be
 *  enabled.
 */
#define DECOUPLE_BCAST_PROCESS 1

/** 
 * #####REGARDING IN-ORDER MESSAGE DELIVERY BETWEEN A PAIR OF 
 * PROCESSORS#####: 
 *  
 * Since the lapi doesn't guarantee the order of msg delivery 
 * (via network) between a pair of processors, we need to ensure 
 * this order via msg seq no and a window sliding based msg 
 * receiving scheme. So two extra fields are added to the basic 
 * msg header: srcPe and seqno. For node messages, we process it 
 * like a msg to be delivered to the first proc (rank=0) of that 
 * node. 
 *  
 * BTW: The in-order delivery between two processors in the same
 * node is guaranteed in the SMP mode as the msg transfer 
 * doesn't go through LAPI. 
 *  
 * The msg transferred through LAPI (via network) is always 
 * delivered to the first proc (whose rank is 0) on that node! 
 *  
 * --Chao Mei 
 */
#define ENSURE_MSG_PAIRORDER 1

#if ENSURE_MSG_PAIRORDER

#define MAX_MSG_SEQNO 65535
/* MAX_WINDOW_SIZE should be smaller than MAX_MSG_SEQNO, and MAX(unsigned char) */
#define MAX_WINDOW_SIZE 128
#define INIT_WINDOW_SIZE 8

/* The lock to ensure the completion handler (PumpMsgsComplete) is thread-safe */
CmiNodeLock cmplHdlrThdLock = NULL;

/** 
 *  expectedMsgSeqNo is an int array of size "#procs". It tracks
 *  the expected seqno recved from other procs to this proc.
 *  
 *  nextMsgSeqNo is an int array of size "#procs". It tracks
 *  the next seqno of the msg to be sent from this proc to other
 *  procs.
 *  
 *  oooMsgBuffer is an array of sizeof(void **)*(#procs), each
 * element (created on demand) points to a window (array) size 
 * of CUR_WINDOW_SIZE which buffers the out-of-order incoming 
 * messages (a (void *) array) 
 *  
 * oooMaxOffset indicates the maximum offset of the ooo msg 
 * ahead of the expected msg. The offset begins with 1, i.e., 
 * (offset-1) is the index of the ooo msg in the window 
 * (oooMsgBuffer) 
 *  
 *  --Chao Mei
 */

typedef struct MsgOrderInfoStruct{
    /* vars used on sender side */
    int *nextMsgSeqNo;

    /* vars used on recv side */
    int *expectedMsgSeqNo;        
    void ***oooMsgBuffer;
    unsigned char *oooMaxOffset;
    unsigned char *CUR_WINDOW_SIZE;
}MsgOrderInfo;

/** 
 *  broadcast msgs and p2p msgs uses different tracks of seq no
 *  as is the case with "net" layers.
 */
CpvDeclare(MsgOrderInfo, p2pMsgSeqInfo);
CpvDeclare(MsgOrderInfo, bcastMsgSeqInfo);


#endif

/*
    To reduce the buffer used in broadcast and distribute the load from
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of
  spanning tree broadcast algorithm.
    This will use the fourth short in message as an indicator of spanning tree
  root.
*/

#undef CMK_BROADCAST_SPANNING_TREE
#define CMK_BROADCAST_SPANNING_TREE 0

/*#define BROADCAST_SPANNING_FACTOR        CMK_SPANTREE_MAXSPAN*/
#define BROADCAST_SPANNING_FACTOR   2

#undef CMK_BROADCAST_HYPERCUBE
#define CMK_BROADCAST_HYPERCUBE     1

/**
 * The broadcast root of a msg. 
 * 0: indicate a non-broadcast msg 
 *  
 * >0: indicate a proc-level broadcast msg ("root-1" indicates 
 * the proc that starts a broadcast)
 *  
 * <0: indicate a node-level broadcast msg ("-root-1" indicates 
 * the node that starts a broadcast) 
 *  
 * On BG/P, we have a separate broadcast queue on each 
 * processor. This will allow us to use the separate thread 
 * (comm thread) to offload the broadcast operation from worker 
 * threads --Chao Mei 
 */
#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank
/* The actual msg size including the msg header */
#define CMI_MSG_SIZE(msg)                ((CmiMsgHeaderBasic *)msg)->size

#define CMI_MSG_SRCPE(msg)               ((CmiMsgHeaderBasic *)msg)->srcpe
#define CMI_MSG_SEQNO(msg)               ((CmiMsgHeaderBasic *)msg)->seqno

CpvDeclare(unsigned , networkProgressCount);
int networkProgressPeriod = 1000;

static int lapiDebugMode=0;
CsvDeclare(int, lapiInterruptMode);

static void ConverseRunPE(int everReturn);
static void PerrorExit(const char *msg);

static int Cmi_nodestart;   /* First processor in this node - stupid need due to machine-smp.h that uses it!!  */

static volatile int commThdExit = 0;

#include "machine-smp.c"


/* Variables describing the processor ID */

/* non-smp mode */
#if CMK_SHARED_VARS_UNAVAILABLE
/* Should the non-smp also needs the concept of node which equals to processor -Chao Mei */
int _Cmi_mype;
int _Cmi_numpes;
int _Cmi_myrank;

void CmiMemLock() {}
void CmiMemUnlock() {}

static struct CmiStateStruct Cmi_state;

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

void CmiYield() {
    sleep(0);
}

#elif CMK_SHARED_VARS_POSIX_THREADS_SMP

int _Cmi_numpes;
int _Cmi_mynodesize;
int _Cmi_mynode;
int _Cmi_numnodes;

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
    return (pe / _Cmi_mynodesize);
}
int CmiRankOf(int pe)      {
    return (pe % _Cmi_mynodesize);
}
#endif

CpvDeclare(void*, CmiLocalQueue);

#if DECOUPLE_BCAST_PROCESS
CpvDeclare(PCQueue, procBcastQ);
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(PCQueue, nodeBcastQ);
#endif
#endif

#if CMK_NODE_QUEUE_AVAILABLE
#define DGRAM_NODEMESSAGE   (0x7EFB)
#define NODE_BROADCAST_OTHERS (-1)
#define NODE_BROADCAST_ALL    (-2)
#endif

/* The way to read this macro
 * "routine args" becomes a function call inside the parameters of check_lapi_err,
 * and it returns a int as returnCode;
 * "#routine" turns the "routine" as a string
 * __LINE__ is the line number in the source file
 * -Chao Mei
 */
#define check_lapi(routine,args) \
        check_lapi_err(routine args, #routine, __LINE__);

static void check_lapi_err(int returnCode,const char *routine,int line) {
    if (returnCode!=LAPI_SUCCESS) {
        char errMsg[LAPI_MAX_ERR_STRING];
        LAPI_Msg_string(returnCode,errMsg);
        fprintf(stderr,"Fatal LAPI error while executing %s at %s:%d\n"
                "  Description: %s\n", routine, __FILE__, line, errMsg);
        CmiAbort("Fatal LAPI error");
    }
}

static void lapi_err_hndlr(lapi_handle_t *hndl, int *error_code, 
                            lapi_err_t *err_type, int *task_ID, int *src){
    char errstr[LAPI_MAX_ERR_STRING];
    LAPI_Msg_string(*error_code, errstr);
    fprintf(stderr, "ERROR IN LAPI: %s for task %d at src %d\n", errstr, *task_ID, *src);
    LAPI_Term(*hndl);
    exit(1);
}

/**
 * The lapiContext stands for the lapi context for a single lapi 
 * task. And inside one lapi task, only one lapi context could 
 * be created via lapi_init. In SMP mode, this context is 
 * created by proc of rank 0, and then it is shared among all 
 * cores on a node (threads) --Chao Mei 
 * 
 */
static lapi_handle_t lapiContext;
static lapi_long_t lapiHeaderHandler = 1;

/**
 * Note on broadcast functions: 
 * The converse QD may be wrong if using spanning tree or 
 * hypercube schemes to send messages --Chao Mei 
 */
void SendMsgToPeers(int size, char *msg, int includeSelf);
#if ENSURE_MSG_PAIRORDER
void SendSpanningChildren(int size, char *msg, int srcPe, int *seqNoArr);
void SendHypercube(int size, char *msg, int srcPe, int *seqNoArr);
#else
void SendSpanningChildren(int size, char *msg);
void SendHypercube(int size, char *msg);
#endif

#if CMK_NODE_QUEUE_AVAILABLE
/** 
 *  The sending schemes of the following two functions are very
 *  similar to its corresponding proc-level functions
 */
void SendSpanningChildrenNode(int size, char *msg);
void SendHypercubeNode(int size, char *msg);
#endif

/**
 * There is a function "CkCopyMsg" in the charm++ level, which
 * considers more than mere memcpy as there could be var size msg
 * --Chao Mei
 */
char *CopyMsg(char *msg, int len) {
    char *copy = (char *)CmiAlloc(len);
    if (!copy)
        fprintf(stderr, "Out of memory\n");
    memcpy(copy, msg, len);
    return copy;
}
 
CsvDeclare(CmiNodeState, NodeState);

#if CMK_IMMEDIATE_MSG
#include "immediate.c"
#endif

/** 
 *  This function will be never called in the comm thd!
 *  It is only used in the non-SMP mode, or in SMP mode w/o comm
 *  thd! --Chao Mei
 */
static void AdvanceCommunication(){
#if !CMK_SMP || CMK_SMP_NO_COMMTHD

    if(!CsvAccess(lapiInterruptMode)) check_lapi(LAPI_Probe,(lapiContext));
#if CMK_IMMEDIATE_MSG
    /** 
     * Immediate msgs are handled in CmiPushNode and 
     * CmiSendNodeSelf, CmiPushPe and CmiSendSelf in the non-SMP 
     * case, but in SMP case, those four functions could be called 
     * in the completition handler where "CmiMyPe(), CmiMyRank()" 
     * will be wrong as the "CmiGetState()" will not return a right 
     * proc-specific CmiState!! So immediate messages are handled 
     * when proc is idle. This may cause a big delay for processing 
     * immdiate messages in SMP mode if there's not a dedicated 
     * communication thread. 
     *  
     * Even in non-SMP, inside imm msg handlers, array proxy 
     * messages could be sent. If handled in LAPI internal 
     * completion thread, it will cause data-racing problems for 
     * data structures that maintain charm array info. So it needs 
     * to be handled in the context of a proc. 
     *  
     * -Chao Mei 
     */
    MACHSTATE1(2, "[%p] Handling Immediate Message begin{",CmiGetState());
    CmiHandleImmediate();
    MACHSTATE1(2, "[%p] Handling Immediate Message end}",CmiGetState());
#endif

#endif
}

/* non-smp CmiStartThreads. -Chao Mei */
#if CMK_SHARED_VARS_UNAVAILABLE
/* To conform with the call made in SMP mode */
static void CmiStartThreads(char **argv) {
    CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
}
#endif

/* ===== Beginging of functions regarding ensure in-order msg delivery ===== */
#if ENSURE_MSG_PAIRORDER

/**
 * "setNextMsgSeqNo" actually sets the current seqno, the 
 * "getNextMsgSeqNo" will increment the seqno, i.e., 
 * "getNextMgSeqNo" returns the next seqno based on the previous 
 * seqno stored in the seqno array. 
 * --Chao Mei 
 */
static int getNextMsgSeqNo(int *seqNoArr, int destPe){
    int ret = seqNoArr[destPe];
    ret++;
    return ret;
}
static void setNextMsgSeqNo(int *seqNoArr, int destPe, int val){
    /* the seq no. may fast-forward to a new round (i.e., starting from 1 again!) */
    if(val>=MAX_MSG_SEQNO) val -= MAX_MSG_SEQNO;
    seqNoArr[destPe] = val;
}

#define getNextExpectedMsgSeqNo(seqNoArr,srcPe) getNextMsgSeqNo(seqNoArr, srcPe)
#define setNextExpectedMsgSeqNo(seqNoArr,srcPe,val) setNextMsgSeqNo(seqNoArr, srcPe, val)

#endif
/* ===== End of functions regarding ensure in-order msg delivery ===== */

/* ===========CmiPushPe and CmiPushNode============*/
/* Add a message to this processor's receive queue, pe is a rank */
void CmiPushPE(int pe,void *msg) {
    CmiState cs = CmiGetStateN(pe);
    MACHSTATE3(3,"Pushing message(%p) into rank %d's queue %p {",msg,pe,(cs->recv));

#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        MACHSTATE1(3, "[%p] Push Immediate Message begin{",CmiGetState());
        CMI_DEST_RANK(msg) = pe;
        CmiPushImmediateMsg(msg);
        MACHSTATE1(3, "[%p] Push Immediate Message end }",CmiGetState());
        return;
    }
#endif

    /* Note atomicity is guaranteed inside pcqueue data structure --Chao Mei */
    PCQueuePush(cs->recv,msg);

    CmiIdleLock_addMessage(&cs->idle);
    MACHSTATE1(3,"} Pushing message into rank %d's queue done",pe);
}

#if CMK_NODE_QUEUE_AVAILABLE
/*Add a message to this processor's receive queue */
/*Note: CmiPushNode is essentially same with CimSendNodeSelf */
static void CmiPushNode(void *msg) {    
    MACHSTATE1(3,"[%p] Pushing message into NodeRecv queue",CmiGetState());

#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        MACHSTATE1(3, "[%p] Push Immediate Message begin {",CmiGetState());
        CMI_DEST_RANK(msg) = 0;
        CmiPushImmediateMsg(msg);
        MACHSTATE1(3, "[%p] Push Immediate Message end }",CmiGetState());
        return;
    }
#endif
    /* CmiNodeRecvLock may not be needed  --Chao Mei*/
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv,msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);

    CmiState cs=CmiGetStateN(0);
    CmiIdleLock_addMessage(&cs->idle);

    MACHSTATE(3,"Pushing message into NodeRecv queue {");
}
#endif

/* ======Beginning of helper functions for processing an incoming (network) message ======*/
/* Process a proc-level broadcast message */
static void ProcessProcBroadcastMsg(char *msg){
    int nbytes = CMI_MSG_SIZE(msg);    
#if ENSURE_MSG_PAIRORDER
    MACHSTATE3(2,"[%p] the broadcast msg is from pe=%d with seq no=%d", CmiGetState(), CMI_MSG_SRCPE(msg), CMI_MSG_SEQNO(msg));
    #if CMK_BROADCAST_SPANNING_TREE    
    SendSpanningChildren(nbytes, msg, CmiNodeFirst(CmiMyNode()), CpvAccessOther(bcastMsgSeqInfo, 0).nextMsgSeqNo);
    #elif CMK_BROADCAST_HYPERCUBE    
    SendHypercube(nbytes, msg, CmiNodeFirst(CmiMyNode()), CpvAccessOther(bcastMsgSeqInfo, 0).nextMsgSeqNo);
    #endif
#else
    #if CMK_BROADCAST_SPANNING_TREE
    SendSpanningChildren(nbytes, msg);
    #elif CMK_BROADCAST_HYPERCUBE
    SendHypercube(nbytes, msg);
    #endif
#endif                     
#if CMK_SMP
    SendMsgToPeers(nbytes, msg, 1);                   
    CmiFree(msg);
#else
    /* nonsmp case */
    CmiPushPE(0, msg);
#endif
}

#if CMK_NODE_QUEUE_AVAILABLE
/* Process a node-level broadcast message */
static void ProcessNodeBroadcastMsg(char *msg){
    int nbytes = CMI_MSG_SIZE(msg);
#if CMK_BROADCAST_SPANNING_TREE
    SendSpanningChildrenNode(nbytes, msg);
#elif CMK_BROADCAST_HYPERCUBE
    SendHypercubeNode(nbytes, msg);
#endif            
    CmiPushNode(msg);
}
#endif

/* Pull msgs from two queues, this function should not be called from cmpl_hdlr thread */
static void ProcessBroadcastMsg(int pullRank){
    char *msg;
    do{
        msg = PCQueuePop(CpvAccessOther(procBcastQ, pullRank));
        if(msg) {
            MACHSTATE2(4, "[%p]: process a proc-level bcast msg %p begin{", CmiGetState(), msg);
            ProcessProcBroadcastMsg(msg);
            MACHSTATE2(4, "[%p]: process a proc-level bcast msg %p end}", CmiGetState(), msg);
        }else{ 
            break;
        }
    }while (1);
#if CMK_NODE_QUEUE_AVAILABLE
    do{
        msg = PCQueuePop(CsvAccess(nodeBcastQ));
        if(msg) {
            MACHSTATE2(4, "[%p]: process a node-level bcast msg %p begin{", CmiGetState(), msg);
            ProcessNodeBroadcastMsg(msg);
            MACHSTATE2(4, "[%p]: process a node-level bcast msg %p end}", CmiGetState(), msg);
        }else{ 
            break;
        }
    }while (1);    
#endif
}


#if ENSURE_MSG_PAIRORDER
/* return 1 if this msg is an out-of-order incoming message */

/**
 * Returns 1 if this "msg" is an out-of-order message, or 
 * this "msg" is a late message which triggers the process 
 * of all buffered ooo msgs. 
 * --Chao Mei 
 */
static int CheckMsgInOrder(char *msg, MsgOrderInfo *info){
    int srcpe, destrank; 
    int incomingSeqNo, expectedSeqNo;
    int curOffset, maxOffset;
    int i, curWinSize;
    void **destMsgBuffer = NULL;

    /* numMsg is the number of msgs to be processed in this buffer*/
    /* Reason to have this extra copy of msgs to be processed: Reduce the atomic granularity */
    void **toProcessMsgBuffer;
    int numMsgs = 0;    
 
    srcpe = CMI_MSG_SRCPE(msg);
    destrank = CMI_DEST_RANK(msg);  
    incomingSeqNo = CMI_MSG_SEQNO(msg);
    
    CmiLock(cmplHdlrThdLock);

    expectedSeqNo = getNextExpectedMsgSeqNo(info->expectedMsgSeqNo, srcpe);
    if(expectedSeqNo == incomingSeqNo){
        /* Two cases: has ooo msg buffered or not */
        maxOffset = (info->oooMaxOffset)[srcpe];
        if(maxOffset>0) {
            MACHSTATE1(4, "Processing all buffered ooo msgs (maxOffset=%d) including the just recved begin {", maxOffset);
            curWinSize = info->CUR_WINDOW_SIZE[srcpe];
            toProcessMsgBuffer = malloc((curWinSize+1)*sizeof(void *));
            /* process the msg just recved */
            toProcessMsgBuffer[numMsgs++] = msg;            
            /* process the buffered ooo msg until the first empty slot in the window */
            destMsgBuffer = (info->oooMsgBuffer)[srcpe];                       
            for(curOffset=0; curOffset<maxOffset; curOffset++) {
                char *curMsg = destMsgBuffer[curOffset];
                if(curMsg == NULL){
                    CmiAssert(curOffset!=(maxOffset-1));                                        
                    break;
                }
                toProcessMsgBuffer[numMsgs++] = curMsg;
                destMsgBuffer[curOffset] = NULL;
            }            
            /* Update expected seqno, maxOffset and slide the window */
            if(curOffset < maxOffset) {
                int i;
                /** 
                 * now, the seqno of the next to-be-recved msg should be 
                 * "expectedSeqNo+curOffset+1" as the seqno of the just 
                 * processed msg is "expectedSeqNo+curOffset. We need to slide 
                 * the msg buffer window from "curOffset+1" because the first 
                 * element of the buffer window should always points to the ooo 
                 * msg that's 1 in terms of seqno ahead of the next to-be-recved 
                 * msg. --Chao Mei 
                 */                
                
                /* moving [curOffset+1, maxOffset) to [0, maxOffset-curOffset-1) in the window */
                /* The following two loops could be combined --Chao Mei */
                for(i=0; i<maxOffset-curOffset-1; i++){
                    destMsgBuffer[i] = destMsgBuffer[curOffset+i+1];
                }
                for(i=maxOffset-curOffset-1; i<maxOffset; i++) {
                    destMsgBuffer[i] = NULL;
                }
                (info->oooMaxOffset)[srcpe] = maxOffset-curOffset-1;
                setNextExpectedMsgSeqNo(info->expectedMsgSeqNo, srcpe, expectedSeqNo+curOffset);
            }else{
                /* there's no remaining buffered ooo msgs */
                (info->oooMaxOffset)[srcpe] = 0;
                setNextExpectedMsgSeqNo(info->expectedMsgSeqNo, srcpe, expectedSeqNo+maxOffset);
            }

            CmiUnlock(cmplHdlrThdLock);
                        
            /* Process the msgs */
            for(i=0; i<numMsgs; i++){
                char *curMsg = toProcessMsgBuffer[i];                
                if(CMI_BROADCAST_ROOT(curMsg)>0) {
                                        
                #if DECOUPLE_BCAST_PROCESS
                    PCQueuePush(CpvAccessOther(procBcastQ, 0), curMsg);
                #else
                    ProcessProcBroadcastMsg(curMsg);
                #endif

                }else{
                    CmiPushPE(CMI_DEST_RANK(curMsg), curMsg);
                }
            }

            free(toProcessMsgBuffer);

            MACHSTATE1(4, "Processing all buffered ooo msgs (actually processed %d) end }", curOffset);
            /** 
             * Since we have processed all buffered ooo msgs including 
             * this just recved one, 1 should be returned so that this 
             * msg no longer needs processing 
             */
            return 1;
        }else{
            /* An expected msg recved without any ooo msg buffered */
            MACHSTATE1(4, "Receiving an expected msg with seqno=%d\n", incomingSeqNo);
            setNextExpectedMsgSeqNo(info->expectedMsgSeqNo, srcpe, expectedSeqNo);

            CmiUnlock(cmplHdlrThdLock);
            return 0;
        }        
    }

    MACHSTATE2(4, "Receiving an out-of-order msg with seqno=%d, but expect seqno=%d", incomingSeqNo, expectedSeqNo);
    curWinSize = info->CUR_WINDOW_SIZE[srcpe];
    if((info->oooMsgBuffer)[srcpe]==NULL) {        
        (info->oooMsgBuffer)[srcpe] = malloc(curWinSize*sizeof(void *));
        memset((info->oooMsgBuffer)[srcpe], 0, curWinSize*sizeof(void *));
    }
    destMsgBuffer = (info->oooMsgBuffer)[srcpe];
    curOffset = incomingSeqNo - expectedSeqNo;
    maxOffset = (info->oooMaxOffset)[srcpe];
    if(curOffset<0) {
        /* It's possible that the seqNo starts with another round (exceeding MAX_MSG_SEQNO) with 1 */
        curOffset += MAX_MSG_SEQNO;
    }
    if(curOffset > curWinSize) {
        int newWinSize;
        if(curOffset > MAX_WINDOW_SIZE) {
            CmiAbort("Exceeding the MAX_WINDOW_SIZE!\n");
        }
        newWinSize = ((curOffset/curWinSize)+1)*curWinSize;
        /*CmiPrintf("[%d]: WARNING: INCREASING WINDOW SIZE FROM %d TO %d\n", CmiMyPe(), curWinSize, newWinSize);*/
        (info->oooMsgBuffer)[srcpe] = malloc(newWinSize*sizeof(void *));
        memset((info->oooMsgBuffer)[srcpe], 0, newWinSize*sizeof(void *));
        memcpy((info->oooMsgBuffer)[srcpe], destMsgBuffer, curWinSize*sizeof(void *));
        info->CUR_WINDOW_SIZE[srcpe] = newWinSize;
        free(destMsgBuffer);
        destMsgBuffer = (info->oooMsgBuffer)[srcpe];
    }    
    CmiAssert(destMsgBuffer[curOffset-1] == NULL);
    destMsgBuffer[curOffset-1] = msg;
    if(curOffset > maxOffset) (info->oooMaxOffset)[srcpe] = curOffset;

    CmiUnlock(cmplHdlrThdLock);
    return 1;
}

#endif
/* ======End of helper functions for processing an incoming (network) message ======*/

/* ======Begining of lapi callbacks such as the completion handler on the sender and recver side ======*/
/** 
  * lapi completion handler on the recv side. It's responsible to push messages 
  * to the destination proc or relay broadcast messages. --Chao Mei 
  *  
  * Note: The completion handler could be executed on any cores within a node ??? 
  * So in SMP mode when there's a comm thread, the completion handler should be carefully 
  * dealt with. 
  *  
  * Given lapi also provides an internal lapi thread to deal with network progress which 
  * will call this function (???), we should be careful with the following situations: 
  * 1) non SMP mode, with interrupt (lapi internal completion thread) 
  * 2) non SMP mode, with polling (machine layer is responsible for network progress) 
  * 3) SMP mode, no comm thread, with polling 
  * 4) SMP mode, no comm thread, with interrupt 
  * 5) SMP mode, with comm thread, with polling (not yet implemented, comm server is empty right now)
  * 6) SMP mode, with comm thread, with interrupt?? 
  *  
  * Currently, SMP mode without comm thread is undergoing implementation. 
  *  
  * This function is executed by LAPI internal threads. It seems that the number of internal 
  * completion handler threads could vary during the program. LAPI adaptively creates more 
  * threads if there are more outstanding messages!!!! This means pcqueue needs protection 
  * even in the nonsmp case!!!!
  *  
  * --Chao Mei 
  */
static void PumpMsgsComplete(lapi_handle_t *myLapiContext, void *am_info) {
    int i;
    char *msg = am_info;
    int broot, destrank;

    MACHSTATE3(2,"[%p] PumpMsgsComplete with msg %p (isImm=%d) begin {",CmiGetState(), msg, CmiIsImmediate(msg));    
#if ENSURE_MSG_PAIRORDER
    MACHSTATE3(2,"msg %p info: srcpe=%d, seqno=%d", msg, CMI_MSG_SRCPE(msg), CMI_MSG_SEQNO(msg));
#endif
    /**
     * First, we check if the msg is a broadcast msg via spanning 
     * tree. If it is, it needs to call SendSpanningChildren to 
     * relay the broadcast, and then send the msg to every cores on 
     * this node.
     *  
     * After the first check, we deal with normal messages. 
     * --Chao Mei
     */
/* It's the right place to relay the broadcast message */
    /**
     * 1. For in-order delivery, because this is the handler for 
     * receiving a message, and we assume the cross-network msgs are 
     * always delivered to the first proc (rank 0) of this node, we 
     * select the srcpe of the bcast msgs and the next msg seq no 
     * correspondingly. 
     *  
     * 2. TODO: checking the in-order delivery of p2p msgs!! 
     *  
     * --Chao Mei 
     */
#if ENSURE_MSG_PAIRORDER 
    broot = CMI_BROADCAST_ROOT(msg);
    destrank = CMI_DEST_RANK(msg);
    /* Only check proc-level msgs */
    if (broot>=0
#if CMK_NODE_QUEUE_AVAILABLE
        && destrank != DGRAM_NODEMESSAGE
#endif
    )
    {
        MsgOrderInfo *info;        
        if(broot>0){
            info = &CpvAccessOther(bcastMsgSeqInfo, destrank);
            MACHSTATE1(2, "Check msg in-order for bcast msg %p", msg);
        } else {
            info = &CpvAccessOther(p2pMsgSeqInfo, destrank);
            MACHSTATE1(2, "Check msg in-order for p2p msg %p", msg);
        }
        if(CheckMsgInOrder(msg,info)) {
            MACHSTATE(2,"} PumpMsgsComplete end ");
            return;
        }
    }
#endif    

#if CMK_BROADCAST_SPANNING_TREE || CMK_BROADCAST_HYPERCUBE
    if (CMI_BROADCAST_ROOT(msg)>0) {
        MACHSTATE2(2,"[%p] Recved a proc-level broadcast msg %p",CmiGetState(), msg);     
                
    #if DECOUPLE_BCAST_PROCESS
        PCQueuePush(CpvAccessOther(procBcastQ, 0), msg);
    #else
        ProcessProcBroadcastMsg(msg);
    #endif

        MACHSTATE(2,"} PumpMsgsComplete end ");
        return;
    }

#if CMK_NODE_QUEUE_AVAILABLE
    if(CMI_BROADCAST_ROOT(msg) < 0) {
        MACHSTATE1(2,"[%p] Recved a node-level broadcast msg",CmiGetState());  
        
    #if DECOUPLE_BCAST_PROCESS
        PCQueuePush(CsvAccess(nodeBcastQ), msg);
    #else
        ProcessNodeBroadcastMsg(msg);
    #endif

        MACHSTATE(2,"} PumpMsgsComplete end ");
        return;
    }
#endif

#endif

#if CMK_NODE_QUEUE_AVAILABLE
    if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
        CmiPushNode(msg);
    else{
        MACHSTATE3(2,"[%p] Recv a p2p msg from pe=%d with seq no=%d", CmiGetState(), CMI_MSG_SRCPE(msg), CMI_MSG_SEQNO(msg));
        CmiPushPE(CMI_DEST_RANK(msg), msg);
    }
#else
    CmiPushPE(CMI_DEST_RANK(msg), msg);
#endif

    MACHSTATE(2,"} PumpMsgsComplete end ");
    return;
}

/** lapi header handler: executed on the recv side, when the
 *  first packet of the recving msg arrives, it is called to
 *  prepare the memory buffer in the user space for recving the
 *  data --Chao Mei
 */
static void* PumpMsgsBegin(lapi_handle_t *myLapiContext,
                           void *hdr, uint *uhdr_len,
                           lapi_return_info_t *msg_info,
                           compl_hndlr_t **comp_h, void **comp_am_info) {
    void *msg_buf;
    MACHSTATE1(2,"[%p] PumpMsgsBegin begin {",CmiGetState());
    /* prepare the space for receiving the data, set the completion handler to
       be executed inline */
    msg_buf = (void *)CmiAlloc(msg_info->msg_len);

    msg_info->ret_flags = LAPI_SEND_REPLY;
    *comp_h = PumpMsgsComplete;
    *comp_am_info = msg_buf;
    MACHSTATE(2,"} PumpMsgsBegin end");
    return msg_buf;

}

/* The following two are lapi sender handlers */
static void ReleaseMsg(lapi_handle_t *myLapiContext, void *msg, lapi_sh_info_t *info) {
    MACHSTATE2(2,"[%p] ReleaseMsg begin %p {",CmiGetState(),msg);
    check_lapi_err(info->reason, "ReleaseMsg", __LINE__);
    CmiFree(msg);
    MACHSTATE(2,"} ReleaseMsg end");
}

static void DeliveredMsg(lapi_handle_t *myLapiContext, void *msg, lapi_sh_info_t *info) {
    MACHSTATE1(2,"[%p] DeliveredMsg begin {",CmiGetState());
    check_lapi_err(info->reason, "DeliveredMsg", __LINE__);
    *((int *)msg) = *((int *)msg) - 1;
    MACHSTATE(2,"} DeliveredMsg end");
}
/* ======End of lapi callbacks such as the completion handler on the sender and recver side ======*/


#if CMK_NODE_QUEUE_AVAILABLE
char *CmiGetNonLocalNodeQ(void) {
    CmiState cs = CmiGetState();
    char *result = 0;
    CmiIdleLock_checkMessage(&cs->idle);
    if (!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
        MACHSTATE2(3,"[%p] CmiGetNonLocalNodeQ begin %d {",CmiGetState(),CmiMyPe());
        CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
        result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
        CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
        MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
    }
    return result;
}
#endif

void *CmiGetNonLocal(void) {    
    CmiState cs = CmiGetState();
    MACHSTATE2(3, "[%p] CmiGetNonLocal begin %d{", cs, CmiMyPe());
    CmiIdleLock_checkMessage(&cs->idle); 
       
#if DECOUPLE_BCAST_PROCESS
#if !CMK_SMP
    ProcessBroadcastMsg(0);
#elif CMK_SMP_NO_COMMTHD
    if(CmiMyRank()==0) ProcessBroadcastMsg(0);
#endif
#endif
           
    void *msg = PCQueuePop(cs->recv);
    MACHSTATE3(3,"[%p] CmiGetNonLocal from queue %p with msg %p end }",CmiGetState(),(cs->recv), msg);    
    return msg;

}

/**
 * TODO: What will be the effects if calling LAPI_Probe in the 
 * interrupt mode??? --Chao Mei 
 */

/* user call to handle immediate message, since there is no ServerThread polling
   messages (lapi does all the polling) every thread is authorized to process
   immediate messages. If we are not in lapiInterruptMode check for progress.
*/
void CmiMachineProgressImpl(){
    if (!CsvAccess(lapiInterruptMode)) check_lapi(LAPI_Probe,(lapiContext));

#if CMK_IMMEDIATE_MSG
    MACHSTATE1(2, "[%p] Handling Immediate Message begin {",CmiGetState());
    CmiHandleImmediate();
    MACHSTATE1(2, "[%p] Handling Immediate Message end }",CmiGetState());
#endif

#if CMK_SMP && !CMK_SMP_NO_COMMTHD && DECOUPLE_BCAST_PROCESS
    if(CmiMyRank()==CmiMyNodeSize()) ProcessBroadcastMsg(0);
#endif
}

/*TODO: does lapi provide any Barrrier related functions as DCMF provides??? --Chao Mei */
/* Barrier needs to be implemented!!! -Chao Mei */
/* These two barriers are only needed by CmiTimerInit to synchronize all the
   threads. They do not need to provide a general barrier. */
int CmiBarrier() {
    return 0;
}
int CmiBarrierZero() {
    return 0;
}

/********************* MESSAGE SEND FUNCTIONS ******************/

/** 
 * "deliverable": used to know if the message can be encoded 
 * into the destPE queue withtout duplication (for usage of 
 * SMP). If it has already been duplicated (and therefore is 
 * deliverable), we do not want to copy it again, while if it 
 * has been copied we must do it before enqueuing it. 
 *  
 * The general send function for all Cmi send functions.
 */
void lapiSendFn(int destPE, int size, char *msg, scompl_hndlr_t *shdlr, void *sinfo, int deliverable) {
    /* CmiState cs = CmiGetState(); */
    CmiUInt2  rank, node;
    lapi_xfer_t xfer_cmd;

    MACHSTATE3(2,"lapiSendFn to destPE=%d with msg %p (isImm=%d) begin {",destPE,msg, CmiIsImmediate(msg));
    MACHSTATE3(2, "inside lapiSendFn 1: size=%d, sinfo=%p, deliverable=%d", size, sinfo, deliverable);
    
#if ENSURE_MSG_PAIRORDER
    MACHSTATE3(2, "inside lapiSendFn 2: msg src->dest (%d->%d), seqno=%d", CMI_MSG_SRCPE(msg), destPE, CMI_MSG_SEQNO(msg));
#endif

    node = CmiNodeOf(destPE);
    /** 
     *  The rank of the msg should be set before calling
     *  lapiSendFn!!
     *  The rank could be DGRAM_NODEMESSAGE which indicates
     *  a node-level message.
     */
#if CMK_SMP
    /*CMI_DEST_RANK(msg) = rank;*/
    if (node == CmiMyNode())  {
        rank = CmiRankOf(destPE);
        MACHSTATE2(2,"[%p] inside lapiSendFn for intra-node message (%p)",CmiGetState(), msg);
        if (deliverable) {
            CmiPushPE(rank, msg);
            /* the acknowledge of delivery must not be called */
        } else {
            CmiPushPE(rank, CopyMsg(msg, size));
            /* acknowledge that the message has been delivered */
            lapi_sh_info_t lapiInfo;
            lapiInfo.src = node;
            lapiInfo.reason = LAPI_SUCCESS;
            (*shdlr)(&lapiContext, sinfo, &lapiInfo);
        }
        return;
    }
#endif
    
    MACHSTATE2(2, "Ready to call LAPI_Xfer with destPe=%d, destRank=%d",destPE,CMI_DEST_RANK(msg));

    xfer_cmd.Am.Xfer_type = LAPI_AM_XFER;
    xfer_cmd.Am.flags     = 0;
    xfer_cmd.Am.tgt       = node;
    xfer_cmd.Am.hdr_hdl   = lapiHeaderHandler;
    xfer_cmd.Am.uhdr_len  = 0;
    xfer_cmd.Am.uhdr      = NULL;
    xfer_cmd.Am.udata     = msg;
    xfer_cmd.Am.udata_len = size;
    xfer_cmd.Am.shdlr     = shdlr;
    xfer_cmd.Am.sinfo     = sinfo;
    xfer_cmd.Am.tgt_cntr  = NULL;
    xfer_cmd.Am.org_cntr  = NULL;
    xfer_cmd.Am.cmpl_cntr = NULL;

    check_lapi(LAPI_Xfer,(lapiContext, &xfer_cmd));

    MACHSTATE(2,"} lapiSendFn end");
}

static void CmiSendSelf(char *msg) {
    MACHSTATE1(3,"[%p] Sending itself a message {",CmiGetState());

#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        MACHSTATE1(3, "[%p] Push Immediate Message begin {",CmiGetState());        
        CmiPushImmediateMsg(msg);
        MACHSTATE1(3, "[%p] Push Immediate Message end }",CmiGetState());        
        return;
    }
#endif
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CmiGetState()->localqueue,msg);

    MACHSTATE(3,"} Sending itself a message");
}

void CmiSyncSendFn(int destPE, int size, char *msg) {
    CmiState cs = CmiGetState();
    char *dupmsg = CopyMsg(msg, size);

    MACHSTATE1(3,"[%p] Sending sync message begin {",CmiGetState());
    CMI_BROADCAST_ROOT(dupmsg) = 0;
    CMI_DEST_RANK(dupmsg) = CmiRankOf(destPE);

    if (cs->pe==destPE) {
        CmiSendSelf(dupmsg);
    } else {
    #if ENSURE_MSG_PAIRORDER
        CMI_MSG_SRCPE(dupmsg) = CmiMyPe();        
        CMI_MSG_SEQNO(dupmsg) = getNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE);
        setNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE, CMI_MSG_SEQNO(dupmsg));
    #endif

    #if ENABLE_CONVERSE_QD
        CQdCreate(CpvAccess(cQdState), 1);
    #endif
        lapiSendFn(destPE, size, dupmsg, ReleaseMsg, dupmsg, 1);
    }
    MACHSTATE(3,"} Sending sync message end");
}

int CmiAsyncMsgSent(CmiCommHandle handle) {
    return (*((int *)handle) == 0)?1:0;
}

void CmiReleaseCommHandle(CmiCommHandle handle) {
#ifndef CMK_OPTIMIZE
    if (*((int *)handle) != 0) CmiAbort("Released a CmiCommHandle not free!");
#endif
    free(handle);
}

/* the CmiCommHandle returned is a pointer to the location of an int. When it is
   set to 1 the message is available. */
CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg) {
    MACHSTATE1(3,"[%p] Sending async message begin {",CmiGetState());
    void *handle;
    CmiState cs = CmiGetState();
    CMI_BROADCAST_ROOT(msg) = 0;
    CMI_DEST_RANK(msg) = CmiRankOf(destPE);

    /* if we are the destination, send ourself a copy of the message */
    if (cs->pe==destPE) {
        CmiSendSelf(CopyMsg(msg, size));
        MACHSTATE(3,"} Sending async message end");
        return 0;
    }

    handle = malloc(sizeof(int));
    *((int *)handle) = 1;

#if ENSURE_MSG_PAIRORDER
    CMI_MSG_SRCPE(msg) = CmiMyPe();    
    CMI_MSG_SEQNO(msg) = getNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE);
    setNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE, CMI_MSG_SEQNO(msg));
#endif

#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), 1);
#endif
    lapiSendFn(destPE, size, msg, DeliveredMsg, handle, 0);
    /* the message may have been duplicated and already delivered if we are in SMP
       mode and the destination is on the same node, but there is no optimized
       check for that. */
    MACHSTATE(3,"} Sending async message end");
    return handle;
}

void CmiFreeSendFn(int destPE, int size, char *msg) {
    MACHSTATE1(3,"[%p] Sending sync free message begin {",CmiGetState());
    CmiState cs = CmiGetState();
    CMI_BROADCAST_ROOT(msg) = 0;
    CMI_DEST_RANK(msg) = CmiRankOf(destPE);

    if (cs->pe==destPE) {
        CmiSendSelf(msg);
    } else {
    #if ENSURE_MSG_PAIRORDER
        CMI_MSG_SRCPE(msg) = CmiMyPe();        
        CMI_MSG_SEQNO(msg) = getNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE);
        setNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE, CMI_MSG_SEQNO(msg));
    #endif
    #if ENABLE_CONVERSE_QD
        CQdCreate(CpvAccess(cQdState), 1);
    #endif        
        lapiSendFn(destPE, size, msg, ReleaseMsg, msg, 1);
        /*CmiAsyncSendFn(destPE, size, msg);*/
    }
    MACHSTATE(3,"} Sending sync free message end");
}

/* ===========Node level p2p send functions============= */
#if CMK_NODE_QUEUE_AVAILABLE
static void CmiSendNodeSelf(char *msg) {
    CmiState cs;
    MACHSTATE1(3,"[%p] Sending itself a node message {",CmiGetState());

#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
        MACHSTATE1(3, "[%p] Push Immediate Message {",CmiGetState());
        CMI_DEST_RANK(msg) = 0;
        CmiPushImmediateMsg(msg);
        MACHSTATE(3, "} Push Immediate Message end");
        return;
    }
#endif
    CQdCreate(CpvAccess(cQdState), 1);
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv, msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);

    cs=CmiGetStateN(0);
    CmiIdleLock_addMessage(&cs->idle);

    MACHSTATE(3,"} Sending itself a node message");
}

/*TODO: not sure whether the in-order delivery affects for node messages?? --Chao Mei */
void CmiSyncNodeSendFn(int destNode, int size, char *msg) {
    char *dupmsg = CopyMsg(msg, size);

    MACHSTATE1(3,"[%p] Sending sync node message begin {",CmiGetState());
    CMI_BROADCAST_ROOT(dupmsg) = 0;
    CMI_DEST_RANK(dupmsg) = DGRAM_NODEMESSAGE;

    if (CmiMyNode()==destNode) {
        CmiSendNodeSelf(dupmsg);
    } else {
    #if ENABLE_CONVERSE_QD
        CQdCreate(CpvAccess(cQdState), 1);
    #endif
        lapiSendFn(CmiNodeFirst(destNode), size, dupmsg, ReleaseMsg, dupmsg, 1);
    }
    MACHSTATE(3,"} Sending sync node message end");
}

CmiCommHandle CmiAsyncNodeSendFn(int destNode, int size, char *msg) {
    void *handle;
    CMI_BROADCAST_ROOT(msg) = 0;
    CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;

    MACHSTATE1(3,"[%p] Sending async node message begin {",CmiGetState());
    /* if we are the destination, send ourself a copy of the message */
    if (CmiMyNode()==destNode) {
        CmiSendNodeSelf(CopyMsg(msg, size));
        MACHSTATE(3,"} Sending async node message end");
        return 0;
    }

    handle = malloc(sizeof(int));
    *((int *)handle) = 1;

#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), 1);
#endif
    lapiSendFn(CmiNodeFirst(destNode), size, msg, DeliveredMsg, handle, 0);
    /* the message may have been duplicated and already delivered if we are in SMP
       mode and the destination is on the same node, but there is no optimized
       check for that. */
    MACHSTATE(3,"} Sending async node message end");
    return handle;

}

void CmiFreeNodeSendFn(int destNode, int size, char *msg) {
    CMI_BROADCAST_ROOT(msg) = 0;
    CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;

    MACHSTATE1(3,"[%p] Sending sync free node message begin {",CmiGetState());
    if (CmiMyNode()==destNode) {
        CmiSendNodeSelf(msg);
    } else {
    #if ENABLE_CONVERSE_QD
        CQdCreate(CpvAccess(cQdState), 1);
    #endif
        lapiSendFn(CmiNodeFirst(destNode), size, msg, ReleaseMsg, msg, 1);
    }
    MACHSTATE(3,"} Sending sync free node message end");
}
#endif

/*********************** BROADCAST FUNCTIONS **********************/
#if CMK_SMP
/** 
  * Sending msgs to cores in the same node 
  * "includeSelf" indicates whether the msg should be sent to 
  * the proc of rank CMI_DEST_RANK(msg) 
  * --Chao mei 
  *  
  */
void SendMsgToPeers(int size, char *msg, int includeSelf){
    if(includeSelf) {
        int i;
        for(i=0; i<CmiMyNodeSize(); i++) {
            char *dupmsg = CopyMsg(msg,size);
            CmiPushPE(i,dupmsg);
        }
    }else{
        int i;
        int excludeRank = CMI_DEST_RANK(msg);
        for(i=excludeRank+1; i<CmiMyNodeSize(); i++) {
            char *dupmsg = CopyMsg(msg,size);
            CmiPushPE(i,dupmsg);
        }
        for(i=0; i<excludeRank; i++) {
            char *dupmsg = CopyMsg(msg,size);
            CmiPushPE(i,dupmsg);
        }
    }
}
#endif

/** 
 * SendSpanningChildren only sends inter-node messages. The 
 * intra-node messages are delivered via SendMsgToPeers if it is 
 * in SMP mode. 
 */
#if ENSURE_MSG_PAIRORDER
void SendSpanningChildren(int size, char *msg, int srcPe, int *seqNoArr){
#else
void SendSpanningChildren(int size, char *msg) {
#endif
    int startproc = CMI_BROADCAST_ROOT(msg)-1;
    int startnode = CmiNodeOf(startproc);
    int i, rp;
    char *dupmsg;

    CmiAssert(startnode>=0 && startnode<CmiNumNodes());
    
    MACHSTATE3(3, "[%p] SendSpanningChildren on proc=%d with start proc %d begin {",CmiGetState(), CmiMyPe(), startproc);  
    /* rp is the relative node id from start node */  
    rp = CmiMyNode() - startnode;
    if(rp<0) rp += CmiNumNodes();
    for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int p = BROADCAST_SPANNING_FACTOR*rp + i;
        if (p > CmiNumNodes() - 1) break;
        p += startnode;
        p = p%CmiNumNodes();
        
#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        lapiSendFn(CmiNodeFirst(p), size, msg, ReleaseMsg, msg, 0);
#else
        dupmsg = CopyMsg(msg, size);
    #if ENSURE_MSG_PAIRORDER
        CMI_MSG_SRCPE(dupmsg) = srcPe;
        CMI_MSG_SEQNO(dupmsg) = getNextMsgSeqNo(seqNoArr, CmiNodeFirst(p));
        setNextMsgSeqNo(seqNoArr, CmiNodeFirst(p), CMI_MSG_SEQNO(dupmsg));
    #endif
        lapiSendFn(CmiNodeFirst(p), size, dupmsg, ReleaseMsg, dupmsg, 1);
#endif
    }    

    MACHSTATE3(3, "[%p] SendSpanningChildren on proc=%d with start proc %d end }",CmiGetState(), CmiMyPe(), startproc);    
}

/* Return the smallest integer that is larger than or equal to log2(i), e.g CmiLog2(14) = 4*/
static int CmiLog2 (int i) {
    int m;
    for (m=0; i>(1<<m); ++m);
    return m;
}

/* send msg along the hypercube in broadcast. (Chao Mei) */
#if ENSURE_MSG_PAIRORDER
void SendHypercube(int size, char *msg, int srcPe, int *seqNoArr){
#else
void SendHypercube(int size, char *msg) {
#endif
    int i, dist;   
    char *dupmsg;
    int dims = CmiLog2(CmiNumNodes());
    int startproc = CMI_BROADCAST_ROOT(msg)-1;
    int startnode = CmiNodeOf(startproc);
    /* relative proc id to startnode */
    int rp = CmiMyNode() - startnode;    
    if(rp < 0) rp += CmiNumNodes();
    dist = rp;

    MACHSTATE3(3, "[%p] SendHypercube on proc=%d with start proc %d begin {",CmiGetState(), CmiMyPe(), startproc);    
    for(i=0; i<dims; i++) { 
        if((dist & 1) == 1) break;

        /* destnode is still the relative node id from startnode */
        int destnode = rp + (1<<i);
        if(destnode > CmiNumNodes()-1) break; 
        
        destnode += startnode;
        destnode = destnode % CmiNumNodes();

        CmiAssert(destnode != CmiMyNode());
               
#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        lapiSendFn(CmiNodeFirst(destnode), size, msg, ReleaseMsg, msg, 0);
#else
        dupmsg = CopyMsg(msg, size);
    #if ENSURE_MSG_PAIRORDER
        CMI_MSG_SRCPE(dupmsg) = srcPe;
        CMI_MSG_SEQNO(dupmsg) = getNextMsgSeqNo(seqNoArr, CmiNodeFirst(destnode));
        setNextMsgSeqNo(seqNoArr, CmiNodeFirst(destnode), CMI_MSG_SEQNO(dupmsg));
    #endif
        lapiSendFn(CmiNodeFirst(destnode), size, dupmsg, ReleaseMsg, dupmsg, 1);
#endif        
        dist = dist >> 1;    
    }    
    MACHSTATE3(3, "[%p] SendHypercube on proc=%d with start proc %d end }",CmiGetState(), CmiMyPe(), startproc);    
}

void CmiSyncBroadcastGeneralFn(int size, char *msg) {    /* ALL_EXCEPT_ME  */
    int i, rank;
    MACHSTATE3(3,"[%p] Sending sync broadcast message %p with size %d begin {",CmiGetState(), msg, size);

#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
#endif

#if CMK_BROADCAST_SPANNING_TREE
    CMI_BROADCAST_ROOT(msg) = CmiMyPe()+1;
    /** 
     * since the broadcast msg will be relayed separately on 
     * remote procs by SendSpanningChildren, the actual msg size 
     * needs to be recorded in the header for proper memory 
     * allocation. E.g. the bcast msg is relayed in PumpMsgComplete.
     * But at that time, the actual msg size is not known if the 
     * filed inside the msg header is not set. The unknown actual 
     * msg size will cause the relay of msg fail because CopyMsg 
     * doesn't have a correct msg size input!! -Chao Mei 
     */
    CMI_MSG_SIZE(msg) = size;
    /* node-aware spanning tree, so bcast msg is always delivered to the first core on each node */
    CMI_DEST_RANK(msg) = 0;
#if ENSURE_MSG_PAIRORDER    
    SendSpanningChildren(size, msg, CmiMyPe(), CpvAccess(bcastMsgSeqInfo).nextMsgSeqNo);
#else
    SendSpanningChildren(size, msg);
#endif
        
#if CMK_SMP
    CMI_DEST_RANK(msg) = CmiMyRank();
    SendMsgToPeers(size, msg, 0);
#endif

#elif CMK_BROADCAST_HYPERCUBE 
   
    CMI_BROADCAST_ROOT(msg) = CmiMyPe()+1;
    CMI_MSG_SIZE(msg) = size;
    CMI_DEST_RANK(msg) = 0;    

#if ENSURE_MSG_PAIRORDER    
    SendHypercube(size, msg, CmiMyPe(), CpvAccess(bcastMsgSeqInfo).nextMsgSeqNo);
#else
    SendHypercube(size, msg);
#endif
      
#if CMK_SMP
    CMI_DEST_RANK(msg) = CmiMyRank();
    SendMsgToPeers(size, msg, 0);
#endif

#else
    CmiState cs = CmiGetState();
    char *dupmsg;

    CMI_BROADCAST_ROOT(msg) = 0;
#if CMK_BROADCAST_USE_CMIREFERENCE
    for (i=cs->pe+1; i<CmiNumPes(); i++) {
        CmiReference(msg);
        lapiSendFn(i, size, msg, ReleaseMsg, msg, 0);
        /*CmiSyncSendFn(i, size, msg) ;*/
    }
    for (i=0; i<cs->pe; i++) {
        CmiReference(msg);
        lapiSendFn(i, size, msg, ReleaseMsg, msg, 0);
        /*CmiSyncSendFn(i, size,msg) ;*/
    }
#else
#if ENSURE_MSG_PAIRORDER
    CMI_MSG_SRCPE(msg) = CmiMyPe();
#endif
    for (i=cs->pe+1; i<CmiNumPes(); i++) {
        dupmsg = CopyMsg(msg, size);
        CMI_DEST_RANK(dupmsg) = CmiRankOf(i);
    #if ENSURE_MSG_PAIRORDER        
        CMI_MSG_SEQNO(dupmsg) = getNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, i);
        setNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, i, CMI_MSG_SEQNO(dupmsg));
    #endif
        lapiSendFn(i, size, dupmsg, ReleaseMsg, dupmsg, 1);        
    }
    for (i=0; i<cs->pe; i++) {
        dupmsg = CopyMsg(msg, size);
    #if ENSURE_MSG_PAIRORDER        
        CMI_MSG_SEQNO(dupmsg) = getNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, i);
        setNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, i, CMI_MSG_SEQNO(dupmsg));
    #endif
        CMI_DEST_RANK(dupmsg) = CmiRankOf(i);
        lapiSendFn(i, size, dupmsg, ReleaseMsg, dupmsg, 1);        
    }
#endif
#endif

    MACHSTATE(3,"} Sending sync broadcast message end");
}

CmiCommHandle CmiAsyncBroadcastGeneralFn(int size, char *msg) {
#if ENSURE_MSG_PAIRORDER
    /* Not sure how to add the msg seq no for async broadcast messages --Chao Mei */
    /* so abort here ! */
    CmiAssert(0);
    return 0;
#else
    CmiState cs = CmiGetState();
    int i, rank;
#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
#endif
    MACHSTATE1(3,"[%p] Sending async broadcast message from {",CmiGetState());
    CMI_BROADCAST_ROOT(msg) = 0;
    void *handle = malloc(sizeof(int));
    *((int *)handle) = CmiNumPes()-1;

    for (i=cs->pe+1; i<CmiNumPes(); i++) {
        CMI_DEST_RANK(msg) = CmiRankOf(i);
        lapiSendFn(i, size, msg, DeliveredMsg, handle, 0);
    }
    for (i=0; i<cs->pe; i++) {
        CMI_DEST_RANK(msg) = CmiRankOf(i);
        lapiSendFn(i, size, msg, DeliveredMsg, handle, 0);
    }

    MACHSTATE(3,"} Sending async broadcast message end");
    return handle;
#endif
}

void CmiSyncBroadcastFn(int size, char *msg) {
    /*CMI_DEST_RANK(msg) = 0;*/
    CmiSyncBroadcastGeneralFn(size, msg);
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg) {
    /*CMI_DEST_RANK(msg) = 0;*/
    return CmiAsyncBroadcastGeneralFn(size, msg);
}

void CmiFreeBroadcastFn(int size, char *msg) {
    CmiSyncBroadcastFn(size,msg);
    CmiFree(msg);
}

void CmiSyncBroadcastAllFn(int size, char *msg) {       /* All including me */
    CmiSendSelf(CopyMsg(msg, size));
    CmiSyncBroadcastFn(size, msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg) {
    CmiSendSelf(CopyMsg(msg, size));
    return CmiAsyncBroadcastFn(size, msg);
}

void CmiFreeBroadcastAllFn(int size, char *msg) {       /* All including me */
    CmiSendSelf(CopyMsg(msg, size));
    CmiSyncBroadcastFn(size, msg);
    CmiFree(msg);
}

#if CMK_NODE_QUEUE_AVAILABLE
void SendSpanningChildrenNode(int size, char *msg){    
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;    
    int i, rp;
    char *dupmsg;

    MACHSTATE3(2, "[%p] SendSpanningChildrenNode on node %d with startnode %d", CmiGetState(), CmiMyNode(), startnode);
    CmiAssert(startnode>=0 && startnode<CmiNumNodes());
        
    rp = CmiMyNode() - startnode;
    if (rp<0) rp+=CmiNumNodes();        
    for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int p = BROADCAST_SPANNING_FACTOR*rp + i;
        if (p > CmiNumNodes() - 1) break;
        p += startnode;
        p = p%CmiNumNodes();
        
#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        lapiSendFn(CmiNodeFirst(p), size, msg, ReleaseMsg, msg, 0);
#else
        dupmsg = CopyMsg(msg, size);
        lapiSendFn(CmiNodeFirst(p), size, dupmsg, ReleaseMsg, dupmsg, 1);
#endif
    }
    MACHSTATE3(3, "[%p] SendSpanningChildrenNode on node=%d with start node %d end }",CmiGetState(), CmiMyNode(), startnode);
}

/* send msg along the hypercube in broadcast. (Chao Mei) */
void SendHypercubeNode(int size, char *msg) {
    int i, dist;   
    char *dupmsg;
    int dims = CmiLog2(CmiNumNodes());
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;
    int rp = CmiMyNode() - startnode;    
    if(rp < 0) rp += CmiNumNodes();
    dist = rp;

    MACHSTATE3(3, "[%p] SendHypercubeNode on node=%d with start node %d begin {",CmiGetState(), CmiMyNode(), startnode);
    for(i=0; i<dims; i++) { 
        if((dist & 1) == 1) break;
        
        int destnode = rp + (1<<i);
        if(destnode > CmiNumNodes()-1) break;
        
        destnode += startnode;
        destnode = destnode % CmiNumNodes();        
                        
        CmiAssert(destnode != CmiMyNode());

#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        lapiSendFn(CmiNodeFirst(destnode), size, msg, ReleaseMsg, msg, 0);
#else
        dupmsg = CopyMsg(msg, size);
        lapiSendFn(CmiNodeFirst(destnode), size, dupmsg, ReleaseMsg, dupmsg, 1);
#endif        
        dist = dist >> 1;    
    }
    MACHSTATE3(3, "[%p] SendHypercubeNode on node=%d with start node %d end }",CmiGetState(), CmiMyNode(), startnode);  
}

void CmiSyncNodeBroadcastGeneralFn(int size, char *msg) {    /* ALL_EXCEPT_THIS_NODE  */     
#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
#endif
#if CMK_BROADCAST_SPANNING_TREE

    MACHSTATE1(3,"[%p] Sending sync node broadcast message (use spanning tree) begin {",CmiGetState());
    CMI_BROADCAST_ROOT(msg) = -(CmiMyNode()+1);
    CMI_MSG_SIZE(msg) = size;
    CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;    
    SendSpanningChildrenNode(size, msg);
    
#elif CMK_BROADCAST_HYPERCUBE

    MACHSTATE1(3,"[%p] Sending sync node broadcast message (use Hypercube) begin {",CmiGetState());
    CMI_BROADCAST_ROOT(msg) = -(CmiMyNode()+1);
    CMI_MSG_SIZE(msg) = size;
    CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;    
    SendHypercubeNode(size, msg);
    
#else      
    char *dupmsg;
    int i;
    MACHSTATE1(3,"[%p] Sending sync node broadcast message (use p2p) begin {",CmiGetState());

    CMI_BROADCAST_ROOT(msg) = 0;
    CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
    for (i=CmiMyNode()+1; i<CmiNumNodes(); i++) {
        dupmsg = CopyMsg(msg, size);        
        lapiSendFn(CmiNodeFirst(i), size, dupmsg, ReleaseMsg, dupmsg, 1);        
    }
    for (i=0; i<CmiMyNode(); i++) {
        dupmsg = CopyMsg(msg, size);        
        lapiSendFn(CmiNodeFirst(i), size, dupmsg, ReleaseMsg, dupmsg, 1);        
    }
#endif
    MACHSTATE(3,"} Sending sync node broadcast message end");
}

CmiCommHandle CmiAsyncNodeBroadcastGeneralFn(int size, char *msg) {
    int i;

#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
#endif

    MACHSTATE1(3,"[%p] Sending async node broadcast message from {",CmiGetState());
    CMI_BROADCAST_ROOT(msg) = 0;
    CMI_DEST_RANK(msg) =DGRAM_NODEMESSAGE;
    void *handle = malloc(sizeof(int));
    *((int *)handle) = CmiNumNodes()-1;
    for (i=CmiMyNode()+1; i<CmiNumNodes(); i++) {        
        lapiSendFn(CmiNodeFirst(i), size, msg, DeliveredMsg, handle, 0);
    }
    for (i=0; i<CmiMyNode(); i++) {        
        lapiSendFn(CmiNodeFirst(i), size, msg, DeliveredMsg, handle, 0);
    }

    MACHSTATE(3,"} Sending async broadcast message end");
    return handle;
}

void CmiSyncNodeBroadcastFn(int size, char *msg) {
    /*CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;*/
    CmiSyncNodeBroadcastGeneralFn(size, msg);
}

CmiCommHandle CmiAsyncNodeBroadcastFn(int size, char *msg) {
    /*CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;*/
    return CmiAsyncNodeBroadcastGeneralFn(size, msg);
}

void CmiFreeNodeBroadcastFn(int size, char *msg) {
    CmiSyncNodeBroadcastFn(size, msg);
    CmiFree(msg);
}

void CmiSyncNodeBroadcastAllFn(int size, char *msg) {
    CmiSendNodeSelf(CopyMsg(msg, size));
    CmiSyncNodeBroadcastFn(size, msg);
}

CmiCommHandle CmiAsyncNodeBroadcastAllFn(int size, char *msg) {
    CmiSendNodeSelf(CopyMsg(msg, size));
    return CmiAsyncNodeBroadcastFn(size, msg);
}

void CmiFreeNodeBroadcastAllFn(int size, char *msg) {
    CmiSendNodeSelf(CopyMsg(msg, size));
    CmiSyncNodeBroadcastFn(size, msg);
    CmiFree(msg);
}
#endif

#if ! CMK_MULTICAST_LIST_USE_COMMON_CODE
void CmiSyncListSendFn(int, int *, int, char*) {

}

CmiCommHandle CmiAsyncListSendFn(int, int *, int, char*) {

}

void CmiFreeListSendFn(int, int *, int, char*) {

}
#endif

#if ! CMK_VECTOR_SEND_USES_COMMON_CODE
void CmiSyncVectorSend(int, int, int *, char **) {

}

CmiCommHandle CmiAsyncVectorSend(int, int, int *, char **) {

}

void CmiSyncVectorSendAndFree(int, int, int *, char **) {

}
#endif


/************************** MAIN (non comm related functions) ***********************************/

void ConverseExit(void) {
    MACHSTATE2(2, "[%d-%p] entering ConverseExit begin {",CmiMyPe(),CmiGetState());
    
    /* TODO: Is it necessary to drive the network progress here?? -Chao Mei */

    /* A barrier excluding the comm thread if present */
    CmiNodeBarrier();

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
    /* Signal the comm thd to exit now! */
    if(CmiMyRank()==0) {
        commThdExit = 1;
    }
#endif

    ConverseCommonExit();

#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyPe() == 0) CmiPrintf("End of program\n");
#endif

    CmiNodeBarrier();

    MACHSTATE2(2, "[%d-%p] ConverseExit end }",CmiMyPe(),CmiGetState());
#if CMK_SMP
    if(CmiMyRank()==0) {
        check_lapi(LAPI_Gfence, (lapiContext));
        check_lapi(LAPI_Term, (lapiContext));
        exit(EXIT_SUCCESS);
    }else{
        pthread_exit(NULL);
    }
#else
    check_lapi(LAPI_Gfence, (lapiContext));      
    check_lapi(LAPI_Term, (lapiContext));
    exit(EXIT_SUCCESS);
#endif
}

/* Those be Cpved?? --Chao Mei */
static char     **Cmi_argv;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */

/** 
 *  CmiNotifyIdle is used in non-SMP mode when the proc is idle.
 *  When the proc is idle, the LAPI_Probe is called to make
 *  network progress.
 *  
 *  While in SMP mode, CmiNotifyStillIdle and CmiNotifyBeginIdle
 *  are used. Particularly, when idle, the frequency of calling
 *  lapi_probe (network progress) is given by "sleepMs"
 */
void CmiNotifyIdle(void) {    
    AdvanceCommunication();
    CmiYield();
}

typedef struct {
    int sleepMs; /*Milliseconds to sleep while idle*/
    int nIdles; /*Number of times we've been idle in a row*/
    CmiState cs; /*Machine state*/
} CmiIdleState;

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

#define SPINS_BEFORE_SLEEP     20

static void CmiNotifyStillIdle(CmiIdleState *s) {
    MACHSTATE2(2,"[%p] still idle (%d) begin {",CmiGetState(),CmiMyPe());
    s->nIdles++;
    if (s->nIdles>SPINS_BEFORE_SLEEP) { /*Start giving some time back to the OS*/
        s->sleepMs+=2;
        if (s->sleepMs>10) s->sleepMs=10;
    }
    if (s->sleepMs>0) {
        MACHSTATE1(2,"idle sleep (%d) {",CmiMyPe());
        CmiIdleLock_sleep(&s->cs->idle,s->sleepMs);
        MACHSTATE1(2,"} idle sleep (%d)",CmiMyPe());
    }

    AdvanceCommunication();
    
    MACHSTATE1(2,"still idle (%d) end }",CmiMyPe());
}

#if MACHINE_DEBUG_LOG
CpvDeclare(FILE *, debugLog);
#endif


#if ENSURE_MSG_PAIRORDER
static void initMsgOrderInfo(MsgOrderInfo *info){
    int i;
    info->nextMsgSeqNo = malloc(CmiNumPes()*sizeof(int));
    memset(info->nextMsgSeqNo, 0, CmiNumPes()*sizeof(int));
    
    info->expectedMsgSeqNo = malloc(CmiNumPes()*sizeof(int));
    memset(info->expectedMsgSeqNo, 0, CmiNumPes()*sizeof(int));
    
    info->oooMsgBuffer = malloc(CmiNumPes()*sizeof(void **));
    memset(info->oooMsgBuffer, 0, CmiNumPes()*sizeof(void **));
    
    info->oooMaxOffset = malloc(CmiNumPes()*sizeof(unsigned char));
    memset(info->oooMaxOffset, 0, CmiNumPes()*sizeof(unsigned char));

    info->CUR_WINDOW_SIZE = malloc(CmiNumPes()*sizeof(unsigned char));
    for(i=0; i<CmiNumPes(); i++) info->CUR_WINDOW_SIZE[i] = INIT_WINDOW_SIZE;
}
#endif

/* Only called from communication thread in SMP mode */
static void CommunicationServer(int sleepTime) {
#if CMK_SMP_NO_COMMTHD
    sleep(sleepTime);
#else
    
    if(commThdExit) {
        MACHSTATE2(2, "[%d-%p] comm server exit begin {",CmiMyPe(),CmiGetState());
        ConverseCommonExit();        
        MACHSTATE2(2, "[%d-%p] comm server exit end }",CmiMyPe(),CmiGetState());
        pthread_exit(NULL);
    }

    if(!CsvAccess(lapiInterruptMode)) check_lapi(LAPI_Probe,(lapiContext));

#if CMK_IMMEDIATE_MSG
    /*MACHSTATE1(2, "[%p] Handling Immediate Message begin{",CmiGetState());*/
    CmiHandleImmediate();
    /*MACHSTATE1(2, "[%p] Handling Immediate Message end}",CmiGetState());*/
#endif

#if DECOUPLE_BCAST_PROCESS
    /*MACHSTATE1(2, "[%p] Enter ProcessBroadcastMsg begin{",CmiGetState());*/
    ProcessBroadcastMsg(0);
    /*MACHSTATE1(2, "[%p] Enter ProcessBroadcastMsg end}",CmiGetState());*/
#endif

    /* sleep(sleepTime) */
#endif
}

static void ConverseRunPE(int everReturn) {
    CmiIdleState *s;
    char** CmiMyArgv;
    int i;
    CpvInitialize(void *,CmiLocalQueue);
    CpvInitialize(unsigned, networkProgressCount);

    CpvInitialize(PCQueue, procBcastQ);
    CpvAccess(procBcastQ) = PCQueueCreate();

#if ENSURE_MSG_PAIRORDER
    CpvInitialize(MsgOrderInfo, p2pMsgSeqInfo);
    initMsgOrderInfo(&CpvAccess(p2pMsgSeqInfo));

    CpvInitialize(MsgOrderInfo, bcastMsgSeqInfo); 
    initMsgOrderInfo(&CpvAccess(bcastMsgSeqInfo));    
#endif

#if MACHINE_DEBUG_LOG
    {
        char ln[200];
        sprintf(ln,"debugLog.%d",CmiMyPe());
        CpvInitialize(FILE *, debugLog);
        CpvAccess(debugLog)=fopen(ln,"w");
    }
#endif

    /* To make sure cpvaccess is correct? -Chao Mei */
    CmiNodeAllBarrier();

    /* Added by Chao Mei */
#if CMK_SMP
    if(CmiMyRank()) {
        /* The master core of this node is already initialized */
        lapi_info_t info;
        int testnode, testnumnodes;
        memset(&info,0,sizeof(info));
        /*CpvInitialize(lapi_handle_t *, lapiContext);
        CpvAccess(lapiContext) = CpvAccessOther(lapiContext, 0)+CmiMyRank();
        CpvAccess(lapiContext) = CpvAccessOther(lapiContext, 0);
        */       
        MACHSTATE2(2, "My rank id=%d, lapicontext=%p", CmiMyRank(), &lapiContext);

        check_lapi(LAPI_Qenv,(lapiContext, TASK_ID, &testnode));
        check_lapi(LAPI_Qenv,(lapiContext, NUM_TASKS, &testnumnodes));

        MACHSTATE3(2, "My rank id=%d, Task id=%d, Num tasks=%d", CmiMyRank(), testnode, testnumnodes);
    }
#endif


    MACHSTATE2(2, "[%d] ConverseRunPE (thread %p)",CmiMyRank(),CmiGetState());
	
    CmiNodeAllBarrier();

    MACHSTATE(2, "After NodeBarrier in ConverseRunPE");
    
    CpvAccess(CmiLocalQueue) = CmiGetState()->localqueue;

    if(CmiMyRank())
        CmiMyArgv=CmiCopyArgs(Cmi_argv);
    else
	CmiMyArgv=Cmi_argv;

    CthInit(CmiMyArgv);

    MACHSTATE(2, "After CthInit in ConverseRunPE");

    ConverseCommonInit(CmiMyArgv);

    MACHSTATE(2, "After ConverseCommonInit in ConverseRunPE");

   /**
     * In SMP, the machine layer usually has one comm thd, and it is 
     * designed to be responsible for all network communication. So 
     * if there's no dedicated processor for the comm thread, it has 
     * to share a proc with a worker thread. In this scenario, 
     * the worker thread needs to yield for some time to give CPU 
     * time to comm thread. However, in current configuration, we 
     * will always dedicate one proc for the comm thd, therefore, 
     * such yielding scheme is not necessary.  Besides, avoiding 
     * this yielding scheme improves performance because worker 
     * thread doesn't need to yield and will be more responsive to 
     * incoming messages. So, we will always use CmiNotifyIdle 
     * instead. 
     *  
     * --Chao Mei
     */
#if 0 && CMK_SMP
    s=CmiNotifyGetState();
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
#if !CMK_SMP || CMK_SMP_NO_COMMTHD
    /* If there's comm thread, then comm thd is responsible for advancing comm */
    if(!CsvAccess(lapiInterruptMode)) {
        CcdCallOnConditionKeep(CcdPERIODIC_10ms, (CcdVoidFn)AdvanceCommunication, NULL);
    }
#endif
#endif

#if CMK_IMMEDIATE_MSG
    /* Converse initialization finishes, immediate messages can be processed.
       node barrier previously should take care of the node synchronization */
    _immediateReady = 1;
#endif

    /* communication thread */
    if (CmiMyRank() == CmiMyNodeSize()) {
        Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
        MACHSTATE2(3, "[%p]: Comm thread on node %d is going to be a communication server", CmiGetState(), CmiMyNode());            
        while(1) CommunicationServer(5);        
    } else { /* worker thread */
        if (!everReturn) {
            Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
            MACHSTATE1(3, "[%p]: Worker thread is going to work", CmiGetState());
            if (Cmi_usrsched==0) CsdScheduler(-1);
            ConverseExit();
        }
    }
}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret) {
    int n,i;

    lapi_info_t info;

#if ENSURE_MSG_PAIRORDER
   cmplHdlrThdLock = CmiCreateLock();
#endif

    /* processor per node */
    /** 
     *  We have to determin the ppn at this point in order to create
     *  the corresponding number of lapiContext instances.
     */
#if CMK_SMP
    CmiMyNodeSize() = 1;
    CmiGetArgInt(argv,"+ppn", &CmiMyNodeSize());
#else
    if (CmiGetArgFlag(argv,"+ppn")) {
        CmiAbort("+ppn cannot be used in non SMP version!\n");
    }
#endif

    memset(&info,0,sizeof(info));

    /* Register error handler (redundant?) -- added by Chao Mei*/
    info.err_hndlr = (LAPI_err_hndlr *)lapi_err_hndlr;

    /* Indicates the number of completion handler threads to create */
    /* The number of completion hndlr thds will affect the atomic PCQueue operations!! */
    /* NOTE: num_compl_hndlr_thr is obsolete now! --Chao Mei */
    /* info.num_compl_hndlr_thr = 1; */

    check_lapi(LAPI_Init,(&lapiContext, &info));
    
    /* It's a good idea to start with a fence,
       because packets recv'd before a LAPI_Init are just dropped. */
    check_lapi(LAPI_Gfence,(lapiContext));

    check_lapi(LAPI_Qenv,(lapiContext, TASK_ID, &CmiMyNode()));
    check_lapi(LAPI_Qenv,(lapiContext, NUM_TASKS, &CmiNumNodes()));

    /* Make polling as the default mode as real apps have better perf */
    CsvAccess(lapiInterruptMode) = 0;
    if(CmiGetArgFlag(argv,"+poll")) CsvAccess(lapiInterruptMode) = 0;
    if(CmiGetArgFlag(argv,"+nopoll")) CsvAccess(lapiInterruptMode) = 1;    

    check_lapi(LAPI_Senv,(lapiContext, ERROR_CHK, lapiDebugMode));
    check_lapi(LAPI_Senv,(lapiContext, INTERRUPT_SET, CsvAccess(lapiInterruptMode)));

    if(CmiMyNode()==0) {
        printf("Running lapi in interrupt mode: %d\n", CsvAccess(lapiInterruptMode));
        printf("Running lapi with %d completion handler threads.\n", info.num_compl_hndlr_thr);
    }

    /** 
     *  Associate PumpMsgsBegin with var "lapiHeaderHandler". Then inside Xfer calls,
     *  lapiHeaderHandler could be used to indicate the callback
     *  instead of PumpMsgsBegin --Chao Mei
     */
    check_lapi(LAPI_Addr_set,(lapiContext,(void *)PumpMsgsBegin,lapiHeaderHandler));

    CmiNumPes() = CmiNumNodes() * CmiMyNodeSize();
    Cmi_nodestart = CmiMyNode() * CmiMyNodeSize();

    Cmi_argv = argv;
    Cmi_startfn = fn;
    Cmi_usrsched = usched;

    if (CmiGetArgFlag(argv,"++debug")) {  /*Pause so user has a chance to start and attach debugger*/
        printf("CHARMDEBUG> Processor %d has PID %d\n",CmiMyNode(),getpid());
        if (!CmiGetArgFlag(argv,"++debug-no-pause"))
            sleep(30);
    }

    CsvInitialize(CmiNodeState, NodeState);
    CmiNodeStateInit(&CsvAccess(NodeState));

#if CMK_NODE_QUEUE_AVAILABLE
    CsvInitialize(PCQueue, nodeBcastQ);
    CsvAccess(nodeBcastQ) = PCQueueCreate();
#endif

    CmiStartThreads(argv);

    ConverseRunPE(initret);
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message) {
    CmiError(message);
    LAPI_Term(lapiContext);
    exit(1);
}

static void PerrorExit(const char *msg) {
    perror(msg);
    LAPI_Term(lapiContext);
    exit(1);
}

