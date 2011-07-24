/*****************************************************************************
LAPI version of machine layer
Based on the template machine layer

Developed by
Filippo Gioachin   03/23/05
Chao Mei 01/28/2010, 05/07/2011
************************************************************************/

/** @file
 * LAPI based machine layer
 * @ingroup Machine
 */
/*@{*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#include <lapi.h>
#include "converse.h"

/*Support for ++debug: */
#include <unistd.h> /*For getpid()*/
#include <stdlib.h> /*For sleep()*/

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
 *   (If ENSURE_MSG_PAIRORDER is 1, then setting CMK_OFFLOAD_BCAST_PROCESS to 1
 *   will make the msg seqno increase at step of 1 w/o data race;
 *     CMK_OFFLOAD_BCAST_PROCESS = 0|1;
 * ===========================================================
 * 2. SMP mode without comm thd:
 *    CMK_SMP = 1;
 *    CMK_PCQUEUE_LOCK = 1;
 *    CMK_SMP_NO_COMMTHD = 1;
 *
 *    ENSURE_MSG_PAIRORDER and ENABLE_CONVERSE_QD have same options as in non-SMP mode;
 *
 *    CMK_OFFLOAD_BCAST_PROCESS has same options as in non-SMP mode;
 * ===========================================================
 *  3. SMP mode with comm thd:
 *     CMK_SMP = 1;
 *     CMK_PCQUEUE_LOCK = 1;
 *     CMK_SMP_NO_COMMTHD = 0;
 *
 *     ENSURE_MSG_PAIRORDER and ENABLE_CONVERSE_QD have same options as in non-SMP mode;
 *
 *     (The following must be set with 1 as bcast msg is dealt with in comm thd!)
 *     CMK_OFFLOAD_BCAST_PROCESS = 1;
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
 *        (enable CMK_OFFLOAD_BCAST_PROCESS)  or in internal LAPI completion
 *        handler threads;
 *     c) the destination rank of proc-level bcast msg is always 0;
 *  3. SMP w/ comm thd:
 *     a) imm msgs are processed in comm thd;
 *     b) forwarding bcast msgs is done in comm thd;
 *     c) same with 2 c)
 *
 */
/* =========END OF EXPLANATION OF MACRO USAGE=============*/

#include "machine.h"

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
CmiNodeLock CmiCreateLock() {
    CmiNodeLock lk = (CmiNodeLock)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(lk,(pthread_mutexattr_t *)0);
    return lk;
}
#define CmiLock(lock) (pthread_mutex_lock(lock))
#define CmiUnlock(lock) (pthread_mutex_unlock(lock))
#define CmiTryLock(lock) (pthread_mutex_trylock(lock))
void CmiDestroyLock(CmiNodeLock lock) {
    pthread_mutex_destroy(lock);
    free(lock);
}
#define CMK_PCQUEUE_LOCK 1
#endif
#include "pcqueue.h"

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
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
#define ENSURE_MSG_PAIRORDER 0

#if ENSURE_MSG_PAIRORDER

/* NOTE: this feature requires O(P) space on every proc to be functional */

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

typedef struct MsgOrderInfoStruct {
    /* vars used on sender side */
    int *nextMsgSeqNo;

    /* vars used on recv side */
    int *expectedMsgSeqNo;
    void ***oooMsgBuffer;
    unsigned char *oooMaxOffset;
    unsigned char *CUR_WINDOW_SIZE;
} MsgOrderInfo;

/**
 * Once p2p msgs are ensured in-order delivery for a pair
 * of procs, then the bcast msg is guaranteed correspondently
 * as the order of msgs sent is fixed (i.e. the spanning tree
 * or the hypercube is fixed)
 */
CpvDeclare(MsgOrderInfo, p2pMsgSeqInfo);
#endif

/**
 *  Enable this macro will offload the broadcast relay from the
 *  internal completion handler thread. This will make the msg
 *  seqno free of data-race. In SMP mode with comm thread where
 *  comm thread will forward bcast msgs, this macro should be
 *  enabled.
 */
/* TODO: need re-consideration */
#define CMK_OFFLOAD_BCAST_PROCESS 1
/* When ENSURE_MSG_PAIRORDER is enabled, CMK_OFFLOAD_BCAST_PROCESS
 * requires to be defined because if the bcast message is processed
 * in the lapi completion handler (the internal lapi thread), then
 * there's possibility of data races in setting the sequence number.
 * In SMP mode, the bcast forwarding should be offloaded from the
 * completion handler to the comm thread to reduce the overhead if
 * there's comm thread. If there's no commthread, and if cpv is
 * tls-based, then CMK_OFFLOAD_BCAST_PROCESS also requires enabled
 * because there's charm proc-private variable access would be
 * incorrect in lapi's internal threads. -Chao Mei
 */
#if (CMK_SMP && (!CMK_SMP_NO_COMMTHD || (CMK_TLS_THREAD && !CMK_NOT_USE_TLS_THREAD))) || ENSURE_MSG_PAIRORDER
#undef CMK_OFFLOAD_BCAST_PROCESS
#define CMK_OFFLOAD_BCAST_PROCESS 1
#endif

#if CMK_OFFLOAD_BCAST_PROCESS
CsvDeclare(PCQueue, procBcastQ);
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(PCQueue, nodeBcastQ);
#endif
#endif

/* =======End of Definitions of Performance-Specific Macros =======*/


/* =======Beginning of Definitions of Msg Header Specific Macros =======*/
/* msg size and srcpe are required info as they will be used for forwarding
 * bcast msg and for ensuring message ordering
 */
#define CMI_MSG_SRCPE(msg)               ((CmiMsgHeaderBasic *)msg)->srcpe
#define CMI_MSG_SEQNO(msg)               ((CmiMsgHeaderBasic *)msg)->seqno
/* =======End of Definitions of Msg Header Specific Macros =======*/

/* =====Beginning of Definitions of Message-Corruption Related Macros=====*/
/* =====End of Definitions of Message-Corruption Related Macros=====*/


/* =====Beginning of Declarations of Machine Specific Variables===== */

static int lapiDebugMode=0;
CsvDeclare(int, lapiInterruptMode);

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
/* =====End of Declarations of Machine Specific Variables===== */


/* =====Beginning of Declarations of Machine Specific Functions===== */

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
                           lapi_err_t *err_type, int *task_ID, int *src) {
    char errstr[LAPI_MAX_ERR_STRING];
    LAPI_Msg_string(*error_code, errstr);
    fprintf(stderr, "ERROR IN LAPI: %s for task %d at src %d\n", errstr, *task_ID, *src);
    LAPI_Term(*hndl);
    exit(1);
}

/* ===== Beginging of functions regarding ensure in-order msg delivery ===== */
#if ENSURE_MSG_PAIRORDER

/**
 * "setNextMsgSeqNo" actually sets the current seqno, the
 * "getNextMsgSeqNo" will increment the seqno, i.e.,
 * "getNextMsgSeqNo" returns the next seqno based on the previous
 * seqno stored in the seqno array.
 * --Chao Mei
 */
static int getNextMsgSeqNo(int *seqNoArr, int destPe) {
    int ret = seqNoArr[destPe];
    ret++;
    return ret;
}
static void setNextMsgSeqNo(int *seqNoArr, int destPe, int val) {
    /* the seq no. may fast-forward to a new round (i.e., starting from 1 again!) */
    if (val>=MAX_MSG_SEQNO) val -= MAX_MSG_SEQNO;
    seqNoArr[destPe] = val;
}

#define getNextExpectedMsgSeqNo(seqNoArr,pe) getNextMsgSeqNo(seqNoArr, pe)
#define setNextExpectedMsgSeqNo(seqNoArr,pe,val) setNextMsgSeqNo(seqNoArr, pe, val)

static int checkMsgInOrder(char *msg, MsgOrderInfo *info);
#endif
/* ===== End of functions regarding ensure in-order msg delivery ===== */


/* The machine-specific send function */
static CmiCommHandle MachineSendFuncForLAPI(int destNode, int size, char *msg, int mode);
#define LrtsSendFunc MachineSendFuncForLAPI

/* ### Beginning of Machine-startup Related Functions ### */
static void MachineInitForLAPI(int *argc, char ***argv, int *numNodes, int *myNodeID);
#define LrtsInit MachineInitForLAPI

static void MachinePreCommonInitForLAPI(int everReturn);
static void MachinePostCommonInitForLAPI(int everReturn);
#define LrtsPreCommonInit MachinePreCommonInitForLAPI
#define LrtsPostCommonInit MachinePostCommonInitForLAPI
/* ### End of Machine-startup Related Functions ### */

/* ### Beginning of Machine-running Related Functions ### */
static void AdvanceCommunicationForLAPI();
#define LrtsAdvanceCommunication AdvanceCommunicationForLAPI

static void DrainResourcesForLAPI(); /* used when exit */
#define LrtsDrainResources DrainResourcesForLAPI

static void MachineExitForLAPI();
#define LrtsExit MachineExitForLAPI
/* ### End of Machine-running Related Functions ### */

/* ### Beginning of Idle-state Related Functions ### */
/* ### End of Idle-state Related Functions ### */

static void MachinePostNonLocalForLAPI();
#define LrtsPostNonLocal MachinePostNonLocalForLAPI

/* =====End of Declarations of Machine Specific Functions===== */

/**
 *  Macros that overwrites the common codes, such as
 *  CMK_SMP_NO_COMMTHD, NETWORK_PROGRESS_PERIOD_DEFAULT,
 *  USE_COMMON_SYNC_P2P, CMK_HAS_SIZE_IN_MSGHDR,
 *  CMK_OFFLOAD_BCAST_PROCESS etc.
 */
/* For async msg sending ops, using lapi specific implementations */
#define USE_COMMON_ASYNC_BCAST 0
#define CMK_OFFLOAD_BCAST_PROCESS 1
#include "machine-common.h"
#include "machine-common.c"

/* The machine specific msg-sending function */

/* ######Beginning of functions for sending a msg ###### */
//lapi sending completion callback
/* The following two are callbacks for sync and async send respectively */
static void ReleaseMsg(lapi_handle_t *myLapiContext, void *msg, lapi_sh_info_t *info) {
    MACHSTATE2(2,"[%d] ReleaseMsg begin %p {",CmiMyNode(),msg);
    check_lapi_err(info->reason, "ReleaseMsg", __LINE__);
    CmiFree(msg);
    MACHSTATE(2,"} ReleaseMsg end");
}

static void DeliveredMsg(lapi_handle_t *myLapiContext, void *msg, lapi_sh_info_t *info) {
    MACHSTATE1(2,"[%d] DeliveredMsg begin {",CmiMyNode());
    check_lapi_err(info->reason, "DeliveredMsg", __LINE__);
    *((int *)msg) = *((int *)msg) - 1;
    MACHSTATE(2,"} DeliveredMsg end");
}

static INLINE_KEYWORD void lapiSendFn(int destNode, int size, char *msg, scompl_hndlr_t *shdlr, void *sinfo) {
    lapi_xfer_t xfer_cmd;

    MACHSTATE3(2,"lapiSendFn to destNode=%d with msg %p (isImm=%d) begin {",destNode,msg, CmiIsImmediate(msg));
    MACHSTATE3(2, "inside lapiSendFn 1: size=%d, sinfo=%p, deliverable=%d", size, sinfo, deliverable);

    MACHSTATE2(2, "Ready to call LAPI_Xfer with destNode=%d, destRank=%d",destNode,CMI_DEST_RANK(msg));

    xfer_cmd.Am.Xfer_type = LAPI_AM_XFER;
    xfer_cmd.Am.flags     = 0;
    xfer_cmd.Am.tgt       = destNode;
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

static CmiCommHandle MachineSendFuncForLAPI(int destNode, int size, char *msg, int mode) {
    scompl_hndlr_t *shdlr = NULL;
    void *sinfo = NULL;

    if (mode==P2P_SYNC) {
        shdlr = ReleaseMsg;
        sinfo = (void *)msg;
    } else if (mode==P2P_ASYNC) {
        shdlr = DeliveredMsg;
        sinfo = malloc(sizeof(int));
        *((int *)sinfo) = 1;
    }

    CMI_MSG_SIZE(msg) = size;

#if ENSURE_MSG_PAIRORDER
#if CMK_NODE_QUEUE_AVAILABLE
    if (CMI_DEST_RANK(msg) == DGRAM_NODEMESSAGE) {
        lapiSendFn(destNode, size, msg, shdlr, sinfo);
        return sinfo;
    }
#endif
    int destPE = CmiNodeFirst(destNode)+CMI_DEST_RANK(msg);
    CMI_MSG_SRCPE(msg) = CmiMyPe();
    /* Note: This could be executed on comm threads, where CmiMyPe() >= CmiNumPes() */
    CMI_MSG_SEQNO(msg) = getNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE);
    setNextMsgSeqNo(CpvAccess(p2pMsgSeqInfo).nextMsgSeqNo, destPE, CMI_MSG_SEQNO(msg));
#endif

    lapiSendFn(destNode, size, msg, shdlr, sinfo);
    return sinfo;
}

/* Lapi-specific implementation of async msg sending operations */
#if !USE_COMMON_ASYNC_BCAST
CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg) {
#if ENSURE_MSG_PAIRORDER
    /* Not sure how to add the msg seq no for async broadcast messages --Chao Mei */
    /* so abort here ! */
    CmiAssert(0);
    return 0;
#else
    int i, rank;
    int mype = CmiMyPe();
#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
#endif
    MACHSTATE1(3,"[%d] Sending async broadcast message from {",CmiMyNode());
    CMI_BROADCAST_ROOT(msg) = 0;
    void *handle = malloc(sizeof(int));
    *((int *)handle) = CmiNumPes()-1;

    for (i=mype+1; i<CmiNumPes(); i++) {
        CMI_DEST_RANK(msg) = CmiRankOf(i);
        lapiSendFn(CmiNodeOf(i), size, msg, DeliveredMsg, handle);
    }
    for (i=0; i<mype; i++) {
        CMI_DEST_RANK(msg) = CmiRankOf(i);
        lapiSendFn(CmiNodeOf(i), size, msg, DeliveredMsg, handle);
    }

    MACHSTATE(3,"} Sending async broadcast message end");
    return handle;
#endif
}

#if CMK_NODE_QUEUE_AVAILABLE
CmiCommHandle CmiAsyncNodeBroadcastFn(int size, char *msg) {
    int i;

#if ENABLE_CONVERSE_QD
    CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
#endif

    MACHSTATE1(3,"[%d] Sending async node broadcast message from {",CmiMyNode());
    CMI_BROADCAST_ROOT(msg) = 0;
    CMI_DEST_RANK(msg) =DGRAM_NODEMESSAGE;
    void *handle = malloc(sizeof(int));
    *((int *)handle) = CmiNumNodes()-1;
    for (i=CmiMyNode()+1; i<CmiNumNodes(); i++) {
        lapiSendFn(i, size, msg, DeliveredMsg, handle);
    }
    for (i=0; i<CmiMyNode(); i++) {
        lapiSendFn(i, size, msg, DeliveredMsg, handle);
    }

    MACHSTATE(3,"} Sending async broadcast message end");
    return handle;
}
#endif
#endif/* end of !USE_COMMON_ASYNC_BCAST */

int CmiAsyncMsgSent(CmiCommHandle handle) {
    return (*((int *)handle) == 0)?1:0;
}

void CmiReleaseCommHandle(CmiCommHandle handle) {
#ifndef CMK_OPTIMIZE
    if (*((int *)handle) != 0) CmiAbort("Released a CmiCommHandle not free!");
#endif
    free(handle);
}
/* ######End of functions for sending a msg ###### */

/* ######Beginning of functions for receiving a msg ###### */
/* lapi recv callback when the first packet of the msg arrives as header handler*/
static void* PumpMsgsBegin(lapi_handle_t *myLapiContext,
                           void *hdr, uint *uhdr_len,
                           lapi_return_info_t *msg_info,
                           compl_hndlr_t **comp_h, void **comp_am_info);
/* lapi recv completion callback when all the msg is received */
static void PumpMsgsComplete(lapi_handle_t *myLapiContext, void *am_info);

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
    MACHSTATE1(2,"[%d] PumpMsgsBegin begin {",CmiMyNode());
    /* prepare the space for receiving the data, set the completion handler to
       be executed inline */
    msg_buf = (void *)CmiAlloc(msg_info->msg_len);

    msg_info->ret_flags = LAPI_SEND_REPLY;
    *comp_h = PumpMsgsComplete;
    *comp_am_info = msg_buf;
    MACHSTATE(2,"} PumpMsgsBegin end");
    return msg_buf;

}

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

    MACHSTATE3(2,"[%d] PumpMsgsComplete with msg %p (isImm=%d) begin {",CmiMyNode(), msg, CmiIsImmediate(msg));
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
       ) {
        MsgOrderInfo *info;
        info = &CpvAccessOther(p2pMsgSeqInfo, destrank);
        MACHSTATE1(2, "Check msg in-order for p2p msg %p", msg);

        if (checkMsgInOrder(msg,info)) {
            MACHSTATE(2,"} PumpMsgsComplete end ");
            return;
        }
    }
#endif

    handleOneRecvedMsg(CMI_MSG_SIZE(msg), msg);

    MACHSTATE(2,"} PumpMsgsComplete end ");
    return;
}

/* utility function for ensuring the message pair-ordering */
#if ENSURE_MSG_PAIRORDER
/* return 1 if this msg is an out-of-order incoming message */
/**
 * Returns 1 if this "msg" is an out-of-order message, or
 * this "msg" is a late message which triggers the process
 * of all buffered ooo msgs.
 * --Chao Mei
 */
static int checkMsgInOrder(char *msg, MsgOrderInfo *info) {
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
    if (expectedSeqNo == incomingSeqNo) {
        /* Two cases: has ooo msg buffered or not */
        maxOffset = (info->oooMaxOffset)[srcpe];
        if (maxOffset>0) {
            MACHSTATE1(4, "Processing all buffered ooo msgs (maxOffset=%d) including the just recved begin {", maxOffset);
            curWinSize = info->CUR_WINDOW_SIZE[srcpe];
            toProcessMsgBuffer = malloc((curWinSize+1)*sizeof(void *));
            /* process the msg just recved */
            toProcessMsgBuffer[numMsgs++] = msg;
            /* process the buffered ooo msg until the first empty slot in the window */
            destMsgBuffer = (info->oooMsgBuffer)[srcpe];
            for (curOffset=0; curOffset<maxOffset; curOffset++) {
                char *curMsg = destMsgBuffer[curOffset];
                if (curMsg == NULL) {
                    CmiAssert(curOffset!=(maxOffset-1));
                    break;
                }
                toProcessMsgBuffer[numMsgs++] = curMsg;
                destMsgBuffer[curOffset] = NULL;
            }
            /* Update expected seqno, maxOffset and slide the window */
            if (curOffset < maxOffset) {
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
                for (i=0; i<maxOffset-curOffset-1; i++) {
                    destMsgBuffer[i] = destMsgBuffer[curOffset+i+1];
                }
                for (i=maxOffset-curOffset-1; i<maxOffset; i++) {
                    destMsgBuffer[i] = NULL;
                }
                (info->oooMaxOffset)[srcpe] = maxOffset-curOffset-1;
                setNextExpectedMsgSeqNo(info->expectedMsgSeqNo, srcpe, expectedSeqNo+curOffset);
            } else {
                /* there's no remaining buffered ooo msgs */
                (info->oooMaxOffset)[srcpe] = 0;
                setNextExpectedMsgSeqNo(info->expectedMsgSeqNo, srcpe, expectedSeqNo+maxOffset);
            }

            CmiUnlock(cmplHdlrThdLock);

            /* Process the msgs */
            for (i=0; i<numMsgs; i++) {
                char *curMsg = toProcessMsgBuffer[i];
                if (CMI_BROADCAST_ROOT(curMsg)>0) {

#if CMK_OFFLOAD_BCAST_PROCESS
                    PCQueuePush(CsvAccess(procBcastQ), curMsg);
#else
                    processProcBcastMsg(CMI_MSG_SIZE(curMsg), curMsg);
#endif
                } else {
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
        } else {
            /* An expected msg recved without any ooo msg buffered */
            MACHSTATE1(4, "Receiving an expected msg with seqno=%d\n", incomingSeqNo);
            setNextExpectedMsgSeqNo(info->expectedMsgSeqNo, srcpe, expectedSeqNo);

            CmiUnlock(cmplHdlrThdLock);
            return 0;
        }
    }

    MACHSTATE2(4, "Receiving an out-of-order msg with seqno=%d, but expect seqno=%d", incomingSeqNo, expectedSeqNo);
    curWinSize = info->CUR_WINDOW_SIZE[srcpe];
    if ((info->oooMsgBuffer)[srcpe]==NULL) {
        (info->oooMsgBuffer)[srcpe] = malloc(curWinSize*sizeof(void *));
        memset((info->oooMsgBuffer)[srcpe], 0, curWinSize*sizeof(void *));
    }
    destMsgBuffer = (info->oooMsgBuffer)[srcpe];
    curOffset = incomingSeqNo - expectedSeqNo;
    maxOffset = (info->oooMaxOffset)[srcpe];
    if (curOffset<0) {
        /* It's possible that the seqNo starts with another round (exceeding MAX_MSG_SEQNO) with 1 */
        curOffset += MAX_MSG_SEQNO;
    }
    if (curOffset > curWinSize) {
        int newWinSize;
        if (curOffset > MAX_WINDOW_SIZE) {
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
    if (curOffset > maxOffset) (info->oooMaxOffset)[srcpe] = curOffset;

    CmiUnlock(cmplHdlrThdLock);
    return 1;
}
#endif /* end of ENSURE_MSG_PAIRORDER */

/* ######End of functions for receiving a msg ###### */

/* ######Beginning of functions related with communication progress ###### */

static INLINE_KEYWORD void AdvanceCommunicationForLAPI() {
    /* What about CMK_SMP_NO_COMMTHD in the original implementation?? */
    /* It does nothing but sleep */
    if (!CsvAccess(lapiInterruptMode)) check_lapi(LAPI_Probe,(lapiContext));
}
/* ######End of functions related with communication progress ###### */

static void MachinePostNonLocalForLAPI() {
    /* None here */
}

/**
 * TODO: What will be the effects if calling LAPI_Probe in the
 * interrupt mode??? --Chao Mei
 */
/* Network progress function is used to poll the network when for
   messages. This flushes receive buffers on some  implementations*/
#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl() {
    if (!CsvAccess(lapiInterruptMode)) check_lapi(LAPI_Probe,(lapiContext));

#if CMK_IMMEDIATE_MSG
    MACHSTATE1(2, "[%d] Handling Immediate Message begin {",CmiMyNode());
    CmiHandleImmediate();
    MACHSTATE1(2, "[%d] Handling Immediate Message end }",CmiMyNode());
#endif

#if CMK_SMP && !CMK_SMP_NO_COMMTHD && CMK_OFFLOAD_BCAST_PROCESS
    if (CmiMyRank()==CmiMyNodeSize()) processBcastQs(); /* FIXME ????????????????*/
#endif
}
#endif

/* ######Beginning of functions related with exiting programs###### */
void DrainResourcesForLAPI() {
    /* None here */
}

void MachineExitForLAPI(void) {
    check_lapi(LAPI_Gfence, (lapiContext));
    check_lapi(LAPI_Term, (lapiContext));
    exit(EXIT_SUCCESS);
}
/* ######End of functions related with exiting programs###### */


/* ######Beginning of functions related with starting programs###### */
/**
 *  Obtain the number of nodes, my node id, and consuming machine layer
 *  specific arguments
 */
static void MachineInitForLAPI(int *argc, char ***argv, int *numNodes, int *myNodeID) {

    lapi_info_t info;
    char **largv = *argv;

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

    check_lapi(LAPI_Qenv,(lapiContext, TASK_ID, myNodeID));
    check_lapi(LAPI_Qenv,(lapiContext, NUM_TASKS, numNodes));

    /* Make polling as the default mode as real apps have better perf */
    CsvAccess(lapiInterruptMode) = 0;
    if (CmiGetArgFlag(largv,"+poll")) CsvAccess(lapiInterruptMode) = 0;
    if (CmiGetArgFlag(largv,"+nopoll")) CsvAccess(lapiInterruptMode) = 1;

    check_lapi(LAPI_Senv,(lapiContext, ERROR_CHK, lapiDebugMode));
    check_lapi(LAPI_Senv,(lapiContext, INTERRUPT_SET, CsvAccess(lapiInterruptMode)));

    if (*myNodeID == 0) {
        printf("Running lapi in interrupt mode: %d\n", CsvAccess(lapiInterruptMode));
        printf("Running lapi with %d completion handler threads.\n", info.num_compl_hndlr_thr);
    }

    /**
     *  Associate PumpMsgsBegin with var "lapiHeaderHandler". Then inside Xfer calls,
     *  lapiHeaderHandler could be used to indicate the callback
     *  instead of PumpMsgsBegin --Chao Mei
     */
    check_lapi(LAPI_Addr_set,(lapiContext,(void *)PumpMsgsBegin,lapiHeaderHandler));

    if (CmiGetArgFlag(largv,"++debug")) {  /*Pause so user has a chance to start and attach debugger*/
        printf("CHARMDEBUG> Processor %d has PID %d\n",*myNodeID,getpid());
        if (!CmiGetArgFlag(largv,"++debug-no-pause"))
            sleep(30);
    }

#if ENSURE_MSG_PAIRORDER
    cmplHdlrThdLock = CmiCreateLock();
#endif
}

#if MACHINE_DEBUG_LOG
CpvDeclare(FILE *, debugLog);
#endif

#if ENSURE_MSG_PAIRORDER
static void initMsgOrderInfo(MsgOrderInfo *info) {
    int i;
    int totalPEs = CmiNumPes();
#if CMK_SMP && CMK_OFFLOAD_BCAST_PROCESS
    /* the comm thread will also access such info */
    totalPEs += CmiNumNodes();
#endif
    info->nextMsgSeqNo = malloc(totalPEs*sizeof(int));
    memset(info->nextMsgSeqNo, 0, totalPEs*sizeof(int));

    info->expectedMsgSeqNo = malloc(totalPEs*sizeof(int));
    memset(info->expectedMsgSeqNo, 0, totalPEs*sizeof(int));

    info->oooMsgBuffer = malloc(totalPEs*sizeof(void **));
    memset(info->oooMsgBuffer, 0, totalPEs*sizeof(void **));

    info->oooMaxOffset = malloc(totalPEs*sizeof(unsigned char));
    memset(info->oooMaxOffset, 0, totalPEs*sizeof(unsigned char));

    info->CUR_WINDOW_SIZE = malloc(totalPEs*sizeof(unsigned char));
    for (i=0; i<totalPEs; i++) info->CUR_WINDOW_SIZE[i] = INIT_WINDOW_SIZE;
}
#endif

static void MachinePreCommonInitForLAPI(int everReturn) {
#if ENSURE_MSG_PAIRORDER
    CpvInitialize(MsgOrderInfo, p2pMsgSeqInfo);
    initMsgOrderInfo(&CpvAccess(p2pMsgSeqInfo));
#endif

#if MACHINE_DEBUG_LOG
    {
        char ln[200];
        sprintf(ln,"debugLog.%d",CmiMyPe());
        CpvInitialize(FILE *, debugLog);
        CpvAccess(debugLog)=fopen(ln,"w");
    }
#endif

}

static void MachinePostCommonInitForLAPI(int everReturn) {
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

    /* Not registering any Idle-state related functions right now!! */

#if 0 && CMK_SMP
    s=CmiNotifyGetState();
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
#if !CMK_SMP || CMK_SMP_NO_COMMTHD
    /* If there's comm thread, then comm thd is responsible for advancing comm */
    if (!CsvAccess(lapiInterruptMode)) {
        CcdCallOnConditionKeep(CcdPERIODIC_10ms, (CcdVoidFn)AdvanceCommunicationForLAPI, NULL);
    }
#endif
#endif
}
/* ######End of functions related with starting programs###### */

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

/* What's the difference of this function with CmiAbort????
   and whether LAPI_Term is needed?? It should be shared across
   all machine layers.
static void PerrorExit(const char *msg) {
    perror(msg);
    LAPI_Term(lapiContext);
    exit(1);
}
*/

/* Barrier related functions */
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


/*@}*/

