/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** @file
 * MPI based machine layer
 * @ingroup Machine
 */
/*@{*/

#include <stdio.h>
#include <errno.h>
#include "converse.h"
#include <mpi.h>
#if CMK_TIMER_USE_XT3_DCLOCK
#include <catamount/dclock.h>
#endif


#ifdef AMPI
#  warning "We got the AMPI version of mpi.h, instead of the system version--"
#  warning "   Try doing an 'rm charm/include/mpi.h' and building again."
#  error "Can't build Charm++ using AMPI version of mpi.h header"
#endif

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
#include <stdlib.h> /*For sleep()*/

#include "machine.h"
#include "pcqueue.h"

/* =======Beginning of Definitions of Performance-Specific Macros =======*/
/* Whether to use multiple send queue in SMP mode */
#define MULTI_SENDQUEUE    0

/* ###Beginning of flow control related macros ### */
#define CMI_EXERT_SEND_CAP 0
#define CMI_EXERT_RECV_CAP 0

#define CMI_DYNAMIC_EXERT_CAP 0
/* This macro defines the max number of msgs in the sender msg buffer
 * that is allowed for recving operation to continue
 */
static int CMI_DYNAMIC_OUTGOING_THRESHOLD=4;
#define CMI_DYNAMIC_MAXCAPSIZE 1000
static int CMI_DYNAMIC_SEND_CAPSIZE=4;
static int CMI_DYNAMIC_RECV_CAPSIZE=3;
/* initial values, -1 indiates there's no cap */
static int dynamicSendCap = CMI_DYNAMIC_MAXCAPSIZE;
static int dynamicRecvCap = CMI_DYNAMIC_MAXCAPSIZE;

#if CMI_EXERT_SEND_CAP
#define SEND_CAP 3
#endif

#if CMI_EXERT_RECV_CAP
#define RECV_CAP 2
#endif
/* ###End of flow control related macros ### */

/* ###Beginning of machine-layer-tracing related macros ### */
#if CMK_TRACE_ENABLED && CMK_SMP_TRACE_COMMTHREAD
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
/* ###End of machine-layer-tracing related macros ### */

/* ###Beginning of POST_RECV related macros ### */
/*
 * If MPI_POST_RECV is defined, we provide default values for
 * size and number of posted recieves. If MPI_POST_RECV_COUNT
 * is set then a default value for MPI_POST_RECV_SIZE is used
 * if not specified by the user.
 */
#define MPI_POST_RECV 0

/* Making those parameters configurable for testing them easily */

#if MPI_POST_RECV
static int MPI_POST_RECV_COUNT=10;
static int MPI_POST_RECV_LOWERSIZE=2000;
static int MPI_POST_RECV_UPPERSIZE=4000;
static int MPI_POST_RECV_SIZE;

CpvDeclare(unsigned long long, Cmi_posted_recv_total);
CpvDeclare(unsigned long long, Cmi_unposted_recv_total);
CpvDeclare(MPI_Request*, CmiPostedRecvRequests); /* An array of request handles for posted recvs */
CpvDeclare(char*,CmiPostedRecvBuffers);
#endif

/* to avoid MPI's in order delivery, changing MPI Tag all the time */
#define TAG     1375
#if MPI_POST_RECV
#define POST_RECV_TAG       (TAG+1)
#define BARRIER_ZERO_TAG  TAG
#else
#define BARRIER_ZERO_TAG   (TAG-1)
#endif
/* ###End of POST_RECV related related macros ### */

#if CMK_BLUEGENEL
#define MAX_QLEN 8
#define NETWORK_PROGRESS_PERIOD_DEFAULT 16
#else
#define NETWORK_PROGRESS_PERIOD_DEFAULT 0
#define MAX_QLEN 200
#endif
/* =======End of Definitions of Performance-Specific Macros =======*/


/* =====Beginning of Definitions of Message-Corruption Related Macros=====*/
#define CMI_MAGIC(msg)			 ((CmiMsgHeaderBasic *)msg)->magic
#define CHARM_MAGIC_NUMBER		 126

#if CMK_ERROR_CHECKING
extern unsigned char computeCheckSum(unsigned char *data, int len);
static int checksum_flag = 0;
#define CMI_SET_CHECKSUM(msg, len)	\
	if (checksum_flag)  {	\
	  ((CmiMsgHeaderBasic *)msg)->cksum = 0; 	\
	  ((CmiMsgHeaderBasic *)msg)->cksum = computeCheckSum((unsigned char*)msg, len);	\
	}
#define CMI_CHECK_CHECKSUM(msg, len)	\
	if (checksum_flag) 	\
	  if (computeCheckSum((unsigned char*)msg, len) != 0) 	\
	    CmiAbort("Fatal error: checksum doesn't agree!\n");
#else
#define CMI_SET_CHECKSUM(msg, len)
#define CMI_CHECK_CHECKSUM(msg, len)
#endif
/* =====End of Definitions of Message-Corruption Related Macros=====*/


/* =====Beginning of Declarations of Machine Specific Variables===== */
#include <signal.h>
void (*signal_int)(int);

static int _thread_provided = -1; /* Indicating MPI thread level */
static int idleblock = 0;

/* A simple list for msgs that have been sent by MPI_Isend */
typedef struct msg_list {
    char *msg;
    struct msg_list *next;
    int size, destpe;
#if CMK_SMP_TRACE_COMMTHREAD
    int srcpe;
#endif
    MPI_Request req;
} SMSG_LIST;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

int MsgQueueLen=0;
static int request_max;
/*FLAG: consume outstanding Isends in scheduler loop*/
static int no_outstanding_sends=0;

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

typedef struct ProcState {
#if MULTI_SENDQUEUE
    PCQueue      sendMsgBuf;       /* per processor message sending queue */
#endif
    CmiNodeLock  recvLock;		    /* for cs->recv */
} ProcState;
static ProcState  *procState;

#if CMK_SMP && !MULTI_SENDQUEUE
static PCQueue sendMsgBuf;
static CmiNodeLock  sendMsgBufLock = NULL;        /* for sendMsgBuf */
#endif
/* =====End of Declarations of Machine Specific Variables===== */


/* =====Beginning of Declarations of Machine Specific Functions===== */
/* Utility functions */
#if CMK_BLUEGENEL
extern void MPID_Progress_test();
#endif
static size_t CmiAllAsyncMsgsSent(void);
static void CmiReleaseSentMessages(void);
static int PumpMsgs(void);
static void PumpMsgsBlocking(void);

#if CMK_SMP
static int MsgQueueEmpty();
static int RecvQueueEmpty();
static int SendMsgBuf();
static  void EnqueueMsg(void *m, int size, int node);
#endif

/* The machine-specific send function */
static CmiCommHandle MachineSpecificSendForMPI(int destNode, int size, char *msg, int mode);
#define LrtsSendFunc MachineSpecificSendForMPI

/* ### Beginning of Machine-startup Related Functions ### */
static void MachineInitForMPI(int *argc, char ***argv, int *numNodes, int *myNodeID);
#define LrtsInit MachineInitForMPI

static void MachinePreCommonInitForMPI(int everReturn);
static void MachinePostCommonInitForMPI(int everReturn);
#define LrtsPreCommonInit MachinePreCommonInitForMPI
#define LrtsPostCommonInit MachinePostCommonInitForMPI
/* ### End of Machine-startup Related Functions ### */

/* ### Beginning of Machine-running Related Functions ### */
static void AdvanceCommunicationForMPI();
#define LrtsAdvanceCommunication AdvanceCommunicationForMPI

static void DrainResourcesForMPI(); /* used when exit */
#define LrtsDrainResources DrainResourcesForMPI

static void MachineExitForMPI();
#define LrtsExit MachineExitForMPI
/* ### End of Machine-running Related Functions ### */

/* ### Beginning of Idle-state Related Functions ### */
void CmiNotifyIdleForMPI(void);
/* ### End of Idle-state Related Functions ### */

static void MachinePostNonLocalForMPI();
#define LrtsPostNonLocal MachinePostNonLocalForMPI

/* =====End of Declarations of Machine Specific Functions===== */

/**
 *  Macros that overwrites the common codes, such as
 *  CMK_SMP_NO_COMMTHD, NETWORK_PROGRESS_PERIOD_DEFAULT,
 *  USE_COMMON_SYNC_P2P, CMK_HAS_SIZE_IN_MSGHDR,
 *  CMK_OFFLOAD_BCAST_PROCESS etc.
 */
#define CMK_HAS_SIZE_IN_MSGHDR 0
#include "machine-lrts.h"
#include "machine-common-core.c"

/* The machine specific msg-sending function */

#if CMK_SMP
static void EnqueueMsg(void *m, int size, int node) {
    SMSG_LIST *msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
    MACHSTATE1(3,"EnqueueMsg to node %d {{ ", node);
    msg_tmp->msg = m;
    msg_tmp->size = size;
    msg_tmp->destpe = node;
    msg_tmp->next = 0;

#if CMK_SMP_TRACE_COMMTHREAD
    msg_tmp->srcpe = CmiMyPe();
#endif

#if MULTI_SENDQUEUE
    PCQueuePush(procState[CmiMyRank()].sendMsgBuf,(char *)msg_tmp);
#else
    /*CmiLock(sendMsgBufLock);*/
    PCQueuePush(sendMsgBuf,(char *)msg_tmp);
    /*CmiUnlock(sendMsgBufLock);*/
#endif

    MACHSTATE3(3,"}} EnqueueMsg to %d finish with queue %p len: %d", node, sendMsgBuf, PCQueueLength(sendMsgBuf));
}
#endif

/* The function that calls MPI_Isend so that both non-SMP and SMP could use */
static CmiCommHandle MPISendOneMsg(SMSG_LIST *smsg) {
    int node = smsg->destpe;
    int size = smsg->size;
    char *msg = smsg->msg;

#if !CMI_DYNAMIC_EXERT_CAP && !CMI_EXERT_SEND_CAP
    while (MsgQueueLen > request_max) {
        CmiReleaseSentMessages();
        PumpMsgs();
    }
#endif

    MACHSTATE2(3,"MPI_send to node %d rank: %d{", node, CMI_DEST_RANK(msg));
#if CMK_ERROR_CHECKING
    CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
    CMI_SET_CHECKSUM(msg, size);
#endif

#if MPI_POST_RECV
    if (size>=MPI_POST_RECV_LOWERSIZE && size <= MPI_POST_RECV_UPPERSIZE) {
        START_EVENT();
        if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,POST_RECV_TAG,MPI_COMM_WORLD,&(smsg->req)))
            CmiAbort("MPISendOneMsg: MPI_Isend failed!\n");
        /*END_EVENT(40);*/
    } else {
        START_EVENT();
        if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(smsg->req)))
            CmiAbort("MPISendOneMsg: MPI_Isend failed!\n");
        /*END_EVENT(40);*/
    }
#else
    START_EVENT();
    if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(smsg->req)))
        CmiAbort("MPISendOneMsg: MPI_Isend failed!\n");
    /*END_EVENT(40);*/
#endif

#if CMK_SMP_TRACE_COMMTHREAD
    traceBeginCommOp(msg);
    traceChangeLastTimestamp(CpvAccess(projTraceStart));
    /* traceSendMsgComm must execute after traceBeginCommOp because
         * we pretend we execute an entry method, and inside this we
         * pretend we will send another message. Otherwise how could
         * a message creation just before an entry method invocation?
         * If such logic is broken, the projections will not trace
         * messages correctly! -Chao Mei
         */
    traceSendMsgComm(msg);
    traceEndCommOp(msg);
#if CMI_MPI_TRACE_MOREDETAILED
    char tmp[64];
    sprintf(tmp, "MPI_Isend: from proc %d to proc %d", smsg->srcpe, CmiNodeFirst(node)+CMI_DEST_RANK(msg));
    traceUserSuppliedBracketedNote(tmp, 40, CpvAccess(projTraceStart), CmiWallTimer());
#endif
#endif

    MACHSTATE(3,"}MPI_send end");
    MsgQueueLen++;
    if (sent_msgs==0)
        sent_msgs = smsg;
    else
        end_sent->next = smsg;
    end_sent = smsg;
    return (CmiCommHandle) &(smsg->req);
}

static CmiCommHandle MachineSpecificSendForMPI(int destNode, int size, char *msg, int mode) {
    /* Ignoring the mode for MPI layer */

    CmiState cs = CmiGetState();
    SMSG_LIST *msg_tmp;
    int  rank;

    CmiAssert(destNode != CmiMyNode());
#if CMK_SMP
    EnqueueMsg(msg, size, destNode);
    return 0;
#else
    /* non smp */
    msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
    msg_tmp->msg = msg;
    msg_tmp->destpe = destNode;
    msg_tmp->size = size;
    msg_tmp->next = 0;
    return MPISendOneMsg(msg_tmp);
#endif
}

static size_t CmiAllAsyncMsgsSent(void) {
    SMSG_LIST *msg_tmp = sent_msgs;
    MPI_Status sts;
    int done;

    while (msg_tmp!=0) {
        done = 0;
        if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts))
            CmiAbort("CmiAllAsyncMsgsSent: MPI_Test failed!\n");
        if (!done)
            return 0;
        msg_tmp = msg_tmp->next;
        /*    MsgQueueLen--; ????? */
    }
    return 1;
}

int CmiAsyncMsgSent(CmiCommHandle c) {

    SMSG_LIST *msg_tmp = sent_msgs;
    int done;
    MPI_Status sts;

    while ((msg_tmp) && ((CmiCommHandle)&(msg_tmp->req) != c))
        msg_tmp = msg_tmp->next;
    if (msg_tmp) {
        done = 0;
        if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts))
            CmiAbort("CmiAsyncMsgSent: MPI_Test failed!\n");
        return ((done)?1:0);
    } else {
        return 1;
    }
}

void CmiReleaseCommHandle(CmiCommHandle c) {
    return;
}

/* ######Beginning of functions related with communication progress ###### */
static void CmiReleaseSentMessages(void) {
    SMSG_LIST *msg_tmp=sent_msgs;
    SMSG_LIST *prev=0;
    SMSG_LIST *temp;
    int done;
    MPI_Status sts;

#if CMK_BLUEGENEL
    MPID_Progress_test();
#endif

    MACHSTATE1(2,"CmiReleaseSentMessages begin on %d {", CmiMyPe());
    while (msg_tmp!=0) {
        done =0;
#if CMK_SMP_TRACE_COMMTHREAD || CMK_TRACE_COMMOVERHEAD
        double startT = CmiWallTimer();
#endif
        if (MPI_Test(&(msg_tmp->req), &done, &sts) != MPI_SUCCESS)
            CmiAbort("CmiReleaseSentMessages: MPI_Test failed!\n");
        if (done) {
            MACHSTATE2(3,"CmiReleaseSentMessages release one %d to %d", CmiMyPe(), msg_tmp->destpe);
            MsgQueueLen--;
            /* Release the message */
            temp = msg_tmp->next;
            if (prev==0) /* first message */
                sent_msgs = temp;
            else
                prev->next = temp;
            CmiFree(msg_tmp->msg);
            CmiFree(msg_tmp);
            msg_tmp = temp;
        } else {
            prev = msg_tmp;
            msg_tmp = msg_tmp->next;
        }
#if CMK_SMP_TRACE_COMMTHREAD || CMK_TRACE_COMMOVERHEAD
        {
            double endT = CmiWallTimer();
            /* only record the event if it takes more than 1ms */
            if (endT-startT>=0.001) traceUserSuppliedBracketedNote("MPI_Test: release a msg", 60, startT, endT);
        }
#endif
    }
    end_sent = prev;
    MACHSTATE(2,"} CmiReleaseSentMessages end");
}

static int PumpMsgs(void) {
    int nbytes, flg, res;
    char *msg;
    MPI_Status sts;
    int recd=0;

#if CMI_EXERT_RECV_CAP || CMI_DYNAMIC_EXERT_CAP
    int recvCnt=0;
#endif

#if CMK_BLUEGENEL
    MPID_Progress_test();
#endif

    MACHSTATE(2,"PumpMsgs begin {");

#if CMI_DYNAMIC_EXERT_CAP
    dynamicRecvCap = CMI_DYNAMIC_MAXCAPSIZE;
#endif

    while (1) {
#if CMI_EXERT_RECV_CAP
        if (recvCnt==RECV_CAP) break;
#elif CMI_DYNAMIC_EXERT_CAP
        if (recvCnt >= dynamicRecvCap) break;
#endif

        /* First check posted recvs then do  probe unmatched outstanding messages */
#if MPI_POST_RECV
        int completed_index=-1;
        if (MPI_SUCCESS != MPI_Testany(MPI_POST_RECV_COUNT, CpvAccess(CmiPostedRecvRequests), &completed_index, &flg, &sts))
            CmiAbort("PumpMsgs: MPI_Testany failed!\n");
        if (flg) {
            if (MPI_SUCCESS != MPI_Get_count(&sts, MPI_BYTE, &nbytes))
                CmiAbort("PumpMsgs: MPI_Get_count failed!\n");

            recd = 1;
            msg = (char *) CmiAlloc(nbytes);
            memcpy(msg,&(CpvAccess(CmiPostedRecvBuffers)[completed_index*MPI_POST_RECV_SIZE]),nbytes);
            /* and repost the recv */

            START_EVENT();

            if (MPI_SUCCESS != MPI_Irecv(  &(CpvAccess(CmiPostedRecvBuffers)[completed_index*MPI_POST_RECV_SIZE])	,
                                           MPI_POST_RECV_SIZE,
                                           MPI_BYTE,
                                           MPI_ANY_SOURCE,
                                           POST_RECV_TAG,
                                           MPI_COMM_WORLD,
                                           &(CpvAccess(CmiPostedRecvRequests)[completed_index])  ))
                CmiAbort("PumpMsgs: MPI_Irecv failed!\n");

            END_EVENT(50);

            CpvAccess(Cmi_posted_recv_total)++;
        } else {
            res = MPI_Iprobe(MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &flg, &sts);
            if (res != MPI_SUCCESS)
                CmiAbort("MPI_Iprobe failed\n");
            if (!flg) break;
            recd = 1;
            MPI_Get_count(&sts, MPI_BYTE, &nbytes);
            msg = (char *) CmiAlloc(nbytes);

            START_EVENT();

            if (MPI_SUCCESS != MPI_Recv(msg,nbytes,MPI_BYTE,sts.MPI_SOURCE,sts.MPI_TAG, MPI_COMM_WORLD,&sts))
                CmiAbort("PumpMsgs: MPI_Recv failed!\n");

            END_EVENT(30);

            CpvAccess(Cmi_unposted_recv_total)++;
        }
#else
        /* Original version */
#if CMK_SMP_TRACE_COMMTHREAD || CMK_TRACE_COMMOVERHEAD
        double startT = CmiWallTimer();
#endif
        res = MPI_Iprobe(MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &flg, &sts);
        if (res != MPI_SUCCESS)
            CmiAbort("MPI_Iprobe failed\n");

        if (!flg) break;
#if CMK_SMP_TRACE_COMMTHREAD || CMK_TRACE_COMMOVERHEAD
        {
            double endT = CmiWallTimer();
            /* only trace the probe that last longer than 1ms */
            if (endT-startT>=0.001) traceUserSuppliedBracketedNote("MPI_Iprobe before a recv call", 70, startT, endT);
        }
#endif

        recd = 1;
        MPI_Get_count(&sts, MPI_BYTE, &nbytes);
        msg = (char *) CmiAlloc(nbytes);

        START_EVENT();

        if (MPI_SUCCESS != MPI_Recv(msg,nbytes,MPI_BYTE,sts.MPI_SOURCE,sts.MPI_TAG, MPI_COMM_WORLD,&sts))
            CmiAbort("PumpMsgs: MPI_Recv failed!\n");

        /*END_EVENT(30);*/

#endif

#if CMK_SMP_TRACE_COMMTHREAD
        traceBeginCommOp(msg);
        traceChangeLastTimestamp(CpvAccess(projTraceStart));
        traceEndCommOp(msg);
#if CMI_MPI_TRACE_MOREDETAILED
        char tmp[32];
        sprintf(tmp, "MPI_Recv: to proc %d", CmiNodeFirst(CmiMyNode())+CMI_DEST_RANK(msg));
        traceUserSuppliedBracketedNote(tmp, 30, CpvAccess(projTraceStart), CmiWallTimer());
#endif
#elif CMK_TRACE_COMMOVERHEAD
        char tmp[32];
        sprintf(tmp, "MPI_Recv: to proc %d", CmiNodeFirst(CmiMyNode())+CMI_DEST_RANK(msg));
        traceUserSuppliedBracketedNote(tmp, 30, CpvAccess(projTraceStart), CmiWallTimer());
#endif


        MACHSTATE2(3,"PumpMsgs recv one from node:%d to rank:%d", sts.MPI_SOURCE, CMI_DEST_RANK(msg));
        CMI_CHECK_CHECKSUM(msg, nbytes);
#if CMK_ERROR_CHECKING
        if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
            CmiPrintf("Charm++ Abort: Non Charm++ Message Received of size %d. \n", nbytes);
            CmiFree(msg);
            CmiAbort("Abort!\n");
            continue;
        }
#endif

        handleOneRecvedMsg(nbytes, msg);

#if CMI_EXERT_RECV_CAP
        recvCnt++;
#elif CMI_DYNAMIC_EXERT_CAP
        recvCnt++;
#if CMK_SMP
        /* check sendMsgBuf to get the  number of messages that have not been sent
             * which is only available in SMP mode
         * MsgQueueLen indicates the number of messages that have not been released
             * by MPI
             */
        if (PCQueueLength(sendMsgBuf) > CMI_DYNAMIC_OUTGOING_THRESHOLD
                || MsgQueueLen > CMI_DYNAMIC_OUTGOING_THRESHOLD) {
            dynamicRecvCap = CMI_DYNAMIC_RECV_CAPSIZE;
        }
#else
        /* MsgQueueLen indicates the number of messages that have not been released
             * by MPI
             */
        if (MsgQueueLen > CMI_DYNAMIC_OUTGOING_THRESHOLD) {
            dynamicRecvCap = CMI_DYNAMIC_RECV_CAPSIZE;
        }
#endif

#endif

    }

    MACHSTATE(2,"} PumpMsgs end ");
    return recd;
}

/* blocking version */
static void PumpMsgsBlocking(void) {
    static int maxbytes = 20000000;
    static char *buf = NULL;
    int nbytes, flg;
    MPI_Status sts;
    char *msg;
    int recd=0;

    if (!PCQueueEmpty(CmiGetState()->recv)) return;
    if (!CdsFifo_Empty(CpvAccess(CmiLocalQueue))) return;
    if (!CqsEmpty(CpvAccess(CsdSchedQueue))) return;
    if (sent_msgs)  return;

#if 0
    CmiPrintf("[%d] PumpMsgsBlocking. \n", CmiMyPe());
#endif

    if (buf == NULL) {
        buf = (char *) CmiAlloc(maxbytes);
        _MEMCHECK(buf);
    }


#if MPI_POST_RECV
#warning "Using MPI posted receives and PumpMsgsBlocking() will break"
    CmiAbort("Unsupported use of PumpMsgsBlocking. This call should be extended to check posted recvs, cancel them all, and then wait on any incoming message, and then re-post the recvs");
#endif

    START_EVENT();

    if (MPI_SUCCESS != MPI_Recv(buf,maxbytes,MPI_BYTE,MPI_ANY_SOURCE,TAG, MPI_COMM_WORLD,&sts))
        CmiAbort("PumpMsgs: PMP_Recv failed!\n");

    /*END_EVENT(30);*/

    MPI_Get_count(&sts, MPI_BYTE, &nbytes);
    msg = (char *) CmiAlloc(nbytes);
    memcpy(msg, buf, nbytes);

#if CMK_SMP_TRACE_COMMTHREAD
    traceBeginCommOp(msg);
    traceChangeLastTimestamp(CpvAccess(projTraceStart));
    traceEndCommOp(msg);
#if CMI_MPI_TRACE_MOREDETAILED
    char tmp[32];
    sprintf(tmp, "To proc %d", CmiNodeFirst(CmiMyNode())+CMI_DEST_RANK(msg));
    traceUserSuppliedBracketedNote(tmp, 30, CpvAccess(projTraceStart), CmiWallTimer());
#endif
#endif

    handleOneRecvedMsg(nbytes, msg);
}


#if CMK_SMP

/* called by communication thread in SMP */
static int SendMsgBuf() {
    SMSG_LIST *msg_tmp;
    char *msg;
    int node, rank, size;
    int i;
    int sent = 0;

#if CMI_EXERT_SEND_CAP || CMI_DYNAMIC_EXERT_CAP
    int sentCnt = 0;
#endif

#if CMI_DYNAMIC_EXERT_CAP
    dynamicSendCap = CMI_DYNAMIC_MAXCAPSIZE;
#endif

    MACHSTATE(2,"SendMsgBuf begin {");
#if MULTI_SENDQUEUE
    for (i=0; i<_Cmi_mynodesize+1; i++) { /* subtle: including comm thread */
        if (!PCQueueEmpty(procState[i].sendMsgBuf)) {
            msg_tmp = (SMSG_LIST *)PCQueuePop(procState[i].sendMsgBuf);
#else
    /* single message sending queue */
    /* CmiLock(sendMsgBufLock); */
    msg_tmp = (SMSG_LIST *)PCQueuePop(sendMsgBuf);
    /* CmiUnlock(sendMsgBufLock); */
    while (NULL != msg_tmp) {
#endif
            MPISendOneMsg(msg_tmp);
            sent=1;

#if CMI_EXERT_SEND_CAP
            if (++sentCnt == SEND_CAP) break;
#elif CMI_DYNAMIC_EXERT_CAP
            if (++sentCnt >= dynamicSendCap) break;
            if (MsgQueueLen > CMI_DYNAMIC_OUTGOING_THRESHOLD)
                dynamicSendCap = CMI_DYNAMIC_SEND_CAPSIZE;
#endif

#if ! MULTI_SENDQUEUE
            /* CmiLock(sendMsgBufLock); */
            msg_tmp = (SMSG_LIST *)PCQueuePop(sendMsgBuf);
            /* CmiUnlock(sendMsgBufLock); */
#endif
        }
#if MULTI_SENDQUEUE
    }
#endif
    MACHSTATE(2,"}SendMsgBuf end ");
    return sent;
}

static int MsgQueueEmpty() {
    int i;
#if MULTI_SENDQUEUE
    for (i=0; i<_Cmi_mynodesize; i++)
        if (!PCQueueEmpty(procState[i].sendMsgBuf)) return 0;
#else
    return PCQueueEmpty(sendMsgBuf);
#endif
    return 1;
}

/* test if all processors recv queues are empty */
static int RecvQueueEmpty() {
    int i;
    for (i=0; i<_Cmi_mynodesize; i++) {
        CmiState cs=CmiGetStateN(i);
        if (!PCQueueEmpty(cs->recv)) return 0;
    }
    return 1;
}


#define REPORT_COMM_METRICS 0
#if REPORT_COMM_METRICS
static double pumptime = 0.0;
                         static double releasetime = 0.0;
                                                     static double sendtime = 0.0;
#endif

#endif //end of CMK_SMP

static void AdvanceCommunicationForMPI() {
#if REPORT_COMM_METRICS
    double t1, t2, t3, t4;
    t1 = CmiWallTimer();
#endif

#if CMK_SMP
    PumpMsgs();

#if REPORT_COMM_METRICS
    t2 = CmiWallTimer();
#endif

    CmiReleaseSentMessages();
#if REPORT_COMM_METRICS
    t3 = CmiWallTimer();
#endif

    SendMsgBuf();

#if REPORT_COMM_METRICS
    t4 = CmiWallTimer();
    pumptime += (t2-t1);
    releasetime += (t3-t2);
    sendtime += (t4-t3);
#endif

#else /* non-SMP case */
    CmiReleaseSentMessages();

#if REPORT_COMM_METRICS
    t2 = CmiWallTimer();
#endif
    PumpMsgs();

#if REPORT_COMM_METRICS
    t3 = CmiWallTimer();
    pumptime += (t3-t2);
    releasetime += (t2-t1);
#endif

#endif /* end of #if CMK_SMP */
}
/* ######End of functions related with communication progress ###### */

static void MachinePostNonLocalForMPI() {
#if !CMK_SMP
    if (no_outstanding_sends) {
        while (MsgQueueLen>0) {
            AdvanceCommunicationForMPI();
        }
    }

    /* FIXME: I don't think the following codes are needed because
     * it repeats the same job of the next call of CmiGetNonLocal
     */
#if 0
    if (!msg) {
        CmiReleaseSentMessages();
        if (PumpMsgs())
            return  PCQueuePop(cs->recv);
        else
            return 0;
    }
#endif
#endif
}

/* Idle-state related functions: called in non-smp mode */
void CmiNotifyIdleForMPI(void) {
    CmiReleaseSentMessages();
    if (!PumpMsgs() && idleblock) PumpMsgsBlocking();
}

/* Network progress function is used to poll the network when for
   messages. This flushes receive buffers on some  implementations*/
#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl() {
#if !CMK_SMP
    PumpMsgs();
#if CMK_IMMEDIATE_MSG
    CmiHandleImmediate();
#endif
#else
    /*Not implemented yet. Communication server does not seem to be
      thread safe, so only communication thread call it */
    if (CmiMyRank() == CmiMyNodeSize())
        CommunicationServerThread(0);
#endif
}
#endif

/* ######Beginning of functions related with exiting programs###### */
void DrainResourcesForMPI() {
#if !CMK_SMP
    while (!CmiAllAsyncMsgsSent()) {
        PumpMsgs();
        CmiReleaseSentMessages();
    }
#else
    while (!MsgQueueEmpty() || !CmiAllAsyncMsgsSent()) {
        CmiReleaseSentMessages();
        SendMsgBuf();
        PumpMsgs();
    }
#endif
    MACHSTATE(2, "Machine exit barrier begin {");
    START_EVENT();
    if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD))
        CmiAbort("DrainResourcesForMPI: MPI_Barrier failed!\n");
    END_EVENT(10);
    MACHSTATE(2, "} Machine exit barrier end");
}

void MachineExitForMPI(void) {
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    int doPrint = 0;
#if CMK_SMP
    if (CmiMyNode()==0) doPrint = 1;
#else
    if (CmiMyPe()==0) doPrint = 1;
#endif

    if (doPrint) {
#if MPI_POST_RECV
        CmiPrintf("%llu posted receives,  %llu unposted receives\n", CpvAccess(Cmi_posted_recv_total), CpvAccess(Cmi_unposted_recv_total));
#endif
    }
#endif

#if REPORT_COMM_METRICS
#if CMK_SMP
    CmiPrintf("Report comm metrics for node %d[%d-%d]: pumptime: %f, releasetime: %f, senttime: %f\n",
              CmiMyNode(), CmiNodeFirst(CmiMyNode()), CmiNodeFirst(CmiMyNode())+CmiMyNodeSize()-1,
              pumptime, releasetime, sendtime);
#else
    CmiPrintf("Report comm metrics for proc %d: pumptime: %f, releasetime: %f, senttime: %f\n",
              CmiMyPe(), pumptime, releasetime, sendtime);
#endif
#endif

#if ! CMK_AUTOBUILD
    signal(SIGINT, signal_int);
    MPI_Finalize();
#endif
    exit(0);
}

static int machine_exit_idx;
static void machine_exit(char *m) {
    EmergencyExit();
    /*printf("--> %d: machine_exit\n",CmiMyPe());*/
    fflush(stdout);
    CmiNodeBarrier();
    if (CmiMyRank() == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        /*printf("==> %d: passed barrier\n",CmiMyPe());*/
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        while (1) CmiYield();
    }
}

static void KillOnAllSigs(int sigNo) {
    static int already_in_signal_handler = 0;
    char *m;
    if (already_in_signal_handler) MPI_Abort(MPI_COMM_WORLD,1);
    already_in_signal_handler = 1;
#if CMK_CCS_AVAILABLE
    if (CpvAccess(cmiArgDebugFlag)) {
        CpdNotify(CPD_SIGNAL, sigNo);
        CpdFreeze();
    }
#endif
    CmiError("------------- Processor %d Exiting: Caught Signal ------------\n"
             "Signal: %d\n",CmiMyPe(),sigNo);
    CmiPrintStackTrace(1);

    m = CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(m, machine_exit_idx);
    CmiSyncBroadcastAndFree(CmiMsgHeaderSizeBytes, m);
    machine_exit(m);
}
/* ######End of functions related with exiting programs###### */


/* ######Beginning of functions related with starting programs###### */
static void registerMPITraceEvents() {
#if CMI_MPI_TRACE_USEREVENTS && CMK_TRACE_ENABLED && !CMK_TRACE_IN_CHARM
    traceRegisterUserEvent("MPI_Barrier", 10);
    traceRegisterUserEvent("MPI_Send", 20);
    traceRegisterUserEvent("MPI_Recv", 30);
    traceRegisterUserEvent("MPI_Isend", 40);
    traceRegisterUserEvent("MPI_Irecv", 50);
    traceRegisterUserEvent("MPI_Test", 60);
    traceRegisterUserEvent("MPI_Iprobe", 70);
#endif
}

#if MACHINE_DEBUG_LOG
FILE *debugLog = NULL;
#endif

static char *thread_level_tostring(int thread_level) {
#if CMK_MPI_INIT_THREAD
    switch (thread_level) {
    case MPI_THREAD_SINGLE:
        return "MPI_THREAD_SINGLE";
    case MPI_THREAD_FUNNELED:
        return "MPI_THREAD_FUNNELED";
    case MPI_THREAD_SERIALIZED:
        return "MPI_THREAD_SERIALIZED";
    case MPI_THREAD_MULTIPLE :
        return "MPI_THREAD_MULTIPLE ";
    default: {
        char *str = (char*)malloc(5);
        sprintf(str,"%d", thread_level);
        return str;
    }
    }
    return  "unknown";
#else
    char *str = (char*)malloc(5);
    sprintf(str,"%d", thread_level);
    return str;
#endif
}

/**
 *  Obtain the number of nodes, my node id, and consuming machine layer
 *  specific arguments
 */
static void MachineInitForMPI(int *argc, char ***argv, int *numNodes, int *myNodeID) {
    int n,i;
    int ver, subver;
    int provided;
    int thread_level;
    int myNID;
    int largc=*argc;
    char** largv=*argv;

#if MACHINE_DEBUG
    debugLog=NULL;
#endif
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
    _main(largc,largv);
#endif
#endif

#if CMK_MPI_INIT_THREAD
#if CMK_SMP
    thread_level = MPI_THREAD_FUNNELED;
#else
    thread_level = MPI_THREAD_SINGLE;
#endif
    MPI_Init_thread(argc, argv, thread_level, &provided);
    _thread_provided = provided;
#else
    MPI_Init(argc, argv);
    thread_level = 0;
    provided = -1;
#endif
    largc = *argc;
    largv = *argv;
    MPI_Comm_size(MPI_COMM_WORLD, numNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, myNodeID);

    myNID = *myNodeID;

    MPI_Get_version(&ver, &subver);
    if (myNID == 0) {
        printf("Charm++> Running on MPI version: %d.%d multi-thread support: %s (max supported: %s)\n", ver, subver, thread_level_tostring(thread_level), thread_level_tostring(provided));
    }

    idleblock = CmiGetArgFlag(largv, "+idleblocking");
    if (idleblock && _Cmi_mynode == 0) {
        printf("Charm++: Running in idle blocking mode.\n");
    }

    /* setup signal handlers */
    signal(SIGSEGV, KillOnAllSigs);
    signal(SIGFPE, KillOnAllSigs);
    signal(SIGILL, KillOnAllSigs);
    signal_int = signal(SIGINT, KillOnAllSigs);
    signal(SIGTERM, KillOnAllSigs);
    signal(SIGABRT, KillOnAllSigs);
#   if !defined(_WIN32) || defined(__CYGWIN__) /*UNIX-only signals*/
    signal(SIGQUIT, KillOnAllSigs);
    signal(SIGBUS, KillOnAllSigs);
#   endif /*UNIX*/

#if CMK_NO_OUTSTANDING_SENDS
    no_outstanding_sends=1;
#endif
    if (CmiGetArgFlag(largv,"+no_outstanding_sends")) {
        no_outstanding_sends = 1;
        if (myNID == 0)
            printf("Charm++: Will%s consume outstanding sends in scheduler loop\n",
                   no_outstanding_sends?"":" not");
    }

    request_max=MAX_QLEN;
    CmiGetArgInt(largv,"+requestmax",&request_max);
    /*printf("request max=%d\n", request_max);*/

#if MPI_POST_RECV
    CmiGetArgInt(largv, "+postRecvCnt", &MPI_POST_RECV_COUNT);
    CmiGetArgInt(largv, "+postRecvLowerSize", &MPI_POST_RECV_LOWERSIZE);
    CmiGetArgInt(largv, "+postRecvUpperSize", &MPI_POST_RECV_UPPERSIZE);
    if (MPI_POST_RECV_COUNT<=0) MPI_POST_RECV_COUNT=1;
    if (MPI_POST_RECV_LOWERSIZE>MPI_POST_RECV_UPPERSIZE) MPI_POST_RECV_UPPERSIZE = MPI_POST_RECV_LOWERSIZE;
    MPI_POST_RECV_SIZE = MPI_POST_RECV_UPPERSIZE;
    if (myNID==0) {
        printf("Charm++: using post-recv scheme with %d pre-posted recvs ranging from %d to %d (bytes)\n",
               MPI_POST_RECV_COUNT, MPI_POST_RECV_LOWERSIZE, MPI_POST_RECV_UPPERSIZE);
    }
#endif

#if CMI_DYNAMIC_EXERT_CAP
    CmiGetArgInt(largv, "+dynCapThreshold", &CMI_DYNAMIC_OUTGOING_THRESHOLD);
    CmiGetArgInt(largv, "+dynCapSend", &CMI_DYNAMIC_SEND_CAPSIZE);
    CmiGetArgInt(largv, "+dynCapRecv", &CMI_DYNAMIC_RECV_CAPSIZE);
    if (myNID==0) {
        printf("Charm++: using dynamic flow control with outgoing threshold %d, send cap %d, recv cap %d\n",
               CMI_DYNAMIC_OUTGOING_THRESHOLD, CMI_DYNAMIC_SEND_CAPSIZE, CMI_DYNAMIC_RECV_CAPSIZE);
    }
#endif

    /* checksum flag */
    if (CmiGetArgFlag(largv,"+checksum")) {
#if CMK_ERROR_CHECKING
        checksum_flag = 1;
        if (myNID == 0) CmiPrintf("Charm++: CheckSum checking enabled! \n");
#else
        if (myNID == 0) CmiPrintf("Charm++: +checksum ignored in optimized version! \n");
#endif
    }

    {
        int debug = CmiGetArgFlag(largv,"++debug");
        int debug_no_pause = CmiGetArgFlag(largv,"++debug-no-pause");
        if (debug || debug_no_pause) {  /*Pause so user has a chance to start and attach debugger*/
#if CMK_HAS_GETPID
            printf("CHARMDEBUG> Processor %d has PID %d\n",myNID,getpid());
            fflush(stdout);
            if (!debug_no_pause)
                sleep(15);
#else
            printf("++debug ignored.\n");
#endif
        }
    }

    procState = (ProcState *)malloc((_Cmi_mynodesize+1) * sizeof(ProcState));
    for (i=0; i<_Cmi_mynodesize+1; i++) {
#if MULTI_SENDQUEUE
        procState[i].sendMsgBuf = PCQueueCreate();
#endif
        procState[i].recvLock = CmiCreateLock();
    }
#if CMK_SMP
#if !MULTI_SENDQUEUE
    sendMsgBuf = PCQueueCreate();
    sendMsgBufLock = CmiCreateLock();
#endif
#endif
}

static void MachinePreCommonInitForMPI(int everReturn) {
#if MPI_POST_RECV
    int doInit = 1;
    int i;

#if CMK_SMP
    if (CmiMyRank() != CmiMyNodeSize()) doInit = 0;
#endif

    /* Currently, in mpi smp, the main thread will be the comm thread, so
     *	only the comm thread should post recvs. Cpvs, however, need to be
     * created on rank 0 (the ptrs to the actual cpv memory), while
     * other ranks are busy waiting for this to finish.	So cpv initialize
     * routines have to be called on every ranks, although they are only
     * useful on comm thread (whose rank is not zero) -Chao Mei
     */
    CpvInitialize(unsigned long long, Cmi_posted_recv_total);
    CpvInitialize(unsigned long long, Cmi_unposted_recv_total);
    CpvInitialize(MPI_Request*, CmiPostedRecvRequests);
    CpvInitialize(char*,CmiPostedRecvBuffers);

    if (doInit) {
        /* Post some extra recvs to help out with incoming messages */
        /* On some MPIs the messages are unexpected and thus slow */

        /* An array of request handles for posted recvs */
        CpvAccess(CmiPostedRecvRequests) = (MPI_Request*)malloc(sizeof(MPI_Request)*MPI_POST_RECV_COUNT);

        /* An array of buffers for posted recvs */
        CpvAccess(CmiPostedRecvBuffers) = (char*)malloc(MPI_POST_RECV_COUNT*MPI_POST_RECV_SIZE);

        /* Post Recvs */
        for (i=0; i<MPI_POST_RECV_COUNT; i++) {
            if (MPI_SUCCESS != MPI_Irecv(  &(CpvAccess(CmiPostedRecvBuffers)[i*MPI_POST_RECV_SIZE])	,
                                           MPI_POST_RECV_SIZE,
                                           MPI_BYTE,
                                           MPI_ANY_SOURCE,
                                           POST_RECV_TAG,
                                           MPI_COMM_WORLD,
                                           &(CpvAccess(CmiPostedRecvRequests)[i])  ))
                CmiAbort("MPI_Irecv failed\n");
        }
    }
#endif

}

static void MachinePostCommonInitForMPI(int everReturn) {
    CmiIdleState *s=CmiNotifyGetState();
    machine_exit_idx = CmiRegisterHandler((CmiHandler)machine_exit);

#if CMI_MPI_TRACE_USEREVENTS && CMK_TRACE_ENABLED && !CMK_TRACE_IN_CHARM
    CpvInitialize(double, projTraceStart);
    /* only PE 0 needs to care about registration (to generate sts file). */
    if (CmiMyPe() == 0) {
        registerMachineUserEventsFunction(&registerMPITraceEvents);
    }
#endif

#if CMK_SMP
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdleForMPI,NULL);
#endif

#if MACHINE_DEBUG_LOG
    if (CmiMyRank() == 0) {
        char ln[200];
        sprintf(ln,"debugLog.%d",CmiMyNode());
        debugLog=fopen(ln,"w");
    }
#endif
}
/* ######End of functions related with starting programs###### */

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message) {
    char *m;
    /* if CharmDebug is attached simply try to send a message to it */
#if CMK_CCS_AVAILABLE
    if (CpvAccess(cmiArgDebugFlag)) {
        CpdNotify(CPD_ABORT, message);
        CpdFreeze();
    }
#endif
    CmiError("------------- Processor %d Exiting: Called CmiAbort ------------\n"
             "Reason: %s\n",CmiMyPe(),message);
    /*  CmiError(message); */
    CmiPrintStackTrace(0);
    m = CmiAlloc(CmiMsgHeaderSizeBytes);
    CmiSetHandler(m, machine_exit_idx);
    CmiSyncBroadcastAndFree(CmiMsgHeaderSizeBytes, m);
    machine_exit(m);
    /* Program never reaches here */
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/**************************  TIMER FUNCTIONS **************************/
#if CMK_TIMER_USE_SPECIAL || CMK_TIMER_USE_XT3_DCLOCK

/* MPI calls are not threadsafe, even the timer on some machines */
static CmiNodeLock  timerLock = 0;
                                static int _absoluteTime = 0;
                                                           static double starttimer = 0;
                                                                                      static int _is_global = 0;

int CmiTimerIsSynchronized() {
    int  flag;
    void *v;

    /*  check if it using synchronized timer */
    if (MPI_SUCCESS != MPI_Attr_get(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &v, &flag))
        printf("MPI_WTIME_IS_GLOBAL not valid!\n");
    if (flag) {
        _is_global = *(int*)v;
        if (_is_global && CmiMyPe() == 0)
            printf("Charm++> MPI timer is synchronized\n");
    }
    return _is_global;
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
    _absoluteTime = CmiGetArgFlagDesc(argv,"+useAbsoluteTime", "Use system's absolute time as wallclock time.");
    if (_absoluteTime && CmiMyPe() == 0)
        printf("Charm++> absolute MPI timer is used\n");

    _is_global = CmiTimerIsSynchronized();

    if (_is_global) {
        if (CmiMyRank() == 0) {
            double minTimer;
#if CMK_TIMER_USE_XT3_DCLOCK
            starttimer = dclock();
#else
            starttimer = MPI_Wtime();
#endif

            MPI_Allreduce(&starttimer, &minTimer, 1, MPI_DOUBLE, MPI_MIN,
                          MPI_COMM_WORLD );
            starttimer = minTimer;
        }
    } else { /* we don't have a synchronous timer, set our own start time */
        CmiBarrier();
        CmiBarrier();
        CmiBarrier();
#if CMK_TIMER_USE_XT3_DCLOCK
        starttimer = dclock();
#else
        starttimer = MPI_Wtime();
#endif
    }

#if 0 && CMK_SMP && CMK_MPI_INIT_THREAD
    if (CmiMyRank()==0 && _thread_provided == MPI_THREAD_SINGLE)
        timerLock = CmiCreateLock();
#endif
    CmiNodeAllBarrier();          /* for smp */
}

/**
 * Since the timerLock is never created, and is
 * always NULL, then all the if-condition inside
 * the timer functions could be disabled right
 * now in the case of SMP. --Chao Mei
 */
double CmiTimer(void) {
    double t;
#if 0 && CMK_SMP
    if (timerLock) CmiLock(timerLock);
#endif

#if CMK_TIMER_USE_XT3_DCLOCK
    t = dclock();
#else
    t = MPI_Wtime();
#endif

#if 0 && CMK_SMP
    if (timerLock) CmiUnlock(timerLock);
#endif

    return _absoluteTime?t: (t-starttimer);
}

double CmiWallTimer(void) {
    double t;
#if 0 && CMK_SMP
    if (timerLock) CmiLock(timerLock);
#endif

#if CMK_TIMER_USE_XT3_DCLOCK
    t = dclock();
#else
    t = MPI_Wtime();
#endif

#if 0 && CMK_SMP
    if (timerLock) CmiUnlock(timerLock);
#endif

    return _absoluteTime? t: (t-starttimer);
}

double CmiCpuTimer(void) {
    double t;
#if 0 && CMK_SMP
    if (timerLock) CmiLock(timerLock);
#endif
#if CMK_TIMER_USE_XT3_DCLOCK
    t = dclock() - starttimer;
#else
    t = MPI_Wtime() - starttimer;
#endif
#if 0 && CMK_SMP
    if (timerLock) CmiUnlock(timerLock);
#endif
    return t;
}

#endif

/************Barrier Related Functions****************/
/* must be called on all ranks including comm thread in SMP */
int CmiBarrier() {
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

        if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD))
            CmiAbort("Timernit: MPI_Barrier failed!\n");

        /*END_EVENT(10);*/
    }
    CmiNodeAllBarrier();
    return 0;
}

/* CmiBarrierZero make sure node 0 is the last one exiting the barrier */
int CmiBarrierZero() {
    int i;
#if CMK_SMP
    if (CmiMyRank() == CmiMyNodeSize())
#else
    if (CmiMyRank() == 0)
#endif
    {
        char msg[1];
        MPI_Status sts;
        if (CmiMyNode() == 0)  {
            for (i=0; i<CmiNumNodes()-1; i++) {
                START_EVENT();

                if (MPI_SUCCESS != MPI_Recv(msg,1,MPI_BYTE,MPI_ANY_SOURCE,BARRIER_ZERO_TAG, MPI_COMM_WORLD,&sts))
                    CmiPrintf("MPI_Recv failed!\n");

                END_EVENT(30);
            }
        } else {
            START_EVENT();

            if (MPI_SUCCESS != MPI_Send((void *)msg,1,MPI_BYTE,0,BARRIER_ZERO_TAG,MPI_COMM_WORLD))
                printf("MPI_Send failed!\n");

            END_EVENT(20);
        }
    }
    CmiNodeAllBarrier();
    return 0;
}

/*@}*/

