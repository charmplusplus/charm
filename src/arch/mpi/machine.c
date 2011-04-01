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
static void sleep(int secs) {Sleep(1000*secs);}
#else
#include <unistd.h> /*For getpid()*/
#endif
#include <stdlib.h> /*For sleep()*/

#define MULTI_SENDQUEUE    0

#if defined(CMK_SHARED_VARS_POSIX_THREADS_SMP)
#define CMK_SMP 1
#endif

#define CMI_EXERT_SEND_CAP 0
#define CMI_EXERT_RECV_CAP 0

#if CMK_SMP
/* currently only considering the smp case */
#define CMI_DYNAMIC_EXERT_CAP 1
/* This macro defines the max number of msgs in the sender msg buffer 
 * that is allowed for recving operation to continue
 */
#define CMI_DYNAMIC_OUTGOING_THRESHOLD 10
#define CMI_DYNAMIC_SEND_CAPSIZE 10
#define CMI_DYNAMIC_RECV_CAPSIZE 10
/* initial values, -1 indiates there's no cap */
static int dynamicSendCap = -1;
static int dynamicRecvCap = -1;
#endif

#if CMI_EXERT_SEND_CAP
#define SEND_CAP 3
#endif

#if CMI_EXERT_RECV_CAP
#define RECV_CAP 2
#endif


#if CMK_SMP_TRACE_COMMTHREAD
#define CMI_MPI_TRACE_MOREDETAILED 0
#undef CMI_MPI_TRACE_USEREVENTS
#define CMI_MPI_TRACE_USEREVENTS 1
#endif

#define CMK_TRACE_COMMOVERHEAD 0
#if CMK_TRACE_COMMOVERHEAD
#undef CMI_MPI_TRACE_USEREVENTS
#define CMI_MPI_TRACE_USEREVENTS 1
#endif

#include "machine.h"

#include "pcqueue.h"

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))

#if CMK_BLUEGENEL
#define MAX_QLEN 8
#define NETWORK_PROGRESS_PERIOD_DEFAULT 16
#else
#define NETWORK_PROGRESS_PERIOD_DEFAULT 0
#define MAX_QLEN 200
#endif

#if CMI_MPI_TRACE_USEREVENTS && CMK_TRACE_ENABLED && ! CMK_TRACE_IN_CHARM
CpvStaticDeclare(double, projTraceStart);
# define  START_EVENT()  CpvAccess(projTraceStart) = CmiWallTimer();
# define  END_EVENT(x)   traceUserBracketEvent(x, CpvAccess(projTraceStart), CmiWallTimer());
#else
# define  START_EVENT()
# define  END_EVENT(x)
#endif

/*
    To reduce the buffer used in broadcast and distribute the load from
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of
  spanning tree broadcast algorithm.
    This will use the fourth short in message as an indicator of spanning tree
  root.
*/
#define CMK_BROADCAST_SPANNING_TREE    1
#define CMK_BROADCAST_HYPERCUBE        0

#define BROADCAST_SPANNING_FACTOR      4
/* The number of children used when a msg is broadcast inside a node */
#define BROADCAST_SPANNING_INTRA_FACTOR      8

#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_GET_CYCLE(msg)               ((CmiMsgHeaderBasic *)msg)->root

#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank
#define CMI_MAGIC(msg)			 ((CmiMsgHeaderBasic *)msg)->magic

/* FIXME: need a random number that everyone agrees ! */
#define CHARM_MAGIC_NUMBER		 126

#if CMK_ERROR_CHECKING
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

#if CMK_BROADCAST_SPANNING_TREE || CMK_BROADCAST_HYPERCUBE
#define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);
#else
#define CMI_SET_BROADCAST_ROOT(msg, root) 
#endif


/** 
    If MPI_POST_RECV is defined, we provide default values for size 
    and number of posted recieves. If MPI_POST_RECV_COUNT is set
    then a default value for MPI_POST_RECV_SIZE is used if not specified
    by the user.
*/
#ifdef MPI_POST_RECV
#define MPI_POST_RECV_COUNT 10
#undef MPI_POST_RECV
#endif
#if MPI_POST_RECV_COUNT > 0
#warning "Using MPI posted receives which have not yet been tested"
#ifndef MPI_POST_RECV_SIZE
#define MPI_POST_RECV_SIZE 200
#endif
/* #undef  MPI_POST_RECV_DEBUG  */
CpvDeclare(unsigned long long, Cmi_posted_recv_total);
CpvDeclare(unsigned long long, Cmi_unposted_recv_total);
CpvDeclare(MPI_Request*, CmiPostedRecvRequests); /* An array of request handles for posted recvs */
CpvDeclare(char*,CmiPostedRecvBuffers);
#endif

/*
 to avoid MPI's in order delivery, changing MPI Tag all the time
*/
#define TAG     1375

#if MPI_POST_RECV_COUNT > 0
#define POST_RECV_TAG TAG+1
#define BARRIER_ZERO_TAG TAG
#else
#define BARRIER_ZERO_TAG     1375
#endif

#include <signal.h>
void (*signal_int)(int);

/*
static int mpi_tag = TAG;
#define NEW_MPI_TAG	mpi_tag++; if (mpi_tag == MPI_TAG_UB) mpi_tag=TAG;
*/

static int        _thread_provided = -1;
int 		  _Cmi_numpes;
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_numnodes;  /* Total number of address spaces */
int               _Cmi_numpes;    /* Total number of processors */
static int        Cmi_nodestart; /* First processor in this address space */
CpvDeclare(void*, CmiLocalQueue);

/*Network progress utility variables. Period controls the rate at
  which the network poll is called */
CpvDeclare(unsigned , networkProgressCount);
int networkProgressPeriod;

int 		  idleblock = 0;

#define BLK_LEN  512

#if CMK_NODE_QUEUE_AVAILABLE
#define DGRAM_NODEMESSAGE   (0xFB)

#define NODE_BROADCAST_OTHERS (-1)
#define NODE_BROADCAST_ALL    (-2)
#endif

#if 0
static void **recdQueue_blk;
static unsigned int recdQueue_blk_len;
static unsigned int recdQueue_first;
static unsigned int recdQueue_len;
static void recdQueueInit(void);
static void recdQueueAddToBack(void *element);
static void *recdQueueRemoveFromFront(void);
#endif

static void ConverseRunPE(int everReturn);
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);

typedef struct msg_list {
     char *msg;
     struct msg_list *next;
     int size, destpe;

#if CMK_SMP_TRACE_COMMTHREAD
	int srcpe;
#endif	
	
     MPI_Request req;
} SMSG_LIST;

int MsgQueueLen=0;
static int request_max;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

static int Cmi_dim;

static int no_outstanding_sends=0; /*FLAG: consume outstanding Isends in scheduler loop*/

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

void CmiAbort(const char *message);
static void PerrorExit(const char *msg);

void SendSpanningChildren(int size, char *msg);
void SendHypercube(int size, char *msg);

static void PerrorExit(const char *msg)
{
  perror(msg);
  exit(1);
}

extern unsigned char computeCheckSum(unsigned char *data, int len);

/**************************  TIMER FUNCTIONS **************************/

#if CMK_TIMER_USE_SPECIAL || CMK_TIMER_USE_XT3_DCLOCK

/* MPI calls are not threadsafe, even the timer on some machines */
static CmiNodeLock  timerLock = 0;
static int _absoluteTime = 0;
static double starttimer = 0;
static int _is_global = 0;

int CmiTimerIsSynchronized()
{
  int  flag;
  void *v;

  /*  check if it using synchronized timer */
  if (MPI_SUCCESS != MPI_Attr_get(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &v, &flag))
    printf("MPI_WTIME_IS_GLOBAL not valid!\n");
  if (flag) {
    _is_global = *(int*)v;
    if (_is_global && CmiMyPe() == 0)
      printf("Charm++> MPI timer is synchronized!\n");
  }
  return _is_global;
}

int CmiTimerAbsolute()
{       
  return _absoluteTime;
}

double CmiStartTimer()
{
  return 0.0;
}

double CmiInitTime()
{
  return starttimer;
}

void CmiTimerInit(char **argv)
{
  _absoluteTime = CmiGetArgFlagDesc(argv,"+useAbsoluteTime", "Use system's absolute time as wallclock time.");

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
  }
  else {  /* we don't have a synchronous timer, set our own start time */
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
double CmiTimer(void)
{
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

double CmiWallTimer(void)
{
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

double CmiCpuTimer(void)
{
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

/* must be called on all ranks including comm thread in SMP */
int CmiBarrier()
{
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
int CmiBarrierZero()
{
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
    }
    else {
      START_EVENT();

      if (MPI_SUCCESS != MPI_Send((void *)msg,1,MPI_BYTE,0,BARRIER_ZERO_TAG,MPI_COMM_WORLD))
         printf("MPI_Send failed!\n");

      END_EVENT(20);
    }
  }
  CmiNodeAllBarrier();
  return 0;
}

typedef struct ProcState {
#if MULTI_SENDQUEUE
PCQueue      sendMsgBuf;       /* per processor message sending queue */
#endif
CmiNodeLock  recvLock;		    /* for cs->recv */
} ProcState;

static ProcState  *procState;

#if CMK_SMP

#if !MULTI_SENDQUEUE
static PCQueue sendMsgBuf;
static CmiNodeLock  sendMsgBufLock = NULL;        /* for sendMsgBuf */
#endif

#endif

/************************************************************
 *
 * Processor state structure
 *
 ************************************************************/

/* fake Cmi_charmrun_fd */
static int Cmi_charmrun_fd = 0;
#include "machine-smp.c"

CsvDeclare(CmiNodeState, NodeState);

#include "immediate.c"

#if CMK_SHARED_VARS_UNAVAILABLE
/************ non SMP **************/
static struct CmiStateStruct Cmi_state;
int _Cmi_mype;
int _Cmi_myrank;

void CmiMemLock() {}
void CmiMemUnlock() {}

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

void CmiYield(void) { sleep(0); }

static void CmiStartThreads(char **argv)
{
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  _Cmi_mype = Cmi_nodestart;
  _Cmi_myrank = 0;
}
#endif	/* non smp */

/*Add a message to this processor's receive queue, pe is a rank */
void CmiPushPE(int pe,void *msg)
{
  CmiState cs = CmiGetStateN(pe);
  MACHSTATE2(3,"Pushing message into rank %d's queue %p{",pe, cs->recv);
#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
/*
CmiPrintf("[node %d] Immediate Message hdl: %d rank: %d {{. \n", CmiMyNode(), CmiGetHandler(msg), pe);
    CmiHandleMessage(msg);
CmiPrintf("[node %d] Immediate Message done.}} \n", CmiMyNode());
*/
    /**(CmiUInt2 *)msg = pe;*/
    CMI_DEST_RANK(msg) = pe;
    CmiPushImmediateMsg(msg);
    return;
  }
#endif

#if CMK_SMP
  CmiLock(procState[pe].recvLock);
#endif
  PCQueuePush(cs->recv,msg);
#if CMK_SMP
  CmiUnlock(procState[pe].recvLock);
#endif
  CmiIdleLock_addMessage(&cs->idle);
  MACHSTATE1(3,"} Pushing message into rank %d's queue done",pe);
}

#if CMK_NODE_QUEUE_AVAILABLE
/*Add a message to this processor's receive queue */
static void CmiPushNode(void *msg)
{
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

#ifndef CmiMyPe
int CmiMyPe(void)
{
  return CmiGetState()->pe;
}
#endif

#ifndef CmiMyRank
int CmiMyRank(void)
{
  return CmiGetState()->rank;
}
#endif

#ifndef CmiNodeFirst
int CmiNodeFirst(int node) { return node*_Cmi_mynodesize; }
int CmiNodeSize(int node)  { return _Cmi_mynodesize; }
#endif

#ifndef CmiNodeOf
int CmiNodeOf(int pe)      { return (pe/_Cmi_mynodesize); }
int CmiRankOf(int pe)      { return pe%_Cmi_mynodesize; }
#endif

static size_t CmiAllAsyncMsgsSent(void)
{
   SMSG_LIST *msg_tmp = sent_msgs;
   MPI_Status sts;
   int done;

   while(msg_tmp!=0) {
    done = 0;
    if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts))
      CmiAbort("CmiAllAsyncMsgsSent: MPI_Test failed!\n");
    if(!done)
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
  if(msg_tmp) {
    done = 0;
    if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts))
      CmiAbort("CmiAsyncMsgSent: MPI_Test failed!\n");
    return ((done)?1:0);
  } else {
    return 1;
  }
}

void CmiReleaseCommHandle(CmiCommHandle c)
{
  return;
}

#if CMK_BLUEGENEL
extern void MPID_Progress_test();
#endif

void CmiReleaseSentMessages(void)
{
  SMSG_LIST *msg_tmp=sent_msgs;
  SMSG_LIST *prev=0;
  SMSG_LIST *temp;
  int done;
  MPI_Status sts;

#if CMK_BLUEGENEL
  MPID_Progress_test();
#endif

  MACHSTATE1(2,"CmiReleaseSentMessages begin on %d {", CmiMyPe());
  while(msg_tmp!=0) {
    done =0;
#if CMK_SMP_TRACE_COMMTHREAD || CMK_TRACE_COMMOVERHEAD
    double startT = CmiWallTimer();
#endif
    if(MPI_Test(&(msg_tmp->req), &done, &sts) != MPI_SUCCESS)
      CmiAbort("CmiReleaseSentMessages: MPI_Test failed!\n");
    if(done) {
      MACHSTATE2(3,"CmiReleaseSentMessages release one %d to %d", CmiMyPe(), msg_tmp->destpe);
      MsgQueueLen--;
      /* Release the message */
      temp = msg_tmp->next;
      if(prev==0)  /* first message */
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
    if(endT-startT>=0.001) traceUserSuppliedBracketedNote("MPI_Test: release a msg", 60, startT, endT);
    }
#endif
  }
  end_sent = prev;
  MACHSTATE(2,"} CmiReleaseSentMessages end");
}

int PumpMsgs(void)
{
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

	
  while(1) {
#if CMI_EXERT_RECV_CAP
	if(recvCnt==RECV_CAP) break;
#elif CMI_DYNAMIC_EXERT_CAP
	if(recvCnt == dynamicRecvCap) break;
#endif
	  
    /* First check posted recvs then do  probe unmatched outstanding messages */
#if MPI_POST_RECV_COUNT > 0 
    int completed_index=-1;
    if(MPI_SUCCESS != MPI_Testany(MPI_POST_RECV_COUNT, CpvAccess(CmiPostedRecvRequests), &completed_index, &flg, &sts))
        CmiAbort("PumpMsgs: MPI_Testany failed!\n");
    if(flg){
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
    }
    else {
        res = MPI_Iprobe(MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &flg, &sts);
        if(res != MPI_SUCCESS)
        CmiAbort("MPI_Iprobe failed\n");
        if(!flg) break;
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
    if(res != MPI_SUCCESS)
      CmiAbort("MPI_Iprobe failed\n");

    if(!flg) break;
#if CMK_SMP_TRACE_COMMTHREAD || CMK_TRACE_COMMOVERHEAD
    {
    double endT = CmiWallTimer();
    /* only trace the probe that last longer than 1ms */
    if(endT-startT>=0.001) traceUserSuppliedBracketedNote("MPI_Iprobe before a recv call", 70, startT, endT);
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
	
#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_BROADCAST_ROOT(msg))
      SendSpanningChildren(nbytes, msg);
#elif CMK_BROADCAST_HYPERCUBE
    if (CMI_BROADCAST_ROOT(msg))
      SendHypercube(nbytes, msg);
#endif
	
	/* In SMP mode, this push operation needs to be executed
     * after forwarding broadcast messages. If it is executed
     * earlier, then during the bcast msg forwarding period,	
	 * the msg could be already freed on the worker thread.
	 * As a result, the forwarded message could be wrong! 
	 * --Chao Mei
	 */
#if CMK_NODE_QUEUE_AVAILABLE
    if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
      CmiPushNode(msg);
    else
#endif
	CmiPushPE(CMI_DEST_RANK(msg), msg);	
	
#if CMI_EXERT_RECV_CAP || CMI_DYNAMIC_EXERT_CAP
	recvCnt++;
	/* check sendMsgBuf  to get the  number of messages that have not been sent */
	/* MsgQueueLen indicates the number of messages that have not been released by MPI */
	if(PCQueueLength(sendMsgBuf) > CMI_DYNAMIC_OUTGOING_THRESHOLD){
		dynamicRecvCap = CMI_DYNAMIC_RECV_CAPSIZE;
	}
#endif	
	
  }

  
#if CMK_IMMEDIATE_MSG && !CMK_SMP
  CmiHandleImmediate();
#endif
  
  MACHSTATE(2,"} PumpMsgs end ");
  return recd;
}

/* blocking version */
static void PumpMsgsBlocking(void)
{
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


#if MPI_POST_RECV_COUNT > 0
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

#if CMK_BROADCAST_SPANNING_TREE
   if (CMI_BROADCAST_ROOT(msg))
      SendSpanningChildren(nbytes, msg);
#elif CMK_BROADCAST_HYPERCUBE
   if (CMI_BROADCAST_ROOT(msg))
      SendHypercube(nbytes, msg);
#endif
  
	/* In SMP mode, this push operation needs to be executed
     * after forwarding broadcast messages. If it is executed
     * earlier, then during the bcast msg forwarding period,	
	 * the msg could be already freed on the worker thread.
	 * As a result, the forwarded message could be wrong! 
	 * --Chao Mei
	 */  
#if CMK_NODE_QUEUE_AVAILABLE
   if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
      CmiPushNode(msg);
   else
#endif
      CmiPushPE(CMI_DEST_RANK(msg), msg);
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

#if CMK_SMP

static int inexit = 0;
static CmiNodeLock  exitLock = 0;

static int MsgQueueEmpty()
{
  int i;
#if MULTI_SENDQUEUE
  for (i=0; i<_Cmi_mynodesize; i++)
    if (!PCQueueEmpty(procState[i].sendMsgBuf)) return 0;
#else
  return PCQueueEmpty(sendMsgBuf);
#endif
  return 1;
}

static int SendMsgBuf();

/* test if all processors recv queues are empty */
static int RecvQueueEmpty()
{
  int i;
  for (i=0; i<_Cmi_mynodesize; i++) {
    CmiState cs=CmiGetStateN(i);
    if (!PCQueueEmpty(cs->recv)) return 0;
  }
  return 1;
}

/**
CommunicationServer calls MPI to send messages in the queues and probe message from network.
*/

#define REPORT_COMM_METRICS 0
#if REPORT_COMM_METRICS
static double pumptime = 0.0;
static double releasetime = 0.0;
static double sendtime = 0.0;
#endif

static void CommunicationServer(int sleepTime)
{
  int static count=0;
/*
  count ++;
  if (count % 10000000==0) MACHSTATE(3, "Entering CommunicationServer {");
*/
#if REPORT_COMM_METRICS
  double t1, t2, t3, t4;
  t1 = CmiWallTimer();
#endif
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
/*
  if (count % 10000000==0) MACHSTATE(3, "} Exiting CommunicationServer.");
*/
  if (inexit == CmiMyNodeSize()) {
    MACHSTATE(2, "CommunicationServer exiting {");
#if 0
    while(!MsgQueueEmpty() || !CmiAllAsyncMsgsSent() || !RecvQueueEmpty()) {
#endif
    while(!MsgQueueEmpty() || !CmiAllAsyncMsgsSent()) {
      CmiReleaseSentMessages();
      SendMsgBuf();
      PumpMsgs();
    }
    MACHSTATE(2, "CommunicationServer barrier begin {");

    START_EVENT();

    if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD))
      CmiAbort("ConverseExit: MPI_Barrier failed!\n");

    END_EVENT(10);

    MACHSTATE(2, "} CommunicationServer barrier end");
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyNode() == 0){
      CmiPrintf("End of program\n");
    }
#endif
    MACHSTATE(2, "} CommunicationServer EXIT");

    ConverseCommonExit();   
#if REPORT_COMM_METRICS
    CmiPrintf("Report comm metrics from node %d[%d-%d]: pumptime: %f, releasetime: %f, senttime: %f\n", CmiMyNode(), CmiNodeFirst(CmiMyNode()), CmiNodeFirst(CmiMyNode())+CmiMyNodeSize()-1, pumptime, releasetime, sendtime);
#endif

#if ! CMK_AUTOBUILD
    signal(SIGINT, signal_int);
    MPI_Finalize();
    #endif
    exit(0);
  }
}

#endif

static void CommunicationServerThread(int sleepTime)
{
#if CMK_SMP
  CommunicationServer(sleepTime);
#endif
#if CMK_IMMEDIATE_MSG
  CmiHandleImmediate();
#endif
}

#if CMK_NODE_QUEUE_AVAILABLE
char *CmiGetNonLocalNodeQ(void)
{
  CmiState cs = CmiGetState();
  char *result = 0;
  CmiIdleLock_checkMessage(&cs->idle);
/*  if(!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {  */
    MACHSTATE1(3,"CmiGetNonLocalNodeQ begin %d {", CmiMyPe());
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
/*  }  */

  return result;
}
#endif

void *CmiGetNonLocal(void)
{
  static int count=0;
  CmiState cs = CmiGetState();
  void *msg;

#if ! CMK_SMP
  if (CmiNumPes() == 1) return NULL;
#endif

  CmiIdleLock_checkMessage(&cs->idle);
  /* although it seems that lock is not needed, I found it crashes very often
     on mpi-smp without lock */

#if ! CMK_SMP
  CmiReleaseSentMessages();
  PumpMsgs();
#endif

  /* CmiLock(procState[cs->rank].recvLock); */
  msg =  PCQueuePop(cs->recv);
  /* CmiUnlock(procState[cs->rank].recvLock); */

/*
  if (msg) {
    MACHSTATE2(3,"CmiGetNonLocal done on pe %d for queue %p", CmiMyPe(), cs->recv); }
  else {
    count++;
    if (count%1000000==0) MACHSTATE2(3,"CmiGetNonLocal empty on pe %d for queue %p", CmiMyPe(), cs->recv);
  }
*/
#if ! CMK_SMP
  if (no_outstanding_sends) {
    while (MsgQueueLen>0) {
      CmiReleaseSentMessages();
      PumpMsgs();
    }
  }

  if(!msg) {
    CmiReleaseSentMessages();
    if (PumpMsgs())
      return  PCQueuePop(cs->recv);
    else
      return 0;
  }
#endif

  return msg;
}

/* called in non-smp mode */
void CmiNotifyIdle(void)
{
  CmiReleaseSentMessages();
  if (!PumpMsgs() && idleblock) PumpMsgsBlocking();
}


/********************************************************
    The call to probe immediate messages has been renamed to
    CmiMachineProgressImpl
******************************************************/
/* user call to handle immediate message, only useful in non SMP version
   using polling method to schedule message.
*/
/*
#if CMK_IMMEDIATE_MSG
void CmiProbeImmediateMsg()
{
#if !CMK_SMP
  PumpMsgs();
  CmiHandleImmediate();
#endif
}
#endif
*/

/* Network progress function is used to poll the network when for
   messages. This flushes receive buffers on some  implementations*/
#if CMK_MACHINE_PROGRESS_DEFINED
void CmiMachineProgressImpl()
{
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

/********************* MESSAGE SEND FUNCTIONS ******************/

CmiCommHandle CmiAsyncSendFn_(int destPE, int size, char *msg);

static void CmiSendSelf(char *msg)
{
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
      /* CmiBecomeNonImmediate(msg); */
      CmiPushImmediateMsg(msg);
      CmiHandleImmediate();
      return;
    }
#endif
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}

void CmiSyncSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);

  CMI_SET_BROADCAST_ROOT(dupmsg, 0);

  if (cs->pe==destPE) {
    CmiSendSelf(dupmsg);
  }
  else
    CmiAsyncSendFn_(destPE, size, dupmsg);
}

#if CMK_SMP

/* called by communication thread in SMP */
static int SendMsgBuf()
{
  SMSG_LIST *msg_tmp;
  char *msg;
  int node, rank, size;
  int i;
  int sent = 0;

#if CMI_EXERT_SEND_CAP || CMI_DYNAMIC_EXERT_CAP
	int sentCnt = 0;
#endif	
	
  MACHSTATE(2,"SendMsgBuf begin {");
#if MULTI_SENDQUEUE
  for (i=0; i<_Cmi_mynodesize=1; i++)  /* subtle: including comm thread */
  {
    if (!PCQueueEmpty(procState[i].sendMsgBuf))
    {
      msg_tmp = (SMSG_LIST *)PCQueuePop(procState[i].sendMsgBuf);
#else
    /* single message sending queue */
    /* CmiLock(sendMsgBufLock); */
    msg_tmp = (SMSG_LIST *)PCQueuePop(sendMsgBuf);
    /* CmiUnlock(sendMsgBufLock); */
    while (NULL != msg_tmp)
    {
#endif
      node = msg_tmp->destpe;
      size = msg_tmp->size;
      msg = msg_tmp->msg;
      msg_tmp->next = 0;
		
#if !CMI_DYNAMIC_EXERT_CAP && !CMI_EXERT_SEND_CAP
      while (MsgQueueLen > request_max) {
		CmiReleaseSentMessages();
		PumpMsgs();
      }
#endif
	  
      MACHSTATE2(3,"MPI_send to node %d rank: %d{", node, CMI_DEST_RANK(msg));
#if CMK_ERROR_CHECKING
      CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
#endif
      CMI_SET_CHECKSUM(msg, size);

#if MPI_POST_RECV_COUNT > 0
        if(size <= MPI_POST_RECV_SIZE){

          START_EVENT();
          if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,POST_RECV_TAG,MPI_COMM_WORLD,&(msg_tmp->req)))
                CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");

          STOP_EVENT(40);
        }
        else {
            START_EVENT();
            if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(msg_tmp->req)))
                CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
            STOP_EVENT(40);
        }
#else
        START_EVENT();
        if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(msg_tmp->req)))
            CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
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
	sprintf(tmp, "MPI_Isend: from proc %d to proc %d", msg_tmp->srcpe, CmiNodeFirst(node)+CMI_DEST_RANK(msg));
	traceUserSuppliedBracketedNote(tmp, 40, CpvAccess(projTraceStart), CmiWallTimer());
	#endif
#endif
		
		
      MACHSTATE(3,"}MPI_send end");
      MsgQueueLen++;
      if(sent_msgs==0)
        sent_msgs = msg_tmp;
      else
        end_sent->next = msg_tmp;
      end_sent = msg_tmp;
      sent=1;
	  
#if CMI_EXERT_SEND_CAP	  
	  if(++sentCnt == SEND_CAP) break;
#elif CMI_DYNAMIC_EXERT_CAP
	  if(++sentCnt == dynamicSendCap) break;
	  if(MsgQueueLen > CMI_DYNAMIC_OUTGOING_THRESHOLD)
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

void EnqueueMsg(void *m, int size, int node)
{
  SMSG_LIST *msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  MACHSTATE1(3,"EnqueueMsg to node %d {{ ", node);
  msg_tmp->msg = m;
  msg_tmp->size = size;
  msg_tmp->destpe = node;
	
#if CMK_SMP_TRACE_COMMTHREAD
	msg_tmp->srcpe = CmiMyPe();
#endif	

#if MULTI_SENDQUEUE
  PCQueuePush(procState[CmiMyRank()].sendMsgBuf,(char *)msg_tmp);
#else
  CmiLock(sendMsgBufLock);
  PCQueuePush(sendMsgBuf,(char *)msg_tmp);
  CmiUnlock(sendMsgBufLock);
#endif
	
  MACHSTATE3(3,"}} EnqueueMsg to %d finish with queue %p len: %d", node, sendMsgBuf, PCQueueLength(sendMsgBuf));
}

#endif

CmiCommHandle CmiAsyncSendFn_(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  SMSG_LIST *msg_tmp;
  CmiUInt2  rank, node;

  if(destPE == cs->pe) {
    char *dupmsg = (char *) CmiAlloc(size);
    memcpy(dupmsg, msg, size);
    CmiSendSelf(dupmsg);
    return 0;
  }
  CQdCreate(CpvAccess(cQdState), 1);
#if CMK_SMP
  node = CmiNodeOf(destPE);
  rank = CmiRankOf(destPE);
  if (node == CmiMyNode())  {
    CmiPushPE(rank, msg);
    return 0;
  }
  CMI_DEST_RANK(msg) = rank;
  EnqueueMsg(msg, size, node);
  return 0;
#else
  /* non smp */
  CMI_DEST_RANK(msg) = 0;	/* rank is always 0 */
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  while (MsgQueueLen > request_max) {
	/*printf("Waiting for %d messages to be sent\n", MsgQueueLen);*/
	CmiReleaseSentMessages();
	PumpMsgs();
  }
#if CMK_ERROR_CHECKING
  CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
#endif
  CMI_SET_CHECKSUM(msg, size);

#if MPI_POST_RECV_COUNT > 0
        if(size <= MPI_POST_RECV_SIZE){

          START_EVENT();
          if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,destPE,POST_RECV_TAG,MPI_COMM_WORLD,&(msg_tmp->req)))
                CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
          END_EVENT(40);
        }
        else {
          START_EVENT();
          if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,destPE,TAG,MPI_COMM_WORLD,&(msg_tmp->req)))
                CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
          END_EVENT(40);
        }
#else
  START_EVENT();
  if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,destPE,TAG,MPI_COMM_WORLD,&(msg_tmp->req)))
    CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
  /*END_EVENT(40);*/
  #if CMK_TRACE_COMMOVERHEAD
	char tmp[64];
	sprintf(tmp, "MPI_Isend: from proc %d to proc %d", CmiMyPe(), destPE);
	traceUserSuppliedBracketedNote(tmp, 40, CpvAccess(projTraceStart), CmiWallTimer());
  #endif
#endif

  MsgQueueLen++;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
  return (CmiCommHandle) &(msg_tmp->req);
#endif              /* non-smp */
}

CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  CMI_SET_BROADCAST_ROOT(msg, 0);
  CmiAsyncSendFn_(destPE, size, msg);
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  CMI_SET_BROADCAST_ROOT(msg, 0);

  if (cs->pe==destPE) {
    CmiSendSelf(msg);
  } else {
    CmiAsyncSendFn_(destPE, size, msg);
  }
}

/*********************** BROADCAST FUNCTIONS **********************/

/* same as CmiSyncSendFn, but don't set broadcast root in msg header */
void CmiSyncSendFn1(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);
  if (cs->pe==destPE)
    CmiSendSelf(dupmsg);
  else
    CmiAsyncSendFn_(destPE, size, dupmsg);
}

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg)
{
  CmiState cs = CmiGetState();
  int startpe = CMI_BROADCAST_ROOT(msg)-1;
  int startnode = CmiNodeOf(startpe);
  int i, exceptRank;
	
   /* first send msgs to other nodes */
  CmiAssert(startnode >=0 &&  startnode<CmiNumNodes());
  for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
    int nd = CmiMyNode()-startnode;
    if (nd<0) nd+=CmiNumNodes();
    nd = BROADCAST_SPANNING_FACTOR*nd + i;
    if (nd > CmiNumNodes() - 1) break;
    nd += startnode;
    nd = nd%CmiNumNodes();
    CmiAssert(nd>=0 && nd!=CmiMyNode());	
	#if CMK_SMP
	/* always send to the first rank of other nodes */
	char *newmsg = CmiCopyMsg(msg, size);
	CMI_DEST_RANK(newmsg) = 0;
    EnqueueMsg(newmsg, size, nd);
	#else
	CmiSyncSendFn1(nd, size, msg);
	#endif
  }
#if CMK_SMP  
   /* second send msgs to my peers on this node */
  /* FIXME: now it's just a flat p2p send!! When node size is large,
   * it should also be sent in a tree
   */
   exceptRank = CMI_DEST_RANK(msg);
   for(i=0; i<exceptRank; i++){
	   CmiPushPE(i, CmiCopyMsg(msg, size));
   }
   for(i=exceptRank+1; i<CmiMyNodeSize(); i++){
	   CmiPushPE(i, CmiCopyMsg(msg, size));
   }
#endif
}

#include <math.h>

/* send msg along the hypercube in broadcast. (Sameer) */
void SendHypercube(int size, char *msg)
{
  CmiState cs = CmiGetState();
  int startpe = CMI_BROADCAST_ROOT(msg)-1;
  int startnode = CmiNodeOf(startpe);
  int i, exceptRank, cnt, tmp, relPE;
  int dims=0;

  /* dims = ceil(log2(CmiNumNodes)) except when #nodes is 1*/
  tmp = CmiNumNodes()-1;
  while(tmp>0){
	  dims++;
	  tmp = tmp >> 1;
  }
  if(CmiNumNodes()==1) dims=1;
  
   /* first send msgs to other nodes */  
  relPE = CmiMyNode()-startnode;
  if(relPE < 0) relPE += CmiNumNodes();
  cnt=0;
  tmp = relPE;
  /* count how many zeros (in binary format) relPE has */
  for(i=0; i<dims; i++, cnt++){
    if(tmp & 1 == 1) break;
    tmp = tmp >> 1;
  }
  
  /*CmiPrintf("ND[%d]: SendHypercube with spe=%d, snd=%d, relpe=%d, cnt=%d\n", CmiMyNode(), startpe, startnode, relPE, cnt);*/
  for (i = cnt-1; i >= 0; i--) {
    int nd = relPE + (1 << i);
	if(nd >= CmiNumNodes()) continue;
	nd = (nd+startnode)%CmiNumNodes();
	/*CmiPrintf("ND[%d]: send to node %d\n", CmiMyNode(), nd);*/
#if CMK_SMP
    /* always send to the first rank of other nodes */
    char *newmsg = CmiCopyMsg(msg, size);
    CMI_DEST_RANK(newmsg) = 0;
    EnqueueMsg(newmsg, size, nd);
#else
	CmiSyncSendFn1(nd, size, msg);
#endif
  }
  
#if CMK_SMP
   /* second send msgs to my peers on this node */
   /* FIXME: now it's just a flat p2p send!! When node size is large,
    * it should also be sent in a tree
    */
   exceptRank = CMI_DEST_RANK(msg);
   for(i=0; i<exceptRank; i++){
	   CmiPushPE(i, CmiCopyMsg(msg, size));
   }
   for(i=exceptRank+1; i<CmiMyNodeSize(); i++){
	   CmiPushPE(i, CmiCopyMsg(msg, size));
   }
#endif
}

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
  CmiState cs = CmiGetState();

#if CMK_SMP	
  /* record the rank to avoid re-sending the msg in SendSpanningChildren */
  CMI_DEST_RANK(msg) = CmiMyRank();
#endif
	
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);

#elif CMK_BROADCAST_HYPERCUBE
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendHypercube(size, msg);

#else
  int i;

  for ( i=cs->pe+1; i<_Cmi_numpes; i++ )
    CmiSyncSendFn(i, size,msg) ;
  for ( i=0; i<cs->pe; i++ )
    CmiSyncSendFn(i, size,msg) ;
#endif

  /*CmiPrintf("In  SyncBroadcast broadcast\n");*/
}


/*  FIXME: luckily async is never used  G. Zheng */
CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg)
{
  CmiState cs = CmiGetState();
  int i ;

  for ( i=cs->pe+1; i<_Cmi_numpes; i++ )
    CmiAsyncSendFn(i,size,msg) ;
  for ( i=0; i<cs->pe; i++ )
    CmiAsyncSendFn(i,size,msg) ;

  /*CmiPrintf("In  AsyncBroadcast broadcast\n");*/
  CmiAbort("CmiAsyncBroadcastFn should never be called");
  return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastFn(int size, char *msg)
{
   CmiSyncBroadcastFn(size,msg);
   CmiFree(msg);
}

void CmiSyncBroadcastAllFn(int size, char *msg)        /* All including me */
{

#if CMK_SMP	
  /* record the rank to avoid re-sending the msg in SendSpanningChildren */
  CMI_DEST_RANK(msg) = CmiMyRank();
#endif

#if CMK_BROADCAST_SPANNING_TREE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);

#elif CMK_BROADCAST_HYPERCUBE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendHypercube(size, msg);

#else
    int i ;

  for ( i=0; i<_Cmi_numpes; i++ )
    CmiSyncSendFn(i,size,msg) ;
#endif

  /*CmiPrintf("In  SyncBroadcastAll broadcast\n");*/
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)
{
  int i ;

  for ( i=1; i<_Cmi_numpes; i++ )
    CmiAsyncSendFn(i,size,msg) ;

  CmiAbort("In  AsyncBroadcastAll broadcast\n");

  return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastAllFn(int size, char *msg)  /* All including me */
{
#if CMK_SMP	
  /* record the rank to avoid re-sending the msg in SendSpanningChildren */
  CMI_DEST_RANK(msg) = CmiMyRank();
#endif

#if CMK_BROADCAST_SPANNING_TREE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);

#elif CMK_BROADCAST_HYPERCUBE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendHypercube(size, msg);

#else
  int i ;

  for ( i=0; i<_Cmi_numpes; i++ )
    CmiSyncSendFn(i,size,msg) ;
#endif
  CmiFree(msg) ;
  /*CmiPrintf("In FreeBroadcastAll broadcast\n");*/
}

#if CMK_NODE_QUEUE_AVAILABLE

static void CmiSendNodeSelf(char *msg)
{
#if CMK_IMMEDIATE_MSG
#if 0
    if (CmiIsImmediate(msg) && !_immRunning) {
      /*CmiHandleImmediateMessage(msg); */
      CmiPushImmediateMsg(msg);
      CmiHandleImmediate();
      return;
    }
#endif
    if (CmiIsImmediate(msg))
    {
      CmiPushImmediateMsg(msg);
      if (!_immRunning) CmiHandleImmediate();
      return;
    }
#endif
    CQdCreate(CpvAccess(cQdState), 1);
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv, msg);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
}

CmiCommHandle CmiAsyncNodeSendFn(int dstNode, int size, char *msg)
{
  int i;
  SMSG_LIST *msg_tmp;
  char *dupmsg;

  CMI_SET_BROADCAST_ROOT(msg, 0);
  CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
  switch (dstNode) {
  case NODE_BROADCAST_ALL:
    CmiSendNodeSelf((char *)CmiCopyMsg(msg,size));
  case NODE_BROADCAST_OTHERS:
    CQdCreate(CpvAccess(cQdState), _Cmi_numnodes-1);
    for (i=0; i<_Cmi_numnodes; i++)
      if (i!=_Cmi_mynode) {
        EnqueueMsg((char *)CmiCopyMsg(msg,size), size, i);
      }
    break;
  default:
    dupmsg = (char *)CmiCopyMsg(msg,size);
    if(dstNode == _Cmi_mynode) {
      CmiSendNodeSelf(dupmsg);
    }
    else {
      CQdCreate(CpvAccess(cQdState), 1);
      EnqueueMsg(dupmsg, size, dstNode);
    }
  }
  return 0;
}

void CmiSyncNodeSendFn(int p, int s, char *m)
{
  CmiAsyncNodeSendFn(p, s, m);
}

/* need */
void CmiFreeNodeSendFn(int p, int s, char *m)
{
  CmiAsyncNodeSendFn(p, s, m);
  CmiFree(m);
}

/* need */
void CmiSyncNodeBroadcastFn(int s, char *m)
{
  CmiAsyncNodeSendFn(NODE_BROADCAST_OTHERS, s, m);
}

CmiCommHandle CmiAsyncNodeBroadcastFn(int s, char *m)
{
}

/* need */
void CmiFreeNodeBroadcastFn(int s, char *m)
{
  CmiAsyncNodeSendFn(NODE_BROADCAST_OTHERS, s, m);
  CmiFree(m);
}

void CmiSyncNodeBroadcastAllFn(int s, char *m)
{
  CmiAsyncNodeSendFn(NODE_BROADCAST_ALL, s, m);
}

CmiCommHandle CmiAsyncNodeBroadcastAllFn(int s, char *m)
{
  CmiAsyncNodeSendFn(NODE_BROADCAST_ALL, s, m);
}

/* need */
void CmiFreeNodeBroadcastAllFn(int s, char *m)
{
  CmiAsyncNodeSendFn(NODE_BROADCAST_ALL, s, m);
  CmiFree(m);
}
#endif

/************************** MAIN ***********************************/
#define MPI_REQUEST_MAX 16      /* 1024*10 */

void ConverseExit(void)
{
#if ! CMK_SMP
  while(!CmiAllAsyncMsgsSent()) {
    PumpMsgs();
    CmiReleaseSentMessages();
  }
  if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD))
    CmiAbort("ConverseExit: MPI_Barrier failed!\n");

  ConverseCommonExit();
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
  if (CmiMyPe() == 0){
    CmiPrintf("End of program\n");
#if MPI_POST_RECV_COUNT > 0
    CmiPrintf("%llu posted receives,  %llu unposted receives\n", CpvAccess(Cmi_posted_recv_total), CpvAccess(Cmi_unposted_recv_total));
#endif
}
#endif
#if ! CMK_AUTOBUILD
  signal(SIGINT, signal_int);
  MPI_Finalize();
#endif
  exit(0);

#else
    /* SMP version, communication thread will exit */
  ConverseCommonExit();
  /* atomic increment */
  CmiLock(exitLock);
  inexit++;
  CmiUnlock(exitLock);
  while (1) CmiYield();
#endif
}

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


static char     **Cmi_argv;
static char     **Cmi_argvcopy;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */

typedef struct {
  int sleepMs; /*Milliseconds to sleep while idle*/
  int nIdles; /*Number of times we've been idle in a row*/
  CmiState cs; /*Machine state*/
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void)
{
  CmiIdleState *s=(CmiIdleState *)malloc(sizeof(CmiIdleState));
  s->sleepMs=0;
  s->nIdles=0;
  s->cs=CmiGetState();
  return s;
}

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  s->sleepMs=0;
  s->nIdles=0;
}

static void CmiNotifyStillIdle(CmiIdleState *s)
{
#if ! CMK_SMP
  CmiReleaseSentMessages();
  PumpMsgs();
#else
/*  CmiYield();  */
#endif

#if 1
  {
  int nSpins=20; /*Number of times to spin before sleeping*/
  MACHSTATE1(2,"still idle (%d) begin {",CmiMyPe())
  s->nIdles++;
  if (s->nIdles>nSpins) { /*Start giving some time back to the OS*/
    s->sleepMs+=2;
    if (s->sleepMs>10) s->sleepMs=10;
  }
  /*Comm. thread will listen on sockets-- just sleep*/
  if (s->sleepMs>0) {
    MACHSTATE1(2,"idle lock(%d) {",CmiMyPe())
    CmiIdleLock_sleep(&s->cs->idle,s->sleepMs);
    MACHSTATE1(2,"} idle lock(%d)",CmiMyPe())
  }
  MACHSTATE1(2,"still idle (%d) end {",CmiMyPe())
  }
#endif
}

#if MACHINE_DEBUG_LOG
FILE *debugLog = NULL;
#endif

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

static void ConverseRunPE(int everReturn)
{
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

  ConverseCommonInit(CmiMyArgv);
  machine_exit_idx = CmiRegisterHandler((CmiHandler)machine_exit);

#if CMI_MPI_TRACE_USEREVENTS && CMK_TRACE_ENABLED && !CMK_TRACE_IN_CHARM
  CpvInitialize(double, projTraceStart);
  /* only PE 0 needs to care about registration (to generate sts file). */
  if (CmiMyPe() == 0) {
    registerMachineUserEventsFunction(&registerMPITraceEvents);
  }
#endif

  /* initialize the network progress counter*/
  /* Network progress function is used to poll the network when for
     messages. This flushes receive buffers on some  implementations*/
  CpvInitialize(int , networkProgressCount);
  CpvAccess(networkProgressCount) = 0;

#if CMK_SMP
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
#endif

#if MACHINE_DEBUG_LOG
  if (CmiMyRank() == 0) {
    char ln[200];
    sprintf(ln,"debugLog.%d",CmiMyNode());
    debugLog=fopen(ln,"w");
  }
#endif

  /* Converse initialization finishes, immediate messages can be processed.
     node barrier previously should take care of the node synchronization */
  _immediateReady = 1;

  /* communication thread */
  if (CmiMyRank() == CmiMyNodeSize()) {
    Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
    while (1) CommunicationServerThread(5);
  }
  else {  /* worker thread */
  if (!everReturn) {
    Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
    if (Cmi_usrsched==0) CsdScheduler(-1);
    ConverseExit();
  }
  }
}

static char *thread_level_tostring(int thread_level)
{
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

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int n,i;
  int ver, subver;
  int provided;
  int thread_level;

#if MACHINE_DEBUG
  debugLog=NULL;
#endif
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif

#if CMK_MPI_INIT_THREAD
#if CMK_SMP
  thread_level = MPI_THREAD_FUNNELED;
#else
  thread_level = MPI_THREAD_SINGLE;
#endif
  MPI_Init_thread(&argc, &argv, thread_level, &provided);
  _thread_provided = provided;
#else
  MPI_Init(&argc, &argv);
  thread_level = 0;
  provided = -1;
#endif
  MPI_Comm_size(MPI_COMM_WORLD, &_Cmi_numnodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &_Cmi_mynode);

  MPI_Get_version(&ver, &subver);
  if (_Cmi_mynode == 0) {
    printf("Charm++> Running on MPI version: %d.%d multi-thread support: %s (max supported: %s)\n", ver, subver, thread_level_tostring(thread_level), thread_level_tostring(provided));
  }

  /* processor per node */
  _Cmi_mynodesize = 1;
  if (!CmiGetArgInt(argv,"+ppn", &_Cmi_mynodesize))
    CmiGetArgInt(argv,"++ppn", &_Cmi_mynodesize);
#if ! CMK_SMP
  if (_Cmi_mynodesize > 1 && _Cmi_mynode == 0)
    CmiAbort("+ppn cannot be used in non SMP version!\n");
#endif
  idleblock = CmiGetArgFlag(argv, "+idleblocking");
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
/*#     if CMK_HANDLE_SIGUSR
  signal(SIGUSR1, HandleUserSignals);
  signal(SIGUSR2, HandleUserSignals);
#     endif*/
#   endif /*UNIX*/
  
#if CMK_NO_OUTSTANDING_SENDS
  no_outstanding_sends=1;
#endif
  if (CmiGetArgFlag(argv,"+no_outstanding_sends")) {
    no_outstanding_sends = 1;
    if (_Cmi_mynode == 0)
      printf("Charm++: Will%s consume outstanding sends in scheduler loop\n",
     	no_outstanding_sends?"":" not");
  }
  _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
  Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
  Cmi_argvcopy = CmiCopyArgs(argv);
  Cmi_argv = argv; Cmi_startfn = fn; Cmi_usrsched = usched;
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=_Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;
 /* CmiSpanTreeInit();*/
  request_max=MAX_QLEN;
  CmiGetArgInt(argv,"+requestmax",&request_max);
  /*printf("request max=%d\n", request_max);*/

  /* checksum flag */
  if (CmiGetArgFlag(argv,"+checksum")) {
#if !CMK_OPTIMIZE
    checksum_flag = 1;
    if (_Cmi_mynode == 0) CmiPrintf("Charm++: CheckSum checking enabled! \n");
#else
    if (_Cmi_mynode == 0) CmiPrintf("Charm++: +checksum ignored in optimized version! \n");
#endif
  }

  {
  int debug = CmiGetArgFlag(argv,"++debug");
  int debug_no_pause = CmiGetArgFlag(argv,"++debug-no-pause");
  if (debug || debug_no_pause)
  {   /*Pause so user has a chance to start and attach debugger*/
#if CMK_HAS_GETPID
    printf("CHARMDEBUG> Processor %d has PID %d\n",_Cmi_mynode,getpid());
    fflush(stdout);
    if (!debug_no_pause)
      sleep(15);
#else
    printf("++debug ignored.\n");
#endif
  }
  }

#if MPI_POST_RECV_COUNT > 0

  CpvInitialize(unsigned long long, Cmi_posted_recv_total);
  CpvInitialize(unsigned long long, Cmi_unposted_recv_total);
  CpvInitialize(MPI_Request*, CmiPostedRecvRequests); 
  CpvInitialize(char*,CmiPostedRecvBuffers);

    /* Post some extra recvs to help out with incoming messages */
    /* On some MPIs the messages are unexpected and thus slow */

    /* An array of request handles for posted recvs */
    CpvAccess(CmiPostedRecvRequests) = (MPI_Request*)malloc(sizeof(MPI_Request)*MPI_POST_RECV_COUNT);

    /* An array of buffers for posted recvs */
    CpvAccess(CmiPostedRecvBuffers) = (char*)malloc(MPI_POST_RECV_COUNT*MPI_POST_RECV_SIZE);

    /* Post Recvs */
    for(i=0; i<MPI_POST_RECV_COUNT; i++){
        if(MPI_SUCCESS != MPI_Irecv(  &(CpvAccess(CmiPostedRecvBuffers)[i*MPI_POST_RECV_SIZE])	,
                    MPI_POST_RECV_SIZE,
                    MPI_BYTE,
                    MPI_ANY_SOURCE,
                    POST_RECV_TAG,
                    MPI_COMM_WORLD,
		    &(CpvAccess(CmiPostedRecvRequests)[i])  ))
	  CmiAbort("MPI_Irecv failed\n");
    }

#endif



  /* CmiTimerInit(); */

#if 0
  CthInit(argv);
  ConverseCommonInit(argv);

  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,CmiNotifyIdle,NULL);
  if (initret==0) {
    fn(CmiGetArgc(argv), argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
#endif

  CsvInitialize(CmiNodeState, NodeState);
  CmiNodeStateInit(&CsvAccess(NodeState));

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
  exitLock = CmiCreateLock();            /* exit count lock */
#endif

  /* Network progress function is used to poll the network when for
     messages. This flushes receive buffers on some  implementations*/
  networkProgressPeriod = NETWORK_PROGRESS_PERIOD_DEFAULT;
  CmiGetArgInt(argv, "+networkProgressPeriod", &networkProgressPeriod);

  CmiStartThreads(argv);
  ConverseRunPE(initret);
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message)
{
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


#if 0

/* ****************************************************************** */
/*    The following internal functions implement recd msg queue       */
/* ****************************************************************** */

static void ** AllocBlock(unsigned int len)
{
  void ** blk;

  blk=(void **)CmiAlloc(len*sizeof(void *));
  if(blk==(void **)0) {
    CmiError("Cannot Allocate Memory!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return blk;
}

static void
SpillBlock(void **srcblk, void **destblk, unsigned int first, unsigned int len)
{
  memcpy(destblk, &srcblk[first], (len-first)*sizeof(void *));
  memcpy(&destblk[len-first],srcblk,first*sizeof(void *));
}

void recdQueueInit(void)
{
  recdQueue_blk = AllocBlock(BLK_LEN);
  recdQueue_blk_len = BLK_LEN;
  recdQueue_first = 0;
  recdQueue_len = 0;
}

void recdQueueAddToBack(void *element)
{
#if NODE_0_IS_CONVHOST
  inside_comm = 1;
#endif
  if(recdQueue_len==recdQueue_blk_len) {
    void **blk;
    recdQueue_blk_len *= 3;
    blk = AllocBlock(recdQueue_blk_len);
    SpillBlock(recdQueue_blk, blk, recdQueue_first, recdQueue_len);
    CmiFree(recdQueue_blk);
    recdQueue_blk = blk;
    recdQueue_first = 0;
  }
  recdQueue_blk[(recdQueue_first+recdQueue_len++)%recdQueue_blk_len] = element;
#if NODE_0_IS_CONVHOST
  inside_comm = 0;
#endif
}


void * recdQueueRemoveFromFront(void)
{
  if(recdQueue_len) {
    void *element;
    element = recdQueue_blk[recdQueue_first++];
    recdQueue_first %= recdQueue_blk_len;
    recdQueue_len--;
    return element;
  }
  return 0;
}

#endif

/*@}*/
