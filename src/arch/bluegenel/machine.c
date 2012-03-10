/** @file
 * MPI based machine layer
 * @ingroup Machine
 */
/*@{*/

#include <stdio.h>
#include <errno.h>
#include "converse.h"
#include <mpi.h>

#ifdef AMPI
#  warning "We got the AMPI version of mpi.h, instead of the system version--"
#  warning "   Try doing an 'rm charm/include/mpi.h' and building again."
#  error "Can't build Charm++ using AMPI version of mpi.h header"
#endif

/*Support for ++debug: */
#include <unistd.h> /*For getpid()*/
#include <stdlib.h> /*For sleep()*/

#if defined(CMK_SHARED_VARS_POSIX_THREADS_SMP)
#define CMK_SMP 1
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


/*
    To reduce the buffer used in broadcast and distribute the load from 
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of 
  spanning tree broadcast algorithm.
    This will use the fourth short in message as an indicator of spanning tree
  root.
*/
#if CMK_SMP
#define CMK_BROADCAST_SPANNING_TREE    0
#else
#define CMK_BROADCAST_SPANNING_TREE    0
#define CMK_BROADCAST_HYPERCUBE        1
#endif

#define BROADCAST_SPANNING_FACTOR      4

#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_GET_CYCLE(msg)               ((CmiMsgHeaderBasic *)msg)->root

#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank
#define CMI_MAGIC(msg)			 ((CmiMsgHeaderBasic *)msg)->magic

/* FIXME: need a random number that everyone agrees ! */
#define CHARM_MAGIC_NUMBER		 126

#if !CMK_OPTIMIZE
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

#if CMK_BROADCAST_SPANNING_TREE
#  define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);
#else
#  define CMI_SET_BROADCAST_ROOT(msg, root)
#endif

#if CMK_BROADCAST_HYPERCUBE
#  define CMI_SET_CYCLE(msg, cycle)  CMI_GET_CYCLE(msg) = (cycle);
#else
#  define CMI_SET_CYCLE(msg, cycle)
#endif

/*
 to avoid MPI's in order delivery, changing MPI Tag all the time
*/
#define TAG     1375
/*
static int mpi_tag = TAG;
#define NEW_MPI_TAG	mpi_tag++; if (mpi_tag == MPI_TAG_UB) mpi_tag=TAG;
*/

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

#if CMK_TIMER_USE_SPECIAL

/* MPI calls are not threadsafe, even the timer on some machines */
static CmiNodeLock  timerLock = 0;
static double starttimer = 0;
static int _is_global = 0;

int CmiTimerIsSynchronized(void)
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

void CmiTimerInit(char **argv)
{
  _is_global = CmiTimerIsSynchronized();

  if (CmiMyRank() == 0) {
    if (_is_global) { 
      double minTimer;
      starttimer = MPI_Wtime();
      MPI_Allreduce(&starttimer, &minTimer, 1, MPI_DOUBLE, MPI_MIN, 
                                  MPI_COMM_WORLD );
      starttimer = minTimer;
    }
    else {
      /* we don't have a synchronous timer, set our own start time */
      CmiBarrier();
      CmiBarrier();
      CmiBarrier();
      starttimer = MPI_Wtime();
    }
  }
  CmiNodeAllBarrier();          /* for smp */
/*  timerLock = CmiCreateLock(); */
}

double CmiTimer(void)
{
  double t;
#if CMK_SMP
  if (timerLock) CmiLock(timerLock);
#endif
  t = MPI_Wtime() - starttimer;
#if CMK_SMP
  if (timerLock) CmiUnlock(timerLock);
#endif
  return t;
}

double CmiWallTimer(void)
{
  double t;
#if CMK_SMP
  if (timerLock) CmiLock(timerLock);
#endif
  t = MPI_Wtime() - starttimer;
#if CMK_SMP
  if (timerLock) CmiUnlock(timerLock);
#endif
  return t;
}

double CmiCpuTimer(void)
{
  double t;
#if CMK_SMP
  if (timerLock) CmiLock(timerLock);
#endif
  t = MPI_Wtime() - starttimer;
#if CMK_SMP
  if (timerLock) CmiUnlock(timerLock);
#endif
  return t;
}

#endif

int CmiBarrier()
{
  if (CmiMyRank() == 0)
    if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD))
        CmiAbort("Timernit: MPI_Barrier failed!\n");
  return 0;
}

/* CmiBarrierZero make sure node 0 is the last one exiting the barrier */
int CmiBarrierZero()
{
  int i;
  if (CmiMyRank() == 0) {
    char msg[1];
    MPI_Status sts;
    if (CmiMyNode() == 0)  {
      for (i=0; i<CmiNumNodes()-1; i++) {
        if (MPI_SUCCESS != MPI_Recv(msg,1,MPI_BYTE,MPI_ANY_SOURCE,TAG, MPI_COMM_WORLD,&sts))
          printf("MPI_Recv failed!\n");
      }
    }
    else {
      if (MPI_SUCCESS != MPI_Send((void *)msg,1,MPI_BYTE,0,TAG,MPI_COMM_WORLD))
         printf("MPI_Send failed!\n");
    }
  }
  CmiNodeAllBarrier();
  return 0;
}

typedef struct ProcState {
/* PCQueue      sendMsgBuf; */      /* per processor message sending queue */
CmiNodeLock  recvLock;		    /* for cs->recv */
} ProcState;

static ProcState  *procState;

#if CMK_SMP

static PCQueue sendMsgBuf;
static CmiNodeLock  sendMsgBufLock = NULL;        /* for sendMsgBuf */

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

#if ! CMK_SMP
/************ non SMP **************/
static struct CmiStateStruct Cmi_state;
int _Cmi_mype;
int _Cmi_myrank;

void CmiMemLock(void) {}
void CmiMemUnlock(void) {}

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

static int CmiAllAsyncMsgsSent(void)
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

#if CMK_BLUEGENEL
  MPID_Progress_test();
#endif

  MACHSTATE(2,"PumpMsgs begin {");
  while(1) {
    flg = 0;
    res = MPI_Iprobe(MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &flg, &sts);
    if(res != MPI_SUCCESS)
      CmiAbort("MPI_Iprobe failed\n");
    if(!flg) break;
    recd = 1;
    MPI_Get_count(&sts, MPI_BYTE, &nbytes);
    msg = (char *) CmiAlloc(nbytes);
    if (MPI_SUCCESS != MPI_Recv(msg,nbytes,MPI_BYTE,sts.MPI_SOURCE,sts.MPI_TAG, MPI_COMM_WORLD,&sts)) 
      CmiAbort("PumpMsgs: MPI_Recv failed!\n");

    MACHSTATE2(3,"PumpMsgs recv one from node:%d to rank:%d", sts.MPI_SOURCE, CMI_DEST_RANK(msg));
    CMI_CHECK_CHECKSUM(msg, nbytes);
    if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
      CmiPrintf("Charm++ Warning: Non Charm++ Message Received. \n");
      CmiFree(msg);
      continue;
    }
#if CMK_NODE_QUEUE_AVAILABLE
    if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
      CmiPushNode(msg);
    else
#endif
      CmiPushPE(CMI_DEST_RANK(msg), msg);

#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_BROADCAST_ROOT(msg))
      SendSpanningChildren(nbytes, msg);
#elif CMK_BROADCAST_HYPERCUBE
    if (CMI_GET_CYCLE(msg))
      SendHypercube(nbytes, msg);
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

  if (MPI_SUCCESS != MPI_Recv(buf,maxbytes,MPI_BYTE,MPI_ANY_SOURCE,TAG, MPI_COMM_WORLD,&sts)) 
      CmiAbort("PumpMsgs: PMP_Recv failed!\n");
   MPI_Get_count(&sts, MPI_BYTE, &nbytes);
   msg = (char *) CmiAlloc(nbytes);
   memcpy(msg, buf, nbytes);

#if CMK_NODE_QUEUE_AVAILABLE
   if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
      CmiPushNode(msg);
   else
#endif
      CmiPushPE(CMI_DEST_RANK(msg), msg);

#if CMK_BROADCAST_SPANNING_TREE
   if (CMI_BROADCAST_ROOT(msg))
      SendSpanningChildren(nbytes, msg);
#elif CMK_BROADCAST_HYPERCUBE
   if (CMI_GET_CYCLE(msg))
      SendHypercube(nbytes, msg);
#endif
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

#if CMK_SMP

static int inexit = 0;

static int MsgQueueEmpty()
{
  int i;
#if 0
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
static void CommunicationServer(int sleepTime)
{
  int static count=0;
/*
  count ++;
  if (count % 10000000==0) MACHSTATE(3, "Entering CommunicationServer {");
*/
  PumpMsgs();
  CmiReleaseSentMessages();
  SendMsgBuf(); 
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
    if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD))
      CmiAbort("ConverseExit: MPI_Barrier failed!\n");
    MACHSTATE(2, "} CommunicationServer barrier end");
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyNode() == 0){
      CmiPrintf("End of program\n");
    }
#endif
    MACHSTATE(2, "} CommunicationServer EXIT");
    MPI_Finalize();
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
  if(!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
    MACHSTATE1(3,"CmiGetNonLocalNodeQ begin %d {", CmiMyPe());
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
  }
  return result;
}
#endif

void *CmiGetNonLocal(void)
{
  static int count=0;
  CmiState cs = CmiGetState();
  void *msg;
  CmiIdleLock_checkMessage(&cs->idle);
  /* although it seems that lock is not needed, I found it crashes very often
     on mpi-smp without lock */

  CmiReleaseSentMessages();
  PumpMsgs();
  
  CmiLock(procState[cs->rank].recvLock);
  msg =  PCQueuePop(cs->recv); 
  CmiUnlock(procState[cs->rank].recvLock);

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
void CmiMachineProgressImpl()
{
#if !CMK_SMP
  PumpMsgs();
  //MPID_Progress_test();
#if CMK_IMMEDIATE_MSG
    CmiHandleImmediate();
#endif
#else
    /*Not implemented yet. Communication server does not seem to be
      thread safe */
    /* CommunicationServerThread(0); */
#endif
}

/********************* MESSAGE SEND FUNCTIONS ******************/

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
    CmiAsyncSendFn(destPE, size, dupmsg);
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

  MACHSTATE(2,"SendMsgBuf begin {");
#if 0
  for (i=0; i<_Cmi_mynodesize; i++)
  {
    while (!PCQueueEmpty(procState[i].sendMsgBuf))
    {
      msg_tmp = (SMSG_LIST *)PCQueuePop(procState[i].sendMsgBuf);
#else
    /* single message sending queue */
    CmiLock(sendMsgBufLock);
    msg_tmp = (SMSG_LIST *)PCQueuePop(sendMsgBuf);
    CmiUnlock(sendMsgBufLock);
    while (NULL != msg_tmp)
    {
#endif
      node = msg_tmp->destpe;
      size = msg_tmp->size;
      msg = msg_tmp->msg;
      msg_tmp->next = 0;
      while (MsgQueueLen > request_max) {
	CmiReleaseSentMessages();
	PumpMsgs();
      }
      MACHSTATE2(3,"MPI_send to node %d rank: %d{", node, CMI_DEST_RANK(msg));
      CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
      CMI_SET_CHECKSUM(msg, size);
      if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
        CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
      MACHSTATE(3,"}MPI_send end");
      MsgQueueLen++;
      if(sent_msgs==0)
        sent_msgs = msg_tmp;
      else
        end_sent->next = msg_tmp;
      end_sent = msg_tmp;
      sent=1;
      CmiLock(sendMsgBufLock);
      msg_tmp = (SMSG_LIST *)PCQueuePop(sendMsgBuf);
      CmiUnlock(sendMsgBufLock);
    }
#if 0
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
  CmiLock(sendMsgBufLock);
  PCQueuePush(sendMsgBuf,(char *)msg_tmp);
  CmiUnlock(sendMsgBufLock);
  MACHSTATE3(3,"}} EnqueueMsg to %d finish with queue %p len: %d", node, sendMsgBuf, PCQueueLength(sendMsgBuf));
}

#endif

CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
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
  CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
  CMI_SET_CHECKSUM(msg, size);
  if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,destPE,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
    CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
  MsgQueueLen++;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
  return (CmiCommHandle) &(msg_tmp->req);
#endif
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  CMI_SET_BROADCAST_ROOT(msg, 0);

  if (cs->pe==destPE) {
    CmiSendSelf(msg);
  } else {
    CmiAsyncSendFn(destPE, size, msg);
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
    CmiAsyncSendFn(destPE, size, dupmsg);
}

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg)
{
  CmiState cs = CmiGetState();
  int startpe = CMI_BROADCAST_ROOT(msg)-1;
  int i;

  CmiAssert(startpe>=0 && startpe<_Cmi_numpes);

  for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
    int p = cs->pe-startpe;
    if (p<0) p+=_Cmi_numpes;
    p = BROADCAST_SPANNING_FACTOR*p + i;
    if (p > _Cmi_numpes - 1) break;
    p += startpe;
    p = p%_Cmi_numpes;
    CmiAssert(p>=0 && p<_Cmi_numpes && p!=cs->pe);
    CmiSyncSendFn1(p, size, msg);
  }
}

#include <math.h>

/* send msg along the hypercube in broadcast. (Sameer) */
void SendHypercube(int size, char *msg)
{
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

    if(p < CmiNumPes()) {
      CMI_SET_CYCLE(msg, i + 1);
      CmiSyncSendFn1(p, size, msg);
    }
  }
}

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);
  
#elif CMK_BROADCAST_HYPERCUBE
  CMI_SET_CYCLE(msg, 0);
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
 
#if CMK_BROADCAST_SPANNING_TREE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);

#elif CMK_BROADCAST_HYPERCUBE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_CYCLE(msg, 0);
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

#if CMK_BROADCAST_SPANNING_TREE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);

#elif CMK_BROADCAST_HYPERCUBE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  CMI_SET_CYCLE(msg, 0);
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
    if (CmiIsImmediate(msg) && !_immRunning) {
      /*CmiHandleImmediateMessage(msg); */
      CmiPushImmediateMsg(msg);
      CmiHandleImmediate();
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
#define MPI_REQUEST_MAX 16      //1024*10 

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
  MPI_Finalize();
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
  if (CmiMyPe() == 0){
    CmiPrintf("End of program\n");
  }
#endif
  exit(0);
#else
  /* SMP version, communication thread will exit */
  ConverseCommonExit();
  /* atomic increment */
  CmiCommLock();
  inexit++;
  CmiCommUnlock();
  while (1) CmiYield();
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

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int n,i;

#if MACHINE_DEBUG
  debugLog=NULL;
#endif
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &_Cmi_numnodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &_Cmi_mynode);
  /* processor per node */
  _Cmi_mynodesize = 1;
  CmiGetArgInt(argv,"+ppn", &_Cmi_mynodesize);
#if ! CMK_SMP
  if (_Cmi_mynodesize > 1 && _Cmi_mynode == 0) 
    CmiAbort("+ppn cannot be used in non SMP version!\n");
#endif
  idleblock = CmiGetArgFlag(argv, "+idleblocking");
  if (idleblock && _Cmi_mynode == 0) {
    CmiPrintf("Charm++: Running in idle blocking mode.\n");
  }

#if CMK_NO_OUTSTANDING_SENDS
  no_outstanding_sends=1;
#endif
  if (CmiGetArgFlag(argv,"+no_outstanding_sends")) {
    no_outstanding_sends = 1;
    if (_Cmi_mynode == 0)
      CmiPrintf("Charm++: Will%s consume outstanding sends in scheduler loop\n",
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

  if (CmiGetArgFlag(argv,"++debug"))
  {   /*Pause so user has a chance to start and attach debugger*/
    printf("CHARMDEBUG> Processor %d has PID %d\n",_Cmi_mynode,getpid());
    if (!CmiGetArgFlag(argv,"++debug-no-pause"))
      sleep(10);
  }

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
    /*    procState[i].sendMsgBuf = PCQueueCreate();   */
    procState[i].recvLock = CmiCreateLock();
  }
#if CMK_SMP
  sendMsgBuf = PCQueueCreate();
  sendMsgBufLock = CmiCreateLock();
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
  CmiError("------------- Processor %d Exiting: Called CmiAbort ------------\n"
        "Reason: %s\n",CmiMyPe(),message);
 /*  CmiError(message); */
  CmiPrintStackTrace(0);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

/*@}*/
