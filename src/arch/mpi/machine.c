/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
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
#define MAX_QLEN 200

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
#define CMK_BROADCAST_SPANNING_TREE    1
#define CMK_BROADCAST_HYPERCUBE        0
#endif

#define BROADCAST_SPANNING_FACTOR      4

#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_GET_CYCLE(msg)               ((CmiMsgHeaderBasic *)msg)->root

#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank

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


#define TAG     1375

int Cmi_numpes;
int               Cmi_mynode;    /* Which address space am I */
int               Cmi_mynodesize;/* Number of processors in my address space */
int               Cmi_numnodes;  /* Total number of address spaces */
int               Cmi_numpes;    /* Total number of processors */
static int        Cmi_nodestart; /* First processor in this address space */ 
CpvDeclare(void*, CmiLocalQueue);

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

static int MsgQueueLen=0;
static int request_max;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

static int Cmi_dim;

static int no_outstanding_sends=0; /*FLAG: consume outstanding Isends in scheduler loop*/

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

double starttimer;

void CmiAbort(const char *message);
static void PerrorExit(const char *msg);

void SendSpanningChildren(int size, char *msg);
void SendHypercube(int size, char *msg);

static void PerrorExit(const char *msg)
{
  perror(msg);
  exit(1);
}


char *CopyMsg(char *msg, int len)
{
  char *copy = (char *)CmiAlloc(len);
  if (!copy)
      fprintf(stderr, "Out of memory\n");
  memcpy(copy, msg, len);
  return copy;
}

/**************************  TIMER FUNCTIONS **************************/

#if CMK_TIMER_USE_SPECIAL

/* MPI calls are not threadsafe, even the timer on some machines */
static CmiNodeLock  timerLock = NULL;

void CmiTimerInit(void)
{
  starttimer = PMPI_Wtime();
/*  timerLock = CmiCreateLock(); */
}

double CmiTimer(void)
{
  double t;
  if (timerLock) CmiLock(timerLock);
  t = PMPI_Wtime() - starttimer;
  if (timerLock) CmiUnlock(timerLock);
  return t;
}

double CmiWallTimer(void)
{
  double t;
#if CMK_SMP
  if (timerLock) CmiLock(timerLock);
#endif
  t = PMPI_Wtime() - starttimer;
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
  t = PMPI_Wtime() - starttimer;
#if CMK_SMP
  if (timerLock) CmiUnlock(timerLock);
#endif
  return t;
}

#endif


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
int Cmi_mype;
int Cmi_myrank;

void CmiMemLock(void) {}
void CmiMemUnlock(void) {}

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

void CmiYield(void) { sleep(0); }

static void CmiStartThreads(char **argv)
{
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  Cmi_mype = Cmi_nodestart;
  Cmi_myrank = 0;
}      
#endif	/* non smp */

/*Add a message to this processor's receive queue, pe is a rank */
static void CmiPushPE(int pe,void *msg)
{
  CmiState cs = CmiGetStateN(pe);
  MACHSTATE2(3,"Pushing message into rank %d's queue %p{",pe, cs->recv);
#if CMK_IMMEDIATE_MSG
  if (CmiGetHandler(msg) == CpvAccessOther(CmiImmediateMsgHandlerIdx,pe)) {
/*
CmiPrintf("[node %d] Immediate Message {{. \n", CmiMyNode());
    CmiHandleMessage(msg);
CmiPrintf("[node %d] Immediate Message done.}} \n", CmiMyNode());
*/
    *(CmiUInt2 *)msg = pe;
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
  if (CmiGetHandler(msg) == CpvAccessOther(CmiImmediateMsgHandlerIdx,0)) {
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
int CmiNodeFirst(int node) { return node*Cmi_mynodesize; }
int CmiNodeSize(int node)  { return Cmi_mynodesize; }
#endif

#ifndef CmiNodeOf
int CmiNodeOf(int pe)      { return (pe/Cmi_mynodesize); }
int CmiRankOf(int pe)      { return pe%Cmi_mynodesize; }
#endif

static int CmiAllAsyncMsgsSent(void)
{
   SMSG_LIST *msg_tmp = sent_msgs;
   MPI_Status sts;
   int done;
     
   while(msg_tmp!=0) {
    done = 0;
    if (MPI_SUCCESS != PMPI_Test(&(msg_tmp->req), &done, &sts)) 
      CmiAbort("CmiAllAsyncMsgsSent: PMPI_Test failed!\n");
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
    if (MPI_SUCCESS != PMPI_Test(&(msg_tmp->req), &done, &sts)) 
      CmiAbort("CmiAsyncMsgSent: PMPI_Test failed!\n");
    return ((done)?1:0);
  } else {
    return 1;
  }
}

void CmiReleaseCommHandle(CmiCommHandle c)
{
  return;
}

static void CmiReleaseSentMessages(void)
{
  SMSG_LIST *msg_tmp=sent_msgs;
  SMSG_LIST *prev=0;
  SMSG_LIST *temp;
  int done;
  MPI_Status sts;
     
  MACHSTATE1(2,"CmiReleaseSentMessages begin on %d {", CmiMyPe());
  while(msg_tmp!=0) {
    done =0;
    if(PMPI_Test(&(msg_tmp->req), &done, &sts) != MPI_SUCCESS)
      CmiAbort("CmiReleaseSentMessages: PMPI_Test failed!\n");
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

static int PumpMsgs(void)
{
  int nbytes, flg, res;
  char *msg;
  MPI_Status sts;
  int recd=0;

  MACHSTATE(2,"PumpMsgs begin {");
  while(1) {
    flg = 0;
    res = PMPI_Iprobe(MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &flg, &sts);
    if(res != MPI_SUCCESS)
      CmiAbort("PMPI_Iprobe failed\n");
    if(!flg) break;
    recd = 1;
    PMPI_Get_count(&sts, MPI_BYTE, &nbytes);
    msg = (char *) CmiAlloc(nbytes);
    if (MPI_SUCCESS != PMPI_Recv(msg,nbytes,MPI_BYTE,sts.MPI_SOURCE,TAG, MPI_COMM_WORLD,&sts)) 
      CmiAbort("PumpMsgs: PMPI_Recv failed!\n");

    MACHSTATE2(3,"PumpMsgs recv one from node:%d to rank:%d", sts.MPI_SOURCE, CMI_DEST_RANK(msg));
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

  if (buf == NULL) buf = (char *) CmiAlloc(maxbytes);

  if (MPI_SUCCESS != PMPI_Recv(buf,maxbytes,MPI_BYTE,MPI_ANY_SOURCE,TAG, MPI_COMM_WORLD,&sts)) 
      CmiAbort("PumpMsgs: PMP_Recv failed!\n");
   PMPI_Get_count(&sts, MPI_BYTE, &nbytes);
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
  for (i=0; i<Cmi_mynodesize; i++)
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
  for (i=0; i<Cmi_mynodesize; i++) {
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
    if (MPI_SUCCESS != PMPI_Barrier(MPI_COMM_WORLD))
      CmiAbort("ConverseExit: PMPI_Barrier failed!\n");
    MACHSTATE(2, "} CommunicationServer barrier end");
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyNode() == 0){
      CmiPrintf("End of program\n");
    }
#endif
    MACHSTATE(2, "} CommunicationServer EXIT");
    PMPI_Finalize();
    exit(0);   
  }
}


static void CommunicationServerThread(int sleepTime)
{
#if CMK_SMP
  CommunicationServer(sleepTime);
#endif
#if CMK_IMMEDIATE_MSG
  CmiHandleImmediate();
#endif
}

#endif


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

void CmiNotifyIdle(void)
{
  CmiReleaseSentMessages();
  if (!PumpMsgs() && idleblock) PumpMsgsBlocking();
}
 
/* user call to handle immediate message, only useful in non SMP version
   using polling method to schedule message.
*/
void CmiProbeImmediateMsg()
{
#if CMK_IMMEDIATE_MSG && !CMK_SMP
  PumpMsgs();
  CmiHandleImmediate();
#endif
}

/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);

  CMI_SET_BROADCAST_ROOT(dupmsg, 0);

  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
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
  for (i=0; i<Cmi_mynodesize; i++)
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
      if (MPI_SUCCESS != PMPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
        CmiAbort("CmiAsyncSendFn: PMPI_Isend failed!\n");
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
     
  CQdCreate(CpvAccess(cQdState), 1);
  if(destPE == cs->pe) {
    char *dupmsg = (char *) CmiAlloc(size);
    memcpy(dupmsg, msg, size);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
    return 0;
  }
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
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  while (MsgQueueLen > request_max) {
	/*printf("Waiting for %d messages to be sent\n", MsgQueueLen);*/
	CmiReleaseSentMessages();
	PumpMsgs();
  }
  if (MPI_SUCCESS != PMPI_Isend((void *)msg,size,MPI_BYTE,destPE,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
    CmiAbort("CmiAsyncSendFn: PMPI_Isend failed!\n");
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
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
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
  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
  }
  else
    CmiAsyncSendFn(destPE, size, dupmsg);
}

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg)
{
  CmiState cs = CmiGetState();
  int startpe = CMI_BROADCAST_ROOT(msg)-1;
  int i;

  assert(startpe>=0 && startpe<Cmi_numpes);

  for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
    int p = cs->pe-startpe;
    if (p<0) p+=Cmi_numpes;
    p = BROADCAST_SPANNING_FACTOR*p + i;
    if (p > Cmi_numpes - 1) break;
    p += startpe;
    p = p%Cmi_numpes;
    assert(p>=0 && p<Cmi_numpes && p!=cs->pe);
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

  /* assert(startpe>=0 && startpe<Cmi_numpes); */

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
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, Cmi_mype+1);
  SendSpanningChildren(size, msg);
  
#elif CMK_BROADCAST_HYPERCUBE
  CMI_SET_CYCLE(msg, 0);
  SendHypercube(size, msg);
    
#else
  CmiState cs = CmiGetState();
  int i;

  for ( i=cs->pe+1; i<Cmi_numpes; i++ ) 
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

  for ( i=cs->pe+1; i<Cmi_numpes; i++ ) 
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
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
#endif

  /*CmiPrintf("In  SyncBroadcastAll broadcast\n");*/
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)  
{
  int i ;

  for ( i=1; i<Cmi_numpes; i++ ) 
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
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
#endif
  CmiFree(msg) ;
  /*CmiPrintf("In FreeBroadcastAll broadcast\n");*/
}

#if CMK_NODE_QUEUE_AVAILABLE

CmiCommHandle CmiAsyncNodeSendFn(int dstNode, int size, char *msg)
{
  int i;
  SMSG_LIST *msg_tmp;
  char *dupmsg;
     
  CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
  switch (dstNode) {
  case NODE_BROADCAST_ALL:
    CQdCreate(CpvAccess(cQdState), 1);
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    PCQueuePush(CsvAccess(NodeState).NodeRecv,(char *)CopyMsg(msg,size));
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
  case NODE_BROADCAST_OTHERS:
    CQdCreate(CpvAccess(cQdState), Cmi_numnodes-1);
    for (i=0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode) {
        EnqueueMsg((char *)CopyMsg(msg,size), size, i);
      }
    break;
  default:
    CQdCreate(CpvAccess(cQdState), 1);
    dupmsg = (char *)CopyMsg(msg,size);
    if(dstNode == Cmi_mynode) {
      CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
      PCQueuePush(CsvAccess(NodeState).NodeRecv, dupmsg);
      CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
      return 0;
    }
    else {
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
#define MPI_REQUEST_MAX 1024*10 

void ConverseExit(void)
{
#if ! CMK_SMP
  while(!CmiAllAsyncMsgsSent()) {
    PumpMsgs();
    CmiReleaseSentMessages();
  }
  if (MPI_SUCCESS != PMPI_Barrier(MPI_COMM_WORLD)) 
    CmiAbort("ConverseExit: PMPI_Barrier failed!\n");
  ConverseCommonExit();
  PMPI_Finalize();
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

  Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
  /* communication thread */
  if (CmiMyRank() == CmiMyNodeSize()) {
    while (1) CommunicationServerThread(5);
  }
  else {  /* worker thread */
  if (!everReturn) {
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
  
  PMPI_Init(&argc, &argv);
  PMPI_Comm_size(MPI_COMM_WORLD, &Cmi_numnodes);
  PMPI_Comm_rank(MPI_COMM_WORLD, &Cmi_mynode);
  /* processor per node */
  Cmi_mynodesize = 1;
  CmiGetArgInt(argv,"+ppn", &Cmi_mynodesize);
#if ! CMK_SMP
  if (Cmi_mynodesize > 1 && Cmi_mynode == 0) 
    CmiAbort("+ppn cannot be used in non SMP version!\n");
#endif
  idleblock = CmiGetArgFlag(argv, "+idleblocking");
  if (idleblock && Cmi_mynode == 0) {
    CmiPrintf("Charm++: Running in idle blocking mode.\n");
  }

#if CMK_NO_OUTSTANDING_SENDS
  no_outstanding_sends=1;
#endif
  if (CmiGetArgInt(argv,"+no_outstanding_sends",&no_outstanding_sends) && Cmi_mynode == 0) {
     CmiPrintf("Charm++: Will%s consume outstanding sends in scheduler loop\n",
     	no_outstanding_sends?"":" not");
  }
  Cmi_numpes = Cmi_numnodes * Cmi_mynodesize;
  Cmi_nodestart = Cmi_mynode * Cmi_mynodesize;
  Cmi_argvcopy = CmiCopyArgs(argv);
  Cmi_argv = argv; Cmi_startfn = fn; Cmi_usrsched = usched;
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;
 /* CmiSpanTreeInit();*/
  request_max=MAX_QLEN;
  CmiGetArgInt(argv,"+requestmax",&request_max);
  /*printf("request max=%d\n", request_max);*/
  if (CmiGetArgFlag(argv,"++debug"))
  {   /*Pause so user has a chance to start and attach debugger*/
    printf("CHARMDEBUG> Processor %d has PID %d\n",Cmi_mynode,getpid());
    if (!CmiGetArgFlag(argv,"++debug-no-pause"))
      sleep(10);
  }

  CmiTimerInit();

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

  procState = (ProcState *)malloc(Cmi_mynodesize * sizeof(ProcState));
  for (i=0; i<Cmi_mynodesize+1; i++) {
/*    procState[i].sendMsgBuf = PCQueueCreate();   */
    procState[i].recvLock = CmiCreateLock();
  }
#if CMK_SMP
  sendMsgBuf = PCQueueCreate();
  sendMsgBufLock = CmiCreateLock();
#endif

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
  CmiError(message);
  PMPI_Abort(MPI_COMM_WORLD, 1);
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
    PMPI_Abort(MPI_COMM_WORLD, 1);
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
