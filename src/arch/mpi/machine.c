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

#ifndef CMK_SMP
#if defined(CMK_SHARED_VARS_POSIX_THREADS_SMP)
# define CMK_SMP 1
#endif
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

typedef struct msg_list {
     MPI_Request req;
     char *msg;
     struct msg_list *next;
     int size, destpe;
} SMSG_LIST;

static int MsgQueueLen=0;
static int request_max;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

static int Cmi_dim;

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

void CmiTimerInit(void)
{
  starttimer = PMPI_Wtime();
}

double CmiTimer(void)
{
  return PMPI_Wtime() - starttimer;
}

double CmiWallTimer(void)
{
  return PMPI_Wtime() - starttimer;
}

double CmiCpuTimer(void)
{
  return PMPI_Wtime() - starttimer;
}

static PCQueue  *msgBuf;

/************************************************************
 * 
 * Processor state structure
 *
 ************************************************************/

#if CMK_NODE_QUEUE_AVAILABLE
CsvStaticDeclare(CmiNodeLock, CmiNodeRecvLock);
CsvStaticDeclare(PCQueue, NodeRecv);
#endif

/* fake Cmi_charmrun_fd */
static int Cmi_charmrun_fd = 0;
#include "machine-smp.c"

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
#endif

/*Add a message to this processor's receive queue */
static void CmiPushPE(int pe,void *msg)
{
  CmiState cs=CmiGetStateN(pe);
  MACHSTATE1(2,"Pushing message into %d's queue",pe);
  CmiIdleLock_addMessage(&cs->idle); 
  PCQueuePush(cs->recv,msg);
}

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
    MsgQueueLen--;
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
  int locked = 0;
     
  while(msg_tmp!=0) {
    done =0;
    if(PMPI_Test(&(msg_tmp->req), &done, &sts) != MPI_SUCCESS)
      CmiAbort("CmiReleaseSentMessages: PMPI_Test failed!\n");
    if(done) {
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
}

static int PumpMsgs(void)
{
  int nbytes, flg, res;
  char *msg;
  MPI_Status sts;
  int recd=0;

  while(1) {
    flg = 0;
    res = PMPI_Iprobe(MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &flg, &sts);
    if(res != MPI_SUCCESS)
      CmiAbort("PMPI_Iprobe failed\n");
    if(!flg)
      return recd;
    recd = 1;
    PMPI_Get_count(&sts, MPI_BYTE, &nbytes);
    msg = (char *) CmiAlloc(nbytes);
    if (MPI_SUCCESS != PMPI_Recv(msg,nbytes,MPI_BYTE,sts.MPI_SOURCE,TAG, MPI_COMM_WORLD,&sts)) 
      CmiAbort("PumpMsgs: PMPI_Recv failed!\n");

#if CMK_NODE_QUEUE_AVAILABLE
    if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
      PCQueuePush(CsvAccess(NodeRecv), msg);
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
}

/* blocking version */
static void PumpMsgsBlocking(void)
{
  static int maxbytes = 20000000;
  static char *buf = NULL;
  int nbytes, flg, res;
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
      PCQueuePush(CsvAccess(NodeRecv), msg);
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

static int inexit = 0;

static int MsgQueueEmpty()
{
  int i;
  for (i=0; i<Cmi_mynodesize; i++)
    if (!PCQueueEmpty(msgBuf[i])) return 0;
  return 1;
}

#if CMK_SMP
/**
CommunicationServer calls MPI to send messages in the queues and probe message from network.
*/
static void CommunicationServer(int sleepTime)
{
  CmiReleaseSentMessages();
  SendMsgBuf(); 
  if (!PumpMsgs()) CmiYield(); 
  if (inexit == 1) {
    while(!MsgQueueEmpty() || !CmiAllAsyncMsgsSent()) {
      CmiReleaseSentMessages();
      SendMsgBuf(); 
      PumpMsgs();
    }
    if (MPI_SUCCESS != PMPI_Barrier(MPI_COMM_WORLD))
      CmiAbort("ConverseExit: PMPI_Barrier failed!\n");
    PMPI_Finalize();
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyNode() == 0){
      CmiPrintf("End of program\n");
    }
#endif
    exit(0);   
  }
}
#endif

#if CMK_NODE_QUEUE_AVAILABLE
char *CmiGetNonLocalNodeQ(void)
{
  char *result = 0;
  if(!PCQueueEmpty(CsvAccess(NodeRecv))) {
    CmiLock(CsvAccess(CmiNodeRecvLock));
    result = (char *) PCQueuePop(CsvAccess(NodeRecv));
    CmiUnlock(CsvAccess(CmiNodeRecvLock));
  }
  return result;
}
#endif

void *CmiGetNonLocal(void)
{
  CmiState cs = CmiGetState();
  void *msg;
  CmiIdleLock_checkMessage(&cs->idle);
  msg =  PCQueuePop(cs->recv); 
#if ! CMK_SMP
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

static void SendMsgBuf()
{
  SMSG_LIST *msg_tmp;
  char *msg;
  int node, rank, size;
  int i;

  for (i=0; i<Cmi_mynodesize; i++)
  {
    while (!PCQueueEmpty(msgBuf[i]))
    {
      msg_tmp = (SMSG_LIST *)PCQueuePop(msgBuf[i]);
      node = msg_tmp->destpe;
      size = msg_tmp->size;
      msg = msg_tmp->msg;
      msg_tmp->next = 0;
      while (MsgQueueLen > request_max) {
	/*printf("Waiting for %d messages to be sent\n", MsgQueueLen);*/
	CmiReleaseSentMessages();
	PumpMsgs();
      }
      if (MPI_SUCCESS != PMPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
        CmiAbort("CmiAsyncSendFn: PMPI_Isend failed!\n");
      MsgQueueLen++;
      if(sent_msgs==0)
        sent_msgs = msg_tmp;
      else
        end_sent->next = msg_tmp;
      end_sent = msg_tmp;
    }
  } 
}

#if CMK_SMP
#define EnqueueMsg(m, size, node)    { 	\
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));	\
  msg_tmp->msg = m;	\
  msg_tmp->size = size;	\
  msg_tmp->destpe = node;	\
  PCQueuePush(msgBuf[CmiMyRank()],(char *)msg_tmp);	\
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
  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, Cmi_mype+1);
  SendSpanningChildren(size, msg);
  
#elif CMK_BROADCAST_HYPERCUBE
  CMI_SET_CYCLE(msg, 0);
  SendHypercube(size, msg);
    
#else
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

  /*CmiPrintf("In  AsyncBroadcastAll broadcast\n");*/
    
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
    PCQueuePush(CsvAccess(NodeRecv),(char *)CopyMsg(msg,size));
  case NODE_BROADCAST_OTHERS:
    CQdCreate(CpvAccess(cQdState), Cmi_mynode-1);
    for (i=0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode) {
        EnqueueMsg((char *)CopyMsg(msg,size), size, i);
      }
    break;
  default:
    CQdCreate(CpvAccess(cQdState), 1);
    dupmsg = (char *) CmiAlloc(size);
    memcpy(dupmsg, msg, size);
    if(dstNode == Cmi_mynode) {
      PCQueuePush(CsvAccess(NodeRecv), dupmsg);
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
  inexit = 1;
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
  CmiYield();
#endif

#if 0
  int nSpins=20; /*Number of times to spin before sleeping*/
  s->nIdles++;
  if (s->nIdles>nSpins) { /*Start giving some time back to the OS*/
    s->sleepMs+=2;
    if (s->sleepMs>10) s->sleepMs=10;
  }
  /*Comm. thread will listen on sockets-- just sleep*/
  if (s->sleepMs>0) {
    MACHSTATE1(3,"idle lock(%d) {",CmiMyPe())
    CmiIdleLock_sleep(&s->cs->idle,s->sleepMs);
    MACHSTATE1(3,"} idle lock(%d)",CmiMyPe())
  }       
#endif
}

static void ConverseRunPE(int everReturn)
{
  CmiIdleState *s=CmiNotifyGetState();
  CmiState cs;
  char** CmiMyArgv;
  CmiNodeBarrier();
  cs = CmiGetState();
  CpvInitialize(void *,CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = cs->localqueue;

  if (CmiMyRank())
    CmiMyArgv=CmiCopyArgs(Cmi_argvcopy);
  else   
    CmiMyArgv=Cmi_argv;
    
  CthInit(CmiMyArgv);
#if MACHINE_DEBUG_LOG
  {
    char ln[200];
    sprintf(ln,"debugLog.%d",CmiMyPe());
    debugLog=fopen(ln,"w");
  }
#endif

  ConverseCommonInit(CmiMyArgv);

#if CMK_SMP
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
#endif

  if (!everReturn) {
    Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
    if (Cmi_usrsched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int n,i ;
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
  Cmi_numpes = Cmi_numnodes * Cmi_mynodesize;
  Cmi_nodestart = Cmi_mynode * Cmi_mynodesize;
  Cmi_argvcopy = CmiCopyArgs(argv);
  Cmi_argv = argv; Cmi_startfn = fn; Cmi_usrsched = usched;
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;
 /* CmiSpanTreeInit();*/
  i=0;
  request_max=MAX_QLEN;
  CmiGetArgInt(argv,"+requestmax",&request_max);
  /*printf("request max=%d\n", request_max);*/
  if (CmiGetArgFlag(argv,"++debug"))
  {   /*Pause so user has a chance to start and attach debugger*/
    printf("CHARMDEBUG> Processor %d has PID %d\n",CmiMyNode(),getpid());
    if (!CmiGetArgFlag(argv,"++debug-no-pause"))
      sleep(10);
  }

  CmiTimerInit();
#if CMK_SMP
  msgBuf = (PCQueue *)malloc(Cmi_mynodesize * sizeof(PCQueue));
  for (i=0; i<Cmi_mynodesize; i++)
    msgBuf[i] = PCQueueCreate();
#endif

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
