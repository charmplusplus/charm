/** @file
 * machine.c on IBM BlueGene/L Message Layer
 */
/*@{*/

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "machine.h"
#include "converse.h"
#include "pcqueue.h"

/* For Gheorghe's msg templates */
#include "templates/TorusDirectMessage.h"
typedef BGLML::TorusDirectMessage<BGLML::DetermRoutePacket> DetermMsg;
/* #include "BGLML/templates/TorusDynamicMessage.h" 
typedef BGLML::TorusDynamicMessage<BGLML::DynamicRoutePacket> DynamicMsg; */

#define EAGER 1

#include "bglml/BLMPI_EagerProtocol.h"
#include "bglml/BLMPI_RzvProtocol.h"

#if 0
#define BGL_DEBUG CmiPrintf
#else
#define BGL_DEBUG // CmiPrintf 
#endif

inline char *ALIGN_16(char *p){
  return((char *)((((unsigned long)p)+0xf)&0xfffffff0));
}

static char *_msgr_buf = NULL;
static BLMPI_Messager_t* _msgr = NULL;
static void ** _recvArray;
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
#endif /* CMK_SMP */

#define BROADCAST_SPANNING_FACTOR      4

#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_GET_CYCLE(msg)               ((CmiMsgHeaderBasic *)msg)->root

#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank
#define CMI_MAGIC(msg)                   ((CmiMsgHeaderBasic *)msg)->magic

/* FIXME: need a random number that everyone agrees ! */
#define CHARM_MAGIC_NUMBER               126

#if !CMK_OPTIMIZE
static int checksum_flag = 0;
extern "C" unsigned char computeCheckSum(unsigned char *data, int len);
#define CMI_SET_CHECKSUM(msg, len)      \
        if (checksum_flag)  {   \
          ((CmiMsgHeaderBasic *)msg)->cksum = 0;        \
          ((CmiMsgHeaderBasic *)msg)->cksum = computeCheckSum((unsigned char*)msg, len);        \
        }
#define CMI_CHECK_CHECKSUM(msg, len)    \
        if (checksum_flag)      \
          if (computeCheckSum((unsigned char*)msg, len) != 0)   \
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

int 		  _Cmi_numpes;
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_numnodes;  /* Total number of address spaces */
static int        Cmi_nodestart; /* First processor in this address space */
CpvDeclare(void*, CmiLocalQueue);

int 		  idleblock = 0;

#include "machine-smp.c"
CsvDeclare(CmiNodeState, NodeState);
#include "immediate.c"

#if !CMK_SMP
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
#endif  /* !CMK_SMP */

/*Add a message to this processor's receive queue, pe is a rank */
static void CmiPushPE(int pe,void *msg)
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
  PCQueuePush(cs->recv,(char *)msg);
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
#endif /* CMK_NODE_QUEUE_AVAILABLE */

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

extern "C" void ConverseCommonInit(char **argv);
extern "C" void ConverseCommonExit(void);
extern "C" void CthInit(char **argv);

static int PumpMsgs();
static void AdvanceCommunications();
static void ConverseRunPE(int everReturn);
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);

void SendSpanningChildren(int size, char *msg);
void SendHypercube(int size, char *msg);

typedef struct msg_list {
  BGLQuad info;
  char *msg;
  char *send_buf;
} SMSG_LIST;

#define MAX_QLEN 32
static int msgQueueLen = 0;
static int request_max;
static int no_outstanding_sends=0; /*FLAG: consume outstanding Isends in scheduler loop*/

static int outstanding_recvs = 0;

static int Cmi_dim;	/* hypercube dim of network */

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
  CmiIdleState *s=(CmiIdleState *)CmiAlloc(sizeof(CmiIdleState));
  s->sleepMs=0;
  s->nIdles=0;
  s->cs=CmiGetState();
  return s;
}

typedef struct ProcState {
/* PCQueue      sendMsgBuf; */      /* per processor message sending queue */
CmiNodeLock  recvLock;              /* for cs->recv */
} ProcState;

static ProcState  *procState;

struct CmiMsgInfo{
#if EAGER
  BLMPI_Eager_Recv_t recv;
#else
  BLMPI_Rzv_Recv_t recv;
#endif
  char* msgptr;
  int sndlen;
};

/* send done callback: sets the smsg entry to done */
static void send_done(void *data){
  SMSG_LIST *msg_tmp = (SMSG_LIST *)ALIGN_16((char *)data);
  CmiFree(msg_tmp->msg);
  msg_tmp->msg=NULL;
  CmiFree(msg_tmp->send_buf);
  msg_tmp->send_buf=NULL;
  CmiFree(data);
  data=NULL;
  msgQueueLen--;
}

/* recv done callback: push the recved msg to recv queue */
static void recv_done(void *clientdata){
  struct CmiMsgInfo *info = (struct CmiMsgInfo *)ALIGN_16((char *)clientdata);
  char* msg = info->msgptr;
  int sndlen = info->sndlen;

  /* then we do what PumpMsgs used to do: 
   * push msg to recv queue */
  CMI_CHECK_CHECKSUM(msg, sndlen);
  if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
    CmiFree(clientdata);
    CmiAbort("Charm++ Warning: Non Charm++ Message Received. \n");
    return;
  }
#if CMK_NODE_QUEUE_AVAILABLE
  if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
    CmiPushNode(msg);
  else
#endif
    CmiPushPE(CMI_DEST_RANK(msg), (void *)msg);

#if CMK_BROADCAST_SPANNING_TREE
  if (CMI_BROADCAST_ROOT(msg)){
    SendSpanningChildren(sndlen, msg);
  }
#elif CMK_BROADCAST_HYPERCUBE
  if (CMI_GET_CYCLE(msg))
    SendHypercube(sndlen, msg);
#endif

  CmiFree(clientdata);
  outstanding_recvs--;
}

/* first packet recv callback, gets recv_done for the whole msg */
#if EAGER
BLMPI_Eager_Recv_t * first_pkt_recv_done (
#else
BLMPI_Rzv_Recv_t * first_pkt_recv_done   (
#endif
					  const BGLQuad    * msginfo,
					  unsigned 	     senderrank,
					  const unsigned     sndlen,
					  unsigned         * rcvlen,
					  char            ** buffer,
					  void           (** cb_done)(void *),
					  void            ** clientdata
					 )
{
  outstanding_recvs++;
  /* printf ("Receiving %d bytes\n", sndlen); */
  *rcvlen = sndlen>0?sndlen:1;	/* to avoid malloc(0) which might return NULL */
  *buffer = (char *)CmiAlloc(sndlen);
  *cb_done = recv_done;
  *clientdata = CmiAlloc(sizeof(struct CmiMsgInfo)+16);
  struct CmiMsgInfo *info = (struct CmiMsgInfo *)ALIGN_16((char *)(*clientdata));
  info->msgptr = *buffer;
  info->sndlen = sndlen;

#if EAGER
  return (BLMPI_Eager_Recv_t *)(&(info->recv));
#else
  return (BLMPI_Rzv_Recv_t *)(&(info->recv));
#endif
}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret){
  int n, i;

  _msgr_buf = (char *)CmiAlloc(sizeof(BLMPI_Messager_t)+16);
  _msgr = (BLMPI_Messager_t *)ALIGN_16(_msgr_buf);
  BLMPI_Messager_Init(_msgr, BGL_AppMutexs, BGL_AppBarriers);

  _Cmi_numnodes = BLMPI_Messager_size(_msgr);
  _Cmi_mynode = BLMPI_Messager_rank(_msgr);

  /* Eager protocol init */
  _recvArray = (void **) CmiAlloc (sizeof(void *) * _Cmi_numnodes);
#if EAGER
  BLMPI_Eager_Init(_msgr, first_pkt_recv_done, _recvArray, 11, 12);
#else
  BLMPI_Rzv_Init(_msgr, first_pkt_recv_done, 11, 12, 3, 13);
#endif

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
      CmiPrintf("Charm++: Will%s consume outstanding sends in scheduler loop\n",        no_outstanding_sends?"":" not");
  }

  _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
  Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
  Cmi_argvcopy = CmiCopyArgs(argv);
  Cmi_argv = argv; Cmi_startfn = fn; Cmi_usrsched = usched;

  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=_Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;

  request_max=MAX_QLEN;
  CmiGetArgInt(argv,"+requestmax",&request_max);
 
  /* checksum flag */
  if (CmiGetArgFlag(argv,"+checksum")) {
#if !CMK_OPTIMIZE
    checksum_flag = 1;
    if (_Cmi_mynode == 0) CmiPrintf("Charm++: CheckSum checking enabled! \n");
#else
    if (_Cmi_mynode == 0) CmiPrintf("Charm++: +checksum ignored in optimized version! \n");
#endif
  }

  CsvInitialize(CmiNodeState, NodeState);
  CmiNodeStateInit(&CsvAccess(NodeState));

  procState = (ProcState *)CmiAlloc((_Cmi_mynodesize+1) * sizeof(ProcState));
  for (i=0; i<_Cmi_mynodesize+1; i++) {
/*    procState[i].sendMsgBuf = PCQueueCreate();   */
    procState[i].recvLock = CmiCreateLock();
  }

  /* Network progress function is used to poll the network when for
     messages. This flushes receive buffers on some  implementations*/
  networkProgressPeriod = 0;
  CmiGetArgInt(argv, "+networkProgressPeriod", &networkProgressPeriod);

  CmiStartThreads(argv);
  ConverseRunPE(initret);
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

static void CommunicationServer(int sleepTime){
  int static count=0;

  SendMsgBuf(); 
  AdvanceCommunications();
/*
  if (count % 10000000==0) MACHSTATE(3, "} Exiting CommunicationServer.");
*/
  if (inexit == CmiMyNodeSize()) {
    MACHSTATE(2, "CommunicationServer exiting {");
#if 0
    while(!MsgQueueEmpty() || !CmiAllAsyncMsgsSent() || !RecvQueueEmpty()) {
#endif
    while(!MsgQueueEmpty() || !CmiAllAsyncMsgsSent()) {
      SendMsgBuf(); 
      AdvanceCommunications();
    }
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyNode() == 0){
      CmiPrintf("End of program\n");
    }
#endif
    MACHSTATE(2, "} CommunicationServer EXIT");
    exit(0);   
  }
}
#endif /* CMK_SMP */

static void CommunicationServerThread(int sleepTime){
#if CMK_SMP
  CommunicationServer(sleepTime);
#endif
#if CMK_IMMEDIATE_MSG
  CmiHandleImmediate();
#endif
}

void ConverseExit(void){
/* #if ! CMK_SMP */
/* we don't have async send yet
  while(!CmiAllAsyncMsgsSent()) {
    AdvanceCommunications();
  }
*/
  while(msgQueueLen) {
    AdvanceCommunications();
  }

  ConverseCommonExit();

  CmiFree(procState);
  CmiFree(_msgr_buf);
  CmiFree(_recvArray); 
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
  if (CmiMyPe() == 0){
    CmiPrintf("End of program\n");
  }
#endif

  exit(0);
}

/* exit() called on any node would abort the whole program */
void CmiAbort(const char * message){
  CmiError("------------- Processor %d Exiting: Called CmiAbort ------------\n"
        "{snd:%d,rcv:%d} Reason: %s\n",CmiMyPe(),msgQueueLen,outstanding_recvs,message);
  CmiPrintStackTrace(0);
  exit(-1);
}

void *CmiGetNonLocal(){
  static int count=0;
  CmiState cs = CmiGetState();
  void *msg;
  CmiIdleLock_checkMessage(&cs->idle);
  /* although it seems that lock is not needed, I found it crashes very often
     on mpi-smp without lock */

  AdvanceCommunications();
  
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
#if !CMK_SMP
  if (no_outstanding_sends) {
    while (msgQueueLen>0) {
      AdvanceCommunications();
    }
  }
  
  if(!msg) {
    AdvanceCommunications();
    return PCQueuePop(cs->recv);
/*
    if (PumpMsgs())
      return  PCQueuePop(cs->recv);
    else
      return 0;
*/
  }
#endif /* !CMK_SMP */
  return msg;
}

static void CmiSendSelf(char *msg){
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

/* The general free send function
 * Send is synchronous, and free msg after posted
 */
void  CmiGeneralFreeSend(int destPE, int size, char* msg){
  CmiState cs = CmiGetState();
  if(destPE==cs->pe){
    CmiSendSelf(msg);
    return;
  }
 
  void *msg_tmp_buf = CmiAlloc(sizeof(SMSG_LIST)+16); 
  SMSG_LIST *msg_tmp = (SMSG_LIST *)ALIGN_16((char *)msg_tmp_buf);
  msg_tmp->msg = msg;

#if EAGER
  msg_tmp->send_buf = (char *) CmiAlloc (sizeof(BLMPI_Eager_Send_t)+16);
  BLMPI_Eager_Send_t *send = (BLMPI_Eager_Send_t *)ALIGN_16(msg_tmp->send_buf);
#else
  msg_tmp->send_buf = (char *) CmiAlloc (sizeof(BLMPI_Rzv_Send_t)+16);
  BLMPI_Rzv_Send_t *send = (BLMPI_Rzv_Send_t *)ALIGN_16(msg_tmp->send_buf);
#endif

  CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
  CMI_SET_CHECKSUM(msg, size);

  CQdCreate(CpvAccess(cQdState), 1);
#if EAGER
  BLMPI_Eager_Send(send, _msgr, &(msg_tmp->info), msg, size, destPE, send_done,(void *)msg_tmp_buf);
#else
  BLMPI_Rzv_Send(send, _msgr, &(msg_tmp->info), msg, size, destPE, send_done,(void *)msg_tmp_buf);
#endif
  msgQueueLen++;

  AdvanceCommunications();
}

void CmiSyncSendFn(int destPE, int size, char *msg){
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiFreeSendFn(destPE,size,copymsg);
}
void CmiFreeSendFn(int destPE, int size, char *msg){
  CMI_SET_BROADCAST_ROOT(msg,0);
  CmiGeneralFreeSend(destPE,size,msg);
}

/* same as CmiSyncSendFn, but don't set broadcast root in msg header */
void CmiSyncSendFn1(int destPE, int size, char *msg)
{
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiGeneralFreeSend(destPE,size,copymsg);
}

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg)
{
  CmiState cs = CmiGetState();
  int startpe = CMI_BROADCAST_ROOT(msg)-1;
  int i;
  CmiAssert(startpe>=0 && startpe<_Cmi_numpes);
  int dist = cs->pe-startpe;
  if(dist<0) dist+=_Cmi_numpes;
  for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
    int p = BROADCAST_SPANNING_FACTOR*dist + i;
    if (p > _Cmi_numpes - 1) break;
    p += startpe;
    p = p%_Cmi_numpes;
    CmiAssert(p>=0 && p<_Cmi_numpes && p!=cs->pe);
    CmiSyncSendFn1(p, size, msg);
  }
}

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

void CmiSyncBroadcastFn(int size, char *msg){
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiFreeBroadcastFn(size,copymsg);
}
void CmiFreeBroadcastFn(int size, char *msg){
  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);
  CmiFree(msg);
#elif CMK_BROADCAST_HYPERCUBE
  CMI_SET_CYCLE(msg, 0);
  SendHypercube(size, msg);
  CmiFree(msg);
#else
  int i;
  for ( i=cs->pe+1; i<_Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg);
  for ( i=0; i<cs->pe; i++ ) 
    CmiSyncSendFn(i,size,msg);
  CmiFree(msg);
#endif
}

void CmiSyncBroadcastAllFn(int size, char *msg){
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiFreeBroadcastAllFn(size,copymsg);
}
void CmiFreeBroadcastAllFn(int size, char *msg){
  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);
  CmiSyncSendFn(cs->pe,size,msg);
  CmiFree(msg);
#elif CMK_BROADCAST_HYPERCUBE
  CmiSyncSendFn(cs->pe,size,msg);
  CMI_SET_CYCLE(msg, 0);
  SendHypercube(size, msg);
  CmiFree(msg);
#else
  int i ;
  for ( i=0; i<_Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg);
  CmiFree(msg);
#endif
}

/* Poll the network for messages */
/* Poll the network and when a message arrives and insert this arrived
   message into the local queue. For SMP this message would have to be
   inserted into the thread's queue with the correct rank **/
/* Pump messages is called when the processor goes idle */
static int PumpMsgs(void){
  int flag = BLMPI_Messager_advance(_msgr);
  while(BLMPI_Messager_advance(_msgr)) ;
  return flag; 
}
static void AdvanceCommunications(void){
  while(outstanding_recvs){
    BLMPI_Messager_advance(_msgr);
  }
  while(msgQueueLen>request_max){
    BLMPI_Messager_advance(_msgr);
  }
  while(BLMPI_Messager_advance(_msgr)) ;
}

void CmiNotifyIdle(){
//  if (!PumpMsgs() && idleblock) AdvanceCommunications();
  AdvanceCommunications();
}

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  s->sleepMs=0;
  s->nIdles=0;
}

static void CmiNotifyStillIdle(CmiIdleState *s)
{ 
#if ! CMK_SMP
  AdvanceCommunications();
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

/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/************ Recommended routines ***********************/
/************ You dont have to implement these but they are supported
 in the converse syntax and some rare programs may crash. But most
 programs dont need them. *************/

CmiCommHandle CmiAsyncSendFn(int, int, char *){
  CmiAbort("CmiAsyncSendFn not implemented.");
  return (CmiCommHandle) 0;
}
CmiCommHandle CmiAsyncBroadcastFn(int, char *){
  CmiAbort("CmiAsyncBroadcastFn not implemented.");
  return (CmiCommHandle) 0;
}
CmiCommHandle CmiAsyncBroadcastAllFn(int, char *){
  CmiAbort("CmiAsyncBroadcastAllFn not implemented.");
  return (CmiCommHandle) 0;
}

int           CmiAsyncMsgSent(CmiCommHandle handle){
  CmiAbort("CmiAsyncMsgSent not implemented.");
  return 0;
}
void          CmiReleaseCommHandle(CmiCommHandle handle){
  CmiAbort("CmiReleaseCommHandle not implemented.");
}


/*==========================================================*/
/*==========================================================*/
/*==========================================================*/

/* Optional routines which could use common code which is shared with
   other machine layer implementations. */

/* MULTICAST/VECTOR SENDING FUNCTIONS

 * In relations to some flags, some other delivery functions may be needed.
 */

#if ! CMK_MULTICAST_LIST_USE_COMMON_CODE
void          CmiSyncListSendFn(int npes, int *pes, int size, char *msg){
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiFreeListSendFn(npes, pes, size, msg);
}

void          CmiFreeListSendFn(int npes, int *pes, int size, char *msg){
  CMI_SET_BROADCAST_ROOT(msg,0);
  for(int i=0;i<npes;i++){
    if(pes[i] == CmiMyPe()){
      CmiSendSelf(msg);
    }else{
#if 0
      CmiReference(msg);
      CmiGeneralFreeSend(pes[i],size,msg);
#else
      CmiSyncSendFn(pes[i],size,msg);
#endif
    }
  }
  CmiFree(msg);
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int size, char *msg){
  CmiAbort("CmiAsyncListSendFn not implemented.");
  return (CmiCommHandle) 0;
}
#endif

#if ! CMK_MULTICAST_GROUP_USE_COMMON_CODE
void          CmiSyncMulticastFn(CmiGroup, int, char*);
CmiCommHandle CmiAsyncMulticastFn(CmiGroup, int, char*);
void          CmiFreeMulticastFn(CmiGroup, int, char*);
#endif

#if ! CMK_VECTOR_SEND_USES_COMMON_CODE
void          CmiSyncVectorSend(int, int, int *, char **);
CmiCommHandle CmiAsyncVectorSend(int, int, int *, char **);
void          CmiSyncVectorSendAndFree(int, int, int *, char **);
#endif


/** NODE SENDING FUNCTIONS

 * If there is a node queue, and we consider also nodes as entity (tipically in
 * SMP versions), these functions are needed.
 */

#if CMK_NODE_QUEUE_AVAILABLE

void          CmiSyncNodeSendFn(int, int, char *);
CmiCommHandle CmiAsyncNodeSendFn(int, int, char *);
void          CmiFreeNodeSendFn(int, int, char *);

void          CmiSyncNodeBroadcastFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastFn(int, char *);
void          CmiFreeNodeBroadcastFn(int, char *);

void          CmiSyncNodeBroadcastAllFn(int, char *);
CmiCommHandle CmiAsyncNodeBroadcastAllFn(int, char *);
void          CmiFreeNodeBroadcastAllFn(int, char *);

#endif


/** GROUPS DEFINITION

 * For groups of processors (establishing and managing) some more functions are
 * needed, they also con be found in common code (convcore.c) or here.
 */

#if ! CMK_MULTICAST_DEF_USE_COMMON_CODE
void     CmiGroupInit();
CmiGroup CmiEstablishGroup(int npes, int *pes);
void     CmiLookupGroup(CmiGroup grp, int *npes, int **pes);
#endif


/** MESSAGE DELIVERY FUNCTIONS

 * In order to deliver the messages to objects (either converse register
 * handlers, or charm objects), a scheduler is needed. The one implemented in
 * convcore.c can be used, or a new one can be implemented here. At present, all
 * machines use the default one, exept sim-linux.

 * If the one in convcore.c is used, still one function is needed.
 */

#if CMK_CMIDELIVERS_USE_COMMON_CODE /* use the default one */

/* already declared
CpvDeclare(void*, CmiLocalQueue);
*/

#elif /* reimplement the scheduler and delivery */

void CsdSchedulerState_new(CsdSchedulerState_t *state);
void *CsdNextMessage(CsdSchedulerState_t *state);
int  CsdScheduler(int maxmsgs);

void CmiDeliversInit();
int  CmiDeliverMsgs(int maxmsgs);
void CmiDeliverSpecificMsg(int handler);

#endif


/** SHARED VARIABLES DEFINITIONS

 * In relation to which CMK_SHARED_VARS_ flag is set, different
 * functions/variables need to be defined and initialized correctly.
 */

#if CMK_SHARED_VARS_UNAVAILABLE /* Non-SMP version of shared vars. */

#if 0 /* already defined */
int _Cmi_mype;
int _Cmi_numpes;
int _Cmi_myrank; /* Normally zero; only 1 during SIGIO handling */
#endif

void CmiMemLock();
void CmiMemUnlock();

#endif

#if CMK_SHARED_VARS_POSIX_THREADS_SMP /*Used by the net-*-smp versions*/

int _Cmi_numpes;
int _Cmi_mynodesize;
int _Cmi_mynode;
int _Cmi_numnodes;

int CmiMyPe();
int CmiMyRank();
int CmiNodeFirst(int node);
int CmiNodeSize(int node);
int CmiNodeOf(int pe);
int CmiRankOf(int pe);

/* optional, these functions are implemented in "machine-smp.c", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier(void);
void CmiNodeAllBarrier(void);
CmiNodeLock CmiCreateLock();
void CmiDestroyLock(CmiNodeLock lock);

#endif

/* NOT VERY USEFUL */
#if CMK_SHARED_VARS_EXEMPLAR /* Used only by HP Exemplar version */

int _Cmi_numpes;
int _Cmi_mynodesize;

void CmiMemLock();
void CmiMemUnlock();
void *CmiSvAlloc(int);

/* optional, these functions are implemented in "machine-smp.c", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier(void);
CmiNodeLock CmiCreateLock(void);

#endif

/* NOT VERY USEFUL */
#if CMK_SHARED_VARS_UNIPROCESSOR /*Used only by uth- and sim- versions*/

int _Cmi_mype;
int _Cmi_numpes;

void         CmiLock(CmiNodeLock lock);
void         CmiUnlock(CmiNodeLock lock);
int          CmiTryLock(CmiNodeLock lock);

/* optional, these functions are implemented in "machine-smp.c", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier();
void CmiNodeAllBarrier();
CmiNodeLock  CmiCreateLock(void);
void         CmiDestroyLock(CmiNodeLock lock);

#endif

/* NOT VERY USEFUL */
#if CMK_SHARED_VARS_PTHREADS /*Used only by origin-pthreads*/

int CmiMyPe();
int _Cmi_numpes;

void CmiMemLock();
void CmiMemUnlock();

void         CmiLock(CmiNodeLock lock);
void         CmiUnlock(CmiNodeLock lock);
int          CmiTryLock(CmiNodeLock lock);

/* optional, these functions are implemented in "machine-smp.c", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier();
void CmiNodeAllBarrier();
CmiNodeLock  CmiCreateLock(void);
void         CmiDestroyLock(CmiNodeLock lock);

#endif

/* NOT VERY USEFUL */
#if CMK_SHARED_VARS_NT_THREADS /*Used only by win32 versions*/

int _Cmi_numpes;
int _Cmi_mynodesize;
int _Cmi_mynode;
int _Cmi_numnodes;

int CmiMyPe();
int CmiMyRank();
int CmiNodeFirst(int node);
int CmiNodeSize(int node);
int CmiNodeOf(int pe);
int CmiRankOf(int pe);

/* optional, these functions are implemented in "machine-smp.c", so including
   this file avoid the necessity to reimplement them.
 */
void CmiNodeBarrier(void);
void CmiNodeAllBarrier(void);
CmiNodeLock CmiCreateLock(void);
void CmiDestroyLock(CmiNodeLock lock);

#endif


/** TIMERS DEFINITIONS

 * In relation to what CMK_TIMER_USE_ is selected, some * functions may need to
 * be implemented.
 */

/* If all the CMK_TIMER_USE_ are set to 0, the following timer functions are
   needed. */

void   CmiTimerInit();
double CmiTimer();
double CmiWallTimer();
double CmiCpuTimer();
int    CmiTimerIsSynchronized();

/** PRINTF FUNCTIONS

 * Default code is provided in convcore.c but for particular architectures they
 * can be reimplemented. At present only net- versions reimplement them.

 */

#if CMK_CMIPRINTF_IS_A_BUILTIN

void CmiPrintf(const char *, ...);
void CmiError(const char *, ...);
int  CmiScanf(const char *, ...);

#endif


/** SPANNING TREE

 * During some working operations (such as quiescence detection), spanning trees
 * are used. Default code in convcore.c can be used, or a new definition can be
 * implemented here.
 */

#if ! CMK_SPANTREE_USE_COMMON_CODE

int      CmiNumSpanTreeChildren(int) ;
int      CmiSpanTreeParent(int) ;
void     CmiSpanTreeChildren(int node, int *children);

int      CmiNumNodeSpanTreeChildren(int);
int      CmiNodeSpanTreeParent(int) ;
void     CmiNodeSpanTreeChildren(int node, int *children) ;

#endif



/** IMMEDIATE MESSAGES

 * If immediate messages are supported, the following function is needed. There
 * is an exeption if the machine progress is also defined (see later for this).

 * Moreover, the file "immediate.c" should be included, otherwise all its
 * functions and variables have to be redefined.
*/

#if CMK_CCS_AVAILABLE

#include "immediate.c"

#if ! CMK_MACHINE_PROGRESS_DEFINED /* Hack for some machines */
void CmiProbeImmediateMsg();
#endif

#endif


/** MACHINE PROGRESS DEFINED

 * Some machines (like BlueGene/L) do not have coprocessors, and messages need
 * to be pulled out of the network manually. For this reason the following
 * functions are needed. Notice that the function "CmiProbeImmediateMsg" must
 * not be defined anymore.
 */

#if CMK_MACHINE_PROGRESS_DEFINED

CpvDeclare(int, networkProgressCount);
int  networkProgressPeriod;

void CmiMachineProgressImpl()
{
#if !CMK_SMP
    AdvanceCommunications();
#if CMK_IMMEDIATE_MSG
    CmiHandleImmediate();
#endif
#else
    /*Not implemented yet. Communication server does not seem to be
      thread safe */
    /* CommunicationServerThread(0); */
#endif
}

#endif


/* Dummy implementation */
extern "C" void CmiBarrier()
{
}

