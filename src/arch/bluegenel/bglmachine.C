#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "machine.h"
#include "converse.h"
#include "pcqueue.h"
#include <stack>

#include "rts.h"
#include "bgml.h"

#if CMK_PERSISTENT_COMM
#include "persist_impl.h"
#endif

#if 0
#define BGL_DEBUG CmiPrintf
#else
#define BGL_DEBUG // CmiPrintf
#endif

extern "C"
void        BG2S_UnOrdered_Send     (BG2S_t               * request,
				     const BGML_Callback_t* cb_info,
				     const BGQuad         * msginfo,
				     const char           * sndbuf,
				     unsigned               sndlen,
				     unsigned               destrank);

int phscount;
int vnpeer = -1;

inline char *ALIGN_16(char *p){
  return((char *)((((unsigned long)p)+0xf)&0xfffffff0));
}

PCQueue message_q;                   //queue to receive incoming messages
PCQueue broadcast_q;                 //queue to send broadcast messages

#define PROGRESS_PERIOD 8
#define PROGRESS_CYCLES 4000           //10k cycles

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

#define MAX_OUTSTANDING  1024
#define MAX_POSTED 64
#define MAX_QLEN 1024
#define MAX_BYTES 1000000

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
          if (computeCheckSum((unsigned char*)msg, len) != 0)  { \
            printf("\n\n------------------------------\n\nReceiver %d size %d:", CmiMyPe(), len); \
            for(int count = 0; count < len; count++) { \
                printf("%2x", msg[count]);                 \
            }                                             \
            printf("------------------------------\n\n"); \
            CmiAbort("Fatal error: checksum doesn't agree!\n"); \
          }
#else
#define CMI_SET_CHECKSUM(msg, len)
#define CMI_CHECK_CHECKSUM(msg, len)
#endif

#define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);

#if CMK_BROADCAST_HYPERCUBE
#  define CMI_SET_CYCLE(msg, cycle)  CMI_GET_CYCLE(msg) = (cycle);
#else
#  define CMI_SET_CYCLE(msg, cycle)
#endif

int               _Cmi_numpes;
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_numnodes;  /* Total number of address spaces */
static int        Cmi_nodestart; /* First processor in this address space */
CpvDeclare(void*, CmiLocalQueue);

int               idleblock = 0;

#include "machine-smp.c"
CsvDeclare(CmiNodeState, NodeState);
#include "immediate.c"

static inline void AdvanceCommunications(int max_out = MAX_OUTSTANDING);
static int outstanding_recvs = 0;


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

int received_immediate = 0;

/*Add a message to this processor's receive queue, pe is a rank */
void CmiPushPE(int pe,void *msg)
{
  CmiState cs = CmiGetStateN(pe);
  MACHSTATE2(3,"Pushing message into rank %d's queue %p{",pe, cs->recv);
#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
    /**(CmiUInt2 *)msg = pe;*/
    received_immediate = 1;
    CMI_DEST_RANK(msg) = pe;
    CmiPushImmediateMsg(msg);
    return;
  }
#endif
#if CMK_SMP
  CmiLock(procState[pe].recvLock);
#endif

  PCQueuePush(cs->recv,(char *)msg);
  //printf("%d: {%d} PCQueue length = %d, msg = %x\n", CmiMyPe(), bgx_in_interrupt, PCQueueLength(cs->recv), msg);
  
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


static int msgQueueLen = 0;
static int msgQBytes = 0;
static int numPosted = 0;

static int request_max;
static int maxMessages;
static int maxBytes;
static int no_outstanding_sends=0; /*FLAG: consume outstanding Isends in scheduler loop*/
static int progress_cycles;

static int Cmi_dim;     /* hypercube dim of network */

static char     **Cmi_argv;
static char     **Cmi_argvcopy;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */

extern "C" void ConverseCommonInit(char **argv);
extern "C" void ConverseCommonExit(void);
extern "C" void CthInit(char **argv);

static inline void SendMsgsUntil(int, int);
static void ConverseRunPE(int everReturn);
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);

void SendSpanningChildren(int size, char *msg);
void SendHypercube(int size, char *msg);

typedef struct msg_list {
  BG2S_t             send;
  BGQuad             info;
  BGML_Callback_t    cb;
  char              *msg;
  int                size;
  int                destpe;
  int               *pelist;

#if CMK_PERSISTENT_COMM
  PersistentHandle phs;
  int phscount;
  int phsSize;
#endif
} SMSG_LIST;


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

/* send done callback: sets the smsg entry to done */
static void send_done(void *data){
  SMSG_LIST *msg_tmp = (SMSG_LIST *)(data);
  CmiFree(msg_tmp->msg);
  msgQBytes -= msg_tmp->size;
  free(data);

  msgQueueLen--;
  numPosted --;
}

/* send done callback: sets the smsg entry to done */
static void send_multi_done(void *data){
  SMSG_LIST *msg_tmp = (SMSG_LIST *)(data);
  CmiFree(msg_tmp->msg);
  free(msg_tmp->pelist);
  free(data);

  msgQBytes -= msg_tmp->size;
  msgQueueLen --;
  numPosted --;
}


//Called on receiving a persistent message
void persist_recv_done(void *clientdata) {

  char *msg = (char *) clientdata;
  int sndlen = ((CmiMsgHeaderBasic *) msg)->size;

  ///  printf("[%d] persistent receive of size %d\n", CmiMyPe(), sndlen);
  
  //Cannot broadcast Persistent message

  CMI_CHECK_CHECKSUM(msg, sndlen);
  if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
    CmiAbort("Charm++ Warning: Non Charm++ Message Received. \n");
    return;
  }
  
  CmiReference(msg);

#if CMK_NODE_QUEUE_AVAILABLE
  if (CMI_DEST_RANK(msg) == DGRAM_NODEMESSAGE)
    CmiPushNode(msg);
  else
#endif
    CmiPushPE(CMI_DEST_RANK(msg), (void *)msg);
}

/* recv done callback: push the recved msg to recv queue */
static void recv_done(void *clientdata){

  char *msg = (char *) clientdata;
  int sndlen = ((CmiMsgHeaderBasic *) msg)->size;
  
  /* then we do what PumpMsgs used to do:
   * push msg to recv queue */
  
  CMI_CHECK_CHECKSUM(msg, sndlen);
  if (CMI_MAGIC(msg) != CHARM_MAGIC_NUMBER) { /* received a non-charm msg */
    CmiAbort("Charm++ Warning: Non Charm++ Message Received. \n");
    return;
  }

#if CMK_NODE_QUEUE_AVAILABLE
  if (CMI_DEST_RANK(msg) == DGRAM_NODEMESSAGE)
    CmiPushNode(msg);
  else
#endif
    CmiPushPE(CMI_DEST_RANK(msg), (void *)msg);

#if CMK_BROADCAST_SPANNING_TREE | CMK_BROADCAST_HYPERCUBE
  if(CMI_BROADCAST_ROOT(msg) != 0) {
    //printf ("%d: Receiving bcast message %d bytes\n", CmiMyPe(), sndlen);
    PCQueuePush(broadcast_q, msg);
  }
#endif


  //rts_dcache_evict_normal();

  outstanding_recvs --;
}


void     short_pkt_recv (const BGQuad     * info,
			 unsigned           senderrank,
			 const char       * buffer,
			 const unsigned     sndlen)
{
  outstanding_recvs ++;
  int alloc_size = sndlen;
  
  //printf ("%d: Receiving short message %d bytes\n", CmiMyPe(), sndlen);
  
  char * new_buffer = (char *)CmiAlloc(alloc_size);
  memcpy (new_buffer, buffer, sndlen);
  recv_done (new_buffer);
}


BG2S_t * first_pkt_recv_done (const BGQuad      * info,
			      unsigned            senderrank,
			      const unsigned      sndlen,
			      unsigned          * rcvlen,
			      char             ** buffer,
			      BGML_Callback_t   * cb 
			      )
{
  outstanding_recvs ++;
  int alloc_size = sndlen + sizeof(BG2S_t) + 16;
  
  //printf ("%d: {%d} Receiving message %d bytes\n", CmiMyPe(), bgx_in_interrupt, sndlen);

  /* printf ("Receiving %d bytes\n", sndlen); */
  *rcvlen = sndlen>0?sndlen:1;  /* to avoid malloc(0) which might
                                   return NULL */

  *buffer = (char *)CmiAlloc(alloc_size);
  cb->function = recv_done;
  cb->clientdata = *buffer;

  return (BG2S_t *) ALIGN_16(*buffer + sndlen);
}

void sendBroadcastMessages() __attribute__((noinline));

void sendBroadcastMessages() {
  while(!PCQueueEmpty(broadcast_q)) {
    char *msg = (char *) PCQueuePop(broadcast_q);
    
#if CMK_BROADCAST_SPANNING_TREE
    SendSpanningChildren(((CmiMsgHeaderBasic *) msg)->size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    SendHypercube(((CmiMsgHeaderBasic *) msg)->size, msg);
#endif
    
    //CMI_CHECK_CHECKSUM(msg, (((CmiMsgHeaderBasic *) msg)->size));
  }
}

unsigned int *ranklist;

// #include "BGLTorus.h"
// CpvDeclare(BGLTorusManager*, tmanager);

BGTsC_t        barrier;

CpvDeclare(unsigned, networkProgressCount);
int  networkProgressPeriod;

// -----------------------------------------
// Rectangular broadcast implementation 
// -----------------------------------------

typedef struct rectbcast_msg {
  BGTsRC_t           request;
  BGML_Callback_t    cb;
  char              *msg;
} RectBcastInfo;


static void bcast_done (void *data) {  
  RectBcastInfo *rinfo = (RectBcastInfo *) data;  
  CmiFree (rinfo->msg);
  free (rinfo);
}


typedef void * (*requestFnType) (unsigned);

requestFnType requestFn;

static  void *   getRectBcastRequest (unsigned comm) {
  return requestFn (comm);
}


static  void *  bcast_recv     (unsigned               root,
				unsigned               comm,
				const unsigned         sndlen,
				unsigned             * rcvlen,
				char                ** rcvbuf,
				BGML_Callback_t      * const cb) {
  
  //  printf ("%d: in bcast recv \n", CmiMyPe());

  int alloc_size = sndlen + sizeof(BGTsRC_t) + 16;
  
  *rcvlen = sndlen>0?sndlen:1;  /* to avoid malloc(0) which might
                                   return NULL */
  
  *rcvbuf       =  (char *)CmiAlloc(alloc_size);
  cb->function  =   recv_done;
  cb->clientdata = *rcvbuf;

  BGTsRC_t *request = (BGTsRC_t *) ALIGN_16 (*rcvbuf + sndlen);
  memset (request, 0, sizeof(BGTsRC_t));
  return request;
}


extern "C"  
void bgl_machine_RectBcast (unsigned                 commid,
			    char                   * sndbuf,
			    unsigned                 sndlen) 
{
  CMI_SET_BROADCAST_ROOT(sndbuf,0);  
  CMI_MAGIC(sndbuf) = CHARM_MAGIC_NUMBER;
  ((CmiMsgHeaderBasic *)sndbuf)->size = sndlen;  
  CMI_SET_CHECKSUM(sndbuf, sndlen);
  
  RectBcastInfo *rinfo  =   (RectBcastInfo *) malloc (sizeof(RectBcastInfo)); 

  rinfo->cb.function    =   bcast_done;
  rinfo->cb.clientdata  =   rinfo;
  rinfo->msg            =   sndbuf;
  
  memset (&rinfo->request, 0, sizeof(BGTsRC_t));

  BGTsRC_AsyncBcast_start (commid, &rinfo->request, &rinfo->cb, sndbuf, sndlen);
}

extern "C"   
void    *         bgl_machine_RectBcastInit  (unsigned               commID,
					      const BGTsRC_Geometry_t* geometry) {
  
  BGTsRC_t *request =  (BGTsRC_t *) malloc (sizeof (BGTsRC_t));
  memset(request, 0,  sizeof( BGTsRC_t));
  BGTsRC_AsyncBcast_init  (request, commID,  geometry);
  
  return request;
}


extern "C" 
void  bgl_machine_RectBcastConfigure (requestFnType fn) {
  requestFn = fn;
}



//--------------------------------------------------------------
//----- End Rectangular Broadcast Implementation ---------------
//--------------------------------------------------------------


void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret){
  int n, i;
  
  BGML_Messager_Init();
  BG2S_Configure (short_pkt_recv, first_pkt_recv_done, NULL);
  
  //BGTr_Configure ( 1 /* use_coprocessor  */, 
  //               1 /* protocolTreshold */, 
  //               1 /* highPrecision    */); 
  
  BGTsC_Configure (NULL, getRectBcastRequest, bcast_recv);

  _Cmi_numnodes = BGML_Messager_size();
  _Cmi_mynode = BGML_Messager_rank();
  
  message_q = PCQueueCreate();
  broadcast_q = PCQueueCreate();

  unsigned rank = BGML_Messager_rank();
  unsigned size = BGML_Messager_size();
  
  //vnpeer = BGML_Messager_vnpeerrank();
  
#if 1
  ranklist = (unsigned int *)malloc (size * sizeof(int));
  for(int count = 0; count < size; count ++)
    ranklist[count] = count;
  
  /* global barrier initialize */
  BGTsC_Barrier_init (&barrier, 0, size, ranklist);
#endif
  
  CmiBarrier();    
  CmiBarrier();    
  CmiBarrier();    

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

  // CpvInitialize(BGLTorusManager*, tmanager);
  // CpvAccess(tmanager) = new BGLTorusManager();


  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=_Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;

  request_max=MAX_POSTED;
  CmiGetArgInt(argv,"+requestmax",&request_max);
 
  maxMessages = MAX_QLEN;
  maxBytes = MAX_BYTES;

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
  networkProgressPeriod = PROGRESS_PERIOD;
  CmiGetArgInt(argv, "+networkProgressPeriod", &networkProgressPeriod);
  
  progress_cycles = PROGRESS_CYCLES;
  CmiGetArgInt(argv, "+progressCycles", &progress_cycles);

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

  CmiBarrier();

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

extern "C" void BGML_Messager_dumpTimers();

void ConverseExit(void){
  /* #if ! CMK_SMP */

  while(msgQueueLen > 0 || outstanding_recvs > 0) {
    AdvanceCommunications();
  }

  CmiBarrier();

  ConverseCommonExit();  

  if(CmiMyPe()%101 == 0)
    BGML_Messager_dumpTimers();
  
  CmiFree(procState);
  
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

  while(msgQueueLen > 0 || outstanding_recvs > 0) {
    AdvanceCommunications();
  }
  
  //CmiBarrier(); 

  assert (0);
}

void *CmiGetNonLocal(){
  static int count=0;
  CmiState cs = CmiGetState();

  void *msg = NULL;
  //CmiIdleLock_checkMessage(&cs->idle);
  /* although it seems that lock is not needed, I found it crashes very often
     on mpi-smp without lock */
  
  AdvanceCommunications();
  
  //CmiLock(procState[cs->rank].recvLock);
  
  if (PCQueueLength(cs->recv) > 0)
    msg =  PCQueuePop(cs->recv); 

  //CmiUnlock(procState[cs->rank].recvLock);
  
#if !CMK_SMP
#if BLOCKING_COMM
  if (no_outstanding_sends) {
    SendMsgsUntil(0, 0);
  }
#endif
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


#if CMK_PERSISTENT_COMM
#include "persistent.C"
#endif

inline void machineSend(SMSG_LIST *msg_tmp) {
  
  CQdCreate(CpvAccess(cQdState), 1);
  numPosted ++;

  if(msg_tmp->destpe == CmiMyPe())
    CmiAbort("Sending to self\n");
  
#if CMK_PERSISTENT_COMM
  if(msg_tmp->phs) {
    if(machineSendPersistentMsg(msg_tmp))
      return;
  }
#endif
  
  CmiAssert(msg_tmp->destpe >= 0 && msg_tmp->destpe < CmiNumPes());
  msg_tmp->cb.function     =   send_done;
  msg_tmp->cb.clientdata   =   msg_tmp;

  BG2S_UnOrdered_Send (&msg_tmp->send, &msg_tmp->cb, &msg_tmp->info, 
		       msg_tmp->msg, msg_tmp->size, msg_tmp->destpe);    
}

void sendQueuedMessages() __attribute__((noinline));

void sendQueuedMessages() {
  while(numPosted <= request_max && !PCQueueEmpty(message_q)) {
    SMSG_LIST *msg_tmp = (SMSG_LIST *)PCQueuePop(message_q);    
    machineSend(msg_tmp);
  }
}


static int enableBatchSends = 1;

/* The general free send function
 * Send is synchronous, and free msg after posted
 */
inline void  CmiGeneralFreeSend(int destPE, int size, char* msg){
  
  CmiState cs = CmiGetState();
  if(destPE==cs->pe){
    CmiSendSelf(msg);
    return;
  }
  
  if(numPosted >= 16 /*|| !enableBatchSends*/)
    SendMsgsUntil(maxMessages, maxBytes);
  
  SMSG_LIST *msg_tmp = (SMSG_LIST *) malloc(sizeof(SMSG_LIST));
  msg_tmp->destpe = destPE;
  msg_tmp->size = size;
  msg_tmp->msg = msg;
  
#if CMK_PERSISTENT_COMM
  msg_tmp->phs = phs;
  msg_tmp->phscount = phscount;
  msg_tmp->phsSize = phsSize;
#endif

  //sendQueuedMessages();
  while(numPosted <= request_max && !PCQueueEmpty(message_q)) {
    SMSG_LIST *msg_tmp = (SMSG_LIST *)PCQueuePop(message_q);    
    machineSend(msg_tmp);
  }  
  
  if(numPosted > request_max) {
    PCQueuePush(message_q, (char *)msg_tmp);
  }    
  else 
    machineSend(msg_tmp);    
  
  msgQBytes += size;
  msgQueueLen++;
  
}

extern "C"
void BGML_Multicast (BG2S_t                 * sender,
		     const BGML_Callback_t  * cb_info,
		     const BGQuad           * msginfo,
		     const char             * sndbuf,
		     unsigned                 sndlen,
		     unsigned               * destranks,
		     unsigned                 nranks);


/* Multicast message to a list of destinations
 */
inline void  machineMulticast(int npes, int *pelist, int size, char* msg){  
  CQdCreate(CpvAccess(cQdState), npes);
  
  CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
  ((CmiMsgHeaderBasic *)msg)->size = size;  
  CMI_SET_BROADCAST_ROOT(msg,0);
  CMI_SET_CHECKSUM(msg, size);
  
  SMSG_LIST *msg_tmp = (SMSG_LIST *) malloc(sizeof(SMSG_LIST));
  //memset (msg_tmp, 0, sizeof(SMSG_LIST));
  
  msg_tmp->destpe    = -1;      //multicast operation
  msg_tmp->size      = size * npes;
  msg_tmp->msg       = msg;
  msg_tmp->pelist    = pelist;
  
  msgQBytes  += size * npes;   
  msgQueueLen ++;
  numPosted  ++;
  
  msg_tmp->cb.function    =   send_multi_done;
  msg_tmp->cb.clientdata  =   msg_tmp;
  
  BGML_Multicast (&msg_tmp->send, &msg_tmp->cb, &msg_tmp->info, 
		  msg_tmp->msg, size, (unsigned *)pelist, npes); 
}
 

void CmiSyncSendFn(int destPE, int size, char *msg){
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiFreeSendFn(destPE,size,copymsg);
}

void CmiFreeSendFn(int destPE, int size, char *msg){
  CMI_SET_BROADCAST_ROOT(msg,0);
  CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
  ((CmiMsgHeaderBasic *)msg)->size = size;  
  CMI_SET_CHECKSUM(msg, size);

  CmiGeneralFreeSend(destPE,size,msg);
}

/* same as CmiSyncSendFn, but don't set broadcast root in msg header */
void CmiSyncSendFn1(int destPE, int size, char *msg)
{
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg, msg, size);

  //  asm volatile("sync" ::: "memory");

  CMI_MAGIC(copymsg) = CHARM_MAGIC_NUMBER;
  ((CmiMsgHeaderBasic *)copymsg)->size = size;  
  CMI_SET_CHECKSUM(copymsg, size);
  
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
  for (i=1; i <= BROADCAST_SPANNING_FACTOR; i++) {
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

      CmiAssert(p>=0 && p<_Cmi_numpes && p!=cs->pe);
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
  
  //printf("%d: Calling Broadcast %d\n", CmiMyPe(), size);
  /*
  printf("------------------------------\n\nSender %d :", CmiMyPe());
  
  for(int count = 0; count < size; count++) {
    printf("%2x", msg[count]);
  } 
  
  printf("------------------------------\n\n");
  */


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
  //SendMsgsUntil(0,0);
}

void CmiSyncBroadcastAllFn(int size, char *msg){
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiFreeBroadcastAllFn(size,copymsg);
}

void CmiFreeBroadcastAllFn(int size, char *msg){
  
  //printf("%d: Calling All Broadcast %d\n", CmiMyPe(), size);

  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
  CmiSyncSendFn(cs->pe,size,msg);
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);
  CmiFree(msg);
#elif CMK_BROADCAST_HYPERCUBE
  CmiSyncSendFn(cs->pe,size,msg);
  CMI_SET_CYCLE(msg, 0);
  SendHypercube(size, msg);
  CmiFree(msg);
#else
  int i ;

  CmiSyncSendFn(CmiMyPe(), size, msg);
  
  for ( i=0; i<_Cmi_numpes; i++ ) {
    if(i== CmiMyPe())
      continue;

    CmiSyncSendFn(i,size,msg);
  }
  CmiFree(msg);
#endif
  
}

static unsigned long long lastProgress = 0;

static inline void AdvanceCommunications(int max_out){
  
  if (PCQueueLength (broadcast_q) > 0)
    sendBroadcastMessages();

#if BLOCKING_COMM
  while(msgQueueLen > maxMessages && msgQBytes > maxBytes){
    while(BGML_Messager_advance()>0) ;
    sendQueuedMessages();
  }
  
  int target = outstanding_recvs - max_out;
  while(target > 0){
    while(BGML_Messager_advance()>0) ;
    target = outstanding_recvs - max_out;
  }
#endif
  
  while(BGML_Messager_advance()>0);

  if (PCQueueLength (broadcast_q) > 0)
    sendBroadcastMessages();
  
#if CMK_IMMEDIATE_MSG
  if(received_immediate) {
    CmiHandleImmediate();
  }
  received_immediate = 0;
#endif
  
  if (numPosted <= request_max) 
    sendQueuedMessages();
}

static inline void SendMsgsUntil(int targetm, int targetb){

  sendBroadcastMessages();
  
  while(msgQueueLen>targetm && msgQBytes > targetb){
    while(BGML_Messager_advance()>0) ;
    sendQueuedMessages();
  }
  
  while ( BGML_Messager_advance() > 0 );
  
  sendBroadcastMessages();
}

void CmiNotifyIdle(){  
  AdvanceCommunications();

  CmiState cs = CmiGetStateN(pe);
  //int length = PCQueueLength(cs->recv);
  //if(length != 0)
  //printf("%d: {%d} PCQueue length = %d\n", CmiMyPe(), bgx_in_interrupt, length);
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

void CmiSyncListSendFn(int npes, int *pes, int size, char *msg){
  char *copymsg;
  copymsg = (char *)CmiAlloc(size);
  memcpy(copymsg,msg,size);
  CmiFreeListSendFn(npes, pes, size, msg);
}

#define OPTIMIZED_MULTICAST  0

static int iteration_count = 0;

void CmiFreeListSendFn(int npes, int *pes, int size, char *msg) {
  CMI_SET_BROADCAST_ROOT(msg,0);  
  CMI_MAGIC(msg) = CHARM_MAGIC_NUMBER;
  ((CmiMsgHeaderBasic *)msg)->size = size;  
  CMI_SET_CHECKSUM(msg, size);
  
  //printf("%d: In Free List Send Fn\n", CmiMyPe());
  
  CmiBecomeImmediate(msg);

  iteration_count ++;
#if 0
  if(iteration_count > 700) {    
    char pbuf[10000];
    char tbuf[64];
    
    sprintf(pbuf, "PE %d sending %d messages of size %d\n", CmiMyPe(), npes, size);
    for(int pcount = 0; pcount < npes; pcount++) {
      sprintf(tbuf," %d,", pes[pcount]);
      strcat(pbuf, tbuf);
    }
    printf("\n%s\n", pbuf);
  }
#endif
  
  int i, count = 0, my_loc = -1;
  for(i=0; i<npes; i++) {
    if(pes[i] == CmiMyPe() || pes[i] == vnpeer) {
      CmiSyncSend(pes[i], size, msg);
      my_loc = i;
    }
  }
  
#if OPTIMIZED_MULTICAST
  if (iteration_count > 500 && (npes > 1) && (CmiNumPes() >= 512)) {    
    int *newpelist = (int *) malloc (sizeof(int) * npes);
    int new_npes = 0;
    
    for(i=0; i<npes; i++) {
      if(pes[i] == CmiMyPe() || pes[i] == vnpeer) 
	continue;
      else
	newpelist[new_npes++] = pes[i];
    }

    machineMulticast (new_npes, newpelist, size, msg);
    AdvanceCommunications();
    return;
  }
#endif
  
  for(i=0;i<npes;i++) {
    if(pes[i] == CmiMyPe() || pes[i] == vnpeer);
    else if(i < npes - 1){
      CmiReference(msg);
      CmiGeneralFreeSend(pes[i], size, msg);
      
      //CmiSyncSend(pes[i], size, msg);
    }
    
#if CMK_PERSISTENT_COMM
    if(phs) 
      phscount ++;
#endif
  }
  
  if (npes  && (pes[npes-1] != CmiMyPe() && pes[npes-1] != vnpeer))
    CmiSyncSendAndFree(pes[npes-1], size, msg);
  else 
    CmiFree(msg);
  
  phscount = 0;
  
  AdvanceCommunications();
}


CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int size, char *msg){
  CmiAbort("CmiAsyncListSendFn not implemented.");
  return (CmiCommHandle) 0;
}
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



void CmiMachineProgressImpl()
{
#if 0
  unsigned long long new_time = rts_get_timebase();  
  if(new_time < lastProgress + progress_cycles) {
    return;
  }
  lastProgress = new_time;
#endif

#if !CMK_SMP
  AdvanceCommunications();
#else
  /*Not implemented yet. Communication server does not seem to be
    thread safe */
  /* CommunicationServerThread(0); */
#endif  
}

#endif

/* Dummy implementation */
extern "C" int CmiBarrier()
{
  BGTsC_Barrier(0);
  //BGTr_Barrier(1);
  return 0;
}

