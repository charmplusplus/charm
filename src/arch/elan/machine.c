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
//#include <mpi.h>

#include <elan/elan.h>


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
#endif

#define BROADCAST_SPANNING_FACTOR      4

/*#define CMI_BROADCAST_ROOT(msg)          ((CmiUInt2 *)(msg))[4]*/
#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank

#if CMK_BROADCAST_SPANNING_TREE
#  define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);
#else
#  define CMI_SET_BROADCAST_ROOT(msg, root)
#endif

ELAN_BASE     *elan_base;
ELAN_TPORT    *elan_port;
ELAN_QUEUE    *elan_q;
#define SMALL_MESSAGE_SIZE 5000
#define SYNC_MESSAGE_SIZE 65536

ELAN_EVENT *esmall, *emid, *elarge;
#define TAG_SMALL 0x69
#define TAG_LARGE 0x79

int Cmi_numpes;
int               Cmi_mynode;    /* Which address space am I */
int               Cmi_mynodesize;/* Number of processors in my address space */
int               Cmi_numnodes;  /* Total number of address spaces */
int               Cmi_numpes;    /* Total number of processors */
static int        Cmi_nodestart; /* First processor in this address space */ 
CpvDeclare(void*, CmiLocalQueue);

#define BLK_LEN  512

static int MsgQueueLen=0;
static int request_max;
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
    /*MPI_Request req;*/
    ELAN_EVENT *e;
    char *msg;
    struct msg_list *next;
    int size, destpe;
} SMSG_LIST;

static int Cmi_dim;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

double starttimer;

void CmiAbort(const char *message);
static void PerrorExit(const char *msg);

void SendSpanningChildren(int size, char *msg);

static void PerrorExit(const char *msg)
{
  perror(msg);
  exit(1);
}

/**************************  TIMER FUNCTIONS **************************/

void CmiTimerInit(void)
{
    starttimer =  elan_clock(elan_base->state)/1e9; /* MPI_Wtime();*/
}

double CmiTimer(void)
{
  //return 0;
  return elan_clock(elan_base->state)/1e9 - starttimer;
}

double CmiWallTimer(void)
{
  //return 0;
  return elan_clock(elan_base->state)/1e9 - starttimer;/*MPI_Wtime() - starttimer;*/
}

double CmiCpuTimer(void)
{
  return elan_clock(elan_base->state)/1e9 - starttimer;/*MPI_Wtime() - starttimer;*/
}

static PCQueue   msgBuf;

/************************************************************
 * 
 * Processor state structure
 *
 ************************************************************/

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
   //MPI_Status sts;
   int done;
     
   while(msg_tmp!=0) {
    done = 0;
    /*
    if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts)) 
      CmiAbort("CmiAllAsyncMsgsSent: MPI_Test failed!\n");
    */
    
    if(elan_tportTxDone(msg_tmp->e))
      done = 1;

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
  /*
  MPI_Status sts;
  */

  while ((msg_tmp) && ((CmiCommHandle)(msg_tmp->e) != c))
    msg_tmp = msg_tmp->next;

  if(msg_tmp) {
    done = 0;

    /*
    if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts)) 
      CmiAbort("CmiAsyncMsgSent: MPI_Test failed!\n");
    */

    if(elan_tportTxDone(msg_tmp->e))
      done = 1;

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
  /*  MPI_Status sts; */
  int locked = 0;
     
  while(msg_tmp!=0) {
    done =0;
    
    /*
    if(MPI_Test(&(msg_tmp->req), &done, &sts) != MPI_SUCCESS)
      CmiAbort("CmiReleaseSentMessages: MPI_Test failed!\n");
    */
    
    if(elan_tportTxDone(msg_tmp->e)) {
      elan_tportTxWait(msg_tmp->e);
      done = 1;
    }

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

int PumpMsgs(void)
{

  static int recv_small_done = 0;
  static int recv_large_done = 0;

  static char *sbuf;
  static char *lbuf;

  int flg, res;
  char *msg = 0;

  int recd=0;
  int size= 0;

  while(1) {
    msg = 0;

    if(!recv_small_done) {
      sbuf = (char *) CmiAlloc(SMALL_MESSAGE_SIZE);
      esmall = elan_tportRxStart(elan_port, 0, 0, 0, -1, TAG_SMALL, sbuf, SMALL_MESSAGE_SIZE);
      recv_small_done = 1;
    }
    
    if(!recv_large_done) {
      elarge = elan_tportRxStart(elan_port, ELAN_TPORT_RXPROBE, 0, 0, -1, TAG_LARGE, NULL, 1);
      recv_large_done = 1;
    }
    
    if(elan_tportRxDone(esmall)) {
      elan_tportRxWait(esmall, NULL, NULL, &size );
      //      CmiPrintf("Received small Message in %d %d\n", CmiMyPe(), size);

      msg = sbuf;
      recv_small_done = 0;
      flg = 1;

      CmiPushPE(CMI_DEST_RANK(msg), msg);
      
#if CMK_BROADCAST_SPANNING_TREE
      if (CMI_BROADCAST_ROOT(msg))
	SendSpanningChildren(size, msg);
#endif
    }
    
    if(elan_tportRxDone(elarge)) {
      elan_tportRxWait(elarge, NULL, NULL, &size );
      //      CmiPrintf("Received large Message in %d %d\n", CmiMyPe(), size);
      //printf("%d, ", size);
      
      lbuf = (char *) CmiAlloc(size);
      elarge = elan_tportRxStart(elan_port, 0, 0, 0, -1, TAG_LARGE, lbuf,size);
      elan_tportRxWait(elarge, NULL, NULL, &size);
      
      msg = lbuf;
      recv_large_done = 0;
      flg = 1;
      
      CmiPushPE(CMI_DEST_RANK(msg), msg);
#if CMK_BROADCAST_SPANNING_TREE
      if (CMI_BROADCAST_ROOT(msg))
	SendSpanningChildren(size, msg);
#endif
    }
    
    if(!flg)
      return recd;
    
    recd = 1;
    flg = 0;
  }
  return recd;
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

static int inexit = 0;

static void CommunicationServer(int sleepTime)
{
  SendMsgBuf(); 
  CmiReleaseSentMessages();
  PumpMsgs();
  if (inexit == 1) {
    while(!PCQueueEmpty(msgBuf) || !CmiAllAsyncMsgsSent()) {
      SendMsgBuf(); 
      PumpMsgs();
      CmiReleaseSentMessages();
    }
    
    /*
    if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD))
      CmiAbort("ConverseExit: MPI_Barrier failed!\n");
    MPI_Finalize();
    */
    
    elan_gsync(elan_base->allGroup);

#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyNode() == 0){
      CmiPrintf("End of program\n");
    }
#endif
    exit(0);   
  }
}

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
  PumpMsgs();
}
 
/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);

  //  CmiPrintf("Setting root to %d\n", 0);
  CMI_SET_BROADCAST_ROOT(dupmsg, 0);

  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
  }
  else
    CmiAsyncSendFn(destPE, size, dupmsg);
}

static int SendMsgBuf()
{
  SMSG_LIST *msg_tmp;
  char *msg;
  int node, rank, size;

  while (!PCQueueEmpty(msgBuf))
  {
    msg_tmp = (SMSG_LIST *)PCQueuePop(msgBuf);
    node = msg_tmp->destpe;
    size = msg_tmp->size;
    msg = msg_tmp->msg;
    msg_tmp->next = 0;
    while (MsgQueueLen > request_max) {
	/*printf("Waiting for %d messages to be sent\n", MsgQueueLen);*/
	CmiReleaseSentMessages();
	PumpMsgs();
    }
    
    /*
    if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
      CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
    */

    msg_tmp->e = elan_tportTxStart(elan_port, (size <= SYNC_MESSAGE_SIZE)? 0: ELAN_TPORT_TXSYNC, node, CmiMyNode(), (size <= SMALL_MESSAGE_SIZE)? TAG_SMALL : TAG_LARGE, msg, size);
    
    MsgQueueLen++;
    if(sent_msgs==0)
      sent_msgs = msg_tmp;
    else
      end_sent->next = msg_tmp;
    end_sent = msg_tmp;
  }
}

CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  
  //  CmiPrintf("In CmiAsyncSendFn\n");

  CmiState cs = CmiGetState();
  SMSG_LIST *msg_tmp;
  CmiUInt2  rank, node;
  /*  MPI_Status my_status;*/
     
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
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->size = size;
  msg_tmp->destpe = node;
  PCQueuePush(msgBuf,(char *)msg_tmp);
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
  
  /*
  if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,destPE,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
    CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
  */

  //  CmiPrintf("Sending Message to %d from %d %d TAG = %d\n", destPE, CmiMyNode(), size, (size <= SMALL_MESSAGE_SIZE)? TAG_SMALL : TAG_LARGE);
  msg_tmp->e = elan_tportTxStart(elan_port, 0, destPE, CmiMyNode(), (size <= SMALL_MESSAGE_SIZE)? TAG_SMALL : TAG_LARGE, msg, size);
  
  /*
    if(size <= SMALL_MESSAGE_SIZE)
    elan_tportTxWait(msg_tmp->e);
  */
  
  MsgQueueLen++;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
  return (CmiCommHandle) msg_tmp->e;
#endif
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();

  //  CmiPrintf("Setting root to %d\n", 0);
  CMI_SET_BROADCAST_ROOT(msg, 0);

  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
  } 
  else { 
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
  
  //  CmiPrintf("startpe = %d\n", startpe);
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

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
  //  CmiPrintf("Setting root to %d\n", Cmi_mype+1);
  CMI_SET_BROADCAST_ROOT(msg, Cmi_mype+1);
  SendSpanningChildren(size, msg);
#else
  int i ;
     
  for ( i=cs->pe+1; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
  for ( i=0; i<cs->pe; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
#endif
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
  //  CmiPrintf("Setting root to %d cs\n", cs->pe+1);
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);
#else
  int i ;
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
#endif
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)  
{
  int i ;

  for ( i=1; i<Cmi_numpes; i++ ) 
    CmiAsyncSendFn(i,size,msg) ;
  return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastAllFn(int size, char *msg)  /* All including me */
{
#if CMK_BROADCAST_SPANNING_TREE
  CmiState cs = CmiGetState();
  CmiSyncSendFn(cs->pe, size,msg) ;
  //  CmiPrintf("Setting root to %d cs\n", cs->pe+1);
  CMI_SET_BROADCAST_ROOT(msg, cs->pe+1);
  SendSpanningChildren(size, msg);
#else
  int i ;
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
#endif
  CmiFree(msg) ;
}

void ConverseExit(void)
{
#if ! CMK_SMP
  while(!CmiAllAsyncMsgsSent()) {
    PumpMsgs();
    CmiReleaseSentMessages();
  }

  elan_gsync(elan_base->allGroup); 

  ConverseCommonExit();
  
  /*
    if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD)) 
    CmiAbort("ConverseExit: MPI_Barrier failed!\n");
    MPI_Finalize();
  */

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
  CmiMyArgv=CmiCopyArgs(Cmi_argv);
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
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,CmiNotifyBeginIdle,(void *)s);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,CmiNotifyStillIdle,(void *)s);
#else
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,CmiNotifyIdle,NULL);
#endif
  
  PumpMsgs();
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
  /*
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &Cmi_numnodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &Cmi_mynode);
  */

  putenv("LIBELAN_SHM_ENABLE=0");

  if (!(elan_base = elan_baseInit())) {
    perror("Failed elan_baseInit()");
    exit(1);
  }

  if ((elan_q = elan_allocQueue(elan_base->state)) == NULL) {
    
    perror( "elan_allocQueue failed" );
    exit (1);
  }
  
  int nslots = elan_base->tport_nslots;
  if(nslots < elan_base->state->nvp)
    nslots = elan_base->state->nvp;
  //  if(nslots > 256)
  //nslots = 256;
  nslots += 16;

  //  nslots = 256;

  if (!(elan_port = elan_tportInit(elan_base->state,
				   (ELAN_QUEUE *)elan_q,
				   /*elan_main2elan(elan_base->state, q),*/
				   nslots /*elan_base->tport_nslots*/, 
				   elan_base->tport_smallmsg,
				   elan_base->tport_bigmsg,
				   elan_base->waitType, elan_base->retryCount,
				   &(elan_base->shm_key),
				   elan_base->shm_fifodepth, elan_base->shm_fragsize))) {
    
    perror("Failed to to initialise TPORT");
    exit(1);
  }

  elan_gsync(elan_base->allGroup);

  Cmi_numnodes = elan_base->state->nvp;
  Cmi_mynode =  elan_base->state->vp;

  /* processor per node */
  Cmi_mynodesize = 1;
  CmiGetArgInt(argv,"+ppn", &Cmi_mynodesize);
#if ! CMK_SMP
  if (Cmi_mynodesize > 1 && Cmi_mynode == 0) 
    CmiAbort("+ppn cannot be used in non SMP version!\n");
#endif
  Cmi_numpes = Cmi_numnodes * Cmi_mynodesize;
  Cmi_nodestart = Cmi_mynode * Cmi_mynodesize;
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
  msgBuf = PCQueueCreate();

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
  /*
  MPI_Abort(MPI_COMM_WORLD, 1);
  */
  exit(1);
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
    exit(1);
    /*MPI_Abort(MPI_COMM_WORLD, 1);*/
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
