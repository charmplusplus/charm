#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <errno.h>
#include "converse.h"
#include <elan/elan.h>

/*Support for ++debug: */
#include <unistd.h> /*For getpid()*/
#include <stdlib.h> /*For sleep()*/

#include "machine.h"
#include "pcqueue.h"

#define MAX_QLEN 200

/*
  To reduce the buffer used in broadcast and distribute the load from 
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of 
  spanning tree broadcast algorithm.
  This will use the fourth short in message as an indicator of spanning tree
  root.
*/
#define CMK_BROADCAST_SPANNING_TREE    1
#define BROADCAST_SPANNING_FACTOR      4

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
#define SMALL_MESSAGE_SIZE 8192       /* Message sizes greater will be 
					  probe received adding 5us overhead*/
#define SYNC_MESSAGE_SIZE 512        /* Message sizes greater will be 
					  sent synchronously thus avoiding copying*/
#define NON_BLOCKING_MSG 128           /* Message sizes greater 
					  than this will be sent asynchronously*/

ELAN_EVENT *esmall, *elarge;
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

#define SIZEFIELD(m) ((int *)((char *)(m)-2*sizeof(int)))[0]

static int MsgQueueLen=0;
static int request_max;

static void* localMsgBuf;
static void ConverseRunPE(int everReturn);

typedef struct msg_list {
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
    starttimer =  elan_clock(elan_base->state); 
}

double CmiTimer(void)
{
  return (elan_clock(elan_base->state) - starttimer)/1e9;
}

double CmiWallTimer(void)
{
  return (elan_clock(elan_base->state) - starttimer)/1e9;
}

double CmiCpuTimer(void)
{
  return (elan_clock(elan_base->state) - starttimer)/1e9;
}

static PCQueue   msgBuf;

/************************************************************
 * 
 * Processor state structure
 *
 ************************************************************/

/*****
      SMP version Extend later, currently only NON SMP version 
***************/

#include "machine-smp.c"

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

   int done;
     
   while(msg_tmp!=0) {
    done = 0;
    
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

  while ((msg_tmp) && ((CmiCommHandle)(msg_tmp->e) != c))
    msg_tmp = msg_tmp->next;

  if(msg_tmp) {
    done = 0;
    
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
  int locked = 0;
     
  while(msg_tmp!=0) {
    done =0;
    
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
      
      if(CMI_DEST_RANK(msg_tmp->msg) != 1000) {
	if(SIZEFIELD(msg_tmp->msg) == SMALL_MESSAGE_SIZE)
	  CqsEnqueue(localMsgBuf, msg_tmp->msg);
	else
	  CmiFree(msg_tmp->msg);
      }
      
      CmiFree(msg_tmp);
      msg_tmp = temp;
    } else {
      prev = msg_tmp;
      msg_tmp = msg_tmp->next;
    }
  }
  end_sent = prev;
}

/* retflag = 0, receive as many messages as can be and then post another receive
   retflag = 1, receive the first message and return */
int PumpMsgs(int retflag)
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
      if(!CqsEmpty(localMsgBuf))
	CqsDequeue(localMsgBuf, &sbuf);
      else
	sbuf = (char *) CmiAlloc(SMALL_MESSAGE_SIZE);
      
      esmall = elan_tportRxStart(elan_port, 0, 0, 0, -1, TAG_SMALL, sbuf, SMALL_MESSAGE_SIZE);
      recv_small_done = 1;
    }
    
    if(!recv_large_done) {
      elarge = elan_tportRxStart(elan_port, ELAN_TPORT_RXPROBE, 0, 0, -1, TAG_LARGE, NULL, 1);
      ///lbuf = (char *) CmiAlloc(65540);
      ///elarge = elan_tportRxStart(elan_port, 0, 0, 0, -1, TAG_LARGE, lbuf, 65540);
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

    if (retflag)
      return flg;

    recd = 1;
    flg = 0;
  }
  return recd;
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/
void *CmiGetNonLocal(void)
{
  CmiState cs = CmiGetState();
  void *msg = NULL;
  CmiIdleLock_checkMessage(&cs->idle);
  msg =  PCQueuePop(cs->recv); 

  if(!msg) {
    CmiReleaseSentMessages();
    if (PumpMsgs(1))
      return  PCQueuePop(cs->recv);
    else
      return 0;
  }
  return msg;
}

void CmiNotifyIdle(void)
{
  CmiReleaseSentMessages();
  PumpMsgs(1);
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

CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  
  //  CmiPrintf("In CmiAsyncSendFn\n");

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

  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  while (MsgQueueLen > request_max) {
    /*printf("Waiting for %d messages to be sent\n", MsgQueueLen);*/
    CmiReleaseSentMessages();
    PumpMsgs(0);
  }
  
  msg_tmp->e = elan_tportTxStart(elan_port, 
				 (size <= SYNC_MESSAGE_SIZE)? 0: ELAN_TPORT_TXSYNC, 
				 destPE, CmiMyPe(), 
				 (size <= SMALL_MESSAGE_SIZE)? TAG_SMALL:TAG_LARGE, 
				 msg, size);
  
  MsgQueueLen++;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
  return (CmiCommHandle) msg_tmp->e;
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  CMI_SET_BROADCAST_ROOT(msg, 0);

  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
  } 
  else { 
    if(size <= NON_BLOCKING_MSG) {
      (void)elan_tportTxWait(elan_tportTxStart(elan_port, 0, destPE, CmiMyPe(), TAG_SMALL, msg, size));

      if(SIZEFIELD(msg) == SMALL_MESSAGE_SIZE)
	CqsEnqueue(localMsgBuf, msg);
      else
	CmiFree(msg);
    }
    else
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

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
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
  while(!CmiAllAsyncMsgsSent()) {
    PumpMsgs(0);
    CmiReleaseSentMessages();
  }

  elan_gsync(elan_base->allGroup); 

  ConverseCommonExit();

#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
  if (CmiMyPe() == 0){
    CmiPrintf("End of program\n");
  }
#endif
  exit(0);
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

  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,CmiNotifyIdle,NULL);
  
  PumpMsgs(0);
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

  putenv("LIBELAN_SHM_ENABLE=0");

  localMsgBuf = CqsCreate();

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
  if(nslots > 256)
    nslots = 256;

  if (!(elan_port = elan_tportInit(elan_base->state,
				   (ELAN_QUEUE *)elan_q,
				   /*elan_main2elan(elan_base->state, q),*/
				   nslots /*elan_base->tport_nslots*/, 
				   elan_base->tport_smallmsg,
				   elan_base->tport_bigmsg,
				   elan_base->waitType, elan_base->retryCount,
				   &(elan_base->shm_key),
				   elan_base->shm_fifodepth, 
				   elan_base->shm_fragsize))) {
    
    perror("Failed to to initialise TPORT");
    exit(1);
  }
  
  elan_gsync(elan_base->allGroup);
  
  Cmi_numnodes = elan_base->state->nvp;
  Cmi_mynode =  elan_base->state->vp;

  /* processor per node */
  Cmi_mynodesize = 1;
  CmiGetArgInt(argv,"+ppn", &Cmi_mynodesize);

  if (Cmi_mynodesize > 1 && Cmi_mynode == 0) 
    CmiAbort("+ppn cannot be used in non SMP version!\n");
  
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
  exit(1);
}
