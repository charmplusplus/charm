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

#define MAX_QLEN 2000
#define MAX_BYTES 10000000

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
#define CMI_MSG_TYPE(msg)                ((CmiMsgHeaderBasic *)msg)->type

#if CMK_BROADCAST_SPANNING_TREE
#  define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);
#else
#  define CMI_SET_BROADCAST_ROOT(msg, root)
#endif

ELAN_BASE     *elan_base;
ELAN_TPORT    *elan_port;
ELAN_QUEUE    *elan_q;
#define SMALL_MESSAGE_SIZE 20000     /* for comm bench */
                                     /* Message sizes greater will be 
					  probe received adding 5us overhead*/
#define SYNC_MESSAGE_SIZE 20000
                                       /* Message sizes greater will be 
				       sent synchronously thus avoiding copying*/

#define NON_BLOCKING_MSG  256          /* Message sizes greater 
					  than this will be sent asynchronously*/
#define RECV_MSG_Q_SIZE 16

ELAN_EVENT *esmall[RECV_MSG_Q_SIZE], *elarge;
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
static int MsgQueueBytes=0;
static int request_max;
static int request_bytes;

#include "queueing.h"

Queue localMsgBuf;

int outstandingMsgs[3000];
int ppn_factor = 1;                    //The size affinity group for 
                                       //preventing stretches.
int stretchFlag = 0;
int blockingReceiveFlag = 0;

static void ConverseRunPE(int everReturn);

typedef struct msg_list {
  ELAN_EVENT *e;
  char *msg;
  struct msg_list *next;
  int size, destpe;
  int sent;
} SMSG_LIST;

static int Cmi_dim;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;
static SMSG_LIST *cur_unsent=0;

PersistentHandle  *phs = NULL;
int phsSize;

void PumpPersistent();
void CmiSendPersistentMsg(PersistentHandle h, int destPE, int size, void *m);
void CmiSyncSendPersistent(int destPE, int size, char *msg, PersistentHandle h);

void ElanSendQueuedMessages();

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

CsvDeclare(CmiNodeState, NodeState);

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
#if CMK_IMMEDIATE_MSG
  if (CmiGetHandler(msg) == CpvAccessOther(CmiImmediateMsgHandlerIdx,0)) {
//CmiPrintf("[node %d] Immediate Message %d %d {{. \n", CmiMyNode(), CmiGetHandler(msg), _ImmediateMsgHandlerIdx);
    CmiHandleMessage(msg);
//CmiPrintf("[node %d] Immediate Message done.}} \n", CmiMyNode());
    return;
  }
#endif
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
     
   while((msg_tmp!=0) && (msg_tmp->e != NULL)){
    done = 0;
    
    if(elan_tportTxDone(msg_tmp->e))
      done = 1;
    else 
      elan_deviceCheck(elan_base->state);

    if(!done)
      return 0;
    msg_tmp = msg_tmp->next;
    //    MsgQueueLen--;
   }
   return 1;
}

int CmiAsyncMsgSent(CmiCommHandle c) {
     
  SMSG_LIST *msg_tmp = sent_msgs;
  int done;

  while ((msg_tmp) && (msg_tmp ->e != NULL) && ((CmiCommHandle)(msg_tmp->e) != c))
    msg_tmp = msg_tmp->next;

  if(msg_tmp) {
    done = 0;
    
    if(elan_tportTxDone(msg_tmp->e))
      done = 1;
    else 
      elan_deviceCheck(elan_base->state);
    
    return ((done)?1:0);
  } else {
    return 1;
  }
}

void CmiReleaseCommHandle(CmiCommHandle c)
{
  return;
}

void release_pmsg_list();

static void CmiReleaseSentMessages(void)
{
  SMSG_LIST *msg_tmp=sent_msgs;
  SMSG_LIST *prev=0;
  SMSG_LIST *temp;
  int done;
  int locked = 0;

#ifndef CMK_OPTIMIZE 
  double rel_start_time = CmiWallTimer();
#endif
     
  while(msg_tmp != NULL){
    if(msg_tmp->sent) {
      done =0;
      
      if(elan_tportTxDone(msg_tmp->e)) {
	elan_tportTxWait(msg_tmp->e);
	done = 1;
      }
      else 
	elan_deviceCheck(elan_base->state);
      
      if(done) {
	MsgQueueLen--;
	MsgQueueBytes -= msg_tmp->size;
	
	outstandingMsgs[msg_tmp->destpe/ppn_factor] = 0;

	/* Release the message */
	temp = msg_tmp->next;
	if(prev==0)  /* first message */
	  sent_msgs = temp;
	else
	  prev->next = temp;
	
	if(CMI_MSG_TYPE(msg_tmp->msg) != 2) {
	  if(SIZEFIELD(msg_tmp->msg) == SMALL_MESSAGE_SIZE) {
	    //	  CmiPrintf("ELAN Returning message to queue\n");
	    CqsEnqueue(localMsgBuf, msg_tmp->msg);
	  }
	  else
	    CmiFree(msg_tmp->msg);
	}
	else if(CMI_MSG_TYPE(msg_tmp->msg) == 2) {
	  // Dont do any thing the message has been statically allocated
	}
	else
	  CmiFree(msg_tmp->msg);
      
	CmiFree(msg_tmp);
	msg_tmp = temp;
      } else {
	prev = msg_tmp;
	msg_tmp = msg_tmp->next;
      }
    }
    else {
      prev = msg_tmp;
      msg_tmp = msg_tmp->next;
    }
  }
  
  //if(cur_unsent == NULL)
  end_sent = prev;

#if CMK_PERSISTENT_COMM
  release_pmsg_list();
#endif

#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
  double rel_end_time = CmiWallTimer();
  if(rel_end_time > rel_start_time + 50/1e6)
    traceUserBracketEvent(20, rel_start_time, rel_end_time);
  if((rel_end_time > rel_start_time + 5/1e3) && stretchFlag)
    CmiPrintf("%d:Stretched Release Sent Msgs at %5.3lfs of %5.5lf ms\n", CmiMyPe(), rel_end_time, (rel_end_time - rel_start_time)*1e3);
#endif
#endif
}

/* retflag = 0, receive as many messages as can be and then post another receive
   retflag = 1, receive the first message and return */
int PumpMsgs(int retflag)
{

  static char recv_small_done[RECV_MSG_Q_SIZE];
  static int recv_large_done = 0;

  static char *sbuf[RECV_MSG_Q_SIZE];
  static char *lbuf;

  static int event_idx = 0;
  static int post_idx = 0;
  static int step1 = 0;

  int flg, res;
  char *msg = 0;

  int recd=0;
  int size= 0;

#ifndef CMK_OPTIMIZE 
  double pmp_start_time = CmiWallTimer();
#endif

  int ecount = 0;
  while(1) {
    msg = 0;
    
    ecount = 0;
    for(int rcount = 0; rcount < RECV_MSG_Q_SIZE; rcount ++){
      ecount = (rcount + post_idx) % RECV_MSG_Q_SIZE;
      if(!recv_small_done[ecount]) {

	//if(CmiMyPe() == 0)
	//	CmiPrintf("%d:Posting %d, %d\n", CmiMyPe(), ecount, post_idx);

	if(!CqsEmpty(localMsgBuf)) {
	  //CmiPrintf("ELAN Getting message from queue\n");
	  CqsDequeue(localMsgBuf, (void *)&sbuf[ecount]);
	}
	else
	  sbuf[ecount] = (char *) CmiAlloc(SMALL_MESSAGE_SIZE);
	
	esmall[ecount] = elan_tportRxStart(elan_port, 0, 0, 0, -1, TAG_SMALL, sbuf[ecount], SMALL_MESSAGE_SIZE);
	recv_small_done[ecount] = 1;
      }
      else {
	ecount = (ecount + RECV_MSG_Q_SIZE - 1) % RECV_MSG_Q_SIZE;
	break;
      }
    }
    post_idx = ecount + 1;
    
    if(!recv_large_done) {
      //CmiPrintf("%d:Posting a probe \n", CmiMyPe());
      elarge = elan_tportRxStart(elan_port, ELAN_TPORT_RXPROBE, 0, 0, -1, TAG_LARGE, NULL, 0);
      recv_large_done = 1;
    }
    
    if(!step1 && (elan_tportRxDone(elarge) || retflag == 2)) {
      elan_tportRxWait(elarge, NULL, NULL, &size );
      
      if (blockingReceiveFlag)
	CmiPrintf("Received large Message in %d %d\n", CmiMyPe(), size);

      lbuf = (char *) CmiAlloc(size);
      elarge = elan_tportRxStart(elan_port, 0, 0, 0, -1, TAG_LARGE, lbuf,size);
      step1 = 1;

      if(retflag == 2)
        retflag = 0;
    }

    if(step1 && (elan_tportRxDone(elarge) || retflag == 2)) {
      elan_tportRxWait(elarge, NULL, NULL, &size);
      
      msg = lbuf;
      recv_large_done = 0;
      flg = 1;
      
      CmiPushPE(CMI_DEST_RANK(msg), msg);
#if CMK_BROADCAST_SPANNING_TREE
      if (CMI_BROADCAST_ROOT(msg))
        SendSpanningChildren(size, msg);
#endif
      step1 = 0;

      if(retflag == 2)
        retflag = 0;
    }
    
    ecount = 0;
    for(int rcount = 0; rcount < RECV_MSG_Q_SIZE; rcount ++){
      ecount = (rcount + event_idx) % RECV_MSG_Q_SIZE;
      if(elan_tportRxDone(esmall[ecount]) || retflag == 3) {
	
	//CmiPrintf("%d:Receiving %d, %d\n", CmiMyPe(), ecount, event_idx);

	elan_tportRxWait(esmall[ecount], NULL, NULL, &size );
	
	msg = sbuf[ecount];
	recv_small_done[ecount] = 0;
	sbuf[ecount] = NULL;
	flg = 1;
	
	CmiPushPE(CMI_DEST_RANK(msg), msg);
	
#if CMK_BROADCAST_SPANNING_TREE
	if (CMI_BROADCAST_ROOT(msg))
	  SendSpanningChildren(size, msg);
#endif
	if(retflag == 3)
	  retflag = 0;
      }
      else {
	elan_deviceCheck(elan_base->state);
	ecount = (ecount + RECV_MSG_Q_SIZE - 1) % RECV_MSG_Q_SIZE;
	break;
      }
    }
    event_idx = ecount + 1;

#if CMK_PERSISTENT_COMM
    PumpPersistent();
#endif

    if(!flg) {
#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
      double pmp_end_time = CmiWallTimer();
      if(pmp_end_time > pmp_start_time + 50/1e6)
	traceUserBracketEvent(10, pmp_start_time, pmp_end_time);
      if((pmp_end_time > pmp_start_time + 5/1e3) && stretchFlag)
	CmiPrintf("%d:Stretched Pump Msgs at %5.3lfs of %5.5lf ms\n", CmiMyPe(), pmp_end_time, (pmp_end_time - pmp_start_time)*1e3);
#endif
#endif
      return recd;    
    }

    if (retflag) {
#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
      double pmp_end_time = CmiWallTimer();
      if(pmp_end_time > pmp_start_time + 50/1e6)
	traceUserBracketEvent(10, pmp_start_time, pmp_end_time);
      if((pmp_end_time > pmp_start_time + 5/1e3) && stretchFlag)
	CmiPrintf("%d:Stretched Pump Msgs at %5.3lfs of %5.5lf ms\n", CmiMyPe(), pmp_end_time, (pmp_end_time - pmp_start_time)*1e3);
#endif
#endif
      return flg;
    }

    recd = 1;
    flg = 0;
  }
  return recd;
}

void *remote_get(void * srcptr, void *destptr, int size, int srcPE){
  return (void *)elan_get(elan_base->state, srcptr, destptr, size, srcPE);
}

int remote_get_done(void *e){
  ELAN_EVENT *evt = (ELAN_EVENT *)e;

  int flag = elan_poll(evt, ELAN_POLL_EVENT);
  /*
  if(flag) {
    elan_wait(evt, ELAN_WAIT_EVENT);
    return 1;
  }
  else
    return 0;  
  */
  return flag;
}

void remote_get_wait_all(){
  elan_getWaitAll(elan_base->state, ELAN_WAIT_EVENT);
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
    ElanSendQueuedMessages();
    if (PumpMsgs(0))  // PumpMsgs(1)
      return  PCQueuePop(cs->recv);
    else
      return 0;
  }
  return msg;
}

void CmiPing() {
  CmiReleaseSentMessages();
  //PumpMsgs(0);
  ElanSendQueuedMessages();
}

void enableBlockingReceives(){
  blockingReceiveFlag = 1;
}

void CmiNotifyIdle(void)
{
  static int previousSleepTime = 0;
  CmiReleaseSentMessages();
  ElanSendQueuedMessages();

  if(!PumpMsgs(0) && blockingReceiveFlag /*&& (CmiMyPe() % 4 == 0)*/){
    if (!PCQueueEmpty(CmiGetState()->recv)) return; 
    if (!CdsFifo_Empty(CpvAccess(CmiLocalQueue))) return;
    if (!CqsEmpty(CpvAccess(CsdSchedQueue))) return;
    if (cur_unsent) return;
    PumpMsgs(3); 
  }
  /*
  else if(!PumpMsgs(0)){
    int curTime = CmiWallTimer() * 1000;
    if (((curTime - previousSleepTime > 10) && (CmiMyPe() % 4 == 0)) 
	||((curTime - previousSleepTime > 20) && (CmiMyPe() % 4 != 0))){
      usleep(0);
      previousSleepTime = CmiWallTimer() * 1000;
    }
  }
  */
}
 
#if CMK_IMMEDIATE_MSG
void CmiProbeImmediateMsg()
{
  PumpMsgs(0);
}
#endif
/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(int destPE, int size, char *msg)
{
  CmiState cs = CmiGetState();
  char *dupmsg;
  if(size <= SMALL_MESSAGE_SIZE){
    if(!CqsEmpty(localMsgBuf)) 
      CqsDequeue(localMsgBuf, (void **)&dupmsg);
    else 
      dupmsg = CmiAlloc(SMALL_MESSAGE_SIZE);
    
    CMI_MSG_TYPE(dupmsg) = 1;
  }
  else
    dupmsg = (char *) CmiAlloc(size);
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

CmiCommHandle ElanSendFn(int destPE, int size, char *msg, int flag)
{
  
  //CmiPrintf("In CmiAsyncSendFn %d %d %d\n", CmiMyPe(), destPE, size);

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

#if CMK_PERSISTENT_COMM
  if (phs) {
    CmiAssert(phsSize == 1);
    CmiSendPersistentMsg(*phs, destPE, size, msg);
    return NULL;
  }
#endif
  
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  msg_tmp->size = size;
  msg_tmp->sent = 0;
  msg_tmp->e = NULL;
  msg_tmp->destpe = destPE;

  if ((MsgQueueLen > request_max || MsgQueueBytes > request_bytes) && (!flag)) {
    CmiReleaseSentMessages();
    PumpMsgs(0);
  }
  
  SMSG_LIST * ptr = cur_unsent, *new_unsent = NULL;
  while (MsgQueueLen <= request_max && MsgQueueBytes <= request_bytes && ptr != NULL) {
    if(!outstandingMsgs[ptr->destpe/ppn_factor] && !ptr->sent){
      ptr->e = elan_tportTxStart(elan_port, (ptr->size <= SYNC_MESSAGE_SIZE)? 
				 0: ELAN_TPORT_TXSYNC, ptr->destpe, CmiMyPe(), 
				 (ptr->size <= SMALL_MESSAGE_SIZE)? 
				 TAG_SMALL:TAG_LARGE, ptr->msg, ptr->size);
      ptr->sent = 1;
    
      MsgQueueLen++;
      MsgQueueBytes += ptr->size;
      
      outstandingMsgs[ptr->destpe/ppn_factor] = 1;
    }
    else if ((!ptr->sent) && (new_unsent == NULL))
      new_unsent = ptr;
    
    ptr = ptr->next;
  }
  
  if(new_unsent)
    cur_unsent = new_unsent;
  else
    cur_unsent = ptr;

  if(MsgQueueLen > request_max || MsgQueueBytes > request_bytes 
     || outstandingMsgs[destPE/ppn_factor]){
    
    if(sent_msgs==0)
      sent_msgs = msg_tmp;
    else
      end_sent->next = msg_tmp;
    end_sent = msg_tmp;
    
    if(cur_unsent == 0)
      cur_unsent = msg_tmp;

    //CmiPrintf("%d: HERE queuing messages for %d\n", CmiMyPe(), destPE);
    //CmiPrintf("HERE %d %d\n", MsgQueueLen, MsgQueueBytes);
  }
  else{
    msg_tmp->e = elan_tportTxStart(elan_port, (size <= SYNC_MESSAGE_SIZE)? 
				   0: ELAN_TPORT_TXSYNC, destPE, CmiMyPe(), 
				   (size <= SMALL_MESSAGE_SIZE)? 
				   TAG_SMALL:TAG_LARGE, msg, size);
    msg_tmp->sent = 1;

    MsgQueueLen++;
    MsgQueueBytes += size;

    outstandingMsgs[destPE/ppn_factor] = 1;

    if(sent_msgs==0)
      sent_msgs = msg_tmp;
    else
      end_sent->next = msg_tmp;
    end_sent = msg_tmp;
    return (CmiCommHandle) msg_tmp->e;
  }
  return NULL;
}

void ElanSendQueuedMessages() {
  SMSG_LIST * ptr = cur_unsent, *new_unsent = NULL;
  while (MsgQueueLen <= request_max && MsgQueueBytes <= request_bytes && ptr != NULL) {
    if(!outstandingMsgs[ptr->destpe/ppn_factor] && !ptr->sent){

      ptr->e = elan_tportTxStart(elan_port, (ptr->size <= SYNC_MESSAGE_SIZE)? 
				 0: ELAN_TPORT_TXSYNC, ptr->destpe, CmiMyPe(), 
				 (ptr->size <= SMALL_MESSAGE_SIZE)? 
				 TAG_SMALL:TAG_LARGE, ptr->msg, ptr->size);
      ptr->sent = 1;
      
      MsgQueueLen++;
      MsgQueueBytes += ptr->size;
      
      outstandingMsgs[ptr->destpe/ppn_factor] = 1;
    }
    else if ((!ptr->sent) && (new_unsent == NULL))
      new_unsent = ptr;
    
    ptr = ptr->next;
  }
  
  if(new_unsent)
    cur_unsent = new_unsent;
  else
    cur_unsent = ptr;
}

CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg){
  return ElanSendFn(destPE, size, msg, 0);
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
#ifndef CMK_OPTIMIZE 
  double snd_start_time = CmiWallTimer();
#endif

  CmiState cs = CmiGetState();
  CMI_SET_BROADCAST_ROOT(msg, 0);

  if (cs->pe==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
  } 
  else { 
    if(size <= NON_BLOCKING_MSG) {
      (void)elan_tportTxWait(elan_tportTxStart(elan_port, 0, destPE, CmiMyPe(), TAG_SMALL, msg, size));

      if(SIZEFIELD(msg) == SMALL_MESSAGE_SIZE) {
	//	CmiPrintf("ELAN Returning message to queue\n");
	CqsEnqueue(localMsgBuf, msg);
      }
      else
	CmiFree(msg);
    }
    else
      CmiAsyncSendFn(destPE, size, msg);
  }

#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
  double snd_end_time = CmiWallTimer();
  if(snd_end_time > snd_start_time + 5/1e6) 
    traceUserBracketEvent(30, snd_start_time, snd_end_time);
  if((snd_end_time > snd_start_time + 5/1e3) && stretchFlag)
      CmiPrintf("%d:Stretched Send to %d at %5.3lfs of %5.5lf ms\n", CmiMyPe(), destPE, snd_end_time, (snd_end_time - snd_start_time)*1e3);
#endif
#endif
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
    ElanSendFn(destPE, size, dupmsg, 1);
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
  while(!CmiAllAsyncMsgsSent() || cur_unsent ) {
    PumpMsgs(0);
    ElanSendQueuedMessages();
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
  /*  since elan version is not a SMP version */
  /*
  CmiMyArgv=CmiCopyArgs(Cmi_argv);
  */
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

  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
  
  PumpMsgs(0);
  elan_gsync(elan_base->allGroup);

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

  putenv("LIBELAN_SHM_ENABLE=1");

  localMsgBuf = CqsCreate();

  if (!(elan_base = elan_baseInit())) {
    perror("Failed elan_baseInit()");
    exit(1);
  }

  elan_gsync(elan_base->allGroup);
  
  if ((elan_q = elan_gallocQueue(elan_base, elan_base->allGroup)) == NULL) {
    
    perror( "elan_gallocQueue failed" );
    exit (1);
  }
  
  int nslots = elan_base->tport_nslots;
  
  //if(nslots < elan_base->state->nvp)
  //nslots = elan_base->state->nvp;
  //if(nslots > 256)
  //nslots = 256;

  if (!(elan_port = elan_tportInit(elan_base->state,
				   elan_q,
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
  request_bytes = MAX_BYTES;
  
  CmiGetArgInt(argv,"+requestmax",&request_max);
  CmiGetArgInt(argv,"+requestbytes",&request_bytes);

  /*printf("request max=%d\n", request_max);*/
  if (CmiGetArgFlag(argv,"++debug"))
  {   /*Pause so user has a chance to start and attach debugger*/
    printf("CHARMDEBUG> Processor %d has PID %d\n",CmiMyNode(),getpid());
    if (!CmiGetArgFlag(argv,"++debug-no-pause"))
      sleep(10);
  }

  CmiTimerInit();
  msgBuf = PCQueueCreate();

  CsvInitialize(CmiNodeState, NodeState);
  CmiNodeStateInit(&CsvAccess(NodeState));

  int rms_nodes = 1, rms_procs = 1;

  if(getenv("RMS_NODES") != NULL)
    rms_nodes = atoi(getenv("RMS_NODES"));
  if(getenv("RMS_PROCS") != NULL)
    rms_procs = atoi(getenv("RMS_PROCS"));
  ppn_factor = (rms_procs/rms_nodes);   //4 nodes is the stretch group affinity
  if(ppn_factor == 0)   //debug
    ppn_factor = 1;

  ppn_factor = 4;
  
  //CmiPrintf("ppn_factor = %d\n", ppn_factor);

  CmiStartThreads(argv);
  ConverseRunPE(initret);

#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
  traceRegisterUserEvent("Pump Messages", 10);
  traceRegisterUserEvent("Release Sent Messages", 20);
  traceRegisterUserEvent("ELAN Send", 30);
#endif
#endif
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message)
{
  CmiError(message);
  *((int *)NULL) = 0;
  exit(1);
}

void CmiSyncListSendFn(int npes, int *pes, int len, char *msg)
{
  CmiError("ListSend not implemented.");
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int len, char *msg)
{
  CmiError("ListSend not implemented.");
  return (CmiCommHandle) 0;
}

void CmiFreeListSendFn(int npes, int *pes, int len, char *msg)
{
  //  CmiError("ListSend not implemented.");
  //CmiPrintf("[%d] CmiFreeListSendFn %d\n", CmiMyPe(), usePhs);
  
  int i;
#if CMK_PERSISTENT_COMM
  if (phs) {
    CmiAssert(phsSize == npes);
    for(i=0;i<npes;i++) 
      CmiSyncSendPersistent(pes[i], len, msg, phs[i]);
  }
  else 
#endif
    for(i=0;i<npes;i++)
      CmiSyncSend(pes[i], len, msg);

  CmiFree(msg);

  /*
    for(i=0;i<npes;i++) {
    }
  */
}


#if CMK_PERSISTENT_COMM
#include "persistent.c"
#endif
