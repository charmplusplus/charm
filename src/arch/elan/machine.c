/* Charm++ Machine Layer for ELAN network interface 
Developed by Sameer Kumar
*/

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <errno.h>
#include "converse.h"
#include <elan/elan.h>
#include <elan3/elan3.h>

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

#if CMK_BROADCAST_SPANNING_TREE
#  define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);
#else
#  define CMI_SET_BROADCAST_ROOT(msg, root)
#endif

ELAN_BASE     *elan_base;
ELAN_TPORT    *elan_port;
ELAN_QUEUE    *elan_q;

const int SMALL_MESSAGE_SIZE=8192;  /* Smallest message size queue 
                                       used for receiving short messages */
                                     
const int MID_MESSAGE_SIZE=65536;     /* Queue for larger messages 
                                          which need pre posted receives
                                          Message sizes greater will be 
                                          probe received adding 5us overhead*/

#define SYNC_MESSAGE_SIZE MID_MESSAGE_SIZE  
                             /* Message sizes greater will be 
                                sent synchronously thus avoiding copying*/

#define NON_BLOCKING_MSG  16     /* Message sizes greater 
                                    than this will be sent asynchronously*/
#define RECV_MSG_Q_SIZE 8
#define MID_MSG_Q_SIZE  8

ELAN_EVENT *esmall[RECV_MSG_Q_SIZE], *emid[MID_MSG_Q_SIZE], *elarge;

#define TAG_SMALL 0x1
#define TAG_LARGE_HEADER 0x3     /* Header that a large message is coming
                                    Not implemented yet */
#define TAG_MID   0x10
#define TAG_LARGE 0x100

int _Cmi_numpes;
int               _Cmi_mynode;    /* Which address space am I */
int               _Cmi_mynodesize;/* Number of processors in my address space */
int               _Cmi_numnodes;  /* Total number of address spaces */
int               _Cmi_numpes;    /* Total number of processors */
static int        Cmi_nodestart; /* First processor in this address space */ 
CpvDeclare(void*, CmiLocalQueue);

#define BLK_LEN  512

static int MsgQueueLen=0;
static int MsgQueueBytes=0;
static int request_max;
static int request_bytes;

#include "pcqueue.h"
PCQueue localSmallBufferQueue;
PCQueue localMidBufferQueue;

int outstandingMsgs[3000];

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
void CmiSyncSendPersistent(int destPE, int size, char *msg, 
                           PersistentHandle h);

void ElanSendQueuedMessages();
static void CmiReleaseSentMessages();

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

double starttimer;

void CmiAbort(const char *message);
static void PerrorExit(const char *msg);

void SendSpanningChildren(int size, char *msg);

#define TYPE_FIELD(buf) ((int *)buf)[0]
#define SIZE_FIELD(buf) ((int *)buf)[1]
#define CONV_SIZE_FIELD(buf) ((int *)buf)[2]
#define REF_FIELD(buf)  ((int *)buf)[3]

#define CONV_BUF_START(res) ((char *)res - 2*sizeof(int))
#define USER_BUF_START(res) ((char *)res - 4*sizeof(int))

#define DYNAMIC_MESSAGE 0
#define STATIC_MESSAGE 1
#define ELAN_MESSAGE 3


static void PerrorExit(const char *msg)
{
  perror(msg);
  exit(1);
}

/**************************  TIMER FUNCTIONS **************************/

#if CMK_TIMER_USE_SPECIAL

#include <sys/timers.h>
void CmiTimerInit(void)
{
    struct timespec tp;
    //starttimer =  elan_clock(elan_base->state); 
    getclock(TIMEOFDAY, &tp);
    starttimer = tp.tv_nsec;
    starttimer /= 1e9;
    starttimer += tp.tv_sec;
}

double CmiTimer(void)
{
    struct timespec tp;
    double cur_time;
    getclock(TIMEOFDAY, &tp);
    cur_time = tp.tv_nsec;
    cur_time  /= 1e9;
    cur_time += tp.tv_sec;
    return cur_time - starttimer;
    //return (elan_clock(elan_base->state) - starttimer)/1e9;
}

double CmiWallTimer(void)
{
    struct timespec tp;
    double cur_time;
    getclock(TIMEOFDAY, &tp);
    cur_time = tp.tv_nsec;
    cur_time  /= 1e9;
    cur_time += tp.tv_sec;
    return cur_time - starttimer;

    //return (elan_clock(elan_base->state) - starttimer)/1e9;
}

double CmiCpuTimer(void)
{
    struct timespec tp;
    double cur_time;
    getclock(TIMEOFDAY, &tp);
    cur_time = tp.tv_nsec;
    cur_time  /= 1e9;
    cur_time += tp.tv_sec;
    return cur_time - starttimer;

    //return (elan_clock(elan_base->state) - starttimer)/1e9;
}

#endif

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
int _Cmi_mype;
int _Cmi_myrank;

void CmiMemLock(void) {}
void CmiMemUnlock(void) {}

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

static void CmiStartThreads(char **argv)
{
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  _Cmi_mype = Cmi_nodestart;
  _Cmi_myrank = 0;
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
int CmiNodeFirst(int node) { return node*_Cmi_mynodesize; }
int CmiNodeSize(int node)  { return _Cmi_mynodesize; }
#endif

#ifndef CmiNodeOf
int CmiNodeOf(int pe)      { return (pe/_Cmi_mynodesize); }
int CmiRankOf(int pe)      { return pe%_Cmi_mynodesize; }
#endif

int CmiAllAsyncMsgsSent(void)
{
   SMSG_LIST *msg_tmp = sent_msgs;

   int done;
   
   CmiReleaseSentMessages();

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

  while ((msg_tmp) && (msg_tmp ->e != NULL) && 
         ((CmiCommHandle)(msg_tmp->e) != c))
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
              
              outstandingMsgs[msg_tmp->destpe] = 0;
              
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
      else {
          prev = msg_tmp;
          msg_tmp = msg_tmp->next;
      }
  }
  
  end_sent = prev;

#if CMK_PERSISTENT_COMM
  release_pmsg_list();
#endif

#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
  double rel_end_time = CmiWallTimer();
  if(rel_end_time > rel_start_time + 50/1e6)
      traceUserBracketEvent(20, rel_start_time, rel_end_time);
#endif
#endif
}

/* retflag = 0, receive as many messages as can be 
   and then post another receive
   retflag = 1, receive the first message and return 
   retflag =3 blocking receives

   Pump Msgs posts a circular queue of receives. The main idea 
   is that if a large number of receives are being posted only 
   minimum receive events should be polled. 

   So there are two indices, event_idx and post_idx. event_idx points
   to the first posted receive. So if a large number of messages come
   for a particular tag then this position in the queue will receive
   the first message. The new receives should be posted from
   post_idx. Notice that event_idx and post_idx are not the same. If a
   large number of messages are received then receives should be
   posted from the position of where first message was received.

*/
int PumpMsgs(int retflag)
{

    static char recv_small_done[RECV_MSG_Q_SIZE]; /*A list of flags which 
                                                    tells if a particular 
                                                    slot in the queue has 
                                                    a receive posted or not. */
    static char recv_mid_done[MID_MSG_Q_SIZE];  
    static int recv_large_done = 0;
    
    static char *sbuf[RECV_MSG_Q_SIZE];    /* Buffer of pointer to the 
                                              messages received */
    static char *mbuf[MID_MSG_Q_SIZE];
    static char *lbuf;
    
    static int event_idx = 0;             /* As defined earlier */
    static int post_idx = 0;

    static int event_m_idx = 0;
    static int post_m_idx = 0;

    static int nlarge_torecv = 0; /*this variable specifies how many
                                    large messages need to be received
                                    before we can block again.*/

    static int step1 = 0;   /* Large message are received in two
                               steps, first the envelope is probed by
                               posting a receive and then memory is
                               allocated for the message and the
                               message is finally received */

    int flg, res, rcount, mcount;
    char *msg = 0;
    
    int recd=0;
    int size= 0;
    int tag=0;
    
#ifndef CMK_OPTIMIZE 
    double pmp_start_time = CmiWallTimer();
#endif

    int ecount = 0, emcount = 0;
    while(1) {
        msg = 0;
        
        ecount = 0;
        for(rcount = 0; rcount < RECV_MSG_Q_SIZE; rcount ++){
            ecount = (rcount + post_idx) % RECV_MSG_Q_SIZE;
            if(!recv_small_done[ecount]) {
                sbuf[ecount] = (char *) CmiAlloc(SMALL_MESSAGE_SIZE);
		
                esmall[ecount] = elan_tportRxStart(elan_port, 0, 0, 0, 1, 
                                                   TAG_SMALL, sbuf[ecount], 
                                                   SMALL_MESSAGE_SIZE);
                recv_small_done[ecount] = 1;
            }
            else {
                ecount = (ecount + RECV_MSG_Q_SIZE - 1) % RECV_MSG_Q_SIZE;
                break;
            }
        }
        post_idx = ecount + 1;

        emcount = 0;
        for(mcount = 0; mcount < MID_MSG_Q_SIZE; mcount ++){
            emcount = (mcount + post_m_idx) % MID_MSG_Q_SIZE;
            if(!recv_mid_done[emcount]) {
                mbuf[emcount] = (char *) CmiAlloc(MID_MESSAGE_SIZE);
						
                emid[emcount] = elan_tportRxStart(elan_port, 0, 0, 0, -1, 
                                                  TAG_MID, mbuf[emcount], 
                                                  MID_MESSAGE_SIZE);
                recv_mid_done[emcount] = 1;
            }
            else {
                emcount = (emcount + MID_MSG_Q_SIZE - 1) % MID_MSG_Q_SIZE;
                break;
            }
        }
        post_m_idx = emcount + 1;        
        
        if(!recv_large_done) {
            elarge = elan_tportRxStart(elan_port, ELAN_TPORT_RXPROBE, 0, 0, 
                                       -1, TAG_LARGE, NULL, 0);
            recv_large_done = 1;
        }
    
        if(!step1 && elan_tportRxDone(elarge)) {
            elan_tportRxWait(elarge, NULL, NULL, &size );
      
            lbuf = (char *) CmiAlloc(size);
            elarge = elan_tportRxStart(elan_port, 0, 0, 0, -1, TAG_LARGE, 
                                       lbuf,size);
            step1 = 1;
        }
        
        if(step1 && elan_tportRxDone(elarge)) {
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
            
            if(blockingReceiveFlag)
                nlarge_torecv --;
        }

        emcount = 0;
        for(mcount = 0; mcount < MID_MSG_Q_SIZE; mcount ++){
            emcount = (mcount + event_m_idx) % MID_MSG_Q_SIZE;
            if(elan_tportRxDone(emid[emcount])) {
                elan_tportRxWait(emid[emcount], NULL, NULL, &size);
                
                msg = mbuf[emcount];
                mbuf[emcount] = NULL;
                
                recv_mid_done[emcount] = 0;
                flg = 1;
                
                CmiPushPE(CMI_DEST_RANK(msg), msg);
                
#if CMK_BROADCAST_SPANNING_TREE
                if (CMI_BROADCAST_ROOT(msg))
                    SendSpanningChildren(size, msg);
#endif
                if(blockingReceiveFlag)
                    nlarge_torecv --;
            }
            else {
                elan_deviceCheck(elan_base->state);
                emcount = (emcount + MID_MSG_Q_SIZE - 1) % MID_MSG_Q_SIZE;
                break;
            }
        }
        event_m_idx = emcount + 1;
        
        ecount = 0;
        for(rcount = 0; rcount < RECV_MSG_Q_SIZE; rcount ++){
            ecount = (rcount + event_idx) % RECV_MSG_Q_SIZE;
            if(elan_tportRxDone(esmall[ecount]) || 
               (retflag == 3 && nlarge_torecv == 0 && !flg)) {
                elan_tportRxWait(esmall[ecount], NULL, &tag, &size );
                
                msg = sbuf[ecount];
                sbuf[ecount] = NULL;
                
                recv_small_done[ecount] = 0;
                    
                if(tag == TAG_SMALL) {
                    flg = 1;
                    CmiPushPE(CMI_DEST_RANK(msg), msg);                
#if CMK_BROADCAST_SPANNING_TREE
                    if (CMI_BROADCAST_ROOT(msg))
                        SendSpanningChildren(size, msg);
#endif
                }
                else if(tag == TAG_LARGE_HEADER) {
                    //CmiPrintf("[%d] Received Header\n", CmiMyPe());
                    nlarge_torecv ++;
                    CmiFree(msg);
                }

                if(retflag == 3)
                    retflag = 1;
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
    return flag;
}

/*
void remote_get_wait_all(){
    elan_getWaitAll(elan_base->state, ELAN_WAIT_EVENT);
}

void remote_put_wait_all(){
    elan_putWaitAll(elan_base->state, ELAN_WAIT_EVENT);
}
*/

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
        if (PumpMsgs(1))  // PumpMsgs(0)
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

void elan_machine_broadcast(int size, void *buffer, int src){
    elan_bcast( elan_base->allGroup, buffer, size, src, 0); 
}

void enableBlockingReceives(){
    blockingReceiveFlag = 1;
}

void disableBlockingReceives(){
    blockingReceiveFlag = 0;
}

void CmiNotifyIdle(void)
{
    static int previousSleepTime = 0;
    CmiReleaseSentMessages();
    ElanSendQueuedMessages();
    
    if(!PumpMsgs(1) && blockingReceiveFlag /*&& (CmiMyPe() % 4 == 0)*/){
        if (!PCQueueEmpty(CmiGetState()->recv)) return; 
        if (!CdsFifo_Empty(CpvAccess(CmiLocalQueue))) return;
        if (!CqsEmpty(CpvAccess(CsdSchedQueue))) return;
        if (cur_unsent) return;
        PumpMsgs(3); 
    }
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


void ElanBasicSendFn(SMSG_LIST * ptr){
    int tag = 0, sync_mode = 0;
    
    if (ptr->size <= SMALL_MESSAGE_SIZE)
        tag = TAG_SMALL;
    else if (ptr->size < MID_MESSAGE_SIZE)
        tag = TAG_MID;
    else
        tag = TAG_LARGE;

    if(ptr->size > SYNC_MESSAGE_SIZE)
        sync_mode = ELAN_TPORT_TXSYNC;
    
    int tiny_msg = 0; //A sizeof(int) byte message 
    //sent to wake up a blocked process

    if(ptr->size > SMALL_MESSAGE_SIZE && blockingReceiveFlag) {
        elan_tportTxWait(elan_tportTxStart(elan_port, 0, ptr->destpe, 
                                           CmiMyPe(), TAG_LARGE_HEADER, 
                                           &tiny_msg, sizeof(int)));
    }

    ptr->e = elan_tportTxStart(elan_port, sync_mode, ptr->destpe, CmiMyPe(),
                               tag, ptr->msg, ptr->size);
    ptr->sent = 1;
    
    MsgQueueLen++;
    MsgQueueBytes += ptr->size;
    
    outstandingMsgs[ptr->destpe] = 1;
}

CmiCommHandle ElanSendFn(int destPE, int size, char *msg, int flag)
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
    
    if ((MsgQueueLen > request_max || MsgQueueBytes > request_bytes) 
        && (!flag)) {
        CmiReleaseSentMessages();
        PumpMsgs(1); //PumpMsgs(0) 
    }

    ElanSendQueuedMessages();

    if(MsgQueueLen > request_max || MsgQueueBytes > request_bytes 
       || outstandingMsgs[destPE]){
        
        if(sent_msgs==0)
            sent_msgs = msg_tmp;
        else
            end_sent->next = msg_tmp;
        end_sent = msg_tmp;
        
        if(cur_unsent == 0)
            cur_unsent = msg_tmp;
        
    }
    else{
        ElanBasicSendFn(msg_tmp);
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
    while (MsgQueueLen <= request_max && MsgQueueBytes <= request_bytes 
           && ptr != NULL) {

        if(!outstandingMsgs[ptr->destpe] && !ptr->sent)
            ElanBasicSendFn(ptr);
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

static void registerElanEvents() {
#ifndef CMK_OPTIMIZE
#if ! CMK_TRACE_IN_CHARM
    traceRegisterUserEvent("Pump Messages", 10);
    traceRegisterUserEvent("Release Sent Messages", 20);
    traceRegisterUserEvent("ELAN Send", 30);
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
  
  assert(startpe>=0 && startpe<_Cmi_numpes);

  for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
    int p = cs->pe-startpe;
    if (p<0) p+=_Cmi_numpes;
    p = BROADCAST_SPANNING_FACTOR*p + i;
    if (p > _Cmi_numpes - 1) break;
    p += startpe;
    p = p%_Cmi_numpes;
    assert(p>=0 && p<_Cmi_numpes && p!=cs->pe);
    CmiSyncSendFn1(p, size, msg);
  }
}

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
  CmiState cs = CmiGetState();
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, _Cmi_mype+1);
  SendSpanningChildren(size, msg);
#else
  int i ;
     
  for ( i=cs->pe+1; i<_Cmi_numpes; i++ ) 
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

  for ( i=cs->pe+1; i<_Cmi_numpes; i++ ) 
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
  
  for ( i=0; i<_Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
#endif
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)  
{
  int i ;

  for ( i=1; i<_Cmi_numpes; i++ ) 
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
     
  for ( i=0; i<_Cmi_numpes; i++ ) 
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

  // register elan events before trace module destoried
  registerElanEvents();

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

void elan_barrier(){
  elan_gsync(elan_base->allGroup);
}

void *elan_CmiAlloc(int size){
    char *res = NULL;
    char *buf;
    
    if(size <= SMALL_MESSAGE_SIZE + 2 * sizeof(int)) {
        size = SMALL_MESSAGE_SIZE + 4 * sizeof(int);
        if(!PCQueueEmpty(localSmallBufferQueue))
            buf = PCQueuePop(localSmallBufferQueue);
        else
            buf = (char *)malloc_nomigrate(size);
    }
    else if(size <= MID_MESSAGE_SIZE + 2 * sizeof(int)) {
        size = MID_MESSAGE_SIZE + 4 * sizeof(int);
        if(!PCQueueEmpty(localMidBufferQueue))
            buf = PCQueuePop(localMidBufferQueue);
        else
            buf = (char *)malloc_nomigrate(size);
    }
    else
        buf =(char *)malloc_nomigrate(size + 2 * sizeof(int));

    TYPE_FIELD(buf) = DYNAMIC_MESSAGE;
    SIZE_FIELD(buf) = size;
    res = (char *)((char *)buf + 2 * sizeof(int));
    return res;
}

void elan_CmiFree(void *res){

    char *buf = CONV_BUF_START(res);
    
    int type = TYPE_FIELD(buf);

    if(type == STATIC_MESSAGE)
        return;

    if(type == DYNAMIC_MESSAGE) {    
        
        //Called from Cmifree so we know the size and 
        //we dont hve to store it again
        int size = SIZE_FIELD(buf);
        if(size == SMALL_MESSAGE_SIZE + 4 * sizeof(int))
            PCQueuePush(localSmallBufferQueue, buf);
        else if (size == MID_MESSAGE_SIZE + 4 * sizeof(int))
            PCQueuePush(localMidBufferQueue, buf);
        else 
            free_nomigrate(buf);
        return;
    }

    //ELAN_MESSAGE
    elan_freeElan(elan_base->state, 
                  elan_elan2sdram(elan_base->state, 
                                  elan_main2elan(elan_base->state, buf)));
}

//Called from the application for static messages which 
//should not be freed by the system.
void *elan_CmiStaticAlloc(int size){
    char *res = NULL;
    char *buf =(char *)malloc_nomigrate(size + 4 *sizeof(int));

    TYPE_FIELD(buf) = STATIC_MESSAGE;
    SIZE_FIELD(buf) = size + 4 * sizeof(int);
    CONV_SIZE_FIELD(buf) = size;
    REF_FIELD(buf) = 0;    

    res = buf + 4 *sizeof(int);
    return res;
}

void elan_CmiStaticFree(void *res){
    char *buf = USER_BUF_START(res);
    if(TYPE_FIELD(buf) != STATIC_MESSAGE)
        return;

    free_nomigrate(buf);
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

  localSmallBufferQueue = PCQueueCreate();
  localMidBufferQueue = PCQueueCreate();

  if (!(elan_base = elan_baseInit())) {
    perror("Failed elan_baseInit()");
    exit(1);
  }

  elan_gsync(elan_base->allGroup);
  
  if ((elan_q = elan_gallocQueue(elan_base, elan_base->allGroup)) == NULL) {
    
    perror( "elan_gallocQueue failed" );
    exit (1);
  }
  
  int nslots = 32; //elan_base->tport_nslots * 2;
  
  //if(nslots < elan_base->state->nvp)
  //  nslots = elan_base->state->nvp;
  //if(nslots > 256)
  //  nslots = 256;
  
  if (!(elan_port = elan_tportInit(elan_base->state,
				   elan_q,
				   nslots /*elan_base->tport_nslots*/, 
				   elan_base->tport_smallmsg,
				   MID_MESSAGE_SIZE, //elan_base->tport_bigmsg,
				   elan_base->waitType, elan_base->retryCount,
				   &(elan_base->shm_key),
				   elan_base->shm_fifodepth, 
				   elan_base->shm_fragsize))) {
    
    perror("Failed to to initialise TPORT");
    exit(1);
  }
  
  elan_gsync(elan_base->allGroup);

  _Cmi_numnodes = elan_base->state->nvp;
  _Cmi_mynode =  elan_base->state->vp;

  /* processor per node */
  _Cmi_mynodesize = 1;
  CmiGetArgInt(argv,"+ppn", &_Cmi_mynodesize);

  if (_Cmi_mynodesize > 1 && _Cmi_mynode == 0) 
    CmiAbort("+ppn cannot be used in non SMP version!\n");
  
  _Cmi_numpes = _Cmi_numnodes * _Cmi_mynodesize;
  Cmi_nodestart = _Cmi_mynode * _Cmi_mynodesize;
  Cmi_argv = argv; Cmi_startfn = fn; Cmi_usrsched = usched;
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=_Cmi_numpes; n>1; n/=2 )
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

  if (CmiGetArgFlag(argv,"+enableBlockingReceives")) {
      blockingReceiveFlag = 1;
  }

  CmiTimerInit();
  msgBuf = PCQueueCreate();

  CsvInitialize(CmiNodeState, NodeState);
  CmiNodeStateInit(&CsvAccess(NodeState));

  int rms_nodes = 1, rms_procs = 1;

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

extern void CmiReference(void *blk);

#define ELAN_BUF_SIZE 4096
#define USE_NIC_MULTICAST 0
void CmiFreeListSendFn(int npes, int *pes, int len, char *msg)
{  
    static int ppn = 0;

    if(ppn == 0) {
        int rms_nodes = 1;
        int rms_procs = 1;
        if(getenv("RMS_NODES"))
            rms_nodes = atoi(getenv("RMS_NODES"));
        
        if(getenv("RMS_PROCS"))
            rms_procs = atoi(getenv("RMS_PROCS"));

        ppn = rms_procs/rms_nodes;
        if(ppn == 0)
            ppn = 4;
        //CmiPrintf("CmiListSyncSendAndFree PPN=%d\n", ppn);
    }    

    char* elan_buf = NULL;
    int i;

    char *msg_start = USER_BUF_START(msg);
    int rflag = REF_FIELD(msg_start); 

    if(rflag != 1) {   
        //Is being referenced by the application dont mess around
        for(i=0;i<npes;i++)
            CmiSyncSend(pes[i], len, msg);                        
        return;
    }

    for(i=0;i<npes;i++) {
        if(pes[i] != CmiMyPe()) {
            CmiReference(msg);   
            if(pes[i]/ppn == CmiMyPe()/ppn) 
                //dest in my node, send right away
                CmiSyncSendAndFree(pes[i], len, msg);            
        }              
        else  //Local message copy and send
            CmiSyncSend(pes[i], len, msg);        
    }

    CmiFree(msg);

#if USE_NIC_MULTICAST   
    //ULTIMATE HACK FOR PERFORMANCE, CLEAN UP LATER   
    if(len < ELAN_BUF_SIZE) {
        //Attempt to speedup Namd multicast, copy the message to local
        //elan memory and then send it several times from there. Should
        //improve performance as bandwidth is increased due to lower DMA
        //contention.
        
        void *elan_addr = elan_allocElan(elan_base->state, 8, ELAN_BUF_SIZE 
                                         + 4*sizeof(int));
        if(elan_addr == 0)
            CmiPrintf("ELAN ALLOC FAILED\n");
        
        elan_buf = (char*)elan_elan2main(elan_base->state, 
                                         elan_sdram2elan
                                         (elan_base->state, 
                                          elan_addr));            
        
        TYPE_FIELD(msg_start) = ELAN_MESSAGE;
        SIZE_FIELD(msg_start) = ELAN_BUF_SIZE + 4*sizeof(int);
        CONV_SIZE_FIELD(msg_start) = ELAN_BUF_SIZE;
        
        elan_wait(elan_put(elan_base->state, msg_start, elan_buf,
                           len + 4*sizeof(int), CmiMyPe()), ELAN_WAIT_EVENT);
        //memcpy(elan_buf, msg_start, len+ 4*sizeof(int));
        
        TYPE_FIELD(msg_start) = DYNAMIC_MESSAGE;
        SIZE_FIELD(msg_start) = len + 4 *sizeof(int);
        CONV_SIZE_FIELD(msg_start) = len;
        REF_FIELD(msg_start) = 1;
        
        //Actually free the message
        CmiFree(msg);

        msg = elan_buf + 4*sizeof(int);
    }
#endif  

    /*
      #if CMK_PERSISTENT_COMM
      if (phs) {
      CmiAssert(phsSize == npes);
      for(i=0;i<npes;i++) 
      CmiSyncSendPersistent(pes[i], len, msg, phs[i]);
      }
      else 
      #endif
    */
        
    for(i=0;i<npes;i++)
        if(pes[i] != CmiMyPe() && pes[i]/ppn != CmiMyPe()/ppn)
            //dest not in my node
            CmiSyncSendAndFree(pes[i], len, msg);        
}


#if CMK_PERSISTENT_COMM
#include "persistent.c"
#endif

#include "immediate.c"
