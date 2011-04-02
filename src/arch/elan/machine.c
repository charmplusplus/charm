/** @file
 * Elan machine layer
 * @ingroup Machine
*/

/* Charm++ Machine Layer for ELAN network interface 
Developed by Sameer Kumar
*/

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <errno.h>
#include "converse.h"
#include <elan/elan.h>
/*#include <elan3/elan3.h>*/

/*Support for ++debug: */
#include <unistd.h> /*For getpid()*/
#include <stdlib.h> /*For sleep()*/

#include "machine.h"
#include "pcqueue.h"

#if CMK_PERSISTENT_COMM
#include "persist_impl.h"
#endif

/* copy from elan/version.h */
#ifndef QSNETLIBS_VERSION_CODE
#define QSNETLIBS_VERSION(a,b,c)        (((a) << 16) + ((b) << 8) + (c))
#define QSNETLIBS_VERSION_CODE          QSNETLIBS_VERSION(1,3,0)
#endif

#define MAX_QLEN 1000
#define MAX_BYTES 1000000

#define USE_SHM 0

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
#define CMI_MESSAGE_SIZE(msg)            ((CmiMsgHeaderBasic *)msg)->size

#if CMK_BROADCAST_SPANNING_TREE
#  define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);
#else
#  define CMI_SET_BROADCAST_ROOT(msg, root)
#endif

ELAN_BASE     *elan_base;
ELAN_TPORT    *elan_port;
ELAN_QUEUE    *elan_q;

int enableGetBasedSend = 1;
int enableBufferPooling = 0;

int SMALL_MESSAGE_SIZE=4080;  /* Smallest message size queue 
                                 used for receiving short messages */
                                     
int MID_MESSAGE_SIZE=65536;     /* Queue for larger messages 
                                   which need pre posted receives
                                   Message sizes greater will be 
                                   probe received adding 5us overhead*/
#define SYNC_MESSAGE_SIZE MID_MESSAGE_SIZE * 10
                               /* Message sizes greater will be 
                                  sent synchronously thus avoiding copying*/

#define NON_BLOCKING_MSG  4     /* Message sizes greater 
                                    than this will be sent asynchronously*/
#define RECV_MSG_Q_SIZE  8   //Maximim queue size for short messages
#define MID_MSG_Q_SIZE   4   //Maximum queue size for mid-range messages

//Actual sizes, can also be set from the command line
int smallQSize = RECV_MSG_Q_SIZE;
int midQSize = MID_MSG_Q_SIZE;

ELAN_EVENT *esmall[RECV_MSG_Q_SIZE], *emid[MID_MSG_Q_SIZE], *elarge;

#define TAG_SMALL 0x1
#define TAG_LARGE_HEADER 0x3     /* Header that a large message is coming*/
#define TAG_GET_BASED_SEND 0x5     /* Header that a large message is coming
                                    as a get request*/
#define TAG_MID   0x10
#define TAG_LARGE 0x100

/*Release sent messages status to check for*/
#define BASIC_SEND 0
#define GET_BASED_SEND 1
#define RECEIVE_GET   2
#define GET_FINISHED_RECEIVE 3

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

typedef struct {
    char header[CmiMsgHeaderSizeBytes];
    int size;
    char* src_addr;
    char* flag_addr;
} GetHeader;

typedef struct msg_list {
    ELAN_EVENT *e;
    char *msg;
    struct msg_list *next;
    int size, destpe;
    int sent;
    
    //Fields for get based send
    int status;
    long done;
    long *flag_addr;
    char *gmsg;
    char *newmsg;       
    int is_broadcast;
} SMSG_LIST;


static int Cmi_dim;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;
static SMSG_LIST *cur_unsent=0;

void ElanSendQueuedMessages();
static int CmiReleaseSentMessages();

void ElanGetBasedSend(SMSG_LIST *ptr);
void handleGetHeader(char *msg, int src);
void processGetEnv(SMSG_LIST *ptr);

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

double starttimer;

void CmiAbort(const char *message);
static void PerrorExit(const char *msg);

void SendSpanningChildren(int size, char *msg);

typedef struct __elanChunkHeader {
  int type;
  int size;
} ElanChunkHeader;

typedef struct __chunkHeader {
  ElanChunkHeader elan;
  CmiChunkHeader conv;
} ChunkHeader;

#define TYPE_FIELD(buf)       (((ChunkHeader*)(buf))->elan.type)
#define SIZE_FIELD(buf)       (((ChunkHeader*)(buf))->elan.size)
#define CONV_SIZE_FIELD(buf)  (((ChunkHeader*)(buf))->conv.size)
#define REF_FIELD(buf)        (((ChunkHeader*)(buf))->conv.ref)

// CONV_BUF_START moves the res from pointing to the start of the elan chunk to the start of the converse chunk
#define CONV_BUF_START(res)     ((char*)(res) + sizeof(ElanChunkHeader))

// MACHINE_BUF_START moves the res from pointing to the start of the converse chunk to the start of the elan chunk
#define MACHINE_BUF_START(res)  ((char*)(res) - sizeof(ElanChunkHeader))

// USER_BUF_START moves the res from pointing to the start of the payload to the start of the elan chunk
#define USER_BUF_START(res)     ((char*)(res) - sizeof(ChunkHeader))

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
void CmiTimerInit(char **argv)
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

#include "immediate.c"

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
void CmiPushPE(int pe,void *msg)
{
  CmiState cs=CmiGetStateN(pe);
  MACHSTATE1(2,"Pushing message into %d's queue",pe);
#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
      /*CmiPrintf("[node %d] Immediate Message %d %d {{. \n", CmiMyNode(), CmiGetHandler(msg), _ImmediateMsgHandlerIdx);*/
      /*CmiHandleMessage(msg);*/
      CmiPushImmediateMsg(msg);
      /*CmiPrintf("[node %d] Immediate Message done.}} \n", CmiMyNode());*/
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
   SMSG_LIST *msg_tmp = NULL; 

   int done;
   
   CmiReleaseSentMessages();

   msg_tmp = sent_msgs;
   while((msg_tmp != cur_unsent) && (msg_tmp->e != NULL)){
       done = 0;
    
       if(elan_tportTxDone(msg_tmp->e))
           done = 1;
       else
#if USE_SHM 
           elan_deviceCheck(elan_base->state);
#else 
       ;
#endif

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
#if USE_SHM 
           elan_deviceCheck(elan_base->state);
#else 
       ;
#endif
    
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

#define MAX_RELEASE_POLL 4096

static int CmiReleaseSentMessages(void)
{
    SMSG_LIST *msg_tmp=sent_msgs;
    SMSG_LIST *prev=0;
    SMSG_LIST *temp;
    int done;
    int locked = 0;
    
#ifndef CMK_OPTIMIZE 
    double rel_start_time = CmiWallTimer();
#endif

#if CMK_PERSISTENT_COMM
  release_pmsg_list();
#endif

  int ncheck = MAX_RELEASE_POLL;

  while(msg_tmp != NULL && ncheck > 0){
      if(msg_tmp->sent) {
          done =0;

          if(msg_tmp->status == BASIC_SEND) {
              if(elan_tportTxDone(msg_tmp->e)) {
                  elan_tportTxWait(msg_tmp->e);
                  done = 1;
              }          
              else 
#if USE_SHM 
                  elan_deviceCheck(elan_base->state);
#else 
              ;
#endif
          }
          else {
              processGetEnv(msg_tmp);
              done = msg_tmp->done;
          }
          
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
              ncheck --;
          }
      }
      else {
          prev = msg_tmp;
          msg_tmp = msg_tmp->next;
      }
  }
  
  //if(msg_tmp)
  //  elan_deviceCheck(elan_base->state);

  end_sent = prev;

#if CMK_PERSISTENT_COMM
  release_pmsg_list();
#endif

#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
  {
  double rel_end_time = CmiWallTimer();
  if(rel_end_time > rel_start_time + 50/1e6)
      traceUserBracketEvent(20, rel_start_time, rel_end_time);
  }
#endif
#endif

  //So messages not finished sending
  if(msg_tmp != cur_unsent)
      return 0;
  else
      //All messages sent
      return 1;
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
#if QSNETLIBS_VERSION_CODE > QSNETLIBS_VERSION(1,4,0)
    unsigned long size= 0;
#else
    int size= 0;
#endif
    int tag=0;
    int src=-1;
    
#ifndef CMK_OPTIMIZE 
    double pmp_start_time = CmiWallTimer();
#endif

    int ecount = 0, emcount = 0;

#if CMK_PERSISTENT_COMM
    if (PumpPersistent()) return 1;
#endif
        
    while(1) {
        msg = 0;
        
        ecount = 0;
        for(rcount = 0; rcount < smallQSize; rcount ++){
            ecount = (rcount + post_idx) % smallQSize;
            if(!recv_small_done[ecount]) {
                sbuf[ecount] = (char *) CmiAlloc(SMALL_MESSAGE_SIZE);
		
                esmall[ecount] = elan_tportRxStart(elan_port, 0, 0, 0, 1, 
                                                   TAG_SMALL, sbuf[ecount], 
                                                   SMALL_MESSAGE_SIZE);
                recv_small_done[ecount] = 1;
            }
            else {
                ecount = (ecount + smallQSize - 1) % smallQSize;
                break;
            }
        }
        post_idx = ecount + 1;

        emcount = 0;
        for(mcount = 0; mcount < midQSize; mcount ++){
            emcount = (mcount + post_m_idx) % midQSize;
            if(!recv_mid_done[emcount]) {
                mbuf[emcount] = (char *) CmiAlloc(MID_MESSAGE_SIZE);
						
                emid[emcount] = elan_tportRxStart(elan_port, 0, 0, 0, -1, 
                                                  TAG_MID, mbuf[emcount], 
                                                  MID_MESSAGE_SIZE);
                recv_mid_done[emcount] = 1;
            }
            else {
                emcount = (emcount + midQSize - 1) % midQSize;
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
        for(mcount = 0; mcount < midQSize; mcount ++){
            emcount = (mcount + event_m_idx) % midQSize;
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
#if USE_SHM 
                elan_deviceCheck(elan_base->state);
#else 
                ;
#endif
                emcount = (emcount + midQSize - 1) % midQSize;
                break;
            }
        }
        event_m_idx = emcount + 1;
        
        ecount = 0;
        for(rcount = 0; rcount < smallQSize; rcount ++){
            ecount = (rcount + event_idx) % smallQSize;
            if(elan_tportRxDone(esmall[ecount]) || 
               (retflag == 3 && nlarge_torecv == 0 && !flg)) {
                elan_tportRxWait(esmall[ecount], &src, &tag, &size );
                
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
                else if(tag == TAG_GET_BASED_SEND) {
                    handleGetHeader(msg, src);
                }                    

                if(retflag == 3)
                    retflag = 1;
            }
            else {
#if USE_SHM 
                elan_deviceCheck(elan_base->state);
#else 
                ;
#endif
                ecount = (ecount + smallQSize - 1) % smallQSize;
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
#if CMK_IMMEDIATE_MSG && !CMK_SMP
            CmiHandleImmediate();
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
#if CMK_IMMEDIATE_MSG && !CMK_SMP
            CmiHandleImmediate();
#endif
            return flg;
        }
        
        recd = 1;
        flg = 0;
    }
#if CMK_IMMEDIATE_MSG && !CMK_SMP
    CmiHandleImmediate();
#endif
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
    register CmiState cs = CmiGetState();
    register void *msg = NULL;
    CmiIdleLock_checkMessage(&cs->idle);
    
    msg =  PCQueuePop(cs->recv); 
    
    if(msg) {
        return msg;
    }

    //get new messages and flush receive buffers
    PumpMsgs(1);           // PumpMsgs(0)
    //we are idle do more work
    CmiReleaseSentMessages();
    ElanSendQueuedMessages();

    msg =  PCQueuePop(cs->recv); 
    return msg;
}

void CmiPing() {
    CmiReleaseSentMessages();
    PumpMsgs(0);
    ElanSendQueuedMessages();
}


void enableBlockingReceives(){
    blockingReceiveFlag = 1;
}

void disableBlockingReceives(){
    blockingReceiveFlag = 0;
}

static int toggle = 0;  //Blocking receive posted only after all idle
                        //handlers are called

void CmiNotifyIdle(void)
{
    static int previousSleepTime = 0;
    CmiReleaseSentMessages();
    ElanSendQueuedMessages();

    PumpMsgs(1);    
    toggle = 0;
}

void CmiNotifyStillIdle(void)
{
    static int previousSleepTime = 0;
    CmiReleaseSentMessages();
    ElanSendQueuedMessages();
    
    if(!PumpMsgs(1) && blockingReceiveFlag && toggle){
        if (!PCQueueEmpty(CmiGetState()->recv)) return; 
        if (!CdsFifo_Empty(CpvAccess(CmiLocalQueue))) return;
        if (!CqsEmpty(CpvAccess(CsdSchedQueue))) return;
        if (sent_msgs) return;
        if (cur_unsent) return;
        PumpMsgs(3); 
    }
    toggle = 1;
}


 
#if CMK_IMMEDIATE_MSG
void CmiProbeImmediateMsg()
{
    PumpMsgs(0);
}
#endif
/********************* MESSAGE SEND FUNCTIONS ******************/

static void CmiSendSelf(char *msg)
{
#if CMK_IMMEDIATE_MSG
    if (CmiIsImmediate(msg)) {
      /* CmiBecomeNonImmediate(msg); */
      CmiHandleImmediateMessage(msg);
      return;
    }
#endif
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}

void CmiSyncSendFn(int destPE, int size, char *msg)
{
    CmiState cs = CmiGetState();
    
    char *dupmsg;
    dupmsg = (char *) CmiAlloc(size);
    memcpy(dupmsg, msg, size);
    
    //  CmiPrintf("Setting root to %d\n", 0);
    CMI_SET_BROADCAST_ROOT(dupmsg, 0);
    
    if (cs->pe==destPE)
        CmiSendSelf(dupmsg);
    else
        CmiAsyncSendFn(destPE, size, dupmsg);
}

void ElanBasicSendFn(SMSG_LIST * ptr){
    int tag = 0, sync_mode = 0;
    int tiny_msg = 0;
    
    ptr->status = BASIC_SEND;
    
    if (ptr->size <= SMALL_MESSAGE_SIZE)
        tag = TAG_SMALL;
    else if (ptr->size < MID_MESSAGE_SIZE)
        tag = TAG_MID;
    else {        
        if(!ptr->is_broadcast && enableGetBasedSend) {
            ElanGetBasedSend(ptr);
            return;
        }
        
        tag = TAG_LARGE;
    }

    //if(ptr->size > SYNC_MESSAGE_SIZE)
    //  sync_mode = ELAN_TPORT_TXSYNC;
    
    tiny_msg = 0; //A sizeof(int) byte message 
    //sent to wake up a blocked process
    
    //WAKE A PROCESS SLEEPING ON A BLOCKING RECEIVE UP,
    //WITH A SMALL MESSAGE THAT MATCHES THE TAG OF A SMALL MESSAGE
    //BUT IS ACTUALLY A MID OR LARGE MESSAGE
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
    
    if(destPE == cs->pe) {
        char *dupmsg = (char *) CmiAlloc(size);
        memcpy(dupmsg, msg, size);
        CmiSendSelf(dupmsg);
        return 0;
    }
    
    CQdCreate(CpvAccess(cQdState), 1);
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
    msg_tmp->is_broadcast = flag;

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

void ElanGetBasedSend(SMSG_LIST *msg_tmp){
    //CmiPrintf("using get based send\n");
    GetHeader *gmsg = (GetHeader *) CmiAlloc(sizeof(GetHeader));

    gmsg->src_addr = msg_tmp->msg;
    gmsg->size = msg_tmp->size;
    CMI_SET_BROADCAST_ROOT(gmsg, 0);
    
    msg_tmp->sent = 1;
    msg_tmp->e = NULL;
    msg_tmp->status = GET_BASED_SEND;
    msg_tmp->done = 0;
    
    msg_tmp->gmsg = (char *)gmsg;
    gmsg->flag_addr = (char *)&(msg_tmp->done);
    
    msg_tmp->e = elan_tportTxStart(elan_port, 0, msg_tmp->destpe, CmiMyPe(), 
                                   TAG_GET_BASED_SEND, gmsg, sizeof(GetHeader));
    
    MsgQueueLen++;
    MsgQueueBytes += msg_tmp->size;
    
    outstandingMsgs[msg_tmp->destpe] = 1;
}

void handleGetHeader(char *msg, int src){
    GetHeader *gmsg = (GetHeader *) msg;

    char *newmsg = CmiAlloc(gmsg->size);
    
    SMSG_LIST *msg_tmp = (SMSG_LIST *)CmiAlloc(sizeof(SMSG_LIST));
    msg_tmp->msg = msg;
    msg_tmp->next = 0;
    msg_tmp->size = gmsg->size;
    msg_tmp->sent = 1;
    msg_tmp->destpe = src;
    msg_tmp->status = RECEIVE_GET;
    msg_tmp->done = 0;
    msg_tmp->flag_addr = (long *)gmsg->flag_addr;
    msg_tmp->newmsg = newmsg;
    
    msg_tmp->e = elan_get(elan_base->state, gmsg->src_addr, msg_tmp->newmsg, gmsg->size, src);
    
    if(sent_msgs==0) {
        sent_msgs = msg_tmp;
        end_sent = msg_tmp;
    }
    else {
        msg_tmp->next = sent_msgs;
        sent_msgs = msg_tmp;
    }
}

long trueFlag = 1;
void processGetEnv(SMSG_LIST *ptr){

    if(ptr->status == BASIC_SEND)
        return;

    if(ptr->status == GET_BASED_SEND) {
        if(ptr->gmsg != NULL) {
            if(!elan_tportTxDone(ptr->e)) {
#if USE_SHM 
                elan_deviceCheck(elan_base->state);
#else 
                ;
#endif
                return;                    
            } 
            
            elan_tportTxWait(ptr->e);
            CmiFree(ptr->gmsg);
            ptr->gmsg =  NULL;
        }
        
        return;
    }        

    //RECEIVE_GET or FINISHED_RECEIVE, a get or a put poll in either case
    int flag = elan_poll(ptr->e, ELAN_POLL_EVENT);

    if(!flag)
        return;
    
    if(ptr->status == RECEIVE_GET){
        ptr->e = elan_put(elan_base->state, &trueFlag, ptr->flag_addr,
                          sizeof(long), ptr->destpe);
        ptr->status = GET_FINISHED_RECEIVE;
        CmiPushPE(0, ptr->newmsg);
        
        /*
          #if CMK_BROADCAST_SPANNING_TREE
          if (CMI_BROADCAST_ROOT(ptr->newmsg))
          SendSpanningChildren(ptr->size, ptr->newmsg);
          #endif
        */
        
        return;
    }
    
    if (ptr->status == GET_FINISHED_RECEIVE)
        ptr->done = 1;
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
        CmiSendSelf(msg);
    } 
    else { 
        if(size <= NON_BLOCKING_MSG) {
            CQdCreate(CpvAccess(cQdState), 1);
            (void)elan_tportTxWait(elan_tportTxStart(elan_port, 0, destPE, CmiMyPe(), TAG_SMALL, msg, size));
            CmiFree(msg);
        }
        else
            CmiAsyncSendFn(destPE, size, msg);
    }
    
#ifndef CMK_OPTIMIZE 
#if ! CMK_TRACE_IN_CHARM
    {
    double snd_end_time = CmiWallTimer();
    if(snd_end_time > snd_start_time + 5/1e6) 
        traceUserBracketEvent(30, snd_start_time, snd_end_time);
    if((snd_end_time > snd_start_time + 5/1e3) && stretchFlag)
        CmiPrintf("%d:Stretched Send to %d at %5.3lfs of %5.5lf ms\n", CmiMyPe(), destPE, snd_end_time, (snd_end_time - snd_start_time)*1e3);
    }
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
        CmiSendSelf(dupmsg);
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
  CMI_SET_BROADCAST_ROOT(msg, CmiMyPe() + 1);
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
    while(!CmiAllAsyncMsgsSent() || cur_unsent) {
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

  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,NULL);
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
  
  PumpMsgs(0);
  elan_gsync(elan_base->allGroup);

  /* Converse initialization finishes, immediate messages can be processed.
     node barrier previously should take care of the node synchronization */
  _immediateReady = 1;

  if (!everReturn) {
    Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
    if (Cmi_usrsched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

//fast low level global barrier
void elan_barrier(){
    
    while(!CmiReleaseSentMessages() || cur_unsent) {
        PumpMsgs(0);
        ElanSendQueuedMessages();
    }   
    
    elan_gsync(elan_base->allGroup);
}

//synchronous fast hardware broadcast
//size = size of the message to be broadcast
//buffer = address of the message
//root = root and source of the broadcast
void elan_machine_broadcast(int size, void *buffer, int root) {
    
    while(!CmiReleaseSentMessages() || cur_unsent) {
        PumpMsgs(0);
        ElanSendQueuedMessages();
    }   
    
    elan_bcast(elan_base->allGroup, buffer, size, root, 0); 
}

typedef void (* ELAN_REDUCER)(void *in, void *inout, int *count, void *handle);

//nelem = number of elements
//size = size of data type
//data = data to be reduced
//fn = function pointer of the function which will do the reduction
//dest = destination buffer where data is finally stored on all processors
void elan_machine_allreduce(int nelem, int size, void * data, void *dest, 
                            ELAN_REDUCER fn){
    
    while(!CmiReleaseSentMessages() || cur_unsent) {
        PumpMsgs(0);
        ElanSendQueuedMessages();
    }   
    
    //ELAN reduction call here
    elan_reduce (elan_base->allGroup, data, dest, size, nelem, fn, NULL, 0, 0, ELAN_REDUCE_COMMUTE | ELAN_RESULT_ALL | elan_base->group_flags, 0);
}

//nelem = number of elements
//size = size of data type
//data = data to be reduced
//fn = function pointer of the function which will do the reduction
//dest = destination buffer where data is finally stored, NULL on all non root processors
//root = root of the reduction where the data will be returned
void elan_machine_reduce(int nelem, int size, void * data, void *dest, ELAN_REDUCER fn, int root){
    
    while(sent_msgs || cur_unsent) {
        CmiReleaseSentMessages();
        PumpMsgs(0);
        ElanSendQueuedMessages();
    }   
    
    printf("Machine Called Reduce %d\n", sent_msgs);

    //ELAN reduction call here
    elan_reduce (elan_base->allGroup, data, dest, size, nelem, fn, NULL, 0, 0, ELAN_REDUCE_COMMUTE|elan_base->group_flags , root);
}

// NOTE: The "size" parameter already includes the size of the CmiChunkHeader data structure.
void *elan_CmiAlloc(int size){
    char *res = NULL;
    char *buf;
    
    int alloc_size = size;
    if(enableBufferPooling) {

        if (size <= SMALL_MESSAGE_SIZE + sizeof(ElanChunkHeader)) {
	   alloc_size = SMALL_MESSAGE_SIZE + sizeof(ChunkHeader);
           size = SMALL_MESSAGE_SIZE + sizeof(CmiChunkHeader);

#if CMK_PERSISTENT_COMM
            //Put a footer at the end the message of it. This footer only will be sent with the pers. message
            alloc_size += sizeof(int)*2;
#endif

            if(!PCQueueEmpty(localSmallBufferQueue))
                buf = PCQueuePop(localSmallBufferQueue);
            else
                buf = (char *)malloc_nomigrate(alloc_size);
        }
        /*
        else if (size < MID_MESSAGE_SIZE + sizeof(ElanChunkHeader)) {
            alloc_size = MID_MESSAGE_SIZE + sizeof(ChunkHeader);
            size = MID_MESSAGE_SIZE + sizeof(CmiChunkHeader);

#if CMK_PERSISTENT_COMM
            //Put a footer at the end the message of it. This footer will only be sent with the pers. message
            alloc_size += sizeof(int)*2;
#endif
            
            if(!PCQueueEmpty(localMidBufferQueue))
                buf = PCQueuePop(localMidBufferQueue);
            else
                buf = (char *)malloc_nomigrate(alloc_size);
        }
        */
        else {

	    alloc_size = size + sizeof(ElanChunkHeader);
            
#if CMK_PERSISTENT_COMM
            //Put a footer at the end the message of it. This footer will be sent with the pers. message
            alloc_size += sizeof(int)*2;
#endif            
            
            buf =(char *)malloc_nomigrate(alloc_size);
        }
    }
    else {

        alloc_size = size + sizeof(ElanChunkHeader);
#if CMK_PERSISTENT_COMM
        //Put a footer at the end the message of it. This footer will be sent with the pers. message
        alloc_size += sizeof(int)*2;
#endif
        buf =(char *)malloc_nomigrate(alloc_size);
    }
    
    TYPE_FIELD(buf) = DYNAMIC_MESSAGE;
    SIZE_FIELD(buf) = size; //size of user part of the buffer, excludes machine header and persistent footer
    res = CONV_BUF_START(buf);  //That is where the converse message starts
    return res;
}

void elan_CmiFree(void *res){

    char *buf = MACHINE_BUF_START(res);    
    int type = TYPE_FIELD(buf);

    if(type == STATIC_MESSAGE)
        return;
    
    if(type == DYNAMIC_MESSAGE) {    
        
        if(enableBufferPooling) {
            //Called from Cmifree so we know the size and 
            //we dont hve to store it again
            int size = SIZE_FIELD(buf);

            if (size == SMALL_MESSAGE_SIZE + sizeof(ElanChunkHeader))

                //I knew I allocated the SMALL_MESSSAGE_SIZE of user data, 
                //so I can put it back to the pool
                PCQueuePush(localSmallBufferQueue, buf);
            /*
              else if (size == MID_MESSAGE_SIZE + sizeof(ElanChunkHeader))
              PCQueuePush(localMidBufferQueue, buf);
            */
            else 
                free_nomigrate(buf);
        }
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

    char *buf = (char*)malloc_nomigrate(size + sizeof(ChunkHeader));

    TYPE_FIELD(buf) = STATIC_MESSAGE;
    SIZE_FIELD(buf) = size + sizeof(ChunkHeader);
    CONV_SIZE_FIELD(buf) = size;
    REF_FIELD(buf) = 0;    

    res = buf + sizeof(ChunkHeader);
    return res;
}

void elan_CmiStaticFree(void *res){
    char *buf = USER_BUF_START(res);
    if(TYPE_FIELD(buf) != STATIC_MESSAGE)
        return;

    free_nomigrate(buf);
}

/*
//Called from the application for messages allocated in the NIC
void *elan_CmiAllocElan(int size){
    char *res = NULL;
    char *buf = NULL;

    void *elan_addr = elan_allocElan(elan_base->state, 8, len + sizeof(ChunkHeader));

    if(elan_addr == 0)
        CmiPrintf("ELAN ALLOC FAILED\n");
    
    elan_buf = (char*)elan_elan2main(elan_base->state, elan_sdram2elan
                                     (elan_base->state, elan_addr));            

    TYPE_FIELD(buf) = ELAN_MESSAGE;
    SIZE_FIELD(buf) = size + sizeof(ChunkHeader);
    CONV_SIZE_FIELD(buf) = size;
    REF_FIELD(buf) = 0;    

    res = buf + sizeof(ChunkHeader);
    return res;
}
*/

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int n,i ;
  int nslots;

#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif

#if USE_SHM 
  putenv("LIBELAN_SHM_ENABLE=1");
#else
  putenv("LIBELAN_SHM_ENABLE=0");
#endif

  localSmallBufferQueue = PCQueueCreate();
  localMidBufferQueue = PCQueueCreate();

  if (!(elan_base = 
#if QSNETLIBS_VERSION_CODE > QSNETLIBS_VERSION(1,4,0)
	elan_baseInit(0)
#else
	elan_baseInit()
#endif
     ))
  {
      perror("Failed elan_baseInit()");
      exit(1);
  }

  elan_gsync(elan_base->allGroup);
  
  if ((elan_q = elan_gallocQueue(elan_base, elan_base->allGroup)) == NULL) {
    
    perror( "elan_gallocQueue failed" );
    exit (1);
  }
  
  nslots = elan_base->tport_nslots * 2;
  
  //if(nslots < elan_base->state->nvp)
  //  nslots = elan_base->state->nvp;
  //if(nslots > 256)
  //  nslots = 256;
  
  if (!(elan_port = elan_tportInit(elan_base->state,
				   elan_q,
                                   nslots,
				   //elan_base->tport_nslots, 
				   elan_base->tport_smallmsg,
				   //MID_MESSAGE_SIZE, 
                                   elan_base->tport_bigmsg,
#if QSNETLIBS_VERSION_CODE > QSNETLIBS_VERSION(1,4,1)
	 			   10000000, //elan_base->tport_stripemsg,
#endif
				   elan_base->waitType, elan_base->retryCount,
				   &(elan_base->shm_key),
				   elan_base->shm_fifodepth, 
				   elan_base->shm_fragsize
#if QSNETLIBS_VERSION_CODE > QSNETLIBS_VERSION(1,4,11)
	 			   ,0
#endif
))) {
    
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

  if (CmiGetArgFlag(argv,"+enableBufferPooling")) {
      enableBufferPooling = 1;
  }

  CmiGetArgInt(argv,"+smallMessageSize", &SMALL_MESSAGE_SIZE);
  CmiGetArgInt(argv,"+midMessageSize", &MID_MESSAGE_SIZE);

  CmiGetArgInt(argv,"+smallQSize", &smallQSize);
  CmiGetArgInt(argv,"+midQSize", &midQSize);

  if(smallQSize > RECV_MSG_Q_SIZE) {
      CmiPrintf("Warning : resetting smallQSize to %d\n", RECV_MSG_Q_SIZE);
      smallQSize = RECV_MSG_Q_SIZE;
  }
  
  if(midQSize > MID_MSG_Q_SIZE) {
      CmiPrintf("Warning : resetting midQSize to %d\n", MID_MSG_Q_SIZE);
      midQSize = MID_MSG_Q_SIZE;
  }

  enableGetBasedSend = CmiGetArgFlag(argv,"+enableGetBasedSend");

  //CmiPrintf("SMALL_MESSAGE_SIZE = %d\n", SMALL_MESSAGE_SIZE);

  CmiTimerInit(argv);
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
#if 1
#define ELAN_BUF_SIZE MID_MESSAGE_SIZE
#define USE_NIC_MULTICAST 0

void CmiFreeListSendFn(int npes, int *pes, int len, char *msg)
{  
    static int ppn = 0;
    char* elan_buf = NULL;
    char *msg_start;
    int rflag;
    int i;

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

	//ppn = 1;
        //CmiPrintf("CmiListSyncSendAndFree PPN=%d\n", ppn);
    }    

    for(i=0;i<npes;i++) 
        if(pes[i] == CmiMyPe()) {
            CmiSyncSend(pes[i], len, msg); 
        }

    int nremote = 0;
    msg_start = USER_BUF_START(msg);
    rflag = REF_FIELD(msg_start); 

    for(i=0;i<npes;i++) {
        if(pes[i] != CmiMyPe()) {
            if(pes[i]/ppn == CmiMyPe()/ppn) {
                CmiReference(msg);               
                //dest in my node, send right away
                CmiSyncSendAndFree(pes[i], len, msg);  
            }
            else
                nremote++;
        } 
    }
    
    if(nremote == 0) {
        CmiFree(msg);
        return;
    }
    
    REF_FIELD(msg_start) += nremote - 1;

#if USE_NIC_MULTICAST   
    //ULTIMATE HACK FOR PERFORMANCE, CLEAN UP LATER   
    if(len > 2048 && len < MID_MESSAGE_SIZE && nremote > 4) {
        //Attempt to speedup Namd multicast, copy the message to local
        //elan memory and then send it several times from there. Should
        //improve performance as bandwidth is increased due to lower DMA
        //contention.
        
        void *elan_addr = elan_allocElan(elan_base->state, 8, len + sizeof(ChunkHeader));
        if(elan_addr == 0) {
            CmiPrintf("ELAN ALLOC FAILED, sending data from main memory instead\n");
        }
        else {
            elan_buf = (char*)elan_elan2main(elan_base->state, 
                                             elan_sdram2elan
                                             (elan_base->state, 
                                              elan_addr));            
            
            int old_size_field = SIZE_FIELD(msg_start);
            int old_conv_size_field = CONV_SIZE_FIELD(msg_start);
            int old_ref_field = REF_FIELD(msg_start);
            
            TYPE_FIELD(msg_start) = ELAN_MESSAGE;
            SIZE_FIELD(msg_start) = len + sizeof(ChunkHeader);
            CONV_SIZE_FIELD(msg_start) = len;
            REF_FIELD(msg_start) = nremote; 
            
            elan_wait(elan_get(elan_base->state,
                               msg_start,
                               elan_buf,
                               len + sizeof(ChunkHeader),
                               CmiMyPe()
                              ),
                      ELAN_WAIT_EVENT
                     );
	    //memcpy(elan_buf, msg_start, len + sizeof(ChunkHeader));
            
            REF_FIELD(msg_start) = old_ref_field; 
            TYPE_FIELD(msg_start) = DYNAMIC_MESSAGE;
            SIZE_FIELD(msg_start) = old_size_field ;
            CONV_SIZE_FIELD(msg_start) = old_conv_size_field;
            
            //Actually free the message        
            for(i=0;i<npes;i++)
                if(pes[i] != CmiMyPe() && pes[i]/ppn != CmiMyPe()/ppn)
                    CmiFree(msg);                
            
            msg = elan_buf + sizeof(ChunkHeader);
        }
    }
#endif  
    
    for(i=0;i<npes;i++) {
        if(pes[i] != CmiMyPe() && pes[i]/ppn != CmiMyPe()/ppn) {
            //dest not in my node
            CmiSyncSendAndFree(pes[i], len, msg);        
            //if(len > SMALL_MESSAGE_SIZE)
            //  elan_barrier();                
        }        
    }
}

#else

void CmiFreeListSendFn(int npes, int *pes, int len, char *msg)
{
    int i;
    for(i=0;i<npes;i++) {
        CmiSyncSend(pes[i], len, msg);
    }
    CmiFree(msg);
}
#endif 

void CmiBarrier()
{
    elan_gsync(elan_base->allGroup);
}

/* a simple barrier - everyone sends a message to pe 0 and go on */
/* it is ok here since we have real elan barrier */
void CmiBarrierZero()
{
    elan_gsync(elan_base->allGroup);
}

#if CMK_PERSISTENT_COMM
#include "persistent.c"
#endif

