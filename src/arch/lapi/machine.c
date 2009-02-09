/*****************************************************************************
LAPI version of machine layer
Based on the template machine layer

Developed by 
Filippo Gioachin   03/23/05
************************************************************************/

#include <lapi.h>

#include "converse.h"

/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
*/
#include <assert.h>
#include <errno.h>

/*Support for ++debug: */
#include <unistd.h> /*For getpid()*/
#include <stdlib.h> /*For sleep()*/

#include "machine.h"
#include "pcqueue.h"

/* #define MAX_QLEN 200 */

/*
    To reduce the buffer used in broadcast and distribute the load from 
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of 
  spanning tree broadcast algorithm.
    This will use the fourth short in message as an indicator of spanning tree
  root.
*/

#define BROADCAST_SPANNING_FACTOR        CMK_SPANTREE_MAXSPAN

#define CMI_BROADCAST_ROOT(msg)          ((CmiMsgHeaderBasic *)msg)->root
#define CMI_DEST_RANK(msg)               ((CmiMsgHeaderBasic *)msg)->rank


static int lapiDebugMode=0;
/* static int lapiInterruptMode=0; */

CsvDeclare(int, lapiInterruptMode);

/* static int SHORT_MESSAGE_SIZE=512; */

static void ConverseRunPE(int everReturn);
static void PerrorExit(const char *msg);
static int Cmi_nodestart;   /* First processor in this node - stupid need due to
			       machine-smp.h that uses it!!  */

#include "machine-smp.c"


/* Variables describing the processor ID */

#if CMK_SHARED_VARS_UNAVAILABLE

int _Cmi_mype;
int _Cmi_numpes;
int _Cmi_myrank;

void CmiMemLock() {}
void CmiMemUnlock() {}

static struct CmiStateStruct Cmi_state;

#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

void CmiYield() { sleep(0); }

#elif CMK_SHARED_VARS_POSIX_THREADS_SMP

int _Cmi_numpes;
int _Cmi_mynodesize;
int _Cmi_mynode;
int _Cmi_numnodes;

int CmiMyPe(void) { return CmiGetState()->pe; }

int CmiMyRank(void) { return CmiGetState()->rank; }

int CmiNodeFirst(int node) { return node*_Cmi_mynodesize; }
int CmiNodeSize(int node)  { return _Cmi_mynodesize; }

int CmiNodeOf(int pe)      { return (pe/_Cmi_mynodesize); }
int CmiRankOf(int pe)      { return pe-(CmiNodeOf(pe)*_Cmi_mynodesize); }

#endif

CpvDeclare(void*, CmiLocalQueue);

#define DGRAM_NODEMESSAGE   (0xFB)

#if CMK_NODE_QUEUE_AVAILABLE
#define NODE_BROADCAST_OTHERS (-1)
#define NODE_BROADCAST_ALL    (-2)
#endif

#define check_lapi(routine,args) \
        check_lapi_err(routine args, #routine, __LINE__);

static void check_lapi_err(int returnCode,const char *routine,int line) {
        if (returnCode!=LAPI_SUCCESS) {
                char errMsg[LAPI_MAX_ERR_STRING];
                LAPI_Msg_string(returnCode,errMsg);
                fprintf(stderr,"Fatal LAPI error while executing %s at %s:%d\n"
                        "  Description: %s\n", routine, __FILE__, line, errMsg);
                CmiAbort("Fatal LAPI error");
        }
}

/*
static void CommunicationServer(int sleepTime);
static void CommunicationServerThread(int sleepTime);

typedef struct msg_list {
  char *msg;
  struct msg_list *next;
  int size, destpe;
  
  // LAPI Stuff Here 
  lapi_cntr_t lapiSendCounter;
  
} SMSG_LIST;

static int MsgQueueLen=0;
static int request_max;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;
*/

/* static int no_outstanding_sends=0; */
/*FLAG: consume outstanding Isends in scheduler loop*/

static lapi_handle_t lapiContext;
static lapi_long_t lapiHeaderHandler = 1;

/* double starttimer; */

/* void CmiAbort(const char *message); */

void SendSpanningChildren(int size, char *msg);
void SendHypercube(int size, char *msg);

char *CopyMsg(char *msg, int len) {
  char *copy = (char *)CmiAlloc(len);
  if (!copy)
      fprintf(stderr, "Out of memory\n");
  memcpy(copy, msg, len);
  return copy;
}

typedef struct ProcState {
  CmiNodeLock  recvLock;		    /* for cs->recv */
} ProcState;

static ProcState  *procState;

/*
#if CMK_SMP

static PCQueue sendMsgBuf;
static CmiNodeLock  sendMsgBufLock = NULL;        // for sendMsgBuf 

#endif
*/

#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(CmiNodeState, NodeState);
#endif

#if CMK_IMMEDIATE_MSG
#include "immediate.c"
#endif

#if CMK_SHARED_VARS_UNAVAILABLE

/* To conform with the call made in SMP mode */
static void CmiStartThreads(char **argv) {
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  /* _Cmi_mype = Cmi_nodestart; // already set! 
  */
  _Cmi_myrank = 0;
}
#endif

/* Add a message to this processor's receive queue, pe is a rank */
static void CmiPushPE(int pe,void *msg) {
  CmiState cs = CmiGetStateN(pe);
  MACHSTATE3(3,"[%p] Pushing message into rank %d's queue %p {",CmiGetState(),pe, cs->recv);

#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
    MACHSTATE1(3, "[%p] Handling Immediate Message",CmiGetState());
    CMI_DEST_RANK(msg) = pe;
    CmiPushImmediateMsg(msg);
    CmiHandleImmediate();
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
static void CmiPushNode(void *msg) {
  MACHSTATE1(3,"[%p] Pushing message into NodeRecv queue",CmiGetState());

#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
    MACHSTATE1(3, "[%p] Handling Immediate Message",CmiGetState());
    CMI_DEST_RANK(msg) = 0;
    CmiPushImmediateMsg(msg);
    CmiHandleImmediate();
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

  MACHSTATE(3,"Pushing message into NodeRecv queue {");
}
#endif

/*
Functions to release the space used by sent messages. This is handled by the
sender completion handler.

static int CmiAllAsyncMsgsSent(void) {
   SMSG_LIST *msg_tmp = sent_msgs;
   
   int done;
     
   while(msg_tmp!=0) {
    done = 0;

    check_lapi(LAPI_Getcntr,(lapiContext, &msg_tmp->lapiSendCounter, &done));

    if(!done)
      return 0;
    
    msg_tmp = msg_tmp->next;
   }
   return 1;
}

int CmiAsyncMsgSent(CmiCommHandle c) {
     
  SMSG_LIST *msg_tmp = sent_msgs;
  int done;

  while ((msg_tmp) && ((CmiCommHandle)(msg_tmp) != c))
    msg_tmp = msg_tmp->next;
  if(msg_tmp) {
    done = 0;

    check_lapi(LAPI_Getcntr,(lapiContext, &msg_tmp->lapiSendCounter, &done));

    return ((done)?1:0);
  } else {
    return 1;
  }
}

static void CmiReleaseSentMessages(void) {
  SMSG_LIST *msg_tmp=sent_msgs;
  SMSG_LIST *prev=0;
  SMSG_LIST *temp;
  int done;
     
  MACHSTATE1(2,"CmiReleaseSentMessages begin on %d {", CmiMyPe());

  while(msg_tmp!=0) {
    done =0;
    
    check_lapi(LAPI_Getcntr,(lapiContext, &msg_tmp->lapiSendCounter, &done));

    if(done) {
      MACHSTATE2(3,"CmiReleaseSentMessages release one %d to %d", CmiMyPe(), msg_tmp->destpe);
      MsgQueueLen--;
      
      // Release the message
      temp = msg_tmp->next;
      if(prev==0)  // first message
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
*/

/* lapi completion handler */
static void PumpMsgsComplete(lapi_handle_t *myLapiContext, void *am_info) {
  int i;
  char *msg = am_info;
  
  int nbytes = ((CmiMsgHeaderBasic *)msg)->size;
  
  MACHSTATE1(2,"[%p] PumpMsgsComplete begin {",CmiGetState());
  
#if CMK_NODE_QUEUE_AVAILABLE
  if (CMI_DEST_RANK(msg)==DGRAM_NODEMESSAGE)
    CmiPushNode(msg);
  else
#endif
    CmiPushPE(CMI_DEST_RANK(msg), msg);

  /*
  if(!myLapiContext)
    return;
  */

#if CMK_BROADCAST_SPANNING_TREE
  if (CMI_BROADCAST_ROOT(msg)) {
    SendSpanningChildren(nbytes, msg);
  }
  
#elif CMK_BROADCAST_HYPERCUBE
  if (CMI_BROADCAST_ROOT(msg)) {
    SendHypercube(nbytes, msg);
  }
#endif

#if CMK_SMP
  if (CMI_BROADCAST_ROOT(msg) && CMI_DEST_RANK(msg)!=DGRAM_NODEMESSAGE) {
    assert(CMI_DEST_RANK(msg)==0);
    for (i=1; i<CmiMyNodeSize(); ++i) {
      CmiPushPE(i, msg);
    }
  }
#endif

  MACHSTATE(2,"} PumpMsgsComplete end ");
  return;
}

/* lapi header handler */
static void* PumpMsgsBegin(lapi_handle_t *myLapiContext,
			   void *hdr, uint *uhdr_len,
			   lapi_return_info_t *msg_info,
			   compl_hndlr_t **comp_h, void **comp_am_info) {
  void *msg_buf;
  MACHSTATE1(2,"[%p] PumpMsgsBegin begin {",CmiGetState());
  /*
  if(msg_info->udata_one_pkt_ptr != NULL) {
    // it means that all the data has already arrived
    msg_buf = CopyMsg(msg_info->udata_one_pkt_ptr, msg_info->msg_len);
    PumpMsgsComplete(myLapiContext, msg_buf);

    *comp_h = NULL;
    *comp_am_info = NULL;
    MACHSTATE(2,"} PumpMsgsBegin end single message");
    return NULL;
  } else {
  */
    /* prepare the space for receiving the data, set the completion handler to
       be executed inline */
    msg_buf = (void *)CmiAlloc(msg_info->msg_len);
    
    msg_info->ret_flags = LAPI_SEND_REPLY;
    *comp_h = PumpMsgsComplete;
    *comp_am_info = msg_buf;
    MACHSTATE(2,"} PumpMsgsBegin end");
    return msg_buf;
    /*}*/
}

/* lapi sender handlers*/
static void ReleaseMsg(lapi_handle_t *myLapiContext, void *msg, lapi_sh_info_t *info) {
  MACHSTATE2(2,"[%p] ReleaseMsg begin %p {",CmiGetState(),msg);
  check_lapi_err(info->reason, "ReleaseMsg", __LINE__);
  CmiFree(msg);
  MACHSTATE(2,"} ReleaseMsg end");
}

static void DeliveredMsg(lapi_handle_t *myLapiContext, void *msg, lapi_sh_info_t *info) {
  MACHSTATE1(2,"[%p] DeliveredMsg begin {",CmiGetState());
  check_lapi_err(info->reason, "DeliveredMsg", __LINE__);
  *((int *)msg) = *((int *)msg) - 1;
  MACHSTATE(2,"} DeliveredMsg end");
}

/*
#if CMK_SMP

static int inexit = 0;

static int MsgQueueEmpty() {
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

// test if all processors recv queues are empty
static int RecvQueueEmpty() {
  int i;
  for (i=0; i<_Cmi_mynodesize; i++) {
    CmiState cs=CmiGetStateN(i);
    if (!PCQueueEmpty(cs->recv)) return 0;
  }
  return 1;
}
*/

/**
CommunicationServer calls MPI to send messages in the queues and probe message from network.
*/
/*
static void CommunicationServer(int sleepTime) {
  // PumpMsgs();
  CmiReleaseSentMessages();
  SendMsgBuf(); 

  if (inexit == CmiMyNodeSize()) {
    MACHSTATE(2, "CommunicationServer exiting {");
#if 0
    while(!MsgQueueEmpty() || !CmiAllAsyncMsgsSent() || !RecvQueueEmpty()) {
#endif
    while(!MsgQueueEmpty() || !CmiAllAsyncMsgsSent()) {
      CmiReleaseSentMessages();
      SendMsgBuf(); 
      // PumpMsgs();
    }

    MACHSTATE(2, "CommunicationServer barrier begin {");

    check_lapi(LAPI_Gfence,(lapiContext));

    MACHSTATE(2, "} CommunicationServer barrier end");
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyNode() == 0){
      CmiPrintf("End of program\n");
    }
#endif
    MACHSTATE(2, "} CommunicationServer EXIT");
    check_lapi(LAPI_Term,(lapiContext));
    exit(0);   
  }
}
#endif
*/

/*
static void CommunicationServerThread(int sleepTime) {
#if CMK_SMP
  CommunicationServer(sleepTime);
#endif
#if CMK_IMMEDIATE_MSG
  CmiHandleImmediate();
#endif
}
*/

#if CMK_NODE_QUEUE_AVAILABLE
char *CmiGetNonLocalNodeQ(void) {
  CmiState cs = CmiGetState();
  char *result = 0;
  CmiIdleLock_checkMessage(&cs->idle);
  if(!PCQueueEmpty(CsvAccess(NodeState).NodeRecv)) {
    MACHSTATE2(3,"[%p] CmiGetNonLocalNodeQ begin %d {",CmiGetState(),CmiMyPe());
    CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
    result = (char *) PCQueuePop(CsvAccess(NodeState).NodeRecv);
    CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);
    MACHSTATE1(3,"} CmiGetNonLocalNodeQ end %d ", CmiMyPe());
  }
  return result;
}
#endif

void *CmiGetNonLocal(void) {
  CmiState cs = CmiGetState();
  CmiIdleLock_checkMessage(&cs->idle);
  MACHSTATE2(3,"[%p] CmiGetNonLocal %d",CmiGetState(),CmiMyPe());
  return PCQueuePop(cs->recv);
  /*
  static int count=0;
  CmiState cs = CmiGetState();
  void *msg;
  CmiIdleLock_checkMessage(&cs->idle);

  // un-necessary locking since only one processor pulls the queue, and the queue is lock safe

  //CmiLock(procState[cs->rank].recvLock);
  msg =  PCQueuePop(cs->recv); 
  //CmiUnlock(procState[cs->rank].recvLock);

#if ! CMK_SMP
  if (no_outstanding_sends) {
    while (MsgQueueLen>0) {
      CmiReleaseSentMessages();
      // PumpMsgs() ??
    }
  }

  if(!msg) {
    CmiReleaseSentMessages();
    // Potentially put a flag here!!
    return  PCQueuePop(cs->recv);
  }
#endif
  return msg;
*/
}

void CmiNotifyIdle(void) {
  /* CmiReleaseSentMessages(); */
  LAPI_Probe(lapiContext);
  CmiYield();
}
 
/* user call to handle immediate message, since there is no ServerThread polling
   messages (lapi does all the polling) every thread is authorized to process
   immediate messages. If we are not in lapiInterruptMode check for progress.
*/
#if CMK_IMMEDIATE_MSG
void CmiMachineProgressImpl() {
  MACHSTATE1(2,"[%p] Probing Immediate Messages",CmiGetState());
  if (!CsvAccess(lapiInterruptMode)) LAPI_Probe(lapiContext);
  MACHSTATE1(3, "[%p] Handling Immediate Message",CmiGetState());
  CmiHandleImmediate();
}
#endif

/* These two barriers are only needed by CmiTimerInit to synchronize all the
   threads. They do not need to provide a general barrier. */
int CmiBarrier() {return 0;}
int CmiBarrierZero() {return 0;}

/********************* MESSAGE SEND FUNCTIONS ******************/

static void CmiSendSelf(char *msg) {
  MACHSTATE1(3,"[%p] Sending itself a message {",CmiGetState());

#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
    MACHSTATE1(3, "[%p] Handling Immediate Message",CmiGetState());
    /* CmiBecomeNonImmediate(msg); */
    CmiPushImmediateMsg(msg);
    CmiHandleImmediate();
    return;
  }
#endif
  CQdCreate(CpvAccess(cQdState), 1);
  CdsFifo_Enqueue(CmiGetState()->localqueue,msg);

  MACHSTATE(3,"} Sending itself a message");
}

/* the field deliverable is used to know if the message can be encoded into the
   destPE queue without duplication (for usage of SMP). If it has already been
   duplicated (and therefore is deliverable), we do not want to copy it again,
   while if it has not been copied we must do it before enqueuing it. */
void lapiSendFn(int destPE, int size, char *msg, scompl_hndlr_t *shdlr, void *sinfo, int deliverable) {
  /* CmiState cs = CmiGetState(); */
  CmiUInt2  rank, node;
  lapi_xfer_t xfer_cmd;
     
  MACHSTATE1(2,"[%p] lapiSendFn begin {",CmiGetState());
  CQdCreate(CpvAccess(cQdState), 1);
  node = CmiNodeOf(destPE);
#if CMK_SMP
  rank = CmiRankOf(destPE);
  
  if (node == CmiMyNode())  {
    if (deliverable) {
      CmiPushPE(rank, msg);
      /* the acknowledge of delivery must not be called */
    } else {
      CmiPushPE(rank, CopyMsg(msg, size));
      /* acknowledge that the message has been delivered */
      lapi_sh_info_t lapiInfo;
      lapiInfo.src = node;
      lapiInfo.reason = LAPI_SUCCESS;
      (*shdlr)(&lapiContext, sinfo, &lapiInfo);
    }
    return;
  }
  /*
  CMI_DEST_RANK(msg) = rank;
#else
  // non smp
  CMI_DEST_RANK(msg) = 0;	// rank is always 0
  */
#endif
  /* The rank is now set by the caller function! */

  /* send the message out of the processor */

  /*
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  while (MsgQueueLen > request_max) {
    //printf("Waiting for %d messages to be sent\n", MsgQueueLen);
    CmiReleaseSentMessages();
  }

  check_lapi(LAPI_Setcntr, (*my_context, &msg_tmp->lapiSendCounter, 0));
  */
  if (CMI_DEST_RANK(msg) > 10) MACHSTATE2(5, "Error!! in lapiSendFn! destPe=%d, destRank=%d",destPE,CMI_DEST_RANK(msg));

  xfer_cmd.Am.Xfer_type = LAPI_AM_XFER;
  xfer_cmd.Am.flags     = 0;
  xfer_cmd.Am.tgt       = node;
  xfer_cmd.Am.hdr_hdl   = lapiHeaderHandler;
  xfer_cmd.Am.uhdr_len  = 0;
  xfer_cmd.Am.uhdr      = NULL;
  xfer_cmd.Am.udata     = msg;
  xfer_cmd.Am.udata_len = size;
  xfer_cmd.Am.shdlr     = shdlr;
  xfer_cmd.Am.sinfo     = sinfo;
  xfer_cmd.Am.tgt_cntr  = NULL;
  xfer_cmd.Am.org_cntr  = NULL;
  xfer_cmd.Am.cmpl_cntr = NULL;

  check_lapi(LAPI_Xfer,(lapiContext, &xfer_cmd));

  MACHSTATE(2,"} lapiSendFn end");
  /*
  if(size < SHORT_MESSAGE_SIZE && my_context != NULL) {
    check_lapi(LAPI_Amsend,(*my_context, node, (void *)PumpMsgsBegin, 
			    msg, size, 0, 0, NULL, NULL,
			    &msg_tmp->lapiSendCounter));      
  }
  else {

    if(my_context == NULL)
      my_context = &lapiContext;
    
    check_lapi(LAPI_Amsend,(*my_context, destPE,
			    (void *)PumpMsgsBegin, 0, 0, msg, size,
			    NULL, NULL, &msg_tmp->lapiSendCounter));
  }
  */
  /*
  MsgQueueLen++;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
  return (CmiCommHandle) msg_tmp;
  */
}

void CmiSyncSendFn(int destPE, int size, char *msg) {
  CmiState cs = CmiGetState();
  /*char *dupmsg = (char *) CmiAlloc(size);
    memcpy(dupmsg, msg, size); */
  char *dupmsg = CopyMsg(msg, size);

  MACHSTATE1(3,"[%p] Sending sync message begin {",CmiGetState());
  CMI_BROADCAST_ROOT(dupmsg) = 0;
  CMI_DEST_RANK(dupmsg) = CmiRankOf(destPE);

  if (cs->pe==destPE) {
    CmiSendSelf(dupmsg);
  } else {
    lapiSendFn(destPE, size, dupmsg, ReleaseMsg, dupmsg, 1);
    /*CmiAsyncSendFn(destPE, size, dupmsg); */
  }
  MACHSTATE(3,"} Sending sync message end");
}

/*
#if CMK_SMP

// called by communication thread in SMP
static int SendMsgBuf() {
  SMSG_LIST *msg_tmp;
  char *msg;
  int node, rank, size;
  int i;
  int sent = 0;

  MACHSTATE(2,"SendMsgBuf begin {");

  // single message sending queue
  CmiLock(sendMsgBufLock);
  msg_tmp = (SMSG_LIST *)PCQueuePop(sendMsgBuf);
  CmiUnlock(sendMsgBufLock);
  while (NULL != msg_tmp)
    {

      node = msg_tmp->destpe;
      size = msg_tmp->size;
      msg = msg_tmp->msg;
      msg_tmp->next = 0;
      while (MsgQueueLen > request_max) {
	CmiReleaseSentMessages();
	// PumpMsgs();
      }
      
      MACHSTATE2(3,"LAPI_Amsend to node %d rank: %d{", node,CMI_DEST_RANK(msg));
      
      check_lapi(LAPI_Setcntr, (lapiContext, &msg_tmp->lapiSendCounter, 0));

      if(size < SHORT_MESSAGE_SIZE) 
	check_lapi(LAPI_Amsend,(lapiContext, node,
				(void *)PumpMsgsBegin, msg, size, 0, 0
				NULL, &msg_tmp->lapiSendCounter, NULL));      
      else
	check_lapi(LAPI_Amsend,(lapiContext, node,
				(void *)PumpMsgsBegin, 0,0, msg, size,
				NULL, &msg_tmp->lapiSendCounter, NULL));

      MACHSTATE(3,"}LAPI_Amsend end");
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
  MACHSTATE(2,"}SendMsgBuf end ");
  return sent;
}

void EnqueueMsg(void *m, int size, int node) {
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
*/

/*
CmiCommHandle lapiSendFn(lapi_handle_t *my_context, int destPE, 
			     int size, char *msg) {
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
  
  // non smp
  CMI_DEST_RANK(msg) = 0;	// rank is always 0
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  while (MsgQueueLen > request_max) {
    //printf("Waiting for %d messages to be sent\n", MsgQueueLen);
    CmiReleaseSentMessages();
  }

  check_lapi(LAPI_Setcntr, (*my_context, &msg_tmp->lapiSendCounter, 0));

  if(size < SHORT_MESSAGE_SIZE && my_context != NULL) {
    check_lapi(LAPI_Amsend,(*my_context, node, (void *)PumpMsgsBegin, 
			    msg, size, 0, 0, NULL, NULL,
			    &msg_tmp->lapiSendCounter));      
  }
  else {

    if(my_context == NULL)
      my_context = &lapiContext;
    
    check_lapi(LAPI_Amsend,(*my_context, destPE,
			    (void *)PumpMsgsBegin, 0, 0, msg, size,
			    NULL, NULL, &msg_tmp->lapiSendCounter));
  }

  MsgQueueLen++;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
  return (CmiCommHandle) msg_tmp;
#endif
}
*/

int CmiAsyncMsgSent(CmiCommHandle handle) {
  return (*((int *)handle) == 0)?1:0;
}

void CmiReleaseCommHandle(CmiCommHandle handle) {
#ifndef CMK_OPTIMIZE
  if (*((int *)handle) != 0) CmiAbort("Released a CmiCommHandle not free!");
#endif
  free(handle);
}

/* the CmiCommHandle returned is a pointer to the location of an int. When it is
   set to 1 the message is available. */
CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg) {
  MACHSTATE1(3,"[%p] Sending async message begin {",CmiGetState());
  void *handle;
  CmiState cs = CmiGetState();
  CMI_BROADCAST_ROOT(msg) = 0;
  CMI_DEST_RANK(msg) = CmiRankOf(destPE);

  /* if we are the destination, send ourself a copy of the message */
  if (cs->pe==destPE) {
    CmiSendSelf(CopyMsg(msg, size));
    MACHSTATE(3,"} Sending async message end");
    return 0;
  }

  handle = malloc(sizeof(int));
  *((int *)handle) = 1;
  lapiSendFn(destPE, size, msg, DeliveredMsg, handle, 0);
  /* the message may have been duplicated and already delivered if we are in SMP
     mode and the destination is on the same node, but there is no optimized
     check for that. */
  MACHSTATE(3,"} Sending async message end");
  return handle;
}

void CmiFreeSendFn(int destPE, int size, char *msg) {
  MACHSTATE1(3,"[%p] Sending sync free message begin {",CmiGetState());
  CmiState cs = CmiGetState();
  CMI_BROADCAST_ROOT(msg) = 0;
  CMI_DEST_RANK(msg) = CmiRankOf(destPE);

  if (cs->pe==destPE) {
    CmiSendSelf(msg);
  } else {
    lapiSendFn(destPE, size, msg, ReleaseMsg, msg, 1);
    /*CmiAsyncSendFn(destPE, size, msg);*/
  }
  MACHSTATE(3,"} Sending sync free message end");
}

#if CMK_NODE_QUEUE_AVAILABLE

static void CmiSendNodeSelf(char *msg) {
  CmiState cs;
  MACHSTATE1(3,"[%p] Sending itself a node message {",CmiGetState());

#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
    MACHSTATE1(3, "[%p] Handling Immediate Message {",CmiGetState());
    CMI_DEST_RANK(msg) = 0;
    CmiPushImmediateMsg(msg);
    CmiHandleImmediate();
    MACHSTATE(3, "} Handling Immediate Message end");
    return;
  }
#endif
  CQdCreate(CpvAccess(cQdState), 1);
  CmiLock(CsvAccess(NodeState).CmiNodeRecvLock);
  PCQueuePush(CsvAccess(NodeState).NodeRecv, msg);
  CmiUnlock(CsvAccess(NodeState).CmiNodeRecvLock);

  cs=CmiGetStateN(0);
  CmiIdleLock_addMessage(&cs->idle);

  MACHSTATE(3,"} Sending itself a node message");
}

void CmiSyncNodeSendFn(int destNode, int size, char *msg) {
  char *dupmsg = CopyMsg(msg, size);

  MACHSTATE1(3,"[%p] Sending sync node message begin {",CmiGetState());
  CMI_BROADCAST_ROOT(dupmsg) = 0;
  CMI_DEST_RANK(dupmsg) = DGRAM_NODEMESSAGE;

  if (CmiMyNode()==destNode) {
    CmiSendNodeSelf(dupmsg);
  } else {
    lapiSendFn(CmiNodeFirst(destNode), size, dupmsg, ReleaseMsg, dupmsg, 1);
  }
  MACHSTATE(3,"} Sending sync node message end");
}

CmiCommHandle CmiAsyncNodeSendFn(int destNode, int size, char *msg) {
  void *handle;
  CMI_BROADCAST_ROOT(msg) = 0;
  CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;

  MACHSTATE1(3,"[%p] Sending async node message begin {",CmiGetState());
  /* if we are the destination, send ourself a copy of the message */
  if (CmiMyNode()==destNode) {
    CmiSendNodeSelf(CopyMsg(msg, size));
    MACHSTATE(3,"} Sending async node message end");
    return 0;
  }

  handle = malloc(sizeof(int));
  *((int *)handle) = 1;
  lapiSendFn(CmiNodeFirst(destNode), size, msg, DeliveredMsg, handle, 0);
  /* the message may have been duplicated and already delivered if we are in SMP
     mode and the destination is on the same node, but there is no optimized
     check for that. */
  MACHSTATE(3,"} Sending async node message end");
  return handle;

  /*
  int i;
  SMSG_LIST *msg_tmp;
  char *dupmsg;
     
  CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
  switch (dstNode) {
  case NODE_BROADCAST_ALL:
    CmiSendNodeSelf((char *)CopyMsg(msg,size));
  case NODE_BROADCAST_OTHERS:
    CQdCreate(CpvAccess(cQdState), _Cmi_numnodes-1);
    for (i=0; i<_Cmi_numnodes; i++)
      if (i!=_Cmi_mynode) {
        EnqueueMsg((char *)CopyMsg(msg,size), size, i);
      }
    break;
  default:
    dupmsg = (char *)CopyMsg(msg,size);
    if(dstNode == _Cmi_mynode) {
      CmiSendNodeSelf(dupmsg);
    }
    else {
      CQdCreate(CpvAccess(cQdState), 1);
      EnqueueMsg(dupmsg, size, dstNode);
    }
  }
  return 0;
  */
}

void CmiFreeNodeSendFn(int destNode, int size, char *msg) {
  CMI_BROADCAST_ROOT(msg) = 0;
  CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;

  MACHSTATE1(3,"[%p] Sending sync free node message begin {",CmiGetState());
  if (CmiMyNode()==destNode) {
    CmiSendNodeSelf(msg);
  } else {
    lapiSendFn(CmiNodeFirst(destNode), size, msg, ReleaseMsg, msg, 1);
  }
  MACHSTATE(3,"} Sending sync free node message end");
}

#endif

/*********************** BROADCAST FUNCTIONS **********************/

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg) {
  int startnode = CMI_BROADCAST_ROOT(msg)-1;
  int i;
  char *dupmsg;

  assert(startnode>=0 && startnode<CmiNumNodes());

  for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
    int p = CmiMyNode() - startnode;
    if (p<0) p+=CmiNumNodes();
    p = BROADCAST_SPANNING_FACTOR*p + i;
    if (p > CmiNumNodes() - 1) break;
    p += startnode;
    p = p%CmiNumNodes();
    assert(p>=0 && p<CmiNumNodes() && p!=CmiMyNode());
#if CMK_BROADCAST_USE_CMIREFERENCE
    CmiReference(msg);
    lapiSendFn(CmiNodeFirst(p), size, msg, ReleaseMsg, msg, 0);
#else
    dupmsg = CopyMsg(msg, size);
    lapiSendFn(CmiNodeFirst(p), size, dupmsg, ReleaseMsg, dupmsg, 1);
#endif
    /*CmiSyncSendFn1(p, size, msg); */
  }
}

int Cmi_log_of_2 (int i) {
  int m;
  for (m=0; i>(1<<m); ++m);
  return m;
}

/* send msg along the hypercube in broadcast. (Filippo Gioachin) */
void SendHypercube(int size, char *msg) {
  int srcPeNumber, tmp, k, num_pes, *dest_pes;

  srcPeNumber = CMI_BROADCAST_ROOT(msg);
  tmp = srcPeNumber ^ CmiMyPe();
  k = Cmi_log_of_2(CmiNumPes()) + 2;
  if (tmp) {
    do {--k;} while (!(tmp>>k));
  }

  /* now 'k' is the last dimension in the hypercube used for exchange */
  CMI_BROADCAST_ROOT(msg) = CmiMyPe();  /* where the message is coming from */
  dest_pes = (int *)malloc(k*sizeof(int));
  --k;  /* next dimension in the cube to be used */
  num_pes = HypercubeGetBcastDestinations(CmiMyPe(), CmiNumPes(), k, dest_pes);
  for (k=0; k<num_pes; ++k) {
#if CMI_BROADCAST_USE_CMIREFERENCE
    CmiReference(msg);
    CmiSyncSendAndFree(dest_pes[k], size, msg);
#else
    CmiSyncSend(dest_pes[k], size, msg);
#endif
  }
  free(dest_pes);
}

void CmiSyncBroadcastGeneralFn(int size, char *msg) {    /* ALL_EXCEPT_ME  */
  int i, rank;
  MACHSTATE1(3,"[%p] Sending sync broadcast message begin {",CmiGetState());
#if CMK_BROADCAST_SPANNING_TREE
  CMI_BROADCAST_ROOT(msg) = CmiMyPe()+1;
  SendSpanningChildren(size, msg);
  
#elif CMK_BROADCAST_HYPERCUBE
  CMI_BROADCAST_ROOT(msg) = CmiMyPe()+1;
  SendHypercube(size, msg);
    
#else
  CmiState cs = CmiGetState();
  char *dupmsg;

  CMI_BROADCAST_ROOT(msg) = 0;
#if CMK_BROADCAST_USE_CMIREFERENCE
  for (i=cs->pe+1; i<CmiNumPes(); i++) {
    CmiReference(msg);
    lapiSendFn(i, size, msg, ReleaseMsg, msg, 0);
    /*CmiSyncSendFn(i, size, msg) ;*/
  }
  for (i=0; i<cs->pe; i++) {
    CmiReference(msg);
    lapiSendFn(i, size, msg, ReleaseMsg, msg, 0);
    /*CmiSyncSendFn(i, size,msg) ;*/
  }
#else
  for (i=cs->pe+1; i<CmiNumPes(); i++) {
    dupmsg = CopyMsg(msg, size);
    lapiSendFn(i, size, dupmsg, ReleaseMsg, dupmsg, 1);
    /*CmiSyncSendFn(i, size, msg) ;*/
  }
  for (i=0; i<cs->pe; i++) {
    dupmsg = CopyMsg(msg, size);
    lapiSendFn(i, size, dupmsg, ReleaseMsg, dupmsg, 1);
    /*CmiSyncSendFn(i, size,msg) ;*/
  }
#endif
#endif

#if CMK_SMP
  /* deliver local node messages */
  if (CMI_DEST_RANK(msg)!=DGRAM_NODEMESSAGE) {
    rank = CmiMyRank();
    for (i=0; i<CmiMyNodeSize(); ++i) {
      if (i != rank) CmiPushPE(i, CopyMsg(msg, size));
    }
  }
#endif
  MACHSTATE(3,"} Sending sync broadcast message end");
}

CmiCommHandle CmiAsyncBroadcastGeneralFn(int size, char *msg) {
  CmiState cs = CmiGetState();
  int i, rank;

  MACHSTATE1(3,"[%p] Sending async broadcast message from {",CmiGetState());
  CMI_BROADCAST_ROOT(msg) = 0;
  void *handle = malloc(sizeof(int));
  *((int *)handle) = CmiNumPes()-1;
  for (i=cs->pe+1; i<CmiNumPes(); i++) {
    lapiSendFn(i, size, msg, DeliveredMsg, handle, 0);
  }
  for (i=0; i<cs->pe; i++) {
    lapiSendFn(i, size, msg, DeliveredMsg, handle, 0);
  }
#if CMK_SMP
  /* deliver local node messages */
  if (CMI_DEST_RANK(msg)!=DGRAM_NODEMESSAGE) {
    rank = CmiMyRank();
    for (i=0; i<CmiMyNodeSize(); ++i) {
      if (i != rank) CmiPushPE(i, CopyMsg(msg, size));
    }
  }
#endif
  MACHSTATE(3,"} Sending async broadcast message end");
  return handle;
}

void CmiSyncBroadcastFn(int size, char *msg) {
  CMI_DEST_RANK(msg) = 0;
  CmiSyncBroadcastGeneralFn(size, msg);
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg) {
  CMI_DEST_RANK(msg) = 0;
  return CmiAsyncBroadcastGeneralFn(size, msg);
}

void CmiFreeBroadcastFn(int size, char *msg) {
   CmiSyncBroadcastFn(size,msg);
   CmiFree(msg);
}

void CmiSyncBroadcastAllFn(int size, char *msg) {       /* All including me */
  CmiSendSelf(CopyMsg(msg, size));
  CmiSyncBroadcastFn(size, msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg) {
  CmiSendSelf(CopyMsg(msg, size));
  return CmiAsyncBroadcastFn(size, msg);
}

void CmiFreeBroadcastAllFn(int size, char *msg) {       /* All including me */
  CmiSendSelf(CopyMsg(msg, size));
  CmiSyncBroadcastFn(size, msg);
  CmiFree(msg);
}

#if CMK_NODE_QUEUE_AVAILABLE

void CmiSyncNodeBroadcastFn(int size, char *msg) {
  CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
  CmiSyncBroadcastGeneralFn(size, msg);
}

CmiCommHandle CmiAsyncNodeBroadcastFn(int size, char *msg) {
   CMI_DEST_RANK(msg) = DGRAM_NODEMESSAGE;
   CmiAsyncBroadcastGeneralFn(size, msg);
}

void CmiFreeNodeBroadcastFn(int size, char *msg) {
  CmiSyncNodeBroadcastFn(size, msg);
  CmiFree(msg);
}

void CmiSyncNodeBroadcastAllFn(int size, char *msg) {
  CmiSendNodeSelf(msg);
  CmiSyncNodeBroadcastFn(size, msg);
}

CmiCommHandle CmiAsyncNodeBroadcastAllFn(int size, char *msg) {
  CmiSendNodeSelf(msg);
  return CmiAsyncNodeBroadcastFn(size, msg);
}

void CmiFreeNodeBroadcastAllFn(int size, char *msg) {
  CmiSendNodeSelf(CopyMsg(msg, size));
  CmiSyncNodeBroadcastFn(size, msg);
  CmiFree(msg);
}

#endif

#if ! CMK_MULTICAST_LIST_USE_COMMON_CODE

void CmiSyncListSendFn(int, int *, int, char*) {

}

CmiCommHandle CmiAsyncListSendFn(int, int *, int, char*) {

}

void CmiFreeListSendFn(int, int *, int, char*) {

}

#endif

#if ! CMK_VECTOR_SEND_USES_COMMON_CODE

void CmiSyncVectorSend(int, int, int *, char **) {

}

CmiCommHandle CmiAsyncVectorSend(int, int, int *, char **) {

}

void CmiSyncVectorSendAndFree(int, int, int *, char **) {

}

#endif


/************************** MAIN ***********************************/

static volatile int inexit = 0;

void ConverseExit(void) {
  MACHSTATE2(2, "[%d-%p] entering ConverseExit",CmiMyPe(),CmiGetState());
#if CMK_SMP
  if (CmiMyRank() != 0) {
    CmiCommLock();
    inexit++;
    CmiCommUnlock();

    /* By leaving this function to return, the caller function (ConverseRunPE)
       will also terminate and return. Since that functions was called by the
       thread constructor (call_startfn of machine-smp.c), the thread will be
       terminated. */
  } else {
    /* processor 0 */
    CmiState cs = CmiGetState();
    MACHSTATE2(2, "waiting for inexit (%d) to be %d",inexit,CmiMyNodeSize()-1);
    while (inexit != CmiMyNodeSize()-1) {
      CmiIdleLock_sleep(&cs->idle,10);
    }
    /* ok, all threads synchronized! */
#endif
    check_lapi(LAPI_Gfence, (lapiContext));

    ConverseCommonExit();
    check_lapi(LAPI_Term, (lapiContext));

#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
    if (CmiMyPe() == 0) CmiPrintf("End of program\n");
#endif

    exit(0);
#if CMK_SMP
  }
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

static CmiIdleState *CmiNotifyGetState(void) {
  CmiIdleState *s=(CmiIdleState *)malloc(sizeof(CmiIdleState));
  s->sleepMs=0;
  s->nIdles=0;
  s->cs=CmiGetState();
  return s;
}

static void CmiNotifyBeginIdle(CmiIdleState *s) {
  s->sleepMs=0;
  s->nIdles=0;
}

#define SPINS_BEFORE_SLEEP     20
    
static void CmiNotifyStillIdle(CmiIdleState *s) {
  MACHSTATE2(2,"[%p] still idle (%d) begin {",CmiGetState(),CmiMyPe());
  s->nIdles++;
  if (s->nIdles>SPINS_BEFORE_SLEEP) { /*Start giving some time back to the OS*/
    s->sleepMs+=2;
    if (s->sleepMs>10) s->sleepMs=10;
  }
  if (s->sleepMs>0) {
    MACHSTATE1(2,"idle sleep (%d) {",CmiMyPe());
    CmiIdleLock_sleep(&s->cs->idle,s->sleepMs);
    MACHSTATE1(2,"} idle sleep (%d)",CmiMyPe());
  }       
  LAPI_Probe(lapiContext);
  MACHSTATE1(2,"still idle (%d) end {",CmiMyPe());
}

static void ConverseRunPE(int everReturn) {
  CmiIdleState *s;
  char** CmiMyArgv;
  CpvInitialize(void *,CmiLocalQueue);

  MACHSTATE2(2, "[%d] ConverseRunPE (thread %p)",CmiMyRank(),CmiGetState());
  /* No communication thread */
  s=CmiNotifyGetState();
  /*CmiState cs;*/
  CmiNodeBarrier();
  /*cs = CmiGetState();*/

  CpvAccess(CmiLocalQueue) = CmiGetState()->localqueue;

  CmiMyArgv=CmiCopyArgs(Cmi_argv);
    
  CthInit(CmiMyArgv);

  ConverseCommonInit(CmiMyArgv);

#if CMK_SMP
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)CmiNotifyBeginIdle,(void *)s);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyStillIdle,(void *)s);
#else
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,(CcdVoidFn)CmiNotifyIdle,NULL);
#endif

#if CMK_IMMEDIATE_MSG
  /* Converse initialization finishes, immediate messages can be processed.
     node barrier previously should take care of the node synchronization */
  _immediateReady = 1;
#endif

  /*if (CmiMyRank() == CmiMyNodeSize()) return;*/
  /* communication thread */
  if (CmiMyRank() == CmiMyNodeSize()) {
    Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
    while (1) sleep(1); /*CommunicationServerThread(5);*/
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

  lapi_info_t info;
  /*int LAPI_Init_rc;*/
  memset(&info,0,sizeof(info));
  check_lapi(LAPI_Init,(&lapiContext, &info));
  /*
  info.err_hndlr=NULL;
  LAPI_Init_rc = LAPI_Init(&lapiContext, &info);
  if (LAPI_Init_rc == LAPI_ERR_BAD_PARAMETER) {
    printf("Error during LAPI_Init.  This normally indicates that \n"
	   " your environment is not properly configured for LAPI--\n"
	   "     MP_MSG_API = %s (should be lapi or mpi,lapi)\n"
	   "     MP_EUILIB = %s (should be us or ip)\n",
	   getenv("MP_MSG_API"),
	   getenv("MP_EULIB"));
  }
  check_lapi(LAPI_Init_rc,);
  */
  
  /* It's a good idea to start with a fence,
     because packets recv'd before a LAPI_Init are just dropped. */
  check_lapi(LAPI_Gfence,(lapiContext));
  
  CsvAccess(lapiInterruptMode) = 0;
  CsvAccess(lapiInterruptMode) = CmiGetArgFlag(argv,"+poll")?0:1;
  CsvAccess(lapiInterruptMode) = CmiGetArgFlag(argv,"+nopoll")?1:0;

  check_lapi(LAPI_Senv,(lapiContext, ERROR_CHK, lapiDebugMode));
  check_lapi(LAPI_Senv,(lapiContext, INTERRUPT_SET, CsvAccess(lapiInterruptMode)));
  
  check_lapi(LAPI_Qenv,(lapiContext, TASK_ID, &CmiMyNode()));
  check_lapi(LAPI_Qenv,(lapiContext, NUM_TASKS, &CmiNumNodes()));

  check_lapi(LAPI_Addr_set,(lapiContext,(void *)PumpMsgsBegin,lapiHeaderHandler));

  /* processor per node */
#if CMK_SMP
  CmiMyNodeSize() = 1;
  CmiGetArgInt(argv,"+ppn", &CmiMyNodeSize());
#else
  if (CmiGetArgFlag(argv,"+ppn")) {
    CmiAbort("+ppn cannot be used in non SMP version!\n");
  }
#endif

  /*
#if CMK_NO_OUTSTANDING_SENDS
  no_outstanding_sends=1;
#endif
  if (CmiGetArgInt(argv,"+no_outstanding_sends",&no_outstanding_sends) && _Cmi_mynode == 0) {
     CmiPrintf("Charm++: Will%s consume outstanding sends in scheduler loop\n",
     	no_outstanding_sends?"":" not");
  }
  */
  CmiNumPes() = CmiNumNodes() * CmiMyNodeSize();
  Cmi_nodestart = CmiMyNode() * CmiMyNodeSize();
  /*Cmi_argvcopy = CmiCopyArgs(argv);*/
  Cmi_argv = argv;
  Cmi_startfn = fn;
  Cmi_usrsched = usched;
 /* CmiSpanTreeInit();*/
  /*
  request_max=MAX_QLEN;
  CmiGetArgInt(argv,"+requestmax",&request_max);
  */
  /*printf("request max=%d\n", request_max);*/
  if (CmiGetArgFlag(argv,"++debug"))
  {   /*Pause so user has a chance to start and attach debugger*/
    printf("CHARMDEBUG> Processor %d has PID %d\n",CmiMyNode(),getpid());
    if (!CmiGetArgFlag(argv,"++debug-no-pause"))
      sleep(10);
  }

  /*CmiTimerInit();*/

#if CMK_NODE_QUEUE_AVAILABLE
  CsvInitialize(CmiNodeState, NodeState);
  CmiNodeStateInit(&CsvAccess(NodeState));
#endif

  procState = (ProcState *)malloc((CmiMyNodeSize()) * sizeof(ProcState));
  for (i=0; i<CmiMyNodeSize(); i++) {
/*    procState[i].sendMsgBuf = PCQueueCreate();   */
    procState[i].recvLock = CmiCreateLock();
  }
  /*
#if CMK_SMP
  sendMsgBuf = PCQueueCreate();
  sendMsgBufLock = CmiCreateLock();
#endif
  */

#if MACHINE_DEBUG_LOG
  {
    char ln[200];
    sprintf(ln,"debugLog.%d",CmiMyNode());
    debugLog=fopen(ln,"w");
  }
#endif
  for (i=0; i<10; ++i) MACHSTATE2(2, "Rankof(%d) = %d",i,CmiRankOf(i));

  MACHSTATE(2, "Starting threads");
  CmiStartThreads(argv);
  ConverseRunPE(initret);
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message) {
  CmiError(message);
  check_lapi(LAPI_Term,(lapiContext));
  exit(1);
}

static void PerrorExit(const char *msg) {
  perror(msg);
  check_lapi(LAPI_Term,(lapiContext));
  exit(1);
}

