/** @file
 * Myrinet API GM implementation of Converse NET version
 * @ingroup NET
 * contains only GM API specific code for:
 * - CmiMachineInit()
 * - CmiCommunicationInit()
 * - CmiNotifyIdle()
 * - DeliverViaNetwork()
 * - CommunicationServer()
 * - CmiMachineExit()

  written by 
  Gengbin Zheng, gzheng@uiuc.edu  4/22/2001
  
  ChangeLog:
  * 3/7/2004,  Gengbin Zheng
    implemented fault tolerant gm layer. When GM detects a catastrophic error,
    it temporarily disables the delivery of all messages with the same sender 
    port, target port, and priority as the message that experienced the error. 
    This layer needs to properly handle the error message of GM and resume 
    the port.

  TODO:
  1. DMAable buffer reuse;
*/

/**
 * @addtogroup NET
 * @{
 */

#ifdef GM_API_VERSION_2_0
#if GM_API_VERSION >= GM_API_VERSION_2_0
#define CMK_USE_GM2    1
#endif
#endif

/* default as in busywaiting mode */
#undef CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT
#undef CMK_WHEN_PROCESSOR_IDLE_USLEEP
#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT 1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP 0

#ifdef __ONESIDED_IMPL
#ifdef __ONESIDED_GM_HARDWARE
#include "conv-onesided.h"
int getSrcHandler;
int getDestHandler;
void handleGetSrc(void *msg);
void handleGetDest(void *msg);
#endif
#endif

static gm_alarm_t gmalarm;

/*#define CMK_USE_CHECKSUM*/

/******************************************************************************
 *
 *  GM layer network statistics collection
 *
 *****************************************************************************/

#define GM_STATS		0

#if GM_STATS
static FILE *gmf;			/* one file per processor */
static int  *gm_stats;			/* send count for each size */
static int   possible_streamed = 0;	/* possible streaming counts */
static int   defrag = 0;		/* number of defragment */
static int   maxQueueLength = 0;	/* maximum send queue length */
#endif

/******************************************************************************
 *
 * Send messages pending queue (used internally)
 *
 *****************************************************************************/


/* max length of pending messages */
#define MAXPENDINGSEND  500

typedef struct PendingMsgStruct
{
  void *msg;
  int length;		/* length of message */
  int size;		/* size of message, usually around log2(length)  */
  int mach_id;		/* receiver machine id */
  int dataport;		/* receiver data port */
  int node_idx;		/* receiver pe id */
  struct PendingMsgStruct *next;
}
*PendingMsg;

static int pendinglen = 0;

/* reuse PendingMsg memory */
static PendingMsg pend_freelist=NULL;

#define FreePendingMsg(d) 	\
  d->next = pend_freelist;\
  pend_freelist = d;\

#define MallocPendingMsg(d) \
  d = pend_freelist;\
  if (d==0) {d = ((PendingMsg)malloc(sizeof(struct PendingMsgStruct)));\
             _MEMCHECK(d);\
  } else pend_freelist = d->next;

void enqueue_sending(char *msg, int length, OtherNode node, int size)
{
  PendingMsg pm;
  MallocPendingMsg(pm);
  pm->msg = msg;
  pm->length = length;
  pm->mach_id = node->mach_id;
  pm->dataport = node->dataport;
  pm->node_idx = node-nodes;
  pm->size = size;
  pm->next = NULL;
  if (node->sendhead == NULL) {
    node->sendhead = node->sendtail = pm;
  }
  else {
    node->sendtail->next = pm;
    node->sendtail = pm;
  }
  pendinglen ++;
#if GM_STATS
  if (pendinglen > maxQueueLength) maxQueueLength = pendinglen;
#endif
}

#define peek_sending(node) (node->sendhead)

#define dequeue_sending(node)  \
  if (node->sendhead != NULL) {	\
    node->sendhead = node->sendhead->next;	\
    pendinglen --;	\
  }

static void alarmcallback (void *context) {
  MACHSTATE(4,"GM Alarm callback executed")
}
static int processEvent(gm_recv_event_t *e);
static void send_progress();
static void alarmInterrupt(int arg);
static int gmExit(int code,const char *msg);
static char *getErrorMsg(gm_status_t status);

/******************************************************************************
 *
 * DMA message pool
 *
 *****************************************************************************/

#define CMK_MSGPOOL  1

#define MAXMSGLEN  200

static char* msgpool[MAXMSGLEN];
static int msgNums = 0;

static int maxMsgSize = 0;

#define putPool(msg) 	{	\
  if (msgNums == MAXMSGLEN) gm_dma_free(gmport, msg);	\
  else msgpool[msgNums++] = msg; }	

#define getPool(msg, len)	{	\
  if (msgNums == 0) msg  = gm_dma_malloc(gmport, maxMsgSize);	\
  else msg = msgpool[--msgNums];	\
}


/******************************************************************************
 *
 * CmiNotifyIdle()-- wait until a packet comes in
 *
 *****************************************************************************/
typedef struct {
char none;  
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void) { return NULL; }

static void CmiNotifyStillIdle(CmiIdleState *s);

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  CmiNotifyStillIdle(s);
}



static void CmiNotifyStillIdle(CmiIdleState *s)
{
#if CMK_SHARED_VARS_UNAVAILABLE
  /*No comm. thread-- listen on sockets for incoming messages*/
  int nreadable;
  gm_recv_event_t *e;
  int pollMs = 4;

#define SLEEP_USING_ALARM 0
#if SLEEP_USING_ALARM /*Enable the alarm, so we don't sleep forever*/
  gm_set_alarm (gmport, &gmalarm, (gm_u64_t) pollMs*1000, alarmcallback,
                    (void *)NULL );
#endif

#if SLEEP_USING_ALARM
  MACHSTATE(3,"Blocking on receive {")
  e = gm_blocking_receive_no_spin(gmport);
  MACHSTATE(3,"} receive returned");
#else
  MACHSTATE(3,"CmiNotifyStillIdle NonBlocking on receive {")
  e = gm_receive(gmport);
  MACHSTATE(3,"} CmiNotifyStillIdle nonblocking receive returned");
#endif

#if SLEEP_USING_ALARM /*Cancel the alarm*/
  gm_cancel_alarm (&gmalarm);
#endif
  
  /* have to handle this event now */
  CmiCommLock();
  inProgress[CmiMyRank()] += 1;
  nreadable = processEvent(e);
  CmiCommUnlock();
  inProgress[CmiMyRank()] -= 1;
  if (nreadable) {
    return;
  }
#else
  /*Comm. thread will listen on sockets-- just sleep*/
  CmiIdleLock_sleep(&CmiGetState()->idle,5);
#endif
}

void CmiNotifyIdle(void) {
  CmiNotifyStillIdle(NULL);
}

/****************************************************************************
 *                                                                          
 * CheckSocketsReady
 *
 * Checks both sockets to see which are readable and which are writeable.
 * We check all these things at the same time since this can be done for
 * free with ``select.'' The result is stored in global variables, since
 * this is essentially global state information and several routines need it.
 *
 ***************************************************************************/

int CheckSocketsReady(int withDelayMs)
{   
  int nreadable;
  CMK_PIPE_DECL(withDelayMs);

  CmiStdoutAdd(CMK_PIPE_SUB);
  if (Cmi_charmrun_fd!=-1) CMK_PIPE_ADDREAD(Cmi_charmrun_fd);

  nreadable=CMK_PIPE_CALL();
  ctrlskt_ready_read = 0;
  dataskt_ready_read = 0;
  dataskt_ready_write = 0;
  
  if (nreadable == 0) {
    MACHSTATE(1,"} CheckSocketsReady (nothing readable)")
    return nreadable;
  }
  if (nreadable==-1) {
    CMK_PIPE_CHECKERR();
    MACHSTATE(2,"} CheckSocketsReady (INTERRUPTED!)")
    return CheckSocketsReady(0);
  }
  CmiStdoutCheck(CMK_PIPE_SUB);
  if (Cmi_charmrun_fd!=-1) 
          ctrlskt_ready_read = CMK_PIPE_CHECKREAD(Cmi_charmrun_fd);
  MACHSTATE(1,"} CheckSocketsReady")
  return nreadable;
}

/***********************************************************************
 * CommunicationServer()
 * 
 * This function does the scheduling of the tasks related to the
 * message sends and receives. 
 * It first check the charmrun port for message, and poll the gm event
 * for send complete and outcoming messages.
 *
 ***********************************************************************/

/* always called from interrupt */
static void ServiceCharmrun_nolock()
{
  int again = 1;
  MACHSTATE(2,"ServiceCharmrun_nolock begin {")
  while (again)
  {
  again = 0;
  CheckSocketsReady(0);
  if (ctrlskt_ready_read) { ctrl_getone(); again=1; }
  if (CmiStdoutNeedsService()) { CmiStdoutService(); }
  }
  MACHSTATE(2,"} ServiceCharmrun_nolock end")
}

static void CommunicationServer_nolock(int withDelayMs) {
  gm_recv_event_t *e;

  MACHSTATE(2,"CommunicationServer_nolock start {")
  while (1) {
    MACHSTATE(3,"Non-blocking receive {")
    e = gm_receive(gmport);
    MACHSTATE(3,"} Non-blocking receive")
    if (!processEvent(e)) break;
  }
  MACHSTATE(2,"}CommunicationServer_nolock end")
}

/*
0: from smp thread
1: from interrupt
2: from worker thread
   Note in netpoll mode, charmrun service is only performed in interrupt, 
 pingCharmrun is from sig alarm, so it is lock free 
*/
static void CommunicationServer(int withDelayMs, int where)
{
  /* standalone mode */
  if (Cmi_charmrun_pid == 0 && gmport == NULL) return;

  MACHSTATE2(2,"CommunicationServer(%d) from %d {",withDelayMs, where)

  if (where == 1) {
    /* don't service charmrun if converse exits, this fixed a hang bug */
    if (!machine_initiated_shutdown) ServiceCharmrun_nolock();
    return;
  }

  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);

  CmiCommLock();
  inProgress[CmiMyRank()] += 1;
  CommunicationServer_nolock(withDelayMs);
  CmiCommUnlock();
  inProgress[CmiMyRank()] -= 1;

#if CMK_IMMEDIATE_MSG
  if (where == 0)
  CmiHandleImmediate();
#endif

  MACHSTATE(2,"} CommunicationServer")
}

static void processMessage(char *msg, int len)
{
  char *newmsg;
  int rank, srcpe, seqno, magic, i;
  unsigned int broot;
  int size;
  unsigned char checksum;
  
  if (len >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);
#ifdef CMK_USE_CHECKSUM
    checksum = computeCheckSum(msg, len);
    if (checksum == 0)
#else
    if (magic == (Cmi_charmrun_pid&DGRAM_MAGIC_MASK))
#endif
    {
      OtherNode node = nodes_by_pe[srcpe];
      /* check seqno */
      if (seqno == node->recv_expect) {
	node->recv_expect = ((seqno+1)&DGRAM_SEQNO_MASK);
      }
      else if (seqno < node->recv_expect) {
        CmiPrintf("[%d] Warning: Past packet received from PE %d, something wrong with GM hardware? (expecting: %d seqno: %d)\n", CmiMyPe(), srcpe, node->recv_expect, seqno);
	CmiPrintf("\n\n\t\t[%d] packet ignored!\n\n");
	return;
      }
      else {
         CmiPrintf("[%d] Error detected - Packet out of order from PE %d, something wrong with GM hardware? (expecting: %d got: %d)\n", CmiMyPe(), srcpe, node->recv_expect, seqno);
         CmiAbort("\n\n\t\tPacket out of order!!\n\n");
      }
      newmsg = node->asm_msg;
      if (newmsg == 0) {
        size = CmiMsgHeaderGetLength(msg);
        if (len > size) {
         CmiPrintf("size: %d, len:%d.\n", size, len);
         CmiAbort("\n\n\t\tLength mismatch!!\n\n");
        }
        newmsg = (char *)CmiAlloc(size);
        _MEMCHECK(newmsg);
        memcpy(newmsg, msg, len);
        node->asm_rank = rank;
        node->asm_total = size;
        node->asm_fill = len;
        node->asm_msg = newmsg;
      } else {
        size = len - DGRAM_HEADER_SIZE;
        if (node->asm_fill+size > node->asm_total) {
         CmiPrintf("asm_total: %d, asm_fill: %d, len:%d.\n", node->asm_total, node->asm_fill, len);
         CmiAbort("\n\n\t\tLength mismatch!!\n\n");
        }
        memcpy(newmsg + node->asm_fill, msg+DGRAM_HEADER_SIZE, size);
        node->asm_fill += size;
      }
      /* get a full packet */
      if (node->asm_fill == node->asm_total) {
        int total_size = node->asm_total;
        node->asm_msg = 0;

      /* do it after integration - the following function may re-entrant */
#if CMK_BROADCAST_SPANNING_TREE
        if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
           )
          SendSpanningChildren(NULL, 0, total_size, newmsg, broot, rank);
#elif CMK_BROADCAST_HYPERCUBE
        if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
           )
          SendHypercube(NULL, 0, total_size, newmsg, broot, rank);
#endif

        switch (rank) {
        case DGRAM_BROADCAST: {
          for (i=1; i<_Cmi_mynodesize; i++)
            CmiPushPE(i, CopyMsg(newmsg, total_size));
          CmiPushPE(0, newmsg);
          break;
        }
#if CMK_NODE_QUEUE_AVAILABLE
        case DGRAM_NODEBROADCAST: 
        case DGRAM_NODEMESSAGE: {
          CmiPushNode(newmsg);
          break;
        }
#endif
        default:
          CmiPushPE(rank, newmsg);
        }    /* end of switch */
      }
    } 
    else {
#ifdef CMK_USE_CHECKSUM
      CmiPrintf("[%d] message ignored: checksum (%d) not 0!\n", CmiMyPe(), checksum);
#else
      CmiPrintf("[%d] message ignored: magic not agree:%d != %d!\n", 
                 CmiMyPe(), magic, Cmi_charmrun_pid&DGRAM_MAGIC_MASK);
#endif
      CmiPrintf("recved: rank:%d src:%d mag:%d seqno:%d len:%d\n", rank, srcpe, magic, seqno, len);
    }
  } 
  else {
      CmiPrintf("[%d] message ignored: size is too small: %d!\n", CmiMyPe(), len);
      CmiPrintf("[%d] possible size: %d\n", CmiMsgHeaderGetLength(msg));
  }
}

/* return 1 - recv'ed  0 - no msg */
static int processEvent(gm_recv_event_t *e)
{
  int size, len;
  char *msg, *buf;
  int status = 1;
  switch (gm_ntohc(e->recv.type))
  {
      /* avoid copy from message to buffer */
    case GM_FAST_PEER_RECV_EVENT:
    case GM_FAST_RECV_EVENT:
      MACHSTATE(4,"Incoming message")
      msg = gm_ntohp(e->recv.message);
      len = gm_ntohl(e->recv.length);
      processMessage(msg, len);
      break;
/*
    case GM_FAST_HIGH_PEER_RECV_EVENT:
    case GM_FAST_HIGH_RECV_EVENT:
*/
    case GM_HIGH_RECV_EVENT:
    case GM_RECV_EVENT:
      MACHSTATE(4,"Incoming message")
      size = gm_ntohc(e->recv.size);
      msg = gm_ntohp(e->recv.buffer);
      len = gm_ntohl(e->recv.length);
      processMessage(msg, len);
      gm_provide_receive_buffer(gmport, msg, size, GM_HIGH_PRIORITY);
      break;
    case GM_NO_RECV_EVENT:
      return 0;
    case GM_ALARM_EVENT:
      status = 0;
    default:
      MACHSTATE1(3,"Unrecognized GM event %d", gm_ntohc(e->recv.type))
      gm_unknown(gmport, e);
  }
  return status;
}


#ifdef __FAULT__ 
void drop_send_callback(struct gm_port *p, void *context, gm_status_t status)
{
  PendingMsg out = (PendingMsg)context;
  void *msg = out->msg;

  printf("[%d] drop_send_callback dropped msg: %p\n", CmiMyPe(), msg);
#if !CMK_MSGPOOL
  gm_dma_free(gmport, msg);
#else
  putPool(msg);
#endif

  FreePendingMsg(out);
}
#else
void send_callback(struct gm_port *p, void *context, gm_status_t status);
void drop_send_callback(struct gm_port *p, void *context, gm_status_t status)
{
  PendingMsg out = (PendingMsg)context;
  void *msg = out->msg;
  gm_send_with_callback(gmport, msg, out->size, out->length,
                        GM_HIGH_PRIORITY, out->mach_id, out->dataport, 
                        send_callback, out);
}
#endif

void send_callback(struct gm_port *p, void *context, gm_status_t status)
{
  PendingMsg out = (PendingMsg)context;
  void *msg = out->msg;
  unsigned char cksum;
  OtherNode  node = nodes+out->node_idx;

  if (status != GM_SUCCESS) { 
    int srcpe, seqno, magic, broot;
    char rank;
    char *errmsg;
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);
    errmsg = getErrorMsg(status);
    CmiPrintf("GM Error> PE:%d send to msg %p node %d rank %d mach_id %d port %d len %d size %d failed to complete (error %d): %s\n", srcpe, msg, out->node_idx, rank, out->mach_id, out->dataport, out->length, out->size, status, errmsg); 
    switch (status) {
#ifdef __FAULT__ 
      case GM_SEND_DROPPED: {
        OtherNode node = nodes + out->node_idx;
        if (out->mach_id == node->mach_id && out->dataport == node->dataport) {
          /* it not crashed, resent */
          gm_send_with_callback(gmport, msg, out->size, out->length, 
                            GM_HIGH_PRIORITY, out->mach_id, out->dataport, 
                            send_callback, out);
          return;
        }
      }
      default: {
        gm_drop_sends (gmport, GM_HIGH_PRIORITY, out->mach_id, out->dataport,
		                             drop_send_callback, out);
        return;
      }
#else
      case GM_SEND_TIMED_OUT: {
 	CmiPrintf("gm send_callback timeout, drop sends and re-enable send. \n");
        gm_drop_sends (gmport, GM_HIGH_PRIORITY, out->mach_id, out->dataport,
                         drop_send_callback, out);
	node->disable=1;	/* disable normal send */
        return;
      }
      case GM_SEND_DROPPED:
        CmiPrintf("Got DROPPED_SEND notification, resend\n");
        gm_send_with_callback(gmport, msg, out->size, out->length,
                            GM_HIGH_PRIORITY, out->mach_id, out->dataport, 
                            send_callback, out);
        return;
      default:
        CmiAbort("gm send_callback failed");
#endif
    }
  }

#ifdef CMK_USE_CHECKSUM
/*
  {
  cksum = computeCheckSum((unsigned char*)msg, out->length);
  if (cksum != 0) {
    CmiPrintf("[%d] Message altered during send, checksum (%d) does not agree!\n", CmiMyPe(), cksum);
    CmiAbort("Myrinet error was detected!\n");
  }
  }
*/
#endif

#if !CMK_MSGPOOL
  gm_dma_free(gmport, msg);
#else
  putPool(msg);
#endif

  gm_free_send_token (gmport, GM_HIGH_PRIORITY);
  FreePendingMsg(out);

  /* send message pending in gm firmware */
  node->gm_pending --;
  if (node->disable && node->gm_pending == 0) node->disable = 0;

  /* since we have one free send token, start next send */
  send_progress();
}

static void send_progress()
{
  static int nextnode = 0;
  int skip;
  OtherNode node;
  PendingMsg  out;

  while (1)
  {
    int sent = 0;
    for (skip=0; skip<_Cmi_numnodes; skip++) {
      node = nodes+nextnode;
      nextnode = (nextnode + 1) % _Cmi_numnodes;
      if (node->disable) continue;
      out = peek_sending(node);
      if (!out) continue;
      if (gm_alloc_send_token(gmport, GM_HIGH_PRIORITY)) {
        if (dataport == out->dataport) {
          gm_send_to_peer_with_callback(gmport, out->msg, out->size, 
                            out->length, GM_HIGH_PRIORITY, out->mach_id,
                            send_callback, out);
        }
        else {
          gm_send_with_callback(gmport, out->msg, out->size, out->length, 
                            GM_HIGH_PRIORITY, out->mach_id, out->dataport, 
                            send_callback, out);
        }
 	node->gm_pending ++;
        sent=1;
        /* dequeue out, but not free it, used at callback */
        dequeue_sending(node);
#if GM_STATS
        gm_stats[out->size] ++;
        /* if we streaming, count how many message we possibly can combine */
        {
        PendingMsg  curout;
        curout = peek_sending(node);
        if (curout) {
          out = curout->next;
          while (out) {
            possible_streamed ++;
            out = out->next;
          }
        }
        }
#endif
      }
      else return;		/* no send token */
    }		/* end for */
    if (sent==0) return;
  }
}


/***********************************************************************
 * DeliverViaNetwork()
 *
 * This function is responsible for all non-local transmission. It
 * first allocate a send token, if fails, put the send message to
 * penging message queue, otherwise invoke the GM send.
 ***********************************************************************/

void EnqueueOutgoingDgram
     (OutgoingMsg ogm, char *ptr, int dlen, OtherNode node, int rank, int broot)
{
  char *buf;
  int size, len, seqno;
  int alloclen, allocSize;

/* CmiPrintf("DeliverViaNetwork: size:%d\n", size); */

  /* don't have to worry about ref count because we do copy, and
     ogm can be free'ed right away */
  /* ogm->refcount++; */

  len = dlen + DGRAM_HEADER_SIZE;

  /* allocate DMAable memory to prepare sending */
  /* FIXME: another memory copy here from user buffer to DMAable buffer */
  /* which however means the user buffer is untouched and can be reused */
#if !CMK_MSGPOOL
  buf = (char *)gm_dma_malloc(gmport, len);
#else
  getPool(buf, len);
#endif
  _MEMCHECK(buf);

  seqno = node->send_next;
  DgramHeaderMake(buf, rank, ogm->src, Cmi_charmrun_pid, seqno, broot);
  node->send_next = ((seqno+1)&DGRAM_SEQNO_MASK);
  memcpy(buf+DGRAM_HEADER_SIZE, ptr, dlen);
#ifdef CMK_USE_CHECKSUM
  {
  DgramHeader *head = (DgramHeader *)buf;
  head->magic ^= computeCheckSum(buf, len);
  }
#endif
  size = gm_min_size_for_length(len);

  /* if queue is not empty, enqueue msg. this is to guarantee the order */
  if (pendinglen != 0) {
    /* this potential screw up broadcast, because bcast packets can not be
       interrupted by other sends in CommunicationServer_nolock */
    enqueue_sending(buf, len, node, size);
    return;
  }
  enqueue_sending(buf, len, node, size);
  send_progress();
}

/* copy is ignored, since we always copy */
void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy)
{
  int size; char *data;
 
  size = ogm->size - DGRAM_HEADER_SIZE;
  data = ogm->data + DGRAM_HEADER_SIZE;
#if GM_STATS
  if (size > Cmi_dgram_max_data) defrag ++;
#endif
  while (size > Cmi_dgram_max_data) {
    EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank, broot);
    data += Cmi_dgram_max_data;
    size -= Cmi_dgram_max_data;
  }
  if (size>0) EnqueueOutgoingDgram(ogm, data, size, node, rank, broot);

  /* a simple flow control */
  while (pendinglen >= MAXPENDINGSEND) {
      /* pending max len exceeded, busy wait until get a token 
         Doing this surprisingly improve the performance by 2s for 200MB msg */
      MACHSTATE(4,"Polling until token available")
      CommunicationServer_nolock(0);
  }
}

/* simple barrier at machine layer */
/* assuming no other flying messages */
static void send_callback_nothing(struct gm_port *p, void *context, gm_status_t status)
{
  gm_dma_free(gmport, context);
}

static void sendBarrierMessage(int pe)
{
  int len = 32;
  char *buf = (char *)gm_dma_malloc(gmport, len);
  int size = gm_min_size_for_length(len);
  OtherNode  node = nodes + pe;
  CmiAssert(buf);
  gm_send_with_callback(gmport, buf, size, len,
              GM_HIGH_PRIORITY, node->mach_id, node->dataport,
              send_callback_nothing, buf);
}

static void recvBarrierMessage()
{
  gm_recv_event_t *e;
  int size, len;
  char *msg;
  while (1) {
    e = gm_receive(gmport);
    switch (gm_ntohc(e->recv.type))
    {
      case GM_HIGH_RECV_EVENT:
      case GM_RECV_EVENT:
        MACHSTATE(4,"Incoming message")
        size = gm_ntohc(e->recv.size);
        msg = gm_ntohp(e->recv.buffer);
        len = gm_ntohl(e->recv.length);
        gm_provide_receive_buffer(gmport, msg, size, GM_HIGH_PRIORITY);
        return;
      case GM_NO_RECV_EVENT:
        continue ;
      default:
        MACHSTATE1(3,"Unrecognized GM event %d", gm_ntohc(e->recv.type))
        gm_unknown(gmport, e);
    }
  }
}

/* happen at node level */
int CmiBarrier()
{
  int len, size, i;
  int status;
  int count = 0;
  OtherNode node;
  int numnodes = CmiNumNodes();
  if (CmiMyRank() == 0) {
    /* every one send to pe 0 */
    if (CmiMyNode() != 0) {
      sendBarrierMessage(0);
    }
    /* printf("[%d] HERE\n", CmiMyPe()); */
    if (CmiMyNode() == 0) 
    {
      for (count = 1; count < numnodes; count ++) 
      {
        recvBarrierMessage();
      }
      /* pe 0 broadcast */
      for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int p = i;
        if (p > numnodes - 1) break;
        /* printf("[%d] BD => %d \n", CmiMyPe(), p); */
        sendBarrierMessage(p);
      }
    }
    /* non 0 node waiting */
    if (CmiMyNode() != 0) 
    {
      recvBarrierMessage();
      for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int p = CmiMyNode();
        p = BROADCAST_SPANNING_FACTOR*p + i;
        if (p > numnodes - 1) break;
        p = p%numnodes;
        /* printf("[%d] RELAY => %d \n", CmiMyPe(), p); */
        sendBarrierMessage(p);
      }
    }
  }
  CmiNodeAllBarrier();
  /* printf("[%d] OUT of barrier \n", CmiMyPe()); */
  return 0;
}

/* everyone sends a message to pe 0 and go on */
int CmiBarrierZero()
{
  int i;

  if (CmiMyRank() == 0) {
    if (CmiMyNode()) {
      sendBarrierMessage(0);
    }
    else {
      for (i=0; i<CmiNumNodes()-1; i++)
      {
        recvBarrierMessage();
      }
    }
  }
  CmiNodeAllBarrier();
  return 0;
}

/***********************************************************************
 * CmiMachineInit()
 *
 * This function intialize the GM board. Set receive buffer
 *
 ***********************************************************************/

static int maxsize;

void CmiMachineInit(char **argv)
{
  int dataport_max=16; /*number of largest GM port to check*/
  gm_status_t status;
  int device, i, j;
  int retry = 10;
  char *buf;
  int mlen;

  MACHSTATE(3,"CmiMachineInit {");

  gmport = NULL;
  /* standalone mode */
  if (dataport == -1) return; 

  /* try a few times init gm */
  for (i=0; i<retry; i++) {
    status = gm_init();
    if (status == GM_SUCCESS) break;
    sleep(1);
  }
  if (status != GM_SUCCESS) { 
    printf("Cannot open GM library (does the machine have a GM card?)\n");
    gm_perror("gm_init", status); 
    return; 
  }
  
  device = 0;
  for (dataport=2;dataport<dataport_max;dataport++) {
    char portname[200];
    sprintf(portname, "converse_port%d_%d", Cmi_charmrun_pid, _Cmi_mynode);
#if CMK_USE_GM2
    status = gm_open(&gmport, device, dataport, portname, GM_API_VERSION_2_0);
#else
    status = gm_open(&gmport, device, dataport, portname, GM_API_VERSION_1_1);
#endif
    if (status == GM_SUCCESS) { break; }
  }
  if (dataport==dataport_max) 
  { /* Couldn't open any GM port... */
    dataport=0;
    return;
  }
  
  /* get our node id */
  status = gm_get_node_id(gmport, (unsigned int *)&Cmi_mach_id);
  if (status != GM_SUCCESS) { gm_perror("gm_get_node_id", status); return; }
#if CMK_USE_GM2
  gm_node_id_to_global_id(gmport, (unsigned int)Cmi_mach_id, (unsigned int*)&Cmi_mach_id);
#endif
  
  /* default abort will take care of gm clean up */
  skt_set_abort(gmExit);

  /* set up recv buffer */
/*
  maxsize = gm_min_size_for_length(4096);
  Cmi_dgram_max_data = 4096 - DGRAM_HEADER_SIZE;
*/
  maxsize = 16;
  CmiGetArgIntDesc(argv,"+gm_maxsize",&maxsize,"maximum packet size in rank (2^maxsize)");

#if GM_STATS
  gm_stats = (int*)malloc((maxsize+1) * sizeof(int));
  for (i=0; i<=maxsize; i++) gm_stats[i] = 0;
#endif

  for (i=1; i<=maxsize; i++) {
    int len = gm_max_length_for_size(i);
    int num = 2;

    maxMsgSize = len;

    if (i<5) num = 0;
    else if (i<7)  num = 4;
    else if (i<13)  num = 20;
    else if (i<17)  num = 10;
    else if (i>22) num = 1;
    for (j=0; j<num; j++) {
      buf = gm_dma_malloc(gmport, len);
      _MEMCHECK(buf);
      gm_provide_receive_buffer(gmport, buf, i, GM_HIGH_PRIORITY);
    }
  }
  Cmi_dgram_max_data = maxMsgSize - DGRAM_HEADER_SIZE;

  status = gm_set_acceptable_sizes (gmport, GM_HIGH_PRIORITY, (1<<(maxsize+1))-1);

  gm_free_send_tokens (gmport, GM_HIGH_PRIORITY,
                       gm_num_send_tokens (gmport));

#if CMK_MSGPOOL
  msgpool[msgNums++]  = gm_dma_malloc(gmport, maxMsgSize);
#endif

  /* alarm will ping charmrun */
  gm_initialize_alarm(&gmalarm);

  MACHSTATE(3,"} CmiMachineInit");
}

void CmiCommunicationInit(char **argv)
{
}

void CmiMachineExit()
{
#if GM_STATS
  int i;
  int mype;
  char fname[128];
  sprintf(fname, "gm-stats.%d", CmiMyPe());
  gmf = fopen(fname, "w");
  mype = CmiMyPe();
  for (i=5; i<=maxsize; i++)  {
    fprintf(gmf, "[%d] size:%d count:%d\n", mype, i, gm_stats[i]);
  }
  fprintf(gmf, "[%d] max quelen: %d possible streaming: %d  defrag: %d \n", mype, maxQueueLength, possible_streamed, defrag);
  fclose(gmf);
#endif
}

void CmiGmConvertMachineID(unsigned int *mach_id)
{
#if CMK_USE_GM2 
    gm_status_t status;
    int newid;
    /* skip if running without charmrun */
    if (Cmi_charmrun_pid == 0 && gmport == NULL) return;
    status = gm_global_id_to_node_id(gmport, *mach_id, &newid);
    if (status == GM_SUCCESS) *mach_id = newid;
#endif
}

/* make sure other gm nodes are accessible in routing table */
void CmiCheckGmStatus()
{
  int i;
  int doabort = 0;
  if (Cmi_charmrun_pid == 0 && gmport == NULL) return;
  if (gmport == NULL) machine_exit(1);
  for (i=0; i<_Cmi_numnodes; i++) {
    gm_status_t status;
    char uid[6], str[100];
    unsigned int mach_id=nodes[i].mach_id;
    status = gm_node_id_to_unique_id(gmport, mach_id, uid);
    if (status != GM_SUCCESS || ( uid[0]==0 && uid[1]== 0 
         && uid[2]==0 && uid[3]==0 && uid[4]==0 && uid[5]==0)) { 
      CmiPrintf("Error> gm node %d can't contact node %d. \n", CmiMyPe(), i);
      doabort = 1;
    }
    /*CmiPrintf("[%d]: %d mach:%d ip:%d %d %d %d\n", CmiMyPe(), i, mach_id, nodes[i].IP,uid[0], uid[3], uid[5]);*/
  }
  if (doabort) CmiAbort("");
}

static int gmExit(int code,const char *msg)
{
  fprintf(stderr,"Fatal socket error: code %d-- %s\n",code,msg);
  machine_exit(code);
}


static char *getErrorMsg(gm_status_t status)
{
  char *errmsg;
  switch (status) {
  case GM_SEND_TIMED_OUT:
    errmsg = "send time out"; break;
  case GM_SEND_REJECTED:
    errmsg = "send rejected"; break;
  case GM_SEND_TARGET_NODE_UNREACHABLE:
    errmsg = "target node unreachable"; break;
  case GM_SEND_TARGET_PORT_CLOSED:
    errmsg = "target port closed"; break;
  case GM_SEND_DROPPED:
    errmsg = "send dropped"; break;
  default:
    errmsg = "unknown error"; break;
  }
  return errmsg;
}

#ifdef __ONESIDED_IMPL
#ifdef __ONESIDED_GM_HARDWARE

struct CmiCb {
  CmiRdmaCallbackFn fn;
  void *param;
};
typedef struct CmiCb CmiCb;

struct CmiRMA {
  int type;
  union {
    int completed;
    CmiCb *cb;
  } ready;
};
typedef struct CmiRMA CmiRMA;

struct CmiRMAMsg {
  char core[CmiMsgHeaderSizeBytes];
  CmiRMA* stat;
};
typedef struct CmiRMAMsg CmiRMAMsg;

struct RMAPutMsg {
  char core[CmiMsgHeaderSizeBytes];
  void *Saddr;
  void *Taddr;
  unsigned int size;
  unsigned int targetId;
  unsigned int sourceId;
  unsigned int dataport;
  CmiRMA *stat;
};
typedef struct RMAPutMsg RMAPutMsg;

void *CmiDMAAlloc(int size) {
  void *addr = gm_dma_calloc(gmport, 1, gm_max_length_for_size(size));
  //gm_allow_remote_memory_access(gmport);
  return addr;
}

int CmiRegisterMemory(void *addr, unsigned int size) {
  gm_status_t status;
  status = gm_register_memory(gmport, addr, size);
  if (status != GM_SUCCESS) { 
    CmiPrintf("Cannot register memory at %p of size %d\n",addr,size);
    gm_perror("registerMemory", status); 
    return 0;
  }
  //gm_allow_remote_memory_access(gmport);
  return 1;
}

int CmiUnRegisterMemory(void *addr, unsigned int size) {
  gm_status_t status;
  status = gm_deregister_memory(gmport, addr, size);
  if (status != GM_SUCCESS) { 
    CmiPrintf("Cannot unregister memory at %p of size %d\n",addr,size);
    gm_perror("UnregisterMemory", status); 
    return 0;
  }
  return 1;
}

void put_callback(struct gm_port *p, void *context, gm_status_t status)
{
  RMAPutMsg *out = (RMAPutMsg*)context;
  unsigned int destMachId = (nodes_by_pe[out->targetId])->mach_id;
  if (status != GM_SUCCESS) { 
    switch (status) {
      case GM_SEND_DROPPED:
        CmiPrintf("Got DROPPED_PUT notification, resend\n");
	gm_directed_send_with_callback(gmport,out->Saddr,(gm_remote_ptr_t)(out->Taddr),
				       out->size,GM_HIGH_PRIORITY, 
				       destMachId, out->dataport, 
				       put_callback, out);
        return;
      default:
        CmiAbort("gm send_callback failed");
    }
  }
  if(out->stat->type==1) {
    out->stat->ready.completed = 1;
    //the handle is active, and the user will clean it
  }
  else {
    (*(out->stat->ready.cb->fn))(out->stat->ready.cb->param);
    //I do not undestand, on turing the following two frees become double frees??
    CmiFree(out->stat->ready.cb);
    CmiFree(out->stat); //clean up the internal handle
  }
  gm_free_send_token (gmport, GM_HIGH_PRIORITY);
  return;
}

void *CmiPut(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size) {
  unsigned int dataport = (nodes_by_pe[targetId])->dataport;
  unsigned int destMachId = (nodes_by_pe[targetId])->mach_id;
  RMAPutMsg *context = (RMAPutMsg*)CmiAlloc(sizeof(RMAPutMsg));
  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->dataport = dataport;
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 1;
  context->stat->ready.completed = 0;
  void *stat = (void*)(context->stat); 
  //get a token before the put
  if (gm_alloc_send_token(gmport, GM_HIGH_PRIORITY)) {
    //perform the put
    gm_directed_send_with_callback(gmport,Saddr,(gm_remote_ptr_t)Taddr,size,
				   GM_HIGH_PRIORITY,destMachId,dataport,
				   put_callback,(void*)context);
  }
  return stat;
}

void CmiPutCb(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size, CmiRdmaCallbackFn fn, void *param) {
  unsigned int dataport = (nodes_by_pe[targetId])->dataport;
  unsigned int destMachId = (nodes_by_pe[targetId])->mach_id;
  RMAPutMsg *context = (RMAPutMsg*)CmiAlloc(sizeof(RMAPutMsg));

  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->dataport = dataport;
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 0;
  context->stat->ready.cb = (CmiCb*)CmiAlloc(sizeof(CmiCb));
  context->stat->ready.cb->fn = fn;
  context->stat->ready.cb->param = param;
  //get a token before the put
  if (gm_alloc_send_token(gmport, GM_HIGH_PRIORITY)) {
    //perform the put
    gm_directed_send_with_callback(gmport,Saddr,(gm_remote_ptr_t)Taddr,size,
				   GM_HIGH_PRIORITY,destMachId,dataport,
				   put_callback,(void*)context);
  }
  return;
}

void handleGetSrc(void *msg) {
  CmiRMA* stat = ((CmiRMAMsg*)msg)->stat;
  if(stat->type==1) {
    stat->ready.completed = 1;
    //the handle is active, and the user will clean it
  }
  else {
    (*(stat->ready.cb->fn))(stat->ready.cb->param);
    //I do not undestand, on turing the following two frees become double frees??
    CmiFree(stat->ready.cb);
    CmiFree(stat); //clean up the internal handle
  }
  CmiFree(msg);
  return;
}

void get_callback_dest(struct gm_port *p, void *context, gm_status_t status)
{
  RMAPutMsg *out = (RMAPutMsg*)context;
  int sizeRmaStat = sizeof(CmiRMAMsg);
  char *msgRmaStat;
  unsigned int srcMachId = (nodes_by_pe[out->targetId])->mach_id;
  if (status != GM_SUCCESS) { 
    switch (status) {
      case GM_SEND_DROPPED:
        CmiPrintf("Got DROPPED_PUT notification, resend\n");
	gm_directed_send_with_callback(gmport,out->Saddr,(int)(out->Taddr),
				       out->size,GM_HIGH_PRIORITY, 
				       out->targetId, out->dataport, 
				       get_callback_dest, out);
        return;
      default:
        CmiAbort("gm send_callback failed");
    }
  }
  gm_free_send_token (gmport, GM_HIGH_PRIORITY);
  //send a message to update the context status on the source node
  msgRmaStat = (void*)CmiAlloc(sizeRmaStat);
  ((CmiRMAMsg*)msgRmaStat)->stat = out->stat;
  CmiSetHandler(msgRmaStat,getSrcHandler);
  CmiSyncSendAndFree(out->targetId,sizeRmaStat,msgRmaStat);
  return;
}

void handleGetDest(void *msg) {
  RMAPutMsg *context1 = (RMAPutMsg*)msg;
  RMAPutMsg *context = (RMAPutMsg*)CmiAlloc(sizeof(RMAPutMsg));
  unsigned int srcMachId = (nodes_by_pe[context1->sourceId])->mach_id;
  context->Saddr = context1->Taddr;
  context->Taddr = context1->Saddr;
  context->size = context1->size;
  context->targetId = context1->sourceId;
  context->sourceId = context1->targetId;
  context->dataport = context1->dataport;
  context->stat = context1->stat;
  //get a token before the put
  if (gm_alloc_send_token(gmport, GM_HIGH_PRIORITY)) {
    //perform the put
    gm_directed_send_with_callback(gmport,context->Saddr,(gm_remote_ptr_t)context->Taddr,context->size,
				   GM_HIGH_PRIORITY,srcMachId,context->dataport,
				   get_callback_dest,(void*)context);
  }
  CmiFree(msg);
  return;
}
/* Send a converse message to the destination to perform a get from destination.
 * The destination then sends a put to write the actual message
 * Then it sends a converse message to the sender to update the completed flag
 */
void *CmiGet(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size) {
  int dataport = (nodes_by_pe[targetId])->dataport;
  int sizeRma;
  char *msgRma;
  RMAPutMsg *context;
  sizeRma = sizeof(RMAPutMsg);
  msgRma = (void*)CmiAlloc(sizeRma);

  context = (RMAPutMsg*)msgRma;
  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->dataport = dataport;
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 1;
  context->stat->ready.completed = 0;
  void *stat = (void*)(context->stat);

  CmiSetHandler(msgRma,getDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);

  return stat;
}

void CmiGetCb(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size, CmiRdmaCallbackFn fn, void *param) {
  int dataport = (nodes_by_pe[targetId])->dataport;
  int sizeRma;
  char *msgRma;
  RMAPutMsg *context;
  sizeRma = sizeof(RMAPutMsg);
  msgRma = (void*)CmiAlloc(sizeRma);

  context = (RMAPutMsg*)msgRma;
  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->dataport = dataport;
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 0;
  context->stat->ready.cb = (CmiCb*)CmiAlloc(sizeof(CmiCb));
  context->stat->ready.cb->fn = fn;
  context->stat->ready.cb->param = param;

  CmiSetHandler(msgRma,getDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);
  return;
}


int CmiWaitTest(void *obj){
  CmiRMA *stat = (CmiRMA*)obj;
  return stat->ready.completed;
}

#endif
#endif

/*@}*/
