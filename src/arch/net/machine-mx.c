
/** @file
 * Myrinet API GM implementation of Converse NET version
 * @ingroup NET
 * contains only MX API specific code for:
 * - CmiMachineInit()
 * - CmiCommunicationInit()
 * - CmiNotifyIdle()
 * - DeliverViaNetwork()
 * - CommunicationServer()
 * - CmiMachineExit()

  written by 
  Yan Shi, yanshi@uiuc.edu        2/1/2006
  Gengbin Zheng, gzheng@uiuc.edu  2/3/2006
  
  ChangeLog:
  * 2/3/2006:  Gengbin Zheng
    implemented packetization, and fix a bug related to buffer reuse/change
    in pending send 
  * 2/7/2006:  Gengbin Zheng
    implement active message mode using callback
    short message pingpong time improved by 0.5us
  * 2/8/2006:  Gengbin Zheng
    implement buffering of future messages (I haven't seen out-of-order
    messages so far, but it may happen)
    Using POOL relies on charmrun set MX_MONOTHREAD=1.
*/

/**
 * @addtogroup NET
 * @{
 */

/* use unexp callback */
#define MX_ACTIVE_MESSAGE                     1

/*#define CMK_USE_CHECKSUM                      0*/

/* default as in busywaiting mode */
#undef CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT
#undef CMK_WHEN_PROCESSOR_IDLE_USLEEP
#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT  1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP    0


/******************************************************************************
 *
 * Send messages pending queue (used internally)
 *
 *****************************************************************************/

typedef struct PendingSentMsgStruct
{
  OutgoingMsg  ogm;
  char *data;
  struct PendingSentMsgStruct *next;
  mx_request_t handle;
  int flag;			/* used for active message mode */
}
*PendingSentMsg;

#define CMK_PMPOOL  1

#if CMK_PMPOOL
#define MAXPMS 200
static PendingSentMsg pmpool[MAXPMS];
static int pmNums = 0;

#define putPool(pm) 	{	\
  if (pmNums == MAXPMS) free(pm);	\
  else pmpool[pmNums++] = pm; }	

#define getPool(pm)	{	\
  if (pmNums == 0) {pm = (PendingSentMsg)malloc(sizeof(struct PendingSentMsgStruct));}	\
  else { pm = pmpool[--pmNums];	}\
}
#else
#define putPool(pm) { free(pm); }
#define getPool(pm) { pm = (PendingSentMsg)malloc(sizeof(struct PendingSentMsgStruct)); _MEMCHECK(pm);}
#endif

static PendingSentMsg sent_handles=NULL;     /* head of queue  */
static PendingSentMsg sent_handles_end=NULL; /* end of the queue */

#define NewPendingSentMsg(pm, ogm) \
  { getPool(pm);        \
    pm->next=NULL; pm->ogm=ogm; pm->data=data; \
    MACHSTATE1(1,"alloc msg %p",pm);\
  }

#define InsertPendingSentMsg(pm) \
  { if(sent_handles_end==NULL) {sent_handles=pm;}  \
    else {sent_handles_end->next=pm;} \
    sent_handles_end=pm; MACHSTATE(1,"Insert done");}

#define FreePendingSentMsg(pm) \
   { sent_handles=pm->next;	\
     if (sent_handles == NULL) sent_handles_end = NULL; \
     if (pm->ogm) {pm->ogm->refcount--; GarbageCollectMsg(pm->ogm);} \
     else CmiFree(pm->data); \
     putPool(pm); }

CmiUInt8 MATCH_FILTER = 0x11111111FFFFFFFFLL;
CmiUInt8 MATCH_MASK   = 0xffffffffffffffffLL;

static int processMessage(char *msg, int len);
static const char *getErrorMsg(mx_return_t rc);
static void processStatusCode(mx_status_t status);
static void PumpMsgs(int getone);
static void ReleaseSentMsgs(void);
#if MX_ACTIVE_MESSAGE
static void PumpEvents(int getone);
static volatile int gotone = 0;
#endif

/******************************************************************************
 *
 * CmiNotifyIdle()-- wait until a packet comes in
 *
 *****************************************************************************/
typedef struct {
char none;  
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void) { return NULL; }

static void CmiNotifyStillIdle(CmiIdleState *s)
{
  int sleep = 1;
  MACHSTATE(1,"CmiNotifyStillIdle {");
#if 0
  CommunicationServer(0, COMM_SERVER_FROM_WORKER);
#else
#if MX_ACTIVE_MESSAGE
  CmiCommLock();
  PumpEvents(1);
  CmiCommUnlock();
#else
#if CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT
  sleep = 0;
#endif
  CmiCommLock();
  ReleaseSentMsgs();
  PumpMsgs(sleep);			/* busy waiting */
  CmiCommUnlock();
#endif
#endif
  MACHSTATE(1,"} CmiNotifyStillIdle");
}

void CmiNotifyIdle(void) {
  CmiNotifyStillIdle(NULL);
}

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  CmiNotifyStillIdle(s);
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

static void PumpMsgs(int getone) {
  mx_return_t rc;
  mx_status_t status;
  uint32_t result;
  mx_segment_t buffer_desc;
  mx_request_t recv_handle; 

  MACHSTATE1(2,"PumpMsgs(%d) {", getone);
  while (1) {
    if (getone)
      rc = mx_probe(endpoint, 1, MATCH_FILTER, MATCH_MASK, &status, &result);
    else
      rc = mx_iprobe(endpoint, MATCH_FILTER, MATCH_MASK, &status, &result);
    if(rc != MX_SUCCESS) {
      const char *errmsg = getErrorMsg(rc);
      CmiPrintf("mx_iprobe error: %s\n", errmsg);
      CmiAbort("mx_iprobe Abort");
    } 
    if (result == 0) {         /* no incoming */
      break;
    }
    MACHSTATE(2,"PumpMsgs recv one");
    buffer_desc.segment_length = status.msg_length;
    buffer_desc.segment_ptr = (char *) CmiAlloc(status.msg_length);
    MACHSTATE(1,"Non-blocking receive {")
    MACHSTATE1(1," size %d", status.msg_length); 
    rc = mx_irecv(endpoint, &buffer_desc, 1, MATCH_FILTER, MATCH_MASK, NULL, &recv_handle);
    if (rc != MX_SUCCESS) {
      const char *errmsg = getErrorMsg(rc);
      CmiPrintf("mx_irecv error: %s\n", errmsg);
      CmiAbort("Abort");
    }
    MACHSTATE1(1,"} Non-blocking receive return %d", rc);
again:
    rc = mx_wait(endpoint, &recv_handle, MX_INFINITE, &status, &result);
    /*rc = mx_test(endpoint, &recv_handle, &status, &result);*/
    MACHSTATE3(1,"mx_wait return rc=%d result=%d status=%d", rc, result, status.code);
    if (rc != MX_SUCCESS) {
      const char *errmsg = getErrorMsg(rc);
      CmiPrintf("mx_wait error: %s\n", errmsg);
      CmiAbort("Abort");
    }
    if(result==0) {
      CmiPrintf("mx_wait error: TIME OUT\n");
      goto again;
    }
    else {
      processMessage(buffer_desc.segment_ptr, buffer_desc.segment_length);
    }
    if (getone) break;
  }    /* end while */
  MACHSTATE1(2,"} PumpMsgs(%d)", getone);
}

#if MX_ACTIVE_MESSAGE
static void PumpEvents(int getone) {
  mx_return_t rc;
  mx_status_t status;
  uint32_t result;
  mx_segment_t buffer_desc;
  mx_request_t recv_handle; 

  while (1) {
    rc = mx_ipeek(endpoint, &recv_handle, &result);
    if (rc != MX_SUCCESS) {
      const char *errmsg = getErrorMsg(rc);
      CmiPrintf("mx_ipeek error: %s\n", errmsg);
      CmiAbort("Abort");
    }
    if (result == 0) break;
    rc = mx_test(endpoint, &recv_handle, &status, &result);
    /*rc = mx_wait(endpoint, &recv_handle, MX_INFINITE, &status, &result);*/
    if (rc != MX_SUCCESS) {
      const char *errmsg = getErrorMsg(rc);
      CmiPrintf("mx_wait error: %s\n", errmsg);
      CmiAbort("Abort");
    }
    if(result==0) {
      CmiAbort("mx_test or wait: TIME OUT\n");  /* this should never happen */
    }
    else {
      PendingSentMsg pm = (PendingSentMsg)status.context;
      if (pm->flag == 1) {    /* send */
        if (pm->ogm) {pm->ogm->refcount--; GarbageCollectMsg(pm->ogm);}
        else CmiFree(pm->data);
      }
      else if (pm->flag == 2) {                  /* recv */
#if MX_ACTIVE_MESSAGE
        if (status.msg_length == 4)  {
         gotone ++;
         CmiFree(pm->data);
        }
        else
#endif
        processMessage(pm->data, status.msg_length);
      }
      else {
        CmiAbort("Invalid PendingSentMsg!");
      }
      putPool(pm);
    }
    if (getone) break;
  }
}

/* active message model, default */
void recv_callback(void * context, uint64_t match_info, int length)
{
  mx_segment_t buffer_desc;
  mx_request_t recv_handle; 
  mx_return_t rc;
  mx_status_t status;
  uint32_t result;
  PendingSentMsg pm;

  buffer_desc.segment_length = length;
  buffer_desc.segment_ptr = (char *) CmiAlloc(length);
  getPool(pm);
  pm->flag = 2;
  pm->data = buffer_desc.segment_ptr;
  if (MATCH_FILTER != match_info) {
    CmiAbort("Invalid match_info");
  }
  rc = mx_irecv(endpoint, &buffer_desc, 1, MATCH_FILTER, MATCH_MASK, pm, &recv_handle);
  if (rc != MX_SUCCESS) {
    const char *errmsg = getErrorMsg(rc);
    CmiPrintf("mx_irecv error: %s\n", errmsg);
    CmiAbort("Abort");
  }
  if (1) {
    rc = mx_test(endpoint, &recv_handle, &status, &result);
    if (rc != MX_SUCCESS) {
      const char *errmsg = getErrorMsg(rc);
      CmiPrintf("mx_wait error: %s\n", errmsg);
      CmiAbort("Abort");
    }
    if(result==0) {
      return;
    }
    else {
      processStatusCode(status);
      CmiPrintf("PUSH HERE\n");
      processMessage(pm->data, status.msg_length);
      putPool(pm);
    }
  }
}
#endif

#define test_send_complete(handle, status, result) \
            {	\
              mx_return_t rc;	\
              rc = mx_test(endpoint, &(handle), &status, &result);	\
              if (rc != MX_SUCCESS) {	\
                  MACHSTATE1(3," mx_test returns %d", rc);	\
                  CmiAbort("mx_test failed\n");	\
              }	\
            }

static void ReleaseSentMsgs(void) {
    MACHSTATE(2,"ReleaseSentMsgs {");
    mx_return_t rc;
    mx_status_t status;
    unsigned int result;
    PendingSentMsg next, pm = sent_handles;
    while (pm!=NULL) {
      test_send_complete(pm->handle, status, result);
      next = pm->next;
      if(result!=0 && status.code == MX_STATUS_SUCCESS) {
        MACHSTATE1(2,"Sent complete. Free sent msg size %d", status.msg_length);
	FreePendingSentMsg(pm);
      }
      else 
        break;
      pm = next;
    }
    MACHSTATE(2,"} ReleaseSentMsgs");
}
 
static void CommunicationServer_nolock(int withDelayMs) {
  if (endpoint == NULL) return;
  MACHSTATE(2,"CommunicationServer_nolock start {")
#if MX_ACTIVE_MESSAGE
  PumpEvents(0);
#else
  PumpMsgs(0);
  ReleaseSentMsgs();
#endif
  MACHSTATE(2,"}CommunicationServer_nolock end");
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
  if (Cmi_charmrun_pid == 0 && endpoint == NULL) return;

  MACHSTATE2(2,"CommunicationServer(%d) from %d {",withDelayMs, where)

  if (where == COMM_SERVER_FROM_WORKER && machine_initiated_shutdown) {
      /* Converse exit, wait for pingCharm to quit */
    return;
  }

  if (where == COMM_SERVER_FROM_INTERRUPT) {
      /* don't service charmrun if converse exits, this fixed a hang bug */
    if (!machine_initiated_shutdown) ServiceCharmrun_nolock();
    return;
  }
  else if (where == COMM_SERVER_FROM_SMP || where == COMM_SERVER_FROM_WORKER) {
    if (machine_initiated_shutdown) ServiceCharmrun_nolock();  /* to exit */
  }

  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);

  CmiCommLock();
  CommunicationServer_nolock(withDelayMs);
  CmiCommUnlock();

#if CMK_IMMEDIATE_MSG
  if (where == COMM_SERVER_FROM_SMP)
    CmiHandleImmediate();
#endif

  MACHSTATE(2,"} CommunicationServer")
}

void processFutureMessages(OtherNode node)
{
  if (!CdsFifo_Empty(node->futureMsgs)) {
    int len = CdsFifo_Length(node->futureMsgs);
    CmiPrintf("[%d] processFutureMessages %d\n", CmiMyPe(), len);
    int i=0;
    while (i<len) {
      FutureMessage f = (FutureMessage)CdsFifo_Dequeue(node->futureMsgs);
      int status = processMessage(f->msg, f->len);
      free(f);
      i++;
    }
  }
}

static int processMessage(char *msg, int len)
{
  char *newmsg;
  int rank, srcpe, seqno, magic, i;
  unsigned int broot;
  int size;
  unsigned char checksum;

  if (len >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);

    MACHSTATE2(1, "Break header Cmi-charmrun_id=%d, magic=%d", Cmi_charmrun_pid, magic);
    MACHSTATE3(1, "srcpe=%d, seqno=%d, rank=%d", srcpe, seqno, rank);    
 
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
        CmiPrintf("[%d] Warning: Past packet received from PE %d (expecting: %d seqno: %d)\n", CmiMyPe(), srcpe, node->recv_expect, seqno);
	CmiPrintf("\n\n\t\t[%d] packet ignored!\n\n");
	return 0;
      }
      else {
        CmiPrintf("[%d] Error detected - Packet out of order from PE %d (expecting: %d got: %d)\n", CmiMyPe(), srcpe, node->recv_expect, seqno);
/*
        CmiAbort("\n\n\t\tPacket out of order!!\n\n");
*/
        FutureMessage f = (FutureMessage)malloc(sizeof(struct FutureMessageStruct));
        f->msg = msg;
        f->len = len;
        CdsFifo_Enqueue(node->futureMsgs, f);
        return 0;
      }

      newmsg = node->asm_msg;
      if (newmsg == 0) {
        size = CmiMsgHeaderGetLength(msg);
        if (size != len) {
          newmsg = (char *)CmiAlloc(size);
          _MEMCHECK(newmsg);
          if (len > size) {
           CmiPrintf("size: %d, len:%d.\n", size, len);
           CmiAbort("\n\n\t\tLength mismatch!!\n\n");
          }
          memcpy(newmsg, msg, len);
          CmiFree(msg);			/* free original msg */
        }
        else 
          newmsg = msg;
        node->asm_rank = rank;
        node->asm_total = size;
        node->asm_fill = len;
        node->asm_msg = newmsg;
      }
      else {
        size = len - DGRAM_HEADER_SIZE;
        if (node->asm_fill+size > node->asm_total) {
         CmiPrintf("asm_total: %d, asm_fill: %d, len:%d.\n", node->asm_total, node->asm_fill, len);
         CmiAbort("\n\n\t\tLength mismatch!!\n\n");
        }
        memcpy(newmsg + node->asm_fill, msg+DGRAM_HEADER_SIZE, size);
        CmiFree(msg);			/* free original msg */
        node->asm_fill += size;
      }
       	
      /* get a full packet */
      if (node->asm_fill == node->asm_total) {
        switch (rank) {
        case DGRAM_BROADCAST: {
          for (i=1; i<_Cmi_mynodesize; i++)
            CmiPushPE(i, CopyMsg(newmsg, node->asm_total));
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
        }
        node->asm_msg = 0;
          /* do it after integration - the following function may re-entrant */
#if CMK_BROADCAST_SPANNING_TREE
      if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         )
        SendSpanningChildren(NULL, 0, node->asm_total, newmsg, broot, rank);
#elif CMK_BROADCAST_HYPERCUBE
      if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         )
        SendHypercube(NULL, 0, node->asm_total, newmsg, broot, rank);
#endif
      }
      processFutureMessages(node);
    } 
    else {   /* checksum failed */
#ifdef CMK_USE_CHECKSUM
      CmiPrintf("[%d] message ignored: checksum (%d) not 0!\n", CmiMyPe(), checksum);
#else
      CmiPrintf("[%d] message ignored: magic not agree:%d != %d!\n", 
                 CmiMyPe(), magic, Cmi_charmrun_pid&DGRAM_MAGIC_MASK);
#endif
      CmiPrintf("[%d] recved: rank:%d src:%d magic:%d seqno:%d len:%d\n", CmiMyPe(), rank, srcpe, magic, seqno,
len);
    }
  }
  else {
      CmiPrintf("[%d] message ignored: size is too small: %d!\n", CmiMyPe(), len);
      CmiPrintf("[%d] possible size: %d\n", CmiMsgHeaderGetLength(msg));
  }

  return 1;
}

/***********************************************************************
 * DeliverViaNetwork()
 *
 * This function is responsible for all non-local transmission. It
 * first allocate a send token, if fails, put the send message to
 * penging message queue, otherwise invoke the GM send.
 ***********************************************************************/

void EnqueueOutgoingDgram
     (OutgoingMsg ogm, char *ptr, int dlen, OtherNode node, int rank, int broot, int copy)
{
  int size, len, seqno;
  mx_return_t rc;
  mx_request_t sent_handle;
  mx_segment_t buffer_desc;
  uint32_t result;
  char *data;

  len = dlen + DGRAM_HEADER_SIZE;;

  if (copy) {
    data = CopyMsg(ptr-DGRAM_HEADER_SIZE, len);
  }
  else {
    data = ptr-DGRAM_HEADER_SIZE;
    ogm->refcount++; 
  }

  seqno = node->send_next;
  MACHSTATE5(1, "[%d] SEQNO: %d to node %d rank: %d %d", CmiMyPe(), seqno, node-nodes, rank, broot);
  DgramHeaderMake(data, rank, ogm->src, Cmi_charmrun_pid, seqno, broot);
  node->send_next = ((seqno+1)&DGRAM_SEQNO_MASK);
#ifdef CMK_USE_CHECKSUM
  {
  DgramHeader *head = (DgramHeader *)data;
  head->magic ^= computeCheckSum(data, len);
  }
#endif

  MACHSTATE1(2, "EnqueueOutgoingDgram { len=%d", len);
  /* MX will put outgoing message in queue and progress to send */
  /* Note: Assume that MX provides unlimited buffers 
       so no user maintain is required */
  buffer_desc.segment_ptr = data;
  buffer_desc.segment_length = len;
  PendingSentMsg pm;
  if (copy)  ogm = NULL;
  NewPendingSentMsg(pm, ogm);
  pm->flag = 1;
  rc = mx_isend(endpoint, &buffer_desc, 1, node->endpoint_addr, MATCH_FILTER, pm, &(pm->handle));
  if (rc != MX_SUCCESS) {
    MACHSTATE1(3," mx_isend returns %d", rc);
    CmiAbort("mx_isend failed\n");
  }
#if !MX_ACTIVE_MESSAGE
  InsertPendingSentMsg(pm);
#endif
  MACHSTATE(2, "} EnqueueOutgoingDgram");
}

/* can not guarantee that buffer is not altered after return, so it is not
safe */
void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy)
{
  int size; char *data;

  size = ogm->size - DGRAM_HEADER_SIZE;
  data = ogm->data + DGRAM_HEADER_SIZE;

  MACHSTATE3(2, "DeliverViaNetwork { : size:%d, to node mach_id=%d, nic=%ld", ogm->size, node->mach_id, node->nic_id);

  while (size > Cmi_dgram_max_data) {
    copy = 1;     /* since we are packetizing, we need to copy anyway now */
    EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank, broot, copy);
    data += Cmi_dgram_max_data;
    size -= Cmi_dgram_max_data;
  }
  if (size>0) EnqueueOutgoingDgram(ogm, data, size, node, rank, broot, copy);

  MACHSTATE(2, "} DeliverViaNetwork");
}

static void sendBarrierMessage(int pe)
{
  mx_request_t send_handle;
  mx_segment_t buffer_desc;
  mx_return_t rc;
  mx_status_t status;
  uint32_t result;
  char msg[4];

  OtherNode  node = nodes + pe;
  buffer_desc.segment_ptr = msg;
  buffer_desc.segment_length = 4;
  rc = mx_isend(endpoint, &buffer_desc, 1, node->endpoint_addr, MATCH_FILTER, NULL, &send_handle);
  do {
    rc = mx_test(endpoint, &send_handle, &status, &result);
  } while (rc != MX_SUCCESS || result==0);
}

static void recvBarrierMessage()
{
  mx_segment_t buffer_desc;
  char msg[4];
  mx_return_t rc;
  mx_status_t status;
  mx_request_t recv_handle;
  uint32_t result;

#if MX_ACTIVE_MESSAGE
  while (gotone == 0) {
    mx_progress(endpoint);
    PumpEvents(1);
  }
  gotone--;
#else
  do {
  rc = mx_probe(endpoint, 100, MATCH_FILTER, MATCH_MASK, &status, &result);
  } while (result == 0);
  CmiAssert(status.msg_length == 4);

  buffer_desc.segment_length = 4;
  buffer_desc.segment_ptr = msg;
  rc = mx_irecv(endpoint, &buffer_desc, 1, MATCH_FILTER, MATCH_MASK, NULL, &recv_handle);
  do {
  rc = mx_wait(endpoint, &recv_handle, MX_INFINITE, &status, &result);
  } while (rc!=MX_SUCCESS || result == 0);
#endif
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
  MACHSTATE(3,"CmiMachineInit {");
  mx_return_t  rc;
  endpoint = NULL;
  /* standalone mode */
  if (dataport == -1) return; 

  rc = mx_init();
  if (rc != MX_SUCCESS) { 
    MACHSTATE1(3," mx_init returns %d", rc);
    printf("Cannot open MX library (does the machine have a GM card?)\n");
    return; 
  }

  rc = mx_open_endpoint(MX_ANY_NIC, MX_ANY_ENDPOINT, MX_FILTER, 0, 0, &endpoint);
  if (rc != MX_SUCCESS) {
    MACHSTATE1(3," open endpoint address returns %d", rc);
    printf("Cannot open endpoint address\n");
    return;
  }

  /* get endpoint address of local endpoint */
  rc = mx_get_endpoint_addr(endpoint, &endpoint_addr);
  if (rc != MX_SUCCESS) {
    MACHSTATE1(3," get endpoint address returns %d", rc);
    printf("Cannot get endpoint address\n");
    return;
  }

  /* get NIC id and endpoint id */	
  rc = mx_decompose_endpoint_addr(endpoint_addr, &Cmi_nic_id, (uint32_t*)&Cmi_mach_id);
  if (rc != MX_SUCCESS) {
    MACHSTATE1(3," mx_decompose_endpoint returns %d", rc);
    printf("Cannot decompose endpoint address\n");
    return;
  }

  dataport = 1;     /* fake it so that charmrun checking won't fail */

  MATCH_FILTER &= Cmi_charmrun_pid;

  Cmi_dgram_max_data = 1024-DGRAM_HEADER_SIZE;

#if  MX_ACTIVE_MESSAGE
  mx_register_unexp_callback(endpoint, recv_callback, NULL);
#endif

  MACHSTATE(3,"} CmiMachineInit");
}

void CmiMXMakeConnection();

void CmiCommunicationInit(char **argv)
{
  CmiMXMakeConnection();
}

void CmiMachineExit()
{
  MACHSTATE(3, "CmiMachineExit {");
  mx_return_t  rc;
  if (endpoint) {
    rc = mx_close_endpoint(endpoint);
    if(rc!=MX_SUCCESS){
      MACHSTATE1(3, "mx_close_endpoint returns %d", rc);
      printf("Can't do mx_close_endpoint\n");   	
      return;	
    }
    endpoint = NULL;
    rc = mx_finalize();
    if(rc!=MX_SUCCESS){
      MACHSTATE1(3, "mx_finalize returns %d", rc);
      printf("Can't do mx_finalize\n");
      return;
    }
  }
  MACHSTATE(3, "} CmiMachineExit");
}

/* make sure other gm nodes are accessible in routing table */
void CmiMXMakeConnection()
{
  int i;
  int doabort = 0;
  if (Cmi_charmrun_pid == 0 && endpoint == NULL) return;
  if (endpoint == NULL) machine_exit(1);
  MACHSTATE(3,"CmiMXMakeConnection {");
  for (i=0; i<_Cmi_numnodes; i++) {
    mx_return_t  rc;
    char ip_str[128];
    skt_print_ip(ip_str, nodes[i].IP);
    rc = mx_connect(endpoint, nodes[i].nic_id, nodes[i].mach_id, MX_FILTER, MX_INFINITE, &nodes[i].endpoint_addr); 
    if (rc != MX_SUCCESS) {
      CmiPrintf("Error> mx node %d can't contact node %d. \n", CmiMyPe(), i);
      doabort = 1;
    }
  }
  if (doabort) CmiAbort("CmiMXMakeConnection");
  MACHSTATE(3,"}CmiMXMakeConnection");
}

static const char *getErrorMsg(mx_return_t rc)
{
/*
  char *errmsg;
  switch (rc) {
  case MX_SUCCESS:  return "MX_SUCCESS";
  case MX_NO_RESOURCES: return "MX_NO_RESOURCES";
  };
  return "Unknown MX error message";
*/
  return mx_strerror(rc);
}

static void processStatusCode(mx_status_t status){
  const char *str = mx_strstatus(status.code);
  CmiPrintf("processStatusCode: %s\n", str);
  MACHSTATE1(4, "%s", str);
}

/*@}*/
