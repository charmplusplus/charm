
/** @file
 * Myrinet API GM implementation of Converse NET version
 * @ingroup NET
 * contains only GM API specific code for:
 * - CmiMachineInit()
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

/* default as in busywaiting mode */
#undef CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT
#undef CMK_WHEN_PROCESSOR_IDLE_USLEEP
#define CMK_WHEN_PROCESSOR_IDLE_BUSYWAIT 1
#define CMK_WHEN_PROCESSOR_IDLE_USLEEP 0


/******************************************************************************
 *
 * Send messages pending queue (used internally)
 *
 *****************************************************************************/

/* max length of pending messages */
#if 0
#define MAXPENDINGSEND  500
//#if 0
typedef struct PendingMsgStruct
{
  void *msg;
  int length;		/* length of message */
  int size;		/* size of message, usually around log2(length)  */
  int mach_id;                /* receiver machine id(GM) */
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
#endif


typedef struct PendingSentMsgStruct
{
  mx_request_t handle;
  OutgoingMsg ogm;
  char *data;
  struct PendingSentMsgStruct *next;
  struct PendingSentMsgStruct *prev;
}
*PendingSentMsg;

static PendingSentMsg sent_handles=NULL;     //where the newly sent goes --> end of queue  
static PendingSentMsg sent_handles_end=NULL; //where the next pointer points to --> start of the queue

#define InsertPendingSentMsg(pm, ogm) \
  {pm = (PendingSentMsg)CmiAlloc(sizeof(struct PendingSentMsgStruct));\
   _MEMCHECK(pm); MACHSTATE1(3,"alloc msg %u",pm);\
   pm->prev=sent_handles_end; pm->next=NULL; \
   if(sent_handles==NULL) {sent_handles=sent_handles_end=pm;}\
   else {sent_handles_end->next=pm;} MACHSTATE(3,"Here2");\
   sent_handles_end=pm; pm->ogm=ogm; pm->data=data;MACHSTATE(3,"Insert done");}\

#define FreePendingSentMsg(pm) \
   { if(pm!=NULL) { if(pm->prev==NULL && pm->next==NULL)\
                      {sent_handles_end=NULL;sent_handles=NULL;}\
                    else if(pm->prev!=NULL && pm->next==NULL)\
			{pm->prev->next=NULL;sent_handles_end=pm->prev;}\
                    else if(pm->next!=NULL && pm->prev==NULL)\
                      {pm->next->prev=NULL;sent_handles=pm->next;}\
		    else {pm->next->prev=pm->prev;pm->prev->next=pm->next;}\
                    if (pm->ogm) {pm->ogm->refcount--; GarbageCollectMsg(pm->ogm);} \
		    else CmiFree(pm->data); \
                    CmiFree(pm); } }	                				

unsigned long MATCH_FILTER = 0x1111111122223333L;
unsigned long MATCH_MASK   = 0xffffffffffffffffL;
#if 0 
void enqueue_sending(char *msg, int length, OtherNode node, int size)
{
  MACHSTATE(5,"enqueue_sending {");
  mx_return_t rc;
  mx_request_t sent_handle;
  mx_segment_t buffer_desc;
  buffer_desc.segment_ptr = msg;
  buffer_desc.segment_length = length;
  PendingSentMsg pm;
  MACHSTATE1(5," mx_isend to endpoint_addr=%d, 1", node->endpoint_addr);
  InsertPendingSentMsg(pm,msg); 
  MACHSTATE1(5," mx_isend to endpoint_addr=%d, 2", node->endpoint_addr);
  rc = mx_isend(endpoint, &buffer_desc, 1, node->endpoint_addr, MATCH_FILTER, NULL, &(pm->handle));  
  MACHSTATE1(5," mx_isend returns %d", rc);
  if (rc != MX_SUCCESS) {
    MACHSTATE1(3," mx_isend returns %d", rc);
    printf("Cannot mx_isend\n");
    return;
  } 
  MACHSTATE(5,"} enqueue_sending");
}
#endif

static void send_progress();
static void alarmInterrupt(int arg);
static void processMessage(char *msg, int len);
static void processMXError(mx_status_t status);

/******************************************************************************
 *
 * DMA message pool
 *
 *****************************************************************************/
#if 0
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

static void CmiNotifyStillIdle(CmiIdleState *s);

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  CmiNotifyStillIdle(s);
}



static void CmiNotifyStillIdle(CmiIdleState *s)
{
  MACHSTATE(1,"CmiNotifyStillIdle {");
  CommunicationServer(0,0);
  MACHSTATE(1,"} CmiNotifyStillIdle");
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

static void PumpMsgs(void) {
  mx_return_t rc;
  mx_status_t status;
  PendingSentMsg pm, ptr=sent_handles;
  unsigned int result;
  mx_segment_t buffer_desc;
  mx_request_t recv_handle; 
  while (1) {
    rc = mx_iprobe(endpoint, MATCH_FILTER, MATCH_MASK, &status, &result);
    if(rc != MX_SUCCESS) {
      MACHSTATE1(3," mx_iprobe returns %d", rc);
      CmiAbort("Cannot mx_iprobe)\n");
      return;
    } 
    if(result) { 
      buffer_desc.segment_length = status.msg_length;
      buffer_desc.segment_ptr = (char *) CmiAlloc(status.msg_length);
      MACHSTATE(1,"Non-blocking receive {")
      MACHSTATE1(1," size %d", status.msg_length); 
      rc = mx_irecv(endpoint, &buffer_desc, 1, MATCH_FILTER, MATCH_MASK, NULL, &recv_handle);
      MACHSTATE1(1,"} Non-blocking receive return %d", rc);
      rc = mx_wait(endpoint, &recv_handle, MX_INFINITE, &status, &result);
      MACHSTATE3(1,"mx_wait return rc=%d result=%d status=%d", rc, result, status.code);
      if(result==0) processMXError(status);
      else processMessage(buffer_desc.segment_ptr, buffer_desc.segment_length);
    }
    else break;  // no incoming message ready
  }
}

static void processMXError(mx_status_t status){
  switch(status.code){
    case MX_STATUS_SUCCESS: MACHSTATE(4,"mx_wait successful"); break;
    case MX_STATUS_PENDING: MACHSTATE(4,"mx_wait pending"); break;
    case MX_STATUS_BUFFERED: MACHSTATE(4,"mx_wait buffered"); break;
    case MX_STATUS_REJECTED: MACHSTATE(4,"mx_wait rejected"); break;
    case MX_STATUS_TIMEOUT: MACHSTATE(4,"mx_wait timeout"); break;
    default: MACHSTATE(4,"mx_wait returns other");
  }
}

static void ReleaseSentMsgs(void) {
    MACHSTATE(2,"ReleaseSentMsgs {");
    mx_return_t rc;
    mx_status_t status;
    unsigned int result;
    PendingSentMsg next, pm = sent_handles;
    while (pm!=NULL) {
      rc = mx_test(endpoint, &(pm->handle), &status, &result);
      if(rc != MX_SUCCESS)
        break;
      next = pm->next;
      if(result!=0 && status.code == MX_STATUS_SUCCESS) {
        MACHSTATE1(2," Sent complete. Free sent msg size %d", status.msg_length);
	FreePendingSentMsg(pm);
      }
      pm = next;
    }
    MACHSTATE(2,"} ReleaseSentMsgs");
}
 
static void CommunicationServer_nolock(int withDelayMs) {
  MACHSTATE(2,"CommunicationServer_nolock start {")
  PumpMsgs();
  ReleaseSentMsgs();
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

  if (where == 2 && machine_initiated_shutdown) {
    return;
  }
  if (where == 1) {
    /* don't service charmrun if converse exits, this fixed a hang bug */
    if (!machine_initiated_shutdown) ServiceCharmrun_nolock();
    return;
  }

  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);

  CmiCommLock();
  CommunicationServer_nolock(withDelayMs);
  CmiCommUnlock();

#if CMK_IMMEDIATE_MSG
  if (where == 0)
  CmiHandleImmediate();
#endif

  MACHSTATE(2,"} CommunicationServer")
}

static void processMessage(char *msg, int len)
{
  int rank, srcpe, seqno, magic, i;
  unsigned int broot;
  int size;
  unsigned char checksum;

  DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);

  DgramHeader *header = (DgramHeader *)(msg); 
  MACHSTATE2(8, "Break header Cmi-charmrun_id=%d, magic=%d", Cmi_charmrun_pid, header->magic);
  MACHSTATE3(8, "srcpe=%d, seqno=%d, rank=%d", srcpe, seqno, rank);    
 
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
	return;
      }
      else {
         CmiPrintf("[%d] Error detected - Packet out of order from PE %d,(expecting: %d got: %d)\n", CmiMyPe(), srcpe, node->recv_expect, seqno);
         CmiAbort("\n\n\t\tPacket out of order!!\n\n");
      }

      size = CmiMsgHeaderGetLength(msg);
      node->asm_rank = rank;
      node->asm_total = size;
      node->asm_fill = len;
      node->asm_msg = msg;
       	
      CmiAssert(size == len);

      /* get a full packet */
      if (node->asm_fill == node->asm_total) {
        switch (rank) {
        case DGRAM_BROADCAST: {
          for (i=1; i<_Cmi_mynodesize; i++)
            CmiPushPE(i, CopyMsg(msg, node->asm_total));
          CmiPushPE(0, msg);
          break;
        }
#if CMK_NODE_QUEUE_AVAILABLE
        case DGRAM_NODEBROADCAST: 
        case DGRAM_NODEMESSAGE: {
          CmiPushNode(msg);
          break;
        }
#endif
        default:
          CmiPushPE(rank, msg);
        }
        node->asm_msg = 0;
      }
      /* do it after integration - the following function may re-entrant */
#if CMK_BROADCAST_SPANNING_TREE
      if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         )
        SendSpanningChildren(NULL, 0, len, msg, broot, rank);
#elif CMK_BROADCAST_HYPERCUBE
      if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         )
        SendHypercube(NULL, 0, len, msg, broot, rank);
#endif
  } 
  else {
#ifdef CMK_USE_CHECKSUM
      CmiPrintf("[%d] message ignored: checksum (%d) not 0!\n", CmiMyPe(), checksum);
#else
      CmiPrintf("[%d] message ignored: magic not agree:%d != %d!\n", 
                 CmiMyPe(), magic, Cmi_charmrun_pid&DGRAM_MAGIC_MASK);
#endif
      CmiPrintf("recved: rank:%d src:%d mag:%d seqno:%d len:%d\n", rank, srcpe, magic, seqno,
len);
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
     (OutgoingMsg ogm, char *ptr, int dlen, OtherNode node, int rank, int broot, int copy)
{
  int size, len, seqno;
  int alloclen, allocSize;
  uint32_t result;
  char *data;

  if (copy) {
    data = CopyMsg(ptr, dlen);
  }
  else {
    data = ptr;
    ogm->refcount++; 
  }

  len = dlen;

  seqno = node->send_next;
//if (CmiMyPe() == 0) CmiPrintf("[%d] SEQNO: %d to %d\n", CmiMyPe(), seqno, node-nodes);
  MACHSTATE5(2, "[%d] SEQNO: %d to %d rank: %d %d", CmiMyPe(), seqno, node-nodes, rank, broot);
  DgramHeaderMake(data, rank, ogm->src, Cmi_charmrun_pid, seqno, broot);
#if 0
  DgramHeader *header = (DgramHeader *)(ptr);
  MACHSTATE2(3,"Make header Cmi_charmrun_pid=%d magic=%d", Cmi_charmrun_pid, header->magic);
  MACHSTATE3(3,"srcpe=%d, seqno=%d, rank=%d", header->srcpe, header->seqno, header->dstrank);
#endif
  node->send_next = ((seqno+1)&DGRAM_SEQNO_MASK);

  MACHSTATE1(2, "EnqueueOutgoingDgram { len=%d", len);
  /* MX will put outgoing message in queue and progress to send */
  /* Note: Assume that MX provides unlimited buffers 
       so no user maintain is required */
  mx_return_t rc;
  mx_request_t sent_handle;
  mx_segment_t buffer_desc;
  buffer_desc.segment_ptr = data;
  buffer_desc.segment_length = len;
  PendingSentMsg pm;
  if (copy)  ogm = NULL;
  InsertPendingSentMsg(pm,ogm);
  rc = mx_isend(endpoint, &buffer_desc, 1, node->endpoint_addr, MATCH_FILTER, NULL, &(pm->handle));
  if (rc != MX_SUCCESS) {
    MACHSTATE1(3," mx_isend returns %d", rc);
    CmiAbort("mx_isend failed\n");
  }
#if 0
  /* wait for it to be safe to change values in workspace */
  do {
    rc = mx_ibuffered(endpoint, &pm->handle, &result);
  } while (rc == MX_SUCCESS && !result);
#endif
  //if(ogm->refcount==0) CmiFree(ogm);  //Garbage collection will do it
  MACHSTATE(2, "} EnqueueOutgoingDgram");
}

/* can not guarantee that buffer is not altered after return, so it is not
safe */
void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy)
{
  int size; char *data;
  size = ogm->size;
  data = ogm->data;
  MACHSTATE3(2, "DeliverViaNetwork { : size:%d, to node mach_id=%d, nic=%ld", size,
node->mach_id, node->nic_id);
  if (size>0) EnqueueOutgoingDgram(ogm, data, size, node, rank, broot, copy);

#if 0
  /* a simple flow control */
  while (pendinglen >= MAXPENDINGSEND) {
      /* pending max len exceeded, busy wait until get a token 
         Doing this surprisingly improve the performance by 2s for 200MB msg */
      MACHSTATE(4,"Polling until token available")
      CommunicationServer_nolock(0);
  }
#endif
  MACHSTATE(2, "} DeliverViaNetwork");
}

#if 0
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
void CmiBarrier()
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
}

/* everyone sends a message to pe 0 and go on */
void CmiBarrierZero()
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
}
#endif

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
  MACHSTATE(3,"} CmiMachineInit");
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

/*@}*/
