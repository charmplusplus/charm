/*
  Myrinet API GM implementation of Converse NET version
  contains only GM API specific code for:
  * CmiMachineInit()
  * CmiNotifyIdle()
  * DeliverViaNetwork()
  * CommunicationServer()
  * CommunicationServerThread()

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

static gm_alarm_t gmalarm;

/******************************************************************************
 *
 * Send messages pending queue (used internally)
 *
 *****************************************************************************/


/* max length of pending messages */
#define MAXPENDINGSEND  200

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

static PendingMsg  sendhead = NULL, sendtail = NULL;
static int pendinglen = 0;

void enqueue_sending(char *msg, int length, OtherNode node, int size)
{
  PendingMsg pm = (PendingMsg) malloc(sizeof(struct PendingMsgStruct));
  pm->msg = msg;
  pm->length = length;
  pm->mach_id = node->mach_id;
  pm->dataport = node->dataport;
  pm->node_idx = node-nodes;
  pm->size = size;
  pm->next = NULL;
  if (sendhead == NULL) {
    sendhead = sendtail = pm;
  }
  else {
    sendtail->next = pm;
    sendtail = pm;
  }
  pendinglen ++;
}

#define peek_sending() (sendhead)

void dequeue_sending()
{
  if (sendhead == NULL) return;
  sendhead = sendhead->next;
  pendinglen --;
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

#define CMK_MSGPOOL  0

#define MAXMSGLEN  20

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
  MACHSTATE(3,"NonBlocking on receive {")
  e = gm_receive(gmport);
  MACHSTATE(3,"} nonblocking receive returned");
#endif

#if SLEEP_USING_ALARM /*Cancel the alarm*/
  gm_cancel_alarm (&gmalarm);
#endif
  
  /* have to handle this event now */
  CmiCommLock();
  nreadable = processEvent(e);
  CmiCommUnlock();
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


static void CommunicationServer_nolock(int withDelayMs) {
  gm_recv_event_t *e;
  int size, len;
  char *msg, *buf;
  while (1) {
    CheckSocketsReady(0);
    if (ctrlskt_ready_read) { ctrl_getone(); }
    if (CmiStdoutNeedsService()) { CmiStdoutService(); }

    MACHSTATE(3,"Non-blocking receive {")
    e = gm_receive(gmport);
    MACHSTATE(3,"} Non-blocking receive")
    if (!processEvent(e)) break;
  }
}

static void CommunicationServer(int withDelayMs)
{
  MACHSTATE1(2,"CommunicationServer(%d)",withDelayMs)
  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);

  CmiCommLock();

  CommunicationServer_nolock(withDelayMs);

  CmiCommUnlock();
  MACHSTATE(2,"} CommunicationServer")
}

/* similar to CommunicationServer, but it is called by communication thread
   or in interrupt */
static void CommunicationServerThread(int sleepTime)
{
  CommunicationServer(sleepTime);
#if CMK_IMMEDIATE_MSG
  CmiHandleImmediate();
#endif
}

static void processMessage(char *msg, int len)
{
  char *newmsg;
  int rank, srcpe, seqno, magic, i;
  int size;
  
  if (len >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno);
    if (magic == (Cmi_charmrun_pid&DGRAM_MAGIC_MASK)) {
      OtherNode node = nodes_by_pe[srcpe];
      newmsg = node->asm_msg;
      if (newmsg == 0) {
        size = CmiMsgHeaderGetLength(msg);
        newmsg = (char *)CmiAlloc(size);
        if (!newmsg)
          fprintf(stderr, "%d: Out of mem\n", _Cmi_mynode);
        if (size < len) KillEveryoneCode(4559312);
        memcpy(newmsg, msg, len);
        node->asm_rank = rank;
        node->asm_total = size;
        node->asm_fill = len;
        node->asm_msg = newmsg;
      } else {
        size = len - DGRAM_HEADER_SIZE;
        memcpy(newmsg + node->asm_fill, msg+DGRAM_HEADER_SIZE, size);
        node->asm_fill += size;
      }
      if (node->asm_fill > node->asm_total)
         CmiAbort("\n\n\t\tLength mismatch!!\n\n");
      if (node->asm_fill == node->asm_total) {
        if (rank == DGRAM_BROADCAST) {
          for (i=1; i<_Cmi_mynodesize; i++)
            CmiPushPE(i, CopyMsg(newmsg, len));
          CmiPushPE(0, newmsg);
        } else {
#if CMK_NODE_QUEUE_AVAILABLE
           if (rank==DGRAM_NODEMESSAGE) {
             CmiPushNode(newmsg);
           }
           else
#endif
             CmiPushPE(rank, newmsg);
        }
        node->asm_msg = 0;
      }
    } 
    else {
      CmiPrintf("[%d] message ignored1: magic not agree:%d != %d!\n", CmiMyPe(), magic, Cmi_charmrun_pid&DGRAM_MAGIC_MASK);
      CmiPrintf("recv: rank:%d src:%d mag:%d\n", rank, srcpe, magic);
    }
  } 
  else CmiPrintf("message ignored2!\n");
}

static int processEvent(gm_recv_event_t *e)
{
  int size, len;
  char *msg, *buf;
  int status = 1;
  switch (gm_ntohc(e->recv.type))
  {
    case GM_HIGH_RECV_EVENT:
    case GM_RECV_EVENT:
      MACHSTATE(4,"Incoming message")
      size = gm_ntohc(e->recv.size);
      msg = gm_ntohp(e->recv.buffer);
      len = gm_ntohl(e->recv.length);
      processMessage(msg, len);
      gm_provide_receive_buffer(gmport, msg, size, GM_LOW_PRIORITY);
      break;
    case GM_NO_RECV_EVENT:
      return 0;
    case GM_ALARM_EVENT:
      status = 0;
    default:
      MACHSTATE1(3,"Unrecognized GM event %d",evt)
      gm_unknown(gmport, e);
  }
  return status;
}


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

  free(out);
}

void send_callback(struct gm_port *p, void *context, gm_status_t status)
{
  PendingMsg out = (PendingMsg)context;
  void *msg = out->msg;

  if (status != GM_SUCCESS) { 
    int srcpe, seqno, magic;
    char rank;
    char *errmsg;
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno);
    errmsg = getErrorMsg(status);
    CmiPrintf("GM Error> PE:%d send to msg %p node %d rank %d mach_id %d port %d len %d size %d failed to complete (error %d): %s\n", srcpe, msg, out->node_idx, rank, out->mach_id, out->dataport, out->length, out->size, status, errmsg); 
#ifdef __FAULT__ 
    if (status != GM_SEND_DROPPED) {
      gm_drop_sends (gmport, GM_LOW_PRIORITY, out->mach_id, out->dataport,
		                             drop_send_callback, out);
      return;
    }
    else {
      OtherNode node = nodes + out->node_idx;
      if (out->mach_id == node->mach_id && out->dataport == node->dataport) {
        /* it not crashed, resent */
        gm_send_with_callback(gmport, msg, out->size, out->length, 
                            GM_LOW_PRIORITY, out->mach_id, out->dataport, 
                            send_callback, out);
        return;
      }
    }
#else
     CmiAbort("send_callback");
#endif
  }

#if !CMK_MSGPOOL
  gm_dma_free(gmport, msg);
#else
  putPool(msg);
#endif

  gm_free_send_token (gmport, GM_LOW_PRIORITY);
  free(out);

  /* since we have one free send token, start next send */
  send_progress();
}


static void send_progress()
{
  PendingMsg  out;

  while (1)
  {
    out = peek_sending();
    if (out && gm_alloc_send_token(gmport, GM_LOW_PRIORITY)) {
      gm_send_with_callback(gmport, out->msg, out->size, out->length, 
                            GM_LOW_PRIORITY, out->mach_id, out->dataport, 
                            send_callback, out);
      dequeue_sending();
    }
    else break;
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
        (OutgoingMsg ogm, char *ptr, int dlen, OtherNode node, int rank)
{
  char *buf;
  int size, len;
  int alloclen, allocSize;

/* CmiPrintf("DeliverViaNetwork: size:%d\n", size); */

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

  DgramHeaderMake(buf, rank, ogm->src, Cmi_charmrun_pid, node->send_next);
  memcpy(buf+DGRAM_HEADER_SIZE, ptr, dlen);
  size = gm_min_size_for_length(len);

  /* if queue is not empty, enqueue msg. this is to guarantee the order */
  if (pendinglen != 0) {
    while (pendinglen == MAXPENDINGSEND) {
      /* pending max len exceeded, busy wait until get a token 
         Doing this surprisingly improve the performance by 2s for 200MB msg */
      MACHSTATE(4,"Polling until token available")
      CommunicationServer_nolock(0);
    }
    enqueue_sending(buf, len, node, size);
    return;
  }
  enqueue_sending(buf, len, node, size);
  send_progress();
}

void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank)
{
  int size; char *data;
 
  size = ogm->size - DGRAM_HEADER_SIZE;
  data = ogm->data + DGRAM_HEADER_SIZE;
  while (size > Cmi_dgram_max_data) {
    EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank);
    data += Cmi_dgram_max_data;
    size -= Cmi_dgram_max_data;
  }
  if (size) EnqueueOutgoingDgram(ogm, data, size, node, rank);
}

/* simple barrier at machine layer */
/* assuming no other flying messages */
static void send_callback_nothing(struct gm_port *p, void *context, gm_status_t status)
{
}

void CmiBarrier()
{
  int len, size, i;
  int status;
  int count = 0;
  OtherNode node;
  gm_recv_event_t *e;
  char *buf, *msg;
  /* every one send to pe 0 */
  if (CmiMyPe() != 0) {
    len = 32;
    buf = (char *)gm_dma_malloc(gmport, len);
    size = gm_min_size_for_length(len);
    node = nodes;
    gm_send_with_callback(gmport, buf, size, len,
                            GM_LOW_PRIORITY, node->mach_id, node->dataport,
                            send_callback_nothing, NULL);
  }
  /* printf("[%d] HERE\n", CmiMyPe()); */
  if (CmiMyPe() == 0) 
  {
    count = 1;
    while (count != CmiNumPes()) 
    {
      e = gm_receive(gmport);
      switch (gm_ntohc(e->recv.type))
      {
        case GM_HIGH_RECV_EVENT:
        case GM_RECV_EVENT:
          MACHSTATE(4,"Incoming message")
          size = gm_ntohc(e->recv.size);
          msg = gm_ntohp(e->recv.buffer);
          len = gm_ntohl(e->recv.length);
          count ++;
          gm_provide_receive_buffer(gmport, msg, size, GM_LOW_PRIORITY);
          break;
        case GM_NO_RECV_EVENT:
          continue ;
        default:
          MACHSTATE1(3,"Unrecognized GM event %d",evt)
          gm_unknown(gmport, e);
      }
    }
    /* pe 0 broadcast */
    for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
      int p = CmiMyPe();
      p = BROADCAST_SPANNING_FACTOR*p + i;
      if (p > _Cmi_numpes - 1) break;
      len = 32;
      buf = (char *)gm_dma_malloc(gmport, len);
      size = gm_min_size_for_length(len);
      node = nodes + p;
      /* printf("[%d] BD => %d \n", CmiMyPe(), p); */
      gm_send_with_callback(gmport, buf, size, len,
                            GM_LOW_PRIORITY, node->mach_id, node->dataport,
                            send_callback_nothing, NULL);
    }
  }
  /* non 0 pe waiting */
  if (CmiMyPe() != 0) 
  {
   retry:
    e = gm_receive(gmport);
    switch (gm_ntohc(e->recv.type))
    {
      case GM_HIGH_RECV_EVENT:
      case GM_RECV_EVENT:
        size = gm_ntohc(e->recv.size);
        msg = gm_ntohp(e->recv.buffer);
        len = gm_ntohl(e->recv.length);
        gm_provide_receive_buffer(gmport, msg, size, GM_LOW_PRIORITY);
        break;
      case GM_NO_RECV_EVENT:
      case GM_ALARM_EVENT:
        goto retry;
      default:
        gm_unknown(gmport, e);
        goto retry;
    }
    for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
      int p = CmiMyPe();
      p = BROADCAST_SPANNING_FACTOR*p + i;
      if (p > _Cmi_numpes - 1) break;
      p = p%_Cmi_numpes;
      len = 32;
      buf = (char *)gm_dma_malloc(gmport, len);
      size = gm_min_size_for_length(len);
      node = nodes + p;
      /* printf("[%d] RELAY => %d \n", CmiMyPe(), p); */
      gm_send_with_callback(gmport, buf, size, len,
                            GM_LOW_PRIORITY, node->mach_id, node->dataport,
                            send_callback_nothing, NULL);
    }
  }
  /* printf("[%d] OUT of barrier \n", CmiMyPe());  */
}

/***********************************************************************
 * CmiMachineInit()
 *
 * This function intialize the GM board. Set receive buffer
 *
 ***********************************************************************/

void CmiMachineInit(char **argv)
{
  int dataport_max=16; /*number of largest GM port to check*/
  gm_status_t status;
  int device, i, j, maxsize;
  char *buf;
  int mlen;

  gmport = NULL;
  if (dataport == -1) 
  { /* Can't do standalone mode without mucking with broadcast, etc. */
    fprintf(stderr,
    "ERROR: Standalone mode not supported under net-linux gm.\n"
    "You must either run using charmrun or rebuild using just net-linux.\n");
    machine_initiated_shutdown=1;
    exit(1);
  }

  status = gm_init();
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
  gm_node_id_to_global_id(gmport, Cmi_mach_id, &Cmi_mach_id);
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

  for (i=1; i<maxsize; i++) {
    int len = gm_max_length_for_size(i);
    int num = 2;

    maxMsgSize = len;

    if (i<5) num = 0;
    else if (i<9)  num = 4;
    else if (i<17)  num = 20;
    else if (i>22) num = 1;
    for (j=0; j<num; j++) {
      buf = gm_dma_malloc(gmport, len);
      _MEMCHECK(buf);
      gm_provide_receive_buffer(gmport, buf, i, GM_LOW_PRIORITY);
    }
  }
  Cmi_dgram_max_data = maxMsgSize - DGRAM_HEADER_SIZE;

  status = gm_set_acceptable_sizes (gmport, GM_LOW_PRIORITY, (1<<(maxsize))-1);

  gm_free_send_tokens (gmport, GM_LOW_PRIORITY,
                       gm_num_send_tokens (gmport));

#if CMK_MSGPOOL
  msgpool[msgNums++]  = gm_dma_malloc(gmport, maxMsgSize);
#endif

  /* alarm will ping charmrun */
  gm_initialize_alarm(&gmalarm);

}

void CmiGmConvertMachineID(unsigned int *mach_id)
{
#if CMK_USE_GM2 
    gm_status_t status;
    int newid;
    status = gm_global_id_to_node_id(gmport, *mach_id, &newid);
    if (status == GM_SUCCESS) *mach_id = newid;
#endif
}

/* make sure other gm nodes are accessible in routing table */
void CmiCheckGmStatus()
{
  int i;
  int doabort = 0;
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
    errmsg = ""; break;
  }
  return errmsg;
}
