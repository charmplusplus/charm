/*
  Myrinet API GM implementation of Converse NET version
  contains only GM API specific code for:
  * CmiMachineInit()
  * CmiNotifyIdle()
  * DeliverViaNetwork()
  * CommunicationServer()

  written by 
  Gengbin Zheng, gzheng@uiuc.edu  4/22/2001

  TODO:
  1. DMAable buffer reuse;
*/


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
  OtherNode   node;	/* receiver node */
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
  pm->node = node;
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

static void alarmcallback (void *context);
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
  
} CmiIdleState;

static CmiIdleState *CmiNotifyGetState(void) { return NULL; }

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  
}

static void CmiNotifyStillIdle(CmiIdleState *s)
{
#if CMK_SHARED_VARS_UNAVAILABLE
  /*No comm. thread-- listen on sockets for incoming messages*/
  int nreadable;
  gm_recv_event_t *e;
  int pollMs = 5;

  if (Cmi_idlepoll) {
    if (Cmi_netpoll) CommunicationServer(0);
    return;
  }

/*
  gm_set_alarm (gmport, &gmalarm, (gm_u64_t) pollMs*1000, alarmcallback,
                    (void *)NULL );
*/
  e = gm_blocking_receive_no_spin(gmport);
  /* have to handle this event now */
  CmiCommLock();
  nreadable = processEvent(e);
  CmiCommUnlock();
  if (nreadable) {
    return;
  }
  if (Cmi_netpoll) CommunicationServer(5);
#else
  /*Comm. thread will listen on sockets-- just sleep*/
  CmiIdleLock_sleep(&CmiGetState()->idle,5);
#endif
}

void CmiNotifyIdle(void) {
  CmiNotifyStillIdle(NULL);
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

static void CommunicationServer(int withDelayMs)
{
  gm_recv_event_t *e;
  int size, len;
  char *msg, *buf;

  CmiCommLockOrElse({
    MACHSTATE(3,"Attempted to re-enter comm. server!")
    return;
  });
  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);
/*
#if CMK_SHARED_VARS_UNAVAILABLE
  if (terrupt)
  {
      return;
  }
  terrupt++;
#endif
*/
  CmiCommLock();

  while (1) {
    CheckSocketsReady(0);
/*
    CmiCommLock();
*/
    if (ctrlskt_ready_read) { ctrl_getone(); }
    e = gm_receive(gmport);
    if (!processEvent(e)) break;
/*
    CmiCommUnlock();
*/
  }

/*
#if CMK_SHARED_VARS_UNAVAILABLE
  terrupt--;
#endif
*/

  CmiCommUnlock();
  MACHSTATE(2,"} CommunicationServer")

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
          fprintf(stderr, "%d: Out of mem\n", Cmi_mynode);
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
          for (i=1; i<Cmi_mynodesize; i++)
            CmiPushPE(i, CopyMsg(newmsg, len));
          CmiPushPE(0, newmsg);
        } else {
#if CMK_NODE_QUEUE_AVAILABLE
           if (rank==DGRAM_NODEMESSAGE) {
             PCQueuePush(CsvAccess(NodeRecv), newmsg);
           }
           else
#endif
             CmiPushPE(rank, newmsg);
        }
        node->asm_msg = 0;
      }
    } 
    else {
      CmiPrintf("message ignored1: magic not agree:%d != %d!\n", magic, Cmi_charmrun_pid&DGRAM_MAGIC_MASK);
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
      gm_unknown(gmport, e);
    }
    return status;
}


void send_callback(struct gm_port *p, void *msg, gm_status_t status)
{
  if (status != GM_SUCCESS) { 
    int srcpe, seqno, magic;
    char rank;
    char *errmsg;
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno);
    errmsg = getErrorMsg(status);
    CmiPrintf("GM Error> PE:%d send to %d failed to complete (error %d): %s\n", srcpe, rank, status, errmsg); 
    CmiAbort("");
  }

#if !CMK_MSGPOOL
  gm_dma_free(gmport, msg);
#else
  putPool(msg);
#endif
  gm_free_send_token (gmport, GM_LOW_PRIORITY);

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
      OtherNode node = out->node;
      char *msg = out->msg;
      gm_send_with_callback(gmport, msg, out->size, out->length, 
                            GM_LOW_PRIORITY, *(int *)&node->IP, node->dataport, 
                            send_callback, msg);
      dequeue_sending();
      free(out);
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
      /* pending max len exceeded, busy wait until get a token */
/*      CmiPrintf("pending max len exceeded.\n"); */
        CommunicationServer(0);
    }
    enqueue_sending(buf, len, node, size);
    return;
  }
  /* see if we can get a send token from gm */
  if (!gm_alloc_send_token(gmport, GM_LOW_PRIORITY)) {
    /* save to pending send list */
    enqueue_sending(buf, len, node, size);
    return;
  }
  gm_send_with_callback(gmport, buf, size, len, 
                        GM_LOW_PRIORITY, *(int *)&node->IP, node->dataport, 
                        send_callback, buf);
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
  EnqueueOutgoingDgram(ogm, data, size, node, rank);
}

/***********************************************************************
 * CmiMachineInit()
 *
 * This function intialize the GM board. Set receive buffer
 *
 ***********************************************************************/

void CmiMachineInit()
{
  gm_status_t status;
  int device, i, j, maxsize;
  char portname[200];
  char *buf;
  int mlen;

  gmport = NULL;
  if (dataport == -1) return;

  status = gm_init();
  if (status != GM_SUCCESS) { gm_perror("gm_init", status); return; }

  device = 0;
  sprintf(portname, "port%d%d", Cmi_charmrun_pid, Cmi_mynode);
  status = gm_open(&gmport, device, dataport, portname, GM_API_VERSION_1_1);
  if (status != GM_SUCCESS) { return; }

  /* default abort will take care of gm clean up */
  skt_set_abort(gmExit);

  /* set up recv buffer */
/*
  maxsize = gm_min_size_for_length(4096);
  Cmi_dgram_max_data = 4096 - DGRAM_HEADER_SIZE;
*/
  maxsize = 16;

  for (i=1; i<maxsize; i++) {
    int len = gm_max_length_for_size(i);
    int num = 2;

    maxMsgSize = len;

    if (i<5) num = 0;
    else if (i<11 && i>6)  num = 20;
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


/* make sure other gm nodes are accessible in routing table */
void CmiCheckGmStatus()
{
  int i;
  int doabort = 0;
  if (gmport == NULL) machine_exit(1);
  for (i=0; i<Cmi_numnodes; i++) {
    gm_status_t status;
    char uid[6], str[100];
    unsigned int ip;
    memcpy(&ip, &nodes[i].IP, sizeof(nodes[i].IP));
    status = gm_node_id_to_unique_id(gmport, ip, uid);
    if (status != GM_SUCCESS || ( uid[0]==0 && uid[1]== 0 
         && uid[2]==0 && uid[3]==0 && uid[4]==0 && uid[5]==0)) { 
      CmiPrintf("Error> gm node %d doesn't know node %d. \n", CmiMyPe(), i);
      doabort = 1;
    }
/*    CmiPrintf("%d: ip:%d %d %d %d\n", CmiMyPe(), nodes[i].IP,uid[0], uid[3], uid[5]); */
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
  default:
    errmsg = ""; break;
  }
  return errmsg;
}
