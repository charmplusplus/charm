/*
  Myrinet API GM implementation of Converse NET version
  contains only GM API specific code for:
  * CmiNotifyIdle()
  * DeliverViaNetwork()
  * CommunicationServer()

  written by 
  Gengbin Zheng, gzheng@uiuc.edu  4/22/2001
*/


#define MAXPENDINGSEND  50

typedef struct PendingMsgStruct
{
  OutgoingMsg ogm;
  OtherNode   node;
  int size;
  struct PendingMsgStruct *next;
}
*PendingMsg;

static PendingMsg  sendhead = 0, sendtail = 0;
static int pendinglen = 0;

static gm_alarm_t gmalarm;

void enqueue_sending(OutgoingMsg out, OtherNode node, int size)
{
  PendingMsg pm = (PendingMsg) malloc(sizeof(struct PendingMsgStruct));
  pm->ogm = out;
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

/******************************************************************************
 *
 * CmiNotifyIdle()-- wait until a packet comes in
 *
 *****************************************************************************/

void CmiNotifyIdle(void)
{
  struct timeval tv;
#if CMK_SHARED_VARS_UNAVAILABLE
  /*No comm. thread-- listen on sockets for incoming messages*/
#if !CMK_USE_POLL
  static fd_set rfds;
  tv.tv_sec=0; tv.tv_usec=5000;
  FD_ZERO(&rfds); 
  if (Cmi_charmrun_fd!=-1)
    FD_SET(Cmi_charmrun_fd, &rfds);
  select(FD_SETSIZE,&rfds,NULL,0,&tv);
#else
  struct pollfd fds[2]; int n = 0;
  int nreadable;
  gm_recv_event_t *e;
  int pollMs = 5;

  gm_set_alarm (gmport, &gmalarm, (gm_u64_t) pollMs*1000, alarmcallback,
                    (void *)NULL );
  e = gm_blocking_receive_no_spin(gmport);
  /* have to handle this event now */
  CmiCommLock();
  nreadable = processEvent(e);
  CmiCommUnlock();
  if (nreadable) return;
  pollMs = 0;
  if (Cmi_charmrun_fd!=-1) {
    fds[n].fd = Cmi_charmrun_fd;
    fds[n].events = POLLIN;
    n++;
  }
  poll(fds, n, pollMs);
#endif
  if (Cmi_netpoll) CommunicationServer(5);
#else
  /*Comm. thread will listen on sockets-- just sleep*/
  tv.tv_sec=0; tv.tv_usec=1000;
  select(0,NULL,NULL,NULL,&tv);
#endif
}

static void alarmcallback (void *context)
{
}

static void processMessage(char *msg, int len)
{
  char *newmsg;
  int rank, srcpe, seqno, magic, i;
  
  if (len >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno);
    if (magic == (Cmi_charmrun_pid&DGRAM_MAGIC_MASK)) {
      /* node = nodes_by_pe[srcpe]; */
      newmsg = gm_dma_malloc(gmport, len);
      _MEMCHECK(newmsg);
      memcpy(newmsg, msg, len);
      if (rank == DGRAM_BROADCAST) {
        for (i=1; i<Cmi_mynodesize; i++)
          PCQueuePush(CmiGetStateN(i)->recv, CopyMsg(msg, len));
        PCQueuePush(CmiGetStateN(0)->recv, newmsg);
      } else {
#if CMK_NODE_QUEUE_AVAILABLE
         if (rank==DGRAM_NODEMESSAGE) {
           PCQueuePush(CsvAccess(NodeRecv), newmsg);
         }
         else
#endif
           PCQueuePush(CmiGetStateN(rank)->recv, newmsg);
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

    switch (gm_ntohc(e->recv.type))
    {
    case GM_HIGH_RECV_EVENT:
    case GM_RECV_EVENT:
      size = gm_ntohc(e->recv.size);
//CmiPrintf("size:%d\n",size);
      msg = gm_ntohp(e->recv.buffer);
      len = gm_ntohl(e->recv.length);
      if (CmiMsgHeaderGetLength(msg) != len) CmiPrintf("Message corrupted!\n");;
      processMessage(msg, len);
//CmiPrintf("receive tkens:%d\n", gm_num_receive_tokens(gmport));
      gm_provide_receive_buffer(gmport, msg, size, GM_LOW_PRIORITY);
      break;
    case GM_NO_RECV_EVENT:
      return 0;
    default:
      gm_unknown(gmport, e);
    }
    return 1;
}

static void CommunicationServer(int withDelayMs)
{
  gm_recv_event_t *e;
  int size, len;
  char *msg, *buf;

  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);
  if (Cmi_charmrun_fd==-1) return; /*Standalone mode*/
#if CMK_SHARED_VARS_UNAVAILABLE
  if (terrupt)
  {
      return;
  }
  terrupt++;
#endif
  CmiCommLock();
  Cmi_clock = GetClock();
  if (Cmi_clock > Cmi_check_last + Cmi_check_delay) {
    ctrl_sendone_nolock("ping",NULL,0,NULL,0);
    Cmi_check_last = Cmi_clock;
  }

  while (1) {
    CheckSocketsReady(0);
    if (ctrlskt_ready_read) { ctrl_getone(); }
    e = gm_receive(gmport);
    if (!processEvent(e)) break;
  }

  CmiCommUnlock();
#if CMK_SHARED_VARS_UNAVAILABLE
  terrupt--;
#endif

}


void send_callback(struct gm_port *p, void *msg, gm_status_t status)
{
  OutgoingMsg ogm = (OutgoingMsg)msg;

  if (status != GM_SUCCESS) { 
    CmiPrintf("error in send. %d\n", status); 
    CmiAbort("");
  }

  gm_free_send_token (gmport, GM_LOW_PRIORITY);
  ogm->refcount--;
  GarbageCollectMsg(ogm);

  /* since we have one free senf token, start next send */
  send_progress();
}

static void send_progress()
{
  PendingMsg  out;

  while (1)
  {
    out = peek_sending();
    if (out && gm_alloc_send_token(gmport, GM_LOW_PRIORITY)) {
      OutgoingMsg ogm = out->ogm;
      OtherNode node = out->node;
      gm_send_with_callback(gmport, ogm->data, out->size, ogm->size, 
                            GM_LOW_PRIORITY, node->IP, node->dataport, 
                            send_callback, ogm);
      dequeue_sending();
      free(out);
    }
    else break;
  }
}


void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank)
{
  int size = gm_min_size_for_length(ogm->size);

  DgramHeaderMake(ogm->data, rank, ogm->src, Cmi_charmrun_pid, node->send_next);
  ogm->refcount++;

//CmiPrintf("DeliverViaNetwork: size:%d\n", size);

  /* if queue is not empty, this is to gurantee the sequence */
  if (pendinglen != 0) {
    while (pendinglen == MAXPENDINGSEND) {
      /* pending max len exceeded, spin until get a token */
//      CmiPrintf("pending max len exceeded.\n");
        CommunicationServer(0);
    }
    enqueue_sending(ogm, node, size);
    return;
  }
  /* see if we can get a send token from gm */
  if (!gm_alloc_send_token(gmport, GM_LOW_PRIORITY)) {
    /* save to pending send list */
    enqueue_sending(ogm, node, size);
    return;
  }
  gm_send_with_callback(gmport, ogm->data, size, ogm->size, GM_LOW_PRIORITY, node->IP, node->dataport, send_callback, ogm);
}


void initialize_gm()
{
  gm_status_t status;
  int device, i, j, maxsize;
  char portname[200];
  char *buf;

  gmport = NULL;
  if (dataport == -1) return;

  status = gm_init();
  if (status != GM_SUCCESS) { gm_perror("gm_init", status); return; }

  device = 0;
  sprintf(portname, "port%d%d", Cmi_charmrun_pid, Cmi_mynode);
  status = gm_open(&gmport, device, dataport, portname, GM_API_VERSION_1_1);
  if (status != GM_SUCCESS) { return; }

  /* set up recv buffer */
  maxsize = 24;
  for (i=1; i<maxsize; i++) {
    int len = gm_max_length_for_size(i);
    int num = 2;
    if (i<5) num = 0;
    else if (i<11 && i>6)  num = 20;
    else if (i>22) num = 1;
    for (j=0; j<num; j++) {
      buf = gm_dma_malloc(gmport, len);
      _MEMCHECK(buf);
      gm_provide_receive_buffer(gmport, buf, i, GM_LOW_PRIORITY);
    }
  }

  status = gm_set_acceptable_sizes (gmport, GM_LOW_PRIORITY, (1<<(maxsize))-1);

  gm_free_send_tokens (gmport, GM_LOW_PRIORITY,
                       gm_num_send_tokens (gmport));

  gm_initialize_alarm(&gmalarm);
}

