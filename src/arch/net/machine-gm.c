#include <assert.h>

static void process_message(char *msg, int len)
{
  char *newmsg;
  int rank, srcpe, seqno, magic, i;
  
  if (len >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(msg, rank, srcpe, magic, seqno);
    if (magic == (Cmi_charmrun_pid&DGRAM_MAGIC_MASK)) {
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
CmiPrintf("recv: rank:%d src:%d mag:%d\n", rank, srcpe, magic);
      CmiPrintf("message ignored1: magic not agree:%d!\n", Cmi_charmrun_pid&DGRAM_MAGIC_MASK);
    }
  } 
  else CmiPrintf("message ignored2!\n");
}

static void CommunicationServer(int withDelayMs)
{
  gm_recv_event_t *e;
  int exitok = 0;
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

  while (!exitok) {
    CheckSocketsReady(0);
    if (ctrlskt_ready_read) { ctrl_getone(); }
    e = gm_receive(gmport);
    switch (gm_ntohc(e->recv.type))
    {
    case GM_HIGH_RECV_EVENT:
    case GM_RECV_EVENT:
      size = gm_ntohc(e->recv.size);
      assert(size < 22);
//CmiPrintf("size:%d\n",size);
      msg = gm_ntohp(e->recv.buffer);
      len = gm_ntohl(e->recv.length);
      if (CmiMsgHeaderGetLength(msg) != len) CmiPrintf("Message corrupted!\n");;
      process_message(msg, len);
//CmiPrintf("receive tkens:%d\n", gm_num_receive_tokens(gmport));
      gm_provide_receive_buffer(gmport, msg, size, GM_LOW_PRIORITY);
      break;
    case GM_NO_RECV_EVENT:
      exitok = 1;
      break;
    default:
      gm_unknown(gmport, e);
    }
  }

  CmiCommUnlock();
#if CMK_SHARED_VARS_UNAVAILABLE
  terrupt--;
#endif

}

void send_callback(struct gm_port *p, void *msg, gm_status_t status)
{
  OutgoingMsg ogm = (OutgoingMsg)msg;
//CmiPrintf("send_callback called\n");

  if (status != GM_SUCCESS) { 
    CmiPrintf("error in send. %d\n", status); 
  }

  gm_free_send_token (gmport, GM_LOW_PRIORITY);
  ogm->refcount--;
  GarbageCollectMsg(ogm);
}


void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank)
{
  int magic;

  int size = gm_min_size_for_length(ogm->size);
  if (gmport == 0) return;

  DgramHeaderMake(ogm->data, rank, ogm->src, Cmi_charmrun_pid, node->send_next);
  ogm->refcount++;
  while (!gm_alloc_send_token(gmport, GM_LOW_PRIORITY)) {
//    CmiPrintf("gm_alloc_send_token failed.\n");
    CommunicationServer(0);
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
    if (i<6) num = 0;
    else if (i<11 && i>6) num = 18;
    else if (i>22) num = 1;
    for (j=0; j<num; j++) {
      buf = gm_dma_malloc(gmport, len);
      assert(buf);
      gm_provide_receive_buffer(gmport, buf, i, GM_LOW_PRIORITY);
    }
  }

  status = gm_set_acceptable_sizes (gmport, GM_LOW_PRIORITY, (1<<(maxsize))-1);

  gm_free_send_tokens (gmport, GM_LOW_PRIORITY,
                       gm_num_send_tokens (gmport));
}

