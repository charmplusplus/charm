/*
  TCP implementation of Converse NET version
  contains only TCP specific code for:
  * CmiMachineInit()
  * CmiNotifyIdle()
  * DeliverViaNetwork()
  * CommunicationServer()

  written by 
  Gengbin Zheng, gzheng@uiuc.edu  12/21/2001

*/

void ReceiveDatagram(SOCKET fd);
int TransmitDatagram(int pe);

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
  CommunicationServer(1);
#else
  /*Comm. thread will listen on sockets-- just sleep*/
  CmiIdleLock_sleep(&CmiGetState()->idle,5);
#endif
}

void CmiNotifyIdle(void) {
  CmiNotifyStillIdle(NULL);
}

#if CMK_USE_POLL
struct pollfd *fds=0;
int numSocks=0;
int numDataSocks=0;

int CmiSetupSockets()
{
  int i;
  numSocks = 0;
  if (!fds)
    fds = (struct pollfd  *)malloc((CmiNumPes()+5)*sizeof(struct pollfd));
  MACHSTATE(2,"CmiSetupSockets")
  if (dataskt!=-1) {
    for (i=0; i<CmiNumPes(); i++)
    {
/*CmiPrintf("[%d] %d - %d\n", CmiMyPe(), i, nodes[i].sock);*/
      if (i == CmiMyPe()) continue;
      fds[numSocks].fd = nodes[i].sock;
      fds[numSocks].events = POLLIN;
      if (nodes[i].send_queue_h) fds[numSocks].events |= POLLOUT;
      numSocks++;
    }
  }
  if (Cmi_charmrun_fd!=-1) {
    fds[numSocks].fd = Cmi_charmrun_fd;
    fds[numSocks].events = POLLIN;
    numSocks ++;
  }
}

void checkSocks()
{
  int n = 0, pe;
  if (dataskt!=-1) {
    for (pe=0; pe<CmiNumPes(); pe++)
    {
      if (pe == CmiMyPe()) continue;
      if (fds[n].revents & POLLIN) {
	dataskt_ready_read = 1;
        MACHSTATE1(2,"go to ReceiveDatagram %d", pe)
	ReceiveDatagram(fds[n].fd);
      }
      if (fds[n].revents & POLLOUT) {
        MACHSTATE1(2,"go to TransmitDatagram %d", pe)
	TransmitDatagram(pe);
      }
      n++;
    }
  }
  if (Cmi_charmrun_fd!=-1) {
    ctrlskt_ready_read = fds[n].revents & POLLIN;
    n++;
  }
}
#endif

int CheckSocketsReady(int withDelayMs)
{   
  int nreadable,i;
#if !CMK_USE_POLL
  static fd_set rfds; 
  static fd_set wfds; 
  struct timeval tmo;
  
  MACHSTATE(2,"CheckSocketsReady {")
  FD_ZERO(&rfds);FD_ZERO(&wfds);
  if (Cmi_charmrun_fd!=-1)
  	FD_S1T(Cmi_charmrun_fd, &rfds);
  if (dataskt!=-1) {
  for (i=0; i<Cmi_numnodes-1; i++)
  {
    FD_SET(nodes[i].recvsock, &rfds);
  }
  }
  tmo.tv_sec = 0;
  tmo.tv_usec = withDelayMs*1000;
  nreadable = select(FD_SETSIZE, &rfds, &wfds, NULL, &tmo);
#else
  int n = 0;
  MACHSTATE(1,"CheckSocketsReady {")
  CmiSetupSockets();
  MACHSTATE2(2,"CheckSocketsReady %d, %d socks", withDelayMs, numSocks)
  nreadable = poll(fds, numSocks, withDelayMs);
  MACHSTATE2(2,"CheckSocketsReady after poll %d %d", withDelayMs, nreadable)
#endif
  ctrlskt_ready_read = 0;
  dataskt_ready_read = 0;
  dataskt_ready_write = 0;

  if (nreadable == 0) {
    MACHSTATE(1,"} CheckSocketsReady (nothing readable)")
    return nreadable;
  }
  if (nreadable==-1) {
#if defined(_WIN32) && !defined(__CYGWIN__)
/* Win32 socket seems to randomly return inexplicable errors
here-- WSAEINVAL, WSAENOTSOCK-- yet everything is actually OK. 
	int err=WSAGetLastError();
	CmiPrintf("(%d)Select returns -1; errno=%d, WSAerr=%d\n",withDelayMs,errno,err);
*/
#else
	if (errno!=EINTR)
		KillEveryone("Socket error in CheckSocketsReady!\n");
#endif
    MACHSTATE(2,"} CheckSocketsReady (INTERRUPTED!)")
    return CheckSocketsReady(0);
  }
#if !CMK_USE_POLL
  if (Cmi_charmrun_fd!=-1)
	ctrlskt_ready_read = (FD_ISSET(Cmi_charmrun_fd, &rfds));
  if (dataskt!=-1) {
	dataskt_ready_write = (FD_ISSET(dataskt, &wfds));
  for (i=0; i<Cmi_numnodes-1; i++)
  {
    if (FD_ISSET(nodes[i].recvsock, &rfds)) {
	dataskt_ready_read = 1;
	ReceiveDatagram(nodes[i].recvsock);
    }
  }
  }
#else
  checkSocks();
#endif
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


static void CommunicationServer(int sleepTime)
{
  unsigned int nTimes=0; /* Loop counter */
  CmiCommLockOrElse({
    MACHSTATE(4,"Attempted to re-enter comm. server!") 
    return;
  });
  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);
  MACHSTATE2(sleepTime?3:2,"CommunicationsServer(%d,%d) {",
	     sleepTime,writeableAcks||writeableDgrams)  
#if !CMK_SHARED_VARS_UNAVAILABLE /*SMP mode: comm. lock is precious*/
  if (sleepTime!=0) {/*Sleep *without* holding the comm. lock*/
    MACHSTATE(2,"CommServer going to sleep (NO LOCK)");
    if (CheckSocketsReady(sleepTime)<=0) {
      MACHSTATE(2,"CommServer finished without anything happening.");
    }
  }
  sleepTime=0;
#endif
  CmiCommLock();
/*  CommunicationsClock(); */
  /*Don't sleep if a signal has stored messages for us*/
  if (sleepTime&&CmiGetState()->idle.hasMessages) sleepTime=0;
  MACHSTATE(2," enter CheckSocket") 
  while (CheckSocketsReady(sleepTime)>0) {
    int again=0;
    MACHSTATE(2," after CheckSocket") 
    sleepTime=0;
    if (ctrlskt_ready_read) {again=1;ctrl_getone();}
    if (dataskt_ready_read) {again=1;}
    break;
    if (!again) break; /* Nothing more to do */
#if 0
    if ((nTimes++ &16)==15) {
      /*We just grabbed a whole pile of packets-- try to retire a few*/
      CommunicationsClock();
    }
#endif
  }
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

void ReceiveDatagram(SOCKET fd)
{
  ChMessage msg;
  int rank, srcpe, seqno, magic, i;
  int size;
  int ok;
  double t;

  if (-1==ChMessage_recv(fd,&msg))
    CmiAbort("Error in ReceiveDatagram.");

  processMessage(msg.data, msg.len);

  ChMessage_free(&msg);

}


/***********************************************************************
 * DeliverViaNetwork()
 *
 * This function is responsible for all non-local transmission. It
 * first allocate a send token, if fails, put the send message to
 * penging message queue, otherwise invoke the GM send.
 ***********************************************************************/

int TransmitImplicitDgram(ImplicitDgram dg)
{
  ChMessageHeader msg;
  char *data; DgramHeader *head; int len; DgramHeader temp;
  OtherNode dest;
  int retval;
  
  MACHSTATE2(2,"  TransmitImplicitDgram (%d bytes) [%d]",dg->datalen,dg->seqno)
  len = dg->datalen+DGRAM_HEADER_SIZE;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_charmrun_pid, dg->seqno);
  LOG(Cmi_clock, Cmi_nodestart, 'T', dest->nodestart, dg->seqno);
  ChMessageHeader_new("data", len, &msg);
  if (-1==skt_sendN(dest->sock,(const char *)&msg,sizeof(msg))) 
    CmiAbort("EnqueueOutgoingDgram"); 
  if (-1==skt_sendN(dest->sock,head,len))
    CmiAbort("EnqueueOutgoingDgram"); 
  *head = temp;
  dest->stat_send_pkt++;
  return 1;
}

int TransmitDatagram(int pe)
{
  ImplicitDgram dg; OtherNode node;
  int count;
  unsigned int seqno;
  
/*CmiPrintf("[%d] TransmitDatagram to %d\n", CmiMyPe(), pe);*/
  node = nodes+pe;
  dg = node->send_queue_h;
  if (dg) {
    if (TransmitImplicitDgram(dg)) {
      node->send_queue_h = dg->next;
      DiscardImplicitDgram(dg);
    }
  }
  return 0;
}

void EnqueueOutgoingDgram
        (OutgoingMsg ogm, char *ptr, int len, OtherNode node, int rank)
{
  int seqno, dst, src; ImplicitDgram dg;
  src = ogm->src;
  dst = ogm->dst;
  seqno = node->send_next;
  node->send_next = ((seqno+1)&DGRAM_SEQNO_MASK);
  MallocImplicitDgram(dg);
  dg->dest = node;
  dg->srcpe = src;
  dg->rank = rank;
  dg->seqno = seqno;
  dg->dataptr = ptr;
  dg->datalen = len;
  dg->ogm = ogm;
  ogm->refcount++;
  dg->next = 0;
  if (node->send_queue_h == 0) {
    node->send_queue_h = dg;
    node->send_queue_t = dg;
  } else {
    node->send_queue_t->next = dg;
    node->send_queue_t = dg;
  }
}

void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank)
{
  int size; char *data;

/*CmiPrintf("DeliverViaNetwork to %d\n", node->nodestart);*/
/*CmiPrintf("send time: %fus\n", (CmiWallTimer()-t)*1.0e6); */
 
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
  Cmi_dgram_max_data = 1400-DGRAM_HEADER_SIZE;
}


static void open_tcp_sockets()
{
  int i, ok, pe;
  int mype, numpes;
  SOCKET skt;

/*
  FILE *fp;
  char file[100];
  sprintf(file, "out.%d", Cmi_mynode);
  fp = fopen(file, "a");
  fprintf(fp, "%d %d\n", Cmi_mynode, Cmi_numnodes);
  fclose(fp);
  mype = Cmi_mynode;
  numpes = Cmi_numnodes;
*/
  mype = CmiMyPe();
  numpes = CmiNumPes();
  MACHSTATE2(2,"  open_tcp_sockets (%d:%d)", mype, numpes);
  for (i=0; i<mype; i++) {
    unsigned int clientPort;
    skt_ip_t clientIP;
    skt = skt_accept(dataskt, &clientIP,&clientPort);
    if (skt<0) KillEveryoneCode(98246554);
    ok = skt_recvN(skt, &pe, sizeof(int));
    if (ok<0) KillEveryoneCode(98246556);
/*printf("recv %d %d\n", mype, pe);*/
    nodes[pe].sock = skt;
  }
  for (pe=mype+1; pe<numpes; pe++) {
    skt = skt_connect(nodes[pe].IP, nodes[pe].dataport, 300);
    if (skt<0) KillEveryoneCode(894788843);
    ok = skt_sendN(skt, &mype, sizeof(int));
    if (ok<0) KillEveryoneCode(98246556);
/*printf("send %d %d\n", mype, pe); */
    nodes[pe].sock = skt;
  }
}


