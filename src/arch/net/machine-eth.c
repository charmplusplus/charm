/*
  UDP implementation of Converse NET version
  contains only UDP specific code for:
  * CmiMachineInit()
  * CmiNotifyIdle()
  * DeliverViaNetwork()
  * CommunicationServer()

  moved from machine.c by 
  Gengbin Zheng, gzheng@uiuc.edu  4/22/2001
*/


/******************************************************************************
 *
 * CmiNotifyIdle()-- wait until a packet comes in
 *
 *****************************************************************************/
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

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  s->sleepMs=0;
  s->nIdles=0;
}

static void CmiNotifyStillIdle(CmiIdleState *s)
{
#if CMK_SHARED_VARS_UNAVAILABLE
  int nSpins=0; /*Number of times to spin before sleeping*/
#else
  int nSpins=20; /*Number of times to spin before sleeping*/
#endif

  s->nIdles++;
  if (s->nIdles>nSpins) { /*Start giving some time back to the OS*/
    s->sleepMs+=2;
    if (s->sleepMs>10) s->sleepMs=10;
  }
#if CMK_SHARED_VARS_UNAVAILABLE
  /*No comm. thread-- listen on sockets for incoming messages*/
  MACHSTATE(3,"idle commserver {")
  if (s->nIdles%4 ==3) CommunicationsClock();
  CommunicationServer(s->sleepMs);
  MACHSTATE(3,"} idle commserver")
#else
  /*Comm. thread will listen on sockets-- just sleep*/
  if (s->sleepMs>0)
    CmiIdleLock_sleep(&s->cs->idle,s->sleepMs);
#endif
}

void CmiNotifyIdle(void) {
  CmiIdleState s;
  s.sleepMs=5;
  CmiNotifyStillIdle(&s);
}

/***********************************************************************
 * TransmitAckDatagram
 *
 * This function sends the ack datagram, after setting the window
 * array to show which of the datagrams in the current window have been
 * received. The sending side will then use this information to resend
 * packets, mark packets as received, etc. This system also prevents
 * multiple retransmissions/acks when acks are lost.
 ***********************************************************************/
static int TransmitUDP(OtherNode node,void *data,int len)
{
  int retval = sendto(dataskt, data,len, 0,
	 (struct sockaddr *)&(node->addr),sizeof(node->addr));
  if (retval==-1) {
#if 0
    if (errno==EWOULDBLOCK) return 0; /*Outgoing full-- try again later*/
#endif
    if (errno==EINTR) return 0; /*Interrutped*/
    CmiAbort("Error sending UDP datagram to other node");
  }
  return 1;
}


static int TransmitAckDatagram(OtherNode node)
{
  DgramAck ack; int i, seqno, slot; ExplicitDgram dg;
  int retval;
  
  seqno = node->recv_next-1; /* Meaning: I have everything up to here*/
  MACHSTATE2(4,"   *** TransmitAck  (%d) to %d",seqno,node->nodestart)
  DgramHeaderMake(&ack, DGRAM_ACKNOWLEDGE, Cmi_nodestart, Cmi_charmrun_pid, seqno);
  LOG(Cmi_clock, Cmi_nodestart, 'A', node->nodestart, seqno);
  if (0==TransmitUDP(node,(void *)&ack,
	 DGRAM_HEADER_SIZE + Cmi_window_size + sizeof(unsigned int)))
    return 0;
  node->stat_send_ack++;
  if (node->recv_winsz) { /*We still have unintegrated packets*/
      node->recv_ack_cnt  = 1;
      node->recv_ack_time = Cmi_clock + Cmi_ack_delay;
      writeableAcks=1;
  } else { /*We've acknowledged everything we know about*/
      node->recv_ack_cnt  = 0;
      node->recv_ack_time = 1.0e30;
  }
  return 1;
}

/***********************************************************************
 * TransmitAcknowledgement
 *
 * This function sends the ack datagrams, after checking to see if the 
 * Recv Window is atleast half-full. After that, if the Recv window size 
 * is 0, then the count of un-acked datagrams, and the time at which
 * the ack should be sent is reset.
 ***********************************************************************/
static int TryTransmitAcknowledgement(OtherNode node)
{
#if 0
  printf("ACK> %d packets outstanding; clock %.2f of %.2f\n",
	 node->recv_ack_cnt,Cmi_clock, node->recv_ack_time);
#endif
  if ((node->recv_ack_cnt > Cmi_half_window) ||
      (Cmi_clock >= node->recv_ack_time)) {
    TransmitAckDatagram(node);
    return 1;
  }
  return 0;
}

void TransmitAcknowledgement()
{
  int skip; static int nextnode=0; OtherNode node;
  MACHSTATE(2,"  TransmitAcknowledgement {")
  writeableAcks=0;
  for (skip=0; skip<Cmi_numnodes; skip++) {
    node = nodes+nextnode;
    nextnode = (nextnode + 1) % Cmi_numnodes;
    if (node->recv_ack_cnt) 
      if (!TryTransmitAcknowledgement(node)) {
	MACHSTATE(2,"    Won't yet send ack")
	writeableAcks=1;
      }
  }
  MACHSTATE(2,"  } TransmitAcknowledgement")
}

/***********************************************************************
 * TransmitImplicitDgram
 *
 * This function does the actual work of (re)sending a datagram.
 ***********************************************************************/
static int TransmitImplicitDgram(ImplicitDgram dg,int isRetransmit)
{
  char *data; DgramHeader *head; int len; DgramHeader temp;
  OtherNode dest;
  int retval;

  MACHSTATE2(4,"  *** TransmitImplicit (%d) to %d",dg->seqno,dg->dest->nodestart)
  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head; /*Save message data trashed by DgramHeader*/ 
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_charmrun_pid, dg->seqno);
  LOG(Cmi_clock, Cmi_nodestart, isRetransmit?'P':'T', dest->nodestart, dg->seqno);
  if (0==TransmitUDP(dest,(void *)head,len + DGRAM_HEADER_SIZE)) {
    *head = temp; /*Restore message data under DgramHeader*/ 
    return 0;
  }
  *head = temp; /*Restore message data under DgramHeader*/ 
  if (isRetransmit) dest->stat_resend_pkt++;  
  else dest->stat_send_pkt++;
  return 1;
}

/***********************************************************************
 * TransmitDatagram()
 *
 * This function fills up the Send Window with the contents of the
 * Send Queue. It also sets the node->send_primer variable, which
 * indicates when a retransmission will be attempted.
 ***********************************************************************/
static int TryTransmitDatagram(OtherNode node,ImplicitDgram dg)
{
  unsigned int seqno = dg->seqno;
  int slot = seqno % Cmi_window_size;
  if (node->send_window[slot] == 0) {
    node->send_queue_h = dg->next;
    node->send_window[slot] = dg;
    TransmitImplicitDgram(dg,0);
    if (seqno == ((node->send_last+1)&DGRAM_SEQNO_MASK))
      node->send_last = seqno;
    node->send_primer = Cmi_clock + Cmi_delay_retransmit;
    return 1;
  }
  return 0;
}


void TransmitDatagram()
{
  ImplicitDgram dg; OtherNode node;
  static int nextnode=0; int skip, count;
  unsigned int seqno;
  MACHSTATE(2,"  TransmitDatagram {")
  writeableDgrams=0;
  for (skip=0; skip<Cmi_numnodes; skip++) {
    node = nodes+nextnode;
    nextnode = (nextnode + 1) % Cmi_numnodes;
    while (node->send_queue_h) { 
      MACHSTATE(2," Transmitting delayed packets ");
      if (!TryTransmitDatagram(node,node->send_queue_h)) {
	MACHSTATE1(5," **   Delaying transmit-- send window full ** (%d) ",node->send_queue_h->seqno);
	writeableDgrams=1;
	break;
      }
    }
    if (Cmi_clock > node->send_primer) 
    { /*Time to retransmit*/
      int packetsSent=0;
      for (count=0; count<node->retransmit_leash; count++) {
	dg = node->send_window[(node->send_good+1+count)%Cmi_window_size];
	if (dg) { /*Retransmit the first un-ack'd packet*/
	  MACHSTATE1(5," **   Timeout--retransmitting datagram ** (%d)",dg->seqno);
	  TransmitImplicitDgram(dg,1);
	  packetsSent++;
	}
      }
      node->send_primer = Cmi_clock + Cmi_delay_retransmit;
      node->retransmit_leash=1+node->retransmit_leash/2; /*Halve the leash length*/
      node->stat_consec_resend+=packetsSent;
    }
  }
  
  MACHSTATE(2,"  } TransmitDatagram")
}

/***********************************************************************
 * EnqueOutgoingDgram()
 *
 * This function enqueues the datagrams onto the Send queue of the
 * sender, after setting appropriate data values into each of the
 * datagrams. 
 ***********************************************************************/
void EnqueueOutgoingDgram
        (OutgoingMsg ogm, char *ptr, int len, OtherNode node, int rank)
{
  int seqno, dst, src; ImplicitDgram dg;
  MACHSTATE(2,"    EnqueueOutgoing")
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
  if (!TryTransmitDatagram(node,dg)) {
    MACHSTATE1(5," **   Delaying outgoing datagram ** (%d) ",dg->seqno)
    writeableDgrams=1;
  } else {
    MACHSTATE(3,"      Sending outgoing datagram")
  }
}


/***********************************************************************
 * DeliverViaNetwork()
 *
 * This function is responsible for all non-local transmission. This
 * function takes the outgoing messages, splits it into datagrams and
 * enqueues them into the Send Queue.
 ***********************************************************************/
void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank)
{
  int size; char *data;
  OtherNode myNode = nodes+CmiMyNode();
  MACHSTATE(1,"  DeliverViaNetwork {")
 
  size = ogm->size - DGRAM_HEADER_SIZE;
  data = ogm->data + DGRAM_HEADER_SIZE;
  while (size > Cmi_dgram_max_data) {
    EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank);
    data += Cmi_dgram_max_data;
    size -= Cmi_dgram_max_data;
  }
  EnqueueOutgoingDgram(ogm, data, size, node, rank);
  MACHSTATE(1,"  } DeliverViaNetwork")

  myNode->sent_msgs++;
  myNode->sent_bytes += ogm->size;
}

/***********************************************************************
 * AssembleDatagram()
 *
 * This function does the actual assembly of datagrams into a
 * message. node->asm_msg holds the current message being
 * assembled. Once the message assemble is complete (known by checking
 * if the total number of datagrams is equal to the number of datagrams
 * constituting the assembled message), the message is pushed into the
 * Producer-Consumer queue
 ***********************************************************************/
void AssembleDatagram(OtherNode node, ExplicitDgram dg)
{
  int i;
  unsigned int size; char *msg;
  OtherNode myNode = nodes+CmiMyNode();

  MACHSTATE2(4,"      AssembleDatagram (%d) from %d",dg->seqno,node->nodestart)
  LOG(Cmi_clock, Cmi_nodestart, 'X', dg->srcpe, dg->seqno);
  msg = node->asm_msg;
  if (msg == 0) {
    size = CmiMsgHeaderGetLength(dg->data);
    msg = (char *)CmiAlloc(size);
    if (!msg)
      fprintf(stderr, "%d: Out of mem\n", Cmi_mynode);
    if (size < dg->len) KillEveryoneCode(4559312);
    memcpy(msg, (char*)(dg->data), dg->len);
    node->asm_rank = dg->rank;
    node->asm_total = size;
    node->asm_fill = dg->len;
    node->asm_msg = msg;
  } else {
    size = dg->len - DGRAM_HEADER_SIZE;
    memcpy(msg + node->asm_fill, ((char*)(dg->data))+DGRAM_HEADER_SIZE, size);
    node->asm_fill += size;
  }
  if (node->asm_fill > node->asm_total)
      fprintf(stderr, "\n\n\t\tLength mismatch!!\n\n");
  if (node->asm_fill == node->asm_total) {
    if (node->asm_rank == DGRAM_BROADCAST) {
      int len = node->asm_total;
      for (i=1; i<Cmi_mynodesize; i++)
	CmiPushPE(i, CopyMsg(msg, len));
      CmiPushPE(0, msg);
    } else {
#if CMK_NODE_QUEUE_AVAILABLE
         if (node->asm_rank==DGRAM_NODEMESSAGE) {
	   PCQueuePush(CsvAccess(NodeRecv), msg);
         }
	 else
#endif
	   CmiPushPE(node->asm_rank, msg);
    }
    node->asm_msg = 0;
    myNode->recd_msgs++;
    myNode->recd_bytes += node->asm_total;
  }
  FreeExplicitDgram(dg);
}


/***********************************************************************
 * AssembleReceivedDatagrams()
 *
 * This function assembles the datagrams received so far, into a
 * single message. This also results in part of the Receive Window being 
 * freed.
 ***********************************************************************/
void AssembleReceivedDatagrams(OtherNode node)
{
  unsigned int next, slot; ExplicitDgram dg;
  MACHSTATE(2,"    AssembleDatagrams {")  
  next = node->recv_next;
  while (1) {
    slot = (next % Cmi_window_size);
    dg = node->recv_window[slot];
    if (dg == 0) break;
    AssembleDatagram(node, dg);
    node->recv_window[slot] = 0;
    node->recv_winsz--;
    next = ((next + 1) & DGRAM_SEQNO_MASK);
  }
  MACHSTATE(2,"    } AssembleDatagrams")
  node->recv_next = next;
}




/************************************************************************
 * IntegrateMessageDatagram()
 *
 * This function integrates the received datagrams. It first
 * increments the count of un-acked datagrams. (This is to aid the
 * heuristic that an ack should be sent when the Receive window is half
 * full). If the current datagram is the first missing packet, then this 
 * means that the datagram that was missing in the incomplete sequence
 * of datagrams so far, has arrived, and hence the datagrams can be
 * assembled. 
 ************************************************************************/


void IntegrateMessageDatagram(ExplicitDgram dg)
{
  int seqno;
  unsigned int slot; OtherNode node;

  MACHSTATE(2,"  IntegrateMessageDatagram {")  
  LOG(Cmi_clock, Cmi_nodestart, 'M', dg->srcpe, dg->seqno);
  node = nodes_by_pe[dg->srcpe];
  node->stat_recv_pkt++;
  seqno = dg->seqno;
  node->recv_ack_cnt++;
  if (node->recv_ack_time > Cmi_clock+Cmi_ack_delay)
    node->recv_ack_time = Cmi_clock + Cmi_ack_delay;
  if (((seqno - node->recv_next) & DGRAM_SEQNO_MASK) < Cmi_window_size) {
    slot = (seqno % Cmi_window_size);
    if (node->recv_window[slot] == 0) {
      node->recv_window[slot] = dg;
      node->recv_winsz++;
      if (seqno == node->recv_next)
	AssembleReceivedDatagrams(node);
      LOG(Cmi_clock, Cmi_nodestart, 'Y', node->recv_next, dg->seqno);
      if (!TryTransmitAcknowledgement(node))
        writeableAcks=1;
      return;
    }
    MACHSTATE1(5,"  Already have seqno %d packet",dg->seqno)    
  } 
  else 
  { /*We already have this datagram-- try to resynchronize*/
    MACHSTATE2(5,"  Throwing away wildly-unexpected seqno %d packet (ready for %d or better)",dg->seqno,node->recv_next)  
      /*TransmitAckDatagram(node);*/
     node->recv_ack_time=Cmi_clock;
  }
  LOG(Cmi_clock, Cmi_nodestart, 'y', node->recv_next, dg->seqno);
  FreeExplicitDgram(dg);
}



/***********************************************************************
 * IntegrateAckDatagram()
 * 
 * This function is called on the message sending side, on receipt of
 * an ack for a message that it sent. Since messages and acks could be 
 * lost, our protocol works in such a way that acks for higher sequence
 * numbered packets act as implict acks for lower sequence numbered
 * packets, in case the acks for the lower sequence numbered packets
 * were lost.

 * Recall that the Send and Receive windows are circular queues, and the
 * sequence numbers of the packets (datagrams) are monotically
 * increasing. Hence an ack for packet n implicitly acks all packets
 * numbered less than or equal to n.
 * 
 ***********************************************************************/
void IntegrateAckDatagram(ExplicitDgram dg)
{
  OtherNode node; 
  int i; 
  unsigned int ackseqno;
  int packetsAckd;

  node = nodes_by_pe[dg->srcpe];
  ackseqno = dg->seqno;
  MACHSTATE2(4,"  IntegrateAckDatagram (%d) from %d",ackseqno,dg->srcpe)

  node->stat_recv_ack++;
  LOG(Cmi_clock, Cmi_nodestart, 'R', node->nodestart, dg->seqno);

  /* check that the ack being received is within our window */
  if (!seqno_in_window(ackseqno,node->send_good+1))
  {
      MACHSTATE3(5," *** Discarding inappropriate ack (%d) from %d (expected %d or better) ***",ackseqno,dg->srcpe,node->send_good+1)
      FreeExplicitDgram(dg);
      return;
  } 
  
  /*Discard all the packets that made it*/
  packetsAckd=0;
  for (i=0; i<Cmi_window_size;i++) {
    ImplicitDgram idg=node->send_window[i];
    if (idg!=NULL && seqno_le(idg->seqno,ackseqno)) {
      /*This datagram arrived safely*/
      packetsAckd++;
      LOG(Cmi_clock, Cmi_nodestart, 'r', node->nodestart, seqno);
      node->send_window[i] = 0;
      DiscardImplicitDgram(idg);
    }
  }
  if (packetsAckd>0) {
    writeableDgrams=1; /*May have freed up some send slots*/
    node->retransmit_leash++; /*Some data actually made it*/
    if (node->retransmit_leash>Cmi_window_size) 
      node->retransmit_leash=Cmi_window_size;
    node->stat_consec_resend=0;
    node->send_good=ackseqno;
  }
  node->stat_ack_pkts+=packetsAckd;
  FreeExplicitDgram(dg);  
}

/*Grab the next datagram off this port.
Returns whether to call it again (1) or 
go on to other work (0).*/
int ReceiveDatagram()
{
  ExplicitDgram dg; int ok, magic;
  MallocExplicitDgram(dg);
  MACHSTATE(3,"ReceiveDatagram {")  
  ok = recv(dataskt,(char*)(dg->data),Cmi_max_dgram_size,0);
  /*ok = recvfrom(dataskt,(char*)(dg->data),Cmi_max_dgram_size,0, 0, 0);*/
  if (ok < 0) {
    FreeExplicitDgram(dg);
    if (errno == EINTR) return 1;  /* A SIGIO interrupted the receive */
#if !defined(_WIN32) || defined(__CYGWIN__) 
	if (errno == EWOULDBLOCK) return 0; /* No more messages on that socket. */
    if (errno == ECONNREFUSED) return 0;  /* A "Host unreachable" ICMP packet came in */
#endif
    CmiPrintf("ReceiveDatagram: recv: %s\n", strerror(errno)) ;
    KillEveryoneCode(37489437);
  }
  dg->len = ok;
#define SIMULATE_PACKET_LOSS 0
#if SIMULATE_PACKET_LOSS /*Randomly drop some incoming packets*/
  if (((rand()+(int)(100.0*CmiWallTimer()))%32)==0) { 
    printf("machine-eth.c intentionally dropping packet (net-debugging)\n");
    FreeExplicitDgram(dg);
    return 0;
  }
#endif
  if (ok >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(dg->data, dg->rank, dg->srcpe, magic, dg->seqno);
    if (magic == (Cmi_charmrun_pid&DGRAM_MAGIC_MASK)) {
      if (dg->rank == DGRAM_ACKNOWLEDGE)
	IntegrateAckDatagram(dg);
      else IntegrateMessageDatagram(dg);
      MACHSTATE(2,"} ReceiveDatagram")  
      return 1;
    }
    CmiError("Converse> Incorrect magic number %d on incoming UDP packet!\n",magic);
  }
  CmiError("Converse> Dropping strange %d-byte UDP packet!\n",ok,magic);
  MACHSTATE(5,"} Received Wierd Datagram!")
  FreeExplicitDgram(dg);
  return 1;
}


/***********************************************************************
 * CommunicationServer()
 * 
 * This function does the scheduling of the tasks related to the
 * message sends and receives. It is called from the CmiGeneralSend()
 * function, and periodically from the CommunicationInterrupt() (in case 
 * of the single processor version), and from the comm_thread (for the
 * SMP version). Based on which of the data/control read/write sockets
 * are ready, the corresponding tasks are called
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
  MACHSTATE1(sleepTime?3:2,"CommunicationsServer(%d)",sleepTime)  
#if !CMK_SHARED_VARS_UNAVAILABLE /*SMP mode: comm. lock is precious*/
  if (sleepTime!=0) /*Sleep *without* holding the comm. lock*/
    if (CheckSocketsReady(sleepTime)<=0) {
      MACHSTATE(2,"CommServer finished without anything happening.");
    }
  sleepTime=0;
#endif
  CmiCommLock();
  /*Don't sleep if a signal has stored messages for us*/
  if (sleepTime&&CmiGetState()->idle.hasMessages) sleepTime=0;
  while (CheckSocketsReady(sleepTime)>0) {
    sleepTime=0;
    MACHSTATE(2," -> Sockets Readable") 
    if (ctrlskt_ready_read) ctrl_getone();
    if (dataskt_ready_read) ReceiveDatagram();
    if (nTimes++ > 20) {
      /*We just grabbed a whole pile of packets-- try to retire a few*/
      CommunicationsClock();
      break;
    }
  }
  if (writeableAcks) TransmitAcknowledgement();
  if (writeableDgrams) TransmitDatagram();
  CmiCommUnlock();
  MACHSTATE(2,"} CommunicationServer")  
}

void CmiMachineInit()
{
}

