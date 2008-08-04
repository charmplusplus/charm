/** @file
 * ibverbs unreliable datagram implementation of Converse NET version
 * @ingroup NET
 * contains only ibverbs specific code for:
 * - CmiMachineInit()
 * - CmiCommunicationInit()
 * - CmiNotifyIdle()
 * - DeliverViaNetwork()
 * - CommunicationServer()

   Eric Shook and Esteban Meneses - Jul 22, 2008
*/

/**
 * @addtogroup NET
 * @{
 */

// FIXME: Note: Charm does not guarantee in order messages - can use for bettter performance


#include <infiniband/verbs.h>

#define WC_LIST_SIZE 32

#define INFIPACKETCODE_DATA 1

#define METADATAFIELD(m) (((infiCmiChunkHeader *)m)[-1].metaData)


enum ibv_mtu mtu = IBV_MTU_2048;
static int mtu_size;
static int maxrecvbuffers;
static int maxtokens;


struct ibudstruct {
	struct ibv_device **devlist;
	struct ibv_device *dev;
};

struct ibudstruct ibud;

struct infiPacketHeader{
	char code;
	int nodeNo;
};

/** Represents a qp used to send messages to another node
 There is one for each remote node */
struct infiAddr {
	int lid,qpn,psn;
};


struct infiContext {
	struct ibv_context	*context;
	
	fd_set  asyncFds;
	struct timeval tmo;
	
	int ibPort;
	struct ibv_pd		*pd;
	struct ibv_cq		*sendCq;
	struct ibv_cq   	*recvCq;
	struct ibv_srq  	*srq;
	struct ibv_mr		*mr;
	
	struct ibv_qp		*qp; 	//Array of qps (numNodes long) to temporarily store the queue pairs
					//It is used between CmiMachineInit and the call to node_addresses_store
					//when the qps are stored in the corresponding OtherNodes

	struct infiAddr localAddr; //store the lid,qpn,msn address of ur qpair until they are sent

	struct infiPacketHeader header;
	int sendCqSize,recvCqSize;

	void *buffer; // Registered memory buffer for msg's
};

static struct infiContext *context;


typedef struct infiPacketStruct {
        char *buf;
        int size;
        struct infiPacketHeader header;
        struct ibv_mr *keyHeader;
        struct OtherNodeStruct *destNode;
        struct infiPacketStruct *next;
        OutgoingMsg ogm;
        struct ibv_sge elemList[2];
        struct ibv_send_wr wr;
}* infiPacket;

struct infiOtherNodeData{
	int state;// does it expect a packet with a header (first packet) or one without
	int totalTokens;
	int tokensLeft;
	int nodeNo;

	int postedRecvs;
	int broot;//needed to store the root of a multi-packet broadcast sent along a spanning tree or hypercube
	struct infiAddr qp;
};



typedef struct {
  int sleepMs; /*Milliseconds to sleep while idle*/
  int nIdles;  /*Number of times we've been idle in a row*/
  CmiState cs; /*Machine state*/
} CmiIdleState;


/*
#define FreeInfiPacket(pkt){ \
        pkt->size = -1;\
        pkt->ogm=NULL;\
        pkt->next = context->infiPacketFreeList; \
        context->infiPacketFreeList = pkt; \
}

#define MallocInfiPacket(pkt) { \
	infiPacket p = context->infiPacketFreeList; \
	if(p == NULL){ p = newPacket();} \
	else{context->infiPacketFreeList = p->next; } \
	pkt = p;\
}
*/




static CmiIdleState *CmiNotifyGetState(void) { 
	return NULL; 
}

static void CmiNotifyStillIdle(CmiIdleState *s); 
static void CmiNotifyBeginIdle(CmiIdleState *s) {
  CmiNotifyStillIdle(s);
}


static inline  void CommunicationServer_lock(int toBuffer);
static inline  void CommunicationServer_nolock(int toBuffer);

static void CmiNotifyStillIdle(CmiIdleState *s) { 
#if CMK_SMP
        CommunicationServer_lock(0);
#else
        CommunicationServer_nolock(0);
#endif
}

/******************************************************************************
 *
 * CmiNotifyIdle()-- wait until a packet comes in
 *
 *****************************************************************************/

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
  int nreadable,dataWrite=writeableDgrams || writeableAcks;
  CMK_PIPE_DECL(withDelayMs);


#if CMK_USE_KQUEUE && 0
  // This implementation doesn't yet work, but potentially is much faster

  /* Only setup the CMK_PIPE structures the first time they are used. 
     This makes the kqueue implementation much faster.
  */
  static int first = 1;
  if(first){
    first = 0;
    CmiStdoutAdd(CMK_PIPE_SUB);
    if (Cmi_charmrun_fd!=-1) { CMK_PIPE_ADDREAD(Cmi_charmrun_fd); }
    else return 0; /* If there's no charmrun, none of this matters. */
    if (dataskt!=-1) {
      CMK_PIPE_ADDREAD(dataskt); 
      CMK_PIPE_ADDWRITE(dataskt);
    }
  }

#else  
  CmiStdoutAdd(CMK_PIPE_SUB);  
  if (Cmi_charmrun_fd!=-1) { CMK_PIPE_ADDREAD(Cmi_charmrun_fd); }  
  else return 0; /* If there's no charmrun, none of this matters. */  
  if (dataskt!=-1) {  
    { CMK_PIPE_ADDREAD(dataskt); }  
    if (dataWrite)  
      CMK_PIPE_ADDWRITE(dataskt);  
  }  
#endif 

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
  if (dataskt!=-1) {
	dataskt_ready_read = CMK_PIPE_CHECKREAD(dataskt);
	if (dataWrite)
		dataskt_ready_write = CMK_PIPE_CHECKWRITE(dataskt);
  }
  return nreadable;
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
//													OPTIONAL - can be removed
/*
void TransmitAckDatagram(OtherNode node)
{
  DgramAck ack; int i, seqno, slot; ExplicitDgram dg;
  int retval;
  
  seqno = node->recv_next;
  MACHSTATE2(3,"  TransmitAckDgram [seq %d to 'pe' %d]",seqno,node->nodestart)
  DgramHeaderMake(&ack, DGRAM_ACKNOWLEDGE, Cmi_nodestart, Cmi_charmrun_pid, seqno, 0);
  LOG(Cmi_clock, Cmi_nodestart, 'A', node->nodestart, seqno);
  for (i=0; i<Cmi_window_size; i++) {
    slot = seqno % Cmi_window_size;
    dg = node->recv_window[slot];
    ack.window[i] = (dg && (dg->seqno == seqno));
    seqno = ((seqno+1) & DGRAM_SEQNO_MASK);
  }
  memcpy(&ack.window[Cmi_window_size], &(node->send_ack_seqno), 
          sizeof(unsigned int));
  node->send_ack_seqno = ((node->send_ack_seqno + 1) & DGRAM_SEQNO_MASK);
  retval = (-1);
  while(retval==(-1))
    retval = sendto(dataskt, (char *)&ack,
	 DGRAM_HEADER_SIZE + Cmi_window_size + sizeof(unsigned int), 0,
	 (struct sockaddr *)&(node->addr),
	 sizeof(struct sockaddr_in));
  node->stat_send_ack++;
}
*/

/***********************************************************************
 * TransmitImplicitDgram
 * TransmitImplicitDgram1
 *
 * These functions do the actual work of sending a UDP datagram.
 ***********************************************************************/
//													OPTIONAL - can be removed
/*
void TransmitImplicitDgram(ImplicitDgram dg)
{
  char *data; DgramHeader *head; int len; DgramHeader temp;
  OtherNode dest;
  int retval;
  
  MACHSTATE3(3,"  TransmitImplicitDgram (%d bytes) [seq %d to 'pe' %d]",
	     dg->datalen,dg->seqno,dg->dest->nodestart)
  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_charmrun_pid, dg->seqno, dg->broot);
  LOG(Cmi_clock, Cmi_nodestart, 'T', dest->nodestart, dg->seqno);
  retval = (-1);
  while(retval==(-1))
    retval = sendto(dataskt, (char *)head, len + DGRAM_HEADER_SIZE, 0,
	      (struct sockaddr *)&(dest->addr), sizeof(struct sockaddr_in));
  *head = temp;
  dest->stat_send_pkt++;
}

void TransmitImplicitDgram1(ImplicitDgram dg)
{
  char *data; DgramHeader *head; int len; DgramHeader temp;
  OtherNode dest;
  int retval;

  MACHSTATE3(4,"  RETransmitImplicitDgram (%d bytes) [seq %d to 'pe' %d]",
	     dg->datalen,dg->seqno,dg->dest->nodestart)
  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_charmrun_pid, dg->seqno, dg->broot);
  LOG(Cmi_clock, Cmi_nodestart, 'P', dest->nodestart, dg->seqno);
  retval = (-1);
  while (retval == (-1))
    retval = sendto(dataskt, (char *)head, len + DGRAM_HEADER_SIZE, 0,
	      (struct sockaddr *)&(dest->addr), sizeof(struct sockaddr_in));
  *head = temp;
  dest->stat_resend_pkt++;
}
*/

/***********************************************************************
 * TransmitAcknowledgement
 *
 * This function sends the ack datagrams, after checking to see if the 
 * Recv Window is atleast half-full. After that, if the Recv window size 
 * is 0, then the count of un-acked datagrams, and the time at which
 * the ack should be sent is reset.
 ***********************************************************************/
//													OPTIONAL - can be removed
/*
int TransmitAcknowledgement()
{
  int skip; static int nextnode=0; OtherNode node;
  for (skip=0; skip<_Cmi_numnodes; skip++) {
    node = nodes+nextnode;
    nextnode = (nextnode + 1) % _Cmi_numnodes;
    if (node->recv_ack_cnt) {
      if ((node->recv_ack_cnt > Cmi_half_window) ||
	  (Cmi_clock >= node->recv_ack_time)) {
	TransmitAckDatagram(node);
	if (node->recv_winsz) {
	  node->recv_ack_cnt  = 1;
	  node->recv_ack_time = Cmi_clock + Cmi_ack_delay;
	} else {
	  node->recv_ack_cnt  = 0;
	  node->recv_ack_time = 0.0;
	}
	return 1;
      }
    }
  }
  return 0;
}
*/

/***********************************************************************
 * TransmitDatagram()
 *
 * This function fills up the Send Window with the contents of the
 * Send Queue. It also sets the node->send_primer variable, which
 * indicates when a retransmission will be attempted.
 ***********************************************************************/
//													OPTIONAL - can be removed
/*
int TransmitDatagram()
{
  ImplicitDgram dg; OtherNode node;
  static int nextnode=0; int skip, count, slot;
  unsigned int seqno;
  
  for (skip=0; skip<_Cmi_numnodes; skip++) {
    node = nodes+nextnode;
    nextnode = (nextnode + 1) % _Cmi_numnodes;
    dg = node->send_queue_h;
    if (dg) {
      seqno = dg->seqno;
      slot = seqno % Cmi_window_size;
      if (node->send_window[slot] == 0) {
	node->send_queue_h = dg->next;
	node->send_window[slot] = dg;
	TransmitImplicitDgram(dg);
	if (seqno == ((node->send_last+1)&DGRAM_SEQNO_MASK))
	  node->send_last = seqno;
	node->send_primer = Cmi_clock + Cmi_delay_retransmit;
	return 1;
      }
    }
    if (Cmi_clock > node->send_primer) {
      slot = (node->send_last % Cmi_window_size);
      for (count=0; count<Cmi_window_size; count++) {
	dg = node->send_window[slot];
	if (dg) break;
	slot = ((slot+Cmi_window_size-1) % Cmi_window_size);
      }
      if (dg) {
	TransmitImplicitDgram1(node->send_window[slot]);
	node->send_primer = Cmi_clock + Cmi_delay_retransmit;
	return 1;
      }
    }
  }
  return 0;
}
*/

/***********************************************************************
 * EnqueOutgoingDgram()
 *
 * This function enqueues the datagrams onto the Send queue of the
 * sender, after setting appropriate data values into each of the
 * datagrams. 
 ***********************************************************************/
//													OPTIONAL - can be removed - don't remove

void EnqueueOutgoingDgram
        (OutgoingMsg ogm, char *ptr, int len, OtherNode node, int rank, int broot)
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
  dg->broot = broot;
  dg->dataptr = ptr;
  dg->datalen = len;
  dg->ogm = ogm;
//  ogm->refcount++;
  dg->next = 0;
  if (node->send_queue_h == 0) {
    node->send_queue_h = dg;
    node->send_queue_t = dg;
  } else {
    node->send_queue_t->next = dg;
    node->send_queue_t = dg;
  }
}


static void inline EnqueuePacket(OtherNode node,infiPacket packet,int size,struct ibv_mr *dataKey){
	int retval;
	struct ibv_send_wr *bad_wr=NULL;

	packet->elemList[1].addr = (uintptr_t)packet->buf;
	packet->elemList[1].length = size;
	packet->elemList[1].lkey = dataKey->lkey;
	
	packet->destNode = node;
	
	//if(retval = ibv_post_send(node->infiData->qp,&(packet->wr),&bad_wr)){
	if(retval = ibv_post_send(context->qp,&(packet->wr),&bad_wr)){ 
		CmiPrintf("[%d] Sending to node %d failed with return value %d\n",_Cmi_mynode,node->infiData->nodeNo,retval);
		CmiAssert(0);
	}

//        MACHSTATE4(3,"Packet send size %d node %d tokensLeft %d psn %d",size,packet->destNode->infiData->nodeNo,context->tokensLeft,packet->header.psn);
}

//static void inline EnqueueDataPacket(OutgoingMsg ogm, OtherNode node, int rank,char *data,int size,int broot,int copy){
static void inline EnqueueDataPacket(OutgoingMsg ogm, char *data, int size, OtherNode node, int rank, int broot) {

	infiPacket packet;
//	MallocInfiPacket(packet);
	packet->size = size;
	packet->buf=data;
	
	//the nodeNo is added at time of packet allocation
	packet->header.code = INFIPACKETCODE_DATA;
	
//	ogm->refcount++;
	packet->ogm = ogm;
	
/*
	// FIXME: look at line 328 & 454 of example code
	struct ibv_mr *key = METADATAFIELD(ogm->data)->key;
	CmiAssert(key != NULL);
	
	EnqueuePacket(node,packet,size,key);
*/
}




/***********************************************************************
 * DeliverViaNetwork()
 *
 * This function is responsible for all non-local transmission. This
 * function takes the outgoing messages, splits it into datagrams and
 * enqueues them into the Send Queue.
 ***********************************************************************/
void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy) {
	int size; char *data;
	size=ogm->size;
	data=ogm->data;
	DgramHeaderMake(data, rank, ogm->src, Cmi_charmrun_pid, 1, broot); // May not be needed
	CmiMsgHeaderSetLength(data,size);
	while(size>Cmi_dgram_max_data) {
		//EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank, broot);
		EnqueueDataPacket(ogm, data, Cmi_dgram_max_data, node, rank, broot);
//		EnqueueDataPacket(ogm,node,rank,data,dataSize,broot,copy);
		size -= Cmi_dgram_max_data;
		data += Cmi_dgram_max_data;

	}
	if(size>0)
		EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank, broot);

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
//													OPTIONAL - can be removed
/*
void AssembleDatagram(OtherNode node, ExplicitDgram dg)
{
  int i;
  unsigned int size; char *msg;
  OtherNode myNode = nodes+CmiMyNode();
  
  MACHSTATE3(2,"  AssembleDatagram [seq %d from 'pe' %d, packet len %d]",
  	dg->seqno,node->nodestart,dg->len)
  LOG(Cmi_clock, Cmi_nodestart, 'X', dg->srcpe, dg->seqno);
  msg = node->asm_msg;
  if (msg == 0) {
    size = CmiMsgHeaderGetLength(dg->data);
    MACHSTATE3(4,"  Assemble new datagram seq %d from 'pe' %d, len %d",
    	dg->seqno,node->nodestart,size)
    msg = (char *)CmiAlloc(size);
    if (!msg)
      fprintf(stderr, "%d: Out of mem\n", _Cmi_mynode);
    if (size < dg->len) KillEveryoneCode(4559312);
#ifndef CMK_OPTIMIZE
    setMemoryTypeMessage(msg);
#endif
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
  MACHSTATE3(2,"  AssembleDatagram: now have %d of %d bytes from %d",
  	node->asm_fill, node->asm_total, node->nodestart)
  if (node->asm_fill > node->asm_total) {
      fprintf(stderr, "\n\n\t\tLength mismatch!!\n\n");
      fflush(stderr);
      MACHSTATE4(5,"Length mismatch seq %d, from 'pe' %d, fill %d, total %d\n", dg->seqno,node->nodestart,node->asm_fill,node->asm_total)
      KillEveryoneCode(4559313);
  }
  if (node->asm_fill == node->asm_total) {
    // spanning tree broadcast - send first to avoid invalid msg ptr 
#if CMK_BROADCAST_SPANNING_TREE
    if (node->asm_rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE 
          || node->asm_rank == DGRAM_NODEBROADCAST
#endif
      )
        SendSpanningChildren(NULL, 0, node->asm_total, msg, dg->broot, dg->rank);
#elif CMK_BROADCAST_HYPERCUBE
    if (node->asm_rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || node->asm_rank == DGRAM_NODEBROADCAST
#endif
      )
        SendHypercube(NULL, 0, node->asm_total, msg, dg->broot, dg->rank);
#endif
    if (node->asm_rank == DGRAM_BROADCAST) {
      int len = node->asm_total;
      for (i=1; i<_Cmi_mynodesize; i++)
         CmiPushPE(i, CopyMsg(msg, len));
      CmiPushPE(0, msg);
    } else {
#if CMK_NODE_QUEUE_AVAILABLE
         if (node->asm_rank==DGRAM_NODEMESSAGE ||
	     node->asm_rank==DGRAM_NODEBROADCAST) 
	 {
	   CmiPushNode(msg);
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
*/


/***********************************************************************
 * AssembleReceivedDatagrams()
 *
 * This function assembles the datagrams received so far, into a
 * single message. This also results in part of the Receive Window being 
 * freed.
 ***********************************************************************/
//													OPTIONAL - can be removed
/*
void AssembleReceivedDatagrams(OtherNode node)
{
  unsigned int next, slot; ExplicitDgram dg;
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
  node->recv_next = next;
}
*/



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

//													OPTIONAL - can be removed
/*
void IntegrateMessageDatagram(ExplicitDgram dg)
{
  int seqno;
  unsigned int slot; OtherNode node;

  LOG(Cmi_clock, Cmi_nodestart, 'M', dg->srcpe, dg->seqno);
  MACHSTATE2(2,"  IntegrateMessageDatagram [seq %d from pe %d]", dg->seqno,dg->srcpe)

  node = nodes_by_pe[dg->srcpe];
  node->stat_recv_pkt++;
  seqno = dg->seqno;
  writeableAcks=1;
  node->recv_ack_cnt++;
  if (node->recv_ack_time == 0.0)
    node->recv_ack_time = Cmi_clock + Cmi_ack_delay;
  if (((seqno - node->recv_next) & DGRAM_SEQNO_MASK) < Cmi_window_size) {
    slot = (seqno % Cmi_window_size);
    if (node->recv_window[slot] == 0) {
      node->recv_window[slot] = dg;
      node->recv_winsz++;
      if (seqno == node->recv_next)
	AssembleReceivedDatagrams(node);
      if (seqno > node->recv_expect)
	node->recv_ack_time = 0.0;
      if (seqno >= node->recv_expect)
	node->recv_expect = ((seqno+1)&DGRAM_SEQNO_MASK);
      LOG(Cmi_clock, Cmi_nodestart, 'Y', node->recv_next, dg->seqno);
      return;
    }
  }
  LOG(Cmi_clock, Cmi_nodestart, 'y', node->recv_next, dg->seqno);
  FreeExplicitDgram(dg);
}
*/


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
 * increasing. Hence it is important to know for which sequence number
 * the ack is for, and to correspodinly relate that to tha actual packet 
 * sitting in the Send window. Since every 20th packet occupies the same
 * slot in the windows, a number of sanity checks are required for our
 * protocol to work. 
 * 1. If the ack number (first missing packet sequence number) is less
 * than the last ack number received then this ack can be ignored. 

 * 2. The last ack number received must be set to the current ack
 * sequence number (This is done only if 1. is not true).

 * 3. Now the whole Send window is examined, in a kind of reverse
 * order. The check starts from a sequence number = 20 + the first
 * missing packet's sequence number. For each of these sequence numbers, 
 * the slot in the Send window is checked for existence of a datagram
 * that should have been sent. If there is no datagram, then the search
 * advances. If there is a datagram, then the sequence number of that is 
 * checked with the expected sequence number for the current iteration
 * (This is decremented in each iteration of the loop).

 * If the sequence numbers do not match, then checks are made (for
 * the unlikely scenarios where the current slot sequence number is 
 * equal to the first missing packet's sequence number, and where
 * somehow, packets which have greater sequence numbers than allowed for 
 * the current window)

 * If the sequence numbers DO match, then the flag 'rxing' is
 * checked. The semantics for this flag is that : If any packet with a
 * greater sequence number than the current packet (and hence in the
 * previous iteration of the for loop) has been acked, then the 'rxing'
 * flag is set to 1, to imply that all the packets of lower sequence
 * number, for which the ack->window[] element does not indicate that the 
 * packet has been received, must be retransmitted.
 * 
 ***********************************************************************/

//													OPTIONAL - can be removed
/*
void IntegrateAckDatagram(ExplicitDgram dg)
{
  OtherNode node; DgramAck *ack; ImplicitDgram idg;
  int i; unsigned int slot, rxing, dgseqno, seqno, ackseqno;
  int diff;
  unsigned int tmp;

  node = nodes_by_pe[dg->srcpe];
  ack = ((DgramAck*)(dg->data));
  memcpy(&ackseqno, &(ack->window[Cmi_window_size]), sizeof(unsigned int));
  dgseqno = dg->seqno;
  seqno = (dgseqno + Cmi_window_size) & DGRAM_SEQNO_MASK;
  slot = seqno % Cmi_window_size;
  rxing = 0;
  node->stat_recv_ack++;
  LOG(Cmi_clock, Cmi_nodestart, 'R', node->nodestart, dg->seqno);

  tmp = node->recv_ack_seqno;
  // check that the ack being received is actually appropriate 
  if ( !((node->recv_ack_seqno >= 
	  ((DGRAM_SEQNO_MASK >> 1) + (DGRAM_SEQNO_MASK >> 2))) &&
	 (ackseqno < (DGRAM_SEQNO_MASK >> 1))) &&
       (ackseqno <= node->recv_ack_seqno))
    {
      FreeExplicitDgram(dg);
      return;
    } 
  // higher ack so adjust 
  node->recv_ack_seqno = ackseqno;
  writeableDgrams=1; // May have freed up some send slots 
  
  for (i=Cmi_window_size-1; i>=0; i--) {
    slot--; if (slot== ((unsigned int)-1)) slot+=Cmi_window_size;
    seqno = (seqno-1) & DGRAM_SEQNO_MASK;
    idg = node->send_window[slot];
    if (idg) {
      if (idg->seqno == seqno) {
	if (ack->window[i]) {
	  // remove those that have been received and are within a window
	  // of the first missing packet 
	  node->stat_ack_pkts++;
	  LOG(Cmi_clock, Cmi_nodestart, 'r', node->nodestart, seqno);
	  node->send_window[slot] = 0;
	  DiscardImplicitDgram(idg);
	  rxing = 1;
	} else if (rxing) {
	  node->send_window[slot] = 0;
	  idg->next = node->send_queue_h;
	  if (node->send_queue_h == 0) {
	    node->send_queue_t = idg;
	  }
	  node->send_queue_h = idg;
	}
      } else {
        diff = dgseqno >= idg->seqno ? 
	  ((dgseqno - idg->seqno) & DGRAM_SEQNO_MASK) :
	  ((dgseqno + (DGRAM_SEQNO_MASK - idg->seqno) + 1) & DGRAM_SEQNO_MASK);
	  
	if ((diff <= 0) || (diff > Cmi_window_size))
	{
	  continue;
	}

        // if ack is really less than our packet seq (consider wrap around) 
        if (dgseqno < idg->seqno && (idg->seqno - dgseqno <= Cmi_window_size))
        {
          continue;
        }
        if (dgseqno == idg->seqno)
        {
	  continue;
        }
	node->stat_ack_pkts++;
	LOG(Cmi_clock, Cmi_nodestart, 'o', node->nodestart, idg->seqno);
	node->send_window[slot] = 0;
	DiscardImplicitDgram(idg);
      }
    }
  }
  FreeExplicitDgram(dg);  
}

//													OPTIONAL - can be removed
void ReceiveDatagram()
{
  ExplicitDgram dg; int ok, magic;
  MACHLOCK_ASSERT(comm_flag,"ReceiveDatagram")
  MallocExplicitDgram(dg);
  ok = recv(dataskt,(char*)(dg->data),Cmi_max_dgram_size,0);
  //ok = recvfrom(dataskt,(char*)(dg->data),Cmi_max_dgram_size,0, 0, 0);
  if (ok < 0) {
    MACHSTATE1(4,"  recv dgram failed (errno=%d)",errno)
    FreeExplicitDgram(dg);
    if (errno == EINTR) return;  // A SIGIO interrupted the receive 
    if (errno == EAGAIN) return; // Just try again later 
#if !defined(_WIN32) || defined(__CYGWIN__) 
    if (errno == EWOULDBLOCK) return; // No more messages on that socket. 
    if (errno == ECONNREFUSED) return;  // A "Host unreachable" ICMP packet came in 
#endif
    CmiPrintf("ReceiveDatagram: recv: %s(%d)\n", strerror(errno), errno) ;
    KillEveryoneCode(37489437);
  }
  dg->len = ok;

  if (ok >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(dg->data, dg->rank, dg->srcpe, magic, dg->seqno, dg->broot);
    MACHSTATE3(2,"  recv dgram [seq %d, for rank %d, from pe %d]",
	       dg->seqno,dg->rank,dg->srcpe)
    if (magic == (Cmi_charmrun_pid&DGRAM_MAGIC_MASK))
    {
      if (dg->rank == DGRAM_ACKNOWLEDGE)
	IntegrateAckDatagram(dg);
      else IntegrateMessageDatagram(dg);
    } else FreeExplicitDgram(dg);
  } else {
    MACHSTATE1(4,"  recv dgram failed (len=%d)",ok)
    FreeExplicitDgram(dg);
  }
}
*/




static void ServiceCharmrun_nolock() {
	int again = 1;
	MACHSTATE(2,"ServiceCharmrun_nolock begin {")
	while (again) {
		again = 0;
		CheckSocketsReady(0);
		if (ctrlskt_ready_read) { 	// FIXME: this is set in another call 
			ctrl_getone();
			again=1; 
		}
		if (CmiStdoutNeedsService())
			CmiStdoutService();
	}
	MACHSTATE(2,"} ServiceCharmrun_nolock end")
}

void processSendWC(struct ibv_wc *recvWC) {
	//nothing really
	// ibv_post_send() 
}
void processRecvWC(struct ibv_wc *recvWC,const int toBuffer) {
	//ibv_recv ...
}


static inline int pollCq(const int toBuffer,struct ibv_cq *cq) {
	int i;
	int ne;
	struct ibv_wc wc[WC_LIST_SIZE];

	MACHSTATE1(2,"pollCq %d (((",toBuffer);
	ne = ibv_poll_cq(cq,WC_LIST_SIZE,&wc[0]);
	
	if(ne != 0){
		MACHSTATE1(3,"pollCq ne %d",ne);
		if(ne<0)
			CmiAbort("ibv_poll_cq error");
	}
	
	for(i=0;i<ne;i++){
		CmiAssert(wc[i].status==IBV_WC_SUCCESS);

		switch(wc[i].opcode){
			case IBV_WC_SEND: //sending message
				processSendWC(&wc[i]);
				break;
			case IBV_WC_RECV: // recving message
				processRecvWC(&wc[i],toBuffer);
				break;
			default:
				CmiAbort("Wrong type of work completion object in cq");
				break;
		}
			
	}
	MACHSTATE1(2,"))) pollCq %d",toBuffer);
	return ne;

}

static inline  void CommunicationServer_lock(int toBuffer) {
        CmiCommLock();
        CommunicationServer_nolock(0);
        CmiCommUnlock();
}

static inline  void CommunicationServer_nolock(int toBuffer) {
/*
	if(_Cmi_numnodes <= 1){
		pollCmiDirectQ();
		return;
	} 
*/
	MACHSTATE(2,"CommServer_nolock{");
        
//	pollCmiDirectQ();	// FIXME: not sure what this does...
  
	pollCq(toBuffer,context->sendCq);
	pollCq(toBuffer,context->recvCq);

//	if(toBuffer == 0)
//		processAllBufferedMsgs();	// FIXME : I don't think we need buf'ed msgs

	MACHSTATE(2,"} CommServer_nolock ne");
}




static uint16_t getLocalLid(struct ibv_context *dev_context, int port){
	struct ibv_port_attr attr;
	if (ibv_query_port(dev_context, port, &attr))
		return 0;

	return attr.lid;
}


struct infiAddr* initinfiAddr(int node,int lid,int qpn,int psn) {
	struct infiAddr *addr=malloc(sizeof(struct infiAddr));

	addr->lid=lid;
	addr->qpn=qpn;
	addr->psn=psn;

	return addr;
}

struct infiOtherNodeData *initinfiData(int node,int lid,int qpn,int psn) {
//struct infiOtherNodeData *initInfiOtherNodeData(int node,int addr[3]){
        struct infiOtherNodeData *ret=malloc(sizeof(struct infiOtherNodeData));
	//set qp
	ret->qp.lid=lid;
	ret->qp.qpn=qpn;
	ret->qp.psn=psn;

        ret->nodeNo = node;
//        ret->state = INFI_HEADER_DATA; // FIXME: undefined

//        ret->qp = context->qp[node];
//      ret->totalTokens = tokensPerProcessor;
//      ret->tokensLeft = tokensPerProcessor;
//      ret->postedRecvs = tokensPerProcessor;
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
void CmiHandleImmediate();
static void CommunicationServer(int sleepTime, int where) {
/*  0: from smp thread
    1: from interrupt
    2: from worker thread
*/

	if(where==COMM_SERVER_FROM_INTERRUPT)
		return;
#if CMK_SMP
	if(where == COMM_SERVER_FROM_WORKER)
		return;
	if(where == COMM_SERVER_FROM_SMP) {
		ServiceCharmrun_nolock();
	}
        CommunicationServer_lock(0);
#else
	ServiceCharmrun_nolock(); 
	CommunicationServer_nolock(0);
#endif
}

static void sendBarrierMessage(int pe)
{
  char buf[32];
  OtherNode  node = nodes + pe;
  int retval = -1;
  while (retval == -1) {
     retval = sendto(dataskt, (char *)buf, 32, 0,
	 (struct sockaddr *)&(node->addr),
	 sizeof(struct sockaddr_in));
  }
}

static void recvBarrierMessage()
{
  char buf[32];
  int nreadable, ok, s;
  
  if (dataskt!=-1) {
        do {
        CMK_PIPE_DECL(10);
	CMK_PIPE_ADDREAD(dataskt);
          nreadable=CMK_PIPE_CALL();
          if (nreadable == 0) continue;
          s = CMK_PIPE_CHECKREAD(dataskt);
          if (s) break;
        } while (1);
        ok = recv(dataskt,buf,32,0);
        CmiAssert(ok >= 0);
  }
}

/* happen at node level */
/* must be called on every PE including communication processors */
int CmiBarrier()
{
  int len, size, i;
  int status;
  int count = 0;
  OtherNode node;
  int numnodes = CmiNumNodes();

  if (Cmi_netpoll == 0) return -1;

  if (CmiMyRank() == 0) {
    /* every one send to pe 0 */
    if (CmiMyNode() != 0) {
      sendBarrierMessage(0);
    }
    if (CmiMyNode() == 0)  {
      for (count = 1; count < numnodes; count ++) {
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
    if (CmiMyNode() != 0) {
      recvBarrierMessage();
      for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int p = CmiMyNode();
        p = BROADCAST_SPANNING_FACTOR*p + i;
        if (p > numnodes - 1) break;
        p = p%numnodes;
        /* printf("[%d] RELAY => %d \n", CmiMyPe(), p);  */
        sendBarrierMessage(p);
      }
    }
  }
  CmiNodeAllBarrier();
  /* printf("[%d] OUT of barrier \n", CmiMyPe()); */
  return 0;
}


int CmiBarrierZero()
{
  int i;

  if (Cmi_netpoll == 0) return -1;

  if (CmiMyRank() == 0) {
    if (CmiMyNode()) {
      sendBarrierMessage(0);
    } else {
      for (i=0; i<CmiNumNodes()-1; i++) {
        recvBarrierMessage();
      }
    }
  }
  CmiNodeAllBarrier();
  return 0;
}

void createqp(struct ibv_device *dev){
	context->localAddr.lid=getLocalLid(context->context,context->ibPort);

	context->sendCqSize = maxrecvbuffers+2;
	context->sendCq = ibv_create_cq(context->context,context->sendCqSize,NULL,NULL,0);
	CmiAssert(context->sendCq != NULL);
	MACHSTATE1(3,"sendCq created %p",context->sendCq);
	
	context->recvCqSize = maxrecvbuffers;
	context->recvCq = ibv_create_cq(context->context,context->recvCqSize,NULL,NULL,0);
	CmiAssert(context->recvCq != NULL);
	MACHSTATE2(3,"recvCq created %p %d",context->recvCq,context->recvCqSize);

	struct ibv_qp_init_attr initAttr = {
		.qp_type = IBV_QPT_RC,
		.send_cq = context->sendCq,
		.recv_cq = context->recvCq,
		.srq	 = context->srq,
		.sq_sig_all = 0,
		.qp_context = NULL,
			.cap     = {
			.max_send_wr  = maxrecvbuffers,
			.max_send_sge = 2,
		},
	};
	struct ibv_qp_attr attr;

	attr.qp_state        = IBV_QPS_INIT;
	attr.pkey_index      = 0;
	attr.port_num        = context->ibPort;
	attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

	context->qp = ibv_create_qp(context->pd,&initAttr);
	CmiAssert(context->qp != NULL);
	MACHSTATE1(3,"qp created %p",context->qp);
			
	ibv_modify_qp(context->qp, &attr,
		IBV_QP_STATE              |
		IBV_QP_PKEY_INDEX         |
		IBV_QP_PORT               |
		IBV_QP_ACCESS_FLAGS);		

	context->localAddr.qpn = context->qp->qp_num;
	context->localAddr.psn = lrand48() & 0xffffff;
	MACHSTATE3(4,"qp information (lid=%i qpn=%i psn=%i)\n",context->localAddr.lid,context->localAddr.qpn,context->localAddr.psn);
}


void CmiMachineInit(char **argv)
{
	struct ibv_device **devlist;
	struct ibv_device *dev;
	int i;
	int calcmaxsize;
	int packetsize;
	int lid;

	MACHSTATE(3,"CmiMachineInit {");
	MACHSTATE2(3,"_Cmi_numnodes %d CmiNumNodes() %d",_Cmi_numnodes,CmiNumNodes());
	MACHSTATE1(3,"CmiMyNodeSize() %d",CmiMyNodeSize());
	MACHSTATE1(3,"CmiMyNodeSize() %d",CmiMyNodeSize());

	mtu_size=1200;
	packetsize = mtu_size*4;
	Cmi_dgram_max_data=packetsize-sizeof(struct infiPacketHeader);
	CmiAssert(Cmi_dgram_max_data>1);
	
	calcmaxsize=8000;

	maxrecvbuffers=calcmaxsize;
	maxtokens = calcmaxsize;
	
	ibud.devlist = ibv_get_device_list(NULL);
	CmiAssert(ibud.devlist != NULL);

	dev = *(ibud.devlist);
	CmiAssert(dev != NULL);

	MACHSTATE1(3,"device name %s",ibv_get_device_name(dev));

	context = (struct infiContext *)malloc(sizeof(struct infiContext));
	
	MACHSTATE1(3,"context allocated %p",context);
	
	context->ibPort = 1;
	context->context = ibv_open_device(dev);  //the context for this infiniband device 
	CmiAssert(context->context != NULL);
	
	MACHSTATE1(3,"device opened %p",context->context);

	context->pd = ibv_alloc_pd(context->context); //protection domain
	CmiAssert(context->pd != NULL);

	context->header.nodeNo = _Cmi_mynode;

	context->buffer=malloc(sizeof(int)*maxrecvbuffers);

	context->mr=ibv_reg_mr(context->pd, context->buffer, sizeof(int)*maxrecvbuffers, IBV_ACCESS_LOCAL_WRITE);

// FIXME: move createqp into the if statement?
//	if(_Cmi_numnodes>1) {
		createqp(dev);
//	}

	MACHSTATE(3,"} CmiMachineInit");
}

void CmiCommunicationInit(char **argv) {
}

void CmiMachineExit()
{
	ibv_dereg_mr(context->mr);
	free(context->buffer);
	ibv_destroy_qp(context->qp);
	ibv_dealloc_pd(context->pd); 
	ibv_close_device(context->context);
	ibv_free_device_list(ibud.devlist);
}

