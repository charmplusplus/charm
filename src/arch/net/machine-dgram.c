/*
  Datagram implementation of Converse NET version

  moved from machine.c by 
  Orion Sky Lawlor, olawlor@acm.org, 7/25/2001
*/

#define DGRAM_HEADER_SIZE 8

#define CmiMsgHeaderSetLength(msg, len) (((int*)(msg))[2] = (len))
#define CmiMsgHeaderGetLength(msg)      (((int*)(msg))[2])
#define CmiMsgNext(msg) (*((void**)(msg)))

#define DGRAM_SRCPE_MASK    (0xFFFF)
#define DGRAM_MAGIC_MASK    (0xFF)
#define DGRAM_SEQNO_MASK    (0xFFFFFFFFu)

#if CMK_NODE_QUEUE_AVAILABLE
#define DGRAM_NODEMESSAGE   (0xFB)
#endif
#define DGRAM_DSTRANK_MAX   (0xFC)
#define DGRAM_SIMPLEKILL    (0xFD)
#define DGRAM_BROADCAST     (0xFE)
#define DGRAM_ACKNOWLEDGE   (0xFF)

typedef struct { char data[DGRAM_HEADER_SIZE]; } DgramHeader;

/* the window size needs to be Cmi_window_size + sizeof(unsigned int) bytes) */
typedef struct { DgramHeader head; char window[1024]; } DgramAck;

#define DgramHeaderMake(ptr, dstrank, srcpe, magic, seqno) { \
   ((unsigned short *)ptr)[0] = srcpe; \
   ((unsigned short *)ptr)[1] = ((magic & DGRAM_MAGIC_MASK)<<8) | dstrank; \
   ((unsigned int *)ptr)[1] = seqno; \
}

#define DgramHeaderBreak(ptr, dstrank, srcpe, magic, seqno) { \
   unsigned short tmp; \
   srcpe = ((unsigned short *)ptr)[0]; \
   tmp = ((unsigned short *)ptr)[1]; \
   dstrank = (tmp&0xFF); magic = (tmp>>8); \
   seqno = ((unsigned int *)ptr)[1]; \
}

#define PE_BROADCAST_OTHERS (-1)
#define PE_BROADCAST_ALL    (-2)

#if CMK_NODE_QUEUE_AVAILABLE
#define NODE_BROADCAST_OTHERS (-1)
#define NODE_BROADCAST_ALL    (-2)
#endif

/********* Startup and Command-line args ********/
static int    Cmi_max_dgram_size;
static int    Cmi_os_buffer_size;
static int    Cmi_window_size;
static int    Cmi_half_window;
static double Cmi_delay_retransmit;
static double Cmi_ack_delay;
static int    Cmi_dgram_max_data;
static int    Cmi_comm_periodic_delay;
static int    Cmi_comm_clock_delay;
static int writeableAcks,writeableDgrams;/*Write-queue counts (to know when to sleep)*/

static void setspeed_atm()
{
  Cmi_max_dgram_size   = 2048;
  Cmi_os_buffer_size   = 50000;
  Cmi_window_size      = 20;
  Cmi_delay_retransmit = 0.0150;
  Cmi_ack_delay        = 0.0035;
}

static void setspeed_eth()
{
  Cmi_max_dgram_size   = 1400;
  Cmi_window_size      = 40;
  Cmi_os_buffer_size   = Cmi_window_size*Cmi_max_dgram_size;
  Cmi_delay_retransmit = 0.0400;
  Cmi_ack_delay        = 0.0050;
}

static void extract_args(char **argv)
{
  int ms;
  setspeed_eth();
  if (CmiGetArgFlag(argv,"+atm"))
    setspeed_atm();
  if (CmiGetArgFlag(argv,"+eth"))
    setspeed_eth();
  CmiGetArgInt(argv,"+max_dgram_size",&Cmi_max_dgram_size);
  CmiGetArgInt(argv,"+window_size",&Cmi_window_size);
  CmiGetArgInt(argv,"+os_buffer_size",&Cmi_os_buffer_size);
  if (CmiGetArgInt(argv,"+delay_retransmit",&ms))
	  Cmi_delay_retransmit=0.001*ms;
  if (CmiGetArgInt(argv,"+ack_delay",&ms))
	  Cmi_delay_retransmit=0.001*ms;
  extract_common_args(argv);
  Cmi_dgram_max_data = Cmi_max_dgram_size - DGRAM_HEADER_SIZE;
  Cmi_half_window = Cmi_window_size >> 1;
  if ((Cmi_window_size * Cmi_max_dgram_size) > Cmi_os_buffer_size)
    KillEveryone("Window size too big for OS buffer.");
  Cmi_comm_periodic_delay=(int)(1000*Cmi_delay_retransmit);
  if (Cmi_comm_periodic_delay>60) Cmi_comm_periodic_delay=60;
  Cmi_comm_clock_delay=(int)(1000*Cmi_ack_delay);
}

/* Compare seqnos using modular arithmetic */
static int seqno_in_window(unsigned int seqno,unsigned int winStart)
{
  return ((DGRAM_SEQNO_MASK&(seqno-winStart)) < Cmi_window_size);
}
static int seqno_lt(unsigned int seqA,unsigned int seqB)
{
  unsigned int del=seqB-seqA;
  return (del>0u) && (del<(DGRAM_SEQNO_MASK/2));
}
static int seqno_le(unsigned int seqA,unsigned int seqB)
{
  unsigned int del=seqB-seqA;
  return (del>=0u) && (del<(DGRAM_SEQNO_MASK/2));
}

/*****************************************************************************
 *
 * Communication Structures
 *
 *****************************************************************************/

typedef struct OutgoingMsgStruct
{
  struct OutgoingMsgStruct *next;
  int   src, dst;
  int   size;
  char *data;
  int   refcount;
  int   freemode;
}
*OutgoingMsg;

typedef struct ExplicitDgramStruct
{
  struct ExplicitDgramStruct *next;
  int  srcpe, rank, seqno;
  unsigned int len, dummy; /* dummy to fix bug in rs6k alignment */
  double data[1];
}
*ExplicitDgram;

typedef struct ImplicitDgramStruct
{
  struct ImplicitDgramStruct *next;
  struct OtherNodeStruct *dest;
  int srcpe, rank, seqno;
  char  *dataptr;
  int    datalen;
  OutgoingMsg ogm;
}
*ImplicitDgram;

typedef struct OtherNodeStruct
{
  int nodestart, nodesize;
  skt_ip_t IP;
  unsigned int dataport;
  struct sockaddr_in addr;

  unsigned int             send_last;    /* seqno of last dgram sent */
  ImplicitDgram           *send_window;  /* datagrams sent, not acked */
  ImplicitDgram            send_queue_h; /* head of send queue */
  ImplicitDgram            send_queue_t; /* tail of send queue */
  unsigned int             send_next;    /* next seqno to go into queue */
  unsigned int             send_good;    /* last acknowledged seqno */
  double                   send_primer;  /* time to send retransmit */
  unsigned int             send_ack_seqno; /* next ack seqno to send */
  int                      retransmit_leash; /*Maximum number of packets to retransmit*/

  int                      asm_rank;
  int                      asm_total;
  int                      asm_fill;
  char                    *asm_msg;
  
  int                      recv_ack_cnt; /* number of unacked dgrams */
  double                   recv_ack_time;/* time when ack should be sent */
  unsigned int             recv_expect;  /* next dgram to expect */
  ExplicitDgram           *recv_window;  /* Packets received, not integrated */
  int                      recv_winsz;   /* Number of packets in recv window */
  unsigned int             recv_next;    /* Seqno of first missing packet */
  unsigned int             recv_ack_seqno; /* last ack seqno received */

  unsigned int             stat_total_intr; /* Total Number of Interrupts */
  unsigned int             stat_proc_intr;  /* Processed Interrupts */
  unsigned int             stat_send_pkt;   /* number of packets sent */
  unsigned int             stat_resend_pkt; /* number of packets resent */
  unsigned int             stat_send_ack;   /* number of acks sent */
  unsigned int             stat_recv_pkt;   /* number of packets received */
  unsigned int             stat_recv_ack;   /* number of acks received */
  unsigned int             stat_ack_pkts;   /* packets acked */
  unsigned int             stat_consec_resend; /*Packets retransmitted since last ack*/ 

  int sent_msgs;
  int recd_msgs;
  int sent_bytes;
  int recd_bytes;
}
*OtherNode;

static void OtherNode_init(OtherNode node)
{
    int i;
    node->send_primer = 1.0e30; /*Don't retransmit until needed*/
    node->retransmit_leash = 1; /*Start with short leash*/
    node->send_last=0;
    node->send_window =
      (ImplicitDgram*)malloc(Cmi_window_size*sizeof(ImplicitDgram));
    for (i=0;i<Cmi_window_size;i++) node->send_window[i]=NULL;
    node->send_queue_h=node->send_queue_t=NULL;
    node->send_next=0;
    node->send_good=(unsigned int)(-1);
    node->send_ack_seqno=0;

    node->asm_rank=0;
    node->asm_total=0;
    node->asm_fill=0;
    node->asm_msg=0;
    
    node->recv_ack_cnt=0;
    node->recv_ack_time=1.0e30;
    node->recv_ack_seqno=0;
    node->recv_expect=0;
    node->recv_window =
      (ExplicitDgram*)malloc(Cmi_window_size*sizeof(ExplicitDgram));
    for (i=0;i<Cmi_window_size;i++) node->recv_window[i]=NULL;    
    node->recv_winsz=0;
    node->recv_next=0;

    node->stat_total_intr=0;
    node->stat_proc_intr=0;
    node->stat_send_pkt=0;
    node->stat_resend_pkt=0;
    node->stat_send_ack=0; 
    node->stat_recv_pkt=0;      
    node->stat_recv_ack=0;        
    node->stat_ack_pkts=0;

    node->sent_msgs = 0;
    node->recd_msgs = 0;
    node->sent_bytes = 0;
    node->recd_bytes = 0;
}

static OtherNode *nodes_by_pe;  /* OtherNodes indexed by processor number */
static OtherNode  nodes;        /* Indexed only by ``node number'' */

/* initnode node table reply format:
 +------------------------------------------------------- 
 | 4 bytes  |   Number of nodes n                       ^
 |          |   (big-endian binary integer)       4+12*n bytes
 +-------------------------------------------------     |
 ^  |        (one entry for each node)            ^     |
 |  | 4 bytes  |   Number of PEs for this node    |     |
 n  | 4 bytes  |   IP address of this node   12*n bytes |
 |  | 4 bytes  |   Data (UDP) port of this node   |     |
 v  |          |   (big-endian binary integers)   v     v
 ---+----------------------------------------------------
*/
static void node_addresses_store(ChMessage *msg)
{
  ChMessageInt_t *n32=(ChMessageInt_t *)msg->data;
  ChNodeinfo *d=(ChNodeinfo *)(n32+1);
  int nodestart;
  int i,j;
  Cmi_numnodes=ChMessageInt(n32[0]);
  
  if ((sizeof(ChMessageInt_t)+sizeof(ChNodeinfo)*Cmi_numnodes)
         !=(unsigned int)msg->len)
    {printf("Node table has inconsistent length!");machine_exit(1);}
  nodes = (OtherNode)malloc(Cmi_numnodes * sizeof(struct OtherNodeStruct));
  nodestart=0;
  for (i=0; i<Cmi_numnodes; i++) {
    nodes[i].nodestart = nodestart;
    nodes[i].nodesize  = ChMessageInt(d[i].nPE);
    nodes[i].IP=d[i].IP;
    if (i==Cmi_mynode) {
      Cmi_nodestart=nodes[i].nodestart;
      Cmi_mynodesize=nodes[i].nodesize;
      Cmi_self_IP=nodes[i].IP;
    }
    nodes[i].dataport = ChMessageInt(d[i].dataport);
    nodes[i].addr = skt_build_addr(nodes[i].IP,nodes[i].dataport);
    nodestart+=nodes[i].nodesize;
  }
  Cmi_numpes=nodestart;
  nodes_by_pe = (OtherNode*)malloc(Cmi_numpes * sizeof(OtherNode));
  _MEMCHECK(nodes_by_pe);
  for (i=0; i<Cmi_numnodes; i++) {
    OtherNode node = nodes + i;
    OtherNode_init(node);
    for (j=0; j<node->nodesize; j++)
      nodes_by_pe[j + node->nodestart] = node;
  }
}

/**
 * Printing Net Statistics -- milind
 */
static char statstr[10000];

void printNetStatistics(void)
{
  char tmpstr[1024];
  OtherNode myNode;
  int i;
  unsigned int send_pkt=0, resend_pkt=0, recv_pkt=0, send_ack=0;
  unsigned int recv_ack=0, ack_pkts=0;

  myNode = nodes+CmiMyNode();
  sprintf(tmpstr, "***********************************\n");
  strcpy(statstr, tmpstr);
  sprintf(tmpstr, "Net Statistics For Node %u\n", CmiMyNode());
  strcat(statstr, tmpstr);
  sprintf(tmpstr, "Interrupts: %u \tProcessed: %u\n",
                  myNode->stat_total_intr, myNode->stat_proc_intr);
  strcat(statstr, tmpstr);
  sprintf(tmpstr, "Total Msgs Sent: %u \tTotal Bytes Sent: %u\n",
                  myNode->sent_msgs, myNode->sent_bytes);
  strcat(statstr, tmpstr);
  sprintf(tmpstr, "Total Msgs Recv: %u \tTotal Bytes Recv: %u\n",
                  myNode->recd_msgs, myNode->recd_bytes);
  strcat(statstr, tmpstr);
  sprintf(tmpstr, "***********************************\n");
  strcat(statstr, tmpstr);
  sprintf(tmpstr, "[Num]\tSENDTO\tRESEND\tRECV\tACKSTO\tACKSFRM\tPKTACK\n");
  strcat(statstr,tmpstr);
  sprintf(tmpstr, "=====\t======\t======\t====\t======\t=======\t======\n");
  strcat(statstr,tmpstr);
  for(i=0;i<CmiNumNodes();i++) {
    OtherNode node = nodes+i;
    sprintf(tmpstr, "[%u]\t%u\t%u\t%u\t%u\t%u\t%u\n",
                     i, node->stat_send_pkt, node->stat_resend_pkt,
		     node->stat_recv_pkt, node->stat_send_ack,
		     node->stat_recv_ack, node->stat_ack_pkts);
    strcat(statstr, tmpstr);
    send_pkt += node->stat_send_pkt;
    recv_pkt += node->stat_recv_pkt;
    resend_pkt += node->stat_resend_pkt;
    send_ack += node->stat_send_ack;
    recv_ack += node->stat_recv_ack;
    ack_pkts += node->stat_ack_pkts;
  }
  sprintf(tmpstr, "[TOTAL]\t%u\t%u\t%u\t%u\t%u\t%u\n",
                     send_pkt, resend_pkt,
		     recv_pkt, send_ack,
		     recv_ack, ack_pkts);
  strcat(statstr, tmpstr);
  sprintf(tmpstr, "***********************************\n");
  strcat(statstr, tmpstr);
  CmiPrintf(statstr);
}


/************** free list management *****************/

static ExplicitDgram Cmi_freelist_explicit;
static ImplicitDgram Cmi_freelist_implicit;
/*static OutgoingMsg   Cmi_freelist_outgoing;*/

#define FreeImplicitDgram(dg) {\
  ImplicitDgram d=(dg);\
  d->next = Cmi_freelist_implicit;\
  Cmi_freelist_implicit = d;\
}

#define MallocImplicitDgram(dg) {\
  ImplicitDgram d = Cmi_freelist_implicit;\
  if (d==0) {d = ((ImplicitDgram)malloc(sizeof(struct ImplicitDgramStruct)));\
             _MEMCHECK(d);\
  } else Cmi_freelist_implicit = d->next;\
  dg = d;\
}

#define FreeExplicitDgram(dg) {\
  ExplicitDgram d=(dg);\
  d->next = Cmi_freelist_explicit;\
  Cmi_freelist_explicit = d;\
}

#define MallocExplicitDgram(dg) {\
  ExplicitDgram d = Cmi_freelist_explicit;\
  if (d==0) { d = ((ExplicitDgram)malloc \
		   (sizeof(struct ExplicitDgramStruct) + Cmi_max_dgram_size));\
              _MEMCHECK(d);\
  } else Cmi_freelist_explicit = d->next;\
  dg = d;\
}

/* Careful with these next two, need concurrency control */

#define FreeOutgoingMsg(m) (free(m))
#define MallocOutgoingMsg(m)\
    {(m=(OutgoingMsg)malloc(sizeof(struct OutgoingMsgStruct))); _MEMCHECK(m);}

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

static int ctrlskt_ready_read;
static int dataskt_ready_read;
static int dataskt_ready_write;

int CheckSocketsReady(int withDelayMs)
{   
  int nreadable;
#if !CMK_USE_POLL
  static fd_set rfds; 
  static fd_set wfds; 
  struct timeval tmo;
  
  MACHSTATE(1,"CheckSocketsReady {")
  FD_ZERO(&rfds);FD_ZERO(&wfds);
  if (Cmi_charmrun_fd!=-1)
  	FD_SET(Cmi_charmrun_fd, &rfds);
  if (dataskt!=-1) {
  	FD_SET(dataskt, &rfds);
  	if (writeableDgrams || writeableAcks)
  	  FD_SET(dataskt, &wfds); /*Outgoing queue is nonempty*/
  }
  tmo.tv_sec = 0;
  tmo.tv_usec = withDelayMs*1000;
  nreadable = select(FD_SETSIZE, &rfds, &wfds, NULL, &tmo);
#else
  struct pollfd fds[3]; 
  int n = 0;
  MACHSTATE(1,"CheckSocketsReady {")
  if (Cmi_charmrun_fd!=-1) {
    fds[n].fd = Cmi_charmrun_fd;
    fds[n].events = POLLIN;
    n++;
  }
  if (dataskt!=-1) {
    fds[n].fd = dataskt;
    fds[n].events = POLLIN;
    if (writeableDgrams || writeableAcks)  fds[n].events |= POLLOUT;
    n++;
  }
  nreadable = poll(fds, n, withDelayMs);
#endif
  ctrlskt_ready_read = 0;
  dataskt_ready_read = 0;
  dataskt_ready_write = 0;

  if (nreadable == 0) {
    MACHSTATE(1,"} CheckSocketsReady (nothing readable)")
    return nreadable;
  }
  if (nreadable==-1) {
    if (errno && errno!=EINTR)
      KillEveryone("Socket error in CheckSocketsReady!\n");
    MACHSTATE(2,"} CheckSocketsReady (INTERRUPTED!)")
    return CheckSocketsReady(0);
  }
#if !CMK_USE_POLL
  if (Cmi_charmrun_fd!=-1)
	ctrlskt_ready_read = (FD_ISSET(Cmi_charmrun_fd, &rfds));
  if (dataskt!=-1) {
  	dataskt_ready_read = (FD_ISSET(dataskt, &rfds));
	dataskt_ready_write = (FD_ISSET(dataskt, &wfds));
  }
#else
  if (dataskt!=-1) {
    n--;
    dataskt_ready_read = fds[n].revents & POLLIN;
    dataskt_ready_write = fds[n].revents & POLLOUT;
  }
  if (Cmi_charmrun_fd!=-1) {
    n--;
    ctrlskt_ready_read = fds[n].revents & POLLIN;
  }
#endif
  MACHSTATE(1,"} CheckSocketsReady")
  return nreadable;
}

/******************************************************************************
 *
 * Transmission Code
 *
 *****************************************************************************/

void GarbageCollectMsg(OutgoingMsg ogm)
{
  if (ogm->refcount == 0) {
    if (ogm->freemode == 'A') {
      ogm->freemode = 'X';
    } else {
      CmiFree(ogm->data);
      FreeOutgoingMsg(ogm);
    }
  }
}

void DiscardImplicitDgram(ImplicitDgram dg)
{
  OutgoingMsg ogm;
  ogm = dg->ogm;
  ogm->refcount--;
  GarbageCollectMsg(ogm);
  FreeImplicitDgram(dg);
}

/*
 Check the real-time clock and perform periodic tasks.
 Must be called with comm. lock held.
 */
static double Cmi_ack_last, Cmi_check_last;
static void CommunicationsClock(void)
{
  MACHSTATE(3,"CommunicationsClock");
  Cmi_clock = GetClock();
  if (Cmi_clock > Cmi_ack_last + 0.5*Cmi_ack_delay) {
    MACHSTATE(4,"CommunicationsClock timing out acks");    
    Cmi_ack_last=Cmi_clock;
    writeableAcks=1;
    writeableDgrams=1;
  }
  
  if (Cmi_clock > Cmi_check_last + Cmi_check_delay) {
    MACHSTATE(4,"CommunicationsClock pinging charmrun");       
    Cmi_check_last = Cmi_clock; 
    ctrl_sendone_nolock("ping",NULL,0,NULL,0); /*Charmrun may have died*/
  }
}

#if CMK_SHARED_VARS_UNAVAILABLE
static void CommunicationsClockCaller(void *ignored)
{
  CmiCommLock();
  CommunicationsClock();
  CmiCommUnlock();
  CcdCallFnAfter(CommunicationsClockCaller,NULL,Cmi_comm_clock_delay);  
}

static void CommunicationPeriodic(void) 
{ /*Poll on the communications server*/
  CommunicationServer(0);
}

static void CommunicationPeriodicCaller(void *ignored)
{
  CommunicationPeriodic();
  CcdCallFnAfter(CommunicationPeriodicCaller,NULL,Cmi_comm_periodic_delay);
}
#endif

#if !CMK_USE_GM

#include "machine-eth.c"

#else

#include "machine-gm.c"

#endif
