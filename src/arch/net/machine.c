/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/


/******************************************************************************
 *
 * THE DATAGRAM STREAM
 *
 * Messages are sent using UDP datagrams.  The sender allocates a
 * struct for each datagram to be sent.  These structs stick around
 * until slightly after the datagram is acknowledged.
 *
 * Datagrams are transmitted node-to-node (as opposed to pe-to-pe).
 * Each node has an OtherNode struct for every other node in the
 * system.  The OtherNode struct contains:
 *
 *   send_queue   (all datagram-structs not yet transmitted)
 *   send_window  (all datagram-structs transmitted but not ack'd)
 *
 * When an acknowledgement comes in, all packets in the send-window
 * are either marked as acknowledged or pushed back into the send
 * queue for retransmission.
 *
 * THE OUTGOING MESSAGE
 *
 * When you send or broadcast a message, the first thing the system
 * does is system creates an OutgoingMsg struct to represent the
 * operation.  The OutgoingMsg contains a very direct expression
 * of what you want to do:
 *
 * OutgoingMsg:
 *
 *   size      --- size of message in bytes
 *   data      --- pointer to the buffer containing the message
 *   src       --- processor which sent the message
 *   dst       --- destination processor (-1=broadcast, -2=broadcast all)
 *   freemode  --- see below.
 *   refcount  --- see below.
 *
 * The OutgoingMsg is kept around until the transmission is done, then
 * it is garbage collected --- the refcount and freemode fields are
 * to assist garbage collection.
 *
 * The freemode indicates which kind of buffer-management policy was
 * used (sync, async, or freeing).  The sync policy is handled
 * superficially by immediately converting sync sends into freeing
 * sends.  Thus, the freemode can either be 'A' (async) or 'F'
 * (freeing).  If the freemode is 'F', then garbage collection
 * involves freeing the data and the OutgoingMsg structure itself.  If
 * the freemode is 'A', then the only cleanup is to change the
 * freemode to 'X', a condition which is then detectable by
 * CmiAsyncMsgSent.  In this case, the actual freeing of the
 * OutgoingMsg is done by CmiReleaseCommHandle.
 *
 * When the transmission is initiated, the system computes how many
 * datagrams need to be sent, total.  This number is stored in the
 * refcount field.  Each time a datagram is delivered, the refcount
 * is decremented, when it reaches zero, cleanup is performed.  There
 * are two exceptions to this rule.  Exception 1: if the OutgoingMsg
 * is a send (not a broadcast) and can be performed with shared
 * memory, the entire datagram system is bypassed, the message is
 * simply delivered and freed, not using the refcount mechanism at
 * all.  Exception 2: If the message is a broadcast, then part of the
 * broadcast that can be done via shared memory is performed prior to
 * initiating the datagram/refcount system.
 *
 * DATAGRAM FORMATS AND MESSAGE FORMATS
 *
 * Datagrams have this format:
 *
 *   srcpe   (16 bits) --- source processor number.
 *   magic   ( 8 bits) --- magic number to make sure DG is good.
 *   dstrank ( 8 bits) --- destination processor rank.
 *   seqno   (32 bits) --- packet sequence number.
 *   data    (XX byte) --- user data.
 *
 * The only reason the srcpe is in there is because the receiver needs
 * to know which receive window to use.  The dstrank field is needed
 * because transmission is node-to-node.  Once the message is
 * assembled by the node, it must be delivered to the appropriate PE.
 * The dstrank field is used to encode certain special-case scenarios.
 * If the dstrank is DGRAM_BROADCAST, the transmission is a broadcast,
 * and should be delivered to all processors in the node.  If the dstrank
 * is DGRAM_ACKNOWLEDGE, the datagram is an acknowledgement datagram, in
 * which case the srcpe is the number of the acknowledger, the seqno is
 * always zero, and the user data is a list of the seqno's being
 * acknowledged.  There may be other dstrank codes for special functions.
 *
 * To send a message, one chops it up into datagrams and stores those
 * datagrams in a send-queue.  These outgoing datagrams aren't stored
 * in the explicit format shown above.  Instead, they are stored as
 * ImplicitDgrams, which contain the datagram header and a pointer to
 * the user data (which is in the user message buffer, which is in the
 * OutgoingMsg).  At transmission time these are combined together.

 * The combination of the datagram header with the user's data is
 * performed right in the user's message buffer.  Note that the
 * datagram header is exactly 64 bits.  One simply overwrites 64 bits
 * of the user's message with a datagram header, sends the datagram
 * straight from the user's message buffer, then restores the user's
 * buffer to its original state.  There is a small problem with the
 * first datagram of the message: one needs 64 bits of space to store
 * the datagram header.  To make sure this space is there, we added a
 * 64-bit unused space to the front of the Cmi message header.  In
 * addition to this, we also add 32 bits to the Cmi message header
 * to make room for a length-field, making it possible to identify
 * message boundaries.
 *
 * CONCURRENCY CONTROL
 *
 * This has changed recently.
 *
 * EFFICIENCY NOTES
 *
 * The sender-side does little copying.  The async and freeing send
 * routines do no copying at all.  The sync send routines copy the
 * message, then use the freeing-send routines.  The other alternative
 * is to not copy the message, and use the async send mechanism
 * combined with a blocking wait.  Blocking wait seems like a bad
 * idea, since it could take a VERY long time to get all those
 * datagrams out the door.
 *
 * The receiver side, unfortunately, must copy.  To avoid copying,
 * it would have to receive directly into a preallocated message buffer.
 * Unfortunately, this can't work: there's no way to know how much
 * memory to preallocate, and there's no way to know which datagram
 * is coming next.  Thus, we receive into fixed-size (large) datagram
 * buffers.  These are then inspected, and the messages extracted from
 * them.
 *
 * Note that we are allocating a large number of structs: OutgoingMsg's,
 * ImplicitDgrams, ExplicitDgrams.  By design, each of these structs
 * is a fixed-size structure.  Thus, we can do memory allocation by
 * simply keeping a linked-list of unused structs around.  The only
 * place where expensive memory allocation is performed is in the
 * sync routines.
 *
 * Since the datagrams from one node to another are fully ordered,
 * there is slightly more ordering than is needed: in theory, the
 * datagrams of one message don't need to be ordered relative to the
 * datagrams of another.  This was done to simplify the sequencing
 * mechanisms: implementing a fully-ordered stream is much simpler
 * than a partially-ordered one.  It also makes it possible to
 * modularize, layering the message transmitter on top of the
 * datagram-sequencer.  In other words, it was just easier this way.
 * Hopefully, this won't cause serious degradation: LAN's rarely get
 * datagrams out of order anyway.
 *
 * A potential efficiency problem is the lack of message-combining.
 * One datagram could conceivably contain several messages.  This
 * might be more efficient, it's not clear how much overhead is
 * involved in sending a short datagram.  Message-combining isn't
 * really ``integrated'' into the design of this software, but you
 * could fudge it as follows.  Whenever you pull a short datagram from
 * the send-queue, check the next one to see if it's also a short
 * datagram.  If so, pack them together into a ``combined'' datagram.
 * At the receive side, simply check for ``combined'' datagrams, and
 * treat them as if they were simply two datagrams.  This would
 * require extra copying.  I have no idea if this would be worthwhile.
 *
 *****************************************************************************/

/*****************************************************************************
 *
 * Include Files
 *
 ****************************************************************************/

/* About CMK_TRUECRASH:  
   When debugging Charm++/Converse, CmiAbort is your enemy.
   Uncommenting the define below will cause the program to crash where the 
   problem occurs instead of calling host_abort which lets the program 
   exit gracefully and lose all the debugging info... */

#include "converse.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h> /*<- was <varargs.h>*/
#include <string.h>

#include "conv-ccs.h"
#include "ccs-server.h"
#include "sockRoutines.h"

#if defined(_WIN32) && ! defined(__CYGWIN__)
/*For windows systems:*/
#  include <windows.h>
#  include <wincon.h>
#  include <sys/types.h>
#  include <sys/timeb.h>
#  define fdopen _fdopen
#  define SIGBUS -1  /*These signals don't exist in Win32*/
#  define SIGKILL -1
#  define SIGQUIT -1

#else /*UNIX*/
#  include <pwd.h>
#  include <unistd.h>
#  include <sys/file.h>
#endif

#define PRINTBUFSIZE 16384

static void CommunicationServer(int withDelayMs);
extern int CmemInsideMem();
extern void CmemCallWhenMemAvail();
static void ConverseRunPE(int everReturn);
void CmiYield();
void ConverseCommonExit(void);

/****************************************************************************
 *
 * Handling Errors
 *
 * Errors should be handled by printing a message on stderr and
 * calling exit(1).  Nothing should be sent to charmrun, no attempt at
 * communication should be made.  The other processes will notice the
 * abnormal termination and will deal with it.
 *
 * Rationale: if an error triggers an attempt to send a message,
 * the attempt to send a message is likely to trigger another error,
 * leading to an infinite loop and a process that spins instead of
 * shutting down.
 *
 *****************************************************************************/

static void charmrun_abort(const char*);

static void KillEveryone(const char *msg)
{
  charmrun_abort(msg);
  exit(1);
}

static void KillEveryoneCode(n)
int n;
{
  char _s[100];
  sprintf(_s, "[%d] Fatal error #%d\n", CmiMyPe(), n);
  charmrun_abort(_s);
  exit(1);
}

static void KillOnAllSigs(int sigNo)
{
  char _s[100];
  const char *sig=" received signal";
  if (sigNo==SIGSEGV) sig=": segmentation violation.\nDid you dereference a null pointer?";
  if (sigNo==SIGFPE) sig=": floating point exception.\nDid you divide by zero?";
  if (sigNo==SIGILL) sig=": illegal instruction";
  if (sigNo==SIGBUS) sig=": bus error";
  if (sigNo==SIGKILL) sig=": caught signal KILL";
  if (sigNo==SIGQUIT) sig=": caught signal QUIT";
  
  sprintf(_s, "ERROR> Node program on PE %d%s\n", CmiMyPe(),sig);
  charmrun_abort(_s);
  exit(1);
}

#if !defined(_WIN32) || defined(__CYGWIN__)
static void HandleUserSignals(int signum)
{
  int condnum = ((signum==SIGUSR1) ? CcdSIGUSR1 : CcdSIGUSR2);
  CcdRaiseCondition(condnum);
}
#endif

static void PerrorExit(const char *msg)
{
  perror(msg);
  exit(1);
}

static void KillOnSIGPIPE(int dummy)
{
  fprintf(stderr,"charmrun exited, terminating.\n");
  exit(0);
}


/*****************************************************************************
 *
 *     Utility routines for network machine interface.
*
 *****************************************************************************/

double GetClock()
{
#if defined(_WIN32) && !defined(__CYGWIN__)
  struct _timeb tv; 
  _ftime(&tv);
  return (tv.time * 1.0 + tv.millitm * 1.0E-3);
#else
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, NULL);
  if (ok<0) { perror("gettimeofday"); KillEveryoneCode(9343112); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
#endif
}

char *CopyMsg(char *msg, int len)
{
  char *copy = (char *)CmiAlloc(len);
  if (!copy)
      fprintf(stderr, "Out of memory\n");
  memcpy(copy, msg, len);
  return copy;
}

/*****************************************************************************
 *
 * Producer-Consumer Queues
 *
 * This queue implementation enables a producer and a consumer to
 * communicate via a queue.  The queues are optimized for this situation,
 * they don't require any operating system locks (they do require 32-bit
 * reads and writes to be atomic.)  Cautions: there can only be one
 * producer, and one consumer.  These queues cannot store null pointers.
 *
 ****************************************************************************/

#define PCQueueSize 0x100

typedef struct CircQueueStruct
{
  struct CircQueueStruct *next;
  int push;
  int pull;
  char *data[PCQueueSize];
}
*CircQueue;

typedef struct PCQueueStruct
{
  CircQueue head;
  CircQueue tail;
}
*PCQueue;

/* static CircQueue Cmi_freelist_circqueuestruct = 0;
   static int freeCount = 0; */

#define FreeCircQueueStruct(dg) {\
  CircQueue d;\
  CmiMemLock();\
  d=(dg);\
  d->next = Cmi_freelist_circqueuestruct;\
  Cmi_freelist_circqueuestruct = d;\
  freeCount++;\
  CmiMemUnlock();\
}

#define MallocCircQueueStruct(dg) {\
  CircQueue d;\
  CmiMemLock();\
  d = Cmi_freelist_circqueuestruct;\
  if (d==(CircQueue)0){\
    d = ((CircQueue)calloc(1, sizeof(struct CircQueueStruct)));\
  }\
  else{\
    freeCount--;\
    Cmi_freelist_circqueuestruct = d->next;\
    }\
  dg = d;\
  CmiMemUnlock();\
}


PCQueue PCQueueCreate()
{
  CircQueue circ;
  PCQueue Q;

  /* MallocCircQueueStruct(circ); */
  circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));

  Q = (PCQueue)malloc(sizeof(struct PCQueueStruct));
  _MEMCHECK(Q);
  Q->head = circ;
  Q->tail = circ;
  return Q;
}

int PCQueueEmpty(PCQueue Q)
{
  CircQueue circ = Q->head;
  char *data = circ->data[circ->pull];
  return (data == 0);
}

char *PCQueuePop(PCQueue Q)
{
  CircQueue circ; int pull; char *data;

    circ = Q->head;
    pull = circ->pull;
    data = circ->data[pull];
    if (data) {
      circ->pull = (pull + 1);
      circ->data[pull] = 0;
      if (pull == PCQueueSize - 1) { /* just pulled the data from the last slot
                                     of this buffer */
        Q->head = circ-> next; /* next buffer must exist, because "Push"  */
	
	/* FreeCircQueueStruct(circ); */
        free(circ);
	
	/* links in the next buffer *before* filling */
                               /* in the last slot. See below. */
      }
      return data;
    }
    else { /* queue seems to be empty. The producer may be adding something
              to it, but its ok to report queue is empty. */
      return 0;
    }
}

void PCQueuePush(PCQueue Q, char *data)
{
  CircQueue circ, circ1; int push;
  
  circ1 = Q->tail;
  push = circ1->push;
  if (push == (PCQueueSize -1)) { /* last slot is about to be filled */
    /* this way, the next buffer is linked in before data is filled in 
       in the last slot of this buffer */

    circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));
    /* MallocCircQueueStruct(circ); */

    Q->tail->next = circ;
    Q->tail = circ;
  }
  circ1->data[push] = data;
  circ1->push = (push + 1);
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message)
{
#if CMK_TRUECRASH
  CmiPrintf("%s", message);
  *(int *)NULL = 0; /*Write to null, causing bus error*/
#else
  charmrun_abort(message);
  exit(1);
#endif
}


/******************************************************************************
 *
 * CmiEnableAsyncIO
 *
 * The net and tcp versions use a bunch of unix processes talking to each
 * other via file descriptors.  We need for a signal SIGIO to be generated
 * each time a message arrives, making it possible to write a signal
 * handler to handle the messages.  The vast majority of unixes can,
 * in fact, do this.  However, there isn't any standard for how this is
 * supposed to be done, so each version of UNIX has a different set of
 * calls to turn this signal on.  So, there is like one version here for
 * every major brand of UNIX.
 *
 *****************************************************************************/

#if CMK_ASYNC_USE_FIOASYNC_AND_FIOSETOWN
#include <sys/filio.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  int pid = getpid();
  int async = 1;
  if ( ioctl(fd, FIOSETOWN, &pid) < 0  ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( ioctl(fd, FIOASYNC, &async) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

#if CMK_ASYNC_USE_FIOASYNC_AND_SIOCSPGRP
#include <sys/filio.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  int pid = -getpid();
  int async = 1;
  if ( ioctl(fd, SIOCSPGRP, &pid) < 0  ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( ioctl(fd, FIOASYNC, &async) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

#if CMK_ASYNC_USE_FIOSSAIOSTAT_AND_FIOSSAIOOWN
#include <sys/ioctl.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  int pid = getpid();
  int async = 1;
  if ( ioctl(fd, FIOSSAIOOWN, &pid) < 0  ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( ioctl(fd, FIOSSAIOSTAT, &async) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

#if CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN
#include <fcntl.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  if ( fcntl(fd, F_SETOWN, getpid()) < 0 ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( fcntl(fd, F_SETFL, FASYNC) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

/*****************************************************************************
 *
 * Communication Structures
 *
 *****************************************************************************/

#define DGRAM_HEADER_SIZE 8

#define CmiMsgHeaderSetLength(msg, len) (((int*)(msg))[2] = (len))
#define CmiMsgHeaderGetLength(msg)      (((int*)(msg))[2])
#define CmiMsgNext(msg) (*((void**)(msg)))

#define DGRAM_SRCPE_MASK    (0xFFFF)
#define DGRAM_MAGIC_MASK    (0xFF)
#define DGRAM_SEQNO_MASK    (0xFFFFFFFF)

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
  unsigned int IP, dataport;
  struct sockaddr_in addr;

  double                   send_primer;  /* time to send primer packet */
  unsigned int             send_last;    /* seqno of last dgram sent */
  ImplicitDgram           *send_window;  /* datagrams sent, not acked */
  ImplicitDgram            send_queue_h; /* head of send queue */
  ImplicitDgram            send_queue_t; /* tail of send queue */
  unsigned int             send_next;    /* next seqno to go into queue */
  unsigned int             send_ack_seqno; /* next ack seqno to send */

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
 
  int sent_msgs;
  int recd_msgs;
  int sent_bytes;
  int recd_bytes;
}
*OtherNode;

static OtherNode *nodes_by_pe;  /* OtherNodes indexed by processor number */
static OtherNode  nodes;        /* Indexed only by ``node number'' */

typedef struct CmiStateStruct
{
  int pe, rank;
  PCQueue recv;
  void *localqueue;
}
*CmiState;

#if CMK_NODE_QUEUE_AVAILABLE
CsvStaticDeclare(CmiNodeLock, CmiNodeRecvLock);
CsvStaticDeclare(PCQueue, NodeRecv);
#endif

void CmiStateInit(int pe, int rank, CmiState state)
{
  state->pe = pe;
  state->rank = rank;
  state->recv = PCQueueCreate();
  state->localqueue = CdsFifo_Create();
#if CMK_NODE_QUEUE_AVAILABLE
  CsvInitialize(CmiNodeLock, CmiNodeRecvLock);
  CsvInitialize(PCQueue, NodeRecv);
  if (rank==0) {
    CsvAccess(CmiNodeRecvLock) = CmiCreateLock();
    CsvAccess(NodeRecv) = PCQueueCreate();
  }
#endif
}

static ExplicitDgram Cmi_freelist_explicit;
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

/******************************************************************************
 *
 * Configuration Data
 *
 * This data is all read in from the NETSTART variable (provided by the
 * charmrun) and from the command-line arguments.  Once read in, it is never
 * modified.
 *
 *****************************************************************************/

int               Cmi_mynode;    /* Which address space am I */
int               Cmi_mynodesize;/* Number of processors in my address space */
int               Cmi_numnodes;  /* Total number of address spaces */
int               Cmi_numpes;    /* Total number of processors */
static int        Cmi_nodestart; /* First processor in this address space */
static int        Cmi_self_IP;
static int        Cmi_charmrun_IP; /*Address of charmrun machine*/
static int        Cmi_charmrun_port;
static int        Cmi_charmrun_pid;
static int        Cmi_charmrun_fd;
static int    Cmi_max_dgram_size;
static int    Cmi_os_buffer_size;
static int    Cmi_window_size;
static int    Cmi_half_window;
static double Cmi_delay_retransmit;
static double Cmi_ack_delay;
static int    Cmi_dgram_max_data;
static int    Cmi_tickspeed;
static int    Cmi_netpoll;
static int    Cmi_syncprint;

static int Cmi_print_stats = 0;

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

static ImplicitDgram Cmi_freelist_implicit;
static void setspeed_atm()
{
  Cmi_max_dgram_size   = 2048;
  Cmi_os_buffer_size   = 50000;
  Cmi_window_size      = 20;
  Cmi_delay_retransmit = 0.0150;
  Cmi_ack_delay        = 0.0035;
  Cmi_tickspeed        = 10000;
}

static void setspeed_eth()
{
  Cmi_max_dgram_size   = 2048;
  Cmi_os_buffer_size   = 50000;
  Cmi_window_size      = 20;
  Cmi_delay_retransmit = 0.0400;
  Cmi_ack_delay        = 0.0100;
  Cmi_tickspeed        = 10000;
}

static void parse_netstart()
{
  char *ns;
  int nread;
  ns = getenv("NETSTART");
  if (ns!=0) 
  {/*Read values set by Charmrun*/
  	nread = sscanf(ns, "%d%d%d%d",
		 &Cmi_mynode,
		 &Cmi_charmrun_IP, &Cmi_charmrun_port, 
		 &Cmi_charmrun_pid);

  	if (nread!=4) {
  		fprintf(stderr,"Error parsing NETSTART '%s'\n",ns);
  		exit(1);
  	}
  } else 
  {/*No charmrun-- set flag values for standalone operation*/
  	Cmi_mynode=0;
  	Cmi_charmrun_IP=0;
  	Cmi_charmrun_port=0;
  	Cmi_charmrun_pid=0;
  }
}

static void extract_args(argv)
char **argv;
{
  setspeed_eth();
  if (CmiGetArgFlag(argv,"+atm"))
    setspeed_atm();
  if (CmiGetArgFlag(argv,"+eth"))
    setspeed_eth();
  if (CmiGetArgFlag(argv,"+stats"))
    Cmi_print_stats = 1;
  Cmi_dgram_max_data = Cmi_max_dgram_size - DGRAM_HEADER_SIZE;
  Cmi_half_window = Cmi_window_size >> 1;
  if ((Cmi_window_size * Cmi_max_dgram_size) > Cmi_os_buffer_size)
    KillEveryone("Window size too big for OS buffer.");
}

/******************************************************************************
 *
 * Packet Performance Logging
 *
 * This module is designed to give a detailed log of the packets and their
 * acknowledgements, for performance tuning.  It can be disabled.
 *
 *****************************************************************************/

#define LOGGING 0

#if LOGGING

typedef struct logent {
  double time;
  int seqno;
  int srcpe;
  int dstpe;
  int kind;
} *logent;


logent log;
int    log_pos;
int    log_wrap;

static void log_init()
{
  log = (logent)malloc(50000 * sizeof(struct logent));
  _MEMCHECK(log);
  log_pos = 0;
  log_wrap = 0;
}

static void log_done()
{
  char logname[100]; FILE *f; int i, size;
  sprintf(logname, "log.%d", Cmi_mynode);
  f = fopen(logname, "w");
  if (f==0) KillEveryone("fopen problem");
  if (log_wrap) size = 50000; else size=log_pos;
  for (i=0; i<size; i++) {
    logent ent = log+i;
    fprintf(f, "%1.4f %d %c %d %d\n",
	    ent->time, ent->srcpe, ent->kind, ent->dstpe, ent->seqno);
  }
  fclose(f);
}

void printLog(void)
{
  char logname[100]; FILE *f; int i, j, size;
  static int logged = 0;
  if (logged)
      return;
  logged = 1;
  CmiPrintf("Logging: %d\n", Cmi_mynode);
  sprintf(logname, "log.%d", Cmi_mynode);
  f = fopen(logname, "w");
  if (f==0) KillEveryone("fopen problem");
  for (i = 5000; i; i--)
  {
  /*for (i=0; i<size; i++) */
    j = log_pos - i;
    if (j < 0)
    {
        if (log_wrap)
	    j = 5000 + j;
	else
	    j = 0;
    };
    {
    logent ent = log+j;
    fprintf(f, "%1.4f %d %c %d %d\n",
	    ent->time, ent->srcpe, ent->kind, ent->dstpe, ent->seqno);
    }
  }
  fclose(f);
  CmiPrintf("Done Logging: %d\n", Cmi_mynode);
}

#define LOG(t,s,k,d,q) { if (log_pos==50000) { log_pos=0; log_wrap=1;} { logent ent=log+log_pos; ent->time=t; ent->srcpe=s; ent->kind=k; ent->dstpe=d; ent->seqno=q; log_pos++; }}

#endif

#if !LOGGING

#define log_init() /*empty*/
#define log_done() /*empty*/
#define printLog() /*empty*/
#define LOG(t,s,k,d,q) /*empty*/

#endif

/******************************************************************************
 *
 * Node state
 *
 *****************************************************************************/

static unsigned int dataport=0;
static SOCKET       dataskt;

static CmiNodeLock    Cmi_scanf_mutex;
static double         Cmi_clock;
static double         Cmi_check_last = 0.0;
static double         Cmi_check_delay = 3.0;

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
  static fd_set rfds; 
  static fd_set wfds; 
  struct timeval tmo;
  int nreadable;
  
  if (Cmi_charmrun_fd!=-1)
  	FD_SET(Cmi_charmrun_fd, &rfds);
  if (dataskt!=-1) {
  	FD_SET(dataskt, &rfds);
  	FD_SET(dataskt, &wfds);
  }
  tmo.tv_sec = 0;
  tmo.tv_usec = withDelayMs*1000;
  nreadable = select(FD_SETSIZE, &rfds, &wfds, NULL, &tmo);
  if (nreadable <= 0) {
    ctrlskt_ready_read = 0;
    dataskt_ready_read = 0;
    dataskt_ready_write = 0;
    return nreadable;
  }
  if (Cmi_charmrun_fd!=-1)
	ctrlskt_ready_read = (FD_ISSET(Cmi_charmrun_fd, &rfds));
  if (dataskt!=-1) {
  	dataskt_ready_read = (FD_ISSET(dataskt, &rfds));
  	dataskt_ready_write = (FD_ISSET(dataskt, &wfds));
  }
  return nreadable;
}
/******************************************************************************
 *
 * OS Threads
 *
 * This version of converse is for multiple-processor workstations,
 * and we assume that the OS provides threads to gain access to those
 * multiple processors.  This section contains an interface layer for
 * the OS specific threads package.  It contains routines to start
 * the threads, routines to access their thread-specific state, and
 * routines to control mutual exclusion between them.
 *
 * In addition, we wish to support nonthreaded operation.  To do this,
 * we provide a version of these functions that uses the main/only thread
 * as a single PE, and simulates a communication thread using interrupts.
 *
 *
 * CmiStartThreads()
 *
 *    Allocates one CmiState structure per PE.  Initializes all of
 *    the CmiState structures using the function CmiStateInit.
 *    Starts processor threads 1..N (not 0, that's the one
 *    that calls CmiStartThreads), as well as the communication
 *    thread.  Each processor thread (other than 0) must call ConverseInitPE
 *    followed by Cmi_startfn.  The communication thread must be an infinite
 *    loop that calls the function CommunicationServer over and over.
 *
 * CmiGetState()
 *
 *    When called by a PE-thread, returns the processor-specific state
 *    structure for that PE.
 *
 * CmiGetStateN(int n)
 *
 *    returns processor-specific state structure for the PE of rank n.
 *
 * CmiMemLock() and CmiMemUnlock()
 *
 *    The memory module calls these functions to obtain mutual exclusion
 *    in the memory routines, and to keep interrupts from reentering malloc.
 *
 * CmiCommLock() and CmiCommUnlock()
 *
 *    These functions lock a mutex that insures mutual exclusion in the
 *    communication routines.
 *
 * CmiMyPe() and CmiMyRank()
 *
 *    The usual.  Implemented here, since a highly-optimized version
 *    is possible in the nonthreaded case.
 *

  
  FIXME: There is horrible duplication of code (e.g. locking code)
   both here and in converse.h.  It could be much shorter.  OSL 9/9/2000

 *****************************************************************************/

#if CMK_SHARED_VARS_NT_THREADS

static HANDLE memmutex;
void CmiMemLock() { WaitForSingleObject(memmutex, INFINITE); }
void CmiMemUnlock() { ReleaseMutex(memmutex); }

static DWORD Cmi_state_key = 0xFFFFFFFF;
static CmiState     Cmi_state_vector = 0;

CmiState CmiGetState()
{
    CmiState result = 0;

  if(Cmi_state_key == 0xFFFFFFFF) return 0;
  
  result = (CmiState)TlsGetValue(Cmi_state_key);
  if(result == 0) PerrorExit("CmiGetState: TlsGetValue");
  return result;
}

int CmiMyPe()
{  
  CmiState result = 0;

  if(Cmi_state_key == 0xFFFFFFFF) return -1;
  result = (CmiState)TlsGetValue(Cmi_state_key);

  if(result == 0) PerrorExit("CmiMyPe: TlsGetValue");

  return result->pe;
}

int CmiMyRank()
{
  CmiState result = 0;

  if(Cmi_state_key == 0xFFFFFFFF) return 0;
  result = (CmiState)TlsGetValue(Cmi_state_key);
  if(result == 0) PerrorExit("CmiMyRank: TlsGetValue");
  return result->rank;
}

int CmiNodeFirst(int node) { return nodes[node].nodestart; }
int CmiNodeSize(int node)  { return nodes[node].nodesize; }
int CmiNodeOf(int pe)      { return (nodes_by_pe[pe] - nodes); }
int CmiRankOf(int pe)      { return pe - (nodes_by_pe[pe]->nodestart); }

CmiNodeLock CmiCreateLock()
{
  HANDLE hMutex = CreateMutex(NULL, FALSE, NULL);
  return hMutex;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  CloseHandle(lk);
}

void CmiYield() 
{ 
  Sleep(0);
}

#define CmiGetStateN(n) (Cmi_state_vector+(n))

static HANDLE comm_mutex;
#define CmiCommLock() (WaitForSingleObject(comm_mutex, INFINITE))
#define CmiCommUnlock() (ReleaseMutex(comm_mutex))

static DWORD WINAPI comm_thread(LPVOID dummy)
{  
  while (1) CommunicationServer(500);
  return 0;/*should never get here*/
}

static DWORD WINAPI call_startfn(LPVOID vindex)
{
  int index = (int)vindex;
 
  CmiState state = Cmi_state_vector + index;
  if(Cmi_state_key == 0xFFFFFFFF) PerrorExit("TlsAlloc");
  if(TlsSetValue(Cmi_state_key, (LPVOID)state) == 0) PerrorExit("TlsSetValue");

  ConverseRunPE(0);
  return 0;
}


/*Classic sense-reversing barrier algorithm.
FIXME: This should be the barrier implementation for 
all thread types.
*/
static HANDLE barrier_mutex;
static volatile int    barrier_wait[2] = {0,0};
static volatile int    barrier_which = 0;

void  CmiNodeBarrier(void)
{
  int doWait = 1;
  int which;

  WaitForSingleObject(barrier_mutex, INFINITE);
  which=barrier_which;
  barrier_wait[which]++;
  if (barrier_wait[which] == Cmi_mynodesize) {
    barrier_which = !which;
    barrier_wait[barrier_which] = 0;/*Reset new counter*/
    doWait = 0;
  }
  ReleaseMutex(barrier_mutex);

  if (doWait)
      while(barrier_wait[which] != Cmi_mynodesize)
		  sleep(0);/*<- could also just spin here*/
}

static void CmiStartThreads()
{
  int     i;
  DWORD   threadID;
  HANDLE  thr;
  int     val = 0;

  comm_mutex = CmiCreateLock();
  memmutex = CmiCreateLock();
  barrier_mutex = CmiCreateLock();

  Cmi_state_key = TlsAlloc();
  if(Cmi_state_key == 0xFFFFFFFF) PerrorExit("TlsAlloc main");
  
  Cmi_state_vector =
    (CmiState)calloc(Cmi_mynodesize, sizeof(struct CmiStateStruct));
  
  for (i=0; i<Cmi_mynodesize; i++)
    CmiStateInit(i+Cmi_nodestart, i, CmiGetStateN(i));
  
  for (i=1; i<Cmi_mynodesize; i++) {
    if((thr = CreateThread(NULL, 0, call_startfn, (LPVOID)i, 0, &threadID)) 
       == NULL) PerrorExit("CreateThread");
    CloseHandle(thr);
  }
  
  if(TlsSetValue(Cmi_state_key, (LPVOID)Cmi_state_vector) == 0) 
    PerrorExit("TlsSetValue");
  
  if((thr = CreateThread(NULL, 0, comm_thread, 0, 0, &threadID)) == NULL) 
     PerrorExit("CreateThread");
  CloseHandle(thr);
}

#endif

#if CMK_SHARED_VARS_SUN_THREADS

static mutex_t memmutex;
void CmiMemLock() { mutex_lock(&memmutex); }
void CmiMemUnlock() { mutex_unlock(&memmutex); }

static thread_key_t Cmi_state_key;
static CmiState     Cmi_state_vector;

CmiState CmiGetState()
{
  CmiState result = 0;
  thr_getspecific(Cmi_state_key, (void **)&result);
  return result;
}

int CmiMyPe()
{
  CmiState result = 0;
  thr_getspecific(Cmi_state_key, (void **)&result);
  return result->pe;
}

int CmiMyRank()
{
  CmiState result = 0;
  thr_getspecific(Cmi_state_key, (void **)&result);
  return result->rank;
}

int CmiNodeFirst(int node) { return nodes[node].nodestart; }
int CmiNodeSize(int node)  { return nodes[node].nodesize; }
int CmiNodeOf(int pe)      { return (nodes_by_pe[pe] - nodes); }
int CmiRankOf(int pe)      { return pe - (nodes_by_pe[pe]->nodestart); }

CmiNodeLock CmiCreateLock()
{
  CmiNodeLock lk = (CmiNodeLock)malloc(sizeof(mutex_t));
  _MEMCHECK(lk);
  mutex_init(lk,0,0);
  return lk;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  mutex_destroy(lk);
  free(lk);
}

void CmiYield() { thr_yield(); }

int barrier = 0;
cond_t barrier_cond = DEFAULTCV;
mutex_t barrier_mutex = DEFAULTMUTEX;

void CmiNodeBarrier(void)
{
  mutex_lock(&barrier_mutex);
  barrier++;
  if(barrier != CmiMyNodeSize())
    cond_wait(&barrier_cond, &barrier_mutex);
  else{
    barrier = 0;
    cond_broadcast(&barrier_cond);
  }
  mutex_unlock(&barrier_mutex);
}

#define CmiGetStateN(n) (Cmi_state_vector+(n))

static mutex_t comm_mutex;
#define CmiCommLock() (mutex_lock(&comm_mutex))
#define CmiCommUnlock() (mutex_unlock(&comm_mutex))

static void comm_thread(void)
{
  while (1) CommunicationServer(500);
}

static void *call_startfn(void *vindex)
{
  int index = (int)vindex;
  CmiState state = Cmi_state_vector + index;
  thr_setspecific(Cmi_state_key, state);
  ConverseRunPE(0);
}

static void CmiStartThreads()
{
  int i, ok;
  unsigned int pid;
  
  thr_setconcurrency(Cmi_mynodesize);
  thr_keycreate(&Cmi_state_key, 0);
  Cmi_state_vector =
    (CmiState)calloc(Cmi_mynodesize, sizeof(struct CmiStateStruct));
  for (i=0; i<Cmi_mynodesize; i++)
    CmiStateInit(i+Cmi_nodestart, i, CmiGetStateN(i));
  for (i=1; i<Cmi_mynodesize; i++) {
    ok = thr_create(0, 256000, call_startfn, (void *)i, THR_DETACHED|THR_BOUND, &pid);
    if (ok<0) PerrorExit("thr_create");
  }
  thr_setspecific(Cmi_state_key, Cmi_state_vector);
  ok = thr_create(0, 256000, (void *(*)(void *))comm_thread, 0, 0, &pid);
  if (ok<0) PerrorExit("thr_create comm");
}

#endif

#if CMK_SHARED_VARS_POSIX_THREADS_SMP

static pthread_mutex_t memmutex;
void CmiMemLock() { pthread_mutex_lock(&memmutex); }
void CmiMemUnlock() { pthread_mutex_unlock(&memmutex); }

static pthread_key_t Cmi_state_key;
static CmiState     Cmi_state_vector;

CmiState CmiGetState()
{
  CmiState result = 0;
  result = pthread_getspecific(Cmi_state_key);
  return result;
}

int CmiMyPe()
{
  CmiState result = 0;
  result = pthread_getspecific(Cmi_state_key);
  return result->pe;
}

int CmiMyRank()
{
  CmiState result = 0;
  result = pthread_getspecific(Cmi_state_key);
  return result->rank;
}

int CmiNodeFirst(int node) { return nodes[node].nodestart; }
int CmiNodeSize(int node)  { return nodes[node].nodesize; }
int CmiNodeOf(int pe)      { return (nodes_by_pe[pe] - nodes); }
int CmiRankOf(int pe)      { return pe - (nodes_by_pe[pe]->nodestart); }

CmiNodeLock CmiCreateLock()
{
  CmiNodeLock lk = (CmiNodeLock)malloc(sizeof(pthread_mutex_t));
  _MEMCHECK(lk);
  pthread_mutex_init(lk,(pthread_mutexattr_t *)0);
  return lk;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  pthread_mutex_destroy(lk);
  free(lk);
}

void CmiYield() { sched_yield(); }

int barrier;
pthread_cond_t barrier_cond;
pthread_mutex_t barrier_mutex;

void CmiNodeBarrier(void)
{
  pthread_mutex_lock(&barrier_mutex);
  barrier++;
  if(barrier != CmiMyNodeSize())
    pthread_cond_wait(&barrier_cond, &barrier_mutex);
  else{
    barrier = 0;
    pthread_cond_broadcast(&barrier_cond);
  }
  pthread_mutex_unlock(&barrier_mutex);
}


#define CmiGetStateN(n) (Cmi_state_vector+(n))

static pthread_mutex_t comm_mutex;
#define CmiCommLock() (pthread_mutex_lock(&comm_mutex))
#define CmiCommUnlock() (pthread_mutex_unlock(&comm_mutex))

static void comm_thread(void)
{
  while (1) CommunicationServer(500);
}

static void *call_startfn(void *vindex)
{
  int index = (int)vindex;
  CmiState state = Cmi_state_vector + index;
  pthread_setspecific(Cmi_state_key, state);
  ConverseRunPE(0);
  return 0;
}

static void CmiStartThreads()
{
  pthread_t pid;
  int i, ok;
  pthread_attr_t attr;
  
  pthread_key_create(&Cmi_state_key, 0);
  Cmi_state_vector =
    (CmiState)calloc(Cmi_mynodesize, sizeof(struct CmiStateStruct));
  for (i=0; i<Cmi_mynodesize; i++)
    CmiStateInit(i+Cmi_nodestart, i, CmiGetStateN(i));
  for (i=1; i<Cmi_mynodesize; i++) {
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    ok = pthread_create(&pid, &attr, call_startfn, (void *)i);
    if (ok<0) PerrorExit("pthread_create"); 
    pthread_attr_destroy(&attr);
  }
  pthread_setspecific(Cmi_state_key, Cmi_state_vector);
  ok = pthread_create(&pid, NULL, (void *(*)(void *))comm_thread, 0);
  if (ok<0) PerrorExit("pthread_create comm"); 
}

#endif

#if CMK_SHARED_VARS_UNAVAILABLE

static volatile int memflag;
void CmiMemLock() { memflag=1; }
void CmiMemUnlock() { memflag=0; }

static struct CmiStateStruct Cmi_state;
int Cmi_mype;
int Cmi_myrank;
#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

static int comm_flag;
#define CmiCommLock() (comm_flag=1)
#define CmiCommUnlock() (comm_flag=0)

void CmiYield() { sleep(0); }

int interruptFlag;

static unsigned int terrupt;
static void CommunicationInterrupt(int arg)
{
  nodes[CmiMyNode()].stat_total_intr++;
  if (comm_flag) return;
  if (memflag) return;
  nodes[CmiMyNode()].stat_proc_intr++;
  CommunicationServer(0);
}

extern void CmiSignal(int sig1, int sig2, int sig3, void (*handler)());

static void CmiStartThreads()
{
  struct itimerval i;
  
  if ((Cmi_numpes != Cmi_numnodes) || (Cmi_mynodesize != 1))
    KillEveryone
      ("Multiple cpus unavailable, don't use cpus directive in nodesfile.\n");
  
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  Cmi_mype = Cmi_nodestart;
  Cmi_myrank = 0;
  
  if (!Cmi_netpoll) 
  {
#if CMK_ASYNC_NOT_NEEDED
    CmiSignal(SIGALRM, 0, 0, CommunicationInterrupt);
#else
    CmiSignal(SIGALRM, SIGIO, 0, CommunicationInterrupt);
    if (dataskt!=-1) CmiEnableAsyncIO(dataskt);
    if (Cmi_charmrun_fd!=-1) CmiEnableAsyncIO(Cmi_charmrun_fd);
#endif
  }
  
  /* if running on only one node, the only thing an interrupt
  is used for is to check if charmrun has been killed. And this is
  done only Cmi_check_delay seconds, so no need to have tickspeed
  any faster than that. */

  if(Cmi_numnodes==1) Cmi_tickspeed = (int)(Cmi_check_delay*1000000.0);

  if (!Cmi_netpoll) 
  {
  /*This will send us a SIGALRM every Cmi_tickspeed microseconds,
    which will call the CommunicationInterrupt routine above.*/
    i.it_interval.tv_sec = 0;
    i.it_interval.tv_usec = Cmi_tickspeed;
    i.it_value.tv_sec = 0;
    i.it_value.tv_usec = Cmi_tickspeed;
    setitimer(ITIMER_REAL, &i, NULL);
  }
}

#endif

CpvDeclare(void *, CmiLocalQueue);
CpvStaticDeclare(char *, internal_printf_buffer);

/****************************************************************************
 *
 * ctrl_sendone
 *
 * Careful, don't call the locking version from inside the 
 * communication thread, you'll mess up the communications mutex.
 *
 ****************************************************************************/

static int sendone_abort_fn(int code,const char *msg) {
	fprintf(stderr,"Socket error %d in ctrl_sendone! %s\n",code,msg);
	exit(1);
	return -1;
}

static void ctrl_sendone_nolock(const char *type,
				const char *data1,int dataLen1,
				const char *data2,int dataLen2)
{
  ChMessageHeader hdr;
  if (Cmi_charmrun_fd==-1) 
  	charmrun_abort("ctrl_sendone called in standalone!\n");
  ChMessageHeader_new(type,dataLen1+dataLen2,&hdr);
  skt_sendN(Cmi_charmrun_fd,(const char *)&hdr,sizeof(hdr));
  if (dataLen1>0) skt_sendN(Cmi_charmrun_fd,data1,dataLen1);
  if (dataLen2>0) skt_sendN(Cmi_charmrun_fd,data2,dataLen2);
}

static void ctrl_sendone_locking(const char *type,
				const char *data1,int dataLen1,
				const char *data2,int dataLen2)
{
  skt_abortFn oldAbort;
  CmiCommLock();
  oldAbort=skt_set_abort(sendone_abort_fn);
  ctrl_sendone_nolock(type,data1,dataLen1,data2,dataLen2);
  skt_set_abort(oldAbort);
  CmiCommUnlock();
}

static int ignore_further_errors(int c,const char *msg) {exit(2);return -1;}
static void charmrun_abort(const char *s)
{
  if (Cmi_charmrun_fd==-1) {/*Standalone*/
  	fprintf(stderr,"Charm++ fatal error:\n%s\n",s);
  	abort();
  } else {
  	skt_set_abort(ignore_further_errors);
  	ctrl_sendone_nolock("abort",s,strlen(s)+1,NULL,0);
  }
}

/****************************************************************************
 *
 * ctrl_getone
 *
 * This is handled only by the communications thread.
 *
 ****************************************************************************/


static void ctrl_getone()
{
  ChMessage msg;
  ChMessage_recv(Cmi_charmrun_fd,&msg);

  if (strcmp(msg.header.type,"die")==0) {
    fprintf(stderr,"aborting: %s\n",msg.data);
    log_done();
    ConverseCommonExit();
    exit(0);
#if CMK_CCS_AVAILABLE
  } else if (strcmp(msg.header.type, "req_fw")==0) {
    CcsImplHeader *hdr=(CcsImplHeader *)msg.data;
	/*Sadly, I *can't* do a:
      CcsImpl_netRequest(hdr,msg.data+sizeof(CcsImplHeader));
	here, because I can't send converse messages in the
	communcation thread.  I *can* poke this message into 
	any convenient processor's queue, though:  (OSL, 9/14/2000)
	*/
	int pe=0;/*<- node-local processor number. Any one will do.*/
	char *cmsg=CcsImpl_ccs2converse(hdr,msg.data+sizeof(CcsImplHeader),NULL);
	PCQueuePush(CmiGetStateN(pe)->recv,cmsg);
#endif
  }  else {
  /* We do not use KillEveryOne here because it calls CmiMyPe(),
   * which is not available to the communication thread on an SMP version.
   */
    charmrun_abort("ERROR> Unrecognized message from charmrun.\n");
    exit(1);
  }
  
  ChMessage_free(&msg);
}

#if CMK_CCS_AVAILABLE
/*Deliver this reply data to this reply socket.
  The data is forwarded to CCS server via charmrun.*/
void CcsImpl_reply(SOCKET replFd,int repLen,const void *repData)
{
  ChMessageInt_t skt=ChMessageInt_new(replFd);
  ctrl_sendone_locking("reply_fw",(const char *)&skt,sizeof(skt),
		       repData,repLen);  
}
#endif

/*****************************************************************************
 *
 * node_addresses
 *
 *  These two functions fill the node-table.
 *
 *
 *   This node, like all others, first sends its own address to charmrun
 *   using this command:
 *
 *     Type: nodeinfo
 *     Data: Big-endian 4-byte ints
 *           <my-node #><Dataport>
 *
 *   When charmrun has all the addresses, he sends this table to me:
 *
 *     Type: nodes
 *     Data: Big-endian 4-byte ints
 *           <number of nodes n>
 *           <#PEs><IP><Dataport> Node 0
 *           <#PEs><IP><Dataport> Node 1
 *           ...
 *           <#PEs><IP><Dataport> Node n-1
 *
 *****************************************************************************/

static void node_addresses_store(ChMessage *msg);

/*Note: node_addresses_obtain is called before starting
  threads, so no locks are needed (or valid!)*/
static void node_addresses_obtain(char **argv)
{
  ChMessage nodetabmsg; /* info about all nodes*/
  if (Cmi_charmrun_fd==-1) 
  {/*Standalone-- fake a single-node nodetab message*/
  	int npes=1;
  	int fakeLen=4*sizeof(ChMessageInt_t);
  	ChMessageInt_t *fakeTab=(ChMessageInt_t *)malloc(fakeLen);
  	CmiGetArgInt(argv,"+p",&npes);
#if CMK_SHARED_VARS_UNAVAILABLE
	if (npes!=1) {
		fprintf(stderr,
			"To use multiple processors, you must run this program as:\n"
			" > charmrun +p%d %s <args>\n"
			"or build the %s-smp version of Charm++.\n",
			npes,argv[0],CMK_MACHINE_NAME);
		exit(1);
	}
#endif
	fakeTab[0]=ChMessageInt_new(1);
	fakeTab[1]=ChMessageInt_new(npes);
	fakeTab[2]=fakeTab[3]=ChMessageInt_new(0);
 	nodetabmsg.len=fakeLen;
 	nodetabmsg.data=(char *)fakeTab;
  }
  else 
  { /*Contact charmrun for machine info.*/
  	ChMessageInt_t info[2]; /*Info. about my node for charmrun*/
  	int infoLen=2*sizeof(ChMessageInt_t);
 	
  	info[0]=ChMessageInt_new(Cmi_mynode);
  	info[1]=ChMessageInt_new(dataport);

  	/*Send our node info. to charmrun.
  	CommLock hasn't been initialized yet-- 
  	use non-locking version*/  
  	ctrl_sendone_nolock("initnode",(const char *)&info[0],infoLen,NULL,0);
  
  	/*We get the other node addresses from a message sent
  	  back via the charmrun control port.*/
  	if (!skt_select1(Cmi_charmrun_fd,600*1000)) CmiAbort("Timeout waiting for nodetab!\n");
  	ChMessage_recv(Cmi_charmrun_fd,&nodetabmsg);
  }
  node_addresses_store(&nodetabmsg);
  ChMessage_free(&nodetabmsg);
}

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
  ChMessageInt_t *d=(ChMessageInt_t *)msg->data;
  int nodestart;
  int i,j;
  OtherNode ntab, *bype;
  Cmi_numnodes=ChMessageInt(*d++);
  if (sizeof(ChMessageInt_t)*(1+3*Cmi_numnodes)!=(unsigned int)msg->len)
    {printf("Node table has inconsistent length!");abort();}
  ntab = (OtherNode)calloc(Cmi_numnodes, sizeof(struct OtherNodeStruct));
  nodestart=0;
  for (i=0; i<Cmi_numnodes; i++) {
    ntab[i].nodestart = nodestart;
    ntab[i].nodesize  = ChMessageInt(*d++);
    ntab[i].IP = ChMessageInt(*d++);;
    if (i==Cmi_mynode) {
      Cmi_nodestart=ntab[i].nodestart;
      Cmi_mynodesize=ntab[i].nodesize;
      Cmi_self_IP=ntab[i].IP;
    }
    ntab[i].dataport = ChMessageInt(*d++);
    ntab[i].addr = skt_build_addr(ntab[i].IP,ntab[i].dataport);
    nodestart+=ntab[i].nodesize;
  }
  Cmi_numpes=nodestart;
  bype = (OtherNode*)malloc(Cmi_numpes * sizeof(OtherNode));
  _MEMCHECK(bype);
  for (i=0; i<Cmi_numnodes; i++) {
    OtherNode node = ntab + i;
    node->sent_msgs = 0;
    node->recd_msgs = 0;
    node->sent_bytes = 0;
    node->recd_bytes = 0;
    node->send_window =
      (ImplicitDgram*)calloc(Cmi_window_size, sizeof(ImplicitDgram));
    node->recv_window =
      (ExplicitDgram*)calloc(Cmi_window_size, sizeof(ExplicitDgram));
    for (j=0; j<node->nodesize; j++)
      bype[j + node->nodestart] = node;
  }
  nodes_by_pe = bype;
  nodes = ntab;
}

/*****************************************************************************
 *
 * CmiPrintf, CmiError, CmiScanf
 *
 *****************************************************************************/

static void InternalPrintf(f, l) char *f; va_list l;
{
  ChMessage replymsg;
  char *buffer = CpvAccess(internal_printf_buffer);
  vsprintf(buffer, f, l);
  if (Cmi_charmrun_fd!=-1) {
    if(Cmi_syncprint) {
  	  ctrl_sendone_locking("printsync", buffer,strlen(buffer)+1,NULL,0);
      CmiCommLock();
  	  ChMessage_recv(Cmi_charmrun_fd,&replymsg);
  	  ChMessage_free(&replymsg);
      CmiCommUnlock();
    } else {
  	  ctrl_sendone_locking("print", buffer,strlen(buffer)+1,NULL,0);
    }
  } else {
  	fprintf(stdout,"%s",buffer);
  }
}

static void InternalError(f, l) char *f; va_list l;
{
  ChMessage replymsg;
  char *buffer = CpvAccess(internal_printf_buffer);
  vsprintf(buffer, f, l);
  if (Cmi_charmrun_fd!=-1) {
    if(Cmi_syncprint) {
  	  ctrl_sendone_locking("printerrsync", buffer,strlen(buffer)+1,NULL,0);
      CmiCommLock();
  	  ChMessage_recv(Cmi_charmrun_fd,&replymsg);
  	  ChMessage_free(&replymsg);
      CmiCommUnlock();
    } else {
  	  ctrl_sendone_locking("printerr", buffer,strlen(buffer)+1,NULL,0);
    }
  } else {
  	fprintf(stderr,"%s",buffer);
  }
}

static int InternalScanf(fmt, l)
    char *fmt;
    va_list l;
{
  ChMessage replymsg;
  char *ptr[20];
  char *p; int nargs, i;
  nargs=0;
  p=fmt;
  while (*p) {
    if ((p[0]=='%')&&(p[1]=='*')) { p+=2; continue; }
    if ((p[0]=='%')&&(p[1]=='%')) { p+=2; continue; }
    if (p[0]=='%') { nargs++; p++; continue; }
    if (*p=='\n') *p=' '; p++;
  }
  if (nargs > 18) KillEveryone("CmiScanf only does 18 args.\n");
  for (i=0; i<nargs; i++) ptr[i]=va_arg(l, char *);
  CmiLock(Cmi_scanf_mutex);
  if (Cmi_charmrun_fd!=-1)
  {/*Send charmrun the format string*/
  	ctrl_sendone_locking("scanf", fmt, strlen(fmt)+1,NULL,0);
  	/*Wait for the reply (characters to scan) from charmrun*/
  	CmiCommLock();
  	ChMessage_recv(Cmi_charmrun_fd,&replymsg);
  	i = sscanf((char*)replymsg.data, fmt,
		     ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3], ptr[ 4], ptr[ 5],
		     ptr[ 6], ptr[ 7], ptr[ 8], ptr[ 9], ptr[10], ptr[11],
		     ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17]);
  	ChMessage_free(&replymsg);
  	CmiCommUnlock();
  } else 
  {/*Just do the scanf normally*/
  	i=scanf(fmt, ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3], ptr[ 4], ptr[ 5],
		     ptr[ 6], ptr[ 7], ptr[ 8], ptr[ 9], ptr[10], ptr[11],
		     ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17]);
  }
  CmiUnlock(Cmi_scanf_mutex);
  return i;
}

/*New stdarg.h declarations*/
void CmiPrintf(const char *fmt, ...)
{
  va_list p; va_start(p, fmt);
  InternalPrintf(fmt, p);
  va_end(p);
}

void CmiError(const char *fmt, ...)
{
  va_list p; va_start (p, fmt);
  InternalError(fmt, p);
  va_end(p);
}

int CmiScanf(const char *fmt, ...)
{
  va_list p; int i; va_start(p, fmt);
  i = InternalScanf(fmt, p);
  va_end(p);
  return i;
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


/***********************************************************************
 * TransmitAckDatagram
 *
 * This function sends the ack datagram, after setting the window
 * array to show which of the datagrams in the current window have been
 * received. The sending side will then use this information to resend
 * packets, mark packets as received, etc. This system also prevents
 * multiple retransmissions/acks when acks are lost.
 ***********************************************************************/
void TransmitAckDatagram(OtherNode node)
{
  DgramAck ack; int i, seqno, slot; ExplicitDgram dg;
  int retval;
  
  seqno = node->recv_next;
  DgramHeaderMake(&ack, DGRAM_ACKNOWLEDGE, Cmi_nodestart, Cmi_charmrun_pid, seqno);
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


/***********************************************************************
 * TransmitImplicitDgram
 * TransmitImplicitDgram1
 *
 * These functions do the actual work of sending a UDP datagram.
 ***********************************************************************/
void TransmitImplicitDgram(ImplicitDgram dg)
{
  char *data; DgramHeader *head; int len; DgramHeader temp;
  OtherNode dest;
  int retval;

  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_charmrun_pid, dg->seqno);
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

  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_charmrun_pid, dg->seqno);
  LOG(Cmi_clock, Cmi_nodestart, 'P', dest->nodestart, dg->seqno);
  retval = (-1);
  while (retval == (-1))
    retval = sendto(dataskt, (char *)head, len + DGRAM_HEADER_SIZE, 0,
	      (struct sockaddr *)&(dest->addr), sizeof(struct sockaddr_in));
  *head = temp;
  dest->stat_resend_pkt++;
}


/***********************************************************************
 * TransmitAcknowledgement
 *
 * This function sends the ack datagrams, after checking to see if the 
 * Recv Window is atleast half-full. After that, if the Recv window size 
 * is 0, then the count of un-acked datagrams, and the time at which
 * the ack should be sent is reset.
 ***********************************************************************/
int TransmitAcknowledgement()
{
  int skip; static int nextnode=0; OtherNode node;
  
  for (skip=0; skip<Cmi_numnodes; skip++) {
    node = nodes+nextnode;
    nextnode = (nextnode + 1) % Cmi_numnodes;
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


/***********************************************************************
 * TransmitDatagram()
 *
 * This function fills up the Send Window with the contents of the
 * Send Queue. It also sets the node->send_primer variable, which
 * indicates when a retransmission will be attempted.
 ***********************************************************************/
int TransmitDatagram()
{
  ImplicitDgram dg; OtherNode node;
  static int nextnode=0; int skip, count, slot;
  unsigned int seqno;
  
  for (skip=0; skip<Cmi_numnodes; skip++) {
    node = nodes+nextnode;
    nextnode = (nextnode + 1) % Cmi_numnodes;
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
 
  size = ogm->size - DGRAM_HEADER_SIZE;
  data = ogm->data + DGRAM_HEADER_SIZE;
  while (size > Cmi_dgram_max_data) {
    EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank);
    data += Cmi_dgram_max_data;
    size -= Cmi_dgram_max_data;
  }
  EnqueueOutgoingDgram(ogm, data, size, node, rank);

  myNode->sent_msgs++;
  myNode->sent_bytes += ogm->size;
}

#if CMK_NODE_QUEUE_AVAILABLE

/***********************************************************************
 * DeliverOutgoingNodeMessage()
 *
 * This function takes care of delivery of outgoing node messages from the
 * sender end. Broadcast messages are divided into sets of messages that 
 * are bound to the local node, and to remote nodes. For local
 * transmission, the messages are directly pushed into the recv
 * queues. For non-local transmission, the function DeliverViaNetwork()
 * is called
 ***********************************************************************/
void DeliverOutgoingNodeMessage(OutgoingMsg ogm)
{
  int i, rank, dst; OtherNode node;

  dst = ogm->dst;
  switch (dst) {
  case NODE_BROADCAST_OTHERS:
    for (i = 0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_NODEMESSAGE);
    GarbageCollectMsg(ogm);
    break;
  case NODE_BROADCAST_ALL:
    PCQueuePush(CsvAccess(NodeRecv),CopyMsg(ogm->data,ogm->size));
    for (i=0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_NODEMESSAGE);
    GarbageCollectMsg(ogm);
    break;
  default:
    node = nodes+dst;
    rank=DGRAM_NODEMESSAGE;
    if (dst != Cmi_mynode) {
      DeliverViaNetwork(ogm, node, rank);
      GarbageCollectMsg(ogm);
    } else {
      if (ogm->freemode == 'A') {
	PCQueuePush(CsvAccess(NodeRecv),CopyMsg(ogm->data,ogm->size));
	ogm->freemode = 'X';
      } else {
	PCQueuePush(CsvAccess(NodeRecv), ogm->data);
	FreeOutgoingMsg(ogm);
      }
    }
  }
}

#else

#define DeliverOutgoingNodeMessage(msg) DeliverOutgoingMessage(msg)

#endif

/***********************************************************************
 * DeliverOutgoingMessage()
 *
 * This function takes care of delivery of outgoing messages from the
 * sender end. Broadcast messages are divided into sets of messages that 
 * are bound to the local node, and to remote nodes. For local
 * transmission, the messages are directly pushed into the recv
 * queues. For non-local transmission, the function DeliverViaNetwork()
 * is called
 ***********************************************************************/
void DeliverOutgoingMessage(OutgoingMsg ogm)
{
  int i, rank, dst; OtherNode node;
  
  dst = ogm->dst;
  switch (dst) {
  case PE_BROADCAST_ALL:
    for (rank = 0; rank<Cmi_mynodesize; rank++) {
      PCQueuePush(CmiGetStateN(rank)->recv,CopyMsg(ogm->data,ogm->size));
    }
    for (i=0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_BROADCAST);
    GarbageCollectMsg(ogm);
    break;
  case PE_BROADCAST_OTHERS:
    for (rank = 0; rank<Cmi_mynodesize; rank++)
      if (rank + Cmi_nodestart != ogm->src) {
	PCQueuePush(CmiGetStateN(rank)->recv,CopyMsg(ogm->data,ogm->size));
      }
    for (i = 0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_BROADCAST);
    GarbageCollectMsg(ogm);
    break;
  default:
    node = nodes_by_pe[dst];
    rank = dst - node->nodestart;
    if (node->nodestart != Cmi_nodestart) {
      DeliverViaNetwork(ogm, node, rank);
      GarbageCollectMsg(ogm);
    } else {
      if (ogm->freemode == 'A') {
	PCQueuePush(CmiGetStateN(rank)->recv,CopyMsg(ogm->data,ogm->size));
	ogm->freemode = 'X';
      } else {
	PCQueuePush(CmiGetStateN(rank)->recv, ogm->data);
	FreeOutgoingMsg(ogm);
      }
    }
  }
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
	PCQueuePush(CmiGetStateN(i)->recv, CopyMsg(msg, len));
      PCQueuePush(CmiGetStateN(0)->recv, msg);
    } else {
#if CMK_NODE_QUEUE_AVAILABLE
         if (node->asm_rank==DGRAM_NODEMESSAGE) {
	   PCQueuePush(CsvAccess(NodeRecv), msg);
         }
	 else
#endif
	   PCQueuePush(CmiGetStateN(node->asm_rank)->recv, msg);
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

  LOG(Cmi_clock, Cmi_nodestart, 'M', dg->srcpe, dg->seqno);
  node = nodes_by_pe[dg->srcpe];
  node->stat_recv_pkt++;
  seqno = dg->seqno;
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
  /* check that the ack being received is actually appropriate */
  if ( !((node->recv_ack_seqno >= 
	  ((DGRAM_SEQNO_MASK >> 1) + (DGRAM_SEQNO_MASK >> 2))) &&
	 (ackseqno < (DGRAM_SEQNO_MASK >> 1))) &&
       (ackseqno <= node->recv_ack_seqno))
    {
      FreeExplicitDgram(dg);
      return;
    } 
  /* higher ack so adjust */
  node->recv_ack_seqno = ackseqno;
  
  for (i=Cmi_window_size-1; i>=0; i--) {
    slot--; if (slot== ((unsigned int)-1)) slot+=Cmi_window_size;
    seqno = (seqno-1) & DGRAM_SEQNO_MASK;
    idg = node->send_window[slot];
    if (idg) {
      if (idg->seqno == seqno) {
	if (ack->window[i]) {
	  /* remove those that have been received and are within a window
	     of the first missing packet */
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

        if (dgseqno < idg->seqno)
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

void ReceiveDatagram()
{
  ExplicitDgram dg; int ok, magic;
  MallocExplicitDgram(dg);
  ok = recv(dataskt,(char*)(dg->data),Cmi_max_dgram_size,0);
  /*ok = recvfrom(dataskt,(char*)(dg->data),Cmi_max_dgram_size,0, 0, 0);*/
  /* if (ok<0) { perror("recv"); KillEveryoneCode(37489437); } */
  if (ok < 0) {
    if (errno == EINTR) return;          /* ignore the error.  G. Zheng */
    CmiPrintf("ReceiveDatagram: recv: %s\n", strerror(errno)) ;
    KillEveryoneCode(37489437);
  }
  dg->len = ok;
  if (ok >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(dg->data, dg->rank, dg->srcpe, magic, dg->seqno);
    if (magic == (Cmi_charmrun_pid&DGRAM_MAGIC_MASK)) {
      if (dg->rank == DGRAM_ACKNOWLEDGE)
	IntegrateAckDatagram(dg);
      else IntegrateMessageDatagram(dg);
    } else FreeExplicitDgram(dg);
  } else FreeExplicitDgram(dg);
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
static void CommunicationServer(int withDelayMs)
{
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
    CheckSocketsReady(withDelayMs);
    if (ctrlskt_ready_read) { ctrl_getone(); continue; }
    if (dataskt_ready_read) { ReceiveDatagram(); continue; }
    if (dataskt_ready_write) { if (TransmitAcknowledgement()) continue; }
    if (dataskt_ready_write) { if (TransmitDatagram()) continue; }
    break;
  }
  CmiCommUnlock();
#if CMK_SHARED_VARS_UNAVAILABLE
  terrupt--;
#endif
}

/******************************************************************************
 *
 * CmiGetNonLocal
 *
 * The design of this system is that the communication thread does all the
 * work, to eliminate as many locking issues as possible.  This is the only
 * part of the code that happens in the receiver-thread.
 *
 * This operation is fairly cheap, it might be worthwhile to inline
 * the code into CmiDeliverMsgs to reduce function call overhead.
 *
 *****************************************************************************/

#if CMK_NODE_QUEUE_AVAILABLE
char *CmiGetNonLocalNodeQ()
{
  char *result = 0;
  if (Cmi_netpoll) CommunicationServer(0);
  if(!PCQueueEmpty(CsvAccess(NodeRecv))) {
    CmiLock(CsvAccess(CmiNodeRecvLock));
    result = (char *) PCQueuePop(CsvAccess(NodeRecv));
    CmiUnlock(CsvAccess(CmiNodeRecvLock));
  }
  return result;
}
#endif

char *CmiGetNonLocal()
{
  CmiState cs = CmiGetState();
  return (char *) PCQueuePop(cs->recv);
}

/******************************************************************************
 *
 * CmiNotifyIdle()-- wait until a packet comes in
 *
 *****************************************************************************/

void CmiNotifyIdle()
{
  struct timeval tv;
#if CMK_SHARED_VARS_UNAVAILABLE
  /*No comm. thread-- listen on sockets for incoming messages*/
  static fd_set rfds;
  static fd_set wfds;
  tv.tv_sec=0; tv.tv_usec=5000;
  if (Cmi_charmrun_fd!=-1)
    FD_SET(Cmi_charmrun_fd, &rfds);
  if (dataskt!=-1) {
    FD_SET(dataskt, &rfds);
    FD_SET(dataskt, &wfds);
  }
  select(FD_SETSIZE,&rfds,&wfds,0,&tv);
  if (Cmi_netpoll) CommunicationServer(5);
#else
  /*Comm. thread will listen on sockets-- just sleep*/
  tv.tv_sec=0; tv.tv_usec=1000;
  select(0,NULL,NULL,NULL,&tv);
#endif
}

#if CMK_NODE_QUEUE_AVAILABLE

/******************************************************************************
 *
 * CmiGeneralNodeSend
 *
 * Description: This is a generic message sending routine. All the
 * converse message send functions are implemented in terms of this
 * function. (By setting appropriate flags (eg freemode) that tell
 * CmiGeneralSend() how exactly to handle the particular case of
 * message send)
 *
 *****************************************************************************/

CmiCommHandle CmiGeneralNodeSend(int pe, int size, int freemode, char *data)
{
  CmiState cs = CmiGetState(); OutgoingMsg ogm;

  if (freemode == 'S') {
    char *copy = (char *)CmiAlloc(size);
  if (!copy)
      fprintf(stderr, "%d: Out of mem\n", Cmi_mynode);
    memcpy(copy, data, size);
    data = copy; freemode = 'F';
  }

  MallocOutgoingMsg(ogm);
  CmiMsgHeaderSetLength(data, size);
  ogm->size = size;
  ogm->data = data;
  ogm->src = cs->pe;
  ogm->dst = pe;
  ogm->freemode = freemode;
  ogm->refcount = 0;
  CmiCommLock();
  DeliverOutgoingNodeMessage(ogm);
  CmiCommUnlock();
  CommunicationServer(0);
  return (CmiCommHandle)ogm;
}
#endif

/******************************************************************************
 *
 * CmiGeneralSend
 *
 * Description: This is a generic message sending routine. All the
 * converse message send functions are implemented in terms of this
 * function. (By setting appropriate flags (eg freemode) that tell
 * CmiGeneralSend() how exactly to handle the particular case of
 * message send)
 *
 *****************************************************************************/

CmiCommHandle CmiGeneralSend(int pe, int size, int freemode, char *data)
{
  CmiState cs = CmiGetState(); OutgoingMsg ogm;

  if (freemode == 'S') {
    char *copy = (char *)CmiAlloc(size);
  if (!copy)
      fprintf(stderr, "%d: Out of mem\n", Cmi_mynode);
    memcpy(copy, data, size);
    data = copy; freemode = 'F';
  }

  if (pe == cs->pe) {
    CdsFifo_Enqueue(cs->localqueue, data);
    if (freemode == 'A') {
      MallocOutgoingMsg(ogm);
      ogm->freemode = 'X';
      return ogm;
    } else return 0;
  }
  
  MallocOutgoingMsg(ogm);
  CmiMsgHeaderSetLength(data, size);
  ogm->size = size;
  ogm->data = data;
  ogm->src = cs->pe;
  ogm->dst = pe;
  ogm->freemode = freemode;
  ogm->refcount = 0;
  CmiCommLock();
  DeliverOutgoingMessage(ogm);
  CmiCommUnlock();
  CommunicationServer(0);
  return (CmiCommHandle)ogm;
}

void CmiSyncSendFn(int p, int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), 1);
  CmiGeneralSend(p,s,'S',m); 
}

CmiCommHandle CmiAsyncSendFn(int p, int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), 1);
  return CmiGeneralSend(p,s,'A',m); 
}

void CmiFreeSendFn(int p, int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), 1);
  CmiGeneralSend(p,s,'F',m); 
}

void CmiSyncBroadcastFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumPes()-1); 
  CmiGeneralSend(PE_BROADCAST_OTHERS,s,'S',m); 
}

CmiCommHandle CmiAsyncBroadcastFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumPes()-1); 
  return CmiGeneralSend(PE_BROADCAST_OTHERS,s,'A',m); 
}

void CmiFreeBroadcastFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
  CmiGeneralSend(PE_BROADCAST_OTHERS,s,'F',m); 
}

void CmiSyncBroadcastAllFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumPes()); 
  CmiGeneralSend(PE_BROADCAST_ALL,s,'S',m); 
}

CmiCommHandle CmiAsyncBroadcastAllFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumPes()); 
  return CmiGeneralSend(PE_BROADCAST_ALL,s,'A',m); 
}

void CmiFreeBroadcastAllFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumPes()); 
  CmiGeneralSend(PE_BROADCAST_ALL,s,'F',m); 
}

#if CMK_NODE_QUEUE_AVAILABLE

void CmiSyncNodeSendFn(int p, int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), 1);
  CmiGeneralNodeSend(p,s,'S',m); 
}

CmiCommHandle CmiAsyncNodeSendFn(int p, int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), 1);
  return CmiGeneralNodeSend(p,s,'A',m); 
}

void CmiFreeNodeSendFn(int p, int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), 1);
  CmiGeneralNodeSend(p,s,'F',m); 
}

void CmiSyncNodeBroadcastFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
  CmiGeneralNodeSend(NODE_BROADCAST_OTHERS,s,'S',m); 
}

CmiCommHandle CmiAsyncNodeBroadcastFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
  return CmiGeneralNodeSend(NODE_BROADCAST_OTHERS,s,'A',m);
}

void CmiFreeNodeBroadcastFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
  CmiGeneralNodeSend(NODE_BROADCAST_OTHERS,s,'F',m); 
}

void CmiSyncNodeBroadcastAllFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumNodes());
  CmiGeneralNodeSend(NODE_BROADCAST_ALL,s,'S',m); 
}

CmiCommHandle CmiAsyncNodeBroadcastAllFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumNodes());
  return CmiGeneralNodeSend(NODE_BROADCAST_ALL,s,'A',m); 
}

void CmiFreeNodeBroadcastAllFn(int s, char *m)
{ 
  CQdCreate(CpvAccess(cQdState), CmiNumNodes());
  CmiGeneralNodeSend(NODE_BROADCAST_ALL,s,'F',m); 
}
#endif

/******************************************************************************
 *
 * Comm Handle manipulation.
 *
 *****************************************************************************/

int CmiAsyncMsgSent(CmiCommHandle handle)
{
  return (((OutgoingMsg)handle)->freemode == 'X');
}

void CmiReleaseCommHandle(CmiCommHandle handle)
{
  FreeOutgoingMsg(((OutgoingMsg)handle));
}

/******************************************************************************
 *
 * Main code, Init, and Exit
 *
 *****************************************************************************/
extern void CthInit(char **argv);
extern void ConverseCommonInit(char **);

static char     **Cmi_argv;
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */

static void ConverseRunPE(int everReturn)
{
  CmiState cs;
  char** CmiMyArgv;
  CmiNodeBarrier();
  cs = CmiGetState();
  CpvInitialize(char *, internal_printf_buffer);
  
  CpvAccess(internal_printf_buffer) = (char *) malloc(PRINTBUFSIZE);
  _MEMCHECK(CpvAccess(internal_printf_buffer));
  CmiMyArgv=CmiCopyArgs(Cmi_argv);
  CthInit(CmiMyArgv);
  ConverseCommonInit(CmiMyArgv);
  CpvInitialize(void *,CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = cs->localqueue;

  if (!everReturn) {
    Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
    if (Cmi_usrsched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

void ConverseExit()
{
  if (CmiMyRank()==0) {
    if(Cmi_print_stats)
      printNetStatistics();
    log_done();
    ConverseCommonExit();
  }
  if (Cmi_charmrun_fd==-1)
  	exit(0); /*Standalone version-- just leave*/
  else {
  	ctrl_sendone_locking("ending",NULL,0,NULL,0); /* this causes charmrun to go away */
 	while (1) {
 		if (Cmi_netpoll) CommunicationServer(500);
 		CmiYield();/*Loop until charmrun dies, which will be caught by comm. thread*/
 	}
  }
}

static void exitDelay(void)
{
  printf("Program finished.\n");
#if 0
  fgetc(stdin);
#endif
}

static void set_signals(void)
{
#if !CMK_TRUECRASH
  signal(SIGSEGV, KillOnAllSigs);
  signal(SIGFPE, KillOnAllSigs);
  signal(SIGILL, KillOnAllSigs);
  signal(SIGINT, KillOnAllSigs);
  signal(SIGTERM, KillOnAllSigs);
  signal(SIGABRT, KillOnAllSigs);
#  if !defined(_WIN32) || defined(__CYGWIN__) /*UNIX-only signals*/
  signal(SIGQUIT, KillOnAllSigs);
  signal(SIGBUS, KillOnAllSigs);
  signal(SIGPIPE, KillOnSIGPIPE);
#    if CMK_HANDLE_SIGUSR
  signal(SIGUSR1, HandleUserSignals);
  signal(SIGUSR2, HandleUserSignals);
#    endif
#  endif /*UNIX*/
#endif /*CMK_TRUECRASH*/
}

/*Socket idle function to use before addresses have been
  obtained.  During the real program, we idle with CmiYield.
*/
static void obtain_idleFn(void) {sleep(0);}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usc, int everReturn)
{
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  Cmi_argv = argv; Cmi_startfn = fn; Cmi_usrsched = usc;
  Cmi_netpoll=CmiGetArgFlag(argv,"+netpoll");
  Cmi_syncprint=CmiGetArgFlag(argv,"+syncprint");
  skt_init();
  atexit(exitDelay);
  parse_netstart();
  extract_args(argv);
  log_init();
  Cmi_scanf_mutex = CmiCreateLock();

  skt_set_idle(obtain_idleFn);
  if (Cmi_charmrun_IP!=0) {
  	set_signals();
  	dataskt=skt_datagram(&dataport, Cmi_os_buffer_size);
  	Cmi_charmrun_fd = skt_connect(Cmi_charmrun_IP, Cmi_charmrun_port, 1800);
  } else {/*Standalone operation*/
  	printf("Charm++: standalone mode (not using charmrun)\n");
  	dataskt=-1;
  	Cmi_charmrun_fd=-1;
  }
  node_addresses_obtain(argv);
  skt_set_idle(CmiYield);
  Cmi_check_delay = 2.0+0.5*Cmi_numnodes;
  CmiStartThreads();
  ConverseRunPE(everReturn);
}
