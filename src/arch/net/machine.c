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

#define MACHINE_DEBUG 0
#if MACHINE_DEBUG
/*Controls amount of debug messages: 1 (the lowest priority) is 
extremely verbose, 2 shows most procedure entrance/exits, 
3 shows most communication, and 5 only shows rare or unexpected items.
Displaying lower priority messages doesn't stop higher priority ones.
*/
#define MACHINE_DEBUG_PRIO 2
#define MACHINE_DEBUG_LOG 0 /*Controls whether output goes to log file*/

FILE *debugLog;
# define MACHSTATE_I(prio,args) if ((prio)>=MACHINE_DEBUG_PRIO) {\
	fprintf args ; fflush(debugLog); }
# define MACHSTATE(prio,str) \
	MACHSTATE_I(prio,(debugLog,"[%.3f]> "str"\n",CmiWallTimer()))
# define MACHSTATE1(prio,str,a) \
	MACHSTATE_I(prio,(debugLog,"[%.3f]> "str"\n",CmiWallTimer(),a))
# define MACHSTATE2(prio,str,a,b) \
	MACHSTATE_I(prio,(debugLog,"[%.3f]> "str"\n",CmiWallTimer(),a,b))
# define MACHSTATE3(prio,str,a,b,c) \
	MACHSTATE_I(prio,(debugLog,"[%.3f]> "str"\n",CmiWallTimer(),a,b,c))
#else
# define MACHINE_DEBUG_LOG 0
# define MACHSTATE(n,x) /*empty*/
# define MACHSTATE1(n,x,a) /*empty*/
# define MACHSTATE2(n,x,a,b) /*empty*/
# define MACHSTATE3(n,x,a,b,c) /*empty*/
#endif

#if CMK_USE_POLL
#include <poll.h>
#endif

#if CMK_USE_GM
#include "gm.h"
struct gm_port *gmport = NULL;
int  portFinish = 0;
#endif

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
void CmiYield(void);
void ConverseCommonExit(void);

static unsigned int dataport=0;
static SOCKET       dataskt;

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

static void machine_exit(int status)
{
#if CMK_USE_GM
  if (gmport) { 
    gm_close(gmport); gmport = 0;
    gm_finalize();
  }
  exit(status);
#else
  exit(status);
#endif
}

static void charmrun_abort(const char*);

static void KillEveryone(const char *msg)
{
  charmrun_abort(msg);
  machine_exit(1);
}

static void KillEveryoneCode(n)
int n;
{
  char _s[100];
  sprintf(_s, "[%d] Fatal error #%d\n", CmiMyPe(), n);
  charmrun_abort(_s);
  machine_exit(1);
}

static void KillOnAllSigs(int sigNo)
{
  char _s[100];
  const char *sig=" received signal";
  if (sigNo==SIGSEGV) sig=": segmentation violation.\n"
	"Try running with '++debug', or linking with '-memory paranoid'";
  if (sigNo==SIGFPE) sig=": floating point exception.\nDid you divide by zero?";
  if (sigNo==SIGILL) sig=": illegal instruction";
  if (sigNo==SIGBUS) sig=": bus error";
  if (sigNo==SIGKILL) sig=": caught signal KILL";
  if (sigNo==SIGQUIT) sig=": caught signal QUIT";
  
  sprintf(_s, "ERROR> Node program on PE %d%s\n", CmiMyPe(),sig);
  charmrun_abort(_s);
  machine_exit(1);
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
  machine_exit(1);
}

/*****************************************************************************
 *
 *     Utility routines for network machine interface.
*
 *****************************************************************************/

double GetClock(void)
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


PCQueue PCQueueCreate(void)
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

static int  Cmi_truecrash;

void CmiAbort(const char *message)
{
  if(Cmi_truecrash) {
    printf("CHARM++ FATAL ERROR: %s\n", message);
    *(int *)NULL = 0; /*Write to null, causing bus error*/
  } else {
    charmrun_abort(message);
    machine_exit(1);
  }
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

#if CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN
#include <fcntl.h>
void CmiEnableAsyncIO(int fd)
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
#else
void CmiEnableAsyncIO(int fd) { }
#endif

#if 0
void CmiEnableNonblockingIO(int fd) {
  int on=1;
  if (fcntl(fd,F_SETFL,O_NONBLOCK,&on)<0) {
    CmiError("setting nonblocking IO: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

/***********************************************************
 * SMP Idle Locking
 *   In an SMP system, idle processors need to sleep on a
 * lock so that if a message for them arrives, they can be
 * woken up.
 **********************************************************/

#if CMK_SHARED_VARS_NT_THREADS
typedef struct {
  int hasMessages; /*Is there a message waiting?*/
  volatile int isSleeping; /*Are we asleep in this cond?*/
  HANDLE sem;
} CmiIdleLock;

static void CmiIdleLock_init(CmiIdleLock *l) {
  l->hasMessages=0;
  l->isSleeping=0;
  l->sem=CreateSemaphore(NULL,0,1, NULL);
}

static void CmiIdleLock_sleep(CmiIdleLock *l,int msTimeout) {
  if (l->hasMessages) return;
  l->isSleeping=1;
  MACHSTATE(4,"Processor going to sleep {")
  WaitForSingleObject(l->sem,msTimeout);
  MACHSTATE(4,"} Processor awake again")
  l->isSleeping=0;
}

static void CmiIdleLock_addMessage(CmiIdleLock *l) {
  l->hasMessages=1;
  if (l->isSleeping) { /*The PE is sleeping on this lock-- wake him*/  
    MACHSTATE(4,"Waking sleeping processor")
    ReleaseSemaphore(l->sem,1,NULL);
  }
}
static void CmiIdleLock_checkMessage(CmiIdleLock *l) {
  l->hasMessages=0;
}

#elif CMK_SHARED_VARS_POSIX_THREADS_SMP
typedef struct {
  volatile int hasMessages; /*Is there a message waiting?*/
  volatile int isSleeping; /*Are we asleep in this cond?*/
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} CmiIdleLock;

static void CmiIdleLock_init(CmiIdleLock *l) {
  l->hasMessages=0;
  l->isSleeping=0;
  pthread_mutex_init(&l->mutex,NULL);
  pthread_cond_init(&l->cond,NULL);
}

static void getTimespec(int msFromNow,struct timespec *dest) {
  struct timeval cur;
  int secFromNow;
  /*Get the current time*/
  gettimeofday(&cur,NULL);
  dest->tv_sec=cur.tv_sec;
  dest->tv_nsec=cur.tv_usec*1000;
  /*Add in the wait time*/
  secFromNow=msFromNow/1000;
  msFromNow-=secFromNow*1000;
  dest->tv_sec+=secFromNow;
  dest->tv_nsec+=1000*1000*msFromNow;
  /*Wrap around if we overflowed the nsec field*/
  while (dest->tv_nsec>1000000000u) {
    dest->tv_nsec-=1000000000;
    dest->tv_sec++;
  }
}

static void CmiIdleLock_sleep(CmiIdleLock *l,int msTimeout) {
  struct timespec wakeup;

  if (l->hasMessages) return;
  l->isSleeping=1;
  MACHSTATE(4,"Processor going to sleep {")
  pthread_mutex_lock(&l->mutex);
  getTimespec(msTimeout,&wakeup);
  while (!l->hasMessages)
    if (ETIMEDOUT==pthread_cond_timedwait(&l->cond,&l->mutex,&wakeup))
      break;
  pthread_mutex_unlock(&l->mutex);
  MACHSTATE(4,"} Processor awake again")
  l->isSleeping=0;
}
static void CmiIdleLock_wakeup(CmiIdleLock *l) {
  l->hasMessages=1;
  MACHSTATE(4,"Waking sleeping processor")
  /*The PE is sleeping on this condition variable-- wake him*/
  pthread_mutex_lock(&l->mutex);
  pthread_cond_signal(&l->cond);
  pthread_mutex_unlock(&l->mutex);
}

static void CmiIdleLock_addMessage(CmiIdleLock *l) {
  if (l->isSleeping) CmiIdleLock_wakeup(l);
  l->hasMessages=1;
}
static void CmiIdleLock_checkMessage(CmiIdleLock *l) {
  l->hasMessages=0;
}

#else /* Non-SMP version-- infinitely simpler */
typedef struct {
  int hasMessages;
} CmiIdleLock;
#define CmiIdleLock_sleep(x) /*empty*/

static void CmiIdleLock_init(CmiIdleLock *l) {
  l->hasMessages=0;
}
static void CmiIdleLock_addMessage(CmiIdleLock *l) {
  l->hasMessages=1;
}
static void CmiIdleLock_checkMessage(CmiIdleLock *l) {
  l->hasMessages=0;
}

#endif

/************************************************************
 * 
 * Processor state structure
 *
 ************************************************************/

typedef struct CmiStateStruct
{
  int pe, rank;
  PCQueue recv;
  void *localqueue;
  CmiIdleLock idle;
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
  CmiIdleLock_init(&state->idle);
#if CMK_NODE_QUEUE_AVAILABLE
  CsvInitialize(CmiNodeLock, CmiNodeRecvLock);
  CsvInitialize(PCQueue, NodeRecv);
  if (rank==0) {
    CsvAccess(CmiNodeRecvLock) = CmiCreateLock();
    CsvAccess(NodeRecv) = PCQueueCreate();
  }
#endif
}

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
static skt_ip_t   Cmi_self_IP;
static skt_ip_t   Cmi_charmrun_IP; /*Address of charmrun machine*/
static int        Cmi_charmrun_port;
static int        Cmi_charmrun_pid;
static int        Cmi_charmrun_fd;

static int    Cmi_netpoll;
static int    Cmi_idlepoll;
static int    Cmi_syncprint;
static int Cmi_print_stats = 0;

static void parse_netstart(void)
{
  char *ns;
  int nread;
  int port;
  ns = getenv("NETSTART");
  if (ns!=0) 
  {/*Read values set by Charmrun*/
        char Cmi_charmrun_name[1024];
        nread = sscanf(ns, "%d%s%d%d%d",
                 &Cmi_mynode,
                 Cmi_charmrun_name, &Cmi_charmrun_port,
                 &Cmi_charmrun_pid, &port);
	Cmi_charmrun_IP=skt_lookup_ip(Cmi_charmrun_name);

        if (nread!=5) {
                fprintf(stderr,"Error parsing NETSTART '%s'\n",ns);
                exit(1);
        }
#if CMK_USE_GM
        /* port is only useful for Myrinet */
        dataport = port;
#endif
  } else 
  {/*No charmrun-- set flag values for standalone operation*/
  	Cmi_mynode=0;
  	Cmi_charmrun_IP=skt_invalid_ip;
  	Cmi_charmrun_port=0;
  	Cmi_charmrun_pid=0;
#if CMK_USE_GM
        dataport = -1;
#endif
  }
}

static void extract_common_args(char **argv)
{
  if (CmiGetArgFlag(argv,"+stats"))
    Cmi_print_stats = 1;
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

static void log_init(void)
{
  log = (logent)malloc(50000 * sizeof(struct logent));
  _MEMCHECK(log);
  log_pos = 0;
  log_wrap = 0;
}

static void log_done(void)
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


static CmiNodeLock    Cmi_scanf_mutex;
static double         Cmi_clock;
static double         Cmi_check_delay = 3.0;

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

/************************ Win32 kernel SMP threads **************/
#if CMK_SHARED_VARS_NT_THREADS

CmiNodeLock CmiMemLock_lock;
static HANDLE comm_mutex;
#define CmiCommLockOrElse(x) /*empty*/
#define CmiCommLock() (WaitForSingleObject(comm_mutex, INFINITE))
#define CmiCommUnlock() (ReleaseMutex(comm_mutex))

static DWORD Cmi_state_key = 0xFFFFFFFF;
static CmiState     Cmi_state_vector = 0;

#ifdef CMK_OPTIMIZE
#  define CmiGetState() ((CmiState)TlsGetValue(Cmi_state_key))
#else
CmiState CmiGetState()
{
  CmiState result;
  result = (CmiState)TlsGetValue(Cmi_state_key);
  if(result == 0) PerrorExit("CmiGetState: TlsGetValue");
  return result;
}
#endif

CmiNodeLock CmiCreateLock(void)
{
  HANDLE hMutex = CreateMutex(NULL, FALSE, NULL);
  return hMutex;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  CloseHandle(lk);
}

void CmiYield(void) 
{ 
  Sleep(0);
}

#define CmiGetStateN(n) (Cmi_state_vector+(n))

static DWORD WINAPI comm_thread(LPVOID dummy)
{  
  if (Cmi_charmrun_fd!=-1)
    while (1) CommunicationServer(5);
  return 0;
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

static void CmiStartThreads(char **argv)
{
  int     i;
  DWORD   threadID;
  HANDLE  thr;
  int     val = 0;

  CmiMemLock_lock=CmiCreateLock();
  comm_mutex = CmiCreateLock();
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
/***************** Pthreads kernel SMP threads ******************/
#if CMK_SHARED_VARS_POSIX_THREADS_SMP

static pthread_key_t Cmi_state_key;
static CmiState     Cmi_state_vector;
CmiNodeLock CmiMemLock_lock;

#define CmiGetState() ((CmiState)pthread_getspecific(Cmi_state_key))

CmiNodeLock CmiCreateLock(void)
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

void CmiYield(void) { sched_yield(); }

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

static CmiNodeLock comm_mutex;

#define CmiCommLockOrElse(x) /*empty*/

#if 1
/*Regular comm. lock*/
#  define CmiCommLock() CmiLock(comm_mutex)
#  define CmiCommUnlock() CmiUnlock(comm_mutex)
#else
/*Verbose debugging comm. lock*/
static int comm_mutex_isLocked=0;
void CmiCommLock(void) {
	if (comm_mutex_isLocked) 
		CmiAbort("CommLock: already locked!\n");
	CmiLock(comm_mutex);
	comm_mutex_isLocked=1;
}
void CmiCommUnlock(void) {
	if (!comm_mutex_isLocked)
		CmiAbort("CommUnlock: double unlock!\n");
	comm_mutex_isLocked=0;
	CmiUnlock(comm_mutex);
}
#endif

static void comm_thread(void)
{
  while (1) CommunicationServer(5);
}

static void *call_startfn(void *vindex)
{
  int index = (int)vindex;
  CmiState state = Cmi_state_vector + index;
  pthread_setspecific(Cmi_state_key, state);
  ConverseRunPE(0);
  return 0;
}

static void CmiStartThreads(char **argv)
{
  pthread_t pid;
  int i, ok;
  pthread_attr_t attr;

  CmiMemLock_lock=CmiCreateLock();
  comm_mutex=CmiCreateLock();

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

/************************ No kernel SMP threads ***************/
#if CMK_SHARED_VARS_UNAVAILABLE

static volatile int memflag=0;
void CmiMemLock() { memflag++; }
void CmiMemUnlock() { memflag--; }

static volatile int comm_flag=0;
#define CmiCommLockOrElse(dothis) if (comm_flag!=0) dothis
#if 1
#  define CmiCommLock() (comm_flag=1)
#else
void CmiCommLock(void) {
  if (comm_flag!=0) CmiAbort("Comm lock *not* reentrant!\n");
  comm_flag=1;
}
#endif
#define CmiCommUnlock() (comm_flag=0)

static struct CmiStateStruct Cmi_state;
int Cmi_mype;
int Cmi_myrank;
#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

void CmiYield(void) { sleep(0); }

static void CommunicationInterrupt(int ignored)
{
  if (memflag) return;
  MACHSTATE(2,"--BEGIN SIGIO--")
  CommunicationServer(0);
  MACHSTATE(2,"--END SIGIO--")
}

extern void CmiSignal(int sig1, int sig2, int sig3, void (*handler)());

static void CmiStartThreads(char **argv)
{
  if ((Cmi_numpes != Cmi_numnodes) || (Cmi_mynodesize != 1))
    KillEveryone
      ("Multiple cpus unavailable, don't use cpus directive in nodesfile.\n");
  
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  Cmi_mype = Cmi_nodestart;
  Cmi_myrank = 0;
  
#if !CMK_ASYNC_NOT_NEEDED
  if (!Cmi_netpoll) {
    CmiSignal(SIGIO, 0, 0, CommunicationInterrupt);
    if (dataskt!=-1) CmiEnableAsyncIO(dataskt);
    if (Cmi_charmrun_fd!=-1) CmiEnableAsyncIO(Cmi_charmrun_fd);
  }
#endif
}

#endif

CpvDeclare(void *, CmiLocalQueue);
CpvStaticDeclare(char *, internal_printf_buffer);


#ifndef CmiMyPe
int CmiMyPe(void) 
{ 
  return CmiGetState()->pe; 
}
#endif
#ifndef CmiMyRank
int CmiMyRank(void)
{
  return CmiGetState()->rank;
}
#endif

/***************************************************************
 Communication with charmrun:
 We can send (ctrl_sendone) and receive (ctrl_getone)
 messages on a TCP socket connected to charmrun.
 This is used for printfs, CCS, etc; and also for
 killing ourselves if charmrun dies.
*/

/*This flag prevents simultanious outgoing
messages on the charmrun socket.  It is protected
by the commlock.*/
static int Cmi_charmrun_fd_sendflag=0;

/* ctrl_sendone */
static int sendone_abort_fn(int code,const char *msg) {
	fprintf(stderr,"Socket error %d in ctrl_sendone! %s\n",code,msg);
	machine_exit(1);
	return -1;
}

static void ctrl_sendone_nolock(const char *type,
				const char *data1,int dataLen1,
				const char *data2,int dataLen2)
{
  ChMessageHeader hdr;
  if (Cmi_charmrun_fd==-1) 
  	charmrun_abort("ctrl_sendone called in standalone!\n");
  Cmi_charmrun_fd_sendflag=1;
  ChMessageHeader_new(type,dataLen1+dataLen2,&hdr);
  skt_sendN(Cmi_charmrun_fd,(const char *)&hdr,sizeof(hdr));
  if (dataLen1>0) skt_sendN(Cmi_charmrun_fd,data1,dataLen1);
  if (dataLen2>0) skt_sendN(Cmi_charmrun_fd,data2,dataLen2);
  Cmi_charmrun_fd_sendflag=0;
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

static double Cmi_check_last;

static void pingCharmrun(int ignored) 
{
  double clock=GetClock();
  if (clock > Cmi_check_last + Cmi_check_delay) {
    MACHSTATE(1,"CommunicationsClock pinging charmrun");       
    Cmi_check_last = clock; 
    CmiCommLockOrElse(return;); /*Already busy doing communication*/
    if (Cmi_charmrun_fd_sendflag) return; /*Busy talking to charmrun*/
    CmiCommLock();
    ctrl_sendone_nolock("ping",NULL,0,NULL,0); /*Charmrun may have died*/
    CmiCommUnlock();
  }
}

/* periodic charm ping, for gm and netpoll */
static void pingCharmrunPeriodic(int ignored)
{
  pingCharmrun(ignored);
  CcdCallFnAfter(pingCharmrunPeriodic,NULL,1000);
}

static int ignore_further_errors(int c,const char *msg) {machine_exit(2);return -1;}
static void charmrun_abort(const char *s)
{
  if (Cmi_charmrun_fd==-1) {/*Standalone*/
  	fprintf(stderr,"Charm++ fatal error:\n%s\n",s);
  	abort();
  } else {
	char msgBuf[80];
  	skt_set_abort(ignore_further_errors);
	sprintf(msgBuf,"Fatal error on PE %d> ",CmiMyPe());
  	ctrl_sendone_nolock("abort",msgBuf,strlen(msgBuf),s,strlen(s)+1);
  }
}

/* ctrl_getone */

/*Add a message to this processor's receive queue */
static void CmiPushPE(int pe,void *msg)
{
  CmiState cs=CmiGetStateN(pe);
  MACHSTATE1(2,"Pushing message into %d's queue",pe);
  CmiIdleLock_addMessage(&cs->idle);
  PCQueuePush(cs->recv,msg);
}

static void ctrl_getone(void)
{
  ChMessage msg;
  ChMessage_recv(Cmi_charmrun_fd,&msg);

  if (strcmp(msg.header.type,"die")==0) {
    fprintf(stderr,"aborting: %s\n",msg.data);
    log_done();
    ConverseCommonExit();
    machine_exit(0);
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
	void *cmsg=(void *)CcsImpl_ccs2converse(hdr,msg.data+sizeof(CcsImplHeader),NULL);
	CmiPushPE(pe,cmsg);
#endif
  }  else {
  /* We do not use KillEveryOne here because it calls CmiMyPe(),
   * which is not available to the communication thread on an SMP version.
   */
    charmrun_abort("ERROR> Unrecognized message from charmrun.\n");
    machine_exit(1);
  }
  
  ChMessage_free(&msg);
}

#if CMK_CCS_AVAILABLE
/*Deliver this reply data to this reply socket.
  The data is forwarded to CCS server via charmrun.*/
void CcsImpl_reply(CcsImplHeader *hdr,int repLen,const void *repData)
{
  ctrl_sendone_locking("reply_fw",(const char *)hdr,sizeof(CcsImplHeader),
		       repData,repLen);  
}
#endif

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
  if(Cmi_syncprint) {
  	  ctrl_sendone_locking("printsync", buffer,strlen(buffer)+1,NULL,0);
	  CmiCommLock();
  	  ChMessage_recv(Cmi_charmrun_fd,&replymsg);
  	  ChMessage_free(&replymsg);
	  CmiCommUnlock();
  } else {
  	  ctrl_sendone_locking("print", buffer,strlen(buffer)+1,NULL,0);
  }
}

static void InternalError(f, l) char *f; va_list l;
{
  ChMessage replymsg;
  char *buffer = CpvAccess(internal_printf_buffer);
  vsprintf(buffer, f, l);
  if(Cmi_syncprint) {
  	  ctrl_sendone_locking("printerrsync", buffer,strlen(buffer)+1,NULL,0);
	  CmiCommLock();
  	  ChMessage_recv(Cmi_charmrun_fd,&replymsg);
  	  ChMessage_free(&replymsg);
	  CmiCommUnlock();
  } else {
  	  ctrl_sendone_locking("printerr", buffer,strlen(buffer)+1,NULL,0);
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
  if (Cmi_charmrun_fd!=-1)
    InternalPrintf(fmt, p);
  else
    vfprintf(stdout,fmt,p);
  va_end(p);
}

void CmiError(const char *fmt, ...)
{
  va_list p; va_start (p, fmt);
  if (Cmi_charmrun_fd!=-1)
    InternalError(fmt, p);
  else
    vfprintf(stderr,fmt,p);
  va_end(p);
}

int CmiScanf(const char *fmt, ...)
{
  va_list p; int i; va_start(p, fmt);
  i = InternalScanf(fmt, p);
  va_end(p);
  return i;
}


/***************************************************************************
 * Message Delivery:
 *
 ***************************************************************************/

#include "machine-dgram.c"


#ifndef CmiNodeFirst
int CmiNodeFirst(int node) { return nodes[node].nodestart; }
int CmiNodeSize(int node)  { return nodes[node].nodesize; }
#endif

#ifndef CmiNodeOf
int CmiNodeOf(int pe)      { return (nodes_by_pe[pe] - nodes); }
int CmiRankOf(int pe)      { return pe - (nodes_by_pe[pe]->nodestart); }
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

/*Note: node_addresses_obtain is called before starting
  threads, so no locks are needed (or valid!)*/
static void node_addresses_obtain(char **argv)
{
  ChMessage nodetabmsg; /* info about all nodes*/
  if (Cmi_charmrun_fd==-1) 
  {/*Standalone-- fake a single-node nodetab message*/
  	int npes=1;
  	int fakeLen=sizeof(ChSingleNodeinfo);
  	ChSingleNodeinfo *fakeTab;
	ChMessage_new("nodeinfo",sizeof(ChSingleNodeinfo),&nodetabmsg);
	fakeTab=(ChSingleNodeinfo *)(nodetabmsg.data);
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
	/*This is a stupid hack: we expect the *number* of nodes
	followed by ChNodeinfo structs; so we use a ChSingleNodeinfo
	(which happens to have exactly that layout!) and stuff
	a 1 into the "node number" slot
	*/
	fakeTab->nodeNo=ChMessageInt_new(1); /* <- hack */
	fakeTab->info.nPE=ChMessageInt_new(npes);
	fakeTab->info.dataport=ChMessageInt_new(0);
	fakeTab->info.IP=skt_invalid_ip;
  }
  else 
  { /*Contact charmrun for machine info.*/
	ChSingleNodeinfo me;

  	me.nodeNo=ChMessageInt_new(Cmi_mynode);
	/*The nPE and IP fields are set by charmrun--
	  these values don't matter.
	*/
	me.info.nPE=ChMessageInt_new(0);
	me.info.IP=skt_invalid_ip;
#if !CMK_USE_GM
  	me.info.dataport=ChMessageInt_new(dataport);
#else
        {
        /* get and send node id */
        unsigned int nodeid;
        if (gmport == NULL) nodeid = 0;
        else {
          gm_status_t status;
          status = gm_get_node_id(gmport, &nodeid);
          if (status != GM_SUCCESS) nodeid = 0;
        }
        me.info.dataport=ChMessageInt_new(nodeid);
        }
#endif

  	/*Send our node info. to charmrun.
  	CommLock hasn't been initialized yet-- 
  	use non-locking version*/
  	ctrl_sendone_nolock("initnode",(const char *)&me,sizeof(me),NULL,0);
  
  	/*We get the other node addresses from a message sent
  	  back via the charmrun control port.*/
  	if (!skt_select1(Cmi_charmrun_fd,600*1000)) CmiAbort("Timeout waiting for nodetab!\n");
  	ChMessage_recv(Cmi_charmrun_fd,&nodetabmsg);
  }
  node_addresses_store(&nodetabmsg);
  ChMessage_free(&nodetabmsg);
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
  case NODE_BROADCAST_ALL:
    PCQueuePush(CsvAccess(NodeRecv),CopyMsg(ogm->data,ogm->size));
    /*case-fallthrough (no break)-- deliver to all other processors*/
  case NODE_BROADCAST_OTHERS:
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
      CmiPushPE(rank,CopyMsg(ogm->data,ogm->size));
    }
    for (i=0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_BROADCAST);
    GarbageCollectMsg(ogm);
    break;
  case PE_BROADCAST_OTHERS:
    for (rank = 0; rank<Cmi_mynodesize; rank++)
      if (rank + Cmi_nodestart != ogm->src) {
	CmiPushPE(rank,CopyMsg(ogm->data,ogm->size));
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
	CmiPushPE(rank,CopyMsg(ogm->data,ogm->size));
	ogm->freemode = 'X';
      } else {
	CmiPushPE(rank, ogm->data);
	FreeOutgoingMsg(ogm);
      }
    }
  }
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
char *CmiGetNonLocalNodeQ(void)
{
  char *result = 0;
  if(!PCQueueEmpty(CsvAccess(NodeRecv))) {
    CmiLock(CsvAccess(CmiNodeRecvLock));
    result = (char *) PCQueuePop(CsvAccess(NodeRecv));
    CmiUnlock(CsvAccess(CmiNodeRecvLock));
  }
  return result;
}
#endif

void *CmiGetNonLocal(void)
{
  CmiState cs = CmiGetState();
  CmiIdleLock_checkMessage(&cs->idle);
  return (void *) PCQueuePop(cs->recv);
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
#if 1
/*CMK_SHARED_VARS_UNAVAILABLE*/
  CommunicationServer(0);
#endif
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
#if 1
/*CMK_SHARED_VARS_UNAVAILABLE*/
  CommunicationServer(0);
#endif
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
  CmiIdleState *s=CmiNotifyGetState();
  CmiState cs;
  char** CmiMyArgv;
  CmiNodeBarrier();
  cs = CmiGetState();
  CpvInitialize(char *, internal_printf_buffer);  
  CpvAccess(internal_printf_buffer) = (char *) malloc(PRINTBUFSIZE);
  _MEMCHECK(CpvAccess(internal_printf_buffer));
  CpvInitialize(void *,CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = cs->localqueue;
  CmiMyArgv=CmiCopyArgs(Cmi_argv);
  CthInit(CmiMyArgv);
#if MACHINE_DEBUG_LOG
  {
    char ln[200];
    sprintf(ln,"debugLog.%d",CmiMyPe());
    debugLog=fopen(ln,"w");
  }
#endif

  /* better to show the status here */
  if (Cmi_netpoll == 1 && CmiMyPe() == 0)
    CmiPrintf("Charm++: scheduler running in netpoll mode.\n");

#if CMK_USE_GM
  CmiCheckGmStatus();
#endif

  ConverseCommonInit(CmiMyArgv);
  
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,CmiNotifyBeginIdle,(void *)s);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,CmiNotifyStillIdle,(void *)s);
#if CMK_SHARED_VARS_UNAVAILABLE
  if (Cmi_netpoll) /*Repeatedly call CommServer*/
    CcdCallOnConditionKeep(CcdPERIODIC,CommunicationPeriodic,NULL);
  else /*Only need this for retransmits*/
    CcdCallOnConditionKeep(CcdPERIODIC_10ms,CommunicationPeriodic,NULL);
#endif

  if (CmiMyRank()==0 && Cmi_charmrun_fd!=-1) {
#if CMK_SHARED_VARS_UNAVAILABLE
    if (Cmi_netpoll == 1) {
    /* gm cannot live with setitimer */
    CcdCallFnAfter(pingCharmrunPeriodic,NULL,1000);
    }
    else {
    /*Occasionally ping charmrun, to test if it's dead*/
    struct itimerval i;
    CmiSignal(SIGALRM, 0, 0, pingCharmrun);
    i.it_interval.tv_sec = 1;
    i.it_interval.tv_usec = 0;
    i.it_value.tv_sec = 1;
    i.it_value.tv_usec = 0;
    setitimer(ITIMER_REAL, &i, NULL);
    }

#if ! CMK_USE_GM
    /*Occasionally check for retransmissions, outgoing acks, etc.*/
    /*no need in GM case */
    CcdCallFnAfter(CommunicationsClockCaller,NULL,Cmi_comm_clock_delay);
#endif
#endif
    
    /*Initialize the clock*/
    Cmi_clock=GetClock();
  }

  if (!everReturn) {
    Cmi_startfn(CmiGetArgc(CmiMyArgv), CmiMyArgv);
    if (Cmi_usrsched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

void ConverseExit(void)
{
  if (CmiMyRank()==0) {
    if(Cmi_print_stats)
      printNetStatistics();
    log_done();
    ConverseCommonExit();
    if (Cmi_charmrun_fd==-1) exit(0); /*Standalone version-- just leave*/
  }
  if (Cmi_charmrun_fd!=-1) {
  	ctrl_sendone_locking("ending",NULL,0,NULL,0); /* this causes charmrun to go away */
#if CMK_SHARED_VARS_UNAVAILABLE
 	while (1) CommunicationServer(500);
#endif
  }
/*Comm. thread will kill us.*/
  while (1) CmiYield();
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
  if(!Cmi_truecrash) {
    signal(SIGSEGV, KillOnAllSigs);
    signal(SIGFPE, KillOnAllSigs);
    signal(SIGILL, KillOnAllSigs);
    signal(SIGINT, KillOnAllSigs);
    signal(SIGTERM, KillOnAllSigs);
    signal(SIGABRT, KillOnAllSigs);
#   if !defined(_WIN32) || defined(__CYGWIN__) /*UNIX-only signals*/
    signal(SIGQUIT, KillOnAllSigs);
    signal(SIGBUS, KillOnAllSigs);
#     if CMK_HANDLE_SIGUSR
    signal(SIGUSR1, HandleUserSignals);
    signal(SIGUSR2, HandleUserSignals);
#     endif
#   endif /*UNIX*/
  }
}

/*Socket idle function to use before addresses have been
  obtained.  During the real program, we idle with CmiYield.
*/
static void obtain_idleFn(void) {sleep(0);}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usc, int everReturn)
{
#if MACHINE_DEBUG
  debugLog=stdout;
#endif
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  Cmi_argv = argv; Cmi_startfn = fn; Cmi_usrsched = usc;
  Cmi_netpoll = 0;
#if CMK_NETPOLL
  Cmi_netpoll = 1;
#endif
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  Cmi_idlepoll = 0;
#else
  Cmi_idlepoll = 1;
#endif
  Cmi_truecrash = 0;
  if (CmiGetArgFlag(argv,"+truecrash") ||
      CmiGetArgFlag(argv,"++debug")) Cmi_truecrash = 1;
    /* netpoll disable signal */
  if (CmiGetArgFlag(argv,"+netpoll")) Cmi_netpoll = 1;
    /* idlepoll use poll instead if sleep when idle */
  if (CmiGetArgFlag(argv,"+idlepoll")) Cmi_idlepoll = 1;
    /* idlesleep use sleep instead if busywait when idle */
  if (CmiGetArgFlag(argv,"+idlesleep")) Cmi_idlepoll = 0;
  Cmi_syncprint = CmiGetArgFlag(argv,"+syncprint");

  MACHSTATE2(5,"Init: (netpoll=%d), (idlepoll=%d)",Cmi_netpoll,Cmi_idlepoll);

  skt_init();
  atexit(exitDelay);
  parse_netstart();
  extract_args(argv);
  log_init();
  Cmi_scanf_mutex = CmiCreateLock();

  skt_set_idle(obtain_idleFn);
  if (!skt_ip_match(Cmi_charmrun_IP,skt_invalid_ip)) {
  	set_signals();
#if !CMK_USE_GM
  	dataskt=skt_datagram(&dataport, Cmi_os_buffer_size);
#else
        dataskt=-1;
#endif
  	Cmi_charmrun_fd = skt_connect(Cmi_charmrun_IP, Cmi_charmrun_port, 1800);
  } else {/*Standalone operation*/
  	printf("Charm++: standalone mode (not using charmrun)\n");
  	dataskt=-1;
  	Cmi_charmrun_fd=-1;
  }

  CmiMachineInit();

  node_addresses_obtain(argv);
  skt_set_idle(CmiYield);
  Cmi_check_delay = 2.0+0.5*Cmi_numnodes;
  if (Cmi_charmrun_fd==-1) /*Don't bother with check in standalone mode*/
	Cmi_check_delay=1.0e30;
  CmiStartThreads(argv);
  ConverseRunPE(everReturn);
}















