
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
 * In addition, every node has one of the following:
 *
 *   retransmit-queue (all datagram-structs that may need retransmitting)
 *
 * The send-queue is the starting point, when one wants to send
 * something, one allocates datagram-structs and pushes them onto a
 * send-queue.  As soon as one slot is available in the send-window,
 * one datagram is popped from the send-queue and transmitted.  The
 * transmitted datagram is then added to both the send-window and the
 * retransmit-queue.  It remains in the send-window until it is
 * acknowledged.  When acknowledged, the datagram-struct is flagged as
 * ``acknowledged'' and removed from the send-window.  Meanwhile, a
 * timer-based routine pops datagrams from the retransmit-queue.  If a
 * datagram is popped which has been flagged as ``acknowledged'', it
 * is freed.  If a datagram is popped which has not been flagged, it
 * is retransmitted and pushed back on the end of the retransmit
 * queue.
 *
 * There is a slight subtlety here.  When we transmit a datagram, the
 * OS may reply that the OS buffers are full.  In such a case, the
 * datagram is still considered ``transmitted.''  This causes them
 * to enter the send-window where they remain for a while.  Of course,
 * they don't get acknowledged, so they get retransmitted.  This
 * simple expedient means we don't have to add special code to handle
 * the OS buffering problem.  To make this work well, we set the
 * retransmit time for such packets very low.
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
 *   magic   (16 bits) --- magic number to make sure DG is good.
 *   dstrank ( 8 bits) --- destination processor rank.
 *   seqno   (24 bits) --- packet sequence number.
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
 * Almost all the transmission work is done by an interrupt routine.
 * The interrupt routine acts like a server process which does 99% of
 * the communication work.  The send-windows and other structures are
 * only accessed from within the interrupt routine.  Having a server
 * to handle the communication eliminates contention over the
 * send-windows and other communication structures.  Thus, no locking
 * of these structures is needed.
 *
 * For every PE-thread, there is one circular queue for that thread to
 * feed messages to the communication-interrupt.  And, for every
 * PE-thread, there is one circular queue for that thread to retrieve
 * messages from the communication-interrupt.  Thus, each circular
 * queue has one thread (or interrupt routine) as a producer, and one
 * thread as a consumer.  Locking is easily avoided in this case.
 * Since the circular queues are of finite size, this appears to be a
 * case of finite buffering. However, the interrupt routine does
 * infinite buffering, this turns out not to be a problem.
 *
 * The communication routine must watch for several events that can
 * happen spontaneously (from the point of view of the communication
 * thread).  Here is a complete list of the spontaneously-occurring
 * events that must be watched for: 1. A sender pushes a message into
 * its send-circular.  2. A recv-circular which was full becomes
 * not-full, making room for more received messages.  3. A datagram
 * appears on the datagram socket.  4. A timeout occurs, indicating
 * that a datagram needs to be retransmitted.
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

/*
 * I_Hate_C because the ansi prototype for a varargs function is incompatible
 * with the K&R definition of that varargs function.  Eg, this doesn't compile:
 *
 * void CmiPrintf(char *, ...);
 *
 * void CmiPrintf(va_alist) va_dcl
 * {
 *    ...
 * }
 *
 * I can't define the function in an ANSI way, because our stupid SUNs dont
 * yet have stdarg.h, even though they have gcc (which is ANSI).  So I have
 * to leave the definition of CmiPrintf as a K&R form, but I have to
 * deactivate the protos or the compiler barfs.  That's why I_Hate_C.
 *
 */

#define CmiPrintf I_Hate_C_1
#define CmiError  I_Hate_C_2
#define CmiScanf  I_Hate_C_3
#include "converse.h"
#undef CmiPrintf
#undef CmiError
#undef CmiScanf
void CmiPrintf();
void CmiError();
int  CmiScanf();

#include <sys/types.h>
#include <stdio.h>
#include <ctype.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <rpc/rpc.h>
#include <errno.h>
#include <setjmp.h>
#include <pwd.h>
#include <stdlib.h>
#include <signal.h>
#include <varargs.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/param.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <varargs.h>

#if CMK_STRINGS_USE_STRINGS_H
#include <strings.h>
#endif

#if CMK_STRINGS_USE_STRING_H
#include <string.h>
#endif

#if CMK_STRINGS_USE_OWN_DECLARATIONS
char *strchr(), *strrchr(), *strdup();
#endif

#if CMK_RSH_IS_A_COMMAND
#define RSH_CMD "rsh"
#endif
#if CMK_RSH_USE_REMSH
#define RSH_CMD "remsh"
#endif

static void KillEveryone();
static void KillEveryoneCode();
static void CommunicationServer();
extern int CmemInsideMem();
extern void CmemCallWhenMemAvail();

void *FIFO_Create();
void *FIFO_Peek(void *);
void  FIFO_Pop(void *);
void  FIFO_EnQueue(void *, void *);

/*****************************************************************************
 *
 *     Utility routines for network machine interface.
 *
 *
 * zap_newline(char *s)
 *
 *   - Remove the '\n' from the end of a string.
 *
 * char *skipblanks(char *s)
 *
 *   - advance pointer over blank characters
 *
 * char *skipstuff(char *s)
 *
 *   - advance pointer over nonblank characters
 *
 * char *strdupl(char *s)
 *
 *   - return a freshly-allocated duplicate of a string
 *
 *****************************************************************************/

static void zap_newline(s) char *s;
{
  char *p;
  p = s + strlen(s)-1;
  if (*p == '\n') *p = '\0';
}

static char *skipblanks(p) char *p;
{
  while ((*p==' ')||(*p=='\t')||(*p=='\n')) p++;
  return p;
}

static char *skipstuff(p) char *p;
{
  while ((*p)&&(*p!=' ')&&(*p!='\t')) p++;
  return p;
}

static char *strdupl(s) char *s;
{
  int len = strlen(s);
  char *res = (char *)malloc(len+1);
  strcpy(res, s);
  return res;
}

double GetClock()
{
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, NULL);
  if (ok<0) KillEveryoneCode(9343112);
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

char *CopyMsg(char *msg, int len)
{
  char *copy = (char *)CmiAlloc(len);
  memcpy(copy, msg, len);
  return copy;
}

static char *parseint(p, value) char *p; int *value;
{
  int val = 0;
  while (((*p)==' ')||((*p)=='.')) p++;
  if (((*p)<'0')||((*p)>'9')) KillEveryone("badly-formed number");
  while ((*p>='0')&&(*p<='9')) { val*=10; val+=(*p)-'0'; p++; }
  *value = val;
  return p;
}

static char *DeleteArg(argv)
char **argv;
{
  char *res = argv[0];
  if (res==0) KillEveryone("Illegal Arglist");
  while (*argv) { argv[0]=argv[1]; argv++; }
  return res;
}

static void jsleep(int sec, int usec)
{
  struct timeval tm;
  tm.tv_sec = sec;
  tm.tv_usec = usec;
  select(0,0,0,0,&tm);
}

static void writeall(int fd, char *buf, int size)
{
  int ok;
  while (size) {
    ok = write(fd, buf, size);
    if (ok<=0) KillEveryone("write on tcp socket failed.");
    size-=ok; buf+=ok;
  }
}

static int wait_readable(fd, sec) int fd; int sec;
{
  fd_set rfds;
  struct timeval tmo;
  int begin, nreadable;
  
  begin = time(0);
  FD_ZERO(&rfds);
  FD_SET(fd, &rfds);
  while(1) {
    tmo.tv_sec = (time(0) - begin) + sec;
    tmo.tv_usec = 0;
    nreadable = select(FD_SETSIZE, &rfds, NULL, NULL, &tmo);
    if ((nreadable<0)&&(errno==EINTR)) continue;
    if (nreadable == 0) { errno=ETIMEDOUT; return -1; }
    return 0;
  }
}

/******************************************************************************
 *
 * Configuration Data
 *
 * This data is all read in from the NETSTART variable (provided by the
 * host) and from the command-line arguments.  Once read in, it is never
 * modified.
 *
 * There should be an option, such as "++tune atm" or something like
 * that, which controls all the tuning parameters at once.
 *
 *****************************************************************************/

int           Cmi_numpes;    /* Total number of processors */
int           Cmi_nodesize;  /* Number of processors in my address space */
static int    Cmi_numnodes;  /* Total number of address spaces */
static int    Cmi_nodenum;   /* Which address space am I */
static int    Cmi_nodestart; /* First processor in this address space */
static char **Cmi_argv;
static int    Cmi_argc;
static int    Cmi_host_IP;
static int    Cmi_self_IP;
static int    Cmi_host_port;
static char   Cmi_host_IP_str[16];
static char   Cmi_self_IP_str[16];

static int    Cmi_max_dgram_size   = 2048;
static int    Cmi_os_buffer_size   = 50000;
static int    Cmi_window_size      = 50;
static double Cmi_delay_retransmit = 0.050;
static double Cmi_ack_delay        = 0.025;
static int    Cmi_dgram_max_data   = 2040;

static void parse_netstart()
{
  char *ns;
  int nread;
  ns = getenv("NETSTART");
  if (ns==0) goto abort;
  nread = sscanf(ns, "%d%d%d%d%d%d%d%d",
		 &Cmi_numnodes, &Cmi_nodenum,
		 &Cmi_nodestart, &Cmi_nodesize, &Cmi_numpes,
		 &Cmi_self_IP, &Cmi_host_IP, &Cmi_host_port);
  if (nread!=8) goto abort;
  sprintf(Cmi_self_IP_str,"%d.%d.%d.%d",
	  (Cmi_self_IP>>24)&0xFF,(Cmi_self_IP>>16)&0xFF,
	  (Cmi_self_IP>>8)&0xFF,Cmi_self_IP&0xFF);
  sprintf(Cmi_host_IP_str,"%d.%d.%d.%d",
	  (Cmi_host_IP>>24)&0xFF,(Cmi_host_IP>>16)&0xFF,
	  (Cmi_host_IP>>8)&0xFF,Cmi_host_IP&0xFF);
  return;
 abort:
  KillEveryone("program not started using 'conv-host' utility. aborting.\n");
  exit(1);
}

static void extract_args(argv)
char **argv;
{
}

/**************************************************************************
 *
 * SKT - socket routines
 *
 *
 * void skt_server(unsigned int *ppo, unsigned int *pfd)
 *
 *   - create a tcp server socket.  Performs the whole socket/bind/listen
 *     procedure.  Returns the the port of the socket and the file descriptor.
 *
 * void skt_datagram(unsigned int *ppo, unsigned int *pfd, int bufsize)
 *
 *   - creates a UDP datagram socket.  Performs the whole socket/bind/
 *     getsockname procedure.  Returns the port of the socket and
 *     the file descriptor.  Bufsize, if nonzero, controls the amount
 *     of buffer space the kernel sets aside for the socket.
 *
 * void skt_accept(int src,
 *                 unsigned int *pip, unsigned int *ppo, unsigned int *pfd)
 *
 *   - accepts a connection to the specified socket.  Returns the
 *     IP of the caller, the port number of the caller, and the file
 *     descriptor to talk to the caller.
 *
 * int skt_connect(unsigned int ip, int port, int timeout)
 *
 *   - Opens a connection to the specified server.  Returns a socket for
 *     communication.
 *
 *
 **************************************************************************/

static void skt_server(ppo, pfd)
unsigned int *ppo;
unsigned int *pfd;
{
  int fd= -1;
  int ok, len;
  struct sockaddr_in addr;
  
  retry: fd = socket(PF_INET, SOCK_STREAM, 0);
  if ((fd<0)&&(errno==EINTR)) goto retry;
  if (fd < 0) { perror("socket"); KillEveryoneCode(93483); }
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  ok = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  if (ok < 0) { perror("bind"); KillEveryoneCode(22933); }
  ok = listen(fd,5);
  if (ok < 0) { perror("listen"); KillEveryoneCode(3948); }
  len = sizeof(addr);
  ok = getsockname(fd, (struct sockaddr *)&addr, &len);
  if (ok < 0) { perror("getsockname"); KillEveryoneCode(93583); }

  *pfd = fd;
  *ppo = ntohs(addr.sin_port);
}

static void skt_datagram(ppo, pfd, bufsize)
unsigned int *ppo;
unsigned int *pfd;
unsigned int  bufsize;
{
  struct sockaddr_in name;
  int length, ok, skt;
  
  /* Create data socket */
  retry: skt = socket(AF_INET,SOCK_DGRAM,0);
  if ((skt<0)&&(errno==EINTR)) goto retry;
  if (skt < 0)
    { perror("socket"); KillEveryoneCode(8934); }
  name.sin_family = AF_INET;
  name.sin_port = 0;
  name.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(skt, (struct sockaddr *)&name, sizeof(name)) == -1)
    { perror("binding data socket"); KillEveryoneCode(2983); }
  length = sizeof(name);
  if (getsockname(skt, (struct sockaddr *)&name , &length))
    { perror("getting socket name"); KillEveryoneCode(39483); }

  if (bufsize) {
    int len = sizeof(int);
    ok = setsockopt(skt, SOL_SOCKET , SO_RCVBUF , (char *)&bufsize, len);
    if (ok < 0) KillEveryoneCode(35782);
    ok = setsockopt(skt, SOL_SOCKET , SO_SNDBUF , (char *)&bufsize, len);
    if (ok < 0) KillEveryoneCode(35783);
  }     

  *pfd = skt;
  *ppo = htons(name.sin_port);
}

static void skt_accept(src, pip, ppo, pfd)
int src;
unsigned int *pip;
unsigned int *ppo;
unsigned int *pfd;
{
  int i, fd, ok;
  struct sockaddr_in remote;
  i = sizeof(remote);
 acc:
  fd = accept(src, (struct sockaddr *)&remote, &i);
  if ((fd<0)&&(errno==EINTR)) goto acc;
  if (fd<0) { perror("accept"); KillEveryoneCode(39489); }
  *pip=htonl(remote.sin_addr.s_addr);
  *ppo=htons(remote.sin_port);
  *pfd=fd;
}

static int skt_connect(ip, port, seconds)
unsigned int ip; int port; int seconds;
{
  struct sockaddr_in remote; short sport=port;
  int fd, ok, len, retry, begin;
    
  /* create an address structure for the server */
  memset(&remote, 0, sizeof(remote));
  remote.sin_family = AF_INET;
  remote.sin_port = htons(sport);
  remote.sin_addr.s_addr = htonl(ip);
    
  begin = time(0); ok= -1;
  while (time(0)-begin < seconds) {
  sock:
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if ((fd<0)&&(errno==EINTR)) goto sock;
    if (fd < 0) { perror("socket"); exit(1); }
    
  conn:
    ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
    if (ok>=0) break;
    close(fd);
    switch (errno) {
    case EINTR: break;
    case ECONNREFUSED: jsleep(1,0); break;
    case EADDRINUSE: jsleep(1,0); break;
    case EADDRNOTAVAIL: jsleep(5,0); break;
    default: KillEveryone(strerror(errno));
    }
  }
  if (ok<0) {
    KillEveryone(strerror(errno)); exit(1);
  }
  return fd;
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

PCQueue PCQueueCreate()
{
  CircQueue circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));
  PCQueue Q = (PCQueue)malloc(sizeof(struct PCQueueStruct));
  Q->head = circ;
  Q->tail = circ;
  return Q;
}

char *PCQueuePop(PCQueue Q)
{
  CircQueue circ; int pull; char *data;

  while (1) {
    circ = Q->head;
    pull = circ->pull;
    data = circ->data[pull];
    if (data) {
      circ->pull = (pull + 1) & (PCQueueSize-1);
      circ->data[pull] = 0;
      return data;
    }
    if (Q->tail == circ)
      return 0;
    Q->head = circ->next;
    free(circ);
  }
}

void PCQueuePush(PCQueue Q, char *data)
{
  CircQueue circ; int push;
  
  circ = Q->tail;
  push = circ->push;
  if (circ->data[push] == 0) {
    circ->data[push] = data;
    circ->push = (push + 1) & (PCQueueSize-1);
    return;
  }
  circ = (CircQueue)calloc(1, sizeof(struct CircQueueStruct));
  circ->push = 1;
  circ->data[0] = data;
  Q->tail->next = circ;
  Q->tail = circ;
}

/*****************************************************************************
 *
 * CmiAlloc, CmiSize, and CmiFree
 *
 * Note: this allocator is only used for messages.  Everything else
 * is allocated with the less-expensive ``malloc''.
 *
 *****************************************************************************/

void *CmiAlloc(size)
int size;
{
  char *res;
  res =(char *)malloc(size+8);
  if (res==0) KillEveryone("Memory allocation failed.");
  ((int *)res)[0]=size;
  return (void *)(res+8);
}

int CmiSize(blk)
void *blk;
{
  return ((int *)(((char *)blk)-8))[0];
}

void CmiFree(blk)
void *blk;
{
  free(((char *)blk)-8);
}


/*****************************************************************************
 *                                                                           
 * Neighbour-Lookup functions.                                               
 *                                                                           
 * the neighbour information is computed dynamically.  It imposes a
 * (maybe partial) hypercube on the machine.
 *                                                                           
 *****************************************************************************/
 
long CmiNumNeighbours(node)
int node;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < CmiNumPes()) count++;
    bit = bit<<1; 
    if (bit > CmiNumPes()) break;
  }
  return count;
}
 
int CmiGetNodeNeighbours(node, neighbours)
int node, *neighbours;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < CmiNumPes()) neighbours[count++] = neighbour;
    bit = bit<<1; 
    if (bit > CmiNumPes()) break;
  }
  return count;
}
 
int CmiNeighboursIndex(node, nbr)
int node, nbr;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < CmiNumPes()) { if (nbr==neighbour) return count; count++; }
    bit = bit<<=1; 
    if (bit > CmiNumPes()) break;
  }
  return(-1);
}

/*****************************************************************************
 *
 * Communication Structures
 *
 *****************************************************************************/

#define DGRAM_HEADER_SIZE 8
#define DGRAM_MAGIC 0x1234

#define CmiMsgHeaderSetLength(msg, len) (((int*)(msg))[2] = (len))
#define CmiMsgHeaderGetLength(msg)      (((int*)(msg))[2])
#define CmiMsgNext(msg) (*((void**)(msg)))

#define DGRAM_SRCPE_MASK    (0xFFFF)
#define DGRAM_MAGIC_MASK    (0xFFFF)
#define DGRAM_SEQNO_MASK    (0xFFFFFF)

#define DGRAM_DSTRANK_MAX   (0xFC)
#define DGRAM_SIMPLEKILL    (0xFD)
#define DGRAM_BROADCAST     (0xFE)
#define DGRAM_ACKNOWLEDGE   (0xFF)

typedef struct { char data[DGRAM_HEADER_SIZE]; } DgramHeader;

#define DgramHeaderMake(ptr, dstrank, srcpe, magic, seqno) { \
   ((unsigned short *)ptr)[0] = srcpe; \
   ((unsigned short *)ptr)[1] = magic; \
   ((unsigned int *)ptr)[1] = (seqno<<8) | dstrank; \
}

#define DgramHeaderBreak(ptr, dstrank, srcpe, magic, seqno) { \
   unsigned int tmp; \
   srcpe = ((unsigned short *)ptr)[0]; \
   magic = ((unsigned short *)ptr)[1]; \
   tmp = ((unsigned int *)ptr)[1]; \
   dstrank = (tmp&0xFF); seqno = (tmp>>8); \
}

#define PE_BROADCAST_OTHERS (-1)
#define PE_BROADCAST_ALL    (-2)


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
  unsigned int  len;
  unsigned char data[1];
}
*ExplicitDgram;

typedef struct ImplicitDgramStruct
{
  struct ImplicitDgramStruct *next;
  struct OtherNodeStruct *dest;
  int srcpe, rank, seqno;
  char  *dataptr;
  int    datalen;
  double nextxmit;
  int    acknowledged;
  OutgoingMsg ogm;
}
*ImplicitDgram;

typedef struct OtherNodeStruct
{
  int nodestart, nodesize;
  unsigned int IP, dataport, ctrlport;
  struct sockaddr_in addr;
  
  int                      ack_count;
  double                   ack_time;
  
  int                      asm_rank;
  int                      asm_total;
  int                      asm_fill;
  char                    *asm_msg;
  
  ExplicitDgram           *recv_window;
  unsigned int             recv_next;
  
  ImplicitDgram           *send_window;
  void                    *send_queue;
  unsigned int             send_next;
}
*OtherNode;

typedef struct CmiStateStruct
{
  int pe, rank;
  PCQueue send;
  PCQueue recv;
  void *localqueue;
}
*CmiState;

void CmiStateInit(int pe, int rank, CmiState state)
{
  state->pe = pe;
  state->rank = rank;
  state->send = PCQueueCreate();
  state->recv = PCQueueCreate();
  state->localqueue = FIFO_Create();
}


static ImplicitDgram Cmi_freelist_implicit;
static ExplicitDgram Cmi_freelist_explicit;
static OutgoingMsg   Cmi_freelist_outgoing;

#define FreeImplicitDgram(dg) {\
  ImplicitDgram d=(dg);\
  d->next = Cmi_freelist_implicit;\
  Cmi_freelist_implicit = d;\
}

#define MallocImplicitDgram(dg) {\
  ImplicitDgram d = Cmi_freelist_implicit;\
  if (d==0) d = ((ImplicitDgram)malloc(sizeof(struct ImplicitDgramStruct)));\
  else Cmi_freelist_implicit = d->next;\
  dg = d;\
}

#define FreeExplicitDgram(dg) {\
  ExplicitDgram d=(dg);\
  d->next = Cmi_freelist_explicit;\
  Cmi_freelist_explicit = d;\
}

#define MallocExplicitDgram(dg) {\
  ExplicitDgram d = Cmi_freelist_explicit;\
  if (d==0) d = ((ExplicitDgram)malloc \
		   (sizeof(struct ExplicitDgramStruct) + Cmi_max_dgram_size));\
  else Cmi_freelist_explicit = d->next;\
  dg = d;\
}

/* Careful with these next two, need concurrency control */

#define FreeOutgoingMsg(m) (free(m))
#define MallocOutgoingMsg(m)\
   (m=(OutgoingMsg)malloc(sizeof(struct OutgoingMsgStruct)))

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

char *log;
int   log_size;

static void log_init()
{
  log = (char *)malloc(10001000);
  log_size = 0;
}

static void log_done()
{
  char buffer[1000]; int f;
  sprintf(buffer, "log.%d", Cmi_nodenum);
  f = open(buffer, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (f<0) { perror("open"); exit(1); }
  writeall(f, log, log_size);
  close(f);
}

static int log_printf(va_alist) va_dcl
{
  char *f; int len;
  va_list p;
  va_start(p);
  f = va_arg(p, char *);
  if (log_size >= 1000000) return;
  vsprintf(log+log_size, f, p);
  log_size += strlen(log+log_size);
}

#define LOG(x) log_printf x

#endif


#if !LOGGING

#define log_init() 0
#define log_done() 0
#define LOG(x) 0

#endif

/******************************************************************************
 *
 * Node state
 *
 *****************************************************************************/

static int        ctrlport, dataport, ctrlskt, dataskt;

static OtherNode *nodes_by_pe;  /* OtherNodes indexed by processor number */
static OtherNode  nodes;        /* Indexed only by ``node number'' */

static int        Cmi_shutdown_done;
static CmiMutex   Cmi_scanf_mutex;
static char      *Cmi_scanf_data;
static int        Cmi_shutdown_done; 
static int        Cmi_clock;

static void *transmit_queue;
static void *retransmit_queue;
static void *ack_queue;

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
static int ctrlskt_ready_write;
static int dataskt_ready_read;
static int dataskt_ready_write;

void CheckSocketsReady()
{
  static fd_set rfds; 
  static fd_set wfds; 
  struct timeval tmo;
  int nreadable;
  
  FD_SET(dataskt, &rfds);
  FD_SET(ctrlskt, &rfds);
  FD_SET(dataskt, &wfds);
  FD_SET(ctrlskt, &wfds);
  tmo.tv_sec = 0;
  tmo.tv_usec = 0;
  nreadable = select(FD_SETSIZE, &rfds, &wfds, NULL, &tmo);
  if (nreadable <= 0) {
    ctrlskt_ready_read = 0;
    ctrlskt_ready_write = 0;
    dataskt_ready_read = 0;
    dataskt_ready_write = 0;
    return;
  }
  ctrlskt_ready_read = (FD_ISSET(ctrlskt, &rfds));
  dataskt_ready_read = (FD_ISSET(dataskt, &rfds));
  ctrlskt_ready_write = (FD_ISSET(ctrlskt, &wfds));
  dataskt_ready_write = (FD_ISSET(dataskt, &wfds));
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
 * switch_to_multithreaded_mode()
 *
 *    Allocates one CmiState structure per PE.  Initializes all of
 *    the CmiState structures using the function CmiStateInit.
 *    Then, starts all the threads: one per PE, plus a communication
 *    thread.  The communication thread must be an infinite loop
 *    that calls the function CommunicationServer over and over, with
 *    a short delay between each call (ideally, the delay should end
 *    when a datagram arrives or when somebody notifies: see below).
 *    switch_to_multithreaded_mode never returns.
 *
 * NotifyCommunicationThread
 *
 *    Used by the PE's to notify the communication thread that a message
 *    has just been inserted into a send-queue.  This may cause the
 *    communication thread to pick up the message and packetize it more
 *    quickly than if the message is simply deposited in the send-queue
 *    without notification.
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
 * CmiMutex, CmiMutexLock(), CmiMutexUnlock()
 *
 *    Lock/Unlock a mutex.  These are in converse.h
 *
 * CmiMemLock() and CmiMemUnlock()
 *
 *    The memory module calls these functions to obtain mutual exclusion
 *    in the memory routines, and to keep interrupts from reentering malloc.
 *
 * CmiMyPe() and CmiMyRank()
 *
 *    The usual.  Implemented here, since a highly-optimized version
 *    is possible in the nonthreaded case.
 *
 *****************************************************************************/

#if CMK_SHARED_VARS_SUN_THREADS

static mutex_t memmutex;
void CmiMemLock() { mutex_lock(&memmutex); }
void CmiMemUnlock() { mutex_unlock(&memmutex); }
#define CmiMemBusy() 0

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

#define CmiGetStateN(n) (Cmi_state_vector+(n))

static void *call_user_main(void *vindex)
{
  int index = (int)vindex;
  CmiState state = Cmi_state_vector + index;
  thr_setspecific(Cmi_state_key, state);
  state->pe = Cmi_nodestart + index;
  state->rank = index;
  user_main(Cmi_argc, Cmi_argv);
  thr_exit(0);
}

static void switch_to_multithreaded_mode()
{
  int i, pid, ok; struct timeval tmo; fd_set rfds;
  
  thr_keycreate(&Cmi_state_key, 0);
  Cmi_state_vector =
    (CmiState)calloc(Cmi_nodesize, sizeof(struct CmiStateStruct));
  for (i=0; i<Cmi_nodesize; i++)
    CmiStateInit(i+Cmi_nodestart, i, CmiGetStateN(i));
  for (i=0; i<Cmi_nodesize; i++) {
    ok = thr_create(0, 256000, call_user_main, (void *)i,
	       THR_DETACHED|THR_BOUND, &pid);
    if (ok<0) { perror("thr_create"); exit(1); }
  }
  CmiSignal(SIGALRM, SIG_IGN);
  while (Cmi_shutdown_done == 0) {
    CommunicationServer();
    tmo.tv_sec = 0;
    tmo.tv_usec = 10000;
    FD_ZERO(&rfds);
    FD_SET(dataskt, &rfds);
    FD_SET(ctrlskt, &rfds);
    select(FD_SETSIZE, &rfds, 0, 0, &tmo);
  }
  thr_exit(0);
}

static double last_notify;

static void NotifyCommunicationThread()
{
  struct sockaddr_in addr; double now;
  if (dataskt_ready_write == 0) return;
  now = GetClock();
  if (now < last_notify + 0.005) return;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(dataport);
  addr.sin_addr.s_addr = htonl(0x7F000001);
  sendto(dataskt, "x", 1, 0, (struct sockaddr *)&addr, sizeof(addr));
}

#endif

#if CMK_SHARED_VARS_UNAVAILABLE

static int memflag;
void CmiMemLock() { memflag=1; }
void CmiMemUnlock() { memflag=0; }
#define CmiMemBusy() (memflag)

static struct CmiStateStruct Cmi_state;
int Cmi_mype;
int Cmi_myrank;
#define CmiGetState() (&Cmi_state)
#define CmiGetStateN(n) (&Cmi_state)

static void switch_to_multithreaded_mode()
{
  struct itimerval i;

  if ((Cmi_numpes != Cmi_numnodes) || (Cmi_nodesize != 1))
    KillEveryone
      ("Multiple cpus unavailable, don't use cpus directive in nodesfile.\n");
  
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  Cmi_mype = Cmi_nodestart;
  Cmi_myrank = 0;
  
  CmiSignal(SIGALRM, CommunicationServer);
  CmiSignal(SIGIO,   CommunicationServer);
  CmiEnableAsyncIO(dataskt);
  CmiEnableAsyncIO(ctrlskt);
  i.it_interval.tv_sec = 0;
  i.it_interval.tv_usec = 10000;
  i.it_value.tv_sec = 0;
  i.it_value.tv_usec = 10000;
  setitimer(ITIMER_REAL, &i, NULL);
  
  user_main(Cmi_argc, Cmi_argv);
  exit(0);
}

static void NotifyCommunicationThread()
{
  struct itimerval value;
  if (dataskt_ready_write == 0) return;
  getitimer(ITIMER_REAL, &value);
  if (value.it_value.tv_usec > 9000) return;
  value.it_value.tv_usec = 10000;
  setitimer(ITIMER_REAL, &value, 0);
  kill(0, SIGALRM);
}

#endif

CpvDeclare(void *, CmiLocalQueue);

/****************************************************************************
 *                                                                          
 * Fast shutdown                                                            
 *                                                                          
 ****************************************************************************/

static void KillIndividual(int ip,int port,int timeout,char *cmd,char *flag)
{
  int fd;
  if (*flag) return;
  fd = skt_connect(ip, port, timeout);
  if (fd>=0) {
    writeall(fd, cmd, strlen(cmd));
    close(fd);
    *flag = 1;
  }
}

static void KillEveryone(msg)
char *msg;
{
  char cmd[1024]; char *killed; char host=0; int i;
  if (nodes == 0) {
    fprintf(stderr,"%s\n", msg);
    exit(1);
  }
  sprintf(cmd,"die %s",msg);
  killed = (char *)calloc(1, Cmi_numnodes);
  KillIndividual(Cmi_host_IP, Cmi_host_port, 2, cmd, &host);
  for (i=0; i<Cmi_numnodes; i++)
    KillIndividual(nodes[i].IP, nodes[i].ctrlport, 2, cmd, killed+i);
  KillIndividual(Cmi_host_IP, Cmi_host_port, 10, cmd, &host);
  for (i=0; i<Cmi_numnodes; i++)
    KillIndividual(nodes[i].IP, nodes[i].ctrlport, 10, cmd, killed+i);
  KillIndividual(Cmi_host_IP, Cmi_host_port, 30, cmd, &host);
  for (i=0; i<Cmi_numnodes; i++)
    KillIndividual(nodes[i].IP, nodes[i].ctrlport, 30, cmd, killed+i);
  exit(1);
}

static void KillEveryoneCode(n)
int n;
{
  char buffer[1024];
  sprintf(buffer,"Internal error #%d (node %d)\n(Contact CHARM developers)\n",
	  n,CmiMyPe());
  KillEveryone(buffer);
}

static void KillOnSegv()
{
  char buffer[1024];
  sprintf(buffer, "Node %d: Segmentation Fault.\n",CmiMyPe());
  KillEveryone(buffer);
}

static void KillOnBus()
{
  char buffer[1024];
  sprintf(buffer, "Node %d: Bus Error.\n",CmiMyPe());
  KillEveryone(buffer);
}

static void KillOnIntr()
{
  char buffer[1024];
  sprintf(buffer, "Node %d: Interrupted.\n",CmiMyPe());
  KillEveryone(buffer);
}

static void KillOnCrash()
{
  char buffer[1024];
  sprintf(buffer, "Node %d: Crashed.\n",CmiMyPe());
  KillEveryone(buffer);
}

static void KillInit()
{
  CmiSignal(SIGSEGV, KillOnSegv);
  CmiSignal(SIGBUS,  KillOnBus);
  CmiSignal(SIGILL,  KillOnCrash);
  CmiSignal(SIGABRT, KillOnCrash);
  CmiSignal(SIGFPE,  KillOnCrash);
  CmiSignal(SIGPIPE, KillOnCrash);
  CmiSignal(SIGURG,  KillOnCrash);

#ifdef SIGSYS
  CmiSignal(SIGSYS,  KillOnCrash);
#endif

  CmiSignal(SIGTERM, KillOnIntr);
  CmiSignal(SIGQUIT, KillOnIntr);
  CmiSignal(SIGINT,  KillOnIntr);
}

/****************************************************************************
 *
 * ctrl_sendone
 *
 * Any thread can call this.  There's no need for concurrency control.
 *
 ****************************************************************************/

static void ctrl_sendone(va_alist) va_dcl
{
  char buffer[1024];
  char *f; int fd, delay;
  va_list p;
  va_start(p);
  delay = va_arg(p, int);
  f = va_arg(p, char *);
  vsprintf(buffer, f, p);
  fd = skt_connect(Cmi_host_IP, Cmi_host_port, delay);
  if (fd<0) KillEveryone("cannot contact host");
  writeall(fd, buffer, strlen(buffer));
  shutdown(fd, 1);
  while (read(fd, buffer, 1023)>0);
  close(fd);
}

/****************************************************************************
 *
 * ctrl_getone
 *
 * This is handled only by the communication interrupt.
 *
 ****************************************************************************/

static void node_addresses_store();

static void ctrl_getone()
{
  char line[10000];
  int ok, ip, port, fd;  FILE *f;
  skt_accept(ctrlskt, &ip, &port, &fd);
  f = fdopen(fd,"r");
  while (fgets(line, 9999, f)) {
    if (strncmp(line,"aval addr ",10)==0) {
      node_addresses_store(line);
    }
    else if (strncmp(line,"aval done ",10)==0) {
      Cmi_shutdown_done = 1;
    }
    else if (strncmp(line,"scanf-data ",11)==0) {
      Cmi_scanf_data=strdupl(line+11);
    } else if (strncmp(line,"die ",4)==0) {
      fprintf(stderr,"aborting: %s\n",line+4);
      exit(0);
    }
    else KillEveryoneCode(2932);
  }
  fclose(f);
  close(fd);
}

/*****************************************************************************
 *
 * node_addresses
 *
 *  These two functions fill the node-table.
 *
 *
 *   This node, like all others, first sends its own address to the host
 *   using this command:
 *
 *     aset addr <my-nodeno> <ip-addr>.<ctrlport>.<dataport>.<nodestart>.<nodesize>
 *
 *   Then requests all addresses from the host using this command:
 *
 *     aget <my-ip-addr> <my-ctrlport> addr 0 <numnodes>
 *
 *   when the host has all the addresses, he sends a table to me:
 *
 *     aval addr <ip-addr-0>.<ctrlport-0>.<dataport-0> ...
 *
 *****************************************************************************/

static void node_addresses_obtain()
{
  ctrl_sendone(120, "aset addr %d %s.%d.%d.%d.%d\n",
	       Cmi_nodenum, Cmi_self_IP_str, ctrlport, dataport,
	       Cmi_nodestart, Cmi_nodesize);
  ctrl_sendone(120, "aget %s %d addr 0 %d\n",
	       Cmi_self_IP_str,ctrlport,Cmi_numnodes-1);
  while (nodes == 0) {
    if (wait_readable(ctrlskt, 300)<0)
      { perror("waiting for data"); KillEveryoneCode(21323); }
    ctrl_getone();
  }
}

static void node_addresses_store(addrs) char *addrs;
{
  char *p, *e; int i, j, lo, hi;
  OtherNode ntab, *bype;
  if (strncmp(addrs,"aval addr ",10)!=0) KillEveryoneCode(83473);
  ntab = (OtherNode)calloc(Cmi_numnodes, sizeof(struct OtherNodeStruct));
  p = skipblanks(addrs+10);
  p = parseint(p,&lo);
  p = parseint(p,&hi);
  if ((lo!=0)||(hi!=Cmi_numnodes - 1)) KillEveryoneCode(824793);
  for (i=0; i<Cmi_numnodes; i++) {
    unsigned int ip0,ip1,ip2,ip3,cport,dport,nodestart,nodesize,ip;
    p = parseint(p,&ip0);
    p = parseint(p,&ip1);
    p = parseint(p,&ip2);
    p = parseint(p,&ip3);
    p = parseint(p,&cport);
    p = parseint(p,&dport);
    p = parseint(p,&nodestart);
    p = parseint(p,&nodesize);
    ip = (ip0<<24)+(ip1<<16)+(ip2<<8)+ip3;
    ntab[i].nodestart = nodestart;
    ntab[i].nodesize  = nodesize;
    ntab[i].IP = ip;
    ntab[i].ctrlport = cport;
    ntab[i].dataport = dport;
    ntab[i].addr.sin_family      = AF_INET;
    ntab[i].addr.sin_port        = htons(dport);
    ntab[i].addr.sin_addr.s_addr = htonl(ip);
  }
  p = skipblanks(p);
  if (*p!=0) KillEveryoneCode(82283);
  bype = (OtherNode*)malloc(Cmi_numpes * sizeof(OtherNode));
  for (i=0; i<Cmi_numnodes; i++) {
    OtherNode node = ntab + i;
    node->send_queue = FIFO_Create();
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
  char *p, *buf;
  char buffer[8192];
  vsprintf(buffer, f, l);
  buf = buffer;
  while (*buf) {
    p = strchr(buf, '\n');
    if (p) {
      *p=0; ctrl_sendone(120, "print %s\n", buf);
      *p='\n'; buf=p+1;
    } else {
      ctrl_sendone(120, "princ %s\n", buf);
      break;
    }
  }
}

static void InternalError(f, l) char *f; va_list l;
{
  char *p, *buf;
  char buffer[8192];
  vsprintf(buffer, f, l);
  buf = buffer;
  while (*buf) {
    p = strchr(buf, '\n');
    if (p) {
      *p=0; ctrl_sendone(120, "printerr %s\n", buf);
      *p='\n'; buf = p+1;
    } else {
      ctrl_sendone(120, "princerr %s\n", buf);
      break;
    }
  }
}

static int InternalScanf(fmt, l)
    char *fmt;
    va_list l;
{
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
  CmiMutexLock(&Cmi_scanf_mutex);
  ctrl_sendone(120, "scanf %s %d %s", Cmi_self_IP_str, ctrlport, fmt);
  while (Cmi_scanf_data == 0) jsleep(0, 250000);
  i = sscanf(Cmi_scanf_data, fmt,
	     ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3], ptr[ 4], ptr[ 5],
	     ptr[ 6], ptr[ 7], ptr[ 8], ptr[ 9], ptr[10], ptr[11],
	     ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17]);
  free(Cmi_scanf_data);
  Cmi_scanf_data=0;
  CmiMutexUnlock(&Cmi_scanf_mutex);
  return i;
}

void CmiPrintf(va_alist) va_dcl
{
  va_list p; char *f; va_start(p); f = va_arg(p, char *);
  InternalPrintf(f, p);
}

void CmiError(va_alist) va_dcl
{
  va_list p; char *f; va_start(p); f = va_arg(p, char *);
  InternalError(f, p);
}

int CmiScanf(va_alist) va_dcl
{
  va_list p; char *f; va_start(p); f = va_arg(p, char *);
  return InternalScanf(f, p);
}


/******************************************************************************
 *
 * Transmission Code
 *
 *****************************************************************************/

int TransmitAcknowledgement()
{
  OtherNode node; int i, extra;
  struct { DgramHeader head; int others[1024]; } ack;
  
  node = (OtherNode)FIFO_Peek(ack_queue);
  if (node == 0) return 0;
  if (Cmi_clock < node->ack_time) return 0;
  FIFO_Pop(ack_queue);
  
  DgramHeaderMake(&ack, DGRAM_ACKNOWLEDGE,
		  Cmi_nodestart, 0x1234, node->recv_next);
  extra = 0;
  for (i=0; i<Cmi_window_size; i++) {
    if (node->recv_window[i]) {
      ack.others[extra] = node->recv_window[i]->seqno;
      extra++;
    }
  }
  LOG(("%1.4f %d A %d %d",
       Cmi_clock, Cmi_nodestart,node->nodestart,node->recv_next));
  for (i=0; i<Cmi_window_size; i++)
    if (node->recv_window[i])
      LOG((" %d", node->recv_window[i]->seqno));
  LOG(("\n"));
  sendto(dataskt, (char *)&ack,
	 DGRAM_HEADER_SIZE + extra * sizeof(int), 0,
	 (struct sockaddr *)&(node->addr),
	 sizeof(struct sockaddr_in));
  node->ack_time = 0.0;
  node->ack_count = 0;
  return 1;
}

void ScheduleAcknowledgement(OtherNode node)
{
  node->ack_count++;
  if (node->ack_count == 1) {
    node->ack_time = Cmi_clock + Cmi_ack_delay;
    FIFO_EnQueue(ack_queue, (void *)node);
  }
}

void DiscardImplicitDgram(ImplicitDgram dg)
{
  OutgoingMsg ogm;
  ogm = dg->ogm;
  ogm->refcount--;
  if (ogm->refcount == 0) {
    if (ogm->freemode == 'A') {
      ogm->freemode = 'X';
    } else {
      CmiFree(ogm->data);
      FreeOutgoingMsg(ogm);
    }
  }
  FreeImplicitDgram(dg);
}

void TransmitImplicitDgram(ImplicitDgram dg)
{
  char *data; DgramHeader *head; int ok, len; DgramHeader temp;
  OtherNode dest;

  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, DGRAM_MAGIC, dg->seqno);
  LOG(("%1.4f %d T %d %d\n",
       Cmi_clock, Cmi_nodestart,dest->nodestart,dg->seqno));
  ok = sendto(dataskt, (char *)head, len + DGRAM_HEADER_SIZE, 0,
	      (struct sockaddr *)&(dest->addr), sizeof(struct sockaddr_in));
  *head = temp;
}

int TransmitDatagram()
{
  ImplicitDgram dg; OtherNode node; 
  
  FIFO_DeQueue(transmit_queue, &dg);
  if (dg) {
    TransmitImplicitDgram(dg);
    dg->nextxmit = Cmi_clock + Cmi_delay_retransmit;
    FIFO_EnQueue(retransmit_queue, (void *)dg);
    return 1;
  }
  
  while (1) {
    dg = (ImplicitDgram)FIFO_Peek(retransmit_queue);
    if (dg == 0) return 0;
    if (dg->acknowledged) {
      FIFO_Pop(retransmit_queue);
      DiscardImplicitDgram(dg);
      continue;
    }
    if (dg->nextxmit > Cmi_clock) return 0;
    FIFO_Pop(retransmit_queue);
    TransmitImplicitDgram(dg);
    dg->nextxmit = Cmi_clock + Cmi_delay_retransmit;
    FIFO_EnQueue(retransmit_queue, (void *)dg);
    return 1;
  }
}

void MoveDatagramsIntoSendWindow(OtherNode node)
{
  ImplicitDgram dg; unsigned int seqno, slot; int ok;
  
  while (1) {
    dg = (ImplicitDgram)FIFO_Peek(node->send_queue);
    if (dg == 0) break;
    seqno = dg->seqno;
    slot = seqno % Cmi_window_size;
    if (node->send_window[slot]) break;
    FIFO_Pop(node->send_queue);
    node->send_window[slot] = dg;
    dg->nextxmit = 0;
    FIFO_EnQueue(transmit_queue, (void *)dg);
  }
}

void EnqueueOutgoingDgram
        (OutgoingMsg ogm, char *ptr, int len, OtherNode node, int rank)
{
  unsigned int seqno, slot; int dst, src, dstrank; ImplicitDgram dg;
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
  dg->acknowledged = 0;
  dg->ogm = ogm;
  ogm->refcount++;
  FIFO_EnQueue(node->send_queue, (void *)dg);
}

void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank)
{
  int size, seqno; char *data;
  
  size = ogm->size - DGRAM_HEADER_SIZE;
  data = ogm->data + DGRAM_HEADER_SIZE;
  while (size > Cmi_dgram_max_data) {
    EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank);
    data += Cmi_dgram_max_data;
    size -= Cmi_dgram_max_data;
  }
  EnqueueOutgoingDgram(ogm, data, size, node, rank);
  MoveDatagramsIntoSendWindow(node);
}

void DeliverOutgoingMessage(OutgoingMsg ogm)
{
  int i, rank, dst; OtherNode node;
  
  dst = ogm->dst;
  switch (dst) {
  case PE_BROADCAST_ALL:
    for (rank = 0; rank<Cmi_nodesize; rank++)
      PCQueuePush(CmiGetStateN(rank)->recv,CopyMsg(ogm->data,ogm->size));
    for (i=0; i<Cmi_numnodes; i++)
      if (i!=Cmi_nodenum)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_BROADCAST);
    break;
  case PE_BROADCAST_OTHERS:
    for (rank = 0; rank<Cmi_nodesize; rank++)
      if (rank + Cmi_nodestart != ogm->src)
	PCQueuePush(CmiGetStateN(rank)->recv,CopyMsg(ogm->data,ogm->size));
    for (i = 0; i<Cmi_numnodes; i++)
      if (i!=Cmi_nodenum)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_BROADCAST);
    break;
  default:
    node = nodes_by_pe[dst];
    rank = dst - node->nodestart;
    if (node->nodestart != Cmi_nodestart)
      DeliverViaNetwork(ogm, node, rank);
    else {
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

int CollectOutgoingMessages()
{
  CmiState cs; int rank, pull, action=0;
  OutgoingMsg ogm;
  
  for (rank=0; rank<Cmi_nodesize; rank++) {
    cs = CmiGetStateN(rank);
    ogm = (OutgoingMsg)PCQueuePop(cs->send);
    if (ogm) {
      LOG(("%1.4f %d O\n", Cmi_clock, Cmi_nodestart));
      action = 1;
      DeliverOutgoingMessage(ogm);
    }
  }
  return action;
}

void AssembleDatagram(OtherNode node, ExplicitDgram dg)
{
  int i, size; char *msg;
  
  msg = node->asm_msg;
  if (msg == 0) {
    size = CmiMsgHeaderGetLength(dg->data);
    msg = (char *)CmiAlloc(size);
    if (size < dg->len) KillEveryoneCode(4559312);
    memcpy(msg, dg->data, dg->len);
    node->asm_rank = dg->rank;
    node->asm_total = size;
    node->asm_fill = dg->len;
    node->asm_msg = msg;
  } else {
    size = dg->len - DGRAM_HEADER_SIZE;
    memcpy(msg + node->asm_fill, dg->data + DGRAM_HEADER_SIZE, size);
    node->asm_fill += size;
  }
  if (node->asm_fill == node->asm_total) {
    if (node->asm_rank == DGRAM_BROADCAST) {
      int len = node->asm_total;
      for (i=1; i<Cmi_nodesize; i++)
	PCQueuePush(CmiGetStateN(i)->recv, CopyMsg(msg, len));
      PCQueuePush(CmiGetStateN(0)->recv, msg);
    } else PCQueuePush(CmiGetStateN(node->asm_rank)->recv, msg);
    node->asm_msg = 0;
  }
  FreeExplicitDgram(dg);
}

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
    next = ((next + 1) & DGRAM_SEQNO_MASK);
  }
  node->recv_next = next;
}

void IntegrateMessageDatagram(ExplicitDgram dg)
{
  unsigned int seqno, slot; OtherNode node;

  LOG(("%1.4f %d M %d %d\n", Cmi_clock, Cmi_nodestart, dg->srcpe, dg->seqno));
  node = nodes_by_pe[dg->srcpe];
  seqno = dg->seqno;
  ScheduleAcknowledgement(node);
  if (((seqno - node->recv_next)&DGRAM_SEQNO_MASK) < Cmi_window_size) {
    slot = (seqno % Cmi_window_size);
    if (node->recv_window[slot] == 0) {
      node->recv_window[slot] = dg;
      if (seqno == node->recv_next)
	AssembleReceivedDatagrams(node);
      return;
    }
  }
  FreeExplicitDgram(dg);
}

void IntegrateAcknowledgement(OtherNode node, unsigned int seqno)
{
  unsigned int slot = (seqno % Cmi_window_size);
  ImplicitDgram idg;

  idg = node->send_window[slot];
  if ((idg)&&(idg->seqno == seqno)) {
    idg->acknowledged = 1;
    node->send_window[slot] = 0;
  }
}

void IntegrateAckDatagram(ExplicitDgram dg)
{
  unsigned int seqno; int i, count, *data;
  OtherNode node;
  
  node = nodes_by_pe[dg->srcpe];
  data = ((int*)(dg->data + DGRAM_HEADER_SIZE));
  count = (dg->len - DGRAM_HEADER_SIZE) / sizeof(int);
  LOG(("%1.4f %d R %d %d", 
       Cmi_clock, Cmi_nodestart, node->nodestart, dg->seqno));
  for (i=1; i<=Cmi_window_size; i++) {
    seqno = (dg->seqno - i) & DGRAM_SEQNO_MASK;
    IntegrateAcknowledgement(node, seqno);
  }
  while (count) {
    LOG((" %d", *data));
    IntegrateAcknowledgement(node, *data);
    count--; data++;
  }
  LOG(("\n"));
  MoveDatagramsIntoSendWindow(node);
  FreeExplicitDgram(dg);
}

void ReceiveDatagram()
{
  ExplicitDgram dg; int ok, magic;
  MallocExplicitDgram(dg);
  ok = recv(dataskt,dg->data,Cmi_max_dgram_size,0);
  if (ok<0) KillEveryoneCode(37489437);
  dg->len = ok;
  if (ok >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(dg->data, dg->rank, dg->srcpe, magic, dg->seqno);
    if (magic == 0x1234) {
      if (dg->rank == DGRAM_ACKNOWLEDGE)
	IntegrateAckDatagram(dg);
      else IntegrateMessageDatagram(dg);
    } else FreeExplicitDgram(dg);
  } else FreeExplicitDgram(dg);
}

static void CommunicationServer()
{
  if (CmiMemBusy()) return;
  LOG(("%1.4f %d I\n", GetClock(), Cmi_nodestart));
  while (1) {
    Cmi_clock = GetClock();
    CheckSocketsReady();
    if (ctrlskt_ready_read) { ctrl_getone(); continue; }
    if (dataskt_ready_write) { if (TransmitAcknowledgement()) continue; }
    if (dataskt_ready_read) { ReceiveDatagram(); continue; }
    if (dataskt_ready_write) {
      CollectOutgoingMessages();
      if (TransmitDatagram()) continue;
    }
    break;
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

char *CmiGetNonLocal()
{
  CmiState cs = CmiGetState();
  void *result = PCQueuePop(cs->recv);
  return result;
}

/******************************************************************************
 *
 * CmiGeneralSend
 *
 * The design of this system is that the CommunicationThread does all the
 * work, to eliminate as many locking issues as possible.  This is the
 * only part of the code that happens in the sender-thread
 *
 *****************************************************************************/

CmiCommHandle CmiGeneralSend(int pe, int size, int freemode, char *data)
{
  CmiState cs = CmiGetState(); OutgoingMsg ogm;

  if (pe > Cmi_numpes) *((int*)0)=0;
  if (freemode == 'S') {
    char *copy = (char *)CmiAlloc(size);
    memcpy(copy, data, size);
    data = copy; freemode = 'F';
  }

  if (pe == cs->pe) {
    FIFO_EnQueue(cs->localqueue, data);
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
  PCQueuePush(cs->send, (char *)ogm);
  NotifyCommunicationThread();
  return (CmiCommHandle)ogm;
}

void CmiSyncSendFn(int p, int s, char *m)
{ CmiGeneralSend(p,s,'S',m); }

CmiCommHandle CmiAsyncSendFn(int p, int s, char *m)
{ return CmiGeneralSend(p,s,'A',m); }

void CmiFreeSendFn(int p, int s, char *m)
{ CmiGeneralSend(p,s,'F',m); }

void CmiSyncBroadcastFn(int s, char *m)
{ CmiGeneralSend(PE_BROADCAST_OTHERS,s,'S',m); }

CmiCommHandle CmiAsyncBroadcastFn(int s, char *m)
{ return CmiGeneralSend(PE_BROADCAST_OTHERS,s,'A',m); }

void CmiFreeBroadcastFn(int s, char *m)
{ CmiGeneralSend(PE_BROADCAST_OTHERS,s,'F',m); }

void CmiSyncBroadcastAllFn(int s, char *m)
{ CmiGeneralSend(PE_BROADCAST_ALL,s,'S',m); }

CmiCommHandle CmiAsyncBroadcastAllFn(int s, char *m)
{ return CmiGeneralSend(PE_BROADCAST_ALL,s,'A',m); }

void CmiFreeBroadcastAllFn(int s, char *m)
{ CmiGeneralSend(PE_BROADCAST_ALL,s,'F',m); }

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

CmiInitMc(argv)
char **argv;
{
  CmiState cs = CmiGetState();
  CpvInitialize(void *,CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = cs->localqueue;
}

CmiExit()
{
  ctrl_sendone(120,"aset done %d TRUE\n",CmiMyPe());
  while (Cmi_shutdown_done == 0) jsleep(0, 250000);
  ctrl_sendone(120,"ending\n");
  if (CmiMyRank()==0) log_done();
}

main(argc, argv)
int argc;
char **argv;
{
  int i, ok; struct timeval tv;

#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif

  Cmi_argc = argc;
  Cmi_argv = argv;
  parse_netstart();
  extract_args(argv);
  
  log_init();
  transmit_queue = FIFO_Create();
  retransmit_queue = FIFO_Create();
  ack_queue = FIFO_Create();
  skt_datagram(&dataport, &dataskt, Cmi_os_buffer_size);
  skt_server(&ctrlport, &ctrlskt);
  KillInit();
  ctrl_sendone(120,"notify-die %s %d\n",Cmi_self_IP_str,ctrlport);
  ctrl_sendone(120,"aget %s %d done 0 %d\n",
	       Cmi_self_IP_str,ctrlport,Cmi_numpes-1);
  node_addresses_obtain();
  CmiTimerInit();
  switch_to_multithreaded_mode();
}

