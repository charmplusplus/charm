
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
void ConverseInitPE(void);
void *FIFO_Create(void);
int   FIFO_Fill(void *);
void *FIFO_Peek(void *);
void  FIFO_Pop(void *);
void  FIFO_EnQueue(void *, void *);
void  FIFO_EnQueue_Front(void *, void *);
void CmiYield();

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

void jmemcpy(char *dst, char *src, int len)
{
  char *sdend = (char *)(((int)(src + len)) & ~(sizeof(double)-1));
  while (src != sdend) {
    *((double*)dst) = *((double*)src);
    dst+=sizeof(double); src+=sizeof(double);
  }
  len &= (sizeof(double)-1);
  while (len) { *dst++ = *src++; len--; }
}

char *CopyMsg(char *msg, int len)
{
  char *copy = (char *)CmiAlloc(len);
  jmemcpy(copy, msg, len);
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

static int CountArgs(char **argv)
{
  int count=0;
  while (argv[0]) { count++; argv++; }
  return count;
}


static void jsleep(int sec, int usec)
{
  int ntimes,i;
  struct timeval tm;

  ntimes = sec*200 + usec/5000;
  for(i=0;i<ntimes;i++) {
    tm.tv_sec = 0;
    tm.tv_usec = 5000;
    while(1) {
      if (select(0,NULL,NULL,NULL,&tm)==0) break;
      if ((errno!=EBADF)&&(errno!=EINTR)) return;
    }
  }
}

void writeall(int fd, char *buf, int size)
{
  int ok;
  while (size) {
    retry:
    CmiYield();
    ok = write(fd, buf, size);
    if ((ok<0)&&(errno==EBADF)) goto retry;
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
    if ((nreadable<0)&&((errno==EINTR)||(errno==EBADF))) continue;
    if (nreadable == 0) { errno=ETIMEDOUT; return -1; }
    return 0;
  }
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
  
retry:
  CmiYield();
  fd = socket(PF_INET, SOCK_STREAM, 0);
  if ((fd<0)&&((errno==EINTR)||(errno==EBADF))) goto retry;
  if (fd < 0) { perror("socket 1"); KillEveryoneCode(93483); }
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
retry:
  CmiYield();
  skt = socket(AF_INET,SOCK_DGRAM,0);
  if ((skt<0)&&((errno==EINTR)||(errno==EBADF))) goto retry;
  if (skt < 0)
    { perror("socket 2"); KillEveryoneCode(8934); }
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
  CmiYield();
  fd = accept(src, (struct sockaddr *)&remote, &i);
  if ((fd<0)&&((errno==EINTR)||(errno==EBADF)||(errno==EPROTO))) goto acc;
  if (fd<0) { perror("accept"); KillEveryoneCode(39489); }
  *pip=htonl(remote.sin_addr.s_addr);
  *ppo=htons(remote.sin_port);
  *pfd=fd;
}

int skt_connect(ip, port, seconds)
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
    if ((fd<0)&&((errno==EINTR)||(errno==EBADF))) goto sock;
    if (fd < 0) { perror("socket 3"); exit(1); }
    
  conn:
    ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
    if (ok>=0) break;
    close(fd);
    switch (errno) {
    case EINTR: case EBADF: case EALREADY: break;
    case ECONNREFUSED: jsleep(1,0); break;
    case EADDRINUSE: jsleep(1,0); break;
    case EADDRNOTAVAIL: jsleep(5,0); break;
    default: return -1;
    }
  }
  if (ok<0) return -1;
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

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(char *message)
{
  CmiError(message);
  KillEveryone("");
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

typedef struct { DgramHeader head; char window[1024]; } DgramAck;

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
  unsigned int IP, dataport, ctrlport;
  struct sockaddr_in addr;

  double                   send_give_up; /* time to give up on retrying */
  double                   send_primer;  /* time to send primer packet */
  unsigned int             send_last;    /* seqno of last dgram sent */
  ImplicitDgram           *send_window;  /* datagrams sent, not acked */
  ImplicitDgram            send_queue_h; /* head of send queue */
  ImplicitDgram            send_queue_t; /* tail of send queue */
  unsigned int             send_next;    /* next seqno to go into queue */
  
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
}
*OtherNode;

typedef struct CmiStateStruct
{
  int pe, rank;
  PCQueue recv;
  void *localqueue;
}
*CmiState;

void CmiStateInit(int pe, int rank, CmiState state)
{
  state->pe = pe;
  state->rank = rank;
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
 * Configuration Data
 *
 * This data is all read in from the NETSTART variable (provided by the
 * host) and from the command-line arguments.  Once read in, it is never
 * modified.
 *
 *****************************************************************************/


int               Cmi_numpes;    /* Total number of processors */
int               Cmi_mynodesize;/* Number of processors in my address space */
static int        Cmi_mynode;    /* Which address space am I */
static int        Cmi_numnodes;  /* Total number of address spaces */
static int        Cmi_nodestart; /* First processor in this address space */
static CmiStartFn Cmi_startfn;   /* The start function */
static int        Cmi_usrsched;  /* Continue after start function finishes? */
static char     **Cmi_argv;
static int        Cmi_host_IP;
static int        Cmi_self_IP;
static int        Cmi_host_port;
static int        Cmi_host_pid;
static char       Cmi_host_IP_str[16];
static char       Cmi_self_IP_str[16];

static int    Cmi_max_dgram_size;
static int    Cmi_os_buffer_size;
static int    Cmi_window_size;
static int    Cmi_half_window;
static double Cmi_delay_retransmit;
static double Cmi_ack_delay;
static int    Cmi_dgram_max_data;
static int    Cmi_tickspeed;

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
  if (ns==0) goto abort;
  nread = sscanf(ns, "%d%d%d%d%d%d%d%d",
		 &Cmi_numnodes, &Cmi_mynode,
		 &Cmi_nodestart, &Cmi_mynodesize, &Cmi_numpes,
		 &Cmi_self_IP, &Cmi_host_IP, &Cmi_host_port, &Cmi_host_pid);
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
  setspeed_eth();
  while (*argv) {
    if (strcmp(*argv,"++atm")==0) {
      setspeed_atm();
      DeleteArg(argv);
    } else if (strcmp(*argv,"++eth")==0) {
      setspeed_eth();
      DeleteArg(argv);
    } else argv++;
  }
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
  log_pos = 0;
  log_wrap = 0;
}

static void log_done()
{
  char logname[100]; FILE *f; int i, size;
  sprintf(logname, "log.%d", Cmi_mynode);
  f = fopen(logname, "w");
  if (f==0) { perror("fopen"); exit(1); }
  if (log_wrap) size = 50000; else size=log_pos;
  for (i=0; i<size; i++) {
    logent ent = log+i;
    fprintf(f, "%1.4f %d %c %d %d\n",
	    ent->time, ent->srcpe, ent->kind, ent->dstpe, ent->seqno);
  }
  fclose(f);
}

#define LOG(t,s,k,d,q) { if (log_pos==50000) { log_pos=0; log_wrap=1;} { logent ent=log+log_pos; ent->time=t; ent->srcpe=s; ent->kind=k; ent->dstpe=d; ent->seqno=q; log_pos++; }}

#endif


#if !LOGGING

#define log_init() 0
#define log_done() 0
#define LOG(t,s,k,d,q) 0

#endif

/******************************************************************************
 *
 * Node state
 *
 *****************************************************************************/

static int        ctrlport, dataport, ctrlskt, dataskt;

static OtherNode *nodes_by_pe;  /* OtherNodes indexed by processor number */
static OtherNode  nodes;        /* Indexed only by ``node number'' */

static volatile int          Cmi_shutdown_initiated;
static CmiNodeLock  Cmi_scanf_mutex;
static volatile char        *Cmi_scanf_data;
static double       Cmi_clock;

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
 * CmiStartThreads()
 *
 *    Allocates one CmiState structure per PE.  Initializes all of
 *    the CmiState structures using the function CmiStateInit.
 *    Starts processor threads 1..N (not 0, that's the one
 *    that calls CmiStartThreads), as well as the communication
 *    thread.  Each processor thread (other than 0) must call ConverseInitPE
 *    followed by Cmi_startfn.  The communication thread must be an infinite
 *    loop that calls the function CommunicationServer over and over,
 *    with a short delay between each call (ideally, the delay should end
 *    when a datagram arrives or when somebody notifies: see below).
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
 *****************************************************************************/

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
  mutex_init(lk,0,0);
}

void CmiDestroyLock(CmiNodeLock lk)
{
  mutex_destroy(lk);
  free(lk);
}

void CmiYield() { thr_yield(); }

#define CmiGetStateN(n) (Cmi_state_vector+(n))

static mutex_t comm_mutex;
#define CmiCommLock() (mutex_lock(&comm_mutex))
#define CmiCommUnlock() (mutex_unlock(&comm_mutex))

static void comm_thread(void)
{
  struct timeval tmo; fd_set rfds;
  while (Cmi_shutdown_initiated == 0) {
    CmiCommLock();
    CommunicationServer();
    CmiCommUnlock();
    tmo.tv_sec = 0;
    tmo.tv_usec = Cmi_tickspeed;
    FD_ZERO(&rfds);
    FD_SET(dataskt, &rfds);
    FD_SET(ctrlskt, &rfds);
    select(FD_SETSIZE, &rfds, 0, 0, &tmo);
  }
  thr_exit(0);
}

static void *call_startfn(void *vindex)
{
  int index = (int)vindex;
  CmiState state = Cmi_state_vector + index;
  thr_setspecific(Cmi_state_key, state);
  ConverseInitPE();
  Cmi_startfn(CountArgs(Cmi_argv), Cmi_argv);
  if (Cmi_usrsched == 0) CsdScheduler(-1);
  ConverseExit();
  thr_exit(0);
}

static void CmiStartThreads()
{
  int i, pid, ok;
  
  thr_setconcurrency(Cmi_mynodesize);
  thr_keycreate(&Cmi_state_key, 0);
  Cmi_state_vector =
    (CmiState)calloc(Cmi_mynodesize, sizeof(struct CmiStateStruct));
  for (i=0; i<Cmi_mynodesize; i++)
    CmiStateInit(i+Cmi_nodestart, i, CmiGetStateN(i));
  for (i=1; i<Cmi_mynodesize; i++) {
    ok = thr_create(0, 256000, call_startfn, (void *)i, THR_BOUND, &pid);
    if (ok<0) { perror("thr_create"); exit(1); }
  }
  thr_setspecific(Cmi_state_key, Cmi_state_vector);
  ok = thr_create(0, 256000, (void *(*)(void *))comm_thread, 0, 0, &pid);
  if (ok<0) { perror("thr_create"); exit(1); }
}

static void CmiJoinThreads()
{
  int i, ok; void *status;
  for(i=0; i<Cmi_mynodesize; i++) {
    ok = thr_join(0, 0, &status);
    if (ok<0) perror("thr_join");
  }
}

#endif

#if CMK_SHARED_VARS_UNAVAILABLE

static int memflag;
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

void CmiYield() { jsleep(0,100); }

static void CommunicationInterrupt()
{
  if (comm_flag) return;
  if (memflag) return;
  CommunicationServer();
}

static void CmiStartThreads()
{
  struct itimerval i;
  
  if ((Cmi_numpes != Cmi_numnodes) || (Cmi_mynodesize != 1))
    KillEveryone
      ("Multiple cpus unavailable, don't use cpus directive in nodesfile.\n");
  
  CmiStateInit(Cmi_nodestart, 0, &Cmi_state);
  Cmi_mype = Cmi_nodestart;
  Cmi_myrank = 0;
  
#if CMK_ASYNC_NOT_NEEDED
  CmiSignal(SIGALRM, 0, 0, CommunicationInterrupt);
#else
  CmiSignal(SIGALRM, SIGIO, 0, CommunicationInterrupt);
  CmiEnableAsyncIO(dataskt);
  CmiEnableAsyncIO(ctrlskt);
#endif

  i.it_interval.tv_sec = 0;
  i.it_interval.tv_usec = Cmi_tickspeed;
  i.it_value.tv_sec = 0;
  i.it_value.tv_usec = Cmi_tickspeed;
  setitimer(ITIMER_REAL, &i, NULL);
}

static void CmiJoinThreads() {}

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
  log_done();
  ConverseCommonExit();
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
  CmiSignal(SIGSEGV, 0, 0, KillOnSegv);
  CmiSignal(SIGBUS,  0, 0, KillOnBus);
  CmiSignal(SIGILL,  0, 0, KillOnCrash);
  CmiSignal(SIGABRT, 0, 0, KillOnCrash);
  CmiSignal(SIGFPE,  0, 0, KillOnCrash);
  CmiSignal(SIGPIPE, 0, 0, KillOnCrash);
  CmiSignal(SIGURG,  0, 0, KillOnCrash);

#ifdef SIGSYS
  CmiSignal(SIGSYS,  0, 0, KillOnCrash);
#endif

  CmiSignal(SIGTERM, 0, 0, KillOnIntr);
  CmiSignal(SIGQUIT, 0, 0, KillOnIntr);
  CmiSignal(SIGINT,  0, 0, KillOnIntr);
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
  char *f; int fd, delay; char c;
  va_list p;
  va_start(p);
  delay = va_arg(p, int);
  f = va_arg(p, char *);
  vsprintf(buffer, f, p);
  fd = skt_connect(Cmi_host_IP, Cmi_host_port, delay);
  if (fd<0) KillEveryone("cannot contact host");
  writeall(fd, buffer, strlen(buffer));
#if CMK_SYNCHRONIZE_ON_TCP_CLOSE
  shutdown(fd, 1);
  while (read(fd, &c, 1)==EINTR);
  close(fd);
#else
  close(fd);
#endif
}

/****************************************************************************
 *
 * ctrl_getone
 *
 * This is handled only by the communication interrupt.
 *
 ****************************************************************************/

static void node_addresses_store();

#if CMK_CCS_AVAILABLE
CpvExtern(int, strHandlerID);
#endif

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
      Cmi_shutdown_initiated = 1;
    }
    else if (strncmp(line,"scanf-data ",11)==0) {
      Cmi_scanf_data=strdupl(line+11);
#if CMK_CCS_AVAILABLE
    } else if (strncmp(line, "req ", 4)==0) {
      char cmd[5], *msg;
      int pe, size, len;
      sscanf(line, "%s%d%d", cmd, &pe, &size);
      len = strlen(line);
      msg = (char *) CmiAlloc(len+size+CmiMsgHeaderSizeBytes);
      CmiSetHandler(msg, CpvAccess(strHandlerID));
      strcpy(msg+CmiMsgHeaderSizeBytes, line);
      fread(msg+CmiMsgHeaderSizeBytes+len, 1, size, f);
      PCQueuePush(CmiGetStateN(pe)->recv, msg);
#endif
    } else if (strncmp(line,"die ",4)==0) {
      fprintf(stderr,"aborting: %s\n",line+4);
      log_done();
      ConverseCommonExit();
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
	       Cmi_mynode, Cmi_self_IP_str, ctrlport, dataport,
	       Cmi_nodestart, Cmi_mynodesize);
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
  CmiLock(Cmi_scanf_mutex);
  ctrl_sendone(120, "scanf %s %d %s", Cmi_self_IP_str, ctrlport, fmt);
  while (Cmi_scanf_data == 0) jsleep(0, 250000);
  i = sscanf(Cmi_scanf_data, fmt,
	     ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3], ptr[ 4], ptr[ 5],
	     ptr[ 6], ptr[ 7], ptr[ 8], ptr[ 9], ptr[10], ptr[11],
	     ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17]);
  free(Cmi_scanf_data);
  Cmi_scanf_data=0;
  CmiUnlock(Cmi_scanf_mutex);
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

int TransmitAckDatagram(OtherNode node)
{
  DgramAck ack; unsigned int i, seqno, slot; ExplicitDgram dg;
  
  seqno = node->recv_next;
  DgramHeaderMake(&ack, DGRAM_ACKNOWLEDGE, Cmi_nodestart, Cmi_host_pid, seqno);
  LOG(Cmi_clock, Cmi_nodestart, 'A', node->nodestart, seqno);
  for (i=0; i<Cmi_window_size; i++) {
    slot = seqno % Cmi_window_size;
    dg = node->recv_window[slot];
    ack.window[i] = (dg && (dg->seqno == seqno));
    seqno = ((seqno+1) & DGRAM_SEQNO_MASK);
  }
  sendto(dataskt, (char *)&ack,
	 DGRAM_HEADER_SIZE + Cmi_window_size, 0,
	 (struct sockaddr *)&(node->addr),
	 sizeof(struct sockaddr_in));
}

void TransmitImplicitDgram(ImplicitDgram dg)
{
  char *data; DgramHeader *head; int len; DgramHeader temp;
  OtherNode dest;

  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_host_pid, dg->seqno);
  LOG(Cmi_clock, Cmi_nodestart, 'T', dest->nodestart, dg->seqno);
  sendto(dataskt, (char *)head, len + DGRAM_HEADER_SIZE, 0,
	      (struct sockaddr *)&(dest->addr), sizeof(struct sockaddr_in));
  *head = temp;
}

void TransmitImplicitDgram1(ImplicitDgram dg)
{
  char *data; DgramHeader *head; int len; DgramHeader temp;
  OtherNode dest;

  len = dg->datalen;
  data = dg->dataptr;
  head = (DgramHeader *)(data - DGRAM_HEADER_SIZE);
  temp = *head;
  dest = dg->dest;
  DgramHeaderMake(head, dg->rank, dg->srcpe, Cmi_host_pid, dg->seqno);
  LOG(Cmi_clock, Cmi_nodestart, 'P', dest->nodestart, dg->seqno);
  sendto(dataskt, (char *)head, len + DGRAM_HEADER_SIZE, 0,
	      (struct sockaddr *)&(dest->addr), sizeof(struct sockaddr_in));
  *head = temp;
}

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

int TransmitDatagram()
{
  ImplicitDgram dg; OtherNode node;
  static int nextnode=0; unsigned int skip, count, slot, seqno;
  
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
	node->send_give_up = Cmi_clock + 300; /* five minutes */
	node->send_primer = Cmi_clock + Cmi_delay_retransmit;
	/* Note --- give up delay is 5 min because I/O could stall node */
	return 1;
      }
    }
    if (Cmi_clock > node->send_primer) {
      slot = (node->send_last % Cmi_window_size);
      for (count=0; count<Cmi_window_size; count++) {
	dg = node->send_window[slot];
	if (dg) break;
	slot = ((slot-1) % Cmi_window_size);
      }
      if (dg) {
	TransmitImplicitDgram1(node->send_window[slot]);
	node->send_primer = Cmi_clock + Cmi_delay_retransmit;
	if (Cmi_clock > node->send_give_up)
	  KillEveryone("processor not responding");
	return 1;
      }
    }
  }
  return 0;
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
  int size, seqno; char *data;
  
  size = ogm->size - DGRAM_HEADER_SIZE;
  data = ogm->data + DGRAM_HEADER_SIZE;
  while (size > Cmi_dgram_max_data) {
    EnqueueOutgoingDgram(ogm, data, Cmi_dgram_max_data, node, rank);
    data += Cmi_dgram_max_data;
    size -= Cmi_dgram_max_data;
  }
  EnqueueOutgoingDgram(ogm, data, size, node, rank);
}

void DeliverOutgoingMessage(OutgoingMsg ogm)
{
  int i, rank, dst; OtherNode node;
  
  dst = ogm->dst;
  switch (dst) {
  case PE_BROADCAST_ALL:
    for (rank = 0; rank<Cmi_mynodesize; rank++)
      PCQueuePush(CmiGetStateN(rank)->recv,CopyMsg(ogm->data,ogm->size));
    for (i=0; i<Cmi_numnodes; i++)
      if (i!=Cmi_mynode)
	DeliverViaNetwork(ogm, nodes + i, DGRAM_BROADCAST);
    GarbageCollectMsg(ogm);
    break;
  case PE_BROADCAST_OTHERS:
    for (rank = 0; rank<Cmi_mynodesize; rank++)
      if (rank + Cmi_nodestart != ogm->src)
	PCQueuePush(CmiGetStateN(rank)->recv,CopyMsg(ogm->data,ogm->size));
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

void AssembleDatagram(OtherNode node, ExplicitDgram dg)
{
  int i, size; char *msg;
  
  msg = node->asm_msg;
  if (msg == 0) {
    size = CmiMsgHeaderGetLength(dg->data);
    msg = (char *)CmiAlloc(size);
    if (size < dg->len) KillEveryoneCode(4559312);
    jmemcpy(msg, (char*)(dg->data), dg->len);
    node->asm_rank = dg->rank;
    node->asm_total = size;
    node->asm_fill = dg->len;
    node->asm_msg = msg;
  } else {
    size = dg->len - DGRAM_HEADER_SIZE;
    jmemcpy(msg + node->asm_fill, ((char*)(dg->data))+DGRAM_HEADER_SIZE, size);
    node->asm_fill += size;
  }
  if (node->asm_fill == node->asm_total) {
    if (node->asm_rank == DGRAM_BROADCAST) {
      int len = node->asm_total;
      for (i=1; i<Cmi_mynodesize; i++)
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
    node->recv_winsz--;
    next = ((next + 1) & DGRAM_SEQNO_MASK);
  }
  node->recv_next = next;
}

void IntegrateMessageDatagram(ExplicitDgram dg)
{
  unsigned int seqno, slot; OtherNode node;

  LOG(Cmi_clock, Cmi_nodestart, 'M', dg->srcpe, dg->seqno);
  node = nodes_by_pe[dg->srcpe];
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
      return;
    }
  }
  FreeExplicitDgram(dg);
}

void IntegrateAckDatagram(ExplicitDgram dg)
{
  OtherNode node; DgramAck *ack; ImplicitDgram idg;
  int i; unsigned int slot, rxing, dgseqno, seqno;
  
  node = nodes_by_pe[dg->srcpe];
  ack = ((DgramAck*)(dg->data));
  dgseqno = dg->seqno;
  seqno = (dgseqno + Cmi_window_size) & DGRAM_SEQNO_MASK;
  slot = seqno % Cmi_window_size;
  rxing = 0;
  LOG(Cmi_clock, Cmi_nodestart, 'R', node->nodestart, dg->seqno);
  for (i=Cmi_window_size-1; i>=0; i--) {
    slot--; if (slot== ((unsigned int)-1)) slot+=Cmi_window_size;
    seqno = (seqno-1) & DGRAM_SEQNO_MASK;
    idg = node->send_window[slot];
    if (idg) {
      if (idg->seqno == seqno) {
	if (ack->window[i]) {
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
      } else if (((idg->seqno - dgseqno) & DGRAM_SEQNO_MASK)>=Cmi_window_size){
	LOG(Cmi_clock, Cmi_nodestart, 'r', node->nodestart, idg->seqno);
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
  if (ok<0) KillEveryoneCode(37489437);
  dg->len = ok;
  if (ok >= DGRAM_HEADER_SIZE) {
    DgramHeaderBreak(dg->data, dg->rank, dg->srcpe, magic, dg->seqno);
    if (magic == Cmi_host_pid) {
      if (dg->rank == DGRAM_ACKNOWLEDGE)
	IntegrateAckDatagram(dg);
      else IntegrateMessageDatagram(dg);
    } else FreeExplicitDgram(dg);
  } else FreeExplicitDgram(dg);
}

static void CommunicationServer()
{
  LOG(GetClock(), Cmi_nodestart, 'I', 0, 0);
  while (1) {
    Cmi_clock = GetClock();
    CheckSocketsReady();
    if (ctrlskt_ready_read) { ctrl_getone(); continue; }
    if (dataskt_ready_write) { if (TransmitAcknowledgement()) continue; }
    if (dataskt_ready_read) { ReceiveDatagram(); continue; }
    if (dataskt_ready_write) { if (TransmitDatagram()) continue; }
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
 * CmiNotifyIdle()
 *
 *****************************************************************************/

void CmiNotifyIdle()
{
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  struct timeval tv;
  tv.tv_sec=0; tv.tv_usec=5000;
  select(0,0,0,0,&tv);
#else
  CmiCommLock();
  CommunicationServer();
  CmiCommUnlock();
#endif
}

/******************************************************************************
 *
 * CmiGeneralSend
 *
 *****************************************************************************/

CmiCommHandle CmiGeneralSend(int pe, int size, int freemode, char *data)
{
  CmiState cs = CmiGetState(); OutgoingMsg ogm;

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
  CmiCommLock();
  DeliverOutgoingMessage(ogm);
  CommunicationServer();
  CmiCommUnlock();
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

void ConverseInitPE()
{
  CmiState cs = CmiGetState();
  CthInit(Cmi_argv);
  ConverseCommonInit(Cmi_argv);
  CpvInitialize(void *,CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = cs->localqueue;
}

void ConverseExit()
{
  ctrl_sendone(120,"aset done %d TRUE\n",CmiMyPe());
  while (Cmi_shutdown_initiated == 0) CmiYield();
  ctrl_sendone(120,"ending\n");
  if (CmiMyRank()==0) {
    log_done();
    ConverseCommonExit();
    CmiJoinThreads();
  }
}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usc, int ret)
{
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  Cmi_argv = argv;
  Cmi_startfn = fn;
  Cmi_usrsched = usc;
  parse_netstart();
  extract_args(argv);
  log_init();
  skt_datagram(&dataport, &dataskt, Cmi_os_buffer_size);
  skt_server(&ctrlport, &ctrlskt);
  KillInit();
  ctrl_sendone(120,"notify-die %s %d\n",Cmi_self_IP_str,ctrlport);
  ctrl_sendone(120,"aget %s %d done 0 %d\n",
	       Cmi_self_IP_str,ctrlport,Cmi_numpes-1);
  node_addresses_obtain();
  Cmi_scanf_mutex = CmiCreateLock();
  CmiTimerInit();
  CmiStartThreads();
  ConverseInitPE();
  if (ret==0) {
    fn(CountArgs(argv), argv);
    if (usc==0) CsdScheduler(-1);
    ConverseExit();
  }
}
