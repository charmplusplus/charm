/*****************************************************************************
 *
 * Machine-Specific Definitions
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

#if CMK_TIMER_USE_TIMES
#include <sys/times.h>
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

#ifdef DEBUG
#define TRACE(p) p
#else
#define TRACE(p)
#endif

extern int CmemInsideMem();
extern void CmemCallWhenMemAvail();

/******************************************************************************
 *
 * Memory Mutual Exclusion
 *
 * This code is designed to keep malloc from being called reentrantly.
 * Malloc is a special case, because malloc must be operational right
 * from the start. Thus, it's mutual exclusion mechanism must be free
 * of dependencies on anything that requires initialization.
 *
 * The remainder of the system uses a mutual exclusion flag ``busy'' in
 * the cmi_state record.
 *
 *****************************************************************************/

#if CMK_SHARED_VARS_UNAVAILABLE

static int memflag;
void CmiMemLock() { memflag = 1; }
void CmiMemUnlock() { memflag = 0; }
#define CmiMemBusy() memflag

#endif

#if CMK_SHARED_VARS_SUN_THREADS

static mutex_t memmutex;
static int     memflag;
void CmiMemLock()
{
  mutex_lock(&memmutex);
  memflag=1;
}
void CmiMemUnlock()
{
  memflag=0;
  mutex_unlock(&memmutex);
}
#define CmiMemBusy() (memflag)

#endif

/******************************************************************************
 *
 * Protocol State Variables
 *
 *****************************************************************************/

#define MAX_NODESIZE 64
#define MAX_NODES 128

typedef struct {
  int IP;
  int dataport;
  int ctrlport;
  struct sockaddr_in addr;
} node_info;

/* Types of messages. */

# define SEND  1
# define ACK  2

typedef unsigned char BYTE;

/* In bytes.  Make sure this is greater than zero! */
/* Works out to be about 2018 bytes. */

# define MAXDSIZE (CMK_DGRAM_MAX_SIZE - sizeof(DATA_HDR))

/* Format of the header sent with each fragment. */
typedef struct DATA_HDR
{
  unsigned int seq_num;   /* Sequence Number */
  unsigned int PeNum;     /* always penum of the originator*/
  unsigned int pktidx;    /* Index of packet within message */
  unsigned int rem_size;  /*Size of remaining message including this packet*/
                          /*If this is zero, it is ACK otherwise SEND */
}
DATA_HDR;

/* Header combined with data fragment. */
typedef struct msgspace
{
  DATA_HDR hd;
  char data[MAXDSIZE];
}
msgspace;

typedef struct
{
  DATA_HDR        *packet;
  unsigned int     seq_num;
  double           send_time;
}
WindowElement;

typedef struct pkt_queue_elem
{
  DATA_HDR *packet;
  struct pkt_queue_elem *nextptr;
}
PacketQueueElem;

typedef struct msg_queue_elem
{
  char *mesg;
  struct msg_queue_elem *nextptr;
}
MsgQueueElem;

typedef struct new_msg
{
  int numpackets;
  int numfrags;
  char *mesg;
}
NewMessage;

#define MAX_SEQ_NUM 0xFFFFFFFF     /* 2^32 - 1 */

/*****************************************************************************
 *
 * These variables contain all the processor-specific state.
 *
 *****************************************************************************/

typedef struct Cmi_state_struct
{
  int ctrl_port;
  int ctrl_skt;
  int data_port;
  int data_skt;
  
  node_info  node_table[MAX_NODES];
  int        node_table_fill;
  
  int all_done;
  char *scanf_data;
  double CmiNow;
  FILE *outlog_file;
  
  int Communication_init;            /* Communication set up yet? */
  WindowElement **send_window;       /* packets awaiting acks  */
  PacketQueueElem **transmit_head;   /* packets awaiting transmission */
  PacketQueueElem **transmit_tail; 
  int *first_window_index; 
  int *last_window_index;
  int *cur_window_size;
  unsigned int  *next_seq_num;
  int *timeout_factor;
  WindowElement **recv_window;    /* packets awaiting acks  */
  int *next_window_index;         /* index of 1st entry in recv window */
  unsigned int *expected_seq_num; /* next sequence number expected */
  NewMessage *recd_messages;
  int *needack;
  DATA_HDR ack;
  MsgQueueElem *recd_msg_head, *recd_msg_tail;
  
  int NumAlloc;
  int NumFree;
  int NumIntr;
  int NumIntrCalls;
  int NumOutsideMc;
  int NumRetransmits;
  int NumAcksSent;
  int NumUseless;
  int NumSends;
  int NumEvents;

  int mype;
  int myrank;
  int busy;
} *cmi_state;


#if CMK_SHARED_VARS_SUN_THREADS

static int cmi_state_key;
cmi_state CmiState()
{
  cmi_state result = 0;
  thr_getspecific(cmi_state_key, (void **)&result);
  return result;
}
int CmiMyPe() { return CmiState()->mype; }
int CmiMyRank() { return CmiState()->myrank; }

#endif

#if CMK_SHARED_VARS_UNAVAILABLE

static struct Cmi_state_struct Cmi_state;
#define CmiState() (&Cmi_state)
int Cmi_mype;
int Cmi_myrank;

#endif

CpvDeclare(void *, CmiLocalQueue);

/*****************************************************************************
 *
 * Statistics Module
 *
 *****************************************************************************/

CmiStatsInit()
{
  cmi_state cs = CmiState();
  cs->NumAlloc = 0;
  cs->NumFree = 0;
  cs->NumIntr = 0;
  cs->NumIntrCalls = 0;
  cs->NumOutsideMc = 0;
  cs->NumRetransmits = 0;
  cs->NumAcksSent = 0;
  cs->NumUseless = 0;
  cs->NumSends = 0;
  cs->NumEvents = 0;
}

CmiStatsPrint()
{
  cmi_state cs = CmiState();
  CmiPrintf("NumAlloc = %d\n", cs->NumAlloc);
  CmiPrintf("NumFree = %d\n", cs->NumFree);
  CmiPrintf("NumIntr = %d\n", cs->NumIntr);
  CmiPrintf("NumIntrCalls = %d\n", cs->NumIntrCalls);
  CmiPrintf("NumOutsideMc = %d\n", cs->NumOutsideMc);
  CmiPrintf("NumRetransmits = %d\n", cs->NumRetransmits);
  CmiPrintf("NumAcksSent = %d\n", cs->NumAcksSent);
  CmiPrintf("NumUseless = %d\n", cs->NumUseless);
  CmiPrintf("NumSends = %d\n", cs->NumSends);
  CmiPrintf("NumEvents = %d\n", cs->NumEvents);
}

static void outlog_init(outputfile, index) char *outputfile; int index;
{
  cmi_state cs = CmiState();
  char fn[MAXPATHLEN];
  if (outputfile) {
    sprintf(fn,outputfile,index);
    cs->outlog_file = fopen(fn,"w");
  }
  else cs->outlog_file = 0;
}

static void outlog_done()
{
  cmi_state cs = CmiState();
  fclose(cs->outlog_file);
}

static void outlog_output(buf) char *buf;
{
  cmi_state cs = CmiState();
  if (cs->outlog_file) {
    fprintf(cs->outlog_file,"%s",buf);
    fflush(cs->outlog_file);
  }
}


/*****************************************************************************
 *
 * CmiAlloc, CmiSize, and CmiFree
 *
 *****************************************************************************/

void *CmiAlloc(size)
int size;
{
  cmi_state cs = CmiState();
  char *res;
  CmiInterruptsBlock();
  cs->NumAlloc++;
  res =(char *)malloc(size+8);
  CmiInterruptsRelease();
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
  cmi_state cs = CmiState();
  CmiInterruptsBlock();
  cs->NumFree++;
  free(((char *)blk)-8);
  CmiInterruptsRelease();
}

/*****************************************************************************
 *
 *     Utility routines for network machine interface.
 *
 *
 * zap_newline(char *s)
 *
 *   - Remove the '\n' from the end of a string.
 *
 * char *substr(char *lo, char *hi)
 *
 *   - return an allocated copy of a string subsequence
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
 * int my_sendto
 *     (int s, char *msg, int len, int flags, (struct sockaddr *)to, int tolen)
 * 
 *   - performs a "sendto", automatically retrying on trivial errors.
 *
 *****************************************************************************/

static void jsleep(int sec, int usec)
{
  struct timeval tm;
  tm.tv_sec = sec;
  tm.tv_usec = usec;
  while (1) {
    if (select(0,0,0,0,&tm)==0) break;
    if (errno!=EINTR) break;
  }
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

static void zap_newline(s) char *s;
{
  char *p;
  p = s + strlen(s)-1;
  if (*p == '\n') *p = '\0';
}

static char *substr(lo, hi) char *lo; char *hi;
{
  int len = hi-lo;
  char *res = (char *)CmiAlloc(1+len);
  memcpy(res, lo, len);
  res[len]=0;
  return res;
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

static char *readint(p, value) char *p; int *value;
{
  int val = 0;
  while (((*p)==' ')||((*p)=='.')) p++;
  if (((*p)<'0')||((*p)>'9')) KillEveryone("badly-formed number");
  while ((*p>='0')&&(*p<='9')) { val*=10; val+=(*p)-'0'; p++; }
  *value = val;
  return p;
}

static char *strdupl(s) char *s;
{
  int len = strlen(s);
  char *res = (char *)CmiAlloc(len+1);
  strcpy(res, s);
  return res;
}

static int my_sendto(s, msg, len, flags, to, tolen)
int s; char *msg; int len; int flags; struct sockaddr *to; int tolen;
{
  int ok;

/* I tried this, but didn't seem to have impact.
   We should make sure a poll/check like this is not needed, and then 
   delete this comment --- Sanjay 

  fd_set wfds;
  int nwritable;
  FD_ZERO(&wfds);
  FD_SET(s, &wfds);
  nwritable = select(FD_SETSIZE, NULL, &wfds, NULL, NULL);  
*/

  while((ok = sendto(s, msg, len, flags, to, tolen))<0);
  if (ok != len) KillEveryoneCode(21);
  return ok;
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

static char *DeleteArg(argv)
char **argv;
{
  char *res = argv[0];
  if (res==0) KillEveryone("Illegal Arglist");
  while (*argv) { argv[0]=argv[1]; argv++; }
  return res;
}

/**************************************************************************
 *
 * SKT - socket routines
 *
 *
 * void skt_server(unsigned int *ppo, unsigned int *pfd)
 *
 *   - create a tcp server socket.  Performs the whole socket/bind/listen
 *     procedure.  Returns the IP address of the socket (eg, the IP of the
 *     current machine), the port of the socket, and the file descriptor.
 *
 * void skt_datagram(unsigned int *ppo, unsigned int *pfd)
 *
 *   - creates a UDP datagram socket.  Performs the whole socket/bind/
 *     getsockname procedure.  Returns the IP address of the socket (eg,
 *     the IP address of the current machine), the port of the socket, and
 *     the file descriptor.
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

static void skt_datagram(ppo, pfd)
unsigned int *ppo;
unsigned int *pfd;
{
  struct sockaddr_in name;
  int length, ok, skt;
  int optval, optlen;

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

  optlen = 4;
  /*  getsockopt(skt, SOL_SOCKET , SO_RCVBUF , (char *) &optval, &optlen); */
  optval = CMK_DGRAM_BUF_SIZE;
  if (setsockopt(skt, SOL_SOCKET , SO_RCVBUF , (char *) &optval, optlen) < 0)
    {perror("setting socket rcv bufer size");
     KillEveryoneCode(35782); }
  if (setsockopt(skt, SOL_SOCKET , SO_SNDBUF , (char *) &optval, optlen) < 0)
    {perror("setting socket rcv bufer size");
     KillEveryoneCode(35782); }


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

/******************************************************************************
 *
 * Configuration Data
 *
 * This configuration data must be parsed before the CPV/CSV system
 * is initialized.  Therefore, it must be stored in variables which are
 * neither CPV or CSV.  We use static variables (informal CSV).
 *
 * Note that the default resend-wait and resend-fail are 10 and 600000
 * milliseconds, respectively. Both are set very high, to
 * ameliorate a known bug in the machine version.  If an entry-point is
 * executing a CmiScanf, then it won't acknowledge messages.  To everyone
 * else, it appears dead.  Therefore, the default_resend_fail is set very
 * high to keep things going during long CmiScanfs.  (You have 10 minutes to
 * respond).
 *
 *****************************************************************************/

int Cmi_numpes;
int Cmi_nodesize;

static char **conf_argv;
static int    conf_argc;
static int    conf_node_start; /* PE number of first processor in this node */
static int    conf_host_IP;
static int    conf_self_IP;
static int    conf_host_port;
static char   conf_host_IP_str[16];
static char   conf_self_IP_str[16];
static char   conf_topology;
static char  *conf_outfile;
static double conf_resend_wait;
static double conf_resend_fail;
static int    conf_enableinterrupts;

static void conf_parse_netstart()
{
  char *ns;
  int nread, rank;
  ns = getenv("NETSTART");
  if (ns==0) goto abort;
  nread = sscanf(ns, "%d%d%d%d%d%d%d",
		 &conf_node_start, &Cmi_numpes, &Cmi_nodesize, &rank,
		 &conf_self_IP, &conf_host_IP, &conf_host_port);
  if (nread!=7) goto abort;
  if (rank!=0) goto abort;
  sprintf(conf_self_IP_str,"%d.%d.%d.%d",
	  (conf_self_IP>>24)&0xFF,(conf_self_IP>>16)&0xFF,
	  (conf_self_IP>>8)&0xFF,conf_self_IP&0xFF);
  sprintf(conf_host_IP_str,"%d.%d.%d.%d",
	  (conf_host_IP>>24)&0xFF,(conf_host_IP>>16)&0xFF,
	  (conf_host_IP>>8)&0xFF,conf_host_IP&0xFF);
  return;
 abort:
  fprintf(stderr,"program not started using 'conv-host' utility. aborting.\n");
  exit(1);
}

static void conf_extract_args(argv)
char **argv;
{
  conf_resend_wait =   0.030; /* This could be m/c dependent -- -Sanjay */
  conf_resend_fail = 600.000;
  conf_topology = 'H';
  conf_outfile = NULL;

#if CMK_SHARED_VARS_UNAVAILABLE
  conf_enableinterrupts = 1;
#endif

  while (*argv) {
    if (strcmp(*argv,"++resend-wait")==0) {
      DeleteArg(argv); conf_resend_wait = atoi(DeleteArg(argv)) *.001;
    } else
    if (strcmp(*argv,"++resend-fail")==0) {
      DeleteArg(argv); conf_resend_fail = atoi(DeleteArg(argv)) * .001;
    } else
    if (strcmp(*argv,"++no-interrupts")==0) {
      DeleteArg(argv); conf_enableinterrupts=0;
    } else
    if (strcmp(*argv,"++topology")==0) {
      DeleteArg(argv); conf_topology=DeleteArg(argv)[0];
    } else
    if (strcmp(*argv,"++outputfile")==0) {
      DeleteArg(argv); conf_outfile=DeleteArg(argv);
    } else
    argv++;
  }
}

/****************************************************************************
 *                                                                          
 * Fast shutdown                                                            
 *                                                                          
 ****************************************************************************/

static int KillIndividual(cmd, ip, port, sec)
char *cmd; unsigned int ip; unsigned int port; int sec;
{
  int fd;
  fd = skt_connect(ip, port, sec);
  if (fd<0) return -1;
  writeall(fd, cmd, strlen(cmd));
  close(fd);
  return 0;  
}

static void KillEveryone(msg)
char *msg;
{
  cmi_state cs = CmiState();
  char buffer[1024]; int i;
  sprintf(buffer,"die %s",msg);
  KillIndividual(buffer, conf_host_IP, conf_host_port, 30);
  for (i=0; i<CmiNumPes(); i++)
    KillIndividual(buffer, cs->node_table[i].IP, cs->node_table[i].ctrlport,3);
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
  sprintf(buffer, "Node %d: Segmentation fault.\n",CmiMyPe());
  KillEveryone(buffer);
}

static void KillOnIntr()
{
  char buffer[1000];
  sprintf(buffer, "Node %d: Interrupted.\n",CmiMyPe());
  KillEveryone(buffer);
}

static void KillInit()
{
  signal(SIGSEGV, KillOnSegv);
  signal(SIGBUS,  KillOnSegv);
  signal(SIGILL,  KillOnSegv);
  signal(SIGABRT, KillOnSegv);
  signal(SIGFPE,  KillOnSegv);

#ifdef SIGSYS
  signal(SIGSYS,  KillOnSegv);
#endif

  signal(SIGPIPE, KillOnSegv);
  signal(SIGURG,  KillOnSegv);

  signal(SIGTERM, KillOnIntr);
  signal(SIGQUIT, KillOnIntr);
  signal(SIGINT,  KillOnIntr);
}

/****************************************************************************
 *
 * ctrl_getone
 *
 * Receive a command (on the control socket) from the host or another node,
 * and process it.  This is just a dispatcher, none of the actual processing
 * is done here.
 *
 ****************************************************************************/

static void node_addresses_store();

static void ctrl_sendone(va_alist) va_dcl
{
  char buffer[1024];
  char *f; int fd, delay;
  va_list p;
  va_start(p);
  delay = va_arg(p, int);
  f = va_arg(p, char *);
  vsprintf(buffer, f, p);
  fd = skt_connect(conf_host_IP, conf_host_port, delay);
  if (fd<0) KillEveryone("cannot contact host");
  writeall(fd, buffer, strlen(buffer));
  shutdown(fd, 1);
  while (read(fd, buffer, 1023)>0);
  close(fd);
}

static void ctrl_getone()
{
  cmi_state cs = CmiState();
  char line[10000];
  int ok, ip, port, fd;  FILE *f;
  skt_accept(cs->ctrl_skt, &ip, &port, &fd);
  f = fdopen(fd,"r");
  while (fgets(line, 9999, f)) {
    if      (strncmp(line,"aval addr ",10)==0) node_addresses_store(line);
    else if (strncmp(line,"aval done ",10)==0) cs->all_done = 1;
    else if (strncmp(line,"scanf-data ",11)==0)cs->scanf_data=strdupl(line+11);
    else if (strncmp(line,"die ",4)==0) {
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
 *     aset addr <my-nodeno> <my-ip-addr>.<my-ctrlport>.<my-dataport>
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

static void node_addresses_receive()
{
  cmi_state cs = CmiState();
  ctrl_sendone(120, "aset addr %d %s.%d.%d\n",
         CmiMyPe(), conf_self_IP_str, cs->ctrl_port, cs->data_port);
  ctrl_sendone(120, "aget %s %d addr 0 %d\n",
    conf_self_IP_str,cs->ctrl_port,CmiNumPes()-1);
  while (cs->node_table_fill != CmiNumPes()) {
    if (wait_readable(cs->ctrl_skt, 300)<0)
      { perror("waiting for data"); KillEveryoneCode(21323); }
    ctrl_getone();
  }
}

static void node_addresses_store(addrs) char *addrs;
{
  cmi_state cs = CmiState();
  char *p, *e; int i, lo, hi;
  if (strncmp(addrs,"aval addr ",10)!=0) KillEveryoneCode(83473);
  p = skipblanks(addrs+10);
  p = readint(p,&lo);
  p = readint(p,&hi);
  if ((lo!=0)||(hi!=CmiNumPes()-1)) KillEveryoneCode(824793);
  for (i=0; i<CmiNumPes(); i++) {
    unsigned int ip0,ip1,ip2,ip3,cport,dport,ip;
    p = readint(p,&ip0);
    p = readint(p,&ip1);
    p = readint(p,&ip2);
    p = readint(p,&ip3);
    p = readint(p,&cport);
    p = readint(p,&dport);
    ip = (ip0<<24)+(ip1<<16)+(ip2<<8)+ip3;
    cs->node_table[i].IP = ip;
    cs->node_table[i].ctrlport = cport;
    cs->node_table[i].dataport = dport;
    cs->node_table[i].addr.sin_family      = AF_INET;
    cs->node_table[i].addr.sin_port        = htons(dport);
    cs->node_table[i].addr.sin_addr.s_addr = htonl(ip);
  }
  p = skipblanks(p);
  if (*p!=0) KillEveryoneCode(82283);
  cs->node_table_fill = CmiNumPes();
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
  outlog_output(buf);
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
  outlog_output(buf);
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
  cmi_state cs = CmiState();
  static int CmiProbe();
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
  ctrl_sendone(120, "scanf %s %d %s", conf_self_IP_str, cs->ctrl_port, fmt);
  while (cs->scanf_data==0) CmiProbe();
  i = sscanf(cs->scanf_data, fmt,
         ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3], ptr[ 4], ptr[ 5],
         ptr[ 6], ptr[ 7], ptr[ 8], ptr[ 9], ptr[10], ptr[11],
         ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17]);
  CmiFree(cs->scanf_data);
  cs->scanf_data=0;
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

/****************************************************************************/
/* ROUTINES FOR SENDING/RECEIVING MESSAGES
**
** Definition of terms:
** Synchronized send -- The send doesn't return until the buffer space of
**  the data sent is free for re-use, that is, until a write to that buffer
**  space will not cause the message being sent to be disrupted or altered
**  in any way.
**
** Synchronized receive -- The receive doesn't return until all of the data
**  has been placed in the receiving buffer.
**
** -CW
*/
/****************************************************************************/


/* Send an acknowledging message to the PE penum.
** The acknowledgement has the same header, except that rem_size == 0
*/
static void send_ack(packet,penum) 
     DATA_HDR *packet; 
     int penum;
{
  cmi_state cs = CmiState();
  cs->NumSends++;
#if CMK_IS_HETERO
  packet->seq_num = htonl(packet->seq_num);
  packet->PeNum = htonl(packet->PeNum);
  packet->pktidx = htonl(packet->pktidx);
  packet->rem_size = htonl(packet->rem_size);
#endif
  my_sendto(cs->data_skt, (char *)packet, sizeof(DATA_HDR), 0, 
    (struct sockaddr *)&(cs->node_table[penum].addr),
    sizeof(struct sockaddr_in));
}


/* This section implements a send sliding window protocol for the IPnet 
   implementation of the Chare kernel. The modification to Chris Walquist's 
   implementation are as follows:

   1) The sends are not synchronous, i.e. every pack is not acknowledged before
      the next packet is sent. Instead, packets are sent until CMK_DGRAM_WINDOW_SIZE 
      packets remain unacknowledged.

   2) An ack packet with sequence number N acknowledges all packets upto and
      including N. Thus, it is not necessary for all packets to be explicitly
      acknowledged.

   3) After the sends have been completed, no waits are performed. Instead,
      execution of the Loop resumes. During each pump message phase of the 
      loop, the sliding window is updated as and when ack packets are received.

   4) One send window per destination is necessary for sequence numbers of
      acks to be used as required in (2).

              Questions, Comments and Criticisms to be directed to

             Balkrishna Ramkumar
            ramkumar@crhc.uiuc.edu

*/
   
      
static void SendWindowInit()
{
  cmi_state cs = CmiState();
  int mype = CmiMyPe();
  int numpe = CmiNumPes();
  int i,j;

  cs->send_window=(WindowElement **)CmiAlloc(numpe*sizeof(WindowElement *));
  cs->transmit_head=
    (PacketQueueElem **)CmiAlloc(numpe*sizeof(PacketQueueElem *));
  cs->transmit_tail=
    (PacketQueueElem **)CmiAlloc(numpe*sizeof(PacketQueueElem *));
  cs->first_window_index = (int *)CmiAlloc(numpe*sizeof(int));
  cs->last_window_index = (int *)CmiAlloc(numpe*sizeof(int));
  cs->cur_window_size = (int *) CmiAlloc(numpe * sizeof(int));
  cs->next_seq_num = (unsigned int  *) CmiAlloc(numpe * sizeof(unsigned int ));
  cs->timeout_factor = (int *)CmiAlloc(numpe * sizeof(int)) ;

  for (i = 0; i < numpe; i++) {
    if (i != mype) {
      cs->send_window[i]=
	(WindowElement *)CmiAlloc(CMK_DGRAM_WINDOW_SIZE*sizeof(WindowElement));
      for (j = 0; j < CMK_DGRAM_WINDOW_SIZE; j++) {
        cs->send_window[i][j].packet = NULL;
        cs->send_window[i][j].seq_num = 0;
        cs->send_window[i][j].send_time = 0.0;
      }
    }
    else cs->send_window[i] = NULL;  /* never used */

    cs->first_window_index[i] = 0;
    cs->last_window_index[i] = 0;
    cs->cur_window_size[i] = 0;
    cs->next_seq_num[i] = 0;
    cs->transmit_head[i] = NULL;
    cs->transmit_tail[i] = NULL;
    cs->timeout_factor[i] = 1 ;
  }
}

/* This routine adds a packet to the tail of the transmit queue */

static void InsertInTransmitQueue(packet, destpe)
DATA_HDR *packet;
int destpe;
{
  cmi_state cs = CmiState();
  PacketQueueElem *newelem;

  newelem = (PacketQueueElem *) CmiAlloc(sizeof(PacketQueueElem));
  newelem->packet = packet;
  newelem->nextptr = NULL;
  if  (cs->transmit_tail[destpe] == NULL)
    cs->transmit_head[destpe] = newelem;
  else 
    cs->transmit_tail[destpe]->nextptr = newelem;
  cs->transmit_tail[destpe] = newelem;
}


/* This routine returns the packet at the head of the transmit queue. If no
   packets exist, NULL is returned */

static DATA_HDR *GetTransmitPacket(destpe)
int destpe;
{
  cmi_state cs = CmiState();
  PacketQueueElem *elem;
  DATA_HDR *packet;

  if  (cs->transmit_head[destpe] == NULL)
    return NULL;
  elem = cs->transmit_head[destpe];
  cs->transmit_head[destpe] = cs->transmit_head[destpe]->nextptr;
  if (cs->transmit_head[destpe] == NULL)
    cs->transmit_tail[destpe] = NULL;
  packet = elem->packet;
  CmiFree(elem);
  return packet;
}

/* This routine adds the packet to the send window. If the addition is
   successful, the function returns 1. If the window is full, it returns
   0 - the calling function will then have to insert the packet into 
   the transmit queue */

static int AddToSendWindow(packet, destpe)
DATA_HDR *packet;
int destpe;
{
  cmi_state cs = CmiState();
  int lwi = cs->last_window_index[destpe];
  cs->send_window[destpe][lwi].packet = packet;
  cs->send_window[destpe][lwi].seq_num = cs->next_seq_num[destpe];
  cs->send_window[destpe][lwi].send_time = cs->CmiNow;
  packet->seq_num = cs->next_seq_num[destpe];
  cs->next_seq_num[destpe]++;
  cs->last_window_index[destpe] = (lwi + 1) % CMK_DGRAM_WINDOW_SIZE;
  cs->cur_window_size[destpe]++;
  return 1;
}

static SendPackets(destpe)
int destpe;
{
  cmi_state cs = CmiState();
  DATA_HDR *GetTransmitPacket();
  DATA_HDR *packet;
  int bytes_sent;
  unsigned int act_size;
  int i;

  cs->CmiNow = CmiWallTimer();
  while (cs->cur_window_size[destpe] < CMK_DGRAM_WINDOW_SIZE &&
     ((packet = GetTransmitPacket(destpe)) != NULL))
  {
    AddToSendWindow(packet, destpe);
    TRACE(CmiPrintf("Node %d: sending packet seq_num=%d, rem_size=%d\n",
           CmiMyPe(),
           packet->seq_num, 
           ntohl(packet->rem_size)));
    
    cs->NumSends++ ;
    act_size=(packet->rem_size<MAXDSIZE)?packet->rem_size:MAXDSIZE;
#if CMK_IS_HETERO
    packet->seq_num = htonl(packet->seq_num);
    packet->PeNum = htonl(packet->PeNum);
    packet->pktidx = htonl(packet->pktidx);
    packet->rem_size = htonl(packet->rem_size);
#endif
    bytes_sent = my_sendto(cs->data_skt, (char *)packet,
          act_size + sizeof(DATA_HDR),
          0, (struct sockaddr *)&(cs->node_table[destpe].addr),
          sizeof(struct sockaddr_in));
  }
}

/* This routine updates the send window upon receipt of an ack. Old acks
   are ignored */

static UpdateSendWindow(ack, sourcepe)
DATA_HDR *ack;
int sourcepe;
{
  cmi_state cs = CmiState();
  int i, index, found, count;
  DATA_HDR *PacketsToBeFreed[CMK_DGRAM_WINDOW_SIZE];

  if (cs->cur_window_size[sourcepe] == 0)  /* empty window */
    return;

  index = cs->first_window_index[sourcepe];
  found = 0;
  count = 0;
  while (count < cs->cur_window_size[sourcepe] && !found) 
  {
    found = (cs->send_window[sourcepe][index].seq_num == ack->seq_num);
    index = (index + 1) % CMK_DGRAM_WINDOW_SIZE;
    count++;
  }
  if (found) {
    TRACE(CmiPrintf("Node %d: received ack with seq_num %d\n",
           CmiMyPe(), ack->seq_num)); 
    index = cs->first_window_index[sourcepe];
    for (i = 0; i < count; i++)
    {
      PacketsToBeFreed[i] = cs->send_window[sourcepe][index].packet;
      cs->send_window[sourcepe][index].packet = NULL;
      index = (index + 1) % CMK_DGRAM_WINDOW_SIZE;
    }
    cs->first_window_index[sourcepe] = index;
    cs->cur_window_size[sourcepe] -= count;
  }
  SendPackets(sourcepe);   /* any untransmitted pkts */
  if(found)
    for(i=0;i<count;i++)
	    CmiFree(PacketsToBeFreed[i]);
}

/* This routine extracts a packet with a given sequence number. It returns
   NULL if no packet with the given sequence number exists in the window 
   This will be used primarily if retransmission is required - the packet 
   is not removed from the window */

static int RetransmitPackets()
{
  cmi_state cs = CmiState();
  int i, index, fnord, act_size;
  DATA_HDR *packet;
  int sending=0;

  cs->CmiNow = CmiWallTimer();
  for (i = 0; i < CmiNumPes(); i++) {
    index = cs->first_window_index[i];
    if (cs->cur_window_size[i] > 0) {
      sending = 1;
      if ((cs->CmiNow - cs->send_window[i][index].send_time) > 
          (conf_resend_wait * cs->timeout_factor[i])) {
        if (conf_resend_wait * cs->timeout_factor[i] > conf_resend_fail) {
          KillEveryone("retransmission failed, timeout.");
        }
        packet = cs->send_window[i][index].packet;
        cs->NumRetransmits++;
        cs->NumSends++;
        act_size=(packet->rem_size<MAXDSIZE)?packet->rem_size:MAXDSIZE;
#if CMK_IS_HETERO
        packet->seq_num = htonl(packet->seq_num);
        packet->PeNum = htonl(packet->PeNum);
        packet->pktidx = htonl(packet->pktidx);
        packet->rem_size = htonl(packet->rem_size);
#endif
        my_sendto(cs->data_skt, (char *)packet,
            act_size + sizeof(DATA_HDR), 0,
	    (struct sockaddr *)&(cs->node_table[i].addr),
	    sizeof(struct sockaddr_in)); 
        cs->send_window[i][index].send_time = cs->CmiNow;
      }
    }
  }
  return sending;
}


/* This section implements a receive sliding window protocol for the IPnet 
   implementation of the Chare kernel. The modification to Chris Walquist's 
   implementation are as follows:

   1) An ack packet with sequence number N acknowledges all packets upto and
      including N. Thus, it is not necessary for all packets to be explicitly
      acknowledged.

   2) The sliding window will guarantee that packets are delivered in order
      once successfully received. 

   3) One receive window per sender is necessary.

              Questions, Comments and Criticisms to be directed to

             Balkrishna Ramkumar
            ramkumar@crhc.uiuc.edu
*/
   

static RecvWindowInit()
{
  cmi_state cs = CmiState();
  int numpe = CmiNumPes();
  int mype = CmiMyPe();
  int i,j;

  cs->recv_window = (WindowElement **)CmiAlloc(numpe*sizeof(WindowElement *));
  cs->next_window_index = (int *) CmiAlloc(numpe*sizeof(int));
  cs->expected_seq_num = (unsigned int *)CmiAlloc(numpe*sizeof(unsigned int));
  cs->recd_messages = (NewMessage *)CmiAlloc(numpe * sizeof(NewMessage));
  cs->needack = (int *) CmiAlloc(numpe * sizeof(int));
  for (i = 0; i < numpe; i++) {
    if (i != mype) {
      cs->recv_window[i]=
	(WindowElement *)CmiAlloc(CMK_DGRAM_WINDOW_SIZE*sizeof(WindowElement));
      for (j = 0; j < CMK_DGRAM_WINDOW_SIZE; j++) {
        cs->recv_window[i][j].packet = NULL;
        cs->recv_window[i][j].seq_num = 0;
        cs->recv_window[i][j].send_time = 0.0;
      }
    }
    else 
      cs->recv_window[i] = NULL;  /* never used */

    cs->next_window_index[i] = 0;
    cs->expected_seq_num[i] = 0;
    cs->recd_messages[i].numpackets = 0;
    cs->recd_messages[i].numfrags = 0;
    cs->recd_messages[i].mesg = NULL;
    cs->needack[i] = 0;
  }
  cs->recd_msg_head = NULL;
  cs->recd_msg_tail = NULL;

  cs->ack.PeNum = CmiMyPe();
  cs->ack.rem_size = 0;
}




/* This routine tries to add an incoming packet to the recv window. 
   If the packet has already been received, the routine discards it and
   returns 0. Otherwise the routine inserts it into the window and
   returns 1
*/

static int AddToReceiveWindow(packet, sourcepe)
     DATA_HDR *packet;
     int sourcepe;
{
  cmi_state cs = CmiState();
  int index;
  unsigned int seq_num = packet->seq_num;
  unsigned int last_seq_num =
    cs->expected_seq_num[sourcepe] + CMK_DGRAM_WINDOW_SIZE - 1;

  /* 
     Note that seq_num cannot be > last_seq_num.
     Otherwise,  > last_seq_num - CMK_DGRAM_WINDOW_SIZE has been ack'd.
     If that were the case, 
     expected_seq_num[sourcepe] > > last_seq_num - CMK_DGRAM_WINDOW_SIZE,
     which is a contradiction
   */

  if (cs->expected_seq_num[sourcepe] < last_seq_num) {
    if (seq_num < cs->expected_seq_num[sourcepe]) {
      CmiFree(packet);  /* already received */
      cs->needack[sourcepe] = 1;
      cs->NumUseless++ ;
      return 0;
    }
    else {
      index = (cs->next_window_index[sourcepe] + 
      seq_num - cs->expected_seq_num[sourcepe]) % CMK_DGRAM_WINDOW_SIZE;
      /* put needack and NumUseless++ here ??? */
      if (cs->recv_window[sourcepe][index].packet) 
        CmiFree(packet);
      else {
        cs->recv_window[sourcepe][index].packet = packet;
        cs->recv_window[sourcepe][index].seq_num = seq_num;
        TRACE(CmiPrintf("Node %d: Inserting packet %d at index %d in recv window\n",
               CmiMyPe(),
               seq_num, index)); 
      }
    }
  }
  return 1;
}

/* this routine supplies the next packet in the receive window and updates
   the window. The window entry - packet + sequence number, is no longer
   available. If the first entry in the window is a null packet - i.e. it
   has not arrived, the routine returns NULL 
   */

static DATA_HDR *ExtractNextPacket(sourcepe)
     int sourcepe;
{
  cmi_state cs = CmiState();
  DATA_HDR *packet;
  int index = cs->next_window_index[sourcepe];

  packet = cs->recv_window[sourcepe][index].packet;
  if (packet != NULL) {
    cs->recv_window[sourcepe][index].packet = NULL;
    cs->needack[sourcepe] = 1;
    cs->expected_seq_num[sourcepe]++;
    cs->next_window_index[sourcepe]=
      (cs->next_window_index[sourcepe] + 1) % CMK_DGRAM_WINDOW_SIZE;
  }
  return packet;
}


/* This routine inserts a completed messages into the recd_msg queue */

static InsertInMessageQueue(mesg)
     char *mesg;
{
  cmi_state cs = CmiState();
  MsgQueueElem *newelem;

  newelem = (MsgQueueElem *) CmiAlloc(sizeof(MsgQueueElem));
  newelem->mesg = mesg;
  newelem->nextptr = NULL;
  if  (cs->recd_msg_tail == NULL)
    cs->recd_msg_head = newelem;
  else 
    cs->recd_msg_tail->nextptr = newelem;
  cs->recd_msg_tail = newelem;
}


static void ConstructMessages()
{
  cmi_state cs = CmiState();
  int i;
  DATA_HDR *packet;
  DATA_HDR *ExtractNextPacket();
  NewMessage *msg;
  unsigned int numfrags, msglength, rsize, offset;

  TRACE(CmiPrintf("Node %d: in ConstructMessages().\n",CmiMyPe()));
  for (i=0; i < CmiNumPes(); i++) {
    if (i != CmiMyPe()) {
      msg = cs->recd_messages+i;
      packet = ExtractNextPacket(i);
      while (packet != NULL) {
        offset = packet->pktidx*MAXDSIZE;
        rsize = packet->rem_size;
        if(msg->numpackets==0) {
          msglength = offset + packet->rem_size;
          msg->mesg = (char *) CmiAlloc(msglength);
          msg->numfrags = packet->pktidx + 
                 ((packet->rem_size-1)/MAXDSIZE) + 1;
        }
        memcpy(msg->mesg+offset,packet+1,(rsize<MAXDSIZE)?rsize:MAXDSIZE);
        msg->numpackets++;
        CmiFree(packet);
        if (msg->numpackets == msg->numfrags) {
          /* msg complete */
          InsertInMessageQueue(msg->mesg);
          msg->numpackets = 0;
        }
        packet = ExtractNextPacket(i);
      }
    }
  }
}



static void AckReceivedMsgs()
{
  cmi_state cs = CmiState();
  int i;
  for (i = 0; i < CmiNumPes(); i++) {
    if (cs->needack[i])  {
      cs->needack[i] = 0;
      cs->ack.seq_num = cs->expected_seq_num[i] - 1;
      if (CmiMyPe() < CmiNumPes())
        TRACE(CmiPrintf("Node %d: acking seq_num %d on window %d\n",
            CmiMyPe(), ack.seq_num, i)); 
      cs->NumAcksSent++ ;
      cs->ack.seq_num = cs->ack.seq_num;
      send_ack(&(cs->ack),i);
    }
  }
}

#define shift(root, num) ((num+total-root)%CmiNumPes())
#define unshift(root, num) ((num+root)%CmiNumPes())

static void my_children(root, pchild1, pchild2)
    int root, *pchild1, *pchild2;
{
  int mynum = CmiMyPe();
  int total = CmiNumPes();
  int p1, p2;

  mynum = shift(root, mynum);
  p1 = 2*mynum+1;
  p2 = 2*mynum+2;
  if(p1 >= total) 
	 *pchild1 = (-1);
  else
    *pchild1 = unshift(root, p1);
  if(p2 >= total) 
	 *pchild2 = (-1);
  else
    *pchild2 = unshift(root, p2);
}

static int data_getone()
{
  cmi_state cs = CmiState();
  msgspace *recv_buf=NULL;
  struct sockaddr_in src;
  int i, srclen=sizeof(struct sockaddr_in);
  node_info *sender; int kind;
  int AddToReceiveWindow();
  int n;

  recv_buf = (msgspace *)CmiAlloc(CMK_DGRAM_MAX_SIZE);
  do n=recvfrom(cs->data_skt,(char *)recv_buf,CMK_DGRAM_MAX_SIZE,0,(struct sockaddr *)&src,&srclen);
  while ((n<0)&&(errno==EINTR));
  if (n<0) { KillEveryone(strerror(errno)); }
  recv_buf->hd.seq_num = ntohl(recv_buf->hd.seq_num);
  recv_buf->hd.PeNum = ntohl(recv_buf->hd.PeNum);
  recv_buf->hd.pktidx = ntohl(recv_buf->hd.pktidx);
  recv_buf->hd.rem_size = ntohl(recv_buf->hd.rem_size);
  kind = (recv_buf->hd.rem_size)?SEND:ACK;
  if (kind == ACK) {
    UpdateSendWindow(recv_buf, (int) recv_buf->hd.PeNum);
    CmiFree(recv_buf);
  } else if (kind == SEND) {
    sender = cs->node_table + recv_buf->hd.PeNum;
    if((sender->dataport)&&(sender->dataport!=htons(src.sin_port)))
      KillEveryoneCode(38473);
    AddToReceiveWindow(recv_buf, (int) recv_buf->hd.PeNum);
  } 
  return kind;
}

static int dgram_scan()
{
  cmi_state cs = CmiState();
  fd_set rfds;
  struct timeval tmo;
  int nreadable, gotsend=0;

  while (1) {
    FD_ZERO(&rfds);
    FD_SET(cs->data_skt, &rfds);
    FD_SET(cs->ctrl_skt, &rfds);
    tmo.tv_sec = 0;
    tmo.tv_usec = 0;
    do nreadable = select(FD_SETSIZE, &rfds, NULL, NULL, &tmo);
    while ((nreadable<0)&&(errno==EINTR));
    if (nreadable <= 0) break;
    if (FD_ISSET(cs->ctrl_skt, &rfds))
      ctrl_getone();
    if (FD_ISSET(cs->data_skt, &rfds)) {
      int kind = data_getone();
      if (kind==SEND) gotsend=1;
    }
  }
  if (gotsend) {
    ConstructMessages();
    AckReceivedMsgs();
  }
  return gotsend;
}

#if CMK_ASYNC_DOESNT_WORK_USE_TIMER_INSTEAD

static int ticker_countup = 0;

static void ticker_reset()
{
  cmi_state cs = CmiState();
  struct itimerval i;
  if (ticker_countup < 8) {
    i.it_interval.tv_sec = 0;
    i.it_interval.tv_usec = 25000;
    i.it_value.tv_sec = 0;
    i.it_value.tv_usec = 25000;
  } else {
    i.it_interval.tv_sec = 0;
    i.it_interval.tv_usec = 250000;
    i.it_value.tv_sec = 0;
    i.it_value.tv_usec = 250000;
  }
  setitimer(ITIMER_REAL, &i, NULL);
}

static void InterruptHandler()
{
  cmi_state cs = CmiState();
  int prevmask ;
  int dgram_scan();
  if (CmiMemBusy() || (cs->busy)) return;
  cs->NumIntrCalls++;
  cs->NumOutsideMc++;
  ticker_countup++;
  sighold(SIGALRM);
  
  if (dgram_scan()) ticker_countup = 0;
  RetransmitPackets();
  ticker_reset();

  sigrelse(SIGALRM);
  cs->NumIntr++;
}

static void InterruptInit()
{
  if (conf_enableinterrupts) {
    CmiSignal(SIGALRM, InterruptHandler);
    ticker_reset();
  }
}

#else

static void InterruptHandler()
{
  cmi_state cs = CmiState();
  int prevmask;
  int dgram_scan();

  if (CmiMemBusy() || (cs->busy)) return;

  cs->NumIntrCalls++;
  cs->NumOutsideMc++;
  sighold(SIGIO) ;
  
  dgram_scan();
  RetransmitPackets();

  sigrelse(SIGIO) ;
  cs->NumIntr++ ;
}

static void InterruptInit()
{
  cmi_state cs = CmiState();
  if (conf_enableinterrupts) {
    CmiSignal(SIGIO, InterruptHandler);
    CmiEnableAsyncIO(cs->data_skt);
  }
}

#endif

/*
** Synchronized send of "msg", of "size" bytes, to "destPE".
** -CW
*/
static int netSend(destPE, size, msg) 
     int destPE, size; 
     char * msg; 
{
  cmi_state cs = CmiState();
  DATA_HDR *hd;
  unsigned int pktnum = 0;
  unsigned int numfrag = ((size-1)/MAXDSIZE) + 1;

  cs->busy++;
  
  if (!cs->Communication_init) return -1;
  
  if (destPE==CmiMyPe()) {
    CmiPrintf("netSend to self illegal.\n");
    exit(1);
  } 
  
  for(;pktnum<(numfrag-1);pktnum++) {
    hd = (DATA_HDR *)CmiAlloc(CMK_DGRAM_MAX_SIZE);
    hd->pktidx = pktnum;
    hd->rem_size = size;
    hd->PeNum = CmiMyPe();
    memcpy((hd+1), msg, MAXDSIZE);
    InsertInTransmitQueue(hd, destPE);
    msg += MAXDSIZE;
    size -= MAXDSIZE;
  }
  hd = (DATA_HDR *)CmiAlloc(sizeof(DATA_HDR)+size);
  hd->pktidx = pktnum;
  hd->rem_size = size;
  hd->PeNum = CmiMyPe();
  memcpy((hd+1), msg, size);
  InsertInTransmitQueue(hd, destPE);
  
  SendPackets(destPE);
  
  cs->busy--;
}

static int netBcast(size, msg) 
     int size; 
     char * msg; 
{
  cmi_state cs = CmiState();
  DATA_HDR *hd,*hd2;
  unsigned int pktnum = 0;
  unsigned int numfrag = ((size-1)/MAXDSIZE) + 1;
  unsigned int destPE1, destPE2;

  my_children(CmiMyPe(), &destPE1, &destPE2);
  if(destPE1==(-1) || destPE2==(-1)){
	 if(destPE1!=(-1))
		netSend(destPE1, size, msg);
    if(destPE2!=(-1))
		netSend(destPE2, size, msg);
    return 1;
  }

  cs->busy++;
  
  if (!cs->Communication_init) return -1;
  
  for(;pktnum<(numfrag-1);pktnum++) {
    hd = (DATA_HDR *)CmiAlloc(CMK_DGRAM_MAX_SIZE);
    hd->pktidx = pktnum;
    hd->rem_size = size;
    hd->PeNum = (1<<31) | (CmiMyPe()<<16) | (CmiMyPe());
    memcpy((hd+1), msg, MAXDSIZE);
	 hd2 = (DATA_HDR *)CmiAlloc(CMK_DGRAM_MAX_SIZE);
	 memcpy(hd2,hd,CMK_DGRAM_MAX_SIZE);
    InsertInTransmitQueue(hd, destPE1);
    InsertInTransmitQueue(hd2, destPE2);
    msg += MAXDSIZE;
    size -= MAXDSIZE;
  }
  hd = (DATA_HDR *)CmiAlloc(sizeof(DATA_HDR)+size);
  hd->pktidx = pktnum;
  hd->rem_size = size;
  hd->PeNum = (1<<31) | (CmiMyPe()<<16) | (CmiMyPe());
  memcpy((hd+1), msg, size);
  hd2 = (DATA_HDR *)CmiAlloc(sizeof(DATA_HDR)+size);
  memcpy(hd2,hd,sizeof(DATA_HDR)+size);
  InsertInTransmitQueue(hd, destPE1);
  InsertInTransmitQueue(hd2, destPE2);
  
  SendPackets(destPE1);
  SendPackets(destPE2);
  
  cs->busy--;
}

static int netSendV(destPE, n, sizes, msgs) 
     int destPE, n, *sizes; 
     char **msgs; 
{
  cmi_state cs = CmiState();
  DATA_HDR *hd;
  unsigned int pktnum = 0;
  unsigned int numfrag;
  unsigned int size,cur,cursize;
  unsigned int remfill;
  char *tmpsrc, *tmpdst;

  for(size=0,cur=0;cur<n;size += sizes[cur++]);
  numfrag  = ((size-1)/MAXDSIZE) + 1;

  cs->busy++;
  
  if (!cs->Communication_init) return -1;
  
  if (destPE==CmiMyPe()) {
    CmiPrintf("netSend to self illegal.\n");
    exit(1);
  } 
  
  cur = 0;
  cursize = sizes[0];
  tmpsrc = msgs[0];

  for(;pktnum<(numfrag-1);pktnum++) {
    hd = (DATA_HDR *)CmiAlloc(CMK_DGRAM_MAX_SIZE);
    hd->pktidx = pktnum;
    hd->rem_size = size;
    hd->PeNum = CmiMyPe();
	remfill = MAXDSIZE;
	tmpdst = (char *) (hd+1);
	while(remfill>0){ /* fill the datagram here */
		if(remfill > cursize) { /*entire msg to be copied */
			memcpy(tmpdst, tmpsrc, cursize);
			remfill -= cursize;
			tmpdst += cursize;
			cursize = sizes[++cur];
			tmpsrc = msgs[cur];
		}
		else{ /*partial message to be copied*/
			memcpy(tmpdst, tmpsrc, remfill);
			tmpsrc += remfill;
			cursize -= remfill;
			remfill = 0;
		}
	}
    InsertInTransmitQueue(hd, destPE);
    size -= MAXDSIZE;
  }
  hd = (DATA_HDR *)CmiAlloc(sizeof(DATA_HDR)+size);
  hd->pktidx = pktnum;
  hd->rem_size = size;
  hd->PeNum = CmiMyPe();
  remfill = size;
  tmpdst = (char *) (hd+1);
  while(remfill>0){ /* fill the datagram here */
    if(remfill > cursize) { /*entire msg to be copied */
      memcpy(tmpdst, tmpsrc, cursize);
      remfill -= cursize;
      tmpdst += cursize;
      cursize = sizes[++cur];
      tmpsrc = msgs[cur];
    }
    else{ /*partial message to be copied*/
      memcpy(tmpdst, tmpsrc, remfill);
      tmpsrc += remfill;
      cursize -= remfill;
      remfill = 0;
    }
  }
  InsertInTransmitQueue(hd, destPE);
  
  SendPackets(destPE);
  
  cs->busy--;
}

static int CmiProbe() 
{
  int val;
  int dgram_scan();
  cmi_state cs = CmiState();

  cs->busy++;

  dgram_scan();
  RetransmitPackets();
  val = (cs->recd_msg_head != NULL);

  cs->busy--;
  return (val);
}

void *CmiGetNonLocal()
{
  cmi_state cs = CmiState();
  MsgQueueElem *msgelem;
  char *newmsg=NULL;
  int i;

  cs->busy++;

  dgram_scan();
  RetransmitPackets();
  if (cs->recd_msg_head!=NULL) {
    msgelem = cs->recd_msg_head;
    newmsg = msgelem->mesg;
    if (cs->recd_msg_head == cs->recd_msg_tail) {
      cs->recd_msg_head = NULL;
      cs->recd_msg_tail = NULL;
    }
    else 
      cs->recd_msg_head = cs->recd_msg_head->nextptr;
    CmiFree(msgelem);
  } 
  cs->busy--;
  return newmsg;
}

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char *msg;
{
  if (CmiMyPe()==destPE) {
    char *msg1 = (char *)CmiAlloc(size);
    memcpy(msg1,msg,size);
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg1);
  } else netSend(destPE,size,msg);
}

void CmiSyncVectorSend(destPE, n, sizes, msgs)
int destPE, n;
int *sizes;
char **msgs;
{
  int total_size,i;

  if (CmiMyPe()==destPE) {
    char *msg1,*tmp;
	for(total_size=0,i=0;i<n;total_size += sizes[i++]);
	tmp = msg1 = (char *)CmiAlloc(total_size);
	for(i=0;i<n;i++) {
    	memcpy(tmp,msgs[i],sizes[i]);
		tmp += sizes[i];
	}
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg1);
  } else netSendV(destPE,n,sizes,msgs);
}

CmiCommHandle CmiAsyncVectorSend(destPE, n, sizes, msgs)
int destPE, n;
int *sizes;
char **msgs;
{
  CmiSyncVectorSend(destPE,n,sizes,msgs);
  return NULL;
}

void CmiSyncVectorSendAndFree(destPE, n, sizes, msgs)
int destPE, n;
int *sizes;
char **msgs;
{
  int i;
  CmiSyncVectorSend(destPE,n,sizes,msgs);
  for(i=0;i<n;i++) CmiFree(msgs[i]);
  CmiFree(sizes);
  CmiFree(msgs);
}

CmiCommHandle CmiAsyncSendFn(destPE, size, msg)
     int destPE, size;
     char *msg;
{
  CmiSyncSendFn(destPE, size, msg);
  return NULL;
}

void CmiFreeSendFn(destPE, size, msg)
     int destPE, size;
     char *msg;
{
  if (CmiMyPe()==destPE) {
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
  } else {
    CmiSyncSendFn(destPE, size, msg);
    CmiFree(msg);
  }
}

void CmiSyncBroadcastFn(size,msg)
     int size; char *msg;
{
  int i;
  for (i=0;i<CmiNumPes();i++) {
    if (i != CmiMyPe()) 
      netSend(i,size,msg);
  }
}

CmiCommHandle CmiAsyncBroadcastFn(size, msg)
     int size; char *msg;
{
  CmiSyncBroadcastFn(size, msg);
  return 0;
}

void CmiFreeBroadcastFn(size,msg)
     int size; char *msg;
{
  int i;
  for (i=0;i<CmiNumPes();i++) {
    if (i != CmiMyPe()) 
      netSend(i,size,msg);
  }
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(size,msg)
     int size; char *msg;
{
  int i;
  char *msg1;
  for (i=0;i<CmiNumPes();i++)
    if (i != CmiMyPe()) 
      netSend(i,size,msg);
  msg1 = (char *)CmiAlloc(size);
  memcpy(msg1,msg,size);
  FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg1);
}

CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
     int size; char *msg;
{
  CmiSyncBroadcastAllFn(size, msg);
  return 0;
}

void CmiFreeBroadcastAllFn(size,msg)
     int size; char *msg;
{
  int i;
  for (i=0;i<CmiNumPes();i++)
    if (i != CmiMyPe()) 
      netSend(i,size,msg);
  FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
}

static void CmiSleep()
{
  if (conf_enableinterrupts) sigpause(0L);
}

int CmiAsyncMsgSent(handle)
     CmiCommHandle handle;
{
  return 1;
}

void CmiReleaseCommHandle(handle)
     CmiCommHandle handle;
{
}

/******************************************************************************
 *
 * CmiInitMc and CmiExit
 *
 *****************************************************************************/

CmiInitMc(argv)
char **argv;
{
  void *FIFO_Create();
  cmi_state cs = CmiState();
  
  cs->Communication_init = 0;
  cs->busy++;
  CpvInitialize(void *,CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = FIFO_Create();
  skt_datagram(&cs->data_port, &cs->data_skt);
  skt_server(&cs->ctrl_port, &cs->ctrl_skt);
  KillInit();
  ctrl_sendone(120,"notify-die %s %d\n",conf_self_IP_str,cs->ctrl_port);
  outlog_init(conf_outfile, CmiMyPe());
  node_addresses_receive();
  CmiTimerInit();
  SendWindowInit();
  RecvWindowInit();
  InterruptInit();
  cs->Communication_init = 1;
  cs->busy--;
}

CmiExit()
{
  int begin;
  cmi_state cs = CmiState();
  
  cs->busy++;
  ctrl_sendone(120,"aget %s %d done 0 %d\n",
         conf_self_IP_str,cs->ctrl_port,CmiNumPes()-1);
  ctrl_sendone(120,"aset done %d TRUE\n",CmiMyPe());
  ctrl_sendone(120,"ending\n");
  begin = time(0);
  while(!cs->all_done && (time(0)<begin+120))
    { RetransmitPackets(); dgram_scan(); sleep(1); }
  outlog_done();
  cs->busy--;
}

#if CMK_SHARED_VARS_UNAVAILABLE
main(argc, argv)
int argc;
char **argv;
{
  int i, pid;

#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  conf_parse_netstart();
  conf_extract_args(argv);

  for (i=Cmi_nodesize-1; i>=0; i--) {
    if (i!=0) pid=fork();
    if ((i==0)||(pid==0)) {
      Cmi_mype = conf_node_start + i;
      Cmi_myrank = 0;
      CmiState()->mype = Cmi_mype;
      CmiState()->myrank = 0;
      CmiState()->busy = 0;
      user_main(argc, argv);
      exit(0);
    }
  }
}
#endif

#if CMK_SHARED_VARS_SUN_THREADS

static void *call_user_main(void *vindex)
{
  int index = (int)vindex;
  cmi_state state = (cmi_state)malloc(sizeof(struct Cmi_state_struct));
  thr_setspecific(cmi_state_key, state);
  state->mype = conf_node_start + index;
  state->myrank = index;
  state->busy = 0;
  user_main(conf_argc, conf_argv);
  thr_exit(0);
}

main(argc, argv)
int argc;
char **argv;
{
  int i, ok; thread_t pid;
  
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  conf_argc = argc;
  conf_argv = argv;
  conf_parse_netstart();
  conf_extract_args(argv);
  thr_keycreate(&cmi_state_key, 0);
  for (i=0; i<Cmi_nodesize; i++) {
    thr_create(0, 256000, call_user_main, (void *)i,
	       THR_DETACHED|THR_BOUND, &pid);
    if (ok<0) { perror("thr_create"); exit(1); }
  }
  thr_exit(0);
}
#endif

