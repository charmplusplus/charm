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
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <rpc/rpc.h>
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

#ifdef DEBUG
#define TRACE(p) p
#else
#define TRACE(p)
#endif

extern int CmemInsideMem();
extern void CmemCallWhenMemAvail();
static void CmiSleep();

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
 *     Utility routines for network machine interface.
 *
 *
 * Bcopy(char *s1, char *s2, int size)
 *
 *    - Temporary bcopy routine.  There seems to be a problem with the
 *      usual one
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
 *****************************************************************************/


static void Bcopy (s1,s2,size) char *s1,*s2; int size;
{
  int i;
  for (i=0;i<size;i++) s2[i]=s1[i];
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

static int readall(fd, buff, len)
     int fd; char *buff; int len;
{
  int nread, i;
  while (len>0) {
    nread = read(fd, buff, len);
    if (nread==0) return -1;
    if ((nread<0)&&(errno!=EWOULDBLOCK)&&(errno!=EAGAIN))
      KillEveryoneCode(43491550);
    if (nread>0) { len -= nread; buff += nread; }
  }
  return 0;
}

static void writeall(fd, buff, len)
     int fd; char *buff; int len;
{
  int nwrote;
  while (len > 0) {
    nwrote = write(fd, buff, len);
    if (nwrote==0)
      KillEveryoneCode(83794837);
    if ((nwrote<0)&&(errno!=EWOULDBLOCK)&&(errno!=EAGAIN))
      KillEveryoneCode(43491550);
    if (nwrote>0) { len -= nwrote; buff += nwrote; }
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
 retry:
  tmo.tv_sec = (time(0) - begin) + sec;
  tmo.tv_usec = 0;
  nreadable = select(FD_SETSIZE, &rfds, NULL, NULL, &tmo);
  if ((nreadable<0)&&(errno==EINTR)) goto retry;
  if (nreadable == 0) { errno=ETIMEDOUT; return -1; }
  return 0;
}

/*****************************************************************************
 *
 * Logging Module
 *
 *****************************************************************************/

static FILE *outlog_file;

static void outlog_init(outputfile, index) char *outputfile; int index;
{
  char fn[MAXPATHLEN];
  if (outputfile) {
    sprintf(fn,outputfile,index);
    outlog_file = fopen(fn,"w");
  }
  else outlog_file = 0;
}

static void outlog_done()
{
  fclose(outlog_file);
}

static void outlog_output(buf) char *buf;
{
  if (outlog_file) {
    fprintf(outlog_file,"%s",buf);
    fflush(outlog_file);
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
 *     procedure.  Returns the IP address of the socket (eg, the IP of the
 *     current machine), the port of the socket, and the file descriptor.
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

static void skt_accept(src, pip, ppo, pfd)
int src;
unsigned int *pip;
unsigned int *ppo;
unsigned int *pfd;
{
  int i, fd, ok, flag;
  struct sockaddr_in remote;
  i = sizeof(remote);
 acc:
  fd = accept(src, (struct sockaddr *)&remote, &i);
  if ((fd<0)&&(errno==EINTR)) goto acc;
  if (fd<0) { perror("accept"); KillEveryoneCode(39489); }
  flag = 1;
  ok = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
  if (ok<0) { perror("setsockopt"); KillEveryoneCode(99331); }
  *pip=htonl(remote.sin_addr.s_addr);
  *ppo=htons(remote.sin_port);
  *pfd=fd;
}

static int skt_connect(ip, port, seconds)
unsigned int ip; int port; int seconds;
{
  struct sockaddr_in remote; short sport=port;
  int fd, ok, len, retry, begin, flag;
    
  /* create an address structure for the server */
  memset(&remote, 0, sizeof(remote));
  remote.sin_family = AF_INET;
  remote.sin_port = htons(sport);
  remote.sin_addr.s_addr = htonl(ip);
    
  begin = time(0);
  while (time(0)-begin < seconds) {
  sock:
    fd = socket(AF_INET, SOCK_STREAM, 0);
    if ((fd<0)&&(errno==EINTR)) goto sock;
    if (fd < 0) { perror("socket"); exit(1); }
    flag = 1;
    ok = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    if (ok<0) { perror("setsockopt"); KillEveryoneCode(20033271); }
    
  conn:
    ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
    if ((ok<0)&&(errno==EINTR)) { close(fd); goto sock; }
    if ((ok<0)&&(errno==EADDRINUSE)) { close(fd); goto sock; }
    if (ok>=0) break;
    if (errno!=ECONNREFUSED) break;
    
    sleep(1);
  }
  if (ok<0) {
    perror("connect"); exit(1);
  }
  return fd;
}


/*****************************************************************************
 *
 * Readonly Data
 *
 *****************************************************************************/

static char  *outputfile;

CpvDeclare(int, Cmi_mype);
CpvDeclare(int, Cmi_numpes);

static int    host_IP;
static char   host_IP_str[16];
static int    host_port;
static int    self_IP;
static char   self_IP_str[16];
static int    ctrl_port, ctrl_skt;
static int    data_port, data_skt;

#define MAX_NODES 100

typedef struct commhandle {
  char *msg;
  char *ptr;
  char *end;
  int   autofree;
  int   done;
  struct commhandle *next;
} *commhandle;

typedef struct {
  int IP;
  int dataport;
  int ctrlport;
  int talk_skt;
  /* Incoming Message: a single buffer */
  char *icm_msg;
  char *icm_ptr;
  char *icm_end;
  /* Outgoing Message */ 
  commhandle ogm_head;
  commhandle ogm_tail;
} node_info;

/* Information table about host and other nodes. */
static node_info node_table[MAX_NODES];
static int       node_table_fill;


static void ParseNetstart()
{
  char *ns;
  int nread;
  ns = getenv("NETSTART");
  if (ns==0) goto abort;
  nread = sscanf(ns, "%d%d%d%d%d",
		 &CpvAccess(Cmi_mype),&CpvAccess(Cmi_numpes),&self_IP,&host_IP,&host_port);
  if (nread!=5) goto abort;
  return;
 abort:
  fprintf(stderr,"program not started using 'conv-host' utility. aborting.\n");
  exit(1);
}

static char *DeleteArg(argv)
char **argv;
{
  char *res = argv[0];
  if (res==0) KillEveryone("Illegal Arglist");
  while (*argv) { argv[0]=argv[1]; argv++; }
  return res;
}

static void ExtractArgs(argv)
char **argv;
{
  outputfile = NULL;

  while (*argv) {
    if (strcmp(*argv,"++outputfile")==0) {
      DeleteArg(argv); outputfile=DeleteArg(argv);
    } else
    argv++;
  }
}

static void InitializePorts()
{
  skt_server(&data_port, &data_skt);
  skt_server(&ctrl_port, &ctrl_skt);
  
  sprintf(self_IP_str,"%d.%d.%d.%d",
	  (self_IP>>24)&0xFF,(self_IP>>16)&0xFF,
	  (self_IP>>8)&0xFF,self_IP&0xFF);
  sprintf(host_IP_str,"%d.%d.%d.%d",
	  (host_IP>>24)&0xFF,(host_IP>>16)&0xFF,
	  (host_IP>>8)&0xFF,host_IP&0xFF);
}

/****************************************************************************
 *
 * ctrl_sendone && ctrl_getone
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
  fd = skt_connect(host_IP, host_port, delay);
  if (fd<0) KillEveryone("cannot contact host");
  write(fd, buffer, strlen(buffer));
  shutdown(fd, 1);
  while (read(fd, buffer, 1023)>0);
  close(fd);
}

static char *scanf_data = 0;
static int all_done = 0;
static void node_addresses_store();

static void ctrl_getone()
{
  char line[10000];
  int ok, ip, port, fd;  FILE *f;
  skt_accept(ctrl_skt, &ip, &port, &fd);
  f = fdopen(fd,"r");
  while (fgets(line, 9999, f)) {
    if      (strncmp(line,"aval addr ",10)==0) node_addresses_store(line);
    else if (strncmp(line,"aval done ",10)==0) all_done = 1;
    else if (strncmp(line,"scanf-data ",11)==0) scanf_data=strdupl(line+11);
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
 * Kill handling (much simpler now)
 *
 ****************************************************************************/

static void KillEveryone(char *s)
{
  ctrl_sendone(120, "die %s\n", s);
  exit(1);
}

static void KillEveryoneCode(n)
int n;
{
  ctrl_sendone(120, "die Internal error #%d (node %d)\n(Contact CHARM developers)\n", n,CpvAccess(Cmi_mype));
  exit(1);
}

static void KillOnSegv()
{
  ctrl_sendone(120, "die Node %d: Segmentation fault.\n",CpvAccess(Cmi_mype));
  exit(1);
}

static void KillOnIntr()
{
  ctrl_sendone(120, "die Node %d: Interrupted.\n",CpvAccess(Cmi_mype));
  exit(1);
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
  ctrl_sendone(120, "aset addr %d %s.%d.%d\n",
	       CpvAccess(Cmi_mype), self_IP_str, ctrl_port, data_port);
  ctrl_sendone(120, "aget %s %d addr 0 %d\n",
    self_IP_str,ctrl_port,CpvAccess(Cmi_numpes)-1);
  while (node_table_fill != CpvAccess(Cmi_numpes)) {
    if (wait_readable(ctrl_skt, 300)<0)
      { perror("waiting for data"); KillEveryoneCode(21323); }
    ctrl_getone();
  }
}

static void node_addresses_store(addrs) char *addrs;
{
  char *p, *e; int i, lo, hi;
  if (strncmp(addrs,"aval addr ",10)!=0) KillEveryoneCode(83473);
  p = skipblanks(addrs+10);
  p = readint(p,&lo);
  p = readint(p,&hi);
  if ((lo!=0)||(hi!=CpvAccess(Cmi_numpes)-1)) KillEveryoneCode(824793);
  for (i=0; i<CpvAccess(Cmi_numpes); i++) {
    unsigned int ip0,ip1,ip2,ip3,cport,dport;
    p = readint(p,&ip0);
    p = readint(p,&ip1);
    p = readint(p,&ip2);
    p = readint(p,&ip3);
    p = readint(p,&cport);
    p = readint(p,&dport);
    node_table[i].IP = (ip0<<24)+(ip1<<16)+(ip2<<8)+ip3;
    node_table[i].ctrlport = cport;
    node_table[i].dataport = dport;
    node_table[i].talk_skt = 0;
    node_table[i].icm_msg = 0;
    node_table[i].ogm_head = 0;
  }
  p = skipblanks(p);
  if (*p!=0) KillEveryoneCode(82283);
  node_table_fill = CpvAccess(Cmi_numpes);
}

/*****************************************************************************
 *
 * open_talk_sockets
 *
 * open one connection to every other node.
 *
 *****************************************************************************/

static void open_talk_sockets()
{
  int i, net_mype, net_hispe, ok, ip, port, skt, pe;
  net_mype = htonl(CpvAccess(Cmi_mype));
  for (i=0; i<CpvAccess(Cmi_mype); i++) {
    skt_accept(data_skt, &ip, &port, &skt);
    ok = readall(skt, &net_hispe, 4);
    if (ok<0) KillEveryoneCode(98246556);
    pe = htonl(net_hispe);
    node_table[pe].talk_skt = skt;
  }
  for (pe=CpvAccess(Cmi_mype)+1; pe<CpvAccess(Cmi_numpes); pe++) {
    skt = skt_connect(node_table[pe].IP, node_table[pe].dataport, 300);
    if (skt<0) KillEveryoneCode(894788843);
    writeall(skt, &net_mype, 4);
    node_table[pe].talk_skt = skt;
  }
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
  ctrl_sendone(120, "scanf %s %d %s", self_IP_str, ctrl_port, fmt);
  while (scanf_data==0) CmiSleep();
  i = sscanf(scanf_data, fmt,
         ptr[ 0], ptr[ 1], ptr[ 2], ptr[ 3], ptr[ 4], ptr[ 5],
         ptr[ 6], ptr[ 7], ptr[ 8], ptr[ 9], ptr[10], ptr[11],
         ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17]);
  CmiFree(scanf_data);
  scanf_data=0;
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
    if (neighbour < CpvAccess(Cmi_numpes)) count++;
    bit<<1; if (bit > CpvAccess(Cmi_numpes)) break;
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
    if (neighbour < CpvAccess(Cmi_numpes)) neighbours[count++] = neighbour;
    bit<<1; if (bit > CpvAccess(Cmi_numpes)) break;
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
    if (neighbour < CpvAccess(Cmi_numpes)) { if (nbr==neighbour) return count; count++; }
    bit<<=1; if (bit > CpvAccess(Cmi_numpes)) break;
  }
  return(-1);
}

/*****************************************************************************
 *
 * Low-Level Transmission and Reception routines
 *
 *****************************************************************************/

CpvDeclare(void *, CmiLocalQueue);
static fd_set fds_read;
static fd_set fds_write;

static CmiCommHandle CmiAllocCommHandle()
{
  CmiCommHandle h = (CmiCommHandle)malloc(sizeof(struct commhandle));
  return h;
}

void CmiReleaseCommHandle(handle)
     CmiCommHandle handle;
{
  free(handle);
}

/* Only call this when the descriptor is readable */
static void pump_icms(pe)
     int pe;
{
  int head[2], netlen, len, ok;
  node_info *ni = node_table + pe;
  int skt = ni->talk_skt;
  if (ni->icm_msg == 0) {
    ok = readall(skt, head, 8);
    if (ok<0) KillEveryoneCode(23323335);
    len = head[1];
    if (len<8) {
      CmiError("Error in message protocol: len=%d.\n",len);
      KillEveryoneCode(87949839);
    }
    ni->icm_msg = (char *)CmiAlloc(len);
    ni->icm_ptr = ni->icm_msg + 8;
    ni->icm_end = ni->icm_msg + len;
    ((int *)(ni->icm_msg))[0]=head[0];
  }
  len = ni->icm_end - ni->icm_ptr;
  if (len) {
    ok = read(skt, ni->icm_ptr, len);
    if ((ok<0)&&(errno!=EWOULDBLOCK)&&(errno!=EAGAIN))
      KillEveryoneCode(8966374);
    if (ok==0) exit(0);
    if (ok>0) {
      ni->icm_ptr += ok;
    }
  }
  if (ni->icm_ptr == ni->icm_end) {
    char *msg = ni->icm_msg;
    ni->icm_msg = 0;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
  }
}

/* Only call this when the descriptor is writeable */
static void pump_ogms(pe)
     int pe;
{
  int ok, bytes;
  node_info *ni = node_table + pe;
  commhandle h = ni->ogm_head;
  if (h) {
    int skt = ni->talk_skt;
    bytes = (h->end - h->ptr);
    if (bytes > 128000) bytes=128000;
    ok = write(skt, h->ptr, bytes);
    if (ok<0) {
      if ((errno!=EWOULDBLOCK)&&(errno!=EAGAIN))
	KillEveryoneCode(7458781);
    } else {
      h->ptr += ok;
      if (h->ptr == h->end) {
	h->done = 1;
	ni->ogm_head = h->next;
	if (ni->ogm_head==0) 
	  { FD_CLR(skt, &fds_write); }
	if (h->autofree) { CmiFree(h->msg); free(h); }
      }
    }
  }
}

int pumpmsgs()
{
  int i; fd_set r; fd_set w; int action=1;
  struct timeval tv; int ok, anyaction=0;
  while (action) {
    action = 0;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    /* INEFFICIENCY HERE */
    memcpy(&r, &fds_read, sizeof(r));
    memcpy(&w, &fds_write, sizeof(w));
    ok = select(FD_SETSIZE, &r, &w, 0, &tv);
    if (ok<=0) break;
    /* INEFFICIENCY HERE */
    for (i=0; i<CpvAccess(Cmi_numpes); i++)
      if (FD_ISSET(node_table[i].talk_skt, &w))
	{ pump_ogms(i); action=1; }
    for (i=0; i<CpvAccess(Cmi_numpes); i++)
      if (FD_ISSET(node_table[i].talk_skt, &r))
	{ pump_icms(i); action=1; }
    if (FD_ISSET(ctrl_skt, &r))
      ctrl_getone();
    if (action) anyaction=1;
  }
}

#if CMK_ASYNC_DOESNT_WORK_USE_TIMER_INSTEAD

static int ticker_countup = 0;

static void ticker_reset()
{
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
  CmiInterruptHeader(InterruptHandler);
  ticker_countup++;
  if (pumpmsgs()) ticker_countup=0;
  ticker_reset();
}

static void InterruptInit()
{
  int i;
  FD_ZERO(&fds_read);
  FD_ZERO(&fds_write);
  FD_SET(ctrl_skt, &fds_read);
  CmiSignal(SIGALRM, InterruptHandler);
  ticker_reset();
  for (i=0; i<CpvAccess(Cmi_numpes); i++) {
    if (i!=CpvAccess(Cmi_mype)) {
      FD_SET(node_table[i].talk_skt, &fds_read);
    }
  }
}

#else

static void InterruptHandler()
{
  CmiInterruptHeader(InterruptHandler);
  pumpmsgs();
}

static void InterruptInit()
{
  int i;
  FD_ZERO(&fds_read);
  FD_ZERO(&fds_write);
  FD_SET(ctrl_skt, &fds_read);
  CmiSignal(SIGIO, InterruptHandler);
  for (i=0; i<CpvAccess(Cmi_numpes); i++) {
    if (i!=CpvAccess(Cmi_mype)) {
      CmiEnableAsyncIO(node_table[i].talk_skt);
      FD_SET(node_table[i].talk_skt, &fds_read);
    }
  }
}

#endif


static commhandle netsend(autofree, destPE, size, msg) 
     int autofree, destPE, size; 
     char * msg; 
{
  node_info *ni = node_table+destPE;
  commhandle h = (commhandle)CmiAllocCommHandle();
  int skt = ni->talk_skt;
  if (size<CmiMsgHeaderSizeBytes)
    KillEveryone("Message too short (no converse header!)");
  h->msg        = msg;
  h->ptr        = msg;
  h->end        = msg+size;
  h->autofree   = autofree;
  h->done       = 0;
  h->next       = 0;
  ((int *)msg)[1]=size;
  CmiInterruptsBlock();
  if (ni->ogm_head==0) {
    ni->ogm_head = h;
    ni->ogm_tail = h;
    FD_SET(ni->talk_skt, &fds_write);
    pump_ogms(destPE);
  } else {
    ni->ogm_tail->next = h;
    ni->ogm_tail = h;
  }
  CmiInterruptsRelease();
  return h;
}

int CmiAsyncMsgSent(h)
     CmiCommHandle h;
{
  if (h==0) return 1;
  CmiInterruptsBlock();
  pumpmsgs();
  CmiInterruptsRelease();
  return ((commhandle)h)->done;
}

void *CmiGetNonLocal()
{
  CmiInterruptsBlock();
  pumpmsgs();
  CmiInterruptsRelease();
  return 0;
}

static void CmiSleep()
{
  CmiInterruptsBlock();
  pumpmsgs();
  CmiInterruptsRelease();
}

/*****************************************************************************
 *
 * The CmiDeliver routines.
 *
 * This version doesn't use the common CmiDelivers because it needs to
 * wrap access to CmiLocalQueue in a interrupt-blocking section.
 *
 ****************************************************************************/

CpvStaticDeclare(int, CmiBufferGrabbed);

void CmiDeliversInit()
{
  CpvInitialize(int, CmiBufferGrabbed);
  CpvAccess(CmiBufferGrabbed) = 0;
}

void CmiGrabBuffer()
{
  CpvAccess(CmiBufferGrabbed) = 1;
}

int CmiDeliverMsgs(maxmsgs)
int maxmsgs;
{
  void *msg;
  
  while (1) {
    CmiInterruptsBlock();
    pumpmsgs();
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    CmiInterruptsRelease();
    if (msg==0) break;
    CpvAccess(CmiBufferGrabbed)=0;
    (CmiGetHandlerFunction(msg))(msg);
    if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg);
    maxmsgs--; if (maxmsgs==0) break;
  }
  return maxmsgs;
}

/*
 * CmiDeliverSpecificMsg(lang)
 *
 * - waits till a message with the specified handler is received,
 *   then delivers it.
 *
 */

void CmiDeliverSpecificMsg(handler)
int handler;
{
  void *msg;
  CmiInterruptsBlock();
  while (1) {
    pumpmsgs();
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    if (msg) {
      if (CmiGetHandler(msg)==handler) {
	CpvAccess(CmiBufferGrabbed)=0;
	CmiInterruptsRelease();
	(CmiGetHandlerFunction(msg))(msg);
	if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg);
	return;
      } else FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
    }
  }
}

/*****************************************************************************
 *
 * High-Level Transmission and Reception routines
 *
 *****************************************************************************/


void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char *msg;
{
  if (CpvAccess(Cmi_mype)==destPE) {
    char *msg1 = (char *)CmiAlloc(size);
    memcpy(msg1,msg,size);
    CmiInterruptsBlock();
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg1);
    CmiInterruptsRelease();
  } else {
    commhandle h = netsend(0,destPE,size,msg);
    while (h->done == 0) CmiSleep();
    CmiReleaseCommHandle((CmiCommHandle)h);
  }
}

CmiCommHandle CmiAsyncSendFn(destPE, size, msg)
     int destPE, size;
     char *msg;
{
  if (CpvAccess(Cmi_mype)==destPE) {
    char *msg1 = (char *)CmiAlloc(size);
    memcpy(msg1,msg,size);
    CmiInterruptsBlock();
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg1);
    CmiInterruptsRelease();
    return NULL;
  } else {
    commhandle h = netsend(0,destPE,size,msg);
    return (CmiCommHandle)h;
  }
}

void CmiFreeSendFn(destPE, size, msg)
     int destPE, size;
     char *msg;
{
  if (CpvAccess(Cmi_mype)==destPE) {
    CmiInterruptsBlock();
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
    CmiInterruptsRelease();
  } else {
    netsend(1,destPE,size,msg);
  }
}

void CmiSyncBroadcastFn(size,msg)
     int size; char *msg;
{
  int i;
  for (i=0;i<CpvAccess(Cmi_numpes);i++) {
    if (i != CpvAccess(Cmi_mype))
      CmiSyncSendFn(i,size,msg);
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
  for (i=0;i<CpvAccess(Cmi_numpes);i++) {
    if (i != CpvAccess(Cmi_mype)) 
      CmiSyncSendFn(i,size,msg);
  }
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(size,msg)
     int size; char *msg;
{
  int i;
  char *msg1;
  for (i=0;i<CpvAccess(Cmi_numpes);i++)
    if (i != CpvAccess(Cmi_mype)) 
      CmiSyncSendFn(i,size,msg);
  msg1 = (char *)CmiAlloc(size);
  memcpy(msg1,msg,size);
  CmiInterruptsBlock();
  FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg1);
  CmiInterruptsRelease();
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
  for (i=0;i<CpvAccess(Cmi_numpes);i++)
    if (i != CpvAccess(Cmi_mype)) 
      CmiSyncSendFn(i,size,msg);
  CmiInterruptsBlock();
  FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
  CmiInterruptsRelease();
}

/******************************************************************************
 *
 * CmiInitMc and CmiExit
 *
 *****************************************************************************/

CmiInitMc(argv)
char **argv;
{
  static int initmc=0;
  void *FIFO_Create();
  
  CmiInterruptsBlock();
  if (initmc==1) KillEveryone("CmiInit called twice");
  initmc=1;
  
  ExtractArgs(argv);
  ParseNetstart();
  InitializePorts();
  CpvAccess(CmiLocalQueue) = FIFO_Create();
  KillInit();
  ctrl_sendone(120,"notify-die %s %d\n",self_IP_str,ctrl_port);
  outlog_init(outputfile, CpvAccess(Cmi_mype));
  node_addresses_receive();
  open_talk_sockets();
  CmiTimerInit();
  InterruptInit();
  pumpmsgs();
  CmiInterruptsRelease();
}

CmiExit()
{
  static int exited;
  int begin;
  
  if (exited==1) KillEveryone("CmiExit called twice");
  exited=1;
  CmiInterruptsBlock();
  ctrl_sendone(120,"aget %s %d done 0 %d\n",
	       self_IP_str,ctrl_port,CpvAccess(Cmi_numpes)-1);
  ctrl_sendone(120,"aset done %d TRUE\n",CpvAccess(Cmi_mype));
  ctrl_sendone(120,"ending\n");
  outlog_done();
  begin = time(0);
  while(!all_done && (time(0)<begin+120)) 
    ctrl_getone();
  CmiInterruptsRelease();
}

main(argc, argv)
int argc;
char **argv;
{
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
setbuf(stderr,0);
user_main(argc, argv);
}
