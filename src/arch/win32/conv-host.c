/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/


#include "conv-mach.h"

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include <sys/stat.h>
#include <time.h>
#include <winsock2.h>
#include <winbase.h>
#include <io.h>
#include <fcntl.h>
#include <errno.h>
#include <process.h>
#include <direct.h>
#include <string.h>
#include <stdarg.h>
#include "daemon.h"

#define  MAXPATHLEN   1024


#if CMK_STRINGS_USE_STRINGS_H
#include <strings.h>
#endif

#if CMK_STRINGS_USE_STRING_H
#include <string.h>
#endif

#if CMK_STRINGS_USE_OWN_DECLARATIONS
char *strchr(), *strrchr(), *strdup();
#endif

#if CMK_WAIT_USES_WAITFLAGS_H
#include <waitflags.h>
#endif
#if CMK_WAIT_USES_SYS_WAIT_H
#include <sys/wait.h>
#endif

#if CMK_STRERROR_USE_SYS_ERRLIST
extern char *sys_errlist[];
#define strerror(i) (sys_errlist[i])
#endif

#if CMK_RSH_IS_A_COMMAND
#define RSH_CMD "rsh"
#endif

#if CMK_RSH_USE_REMSH
#define RSH_CMD "remsh"
#endif

/* keep this value in sync with converse.h */
#define CMK_CCS_VERSION "1"

//#if CMK_CCS_AVAILABLE
SOCKET CcsClientFd;
SOCKET myFd;
unsigned int clientIP;
unsigned int clientKillPort;
//#endif

#define DEBUGF(x) /*printf x*/

int RecvSocketN(SOCKET hSocket,BYTE *pBuff,int nBytes)
{
  int nLeft;
  int nRead;
  int nTotal = 0;

  nLeft = nBytes;
  while (0 < nLeft)
  {
    nRead = recv(hSocket,pBuff,nLeft,0);
    if (SOCKET_ERROR == nRead)
    {
      return nRead;
    }
    else
    {
      nLeft -= nRead;
      pBuff += nRead;
      nTotal += nRead;
    }
  }

  return nTotal;
}

int SendSocketN(SOCKET hSocket,BYTE *pBuff,int nBytes)
{
  int nLeft,nWritten;
  int nTotal = 0;

  nLeft = nBytes;
  while (0 < nLeft)
  {
    nWritten = send(hSocket,pBuff,nLeft,0);
    if (SOCKET_ERROR == nWritten)
    {
      return nWritten;
    }
    else
    {
      nLeft -= nWritten;
      pBuff += nWritten;
      nTotal += nWritten;
    }
  }
  return nTotal;
}

static void jsleep(int sec, int usec)
{  
  int ntimes,i;
  struct timeval tm;

  ntimes = (sec*1000000+usec)/5000;
  for(i=0;i<ntimes;i++) 
  {
    tm.tv_sec = 0;
    tm.tv_usec = 5000;
    while(1) 
    {
      if (select(0,NULL,NULL,NULL,&tm)==0) break;
      if (WSAGetLastError()!= WSAEINTR) return;
    }
  }
}


int probefile(char *path)
{  
  struct _stat s;
  int ok = _stat(path, &s);
  if (ok<0) return 0;
  if ((s.st_mode & _S_IFMT) != _S_IFREG) return 0;
  if (s.st_size==0) return 0;
  return 1;
}


/**************************************************************************
 *
 * ping_developers
 *
 * Sends a single UDP packet to the charm developers notifying them
 * that charm is in use.
 *
 **************************************************************************/

void ping_developers()
{
#ifdef NOTIFY
  char               info[1000];
  struct sockaddr_in addr;
  int                infoSize = 999;
  SOCKET             skt;
  
  skt = socket(AF_INET, SOCK_DGRAM, 0);
  if (skt == INVALID_SOCKET) return;
  
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(6571);
  addr.sin_addr.s_addr = htonl(0x80aef1d3);
  sprintf(info,"%s",GetUserName(info, &infoSize));
  
  sendto(skt, info, strlen(info), 0, (struct sockaddr *)&addr, sizeof(addr));
  closesocket(skt);
#endif /* NOTIFY */
}

/**************************************************************************
 *
 * Pathfix : alters a path according to a set of rewrite rules
 *
 *************************************************************************/

typedef struct pathfixlist {
  char *s1;
  char *s2;
  struct pathfixlist *next;
} *pathfixlist;

pathfixlist pathfix_append(char *s1, char *s2, pathfixlist l)
{
  pathfixlist pf = (pathfixlist)malloc(sizeof(struct pathfixlist));
  pf->s1 = s1;
  pf->s2 = s2;
  pf->next = l;
  return pf;
}

char *pathfix(char *path, pathfixlist fixes)
{
  char buffer[MAXPATHLEN]; pathfixlist l; 
  char buf2[MAXPATHLEN]; 
  char *offs; int mod, len;
  strcpy(buffer,path);
  mod = 1;
  while (mod) {
    mod = 0;
    for (l=fixes; l; l=l->next) {
      len = strlen(l->s1);
      offs = strstr(buffer, l->s1);
      if (offs) {
  offs[0]=0;
  sprintf(buf2,"%s%s%s",buffer,l->s2,offs+len);
  strcpy(buffer,buf2);
  mod = 1;
      }
    }
  }
  return strdup(buffer);
}

/**************************************************************************
 *
 * SKT - socket routines
 *
 * Uses Module: SCHED  [implicitly TIMEVAL, QUEUE, THREAD]
 *
 *
 * unsigned int skt_ip()
 *
 *   - returns the IP address of the current machine.
 *
 * void skt_server(unsigned int *pip, unsigned int *ppo, unsigned int *pfd)
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
 * int skt_connect(unsigned int ip, int port)
 *
 *   - Opens a connection to the specified server.  Returns a socket for
 *     communication.
 *
 *
 **************************************************************************/

unsigned int skt_ip()
{  
  static unsigned int ip = 0;
  struct hostent *hostent;
  char hostname[100];
  
  if (ip==0) 
  {
    if (gethostname(hostname, 99)<0) ip=0x7f000001;
    hostent = gethostbyname(hostname);
    if (hostent == 0) return 0x7f000001;
    ip = *((int *)(hostent->h_addr_list[0]));
  }
  return ip;
}



void skt_server(unsigned int *pip, unsigned int *ppo, SOCKET *pfd)
{  
  SOCKET             fd = INVALID_SOCKET;
  int                ok, len;
  struct sockaddr_in addr;
  
  fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd == INVALID_SOCKET) { fprintf(stderr, "ERROR> socket: %d", WSAGetLastError()); exit(1); }
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  if(*ppo) { addr.sin_port = htons(*ppo); }
  ok = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  if (ok == SOCKET_ERROR) { fprintf(stderr, "ERROR> bind: %d", WSAGetLastError()); exit(1); }
  
  ok = listen(fd,5);
  if (ok == SOCKET_ERROR) { fprintf(stderr, "ERROR> listen: %d", WSAGetLastError()); exit(1); }
  len = sizeof(addr);
  ok = getsockname(fd, (struct sockaddr *)&addr, &len);
  if (ok == SOCKET_ERROR) { fprintf(stderr, "ERROR> getsockname: %d", WSAGetLastError()); exit(1); }

  *pfd = fd;
  *pip = skt_ip();
  *ppo = ntohs(addr.sin_port);
}


void skt_accept(int src, unsigned int *pip, unsigned int *ppo, SOCKET *pfd)
{
  int i;
  SOCKET fd;
  struct sockaddr_in remote;
  
  i = sizeof(remote);
 acc:
  fd = accept(src, (struct sockaddr *)&remote, &i);
  if ((fd == INVALID_SOCKET)&&(WSAGetLastError()==WSAEINTR)) goto acc;
  if ((fd == INVALID_SOCKET)&&(WSAGetLastError()==WSAEMFILE)) goto acc;
  if (fd == INVALID_SOCKET) 
  {
    fprintf(stderr, "accept : %d", WSAGetLastError()); 
    exit(1);
  }
  
  *pip = htonl(remote.sin_addr.s_addr);
  *ppo = htons(remote.sin_port);
  *pfd = fd;
}



SOCKET skt_connect(unsigned int ip, unsigned int port)
{
  struct sockaddr_in remote; 
  short sport=port;
  int    ok;
  SOCKET fd;
    
  /* create an address structure for the server */
  memset(&remote, 0, sizeof(remote));
  remote.sin_family = AF_INET;
  remote.sin_port = htons(sport);
  remote.sin_addr.s_addr = htonl(ip);
    
  DEBUGF(("Connecting to %d, at port = %d\n", ip, sport));
 
sock:
  fd = socket(AF_INET, SOCK_STREAM, 0);
  if ((fd==INVALID_SOCKET)&&(WSAGetLastError()==WSAEMFILE)) goto sock;
  if (fd == INVALID_SOCKET) return fd;
  
  ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
  if (ok == SOCKET_ERROR) {
    switch (WSAGetLastError()) 
  {
    case WSAEADDRINUSE: 
      closesocket(fd); 
      goto sock;
    default: 
      return INVALID_SOCKET;
    }
  }
  return fd;
}

/****************************************************************************
 *
 * Miscellaneous minor routines.
 *
 ****************************************************************************/


void zap_newline(char *s)
{
  char *p;
  p = s + strlen(s)-1;
  if (*p == '\n') *p = '\0';
}


char *substr(char *lo, char *hi)
{
  int len = hi-lo;
  char *res = (char *)malloc(1+len);
  memcpy(res, lo, len);
  res[len]=0;
  return res;
}


int subeqs(char *lo, char *hi, char *str)
{
  int len = strlen(str);
  if (hi-lo != len) return 0;
  if (memcmp(lo, str, len)) return 0;
  return 1;
}

/* advance pointer over blank characters */
char *skipblanks(char *p)
{
  while ((*p==' ')||(*p=='\t')) p++;
  return p;
}

/* advance pointer over nonblank characters */

char *skipstuff(char *p)
{
  while ((*p)&&(*p!=' ')&&(*p!='\t')) p++;
  return p;
}



char *text_ip(unsigned int ip)
{
  static char buffer[100];
  sprintf(buffer,"%d.%d.%d.%d",
      (ip>>24)&0xFF, (ip>>16)&0xFF, (ip>>8)&0xFF, (ip>>0)&0xFF);
  
  return buffer;
}


int readhex(FILE *f, int len)
{
  char buffer[100];
  char *p;
  int res;
  
  if (fread(buffer, len, 1, f)!=1) return -1;
  buffer[len]=0;
  res=strtol(buffer, &p, 16);
  if (p!=buffer+len) return -1;
  return res;
}


static char *parseint(char *p, int *value)
{
  int val = 0;
  
  while (((*p)==' ')||((*p)=='.')) p++;
  
  if (((*p)<'0')||((*p)>'9')) 
  {
    fprintf(stderr,"badly-formed number");
    exit(1);
  }
  
  while ((*p>='0')&&(*p<='9')) { val*=10; val+=(*p)-'0'; p++; }
  *value = val;
  return p;
}


/* getenv_display is never used */
char *getenv_display()
{
  static char result[100];
  char *e, *p;
  
  e = getenv("DISPLAY");
  if (e==0) return 0;
  p = strrchr(e, ':');
  if (p==0) return 0;
  if ((e[0]==':')||(strncmp(e,"unix:",5)==0)) 
  {
    sprintf(result,"%s:%s",text_ip(skt_ip()),p+1);
  }  
  else strcpy(result, e);
  return result;
}

char *mylogin()
{  
  char *name;
  DWORD size = 1000;

  name = (char *) malloc(size);

  if (GetUserName(name, &size)==0) { perror("ERROR> GetUserName()"); exit(1); }
  return name;
} 

/* Given the name of a host, find it's IP address by DNS lookup, return a big endian address i.e. in network byte order*/
unsigned int lookup_ip(char *name)
{  
  struct hostent *h;  
  unsigned int ip1,ip2,ip3,ip4; 
  int nread;
  
  nread = sscanf(name,"%d.%d.%d.%d",&ip1,&ip2,&ip3,&ip4);
  if (nread==4) return (ip1<<24)|(ip2<<16)|(ip3<<8)|ip4;
  
  h = gethostbyname(name);
  if (h==0) return 0;
  return *((int *)(h->h_addr_list[0]));
}


void strsubst(char *str, char c1, char c2)
{
  while (*str) 
  {
    if (*str==c1) *str=c2;
    str++;
  }
}

/*****************************************************************************
 *                                                                           *
 * PPARAM - obtaining "program parameters" from the user.                    *
 *                                                                           *
 *****************************************************************************/

typedef struct ppdef {  
  union
    {
    double r;
    int i;
    char *s;
    int f;
    } value;
  char *lname;
  char *doc;
  char  type;
  struct ppdef *next;
} *ppdef;

static ppdef   ppdefs;

static int     pparam_pos;
static char  **pparam_argv;
static char    pparam_optc = '-';
char           pparam_error[100];

/* Finds a name in a list of struct ppdefs */
static ppdef pparam_find(char *lname)
{  
  ppdef def;
  
  for (def=ppdefs; def; def=def->next)
    if (strcmp(def->lname, lname)==0) break;
      
  return def;
}

static ppdef pparam_cell(char *lname)
{  
  ppdef def = pparam_find(lname);
  if (def) return def;
  
  def = (ppdef) malloc(sizeof(struct ppdef));
  def->lname = _strdup(lname);
  def->type  = 's';
  def->doc   = "(undocumented)";
  def->next  = ppdefs;
  ppdefs = def;
  return def;
}

void pparam_doc(char *lname, char *doc)
{
  ppdef def = pparam_cell(lname);
  def->doc = _strdup(doc);
}

void pparam_defint(char *lname, int value)
{
  ppdef def = pparam_cell(lname);
  def->type  = 'i';
  def->value.i = value;
}

void pparam_defreal(char *lname, double value)
{
  ppdef def = pparam_cell(lname);
  def->type  = 'r';
  def->value.r = value;
}

void pparam_defstr(char *lname, char *value)
{
  ppdef def = pparam_cell(lname);
  def->type  = 's';
  def->value.s = value;
}

void pparam_defflag(char *lname)
{
  ppdef def = pparam_cell(lname);
  def->type  = 'f';
  def->value.f = 0;
}

static ppdef pparam_hfind(char *lname)
{
  ppdef def = pparam_find(lname);
  if (def) return def;
  fprintf(stderr,"ERROR> No such program parameter %s\n",lname);
  exit(1);
}

int pparam_getint(char *lname)
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 'i') return 0;
  return def->value.i;
}

double pparam_getreal(char *lname)
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 'r') return 0.0;
  return def->value.r;
}

char *pparam_getstr(char *lname)
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 's') return 0;
  return def->value.s;
}

int pparam_getflag(char *lname)
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 'f') return 0;
  return def->value.f;
}

static int pparam_setdef(ppdef def, char *value)
{
  char *p;
  
  switch(def->type)
    {
    case 'i' :
      def->value.i = strtol(value, &p, 10);
      if (*p) return -1;
      return 0;
    case 'r' :
      def->value.r = strtod(value, &p);
      if (*p) return -1;
      return 0;
    case 's' :
      def->value.s = value;
      return 0;
    case 'f' :
      def->value.i = strtol(value, &p, 10);
      if (*p) return -1;
      return 0;
    default:
      return -1;
    }  
}

int pparam_set(char *lname, char *value)
{
  ppdef def = pparam_cell(lname);
  return pparam_setdef(def, value);
}

char *pparam_getdef(ppdef def)
{  
  static char result[100];
  
  switch(def->type)
    {
    case 'i': sprintf(result,"%d", def->value.i); return result;
    case 'r': sprintf(result,"%lf",def->value.r); return result;
    case 's': return def->value.s;
    case 'f': sprintf(result,"%d", def->value.f); return result;
    default : return NULL;
    }
}

void pparam_printdocs()
{  
  ppdef def; 
  int   i, len, maxname, maxdoc;
  
  maxname = 0;
  maxdoc = 0;
  for (def=ppdefs; def; def=def->next)
    {
    len = strlen(def->lname);
    if (len>maxname) maxname=len;
    len = strlen(def->doc);
    if (len>maxdoc) maxdoc=len;
    }
  
  fprintf(stderr,"\n");
  fprintf(stderr,"parameters recognized are:\n");
  fprintf(stderr,"\n");
  
  for (def=ppdefs; def; def=def->next)
    {
    len = strlen(def->lname);
    fprintf(stderr,"  %c%c%s ",pparam_optc,pparam_optc,def->lname);
    for(i=0; i<maxname-len; i++) fprintf(stderr," ");
    len = strlen(def->doc);
    fprintf(stderr,"  %s ",def->doc);
    for(i=0; i<maxdoc-len; i++) fprintf(stderr," ");
    fprintf(stderr,"[%s]\n",pparam_getdef(def));
    }
  fprintf(stderr,"\n");
}

void pparam_delarg(int i)
{
  int j;
  for (j=i; pparam_argv[j]; j++)
    pparam_argv[j]=pparam_argv[j+1];
}

int pparam_countargs(char **argv)
{
  int argc;
  for (argc=0; argv[argc]; argc++);
  return argc;
}

int pparam_parseopt()
{  
  int   ok; 
  ppdef def;
  
  char *opt = pparam_argv[pparam_pos];
  
  /* handle ++ by skipping to end */
  if ((opt[1]=='+')&&(opt[2]=='\0'))
    {
    pparam_delarg(pparam_pos);
    while (pparam_argv[pparam_pos]) pparam_pos++;
    return 0;
    }
  
  /* handle + by itself - an error */
  if (opt[1]==0) 
  {
    sprintf(pparam_error,"Illegal option +\n");
    return -1;
    }
  
  /* look up option definition */
  if (opt[1]=='+') def = pparam_find(opt+2);
  else
    {
    char name[2];
    name[0]=opt[1];
    name[1]='\0';
    def = pparam_find(name);
    }
  
  if (def==0)
    {
      pparam_pos++;
      return 0;
    }
  
  /* handle flag-options */
  if ((def->type=='f')&&(opt[1]!='+')&&(opt[2]))
    {
    sprintf(pparam_error,"Option %s should not include a value",opt);
    return -1;
    }
  
  if (def->type=='f')
    {
    def->value.f = 1;
    pparam_delarg(pparam_pos);
    return 0;
    }
  
  /* handle non-flag options */
  if ((opt[1]=='+')||(opt[2]=='\0'))
    {
    pparam_delarg(pparam_pos);
    opt = pparam_argv[pparam_pos];
    }
  
  else opt+=2;
  
  if ((opt == NULL)||(opt[0] == '\0'))
    {
    sprintf(pparam_error,"%s must be followed by a value.",opt);
    return -1;
    }
  
  ok = pparam_setdef(def, opt);
  pparam_delarg(pparam_pos);
  
  if (ok<0)
    {
    sprintf(pparam_error,"Illegal value for %s",opt);
    return -1;
    }
  
  return 0;
}


int pparam_parsecmd(char optchr, char **argv)
{  
  pparam_error[0]='\0';
  pparam_argv = argv;
  pparam_optc = optchr;
  pparam_pos  = 0;
  
  while(1)
    {
    char *opt = pparam_argv[pparam_pos];
    if (opt==NULL) break;
    if (opt[0]!=optchr) pparam_pos++;
    else if (pparam_parseopt()<0) return -1;
    }
  
  return 0;
}

/****************************************************************************
 *                                                                           
 * xstr                                                                      
 *                                                                           
 *  extendable (and otherwise dynamically-changing) strings.                 
 *                                                                           
 *  These are handy for implementing character buffers of all types.         
 *                                                                           
 *  This module tries to guarantee reasonable speed efficiency for all       
 *  operations.                                                              
 *                                                                           
 *  Each xstr takes around 3*len bytes (where 'len' is the length of the     
 *  string being stored), so it isn't very space efficient.  This is done    
 *  to improve the efficiency of updates.                                    
 *                                                                           
 *  xstr_alloc()                                                             
 *                                                                           
 *      - allocates an empty buffer.                                         
 *                                                                           
 *  xstr_free(str)                                                           
 *                                                                           
 *      - frees an allocated buffer.                                         
 *                                                                           
 *  xstr_lptr(s)                                                             
 *                                                                           
 *      - returns a pointer to leftmost char in buffer.                      
 *                                                                           
 *  xstr_rptr(s)                                                             
 *                                                                           
 *      - returns a pointer beyond rightmost char in buffer.                 
 *                                                                           
 *  xstr_rexpand(str, nbytes)                                                
 *                                                                           
 *     - add uninitialized bytes to the right end of the string.             
 *                                                                           
 *  xstr_lexpand(str, nbytes)                                                
 *                                                                           
 *     - add uninitialized bytes to the left end of the string.              
 *                                                                           
 *  xstr_rshrink(str, nbytes)                                                
 *                                                                           
 *     - remove bytes from the right end of the string.                      
 *                                                                           
 *  xstr_lshrink(str, nbytes)                                                
 *                                                                           
 *     - remove bytes from the left end of the string.                       
 *                                                                           
 *  xstr_read(str, fd)
 *
 *     - read bytes from the specified FD into the right end of xstr.
 *
 *  xstr_write(str, bytes, nbytes)                                           
 *                                                                           
 *     - append the specified bytes to the right end of the xstr.            
 *                                                                           
 *  xstr_printf(str, ...)                                                    
 *                                                                           
 *     - print the specified message onto the end of the xstr.               
 *                                                                           
 *****************************************************************************/

typedef struct xstr {
    char *lptr;
    char *rptr;
    char *lend;
    char *rend;
} *xstr;

char *xstr_lptr(xstr l) { return l->lptr; }
char *xstr_rptr(xstr l) { return l->rptr; }

int xstr_len(xstr l)
{
  return l->rptr - l->lptr;
}

xstr xstr_alloc()
{
  xstr res = (xstr)malloc(sizeof(struct xstr));
  res->lend = (char *)malloc(257);
  res->lptr = res->lend + 128;
  res->rptr = res->lend + 128;
  res->rend = res->lend + 256;
  *(res->rptr) = 0;
  return res;
}

void xstr_free(xstr s)
{
  free(s->lend);
  free(s);
}

void xstr_rexpand(xstr l, int nbytes)
{
  int  uspace, needed; 
  char *nbuf;
  
  if (l->rend - l->rptr>=nbytes) { l->rptr += nbytes; return; }
  
  uspace = (l->rptr - l->lptr);
  needed = uspace + nbytes;
  if (needed<64) needed=64;
  nbuf = (char *)malloc(1+(needed*3));
  memcpy(nbuf+needed, l->lptr, uspace);
  free(l->lend);
  l->lend = nbuf;
  l->lptr = nbuf + needed;
  l->rptr = nbuf + needed + uspace + nbytes;
  l->rend = nbuf + needed + needed + needed;
  *(l->rptr) = 0;
}

void xstr_lexpand(xstr l, int nbytes)
{
  int  uspace, needed; 
  char *nbuf;
  
  if (l->rend - l->rptr>=nbytes) { l->rptr += nbytes; return; }
  uspace = (l->rptr - l->lptr);
  needed = uspace + nbytes;
  if (needed<64) needed=64;
  nbuf = (char *)malloc(1+(needed*3));
  memcpy(nbuf+needed+nbytes, l->lptr, uspace);
  free(l->lend);
  l->lend = nbuf;
  l->lptr = nbuf + needed;
  l->rptr = nbuf + needed + uspace + nbytes;
  l->rend = nbuf + needed + needed + needed;
  *(l->rptr) = 0;
}

void xstr_rshrink(xstr l, int nbytes)
{
  if (l->rptr - l->lptr < nbytes) { l->rptr=l->lptr; return; }
  l->rptr -= nbytes;
  *(l->rptr) = 0;
}

void xstr_lshrink(xstr l, int nbytes)
{
  if (l->rptr - l->lptr < nbytes) { l->lptr=l->rptr; return; }
  l->lptr += nbytes;
}

void xstr_write(xstr l, char *bytes, int nbytes)
{
  xstr_rexpand(l, nbytes);
  memcpy(xstr_lptr(l)+xstr_len(l)-nbytes, bytes, nbytes);
}


int xstr_read(xstr l, HANDLE fd)
{
  int nread, done;
  xstr_rexpand(l, 1024);
  done = ReadFile(fd, xstr_rptr(l)-1024, 1024, &nread, NULL);
  if (done == 0) 
  {
    xstr_rshrink(l, 1024);
    nread = -1;
  }
  else xstr_rshrink(l, 1024-nread);
  return nread;
}


int xstr_readsock(xstr l, SOCKET fd)
{
  int nread;
  xstr_rexpand(l, 1024);
  nread = recv(fd, xstr_rptr(l)-1024, 1024, 0);
  if (nread == SOCKET_ERROR) 
  {
    xstr_rshrink(l, 1024);
    nread = -1;
  }
  else xstr_rshrink(l, 1024-nread);
  return nread;
}


void xstr_printf(xstr buf, ...)
{
  char buffer[10000];
  char *fmt;
  va_list ap;

  va_start(ap, buf);
  fmt = va_arg(ap, char *);
  vsprintf(buffer, fmt, ap);
  xstr_write(buf, buffer, strlen(buffer));
  va_end(ap);
}


char *xstr_gets(char *buff, int size, xstr s)
{
  char *p; 
  int len;
  xstr_rptr(s)[0]=0;
  p = strchr(xstr_lptr(s),'\n');
  if (p==0) return 0;
  *p = '\0';
  len = p - xstr_lptr(s);
  if (len > size) len=size;
  memcpy(buff, xstr_lptr(s), len);
  buff[len] = 0;
  xstr_lshrink(s, len+1);
  return buff;
}

/****************************************************************************
 * 
 * ARG
 *
 * The following module computes a whole bunch of miscellaneous values, which
 * are all constant throughout the program.  Naturally, this includes the
 * value of the command-line arguments.
 *
 *****************************************************************************/


#define MAX_NODES       1000
#define MAX_LINE_LENGTH 1000

char **arg_argv;
int    arg_argc;

int   arg_requested_pes;
int   arg_timeout;
int   arg_verbose;
int   arg_debug;
int   arg_debug_no_pause;
int   arg_in_xterm;
int   arg_maxrsh;
char *arg_nodelist;
char *arg_nodegroup;
char *arg_display;
char *arg_nodeprog_a;
char *arg_nodeprog_r;
char *arg_rshprog;
char *arg_currdir_a;
char *arg_currdir_r;
char *arg_mylogin;
char *arg_myhome;
char *arg_shell;

#if CMK_CCS_AVAILABLE
int   arg_server;
int   arg_server_port;
#endif

#if CMK_DEBUG_MODE
int   arg_gdbinterface;
int   arg_initial_bp;
#endif

void arg_init(int argc, char **argv)
{  
  static char buf[1024]; 
  
  pparam_defint ("p"             ,  MAX_NODES);
  pparam_defint ("timeout"       ,  2);
  pparam_defflag("verbose"           );
  
  // Don't know yet if we need the debug flags
  //pparam_defflag("debug"             );
  //pparam_defflag("debug-no-pause"    );

/* Got to do something about CCS doesn't run yet */

#if CMK_CCS_AVAILABLE
  pparam_defflag("server"            );
  pparam_defint ("server-port",    0 );
#endif

/*
#if CMK_DEBUG_MODE
  pparam_defflag("gdbinterface"      );
  pparam_defflag("initial_bp"        );
#endif
*/
  // No xterms or rshs being used decide what to do with these
  //pparam_defflag("in-xterm"          );
  //pparam_defint ("maxrsh"        ,  16);
  
  pparam_defstr ("nodelist"      ,  NULL);
  pparam_defstr ("nodegroup"     ,  "main");
  //pparam_defstr ("remote-shell"  ,  NULL);
  
  pparam_doc("p",             "number of processes to create");
  pparam_doc("timeout",       "seconds to wait per host connection");
  //pparam_doc("in-xterm",      "Run each node in an xterm window");
  pparam_doc("verbose",       "Print diagnostic messages");
  //pparam_doc("debug",         "Run each node under gdb in an xterm window");
  //pparam_doc("debug-no-pause","Like debug, except doesn't pause at beginning");

#if CMK_CCS_AVAILABLE
  pparam_doc("server",        "Enable client-server mode");
  pparam_doc("server-port",   "Port to listen to for CCS requests");
#endif

/*
#if CMK_DEBUG_MODE
  pparam_doc("gdbinterface",  "Allow the gdb interface to be integrated");
  pparam_doc("initial_bp"  ,  "Allow the program to break at the initial CsdScheduler call");
#endif
*/
  //pparam_doc("maxrsh",        "Maximum number of rsh's to run at a time");
  pparam_doc("nodelist",      "file containing list of nodes");
  pparam_doc("nodegroup",     "which group of nodes to use");
  //pparam_doc("remote-shell",  "which remote shell to use");

  if (pparam_parsecmd('+', argv) < 0) 
  {
    fprintf(stderr,"ERROR> syntax: %s\n", pparam_error);
    pparam_printdocs();
    exit(1);
  }
  
  arg_argv = argv+2;
  arg_argc = pparam_countargs(argv+2);
  
  arg_requested_pes  = pparam_getint("p");
  arg_timeout        = pparam_getint("timeout");
  //arg_in_xterm       = pparam_getflag("in-xterm");
  arg_verbose        = pparam_getflag("verbose");
  //arg_debug          = pparam_getflag("debug");
  //arg_debug_no_pause = pparam_getflag("debug-no-pause");

#if CMK_CCS_AVAILABLE
  arg_server         = pparam_getflag("server");
  arg_server_port    = pparam_getint("server-port");
#endif

/*
#if CMK_DEBUG_MODE
  arg_gdbinterface   = pparam_getflag("gdbinterface");
  arg_initial_bp     = pparam_getflag("initial_bp");
#endif
*/
  //arg_maxrsh         = pparam_getint("maxrsh");
  arg_nodelist       = pparam_getstr("nodelist");
  arg_nodegroup      = pparam_getstr("nodegroup");
  //arg_shell          = pparam_getstr("remote-shell");

  //arg_verbose = arg_verbose || arg_debug || arg_debug_no_pause;
  
  /* Find the current value of the CONV_RSH variable */
  //if(!arg_shell)
    //arg_shell = getenv_rsh();

  /* Find the current value of the DISPLAY variable, there is no DISPLAY variable to speak of */  
  /*arg_display = getenv_display();
  if ((arg_debug || arg_debug_no_pause || arg_in_xterm) && (arg_display==0)) {
    fprintf(stderr,"ERROR> DISPLAY must be set to use debugging mode\n");
    exit(1);
  }
*/
  
/* conv-host assumes that the node program lies at the same location as itself on each node, 
   need to find a way to start up the node program at different locations on different nodes */

  /* find the current directory, absolute version */
  _getcwd(buf, 1023);
  arg_currdir_a = _strdup(buf);
  
  /* find the node-program, absolute version */
  if (argc<2) 
  {
    fprintf(stderr,"ERROR> You must specify a node-program.\n");
    exit(1);
  }
  
  if (argv[1][1]==':') 
  {
    arg_nodeprog_a = argv[1];
  } 
  else 
  {
    sprintf(buf,"%s\\%s",arg_currdir_a, argv[1]);
    arg_nodeprog_a = strdup(buf);
  }

  arg_mylogin = mylogin();
  
  /* No concept of a home directory really, at least I don't know of one */
  //arg_myhome = (char *)malloc(MAXPATHLEN);
  //strcpy(arg_myhome, getenv("HOME"));
  arg_shell = NULL;
}

/****************************************************************************
 *                                                                           
 * NODETAB:  The nodes file and nodes table.
 *
 ****************************************************************************/

char *nodetab_file_find()
{
  /* Find a nodes-file as specified by ++nodelist */
  if (arg_nodelist) 
  {
    char *path = arg_nodelist;
    if (probefile(path)) return _strdup(path);
    fprintf(stderr,"ERROR> No such nodelist file %s\n",path);
    exit(1);
  }
  
  /* Find a nodes-file as specified by getenv("NODELIST") */
  if (getenv("NODELIST")) 
  {
    char *path = getenv("NODELIST");        
    if (path && probefile(path)) return _strdup(path);
    fprintf(stderr,"ERROR> Cannot find nodelist file %s\n",path);
    exit(1);
  }
  
  /* Find a nodes-file by looking under 'nodelist' in the current directory */
  if (probefile(".\\nodelist")) return _strdup(".\\nodelist");
  
  /*
  if (getenv("HOME")) 
  {
    char buffer[MAXPATHLEN];
    sprintf(buffer,"%s/.nodelist", getenv("HOME"));
    if (probefile(buffer)) return _strdup(buffer);
  }
  */
  
  fprintf(stderr,"ERROR> Cannot find a nodes file.\n");
  exit(1);
}



/* Structure encapsulating information about each host, we keep login, passwd, shell information but may never actually 
   use it, will reach a decision on this later */

typedef struct nodetab_host {
  
  char         *name;
  char         *login;
  char         *passwd;
  pathfixlist  pathfixes;
  char         *ext;
  char         *setup;
  int          cpus;
  int          rank;
  double       speed;
  unsigned int ip;
  char         *shell;
} *nodetab_host;

nodetab_host *nodetab_table;
int           nodetab_max;
int           nodetab_size;
int          *nodetab_rank0_port;
int          *nodetab_rank0_table;
int           nodetab_rank0_size;

char         *default_login;
char         *default_group;
char         *default_passwd;
pathfixlist  default_pathfixes;
char         *default_ext;
char         *default_setup;
double       default_speed;
int          default_cpus;
int          default_rank;
char         *default_shell;

char         *host_login;
char         *host_group;
char         *host_passwd;
pathfixlist  host_pathfixes;
char         *host_ext;
char         *host_setup;
double       host_speed;
int          host_cpus;
int          host_rank;
char         *host_shell;

void nodetab_reset()
{  
  default_login     = "*";
  default_group     = "*";
  default_passwd    = "*";
  default_pathfixes = NULL;
  default_ext       = "*";
  default_setup     = "*";
  default_speed     = 1.0;
  default_cpus      = 1;
  default_rank      = 0;
  default_shell     = arg_shell;
}

void host_reset()
{
  host_login = default_login;
  host_group = default_group;
  host_passwd = default_passwd;
  host_pathfixes = default_pathfixes;
  host_ext = default_ext;
  host_setup = default_setup;
  host_speed = default_speed;
  host_cpus = default_cpus;
  host_rank = default_rank;
  host_shell = default_shell;
}

void nodetab_add(nodetab_host res)
{  
  if (res->rank == 0)
    nodetab_rank0_table[nodetab_rank0_size++] = nodetab_size;
  nodetab_table[nodetab_size] = 
    (struct nodetab_host *) malloc(sizeof(struct nodetab_host));
  memcpy(nodetab_table[nodetab_size++], res, sizeof(struct nodetab_host));
}

void nodetab_makehost(char *host)
{
  nodetab_host res;
  int ip;
  
  ip = lookup_ip(host);
  
  if (ip==0) 
  {
    fprintf(stderr,"ERROR> Cannot obtain IP address of %s\n", host);
    exit(1);
  }
  
  if (nodetab_size == nodetab_max) return;
  res = (nodetab_host)malloc(sizeof(struct nodetab_host));
  res->name =   host;
  res->login =  host_login;
  res->passwd = host_passwd;
  res->pathfixes = host_pathfixes;
  res->ext = host_ext;
  res->setup = host_setup;
  res->speed = host_speed;
  res->rank = host_rank;
  res->cpus = host_cpus;
  res->ip = ip;
  res->shell = host_shell;
  nodetab_add(res);
}

void setup_host_args(char *args)
{
  while(*args != 0) 
  {
    char *b1 = args, *e1 = skipstuff(b1);
    char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
    
    args = skipblanks(e2);
    if (subeqs(b1,e1,"++login")) host_login = substr(b2,e2);
    else if (subeqs(b1,e1,"++passwd")) host_passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"++shell")) host_shell = substr(b2,e2);
    else if (subeqs(b1,e1,"++speed")) host_speed = atof(b2);
    else if (subeqs(b1,e1,"++cpus")) host_cpus = atol(b2);
    else if (subeqs(b1,e1,"++pathfix")) 
    {    
      char *b3 = skipblanks(e2), *e3 = skipstuff(b3);
      args = skipblanks(e3);
      host_pathfixes = pathfix_append(substr(b2,e2),substr(b3,e3),host_pathfixes);
    } 
    else if (subeqs(b1,e1,"++ext")) host_ext = substr(b2,e2);
    else if (subeqs(b1,e1,"++setup")) host_setup = strdup(b2);
  }
}

void setup_group_args(char *args)
{  
  while(*args != 0) 
  {
    char *b1 = args, *e1 = skipstuff(b1);
    char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
    args = skipblanks(e2);
    if (subeqs(b1,e1,"++login")) default_login = substr(b2,e2);
    else if (subeqs(b1,e1,"++passwd")) default_passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"++shell")) default_shell = substr(b2,e2);
    else if (subeqs(b1,e1,"++speed")) default_speed = atof(b2);
    else if (subeqs(b1,e1,"++cpus")) default_cpus = atol(b2);
    else if (subeqs(b1,e1,"++pathfix")) 
    {
      char *b3 = skipblanks(e2), *e3 = skipstuff(b3);
      args = skipblanks(e3);
      default_pathfixes = pathfix_append(substr(b2,e2),substr(b3,e3),
                                       default_pathfixes);
    } 
    else if (subeqs(b1,e1,"++ext")) default_ext = substr(b2,e2);
    else if (subeqs(b1,e1,"++setup")) default_setup = strdup(b2);
  }
}


/* Read the 'nodelist' file, parse it and set up the node table appropriately */
void nodetab_init()
{  
  FILE *f,*fopen();
  char *nodesfile; //nodetab_host node;
  char input_line[MAX_LINE_LENGTH];
  int  rightgroup, basicsize, i, remain;
  char *b1, *e1, *b2, *e2, *b3, *e3;
  
  /* Open the NODES_FILE. */
  nodesfile = nodetab_file_find();
  if(arg_verbose)
    fprintf(stderr, "INFO> using %s as nodesfile\n", nodesfile);
  if (!(f = fopen(nodesfile,"r"))) 
  {
    fprintf(stderr,"ERROR> Cannot read %s: %s\n",nodesfile,strerror(errno));
    exit(1);
  }
  
  nodetab_table = (nodetab_host*)malloc(arg_requested_pes*sizeof(nodetab_host));
  nodetab_rank0_table = (int*)malloc(arg_requested_pes*sizeof(int));
  nodetab_rank0_port = (int*)malloc(arg_requested_pes*sizeof(int));
  nodetab_max = arg_requested_pes;
  
  nodetab_reset();
  rightgroup = (strcmp(arg_nodegroup,"main")==0);
  
  while(fgets(input_line,sizeof(input_line)-1,f)!=0) 
  {
    if (nodetab_size == arg_requested_pes) break;
    if (input_line[0]=='#') continue;
    zap_newline(input_line);
    b1 = skipblanks(input_line);
    e1 = skipstuff(b1); b2 = skipblanks(e1); 
    e2 = skipstuff(b2); b3 = skipblanks(e2);
    e3 = skipstuff(b3);
    if (*b1==0) continue;
    if (strcmp(default_login, "*")==0) default_login = arg_mylogin;
    if (strcmp(default_ext, "*")==0) default_ext = "";
    if  (subeqs(b1,e1,"login")&&(*b3==0))    default_login = substr(b2,e2);
    else if (subeqs(b1,e1,"passwd")&&(*b3==0))   default_passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"shell")&&(*b3==0))   default_shell = substr(b2,e2);
    else if (subeqs(b1,e1,"speed")&&(*b3==0))    default_speed = atof(b2);
    else if (subeqs(b1,e1,"cpus")&&(*b3==0))     default_cpus = atol(b2);
    else if (subeqs(b1,e1,"pathfix")) 
    default_pathfixes=pathfix_append(substr(b2,e2),substr(b3,e3),
                                       default_pathfixes);
    else if (subeqs(b1,e1,"ext")&&(*b3==0))      default_ext = substr(b2,e2);
    else if (subeqs(b1,e1,"setup"))              default_setup = strdup(b2);
    else if (subeqs(b1,e1,"host")) 
    {
      if (rightgroup) 
      {
        host_reset();
        setup_host_args(b3);
        for (host_rank=0; host_rank<host_cpus; host_rank++)
          nodetab_makehost(substr(b2,e2));
      }
    } 
    else if (subeqs(b1,e1, "group")) 
    {
      nodetab_reset();
      rightgroup = subeqs(b2,e2,arg_nodegroup);
      if(rightgroup) setup_group_args(b3);
    } 
    else 
    {
      fprintf(stderr,"ERROR> unrecognized command in nodesfile:\n");
      fprintf(stderr,"ERROR> %s\n", input_line);
      exit(1);
    }
  }
  
  basicsize = nodetab_size;
  if (basicsize==0) 
  {
    fprintf(stderr,"ERROR> No hosts in group %s\n", arg_nodegroup);
    exit(1);
  }
  
  while ((nodetab_size < arg_requested_pes)&&(arg_requested_pes!=MAX_NODES)) 
  {
    nodetab_host h = nodetab_table[nodetab_size-basicsize];
    nodetab_add(h);
  }
  
  for (i=0; i<nodetab_size; i++) 
  {
    if (nodetab_table[i]->rank == 0)
      remain = nodetab_size - i;
    if (nodetab_table[i]->cpus > remain)
      nodetab_table[i]->cpus = remain;
  }
  fclose(f);
}


nodetab_host nodetab_getinfo(int i)
{
  if (nodetab_table==0) 
  {
    fprintf(stderr,"ERROR> Node table not initialized.\n");
    exit(1);
  }
  if ((i<0)||(i>=nodetab_size)) 
  {
    fprintf(stderr,"ERROR> No such node %d\n",i);
    exit(1);
  }
  return nodetab_table[i];
}


char        *nodetab_name(int i)      { return nodetab_getinfo(i)->name; }
char        *nodetab_login(int i)     { return nodetab_getinfo(i)->login; }
char        *nodetab_passwd(int i)    { return nodetab_getinfo(i)->passwd; }
char        *nodetab_setup(int i)     { return nodetab_getinfo(i)->setup; }
pathfixlist  nodetab_pathfixes(int i) { return nodetab_getinfo(i)->pathfixes; }
char        *nodetab_ext(int i)       { return nodetab_getinfo(i)->ext; }
char        *nodetab_shell(int i)     { return nodetab_getinfo(i)->shell; }
unsigned int nodetab_ip(int i)        { return nodetab_getinfo(i)->ip; }
unsigned int nodetab_cpus(int i)      { return nodetab_getinfo(i)->cpus; }
unsigned int nodetab_rank(int i)      { return nodetab_getinfo(i)->rank; }
 

/****************************************************************************
 *
 * ARVAR - array variables
 *
 * This host can store arrays for its clients.
 *
 * arvar_set(char *var, int index, char *val)
 *
 *  - set the specified position in the specified array to the specified
 *    value.  If the named array does not exist, it is created.  If the
 *    specified index does not exist, it is created.
 *
 * arvar_get(char *var, int lo, int hi)
 *
 *  - retrieve the specified subrange of the array.  If the entire subrange
 *    has not been assigned, NULL is returned. Values are separated by
 *    spaces.
 *
 ****************************************************************************/

typedef struct arvar {
  char *name;
  int lo;
  int hi;
  int tot;
  char **data;
  struct arvar *next;
} *arvar;

arvar arvar_list;

arvar arvar_find(char *var)
{
  arvar v = arvar_list;
  while (v && (strcmp(v->name, var))) v=v->next;
  return v;
}


arvar arvar_obtain(char *var)
{
  arvar v = arvar_find(var);
  if (v==0) 
  { 
    v = (arvar)malloc(sizeof(struct arvar));
    v->name = strdup(var);
    v->lo = 0;
    v->hi = -1;
    v->data = 0;
    v->next = arvar_list;
    arvar_list = v;
  }
  return v;
}


void arvar_set(char *var, int index, char *val)
{
  char *old;
  int  lo, hi;
  char **data;
  
  arvar v = arvar_obtain(var);
  data = v->data; lo = v->lo; hi = v->hi;
  if ((index<lo)||(index>=hi)) 
  {
    if (lo>hi) lo=hi=index;
    if (index<lo) lo=index;
    if (index>=hi) hi=index+1;
    data = (char **)calloc(1, (hi-lo)*sizeof(char *));
    if (v->data) 
    {
      memcpy(data+((v->lo)-lo), v->data, (v->hi - v->lo)*sizeof(char *));
      free(v->data);
    }
    v->data = data; v->lo = lo; v->hi = hi;
  }
  old = v->data[index-(v->lo)];
  if (old) free(old);
  else v->tot++;
  v->data[index-(v->lo)] = strdup(val);
}


char *arvar_get(char *var, int lo, int hi)
{
  int len=0; 
  int i;
  arvar v = arvar_find(var);
  char **data, *res;
  
  if (v==0) return 0;
  data = v->data;
  if (lo<v->lo) return 0;
  if (hi>v->hi) return 0;
  for (i=lo; i<hi; i++) 
  {
      char *val = data[i-(v->lo)];
    if (val==0) return 0;
    len += strlen(val)+1;
  }
  res = (char *)malloc(len);
  len=0;
  for (i=lo; i<hi; i++) 
  {
    char *val = data[i-(v->lo)];
    strcpy(res+len, val);
    len+=strlen(val);
    res[len++] = ' ';
  }
  res[--len] = 0;
  return res;
}

/****************************************************************************
 *
 * input handling
 *
 * You can use this module to read the standard input.  It supports
 * one odd function, input_scanf_chars, which is what makes it useful.
 * if you use this module, you may not read stdin yourself.
 *
 * void input_init(void)
 * char *input_gets(void)
 * char *input_scanf_chars(char *fmt)
 *
 ****************************************************************************/

char *input_buffer;

void input_extend()
{
  char line[1024];
  int len = input_buffer?strlen(input_buffer):0;
  fflush(stdout);
  if (fgets(line, 1023, stdin)==0) 
  { 
      fprintf(stderr,"end-of-file on stdin");
    exit(1);
  }
  input_buffer = realloc(input_buffer, len + strlen(line) + 1);
  strcpy(input_buffer+len, line);
}

void input_init()
{
  input_buffer = strdup("");
}

char *input_extract(int nchars)
{
  char *res = substr(input_buffer, input_buffer+nchars);
  char *tmp = substr(input_buffer+nchars, input_buffer+strlen(input_buffer));
  free(input_buffer);
  input_buffer = tmp;
  return res;
}

char *input_gets()
{
  char *p, *res; 
  int  len;
  while(1) 
  {
    p = strchr(input_buffer,'\n');
    if (p) break;
    input_extend();
  }
  len = p-input_buffer;
  res = input_extract(len+1);
  res[len]=0;
  return res;
}

char *input_scanf_chars(char *fmt)
{
  
  char buf[8192]; int len, pos;
  static int fd; static FILE *file;
  fflush(stdout);
  
  if (file==0) 
  {
    _unlink("\\tmp\\fnord");
    fd = _open("\\tmp\\fnord", _O_RDWR | _O_CREAT | _O_TRUNC);
    if (fd<0) 
    { 
      fprintf(stderr,"cannot open temp file \\tmp\\fnord");
      exit(1);
    }
    
    file = _fdopen(fd, "r+");
    unlink("\\tmp\\fnord");
  }
  
  while (1) 
  {
    len = strlen(input_buffer);
    rewind(file);
    fwrite(input_buffer, len, 1, file);
    fflush(file);
    rewind(file);
    _chsize(fd, len);
    fscanf(file, fmt, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf);
    pos = ftell(file);
    if (pos<len) break;
    input_extend();
  }
  return input_extract(pos);
}


/****************************************************************************
 *
 * REQUEST SERVICER
 *
 * The request servicer accepts connections on a TCP port.  The client
 * sends a sequence of commands (each is one line).  It then closes the
 * connection.  The server must then contact the client, sending replies.
 *
 ****************************************************************************/

#define REQ_MAXFD 10 /* max file descriptors per process */

typedef struct req_node {
  struct req_node *next;
  char request[1];
} *req_node;

typedef struct req_pipe
{
  HANDLE fd[2];
  xstr buffer;
  HANDLE tHandle;
  DWORD  threadID;
  unsigned int startnode, endnode;
} *req_pipe;

unsigned int    req_host_IP;
struct req_pipe req_pipes[REQ_MAXFD];
int             req_numworkers;
int             req_workers_registered;
req_node        req_saved;
int             req_ending=0;


#define REQ_OK 0
#define REQ_POSTPONE -1
#define REQ_FAILED -2
#define REQ_OK_AWAKEN -3



int req_handle_aset(char *line)
{
  char cmd[100], var[100], val[1000]; 
  int index, ok;
  
  ok = sscanf(line,"%s %s %d %s",cmd,var,&index,val);
  if (ok!=4) return REQ_FAILED;
  
  arvar_set(var,index,val);
  return REQ_OK_AWAKEN;
}



void req_write_to_client(int fd, char *buf, int size)
{
  int ok;
  while (size) 
  {
    ok = write(fd, buf, size);
    if (ok<=0) 
    {
      fprintf(stderr,"failed to send data to client");
      exit(1);
    }
    size-=ok; buf+=ok;
  }
}


/* Only place where request handler uses sockets alter suitably */
int req_reply(int ip, int port, char *pre, char *ans)
{
  char buf[1000];
  int len;
  int fd = skt_connect(ip, port);
  if (fd == INVALID_SOCKET) return REQ_FAILED;

  /*Prepare the message buffer*/
  sprintf(buf,"%s %s",pre,ans);
  if (strchr(buf,'\n')) *strchr(buf,'\n')=0;/*Replace newline with terminating NULL*/
  len=strlen(buf)+1;/*Send string (plus terminating NULL)*/

  send(fd,(char *)&len,sizeof(int),0);/*Send string length*/
  send(fd,buf,len,0);
  closesocket(fd);
  return REQ_OK;
}


int req_handle_aget(char *line)
{
  
  char cmd[100], host[1000], pre[1000], var[100]; 
  int ip, port, lo, hi, ok;
  char *res;
  
  ok = sscanf(line,"%s %s %d %s %d %d",cmd,host,&port,var,&lo,&hi);
  DEBUGF(("%s, %s, %d, %s, %d, %d\n", cmd, host, port, var, lo, hi));
  DEBUGF(("Handling aget\n"));
  if (ok!=6) return REQ_FAILED;
  ip = lookup_ip(host);
  if (ip==0) return REQ_FAILED;
  res = arvar_get(var,lo,hi+1);
  if (res==0) return REQ_POSTPONE;
  sprintf(pre, "aval %s %d %d", var, lo, hi);
  ok = req_reply(ip, port, pre, res);
  free(res);
  DEBUGF(("aget, handled -> %s\n", pre));
  return ok;
}


int req_handle_print(char *line)
{
  printf("%s",line+6);
  fflush(stdout);
  return REQ_OK;
}


int req_handle_printerr(char *line)
{
  fprintf(stderr,"%s",line+9);
  fflush(stderr);
  return REQ_OK;
}


int req_handle_ending(char *line)
{  
  req_ending++;
  //fprintf(stderr, "In req_handle_ending\n");
  
  if (req_ending == nodetab_size)
  {

#if CMK_WEB_MODE
    printf("End of program\n");
    if(clientIP != 0)
    {
      fd = skt_connect(clientIP, clientKillPort);
      if (fd != INVALID_SOCKET) 
        SendSocketN(fd, "die\n", strlen("die\n"));
    }
#endif
    
    exit(0);
  }
  return REQ_OK;
}


int req_handle_abort(char *line)
{
  fprintf(stderr, "%s", line+6);
  exit(1);
}

int req_handle_scanf(char *line)
{
  char cmd[100], host[100]; 
  int ip, port, ok;
  char *fmt, *res, *p;
  
  ok = sscanf(line,"%s %s %d",cmd,host,&port);
  if (ok!=3) return REQ_FAILED;
  ip = lookup_ip(host);
  if (ip==0) return REQ_FAILED;
  fmt = line;
  fmt = skipblanks(fmt); fmt = skipstuff(fmt);
  fmt = skipblanks(fmt); fmt = skipstuff(fmt);
  fmt = skipblanks(fmt); fmt = skipstuff(fmt);
  fmt = fmt+1;
  res = input_scanf_chars(fmt);
  p = res; while (*p) { if (*p=='\n') *p=' '; p++; }
  req_reply(ip, port, "scanf-data ", res);
  free(res);
  return REQ_OK;
}



/* Don't use this it doesn't work */
int req_handle_getinfo(char *line)
{  
  char reply[1024], ans[1024], pre[1024];
  char cmd[100], *res, *origres;
  int port, nread, i;
  int ip;

  nread = sscanf(line, "%s%d%d", cmd, &ip, &port);

  if(nread != 3) return REQ_FAILED;
  origres = res = arvar_get("addr", 0, nodetab_rank0_size);
  if(res==0) return REQ_POSTPONE;
  strcpy(pre, "info");
  reply[0] = 0;
  sprintf(ans, "%d ", nodetab_rank0_size);
  strcat(reply, ans);
  for(i=0;i<nodetab_rank0_size;i++) 
  {
    sprintf(ans, "%d ", nodetab_cpus(i));
    strcat(reply, ans);
  }
  for(i=0;i<nodetab_rank0_size;i++) 
  {
    sprintf(ans, "%d ", nodetab_ip(i));
    strcat(reply, ans);
  }
  for(i=0;i<nodetab_rank0_size;i++) 
  {
    unsigned int ip0,ip1,ip2,ip3,cport,dport,nodestart,nodesize;
    res = parseint(res,&ip0);
    res = parseint(res,&ip1);
    res = parseint(res,&ip2);
    res = parseint(res,&ip3);
    res = parseint(res,&cport);
    res = parseint(res,&dport);
    res = parseint(res,&nodestart);
    res = parseint(res,&nodesize);
    sprintf(ans, "%d ", cport);
    strcat(reply, ans);
  }
  
  req_reply(ip, port, pre, reply);
  free(origres);
  return REQ_OK;
}


int req_handle_worker(char *line)
{  
  unsigned int workerno, ip, port, i; 
  req_pipe p;
  
  sscanf(line+7,"%d%d%d",&workerno,&ip,&port);
  p = req_pipes + workerno;
  for (i=p->startnode; i<p->endnode; i++) 
  {
    nodetab_rank0_port[i] = port;
    req_host_IP = ip;
  }

  DEBUGF(("Master Thread: worker %d registered, with ip = %d, port = %d\n", workerno, ip, port));
  req_workers_registered++;
  return REQ_OK;
};


int req_handle(char *line)
{
  char cmd[100];
  sscanf(line,"%s",cmd);

  DEBUGF(("Request Handler: req -> %s\n", line));
  
  if      (strcmp(cmd,"aset")==0)       return req_handle_aset(line);
  else if (strcmp(cmd,"aget")==0)       return req_handle_aget(line);
  else if (strcmp(cmd,"scanf")==0)      return req_handle_scanf(line);
  else if (strcmp(cmd,"ping")==0)       return REQ_OK;
  else if (strcmp(cmd,"print")==0)      return req_handle_print(line);
  else if (strcmp(cmd,"printerr")==0)   return req_handle_printerr(line);
  else if (strcmp(cmd,"ending")==0)     return req_handle_ending(line);
  else if (strcmp(cmd,"abort")==0)      return req_handle_abort(line);
  else if (strcmp(cmd,"getinfo")==0)    return req_handle_getinfo(line);

  else if (strcmp(cmd,"worker")==0)     return req_handle_worker(line);
  else return REQ_FAILED;
}

void req_run_saved()
{
  int ok;
  
  req_node saved = req_saved;
  req_saved = 0;
  while (saved) 
  {
    req_node t = saved;
    saved = saved->next;
    ok = req_handle(t->request);
    if (ok==REQ_POSTPONE) 
    {
      t->next = req_saved;
      req_saved = t;
    } 
    else if (ok==REQ_FAILED) 
    {
      fprintf(stderr,"bad request: %s\n",t->request);
      free(t);
    } 
    else free(t);
  }
}


void req_write_to_host(HANDLE fd, char *buf, int size)
{
  int ok;
  int nwrote;
  
  while (size) 
  {
    ok = WriteFile(fd, buf, size, &nwrote, NULL);
    if ((ok == 0)&&(errno==EPIPE)) exit(0);
    if (ok == 0) 
    {
      fprintf(stderr, "worker writing to host: %d", GetLastError());
      exit(1);
    }
    size -= nwrote; buf += nwrote;
  }
}


void req_serve_client(int workerno)
{
  req_pipe worker;
  xstr buffer;
  int status, nread, len; char *head;
  req_node n;
  int CcsActiveFlag = 0;

  if(workerno == -1)
  {
    buffer = xstr_alloc();
    nread = xstr_readsock(buffer, CcsClientFd);

    /*** debugging ***/
    /*
    printf("Read the first part : %s, %d\n", xstr_lptr(buffer), nread);
    fflush(stdout);
    */

    /* if necessary, read the rest */
    while(nread != 0)
    {
      nread = xstr_readsock(buffer, CcsClientFd);
      
      /*** debugging ***/
      /*
      printf("Read the second part: %s\n", xstr_lptr(buffer));
      fflush(stdout);
      */
    }

  }
   
  else
  {
    worker = req_pipes+workerno;
    buffer = worker->buffer;
    nread  = xstr_read(buffer, worker->fd[0]);
  }

  if (nread<0) 
  {
    fprintf(stderr,"aborting: node terminated abnormally.\n");
    exit(1);
  }
  if((nread==0) && (workerno != -1)) return;
  
  while (1) 
  {
    head = xstr_lptr(buffer);
    len = strlen(head);
    if ((head+len == xstr_rptr(buffer)) && (workerno != -1)) 
    {
      break;
    }
    status = req_handle(head);
    switch (status) 
    {
      case REQ_OK: 
        if(workerno == -1) CcsActiveFlag = 1;
        break;
      case REQ_FAILED: 
        fprintf(stderr,"bad request: %s: %d\n",head, strlen(head)); 
        abort();
        break;
      case REQ_OK_AWAKEN: 
        req_run_saved(); 
        break;    
      case REQ_POSTPONE:
        n = (req_node)malloc(sizeof(struct req_node)+len);
        strcpy(n->request, head);
        n->next = req_saved;
        req_saved = n;
        if(workerno == -1) CcsActiveFlag = 1;
    }

    xstr_lshrink(buffer, len+1);
    
    if(CcsActiveFlag == 1) break;
  }
}


void req_poll()
{    
  int           i, peek, done;
  DWORD         bytesRead, totalBytes, bytesLeft;
  char          buffer[1024];
  static int    fin, *reg=NULL;

#if CMK_CCS_AVAILABLE
  unsigned int   clientIP;
  unsigned int   clientPortNo;
  int            status;
  struct fd_set  rfds;
  struct timeval tmo;
#endif
  
  if (!reg)
    reg = (int *) calloc(REQ_MAXFD, sizeof(int));

/*  The master thread serves only registration requests at first,
  only after all the workers have registered does it process other requests */

  if (!fin)
  {
    done = 1;
    for (i = 0; i <req_numworkers; i++)
      if (reg[i] == 0) done = 0;
    if (done) 
    {
      fin = 1;
      for (i = 0; i <req_numworkers; i++) reg[i] = 0;
    }
  }


#if CMK_CCS_AVAILABLE
  tmo.tv_sec = 0;
  tmo.tv_usec = 500000;

  FD_ZERO(&rfds);
  if (arg_server == 1) FD_SET(myFd, &rfds);
  status = select(FD_SETSIZE, &rfds, 0, 0, &tmo);

  if (arg_server ==1) 
  {
    if(FD_ISSET(myFd, &rfds))
    {
      /*      printf("Activity detected on client socket %d\n",myFd); */
      skt_accept(myFd, &clientIP, &clientPortNo, &CcsClientFd);
      /* printf("Accept over\n"); */
      fflush(stdout);
      req_serve_client(-1);
    }
  }
#endif

  /* Master thread peeks into each pipe and serves worker request if any activity is detected */
  for (i=0; i < req_numworkers; i++)
  {
    if (!reg[i]) 
      peek = PeekNamedPipe(req_pipes[i].fd[0], buffer, 
                  1024, &bytesRead, &totalBytes, &bytesLeft);

    if (peek == 0) 
    {
      fprintf(stderr, "PeekNamedPipe: %d", GetLastError());
      exit(1);
    }
    if (bytesRead > 0) 
    {
      req_serve_client(i);
      if (!fin) reg[i] = 1;
    }
  }
}


int req_worker(int workerno)
{  
  int            numclients;
  unsigned int   master_ip, master_port; 
  SOCKET         master_fd, client_fd;
  unsigned int   client_ip, client_port;
  int            i, status, nread, len, timeout;
  SOCKET         client[REQ_MAXFD];
  HANDLE         hostfd;
  xstr           clientbuf[REQ_MAXFD];
  struct fd_set  rfds; 
  xstr           buffer;
  struct timeval tv;
  char           reply[1024], *head;

  hostfd     = req_pipes[workerno].fd[1];
  numclients = req_pipes[workerno].endnode - req_pipes[workerno].startnode;
    master_port = 0;
  skt_server(&master_ip, &master_port, &master_fd);
  sprintf(reply,"worker %d %d %d", workerno, master_ip, master_port);
  req_write_to_host(hostfd, reply, strlen(reply)+1);

  DEBUGF(("Worker no. %d reporting, serverIP = %d, serverPort = %d\n", workerno, master_ip, master_port));
  DEBUGF(("Message sent to host: %s\n", reply));
  
  timeout = (nodetab_rank0_size*arg_timeout) + 60;
  DEBUGF(("Got here, timeout = %d\n", timeout));

  for (i=0; i<numclients; i++) 
  {
    DEBUGF(("Waiting for clients to connect.\n"));
    while (1) 
    {
      if (timeout <= 0) 
      {
        sprintf(reply,"abort timeout waiting for nodes to connect\n");
        req_write_to_host(hostfd, reply, strlen(reply)+1);
        return 1;
      }
    
      FD_ZERO(&rfds);
      FD_SET(master_fd, &rfds);
      tv.tv_sec = 1; tv.tv_usec = 0;
      status = select(FD_SETSIZE, &rfds, 0, 0, &tv);
      if (status == 1) break;
      if (status == SOCKET_ERROR) { fprintf(stderr, "accept: %d", WSAGetLastError()); return 1; }
      if (status == 0) { timeout--; continue; }
    }
    
    skt_accept(master_fd, &client_ip, &client_port, &client_fd);
    client[i] = client_fd;
    clientbuf[i] = xstr_alloc();
    DEBUGF(("Client %d connected\n", i));
  }
  
  while (1) 
  {

    DEBUGF(("Worker probing for client requests\n"));
    /* probe all file descriptors */
    FD_ZERO(&rfds);
    for (i=0; i<numclients; i++)
      FD_SET(client[i], &rfds);
    
    status = select(FD_SETSIZE, &rfds, 0, 0, 0);
    
    if (status == SOCKET_ERROR) 
    {
      fprintf(stderr, "worker error of some sort: %d", WSAGetLastError());
      return 1;
    }
    
    for(i=0; i<numclients; i++) 
    {
      if (FD_ISSET(client[i], &rfds)) 
      {
  
        /**** Debugging ***/
        DEBUGF(("Activity on %d\n", i));

        buffer = clientbuf[i];
        nread = xstr_readsock(buffer, client[i]);
        DEBUGF(("Read %d bytes\n", nread));
        if (nread<=0) { DEBUGF(("Exiting weirdly...\n")); return 1; }
        while (1) 
        {
          head = xstr_lptr(buffer);
          len = strlen(head);
          if (head+len == xstr_rptr(buffer)) break;
          req_write_to_host(hostfd, head, len+1);
          xstr_lshrink(buffer, len+1);
        }
      }
    }
  }
}


DWORD WINAPI WorkerThread(LPVOID lpvThreadParm)
{
  DWORD dwResult;
  dwResult = req_worker(*((int *) lpvThreadParm));

  ExitThread(dwResult);
  return 0; //never should get here
}


void req_start_workers()
{
  int i, nodenum, nclients;
  int *workerID;
  
  workerID = (int *) malloc(REQ_MAXFD*sizeof(int));
  req_numworkers = (nodetab_rank0_size+REQ_MAXFD-1) / REQ_MAXFD;
  
  if (req_numworkers > REQ_MAXFD) 
  {
    fprintf(stderr,"Too many PEs, file descriptors exceeded.\n");
    exit(1);
  }
  nodenum = 0;

  DEBUGF(("INFO> In req_start_workers, about to create worker threads\n"));
  
  for (i=0; i<req_numworkers; i++) 
  {
    nclients = REQ_MAXFD;
    if (nclients > (nodetab_rank0_size-nodenum))
      nclients = (nodetab_rank0_size-nodenum);
    req_pipes[i].startnode = nodenum;
    req_pipes[i].endnode = nodenum + nclients;
    CreatePipe(&req_pipes[i].fd[0], &req_pipes[i].fd[1], NULL, 0);
    workerID[i] = i;
    req_pipes[i].tHandle = CreateThread(NULL, 0, WorkerThread, 
                        (LPVOID) (&workerID[i]), 0,  &req_pipes[i].threadID);
    //if (req_pipes[i].pid==0) req_worker(i);
    req_pipes[i].buffer = xstr_alloc();
    //_close(req_pipes[i].fd[1]);
    nodenum += nclients;
  }
  
  DEBUGF(("INFO> In req_start_workers, worker threads successfully created\n"));

}



void start_nodes()
{
  taskStruct task;
  char argBuffer[5000];//Buffer to hold assembled program arguments
  int i,nodeNumber;

    /*Set the parts of the task structure that will be the same for all nodes*/
  strcpy(task.pgm,arg_nodeprog_a);
  /*Figure out the command line arguments (same for all PEs)*/
  argBuffer[0]=0;
  for (i=0;arg_argv[i];i++) 
  {
    strcat(argBuffer," ");
    strcat(argBuffer,arg_argv[i]);
  }
  task.argLength=strlen(argBuffer);
  strcpy(task.cwd,arg_currdir_a);/*HACK!  The run directory needs to come from nodelist file*/
  
  task.magic=DAEMON_MAGIC;

/*Start up the user program, by sending a message
  to PE 0 on each node.*/
  for (nodeNumber=0;nodeNumber<nodetab_rank0_size;nodeNumber++)
  {
    char statusCode='N';/*Default error code-- network problem*/
    int fd;
    int pe0=nodetab_rank0_table[nodeNumber];
    /*Set the node-varying parts of the task structure*/
    
    task.Cmi_numpes=nodetab_size;
    task.Cmi_mynode=nodeNumber;
    task.Cmi_mynodesize=nodetab_cpus(pe0);
    task.Cmi_numnodes=nodetab_rank0_size;
    task.Cmi_nodestart=pe0;
  
    task.Cmi_self_IP=htonl(nodetab_ip(pe0));
    task.Cmi_host_IP=htonl(req_host_IP);
    task.Cmi_host_port=nodetab_rank0_port[nodeNumber];
    task.Cmi_host_pid=(getpid()&0x7FFF);

    DEBUGF(("Task struct contents:\n"));
    DEBUGF(("task.pgm = %s\n", task.pgm));
    DEBUGF(("task.cwd = %s\n", task.cwd));
    DEBUGF(("task.Cmi_numpes = %d\n", task.Cmi_numpes));
    DEBUGF(("task.Cmi_mynode = %d\n", task.Cmi_mynode));
    DEBUGF(("task.Cmi_mynodesize = %d\n", task.Cmi_mynodesize));
    DEBUGF(("task.Cmi_numnodes = %d\n", task.Cmi_numnodes));
    DEBUGF(("task.Cmi_nodestart = %d\n", task.Cmi_nodestart));
    DEBUGF(("task.Cmi_self_IP = %d\n", task.Cmi_self_IP));
    DEBUGF(("task.Cmi_host_IP = %d\n", task.Cmi_host_IP));
    DEBUGF(("task.Cmi_host_port = %d\n", task.Cmi_host_port));

    /*Send request out to remote node*/
    fd = skt_connect(task.Cmi_self_IP, DAEMON_IP_PORT);
    if (fd>0)
    {/*Contact!  Ask the daemon to start the program*/
      SendSocketN(fd, (BYTE *)&task, sizeof(task));
      SendSocketN(fd, (BYTE *)argBuffer, strlen(argBuffer));
      RecvSocketN(fd, (BYTE *)&statusCode,sizeof(char));
    }
    if (statusCode!='G')
    {/*Something went wrong--*/
      fprintf(stderr,"Error '%c' starting remote node program--\n%s\n",statusCode,
        daemon_status2msg(statusCode));
      exit(1);
    }
  }
}


/****************************************************************************
 *
 *  The Main Program
 *
 ****************************************************************************/

int main(int argc, char **argv)
{  
#if CMK_CCS_AVAILABLE
  unsigned int myIP, myPortNo;
#endif
  WSADATA  WSAData;

  WSAStartup(0x0002, &WSAData);
  srand(time(NULL));
  
  /* notify charm developers that charm is in use */
  ping_developers();
  
  /* Compute the values of all constants */
  arg_init(argc, argv);
  if(arg_verbose)
    fprintf(stderr, "INFO> conv-host started...\n");
  
  /* Initialize the node-table by reading nodesfile */
  nodetab_init();

#if CMK_CCS_AVAILABLE
  if(arg_server == 1){
    myPortNo = arg_server_port;
    skt_server(&myIP, &myPortNo, &myFd);
    printf("ccs: %s\nccs: Server IP = %u, Server port = %u $\n", 
           CMK_CCS_VERSION, myIP, myPortNo);
    fflush(stdout);
  }
#endif

  DEBUGF(("INFO> node table initialized\n"));

  DEBUGF(("INFO> starting workers\n"));

  /* Start the worker processes */
  req_start_workers();

  /* Wait until the workers have registered */
  while (req_workers_registered < req_numworkers) req_poll();
  /* Initialize the IO module */
  input_init();
  /* start the node processes */
  start_nodes();

  DEBUGF(("INFO> node programs started\n"));
  
  /* enter request-service mode */
  while (1) req_poll();
  WSACleanup();

}

