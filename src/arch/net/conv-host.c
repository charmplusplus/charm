/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/


#include "conv-mach.h"
#include "converse.h"

#include <sys/types.h>
#include <stdio.h>
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <netinet/in.h>
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

#if CMK_CCS_AVAILABLE
int CcsClientFd;
int myFd;
#endif

#if CMK_DEBUG_MODE
unsigned int clientIP = 0;
int clientPort;
int clientKillPort;
int *serverCCSPorts;
int *openStatus;
#define MAX_GDBS 200
#define MAXREADBYTES 2048
#endif

static void jsleep(int sec, int usec)
{
  int ntimes,i;
  struct timeval tm;

  ntimes = (sec*1000000+usec)/5000;
  for(i=0;i<ntimes;i++) {
    tm.tv_sec = 0;
    tm.tv_usec = 5000;
    while(1) {
      if (select(0,NULL,NULL,NULL,&tm)==0) break;
      if (errno!=EINTR) return;
    }
  }
}

int probefile(path)
    char *path;
{
  struct stat s;
  int ok = stat(path, &s);
  if (ok<0) return 0;
  if (!S_ISREG(s.st_mode)) return 0;
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
  char info[1000];
  struct sockaddr_in addr;
  int skt;
  skt = socket(AF_INET, SOCK_DGRAM, 0);
  if (skt < 0) return;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(6571);
  addr.sin_addr.s_addr = htonl(0x80aef1d3);
  sprintf(info,"%d",getuid());
  sendto(skt, info, strlen(info), 0, (struct sockaddr *)&addr, sizeof(addr));
  close(skt);
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
  if (ip==0) {
    if (gethostname(hostname, 99)<0) ip=0x7f000001;
    hostent = gethostbyname(hostname);
    if (hostent == 0) return 0x7f000001;
    ip = htonl(*((int *)(hostent->h_addr_list[0])));
  }
  return ip;
}

void skt_server(pip,ppo,pfd)
    unsigned int *pip;
    unsigned int *ppo;
    unsigned int *pfd;
{
  int fd= -1;
  int ok, len;
  struct sockaddr_in addr;
  
  fd = socket(PF_INET, SOCK_STREAM, 0);
  if (fd < 0) { perror("ERROR> socket"); exit(1); }
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  if(*ppo != 0) { addr.sin_port = htons(*ppo); }
  ok = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  if (ok < 0) { perror("ERROR> bind"); exit(1); }
  ok = listen(fd,5);
  if (ok < 0) { perror("ERROR> listen"); exit(1); }
  len = sizeof(addr);
  ok = getsockname(fd, (struct sockaddr *)&addr, &len);
  if (ok < 0) { perror("ERROR> getsockname"); exit(1); }

  *pfd = fd;
  *pip = skt_ip();
  *ppo = ntohs(addr.sin_port);
}

void skt_accept(src,pip,ppo,pfd)
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
  if ((fd<0)&&(errno==EMFILE)) goto acc;
  if (fd<0) {
    perror("accept"); 
    exit(1);
  }
  *pip=htonl(remote.sin_addr.s_addr);
  *ppo=htons(remote.sin_port);
  *pfd=fd;
}

int skt_connect(ip, port)
    unsigned int ip; int port;
{
  struct sockaddr_in remote; short sport=port;
  int fd, ok, len;
    
  /* create an address structure for the server */
  memset(&remote, 0, sizeof(remote));
  remote.sin_family = AF_INET;
  remote.sin_port = htons(sport);
  remote.sin_addr.s_addr = htonl(ip);
    
 sock:
  fd = socket(AF_INET, SOCK_STREAM, 0);
  if ((fd<0)&&(errno=EMFILE)) goto sock;
  if (fd < 0) return -1;
  
 conn:
  ok = connect(fd, (struct sockaddr *)&(remote), sizeof(remote));
  if (ok<0) {
    switch (errno) {
    case EADDRINUSE: close(fd); goto sock;
    default: return -1;
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
	  (ip>>24)&0xFF,
	  (ip>>16)&0xFF,
	  (ip>>8)&0xFF,
	  (ip>>0)&0xFF);
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
  if (((*p)<'0')||((*p)>'9')) {
    fprintf(stderr,"badly-formed number");
    exit(1);
  }
  while ((*p>='0')&&(*p<='9')) { val*=10; val+=(*p)-'0'; p++; }
  *value = val;
  return p;
}

char *getenv_rsh()
{
  char *e;

  e = getenv("CONV_RSH");
  return e ? e : RSH_CMD;
}

char *getenv_display()
{
  static char result[100];
  char *e, *p;
  
  e = getenv("DISPLAY");
  if (e==0) return 0;
  p = strrchr(e, ':');
  if (p==0) return 0;
  if ((e[0]==':')||(strncmp(e,"unix:",5)==0)) {
    sprintf(result,"%s:%s",text_ip(skt_ip()),p+1);
  }
  else strcpy(result, e);
  return result;
}

char *mylogin()
{
  struct passwd *self;

  self = getpwuid(getuid());
  if (self==0) { perror("ERROR> getpwuid"); exit(1); }
  return self->pw_name;
} 

unsigned int lookup_ip(char *name)
{
  struct hostent *h;
  unsigned int ip1,ip2,ip3,ip4; int nread;
  nread = sscanf(name,"%d.%d.%d.%d",&ip1,&ip2,&ip3,&ip4);
  if (nread==4) return (ip1<<24)|(ip2<<16)|(ip3<<8)|ip4;
  h = gethostbyname(name);
  if (h==0) return 0;
  return htonl(*((int *)(h->h_addr_list[0])));
}

void strsubst(char *str, char c1, char c2)
{
  while (*str) {
    if (*str==c1) *str=c2;
    str++;
  }
}

/*****************************************************************************
 *                                                                           *
 * PPARAM - obtaining "program parameters" from the user.                    *
 *                                                                           *
 *****************************************************************************/

typedef struct ppdef
{
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
}
*ppdef;

static ppdef ppdefs;

static int     pparam_pos;
static char  **pparam_argv;
static char    pparam_optc='-';
char           pparam_error[100];

static ppdef pparam_find(lname)
    char *lname;
{
  ppdef def;
  for (def=ppdefs; def; def=def->next)
    if (strcmp(def->lname, lname)==0)
      return def;
  return 0;
}

static ppdef pparam_cell(lname)
    char *lname;
{
  ppdef def = pparam_find(lname);
  if (def) return def;
  def = (ppdef)malloc(sizeof(struct ppdef));
  def->lname = lname;
  def->type  = 's';
  def->doc   = "(undocumented)";
  def->next  = ppdefs;
  ppdefs = def;
  return def;
}

void pparam_doc(lname, doc)
    char *lname; char *doc;
{
  ppdef def = pparam_cell(lname);
  def->doc = doc;
}

void pparam_defint(lname, value)
    char *lname; int value;
{
  ppdef def = pparam_cell(lname);
  def->type  = 'i';
  def->value.i = value;
}

void pparam_defreal(lname, value)
    char *lname; double value;
{
  ppdef def = pparam_cell(lname);
  def->type  = 'r';
  def->value.r = value;
}

void pparam_defstr(lname, value)
    char *lname; char *value;
{
  ppdef def = pparam_cell(lname);
  def->type  = 's';
  def->value.s = value;
}

void pparam_defflag(lname)
    char *lname;
{
  ppdef def = pparam_cell(lname);
  def->type  = 'f';
  def->value.f = 0;
}

static ppdef pparam_hfind(lname)
    char *lname;
{
  ppdef def = pparam_find(lname);
  if (def) return def;
  fprintf(stderr,"ERROR> No such program parameter %s\n",lname);
  exit(1);
}

int pparam_getint(lname)
    char *lname;
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 'i') return 0;
  return def->value.i;
}

double pparam_getreal(lname)
    char *lname;
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 'r') return 0.0;
  return def->value.r;
}

char *pparam_getstr(lname)
    char *lname;
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 's') return 0;
  return def->value.s;
}

int pparam_getflag(lname)
    char *lname;
{
  ppdef def = pparam_hfind(lname);
  if (def->type != 'f') return 0;
  return def->value.f;
}

static int pparam_setdef(def, value)
    ppdef def; char *value;
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
    }
}

int pparam_set(lname, value)
    char *lname; char *value;
{
  ppdef def = pparam_cell(lname);
  return pparam_setdef(def, value);
}

char *pparam_getdef(def)
    ppdef def;
{
  static char result[100];
  switch(def->type)
    {
    case 'i': sprintf(result,"%d", def->value.i); return result;
    case 'r': sprintf(result,"%lf",def->value.r); return result;
    case 's': return def->value.s;
    case 'f': sprintf(result,"%d", def->value.f); return result;
    }
}

void pparam_printdocs()
{
  ppdef def; int i, len, maxname, maxdoc;
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

void pparam_delarg(i)
    int i;
{
  int j;
  for (j=i; pparam_argv[j]; j++)
    pparam_argv[j]=pparam_argv[j+1];
}

int pparam_countargs(argv)
    char **argv;
{
  int argc;
  for (argc=0; argv[argc]; argc++);
  return argc;
}

int pparam_parseopt()
{
  int ok; ppdef def;
  char *opt = pparam_argv[pparam_pos];
  /* handle ++ by skipping to end */
  if ((opt[1]=='+')&&(opt[2]==0))
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
      name[1]=0;
      def = pparam_find(name);
    }
  if (def==0)
    {
      pparam_pos++;
      return 0;
/*
   sprintf(pparam_error,"Option %s not recognized.",opt);
   return -1;
*/
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
  if ((opt[1]=='+')||(opt[2]==0))
    {
      pparam_delarg(pparam_pos);
      opt = pparam_argv[pparam_pos];
    }
  else opt+=2;
  if ((opt == 0)||(opt[0] == 0))
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

int pparam_parsecmd(optchr, argv)
    char optchr; char **argv;
{
  pparam_error[0]=0;
  pparam_argv = argv;
  pparam_optc = optchr;
  pparam_pos  = 0;
  while(1)
    {
      char *opt = pparam_argv[pparam_pos];
      if (opt==0) break;
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

typedef struct xstr
    {
    char *lptr;
    char *rptr;
    char *lend;
    char *rend;
    }
    *xstr;

char *xstr_lptr(l) xstr l; { return l->lptr; }
char *xstr_rptr(l) xstr l; { return l->rptr; }

int xstr_len(l)
    xstr l;
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

void xstr_free(s)
    xstr s;
{
  free(s->lend);
  free(s);
}

void xstr_rexpand(l, nbytes)
    xstr l; int nbytes;
{
  int lspace, rspace, uspace, needed; char *nbuf;
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

void xstr_lexpand(l, nbytes)
    xstr l; int nbytes;
{
  int lspace, rspace, uspace, needed; char *nbuf;
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

void xstr_rshrink(l, nbytes)
    xstr l; int nbytes;
{
  if (l->rptr - l->lptr < nbytes) { l->rptr=l->lptr; return; }
  l->rptr -= nbytes;
  *(l->rptr) = 0;
}

void xstr_lshrink(l, nbytes)
    xstr l; int nbytes;
{
  if (l->rptr - l->lptr < nbytes) { l->lptr=l->rptr; return; }
  l->lptr += nbytes;
}

void xstr_write(l, bytes, nbytes)
    xstr l; char *bytes; int nbytes;
{
  xstr_rexpand(l, nbytes);
  memcpy(xstr_lptr(l)+xstr_len(l)-nbytes, bytes, nbytes);
}

int xstr_read(xstr l, int fd)
{
  int nread;
  xstr_rexpand(l, 1024);
  nread = read(fd, xstr_rptr(l)-1024, 1024);
  if (nread<0) xstr_rshrink(l, 1024);
  else xstr_rshrink(l, 1024-nread);
  return nread;
}

void xstr_printf(va_alist) va_dcl
{
  char buffer[10000];
  xstr l; char *fmt;
  va_list p;
  va_start(p);
  l = va_arg(p, xstr);
  fmt = va_arg(p, char *);
  vsprintf(buffer, fmt, p);
  xstr_write(l, buffer, strlen(buffer));
}

char *xstr_gets(buff, size, s)
    char *buff; int size; xstr s;
{
  char *p; int len;
  xstr_rptr(s)[0]=0;
  p = strchr(xstr_lptr(s),'\n');
  if (p==0) return 0;
  *p = 0;
  len = p - xstr_lptr(s);
  if (len > size) len=size;
  memcpy(buff, xstr_lptr(s), len);
  buff[len] = 0;
  xstr_lshrink(s, len+1);
  return buff;
}

/****************************************************************************
 *
 * PROG - much like 'popen', but better.
 *
 *
 * typedef prog
 *
 *  - represents an opened, running program.
 *
 * prog prog_start(char *prog, char **argv, use_err)
 *
 *  - starts a program in the background (with the same args as execv).
 *    The prog returned can be used to read the standard input and standard
 *    output of the resulting program.  'use_err' is a flag, if zero, then
 *    the program's standard error is merged with its standard output,
 *    otherwise it is kept separate.
 *
 *    The program P has three file descriptors (P->ifd, P->ofd, P->efd)
 *    which can be used to access its standard input, output, and error.
 *    In addition, it has three xstr buffers (P->ibuf, P->obuf, P->ebuf)
 *    which are used for buffered IO on those descriptors.
 *
 * int prog_flush(prog p)
 *
 *  - flushes the contents of P->ibuf into P->ifd.
 *
 * void prog_iclose(prog p)
 * 
 *  - close the input-side of the specified program.  You may not write to
 *    the standard input of the program after prog_iclose.
 *
 * void prog_close(prog p)
 *
 *  - close the standard inputs and outputs of the specified program,
 *    and free all resources used by the handle.  The program may continue
 *    to exist in the background if it is capable of doing so without a
 *    standard input and output.
 *
 ****************************************************************************/

typedef struct prog
{
  int ifd; xstr ibuf;
  int ofd; xstr obuf;
  int efd; xstr ebuf;
  int pid;
}
*prog;

int prog_flush(c)
    prog c;
{
  xstr ibuf = c->ibuf;
  int ifd = c->ifd;
  
  if (ibuf==0) return;
  while (xstr_lptr(ibuf)!=xstr_rptr(ibuf))
    {
      int nwrote = write(ifd, xstr_lptr(ibuf), xstr_len(ibuf));
      if (nwrote < 0) return -1;
      if (nwrote==0)
	{ fprintf(stderr,"ERROR> write returned 0???\n"); exit(1); }
      xstr_lshrink(ibuf, nwrote);
    }
  return 0;
}

void prog_iclose(c)
    prog c;
{
  prog_flush(c);
  if (c->ibuf) { xstr_free(c->ibuf); close(c->ifd); }
  c->ibuf = 0;
}

void prog_close(c)
    prog c;
{
  prog_flush(c);
  if (c->ibuf) { xstr_free(c->ibuf); close(c->ifd); }
  if (c->obuf) { xstr_free(c->obuf); close(c->ofd); }
  if (c->ebuf) { xstr_free(c->ebuf); close(c->efd); }
  free(c);
}

prog prog_make(ifd, ofd, efd, pid)
    int ifd, ofd, efd, pid;
{
  prog res = (prog)malloc(sizeof(struct prog));
  res->ifd = ifd;
  res->ofd = ofd;
  res->efd = efd;
  res->ibuf = (ifd >= 0) ? xstr_alloc() : NULL;
  res->obuf = (ofd >= 0) ? xstr_alloc() : NULL;
  res->ebuf = (efd >= 0) ? xstr_alloc() : NULL;
  res->pid = pid;
  return res;
}

prog prog_start(p, argv, useerr)
    char *p; char **argv; int useerr;
{
  int p_stdin[2];
  int p_stdout[2];
  int p_stderr[2];
  int pid;
  p_stdin[0]= -1; p_stdout[0]= -1; p_stderr[0]= -1;
  p_stdin[1]= -1; p_stdout[1]= -1; p_stderr[1]= -1;
  if (pipe(p_stdin )<0) goto abort;
  if (pipe(p_stdout)<0) goto abort;
  if (pipe(p_stderr)<0) goto abort;
  pid = 0;
  pid = fork();
  if (pid < 0) goto abort;
  if (pid == 0)
    {
      int i;
      dup2(p_stdin[0],0);
      dup2(p_stdout[1],1);
      dup2(useerr?p_stderr[1]:p_stdout[1],2);
      for(i=3; i<128; i++) close(i);
      execvp(p, argv);
      exit(1);
    }
  close(p_stdin[0]);
  close(p_stdout[1]);
  close(p_stderr[1]);
  return prog_make(p_stdin[1], p_stdout[0], p_stderr[0], pid);
 abort:
  if (p_stdin[0]!= -1) close(p_stdin[0]);
  if (p_stdin[1]!= -1) close(p_stdin[1]);
  if (p_stdout[0]!= -1) close(p_stdout[0]);
  if (p_stdout[1]!= -1) close(p_stdout[1]);
  if (p_stderr[0]!= -1) close(p_stderr[0]);
  if (p_stderr[1]!= -1) close(p_stderr[1]);
  return 0;
}

#if CMK_DEBUG_MODE

/***********************************************************************
 * PROG_GDB : This is an extension to prog, which allows the the debugger
 * to interface with gdb, and allow the debugger client to have a gdb
 * like environnment.
 *
 * typedef prog_gdb : an extension to prog, but additionally, the 'pe' 
 * and the 'open' field (open = 0/1 depending on whether gdb has
 * 'broken out' of execution.
 *
 * void checkForGDBActivitity() - scans all the prog_gdb instances,
 * and sees if any activity (GDB break out) is present. If so, sets
 * the openProcessingNeeded flag to 1.
 *
 * void processOnOpen() - Carries out processing on all the 'open'
 * prog_gdb instances. This may involve executing a user command
 * on the input buffer of a particular prog_gdb
 *
 * sendCCSReply(unsigned ip, unsigned port, int destProc, int size, void *msg)
 * Sends the message to the client in the CCS format.
 *
 * int readUntilString(int fd, char *buffer, char *match) - reads from 
 * the specified file descriptor, until it finds the specified
 * string. Stores the result in 'buffer'. It returns the number of 
 * bytes read
 *
 ***********************************************************************/

typedef struct prog_gdb {
  prog p;
  int pe;
  int open;
} prog_gdb;

void writeall(int fd, char *buf, int size)
{
  int ok;
  while (size) {
  retry:
    ok = write(fd, buf, size);
    if ((ok<0)&&((errno==EBADF)||(errno==EINTR))) goto retry;
    if (ok<=0) {
      fprintf(stderr, "Write failed ..\n");
      exit(1);
    }
    size-=ok; buf+=ok;
  }
}

prog_gdb rsh_prog_gdb[MAX_GDBS];
int numGDBs;
int GDBActiveFlag = 0;
int openProcessingNeeded = 0;
char *command;
int commandPresent = 0;
int commandSent = 0;
int destProcessor = -1;

void sendCCSReply(unsigned int ip, unsigned int port, int destProc, int size, void *msg){
  char cmd[100];
  int fd;

  fd = skt_connect(ip, port);
  if (fd<0) {
    printf("client Exited : %d %d\n", ip, port);
    return; /* maybe the requester exited */
  }
  sprintf(cmd, "reply %d %d\n", size, destProc);
  writeall(fd, cmd, strlen(cmd));
  writeall(fd, msg, size);

  close(fd);
}

int readUntilString(int fd, char *buffer, char *match){
  int readBytes = 0, totalReadBytes = 0;
  int found = 0;
  char *str;
  int junk;
  
  /*
  printf("Inside readUntilString()\n");
  fflush(stdout);
  */
  
  while(!found){
    readBytes = read(fd, buffer + totalReadBytes, MAXREADBYTES);
    totalReadBytes += readBytes;
    buffer[totalReadBytes] = 0; 
	
    str = strstr(buffer, match);
    if(str != NULL){
      found = 1;
    }
    
    /*
    printf("string read = %s: %d : %d\n", buffer, strlen(buffer), totalReadBytes);
    fflush(stdout);
    */
  }
  
  return(totalReadBytes);
}

void checkForGDBActivity(){
  fd_set rfds;
  int maxfd,i;
  int nreadable;
  long diff;
  struct timeval tmo, begin, current;

  /* Debugging */
  char buffer[1000];
  int nread;

  maxfd = 0;
  FD_ZERO(&rfds);
  for(i = 0; i < numGDBs; i++){
    FD_SET((rsh_prog_gdb[i].p)->ofd, &rfds);
    if ((rsh_prog_gdb[i].p)->ofd > maxfd) maxfd = (rsh_prog_gdb[i].p)->ofd;
  }

  tmo.tv_sec = 0;
  tmo.tv_usec = 500000;
  gettimeofday(&begin, NULL);
  while(1){
    nreadable = select(maxfd + 1, &rfds, NULL, NULL, &tmo);
    if ((nreadable<0)&&((errno==EINTR)||(errno==EBADF))) {
      gettimeofday(&current, NULL);
      diff = (begin.tv_sec == current.tv_sec) ? current.tv_usec -
	begin.tv_usec : (1000000 + current.tv_usec - begin.tv_usec);
      if(diff < 0) break;
      else tmo.tv_usec = diff;
      continue;
    }
    break;
  }
  if(nreadable > 0) {
    openProcessingNeeded = 1;
    for(i = 0; i < numGDBs; i++){
      if(FD_ISSET((rsh_prog_gdb[i].p)->ofd, &rfds)){
	rsh_prog_gdb[i].open = 1;

	/*
	printf("activity seen on : %d\n", i);
	fflush(stdout);
	*/

	/* Debugging */
	/* nread = read((rsh_prog_gdb[i].p)->ofd, buffer, 1000);
	if(nread < 0){
	  printf("Error in read...\n");
	  exit(1);
	}
	else{
	  printf("read stuff = %s, %d\n", buffer, nread);
	}
	*/
	/* */

      }
      else
	rsh_prog_gdb[i].open = 0;
    }
  }
  else{
    openProcessingNeeded = 0;
  }
}

void processOnOpen(void)
{
  int readBytes, writeBytes, i;
  char buffer[MAXREADBYTES];
  int fd;
  char line[1024];
  char gdbRequest[] = "gdbRequest";
  int size = strlen(gdbRequest) + 1;
  int junkIP = 1234;
  int junkPort = 56789;
  
  unsigned int nodetab_ip(int);

  for(i = 0; i < numGDBs; i++){
    if(rsh_prog_gdb[i].open == 1){
      readBytes = readUntilString((rsh_prog_gdb[i].p)->ofd, buffer, "(gdb) ");
      if(readBytes < 0){
        printf("Error reading gdb pipe %d", i);
        exit(1);
      }
      /*
      printf("Sending CCS Reply for node %d:buffer contents:%s\n", i, buffer);
      fflush(stdout);
      */
      sendCCSReply(clientIP, clientKillPort, i, strlen(buffer) + 1, buffer);
      openStatus[i] = 1;
    }
  }
  if(commandPresent == 1){
    if(openStatus[destProcessor] == 1){
      rsh_prog_gdb[destProcessor].open = 0;
      writeBytes = write((rsh_prog_gdb[destProcessor].p)->ifd, command,
                         strlen(command));
      if(writeBytes < 0){
        printf("Error executing gdb command : %s : %d : proc %d\n",
               command, strlen(command), destProcessor);
        exit(1);
      }
      fd = skt_connect(clientIP, clientKillPort);
      writeall(fd, "unfreeze", strlen("unfreeze") + 1);
      close(fd);

      /*
      printf("1.sent unfreeze (to Debugger Client)...%s %d\n", command, writeBytes);
      */

      if(strncmp(command, "c", 1) == 0){
        int nread;

        /* Debugging */
        /* printf("CHECK : 'continue' pressed\n"); */

        nread = read((rsh_prog_gdb[destProcessor].p)->ofd, buffer, 1000);
        if(nread < 0){
          printf("Error in read...\n");
          exit(1);
        }
      }

      openStatus[destProcessor] = 0;
      free(command);
      commandPresent = 0;
      commandSent = 0;
      command = 0;
      destProcessor = -1;
    }
    else{
      fd = skt_connect(nodetab_ip(destProcessor), serverCCSPorts[destProcessor]);
      sprintf(line, "req %d %d %d %d %s\n", destProcessor, size, junkIP, junkPort, "DebugHandler");
      write(fd, line, strlen(line));
      write(fd, gdbRequest, size);
      close(fd);
    }
  }

  openProcessingNeeded = 0;
}

#endif

/****************************************************************************
 * 
 * ARG
 *
 * The following module computes a whole bunch of miscellaneous values, which
 * are all constant throughout the program.  Naturally, this includes the
 * value of the command-line arguments.
 *
 *****************************************************************************/


#define MAX_NODES 1000
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
char *arg_shell;
char *arg_debugger;
char *arg_xterm;

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
  static char buf[1024]; int len, i;
  
  pparam_defint ("p"             ,  MAX_NODES);
  pparam_defint ("timeout"       ,  2);
  pparam_defflag("verbose"           );
  pparam_defflag("debug"             );
  pparam_defflag("debug-no-pause"    );
#if CMK_CCS_AVAILABLE
  pparam_defflag("server"            );
  pparam_defint ("server-port"   , 0 );
#endif
#if CMK_DEBUG_MODE
  pparam_defflag("gdbinterface"      );
  pparam_defflag("initial_bp"        );
#endif
  pparam_defflag("in-xterm"          );
  pparam_defint ("maxrsh"        ,  16);
  pparam_defstr ("nodelist"      ,  0);
  pparam_defstr ("nodegroup"     ,  "main");
  pparam_defstr ("remote-shell"  ,  0);
  pparam_defstr ("debugger"      ,  0);
  pparam_defstr ("xterm"         ,  0);
  
  pparam_doc("p",             "number of processes to create");
  pparam_doc("timeout",       "seconds to wait per host connection");
  pparam_doc("in-xterm",      "Run each node in an xterm window");
  pparam_doc("verbose",       "Print diagnostic messages");
  pparam_doc("debug",         "Run each node under gdb in an xterm window");
  pparam_doc("debug-no-pause","Like debug, except doesn't pause at beginning");
#if CMK_CCS_AVAILABLE
  pparam_doc("server",        "Enable client-server mode");
  pparam_doc("server-port",   "Port to listen for CCS requests");
#endif
#if CMK_DEBUG_MODE
  pparam_doc("gdbinterface",  "Allow the gdb interface to be integrated");
  pparam_doc("initial_bp"  ,  "Allow the program to break at the initial CsdScheduler call");
#endif
  pparam_doc("maxrsh",        "Maximum number of rsh's to run at a time");
  pparam_doc("nodelist",      "file containing list of nodes");
  pparam_doc("nodegroup",     "which group of nodes to use");
  pparam_doc("remote-shell",  "which remote shell to use");
  pparam_doc("debugger",      "which debugger to use");
  pparam_doc("xterm",         "which xterm to use");

  if (pparam_parsecmd('+', argv) < 0) {
    fprintf(stderr,"ERROR> syntax: %s\n",pparam_error);
    pparam_printdocs();
    exit(1);
  }
  arg_argv = argv+2;
  arg_argc = pparam_countargs(argv+2);
  
  arg_requested_pes  = pparam_getint("p");
  arg_timeout        = pparam_getint("timeout");
  arg_in_xterm       = pparam_getflag("in-xterm");
  arg_verbose        = pparam_getflag("verbose");
  arg_debug          = pparam_getflag("debug");
  arg_debug_no_pause = pparam_getflag("debug-no-pause");

#if CMK_CCS_AVAILABLE
  arg_server         = pparam_getflag("server");
  arg_server_port    = pparam_getint("server-port");
#endif
#if CMK_DEBUG_MODE
  arg_gdbinterface   = pparam_getflag("gdbinterface");
  arg_initial_bp     = pparam_getflag("initial_bp");
#endif
  arg_maxrsh         = pparam_getint("maxrsh");
  arg_nodelist       = pparam_getstr("nodelist");
  arg_nodegroup      = pparam_getstr("nodegroup");
  arg_shell          = pparam_getstr("remote-shell");
  arg_debugger       = pparam_getstr("debugger");
  arg_xterm          = pparam_getstr("xterm");

  arg_verbose = arg_verbose || arg_debug || arg_debug_no_pause;
  
  /* Find the current value of the CONV_RSH variable */
  if(!arg_shell)
    arg_shell = getenv_rsh();

  /* Find the current value of the DISPLAY variable */
  arg_display = getenv_display();
  if ((arg_debug || arg_debug_no_pause || arg_in_xterm) && (arg_display==0)) {
    fprintf(stderr,"ERROR> DISPLAY must be set to use debugging mode\n");
    exit(1);
  }

  /* default debugger is gdb */
  if(!arg_debugger)
    arg_debugger = "gdb" ;
  /* default xterm is xterm */
  if(!arg_xterm)
    arg_xterm = "xterm" ;

  /* find the current directory, absolute version */
  getcwd(buf, 1023);
  arg_currdir_a = strdup(buf);
  
  /* find the node-program, absolute version */
  if (argc<2) {
    fprintf(stderr,"ERROR> You must specify a node-program.\n");
    exit(1);
  }
  if (argv[1][0]=='/') {
    arg_nodeprog_a = argv[1];
  } else {
    sprintf(buf,"%s/%s",arg_currdir_a,argv[1]);
    arg_nodeprog_a = strdup(buf);
  }

  arg_mylogin = mylogin();
}

/****************************************************************************
 *                                                                           
 * NODETAB:  The nodes file and nodes table.
 *
 ****************************************************************************/

char *nodetab_file_find()
{
  /* Find a nodes-file as specified by ++nodelist */
  if (arg_nodelist) {
    char *path = arg_nodelist;
    if (probefile(path)) return strdup(path);
    fprintf(stderr,"ERROR> No such nodelist file %s\n",path);
    exit(1);
  }
  /* Find a nodes-file as specified by getenv("NODELIST") */
  if (getenv("NODELIST")) {
    char *path = getenv("NODELIST");        
    if (path && probefile(path)) return strdup(path);
    fprintf(stderr,"ERROR> Cannot find nodelist file %s\n",path);
    exit(1);
  }
  /* Find a nodes-file by looking under 'nodelist' in the current directory */
  if (probefile("./nodelist")) return strdup("./nodelist");
  if (getenv("HOME")) {
    char buffer[MAXPATHLEN];
    sprintf(buffer,"%s/.nodelist",getenv("HOME"));
    if (probefile(buffer)) return strdup(buffer);
  }
  fprintf(stderr,"ERROR> Env. Var. HOME not set. Cannot find a nodes file.\n");
  exit(1);
}

typedef struct nodetab_host {
  char    *name;
  char    *login;
  char    *passwd;
  pathfixlist pathfixes;
  char    *ext;
  char    *setup;
  int      cpus;
  int      rank;
  double   speed;
  unsigned int ip;
  char    *shell;
  char    *debugger ;
  char    *xterm ;
} *nodetab_host;

nodetab_host *nodetab_table;
int           nodetab_max;
int           nodetab_size;
int          *nodetab_rank0_port;
int          *nodetab_rank0_table;
int           nodetab_rank0_size;

char        *default_login;
char        *default_group;
char        *default_passwd;
pathfixlist  default_pathfixes;
char        *default_ext;
char        *default_setup;
double       default_speed;
int          default_cpus;
int          default_rank;
char        *default_shell;
char        *default_debugger;
char        *default_xterm;

char        *host_login;
char        *host_group;
char        *host_passwd;
pathfixlist  host_pathfixes;
char        *host_ext;
char        *host_setup;
double       host_speed;
int          host_cpus;
int          host_rank;
char        *host_shell;
char        *host_debugger;
char        *host_xterm;

void nodetab_reset()
{
  default_login = "*";
  default_group = "*";
  default_passwd = "*";
  default_pathfixes = 0;
  default_ext = "*";
  default_setup = "*";
  default_speed = 1.0;
  default_cpus = 1;
  default_rank = 0;
  default_shell = arg_shell;
  default_debugger = arg_debugger;
  default_xterm = arg_xterm;
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
  host_debugger = default_debugger;
  host_xterm = default_xterm;
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
  if (ip==0) {
    fprintf(stderr,"ERROR> Cannot obtain IP address of %s\n", host);
    exit(1);
  }
  if (nodetab_size == nodetab_max) return;
  res = (nodetab_host)malloc(sizeof(struct nodetab_host));
  res->name = host;
  res->login = host_login;
  res->passwd = host_passwd;
  res->pathfixes = host_pathfixes;
  res->ext = host_ext;
  res->setup = host_setup;
  res->speed = host_speed;
  res->rank = host_rank;
  res->cpus = host_cpus;
  res->ip = ip;
  res->shell = host_shell;
  res->debugger = host_debugger;
  res->xterm = host_xterm;
  nodetab_add(res);
}

void setup_host_args(char *args)
{
  while(*args != 0) {
    char *b1 = args, *e1 = skipstuff(b1);
    char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
    args = skipblanks(e2);
    if (subeqs(b1,e1,"++login")) host_login = substr(b2,e2);
    else if (subeqs(b1,e1,"++passwd")) host_passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"++shell")) host_shell = substr(b2,e2);
    else if (subeqs(b1,e1,"++debugger")) host_debugger = substr(b2,e2);
    else if (subeqs(b1,e1,"++xterm")) host_xterm = substr(b2,e2);
    else if (subeqs(b1,e1,"++speed")) host_speed = atof(b2);
    else if (subeqs(b1,e1,"++cpus")) host_cpus = atol(b2);
    else if (subeqs(b1,e1,"++pathfix")) {
      char *b3 = skipblanks(e2), *e3 = skipstuff(b3);
      args = skipblanks(e3);
      host_pathfixes=pathfix_append(substr(b2,e2),substr(b3,e3),host_pathfixes);
    } else if (subeqs(b1,e1,"++ext")) host_ext = substr(b2,e2);
    else if (subeqs(b1,e1,"++setup")) host_setup = strdup(b2);
  }
}

void setup_group_args(char *args)
{
  while(*args != 0) {
    char *b1 = args, *e1 = skipstuff(b1);
    char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
    args = skipblanks(e2);
    if (subeqs(b1,e1,"++login")) default_login = substr(b2,e2);
    else if (subeqs(b1,e1,"++passwd")) default_passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"++shell")) default_shell = substr(b2,e2);
    else if (subeqs(b1,e1,"++debugger")) default_debugger = substr(b2,e2);
    else if (subeqs(b1,e1,"++xterm")) default_xterm = substr(b2,e2);
    else if (subeqs(b1,e1,"++speed")) default_speed = atof(b2);
    else if (subeqs(b1,e1,"++cpus")) default_cpus = atol(b2);
    else if (subeqs(b1,e1,"++pathfix")) {
      char *b3 = skipblanks(e2), *e3 = skipstuff(b3);
      args = skipblanks(e3);
      default_pathfixes=pathfix_append(substr(b2,e2),substr(b3,e3),
                                       default_pathfixes);
    } else if (subeqs(b1,e1,"++ext")) default_ext = substr(b2,e2);
    else if (subeqs(b1,e1,"++setup")) default_setup = strdup(b2);
  }
}

void nodetab_init()
{
  FILE *f,*fopen();
  char *nodesfile; nodetab_host node;
  char input_line[MAX_LINE_LENGTH];
  int nread, rightgroup, basicsize, i, remain;
  char *b1, *e1, *b2, *e2, *b3, *e3;
  
  /* Open the NODES_FILE. */
  nodesfile = nodetab_file_find();
  if(arg_verbose)
    fprintf(stderr, "INFO> using %s as nodesfile\n", nodesfile);
  if (!(f = fopen(nodesfile,"r"))) {
    fprintf(stderr,"ERROR> Cannot read %s: %s\n",nodesfile,strerror(errno));
    exit(1);
  }
  
  nodetab_table=(nodetab_host*)malloc(arg_requested_pes*sizeof(nodetab_host));
  nodetab_rank0_table=(int*)malloc(arg_requested_pes*sizeof(int));
  nodetab_rank0_port=(int*)malloc(arg_requested_pes*sizeof(int));
  nodetab_max=arg_requested_pes;
  
  nodetab_reset();
  rightgroup = (strcmp(arg_nodegroup,"main")==0);
  
  while(fgets(input_line,sizeof(input_line)-1,f)!=0) {
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
    if      (subeqs(b1,e1,"login")&&(*b3==0))    default_login = substr(b2,e2);
    else if (subeqs(b1,e1,"passwd")&&(*b3==0))   default_passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"shell")&&(*b3==0))   default_shell = substr(b2,e2);
    else if (subeqs(b1,e1,"debugger")&&(*b3==0)) default_debugger = substr(b2,e2);
    else if (subeqs(b1,e1,"xterm")&&(*b3==0)) default_xterm = substr(b2,e2);
    else if (subeqs(b1,e1,"speed")&&(*b3==0))    default_speed = atof(b2);
    else if (subeqs(b1,e1,"cpus")&&(*b3==0))     default_cpus = atol(b2);
    else if (subeqs(b1,e1,"pathfix")) 
      default_pathfixes=pathfix_append(substr(b2,e2),substr(b3,e3),
                                       default_pathfixes);
    else if (subeqs(b1,e1,"ext")&&(*b3==0))      default_ext = substr(b2,e2);
    else if (subeqs(b1,e1,"setup"))              default_setup = strdup(b2);
    else if (subeqs(b1,e1,"host")) {
      if (rightgroup) {
        host_reset();
        setup_host_args(b3);
	for (host_rank=0; host_rank<host_cpus; host_rank++)
	  nodetab_makehost(substr(b2,e2));
      }
    } else if (subeqs(b1,e1, "group")) {
      nodetab_reset();
      rightgroup = subeqs(b2,e2,arg_nodegroup);
      if(rightgroup) setup_group_args(b3);
    } else {
      fprintf(stderr,"ERROR> unrecognized command in nodesfile:\n");
      fprintf(stderr,"ERROR> %s\n", input_line);
      exit(1);
    }
  }
  basicsize = nodetab_size;
  if (basicsize==0) {
    fprintf(stderr,"ERROR> No hosts in group %s\n", arg_nodegroup);
    exit(1);
  }
  while ((nodetab_size < arg_requested_pes)&&(arg_requested_pes!=MAX_NODES)) {
    nodetab_host h = nodetab_table[nodetab_size-basicsize];
    nodetab_add(h);
  }
  for (i=0; i<nodetab_size; i++) {
    if (nodetab_table[i]->rank == 0)
      remain = nodetab_size - i;
    if (nodetab_table[i]->cpus > remain)
      nodetab_table[i]->cpus = remain;
  }
  fclose(f);
}

nodetab_host nodetab_getinfo(i)
  int i;
{
  if (nodetab_table==0) {
    fprintf(stderr,"ERROR> Node table not initialized.\n");
    exit(1);
  }
  if ((i<0)||(i>=nodetab_size)) {
    fprintf(stderr,"ERROR> No such node %d\n",i);
    exit(1);
  }
  return nodetab_table[i];
}

char        *nodetab_name(i) int i;     { return nodetab_getinfo(i)->name; }
char        *nodetab_login(i) int i;    { return nodetab_getinfo(i)->login; }
char        *nodetab_passwd(i) int i;   { return nodetab_getinfo(i)->passwd; }
char        *nodetab_setup(i) int i;    { return nodetab_getinfo(i)->setup; }
pathfixlist  nodetab_pathfixes(i) int i;{ return nodetab_getinfo(i)->pathfixes; }
char        *nodetab_ext(i) int i;      { return nodetab_getinfo(i)->ext; }
char        *nodetab_shell(i) int i;      { return nodetab_getinfo(i)->shell; }
char        *nodetab_debugger(i) int i;      { return nodetab_getinfo(i)->debugger; }
char        *nodetab_xterm(i) int i;      { return nodetab_getinfo(i)->xterm; }
unsigned int nodetab_ip(i) int i;       { return nodetab_getinfo(i)->ip; }
unsigned int nodetab_cpus(i) int i;     { return nodetab_getinfo(i)->cpus; }
unsigned int nodetab_rank(i) int i;     { return nodetab_getinfo(i)->rank; }
 
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

typedef struct arvar
{
  char *name;
  int lo;
  int hi;
  int tot;
  char **data;
  struct arvar *next;
}
*arvar;

arvar arvar_list;

arvar arvar_find(var)
    char *var;
{
  arvar v = arvar_list;
  while (v && (strcmp(v->name, var))) v=v->next;
  return v;
}

arvar arvar_obtain(var)
    char *var;
{
  arvar v = arvar_find(var);
  if (v==0) { 
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

void arvar_set(var, index, val)
    char *var; int index; char *val;
{
  char *old;
  int i, lo, hi;
  char **data;
  arvar v = arvar_obtain(var);
  data = v->data; lo = v->lo; hi = v->hi;
  if ((index<lo)||(index>=hi)) {
    if (lo>hi) lo=hi=index;
    if (index<lo) lo=index;
    if (index>=hi) hi=index+1;
    data = (char **)calloc(1, (hi-lo)*sizeof(char *));
    if (v->data) {
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

char *arvar_get(var, lo, hi)
    char *var; int lo; int hi;
{
  int len=0; int i;
  arvar v = arvar_find(var);
  char **data, *res;
  if (v==0) return 0;
  data = v->data;
  if (lo<v->lo) return 0;
  if (hi>v->hi) return 0;
  for (i=lo; i<hi; i++) {
    char *val = data[i-(v->lo)];
    if (val==0) return 0;
    len += strlen(val)+1;
  }
  res = (char *)malloc(len);
  len=0;
  for (i=lo; i<hi; i++) {
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
  if (fgets(line, 1023, stdin)==0) { 
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

char *input_extract(nchars)
    int nchars;
{
  char *res = substr(input_buffer, input_buffer+nchars);
  char *tmp = substr(input_buffer+nchars, input_buffer+strlen(input_buffer));
  free(input_buffer);
  input_buffer = tmp;
  return res;
}

char *input_gets()
{
  char *p, *res; int len;
  while(1) {
    p = strchr(input_buffer,'\n');
    if (p) break;
    input_extend();
  }
  len = p-input_buffer;
  res = input_extract(len+1);
  res[len]=0;
  return res;
}

char *input_scanf_chars(fmt)
    char *fmt;
{
  char buf[8192]; int len, pos;
  static int fd; static FILE *file;
  fflush(stdout);
  if (file==0) {
    unlink("/tmp/fnord");
    fd = open("/tmp/fnord",O_RDWR | O_CREAT | O_TRUNC);
    if (fd<0) { 
      fprintf(stderr,"cannot open temp file /tmp/fnord");
      exit(1);
    }
    file = fdopen(fd, "r+");
    unlink("/tmp/fnord");
  }
  while (1) {
    len = strlen(input_buffer);
    rewind(file);
    fwrite(input_buffer, len, 1, file);
    fflush(file);
    rewind(file);
    ftruncate(fd, len);
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

typedef struct req_node
{
  struct req_node *next;
  char request[1];
}
*req_node;

typedef struct req_pipe
{
  int fd[2];
  xstr buffer;
  int pid;
  unsigned int startnode, endnode;
}
*req_pipe;

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

int req_handle_aset(line)
    char *line;
{
  char cmd[100], var[100], val[1000]; int index, ok;
  ok = sscanf(line,"%s %s %d %s",cmd,var,&index,val);
  if (ok!=4) return REQ_FAILED;
  arvar_set(var,index,val);
  return REQ_OK_AWAKEN;
}

void req_write_to_client(int fd, char *buf, int size)
{
  int ok;
  while (size) {
    ok = write(fd, buf, size);
    if (ok<=0) {
      fprintf(stderr,"failed to send data to client");
      exit(1);
    }
    size-=ok; buf+=ok;
  }
}

int req_reply(ip, port, pre, ans)
    int ip; int port; char *pre; char *ans;
{
  int fd = skt_connect(ip, port);
  if (fd<=0) return REQ_FAILED;
  write(fd, pre, strlen(pre));
  write(fd, " ", 1);
  write(fd, ans, strlen(ans));
  close(fd);
  return REQ_OK;
}

int req_handle_aget(line)
    char *line;
{
  char cmd[100], host[1000], pre[1000], var[100]; int ip, port, lo, hi, ok;
  char *res;
  ok = sscanf(line,"%s %s %d %s %d %d",cmd,host,&port,var,&lo,&hi);
  if (ok!=6) return REQ_FAILED;
  ip = lookup_ip(host);
  if (ip==0) return REQ_FAILED;
  res = arvar_get(var,lo,hi+1);
  if (res==0) return REQ_POSTPONE;
  sprintf(pre, "aval %s %d %d", var, lo, hi);
  ok = req_reply(ip,port,pre,res);
  free(res);
  return ok;
}

int req_handle_print(line)
    char *line;
{
  printf("%s",line+6);
  fflush(stdout);
  return REQ_OK;
}

int req_handle_printerr(line)
    char *line;
{
  fprintf(stderr,"%s",line+9);
  fflush(stderr);
  return REQ_OK;
}

int req_handle_ending(line)
    char *line;
{
  int fd;

  req_ending++;
  if (req_ending == nodetab_size){
#if (CMK_DEBUG_MODE || CMK_WEB_MODE)
    printf("End of program\n");
    if(clientIP != 0){
      fd = skt_connect(clientIP, clientKillPort);
      if (fd>0) 
	write(fd, "die\n", strlen("die\n"));
    }
#endif
    exit(0);
  }
  return REQ_OK;
}

int req_handle_abort(line)
    char *line;
{
  fprintf(stderr, "%s", line+6);
  exit(1);
}

int req_handle_scanf(line)
    char *line;
{
  char cmd[100], host[100]; int ip, port, ok;
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

#if CMK_DEBUG_MODE
int req_handle_clientdata(line)
     char *line;
{
  int nread;
  char cmd[100];

  nread = sscanf(line, "%s%d", cmd, &clientKillPort);
  /* printf("Client Kill Port = %d\n", clientKillPort); */
  
  if(arg_initial_bp == 1){
    if(arg_gdbinterface == 1){
      GDBActiveFlag = 1;
    }
  }
  return REQ_OK;
}

/* To be removed Later*/
int req_handle_gdb_freezeport_data(line)
     char *line;
{

  int nread;
  char cmd[100];
  int pe, statePort, freezePort;

  nread = sscanf(line, "%s%d%d", cmd, &pe, &freezePort);
  /* clientFreezePort[pe] = freezePort; */
  return REQ_OK;
}


int req_handle_gdb_command(char *line)
{
  int nread;
  char cmd[100];
  int len, i;

  nread = sscanf(line, "%s%d", cmd, &len);
  command = (char *)malloc(sizeof(char) * (len + 2));
  commandPresent = 1;
  nread = sscanf(line, "%s%d%s%d", cmd, &len, command, &destProcessor);
  command[len] = '\n';
  command[len + 1] = '\0';
  for(i = 0; i < strlen(command); i++){
    if(command[i] == '@'){
      command[i] = ' ';
      break;
    }
  }

  /**** Debugging *****/
  /*printf("GDB command from client for proc %d = %s, %d\n", destProcessor, 
    command, strlen(command));
  fflush(stdout);
  */
  /** ***/

  return REQ_OK;
}
#endif

int req_handle_getinfo(line)
    char *line;
{
  char reply[1024], ans[1024], pre[1024];
  char cmd[100], *res, *origres;
  int port, nread, i;
  int ip;

  nread = sscanf(line, "%s%d%d", cmd, &ip, &port);

#if CMK_DEBUG_MODE
  clientIP = ip;
  clientPort = port;
  serverCCSPorts = (int *)malloc(numGDBs * sizeof(int));
  openStatus = (int *)malloc(numGDBs * sizeof(int));
  for(i = 0; i < numGDBs; i++)
    openStatus[i] = 0;
#endif

  /**** Debugging *****/
  /*
  printf("in req_handle_getinfo : %s\n", line);
  fflush(stdout);
  */


  if(nread != 3) return REQ_FAILED;
  origres = res = arvar_get("addr", 0, nodetab_rank0_size);
  if(res==0) return REQ_POSTPONE;
  strcpy(pre, "info");
  reply[0] = 0;
  sprintf(ans, "%d ", nodetab_rank0_size);
  strcat(reply, ans);
  for(i=0;i<nodetab_rank0_size;i++) {
    sprintf(ans, "%d ", nodetab_cpus(i));
    strcat(reply, ans);
  }
  for(i=0;i<nodetab_rank0_size;i++) {
    sprintf(ans, "%d ", nodetab_ip(i));
    strcat(reply, ans);
  }
  for(i=0;i<nodetab_rank0_size;i++) {
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

#if CMK_DEBUG_MODE
    serverCCSPorts[i] = cport;
#endif

    strcat(reply, ans);
  }
  req_reply(ip, port, pre, reply);
  free(origres);
  return REQ_OK;
}

int req_handle_worker(char *line)
{
  unsigned int workerno, ip, port, i, pe; req_pipe p;
  sscanf(line+7,"%d%d%d",&workerno,&ip,&port);
  p = req_pipes + workerno;
  for (i=p->startnode; i<p->endnode; i++) {
    nodetab_rank0_port[i] = port;
    req_host_IP = ip;
  }
  req_workers_registered++;
  return REQ_OK;
};

int req_handle(line)
    char *line;
{
  char cmd[100];
  sscanf(line,"%s",cmd);
  if      (strcmp(cmd,"aset")==0)       return req_handle_aset(line);
  else if (strcmp(cmd,"aget")==0)       return req_handle_aget(line);
  else if (strcmp(cmd,"scanf")==0)      return req_handle_scanf(line);
  else if (strcmp(cmd,"ping")==0)       return REQ_OK;
  else if (strcmp(cmd,"print")==0)      return req_handle_print(line);
  else if (strcmp(cmd,"printerr")==0)   return req_handle_printerr(line);
  else if (strcmp(cmd,"ending")==0)     return req_handle_ending(line);
  else if (strcmp(cmd,"abort")==0)      return req_handle_abort(line);
  else if (strcmp(cmd,"getinfo")==0)    return req_handle_getinfo(line);
#if CMK_DEBUG_MODE
  else if (strcmp(cmd,"clientdata")==0) return
					  req_handle_clientdata(line);
  else if (strcmp(cmd,"gdb_freezeport_data") == 0)
    return req_handle_gdb_freezeport_data(line);
  else if (strcmp(cmd, "gdb_command") == 0)
    return req_handle_gdb_command(line);
#endif
  else if (strcmp(cmd,"worker")==0)     return req_handle_worker(line);
  else return REQ_FAILED;
}

void req_run_saved()
{
  int ok;
  req_node saved = req_saved;
  req_saved = 0;
  while (saved) {
    req_node t = saved;
    saved = saved->next;
    ok = req_handle(t->request);
    if (ok==REQ_POSTPONE) {
      t->next = req_saved;
      req_saved = t;
    } else if (ok==REQ_FAILED) {
      fprintf(stderr,"bad request: %s\n",t->request);
      free(t);
    } else free(t);
  }
}

void req_write_to_host(int fd, char *buf, int size)
{
  int ok;
  while (size) {
    ok = write(fd, buf, size);
    if ((ok<0)&&(errno==EPIPE)) exit(0);
    if (ok<0) {
      perror("worker writing to host");
      exit(1);
    }
    size-=ok; buf+=ok;
  }
}

void req_serve_client(int workerno)
{
  req_pipe worker;
  xstr buffer;
  int status, nread, len; char *head;
  req_node n;
  int CcsActiveFlag = 0;

  if(workerno == -1){
    buffer = xstr_alloc();
    nread = xstr_read(buffer, CcsClientFd);

    /*** debugging ***/
    /*
    printf("Read the first part : %s, %d\n", xstr_lptr(buffer), nread);
    fflush(stdout);
    */

    /* if necessary, read the rest */
    while(nread != 0){
      nread = xstr_read(buffer, CcsClientFd);
      
      /*** debugging ***/
      /*
      printf("Read the second part: %s\n", xstr_lptr(buffer));
      fflush(stdout);
      */
    }
  }
  else{
    worker = req_pipes+workerno;
    buffer = worker->buffer;
    nread = xstr_read(buffer, worker->fd[0]);
  }

  if (nread<0) {
    fprintf(stderr,"aborting: node terminated abnormally.\n");
    exit(1);
  }
  if((nread==0) && (workerno != -1)) return;
  while (1) {
    head = xstr_lptr(buffer);
    len = strlen(head);
    if ((head+len == xstr_rptr(buffer)) && (workerno != -1)) {
      break;
    }
    status = req_handle(head);
    switch (status) {
    case REQ_OK: 
      if(workerno == -1) CcsActiveFlag = 1;
      break;
    case REQ_FAILED: 
      fprintf(stderr,"bad request: %s: %d\n",head, strlen(head)); 
      abort();
      break;
    case REQ_OK_AWAKEN: req_run_saved(); break;
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
  fd_set rfds; int status, i;
  int clientIP, clientPortNo;
#if CMK_DEBUG_MODE
  struct timeval tmo;

  tmo.tv_sec = 0;
  tmo.tv_usec = 500000;
#endif

  FD_ZERO(&rfds);
  for (i=0; i<req_numworkers; i++)
    FD_SET(req_pipes[i].fd[0], &rfds);
#if CMK_CCS_AVAILABLE
  if (arg_server == 1) FD_SET(myFd, &rfds);
#endif

#if CMK_DEBUG_MODE
  status = select(FD_SETSIZE, &rfds, 0, 0, &tmo);
#else
  status = select(FD_SETSIZE, &rfds, 0, 0, 0);
#endif
  for (i=0; i<req_numworkers; i++)
    if (FD_ISSET(req_pipes[i].fd[0], &rfds))
      req_serve_client(i);

#if CMK_CCS_AVAILABLE
  if (arg_server ==1) {
    if(FD_ISSET(myFd, &rfds)){
      /*      printf("Activity detected on client socket %d\n",myFd); */
      skt_accept(myFd, &clientIP, &clientPortNo, &CcsClientFd);
      /* printf("Accept over\n"); */
      fflush(stdout);
      req_serve_client(-1);
    }
  }
#endif
#if CMK_DEBUG_MODE
  if(arg_server == 1){
    if(GDBActiveFlag == 1){
      /* printf("Entering gdb interface check\n"); */
      fflush(stdout);
      checkForGDBActivity();
      if (openProcessingNeeded == 1){
        /* printf("activity detected on the gdb pipes\n"); */
        fflush(stdout);
        processOnOpen();
      }
      else{
        if((commandPresent == 1) && (commandSent == 0)){
          if(openStatus[destProcessor] == 0){
            int fd;
            char gdbRequest[] = "gdbRequest";
            int size = strlen(gdbRequest) + 1;
            int junkIP = 1234;
            int junkPort = 56789;
            char line[1024];

            commandSent = 1;
            /* printf("Sending request to processor %d\n", destProcessor); */
            fd = skt_connect(nodetab_ip(destProcessor), serverCCSPorts[destProcessor]);
            sprintf(line, "req %d %d %d %d %s\n", destProcessor, size, junkIP, junkPort, "DebugHandler");
            write(fd, line, strlen(line));
            write(fd, gdbRequest, size);
            close(fd);
          }
          else{
            int writeBytes, fd;
            char buffer[MAXREADBYTES];

            writeBytes = write((rsh_prog_gdb[destProcessor].p)->ifd, command,
                               strlen(command));
            if(writeBytes < 0){
              printf("Error executing gdb command : %s : %d : proc %d\n",
                     command, strlen(command), destProcessor);
              exit(1);
            }
            fd = skt_connect(clientIP, clientKillPort);
            writeall(fd, "unfreeze", strlen("unfreeze") + 1);
            close(fd);

	    /*
	      printf("2.sent unfreeze (to Debugger Client)...%s %d %d\n", command,writeBytes, destProcessor);
	    */

            if(strncmp(command, "c", 1) == 0){
              int nread;

              /* Debugging */
              /* printf("CHECK2 : 'continue' pressed\n"); */

              nread = read((rsh_prog_gdb[destProcessor].p)->ofd, buffer, 1000);
              if(nread < 0){
                printf("Error in read...\n");
                exit(1);
              }
            }

            openStatus[destProcessor] = 0;
            free(command);
            commandPresent = 0;
            commandSent = 0;
            command = 0;
            destProcessor = -1;
          }
        }
      }
    }
  }
#endif
}

void req_worker(int workerno)
{
  int numclients;
  int master_ip, master_port, master_fd;
  int client_ip, client_port, client_fd;
  int i, hostfd, status, nread, len, timeout;
  int client[REQ_MAXFD];
  xstr clientbuf[REQ_MAXFD];
  fd_set rfds; xstr buffer;
  struct timeval tv;
  char reply[1024], *head;

  hostfd = req_pipes[workerno].fd[1];
  numclients = req_pipes[workerno].endnode - req_pipes[workerno].startnode;
  for (i=3; i<512; i++) if (i!=hostfd) close(i);
  master_port = 0;
  skt_server(&master_ip, &master_port, &master_fd);
  sprintf(reply,"worker %d %d %d", workerno, master_ip, master_port);
  req_write_to_host(hostfd, reply, strlen(reply)+1);
  timeout = (nodetab_rank0_size*arg_timeout) + 60;
  for (i=0; i<numclients; i++) {
    while (1) {
      if (timeout <= 0) {
	sprintf(reply,"abort timeout waiting for nodes to connect\n");
        req_write_to_host(hostfd, reply, strlen(reply)+1);
	exit(1);
      }
      FD_ZERO(&rfds);
      FD_SET(master_fd, &rfds);
      tv.tv_sec = 1; tv.tv_usec = 0;
      status = select(FD_SETSIZE, &rfds, 0, 0, &tv);
      if (status==1) break;
      if (status<0) { perror("accept"); exit(1); }
      if (status==0) { timeout--; continue; }
    }
    skt_accept(master_fd, &client_ip, &client_port, &client_fd);
    client[i] = client_fd;
    clientbuf[i] = xstr_alloc();
  }
  while (1) {
    /* probe all file descriptors */
    FD_ZERO(&rfds);
    for (i=0; i<numclients; i++)
      FD_SET(client[i], &rfds);
    status = select(FD_SETSIZE, &rfds, 0, 0, 0);
    if (status < 0) {
      perror("worker error of some sort");
      exit(1);
    }
    for(i=0; i<numclients; i++) {
      if (FD_ISSET(client[i], &rfds)) {
	
	/**** Debugging ***/
	/* printf("Activity on %d\n", i);*/


	buffer = clientbuf[i];
	nread = xstr_read(buffer, client[i]);
	if (nread<=0) exit(1);
	while (1) {
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

void req_start_workers()
{
  int i, j, nodenum, nclients;
  
  req_numworkers = (nodetab_rank0_size+REQ_MAXFD-1) / REQ_MAXFD;
  if (req_numworkers > REQ_MAXFD) {
    fprintf(stderr,"Too many PEs, file descriptors exceeded.\n");
    exit(1);
  }
  nodenum = 0;
  for (i=0; i<req_numworkers; i++) {
    nclients = REQ_MAXFD;
    if (nclients > (nodetab_rank0_size-nodenum))
      nclients = (nodetab_rank0_size-nodenum);
    req_pipes[i].startnode = nodenum;
    req_pipes[i].endnode = nodenum + nclients;
    pipe(req_pipes[i].fd);
    req_pipes[i].pid = fork();
    if (req_pipes[i].pid==0) req_worker(i);
    req_pipes[i].buffer = xstr_alloc();
    close(req_pipes[i].fd[1]);
    nodenum += nclients;
  }
}

/****************************************************************************/
/*                                                                          */
/* start_nodes                                                              */
/*                                                                          */
/* this starts all the node programs.  It executes fully in the background. */
/*                                                                          */
/****************************************************************************/

prog rsh_start(nodeno)
    int nodeno;
{
  char *rshargv[6];
  prog rsh;
  
  rshargv[0]=nodetab_shell(nodeno);
  rshargv[1]=nodetab_name(nodeno);
  rshargv[2]="-l";
  rshargv[3]=nodetab_login(nodeno);
  rshargv[4]="exec /bin/csh -f";
  rshargv[5]=0;

  rsh = prog_start(nodetab_shell(nodeno), rshargv, 1);
  if ((rsh==0)&&(errno!=EMFILE)) { perror("ERROR> starting rsh"); exit(1); }
  if (rsh==0)
    {
      fprintf(stderr,"caution: cannot start specified number of rsh's\n");
      fprintf(stderr,"(not enough file descriptors available).\n");
    }
  if (rsh && arg_verbose)
    fprintf(stderr,"INFO> node %d: rsh initiated...\n",nodeno);
  return rsh;
}

int rsh_pump(p, nodeno, rank0no, argv)
    prog p; int nodeno, rank0no; char **argv;
{
  static char buf[1024];
  int len;
  xstr ibuf = p->ibuf;
  int randno = rand();
  /* int randno = CrnRand(); */
  
#if CMK_DEBUG_MODE
  if(!arg_gdbinterface)
#endif
    xstr_printf(ibuf,"echo 'remote responding...'\n");

  xstr_printf(ibuf,"if ( -x ~/.conv-hostrc )   source ~/.conv-hostrc\n");
  if (arg_display)
    xstr_printf(ibuf,"setenv DISPLAY %s\n",arg_display);
  xstr_printf(ibuf,"setenv NETSTART '%d %d %d %d %d %d %d %d %d'\n",
	      nodetab_rank0_size, rank0no,
	      nodeno, nodetab_cpus(nodeno), nodetab_size,
	      nodetab_ip(nodeno), req_host_IP,
	      nodetab_rank0_port[rank0no], (getpid()&0x7FFF));
  prog_flush(p);
  
  /* find the node-program */
  arg_nodeprog_r = pathfix(arg_nodeprog_a, nodetab_pathfixes(nodeno));
  
  /* find the current directory, relative version */
  arg_currdir_r = pathfix(arg_currdir_a, nodetab_pathfixes(nodeno));

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    xstr_printf(ibuf,"foreach dir ($path)\n");
    xstr_printf(ibuf,"  if (-e $dir/%s) setenv F_XTERM $dir/%s\n", 
                     nodetab_xterm(nodeno), nodetab_xterm(nodeno));
    xstr_printf(ibuf,"  if (-e $dir/xrdb) setenv F_XRDB $dir/xrdb\n");
    xstr_printf(ibuf,"end\n");
    xstr_printf(ibuf,"if ($?F_XTERM == 0) then\n");
    xstr_printf(ibuf,"   echo '%s not in path --- set your path in your cshrc.'\n", nodetab_xterm(nodeno));
    xstr_printf(ibuf,"   kill -9 $$\n");
    xstr_printf(ibuf,"endif\n");
    xstr_printf(ibuf,"if ($?F_XRDB == 0) then\n");
    xstr_printf(ibuf,"   echo 'xrdb not in path - set your path in your cshrc.'\n");
    xstr_printf(ibuf,"   kill -9 $$\n");
    xstr_printf(ibuf,"endif\n");
    if(arg_verbose) xstr_printf(ibuf,"echo 'using xterm ' $F_XTERM\n");
    prog_flush(p);
  }

  if (arg_debug || arg_debug_no_pause 
#if CMK_DEBUG_MODE
      || arg_gdbinterface
#endif
     )
  	{
	  if ( strcmp(nodetab_debugger(nodeno), "gdb") == 0 ) {
            xstr_printf(ibuf,"foreach dir ($path)\n");
            xstr_printf(ibuf,"  if (-e $dir/gdb) setenv F_GDB $dir/gdb\n");
            xstr_printf(ibuf,"end\n");
            xstr_printf(ibuf,"if ($?F_GDB == 0) then\n");
            xstr_printf(ibuf,"   echo 'gdb not in path - set your path in your cshrc.'\n");
            xstr_printf(ibuf,"   kill -9 $$\n");
            xstr_printf(ibuf,"endif\n");
            prog_flush(p);
	  } else if ( strcmp(nodetab_debugger(nodeno), "dbx") == 0 ) {
            xstr_printf(ibuf,"foreach dir ($path)\n");
            xstr_printf(ibuf,"  if (-e $dir/dbx) setenv F_DBX $dir/dbx\n");
            xstr_printf(ibuf,"end\n");
            xstr_printf(ibuf,"if ($?F_DBX == 0) then\n");
            xstr_printf(ibuf,"   echo 'dbx not in path - set your path in your cshrc.'\n");
            xstr_printf(ibuf,"   kill -9 $$\n");
            xstr_printf(ibuf,"endif\n");
            prog_flush(p);
	  } else {
	    fprintf(stderr, "Unknown Debugger: %s.\n exiting.\n", 
	      nodetab_debugger(nodeno));
	    exit(1);
	  }
       }

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    xstr_printf(ibuf,"xrdb -query > /dev/null\n");
    xstr_printf(ibuf,"if ($status != 0) then\n");
    xstr_printf(ibuf,"  echo 'Cannot contact X Server.  You probably'\n");
    xstr_printf(ibuf,"  echo 'need to run xhost to authorize connections.'\n");
    xstr_printf(ibuf,"  echo '(See manual for xhost for security issues)'\n");
    xstr_printf(ibuf,"  kill -9 $$\n");
    xstr_printf(ibuf,"endif\n");
    prog_flush(p);
  }
  
  xstr_printf(ibuf,"if (! -x %s) then\n",arg_nodeprog_r);
  xstr_printf(ibuf,"  echo 'Cannot locate this node-program:'\n");
  xstr_printf(ibuf,"  echo '%s'\n",arg_nodeprog_r);
  xstr_printf(ibuf,"  kill -9 $$\n");
  xstr_printf(ibuf,"endif\n");
  
  xstr_printf(ibuf,"cd %s\n",arg_currdir_r);
  xstr_printf(ibuf,"if ($status == 1) then\n");
  xstr_printf(ibuf,"  echo 'Cannot propagate this current directory:'\n"); 
  xstr_printf(ibuf,"  echo '%s'\n",arg_currdir_r);
  xstr_printf(ibuf,"  kill -9 $$\n");
  xstr_printf(ibuf,"endif\n");
  
  if (strcmp(nodetab_setup(nodeno),"*")) {
    xstr_printf(ibuf,"cd .\n");
    xstr_printf(ibuf,"%s\n",nodetab_setup(nodeno));
    xstr_printf(ibuf,"if ($status == 1) then\n");
    xstr_printf(ibuf,"  echo 'this initialization command failed:'\n");
    xstr_printf(ibuf,"  echo '\"%s\"'\n",nodetab_setup(nodeno));
    xstr_printf(ibuf,"  echo 'edit your nodes file to fix it.'\n");
    xstr_printf(ibuf,"  kill -9 $$\n");
    xstr_printf(ibuf,"endif\n");
  }
  
  if (arg_debug || arg_debug_no_pause 
#if CMK_DEBUG_MODE
      || arg_gdbinterface
#endif
     ) {
	 if ( strcmp(nodetab_debugger(nodeno), "gdb") == 0 ) {
           xstr_printf(ibuf,"cat > /tmp/gdb%08x << END_OF_SCRIPT\n",randno);
           xstr_printf(ibuf,"shell rm -f /tmp/gdb%08x\n",randno);
           xstr_printf(ibuf,"handle SIGPIPE nostop noprint\n");
           xstr_printf(ibuf,"handle SIGWINCH nostop noprint\n");
           xstr_printf(ibuf,"handle SIGWAITING nostop noprint\n");
           xstr_printf(ibuf,"set args");
           while (*argv) { xstr_printf(ibuf," %s",*argv); argv++; }
           xstr_printf(ibuf,"\n");
           if (arg_debug_no_pause) xstr_printf(ibuf,"run\n");
           xstr_printf(ibuf,"END_OF_SCRIPT\n");
           if( arg_debug || arg_debug_no_pause){
             xstr_printf(ibuf,"$F_XTERM");
             xstr_printf(ibuf," -title 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
             xstr_printf(ibuf," -e $F_GDB %s -x /tmp/gdb%08x",arg_nodeprog_r,randno);
             xstr_printf(ibuf," < /dev/null >& /dev/null &");
             xstr_printf(ibuf,"\n");
           }
#if CMK_DEBUG_MODE
           else if( arg_gdbinterface){
             xstr_printf(ibuf,"$F_GDB -silent %s -x /tmp/gdb%08x", arg_nodeprog_r,randno);
             xstr_printf(ibuf,"\n");
           }
#endif
        } else if ( strcmp(nodetab_debugger(nodeno), "dbx") == 0 ) {
          xstr_printf(ibuf,"cat > /tmp/dbx%08x << END_OF_SCRIPT\n",randno);
          xstr_printf(ibuf,"sh rm -f /tmp/dbx%08x\n",randno);
          xstr_printf(ibuf,"dbxenv suppress_startup_message 5.0\n");
          xstr_printf(ibuf,"ignore SIGPOLL\n");
          xstr_printf(ibuf,"ignore SIGPIPE\n");
          xstr_printf(ibuf,"ignore SIGWINCH\n");
          xstr_printf(ibuf,"ignore SIGWAITING\n");
          xstr_printf(ibuf,"END_OF_SCRIPT\n");
          if( arg_debug || arg_debug_no_pause){
            xstr_printf(ibuf,"$F_XTERM");
            xstr_printf(ibuf," -title 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
            xstr_printf(ibuf," -e $F_DBX %s ",arg_debug_no_pause?"-r":"");
	    if(arg_debug) {
              xstr_printf(ibuf,"-c \'runargs ");
              while (*argv) { xstr_printf(ibuf,"%s ",*argv); argv++; }
              xstr_printf(ibuf,"\' ");
	    }
	    xstr_printf(ibuf, "-s/tmp/dbx%08x %s",randno,arg_nodeprog_r);
	    if(arg_debug_no_pause) {
              while (*argv) { xstr_printf(ibuf," %s",*argv); argv++; }
	    }
            xstr_printf(ibuf," < /dev/null >& /dev/null &");
            xstr_printf(ibuf,"\n");
          }
	} else { 
	  fprintf(stderr, "Unknown debugger: %s.\n Exiting.\n", 
	    nodetab_debugger(nodeno));
	  exit(1);
	}
  } else if (arg_in_xterm) {
    if(arg_verbose) {
      fprintf(stderr, "INFO> node %d: xterm is %s\n", 
              nodeno, nodetab_xterm(nodeno));
    }
    xstr_printf(ibuf,"cat > /tmp/inx%08x << END_OF_SCRIPT\n", randno);
    xstr_printf(ibuf,"#!/bin/sh\n");
    xstr_printf(ibuf,"rm -f /tmp/inx%08x\n",randno);
    xstr_printf(ibuf,"%s", arg_nodeprog_r);
    while (*argv) { xstr_printf(ibuf," %s",*argv); argv++; }
    xstr_printf(ibuf,"\n");
    xstr_printf(ibuf,"echo 'program exited with code '\\$?\n");
    xstr_printf(ibuf,"read eoln\n");
    xstr_printf(ibuf,"END_OF_SCRIPT\n");
    xstr_printf(ibuf,"chmod 700 /tmp/inx%08x\n", randno);
    xstr_printf(ibuf,"$F_XTERM");
    xstr_printf(ibuf," -title 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
    xstr_printf(ibuf," -sl 5000");
    xstr_printf(ibuf," -e /tmp/inx%08x", randno);
    xstr_printf(ibuf," < /dev/null >& /dev/null &");
    xstr_printf(ibuf,"\n");
  } else {
    xstr_printf(ibuf,"%s",arg_nodeprog_r);
    while (*argv) { xstr_printf(ibuf," %s",*argv); argv++; }
    xstr_printf(ibuf," < /dev/null >& /dev/null &");
    xstr_printf(ibuf,"\n");
  }
  
#if CMK_DEBUG_MODE
  if(!arg_gdbinterface){
#endif
    xstr_printf(ibuf,"echo 'rsh phase successful.'\n");
    xstr_printf(ibuf,"kill -9 $$\n");
#if CMK_DEBUG_MODE
  }
#endif
  prog_flush(p);
  
}


int start_nodes()
{
  prog        rsh_prog[200];
  int         rsh_node[200];
  int         rsh_nstarted;
  int         rsh_nfinished;
  int         rsh_freeslot;
  int         rsh_maxsim;
  fd_set rfds; int i; prog p; int pe;

  /* Return immediately.  That way, the nodes which are starting */
  /* Will be able to establish communication with the host */
  /* if (fork()) return; */

  /* Obtain the values from the command line options */
  rsh_maxsim = arg_maxrsh;
  if (rsh_maxsim < 1) rsh_maxsim=1;
  if (rsh_maxsim > nodetab_rank0_size) rsh_maxsim=nodetab_rank0_size;
  if (rsh_maxsim > 200) rsh_maxsim=200;

  /* start initial group of rsh's */
  for (i=0; i<rsh_maxsim; i++) {
    pe = nodetab_rank0_table[i];
    p = rsh_start(pe);
    if (p==0) { rsh_maxsim=i; break; }
    rsh_pump(p, pe, i, arg_argv);
    rsh_prog[i] = p;
    rsh_node[i] = pe;
#if CMK_DEBUG_MODE
    rsh_prog_gdb[numGDBs].p = p;
    rsh_prog_gdb[numGDBs].open = 0;
    rsh_prog_gdb[numGDBs].pe = numGDBs;
    numGDBs++;
#endif
  }
  if (rsh_maxsim==0) { perror("ERROR> starting rsh"); exit(1); }
  rsh_nstarted = rsh_maxsim;
  rsh_nfinished = 0;

  while (rsh_nfinished < nodetab_rank0_size) {
    int maxfd=0; int ok;
    FD_ZERO(&rfds);
    for (i=0; i<rsh_maxsim; i++) {
      p = rsh_prog[i];
      if (p==0) continue;
      FD_SET(p->ofd, &rfds);
      if (p->ofd > maxfd) maxfd = p->ofd;
    }
    do ok = select(maxfd+1, &rfds, NULL, NULL, NULL);
    while ((ok<0)&&(errno==EINTR));
    if (ok<0) { perror("ERROR> select"); exit(1); }
    do ok = waitpid((pid_t)(-1), NULL, WNOHANG);
    while (ok>0);
    for (i=0; i<rsh_maxsim; i++) {
      char *line = 0;
      char buffer[1000];
      int nread, done = 0;
      p = rsh_prog[i];
      if (p==0) continue;
      if (!FD_ISSET(p->ofd, &rfds)) continue;
      do nread = read(p->ofd, buffer, 1000);
      while ((nread<0)&&(errno==EINTR));
      if (nread<0) { perror("ERROR> read"); exit(1); }
      if (nread==0) {
        fprintf(stderr,"ERROR> node %d: rsh phase failed.\n",rsh_node[i]);
        exit(1);
      }
      xstr_write(p->obuf, buffer, nread);
      while (1) {
#if CMK_DEBUG_MODE
        if(!arg_gdbinterface){
#endif
        line = xstr_gets(buffer, 999, p->obuf);
#if CMK_DEBUG_MODE
        }
        else{
          line = buffer;
        }
#endif
        if (line==0) break;
#if CMK_DEBUG_MODE
        if(!arg_gdbinterface) {
#endif
          if (strncmp(line,"[1] ",4)==0) continue;
          if (arg_verbose ||
              (strcmp(line,"rsh phase successful.")
               &&strcmp(line,"remote responding...")))
            fprintf(stderr,"INFO> node %d: %s\n",rsh_node[i],line);
          if (strcmp(line,"rsh phase successful.")==0) { done=1; break; }
#if CMK_DEBUG_MODE
        } else{
          if (strncmp(line, "(gdb)", 5) == 0) {
            xstr_printf(p->ibuf, "b dummyF\n");

	    if(arg_initial_bp == 1)
	      xstr_printf(p->ibuf, "b CsdScheduler\n");

            prog_flush(p);
            /* Flush out the initial output */
            nread = readUntilString(p->ofd, buffer, "(gdb) ");
            if(nread < 0){
              printf("Error in read...\n");
              exit(1);
            }
            else{
              /* printf("read stuff = %s, %d\n", buffer, nread); */
            }
            xstr_printf(p->ibuf, "run\n");
            prog_flush(p);
            /* Flush the output again */
            nread = read(p->ofd, buffer, 1000);
            if(nread < 0){
              printf("Error in read...\n");
              exit(1);
            }
            else{
              /* printf("read stuff = %s, %d\n", buffer, nread); */
            }

            done=1;
            break;
          }
        }
#endif
      }
      if (!done) continue;
      rsh_nfinished++;

      rsh_prog[i] = 0;
      if (rsh_nstarted==nodetab_rank0_size) break;
      pe = nodetab_rank0_table[rsh_nstarted];
      p = rsh_start(pe);
      if (p==0) { perror("ERROR> starting rsh"); exit(1); }
      rsh_pump(p, pe, rsh_nstarted, arg_argv);
      rsh_prog[i] = p;
      rsh_node[i] = pe;
#if CMK_DEBUG_MODE
      rsh_prog_gdb[numGDBs].p = p;
      rsh_prog_gdb[numGDBs].open = 0;
      rsh_prog_gdb[numGDBs].pe = numGDBs;
      numGDBs++;
#endif
      rsh_nstarted++;
    }
  }
}

/****************************************************************************
 *
 *  The Main Program
 *
 ****************************************************************************/

main(argc, argv)
    int argc; char **argv;
{
  unsigned int myIP, myPortNo;

  srand(time(0));
  /* CrnSrand((int) time(0)); */
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

  /* Start the worker processes */
  req_start_workers();
  /* Wait until the workers have registered */
  while (req_workers_registered < req_numworkers) req_poll();
  /* Initialize the IO module */
  input_init();
  /* start the node processes */
  start_nodes();


#if CMK_DEBUG_MODE
  if(arg_initial_bp == 0){
    if(arg_gdbinterface == 1){
      GDBActiveFlag = 1;
    }
  }
#endif

  /* enter request-service mode */
  while (1) req_poll();
}

