
#include "conv-mach.h"

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

/****************************************************************************
 *
 * Death-notification
 *
 ****************************************************************************/

int *notify_ip;
int *notify_port;
int  notify_count;
int  notify_max;

void notify_die(ip, port)
    int ip; int port;
{
  if (notify_count==notify_max) {
    notify_max  = (notify_max*2)+100;
    if (notify_ip  ==0) notify_ip   = (int *)malloc(sizeof(int));
    if (notify_port==0) notify_port = (int *)malloc(sizeof(int));
    notify_ip   = (int *)realloc(notify_ip,   notify_max*sizeof(int));
    notify_port = (int *)realloc(notify_port, notify_max*sizeof(int));
  }
  notify_ip[notify_count] = ip;
  notify_port[notify_count] = port;
  notify_count++;
}

void notify_die_doit(msg)
    char *msg;
{
  int skt_connect();
  char buffer[1024];
  int i, fd;
  sprintf(buffer,"die %s\n",msg);
  for (i=0; i<notify_count; i++) {
    int ip = notify_ip[i];
    int port = notify_port[i];
    fd = skt_connect(ip, port);
    if (fd>=0) { write(fd, buffer, strlen(buffer)); close(fd); }
  }
  fprintf(stderr,"aborting: %s\n",msg);
  exit(1);
}

void notify_abort()
{
  notify_die_doit("");
}

void notify_die_segv()
{
  notify_die_doit("host: seg fault.");
}

void notify_die_intr()
{
  notify_die_doit("host: interrupted.");
}

void notify_die_init()
{
  signal(SIGSEGV, notify_die_segv);
  signal(SIGBUS,  notify_die_segv);
  signal(SIGILL,  notify_die_segv);
  signal(SIGABRT, notify_die_segv);
  signal(SIGFPE,  notify_die_segv);

#ifdef SIGSYS
  signal(SIGSYS,  notify_die_segv);
#endif
  
  signal(SIGPIPE, notify_die_segv);
  signal(SIGURG,  notify_die_segv);

  signal(SIGTERM, notify_die_intr);
  signal(SIGQUIT, notify_die_intr);
  signal(SIGINT,  notify_die_intr);
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
  if (fd < 0) { perror("socket"); exit(1); }
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  ok = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  if (ok < 0) { perror("bind"); exit(1); }
  ok = listen(fd,5);
  if (ok < 0) { perror("listen"); exit(1); }
  len = sizeof(addr);
  ok = getsockname(fd, (struct sockaddr *)&addr, &len);
  if (ok < 0) { perror("getsockname"); exit(1); }

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
#if CMK_FIX_HP_CONNECT_BUG
  jsleep(0,50000);
#endif
  if ((fd<0)&&(errno==EINTR)) goto acc;
  if ((fd<0)&&(errno==EMFILE)) goto acc;
  if (fd<0) { perror("accept"); notify_abort(); }
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

zap_newline(s)
    char *s;
{
  char *p;
  p = s + strlen(s)-1;
  if (*p == '\n') *p = '\0';
}

char *substr(lo, hi)
    char *lo; char *hi;
{
  int len = hi-lo;
  char *res = (char *)malloc(1+len);
  memcpy(res, lo, len);
  res[len]=0;
  return res;
}

int subeqs(lo, hi, str)
     char *lo; char *hi; char *str;
{
  int len = strlen(str);
  if (hi-lo != len) return 0;
  if (memcmp(lo, str, len)) return 0;
  return 1;
}

/* advance pointer over blank characters */
char *skipblanks(p)
    char *p;
{
  while ((*p==' ')||(*p=='\t')) p++;
  return p;
}

/* advance pointer over nonblank characters */
char *skipstuff(p)
    char *p;
{
  while ((*p)&&(*p!=' ')&&(*p!='\t')) p++;
  return p;
}

char *text_ip(ip)
    unsigned int ip;
{
  static char buffer[100];
  sprintf(buffer,"%d.%d.%d.%d",
	  (ip>>24)&0xFF,
	  (ip>>16)&0xFF,
	  (ip>>8)&0xFF,
	  (ip>>0)&0xFF);
  return buffer;
}

int readhex(f, len)
    FILE *f; int len;
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
  if (self==0) { perror("getpwuid"); exit(1); }
  return self->pw_name;
} 

unsigned int lookup_ip(name)
    char *name;
{
  struct hostent *h;
  unsigned int ip1,ip2,ip3,ip4; int nread;
  nread = sscanf(name,"%d.%d.%d.%d",&ip1,&ip2,&ip3,&ip4);
  if (nread==4) return (ip1<<24)|(ip2<<16)|(ip3<<8)|ip4;
  h = gethostbyname(name);
  if (h==0) return 0;
  return htonl(*((int *)(h->h_addr_list[0])));
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
  printf("No such program parameter %s\n",lname);
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
  printf("\n");
  printf("parameters recognized are:\n");
  printf("\n");
  for (def=ppdefs; def; def=def->next)
    {
      len = strlen(def->lname);
      printf("  %c%c%s ",pparam_optc,pparam_optc,def->lname);
      for(i=0; i<maxname-len; i++) printf(" ");
      len = strlen(def->doc);
      printf("  %s ",def->doc);
      for(i=0; i<maxdoc-len; i++) printf(" ");
      printf("[%s]\n",pparam_getdef(def));
    }
  printf("\n");
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
 * PATH                                                                      
 *                                                                           
 * path_simplify(P)                                                          
 *                                                                           
 *  - P is a pointer to a buffer containing a path.  All ".." and "."        
 *    components of the path are expanded out.                               
 *                                                                           
 * path_concat(P, rel)                                                       
 *                                                                           
 *  - P is a pointer to a buffer containing a path, rel is another path      
 *    relative to P.  The logical concatenation of the two paths are         
 *    stored in the buffer.                                                  
 *                                                                           
 * path_absolute(P)                                                          
 *                                                                           
 *  - If P is a relative path, it is assumed to be relative to the current   
 *    directory, and it is thereby converted into an absolute path.          
 *                                                                           
 * path_search(name, searchpath)                                             
 *                                                                           
 *  - name is a pointer to a buffer containing a program name or program     
 *    path, and searchpath is a string containing a list of directories      
 *   (as might be returned by getenv("PATH")).  The program's executable     
 *    is located and its absolute, readlinked path is stored in the name     
 *    buffer.                                                                
 *                                                                           
 * int path_isprefix(path1, path2)
 * 
 *  -  Routine to check whether path1 is a prefix of path2.  Returns 0 if
 *     not a prefix, or number of chars to chop off 'ipath' if it is a
 *     prefix.  path1 must be a path without a trailing slash.
 *
 *****************************************************************************/

static char *path_segs[100];
static int   path_nsegs;

static void path_segs_free()
{
  while (path_nsegs) free(path_segs[--path_nsegs]);
}

static void path_dissect(path)
    char *path;
{
  char buf[1000];
  int len=0;
  while (1)
    {
      if ((*path=='/')||(*path==0))
	{
	  buf[len]=0;
	  path_segs[path_nsegs++] = strdup(buf);
	  len=0;
	}
      else buf[len++] = *path;
      if (*path==0) break;
      path++;
    }
}

static void path_reduce()
{
  int src, dst;
  src = 0; dst = 0;
  for (src=0; src<path_nsegs; src++)
    {
      char *t;
      if ((strcmp(path_segs[src],"")==0)&&(src!=0)) continue;
      if (strcmp(path_segs[src],".")==0) continue;
      if (strcmp(path_segs[src],"..")==0) { if (dst) dst--; continue; }
      t = path_segs[dst]; path_segs[dst]=path_segs[src]; path_segs[src]=t;
      dst ++;
    }
  while (src>dst) free(path_segs[--src]);
  path_nsegs = dst;
}

static void path_reconstitute(buff)
    char *buff;
{
  int i;
  for (i=0; i<path_nsegs; i++)
    {
      strcpy(buff, path_segs[i]);
      buff+=strlen(buff);
      *buff++ = '/';
    }
  *(--buff)=0;
}

void path_simplify(path)
    char *path;
{
  path_nsegs = 0;
  path_dissect(path);
  path_reduce();
  path_reconstitute(path);
  path_segs_free();
}

void path_concat(base, rel)
    char *base; char *rel;
{
  path_nsegs = 0;
  if (rel[0]!='/') path_dissect(base);
  path_dissect(rel);
  path_reduce();
  path_reconstitute(base);
  path_segs_free();
}

void path_absolute(path)
    char *path;
{
  char buff[1024];
  if (path[0]=='/') return;
  getcwd(buff, 1023);
  path_concat(buff, path);
  strcpy(path, buff);
}

int path_exists(path)
    char *path;
{
  struct stat s;
  int ok = stat(path, &s);
  if (ok>=0) return 1;
  return 0;
}

int path_executable(path)
    char *path;
{
  struct stat s;
  int ok = stat(path, &s);
  if (ok<0) return 0;
  if (!S_ISREG(s.st_mode)) return 0;
  if((s.st_mode&S_IXOTH)&&(s.st_mode&S_IROTH)) return 1;
  if((s.st_mode&S_IXGRP)&&(s.st_mode&S_IRGRP)&&(s.st_gid==getgid()))return 1;
  if((s.st_mode&S_IXUSR)&&(s.st_mode&S_IRUSR)&&(s.st_uid==getuid()))return 1;
  return 0;
}

int path_nonzero(path)
    char *path;
{
  struct stat s;
  int ok = stat(path, &s);
  if (ok<0) return 0;
  if (!S_ISREG(s.st_mode)) return 0;
  if (s.st_size==0) return 0;
  return 1;
}

int path_search(prog, path)
    char *prog; char *path;
{
  char *end;
  if (strchr(prog,'/'))
    {
      path_absolute(prog);
      if (path_exists(prog)) return 0;
      prog[0]=0; return -1;
    }
  if ((path)&&(*path)) while (1)
    {
      char buff[1024];
      int len;
      end = strchr(path, ':');
      if (end==0) { end=path+strlen(path); }
      len = (end - path);
      memcpy(buff, path, len);
      buff[len]=0;
      path_concat(buff, prog);
      path_absolute(buff);
      if (path_executable(buff)) { strcpy(prog, buff); return 0; }
      if (*end==0) break;
      path=end+1;
    }
  prog[0]=0; errno=ENOENT; return -1;
}

int path_isprefix(ipre, ipath)
    char *ipre; char *ipath;
{
  char pre[MAXPATHLEN];
  char path[MAXPATHLEN];
  struct stat preinfo;
  struct stat pathinfo;
  int ok, prelen; char *p;
  strcpy(pre, ipre);
  strcpy(path, ipath);
  prelen = strlen(pre);
  if (prelen==0) return 0;
  if (pre[prelen-1]=='/') return 0;
  if (strncmp(pre, path, prelen)==0) return prelen;
  ok = stat(pre, &preinfo);
  if (ok<0) return 0;
  p=path;
  while (1) {
    int ch = *p;
    if ((ch=='/')||(ch==0)) {
      *p = 0;
      ok = stat(path, &pathinfo);
      if (ok<0) return 0;
      if ((pathinfo.st_ino == preinfo.st_ino)&&
          (pathinfo.st_dev == preinfo.st_dev)) return p-path;
      *p = ch;
    }
    if (ch==0) break;
    p++;
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
}

void xstr_rshrink(l, nbytes)
    xstr l; int nbytes;
{
  if (l->rptr - l->lptr < nbytes) { l->rptr=l->lptr; return; }
  l->rptr -= nbytes;
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
	{ fprintf(stderr,"error: write returned 0???\n"); exit(1); }
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
      execv(p, argv);
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

arg_init(argc, argv)
    int argc; char **argv;
{
  static char buf[1024]; int len, i;
  
  pparam_defint ("p"             ,  MAX_NODES);
  pparam_defflag("debug"             );
  pparam_defflag("debug-no-pause"    );
  pparam_defflag("in-xterm"          );
  pparam_defint ("maxrsh"        ,  5);
  pparam_defstr ("nodelist"      ,  0);
  pparam_defstr ("nodegroup"     ,  "main");
  
  pparam_doc("p",             "number of processes to create");
  pparam_doc("in-xterm",      "Run each node in an xterm window");
  pparam_doc("debug",         "Run each node under gdb in an xterm window");
  pparam_doc("debug-no-pause","Like debug, except doesn't pause at beginning");
  pparam_doc("maxrsh",        "Maximum number of rsh's to run at a time");
  pparam_doc("nodelist",      "file containing list of nodes");
  pparam_doc("nodegroup",     "which group of nodes to use");

  if (pparam_parsecmd('+', argv) < 0) {
    fprintf(stderr,"syntax: %s\n",pparam_error);
    pparam_printdocs();
    exit(1);
  }
  arg_argv = argv+2;
  arg_argc = pparam_countargs(argv+2);

  arg_requested_pes  = pparam_getint("p");
  arg_in_xterm       = pparam_getflag("in-xterm");
  arg_debug          = pparam_getflag("debug");
  arg_debug_no_pause = pparam_getflag("debug-no-pause");
  arg_maxrsh         = pparam_getint("maxrsh");
  arg_nodelist       = pparam_getstr("nodelist");
  arg_nodegroup      = pparam_getstr("nodegroup");
  
  /* Find the current value of the DISPLAY variable */
  arg_display = getenv_display();
  if ((arg_debug || arg_debug_no_pause || arg_in_xterm) && (arg_display==0)) {
    fprintf(stderr,"DISPLAY must be set to use debugging mode\n");
    exit(1);
  }

  /* find the node-program, absolute version */
  if (argc<2) {
    fprintf(stderr,"You must specify a node-program.\n");
    exit(1);
  }
  strcpy(buf, argv[1]);
  path_search(buf, getenv("PATH"));
  if (buf[0]==0)
    { fprintf(stderr,"No such program %s\n",argv[1]); exit(1); }
  arg_nodeprog_a = strdup(buf);

  strcpy(buf, RSH_CMD);
  path_search(buf, getenv("PATH"));
  if (buf[0]==0)
    { fprintf(stderr,"Cannot find '%s' in path.\n", RSH_CMD); exit(1); }
  arg_rshprog = strdup(buf);

  /* find the current directory, absolute version */
  getcwd(buf, 1023);
  arg_currdir_a = strdup(buf);

  arg_mylogin = mylogin();
  arg_myhome = (char *) malloc(MAXPATHLEN);
  strcpy(arg_myhome, getenv("HOME"));
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
    if (path_nonzero(path)) return strdup(path);
    fprintf(stderr,"No such nodelist file %s\n",path);
    exit(1);
  }
  /* Find a nodes-file as specified by getenv("NODELIST") */
  if (getenv("NODELIST")) {
    char *path = getenv("NODELIST");        
    if (path && path_nonzero(path)) return strdup(path);
    fprintf(stderr,"Cannot find nodelist file %s\n",path);
    exit(1);
  }
  /* Find a nodes-file by looking under 'nodelist' in the current directory */
  if (path_nonzero("./nodelist")) return strdup("./nodelist");
  if (getenv("HOME")) {
    char buffer[MAXPATHLEN];
    strcpy(buffer,getenv("HOME"));
    path_concat(buffer,".nodelist");
    if (path_nonzero(buffer)) return strdup(buffer);
  }
  fprintf(stderr,"Cannot find a nodes file.\n");
  exit(1);
}


typedef struct nodetab_host {
  char *name;
  char *login;
  char *passwd;
  char *home;
  char *ext;
  char *setup;
  int   cpus;
  int   rank;
  double speed;
  unsigned int ip;
} *nodetab_host;

char    *default_login = "*";
char    *default_group = "*";
char    *default_passwd = "*";
char    *default_home = "*";
char    *default_ext = "*";
char    *default_setup = "*";
double   default_speed = 1.0;
int      default_cpus = 1;
int      default_rank = 0;

nodetab_host *nodetab_table;
int           nodetab_max;
int           nodetab_size;
int          *nodetab_rank0_table;
int           nodetab_rank0_size;

void nodetab_add(nodetab_host res)
{
  if (res->rank == 0)
    nodetab_rank0_table[nodetab_rank0_size++] = nodetab_size;
  nodetab_table[nodetab_size++] = res;
}

void nodetab_makehost(char *host)
{
  nodetab_host res;
  int ip;
  ip = lookup_ip(host);
  if (ip==0) {
    fprintf(stderr,"Cannot obtain IP address of %s\n", host);
    exit(0);
  }
  if (nodetab_size == nodetab_max) return;
  res = (nodetab_host)malloc(sizeof(struct nodetab_host));
  res->name = host;
  res->login = default_login;
  res->passwd = default_passwd;
  res->home = default_home;
  res->ext = default_ext;
  res->setup = default_setup;
  res->speed = default_speed;
  res->rank = default_rank;
  res->cpus = default_cpus;
  res->ip = ip;
  nodetab_add(res);
}

void nodetab_init()
{
  FILE *f,*fopen();
  char *nodesfile; nodetab_host node;
  char input_line[MAX_LINE_LENGTH];
  int nread, rightgroup, basicsize, i;
  char *b1, *e1, *b2, *e2, *b3, *e3;
  
  /* Open the NODES_FILE. */
  nodesfile = nodetab_file_find();
  if (!(f = fopen(nodesfile,"r"))) {
    fprintf(stderr,"Cannot read %s: %s\n",nodesfile,strerror(errno));
    exit(1);
  }
  
  nodetab_table=(nodetab_host*)malloc(arg_requested_pes*sizeof(nodetab_host));
  nodetab_rank0_table=(int*)malloc(arg_requested_pes*sizeof(int));
  nodetab_max=arg_requested_pes;
  
  rightgroup = (strcmp(arg_nodegroup,"main")==0);
  
  while(fgets(input_line,sizeof(input_line)-1,f)!=0) {
    if (nodetab_size == arg_requested_pes) break;
    if (input_line[0]=='#') continue;
    zap_newline(input_line);
    b1 = skipblanks(input_line);
    e1 = skipstuff(b1); b2 = skipblanks(e1); 
    e2 = skipstuff(b2); b3 = skipblanks(e2);
    if (*b1==0) continue;
    if (strcmp(default_login, "*")==0) default_login = arg_mylogin;
    if (strcmp(default_home, "*")==0) default_home = arg_myhome;
    if (strcmp(default_ext, "*")==0) default_ext = "";
    if      (subeqs(b1,e1,"login")&&(*b3==0))  default_login = substr(b2,e2);
    else if (subeqs(b1,e1,"passwd")&&(*b3==0)) default_passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"speed")&&(*b3==0))  default_speed = atof(b2);
    else if (subeqs(b1,e1,"cpus")&&(*b3==0))   default_cpus = atol(b2);
    else if (subeqs(b1,e1,"home")&&(*b3==0))   default_home = substr(b2,e2);
    else if (subeqs(b1,e1,"ext")&&(*b3==0))    default_ext = substr(b2,e2);
    else if (subeqs(b1,e1,"setup"))            default_setup = strdup(b2);
    else if (subeqs(b1,e1,"host")&&(*b3==0)) {
      if (rightgroup)
	for (default_rank=0; default_rank<default_cpus; default_rank++)
	  nodetab_makehost(substr(b2,e2));
    } else if (subeqs(b1,e1, "group")&&(*b3==0)) {
      rightgroup = subeqs(b2,e2,arg_nodegroup);
    } else {
      fprintf(stderr,"unrecognized command in nodesfile:\n");
      fprintf(stderr,"%s\n", input_line);
      exit(1);
    }
  }
  basicsize = nodetab_size;
  if (basicsize==0) {
    fprintf(stderr,"No hosts in group %s\n", arg_nodegroup);
    exit(1);
  }
  while ((nodetab_size < arg_requested_pes)&&(arg_requested_pes!=MAX_NODES)) {
    nodetab_host h = nodetab_table[nodetab_size-basicsize];
    nodetab_add(h);
  }
  fclose(f);
}

nodetab_host nodetab_getinfo(i)
  int i;
{
  if (nodetab_table==0) {
    fprintf(stderr,"Node table not initialized.\n");
    exit(1);
  }
  if ((i<0)||(i>=nodetab_size)) {
    fprintf(stderr,"No such node %d\n",i);
    exit(1);
  }
  return nodetab_table[i];
}

char        *nodetab_name(i) int i;    { return nodetab_getinfo(i)->name; }
char        *nodetab_login(i) int i;   { return nodetab_getinfo(i)->login; }
char        *nodetab_passwd(i) int i;  { return nodetab_getinfo(i)->passwd; }
char        *nodetab_setup(i) int i;   { return nodetab_getinfo(i)->setup; }
char        *nodetab_home(i) int i;    { return nodetab_getinfo(i)->home; }
char        *nodetab_ext(i) int i;     { return nodetab_getinfo(i)->ext; }
unsigned int nodetab_ip(i) int i;      { return nodetab_getinfo(i)->ip; }
unsigned int nodetab_cpus(i) int i;    { return nodetab_getinfo(i)->cpus; }
unsigned int nodetab_rank(i) int i;    { return nodetab_getinfo(i)->rank; }

 
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
  if (fgets(line, 1023, stdin)==0) { notify_die_doit("end-of-file on stdin"); }
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
    if (fd<0) { notify_die_doit("cannot open temp file /tmp/fnord"); }
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

typedef struct req_node
{
  struct req_node *next;
  char request[1];
}
*req_node;

unsigned int req_fd;
unsigned int req_ip;
unsigned int req_port;
req_node     req_saved;

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

int req_reply(ip, port, pre, ans)
    int ip; int port; char *pre; char *ans;
{
  int fd = skt_connect(ip, port);
  if (fd<=0) return REQ_FAILED;
  write(fd, pre, strlen(pre));
  write(fd, " ", 1);
  write(fd, ans, strlen(ans));
  write(fd, "\n", 1);
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
  printf("%s\n",line+6);
  return REQ_OK;
}

int req_handle_princ(line)
    char *line;
{
  printf("%s",line+6);
  return REQ_OK;
}

int req_handle_printerr(line)
    char *line;
{
  fprintf(stderr,"%s\n",line+9);
  return REQ_OK;
}

int req_handle_princerr(line)
    char *line;
{
  fprintf(stderr,"%s",line+9);
  return REQ_OK;
}

int req_handle_notify_die(line)
    char *line;
{
  char cmd[100], host[100]; int ip, port, nread;
  nread = sscanf(line,"%s%s%d",cmd,host,&port);
  if (nread != 3) return REQ_FAILED;
  ip = lookup_ip(host);
  if (ip==0) return REQ_FAILED;
  notify_die(ip, port);
  return REQ_OK;
}

int req_handle_die(line)
    char *line;
{
  notify_die_doit(line+4);
}

int ending_count=0;

int req_handle_ending(line)
    char *line;
{
  ending_count++;
  if (ending_count == nodetab_size) exit(0);
  return REQ_OK;
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

int req_handle(line)
    char *line;
{
  char cmd[100];
  sscanf(line,"%s",cmd);
  if      (strcmp(cmd,"aset")==0)       return req_handle_aset(line);
  else if (strcmp(cmd,"aget")==0)       return req_handle_aget(line);
  else if (strcmp(cmd,"scanf")==0)      return req_handle_scanf(line);
  else if (strcmp(cmd,"print")==0)      return req_handle_print(line);
  else if (strcmp(cmd,"princ")==0)      return req_handle_princ(line);
  else if (strcmp(cmd,"printerr")==0)   return req_handle_printerr(line);
  else if (strcmp(cmd,"princerr")==0)   return req_handle_princerr(line);
  else if (strcmp(cmd,"ending")==0)     return req_handle_ending(line);
  else if (strcmp(cmd,"notify-die")==0) return req_handle_notify_die(line);
  else if (strcmp(cmd,"die")==0)        return req_handle_die(line);
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

req_serve_client(f)
    FILE *f;
{
  while (1) {
  }
}

req_serve()
{
  char line[1000]; int status; FILE *f;
  int client_ip, client_port, client_fd; req_node n;
  while (1) {
    skt_accept(req_fd, &client_ip, &client_port, &client_fd);
    if (client_fd==0) {
      perror("accept");
      notify_abort();
    }
    f = fdopen(client_fd, "r+");
    line[0]=0;
    fgets(line, 999, f);
    fclose(f); close(client_fd);
    if (line[0]==0) continue;
    zap_newline(line);
    status = req_handle(line);
    switch (status) {
    case REQ_OK: break;
    case REQ_FAILED: fprintf(stderr,"bad request: %s\n",line); break;
    case REQ_OK_AWAKEN: req_run_saved(); break;
    case REQ_POSTPONE:
      n = (req_node)malloc(sizeof(struct req_node)+strlen(line));
      strcpy(n->request, line);
      n->next = req_saved;
      req_saved = n;
    }
    line[0]=0;
  }
}

req_init()
{
  skt_server(&req_ip, &req_port, &req_fd);
  req_saved = 0;
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
  
  rshargv[0]=RSH_CMD;
  rshargv[1]=nodetab_name(nodeno);
  rshargv[2]="-l";
  rshargv[3]=nodetab_login(nodeno);
  rshargv[4]="exec /bin/csh -f";
  rshargv[5]=0;
  rsh = prog_start(arg_rshprog, rshargv, 0);
  if ((rsh==0)&&(errno!=EMFILE)) { perror("starting rsh"); exit(1); }
  if (rsh==0)
    {
      fprintf(stderr,"caution: cannot start specified number of rsh's\n");
      fprintf(stderr,"(not enough file descriptors available).\n");
    }
  if (rsh && (arg_debug || arg_debug_no_pause))
    fprintf(stderr,"node %d: rsh initiated...\n",nodeno);
  return rsh;
}

int rsh_pump(p, nodeno, rank0no, argv)
    prog p; int nodeno, rank0no; char **argv;
{
  static char buf[1024];
  int len;
  xstr ibuf = p->ibuf;
  int randno = rand();
  
  xstr_printf(ibuf,"echo 'remote responding...'\n");

  if (arg_display)
    xstr_printf(ibuf,"setenv DISPLAY %s\n",arg_display);
  xstr_printf(ibuf,"setenv NETSTART '%d %d %d %d %d %d %d %d %d'\n",
	      nodetab_rank0_size, rank0no,
	      nodeno, nodetab_cpus(nodeno), nodetab_size,
	      nodetab_ip(nodeno), req_ip, req_port, (getpid()&0x7FFF));
  prog_flush(p);
  
  /* find the node-program, relative version */
  sprintf(buf,"%s",getenv("HOME"));
  if ((len=path_isprefix(buf,arg_nodeprog_a))!=0) {
    sprintf(buf,"%s/%s%s",nodetab_home(nodeno),
			  arg_nodeprog_a+len,nodetab_ext(nodeno));
    arg_nodeprog_r = strdup(buf);
  }
  else {
    sprintf(buf,"%s%s",arg_nodeprog_a,nodetab_ext(nodeno));
    arg_nodeprog_r = strdup(buf);
  }

  /* find the current directory, relative version */
  sprintf(buf,"%s",getenv("HOME"));
  if ((len=path_isprefix(buf, arg_currdir_a))!=0) {
    sprintf(buf,"%s/%s",nodetab_home(nodeno),arg_currdir_a+len);
    arg_currdir_r = strdup(buf);
  }
  else arg_currdir_r = arg_currdir_a;

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    xstr_printf(ibuf,"foreach dir ($path)\n");
    xstr_printf(ibuf,"  if (-e $dir/xterm) setenv F_XTERM $dir/xterm\n");
    xstr_printf(ibuf,"  if (-e $dir/xrdb) setenv F_XRDB $dir/xrdb\n");
    xstr_printf(ibuf,"end\n");
    xstr_printf(ibuf,"if ($?F_XTERM == 0) then\n");
    xstr_printf(ibuf,"   echo 'xterm not in path --- set your path in your cshrc.'\n");
    xstr_printf(ibuf,"   kill -9 $$\n");
    xstr_printf(ibuf,"endif\n");
    xstr_printf(ibuf,"if ($?F_XRDB == 0) then\n");
    xstr_printf(ibuf,"   echo 'xrdb not in path - set your path in your cshrc.'\n");
    xstr_printf(ibuf,"   kill -9 $$\n");
    xstr_printf(ibuf,"endif\n");
    prog_flush(p);
  }

  if (arg_debug || arg_debug_no_pause) {
    xstr_printf(ibuf,"foreach dir ($path)\n");
    xstr_printf(ibuf,"  if (-e $dir/gdb) setenv F_GDB $dir/gdb\n");
    xstr_printf(ibuf,"end\n");
    xstr_printf(ibuf,"if ($?F_GDB == 0) then\n");
    xstr_printf(ibuf,"   echo 'gdb not in path - set your path in your cshrc.'\n");
    xstr_printf(ibuf,"   kill -9 $$\n");
    xstr_printf(ibuf,"endif\n");
    prog_flush(p);
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
  
  if (arg_debug || arg_debug_no_pause) {
    xstr_printf(ibuf,"cat > /tmp/gdb%08x << END_OF_SCRIPT\n",randno);
    xstr_printf(ibuf,"shell rm -f /tmp/gdb%08x\n",randno);
    xstr_printf(ibuf,"handle SIGWINCH nostop noprint\n");
    xstr_printf(ibuf,"handle SIGWAITING nostop noprint\n");
    xstr_printf(ibuf,"set args");
    while (*argv) { xstr_printf(ibuf," %s",*argv); argv++; }
    xstr_printf(ibuf,"\n");
    if (arg_debug_no_pause) xstr_printf(ibuf,"run\n");
    xstr_printf(ibuf,"END_OF_SCRIPT\n");
    xstr_printf(ibuf,"$F_XTERM");
    xstr_printf(ibuf," -T 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
    xstr_printf(ibuf," -n 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
    xstr_printf(ibuf," -e $F_GDB %s -x /tmp/gdb%08x",arg_nodeprog_r,randno);
    xstr_printf(ibuf," < /dev/null >& /dev/null &");
    xstr_printf(ibuf,"\n");
  } else if (arg_in_xterm) {
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
    xstr_printf(ibuf," -T 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
    xstr_printf(ibuf," -n 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
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
  
  xstr_printf(ibuf,"echo 'rsh phase successful.'\n");
  xstr_printf(ibuf,"kill -9 $$\n");
  
  prog_flush(p);
  prog_iclose(p);
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
  }
  if (rsh_maxsim==0) { perror("starting rsh"); exit(1); }
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
    if (ok<0) { perror("select"); exit(1); }
    do ok = waitpid((pid_t)(-1), NULL, WNOHANG);
    while (ok>=0);
    for (i=0; i<rsh_maxsim; i++) {
      char *line = 0;
      char buffer[1000];
      int nread, done = 0;
      p = rsh_prog[i];
      if (p==0) continue;
      if (!FD_ISSET(p->ofd, &rfds)) continue;
      do nread = read(p->ofd, buffer, 1000);
      while ((nread<0)&&(errno==EINTR));
      if (nread<0) { perror("read"); exit(1); }
      if (nread==0) {
	fprintf(stderr,"node %d: rsh phase failed.\n",rsh_node[i]);
	exit(1);
      }
      xstr_write(p->obuf, buffer, nread);
      while (1) {
	line = xstr_gets(buffer, 999, p->obuf);
	if (line==0) break;
	if (strncmp(line,"[1] ",4)==0) continue;
	if (arg_debug || arg_debug_no_pause ||
	    (strcmp(line,"rsh phase successful.")
	     &&strcmp(line,"remote responding...")))
	  fprintf(stderr,"node %d: %s\n",rsh_node[i],line);
	if (strcmp(line,"rsh phase successful.")==0) { done=1; break; }
      }
      if (!done) continue;
      rsh_nfinished++;
      prog_close(p);
      rsh_prog[i] = 0;
      if (rsh_nstarted==nodetab_rank0_size) break;
      pe = nodetab_rank0_table[rsh_nstarted];
      p = rsh_start(pe);
      if (p==0) { perror("starting rsh"); exit(1); }
      rsh_pump(p, pe, rsh_nstarted, arg_argv);
      rsh_prog[i] = p;
      rsh_node[i] = pe;
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
  srand(time(0));
  /* notify charm developers that charm is in use */
  ping_developers();
  /* Compute the values of all constants */
  arg_init(argc, argv);
  /* Initialize the node-table by reading nodesfile */
  nodetab_init();
  /* Initialize the request-server */
  req_init();
  /* Initialize the kill-handler */
  notify_die_init();
  /* Initialize the IO module */
  input_init();
  /* start the node processes */
  start_nodes();
  /* enter request-service mode */
  req_serve();
}

