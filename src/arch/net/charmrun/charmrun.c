/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
#include "conv-mach.h"
#include "converse.h"

#include "../sockRoutines.h"
#include "../sockRoutines.c"
#include "../ccs-auth.h"
#include "../ccs-auth.c"
#include "../ccs-server.h"
#include "../ccs-server.c"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <setjmp.h>
#include <stdlib.h>
#include <signal.h>
#include <fcntl.h>
#include <time.h>
#if CMK_SCYLD
#include <sys/bproc.h>
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
/*Win32 has screwy names for the standard UNIX calls:*/
#define getcwd _getcwd
#define strdup _strdup
#define unlink _unlink
#define open _open
#define fdopen _fdopen
#define ftruncate _chsize
#include <winbase.h>
#include <direct.h>
#include <io.h>
#include <sys/timeb.h>
#include <process.h>
#define DIRSEP "\\"
#define SIGBUS -1  /*These signals don't exist in Win32*/
#define SIGKILL -1
#define SIGQUIT -1


#else /*UNIX*/
#include <pwd.h>   /*getcwd*/
#include <unistd.h>
#include <varargs.h>
#define DIRSEP "/"
#endif

#if CMK_RSH_NOT_NEEDED /*No RSH-- use daemon to start node-programs*/
#  define CMK_USE_RSH 0

#else /*Use RSH to start node-programs*/
#  define CMK_USE_RSH 1
#ifndef __CYGWIN__
#  include <rpc/rpc.h>
#else
#  include <w32api/rpc.h>
#endif
#  if CMK_RSH_IS_A_COMMAND
#    define RSH_CMD "rsh"
#  endif

#  if CMK_RSH_USE_REMSH
#    define RSH_CMD "remsh"
#  endif
#endif

#include "../daemon.h"

#define DEBUGF(x) /*printf x*/

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif


int probefile(path)
    char *path;
{
	FILE *f=fopen(path,"r");
	if (f==NULL) return 0;
	fclose(f);
	return 1;
}

char *mylogin(void)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
	static char name[100]={'d','u','n','n','o',0};
	int len=100;
	GetUserName(name,&len);
	return name;
#else /*UNIX*/
  struct passwd *self;

  self = getpwuid(getuid());
  if (self==0) { return "unknown"; }
  return self->pw_name;
#endif
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
  /*This is the resolved IP address of elegance.cs.uiuc.edu */
  skt_ip_t destination_ip=skt_lookup_ip("128.174.241.211");
  unsigned int destination_port=6571;
  struct sockaddr_in addr=skt_build_addr(destination_ip,destination_port);
  SOCKET             skt;
  
  skt = socket(AF_INET, SOCK_DGRAM, 0);
  if (skt == INVALID_SOCKET) return;

  sprintf(info,"%s",mylogin());
  
  sendto(skt, info, strlen(info), 0, (struct sockaddr *)&addr, sizeof(addr));
  skt_close(skt);
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

/****************************************************************************
 *
 * Miscellaneous minor routines.
 *
 ****************************************************************************/

int is_quote(char c)
{
  return (c=='\'' || c == '"');
}

void zap_newline(char *s)
{
  char *p;
  p = s + strlen(s)-1;
  if (*p == '\n') *p = '\0';
}

/* get substring from lo to hi, remove quote chars */
char *substr(char *lo, char *hi)
{
  int len;
  char *res;
  if (is_quote(*lo)) lo++;
  if (is_quote(*(hi-1))) hi--;
  len = hi-lo;
  res = (char *)malloc(1+len);
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

/* advance pointer over nonblank characters and a quoted string */
char *skipstuff(char *p)
{
  char quote = 0;
  if (*p && (*p=='\'' || *p=='"')) { quote=*p; p++; }
  if (quote != 0) {
    while (*p&&*p!=quote) p++;
    if (*p!=quote) {
      fprintf(stderr, "ERROR> Unmatched quote in nodelist file.\n");
      exit(1);
    }
    p++;
  }
  else
    while ((*p)&&(*p!=' ')&&(*p!='\t')) p++;
  return p;
}

#if CMK_USE_RSH
char *getenv_rsh()
{
  char *e;

  e = getenv("CONV_RSH");
  return e ? e : RSH_CMD;
}
#endif

#if !defined(_WIN32) || defined(__CYGWIN__)
char *getenv_display()
{
  static char result[100],ipBuf[200];
  char *e, *p;
  
  e = getenv("DISPLAY");
  if (e==0) return NULL;
  p = strrchr(e, ':');
  if (p==0) return NULL;
  if ((e[0]==':')||(strncmp(e,"unix:",5)==0)) {
    sprintf(result,"%s:%s",skt_print_ip(ipBuf,skt_my_ip()),p+1);
  }
  else strcpy(result, e);
  return result;
}
#endif

/*****************************************************************************
 *                                                                           *
 * PPARAM - obtaining "program parameters" from the user.                    *
 *                                                                           *
 *****************************************************************************/

typedef struct ppdef
{
  union {
      int *i;
      double *r;
      char **s;
      int *f;
    } where;/*Where to store result*/
  const char *lname; /*Argument name on command line*/
  const char *doc;
  char  type; /*One of i, r, s, f.*/
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



void pparam_int(int *where,int defValue,
				   const char *arg,const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type  = 'i';
  def->where.i = where; *where=defValue;
  def->lname=arg;
  def->doc=doc;
}

void pparam_flag(int *where,int defValue,
				   const char *arg,const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type  = 'f';
  def->where.f = where; *where=defValue;
  def->lname=arg;
  def->doc=doc;
}

void pparam_real(double *where,double defValue,
				   const char *arg,const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type  = 'r';
  def->where.r = where; *where=defValue;
  def->lname=arg;
  def->doc=doc;
}
void pparam_str(char **where,char *defValue,
				   const char *arg,const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type  = 's';
  def->where.s = where; *where=defValue;
  def->lname=arg;
  def->doc=doc;
}

static int pparam_setdef(def, value)
    ppdef def; char *value;
{
  char *p;
  switch(def->type)
    {
    case 'i' :
      *def->where.i = strtol(value, &p, 10);
      if (*p) return -1;
      return 0;
    case 'r' :
      *def->where.r = strtod(value, &p);
      if (*p) return -1;
      return 0;
    case 's' :
      *def->where.s = strdup(value);
      return 0;
    case 'f' :
      *def->where.f = strtol(value, &p, 10);
      if (*p) return -1;
      return 0;
    }
  return -1;
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
    case 'i': sprintf(result,"%d", *def->where.i); return result;
    case 'r': sprintf(result,"%f",*def->where.r); return result;
    case 's': return *def->where.s?*def->where.s:"";
    case 'f': sprintf(result,"%d", *def->where.f); return result;
    }
  return NULL;
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
  int ok; ppdef def=NULL;
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
  if (def==NULL)
  {
    if (opt[1]=='+')
    {
       sprintf(pparam_error,"Option %s not recognized.",opt);
       return -1;
    } else {
	   /*Unrecognized + option-- skip it.*/
	   pparam_pos++;
	   return 0;
	}
  }
  /* handle flag-options */
  if ((def->type=='f')&&(opt[1]!='+')&&(opt[2]))
    {
      sprintf(pparam_error,"Option %s should not include a value",opt);
      return -1;
    }
  if (def->type=='f')
    {
      *def->where.f = 1;
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
char *arg_nodelist;
char *arg_nodegroup;

int   arg_debug;
int   arg_debug_no_pause;

int   arg_local;	/* start node programs directly by exec on localhost */

int   arg_help;		/* print help message */
int   arg_usehostname;

#if CMK_USE_RSH
int   arg_maxrsh;
char *arg_shell;
int   arg_in_xterm;
char *arg_debugger;
char *arg_xterm;
char *arg_display;
char *arg_mylogin;
#endif

char *arg_nodeprog_a;
char *arg_nodeprog_r;
char *arg_currdir_a;
char *arg_currdir_r;

int   arg_server;
int   arg_server_port=0;
char *arg_server_auth=NULL;

#if CMK_SCYLD
int   arg_startpe;
#endif

void arg_init(int argc, char **argv)
{
  static char buf[1024];

  pparam_int(&arg_requested_pes, 1, "p",             "number of processes to create");
  pparam_int(&arg_timeout,      60, "timeout",       "seconds to wait per host connection");
  pparam_flag(&arg_verbose,      0, "verbose",       "Print diagnostic messages");
  pparam_str(&arg_nodelist,      0, "nodelist",      "file containing list of nodes");
  pparam_str(&arg_nodegroup,"main", "nodegroup",     "which group of nodes to use");

#if CMK_CCS_AVAILABLE
  pparam_flag(&arg_server,       0, "server",        "Enable client-server (CCS) mode");
  pparam_int(&arg_server_port,   0, "server-port",   "Port to listen for CCS requests");
  pparam_str(&arg_server_auth,   0, "server-auth",   "CCS Authentication file");
#endif
  pparam_flag(&arg_local,	0, "local", "Start node programs locally without daemon");
  pparam_flag(&arg_usehostname,  0, "usehostname", "Send nodes our symbolic hostname instead of IP address");
#if CMK_USE_RSH
  pparam_flag(&arg_debug,         0, "debug",         "Run each node under gdb in an xterm window");
  pparam_flag(&arg_debug_no_pause,0, "debug-no-pause","Like debug, except doesn't pause at beginning");
  pparam_int(&arg_maxrsh,        16, "maxrsh",        "Maximum number of rsh's to run at a time");
  pparam_str(&arg_shell,          0, "remote-shell",  "which remote shell to use");
  pparam_str(&arg_debugger,       0, "debugger",      "which debugger to use");
  pparam_str(&arg_display,        0, "display",       "X Display for xterm");
  pparam_flag(&arg_in_xterm,      0, "in-xterm",      "Run each node in an xterm window");
  pparam_str(&arg_xterm,          0, "xterm",         "which xterm to use");
#endif
#ifdef CMK_SCYLD
  pparam_int(&arg_startpe,   0, "startpe",   "first pe to start job(SCYLD)");
#endif
  pparam_flag(&arg_help,	0, "help", "print help messages");

  if (pparam_parsecmd('+', argv) < 0) {
    fprintf(stderr,"ERROR> syntax: %s\n",pparam_error);
    pparam_printdocs();
    exit(1);
  }

  if (arg_help) {
    pparam_printdocs();
    exit(0);
  }

  arg_argv = argv+1; /*Skip over charmrun (0) here and program name (1) later*/
  arg_argc = pparam_countargs(arg_argv);
  if (arg_argc<1) {
    fprintf(stderr,"ERROR> You must specify a node-program.\n");
    pparam_printdocs();
    exit(1);
  }
  arg_argv++; arg_argc--;

  if (arg_server_port || arg_server_auth) arg_server=1;

  if (arg_debug || arg_debug_no_pause) {
	arg_verbose=1;
	/*Pass ++debug along to program (used by machine.c)*/
	arg_argv[arg_argc++]="++debug";
  }

#if CMK_USE_RSH
  /* Find the current value of the CONV_RSH variable */
  if(!arg_shell) arg_shell = getenv_rsh();

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

  arg_mylogin = mylogin();
#endif

  /* find the current directory, absolute version */
  getcwd(buf, 1023);
  arg_currdir_a = strdup(buf);
  
  /* find the node-program, absolute version */
  arg_nodeprog_r = argv[1];

#if defined(_WIN32) && !defined(__CYGWIN__)
  if (argv[1][1]==':') { /*E.g.: "C:\foo\bar.exe*/
#else
  if (argv[1][0]=='/') { /*E.g.: "\foo\bar"*/
#endif
	  /*Absolute path to node-program*/
    arg_nodeprog_a = argv[1];
  } else {
    sprintf(buf,"%s%s%s",arg_currdir_a,DIRSEP,arg_nodeprog_r);
    arg_nodeprog_a = strdup(buf);
  }
}

/****************************************************************************
 *                                                                           
 * NODETAB:  The nodes file and nodes table.
 *
 ****************************************************************************/

static int portOk = 1;
static const char *nodetab_tempName=NULL;
char *nodetab_file_find()
{
  char buffer[MAXPATHLEN];

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
#if defined(_WIN32) && !defined(__CYGWIN__)
  tmpnam(buffer);
  nodetab_tempName=strdup(buffer);
#else /*UNIX*/
  if (getenv("HOME")) {
    sprintf(buffer,"%s/.nodelist",getenv("HOME"));
  }
#endif
  if (!probefile(buffer)) 
  {
    /*Create a simple nodelist in the user's home*/
    FILE *f=fopen(buffer,"w");
    if (f==NULL) {
      fprintf(stderr,"ERROR> Cannot create a 'nodelist' file.\n");
      exit(1);
    }
    fprintf(f,"group main\nhost localhost\n");
    fclose(f);
  }
  return strdup(buffer);
}

typedef struct nodetab_host {
  char    *name;  /*Host DNS name*/
  skt_ip_t ip; /*IP address of host*/
  pathfixlist pathfixes;
  char    *ext;  /*FIXME: What the heck is this?  OSL 9/8/00*/
  int      cpus;  /* # of physical CPUs*/
  int      rank;  /*Rank of this CPU*/
  double   speed; /*Relative speed of each CPU*/
  /*These fields are set during node-startup*/
  int     dataport;/*UDP port number*/
  SOCKET  ctrlfd;/*Connection to control port*/
#if CMK_USE_RSH
  char    *shell;  /*Rsh to use*/
  char    *debugger ; /*Debugger to use*/
  char    *xterm ;  /*Xterm to use*/
  char    *login;  /*User login name to use*/
  char    *passwd;  /*User login password*/
  char    *setup;  /*Commands to execute on login*/
#endif
} nodetab_host;

nodetab_host **nodetab_table;
int           nodetab_max;
int           nodetab_size;
int          *nodetab_rank0_table;
int           nodetab_rank0_size;

void nodetab_reset(nodetab_host *h)
{
  h->name="SET_H->NAME";
  h->ip=skt_invalid_ip;
  h->pathfixes = 0;
  h->ext = "*";
  h->speed = 1.0;
  h->cpus = 1;
  h->rank = 0;
  h->dataport=-1;
  h->ctrlfd=-1;
#if CMK_USE_RSH
  h->shell = arg_shell;
  h->debugger = arg_debugger;
  h->xterm = arg_xterm;
  h->login = arg_mylogin;
  h->passwd = "*";
  h->setup = "*";
#endif
}

void nodetab_add(nodetab_host *h)
{
  if (h->rank == 0)
    nodetab_rank0_table[nodetab_rank0_size++] = nodetab_size;
  nodetab_table[nodetab_size] = (nodetab_host *) malloc(sizeof(nodetab_host));

  if (arg_verbose) {
    char ips[200];
    skt_print_ip(ips,h->ip);
    printf("Charmrun> adding client %d: \"%s\", IP:%s\n", nodetab_size, h->name, ips);
  }

  *nodetab_table[nodetab_size++] = *h;
}

void nodetab_makehost(char *name,nodetab_host *h)
{
  h->name=strdup(name);
  h->ip = skt_innode_lookup_ip(name);
  if (skt_ip_match(h->ip,skt_invalid_ip)) {
    fprintf(stderr,"ERROR> Cannot obtain IP address of %s\n", name);
    exit(1);
  }
  if (nodetab_size == nodetab_max) return;
  nodetab_add(h);
}

char *nodetab_args(char *args,nodetab_host *h)
{
  while(*args != 0) {
    char *b1 = skipblanks(args), *e1 = skipstuff(b1);
    char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
    while (*b1=='+') b1++;/*Skip over "++" on parameters*/
#if CMK_USE_RSH
    if (subeqs(b1,e1,"login")) h->login = substr(b2,e2);
    else if (subeqs(b1,e1,"passwd")) h->passwd = substr(b2,e2);
    else if (subeqs(b1,e1,"setup")) h->setup = strdup(b2);
    else if (subeqs(b1,e1,"shell")) h->shell = substr(b2,e2);
    else if (subeqs(b1,e1,"debugger")) h->debugger = substr(b2,e2);
    else if (subeqs(b1,e1,"xterm")) h->xterm = substr(b2,e2);
    else 
#endif
    if (subeqs(b1,e1,"speed")) h->speed = atof(b2);
    else if (subeqs(b1,e1,"cpus")) h->cpus = atol(b2);
    else if (subeqs(b1,e1,"pathfix")) {
      char *b3 = skipblanks(e2), *e3 = skipstuff(b3);
      args = skipblanks(e3);
      h->pathfixes=pathfix_append(substr(b2,e2),substr(b3,e3),h->pathfixes);
    } 
    else if (subeqs(b1,e1,"ext")) h->ext = substr(b2,e2);
    else return args;
    args = skipblanks(e2);
  }
  return args;
}

/*  setup nodetab as localhost only */
void nodetab_init_for_local()
{
  int tablesize, i;
  nodetab_host group;

  tablesize = arg_requested_pes;
  nodetab_table=(nodetab_host**)malloc(tablesize*sizeof(nodetab_host*));
  nodetab_rank0_table=(int*)malloc(tablesize*sizeof(int));
  nodetab_max=tablesize;

  nodetab_reset(&group);
  for (i=0; i<tablesize; i++) {
    char *hostname = "localhost";
    nodetab_makehost(hostname, &group);
  }
}

void nodetab_init()
{
  FILE *f,*fopen();
  char *nodesfile; 
  nodetab_host global,group,host;
  char input_line[MAX_LINE_LENGTH];
  int rightgroup, basicsize, i, remain;
  
  /* if arg_local is set, ignore the nodelist file */
  if (arg_local) {
    nodetab_init_for_local();
    return;
  }

  /* Open the NODES_FILE. */
  nodesfile = nodetab_file_find();
  if(arg_verbose)
    fprintf(stderr, "Charmrun> using %s as nodesfile\n", nodesfile);
  if (!(f = fopen(nodesfile,"r"))) {
    fprintf(stderr,"ERROR> Cannot read %s: %s\n",nodesfile,strerror(errno));
    exit(1);
  }
  
  nodetab_table=(nodetab_host**)malloc(arg_requested_pes*sizeof(nodetab_host*));
  nodetab_rank0_table=(int*)malloc(arg_requested_pes*sizeof(int));
  nodetab_max=arg_requested_pes;
  
  nodetab_reset(&global);
  group=global;
  rightgroup = (strcmp(arg_nodegroup,"main")==0);
  
  while(fgets(input_line,sizeof(input_line)-1,f)!=0) {
    if (nodetab_size == arg_requested_pes) break;
    if (input_line[0]=='#') continue;
    zap_newline(input_line);
	if (!nodetab_args(input_line,&global)) {
		/*An option line-- also add options to current group*/
		nodetab_args(input_line,&group);
	}
	else {/*Not an option line*/
		char *b1 = skipblanks(input_line), *e1 = skipstuff(b1);
		char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
		char *b3 = skipblanks(e2);
		if (subeqs(b1,e1,"host")) {
#if CMK_USE_GM
			/*  host [hostname] [port] ...  */
                        char *b4 = b3;
                        char *e4 = skipstuff(b3);
                        b3 = skipblanks(e4);
#endif
			if (rightgroup) {
				host=group;
				nodetab_args(b3,&host);
#if CMK_USE_GM
				host.dataport = atoi(b4);
                                if (host.dataport == 0) { fprintf(stderr, "Missing port number!\n"); exit(1); }
#endif
				for (host.rank=0; host.rank<host.cpus; host.rank++)
					nodetab_makehost(substr(b2,e2),&host);
			}
		} else if (subeqs(b1,e1, "group")) {
			group=global;
			nodetab_args(b3,&group);
			rightgroup = subeqs(b2,e2,arg_nodegroup);
		} else if (b1!=b3) {
			fprintf(stderr,"ERROR> unrecognized command in nodesfile:\n");
			fprintf(stderr,"ERROR> %s\n", input_line);
			exit(1);
		}
	}
  }
  fclose(f);
  if (nodetab_tempName!=NULL) unlink(nodetab_tempName);

  /*Wrap nodes in table around if there aren't enough yet*/
  basicsize = nodetab_size;
  if (basicsize==0) {
    fprintf(stderr,"ERROR> No hosts in group %s\n", arg_nodegroup);
    exit(1);
  }
  while ((nodetab_size < arg_requested_pes)&&(arg_requested_pes!=MAX_NODES))
     nodetab_add(nodetab_table[nodetab_size%basicsize]);
  
  /*Clip off excess CPUs at end*/
  for (i=0; i<nodetab_size; i++) {
    if (nodetab_table[i]->rank == 0)
      remain = nodetab_size - i;
    if (nodetab_table[i]->cpus > remain)
      nodetab_table[i]->cpus = remain;
  }
}

nodetab_host *nodetab_getinfo(int i)
{
  if (nodetab_table==0) {
    fprintf(stderr,"ERROR> Node table not initialized.\n");
    exit(1);
  }
  return nodetab_table[i];
}

char        *nodetab_name(int i)     { return nodetab_getinfo(i)->name; }
pathfixlist  nodetab_pathfixes(int i){ return nodetab_getinfo(i)->pathfixes; }
char        *nodetab_ext(int i)      { return nodetab_getinfo(i)->ext; }
skt_ip_t     nodetab_ip(int i)       { return nodetab_getinfo(i)->ip; }
unsigned int nodetab_cpus(int i)     { return nodetab_getinfo(i)->cpus; }
unsigned int nodetab_rank(int i)     { return nodetab_getinfo(i)->rank; }
int          nodetab_dataport(int i) { return nodetab_getinfo(i)->dataport; }
SOCKET      nodetab_ctrlfd(int i)    { return nodetab_getinfo(i)->ctrlfd;}
#if CMK_USE_RSH
char        *nodetab_setup(int i)    { return nodetab_getinfo(i)->setup; }
char        *nodetab_shell(int i)    { return nodetab_getinfo(i)->shell; }
char        *nodetab_debugger(int i) { return nodetab_getinfo(i)->debugger; }
char        *nodetab_xterm(int i)    { return nodetab_getinfo(i)->xterm; }
char        *nodetab_login(int i)    { return nodetab_getinfo(i)->login; }
char        *nodetab_passwd(int i)   { return nodetab_getinfo(i)->passwd; }
#endif

/****************************************************************************
 *
 * Nodeinfo
 *
 * The global list of node PEs, IPs, and port numbers.
 * Stored in ChMachineInt_t format so the table can easily be sent
 * back to the nodes.
 *
 ****************************************************************************/

static ChNodeinfo *nodeinfo_arr;/*Indexed by node number.*/

void nodeinfo_allocate(void)
{
	nodeinfo_arr=(ChNodeinfo *)malloc(nodetab_rank0_size*sizeof(ChNodeinfo));
}
void nodeinfo_add(const ChSingleNodeinfo *in,SOCKET ctrlfd)
{
	int node=ChMessageInt(in->nodeNo);
	ChNodeinfo i=in->info;
	int nt,pe;
	if (node<0 || node>=nodetab_rank0_size)
		{fprintf(stderr,"Unexpected node %d registered!\n",node);exit(1);}
	nt=nodetab_rank0_table[node];/*Nodetable index for this node*/
	i.nPE=ChMessageInt_new(nodetab_cpus(nt));
#if CMK_USE_GM
        *(int *)&i.IP = ChMessageInt(i.dataport);
        if (ChMessageInt(i.dataport)==0) {
          fprintf(stderr, "Error> Node %d:%s, cannot open GM gm_port %d!\n", nt, nodetab_name(nt), nodetab_dataport(nt));
          portOk = 0;
        }
        i.dataport=ChMessageInt_new(nodetab_dataport(nt));
#else
	i.IP=nodetab_ip(node);
#endif
	nodeinfo_arr[node]=i;
	for (pe=0;pe<nodetab_cpus(nt);pe++)
	  {
	    nodetab_table[nt+pe]->dataport=ChMessageInt(i.dataport);
	    nodetab_table[nt+pe]->ctrlfd=ctrlfd;
	  }
        if (arg_verbose) {
	  char ips[200];
	  skt_print_ip(ips,nodetab_ip(nt));
	  printf("Charmrun> client %d connected (IP=%s)\n", nt, ips);
	}
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

/*FIXME: I am terrified by this routine. OSL 9/8/00*/
char *input_scanf_chars(fmt)
    char *fmt;
{
  char buf[8192]; int len, pos;
  static int fd; static FILE *file;
  fflush(stdout);
  if (file==0) {
	char *tmp=tmpnam(NULL);/*This was once /tmp/fnord*/
    unlink(tmp);
    fd = open(tmp,O_RDWR | O_CREAT | O_TRUNC);
    if (fd<0) { 
      fprintf(stderr,"cannot open temp file /tmp/fnord");
      exit(1);
    }
    file = fdopen(fd, "r+");
    unlink(tmp);
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

/***************************************************************************
CCS Interface:
  Charmrun forwards CCS requests on to the node-programs' control
sockets.
***************************************************************************/

#if CMK_CCS_AVAILABLE

/*The Ccs Server socket became active-- 
rec'v the message and respond to the request,
by forwarding the request to the appropriate node.
 */
void req_ccs_connect(void)
{
  struct {
   ChMessageHeader ch;/*Make a charmrun header*/
   CcsImplHeader hdr;/*Ccs internal header*/
  } h;
  void *reqData;/*CCS request data*/
  int pe,reqBytes;
  if (0==CcsServer_recvRequest(&h.hdr,&reqData))
    return;/*Malformed request*/
  pe=ChMessageInt(h.hdr.pe);
  reqBytes=ChMessageInt(h.hdr.len);

  if (pe<0 || pe>=nodetab_size) {
	pe=0;
	h.hdr.pe=ChMessageInt_new(pe);
  }

  /*Fill out the charmrun header & forward the CCS request*/
  ChMessageHeader_new("req_fw",sizeof(h.hdr)+reqBytes,&h.ch);  

  skt_sendN(nodetab_ctrlfd(pe),&h,sizeof(h));
  skt_sendN(nodetab_ctrlfd(pe),reqData,reqBytes);
  free(reqData);
}

/*
Forward the CCS reply (if any) back to the original requestor,
on the original request socket.
 */
void req_ccs_reply_fw(ChMessage *msg) {
  CcsImplHeader *hdr=(CcsImplHeader *)msg->data;
  CcsServer_sendReply(hdr,msg->len-sizeof(CcsImplHeader),
		      msg->data+sizeof(CcsImplHeader));
  ChMessage_free(msg);
}

#else
void req_ccs_connect(void) {}
void req_handle_ccs(void) {}
#endif /*CMK_CCS_AVAILABLE*/

/****************************************************************************
 *
 * REQUEST SERVICER
 *
 * The request servicer accepts connections on a TCP port.  The client
 * sends a sequence of commands (each is one line).  It then closes the
 * connection.  The server must then contact the client, sending replies.
 *
 ****************************************************************************/

SOCKET *req_clients; /*TCP request sockets for each node*/
int  req_nClients;/*Number of entries in above list (==nodetab_rank0_size)*/
int             req_ending=0;

#define REQ_OK 0
#define REQ_FAILED -1

/* This is the only place where charmrun talks back to anyone. 
*/
int req_reply(SOCKET fd, char *type, 
	      const char *data, int dataLen)
{
  ChMessageHeader msg;
  if (fd == INVALID_SOCKET) return REQ_FAILED;
  ChMessageHeader_new(type,dataLen,&msg);
  skt_sendN(fd,(const char *)&msg,sizeof(msg));
  skt_sendN(fd,data,dataLen);
  return REQ_OK;
}

/* Request handlers:
When a client asks us to do something, these are the
routines that actually respond to the request.
*/
/*Stash this new node's control and data ports.
 */
int req_handle_initnode(ChMessage *msg,SOCKET fd)
{
  if (msg->len!=sizeof(ChSingleNodeinfo)) {
    fprintf(stderr,"Charmrun: Bad initnode data length. Aborting\n");
    fprintf(stderr,"Charmrun: possibly because: %s.\n", msg->data);
    exit(1);
  }
  nodeinfo_add((ChSingleNodeinfo *)msg->data,fd);
  return REQ_OK;
}

/*Get the array of node numbers, IPs, and ports.
This is used by the node-programs to talk to one another.
*/
int req_handle_initnodetab(ChMessage *msg,SOCKET fd)
{
	ChMessageHeader hdr;
	ChMessageInt_t nNodes=ChMessageInt_new(nodetab_rank0_size);
	ChMessageHeader_new("initnodetab",sizeof(ChMessageInt_t)+
			    sizeof(ChNodeinfo)*nodetab_rank0_size,&hdr);
	skt_sendN(fd,(const char *)&hdr,sizeof(hdr));
	skt_sendN(fd,(const char *)&nNodes,sizeof(nNodes));
	skt_sendN(fd,(const char *)nodeinfo_arr,
		  sizeof(ChNodeinfo)*nodetab_rank0_size);
	return REQ_OK;
}

int req_handle_print(ChMessage *msg,SOCKET fd)
{
  printf("%s",msg->data);
  fflush(stdout);
  return REQ_OK;
}


int req_handle_printerr(ChMessage *msg,SOCKET fd)
{
  fprintf(stderr,"%s",msg->data);
  fflush(stderr);
  return REQ_OK;
}


int req_handle_printsyn(ChMessage *msg,SOCKET fd)
{
  printf("%s",msg->data);
  fflush(stdout);
  req_reply(fd, "printdone", "", 1);
  return REQ_OK;
}


int req_handle_printerrsyn(ChMessage *msg,SOCKET fd)
{
  fprintf(stderr,"%s",msg->data);
  fflush(stderr);
  req_reply(fd, "printdone", "", 1);
  return REQ_OK;
}


int req_handle_ending(ChMessage *msg,SOCKET fd)
{  
  int i;
  req_ending++;
    
  if (req_ending == nodetab_size)
  {
    for (i=0;i<req_nClients;i++)
      skt_close(req_clients[i]);
    if (arg_verbose) printf("Charmrun> Graceful exit.\n");
    exit(0);
  }
  return REQ_OK;
}


int req_handle_abort(ChMessage *msg,SOCKET fd)
{
  if (msg->len==0) 
    fprintf(stderr,"Aborting!\n");
  else
    fprintf(stderr, "%s\n", msg->data);
  exit(1);
}

int req_handle_scanf(ChMessage *msg,SOCKET fd)
{
  char *fmt, *res, *p;

  fmt = msg->data;
  fmt[msg->len-1]=0;
  res = input_scanf_chars(fmt);
  p = res; while (*p) { if (*p=='\n') *p=' '; p++; }
  req_reply(fd, "scanf-data", res, strlen(res)+1);
  free(res);
  return REQ_OK;
}

int req_handler_dispatch(ChMessage *msg,SOCKET replyFd)
{
  char *cmd=msg->header.type;
  DEBUGF(("Got request '%s'\n",cmd,replyFd));
       if (strcmp(cmd,"ping")==0)       return REQ_OK;
#if CMK_CCS_AVAILABLE
  else if (strcmp(cmd,"reply_fw")==0)   req_ccs_reply_fw(msg);
#endif
  else if (strcmp(cmd,"print")==0)      return req_handle_print(msg,replyFd);
  else if (strcmp(cmd,"printerr")==0)   return req_handle_printerr(msg,replyFd);
  else if (strcmp(cmd,"printsyn")==0)  return req_handle_printsyn(msg,replyFd);
  else if (strcmp(cmd,"printerrsyn")==0) return req_handle_printerrsyn(msg,replyFd);
  else if (strcmp(cmd,"scanf")==0)      return req_handle_scanf(msg,replyFd);
  else if (strcmp(cmd,"ending")==0)     return req_handle_ending(msg,replyFd);
  else if (strcmp(cmd,"abort")==0)      return req_handle_abort(msg,replyFd);
  else {
        fprintf(stderr,"Charmrun> Bad control socket request '%s'\n",cmd); 
        abort();
  }
  return REQ_OK;
}

void req_serve_client(SOCKET fd)
{
  int status;
  ChMessage msg;
  DEBUGF(("Getting message from client...\n"));
  ChMessage_recv(fd,&msg);
  DEBUGF(("Message is '%s'\n",msg.header.type));
  status = req_handler_dispatch(&msg,fd);
  switch (status) 
  {
    case REQ_OK: break;
    case REQ_FAILED: 
        fprintf(stderr,"Charmrun> Error processing control socket request %s\n",msg.header.type); 
        abort();
        break;
  }
  ChMessage_free(&msg);
}


int ignore_socket_errors(int c,const char *m)
{/*Abandon on further socket errors during error shutdown*/
  exit(2);return -1;
}

/*A socket went bad somewhere!  Immediately disconnect,
which kills everybody.
*/
int socket_error_in_poll(int code,const char *msg)
{
	int i;
	skt_set_abort(ignore_socket_errors);
	fprintf(stderr,"Charmrun: error on request socket--\n"
			"%s\n",msg);
	for (i=0;i<req_nClients;i++)
		skt_close(req_clients[i]);
	exit(1);
	return -1;
}

/*
Wait for incoming requests on all client sockets,
and the CCS socket (if present).
*/
void req_poll()
{
  int status,i;
  fd_set  rfds;
  struct timeval tmo;

  skt_set_abort(socket_error_in_poll);

  tmo.tv_sec = 1;
  tmo.tv_usec = 0;
  FD_ZERO(&rfds);
  for (i=0;i<req_nClients;i++)
	FD_SET(req_clients[i],&rfds);
  if (CcsServer_fd()!=INVALID_SOCKET) FD_SET(CcsServer_fd(),&rfds);
  DEBUGF(("Req_poll: Calling select...\n"));
  status=select(FD_SETSIZE, &rfds, 0, 0, &tmo);
  DEBUGF(("Req_poll: Select returned %d...\n",status));

  if (status==0) return;/*Nothing to do-- timeout*/

  if (status<0) 
	socket_error_in_poll(1359,"Node program terminated unexpectedly!\n");
	
  for (i=0;i<req_nClients;i++)
	if (FD_ISSET(req_clients[i],&rfds))
	/*This client is ready to read*/
		do { req_serve_client(req_clients[i]); }
		while (1==skt_select1(req_clients[i],0));

  if (CcsServer_fd()!=INVALID_SOCKET)
	 if (FD_ISSET(CcsServer_fd(),&rfds)) {
		  DEBUGF(("Activity on CCS server port...\n"));
		  req_ccs_connect();
	 }
}



static unsigned int server_port;
static char server_addr[1024];/* IP address or hostname of charmrun*/
static SOCKET server_fd;

int client_connect_problem(int code,const char *msg)
{/*Called when something goes wrong during a client connect*/
	fprintf(stderr,"Charmrun> error %d attaching to node:\n"
		"%s\n",code,msg);
	exit(1);
	return -1;
}

/*Wait for all the clients to connect to our server port*/
void req_client_connect(void)
{
	int client;
	nodeinfo_allocate();
	req_nClients=nodetab_rank0_size;
	req_clients=(SOCKET *)malloc(req_nClients*sizeof(SOCKET));
	skt_set_abort(client_connect_problem);
	for (client=0;client<req_nClients;client++)
	{/*Wait for the next client to connect to our server port.*/
		unsigned int clientPort;/*These are actually ignored*/
		skt_ip_t clientIP;
		if (arg_verbose) printf("Charmrun> Waiting for %d-th client to connect.\n",client);
		if (0==skt_select1(server_fd,arg_timeout*1000))
			client_connect_problem(client,"Timeout waiting for node-program to connect");
		req_clients[client]=skt_accept(server_fd,&clientIP,&clientPort);
		if (req_clients[client]==SOCKET_ERROR) 
			client_connect_problem(client,"Failure in node accept");
		else 
		{ /*This client has just connected-- fetch his name and IP*/
			ChMessage msg;
			if (!skt_select1(req_clients[client],arg_timeout*1000))
			  client_connect_problem(client,"Timeout on IP request");
			ChMessage_recv(req_clients[client],&msg);
			req_handle_initnode(&msg,req_clients[client]);
			ChMessage_free(&msg);
		}
	}
        if (portOk == 0) exit(1);
	if (arg_verbose) printf("Charmrun> All clients connected.\n");
	for (client=0;client<req_nClients;client++)
	  req_handle_initnodetab(NULL,req_clients[client]);
	if (arg_verbose) printf("Charmrun> IP tables sent.\n");
}

/*Start the server socket the clients will connect to.*/
void req_start_server(void)
{
  skt_ip_t ip=skt_innode_my_ip();
  if (arg_local)
    /* local execution, use localhost always */
    strcpy(server_addr, "127.0.0.1");
  else if (arg_usehostname || skt_ip_match(ip,skt_lookup_ip("127.0.0.1")))
    /*Use symbolic host name as charmrun address*/
    gethostname(server_addr,sizeof(server_addr));
  else
    skt_print_ip(server_addr,ip);

  server_port = 0;
  server_fd=skt_server(&server_port);

  if (arg_verbose) {
    printf("Charmrun> Charmrun = %s, port = %d\n", server_addr, server_port);
  }
  
#if CMK_CCS_AVAILABLE
  if(arg_server == 1) CcsServer_new(NULL,&arg_server_port,arg_server_auth);
#endif
}

/****************************************************************************
 *
 *  The Main Program
 *
 ****************************************************************************/
void start_nodes_daemon(void);
void start_nodes_rsh(void);
#if CMK_SCYLD
void nodetab_init_for_scyld(void);
void start_nodes_scyld(void);
#endif
void start_nodes_local(char **envp);

static void fast_idleFn(void) {sleep(0);}
void finish_nodes(void);

int main(int argc, char **argv, char **envp)
{
  srand(time(0));
  skt_init();
  skt_set_idle(fast_idleFn);
  /* CrnSrand((int) time(0)); */
  /* notify charm developers that charm is in use */
  ping_developers();
  /* Compute the values of all constants */
  arg_init(argc, argv);
  if(arg_verbose) fprintf(stderr, "Charmrun> charmrun started...\n");
#if CMK_SCYLD
  /* check scyld configuration */
  nodetab_init_for_scyld();
#else
  /* Initialize the node-table by reading nodesfile */
  nodetab_init();
#endif

  /* Start the server port */
  req_start_server();
  
  /* Initialize the IO module */
  input_init();
  
  /* start the node processes */
  if (0!=getenv("CONV_DAEMON"))
    start_nodes_daemon();
  else
#if CMK_SCYLD
    start_nodes_scyld();
#else
    if (!arg_local)
      start_nodes_rsh();
    else
      start_nodes_local(envp);
#endif

  if(arg_verbose) fprintf(stderr, "Charmrun> node programs all started\n");

  /* Wait for all clients to connect */
#if !CMK_RSH_KILL
  finish_nodes();
#endif
  req_client_connect();
#if CMK_RSH_KILL
  kill_nodes();
#endif
  if(arg_verbose) fprintf(stderr, "Charmrun> node programs all connected\n");

  /* enter request-service mode */
  while (1) req_poll();
}

/*This little snippet creates a NETSTART 
environment variable entry for the given node #.
It uses the idiotic "return reference to static buffer"
string return idiom.
*/
char *create_netstart(int node)
{
  static char dest[1024];
  int port = 0;
#if CMK_USE_GM
  /* send myrinet port to node program */
  port = nodetab_dataport(node);
#endif
  sprintf(dest,"%d %s %d %d %d",node,server_addr,server_port,getpid()&0x7FFF, port);
  return dest;
}

/* The remainder of charmrun is only concerned with starting all
the node-programs, also known as charmrun clients.  We have to
start nodetab_rank0_size processes on the remote machines.
*/

/*Ask the converse daemon running on each machine to start the node-programs.*/
void start_nodes_daemon(void)
{
  taskStruct task;
  char argBuffer[5000];/*Buffer to hold assembled program arguments*/
  int i,nodeNumber;

  /*Set the parts of the task structure that will be the same for all nodes*/
  /*FIXME: The program path needs to come from the nodelist file*/
  strcpy(task.pgm,arg_nodeprog_a);
  /*Figure out the command line arguments (same for all PEs)*/
  argBuffer[0]=0;
  for (i=0;arg_argv[i];i++) 
  {
    strcat(argBuffer," ");
    strcat(argBuffer,arg_argv[i]);
  }
  task.argLength=ChMessageInt_new(strlen(argBuffer));
  /*FIXME: The run directory needs to come from nodelist file*/
  strcpy(task.cwd,arg_currdir_a);
  
  task.magic=ChMessageInt_new(DAEMON_MAGIC);

/*Start up the user program, by sending a message
  to PE 0 on each node.*/
  for (nodeNumber=0;nodeNumber<nodetab_rank0_size;nodeNumber++)
  {
    char statusCode='N';/*Default error code-- network problem*/
    int fd;
    int pe0=nodetab_rank0_table[nodeNumber];
    
	if (arg_verbose)
	  printf("Charmrun> Starting node program %d on '%s'.\n",nodeNumber,nodetab_name(pe0));

    sprintf(task.env,"NETSTART=%s",create_netstart(nodeNumber));

    /*Send request out to remote node*/
    fd = skt_connect(nodetab_ip(pe0),
		     DAEMON_IP_PORT,30);
    if (fd!=INVALID_SOCKET)
    {/*Contact!  Ask the daemon to start the program*/
      skt_sendN(fd, (const char *)&task, sizeof(task));
      skt_sendN(fd, (const char *)argBuffer, strlen(argBuffer));
      skt_recvN(fd, &statusCode,sizeof(char));
    }
    if (statusCode!='G')
    {/*Something went wrong--*/
      fprintf(stderr,"Error '%c' starting remote node program on %s--\n%s\n",
		  statusCode,nodetab_name(pe0),daemon_status2msg(statusCode));
	  exit(1);
    } else if (arg_verbose)
	  printf("Charmrun> Node program %d started.\n",nodeNumber);
  }
}

#if defined(_WIN32) && !defined(__CYGWIN__)
/*Sadly, interprocess communication on Win32 is quite
  different, so we can't use Rsh on win32 yet.  
  Fall back to the daemon.*/
void start_nodes_rsh() {start_nodes_daemon();}
void finish_nodes(void) {}

void envCat(char *dest,LPTSTR oldEnv)
{
  char *src=oldEnv;
  dest+=strlen(dest);//Advance to end of dest
  dest++;//Advance past terminating NULL character
  while ((*src)!='\0') {
    int adv=strlen(src)+1;//Length of newly-copied string plus NULL
    strcpy(dest,src);//Copy another environment string
    dest+=adv;//Advance past newly-copied string and NULL
    src+=adv;//Ditto for src
  }
  *dest='\0';//Paste on final terminating NULL character
  FreeEnvironmentStrings(oldEnv);
}


/* simple version of charmrun that avoids the rshd or charmd,   */
/* it spawn the node program just on local machine using exec. */
void start_nodes_local(char ** env)
{
  int ret, i;
  PROCESS_INFORMATION pi;     /* process Information for the process spawned */
  char **p;

  char environment[10000];/*Doubly-null terminated environment strings*/
  char cmdLine[10000];/*Program command line, including executable name*/
/*Command line too long.*/
/*
  if (strlen(pparam_argv[1])+strlen(args) > 10000) 
	return 0; 
*/
  strcpy(cmdLine,pparam_argv[1]);
  p = pparam_argv+2;
  while ((*p)!='\0') {
    strcat(cmdLine," ");
    strcat(cmdLine,*p);
    p++;
  }

  for (i=0; i<arg_requested_pes; i++)
  {
    STARTUPINFO si={0};         /* startup info for the process spawned */

    sprintf(environment, "NETSTART=%s",  create_netstart(i));
    /*Paste all system environment strings */
    envCat(environment,GetEnvironmentStrings());
  
    /* Initialise the security attributes for the process 
     to be spawned */
    si.cb = sizeof(si);   
    if (arg_verbose)
      printf("Charmrun> start %d node program on localhost.\n", i);
    ret = CreateProcess(NULL,	/* application name */
		    cmdLine,	/* command line */
		    NULL,/*&sa,*/		/* process SA */
		    NULL,/*&sa,*/		/* thread SA */
		    FALSE,	/* inherit flag */
#if 1
		    CREATE_NEW_PROCESS_GROUP|DETACHED_PROCESS, 
#else
		    CREATE_NEW_PROCESS_GROUP|CREATE_NEW_CONSOLE,
#endif
			/* creation flags */
		    environment,		/* environment block */
		    ".",			/* working directory */
		    &si,			/* startup info */
		    &pi);
 
    if (ret==0)
    {
      /*Something went wrong!  Look up the Windows error code*/
/*
      int error=GetLastError();
      char statusCode=daemon_err2status(error);
      fprintf(logfile,"******************* ERROR *****************\n"
	      "Error in creating process!\n"
	      "Error code = %ld-- %s\n\n\n", error,
	      daemon_status2msg(statusCode));
	  fflush(logfile);
*/
      int error=GetLastError();
      printf("startProcess failed to start process \"%s\" with status: %d\n", pparam_argv[1], error);
      exit(1) ;
    } 
  }
}

#elif CMK_SCYLD

void nodetab_init_for_scyld()
{
  int maxNodes, i, node;
  nodetab_host group;
  int tablesize;

  tablesize = arg_requested_pes;
  maxNodes = bproc_numnodes() + 1;
  if (maxNodes > tablesize) tablesize = maxNodes;
  nodetab_table=(nodetab_host**)malloc(tablesize*sizeof(nodetab_host*));
  nodetab_rank0_table=(int*)malloc(tablesize*sizeof(int));
  nodetab_max=tablesize;

  nodetab_reset(&group);

  /* check which slave nodes available */
  for (i=-1; i<maxNodes; i++) {
    char hostname[256];
    if (bproc_nodestatus(i) != bproc_node_up) continue;
    if (i!= -1 && i<arg_startpe) continue;
    sprintf(hostname, "%d", i);
    nodetab_makehost(hostname, &group);
    if (nodetab_rank0_size == arg_requested_pes) break;
  }
  if (nodetab_rank0_size == 0) {
    fprintf(stderr, "Charmrun> no slave node available!\n");
    exit (1);
  }
  if (arg_verbose)
    printf("Charmrun> There are %d slave nodes available.\n", nodetab_rank0_size);

  /* expand node table to arg_requested_pes */
  if (arg_requested_pes > nodetab_rank0_size) {
    int node = 0;
    int orig_size = nodetab_rank0_size;
    while (nodetab_rank0_size < arg_requested_pes) {
      nodetab_makehost(nodetab_name(node), &group);
      node++; if (node == orig_size) node = 0;
    }
  }
}

void start_nodes_scyld(void)
{
  char *envp[2];
  int i;

  envp[0] = (char *)malloc(256);
  envp[1] = 0;
  for (i=0; i<arg_requested_pes; i++)
  {
    int status = 0;
    int pid;
    int nodeno = atoi(nodetab_name(i));

    if (arg_verbose)
      printf("Charmrun> start node program on slave node: %d.\n", nodeno);
    sprintf(envp[0], "NETSTART=%s",  create_netstart(i));
    pid = 0;
    pid = fork();
    if (pid < 0) exit(1);
    if (pid == 0)
    {
      int fd, fd1 = dup(1);
      if (fd = open("/dev/null", O_RDWR)) {
        dup2(fd, 0); dup2(fd, 1); dup2(fd, 2);
      }
      if (nodeno == -1) {
        status = execve(pparam_argv[1], pparam_argv+1, envp);
        dup2(fd1, 1);
        printf("execve failed to start process \"%s\" with status: %d\n", pparam_argv[1], status);
      }
      else {
        status = bproc_execmove(nodeno, pparam_argv[1], pparam_argv+1, envp);
        dup2(fd1, 1);
        printf("bproc_execmove failed to start remote process \"%s\" with status: %d\n", pparam_argv[1], status);
      }
      kill(getppid(), 9);
      exit(1);
    }
  }
  free(envp[0]);
}
void finish_nodes(void) {}

#else
/*Unix systems can use Rsh normally*/
/********** RSH-ONLY CODE *****************************************/
/*                                                                          */
/* Rsh_etc                                                                  */
/*                                                                          */
/* this starts all the node programs.  It executes fully in the background. */
/*                                                                          */
/****************************************************************************/
#include <sys/wait.h>

extern char **environ;
void removeEnv(const char *doomedEnv)
{ /*Remove a value from the environment list*/
      char **oe, **ie;
      oe=ie=environ;
      while (*ie != NULL) {
        if (0!=strncmp(*ie,doomedEnv,strlen(doomedEnv)))
          *oe++ = *ie;
        ie++;
      }
      *oe=NULL;/*NULL-terminate list*/
}

int rsh_fork(int nodeno,const char *startScript)
{
  char *rshargv[6];
  int pid;
  int num=0;
  char *s, *e;

  s=nodetab_shell(nodeno); e=skipstuff(s);
  while (*s) {
    rshargv[num++]=substr(s, e);
    s = skipblanks(e); e = skipstuff(s);
  }
  rshargv[num++]=nodetab_name(nodeno);
  rshargv[num++]="-l";
  rshargv[num++]=nodetab_login(nodeno);
  rshargv[num++]="/bin/sh -f";
  rshargv[num++]=0;
  if (arg_verbose) printf("Charmrun> Starting %s %s -l %s %s\n",nodetab_shell(nodeno), nodetab_name(nodeno),nodetab_login(nodeno), rshargv[num-2]);
  
  pid = fork();
  if (pid < 0) 
  	{ perror("ERROR> starting rsh"); exit(1); }
  if (pid == 0)
  {/*Child process*/
      int i;
      int fdScript=open(startScript,O_RDONLY);
  /**/  unlink(startScript); /**/
      dup2(fdScript,0);/*Open script as standard input*/
      removeEnv("DISPLAY="); /*No DISPLAY disables ssh's slow X11 forwarding*/
      for(i=3; i<1024; i++) close(i);
      execvp(rshargv[0], rshargv);
      fprintf(stderr,"Charmrun> Couldn't find rsh program '%s'!\n",rshargv[0]);
      exit(1);
  }
  if (arg_verbose)
    fprintf(stderr,"Charmrun> rsh (%s:%dd) started\n",
    	nodetab_name(nodeno),nodeno);
  return pid;
}

void fprint_arg(FILE *f,char **argv)
{
  while (*argv) { 
  	fprintf(f," %s",*argv); 
  	argv++; 
  }
}
void rsh_Find(FILE *f,const char *program,const char *dest)
{
    fprintf(f,"Find %s\n",program);
    fprintf(f,"%s=$loc\n",dest,dest);
}
void rsh_script(FILE *f, int nodeno, int rank0no, char **argv)
{
  char *netstart;
  char *arg_nodeprog_r,*arg_currdir_r;
  char *dbg=nodetab_debugger(nodeno);
  char *host=nodetab_name(nodeno);
  int randno = rand();
#define CLOSE_ALL " < /dev/null 1> /dev/null 2> /dev/null &"

  fprintf(f, /*Echo: prints out status message*/
  	"Echo() {\n"
  	"  echo 'Charmrun rsh(%s.%d)>' $*\n"
  	"}\n",host,nodeno);
  fprintf(f, /*Exit: exits with return code*/
	"Exit() {\n"
	"  if [ $1 -ne 0 ]\n"
	"  then\n"
	"    Echo Exiting with error code $1\n"
	"  fi\n"
#if CMK_RSH_KILL /*End by killing ourselves*/
	"  sleep 5\n" /*Delay until any error messages are flushed*/
	"  kill -9 $$\n"
#else /*Exit normally*/
	"  exit $1\n"
#endif
	"}\n");
  fprintf(f, /*Find: locates a binary program in PATH, sets loc*/
  	"Find() {\n"
  	"  loc=''\n"
  	"  for dir in `echo $PATH | sed -e 's/:/ /g'`\n"
  	"  do\n"
  	"    test -f $dir/$1 && loc=$dir/$1\n"
  	"  done\n"
  	"  if [ \"x$loc\" = x ]\n"
  	"  then\n"
  	"    Echo $1 not found in your PATH \"($PATH)\"--\n"
  	"    Echo set your path in your ~/.charmrunrc\n"
  	"    Exit 1\n"
  	"  fi\n"
  	"}\n",host,nodeno);
  
  if (arg_verbose) fprintf(f,"Echo 'remote responding...'\n");
  
  fprintf(f,"test -f $HOME/.charmrunrc && . $HOME/.charmrunrc\n");
  if (arg_display)
    fprintf(f,"DISPLAY='%s';export DISPLAY\n",arg_display);
  netstart = create_netstart(rank0no);
  fprintf(f,"NETSTART='%s';export NETSTART\n",netstart);
  
  if (arg_verbose) {
    printf("Charmrun> Sending \"%s\" to client %d.\n", netstart, rank0no);
  }
  fprintf(f,"PATH=\"$PATH:/bin:/usr/bin:/usr/X/bin:/usr/X11/bin:/usr/local/bin:"
  	"/usr/X11R6/bin:/usr/openwin/bin\"\n");
  
  /* find the node-program */
  arg_nodeprog_r = pathfix(arg_nodeprog_a, nodetab_pathfixes(nodeno));
  
  /* find the current directory, relative version */
  arg_currdir_r = pathfix(arg_currdir_a, nodetab_pathfixes(nodeno));

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    rsh_Find(f,nodetab_xterm(nodeno),"F_XTERM");
    rsh_Find(f,"xrdb","F_XRDB");
    if(arg_verbose) fprintf(f,"Echo 'using xterm' $F_XTERM\n");
  }

  if (arg_debug || arg_debug_no_pause)
  {/*Look through PATH for debugger*/
    rsh_Find(f,dbg,"F_DBG");
    if (arg_verbose) fprintf(f,"Echo 'using debugger' $F_DBG\n");
  }

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    fprintf(f,"$F_XRDB -query > /dev/null\n");
    fprintf(f,"if test $? != 0\nthen\n");
    fprintf(f,"  Echo 'Cannot contact X Server '$DISPLAY'.  You probably'\n");
    fprintf(f,"  Echo 'need to run xhost to authorize connections.'\n");
    fprintf(f,"  Echo '(See manual for xhost for security issues)'\n");
    fprintf(f,"  Exit 1\n");
    fprintf(f,"fi\n");
  }
  
  fprintf(f,"if test ! -x %s\nthen\n",arg_nodeprog_r);
  fprintf(f,"  Echo 'Cannot locate this node-program:'\n");
  fprintf(f,"  Echo '%s'\n",arg_nodeprog_r);
  fprintf(f,"  Exit 1\n");
  fprintf(f,"fi\n");
  
  fprintf(f,"cd %s\n",arg_currdir_r);
  fprintf(f,"if test $? = 1\nthen\n");
  fprintf(f,"  Echo 'Cannot propagate this current directory:'\n"); 
  fprintf(f,"  Echo '%s'\n",arg_currdir_r);
  fprintf(f,"  Exit 1\n");
  fprintf(f,"fi\n");
  
  if (strcmp(nodetab_setup(nodeno),"*")) {
    fprintf(f,"%s\n",nodetab_setup(nodeno));
    fprintf(f,"if test $? = 1\nthen\n");
    fprintf(f,"  Echo 'this initialization command failed:'\n");
    fprintf(f,"  Echo '\"%s\"'\n",nodetab_setup(nodeno));
    fprintf(f,"  Echo 'edit your nodes file to fix it.'\n");
    fprintf(f,"  Exit 1\n");
    fprintf(f,"fi\n");
  }

  if(arg_verbose) fprintf(f,"Echo 'starting node-program...'\n");  
  if (arg_debug || arg_debug_no_pause ) {
	 if ( strcmp(dbg, "gdb") == 0 ) {
           fprintf(f,"cat > /tmp/gdb%08x << END_OF_SCRIPT\n",randno);
           fprintf(f,"shell /bin/rm -f /tmp/gdb%08x\n",randno);
           fprintf(f,"handle SIGPIPE nostop noprint\n");
           fprintf(f,"handle SIGWINCH nostop noprint\n");
           fprintf(f,"handle SIGWAITING nostop noprint\n");
           fprintf(f,"set args");
           fprint_arg(f,argv);
           fprintf(f,"\n");
           if (arg_debug_no_pause) fprintf(f,"run\n");
           fprintf(f,"END_OF_SCRIPT\n");
           fprintf(f,"$F_XTERM");
           fprintf(f," -title 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
           fprintf(f," -e $F_DBG %s -x /tmp/gdb%08x"CLOSE_ALL"\n",
           	arg_nodeprog_r,randno);
         } else if ( strcmp(dbg, "dbx") == 0 ) {
           fprintf(f,"cat > /tmp/dbx%08x << END_OF_SCRIPT\n",randno);
           fprintf(f,"sh /bin/rm -f /tmp/dbx%08x\n",randno);
           fprintf(f,"dbxenv suppress_startup_message 5.0\n");
           fprintf(f,"ignore SIGPOLL\n");
           fprintf(f,"ignore SIGPIPE\n");
           fprintf(f,"ignore SIGWINCH\n");
           fprintf(f,"ignore SIGWAITING\n");
           fprintf(f,"END_OF_SCRIPT\n");
           fprintf(f,"$F_XTERM");
           fprintf(f," -title 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
           fprintf(f," -e $F_DBG %s ",arg_debug_no_pause?"-r":"");
	   if(arg_debug) {
              fprintf(f,"-c \'runargs ");
              fprint_arg(f,argv);
              fprintf(f,"\' ");
	   }
	   fprintf(f, "-s/tmp/dbx%08x %s",randno,arg_nodeprog_r);
	   if(arg_debug_no_pause) 
              fprint_arg(f,argv);
           fprintf(f,CLOSE_ALL "\n");
	 } else { 
	  fprintf(stderr, "Unknown debugger: %s.\n Exiting.\n", 
	    nodetab_debugger(nodeno));
	 }
  } else if (arg_in_xterm) {
    if(arg_verbose)
      fprintf(stderr, "Charmrun> node %d: xterm is %s\n", 
              nodeno, nodetab_xterm(nodeno));
    fprintf(f,"cat > /tmp/inx%08x << END_OF_SCRIPT\n", randno);
    fprintf(f,"#!/bin/sh\n");
    fprintf(f,"/bin/rm -f /tmp/inx%08x\n",randno);
    fprintf(f,"%s", arg_nodeprog_r);
    fprint_arg(f,argv);
    fprintf(f,"\n");
    fprintf(f,"echo 'program exited with code '\\$?\n");
    fprintf(f,"read eoln\n");
    fprintf(f,"END_OF_SCRIPT\n");
    fprintf(f,"chmod 700 /tmp/inx%08x\n", randno);
    fprintf(f,"$F_XTERM -title 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
    fprintf(f," -sl 5000");
    fprintf(f," -e /tmp/inx%08x", randno);
    fprintf(f,CLOSE_ALL "\n");
  } else {
    fprintf(f,"%s",arg_nodeprog_r);
    fprint_arg(f,argv);
    fprintf(f,CLOSE_ALL "\n");
  }
  if (arg_verbose) fprintf(f,"Echo 'rsh phase successful.'\n");
  fprintf(f,"Exit 0\n");
}

int *rsh_pids=NULL;
void start_nodes_rsh()
{
  int rank0no;
  rsh_pids=(int *)malloc(sizeof(int)*nodetab_rank0_size);
  /*Start up the user program, by sending a message
  to PE 0 on each node.*/
  for (rank0no=0;rank0no<nodetab_rank0_size;rank0no++)
  {
     int pe=nodetab_rank0_table[rank0no];
     FILE *f;
     char startScript[200];
     sprintf(startScript,"/tmp/charmrun.%d.%d",getpid(),pe);
     f=fopen(startScript,"w");
     rsh_script(f,pe,rank0no,arg_argv);
     fclose(f);
     rsh_pids[rank0no]=rsh_fork(pe,startScript);
  }
}

void finish_nodes()
{
  int rank0no;
  if (!rsh_pids) return; /*nothing to do*/
  /*Now wait for all the rsh'es to finish*/
  for (rank0no=0;rank0no<nodetab_rank0_size;rank0no++)
  {
     const char *host=nodetab_name(nodetab_rank0_table[rank0no]);
     int status=0;
     if (arg_verbose) printf("Charmrun> waiting for rsh (%s:%d), pid %d\n",
		host,rank0no,rsh_pids[rank0no]);
     do {
     	waitpid(rsh_pids[rank0no],&status,0);
     } while (!WIFEXITED(status));
     if (WEXITSTATUS(status)!=0)
     {
     	fprintf(stderr,"Charmrun> Error %d returned from rsh (%s:%d)\n",
     		WEXITSTATUS(status),host,rank0no);
     	exit(1);
     }     
  }
  free(rsh_pids);
}

void kill_nodes()
{
  int rank0no;
  if (!rsh_pids) return; /*nothing to do*/
  /*Now wait for all the rsh'es to finish*/
  for (rank0no=0;rank0no<nodetab_rank0_size;rank0no++)
  {
     const char *host=nodetab_name(nodetab_rank0_table[rank0no]);
     int status=0;
     if (arg_verbose) printf("Charmrun> waiting for rsh (%s:%d), pid %d\n",
		host,rank0no,rsh_pids[rank0no]);
     kill(rsh_pids[rank0no],9);
     waitpid(rsh_pids[rank0no],&status,0); /*<- no zombies*/
  }
  free(rsh_pids);
}

/* simple version of charmrun that avoids the rshd or charmd,   */
/* it spawn the node program just on local machine using exec. */
void start_nodes_local(char ** env)
{
  char **envp;
  int i, envc;

  /* copy environ and expanded to hold NETSTART */ 
  for (envc=0; env[envc]; envc++);  envc--;
  envp = (char **)malloc((envc+2)*sizeof(void *));
  for (i=0; i<envc; i++) envp[i] = env[i];
  envp[envc] = (char *)malloc(256);
  envp[envc+1] = 0;

  for (i=0; i<arg_requested_pes; i++)
  {
    int status = 0;
    int pid;

    if (arg_verbose)
      printf("Charmrun> start %d node program on localhost.\n", i);
    sprintf(envp[envc], "NETSTART=%s",  create_netstart(i));
    pid = 0;
    pid = fork();
    if (pid < 0) exit(1);
    if (pid == 0)
    {
      int fd, fd1 = dup(1);
      if (fd = open("/dev/null", O_RDWR)) {
        dup2(fd, 0); dup2(fd, 1); dup2(fd, 2);
      }
      status = execve(pparam_argv[1], pparam_argv+1, envp);
      dup2(fd1, 1);
      printf("execve failed to start process \"%s\" with status: %d\n", pparam_argv[1], status);
      kill(getppid(), 9);
      exit(1);
    }
  }
  free(envp[envc]);
  free(envp);
}

#endif /*CMK_USE_RSH*/




