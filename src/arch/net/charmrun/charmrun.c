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

/**************************************************************************
 *
 * ping_developers
 *
 * Sends a single UDP packet to the charm developers notifying them
 * that charm is in use.
 *
 **************************************************************************/

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

void ping_developers()
{
#ifdef NOTIFY
  char               info[1000];
  struct sockaddr_in addr=skt_build_addr(0x80aef1d3,6571);
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
  static char result[100];
  char *e, *p;
  
  e = getenv("DISPLAY");
  if (e==0) return NULL;
  p = strrchr(e, ':');
  if (p==0) return NULL;
  if ((e[0]==':')||(strncmp(e,"unix:",5)==0)) {
    sprintf(result,"%s:%s",text_ip(skt_my_ip()),p+1);
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
int   arg_server_port;

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
#endif
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

  if (pparam_parsecmd('+', argv) < 0) {
    fprintf(stderr,"ERROR> syntax: %s\n",pparam_error);
    pparam_printdocs();
    exit(1);
  }
  arg_argv = argv+2; /*Skip over conv-host (0) and program name (1)*/
  arg_argc = pparam_countargs(arg_argv);

  arg_verbose = arg_verbose || arg_debug || arg_debug_no_pause;

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
  if (argc<2) {
    fprintf(stderr,"ERROR> You must specify a node-program.\n");
    exit(1);
  }
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
  nodetab_tempName=tmpnam(buffer);
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
  unsigned int ip; /*IP address of host*/
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
  h->ip=0;
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
  *nodetab_table[nodetab_size++] = *h;
}

void nodetab_makehost(char *name,nodetab_host *h)
{
  h->name=strdup(name);
  h->ip = skt_lookup_ip(name);
  if (h->ip==-1) {
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

void nodetab_init()
{
  FILE *f,*fopen();
  char *nodesfile; 
  nodetab_host global,group,host;
  char input_line[MAX_LINE_LENGTH];
  int rightgroup, basicsize, i, remain;
  
  /* Open the NODES_FILE. */
  nodesfile = nodetab_file_find();
  if(arg_verbose)
    fprintf(stderr, "Conv-host> using %s as nodesfile\n", nodesfile);
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
			if (rightgroup) {
				host=group;
				nodetab_args(b3,&host);
				for (host.rank=0; host.rank<host.cpus; host.rank++)
					nodetab_makehost(substr(b2,e2),&host);
			}
		} else if (subeqs(b1,e1, "group")) {
			group=global;
			nodetab_args(b3,&group);
			rightgroup = subeqs(b2,e2,arg_nodegroup);
		} else {
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
  if ((i<0)||(i>=nodetab_size)) {
    fprintf(stderr,"ERROR> No such node %d\n",i);
    exit(1);
  }
  return nodetab_table[i];
}

char        *nodetab_name(int i)     { return nodetab_getinfo(i)->name; }
pathfixlist  nodetab_pathfixes(int i){ return nodetab_getinfo(i)->pathfixes; }
char        *nodetab_ext(int i)      { return nodetab_getinfo(i)->ext; }
unsigned int nodetab_ip(int i)       { return nodetab_getinfo(i)->ip; }
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

typedef struct {
	ChMessageInt_t nPE,IP,dataport;
} single_nodeinfo;

static single_nodeinfo *nodeinfo_arr;/*Indexed by node number.*/

void nodeinfo_allocate(void)
{
	nodeinfo_arr=(single_nodeinfo *)malloc(nodetab_rank0_size*sizeof(single_nodeinfo));
}
void nodeinfo_add(const ChMessageInt_t *nodeInfo,SOCKET ctrlfd)
{
	single_nodeinfo i;
	int node=ChMessageInt(nodeInfo[0]);
	int nt,pe;
	if (node<0 || node>=nodetab_rank0_size)
		{fprintf(stderr,"Unexpected node %d registered!\n",node);exit(1);}
	nt=nodetab_rank0_table[node];/*Nodetable index for this node*/
	i.nPE=ChMessageInt_new(nodetab_cpus(nt));
	i.IP=ChMessageInt_new(nodetab_ip(nt));
	i.dataport=nodeInfo[1];
	nodeinfo_arr[node]=i;
	for (pe=0;pe<nodetab_cpus(nt);pe++)
	  {
	    nodetab_table[nt+pe]->dataport=ChMessageInt(i.dataport);
	    nodetab_table[nt+pe]->ctrlfd=ctrlfd;
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
  Conv-host forwards CCS requests on to the node-programs' control
sockets.
***************************************************************************/

#if CMK_CCS_AVAILABLE

/*The Ccs Server socket became active-- 
rec'v the message and respond to the request,
by forwarding the request to the appropriate node.
 */
void req_ccs_connect(void)
{
  ChMessageHeader ch;/*Make a conv-host header*/
  CcsImplHeader hdr;/*Ccs internal header*/
  void *reqData;/*CCS request data*/
  int pe,reqBytes;
  if (0==CcsServer_recvRequest(&hdr,&reqData))
    return;/*Malformed request*/
  pe=ChMessageInt(hdr.pe);
  reqBytes=ChMessageInt(hdr.len);

  /*Fill out the conv-host header & forward the CCS request*/
  ChMessageHeader_new("req_fw",sizeof(hdr)+reqBytes,&ch);
  skt_sendN(nodetab_ctrlfd(pe),&ch,sizeof(ch));
  skt_sendN(nodetab_ctrlfd(pe),&hdr,sizeof(hdr));
  skt_sendN(nodetab_ctrlfd(pe),reqData,reqBytes);
  free(reqData);
}

/*
Forward the CCS reply (if any) back to the original requestor,
on the original request socket.
 */
void req_ccs_reply_fw(ChMessage *msg) {
  SOCKET fd=(SOCKET)ChMessageInt(*(ChMessageInt_t *)msg->data);
  CcsServer_sendReply(fd,msg->len-sizeof(ChMessageInt_t),
		      msg->data+sizeof(ChMessageInt_t));
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

/* This is the only place where conv-host talks back to anyone. 
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
  if (msg->len!=2*sizeof(ChMessageInt_t)) {
    fprintf(stderr,"Conv-host: Bad initnode data. Aborting\n");
    exit(1);
  }
  nodeinfo_add((ChMessageInt_t *)msg->data,fd);
  return REQ_OK;
}

/*Get the array of node numbers, IPs, and ports.
This is used by the node-programs to talk to one another.
*/
int req_handle_initnodetab(ChMessage *msg,SOCKET fd)
{
	ChMessageHeader hdr;
	ChMessageInt_t nNodes=ChMessageInt_new(nodetab_rank0_size);
	ChMessageHeader_new("initnodetab",sizeof(ChMessageInt_t)*(1+3*nodetab_rank0_size),
		&hdr);
	skt_sendN(fd,(const char *)&hdr,sizeof(hdr));
	skt_sendN(fd,(const char *)&nNodes,sizeof(nNodes));
	skt_sendN(fd,(const char *)nodeinfo_arr,
		  sizeof(ChMessageInt_t)*3*nodetab_rank0_size);
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


int req_handle_ending(ChMessage *msg,SOCKET fd)
{  
  int i;
  req_ending++;
    
  if (req_ending == nodetab_size)
  {
    for (i=0;i<req_nClients;i++)
      skt_close(req_clients[i]);
    if (arg_verbose) printf("Conv-host> Graceful exit.\n");
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
  else if (strcmp(cmd,"scanf")==0)      return req_handle_scanf(msg,replyFd);
  else if (strcmp(cmd,"ending")==0)     return req_handle_ending(msg,replyFd);
  else if (strcmp(cmd,"abort")==0)      return req_handle_abort(msg,replyFd);
  else {
        fprintf(stderr,"conv-host> Bad control socket request %s\n",cmd); 
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
        fprintf(stderr,"conv-host> Error processing control socket request %s\n",msg.header.type); 
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
	fprintf(stderr,"Conv-host: error on request socket--\n"
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
		  req_serve_client(req_clients[i]);

  if (CcsServer_fd()!=INVALID_SOCKET)
	 if (FD_ISSET(CcsServer_fd(),&rfds)) {
		  DEBUGF(("Activity on CCS server port...\n"));
		  req_ccs_connect();
	   }
}



static unsigned int server_ip,server_port;
static SOCKET server_fd;

int client_connect_problem(int code,const char *msg)
{/*Called when something goes wrong during a client connect*/
	fprintf(stderr,"conv-host> error attaching to node %d:\n"
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
		unsigned int clientIP,clientPort;/*These are actually ignored*/
		if (arg_verbose) printf("Conv-host> Waiting for client %d to connect.\n",client);
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
	if (arg_verbose) printf("Conv-host> All clients connected.\n");
	for (client=0;client<req_nClients;client++)
	  req_handle_initnodetab(NULL,req_clients[client]);
	if (arg_verbose) printf("Conv-host> IP tables sent.\n");
}

/*Start the server socket the clients will connect to.*/
void req_start_server(void)
{
  server_port = 0;
  server_ip=skt_my_ip();
  server_fd=skt_server(&server_port);
  DEBUGF(("Conv-host control IP = %d, port = %d\n", server_ip, server_port));
  
#if CMK_CCS_AVAILABLE
  if(arg_server == 1) CcsServer_new(NULL,&arg_server_port);
#endif
}

/****************************************************************************
 *
 *  The Main Program
 *
 ****************************************************************************/
void start_nodes_daemon(void);
void start_nodes_rsh(void);

static void fast_idleFn(void) {sleep(0);}

int main(int argc, char **argv)
{
  srand(time(0));
  skt_init();
  skt_set_idle(fast_idleFn);
  /* CrnSrand((int) time(0)); */
  /* notify charm developers that charm is in use */
  ping_developers();
  /* Compute the values of all constants */
  arg_init(argc, argv);
  if(arg_verbose) fprintf(stderr, "Conv-host> conv-host started...\n");
  /* Initialize the node-table by reading nodesfile */
  nodetab_init();

  /* Start the server port */
  req_start_server();
  
  /* Initialize the IO module */
  input_init();
  
  /* start the node processes */
  if (0!=getenv("CONV-DAEMON"))
    start_nodes_daemon();
  else
    start_nodes_rsh();

  if(arg_verbose) fprintf(stderr, "Conv-host> node programs all started\n");

  /* Wait for all clients to connect */
  req_client_connect();
  if(arg_verbose) fprintf(stderr, "Conv-host> node programs all connected\n");

  /* enter request-service mode */
  while (1) req_poll();
}

/*This little snippet creates a NETSTART 
environment variable entry for the given node #.
It uses the idiotic "return reference to static buffer"
string return idiom.
*/
const char *create_netstart(int node)
{
  static char dest[80];
  sprintf(dest,"%d %d %d %d",node,server_ip,server_port,getpid()&0x7FFF);
  return dest;
}

/* The remainder of conv-host is only concerned with starting all
the node-programs, also known as conv-host clients.  We have to
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
	  printf("Conv-host> Starting node program %d on '%s'.\n",nodeNumber,nodetab_name(pe0));

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
	  printf("Conv-host> Node program %d started.\n",nodeNumber);
  }
}

#if defined(_WIN32) && !defined(__CYGWIN__)
/*Sadly, interprocess communication on Win32 is quite
  different, so we can't use Rsh on win32 yet.  
  Fall back to the daemon.*/
void start_nodes_rsh(void) {start_nodes_daemon();}

#else
/*Unix systems can use Rsh normally*/
/************* RSH-ONLY CODE ***************************
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
  int uspace, needed; char *nbuf;
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
  int uspace, needed; char *nbuf;
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


#if 0
void xstr_printf(xstr l, char *fmt, ...)
{
  va_list p;
  char buffer[10000];
  va_start(p, fmt);
  vsprintf(buffer, fmt, p);
  xstr_write(l, buffer, strlen(buffer));
  va_end(p);
}
#else
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
#endif

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

/******** RSH-ONLY CODE ****************************************************
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

#include <sys/wait.h>

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
  
  if (ibuf==0) return -1;
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
  int status=0;
  prog_flush(c);
  kill(c->pid,SIGKILL);
  waitpid(c->pid,&status,0);
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
      for(i=3; i<1024; i++) close(i);
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

/********** RSH-ONLY CODE *****************************************/
/*                                                                          */
/* Rsh_etc                                                                  */
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
  rshargv[4]="exec /bin/sh -f";
#if CMK_CONV_HOST_WANT_CSH
  rshargv[4]="exec /bin/csh -f";
#endif
  rshargv[5]=0;
  if (arg_verbose) printf("Conv-host> Starting %s %s -l %s %s\n",nodetab_shell(nodeno), nodetab_name(nodeno),nodetab_login(nodeno), rshargv[4]);

  rsh = prog_start(nodetab_shell(nodeno), rshargv, 1);
  if ((rsh==0)&&(errno!=EMFILE)) { perror("ERROR> starting rsh"); exit(1); }
  if (rsh==0)
    {
      fprintf(stderr,"caution: cannot start specified number of rsh's\n");
      fprintf(stderr,"(not enough file descriptors available?)\n");
    }
  if (rsh && arg_verbose)
    fprintf(stderr,"Conv-host> node %d: rsh initiated...\n",nodeno);
  return rsh;
}


void rsh_pump_sh(p, nodeno, rank0no, argv)
    prog p; int nodeno, rank0no; char **argv;
{
  char *arg_nodeprog_r,*arg_currdir_r;
  char *dbg=nodetab_debugger(nodeno);
  xstr ibuf = p->ibuf;
  int randno = rand();
  /* int randno = CrnRand(); */
  
  xstr_printf(ibuf,"echo 'remote responding...'\n");

  xstr_printf(ibuf,"test -f ~/.conv-hostrc && . ~/.conv-hostrc\n");
  if (arg_display)
    xstr_printf(ibuf,"DISPLAY=%s;export DISPLAY\n",arg_display);
  xstr_printf(ibuf,"NETSTART='%s';export NETSTART\n",create_netstart(rank0no));
  prog_flush(p);

  /* find the node-program */
  arg_nodeprog_r = pathfix(arg_nodeprog_a, nodetab_pathfixes(nodeno));
  
  /* find the current directory, relative version */
  arg_currdir_r = pathfix(arg_currdir_a, nodetab_pathfixes(nodeno));

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    xstr_printf(ibuf,"for dir in `echo $PATH | sed -e 's/:/ /g'`; do\n");
    xstr_printf(ibuf,"  test -f $dir/%s && F_XTERM=$dir/%s && export F_XTERM\n", 
                     nodetab_xterm(nodeno), nodetab_xterm(nodeno));
    xstr_printf(ibuf,"  test -f $dir/xrdb && F_XRDB=$dir/xrdb && export F_XRDB\n");
    xstr_printf(ibuf,"done\n");
    xstr_printf(ibuf,"if test -z \"$F_XTERM\";  then\n");
    xstr_printf(ibuf,"   echo '%s not in path --- set your path in your ~/.conv-hostrc or profile.'\n", nodetab_xterm(nodeno));
    xstr_printf(ibuf,"   test -f /bin/sync && /bin/sync\n");
    xstr_printf(ibuf,"   exit 1\n");
    xstr_printf(ibuf,"fi\n");
    xstr_printf(ibuf,"if test -z \"$F_XRDB\"; then\n");
    xstr_printf(ibuf,"   echo 'xrdb not in path - set your path in your ~/.conv-hostrc or profile.'\n");
    xstr_printf(ibuf,"   test -f /bin/sync && /bin/sync\n");
    xstr_printf(ibuf,"   exit 1\n");
    xstr_printf(ibuf,"fi\n");
    if(arg_verbose) xstr_printf(ibuf,"echo 'using xterm' $F_XTERM\n");
    prog_flush(p);
  }

  if (arg_debug || arg_debug_no_pause)
  	{
          xstr_printf(ibuf,"for dir in `echo $PATH | sed -e 's/:/ /g'`; do\n");
          xstr_printf(ibuf,"  test -f $dir/%s && F_DBG=$dir/%s && export F_DBG\n",dbg,dbg);
          xstr_printf(ibuf,"done\n");
          xstr_printf(ibuf,"if -z \"$F_DBG\"; then\n");
          xstr_printf(ibuf,"   echo '%s not in path - set your path in your cshrc.'\n",dbg);
          xstr_printf(ibuf,"   test -f /bin/sync && /bin/sync\n");
          xstr_printf(ibuf,"   exit 1\n");
          xstr_printf(ibuf,"fi\n");
          prog_flush(p);
       }

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    xstr_printf(ibuf,"$F_XRDB -query > /dev/null\n");
    xstr_printf(ibuf,"if test $? != 0; then\n");
    xstr_printf(ibuf,"  echo 'Cannot contact X Server '$DISPLAY'.  You probably'\n");
    xstr_printf(ibuf,"  echo 'need to run xhost to authorize connections.'\n");
    xstr_printf(ibuf,"  echo '(See manual for xhost for security issues)'\n");
    xstr_printf(ibuf,"  test -f /bin/sync && /bin/sync\n");
    xstr_printf(ibuf,"  exit 1\n");
    xstr_printf(ibuf,"fi\n");
    prog_flush(p);
  }
  
  xstr_printf(ibuf,"if test ! -x %s; then\n",arg_nodeprog_r);
  xstr_printf(ibuf,"  echo 'Cannot locate this node-program:'\n");
  xstr_printf(ibuf,"  echo '%s'\n",arg_nodeprog_r);
  xstr_printf(ibuf,"  test -f /bin/sync && /bin/sync\n");
  xstr_printf(ibuf,"  exit 1\n");
  xstr_printf(ibuf,"fi\n");
  
  xstr_printf(ibuf,"cd %s\n",arg_currdir_r);
  xstr_printf(ibuf,"if test $? = 1; then\n");
  xstr_printf(ibuf,"  echo 'Cannot propagate this current directory:'\n"); 
  xstr_printf(ibuf,"  echo '%s'\n",arg_currdir_r);
  xstr_printf(ibuf,"  test -f /bin/sync && /bin/sync\n");
  xstr_printf(ibuf,"  exit 1\n");
  xstr_printf(ibuf,"fi\n");
  
  if (strcmp(nodetab_setup(nodeno),"*")) {
    xstr_printf(ibuf,"cd .\n");
    xstr_printf(ibuf,"%s\n",nodetab_setup(nodeno));
    xstr_printf(ibuf,"if test $? = 1; then\n");
    xstr_printf(ibuf,"  echo 'this initialization command failed:'\n");
    xstr_printf(ibuf,"  echo '\"%s\"'\n",nodetab_setup(nodeno));
    xstr_printf(ibuf,"  echo 'edit your nodes file to fix it.'\n");
    xstr_printf(ibuf,"  test -f /bin/sync && /bin/sync\n");
    xstr_printf(ibuf,"  exit 1\n");
    xstr_printf(ibuf,"fi\n");
  }

  if(arg_verbose) xstr_printf(ibuf,"echo 'starting node-program...'\n");  
  if (arg_debug || arg_debug_no_pause ) {
	 if ( strcmp(dbg, "gdb") == 0 ) {
           xstr_printf(ibuf,"cat > /tmp/gdb%08x << END_OF_SCRIPT\n",randno);
           xstr_printf(ibuf,"shell /bin/rm -f /tmp/gdb%08x\n",randno);
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
             xstr_printf(ibuf," -e $F_DBG %s -x /tmp/gdb%08x",arg_nodeprog_r,randno);
             xstr_printf(ibuf," < /dev/null 2> /dev/null &");
             xstr_printf(ibuf,"\n");
           }
        } else if ( strcmp(dbg, "dbx") == 0 ) {
          xstr_printf(ibuf,"cat > /tmp/dbx%08x << END_OF_SCRIPT\n",randno);
          xstr_printf(ibuf,"sh /bin/rm -f /tmp/dbx%08x\n",randno);
          xstr_printf(ibuf,"dbxenv suppress_startup_message 5.0\n");
          xstr_printf(ibuf,"ignore SIGPOLL\n");
          xstr_printf(ibuf,"ignore SIGPIPE\n");
          xstr_printf(ibuf,"ignore SIGWINCH\n");
          xstr_printf(ibuf,"ignore SIGWAITING\n");
          xstr_printf(ibuf,"END_OF_SCRIPT\n");
          if( arg_debug || arg_debug_no_pause){
            xstr_printf(ibuf,"$F_XTERM");
            xstr_printf(ibuf," -title 'Node %d (%s)' ",nodeno,nodetab_name(nodeno));
            xstr_printf(ibuf," -e $F_DBG %s ",arg_debug_no_pause?"-r":"");
	    if(arg_debug) {
              xstr_printf(ibuf,"-c \'runargs ");
              while (*argv) { xstr_printf(ibuf,"%s ",*argv); argv++; }
              xstr_printf(ibuf,"\' ");
	    }
	    xstr_printf(ibuf, "-s/tmp/dbx%08x %s",randno,arg_nodeprog_r);
	    if(arg_debug_no_pause) {
              while (*argv) { xstr_printf(ibuf," %s",*argv); argv++; }
	    }
            xstr_printf(ibuf," < /dev/null 2> /dev/null &");
            xstr_printf(ibuf,"\n");
          }
	} else { 
	  fprintf(stderr, "Unknown debugger: %s.\n Exiting.\n", 
	    nodetab_debugger(nodeno));
	  exit(1);
	}
  } else if (arg_in_xterm) {
    if(arg_verbose) {
      fprintf(stderr, "Conv-host> node %d: xterm is %s\n", 
              nodeno, nodetab_xterm(nodeno));
    }
    xstr_printf(ibuf,"cat > /tmp/inx%08x << END_OF_SCRIPT\n", randno);
    xstr_printf(ibuf,"#!/bin/sh\n");
    xstr_printf(ibuf,"/bin/rm -f /tmp/inx%08x\n",randno);
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
    xstr_printf(ibuf," < /dev/null 2> /dev/null &");
    xstr_printf(ibuf,"\n");
  } else {
    xstr_printf(ibuf,"%s",arg_nodeprog_r);
    while (*argv) { xstr_printf(ibuf," %s",*argv); argv++; }
    xstr_printf(ibuf," < /dev/null 2> /dev/null &");
    xstr_printf(ibuf,"\n");
  }
  
  xstr_printf(ibuf,"echo 'rsh phase successful.'\n");
  xstr_printf(ibuf,"test -f /bin/sync && /bin/sync\n");
  xstr_printf(ibuf,"exit 0\n");
  prog_flush(p);
  
}

#if CMK_CONV_HOST_WANT_CSH
void rsh_pump_csh(p, nodeno, rank0no, argv)
    prog p; int nodeno, rank0no; char **argv;
{
  char *arg_nodeprog_r,*arg_currdir_r;
  char *dbg=nodetab_debugger(nodeno);
  xstr ibuf = p->ibuf;
  int randno = rand();
  /* int randno = CrnRand(); */
  
  xstr_printf(ibuf,"echo 'remote responding...'\n");

  xstr_printf(ibuf,"if ( -x ~/.conv-hostrc )   source ~/.conv-hostrc\n");
  if (arg_display)
    xstr_printf(ibuf,"setenv DISPLAY %s\n",arg_display);
  xstr_printf(ibuf,"setenv NETSTART '%s'\n",create_netstart(rank0no));
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
    xstr_printf(ibuf,"   exit 1\n");
    xstr_printf(ibuf,"endif\n");
    xstr_printf(ibuf,"if ($?F_XRDB == 0) then\n");
    xstr_printf(ibuf,"   echo 'xrdb not in path - set your path in your cshrc.'\n");
    xstr_printf(ibuf,"   exit 1\n");
    xstr_printf(ibuf,"endif\n");
    if(arg_verbose) xstr_printf(ibuf,"echo 'using xterm' $F_XTERM\n");
    prog_flush(p);
  }

  if (arg_debug || arg_debug_no_pause)
  	{
	  xstr_printf(ibuf,"foreach dir ($path)\n");
          xstr_printf(ibuf,"  if (-e $dir/%s) setenv F_DBG $dir/%s\n",dbg,dbg);
          xstr_printf(ibuf,"end\n");
          xstr_printf(ibuf,"if ($?F_DBG == 0) then\n");
          xstr_printf(ibuf,"   echo '%s not in path - set your path in your cshrc.'\n",dbg);
          xstr_printf(ibuf,"   exit 1\n");
          xstr_printf(ibuf,"endif\n");
          prog_flush(p);
       }

  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    xstr_printf(ibuf,"xrdb -query > /dev/null\n");
    xstr_printf(ibuf,"if ($status != 0) then\n");
    xstr_printf(ibuf,"  echo 'Cannot contact X Server '$DISPLAY'.  You probably'\n");
    xstr_printf(ibuf,"  echo 'need to run xhost to authorize connections.'\n");
    xstr_printf(ibuf,"  echo '(See manual for xhost for security issues)'\n");
    xstr_printf(ibuf,"  exit 1\n");
    xstr_printf(ibuf,"endif\n");
    prog_flush(p);
  }
  
  xstr_printf(ibuf,"if (! -x %s) then\n",arg_nodeprog_r);
  xstr_printf(ibuf,"  echo 'Cannot locate this node-program:'\n");
  xstr_printf(ibuf,"  echo '%s'\n",arg_nodeprog_r);
  xstr_printf(ibuf,"  exit 1\n");
  xstr_printf(ibuf,"endif\n");
  
  xstr_printf(ibuf,"cd %s\n",arg_currdir_r);
  xstr_printf(ibuf,"if ($status == 1) then\n");
  xstr_printf(ibuf,"  echo 'Cannot propagate this current directory:'\n"); 
  xstr_printf(ibuf,"  echo '%s'\n",arg_currdir_r);
  xstr_printf(ibuf,"  exit 1\n");
  xstr_printf(ibuf,"endif\n");
  
  if (strcmp(nodetab_setup(nodeno),"*")) {
    xstr_printf(ibuf,"cd .\n");
    xstr_printf(ibuf,"%s\n",nodetab_setup(nodeno));
    xstr_printf(ibuf,"if ($status == 1) then\n");
    xstr_printf(ibuf,"  echo 'this initialization command failed:'\n");
    xstr_printf(ibuf,"  echo '\"%s\"'\n",nodetab_setup(nodeno));
    xstr_printf(ibuf,"  echo 'edit your nodes file to fix it.'\n");
    xstr_printf(ibuf,"  exit 1\n");
    xstr_printf(ibuf,"endif\n");
  }

  if(arg_verbose) xstr_printf(ibuf,"echo 'starting node-program...'\n");  
  if (arg_debug || arg_debug_no_pause ) {
	 if ( strcmp(dbg, "gdb") == 0 ) {
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
             xstr_printf(ibuf," -e $F_DBG %s -x /tmp/gdb%08x",arg_nodeprog_r,randno);
             xstr_printf(ibuf," < /dev/null >& /dev/null &");
             xstr_printf(ibuf,"\n");
           }
        } else if ( strcmp(dbg, "dbx") == 0 ) {
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
            xstr_printf(ibuf," -e $F_DBG %s ",arg_debug_no_pause?"-r":"");
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
      fprintf(stderr, "Conv-host> node %d: xterm is %s\n", 
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
  
    xstr_printf(ibuf,"echo 'rsh phase successful.'\n");
    xstr_printf(ibuf,"exit 0\n");
  prog_flush(p);
  
}
#endif


void rsh_pump(p, nodeno, rank0no, argv)
    prog p; int nodeno, rank0no; char **argv;
{
#if CMK_CONV_HOST_WANT_CSH
  rsh_pump_csh(p, nodeno, rank0no, argv);
#else
  rsh_pump_sh(p, nodeno, rank0no, argv);
#endif
}

void start_nodes_rsh(void)
{
  prog        rsh_prog[200];
  int         rsh_node[200];
  int         rsh_nstarted;
  int         rsh_nfinished;
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
        line = xstr_gets(buffer, 999, p->obuf);
        if (line==0) break;
          if (strncmp(line,"[1] ",4)==0) continue;
          if (arg_verbose ||
              (strcmp(line,"rsh phase successful.")
               &&strcmp(line,"remote responding...")))
            fprintf(stderr,"Conv-host> node %d: %s\n",rsh_node[i],line);
          if (strcmp(line,"rsh phase successful.")==0) { done=1; break; }
      }
      if (!done) continue;
      rsh_nfinished++;

      prog_close(rsh_prog[i]);
      rsh_prog[i] = 0;
      if (rsh_nstarted==nodetab_rank0_size) break;
      pe = nodetab_rank0_table[rsh_nstarted];
      p = rsh_start(pe);
      if (p==0) { perror("ERROR> starting rsh"); exit(1); }
      rsh_pump(p, pe, rsh_nstarted, arg_argv);
      rsh_prog[i] = p;
      rsh_node[i] = pe;
      rsh_nstarted++;
    }
  }
}

#endif /*CMK_USE_RSH*/
