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
#include <assert.h>
#include <math.h>
#include <limits.h>
#if CMK_BPROC
#include <sys/bproc.h>
#endif
#if CMK_USE_POLL
#include <poll.h>
#endif
#include <sys/stat.h>

#include <map>
#include <string>
#include <vector>
#include <utility>

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
#define SIGBUS -1 /*These signals don't exist in Win32*/
#define SIGKILL -1
#define SIGQUIT -1

#else            /*UNIX*/
#include <pwd.h> /*getcwd*/
#include <unistd.h>
#define DIRSEP "/"
#endif

#define PRINT(a) (arg_quiet ? 1 : printf a)

#if CMK_SSH_NOT_NEEDED /*No SSH-- use daemon to start node-programs*/
#define CMK_USE_SSH 0

#else /*Use SSH to start node-programs*/
#define CMK_USE_SSH 1
#ifdef __MINGW_H
#include <rpc.h>
#elif !defined(__CYGWIN__)
#include <rpc/rpc.h>
#else
#include <w32api/rpc.h>
#endif
#if CMK_SSH_IS_A_COMMAND
#define SSH_CMD "ssh"
#endif

#endif

#include "daemon.h"

/*#define DEBUGF(x) printf x*/
#define DEBUGF(x)

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

const int MAX_NUM_RETRIES = 3;
#include <map>

std::map<SOCKET, int> skt_client_table;
std::map<std::string, int> host_sizes;

const char *nodetab_name(int i);
const char *skt_to_name(SOCKET skt)
{
  if (skt_client_table.find(skt) != skt_client_table.end()) {
    return nodetab_name(skt_client_table[skt]);
  } else {
    return "UNKNOWN";
  }
}
int skt_to_node(SOCKET skt)
{
  if (skt_client_table.find(skt) != skt_client_table.end()) {
    return skt_client_table[skt];
  } else {
    return -1;
  }
}

//#define HSTART
#ifdef HSTART
/*Hierarchical-start routines*/
int mynodes_start; /* To keep a global node numbering */

#endif

static double ftTimer;

double start_timer;

int *ssh_pids = NULL;

double GetClock(void)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
  struct _timeb tv;
  _ftime(&tv);
  return (tv.time * 1.0 + tv.millitm * 1.0E-3);
#else
  struct timeval tv;
  int ok;
  ok = gettimeofday(&tv, NULL);
  if (ok < 0) {
    perror("gettimeofday");
    exit(1);
  }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
#endif
}

int probefile(const char *path)
{
  FILE *f = fopen(path, "r");
  if (f == NULL)
    return 0;
  fclose(f);
  return 1;
}

const char *mylogin(void)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
  static char name[100] = {'d', 'u', 'n', 'n', 'o', 0};
  unsigned int len = 100;
  GetUserName(name, (LPDWORD) &len);
  return name;
#else /*UNIX*/
  struct passwd *self;

  self = getpwuid(getuid());
  if (self == 0) {
#if CMK_HAS_POPEN
    char cmd[16];
    char uname[64];
    FILE *p;
    sprintf(cmd, "id -u -n");
    p = popen(cmd, "r");
    if (p) {
      fscanf(p, "%s", uname);
      pclose(p);
      return strdup(uname);
    } else
      return "unknown";
#else
    return "unknown";
#endif
  }
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
  char info[1000];
  /*This is the resolved IP address of elegance.cs.uiuc.edu */
  skt_ip_t destination_ip = skt_lookup_ip("128.174.241.211");
  unsigned int destination_port = 6571;
  struct sockaddr_in addr = skt_build_addr(destination_ip, destination_port);
  SOCKET skt;

  skt = socket(AF_INET, SOCK_DGRAM, 0);
  if (skt == INVALID_SOCKET)
    return;

  sprintf(info, "%s", mylogin());

  sendto(skt, info, strlen(info), 0, (struct sockaddr *) &addr, sizeof(addr));
  skt_close(skt);
#endif /* NOTIFY */
}

/**************************************************************************
 *
 * Pathfix : alters a path according to a set of rewrite rules
 *
 *************************************************************************/

typedef struct s_pathfixlist {
  char *s1;
  char *s2;
  struct s_pathfixlist *next;
} * pathfixlist;

pathfixlist pathfix_append(char *s1, char *s2, pathfixlist l)
{
  pathfixlist pf = (pathfixlist) malloc(sizeof(s_pathfixlist));
  pf->s1 = s1;
  pf->s2 = s2;
  pf->next = l;
  return pf;
}

char *pathfix(const char *path, pathfixlist fixes)
{
  char buffer[MAXPATHLEN];
  pathfixlist l;
  char buf2[MAXPATHLEN];
  char *offs;
  int mod, len;
  strcpy(buffer, path);
  mod = 1;
  while (mod) {
    mod = 0;
    for (l = fixes; l; l = l->next) {
      len = strlen(l->s1);
      offs = strstr(buffer, l->s1);
      if (offs) {
        offs[0] = 0;
        sprintf(buf2, "%s%s%s", buffer, l->s2, offs + len);
        strcpy(buffer, buf2);
        mod = 1;
      }
    }
  }
  return strdup(buffer);
}

char *pathextfix(const char *path, pathfixlist fixes, char *ext)
{
  char *newpath = pathfix(path, fixes);
  char *ret;
  if (ext == NULL)
    return newpath;
  ret = (char *) malloc(strlen(newpath) + strlen(ext) + 2);
  strcpy(ret, newpath);
  strcat(ret, ext);
  free(newpath);
  return ret;
}

/****************************************************************************
 *
 * Miscellaneous minor routines.
 *
 ****************************************************************************/

int is_quote(char c) { return (c == '\'' || c == '"'); }

void zap_newline(char *s)
{
  char *p;
  p = s + strlen(s) - 1;
  if (*p == '\n')
    *p = '\0';
  /* in case of DOS ^m */
  p--;
  if (*p == '\15')
    *p = '\0';
}

/* get substring from lo to hi, remove quote chars */
char *substr(const char *lo, const char *hi)
{
  int len;
  char *res;
  if (is_quote(*lo))
    lo++;
  if (is_quote(*(hi - 1)))
    hi--;
  len = hi - lo;
  res = (char *) malloc(1 + len);
  memcpy(res, lo, len);
  res[len] = 0;
  return res;
}

int subeqs(const char *lo, const char *hi, const char *str)
{
  int len = strlen(str);
  if (hi - lo != len)
    return 0;
  if (memcmp(lo, str, len))
    return 0;
  return 1;
}

/* advance pointer over blank characters */
const char *skipblanks(const char *p)
{
  while ((*p == ' ') || (*p == '\t'))
    p++;
  return p;
}

/* advance pointer over nonblank characters and a quoted string */
const char *skipstuff(const char *p)
{
  char quote = 0;
  if (*p && (*p == '\'' || *p == '"')) {
    quote = *p;
    p++;
  }
  if (quote != 0) {
    while (*p && *p != quote)
      p++;
    if (*p != quote) {
      fprintf(stderr, "ERROR> Unmatched quote in nodelist file.\n");
      exit(1);
    }
    p++;
  } else
    while ((*p) && (*p != ' ') && (*p != '\t'))
      p++;
  return p;
}

#if CMK_USE_SSH
const char *getenv_ssh()
{
  char *e;

  e = getenv("CONV_RSH");
  return e ? e : SSH_CMD;
}
#endif

#if !defined(_WIN32) || defined(__CYGWIN__)
char *getenv_display()
{
  static char result[100], ipBuf[200];
  char *e, *p;

  e = getenv("DISPLAY");
  if (e == 0)
    return NULL;
  p = strrchr(e, ':');
  if (p == 0)
    return NULL;
  if ((e[0] == ':') || (strncmp(e, "unix:", 5) == 0)) {
    sprintf(result, "%s:%s", skt_print_ip(ipBuf, skt_my_ip()), p + 1);
  } else
    strcpy(result, e);
  return result;
}
char *getenv_display_no_tamper()
{
  static char result[100], ipBuf[200];
  char *e, *p;

  e = getenv("DISPLAY");
  if (e == 0)
    return NULL;
  p = strrchr(e, ':');
  if (p == 0)
    return NULL;
  strcpy(result, e);
  return result;
}

#endif

static unsigned int server_port;
static char server_addr[1024]; /* IP address or hostname of charmrun*/
static SOCKET server_fd;
#if CMK_SHRINK_EXPAND
char *create_netstart(int node);
char *create_oldnodenames();
#endif
/*****************************************************************************
 *                                                                           *
 * PPARAM - obtaining "program parameters" from the user.                    *
 *                                                                           *
 *****************************************************************************/

typedef struct s_ppdef {
  union {
    int *i;
    double *r;
    const char **s;
    int *f;
  } where;           /*Where to store result*/
  const char *lname; /*Argument name on command line*/
  const char *doc;
  char type; /*One of i, r, s, f.*/
  bool initFlag; // if 0 means, user input paramater is inserted. 1 means, it holds a default value
  struct s_ppdef *next;
} * ppdef;

static ppdef ppdefs;

static int pparam_pos;
static const char **pparam_argv;
static char pparam_optc = '-';
char pparam_error[100];

static ppdef pparam_find(const char *lname)
{
  ppdef def;
  for (def = ppdefs; def; def = def->next)
    if (strcmp(def->lname, lname) == 0)
      return def;
  return 0;
}

static ppdef pparam_cell(const char *lname)
{
  ppdef def = pparam_find(lname);
  if (def)
    return def;
  def = (ppdef) malloc(sizeof(s_ppdef));
  def->lname = lname;
  def->type = 's';
  def->doc = "(undocumented)";
  def->next = ppdefs;
  def->initFlag = true;
  ppdefs = def;
  return def;
}

void pparam_int(int *where, int defValue, const char *arg, const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type = 'i';
  def->where.i = where;
  *where = defValue;
  def->lname = arg;
  def->doc = doc;
}

void pparam_flag(int *where, int defValue, const char *arg, const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type = 'f';
  def->where.f = where;
  *where = defValue;
  def->lname = arg;
  def->doc = doc;
}

void pparam_real(double *where, double defValue, const char *arg,
                 const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type = 'r';
  def->where.r = where;
  *where = defValue;
  def->lname = arg;
  def->doc = doc;
}

void pparam_str(const char **where, const char *defValue, const char *arg,
                const char *doc)
{
  ppdef def = pparam_cell(arg);
  def->type = 's';
  def->where.s = where;
  *where = defValue;
  def->lname = arg;
  def->doc = doc;
}

static int pparam_setdef(ppdef def, const char *value)
{
  char *p;
  if (def->initFlag)
    def->initFlag = false;
  else {
    fprintf(stderr, "Option \'%s\' is used more than once. Please remove duplicate arguments for this option\n", def->lname);
    exit(1);
  }

  switch (def->type) {
  case 'i':
    *def->where.i = strtol(value, &p, 10);
    if (*p)
      return -1;
    return 0;
  case 'r':
    *def->where.r = strtod(value, &p);
    if (*p)
      return -1;
    return 0;
  case 's': {
    /* Parse input string and convert a literal "\n" into '\n'. */
    *def->where.s = (char *) calloc(strlen(value) + 1, sizeof(char));
    char *parsed_value = (char *) *def->where.s;
    for (int i = 0, j = 0; i < strlen(value); i++) {
      if (i + 1 < strlen(value)) {
        if (value[i] == '\\' && value[i + 1] == 'n') {
          parsed_value[j++] = '\n';
          i++;
          continue;
        }
      }
      parsed_value[j++] = value[i];
    }
    return 0;
  }
  case 'f':
    *def->where.f = strtol(value, &p, 10);
    if (*p)
      return -1;
    return 0;
  }
  return -1;
}

int pparam_set(char *lname, char *value)
{
  ppdef def = pparam_cell(lname);
  return pparam_setdef(def, value);
}

const char *pparam_getdef(ppdef def)
{
  static char result[100];
  switch (def->type) {
  case 'i':
    sprintf(result, "%d", *def->where.i);
    return result;
  case 'r':
    sprintf(result, "%f", *def->where.r);
    return result;
  case 's':
    return *def->where.s ? *def->where.s : "";
  case 'f':
    sprintf(result, "%d", *def->where.f);
    return result;
  }
  return NULL;
}

void pparam_printdocs()
{
  ppdef def;
  int len, maxname, maxdoc;
  maxname = 0;
  maxdoc = 0;
  for (def = ppdefs; def; def = def->next) {
    len = strlen(def->lname);
    if (len > maxname)
      maxname = len;
    len = strlen(def->doc);
    if (len > maxdoc)
      maxdoc = len;
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "Charmrun Command-line Parameters:\n");
  for (def = ppdefs; def; def = def->next) {
    fprintf(stderr, "  %c%c%-*s ", pparam_optc, pparam_optc, maxname,
            def->lname);
    fprintf(stderr, "  %-*s [%s]\n", maxdoc, def->doc, pparam_getdef(def));
  }
  fprintf(stderr, "\n");
}

void pparam_delarg(int i)
{
  int j;
  for (j = i; pparam_argv[j]; j++)
    pparam_argv[j] = pparam_argv[j + 1];
}

int pparam_countargs(const char **argv)
{
  int argc;
  for (argc = 0; argv[argc]; argc++)
    ;
  return argc;
}

int pparam_parseopt()
{
  int ok;
  ppdef def = NULL;
  const char *opt = pparam_argv[pparam_pos];
  /* handle ++ by skipping to end */
  if ((opt[1] == '+') && (opt[2] == 0)) {
    pparam_delarg(pparam_pos);
    while (pparam_argv[pparam_pos])
      pparam_pos++;
    return 0;
  }
  /* handle + by itself - an error */
  if (opt[1] == 0) {
    sprintf(pparam_error, "Illegal option +\n");
    return -1;
  }
  /* look up option definition */
  if (opt[1] == '+')
    def = pparam_find(opt + 2);
  else {
    char name[2];
    name[0] = opt[1];
    if (strlen(opt) <= 2 || !isalpha(opt[2])) {
      name[1] = 0;
      def = pparam_find(name);
    }
  }
  if (def == NULL) {
    if (opt[1] == '+') {
      sprintf(pparam_error, "Option %s not recognized.", opt);
      return -1;
    } else {
      /*Unrecognized + option-- skip it.*/
      pparam_pos++;
      return 0;
    }
  }
  /* handle flag-options */
  if ((def->type == 'f') && (opt[1] != '+') && (opt[2])) {
    sprintf(pparam_error, "Option %s should not include a value", opt);
    return -1;
  }
  if (def->type == 'f') {
    *def->where.f = 1;
    pparam_delarg(pparam_pos);
    return 0;
  }
  /* handle non-flag options */
  if ((opt[1] == '+') || (opt[2] == 0)) {
    pparam_delarg(pparam_pos);
    opt = pparam_argv[pparam_pos];
  } else
    opt += 2;
  if ((opt == 0) || (opt[0] == 0)) {
    sprintf(pparam_error, "%s must be followed by a value.", opt);
    return -1;
  }
  ok = pparam_setdef(def, opt);
  pparam_delarg(pparam_pos);
  if (ok < 0) {
    sprintf(pparam_error, "Illegal value for %s", opt);
    return -1;
  }
  return 0;
}

int pparam_parsecmd(char optchr, const char **argv)
{
  pparam_error[0] = 0;
  pparam_argv = argv;
  pparam_optc = optchr;
  pparam_pos = 0;
  while (1) {
    const char *opt = pparam_argv[pparam_pos];
    if (opt == 0)
      break;
    if (opt[0] != optchr)
      pparam_pos++;
    else if (pparam_parseopt() < 0)
      return -1;
  }
  return 0;
}

#ifdef HSTART
char **dupargv(char **argv)
{
  int argc;
  char **copy;

  if (argv == NULL)
    return NULL;

  /* the vector */
  for (argc = 0; argv[argc] != NULL; argc++)
    ;
  copy = (char **) malloc((argc + 2) * sizeof(char *));
  if (copy == NULL)
    return NULL;

  /* the strings */
  for (argc = 0; argv[argc] != NULL; argc++) {
    int len = strlen(argv[argc]);
    copy[argc] = malloc(sizeof(char) * (len + 1));
    strcpy(copy[argc], argv[argc]);
  }
  copy[argc] = NULL;
  return copy;
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

#define MAX_LINE_LENGTH 1000

const char **arg_argv;
int arg_argc;

int arg_requested_pes;
int arg_timeout;
int arg_verbose;
const char *arg_nodelist;
const char *arg_nodegroup;
const char *arg_runscript; /* script to run the node-program with */
const char *arg_charmrunip;

int arg_debug;
int arg_debug_no_pause;
int arg_debug_no_xrdb;
int arg_charmdebug;
const char *
    arg_debug_commands; /* commands that are provided by a ++debug-commands
                           flag. These are passed into gdb. */

int arg_quiet;       /* omit charmrun standard output */
int arg_local;       /* start node programs directly by exec on localhost */
int arg_batch_spawn; /* control starting node programs, several at a time */
int arg_scalable_start;

#ifdef HSTART
int arg_hierarchical_start;
int arg_child_charmrun;
#endif
int arg_help; /* print help message */
int arg_ppn;  /* pes per node */
int arg_usehostname;

#if CMK_SHRINK_EXPAND
char **saved_argv;
int saved_argc;
int arg_realloc_pes;
int arg_old_pes;
int arg_shrinkexpand;
int arg_charmrun_port;
const char *arg_shrinkexpand_basedir;
#endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
int arg_read_pes = 0;
#endif

#if CMK_USE_SSH
int arg_maxssh;
const char *arg_shell;
int arg_in_xterm;
const char *arg_debugger;
const char *arg_xterm;
const char *arg_display;
int arg_ssh_display;
const char *arg_mylogin;
#endif
int arg_mpiexec;
int arg_mpiexec_no_n;
int arg_no_va_rand;

const char *arg_nodeprog_a;
const char *arg_nodeprog_r;
char *arg_currdir_a;
char *arg_currdir_r;

int arg_server;
int arg_server_port = 0;
const char *arg_server_auth = NULL;
int replay_single = 0;

#if CMK_BPROC
int arg_startpe;
int arg_endpe;
int arg_singlemaster;
int arg_skipmaster;
#endif

void arg_init(int argc, const char **argv)
{
  static char buf[1024];

  int i, local_def = 0;
#if CMK_CHARMRUN_LOCAL
  local_def = 1; /*++local is the default*/
#endif

  pparam_int(&arg_requested_pes, 1, "p", "number of processes to create");
  pparam_int(&arg_timeout, 60, "timeout",
             "seconds to wait per host connection");
  pparam_flag(&arg_verbose, 0, "verbose", "Print diagnostic messages");
  pparam_flag(&arg_quiet, 0, "quiet", "Omit non-error runtime messages");
  pparam_str(&arg_nodelist, 0, "nodelist", "file containing list of nodes");
  pparam_str(&arg_nodegroup, "main", "nodegroup",
             "which group of nodes to use");
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  pparam_int(&arg_read_pes, 0, "readpe",
             "number of host names to read into the host table");
#endif

#if CMK_CCS_AVAILABLE
  pparam_flag(&arg_server, 0, "server", "Enable client-server (CCS) mode");
  pparam_int(&arg_server_port, 0, "server-port",
             "Port to listen for CCS requests");
  pparam_str(&arg_server_auth, 0, "server-auth", "CCS Authentication file");
#endif
  pparam_flag(&arg_local, local_def, "local",
              "Start node programs locally without daemon");
  pparam_int(&arg_batch_spawn, 0, "batch", "Launch connections to this many "
                                           "node programs at a time, avoiding "
                                           "overloading charmrun pe");
  pparam_flag(&arg_scalable_start, 1, "scalable-start", "scalable start");
#ifdef HSTART
  pparam_flag(&arg_hierarchical_start, 0, "hierarchical-start",
              "hierarchical start");
  pparam_flag(&arg_child_charmrun, 0, "child-charmrun", "child charmrun");
#endif
#if CMK_SHRINK_EXPAND
  pparam_int(&arg_realloc_pes, 1, "newp", "new number of processes to create");
  pparam_int(&arg_old_pes, 1, "oldp", "old number of processes to create");
  pparam_flag(&arg_shrinkexpand, 0, "shrinkexpand", "shrink expand");
  pparam_int(&arg_charmrun_port, 0, "charmrun_port", "make charmrun listen on this port");
#endif
  pparam_flag(&arg_usehostname, 0, "usehostname",
              "Send nodes our symbolic hostname instead of IP address");
  pparam_str(&arg_charmrunip, 0, "useip",
             "Use IP address provided for charmrun IP");
  pparam_flag(&arg_mpiexec, 0, "mpiexec", "use mpiexec to start jobs");
  pparam_flag(&arg_mpiexec_no_n, 0, "mpiexec-no-n", "use mpiexec to start jobs without -n procs");
#if CMK_USE_SSH
  pparam_flag(&arg_debug, 0, "debug",
              "Run each node under gdb in an xterm window");
  pparam_flag(&arg_debug_no_pause, 0, "debug-no-pause",
              "Like debug, except doesn't pause at beginning");
  pparam_str(&arg_debug_commands, 0, "debug-commands",
             "Commands to be run inside gdb at startup");
  pparam_flag(&arg_debug_no_xrdb, 0, "no-xrdb", "Don't check xrdb");

/* When the ++charmdebug flag is used, charmrun listens from its stdin for
   commands, and forwards them to the gdb info program (a child), or to the
   processor gdbs. The stderr is redirected to the stdout, so the two streams
   are mixed together. The channel for stderr is reused to forward the replies
   of gdb back to the java debugger. */
#if !defined(_WIN32)
  pparam_flag(&arg_charmdebug, 0, "charmdebug",
              "Used only when charmrun is started by charmdebug");
#endif

  pparam_int(&arg_maxssh, 16, "maxssh",
             "Maximum number of ssh's to run at a time");
  pparam_str(&arg_shell, 0, "remote-shell",
             "which remote shell to use (default $CONV_RSH or " SSH_CMD);
  pparam_str(&arg_debugger, 0, "debugger", "which debugger to use");
  pparam_str(&arg_display, 0, "display", "X Display for xterm");
  pparam_flag(&arg_ssh_display, 0, "ssh-display",
              "use own X Display for each ssh session");
  pparam_flag(&arg_in_xterm, 0, "in-xterm", "Run each node in an xterm window");
  pparam_str(&arg_xterm, 0, "xterm", "which xterm to use");
#endif
#ifdef CMK_BPROC
  /* options for Scyld */
  pparam_int(&arg_startpe, 0, "startpe", "first pe to start job(SCYLD)");
  pparam_int(&arg_endpe, 1000000, "endpe", "last pe to start job(SCYLD)");
  pparam_flag(&arg_singlemaster, 0, "singlemaster",
              "Only assign one process to master node(SCYLD)");
  pparam_flag(&arg_skipmaster, 0, "skipmaster",
              "Donot assign any process to master node(SCYLD)");
  if (arg_skipmaster && arg_singlemaster) {
    PRINT(("Charmrun> 'singlemaster' is ignored due to 'skipmaster'. \n"));
    arg_singlemaster = 0;
  }
  pparam_flag(&arg_debug, 0, "debug", "turn on more verbose debug print");
#endif
  pparam_str(&arg_runscript, 0, "runscript", "script to run node-program with");
  pparam_flag(&arg_help, 0, "help", "print help messages");
  pparam_int(&arg_ppn, 0, "ppn", "number of pes per node");
  pparam_flag(&arg_no_va_rand, 0, "no-va-randomization",
              "Disables randomization of the virtual address  space");
#ifdef HSTART
  arg_argv = dupargv(argv);
#endif

#if CMK_SHRINK_EXPAND
  /* move it to a function */
  saved_argc = argc;
  saved_argv = (char **) malloc(sizeof(char *) * (saved_argc));
  for (i = 0; i < saved_argc; i++) {
    //  MACHSTATE1(2,"Parameters %s",Cmi_argvcopy[i]);
    saved_argv[i] = (char *) argv[i];
  }
#endif

  if (pparam_parsecmd('+', argv) < 0) {
    fprintf(stderr, "ERROR> syntax: %s\n", pparam_error);
    pparam_printdocs();
    exit(1);
  }

  /* Check for (but do *not* remove) the "-?", "-h", or "--help" flags */
  for (i = 0; argv[i]; i++) {
    if (0 == strcmp(argv[i], "-?") || 0 == strcmp(argv[i], "-h") ||
        0 == strcmp(argv[i], "--help"))
      arg_help = 1;
  }
  if (arg_help) {
    pparam_printdocs();
    /*exit(0);*/
  }

  if ( arg_mpiexec_no_n ) arg_mpiexec = arg_mpiexec_no_n;

#if CMK_SHRINK_EXPAND
  if (arg_shrinkexpand) {
    arg_requested_pes = arg_realloc_pes;
    printf("\n \nCharmrun> %d Reallocated pes\n \n", arg_requested_pes);
  }
#endif

#ifdef HSTART
  if (!arg_hierarchical_start || arg_child_charmrun)
#endif
    arg_argv =
        (argv) + 1; /*Skip over charmrun (0) here and program name (1) later*/
  arg_argc = pparam_countargs(arg_argv);
  if (arg_argc < 1) {
    fprintf(stderr, "ERROR> You must specify a node-program.\n");
    pparam_printdocs();
    exit(1);
  }

#ifdef HSTART
  if (!arg_hierarchical_start || arg_child_charmrun) {
    // Removing nodeprogram from the list
    arg_argv++;
    arg_argc--;
  } else {
    // Removing charmrun from parameters
    arg_argv++;
    arg_argc--;

    arg_argv[arg_argc] = malloc(sizeof(char) * strlen("++child-charmrun"));
    strcpy(arg_argv[arg_argc++], "++child-charmrun");
    arg_argv[arg_argc] = NULL;
  }
#else
  arg_argv++;
  arg_argc--;
#endif

  if (arg_server_port || arg_server_auth)
    arg_server = 1;

  if (arg_verbose) arg_quiet = 0;

  if (arg_debug || arg_debug_no_pause) {
    fprintf(stderr, "Charmrun> scalable start disabled under ++debug:\n"
                    "NOTE: will make an SSH connection per process launched,"
                    " instead of per physical node.\n");
    arg_scalable_start = 0;
    arg_quiet = 0;
    arg_verbose = 1;
    /*Pass ++debug along to program (used by machine.c)*/
    arg_argv[arg_argc++] = "++debug";
  }
  /* pass ++quiet to program */
  if (arg_quiet) arg_argv[arg_argc++] = "++quiet";

  /* Check for +replay-detail to know we have to load only one single processor
   */
  for (i = 0; argv[i]; i++) {
    if (0 == strcmp(argv[i], "+replay-detail")) {
      replay_single = 1;
      arg_requested_pes = 1;
    }
  }

#ifdef CMK_BPROC
  if (arg_local) {
    fprintf(stderr,
            "Warning> ++local cannot be used in bproc version, ignored!\n");
    arg_local = 0;
  }
#endif

#if CMK_USE_SSH
  /* Find the current value of the CONV_RSH variable */
  if (!arg_shell) {
    if (arg_mpiexec)
      arg_shell = "mpiexec";
    else
      arg_shell = getenv_ssh();
  }

  /* Find the current value of the DISPLAY variable */
  if (!arg_display)
    arg_display = getenv_display_no_tamper();
  if ((arg_debug || arg_debug_no_pause || arg_in_xterm) && (arg_display == 0)) {
    fprintf(stderr, "ERROR> DISPLAY must be set to use debugging mode\n");
    exit(1);
  }
  if (arg_debug || arg_debug_no_pause)
    arg_timeout = 8 * 60 * 60; /* Wait 8 hours for ++debug */

  /* default debugger is gdb */
  if (!arg_debugger)
    arg_debugger = "gdb";
  /* default xterm is xterm */
  if (!arg_xterm)
    arg_xterm = "xterm";

  arg_mylogin = mylogin();
#endif

  /* find the current directory, absolute version */
  getcwd(buf, 1023);
  arg_currdir_a = strdup(buf);

  /* find the node-program, absolute version */
  arg_nodeprog_r = argv[1];

  if (arg_nodeprog_r[0] == '-' || arg_nodeprog_r[0] == '+') {
    /*If it starts with - or +, it ain't a node program.
      Chances are, the user screwed up and passed some
      unknown flag to charmrun*/
    fprintf(stderr, "Charmrun does not recognize the flag '%s'.\n", arg_nodeprog_r);
    if (arg_nodeprog_r[0] == '+')
      fprintf(stderr, "Charm++'s flags need to be placed *after* the program name.\n");
    pparam_printdocs();
    exit(1);
  }

#if defined(_WIN32) && !defined(__CYGWIN__)
  if (argv[1][1] == ':' ||
      argv[1][0] == '\\' && argv[1][1] == '\\') { /*E.g.: "C:\foo\bar.exe*/
#else
  if (argv[1][0] == '/') { /*E.g.: "\foo\bar"*/
#endif
    /*Absolute path to node-program*/
    arg_nodeprog_a = argv[1];
  } else {
    sprintf(buf, "%s%s%s", arg_currdir_a, DIRSEP, arg_nodeprog_r);
    arg_nodeprog_a = strdup(buf);
  }
  if (arg_scalable_start) {
    PRINT(("Charmrun> scalable start enabled. \n"));
  }

#ifdef HSTART
  if (arg_hierarchical_start) {
    PRINT(("Charmrun> Hierarchical scalable start enabled. \n"));
    if (arg_debug || arg_debug_no_pause) {
      fprintf(stderr, "Charmrun> Error: ++hierarchical-start does not support "
                      "debugging mode. \n");
      exit(1);
    }
    if (arg_verbose) {
      fprintf(stderr, "Charmrun> Warning: you have enabled verbose output with "
                      "Hierarchical startup, you may get inconsistent verbose "
                      "outputs. \n++hierarchial-start does not support verbose "
                      "mode. \n");
    }

  } else if (arg_child_charmrun) {
    fprintf(
        stderr,
        "Charmrun> Error: ++child-charmrun is not a user-specified flag. \n");
    exit(1);
  }
#endif

  /*If number of pes per node does not divide number of pes*/
  if(arg_requested_pes && arg_ppn){
    if(arg_requested_pes % arg_ppn != 0){
      if(arg_ppn > arg_requested_pes){
	arg_ppn=arg_requested_pes;
	fprintf(stderr, "Charmrun> warning: forced ++ppn = +p = %d\n",arg_ppn);
      }
      else
	{
	  fprintf(
		  stderr,
		  "Charmrun> Error: ++ppn (number of pes per node) does not divide +p (number of pes) \n");
	  exit(1);
	}
    }
  }
}

/****************************************************************************
 *
 * NODETAB:  The nodes file and nodes table.
 *
 ****************************************************************************/

static int portOk = 1;
static const char *nodetab_tempName = NULL;
char *nodetab_file_find()
{
  char buffer[MAXPATHLEN];

  /* Find a nodes-file as specified by ++nodelist */
  if (arg_nodelist) {
    const char *path = arg_nodelist;
    if (probefile(path))
      return strdup(path);
    fprintf(stderr, "ERROR> No such nodelist file %s\n", path);
    exit(1);
  }
  /* Find a nodes-file as specified by getenv("NODELIST") */
  if (getenv("NODELIST")) {
    char *path = getenv("NODELIST");
    if (path && probefile(path))
      return strdup(path);
    // cppcheck-suppress nullPointer
    fprintf(stderr, "ERROR> Cannot find nodelist file %s\n", path);
    exit(1);
  }
  /* Find a nodes-file by looking under 'nodelist' in the current directory */
  if (probefile("./nodelist"))
    return strdup("./nodelist");
#if defined(_WIN32) && !defined(__CYGWIN__)
  tmpnam(buffer);
  nodetab_tempName = strdup(buffer);
#else /*UNIX*/
  if (getenv("HOME")) {
    sprintf(buffer, "%s/.nodelist", getenv("HOME"));
  }
#endif
  if (!probefile(buffer)) {
    /*Create a simple nodelist in the user's home*/
    FILE *f = fopen(buffer, "w");
    if (f == NULL) {
      fprintf(stderr, "ERROR> Cannot create a 'nodelist' file.\n");
      exit(1);
    }
    fprintf(f, "group main\nhost localhost\n");
    fclose(f);
  }
  return strdup(buffer);
}

typedef struct nodetab_host {
  const char *name; /*Host DNS name*/
  skt_ip_t ip;      /*IP address of host*/
  pathfixlist pathfixes;
  char *ext;    /*FIXME: What the heck is this?  OSL 9/8/00*/
  int cpus;     /* # of physical CPUs*/
  int rank;     /*Rank of this CPU*/
  double speed; /*Relative speed of each CPU*/
  int nice;     /* process priority */
  int forks;    /* number of processes to fork on remote node */
  /*These fields are set during node-startup*/
  int dataport;  /*UDP port number*/
  SOCKET ctrlfd; /*Connection to control port*/
#if CMK_USE_SSH
  const char *shell;    /*Ssh to use*/
  const char *debugger; /*Debugger to use*/
  const char *xterm;    /*Xterm to use*/
  const char *login;    /*User login name to use*/
  const char *passwd;   /*User login password*/
  const char *setup;    /*Commands to execute on login*/
#endif

#if CMK_USE_IBVERBS
  ChInfiAddr *qpData;
#endif
#if CMK_USE_IBUD
  ChInfiAddr qp;
#endif

} nodetab_host;

nodetab_host **nodetab_table;
int nodetab_max;
int nodetab_size;
int *nodetab_rank0_table;
int nodetab_rank0_size;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
int loaded_max_pe;
#endif

void nodetab_reset(nodetab_host *h)
{
  h->name = "SET_H->NAME";
  h->ip = _skt_invalid_ip;
  h->pathfixes = 0;
  h->ext = NULL;
  h->speed = 1.0;
  h->cpus = 1;
  h->rank = 0;
  h->nice = -100;
  h->forks = 0;
  h->dataport = -1;
  h->ctrlfd = -1;
#if CMK_USE_SSH
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
    skt_print_ip(ips, h->ip);
    printf("Charmrun> adding client %d: \"%s\", IP:%s\n", nodetab_size, h->name,
           ips);
  }

  *nodetab_table[nodetab_size++] = *h;
}

void nodetab_makehost(const char *name, nodetab_host *h)
{
  h->name = strdup(name);
  h->ip = skt_innode_lookup_ip(name);
  if (skt_ip_match(h->ip, _skt_invalid_ip)) {
#ifdef CMK_BPROC
    /* only the master node is used */
    if (!(1 == arg_requested_pes && atoi(name) == -1))
#endif
    {
      fprintf(stderr, "ERROR> Cannot obtain IP address of %s\n", name);
      exit(1);
    }
  }
}

const char *nodetab_args(const char *args, nodetab_host *h)
{
  if (arg_ppn > 0)
    h->cpus = arg_ppn;
  while (*args != 0) {
    const char *b1 = skipblanks(args), *e1 = skipstuff(b1);
    const char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
    while (*b1 == '+')
      b1++; /*Skip over "++" on parameters*/
#if CMK_USE_SSH
    if (subeqs(b1, e1, "login"))
      h->login = substr(b2, e2);
    else if (subeqs(b1, e1, "passwd"))
      h->passwd = substr(b2, e2);
    else if (subeqs(b1, e1, "setup"))
      h->setup = strdup(b2);
    else if (subeqs(b1, e1, "shell"))
      h->shell = substr(b2, e2);
    else if (subeqs(b1, e1, "debugger"))
      h->debugger = substr(b2, e2);
    else if (subeqs(b1, e1, "xterm"))
      h->xterm = substr(b2, e2);
    else
#endif
        if (subeqs(b1, e1, "speed"))
      h->speed = atof(b2);
    else if (subeqs(b1, e1, "cpus")) {
      if (arg_ppn == 0)
        h->cpus = atol(b2); /* ignore if there is ++ppn */
    } else if (subeqs(b1, e1, "pathfix")) {
      const char *b3 = skipblanks(e2), *e3 = skipstuff(b3);
      args = skipblanks(e3);
      h->pathfixes =
          pathfix_append(substr(b2, e2), substr(b3, e3), h->pathfixes);
      e2 = e3; /* for the skipblanks at the end */
    } else if (subeqs(b1, e1, "ext"))
      h->ext = substr(b2, e2);
    else if (subeqs(b1, e1, "nice"))
      h->nice = atoi(b2);
    else
      return args;
    args = skipblanks(e2);
  }
  return args;
}

/* setup nodetab as localhost only */
void nodetab_init_for_local()
{
  int tablesize, i, done = 0;
  nodetab_host group;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  if (arg_read_pes == 0) {
    arg_read_pes = arg_requested_pes;
  }
#endif

  tablesize = arg_requested_pes;
  nodetab_table = (nodetab_host **) malloc(tablesize * sizeof(nodetab_host *));
  nodetab_rank0_table = (int *) malloc(tablesize * sizeof(int));
  nodetab_max = tablesize;

  nodetab_reset(&group);
  if (arg_ppn == 0)
    arg_ppn = 1;
#if CMK_SHARED_VARS_UNAVAILABLE
  if (arg_ppn > 1) {
    fprintf(stderr, "Warning> Invalid ppn %d in nodelist ignored.\n", arg_ppn);
    arg_ppn = 1;
  }
#endif
  group.cpus = arg_ppn;
  i = 0;
  while (!done) {
    const char *hostname = "127.0.0.1";
    for (group.rank = 0; group.rank < arg_ppn; group.rank++) {
      nodetab_makehost(hostname, &group);
      nodetab_add(&group);
      if (++i == arg_requested_pes) {
        done = 1;
        break;
      }
    }
  }
  host_sizes["127.0.0.1"] = (arg_requested_pes + arg_ppn - 1) / arg_ppn;
}

#ifdef HSTART
/* Sets the parent field of hosts to point to their parent charmrun. The root
 * charmrun will create children for all hosts which are parent of at least one
 * other host*/
int branchfactor;
int nodes_per_child;
int *nodetab_unique_table;
int nodetab_unique_size;
char *nodetab_name(int i);
void nodetab_init_hierarchical_start(void)
{
  int node_start = 0;
  char *node_name;
  nodetab_unique_size = 0;
  nodetab_unique_table = (int *) malloc(nodetab_rank0_size * sizeof(int));
  while (node_start < nodetab_rank0_size) {
    nodetab_unique_table[nodetab_unique_size++] = node_start;
    node_name = nodetab_name(node_start);
    do {
      node_start++;
    } while (node_start < nodetab_rank0_size &&
             (!strcmp(nodetab_name(node_start), node_name)));
  }
  branchfactor = ceil(sqrt(nodetab_unique_size));
  nodes_per_child = round(nodetab_unique_size * 1.0 / branchfactor);
}
#endif

#if CMK_SHRINK_EXPAND
int isPresent(const char *names, char **listofnames)
{
  int k;
  for (k = 0; k < arg_old_pes; k++) {
    if (strcmp(names, listofnames[k]) == 0)
      return 1;
  }
  return 0;
}
void parse_oldnodenames(char **oldnodelist)
{
  char *ns;
  ns = getenv("OLDNODENAMES");
  int i;
  char buffer[1024 * 1000];
  for (i = 0; i < arg_old_pes; i++) {
    oldnodelist[i] = (char *) malloc(100 * sizeof(char));
    int nread = sscanf(ns, "%s %[^\n]", oldnodelist[i], buffer);
    ns = buffer;
  }
}
#endif

void nodetab_init()
{
  FILE *f;
  char *nodesfile;
  nodetab_host global, group, host;
  char input_line[MAX_LINE_LENGTH];
  int rightgroup, i, remain, lineNo;
  /* Store the previous host so we can make sure we aren't mixing localhost and
   * non-localhost */
  char *prevHostName = NULL;
  std::vector< std::pair<int, nodetab_host> > hosts;
  std::multimap<int, nodetab_host> binned_hosts;

  /* if arg_local is set, ignore the nodelist file */
  if (arg_local || arg_mpiexec) {
    nodetab_init_for_local();
    goto fin;
  }

  /* Open the NODES_FILE. */
  nodesfile = nodetab_file_find();
  if (arg_verbose)
    fprintf(stderr, "Charmrun> using %s as nodesfile\n", nodesfile);
  if (!(f = fopen(nodesfile, "r"))) {
    fprintf(stderr, "ERROR> Cannot read %s: %s\n", nodesfile, strerror(errno));
    exit(1);
  }
  free(nodesfile);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  if (arg_read_pes == 0) {
    arg_read_pes = arg_requested_pes;
  }
  nodetab_table =
      (nodetab_host **) malloc(arg_read_pes * sizeof(nodetab_host *));
  nodetab_rank0_table = (int *) malloc(arg_read_pes * sizeof(int));
  nodetab_max = arg_read_pes;
  PRINT(("arg_read_pes %d arg_requested_pes %d\n", arg_read_pes,
          arg_requested_pes));
#else
  nodetab_table =
      (nodetab_host **) malloc(arg_requested_pes * sizeof(nodetab_host *));
  nodetab_rank0_table = (int *) malloc(arg_requested_pes * sizeof(int));
  nodetab_max = arg_requested_pes;
#endif

  nodetab_reset(&global);
  group = global;
  rightgroup = (strcmp(arg_nodegroup, "main") == 0);

  if (arg_ppn == 0)
    arg_ppn = 1;

  lineNo = 1;
  while (fgets(input_line, sizeof(input_line) - 1, f) != 0) {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    if (nodetab_size == arg_read_pes)
      break;
#else
    if (nodetab_size == arg_requested_pes)
      break;
#endif
    if (input_line[0] == '#')
      continue;
    zap_newline(input_line);
    if (!nodetab_args(input_line, &global)) {
      /*An option line-- also add options to current group*/
      nodetab_args(input_line, &group);
    } else { /*Not an option line*/
      const char *b1 = skipblanks(input_line), *e1 = skipstuff(b1);
      const char *b2 = skipblanks(e1), *e2 = skipstuff(b2);
      const char *b3 = skipblanks(e2);
      if (subeqs(b1, e1, "host")) {
        if (rightgroup) {
          /* check if we have a previous host, if it's different than our
           * current host, and if one of them is localhost */
          if (prevHostName && strcmp(b2, prevHostName) &&
              (!strcmp(b2, "localhost") ||
               !strcmp(prevHostName, "localhost"))) {
            fprintf(stderr, "ERROR> Mixing localhost with other hostnames will "
                            "lead to connection failures.\n");
            fprintf(stderr, "ERROR> The problematic line in group %s is: %s\n",
                    arg_nodegroup, input_line);
            exit(1);
          }
          host = group;
          nodetab_args(b3, &host);
#if !CMK_SMP
          /* Non-SMP workaround */
          int cpus = host.cpus;
          host.cpus = 1;
          for (int rank = 0; rank < cpus; rank++)
#else
          for (host.rank = 0; host.rank < host.cpus; host.rank++)
#endif
          {
            nodetab_makehost(substr(b2, e2), &host);
            hosts.push_back(std::make_pair(lineNo, host));
          }
          free(prevHostName);
          prevHostName = strdup(b2);
        }
      } else if (subeqs(b1, e1, "group")) {
        group = global;
        nodetab_args(b3, &group);
        rightgroup = subeqs(b2, e2, arg_nodegroup);
      } else if (b1 != b3) {
        fprintf(stderr, "ERROR> unrecognized command in nodesfile:\n");
        fprintf(stderr, "ERROR> %s\n", input_line);
        exit(1);
      }
    }
    lineNo++;
  }
  fclose(f);
  if (nodetab_tempName != NULL)
    unlink(nodetab_tempName);

  if (hosts.size() == 0) {
    fprintf(stderr, "ERROR> No hosts in group %s\n", arg_nodegroup);
    exit(1);
  }

  /*Wrap nodes in table around if there aren't enough yet*/
  for (int i = 0; binned_hosts.size() < arg_requested_pes; ++i) {
    binned_hosts.insert(hosts[i % hosts.size()]);
    host_sizes[hosts[i % hosts.size()].second.name]++;
  }

  /* Only increase counter for each new process */
  for (std::map<std::string, int>::iterator it = host_sizes.begin();
       it != host_sizes.end(); ++it) {
    it->second = (it->second + arg_ppn - 1) / arg_ppn;
  }

  for (std::multimap<int, nodetab_host>::iterator it = binned_hosts.begin();
       it != binned_hosts.end(); ++it) {
    nodetab_add(&(it->second));
  }

fin:
  /*Clip off excess CPUs at end*/
  for (i = 0; i < nodetab_size; i++) {
    if (nodetab_table[i]->rank == 0)
      remain = nodetab_size - i;
    if (nodetab_table[i]->cpus > remain)
      nodetab_table[i]->cpus = remain;
  }

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  loaded_max_pe = arg_requested_pes - 1;
#endif
#ifdef HSTART
  if (arg_hierarchical_start)
    nodetab_init_hierarchical_start();
#endif

  free(prevHostName);

#if CMK_SHRINK_EXPAND
  if (arg_shrinkexpand &&
      (arg_requested_pes > arg_old_pes)) // modify nodetable ordering
  {
    nodetab_host **reordered_nodetab_table =
        (nodetab_host **) malloc(arg_requested_pes * sizeof(nodetab_host *));
    char **oldnodenames = (char **) malloc(arg_old_pes * sizeof(char *));

    parse_oldnodenames(oldnodenames);
    int newpes = arg_old_pes;
    int oldpes = 0;
    int k;
    for (k = 0; k < nodetab_size; k++) {
      if (isPresent(nodetab_table[k]->name, oldnodenames))
        reordered_nodetab_table[oldpes++] = nodetab_table[k];
      else
        reordered_nodetab_table[newpes++] = nodetab_table[k];
    }
    free(nodetab_table);
    nodetab_table = reordered_nodetab_table;
  }
#endif
}

/* Given a processor number, look up the nodetab info: */
nodetab_host *nodetab_getinfo(int i)
{
  if (nodetab_table == 0) {
    fprintf(stderr, "ERROR> Node table not initialized.\n");
    exit(1);
  }
  return nodetab_table[i];
}

/* Given a node number, look up the nodetab info: */
nodetab_host *nodetab_getnodeinfo(int i)
{
  return nodetab_getinfo(nodetab_rank0_table[i]);
}

/*These routines all take *PE* numbers (NOT node numbers!)*/
const char *nodetab_name(int i) { return nodetab_getinfo(i)->name; }
pathfixlist nodetab_pathfixes(int i) { return nodetab_getinfo(i)->pathfixes; }
char *nodetab_ext(int i) { return nodetab_getinfo(i)->ext; }
skt_ip_t nodetab_ip(int i) { return nodetab_getinfo(i)->ip; }
unsigned int nodetab_cpus(int i) { return nodetab_getinfo(i)->cpus; }
unsigned int nodetab_rank(int i) { return nodetab_getinfo(i)->rank; }
int nodetab_dataport(int i) { return nodetab_getinfo(i)->dataport; }
int nodetab_nice(int i) { return nodetab_getinfo(i)->nice; }
SOCKET nodetab_ctrlfd(int i) { return nodetab_getinfo(i)->ctrlfd; }
#if CMK_USE_SSH
const char *nodetab_setup(int i) { return nodetab_getinfo(i)->setup; }
const char *nodetab_shell(int i) { return nodetab_getinfo(i)->shell; }
const char *nodetab_debugger(int i) { return nodetab_getinfo(i)->debugger; }
const char *nodetab_xterm(int i) { return nodetab_getinfo(i)->xterm; }
const char *nodetab_login(int i) { return nodetab_getinfo(i)->login; }
const char *nodetab_passwd(int i) { return nodetab_getinfo(i)->passwd; }
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

static ChNodeinfo *nodeinfo_arr; /*Indexed by node number.*/

void nodeinfo_allocate(void)
{
  nodeinfo_arr = (ChNodeinfo *) malloc(nodetab_rank0_size * sizeof(ChNodeinfo));
}
void nodeinfo_add(const ChSingleNodeinfo *in, SOCKET ctrlfd)
{
  int node = ChMessageInt(in->nodeNo);
  ChNodeinfo i = in->info;
  unsigned int nt;
  unsigned int pe;
  unsigned int dataport;
  int lid, qpn, psn;
  if (node < 0 || node >= nodetab_rank0_size) {
    fprintf(stderr, "Unexpected node %d registered!\n", node);
    exit(1);
  }
  nt = nodetab_rank0_table[node]; /*Nodetable index for this node*/
  i.nPE = ChMessageInt_new(nodetab_cpus(nt));
  i.nProcessesInPhysNode = ChMessageInt_new(host_sizes[nodetab_name(nt)]);

  if (arg_mpiexec)
    nodetab_getinfo(nt)->ip = i.IP; /* get IP */
  i.IP = nodetab_ip(nt);
#if CMK_USE_IBVERBS
  nodeinfo_arr[node] = i;
  for (pe = 0; pe < nodetab_cpus(nt); pe++) {
    nodetab_table[nt + pe]->ctrlfd = ctrlfd;
  }
/* PRINT(("Charmrun> client %d connected\n", nt)); */
#else
  dataport = ChMessageInt(i.dataport);
  if (0 == dataport) {
    fprintf(stderr, "Node %d could not initialize network!\n", node);
    exit(1);
  }
  nodeinfo_arr[node] = i;
  for (pe = 0; pe < nodetab_cpus(nt); pe++) {
    nodetab_table[nt + pe]->dataport = dataport;
    nodetab_table[nt + pe]->ctrlfd = ctrlfd;
#if CMK_USE_IBUD
    nodetab_table[nt + pe]->qp = i.qp;
#endif
  }
  if (arg_verbose) {
    char ips[200];
    skt_print_ip(ips, nodetab_ip(nt));
    printf("Charmrun> client %d connected (IP=%s data_port=%d)\n", nt, ips,
           dataport);
#if CMK_USE_IBUD
    printf("Charmrun> client %d lid=%d qpn=%i psn=%i\n", nt,
           ChMessageInt(i.qp.lid), ChMessageInt(i.qp.qpn),
           ChMessageInt(i.qp.psn));
#endif
  }
#endif
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
  char *new_input_buffer;
  int len = input_buffer ? strlen(input_buffer) : 0;
  fflush(stdout);
  if (fgets(line, 1023, stdin) == 0) {
    fprintf(stderr, "end-of-file on stdin");
    exit(1);
  }
  new_input_buffer = (char *) realloc(input_buffer, len + strlen(line) + 1);
  if (new_input_buffer == NULL) {
    // could not realloc
    free(input_buffer);
    fprintf(stderr, "Charmrun: Realloc failed");
    exit(1);
  } else {
    input_buffer = new_input_buffer;
  }

  strcpy(input_buffer + len, line);
}

void input_init() { input_buffer = strdup(""); }

char *input_extract(int nchars)
{
  char *res = substr(input_buffer, input_buffer + nchars);
  char *tmp =
      substr(input_buffer + nchars, input_buffer + strlen(input_buffer));
  free(input_buffer);
  input_buffer = tmp;
  return res;
}

char *input_gets()
{
  char *p, *res;
  int len;
  while (1) {
    p = strchr(input_buffer, '\n');
    if (p)
      break;
    input_extend();
  }
  len = p - input_buffer;
  res = input_extract(len + 1);
  res[len] = 0;
  return res;
}

/*FIXME: I am terrified by this routine. OSL 9/8/00*/
char *input_scanf_chars(char *fmt)
{
  char buf[8192];
  int len, pos;
  static int fd;
  static FILE *file;
  fflush(stdout);
  if (file == 0) {
#if CMK_USE_MKSTEMP
    char tmp[128];
    strcpy(tmp, "/tmp/fnordXXXXXX");
    mkstemp(tmp);
#else
    char *tmp = tmpnam(NULL); /*This was once /tmp/fnord*/
#endif
    unlink(tmp);
    fd = open(tmp, O_RDWR | O_CREAT | O_TRUNC, 0664);
    if (fd < 0) {
      fprintf(stderr, "cannot open temp file /tmp/fnord");
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
    fscanf(file, fmt, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf, buf,
           buf, buf, buf, buf, buf, buf, buf);
    pos = ftell(file);
    if (pos < len)
      break;
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
  const void *bufs[3];
  int lens[3];
  struct {
    ChMessageHeader ch; /*Make a charmrun header*/
    CcsImplHeader hdr;  /*Ccs internal header*/
  } h;
  void *reqData; /*CCS request data*/
  int pe, reqBytes;
  if (0 == CcsServer_recvRequest(&h.hdr, &reqData))
    return; /*Malformed request*/
  pe = ChMessageInt(h.hdr.pe);
  reqBytes = ChMessageInt(h.hdr.len);

  if (pe == -1) {
    /*Treat -1 as broadcast and sent to 0 as root of the spanning tree*/
    pe = 0;
  }
  if ((pe <= -nodetab_size || pe >= nodetab_size) && 0 == replay_single) {
/*Treat out of bound values as errors. Helps detecting bugs*/
/* But when virtualized with Bigemulator, we can have more pes than nodetabs */
/* TODO: We should somehow check boundaries also for bigemulator... */
#if !CMK_BIGSIM_CHARM
    if (pe == -nodetab_size)
      fprintf(stderr, "Invalid processor index in CCS request: are you trying "
                      "to do a broadcast instead?");
    else
      fprintf(stderr, "Invalid processor index in CCS request.");
    CcsServer_sendReply(&h.hdr, 0, 0);
    free(reqData);
    return;
#endif
  } else if (pe < -1) {
    /*Treat negative values as multicast to a number of processors specified by
      -pe.
      The pes to multicast to follows sits at the beginning of reqData*/
    reqBytes -= pe * sizeof(ChMessageInt_t);
    pe = ChMessageInt(*(ChMessageInt_t *) reqData);
  }

  if (!check_stdio_header(&h.hdr)) {

#define LOOPBACK 0
#if LOOPBACK /*Immediately reply "there's nothing!" (for performance           \
                testing)*/
    CcsServer_sendReply(&h.hdr, 0, 0);
#else
    int destpe = pe;
#if CMK_BIGSIM_CHARM
    destpe = destpe % nodetab_size;
#endif
    if (replay_single)
      destpe = 0;
    /*Fill out the charmrun header & forward the CCS request*/
    ChMessageHeader_new("req_fw", sizeof(h.hdr) + reqBytes, &h.ch);

    bufs[0] = &h;
    lens[0] = sizeof(h);
    bufs[1] = reqData;
    lens[1] = reqBytes;
    skt_sendV(nodetab_ctrlfd(destpe), 2, bufs, lens);

#endif
  }
  free(reqData);
}

/*
Forward the CCS reply (if any) from this client back to the
original network requestor, on the original request socket.
 */
int req_ccs_reply_fw(ChMessage *msg, SOCKET srcFd)
{
  int len = msg->len; /* bytes of data remaining to receive */

  /* First pull down the CCS header sent by the client. */
  CcsImplHeader hdr;
  skt_recvN(srcFd, &hdr, sizeof(hdr));
  len -= sizeof(hdr);

#define m (4 * 1024)              /* packets of message to recv/send at once */
  if (len < m || hdr.attr.auth) { /* short or authenticated message: grab the
                                     whole thing first */
    void *data = malloc(len);
    skt_recvN(srcFd, data, len);
    CcsServer_sendReply(&hdr, len, data);
    free(data);
  } else { /* long messages: packetize (for pipelined sending; a 2x bandwidth
              improvement!) */
    ChMessageInt_t outLen;
    int destFd; /* destination for data */
    skt_abortFn old = skt_set_abort(reply_abortFn);
    int destErrs = 0;

    destFd = ChMessageInt(hdr.replyFd);
    outLen = ChMessageInt_new(len);
    skt_sendN(destFd, &outLen, sizeof(outLen)); /* first comes the length */
    while (len > 0) {
      char buf[m];
      int r = m;
      if (r > len)
        r = len;
      skt_recvN(srcFd, buf, r);
      if (0 == destErrs) /* don't keep sending to dead clients, but *do* clean
                            out srcFd */
        destErrs |= skt_sendN(destFd, buf, r);
      len -= m;
#undef m
    }
    skt_close(destFd);

    skt_set_abort(old);
  }
  return 0;
}

#else
int req_ccs_reply_fw(ChMessage *msg, SOCKET srcFd) {}
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
/** Macro to switch on the case when charmrun stays up even if
one of the processor crashes*/
/*#define __FAULT__*/

SOCKET *req_clients; /*TCP request sockets for each node*/
#ifdef HSTART
SOCKET *charmrun_fds;
#endif
int req_nClients; /*Number of entries in above list (==nodetab_rank0_size)*/
int req_ending = 0;

/* socket and std streams for the gdb info program */
int gdb_info_pid = 0;
int gdb_info_std[3];
FILE *gdb_stream = NULL;

#define REQ_OK 0
#define REQ_FAILED -1

#ifdef HSTART
int req_reply_child(SOCKET fd, const char *type, const char *data, int dataLen)
{

  int status = req_reply(fd, type, data, dataLen);
  if (status != REQ_OK)
    return status;
  SOCKET clientFd;
  skt_recvN(fd, (const char *) &clientFd, sizeof(SOCKET));
  skt_sendN(fd, (const char *) &clientFd, sizeof(fd));
  return status;
}
#endif
/**
 * @brief This is the only place where charmrun talks back to anyone.
 */
int req_reply(SOCKET fd, const char *type, const char *data, int dataLen)
{
  ChMessageHeader msg;
  if (fd == INVALID_SOCKET)
    return REQ_FAILED;
  ChMessageHeader_new(type, dataLen, &msg);
  skt_sendN(fd, (const char *) &msg, sizeof(msg));
  skt_sendN(fd, data, dataLen);
  return REQ_OK;
}

/* Request handlers:
When a client asks us to do something, these are the
routines that actually respond to the request.
*/
/*Stash this new node's control and data ports.
 */
int req_handle_initnode(ChMessage *msg, SOCKET fd)
{
#if CMK_USE_IBVERBS
  int i;
  ChSingleNodeinfo *nodeInfo = (ChSingleNodeinfo *) msg->data;
  //	printf("Charmrun> msg->len %d sizeof(ChSingleNodeinfo) %d
  // sizeof(ChInfiAddr) %d
  //\n",msg->len,sizeof(ChSingleNodeinfo),sizeof(ChInfiAddr));
  if (msg->len !=
      sizeof(ChSingleNodeinfo) +
          (nodetab_rank0_size - 1) * sizeof(ChInfiAddr)) {
    fprintf(stderr, "Charmrun: Bad initnode data length. Aborting\n");
    fprintf(stderr, "Charmrun: possibly because: %s.\n", msg->data);
    exit(1);
  }
  nodeInfo->info.qpList =
      (ChInfiAddr *) malloc(sizeof(ChInfiAddr) * (nodetab_rank0_size - 1));
  memcpy((char *) nodeInfo->info.qpList, &msg->data[sizeof(ChSingleNodeinfo)],
         sizeof(ChInfiAddr) * (nodetab_rank0_size - 1));
/*	for(i=0;i<nodetab_rank0_size-1;i++){
                printf("i %d  0x%0x 0x%0x
   0x%0x\n",i,ChMessageInt(nodeInfo->info.qpList[i].lid),ChMessageInt(nodeInfo->info.qpList[i].qpn),ChMessageInt(nodeInfo->info.qpList[i].psn));
        }*/
#else
  if (msg->len != sizeof(ChSingleNodeinfo)) {
    fprintf(stderr, "Charmrun: Bad initnode data length. Aborting\n");
    fprintf(stderr, "Charmrun: possibly because: %s.\n", msg->data);
    exit(1);
  }
#endif
  nodeinfo_add((ChSingleNodeinfo *) msg->data, fd);
  return REQ_OK;
}

/**
 * @brief Gets the array of node numbers, IPs, and ports. This is used by the
 * node-programs
 * to talk to one another.
 */
int req_handle_initnodetab(ChMessage *msg, SOCKET fd)
{
  ChMessageHeader hdr;
  ChMessageInt_t nNodes = ChMessageInt_new(nodetab_rank0_size);
  ChMessageHeader_new(
      "initnodetab",
      sizeof(ChMessageInt_t) + sizeof(ChNodeinfo) * nodetab_rank0_size, &hdr);
  skt_sendN(fd, (const char *) &hdr, sizeof(hdr));
  skt_sendN(fd, (const char *) &nNodes, sizeof(nNodes));
  skt_sendN(fd, (const char *) nodeinfo_arr,
            sizeof(ChNodeinfo) * nodetab_rank0_size);

  return REQ_OK;
}

#ifdef HSTART
/* Used for fault tolerance with hierarchical start */
int req_handle_initnodetab1(ChMessage *msg, SOCKET fd)
{
  ChMessageHeader hdr;
  ChMessageInt_t nNodes = ChMessageInt_new(nodetab_rank0_size);
  ChMessageHeader_new("initnttab", sizeof(ChMessageInt_t) +
                                       sizeof(ChNodeinfo) * nodetab_rank0_size,
                      &hdr);
  skt_sendN(fd, (const char *) &hdr, sizeof(hdr));
  skt_sendN(fd, (const char *) &nNodes, sizeof(nNodes));
  skt_sendN(fd, (const char *) nodeinfo_arr,
            sizeof(ChNodeinfo) * nodetab_rank0_size);

  return REQ_OK;
}
/*Get the array of node numbers, IPs, and ports.
This is used by the node-programs to talk to one another.
*/
static int parent_charmrun_fd = -1;
int req_handle_initnodedistribution(ChMessage *msg, SOCKET fd, int client)
{
  int nodes_to_fork =
      nodes_per_child; /* rounding should help in better load distribution*/
  int rank0_start = nodetab_unique_table[client * nodes_per_child];
  int rank0_finish;
  if (client == branchfactor - 1) {
    nodes_to_fork = nodetab_unique_size - client * nodes_per_child;
    rank0_finish = nodetab_rank0_size;
  } else
    rank0_finish =
        nodetab_unique_table[client * nodes_per_child + nodes_to_fork];
  int k;
  ChMessageInt_t *nodemsg = (ChMessageInt_t *) malloc(
      (rank0_finish - rank0_start) * sizeof(ChMessageInt_t));
  for (k = 0; k < rank0_finish - rank0_start; k++)
    nodemsg[k] = ChMessageInt_new(nodetab_rank0_table[rank0_start + k]);
  ChMessageHeader hdr;
  ChMessageInt_t nNodes = ChMessageInt_new(rank0_finish - rank0_start);
  ChMessageInt_t nTotalNodes = ChMessageInt_new(nodetab_rank0_size);
  ChMessageHeader_new("initnodetab",
                      sizeof(ChMessageInt_t) * 2 +
                          sizeof(ChMessageInt_t) * (rank0_finish - rank0_start),
                      &hdr);
  skt_sendN(fd, (const char *) &hdr, sizeof(hdr));
  skt_sendN(fd, (const char *) &nNodes, sizeof(nNodes));
  skt_sendN(fd, (const char *) &nTotalNodes, sizeof(nTotalNodes));
  skt_sendN(fd, (const char *) nodemsg,
            (rank0_finish - rank0_start) * sizeof(ChMessageInt_t));
  free(nodemsg);
  return REQ_OK;
}

ChSingleNodeinfo *myNodesInfo;
int send_myNodeInfo_to_parent()
{
  ChMessageHeader hdr;
  ChMessageInt_t nNodes = ChMessageInt_new(nodetab_rank0_size);
  ChMessageHeader_new("initnodetab",
                      sizeof(ChMessageInt_t) +
                          sizeof(ChSingleNodeinfo) * nodetab_rank0_size,
                      &hdr);
  skt_sendN(parent_charmrun_fd, (const char *) &hdr, sizeof(hdr));
  skt_sendN(parent_charmrun_fd, (const char *) &nNodes, sizeof(nNodes));
  skt_sendN(parent_charmrun_fd, (const char *) myNodesInfo,
            sizeof(ChSingleNodeinfo) * nodetab_rank0_size);

  return REQ_OK;
}
void forward_nodetab_to_children()
{
  /*it just needs to receive and copy the nodetab info if required and send it
   * as it is to its nodes */
  if (!skt_select1(parent_charmrun_fd, 1200 * 1000)) {
    exit(0);
  }
  ChMessage msg;
  ChMessage_recv(parent_charmrun_fd, &msg);

  ChMessageInt_t *nodelistmsg = (ChMessageInt_t *) msg.data;
  int nodetab_Nodes = ChMessageInt(nodelistmsg[0]);
  int client;
  for (client = 0; client < nodetab_rank0_size; client++) {
    SOCKET fd = req_clients[client];
    ChMessageHeader hdr;
    ChMessageInt_t nNodes = ChMessageInt_new(nodetab_Nodes);
    ChMessageHeader_new("initnodetab", sizeof(ChMessageInt_t) +
                                           sizeof(ChNodeinfo) * nodetab_Nodes,
                        &hdr);
    skt_sendN(fd, (const char *) &hdr, sizeof(hdr));
    skt_sendN(fd, (const char *) &nNodes, sizeof(nNodes));
    skt_sendN(fd, (const char *) (nodelistmsg + 1),
              sizeof(ChNodeinfo) * nodetab_Nodes);
  }
}
/*Parent Charmrun receives the nodetab from child and processes it. msg contain
 * array of ChSingleNodeInfo*/
void receive_nodeset_from_child(ChMessage *msg, SOCKET fd)
{
  ChMessageInt_t *n32 = (ChMessageInt_t *) msg->data;
  int numOfNodes = ChMessageInt(n32[0]);
  ChSingleNodeinfo *childNodeInfo = (ChSingleNodeinfo *) (n32 + 1);
  int k;
  for (k = 0; k < numOfNodes; k++)
    nodeinfo_add(childNodeInfo + k, fd);
}

void set_sockets_list(ChMessage *msg, SOCKET fd)
{
  ChMessageInt_t *n32 = (ChMessageInt_t *) msg->data;
  int node_start = ChMessageInt(n32[0]);
  charmrun_fds[node_start / nodes_per_child] = fd;
}
#endif
/* Check this return code from "printf". */
static void checkPrintfError(int err)
{
  if (err < 0) {
    static int warned = 0;
    if (!warned) {
      perror("charmrun WARNING> error in printf");
      warned = 1;
    }
  }
}

int req_handle_print(ChMessage *msg, SOCKET fd)
{
  checkPrintfError(printf("%s", msg->data));
  checkPrintfError(fflush(stdout));
  write_stdio_duplicate(msg->data);
  return REQ_OK;
}

int req_handle_printerr(ChMessage *msg, SOCKET fd)
{
  fprintf(stderr, "%s", msg->data);
  fflush(stderr);
  write_stdio_duplicate(msg->data);
  return REQ_OK;
}

int req_handle_printsyn(ChMessage *msg, SOCKET fd)
{
  checkPrintfError(printf("%s", msg->data));
  checkPrintfError(fflush(stdout));
  write_stdio_duplicate(msg->data);
#ifdef HSTART
  if (arg_hierarchical_start)
    req_reply_child(fd, "printdone", "", 1);
  else
#endif
    req_reply(fd, "printdone", "", 1);
  return REQ_OK;
}

int req_handle_printerrsyn(ChMessage *msg, SOCKET fd)
{
  fprintf(stderr, "%s", msg->data);
  fflush(stderr);
  write_stdio_duplicate(msg->data);
#ifdef HSTART
  if (arg_hierarchical_start)
    req_reply_child(fd, "printdone", "", 1);
  else
#endif
    req_reply(fd, "printdone", "", 1);
  return REQ_OK;
}

int req_handle_ending(ChMessage *msg, SOCKET fd)
{
  int i;
  req_ending++;

#if CMK_SHRINK_EXPAND
  // When using shrink-expand, only PE 0 will send an "ending" request.
#elif (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
  if (req_ending == nodetab_size)
#else
  if (req_ending == arg_requested_pes)
#endif
  {
#if CMK_SHRINK_EXPAND
    ChMessage ackmsg;
    ChMessage_new("realloc_ack", 0, &ackmsg);
    for (i = 0; i < req_nClients; i++) {
      ChMessage_send(req_clients[i], &ackmsg);
    }
#endif

    for (i = 0; i < req_nClients; i++)
      skt_close(req_clients[i]);
    if (arg_verbose)
      printf("Charmrun> Graceful exit.\n");
    exit(0);
  }
  return REQ_OK;
}

int req_handle_barrier(ChMessage *msg, SOCKET fd)
{
  int i;
  static int barrier_count = 0;
  static int barrier_phase = 0;
  barrier_count++;
#ifdef HSTART
  if (barrier_count == arg_requested_pes)
#else
  if (barrier_count == req_nClients)
#endif
  {
    barrier_count = 0;
    barrier_phase++;
    for (i = 0; i < req_nClients; i++)
      if (REQ_OK != req_reply(req_clients[i], "barrier", "", 1)) {
        fprintf(stderr, "req_handle_barrier socket error: %d\n", i);
        abort();
      }
  }
  return REQ_OK;
}

int req_handle_barrier0(ChMessage *msg, SOCKET fd)
{
  int i;
  static int count = 0;
  static SOCKET fd0;
  int pe = atoi(msg->data);
  if (pe == 0)
    fd0 = fd;
  count++;
#ifdef HSTART
  if (count == arg_requested_pes)
#else
  if (count == req_nClients)
#endif
  {
    req_reply(fd0, "barrier0", "", 1); /* only send to node 0 */
    count = 0;
  }
  return REQ_OK;
}

void req_handle_abort(ChMessage *msg, SOCKET fd)
{
  /*fprintf(stderr,"req_handle_abort called \n");*/
  if (msg->len == 0)
    fprintf(stderr, "Aborting!\n");
  else
    fprintf(stderr, "%s\n", msg->data);
  exit(1);
}

int req_handle_scanf(ChMessage *msg, SOCKET fd)
{
  char *fmt, *res, *p;

  fmt = msg->data;
  fmt[msg->len - 1] = 0;
  res = input_scanf_chars(fmt);
  p = res;
  while (*p) {
    if (*p == '\n')
      *p = ' ';
    p++;
  }
#ifdef HSTART
  if (arg_hierarchical_start)
    req_reply_child(fd, "scanf-data", res, strlen(res) + 1);
  else
#endif
    req_reply(fd, "scanf-data", res, strlen(res) + 1);
  free(res);
  return REQ_OK;
}

#if CMK_SHRINK_EXPAND
int req_handle_realloc(ChMessage *msg, SOCKET fd)
{
  printf("Charmrun> Realloc request received %s \n", msg->data);

  /* Exec to clear and restart everything, just preserve contents of
   * netstart*/
  int restart_idx = -1;
  for (int i = 0; i < saved_argc; ++i) {
    if (strcmp(saved_argv[i], "+restart") == 0) {
      restart_idx = i;
      break;
    }
  }

  const char *dir = "/dev/shm";
  for (int i = 0; i < saved_argc; ++i) {
    if (strcmp(saved_argv[i], "+shrinkexpand_basedir") == 0) {
      dir = saved_argv[i+1];
      break;
    }
  }

  const char **ret;
  if (restart_idx == -1) {
    ret = (const char **) malloc(sizeof(char *) * (saved_argc + 10));
  } else {
    ret = (const char **) malloc(sizeof(char *) * (saved_argc + 8));
  }

  int newP = *(int *) (msg->data);
  int oldP = arg_requested_pes;
  printf("Charmrun> newp =  %d oldP = %d \n \n \n", newP, oldP);

  int i;
  for (i = 0; i < saved_argc; i++) {
    ret[i] = saved_argv[i];
  }

  ret[saved_argc + 0] = "++newp";

  char sp_buffer[50];
  sprintf(sp_buffer, "%d", newP);
  ret[saved_argc + 1] = sp_buffer;
  ret[saved_argc + 2] = "++shrinkexpand";
  ret[saved_argc + 3] = "++oldp";

  char sp_buffer1[50];
  sprintf(sp_buffer1, "%d", arg_requested_pes);
  ret[saved_argc + 4] = sp_buffer1;

  char sp_buffer2[6];
  sprintf(sp_buffer2, "%d", server_port);
  ret[saved_argc + 5] = "++charmrun_port";
  ret[saved_argc + 6] = sp_buffer2;

  if (restart_idx == -1) {
    ret[saved_argc + 7] = "+restart";
    ret[saved_argc + 8] = dir;
    ret[saved_argc + 9] = NULL;
  } else {
    ret[restart_idx + 1] = dir;
    ret[saved_argc + 7] = NULL;
  }

  setenv("NETSTART", create_netstart(1), 1);
  setenv("OLDNODENAMES", create_oldnodenames(), 1);

  ChMessage ackmsg;
  ChMessage_new("realloc_ack", 0, &ackmsg);
  for (i = 0; i < req_nClients; i++) {
    ChMessage_send(req_clients[i], &ackmsg);
  }

  skt_client_table.clear();
  skt_close(server_fd);
  skt_close(CcsServer_fd());
  execv(ret[0], (char **)ret);
  printf("Should not be here\n");
  exit(1);

  return REQ_OK;
}
#endif

#ifdef __FAULT__
void restart_node(int crashed_node);
void reconnect_crashed_client(int socket_index, int crashed_node);
void announce_crash(int socket_index, int crashed_node);

static int _last_crash = 0;         /* last crashed pe number */
static int _crash_socket_index = 0; /* last restart socket */
#ifdef HSTART
static int _crash_socket_charmrun_index = 0; /* last restart socket */
int crashed_pe_id;
int restarted_pe_id;
#endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
static int numCrashes = 0; /*number of crashes*/
static SOCKET last_crashed_fd = -1;
#endif

/**
 * @brief Handles an ACK after a crash. Once it has received all the pending
 * acks, it sends the nodetab
 * table to the crashed node.
 */
int req_handle_crashack(ChMessage *msg, SOCKET fd)
{
  static int count = 0;
  count++;
#ifdef HSTART
  if (arg_hierarchical_start) {
    if (count == nodetab_rank0_size - 1) {
      /* only after everybody else update its nodetab, can this
         restarted process continue */
      PRINT(("Charmrun> continue node: %d\n", _last_crash));
      req_handle_initnodetab1(NULL, req_clients[_crash_socket_charmrun_index]);
      _last_crash = 0;
      count = 0;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
      last_crashed_fd = -1;
#endif
    }
  }

  else

#endif
      if (count == req_nClients - 1) {
    // only after everybody else update its nodetab, can this restarted process
    // continue
    PRINT(("Charmrun> continue node: %d\n", _last_crash));
    req_handle_initnodetab(NULL, req_clients[_crash_socket_index]);
    _last_crash = 0;
    count = 0;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    last_crashed_fd = -1;
#endif
  }
}

#ifdef HSTART
/* send initnode to root*/
int set_crashed_socket_id(ChMessage *msg, SOCKET fd)
{
  ChSingleNodeinfo *nodeInfo = (ChSingleNodeinfo *) msg->data;
  int nt = nodetab_rank0_table[ChMessageInt(nodeInfo->nodeNo) - mynodes_start];
  nodeInfo->nodeNo = ChMessageInt_new(nt);
  /* Required for CCS */
  /*Nodetable index for this node*/
  int pe;
  for (pe = 0; pe < nodetab_cpus(nt); pe++) {
    nodetab_table[nt + pe]->ctrlfd = fd;
  }
}

/* Receives new dataport of restarted prcoess	and resends nodetable to
 * everyone*/
int req_handle_crash(ChMessage *msg, SOCKET fd)
{

  ChMessageInt_t oldpe, newpe;
  skt_recvN(fd, (const char *) &oldpe, sizeof(oldpe));
  skt_recvN(fd, (const char *) &newpe, sizeof(newpe));
  *nodetab_table[ChMessageInt(oldpe)] = *nodetab_table[ChMessageInt(newpe)];

  int status = req_handle_initnode(msg, fd);
  int i;
  for (i = 0; i < req_nClients; i++) {
    if (req_clients[i] == fd) {
      break;
    }
  }
  _crash_socket_charmrun_index = i;

  fprintf(stderr, "Root charmrun : Socket %d failed %d\n", fd,
          _crash_socket_charmrun_index);
  fflush(stderr);
  ChSingleNodeinfo *nodeInfo = (ChSingleNodeinfo *) msg->data;
  int crashed_node = ChMessageInt(nodeInfo->nodeNo);
  _last_crash = crashed_node;
  switch (status) {
  case REQ_OK:
    break;
  case REQ_FAILED:
    return REQ_FAILED;
  }

  /* Already processed, so send*/
  int client;
  for (client = 0; client < req_nClients; client++) {
    req_handle_initnodetab(NULL, req_clients[client]);
  }

  /*Anounce crash to all child charmruns*/
  announce_crash(nodetab_rank0_size + 1, crashed_node);
}

#endif
#endif

#ifdef __FAULT__
void error_in_req_serve_client(SOCKET fd)
{
  int i;
  int crashed_node, crashed_pe, node_index, socket_index;
  fprintf(stderr, "Socket %d failed \n", fd);

#ifdef HSTART
  if (arg_hierarchical_start) {
    for (i = mynodes_start; i < mynodes_start + nodetab_rank0_size; i++) {
      if (nodetab_ctrlfd(i) == fd) {
        break;
      }
    }
  }

  else
#endif
    for (i = 0; i < nodetab_max; i++) {
      if (nodetab_ctrlfd(i) == fd) {
        break;
      }
    }

  fflush(stdout);
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
  skt_close(fd);
#endif
  crashed_pe = i;
  node_index = i - nodetab_rank(crashed_pe);
  for (i = 0; i < nodetab_rank0_size; i++) {
    if (node_index == nodetab_rank0_table[i]) {
      break;
    }
  }
  crashed_node = i;

  /** should also send a message to all the other processors telling them that
   * this guy has crashed*/
  /*announce_crash(socket_index,crashed_node);*/
  restart_node(crashed_node);

  fprintf(stderr, "charmrun says Processor %d failed on Node %d\n", crashed_pe,
          crashed_node);
  /** after the crashed processor has been recreated
   it connects to charmrun. That data must now be filled
   into the req_nClients array and the nodetab_table*/

  for (i = 0; i < req_nClients; i++) {
    if (req_clients[i] == fd) {
      break;
    }
  }
  socket_index = i;
  reconnect_crashed_client(socket_index, crashed_node);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  skt_close(fd);
#endif
}
#endif

int req_handler_dispatch(ChMessage *msg, SOCKET replyFd)
{
  char *cmd = msg->header.type;
  int recv_status;
  DEBUGF(("Got request '%s'\n", cmd, replyFd));
#if CMK_CCS_AVAILABLE /* CCS *doesn't* want data yet, for faster forwarding */
  if (strcmp(cmd, "reply_fw") == 0)
    return req_ccs_reply_fw(msg, replyFd);
#endif

  /* grab request data */
  recv_status = ChMessageData_recv(replyFd, msg);
#ifdef __FAULT__
#ifdef HSTART
  if (!arg_hierarchical_start)
#endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    if (recv_status < 0) {
      if (replyFd == last_crashed_fd) {
        return REQ_OK;
      }
      DEBUGF(("recv_status %d on socket %d \n", recv_status, replyFd));
      error_in_req_serve_client(replyFd);
    }
#else
  if (recv_status < 0) {
    error_in_req_serve_client(replyFd);
    return REQ_OK;
  }
#endif
#endif

  if (strcmp(cmd, "ping") == 0)
    return REQ_OK;
  else if (strcmp(cmd, "print") == 0)
    return req_handle_print(msg, replyFd);
  else if (strcmp(cmd, "printerr") == 0)
    return req_handle_printerr(msg, replyFd);
  else if (strcmp(cmd, "printsyn") == 0)
    return req_handle_printsyn(msg, replyFd);
  else if (strcmp(cmd, "printerrsyn") == 0)
    return req_handle_printerrsyn(msg, replyFd);
  else if (strcmp(cmd, "scanf") == 0)
    return req_handle_scanf(msg, replyFd);
  else if (strcmp(cmd, "barrier") == 0)
    return req_handle_barrier(msg, replyFd);
  else if (strcmp(cmd, "barrier0") == 0)
    return req_handle_barrier0(msg, replyFd);
  else if (strcmp(cmd, "ending") == 0)
    return req_handle_ending(msg, replyFd);
  else if (strcmp(cmd, "abort") == 0) {
    req_handle_abort(msg, replyFd);
    return REQ_FAILED;
  }
#ifdef __FAULT__
  else if (strcmp(cmd, "crash_ack") == 0)
    return req_handle_crashack(msg, replyFd);
#ifdef HSTART
  else if (strcmp(cmd, "initnode") == 0)
    return req_handle_crash(msg, replyFd);
#endif
#endif
#if CMK_SHRINK_EXPAND
  else if (strcmp(cmd, "realloc") == 0)
    return req_handle_realloc(msg, replyFd);
#endif
  else {
#ifndef __FAULT__
    fprintf(stderr, "Charmrun> Bad control socket request '%s'\n", cmd);
    abort();
    return REQ_OK;
#endif
  }
  return REQ_OK;
}

void req_serve_client(SOCKET fd)
{
  int recv_status;
  int status;
  ChMessage msg;
  DEBUGF(("Getting message from client...\n"));
  recv_status = ChMessageHeader_recv(fd, &msg);
#ifdef __FAULT__
#ifdef HSTART
  if (!arg_hierarchical_start && recv_status < 0)
    error_in_req_serve_client(fd);
#else
  if (recv_status < 0) {
    error_in_req_serve_client(fd);
    return;
  }
#endif
#endif

  DEBUGF(("Message is '%s'\n", msg.header.type));
  status = req_handler_dispatch(&msg, fd);
  switch (status) {
  case REQ_OK:
    break;
  case REQ_FAILED:
    fprintf(stderr, "Charmrun> Error processing control socket request %s\n",
            msg.header.type);
    abort();
    break;
  }
  ChMessage_free(&msg);
}

#ifdef HSTART
void req_forward_root(SOCKET fd)
{
  int recv_status;
  int status;
  ChMessage msg;
  recv_status = ChMessage_recv(fd, &msg);

  char *cmd = msg.header.type;

#ifdef __FAULT__
  if (recv_status < 0) {
    error_in_req_serve_client(fd);
    return;
  }

  /*called from reconnect_crashed_client */
  if (strcmp(cmd, "initnode") == 0) {
    set_crashed_socket_id(&msg, fd);
  }
#endif

  if (strcmp(cmd, "ping") != 0) {
    status = req_reply(parent_charmrun_fd, cmd, msg.data,
                       ChMessageInt(msg.header.len));

    if (strcmp(cmd, "scanf") == 0 || strcmp(cmd, "printsyn") == 0 ||
        strcmp(cmd, "printerrsyn") == 0)
      skt_sendN(parent_charmrun_fd, (const char *) &fd, sizeof(fd));

#ifdef __FAULT__
    if (strcmp(cmd, "initnode") == 0) {
      ChMessageInt_t oldpe = ChMessageInt_new(crashed_pe_id);
      ChMessageInt_t newpe = ChMessageInt_new(restarted_pe_id);
      skt_sendN(parent_charmrun_fd, (const char *) &oldpe, sizeof(oldpe));
      skt_sendN(parent_charmrun_fd, (const char *) &newpe, sizeof(newpe));
    }
#endif
  }

  switch (status) {
  case REQ_OK:
    break;
  case REQ_FAILED:
    abort();
    break;
  }
  ChMessage_free(&msg);
}

void req_forward_client()
{
  int recv_status;
  int status;
  ChMessage msg;
  recv_status = ChMessage_recv(parent_charmrun_fd, &msg);
  if (recv_status < 0) {
    int i;
    for (i = 0; i < req_nClients; i++)
      skt_close(req_clients[i]);
    exit(0);
  }

  char *cmd = msg.header.type;

  if (strcmp(cmd, "barrier") == 0) {
    int i;
    for (i = 0; i < req_nClients; i++)
      if (REQ_OK != req_reply(req_clients[i], cmd, msg.data,
                              ChMessageInt(msg.header.len))) {
        abort();
      }
    return;
  }
#ifdef __FAULT__
  if (strcmp(cmd, "initnodetab") == 0) {
    if (_last_crash == 0)
      current_restart_phase++;
    int i;
    for (i = 0; i < req_nClients; i++)
      if (_last_crash == 0 || i != _crash_socket_index)
        if (REQ_OK != req_reply(req_clients[i], cmd, msg.data,
                                ChMessageInt(msg.header.len))) {
          abort();
        }
    return;
  }

  if (strcmp(cmd, "crashnode") == 0) {

    int i;
    for (i = 0; i < req_nClients; i++)
      if (_last_crash == 0 || i != _crash_socket_index)
        if (REQ_OK != req_reply(req_clients[i], cmd, msg.data,
                                ChMessageInt(msg.header.len))) {
          abort();
        }
    return;
  }
  if (strcmp(cmd, "initnttab") == 0) {
    _last_crash = 0;
    if (REQ_OK != req_reply(req_clients[_crash_socket_index], "initnodetab",
                            msg.data, ChMessageInt(msg.header.len))) {
      abort();
    }
    return;
  }

#endif

  SOCKET fd;

  /* CCS forward request */
  if (strcmp(cmd, "req_fw") == 0) {
    CcsImplHeader *hdr = (CcsImplHeader *) msg.data;
    int pe = ChMessageInt(hdr->pe);
    fd = nodetab_table[pe]->ctrlfd;
  } else if (strcmp(cmd, "barrier0") == 0) {
    fd = nodetab_table[0]->ctrlfd;
  } else
    skt_recvN(parent_charmrun_fd, (const char *) &fd, sizeof(SOCKET));

  status = req_reply(fd, cmd, msg.data, ChMessageInt(msg.header.len));

  switch (status) {
  case REQ_OK:
    break;
  case REQ_FAILED:
    abort();
    break;
  }
  ChMessage_free(&msg);
}

#endif

int ignore_socket_errors(SOCKET skt, int c, const char *m)
{ /*Abandon on further socket errors during error shutdown*/

#ifndef __FAULT__
  exit(2);
#endif
  return -1;
}

/*A socket went bad somewhere!  Immediately disconnect,
which kills everybody.
*/
int socket_error_in_poll(SOCKET skt, int code, const char *msg)
{
  /*commenting it for fault tolerance*/
  /*ifdef it*/
  int i;
  skt_set_abort(ignore_socket_errors);
  const char *name = skt_to_name(skt);
  fprintf(stderr, "Charmrun> error on request socket to node %d '%s'--\n"
                  "%s\n",
          skt_to_node(skt), name, msg);
#ifndef __FAULT__
  for (i = 0; i < req_nClients; i++)
    skt_close(req_clients[i]);
  exit(1);
#endif
  ftTimer = GetClock();
  return -1;
}

#if CMK_USE_POLL /*poll() version*/
#define CMK_PIPE_DECL(maxn, delayMs)                                           \
  static struct pollfd *fds = NULL;                                            \
  int nFds_sto = 0;                                                            \
  int *nFds = &nFds_sto;                                                       \
  int pollDelayMs = delayMs;                                                   \
  if (fds == NULL)                                                             \
    fds = (struct pollfd *) malloc((maxn) * sizeof(struct pollfd));
#define CMK_PIPE_SUB fds, nFds
#define CMK_PIPE_CALL()                                                        \
  poll(fds, *nFds, pollDelayMs);                                               \
  *nFds = 0

#define CMK_PIPE_PARAM struct pollfd *fds, int *nFds
#define CMK_PIPE_ADDREAD(rd_fd)                                                \
  do {                                                                         \
    fds[*nFds].fd = rd_fd;                                                     \
    fds[*nFds].events = POLLIN;                                                \
    (*nFds)++;                                                                 \
  } while (0)
#define CMK_PIPE_ADDWRITE(wr_fd)                                               \
  do {                                                                         \
    fds[*nFds].fd = wr_fd;                                                     \
    fds[*nFds].events = POLLOUT;                                               \
    (*nFds)++;                                                                 \
  } while (0)
#define CMK_PIPE_CHECKREAD(rd_fd) fds[(*nFds)++].revents &POLLIN
#define CMK_PIPE_CHECKWRITE(wr_fd) fds[(*nFds)++].revents &POLLOUT

#else /*select() version*/

#define CMK_PIPE_DECL(maxn, delayMs)                                           \
  fd_set rfds_sto, wfds_sto;                                                   \
  int nFds = 0;                                                                \
  fd_set *rfds = &rfds_sto, *wfds = &wfds_sto;                                 \
  struct timeval tmo;                                                          \
  FD_ZERO(rfds);                                                               \
  FD_ZERO(wfds);                                                               \
  tmo.tv_sec = delayMs / 1000;                                                 \
  tmo.tv_usec = 1000 * (delayMs % 1000);
#define CMK_PIPE_SUB rfds, wfds
#define CMK_PIPE_CALL() select(FD_SETSIZE, rfds, 0, 0, &tmo)

#define CMK_PIPE_PARAM fd_set *rfds, fd_set *wfds
#define CMK_PIPE_ADDREAD(rd_fd)                                                \
  {                                                                            \
    assert(nFds < FD_SETSIZE);                                                 \
    FD_SET(rd_fd, rfds);                                                       \
    nFds++;                                                                    \
  }
#define CMK_PIPE_ADDWRITE(wr_fd) FD_SET(wr_fd, wfds)
#define CMK_PIPE_CHECKREAD(rd_fd) FD_ISSET(rd_fd, rfds)
#define CMK_PIPE_CHECKWRITE(wr_fd) FD_ISSET(wr_fd, wfds)
#endif

/*
Wait for incoming requests on all client sockets,
and the CCS socket (if present).
*/
void req_poll()
{
  int status, i;
  int readcount;

  CMK_PIPE_DECL(req_nClients + 5, 1000);
  for (i = 0; i < req_nClients; i++)
    CMK_PIPE_ADDREAD(req_clients[i]);
  if (CcsServer_fd() != INVALID_SOCKET)
    CMK_PIPE_ADDREAD(CcsServer_fd());
  if (arg_charmdebug) {
    CMK_PIPE_ADDREAD(0);
    CMK_PIPE_ADDREAD(gdb_info_std[1]);
    CMK_PIPE_ADDREAD(gdb_info_std[2]);
  }

  skt_set_abort(socket_error_in_poll);

  DEBUGF(("Req_poll: Calling select...\n"));
  status = CMK_PIPE_CALL();
  DEBUGF(("Req_poll: Select returned %d...\n", status));

  if (status == 0)
    return; /*Nothing to do-- timeout*/

  if (status < 0) {
    if (errno == EINTR || errno == EAGAIN)
      return;
    fflush(stdout);
    fflush(stderr);
    socket_error_in_poll(-1, 1359, "Node program terminated unexpectedly!\n");
  }
  for (i = 0; i < req_nClients; i++)
    if (CMK_PIPE_CHECKREAD(req_clients[i])) {
      readcount = 10; /*number of successive reads we serve per socket*/
      /*This client is ready to read*/
      do {
        req_serve_client(req_clients[i]);
        readcount--;
      } while (1 == skt_select1(req_clients[i], 0) && readcount > 0);
    }

  if (CcsServer_fd() != INVALID_SOCKET)
    if (CMK_PIPE_CHECKREAD(CcsServer_fd())) {
      DEBUGF(("Activity on CCS server port...\n"));
      req_ccs_connect();
    }

  if (arg_charmdebug) {
    char buf[2048];
    if (CMK_PIPE_CHECKREAD(0)) {
      int indata = read(0, buf, 5);
      buf[indata] = 0;
      if (indata < 5)
        fprintf(stderr, "Error reading command (%s)\n", buf);
      if (strncmp(buf, "info:", 5) == 0) {
        /* Found info command, forward data to gdb info program */
        char c;
        int num = 0;
        // printf("Command to be forwarded\n");
        while (read(0, &c, 1) != -1) {
          buf[num++] = c;
          if (c == '\n' || num >= 2045) {
            write(gdb_info_std[0], buf, num);
            if (c == '\n')
              break;
          }
        }
      }
      // printf("Command from charmdebug: %d(%s)\n",indata,buf);
    }
    /* All streams from gdb are forwarded to the stderr stream through the FILE
       gdb_stream which has been duplicated from stderr */
    /* NOTE: gdb_info_std[2] must be flushed before gdb_info_std[1] because the
       latter contains the string "(gdb) " ending the synchronization. Also the
       std[1] should be read with the else statement. It will not work without.
       */
    if (CMK_PIPE_CHECKREAD(gdb_info_std[2])) {
      int indata = read(gdb_info_std[2], buf, 100);
      /*printf("read data from gdb info stderr %d\n",indata);*/
      if (indata > 0) {
        buf[indata] = 0;
        // printf("printing %s\n",buf);
        // fflush(stdout);
        // fprintf(gdb_stream,"%s",buf);
        fflush(gdb_stream);
      }
    } else if (CMK_PIPE_CHECKREAD(gdb_info_std[1])) {
      int indata = read(gdb_info_std[1], buf, 100);
      /*printf("read data from gdb info stdout %d\n",indata);*/
      if (indata > 0) {
        buf[indata] = 0;
        // printf("printing %s\n",buf);
        // fflush(stdout);
        fprintf(gdb_stream, "%s", buf);
        fflush(gdb_stream);
      }
    }
  }
}

#ifdef HSTART
void req_poll_hierarchical()
{
  int status, i;
  fd_set rfds;
  struct timeval tmo;
  int readcount;

  skt_set_abort(socket_error_in_poll);

  tmo.tv_sec = 1;
  tmo.tv_usec = 0;
  FD_ZERO(&rfds); /* clears set of file descriptor */
  for (i = 0; i < req_nClients; i++)
    FD_SET(req_clients[i], &rfds); /* adds client sockets to rfds set*/
  if (CcsServer_fd() != INVALID_SOCKET)
    FD_SET(CcsServer_fd(), &rfds);
  if (arg_charmdebug) {
    FD_SET(0, &rfds);
    FD_SET(gdb_info_std[1], &rfds);
    FD_SET(gdb_info_std[2], &rfds);
  }

  if (arg_child_charmrun)
    FD_SET(parent_charmrun_fd, &rfds); /* adds client sockets to rfds set*/
  DEBUGF(("Req_poll: Calling select...\n"));
  status = select(FD_SETSIZE, &rfds, 0, 0,
                  &tmo); /* FD_SETSIZE is the maximum number of file
                            descriptors that a fd_set object can hold
                            information about, select returns number of
                            polls gathered */
  DEBUGF(("Req_poll: Select returned %d...\n", status));

  if (status == 0)
    return; /*Nothing to do-- timeout*/
  if (status < 0) {
    fflush(stdout);
    fflush(stderr);
    socket_error_in_poll(1359, "Node program terminated unexpectedly!\n");
  }
  for (i = 0; i < req_nClients; i++)
    if (FD_ISSET(req_clients[i], &rfds)) {
      readcount = 10; /*number of successive reads we serve per socket*/
      /*This client is ready to read*/
      do {
        if (arg_child_charmrun)
          req_forward_root(req_clients[i]);
        else
          req_serve_client(req_clients[i]);
        readcount--;
      } while (1 == skt_select1(req_clients[i], 0) && readcount > 0);
    }

  if (arg_child_charmrun)
    // Forward from root to clients
    if (FD_ISSET(parent_charmrun_fd, &rfds)) {
      readcount = 10; /*number of successive reads we serve per socket*/
      do {
        req_forward_client();
        readcount--;
      } while (1 == skt_select1(parent_charmrun_fd, 0) && readcount > 0);
    }

  /*Wait to receive responses and Forward responses */
  if (CcsServer_fd() != INVALID_SOCKET)
    if (FD_ISSET(CcsServer_fd(), &rfds)) {
      DEBUGF(("Activity on CCS server port...\n"));
      req_ccs_connect();
    }

  if (arg_charmdebug) {
    char buf[2048];
    if (FD_ISSET(0, &rfds)) {
      int indata = read(0, buf, 5);
      buf[indata] = 0;
      if (indata < 5)
        fprintf(stderr, "Error reading command (%s)\n", buf);
      if (strncmp(buf, "info:", 5) == 0) {
        /* Found info command, forward data to gdb info program */
        char c;
        int num = 0;
        // printf("Command to be forwarded\n");
        while (read(0, &c, 1) != -1) {
          buf[num++] = c;
          if (c == '\n' || num >= 2045) {
            write(gdb_info_std[0], buf, num);
            if (c == '\n')
              break;
          }
        }
      }
      // printf("Command from charmdebug: %d(%s)\n",indata,buf);
    }
    /* All streams from gdb are forwarded to the stderr stream through the FILE
       gdb_stream which has been duplicated from stderr */
    /* NOTE: gdb_info_std[2] must be flushed before gdb_info_std[1] because the
       latter contains the string "(gdb) " ending the synchronization. Also the
       std[1] should be read with the else statement. It will not work without.
       */
    if (FD_ISSET(gdb_info_std[2], &rfds)) {
      int indata = read(gdb_info_std[2], buf, 100);
      /*printf("read data from gdb info stderr %d\n",indata);*/
      if (indata > 0) {
        buf[indata] = 0;
        // printf("printing %s\n",buf);
        // fflush(stdout);
        // fprintf(gdb_stream,"%s",buf);
        fflush(gdb_stream);
      }
    } else if (FD_ISSET(gdb_info_std[1], &rfds)) {
      int indata = read(gdb_info_std[1], buf, 100);
      /*printf("read data from gdb info stdout %d\n",indata);*/
      if (indata > 0) {
        buf[indata] = 0;
        // printf("printing %s\n",buf);
        // fflush(stdout);
        fprintf(gdb_stream, "%s", buf);
        fflush(gdb_stream);
      }
    }
  }
}
#endif

#ifdef HSTART
static skt_ip_t parent_charmrun_IP;
static int parent_charmrun_port;
static int parent_charmrun_pid;
static int dataport;
static SOCKET dataskt;
int charmrun_phase = 0;
#endif

int client_connect_problem(SOCKET skt, int code, const char *msg)
{ /*Called when something goes wrong during a client connect*/
  const char *name = skt_to_name(skt);
  fprintf(stderr, "Charmrun> error attaching to node '%s':\n%s\n", name, msg);
  exit(1);
  return -1;
}

/** return 1 if connection is openned succesfully with client**/
int errorcheck_one_client_connect(int client)
{
#ifdef HSTART
  /* Child charmruns are already connected - Do we need to conect again*/
  if (arg_hierarchical_start && !arg_child_charmrun && charmrun_phase == 1)
    return 1;
#endif
  /* FIXME: The error printing functions do a table lookup on the socket to
   *        figure their corresponding host. However, certain failures happen
   *        before we can associate a socket with a particular client, as in
   *        skt_select1 below. In that case, we use a workaround to create a
   *        dummy socket so that the internal error message is printed
   * correctly.
   */
  SOCKET dummy_skt = -10;
  skt_client_table[dummy_skt] = client;

  unsigned int clientPort; /*These are actually ignored*/
  skt_ip_t clientIP;
  if (arg_verbose)
    printf("Charmrun> Waiting for %d-th client to connect.\n", client);
  /* FIXME: why are we passing the client as an error code here? */
  if (0 == skt_select1(server_fd, arg_timeout * 1000))
    client_connect_problem(dummy_skt, client,
                           "Timeout waiting for node-program to connect");

  req_clients[client] = skt_accept(server_fd, &clientIP, &clientPort);
  skt_client_table[req_clients[client]] = client;

  /* FIXME: will this ever be triggered? It seems the skt_abort handler here is
   *        'client_connect_problem', which calls exit(1), so we'd exit
   *        in skt_accept. */
  if (req_clients[client] == SOCKET_ERROR)
    client_connect_problem(dummy_skt, client, "Failure in node accept");

  skt_tcp_no_nagle(req_clients[client]);

  return 1;
};

#if CMK_C_INLINE
inline static
#endif
    void
    read_initnode_one_client(int client)
{
  ChMessage msg;
  if (!skt_select1(req_clients[client], arg_timeout * 1000))
    client_connect_problem(req_clients[client], client,
                           "Timeout on IP request");
  ChMessage_recv(req_clients[client], &msg);
  req_handle_initnode(&msg, req_clients[client]);
  ChMessage_free(&msg);
}

#if CMK_IBVERBS_FAST_START
void req_one_client_partinit(int client)
{
  ChMessage partStartMsg;
  int clientNode;

  if (errorcheck_one_client_connect(client)) {
    if (!skt_select1(req_clients[client], arg_timeout * 1000))
      client_connect_problem(req_clients[client], client,
                             "Timeout on partial init request");

    ChMessage_recv(req_clients[client], &partStartMsg);
    clientNode = ChMessageInt(*(ChMessageInt_t *) partStartMsg.data);
    assert(strncmp(partStartMsg.header.type, "partinit", 8) == 0);
    ChMessage_free(&partStartMsg);
  }
};
#endif

#ifdef HSTART
int nodeCount = 0;
/* To keep a global node numbering */
void add_singlenodeinfo_to_mynodeinfo(ChMessage *msg, SOCKET ctrlfd)
{
  /*add to myNodesInfo */
  ChSingleNodeinfo *nodeInfo = (ChSingleNodeinfo *) msg->data;

  /* need to change nodeNo */
  myNodesInfo[nodeCount].nodeNo = ChMessageInt_new(
      nodetab_rank0_table[ChMessageInt(nodeInfo->nodeNo) - mynodes_start]);
  myNodesInfo[nodeCount++].info = nodeInfo->info;

  /* Required for CCS */
  int nt = nodetab_rank0_table[ChMessageInt(nodeInfo->nodeNo) -
                               mynodes_start]; /*Nodetable index for this node*/
  int pe;
  for (pe = 0; pe < nodetab_cpus(nt); pe++) {
    nodetab_table[nt + pe]->ctrlfd = ctrlfd;
  }
}
#endif

#ifndef HSTART
/* Original Function, need to check if modifications required*/
void req_set_client_connect(int start, int end)
{
  fd_set sockset;
  ChMessage msg;
  int client, i;
  int done, maxdesc;
  int *finished;  // -1 if client i not finished, otherwise the node id of client i
  int curclient, curclientend, curclientstart;

  curclient = curclientend = curclientstart = start;

  finished = (int *) malloc((end - start) * sizeof(int));
  for (i = 0; i < (end - start); i++)
    finished[i] = -1;

#if CMK_USE_IBVERBS && !CMK_IBVERBS_FAST_START
  for (i = start; i < end; i++) {
    errorcheck_one_client_connect(curclientend++);
  }
  if (req_nClients > 1) {
    /*  a barrier to make sure infiniband device gets initialized */
    for (i = start; i < end; i++)
      ChMessage_recv(req_clients[i], &msg);
    for (i = start; i < end; i++)
      req_reply(req_clients[i], "barrier", "", 1);
  }
#endif

  done = 0;
  while (!done) {
/* check server socket for messages */
#if !CMK_USE_IBVERBS || CMK_IBVERBS_FAST_START
    while (curclientstart == curclientend || skt_select1(server_fd, 1) != 0) {
      errorcheck_one_client_connect(curclientend++);
    }
#endif
    /* check appropriate clients for messages */
    for (client = curclientstart; client < curclientend; client++)
      if (req_clients[client] > 0) {
        if (skt_select1(req_clients[client], 1) != 0) {
          ChMessage_recv(req_clients[client], &msg);
          req_handle_initnode(&msg, req_clients[client]);
          finished[client - start] = 
            ChMessageInt(((ChSingleNodeinfo *)msg.data)->nodeNo);
        }
      }

    /* test if done */
    done = 1;
    for (i = curclientstart - start; i < (end - start); i++)
      if (finished[i] == -1) {
        curclientstart = start + i;
        done = 0;
        break;
      }
  }
  ChMessage_free(&msg);

  // correct mapping in skt_client_table so that socket points to node using the socket
  for (i = start; i < (end - start); i++)
    skt_client_table[req_clients[i]] = finished[i];

  free(finished);
}
#else
/*int charmrun_phase =0; meaningful for main charmun to decide what to receive*/
void req_set_client_connect(int start, int end)
{
  fd_set sockset;
  ChMessage msg;
  int client, i;
  int done, maxdesc;
  int *finished;  // -1 if client i not finished, otherwise the node id of client i
  int curclient, curclientend, curclientstart;

  curclient = curclientend = curclientstart = start;

  finished = malloc((end - start) * sizeof(int));
  for (i = 0; i < (end - start); i++)
    finished[i] = -1;

  if (arg_child_charmrun && start == 0)
    myNodesInfo = malloc(sizeof(ChSingleNodeinfo) * nodetab_rank0_size);

#if CMK_USE_IBVERBS && !CMK_IBVERBS_FAST_START
  for (i = start; i < end; i++) {
    errorcheck_one_client_connect(curclientend++);
  }
  if (req_nClients > 1) {
    /*  a barrier to make sure infiniband device gets initialized */
    for (i = start; i < end; i++)
      ChMessage_recv(req_clients[i], &msg);
    for (i = start; i < end; i++)
      req_reply(req_clients[i], "barrier", "", 1);
  }
#endif

  done = 0;
  while (!done) {
/* check server socket for messages */
#if !CMK_USE_IBVERBS || CMK_IBVERBS_FAST_START
    while (curclientstart == curclientend || skt_select1(server_fd, 1) != 0) {
      errorcheck_one_client_connect(curclientend++);
    }
#endif
    /* check appropriate clients for messages */
    for (client = curclientstart; client < curclientend; client++)
      if (req_clients[client] > 0) {
        if (skt_select1(req_clients[client], 1) != 0) {
          ChMessage_recv(req_clients[client], &msg);
          if (!arg_hierarchical_start)
            req_handle_initnode(&msg, req_clients[client]);
          else {
            if (!arg_child_charmrun) {
              if (charmrun_phase == 1)
                receive_nodeset_from_child(&msg, req_clients[client]);
              else
                set_sockets_list(&msg, req_clients[client]);
              // here we need to decide based upon the phase
            } else /* hier-start with 2nd leval*/
              add_singlenodeinfo_to_mynodeinfo(&msg, req_clients[client]);
          }
          finished[client - start] = 
              ChMessageInt(((ChSingleNodeinfo *)msg.data)->nodeNo);
        }
      }

    /* test if done */
    done = 1;
    for (i = curclientstart - start; i < (end - start); i++)
      if (finished[i] == -1) {
        curclientstart = start + i;
        done = 0;
        break;
      }
  }
  ChMessage_free(&msg);

  // correct mapping in skt_client_table so that socket points to node using the socket
  for (i = start; i < (end - start); i++)
    skt_client_table[req_clients[i]] = finished[i];

  free(finished);
}
#endif

/* allow one client to connect */
void req_one_client_connect(int client)
{
  if (errorcheck_one_client_connect(
          client)) { /*This client has just connected-- fetch his name and IP*/
    read_initnode_one_client(client);
  }
}

#if CMK_USE_IBVERBS
/* Each node has sent the qpn data for all the qpns it has created
   This data needs to be sent to all the other nodes
         This needs to be done for all nodes
**/
void exchange_qpdata_clients()
{
  int proc, i;
  for (i = 0; i < nodetab_rank0_size; i++) {
    int nt = nodetab_rank0_table[i]; /*Nodetable index for this node*/
    nodetab_table[nt]->qpData =
        (ChInfiAddr *) malloc(sizeof(ChInfiAddr) * nodetab_rank0_size);
  }
  for (proc = 0; proc < nodetab_rank0_size; proc++) {
    int count = 0;
    for (i = 0; i < nodetab_rank0_size; i++) {
      if (i == proc) {
      } else {
        int nt = nodetab_rank0_table[i]; /*Nodetable index for this node*/
        nodetab_table[nt]->qpData[proc] = nodeinfo_arr[proc].qpList[count];
        //			printf("Charmrun> nt %d proc %d lid 0x%x qpn
        // 0x%x
        // psn
        // 0x%x\n",nt,proc,ChMessageInt(nodetab_table[nt]->qpData[proc].lid),ChMessageInt(nodetab_table[nt]->qpData[proc].qpn),ChMessageInt(nodetab_table[nt]->qpData[proc].psn));
        count++;
      }
    }
    free(nodeinfo_arr[proc].qpList);
  }
};

void send_clients_nodeinfo_qpdata()
{
  int node;
  int msgSize = sizeof(ChMessageInt_t) +
                sizeof(ChNodeinfo) * nodetab_rank0_size +
                sizeof(ChInfiAddr) * nodetab_rank0_size;
  for (node = 0; node < nodetab_rank0_size; node++) {
    int nt = nodetab_rank0_table[node]; /*Nodetable index for this node*/
    //		printf("Charmrun> Node %d proc %d sending initnodetab
    //\n",node,nt);
    ChMessageHeader hdr;
    ChMessageInt_t nNodes = ChMessageInt_new(nodetab_rank0_size);
    ChMessageHeader_new("initnodetab", msgSize, &hdr);
    skt_sendN(nodetab_table[nt]->ctrlfd, (const char *) &hdr, sizeof(hdr));
    skt_sendN(nodetab_table[nt]->ctrlfd, (const char *) &nNodes,
              sizeof(nNodes));
    skt_sendN(nodetab_table[nt]->ctrlfd, (const char *) nodeinfo_arr,
              sizeof(ChNodeinfo) * nodetab_rank0_size);
    skt_sendN(nodetab_table[nt]->ctrlfd,
              (const char *) &nodetab_table[nt]->qpData[0],
              sizeof(ChInfiAddr) * nodetab_rank0_size);
  }
}
#endif

struct timeval tim;
#define getthetime(x)                                                          \
  gettimeofday(&tim, NULL);                                                    \
  x = tim.tv_sec + (tim.tv_usec / 1000000.0);
#define getthetime1(x)                                                         \
  gettimeofday(&tim, NULL);                                                    \
  x = tim.tv_sec;
/*Wait for all the clients to connect to our server port*/
void req_client_connect(void)
{
  int client;
#ifdef HSTART
  if (!arg_hierarchical_start)
#endif
    nodeinfo_allocate();
  req_nClients = nodetab_rank0_size;
  req_clients = (SOCKET *) malloc(req_nClients * sizeof(SOCKET));
  for (client = 0; client < req_nClients; client++)
    req_clients[client] = -1;

  skt_set_abort(client_connect_problem);

#if CMK_IBVERBS_FAST_START
  for (client = 0; client < req_nClients; client++) {
    req_one_client_partinit(client);
  }
  for (client = 0; client < req_nClients; client++) {
    read_initnode_one_client(client);
  }
#else

  req_set_client_connect(0, req_nClients);

#endif

  if (portOk == 0)
    exit(1);
  if (arg_verbose)
    printf("Charmrun> All clients connected.\n");
#if CMK_USE_IBVERBS
  exchange_qpdata_clients();
  send_clients_nodeinfo_qpdata();
#else
#ifdef HSTART
  if (arg_hierarchical_start) {
    /* first we need to send data to parent charmrun and then send the nodeinfo
     * to the clients*/
    send_myNodeInfo_to_parent();
    /*then receive from root */
    forward_nodetab_to_children();
  }

  else
#endif
    for (client = 0; client < req_nClients; client++) {
      req_handle_initnodetab(NULL, req_clients[client]);
    }

#endif
  if (arg_verbose)
    printf("Charmrun> IP tables sent.\n");
}

/*Wait for all the clients to connect to our server port, then collect and send
 * nodetable to all */
#ifdef HSTART
void req_charmrun_connect(void)
{
  //	double t1, t2, t3, t4;
  int client;
  nodeinfo_allocate();
  req_nClients = branchfactor;
  req_clients = (SOCKET *) malloc(req_nClients * sizeof(SOCKET));
  charmrun_fds = (SOCKET *) malloc(req_nClients * sizeof(SOCKET));
  for (client = 0; client < req_nClients; client++)
    req_clients[client] = -1;

  skt_set_abort(client_connect_problem);

#if CMK_IBVERBS_FAST_START
  for (client = 0; client < req_nClients; client++) {
    req_one_client_partinit(client);
  }
  for (client = 0; client < req_nClients; client++) {
    read_initnode_one_client(client);
  }
#else
  // if(!arg_child_charmrun) getthetime(t1);

  req_set_client_connect(0, req_nClients);
// if(!arg_child_charmrun)	getthetime(t2);		/* also need to process
// received nodesets JIT */
#endif

  if (portOk == 0)
    exit(1);
  if (arg_verbose)
    printf("Charmrun> All clients connected.\n");
#if CMK_USE_IBVERBS
  exchange_qpdata_clients();
  send_clients_nodeinfo_qpdata();
#else
  for (client = 0; client < req_nClients; client++) {
    // add flag to check what leval charmrun it is and what phase
    req_handle_initnodedistribution(NULL, charmrun_fds[client], client);
  }
  // getthetime(t3);

  /* Now receive the nodetab from child charmruns*/
  charmrun_phase = 1;

  skt_set_abort(client_connect_problem);

  req_set_client_connect(0, req_nClients);

  /* Already processed, so send*/
  for (client = 0; client < req_nClients; client++) {
    req_handle_initnodetab(NULL, req_clients[client]);
  }
// if(!arg_child_charmrun) getthetime(t4);
#endif
  if (arg_verbose)
    printf("Charmrun> IP tables sent.\n");
  // if(!arg_child_charmrun) printf("Time for charmruns connect= %f , sending
  // nodes to fire= %f, node clients connected= %f n ", t2-t1, t3-t2, t4-t3);
}

#endif

#ifndef CMK_BPROC

void start_one_node_ssh(int rank0no);
void finish_one_node(int rank0no);
void finish_set_nodes(int start, int stop);
int start_set_node_ssh(int client);

void req_client_start_and_connect(void)
{
  int client, c;
  int batch = arg_batch_spawn; /* fire several at a time */
  int clientgroup, clientstart;
  int counter;

#ifdef HSTART
  if (!arg_hierarchical_start)
#endif
    nodeinfo_allocate();
  req_nClients = nodetab_rank0_size;
  req_clients = (SOCKET *) malloc(req_nClients * sizeof(SOCKET));

  skt_set_abort(client_connect_problem);

  client = 0;
  while (client < req_nClients) { /* initiate a batch */
    clientstart = client;

    for (counter = 0; counter < batch;
         counter++) { /* initiate batch number of nodes */
      clientgroup = start_set_node_ssh(client);
      client += clientgroup;
      if (client >= req_nClients) {
        client = req_nClients;
        break;
      }
    }
#if CMK_USE_SSH
    /* ssh x11 forwarding will make sure ssh exit */
    if (!arg_ssh_display)
#endif
      finish_set_nodes(clientstart, client);

#if CMK_IBVERBS_FAST_START
    for (c = clientstart; c < client; c++) {
      req_one_client_partinit(c);
    }
#else
    req_set_client_connect(clientstart, client);
#endif
  }

#if CMK_IBVERBS_FAST_START
  for (client = 0; client < req_nClients; client++) {
    read_initnode_one_client(client);
  }
#endif
  if (portOk == 0)
    exit(1);
  if (arg_verbose)
    printf("Charmrun> All clients connected.\n");

#if CMK_USE_IBVERBS
  exchange_qpdata_clients();
  send_clients_nodeinfo_qpdata();
#else
#ifdef HSTART
  if (arg_hierarchical_start) {
    /* first we need to send data to parent charmrun and then send the nodeinfo
     * to the clients*/
    send_myNodeInfo_to_parent();
    /*then receive from root */
    forward_nodetab_to_children();
  }

  else
#endif
    for (client = 0; client < req_nClients; client++) {
      req_handle_initnodetab(NULL, req_clients[client]);
    }

#endif
  if (arg_verbose)
    printf("Charmrun> IP tables sent.\n");
  free(ssh_pids); /* done with ssh_pids */
}

#endif

/*Start the server socket the clients will connect to.*/
void req_start_server(void)
{
  skt_ip_t ip = skt_innode_my_ip();
  server_port = 0;
#if CMK_SHRINK_EXPAND
  if (arg_shrinkexpand) { // Need port information
    char *ns;
    int nread;
    int port;
    ns = getenv("NETSTART");
    if (ns != 0) { /*Read values set by Charmrun*/
      int node_num, old_charmrun_pid;
      char old_charmrun_name[1024 * 1000];
      nread = sscanf(ns, "%d%s%d%d%d", &node_num, old_charmrun_name,
                     &server_port, &old_charmrun_pid, &port);
      if (nread != 5) {
        fprintf(stderr, "Error parsing NETSTART '%s'\n", ns);
        exit(1);
      }
    }
  }
#endif
  if (arg_local)
    /* local execution, use localhost always */
    strcpy(server_addr, "127.0.0.1");
  else if (arg_charmrunip != NULL)
    /* user specify the IP at +useip */
    strcpy(server_addr, arg_charmrunip);
  else if ((arg_charmrunip = getenv("CHARMRUN_IP")) != NULL)
    /* user specify the env  */
    strcpy(server_addr, arg_charmrunip);
  else if (skt_ip_match(ip, _skt_invalid_ip)) {
    fprintf(stderr, "Charmrun> Warning-- cannot find IP address for your hostname.  "
           "Using loopback.\n");
    strcpy(server_addr, "127.0.0.1");
  } else if (arg_usehostname || skt_ip_match(ip, skt_lookup_ip("127.0.0.1")))
    /*Use symbolic host name as charmrun address*/
    gethostname(server_addr, sizeof(server_addr));
  else
    skt_print_ip(server_addr, ip);

#if CMK_SHRINK_EXPAND
  server_port = arg_charmrun_port;
#else
  server_port = 0;
#endif
  server_fd = skt_server(&server_port);

  if (arg_verbose) {
    printf("Charmrun> Charmrun = %s, port = %d\n", server_addr, server_port);
  }

#if CMK_CCS_AVAILABLE
#ifdef HSTART
  if (!arg_hierarchical_start ||
      (arg_hierarchical_start && !arg_child_charmrun))
#endif
    if (arg_server == 1)
      CcsServer_new(NULL, &arg_server_port, arg_server_auth);
#endif
}

#ifdef HSTART
int unique_node_start;
/* Function copied from machine.c file */
void parse_netstart(void)
{
  char *ns;
  int nread;
  int port;
  ns = getenv("NETSTART");
  if (ns != 0) { /*Read values set by Charmrun*/
    char parent_charmrun_name[1024 * 1000];
    nread = sscanf(ns, "%d%s%d%d%d", &unique_node_start, parent_charmrun_name,
                   &parent_charmrun_port, &parent_charmrun_pid, &port);
    parent_charmrun_IP = skt_lookup_ip(parent_charmrun_name);
    mynodes_start =
        nodetab_unique_table[unique_node_start]; /*Works only when
                                                    init_hierarchical called in
                                                    child charmrun*/

    if (nread != 5) {
      fprintf(stderr, "Error parsing NETSTART '%s'\n", ns);
      exit(1);
    }
  }
#if CMK_USE_IBVERBS | CMK_USE_IBUD
  char *cmi_num_nodes = getenv("CmiNumNodes");
  if (cmi_num_nodes != NULL) {
    sscanf(cmi_num_nodes, "%d", &_Cmi_numnodes);
  }
#endif
}

int nodetab_rank0_size_total;
/* Receive nodes for which I am responsible*/
void my_nodetab_store(ChMessage *msg)
{
  ChMessageInt_t *nodelistmsg = (ChMessageInt_t *) msg->data;
  nodetab_rank0_size = ChMessageInt(nodelistmsg[0]);
  nodetab_rank0_size_total = ChMessageInt(nodelistmsg[1]);
  int k;
  for (k = 0; k < nodetab_rank0_size; k++) {
    nodetab_rank0_table[k] = ChMessageInt(nodelistmsg[k + 2]);
  }
}

/* In hierarchical startup, this function is used by child charmrun to obtains
 * the list of nodes for which it is responsible */
void nodelist_obtain(void)
{
  ChMessage nodelistmsg; /* info about all nodes*/
                         /*Contact charmrun for machine info.*/

#if CMK_USE_IBVERBS
  {
                /*		int qpListSize = (_Cmi_numnodes-1)*sizeof(ChInfiAddr);
				me.info.qpList = malloc(qpListSize);
				copyInfiAddr(me.info.qpList);
				MACHSTATE1(3,"me.info.qpList created and copied size %d bytes",qpListSize);
				ctrl_sendone_nolock("initnode",(const char *)&me,sizeof(me),(const char *)me.info.qpList,qpListSize);
				free(me.info.qpList);
		*/	}
#else
  ChMessageHeader hdr;
  ChMessageInt_t node_start = ChMessageInt_new(unique_node_start);
  ChMessageHeader_new("initnodetab", sizeof(ChMessageInt_t), &hdr);
  skt_sendN(parent_charmrun_fd, (const char *) &hdr, sizeof(hdr));
  skt_sendN(parent_charmrun_fd, (const char *) &node_start, sizeof(node_start));

#endif // CMK_USE_IBVERBS

  /*We get the other node addresses from a message sent
    back via the charmrun control port.*/
  if (!skt_select1(parent_charmrun_fd, 1200 * 1000)) {
    exit(0);
  }
  ChMessage_recv(parent_charmrun_fd, &nodelistmsg);

  my_nodetab_store(&nodelistmsg);
  ChMessage_free(&nodelistmsg);
}

void init_mynodes(void)
{
  parse_netstart();
  if (!skt_ip_match(parent_charmrun_IP, _skt_invalid_ip)) {
    dataskt = skt_server(&dataport);
    parent_charmrun_fd =
        skt_connect(parent_charmrun_IP, parent_charmrun_port, 1800);
  } else {
    parent_charmrun_fd = -1;
  }

  nodelist_obtain();
}
#endif

/****************************************************************************
 *
 *  The Main Program
 *
 ****************************************************************************/
void start_nodes_daemon(void);
void start_nodes_ssh(void);
void start_nodes_mpiexec();
#ifdef HSTART
void start_next_level_charmruns(void);
#endif
#if CMK_BPROC
void nodetab_init_for_scyld(void);
void start_nodes_scyld(void);
#endif
void start_nodes_local(char **envp);
void kill_nodes(void);
void open_gdb_info(void);
void read_global_segments_size(void);

static void fast_idleFn(void) { sleep(0); }
void finish_nodes(void);

int main(int argc, const char **argv, char **envp)
{
  srand(time(0));
  skt_init();
  skt_set_idle(fast_idleFn);
/* CrnSrand((int) time(0)); */
/* notify charm developers that charm is in use */

#ifdef HSTART
  if (!arg_child_charmrun)
#endif
    ping_developers();
  /* Compute the values of all constants */
  arg_init(argc, argv);
  if (arg_verbose)
    fprintf(stderr, "Charmrun> charmrun started...\n");
  start_timer = GetClock();
#if CMK_BPROC
  /* check scyld configuration */
  if (arg_nodelist)
    nodetab_init();
  else
    nodetab_init_for_scyld();
#else
  /* Initialize the node-table by reading nodesfile */
  nodetab_init();
#endif

  /* Start the server port */
  req_start_server();

  /* Initialize the IO module */
  input_init();

#ifdef HSTART
  /* Hierarchical startup*/
  if (arg_child_charmrun) {
    init_mynodes(); /* contacts root charmrun and gets list  of nodes to start*/
  }
#endif
  /* start the node processes */
  if (0 != getenv("CONV_DAEMON"))
    start_nodes_daemon();
  else
#if CMK_BPROC
    start_nodes_scyld();
#else
#if CMK_USE_IBVERBS
    PRINT(("Charmrun> IBVERBS version of charmrun\n"));
#endif

#ifdef HSTART
  /* Hierarchical-startup*/
  if (arg_hierarchical_start) {
    if (!arg_local) {
      if (!arg_child_charmrun) {
        start_next_level_charmruns();
      } else {
        if (!arg_batch_spawn)
          start_nodes_ssh();
        else
          req_client_start_and_connect();
      }
    } else
      start_nodes_local(envp);
  }

  /* Normal startup*/
  else

#endif
  {
    if (!arg_local) {
      if (!arg_batch_spawn) {
#if CMK_SHRINK_EXPAND
        //  modified rsh in shrink expand, need to launch only new ones,
        //  preserve some info between new and old
        if (!arg_shrinkexpand || (arg_requested_pes > arg_old_pes))
#endif
        {
          if (arg_mpiexec)
            start_nodes_mpiexec();
          else
            start_nodes_ssh();
        }
      } else
        req_client_start_and_connect();
    } else
      start_nodes_local(envp);
  }
#endif

  if (arg_charmdebug) {
#if (defined(_WIN32) && !defined(__CYGWIN__)) || CMK_BPROC
    /* Gdb stream (and charmdebug) currently valid only with ssh subsystem */
    fprintf(stderr,
            "Charmdebug is supported currently only with the ssh subsystem\n");
    abort();
#else
    /* Open an additional connection to node 0 with a gdb to grab info */
    PRINT(("opening connection with node 0 for info gdb\n"));
    read_global_segments_size();
    open_gdb_info();
    gdb_stream = fdopen(dup(2), "a");
    dup2(1, 2);
#endif
  }

  if (arg_verbose)
    fprintf(stderr, "Charmrun> node programs all started\n");

/* Wait for all clients to connect */
#ifdef HSTART
  /* Hierarchical startup*/
  if (arg_hierarchical_start) {
#if !CMK_SSH_KILL
    if (!arg_batch_spawn || (!arg_child_charmrun))
      finish_nodes();
#endif

    if (!arg_child_charmrun)
      req_charmrun_connect();
    else if (!arg_batch_spawn)
      req_client_connect();
  }
  /* Normal startup*/
  else
#endif
  {
#if !CMK_SSH_KILL
    if (!arg_batch_spawn)
      finish_nodes();
#endif
    if (!arg_batch_spawn)
      req_client_connect();
  }
#if CMK_SSH_KILL
  kill_nodes();
#endif
  if (arg_verbose)
    fprintf(stderr, "Charmrun> node programs all connected\n");
  /* report time */
  PRINT(("Charmrun> started all node programs in %.3f seconds.\n",
          GetClock() - start_timer));

/* enter request-service mode */
#ifdef HSTART
  if (arg_hierarchical_start)
    while (1)
      req_poll_hierarchical();
  else
#endif
    while (1)
      req_poll();
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
  if (arg_mpiexec)
    sprintf(dest, "$CmiMyNode %s %d %d %d", server_addr, server_port,
            getpid() & 0x7FFF, port);
  else
    sprintf(dest, "%d %s %d %d %d", node, server_addr, server_port,
            getpid() & 0x7FFF, port);
  return dest;
}

#if CMK_SHRINK_EXPAND
/*This little snippet creates a OLDNODENAMES
environment variable entry*/
char *create_oldnodenames()
{
  static char dest1[1024 * 1000];
  int i;
  for (i = 0; i < nodetab_size; i++)
    sprintf(dest1, "%s %s", dest1, (*nodetab_table[i]).name);
  printf("Charmrun> Created oldnames %s \n", dest1);
  return dest1;
}
#endif
/* The remainder of charmrun is only concerned with starting all
the node-programs, also known as charmrun clients.  We have to
start nodetab_rank0_size processes on the remote machines.
*/

/*Ask the converse daemon running on each machine to start the node-programs.*/
void start_nodes_daemon(void)
{
  taskStruct task;
  char argBuffer[5000]; /*Buffer to hold assembled program arguments*/
  int i, nodeNumber;

  /*Set the parts of the task structure that will be the same for all nodes*/
  /*Figure out the command line arguments (same for all PEs)*/
  argBuffer[0] = 0;
  for (i = 0; arg_argv[i]; i++) {
    if (arg_verbose)
      printf("Charmrun> packing arg: %s\n", arg_argv[i]);
    strcat(argBuffer, " ");
    strcat(argBuffer, arg_argv[i]);
  }

  task.magic = ChMessageInt_new(DAEMON_MAGIC);

  /*Start up the user program, by sending a message
    to PE 0 on each node.*/
  for (nodeNumber = 0; nodeNumber < nodetab_rank0_size; nodeNumber++) {
    char nodeArgBuffer[5000]; /*Buffer to hold assembled program arguments*/
    char *argBuf;
    char *arg_nodeprog_r, *arg_currdir_r;
    char statusCode = 'N'; /*Default error code-- network problem*/
    int fd;
    int pe0 = nodetab_rank0_table[nodeNumber];

    arg_currdir_r = pathfix(arg_currdir_a, nodetab_pathfixes(nodeNumber));
    strcpy(task.cwd, arg_currdir_r);
    free(arg_currdir_r);
    arg_nodeprog_r = pathextfix(arg_nodeprog_a, nodetab_pathfixes(nodeNumber),
                                nodetab_ext(nodeNumber));
    strcpy(task.pgm, arg_nodeprog_r);

    if (arg_verbose)
      printf("Charmrun> Starting node program %d on '%s' as %s.\n", nodeNumber,
             nodetab_name(pe0), arg_nodeprog_r);
    free(arg_nodeprog_r);
    sprintf(task.env, "NETSTART=%s", create_netstart(nodeNumber));

    if (nodetab_nice(nodeNumber) != -100) {
      if (arg_verbose)
        fprintf(stderr, "Charmrun> +nice %d\n", nodetab_nice(nodeNumber));
      sprintf(nodeArgBuffer, "%s +nice %d", argBuffer,
              nodetab_nice(nodeNumber));
      argBuf = nodeArgBuffer;
    } else
      argBuf = argBuffer;
    task.argLength = ChMessageInt_new(strlen(argBuf));

    /*Send request out to remote node*/
    fd = skt_connect(nodetab_ip(pe0), DAEMON_IP_PORT, 30);
    if (fd !=
        INVALID_SOCKET) { /*Contact!  Ask the daemon to start the program*/
      skt_sendN(fd, (const char *) &task, sizeof(task));
      skt_sendN(fd, (const char *) argBuf, strlen(argBuf));
      skt_recvN(fd, &statusCode, sizeof(char));
    }
    if (statusCode != 'G') { /*Something went wrong--*/
      fprintf(stderr, "Error '%c' starting remote node program on %s--\n%s\n",
              statusCode, nodetab_name(pe0), daemon_status2msg(statusCode));
      exit(1);
    } else if (arg_verbose)
      printf("Charmrun> Node program %d started.\n", nodeNumber);
  }
}

#if defined(_WIN32) && !defined(__CYGWIN__)
/*Sadly, interprocess communication on Win32 is quite
  different, so we can't use Ssh on win32 yet.
  Fall back to the daemon.*/
void start_nodes_ssh() { start_nodes_daemon(); }
void finish_nodes(void) {}
void start_one_node_ssh(int rank0no) {}
void finish_one_node(int rank0no) {}
void start_nodes_mpiexec() {}

int start_set_node_ssh(int client) { return 0; }
void finish_set_nodes(int start, int stop) {}

void envCat(char *dest, LPTSTR oldEnv)
{
  char *src = oldEnv;
  dest += strlen(dest); // Advance to end of dest
  dest++;               // Advance past terminating NULL character
  while ((*src) != '\0') {
    int adv = strlen(src) + 1; // Length of newly-copied string plus NULL
    strcpy(dest, src);         // Copy another environment string
    dest += adv;               // Advance past newly-copied string and NULL
    src += adv;                // Ditto for src
  }
  *dest = '\0'; // Paste on final terminating NULL character
  FreeEnvironmentStrings(oldEnv);
}

/* simple version of charmrun that avoids the sshd or charmd,   */
/* it spawn the node program just on local machine using exec. */
void start_nodes_local(char **env)
{
  int ret, i;
  PROCESS_INFORMATION pi; /* process Information for the process spawned */
  const char **p;

  char environment[10000]; /*Doubly-null terminated environment strings*/
  char cmdLine[10000];     /*Program command line, including executable name*/
                           /*Command line too long.*/
                           /*
                             if (strlen(pparam_argv[1])+strlen(args) > 10000)
                                   return 0;
                           */
  strcpy(cmdLine, pparam_argv[1]);
  p = pparam_argv + 2;
  while ((*p) != '\0') {
    strcat(cmdLine, " ");
    strcat(cmdLine, *p);
    p++;
  }

  for (i = 0; i < arg_requested_pes; i++) {
    STARTUPINFO si = {0}; /* startup info for the process spawned */

    sprintf(environment, "NETSTART=%s", create_netstart(i));
    /*Paste all system environment strings */
    envCat(environment, GetEnvironmentStrings());

    /* Initialise the security attributes for the process
     to be spawned */
    si.cb = sizeof(si);
    if (arg_verbose)
      printf("Charmrun> start %d node program on localhost.\n", i);

    ret = CreateProcess(NULL,          /* application name */
                        cmdLine,       /* command line */
                        NULL, /*&sa,*/ /* process SA */
                        NULL, /*&sa,*/ /* thread SA */
                        FALSE,         /* inherit flag */
#if 1
                        CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
#else
                        CREATE_NEW_PROCESS_GROUP | CREATE_NEW_CONSOLE,
#endif
                        /* creation flags */
                        environment, /* environment block */
                        ".",         /* working directory */
                        &si,         /* startup info */
                        &pi);

    if (ret == 0) {
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
      int error = GetLastError();
      fprintf(stderr, "startProcess failed to start process \"%s\" with status: %d\n",
             pparam_argv[1], error);
      exit(1);
    }
  }
}

#elif CMK_BPROC

int bproc_nodeisup(int node)
{
  int status = 0;
#if CMK_BPROC_VERSION < 4
  if (bproc_nodestatus(node) == bproc_node_up)
    status = 1;
  if (arg_verbose)
    printf("Charmrun> node %d status: %s\n", node, status ? "up" : "down");
#else
  char nodestatus[128];
  if (node == -1) { /* master node is always up */
    strcpy(nodestatus, "up");
    status = 1;
  }
  if (bproc_nodestatus(node, nodestatus, 128)) {
    if (strcmp(nodestatus, "up") == 0)
      status = 1;
  }
  if (arg_verbose)
    printf("Charmrun> node %d status: %s\n", node, nodestatus);
#endif
  return status;
}

/* ++ppn now is supported in both SMP and non SMP version
   in SMP, ++ppn specifies number of threads on each node;
   in non-SMP, ++ppn specifies number of processes on each node. */
void nodetab_init_for_scyld()
{
  int maxNodes, i, node, npes, rank;
  nodetab_host group;
  int tablesize;

  tablesize = arg_requested_pes;
  maxNodes = bproc_numnodes() + 1;
  if (arg_endpe < maxNodes)
    maxNodes = arg_endpe + 1;
  if (maxNodes > tablesize)
    tablesize = maxNodes;
  nodetab_table = (nodetab_host **) malloc(tablesize * sizeof(nodetab_host *));
  nodetab_rank0_table = (int *) malloc(tablesize * sizeof(int));
  nodetab_max = tablesize;

  nodetab_reset(&group);

  if (arg_ppn == 0)
    arg_ppn = 1;
  /*
#if CMK_SHARED_VARS_UNAVAILABLE
    if (arg_ppn > 1) {
          fprintf(stderr,"Warning> Invalid ppn %d in nodelist ignored.\n",
arg_ppn);
          arg_ppn=1;
    }
#endif
  */
  group.cpus = 1;
  group.rank = 0;

  /* check which slave node is available from frompe to endpe */
  npes = 0;
  for (i = -1; i < maxNodes && npes < arg_requested_pes; i++) {
    char hostname[256];
    if (!bproc_nodeisup(i))
      continue;
    if (i != -1 && i < arg_startpe)
      continue;
    if (i == -1 && arg_skipmaster)
      continue; /* skip master node -1 */
    sprintf(hostname, "%d", i);
#if !CMK_SHARED_VARS_UNAVAILABLE
    if (npes + arg_ppn > arg_requested_pes)
      group.cpus = arg_requested_pes - npes;
    else
      group.cpus = arg_ppn;
#endif
    for (rank = 0; rank < arg_ppn; rank++) {
#if !CMK_SHARED_VARS_UNAVAILABLE
      group.rank = rank;
#endif
      nodetab_makehost(hostname, &group);
      nodetab_add(&group);
      if (++npes == arg_requested_pes)
        break;
    }
  }
  if (nodetab_rank0_size == 0) {
    fprintf(stderr, "Charmrun> no slave node available!\n");
    exit(1);
  }
  if (arg_verbose)
    printf("Charmrun> There are %d slave nodes available.\n",
           nodetab_rank0_size - (arg_skipmaster ? 0 : 1));

  /* expand node table to arg_requested_pes */
  if (arg_requested_pes > npes) {
    int orig_size = npes;
    int node;
    int startnode = 0;
    if (arg_singlemaster && nodetab_rank0_size > 1 && !arg_skipmaster)
      startnode = arg_ppn; /* skip -1 */
    node = startnode;
    while (npes < arg_requested_pes) {
#if !CMK_SHARED_VARS_UNAVAILABLE
      if (npes + arg_ppn > arg_requested_pes)
        group.cpus = arg_requested_pes - npes;
      else
        group.cpus = arg_ppn;
#endif
      for (rank = 0; rank < arg_ppn; rank++) {
#if !CMK_SHARED_VARS_UNAVAILABLE
        group.rank = rank;
#endif
        nodetab_makehost(nodetab_name(node), &group);
        nodetab_add(&group);
        if (++node == orig_size)
          node = startnode;
        if (++npes == arg_requested_pes)
          break;
      }
    }
  }
}

void start_nodes_scyld(void)
{
  char *envp[2];
  int i;

  envp[0] = (char *) malloc(256);
  envp[1] = 0;
  for (i = 0; i < nodetab_rank0_size; i++) {
    int status = 0;
    int pid;
    int pe = nodetab_rank0_table[i];
    int nodeno = atoi(nodetab_name(pe));

    if (arg_verbose)
      printf("Charmrun> start node program on slave node: %d.\n", nodeno);
    sprintf(envp[0], "NETSTART=%s", create_netstart(i));
    pid = 0;
    pid = fork();
    if (pid < 0)
      exit(1);
    if (pid == 0) {
      int fd, fd1 = dup(1);
      if (!(arg_debug || arg_debug_no_pause)) { /* debug mode */
        if (fd = open("/dev/null", O_RDWR)) {
          dup2(fd, 0);
          dup2(fd, 1);
          dup2(fd, 2);
        }
      }
      if (nodeno == -1) {
        status = execve(pparam_argv[1], pparam_argv + 1, envp);
        dup2(fd1, 1);
        fprintf(stderr, "execve failed to start process \"%s\" with status: %d\n",
               pparam_argv[1], status);
      } else {
        status = bproc_execmove(nodeno, pparam_argv[1], pparam_argv + 1, envp);
        dup2(fd1, 1);
        fprintf(stderr, "bproc_execmove failed to start remote process \"%s\" with "
               "status: %d\n",
               pparam_argv[1], status);
      }
      kill(getppid(), 9);
      exit(1);
    }
  }
  free(envp[0]);
}
void finish_nodes(void) {}

#else
/*Unix systems can use Ssh normally*/
/********** SSH-ONLY CODE *****************************************/
/*                                                                          */
/* Ssh_etc                                                                  */
/*                                                                          */
/* this starts all the node programs.  It executes fully in the background. */
/*                                                                          */
/****************************************************************************/
#include <sys/wait.h>

extern char **environ;
void removeEnv(const char *doomedEnv)
{ /*Remove a value from the environment list*/
  char **oe, **ie;
  oe = ie = environ;
  while (*ie != NULL) {
    if (0 != strncmp(*ie, doomedEnv, strlen(doomedEnv)))
      *oe++ = *ie;
    ie++;
  }
  *oe = NULL; /*NULL-terminate list*/
}

int ssh_fork(int nodeno, const char *startScript)
{
  std::vector<const char *> sshargv;
  int pid;
  const char *s, *e;

  s = nodetab_shell(nodeno);
  e = skipstuff(s);
  while (*s) {
    sshargv.push_back(substr(s, e));
    s = skipblanks(e);
    e = skipstuff(s);
  }

  sshargv.push_back(nodetab_name(nodeno));
  sshargv.push_back("-l");
  sshargv.push_back(nodetab_login(nodeno));
  sshargv.push_back("-o");
  sshargv.push_back("KbdInteractiveAuthentication=no");
  sshargv.push_back("-o");
  sshargv.push_back("PasswordAuthentication=no");
  sshargv.push_back("-o");
  sshargv.push_back("NoHostAuthenticationForLocalhost=yes");
  sshargv.push_back("/bin/bash -f");
  sshargv.push_back((const char *) NULL);

  if (arg_verbose) {
    std::string cmd_str = sshargv[0];
    for (int n = 1; n < sshargv.size()-1; ++n)
      cmd_str += " " + std::string(sshargv[n]);
    printf("Charmrun> Starting %s\n", cmd_str.c_str());
  }

  pid = fork();
  if (pid < 0) {
    perror("ERROR> starting remote shell");
    exit(1);
  }
  if (pid == 0) { /*Child process*/
    int fdScript = open(startScript, O_RDONLY);
    /**/ unlink(startScript); /**/
    dup2(fdScript, 0);        /*Open script as standard input*/
    // removeEnv("DISPLAY="); /*No DISPLAY disables ssh's slow X11 forwarding*/
    for (int i = 3; i < 1024; i++)
      close(i);
    execvp(sshargv[0], const_cast<char **>(&sshargv[0]));
    fprintf(stderr, "Charmrun> Couldn't find remote shell program '%s'!\n",
            sshargv[0]);
    exit(1);
  }
  if (arg_verbose)
    fprintf(stderr, "Charmrun> remote shell (%s:%d) started\n",
            nodetab_name(nodeno), nodeno);
  return pid;
}

void fprint_arg(FILE *f, const char **argv)
{
  while (*argv) {
    fprintf(f, " %s", *argv);
    argv++;
  }
}
void ssh_Find(FILE *f, const char *program, const char *dest)
{
  fprintf(f, "Find %s\n", program);
  fprintf(f, "%s=$loc\n", dest);
}
void ssh_script(FILE *f, int nodeno, int rank0no, const char **argv,
                int restart)
{
  char *netstart;
  char *arg_nodeprog_r, *arg_currdir_r;
  const char *dbg = nodetab_debugger(nodeno);
  const char *host = nodetab_name(nodeno);

  if (arg_mpiexec)
    fprintf(f, "#!/bin/sh\n");

  fprintf(f, /*Echo: prints out status message*/
          "Echo() {\n"
          "  echo 'Charmrun remote shell(%s.%d)>' $*\n"
          "}\n",
          host, nodeno);
  fprintf(f, /*Exit: exits with return code*/
          "Exit() {\n"
          "  if [ $1 -ne 0 ]\n"
          "  then\n"
          "    Echo Exiting with error code $1\n"
          "  fi\n"
#if CMK_SSH_KILL /*End by killing ourselves*/
          "  sleep 5\n" /*Delay until any error messages are flushed*/
          "  kill -9 $$\n"
#else            /*Exit normally*/
          "  exit $1\n"
#endif
          "}\n");
  fprintf(f, /*Find: locates a binary program in PATH, sets loc*/
          "Find() {\n"
          "  loc=''\n"
          "  for dir in `echo $PATH | sed -e 's/:/ /g'`\n"
          "  do\n"
          "    test -f \"$dir/$1\" && loc=\"$dir/$1\"\n"
          "  done\n"
          "  if [ \"x$loc\" = x ]\n"
          "  then\n"
          "    Echo $1 not found in your PATH \"($PATH)\"--\n"
          "    Echo set your path in your ~/.charmrunrc\n"
          "    Exit 1\n"
          "  fi\n"
          "}\n");

  if (arg_verbose)
    fprintf(f, "Echo 'remote responding...'\n");

  fprintf(f, "test -f \"$HOME/.charmrunrc\" && . \"$HOME/.charmrunrc\"\n");
  /* let's leave DISPLAY untouched and rely on X11 forwarding,
     changing DISPLAY to charmrun does not always work if X11 forwarding
     presents
  */
  if (arg_display && !arg_ssh_display)
    fprintf(f, "DISPLAY='%s';export DISPLAY\n", arg_display);

#ifdef HSTART
  if (arg_child_charmrun)
    fprintf(f, "NETMAGIC=\"%d\";export NETMAGIC\n",
            parent_charmrun_pid & 0x7FFF);
  else
#endif
    fprintf(f, "NETMAGIC=\"%d\";export NETMAGIC\n", getpid() & 0x7FFF);

  if (arg_mpiexec) {
    fprintf(f, "CmiMyNode=$OMPI_COMM_WORLD_RANK\n");
    fprintf(f, "test -z \"$CmiMyNode\" && CmiMyNode=$MPIRUN_RANK\n");
    fprintf(f, "test -z \"$CmiMyNode\" && CmiMyNode=$PMI_RANK\n");
    fprintf(f, "test -z \"$CmiMyNode\" && CmiMyNode=$PMI_ID\n");
    fprintf(f, "test -z \"$CmiMyNode\" && CmiMyNode=$MP_CHILD\n");
    fprintf(f, "test -z \"$CmiMyNode\" && CmiMyNode=$SLURM_PROCID\n");
    fprintf(f, "test -z \"$CmiMyNode\" && (Echo Could not detect rank from "
               "environment ; Exit 1)\n");
    fprintf(f, "export CmiMyNode\n");
  }
#ifdef HSTART
  else if (arg_hierarchical_start && arg_child_charmrun)
    fprintf(f, "CmiMyNode='%d'; export CmiMyNode\n", mynodes_start + rank0no);
#endif
  else
    fprintf(f, "CmiMyNode='%d'; export CmiMyNode\n", rank0no);

#ifdef HSTART
  if (arg_hierarchical_start && arg_child_charmrun)
    netstart = create_netstart(mynodes_start + rank0no);
  else
#endif
    netstart = create_netstart(rank0no);
  fprintf(f, "NETSTART=\"%s\";export NETSTART\n", netstart);

  fprintf(f, "CmiMyNodeSize='%d'; export CmiMyNodeSize\n",
          nodetab_getnodeinfo(rank0no)->cpus);

  if (restart || arg_mpiexec) /* skip fork */
    fprintf(f, "CmiMyForks='%d'; export CmiMyForks\n", 0);
  else
    fprintf(f, "CmiMyForks='%d'; export CmiMyForks\n",
            nodetab_getnodeinfo(rank0no)->forks);

  if (arg_mpiexec) {
    fprintf(f, "CmiNumNodes=$OMPI_COMM_WORLD_SIZE\n");
    fprintf(f, "test -z \"$CmiNumNodes\" && CmiNumNodes=$MPIRUN_NPROCS\n");
    fprintf(f, "test -z \"$CmiNumNodes\" && CmiNumNodes=$PMI_SIZE\n");
    fprintf(f, "test -z \"$CmiNumNodes\" && CmiNumNodes=$MP_PROCS\n");
    fprintf(f, "test -z \"$CmiNumNodes\" && CmiNumNodes=$SLURM_NTASKS\n");
    fprintf(f, "test -z \"$CmiNumNodes\" && CmiNumNodes=$SLURM_NPROCS\n");
    fprintf(f, "test -z \"$CmiNumNodes\" && (Echo Could not detect node count "
               "from environment ; Exit 1)\n");
    fprintf(f, "export CmiNumNodes\n");
  }
#ifdef HSTART
  else if (arg_hierarchical_start && arg_child_charmrun)
    fprintf(f, "CmiNumNodes='%d'; export CmiNumNodes\n",
            nodetab_rank0_size_total);
#endif

  else
    fprintf(f, "CmiNumNodes='%d'; export CmiNumNodes\n", nodetab_rank0_size);

#ifdef CMK_G95
  fprintf(f, "G95_UNBUFFERED_ALL=TRUE; export G95_UNBUFFERED_ALL\n");
#endif
#ifdef CMK_GFORTRAN
  fprintf(f, "GFORTRAN_UNBUFFERED_ALL=YES; export GFORTRAN_UNBUFFERED_ALL\n");
#endif
#if CMK_USE_MX
  fprintf(f, "MX_MONOTHREAD=1; export MX_MONOTHREAD\n");
/*fprintf(f,"MX_RCACHE=1; export MX_RCACHE\n");*/
#endif
#if CMK_AIX && CMK_SMP
  fprintf(f, "MALLOCMULTIHEAP=1; export MALLOCMULTIHEAP\n");
#endif

  if (arg_verbose) {
    printf("Charmrun> Sending \"%s\" to client %d.\n", netstart, rank0no);
  }
  fprintf(f,
          "PATH=\"$PATH:/bin:/usr/bin:/usr/X/bin:/usr/X11/bin:/usr/local/bin:"
          "/usr/X11R6/bin:/usr/openwin/bin\"\n");

  /* find the node-program */
  arg_nodeprog_r = pathextfix(arg_nodeprog_a, nodetab_pathfixes(nodeno),
                              nodetab_ext(nodeno));

  /* find the current directory, relative version */
  arg_currdir_r = pathfix(arg_currdir_a, nodetab_pathfixes(nodeno));

  if (arg_verbose) {
    printf("Charmrun> find the node program \"%s\" at \"%s\" for %d.\n",
           arg_nodeprog_r, arg_currdir_r, nodeno);
  }
  if (arg_debug || arg_debug_no_pause || arg_in_xterm) {
    ssh_Find(f, nodetab_xterm(nodeno), "F_XTERM");
    if (!arg_ssh_display && !arg_debug_no_xrdb)
      ssh_Find(f, "xrdb", "F_XRDB");
    if (arg_verbose)
      fprintf(f, "Echo 'using xterm' $F_XTERM\n");
  }

  if (arg_debug || arg_debug_no_pause) { /*Look through PATH for debugger*/
    ssh_Find(f, dbg, "F_DBG");
    if (arg_verbose)
      fprintf(f, "Echo 'using debugger' $F_DBG\n");
  }

  if (!arg_ssh_display && !arg_debug_no_xrdb &&
      (arg_debug || arg_debug_no_pause || arg_in_xterm)) {
    /*    if (arg_debug || arg_debug_no_pause || arg_in_xterm) {*/
    fprintf(f, "$F_XRDB -query > /dev/null\n");
    fprintf(f, "if test $? != 0\nthen\n");
    fprintf(f, "  Echo 'Cannot contact X Server '$DISPLAY'.  You probably'\n");
    fprintf(f, "  Echo 'need to run xhost to authorize connections.'\n");
    fprintf(f, "  Echo '(See manual for xhost for security issues)'\n");
    fprintf(f, "  Echo 'Or try ++batch 1 ++ssh-display to rely on SSH X11 "
               "forwarding'\n");
    fprintf(f, "  Exit 1\n");
    fprintf(f, "fi\n");
  }

  fprintf(f, "if test ! -x \"%s\"\nthen\n", arg_nodeprog_r);
  fprintf(f, "  Echo 'Cannot locate this node-program: %s'\n", arg_nodeprog_r);
  fprintf(f, "  Exit 1\n");
  fprintf(f, "fi\n");

  fprintf(f, "cd \"%s\"\n", arg_currdir_r);
  fprintf(f, "if test $? = 1\nthen\n");
  fprintf(f, "  Echo 'Cannot propagate this current directory:'\n");
  fprintf(f, "  Echo '%s'\n", arg_currdir_r);
  fprintf(f, "  Exit 1\n");
  fprintf(f, "fi\n");

  if (strcmp(nodetab_setup(nodeno), "*")) {
    fprintf(f, "%s\n", nodetab_setup(nodeno));
    fprintf(f, "if test $? = 1\nthen\n");
    fprintf(f, "  Echo 'this initialization command failed:'\n");
    fprintf(f, "  Echo '\"%s\"'\n", nodetab_setup(nodeno));
    fprintf(f, "  Echo 'edit your nodes file to fix it.'\n");
    fprintf(f, "  Exit 1\n");
    fprintf(f, "fi\n");
  }

  fprintf(f, "rm -f /tmp/charmrun_err.$$\n");
  if (arg_verbose)
    fprintf(f, "Echo 'starting node-program...'\n");
  /* This is the start of the the run-nodeprogram script */
  fprintf(f, "(");

  if (arg_debug || arg_debug_no_pause) {
    if (strcmp(dbg, "gdb") == 0 || strcmp(dbg, "idb") == 0) {
      fprintf(f, "cat > /tmp/charmrun_gdb.$$ << END_OF_SCRIPT\n");
      if (strcmp(dbg, "idb") == 0) {
        fprintf(f, "set \\$cmdset=\"gdb\"\n");
      }
      fprintf(f, "shell /bin/rm -f /tmp/charmrun_gdb.$$\n");
      fprintf(f, "handle SIGPIPE nostop noprint\n");
      fprintf(f, "handle SIGWINCH nostop noprint\n");
      fprintf(f, "handle SIGWAITING nostop noprint\n");
      if (arg_debug_commands)
        fprintf(f, "%s\n", arg_debug_commands);
      fprintf(f, "set args");
      fprint_arg(f, argv);
      fprintf(f, "\n");
      if (arg_debug_no_pause)
        fprintf(f, "run\n");
      fprintf(f, "END_OF_SCRIPT\n");
      if (arg_runscript)
        fprintf(f, "\"%s\" ", arg_runscript);
      fprintf(f, "$F_XTERM");
      fprintf(f, " -title 'Node %d (%s)' ", nodeno, nodetab_name(nodeno));
      if (strcmp(dbg, "idb") == 0)
        fprintf(f, " -e $F_DBG \"%s\" -c /tmp/charmrun_gdb.$$ \n", arg_nodeprog_r);
      else
        fprintf(f, " -e $F_DBG \"%s\" -x /tmp/charmrun_gdb.$$ \n", arg_nodeprog_r);
    } else if (strcmp(dbg, "dbx") == 0) {
      fprintf(f, "cat > /tmp/charmrun_dbx.$$ << END_OF_SCRIPT\n");
      fprintf(f, "sh /bin/rm -f /tmp/charmrun_dbx.$$\n");
      fprintf(f, "dbxenv suppress_startup_message 5.0\n");
      fprintf(f, "ignore SIGPOLL\n");
      fprintf(f, "ignore SIGPIPE\n");
      fprintf(f, "ignore SIGWINCH\n");
      fprintf(f, "ignore SIGWAITING\n");
      if (arg_debug_commands)
        fprintf(f, "%s\n", arg_debug_commands);
      fprintf(f, "END_OF_SCRIPT\n");
      if (arg_runscript)
        fprintf(f, "\"%s\" ", arg_runscript);
      fprintf(f, "$F_XTERM");
      fprintf(f, " -title 'Node %d (%s)' ", nodeno, nodetab_name(nodeno));
      fprintf(f, " -e $F_DBG %s ", arg_debug_no_pause ? "-r" : "");
      if (arg_debug) {
        fprintf(f, "-c \'runargs ");
        fprint_arg(f, argv);
        fprintf(f, "\' ");
      }
      fprintf(f, "-s/tmp/charmrun_dbx.$$ %s", arg_nodeprog_r);
      if (arg_debug_no_pause)
        fprint_arg(f, argv);
      fprintf(f, "\n");
    } else {
      fprintf(stderr, "Unknown debugger: %s.\n Exiting.\n",
              nodetab_debugger(nodeno));
    }
  } else if (arg_in_xterm) {
    if (arg_verbose)
      fprintf(stderr, "Charmrun> node %d: xterm is %s\n", nodeno,
              nodetab_xterm(nodeno));
    fprintf(f, "cat > /tmp/charmrun_inx.$$ << END_OF_SCRIPT\n");
    fprintf(f, "#!/bin/sh\n");
    fprintf(f, "/bin/rm -f /tmp/charmrun_inx.$$\n");
    fprintf(f, "%s", arg_nodeprog_r);
    fprint_arg(f, argv);
    fprintf(f, "\n");
    fprintf(f, "echo 'program exited with code '\\$?\n");
    fprintf(f, "read eoln\n");
    fprintf(f, "END_OF_SCRIPT\n");
    fprintf(f, "chmod 700 /tmp/charmrun_inx.$$\n");
    if (arg_runscript)
      fprintf(f, "\"%s\" ", arg_runscript);
    fprintf(f, "$F_XTERM -title 'Node %d (%s)' ", nodeno, nodetab_name(nodeno));
    fprintf(f, " -sl 5000");
    fprintf(f, " -e /tmp/charmrun_inx.$$\n");
  } else {
    if (arg_runscript)
      fprintf(f, "\"%s\" ", arg_runscript);
    if (arg_no_va_rand) {
      if (arg_verbose)
        fprintf(stderr, "Charmrun> setarch -R is used.\n");
      fprintf(f, "setarch `uname -m` -R ");
    }
    fprintf(f, "\"%s\" ", arg_nodeprog_r);
    fprint_arg(f, argv);
    if (nodetab_nice(nodeno) != -100) {
      if (arg_verbose)
        fprintf(stderr, "Charmrun> nice -n %d\n", nodetab_nice(nodeno));
      fprintf(f, " +nice %d ", nodetab_nice(nodeno));
    }
    fprintf(f, "\nres=$?\n");
    /* If shared libraries fail to load, the program dies without
       calling charmrun back.  Since we *have* to close down stdin/out/err,
       we have to smuggle this failure information out via a file,
       /tmp/charmrun_err.<pid> */
    fprintf(f, "if [ $res -eq 127 ]\n"
               "then\n"
               "  ( \n" /* Re-run, spitting out errors from a subshell: */
               "    \"%s\" \n"
               "    ldd \"%s\"\n"
               "  ) > /tmp/charmrun_err.$$ 2>&1 \n"
               "fi\n",
            arg_nodeprog_r, arg_nodeprog_r);
  }

  /* End the node-program subshell. To minimize the number
     of open ports on the front-end, we must close down ssh;
     to do this, we have to close stdin, stdout, stderr, and
     run the subshell in the background. */
  fprintf(f, ")");
  fprintf(f, " < /dev/null 1> /dev/null 2> /dev/null");
  if (!arg_mpiexec)
    fprintf(f, " &");
  fprintf(f, "\n");

  if (arg_verbose)
    fprintf(f, "Echo 'remote shell phase successful.'\n");
  fprintf(f, /* Check for startup errors: */
          "sleep 1\n"
          "if [ -r /tmp/charmrun_err.$$ ]\n"
          "then\n"
          "  cat /tmp/charmrun_err.$$ \n"
          "  rm -f /tmp/charmrun_err.$$ \n"
          "  Exit 1\n"
          "fi\n");
  fprintf(f, "Exit 0\n");
  free(arg_currdir_r);
}

/* use the command "size" to get information about the position of the ".data"
   and ".bss" segments inside the program memory */
void read_global_segments_size()
{
  std::vector<const char *> sshargv;
  char *tmp;
  int childPid;

  /* find the node-program */
  arg_nodeprog_r =
      pathextfix(arg_nodeprog_a, nodetab_pathfixes(0), nodetab_ext(0));

  sshargv.push_back(nodetab_shell(0));
  sshargv.push_back(nodetab_name(0));
  sshargv.push_back("-l");
  sshargv.push_back(nodetab_login(0));
  tmp = (char *) malloc(sizeof(char) * 9 + strlen(arg_nodeprog_r));
  sprintf(tmp, "size -A %s", arg_nodeprog_r);
  sshargv.push_back(tmp);
  sshargv.push_back((const char *) NULL);

  childPid = fork();
  if (childPid < 0) {
    perror("ERROR> getting the size of the global variables segments");
    exit(1);
  } else if (childPid == 0) {
    /* child process */
    dup2(2, 1);
    /*printf("executing: \"%s\" \"%s\" \"%s\" \"%s\"
     * \"%s\"\n",sshargv[0],sshargv[1],sshargv[2],sshargv[3],sshargv[4]);*/
    execvp(sshargv[0], const_cast<char **>(&sshargv[0]));
    fprintf(stderr, "Charmrun> Couldn't find remote shell program '%s'!\n",
            sshargv[0]);
    exit(1);
  } else {
    /* else we are in the parent */
    free(tmp);
    waitpid(childPid, NULL, 0);
  }
}

/* open a ssh connection with processor 0 and open a gdb session for info */
void open_gdb_info()
{
  std::vector<const char *> sshargv;
  char *tmp;
  int fdin[2];
  int fdout[2];
  int fderr[2];
  int i;

  /* find the node-program */
  arg_nodeprog_r =
      pathextfix(arg_nodeprog_a, nodetab_pathfixes(0), nodetab_ext(0));

  sshargv.push_back(nodetab_shell(0));
  sshargv.push_back(nodetab_name(0));
  sshargv.push_back("-l");
  sshargv.push_back(nodetab_login(0));
  tmp = (char *) malloc(sizeof(char) * 8 + strlen(arg_nodeprog_r));
  sprintf(tmp, "gdb -q %s", arg_nodeprog_r);
  sshargv.push_back(tmp);
  sshargv.push_back((const char *) NULL);

  pipe(fdin);
  pipe(fdout);
  pipe(fderr);

  gdb_info_pid = fork();
  if (gdb_info_pid < 0) {
    perror("ERROR> starting info gdb");
    exit(1);
  } else if (gdb_info_pid == 0) {
    /* child process */
    close(fdin[1]);
    close(fdout[0]);
    close(fderr[0]);
    PRINT(("executing: \"%s\" \"%s\" \"%s\" \"%s\" \"%s\"\n", sshargv[0],
           sshargv[1], sshargv[2], sshargv[3], sshargv[4]));
    dup2(fdin[0], 0);
    dup2(fdout[1], 1);
    dup2(fderr[1], 2);
    for (i = 3; i < 1024; i++)
      close(i);
    execvp(sshargv[0], const_cast<char **>(&sshargv[0]));
    fprintf(stderr, "Charmrun> Couldn't find remote shell program '%s'!\n",
            sshargv[0]);
    exit(1);
  }
  /* else we are in the parent */
  free(tmp);
  gdb_info_std[0] = fdin[1];
  gdb_info_std[1] = fdout[0];
  gdb_info_std[2] = fderr[0];
  close(fdin[0]);
  close(fdout[1]);
  close(fderr[1]);
}
#ifdef HSTART
void start_next_level_charmruns()
{

  static char buf[1024];
  char *nodeprog_name = strrchr(arg_nodeprog_a, '/');
  nodeprog_name[0] = 0;
  sprintf(buf, "%s%s%s", arg_nodeprog_a, DIRSEP, "charmrun");
  arg_nodeprog_a = strdup(buf);

  int client;
  int nextIndex = 0;
  client = 0;
  while (nextIndex < branchfactor) {
    /* need to index into unique_table*/
    int rank0no = nodetab_unique_table[client];
    int pe = nodetab_rank0_table[rank0no];
    FILE *f;
    char startScript[200];
    sprintf(startScript, "/tmp/charmrun.%d.%d", getpid(), pe);
    f = fopen(startScript, "w");
    if (f == NULL) {
      /* now try current directory */
      sprintf(startScript, "charmrun.%d.%d", getpid(), pe);
      f = fopen(startScript, "w");
      if (f == NULL) {
        fprintf(stderr, "Charmrun> Can not write file %s!\n", startScript);
        exit(1);
      }
    }
    ssh_script(f, pe, client, arg_argv, 0);
    fclose(f);
    if (!ssh_pids)
      ssh_pids = (int *) malloc(sizeof(int) * branchfactor);
    ssh_pids[nextIndex++] = ssh_fork(pe, startScript);
    client += nodes_per_child;
  }
}
#endif

/* returns pid */
void start_one_node_ssh(int rank0no)
{
  int pe = nodetab_rank0_table[rank0no];
  FILE *f;
  char startScript[200];
  sprintf(startScript, "/tmp/charmrun.%d.%d", getpid(), pe);
  f = fopen(startScript, "w");
  if (f == NULL) {
    /* now try current directory */
    sprintf(startScript, "charmrun.%d.%d", getpid(), pe);
    f = fopen(startScript, "w");
    if (f == NULL) {
      fprintf(stderr, "Charmrun> Can not write file %s!\n", startScript);
      exit(1);
    }
  }
  ssh_script(f, pe, rank0no, arg_argv, 0);
  fclose(f);
  if (!ssh_pids)
    ssh_pids = (int *) malloc(sizeof(int) * nodetab_rank0_size);
  ssh_pids[rank0no] = ssh_fork(pe, startScript);
}

int start_set_node_ssh(int client)
{
  /* a search function could be inserted here instead of sequential lookup for
   * more complex node lists (e.g. interleaving) */
  int clientgroup;

#if defined(_WIN32)
  clientgroup = client + 1; /* smp already handles this functionality */
#else

#ifdef HSTART
  if (!arg_scalable_start && !arg_hierarchical_start)
    clientgroup = client + 1; /* only launch 1 core per ssh call */
  else {
    clientgroup = client;
    do {
      clientgroup++; /* add one more client to group if not greater than nodes
                        and shares the same name as client */
      if (clientgroup >= nodetab_rank0_size)
        break;
      if (arg_scalable_start && !arg_hierarchical_start)
        if (strcmp(nodetab_name(clientgroup), nodetab_name(client)))
          break;
      /*Hierarchical-start*/
      if (strcmp(nodetab_name(nodetab_rank0_table[clientgroup]),
                 nodetab_name(nodetab_rank0_table[client])))
        break;
    } while (1);
  }

#else
  if (!arg_scalable_start)
    clientgroup = client + 1; /* only launch 1 core per ssh call */
  else {
    clientgroup = client;
    do {
      clientgroup++; /* add one more client to group if not greater than nodes
                        and shares the same name as client */
    } while (clientgroup < nodetab_rank0_size &&
             (!strcmp(nodetab_getnodeinfo(clientgroup)->name,
                      nodetab_getnodeinfo(client)->name)));
  }
#endif

#endif
  nodetab_getnodeinfo(client)->forks =
      clientgroup - client - 1; /* already have 1 process launching */
  start_one_node_ssh(client);
  return clientgroup - client; /* return number of entries in group */
}

void start_nodes_ssh()
{
  int client, clientgroup;
  ssh_pids = (int *) malloc(sizeof(int) * nodetab_rank0_size);

  if (arg_verbose)
    printf("start_nodes_ssh\n");
  client = 0;
#if CMK_SHRINK_EXPAND
  if (arg_verbose)
    printf("start_nodes_rsh %d %d\n", arg_requested_pes, arg_old_pes);
  if (arg_shrinkexpand) {
    if (arg_requested_pes >= arg_old_pes) { // expand case
      if (arg_verbose)
        printf("Expand %d %d\n", arg_requested_pes, arg_old_pes);
      for (client = 0; client < arg_old_pes; client++)
        ssh_pids[client] = 0;
    } else { // shrink case
      if (arg_verbose)
        printf("Shrink  %d %d\n", arg_requested_pes, arg_old_pes);
      for (client = 0; client < arg_requested_pes; client++)
        ssh_pids[client] = 0;
    }
  }
#endif
  while (client < nodetab_rank0_size) {
    /* start a group of processes per node */
    clientgroup = start_set_node_ssh(client);
    client += clientgroup;
  }
}

/* for mpiexec, for once calling mpiexec to start on all nodes  */
int ssh_fork_one(const char *startScript)
{
  std::vector<const char *> sshargv;
  int pid;
  char npes[128];
  const char *s, *e;

  /* figure out size and dynamic allocate */
  s = nodetab_shell(0);
  e = skipstuff(s);
  while (*s) {
    s = skipblanks(e);
    e = skipstuff(s);
  }

  s = nodetab_shell(0);
  e = skipstuff(s);
  while (*s) {
    sshargv.push_back(substr(s, e));
    s = skipblanks(e);
    e = skipstuff(s);
  }

  if ( ! arg_mpiexec_no_n ) {
    sshargv.push_back("-n");
    sprintf(npes, "%d", nodetab_rank0_size);
    sshargv.push_back(npes);
  }
  sshargv.push_back((char *) startScript);
  sshargv.push_back((const char *) NULL);
  if (arg_verbose)
    printf("Charmrun> Starting %s %s \n", nodetab_shell(0), startScript);

  pid = fork();
  if (pid < 0) {
    perror("ERROR> starting mpiexec");
    exit(1);
  }
  if (pid == 0) { /*Child process*/
    int i;
    /*  unlink(startScript); */
    // removeEnv("DISPLAY="); /*No DISPLAY disables ssh's slow X11 forwarding*/
    for (i = 3; i < 1024; i++)
      close(i);
    execvp(sshargv[0], const_cast<char *const *>(&sshargv[0]));
    fprintf(stderr, "Charmrun> Couldn't find mpiexec program '%s'!\n",
            sshargv[0]);
    exit(1);
  }
  if (arg_verbose)
    fprintf(stderr, "Charmrun> mpiexec started\n");
  return pid;
}

void start_nodes_mpiexec()
{
  int i;

  FILE *f;
  char startScript[200];
  sprintf(startScript, "./charmrun.%d", getpid());
  f = fopen(startScript, "w");
  chmod(startScript, S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IROTH);
  if (f == NULL) {
    /* now try current directory */
    sprintf(startScript, "./charmrun.%d", getpid());
    f = fopen(startScript, "w");
    if (f == NULL) {
      fprintf(stderr, "Charmrun> Can not write file %s!\n", startScript);
      exit(1);
    }
  }
  ssh_script(f, 0, 0, arg_argv, 0);
  fclose(f);
  ssh_pids = (int *) malloc(sizeof(int) * nodetab_rank0_size);
  ssh_pids[0] = ssh_fork_one(startScript);
  for (i = 0; i < nodetab_rank0_size; i++)
    ssh_pids[i] = 0; /* skip finish_nodes */
}

void finish_set_nodes(int start, int stop)
{
  int status, done, i;
  const char *host;

  if (!ssh_pids)
    return; /*nothing to do*/

  std::vector<int> num_retries(stop - start, 0);
  done = 0;
  while (!done) {
    done = 1;
    for (i = start; i < stop; i++) { /* check all nodes */
      if (ssh_pids[i] != 0) {
        done = 0; /* we are not finished yet */
        status = 0;
        waitpid(ssh_pids[i], &status, 0); /* check if the process is finished */
        if (WIFEXITED(status)) {
          if (!WEXITSTATUS(status)) { /* good */
            ssh_pids[i] = 0;          /* process is finished */
          } else {
            host = nodetab_name(nodetab_rank0_table[i]);
            fprintf(stderr,
                    "Charmrun> Error %d returned from remote shell (%s:%d)\n",
                    WEXITSTATUS(status), host, i);

            if (WEXITSTATUS(status) != 255)
              exit(1);

            if (++num_retries[i - start] <= MAX_NUM_RETRIES) {
              fprintf(stderr, "Charmrun> Reconnection attempt %d of %d\n",
                      num_retries[i - start], MAX_NUM_RETRIES);
              start_one_node_ssh(i);
            } else {
              fprintf(
                  stderr,
                  "Charmrun> Too many reconnection attempts; bailing out\n");
              exit(1);
            }
          }
        }
      }
    }
  }
}

void finish_nodes()
{
#ifdef HSTART
  if (arg_hierarchical_start && !arg_child_charmrun)
    finish_set_nodes(0, branchfactor);
  else
#endif
    finish_set_nodes(0, nodetab_rank0_size);
  free(ssh_pids);
}

void kill_nodes()
{
  int rank0no;
  if (!ssh_pids)
    return; /*nothing to do*/
  /*Now wait for all the ssh'es to finish*/
  for (rank0no = 0; rank0no < nodetab_rank0_size; rank0no++) {
    const char *host = nodetab_name(nodetab_rank0_table[rank0no]);
    int status = 0;
    if (arg_verbose)
      printf("Charmrun> waiting for remote shell (%s:%d), pid %d\n", host,
             rank0no, ssh_pids[rank0no]);
    kill(ssh_pids[rank0no], 9);
    waitpid(ssh_pids[rank0no], &status, 0); /*<- no zombies*/
  }
  free(ssh_pids);
}


/* find the absolute path for an executable in the path */
char *find_abs_path(const char *target)
{
  char *thepath=getenv("PATH");
  char *path=strdup(thepath);
  char *subpath=strtok(path,":");
  char *abspath=(char*) malloc(PATH_MAX + strlen(target) + 2);
  while(subpath!=NULL) {
    strcpy(abspath,subpath);
    strcat(abspath,"/");
    strcat(abspath,target);
    if(probefile(abspath)){
      delete path;
      return abspath;
    }
    subpath=strtok(NULL,":");
  }
  free(abspath);
  free(path);
  return NULL;
}

/* simple version of charmrun that avoids the sshd or charmd,   */
/* it spawn the node program just on local machine using exec. */
void start_nodes_local(char **env)
{
  char **envp;
  int envc, rank0no, i;
  int extra = 0;
  
  char **dparamp;
  int dparamc;
#if CMK_AIX && CMK_SMP
  extra = 1;
#endif


  /* copy environ and expanded to hold NETSTART and CmiNumNodes */
  for (envc = 0; env[envc]; envc++)
    ;
  envp = (char **) malloc((envc + 2 + extra + 1) * sizeof(void *));
  for (i = 0; i < envc; i++)
    envp[i] = env[i];
  envp[envc] = (char *) malloc(256);
  envp[envc + 1] = (char *) malloc(256);
#if CMK_AIX && CMK_SMP
  envp[envc + 2] = (char *) malloc(256);
  sprintf(envp[envc + 2], "MALLOCMULTIHEAP=1");
#endif
  envp[envc + 2 + extra] = 0;
  for (i = 0; i < envc; i++)
    envp[i] = env[i];
  envp[envc] = (char *) malloc(256);
  envp[envc + 1] = (char *) malloc(256);
  int dparamoutc=0;
  int dparamoutmax=7;

  /* insert xterm gdb in front of command line and pass args to gdb */
  if(arg_debug || arg_debug_no_pause) {
    int argstringlen=0;
    for (dparamc = 0, argstringlen=0; pparam_argv[dparamc]; dparamc++)
      { 
        if(dparamc>1) argstringlen+=strlen(pparam_argv[dparamc]);
      }
    if(arg_debug_no_pause) dparamoutmax+=2;

    dparamp = (char **) malloc((dparamoutmax) * sizeof(void *));
    char *abs_xterm=find_abs_path(arg_xterm);
    if(!abs_xterm)
      {
      fprintf(stderr, "Charmrun> cannot find xterm for gdb, please add it to your path\n");
      exit(1);
      }
    dparamp[dparamoutc++] = strdup(abs_xterm);
    dparamp[dparamoutc++] = strdup("-e");
    dparamp[dparamoutc++] = strdup(arg_debugger);
    dparamp[dparamoutc++] = strdup(pparam_argv[1]);
    dparamp[dparamoutc++] = strdup("-ex");
    dparamp[dparamoutc] = (char *) malloc(argstringlen + 11 + dparamc);
    strcpy(dparamp[dparamoutc], "set args");
    for(int i=2; i< dparamc; i++)
      {
	strcat(dparamp[dparamoutc], " ");
	strcat(dparamp[dparamoutc], pparam_argv[i]);
      }
    if(arg_debug_no_pause)
      {
	dparamp[++dparamoutc] = strdup("-ex");
	dparamp[++dparamoutc] = strdup("r");
      }
    dparamp[++dparamoutc]=0;  // null terminate your argv or face the wrath of
    // undefined behavior
    if (arg_verbose)
      {
	printf("Charmrun> gdb args : ");
	for (i = 0; i < dparamoutc; i++)
	  printf(" %s ",dparamp[i]);
	printf("\n");
      }
  }
  else
    {
      dparamp=(char **) (pparam_argv+1);
    }

  for (rank0no = 0; rank0no < nodetab_rank0_size; rank0no++) {
    int status = 0;
    int pid;
    int pe = nodetab_rank0_table[rank0no];

    if (arg_verbose)
      printf("Charmrun> start %d node program on localhost.\n", pe);
    sprintf(envp[envc], "NETSTART=%s", create_netstart(rank0no));
    sprintf(envp[envc + 1], "CmiNumNodes=%d", nodetab_rank0_size);
    pid = 0;
    pid = fork();
    if (pid < 0)
      exit(1);
    if (pid == 0) {
      int fd, fd1 = dup(1);
      if (-1 != (fd = open("/dev/null", O_RDWR))) {
        dup2(fd, 0);
        dup2(fd, 1);
        dup2(fd, 2);
      }
      status = execve(dparamp[0],
		      const_cast<char *const *>(dparamp), envp);

      dup2(fd1, 1);
      fprintf(stderr, "execve failed to start process \"%s\" with status: %d\n",
             dparamp[0], status);
      kill(getppid(), 9);
      exit(1);
    }
  }
  if(arg_debug || arg_debug_no_pause)
    {
      for(dparamoutc; dparamoutc>=0;dparamoutc--) free(dparamp[dparamoutc]);
      free(dparamp);
    }
  free(envp[envc]);
  free(envp[envc + 1]);
#if CMK_AIX && CMK_SMP
  free(envp[envc + 2]);
#endif
  free(envp);
}

#ifdef __FAULT__

int current_restart_phase = 1;

void refill_nodetab_entry(int crashed_node);
nodetab_host *replacement_host(int pe);

/**
 * @brief Relaunches a program on the crashed node.
 */
void restart_node(int crashed_node)
{
  int pe = nodetab_rank0_table[crashed_node];
  FILE *f;
  char startScript[200];
  int restart_ssh_pid;
  const char **restart_argv;
  int status = 0;
  char phase_str[10];
  int i;
  /** write the startScript file to be sent**/
  sprintf(startScript, "/tmp/charmrun.%d.%d", getpid(), pe);
  f = fopen(startScript, "w");

  /** add an argument to the argv of the new process
  so that the restarting processor knows that it
  is a restarting processor */
  i = 0;
  while (arg_argv[i] != NULL) {
    i++;
  }
  restart_argv = (const char **) malloc(sizeof(char *) * (i + 4));
  i = 0;
  while (arg_argv[i] != NULL) {
    restart_argv[i] = arg_argv[i];
    i++;
  }
  restart_argv[i] = "+restartaftercrash";
  sprintf(phase_str, "%d", ++current_restart_phase);
  restart_argv[i + 1] = phase_str;
  restart_argv[i + 2] = "+restartisomalloc";
  restart_argv[i + 3] = NULL;

  /** change the nodetable entry of the crashed
processor to connect it to a new one**/
  refill_nodetab_entry(crashed_node);
  ssh_script(f, pe, crashed_node, restart_argv, 1);
  fclose(f);
  /**start the new processor */
  restart_ssh_pid = ssh_fork(pe, startScript);
  /**wait for the reply from the new process*/
  status = 0;
  if (arg_debug_no_pause || arg_debug)
    ;
  else {
    do {
      waitpid(restart_ssh_pid, &status, 0);
    } while (!WIFEXITED(status));
    if (WEXITSTATUS(status) != 0) {
      fprintf(stderr,
              "Charmrun> Error %d returned from new attempted remote shell \n",
              WEXITSTATUS(status));
      exit(1);
    }
  }
  PRINT(("Charmrun finished launching new process in %fs\n",
         GetClock() - ftTimer));
}

void refill_nodetab_entry(int crashed_node)
{
  int pe = nodetab_rank0_table[crashed_node];
  nodetab_host *h = nodetab_table[pe];
  *h = *(replacement_host(pe));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  fprintf(stderr, "Charmrun>>> New pe %d is on host %s \n", pe,
          nodetab_name(pe));
#endif
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
nodetab_host *replacement_host(int pe)
{
  int x = loaded_max_pe + 1;

  x = x % arg_read_pes;
  loaded_max_pe += 1;
  /*  while(x == pe){
   *       x = rand()%nodetab_size;
   *           }*/
  fprintf(stderr, "Charmrun>>> replacing pe %d with %d host %s with %s \n", pe,
          x, nodetab_name(pe), nodetab_name(x));
  return nodetab_table[x];
}
#else
nodetab_host *replacement_host(int pe)
{
  int x = pe;
  while (x == pe) {
#ifdef HSTART
    if (arg_hierarchical_start) {
      x = nodetab_rank0_table[rand() % nodetab_rank0_size];
      crashed_pe_id = pe;
      restarted_pe_id = x;
    } else
#endif
      x = rand() % nodetab_size;
  }
  return nodetab_table[x];
}
#endif

/**
 * @brief Reconnects a crashed node. It waits for the I-tuple from the just
 * relaunched program. It also:
 * i) Broadcast the nodetabtable to every other node.
 * ii) Announces the crash to every other node.
 */
void reconnect_crashed_client(int socket_index, int crashed_node)
{
  int i;
  unsigned int clientPort;
  skt_ip_t clientIP;
  ChSingleNodeinfo *in;
  if (0 == skt_select1(server_fd, arg_timeout * 1000)) {
    client_connect_problem(
        socket_index, socket_index,
        "Timeout waiting for restarted node-program to connect");
  }
  req_clients[socket_index] = skt_accept(server_fd, &clientIP, &clientPort);
  skt_client_table[req_clients[socket_index]] = crashed_node;

  if (req_clients[socket_index] == SOCKET_ERROR) {
    client_connect_problem(socket_index, socket_index,
                           "Failure in restarted node accept");
  } else {
    ChMessage msg;
    if (!skt_select1(req_clients[socket_index], arg_timeout * 1000)) {
      client_connect_problem(socket_index, socket_index,
                             "Timeout on IP request for restarted processor");
    }

#ifdef HSTART
    if (arg_hierarchical_start) {
      req_forward_root(req_clients[socket_index]);
      if (_last_crash != 0) {
        fprintf(stderr, "ERROR> Charmrun detected multiple crashes.\n");
        exit(1);
      }

      _last_crash = crashed_node;
      _crash_socket_index = socket_index;
      return;
    }
#endif
    ChMessage_recv(req_clients[socket_index], &msg);
    if (msg.len != sizeof(ChSingleNodeinfo)) {
      fprintf(stderr, "Charmrun: Bad initnode data length. Aborting\n");
      fprintf(stderr, "Charmrun: possibly because: %s.\n", msg.data);
    }
    fprintf(stderr, "socket_index %d crashed_node %d reconnected fd %d  \n",
            socket_index, crashed_node, req_clients[socket_index]);

    /** update the nodetab entry corresponding to
    this node, skip the restarted one */
    in = (ChSingleNodeinfo *) msg.data;
    nodeinfo_add(in, req_clients[socket_index]);
    for (i = 0; i < req_nClients; i++) {
      if (i != socket_index) {
        req_handle_initnodetab(NULL, req_clients[i]);
      }
    }

    /* tell every one there is a crash */
    announce_crash(socket_index, crashed_node);
    if (_last_crash != 0) {
      fprintf(stderr, "ERROR> Charmrun detected multiple crashes.\n");
      exit(1);
    }
    _last_crash = crashed_node;
    _crash_socket_index = socket_index;
    /*holds the restarted process until I got ack back from
      everyone in req_handle_crashack
      now the restarted one can only continue until
      req_handle_crashack calls req_handle_initnodetab(socket_index)
      req_handle_initnodetab(NULL,req_clients[socket_index]); */
    ChMessage_free(&msg);
  }
}

/**
 * @brief Sends a message announcing the crash to every other node. This message
 * will be used to
 * trigger fault tolerance methods.
 */
void announce_crash(int socket_index, int crashed_node)
{
  int i;
  ChMessageHeader hdr;
  ChMessageInt_t crashNo = ChMessageInt_new(crashed_node);
  ChMessageHeader_new("crashnode", sizeof(ChMessageInt_t), &hdr);
  for (i = 0; i < req_nClients; i++) {
    if (i != socket_index) {
      skt_sendN(req_clients[i], (const char *) &hdr, sizeof(hdr));
      skt_sendN(req_clients[i], (const char *) &crashNo,
                sizeof(ChMessageInt_t));
    }
  }
}

#endif

#endif /*CMK_USE_SSH*/
