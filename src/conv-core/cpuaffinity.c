
/*
 This scheme relies on using IP address to identify nodes and assigning 
cpu affinity.  

 when CMK_NO_SOCKETS, which is typically on cray xt3 and bluegene/L.
 There is no hostname for the compute nodes.
 *
 * last updated 3/20/2010   Gengbin Zheng
 * new options +pemap +commmap takes complex pattern of a list of cores
*/

#define _GNU_SOURCE

#include "converse.h"
#include "sockRoutines.h"

#define DEBUGP(x)   /* CmiPrintf x;  */
CpvDeclare(int, myCPUAffToCore);
#if CMK_OS_IS_LINUX
/* 
 * /proc/<PID>/[task/<TID>]/stat file descriptor 
 * Used to retrieve the info about which physical
 * coer this process or thread is on.
 **/
CpvDeclare(void *, myProcStatFP);
#endif

#if CMK_HAS_SETAFFINITY || defined (_WIN32) || CMK_HAS_BINDPROCESSOR

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else
#define _GNU_SOURCE
#include <sched.h>
//long sched_setaffinity(pid_t pid, unsigned int len, unsigned long *user_mask_ptr);
//long sched_getaffinity(pid_t pid, unsigned int len, unsigned long *user_mask_ptr);
#endif

#if CMK_OS_IS_LINUX
#include <sys/syscall.h>
#endif

#if defined(__APPLE__) 
#include <Carbon/Carbon.h> /* Carbon APIs for Multiprocessing */
#endif

#if defined(ARCH_HPUX11) ||  defined(ARCH_HPUX10)
#include <sys/mpctl.h>
#endif


#define MAX_EXCLUDE      32
static int excludecore[MAX_EXCLUDE] = {-1};
static int excludecount = 0;

static int affinity_doneflag = 0;

static int in_exclude(int core)
{
  int i;
  for (i=0; i<excludecount; i++) if (core == excludecore[i]) return 1;
  return 0;
}

static void add_exclude(int core)
{
  if (in_exclude(core)) return;
  CmiAssert(excludecount < MAX_EXCLUDE);
  excludecore[excludecount++] = core;
}

#if CMK_HAS_BINDPROCESSOR
#include <sys/processor.h>
#endif

#define SET_MASK(cpuid)    \
  /* set the affinity mask if possible */      \
  if ((cpuid / 8) > len) {      \
    printf("Mask size too small to handle requested CPU ID\n");   \
    return -1;      \
  } else {    \
    mask = 1 << cpuid;   /* set the affinity mask exclusively to one CPU */ \
  }


/* This implementation assumes the default x86 CPU mask size used by Linux */
/* For a large SMP machine, this code should be changed to use a variable sized   */
/* CPU affinity mask buffer instead, as the present code will fail beyond 32 CPUs */
int set_cpu_affinity(unsigned int cpuid) {
  unsigned long mask = 0xffffffff;
  unsigned int len = sizeof(mask);
  int retValue = 0;
  int pid;

 #ifdef _WIN32
   HANDLE hProcess;
 #endif
 
#ifdef _WIN32
  SET_MASK(cpuid)
  hProcess = GetCurrentProcess();
  if (SetProcessAffinityMask(hProcess, mask) == 0) {
    return -1;
  }
#elif CMK_HAS_BINDPROCESSOR
  pid = getpid();
  if (bindprocessor(BINDPROCESS, pid, cpuid) == -1) return -1;
#else
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpuid, &cpuset);
  /*SET_MASK(cpuid)*/

  /* PID 0 refers to the current process */
  /*if (sched_setaffinity(0, len, &mask) < 0) {*/
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    perror("sched_setaffinity");
    return -1;
  }
#endif

  return 0;
}

#if CMK_SMP
int set_thread_affinity(int cpuid) {
  unsigned long mask = 0xffffffff;
  unsigned int len = sizeof(mask);

#ifdef _WIN32
  HANDLE hThread;
#endif	
  
#ifdef _WIN32
  SET_MASK(cpuid)
  hThread = GetCurrentThread();
  if (SetThreadAffinityMask(hThread, mask) == 0) {
    return -1;
  }
#elif  CMK_HAS_PTHREAD_SETAFFINITY
  int s, j;
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  CPU_ZERO(&cpuset);
  CPU_SET(cpuid, &cpuset);

  s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    perror("pthread_setaffinity");
    return -1;
  }
#elif CMK_HAS_BINDPROCESSOR
  if (bindprocessor(BINDTHREAD, thread_self(), cpuid) != 0)
    return -1;
#else
  return set_cpu_affinity(cpuid);
#endif

  return 0;
}
#endif


int CmiSetCPUAffinity(int mycore)
{
  int core = mycore;
  if (core < 0) {
    core = CmiNumCores() + core;
  }
  if (core < 0) {
    CmiError("Error: Invalid cpu affinity core number: %d\n", mycore);
    CmiAbort("CmiSetCPUAffinity failed");
  }

  CpvAccess(myCPUAffToCore) = core;

  /* set cpu affinity */
#if CMK_SMP
  return set_thread_affinity(core);
#else
  return set_cpu_affinity(core);
  /* print_cpu_affinity(); */
#endif
}

/* This implementation assumes the default x86 CPU mask size used by Linux */
/* For a large SMP machine, this code should be changed to use a variable sized   */
/* CPU affinity mask buffer instead, as the present code will fail beyond 32 CPUs */
int print_cpu_affinity() {
#ifdef _WIN32
  unsigned long pMask, sMask;
  HANDLE hProcess = GetCurrentProcess();
  if(GetProcessAffinityMask(hProcess, &pMask, &sMask)){
	perror("On Windows: GetProcessAffinityMask");
    return -1;
  }
  
 CmiPrintf("[%d] CPU affinity mask is: 0x%08lx\n", CmiMyPe(), pMask);
  
#elif CMK_HAS_BINDPROCESSOR
  printf("[%d] CPU affinity mask is unknown for AIX. \n", CmiMyPe());
#else
  /*unsigned long mask;
  unsigned int len = sizeof(mask);*/
  cpu_set_t cpuset;
  char str[256], pe[16];
  int i;
  CPU_ZERO(&cpuset);
 
  /* PID 0 refers to the current process */
  /*if (sched_getaffinity(0, len, &mask) < 0) {*/
  if (sched_getaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    perror("sched_getaffinity");
    return -1;
  }

  sprintf(str, "[%d] CPU affinity mask is: ", CmiMyPe());
  for (i = 0; i < CPU_SETSIZE; i++)
        if (CPU_ISSET(i, &cpuset)) {
            sprintf(pe, " %d ", i);
            strcat(str, pe);
        }
  CmiPrintf("%s\n", str);  
#endif
  return 0;
}

#if CMK_SMP
int print_thread_affinity() {
  unsigned long mask;
  size_t len = sizeof(mask);

#if  CMK_HAS_PTHREAD_SETAFFINITY
  int s, j;
  cpu_set_t cpuset;
  pthread_t thread;
  char str[256], pe[16];

  thread = pthread_self();
  s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    perror("pthread_getaffinity");
    return -1;
  }

  sprintf(str, "[%d] %s affinity is: ", CmiMyPe(), CmiMyPe()>=CmiNumPes()?"communication pthread":"pthread");
  for (j = 0; j < CPU_SETSIZE; j++)
        if (CPU_ISSET(j, &cpuset)) {
            sprintf(pe, " %d ", j);
            strcat(str, pe);
        }
  CmiPrintf("%s\n", str);
#endif
  return 0;
}
#endif

int CmiPrintCPUAffinity()
{
#if CMK_SMP
  return print_thread_affinity();
#else
  return print_cpu_affinity();
#endif
}

int CmiOnCore(void) {
#if CMK_OS_IS_LINUX
  /*
   * The info (task_cpu) is read from the Linux /proc virtual file system.
   * The /proc/<PID>/[task/<TID>]/stat is explained in the Linux
   * kernel documentation. The online one could be found in:
   * http://www.mjmwired.net/kernel/Documentation/filesystems/proc.txt
   * Based on the documentation, task_cpu is found at the 39th field in
   * the stat file.
   **/
#define TASK_CPU_POS (39)
  int n;
  char str[128];
  FILE *fp = (FILE *)CpvAccess(myProcStatFP);
  if (fp == NULL) return -1;
  fseek(fp, 0, SEEK_SET);
  for (n=0; n<TASK_CPU_POS; n++)  {
    fscanf(fp, "%s", str);
  }  
  return atoi(str);
#else
  printf("WARNING: CmiOnCore IS NOT SUPPORTED ON THIS PLATFORM\n");
  return -1;
#endif
}


static int cpuAffinityHandlerIdx;
static int cpuAffinityRecvHandlerIdx;

typedef struct _hostnameMsg {
  char core[CmiMsgHeaderSizeBytes];
  int pe;
  skt_ip_t ip;
  int ncores;
  int rank;
  int seq;
} hostnameMsg;

typedef struct _rankMsg {
  char core[CmiMsgHeaderSizeBytes];
  int *ranks;                  /* PE => core rank mapping */
  int *nodes;                  /* PE => node number mapping */
} rankMsg;

static rankMsg *rankmsg = NULL;
static CmmTable hostTable;
static CmiNodeLock affLock = NULL;

/* called on PE 0 */
static void cpuAffinityHandler(void *m)
{
  static int count = 0;
  static int nodecount = 0;
  hostnameMsg *rec;
  hostnameMsg *msg = (hostnameMsg *)m;
  hostnameMsg *tmpm;
  int tag, tag1, pe, myrank;
  int npes = CmiNumPes();

/*   for debug
  char str[128];
  skt_print_ip(str, msg->ip);
  printf("hostname: %d %s\n", msg->pe, str);
*/
  CmiAssert(CmiMyPe()==0 && rankmsg != NULL);
  tag = *(int*)&msg->ip;
  pe = msg->pe;
  if ((rec = (hostnameMsg *)CmmProbe(hostTable, 1, &tag, &tag1)) != NULL) {
    CmiFree(msg);
  }
  else {
    rec = msg;
    rec->seq = nodecount;
    nodecount++;                          /* a new node record */
    CmmPut(hostTable, 1, &tag, msg);
  }
  myrank = rec->rank%rec->ncores;
  while (in_exclude(myrank)) {             /* skip excluded core */
    myrank = (myrank+1)%rec->ncores;
    rec->rank ++;
  }
  rankmsg->ranks[pe] = myrank;             /* core rank */
  rankmsg->nodes[pe] = rec->seq;           /* on which node */
  rec->rank ++;
  count ++;
  if (count == CmiNumPes()) {
    /* CmiPrintf("Cpuaffinity> %d unique compute nodes detected! \n", CmmEntries(hostTable)); */
    tag = CmmWildCard;
    while (tmpm = CmmGet(hostTable, 1, &tag, &tag1)) CmiFree(tmpm);
    CmmFree(hostTable);
#if 1
    /* bubble sort ranks on each node according to the PE number */
    {
    int i,j;
    for (i=0; i<npes-1; i++)
      for(j=i+1; j<npes; j++) {
        if (rankmsg->nodes[i] == rankmsg->nodes[j] && 
              rankmsg->ranks[i] > rankmsg->ranks[j]) 
        {
          int tmp = rankmsg->ranks[i];
          rankmsg->ranks[i] = rankmsg->ranks[j];
          rankmsg->ranks[j] = tmp;
        }
      }
    }
#endif
    CmiSyncBroadcastAllAndFree(sizeof(rankMsg)+CmiNumPes()*sizeof(int)*2, (void *)rankmsg);
  }
}

/* called on each processor */
static void cpuAffinityRecvHandler(void *msg)
{
  int myrank, mynode;
  rankMsg *m = (rankMsg *)msg;
  m->ranks = (int *)((char*)m + sizeof(rankMsg));
  m->nodes = (int *)((char*)m + sizeof(rankMsg) + CmiNumPes()*sizeof(int));
  myrank = m->ranks[CmiMyPe()];
  mynode = m->nodes[CmiMyPe()];

  /*CmiPrintf("[%d %d] set to core #: %d\n", CmiMyNode(), CmiMyPe(), myrank);*/

  if (-1 != CmiSetCPUAffinity(myrank)) {
    DEBUGP(("Processor %d is bound to core #%d on node #%d\n", CmiMyPe(), myrank, mynode));
  }
  else{
    CmiPrintf("Processor %d set affinity failed!\n", CmiMyPe());
    CmiAbort("set cpu affinity abort!\n");
  }
  CmiFree(m);
}

#if defined(_WIN32) && ! defined(__CYGWIN__)
  /* strtok is thread safe in VC++ */
#define strtok_r(x,y,z) strtok(x,y)
#endif

static int search_pemap(char *pecoremap, int pe)
{
  int *map = (int *)malloc(CmiNumPes()*sizeof(int));
  char *ptr = NULL;
  int i, j, count;
  char *str;

  char *mapstr = (char*)malloc(strlen(pecoremap)+1);
  strcpy(mapstr, pecoremap);

  str = strtok_r(mapstr, ",", &ptr);
  count = 0;
  while (str && count < CmiNumPes())
  {
      int hasdash=0, hascolon=0, hasdot=0;
      int start, end, stride=1, block=1;
      for (i=0; i<strlen(str); i++) {
          if (str[i] == '-' && i!=0) hasdash=1;
          if (str[i] == ':') hascolon=1;
	  if (str[i] == '.') hasdot=1;
      }
      if (hasdash) {
          if (hascolon) {
            if (hasdot) {
              if (sscanf(str, "%d-%d:%d.%d", &start, &end, &stride, &block) != 4)
                 printf("Warning: Check the format of \"%s\".\n", str);
            }
            else {
              if (sscanf(str, "%d-%d:%d", &start, &end, &stride) != 3)
                 printf("Warning: Check the format of \"%s\".\n", str);
            }
          }
          else {
            if (sscanf(str, "%d-%d", &start, &end) != 2)
                 printf("Warning: Check the format of \"%s\".\n", str);
          }
      }
      else {
          sscanf(str, "%d", &start);
          end = start;
      }
      if (block > stride) {
        printf("Warning: invalid block size in \"%s\" ignored.\n", str);
        block=1;
      }
      for (i = start; i<=end; i+=stride) {
        for (j=0; j<block; j++) {
          if (i+j>end) break;
          map[count++] = i+j;
          if (count == CmiNumPes()) break;
        }
        if (count == CmiNumPes()) break;
      }
      str = strtok_r(NULL, ",", &ptr);
  }
  i = map[pe % count];

  free(map);
  free(mapstr);
  return i;
}

#if CMK_CRAYXT
extern int getXTNodeID(int mpirank, int nummpiranks);
#endif


void CmiInitCPUAffinity(char **argv)
{
  static skt_ip_t myip;
  int ret, i, exclude;
  hostnameMsg  *msg;
  char *pemap = NULL;
  char *commap = NULL;
 
  int show_affinity_flag;
  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
						"set cpu affinity");

  while (CmiGetArgIntDesc(argv,"+excludecore", &exclude, "avoid core when setting cpuaffinity")) 
    if (CmiMyRank() == 0) add_exclude(exclude);

  CmiGetArgStringDesc(argv, "+pemap", &pemap, "define pe to core mapping");
  if (pemap!=NULL && excludecount>0)
    CmiAbort("Charm++> +pemap can not be used with +excludecore.\n");
  CmiGetArgStringDesc(argv, "+commap", &commap, "define comm threads to core mapping");
  show_affinity_flag = CmiGetArgFlagDesc(argv,"+showcpuaffinity",
						"print cpu affinity");

  cpuAffinityHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityHandler);
  cpuAffinityRecvHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityRecvHandler);

  if (CmiMyRank() ==0) {
     affLock = CmiCreateLock();
  }

  if (!affinity_flag) {
    if (show_affinity_flag) CmiPrintCPUAffinity();
    return;
  }
  else if (CmiMyPe() == 0) {
     CmiPrintf("Charm++> cpu affinity enabled. \n");
     if (excludecount > 0) {
       CmiPrintf("Charm++> cpuaffinity excludes core: %d", excludecore[0]);
       for (i=1; i<excludecount; i++) CmiPrintf(" %d", excludecore[i]);
       CmiPrintf(".\n");
     }
     if (pemap!=NULL)
       CmiPrintf("Charm++> cpuaffinity PE-core map : %s\n", pemap);
  }

  if (CmiMyPe() >= CmiNumPes()) {         /* this is comm thread */
      /* comm thread either can float around, or pin down to the last rank.
         however it seems to be reportedly slower if it is floating */
    CmiNodeAllBarrier();
    if (commap != NULL) {
      int mycore = search_pemap(commap, CmiMyPe()-CmiNumPes());
      printf("Charm++> set comm %d on node %d to core #%d\n", CmiMyPe()-CmiNumPes(), CmiMyNode(), mycore); 
      if (-1 == CmiSetCPUAffinity(mycore))
        CmiAbort("set_cpu_affinity abort!");
    }
    else {
    /* if (CmiSetCPUAffinity(CmiNumCores()-1) == -1) CmiAbort("set_cpu_affinity abort!"); */
    }
    if (pemap == NULL) {
#if CMK_MACHINE_PROGRESS_DEFINED
    while (affinity_doneflag < CmiMyNodeSize())  CmiNetworkProgress();
#else
#if CMK_SMP
    #error "Machine progress call needs to be implemented for cpu affinity!"
#endif
#endif
    }
    CmiNodeAllBarrier();
    if (show_affinity_flag) CmiPrintCPUAffinity();
    return;    /* comm thread return */
  }

  if (pemap != NULL) {
    int mycore = search_pemap(pemap, CmiMyPe());
    CmiPrintf("Charm++> set PE %d on node %d to core #%d\n", CmiMyPe(), CmiMyNode(), mycore); 
    if (mycore >= CmiNumCores()) {
      CmiPrintf("Error> Invalid core number %d, only have %d cores (0-%d) on the node. \n", mycore, CmiNumCores(), CmiNumCores()-1);
      CmiAbort("Invalid core number");
    }
    if (CmiSetCPUAffinity(mycore) == -1) CmiAbort("set_cpu_affinity abort!");
    CmiNodeAllBarrier();
    CmiNodeAllBarrier();
    if (show_affinity_flag) CmiPrintCPUAffinity();
    return;
  }

    /* get my ip address */
  if (CmiMyRank() == 0)
  {
#if CMK_CRAYXT
    ret = getXTNodeID(CmiMyNode(), CmiNumNodes());
    memcpy(&myip, &ret, sizeof(int));
#elif CMK_HAS_GETHOSTNAME
    myip = skt_my_ip();        /* not thread safe, so only calls on rank 0 */
#else
    CmiAbort("Can not get unique name for the compute nodes. \n");
#endif
  }
  CmiNodeAllBarrier();

    /* prepare a msg to send */
  msg = (hostnameMsg *)CmiAlloc(sizeof(hostnameMsg));
  CmiSetHandler((char *)msg, cpuAffinityHandlerIdx);
  msg->pe = CmiMyPe();
  msg->ip = myip;
  msg->ncores = CmiNumCores();
  DEBUGP(("PE %d's node has %d number of cores. \n", CmiMyPe(), msg->ncores));
  msg->rank = 0;
  CmiSyncSendAndFree(0, sizeof(hostnameMsg), (void *)msg);

  if (CmiMyPe() == 0) {
    int i;
    hostTable = CmmNew();
    rankmsg = (rankMsg *)CmiAlloc(sizeof(rankMsg)+CmiNumPes()*sizeof(int)*2);
    CmiSetHandler((char *)rankmsg, cpuAffinityRecvHandlerIdx);
    rankmsg->ranks = (int *)((char*)rankmsg + sizeof(rankMsg));
    rankmsg->nodes = (int *)((char*)rankmsg + sizeof(rankMsg) + CmiNumPes()*sizeof(int));
    for (i=0; i<CmiNumPes(); i++) {
      rankmsg->ranks[i] = 0;
      rankmsg->nodes[i] = -1;
    }

    for (i=0; i<CmiNumPes(); i++) CmiDeliverSpecificMsg(cpuAffinityHandlerIdx);
  }

    /* receive broadcast from PE 0 */
  CmiDeliverSpecificMsg(cpuAffinityRecvHandlerIdx);
  CmiLock(affLock);
  affinity_doneflag++;
  CmiUnlock(affLock);
  CmiNodeAllBarrier();

  if (show_affinity_flag) CmiPrintCPUAffinity();
}

/* called in ConverseCommonInit to initialize basic variables */
void CmiInitCPUAffinityUtil(){
    CpvInitialize(int, myCPUAffToCore);
    CpvAccess(myCPUAffToCore) = -1;
#if CMK_OS_IS_LINUX
    char fname[128];
#if CMK_SMP
    sprintf(fname, "/proc/%d/task/%d/stat", getpid(), syscall(SYS_gettid));
#else
    sprintf(fname, "/proc/%d/stat", getpid());
#endif	
    CpvInitialize(void *, myProcStatFP);
    CpvAccess(myProcStatFP) = (void *)fopen(fname, "r");
/*
    if(CmiMyPe()==0 && CpvAccess(myProcStatFP) == NULL){
        CmiPrintf("WARNING: ERROR IN OPENING FILE %s on PROC %d, CmiOnCore() SHOULDN'T BE CALLED\n", fname, CmiMyPe()); 
*/
    }
#endif
}

#else           /* not supporting affinity */


int CmiPrintCPUAffinity()
{
  CmiPrintf("Warning: CmiPrintCPUAffinity not supported.\n");
}

void CmiInitCPUAffinity(char **argv)
{
  char *pemap = NULL;
  char *commap = NULL;
  int excludecore = -1;
  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
						"set cpu affinity");
  while (CmiGetArgIntDesc(argv,"+excludecore",&excludecore, "avoid core when setting cpuaffinity"));
  CmiGetArgStringDesc(argv, "+pemap", &pemap, "define pe to core mapping");
  CmiGetArgStringDesc(argv, "+commap", &commap, "define comm threads to core mapping");
  if (affinity_flag && CmiMyPe()==0)
    CmiPrintf("sched_setaffinity() is not supported, +setcpuaffinity disabled.\n");
}

/* called in ConverseCommonInit to initialize basic variables */
void CmiInitCPUAffinityUtil(){
    CpvInitialize(int, myCPUAffToCore);
    CpvAccess(myCPUAffToCore) = -1;
#if CMK_OS_IS_LINUX	
    CpvInitialize(void *, myProcStatFP);
    CpvAccess(myProcStatFP) = NULL;
 #endif
}

int CmiOnCore(){
	return -1;
}
#endif
