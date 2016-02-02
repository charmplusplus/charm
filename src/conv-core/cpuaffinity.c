
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

#define DEBUGP(x)    /* CmiPrintf x;  */
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
#include <errno.h>

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


#define MAX_EXCLUDE      64
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
#ifdef CPU_ALLOC
 if ( cpuid >= CPU_SETSIZE ) {
  cpu_set_t *cpusetp;
  size_t size;
  int num_cpus;
  num_cpus = cpuid + 1;
  cpusetp = CPU_ALLOC(num_cpus);
  if (cpusetp == NULL) {
    perror("set_cpu_affinity CPU_ALLOC");
    return -1;
  }
  size = CPU_ALLOC_SIZE(num_cpus);
  CPU_ZERO_S(size, cpusetp);
  CPU_SET_S(cpuid, size, cpusetp);
  if (sched_setaffinity(0, size, cpusetp) < 0) {
    perror("sched_setaffinity dynamically allocated");
    CPU_FREE(cpusetp);
    return -1;
  }
  CPU_FREE(cpusetp);
 } else
#endif
 {
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
#ifdef CPU_ALLOC
 if ( cpuid >= CPU_SETSIZE ) {
  cpu_set_t *cpusetp;
  pthread_t thread;
  size_t size;
  int num_cpus;
  num_cpus = cpuid + 1;
  cpusetp = CPU_ALLOC(num_cpus);
  if (cpusetp == NULL) {
    perror("set_thread_affinity CPU_ALLOC");
    return -1;
  }
  size = CPU_ALLOC_SIZE(num_cpus);
  thread = pthread_self();
  CPU_ZERO_S(size, cpusetp);
  CPU_SET_S(cpuid, size, cpusetp);
  if (errno = pthread_setaffinity_np(thread, size, cpusetp)) {
    perror("pthread_setaffinity dynamically allocated");
    CPU_FREE(cpusetp);
    return -1;
  }
  CPU_FREE(cpusetp);
 } else
#endif
 {
  int s, j;
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  CPU_ZERO(&cpuset);
  CPU_SET(cpuid, &cpuset);

  if (errno = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) {
    perror("pthread_setaffinity");
    return -1;
  }
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
  int j;
  cpu_set_t cpuset;
  pthread_t thread;
  char str[256], pe[16];

  thread = pthread_self();
  
  if (errno = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) {
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

int CmiOnCore() {
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
  if (fp == NULL){
    printf("WARNING: CmiOnCore IS NOT SUPPORTED ON THIS PLATFORM\n");
    return -1;
  }
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
static CmiNodeLock affLock = 0;

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
  int *map = (int *)malloc(CmiNumPesGlobal()*sizeof(int));
  char *ptr = NULL;
  int h, i, j, k, count;
  int plusarr[128];
  char *str;

  char *mapstr = (char*)malloc(strlen(pecoremap)+1);
  strcpy(mapstr, pecoremap);

  str = strtok_r(mapstr, ",", &ptr);
  count = 0;
  while (str && count < CmiNumPesGlobal())
  {
      int hasdash=0, hascolon=0, hasdot=0, hasstar1=0, hasstar2=0, numplus=0;
      int start, end, stride=1, block=1;
      int iter=1;
      plusarr[0] = 0;
      for (i=0; i<strlen(str); i++) {
          if (str[i] == '-' && i!=0) hasdash=1;
          else if (str[i] == ':') hascolon=1;
	  else if (str[i] == '.') hasdot=1;
	  else if (str[i] == 'x') hasstar1=1;
	  else if (str[i] == 'X') hasstar2=1;
	  else if (str[i] == '+') {
            if (str[i+1] == '+' || str[i+1] == '-') {
              printf("Warning: Check the format of \"%s\".\n", str);
            } else if (sscanf(&str[i], "+%d", &plusarr[++numplus]) != 1) {
              printf("Warning: Check the format of \"%s\".\n", str);
              --numplus;
            }
          }
      }
      if (hasstar1 || hasstar2) {
          if (hasstar1) sscanf(str, "%dx", &iter);
          if (hasstar2) sscanf(str, "%dX", &iter);
          while (*str!='x' && *str!='X') str++;
          str++;
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
      //if (CmiMyPe() == 0) printf("iter: %d start: %d end: %d stride: %d, block: %d. plus %d \n", iter, start, end, stride, block, numplus);
      for (k = 0; k<iter; k++) {
        for (i = start; i<=end; i+=stride) {
          for (j=0; j<block; j++) {
            if (i+j>end) break;
            for (h=0; h<=numplus; h++) {
              map[count++] = i+j+plusarr[h];
              if (count == CmiNumPesGlobal()) break;
            }
            if (count == CmiNumPesGlobal()) break;
          }
          if (count == CmiNumPesGlobal()) break;
        }
        if (count == CmiNumPesGlobal()) break;
      }
      str = strtok_r(NULL, ",", &ptr);
  }
  i = map[pe % count];

  free(map);
  free(mapstr);
  return i;
}

#if CMK_CRAYXT || CMK_CRAYXE || CMK_CRAYXC
extern int getXTNodeID(int mpirank, int nummpiranks);
#endif


void CmiInitCPUAffinity(char **argv)
{
  static skt_ip_t myip;
  int ret, i, exclude;
  hostnameMsg  *msg;
  char *pemap = NULL;
  char *commap = NULL;
  char *pemapfile = NULL;
 
  int show_affinity_flag;
  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
						"set cpu affinity");

  while (CmiGetArgIntDesc(argv,"+excludecore", &exclude, "avoid core when setting cpuaffinity"))  {
    if (CmiMyRank() == 0) add_exclude(exclude);
    affinity_flag = 1;
  }

  if (CmiGetArgStringDesc(argv, "+pemapfile", &pemapfile, "define pe to core mapping file")) {
    FILE *fp;
    char buf[128];
    pemap = (char*)malloc(1024);
    fp = fopen(pemapfile, "r");
    if (fp == NULL) CmiAbort("pemapfile does not exist");
    while (!feof(fp)) {
      if (fgets(buf, 128, fp)) {
        if (buf[strlen(buf)-1] == '\n') buf[strlen(buf)-1] = 0;
        strcat(pemap, buf);
      }
    }
    fclose(fp);
    if (CmiMyPe()==0) CmiPrintf("Charm++> read from pemap file '%s': %s\n", pemapfile, pemap);
  }

  CmiGetArgStringDesc(argv, "+pemap", &pemap, "define pe to core mapping");
  if (pemap!=NULL && excludecount>0)
    CmiAbort("Charm++> +pemap can not be used with +excludecore.\n");

  CmiGetArgStringDesc(argv, "+commap", &commap, "define comm threads to core mapping");

  if (pemap!=NULL || commap!=NULL) affinity_flag = 1;

#if CMK_PAMI_LINUX_PPC8
  affinity_flag = 1;
#endif

  show_affinity_flag = CmiGetArgFlagDesc(argv,"+showcpuaffinity",
						"print cpu affinity");

  cpuAffinityHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityHandler);
  cpuAffinityRecvHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityRecvHandler);

  if (CmiMyRank() ==0) {
     affLock = CmiCreateLock();
  }

#if CMK_BLUEGENEP || CMK_BLUEGENEQ
  if(affinity_flag){
      affinity_flag = 0;
      if(CmiMyPe()==0) CmiPrintf("Charm++> cpu affinity setting is not needed on Blue Gene, thus ignored.\n");
  }
  if(show_affinity_flag){
      show_affinity_flag = 0;
      if(CmiMyPe()==0) CmiPrintf("Charm++> printing cpu affinity is not supported on Blue Gene.\n");
  }
#endif

  if (!affinity_flag) {
    if (show_affinity_flag) CmiPrintCPUAffinity();
    return;
  }

  if (CmiMyPe() == 0) {
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
      int mycore = search_pemap(commap, CmiMyPeGlobal()-CmiNumPesGlobal());
      if(CmiMyPe()-CmiNumPes()==0) printf("Charm++> set comm %d on node %d to core #%d\n", CmiMyPe()-CmiNumPes(), CmiMyNode(), mycore); 
      if (-1 == CmiSetCPUAffinity(mycore))
        CmiAbort("set_cpu_affinity abort!");
      CmiNodeAllBarrier();
      if (show_affinity_flag) CmiPrintCPUAffinity();
      return;    /* comm thread return */
    }
    else {
    /* if (CmiSetCPUAffinity(CmiNumCores()-1) == -1) CmiAbort("set_cpu_affinity abort!"); */
#if !CMK_CRAYXT && !CMK_CRAYXE && !CMK_CRAYXC && !CMK_BLUEGENEQ && !CMK_PAMI_LINUX_PPC8
      if (pemap == NULL) {
#if CMK_MACHINE_PROGRESS_DEFINED
        while (affinity_doneflag < CmiMyNodeSize())  CmiNetworkProgress();
#else
#if CMK_SMP
        #error "Machine progress call needs to be implemented for cpu affinity!"
#endif
#endif
      }
#endif
#if CMK_CRAYXT || CMK_CRAYXE || CMK_CRAYXC
      /* if both pemap and commmap are NULL, will compute one */
      if (pemap != NULL)      
#endif
      {
      CmiNodeAllBarrier();
      if (show_affinity_flag) CmiPrintCPUAffinity();
      return;    /* comm thread return */
      }
    }
  }

  if (pemap != NULL && CmiMyPe()<CmiNumPes()) {    /* work thread */
    int mycore = search_pemap(pemap, CmiMyPeGlobal());
    if(show_affinity_flag) CmiPrintf("Charm++> set PE %d on node %d to core #%d\n", CmiMyPe(), CmiMyNode(), mycore); 
    if (mycore >= CmiNumCores()) {
      CmiPrintf("Error> Invalid core number %d, only have %d cores (0-%d) on the node. \n", mycore, CmiNumCores(), CmiNumCores()-1);
      CmiAbort("Invalid core number");
    }
    if (CmiSetCPUAffinity(mycore) == -1) CmiAbort("set_cpu_affinity abort!");
    CmiNodeAllBarrier();
    CmiNodeAllBarrier();
    /* if (show_affinity_flag) CmiPrintCPUAffinity(); */
    return;
  }

#if CMK_CRAYXT || CMK_CRAYXE || CMK_CRAYXC
  {
    int numCores = CmiNumCores();

    int myid = getXTNodeID(CmiMyNodeGlobal(), CmiNumNodesGlobal());
    int myrank;
    int pe, mype = CmiMyPeGlobal();
    int node = CmiMyNodeGlobal();
    int nnodes = 0;
#if CMK_SMP
    if (CmiMyPe() >= CmiNumPes()) {         /* this is comm thread */
      int node = CmiMyPe() - CmiNumPes();
      mype = CmiGetPeGlobal(CmiNodeFirst(node) + CmiMyNodeSize() - 1, CmiMyPartition()); /* last pe on SMP node */
      node = CmiGetNodeGlobal(node, CmiMyPartition());
    }
#endif
    pe = mype - 1;
    while (pe >= 0) {
      int n = CmiNodeOf(pe);
      if (n != node) { nnodes++; node = n; }
      if (getXTNodeID(n, CmiNumNodesGlobal()) != myid) break;
      pe --;
    }
    CmiAssert(numCores > 0);
    myrank = (mype - pe - 1 + nnodes)%numCores;
#if CMK_SMP
    if (CmiMyPe() >= CmiNumPes()) 
        myrank = (myrank + 1)%numCores;
#endif

    if (-1 != CmiSetCPUAffinity(myrank)) {
      DEBUGP(("Processor %d is bound to core #%d on node #%d\n", CmiMyPe(), myrank, mynode));
    }
    else{
      CmiPrintf("Processor %d set affinity failed!\n", CmiMyPe());
      CmiAbort("set cpu affinity abort!\n");
    }
  }
  if (CmiMyPe() < CmiNumPes()) 
  CmiNodeAllBarrier();
  CmiNodeAllBarrier();
#elif CMK_SMP && CMK_PAMI_LINUX_PPC8
#define CMK_PAMI_LINUX_PPC8_CORES_PER_NODE      20
#define CMK_PAMI_LINUX_PPC8_THREADS_PER_CORE     8
#define CMK_PAMI_LINUX_PPC8_SKIP_CORE_0          0
  int cores_per_node = CMK_PAMI_LINUX_PPC8_CORES_PER_NODE;
  int threads_per_core = CMK_PAMI_LINUX_PPC8_THREADS_PER_CORE;

  CmiGetArgInt(argv,"+cores_per_node", &cores_per_node);
  CmiGetArgInt(argv,"+threads_per_core", &threads_per_core);

  int my_core   = CmiMyPe() % cores_per_node;
  int my_core_2 = CmiMyPe() % (cores_per_node/2);
#if CMK_PAMI_LINUX_PPC8_SKIP_CORE_0
  my_core_2 = (my_core_2 + 1) % (CMK_PAMI_LINUX_PPC8_CORES_PER_NODE/2);
#endif

  int cpu = 0;
  if (my_core < (cores_per_node/2))
    cpu = my_core_2 * threads_per_core;
  else
    cpu = (my_core_2 + CMK_PAMI_LINUX_PPC8_CORES_PER_NODE/2) * threads_per_core;

  cpu_set_t cset;
  CPU_ZERO(&cset);
  CPU_SET(cpu, &cset);
  CPU_SET(cpu+1, &cset);
  if(sched_setaffinity(0, sizeof(cpu_set_t), &cset) < 0)
    perror("sched_setaffinity");

  CPU_ZERO(&cset);
  if (sched_getaffinity(0, sizeof(cset), &cset) < 0)
    perror("sched_getaffinity");

  sched_yield();
  if(CmiMyPe() == 0)
    printf("Setting default affinity\n");
  return;
#else
    /* get my ip address */
  if (CmiMyRank() == 0)
  {
#if CMK_HAS_GETHOSTNAME
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
#endif

  if (show_affinity_flag) CmiPrintCPUAffinity();
}

/* called in ConverseCommonInit to initialize basic variables */
void CmiInitCPUAffinityUtil(){
    char fname[64];
    CpvInitialize(int, myCPUAffToCore);
    CpvAccess(myCPUAffToCore) = -1;
#if CMK_OS_IS_LINUX
    CpvInitialize(void *, myProcStatFP);
    CmiLock(_smp_mutex);
#if CMK_SMP
    sprintf(fname, "/proc/%d/task/%d/stat", getpid(), syscall(SYS_gettid));
#else
    sprintf(fname, "/proc/%d/stat", getpid());
#endif
    CpvAccess(myProcStatFP) = (void *)fopen(fname, "r");
    CmiUnlock(_smp_mutex);
/*
    if(CmiMyPe()==0 && CpvAccess(myProcStatFP) == NULL){
        CmiPrintf("WARNING: ERROR IN OPENING FILE %s on PROC %d, CmiOnCore() SHOULDN'T BE CALLED\n", fname, CmiMyPe()); 
    }
*/
#endif
}

#else           /* not supporting affinity */

int CmiSetCPUAffinity(int mycore)
{
  return -1;
}

int CmiPrintCPUAffinity()
{
  CmiPrintf("Warning: CmiPrintCPUAffinity not supported.\n");
  return -1;
}

void CmiInitCPUAffinity(char **argv)
{
  char *pemap = NULL;
  char *pemapfile = NULL;
  char *commap = NULL;
  int excludecore = -1;
  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
						"set cpu affinity");
  while (CmiGetArgIntDesc(argv,"+excludecore",&excludecore, "avoid core when setting cpuaffinity"));
  CmiGetArgStringDesc(argv, "+pemap", &pemap, "define pe to core mapping");
  CmiGetArgStringDesc(argv, "+pemapfile", &pemapfile, "define pe to core mapping file");
  CmiGetArgStringDesc(argv, "+commap", &commap, "define comm threads to core mapping");
  if (affinity_flag && CmiMyPe()==0)
    CmiPrintf("sched_setaffinity() is not supported, +setcpuaffinity disabled.\n");
  if (excludecore != -1 && CmiMyPe()==0)
    CmiPrintf("sched_setaffinity() is not supported, +excludecore disabled.\n");
  if (pemap && CmiMyPe()==0)
    CmiPrintf("sched_setaffinity() is not supported, +pemap disabled.\n");
  if (pemapfile && CmiMyPe()==0)
    CmiPrintf("sched_setaffinity() is not supported, +pemapfile disabled.\n");
  if (commap && CmiMyPe()==0)
    CmiPrintf("sched_setaffinity() is not supported, +commap disabled.\n");
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
  printf("WARNING: CmiOnCore IS NOT SUPPORTED ON THIS PLATFORM\n");
  return -1;
}
#endif
