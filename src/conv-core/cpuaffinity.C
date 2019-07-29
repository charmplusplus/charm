
/*
 This scheme relies on using IP address to identify nodes and assigning 
cpu affinity.  

 when CMK_NO_SOCKETS, which is typically on cray xt3 and bluegene/L.
 There is no hostname for the compute nodes.
 *
 * last updated 3/20/2010   Gengbin Zheng
 * new options +pemap +commmap takes complex pattern of a list of cores
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "converse.h"
#include "sockRoutines.h"
#include "charm-api.h"
#include "hwloc.h"

#if CMK_USE_IBVERBS
#include <infiniband/verbs.h>
#include <hwloc/openfabrics-verbs.h>
#endif

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

CmiHwlocTopology CmiHwlocTopologyLocal;

void CmiInitHwlocTopology(void)
{
    hwloc_topology_t topology;
    int depth;

    /* Allocate and initialize topology object. */
    cmi_hwloc_topology_init(&topology);
    /* Perform the topology detection. */
    cmi_hwloc_topology_load(topology);

    // packages == sockets
    depth = cmi_hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
    CmiHwlocTopologyLocal.num_sockets = depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(topology, depth) : 1;
#if CMK_BLUEGENEQ
  // ignore BG/Q's reserved socket
  if (CmiHwlocTopologyLocal.num_sockets == 17)
    CmiHwlocTopologyLocal.num_sockets = 16;
#endif

    // cores
    depth = cmi_hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    CmiHwlocTopologyLocal.num_cores = depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(topology, depth) : 1;

    // PUs
    depth = cmi_hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    CmiHwlocTopologyLocal.num_pus = depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(topology, depth) : 1;

    cmi_hwloc_topology_destroy(topology);


    // Legacy: Determine the system's total PU count

    cmi_hwloc_topology_init(&topology);
    cmi_hwloc_topology_set_flags(topology, cmi_hwloc_topology_get_flags(topology) | HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM);
    cmi_hwloc_topology_load(topology);

    depth = cmi_hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    CmiHwlocTopologyLocal.total_num_pus = depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(topology, depth) : 1;

    cmi_hwloc_topology_destroy(topology);
}

#if CMK_HAS_SETAFFINITY || defined (_WIN32) || CMK_HAS_BINDPROCESSOR

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else
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

#ifndef _WIN32
static int affMsgsRecvd = 1;  // number of affinity messages received at PE0
static cpu_set_t core_usage;  // used to record union of CPUs used by every PE in physical node
static int aff_is_set = 0;
#endif

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

static int set_process_affinity(hwloc_topology_t topology, hwloc_cpuset_t cpuset)
{
#ifdef _WIN32
  HANDLE process = GetCurrentProcess();
#define PRINTF_PROCESS "p"
#else
  pid_t process = getpid();
#define PRINTF_PROCESS "d"
#endif

  if (cmi_hwloc_set_proc_cpubind(topology, process, cpuset, HWLOC_CPUBIND_PROCESS|HWLOC_CPUBIND_STRICT))
  {
    char *str;
    int error = errno;
    cmi_hwloc_bitmap_asprintf(&str, cpuset);
    CmiPrintf("HWLOC> Couldn't bind to cpuset %s: %s\n", str, strerror(error));
    free(str);
    return -1;
  }

#if CMK_CHARMDEBUG
  if (CmiPhysicalNodeID(CmiMyPe()) == 0)
  {
    char *str;
    cmi_hwloc_bitmap_asprintf(&str, cpuset);
    CmiPrintf("HWLOC> [%d] Process %" PRINTF_PROCESS " bound to cpuset: %s\n", CmiMyPe(), process, str);
    free(str);
  }
#endif

  return 0;
#undef PRINTF_PROCESS
}

#if CMK_SMP
static int set_thread_affinity(hwloc_topology_t topology, hwloc_cpuset_t cpuset)
{
#ifdef _WIN32
  HANDLE thread = GetCurrentThread();
#define PRINTF_THREAD "p"
#else
  pthread_t thread = pthread_self();
#define PRINTF_THREAD "lu"
#endif

  if (cmi_hwloc_set_thread_cpubind(topology, thread, cpuset, HWLOC_CPUBIND_THREAD|HWLOC_CPUBIND_STRICT))
  {
    char *str;
    int error = errno;
    cmi_hwloc_bitmap_asprintf(&str, cpuset);
    CmiPrintf("HWLOC> Couldn't bind to cpuset %s: %s\n", str, strerror(error));
    free(str);
    return -1;
  }

#if CMK_CHARMDEBUG
  if (CmiPhysicalNodeID(CmiMyPe()) == 0)
  {
    char *str;
    cmi_hwloc_bitmap_asprintf(&str, cpuset);
    CmiPrintf("HWLOC> [%d] Thread %" PRINTF_THREAD " bound to cpuset: %s\n", CmiMyPe(), thread, str);
    free(str);
  }
#endif

  return 0;
#undef PRINTF_THREAD
}
#endif


int CmiSetCPUAffinity(int mycore)
{
  int core = mycore;
  if (core < 0) {
    core = CmiNumCores() + core;
  }
  if (core < 0) {
    CmiError("Error: Invalid parameter to CmiSetCPUAffinity: %d\n", mycore);
    CmiAbort("CmiSetCPUAffinity failed!");
  }

  CpvAccess(myCPUAffToCore) = core;

  hwloc_topology_t topology;

  cmi_hwloc_topology_init(&topology);
  cmi_hwloc_topology_load(topology);

  hwloc_obj_t thread_obj = cmi_hwloc_get_pu_obj_by_os_index(topology, core);

  int result = -1;

  if (thread_obj != nullptr)
#if CMK_SMP
    result = set_thread_affinity(topology, thread_obj->cpuset);
#else
    result = set_process_affinity(topology, thread_obj->cpuset);
#endif

  cmi_hwloc_topology_destroy(topology);

  if (result == -1)
    CmiError("Error: CmiSetCPUAffinity failed to bind PE #%d to PU #%d.\n", CmiMyPe(), mycore);

  return result;
}

/* This implementation assumes the default x86 CPU mask size used by Linux */
/* For a large SMP machine, this code should be changed to use a variable sized   */
/* CPU affinity mask buffer instead, as the present code will fail beyond 32 CPUs */
int print_cpu_affinity(void) {
  hwloc_topology_t topology;
  // Allocate and initialize topology object.
  cmi_hwloc_topology_init(&topology);
  // Perform the topology detection.
  cmi_hwloc_topology_load(topology);

  hwloc_cpuset_t cpuset = cmi_hwloc_bitmap_alloc();
  // And try to bind ourself there. */
  if (cmi_hwloc_get_cpubind(topology, cpuset, 0)) {
    int error = errno;
    CmiPrintf("[%d] CPU affinity mask is unknown %s\n", CmiMyPe(), strerror(error));
    cmi_hwloc_bitmap_free(cpuset);
    cmi_hwloc_topology_destroy(topology);
    return -1;
  }

  char *str;
  cmi_hwloc_bitmap_asprintf(&str, cpuset);
  CmiPrintf("[%d] CPU affinity mask is %s\n", CmiMyPe(), str);
  free(str);
  cmi_hwloc_bitmap_free(cpuset);
  cmi_hwloc_topology_destroy(topology);
  return 0;
}

#if CMK_SMP
int print_thread_affinity(void) {
  hwloc_topology_t topology;
  // Allocate and initialize topology object.
  cmi_hwloc_topology_init(&topology);
  // Perform the topology detection.
  cmi_hwloc_topology_load(topology);

#ifdef _WIN32
  HANDLE thread = GetCurrentThread();
#else
  pthread_t thread = pthread_self();
#endif

  hwloc_cpuset_t cpuset = cmi_hwloc_bitmap_alloc();
  // And try to bind ourself there. */
//  if (cmi_hwloc_get_thread_cpubind(topology, thread, cpuset, HWLOC_CPUBIND_THREAD)) {
  if (cmi_hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD) == -1) {
    int error = errno;
    CmiPrintf("[%d] thread CPU affinity mask is unknown %s\n", CmiMyPe(), strerror(error));
    cmi_hwloc_bitmap_free(cpuset);
    cmi_hwloc_topology_destroy(topology);
    return -1;
  }

  char *str;
  cmi_hwloc_bitmap_asprintf(&str, cpuset);
  CmiPrintf("[%d] thread CPU affinity mask is %s\n", CmiMyPe(), str);
  free(str);
  cmi_hwloc_bitmap_free(cpuset);
  cmi_hwloc_topology_destroy(topology);
  return 0;

}
#endif

int CmiPrintCPUAffinity(void)
{
#if CMK_SMP
  return print_thread_affinity();
#else
  return print_cpu_affinity();
#endif
}

#ifndef _WIN32
int get_cpu_affinity(cpu_set_t *cpuset) {
  CPU_ZERO(cpuset);
  if (sched_getaffinity(0, sizeof(cpuset), cpuset) < 0) {
    perror("sched_getaffinity");
    return -1;
  }
  return 0;
}

#if CMK_SMP
int get_thread_affinity(cpu_set_t *cpuset) {
#if CMK_HAS_PTHREAD_SETAFFINITY
  CPU_ZERO(cpuset);
  if (errno = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), cpuset)) {
    perror("pthread_getaffinity");
    return -1;
  }
  return 0;
#else
  return -1;
#endif
}
#endif

int get_affinity(cpu_set_t *cpuset) {
#if CMK_SMP
  return get_thread_affinity(cpuset);
#else
  return get_cpu_affinity(cpuset);
#endif
}
#endif

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
  if (fp == NULL){
    printf("WARNING: CmiOnCore IS NOT SUPPORTED ON THIS PLATFORM\n");
    return -1;
  }
  fseek(fp, 0, SEEK_SET);
  for (n=0; n<TASK_CPU_POS; n++)  {
    if (fscanf(fp, "%127s", str) != 1) {
      CmiAbort("CPU affinity> reading from /proc/<PID>/[task/<TID>]/stat failed!");
    }
  }
  return atoi(str);
#else
  printf("WARNING: CmiOnCore IS NOT SUPPORTED ON THIS PLATFORM\n");
  return -1;
#endif
}


static int cpuAffinityHandlerIdx;
static int cpuAffinityRecvHandlerIdx;
static int cpuPhyNodeAffinityRecvHandlerIdx;

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

typedef struct _affMsg {
  char core[CmiMsgHeaderSizeBytes];
#ifndef _WIN32
  cpu_set_t affinity;
#endif
} affMsg;

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
  void *tmpm;
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
    DEBUGP(("Cpuaffinity> %d unique compute nodes detected! \n", CmmEntries(hostTable)));
    tag = CmmWildCard;
    while ((tmpm = CmmGet(hostTable, 1, &tag, &tag1))) CmiFree(tmpm);
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

  DEBUGP(("[%d %d] set to core #: %d\n", CmiMyNode(), CmiMyPe(), myrank));

  if (-1 != CmiSetCPUAffinity(myrank)) {
    DEBUGP(("Processor %d is bound to core #%d on node #%d\n", CmiMyPe(), myrank, mynode));
  }
  else{
    CmiAbort("CmiSetCPUAffinity failed!");
  }
  CmiFree(m);
}

/* called on first PE in physical node, receive affinity set from other PEs in phy node */
static void cpuPhyNodeAffinityRecvHandler(void *msg)
{
  affMsg *m = (affMsg *)msg;
#if !defined(_WIN32) && defined(CPU_OR)
  CPU_OR(&core_usage, &core_usage, &m->affinity);
  affMsgsRecvd++;
#endif
  CmiFree(m);
}

#if defined(_WIN32)
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

#if CMK_CRAYXE || CMK_CRAYXC
CLINKAGE int getXTNodeID(int mpirank, int nummpiranks);
#endif

/**
 * Check that there are not multiple PEs assigned to the same core.
 * If a pemap has been computed by this module (or passed by the user) this
 * function will print a warning if oversubscription detected. If no affinity
 * has been set explicitly by this module, it will print error and abort if
 * oversubscription detected.
 */
void CmiCheckAffinity(void)
{
#if !defined(_WIN32) && CMK_SMP && CMK_HAS_PTHREAD_SETAFFINITY && defined(CPU_OR)

  if (!CmiCpuTopologyEnabled()) return;  // only works if cpu topology enabled

  if (CmiMyPe() == 0) {
    // wait for every PE affinity from my physical node (for now only done on phy node 0)

    cpu_set_t my_aff;
    if (get_affinity(&my_aff) == -1) CmiAbort("get_affinity failed\n");
    CPU_OR(&core_usage, &core_usage, &my_aff); // add my affinity (pe0)
    int N = CmiNumPesOnPhysicalNode(0);
    while (affMsgsRecvd < N)
      CmiDeliverSpecificMsg(cpuPhyNodeAffinityRecvHandlerIdx);

    // NOTE this test is simple and may not detect every possible case of
    // oversubscription
    if (CPU_COUNT(&core_usage) < N) {
      // TODO suggest command line arguments?
      if (!aff_is_set) {
        CmiAbort("Multiple PEs assigned to same core. Set affinity "
        "options to correct or lower the number of threads, or pass +setcpuaffinity to ignore.\n");
      } else {
        CmiPrintf("WARNING: Multiple PEs assigned to same core, recommend "
        "adjusting processor affinity or passing +CmiSleepOnIdle to reduce "
        "interference.\n");
      }
    }
  } else if ((CmiMyPe() < CmiNumPes()) && (CmiPhysicalNodeID(CmiMyPe()) == 0)) {
    // send my affinity to first PE on physical node (only done on phy node 0 for now)
    affMsg *m = (affMsg*)CmiAlloc(sizeof(affMsg));
    CmiSetHandler((char *)m, cpuPhyNodeAffinityRecvHandlerIdx);
    if (get_affinity(&m->affinity) == -1) { // put my affinity in msg
      CmiFree(m);
      CmiAbort("get_affinity failed\n");
    }
    CmiSyncSendAndFree(0, sizeof(affMsg), (void *)m);
  }
#endif
}

extern int CmiMyLocalRank;

static void bind_process_only(hwloc_obj_type_t process_unit)
{
  hwloc_topology_t topology;
  hwloc_cpuset_t cpuset;
  cmi_hwloc_topology_init(&topology);
  cmi_hwloc_topology_load(topology);


  int process_unitcount = cmi_hwloc_get_nbobjs_by_type(topology, process_unit);
#if CMK_BLUEGENEQ
  // ignore BG/Q's reserved socket
  if (process_unit == HWLOC_OBJ_PACKAGE && process_unitcount == 17)
    process_unitcount = 16;
#endif

  int process_assignment = CmiMyLocalRank % process_unitcount;

  hwloc_obj_t process_obj = cmi_hwloc_get_obj_by_type(topology, process_unit, process_assignment);
  set_process_affinity(topology, process_obj->cpuset);


  cmi_hwloc_topology_destroy(topology);
}

#if CMK_SMP
static void bind_threads_only(hwloc_obj_type_t thread_unit)
{
  hwloc_topology_t topology;
  hwloc_cpuset_t cpuset;
  cmi_hwloc_topology_init(&topology);
  cmi_hwloc_topology_load(topology);


  int thread_unitcount = cmi_hwloc_get_nbobjs_by_type(topology, thread_unit);
#if CMK_BLUEGENEQ
  // ignore BG/Q's reserved socket
  if (thread_unit == HWLOC_OBJ_PACKAGE && thread_unitcount == 17)
    thread_unitcount = 16;
#endif

  int thread_assignment = CmiMyRank() % thread_unitcount;

  hwloc_obj_t thread_obj = cmi_hwloc_get_obj_by_type(topology, thread_unit, thread_assignment);
  hwloc_cpuset_t thread_cpuset = cmi_hwloc_bitmap_dup(thread_obj->cpuset);
  cmi_hwloc_bitmap_singlify(thread_cpuset);
  set_thread_affinity(topology, thread_cpuset);
  cmi_hwloc_bitmap_free(thread_cpuset);


  cmi_hwloc_topology_destroy(topology);
}

static void bind_process_and_threads(hwloc_obj_type_t process_unit, hwloc_obj_type_t thread_unit)
{
  hwloc_topology_t topology;
  hwloc_cpuset_t cpuset;
  cmi_hwloc_topology_init(&topology);
  cmi_hwloc_topology_load(topology);


  int process_unitcount = cmi_hwloc_get_nbobjs_by_type(topology, process_unit);

  int process_assignment = CmiMyLocalRank % process_unitcount;

  hwloc_obj_t process_obj = cmi_hwloc_get_obj_by_type(topology, process_unit, process_assignment);
  set_process_affinity(topology, process_obj->cpuset);

  int thread_unitcount = cmi_hwloc_get_nbobjs_inside_cpuset_by_type(topology, process_obj->cpuset, thread_unit);

  int thread_assignment = CmiMyRank() % thread_unitcount;

  hwloc_obj_t thread_obj = cmi_hwloc_get_obj_inside_cpuset_by_type(topology, process_obj->cpuset, thread_unit, thread_assignment);
  hwloc_cpuset_t thread_cpuset = cmi_hwloc_bitmap_dup(thread_obj->cpuset);
  cmi_hwloc_bitmap_singlify(thread_cpuset);
  set_thread_affinity(topology, thread_cpuset);
  cmi_hwloc_bitmap_free(thread_cpuset);


  cmi_hwloc_topology_destroy(topology);
}
#endif

static int set_default_affinity(void)
{
  char *s;
  int n = -1;

  if ((s = getenv("CmiProcessPerSocket")))
  {
    n = atoi(s);
#if CMK_SMP
    if (getenv("CmiOneWthPerCore"))
      bind_process_and_threads(HWLOC_OBJ_PACKAGE, HWLOC_OBJ_CORE);
    else if (getenv("CmiOneWthPerPU"))
      bind_process_and_threads(HWLOC_OBJ_PACKAGE, HWLOC_OBJ_PU);
    else
#endif
      bind_process_only(HWLOC_OBJ_PACKAGE);
  }
  else if ((s = getenv("CmiProcessPerCore")))
  {
    n = atoi(s);
#if CMK_SMP
    if (getenv("CmiOneWthPerPU"))
      bind_process_and_threads(HWLOC_OBJ_CORE, HWLOC_OBJ_PU);
    else
#endif
      bind_process_only(HWLOC_OBJ_CORE);
  }
  else if ((s = getenv("CmiProcessPerPU")))
  {
    n = atoi(s);
    bind_process_only(HWLOC_OBJ_PU);
  }
  else // if ((s = getenv("CmiProcessPerHost")))
  {
#if CMK_SMP
    if (getenv("CmiOneWthPerSocket"))
    {
      n = 0;
      bind_threads_only(HWLOC_OBJ_PACKAGE);
    }
    else if (getenv("CmiOneWthPerCore"))
    {
      n = 0;
      bind_threads_only(HWLOC_OBJ_CORE);
    }
    else if (getenv("CmiOneWthPerPU"))
    {
      n = 0;
      bind_threads_only(HWLOC_OBJ_PU);
    }
#endif
  }

  return n != -1;
}

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

  show_affinity_flag = CmiGetArgFlagDesc(argv,"+showcpuaffinity", "print cpu affinity");

  CmiAssignOnce(&cpuAffinityHandlerIdx, CmiRegisterHandler((CmiHandler)cpuAffinityHandler));
  CmiAssignOnce(&cpuAffinityRecvHandlerIdx, CmiRegisterHandler((CmiHandler)cpuAffinityRecvHandler));
  CmiAssignOnce(&cpuPhyNodeAffinityRecvHandlerIdx, CmiRegisterHandler((CmiHandler)cpuPhyNodeAffinityRecvHandler));

  /* new style */
  {
      int done = 0;
      CmiNodeAllBarrier();

      /* must bind the rank 0 which is the main thread first */
      /* binding the main thread seems to change binding for all threads */
      if (CmiMyRank() == 0) {
          done = set_default_affinity();
      }

      CmiNodeAllBarrier();

      if (CmiMyRank() != 0) {
          done = set_default_affinity();
      }

      if (done) {
          if (show_affinity_flag) CmiPrintCPUAffinity();
          return;
      }
  }

  if (CmiMyRank() ==0) {
     affLock = CmiCreateLock();
#ifndef _WIN32
     aff_is_set = affinity_flag;
     CPU_ZERO(&core_usage);
#endif
  }

#if CMK_BLUEGENEQ
  if(affinity_flag){
      affinity_flag = 0;
      if(CmiMyPe()==0) CmiPrintf("Charm++> cpu affinity setting is not needed on Blue Gene/Q, thus ignored.\n");
  }
  if(show_affinity_flag){
      show_affinity_flag = 0;
      if(CmiMyPe()==0) CmiPrintf("Charm++> printing cpu affinity is not supported on Blue Gene/Q.\n");
  }
#endif

  if (!affinity_flag) {
    if (show_affinity_flag) {
      CmiPrintCPUAffinity();
      CmiPrintf("Charm++> cpu affinity NOT enabled.\n");
    }
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
      if (CmiPhysicalNodeID(CmiMyPe()) == 0) CmiPrintf("Charm++> set comm %d on node %d to core #%d\n", CmiMyPe()-CmiNumPes(), CmiMyNode(), mycore);
      if (-1 == CmiSetCPUAffinity(mycore))
        CmiAbort("CmiSetCPUAffinity failed!");
      CmiNodeAllBarrier();
      if (show_affinity_flag) CmiPrintCPUAffinity();
      return;    /* comm thread return */
    }
    else {
    /* if (CmiSetCPUAffinity(CmiNumCores()-1) == -1) CmiAbort("CmiSetCPUAffinity failed!"); */
#if !CMK_CRAYXE && !CMK_CRAYXC && !CMK_BLUEGENEQ && !CMK_PAMI_LINUX_PPC8
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
#if CMK_CRAYXE || CMK_CRAYXC
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
    if (CmiSetCPUAffinity(mycore) == -1) CmiAbort("CmiSetCPUAffinity failed!");
    CmiNodeAllBarrier();
    CmiNodeAllBarrier();
    /* if (show_affinity_flag) CmiPrintCPUAffinity(); */
    return;
  }

#if CMK_CRAYXE || CMK_CRAYXC
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
      CmiAbort("CmiSetCPUAffinity failed!");
    }
  }
  if (CmiMyPe() < CmiNumPes()) 
  CmiNodeAllBarrier();
  CmiNodeAllBarrier();
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
void CmiInitCPUAffinityUtil(void){
    char fname[64];
    CpvInitialize(int, myCPUAffToCore);
    CpvAccess(myCPUAffToCore) = -1;
#if CMK_OS_IS_LINUX
    CpvInitialize(void *, myProcStatFP);
    CmiLock(_smp_mutex);
#if CMK_SMP
    sprintf(fname, "/proc/%d/task/%ld/stat", getpid(), syscall(SYS_gettid));
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

int CmiPrintCPUAffinity(void)
{
  CmiPrintf("Warning: CmiPrintCPUAffinity not supported.\n");
  return -1;
}

void CmiCheckAffinity(void) {
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
  CmiGetArgFlagDesc(argv,"+showcpuaffinity", "print cpu affinity");
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
void CmiInitCPUAffinityUtil(void){
    CpvInitialize(int, myCPUAffToCore);
    CpvAccess(myCPUAffToCore) = -1;
#if CMK_OS_IS_LINUX	
    CpvInitialize(void *, myProcStatFP);
    CpvAccess(myProcStatFP) = NULL;
 #endif
}

int CmiOnCore(void){
  printf("WARNING: CmiOnCore IS NOT SUPPORTED ON THIS PLATFORM\n");
  return -1;
}
#endif
