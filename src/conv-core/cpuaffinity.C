
/*
 This scheme relies on using IP address to identify hosts and assign
 cpu affinity.

 When CMK_NO_SOCKETS, which is typically on cray xt3 and bluegene/L,
 there is no hostname.
 *
 * last updated 3/20/2010   Gengbin Zheng
 * new options +pemap +commmap takes complex pattern of a list of cores
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <vector>
#include <queue>
#include <map>
#include <algorithm>

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

// topology is the set of resources available to this process
// legacy_topology includes resources disallowed by the system, to implement CmiNumCores
static hwloc_topology_t topology, legacy_topology;

void CmiInitHwlocTopology(void)
{
    int depth;

    cmi_hwloc_topology_init(&topology);
    cmi_hwloc_topology_load(topology);

    // packages == sockets
    depth = cmi_hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
    CmiHwlocTopologyLocal.num_sockets = depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(topology, depth) : 1;

    // cores
    depth = cmi_hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    CmiHwlocTopologyLocal.num_cores = depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(topology, depth) : 1;

    // PUs
    depth = cmi_hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    CmiHwlocTopologyLocal.num_pus = depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(topology, depth) : 1;


    // Legacy: Determine the system's total PU count

    cmi_hwloc_topology_init(&legacy_topology);
    cmi_hwloc_topology_set_flags(legacy_topology, cmi_hwloc_topology_get_flags(legacy_topology) | HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED);
    cmi_hwloc_topology_load(legacy_topology);

    depth = cmi_hwloc_get_type_depth(legacy_topology, HWLOC_OBJ_PU);
    CmiHwlocTopologyLocal.total_num_pus =
      depth != HWLOC_TYPE_DEPTH_UNKNOWN ? cmi_hwloc_get_nbobjs_by_depth(legacy_topology, depth) : 1;
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

static int set_process_affinity(hwloc_cpuset_t cpuset)
{
#ifdef _WIN32
  HANDLE process = GetCurrentProcess();
# define PRINTF_PROCESS "%p"
#else
  pid_t process = getpid();
# define PRINTF_PROCESS "%d"
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
    CmiPrintf("HWLOC> [%d] Process " PRINTF_PROCESS " bound to cpuset: %s\n", CmiMyPe(), process, str);
    free(str);
  }
#endif

  return 0;

#undef PRINTF_PROCESS
}

#if CMK_SMP
static int set_thread_affinity(hwloc_cpuset_t cpuset)
{
#ifdef _WIN32
  HANDLE thread = GetCurrentThread();
#else
  pthread_t thread = pthread_self();
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
    CmiPrintf("HWLOC> [%d] Thread %p bound to cpuset: %s\n", CmiMyPe(), (const void *)thread, str);
    free(str);
  }
#endif

  return 0;
}
#endif


// Uses PU indices assigned by the OS
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

  hwloc_obj_t thread_obj = cmi_hwloc_get_pu_obj_by_os_index(topology, core);

  int result = -1;

  if (thread_obj != nullptr)
#if CMK_SMP
    result = set_thread_affinity(thread_obj->cpuset);
#else
    result = set_process_affinity(thread_obj->cpuset);
#endif

  if (result == -1)
    CmiError("Error: CmiSetCPUAffinity failed to bind PE #%d to PU P#%d.\n", CmiMyPe(), mycore);

  return result;
}

// Uses logical PU indices as determined by hwloc
int CmiSetCPUAffinityLogical(int mycore)
{
  int core = mycore;
  if (core < 0) {
    core = CmiHwlocTopologyLocal.num_pus + core;
  }
  if (core < 0) {
    CmiError("Error: Invalid parameter to CmiSetCPUAffinityLogical: %d\n", mycore);
    CmiAbort("CmiSetCPUAffinityLogical failed!");
  }

  int thread_unitcount = cmi_hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
  int thread_assignment = core % thread_unitcount;

  hwloc_obj_t thread_obj = cmi_hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, thread_assignment);

  int result = -1;

  if (thread_obj != nullptr)
  {
#if CMK_SMP
    result = set_thread_affinity(thread_obj->cpuset);
#else
    result = set_process_affinity(thread_obj->cpuset);
#endif

    CpvAccess(myCPUAffToCore) = thread_obj->os_index;
  }

  if (result == -1)
    CmiError("Error: CmiSetCPUAffinityLogical failed to bind PE #%d to PU L#%d.\n", CmiMyPe(), mycore);

  return result;
}

/* This implementation assumes the default x86 CPU mask size used by Linux */
/* For a large SMP machine, this code should be changed to use a variable sized   */
/* CPU affinity mask buffer instead, as the present code will fail beyond 32 CPUs */
int print_cpu_affinity(void) {
  hwloc_cpuset_t cpuset = cmi_hwloc_bitmap_alloc();
  // And try to bind ourself there. */
  if (cmi_hwloc_get_cpubind(topology, cpuset, 0)) {
    int error = errno;
    CmiPrintf("[%d] CPU affinity mask is unknown %s\n", CmiMyPe(), strerror(error));
    cmi_hwloc_bitmap_free(cpuset);
    return -1;
  }

  char *str;
  cmi_hwloc_bitmap_asprintf(&str, cpuset);
  CmiPrintf("[%d] CPU affinity mask is %s\n", CmiMyPe(), str);
  free(str);
  cmi_hwloc_bitmap_free(cpuset);
  return 0;
}

#if CMK_SMP
int print_thread_affinity(void) {
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
    return -1;
  }

  char *str;
  cmi_hwloc_bitmap_asprintf(&str, cpuset);
  CmiPrintf("[%d] thread CPU affinity mask is %s\n", CmiMyPe(), str);
  free(str);
  cmi_hwloc_bitmap_free(cpuset);
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
  if ((errno = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), cpuset))) {
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

struct hostnameMsg {
  char core[CmiMsgHeaderSizeBytes];
  int node;
  int pe;
  skt_ip_t ip;
  int seq;
};

struct rankMsg {
  char core[CmiMsgHeaderSizeBytes];
  int *node_ranks_on_host;        /* node => core rank mapping */
  int *node_hosts;                /* node => host number mapping */
  int *node_numpes;               /* node => number of PEs on node */
  int *pe_ranks_on_host;          /* PE => core rank mapping */
  int *pe_hosts;                  /* PE => host number mapping */

  static constexpr size_t size(int numnodes, int numpes)
  {
    return sizeof(rankMsg) + numnodes*sizeof(int)*3 + numpes*sizeof(int)*2;
  }
};

struct affMsg {
  char core[CmiMsgHeaderSizeBytes];
#ifndef _WIN32
  cpu_set_t affinity;
#endif
};

static rankMsg *rankmsg = NULL;
static std::map<skt_ip_t, hostnameMsg *> hostTable;

static std::atomic<bool> cpuAffSyncHandlerDone{};
static std::atomic<bool> cpuAffSyncBroadcastDone{};
static std::atomic<bool> cpuPhyAffCheckDone{};
#if CMK_SMP && !CMK_SMP_NO_COMMTHD
extern void CommunicationServerThread(int sleepTime);
static std::atomic<bool> cpuAffSyncCommThreadDone{};
static std::atomic<bool> cpuPhyAffCheckCommThreadDone{};
#endif

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
static void cpuAffSyncWaitCommThread(std::atomic<bool> & done)
{
  do
    CommunicationServerThread(5);
  while (!done.load());

  CommunicationServerThread(5);
}
#endif

static void cpuAffSyncWait(std::atomic<bool> & done)
{
  do
    CsdSchedulePoll();
  while (!done.load());

  CsdSchedulePoll();
}

/* called on PE 0 */
static void cpuAffinityHandler(void *m)
{
  static int count = 0;
  static int hostcount = 0;
  hostnameMsg *rec;
  hostnameMsg *msg = (hostnameMsg *)m;
  void *tmpm;
  const int npes = CmiNumPes();

/*   for debug
  char str[128];
  skt_print_ip(str, msg->ip);
  printf("hostname: %d %s\n", msg->pe, str);
*/
  CmiAssert(CmiMyPe()==0 && rankmsg != NULL);
  skt_ip_t & ip = msg->ip;
  int node = msg->node;
  int pe = msg->pe;
  auto iter = hostTable.find(ip);
  if (iter != hostTable.end()) {
    rec = iter->second;
    CmiFree(msg);
  }
  else {
    rec = msg;
    rec->seq = hostcount++;
    hostTable.emplace(ip, msg);
  }

  int & node_host = rankmsg->node_hosts[node];
  if (node_host == -1)
  {
    node_host = rec->seq;
  }
  rankmsg->node_numpes[node]++;
  rankmsg->pe_hosts[pe] = rec->seq;

  if (++count == npes)
    cpuAffSyncHandlerDone = true;
}

/*
 * Distributes PUs among PEs in a multi-level round robin fashion.
 *
 * For example, a machine with two sockets x two cores x two PUs,
 *           M0
 *     S0          S1
 *  C0    C1    C2    C3
 * 0  1  2  3  4  5  6  7
 * will result in: 0 4 2 6 1 5 3 7
 *
 * With a single socket containing four cores instead,
 *           M0
 *           S0
 *  C0    C1    C2    C3
 * 0  1  2  3  4  5  6  7
 * the output is: 0 2 4 6 1 3 5 7
 *
 * The function takes a parameter to limit the count of PUs returned.
 * With the single socket example and +p5 in argv, the function returns: 0 2 4 6 1
 * The caller then sorts this list so that PE number locality implies PU locality,
 * while still spreading the work units as much as possible: 0 1 2 4 6
 * With +p8, the result would be: 0 1 2 3 4 5 6 7
 *
 * The +excludecore command line parameter is also respected.
 */
static inline std::vector<hwloc_obj_t> getPUListForAutoAffinity(hwloc_obj_t container, unsigned int numranks)
{
  std::vector<hwloc_obj_t> pu_list;
  std::map<int, std::queue<std::pair<hwloc_obj_t, unsigned int>>> depthqueues;

  depthqueues[container->depth].emplace(container, 0);

  do
  {
    auto & thisqueue = depthqueues.begin()->second;

    // Loop through the current depth level, pushing children into queues for their depth level.
    do
    {
      auto entry = thisqueue.front();
      thisqueue.pop();

      hwloc_obj_t obj = entry.first;
      unsigned int childindex = entry.second;

      if (obj->type == HWLOC_OBJ_PU && !in_exclude(obj->os_index))
      {
        // Add this PU to the list.
        pu_list.push_back(obj);

        // Not just an optimization, the algorithm is pointless without this check and return.
        if (pu_list.size() == numranks)
          return pu_list;
      }

      if (childindex < obj->arity)
      {
        // Push one child at a time into its queue.
        depthqueues[obj->depth].emplace(obj->children[childindex++], 0);

        // If this node has more children, re-add it to the end of the current queue.
        if (childindex < obj->arity)
          thisqueue.emplace(obj, childindex);
      }
    }
    while (!thisqueue.empty());

    // Proceed to the next depth level.
    depthqueues.erase(depthqueues.begin());
  }
  while (!depthqueues.empty());

  return pu_list;
}

/* called on each processor */
static void cpuAffinityRecvHandler(void *msg)
{
  const int mynode = CmiMyNode();
  const int mype = CmiMyPe();
  const int numnodes = CmiNumNodes();
  const int numpes = CmiNumPes();

  rankMsg *m = (rankMsg *)msg;
  size_t siz = sizeof(rankMsg);
  m->node_ranks_on_host = (int *)((char*)m + siz);
  siz += numnodes*sizeof(int);
  m->node_hosts = (int *)((char*)m + siz);
  siz += numnodes*sizeof(int);
  m->node_numpes = (int *)((char*)m + siz);
  siz += numnodes*sizeof(int);
  m->pe_ranks_on_host = (int *)((char*)m + siz);
  siz += numpes*sizeof(int);
  m->pe_hosts = (int *)((char*)m + siz);
  siz += numpes*sizeof(int);
  CmiAssert(siz == rankMsg::size(numnodes, numpes));

  int node_rank = m->node_ranks_on_host[mynode];
  int pe_rank = m->pe_ranks_on_host[mype];
  int myhost = m->pe_hosts[mype];

  int nodes_on_host = 0;
  for (int i = 0; i < numnodes; ++i)
    if (m->node_hosts[i] == myhost)
      ++nodes_on_host;

  // Filter out packages with null cpusets, caused by system resource management
  const int total_package_count = cmi_hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PACKAGE);
  std::vector<hwloc_obj_t> packages;
  packages.reserve(total_package_count);
  for (int i = 0; i < total_package_count; ++i)
  {
    hwloc_obj_t obj = cmi_hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, i);
    if (cmi_hwloc_get_nbobjs_inside_cpuset_by_type(topology, obj->cpuset, HWLOC_OBJ_PU) > 0)
      packages.emplace_back(obj);
  }
  const int package_count = packages.size();

  hwloc_obj_t obj;
  int numranks, myrank;

  if (nodes_on_host > 1 && package_count > 1)
  {
    // If multiple processes are running on a machine with multiple packages, divide the
    // packages among the processes, and bind each process to its package.

    const int mypackage = node_rank % package_count;

    // Count the number of PEs across all logical nodes sharing a package (node_count > package_count).
    int my_first_pe_rank_on_package = 0;
    int pes_on_package = 0;
    for (int i = 0; i < numnodes; ++i)
    {
      if (m->node_hosts[i] == myhost && m->node_ranks_on_host[i] % package_count == mypackage)
      {
        if (i == mynode)
          my_first_pe_rank_on_package = pes_on_package;
        pes_on_package += m->node_numpes[i];
      }
    }

    obj = packages[mypackage];
    numranks = pes_on_package;
    myrank = my_first_pe_rank_on_package + CmiMyRank();

    // Set the process binding to be safe.
    // Also binds the comm thread at least to the package, since it does not execute cpuAffinityRecvHandler.
    if (CmiMyRank() == 0)
      set_process_affinity(obj->cpuset);

    // Ensure no thread sets affinity before the process call, or else the thread call will be overwritten.
    CmiNodeBarrier();
  }
  else
  {
    // Otherwise spread out among the entire machine.

    // Count the number of PEs sharing the machine.
    int pes_on_host = 0;
    for (int i = 0; i < numpes; ++i)
      if (m->pe_hosts[i] == myhost)
        ++pes_on_host;

    obj = cmi_hwloc_get_root_obj(topology);
    numranks = pes_on_host;
    myrank = pe_rank;
  }

  std::vector<hwloc_obj_t> pu_list = getPUListForAutoAffinity(obj, numranks);

  if (pu_list.size() == 0)
    CmiAbort("All eligible PUs have been excluded!");

  std::sort(pu_list.begin(), pu_list.end(),
            [](hwloc_obj_t a, hwloc_obj_t b) -> bool { return a->logical_index < b->logical_index; });

  const int pu = pu_list[myrank % pu_list.size()]->logical_index;

  DEBUGP(("[%d %d] (%d %d) assigning to PU L#%d\n", mynode, mype, node_rank, pe_rank, pu));

  if (-1 != CmiSetCPUAffinityLogical(pu)) {
    DEBUGP(("Processor %d is bound to PU L#%d on host #%d\n", mype, pu, myhost));
  }
  else{
    CmiAbort("CmiSetCPUAffinity failed!");
  }
  CmiFree(m);

  CmiNodeBarrier();

  if (CmiMyRank() == 0)
    cpuAffSyncBroadcastDone = true;
}

/* called on first PE in physical node, receive affinity set from other PEs in phy node */
static void cpuPhyNodeAffinityRecvHandler(void *msg)
{
  static int count = 0;

  affMsg *m = (affMsg *)msg;
#if !defined(_WIN32) && defined(CPU_OR)
  CPU_OR(&core_usage, &core_usage, &m->affinity);
  affMsgsRecvd++;
#endif
  CmiFree(m);

  if (++count == CmiNumPesOnPhysicalNode(0) - 1)
    cpuPhyAffCheckDone = true;
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

  if (CmiNumPes() == 1)
    return;

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
  if (CmiInCommThread())
  {
    cpuAffSyncWaitCommThread(cpuPhyAffCheckCommThreadDone);
  }
  else
#endif
  if (CmiMyPe() == 0)
  {
    // wait for every PE affinity from my physical node (for now only done on phy node 0)

    cpu_set_t my_aff;
    if (get_affinity(&my_aff) == -1) CmiAbort("get_affinity failed\n");
    CPU_OR(&core_usage, &core_usage, &my_aff); // add my affinity (pe0)

    cpuAffSyncWait(cpuPhyAffCheckDone);

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
    CmiNodeBarrier();

    cpuPhyAffCheckCommThreadDone = true;
#endif
  }
  else if (CmiPhysicalNodeID(CmiMyPe()) == 0)
  {
    // send my affinity to first PE on physical node (only done on phy node 0 for now)
    affMsg *m = (affMsg*)CmiAlloc(sizeof(affMsg));
    CmiSetHandler((char *)m, cpuPhyNodeAffinityRecvHandlerIdx);
    if (get_affinity(&m->affinity) == -1) { // put my affinity in msg
      CmiFree(m);
      CmiAbort("get_affinity failed\n");
    }
    CmiSyncSendAndFree(0, sizeof(affMsg), (void *)m);

    CsdSchedulePoll();

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
    CmiNodeBarrier();

    if (CmiMyRank() == 0)
      cpuPhyAffCheckCommThreadDone = true;
#endif
  }
#if CMK_SMP && !CMK_SMP_NO_COMMTHD
  else if (CmiMyRank() == 0)
  {
    cpuPhyAffCheckCommThreadDone = true;
  }
#endif

  CmiBarrier();

  if (CmiMyPe() == 0)
  {
    // NOTE this test is simple and may not detect every possible case of
    // oversubscription
    const int N = CmiNumPesOnPhysicalNode(0);
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
  }
#endif
}

extern int CmiMyLocalRank;

static void bind_process_only(hwloc_obj_type_t process_unit)
{
  hwloc_cpuset_t cpuset;

  int process_unitcount = cmi_hwloc_get_nbobjs_by_type(topology, process_unit);

  int process_assignment = CmiMyLocalRank % process_unitcount;

  hwloc_obj_t process_obj = cmi_hwloc_get_obj_by_type(topology, process_unit, process_assignment);
  set_process_affinity(process_obj->cpuset);
}

#if CMK_SMP
static void bind_threads_only(hwloc_obj_type_t thread_unit)
{
  hwloc_cpuset_t cpuset;

  int thread_unitcount = cmi_hwloc_get_nbobjs_by_type(topology, thread_unit);

  int thread_assignment = CmiMyRank() % thread_unitcount;

  hwloc_obj_t thread_obj = cmi_hwloc_get_obj_by_type(topology, thread_unit, thread_assignment);
  hwloc_cpuset_t thread_cpuset = cmi_hwloc_bitmap_dup(thread_obj->cpuset);
  cmi_hwloc_bitmap_singlify(thread_cpuset);
  set_thread_affinity(thread_cpuset);
  cmi_hwloc_bitmap_free(thread_cpuset);
}

static void bind_process_and_threads(hwloc_obj_type_t process_unit, hwloc_obj_type_t thread_unit)
{
  hwloc_cpuset_t cpuset;

  int process_unitcount = cmi_hwloc_get_nbobjs_by_type(topology, process_unit);

  int process_assignment = CmiMyLocalRank % process_unitcount;

  hwloc_obj_t process_obj = cmi_hwloc_get_obj_by_type(topology, process_unit, process_assignment);
  set_process_affinity(process_obj->cpuset);

  int thread_unitcount = cmi_hwloc_get_nbobjs_inside_cpuset_by_type(topology, process_obj->cpuset, thread_unit);

  int thread_assignment = CmiMyRank() % thread_unitcount;

  hwloc_obj_t thread_obj = cmi_hwloc_get_obj_inside_cpuset_by_type(topology, process_obj->cpuset, thread_unit, thread_assignment);
  hwloc_cpuset_t thread_cpuset = cmi_hwloc_bitmap_dup(thread_obj->cpuset);
  cmi_hwloc_bitmap_singlify(thread_cpuset);
  set_thread_affinity(thread_cpuset);
  cmi_hwloc_bitmap_free(thread_cpuset);
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

// Check if provided mapping string uses logical indices as assigned by hwloc.
// Logical indices are used if the first character of the map string is an L
// (case-insensitive).
static int check_logical_indices(char **mapptr) {
  if ((*mapptr)[0] == 'l' || (*mapptr)[0] == 'L') {
    (*mapptr)++; // Exclude the L character from the string
    return 1;
  }

  return 0;
}

void CmiInitCPUAffinity(char **argv)
{
  int ret, i, exclude;
  hostnameMsg  *msg;
  char *pemap = NULL;
  char *commap = NULL;
  char *pemapfile = NULL;

  int show_affinity_flag;

  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
                                               "set cpu affinity");
  int pemap_logical_flag = 0;
  int commap_logical_flag = 0;

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

  // Check if provided pemap and/or commap use OS indices or logical indices
  if (pemap != NULL) pemap_logical_flag = check_logical_indices(&pemap);
  if (commap != NULL) commap_logical_flag = check_logical_indices(&commap);

  // Issue warning if pemap and commap do not use the same type of indices
  if (pemap != NULL && commap != NULL && pemap_logical_flag != commap_logical_flag) {
    if (CmiMyPe() == 0) {
      CmiPrintf("WARNING: Different types of indices are used for +pemap and +commap.\n");
    }
  }

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
#ifndef _WIN32
     aff_is_set = affinity_flag;
     CPU_ZERO(&core_usage);
#endif
  }

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
       CmiPrintf("Charm++> cpuaffinity PE-core map (%s): %s\n",
           pemap_logical_flag ? "logical indices" : "OS indices", pemap);
  }

  if (pemap != NULL)
  {
#if CMK_SMP && !CMK_SMP_NO_COMMTHD
    if (!CmiInCommThread())
#endif
    {
      int mycore = search_pemap(pemap, CmiMyPeGlobal());
      if (pemap_logical_flag) {
        if (CmiSetCPUAffinityLogical(mycore) == -1) CmiAbort("CmiSetCPUAffinityLogical failed");
      }
      else {
        if (CmiSetCPUAffinity(mycore) == -1) CmiAbort("CmiSetCPUAffinity failed!");
      }
      if (show_affinity_flag) {
        CmiPrintf("Charm++> set PE %d on node %d to PU %c#%d\n", CmiMyPe(), CmiMyNode(),
            pemap_logical_flag ? 'L' : 'P', mycore);
      }
    }
  }
  else
  {
#if CMK_CRAYXE || CMK_CRAYXC
    int numCores = CmiNumCores();

    int myid = getXTNodeID(CmiMyNodeGlobal(), CmiNumNodesGlobal());
    int myrank;
    int pe, mype = CmiMyPeGlobal();
    int node = CmiMyNodeGlobal();
    int nnodes = 0;
#if CMK_SMP && !CMK_SMP_NO_COMMTHD
    if (CmiInCommThread()) {
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
#if CMK_SMP && !CMK_SMP_NO_COMMTHD
    if (CmiInCommThread())
        myrank = (myrank + 1)%numCores;
#endif

    if (-1 != CmiSetCPUAffinity(myrank)) {
      DEBUGP(("Processor %d is bound to core #%d on node #%d\n", CmiMyPe(), myrank, mynode));
    }
    else{
      CmiAbort("CmiSetCPUAffinity failed!");
    }

    CmiNodeAllBarrier();
#else
    /* get my ip address */
    static skt_ip_t myip;
    if (CmiMyRank() == 0)
    {
      myip = skt_my_ip();        /* not thread safe, so only calls on rank 0 */
    }
    CmiNodeAllBarrier();

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
    if (CmiInCommThread())
    {
      cpuAffSyncWaitCommThread(cpuAffSyncCommThreadDone);
    }
    else
#endif
    {
      const int numnodes = CmiNumNodes();
      const int numpes = CmiNumPes();

      if (CmiMyPe() == 0)
      {
        int i;
        rankmsg = (rankMsg *)CmiAlloc(rankMsg::size(numnodes, numpes));
        CmiSetHandler((char *)rankmsg, cpuAffinityRecvHandlerIdx);
        size_t siz = sizeof(rankMsg);
        rankmsg->node_ranks_on_host = (int *)((char*)rankmsg + siz);
        siz += numnodes*sizeof(int);
        rankmsg->node_hosts = (int *)((char*)rankmsg + siz);
        siz += numnodes*sizeof(int);
        rankmsg->node_numpes = (int *)((char*)rankmsg + siz);
        siz += numnodes*sizeof(int);
        rankmsg->pe_ranks_on_host = (int *)((char*)rankmsg + siz);
        siz += numpes*sizeof(int);
        rankmsg->pe_hosts = (int *)((char*)rankmsg + siz);
        siz += numpes*sizeof(int);
        CmiAssert(siz == rankMsg::size(numnodes, numpes));
        for (i=0; i<numnodes; i++) {
          rankmsg->node_ranks_on_host[i] = 0;
          rankmsg->node_hosts[i] = -1;
          rankmsg->node_numpes[i] = 0;
        }
        for (i=0; i<numpes; i++) {
          rankmsg->pe_ranks_on_host[i] = 0;
          rankmsg->pe_hosts[i] = -1;
        }
      }

      /* prepare a msg to send */
      msg = (hostnameMsg *)CmiAlloc(sizeof(hostnameMsg));
      CmiSetHandler((char *)msg, cpuAffinityHandlerIdx);
      msg->node = CmiMyNode();
      msg->pe = CmiMyPe();
      msg->ip = myip;
      CmiSyncSendAndFree(0, sizeof(hostnameMsg), (void *)msg);

      if (CmiMyPe() == 0)
      {
        cpuAffSyncWait(cpuAffSyncHandlerDone);

        const size_t hostcount = hostTable.size();

        DEBUGP(("Cpuaffinity> %zu unique hosts detected!\n", hostcount));
        for (const auto & pair : hostTable)
          CmiFree(pair.second);
        hostTable.clear();

        {
          std::vector<int> ranks_on_host(hostcount);
          for (int i = 0; i < numnodes; i++) {
            rankmsg->node_ranks_on_host[i] = ranks_on_host[rankmsg->node_hosts[i]]++;
          }
          ranks_on_host.assign(hostcount, 0);
          for (int i = 0; i < numpes; i++) {
            rankmsg->pe_ranks_on_host[i] = ranks_on_host[rankmsg->pe_hosts[i]]++;
          }
        }

        CmiSyncBroadcastAllAndFree(rankMsg::size(numnodes, numpes), (void *)rankmsg);
      }

      cpuAffSyncWait(cpuAffSyncBroadcastDone);

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
      if (CmiMyRank() == 0)
        cpuAffSyncCommThreadDone = true;
#endif
    }

    CmiBarrier();
#endif
  }

#if CMK_SMP && !CMK_SMP_NO_COMMTHD
  if (CmiInCommThread() && commap != NULL)
  {
    /* comm thread either can float around, or pin down to the last rank.
       however it seems to be reportedly slower if it is floating */
    int mycore = search_pemap(commap, CmiMyPeGlobal()-CmiNumPesGlobal());
    if (commap_logical_flag) {
      if (-1 == CmiSetCPUAffinityLogical(mycore))
        CmiAbort("CmiSetCPUAffinityLogical failed!");
    }
    else {
      if (-1 == CmiSetCPUAffinity(mycore))
        CmiAbort("CmiSetCPUAffinity failed!");
    }
    if (CmiPhysicalNodeID(CmiMyPe()) == 0) {
      CmiPrintf("Charm++> set comm %d on node %d to PU %c#%d\n",
          CmiMyPe()-CmiNumPes(), CmiMyNode(), commap_logical_flag ? 'L' : 'P', mycore);
    }
  }
#endif

  CmiNodeAllBarrier();

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
