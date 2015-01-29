#include <map>
#include "converse.h"
#include "sockRoutines.h"
#include "cklists.h"

#define DEBUGP(x)  /** CmiPrintf x; */

/** This scheme relies on using IP address to identify physical nodes 
 * written by Gengbin Zheng  9/2008
 *
 * last updated 10/4/2009   Gengbin Zheng
 * added function CmiCpuTopologyEnabled() which retuens 1 when supported
 * when not supported return 0
 * all functions when cputopology not support, now act like a normal non-smp 
 * case and all PEs are unique.
 *
 * major changes 10/28/09   Gengbin Zheng
 * - parameters changed from pe to node to be consistent with the function name
 * - two new functions:   CmiPhysicalNodeID and CmiPhysicalRank
 *
 * 3/5/2010   Gengbin Zheng
 * - use CmiReduce to optimize the collection of node info
 */

#if 1

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>

#if CMK_BLUEGENEL || CMK_BLUEGENEP || CMK_BLUEGENEQ
#include "TopoManager.h"
#endif

#if CMK_BLUEGENEQ
#include "spi/include/kernel/process.h"
#endif

#if CMK_CRAYXT || CMK_CRAYXE || CMK_CRAYXC
extern "C" int getXTNodeID(int mpirank, int nummpiranks);
#endif

#if defined(__APPLE__)  && CMK_HAS_MULTIPROCESSING_H
#include <Carbon/Carbon.h>
#include <Multiprocessing.h>
#endif

#if CMK_BIGSIM_CHARM
#include "middle-blue.h"
using namespace BGConverse;
#endif

extern "C" void CommunicationServerPxshm();

extern "C" int CmiNumCores(void) {
  int a = 1;
#ifdef _WIN32
struct _SYSTEM_INFO sysinfo;
#endif  

  /* Allow the user to override the number of CPUs for use
     in scalability testing, debugging, etc. */
  char *forcecount = getenv("FORCECPUCOUNT");
  if (forcecount != NULL) {
    if (sscanf(forcecount, "%d", &a) == 1) {
      return a; /* if we got a valid count, return it */
    } else {
      a = 1;      /* otherwise use the real available hardware CPU count */
    }
  }

#if defined(__APPLE__)  && CMK_HAS_MULTIPROCESSING_H
  a = MPProcessorsScheduled(); /* Number of active/running CPUs */
#endif

#ifdef _WIN32
  //struct _SYSTEM_INFO sysinfo;  
  GetSystemInfo(&sysinfo);
  a = sysinfo.dwNumberOfProcessors; /* total number of CPUs */
#endif /* _MSC_VER */


#ifdef _SC_NPROCESSORS_ONLN
  a = sysconf(_SC_NPROCESSORS_ONLN); /* number of active/running CPUs */
#ifdef _SC_NPROCESSORS_CONF
  /* also consider CPUs that are temporarily powered down by the OS */
  const int b = sysconf(_SC_NPROCESSORS_CONF);
  if ( b > a ) a = b;
#endif
#elif defined(_SC_CRAY_NCPU)
  a = sysconf(_SC_CRAY_NCPU);
#elif defined(_SC_NPROC_ONLN)
  a = sysconf(_SC_NPROC_ONLN); /* number of active/running CPUs */
#endif
#if CMK_BLUEGENEQ
  a *= Kernel_ProcessCount();
#endif

  if (a < 1) a = 1;

  return a;
}

CpvStaticDeclare(int, cpuTopoHandlerIdx);
CpvStaticDeclare(int, cpuTopoRecvHandlerIdx);

struct _procInfo {
  skt_ip_t ip;
  int pe;
  int ncores;
  int rank;
  int nodeID;
};

typedef struct _hostnameMsg {
  char core[CmiMsgHeaderSizeBytes];
  int n;
  _procInfo *procs;
} hostnameMsg;

typedef struct _nodeTopoMsg {
  char core[CmiMsgHeaderSizeBytes];
  int *nodes;
} nodeTopoMsg;

static nodeTopoMsg *topomsg = NULL;
static CmmTable hostTable;

// nodeIDs[pe] is the node number of processor pe
class CpuTopology {
public:
  static int *nodeIDs;
  static int numPes;
  static int numNodes;
  static CkVec<int> *bynodes;
  static int supported;

    // return -1 when not supported
  int numUniqNodes() {
#if 0
    if (numNodes != 0) return numNodes;
    int n = 0;
    for (int i=0; i<CmiNumPes(); i++) 
      if (nodeIDs[i] > n)
	n = nodeIDs[i];
    numNodes = n+1;
    return numNodes;
#else
    if (numNodes > 0) return numNodes;     // already calculated
    CkVec<int> unodes;
    int i;
    for (i=0; i<numPes; i++)  unodes.push_back(nodeIDs[i]);
    //unodes.bubbleSort(0, numPes-1);
    unodes.quickSort();
    int last = -1;
    std::map<int, int> nodemap;  // nodeIDs can be out of range of [0,numNodes]
    for (i=0; i<numPes; i++)  { 
        if (unodes[i] != last) {
          last=unodes[i];
	  nodemap[unodes[i]] = numNodes;
          numNodes++; 
        }
    }
    if (numNodes == 0)  {
      numNodes = CmiNumNodes();
      numPes = CmiNumPes();
    }
    else {
        // re-number nodeIDs, which may be necessary e.g. on BlueGene/P
      for (i=0; i<numPes; i++) nodeIDs[i] = nodemap[nodeIDs[i]];
      CpuTopology::supported = 1;
    }
    return numNodes;
#endif
  }

  void sort() {
    int i;
    numUniqNodes();
    bynodes = new CkVec<int>[numNodes];
    if (supported) {
      for (i=0; i<numPes; i++){
        CmiAssert(nodeIDs[i] >=0 && nodeIDs[i] <= numNodes); // Sanity check for bug that occurs on mpi-crayxt
        bynodes[nodeIDs[i]].push_back(i);
      }
    }
    else {    /* not supported/enabled */
      for (i=0;i<CmiNumPes();i++)  bynodes[CmiNodeOf(i)].push_back(i);
    }
  }

  void print() {
    int i;
    CmiPrintf("Charm++> Cpu topology info:\n");
    CmiPrintf("PE to node map: ");
    for (i=0; i<CmiNumPes(); i++)
      CmiPrintf("%d ", nodeIDs[i]);
    CmiPrintf("\n");
    CmiPrintf("Node to PE map:\n");
    for (i=0; i<numNodes; i++) {
      CmiPrintf("Chip #%d: ", i);
      for (int j=0; j<bynodes[i].size(); j++)
	CmiPrintf("%d ", bynodes[i][j]);
      CmiPrintf("\n");
    }
  }

};

int *CpuTopology::nodeIDs = NULL;
int CpuTopology::numPes = 0;
int CpuTopology::numNodes = 0;
CkVec<int> *CpuTopology::bynodes = NULL;
int CpuTopology::supported = 0;

static CpuTopology cpuTopo;
static CmiNodeLock topoLock = 0; /* Not spelled 'NULL' to quiet warnings when CmiNodeLock is just 'int' */
static int done = 0;

/* called on PE 0 */
static void cpuTopoHandler(void *m)
{
  _procInfo *rec;
  hostnameMsg *msg = (hostnameMsg *)m;
  char str[256];
  int tag, tag1, pe;

  if (topomsg == NULL) {
    int i;
    hostTable = CmmNew();
    topomsg = (nodeTopoMsg *)CmiAlloc(sizeof(nodeTopoMsg)+CmiNumPes()*sizeof(int));
    CmiSetHandler((char *)topomsg, CpvAccess(cpuTopoRecvHandlerIdx));
    topomsg->nodes = (int *)((char*)topomsg + sizeof(nodeTopoMsg));
    for (i=0; i<CmiNumPes(); i++) topomsg->nodes[i] = -1;
  }
  CmiAssert(topomsg != NULL);

  msg->procs = (_procInfo*)((char*)msg + sizeof(hostnameMsg));
  CmiAssert(msg->n == CmiNumPes());
  for (int i=0; i<msg->n; i++) 
  {
    _procInfo *proc = msg->procs+i;

/*   for debug
  skt_print_ip(str, msg->ip);
  printf("hostname: %d %s\n", msg->pe, str);
*/
    tag = *(int*)&proc->ip;
    pe = proc->pe;
    if ((rec = (_procInfo *)CmmProbe(hostTable, 1, &tag, &tag1)) != NULL) {
    }
    else {
      proc->nodeID = pe;           // we will compact the node ID later
      rec = proc;
      CmmPut(hostTable, 1, &tag, proc);
    }
    topomsg->nodes[pe] = rec->nodeID;
    rec->rank ++;
  }

    // assume all nodes have same number of cores
  int ncores = CmiNumCores();
  if (ncores > 1)
    sprintf(str, "Charm++> Running on %d unique compute nodes (%d-way SMP).\n", CmmEntries(hostTable), ncores);
  else
    sprintf(str, "Charm++> Running on %d unique compute nodes.\n", CmmEntries(hostTable));
  CmiPrintf(str);
    // clean up CmmTable
  hostnameMsg *tmpm;
  tag = CmmWildCard;
  while (tmpm = (hostnameMsg *)CmmGet(hostTable, 1, &tag, &tag1));
  CmmFree(hostTable);
  CmiFree(msg);

  CmiSyncBroadcastAllAndFree(sizeof(nodeTopoMsg)+CmiNumPes()*sizeof(int), (char *)topomsg);
}

/* called on each processor */
static void cpuTopoRecvHandler(void *msg)
{
  nodeTopoMsg *m = (nodeTopoMsg *)msg;
  m->nodes = (int *)((char*)m + sizeof(nodeTopoMsg));

  CmiLock(topoLock);
  if (cpuTopo.nodeIDs == NULL) {
    cpuTopo.nodeIDs = m->nodes;
    cpuTopo.sort();
  }
  else
    CmiFree(m);
  done++;
  CmiUnlock(topoLock);

  //if (CmiMyPe() == 0) cpuTopo.print();
}

// reduction function
static void * combineMessage(int *size, void *data, void **remote, int count) 
{
  int i, j;
  int nprocs = ((hostnameMsg *)data)->n;
  if (count == 0) return data;
  for (i=0; i<count; i++) nprocs += ((hostnameMsg *)remote[i])->n;
  *size = sizeof(hostnameMsg)+sizeof(_procInfo)*nprocs;
  hostnameMsg *msg = (hostnameMsg *)CmiAlloc(*size);
  msg->procs = (_procInfo*)((char*)msg + sizeof(hostnameMsg));
  msg->n = nprocs;
  CmiSetHandler((char *)msg, CpvAccess(cpuTopoHandlerIdx));

  int n=0;
  hostnameMsg *m = (hostnameMsg*)data;
  m->procs = (_procInfo*)((char*)m + sizeof(hostnameMsg));
  for (j=0; j<m->n; j++)
    msg->procs[n++] = m->procs[j];
  for (i=0; i<count; i++) {
    m = (hostnameMsg*)remote[i];
    m->procs = (_procInfo*)((char*)m + sizeof(hostnameMsg));
    for (j=0; j<m->n; j++)
      msg->procs[n++] = m->procs[j];
  }
  return msg;
}

/******************  API implementation **********************/

extern "C" int LrtsCpuTopoEnabled()
{
  return CpuTopology::supported;
}

extern "C" int LrtsPeOnSameNode(int pe1, int pe2)
{
  int *nodeIDs = cpuTopo.nodeIDs;
  if (!cpuTopo.supported || nodeIDs == NULL) return CmiNodeOf(pe1) == CmiNodeOf(pe2);
  else return nodeIDs[pe1] == nodeIDs[pe2];
}

// return -1 when not supported
extern "C" int LrtsNumNodes()
{
  if (!cpuTopo.supported) return CmiNumNodes();
  else return cpuTopo.numUniqNodes();
}

extern "C" int LrtsNodeSize(int node)
{
  return !cpuTopo.supported?CmiNodeSize(node):(int)cpuTopo.bynodes[node].size();
}

// pelist points to system memory, user should not free it
extern "C" void LrtsPeOnNode(int node, int **pelist, int *num)
{
  *num = cpuTopo.bynodes[node].size();
  if (pelist!=NULL && *num>0) *pelist = cpuTopo.bynodes[node].getVec();
}

extern "C" int LrtsRankOf(int pe)
{
  if (!cpuTopo.supported) return CmiRankOf(pe);
  const CkVec<int> &v = cpuTopo.bynodes[cpuTopo.nodeIDs[pe]];
  int rank = 0;  
  int npes = v.size();
  while (rank < npes && v[rank] < pe) rank++;       // already sorted
  CmiAssert(v[rank] == pe);
  return rank;
}

extern "C" int LrtsNodeOf(int pe)
{
  if (!cpuTopo.supported) return CmiNodeOf(pe);
  return cpuTopo.nodeIDs[pe];
}

// the least number processor on the same physical node
extern "C"  int LrtsNodeFirst(int node)
{
  if (!cpuTopo.supported) return CmiNodeFirst(node);
  return cpuTopo.bynodes[node][0];
}


static int _noip = 0;
extern "C" void LrtsInitCpuTopo(char **argv)
{
  static skt_ip_t myip;
  hostnameMsg  *msg;
  double startT;
 
  int obtain_flag = 1;              // default on
  int show_flag = 0;                // default not show topology

  if (CmiMyRank() ==0) {
     topoLock = CmiCreateLock();
  }

#if __FAULT__
  obtain_flag = 0;
#endif
  if(CmiGetArgFlagDesc(argv,"+obtain_cpu_topology",
					   "obtain cpu topology info"))
    obtain_flag = 1;
  if (CmiGetArgFlagDesc(argv,"+skip_cpu_topology",
                               "skip the processof getting cpu topology info"))
    obtain_flag = 0;
  if(CmiGetArgFlagDesc(argv,"+show_cpu_topology",
					   "Show cpu topology info"))
    show_flag = 1;

#if CMK_BIGSIM_CHARM
  if (BgNodeRank() == 0)
#endif
    {
      CpvInitialize(int, cpuTopoHandlerIdx);
      CpvInitialize(int, cpuTopoRecvHandlerIdx);
      CpvAccess(cpuTopoHandlerIdx) =
	CmiRegisterHandler((CmiHandler)cpuTopoHandler);
      CpvAccess(cpuTopoRecvHandlerIdx) =
	CmiRegisterHandler((CmiHandler)cpuTopoRecvHandler);
    }
  if (!obtain_flag) {
    if (CmiMyRank() == 0) cpuTopo.sort();
    CmiNodeAllBarrier();
    CcdRaiseCondition(CcdTOPOLOGY_AVAIL);      // call callbacks
    return;
  }

  if (CmiMyPe() == 0) {
#if CMK_BIGSIM_CHARM
    if (BgNodeRank() == 0)
#endif
      startT = CmiWallTimer();
  }

#if CMK_BIGSIM_CHARM
  if (BgNodeRank() == 0)
  {
    //int numPes = BgNumNodes()*BgGetNumWorkThread();
    int numPes = cpuTopo.numPes = CkNumPes();
    cpuTopo.nodeIDs = new int[numPes];
    CpuTopology::supported = 1;
    int wth = BgGetNumWorkThread();
    for (int i=0; i<numPes; i++) {
      int nid = i / wth;
      cpuTopo.nodeIDs[i] = nid;
    }
    cpuTopo.sort();
  }
  return;
#else

#if CMK_USE_GM
  CmiBarrier();
#endif


#if 0
  if (gethostname(hostname, 999)!=0) {
      strcpy(hostname, "");
  }
#endif
#if CMK_BLUEGENEL || CMK_BLUEGENEP
  if (CmiMyRank() == 0) {
    TopoManager tmgr;

    int numPes = cpuTopo.numPes = CmiNumPes();
    cpuTopo.nodeIDs = new int[numPes];
    CpuTopology::supported = 1;

    int x, y, z, t, nid;
    for(int i=0; i<numPes; i++) {
      tmgr.rankToCoordinates(i, x, y, z, t);
      nid = tmgr.coordinatesToRank(x, y, z, 0);
      cpuTopo.nodeIDs[i] = nid;
    }
    cpuTopo.sort();
    if (CmiMyPe()==0)  CmiPrintf("Charm++> Running on %d unique compute nodes (%d-way SMP).\n", cpuTopo.numNodes, CmiNumCores());
  }
  CmiNodeAllBarrier();
#elif CMK_BLUEGENEQ
  if (CmiMyRank() == 0) {
   TopoManager tmgr;

    int numPes = cpuTopo.numPes = CmiNumPes();
    cpuTopo.nodeIDs = new int[numPes];
    CpuTopology::supported = 1;

    int a, b, c, d, e, t, nid;
    for(int i=0; i<numPes; i++) {
      tmgr.rankToCoordinates(i, a, b, c, d, e, t);
      nid = tmgr.coordinatesToRank(a, b, c, d, e, 0);
      cpuTopo.nodeIDs[i] = nid;
    }
    cpuTopo.sort();
    if (CmiMyPe()==0)  CmiPrintf("Charm++> Running on %d unique compute nodes (%d-way SMP).\n", cpuTopo.numNodes, CmiNumCores());
  }
  CmiNodeAllBarrier();
#elif CMK_CRAYXT || CMK_CRAYXE || CMK_CRAYXC
  if(CmiMyRank() == 0) {
    int numPes = cpuTopo.numPes = CmiNumPes();
    int numNodes = CmiNumNodes();
    cpuTopo.nodeIDs = new int[numPes];
    CpuTopology::supported = 1;

    int nid;
    for(int i=0; i<numPes; i++) {
      nid = getXTNodeID(CmiNodeOf(i), numNodes);
      cpuTopo.nodeIDs[i] = nid;
    }
    int prev = -1;
    nid = -1;

    // this assumes that all cores on a node have consecutive MPI rank IDs
    // and then changes nodeIDs to 0 to numNodes-1
    for(int i=0; i<numPes; i++) {
      if(cpuTopo.nodeIDs[i] != prev) {
	prev = cpuTopo.nodeIDs[i];
	cpuTopo.nodeIDs[i] = ++nid;
      }
      else
	cpuTopo.nodeIDs[i] = nid;
    }
    cpuTopo.sort();
    if (CmiMyPe()==0)  CmiPrintf("Charm++> Running on %d unique compute nodes (%d-way SMP).\n", cpuTopo.numNodes, CmiNumCores());
  }
  CmiNodeAllBarrier();

#else

  bool topoInProgress = true;

  if (CmiMyPe() >= CmiNumPes()) {
    CmiNodeAllBarrier();         // comm thread waiting
#if CMK_MACHINE_PROGRESS_DEFINED
#if ! CMK_CRAYXT
    while (topoInProgress) {
      CmiNetworkProgress();
      CmiLock(topoLock);
      topoInProgress = done < CmiMyNodeSize();
      CmiUnlock(topoLock);
    }
#endif
#endif
    return;    /* comm thread return */
  }

    /* get my ip address */
  if (CmiMyRank() == 0)
  {
  #if CMK_HAS_GETHOSTNAME && !CMK_BLUEGENEQ
    myip = skt_my_ip();        /* not thread safe, so only calls on rank 0 */
    // fprintf(stderr, "[%d] IP is %d.%d.%d.%d\n", CmiMyPe(), myip.data[0],myip.data[1],myip.data[2],myip.data[3]);
  #elif CMK_BPROC
    myip = skt_innode_my_ip();
  #else
    if (!CmiMyPe())
    CmiPrintf("CmiInitCPUTopology Warning: Can not get unique name for the compute nodes. \n");
    _noip = 1; 
  #endif
    cpuTopo.numPes = CmiNumPes();
  }

  CmiNodeAllBarrier();
  if (_noip) return; 

    /* prepare a msg to send */
  msg = (hostnameMsg *)CmiAlloc(sizeof(hostnameMsg)+sizeof(_procInfo));
  msg->n = 1;
  msg->procs = (_procInfo*)((char*)msg + sizeof(hostnameMsg));
  CmiSetHandler((char *)msg, CpvAccess(cpuTopoHandlerIdx));
  msg->procs[0].pe = CmiMyPe();
  msg->procs[0].ip = myip;
  msg->procs[0].ncores = CmiNumCores();
  msg->procs[0].rank = 0;
  msg->procs[0].nodeID = 0;
  CmiReduce(msg, sizeof(hostnameMsg)+sizeof(_procInfo), combineMessage);

  // blocking here
  while (topoInProgress) {
    CsdSchedulePoll();
    CmiLock(topoLock);
    topoInProgress = done < CmiMyNodeSize();
    CmiUnlock(topoLock);
  }

  if (CmiMyPe() == 0) {
#if CMK_BIGSIM_CHARM
    if (BgNodeRank() == 0)
#endif
      CmiPrintf("Charm++> cpu topology info is gathered in %.3f seconds.\n", CmiWallTimer()-startT);
  }
#endif

#endif   /* __BIGSIM__ */

  // now every one should have the node info
  CcdRaiseCondition(CcdTOPOLOGY_AVAIL);      // call callbacks
  if (CmiMyPe() == 0 && show_flag) cpuTopo.print();
}

#else           /* not supporting cpu topology */

extern "C" void LrtsInitCpuTopo(char **argv)
{
  /* do nothing */
  int obtain_flag = CmiGetArgFlagDesc(argv,"+obtain_cpu_topology",
						"obtain cpu topology info");
  CmiGetArgFlagDesc(argv,"+skip_cpu_topology",
                               "skip the processof getting cpu topology info");
  CmiGetArgFlagDesc(argv,"+show_cpu_topology",
					   "Show cpu topology info");
}

#endif

extern "C" int CmiCpuTopologyEnabled()
{
  return LrtsCpuTopoEnabled();
}
extern "C" int CmiPeOnSamePhysicalNode(int pe1, int pe2)
{
  return LrtsPeOnSameNode(pe1, pe2);
}
extern "C" int CmiNumPhysicalNodes()
{
  return LrtsNumNodes();
}
extern "C" int CmiNumPesOnPhysicalNode(int node)
{
  return LrtsNodeSize(node);
}
extern "C" void CmiGetPesOnPhysicalNode(int node, int **pelist, int *num)
{
  LrtsPeOnNode(node, pelist, num);
}
extern "C" int CmiPhysicalRank(int pe)
{
  return LrtsRankOf(pe);
}
extern "C" int CmiPhysicalNodeID(int pe)
{
  return LrtsNodeOf(pe);
}
extern "C" int CmiGetFirstPeOnPhysicalNode(int node)
{
  return LrtsNodeFirst(node);
}
extern "C" void CmiInitCPUTopology(char **argv)
{
  LrtsInitCpuTopo(argv);
}

