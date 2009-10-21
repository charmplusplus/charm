/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

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
 */

#if 1

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>

#if CMK_BLUEGENEL || CMK_BLUEGENEP
#include "TopoManager.h"
#endif

#if CMK_CRAYXT
extern "C" int getXTNodeID(int mype, int numpes);
#endif

#if defined(__APPLE__)  && CMK_HAS_MULTIPROCESSING_H
#include <Carbon/Carbon.h>
#include <Multiprocessing.h>
#endif

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
#elif defined(_SC_CRAY_NCPU)
  a = sysconf(_SC_CRAY_NCPU);
#elif defined(_SC_NPROC_ONLN)
  a = sysconf(_SC_NPROC_ONLN); /* number of active/running CPUs */
#endif
  if (a == -1) a = 1;

#if defined(ARCH_HPUX11) || defined(ARCH_HPUX10)
  a = mpctl(MPC_GETNUMSPUS, 0, 0); /* total number of CPUs */
#endif /* HPUX */

  return a;
}

static int cpuTopoHandlerIdx;
static int cpuTopoRecvHandlerIdx;

typedef struct _hostnameMsg {
  char core[CmiMsgHeaderSizeBytes];
  skt_ip_t ip;
  int pe;
  int ncores;
  int rank;
  int nodeID;
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
    for (i=0; i<CmiNumPes(); i++)  unodes.push_back(nodeIDs[i]);
    //unodes.bubbleSort(0, CmiNumPes()-1);
    unodes.quickSort();
    int last = -1;
    for (i=0; i<CmiNumPes(); i++)  { 
        if (unodes[i] != last) numNodes++; 
        last=unodes[i];
    }
    if (numNodes == 0) 
      numNodes = CmiNumPes();
    else
      CpuTopology::supported = 1;
    return numNodes;
#endif
  }

  void sort() {
    int i;
    numUniqNodes();
    bynodes = new CkVec<int>[numNodes];
    if (supported) {
      for (i=0; i<CmiNumPes(); i++){
        CmiAssert(nodeIDs[i] >=0 && nodeIDs[i] <= numNodes); // Sanity check for bug that occurs on mpi-crayxt
        bynodes[nodeIDs[i]].push_back(i);
      }
    }
    else {    /* not supported */
      for (i=0;i<numNodes;i++)  bynodes[i].push_back(i);
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
int CpuTopology::numNodes = 0;
CkVec<int> *CpuTopology::bynodes = NULL;
int CpuTopology::supported = 0;

static CpuTopology cpuTopo;
static CmiNodeLock topoLock = NULL;
static int done = 0;

/* called on PE 0 */
static void cpuTopoHandler(void *m)
{
  static int count = 0;
  static int nodecount = 0;
  hostnameMsg *rec;
  hostnameMsg *msg = (hostnameMsg *)m;
  hostnameMsg *tmpm;
  char str[128];
  int tag, tag1, pe, myrank;

  if (topomsg == NULL) {
    int i;
    hostTable = CmmNew();
    topomsg = (nodeTopoMsg *)CmiAlloc(sizeof(nodeTopoMsg)+CmiNumPes()*sizeof(int));
    CmiSetHandler((char *)topomsg, cpuTopoRecvHandlerIdx);
    topomsg->nodes = (int *)((char*)topomsg + sizeof(nodeTopoMsg));
    for (i=0; i<CmiNumPes(); i++) topomsg->nodes[i] = -1;
  }
  CmiAssert(topomsg != NULL);

/*   for debug
  skt_print_ip(str, msg->ip);
  printf("hostname: %d %s\n", msg->pe, str);
*/
  tag = *(int*)&msg->ip;
  pe = msg->pe;
  if ((rec = (hostnameMsg *)CmmProbe(hostTable, 1, &tag, &tag1)) != NULL) {
    CmiFree(msg);
  }
  else {
    msg->nodeID = nodecount++;
    rec = msg;
    CmmPut(hostTable, 1, &tag, msg);
  }
  myrank = rec->rank%rec->ncores;
  topomsg->nodes[pe] = rec->nodeID;
  rec->rank ++;
  count ++;
  if (count == CmiNumPes()) {
    char str[256];
    int ncores = CmiNumCores();
    if (ncores > 1)
    sprintf(str, "Charm++> Running on %d unique compute nodes (%d-way SMP).\n", CmmEntries(hostTable), ncores);
    else
    sprintf(str, "Charm++> Running on %d unique compute nodes.\n", CmmEntries(hostTable));
    CmiPrintf(str);
    //hostnameMsg *tmpm;
    tag = CmmWildCard;
    while (tmpm = (hostnameMsg *)CmmGet(hostTable, 1, &tag, &tag1)) CmiFree(tmpm);
    CmmFree(hostTable);

    CmiSyncBroadcastAllAndFree(sizeof(nodeTopoMsg)+CmiNumPes()*sizeof(int), (char *)topomsg);
  }
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
  CmiUnlock(topoLock);

  // if (CmiMyPe() == 0) cpuTopo.print();
}

/******************  API implementation **********************/

extern "C" int CmiCpuTopologyEnabled()
{
  return CpuTopology::supported;
}

extern "C" int CmiOnSamePhysicalNode(int pe1, int pe2)
{
  int *nodeIDs = cpuTopo.nodeIDs;
  if (nodeIDs == NULL) return pe1 == pe2;
  else return nodeIDs[pe1] == nodeIDs[pe2];
}

// return -1 when not supported
extern "C" int CmiNumPhysicalNodes()
{
  return cpuTopo.numUniqNodes();
}

extern "C" int CmiNumPesOnPhysicalNode(int pe)
{
  return !cpuTopo.supported?1:(int)cpuTopo.bynodes[cpuTopo.nodeIDs[pe]].size();
}

// pelist points to system memory, user should not free it
extern "C" void CmiGetPesOnPhysicalNode(int pe, int **pelist, int *num)
{
  CmiAssert(pe >=0 && pe < CmiNumPes());
  if (cpuTopo.supported) {
    *num = cpuTopo.bynodes[cpuTopo.nodeIDs[pe]].size();
    if (pelist!=NULL && *num>0) *pelist = cpuTopo.bynodes[cpuTopo.nodeIDs[pe]].getVec();
  }
  else {
    *num = 1;
    *pelist = cpuTopo.bynodes[pe].getVec();
  }
}

// the least number processor on the same physical node
extern "C"  int CmiGetFirstPeOnPhysicalNode(int pe)
{
  CmiAssert(pe >=0 && pe < CmiNumPes());
  if (!cpuTopo.supported) return pe;
  const CkVec<int> &v = cpuTopo.bynodes[cpuTopo.nodeIDs[pe]];
  return v[0];
}


static int _noip = 0;

extern "C" void CmiInitCPUTopology(char **argv)
{
  static skt_ip_t myip;
  int ret, i;
  hostnameMsg  *msg;
 
  if (CmiMyRank() ==0) {
     topoLock = CmiCreateLock();
  }

  int obtain_flag = CmiGetArgFlagDesc(argv,"+obtain_cpu_topology",
					   "obtain cpu topology info");
#if !defined(__BLUEGENE__)
  obtain_flag = 1;
#endif
  if (CmiGetArgFlagDesc(argv,"+skip_cpu_topology",
                               "skip the processof getting cpu topology info"))
    obtain_flag = 0;

  cpuTopoHandlerIdx =
     CmiRegisterHandler((CmiHandler)cpuTopoHandler);
  cpuTopoRecvHandlerIdx =
     CmiRegisterHandler((CmiHandler)cpuTopoRecvHandler);

  if (!obtain_flag) return;
  else if (CmiMyPe() == 0) {
     CmiPrintf("Charm++> cpu topology info is being gathered.\n");
  }

#if CMK_USE_GM
  CmiBarrier();
#endif

  if (CmiMyPe() >= CmiNumPes()) {
    CmiNodeAllBarrier();         // comm thread waiting
#if CMK_MACHINE_PROGRESS_DEFINED
    while (done < CmiMyNodeSize()) CmiNetworkProgress();
#endif
    return;    /* comm thread return */
  }

#if 0
  if (gethostname(hostname, 999)!=0) {
      strcpy(hostname, "");
  }
#endif
#if CMK_BLUEGENEL || CMK_BLUEGENEP
  if (CmiMyRank() == 0) {
    TopoManager tmgr;

    cpuTopo.numNodes = CmiNumPes();
    cpuTopo.nodeIDs = new int[cpuTopo.numNodes];

    int x, y, z, t, nid;
    for(int i=0; i<cpuTopo.numNodes; i++) {
      tmgr.rankToCoordinates(i, x, y, z, t);
      nid = tmgr.coordinatesToRank(x, y, z, 0);
      cpuTopo.nodeIDs[i] = nid;
    }
    cpuTopo.sort();
  }
  CmiNodeAllBarrier();
  return;
#elif CMK_CRAYXT
  if(CmiMyRank() == 0) {
    cpuTopo.numNodes = CmiNumPes();
    cpuTopo.nodeIDs = new int[cpuTopo.numNodes];

    int nid;
    for(int i=0; i<cpuTopo.numNodes; i++) {
      nid = getXTNodeID(i, cpuTopo.numNodes);
      cpuTopo.nodeIDs[i] = nid;
    }
    int prev = -1;
    nid = -1;

    // this assumes TXYZ mapping and changes nodeIDs
    for(int i=0; i<cpuTopo.numNodes; i++) {
      if(cpuTopo.nodeIDs[i] != prev) {
	prev = cpuTopo.nodeIDs[i];
	cpuTopo.nodeIDs[i] = ++nid;
      }
      else
	cpuTopo.nodeIDs[i] = nid;
    }
    cpuTopo.sort();
    if (CmiMyPe()==0)  CmiPrintf("Charm++> Running on %d unique compute nodes.\n", nid+1);
  }
  CmiNodeAllBarrier();
  return;
#else
  /* get my ip address */
  if (CmiMyRank() == 0)
  {
  #if CMK_HAS_GETHOSTNAME
    myip = skt_my_ip();        /* not thread safe, so only calls on rank 0 */
  #elif CMK_BPROC
    myip = skt_innode_my_ip();
  #else
    if (!CmiMyPe())
    CmiPrintf("CmiInitCPUTopology Warning: Can not get unique name for the compute nodes. \n");
    _noip = 1; 
  #endif
  }

  CmiNodeAllBarrier();
  if (_noip) return; 

    /* prepare a msg to send */
  msg = (hostnameMsg *)CmiAlloc(sizeof(hostnameMsg));
  CmiSetHandler((char *)msg, cpuTopoHandlerIdx);
  msg->pe = CmiMyPe();
  msg->ip = myip;
  msg->ncores = CmiNumCores();
  msg->rank = 0;
  CmiSyncSendAndFree(0, sizeof(hostnameMsg), (char *)msg);

  if (CmiMyPe() == 0) {
    for (i=0; i<CmiNumPes(); i++) CmiDeliverSpecificMsg(cpuTopoHandlerIdx);
    // CsdScheduleCount(CmiNumPes());   // collecting node IP from every processor
  }

  // receive broadcast from PE 0
  CmiDeliverSpecificMsg(cpuTopoRecvHandlerIdx);
  // CsdScheduleCount(1);
  CmiLock(topoLock);
  done++;
  CmiUnlock(topoLock);
#endif

  // now every one should have the node info
}

#else           /* not supporting cpu topology */


extern "C" void CmiInitCPUTopology(char **argv)
{
  /* do nothing */
  int obtain_flag = CmiGetArgFlagDesc(argv,"+obtain_cpu_topology",
						"obtain cpu topology info");
  CmiGetArgFlagDesc(argv,"+skip_cpu_topology",
                               "skip the processof getting cpu topology info");
}

#endif
