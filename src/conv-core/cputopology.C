/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "converse.h"
#include "sockRoutines.h"

#define DEBUGP(x)  /* CmiPrintf x;  */

/*
 This scheme relies on using IP address to identify nodes and assigning 
cpu affinity.  

 when CMK_NO_SOCKETS, which is typically on cray xt3 and bluegene/L.
 There is no hostname for the compute nodes.
*/
#if 1

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


#include <stdlib.h>
#include <stdio.h>


extern "C" int Cmi_num_cores(void) {
  int a = 1;
#ifdef _WIN32
struct _SYSTEM_INFO sysinfo;
#endif  

  /* Allow the user to override the number of CPUs for use
     in scalability testing, debugging, etc. */
  char *forcecount = getenv("VMDFORCECPUCOUNT");
  if (forcecount != NULL) {
    if (sscanf(forcecount, "%d", &a) == 1) {
      return a; /* if we got a valid count, return it */
    } else {
      a = 1;      /* otherwise use the real available hardware CPU count */
    }
  }

#if defined(__APPLE__) 
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

static int cpuAffinityHandlerIdx;
static int cpuAffinityRecvHandlerIdx;

typedef struct _hostnameMsg {
  char core[CmiMsgHeaderSizeBytes];
  int pe;
  skt_ip_t ip;
  int ncores;
  int rank;
  int nodenum;
} hostnameMsg;

typedef struct _nodeTopoMsg {
  char core[CmiMsgHeaderSizeBytes];
  int *nodes;
} nodeTopoMsg;

static nodeTopoMsg *topomsg = NULL;
static CmmTable hostTable;

typedef void (*CallbackFn)(void *cb);

// nodenum[pe] is the node number of processor pe
class CpuTopology {
public:
static int *nodenum;
CallbackFn   fn;
void        *cb;

CpuTopology(): fn(NULL), cb(NULL) {}
};

CpvDeclare(CpuTopology, cpuTopo);
CmiNodeLock topoLock;

/* called on PE 0 */
static void cpuAffinityHandler(void *m)
{
  static int count = 0;
  static int nodecount = 0;
  hostnameMsg *rec;
  hostnameMsg *msg = (hostnameMsg *)m;
  hostnameMsg *tmpm;
  char str[128];
  int tag, tag1, pe, myrank;
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
    msg->nodenum = nodecount++;
    rec = msg;
    CmmPut(hostTable, 1, &tag, msg);
  }
  myrank = rec->rank%rec->ncores;
  topomsg->nodes[pe] = rec->nodenum;
  rec->rank ++;
  count ++;
  if (count == CmiNumPes()) {
    CmiPrintf("Cpuaffinity> %d unique compute nodes detected! \n", CmmEntries(hostTable));
    //hostnameMsg *tmpm;
    tag = CmmWildCard;
    while (tmpm = (hostnameMsg *)CmmGet(hostTable, 1, &tag, &tag1)) CmiFree(tmpm);
    CmmFree(hostTable);
    CmiSyncBroadcastAllAndFree(sizeof(nodeTopoMsg)+CmiNumPes()*sizeof(int), (char *)topomsg);
  }
}

/* called on each processor */
static void cpuAffinityRecvHandler(void *msg)
{
  int myrank;
  nodeTopoMsg *m = (nodeTopoMsg *)msg;
  m->nodes = (int *)((char*)m + sizeof(nodeTopoMsg));

  CmiLock(topoLock);
  if (CpvAccess(cpuTopo).nodenum == NULL)
    CpvAccess(cpuTopo).nodenum = m->nodes;
  else
    CmiFree(m);
  CmiUnlock(topoLock);

    // call callback
  if (CpvAccess(cpuTopo).fn!=NULL)
    CpvAccess(cpuTopo).fn(CpvAccess(cpuTopo).cb);

}

#if CMK_CRAYXT
extern int getXTNodeID(int mype, int numpes);
#endif

// only one callback is allowed right now
extern "C" void CmiRegisterCPUTopologyCallback(CallbackFn fn, void *cb)
{
  CpvAccess(cpuTopo).fn = fn;
  CpvAccess(cpuTopo).cb = cb;
}

extern "C" void CmiInitCPUTopology(char **argv)
{
  static skt_ip_t myip;
  int ret, i;
  hostnameMsg  *msg;
 
  CpvInitialize(CpuTopology, cpuTopo);
  if (CmiMyRank() ==0) {
        topoLock = CmiCreateLock();
  }


  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
						"set cpu affinity");

  cpuAffinityHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityHandler);
  cpuAffinityRecvHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityRecvHandler);

  if (!affinity_flag) return;
  else if (CmiMyPe() == 0) {
     CmiPrintf("Charm++> cpu affinity enabled! \n");
  }

  if (CmiMyPe() >= CmiNumPes()) {
      /* comm thread either can float around, or pin down to the last rank.
         however it seems to be reportedly slower if it is floating */
    CmiNodeAllBarrier();
    return;    /* comm thread return */
  }

#if 0
  if (gethostname(hostname, 999)!=0) {
      strcpy(hostname, "");
  }
#endif

    /* get my ip address */
  if (CmiMyRank() == 0)
  {
#if CMK_CRAYXT
    ret = getXTNodeID(CmiMyPe(), CmiNumPes());
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
  msg->ncores = Cmi_num_cores();
  msg->rank = 0;
  CmiSyncSendAndFree(0, sizeof(hostnameMsg), (char *)msg);

  if (CmiMyPe() == 0) {
    int i;
    hostTable = CmmNew();
    topomsg = (nodeTopoMsg *)CmiAlloc(sizeof(nodeTopoMsg)+CmiNumPes()*sizeof(int));
    CmiSetHandler((char *)topomsg, cpuAffinityRecvHandlerIdx);
    topomsg->nodes = (int *)((char*)topomsg + sizeof(nodeTopoMsg));
    for (i=0; i<CmiNumPes(); i++) topomsg->nodes[i] = -1;
  }
}

#else           /* not supporting affinity */


extern "C" void CmiInitCPUTopology(char **argv)
{
  /* do nothing */
}

#endif
