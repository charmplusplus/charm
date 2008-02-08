/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "converse.h"
#include "sockRoutines.h"

#define DEBUGP(x)   /*  CmiPrintf x;  */

/*
 This scheme relies on using IP address to identify nodes and assigning 
cpu affinity.  

 when CMK_NO_SOCKETS, which is typically on cray xt3 and bluegene/L.
 There is no hostname for the compute nodes.
*/
#if (CMK_HAS_SETAFFINITY || defined (_WIN32)) 

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else
#define _GNU_SOURCE
#include <sched.h>
long sched_setaffinity(pid_t pid, unsigned int len, unsigned long *user_mask_ptr);
long sched_getaffinity(pid_t pid, unsigned int len, unsigned long *user_mask_ptr);
#endif

#if defined(__APPLE__) 
#include <Carbon/Carbon.h> /* Carbon APIs for Multiprocessing */
#endif

#if defined(ARCH_HPUX11) ||  defined(ARCH_HPUX10)
#include <sys/mpctl.h>
#endif


int num_cores(void) {
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

/* This implementation assumes the default x86 CPU mask size used by Linux */
/* For a large SMP machine, this code should be changed to use a variable sized   */
/* CPU affinity mask buffer instead, as the present code will fail beyond 32 CPUs */
int set_cpu_affinity(int cpuid) {
  unsigned long mask = 0xffffffff;
  unsigned int len = sizeof(mask);

 #ifdef _WIN32
   HANDLE hProcess;
 #endif
 
  /* set the affinity mask if possible */
  if ((cpuid / 8) > len) {
    printf("Mask size too small to handle requested CPU ID\n");
    return -1;
  } else {
    mask = 1 << cpuid;   /* set the affinity mask exclusively to one CPU */
  }

#ifdef _WIN32
  //HANDLE hProcess = GetCurrentProcess();
  hProcess = GetCurrentProcess();
  if (SetProcessAffinityMask(hProcess, mask) == 0) {
    return -1;
  }
#else
  /* PID 0 refers to the current process */
  if (sched_setaffinity(0, len, &mask) < 0) {
    perror("sched_setaffinity");
    return -1;
  }
#endif

  return 0;
}

int set_thread_affinity(int cpuid) {
#if CMK_SMP
  unsigned long mask = 0xffffffff;
  unsigned int len = sizeof(mask);

#ifdef _WIN32
  HANDLE hThread;
#endif	
  
  /* set the affinity mask if possible */
  if ((cpuid / 8) > len) {
    printf("Mask size too small to handle requested CPU ID\n");
    return -1;
  } else {
    mask = 1 << cpuid;   /* set the affinity mask exclusively to one CPU */
  }

#ifdef _WIN32
  //HANDLE hThread = GetCurrentThread();
  hThread = GetCurrentThread();
  if (SetThreadAffinityMask(hThread, mask) == 0) {
    return -1;
  }
#elif  CMK_HAS_PTHREAD_SETAFFINITY
  /* PID 0 refers to the current process */
  if (pthread_setaffinity_np(pthread_self(), len, &mask) < 0) {
    perror("pthread_setaffinity");
    return -1;
  }
#else
  return set_cpu_affinity(cpuid);
#endif

  return 0;
#else
  return -1;
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
  
  printf("CPU affinity mask is: %08lx\n", pMask);
  
#else
  unsigned long mask;
  unsigned int len = sizeof(mask);

  /* PID 0 refers to the current process */
  if (sched_getaffinity(0, len, &mask) < 0) {
    perror("sched_getaffinity");
    return -1;
  }

  printf("CPU affinity mask is: %08lx\n", mask);
#endif
  return 0;
}

static int cpuAffinityHandlerIdx;
static int cpuAffinityRecvHandlerIdx;

typedef struct _hostnameMsg {
  char core[CmiMsgHeaderSizeBytes];
  int pe;
  skt_ip_t ip;
  int ncores;
  int rank;
} hostnameMsg;

typedef struct _rankMsg {
  char core[CmiMsgHeaderSizeBytes];
  int *ranks;
} rankMsg;

static rankMsg *rankmsg = NULL;
static CmmTable hostTable;


/* called on PE 0 */
static void cpuAffinityHandler(void *m)
{
  static int count = 0;
  hostnameMsg *rec;
  hostnameMsg *msg = (hostnameMsg *)m;
  hostnameMsg *tmpm;
  char str[128];
  int tag, tag1, pe;
  CmiAssert(rankmsg != NULL);

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
    rec = msg;
    CmmPut(hostTable, 1, &tag, msg);
  }
  rankmsg->ranks[pe] = rec->rank%rec->ncores;
  rec->rank ++;
  count ++;
  if (count == CmiNumPes()) {
    CmiPrintf("Cpuaffinity> %d unique compute nodes detected! \n", CmmEntries(hostTable));
    //hostnameMsg *tmpm;
    tag = CmmWildCard;
    while (tmpm = CmmGet(hostTable, 1, &tag, &tag1)) CmiFree(tmpm);
    CmmFree(hostTable);
    CmiSyncBroadcastAllAndFree(sizeof(rankMsg)+CmiNumPes()*sizeof(int), (void *)rankmsg);
  }
}

static int set_myaffinitity(int myrank)
{
  /* set cpu affinity */
#if CMK_SMP
  return set_thread_affinity(myrank);
#else
  return set_cpu_affinity(myrank);
  /* print_cpu_affinity(); */
#endif
}

/* called on each processor */
static void cpuAffinityRecvHandler(void *msg)
{
  int myrank;
  rankMsg *m = (rankMsg *)msg;
  m->ranks = (int *)((char*)m + sizeof(rankMsg));
  myrank = m->ranks[CmiMyPe()];

  /* CmiPrintf("[%d %d] rank: %d\n", CmiMyNode(), CmiMyPe(), myrank); */

  if (-1 != set_myaffinitity(myrank)) {
    DEBUGP(("Processor %d is bound to core #%d\n", CmiMyPe(), myrank));
  }
  else
    CmiPrintf("Processor %d set affinity failed!\n", CmiMyPe());

  CmiFree(m);
}

#if CMK_XT3
extern int getXT3NodeID(int mype, int numpes);
#endif

void CmiInitCPUAffinity(char **argv)
{
  static skt_ip_t myip;
  int ret;
  hostnameMsg  *msg;
  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
						"set cpu affinity");

  cpuAffinityHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityHandler);
  cpuAffinityRecvHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityRecvHandler);

  if (!affinity_flag) return;
  else if (CmiMyPe() == 0)
     CmiPrintf("Charm++> cpu affinity enabled! \n");

  if (CmiMyPe() >= CmiNumPes()) {
      /* comm thread either can float around, or pin down to the last rank.
         however it seems to be reportedly slower if it is floating */
    set_myaffinitity(num_cores()-1);
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
#if CMK_XT3
    ret = getXT3NodeID(CmiMyPe(), CmiNumPes());
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
  msg->ncores = num_cores();
  msg->rank = 0;
  CmiSyncSendAndFree(0, sizeof(hostnameMsg), (void *)msg);

  if (CmiMyPe() == 0) {
    int i;
    hostTable = CmmNew();
    rankmsg = (rankMsg *)CmiAlloc(sizeof(rankMsg)+CmiNumPes()*sizeof(int));
    CmiSetHandler((char *)rankmsg, cpuAffinityRecvHandlerIdx);
    rankmsg->ranks = (int *)((char*)rankmsg + sizeof(rankMsg));
    for (i=0; i<CmiNumPes(); i++) rankmsg->ranks[i] = 0;
  }
}

#else           /* not supporting affinity */

void CmiInitCPUAffinity(char **argv)
{
  int affinity_flag = CmiGetArgFlagDesc(argv,"+setcpuaffinity",
						"set cpu affinity");
  if (affinity_flag)
    CmiPrintf("sched_setaffinity() is not supported, +affinity_flag disabled.\n");
}

#endif
