#include "converse.h"
#include "sockRoutines.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define _GNU_SOURCE
#include <sched.h>
long sched_setaffinity(pid_t pid, unsigned int len, unsigned long *user_mask_ptr);
long sched_getaffinity(pid_t pid, unsigned int len, unsigned long *user_mask_ptr);

#include <stdlib.h>
#include <stdio.h>

#ifdef _MSC_VER
#include <windows.h>
#include <winbase.h>
#endif

/* needed for call to sysconf() */
#if defined(__sun) || defined(ARCH_IRIX6) || defined(ARCH_IRIX6_64) || defined(ARCH_LINUX) || defined(ARCH_LINUXALHPA) || defined(ARCH_LINUXAMD64) || defined(ARCH_LINUXIA64) || defined(ARCH_LINUXPPC) || defined(ARCH_LINUXPPC64) || defined(_CRAY) || defined(__osf__) || defined(ARCH_AIX4) || defined(ARCH_AIX5) || defined(ARCH_AIX5_64)
#include<unistd.h>
#endif

#if defined(__APPLE__) && defined(VMDTHREADS)
#include <Carbon/Carbon.h> /* Carbon APIs for Multiprocessing */
#endif

#if defined(ARCH_HPUX11) ||  defined(ARCH_HPUX10)
#include <sys/mpctl.h>
#endif


int num_cores(void) {
  int a=1;

  // Allow the user to override the number of CPUs for use
  // in scalability testing, debugging, etc.
  char *forcecount = getenv("VMDFORCECPUCOUNT");
  if (forcecount != NULL) {
    if (sscanf(forcecount, "%d", &a) == 1) {
      return a; // if we got a valid count, return it
    } else {
      a=1;      // otherwise use the real available hardware CPU count
    }
  }

#if defined(__APPLE__) 
  a = MPProcessorsScheduled(); /* Number of active/running CPUs */
#endif

#ifdef _MSC_VER
  struct _SYSTEM_INFO sysinfo;
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

printf("[%d] %d\n", sysconf(_SC_NPROCESSORS_ONLN), a);
  return a;
}

/* This implementation assumes the default x86 CPU mask size used by Linux */
/* For a large SMP machine, this code should be changed to use a variable sized   */
/* CPU affinity mask buffer instead, as the present code will fail beyond 32 CPUs */
int set_cpu_affinity(int cpuid) {
  unsigned long mask = 0xffffffff;
  unsigned int len = sizeof(mask);

  /* set the affinity mask if possible */
  if ((cpuid / 8) > len) {
    printf("Mask size too small to handle requested CPU ID\n");
    return -1;
  } else {
    mask = 1 << cpuid;   /* set the affinity mask exclusively to one CPU */
  }

  /* PID 0 refers to the current process */
  if (sched_setaffinity(0, len, &mask) < 0) {
    perror("sched_setaffinity");
    return -1;
  }

  return 0;
}

/* This implementation assumes the default x86 CPU mask size used by Linux */
/* For a large SMP machine, this code should be changed to use a variable sized   */
/* CPU affinity mask buffer instead, as the present code will fail beyond 32 CPUs */
int print_cpu_affinity() {
  unsigned long mask;
  unsigned int len = sizeof(mask);

  /* PID 0 refers to the current process */
  if (sched_getaffinity(0, len, &mask) < 0) {
    perror("sched_getaffinity");
    return -1;
  }

  printf("CPU affinity mask is: %08lx\n", mask);

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
  char str[128];
  int tag, tag1, pe;
  CmiAssert(rankmsg != NULL);

  skt_print_ip(str, msg->ip);
  printf("hostname: %d %s\n", msg->pe, str);
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
    hostnameMsg *m;
    tag = CmmWildCard;
    while (m = CmmGet(hostTable, 1, &tag, &tag1)) CmiFree(m);
    CmmFree(hostTable);
    CmiSyncBroadcastAllAndFree(sizeof(rankMsg)+CmiNumPes()*sizeof(int), (void *)rankmsg);
  }
}

static void cpuAffinityRecvHandler(void *msg)
{
  int myrank;
  rankMsg *m = (rankMsg *)msg;
  m->ranks = (char*)m + sizeof(rankMsg); 
  myrank = m->ranks[CmiMyPe()];
  CmiPrintf("[%d %d] rank: %d\n", CmiMyNode(), CmiMyPe(), myrank);

  /* set cpu affinity */
  if (set_cpu_affinity(myrank) != -1)
    CmiPrintf("Processor %d is bound to core %d\n", CmiMyPe(), myrank);
  print_cpu_affinity();
  CmiFree(m);
}

void CmiInitCPUAffinity()
{
  hostnameMsg  *msg;
  cpuAffinityHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityHandler);
  cpuAffinityRecvHandlerIdx =
       CmiRegisterHandler((CmiHandler)cpuAffinityRecvHandler);
  if (CmiMyPe() >= CmiNumPes()) return;    /* comm thread return */
#if 0
  if (gethostname(hostname, 999)!=0) {
      strcpy(hostname, "");
  }
#endif
    /* get my ip address */
  skt_ip_t myip = skt_my_ip();

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
    rankmsg->ranks = (char*)rankmsg + sizeof(rankMsg); /* (int *)(void **)&rankmsg->ranks + 1; */
    for (i=0; i<CmiNumPes(); i++) rankmsg->ranks[i] = 0;
  }
}

