#include <stdio.h>
#include <math.h>
#include "machine.h"
#include "converse.h"

#if CMK_TIMER_SIM_USE_TIMES
#include <sys/times.h>
#include <sys/unistd.h>
#endif
#if CMK_TIMER_SIM_USE_GETRUSAGE
#include <sys/time.h>
#include <sys/resource.h>
#endif


void CthInit(char **);
void ConverseCommonInit(char **);
void ConverseCommonExit(void);

static void **McQueue;

int _Cmi_mype;
int _Cmi_numpes;


CsvDeclare(int, CsdStopCount);
CpvDeclare(void*, CmiLocalQueue);
CpvExtern(int, CcdNumChecks);
CpvExtern(int, disable_sys_msgs);

double CmiTimer(void);

static void CsiTimerInit(void);
static double CsiTimer(void);


void CmiDeliversInit(void)
{
}


int CsdScheduler(int maxmsgs)
{
  CmiError("Cannot call scheduling functions in SIM versions.\n");
  exit(0);
}

int CmiDeliverMsgs(int maxmsgs)
{
  CmiError("Cannot call scheduling functions in SIM versions.\n");
  exit(1);
}

void CmiDeliverSpecificMsg(int handler)
{
  CmiError("Cannot call scheduling functions in SIM versions.\n");
  exit(1);
}

CmiUniContextSwitch(int i)
{
  _Cmi_mype = i; 
}

void CmiNotifyIdle(void)
{
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  struct timeval tv;
  tv.tv_sec=0; tv.tv_usec=5000;
  select(0,0,0,0,&tv);
#endif
}

static void CmiNotifyIdleCcd(void *ignored1, double ignored2)
{
    CmiNotifyIdle();
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message, ...)
{
  char newmsg[256];
  va_list args;
  va_start(args, message);
  vsnprintf(newmsg, sizeof(newmsg), message, args);
  va_end(args);
  CmiError(newmsg);
  CmiError("\n");
  exit(1);
  CMI_NORETURN_FUNCTION_END
}

void ConverseExit(int exitcode)
{
  exit(exitcode);
}

/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(int destPE, int size, char *msg)
{
    char *buf;

    buf =  (char *) CmiAlloc(size);
    memcpy(buf,msg,size);
    sim_send_message(_Cmi_mype,buf,size,FALSE,destPE);
}




CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
     CmiSyncSendFn(destPE, size, msg);
     return 0;
}



void CmiFreeSendFn(int destPE, int size, char *msg)
{
     sim_send_message(_Cmi_mype,msg,size,FALSE,destPE);
}



int CmiAsyncMsgSent(CmiCommHandle c)
{
    return 1;
}


void CmiReleaseCommHandle(CmiCommHandle c)
{
}


void CmiNodeBarrier(void)
{
}

void CmiNodeAllBarrier(void)
{
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal(void){return NULL;}




/*********************** BROADCAST FUNCTIONS **********************/


void CmiFreeBroadcastAllFn(int size,  char *msg)
{
    int i;
    for(i=0; i<_Cmi_numpes; i++)
       if (i!= _Cmi_mype) CmiSyncSendFn(i,size,msg);
         
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}


void CmiSyncBroadcastFn(int size,  char *msg)	/* ALL_EXCEPT_ME  */
{
    int i;
    for(i=0; i<_Cmi_numpes; i++)
       if (i!= _Cmi_mype) CmiSyncSendFn(i,size,msg);
}


void CmiSyncBroadcastAllFn(int size,  char *msg)
{
     int i;

     char *buf;

     for(i=0; i<_Cmi_numpes; i++)
        if (i!= _Cmi_mype) CmiSyncSendFn(i,size,msg);

     buf =  (char *) CmiAlloc(size);
     memcpy(buf,msg,size);
     CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),buf);
}



void CmiFreeBroadcastFn(int size,  char *msg)      /* ALL_EXCEPT_ME  */
{
    CmiSyncBroadcastFn(size, msg);
    CmiFree(msg);
}

void CmiWithinNodeBroadcastFn(int size, char* msg) {
  int nodeFirst = CmiNodeFirst(CmiMyNode());
  int nodeLast = nodeFirst + CmiNodeSize(CmiMyNode());
  if (CMI_MSG_NOKEEP(msg)) {
    for (int i = nodeFirst; i < CmiMyPe(); i++) {
      CmiReference(msg);
      CmiFreeSendFn(i, size, msg);
    }
    for (int i = CmiMyPe() + 1; i < nodeLast; i++) {
      CmiReference(msg);
      CmiFreeSendFn(i, size, msg);
    }
  } else {
    for (int i = nodeFirst; i < CmiMyPe(); i++) {
      CmiSyncSendFn(i, size, msg);
    }
    for (int i = CmiMyPe() + 1; i < nodeLast; i++) {
      CmiSyncSendFn(i, size, msg);
    }
  }
  CmiSyncSendAndFree(CmiMyPe(), size, msg);
}

CmiCommHandle CmiAsyncBroadcastFn(int size,  char *msg)	/* ALL_EXCEPT_ME  */
{
        CmiSyncBroadcastFn(size, msg); 
	return 0 ;
}




CmiCommHandle CmiAsyncBroadcastAllFn(int size,  char *msg)
{
        CmiSyncBroadcastAll(size,msg);
	return 0 ;
}


/************************** SETUP ***********************************/

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usc, int initret)
{
  void simulate();
  int i, requested_npe;
  
  if ((usc)||(initret)) {
    fprintf(stderr,"ConverseInit in SIM version is limited:\n");
    fprintf(stderr," 1. User-Calls-Scheduler mode is not supported.\n");
    fprintf(stderr," 2. ConverseInit-Returns mode is not supported.\n");
    exit(1);
  }
  
  
  /* figure out number of processors required */
  
  i = 0; requested_npe = 0;
  CmiGetArgInt(argv,"+p",&requested_npe);
  if (requested_npe <= 0) {
    printf("Error: requested number of processors is invalid %d\n",
	   requested_npe);
    exit(1);
  }

  _Cmi_numpes = requested_npe;
  _Cmi_mype   = 0;

  McQueue = (void **) malloc(requested_npe * sizeof(void *)); 
  for(i=0; i<requested_npe; i++) McQueue[i] = CdsFifo_Create();
  CrnInit();
  sim_initialize("sim.param",requested_npe);
  
  CsiTimerInit();
  for(i=0; i<CmiNumPes(); i++) {
    CmiUniContextSwitch(i);
    CpvInitialize(void*, CmiLocalQueue);
    CpvAccess(CmiLocalQueue) = CdsFifo_Create();
    CthInit(argv);
    ConverseCommonInit(argv);
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE, CmiNotifyIdleCcd, NULL);
    argc=CmiGetArgc(argv);
    fn(argc, CmiCopyArgs(argv));
    CpvAccess(CsdStopFlag) = 0;
  }
  
  CsvAccess(CsdStopCount) = CmiNumPes();
  CmiUniContextSwitch(0);
  
  while (CsvAccess(CsdStopCount)) simulate();
  
  exit(0);
}

/* ********************************************************************* */
/*                      SIMULATOR                                        */
/* ********************************************************************* */



#if CMK_TIMER_SIM_USE_TIMES

static struct tms inittime;

static void CsiTimerInit(void)
{
  times(&inittime);
}

static double CsiTimer(void)
{
  double currenttime;
  int clk_tck;
    struct tms temp;

    times(&temp);
    clk_tck=sysconf(_SC_CLK_TCK);
    currenttime =
     (((temp.tms_utime - inittime.tms_utime)+
       (temp.tms_stime - inittime.tms_stime))*1.0)/clk_tck;
    return (currenttime);
}

#endif

#if CMK_TIMER_SIM_USE_GETRUSAGE

static struct rusage inittime;

static void CsiTimerInit(void)
{
  getrusage(0, &inittime);
}


static double CsiTimer(void) {
  double currenttime;

  struct rusage temp;
  getrusage(0, &temp);
  currenttime =
    (temp.ru_utime.tv_usec - inittime.ru_utime.tv_usec) * 0.000001+
      (temp.ru_utime.tv_sec - inittime.ru_utime.tv_sec) +
        (temp.ru_stime.tv_usec - inittime.ru_stime.tv_usec) * 0.000001+
          (temp.ru_stime.tv_sec - inittime.ru_stime.tv_sec) ;

  return (currenttime);
}

#endif



static double Csi_global_time;
static double Csi_start_time;

void CmiTimerInit(char **argv) { }

double CmiTimer(void)
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

double CmiWallTimer(void)
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

double CmiCpuTimer(void)
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

CmiNodeLock CmiCreateLock(void)
{
  CmiNodeLock lk = (CmiNodeLock)malloc(sizeof(int));
  *lk = 0;
  return lk;
}

void CmiLock(CmiNodeLock lk)
{
  while (*lk) CmiGetNonLocal();
  *lk = 1;
}

void CmiUnlock(CmiNodeLock lk)
{
  if (*lk==0) {
    CmiError("CmiNodeLock not locked, can't unlock.");
    exit(1);
  }
  *lk = 0;
}

int CmiTryLock(CmiNodeLock lk)
{
  if (*lk==0) { *lk=1; return 0; }
  return -1;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  free(lk);
}

#include "ext_func.h"
#include "sim.C"
#include "heap.C"
#include "net.C"
#include "simqmng.C"
#include "simrand.C"
