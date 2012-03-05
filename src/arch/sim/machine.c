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



static void **McQueue;

int _Cmi_mype;
int _Cmi_numpes;


CsvDeclare(int, CsdStopCount);
CpvDeclare(void*, CmiLocalQueue);
CpvExtern(int, CcdNumChecks);
CpvExtern(int, disable_sys_msgs);

double CmiTimer();

static void CsiTimerInit();
static double CsiTimer();


void CmiDeliversInit()
{
}


int CsdScheduler(maxmsgs)
int maxmsgs;
{
  CmiError("Cannot call scheduling functions in SIM versions.\n");
  exit(0);
}

int CmiDeliverMsgs(maxmsgs)
int maxmsgs;
{
  CmiError("Cannot call scheduling functions in SIM versions.\n");
  exit(1);
}

void CmiDeliverSpecificMsg(handler)
int handler;
{
  CmiError("Cannot call scheduling functions in SIM versions.\n");
  exit(1);
}

CmiUniContextSwitch(i)
int i;
{
  _Cmi_mype = i; 
}

void CmiNotifyIdle()
{
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  struct timeval tv;
  tv.tv_sec=0; tv.tv_usec=5000;
  select(0,0,0,0,&tv);
#endif
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message)
{
  CmiError(message);
  exit(1);
}

void ConverseExit(void)
{
  exit(0);
}

/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
    char *buf;

    buf =  (char *) CmiAlloc(size);
    memcpy(buf,msg,size);
    sim_send_message(_Cmi_mype,buf,size,FALSE,destPE);
}




CmiCommHandle CmiAsyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
     CmiSyncSendFn(destPE, size, msg);
     return 0;
}



void CmiFreeSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
     sim_send_message(_Cmi_mype,msg,size,FALSE,destPE);
}



int CmiAsyncMsgSent(c)
CmiCommHandle c ;
{
    return 1;
}


void CmiReleaseCommHandle(c)
CmiCommHandle c ;
{
}


void CmiNodeBarrier()
{
}

void CmiNodeAllBarrier()
{
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal(){return NULL;}




/*********************** BROADCAST FUNCTIONS **********************/


void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
    int i;
    for(i=0; i<_Cmi_numpes; i++)
       if (i!= _Cmi_mype) CmiSyncSendFn(i,size,msg);
         
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
}


void CmiSyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
    int i;
    for(i=0; i<_Cmi_numpes; i++)
       if (i!= _Cmi_mype) CmiSyncSendFn(i,size,msg);
}


void CmiSyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
     int i;

     char *buf;

     for(i=0; i<_Cmi_numpes; i++)
        if (i!= _Cmi_mype) CmiSyncSendFn(i,size,msg);

     buf =  (char *) CmiAlloc(size);
     memcpy(buf,msg,size);
     CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),buf);
}



void CmiFreeBroadcastFn(size, msg)      /* ALL_EXCEPT_ME  */
int size;
char * msg;
{
    CmiSyncBroadcastFn(size, msg);
    CmiFree(msg);
}





CmiCommHandle CmiAsyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
        CmiSyncBroadcastFn(size, msg); 
	return 0 ;
}




CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
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
    CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,CmiNotifyIdle,NULL);
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

static void CsiTimerInit()
{
  times(&inittime);
}

static double CsiTimer()
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

static void CsiTimerInit()
{
  getrusage(0, &inittime);
}


static double CsiTimer() {
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

double CmiTimer()
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

double CmiWallTimer()
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

double CmiCpuTimer()
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

CmiNodeLock CmiCreateLock()
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
#include "sim.c"
#include "heap.c"
#include "net.c"
#include "simqmng.c"
#include "simrand.c"
