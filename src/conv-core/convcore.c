#include <stdio.h>
#include <string.h>
#include "converse.h"
#include "trace.h"
#include <errno.h>

#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
#include <sys/types.h>
#include <sys/time.h>
#endif

#if CMK_TIMER_USE_TIMES
#include <sys/times.h>
#include <limits.h>
#include <unistd.h>
#endif

#if CMK_TIMER_USE_GETRUSAGE
#include <sys/time.h>
#include <sys/resource.h>
#endif


/*****************************************************************************
 *
 * Unix Stub Functions
 *
 ****************************************************************************/

#if CMK_STRERROR_USE_SYS_ERRLIST
extern char *sys_errlist[];
char *strerror(i) int i; { return sys_errlist[i]; }
#endif

#ifdef MEMMONITOR
typedef unsigned long mmulong;
CpvDeclare(mmulong,MemoryUsage);
CpvDeclare(mmulong,HiWaterMark);
CpvDeclare(mmulong,ReportedHiWaterMark);
CpvDeclare(int,AllocCount);
CpvDeclare(int,BlocksAllocated);
#endif

#if CMK_SIGHOLD_USE_SIGMASK
#include <signal.h>
int sighold(sig) int sig;
{ if (sigblock(sigmask(sig)) < 0) return -1;
  else return 0; }
int sigrelse(sig) int sig;
{ if (sigsetmask(sigblock(0)&(~sigmask(sig))) < 0) return -1;
  else return 0; }
#endif

#define MAX_HANDLERS 512

#if CMK_NODE_QUEUE_AVAILABLE
void  *CmiGetNonLocalNodeQ();
#endif
void  *CmiGetNonLocal();
void   CmiNotifyIdle();

CpvDeclare(int, disable_sys_msgs);
CpvExtern(int,    CcdNumChecks) ;
CpvDeclare(void*, CsdSchedQueue);
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(void*, CsdNodeQueue);
CsvDeclare(CmiNodeLock, NodeQueueLock);
#endif
CpvDeclare(int,   CsdStopFlag);


/*****************************************************************************
 *
 * Some of the modules use this in their argument parsing.
 *
 *****************************************************************************/

static char *DeleteArg(argv)
  char **argv;
{
  char *res = argv[0];
  if (res==0) { CmiError("Bad arglist."); exit(1); }
  while (*argv) { argv[0]=argv[1]; argv++; }
  return res;
}


/**
 * Global variable for Trace in converse (moved from charm)
 */

CpvDeclare(int, CtrRecdTraceMsg);
CpvDeclare(int, traceOn);
CpvDeclare(int, CtrLogBufSize);

/*****************************************************************************
 *
 * Statistics: currently, the following statistics are not updated by converse.
 *
 *****************************************************************************/

CpvDeclare(int, CstatsMaxChareQueueLength);
CpvDeclare(int, CstatsMaxForChareQueueLength);
CpvDeclare(int, CstatsMaxFixedChareQueueLength);
CpvStaticDeclare(int, CstatPrintQueueStatsFlag);
CpvStaticDeclare(int, CstatPrintMemStatsFlag);

void CstatsInit(argv)
char **argv;
{
  int argc;
  char **origArgv = argv;
  int trace = 1;

#ifdef MEMMONITOR
  CpvInitialize(mmulong,MemoryUsage);
  CpvAccess(MemoryUsage) = 0;
  CpvInitialize(mmulong,HiWaterMark);
  CpvAccess(HiWaterMark) = 0;
  CpvInitialize(mmulong,ReportedHiWaterMark);
  CpvAccess(ReportedHiWaterMark) = 0;
  CpvInitialize(int,AllocCount);
  CpvAccess(AllocCount) = 0;
  CpvInitialize(int,BlocksAllocated);
  CpvAccess(BlocksAllocated) = 0;
#endif

  CpvInitialize(int, CtrRecdTraceMsg);
  CpvInitialize(int, CtrLogBufSize);
  CpvInitialize(int, CstatsMaxChareQueueLength);
  CpvInitialize(int, CstatsMaxForChareQueueLength);
  CpvInitialize(int, CstatsMaxFixedChareQueueLength);
  CpvInitialize(int, CstatPrintQueueStatsFlag);
  CpvInitialize(int, CstatPrintMemStatsFlag);

  CpvAccess(CtrLogBufSize) = 100000;
  CpvAccess(CstatsMaxChareQueueLength) = 0;
  CpvAccess(CstatsMaxForChareQueueLength) = 0;
  CpvAccess(CstatsMaxFixedChareQueueLength) = 0;
  CpvAccess(CstatPrintQueueStatsFlag) = 0;
  CpvAccess(CstatPrintMemStatsFlag) = 0;

  while (*argv) {
    if (strcmp(*argv, "+mems") == 0) {
      CpvAccess(CstatPrintMemStatsFlag)=1;
      DeleteArg(argv);
    } else
    if (strcmp(*argv, "+qs") == 0) {
      CpvAccess(CstatPrintQueueStatsFlag)=1;
      DeleteArg(argv);
    } else if (strcmp(*argv, "+logsize") == 0) {
      int logsize;
      DeleteArg(argv);
      sscanf(*argv, "%d", &logsize);
      CpvAccess(CtrLogBufSize) = logsize;
      DeleteArg(argv);
    } else if (strcmp(*argv, "+traceoff") == 0) {
      trace = 0;
      DeleteArg(argv);
    } else
    argv++;
  }

  argc = 0; argv=origArgv;
  for(argc=0;argv[argc];argc++);
  traceModuleInit(&argc, argv);
  CpvAccess(traceOn) = (CpvAccess(traceOn) && trace);
  log_init();
}

int CstatMemory(i)
int i;
{
  return 0;
}

int CstatPrintQueueStats()
{
  return CpvAccess(CstatPrintQueueStatsFlag);
}

int CstatPrintMemStats()
{
  return CpvAccess(CstatPrintMemStatsFlag);
}

/*****************************************************************************
 *
 * Cmi handler registration
 *
 *****************************************************************************/

CpvDeclare(CmiHandler*, CmiHandlerTable);
CpvStaticDeclare(int  , CmiHandlerCount);
CpvStaticDeclare(int  , CmiHandlerLocal);
CpvStaticDeclare(int  , CmiHandlerGlobal);
CpvDeclare(int,         CmiHandlerMax);

void CmiNumberHandler(n, h)
int n; CmiHandler h;
{
  CmiHandler *tab;
  int         max = CpvAccess(CmiHandlerMax);

  tab = CpvAccess(CmiHandlerTable);
  if (n >= max) {
    int newmax = ((n<<1)+10);
    int bytes = max*sizeof(CmiHandler);
    int newbytes = newmax*sizeof(CmiHandler);
    CmiHandler *new = (CmiHandler*)CmiAlloc(newbytes);
    memcpy(new, tab, bytes);
    memset(((char *)new)+bytes, 0, (newbytes-bytes));
    free(tab); tab=new;
    CpvAccess(CmiHandlerTable) = tab;
    CpvAccess(CmiHandlerMax) = newmax;
  }
  tab[n] = h;
}

int CmiRegisterHandler(h)
CmiHandler h;
{
  int Count = CpvAccess(CmiHandlerCount);
  CmiNumberHandler(Count, h);
  CpvAccess(CmiHandlerCount) = Count+3;
  return Count;
}

int CmiRegisterHandlerLocal(h)
CmiHandler h;
{
  int Local = CpvAccess(CmiHandlerLocal);
  CmiNumberHandler(Local, h);
  CpvAccess(CmiHandlerLocal) = Local+3;
  return Local;
}

int CmiRegisterHandlerGlobal(h)
CmiHandler h;
{
  int Global = CpvAccess(CmiHandlerGlobal);
  if (CmiMyPe()!=0) 
    CmiError("CmiRegisterHandlerGlobal must only be called on PE 0.\n");
  CmiNumberHandler(Global, h);
  CpvAccess(CmiHandlerGlobal) = Global+3;
  return Global;
}

static void CmiHandlerInit()
{
  CpvInitialize(CmiHandler *, CmiHandlerTable);
  CpvInitialize(int         , CmiHandlerCount);
  CpvInitialize(int         , CmiHandlerLocal);
  CpvInitialize(int         , CmiHandlerGlobal);
  CpvInitialize(int         , CmiHandlerMax);
  CpvAccess(CmiHandlerCount)  = 0;
  CpvAccess(CmiHandlerLocal)  = 1;
  CpvAccess(CmiHandlerGlobal) = 2;
  CpvAccess(CmiHandlerMax) = 100;
  CpvAccess(CmiHandlerTable) = (CmiHandler *)malloc(100*sizeof(CmiHandler)) ;
}


/******************************************************************************
 *
 * CmiTimer
 *
 * Here are two possible implementations of CmiTimer.  Some machines don't
 * select either, and define the timer in machine.c instead.
 *
 *****************************************************************************/

#if CMK_TIMER_USE_TIMES

CpvStaticDeclare(double, clocktick);
CpvStaticDeclare(int,inittime_wallclock);
CpvStaticDeclare(int,inittime_virtual);

void CmiTimerInit()
{
  struct tms temp;
  CpvInitialize(double, clocktick);
  CpvInitialize(int, inittime_wallclock);
  CpvInitialize(int, inittime_virtual);
  CpvAccess(inittime_wallclock) = times(&temp);
  CpvAccess(inittime_virtual) = temp.tms_utime + temp.tms_stime;
  CpvAccess(clocktick) = 1.0 / (sysconf(_SC_CLK_TCK));
}

double CmiWallTimer()
{
  struct tms temp;
  double currenttime;
  int now;

  now = times(&temp);
  currenttime = (now - CpvAccess(inittime_wallclock)) * CpvAccess(clocktick);
  return (currenttime);
}

double CmiCpuTimer()
{
  struct tms temp;
  double currenttime;
  int now;

  times(&temp);
  now = temp.tms_stime + temp.tms_utime;
  currenttime = (now - CpvAccess(inittime_virtual)) * CpvAccess(clocktick);
  return (currenttime);
}

double CmiTimer()
{
  return CmiCpuTimer();
}

#endif

#if CMK_TIMER_USE_GETRUSAGE

CpvStaticDeclare(double, inittime_wallclock);
CpvStaticDeclare(double, inittime_virtual);

void CmiTimerInit()
{
  struct timeval tv;
  struct rusage ru;
  CpvInitialize(double, inittime_wallclock);
  CpvInitialize(double, inittime_virtual);
  gettimeofday(&tv,0);
  CpvAccess(inittime_wallclock) = (tv.tv_sec * 1.0) + (tv.tv_usec*0.000001);
  getrusage(0, &ru); 
  CpvAccess(inittime_virtual) =
    (ru.ru_utime.tv_sec * 1.0)+(ru.ru_utime.tv_usec * 0.000001) +
    (ru.ru_stime.tv_sec * 1.0)+(ru.ru_stime.tv_usec * 0.000001);
}

double CmiCpuTimer()
{
  struct rusage ru;
  double currenttime;

  getrusage(0, &ru);
  currenttime =
    (ru.ru_utime.tv_sec * 1.0)+(ru.ru_utime.tv_usec * 0.000001) +
    (ru.ru_stime.tv_sec * 1.0)+(ru.ru_stime.tv_usec * 0.000001);
  return currenttime - CpvAccess(inittime_virtual);
}

double CmiWallTimer()
{
  struct timeval tv;
  double currenttime;

  gettimeofday(&tv,0);
  currenttime = (tv.tv_sec * 1.0) + (tv.tv_usec * 0.000001);
  return currenttime - CpvAccess(inittime_wallclock);
}

double CmiTimer()
{
  return CmiCpuTimer();
}

#endif


/******************************************************************************
 *
 * CmiEnableAsyncIO
 *
 * The net and tcp versions use a bunch of unix processes talking to each
 * other via file descriptors.  We need for a signal SIGIO to be generated
 * each time a message arrives, making it possible to write a signal
 * handler to handle the messages.  The vast majority of unixes can,
 * in fact, do this.  However, there isn't any standard for how this is
 * supposed to be done, so each version of UNIX has a different set of
 * calls to turn this signal on.  So, there is like one version here for
 * every major brand of UNIX.
 *
 *****************************************************************************/

#if CMK_ASYNC_USE_FIOASYNC_AND_FIOSETOWN
#include <sys/filio.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  int pid = getpid();
  int async = 1;
  if ( ioctl(fd, FIOSETOWN, &pid) < 0  ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( ioctl(fd, FIOASYNC, &async) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

#if CMK_ASYNC_USE_FIOASYNC_AND_SIOCSPGRP
#include <sys/filio.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  int pid = -getpid();
  int async = 1;
  if ( ioctl(fd, SIOCSPGRP, &pid) < 0  ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( ioctl(fd, FIOASYNC, &async) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

#if CMK_ASYNC_USE_FIOSSAIOSTAT_AND_FIOSSAIOOWN
#include <sys/ioctl.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  int pid = getpid();
  int async = 1;
  if ( ioctl(fd, FIOSSAIOOWN, &pid) < 0  ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( ioctl(fd, FIOSSAIOSTAT, &async) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

#if CMK_ASYNC_USE_F_SETFL_AND_F_SETOWN
#include <fcntl.h>
void CmiEnableAsyncIO(fd)
int fd;
{
  if ( fcntl(fd, F_SETOWN, getpid()) < 0 ) {
    CmiError("setting socket owner: %s\n", strerror(errno)) ;
    exit(1);
  }
  if ( fcntl(fd, F_SETFL, FASYNC) < 0 ) {
    CmiError("setting socket async: %s\n", strerror(errno)) ;
    exit(1);
  }
}
#endif

#if CMK_SIGNAL_USE_SIGACTION
#include <signal.h>
void CmiSignal(sig1, sig2, sig3, handler)
int sig1, sig2, sig3;
void (*handler)();
{
  struct sigaction in, out ;
  in.sa_handler = handler;
  sigemptyset(&in.sa_mask);
  if (sig1) sigaddset(&in.sa_mask, sig1);
  if (sig2) sigaddset(&in.sa_mask, sig2);
  if (sig3) sigaddset(&in.sa_mask, sig3);
  in.sa_flags = 0;
  if (sig1) if (sigaction(sig1, &in, &out)<0) exit(1);
  if (sig2) if (sigaction(sig2, &in, &out)<0) exit(1);
  if (sig3) if (sigaction(sig3, &in, &out)<0) exit(1);
}
#endif

#if CMK_SIGNAL_USE_SIGACTION_WITH_RESTART
#include <signal.h>
void CmiSignal(sig1, sig2, sig3, handler)
int sig1, sig2, sig3;
void (*handler)();
{
  struct sigaction in, out ;
  in.sa_handler = handler;
  sigemptyset(&in.sa_mask);
  if (sig1) sigaddset(&in.sa_mask, sig1);
  if (sig2) sigaddset(&in.sa_mask, sig2);
  if (sig3) sigaddset(&in.sa_mask, sig3);
  in.sa_flags = SA_RESTART;
  if (sig1) if (sigaction(sig1, &in, &out)<0) exit(1);
  if (sig2) if (sigaction(sig2, &in, &out)<0) exit(1);
  if (sig3) if (sigaction(sig3, &in, &out)<0) exit(1);
}
#endif

#if CMK_DEBUG_MODE

CpvDeclare(int, freezeModeFlag);
CpvDeclare(int, continueFlag);
CpvDeclare(int, stepFlag);
CpvDeclare(void *, debugQueue);
unsigned int freezeIP;
int freezePort;
char* breakPointHeader;
char* breakPointContents;

static void CpdDebugHandler(char *msg)
{
  char *normMsg;
  char *reply, *temp;
  int index;
  
  if(CcsIsRemoteRequest()) {
    char name[128];
    unsigned int ip, port;
    CcsCallerId(&ip, &port);
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);
    reply = NULL;

    if (strcmp(name, "freeze") == 0) {
      CpdFreeze();
      msgListCleanup();
      msgListCache();
      CmiPrintf("freeze received\n");
    }
    else if (strcmp(name, "unfreeze") == 0) {
      CpdUnFreeze();
      msgListCleanup();
      CmiPrintf("unfreeze received\n");
    }
    else if (strcmp(name, "getObjectList") == 0){
      CmiPrintf("getObjectList received\n");
      reply = getObjectList();
      CmiPrintf("list obtained");
      if(reply == NULL){
	CmiPrintf("list empty");
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      }
      else{
	CmiPrintf("list : %s\n", reply);
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if(strncmp(name,"getObjectContents",strlen("getObjectContents"))==0){
      CmiPrintf("getObjectContents received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getObjectContents(index);
      CmiPrintf("Object Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strcmp(name, "getMsgListSched") == 0){
      CmiPrintf("getMsgListSched received\n");
      reply = getMsgListSched();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListFIFO") == 0){
      CmiPrintf("getMsgListFIFO received\n");
      reply = getMsgListFIFO();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListPCQueue") == 0){
      CmiPrintf("getMsgListPCQueue received\n");
      reply = getMsgListPCQueue();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getMsgListDebug") == 0){
      CmiPrintf("getMsgListDebug received\n");
      reply = getMsgListDebug();
      if(reply == NULL)
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      else{
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if(strncmp(name,"getMsgContentsSched",strlen("getMsgContentsSched"))==0){
      CmiPrintf("getMsgContentsSched received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsSched(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if(strncmp(name,"getMsgContentsFIFO",strlen("getMsgContentsFIFO"))==0){
      CmiPrintf("getMsgContentsFIFO received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsFIFO(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "getMsgContentsPCQueue", strlen("getMsgContentsPCQueue")) == 0){
      CmiPrintf("getMsgContentsPCQueue received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsPCQueue(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "getMsgContentsDebug", strlen("getMsgContentsDebug")) == 0){
      CmiPrintf("getMsgContentsDebug received\n");
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &index);
      reply = getMsgContentsDebug(index);
      CmiPrintf("Message Contents : %s\n", reply);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    } 
    else if (strncmp(name, "step", strlen("step")) == 0){
      CmiPrintf("step received\n");
      CpvAccess(stepFlag) = 1;
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &freezePort);
      freezeIP = ip;
      CpdUnFreeze();
    }
    else if (strncmp(name, "continue", strlen("continue")) == 0){
      CmiPrintf("continue received\n");
      CpvAccess(continueFlag) = 1;
      temp = strstr(name, "#");
      temp++;
      sscanf(temp, "%d", &freezePort);
      freezeIP = ip;
      CpdUnFreeze();
    }
    else if (strcmp(name, "getBreakStepContents") == 0){
      CmiPrintf("getBreakStepContents received\n");
      if(breakPointHeader == 0){
	CcsSendReply(ip, port, strlen("$") + 1, "$");
      }
      else{
	reply = (char *)malloc(strlen(breakPointHeader) + strlen(breakPointContents) + 1);
	strcpy(reply, breakPointHeader);
	strcat(reply, "@");
	strcat(reply, breakPointContents);
	CcsSendReply(ip, port, strlen(reply) + 1, reply);
	free(reply);
      }
    }
    else if (strcmp(name, "getSymbolTableInfo") == 0){
      CmiPrintf("getSymbolTableInfo received");
      reply = getSymbolTableInfo();
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      reply = getBreakPoints();
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
    }
    else if (strncmp(name, "setBreakPoint", strlen("setBreakPoint")) == 0){
      CmiPrintf("setBreakPoint received\n");
      temp = strstr(name, "#");
      temp++;
      setBreakPoints(temp);
    }
    else if (strncmp(name, "backtrace", strlen("backtrace")) == 0){
      CmiPrintf("backtrace received\n");
      /*
      call the necessary function (result goes into reply)
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
      */
    }
    else if (strcmp(name, "quit") == 0){
      CsdExitScheduler();
    }
    else{
      CmiPrintf("incorrect command:%s received,len=%d\n",name,strlen(name));
    }
  }
}

void *FIFO_Create(void);

void CpdInit(void)
{
  CpvInitialize(int, freezeModeFlag);
  CpvAccess(freezeModeFlag) = 0;

  CpvInitialize(int, continueFlag);
  CpvInitialize(int, stepFlag);
  CpvAccess(continueFlag) = 0;
  CpvAccess(stepFlag) = 0;

  CpvInitialize(void *, debugQueue);
  CpvAccess(debugQueue) = FIFO_Create();
    
  CpdInitializeObjectTable();
  CpdInitializeHandlerArray();
  CpdInitializeBreakPoints();

  CcsRegisterHandler("DebugHandler", CpdDebugHandler);
}  

void CpdFreeze(void)
{
  CpvAccess(freezeModeFlag) = 1;
}  

void CpdUnFreeze(void)
{
  CpvAccess(freezeModeFlag) = 0;
}  

#endif

#if CMK_WEB_MODE

unsigned int appletIP;
unsigned int appletPort;

int countMsgs;
int *valueArray;
CpvDeclare(int, CWebPerformanceDataCollectionHandlerIndex);
CpvDeclare(int, CWebHandlerIndex);

#define WEB_INTERVAL 2000

void sendDataFunction(){
  char *reply;
  int i;
  
  /* NOTE : This needs to be made something BETTER */
  reply = (char *)malloc(100 * sizeof(char));
  strcpy(reply, "");
  
  /* create string to send */
  for(i = 0; i < CmiNumPes(); i++)
    sprintf(reply, "%s %d", reply, valueArray[i]);
  
  /* Do the CcsSendReply */
  CcsSendReply(appletIP, appletPort, strlen(reply) + 1, reply);
  CmiPrintf("reply = %s\n", reply);
  free(reply);
  
  countMsgs = 0;
}

void CWebPerformanceDataCollectionHandler(char *msg){
  int src;
  int value;

  if(CmiMyPe() != 0){
    CmiError("Wrong processor....\n");
    /* CmiAbort(); */
  }
  src = ((int *)(msg + CmiMsgHeaderSizeBytes))[0];
  value = ((int *)(msg + CmiMsgHeaderSizeBytes))[1];
  valueArray[src] = value;
  countMsgs++;

  CmiPrintf("In handler for message totalling..%d..\n", CmiMyPe());

  if(countMsgs == CmiNumPes()){
    CmiPrintf("Calling sendDataFunction...%d..\n", CmiMyPe());
    sendDataFunction();
  }
}

static void CWebPerformanceDataFunction(){
  char *msg;
  int randNum;
  int i;
  int msgSize;

  /* some initialization of seed */
  srand(CmiMyPe());

  randNum = CqsLength(CpvAccess(CsdSchedQueue));
  
  msgSize = 2 * sizeof(int) + CmiMsgHeaderSizeBytes;
  msg = (char *)CmiAlloc(msgSize);
  ((int *)(msg + CmiMsgHeaderSizeBytes))[0] = CmiMyPe();
  ((int *)(msg + CmiMsgHeaderSizeBytes))[1] = randNum;
  CmiSetHandler(msg, CpvAccess(CWebPerformanceDataCollectionHandlerIndex));
  CmiSyncSendAndFree(0, msgSize, msg);
  
  CcdCallFnAfter(CWebPerformanceDataFunction, 0, WEB_INTERVAL);
}

static void CWebHandlerOther(char *msg)
{
  /* Ordinary converse message */
  CmiPrintf("getStuff in Processor %d\n", CmiMyPe());
  CcdCallFnAfter(CWebPerformanceDataFunction, 0, WEB_INTERVAL);
}

static void CWebHandler(char *msg){
  int msgSize;
  char *getStuffMsg;
  int i;

  if(CcsIsRemoteRequest()) {
    char name[32];
    unsigned int ip, port;

    CcsCallerId(&ip, &port);
    sscanf(msg+CmiMsgHeaderSizeBytes, "%s", name);

    if(strcmp(name, "getStuff") == 0){
      appletIP = ip;
      appletPort = port;
      valueArray = (int *)malloc(sizeof(int) * CmiNumPes());
      CcdCallFnAfter(CWebPerformanceDataFunction, 0, WEB_INTERVAL);

      CmiPrintf("After setting the Ccd function..\n");

      /*
      reply = (char *)malloc(10 * sizeof(char));
      randNum = random();
      sprintf(reply, "%d", randNum);
      CcsSendReply(ip, port, strlen(reply) + 1, reply);
      free(reply);
      */
      
      for(i = 1; i < CmiNumPes(); i++){
	CmiPrintf("Forwarding message to processor %d\n", i);
	msgSize = CmiMsgHeaderSizeBytes + sizeof(char);
	getStuffMsg = (char *)CmiAlloc(msgSize);
	CmiSetHandler(getStuffMsg, CpvAccess(CWebHandlerIndex));
	CmiSyncSendAndFree(i, msgSize, getStuffMsg);
      }
    }
    else{
      CmiPrintf("incorrect command:%s received, len=%d\n",name,strlen(name));
    }
  }
  else{
    /* Ordinary converse message */
    CmiPrintf("getStuff in Processor %d\n", CmiMyPe());
    CcdCallFnAfter(CWebPerformanceDataFunction, 0, WEB_INTERVAL);
  }
}

void CWebInit(void)
{
  CcsRegisterHandler("MonitorHandler", CWebHandler);

  CpvInitialize(int, CWebHandlerIndex);
  CpvAccess(CWebHandlerIndex) = CmiRegisterHandler(CWebHandler);

  CpvInitialize(int, CWebPerformanceDataCollectionHandlerIndex);
  CpvAccess(CWebPerformanceDataCollectionHandlerIndex) = 
    CmiRegisterHandler(CWebPerformanceDataCollectionHandler);
}
    


#endif

/*****************************************************************************
 *
 * The following is the CsdScheduler function.  A common
 * implementation is provided below.  The machine layer can provide an
 * alternate implementation if it so desires.
 *
 * void CmiDeliversInit()
 *
 *      - CmiInit promises to call this before calling CmiDeliverMsgs
 *        or any of the other functions in this section.
 *
 * int CmiDeliverMsgs(int maxmsgs)
 *
 *      - CmiDeliverMsgs will retrieve up to maxmsgs that were transmitted
 *        with the Cmi, and will invoke their handlers.  It does not wait
 *        if no message is unavailable.  Instead, it returns the quantity
 *        (maxmsgs-delivered), where delivered is the number of messages it
 *        delivered.
 *
 * void CmiDeliverSpecificMsg(int handlerno)
 *
 *      - Waits for a message with the specified handler to show up, then
 *        invokes the message's handler.  Note that unlike CmiDeliverMsgs,
 *        This function _does_ wait.
 *
 * void CmiGrabBuffer(void **bufptrptr)
 *
 *      - When CmiDeliverMsgs or CmiDeliverSpecificMsgs calls a handler,
 *        the handler receives a pointer to a buffer containing the message.
 *        The buffer does not belong to the handler, eg, the handler may not
 *        free the buffer.  Instead, the buffer will be automatically reused
 *        or freed as soon as the handler returns.  If the handler wishes to
 *        keep a copy of the data after the handler returns, it may do so by
 *        calling CmiGrabBuffer and passing it a pointer to a variable which
 *        in turn contains a pointer to the system buffer.  The variable will
 *        be updated to contain a pointer to a handler-owned buffer containing
 *        the same data as before.  The handler then has the responsibility of
 *        making sure the buffer eventually gets freed.  Example:
 *
 * void myhandler(void *msg)
 * {
 *    CmiGrabBuffer(&msg);      // Claim ownership of the message buffer
 *    ... rest of handler ...
 *    CmiFree(msg);             // I have the right to free it or
 *                              // keep it, as I wish.
 * }
 *
 *
 * For this common implementation to work, the machine layer must provide the
 * following:
 *
 * void *CmiGetNonLocal()
 *
 *      - returns a message just retrieved from some other PE, not from
 *        local.  If no such message exists, returns 0.
 *
 * CpvExtern(FIFO_Queue, CmiLocalQueue);
 *
 *      - a FIFO queue containing all messages from the local processor.
 *
 *****************************************************************************/

CpvDeclare(CmiHandler, CsdNotifyIdle);
CpvDeclare(CmiHandler, CsdNotifyBusy);
CpvDeclare(int, CsdStopNotifyFlag);
CpvStaticDeclare(int, CsdIdleDetectedFlag);

void CsdEndIdle()
{
  if(CpvAccess(CsdIdleDetectedFlag)) {
    CpvAccess(CsdIdleDetectedFlag) = 0;
    if(!CpvAccess(CsdStopNotifyFlag)) {
      (CpvAccess(CsdNotifyBusy))();
      if(CpvAccess(traceOn))
        trace_end_idle();
    }
  }
}

void CsdBeginIdle()
{
  if (!CpvAccess(CsdIdleDetectedFlag)) {
    CpvAccess(CsdIdleDetectedFlag) = 1;
    if(!CpvAccess(CsdStopNotifyFlag)) {
      (CpvAccess(CsdNotifyIdle))();
      if(CpvAccess(traceOn))
        trace_begin_idle();
    }
  }

  CmiNotifyIdle();
  CcdRaiseCondition(CcdPROCESSORIDLE) ;
}
  
#if CMK_CMIDELIVERS_USE_COMMON_CODE

CtvStaticDeclare(int, CmiBufferGrabbed);

void CmiGrabBuffer(void **bufptrptr)
{
  CtvAccess(CmiBufferGrabbed) = 1;
}

void CmiHandleMessage(void *msg)
{
#if CMK_DEBUG_MODE
  char *freezeReply;
  int fd;

  extern int skt_connect(int, int, int);
  extern void writeall(int, char *, int);
#endif

  CtvAccess(CmiBufferGrabbed) = 0;

#if CMK_DEBUG_MODE
  
  if(CpvAccess(continueFlag) && (isBreakPoint((char *)msg))) {

    if(breakPointHeader != 0){
      free(breakPointHeader);
      breakPointHeader = 0;
    }
    if(breakPointContents != 0){
      free(breakPointContents);
      breakPointContents = 0;
    }
    
    breakPointHeader = genericViewMsgFunction((char *)msg, 0);
    breakPointContents = genericViewMsgFunction((char *)msg, 1);

    CmiPrintf("BREAKPOINT REACHED :\n");
    CmiPrintf("Header : %s\nContents : %s\n", breakPointHeader, breakPointContents);

    /* Freeze and send a message back */
    CpdFreeze();
    freezeReply = (char *)malloc(strlen("freezing@")+strlen(breakPointHeader)+1);
    sprintf(freezeReply, "freezing@%s", breakPointHeader);
    fd = skt_connect(freezeIP, freezePort, 120);
    if(fd > 0){
      writeall(fd, freezeReply, strlen(freezeReply) + 1);
      close(fd);
    } else {
      CmiPrintf("unable to connect");
    }
    free(freezeReply);
    CpvAccess(continueFlag) = 0;
  } else if(CpvAccess(stepFlag) && (isEntryPoint((char *)msg))){
    if(breakPointHeader != 0){
      free(breakPointHeader);
      breakPointHeader = 0;
    }
    if(breakPointContents != 0){
      free(breakPointContents);
      breakPointContents = 0;
    }

    breakPointHeader = genericViewMsgFunction((char *)msg, 0);
    breakPointContents = genericViewMsgFunction((char *)msg, 1);

    CmiPrintf("STEP POINT REACHED :\n");
    CmiPrintf("Header:%s\nContents:%s\n",breakPointHeader,breakPointContents);

    /* Freeze and send a message back */
    CpdFreeze();
    freezeReply = (char *)malloc(strlen("freezing@")+strlen(breakPointHeader)+1);
    sprintf(freezeReply, "freezing@%s", breakPointHeader);
    fd = skt_connect(freezeIP, freezePort, 120);
    if(fd > 0){
      writeall(fd, freezeReply, strlen(freezeReply) + 1);
      close(fd);
    } else {
      CmiPrintf("unable to connect");
    }
    free(freezeReply);
    CpvAccess(stepFlag) = 0;
  }
#endif  
  (CmiGetHandlerFunction(msg))(msg);
  if (!CtvAccess(CmiBufferGrabbed)) CmiFree(msg);
}

void CmiDeliversInit()
{
  CtvInitialize(int, CmiBufferGrabbed);
  CtvAccess(CmiBufferGrabbed) = 0;
}

int CmiDeliverMsgs(int maxmsgs)
{
  return CsdScheduler(maxmsgs);
}

int CsdScheduler(int maxmsgs)
{
  int *msg;
  void *localqueue = CpvAccess(CmiLocalQueue);
  int cycle = CpvAccess(CsdStopFlag);
  
  if(maxmsgs == 0) {
    while(1) {
      msg = CmiGetNonLocal();
#if CMK_DEBUG_MODE
      if(CpvAccess(freezeModeFlag)==1){

        /* Check if the msg is an debug message to let it go
           else, enqueue in the FIFO
        */

        if(msg != 0){
          if(strncmp((char *)((char *)msg+CmiMsgHeaderSizeBytes),"req",3)!=0) {
            CsdEndIdle();
            FIFO_EnQueue(CpvAccess(debugQueue), msg);
            continue;
          }
        } else {
          continue;
        }
      } else {
        /* If the debugQueue contains any messages, process them */
        while(!FIFO_Empty(CpvAccess(debugQueue))){
          char *queuedMsg;
          FIFO_DeQueue(CpvAccess(debugQueue), &queuedMsg);
          CmiHandleMessage(queuedMsg);
          maxmsgs--; if (maxmsgs==0) return maxmsgs;
        }
      }
#endif
      if (msg==0) FIFO_DeQueue(localqueue, &msg);
#if CMK_NODE_QUEUE_AVAILABLE
      if (msg==0) {
	CmiLock(CsvAccess(NodeQueueLock));
      	msg = CmiGetNonLocalNodeQ();
	if (msg==0 && 
            !CqsPrioGT(CqsGetPriority(CsvAccess(CsdNodeQueue)), 
                       CqsGetPriority(CpvAccess(CsdSchedQueue)))) {
	  CqsDequeue(CsvAccess(CsdNodeQueue),&msg);
	}
	CmiUnlock(CsvAccess(NodeQueueLock));
      }
#endif
      if (msg==0) CqsDequeue(CpvAccess(CsdSchedQueue),&msg);
      if (msg) {
        CmiHandleMessage(msg);
        maxmsgs--;
        if (CpvAccess(CsdStopFlag) != cycle) return maxmsgs;
      } else {
        return maxmsgs;
      }
    }
  }

  while (1) {
    msg = CmiGetNonLocal();
#if CMK_DEBUG_MODE
    if(CpvAccess(freezeModeFlag) == 1){
      
      /* Check if the msg is an debug message to let it go
	 else, enqueue in the FIFO 
      */

      if(msg != 0){
	if(strncmp((char *)((char *)msg+CmiMsgHeaderSizeBytes),"req",3)!=0){
	  CsdEndIdle();
	  FIFO_EnQueue(CpvAccess(debugQueue), msg);
	  continue;
        }
      } else {
	continue;
      }
    } else {
      /* If the debugQueue contains any messages, process them */
      while(!FIFO_Empty(CpvAccess(debugQueue))){
        char *queuedMsg;
	FIFO_DeQueue(CpvAccess(debugQueue), &queuedMsg);
	CmiHandleMessage(queuedMsg);
	maxmsgs--; if (maxmsgs==0) return maxmsgs;	
      }
    }
#endif
    if (msg==0) FIFO_DeQueue(localqueue, &msg);
#if CMK_NODE_QUEUE_AVAILABLE
    if (msg==0) {
      CmiLock(CsvAccess(NodeQueueLock));
      msg = CmiGetNonLocalNodeQ();
      if (msg==0 && 
            !CqsPrioGT(CqsGetPriority(CsvAccess(CsdNodeQueue)), 
                       CqsGetPriority(CpvAccess(CsdSchedQueue)))) {
	  CqsDequeue(CsvAccess(CsdNodeQueue),&msg);
      }
      CmiUnlock(CsvAccess(NodeQueueLock));
    }

#endif
    if (msg==0) CqsDequeue(CpvAccess(CsdSchedQueue),&msg);
    if (msg) {
      CsdEndIdle();
      CmiHandleMessage(msg);
      maxmsgs--; if (maxmsgs==0) return maxmsgs;
      if (CpvAccess(CsdStopFlag) != cycle) return maxmsgs;
    } else {
      CsdBeginIdle();
      if (CpvAccess(CsdStopFlag) != cycle) {
	CsdEndIdle();
	return maxmsgs;
      }
    }
    if (!CpvAccess(disable_sys_msgs))
      if (CpvAccess(CcdNumChecks) > 0)
        CcdCallBacks();
  }
}

void CmiDeliverSpecificMsg(handler)
int handler;
{
  int *msg, *t; int side;
  void *localqueue = CpvAccess(CmiLocalQueue);
 
  side = 0;
  while (1) {
    side ^= 1;
    if (side) msg = CmiGetNonLocal();
    else      FIFO_DeQueue(localqueue, &msg);
    if (msg) {
      if (CmiGetHandler(msg)==handler) {
	CsdEndIdle();
	CmiHandleMessage(msg);
	return;
      } else {
	FIFO_EnQueue(localqueue, msg);
      }
    }
  }
}
 
#endif /* CMK_CMIDELIVERS_USE_COMMON_CODE */

/***************************************************************************
 *
 * Standin Schedulers.
 *
 * We use the following strategy to make sure somebody's always running
 * the scheduler (CsdScheduler).  Initially, we assume the main thread
 * is responsible for this.  If the main thread blocks, we create a
 * "standin scheduler" thread to replace it.  If the standin scheduler
 * blocks, we create another standin scheduler to replace that one,
 * ad infinitum.  Collectively, the main thread and all the standin
 * schedulers are called "scheduling threads".
 *
 * Suppose the main thread is blocked waiting for data, and a standin
 * scheduler is running instead.  Suppose, then, that the data shows
 * up and the main thread is CthAwakened.  This causes a token to be
 * pushed into the queue.  When the standin pulls the token from the
 * queue and handles it, the standin goes to sleep, and control shifts
 * back to the main thread.  In this way, unnecessary standins are put
 * back to sleep.  These sleeping standins are stored on the
 * CthSleepingStandins list.
 *
 ***************************************************************************/

CpvStaticDeclare(CthThread, CthMainThread);
CpvStaticDeclare(CthThread, CthSchedulingThread);
CpvStaticDeclare(CthThread, CthSleepingStandins);
CpvStaticDeclare(int      , CthResumeNormalThreadIdx);
CpvStaticDeclare(int      , CthResumeSchedulingThreadIdx);

/** addition for tracing */
CpvExtern(CthThread, cThread);
/* end addition */

static void CthStandinCode()
{
  while (1) CsdScheduler(0);
}

static CthThread CthSuspendNormalThread()
{
  return CpvAccess(CthSchedulingThread);
}

static void CthEnqueueSchedulingThread(CthThread t);
static CthThread CthSuspendSchedulingThread();

static CthThread CthSuspendSchedulingThread()
{
  CthThread succ = CpvAccess(CthSleepingStandins);
  CthThread me = CthSelf();

  if (succ) {
    CpvAccess(CthSleepingStandins) = CthGetNext(succ);
  } else {
    succ = CthCreate(CthStandinCode, 0, 256000);
    CthSetStrategy(succ,
		   CthEnqueueSchedulingThread,
		   CthSuspendSchedulingThread);
  }
  
  CpvAccess(CthSchedulingThread) = succ;
  return succ;
}

static void CthResumeNormalThread(CthThread t)
{
  CmiGrabBuffer((void**)&t);
  /** addition for tracing */
  CpvAccess(cThread) = t;
  if(CpvAccess(traceOn))
    trace_begin_execute(0);
  /* end addition */
  CthResume(t);
}

static void CthResumeSchedulingThread(CthThread t)
{
  CthThread me = CthSelf();
  CmiGrabBuffer((void**)&t);
  if (me == CpvAccess(CthMainThread)) {
    CthEnqueueSchedulingThread(me);
  } else {
    CthSetNext(me, CpvAccess(CthSleepingStandins));
    CpvAccess(CthSleepingStandins) = me;
  }
  CpvAccess(CthSchedulingThread) = t;
  CthResume(t);
}

static void CthEnqueueNormalThread(CthThread t)
{
  CmiSetHandler(t, CpvAccess(CthResumeNormalThreadIdx));
  CsdEnqueueFifo(t);
}

static void CthEnqueueSchedulingThread(CthThread t)
{
  CmiSetHandler(t, CpvAccess(CthResumeSchedulingThreadIdx));
  CsdEnqueueFifo(t);
}

void CthSetStrategyDefault(CthThread t)
{
  CthSetStrategy(t,
		 CthEnqueueNormalThread,
		 CthSuspendNormalThread);
}

void CthSchedInit()
{
  CpvInitialize(CthThread, CthMainThread);
  CpvInitialize(CthThread, CthSchedulingThread);
  CpvInitialize(CthThread, CthSleepingStandins);
  CpvInitialize(int      , CthResumeNormalThreadIdx);
  CpvInitialize(int      , CthResumeSchedulingThreadIdx);

  CpvAccess(CthMainThread) = CthSelf();
  CpvAccess(CthSchedulingThread) = CthSelf();
  CpvAccess(CthSleepingStandins) = 0;
  CpvAccess(CthResumeNormalThreadIdx) =
    CmiRegisterHandler(CthResumeNormalThread);
  CpvAccess(CthResumeSchedulingThreadIdx) =
    CmiRegisterHandler(CthResumeSchedulingThread);
  CthSetStrategy(CthSelf(),
		 CthEnqueueSchedulingThread,
		 CthSuspendSchedulingThread);
}

void CsdInit(argv)
  char **argv;
{
  void *CqsCreate();

  CpvInitialize(int,   disable_sys_msgs);
  CpvInitialize(void*, CsdSchedQueue);
#if CMK_NODE_QUEUE_AVAILABLE
  CsvInitialize(void*, CsdNodeQueue);
  CsvInitialize(CmiNodeLock, NodeQueueLock);
#endif
  CpvInitialize(int,   CsdStopFlag);
  CpvInitialize(int,   CsdStopNotifyFlag);
  CpvInitialize(int,   CsdIdleDetectedFlag);
  CpvInitialize(CmiHandler,   CsdNotifyIdle);
  CpvInitialize(CmiHandler,   CsdNotifyBusy);
  
  CpvAccess(disable_sys_msgs) = 0;
  CpvAccess(CsdSchedQueue) = CqsCreate();

#if CMK_NODE_QUEUE_AVAILABLE
  if (CmiMyRank() ==0) {
	CsvAccess(NodeQueueLock) = CmiCreateLock();
	CsvAccess(CsdNodeQueue) = CqsCreate();
  }
  CmiNodeBarrier();
#endif

  CpvAccess(CsdStopFlag)  = 0;
  CpvAccess(CsdStopNotifyFlag) = 1;
  CpvAccess(CsdIdleDetectedFlag) = 0;
}


/*****************************************************************************
 *
 * Vector Send
 *
 ****************************************************************************/

#if CMK_VECTOR_SEND_USES_COMMON_CODE

void CmiSyncVectorSend(destPE, n, sizes, msgs)
int destPE, n;
int *sizes;
char **msgs;
{
  int i, total;
  char *mesg, *tmp;
  
  for(i=0,total=0;i<n;i++) total += sizes[i];
  mesg = (char *) CmiAlloc(total);
  for(i=0,tmp=mesg;i<n;i++) {
    memcpy(tmp, msgs[i],sizes[i]);
    tmp += sizes[i];
  }
  CmiSyncSendAndFree(destPE, total, mesg);
}

CmiCommHandle CmiAsyncVectorSend(destPE, n, sizes, msgs)
int destPE, n;
int *sizes;
char **msgs;
{
  CmiSyncVectorSend(destPE,n,sizes,msgs);
  return NULL;
}

void CmiSyncVectorSendAndFree(destPE, n, sizes, msgs)
int destPE, n;
int *sizes;
char **msgs;
{
  int i;

  CmiSyncVectorSend(destPE,n,sizes,msgs);
  for(i=0;i<n;i++) CmiFree(msgs[i]);
  CmiFree(sizes);
  CmiFree(msgs);
}

#endif

/*****************************************************************************
 *
 * Multicast groups
 *
 ****************************************************************************/

#if CMK_MULTICAST_DEF_USE_COMMON_CODE

typedef struct GroupDef
{
  union {
    char core[CmiMsgHeaderSizeBytes];
    struct GroupDef *next;
  } core;
  CmiGroup group;
  int npes;
  int pes[1];
}
*GroupDef;

#define GROUPTAB_SIZE 101

CpvStaticDeclare(int, CmiGroupHandlerIndex);
CpvStaticDeclare(int, CmiGroupCounter);
CpvStaticDeclare(GroupDef *, CmiGroupTable);

void CmiGroupHandler(GroupDef def)
{
  /* receive group definition, insert into group table */
  int i;
  GroupDef *table = CpvAccess(CmiGroupTable);
  unsigned int hashval, bucket;
  CmiGrabBuffer((void*)&def);
  hashval = (def->group.id ^ def->group.pe);
  bucket = hashval % GROUPTAB_SIZE;
  def->core.next = table[bucket];
  table[bucket] = def;
}

CmiGroup CmiEstablishGroup(int npes, int *pes)
{
  /* build new group definition, broadcast it */
  CmiGroup grp; GroupDef def; int len, i;
  grp.id = CpvAccess(CmiGroupCounter)++;
  grp.pe = CmiMyPe();
  len = sizeof(struct GroupDef)+(npes*sizeof(int));
  def = (GroupDef)CmiAlloc(len);
  def->group = grp;
  def->npes = npes;
  for (i=0; i<npes; i++)
    def->pes[i] = pes[i];
  CmiSetHandler(def, CpvAccess(CmiGroupHandlerIndex));
  CmiSyncBroadcastAllAndFree(len, def);
  return grp;
}

void CmiLookupGroup(CmiGroup grp, int *npes, int **pes)
{
  unsigned int hashval, bucket;  GroupDef def;
  GroupDef *table = CpvAccess(CmiGroupTable);
  hashval = (grp.id ^ grp.pe);
  bucket = hashval % GROUPTAB_SIZE;
  for (def=table[bucket]; def; def=def->core.next) {
    if ((def->group.id == grp.id)&&(def->group.pe == grp.pe)) {
      *npes = def->npes;
      *pes = def->pes;
      return;
    }
  }
  *npes = 0; *pes = 0;
}

void CmiGroupInit()
{
  CpvInitialize(int, CmiGroupHandlerIndex);
  CpvInitialize(int, CmiGroupCounter);
  CpvInitialize(GroupDef *, CmiGroupTable);
  CpvAccess(CmiGroupHandlerIndex) = CmiRegisterHandler(CmiGroupHandler);
  CpvAccess(CmiGroupCounter) = 0;
  CpvAccess(CmiGroupTable) =
    (GroupDef*)calloc(GROUPTAB_SIZE, sizeof(GroupDef));
  if (CpvAccess(CmiGroupTable) == 0)
    CmiAbort("Memory Allocation Error");
}

#endif

/*****************************************************************************
 *
 * Common List-Cast and Multicast Code
 *
 ****************************************************************************/

#if CMK_MULTICAST_LIST_USE_COMMON_CODE

void CmiSyncListSendFn(int npes, int *pes, int len, char *msg)
{
  CmiError("ListSend not implemented.");
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int len, char *msg)
{
  CmiError("ListSend not implemented.");
}

void CmiFreeListSendFn(int npes, int *pes, int len, char *msg)
{
  CmiError("ListSend not implemented.");
}

#endif

#if CMK_MULTICAST_GROUP_USE_COMMON_CODE

typedef struct MultiMsg
{
  char core[CmiMsgHeaderSizeBytes];
  CmiGroup group;
  int pos;
  int origlen;
}
*MultiMsg;

CpvDeclare(int, CmiMulticastHandlerIndex);

void CmiMulticastDeliver(MultiMsg msg)
{
  int npes, *pes; int olen, nlen, pos, child1, child2;
  olen = msg->origlen;
  nlen = olen + sizeof(struct MultiMsg);
  CmiLookupGroup(msg->group, &npes, &pes);
  if (pes==0) {
    CmiSyncSendAndFree(CmiMyPe(), nlen, msg);
    return;
  }
  if (npes==0) {
    CmiFree(msg);
    return;
  }
  if (msg->pos == -1) {
    msg->pos=0;
    CmiSyncSendAndFree(pes[0], nlen, msg);
    return;
  }
  pos = msg->pos;
  child1 = ((pos+1)<<1);
  child2 = child1-1;
  if (child1 < npes) {
    msg->pos = child1;
    CmiSyncSend(pes[child1], nlen, msg);
  }
  if (child2 < npes) {
    msg->pos = child2;
    CmiSyncSend(pes[child2], nlen, msg);
  }
  if(olen < sizeof(struct MultiMsg)) {
    memcpy(msg, msg+1, olen);
  } else {
    memcpy(msg, (((char*)msg)+olen), sizeof(struct MultiMsg));
  }
  CmiSyncSendAndFree(CmiMyPe(), olen, msg);
}

void CmiMulticastHandler(MultiMsg msg)
{
  CmiGrabBuffer((void*)&msg);
  CmiMulticastDeliver(msg);
}

void CmiSyncMulticastFn(CmiGroup grp, int len, char *msg)
{
  int newlen; MultiMsg newmsg;
  newlen = len + sizeof(struct MultiMsg);
  newmsg = (MultiMsg)CmiAlloc(newlen);
  if(len < sizeof(struct MultiMsg)) {
    memcpy(newmsg+1, msg, len);
  } else {
    memcpy(newmsg+1, msg+sizeof(struct MultiMsg), len-sizeof(struct MultiMsg));
    memcpy(((char *)newmsg+len), msg, sizeof(struct MultiMsg));
  }
  newmsg->group = grp;
  newmsg->origlen = len;
  newmsg->pos = -1;
  CmiSetHandler(newmsg, CpvAccess(CmiMulticastHandlerIndex));
  CmiMulticastDeliver(newmsg);
}

void CmiFreeMulticastFn(CmiGroup grp, int len, char *msg)
{
  CmiSyncMulticastFn(grp, len, msg);
  CmiFree(msg);
}

CmiCommHandle CmiAsyncMulticastFn(CmiGroup grp, int len, char *msg)
{
  CmiError("Async Multicast not implemented.");
}

void CmiMulticastInit()
{
  CpvInitialize(int, CmiMulticastHandlerIndex);
  CpvAccess(CmiMulticastHandlerIndex) =
    CmiRegisterHandler(CmiMulticastHandler);
}

#endif

/***************************************************************************
 *
 * Memory Allocation routines 
 *
 * A block of memory can consist of multiple chunks.  Each chunk has
 * a sizefield and a refcount.  The first chunk's refcount is a reference
 * count.  That's how many CmiFrees it takes to free the message.
 * Subsequent chunks have a refcount which is less than zero.  This is
 * the offset back to the start of the first chunk.
 *
 ***************************************************************************/

#define SIZEFIELD(m) ((int *)((char *)(m)-2*sizeof(int)))[0]
#define REFFIELD(m) ((int *)((char *)(m)-sizeof(int)))[0]
#define BLKSTART(m) ((char *)m-2*sizeof(int))

void *CmiAlloc(size)
int size;
{
  char *res;
  res =(char *)malloc(size+2*sizeof(int));
  if (res==0) CmiAbort("Memory allocation failed.");

#ifdef MEMMONITOR
  CpvAccess(MemoryUsage) += size+2*sizeof(int);
  CpvAccess(AllocCount)++;
  CpvAccess(BlocksAllocated)++;
  if (CpvAccess(MemoryUsage) > CpvAccess(HiWaterMark)) {
    CpvAccess(HiWaterMark) = CpvAccess(MemoryUsage);
  }
  if (CpvAccess(MemoryUsage) > 1.1 * CpvAccess(ReportedHiWaterMark)) {
    CmiPrintf("HIMEM STAT PE%d: %d Allocs, %d blocks, %lu K, Max %lu K\n",
	    CmiMyPe(), CpvAccess(AllocCount), CpvAccess(BlocksAllocated),
            CpvAccess(MemoryUsage)/1024, CpvAccess(HiWaterMark)/1024);
    CpvAccess(ReportedHiWaterMark) = CpvAccess(MemoryUsage);
  }
  if ((CpvAccess(AllocCount) % 1000) == 0) {
    CmiPrintf("MEM STAT PE%d: %d Allocs, %d blocks, %lu K, Max %lu K\n",
	    CmiMyPe(), CpvAccess(AllocCount), CpvAccess(BlocksAllocated),
            CpvAccess(MemoryUsage)/1024, CpvAccess(HiWaterMark)/1024);
  }
#endif

  ((int *)res)[0]=size;
  ((int *)res)[1]=1;
  return (void *)(res+2*sizeof(int));
}

void CmiReference(blk)
void *blk;
{
  int refCount = REFFIELD(blk);
  if (refCount < 0) {
    blk = (void *)((char*)blk+refCount);
    refCount = REFFIELD(blk);
  }
  REFFIELD(blk) = refCount+1;
}

int CmiSize(blk)
void *blk;
{
  return SIZEFIELD(blk);
}

void CmiFree(blk)
void *blk;
{
  int offset;
  int refCount;

  refCount = REFFIELD(blk);
  if (refCount < 0) {
    blk = (void *)((char*)blk+refCount);
    refCount = REFFIELD(blk);
  }
  if(refCount==0) {
#ifdef MEMMONITOR
    if (SIZEFIELD(blk) > 100000)
      CmiPrintf("MEMSTAT Uh-oh -- SIZEFIELD=%d\n",SIZEFIELD(blk));
    CpvAccess(MemoryUsage) -= (SIZEFIELD(blk) + 2*sizeof(int));
    CpvAccess(BlocksAllocated)--;
    CmiPrintf("Refcount 0 case called\n");
#endif
    free(BLKSTART(blk));
    return;
  }
  refCount--;
  if(refCount==0) {
#ifdef MEMMONITOR
    if (SIZEFIELD(blk) > 100000)
      CmiPrintf("MEMSTAT Uh-oh -- SIZEFIELD=%d\n",SIZEFIELD(blk));
    CpvAccess(MemoryUsage) -= (SIZEFIELD(blk) + 2*sizeof(int));
    CpvAccess(BlocksAllocated)--;
#endif
    free(BLKSTART(blk));
    return;
  }
  REFFIELD(blk) = refCount;
}

/******************************************************************************

  Multiple Send function                               

  ****************************************************************************/

CpvDeclare(int, CmiMainHandlerIDP); /* Main handler that is run on every node */

/****************************************************************************
* DESCRIPTION : This function call allows the user to send multiple messages
*               from one processor to another, all intended for differnet 
*	        handlers.
*
*	        Parameters :
*
*	        destPE, len, int sizes[], char *messages[]
*
* ASSUMPTION  : The sizes[] and the messages[] array begin their indexing FROM 1.
*               (i.e They should have memory allocated for n + 1)
*               This is important to ensure that the call works correctly
*
****************************************************************************/

void CmiMultipleSend(unsigned int destPE, int len, int sizes[], char *msgComps[])
{
  char *header;
  int i;
  int *newSizes;
  char **newMsgComps;
  int mask = ~7; /* to mask off the last 3 bits */
  char *pad = "                 "; /* padding required - 16 bytes long w.case */

  /* Allocate memory for the newSizes array and the newMsgComps array*/
  newSizes = (int *)CmiAlloc(2 * (len + 1) * sizeof(int));
  newMsgComps = (char **)CmiAlloc(2 * (len + 1) * sizeof(char *));

  /* Construct the newSizes array from the old sizes array */
  newSizes[0] = (CmiMsgHeaderSizeBytes + (len + 1)*sizeof(int));
  newSizes[1] = ((CmiMsgHeaderSizeBytes + (len + 1)*sizeof(int) + 7)&mask) - newSizes[0] + 2*sizeof(int);
                     /* To allow the extra 8 bytes for the CmiSize & the Ref Count */

  for(i = 1; i < len + 1; i++){
    newSizes[2*i] = (sizes[i - 1]);
    newSizes[2*i + 1] = ((sizes[i -1] + 7)&mask) - newSizes[2*i] + 2*sizeof(int); 
             /* To allow the extra 8 bytes for the CmiSize & the Ref Count */
  }
    
  header = (char *)CmiAlloc(newSizes[0]*sizeof(char));

  /* Set the len field in the buffer */
  *(int *)(header + CmiMsgHeaderSizeBytes) = len;

  /* and the induvidual lengths */
  for(i = 1; i < len + 1; i++){
    *((int *)(header + CmiMsgHeaderSizeBytes) + i) = newSizes[2*i] + newSizes[2*i + 1];
  }

  /* This message shd be recd by the main handler */
  CmiSetHandler(header, CpvAccess(CmiMainHandlerIDP));
  newMsgComps[0] = header;
  newMsgComps[1] = pad;

  for(i = 1; i < (len + 1); i++){
    newMsgComps[2*i] =  msgComps[i - 1];
    newMsgComps[2*i + 1] = pad;
  }

  CmiSyncVectorSend(destPE, 2*(len + 1), newSizes, newMsgComps);
  CmiFree(newSizes);
  CmiFree(newMsgComps);
  CmiFree(header);
}

/****************************************************************************
* DESCRIPTION : This function initializes the main handler required for the
*               CmiMultipleSendP() function to work. 
*	        
*               This function should be called once in any Converse program
*	        that uses CmiMultipleSendP()
*
****************************************************************************/

static CmiHandler CmiMultiMsgHandler(char *msgWhole);

void CmiInitMultipleSend(void)
{
  CpvInitialize(int,CmiMainHandlerIDP); 
  CpvAccess(CmiMainHandlerIDP) =
    CmiRegisterHandler((CmiHandler)CmiMultiMsgHandler);
}

/****************************************************************************
* DESCRIPTION : This function is the main handler required for the
*               CmiMultipleSendP() function to work. 
*
****************************************************************************/

static void memChop(char *msgWhole);

static CmiHandler CmiMultiMsgHandler(char *msgWhole)
{
  int len;
  int *sizes;
  int i;
  int offset;
  int mask = ~7; /* to mask off the last 3 bits */
  
  /* Number of messages */
  offset = CmiMsgHeaderSizeBytes;
  len = *(int *)(msgWhole + offset);
  offset += sizeof(int);

  /* Allocate array to store sizes */
  sizes = (int *)(msgWhole + offset);
  offset += sizeof(int)*len;

  /* This is needed since the header may or may not be aligned on an 8 bit boundary */
  offset = (offset + 7)&mask;

  /* To cross the 8 bytes inserted in between */
  offset += 2*sizeof(int);

  /* Call memChop() */
  memChop(msgWhole);

  /* Send the messages to their respective handlers (on the same machine) */
  /* Currently uses CmiSyncSend(), later modify to use Scheduler enqueuing */
  for(i = 0; i < len; i++){
    CmiSyncSendAndFree(CmiMyPe(), sizes[i], ((char *)(msgWhole + offset))); 
    offset += sizes[i];
  }
}

static void memChop(char *msgWhole)
{
  int len;
  int *sizes;
  int i;
  int offset;
  int mask = ~7; /* to mask off the last 3 bits */
  
  /* Number of messages */
  offset = CmiMsgHeaderSizeBytes;
  len = *(int *)(msgWhole + offset);
  offset += sizeof(int);

  /* Set Reference count in the CmiAlloc header*/
  /* Reference Count includes the header also, hence (len + 1) */
  ((int *)(msgWhole - sizeof(int)))[0] = len + 1;

  /* Allocate array to store sizes */
  sizes = (int *)(msgWhole + offset);
  offset += sizeof(int)*len;

  /* This is needed since the header may or may not be aligned on an 8 bit boundary */
  offset = (offset + 7)&mask;

  /* To cross the 8 bytes inserted in between */
  offset += 2*sizeof(int);

  /* update the sizes and offsets for all the chunks */
  for(i = 0; i < len; i++){
    /* put in the size value for that part */
    ((int *)(msgWhole + offset - 2*sizeof(int)))[0] = sizes[i] - 2*sizeof(int);
    
    /* now put in the offset (a negative value) to get right back to the begining */
    ((int *)(msgWhole + offset - sizeof(int)))[0] = (-1)*offset;
    
    offset += sizes[i];
  }
}

/*****************************************************************************
 *
 * Converse Client-Server Functions
 *
 *****************************************************************************/

#if CMK_CCS_AVAILABLE

typedef struct CcsListNode {
  char name[32];
  int hdlr;
  struct CcsListNode *next;
}CcsListNode;

CpvStaticDeclare(CcsListNode*, ccsList);
CpvStaticDeclare(int, callerIP);
CpvStaticDeclare(int, callerPort);
CpvDeclare(int, strHandlerID);

static void CcsStringHandlerFn(char *msg)
{
  char cmd[10], hdlrName[32], *cmsg, *omsg=msg;
  int ip, port, pe, size, nread, hdlrID;
  CcsListNode *list = CpvAccess(ccsList);

  msg += CmiMsgHeaderSizeBytes;
  nread = sscanf(msg, "%s%d%d%d%d%s", 
                 cmd, &pe, &size, &ip, &port, hdlrName);
  if(nread!=6) CmiAbort("Garbled message from client");
  while(list!=0) {
    if(strcmp(hdlrName, list->name)==0) {
      hdlrID = list->hdlr;
      break;
    }
    list = list->next;
  }
  if(list==0) CmiAbort("Invalid Service Request\n");
  while(*msg != '\n') msg++;
  msg++;
  cmsg = (char *) CmiAlloc(size+CmiMsgHeaderSizeBytes+1);
  memcpy(cmsg+CmiMsgHeaderSizeBytes, msg, size);
  cmsg[CmiMsgHeaderSizeBytes+size] = '\0';

  CmiSetHandler(cmsg, hdlrID);
  CpvAccess(callerIP) = ip;
  CpvAccess(callerPort) = port;
  CmiHandleMessage(cmsg);
  CmiGrabBuffer((void **)&omsg);
  CmiFree(omsg);
  CpvAccess(callerIP) = 0;
}

static void CcsInit(void)
{
  CpvInitialize(CcsListNode*, ccsList);
  CpvAccess(ccsList) = 0;
  CpvInitialize(int, callerIP);
  CpvAccess(callerIP) = 0;
  CpvInitialize(int, callerPort);
  CpvAccess(callerPort) = 0;
  CpvInitialize(int, strHandlerID);
  CpvAccess(strHandlerID) = CmiRegisterHandler(CcsStringHandlerFn);
}

void CcsUseHandler(char *name, int hdlr)
{
  CcsListNode *list=CpvAccess(ccsList);
  if(list==0) {
    list = (CcsListNode *)malloc(sizeof(CcsListNode));
    CpvAccess(ccsList) = list;
  } else {
    while(list->next != 0) 
      list = list->next;
    list->next = (CcsListNode *)malloc(sizeof(CcsListNode));
    list = list->next;
  }
  strcpy(list->name, name);
  list->hdlr = hdlr;
  list->next = 0;
}

int CcsRegisterHandler(char *name, CmiHandler fn)
{
  int hdlr = CmiRegisterHandlerLocal(fn);
  CcsUseHandler(name, hdlr);
  return hdlr;
}

int CcsEnabled(void)
{
  return 1;
}

int CcsIsRemoteRequest(void)
{
  return (CpvAccess(callerIP) != 0);
}

void CcsCallerId(unsigned int *pip, unsigned int *pport)
{
  *pip = CpvAccess(callerIP);
  *pport = CpvAccess(callerPort);
}

extern int skt_connect(int, int, int);
extern void writeall(int, char *, int);

void CcsSendReply(unsigned int ip, unsigned int port, int size, void *msg)
{
  char cmd[100], c;
  int fd;

  fd = skt_connect(ip, port, 120);
  
  if (fd<0) {
      CmiPrintf("client Exited\n");
      return; /* maybe the requester exited */
  }
  sprintf(cmd, "reply %d\n", size);
  writeall(fd, cmd, strlen(cmd));
  writeall(fd, msg, size);

#if CMK_SYNCHRONIZE_ON_TCP_CLOSE
  shutdown(fd, 1);
  while (read(fd, &c, 1)==EINTR);
  close(fd);
#else
  close(fd);
#endif
}

#endif
/*****************************************************************************
 *
 * Converse Initialization
 *
 *****************************************************************************/

extern void CrnInit(void);

void ConverseCommonInit(char **argv)
{
  CrnInit();
  CmiTimerInit();
  CstatsInit(argv);
  CcdModuleInit(argv);
  CmiHandlerInit();
  CmiMemoryInit(argv);
  CmiDeliversInit();
  CsdInit(argv);
  CthSchedInit();
  CmiGroupInit();
  CmiMulticastInit();
  CmiInitMultipleSend();
  CcsInit();
#if CMK_DEBUG_MODE
  CpdInit();
#endif
#if CMK_WEB_MODE
  CWebInit();
#endif
  CldModuleInit();
}

void ConverseCommonExit(void)
{
  close_log();
}


