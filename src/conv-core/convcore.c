/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <string.h>
#include "converse.h"
#include "conv-trace.h"
#include "conv-ccs.h"
#include <errno.h>

#ifndef WIN32
#include <sys/file.h>
#endif

#ifdef WIN32
#include "queueing.h"

extern void CqsDequeue(Queue, void **);
extern void CqsEnqueueFifo(Queue, void *);
extern void CcdModuleInit(char **);
extern void CmiMemoryInit(char **);
extern void CldModuleInit(void);
extern int  CqsPrioGT(prio, prio);
extern prio CqsGetPriority(Queue);
#define DEBUGF(x)  printf x
#endif

/*
#if NODE_0_IS_CONVHOST
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/time.h>
#endif
*/

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

#ifdef CMK_TIMER_USE_WIN32API
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <sys/types.h>
#include <sys/timeb.h>
#endif

#include "fifo.h"

#if NODE_0_IS_CONVHOST
extern int serverFlag;
extern int hostport, hostskt;
extern int hostskt_ready_read;
extern unsigned int *nodeIPs;
extern unsigned int *nodePorts;
extern void skt_server(int *, int *);
extern void CommunicationServer();
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
CsvDeclare(CmiNodeLock, CsdNodeQueueLock);
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

#if CMK_WEB_MODE
  void initUsage();
#endif

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

  CpvInitialize(int, CstatsMaxChareQueueLength);
  CpvInitialize(int, CstatsMaxForChareQueueLength);
  CpvInitialize(int, CstatsMaxFixedChareQueueLength);
  CpvInitialize(int, CstatPrintQueueStatsFlag);
  CpvInitialize(int, CstatPrintMemStatsFlag);

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
    } else
    argv++;
  }

  argc = 0; argv=origArgv;
  for(argc=0;argv[argc];argc++);
  /*CmiPrintf("argc = %d, argv[0] = %s\n", argc, argv[0]);*/
#ifndef CMK_OPTIMIZE
  traceInit(&argc, argv);
#endif

#if CMK_WEB_MODE
  initUsage();
#endif
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
  _MEMCHECK(CpvAccess(CmiHandlerTable));
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

#if CMK_TIMER_USE_WIN32API

CpvStaticDeclare(double, inittime_wallclock);
CpvStaticDeclare(double, inittime_virtual);

void CmiTimerInit()
{
	struct _timeb tv;
	clock_t       ru;

	CpvInitialize(double, inittime_wallclock);
	CpvInitialize(double, inittime_virtual);
	_ftime(&tv);
	CpvAccess(inittime_wallclock) = tv.time*1.0 + tv.millitm*0.001;
	ru = clock();
	CpvAccess(inittime_virtual) = ((double) ru)/CLOCKS_PER_SEC;
}

double CmiCpuTimer()
{
	clock_t ru;
	double currenttime;

	ru = clock();
	currenttime = (double) ru/CLOCKS_PER_SEC;

	return currenttime - CpvAccess(inittime_virtual);
}

double CmiWallTimer()
{
	struct _timeb tv;
	double currenttime;

	_ftime(&tv);
	currenttime = tv.time*1.0 + tv.millitm*0.001;

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
#ifndef CMK_OPTIMIZE
      if(CpvAccess(traceOn))
        traceEndIdle();
#endif
    }
  }
#if CMK_WEB_MODE
  usageStart();  
#endif
}

void CsdBeginIdle()
{
  if (!CpvAccess(CsdIdleDetectedFlag)) {
    CpvAccess(CsdIdleDetectedFlag) = 1;
    if(!CpvAccess(CsdStopNotifyFlag)) {
      (CpvAccess(CsdNotifyIdle))();
#ifndef CMK_OPTIMIZE
      if(CpvAccess(traceOn))
        traceBeginIdle();
#endif
    }
  }
#if CMK_WEB_MODE
  usageStop();  
#endif
  CmiNotifyIdle();
  CcdRaiseCondition(CcdPROCESSORIDLE) ;
}
  
#if CMK_CMIDELIVERS_USE_COMMON_CODE

CtvStaticDeclare(int, CmiBufferGrabbed);

void CmiGrabBuffer(void **bufptrptr)
{
  CtvAccess(CmiBufferGrabbed) = 1;
}

void CmiReleaseBuffer(void *buffer)
{
  CmiGrabBuffer(&buffer);
  CmiFree(buffer);
}

void CmiHandleMessage(void *msg)
{
#if CMK_DEBUG_MODE
  CpvExtern(int, freezeModeFlag);
  CpvExtern(int, continueFlag);
  CpvExtern(int, stepFlag);
  CpvExtern(void *, debugQueue);
  extern unsigned int freezeIP;
  extern int freezePort;
  extern char* breakPointHeader;
  extern char* breakPointContents;

  char *freezeReply;
  int fd;

  extern int skt_connect(unsigned int, int, int);
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
    _MEMCHECK(freezeReply);
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
    _MEMCHECK(freezeReply);
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
#if CMK_DEBUG_MODE
  CpvExtern(int, freezeModeFlag);
#endif

  int *msg, csdMsgFlag = 0; /* To signal a message coming from the CsdNodeQueue */
  void *localqueue = CpvAccess(CmiLocalQueue);
  int cycle = CpvAccess(CsdStopFlag);
  int pollmode = (maxmsgs==0);
  
#if CMK_DEBUG_MODE
  /* To allow start in freeze state */
  msgListCleanup();
  msgListCache();
#endif

  while (1) {
#if NODE_0_IS_CONVHOST
    if(hostskt_ready_read) CHostGetOne();
#endif
    msg = CmiGetNonLocal();
#if CMK_DEBUG_MODE
    if(CpvAccess(freezeModeFlag) == 1){
      
      /* Check if the msg is an debug message to let it go
	 else, enqueue in the FIFO 
      */

      if(msg != 0){
	if(strncmp((char *)((char *)msg+CmiMsgHeaderSizeBytes),"req",3)!=0){
          /*CQdCreate(CpvAccess(cQdState), 1);*/
	  CsdEndIdle();
	  FIFO_EnQueue(CpvAccess(debugQueue), msg);
	  continue;
        }
      } 
    } else {
      /* If the debugQueue contains any messages, process them */
      while(((!FIFO_Empty(CpvAccess(debugQueue))) && (CpvAccess(freezeModeFlag)==0))){
        char *queuedMsg;
	FIFO_DeQueue(CpvAccess(debugQueue), (void**)&queuedMsg);
	CmiHandleMessage(queuedMsg);
	maxmsgs--; if (maxmsgs==0) return maxmsgs;	
      }
    }
#endif
    if (msg==0) FIFO_DeQueue(localqueue, (void**)&msg);
#if CMK_NODE_QUEUE_AVAILABLE
	csdMsgFlag = 0;
    if (msg==0) msg = CmiGetNonLocalNodeQ();
    if (msg==0 && !CqsEmpty(CsvAccess(CsdNodeQueue))
               && !CqsPrioGT(CqsGetPriority(CsvAccess(CsdNodeQueue)), 
                             CqsGetPriority(CpvAccess(CsdSchedQueue)))) {
      CmiLock(CsvAccess(CsdNodeQueueLock));
      CqsDequeue(CsvAccess(CsdNodeQueue),&msg);
      CmiUnlock(CsvAccess(CsdNodeQueueLock));
	  csdMsgFlag = 1;
    }
#endif
    if (msg && (!csdMsgFlag)) CQdProcess(CpvAccess(cQdState), 1);
	if (msg==0) CqsDequeue(CpvAccess(CsdSchedQueue),&msg);
    if (msg) {
      CsdEndIdle();
      CmiHandleMessage(msg);
      maxmsgs--; if (maxmsgs==0) return maxmsgs;
      if (CpvAccess(CsdStopFlag) != cycle) return maxmsgs;
    } else {
      if(pollmode) return maxmsgs;
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
  int *msg; int side;
  void *localqueue = CpvAccess(CmiLocalQueue);
 
  side = 0;
  while (1) {
    side ^= 1;
    if (side) msg = CmiGetNonLocal();
    else      FIFO_DeQueue(localqueue, (void**)&msg);
    if (msg) {
      if (CmiGetHandler(msg)==handler) {
	CQdProcess(CpvAccess(cQdState), 1);
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
CpvDeclare(CthThread, curThread);
/* end addition */

static void CthStandinCode()
{
  while (1) CsdScheduler(0);
}

static CthThread CthSuspendNormalThread()
{
  return CpvAccess(CthSchedulingThread);
}

static void CthEnqueueSchedulingThread(CthThread t, int, int, int*);
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
  CpvAccess(curThread) = t;
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceResume();
#endif
  /* end addition */
#if CMK_WEB_MODE
  usageStart();  
#endif
  CthResume(t);
}

static void CthResumeSchedulingThread(CthThread t)
{
  CthThread me = CthSelf();
  CmiGrabBuffer((void**)&t);
  if (me == CpvAccess(CthMainThread)) {
    CthEnqueueSchedulingThread(me,CQS_QUEUEING_FIFO, 0, 0);
  } else {
    CthSetNext(me, CpvAccess(CthSleepingStandins));
    CpvAccess(CthSleepingStandins) = me;
  }
  CpvAccess(CthSchedulingThread) = t;
  CthResume(t);
}

static void CthEnqueueNormalThread(CthThread t, int s, int pb, int *prio)
{
  CmiSetHandler(t, CpvAccess(CthResumeNormalThreadIdx));
  CsdEnqueueGeneral(t, s, pb, prio);
}

static void CthEnqueueSchedulingThread(CthThread t, int s, int pb, int *prio)
{
  CmiSetHandler(t, CpvAccess(CthResumeSchedulingThreadIdx));
  CsdEnqueueGeneral(t, s, pb, prio);
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

  CpvInitialize(CthThread, curThread);

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
  CpvInitialize(int,   CsdStopFlag);
  CpvInitialize(int,   CsdStopNotifyFlag);
  CpvInitialize(int,   CsdIdleDetectedFlag);
  CpvInitialize(CmiHandler,   CsdNotifyIdle);
  CpvInitialize(CmiHandler,   CsdNotifyBusy);
  
  CpvAccess(disable_sys_msgs) = 0;
  CpvAccess(CsdSchedQueue) = CqsCreate();

#if CMK_NODE_QUEUE_AVAILABLE
  CsvInitialize(CmiLock, CsdNodeQueueLock);
  CsvInitialize(void*, CsdNodeQueue);
  if (CmiMyRank() ==0) {
	CsvAccess(CsdNodeQueueLock) = CmiCreateLock();
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
  return (CmiCommHandle) 0;
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
  return (CmiCommHandle) 0;
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
  _MEMCHECK(res);

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

static void CmiMultiMsgHandler(char *msgWhole);

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

static void CmiMultiMsgHandler(char *msgWhole)
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
 * Converse Initialization
 *
 *****************************************************************************/

extern void CrnInit(void);

#if CMK_CCS_AVAILABLE
extern void CcsInit(void);
#endif

void ConverseCommonInit(char **argv)
{
#if NODE_0_IS_CONVHOST
  int i,j;
  char *ptr;
#endif
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
  CQdInit();
#if CMK_CCS_AVAILABLE
  CcsInit();
#endif
#if CMK_DEBUG_MODE
  CpdInit();
#endif
#if CMK_WEB_MODE
  CWebInit();
#endif
#if NODE_0_IS_CONVHOST
  CHostInit();
  skt_server(&hostport, &hostskt);
  CmiSignal(SIGALRM, SIGIO, 0, CommunicationServer);
  CmiEnableAsyncIO(hostskt);
  CHostRegister();
 
  /*  if(CmiMyPe() == 0){ */
  i = 0;
  for(ptr = argv[i]; ptr != 0; i++, ptr = argv[i])
    if(strcmp(ptr, "++server") == 0) {
      if (CmiMyPe() == 0)
	serverFlag = 1;
      for(j = i; argv[j] != 0; j++)
	argv[j] = argv[j+1];
      break;
    }
  /*   } */
#endif
  CldModuleInit();
  CrnInit();
}

void ConverseCommonExit(void)
{
#if NODE_0_IS_CONVHOST
  if((CmiMyPe() == 0) && (clientIP != 0)){
    int fd;
    fd = skt_connect(clientIP, clientKillPort, 120);
    if (fd>0){ 
      write(fd, "die\n", strlen("die\n"));
    }
  }
#endif
#ifndef CMK_OPTIMIZE
  traceClose();
#endif
}


#if CMK_CMIPRINTF_IS_JUST_PRINTF

void CmiPrintf(const char *format, ...)
{
  va_list args;
  va_start(args,format);
  vprintf(format, args);
  fflush(stdout);
  va_end(args);
}

void CmiError(const char *format, ...)
{
  va_list args;
  va_start(args,format);
  vfprintf(stderr,format, args);
  fflush(stderr);
  va_end(args);
}

#endif
