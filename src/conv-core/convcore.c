/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile$
 *      $Author$        $Locker$                $State$
 *      $Revision$      $Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
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

void  *CmiGetNonLocal();
void   CmiNotifyIdle();

CpvDeclare(int, disable_sys_msgs);
CpvExtern(int,    CcdNumChecks) ;
CpvDeclare(void*, CsdSchedQueue);
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

  CpvInitialize(int, CtrRecdTraceMsg);
  CpvInitialize(int, CtrLogBufSize);
  CpvInitialize(int, CstatsMaxChareQueueLength);
  CpvInitialize(int, CstatsMaxForChareQueueLength);
  CpvInitialize(int, CstatsMaxFixedChareQueueLength);
  CpvInitialize(int, CstatPrintQueueStatsFlag);
  CpvInitialize(int, CstatPrintMemStatsFlag);

  CpvAccess(CtrLogBufSize) = 10000;
  CpvAccess(CstatsMaxChareQueueLength) = 0;
  CpvAccess(CstatsMaxForChareQueueLength) = 0;
  CpvAccess(CstatsMaxFixedChareQueueLength) = 0;

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
  CpvAccess(traceOn) = trace;
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
CpvStaticDeclare(int  , CmiHandlerMax);

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

CpvExtern(void*, CmiLocalQueue);
CtvStaticDeclare(int, CmiBufferGrabbed);

void CmiGrabBuffer(void **bufptrptr)
{
  CtvAccess(CmiBufferGrabbed) = 1;
}

void CmiHandleMessage(void *msg)
{
  CtvAccess(CmiBufferGrabbed) = 0;
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
      if (msg==0) FIFO_DeQueue(localqueue, &msg);
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
    if (msg==0) FIFO_DeQueue(localqueue, &msg);
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
  CpvInitialize(int,   CsdStopFlag);
  CpvInitialize(int,   CsdStopNotifyFlag);
  CpvInitialize(int,   CsdIdleDetectedFlag);
  CpvInitialize(CmiHandler,   CsdNotifyIdle);
  CpvInitialize(CmiHandler,   CsdNotifyBusy);
  
  CpvAccess(disable_sys_msgs) = 0;
  CpvAccess(CsdSchedQueue) = CqsCreate();
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
 * Converse Initialization
 *
 *****************************************************************************/

void ConverseCommonInit(char **argv)
{
  CstatsInit(argv);
  CcdModuleInit(argv);
  CmiHandlerInit();
  CmiMemoryInit(argv);
  CmiDeliversInit();
  CsdInit(argv);
  CthSchedInit();
  CldModuleInit();
}

void ConverseCommonExit(void)
{
  close_log();
}

/***************************************************************************
 *
 *  Memory Allocation routines 
 *
 *
 ***************************************************************************/

void *CmiAlloc(size)
int size;
{
  char *res;
  res =(char *)malloc(size+8);
  if (res==0) CmiAbort("Memory allocation failed.");
  ((int *)res)[0]=size;
  ((int *)((char *)res + 4))[0]=-1;    /* Reference count value */
  return (void *)(res+8);

}

int CmiSize(blk)
void *blk;
{
  return ((int *)(((char *)blk)-8))[0];
}

void CmiFree(blk)
void *blk;
{
  int offset;
  int refCount;

  refCount = ((int *)((char *)blk - 4))[0];

  /* Check if the reference count is -1 */
  if(refCount == -1){
    free(((char *)blk)-8);
  }
  else{
    CmiPrintf("Calling CmiFree in special case :\n");

    /* if the value is positive then it is a header having the actual refernce count value */
    /* else it is the actual offset to go all the way back */

    if(refCount >= 0){ /* This is the Header for the Multiple messages */

      CmiPrintf("in CmiFree (for header) : refCount = %d\n", refCount);
      
      if(((int *)((char *)blk - 4))[0] == 0){
	free(((char *)blk - 8));
	return;
      }
      ((int *)((char *)blk - 4))[0]--;
      if(((int *)((char *)blk - 4))[0] == 0){
	free(((char *)blk - 8));
      }
    }
    else {
      offset = refCount;
      
      CmiPrintf("in CmiFree : offset = %d\n", offset);
      CmiPrintf("in CmiFree : size = %d\n",((int *)((char *)blk - 8))[0]); 
      
      ((int *)((char *)blk + offset - 4))[0]--;
      
      CmiPrintf("in CmiFree : Ref Count : %d\n",((int *)((char *)blk + offset - 4))[0]);

      if(((int *)((char *)blk + offset - 4))[0] == 0){
	free(((char *)blk + offset - 8));
      }
    }
  }
}

/*********************************************************************************

  Multiple Send function                               

  ********************************************************************************/

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
  newSizes[1] = ((CmiMsgHeaderSizeBytes + (len + 1)*sizeof(int) + 7)&mask) - newSizes[0] + 8;
                     /* To allow the extra 8 bytes for the CmiSize & the Ref Count */

  for(i = 1; i < len + 1; i++){
    newSizes[2*i] = (sizes[i - 1]);
    newSizes[2*i + 1] = ((sizes[i -1] + 7)&mask) - newSizes[2*i] + 8; 
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

void CmiInitMultipleSendRoutine(void)
{
  CpvInitialize(int,CmiMainHandlerIDP); 
  CpvAccess(CmiMainHandlerIDP) = CmiRegisterHandler(CmiMultiMsgHandler);
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
  offset += 8;

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
  ((int *)(msgWhole - 4))[0] = len + 1;

  /* Allocate array to store sizes */
  sizes = (int *)(msgWhole + offset);
  offset += sizeof(int)*len;

  /* This is needed since the header may or may not be aligned on an 8 bit boundary */
  offset = (offset + 7)&mask;

  /* To cross the 8 bytes inserted in between */
  offset += 8;

  /* update the sizes and offsets for all the chunks */
  for(i = 0; i < len; i++){
    /* put in the size value for that part */
    ((int *)(msgWhole + offset - 8))[0] = sizes[i] - 8;
    
    /* now put in the offset (a negative value) to get right back to the begining */
    ((int *)(msgWhole + offset - 4))[0] = (-1)*offset;
    
    offset += sizes[i];
  }
}
