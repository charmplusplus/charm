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
#include <sys/rusage.h>
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
int sighold(sig) int sig;
{ if (sigblock(sigmask(sig)) < 0) return -1;
  else return 0; }
int sigrelse(sig) int sig;
{ if (sigsetmask(sigblock(0)&(~sigmask(sig))) < 0) return -1;
  else return 0; }
#endif

#define MAX_HANDLERS 512

void  *CmiGetNonLocal();

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
  CpvInitialize(int, CstatsMaxChareQueueLength);
  CpvInitialize(int, CstatsMaxForChareQueueLength);
  CpvInitialize(int, CstatsMaxFixedChareQueueLength);
  CpvInitialize(int, CstatPrintQueueStatsFlag);
  CpvInitialize(int, CstatPrintMemStatsFlag);

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
    } else
    argv++;
  }
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
CpvStaticDeclare(int, CmiHandlerCount);

int CmiRegisterHandler(handlerf)
CmiHandler handlerf ;
{
  CpvAccess(CmiHandlerTable)[CpvAccess(CmiHandlerCount)] = handlerf;
  CpvAccess(CmiHandlerCount)++ ;
  return CpvAccess(CmiHandlerCount)-1 ;
}

static void CmiHandlerInit()
{
  CpvInitialize(int, CmiHandlerCount);
  CpvInitialize(CmiHandler *, CmiHandlerTable);
  CpvAccess(CmiHandlerCount) = 0;
  CpvAccess(CmiHandlerTable) =
    (CmiHandler *)CmiAlloc((MAX_HANDLERS + 1) * sizeof(CmiHandler)) ;
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

static double  clocktick;
static int     inittime_wallclock;
static int     inittime_virtual;

void CmiTimerInit()
{
  struct tms temp;
  inittime_wallclock = times(&temp);
  inittime_virtual = temp.tms_utime + temp.tms_stime;
  clocktick = 1.0 / (sysconf(_SC_CLK_TCK));
}

double CmiWallTimer()
{
  struct tms temp;
  double currenttime;
  int now;

  now = times(&temp);
  currenttime = (now - inittime_wallclock) * clocktick;
  return (currenttime);
}

double CmiCpuTimer()
{
  struct tms temp;
  double currenttime;
  int now;

  times(&temp);
  now = temp.tms_stime + temp.tms_utime;
  currenttime = (now - inittime_virtual) * clocktick;
  return (currenttime);
}

double CmiTimer()
{
  return CmiCpuTimer();
}

#endif

#if CMK_TIMER_USE_GETRUSAGE

static double inittime_wallclock;
static double inittime_virtual;

void CmiTimerInit()
{
  struct timeval tv;
  struct rusage ru;
  gettimeofday(&tv);
  inittime_wallclock = (tv.tv_sec * 1.0) + (tv.tv_usec*0.000001);
  getrusage(0, &ru); 
  inittime_virtual =
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
  return currenttime - inittime_virtual;
}

double CmiWallTimer()
{
  struct timeval tv;
  double currenttime;

  getttimeofday(&tv);
  currenttime = (tv.tv_sec * 1.0) + (tv.tv_usec * 0.000001);
  return currenttime - inittime_wallclock;
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
 * Some possible versions of CmiEnableAsyncIO.  Not all machines use CmiEnableAsyncIO,
 * but it's common enough to belong in the common code.
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
void CmiSignal(sig, handler)
int sig;
void (*handler)();
{
  struct sigaction in, out ;
  in.sa_handler = handler;
  sigemptyset(&in.sa_mask);
  in.sa_flags = 0;
  sigaction(sig, &in, &out);
}
#endif

#if CMK_SIGNAL_USE_SIGACTION_WITH_RESTART
#include <signal.h>
void CmiSignal(sig, handler)
int sig;
void (*handler)();
{
  struct sigaction in, out ;
  in.sa_handler = handler ;
  sigemptyset(&in.sa_mask);
  in.sa_flags = SA_RESTART; 
  if(sigaction(sig, &in, &out) == -1)
      exit(1);
}
#endif

#if CMK_SIGNAL_IS_A_BUILTIN
#include <signal.h>
void CmiSignal(sig, handler)
int sig;
void (*handler)();
{
  signal(sig, handler) ;
}
#endif

/*****************************************************************************
 *
 * The following are the CmiDeliverXXX functions.  A common implementation
 * is provided below.  The machine layer can provide an alternate 
 * implementation if it so desires.
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

#if CMK_CMIDELIVERS_USE_COMMON_CODE

CpvStaticDeclare(int, CmiBufferGrabbed);
CpvExtern(void*, CmiLocalQueue);

void CmiDeliversInit()
{
  CpvInitialize(int, CmiBufferGrabbed);
  CpvAccess(CmiBufferGrabbed) = 0;
}

void CmiGrabBuffer()
{
  CpvAccess(CmiBufferGrabbed) = 1;
}

int CmiDeliverMsgs(maxmsgs)
int maxmsgs;
{
  void *msg1, *msg2;
  int counter;
  
  while (1) {
    msg1 = CmiGetNonLocal();
    if (msg1) {
      CpvAccess(CmiBufferGrabbed)=0;
      (CmiGetHandlerFunction(msg1))(msg1);
      if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg1);
      maxmsgs--; if (maxmsgs==0) break;
    }
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg2);
    if (msg2) {
      CpvAccess(CmiBufferGrabbed)=0;
      (CmiGetHandlerFunction(msg2))(msg2);
      if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg2);
      maxmsgs--; if (maxmsgs==0) break;
    }
    if ((msg1==0)&&(msg2==0)) break;
  }
  return maxmsgs;
}

/*
 * CmiDeliverSpecificMsg(lang)
 *
 * - waits till a message with the specified handler is received,
 *   then delivers it.
 *
 */

void CmiDeliverSpecificMsg(handler)
int handler;
{
  int msgType;
  int *msg, *first ;
  
  if ( !FIFO_Empty(CpvAccess(CmiLocalQueue)) ) {
    FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    first = msg;
    do {
      if (CmiGetHandler(msg)==handler) {
	CpvAccess(CmiBufferGrabbed)=0;
	(CmiGetHandlerFunction(msg))(msg);
	if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg);
	return;
      } else {
	FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
      }
      FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
    } while ( msg != first ) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
  }
  
  /* receive message from network */
  while ( 1 ) { /* Loop till proper message is received */
    while ( (msg = CmiGetNonLocal()) == NULL )
        ;
    if ( CmiGetHandler(msg)==handler ) {
      CpvAccess(CmiBufferGrabbed)=0;
      (CmiGetHandlerFunction(msg))(msg);
      if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg);
      return;
    } else {
      FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
    }
  }
}

#endif /* CMK_CMIDELIVERS_USE_COMMON_CODE */

/*****************************************************************************
 *
 * threads: common code.
 *
 * This section contains the following functions, which are common across
 * all implementations:
 *
 * void CthSchedInit()
 *
 *     This must be called before calling CthSetStrategyDefault.
 *
 * void CthSetStrategyDefault(CthThread t)
 *
 *     Sets the scheduling strategy for thread t to be the default strategy.
 *     All threads, when created, are set for the default strategy.  The
 *     default strategy is to awaken threads by inserting them into the
 *     main CsdScheduler queue, and to suspend them by returning control
 *     to the thread running the CsdScheduler.
 *
 *****************************************************************************/

CpvStaticDeclare(CthThread, CthSchedThreadVar);
CpvStaticDeclare(int, CthSchedResumeIndex);

static CthThread CthSchedThread()
{
  return CpvAccess(CthSchedThreadVar);
}

static void CthSchedResume(t)
CthThread t;
{
  CpvAccess(CthSchedThreadVar) = CthSelf();
  CthResume(t);
}

static void CthSchedEnqueue(t)
CthThread t;
{
  CmiSetHandler(t, CpvAccess(CthSchedResumeIndex));
  CsdEnqueueFifo(t);
}

void CthSchedInit()
{
  CpvInitialize(CthThread, CthSchedThreadVar);
  CpvInitialize(int, CthSchedResumeIndex);
  CpvAccess(CthSchedResumeIndex) = CmiRegisterHandler(CthSchedResume);
}

void CthSetStrategyDefault(t)
CthThread t;
{
  CthSetStrategy(t, CthSchedEnqueue, CthSchedThread);
}


/***************************************************************************
 *
 * 
 ***************************************************************************/


CsdInit(argv)
  char **argv;
{
  void *CqsCreate();

  CpvInitialize(int, disable_sys_msgs);
  CpvInitialize(int,   CmiHandlerCount);
  CpvInitialize(void*, CsdSchedQueue);
  CpvInitialize(int,   CsdStopFlag);
  
  CpvAccess(disable_sys_msgs) = 0;
  CpvAccess(CsdSchedQueue) = CqsCreate();
  CpvAccess(CsdStopFlag)  = 0;
}


int CsdScheduler(maxmsgs)
int maxmsgs;
{
  int *msg;
  
  CpvAccess(CsdStopFlag) = 0 ;
  
  while (1) {
    maxmsgs = CmiDeliverMsgs(maxmsgs);
    if (maxmsgs == 0) return maxmsgs;
    
    /* Check Scheduler queue */
    if ( !CqsEmpty(CpvAccess(CsdSchedQueue)) ) {
      CqsDequeue(CpvAccess(CsdSchedQueue),&msg);
      (CmiGetHandlerFunction(msg))(msg);
      if (CpvAccess(CsdStopFlag)) return maxmsgs;
      maxmsgs--; if (maxmsgs==0) return maxmsgs;
    } else { /* Processor is idle */
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
      struct timeval tv;
      tv.tv_usec=10000; tv.tv_sec=0;
      select(0,0,0,0,&tv);
#endif
      CcdRaiseCondition(CcdPROCESSORIDLE) ;
      if (CpvAccess(CsdStopFlag)) return maxmsgs;
    }
    
    if (!CpvAccess(disable_sys_msgs)) {
      if (CpvAccess(CcdNumChecks) > 0) {
	CcdCallBacks();
      }
    }
  }
}
 
/*****************************************************************************
 *
 * Fast Interrupt Blocking Device
 *
 * This is totally portable.  Go figure.  (The rest of it is just macros,
 * see converse.h)
 *
 *****************************************************************************/

CpvDeclare(int,       CmiInterruptsBlocked);
CpvDeclare(CthVoidFn, CmiInterruptFuncSaved);

CmiInterruptsInit()
{
  CpvInitialize(int, CmiInterruptsBlocked);
  CpvInitialize(CthVoidFn, CmiInterruptFuncSaved);
  CpvAccess(CmiInterruptsBlocked) = 0;
  CpvAccess(CmiInterruptFuncSaved) = 0;
}

/*****************************************************************************
 *
 * ConverseInit and ConverseExit
 *
 *****************************************************************************/

ConverseInit(argv)
char **argv;
{
  CstatsInit(argv);
  conv_condsModuleInit(argv);
  CmiHandlerInit();
  CmiMemoryInit(argv);
  CmiDeliversInit();
  /* CmiSpanTreeInit(argv); done in CmiInitMc()  -- Sanjeev 3/5/96 */
  CmiInterruptsInit();
  CmiInitMc(argv);
  CsdInit(argv);
#if CMK_CTHINIT_IS_IN_CONVERSEINIT
  CthInit(argv);
#endif
  CthSchedInit();
}

ConverseExit()
{
  CmiExit();
}

