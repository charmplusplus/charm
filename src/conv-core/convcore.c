/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "converse.h"
#include "conv-trace.h"
#include "sockRoutines.h"
#include "queueing.h"
#include "conv-ccs.h"
#include "ccs-server.h"
#include "memory-isomalloc.h"

extern void CcdModuleInit(char **);
extern void CmiMemoryInit(char **);
extern void CldModuleInit(void);

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

#if CMK_TIMER_USE_RDTSC
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

#ifdef CMK_TIMER_USE_WIN32API
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/timeb.h>
#endif

#include "quiescence.h"

/*****************************************************************************
 *
 * Unix Stub Functions
 *
 ****************************************************************************/

#ifdef MEMMONITOR
typedef unsigned long mmulong;
CpvDeclare(mmulong,MemoryUsage);
CpvDeclare(mmulong,HiWaterMark);
CpvDeclare(mmulong,ReportedHiWaterMark);
CpvDeclare(int,AllocCount);
CpvDeclare(int,BlocksAllocated);
#endif

#define MAX_HANDLERS 512

#if CMK_NODE_QUEUE_AVAILABLE
void  *CmiGetNonLocalNodeQ();
#endif

CpvDeclare(void*, CsdSchedQueue);
#if CMK_NODE_QUEUE_AVAILABLE
CsvDeclare(void*, CsdNodeQueue);
CsvDeclare(CmiNodeLock, CsdNodeQueueLock);
#endif
CpvDeclare(int,   CsdStopFlag);

/*****************************************************************************
 *
 * Argument parsing routines.
 *
 *****************************************************************************/

/*Count the number of non-NULL arguments in list*/
int CmiGetArgc(char **argv)
{
	int i=0,argc=0;
	while (argv[i++]!=NULL)
		argc++;
	return argc;
}

/*Return a new, heap-allocated copy of the argv array*/
char **CmiCopyArgs(char **argv)
{
	int argc=CmiGetArgc(argv);
	char **ret=(char **)malloc(sizeof(char *)*(argc+1));
	int i;
	for (i=0;i<=argc;i++)
		ret[i]=argv[i];
	return ret;
}

/*Delete the first k argument from the given list, shifting
all other arguments down by k spaces.
e.g., argv=={"a","b","c","d",NULL}, k==3 modifies
argv={"d",NULL,"c","d",NULL}
*/
void CmiDeleteArgs(char **argv,int k)
{
	int i=0;
	while ((argv[i]=argv[i+k])!=NULL)
		i++;
}

/*Find the given argment and string option in argv.
If the argument is present, set the string option and
delete both from argv.  If not present, return NULL.
e.g., arg=="-name" returns "bob" from
argv=={"a.out","foo","-name","bob","bar"},
and sets argv={"a.out","foo","bar"};
*/
int CmiGetArgString(char **argv,const char *arg,char **optDest)
{
	int i;
	for (i=0;argv[i]!=NULL;i++)
		if (0==strcmp(argv[i],arg))
		{/*We found the argument*/
			if (argv[i+1]==NULL) CmiAbort("Argument not complete!");
			*optDest=argv[i+1];
			CmiDeleteArgs(&argv[i],2);
			return 1;
		}
	return 0;/*Didn't find the argument*/
}

/*Find the given argument and numeric option in argv.
If the argument is present, parse and set the numeric option,
delete both from argv, and return 1. If not present, return 0.
e.g., arg=="-pack" matches argv=={...,"-pack","27",...},
argv=={...,"-pack0xf8",...}, and argv=={...,"-pack=0777",...};
but not argv=={...,"-packsize",...}.
*/
int CmiGetArgInt(char **argv,const char *arg,int *optDest)
{
	int i;
	int argLen=strlen(arg);
	for (i=0;argv[i]!=NULL;i++)
		if (0==strncmp(argv[i],arg,argLen))
		{/*We *may* have found the argument*/
			const char *opt=NULL;
			int nDel=0;
			switch(argv[i][argLen]) {
			case 0: /* like "-p","27" */
				opt=argv[i+1]; nDel=2; break;
			case '=': /* like "-p=27" */
				opt=&argv[i][argLen+1]; nDel=1; break;
			case '-':case '+':
			case '0':case '1':case '2':case '3':case '4':
			case '5':case '6':case '7':case '8':case '9':
				/* like "-p27" */
				opt=&argv[i][argLen]; nDel=1; break;
			default:
				continue; /*False alarm-- skip it*/
			}
			if (opt==NULL) continue; /*False alarm*/
			if (sscanf(opt,"%i",optDest)<1) {
			/*Bad command line argument-- die*/
				fprintf(stderr,"Cannot parse %s option '%s' "
					"as an integer.\n",arg,opt);
				CmiAbort("Bad command-line argument\n");
			}
			CmiDeleteArgs(&argv[i],nDel);
			return 1;
		}
	return 0;/*Didn't find the argument-- dest is unchanged*/	
}

/*Find the given argument in argv.  If present, delete
it and return 1; if not present, return 0.
e.g., arg=="-foo" matches argv=={...,"-foo",...} but not
argv={...,"-foobar",...}.
*/
int CmiGetArgFlag(char **argv,const char *arg)
{
	int i;
	for (i=0;argv[i]!=NULL;i++)
		if (0==strcmp(argv[i],arg))
		{/*We found the argument*/
			CmiDeleteArgs(&argv[i],1);
			return 1;
		}
	return 0;/*Didn't find the argument*/
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

  if (CmiGetArgFlag(argv,"+mems"))
    CpvAccess(CstatPrintMemStatsFlag)=1;
  if (CmiGetArgFlag(argv,"+qs"))
    CpvAccess(CstatPrintQueueStatsFlag)=1;

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
    CmiHandler *new = (CmiHandler*)malloc(newbytes);
    _MEMCHECK(new);
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

static double inittime_wallclock;
CpvStaticDeclare(double, inittime_virtual);

void CmiTimerInit()
{
  struct timeval tv;
  struct rusage ru;
  CpvInitialize(double, inittime_virtual);
  gettimeofday(&tv,0);
  inittime_wallclock = (tv.tv_sec * 1.0) + (tv.tv_usec*0.000001);
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
  return currenttime - inittime_wallclock;
}

double CmiTimer()
{
  return CmiCpuTimer();
}

#endif

#if CMK_TIMER_USE_RDTSC

static double readMHz(void)
{
  double x;
  char str[1000];
  char buf[100];
  FILE *fp = fopen("/proc/cpuinfo", "r");
  while(fgets(str, 1000, fp)!=0) {
    if(sscanf(str, "cpu MHz%[^:]",buf)==1)
    {
      char *s = strchr(str, ':'); s=s+1;
      sscanf(s, "%lf", &x);
      fclose(fp);
      return x;
    }
  }
  CmiAbort("Cannot read CPU MHz from /proc/cpuinfo file.");
  return 0.0;
}

double cpu_speed_factor;
CpvStaticDeclare(double, inittime_virtual);

void CmiTimerInit()
{
  struct rusage ru;
  cpu_speed_factor = 1.0/(readMHz()*1.0e6); 
  rdtsc(); rdtsc(); rdtsc(); rdtsc(); rdtsc();
  CpvInitialize(double, inittime_virtual);
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

#endif

#if CMK_TIMER_USE_WIN32API

CpvStaticDeclare(double, inittime_wallclock);
CpvStaticDeclare(double, inittime_virtual);

void CmiTimerInit()
{
#ifdef __CYGWIN__
	struct timeb tv;
#else
	struct _timeb tv;
#endif
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
#ifdef __CYGWIN__
	struct timeb tv;
#else
	struct _timeb tv;
#endif
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
 * For this common implementation to work, the machine layer must provide the
 * following:
 *
 * void *CmiGetNonLocal()
 *
 *      - returns a message just retrieved from some other PE, not from
 *        local.  If no such message exists, returns 0.
 *
 * CpvExtern(CdsFifo, CmiLocalQueue);
 *
 *      - a FIFO queue containing all messages from the local processor.
 *
 *****************************************************************************/

void CsdBeginIdle(void)
{
  CcdCallBacks();
  CcdRaiseCondition(CcdPROCESSOR_BEGIN_IDLE) ;
}

void CsdStillIdle(void)
{
  CcdRaiseCondition(CcdPROCESSOR_STILL_IDLE);
}

void CsdEndIdle(void)
{
  CcdRaiseCondition(CcdPROCESSOR_BEGIN_BUSY) ;
}

void CmiHandleMessage(void *msg)
{
	CpvAccess(cQdState)->mProcessed++;
	(CmiGetHandlerFunction(msg))(msg);
}

#if CMK_CMIDELIVERS_USE_COMMON_CODE

void CmiDeliversInit()
{
}

int CmiDeliverMsgs(int maxmsgs)
{
  return CsdScheduler(maxmsgs);
}

void CsdSchedulerState_new(CsdSchedulerState_t *s)
{
	s->localQ=CpvAccess(CmiLocalQueue);
	s->schedQ=CpvAccess(CsdSchedQueue);
#if CMK_NODE_QUEUE_AVAILABLE
	s->nodeQ=CsvAccess(CsdNodeQueue);
	s->nodeLock=CsvAccess(CsdNodeQueueLock);
#endif
}

void *CsdNextMessage(CsdSchedulerState_t *s) {
	void *msg;
	if (NULL!=(msg=CmiGetNonLocal())) return msg;
	if (NULL!=(msg=CdsFifo_Dequeue(s->localQ))) return msg;
#if CMK_NODE_QUEUE_AVAILABLE
	if (NULL!=(msg=CmiGetNonLocalNodeQ())) return msg;
	if (!CqsEmpty(s->nodeQ)
	 && !CqsPrioGT(CqsGetPriority(s->nodeQ),
		       CqsGetPriority(s->schedQ))) {
	  CmiLock(s->nodeLock);
	  CqsDequeue(s->nodeQ,(void **)&msg);
	  CmiUnlock(s->nodeLock);
	  if (msg!=NULL) return msg;
	}
#endif
	CqsDequeue(s->schedQ,(void **)&msg);
	if (msg!=NULL) return msg;
	return NULL;
}

int CsdScheduler(int maxmsgs)
{
	if (maxmsgs<0) CsdScheduleForever();	
	else if (maxmsgs==0)
		CsdSchedulePoll();
	else /*(maxmsgs>0)*/ 
		return CsdScheduleCount(maxmsgs);
	return 0;
}

/*Declare the standard scheduler housekeeping*/
#define SCHEDULE_TOP \
      void *msg;\
      int cycle = CpvAccess(CsdStopFlag); \
      CsdSchedulerState_t state;\
      CsdSchedulerState_new(&state);\

/*A message is available-- process it*/
#define SCHEDULE_MESSAGE \
      CmiHandleMessage(msg);\
      if (CpvAccess(CsdStopFlag) != cycle) break;\

/*No message available-- go (or remain) idle*/
#define SCHEDULE_IDLE \
      if (!isIdle) {isIdle=1;CsdBeginIdle();}\
      else CsdStillIdle();\
      if (CpvAccess(CsdStopFlag) != cycle) {\
	CsdEndIdle();\
	break;\
      }\

void CsdScheduleForever(void)
{
  int isIdle=0;
  SCHEDULE_TOP
  while (1) {
    msg = CsdNextMessage(&state);
    if (msg) { /*A message is available-- process it*/
      if (isIdle) {isIdle=0;CsdEndIdle();}
      SCHEDULE_MESSAGE
    } else { /*No message available-- go (or remain) idle*/
      SCHEDULE_IDLE
    }
    CsdPeriodic();
  }
}
int CsdScheduleCount(int maxmsgs)
{
  int isIdle=0;
  SCHEDULE_TOP
  while (1) {
    msg = CsdNextMessage(&state);
    if (msg) { /*A message is available-- process it*/
      if (isIdle) {isIdle=0;CsdEndIdle();}
      maxmsgs--; 
      SCHEDULE_MESSAGE
      if (maxmsgs==0) break;
    } else { /*No message available-- go (or remain) idle*/
      SCHEDULE_IDLE
    }
    CsdPeriodic();
  }
  return maxmsgs;
}

void CsdSchedulePoll(void)
{
  SCHEDULE_TOP
  while (1)
  {
	CsdPeriodic();
	if (NULL!=(msg = CsdNextMessage(&state)))
	{
	     SCHEDULE_MESSAGE 
     	}
	else break;
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
    else      msg = CdsFifo_Dequeue(localqueue);
    if (msg) {
      if (CmiGetHandler(msg)==handler) {
	CpvAccess(cQdState)->mProcessed++;
	CmiHandleMessage(msg);
	return;
      } else {
	CdsFifo_Enqueue(localqueue, msg);
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


void CthStandinCode()
{
  while (1) CsdScheduler(0);
}

CthThread CthSuspendNormalThread()
{
  return CpvAccess(CthSchedulingThread);
}

void CthEnqueueSchedulingThread(CthThread t, int, int, unsigned int*);
CthThread CthSuspendSchedulingThread();

CthThread CthSuspendSchedulingThread()
{
  CthThread succ = CpvAccess(CthSleepingStandins);

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

void CthResumeNormalThread(CthThread t)
{
#ifndef CMK_OPTIMIZE
  if(CpvAccess(traceOn))
    traceResume();
#endif
  CthResume(t);
}

void CthResumeSchedulingThread(CthThread t)
{
  CthThread me = CthSelf();
  if (me == CpvAccess(CthMainThread)) {
    CthEnqueueSchedulingThread(me,CQS_QUEUEING_FIFO, 0, 0);
  } else {
    CthSetNext(me, CpvAccess(CthSleepingStandins));
    CpvAccess(CthSleepingStandins) = me;
  }
  CpvAccess(CthSchedulingThread) = t;
  CthResume(t);
}

void CthEnqueueNormalThread(CthThread t, int s, 
				   int pb,unsigned int *prio)
{
  CmiSetHandler(t, CpvAccess(CthResumeNormalThreadIdx));
  CsdEnqueueGeneral(t, s, pb, prio);
}

void CthEnqueueSchedulingThread(CthThread t, int s, 
				       int pb,unsigned int *prio)
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
  CpvInitialize(void *, CsdSchedQueue);
  CpvInitialize(int,   CsdStopFlag);
  
  CpvAccess(CsdSchedQueue) = (void *)CqsCreate();

#if CMK_NODE_QUEUE_AVAILABLE
  CsvInitialize(CmiLock, CsdNodeQueueLock);
  CsvInitialize(void *, CsdNodeQueue);
  if (CmiMyRank() ==0) {
	CsvAccess(CsdNodeQueueLock) = CmiCreateLock();
	CsvAccess(CsdNodeQueue) = (void *)CqsCreate();
  }
  CmiNodeBarrier();
#endif

  CpvAccess(CsdStopFlag)  = 0;
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
  /*CmiError("ListSend not implemented.");*/
  for(i=0;i<npes;i++) {
    CmiSyncSend(pes[i], len, msg);
  }
  CmiFree(msg);
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

#define SIMPLE_CMIALLOC 0

#if SIMPLE_CMIALLOC
void *CmiAlloc(int size)
{
	return malloc_nomigrate(size);
}

void CmiReference(void *blk)
{
	CmiAbort("CmiReference not supported!\n");
}

int CmiSize(void *blk)
{
	CmiAbort("CmiSize not supported!\n");
	return 0;
}

void CmiFree(void *blk)
{
	free_nomigrate(blk);
}

#else /*!SIMPLE_CMIALLOC*/

#define SIZEFIELD(m) ((int *)((char *)(m)-2*sizeof(int)))[0]
#define REFFIELD(m) ((int *)((char *)(m)-sizeof(int)))[0]
#define BLKSTART(m) ((char *)m-2*sizeof(int))

void *CmiAlloc(size)
int size;
{
  char *res;
  res =(char *)malloc_nomigrate(size+2*sizeof(int));
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
    free_nomigrate(BLKSTART(blk));
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
    free_nomigrate(BLKSTART(blk));
    return;
  }
  REFFIELD(blk) = refCount;
}
#endif /*!SIMPLE_CMIALLOC*/

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



/******** Idle timeout module (+idletimeout=30) *********/

typedef struct {
  int idle_timeout;/*Milliseconds to wait idle before aborting*/
  int is_idle;/*Boolean currently-idle flag*/
  int call_count;/*Number of timeout calls currently in flight*/
} cmi_cpu_idlerec;

static void on_timeout(cmi_cpu_idlerec *rec)
{
  rec->call_count--;
  if(rec->call_count==0 && rec->is_idle==1) {
    CmiError("Idle time on PE %d exceeded specified timeout.\n", CmiMyPe());
    CmiAbort("Exiting.\n");
  }
}
static void on_idle(cmi_cpu_idlerec *rec)
{
  CcdCallFnAfter((CcdVoidFn)on_timeout, rec, rec->idle_timeout);
  rec->call_count++; /*Keeps track of overlapping timeout calls.*/  
  rec->is_idle = 1;
}
static void on_busy(cmi_cpu_idlerec *rec)
{
  rec->is_idle = 0;
}
static void CIdleTimeoutInit(char **argv)
{
  int idle_timeout=0; /*Seconds to wait*/
  CmiGetArgInt(argv,"+idle-timeout",&idle_timeout);
  if(idle_timeout != 0) {
    cmi_cpu_idlerec *rec=(cmi_cpu_idlerec *)malloc(sizeof(cmi_cpu_idlerec));
    _MEMCHECK(rec);
    rec->idle_timeout=idle_timeout*1000;
    rec->is_idle=0;
    rec->call_count=0;
    CcdCallOnCondition(CcdPROCESSOR_BEGIN_IDLE, (CcdVoidFn)on_idle, rec);
    CcdCallOnCondition(CcdPROCESSOR_BEGIN_BUSY, (CcdVoidFn)on_busy, rec);
  }
}


/*****************************************************************************
 *
 * Converse Initialization
 *
 *****************************************************************************/

extern void CrnInit(void);
extern void CmiIsomallocInit(char **argv);

void ConverseCommonInit(char **argv)
{
  CmiMemoryInit(argv);
  CmiTimerInit();
  CstatsInit(argv);
  CcdModuleInit(argv);
  CmiHandlerInit();
#ifndef CMK_OPTIMIZE
  traceInit(argv);
#endif
#if CMK_CCS_AVAILABLE
  CcsInit(argv);
#endif
  CmiIsomallocInit(argv);
  CpdInit();
  CmiDeliversInit();
  CsdInit(argv);
  CthSchedInit();
  CmiGroupInit();
  CmiMulticastInit();
  CmiInitMultipleSend();
  CQdInit();

  CldModuleInit();
  CrnInit();
  CIdleTimeoutInit(argv);
}

void ConverseCommonExit(void)
{
  CcsImpl_kill();

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

void __cmi_assert(const char *expr, const char *file, int line)
{
  CmiError("[%d] Assertion \"%s\" failed in file %s line %d.\n",
      CmiMyPe(), expr, file, line);
  CmiAbort("");
}
