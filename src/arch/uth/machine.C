/** @file
 * User-level threads machine layer
 * @ingroup Machine
 * @{
 */

#include <stdio.h>
#include <math.h>
#include "converse.h"
#include <atomic>

int               userDrivenMode; /* Set by CharmInit for interop in user driven mode */
std::atomic<int> ckExitComplete {0};

void CthInit(char **);
void ConverseCommonInit(char **);
void ConverseCommonExit(void);

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
  CmiError("%s\n", newmsg);
  exit(1);
  CMI_NORETURN_FUNCTION_END
}

/*****************************************************************************
 *
 * Module variables
 * 
 ****************************************************************************/

int        _Cmi_mype;
int        _Cmi_myrank;
int        _Cmi_numpes;
int        Cmi_nodesize;
int        Cmi_stacksize = 64000;
char     **CmiArgv;
CmiStartFn CmiStart;
int        CmiUsched;
CthThread *CmiThreads;
void*      *CmiQueues;
int       *CmiBarred;
int        CmiNumBarred=0;

CpvDeclare(void*, CmiLocalQueue);

/*****************************************************************************
 *
 * Comm handles are nonexistent in uth version
 *
 *****************************************************************************/

int CmiAsyncMsgSent(CmiCommHandle c)
{
  return 1;
}

void CmiReleaseCommHandle(CmiCommHandle c)
{
}

/********************* CONTEXT-SWITCHING FUNCTIONS ******************/
static int _exitcode = 0;

static void CmiNext(void)
{
  CthThread t; int index; int orig;
  index = (CmiMyPe()+1) % CmiNumPes();
  orig = index;
  while (1) {
    t = CmiThreads[index];
    if ((t)&&(!CmiBarred[index])) break;
    index = (index+1) % CmiNumPes();
    if (index == orig) exit(_exitcode);
  }
  _Cmi_mype = index;
  CthResume(t);
}

void CmiExit(void)
{
  CmiThreads[CmiMyPe()] = 0;
  CmiFree(CthSelf());
  CmiNext();
}

void *CmiGetNonLocal(void)
{
  CmiThreads[CmiMyPe()] = CthSelf();
  CmiNext();
  return 0;
}

void CmiNotifyIdle(void)
{
  CmiThreads[CmiMyPe()] = CthSelf();
  CmiNext();
}

static void CmiNotifyIdleCcd(void *ignored1, double ignored2)
{
    CmiNotifyIdle();
}

void CmiNodeBarrier(void)
{
  int i;
  CmiNumBarred++;
  CmiBarred[CmiMyPe()] = 1;
  if (CmiNumBarred == CmiNumPes()) {
    for (i=0; i<CmiNumPes(); i++) CmiBarred[i]=0;
    CmiNumBarred=0;
  }
  CmiGetNonLocal();
}

void CmiNodeAllBarrier(void)
{
  int i;
  CmiNumBarred++;
  CmiBarred[CmiMyPe()] = 1;
  if (CmiNumBarred == CmiNumPes()) {
    for (i=0; i<CmiNumPes(); i++) CmiBarred[i]=0;
    CmiNumBarred=0;
  }
  CmiGetNonLocal();
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



/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(int destPE, int size, char *msg)
{
  char *buf = (char *)CmiAlloc(size);
  memcpy(buf,msg,size);
  CdsFifo_Enqueue(CmiQueues[destPE],buf);
#if CMI_QD
  CQdCreate(CpvAccess(cQdState), 1);
#endif
}

CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  char *buf = (char *)CmiAlloc(size);
  memcpy(buf,msg,size);
  CdsFifo_Enqueue(CmiQueues[destPE],buf);
#if CMI_QD
  CQdCreate(CpvAccess(cQdState), 1);
#endif
  return 0;
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  CdsFifo_Enqueue(CmiQueues[destPE], msg);
#if CMI_QD
  CQdCreate(CpvAccess(cQdState), 1);
#endif
}

void CmiSyncBroadcastFn(int size, char *msg)
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    if (i != CmiMyPe()) CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size, msg);
  return 0;
}

void CmiFreeBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size, msg);
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(int size, char *msg)
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)
{
  CmiSyncBroadcastAllFn(size,msg);
  return 0 ;
}

void CmiFreeBroadcastAllFn(int size, char *msg)
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    if (i!=CmiMyPe()) CmiSyncSendFn(i,size,msg);
  CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
#if CMI_QD
  CQdCreate(CpvAccess(cQdState), 1);
#endif
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

int CmiBarrier(void)
{
  return -1;
}

int CmiBarrierZero(void)
{
  return -1;
}

/************************** SETUP ***********************************/

static void CmiParseArgs(char **argv)
{
  CmiGetArgInt(argv,"++stacksize",&Cmi_stacksize);
  _Cmi_numpes=1;
  CmiGetArgInt(argv,"+p",&_Cmi_numpes);
  if (CmiNumPes()<1) {
    CmiAbort("Error: must specify number of processors to simulate with +pXXX");
  }
}

char **CmiInitPE(void)
{
  int argc; char **argv;
  argv = CmiCopyArgs(CmiArgv);
  CpvAccess(CmiLocalQueue) = CmiQueues[CmiMyPe()];
  CmiTimerInit(argv);
  ConverseCommonInit(argv);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE, CmiNotifyIdleCcd, NULL);
  return argv;
}

static void CmiCallMain(void *ignored)
{
  char **argv;
  argv = CmiInitPE();
  CmiStart(CmiGetArgc(argv), argv);
  if (CmiUsched==0) CsdScheduler(-1);
  ConverseExit();
}

void ConverseExit(int exitcode)
{
  ConverseCommonExit();
  CmiThreads[CmiMyPe()] = 0;
  _exitcode = exitcode; // Used in CmiNext()
  CmiNext();
}

#if CMK_CONDS_USE_SPECIAL_CODE
static int CmiSwitchToPEFn(int newpe)
{
  int oldpe = _Cmi_mype;
  if (newpe == CcdIGNOREPE) return CcdIGNOREPE;
  _Cmi_mype = newpe;
  return oldpe;
}
#endif


void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  CthThread t; int stacksize, i;
  
  CmiSwitchToPE = CmiSwitchToPEFn;

  CmiArgv = CmiCopyArgs(argv);
  CmiStart = fn;
  CmiUsched = usched;
  CmiParseArgs(CmiArgv);

  CmiInitHwlocTopology();

  CthInit(CmiArgv);
  CpvInitialize(void*, CmiLocalQueue);
  CmiThreads = (CthThread *)CmiAlloc(CmiNumPes()*sizeof(CthThread));
  CmiBarred  = (int       *)CmiAlloc(CmiNumPes()*sizeof(int));
  CmiQueues  = (void**)CmiAlloc(CmiNumPes()*sizeof(void*));
  
  _smp_mutex = CmiCreateLock();

  /* Create threads for all PE except PE 0 */
  for(i=0; i<CmiNumPes(); i++) {
    t = (i==0) ? CthSelf() : CthCreate(CmiCallMain, 0, Cmi_stacksize);
    CmiThreads[i] = t;
    CmiBarred[i] = 0;
    CmiQueues[i] = CdsFifo_Create();
  }
  _Cmi_mype = 0;
  argv = CmiInitPE();
  if (initret==0) {
    fn(CmiGetArgc(argv), argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

/*@}*/
