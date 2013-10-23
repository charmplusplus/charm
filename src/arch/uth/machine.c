/** @file
 * User-level threads machine layer
 * @ingroup Machine
 * @{
 */

#include <stdio.h>
#include <math.h>
#include "converse.h"

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

int CmiAsyncMsgSent(c)
CmiCommHandle c ;
{
  return 1;
}

void CmiReleaseCommHandle(c)
CmiCommHandle c ;
{
}

/********************* CONTEXT-SWITCHING FUNCTIONS ******************/

static void CmiNext()
{
  CthThread t; int index; int orig;
  index = (CmiMyPe()+1) % CmiNumPes();
  orig = index;
  while (1) {
    t = CmiThreads[index];
    if ((t)&&(!CmiBarred[index])) break;
    index = (index+1) % CmiNumPes();
    if (index == orig) exit(0);
  }
  _Cmi_mype = index;
  CthResume(t);
}

void CmiExit()
{
  CmiThreads[CmiMyPe()] = 0;
  CmiFree(CthSelf());
  CmiNext();
}

void *CmiGetNonLocal()
{
  CmiThreads[CmiMyPe()] = CthSelf();
  CmiNext();
  return 0;
}

void CmiNotifyIdle()
{
  CmiThreads[CmiMyPe()] = CthSelf();
  CmiNext();
}

void CmiNodeBarrier()
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

void CmiNodeAllBarrier()
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



/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
  char *buf = (char *)CmiAlloc(size);
  memcpy(buf,msg,size);
  CdsFifo_Enqueue(CmiQueues[destPE],buf);
  CQdCreate(CpvAccess(cQdState), 1);
}

CmiCommHandle CmiAsyncSendFn(destPE, size, msg) 
int destPE;
int size;
char * msg;
{
  char *buf = (char *)CmiAlloc(size);
  memcpy(buf,msg,size);
  CdsFifo_Enqueue(CmiQueues[destPE],buf);
  CQdCreate(CpvAccess(cQdState), 1);
  return 0;
}

void CmiFreeSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
  CdsFifo_Enqueue(CmiQueues[destPE], msg);
  CQdCreate(CpvAccess(cQdState), 1);
}

void CmiSyncBroadcastFn(size, msg)
int size;
char * msg;
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    if (i != CmiMyPe()) CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastFn(size, msg)
int size;
char * msg;
{
  CmiSyncBroadcastFn(size, msg);
  return 0;
}

void CmiFreeBroadcastFn(size, msg)
int size;
char * msg;
{
  CmiSyncBroadcastFn(size, msg);
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
  CmiSyncBroadcastAllFn(size,msg);
  return 0 ;
}

void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
    if (i!=CmiMyPe()) CmiSyncSendFn(i,size,msg);
  CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
  CQdCreate(CpvAccess(cQdState), 1);
}


int CmiBarrier()
{
  return -1;
}

int CmiBarrierZero()
{
  return -1;
}

/************************** SETUP ***********************************/

static void CmiParseArgs(argv)
char **argv;
{
  CmiGetArgInt(argv,"++stacksize",&Cmi_stacksize);
  _Cmi_numpes=1;
  CmiGetArgInt(argv,"+p",&_Cmi_numpes);
  if (CmiNumPes()<1) {
    printf("Error: must specify number of processors to simulate with +pXXX\n",CmiNumPes());
    exit(1);
  }
}

char **CmiInitPE()
{
  int argc; char **argv;
  argv = CmiCopyArgs(CmiArgv);
  CpvAccess(CmiLocalQueue) = CmiQueues[CmiMyPe()];
  CmiTimerInit(argv);
  ConverseCommonInit(argv);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,CmiNotifyIdle,NULL);
  return argv;
}

void CmiCallMain()
{
  char **argv;
  argv = CmiInitPE();
  CmiStart(CmiGetArgc(argv), argv);
  if (CmiUsched==0) CsdScheduler(-1);
  ConverseExit();
}

void ConverseExit()
{
  ConverseCommonExit();
  CmiThreads[CmiMyPe()] = 0;
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


void ConverseInit(argc,argv,fn,usched,initret)
int argc;
char **argv;
CmiStartFn fn;
int usched, initret;
{
  CthThread t; int stacksize, i;
  
  CmiSwitchToPE = CmiSwitchToPEFn;

  CmiArgv = CmiCopyArgs(argv);
  CmiStart = fn;
  CmiUsched = usched;
  CmiParseArgs(CmiArgv);
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
