/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

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

int        Cmi_mype;
int        Cmi_myrank;
int        Cmi_numpes;
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
  Cmi_mype = index;
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



/************************** SETUP ***********************************/

static void CmiParseArgs(argv)
char **argv;
{
  char **argp;
  
  for (argp=argv; *argp; ) {
    if ((strcmp(*argp,"++stacksize")==0)&&(argp[1])) {
      DeleteArg(argp);
      Cmi_stacksize = atoi(*argp);
      DeleteArg(argp);
    } else if ((strcmp(*argp,"+p")==0)&&(argp[1])) {
      Cmi_numpes = atoi(argp[1]);
      argp+=2;
    } else if (sscanf(*argp, "+p%d", &CmiNumPes()) == 1) {
      argp+=1;
    } else argp++;
  }
  
  if (CmiNumPes()<1) {
    printf("Error: must specify number of processors to simulate with +pXXX\n",CmiNumPes());
    exit(1);
  }
}

static char **CopyArgvec(char **src)
{
  int argc; char **argv;
  for (argc=0; src[argc]; argc++);
  argv = (char **)malloc((argc+1)*sizeof(char *));
  memcpy(argv, src, (argc+1)*sizeof(char *));
  return argv;
}

char **CmiInitPE()
{
  int argc; char **argv;
  argv = CopyArgvec(CmiArgv);
  CpvAccess(CmiLocalQueue) = CmiQueues[CmiMyPe()];
  CmiTimerInit();
  ConverseCommonInit(argv);
  return argv;
}

void CmiCallMain()
{
  char **argv;
  int argc;
  argv = CmiInitPE();
  for (argc=0; argv[argc]; argc++);
  CmiStart(argc, argv);
  if (CmiUsched==0) CsdScheduler(-1);
  ConverseExit();
}

void ConverseExit()
{
  ConverseCommonExit();
  CmiThreads[CmiMyPe()] = 0;
  CmiNext();
}

void ConverseInit(argc,argv,fn,usched,initret)
int argc;
char **argv;
CmiStartFn fn;
int usched, initret;
{
  CthThread t; int stacksize, i;
  
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  
  CmiArgv = CopyArgvec(argv);
  CmiStart = fn;
  CmiUsched = usched;
  CmiParseArgs(argv);
  CthInit(argv);
  CpvInitialize(void*, CmiLocalQueue);
  CmiThreads = (CthThread *)CmiAlloc(CmiNumPes()*sizeof(CthThread));
  CmiBarred  = (int       *)CmiAlloc(CmiNumPes()*sizeof(int));
  CmiQueues  = (void**)CmiAlloc(CmiNumPes()*sizeof(void*));
  
  /* Create threads for all PE except PE 0 */
  for(i=0; i<CmiNumPes(); i++) {
    t = (i==0) ? CthSelf() : CthCreate(CmiCallMain, 0, Cmi_stacksize);
    CmiThreads[i] = t;
    CmiBarred[i] = 0;
    CmiQueues[i] = CdsFifo_Create();
  }
  Cmi_mype = 0;
  argv = CmiInitPE();
  if (initret==0) {
    fn(CountArgs(argv), argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

