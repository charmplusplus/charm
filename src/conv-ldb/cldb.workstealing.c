#include <stdlib.h>

#include "converse.h"
#include "cldb.workstealing.h"
#include "queueing.h"
#include "cldb.h"

#define IDLE_IMMEDIATE 		1
#define TRACE_USEREVENTS        0

#define PERIOD 20                /* default: 30 */
#define MSGDELAY 10
#define MAXOVERLOAD 1

#define LOADTHRESH       3


typedef struct CldProcInfo_s {
  double lastCheck;
  int    sent;			/* flag to disable idle work request */
  int    balanceEvt;		/* user event for balancing */
  int    idleEvt;		/* user event for idle balancing */
  int    idleprocEvt;		/* user event for processing idle req */
  double lastBalanceTime;
} *CldProcInfo;


CpvStaticDeclare(CldProcInfo, CldData);
CpvStaticDeclare(int, CldAskLoadHandlerIndex);

void LoadNotifyFn(int l)
{
  CldProcInfo  cldData = CpvAccess(CldData);
  cldData->sent = 0;
}

char *CldGetStrategy(void)
{
  return "work stealing";
}

/* since I am idle, ask for work from neighbors */
static void CldBeginIdle(void *dummy)
{
  CpvAccess(CldData)->lastCheck = CmiWallTimer();
}

static void CldEndIdle(void *dummy)
{
  CpvAccess(CldData)->lastCheck = -1;
}

static void CldStillIdle(void *dummy, double curT)
{
  int i;
  double startT;
  requestmsg msg;
  int myload;
  CldProcInfo  cldData = CpvAccess(CldData);
  int  victim;

  double now = curT;
  double lt = cldData->lastCheck;
  /* only ask for work every 20ms */
  if (cldData->sent && (lt!=-1 && now-lt< PERIOD*0.001)) return;
  cldData->lastCheck = now;

  myload = CldLoad();
  if (myload > 0) return;

  msg.from_pe = CmiMyPe();

  victim = (((CrnRand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());

  CmiSetHandler(&msg, CpvAccess(CldAskLoadHandlerIndex));
  /* fixme */
  //CmiBecomeImmediate(&msg);
  msg.to_rank = CmiRankOf(victim);
  CmiSyncSend(victim, sizeof(requestmsg),(char *)&msg);
  cldData->sent = 1;

#if CMK_TRACE_ENABLED && TRACE_USEREVENTS
  traceUserBracketEvent(cldData->idleEvt, now, CmiWallTimer());
#endif
}

/* immediate message handler, work at node level */
/* send some work to requested proc */
static void CldAskLoadHandler(requestmsg *msg)
{
  int receiver, rank, recvIdx, i;
  int myload = CldLoad();
  double now = CmiWallTimer();
  CldProcInfo  cldData = CpvAccess(CldData);

  /* only give you work if I have more than 1 */
  if (myload>0) {
    int sendLoad;
    receiver = msg->from_pe;
    rank = CmiMyRank();
    if (msg->to_rank != -1) rank = msg->to_rank;
    sendLoad = myload / 2; 
    CmiFree(msg);
    if (sendLoad < 1) return;
    CldMultipleSend(receiver, sendLoad, rank, 0);
  }
}

void CldHandler(void *msg)
{
  CldInfoFn ifn; CldPackFn pfn;
  int len, queueing, priobits; unsigned int *prioptr;
  
  CldRestoreHandler(msg);
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  /*CsdEnqueueGeneral(msg, CQS_QUEUEING_LIFO, priobits, prioptr); */
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
}

void CldBalanceHandler(void *msg)
{
  CldRestoreHandler(msg);
  CldPutToken(msg);
}

void CldEnqueueGroup(CmiGroup grp, void *msg, int infofn)
{
  int len, queueing, priobits,i; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  if (pfn) {
    pfn(&msg);
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  }
  CldSwitchHandler(msg, CpvAccess(CldHandlerIndex));
  CmiSetInfo(msg,infofn);

  CmiSyncMulticastAndFree(grp, len, msg);
}

void CldEnqueueMulti(int npes, int *pes, void *msg, int infofn)
{
  int len, queueing, priobits,i; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  if (pfn) {
    pfn(&msg);
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  }
  CldSwitchHandler(msg, CpvAccess(CldHandlerIndex));
  CmiSetInfo(msg,infofn);
  CmiSyncListSendAndFree(npes, pes, len, msg);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits, avg; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;

  if ((pe == CLD_ANYWHERE) && (CmiNumPes() > 1)) {
      pe = CmiMyPe();
    /* always pack the message because the message may be move away
       to a different processor later by CldGetToken() */
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    if (pfn && CmiNumNodes()>1) {
       pfn(&msg);
       ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CmiSetInfo(msg,infofn);
    CldPutToken(msg);
  } 
  else if ((pe == CmiMyPe()) || (CmiNumPes() == 1)) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    //CsdEnqueueGeneral(msg, CQS_QUEUEING_LIFO, priobits, prioptr);
    CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
  }
  else {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    if (pfn && CmiNodeOf(pe) != CmiMyNode()) {
      pfn(&msg);
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
    CldSwitchHandler(msg, CpvAccess(CldHandlerIndex));
    CmiSetInfo(msg,infofn);
    if (pe==CLD_BROADCAST) 
      CmiSyncBroadcastAndFree(len, msg);
    else if (pe==CLD_BROADCAST_ALL)
      CmiSyncBroadcastAllAndFree(len, msg);
    else CmiSyncSendAndFree(pe, len, msg);
  }
}

void CldNodeEnqueue(int node, void *msg, int infofn)
{
  int len, queueing, priobits, pe, avg; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;
  if ((node == CLD_ANYWHERE) && (CmiNumPes() > 1)) {
      pe = CmiMyPe();
      node = CmiNodeOf(pe);
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CsdNodeEnqueueGeneral(msg, queueing, priobits, prioptr);
  }
  else if ((node == CmiMyNode()) || (CmiNumPes() == 1)) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CsdNodeEnqueueGeneral(msg, queueing, priobits, prioptr);
  } 
  else {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    if (pfn) {
        pfn(&msg);
        ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
    CldSwitchHandler(msg, CpvAccess(CldHandlerIndex));
    CmiSetInfo(msg,infofn);
    if (node==CLD_BROADCAST) { CmiSyncNodeBroadcastAndFree(len, msg); }
    else if (node==CLD_BROADCAST_ALL){CmiSyncNodeBroadcastAllAndFree(len,msg);}
    else CmiSyncNodeSendAndFree(node, len, msg);
  }
}


void CldGraphModuleInit(char **argv)
{
  CpvInitialize(CldProcInfo, CldData);
  CpvInitialize(int, CldLoadResponseHandlerIndex);
  CpvInitialize(int, CldAskLoadHandlerIndex);
  CpvInitialize(int, CldBalanceHandlerIndex);

  CpvAccess(CldData) = (CldProcInfo)CmiAlloc(sizeof(struct CldProcInfo_s));
  CpvAccess(CldData)->lastCheck = -1;
  CpvAccess(CldData)->sent = 0;
#if CMK_TRACE_ENABLED
  CpvAccess(CldData)->balanceEvt = traceRegisterUserEvent("CldBalance", -1);
  CpvAccess(CldData)->idleEvt = traceRegisterUserEvent("CldBalanceIdle", -1);
  CpvAccess(CldData)->idleprocEvt = traceRegisterUserEvent("CldBalanceProcIdle", -1);
#endif

  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldAskLoadHandlerIndex) = 
    CmiRegisterHandler((CmiHandler)CldAskLoadHandler);

  /* communication thread */
  if (CmiMyRank() == CmiMyNodeSize())  return;

#if 1
  /* register idle handlers - when idle, keep asking work from neighbors */
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
      (CcdVoidFn) CldStillIdle, NULL);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,
      (CcdVoidFn) CldStillIdle, NULL);
    if (CmiMyPe() == 0) 
      CmiPrintf("Charm++> Work stealing is enabled. \n");
#endif
}


void CldModuleInit(char **argv)
{
  CpvInitialize(int, CldHandlerIndex);
  CpvInitialize(int, CldRelocatedMessages);
  CpvInitialize(int, CldLoadBalanceMessages);
  CpvInitialize(int, CldMessageChunks);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler(CldHandler);
  CpvAccess(CldRelocatedMessages) = CpvAccess(CldLoadBalanceMessages) = 
  CpvAccess(CldMessageChunks) = 0;

  CldModuleGeneralInit(argv);
  CldGraphModuleInit(argv);

  CpvAccess(CldLoadNotify) = 1;
}

void CldCallback()
{}
