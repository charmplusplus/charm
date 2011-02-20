#include <stdlib.h>

#include "converse.h"
#include "cldb.workstealing.h"
#include "queueing.h"
#include "cldb.h"

#define IDLE_IMMEDIATE 		0
#define TRACE_USEREVENTS        0

#define PERIOD 10                /* default: 30 */
#define MSGDELAY 10
#define MAXOVERLOAD 1

#define LOADTHRESH       3


typedef struct CldProcInfo_s {
  int    balanceEvt;		/* user event for balancing */
  int    idleEvt;		/* user event for idle balancing */
  int    idleprocEvt;		/* user event for processing idle req */
} *CldProcInfo;

int _stealonly1 = 0;

CpvStaticDeclare(CldProcInfo, CldData);
CpvStaticDeclare(int, CldAskLoadHandlerIndex);
CpvStaticDeclare(int, CldAckNoTaskHandlerIndex);

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "work stealing";
}

/* since I am idle, ask for work from neighbors */

static void CldBeginIdle(void *dummy)
{
  int i;
  double startT;
  requestmsg msg;
  int myload;
  int  victim;
  int mype;
  int numpes;

  CcdRaiseCondition(CcdUSER);

  myload = CldLoad();

  mype = CmiMyPe();
  msg.from_pe = mype;
  numpes = CmiNumPes();
  do{
      victim = (((CrnRand()+mype)&0x7FFFFFFF)%numpes);
  }while(victim == mype);

  CmiSetHandler(&msg, CpvAccess(CldAskLoadHandlerIndex));
#if IDLE_IMMEDIATE
  /* fixme */
  CmiBecomeImmediate(&msg);
#endif
  msg.to_rank = CmiRankOf(victim);
  CmiSyncSend(victim, sizeof(requestmsg),(char *)&msg);
  
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

  int sendLoad;
  sendLoad = myload / 2; 
  receiver = msg->from_pe;
  /* only give you work if I have more than 1 */
  if (myload>LOADTHRESH) {
      if(_stealonly1) sendLoad = 1;
      rank = CmiMyRank();
      if (msg->to_rank != -1) rank = msg->to_rank;
      CldMultipleSend(receiver, sendLoad, rank, 0);
  }else
  {
      requestmsg r_msg;
      r_msg.from_pe = CmiMyPe();
      r_msg.to_rank = CmiMyRank();

      CcdRaiseCondition(CcdUSER);

      CmiSetHandler(&r_msg, CpvAccess(CldAckNoTaskHandlerIndex));
      CmiSyncSend(receiver, sizeof(requestmsg),(char *)&r_msg);
    /* send ack indicating there is no task */
  }
  CmiFree(msg);
}

void  CldAckNoTaskHandler(requestmsg *msg)
{
  int victim; 
  int notaskpe = msg->from_pe;
  int mype = CmiMyPe();

  CcdRaiseCondition(CcdUSER);

  do{
      victim = (((CrnRand()+notaskpe)&0x7FFFFFFF)%CmiNumPes());
  }while(victim == mype);

  /* reuse msg */
  msg->to_rank = CmiRankOf(victim);
  msg->from_pe = mype;
  CmiSetHandler(msg, CpvAccess(CldAskLoadHandlerIndex));
  CmiSyncSendAndFree(victim, sizeof(requestmsg),(char *)msg);

}

void CldHandler(void *msg)
{
  CldInfoFn ifn; CldPackFn pfn;
  int len, queueing, priobits; unsigned int *prioptr;
  
  CldRestoreHandler(msg);
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
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
  CpvInitialize(int, CldAskLoadHandlerIndex);
  CpvInitialize(int, CldAckNoTaskHandlerIndex);
  CpvInitialize(int, CldBalanceHandlerIndex);

  CpvAccess(CldData) = (CldProcInfo)CmiAlloc(sizeof(struct CldProcInfo_s));
#if CMK_TRACE_ENABLED
  CpvAccess(CldData)->balanceEvt = traceRegisterUserEvent("CldBalance", -1);
  CpvAccess(CldData)->idleEvt = traceRegisterUserEvent("CldBalanceIdle", -1);
  CpvAccess(CldData)->idleprocEvt = traceRegisterUserEvent("CldBalanceProcIdle", -1);
#endif

  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldAskLoadHandlerIndex) = 
    CmiRegisterHandler((CmiHandler)CldAskLoadHandler);
  
  CpvAccess(CldAckNoTaskHandlerIndex) = 
    CmiRegisterHandler((CmiHandler)CldAckNoTaskHandler);

  /* communication thread */
  if (CmiMyRank() == CmiMyNodeSize())  return;

  _stealonly1 = CmiGetArgFlagDesc(argv, "+stealonly1", "Charm++> Work Stealing, every time only steal 1 task");

  /* register idle handlers - when idle, keep asking work from neighbors */
  if(CmiNumPes() > 1)
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
      (CcdVoidFn) CldBeginIdle, NULL);
  if (CmiMyPe() == 0) 
      CmiPrintf("Charm++> Work stealing is enabled. \n");
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
