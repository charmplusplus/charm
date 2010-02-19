#include <stdlib.h>

#include "converse.h"
#include "cldb.neighbor.h"
#include "queueing.h"
#include "cldb.h"
#include "topology.h"

#define IDLE_IMMEDIATE 		1
#define TRACE_USEREVENTS        0

#define PERIOD 20                /* default: 30 */
#define MAXOVERLOAD 1

typedef struct CldProcInfo_s {
  double lastCheck;
  int    sent;			/* flag to disable idle work request */
  int    balanceEvt;		/* user event for balancing */
  int    idleEvt;		/* user event for idle balancing */
  int    idleprocEvt;		/* user event for processing idle req */
} *CldProcInfo;

extern char *_lbtopo;			/* topology name string */
int _lbsteal = 0;                       /* work stealing flag */

void gengraph(int, int, int, int *, int *);

CpvStaticDeclare(CldProcInfo, CldData);
CpvStaticDeclare(int, CldLoadResponseHandlerIndex);
CpvStaticDeclare(int, CldAskLoadHandlerIndex);
CpvStaticDeclare(int, MinLoad);
CpvStaticDeclare(int, MinProc);
CpvStaticDeclare(int, Mindex);

void LoadNotifyFn(int l)
{
  CldProcInfo  cldData = CpvAccess(CldData);
  cldData->sent = 0;
}

char *CldGetStrategy(void)
{
  return "neighbor";
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

static void CldStillIdle(void *dummy)
{
  int i;
  double startT;
  requestmsg msg;
  int myload;
  CldProcInfo  cldData = CpvAccess(CldData);

  double now = CmiWallTimer();
  double lt = cldData->lastCheck;
  /* only ask for work every 20ms */
  if (cldData->sent && (lt!=-1 && now-lt< PERIOD*0.001)) return;
  cldData->lastCheck = now;

  myload = CldLoad();
  CmiAssert(myload == 0);
  if (myload > 0) return;

  msg.from_pe = CmiMyPe();
  CmiSetHandler(&msg, CpvAccess(CldAskLoadHandlerIndex));
#if ! IDLE_IMMEDIATE
  msg.to_rank = -1;
  CmiSyncMulticast(CpvAccess(neighborGroup), sizeof(requestmsg), &msg);
#else
  /* fixme */
  CmiBecomeImmediate(&msg);
  for (i=0; i<CpvAccess(numNeighbors); i++) {
    msg.to_rank = CmiRankOf(CpvAccess(neighbors)[i].pe);
    CmiSyncNodeSend(CmiNodeOf(CpvAccess(neighbors)[i].pe),sizeof(requestmsg),(char *)&msg);
  }
#endif
  cldData->sent = 1;

#if !defined(CMK_OPTIMIZE) && TRACE_USEREVENTS
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

  /* only give you work if I have more than 1 */
  if (myload>0) {
    int sendLoad;
    receiver = msg->from_pe;
    rank = CmiMyRank();
    if (msg->to_rank != -1) rank = msg->to_rank;
#if IDLE_IMMEDIATE
    /* try the lock */
    if (CmiTryLock(CpvAccessOther(cldLock, rank))) {
      CmiDelayImmediate();		/* postpone immediate message */
      return;
    }
    CmiUnlock(CpvAccessOther(cldLock, rank));  /* release lock, grab later */
#endif
    sendLoad = myload / CpvAccess(numNeighbors) / 2;
    if (sendLoad < 1) sendLoad = 1;
    sendLoad = 1;
    for (i=0; i<CpvAccess(numNeighbors); i++) 
      if (CpvAccess(neighbors)[i].pe == receiver) break;
    CmiAssert(i<CpvAccess(numNeighbors));
    CpvAccess(neighbors)[i].load += sendLoad;
    CldMultipleSend(receiver, sendLoad, rank, 0);
#if 0
#if !defined(CMK_OPTIMIZE) && TRACE_USEREVENTS
    /* this is dangerous since projections logging is not thread safe */
    {
    CldProcInfo  cldData = CpvAccessOther(CldData, rank);
    traceUserBracketEvent(cldData->idleprocEvt, now, CmiWallTimer());
    }
#endif
#endif
  }
  CmiFree(msg);
}

/* balancing by exchanging load among neighbors */

void CldSendLoad()
{
#if CMK_MULTICORE
  /* directly send load to neighbors */
  double myload = CldLoad();
  int nNeighbors = CpvAccess(numNeighbors);
  int i;
  for (i=0; i<nNeighbors; i++) {
    int neighbor_pe = CpvAccess(neighbors)[i].pe;
    int j, found=0;
    for (j=0; j<CpvAccessOther(numNeighbors, neighbor_pe); j++)
      if (CpvAccessOther(neighbors, neighbor_pe)[j].pe == CmiMyPe())
      {
        CpvAccessOther(neighbors, neighbor_pe)[j].load = myload;
        found = 1;
        break;
      }
    CmiAssert(found == 1);
  }
#else
  loadmsg msg;

  msg.pe = CmiMyPe();
  msg.load = CldLoad();
  CmiSetHandler(&msg, CpvAccess(CldLoadResponseHandlerIndex));
  CmiSyncMulticast(CpvAccess(neighborGroup), sizeof(loadmsg), &msg);
  CpvAccess(CldLoadBalanceMessages) += CpvAccess(numNeighbors);
#endif
}

int CldMinAvg()
{
  int sum=0, i;
  static int start=-1;

  int nNeighbors = CpvAccess(numNeighbors);
  if (start == -1)
    start = CmiMyPe() % nNeighbors;

#if 0
    /* update load from neighbors for multicore */
  for (i=0; i<nNeighbors; i++) {
    CpvAccess(neighbors)[i].load = CldLoadRank(CpvAccess(neighbors)[i].pe);
  }
#endif
  CpvAccess(MinProc) = CpvAccess(neighbors)[start].pe;
  CpvAccess(MinLoad) = CpvAccess(neighbors)[start].load;
  sum = CpvAccess(neighbors)[start].load;
  CpvAccess(Mindex) = start;
  for (i=1; i<nNeighbors; i++) {
    start = (start+1) % nNeighbors;
    sum += CpvAccess(neighbors)[start].load;
    if (CpvAccess(MinLoad) > CpvAccess(neighbors)[start].load) {
      CpvAccess(MinLoad) = CpvAccess(neighbors)[start].load;
      CpvAccess(MinProc) = CpvAccess(neighbors)[start].pe;
      CpvAccess(Mindex) = start;
    }
  }
  start = (start+2) % nNeighbors;
  sum += CldLoad();
  if (CldLoad() < CpvAccess(MinLoad)) {
    CpvAccess(MinLoad) = CldLoad();
    CpvAccess(MinProc) = CmiMyPe();
  }
  i = (int)(1.0 + (((float)sum) /((float)(nNeighbors+1))));
  return i;
}

void CldBalance()
{
  int i, j, overload, numToMove=0, avgLoad;
  int totalUnderAvg=0, numUnderAvg=0, maxUnderAvg=0;

#ifndef CMK_OPTIMIZE
  double startT = CmiWallTimer();
#endif

/*CmiPrintf("[%d] CldBalance %f\n", CmiMyPe(), startT);*/
  avgLoad = CldMinAvg();
  overload = CldLoad() - avgLoad;
  if (overload > CldCountTokens())
    overload = CldCountTokens();

  if (overload > MAXOVERLOAD) {
    int nNeighbors = CpvAccess(numNeighbors);
    for (i=0; i<nNeighbors; i++)
      if (CpvAccess(neighbors)[i].load < avgLoad) {
        totalUnderAvg += avgLoad-CpvAccess(neighbors)[i].load;
        if (avgLoad - CpvAccess(neighbors)[i].load > maxUnderAvg)
          maxUnderAvg = avgLoad - CpvAccess(neighbors)[i].load;
        numUnderAvg++;
      }
    if (numUnderAvg > 0)
      for (i=0; ((i<nNeighbors) && (overload>0)); i++) {
	j = (i+CpvAccess(Mindex))%CpvAccess(numNeighbors);
        if (CpvAccess(neighbors)[j].load < avgLoad) {
          numToMove = (avgLoad - CpvAccess(neighbors)[j].load)/numUnderAvg;
          if (numToMove > overload)
            numToMove = overload;
          overload -= numToMove;
	  CpvAccess(neighbors)[j].load += numToMove;
#if CMK_MULTICORE
          CldSimpleMultipleSend(CpvAccess(neighbors)[j].pe, numToMove);
#else
          CldMultipleSend(CpvAccess(neighbors)[j].pe, 
			  numToMove, CmiMyRank(), 
#if CMK_SMP
			  0
#else
			  1
#endif
                          );
#endif
        }
      }
  }
  CldSendLoad();
#if !defined(CMK_OPTIMIZE) && TRACE_USEREVENTS
  traceUserBracketEvent(CpvAccess(CldData)->balanceEvt, startT, CmiWallTimer());
#endif
  CcdCallFnAfterOnPE((CcdVoidFn)CldBalance, NULL, PERIOD, CmiMyPe());

}

void CldLoadResponseHandler(loadmsg *msg)
{
  int i;

  for(i=0; i<CpvAccess(numNeighbors); i++)
    if (CpvAccess(neighbors)[i].pe == msg->pe) {
      CpvAccess(neighbors)[i].load = msg->load;
      break;
    }
  CmiFree(msg);
}

void CldBalanceHandler(void *msg)
{
  CldRestoreHandler(msg);
  CldPutToken(msg);
}

void CldHandler(void *msg)
{
  CldInfoFn ifn; CldPackFn pfn;
  int len, queueing, priobits; unsigned int *prioptr;
  
  CldRestoreHandler(msg);
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  CsdEnqueueGeneral(msg, CQS_QUEUEING_LIFO, priobits, prioptr);
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
  /*
  for(i=0;i<npes;i++) {
    CmiSyncSend(pes[i], len, msg);
  }
  CmiFree(msg);
  */
  CmiSyncListSendAndFree(npes, pes, len, msg);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits, avg; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;

  if ((pe == CLD_ANYWHERE) && (CmiNumPes() > 1)) {
    avg = CldMinAvg();
    if (CldLoad() < avg)
      pe = CmiMyPe();
    else
      pe = CpvAccess(MinProc);
    /* always pack the message because the message may be move away
       to a different processor later by CldGetToken() */
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    if (pfn) {
       pfn(&msg);
       ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
    if (pe != CmiMyPe()) {
      CpvAccess(neighbors)[CpvAccess(Mindex)].load++;
      CpvAccess(CldRelocatedMessages)++;
      CmiSetInfo(msg,infofn);
      CldSwitchHandler(msg, CpvAccess(CldBalanceHandlerIndex));
      CmiSyncSendAndFree(pe, len, msg);
    }
    else {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      CldPutToken(msg);
    }
  } 
  else if ((pe == CmiMyPe()) || (CmiNumPes() == 1)) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CmiSetInfo(msg,infofn);
    CsdEnqueueGeneral(msg, CQS_QUEUEING_LIFO, priobits, prioptr);
  }
  else {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    if (pfn) {
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
    avg = CldMinAvg();
    if (CldLoad() < avg)
      pe = CmiMyPe();
    else
      pe = CpvAccess(MinProc);
    node = CmiNodeOf(pe);
    if (node != CmiMyNode()){
      CpvAccess(neighbors)[CpvAccess(Mindex)].load++;
      CpvAccess(CldRelocatedMessages)++;
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      if (pfn) {
	pfn(&msg);
	ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      }
      CmiSetInfo(msg,infofn);
      CldSwitchHandler(msg, CpvAccess(CldBalanceHandlerIndex));
      CmiSyncNodeSendAndFree(node, len, msg);
    }
    else {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      CldPutToken(msg);
    }
  }
  else if ((node == CmiMyNode()) || (CmiNumPes() == 1)) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CmiSetInfo(msg,infofn);
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

void CldReadNeighborData()
{
  FILE *fp;
  char filename[25];
  int i, *pes;
  
  if (CmiNumPes() <= 1)
    return;
  sprintf(filename, "graph%d/graph%d", CmiNumPes(), CmiMyPe());
  if ((fp = fopen(filename, "r")) == 0) 
    {
      CmiError("Error opening graph init file on PE: %d\n", CmiMyPe());
      return;
    }
  fscanf(fp, "%d", &CpvAccess(numNeighbors));
  CpvAccess(neighbors) = 
    (struct CldNeighborData *)calloc(CpvAccess(numNeighbors), 
				     sizeof(struct CldNeighborData));
  pes = (int *)calloc(CpvAccess(numNeighbors), sizeof(int));
  for (i=0; i<CpvAccess(numNeighbors); i++) {
    fscanf(fp, "%d", &(CpvAccess(neighbors)[i].pe));
    pes[i] = CpvAccess(neighbors)[i].pe;
    CpvAccess(neighbors)[i].load = 0;
  }
  fclose(fp);
  CpvAccess(neighborGroup) = CmiEstablishGroup(CpvAccess(numNeighbors), pes);
}

static void CldComputeNeighborData()
{
  int i, npe;
  int *pes;
  LBtopoFn topofn;
  void *topo;

  topofn = LBTopoLookup(_lbtopo);
  if (topofn == NULL) {
    char str[1024];
    CmiPrintf("SeedLB> Fatal error: Unknown topology: %s. Choose from:\n", _lbtopo);
    printoutTopo();
    sprintf(str, "SeedLB> Fatal error: Unknown topology: %s", _lbtopo);
    CmiAbort(str);
  }
  topo = topofn(CmiNumPes());
  npe = getTopoMaxNeighbors(topo);
  pes = (int *)malloc(npe*sizeof(int));
  getTopoNeighbors(topo, CmiMyPe(), pes, &npe);
#if 0
  {
  char buf[512], *ptr;
  sprintf(buf, "Neighors for PE %d (%d): ", CmiMyPe(), npe);
  ptr = buf + strlen(buf);
  for (i=0; i<npe; i++) {
    CmiAssert(pes[i] < CmiNumPes() && pes[i] != CmiMyPe());
    sprintf(ptr, " %d ", pes[i]);
    ptr += strlen(ptr);
  }
  strcat(ptr, "\n");
  CmiPrintf(buf);
  }
#endif

  CpvAccess(numNeighbors) = npe;
  CpvAccess(neighbors) = 
    (struct CldNeighborData *)calloc(CpvAccess(numNeighbors), 
				     sizeof(struct CldNeighborData));
  for (i=0; i<CpvAccess(numNeighbors); i++) {
    CpvAccess(neighbors)[i].pe = pes[i];
    CpvAccess(neighbors)[i].load = 0;
  }
  CpvAccess(neighborGroup) = CmiEstablishGroup(CpvAccess(numNeighbors), pes);
  free(pes);
}

void CldGraphModuleInit(char **argv)
{
  CpvInitialize(CldProcInfo, CldData);
  CpvInitialize(int, numNeighbors);
  CpvInitialize(int, MinLoad);
  CpvInitialize(int, Mindex);
  CpvInitialize(int, MinProc);
  CpvInitialize(CmiGroup, neighborGroup);
  CpvInitialize(CldNeighborData, neighbors);
  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvInitialize(int, CldLoadResponseHandlerIndex);
  CpvInitialize(int, CldAskLoadHandlerIndex);

  CpvAccess(CldData) = (CldProcInfo)CmiAlloc(sizeof(struct CldProcInfo_s));
  CpvAccess(CldData)->lastCheck = -1;
  CpvAccess(CldData)->sent = 0;
#ifndef CMK_OPTIMIZE
  CpvAccess(CldData)->balanceEvt = traceRegisterUserEvent("CldBalance", -1);
  CpvAccess(CldData)->idleEvt = traceRegisterUserEvent("CldBalanceIdle", -1);
  CpvAccess(CldData)->idleprocEvt = traceRegisterUserEvent("CldBalanceProcIdle", -1);
#endif

  CpvAccess(MinLoad) = 0;
  CpvAccess(Mindex) = 0;
  CpvAccess(MinProc) = CmiMyPe();
  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldLoadResponseHandlerIndex) = 
    CmiRegisterHandler((CmiHandler)CldLoadResponseHandler);
  CpvAccess(CldAskLoadHandlerIndex) = 
    CmiRegisterHandler((CmiHandler)CldAskLoadHandler);

  /* communication thread */
  if (CmiMyRank() == CmiMyNodeSize())  return;

  CmiGetArgStringDesc(argv, "+LBTopo", &_lbtopo, "define load balancing topology");
  if (CmiMyPe() == 0) CmiPrintf("Seed LB> Topology %s\n", _lbtopo);

  if (CmiNumPes() > 1) {
#if 0
    FILE *fp;
    char filename[20];
  
    sprintf(filename, "graph%d/graph%d", CmiNumPes(), CmiMyPe());
    if ((fp = fopen(filename, "r")) == 0)
      {
	if (CmiMyPe() == 0) {
	  CmiPrintf("No proper graph%d directory exists in current directory.\n Generating...  ", CmiNumPes());
	  gengraph(CmiNumPes(), (int)(sqrt(CmiNumPes())+0.5), 234);
	  CmiPrintf("done.\n");
	}
	else {
	  while (!(fp = fopen(filename, "r"))) ;
	  fclose(fp);
	}
      }
    else fclose(fp);
    CldReadNeighborData();
#endif
    CldComputeNeighborData();
#if CMK_MULTICORE
    CmiNodeBarrier();
#endif
    CldBalance();
  }

#if 1
  _lbsteal = CmiGetArgFlagDesc(argv, "+workstealing", "Charm++> Enable work stealing at idle time");
  if (_lbsteal) {
  /* register idle handlers - when idle, keep asking work from neighbors */
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
      (CcdVoidFn) CldStillIdle, NULL);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,
      (CcdVoidFn) CldStillIdle, NULL);
    if (CmiMyPe() == 0) 
      CmiPrintf("Charm++> Work stealing is enabled. \n");
  }
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
