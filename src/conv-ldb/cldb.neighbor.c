#include <stdlib.h>
#include "cldb.neighbor.h"
#define PERIOD 20                /* default: 30 */
#define MAXOVERLOAD 1

#include "converse.h"
#include "queueing.h"
#include "cldb.h"
#include "topology.h"

typedef struct CldProcInfo_s {
  double lastIdle;
  int    balanceEvt;		/* user event for balancing */
  int    idleEvt;		/* user event for idle balancing */
} *CldProcInfo;

extern char *_lbtopo;			/* topology name string */

void gengraph(int, int, int, int *, int *);

CpvStaticDeclare(CldProcInfo, CldData);
CpvDeclare(int, CldLoadResponseHandlerIndex);
CpvDeclare(int, CldAskLoadHandlerIndex);
CpvDeclare(int, MinLoad);
CpvDeclare(int, MinProc);
CpvDeclare(int, Mindex);

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "neighbor";
}

/* since I am idle, ask for work from neighbors */
static void CldBeginIdle(void *dummy)
{
  CpvAccess(CldData)->lastIdle = CmiWallTimer();
}

static void CldEndIdle(void *dummy)
{
  CpvAccess(CldData)->lastIdle = -1;
}

static void CldStillIdle(void *dummy)
{
  double startT;
  loadmsg msg;
  int myload;
  CldProcInfo  cldData = CpvAccess(CldData);

  double t = CmiWallTimer();
  double lt = cldData->lastIdle;
  /* only ask for work every 5ms */
  if (lt!=-1 && t-lt<0.005) {
    return;
  }
  cldData->lastIdle = t;

#ifndef CMK_OPTIMIZE
  startT = CmiWallTimer();
#endif

  myload = CldLoad();
/*  CmiAssert(myload == 0); */
  if (myload > 0) return;

  msg.pe = CmiMyPe();
  msg.load = myload;
  CmiSetHandler(&msg, CpvAccess(CldAskLoadHandlerIndex));
  CmiSyncMulticast(CpvAccess(neighborGroup), sizeof(loadmsg), &msg);

#ifndef CMK_OPTIMIZE
  /* traceUserBracketEvent(cldData->idleEvt, startT, CmiWallTimer()); */
#endif
}

static void CldAskLoadHandler(loadmsg *msg)
{
  /* send some work to this proc */
  int receiver = msg->pe;
  int myload = CldLoad();

  if (myload>1) {
    int sendLoad = myload / CpvAccess(numNeighbors) / 2;
    if (sendLoad < 1) sendLoad = 1;
    sendLoad = 1;
    CldMultipleSend(receiver, sendLoad);
  }
  CmiFree(msg);
}

/* balancing by exchanging load among neighbors */

void CldSendLoad()
{
  loadmsg msg;

  msg.pe = CmiMyPe();
  msg.load = CldLoad();
  CmiSetHandler(&msg, CpvAccess(CldLoadResponseHandlerIndex));
  CmiSyncMulticast(CpvAccess(neighborGroup), sizeof(loadmsg), &msg);
  CpvAccess(CldLoadBalanceMessages) += CpvAccess(numNeighbors);
}

int CldMinAvg()
{
  int sum=0, i;
  static int start=-1;

  if (start == -1)
    start = CmiMyPe() % CpvAccess(numNeighbors);
  CpvAccess(MinProc) = CpvAccess(neighbors)[start].pe;
  CpvAccess(MinLoad) = CpvAccess(neighbors)[start].load;
  sum = CpvAccess(neighbors)[start].load;
  CpvAccess(Mindex) = start;
  for (i=1; i<CpvAccess(numNeighbors); i++) {
    start = (start+1) % CpvAccess(numNeighbors);
    sum += CpvAccess(neighbors)[start].load;
    if (CpvAccess(MinLoad) > CpvAccess(neighbors)[start].load) {
      CpvAccess(MinLoad) = CpvAccess(neighbors)[start].load;
      CpvAccess(MinProc) = CpvAccess(neighbors)[start].pe;
      CpvAccess(Mindex) = start;
    }
  }
  start = (start+2) % CpvAccess(numNeighbors);
  sum += CldLoad();
  if (CldLoad() < CpvAccess(MinLoad)) {
    CpvAccess(MinLoad) = CldLoad();
    CpvAccess(MinProc) = CmiMyPe();
  }
  i = (int)(1.0 + (((float)sum) /((float)(CpvAccess(numNeighbors)+1))));
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
    for (i=0; i<CpvAccess(numNeighbors); i++)
      if (CpvAccess(neighbors)[i].load < avgLoad) {
        totalUnderAvg += avgLoad-CpvAccess(neighbors)[i].load;
        if (avgLoad - CpvAccess(neighbors)[i].load > maxUnderAvg)
          maxUnderAvg = avgLoad - CpvAccess(neighbors)[i].load;
        numUnderAvg++;
      }
    if (numUnderAvg > 0)
      for (i=0; ((i<CpvAccess(numNeighbors)) && (overload>0)); i++) {
	j = (i+CpvAccess(Mindex))%CpvAccess(numNeighbors);
        if (CpvAccess(neighbors)[j].load < avgLoad) {
          numToMove = avgLoad - CpvAccess(neighbors)[j].load;
          if (numToMove > overload)
            numToMove = overload;
          overload -= numToMove;
	  CpvAccess(neighbors)[j].load += numToMove;
          CldMultipleSend(CpvAccess(neighbors)[j].pe, numToMove);
        }
      }
  }
  CldSendLoad();
#ifndef CMK_OPTIMIZE
/*  traceUserBracketEvent(CpvAccess(CldData)->balanceEvt, startT, CmiWallTimer()); */
#endif
  CcdCallFnAfterOnPE((CcdVoidFn)CldBalance, NULL, PERIOD, CmiMyPe());
/*  CcdCallBacksReset(0); */
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
  topo = topofn();
  npe = getTopoMaxNeighbors(topo);
  pes = (int *)malloc(npe*sizeof(int));
  getTopoNeighbors(topo, CmiMyPe(), pes, &npe);
#if 0
  CmiPrintf("Neighors (%d) for: %d\n", npe, CmiMyPe());
  for (i=0; i<npe; i++) {
    CmiAssert(pes[i] < CmiNumPes());
    CmiPrintf(" %d ", pes[i]);
  }
  CmiPrintf("\n");
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
  CpvAccess(CldData)->lastIdle = -1;
#ifndef CMK_OPTIMIZE
  CpvAccess(CldData)->balanceEvt = traceRegisterUserEvent("CldBalance", -1);
  CpvAccess(CldData)->idleEvt = traceRegisterUserEvent("CldBalanceIdle", -1);
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
    CldBalance();
  }

#if 0
  /* register an idle handler */
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,
      (CcdVoidFn) CldStillIdle, NULL);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE,
      (CcdVoidFn) CldStillIdle, NULL);
#endif
#if 0
  /* periodic load balancing */
  CcdCallOnConditionKeep(CcdPERIODIC_10ms, (CcdVoidFn) CldBalance, NULL);
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
}
