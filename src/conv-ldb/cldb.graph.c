#include "cldb.graph.h"
#define PERIOD 20                /* default: 30 */
#define MAXOVERLOAD 1

extern gengraph(int, int, int);

CpvDeclare(int, CldLoadResponseHandlerIndex);
CpvDeclare(int, MinLoad);
CpvDeclare(int, MinProc);
CpvDeclare(int, Mindex);

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "graph";
}

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
  CcdCallFnAfter((CcdVoidFn)CldBalance, NULL, PERIOD);
}

void CldLoadResponseHandler(loadmsg *msg)
{
  int i;

  for(i=0; i<CpvAccess(numNeighbors); i++)
    if (CpvAccess(neighbors)[i].pe == msg->pe) {
      CpvAccess(neighbors)[i].load = msg->load;
      break;
    }
}

void CldBalanceHandler(void *msg)
{
  CmiGrabBuffer((void **)&msg);
  CldRestoreHandler(msg);
  CldPutToken(msg);
}

void CldHandler(void *msg)
{
  CldInfoFn ifn; CldPackFn pfn;
  int len, queueing, priobits; unsigned int *prioptr;
  
  CmiGrabBuffer((void **)&msg);
  CldRestoreHandler(msg);
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  CsdEnqueueGeneral(msg, CQS_QUEUEING_LIFO, priobits, prioptr);
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
    if (pe != CmiMyPe()) {
      CpvAccess(neighbors)[CpvAccess(Mindex)].load++;
      CpvAccess(CldRelocatedMessages)++;
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      if (pfn) {
	pfn(&msg);
	ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      }
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

void CldGraphModuleInit()
{
  FILE *fp;
  char filename[20];
  
  CpvInitialize(int, numNeighbors);
  CpvInitialize(int, MinLoad);
  CpvInitialize(int, Mindex);
  CpvInitialize(int, MinProc);
  CpvInitialize(CmiGroup, neighborGroup);
  CpvInitialize(CldNeighborData, neighbors);
  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvInitialize(int, CldLoadResponseHandlerIndex);

  CpvAccess(MinLoad) = 0;
  CpvAccess(Mindex) = 0;
  CpvAccess(MinProc) = CmiMyPe();
  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldLoadResponseHandlerIndex) = 
    CmiRegisterHandler(CldLoadResponseHandler);

  if (CmiNumPes() > 1) {
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
    CldBalance();
  }
}

void CldModuleInit()
{
  CpvInitialize(int, CldHandlerIndex);
  CpvInitialize(int, CldRelocatedMessages);
  CpvInitialize(int, CldLoadBalanceMessages);
  CpvInitialize(int, CldMessageChunks);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler(CldHandler);
  CpvAccess(CldRelocatedMessages) = CpvAccess(CldLoadBalanceMessages) = 
    CpvAccess(CldMessageChunks) = 0;
  CldModuleGeneralInit();
  CldGraphModuleInit();
}
