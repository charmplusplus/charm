#include "cldb.graph.h"
#define PERIOD 30
#define PERIODE 100
#define THRESHOLD 20.0

CpvDeclare(int, CldLoadResponseHandlerIndex);
CpvDeclare(int, CldRequestResponseHandlerIndex);

int CldAvgNeighborLoad()
{
  int sum=CldCountTokens(), i;
  
  for (i=0; i<CpvAccess(numNeighbors); i++)
    sum += CpvAccess(neighbors)[i].load;
  return (int)((float)sum / (float)(CpvAccess(numNeighbors)+1));
}

void CldSendLoad()
{
  loadmsg msg;
  int i;

  msg.pe = CmiMyPe();
  msg.load = CldCountTokens();
  CmiSetHandler(&msg, CpvAccess(CldLoadResponseHandlerIndex));
/*  for (i=0; i<CpvAccess(numNeighbors); i++)
    CmiSyncSend(CpvAccess(neighbors)[i].pe, sizeof(loadmsg), &msg); */
  CmiSyncMulticast(CpvAccess(neighborGroup), sizeof(loadmsg), &msg);
  CpvAccess(CldLoadBalanceMessages) += CpvAccess(numNeighbors);
}

void CldMultipleSend(int pe, int numToSend)
{
  void **msgs;
  int len, queueing, priobits, *msgSizes, i, numSent, done=0, parcelSize;
  unsigned int *prioptr;
  CldInfoFn ifn;
  CldPackFn pfn;

  if (numToSend == 0)
    return;

  msgs = (void **)calloc(numToSend, sizeof(void *));
  msgSizes = (int *)calloc(numToSend, sizeof(int));

  while (!done) {
    numSent = 0;
    parcelSize = 0;
    for (i=0; i<numToSend; i++) {
      CldGetToken(&msgs[i]);
      if (msgs[i] != 0) {
	done = 1;
	numSent++;
	ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msgs[i]));
	ifn(msgs[i], &pfn, &len, &queueing, &priobits, &prioptr);
	msgSizes[i] = len;
	parcelSize += len;
	CldSwitchHandler(msgs[i], CpvAccess(CldBalanceHandlerIndex));
      }
      else {
	done = 1;
	break;
      }
      if (parcelSize > MAXMSGBFRSIZE) {
	if (i<numToSend-1)
	  done = 0;
	numToSend -= numSent;
	break;
      }
    }
    if (numSent > 1) {
      CmiMultipleSend(pe, numSent, msgSizes, msgs);
      for (i=0; i<numSent; i++)
	CmiFree(msgs[i]);
      CpvAccess(CldRelocatedMessages) += numSent;
      CpvAccess(CldMessageChunks)++;
    }
    else if (numSent == 1) {
      CmiSyncSend(pe, msgSizes[0], msgs[0]);
      CmiFree(msgs[0]);
      CpvAccess(CldRelocatedMessages)++;
      CpvAccess(CldMessageChunks)++;
    }
  }
  free(msgs);
  free(msgSizes);
}

void CldBalance()
{
  int i, overload, numToMove=0, avgLoad;
  int totalUnderAvg=0, numUnderAvg=0, maxUnderAvg=0;

  avgLoad = CldAvgNeighborLoad();
  overload = CldCountTokens() - avgLoad;
  
  if ((float)overload > (THRESHOLD/100.0 * (float)avgLoad)) {
    for (i=0; i<CpvAccess(numNeighbors); i++)
      if (CpvAccess(neighbors)[i].load < avgLoad) {
	totalUnderAvg += avgLoad-CpvAccess(neighbors)[i].load;
	if (avgLoad - CpvAccess(neighbors)[i].load > maxUnderAvg)
	  maxUnderAvg = avgLoad - CpvAccess(neighbors)[i].load;
	numUnderAvg++;
      }
    if (numUnderAvg > 0) 
      for (i=0; i<CpvAccess(numNeighbors); i++) 
	if (CpvAccess(neighbors)[i].load < avgLoad) {
	  numToMove = (int)
	    (((((float)avgLoad - (float)CpvAccess(neighbors)[i].load)
	       / (float)totalUnderAvg)*(float)overload));
	  if (numToMove > avgLoad - CpvAccess(neighbors)[i].load)
	    numToMove = avgLoad - CpvAccess(neighbors)[i].load;
	  CldMultipleSend(CpvAccess(neighbors)[i].pe, numToMove);
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

void CldRedo();

void CldRequestTokens()
{
  requestmsg msg;
  int i;

  msg.pe = CmiMyPe();
  CmiSetHandler(&msg, CpvAccess(CldRequestResponseHandlerIndex));
/*  for (i=0; i<CpvAccess(numNeighbors); i++) 
    CmiSyncSend(CpvAccess(neighbors)[i].pe, sizeof(requestmsg), &msg); */
  CmiSyncMulticast(CpvAccess(neighborGroup), sizeof(requestmsg), &msg);
  CpvAccess(CldLoadBalanceMessages) += CpvAccess(numNeighbors);
  CcdCallFnAfter((CcdVoidFn)CldRedo, NULL, PERIODE);
}
 
void CldRedo()
{
  if (CldCountTokens() == 0)
    CldRequestTokens();
}
 
void CldRequestResponseHandler(requestmsg *msg)
{
  int pe, numToSend;

  pe = msg->pe;
  numToSend = CldCountTokens() / (CpvAccess(numNeighbors) + 1);
  CldMultipleSend(pe, numToSend);
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
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;
  static int first = 1;

  if ((pe == CLD_ANYWHERE) && (CmiNumPes() > 1)) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CmiSetInfo(msg,infofn);
    CldPutToken(msg); 
  } 
  else if ((pe == CmiMyPe()) || (CmiNumPes() == 1)) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CmiSetInfo(msg,infofn);
    CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
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

  if (first == 1) {
    first = 0;
    if (CmiNumPes() > 1) {
      CldBalance(); 
      CldRequestTokens();
    }
  }
}

void CldNotify(int load)
{
  if (load == 2)
    CldRequestTokens(); 
}

void CldReadNeighborData()
{
  FILE *fp;
  char filename[25];
  int i, *pes;
  
  if (CmiNumPes() <= 1)
    return;
  sprintf(filename, "graph%d/graph%d", CmiNumPes(), CmiMyPe());
  if ((fp = fopen(filename, "r")) == 0) {
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
  CpvInitialize(int, numNeighbors);
  CpvInitialize(CmiGroup, neighborGroup);
  CpvInitialize(CldNeighborData, neighbors);
  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvInitialize(int, CldLoadResponseHandlerIndex);
  CpvInitialize(int, CldRequestResponseHandlerIndex);

  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldLoadResponseHandlerIndex) = 
    CmiRegisterHandler(CldLoadResponseHandler);
  CpvAccess(CldRequestResponseHandlerIndex) = 
    CmiRegisterHandler(CldRequestResponseHandler);
  CldReadNeighborData();

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
