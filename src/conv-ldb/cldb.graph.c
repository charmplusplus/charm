#include <stdio.h>
#include "converse.h"
#define PERIOD 30
#define PERIODE 100
#define THRESHOLD 20.0

typedef struct CldNeighborData
{
  int pe, load;
} *CldNeighborData;

CpvDeclare(CldNeighborData, neighbors);
CpvDeclare(int, numNeighbors);
CpvDeclare(int, CldHandlerIndex);
CpvDeclare(int, CldBalanceHandlerIndex);
CpvDeclare(int, CldLoadResponseHandlerIndex);
CpvDeclare(int, CldRequestResponseHandlerIndex);
CpvDeclare(int, CldRelocatedMessages);
CpvDeclare(int, CldLoadBalanceMessages);
CpvDeclare(int, CldMessageChunks);

CpvExtern(int, CldAvgLoad);

int CldAvgNeighborLoad()
{
  int sum=0, i;
  
  for (i=0; i<CpvAccess(numNeighbors); i++)
    sum += CpvAccess(neighbors)[i].load;
  sum += CldCountTokens();
  return (int)((float)sum / (float)(CpvAccess(numNeighbors)+1));
}

void CldRebalance();

void CldSendLoad()
{
  int len, myPe = CmiMyPe(), myLoad = CldCountTokens(), i;
  char *msg, *msgPos;
  
  len = CmiMsgHeaderSizeBytes + 2*sizeof(int);
  msg = (char *)CmiAlloc(len);
  msgPos = (msg + CmiMsgHeaderSizeBytes);
  memcpy((void *)msgPos, &myPe, sizeof(int));
  msgPos += sizeof(int);
  memcpy((void *)msgPos, &myLoad, sizeof(int));
  CmiSetHandler(msg, CpvAccess(CldLoadResponseHandlerIndex));
  for (i=0; i<CpvAccess(numNeighbors); i++)
    CmiSyncSend(CpvAccess(neighbors)[i].pe, len, msg);
  CmiFree(msg);
  CpvAccess(CldLoadBalanceMessages) += CpvAccess(numNeighbors);
}

void CldBalance()
{
  int i, j, overload, numToMove=0, numSent, *msgSizes;
  void **msgs;
  int len, queueing, priobits, totalUnderAvg=0, numUnderAvg=0, maxUnderAvg=0;
  unsigned int *prioptr;
  CldInfoFn ifn; CldPackFn pfn;

  CpvAccess(CldAvgLoad) = CldAvgNeighborLoad();
  overload = CldCountTokens() - CpvAccess(CldAvgLoad);
  
  if ((float)overload > (THRESHOLD/100.0 * (float)CpvAccess(CldAvgLoad))) {
    for (i=0; i<CpvAccess(numNeighbors); i++)
      if (CpvAccess(neighbors)[i].load < CpvAccess(CldAvgLoad)) {
	totalUnderAvg += CpvAccess(CldAvgLoad)-CpvAccess(neighbors)[i].load;
	if (CpvAccess(CldAvgLoad) - CpvAccess(neighbors)[i].load 
	    > maxUnderAvg)
	  maxUnderAvg = CpvAccess(CldAvgLoad) - CpvAccess(neighbors)[i].load;
	numUnderAvg++;
      }
    if (numUnderAvg > 0) {
      msgs = (void **)calloc(maxUnderAvg, sizeof(void *));
      msgSizes = (int *)calloc(maxUnderAvg, sizeof(int));
      for (i=0; i<CpvAccess(numNeighbors); i++) {
	if (CpvAccess(neighbors)[i].load < CpvAccess(CldAvgLoad)) {
	  numSent = 0;
	  numToMove = (int)
	    (((((float)CpvAccess(CldAvgLoad) - (float)CpvAccess(neighbors)[i].load)
	       / (float)totalUnderAvg)*(float)overload));
	  if (numToMove > 
	      CpvAccess(CldAvgLoad) - CpvAccess(neighbors)[i].load)
	    numToMove = CpvAccess(CldAvgLoad) - CpvAccess(neighbors)[i].load;
	  
	  for (j=0; j<numToMove; j++) {
	    CldGetToken(&msgs[j]);
	    if (msgs[j] != 0) {
	      numSent++;
	      ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msgs[j]));
	      ifn(msgs[j], &pfn, &len, &queueing, &priobits, &prioptr);
	      msgSizes[j] = len;
	      CldSwitchHandler(msgs[j], CpvAccess(CldBalanceHandlerIndex));
	    }
	    else
	      break;
	  }
	  if (numSent > 0) {
	    CmiMultipleSend(CpvAccess(neighbors)[i].pe, numSent, msgSizes, 
			    msgs);
	    CpvAccess(CldRelocatedMessages) += numSent;
	    CpvAccess(CldMessageChunks)++;
	  }
	}
      }
      free(msgs);
      free(msgSizes);
    }
  }
  CldSendLoad();
  CldRebalance();
}

void CldRebalance()
{
  CcdCallFnAfter((CcdVoidFn)CldBalance, NULL, PERIOD);
}

void CldLoadResponseHandler(char *msg)
{
  int load, pe, i=0;

  CmiGrabBuffer((void **)&msg);

  memcpy(&pe, msg + CmiMsgHeaderSizeBytes, sizeof(int));
  memcpy(&load, msg + CmiMsgHeaderSizeBytes + sizeof(int), sizeof(int));

  while (i<CpvAccess(numNeighbors)) {
    if (CpvAccess(neighbors)[i].pe == pe) {
      CpvAccess(neighbors)[i].load = load;
      break;
    }
    i++;
  }
}	

void CldRedo();
  
void CldRequestTokens()
{
  int len, myPe = CmiMyPe(), i;
  char *msg, *msgPos;
  
  len = CmiMsgHeaderSizeBytes + sizeof(int);
  msg = (char *)CmiAlloc(len);
  msgPos = (msg + CmiMsgHeaderSizeBytes);
  memcpy((void *)msgPos, &myPe, sizeof(int));
  CmiSetHandler(msg, CpvAccess(CldRequestResponseHandlerIndex));
  for (i=0; i<CpvAccess(numNeighbors); i++) 
    CmiSyncSend(CpvAccess(neighbors)[i].pe, len, msg);
  CpvAccess(CldLoadBalanceMessages) += CpvAccess(numNeighbors);
  CcdCallFnAfter((CcdVoidFn)CldRedo, NULL, PERIODE);
}
 
void CldRedo()
{
  if (CldCountTokens() == 0)
    CldRequestTokens();
}
 
void CldRequestResponseHandler(char *msg)
{
  int pe, numToSend, numSent, j, len, queueing, priobits, *msgSizes;
  void **msgs;
  unsigned int *prioptr;
  CldInfoFn ifn; CldPackFn pfn;

  CmiGrabBuffer((void **)&msg);
  memcpy(&pe, msg + CmiMsgHeaderSizeBytes, sizeof(int));

  numToSend = CldCountTokens() / (CpvAccess(numNeighbors) + 1);

  if (numToSend > 0) {
    msgs = (void **)calloc(numToSend, sizeof(void *));
    msgSizes = (int *)calloc(numToSend, sizeof(int));
    numSent = 0;
    for (j=0; j<numToSend; j++) {
      CldGetToken(&msgs[j]);
      if (msgs[j] != 0) {
	numSent++;
	ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msgs[j]));
	ifn(msgs[j], &pfn, &len, &queueing, &priobits, &prioptr);
	msgSizes[j] = len;
	CldSwitchHandler(msgs[j], CpvAccess(CldBalanceHandlerIndex));
      }
      else
	break;
    }
    if (numSent > 0) {
      CmiMultipleSend(pe, numSent, msgSizes, msgs);
      CpvAccess(CldRelocatedMessages) += numSent;
      CpvAccess(CldMessageChunks)++;
    }
    free(msgs);
    free(msgSizes);
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
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;

  if (CmiGetHandler(msg) >= CpvAccess(CmiHandlerMax)) *((int*)0)=0;

  if (pe == CLD_ANYWHERE)
    {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      CldPutToken(msg); 
    } 
  else if (pe == CmiMyPe())
    {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
    }
  else
    {
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

void CldNotify(int load)
{
  if (load == 2)
    CldRequestTokens(); 
}

void CldReadNeighborData()
{
  FILE *fp;
  char filename[25];
  int i;
  
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
  for (i=0; i<CpvAccess(numNeighbors); i++)
    {
      fscanf(fp, "%d", &(CpvAccess(neighbors)[i].pe));
      CpvAccess(neighbors)[i].load = 0;
    }
  fclose(fp);
}

void CldGraphModuleInit()
{
  CpvInitialize(int, numNeighbors);
  CpvInitialize(CldNeighborData, neighbors);
  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvInitialize(int, CldLoadResponseHandlerIndex);
  CpvInitialize(int, CldRequestResponseHandlerIndex);
  CpvInitialize(int, CldRelocatedMessages);
  CpvInitialize(int, CldLoadBalanceMessages);
  CpvInitialize(int, CldMessageChunks);
  CpvAccess(CldAvgLoad) = CldAvgNeighborLoad();

  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldLoadResponseHandlerIndex) = 
    CmiRegisterHandler(CldLoadResponseHandler);
  CpvAccess(CldRequestResponseHandlerIndex) = 
    CmiRegisterHandler(CldRequestResponseHandler);
  CpvAccess(CldRelocatedMessages) = CpvAccess(CldLoadBalanceMessages) = 
    CpvAccess(CldMessageChunks) = 0;
  CldReadNeighborData();

  CldBalance(); 
  CldRequestTokens();
}

void CldModuleInit()
{
  CpvInitialize(int, CldHandlerIndex);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler(CldHandler);
  CldModuleGeneralInit();
  CldGraphModuleInit();
}
