#include <stdio.h>
#include "converse.h"
#define PERIOD 10

typedef struct CldNeighborData
{
  int pe, load;
} *CldNeighborData;

CpvDeclare(CldNeighborData, neighbors);
CpvDeclare(int, numNeighbors);
CpvDeclare(int, CldHandlerIndex);
CpvDeclare(int, CldBalanceHandlerIndex);
CpvDeclare(int, CldLoadResponseHandlerIndex);
CpvDeclare(int, CldRelocatedMessages);
CpvDeclare(int, CldLoadBalanceMessages);
CpvDeclare(int, CldMessageChunks);

void CldReadNeighborData()
{
  FILE *fp;
  char filename[25];
  int i;
  
  sprintf(filename, "graph/graph%d", CmiMyPe());
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

int CldAvgNeighborLoad()
{
  int sum=0, i;
  
  for (i=0; i<CpvAccess(numNeighbors); i++)
    sum += CpvAccess(neighbors)[i].load;
  sum += CldCountTokens();
  return (int)((float)sum / (float)(CpvAccess(numNeighbors)+1));
}

void CldBalance();

void CldPeriodicBalanceInit()
{
  int len, myPe, myLoad;
  int i;

  static int count=0; 
  
  char *msg, *msgPos;
  
  myPe = CmiMyPe();
  myLoad = CldCountTokens();
  len = CmiMsgHeaderSizeBytes + 2*sizeof(int);

  msg = (char *)CmiAlloc(len);
  msgPos = (msg + CmiMsgHeaderSizeBytes);
  memcpy((void *)msgPos, &myPe, sizeof(int));
  msgPos += sizeof(int);
  memcpy((void *)msgPos, &myLoad, sizeof(int));
  CmiSetHandler(msg, CpvAccess(CldLoadResponseHandlerIndex));
  for (i=0; i<CpvAccess(numNeighbors); i++)
    CmiSyncSend(CpvAccess(neighbors)[i].pe, len, msg);
  CpvAccess(CldLoadBalanceMessages) += CpvAccess(numNeighbors);
  CcdCallFnAfter((CcdVoidFn)CldBalance, NULL, PERIOD/2);
}

void CldBalance()
{
  int avgload, i, j, overload, numToMove, numSent, *msgSizes, 
    numnbrs = CpvAccess(numNeighbors);
  void **msgs;

  int len, queueing, priobits, totalUnderAvg=0, numUnderAvg=0;
  unsigned int *prioptr;
  CldInfoFn ifn;

  avgload = CldAvgNeighborLoad();
  overload = CldCountTokens() - avgload;

  if (overload <= 0)
    {
      CcdCallFnAfter((CcdVoidFn)CldPeriodicBalanceInit, NULL, PERIOD/2);
      return;
    }

  for (i=0; i<numnbrs; i++)
    if (CpvAccess(neighbors)[i].load <= avgload)
      numUnderAvg++;

  numToMove = overload / numUnderAvg;

  /* Change these two lines */
  msgs = (void **)calloc(numToMove, sizeof(void *));
  msgSizes = (int *)calloc(numToMove, sizeof(int));

  for (i=0; i<numnbrs; i++)
    {
      if (CpvAccess(neighbors)[i].load <= avgload)
	{
	  numSent = 0;
          for (j=0; j<numToMove; j++)
            {
              CldGetToken(&msgs[j]);
              if (msgs[j] != 0)
                {
                  numSent++;
                  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msgs[j]));
                  ifn(msgs[j], &len, &queueing, &priobits, &prioptr);
                  msgSizes[j] = len;
                  CldSwitchHandler(msgs[j], CpvAccess(CldBalanceHandlerIndex));
                }
              else
                break;
            }
          CmiMultipleSend(CpvAccess(neighbors)[i].pe, numSent, msgSizes, msgs);
	  CpvAccess(CldRelocatedMessages) += numSent;
	  CpvAccess(CldMessageChunks)++;
        }
    }
  /*
  for (i=0; i<numnbrs; i++)
    if ((j=avgload - CpvAccess(neighbors)[i].load) > 0)
      totalUnderAvg += j;
  i = numSent = 0;
  while ((numSent < overload) && (i<numnbrs))
    {
      if ((numToMove = avgload - CpvAccess(neighbors)[i].load) > 0)
	{
	  numToMove = (int)(((float)numToMove/(float)totalUnderAvg) 
			    * (float)overload);
	  if (numSent + numToMove <= overload)
	    numSent += numToMove;
	  else
	    {
	      numToMove = overload - numSent;
	      numSent = overload;
	    }
	
	  for (j=0; j<numToMove; j++)
	    {
	      CldGetToken(&msgs[j]);
	      ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msgs[j]));
	      ifn(msgs[j], &len, &queueing, &priobits, &prioptr);
	      msgSizes[j] = len;
	      CldSwitchHandler(msgs[j], CpvAccess(CldBalanceHandlerIndex));
	    }
	  CmiMultipleSend(CpvAccess(neighbors)[i].pe, numToMove, msgSizes, msgs);
	}
      i++;
    }
    */
  CcdCallFnAfter((CcdVoidFn)CldPeriodicBalanceInit, NULL, PERIOD/2);
}

void CldLoadResponseHandler(char *msg)
{
  int load, pe, i=0;

  CmiGrabBuffer((void **)&msg);

  memcpy(&pe, msg + CmiMsgHeaderSizeBytes, sizeof(int));
  memcpy(&load, msg + CmiMsgHeaderSizeBytes + sizeof(int), sizeof(int));

  while (i<CpvAccess(numNeighbors))
    {
      if (CpvAccess(neighbors)[i].pe == pe)
        {
          CpvAccess(neighbors)[i].load = load;
          break;
        }
      i++;
    }
}	
  
void CldBalanceHandler(void *msg)
{
  CmiGrabBuffer((void **)&msg);
  CldRestoreHandler(msg);
  CldPutToken(msg);
}


void CldGraphModuleInit()
{
  CpvInitialize(int, numNeighbors);
  CpvInitialize(CldNeighborData, neighbors);
  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvInitialize(int, CldLoadResponseHandlerIndex);
  CpvInitialize(int, CldRelocatedMessages);
  CpvInitialize(int, CldLoadBalanceMessages);
  CpvInitialize(int, CldMessageChunks);
  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldLoadResponseHandlerIndex) = 
    CmiRegisterHandler(CldLoadResponseHandler);
  CpvAccess(CldRelocatedMessages) = CpvAccess(CldLoadBalanceMessages) = 
    CpvAccess(CldMessageChunks) = 0;
  CldReadNeighborData();
  CldPeriodicBalanceInit();
}

void CldHandler(void *msg)
{
  CldInfoFn ifn;
  int len, queueing, priobits; unsigned int *prioptr;
  
  CmiGrabBuffer((void **)&msg);
  CldRestoreHandler(msg);
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &len, &queueing, &priobits, &prioptr);
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
}

void CldEnqueue(int pe, void *msg, int infofn, int packfn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn = (CldPackFn)CmiHandlerToFunction(packfn);

  if (CmiGetHandler(msg) >= CpvAccess(CmiHandlerMax)) *((int*)0)=0;

  if (pe == CLD_ANYWHERE)
    {
      ifn(msg, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      CldPutToken(msg); 
    } 
  else if (pe == CmiMyPe())
    {
      ifn(msg, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
    }
  else
    {
      pfn(&msg);
      ifn(msg, &len, &queueing, &priobits, &prioptr);
      CldSwitchHandler(msg, CpvAccess(CldHandlerIndex));
      CmiSetInfo(msg,infofn);
      if (pe==CLD_BROADCAST) 
	CmiSyncBroadcastAndFree(len, msg);
      else if (pe==CLD_BROADCAST_ALL)
	CmiSyncBroadcastAllAndFree(len, msg);
      else CmiSyncSendAndFree(pe, len, msg);
    }
}

void CldModuleInit()
{
  CpvInitialize(int, CldHandlerIndex);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler(CldHandler);
  CldModuleGeneralInit();
  CldGraphModuleInit();
}
