#include <stdio.h>
#include "converse.h"
#define PERIOD 10000

typedef struct CldField
{
  CLD_STANDARD_FIELD_STUFF
  int infofn, packfn;
} *CldField;

int Cld_fieldsize = sizeof(struct CldField);

typedef struct CldNeighborData
{
  int pe, load;
} *CldNeighborData;

CpvDeclare(CldNeighborData, neighbors);
CpvDeclare(int, numNeighbors);
CpvDeclare(int, responseCount);
CpvDeclare(CmiGroup, CldNeighbors);

CpvDeclare(int, CldHandlerIndex);
CpvDeclare(int, CldBalanceHandlerIndex);
CpvDeclare(int, CldLoadRequestHandlerIndex);
CpvDeclare(int, CldLoadResponseHandlerIndex);

void CldReadNeighborData()
{
  FILE *fp;
  char filename[25];
  int i;
  
  /* CmiError("On %d: Begin CldReadNeighborData\n", CmiMyPe()); */
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
  /* CmiError("On %d: End CldReadNeighborData\n", CmiMyPe()); */
}


void CldSetNeighborLoad(int pe, int load)
{
  int i=0, found=0;
  
  /* CmiError("On %d: Begin CldSetNeighborLoad\n", CmiMyPe()); */
  while (i<CpvAccess(numNeighbors) && !found)
    {
      if (CpvAccess(neighbors)[i].pe == pe)
	{
	  found = 1;
	  CpvAccess(neighbors)[i].load = load;
	}
      i++;
    }
  /* CmiError("On %d: End CldSetNeighborLoad\n", CmiMyPe()); */
}

int CldAvgNeighborLoad()
{
  int sum=0, i;
  
  /* CmiError("On %d: Begin CldAvgNeighborLoad\n", CmiMyPe()); */
  for (i=0; i<CpvAccess(numNeighbors); i++)
    sum += CpvAccess(neighbors)[i].load;
  sum += CldCountTokens();
  /* CmiError("On %d: End CldAvgNeighborLoad\n", CmiMyPe()); */
  return (int)(sum / (CpvAccess(numNeighbors)+1));

}

int CldGetRandomNeighborOrSelf()
{
  int randIndex = (((rand()+CmiMyPe())&0x7FFFFFFF)
		   % (CpvAccess(numNeighbors)+1));
  /* CmiError("On %d: Begin CldGetRandomNeighborOrSelf\n", CmiMyPe()); */  
  /* CmiError("On %d: In CldGetRandomNeighborOrSelf randIndex = %d numNeighbors = %d\n", CmiMyPe(), randIndex, CpvAccess(numNeighbors));    */
  if (randIndex == CpvAccess(numNeighbors))
    {
      /* CmiError("On %d: In CldGetRandomNeighborOrSelf 3\n", CmiMyPe()); */
      return (CmiMyPe());
      /* CmiError("On %d: In CldGetRandomNeighborOrSelf 4\n", CmiMyPe()); */
    }
  else
    {
      /* CmiError("On %d: In CldGetRandomNeighborOrSelf 5\n", CmiMyPe()); */
      return (CpvAccess(neighbors)[randIndex].pe);
      /* CmiError("On %d: In CldGetRandomNeighborOrSelf 6\n", CmiMyPe()); */
    }
}

void CldPeriodicBalanceInit()
{
  int len, myPe;
  int i;
  
  char *msg, *msgPos;
  
  /* Send CldLoadRequestHandler to all neighbors: */
  /* Construct msg with CmiMyPe() in it */
  /* Set message handler to CldLoadRequestHandler */
  /* CmiError("On %d: Begin CldPeriodicBalanceInit\n", CmiMyPe()); */
  myPe = CmiMyPe();
  len = CmiMsgHeaderSizeBytes + sizeof(int) + sizeof(void *);

  /* Temp solution to CmiSyncMulticastAndFree bug */
  for (i=0; i<CpvAccess(numNeighbors); i++){
    msg = (char *)CmiAlloc(len);
    msgPos = (msg + CmiMsgHeaderSizeBytes);
    memcpy((void *)msgPos, &myPe, sizeof(int));
    msgPos += sizeof(int);
    CmiSetHandler(msg, CpvAccess(CldLoadRequestHandlerIndex));
    CmiSyncSendAndFree(CpvAccess(neighbors)[i].pe, len, msg);
  }

  /*  CmiSyncMulticastAndFree(CpvAccess(CldNeighbors), len, msg); */
  /* CmiError("On %d: In CldPeriodicBalanceInit before CcdCallFnAfter\n", CmiMyPe()); */
  CcdCallFnAfter((CcdVoidFn)CldPeriodicBalanceInit, NULL, PERIOD);
  /* CmiError("On %d: End CldPeriodicBalanceInit\n", CmiMyPe()); */
}

void CldBalance()
{
  int avgload, i, j, overload, numToMove, numSent, infofn, packfn, *msgSizes;
  void **msgs;

  int len, queueing, priobits; unsigned int *prioptr; CldField ldbfield;
  CldInfoFn ifn;
  CldPackFn pfn;

  /* CmiError("On %d: Begin CldBalance\n", CmiMyPe()); */
  avgload = CldAvgNeighborLoad();
  overload = CldCountTokens() - avgload;
  if (overload > 0)
    {
      numToMove = overload / CpvAccess(numNeighbors);
      msgs = (void **)calloc(numToMove, sizeof(void *));
      msgSizes = (int *)calloc(numToMove, sizeof(int));
      
      for (i=0; i<CpvAccess(numNeighbors); i++)
	{
	  numSent = 0;
	  for (j=0; j<numToMove; j++)
	    {
	      CldGetToken(&msgs[j], &infofn, &packfn);
	      if (msgs[j] != 0)
		{
		  numSent++;
		  ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
		  ifn(msgs[j], &len, &ldbfield, &queueing, &priobits, 
		      &prioptr);
		  msgSizes[j] = len;
		  CldSwitchHandler(msgs[j], ldbfield, 
				   CpvAccess(CldBalanceHandlerIndex));
		  ldbfield->infofn = infofn;
		  ldbfield->packfn = packfn;
		}
	      else
		break;
	    }
	  CmiMultipleSend(CpvAccess(neighbors)[i].pe, numSent, msgSizes, msgs);
	}
      /* CmiError("On %d: End CldBalance\n", CmiMyPe()); */
    }
}

void CldLoadRequestHandler(char *msg)
{
  int srcpe, len, myPe, myLoad;
  char *newmsg, *msgPos;

  /* CmiError("On %d: Begin CldLoadRequestHandler\n", CmiMyPe()); */
  CmiGrabBuffer((void **)&msg);

  memcpy(&srcpe, msg + CmiMsgHeaderSizeBytes, sizeof(int));

  /* CmiError("On %d: In CldLoadRequestHandler, received request from %d\n", 
	   CmiMyPe(), srcpe); */
 
  myPe = CmiMyPe();
  myLoad = CldCountTokens();
  len = CmiMsgHeaderSizeBytes + 2*sizeof(int) + sizeof(void *);
  newmsg = (char *)CmiAlloc(len);
  
  msgPos = (newmsg + CmiMsgHeaderSizeBytes);
  memcpy((void *)msgPos, &myPe, sizeof(int));
  msgPos += sizeof(int);
  memcpy((void *)msgPos, &myLoad, sizeof(int));
  msgPos += sizeof(int);

  CmiSetHandler(newmsg, CpvAccess(CldLoadResponseHandlerIndex));
  CmiSyncSendAndFree(srcpe, len, newmsg);
  /* CmiError("On %d: End CldLoadRequestHandler\n", CmiMyPe()); */
}

void CldLoadResponseHandler(char *msg)
{
  int load, pe;

  /* CmiError("On %d: Begin CldLoadResponseHandler\n", CmiMyPe()); */
  CmiGrabBuffer((void **)&msg);

  /* Extract load and pe from msg */
  memcpy(&pe, msg + CmiMsgHeaderSizeBytes, sizeof(int));
  memcpy(&load, msg + CmiMsgHeaderSizeBytes + sizeof(int), sizeof(int));

  CldSetNeighborLoad(pe, load);
  CpvAccess(responseCount)++;
  if (CpvAccess(responseCount) == CpvAccess(numNeighbors))
    {
      CldBalance();
      CpvAccess(responseCount) = 0;
    }
  /* CmiError("On %d: End CldLoadResponseHandler\n", CmiMyPe()); */
}	
  
void CldBalanceHandler(void *msg)
{
  CldField ldbfield;
  
  /* CmiError("On %d: Begin CldBalanceHandler\n", CmiMyPe()); */
  CmiGrabBuffer((void **)&msg);
  CldRestoreHandler(msg, &ldbfield);
      
  CldPutToken(msg, ldbfield->infofn, ldbfield->packfn);
  /* CmiError("On %d: End CldBalanceHandler\n", CmiMyPe()); */
}


void CldGraphModuleInit()
{
  int *nbrs, i;

  /* CmiError("On %d: Begin CldGraphModuleInit\n", CmiMyPe()); */
  CpvInitialize(int, numNeighbors);
  CpvInitialize(int, responseCount);
  CpvAccess(responseCount) = 0;
  CpvInitialize(CmiGroup, CldNeighbors);
  CpvInitialize(CldNeighborData, neighbors);

  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvInitialize(int, CldLoadRequestHandlerIndex);
  CpvInitialize(int, CldLoadResponseHandlerIndex);
  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);
  CpvAccess(CldLoadRequestHandlerIndex) = 
    CmiRegisterHandler(CldLoadRequestHandler);
  CpvAccess(CldLoadResponseHandlerIndex) = 
    CmiRegisterHandler(CldLoadResponseHandler);

  CldReadNeighborData();

  /* Build processor group consisting of neighbors, for multicasts */
  nbrs = (int *)calloc(CpvAccess(numNeighbors), sizeof(int));
  for (i=0; i<CpvAccess(numNeighbors); i++)
    nbrs[i] = CpvAccess(neighbors)[i].pe;
  /* CpvAccess(CldNeighbors) = CmiEstablishGroup(CpvAccess(numNeighbors), nbrs); */

  /* Fire off the periodic rebalancing */
  CldPeriodicBalanceInit();
  /* CmiError("On %d: End CldGraphModuleInit\n", CmiMyPe()); */
}

void CldHandler(void *msg)
{
  CldField ldbfield;
  CldInfoFn ifn;
  int len, queueing, priobits; unsigned int *prioptr;
  
  /* CmiError("On %d: Begin CldHandler\n", CmiMyPe()); */
  CmiGrabBuffer((void **)&msg);
  CldRestoreHandler(msg, &ldbfield);
  ifn = (CldInfoFn)CmiHandlerToFunction(ldbfield->infofn);
  ifn(msg, &len, &ldbfield, &queueing, &priobits, &prioptr);
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
      
  /* CmiError("On %d: End CldHandler\n", CmiMyPe()); */
}

void CldEnqueue(int pe, void *msg, int infofn, int packfn)
{
  int len, queueing, priobits; unsigned int *prioptr; CldField ldbfield;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn = (CldPackFn)CmiHandlerToFunction(packfn);

  /* CmiError("On %d: Begin CldEnqueue\n", CmiMyPe()); */
  if (CmiGetHandler(msg) >= CpvAccess(CmiHandlerMax)) *((int*)0)=0;

  if (pe == CLD_ANYWHERE)
    {
      ifn(msg, &len, &ldbfield, &queueing, &priobits, &prioptr);
      ldbfield->infofn = infofn;
      ldbfield->packfn = packfn;
      CldPutToken(msg, infofn, packfn);
    } 
  else if (pe == CmiMyPe())
    {
      ifn(msg, &len, &ldbfield, &queueing, &priobits, &prioptr);
      ldbfield->infofn = infofn;
      ldbfield->packfn = packfn;
      CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
    }
  else
    {
      pfn(&msg);
      ifn(msg, &len, &ldbfield, &queueing, &priobits, &prioptr);
      CldSwitchHandler(msg, ldbfield, CpvAccess(CldHandlerIndex));
      ldbfield->infofn = infofn;
      ldbfield->packfn = packfn;
      if (pe==CLD_BROADCAST) 
	CmiSyncBroadcastAndFree(len, msg);
      else if (pe==CLD_BROADCAST_ALL)
	CmiSyncBroadcastAllAndFree(len, msg);
      else CmiSyncSendAndFree(pe, len, msg);
    }
  /* CmiError("On %d: End CldEnqueue\n", CmiMyPe()); */
}

void CldModuleInit()
{
  /* CmiError("On %d: Begin CldModuleInit\n", CmiMyPe()); */
  CpvInitialize(int, CldHandlerIndex);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler(CldHandler);
  CldModuleGeneralInit();
  CldGraphModuleInit();
  /* CmiError("On %d: End CldModuleInit\n", CmiMyPe()); */
}
