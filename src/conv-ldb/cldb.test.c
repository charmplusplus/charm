/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>

#ifdef WIN32
#include "queueing.h"
extern void CldGetToken(char **);
extern void CldSwitchHandler(char *, int);
extern int CldCountTokens();
extern void CldRestoreHandler(char *);
extern void CldPutToken(char *);
extern void CqsEnqueueGeneral(Queue, void *, unsigned int, unsigned int, unsigned int*);
extern void CldModuleGeneralInit();
#endif

#include "converse.h"
#define PERIOD 100
#define MAXMSGBFRSIZE 100000

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "test";
}

CpvDeclare(int, CldHandlerIndex);
CpvDeclare(int, CldBalanceHandlerIndex);
CpvDeclare(int, CldRelocatedMessages);
CpvDeclare(int, CldLoadBalanceMessages);
CpvDeclare(int, CldMessageChunks);

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
      CmiMultipleSend(pe, numSent, msgSizes, (char**) msgs);
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

void CldDistributeTokens()
{
  int destPe = (CmiMyPe()+1)%CmiNumPes();
  int numToSend;

  numToSend = CldEstimate() / 2;
  if (numToSend > CldCountTokens())
    numToSend = CldCountTokens() / 2;

  if (numToSend > 0)
    CldMultipleSend(destPe, numToSend);

  CcdCallFnAfter((CcdVoidFn)CldDistributeTokens, NULL, PERIOD);
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
}

void CldHelpModuleInit()
{
  CpvInitialize(int, CldBalanceHandlerIndex);

  CpvAccess(CldBalanceHandlerIndex) = 
    CmiRegisterHandler(CldBalanceHandler);

  if (CmiNumPes() > 1)
    CldDistributeTokens();
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
  CldHelpModuleInit();
}
