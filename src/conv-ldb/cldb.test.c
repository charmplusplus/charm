#include "converse.h"
#include "cldb.h"
#include "queueing.h"
#include <stdlib.h>

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "rand";
}

void CldBalanceHandler(void *msg)
{
  CldRestoreHandler(msg);
  CldPutToken(msg);
}

void CldHandler(char *msg)
{
  int len, queueing, priobits;
  unsigned int *prioptr; 
  CldInfoFn ifn; CldPackFn pfn;

  CldRestoreHandler(msg);
  ifn = (CldInfoFn)CmiHandlerToFunction(CmiGetInfo(msg));
  ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
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
  for(i=0;i<npes;i++) {
    CmiSyncSend(pes[i], len, msg);
  }
  CmiFree(msg);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;

  if (pe == CLD_ANYWHERE) {
    pe = (((CrnRand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());
    while (!CldPresentPE(pe))
      pe = (((CrnRand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());
    if (pe != CmiMyPe())
      CpvAccess(CldRelocatedMessages)++;
    if (pe == CmiMyPe()) {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      CldPutToken(msg);
    } 
    else {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      if (pfn) {
	pfn(&msg);
	ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      }
      CldSwitchHandler(msg, CpvAccess(CldBalanceHandlerIndex));
      CmiSetInfo(msg,infofn);
      CmiSyncSendAndFree(pe, len, msg);
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
    if (pe==CLD_BROADCAST) { CmiSyncBroadcastAndFree(len, msg); }
    else if (pe==CLD_BROADCAST_ALL) { CmiSyncBroadcastAllAndFree(len, msg); }
    else CmiSyncSendAndFree(pe, len, msg);
  }
}

void CldNodeEnqueue(int node, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;
  if (node == CLD_ANYWHERE) {
    /* node = (((rand()+CmiMyNode())&0x7FFFFFFF)%CmiNumNodes()); */
    node = (((CrnRand()+CmiMyNode())&0x7FFFFFFF)%CmiNumNodes());
    if (node != CmiMyNode())
      CpvAccess(CldRelocatedMessages)++;
  }
  if (node == CmiMyNode()) {
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

void CldModuleInit(char **argv)
{
  CpvInitialize(int, CldHandlerIndex);
  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler((CmiHandler)CldHandler);
  CpvAccess(CldBalanceHandlerIndex) = CmiRegisterHandler(CldBalanceHandler);
  CpvInitialize(int, CldRelocatedMessages);
  CpvInitialize(int, CldLoadBalanceMessages);
  CpvInitialize(int, CldMessageChunks);
  CpvAccess(CldRelocatedMessages) = CpvAccess(CldLoadBalanceMessages) = 
    CpvAccess(CldMessageChunks) = 0;
  CldModuleGeneralInit(argv);
}
