/*
 a cldb which does nothing. Just send the messge locally if the destination 
 processor is not specified.
*/

#include "converse.h"
#include "queueing.h"
#include "cldb.h"
#include <stdlib.h>

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "none";
}

void CldHandler(char *msg)
{
  int len, queueing, priobits;
  unsigned int *prioptr; CldInfoFn ifn; CldPackFn pfn;
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
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;
  if (pe == CLD_ANYWHERE) {
    pe = CmiMyPe();
  }
  if (pe == CmiMyPe() && !CmiImmIsRunning()) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    /* CsdEnqueueGeneral is not thread or SIGIO safe */
    CsdEnqueueGeneral(msg, queueing, priobits, prioptr);
  } else {
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
    node = CmiMyNode();
  }
  if (node == CmiMyNode() && !CmiImmIsRunning()) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CsdNodeEnqueueGeneral(msg, queueing, priobits, prioptr);
  } else {
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
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler((CmiHandler)CldHandler);
  CpvInitialize(int, CldRelocatedMessages);
  CpvInitialize(int, CldLoadBalanceMessages);
  CpvInitialize(int, CldMessageChunks);
  CpvAccess(CldRelocatedMessages) = CpvAccess(CldLoadBalanceMessages) = 
    CpvAccess(CldMessageChunks) = 0;
  CldModuleGeneralInit(argv);
}

void CldCallback(){}
