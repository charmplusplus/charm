/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/****
  converse ldb for Blue Gene, this is the first version, full of hack and 
  need more work.
***/

#include "converse.h"
#include "blue.h"
#include "cldb.h"
#include "queueing.h"
#include <stdlib.h>

#define  DEBUGF(x)   /*CmiPrintf x;*/

extern int CldPresentPE(int pe);

void LoadNotifyFn(int l)
{
}

char *CldGetStrategy(void)
{
  return "rand";
}

void CldBalanceHandler(void *msg)
{
  CldRestoreHandler((char *)msg);
  CldPutToken((char *)msg);
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
  CldSwitchHandler((char *)msg, CpvAccess(CldHandlerIndex));
  CmiSetInfo(msg,infofn);
  for(i=0;i<npes;i++) {
    CmiSyncSend(pes[i], len, (char *)msg);
  }
  CmiFree(msg);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;
  int size;

  int sx,sy,sz; BgGetSize(&sx, &sy, &sz); size = (sx*sy*sz);

  DEBUGF(("[%d] CldEnqueue pe: %d infofn:%d\n", BgMyNode(), pe, infofn));
  if (pe == CLD_ANYWHERE) {
    pe = (((CrnRand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());
    while (!CldPresentPE(pe))
      pe = (((CrnRand()+CmiMyPe())&0x7FFFFFFF)%CmiNumPes());
    if (pe != CmiMyPe())
      CpvAccess(CldRelocatedMessages)++;
    if (pe == CmiMyPe()) {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      DEBUGF(("CldEnqueue at 1\n"));
/*
      CldPutToken((char *)msg);
*/
      {
      int x,y,z;
      BgGetXYZ(pe, &x, &y, &z);
      DEBUGF(("send to: %d %d %d handle:%d\n", x,y,z, CmiGetHandler(msg)));
      BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(msg), LARGE_WORK, len, (char *)msg);
      }
    } 
    else {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      if (pfn) {
	pfn(&msg);
	ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      }
      DEBUGF(("CldEnqueue at 2\n"));
/*
      CldSwitchHandler((char *)msg, CpvAccess(CldBalanceHandlerIndex));
      CmiSetInfo(msg,infofn);
      CmiSyncSendAndFree(pe, len, (char *)msg);
*/
      {
      int x,y,z;
      if (pe >= size) pe = size-1;
      BgGetXYZ(pe, &x, &y, &z);
      DEBUGF(("send to: %d %d %d handle:%d\n", x,y,z, CmiGetHandler(msg)));
      BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(msg), LARGE_WORK, len, (char *)msg);
      }
    }
  }
  else if ((pe == CmiMyPe()) || (CmiNumPes() == 1)) {
    DEBUGF(("CldEnqueue pe == CmiMyPe()\n"));
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CmiSetInfo(msg,infofn);
/*
    CsdEnqueueGeneral(msg, CQS_QUEUEING_LIFO, priobits, prioptr);
*/
    {
    int x,y,z;
    if (pe >= CmiNumPes()) pe = CmiNumPes()-1;
    BgGetXYZ(pe, &x, &y, &z);
    BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(msg), LARGE_WORK, len, (char *)msg);
    }
  }
  else {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    if (pfn) {
      pfn(&msg);
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    }
/*
    CldSwitchHandler((char *)msg, CpvAccess(CldHandlerIndex));
*/
    CmiSetInfo(msg,infofn);
    DEBUGF(("CldEnqueue pe=%d\n", pe));
    if (pe==CLD_BROADCAST) { 
CmiPrintf("CldEnqueue pe=%d\n", pe); CmiAbort("");
      CmiSyncBroadcastAndFree(len, (char *)msg); }
    else if (pe==CLD_BROADCAST_ALL) { 
CmiPrintf("CldEnqueue pe=%d\n", pe); CmiAbort("");
      CmiSyncBroadcastAllAndFree(len, (char *)msg); 
    }
    else {
/*
      CmiSyncSendAndFree(pe, len, (char *)msg);
*/
      int x,y,z;
      BgGetXYZ(pe, &x, &y, &z);
      DEBUGF(("send to: %d %d %d handle:%d\n", x,y,z, CmiGetHandler(msg)));
      BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(msg), LARGE_WORK, len, msg);
    }
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
    CldSwitchHandler((char *)msg, CpvAccess(CldHandlerIndex));
    CmiSetInfo(msg,infofn);
    if (node==CLD_BROADCAST) { CmiSyncNodeBroadcastAndFree(len, (char *)msg); }
    else if (node==CLD_BROADCAST_ALL){CmiSyncNodeBroadcastAllAndFree(len,(char *)msg);}
    else CmiSyncNodeSendAndFree(node, len, (char *)msg);
  }
}

void CldModuleInit()
{
  CpvInitialize(int, CldHandlerIndex);
  CpvInitialize(int, CldBalanceHandlerIndex);
  CpvAccess(CldHandlerIndex) = CmiRegisterHandler(CldHandler);
  CpvAccess(CldBalanceHandlerIndex) = CmiRegisterHandler(CldBalanceHandler);
  CpvInitialize(int, CldRelocatedMessages);
  CpvInitialize(int, CldLoadBalanceMessages);
  CpvInitialize(int, CldMessageChunks);
  CpvAccess(CldRelocatedMessages) = CpvAccess(CldLoadBalanceMessages) = 
    CpvAccess(CldMessageChunks) = 0;
  CldModuleGeneralInit();
}

