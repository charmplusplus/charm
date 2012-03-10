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

#if CMK_BIGSIM_CHARM

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

#if CMK_BIGSIM_NODE
static int BgMyPe() { return BgMyNode(); }
static int BgNumPes() { int x,y,z; BgGetSize(&x, &y, &z); return (x*y*z); }
#   define BGSENDPE(pe, msg, len)  {	\
      int x,y,z;	\
      BgGetXYZ(pe, &x, &y, &z);	\
      DEBUGF(("send to: (%d %d %d, %d) handle:%d\n", x,y,z,t, CmiGetHandler(msg)));  \
      BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(msg), LARGE_WORK, \
                   len, (char *)msg);	\
      }
#   define BGBDCASTALL(len, msg) 	\
      BgBroadcastAllPacket(CmiGetHandler(msg), LARGE_WORK, len, msg);
#   define BGBDCAST(len, msg) 	\
      BgBroadcastPacketExcept(BgMyNode(), ANYTHREAD, CmiGetHandler(msg), \
                              LARGE_WORK, len, msg);

#elif CMK_BIGSIM_THREAD
static int BgMyPe() { return BgGetGlobalWorkerThreadID(); }
static int BgNumPes() { return BgNumNodes()*BgGetNumWorkThread(); }
#   define BGSENDPE(pe, msg, len)  {	\
      int x,y,z,t;	\
      t = (pe)%BgGetNumWorkThread();	\
      pe = (pe)/BgGetNumWorkThread();	\
      BgGetXYZ(pe, &x, &y, &z);	\
      DEBUGF(("send to: (%d %d %d, %d) handle:%d\n", x,y,z,t, CmiGetHandler(msg)));  \
      BgSendPacket(x,y,z, t, CmiGetHandler(msg), LARGE_WORK, len, (char *)msg);	\
      }
#   define BGBDCASTALL(len, msg) 	\
      BgThreadBroadcastAllPacket(CmiGetHandler(msg), LARGE_WORK, len, msg);
#   define BGBDCAST(len, msg) 	\
      BgThreadBroadcastPacketExcept(BgMyNode(), BgGetThreadID(), \
                                    CmiGetHandler(msg), LARGE_WORK, len, msg);
#   define BGSENDNODE(node, msg, len)  {	\
      int x,y,z;	\
      BgGetXYZ(node, &x, &y, &z);	\
      DEBUGF(("send to: (%d %d %d) handle:%d\n", x,y,z, CmiGetHandler(msg)));  \
      BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(msg), LARGE_WORK, len, (char *)msg);	\
      }
#   define BGNODEBDCASTALL(len, msg) 	\
      BgBroadcastAllPacket(CmiGetHandler(msg), LARGE_WORK, len, msg);
#   define BGNODEBDCAST(len, msg) 	\
      BgBroadcastPacketExcept(BgMyNode(), ANYTHREAD,\
                                    CmiGetHandler(msg), LARGE_WORK, len, msg);
#endif

void CldEnqueueGroup(CmiGroup grp, void *msg, int infofn)
{
  CmiAbort("CldEnqueueGroup not supported!");
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
/*
  CldSwitchHandler((char *)msg, CpvAccess(CldHandlerIndex));
  CmiSetInfo(msg,infofn);
*/
  BgSyncListSend(npes, pes, CmiGetHandler(msg), LARGE_WORK, len, msg);
}

void CldEnqueue(int pe, void *msg, int infofn)
{
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(infofn);
  CldPackFn pfn;

  DEBUGF(("[%d>] CldEnqueue pe: %d infofn:%d\n", BgMyNode(), pe, infofn));
  if (pe == CLD_ANYWHERE) {
    pe = (((CrnRand()+BgMyPe())&0x7FFFFFFF)%BgNumPes());
/*
    while (!CldPresentPE(pe))
      pe = (((CrnRand()+BgMyPe())&0x7FFFFFFF)%BgNumPes());
*/
    if (pe != BgMyPe())
      CpvAccess(CldRelocatedMessages)++;
    if (pe == BgMyPe()) {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      CmiSetInfo(msg,infofn);
      DEBUGF(("CldEnqueue CLD_ANYWHERE (pe == BgMyPe)\n"));
/*
      CldPutToken((char *)msg);
*/
      BGSENDPE(pe, msg, len);
    } 
    else {
      ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      if (pfn) {
	pfn(&msg);
	ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
      }
      DEBUGF(("CldEnqueue at 2 pe=%d\n", pe));
/*
      CldSwitchHandler((char *)msg, CpvAccess(CldBalanceHandlerIndex));
      CmiSetInfo(msg,infofn);
      CmiSyncSendAndFree(pe, len, (char *)msg);
*/
      BGSENDPE(pe, msg, len);
    }
  }
  else if ((pe == BgMyPe()) || (BgNumPes() == 1)) {
    DEBUGF(("CldEnqueue pe == CmiMyPe()\n"));
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
    CmiSetInfo(msg,infofn);
/*
    CsdEnqueueGeneral(msg, CQS_QUEUEING_LIFO, priobits, prioptr);
*/
    pe = BgMyPe();
    BGSENDPE(pe, msg, len);
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
/*
      CmiSyncBroadcastAndFree(len, (char *)msg);
*/
      BGBDCAST(len, (char *)msg);
    }
    else if (pe==CLD_BROADCAST_ALL) { 
/*
      CmiSyncBroadcastAllAndFree(len, (char *)msg); 
*/
      BGBDCASTALL(len, (char *)msg);
    }
    else {
/*
      CmiSyncSendAndFree(pe, len, (char *)msg);
*/
      BGSENDPE(pe, msg, len);
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
    node = (((CrnRand()+BgMyNode())&0x7FFFFFFF)%BgNumNodes());
    if (node != BgMyNode())
      CpvAccess(CldRelocatedMessages)++;
  }
  if (node == BgMyNode()) {
    ifn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
/*
    CsdNodeEnqueueGeneral(msg, queueing, priobits, prioptr);
*/
    BGSENDNODE(node, msg, len);
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
    if (node==CLD_BROADCAST) { 
/*      CmiSyncNodeBroadcastAndFree(len, (char *)msg);  */
      BGNODEBDCAST(len, (char *)msg);
    }
    else if (node==CLD_BROADCAST_ALL){
/*      CmiSyncNodeBroadcastAllAndFree(len,(char *)msg); */
      BGNODEBDCASTALL(len, (char *)msg);
    }
    else {
/*      CmiSyncNodeSendAndFree(node, len, (char *)msg);  */
      BGSENDNODE(node, msg, len);
    }
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

void CldCallback(){}

#endif
