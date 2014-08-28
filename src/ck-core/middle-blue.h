/*****************************************************************************
          Blue Gene Middle Layer for Charm++ program
*****************************************************************************/

#ifndef _MIDDLE_BLUE_H_
#define _MIDDLE_BLUE_H_

#include <memory.h>
#include "converse.h"
#include "blue.h"

#undef CkMyPe
#undef CkNumPes
#undef CkMyRank
#undef CkMyNode
#undef CkNumNodes
#undef CkMyNodeSize
#undef CkNodeOf
#undef CkNodeSize
#undef CkNodeFirst

#undef CmiSyncSend
#undef CmiSyncSendAndFree
#undef CmiSyncBroadcast
#undef CmiSyncBroadcastAndFree
#undef CmiSyncBroadcastAll
#undef CmiSyncBroadcastAllAndFree

#undef CmiSyncNodeSend
#undef CmiSyncNodeSendAndFree
#undef CmiSyncNodeBroadcast
#undef CmiSyncNodeBroadcastAndFree
#undef CmiSyncNodeBroadcastAll
#undef CmiSyncNodeBroadcastAllAndFree


#undef CkWallTimer
#undef CkCpuTimer
#define CkWallTimer     BgGetTime
#define CkCpuTimer	BgGetTime
#define CkVTimer	BgGetTime
#define CkElapse   BgElapse

#define CkRegisterHandler(x)        BgRegisterHandler((BgHandler)(x))
#define CkRegisterHandlerEx(x, p)   BgRegisterHandlerEx((BgHandlerEx)(x), p)
#define CkNumberHandler(n, x)       BgNumberHandler(n, (BgHandler)(x))
#define CkNumberHandlerEx(n, x, p)  BgNumberHandlerEx(n, (BgHandlerEx)(x), p)

#define ConverseExit             BgCharmExit

/**
  This version Blue Gene Charm++ use a whole Blue Gene node as 
  a Charm PE.
*/
#if CMK_BIGSIM_NODE

#define CkpvDeclare 	BnvDeclare
#define CkpvExtern 	BnvExtern
#define CkpvStaticDeclare  BnvStaticDeclare
#define CkpvInitialize 	BnvInitialize
#define CkpvInitialized BnvInitialized
#define CkpvAccess	BnvAccess
#define CkpvAccessOther	BnvAccessOther

namespace BGConverse {

inline int CkMyPe() { return BgMyNode(); }
inline int CkNumPes() { int x,y,z; BgGetSize(&x, &y, &z); return (x*y*z); }
inline int CkMyRank() { return 0; }
inline int BgNodeRank() { return BgMyRank(); }
inline int CkMyNodeSize() { return 1; }

#if 0
static inline void CmiSyncSend(int pe, int nb, char *m) 
{
  int x,y,z;
  char *dupm = (char *)CmiAlloc(nb);

//CmiPrintf("[%d] CmiSyncSend handle:%d\n", CkMyPe(), CmiGetHandler(m));
  memcpy(dupm, m, nb);
  BgGetXYZ(pe, &x, &y, &z);
  BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void CmiSyncSendAndFree(int pe, int nb, char *m)
{
  int x,y,z;
//CmiPrintf("[%d] CmiSyncSendAndFree handle:%d\n", CkMyPe(), CmiGetHandler(m));
  BgGetXYZ(pe, &x, &y, &z);
  BgSendPacket(x,y,z, ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void CmiSyncBroadcast(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
//CmiPrintf("[%d] CmiSyncBroadcast handle:%d\n", CkMyPe(), CmiGetHandler(m));
  memcpy(dupm, m, nb);
  BgBroadcastPacketExcept(CkMyPe(), ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void CmiSyncBroadcastAndFree(int nb, char *m)
{
//CmiPrintf("CmiSyncBroadcastAndFree handle:%d\n", CmiGetHandler(m));
  BgBroadcastPacketExcept(CkMyPe(), ANYTHREAD, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void CmiSyncBroadcastAll(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
//CmiPrintf("CmiSyncBroadcastAll: handle:%d\n", CmiGetHandler(m));
  memcpy(dupm, m, nb);
  BgBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void CmiSyncBroadcastAllAndFree(int nb, char *m)
{
//CmiPrintf("CmiSyncBroadcastAllAndFree: handle:%d\n", CmiGetHandler(m));
  /* broadcast to all nodes */
  BgBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, m);
}
#endif

}  /* end of namespace */


#else   /* end if CMK_BIGSIM_NODE */

/**
  This version of Blue Gene Charm++ use a Blue Gene thread as 
  a Charm PE.
*/

#define CkpvDeclare 	   BpvDeclare
#define CkpvExtern 	   BpvExtern
#define CkpvStaticDeclare  BpvStaticDeclare
#define CkpvInitialize 	   BpvInitialize
#define CkpvInitialized    BpvInitialized
#define CkpvAccess	   BpvAccess
#define CkpvAccessOther	   BpvAccessOther

#define CksvDeclare 	   BnvDeclare
#define CksvExtern 	   BnvExtern
#define CksvStaticDeclare  BnvStaticDeclare
#define CksvInitialize 	   BnvInitialize
#define CksvAccess	   BnvAccess

namespace BGConverse {

static inline int CkMyPe() { return BgGetGlobalWorkerThreadID(); }
static inline int CkNumPes() { return BgNumNodes()*BgGetNumWorkThread(); }
static inline int CkMyRank() { return BgGetThreadID(); }
static inline int BgNodeRank() { return BgMyRank()*BgGetNumWorkThread()+BgGetThreadID(); }
static inline int CkMyNode() { return BgMyNode(); }
static inline int CkNodeOf(int pe) { return pe / BgGetNumWorkThread(); }
static inline int CkNumNodes() { return BgNumNodes(); }
static inline int CkMyNodeSize() { return BgGetNumWorkThread(); }
static inline int CkNodeSize(int node) { return BgGetNumWorkThread(); }
static inline int CkNodeFirst(int node) { return BgGetNumWorkThread()*node; }

static inline void CksdScheduler(int ret) { BgScheduler(ret); }
static inline void CksdExitScheduler() { BgExitScheduler(); }
static inline void CkDeliverMsgs(int nmsg)	{ BgDeliverMsgs(nmsg); }

#ifdef __cplusplus
extern "C"
#endif
void CkReduce(void *msg, int size, CmiReduceMergeFn mergeFn);

}  /* end of namespace */

#endif

#define CmiSyncSend			BgSyncSend
#define CmiSyncSendAndFree		BgSyncSendAndFree
#define CmiSyncBroadcast		BgSyncBroadcast
#define CmiSyncBroadcastAndFree 	BgSyncBroadcastAndFree
#define CmiSyncBroadcastAll		BgSyncBroadcastAll
#define CmiSyncBroadcastAllAndFree	BgSyncBroadcastAllAndFree

#define CmiSyncNodeSend			BgSyncNodeSend
#define CmiSyncNodeSendAndFree		BgSyncNodeSendAndFree
#define CmiSyncNodeBroadcast		BgSyncNodeBroadcast
#define CmiSyncNodeBroadcastAndFree	BgSyncNodeBroadcastAndFree
#define CmiSyncNodeBroadcastAll		BgSyncNodeBroadcastAll
#define CmiSyncNodeBroadcastAllAndFree	BgSyncNodeBroadcastAllAndFree

#undef CmiSyncListSendAndFree
#define CmiSyncListSendAndFree		BgSyncListSendAndFree

#define CmiMultipleSend			BgMultipleSend

#undef CsdEnqueueLifo
//#define CsdEnqueueLifo(m)  CmiSyncSendAndFree(CkMyPe(),((envelope*)m)->getTotalsize(), (char*)(m));
#define CsdEnqueueLifo(m)    		BgEnqueue((char*)m)

#undef CmiNodeAllBarrier
#define CmiNodeAllBarrier()

#undef CmiBarrier
#define CmiBarrier()

/** common functions for two versions */
namespace BGConverse {

static inline void BgCharmExit()
{
//  traceCharmClose();
  if (CkMyPe() == 0)  BgShutdown();
}

}


#endif
