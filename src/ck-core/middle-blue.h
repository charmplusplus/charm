#ifndef _MIDDLE_BLUE_H_
#define _MIDDLE_BLUE_H_

#include "converse.h"
#include "blue.h"

#undef CkMyPe
#undef CkNumPes
#undef CkMyRank

#undef CmiSyncSend
#undef CmiSyncSendAndFree
#undef CmiSyncBroadcast
#undef CmiSyncBroadcastAndFree
#undef CmiSyncBroadcastAll
#undef CmiSyncBroadcastAllAndFree

#define CkRegisterHandler(x)     BgRegisterHandler((BgHandler)(x))
#define CkNumberHandler(n, x)    BgNumberHandler(n, (BgHandler)(x))

#define ConverseExit             BgCharmExit

/**
  This version Blue Gene Charm++ use a whole Blue Gene node as 
  a Charm PE.
*/
#if CMK_BLUEGENE_NODE

#define CkpvDeclare 	BnvDeclare
#define CkpvExtern 	BnvExtern
#define CkpvStaticDeclare  BnvStaticDeclare
#define CkpvInitialize 	BnvInitialize
#define CkpvAccess	BnvAccess
#define CkpvAccessOther	BnvAccessOther

namespace BGConverse {

inline int CkMyPe() { return BgMyNode(); }
inline int CkNumPes() { int x,y,z; BgGetSize(&x, &y, &z); return (x*y*z); }
inline int CkMyRank() { return BgMyRank(); }

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

}  /* end of namespace */



#else
/**
  This version of Blue Gene Charm++ use a Blue Gene thread as 
  a Charm PE.
*/

#define CkpvDeclare 	   BpvDeclare
#define CkpvExtern 	   BpvExtern
#define CkpvStaticDeclare  BpvStaticDeclare
#define CkpvInitialize 	   BpvInitialize
#define CkpvAccess	   BpvAccess
#define CkpvAccessOther	   BpvAccessOther


namespace BGConverse {

static inline int CkMyPe() { return BgGetGlobalWorkerThreadID(); }
static inline int CkNumPes() { return BgGetTotalSize()*BgGetNumWorkThread(); }
static inline int CkMyRank() { return BgMyRank()*BgGetNumWorkThread()+BgGetThreadID(); }

static inline void CmiSyncSend(int pe, int nb, char *m) 
{
  int x,y,z,t;
  char *dupm = (char *)CmiAlloc(nb);

//CmiPrintf("[%d] CmiSyncSend handle:%d\n", CkMyPe(), CmiGetHandler(m));
  memcpy(dupm, m, nb);
  t = pe%BgGetNumWorkThread();
  pe = pe/BgGetNumWorkThread();
  BgGetXYZ(pe, &x, &y, &z);
  BgSendPacket(x,y,z, t, CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void CmiSyncSendAndFree(int pe, int nb, char *m)
{
  int x,y,z,t;
//CmiPrintf("[%d] CmiSyncSendAndFree handle:%d\n", CkMyPe(), CmiGetHandler(m));
  t = pe%BgGetNumWorkThread();
  pe = pe/BgGetNumWorkThread();
  BgGetXYZ(pe, &x, &y, &z);
  BgSendPacket(x,y,z, t, CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void CmiSyncBroadcast(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
//CmiPrintf("[%d] CmiSyncBroadcast handle:%d\n", CkMyPe(), CmiGetHandler(m));
  memcpy(dupm, m, nb);
  BgThreadBroadcastPacketExcept(BgMyNode(), BgGetThreadID(), CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void CmiSyncBroadcastAndFree(int nb, char *m)
{
//CmiPrintf("CmiSyncBroadcastAndFree handle:%d node:%d tid:%d\n", CmiGetHandler(m), BgMyNode(), BgGetThreadID());
  BgThreadBroadcastPacketExcept(BgMyNode(), BgGetThreadID(), CmiGetHandler(m), LARGE_WORK, nb, m);
}

static inline void CmiSyncBroadcastAll(int nb, char *m)
{
  char *dupm = (char *)CmiAlloc(nb);
//CmiPrintf("CmiSyncBroadcastAll: handle:%d\n", CmiGetHandler(m));
  memcpy(dupm, m, nb);
  BgThreadBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, dupm);
}

static inline void CmiSyncBroadcastAllAndFree(int nb, char *m)
{
//CmiPrintf("CmiSyncBroadcastAllAndFree: handle:%d\n", CmiGetHandler(m));
  /* broadcast to all nodes */
  BgThreadBroadcastAllPacket(CmiGetHandler(m), LARGE_WORK, nb, m);
}

}  /* end of namespace */

#endif


/** common functions for two versions */
namespace BGConverse {

static inline void BgCharmExit()
{
  traceCharmClose();
  if (CkMyPe() == 0)  BgShutdown();
}

}




#endif
