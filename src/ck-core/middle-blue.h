#include "converse.h"
#include "blue.h"


#define CkpvDeclare 	BnvDeclare
#define CkpvExtern 	BnvExtern
#define CkpvStaticDeclare  BnvStaticDeclare
#define CkpvInitialize 	BnvInitialize
#define CkpvAccess	BnvAccess

#if 0
#undef CpvDeclare
#undef CpvExtern
#undef CpvStaticDeclare
#undef CpvInitialize
#undef CpvAccess
#define CpvDeclare 	BnvDeclare
#define CpvExtern 	BnvExtern
#define CpvStaticDeclare  BnvStaticDeclare
#define CpvInitialize 	BnvInitialize
#define CpvAccess	BnvAccess
#endif

#undef CkMyPe
#undef CkNumPes
#undef CkMyRank

#define CkRegisterHandler(x)     BgRegisterHandler((BgHandler)(x))
#define CkNumberHandler(n, x)    BgNumberHandler(n, (BgHandler)(x))

#undef CmiSyncSend
#undef CmiSyncSendAndFree
#undef CmiSyncBroadcast
#undef CmiSyncBroadcastAndFree
#undef CmiSyncBroadcastAll
#undef CmiSyncBroadcastAllAndFree

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

}

