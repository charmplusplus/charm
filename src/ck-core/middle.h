#include "converse.h"

#define CkpvDeclare 	CpvDeclare
#define CkpvExtern 	CpvExtern
#define CkpvStaticDeclare  CpvStaticDeclare
#define CkpvInitialize 	CpvInitialize
#define CkpvAccess	CpvAccess

#undef CkMyPe
#undef CkNumPes

#define CkRegisterHandler(x)     CmiRegisterHandler(x)
#define CkNumberHandler(n, x)    CmiNumberHandler(n, x)

#undef CmiSyncSend
#undef CmiSyncSendAndFree
#undef CmiSyncBroadcast
#undef CmiSyncBroadcastAndFree
#undef CmiSyncBroadcastAll
#undef CmiSyncBroadcastAllAndFree


#if ! CMK_NAMESPACES_BROKEN
namespace Converse {
#endif

static inline int CkMyPe() { return CmiMyPe(); }
static inline int CkNumPes() { return CmiNumPes(); }

static inline void CmiSyncSend(int x, int y, char *z) 
{
  CmiSyncSendFn(x, y, z);
}
static inline void CmiSyncSendAndFree(int x, int y, char *z)
{
  CmiFreeSendFn(x, y, z);
}
static inline void CmiSyncBroadcast(int x, char *y)
{
  CmiSyncBroadcastFn(x, y);
}
static inline void CmiSyncBroadcastAndFree(int x, char *y)
{
  CmiFreeBroadcastFn(x, y);
}
static inline void CmiSyncBroadcastAll(int x, char *y)
{
  CmiSyncBroadcastAllFn(x, y);
}
static inline void CmiSyncBroadcastAllAndFree(int x, char *y)
{
  CmiFreeBroadcastAllFn(x, y);
}

#if 0
template <class d>
class Cpv {
public:
#if CMK_SHARED_VARS_UNAVAILABLE
  d data;
#else
  d *data;
#endif
public:
  void init(void) {
  }
  d& operator = (d& val) {data = val.data;}
};
#endif

#if ! CMK_NAMESPACES_BROKEN
}
#endif


/*
#define CpvDeclare(t, v) Cpv<t> v
#define CpvExtern(t,v)   extern Cpv<t> v
#define CpvStaticDeclare(t,v)  static Cpv<t> v
#define CpvInitialize(t,v)     v.init()
#define CpvAccess(v)  v.data
*/
