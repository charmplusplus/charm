#ifndef _MIDDLE_CONV_H_
#define _MIDDLE_CONV_H_

#include "converse.h"

#define CkpvDeclare 	CpvDeclare
#define CkpvExtern 	CpvExtern
#define CkpvStaticDeclare  CpvStaticDeclare
#define CkpvInitialize 	CpvInitialize
#define CkpvInitialized CpvInitialized
#define CkpvAccess	CpvAccess
#define CkpvAccessOther	CpvAccessOther

#define CksvDeclare 	   CsvDeclare
#define CksvExtern 	   CsvExtern
#define CksvStaticDeclare  CsvStaticDeclare
#define CksvInitialize 	   CsvInitialize
#define CksvAccess	   CsvAccess

#define CkReduce    CmiReduce

#undef CkMyPe
#undef CkNumPes

#define CkRegisterHandler(x)          CmiRegisterHandler(x)
#define CkNumberHandler(n, x)         CmiNumberHandler(n, x)
#define CkNumberHandlerEx(n, x, p)    CmiNumberHandlerEx(n, x, p)

#undef CmiSyncSend
#undef CmiSyncSendAndFree
#undef CmiSyncBroadcast
#undef CmiSyncBroadcastAndFree
#undef CmiSyncBroadcastAll
#undef CmiSyncBroadcastAllAndFree
#undef CmiSyncListSend
#undef CmiSyncListSendAndFree
#undef CmiSyncMulticast
#undef CmiSyncMulticastAndFree

#define CksdScheduler			CsdScheduler
#define CksdExitScheduler		CsdExitScheduler
#define CkDeliverMsgs			CmiDeliverMsgs

#define CkVTimer(x)	      0
#define CkElapse(x)   

#if CMK_REPLAYSYSTEM
extern "C" int ConverseDeliver();
#else
#define ConverseDeliver()   1
#endif

#if ! CMK_NAMESPACES_BROKEN
namespace Converse {
#endif

static inline int CkMyPe() { return CmiMyPe(); }
static inline int CkNumPes() { return CmiNumPes(); }

static inline void CmiSyncSend(int x, int y, char *z) 
{
  if (ConverseDeliver()) CmiSyncSendFn(x, y, z);
}
static inline void CmiSyncSendAndFree(int x, int y, char *z)
{
  if (ConverseDeliver()) CmiFreeSendFn(x, y, z);
}
static inline void CmiSyncBroadcast(int x, char *y)
{
  if (ConverseDeliver()) CmiSyncBroadcastFn(x, y);
}
static inline void CmiSyncBroadcastAndFree(int x, char *y)
{
  if (ConverseDeliver()) CmiFreeBroadcastFn(x, y);
}
static inline void CmiSyncBroadcastAll(int x, char *y)
{
  if (ConverseDeliver()) CmiSyncBroadcastAllFn(x, y);
}
static inline void CmiSyncBroadcastAllAndFree(int x, char *y)
{
  if (ConverseDeliver()) CmiFreeBroadcastAllFn(x, y);
}
static inline void CmiSyncListSend(int x, int *y, int w, char *z)
{
  if (ConverseDeliver()) CmiSyncListSendFn(x, y, w, z);
}
static inline void CmiSyncListSendAndFree(int x, int *y, int w, char *z)
{
  if (ConverseDeliver()) CmiFreeListSendFn(x, y, w, z);
}
static inline void CmiSyncMulticast(CmiGroup x, int y, char *z)
{
  if (ConverseDeliver()) CmiSyncMulticastFn(x, y, z);
}
static inline void CmiSyncMulticastAndFree(CmiGroup x, int y, char *z)
{
  if (ConverseDeliver()) CmiFreeMulticastFn(x, y, z);
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



#endif
