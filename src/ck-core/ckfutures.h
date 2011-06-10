#ifndef _CKFUTURES_H_
#define _CKFUTURES_H_

#include "CkFutures.decl.h"

/**
\addtogroup CkFutures
\brief Futures--ways to block Converse threads on remote events.

These routines are implemented in ckfutures.C.
*/
/*@{*/
typedef int CkFutureID;
typedef struct _CkFuture {
  CkFutureID id;
  int        pe;
} CkFuture;
PUPbytes(CkFuture)

extern "C"  CkFuture CkCreateFuture(void);
extern "C"  void  CkSendToFuture(CkFuture fut, void *msg);
extern "C"  void* CkWaitFuture(CkFuture futNum);
extern "C"  void CkReleaseFuture(CkFuture futNum);
extern "C"  int CkProbeFuture(CkFuture futNum);

extern "C"  void* CkRemoteCall(int eIdx, void *msg,const CkChareID *chare);
extern "C"  void* CkRemoteBranchCall(int eIdx, void *msg, CkGroupID gID, int pe);
extern "C"  void* CkRemoteNodeBranchCall(int eIdx, void *msg, CkGroupID gID, int node);
extern "C"  CkFutureID CkRemoteCallAsync(int eIdx, void *msg, const CkChareID *chare);
extern "C"  CkFutureID CkRemoteBranchCallAsync(int eIdx, void *msg, CkGroupID gID, int pe);
extern "C"  CkFutureID CkRemoteNodeBranchCallAsync(int eIdx, void *msg, CkGroupID gID, int node);

extern "C"  void* CkWaitFutureID(CkFutureID futNum);
extern "C"  void CkWaitVoidFuture(CkFutureID futNum);
extern "C"  void CkReleaseFutureID(CkFutureID futNum);
extern "C"  int CkProbeFutureID(CkFutureID futNum);
extern "C"  void  CkSendToFutureID(CkFutureID futNum, void *msg, int pe);
extern "C"  CkFutureID CkCreateAttachedFuture(void *msg);

/* forward declare */
struct CkArrayID;

extern "C"  CkFutureID CkCreateAttachedFutureSend(void *msg, int ep, struct CkArrayID id, CkArrayIndex idx,
                          void(*fptr)(struct CkArrayID, CkArrayIndex,void*,int,int),int size CK_MSGOPTIONAL);
/* extern "C"  CkFutureID CkCreateAttachedFutureSend(void *msg, int ep, void*,void(*fptr)(void*,void*,int,int)); */

extern "C"  void *CkWaitReleaseFuture(CkFutureID futNum);

extern "C"  void _futuresModuleInit(void);

#endif
