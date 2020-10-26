#ifndef _CKFUTURES_H_
#define _CKFUTURES_H_

#include "CkFutures.decl.h"

#ifdef __cplusplus
extern CkMarshallMsg *CkAllocateMarshallMsg(int size, const CkEntryOptions *opts);
#include <type_traits>
#endif

/**
\addtogroup CkFutures
\brief Futures--ways to block Converse threads on remote events.

These routines are implemented in ckfutures.C.
*/
/*@{*/
typedef unsigned long long CkFutureID;
typedef struct _CkFuture {
  CkFutureID id;
  int        pe;
} CkFuture;
PUPbytes(CkFuture)

/* forward declare */
struct CkArrayID;

#ifdef __cplusplus
extern "C" {
#endif

CkFuture CkCreateFuture(void);
void  CkSendToFuture(CkFuture fut, void *msg);
void* CkWaitFuture(CkFuture futNum);
void CkReleaseFuture(CkFuture futNum);
int CkProbeFuture(CkFuture futNum);

void* CkRemoteCall(int eIdx, void *msg,const CkChareID *chare);
void* CkRemoteBranchCall(int eIdx, void *msg, CkGroupID gID, int pe);
void* CkRemoteNodeBranchCall(int eIdx, void *msg, CkGroupID gID, int node);
CkFutureID CkRemoteCallAsync(int eIdx, void *msg, const CkChareID *chare);
CkFutureID CkRemoteBranchCallAsync(int eIdx, void *msg, CkGroupID gID, int pe);
CkFutureID CkRemoteNodeBranchCallAsync(int eIdx, void *msg, CkGroupID gID, int node);

void* CkWaitFutureID(CkFutureID futNum);
void CkWaitVoidFuture(CkFutureID futNum);
void CkReleaseFutureID(CkFutureID futNum);
int CkProbeFutureID(CkFutureID futNum);
void  CkSendToFutureID(CkFutureID futNum, void *msg, int pe);
CkFutureID CkCreateAttachedFuture(void *msg);

CkFutureID CkCreateAttachedFutureSend(void *msg, int ep, struct CkArrayID id, CkArrayIndex idx,
                          void(*fptr)(struct CkArrayID, CkArrayIndex,void*,int,int),int size CK_MSGOPTIONAL);
/* CkFutureID CkCreateAttachedFutureSend(void *msg, int ep, void*,void(*fptr)(void*,void*,int,int)); */

void *CkWaitReleaseFuture(CkFutureID futNum);

void _futuresModuleInit(void);

#ifdef __cplusplus
}

namespace ck {
  template <typename T> class future {
    CkFuture handle_;

  public:
    future() { handle_ = CkCreateFuture(); }
    future(const future<T> &other) { handle_ = other.handle_; }

    T get() {
      CkMarshallMsg *msg = (CkMarshallMsg *)CkWaitFuture(handle_);
      PUP::fromMem p(msg->msgBuf);
      PUP::detail::TemporaryObjectHolder<T> holder;
      p | holder;
      delete msg;
      return std::move(holder.t);
    }

    void set(const T &value) {
      PUP::sizer s;
      s | (typename std::decay<decltype(value)>::type &)value;
      CkMarshallMsg *msg = CkAllocateMarshallMsg(s.size(), NULL);
      PUP::toMem p((void *)msg->msgBuf);
      p | (typename std::decay<decltype(value)>::type &)value;
      CkSendToFuture(handle_, msg);
    }

    bool probe() { return CkProbeFuture(handle_); }
    void release() { CkReleaseFuture(handle_); }
    void pup(PUP::er &p) { p | handle_; }
  };
}
#endif

#endif
