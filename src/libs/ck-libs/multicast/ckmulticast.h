#ifndef _MULTICAST
#define _MULTICAST

class mCastEntry;

class multicastSetupMsg;
class multicastGrpMsg;
class cookieMsg;
class ReductionMsg;
class CkMcastBaseMsg;

typedef mCastEntry * mCastEntryPtr;

#include "CkMulticast.decl.h"

#define MAGIC 88

class CkMcastBaseMsg {
public:
  char magic;
  CkArrayID aid;
  CkSectionCookie _cookie;
  int ep;
public:
  CkMcastBaseMsg(): magic(MAGIC) {}
  static inline int checkMagic(CkMcastBaseMsg *m) { return m->magic == MAGIC; }
  inline int &gpe(void) { return _cookie.pe; }
  inline int &redno(void) { return _cookie.redNo; }
  inline void *&cookie(void) { return _cookie.val; }
};

typedef void (*redClientFn)(CkSectionCookie sid, void *param,int dataSize,void *data);

class CkMulticastMgr: public CkDelegateMgr {
  private:
  public:
    CkMulticastMgr()  {};
    void setSection(CkSectionCookie &id, CkArrayID aid, CkArrayIndexMax *, int n);
    void setSection(CkSectionCookie &id);
    void setSection(CProxySection_ArrayElement &proxy);
    void ArraySectionSend(int ep,void *m, CkArrayID a, CkSectionCookie &s);
    // entry
    void teardown(CkSectionCookie s);
    void freeup(CkSectionCookie s);
    void setup(multicastSetupMsg *);
    void recvCookie(CkSectionCookie sid, CkSectionCookie child);
    void childrenReady(mCastEntry *entry);
    void recvMsg(multicastGrpMsg *m);
    // for reduction
    void setReductionClient(CProxySection_ArrayElement &, redClientFn fn,void *param=NULL);
    void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &sid);
    void rebuild(CkSectionCookie &);
    // entry
    void recvRedMsg(ReductionMsg *msg);
    void updateRedNo(mCastEntryPtr, int red);
  public:
    typedef ReductionMsg *(*reducerFn)(int nMsg,ReductionMsg **msgs);
  private:
    void initCookie(CkSectionCookie sid);
    void resetCookie(CkSectionCookie sid);
    enum {MAXREDUCERS=256};
    static reducerFn reducerTable[MAXREDUCERS];
    void releaseFutureReduceMsgs(mCastEntryPtr entry);
    void releaseBufferedReduceMsgs(mCastEntryPtr entry);
};


extern void CkGetSectionCookie(CkSectionCookie &id, void *msg);

#endif
