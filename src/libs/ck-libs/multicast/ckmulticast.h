#ifndef _MULTICAST
#define _MULTICAST

class mCastEntry;

class multicastSetupMsg;
class multicastGrpMsg;
class cookieMsg;
class ReductionMsg;

typedef mCastEntry * mCastEntryPtr;

#include "CkMulticast.decl.h"

#define MAXMCASTCHILDREN  2

class CkMcastBaseMsg {
public:
  char magic;      // TODO
  unsigned int _gpe, _redNo;
  void *_cookie;
  static const char MAGIC = 88 ;
public:
  CkMcastBaseMsg() { magic = MAGIC; }
  static int checkMagic(CkMcastBaseMsg *m) { return m->magic == MAGIC; }
  unsigned int &gpe(void) { return _gpe; }
  unsigned int &redno(void) { return _redNo; }
  void *&cookie(void) { return _cookie; }
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
    void init(CkSectionCookie sid);
    void reset(CkSectionCookie sid);
    cookieMsg * setup(multicastSetupMsg *);
    void recvMsg(multicastGrpMsg *m);
    // for reduction
    void setReductionClient(CProxySection_ArrayElement &, redClientFn fn,void *param=NULL);
    void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &sid);
    void rebuild(CkSectionCookie &);
    // entry
    void recvRedMsg(ReductionMsg *msg);
    void updateRedNo(mCastEntry *, int red);
  public:
    typedef ReductionMsg *(*reducerFn)(int nMsg,ReductionMsg **msgs);
  private:
    enum {MAXREDUCERS=256};
    static reducerFn reducerTable[MAXREDUCERS];
    void releaseFutureReduceMsgs(mCastEntry *entry);
    void releaseBufferedReduceMsgs(mCastEntry *entry);
};


extern void setSectionCookie(void *msg, CkSectionCookie sid);
extern void CkGetSectionCookie(CkSectionCookie &id, void *msg);

#endif
