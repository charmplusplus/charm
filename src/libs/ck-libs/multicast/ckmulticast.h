#ifndef _MULTICAST
#define _MULTICAST

class mCastCookie;

class multicastSetupMsg;
class multicastGrpMsg;
class cookieMsg;
class ReductionMsg;

#include "CkMulticast.decl.h"

class CkMcastBaseMsg {
public:
  unsigned int _gpe, _redNo;
  void *_cookie;
public:
  unsigned int &gpe(void) { return _gpe; }
  unsigned int &redno(void) { return _redNo; }
  void *&cookie(void) { return _cookie; }
};

#define MAXMCASTCHILDREN  2

typedef void (*redClientFn)(CkSectionID sid, void *param,int dataSize,void *data);

class CkMulticastMgr: public CkDelegateMgr {
  private:
    CkSectionID sid;  // for now
    int idNum;
  public:
    CkMulticastMgr(): idNum(0) {};
    void setSection(CkSectionID &id, CkArrayID aid, CkArrayIndexMax *, int n);
    void setSection(CkSectionID &id);
    void setSection(CProxySection_ArrayElement *proxy);
    void teardown(CkSectionID s);
    void freeup(CkSectionID s);
    void init(CkSectionID sid);
    void reset(CkSectionID sid);
    cookieMsg * setup(multicastSetupMsg *);
    void ArraySend(int ep, void *m, const CkArrayIndexMax &idx, CkArrayID a);
    void ArrayBroadcast(int ep,void *m, CkArrayID a);
    void ArraySectionSend(int ep,void *m, CkArrayID a, CkSectionID &s);
    void recvMsg(multicastGrpMsg *m);
    // for reduction
    void setReductionClient(CkSectionID sid, redClientFn fn,void *param=NULL);
    void setReductionClient(CProxySection_ArrayElement *, redClientFn fn,void *param=NULL);
    void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionID &sid);
    void recvRedMsg(ReductionMsg *msg);
    void rebuild(CkSectionID &);
  public:
    typedef ReductionMsg *(*reducerFn)(int nMsg,ReductionMsg **msgs);
  private:
    enum {MAXREDUCERS=256};
    static reducerFn reducerTable[MAXREDUCERS];
    void releaseFutureReduceMsgs(mCastCookie *entry);
    void releaseBufferedReduceMsgs(mCastCookie *entry);
};


extern void setSectionID(void *msg, CkSectionID sid);
extern void CkGetSectionID(CkSectionID &id, void *msg);

#endif
