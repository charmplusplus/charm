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

#define MAGIC 88              /**< multicast magic number for error checking */

/**
 CkMcastBaseMsg is the base class for all multicast message.
*/
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

/**
  multicast manager is a CkDelegateMgr. It is a Group that can manage
  all sections of different chare arrays, so all functions need a 
  CkSectionCookie parameter to tell CkMulticastMgr which array section
  it should work on.
*/
class CkMulticastMgr: public CkDelegateMgr {
  private:
    /// internal class for the pair of array index and its location.
    class IndexPos {
    public:
      CkArrayIndexMax idx;
      int  pe;
    public:
      IndexPos() {}
      IndexPos(int i): idx(i), pe(i) {}
      IndexPos(CkArrayIndexMax i, int p): idx(i), pe(p) {};
    };
    typedef CkVec<IndexPos>  arrayIndexPosList;

  public:
    CkMulticastMgr()  {};
    void setSection(CkSectionCookie &id, CkArrayID aid, CkArrayIndexMax *, int n);
    void setSection(CkSectionCookie &id);
    void setSection(CProxySection_ArrayElement &proxy);
    void ArraySectionSend(int ep,void *m, CkArrayID a, CkSectionCookie &s);
    // entry
    void teardown(CkSectionCookie s);  /**< entry: tear down the tree */
    void freeup(CkSectionCookie s);    /**< entry: free old tree */
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
