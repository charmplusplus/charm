#ifndef _MULTICAST
#define _MULTICAST

#include "pup.h"
class mCastEntry;

class multicastSetupMsg;
class multicastGrpMsg;
class cookieMsg;
class CkMcastBaseMsg;

typedef mCastEntry * mCastEntryPtr;
PUPbytes(mCastEntryPtr);

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

#if 0
class CkMcastReductionMsg: public CMessage_CkMcastReductionMsg {
friend class CkMulticastMgr;
public:
  int dataSize;
  char *data;
  CkSectionCookie sid;
private:
  CkReduction::reducerType reducer;
  char flag;  // 1: come from array elem 2: come from BOC
  int redNo;
  int gcounter;
  char rebuilt;
  CkCallback callback;   /**< user callback */
public:
  static CkMcastReductionMsg* buildNew(int NdataSize,void *srcData,
		  CkReduction::reducerType reducer=CkReduction::invalid);
  void setCallback(CkCallback &cb) { callback = cb; }
  inline int getSize(void) const {return dataSize;}
  inline void *getData(void) {return data;}
};
#endif

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
    void setReductionClient(CProxySection_ArrayElement &, CkCallback *cb);
    void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &sid);
    void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &sid, CkCallback &cb);
    void rebuild(CkSectionCookie &);
    // entry
    void recvRedMsg(CkReductionMsg *msg);
    void updateRedNo(mCastEntryPtr, int red);
  public:
//    typedef CkMcastReductionMsg *(*reducerFn)(int nMsg,CkMcastReductionMsg **msgs);
  private:
    void initCookie(CkSectionCookie sid);
    void resetCookie(CkSectionCookie sid);
    enum {MAXREDUCERS=256};
//    static CkReduction::reducerFn reducerTable[MAXREDUCERS];
    void releaseFutureReduceMsgs(mCastEntryPtr entry);
    void releaseBufferedReduceMsgs(mCastEntryPtr entry);
    inline CkReductionMsg *buildContributeMsg(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &id, CkCallback &cb);
};


extern void CkGetSectionCookie(CkSectionCookie &id, void *msg);

#endif
