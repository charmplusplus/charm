#ifndef _MULTICAST
#define _MULTICAST

#include "pup.h"
class mCastEntry;

class multicastSetupMsg;
class multicastGrpMsg;
class cookieMsg;
class CkMcastBaseMsg;
class reductionInfo;

typedef mCastEntry * mCastEntryPtr;
PUPbytes(mCastEntryPtr);

#include "CkMulticast.decl.h"

#if 0
class CkMcastReductionMsg: public CMessage_CkMcastReductionMsg {
friend class CkMulticastMgr;
public:
  int dataSize;
  char *data;
  CkSectionInfo sid;
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

typedef void (*redClientFn)(CkSectionInfo sid, void *param,int dataSize,void *data);

/**
  multicast manager is a CkDelegateMgr. It is a Group that can manage
  all sections of different chare arrays, so all functions need a 
  CkSectionInfo parameter to tell CkMulticastMgr which array section
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
      void pup(PUP::er &p){ p|idx; p|pe; }
    };
    typedef CkVec<IndexPos>  arrayIndexPosList;

  public:
    CkMulticastMgr()  { }
    int useDefCtor(void){ return 1; }
    void setSection(CkSectionInfo &id, CkArrayID aid, CkArrayIndexMax *, int n);
    void setSection(CkSectionInfo &id);
    void setSection(CProxySection_ArrayElement &proxy);
    virtual void initDelegateMgr(CProxy *proxy);
    void ArraySectionSend(CkDelegateData *pd,int ep,void *m, CkArrayID a, CkSectionID &s);
    // entry
    void teardown(CkSectionInfo s);  /**< entry: tear down the tree */
    void freeup(CkSectionInfo s);    /**< entry: free old tree */
    void setup(multicastSetupMsg *);  
    void recvCookie(CkSectionInfo sid, CkSectionInfo child);
    void childrenReady(mCastEntry *entry);
    void recvMsg(multicastGrpMsg *m);
    void recvPacket(CkSectionInfo &_cookie, int n, char *data, int seqno, int count, int totalsize, int fromBuffer);
    // for reduction
    void setReductionClient(CProxySection_ArrayElement &, redClientFn fn,void *param=NULL);
    void setReductionClient(CProxySection_ArrayElement &, CkCallback *cb);
    // user should be careful while passing non-default value of fragSize
    // fragSize%sizeof(data_type) should be zero
    void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, int fragSize=-1);
    void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, CkCallback &cb, int fragSize=-1);
    void rebuild(CkSectionInfo &);
    // entry
    void recvRedMsg(CkReductionMsg *msg);
    void updateRedNo(mCastEntryPtr, int red);
  public:
//    typedef CkMcastReductionMsg *(*reducerFn)(int nMsg,CkMcastReductionMsg **msgs);
  private:
    void initCookie(CkSectionInfo sid);
    void resetCookie(CkSectionInfo sid);
    enum {MAXREDUCERS=256};
//    static CkReduction::reducerFn reducerTable[MAXREDUCERS];
    void releaseFutureReduceMsgs(mCastEntryPtr entry);
    void releaseBufferedReduceMsgs(mCastEntryPtr entry);
    inline CkReductionMsg *buildContributeMsg(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &id, CkCallback &cb);
    void reduceFragment (int index, CkSectionInfo& id,
                         mCastEntry* entry, reductionInfo& redInfo,
                         int& updateReduceNo, int currentTreeUp);
    CkReductionMsg* combineFrags (CkSectionInfo& id,
                                  mCastEntry* entry,
                                  reductionInfo& redInfo);
};


extern void CkGetSectionInfo(CkSectionInfo &id, void *msg);

#endif
