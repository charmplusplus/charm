#ifndef __CACHEMANAGER_DATA_STRUCTURES_H__
#define __CACHEMANAGER_DATA_STRUCTURES_H__

#include "charm++.h"

typedef struct _CkCacheUserData {
  CmiUInt8 d0;
  CmiUInt8 d1;
} CkCacheUserData;


template<typename CkCacheKey>
class CkCacheEntryType;

template<typename CkCacheKey>
class CkCacheRequestorData;

template<typename CkCacheKey>
class CkPeCacheEntry;

template<typename PointedType>
struct CkCachePointerContainer {
  PointedType *pointer;

  CkCachePointerContainer() :
    pointer(NULL)
  {}

  CkCachePointerContainer(PointedType *p) :
    pointer(p)
  {}
};

// used by leader from previous iteration to tell all
// PEs on node that it has finished deallocating the
// node cache from the previous iteration; after receipt
// of this message, it is ok for the new leader 
// to allocate the node cache
struct CkCacheDummy {
};

PUPbytes(CkCacheDummy);


template<typename CkCacheKey>
class CkCacheRequestorData {
  public:
  typedef void (*Callback)(CkArrayID, CkArrayIndex&, CkCacheKey, CkCacheUserData &, void*, int);

  public:
  CkCacheUserData userData;
  Callback fn;
  CkArrayID requestorID;
  CkArrayIndex requestorIdx;

  CkCacheRequestorData() {}
  CkCacheRequestorData(CProxyElement_ArrayElement &el, Callback f, CkCacheUserData &data) {
    userData = data;
    requestorID = el.ckGetArrayID();
    requestorIdx = el.ckGetIndex();
    fn = f;
  }
  
  void deliver(CkCacheKey key, void *data, int chunk) {
    fn(requestorID, requestorIdx, key, userData, data, chunk);
  }
};

// used by SMP cache managers to store requestor PEs.
struct CkNodeCacheRequestorData {
  int pe;

  CkNodeCacheRequestorData() {}
  CkNodeCacheRequestorData(int p) : 
    pe(p)
  {}
};

template<typename CkCacheKey>
class CkCacheFillMsg;

template<typename CkCacheKey>
class CkCacheEntryType {
public:
  virtual void *request(CkArrayIndex&, CkCacheKey) = 0;
  virtual void *unpack(CkCacheFillMsg<CkCacheKey> *, int, CkArrayIndex &) = 0;
  virtual void writeback(CkArrayIndex&, CkCacheKey, void *) = 0;
  virtual void free(void *) = 0;
  virtual int size(void *) = 0;
};

template<typename CkCacheKey>
class CkPeCacheEntry {
public:
  CkCacheKey key;
  CkArrayIndex home;
  CkCacheEntryType<CkCacheKey> *type;
  CkVec<CkCacheRequestorData<CkCacheKey> > requestorVec;

  void *data;
  
  bool requestSent;
  bool replyRecvd;
  bool writtenBack;

  CkPeCacheEntry(CkCacheKey key, const CkArrayIndex &home, CkCacheEntryType<CkCacheKey> *type) {
    replyRecvd = false;
    requestSent = false;
    writtenBack = false;
    data = NULL;
    this->key = key;
    this->home = home;
    this->type = type;
  }

  ~CkPeCacheEntry() {
  }

  inline void free(){
    CkAssert(requestorVec.size() == 0);
    if (!writtenBack) writeback();
    type->free(data);
  }

  inline void writeback() {
    type->writeback(home, key, data);
    writtenBack = true;
  }
};

// used to hold requests that arrive before the PE
// has received a valid (shared) cache
template<typename CkCacheKey>
struct BufferedRequest {
  CkCacheKey key;
  CkArrayIndex home;
  int chunk;
  CkCacheEntryType<CkCacheKey> *type;
  CkCacheRequestorData<CkCacheKey> requestor;

  BufferedRequest() {}
  BufferedRequest(CkCacheKey k, const CkArrayIndex &h, int c, CkCacheEntryType<CkCacheKey> *t, const CkCacheRequestorData<CkCacheKey> &r) : 
    key(k),
    home(h),
    chunk(c),
    type(t),
    requestor(r)
  {}
};


#include "threadsafe_hashtable/hashtable_mt.h"

/** NodeCacheEntry represents the entry for a remote 
node that is requested by the chares 
on a processor.
It stores the index of the remote chare from 
which node is to be requested and the local
chares that request it.***/


template<typename CkCacheKey>
struct CkNodeCacheEntry {
  CkCacheKey key;
  void *data;
  CkArrayIndexMax home;
  CkCacheEntryType<CkCacheKey> *type;
  CkVec<CkNodeCacheRequestorData> requestorVec;

  bool writtenBack;

  porlock lk_;

  CkNodeCacheEntry() : 
    data(NULL),
    type(NULL)
  {}

  CkNodeCacheEntry(CkCacheKey k, const CkArrayIndex &h, CkCacheEntryType<CkCacheKey> *t) :
    data(NULL),
    key(k),
    home(h),
    type(t)
  {}

  ~CkNodeCacheEntry(){
  }

  void lock() volatile {
    lk_.lock();
  }

  void unlock(){
    lk_.unlock();
  }

  void free(){
    CkAssert(requestorVec.size() == 0);
    if(!writtenBack) writeback();
    type->free(data);
  }

  void writeback(){
    type->writeback(home, key, data);
    writtenBack = true;
  }
};

class CkCacheArrayCounter : public CkLocIterator {
public:
  int count;
  CkHashtableT<CkArrayIndex, int> registered;
  CkCacheArrayCounter() : count(0) { }
  void addLocation(CkLocation &loc) {
    registered.put(loc.getIndex()) = ++count;
  }
  void reset() {
    count = 0;
    registered.empty();
  }
};


#endif // __CACHEMANAGER_DATA_STRUCTURES_H__

