#ifndef __CACHEMANAGER_H__
#define __CACHEMANAGER_H__

#include <sys/types.h>
#include <vector>
#include <map>
#include <set>
#include "charm++.h"
#include "envelope.h"

#if COSMO_STATS > 0
#include <fstream>
#endif

/** NodeCacheEntry represents the entry for a remote 
node that is requested by the chares 
on a processor.
It stores the index of the remote chare from 
which node is to be requested and the local
chares that request it.***/

typedef CmiUInt8 CkCacheKey;

typedef struct _CkCacheUserData {
  CmiUInt8 d0;
  CmiUInt8 d1;
} CkCacheUserData;


class CkCacheEntryType;
class CkCacheRequestorData;
class CkCacheEntry;

#include "CkCache.decl.h"

class CkCacheStatistics {
  CmiUInt8 dataArrived;
  CmiUInt8 dataTotalArrived;
  CmiUInt8 dataMisses;
  CmiUInt8 dataLocal;
  CmiUInt8 dataError;
  CmiUInt8 totalDataRequested;
  CmiUInt8 maxData;
  int index;

  CkCacheStatistics() : dataArrived(0), dataTotalArrived(0),
    dataMisses(0), dataLocal(0), dataError(0),
    totalDataRequested(0), maxData(0), index(-1) { }

 public:
  CkCacheStatistics(CmiUInt8 pa, CmiUInt8 pta, CmiUInt8 pm,
          CmiUInt8 pl, CmiUInt8 pe, CmiUInt8 tpr,
          CmiUInt8 mp, int i) :
    dataArrived(pa), dataTotalArrived(pta), dataMisses(pm),
    dataLocal(pl), dataError(pe), totalDataRequested(tpr),
    maxData(mp), index(i) { }

  void printTo(CkOStream &os) {
    os << "  Cache: " << dataTotalArrived << " data arrived (corresponding to ";
    os << dataArrived << " messages), " << dataLocal << " from local Chares" << endl;
    if (dataError > 0) {
      os << "Cache: ======>>>> ERROR: " << dataError << " data messages arrived without being requested!! <<<<======" << endl;
    }
    os << "  Cache: " << dataMisses << " misses during computation" << endl;
    os << "  Cache: Maximum of " << maxData << " data stored at a time in processor " << index << endl;
    os << "  Cache: local Chares made " << totalDataRequested << " requests" << endl;
  }

  static CkReduction::reducerType sum;

  static CkReductionMsg *sumFn(int nMsg, CkReductionMsg **msgs) {
    CkCacheStatistics ret;
    ret.maxData = 0;
    for (int i=0; i<nMsg; ++i) {
      CkAssert(msgs[i]->getSize() == sizeof(CkCacheStatistics));
      CkCacheStatistics *data = (CkCacheStatistics *)msgs[i]->getData();
      ret.dataArrived += data->dataArrived;
      ret.dataTotalArrived += data->dataTotalArrived;
      ret.dataMisses += data->dataMisses;
      ret.dataLocal += data->dataLocal;
      ret.totalDataRequested += data->totalDataRequested;
      if (data->maxData > ret.maxData) {
        ret.maxData = data->maxData;
        ret.index = data->index;
      }
    }
    return CkReductionMsg::buildNew(sizeof(CkCacheStatistics), &ret);
  }
};

class CkCacheRequestMsg : public CMessage_CkCacheRequestMsg {
 public:
  CkCacheKey key;
  int replyTo;
  CkCacheRequestMsg(CkCacheKey k, int reply) : key(k), replyTo(reply) { }
};

class CkCacheFillMsg : public CMessage_CkCacheFillMsg {
public:
  CkCacheKey key;
  char *data;
  CkCacheFillMsg (CkCacheKey k) : key(k) {}
};

typedef void (*CkCacheCallback)(CkArrayID, CkArrayIndex&, CkCacheKey, CkCacheUserData &, void*, int);

class CkCacheRequestorData {
public:
  CkCacheUserData userData;
  CkCacheCallback fn;
  CkArrayID requestorID;
  CkArrayIndex requestorIdx;

  CkCacheRequestorData(CProxyElement_ArrayElement &el, CkCacheCallback f, CkCacheUserData &data) {
    userData = data;
    requestorID = el.ckGetArrayID();
    requestorIdx = el.ckGetIndex();
    fn = f;
  }
  
  void deliver(CkCacheKey key, void *data, int chunk) {
    fn(requestorID, requestorIdx, key, userData, data, chunk);
  }
};

class CkCacheEntryType {
public:
  virtual void * request(CkArrayIndex&, CkCacheKey) = 0;
  virtual void * unpack(CkCacheFillMsg *, int, CkArrayIndex &) = 0;
  virtual void writeback(CkArrayIndex&, CkCacheKey, void *) = 0;
  virtual void free(void *) = 0;
  virtual int size(void *) = 0;
};

class CkCacheEntry {
public:
  CkCacheKey key;
  CkArrayIndex home;
  CkCacheEntryType *type;
  std::vector<CkCacheRequestorData> requestorVec;

  void *data;
  
  bool requestSent;
  bool replyRecvd;
  bool writtenBack;
#if COSMO_STATS > 1
  /// total number of requests to this cache entry
  int totalRequests;
  /// total number of requests that missed this entry, if the request is
  /// to another TreePiece in the local processor we never miss
  int misses;
#endif
  CkCacheEntry(CkCacheKey key, CkArrayIndex &home, CkCacheEntryType *type) {
    replyRecvd = false;
    requestSent = false;
    writtenBack = false;
    data = NULL;
    this->key = key;
    this->home = home;
    this->type = type;
    #if COSMO_STATS > 1
    totalRequests=0;
    misses=0;
#endif
  }

  ~CkCacheEntry() {
    CkAssert(requestorVec.empty());
    if (!writtenBack) writeback();
    type->free(data);
  }

  inline void writeback() {
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

class CkCacheManager : public CBase_CkCacheManager {

  /***********************************************************************
   * Variables definitions
   ***********************************************************************/
  
  /// Number of chunks in which the cache is splitted
  int numChunks;
  /// Number of chunks that have already been completely acknowledged
  int finishedChunks;
  
  /// A list of all the elements that are present in the local processor
  /// for the current iteration
  CkCacheArrayCounter localChares;
  /// A list of all the elements that are present in the local processor
  /// for the current iteration with respect to writeback
  CkCacheArrayCounter localCharesWB;
  /// number of chares that have checked in for the next iteration
  int syncdChares;

  /// The number of arrays this Manager serves without support for writeback
  int numLocMgr;
  /// The group ids of the location managers of the arrays this Manager serves
  /// with support for writeback
  CkGroupID *locMgr;
  /// The number of arrays this Manager serves with support for writeback
  int numLocMgrWB;
  /// The group ids of the location managers of the arrays this Manager serves
  /// with support for writeback
  CkGroupID *locMgrWB;

#if COSMO_STATS > 0
  /// particles arrived from remote processors, this counts only the entries in the cache
  CmiUInt8 dataArrived;
  /// particles arrived from remote processors, this counts the real
  /// number of particles arrived
  CmiUInt8 dataTotalArrived;
  /// particles missed while walking the tree for computation
  CmiUInt8 dataMisses;
  /// particles that have been imported from local TreePieces
  CmiUInt8 dataLocal;
  /// particles arrived which were never requested, basically errors
  CmiUInt8 dataError;
  /** counts the total number of particles requested by all
    the chares on the processor***/
  CmiUInt8 totalDataRequested;
  /// maximum number of nodes stored at some point in the cache
  CmiUInt8 maxData;
#endif

  /// weights of the chunks in which the tree is divided, the cache will
  /// update the chunk division based on these values
  CmiUInt8 *chunkWeight;

  /// Maximum number of allowed data stored
  CmiUInt8 maxSize;
  
  /// number of acknowledgements awaited before deleting the chunk
  int *chunkAck;
  /// number of acknowledgements awaited before writing back the chunk
  int *chunkAckWB;

  /// hash table containing all the entries currently in the cache
  std::map<CkCacheKey,CkCacheEntry*> *cacheTable;
  int storedData;

  /// list of all the outstanding requests. The second field is the chunk for
  /// which this request is outstanding
  std::map<CkCacheKey,int> outStandingRequests;
    
  /***********************************************************************
   * Methods definitions
   ***********************************************************************/

 public:
  
  CkCacheManager(int size, CkGroupID gid);
  CkCacheManager(int size, int n, CkGroupID *gid);
  CkCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);
  CkCacheManager(CkMigrateMessage *m);
  ~CkCacheManager() {}
  void pup(PUP::er &p);
 private:
  void init();
 public:

  void * requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType *type, CkCacheRequestorData &req);
  void * requestDataNoFetch(CkCacheKey key, int chunk);
  CkCacheEntry * requestCacheEntryNoFetch(CkCacheKey key, int chunk);
  void recvData(CkCacheFillMsg *msg);
  void recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType *type, int chunk, void *data);

  void cacheSync(int &numChunks, CkArrayIndex &chareIdx, int &localIdx);

  /** Called from the TreePieces to acknowledge that a particular chunk
      can be written back to the original senders */
  void writebackChunk(int num);
  /** Called from the TreePieces to acknowledge that a particular chunk
      has been completely used, and can be deleted */
  void finishedChunk(int num, CmiUInt8 weight);
  /** Called from the TreePieces to acknowledge that they have completely
      finished their computation */

  /** Collect the statistics for the latest iteration */
  void collectStatistics(CkCallback& cb);
  std::map<CkCacheKey,CkCacheEntry*> *getCache();

};

#endif
