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

// template now
//typedef CmiUInt8 CkCacheKey;

typedef struct _CkCacheUserData {
  CmiUInt8 d0;
  CmiUInt8 d1;
} CkCacheUserData;


template<class CkCacheKey> class CkCacheEntryType;
template<class CkCacheKey> class CkCacheRequestorData;
template<class CkCacheKey> class CkCacheEntry;

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

template<class CkCacheKey> 
class CkCacheRequestMsg : public CMessage_CkCacheRequestMsg<CkCacheKey> {
 public:
  CkCacheKey key;
  int replyTo;
  CkCacheRequestMsg(CkCacheKey k, int reply) : key(k), replyTo(reply) { }
};

template<class CkCacheKey>
class CkCacheFillMsg : public CMessage_CkCacheFillMsg<CkCacheKey> {
public:
  CkCacheKey key;
  char *data;
  CkCacheFillMsg (CkCacheKey k) : key(k) {}
};


template<class CkCacheKey>
class CkCacheRequestorData {
public:
  CkCacheUserData userData;
  typedef void (*CkCacheCallback)(CkArrayID, CkArrayIndex&, CkCacheKey, CkCacheUserData &, void*, int);
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

template<class CkCacheKey>
class CkCacheEntryType {
public:
  virtual void * request(CkArrayIndex&, CkCacheKey) = 0;
  virtual void * unpack(CkCacheFillMsg<CkCacheKey> *, int, CkArrayIndex &) = 0;
  virtual void writeback(CkArrayIndex&, CkCacheKey, void *) = 0;
  virtual void free(void *) = 0;
  virtual int size(void *) = 0;
};

template<class CkCacheKey>
class CkCacheEntry {
public:
  CkCacheKey key;
  CkArrayIndex home;
  CkCacheEntryType<CkCacheKey> *type;
  std::vector< CkCacheRequestorData<CkCacheKey> > requestorVec;

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
  CkCacheEntry(CkCacheKey key, CkArrayIndex &home, CkCacheEntryType<CkCacheKey> *type) {
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

template<class CkCacheKey>
class CkCacheManager : public CBase_CkCacheManager<CkCacheKey> {

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
  std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*> *cacheTable;
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
  CkCacheManager(CkMigrateMessage *m): CBase_CkCacheManager<CkCacheKey>(m) { init(); }
  ~CkCacheManager() {}
  void pup(PUP::er &p);
 private:
  void init();
 public:

  void * requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, CkCacheRequestorData<CkCacheKey> &req);
  void * requestDataNoFetch(CkCacheKey key, int chunk);
  CkCacheEntry<CkCacheKey> * requestCacheEntryNoFetch(CkCacheKey key, int chunk);
  void recvData(CkCacheKey key, void *data, 
                CkCacheFillMsg<CkCacheKey> *msg = NULL);
  void recvData(CkCacheFillMsg<CkCacheKey> *msg);
  void recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data);

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
  std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*> *getCache();

};

  // from CkCache.C

  template<class CkCacheKey>
  CkCacheManager<CkCacheKey>::CkCacheManager(int size, CkGroupID gid) {
    init();
    numLocMgr = 1;
    numLocMgrWB = 0;
    locMgr = new CkGroupID[1];
    locMgr[0] = gid;
    maxSize = (CmiUInt8)size * 1024 * 1024;
  }

  template<class CkCacheKey>
  CkCacheManager<CkCacheKey>::CkCacheManager(int size, int n, CkGroupID *gid) {
    init();
    numLocMgr = n;
    numLocMgrWB = 0;
    locMgr = new CkGroupID[n];
    for (int i=0; i<n; ++i) locMgr[i] = gid[i];
    maxSize = (CmiUInt8)size * 1024 * 1024;
  }

  template<class CkCacheKey>
  CkCacheManager<CkCacheKey>::CkCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB) {
    init();
    numLocMgr = n;
    locMgr = new CkGroupID[n];
    for (int i=0; i<n; ++i) locMgr[i] = gid[i];
    numLocMgrWB = nWB;
    locMgrWB = new CkGroupID[nWB];
    for (int i=0; i<nWB; ++i) locMgrWB[i] = gidWB[i];
    maxSize = (CmiUInt8)size * 1024 * 1024;
  }

  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::init() {
    numChunks = 0;
    numLocMgr = 0;
    locMgr = NULL;
    maxSize = 0;
    syncdChares = 0;
    cacheTable = NULL;
    chunkAck = NULL;
    chunkWeight = NULL;
    storedData = 0;
#if COSMO_STATS > 0
    dataArrived = 0;
    dataTotalArrived = 0;
    dataMisses = 0;
    dataLocal = 0;
    totalDataRequested = 0;
#endif
  }

  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::pup(PUP::er &p) {
    p | numLocMgr;
    if (p.isUnpacking()) locMgr = new CkGroupID[numLocMgr];
    PUP::PUParray(p,locMgr,numLocMgr);
    p | numLocMgrWB;
    if (p.isUnpacking()) locMgrWB = new CkGroupID[numLocMgrWB];
    PUP::PUParray(p,locMgrWB,numLocMgrWB);
    p | maxSize;
  }

  template<class CkCacheKey>
  void * CkCacheManager<CkCacheKey>::requestData(CkCacheKey what, CkArrayIndex &_toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, CkCacheRequestorData<CkCacheKey> &req)
  {
    typename std::map<CkCacheKey, CkCacheEntry<CkCacheKey>* >::iterator  p;
    CkArrayIndex toWhom(_toWhom);
    CkAssert(chunkAck[chunk] > 0);
    p = cacheTable[chunk].find(what);
    CkCacheEntry<CkCacheKey> *e;
#if COSMO_STATS > 0
    totalDataRequested++;
#endif
    if (p != cacheTable[chunk].end()) {
      e = p->second;
      CkAssert(e->home == toWhom);
      //CkAssert(e->begin == begin);
      //CkAssert(e->end == end);
#if COSMO_STATS > 1
      e->totalRequests++;
#endif
      if (e->data != NULL) {
        return e->data;
      }
      if (!e->requestSent) {// || _nocache) {
        e->requestSent = true;
        if ((e->data = type->request(toWhom, what)) != NULL) {
          e->replyRecvd = true;
          return e->data;
        }
      }
    } else {
      e = new CkCacheEntry<CkCacheKey>(what, toWhom, type);
#if COSMO_STATS > 1
      e->totalRequests++;
#endif
      cacheTable[chunk][what] = e;
      e->requestSent = true;
      if ((e->data = type->request(toWhom, what)) != NULL) {
        e->replyRecvd = true;
        return e->data;
      }
    }

    e->requestorVec.push_back(req);
    outStandingRequests[what] = chunk;
#if COSMO_STATS > 1
    e->misses++;
#endif
    return NULL;
  }

  template<class CkCacheKey>
  void * CkCacheManager<CkCacheKey>::requestDataNoFetch(CkCacheKey key, int chunk) {
    typename std::map<CkCacheKey,CkCacheEntry<CkCacheKey> *>::iterator p = cacheTable[chunk].find(key);
    if (p != cacheTable[chunk].end()) {
      return p->second->data;
    }
    return NULL;
  }
  
  template<class CkCacheKey>
  CkCacheEntry<CkCacheKey> * CkCacheManager<CkCacheKey>::requestCacheEntryNoFetch(CkCacheKey key, int chunk) {
    typename std::map<CkCacheKey,CkCacheEntry<CkCacheKey> *>::iterator p = cacheTable[chunk].find(key);
    if (p != cacheTable[chunk].end()) {
      return p->second;
    }
    return NULL;
  }
  
  template<class CkCacheKey>
  std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*> *CkCacheManager<CkCacheKey>::getCache(){
    return cacheTable;
  }

template <class CkCacheKey> 
inline void CkCacheManager<CkCacheKey>::recvData(CkCacheKey key, void *data, CkCacheFillMsg<CkCacheKey> *msg) {

    typename std::map<CkCacheKey,int>::iterator pchunk = outStandingRequests.find(key);
    CkAssert(pchunk != outStandingRequests.end());
    int chunk = pchunk->second;
    CkAssert(chunk >= 0 && chunk < numChunks);
    CkAssert(chunkAck[chunk] > 0);
    outStandingRequests.erase(pchunk);

    typename std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*>::iterator p;
    p = cacheTable[chunk].find(key);
    CkAssert(p != cacheTable[chunk].end());
    CkCacheEntry<CkCacheKey> *e = p->second;
    if (msg != NULL) {
      e->data = e->type->unpack(msg, chunk, e->home);
    }
    else {
      e->data = data; 
    }
    storedData += e->type->size(e->data);
    
    typename std::vector<CkCacheRequestorData<CkCacheKey> >::iterator caller;
    for (caller = e->requestorVec.begin(); caller != e->requestorVec.end(); caller++) {
      caller->deliver(key, e->data, chunk);
    }
    e->requestorVec.clear();

}

  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::recvData(CkCacheFillMsg<CkCacheKey> *msg) {
    CkCacheKey key = msg->key;
    recvData(key, NULL, msg);     
  }
  
  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data) {
    typename std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*>::iterator p = cacheTable[chunk].find(key);
    CkCacheEntry<CkCacheKey> *e;
    if (p == cacheTable[chunk].end()) {
      e = new CkCacheEntry<CkCacheKey>(key, from, type);
      cacheTable[chunk][key] = e;
    } else {
      e = p->second;
      storedData -= e->type->size(e->data);
      e->type->writeback(e->home, e->key, e->data);
    }
    e->replyRecvd = true;
    e->data = data;
    storedData += e->type->size(data);
    
    typename std::vector<CkCacheRequestorData<CkCacheKey> >::iterator caller;
    for (caller = e->requestorVec.begin(); caller != e->requestorVec.end(); caller++) {
      caller->deliver(key, e->data, chunk);
    }
    e->requestorVec.clear();
  }

  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::cacheSync(int &_numChunks, CkArrayIndex &chareIdx, int &localIdx) {
    finishedChunks = 0;
    if (syncdChares > 0) {
      _numChunks = numChunks;
      //CkPrintf("Cache %d: sync following\n",thisgroup.idx);
    } else {
      syncdChares = 1;
      //CkPrintf("Cache %d: sync\n",thisgroup.idx);

      localChares.reset();
      localCharesWB.reset();
      for (int i=0; i<numLocMgr; ++i) {
        CkLocMgr *mgr = (CkLocMgr *)CkLocalBranch(locMgr[i]);
        mgr->iterate(localChares);
      }
      for (int i=0; i<numLocMgrWB; ++i) {
        CkLocMgr *mgr = (CkLocMgr *)CkLocalBranch(locMgrWB[i]);
        mgr->iterate(localChares);
        mgr->iterate(localCharesWB);
      }

#if COSMO_STATS > 0
      dataArrived = 0;
      dataTotalArrived = 0;
      dataMisses = 0;
      dataLocal = 0;
      totalDataRequested = 0;
      maxData = 0;
#endif

      for (int chunk=0; chunk<numChunks; ++chunk) {
        CkAssert(cacheTable[chunk].empty());
        CkAssert(chunkAck[chunk]==0);
        CkAssert(chunkAckWB[chunk]==0);
      }
      CkAssert(outStandingRequests.empty());
      storedData = 0;

      if (numChunks != _numChunks) {
        if(numChunks != 0) {
          delete []cacheTable;
          delete []chunkAck;
          delete []chunkAckWB;
          delete []chunkWeight;
        }
	  
        numChunks = _numChunks;
        cacheTable = new std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*>[numChunks];
        chunkAck = new int[numChunks];
        chunkAckWB = new int[numChunks];
        chunkWeight = new CmiUInt8[numChunks];
      }
      for (int i=0; i<numChunks; ++i) {
        chunkAck[i] = localChares.count;
        chunkAckWB[i] = localCharesWB.count;
        chunkWeight[i] = 0;
        //CkPrintf("[%d] CkCache::cacheSync group %d ack %d ackWb %d\n", CkMyPe(), this->thisgroup.idx, chunkAck[i], chunkAckWB[i]);
      }
      
#if COSMO_STATS > 0
      CmiResetMaxMemory();
#endif
    }

    localIdx = localChares.registered.get(chareIdx);
    CkAssert(localIdx != 0);
  }

  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::writebackChunk(int chunk) {
    //CkPrintf("[%d] CkCache::writebackChunk group %d ackWb %d\n", CkMyPe(), this->thisgroup.idx, chunkAckWB[chunk]);
    CkAssert(chunkAckWB[chunk] > 0);
    if (--chunkAckWB[chunk] == 0) {
      // we can safely write back the chunk to the senders
      // at this point no more changes to the data can be made until next fetch

      typename std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*>::iterator iter;
      for (iter = cacheTable[chunk].begin(); iter != cacheTable[chunk].end(); iter++) {
        CkCacheEntry<CkCacheKey> *e = iter->second;
        e->writeback();
      }

    }
  }

  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::finishedChunk(int chunk, CmiUInt8 weight) {
    //CkPrintf("[%d] CkCache::finishedChunk group %d ack %d\n", CkMyPe(), this->thisgroup.idx, chunkAck[chunk]);
    CkAssert(chunkAck[chunk] > 0);
    chunkWeight[chunk] += weight;
    //CkPrintf("Cache %d: finishedChunk %d\n",thisgroup.idx,chunkAck[chunk]);
    if (--chunkAck[chunk] == 0) {
      // we can safely delete the chunk from the cache
      
      // TODO: if chunks are held back due to restrictions, here is a
      // good position to release them

#if COSMO_STATS > 0
      if (maxData < storedData) maxData = storedData;
#endif

      typename std::map<CkCacheKey,CkCacheEntry<CkCacheKey>*>::iterator iter;
      for (iter = cacheTable[chunk].begin(); iter != cacheTable[chunk].end(); iter++) {
        CkCacheEntry<CkCacheKey> *e = iter->second;
        storedData -= e->type->size(e->data);
        
        // TODO: Store communication pattern here

        delete e;
      }
      cacheTable[chunk].clear();
      if (++finishedChunks == numChunks) {
        finishedChunks = 0;
        syncdChares = 0;
      }
    }
  }
  
  template<class CkCacheKey>
  void CkCacheManager<CkCacheKey>::collectStatistics(CkCallback &cb) {
#if COSMO_STATS > 0
    CkCacheStatistics cs(dataArrived, dataTotalArrived,
        dataMisses, dataLocal, dataError, totalDataRequested,
        maxData, CkMyPe());
    contribute(sizeof(CkCacheStatistics), &cs, CkCacheStatistics::sum, cb);
#else
    CkAbort("Invalid call, only valid if COSMO_STATS is defined");
#endif
  }

#define CK_TEMPLATES_ONLY
#include "CkCache.def.h"
#undef CK_TEMPLATES_ONLY

#endif
