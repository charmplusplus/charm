#ifndef __CACHEMANAGER_H__
#define __CACHEMANAGER_H__

#include <sys/types.h>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include "charm++.h"
#include "envelope.h"

#if COSMO_STATS > 0
#include <fstream>
#endif

#include "CkSmpCoordinator.h"
#include "CkCacheStatistics.h"
#include "CkCacheDataStructures.h"

#include "CkCache.decl.h"

template<typename CkCacheKey>
class CkCacheRequestMsg : public CMessage_CkCacheRequestMsg<CkCacheKey> {
 public:
  CkCacheKey key;
  int replyTo;
  CkCacheRequestMsg(CkCacheKey k, int reply) : key(k), replyTo(reply) { }
};

template<typename CkCacheKey>
class CkCacheFillMsg : public CMessage_CkCacheFillMsg<CkCacheKey> {
  public:
  CkCacheKey key;
  char *data;
  CkCacheFillMsg (CkCacheKey k) : key(k) {}
};

// type of hashtable used to implement shared node-level
// cache in SMP version; last template arg is hash 
// function type 
template<typename CkCacheKey>
class SmpCache;

PUPbytes(CkCachePointerContainer<void>);
PUPbytes(CkCachePointerContainer<CmiUInt8>);
PUPbytes(CkCachePointerContainer<CkCacheEntryType<CmiUInt8> >);
PUPbytes(CkCachePointerContainer<SmpCache<CmiUInt8> >);
#if CMK_HAS_INT16
PUPbytes(CkCachePointerContainer<CmiUInt16>);
PUPbytes(CkCachePointerContainer<CkCacheEntryType<CmiUInt16> >);
PUPbytes(CkCachePointerContainer<SmpCache<CmiUInt16> >);
#endif



template<typename CkCacheKey>
class CkCacheManagerBase : public CBase_CkCacheManagerBase<CkCacheKey> {
  /***********************************************************************
   * Variables definitions
   ***********************************************************************/
  
  /// Number of chunks into which the cache is split
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
  /// without support for writeback
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
  std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *> *peCache;
  int storedData;

  /// list of all the outstanding requests. The second field is the chunk for
  /// which this request is outstanding
  std::map<CkCacheKey, int> keyToChunk;

  // whether it is ok to consult cache table; always
  // safe to do so with non-smp version, but for smp 
  // version, is true only after this pe has received
  // the pointer to the shared cache from the leader
  // of the SMP node.
  bool cacheReady;

  // When all the chunks are done, this flag is set to true. This is used to
  // handle the case when a PE finishes all the chunks of all the objects and
  // then receives the leader's information. 
  bool finished_flag;
    
  /***********************************************************************
   * Methods definitions
   ***********************************************************************/

 protected:
  
  CkCacheManagerBase(int size, CkGroupID gid);
  CkCacheManagerBase(int size, int n, CkGroupID *gid);
  CkCacheManagerBase(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);


 public:
  CkCacheManagerBase(CkMigrateMessage *m);
  ~CkCacheManagerBase() {}

  void pup(PUP::er &p);
 private:
  void init();
 public:

  virtual void * requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req);
  virtual void * requestDataNoFetch(CkCacheKey key, int chunk);
  CkPeCacheEntry<CkCacheKey> * requestCacheEntryNoFetch(CkCacheKey key, int chunk);
  virtual void recvData(CkCacheFillMsg<CkCacheKey> *msg);
  virtual void recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data);

  // FIXME haven't defined this yet.
  //CkCacheEntry<CkCacheKey> * requestCacheEntryNoFetch(CkCacheKey key, int chunk);

  void cacheSync(int &numChunks, CkArrayIndex &chareIdx, int &localIdx);

  /** Called from the TreePieces to acknowledge that a particular chunk
      has been completely used, and can be deleted */
  void finishedChunk(int num, CmiUInt8 weight);
  /** Called from the TreePieces to acknowledge that they have completely
      finished their computation */

  /** Collect the statistics for the latest iteration */
  void collectStatistics(CkCallback& cb);
  std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *> *getCache();

  protected:
  bool &ready();
  bool &finishedAll();
  void clearPeCacheChunk(int chunk);
  void postClearPeCacheChunkCheck();
  virtual void deallocateNodeCache(){}

  virtual void bufferRequestUntilInitialization(CkCacheKey what, const CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req); 

  CkPeCacheEntry<CkCacheKey> *findPeCacheEntry(CkCacheKey key, int chunk);

  virtual void *lookupNodeCache(CkCacheKey what, CkArrayIndex &owner, CkCacheEntryType<CkCacheKey> *type, int chunk);

  // although this should be pure virtual, generated
  // charm++ code tries to allocate objects of type
  // CkCacheManagerBase, so we give it empty def
  virtual void allocateNodeCache(int newNumChunks) {}
  void allocatePeCache(int newNumChunks);
  /** Called from the TreePieces to acknowledge that a particular chunk
      can be written back to the original senders */
  virtual void writebackChunk(int chunk) {}

  int &getStoredData();
  std::map<CkCacheKey,int> &getKeyToChunk();
  void printKeyToChunk() const;
  int &getChunkAck(int chunk);
  int &getChunkAckWB(int chunk);
  int getNumChunks() const;
  void moreStoredData(int nBytes);
  void lessStoredData(int nBytes);
  int getFinishedChunks() const;
  int getLocalChareCount() const;

  void peDeliverData(CkPeCacheEntry<CkCacheKey> *entry, void *data, int chunk);
  virtual void peFinishedChunk(int chunk, CmiUInt8 weight) {}
  virtual void freePeEntry(CkPeCacheEntry<CkCacheKey> *e) {}

  std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *> &getPeCache(int chunk);

  void countObjects();

  CkPeCacheEntry<CkCacheKey> *lookupPeCache(CkCacheKey what, int chunk, const CkArrayIndex &toWhom, CkCacheEntryType<CkCacheKey> *type);

  int getSyncdChares() const { return syncdChares; }
};

template<typename CkCacheKey>
class CkCacheManager : public CBase_CkCacheManager<CkCacheKey> {
  public:

  CkCacheManager(int size, CkGroupID gid);
  CkCacheManager(int size, int n, CkGroupID *gid);
  CkCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);

  CkCacheManager(CkMigrateMessage *m);
  void pup(PUP::er &p);

  private:
  void bufferRequestUntilInitialization(CkCacheKey what, const CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req); 

  void allocateNodeCache(int newNumChunks);
  void peFinishedChunk(int chunk, CmiUInt8 weight);
  void freePeEntry(CkPeCacheEntry<CkCacheKey> *e);
  void writebackChunk(int chunk);

  public:
  void *requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req);
  void * requestDataNoFetch(CkCacheKey key, int chunk);
  void recvData(CkCacheFillMsg<CkCacheKey> *msg);
  void recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data);

};

template<typename CkCacheKey>
class CkSmpCacheManager : public CBase_CkSmpCacheManager<CkCacheKey> {
  private:
  CkVec<BufferedRequest<CkCacheKey> > bufferedRequests;
  CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> > coordinatorProxy_;
  CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> > *coordinator_;
  CProxy_CkSmpCacheManager<CkCacheKey> myProxy_;
  // saved registration callback
  CkCallback callback_;
  CkGroupID mcastGrpId_;

  // node-level map from cache key to chunk
  std::map<CkCacheKey, int> nodeKeyToChunk_;

  std::queue<int> chunksToDelete_;
  bool okToAllocateNodeCache_;
  bool okToDeallocateNodeCache_;
  bool pendingRegistration_;

  SmpCache<CkCacheKey> *nodeCache;

  protected:
  CkSmpCacheManager(int size, CkGroupID gid);
  CkSmpCacheManager(int size, int n, CkGroupID *gid);
  CkSmpCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);

  CProxy_CkSmpCacheManager<CkCacheKey> &smpProxy();
  bool &okToAllocateNodeCache() { return okToAllocateNodeCache_; }
  bool &okToDeallocateNodeCache() { return okToDeallocateNodeCache_; }
  bool &pendingRegistration() { return pendingRegistration_; }

  CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> > *&coordinator() { return coordinator_; }

  std::map<CkCacheKey, int> &getNodeKeyToChunk();

  public:
  CkSmpCacheManager();
  CkSmpCacheManager(CkMigrateMessage *m);
  void pup(PUP::er &p);

  protected:
  void bufferRequestUntilInitialization(CkCacheKey what, const CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req); 
  void sendNodeReplies(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data);

  virtual void nodeDeliverData(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data);

  bool isLeader() const;

  SmpCache<CkCacheKey> *&getNodeCache() { return nodeCache; }


  private:
  void smpInit();
  void releaseBufferedRequests();
  CkNodeCacheEntry<CkCacheKey> *findNodeCacheEntry(CkCacheKey key, int chunk);
  void allocateNodeCache(int newNumChunks);
  void peFinishedChunk(int chunk, CmiUInt8 weight);
  void freePeEntry(CkPeCacheEntry<CkCacheKey> *e);
  void writebackChunk(int chunk);
  void clearNodeCacheChunk(int chunk);
  void deallocateNodeCache();

  public:
  void *requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req);
  void * requestDataNoFetch(CkCacheKey key, int chunk);
  void recvData(CkCacheFillMsg<CkCacheKey> *msg);
  void recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data);

  void mcast(CkCachePointerContainer<SmpCache<CkCacheKey> > &container);

  void peFinishedChunkDone();
  void nodeReply(CkCacheKey key, const CkCachePointerContainer<void> &data);

  void setup(CkSmpCacheHandle<CkCacheKey> &handle, const CkCallback &cb);
  void registration(const CkCallback &cb);
  void doneRegistration();
  void doneRegistrationBody();
  void cleanupForCheckpoint(const CkCallback &cb);
};

template<typename CkCacheKey>
class CkOnefetchSmpCacheManager : public CBase_CkOnefetchSmpCacheManager<CkCacheKey> {
  public:
  CkOnefetchSmpCacheManager(int size, CkGroupID gid);
  CkOnefetchSmpCacheManager(int size, int n, CkGroupID *gid);
  CkOnefetchSmpCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);
  void pup(PUP::er &p);

  CkOnefetchSmpCacheManager(CkMigrateMessage *m);
  void nodeRequest(CkCacheKey key, CkArrayIndex &owner, int chunk, const CkCachePointerContainer<CkCacheEntryType<CkCacheKey> > &typeContainer, int requestorPe);


  private:
  void *lookupNodeCache(CkCacheKey what, CkArrayIndex &owner, CkCacheEntryType<CkCacheKey> *type, int chunk);
  void nodeDeliverData(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data);

};

template<typename CkCacheKey>
class CkMultifetchSmpCacheManager : public CBase_CkMultifetchSmpCacheManager<CkCacheKey> {
  public:
  CkMultifetchSmpCacheManager(int size, CkGroupID gid);
  CkMultifetchSmpCacheManager(int size, int n, CkGroupID *gid);
  CkMultifetchSmpCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);

  CkMultifetchSmpCacheManager(CkMigrateMessage *m);
  void pup(PUP::er &p);

  private:
  void *lookupNodeCache(CkCacheKey what, CkArrayIndex &owner, CkCacheEntryType<CkCacheKey> *type, int chunk);
  void nodeDeliverData(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data);
};


template<typename CkCacheKey>
struct CkNonSmpCacheHandle {
  CProxy_CkCacheManager<CkCacheKey> cacheProxy;

  CProxy_CkCacheManager<CkCacheKey> &getCacheProxy(){
    return cacheProxy;
  }

  void pup(PUP::er &p){
    p | cacheProxy;
  }
};

template<typename CkCacheKey>
struct CkSmpCacheHandle {
  CProxy_CkSmpCacheManager<CkCacheKey> cacheProxy;
  CkGroupID mcastGrpId;
  CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> > coordinatorProxy;

  CProxy_CkSmpCacheManager<CkCacheKey> &getCacheProxy(){
    return cacheProxy;
  }
  
  CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> > &getCoordinatorProxy(){
    return coordinatorProxy;
  }

  void pup(PUP::er &p){
    p | cacheProxy;
    p | mcastGrpId;
    p | coordinatorProxy;
  }
};

// to instantiate cache manager 
template<typename CkCacheKey>
class CkNonSmpCacheFactory {
  public:
  static CkNonSmpCacheHandle<CkCacheKey> instantiate(int size, CkGroupID gid);
  static CkNonSmpCacheHandle<CkCacheKey> instantiate(int size, int n, CkGroupID *gid);
  static CkNonSmpCacheHandle<CkCacheKey> instantiate(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);
};

template<typename CkCacheKey>
class CkOnefetchSmpCacheFactory {
  public:
  static CkSmpCacheHandle<CkCacheKey> instantiate(int size, CkGroupID gid);
  static CkSmpCacheHandle<CkCacheKey> instantiate(int size, int n, CkGroupID *gid);
  static CkSmpCacheHandle<CkCacheKey> instantiate(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);
  static void instantiateCoordinator(CkSmpCacheHandle<CkCacheKey> &handle);
};

template<typename CkCacheKey>
class CkMultifetchSmpCacheFactory {
  public:
  static CkSmpCacheHandle<CkCacheKey> instantiate(int size, CkGroupID gid);
  static CkSmpCacheHandle<CkCacheKey> instantiate(int size, int n, CkGroupID *gid);
  static CkSmpCacheHandle<CkCacheKey> instantiate(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB);
  static void instantiateCoordinator(CkSmpCacheHandle<CkCacheKey> &handle);
};

#include <sstream>
#include "CkCache.h"
#include "SmpCache.h"

#define CK_CACHE_VERBOSE /*CkPrintf*/

template<typename CkCacheKey>
CkCacheManagerBase<CkCacheKey>::CkCacheManagerBase(int size, CkGroupID gid) {
  init();
  numLocMgr = 1;
  numLocMgrWB = 0;
  locMgr = new CkGroupID[1];
  locMgr[0] = gid;
  maxSize = (CmiUInt8)size * 1024 * 1024;
}

template<typename CkCacheKey>
CkCacheManagerBase<CkCacheKey>::CkCacheManagerBase(int size, int n, CkGroupID *gid) {
  init();
  numLocMgr = n;
  numLocMgrWB = 0;
  locMgr = new CkGroupID[n];
  for (int i=0; i<n; ++i) locMgr[i] = gid[i];
  maxSize = (CmiUInt8)size * 1024 * 1024;
}

template<typename CkCacheKey>
CkCacheManagerBase<CkCacheKey>::CkCacheManagerBase(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB) {
  init();
  numLocMgr = n;
  locMgr = new CkGroupID[n];
  for (int i=0; i<n; ++i) locMgr[i] = gid[i];
  numLocMgrWB = nWB;
  locMgrWB = new CkGroupID[nWB];
  for (int i=0; i<nWB; ++i) locMgrWB[i] = gidWB[i];
  maxSize = (CmiUInt8)size * 1024 * 1024;
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::init() {
  numChunks = 0;
  numLocMgr = 0;
  locMgr = NULL;
  maxSize = 0;
  syncdChares = 0;
  peCache = NULL;
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
  finishedChunks = 0;
  ready() = false;
  finishedAll() = false;
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::pup(PUP::er &p) {
  p | numLocMgr;
  if (p.isUnpacking()) locMgr = new CkGroupID[numLocMgr];
  PUParray(p,locMgr,numLocMgr);
  p | numLocMgrWB;
  if (p.isUnpacking()) locMgrWB = new CkGroupID[numLocMgrWB];
  PUParray(p,locMgrWB,numLocMgrWB);
  p | maxSize;
  p | cacheReady;
  p | finished_flag;
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::pup(PUP::er &p){
}
 
template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::pup(PUP::er &p){
  p | coordinatorProxy_;
  //if(p.isUnpacking()){
  //  coordinator_ = coordinatorProxy_[CkMyPe()].ckLocal();
  //}
  p | myProxy_;
  p | mcastGrpId_;
  // copy queue contents into CkVec and serialize
  CkVec<int> qCopy;
  if(!p.isUnpacking()){
    while(!chunksToDelete_.empty()){
      qCopy.push_back(chunksToDelete_.front());
      chunksToDelete_.pop();
    }
  }
  p | qCopy;
  if(p.isUnpacking()){
    for(int i = 0; i < qCopy.size(); i++){
      chunksToDelete_.push(qCopy[i]);
    }
  }
  p | okToAllocateNodeCache_;
  p | okToDeallocateNodeCache_;
  p | pendingRegistration_;
}

template<typename CkCacheKey>
void CkOnefetchSmpCacheManager<CkCacheKey>::pup(PUP::er &p){
}

template<typename CkCacheKey>
void CkMultifetchSmpCacheManager<CkCacheKey>::pup(PUP::er &p){
}



template<typename CkCacheKey>
bool &CkCacheManagerBase<CkCacheKey>::ready(){
  return cacheReady;
}

template<typename CkCacheKey>
bool &CkCacheManagerBase<CkCacheKey>::finishedAll(){
  return finished_flag;
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::bufferRequestUntilInitialization(CkCacheKey what, const CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req){
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::bufferRequestUntilInitialization(CkCacheKey what, const CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req){
  // for non smp cache, each pe is responsible for reinitializing its 
  // cache, and no one else can touch it, so as long as requestData()
  // is invoked after cacheSync() has been called at least once, we 
  // should not have to buffer any requests
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::bufferRequestUntilInitialization(CkCacheKey what, const CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req){
  // for the smp cache, only one pe on an smp node can allocate the
  // node-level cache for the entire node. therefore, we must wait until that special
  // pe (the leader) has allocated the node-level cache and sent us a notification.
  // in the meanwhile, objects on the pe may start making requests; these must be 
  // buffered until we have a pointer to the node-level cache.
  bufferedRequests.push_back(BufferedRequest<CkCacheKey>(what, toWhom, chunk, type, req));
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::releaseBufferedRequests(){
  CkAssert(this->ready());
  //CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::releaseBufferedRequests\n", CkMyPe(), this->thisgroup.idx);
  for(int i = 0; i < bufferedRequests.size(); i++){
    BufferedRequest<CkCacheKey> &req = bufferedRequests[i];
    void *data = requestData(req.key, req.home, req.chunk, req.type, req.requestor);
    if(data != NULL){
      req.requestor.deliver(req.key, data, req.chunk);
    }
  }
  bufferedRequests.resize(0);
}

template<typename CkCacheKey>
void * CkCacheManagerBase<CkCacheKey>::requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req){
  CkAbort("Shouldn't call CkCacheManagerBase::requestData; only subclasses\n");
  return NULL;
}

template<typename CkCacheKey>
void * CkCacheManager<CkCacheKey>::requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req){
  // in non-smp version, there is no node-level
  // cache to be allocated on a possibly different
  // core, and pe-level cache must already have
  // been allocated in cacheSync(), so we must
  // be ready to serve data requests.
  CkAssert(this->ready());

  // cache has been initialized; check in pe-level cache 
  // lookupPeCache creates a new pe-level cache entry if required 
  // (i.e. if the data has never been requested on this pe before)
  CkPeCacheEntry<CkCacheKey> *peEntry = this->lookupPeCache(what, chunk, toWhom, type);
  if(peEntry->data == NULL){
    // data not available on pe
    if(!peEntry->requestSent){
      // send out request if you haven't already 
      // done so; also record chunk corresponding 
      // to request key, if making the request
      type->request(toWhom, what);
      peEntry->requestSent = true;
      this->getKeyToChunk()[what] = chunk;
    }
    peEntry->requestorVec.push_back(req);
  }
  return peEntry->data;
}

template<typename CkCacheKey>
void *CkSmpCacheManager<CkCacheKey>::requestData(CkCacheKey what, CkArrayIndex &toWhom, int chunk, CkCacheEntryType<CkCacheKey> *type, const CkCacheRequestorData<CkCacheKey> &req){
  // first check whether cache has been initialized
  if(!this->ready()){
    CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::requestData key %llu !ready\n", CkMyPe(), this->thisgroup.idx, what);
    // if not, buffer request until it has been
    bufferRequestUntilInitialization(what, toWhom, chunk, type, req);
    return NULL;
  }

  // cache has been initialized; check in pe-level cache 
  // lookupPeCache creates a new pe-level cache entry if required 
  // (i.e. if the data has never been requested on this pe before)
  CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::lookupPeCache key %llu\n", CkMyPe(), this->thisgroup.idx, what);
  CkPeCacheEntry<CkCacheKey> *peEntry = this->lookupPeCache(what, chunk, toWhom, type);

  if(peEntry->data != NULL){
    // data is already available in pe-level cache, return it 
    return peEntry->data;
  }
  else{
    // data is not available in pe-level cache.
    // the 'requestSent' variable helps us tell whether the entry we just
    // received from the pe-level cache above already existed before the
    // current object requested it, or it was created by the call to
    // lookupPeCache above in response to this object's request. If the former,
    // then we don't have to lookup the node-level cache, since some object
    // will have previously looked it up for us; we just have to add ourselves
    // to requestor list. otherwise, if the latter, i.e. this is the first
    // object on this PE to be looking up this particular key, we need to look
    // at the node-level cache as well
    if(peEntry->requestSent){
      // some other object before us will already have looked at 
      // the node-level cache. moreover, that object must have found
      // the data pointer of the corresponding node-level entry to be NULL,
      // since otherwise it would just have set the peEntry->data to non-NULL,
      // and this object would have seen it above. so, our PE is yet to receive
      // the data from the remote fetcher. all we have to do is add teh current
      // requestor object to the list of requestors for this data on this PE.
      peEntry->requestorVec.push_back(req);
      return NULL;
    }
    else{
      // no other PE has looked at the node-level cache for this entry before
      // the current object. 
      peEntry->requestSent = true;
      // therefore, no object can have set the peEntry->data pointer for this PE.
      // it is up to us to the current object to do so, if the data is available in 
      // the node-level cache.

      // look for data in node-level cache 
      CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::lookupNodeCache key %llu\n", CkMyPe(), this->thisgroup.idx, what);
      void *data = this->lookupNodeCache(what, toWhom, type, chunk);

      // did the node-level cache have the data?
      if(data != NULL){
        // yes it did
        // set the peEntry's data pointer to non-NULL for all future
        // requestors of this data on this PE
        peEntry->data = data;
      }
      else{
        // no it didn't; 
        // add self to pe-local requestor list, so that
        // when our PE gets the nodeReply() message, the data therein is
        // given to the current object in addition to any others added after
        // this call to requestData.
        peEntry->requestorVec.push_back(req);
        // also, since this the first (and only) time that the node-level
        // cache will be queried for this entry, and the data will be received
        // later, via a nodeReply() message, we should save the chunk for key
        this->getKeyToChunk()[what] = chunk;
      }
      return data;
    }

    CkAbort("CkMultifetchSmpCacheManager::requestData shouldn't be here 1!");
    return NULL;
  }

  CkAbort("CkMultifetchSmpCacheManager::requestData shouldn't be here 2!");
  return NULL;
}

// if you use these versions, you are expecting to find
// the entry in the pe/node cache table
template<typename CkCacheKey>
CkPeCacheEntry<CkCacheKey> * CkCacheManagerBase<CkCacheKey>::findPeCacheEntry(CkCacheKey key, int chunk){
  typename std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *>::iterator it;
  it = peCache[chunk].find(key);
  CkAssert(it != peCache[chunk].end());
  return it->second;
}

template<typename CkCacheKey>
CkNodeCacheEntry<CkCacheKey> * CkSmpCacheManager<CkCacheKey>::findNodeCacheEntry(CkCacheKey key, int chunk){
  CkNodeCacheEntry<CkCacheKey> *entry = NULL;
  entry = this->getNodeCache()->get(chunk, key);
  CkAssert(entry != NULL);
  return entry;
}

template<typename CkCacheKey>
CkPeCacheEntry<CkCacheKey> * CkCacheManagerBase<CkCacheKey>::lookupPeCache(CkCacheKey what, int chunk, const CkArrayIndex &toWhom, CkCacheEntryType<CkCacheKey> *type){
  CkAssert(chunkAck[chunk] > 0);

  typename std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *>::iterator p;
  p = peCache[chunk].find(what);

  CkPeCacheEntry<CkCacheKey> *e;

  if(p == peCache[chunk].end()){
    e = new CkPeCacheEntry<CkCacheKey>(what, toWhom, type);
    // any way to avoid another lookup while inserting?
    peCache[chunk][what] = e;
  }
  else{
    e = p->second;
  }

  return e;
}

template<typename CkCacheKey>
void *CkCacheManagerBase<CkCacheKey>::lookupNodeCache(CkCacheKey what, CkArrayIndex &owner, CkCacheEntryType<CkCacheKey> *type, int chunk){
  return NULL;
}

template<typename CkCacheKey>
void *CkOnefetchSmpCacheManager<CkCacheKey>::lookupNodeCache(CkCacheKey what, CkArrayIndex &owner, CkCacheEntryType<CkCacheKey> *type, int chunk){
  // query node-level hash table
  CkNodeCacheEntry<CkCacheKey> *entry = this->getNodeCache()->get(chunk, what);

  if(entry == NULL || entry->data == NULL){
    // either no entry for this key exists in the node level cache,
    // or there is an entry, but its data is not yet valid; in either
    // case, since in this version of the smp cache, only the fetcher
    // can alter the node-level cache, we must send a message to it
    // to update that cache.
    CProxy_CkOnefetchSmpCacheManager<CkCacheKey>(this->thisProxy)[this->coordinator()->getLeader()].nodeRequest(what, owner, chunk, CkCachePointerContainer<CkCacheEntryType<CkCacheKey> >(type), CkMyPe());
    return NULL;
  }

  return entry->data;
}

template<typename CkCacheKey>
void *CkMultifetchSmpCacheManager<CkCacheKey>::lookupNodeCache(CkCacheKey what, CkArrayIndex &owner, CkCacheEntryType<CkCacheKey> *type, int chunk){
  // query node-level hash table
  CkNodeCacheEntry<CkCacheKey> *entry = this->getNodeCache()->get(chunk, what);
  volatile CkNodeCacheEntry<CkCacheKey> *inserted = NULL;
  // no such entry; i.e. no other pe on this node
  // has made this request previously
  if(entry == NULL){
    // try to insert an entry into the node-level 
    // table
    entry = new CkNodeCacheEntry<CkCacheKey>(what, owner, type);
    inserted = this->getNodeCache()->single_put(chunk, what, entry);

    // since someone else might be trying this at the
    // same time, check whether our entry got through
    // or not
    bool success = (entry == inserted);
    if(success){
      // the entry you tried to insert into the node
      // level table DID go through. therefore, you
      // are the fetcher for this entry

      // ask for the client-requested piece of data
      // from its owner
      type->request(owner, what);
      this->getNodeKeyToChunk()[what] = chunk;

      // no need to add self to list of requestors:
      // could have had it so that in this version,
      // fetcher is always assumed to have requested
      // the data.  but for sake of consistency with
      // Onefetch version, we add the fetcher pe to
      // the list of requestors in the node cache
      // entry.  this is done because in recvData()
      // we erase the keyToChunk record for the key
      // once we receive the data corresponding to
      // it. this happens only on the fetcher. other
      // pes will still have the entry for the key in
      // their keyToChunk maps. therefore, in
      // sendNodeReplis() we should distinguish
      // between passing the recvd data to
      // non-fetchers (via node Reply) and passing it
      // to the fetcher (via peDeliverData()). in
      // order to have the same piece of code do this
      // in the Onefetch and Multifetch cases, we
      // should add the fetcher pe to the list of
      // requestor pes and check for equality of
      // requestorVec[i] and CkMyPe()
    }
    else{
      delete entry;
    }
  }
  else{
    inserted = entry;
  }

  // At this point, either you tried to insert an
  // entry into the table and failed (but got the
  // pointer to the inserted entry), or you found a
  // pre-existing entry in the node-level table.  in
  // either case, you must add yourself to the list
  // of requestors, taking care of the following: in
  // the time between your checking the node cache
  // for the existence of an entry for the given key,
  // and your trying to append yourself to the list
  // of requestors, the fetcher may finish fetching
  // the data and deliver it to the list of
  // requestors (which doesn't include you at this
  // point). So, you must lock the entry while adding
  // yourself to the list. Moreover, after
  // successfully locking the entry, check again
  // whether the data is now available and add self
  // to list only if data is unavailable.

  if(inserted->data == NULL){
    inserted->lock();
    CkNodeCacheEntry<CkCacheKey> *nvInserted = const_cast<CkNodeCacheEntry<CkCacheKey> *>(inserted);
    // have to recheck whether data is null, since fetcher could 
    // have inserted the data after we did the first check, but
    // before we acquired the lock 
    if(nvInserted->data == NULL){
      nvInserted->requestorVec.push_back(CkNodeCacheRequestorData(CkMyPe()));
    }

    void *dataToReturn = nvInserted->data; 

    nvInserted->unlock();
    return dataToReturn;
  }
  else{
    return inserted->data;
  }

  CkAbort("CkMultifetchSmpCacheManager::lookupNodeCache Shouldn't be here!!\n");
  return NULL;
}

template<typename CkCacheKey>
void CkOnefetchSmpCacheManager<CkCacheKey>::nodeRequest(CkCacheKey key, CkArrayIndex &owner, int chunk, const CkCachePointerContainer<CkCacheEntryType<CkCacheKey> > &typeContainer, int requestorPe){ 
  CkAssert(this->isLeader());
  // we cannot assert that the leader will be rdy when it receives
  // a request from a non-leader. this is because the leader may already
  // have finished its chunks (thereby setting rdy to false) 
  // before a request from a non-leader arrives.
  //CkAssert(ready());

  CK_CACHE_VERBOSE("[%d] Cache %d: CkOnefetchSmpCacheManager::nodeRequest key %llu from %d\n", CkMyPe(), this->thisgroup.idx, key, requestorPe);
  CkNodeCacheEntry<CkCacheKey> *entry = this->getNodeCache()->get(chunk, key);
  if(entry == NULL){
    // no pe has requested this key before
    // ask owner for data
    typeContainer.pointer->request(owner, key);
    this->getNodeKeyToChunk()[key] = chunk;

    // create node-level cache entry
    entry = new CkNodeCacheEntry<CkCacheKey>(key, owner, typeContainer.pointer);
    // and insert it into the table
    this->getNodeCache()->put(chunk, key, entry);
    //CK_CACHE_VERBOSE("[%d] Cache %d: CkOnefetchSmpCacheManager::nodeRequest key %llu not found, request sent\n", CkMyPe(), this->thisgroup.idx, key);
  }

  // either you have just inserted this entry,
  // or it already existed
  if(entry->data == NULL){
    //CK_CACHE_VERBOSE("[%d] Cache %d: CkOnefetchSmpCacheManager::nodeRequest key %llu added %d to req list\n", CkMyPe(), this->thisgroup.idx, key, requestorPe);
    entry->requestorVec.push_back(CkNodeCacheRequestorData(requestorPe));
  }
  else{
    // entry has data; return via msg to requestor
    CkAssert(entry->requestorVec.size() == 0);
    //CK_CACHE_VERBOSE("[%d] Cache %d: CkOnefetchSmpCacheManager::nodeRequest key %llu data found!! send to %d\n", CkMyPe(), this->thisgroup.idx, key, requestorPe);
    this->smpProxy()[requestorPe].nodeReply(key, CkCachePointerContainer<void>(entry->data));
  }
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::recvData(CkCacheFillMsg<CkCacheKey> *msg){
  CkAbort("Shouldn't call CkCacheManagerBase::recvData; use subclass instead.\n");
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data){
  CkAbort("Shouldn't call CkCacheManagerBase::recvData(local); use subclass instead.\n");
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::recvData(CkCacheFillMsg<CkCacheKey> *msg){
  CkCacheKey key = msg->key;
  CK_CACHE_VERBOSE("[%d] CkCacheManager::recvData %llu\n", CkMyPe(), key);

  typename std::map<CkCacheKey, int>::iterator ichunk = this->getKeyToChunk().find(key); 
  CkAssert(ichunk != this->getKeyToChunk().end());
  int chunk = ichunk->second;
  this->getKeyToChunk().erase(ichunk);

  CkAssert(chunk >= 0 && chunk < this->getNumChunks());

  // get pe-level cache entry
  CkPeCacheEntry<CkCacheKey> *entry = this->findPeCacheEntry(key, chunk);

  // unpack and size
  void *data = entry->type->unpack(msg, chunk, entry->home);

  this->moreStoredData(entry->type->size(data));
  this->peDeliverData(entry, data, chunk);
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data){
  CkPeCacheEntry<CkCacheKey> *& entry = this->getPeCache(chunk)[key];
  if (entry == NULL) {
    entry = new CkPeCacheEntry<CkCacheKey>(key, from, type);
  } else {
    this->lessStoredData(entry->type->size(entry->data));
    entry->type->writeback(entry->home, entry->key, entry->data);
  }
  this->peDeliverData(entry, data, chunk);
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::recvData(CkCacheFillMsg<CkCacheKey> *msg){
  CkCacheKey key = msg->key;
  CK_CACHE_VERBOSE("[%d] CkSmpCacheManager::recvData %llu\n", CkMyPe(), key);

  typename std::map<CkCacheKey, int>::iterator ichunk = this->getNodeKeyToChunk().find(key); 
  CkAssert(ichunk != this->getNodeKeyToChunk().end());
  int chunk = ichunk->second;
  this->getNodeKeyToChunk().erase(ichunk);

  CkAssert(chunk >= 0 && chunk < this->getNumChunks());

  // unpack and size
  CkNodeCacheEntry<CkCacheKey> *entry = this->findNodeCacheEntry(key, chunk);
  void *data = entry->type->unpack(msg, chunk, entry->home);

  // different behavior for Onefetch and Multifetch
  this->nodeDeliverData(entry, chunk, data);
  this->moreStoredData(entry->type->size(data));
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType<CkCacheKey> *type, int chunk, void *data){
  CK_CACHE_VERBOSE("[%d] CkSmpCacheManager::recvData[local] %llu\n", CkMyPe(), key);
  CkNodeCacheEntry<CkCacheKey> *entry = getNodeCache()->get(chunk, key); 
  if(entry == NULL){
    entry = new CkNodeCacheEntry<CkCacheKey>(key, from, type);
    getNodeCache()->put(chunk, key, entry);
  }

  nodeDeliverData(entry, chunk, data);
}

template<typename CkCacheKey>
int &CkCacheManagerBase<CkCacheKey>::getStoredData(){
  return storedData;
}

template<typename CkCacheKey>
int &CkCacheManagerBase<CkCacheKey>::getChunkAck(int chunk){
  return chunkAck[chunk];
}

template<typename CkCacheKey>
int &CkCacheManagerBase<CkCacheKey>::getChunkAckWB(int chunk){
  return chunkAckWB[chunk];
}

template<typename CkCacheKey>
std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *> &CkCacheManagerBase<CkCacheKey>::getPeCache(int chunk){
  return peCache[chunk];
}

template<typename CkCacheKey>
int CkCacheManagerBase<CkCacheKey>::getNumChunks() const {
  return numChunks;
}

template<typename CkCacheKey>
int CkCacheManagerBase<CkCacheKey>::getFinishedChunks() const {
  return finishedChunks;
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::moreStoredData(int nBytes){
  storedData += nBytes;
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::lessStoredData(int nBytes){
  storedData -= nBytes;
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::nodeDeliverData(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data){
  CkAbort("Shouldn't invoke CkSmpCacheManager::nodeDeliverData; use subclass instead.\n");
}

template<typename CkCacheKey>
void CkOnefetchSmpCacheManager<CkCacheKey>::nodeDeliverData(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data){
  entry->data = data;
  this->sendNodeReplies(entry, chunk, data);
}

template<typename CkCacheKey>
void CkMultifetchSmpCacheManager<CkCacheKey>::nodeDeliverData(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data){

  // XXX - instead of locking this whole function, inside the function
  // lock and copy the requestor list, unlock and then send messages from
  // copied list; this will shorten the critical section duration
  entry->lock();
  entry->data = data;
  this->sendNodeReplies(entry, chunk, data);
  entry->unlock();
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::sendNodeReplies(CkNodeCacheEntry<CkCacheKey> *entry, int chunk, void *data){
  for(int i = 0; i < entry->requestorVec.size(); i++){
    int toPe = entry->requestorVec[i].pe;
    CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::sendNodeReplies key %llu send to %d\n", CkMyPe(), this->thisgroup.idx, entry->key, toPe);
    smpProxy()[toPe].nodeReply(entry->key, CkCachePointerContainer<void>(data));
    /*
    if(toPe != CkMyPe()){
    }
    else{
      CkPeCacheEntry *pentry = findPeCacheEntry(entry->key, chunk); 
      peDeliverData(pentry, data, chunk);
    }
    */
  }

  entry->requestorVec.resize(0);
}

template<typename CkCacheKey>
std::map<CkCacheKey,int> &CkCacheManagerBase<CkCacheKey>::getKeyToChunk(){
  return keyToChunk;
}

template<typename CkCacheKey>
std::map<CkCacheKey,int> &CkSmpCacheManager<CkCacheKey>::getNodeKeyToChunk(){
  return nodeKeyToChunk_;
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::printKeyToChunk() const {
  std::ostringstream oss;
  typename std::map<CkCacheKey,int>::const_iterator it;
  for(it = keyToChunk.begin(); it != keyToChunk.end(); ++it){
    oss << "(" << it->first << "," << it->second << "), ";
  }
  CkPrintf("[%d] printKeyToChunk: %s\n", CkMyPe(), oss.str().c_str());
};

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::nodeReply(CkCacheKey key, const CkCachePointerContainer<void> &dataPtrContainer){
  CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::nodeReply recvd key %llu\n", CkMyPe(), this->thisgroup.idx, key);
  typename std::map<CkCacheKey, int>::iterator it;
  it = this->getKeyToChunk().find(key);
  CkAssert(it != this->getKeyToChunk().end());
  int chunk = it->second;
  this->getKeyToChunk().erase(it);

  CkPeCacheEntry<CkCacheKey> *pentry = this->findPeCacheEntry(key, chunk);
  this->peDeliverData(pentry, dataPtrContainer.pointer, chunk);
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::peDeliverData(CkPeCacheEntry<CkCacheKey> *entry, void *data, int chunk){
  CkAssert(this->getChunkAck(chunk) > 0);
  entry->data = data;
  // deliver to all requesting objects
  for(int i = 0; i < entry->requestorVec.size(); i++){
    entry->requestorVec[i].deliver(entry->key, data, chunk);
  }
  entry->requestorVec.resize(0);
}

template<typename CkCacheKey>
std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *> *
CkCacheManagerBase<CkCacheKey>::getCache(){
  return peCache;
}

template<typename CkCacheKey>
void *
CkCacheManagerBase<CkCacheKey>::requestDataNoFetch(CkCacheKey key, int chunk) {
  CkAbort("Shouldn't call CkCacheManagerBase::requestDataNoFetch; only subclasses\n");
}

template<typename CkCacheKey>
void *
CkCacheManager<CkCacheKey>::requestDataNoFetch(CkCacheKey key, int chunk) {
  typename std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *>::iterator it;
  it = this->getPeCache(chunk).find(key);
  if (it != this->getPeCache(chunk).end()) {
    return it->second->data;
  }
  return NULL;
}
 
template<typename CkCacheKey>
void *
CkSmpCacheManager<CkCacheKey>::requestDataNoFetch(CkCacheKey key, int chunk) {
  CkNodeCacheEntry<CkCacheKey> *nodeEntry = this->getNodeCache()->get(chunk, key);
  if(nodeEntry != NULL && nodeEntry->data != NULL){
    return nodeEntry->data;
  }
  return NULL;
}
 
template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::cacheSync(int &newNumChunks, CkArrayIndex &chareIdx, int &localIdx) {
  // num. finished chunks should have been reset
  // in pstClearPeCacheCheck
  CkAssert(finishedChunks == 0);

  if (syncdChares > 0) {
    newNumChunks = numChunks;
    CK_CACHE_VERBOSE("[%d] Cache %d: sync noalloc\n", CkMyPe(), this->thisgroup.idx);
  } else {
    syncdChares = 1;
    CK_CACHE_VERBOSE("[%d] Cache %d: sync alloc\n", CkMyPe(), this->thisgroup.idx);

    countObjects();

    allocatePeCache(newNumChunks);

#if COSMO_STATS > 0
    CmiResetMaxMemory();
#endif
  }

  localIdx = localChares.registered.get(chareIdx);
  CkAssert(localIdx != 0);

}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::allocatePeCache(int newNumChunks){
  for (int chunk=0; chunk<numChunks; ++chunk) {
    CkAssert(peCache[chunk].empty());
    CkAssert(chunkAck[chunk]==0);
    CkAssert(chunkAckWB[chunk]==0);
  }
  CkAssert(this->getKeyToChunk().empty());
  storedData = 0;

#if COSMO_STATS > 0
  dataArrived = 0;
  dataTotalArrived = 0;
  dataMisses = 0;
  dataLocal = 0;
  totalDataRequested = 0;
  maxData = 0;
#endif

  if (numChunks != newNumChunks) {
    if(numChunks != 0) {
      delete[] peCache;
      delete[] chunkAck;
      delete[] chunkAckWB;
      delete[] chunkWeight;
    }

    numChunks = newNumChunks;
    peCache = new std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *>[numChunks];
    chunkAck = new int[numChunks];
    chunkAckWB = new int[numChunks];
    chunkWeight = new CmiUInt8[numChunks];
  }
  for (int i=0; i<numChunks; ++i) {
    chunkAck[i] = localChares.count;
    chunkAckWB[i] = localCharesWB.count;
    chunkWeight[i] = 0;
  }

  allocateNodeCache(numChunks);
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::allocateNodeCache(int newNumChunks){
  this->ready() = true;
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::allocateNodeCache(int newNumChunks){
  if(isLeader() && okToAllocateNodeCache()){
    CK_CACHE_VERBOSE("[%d] CkCache: %d CkSmpCacheManager::allocateNodeCache\n", CkMyPe(), this->thisgroup.idx);
    if(this->getNodeCache() == NULL){
      this->getNodeCache() = new SmpCache<CkCacheKey>;
    }
    this->getNodeCache()->alloc(newNumChunks, CkMyNode());
    // If the leader is not already finished with all the chunks, only then set
    // ready to true otherwise let it remain false
    if (!this->finishedAll()) {
      this->ready() = true;
    }
    // send a message to all pes on node
    coordinator()->mcast(CkCachePointerContainer<SmpCache<CkCacheKey> >(this->getNodeCache()));
    releaseBufferedRequests();
  }

  okToAllocateNodeCache() = false;
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::mcast(CkCachePointerContainer<SmpCache<CkCacheKey> > &container){
  CK_CACHE_VERBOSE("[%d] CkCache: %d CkSmpCacheManager::recvd node-cache pointer\n", CkMyPe(), this->thisgroup.idx);
  // If all the chunks have been finished, don't set ready() to true because
  // then the next iteration might think that the cache is ready and will start
  // sending requests to the node cache.
  // Since all the chunks are done and the cache was not ready, sync was not
  // sent. Now is the time to send the sync as the leader is waiting for it to
  // deallocate the cache. 
  if (this->finishedAll()) {
    coordinator()->sync(CkCallback(CkIndex_CkSmpCacheManager<CkCacheKey>::peFinishedChunkDone(),
          CkMyPe(), this->thisProxy));
    return;
  }
  if(isLeader()){
    // we cannot assert that the leader will be ready here: imagine that 
    // the leader's objects make no requests for remote data, so that 
    // right after cacheSync(), they call finishedChunk() and the leader, 
    // having allocated a node cache and sent it through mcast() will 
    // later receive the message here, but will already have set rdy to false
    // in finishedChunk.
    //CkAssert(ready());
    
  } else{
    this->getNodeCache() = container.pointer;
    this->ready() = true;
    releaseBufferedRequests();
  }
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::writebackChunk(int chunk) {
  CkAssert(this->getChunkAckWB(chunk) > 0);
  if (--this->getChunkAckWB(chunk) == 0) {
    // we can safely write back the chunk to the senders
    // at this point no more changes to the data can be made until next fetch

    typename std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *>::iterator iter;
    for (iter = this->getPeCache(chunk).begin(); iter != this->getPeCache(chunk).end(); iter++) {
      CkPeCacheEntry<CkCacheKey> *e = iter->second;
      e->writeback();
    }

  }
}

  // XXX - for now, we don't allow writebacks with SmpCaches, since
  // we don't have an efficient scheme for ensuring write/accum. 
  // access mutex
template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::writebackChunk(int chunk){
  CkAbort("CkSmpCacheManager::writebackChunk: use CkNonSmpCache, not CkSmpCacheManager for writeback capability\n");
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::finishedChunk(int chunk, CmiUInt8 weight) {
  CkAssert(chunkAck[chunk] > 0);
  chunkWeight[chunk] += weight;
  --chunkAck[chunk];
  CK_CACHE_VERBOSE("[%d] Cache %d: finishedChunk %d chunkAck %d\n", CkMyPe(), this->thisgroup.idx, chunk, chunkAck[chunk]);
  if (chunkAck[chunk] == 0) {
    peFinishedChunk(chunk, chunkWeight[chunk]);
  }
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::peFinishedChunk(int chunk, CmiUInt8 weight){
  // we can safely delete the chunk from the cache

  // TODO: if chunks are held back due to restrictions, here is a
  // good position to release them

#if 0
  if (maxData < storedData) maxData = storedData;
#endif

  CK_CACHE_VERBOSE("[%d] Cache %d: CkCacheManager::peFinishedChunk %d\n", CkMyPe(), this->thisgroup.idx, chunk);
  this->clearPeCacheChunk(chunk);

  this->postClearPeCacheChunkCheck();
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::postClearPeCacheChunkCheck(){
  finishedChunks++;
  if (finishedChunks == numChunks) {
    finishedChunks = 0;
    syncdChares = 0;
    this->ready() = false;

    // FIXME - the following might happen: tree pieces don't request any remote
    // nodes, and call finishedChunk right away, causing this PE to make do intra-node
    // sync(). meanwhile, leader might have finished serving its clients and could be
    // waiting for intra-node sync() from other PEs. However, this PE didn't wait for
    // the leader's node-level cache pointer from mcast() before doing the sync(), so 
    // it could receive the pointer now. If this PE's tree pieces now ask for remote 
    // data, it will believe it has the pointer for the new iteration, and forward
    // the requests to the leader; this will cause pending deliveries to be added to
    // the node-level cache. Now, the leader will try to deallocate the chunk and will
    // check the entries in the chunk's cache, and will fail because there are pending
    // deliveries. 

    // FIXME - have to somehow ensure that node-level cache pointer is received before
    // sync() on this PE, and that if the TreePieces have already called finishedChunk
    // by the time the cache pointer is received, we do the sync() without waiting for
    // the tree pieces
    

    deallocateNodeCache();
  }
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::deallocateNodeCache(){
  if(isLeader()){
    CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::okToDeallocateNodeCache\n", CkMyPe(), this->thisgroup.idx);
    okToDeallocateNodeCache() = true;
  }
}

template<typename CkCacheKey>
bool CkSmpCacheManager<CkCacheKey>::isLeader() const {
  return coordinator_->isLeader();
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::peFinishedChunk(int chunk, CmiUInt8 weight){
  CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::peFinishedChunk %d chunkAck %d\n", CkMyPe(), this->thisgroup.idx, chunk, this->getChunkAck(chunk));
  this->clearPeCacheChunk(chunk);
  if(isLeader()){
    chunksToDelete_.push(chunk);
  }

  this->finishedAll() = true;

  // Only when this PE is ready, should we participate in the sync to say all
  // the chunks are done. It can so happen that the leader announcement comes
  // after this.
  if (this->ready()) {
    // synchronize all cores on SMP node, so that before freeing the 
    // data associated with this chunk, the leader knows that all
    // objects on all cores of the node are finished with it.

    // XXX have to modify CkSmpCoordinator to accept some user data to reduce 
    coordinator()->sync(CkCallback(CkIndex_CkSmpCacheManager<CkCacheKey>::peFinishedChunkDone(),
      CkMyPe(), this->thisProxy));
  }

  // when all chunks have been finished, this will render 
  // the cache 'not ready', so that objects cannot use it
  // until it has been reinitialized and this core has a 
  // pointer to it
  this->postClearPeCacheChunkCheck();
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::peFinishedChunkDone(){
  CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::peFinishedChunkDone\n", CkMyPe(), this->thisgroup.idx);
  // here, all cores with objects on them (these are the ones that we care about)
  // have finished using a particular chunk; 
  // only the leader from the previous round should actually have
  // chunks to delete. this is because we only enqueue chunks to delete
  // for the leader in the current iteration. 
  while(!chunksToDelete_.empty()){
    int chunk = chunksToDelete_.front();
    clearNodeCacheChunk(chunk);
    chunksToDelete_.pop();
  }

  if(okToDeallocateNodeCache()){
    CkAssert(this->getNodeCache()->checkClear());
    // we assume that no one else needs to know that
    // it is now ok to allocate a new node-level cache
    // since the leader will not have changed between
    // iterations. 
    // (1) dealloc of previous iteration's node cache
    // and (2) cacheSync() leading to alloc of new
    // node cache can happen in any order; 
    okToAllocateNodeCache() = true;
    if(this->getSyncdChares() > 0){
      // if we have received at least one cacheSync(),
      // we will have the right number of chunks for this
      // new iteration
      allocateNodeCache(this->getNumChunks());
    }

    okToDeallocateNodeCache() = false;
    // we might have stalled registration until 
    // the last sync() has been completed.
    if(pendingRegistration()){
      doneRegistrationBody();
    }
  }

  this->finishedAll() = false;


  // FIXME - furthermore, we shouldn't have both a 
  // registration and a node-cache alloc request (i.e. getSyncdChares() > 0)
  // waiting here;

  // A few notes on this "strange" way of tracking chunks and caches to delete

  // Since there is little synchronization between objects and the cache, the
  // objects on this pe might move on to the next iteration (by calling
  // cacheSync()) having called finishedChunk() 'numChunks' number of times
  // previously. Denote the calling of finishedChunk() 'numChunks' times,
  // [finishedChunk]+.

  // As objects move to the next iteration, they needn't wait for the cache to
  // finish deleting the node-level SmpCache structure from the previous
  // iteration.  Therefore, we must explicitly track SmpCache's to delete.
  // Moreover, an arbitrary number of cacheSync-[finishedChunk]+ cycles might
  // occur after a [finishedChunk()]+ but before the cache is able to delete
  // the corresponding node-level SmpCache. Therefore, we need a list of
  // pointers; a single pointer will not suffice.

  // Furthermore, even though we check for isLeader() when storing
  // chunks/caches to delete (before sync()ing) we cannot check for the same
  // condition while actually deleting (after sync()ing), since the leader
  // might have changed in the interim.

  // Finally, we only need to delete one chunk, and possibly one cache per
  // invocation of this synchronization reduction target method, since (i) we
  // will have called sync() as many times as there are chunks to be deleted,
  // and (ii), a cache is only deleted when all its chunks are deleted, and
  // specifically, is deleted when its final chunk is deleted.
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::clearPeCacheChunk(int chunk){
  CK_CACHE_VERBOSE("[%d] Cache %d: CkCacheManagerBase::clearPeCacheChunk %d\n", CkMyPe(), this->thisgroup.idx, chunk);
  typename std::map<CkCacheKey, CkPeCacheEntry<CkCacheKey> *>::iterator it;
  for (it = peCache[chunk].begin(); it != peCache[chunk].end(); it++) {
    CkPeCacheEntry<CkCacheKey> *e = it->second;
    lessStoredData(e->type->size(e->data));

    // TODO: Store communication pattern here

    freePeEntry(e);
    delete e;
  }
  peCache[chunk].clear();
}

template<typename CkCacheKey>
void CkCacheManager<CkCacheKey>::freePeEntry(CkPeCacheEntry<CkCacheKey> *e){
  // since each pe has a separate copy of the data, 
  // it is ok to free it/write back here.
  e->free();
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::freePeEntry(CkPeCacheEntry<CkCacheKey> *e){
  // cannot free pe-level entry's data, since other pes on this node might
  // still be using it; only when the node level cache is being cleared are we
  // allowed to do this.  the thing to note is that we don't copy data from the
  // node-level cache to the pe-level cache, but only copy a pointer; so to
  // delete the message associated with the data here (see
  // CkCacheEntryType::free) will be an error
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::clearNodeCacheChunk(int chunk){
  CK_CACHE_VERBOSE("[%d] Cache %d: CkSmpCacheManager::clearNodeCacheChunk chunk %d\n", CkMyPe(), this->thisgroup.idx, chunk);
  this->getNodeCache()->freeChunk(chunk);
}
  
template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::collectStatistics(CkCallback &cb) {
#if COSMO_STATS > 0
  CkCacheStatistics cs(dataArrived, dataTotalArrived,
      dataMisses, dataLocal, dataError, totalDataRequested,
      maxData, CkMyPe());
  contribute(sizeof(CkCacheStatistics), &cs, CkCacheStatistics::sum, cb);
#else
  CkAbort("Invalid call, only valid if COSMO_STATS is defined");
#endif
}

// ctors

template<typename CkCacheKey>
CkCacheManagerBase<CkCacheKey>::CkCacheManagerBase(CkMigrateMessage *m) :
  CBase_CkCacheManagerBase<CkCacheKey> (m)
{
  init();
}

template<typename CkCacheKey>
CkCacheManager<CkCacheKey>::CkCacheManager(CkMigrateMessage *m) : 
  CBase_CkCacheManager<CkCacheKey>(m)
{}


template<typename CkCacheKey>
CkCacheManager<CkCacheKey>::CkCacheManager(int size, CkGroupID gid) : 
  CBase_CkCacheManager<CkCacheKey>(size, gid)
{}

template<typename CkCacheKey>
CkCacheManager<CkCacheKey>::CkCacheManager(int size, int n, CkGroupID *gid) : 
  CBase_CkCacheManager<CkCacheKey>(size, n, gid)
{}

template<typename CkCacheKey>
CkCacheManager<CkCacheKey>::CkCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB) : 
  CBase_CkCacheManager<CkCacheKey>(size, n, gid, nWB, gidWB)
{}



template<typename CkCacheKey>
CkSmpCacheManager<CkCacheKey>::CkSmpCacheManager(CkMigrateMessage *m) : 
  CBase_CkSmpCacheManager<CkCacheKey>(m)
{
  smpInit();
}

template<typename CkCacheKey>
CkSmpCacheManager<CkCacheKey>::CkSmpCacheManager(int size, CkGroupID gid) : 
  CBase_CkSmpCacheManager<CkCacheKey>(size, gid)
{
  smpInit();
}

template<typename CkCacheKey>
CkSmpCacheManager<CkCacheKey>::CkSmpCacheManager(int size, int n, CkGroupID *gid) : 
  CBase_CkSmpCacheManager<CkCacheKey>(size, n, gid)
{
  smpInit();
}

template<typename CkCacheKey>
CkSmpCacheManager<CkCacheKey>::CkSmpCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB) : 
  CBase_CkSmpCacheManager<CkCacheKey>(size, n, gid, nWB, gidWB)
{
  smpInit();
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::smpInit(){
  myProxy_ = CProxy_CkSmpCacheManager<CkCacheKey>(this->thisProxy);
  this->getNodeCache() = NULL;
  okToAllocateNodeCache() = true;
  okToDeallocateNodeCache() = false;
  pendingRegistration() = false;
}

template<typename CkCacheKey>
CProxy_CkSmpCacheManager<CkCacheKey> &CkSmpCacheManager<CkCacheKey>::smpProxy(){
  return myProxy_;
}

template<typename CkCacheKey>
CkOnefetchSmpCacheManager<CkCacheKey>::CkOnefetchSmpCacheManager(CkMigrateMessage *m) : 
  CBase_CkOnefetchSmpCacheManager<CkCacheKey>(m)
{
}

template<typename CkCacheKey>
CkOnefetchSmpCacheManager<CkCacheKey>::CkOnefetchSmpCacheManager(int size, CkGroupID gid) : 
  CBase_CkOnefetchSmpCacheManager<CkCacheKey>(size, gid)
{}

template<typename CkCacheKey>
CkOnefetchSmpCacheManager<CkCacheKey>::CkOnefetchSmpCacheManager(int size, int n, CkGroupID *gid) : 
  CBase_CkOnefetchSmpCacheManager<CkCacheKey>(size, n, gid)
{}

template<typename CkCacheKey>
CkOnefetchSmpCacheManager<CkCacheKey>::CkOnefetchSmpCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB) : 
  CBase_CkOnefetchSmpCacheManager<CkCacheKey>(size, n, gid, nWB, gidWB)
{}

template<typename CkCacheKey>
CkMultifetchSmpCacheManager<CkCacheKey>::CkMultifetchSmpCacheManager(CkMigrateMessage *m) : 
  CBase_CkMultifetchSmpCacheManager<CkCacheKey>(m)
{
}

template<typename CkCacheKey>
CkMultifetchSmpCacheManager<CkCacheKey>::CkMultifetchSmpCacheManager(int size, CkGroupID gid) : 
  CBase_CkMultifetchSmpCacheManager<CkCacheKey>(size, gid)
{}

template<typename CkCacheKey>
CkMultifetchSmpCacheManager<CkCacheKey>::CkMultifetchSmpCacheManager(int size, int n, CkGroupID *gid) : 
  CBase_CkMultifetchSmpCacheManager<CkCacheKey>(size, n, gid)
{}

template<typename CkCacheKey>
CkMultifetchSmpCacheManager<CkCacheKey>::CkMultifetchSmpCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB) : 
  CBase_CkMultifetchSmpCacheManager<CkCacheKey>(size, n, gid, nWB, gidWB)
{}

template<typename CkCacheKey>
int CkCacheManagerBase<CkCacheKey>::getLocalChareCount() const {
  return localChares.count;
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::setup(CkSmpCacheHandle<CkCacheKey> &handle, const CkCallback &cb){
  coordinatorProxy_ = handle.getCoordinatorProxy();
  coordinator() = coordinatorProxy_[CkMyPe()].ckLocal();
  CkAssert(coordinator() != NULL);
  coordinator()->setup(cb);
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::cleanupForCheckpoint(const CkCallback &cb){
  // Delete the local element of the coordinator array
  coordinator()->ckDestroy();
  this->contribute(cb);
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::registration(const CkCallback &cb){
  this->countObjects();
  int nObjects = this->getLocalChareCount();
  callback_ = cb;
  pendingRegistration() = true;
  // Need to set ready() to false here because, this is the only place to reset
  // ready when there are no objects on a PE. Otherwise the PE will have the old
  // state.
  // If a PE has no objects, it participates in the registration but not in
  // cacheSync and finishedChunks. Since it doesn't participate in
  // finishedChunks, ready() is never set to false and is set to true when the
  // node cache pointer mcast is received. In the next iteration, since ready()
  // was never reset to false in finishedChunks, it assumes that the node cache
  // is ready and sends request to it leading to crash. So reset ready() to
  // false at registration.
  this->ready() = false;
  coordinator()->registration(this, nObjects, 
                            CkCallback(CkIndex_CkSmpCacheManager<CkCacheKey>::doneRegistration(), 
                                CkMyPe(), smpProxy()));
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::doneRegistration(){
  // if there is a pending delete of the node cache,
  // we should hold off until it is done, before
  // returning control to user.
  if(!okToDeallocateNodeCache()){
    doneRegistrationBody();
  }
}

template<typename CkCacheKey>
void CkSmpCacheManager<CkCacheKey>::doneRegistrationBody(){
  // even if this PE was not the leader in the
  // last set of pre-registration iterations, it
  // is now allowed to allocate node cache; it will
  // do so only if it becomes the leader this time
  // around
  // this assertion is equivalent to saying that we
  // don't have an outstanding delete operation for the 
  // node cache
  CkAssert(!okToDeallocateNodeCache());

  okToAllocateNodeCache() = true;
  okToDeallocateNodeCache() = false;
  pendingRegistration() = false;
  this->contribute(callback_);
}

template<typename CkCacheKey>
void CkCacheManagerBase<CkCacheKey>::countObjects(){
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
}

// CkCacheFactory methods
template<typename CkCacheKey>
CkNonSmpCacheHandle<CkCacheKey> CkNonSmpCacheFactory<CkCacheKey>::instantiate(int size, CkGroupID gid){
  CkNonSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkCacheManager<CkCacheKey>::ckNew(size, gid);
  return handle;
}

template<typename CkCacheKey>
CkNonSmpCacheHandle<CkCacheKey> CkNonSmpCacheFactory<CkCacheKey>::instantiate(int size, int n, CkGroupID *gid){
  CkNonSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkCacheManager<CkCacheKey>::ckNew(size, n, gid);
  return handle;
}

template<typename CkCacheKey>
CkNonSmpCacheHandle<CkCacheKey> CkNonSmpCacheFactory<CkCacheKey>::instantiate(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB){
  CkNonSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkCacheManager<CkCacheKey>::ckNew(size, n, gid, nWB, gidWB);
  return handle;
}

template<typename CkCacheKey>
CkSmpCacheHandle<CkCacheKey> CkOnefetchSmpCacheFactory<CkCacheKey>::instantiate(int size, CkGroupID gid){
  CkSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkOnefetchSmpCacheManager<CkCacheKey>::ckNew(size, gid);
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
  return handle;
}

template<typename CkCacheKey>
CkSmpCacheHandle<CkCacheKey> CkOnefetchSmpCacheFactory<CkCacheKey>::instantiate(int size, int n, CkGroupID *gid){
  CkSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkOnefetchSmpCacheManager<CkCacheKey>::ckNew(size, n, gid);
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
  return handle;
}

template<typename CkCacheKey>
CkSmpCacheHandle<CkCacheKey> CkOnefetchSmpCacheFactory<CkCacheKey>::instantiate(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB){
  CkSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkOnefetchSmpCacheManager<CkCacheKey>::ckNew(size, n, gid, nWB, gidWB);
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
  return handle;
}

template<typename CkCacheKey>
CkSmpCacheHandle<CkCacheKey> CkMultifetchSmpCacheFactory<CkCacheKey>::instantiate(int size, CkGroupID gid){
  CkSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkMultifetchSmpCacheManager<CkCacheKey>::ckNew(size, gid);
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
  return handle;
}

template<typename CkCacheKey>
CkSmpCacheHandle<CkCacheKey> CkMultifetchSmpCacheFactory<CkCacheKey>::instantiate(int size, int n, CkGroupID *gid){
  CkSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkMultifetchSmpCacheManager<CkCacheKey>::ckNew(size, n, gid);
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
  return handle;
}

template<typename CkCacheKey>
CkSmpCacheHandle<CkCacheKey> CkMultifetchSmpCacheFactory<CkCacheKey>::instantiate(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB){
  CkSmpCacheHandle<CkCacheKey> handle;
  handle.cacheProxy = CProxy_CkMultifetchSmpCacheManager<CkCacheKey>::ckNew(size, n, gid, nWB, gidWB);
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
  return handle;
}

template<typename CkCacheKey>
void CkOnefetchSmpCacheFactory<CkCacheKey>::instantiateCoordinator(CkSmpCacheHandle<CkCacheKey> &handle){
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
}

template<typename CkCacheKey>
void CkMultifetchSmpCacheFactory<CkCacheKey>::instantiateCoordinator(CkSmpCacheHandle<CkCacheKey> &handle){
  handle.mcastGrpId = CProxy_CkMulticastMgr::ckNew();
  handle.coordinatorProxy = CProxy_CkSmpCoordinator<CkSmpCacheManager<CkCacheKey> >::ckNew(handle.mcastGrpId, CkNumPes());
}

#define CK_TEMPLATES_ONLY
#include "CkCache.def.h"
#undef CK_TEMPLATES_ONLY

#endif
