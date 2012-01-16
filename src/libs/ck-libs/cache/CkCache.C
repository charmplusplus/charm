#include "CkCache.h"

  CkCacheManager::CkCacheManager(int size, CkGroupID gid) {
    init();
    numLocMgr = 1;
    numLocMgrWB = 0;
    locMgr = new CkGroupID[1];
    locMgr[0] = gid;
    maxSize = (CmiUInt8)size * 1024 * 1024;
  }

  CkCacheManager::CkCacheManager(int size, int n, CkGroupID *gid) {
    init();
    numLocMgr = n;
    numLocMgrWB = 0;
    locMgr = new CkGroupID[n];
    for (int i=0; i<n; ++i) locMgr[i] = gid[i];
    maxSize = (CmiUInt8)size * 1024 * 1024;
  }

  CkCacheManager::CkCacheManager(int size, int n, CkGroupID *gid, int nWB, CkGroupID *gidWB) {
    init();
    numLocMgr = n;
    locMgr = new CkGroupID[n];
    for (int i=0; i<n; ++i) locMgr[i] = gid[i];
    numLocMgrWB = nWB;
    locMgrWB = new CkGroupID[nWB];
    for (int i=0; i<n; ++i) locMgrWB[i] = gidWB[i];
    maxSize = (CmiUInt8)size * 1024 * 1024;
  }

  CkCacheManager::CkCacheManager(CkMigrateMessage* m) : CBase_CkCacheManager(m) {
    init();
  }

  void CkCacheManager::init() {
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

  void CkCacheManager::pup(PUP::er &p) {
    CBase_CkCacheManager::pup(p);
    p | numLocMgr;
    if (p.isUnpacking()) locMgr = new CkGroupID[numLocMgr];
    PUParray(p,locMgr,numLocMgr);
    p | numLocMgrWB;
    if (p.isUnpacking()) locMgrWB = new CkGroupID[numLocMgrWB];
    PUParray(p,locMgrWB,numLocMgrWB);
    p | maxSize;
  }

  void * CkCacheManager::requestData(CkCacheKey what, CkArrayIndex &_toWhom, int chunk, CkCacheEntryType *type, CkCacheRequestorData &req){

    std::map<CkCacheKey,CkCacheEntry *>::iterator p;
    CkArrayIndex toWhom(_toWhom);
    CkAssert(chunkAck[chunk] > 0);
    p = cacheTable[chunk].find(what);
    CkCacheEntry *e;
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
      e = new CkCacheEntry(what, toWhom, type);
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

  void * CkCacheManager::requestDataNoFetch(CkCacheKey key, int chunk) {
    std::map<CkCacheKey,CkCacheEntry *>::iterator p = cacheTable[chunk].find(key);
    if (p != cacheTable[chunk].end()) {
      return p->second->data;
    }
    return NULL;
  }
  
  CkCacheEntry * CkCacheManager::requestCacheEntryNoFetch(CkCacheKey key, int chunk) {
    std::map<CkCacheKey,CkCacheEntry *>::iterator p = cacheTable[chunk].find(key);
    if (p != cacheTable[chunk].end()) {
      return p->second;
    }
    return NULL;
  }
  
  std::map<CkCacheKey,CkCacheEntry*> *CkCacheManager::getCache(){
    return cacheTable;
  }

  void CkCacheManager::recvData(CkCacheFillMsg *msg) {
    CkCacheKey key = msg->key;
    std::map<CkCacheKey,int>::iterator pchunk = outStandingRequests.find(key);
    CkAssert(pchunk != outStandingRequests.end());
    int chunk = pchunk->second;
    CkAssert(chunk >= 0 && chunk < numChunks);
    CkAssert(chunkAck[chunk] > 0);
    outStandingRequests.erase(pchunk);
    
    std::map<CkCacheKey,CkCacheEntry*>::iterator p;
    p = cacheTable[chunk].find(key);
    CkAssert(p != cacheTable[chunk].end());
    CkCacheEntry *e = p->second;
    e->data = e->type->unpack(msg, chunk, e->home);
    storedData += e->type->size(e->data);
    
    std::vector<CkCacheRequestorData>::iterator caller;
    for (caller = e->requestorVec.begin(); caller != e->requestorVec.end(); caller++) {
      caller->deliver(key, e->data, chunk);
    }
    e->requestorVec.clear();
  }
  
  void CkCacheManager::recvData(CkCacheKey key, CkArrayIndex &from, CkCacheEntryType *type, int chunk, void *data) {
    std::map<CkCacheKey,CkCacheEntry*>::iterator p = cacheTable[chunk].find(key);
    CkCacheEntry *e;
    if (p == cacheTable[chunk].end()) {
      e = new CkCacheEntry(key, from, type);
      cacheTable[chunk][key] = e;
    } else {
      e = p->second;
      storedData -= e->type->size(e->data);
      e->type->writeback(e->home, e->key, e->data);
    }
    e->replyRecvd = true;
    e->data = data;
    storedData += e->type->size(data);
    
    std::vector<CkCacheRequestorData>::iterator caller;
    for (caller = e->requestorVec.begin(); caller != e->requestorVec.end(); caller++) {
      caller->deliver(key, e->data, chunk);
    }
    e->requestorVec.clear();
  }

  void CkCacheManager::cacheSync(int &_numChunks, CkArrayIndex &chareIdx, int &localIdx) {
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
        cacheTable = new std::map<CkCacheKey,CkCacheEntry*>[numChunks];
        chunkAck = new int[numChunks];
        chunkAckWB = new int[numChunks];
        chunkWeight = new CmiUInt8[numChunks];
      }
      for (int i=0; i<numChunks; ++i) {
        chunkAck[i] = localChares.count;
        chunkAckWB[i] = localCharesWB.count;
        chunkWeight[i] = 0;
      }
      
#if COSMO_STATS > 0
      CmiResetMaxMemory();
#endif
    }

    localIdx = localChares.registered.get(chareIdx);
    CkAssert(localIdx != 0);
  }

  void CkCacheManager::writebackChunk(int chunk) {
    CkAssert(chunkAckWB[chunk] > 0);
    if (--chunkAckWB[chunk] == 0) {
      // we can safely write back the chunk to the senders
      // at this point no more changes to the data can be made until next fetch

      std::map<CkCacheKey,CkCacheEntry*>::iterator iter;
      for (iter = cacheTable[chunk].begin(); iter != cacheTable[chunk].end(); iter++) {
        CkCacheEntry *e = iter->second;
        e->writeback();
      }

    }
  }

  void CkCacheManager::finishedChunk(int chunk, CmiUInt8 weight) {
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

      std::map<CkCacheKey,CkCacheEntry*>::iterator iter;
      for (iter = cacheTable[chunk].begin(); iter != cacheTable[chunk].end(); iter++) {
        CkCacheEntry *e = iter->second;
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
  
  CkReduction::reducerType CkCacheStatistics::sum;

  void CkCacheManager::collectStatistics(CkCallback &cb) {
#if COSMO_STATS > 0
    CkCacheStatistics cs(dataArrived, dataTotalArrived,
        dataMisses, dataLocal, dataError, totalDataRequested,
        maxData, CkMyPe());
    contribute(sizeof(CkCacheStatistics), &cs, CkCacheStatistics::sum, cb);
#else
    CkAbort("Invalid call, only valid if COSMO_STATS is defined");
#endif
  }

#include "CkCache.def.h"
