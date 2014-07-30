#ifndef CK_CACHE_SMP_CACHE_H 
#define CK_CACHE_SMP_CACHE_H 

#include "charm++.h"
#include "threadsafe_hashtable/hashtable_mt.h"
#include "CkCacheDataStructures.h"

template<typename CkCacheKey>
class SmpCache {
  
  static int hash6432shift(CkCacheKey key);

  typedef hashtable_mt<CkCacheKey, CkNodeCacheEntry<CkCacheKey> *, int (*)(CkCacheKey)> CkSmpHashtableType;


  // for debugging 
  int id; 
  /// hash table containing all the entries currently in the SMP node
  CkSmpHashtableType **tables;

  int numChunks;


  /// weights of the chunks in which the tree is divided, the cache will
  /// update the chunk division based on these values
  CmiUInt8 *chunkWeight;
  CmiUInt8 maxSize;

  public:
  SmpCache();
  void alloc(int nchunks, int id);
  size_t size();
  void setMaxSize(CmiUInt8 size);

  CkNodeCacheEntry<CkCacheKey> *get(int chunk, CkCacheKey key);
  void put(int chunk, CkCacheKey key, CkNodeCacheEntry<CkCacheKey> *fetcherEntry);
  CkNodeCacheEntry<CkCacheKey> *single_put(int chunk, CkCacheKey key, CkNodeCacheEntry<CkCacheKey> *entry);

  void writebackChunk(int chunk);
  void freeChunk(int chunk);
  bool checkClear();
  void print(int chunk);

  private:
  static CmiUInt8 InvalidKey;
  static CkNodeCacheEntry<CkCacheKey> *InvalidValue;

};

#include "CkCacheDataStructures.h"
#include "SmpCache.h"
#include "CkCache.h"
#include <sstream>

#define SMP_CACHE_VERBOSE /* CkPrintf */

template<typename CkCacheKey>
SmpCache<CkCacheKey>::SmpCache(){
  chunkWeight = NULL;
  tables = NULL;
  numChunks = 0;
  id = -1;
}

template<typename CkCacheKey>
void SmpCache<CkCacheKey>::alloc(int nchunks, int id){
  this->id = id;
  for(int i = 0; i < numChunks; i++){
    // should have deleted all entries in the cache
    CkAssert(tables[i]->size() == 0);
  }

  if(numChunks != nchunks){
    for(int i = 0; i < numChunks; i++){
      delete tables[i];
    }

    if(tables != NULL) delete[] tables;
    if(chunkWeight != NULL) delete[] chunkWeight;

    numChunks = nchunks;
    tables = new CkSmpHashtableType *[numChunks];

    for(int i = 0; i < numChunks; i++){
      tables[i] = new  
                  CkSmpHashtableType(0, 
                              hash6432shift,
                              SmpCache::InvalidKey,
                              SmpCache::InvalidValue);
    }

    chunkWeight = new CmiUInt8[numChunks];
  }

  for(int i = 0; i < numChunks; i++){
    chunkWeight[i] = 0;
  }
}

template<typename CkCacheKey>
size_t SmpCache<CkCacheKey>::size(){
  size_t sz = 0;
  for(int i = 0; i < numChunks; i++){
    sz += tables[i]->size();
  }
  return sz;
}

template<typename CkCacheKey>
void SmpCache<CkCacheKey>::setMaxSize(CmiUInt8 size){
  // XXX this is not used currently
  maxSize = size;
}

template<typename CkCacheKey>
bool SmpCache<CkCacheKey>::checkClear(){
  bool ret = true;
  for(int i = 0; i < numChunks; i++){
    ret &= (tables[i]->size() == 0);
  }
  return ret;
}

template<typename CkCacheKey>
void SmpCache<CkCacheKey>::writebackChunk(int chunk){
  typename CkSmpHashtableType::enumerator e = tables[chunk]->make_enumerator();

  const volatile typename CkSmpHashtableType::KEY_VALUE *kv; 
  while (NULL!=(kv=e.next())) {
    CkNodeCacheEntry<CkCacheKey> *entry = kv->v;
    entry->writeback();
  }
}

template<typename CkCacheKey>
void SmpCache<CkCacheKey>::print(int chunk){
  typename CkSmpHashtableType::enumerator e = tables[chunk]->make_enumerator();
  std::ostringstream oss;

  const volatile typename CkSmpHashtableType::KEY_VALUE *kv; 
  while (NULL!=(kv=e.next())) {
    CkNodeCacheEntry<CkCacheKey> *entry = kv->v;
    oss << "(" << kv->k << "; [";
    for(int i = 0; i < kv->v->requestorVec.size(); i++){
      oss << kv->v->requestorVec[i].pe << ", ";
    }
    oss << "]), ";
  }

  CkPrintf("[%d] SmpCache::print(%d): %s\n", CkMyPe(),chunk, oss.str().c_str());
}



template<typename CkCacheKey>
void SmpCache<CkCacheKey>::freeChunk(int chunk){
  //print(chunk);
  typename CkSmpHashtableType::enumerator e = tables[chunk]->make_enumerator();

  const volatile typename CkSmpHashtableType::KEY_VALUE *kv; 
  SMP_CACHE_VERBOSE("[%d] CLEAR CHUNK %d atomicAckDecr id %d\n", CkMyPe(), chunk, id);
  while (NULL!=(kv=e.next())) {
    CkNodeCacheEntry<CkCacheKey> *entry = kv->v;
    // writeback and free data
    entry->free();
    delete entry;
  }

  tables[chunk]->reset();
}

template<typename CkCacheKey>
CkNodeCacheEntry<CkCacheKey> *SmpCache<CkCacheKey>::get(int chunk, CkCacheKey key){
    return tables[chunk]->get(key);
}

template<typename CkCacheKey>
void SmpCache<CkCacheKey>::put(int chunk, CkCacheKey key, CkNodeCacheEntry<CkCacheKey> *fetcherEntry){
    tables[chunk]->put(key, fetcherEntry);
}

template<typename CkCacheKey>
CkNodeCacheEntry<CkCacheKey> *SmpCache<CkCacheKey>::single_put(int chunk, CkCacheKey key, CkNodeCacheEntry<CkCacheKey> *entry){
    return tables[chunk]->single_put(key, entry);
}

// hash function for OSL's hashtable_mt.
// This code was adapted from Thomas Wang's website:
// http://www.concentric.net/~ttwang/tech/inthash.htm
template<typename CkCacheKey>
int SmpCache<CkCacheKey>::hash6432shift(CkCacheKey key)
{
    key = (~key) + (key << 18); // key = (key << 18) - key - 1;
    key = key ^ (key >> 31);
    key = key * 21; // key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 11);
    key = key + (key << 6);
    key = key ^ (key >> 22);
    return (int) key;
}

/* statics */

template<typename CkCacheKey>
CmiUInt8 SmpCache<CkCacheKey>::InvalidKey = CmiUInt8(0);

template<typename CkCacheKey>
CkNodeCacheEntry<CkCacheKey> *SmpCache<CkCacheKey>::InvalidValue = NULL;



#endif // CK_CACHE_SMP_CACHE_H 
