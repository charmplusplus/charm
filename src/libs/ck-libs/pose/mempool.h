/// Memory recycling for POSE
/* Allows for the storage and reuse of memory blocks for event messages and 
   checkpointing. */

#ifndef MEMPOOL_H
#define MEMPOOL_H
#include "mempool.decl.h"

extern CkGroupID MemPoolID;  // global readonly to access pool anywhere

// Basic single pool of same-size memory blocks
class Pool
{
 public:
  int numBlocks, blockSize;
  void *memPool[MAX_POOL_SIZE];
  Pool *next, *prev;
  Pool() : numBlocks(0),blockSize(0),next(NULL), prev(NULL){ }
};

// Set of memory pools for various size blocks; 1 MemoryPool per PE
class MemoryPool : public Group {
private:
  Pool *memPools;  // the Pools
  Pool *lastLook;  // last pool looked at
public:
  /// Basic initialization
  MemoryPool() :memPools(NULL),lastLook(NULL){
#ifdef VERBOSE_DEBUG
    CkPrintf("[%d] constructing MemoryPool\n",CkMyPe());
#endif
  }
  MemoryPool(CkMigrateMessage *msg) : Group(msg) { }
  void pup(PUP::er &p) { }
  /// returns number of blocks of size sz in pool
  int CheckPool(int sz); 
  /// returns a block from pool with size sz
  /* Assumes a block of the appropriate size exists! */
  void *GetBlock(int sz); 
  /// puts a block of size sz in appropriate pool
  /* Assumes there is space in the pool for the block! */
  void PutBlock(int sz, void *blk); 
};

#endif
