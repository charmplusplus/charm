/// Memory recycling for POSE
/* Allows for the storage and reuse of memory blocks for event messages and 
   checkpointing. */
#include "charm++.h"
#include "pose.h"
#include "mempool.def.h"

CkGroupID MemPoolID;

// Returns number of blocks of size sz in pool
int MemoryPool::CheckPool(int sz)
{
  if (sz > MAX_RECYCLABLE) return -1;
  if (memPools) {
    if (lastLook && (lastLook->blockSize == sz))
      return (lastLook->numBlocks);
    else {
      lastLook = memPools;
      while (lastLook && (lastLook->blockSize < sz)) 
	lastLook = lastLook->next;
      if (lastLook && (lastLook->blockSize == sz))
	return lastLook->numBlocks;
      return 0;
    }
  }
  else return 0;
}

// Returns a block of size sz from pool
void *MemoryPool::GetBlock(int sz)
{
  if (lastLook && (lastLook->blockSize == sz)) {
    lastLook->numBlocks--;
    return lastLook->memPool[lastLook->numBlocks];
  }
  else {
    lastLook = memPools;
    while (lastLook->blockSize != sz) 
      lastLook = lastLook->next;
    lastLook->numBlocks--;
    return lastLook->memPool[lastLook->numBlocks];
  }
}

// Puts a block of size sz in pool
void MemoryPool::PutBlock(int sz, void *blk)
{
  if (lastLook && (lastLook->blockSize == sz)) {
    lastLook->memPool[lastLook->numBlocks] = blk;
    lastLook->numBlocks++;
  }
  else {
    lastLook = memPools;
    if (!lastLook) {
      Pool *np = new Pool();
      np->blockSize = sz; 
      np->numBlocks = 0;
      np->next = np->prev = NULL;
      np->memPool[np->numBlocks] = blk;
      np->numBlocks++;
      memPools = lastLook = np;
    }
    else if (lastLook->blockSize > sz) {
      Pool *np = new Pool();
      np->blockSize = sz; 
      np->numBlocks = 0;
      np->next = memPools;
      np->prev = NULL;
      lastLook->prev = np;
      np->memPool[np->numBlocks] = blk;
      np->numBlocks++;
      memPools = lastLook = np;
    }
    else if (lastLook->blockSize == sz) {
      lastLook->memPool[lastLook->numBlocks] = blk;
      lastLook->numBlocks++;
    }
    else {
      while (lastLook->next && (lastLook->next->blockSize < sz))
	lastLook = lastLook->next;
      if (lastLook->next) {
	if (lastLook->next->blockSize == sz) {
	  lastLook->next->memPool[lastLook->next->numBlocks] = blk;
	  lastLook->next->numBlocks++;
	}
	else {
	  Pool *np = new Pool();
	  np->blockSize = sz; 
	  np->numBlocks = 0;
	  np->next = lastLook->next;
	  np->prev = lastLook;
	  lastLook->next->prev = np;
	  lastLook->next = np;
	  np->memPool[np->numBlocks] = blk;
	  np->numBlocks++;
	  lastLook = np;
	}
      }
      else {
	Pool *np = new Pool();
	np->blockSize = sz; 
	np->numBlocks = 0;
	np->next = NULL;
	np->prev = lastLook;
	lastLook->next = np;
	np->memPool[np->numBlocks] = blk;
	np->numBlocks++;
	lastLook = np;
      }
    }
  }
}
