/*
** memory_temporal.h
** 
** Made by Terry Wilmarth
** Login   <wilmarth@order.cs.uiuc.edu>
** 
** Started on  Wed Nov  7 15:43:21 2007 Terry Wilmarth
** Last update Wed Nov  7 17:09:23 2007 Terry Wilmarth
*/

/* Temporal memory management:
   This consists of a TimePool, (a group, so there is one per
   processor), which contains a list of "in use" memory, and a list of 
   "not in use" memory (to be recycled).
   These lists are ordered buckets, or TimeBuckets.  Each time bucket
   corresponds to a range of time, and has one or more blocks of
   memory within it.  We wish to keep the number of blocks of memory
   in each time bucket low.
   These blocks are called SuperBlocks, and maintain their own size, a
   reference count, and pointers to the front and empty portions of
   the block.
*/

#ifndef	MEMORY_TEMPORAL_H_
#define	MEMORY_TEMPORAL_H_

#include "pose_config.h"
#include "memory_temporal.decl.h"

extern CkGroupID TempMemID;  // global readonly to access pool anywhere

//#define ALIGN16 // for 16-byte alignment
#define BLOCK_SIZE 4096  // size of SuperBlocks in bytes

/// SuperBlock holds the actual memory block that is allocated in blk
class SuperBlock {
  int refCount; // number of non-freed blocks within this SuperBlock
  char *blk; // pointer to this block
  char *pos; // pointer to next 16-byte aligned free location in block
  int percent_full; // percentage of first SuperBlock that is/was used 
  SuperBlock *nextBlock; // pointer to next SuperBlock in TimeBucket
  
 public:
 SuperBlock() : refCount(0), blk(NULL), pos(NULL) {};
  ~SuperBlock() { free(blk); }
  /// Allocate and set initial values
  void initBlock() {
    refCount = 0; 
    blk = (char *)malloc(BLOCK_SIZE);
    pos = blk;
    nextBlock = NULL;
    percent_full = 0;
  }
  void resetBlock() {
    // assumes this is a recycled SuperBlock, thus blk was already allocated
    refCount = 0; 
    pos = blk;
    nextBlock = NULL;
    percent_full = 0;
  }
  bool noLongerReferenced() { return(refCount == 0); }
  /// return pos, and advance pos by sz, aligned to 16 bytes, inc refCount
  char *sb_alloc(int sz) {
    int remaining = BLOCK_SIZE - (pos - blk);
#ifdef ALIGN16
    int actual_sz = (sz%16 == 0)? sz : (sz+16)/16 * 16;
#else
    int actual_sz = sz;
#endif
    char *ret = NULL;
    if (actual_sz <= remaining) {
      ret = pos;
      pos += actual_sz;
      refCount++;
    }
    percent_full = (int)((float)(pos-blk)/4096.0)*100;
    return ret;
  }
  // dec refCount
  bool sb_free(void *mem) { 
    if ((mem >= blk) && (blk < pos)) {
      refCount--; 
      return true;
    }
    else return false;
  }
  SuperBlock *getNextBlock() { return nextBlock; }
  void setNextBlock(SuperBlock *loc) { nextBlock = loc; }
  int getPercentFull() { return percent_full; }
};

/// TimeBucket associates a time range with (a) large block(s) of memory
class TimeBucket {
  POSE_TimeType start; // start of time range
  POSE_TimeType range; // range of time covered by this bucket
  int numSuperBlocks; // number of SuperBlocks in sBlocks list
  SuperBlock *sBlocks; // list of SuperBlocks
  TimeBucket *nextBucket; // pointer to next Bucket in TimePool
  TimeBucket *prevBucket; // pointer to previous Bucket in TimePool
  SuperBlock **pool;

 public:
  TimeBucket() : start(POSE_UnsetTS), range(POSE_UnsetTS), numSuperBlocks(0),
    sBlocks(NULL), nextBucket(NULL), prevBucket(NULL) {}
  ~TimeBucket() {} // these are garbage collected in the cleanup function
  // Initialize time range and create first SuperBlock
  void initBucket(POSE_TimeType start_t, POSE_TimeType range_t, SuperBlock **p) {
    pool = p;
    start = start_t;
    range = range_t;
    if (!(*pool)) {
      sBlocks = new SuperBlock; // later, check the recycle bin
      sBlocks->initBlock();
    }
    else {
      sBlocks = (*pool);
      (*pool) = (*pool)->getNextBlock();
      sBlocks->resetBlock();
    }
    numSuperBlocks = 1;
  }
  int getStart() { return start; }
  int getRange() { return range; }
  void setStart(int s) { start = s; }
  void setRange(int r) { range = r; }
  bool isVeryFull() {
    if (numSuperBlocks > 1) return true;
    else if (sBlocks->getPercentFull() > 90) return true;
    else return false;
  }
  SuperBlock *getFirstSuperBlock() { return sBlocks; }
  void setPrevBucket(TimeBucket *p) { prevBucket = p; }
  void setNextBucket(TimeBucket *n) { nextBucket = n; }
  TimeBucket *getPrevBucket() { return prevBucket; }
  TimeBucket *getNextBucket() { return nextBucket; }
  // Get some memory in this time range
  char *tb_alloc(int sz) {
    char *newblk = sBlocks->sb_alloc(sz);
    if (!newblk) {
      SuperBlock *tmp;
      if (!(*pool)) {
	tmp = new SuperBlock; // later, check the recycle bin
	tmp->initBlock();
      }
      else {
	tmp = (*pool);
	(*pool) = (*pool)->getNextBlock();
	tmp->resetBlock();
      }
      tmp->setNextBlock(sBlocks);
      sBlocks = tmp;
      numSuperBlocks++;
      newblk = sBlocks->sb_alloc(sz);
    }
    return newblk;
  }
  // "Free" some memory from this time range
  void tb_free(char *mem) {
    SuperBlock *tmp = sBlocks;
    bool done = false;
    while (tmp && !done) {
      done = tmp->sb_free(mem);
      sBlocks = tmp->getNextBlock();
      tmp = sBlocks;
    }
    if (!done) printf("ERROR: block to deallocate not found in time range.\n");
  }
};

class TimePool : public Group {
  TimeBucket *last_in_use;  // head of doubly-linked list
  TimeBucket *first_in_use; // tail of doubly-linked list
  SuperBlock *not_in_use;   // separate singly-linked list
  
  POSE_TimeType min_time;   // blocks older than this can be recycled

  // The following fields are dynamically adjusted with application behavior
  int BLOCK_RANGE; // This is selected and adjusted to avoid having >1
		   // SuperBlock per TimeBucket

  void clean_up(); // Move old defunct SuperBlocks to not_in_use list
 public:
  TimePool() : min_time(POSE_UnsetTS), last_in_use(NULL), first_in_use(NULL), 
    not_in_use(NULL) {}
  TimePool(CkMigrateMessage *) {}
  ~TimePool();
  // Return memory from a time range
  char *tmp_alloc(POSE_TimeType timestamp, int sz_in_bytes);
  // "Free" up memory from a time range
  void tmp_free(POSE_TimeType timestamp, void *mem);
  // Update the minimum time before which SuperBlocks can be recycled
  void set_min_time(POSE_TimeType min_t) { min_time = min_t; clean_up(); }
};

#endif /* !MEMORY_TEMPORAL_H_ */
