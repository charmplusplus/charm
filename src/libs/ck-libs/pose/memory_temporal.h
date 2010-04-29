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

//#define VERBOSE_DEBUG
//#define ALIGN16 // for 16-byte alignment
#define BLOCK_SIZE 8192  // size of SuperBlocks in bytes
#define	RECYCLE_BIN_CAPACITY 100

extern CkGroupID TempMemID;  // global readonly to access pool anywhere

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
  inline void initBlock() {
    refCount = 0; 
    blk = (char *)malloc(BLOCK_SIZE);
    pos = blk;
    nextBlock = NULL;
    percent_full = 0;
  }
  inline void resetBlock() {
    // assumes this is a recycled SuperBlock, thus blk was already allocated
    refCount = 0; 
    pos = blk;
    nextBlock = NULL;
    percent_full = 0;
  }
  inline bool noLongerReferenced() { return(refCount == 0); }
  /// return pos, and advance pos by sz, aligned to 16 bytes, inc refCount
  char *sb_alloc(int sz) {
#ifdef VERBOSE_DEBUG
    CkPrintf("[sb_alloc:\n");
#endif
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
      percent_full = (int)(((float)(pos-blk)/4096.0)*100.0);
    }
#ifdef VERBOSE_DEBUG
    CkPrintf(".sb_alloc]\n");
#endif
    return ret;
  }
  // dec refCount
  bool sb_free(void *mem) { 
#ifdef VERBOSE_DEBUG
    CkPrintf("[sb_free:\n");
#endif
    if ((mem >= blk) && (mem < pos)) {
      refCount--; 
#ifdef VERBOSE_DEBUG
      CkPrintf(".sb_free]\n");
#endif
      return true;
    }
    else {
#ifdef VERBOSE_DEBUG
      CkPrintf(".sb_free]\n");
#endif
      return false;
    }
  }
  inline SuperBlock *getNextBlock() { return nextBlock; }
  inline void setNextBlock(SuperBlock *loc) { nextBlock = loc; }
  inline int getPercentFull() { return percent_full; }
  void sanity_check();
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
  int *poolSz;

 public:
  TimeBucket() : start(POSE_UnsetTS), range(POSE_UnsetTS), numSuperBlocks(0),
    sBlocks(NULL), nextBucket(NULL), prevBucket(NULL) {}
  ~TimeBucket() {} // these are garbage collected in the cleanup function
  // Initialize time range and create first SuperBlock
  void initBucket(POSE_TimeType start_t, POSE_TimeType range_t, SuperBlock **p, int *pSz) {
    pool = p;
    poolSz = pSz;
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
      (*poolSz)--;
    }
    numSuperBlocks = 1;
  }
  inline int getNumSuperBlocks() { return numSuperBlocks; }
  inline int getStart() { return start; }
  inline int getRange() { return range; }
  inline void setStart(int s) { start = s; }
  inline void setRange(int r) { range = r; }
  inline bool isVeryFull() {
    if (numSuperBlocks > 1) return true;
    else if (sBlocks->getPercentFull() > 90) return true;
    else return false;
  }
  inline SuperBlock *getFirstSuperBlock() { return sBlocks; }
  inline void setFirstSuperBlock(SuperBlock *sb) { sBlocks = sb; }
  inline void setPrevBucket(TimeBucket *p) { prevBucket = p; }
  inline void setNextBucket(TimeBucket *n) { nextBucket = n; }
  inline TimeBucket *getPrevBucket() { return prevBucket; }
  inline TimeBucket *getNextBucket() { return nextBucket; }
  // Get some memory in this time range
  char *tb_alloc(int sz) {
#ifdef VERBOSE_DEBUG
    CkPrintf("[tb_alloc:\n");
#endif
    char *newblk = sBlocks->sb_alloc(sz);
    if (!newblk) {
      SuperBlock *tmp;
      if (!(*pool)) {
	tmp = new SuperBlock;
	tmp->initBlock();
      }
      else {
	tmp = (*pool);
	(*pool) = (*pool)->getNextBlock();
	tmp->resetBlock();
	(*poolSz)--;
      }
      tmp->setNextBlock(sBlocks);
      sBlocks = tmp;
      numSuperBlocks++;
#ifdef VERBOSE_DEBUG
      if (numSuperBlocks > 3)
	CkPrintf("WARNING: # SuperBlocks in TimeBucket exceeds 3 at %d.  SUGGESTION: Increase block size.\n", numSuperBlocks);
#endif
      newblk = sBlocks->sb_alloc(sz);
    }
#ifdef VERBOSE_DEBUG
    CkPrintf(".tb_alloc]\n");
#endif
    return newblk;
  }
  // "Free" some memory from this time range
  void tb_free(char *mem) {
#ifdef VERBOSE_DEBUG
    CkPrintf("[tb_free:\n");
#endif
    SuperBlock *tmp = sBlocks;
    bool done = false;
    while (tmp && !done) {
      done = tmp->sb_free(mem);
      if (done) {
	if (tmp->noLongerReferenced())
	  numSuperBlocks--;
      }
      else {
	tmp = tmp->getNextBlock();
      }
    }
    if (!done) CkAbort("ERROR: block to deallocate not found in time range.\n");
#ifdef VERBOSE_DEBUG
    CkPrintf(".tb_free]\n");
#endif
  }
  POSE_TimeType sanity_check(POSE_TimeType last_time);
};

class TimePool : public Group {
  TimeBucket *last_in_use;  // head of doubly-linked list
  TimeBucket *first_in_use; // tail of doubly-linked list
  SuperBlock *not_in_use;   // separate singly-linked list
  int not_in_use_sz;
  
  POSE_TimeType min_time;   // blocks older than this can be recycled

  // The following fields are dynamically adjusted with application behavior
  int BLOCK_RANGE; // This is selected and adjusted to avoid having >1
		   // SuperBlock per TimeBucket

  void clean_up(); // Move old defunct SuperBlocks to not_in_use list
 public:
  TimePool() : min_time(POSE_UnsetTS), last_in_use(NULL), first_in_use(NULL), 
    not_in_use(NULL), not_in_use_sz(0) {}
  TimePool(CkMigrateMessage *msg) : Group(msg) {}
  ~TimePool();
  void pup(PUP::er &p) {}
  // Return memory from a time range
  char *tmp_alloc(POSE_TimeType timestamp, int sz_in_bytes);
  // "Free" up memory from a time range
  void tmp_free(POSE_TimeType timestamp, void *mem);
  // Update the minimum time before which SuperBlocks can be recycled
  inline void set_min_time(POSE_TimeType min_t) { min_time = min_t; clean_up(); }
  void empty_recycle_bin() {
    SuperBlock *b = not_in_use;
    while (not_in_use) {
      not_in_use = not_in_use->getNextBlock();
      delete b;
      b = not_in_use;
    }
  }
  void sanity_check(); 
};

#endif /* !MEMORY_TEMPORAL_H_ */
