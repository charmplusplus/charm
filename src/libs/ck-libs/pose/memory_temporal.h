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

#define BLOCK_SIZE 4096  // size of SuperBlocks in bytes

/// SuperBlock holds the actual memory block that is allocated in blk
class SuperBlock {
  int refCount; // number of non-freed blocks within this SuperBlock
  void *blk; // pointer to this block
  void *pos; // pointer to next 16-byte aligned free location in block
  SuperBlock *nextBlock; // pointer to next SuperBlock in TimeBucket
  
 public:
  SuperBlock() : size(0), refCount(0), blk(NULL), pos(NULL);
  ~SuperBlock();
  /// Allocate and set initial values
  void initBlock(int blk_sz);
  /// return pos, and advance pos by sz, aligned to 16 bytes, inc refCount
  void *sb_alloc(int sz);
  // dec refCount
  void sb_free(void *mem) { refCount--; };
};

/// TimeBucket associates a time range with (a) large block(s) of memory
class TimeBucket {
  POSE_TimeType start; // start of time range
  POSE_TimeType range; // range of time covered by this bucket
  int numSuperBlocks; // number of SuperBlocks in sBlocks list
  SuperBlock *sBlocks; // list of SuperBlocks
  float percent_full; // percentage of first SuperBlock that is/was used 
  TimeBucket *nextBucket; // pointer to next Bucket in TimePool
  TimeBucket *prevBucket; // pointer to previous Bucket in TimePool

 public:
  TimeBucket() : start(POSE_UnsetTS), range(POSE_UnsetTS), numSuperBlocks(0),
    sBlocks(NULL), nextBucket(NULL), prevBucket(NULL) {}
  ~TimeBucket();
  // Initialize time range and create first SuperBlock
  void initBucket(POSE_TimeType start_t, POSE_TimeType range_t, int sz);
  // Get some memory in this time range
  void *tb_alloc(int sz);
  // "Free" some memory from this time range
  void tb_free(void *mem);
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
  void *tmp_alloc(POSE_TimeType timestamp, int sz_in_bytes);
  // "Free" up memory from a time range
  void tmp_free(POSE_TimeType timestamp, void *mem);
  // Update the minimum time before which SuperBlocks can be recycled
  void set_min_time(POSE_TimeType min_t);
};

#endif /* !MEMORY_TEMPORAL_H_ */
