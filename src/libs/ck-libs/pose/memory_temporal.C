/*
** memory_temporal.C
** 
** Made by Terry Wilmarth
*/
#include "charm++.h"
#include "pose.h"
#include "memory_temporal.def.h"

CkGroupID TempMemID;  // global readonly to access pool anywhere

void SuperBlock::sanity_check() {
  // no obvious way to verify the sanity of a SuperBlock
}

POSE_TimeType TimeBucket::sanity_check(POSE_TimeType last_time) {
  CkAssert(start > last_time);
  if (!sBlocks) CkAssert(numSuperBlocks == 0);
  if (sBlocks) {
    int count=0;
    SuperBlock *tmpblk = sBlocks;
    while (tmpblk) {
      if (!tmpblk->noLongerReferenced()) count++;
      tmpblk = tmpblk->getNextBlock();
    }
    CkAssert(count == numSuperBlocks);
    tmpblk = sBlocks;
    while (tmpblk) {
      tmpblk->sanity_check();
      tmpblk = tmpblk->getNextBlock();
    }
  }
  return (start+range-1);
}

void TimePool::clean_up() {
#ifdef VERBOSE_DEBUG
  sanity_check();
#endif
  TimeBucket *tmpbkt = first_in_use;
  while (tmpbkt && (min_time >= (tmpbkt->getStart()+tmpbkt->getRange()))) {
    // move the blocks of this bucket to not_in_use
    SuperBlock *sb = tmpbkt->getFirstSuperBlock();
    while (sb) {
      if (sb->noLongerReferenced()) {
	SuperBlock *next = sb->getNextBlock();
	if (not_in_use_sz < RECYCLE_BIN_CAPACITY) {
	  sb->setNextBlock(not_in_use);
	  not_in_use = sb;
	  not_in_use_sz++;
	}
	else {
#ifdef VERBOSE_DEBUG
	  CkPrintf("INFO: recycle bin at capacity; deleting blocks\n");
#endif
	  delete sb;
	}
	sb = next;
      }
      else break;
    }
    if (!sb) { // bucket is totally empty, delete it
      // move the first_in_use ptr forward
      if (tmpbkt == first_in_use) {
	if (first_in_use == last_in_use) {
	  first_in_use = last_in_use = NULL;
	}
	else {
	  first_in_use = first_in_use->getPrevBucket();
	  first_in_use->setNextBucket(NULL);
	}
	delete tmpbkt;
	tmpbkt = first_in_use;
      }
      else if (tmpbkt == last_in_use) {
	last_in_use = last_in_use->getNextBucket();
	last_in_use->setPrevBucket(NULL);
	delete tmpbkt;
	tmpbkt = NULL;
      }
      else {
	TimeBucket *tmp = tmpbkt->getPrevBucket();
	tmp->setNextBucket(tmpbkt->getNextBucket());
	tmpbkt->getNextBucket()->setPrevBucket(tmp);
	delete tmpbkt;
	tmpbkt = tmp;
      }
    }
    else {
      tmpbkt->setFirstSuperBlock(sb);
      tmpbkt = tmpbkt->getPrevBucket();
    }
  }
#ifdef VERBOSE_DEBUG
  sanity_check();
#endif
}

TimePool::~TimePool() 
{
  TimeBucket *tmpbkt = last_in_use;
  while (tmpbkt) {
    last_in_use = last_in_use->getNextBucket();
    delete tmpbkt;
    tmpbkt = last_in_use;
  }
  SuperBlock *tmpblk = not_in_use;
  while (tmpblk) {
    not_in_use = not_in_use->getNextBlock();
    delete tmpblk;
    tmpblk = not_in_use;
  }
}

// Return memory from a time range
char *TimePool::tmp_alloc(POSE_TimeType timestamp, int sz_in_bytes)
{ // List looks like this:
  //  last (newer, higher ts) .... first (older, lower ts)
#ifdef VERBOSE_DEBUG
  sanity_check();
  CkPrintf("[tmp_alloc:\n");
#endif
  TimeBucket *bkt = last_in_use;
  while (bkt && (timestamp < bkt->getStart())) {
    bkt = bkt->getNextBucket();
  }
  if (!bkt) { // either empty, or ts is older than anything we have
    if (!first_in_use) {  // empty
      first_in_use = new TimeBucket();
      first_in_use->initBucket(timestamp, 1, &not_in_use, &not_in_use_sz);
      last_in_use = first_in_use;
      char *mem = last_in_use->tb_alloc(sz_in_bytes);
#ifdef VERBOSE_DEBUG
      CkPrintf(".tmp_alloc]\n");
      sanity_check();
#endif
      return mem;
    }
    else if (timestamp < first_in_use->getStart()) { //not empty, ts is oldest
      first_in_use->setStart(timestamp);
      char *mem = first_in_use->tb_alloc(sz_in_bytes);
#ifdef VERBOSE_DEBUG
      CkPrintf(".tmp_alloc]\n");
      sanity_check();
#endif
      return mem;
    }
  }
  else if (bkt == last_in_use) {  // we have some options if the target is last
    if (bkt->isVeryFull() && (timestamp >= (bkt->getStart() + bkt->getRange()))) { // let's make a new bucket, timestamp is far out enough
      int start = bkt->getStart()+bkt->getRange();
      int range = timestamp - start + 1;
      last_in_use = new TimeBucket();
      last_in_use->initBucket(start, range, &not_in_use, &not_in_use_sz);
      bkt->setPrevBucket(last_in_use);
      last_in_use->setNextBucket(bkt);
      char *mem = last_in_use->tb_alloc(sz_in_bytes);
#ifdef VERBOSE_DEBUG
      CkPrintf(".tmp_alloc]\n");
      sanity_check();
#endif
      return mem;
    }
    else { // let's put it here, expanding the range if necessary
      if (timestamp >= (bkt->getStart() + bkt->getRange())) {
	bkt->setRange(timestamp - bkt->getStart() + 1);
      }
      char *mem = bkt->tb_alloc(sz_in_bytes);
#ifdef VERBOSE_DEBUG
      CkPrintf(".tmp_alloc]\n");
      sanity_check();
#endif
      return mem;
    }
  }
  else { // this is in the range of this bucket, must put it here
    char *mem = bkt->tb_alloc(sz_in_bytes);
#ifdef VERBOSE_DEBUG
    CkPrintf(".tmp_alloc]\n");
    sanity_check();
#endif
    return mem;
  }
}

// "Free" up memory from a time range
void TimePool::tmp_free(POSE_TimeType timestamp, void *mem) 
{
#ifdef VERBOSE_DEBUG
  sanity_check();
  CkPrintf("[tmp_free:\n");
#endif
  if (mem) {
    TimeBucket *tmpbkt = first_in_use;
    while (tmpbkt && (timestamp >= (tmpbkt->getStart()+tmpbkt->getRange()))) {
      tmpbkt = tmpbkt->getPrevBucket();
    }
    if (tmpbkt) {
      tmpbkt->tb_free((char *)mem);
    }
    else CkAbort("ERROR: Memory in that time range not found for deallocation.\n");
#ifdef VERBOSE_DEBUG
    CkPrintf(".tmp_free]\n");
    sanity_check();
#endif
  }
}

void TimePool::sanity_check() {
  // first check the quality of the list of in-use buckets
  if (!last_in_use) CkAssert(!first_in_use);
  if (!first_in_use) CkAssert(!last_in_use);
  if (last_in_use) CkAssert(!(last_in_use->getPrevBucket()));
  if (first_in_use) CkAssert(!(first_in_use->getNextBucket()));
  TimeBucket *tmpbkt = last_in_use;
  if (tmpbkt) {
    while (tmpbkt->getNextBucket()) {
      CkAssert(tmpbkt->getNextBucket()->getPrevBucket() == tmpbkt);
      tmpbkt = tmpbkt->getNextBucket();
    }
    CkAssert(tmpbkt == first_in_use);
  }
  tmpbkt = first_in_use;
  if (tmpbkt) {
    while (tmpbkt->getPrevBucket()) {
      CkAssert(tmpbkt->getPrevBucket()->getNextBucket() == tmpbkt);
      tmpbkt = tmpbkt->getPrevBucket();
    }
    CkAssert(tmpbkt == last_in_use);
  }
  // ASSERT: Bucket structure of TimePool is fine at this point.
  // Now, we examine the bucket contents
  tmpbkt = first_in_use;
  if (tmpbkt) {
    POSE_TimeType lastTime = POSE_UnsetTS;
    while (tmpbkt) {
      lastTime = tmpbkt->sanity_check(lastTime);
      tmpbkt = tmpbkt->getPrevBucket();
    }
  }
}
