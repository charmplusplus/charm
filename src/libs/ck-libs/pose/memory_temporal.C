/*
** memory_temporal.C
** 
** Made by Terry Wilmarth
*/
#include "charm++.h"
#include "pose.h"
#include "memory_temporal.def.h"

CkGroupID TempMemID;  // global readonly to access pool anywhere

void TimePool::clean_up() {
  TimeBucket *tmpbkt = first_in_use;
  while (tmpbkt && (min_time >= (tmpbkt->getStart()+tmpbkt->getRange()))) {
    // move the blocks of this bucket to not_in_use
    SuperBlock *sb = tmpbkt->getFirstSuperBlock();
    while (sb) {
      if (!sb->noLongerReferenced())
	CkPrintf("ERROR: attempting to delete a SuperBlock that is still referenced.\n");
      SuperBlock *next = sb->getNextBlock();
      sb->setNextBlock(not_in_use);
      not_in_use = sb;
      sb = next;
    }
    // move the first_in_use ptr forward
    first_in_use = first_in_use->getNextBucket();
    delete tmpbkt;
    tmpbkt = first_in_use;
  }
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
{
  TimeBucket *bkt = last_in_use;
  while (bkt && (timestamp < bkt->getStart())) {
    bkt = bkt->getNextBucket();
  }
  if (!bkt) {
    if (!last_in_use) {
      last_in_use = new TimeBucket();
      last_in_use->initBucket(timestamp, 1, &not_in_use);
      first_in_use = last_in_use;
    }
    else if (timestamp < first_in_use->getStart()) {
      first_in_use->setStart(timestamp);
      return first_in_use->tb_alloc(sz_in_bytes);
    }
  }
  else if (bkt == last_in_use) {
    if (bkt->isVeryFull()) {
      int start = bkt->getStart()+bkt->getRange();
      int range = timestamp - start + 1;
      last_in_use = new TimeBucket();
      last_in_use->initBucket(start, range, &not_in_use);
      bkt->setPrevBucket(last_in_use);
      last_in_use->setNextBucket(bkt);
      return last_in_use->tb_alloc(sz_in_bytes);
    }
    else {
      if (timestamp > (bkt->getStart() + bkt->getRange())) {
	bkt->setRange(timestamp - bkt->getStart() + 1);
      }
      return bkt->tb_alloc(sz_in_bytes);
    }
  }
  else {
    return bkt->tb_alloc(sz_in_bytes);
  }
}

// "Free" up memory from a time range
void TimePool::tmp_free(POSE_TimeType timestamp, void *mem) 
{
  TimeBucket *tmpbkt = first_in_use;
  while (tmpbkt && (timestamp >= (tmpbkt->getStart()+tmpbkt->getRange()))) {
    tmpbkt = tmpbkt->getPrevBucket();
  }
  if (tmpbkt) {
    tmpbkt->tb_free((char *)mem);
  }
  else printf("ERROR: Memory in that time range not found for deallocation.\n");
}
