// File: srtable.C
// SRtable is a table that stores timestamped send/recv events of all
// events and cancellations.

// NOTE TO SELF: run w/ +memory_checkfreq=1
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"  // Needed for Charm++ output
#include "srtable.h"
#include "gvt.h"

// Basic constructor
SRtable::SRtable() 
{ 
  gvtWindow = 16;
  numBuckets = 2;
  residuals = recyc = NULL;
  inBuckets = offset = 0;
  bktSz = gvtWindow / numBuckets;
  sends = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  recvs = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  for (int i=0; i<numBuckets; i++) {
    sends[i].count = recvs[i].count = 0;
    sends[i].bucket = recvs[i].bucket = NULL;
    for (int i=0; i<numBuckets; i++) {
      sends[i].initBucket(bktSz, offset+i*bktSz);
      recvs[i].initBucket(bktSz, offset+i*bktSz);
    }
  }
}

// Destructor: needed to free linked lists
SRtable::~SRtable()
{ // why on earth is this implemented like this?
  SRentry *next, *current=residuals;
  while (current) {
    next = current->next;
    current->next = recyc;
    recyc = current;
    current = next;
  }
  for (int i=0; i<numBuckets; i++) {
    sends[i].emptyOutBucket(recyc);
    recvs[i].emptyOutBucket(recyc);
  }
}

// Makes an SRentry out of parameters and sends to other Insert
void SRtable::Insert(int timestamp, int srSt)
{
  SRentry *entry;
  //  sanitize();
  CmiAssert(timestamp >= offset);
  if (timestamp >= offset+gvtWindow) {
    if (recyc) {
      entry = recyc;
      recyc = recyc->next;
      entry->Set(timestamp, srSt, residuals);
      residuals = entry;
    }
    else {
      entry = new SRentry(timestamp, srSt, residuals);
      residuals = entry;
    }
  }
  else {
    inBuckets++;
    if (recyc) {
      entry = recyc;
      recyc = recyc->next;
      entry->Set(timestamp, srSt, NULL);
    }
    else
      entry = new SRentry(timestamp, srSt, NULL);
    int bkt = (timestamp - offset)/bktSz;
    CmiAssert(bkt < numBuckets);
    if (srSt == SEND)  sends[bkt].addToBucket(entry);
    else  recvs[bkt].addToBucket(entry);
  }
  //sanitize();
}

// purge tables below timestamp ts
void SRtable::PurgeBelow(int ts)
{
  //sanitize();
  CmiAssert(ts >= offset);
  int start = (ts - offset)/bktSz, i;
  if (ts == offset) return;  // purge nothing
  else if (ts >= offset + gvtWindow) { // purge everything in buckets
    for (i=0; i<numBuckets; i++) {
      sends[i].emptyOutBucket(recyc);
      recvs[i].emptyOutBucket(recyc);
    }
    offset = offset + gvtWindow;
    inBuckets = 0;
  }
  else { // purge a range of buckets and move higher buckets down
    CmiAssert((start >= 0) && (start < numBuckets));
    for (i=start; i<numBuckets; i++) {
      inBuckets -= sends[i-start].count;
      sends[i-start].emptyOutBucket(recyc);
      sends[i-start].count = sends[i].count;
      sends[i-start].bucket = sends[i].bucket;
      sends[i].bucket = NULL;
      sends[i].count = 0;
      inBuckets -= recvs[i-start].count;
      recvs[i-start].emptyOutBucket(recyc);
      recvs[i-start].count = recvs[i].count;
      recvs[i-start].bucket = recvs[i].bucket;
      recvs[i].bucket = NULL;
      recvs[i].count = 0;
    }
    for (i=(numBuckets-start); i<numBuckets; i++) {
      inBuckets -= sends[i].count;
      inBuckets -= recvs[i].count;
      sends[i].emptyOutBucket(recyc);
      recvs[i].emptyOutBucket(recyc);
    }
    CmiAssert(offset <= (offset + bktSz*start));
    offset = offset + bktSz*start;
  }
  CmiAssert(offset <= ts);
  for (i=0; i<numBuckets; i++) {  // reset the bucket offsets
    sends[i].initBucket(bktSz, offset+i*bktSz);
    recvs[i].initBucket(bktSz, offset+i*bktSz);
  }
  // move contents of first bucket to residuals if in range, otherwise
  // toss out
  SRentry *tmp;
  while (sends[0].bucket) {
    tmp = sends[0].bucket;
    sends[0].bucket = tmp->next;
    if (tmp->timestamp >= ts) {
      tmp->next = residuals;
      residuals = tmp;
    }
    else {
      tmp->next = recyc;
      recyc = tmp;
    }
  }
  while (recvs[0].bucket) {
    tmp = recvs[0].bucket;
    recvs[0].bucket = tmp->next;
    if (tmp->timestamp >= ts) {
      tmp->next = residuals;
      residuals = tmp;
    }
    else {
      tmp->next = recyc;
      recyc = tmp;
    }
  }
  //sanitize();
}

// try to file each residual event in table
void SRtable::FileResiduals()
{
  SRentry *tmp = residuals, *current;
  int bkt;

  //  sanitize();
  residuals = NULL;
  while (tmp) {
    current = tmp;
    tmp = tmp->next;
    current->next = NULL;
    CmiAssert(current->timestamp >= offset);
    if (current->timestamp >= offset+gvtWindow) {
      current->next = residuals;
      residuals = current;
    }
    else {
      bkt = (current->timestamp - offset)/bktSz;
      inBuckets++;
      if (current->sr == SEND)  sends[bkt].addToBucket(current);
      else  recvs[bkt].addToBucket(current);
    }
  }
  //sanitize();
}

// Clears all data from the table
void SRtable::FreeTable()
{
  //  sanitize();
  for (int i=0; i<numBuckets; i++) {
    sends[i].emptyOutBucket(recyc);
    recvs[i].emptyOutBucket(recyc);
  }
  inBuckets = offset = 0;
  SRentry *tmp = residuals, *cur;
  while (tmp) {
    cur = tmp;
    tmp = tmp->next;
    cur->next = recyc;
    recyc = cur;
  }
  residuals = NULL;
  // sanitize();
}

UpdateMsg *SRtable::packTable()
{ // packs only buckets; residuals left behind
  UpdateMsg *um;
  int count=0, i;
  SRentry *j;

  //  sanitize();
  for (i=0; i<numBuckets; i++)
    count = count + sends[i].count + recvs[i].count;
  um = new (count, 8*sizeof(int)) UpdateMsg;
  um->msgCount = count;
  um->gvtW = gvtWindow;
  um->numB = numBuckets;
  um->offset = offset;
  count = 0;
  for (i=0; i<numBuckets; i++) {
    j=sends[i].bucket;
    while (j) {
      um->msgs[count] = *j;
      um->msgs[count].next = NULL;
      count++;
      j = j->next;
    }
    j=recvs[i].bucket;
    while (j) {
      um->msgs[count] = *j;
      um->msgs[count].next = NULL;
      count++;
      j = j->next;
    }
  }
  //sanitize();
  return um;
}

void SRtable::addEntries(UpdateMsg *um)
{
  int i, bkt;
  SRentry *entry, *bkup_resid;
  //  sanitize();

  // backup residuals
  bkup_resid = residuals;
  residuals = NULL;
  // first, resize if necessary
  int oldNumBuckets = numBuckets;
  CmiAssert(offset == um->offset);
  gvtWindow = um->gvtW;
  numBuckets = um->numB;
  // move all elements to residuals
  SRentry *tmp;
  for (int i=0; i<oldNumBuckets; i++) {
    while (sends[i].bucket) {
      tmp = sends[i].bucket;
      sends[i].bucket = tmp->next;
      tmp->next = residuals;
      residuals = tmp;
    }
    while (recvs[i].bucket) {
      tmp = recvs[i].bucket;
      recvs[i].bucket = tmp->next;
      tmp->next = residuals;
      residuals = tmp;
    }
  }
  // free and realloc arrays
  free(sends);
  free(recvs);
  inBuckets = 0;
  bktSz = gvtWindow / numBuckets;
  sends = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  recvs = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  SetOffset(offset);
  FileResiduals();
  // put residuals back
  while (bkup_resid) {
    tmp = bkup_resid;
    bkup_resid = tmp->next;
    tmp->next = residuals;
    residuals = tmp;
  }

  // now move the new stuff in
  for (i=0; i<um->msgCount; i++) {
    CmiAssert(um->msgs[i].timestamp >= offset);
    CmiAssert(um->msgs[i].timestamp < offset+gvtWindow);
    if (recyc) {
      entry = recyc;
      recyc = recyc->next;
      entry->Set(um->msgs[i].timestamp, um->msgs[i].sr, NULL);
    }
    else
      entry = new SRentry(um->msgs[i].timestamp, um->msgs[i].sr, NULL);
    bkt = (um->msgs[i].timestamp - offset)/bktSz;
    CmiAssert((bkt >= 0) && (bkt < numBuckets));
    inBuckets++;
    if (um->msgs[i].sr == SEND)  sends[bkt].addToBucket(entry);
    else  recvs[bkt].addToBucket(entry);
  }
  //sanitize();
}

void SRtable::shrink()
{ 
  // Minimum GVT Window is 8; minimum bucket size is 1
  // Both should always be a power of 2 to make life easy...
  int oldNumBuckets = numBuckets;
  SRentry *tmp, *bkup_resid;

  if (gvtWindow > 16) gvtWindow = gvtWindow-8;
  else return;
  numBuckets = gvtWindow/8;
  // backup residuals
  bkup_resid = residuals;
  residuals = NULL;
  // move all elements to residuals
  for (int i=0; i<oldNumBuckets; i++) {
    while (sends[i].bucket) {
      tmp = sends[i].bucket;
      sends[i].bucket = tmp->next;
      tmp->next = residuals;
      residuals = tmp;
    }
    while (recvs[i].bucket) {
      tmp = recvs[i].bucket;
      recvs[i].bucket = tmp->next;
      tmp->next = residuals;
      residuals = tmp;
    }
  }
  // free and re-malloc arrays
  free(sends);
  free(recvs);
  inBuckets = 0;
  bktSz = gvtWindow / numBuckets;
  sends = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  recvs = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  SetOffset(offset);
  FileResiduals();
  // put residuals back
  while (bkup_resid) {
    tmp = bkup_resid;
    bkup_resid = tmp->next;
    tmp->next = residuals;
    residuals = tmp;
  }
}

void SRtable::expand()
{
  int oldNumBuckets = numBuckets;
  SRentry *tmp, *bkup_resid;

  if (gvtWindow > MAX_GVT_WINDOW) return;
  gvtWindow = gvtWindow+8;
  numBuckets = gvtWindow/8;
  // backup residuals
  bkup_resid = residuals;
  residuals = NULL;
  // move all elements to residuals
  for (int i=0; i<oldNumBuckets; i++) {
    while (sends[i].bucket) {
      tmp = sends[i].bucket;
      sends[i].bucket = tmp->next;
      tmp->next = residuals;
      residuals = tmp;
    }
    while (recvs[i].bucket) {
      tmp = recvs[i].bucket;
      recvs[i].bucket = tmp->next;
      tmp->next = residuals;
      residuals = tmp;
    }
  }
  // free and re-malloc arrays (fastest for expand -- avoids unnecessary copy)
  free(sends);
  free(recvs);
  inBuckets = 0;
  bktSz = gvtWindow / numBuckets;
  sends = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  recvs = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  SetOffset(offset);
  FileResiduals();
  // put residuals back
  while (bkup_resid) {
    tmp = bkup_resid;
    bkup_resid = tmp->next;
    tmp->next = residuals;
    residuals = tmp;
  }
}

void SRtable::dump()
{
  CkPrintf("SRtable::dump: WRITE ME\n");
}

void SRtable::sanitize()
{
  int count;
  SRentry *current;
  for (int i=0; i<numBuckets; i++) {
    count = 0;
    current = sends[i].bucket;
    while (current) {
      count++;
      CmiAssert((current->timestamp >= offset) && 
		(current->timestamp < offset+gvtWindow));
      CmiAssert(current->sr == 0);
      current = current->next;
    }
    CmiAssert(count == sends[i].count);
    count = 0;
    current = recvs[i].bucket;
    while (current) {
      count++;
      CmiAssert((current->timestamp >= offset) && 
		(current->timestamp < offset+gvtWindow));
      CmiAssert(current->sr == 1);
      current = current->next;
    }
    CmiAssert(count == recvs[i].count);
  }
}
