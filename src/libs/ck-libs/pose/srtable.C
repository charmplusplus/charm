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
  residuals = residualsTail = recyc = recycTail = NULL;
  inBuckets = offset = 0;
  bktSz = gvtWindow / numBuckets;
  sends = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  recvs = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  for (int i=0; i<numBuckets; i++) {
    sends[i].initBucket(bktSz, offset+i*bktSz);
    recvs[i].initBucket(bktSz, offset+i*bktSz);
  }
}

// Destructor: needed to free linked lists
SRtable::~SRtable()
{ // why on earth is this implemented like this?
  /*  
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
  */
}

// Makes an SRentry out of parameters and sends to other Insert
void SRtable::Insert(int timestamp, int srSt)
{
  //sanitize();
  SRentry *entry;
  if (timestamp >= offset+gvtWindow) {
    if (recyc) {
      entry = recyc;
      recyc = recyc->next;
      if (!recyc) recycTail = NULL;
      entry->Set(timestamp, srSt, residuals);
      if (residualsTail)
	residuals = entry;
      else
	residuals = residualsTail = entry;
    }
    else {
      entry = new SRentry(timestamp, srSt, residuals);
      if (residualsTail)
	residuals = entry;
      else
	residuals = residualsTail = entry;
    }
  }
  else {
    //    CmiAssert(timestamp >= offset);
    inBuckets++;
    if (recyc) {
      entry = recyc;
      recyc = recyc->next;
      if (!recyc) recycTail = NULL;
      entry->Set(timestamp, srSt, NULL);
    }
    else
      entry = new SRentry(timestamp, srSt, NULL);
    int bkt = (timestamp - offset)/bktSz;
    if (srSt == SEND)  sends[bkt].addToBucket(entry);
    else  recvs[bkt].addToBucket(entry);
  }
  //sanitize();
}

// purge tables below timestamp ts
void SRtable::PurgeBelow(int ts)
{
  //sanitize();
  int start = (ts - offset)/bktSz, i;
  if (ts < offset+bktSz) { // purge nothing: ts is in zero-th bucket
  }
  else if (ts >= offset + gvtWindow) { // purge everything in buckets
    offset = offset + gvtWindow;
    int bktOffset;
    for (i=0; i<numBuckets; i++) {
      sends[i].emptyOutBucket(recyc, recycTail, &recycCount);
      recvs[i].emptyOutBucket(recyc, recycTail, &recycCount);
      bktOffset = offset+i*bktSz;
      sends[i].setBucketOffset(bktOffset);
      recvs[i].setBucketOffset(bktOffset);
    }
    inBuckets = 0;
  }
  else { // purge a range of buckets and move higher buckets down
    //CmiAssert((start >= 0) && (start < numBuckets));
    offset = offset + bktSz*start;
    int offIdx, bktOffset;
    for (i=start; i<numBuckets; i++) {
      offIdx = i-start;
      bktOffset = offset+i*bktSz;
      inBuckets -= sends[offIdx].count;
      sends[offIdx].emptyOutBucket(recyc, recycTail, &recycCount);
      sends[offIdx].count = sends[i].count;
      sends[offIdx].bucket = sends[i].bucket;
      sends[offIdx].bucketTail = sends[i].bucketTail;
      sends[i].initBucket(bktSz, bktOffset);
      inBuckets -= recvs[offIdx].count;
      recvs[offIdx].emptyOutBucket(recyc, recycTail, &recycCount);
      recvs[offIdx].count = recvs[i].count;
      recvs[offIdx].bucket = recvs[i].bucket;
      recvs[offIdx].bucketTail = recvs[i].bucketTail;
      recvs[i].initBucket(bktSz, bktOffset);
      sends[offIdx].offset = recvs[offIdx].offset = offset+offIdx*bktSz;
    }
    for (i=(numBuckets-start); i<start; i++) {
      // purge the in between buckets if there are any
      bktOffset = offset+i*bktSz;
      inBuckets -= sends[i].count;
      inBuckets -= recvs[i].count;
      sends[i].emptyOutBucket(recyc, recycTail, &recycCount);
      recvs[i].emptyOutBucket(recyc, recycTail, &recycCount);
      sends[i].setBucketOffset(bktOffset);
      recvs[i].setBucketOffset(bktOffset);
    }
    //CmiAssert(offset <= (offset + bktSz*start));
  }
  //sanitize();
  // purge entries in first bucket with timestamp < ts
  SRentry *tmp;
  while (sends[0].bucket && (sends[0].bucket->timestamp < ts)) {
    tmp = sends[0].bucket;
    sends[0].bucket = tmp->next;
    sends[0].count--;
    delete(tmp);
    inBuckets--;
  }
  if (!sends[0].bucket) sends[0].bucketTail = NULL;
  while (recvs[0].bucket && (recvs[0].bucket->timestamp < ts)) {
    tmp = recvs[0].bucket;
    recvs[0].bucket = tmp->next;
    recvs[0].count--;
    delete(tmp);
    inBuckets--;
  }
  if (!recvs[0].bucket) recvs[0].bucketTail = NULL;
  //sanitize();
}

// try to file each residual event in table
void SRtable::FileResiduals()
{
  //sanitize();
  SRentry *tmp = residuals, *current;
  int bkt;

  residuals = residualsTail = NULL;
  while (tmp) {
    current = tmp;
    tmp = tmp->next;
    current->next = NULL;
    if (current->timestamp >= offset+gvtWindow) {
      current->next = residuals;
      if (residuals) residuals = current;
      else residuals = residualsTail = current;
    }
    else {
      //CmiAssert(current->timestamp >= offset);
      bkt = (current->timestamp - offset)/bktSz;
      inBuckets++;
      if (current->sr == SEND)  sends[bkt].addToBucket(current);
      else  recvs[bkt].addToBucket(current);
    }
  }
  //sanitize();
}

UpdateMsg *SRtable::packTable()
{ // packs only buckets; residuals left behind
  //sanitize();
  UpdateMsg *um;
  int count=0, i;
  SRentry *j;

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
      // um->msgs[count].next = NULL;
      count++;
      j = j->next;
    }
    j=recvs[i].bucket;
    while (j) {
      um->msgs[count] = *j;
      // um->msgs[count].next = NULL;
      count++;
      j = j->next;
    }
  }
  //sanitize();
  return um;
}

void SRtable::addEntries(UpdateMsg *um)
{
  //  sanitize();
  int i, bkt;
  SRentry *entry;

  // first, resize if necessary
  if ((gvtWindow != um->gvtW) || (numBuckets != um->numB) || 
      (offset != um->offset)) {
    int oldNumBuckets = numBuckets;
    gvtWindow = um->gvtW;
    numBuckets = um->numB;
    offset = um->offset;
    //CmiAssert(offset == um->offset);
    // move all elements to residuals
    for (i=0; i<oldNumBuckets; i++) {
      sends[i].emptyOutBucket(residuals, residualsTail, NULL);
      recvs[i].emptyOutBucket(residuals, residualsTail, NULL);
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
  }
  else if (offset != um->offset)  SetOffset(um->offset);

  // now move the new stuff in
  for (i=0; i<um->msgCount; i++) {
    if (recyc) {
      entry = recyc;
      recyc = recyc->next;
      if (!recyc) recycTail = NULL;
      entry->Set(um->msgs[i].timestamp, um->msgs[i].sr, NULL);
    }
    else
      entry = new SRentry(um->msgs[i].timestamp, um->msgs[i].sr, NULL);
    bkt = (um->msgs[i].timestamp - offset)/bktSz;
    inBuckets++;
    if (um->msgs[i].sr == SEND)  sends[bkt].addToBucket(entry);
    else  recvs[bkt].addToBucket(entry);
  }
  //sanitize();
}

void SRtable::shrink()
{ 
  //sanitize();
  // Minimum GVT Window is 8; minimum bucket size is 1
  int oldNumBuckets = numBuckets, i;
  SRentry *tmp;

  if (gvtWindow > 16) gvtWindow = gvtWindow-8;
  else return;
  numBuckets = gvtWindow/8;
  // move all elements to residuals
  for (i=0; i<oldNumBuckets; i++) {
    sends[i].emptyOutBucket(residuals, residualsTail, NULL);
    recvs[i].emptyOutBucket(residuals, residualsTail, NULL);
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
  //sanitize();
}

void SRtable::expand()
{
  //sanitize();
  int oldNumBuckets = numBuckets, i;
  SRentry *tmp;

  if (gvtWindow > MAX_GVT_WINDOW) return;
  gvtWindow = gvtWindow*2;
  numBuckets = gvtWindow/8;
  // move all elements to residuals
  for (i=0; i<oldNumBuckets; i++) {
    sends[i].emptyOutBucket(residuals, residualsTail, NULL);
    recvs[i].emptyOutBucket(residuals, residualsTail, NULL);
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
  //sanitize();
}

void SRtable::dump()
{
  CkPrintf("SRtable::dump: WRITE ME\n");
}

void SRtable::sanitize()
{
  int totalCount=0, bktCount;
  SRentry *current;
  for (int i=0; i<numBuckets; i++) {
    CmiAssert(sends[i].offset == offset+i*bktSz);
    bktCount = 0;
    current = sends[i].bucket;
    CmiAssert((sends[i].bucket && sends[i].bucketTail) || 
	      (!sends[i].bucket && !sends[i].bucketTail));
    while (current) {
      bktCount++;
      CmiAssert((current->timestamp >= offset) && 
		(current->timestamp < offset+gvtWindow));
      CmiAssert(current->sr == 0);
      CmiAssert((current->timestamp >= sends[i].offset) && 
		(current->timestamp < sends[i].offset+bktSz));
      current = current->next;
    }
    CmiAssert(bktCount == sends[i].count);
    totalCount += bktCount;
    bktCount = 0;
    current = recvs[i].bucket;
    CmiAssert((recvs[i].bucket && recvs[i].bucketTail) || 
	      (!recvs[i].bucket && !recvs[i].bucketTail));
    while (current) {
      bktCount++;
      CmiAssert((current->timestamp >= offset) && 
		(current->timestamp < offset+gvtWindow));
      CmiAssert(current->sr == 1);
      CmiAssert((current->timestamp >= recvs[i].offset) && 
		(current->timestamp < recvs[i].offset+bktSz));
     current = current->next;
    }
    CmiAssert(bktCount == recvs[i].count);
    totalCount += bktCount;
  }
  CmiAssert(totalCount == inBuckets);
  CmiAssert((residuals && residualsTail) || (!residuals && !residualsTail));
  CmiAssert((recyc && recycTail) || (!recyc && !recycTail));
}
