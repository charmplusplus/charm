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
  gvtWindow = 512;
  availableBuckets = numBuckets = 64;
  residuals = residualsTail = recyc = recycTail = NULL;
  recycCount = 0;
  inBuckets = 0;
  offset = 0;
  bktSz = gvtWindow / numBuckets;
  sends = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  recvs = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
  for (int i=0; i<numBuckets; i++) {
    sends[i].initBucket(bktSz, offset+i*bktSz);
    recvs[i].initBucket(bktSz, offset+i*bktSz);
  }
  offset = -1;
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
  SRentry *entry;
  CmiAssert(timestamp >= offset);
  if (offset == -1) offset = 0;
  //sanitize();
  if (timestamp >= offset+gvtWindow) {
    if (recyc) {
      entry = recyc;
      recyc = recyc->next;
      if (!recyc) recycTail = NULL;
      recycCount--;
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
      recycCount--;
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
    if (current->timestamp >= offset+gvtWindow) { // entry not in window
      // put back into residuals
      current->next = residuals;
      if (residuals) residuals = current;
      else residuals = residualsTail = current;
    }
    else { // entry in window; place it in a bucket
      bkt = (current->timestamp - offset)/bktSz;
      inBuckets++;
      if (current->sr == SEND)  sends[bkt].addToBucket(current);
      else  recvs[bkt].addToBucket(current);
    }
  }
  //sanitize();
}

UpdateMsg *SRtable::packTable()
{ // packs entries with two earliest timestamps from buckets into an UpdateMsg
  UpdateMsg *um = new UpdateMsg;

  //sanitize();
  FindEarliest(&(um->earlyTS), &(um->earlySends), &(um->earlyRecvs), 
	       &(um->nextTS), &(um->nextSends), &(um->nextRecvs));
  return um;
}

void SRtable::shrink()
{ // Assumes bucket size is 8; currently not flexible
  // This code shrinks by a single bucket
  int oldNumBuckets = numBuckets, i;
  SRentry *tmp;

  //sanitize();
  if (gvtWindow > 16) gvtWindow = gvtWindow-8;
  else return;
  numBuckets = gvtWindow/8;
  // move elements from last bucket to residuals
  inBuckets = inBuckets - sends[oldNumBuckets-1].count - recvs[oldNumBuckets-1].count;
  sends[oldNumBuckets-1].emptyOutBucket(residuals, residualsTail, NULL);
  recvs[oldNumBuckets-1].emptyOutBucket(residuals, residualsTail, NULL);
  //sanitize();
}

void SRtable::expand()
{ // Assumes bucket size is 8; currently not flexible
  // This code expands by a single bucket
  int oldNumBuckets = numBuckets, i;
  SRentry *tmp;

  //sanitize();
  if (gvtWindow > MAX_GVT_WINDOW) return;
  gvtWindow = gvtWindow*2;
  numBuckets = gvtWindow/8;
  if (numBuckets > availableBuckets) { // need to actually expand table
    // move all elements to residuals
    for (i=0; i<oldNumBuckets; i++) {
      sends[i].emptyOutBucket(residuals, residualsTail, NULL);
      recvs[i].emptyOutBucket(residuals, residualsTail, NULL);
    }
    // free and re-malloc arrays (fastest for expand -- avoids copy)
    free(sends);
    free(recvs);
    inBuckets = 0;
    sends = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
    recvs = (SRbucket *)malloc(numBuckets*sizeof(SRbucket));
    SetOffset(offset);
  }
  FileResiduals();
  //sanitize();
}

void SRtable::dump()
{
  CkPrintf("SRtable::dump():\n");
  CkPrintf("OFFSET=%d #inBuckets=%d gvtWindow=%d numBuckets=%d bktSz=%d\n",
	   offset, inBuckets, gvtWindow, numBuckets, bktSz);
  if (residuals) CkPrintf("Residuals is non-empty.  ");
  else CkPrintf("Residuals is empty.  ");
  if (recyc) CkPrintf("Recyc is non-empty (%d entries).\n", recycCount);
  else CkPrintf("Recyc is empty.\n");
  for (int i=0; i<numBuckets; i++) sends[i].dump();
  for (int j=0; j<numBuckets; j++) recvs[j].dump();    
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
