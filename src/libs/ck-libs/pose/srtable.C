/// SendRecvTable for POSE GVT calculations
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"  // Needed for Charm++ output
#include "srtable.h"
#include "gvt.h"

/// Basic constructor
SRtable::SRtable() 
{ 
  offset = b = size_b = numOverflow = 0;
  for (int i=0; i<MAX_B; i++) {
    buckets[i] = NULL;
    numEntries[i] = 0;
  }
  overflow = NULL;
}

/// Initialize table to a minimum size
void SRtable::Initialize()
{
  offset = 0; b = MAX_B; size_b = 1;
  //sanitize();
}

/// Insert send/recv record sr at timestamp ts
/* NOTE: buckets roughly ordered with decreasing timestamps */
void SRtable::Insert(int ts, int sr)
{
  //sanitize();
  CmiAssert(ts >= offset);
  CmiAssert((sr == 0) || (sr == 1));
  int destBkt = (int)(((float)(ts-offset))/((float)size_b));  // which bucket?
  SRentry *e;
  if (destBkt >= b) { // put in overflow bucket
    e = new SRentry(ts, sr, overflow);
    overflow = e;
  }
  else { // put in buckets[destBkt]
    e = new SRentry(ts, sr, buckets[destBkt]);
    buckets[destBkt] = e;
  }
  //sanitize();
}

/// Insert an existing SRentry e
void SRtable::Insert(SRentry *e)
{
  //sanitize();
  CmiAssert(e != NULL);
  CmiAssert(e->timestamp >= offset);
  int destBkt = (int)(((float)(e->timestamp-offset))/((float)size_b));
  if (destBkt >= b) { // put in overflow bucket
    e->next = overflow;
    overflow = e;
  }
  else { // put in buckets[destBkt]
    e->next = buckets[destBkt];
    buckets[destBkt] = e;
  }
  //sanitize();
}

/// Restructure the table according to new GVT estimate and first send/recv
/** Number of buckets and bucket size are determined from firstTS, and
    entries below newGVTest are discarded. */
void SRtable::Restructure(int newGVTest, int firstTS, int firstSR)
{
  //sanitize();
  //CkPrintf("SRtable::Restructure(%d, %d, %d)\n", newGVTest, firstTS, firstSR);
  // Backup the table to make new one in its place
  int b_old = b, size_b_old = size_b, offset_old = offset;
  SRentry *buckets_old[MAX_B], *overflow_old = overflow, *tmp;
  for (int i=0; i<b_old; i++) {
    buckets_old[i] = buckets[i];
    buckets[i] = NULL;
  }
  overflow = NULL;
  // Build new table
  offset = newGVTest;
  b = firstTS - offset;
  if ((b > MAX_B) || (b == 0)) b = MAX_B;
  size_b = (int)(1.0 + (((float)(firstTS - offset))/((float)b)));

  for (int j=0; j<b_old; j++) {
    tmp = buckets_old[j];
    while (tmp) {
      buckets_old[j] = tmp->next;
      if (tmp->timestamp < offset) delete tmp;
      else Insert(tmp);
      tmp = buckets_old[j];
    }
  }
  tmp = overflow_old;
  while (tmp) {
    overflow_old = tmp->next;
    if (tmp->timestamp < offset) delete tmp;
    else Insert(tmp);
    tmp = overflow_old;
  }
  if (firstSR != -1) Insert(firstTS, firstSR);
  //sanitize();
}

/// Compress and pack table into an UpdateMsg and return it
UpdateMsg *SRtable::PackTable(int pvt)
{
  //sanitize();
  int packSize = 0, nEntries = 0, entryIdx = 0, nBkts = 0;
  int destBkt = (int)(((float)(pvt-offset))/((float)size_b));  // which bucket?
  SRentry *tmp;

  PartialSortTable(pvt);
  nBkts = destBkt;
  if (destBkt > b) { 
    nEntries += numOverflow;
    nBkts = b-1;
  }
  for (int i=0; i<=nBkts; i++) nEntries += numEntries[i];

  packSize = nEntries * sizeof(SRentry);
  UpdateMsg *um = new (packSize, 0) UpdateMsg;
  for (int j=0; j<=nBkts; j++) {
    tmp = buckets[j];
    while (tmp) {
      um->SRs[entryIdx] = *tmp;
      entryIdx++;
      tmp = tmp->next;
    }
  }
  if (destBkt > b) {
    tmp = overflow;
    while (tmp && (tmp->timestamp < pvt)) {
      um->SRs[entryIdx] = *tmp;
      entryIdx++;
      tmp = tmp->next;
    }
  }
  CkAssert(entryIdx <= nEntries);
  um->numEntries = entryIdx;
  //sanitize();
  return um;
}

/// CompressAndSort all buckets with timestamps <= pvt
void SRtable::PartialSortTable(int pvt)
{
  //sanitize();
  int sortall = 0, sortTo,
    destBkt = (int)(((float)(pvt-offset))/((float)size_b));  // which bucket?
  sortTo = destBkt;
  if (destBkt >= b) {
    sortall = 1; 
    sortTo = b-1;
  }
  for (int i=0; i<=sortTo; i++) CompressAndSortBucket(i, 0);
  if (sortall) CompressAndSortBucket(b, 1);
  //sanitize();
}

/// Compress a bucket so all SRentries have unique timestamps and are sorted
void SRtable::CompressAndSortBucket(int i, int is_overflow)
{
  //sanitize();
  SRentry *tmp, *e, *newBucket = NULL, *lastInserted = NULL;
  int nEntries = 0;
  if (is_overflow) tmp = overflow;
  else tmp = buckets[i];
  while (tmp) {
    e = tmp;
    tmp = tmp->next;
    // insert e in newBucket
    if (!newBucket) {
      newBucket = lastInserted = e;
      e->next = NULL;
      nEntries++;
    }
    else if (lastInserted->timestamp == e->timestamp) {
      lastInserted->sends += e->sends;
      lastInserted->recvs += e->recvs;
      delete e;
    }
    else if (lastInserted->timestamp < e->timestamp) {
      while (lastInserted->next && 
	     (lastInserted->next->timestamp < e->timestamp))
	lastInserted = lastInserted->next;
      if (lastInserted->next) {
	if (lastInserted->next->timestamp == e->timestamp) {
	  lastInserted->next->sends += e->sends;
	  lastInserted->next->recvs += e->recvs;
	  lastInserted = lastInserted->next;
	  delete e;
	}
	else {
	  e->next = lastInserted->next;
	  lastInserted->next = e;
	  lastInserted = e;
	  nEntries++;
	}
      }
      else {
	lastInserted->next = e;
	e->next = NULL;
	lastInserted = e;
	nEntries++;
      }
    }
    else if (newBucket->timestamp > e->timestamp) {
      e->next = newBucket;
      newBucket = lastInserted = e;
      nEntries++;
    }
    else if (newBucket->timestamp == e->timestamp) {
      newBucket->sends += e->sends;
      newBucket->recvs += e->recvs;
      delete e;
    }
    else {
      lastInserted = newBucket;
      while (lastInserted->next && 
	     (lastInserted->next->timestamp < e->timestamp))
	lastInserted = lastInserted->next;
      if (lastInserted->next) {
	if (lastInserted->next->timestamp == e->timestamp) {
	  lastInserted->next->sends += e->sends;
	  lastInserted->next->recvs += e->recvs;
	  lastInserted = lastInserted->next;
	  delete e;
	}
	else {
	  e->next = lastInserted->next;
	  lastInserted->next = e;
	  lastInserted = e;
	  nEntries++;
	}
      }
      else {
	lastInserted->next = e;
	e->next = NULL;
	lastInserted = e;
	nEntries++;
      }
    }
  }
  if (is_overflow) { 
    overflow = newBucket;
    numOverflow = nEntries;
  }
  else { 
    buckets[i] = newBucket;
    numEntries[i] = nEntries;
  }
  //sanitize();
}

/// Free all buckets and overflows, reset all counts
void SRtable::FreeTable() 
{
  //sanitize();
  SRentry *tmp;
  for (int i=0; i<b; i++) {
    tmp = buckets[i];
    while (tmp) { 
      buckets[i] = tmp->next;
      delete(tmp);
      tmp = buckets[i];
    }
  }
  tmp = overflow;
  while (tmp) { 
    overflow = tmp->next;
    delete(tmp);
    tmp = overflow;
  }
  offset = b = size_b = 0;
}

/// Dump data fields
void SRtable::dump()
{
  SRentry *tmp;
  CkPrintf("\nSRtable: offset=%d b=%d size_b=%d\n", offset, b, size_b);
  for (int i=0; i<b; i++) {
    tmp = buckets[i];
    CkPrintf("... Bucket[%d]: ", i);
    while (tmp) { 
      tmp->dump();
      tmp = tmp->next;
    }
    CkPrintf("\n");
  }
  tmp = overflow;
  CkPrintf("... Overflow: ");
  while (tmp) {
    tmp->dump();
    tmp = tmp->next;
  }
  CkPrintf("\n");
}

/// Check validity of data fields
void SRtable::sanitize()
{
  int bktMin, bktMax;
  SRentry *tmp;
  CmiAssert(offset > -1);
  CmiAssert((b>-1) && (b <= MAX_B));
  CmiAssert(size_b > -1);
  for (int i=0; i<b; i++) {
    tmp = buckets[i];
    bktMin = i*size_b + offset;
    bktMax = i*size_b + (size_b-1) + offset;
    while (tmp) {
      CkAssert((tmp->timestamp >= bktMin) && (tmp->timestamp <= bktMax));
      tmp->sanitize();
      tmp = tmp->next;
    }
  }
  tmp = overflow;
  while (tmp) {
    CkAssert(tmp->timestamp >= b*size_b + offset);
    tmp->sanitize();
    tmp = tmp->next;
  }
}
