/// SendRecvTable for POSE GVT calculations
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "charm++.h"  // Needed for Charm++ output
#include "srtable.h"
#include "gvt.h"

//#define SR_SANITIZE 1

/// Basic constructor
SRtable::SRtable() 
{ 
  register int i;
  offset = b = size_b = numOverflow = 0;
  for (i=0; i<MAX_B; i++) {
    buckets[i] = end_bucket[i] = NULL;
    numEntries[i] = sends[i] = recvs[i] = 0;
  }
  overflow = end_overflow = NULL;
  ofSends = ofRecvs = 0;
}

/// Initialize table to a minimum size
void SRtable::Initialize()
{
  offset = 0; b = MAX_B; size_b = 1;
#ifdef SR_SANITIZE
  sanitize();
#endif
}

/// Insert send/recv record sr at timestamp ts
/* NOTE: buckets roughly ordered with decreasing timestamps */
void SRtable::Insert(POSE_TimeType ts, int sr)
{
#ifdef SR_SANITIZE
  sanitize();
#endif
  CmiAssert(ts >= offset);
  CmiAssert((sr == 0) || (sr == 1));
  int destBkt = (ts-offset)/size_b;  // which bucket?
  SRentry *e = new SRentry(ts, sr, NULL);
  if (destBkt >= b) { // put in overflow bucket
    if (overflow) {
      if (end_overflow->timestamp == ts) { // an entry at that timestamp exists
	if (sr == SEND) end_overflow->sends++;
	else end_overflow->recvs++;
	delete e;
      }
      else { // no entry with that timestamp is handy
	end_overflow->next = e;
	end_overflow = e;
      }
    }
    else overflow = end_overflow = e;
    if (sr == SEND) ofSends++;
    else ofRecvs++;
  }
  else { // put in buckets[destBkt]
    if (buckets[destBkt]) {
      if (end_bucket[destBkt]->timestamp == ts) { 
	// an entry at that timestamp exists
	if (sr == SEND) end_bucket[destBkt]->sends++;
	else end_bucket[destBkt]->recvs++;
	delete e;
      }
      else { // no entry with that timestamp is handy
	end_bucket[destBkt]->next = e;
	end_bucket[destBkt] = e;
      }
    }
    else buckets[destBkt] = end_bucket[destBkt] = e;
    if (sr == SEND) sends[destBkt]++;
    else recvs[destBkt]++;
  }
#ifdef SR_SANITIZE
  sanitize();
#endif
}

/// Insert an existing SRentry e
void SRtable::Insert(SRentry *e)
{
#ifdef SR_SANITIZE
  sanitize();
#endif
  CmiAssert(e != NULL);
  CmiAssert(e->timestamp >= offset);
  int destBkt = (e->timestamp-offset)/size_b;
  e->next = NULL;
  if (destBkt >= b) { // put in overflow bucket
    if (overflow) {
      end_overflow->next = e;
      end_overflow = e;
    }
    else overflow = end_overflow = e;
    ofSends += e->sends;
    ofRecvs += e->recvs;
  }
  else { // put in buckets[destBkt]
    if (buckets[destBkt]) {
      end_bucket[destBkt]->next = e;
      end_bucket[destBkt] = e;
    }
    else buckets[destBkt] = end_bucket[destBkt] = e;
    sends[destBkt] += e->sends;
    recvs[destBkt] += e->recvs;
  }
#ifdef SR_SANITIZE
  sanitize();
#endif
}

/// Restructure the table according to new GVT estimate and first send/recv
/** Number of buckets and bucket size are determined from firstTS, and
    entries below newGVTest are discarded. */
void SRtable::Restructure(POSE_TimeType newGVTest, POSE_TimeType firstTS, 
			  int firstSR)
{
#ifdef SR_SANITIZE
  sanitize();
#endif
  register int i;
  // Backup the table to make new one in its place
  int sumS_old=0, sumR_old=0, sumS=0, sumR=0;
  int keepBkt = (newGVTest-offset)/size_b;
  if (keepBkt < b)
    for (i=keepBkt; i<b; i++)
      CompressAndSortBucket(i, 0);
  CompressAndSortBucket(b, 1);
  int b_old = b, size_b_old = size_b, offset_old = offset;
  SRentry *buckets_old[MAX_B], *end_bucket_old[MAX_B], 
    *overflow_old, *end_overflow_old, *tmp;
  int sends_old[MAX_B], recvs_old[MAX_B], ofSends_old, ofRecvs_old;
  for (i=0; i<b_old; i++) {
    buckets_old[i] = buckets[i];
    end_bucket_old[i] = end_bucket[i];
    sends_old[i] = sends[i];
    recvs_old[i] = recvs[i];
  }
  overflow_old = overflow;
  end_overflow_old = end_overflow;
  ofSends_old = ofSends;
  ofRecvs_old = ofRecvs;
  for (i=0; i<b; i++) { sumS_old += sends_old[i]; sumR_old += recvs_old[i]; }
  sumS_old += ofSends_old; sumR_old += ofRecvs_old;
  // Build new table
  overflow = end_overflow = NULL;
  ofSends = ofRecvs = 0;
  offset = newGVTest;
  POSE_TimeType d = firstTS - offset;
  if (d < 0) d = 0;
  if ((d > MAX_B) || (d == 0)) b = MAX_B;
  size_b = 1 + d/b;
  for (i=0; i<b; i++) {
    buckets[i] = end_bucket[i] = NULL;
    sends[i] = recvs[i] = 0;
  }

#ifdef SR_SANITIZE
  sanitize();
#endif

  if (keepBkt < b_old) {
    for (i=0; i<keepBkt; i++) { // throw all these away
      tmp = buckets_old[i];
      while (tmp) {
	buckets_old[i] = tmp->next;
	delete tmp;
	tmp = buckets_old[i];
      }
      sumS_old -= sends_old[i];
      sumR_old -= recvs_old[i];
      sends_old[i] = recvs_old[i] = 0;
    }
#ifdef SR_SANITIZE
    sanitize();
#endif
    // carefully sort through this bucket
    tmp = buckets_old[keepBkt];
    while (tmp && (tmp->timestamp < offset)) {
      buckets_old[keepBkt] = tmp->next;
      sumS_old -= tmp->sends;
      sumR_old -= tmp->recvs;
      sends_old[keepBkt] -= tmp->sends;
      recvs_old[keepBkt] -= tmp->recvs;
      delete tmp;
      tmp = buckets_old[keepBkt];
    }
    if (tmp) 
      MapToBuckets(buckets_old[keepBkt], end_bucket_old[keepBkt], 
		   &sends_old[keepBkt], &recvs_old[keepBkt]);
#ifdef SR_SANITIZE
    sanitize();
#endif
    for (i=keepBkt+1; i<b_old; i++) { // keep all of these
      if (buckets_old[i])
	MapToBuckets(buckets_old[i], end_bucket_old[i], 
		     &sends_old[i], &recvs_old[i]);
    }
#ifdef SR_SANITIZE
    sanitize();
#endif
    if (overflow_old)     // keep all of overflow
      MapToBuckets(overflow_old, end_overflow_old, &ofSends_old, 
		   &ofRecvs_old);
#ifdef SR_SANITIZE
    sanitize();
#endif
  }
  else { // throw all buckets away
    for (i=0; i<b_old; i++) {
      tmp = buckets_old[i];
      while (tmp) {
	buckets_old[i] = tmp->next;
	delete tmp;
	tmp = buckets_old[i];
      }
      sumS_old -= sends_old[i];
      sumR_old -= recvs_old[i];
      sends_old[i] = recvs_old[i] = 0;
    }
#ifdef SR_SANITIZE
    sanitize();
#endif
    // carefully sort through overflow
    tmp = overflow_old;
    while (tmp && (tmp->timestamp < offset)) {
      overflow_old = tmp->next;
      sumS_old -= tmp->sends;
      sumR_old -= tmp->recvs;
      ofSends_old -= tmp->sends;
      ofRecvs_old -= tmp->recvs;
      delete tmp;
      tmp = overflow_old;
    }
    if (overflow_old) 
      MapToBuckets(overflow_old, end_overflow_old, &ofSends_old, &ofRecvs_old);
#ifdef SR_SANITIZE
    sanitize();
#endif
  }
  
  for (i=0; i<b; i++) { sumS += sends[i]; sumR += recvs[i]; }
  sumS += ofSends; sumR += ofRecvs;

  if ((sumS_old != sumS) || (sumR_old != sumR))
    CkPrintf("Old sends=%d; old recvs=%d; new sends=%d, new recvs=%d\n", 
	     sumS_old, sumR_old, sumS, sumR);

  if (firstSR != -1) Insert(firstTS, firstSR);
#ifdef SR_SANITIZE
  sanitize();
#endif
}

void SRtable::MapToBuckets(SRentry *bkt, SRentry *endBkt, int *s, 
			   int *r)
{
  int destBkt1 = (bkt->timestamp-offset)/size_b;  
  int destBkt2 = (endBkt->timestamp-offset)/size_b;
  SRentry *tmp = bkt;
  CkAssert(destBkt1 <= destBkt2);
  while (destBkt1 != destBkt2) {
    CkAssert(destBkt1 < destBkt2);
    if (destBkt1 >= b) break;
    bkt = tmp->next;
    (*s) -= tmp->sends;
    (*r) -= tmp->recvs;
#ifdef SR_SANITIZE
  sanitize();
#endif
    if (end_bucket[destBkt1]) { // bucket is non-empty
      if (end_bucket[destBkt1]->timestamp == tmp->timestamp) {
	end_bucket[destBkt1]->sends += tmp->sends;
	end_bucket[destBkt1]->recvs += tmp->recvs;
	sends[destBkt1] += tmp->sends;
	recvs[destBkt1] += tmp->recvs;
	delete tmp;
#ifdef SR_SANITIZE
  sanitize();
#endif
      }
      else {
#ifdef SR_SANITIZE
  sanitize();
#endif
	end_bucket[destBkt1]->next = tmp;
	end_bucket[destBkt1] = tmp;
	tmp->next = NULL;
	sends[destBkt1] += tmp->sends;
	recvs[destBkt1] += tmp->recvs;
#ifdef SR_SANITIZE
  sanitize();
#endif
      }
    }
    else { // bucket is empty
      buckets[destBkt1] = end_bucket[destBkt1] = tmp;
      tmp->next = NULL;
      sends[destBkt1] = tmp->sends;
      recvs[destBkt1] = tmp->recvs;
#ifdef SR_SANITIZE
  sanitize();
#endif
    }
    tmp = bkt;
    destBkt1 = (bkt->timestamp-offset)/size_b;
  }
  if (destBkt1 >= b) {
    if (end_overflow) { // overflow is non-empty
      end_overflow->next = bkt;
      end_overflow = endBkt;
      ofSends += (*s);
      ofRecvs += (*r);
#ifdef SR_SANITIZE
  sanitize();
#endif
    }
    else { // overflow is empty
      overflow = bkt;
      end_overflow = endBkt;
      ofSends = (*s);
      ofRecvs = (*r);
#ifdef SR_SANITIZE
  sanitize();
#endif
    }
  }
  else { // destBkt1 == destBkt2
    if (end_bucket[destBkt1]) { // bucket is non-empty
      end_bucket[destBkt1]->next = bkt;
      end_bucket[destBkt1] = endBkt;
      sends[destBkt1] += (*s);
      recvs[destBkt1] += (*r);
#ifdef SR_SANITIZE
  sanitize();
#endif
    }
    else { // bucket is empty
      buckets[destBkt1] = bkt; 
      end_bucket[destBkt1] = endBkt;
      sends[destBkt1] = (*s);
      recvs[destBkt1] = (*r);
#ifdef SR_SANITIZE
  sanitize();
#endif
    }
  }
}

/// Compress and pack table into an UpdateMsg and return it
UpdateMsg *SRtable::PackTable(POSE_TimeType pvt)
{
#ifdef SR_SANITIZE
  sanitize();
#endif
  register int i;
  int packSize = 0, nEntries = 0, entryIdx = 0, nBkts = 0;
  int destBkt;  // which bucket?
  SRentry *tmp;

  if (pvt == -1) destBkt = b-1;
  else destBkt = (pvt-offset)/size_b;

  SortTable();
  nBkts = destBkt;
  if (destBkt > b) { 
    nEntries += numOverflow;
    nBkts = b-1;
  }
  for (i=0; i<=nBkts; i++) nEntries += numEntries[i];

  packSize = nEntries * sizeof(SRentry);
  UpdateMsg *um = new (packSize, 0) UpdateMsg;
  for (i=0; i<=nBkts; i++) {
    tmp = buckets[i];
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
#ifdef SR_SANITIZE
  sanitize();
#endif
  return um;
}

/// CompressAndSort all buckets with timestamps <= pvt
void SRtable::SortTable()
{
#ifdef SR_SANITIZE
  sanitize();
#endif
  register int i;
  for (i=0; i<b; i++) CompressAndSortBucket(i, 0);
  CompressAndSortBucket(b, 1);
#ifdef SR_SANITIZE
  sanitize();
#endif
}

/// Compress a bucket so all SRentries have unique timestamps and are sorted
void SRtable::CompressAndSortBucket(int i, int is_overflow)
{
#ifdef SR_SANITIZE
  sanitize();
#endif
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
  SRentry *lastEntry = newBucket;
  if (lastEntry) while (lastEntry->next) lastEntry = lastEntry->next;
  if (is_overflow) { 
    overflow = newBucket;
    end_overflow = lastEntry;
    numOverflow = nEntries;
  }
  else { 
    buckets[i] = newBucket;
    end_bucket[i] = lastEntry;
    numEntries[i] = nEntries;
  }
#ifdef SR_SANITIZE
  sanitize();
#endif
}

/// Free all buckets and overflows, reset all counts
void SRtable::FreeTable() 
{
#ifdef SR_SANITIZE
  sanitize();
#endif
  register int i;
  SRentry *tmp;
  for (i=0; i<b; i++) {
    tmp = buckets[i];
    while (tmp) { 
      buckets[i] = tmp->next;
      delete(tmp);
      tmp = buckets[i];
    }
    numEntries[i] = sends[i] = recvs[i] = 0;
  }
  tmp = overflow;
  while (tmp) { 
    overflow = tmp->next;
    delete(tmp);
    tmp = overflow;
  }
  offset = b = size_b = 0;
  ofSends = ofRecvs = 0;
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

/// Check validity of data field
void SRtable::sanitize()
{
  int bktMin, bktMax, sCount, rCount;
  SRentry *tmp;
  CmiAssert(offset > -1);
  CmiAssert((b>-1) && (b <= MAX_B));
  CmiAssert(size_b > -1);
  for (int i=0; i<b; i++) {
    sCount = rCount = 0;
    tmp = buckets[i];
    bktMin = i*size_b + offset;
    bktMax = i*size_b + (size_b-1) + offset;
    if (!tmp) CkAssert(!end_bucket[i]);
    while (tmp) {
      CkAssert((tmp->timestamp >= bktMin) && (tmp->timestamp <= bktMax));
      if (!tmp->next) CkAssert(end_bucket[i] == tmp);
      tmp->sanitize();
      sCount += tmp->sends;
      rCount += tmp->recvs;
      tmp = tmp->next;
    }
    CkAssert(sCount == sends[i]);
    CkAssert(rCount == recvs[i]);
  }
  tmp = overflow;
  sCount = rCount = 0;
  if (!tmp) CkAssert(!end_overflow);
  while (tmp) {
    CkAssert(tmp->timestamp >= b*size_b + offset);
    if (!tmp->next) CkAssert(end_overflow == tmp);
    tmp->sanitize();
    sCount += tmp->sends;
    rCount += tmp->recvs;
    tmp = tmp->next;
  }
  CkAssert(sCount == ofSends);
  CkAssert(rCount == ofRecvs);
}
