// File: srtable.h
// SendRecvTable is a table that stores timestamps of sends and recvs of all
// events and cancellations.
// Used in gvt.*.  

#ifndef SRTABLE_H
#define SRTABLE_H
#include "pose.h"

class UpdateMsg;

class SRentry { // A record for a single send/recv event
  // for now we wish to keep the timestamp associated with a send/recv,
  // since later we may choose to alter the bktSz in SRtable
 public:
  int sr;         // sr denotes either SEND or RECV
  int timestamp;  // timestamp of the message
  SRentry *next;  // this type is used almost always in linked lists
  SRentry() { timestamp = sr = -1; next = NULL; }
  //~SRentry() { next = NULL; }
  SRentry(int ts, int mt, SRentry *p) { timestamp = ts; sr = mt; next = p; }
  int operator==(const SRentry &obj) const {
    return ((timestamp == obj.timestamp) && (sr == obj.sr));
  }
  SRentry &operator=(const SRentry obj) {
    timestamp = obj.timestamp;
    sr = obj.sr;
    return *this;
  }
  void Set(int ts, int mt, SRentry *p) { timestamp = ts; sr = mt; next = p; }
  void dump() { CkPrintf("TS:%d SR:%d", timestamp, sr); }
};

class SRbucket { // A bucket for holding a range of SRentries
 public:
  int count;        // the number of SRentries in bucket
  int bktSz;        // the range of timestamps to be stored in this bucket
  int offset;       // the minimum timestamp to be stored in this bucket
  SRentry *bucket, *bucketTail;  // the entries in this bucket
  SRbucket() { bucket = NULL;  count = 0; bktSz = -1; offset = 0; }
  ~SRbucket() {
    SRentry *next, *current=bucket;
    while (current) {
      next = current->next;
      delete current;
      current = next;
    }
    bucket = bucketTail = NULL;
    count = 0;
  }
  void initBucket(int sz, int os) { 
    bucket = bucketTail = NULL; count = 0; bktSz = sz;  offset = os; 
  }
  void setBucketOffset(int os) { offset = os; }
  int diffBucket(const SRbucket& bkt) { 
    // return the timestamp at which this bucket and bkt first differ
    // assumes bucket is sorted by non-decreasing timestamp
    int i, result=-1;
    SRentry *myCurrent = bucket, *theirCurrent = bkt.bucket;
    while (myCurrent && theirCurrent) {
      if (myCurrent->timestamp != theirCurrent->timestamp) {
	result = myCurrent->timestamp;
	if (theirCurrent->timestamp < result)
	  result = theirCurrent->timestamp;
	break;
      }
      myCurrent = myCurrent->next;
      theirCurrent = theirCurrent->next;
    }
    if (result != -1)
      return result;
    else if (!myCurrent && theirCurrent)
      return theirCurrent->timestamp;
    else if (!theirCurrent && myCurrent)
      return myCurrent->timestamp;
    else return result;
  }
  void addToBucket(SRentry *p) {
    if (!bucket || (p->timestamp <= bucket->timestamp)) {
      p->next = bucket;
      if (!bucket) bucket = bucketTail = p;
      else bucket = p;
      count++;
    }
    else {
      SRentry *current = bucket;
      while (current->next) {
	if (p->timestamp <= current->next->timestamp) {
	  p->next = current->next;
	  if (!current->next) bucketTail = current->next = p;
	  else current->next = p;
	  count++;
	  return;
	}
	current = current->next;
      }
      p->next = current->next;
      if (!current->next) bucketTail = current->next = p;
      else current->next = p;
      count++;
    }
  }
  int findInBucket(SRentry *p) const {
    SRentry *current=bucket;
    if (count == 0) return 0;
    while (current && (current->timestamp <= p->timestamp)) {
      if (p->timestamp == current->timestamp) return 1;
      current = current->next;
    }
    return 0;
  }
  void emptyOutBucket(SRentry *recyc, SRentry *recycTail, int *recycCount) {
    SRentry *tmp;
    if (bucket) {
      if (recycCount == NULL) { // actually moves bucket contents to residuals
	bucketTail->next = recyc; 
	recyc = bucket;
	if (!recycTail) recycTail = bucketTail;
      }
      else if (*recycCount < 500) { // stores bucket entries for recycling
	*recycCount += count;
	bucketTail->next = recyc; 
	recyc = bucket;
	if (!recycTail) recycTail = bucketTail;
      }
      else { // frees the bucket entries
	tmp = bucket;
	while (tmp) {
	  bucket = bucket->next;
	  delete(tmp);
	  tmp = bucket;
	}
      }
      bucket = bucketTail = NULL;
      count = 0;
    }
  }
};

class SRtable {
 private:
  SRentry *residuals, *residualsTail;  // all other send/recv events
  SRentry *recyc, *recycTail;          // SRentries that can be reused
  int recycCount;
 public:
  SRbucket *sends, *recvs; // send/recv events occurring 
                                // at timestamps between gvt and gvt+gvtWindow
  int gvtWindow, numBuckets;
  int offset;                   // gvt offset
  int bktSz;
  int inBuckets;
  SRtable();                    // basic constructor
  ~SRtable();                   // needed to free up the linked lists
  void SetOffset(int gvt) {     // set GVT offset
    //    CmiAssert(offset <= gvt);
    offset = gvt;
    for (int i=0; i<numBuckets; i++) {
      sends[i].initBucket(bktSz, offset+i*bktSz);
      recvs[i].initBucket(bktSz, offset+i*bktSz);
    }
    //sanitize();
  }
  int TestThreshold() {
    //sanitize();
    if (inBuckets == 0) {
      //CkPrintf("^^^^^^^^^^  EXPAND WINDOW!!!     %d\n", inBuckets);
      //expand();
      return 1;
    }
    else if (inBuckets > TBL_THRESHOLD) {
      //CkPrintf("vvvvvvvvvv  SHRINK WINDOW!!!     %d\n", inBuckets);
      //shrink();
      return -1;
    }
    //CkPrintf("----------  WINDOW FINE     %d\n", inBuckets);
    return 0;
  }
  void Insert(int timestamp, int srSt); // insert s/r at timestamp
  int FindDiff() {
    //sanitize();
    int result;
    for (int i=0; i<numBuckets; i++)
      if ((result = sends[i].diffBucket(recvs[i])) != -1)
	return result;
    return -1;
    //sanitize();
  }
  void PurgeBelow(int ts);      // purge table below timestamp ts
  void FileResiduals();         // try to file each residual event in table
  void FreeTable() {            // reset counters & ptrs; free all
    // Clears all data from the table
    //sanitize();
    SRentry *tmp;
    for (int i=0; i<numBuckets; i++) {
      sends[i].emptyOutBucket(recyc, recycTail, &recycCount);
      recvs[i].emptyOutBucket(recyc, recycTail, &recycCount);
    }
    inBuckets = offset = 0;
    tmp = residuals;
    while (tmp) { 
      residuals = tmp->next;
      delete(tmp);
      tmp = residuals;
    }
    residuals = residualsTail = NULL;
    offset = -1;
    //sanitize();
  }
  UpdateMsg *packTable();
  void addEntries(UpdateMsg *um);
  void shrink();
  void expand();
  void dump();                  // dump the table
  void sanitize();
};

#endif
