// File: srtable.h
// SendRecvTable is a table that stores timestamps of sends and recvs of all
// events and cancellations.
// Used in gvt.*.  

#ifndef SRTABLE_H
#define SRTABLE_H

#include "pose.h"

class UpdateMsg;

class SRentry { // A record for a single send/recv event
 public:
  int timestamp, sr;  // sr denotes either SEND or RECV
  SRentry *next;
  SRentry() { timestamp = sr = -1; next = NULL; }
  //~SRentry() { next = NULL; }
  SRentry(int ts, int srSt, SRentry *p) { 
    timestamp = ts; sr = srSt; next = p;
  }
  int operator==(const SRentry &obj) const {
    return ((timestamp == obj.timestamp) && (sr == obj.sr));
  }
  SRentry &operator=(const SRentry obj) {
    timestamp = obj.timestamp;
    sr = obj.sr;
    return *this;
  }
  void Set(int ts, int srSt, SRentry *p) { 
    timestamp = ts; sr = srSt; next = p;
  }
  void dump() { 
    CkPrintf("TS:%d SR:%d", timestamp, sr);
  }
};

class SRbucket { // A bucket for holding a range of SRentries
 public:
  SRentry *bucket;  // should we sort entries into bucket?
  int count, bktSz, offset;
  SRbucket() { bucket = NULL;  count = 0; bktSz = -1; offset = 0; }
  ~SRbucket() {
    SRentry *next, *current=bucket;
    while (current) {
      next = current->next;
      delete current;
      current = next;
    }
    bucket = NULL;
    count = 0;
  }
  void initBucket(int sz, int os) { 
    bucket = NULL; count = 0; bktSz = sz;  offset = os; 
  }
  int diffBucket(const SRbucket& bkt) { 
    int *myTSarray = new int[bktSz], *theirTSarray = new int[bktSz], i, result=-1;
    SRentry *current = bucket;
    CmiAssert(bktSz > 0);
    for (i=0; i<bktSz; i++) myTSarray[i] = theirTSarray[i] = 0;
    while (current) {
      CmiAssert(current->timestamp - offset >= 0);
      CmiAssert(current->timestamp - offset < bktSz);
      myTSarray[current->timestamp - offset]++;
      current = current->next;
    }
    current = bkt.bucket;
    while (current) {
      CmiAssert(current->timestamp - offset >= 0);
      CmiAssert(current->timestamp - offset < bktSz);
      theirTSarray[current->timestamp - offset]++;
      current = current->next;
    }
    for (i=0; i<bktSz; i++) 
      if (myTSarray[i] != theirTSarray[i]) {
	result = i+offset;
	break;
      }
    delete[] myTSarray;
    delete[] theirTSarray;
    return result;
  }
  void addToBucket(SRentry *p) {
    CmiAssert((p->timestamp >= offset) && (p->timestamp < offset+bktSz));
    CmiAssert((p->sr == 0) || (p->sr == 1));
    p->next = bucket;
    bucket = p;
    count++;
  }
  int findInBucket(SRentry *p) const {
    SRentry *current=bucket;
    if (count == 0) return 0;
    while (current) {
      if (p->timestamp == current->timestamp) return 1;
      current = current->next;
    }
    return 0;
  }
  void emptyOutBucket(SRentry *recyc) {
    SRentry *next, *current=bucket;
    if (count == 0) return;
    while (current) {
      next = current->next;
      current->next = recyc;
      recyc = current;
      current = next;
    }
    bucket = NULL;
    count = 0;
  }
};

class SRtable {
 private:
  SRentry *residuals;           // all other send/recv events
  SRentry *recyc;
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
    CmiAssert(offset <= gvt);
    offset = gvt;
    for (int i=0; i<numBuckets; i++) {
      sends[i].initBucket(bktSz, offset+i*bktSz);
      recvs[i].initBucket(bktSz, offset+i*bktSz);
    }
  }
  int TestThreshold() {
    if (inBuckets == 0) {
      //CkPrintf("^^^^^^^^^^  EXPAND WINDOW!!!     %d\n", inBuckets);
      expand();
      return 1;
    }
    else if (inBuckets > TBL_THRESHOLD) {
      //CkPrintf("vvvvvvvvvv  SHRINK WINDOW!!!     %d\n", inBuckets);
      shrink();
      return -1;
    }
    //CkPrintf("----------  WINDOW FINE     %d\n", inBuckets);
    return 0;
  }
  void Insert(int timestamp, int srSt); // insert s/r at timestamp
  int FindDifferenceTimestamp(SRtable *t) {
    // return timestamp with first difference
    int result;
    for (int i=0; i<numBuckets; i++)
      if ((result = sends[i].diffBucket(recvs[i])) > -1)
	return result;
    return gvtWindow + offset;
  }
  void PurgeBelow(int ts);      // purge table below timestamp ts
  void FileResiduals();         // try to file each residual event in table
  void FreeTable();             // reset counters & ptrs; free all
  UpdateMsg *packTable();
  void addEntries(UpdateMsg *um);
  void shrink();
  void expand();
  void dump();                  // dump the table
  void sanitize();
};

#endif
