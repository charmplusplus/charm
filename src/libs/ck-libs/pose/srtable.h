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
  void dump() {
    CkPrintf("bkt: Count=%d bktSz=%d offset=%d\n", count, bktSz, offset);
    SRentry *tmp = bucket;
    while (tmp) {
      CkPrintf("  %d", tmp->timestamp);
      tmp = tmp->next;
    }
    if (bucket) CkPrintf("\n");
  }
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
  void emptyOutBucket(SRentry *&recyc, SRentry *&recycTail, int *recycCount) {
    SRentry *tmp;
    if (bucket) {
      if (recycCount == NULL) { // actually moves bucket contents to residuals
	// recyc is actually a pointer to residuals
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
  int recycCount, availableBuckets;
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
  void Insert(int timestamp, int srSt); // insert s/r at timestamp
  void FindEarliest(int *eTS, int *eS, int *eR, int *nTS, int *nS, int *nR) {
    //sanitize();
    int early1 = -1, early2 = -1;
    SRentry *tmp, *nTmp1=NULL, *nTmp2=NULL;
    if (inBuckets == 0) {
      *eTS = offset + gvtWindow;
      *nTS = *eTS + 1;
      *eS = *eR = *nS = *nR = 0;
    }
    for (int i=0; i<numBuckets; i++) {
      if (sends[i].count > 0) {
	early1 = sends[i].bucket->timestamp;
	if ((recvs[i].count > 0) && 
	    ((early1 > recvs[i].bucket->timestamp) || (early1 == -1)))
	  early1 = recvs[i].bucket->timestamp;
      }
      else if (recvs[i].count > 0) 
	early1 = recvs[i].bucket->timestamp;
      if (early1 != -1) { // earliest has been found!
	// count S/R w/TS early1: first, the sends
	*eTS = early1;
	tmp = sends[i].bucket;
	*eS = 0;
	while (tmp && (tmp->timestamp == early1)) {
	  (*eS)++;
	  tmp = tmp->next;
	}
	if (tmp) { // start here for early2
	  early2 = tmp->timestamp;
	  nTmp1 = tmp;
	}
	// now count the recvs
	tmp = recvs[i].bucket;
	*eR = 0;
	while (tmp && (tmp->timestamp == early1)) {
	  (*eR)++;
	  tmp = tmp->next;
	}
	if (tmp && ((tmp->timestamp < early2) || (early2 == -1))) { 
	  early2 = tmp->timestamp;
	  nTmp2 = tmp;
	}
	// find early2
	if (early2 == -1) { // early2 is not in this bucket
	  for (int j=i+1; j<numBuckets; j++) {
	    if (sends[j].count > 0) {
	      early2 = sends[j].bucket->timestamp;
	      if ((recvs[j].count > 0) && 
		  ((early2 > recvs[j].bucket->timestamp) || (early2 == -1)))
		early2 = recvs[j].bucket->timestamp;
	    }
	    else if (recvs[j].count > 0) 
	      early2 = recvs[j].bucket->timestamp;
	    if (early2 != -1) { // nextEarliest found
	      *nTS = early2;
	      // count S/R w/TS early2: first, the sends
	      tmp = sends[j].bucket;
	      *nS = 0;
	      while (tmp && (tmp->timestamp == early2)) {
		(*nS)++;
		tmp = tmp->next;
	      }
	      // now count the recvs
	      tmp = recvs[j].bucket;
	      *nR = 0;
	      while (tmp && (tmp->timestamp == early2)) {
		(*nR)++;
		tmp = tmp->next;
	      }
	      return;
	    }
	  }
	  // no second earliest timestamp found
	  *nTS = offset + gvtWindow;
	  *nS = *nR = 0;
	}
	else { // early2 is in this bucket
	  *nTS = early2;
	  // count S/R w/TS early2: first, the sends
	  *nS = 0;
	  while (nTmp1 && (nTmp1->timestamp == early2)) {
	    (*nS)++;
	    nTmp1 = nTmp1->next;
	  }
	  // now count the recvs
	  *nR = 0;
	  while (nTmp2 && (nTmp2->timestamp == early2)) {
	    (*nR)++;
	    nTmp2 = nTmp2->next;
	  }
	}
	return;
      }
    }
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
  }
  UpdateMsg *packTable();
  void shrink();
  void expand();
  void dump();                  // dump the table
  void sanitize();
};

#endif
