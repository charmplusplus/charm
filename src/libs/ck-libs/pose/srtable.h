/// SendRecvTable for POSE GVT calculations
#ifndef SRTABLE_H
#define SRTABLE_H
#include "pose.h"

#define MAX_B 10

class UpdateMsg; // from gvt.h

/// An entry for storing the number of sends and recvs at a timestamp
/** This class is used in SendRecvTable to store residual send/recv data */
class SRentry {
 public:
  /// Timestamp of the message
  POSE_TimeType timestamp;  
  /// The number of messages sent with theTimestamp
  int sends;
  /// The number of messages sent with theTimestamp
  int recvs;
  /// Next SRentry in list
  SRentry *next;
  /// Basic constructor
  /** Initializes all data members */
  SRentry() { 
    timestamp = POSE_UnsetTS; sends = recvs = 0; next = NULL; 
  }
  /// Initializing constructor 1
  /** Initializes timestamp & next w/parameters, sends & recvs to 0 */
  SRentry(POSE_TimeType ts, SRentry *p) { 
    timestamp = ts; sends = recvs = 0; next = p; 
  }
  /// Initializing constructor 2
  /** Initializes timestamp, send/recv count and next
      w/parameters */
  SRentry(POSE_TimeType ts, int sr, SRentry *p) {
    timestamp = ts;  next = p; 
    if (sr == SEND) { sends = 1; recvs = 0; }
    else { sends = 0; recvs = 1; }
  }
  /// Initializing constructor
  /** Initializes timestamp and send/recv count w/parameters */
  SRentry(POSE_TimeType ts, int sr) {
    timestamp = ts;  next = NULL; 
    if (sr == SEND) { sends = 1; recvs = 0; }
    else { sends = 0; recvs = 1; }
  }
  /// Assignment operator
  SRentry& operator=(const SRentry& e) {
    timestamp = e.timestamp;
    sends = e.sends;
    recvs = e.recvs;
    return *this;
  }
  /// Dump data fields
  void dump() {
    if (next)
      CkPrintf("TS:%d #s:%d #r:%d n:!NULL ", timestamp, sends, recvs); 
    else CkPrintf("TS:%d #s:%d #r:%d n:NULL ",timestamp, sends, recvs);
  }
  /// Check validity of data fields
  void sanitize() {
    CmiAssert(timestamp >= POSE_UnsetTS); // should be POSE_UnsetTS or > if initialized
    CmiAssert(sends >= 0);  // cannot be less than zero
    CmiAssert(recvs >= 0);  // cannot be less than zero
    if (next == NULL) return;   // next can be NULL
    // if next != NULL, check if pointer looks valid
    POSE_TimeType test_ts = next->timestamp;
    int test_sendCount = next->sends;
    int test_recvCount = next->recvs;
    SRentry *test_next = next->next;
    next->timestamp = test_ts;
    next->sends = test_sendCount;
    next->recvs = test_recvCount;
    next->next = test_next;
  }
};

/// A table for storing the number of sends and recvs at a timestamp
/** This class is used in GVT to keep track of messages sent/received */
class SRtable {
 public:
  /// Base timestamp to index tables
  /** offset is the current GVT */
  POSE_TimeType offset;
  /// Number of buckets to sort sends/recvs into
  /** Recomputed with each new offset */
  int b;
  /// Size of each bucket
  /** Recomputed with each new offset */
  int size_b;
  /// The buckets to sort sends/recvs into
  /** Only entries [0..b-1] are used */
  SRentry *buckets[MAX_B];
  /// Pointers to the last entry of each bucket
  /** Only entries [0..b-1] are used */
  SRentry *end_bucket[MAX_B];
  /// Error checking on bucket counts
  int sends[MAX_B], recvs[MAX_B], ofSends, ofRecvs;
  /// The overflow bucket
  /** What doesn't fit in buckets goes here */
  SRentry *overflow;
  /// End entry of overflow
  SRentry *end_overflow;
  /// Number of distinct timestamp entries per bucket
  /** This is computed in CompressAndSortBucket */
  int numEntries[MAX_B];
  /// Number of distinct entries in overflow bucket
  /** This is computed in CompressAndSortBucket */
  int numOverflow;
  
  /// Basic constructor
  /** Initializes all data fields */
  SRtable();
  /// Destructor
  ~SRtable() { FreeTable(); }
  /// Initialize table to a minimum size
  void Initialize();
  /// Insert send/recv record sr at timestamp ts
  void Insert(POSE_TimeType ts, int sr); 
  /// Insert an existing SRentry e
  void Insert(SRentry *e); 
  /// Restructure the table according to new GVT estimate and first send/recv
  /** Number of buckets and bucket size are determined from firstTS, and
      entries below newGVTest are discarded. */
  void Restructure(POSE_TimeType newGVTest, POSE_TimeType firstTS, int firstSR);
  /// Compress and pack table into an UpdateMsg and return it
  UpdateMsg *PackTable(POSE_TimeType pvt);
  /// CompressAndSort all buckets with timestamps <= pvt
  void PartialSortTable(POSE_TimeType pvt);
  /// Compress a bucket so all SRentries have unique timestamps and are sorted
  void CompressAndSortBucket(int i, int is_overflow);
  /// Free all buckets and overflows, reset all counts
  void FreeTable();
  /// Dump data fields
  void dump();    
  /// Check validity of data fields
  void sanitize();
  /// Test this class
  void self_test();
};

#endif
