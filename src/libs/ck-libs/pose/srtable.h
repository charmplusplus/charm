/// SendRecvTable for POSE GVT calculations
#ifndef SRTABLE_H
#define SRTABLE_H
#include "pose.h"

class UpdateMsg; // from gvt.h

/// An entry for storing the number of sends and recvs at a timestamp
/** This class is used in SendRecvTable to store residual send/recv data */
class SRentry {
  /// Timestamp of the message
  int theTimestamp;  
  /// The number of messages sent with theTimestamp
  int sendCount;
  /// The number of messages sent with theTimestamp
  int recvCount;
  /// Next SRentry in list
  SRentry *nextPtr;
 public:
  /// Basic constructor
  /** Initializes all data members */
  SRentry() { 
    theTimestamp = -1; sendCount = recvCount = 0; nextPtr = NULL; 
  }
  /// Initializing constructor 1
  /** Initializes theTimestamp & nextPtr w/parameters, 
      sendCount & recvCount to 0 */
  SRentry(int ts, SRentry *p) { 
    theTimestamp = ts; sendCount = recvCount = 0; nextPtr = p; 
  }
  /// Initializing constructor 2
  /** Initializes theTimestamp, send/recv count and nextPtr
      w/parameters */
  SRentry(int ts, int sr, SRentry *p) {
    theTimestamp = ts;  nextPtr = p; 
    if (sr == SEND) { sendCount = 1; recvCount = 0; }
    else { sendCount = 0; recvCount = 1; }
  }
  /// Set timestamp
  void setTimestamp(int ts) { theTimestamp = ts; }
  /// Set next pointer
  void setNext(SRentry *p) { nextPtr = p; }
  /// Set sendCount
  void setSends(int n) { sendCount = n; }
  /// Set recvCount
  void setRecvs(int n) { recvCount = n; }
  /// Increment sendCount
  void incSends() { sendCount++; } 
  /// Increment recvCount
  void incRecvs() { recvCount++; }
  /// Get timestamp
  int timestamp() { return theTimestamp; }
  /// Get next pointer
  SRentry *next() { return nextPtr; }
  /// Get sendCount
  int sends() { return sendCount; }
  /// Get recvCount
  int recvs() { return recvCount; }
  /// Dump data fields
  void dump() {
    if (nextPtr)
      CkPrintf("TS:%d #s:%d #r:%d n:!NULL ", theTimestamp, sendCount, 
	       recvCount); 
    else CkPrintf("TS:%d #s:%d #r:%d n:NULL ",theTimestamp, sendCount, 
		  recvCount);
  }
  /// Check validity of data fields
  void sanitize() {
    CmiAssert(theTimestamp >= -1); // should be -1 or > if initialized
    CmiAssert(sendCount >= 0);  // cannot be less than zero
    CmiAssert(recvCount >= 0);  // cannot be less than zero
    if (nextPtr == NULL) return;   // nextPtr can be NULL
    // if nextPtr != NULL, check if pointer looks valid
    int test_ts = nextPtr->timestamp();
    int test_sendCount = nextPtr->sends();
    int test_recvCount = nextPtr->recvs();
    SRentry *test_next = nextPtr->next();
    nextPtr->setTimestamp(test_ts);
    nextPtr->setSends(test_sendCount);
    nextPtr->setRecvs(test_recvCount);
    nextPtr->setNext(test_next);
  }
};

/// An table for storing the number of sends and recvs at a timestamp
/** This class is used in GVT to keep track of messages sent/received */
class SRtable {
 private:
  /// Stores send/recv records ith timestamp >= offset+GVT_WINDOW
  /** One entry per timestamp, all sends/recvs stored in same entry */
  SRentry *residuals;
  /// Helper function to Insert
  /** Stores new send/recv record in residuals */
  void listInsert(int timestamp, int srSt);
 public:
  /// sends[i] is number of sends at timestamp offset+i
  /** Size of table GVT_WINDOW specified in pose.h */
  int sends[GVT_WINDOW];
  /// recvs[i] is number of receives at timestamp offset+i
  /** Size of table GVT_WINDOW specified in pose.h */
  int recvs[GVT_WINDOW];
  /// Base timestamp to index tables
  /** offset is the current GVT */
  int offset;
  /// Basic constructor
  /** Initializes all data fields, including entire sends and recvs arrays */
  SRtable();
  /// Destructor
  ~SRtable() { FreeTable(); }
  /// Insert send/recv record at timestamp
  void Insert(int timestamp, int srSt); 
  /// Purge entries from table with timestamp below ts
  void PurgeBelow(int ts);      
  /// Move entries to table from residuals if timestamp < offset+GVT_WINDOW
  void FileResiduals();         
  /// Find earliest timestamp in table and associated send/recv counts
  /** Returns results by reference */
  void FindEarliest(int *eTS, int *eS, int *eR);
  /// Find earliest timestamp difference between this and a previous table
  /** Returns results by reference */
  void FindEarliestDiff(SRtable *cp, int *eTS);
  /// Copy table to cp
  void CopyTable(SRtable *cp);
  /// Free residual entries, reset counters and pointers
  void FreeTable();
  /// Dump data fields
  void dump();    
  /// Check validity of data fields
  void sanitize();
  /// Test this class
  void self_test();
};

#endif
