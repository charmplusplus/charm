// File: srtable.h
// SendRecvTable is a table that stores timestamps of sends and recvs of all
// events and cancellations.
// Used in gvt.*.  

#ifndef SRTABLE_H
#define SRTABLE_H

#include "pose.h"

class SRentry { // A record for a single send/recv event
 public:
  int timestamp, sr;  // sr denotes either SEND or RECV
  SRentry *next;
  SRentry() { timestamp = sr = -1; next = NULL; }
  SRentry(int ts, int srSt, SRentry *p) { 
    timestamp = ts; sr = srSt; next = p;
  }
  void Set(int ts, int srSt, SRentry *p) { 
    timestamp = ts; sr = srSt; next = p;
  }
  void dump() { 
    CkPrintf("TS:%d SR:%d", timestamp, sr);
  }
};

class SRtable {
 private:
  SRentry *residuals;           // all other send/recv events
 public:
  int sends[GVT_WINDOW], recvs[GVT_WINDOW]; // send & recv events occurring 
                                            // at timestamps between gvt and 
                                            // gvt+GVT_WINDOW
  int offset;                   // gvt offset
  SRtable();                    // basic constructor
  ~SRtable();                   // needed to free up the linked lists
  void SetOffset(int gvt);      // set GVT offset
  void Insert(int timestamp, int srSt); // insert s/r at timestamp
  int FindDifferenceTimestamp(SRtable *t); // return timestamp with first difference
  void PurgeBelow(int ts);      // purge table below timestamp ts
  void FileResiduals();         // try to file each residual event in table
  void FreeTable();             // reset counters & ptrs; free all
  void dump();                  // dump the table
};

#endif
