// File: srtable.h
// SendRecvTable is a table that stores timestamps of sends and recvs of all
// events and cancellations.
// Used in gvt.*.  

#ifndef SRTABLE_H
#define SRTABLE_H
#include "pose.h"

class UpdateMsg;

class SRentry { // A record for a set of send/recv w/same timestamp
 public:
  int timestamp;  // timestamp of the message
  int sendCount, recvCount;  // number of messages with this timestamp
  SRentry *next;  // this type is used almost always in linked lists
  SRentry() { timestamp = -1; sendCount = recvCount = 0; next = NULL; }
  ~SRentry() { }
  SRentry(int ts, SRentry *p) { 
    timestamp = ts; sendCount = recvCount = 0; next = p; 
  }
  void incSends() { sendCount++; }
  void incRecvs() { recvCount++; }
  void dump() { 
    CkPrintf("TS:%d #s:%d #r:%d", timestamp, sendCount, recvCount); 
  }
};

class SRtable {
 private:
  SRentry *residuals;  // all other send/recv events
  void listInsert(int timestamp, int srSt);
 public:
  int sends[GVT_WINDOW], recvs[GVT_WINDOW]; // send/recv events occurring 
  int offset;                   // gvt offset
  SRtable();                    // basic constructor
  ~SRtable();                   // needed to free up the linked lists
  void Insert(int timestamp, int srSt); // insert s/r at timestamp
  void FindEarliest(int *eTS, int *eS, int *eR, int *nTS, int *nS, int *nR);
  void PurgeBelow(int ts);      // purge table below timestamp ts
  void FileResiduals();         // try to file each residual event in table
  void ClearTable() {           // reset counters & ptrs; free all
    //sanitize();
    SRentry *tmp;
    offset = 0;
    for (int i=0; i<GVT_WINDOW; i++)
      sends[i] = recvs[i] = 0;
    tmp = residuals;
    while (tmp) { 
      residuals = tmp->next;
      delete(tmp);
      tmp = residuals;
    }
  }
  UpdateMsg *packTable();
  void dump();                  // dump the table
  void sanitize();
};

#endif
