// File: eventID.h
// Defines: eventID class and methods;
//          GetEventID function generates unique event IDs
// Last Modified: 6.14.01 by Terry L. Wilmarth

#ifndef EVENTID_H
#define EVENTID_H

#include "charm++.h"

// This object uniquely identifies all events
class eventID 
{
 public:
  unsigned int id; // large number for identifier
  int pe;          // pe field ensures uniqueness across PEs
  eventID() { id = 0; pe = CkMyPe(); }          // basic initialization
  void resetEventID() { id = 0; }               // sets to init state
  void incEventID();                            // get next value for eventID
  eventID& operator=(const eventID& e);         // eventID assignment
  int operator==(const eventID& obj);           // eventID equality comparison
  int operator<=(const eventID& obj);           // eventID less/eq comparison
  void dump() { CkPrintf("%d.%d", id, pe); }    // print eventID
  void pup(class PUP::er &p) { p(id); p(pe); }  // pup eventID
};

const eventID& GetEventID();                    // generates unique event IDs

#endif
