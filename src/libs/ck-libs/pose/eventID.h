/// Unique event IDs for POSE events
/** Provides the event ID data structure and function GetEventID to generate
    the unique event IDs */
#ifndef EVENTID_H
#define EVENTID_H
#include "charm++.h"

/// Unique identifier for a POSE event
class eventID 
{
  /// PE identifier field ensures uniqueness across PEs
  int pe;
  /// Object identifier field for ordering
  int obj;
 public:
  /// Large number for identifier unique on creation PE
  unsigned int id; 
  /// Basic Constructor
  eventID() { id = 0; pe = CkMyPe(); obj = 0; }          
  /// Get next value for eventID
  /** increments id field for this eventID */
  void incEventID();                            
  /// Assignment operator
  eventID& operator=(const eventID& e);         
  /// get source PE
  int getPE() { return pe; }
  /// get source obj
  int getObj() { return obj; }
  /// set source obj
  void setObj(int objIdx) { obj = objIdx; }
  /// Equality comparison operator
  int operator==(const eventID& obj);           
  /// Less than/equality comparison operator
  /** Provides a way to sort events by event ID */
  int operator<=(const eventID& obj);           
  /// Less than comparison operator
  /** Provides a way to sort events by event ID */
  int operator<(const eventID& obj);           
  /// Dump all data fields
  void dump() { 
    CmiAssert((pe >= 0) && (pe < CkNumPes())); 
    CkPrintf("%d.%d", id, pe); 
  }    
  char *sdump(char *s) { sprintf(s, "%d.%d", id, pe); return s;}    
  char *sndump(char *s,size_t n) { snprintf(s,n,"%d.%d", id, pe); return s;}
  /// Pack/unpack/sizing operator
  void pup(class PUP::er &p) { p(id); p(pe); }  
  void sanitize() { CkAssert((pe > -1) && (pe < CkNumPes())); }
};

/// Generates and returns unique event IDs
const eventID& GetEventID();                    

#endif
