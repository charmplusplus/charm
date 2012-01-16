/// Unique event IDs for POSE events
/** Provides the event ID data structure and function GetEventID to generate
    the unique event IDs */
#ifndef EVENTID_H
#define EVENTID_H
#include "charm++.h"
#include "limits.h"

#ifdef WIN32
#define snprintf _snprintf
#endif

/// Unique identifier for a POSE event
class eventID 
{
  /// PE identifier field ensures uniqueness across PEs
  int pe;
  /// Control field for ordering events with same timestamp
  int control;
 public:
  /// Large number for identifier unique on creation PE
  unsigned int id; 
  /// Basic Constructor
  eventID() : pe(CkMyPe()), control (0), id(0) { }          

  /// Init to default unused values
  inline void init() { id = 0; pe = -1; control = INT_MIN; }
  /// Get next value for eventID
  /** increments id field for this eventID */
  inline void incEventID() {
    id++;
    if (id == 0) CkPrintf("WARNING: event ID rollover occurred!\n");
  }
  /// Assignment operator
  inline eventID& operator=(const eventID& e) { 
    CmiAssert((e.pe >= 0) || (e.pe < CkNumPes()));
    id = e.id;  pe = e.pe;  control = e.control; return *this;
  }
  /// get source PE
  inline int getPE() { return pe; }
  /// get source control
  inline int getControl() { return control; }
  /// set source control
  inline void setControl(int ctrl) { control = ctrl; }
  /// Equality comparison operator
  inline int operator==(const eventID& o) {
    return ((id == o.id) && (pe == o.pe) && (control == o.control));
  }
  /// Less than/equality comparison operator
  /** Provides a way to sort events by event ID */
  inline int operator<=(const eventID& o) {
    return (control <= o.control);
  }
  /// Less than comparison operator
  /** Provides a way to sort events by event ID */
  inline int operator<(const eventID& o) {
    return (control < o.control);
  }
  /// Greater than/equality comparison operator
  /** Provides a way to sort events by event ID */
  inline int operator>=(const eventID& o) {
    return (control >= o.control);
  }
  /// Greater than comparison operator
  /** Provides a way to sort events by event ID */
  inline int operator> (const eventID& o) {
    return (control >  o.control);
  }

  /// Dump all data fields
  inline void dump() { 
    CmiAssert((pe >= 0) && (pe < CkNumPes())); 
    CkPrintf("%d.%d", id, pe); 
  }    
  inline char *sdump(char *s) { sprintf(s, "%d.%d", id, pe); return s;}    
  inline char *sndump(char *s,size_t n) { snprintf(s,n,"%d.%d", id, pe); return s;}
  /// Pack/unpack/sizing operator
  inline void pup(class PUP::er &p) { p(id); p(pe); }  
  inline void sanitize() { CkAssert((pe > -1) && (pe < CkNumPes())); }
};

/// Generates and returns unique event IDs
const eventID& GetEventID();                    

#endif
