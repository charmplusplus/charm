// File: eventID.C
// Defines: eventID class and methods;
//          GetEventID function generates unique event IDs
// Last Modified: 6.14.01 by Terry L. Wilmarth

#include "eventID.h"

/// Get next value for eventID
void eventID::incEventID() 
{
  id++;
  if (id == 0) CkPrintf("WARNING: event ID rollover occurred!\n");
}

/// Assignment operator
eventID& eventID::operator=(const eventID& e) 
{ 
  CmiAssert((e.pe >= 0) || (e.pe < CkNumPes()));
  id = e.id;  pe = e.pe;  obj = e.obj; return *this;
}

/// Equality comparison operator
int eventID::operator==(const eventID& o) 
{
  return ((id == o.id) && (pe == o.pe) && (obj == o.obj));
}

/// Less than/equality comparison operator
int eventID::operator<=(const eventID& o) 
{
  return ((obj < o.obj) || ((obj == o.obj) && (id <= o.id)));
}

/// Less than comparison operator
/** Provides a way to sort events by event ID */
int eventID::operator<(const eventID& o)
{
  return ((obj < o.obj) || ((obj == o.obj) && (id < o.id)));
}

/// Generates and returns unique event IDs
const eventID& GetEventID() {
  static eventID theEventID;  // initializes to [0.pe] for each pe called on
  theEventID.incEventID();    
  return theEventID;
}
