// File: eventID.C
// Defines: eventID class and methods;
//          GetEventID function generates unique event IDs
// Last Modified: 6.14.01 by Terry L. Wilmarth

#include "eventID.h"

// Change this to next possible value
void eventID::incEventID() 
{
  id++;
  if (id == 0)
    CkPrintf("NOTE: event ID rollover occurred!\n");
}

// eventID assignment
eventID& eventID::operator=(const eventID& e) 
{ 
  id = e.id; 
  pe = e.pe;
  return *this;
}

// eventID equality comparison
int eventID::operator==(const eventID& obj) 
{
  return ((id == obj.id) && (pe == obj.pe));
}

// eventID less/eq comparison
int eventID::operator<=(const eventID& obj) 
{
  return ((pe < obj.pe) || ((pe == obj.pe) && (id <= obj.id)));
}

// Generates unique event IDs
const eventID& GetEventID() {
  static eventID theEventID;  // init to [0.pe] for each pe called on
  theEventID.incEventID();    
  return theEventID;
}
