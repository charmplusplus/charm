// File: eventID.C
// Defines: eventID class and methods;
//          GetEventID function generates unique event IDs
// Last Modified: 6.14.01 by Terry L. Wilmarth

#include "eventID.h"
/* replaced with Cpv code in pose.C
/// Generates and returns unique event IDs
const eventID& GetEventID() {
  static eventID theEventID;  // initializes to [0.pe] for each pe called on
  theEventID.incEventID();    
  return theEventID;
}
*/
