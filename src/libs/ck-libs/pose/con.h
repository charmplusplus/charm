// File: con.h
// Module for conservative simulation strategy class
// Last Modified: 06.05.01 by Terry L. Wilmarth

#ifndef CON_H
#define CON_H

class con : public strat {
 protected:
  virtual void CancelEvents();
 public:
  void initSync() { parent->sync = CONSERVATIVE; }
  virtual void Step();
  int SafeTime() {  // computes the safe time for this object: the object in
    // its current state cannot generate an event or cancellation earlier than
    // its safe time
    int hightime = userObj->OVT(), lowtime;
    
    if ((parent->cancels.earliest < 0) 
	&& (eq->currentPtr->timestamp < 0)
	&& (userObj->OVT() <= localPVT->getGVT()))
      return -1;  // this corresponds to an idle object; ignore its safe time
    
    if (eq->currentPtr->timestamp > hightime)  // check next queued event
      hightime = eq->currentPtr->timestamp;
    
    lowtime = hightime;
    if (parent->cancels.earliest >= 0)  // check cancellations
      lowtime = parent->cancels.earliest;
    
    if (lowtime < hightime)
      return lowtime;
    else 
      return hightime;
  }
};

#endif
