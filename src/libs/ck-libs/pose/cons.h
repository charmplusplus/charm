// File: con.h
// Module for conservative simulation strategy class
// Last Modified: 06.05.01 by Terry L. Wilmarth

#ifndef CONS_H
#define CONS_H

class con : public strat {
 protected:
  virtual void CancelEvents();
 public:
  void initSync() { parent->sync = CONSERVATIVE; }
  virtual void Step();
  POSE_TimeType SafeTime() {  // computes the safe time for this object: the object in
    // its current state cannot generate an event or cancellation earlier than
    // its safe time
    POSE_TimeType hightime = userObj->OVT(), lowtime;
    
    if ((parent->cancels.getEarliest() < 0) 
	&& (eq->currentPtr->timestamp < 0)
	&& (userObj->OVT() <= localPVT->getGVT()))
      return POSE_UnsetTS;  // this corresponds to an idle object; ignore its safe time
    
    if (eq->currentPtr->timestamp > hightime)  // check next queued event
      hightime = eq->currentPtr->timestamp;
    
    lowtime = hightime;
    if (parent->cancels.getEarliest() >= 0)  // check cancellations
      lowtime = parent->cancels.getEarliest();
    
    if (lowtime < hightime)
      return lowtime;
    else 
      return hightime;
  }
};

#endif
