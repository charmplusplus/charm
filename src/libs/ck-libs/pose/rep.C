/// Base class to represent user object
#include "pose.h"

/// Initializer called from poser wrapper constructor
void rep::init(eventMsg *m)
{ 
  ovt = m->timestamp; 
  ort = 0.0;
  parent = m->parent; 
  myStrat = m->str;
  myHandle = parent->thisIndex;
  anti_methods = 0;
}

void rep::update(POSE_TimeType t, double rt) 
{ 
  ovt = (ovt < t) ? t : ovt;
  parent->eq->currentPtr->svt = ovt;
#ifdef POSE_DOP_ON  
  ort = (ort < rt) ? rt : ort;
  parent->eq->currentPtr->srt = ort;
#endif
}

/// Timestamps event message, sets priority, and records in spawned list
void rep::registerTimestamp(int idx, eventMsg *m, POSE_TimeType offset)
{
  PVT *localPVT = (PVT *)CkLocalBranch(ThePVT);
  CmiAssert(ovt+offset >= localPVT->getGVT());
  m->Timestamp(ovt+offset);
  m->setPriority(ovt+offset-INT_MAX);
  m->evID.setObj(myHandle);
  parent->registerSent(ovt+offset);
}
