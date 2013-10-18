/// Base class to represent user object
#include "pose.h"

/// Update the OVT and ORT at event start to auto-elapse to event timestamp
void rep::update(POSE_TimeType t, double rt) { 
  ovt = (ovt < t) ? t : ovt;
  parent->eq->currentPtr->svt = ovt;
#if !CMK_TRACE_DISABLED
  if(pose_config.dop){
    ort = (ort < rt) ? rt : ort;
    parent->eq->currentPtr->srt = ort;
  }
#endif
}

/// Initializer called from poser wrapper constructor
void rep::init(eventMsg *m) { 
  ovt = m->timestamp; 
  ort = 0.0;
  parent = m->parent; 
  myStrat = m->str;
  myHandle = m->parent->thisIndex;
  anti_methods = 0;
  prand_seed = myHandle;
  prand48_seed[0] = (unsigned short int)myHandle;
  prand48_seed[1] = (unsigned short int)myHandle;
  prand48_seed[2] = (unsigned short int)myHandle;
}

/// Timestamps event message, sets priority, and records in spawned list
void rep::registerTimestamp(int idx, eventMsg *m, POSE_TimeType offset)
{
#ifndef SEQUENTIAL_POSE
  PVT *localPVT = (PVT *)CkLocalBranch(ThePVT);
  CmiAssert(ovt+offset >= localPVT->getGVT());
#endif
  m->Timestamp(ovt+offset);
  m->setPriority(ovt+offset-POSE_TimeMax);
  //m->evID.setObj(myHandle);
#ifndef SEQUENTIAL_POSE
  parent->registerSent(ovt+offset);
#endif
}
