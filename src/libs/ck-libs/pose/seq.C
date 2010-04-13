// File: seq.C
// Module for sequential simulation strategy class
#include "pose.h"

void seq::Step() {

  // execute event if checkpointing is not in progress or if the
  // timestamp is before the GVT at which a checkpoint is occurring
  if ((eq->currentPtr->timestamp < seqLastCheckpointGVT) || (!seqCheckpointInProgress)) {

    Event *ev;
    // Prepare to execute an event
    ev = eq->currentPtr;
    currentEvent = ev;
    if (ev->timestamp < POSE_GlobalTS)
      CkPrintf("WARNING: SEQUENTIAL POSE BUG! Event timestamp %d is less than a previous one! This is due to stupid Charm++ Scheduler implementation and needs to be fixed.\n", ev->timestamp);
    POSE_GlobalTS = ev->timestamp;
    parent->ResolveFn(ev->fnIdx, ev->msg);  // execute it
    if (userObj->OVT() > POSE_GlobalClock)
      POSE_GlobalClock = userObj->OVT();
    ev->done = 1;
    eq->ShiftEvent();                       // move on to next event
    eq->CommitDoneEvents(parent);

    // checkpoint if appropriate
    if ((userObj->myHandle == 0) && (seqCheckpointInProgress == 0) && (POSE_GlobalClock > 0) && 
	(((pose_config.checkpoint_gvt_interval > 0) && (POSE_GlobalClock >= (seqLastCheckpointGVT + pose_config.checkpoint_gvt_interval))) || 
	 ((pose_config.checkpoint_time_interval > 0) && 
	  ((CmiWallTimer() + seqStartTime) >= (seqLastCheckpointTime + (double)pose_config.checkpoint_time_interval))))) {
      // start quiescence detection on the sim chare
      seqCheckpointInProgress = 1;
      seqLastCheckpointGVT = POSE_GlobalClock;
      seqLastCheckpointTime = CmiWallTimer() + seqStartTime;
    }

  } else {

    // if in the process of checkpointing, store info so Step() can be
    // called for this event on restart/resume
    Skipped_Event se;
    se.simIndex = userObj->myHandle;
    se.timestamp = eq->tsOfLastInserted;

    // store skipped event
    POSE_Skipped_Events.enq(se);

  }

}
