/// Adaptive Synchronization Strategy No. 3
/** This is the most experimental strategy.  It may or may not differ from 
    adapt at any given time.
    Performs locally available events speculatively, as long as they
    are within a speculative time window. The speculative time window
    shrinks to a minimum size if the object rolls back, otherwise it
    expands by some amount to a maximum if no rollbacks occur. (See
    pose.h for the #defines for these values. If an object has
    multiple events available within the window, they are all executed
    before the object gives up control. When object does give up
    control, it does not schedule any future work it might have
    available.  It will not be able to execute this work until the
    speculative window moves forward. This happens only when a new GVT
    estimate is obtained.  When this happens all objects attempt to
    execute their available events */
#ifndef ADAPT3_H
#define ADAPT3_H

class adapt3 : public opt3 {
  double specTol;
 public:
  adapt3() : specTol(0.05) { 
    STRAT_T = ADAPT3_T; 
    timeLeash = POSE_TimeMax/2;
  }
  virtual void Step();
};

#endif
