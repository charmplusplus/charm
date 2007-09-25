/// Adaptive Synchronization Strategy No. 4
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
#ifndef ADAPT4_H
#define ADAPT4_H

class adapt4 : public opt3 {
  double specTol;
 public:
  int itersAllowed, iter, objUsage;
  adapt4() { 
    itersAllowed=-1;
    iter=0;
    objUsage = pose_config.max_usage * pose_config.store_rate;
    STRAT_T = ADAPT4_T; 
    //timeLeash = POSE_TimeMax/2;
    timeLeash = 1;
    specTol = 0.01;
  }
  virtual void Step();
};

#endif
