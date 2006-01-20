/// Adaptive Synchronization Strategy No. 2
/** This is the experimental strategy.  It may or may not differ from 
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
#ifndef ADAPT2_H
#define ADAPT2_H

class adapt2 : public opt3 {
 public:
  adapt2() { timeLeash = pose_config.spec_window; STRAT_T = ADAPT2_T; }
  virtual void Step();
};

#endif
