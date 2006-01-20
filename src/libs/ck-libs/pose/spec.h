/// Speculative Synchronization Strategy
/** Performs locally available events speculatively, as long as they
    are within a speculative time window. If an object has multiple
    events available within the window, they are all executed
    before the object gives up control. When object does give up
    control, it does not schedule any future work it might have available.
    It will not be able to execute this work until the speculative window
    moves forward. This happens only when a new GVT estimate is obtained.
    When this happens all objects attempt to execute their available events */
#ifndef SPEC_H
#define SPEC_H

class spec : public opt3 {
public:
  spec() { timeLeash = pose_config.spec_window; STRAT_T = SPEC_T; }
  virtual void Step();
};

#endif
