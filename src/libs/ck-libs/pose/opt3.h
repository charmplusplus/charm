/// Optimistic Synchronization Strategy No. 3: Time window
/** Performs locally available events in strict timestamp order, as
    long as they are within a time window. If an object has multiple
    events available at the earliest timestamp, they are all executed
    before the object gives up control. */
#ifndef OPT3_H
#define OPT3_H

class opt3 : public opt {
 protected:
  /// Time window size; fixed to SPEC_WINDOW in pose.h
  POSE_TimeType timeLeash;
public:
  opt3() : timeLeash (pose_config.spec_window){ STRAT_T = OPT3_T; }
  virtual void Step();
};

#endif
