/// Optimistic Synchronization Strategy No. 2
/** Performs locally available events in strict timestamp order. If an
    object has multiple events available at the earliest timestamp, they
    are all executed before the object gives up control. */
#ifndef OPT2_H
#define OPT2_H

class opt2 : public opt {
public:
  opt2() { STRAT_T = OPT2_T; }
  virtual void Step();
};

#endif
