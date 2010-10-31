// File: seq.h
// Module for sequential simulation strategy class
#ifndef SEQ_H
#define SEQ_H

class seq : public strat {
 public:
  seq() { 
    STRAT_T = SEQ_T; 
#ifndef SEQUENTIAL_POSE
    CkAbort("ERROR: can't have sequential posers in parallel simulation!\n");
#endif    
  }
  inline void initSync() { parent->sync = CONSERVATIVE; }
  virtual void Step();
};

#endif
