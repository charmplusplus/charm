// File: rep.h
// Module for basic representation class: what the user class becomes. It adds
// minimal  functionality to the user's code, mostly OVT, and links back to 
// the parent sim object.
// Last Modified: 06.05.01 by Terry L. Wilmarth

#ifndef REP_H
#define REP_H

class rep 
{
 protected:
  sim *parent;             // pointer to wrapper object
  strat *myStrat;          // pointer to strategy
 public:
  int ovt;                 // the object's virtual time
  int myHandle;            // the objects unique handle
  rep() { ovt = 0; }
  rep(int init_ovt) { ovt = init_ovt; }
  virtual ~rep() { }
  void init(eventMsg *m);  // call at start of constructor
  int OVT() { return ovt; }
  void SetOVT(int t) { ovt = t; }
  void elapse(int dt) { ovt += dt; }  // user calls to elapse time
  void update(int t) { ovt = (ovt < t) ? t : ovt; }  // call at start of event
  virtual void terminus() { 
    CkPrintf("Object %d terminus at time %d\n", myHandle, ovt);
  }
  virtual Event *getCommitEvent(Event *e);  // get event to rollback to

  // timestamps event message, sets priority, and makes a record of the send
  virtual void registerTimestamp(int idx, eventMsg *m, int offset);
  virtual void CheckpointAll();        // set checkpoint rate to 1/1
  virtual void ResetCheckpointRate();  // reset checkpoint rate to default

  // required for checkpointing: must provide assignment in all derived classes
  virtual rep& operator=(const rep& obj) { 
    ovt = obj.ovt; 
    return *this;
  }
  virtual void dump(int pdb_level);        // dump the entire rep object
  virtual void pup(PUP::er &p) { p(ovt); } // pup the entire rep object
};

#endif
