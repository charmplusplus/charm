/// Base class to represent user object
/** This is what the user class becomes. It adds minimal functionality
    to the user's code, mostly OVT, and links back to the parent sim object. 
    It also provides the derived templated class chpt for checkpointing. */
#ifndef REP_H
#define REP_H

/// Base representation class
class rep 
{
 protected:
  /// Pointer to poser wrapper
  sim *parent;             
  /// Pointer to synchronization strategy
  strat *myStrat;          
 public:
  /// The object's virtual time (OVT)
  int ovt;
  /// The object's real time (ORT)
  double ort;
  /// the object's unique handle
  /** Initialized to index of poser wrapper in POSE_objects array */
  int myHandle;            
  /// Flag to signify if this is a checkpointed copy of the real object
  int copy;                
  /// Basic Constructor
  rep() { ovt = 0; ort = 0.0; copy = 0; parent = NULL; myStrat = NULL; }
  /// Initializing Constructor
  rep(int init_ovt) { ovt = init_ovt; ort = 0.0; copy = 0; }
  /// Destructor
  virtual ~rep() { }
  /// Initializer called from poser wrapper constructor
  void init(eventMsg *m);  
  /// Return the OVT
  int OVT() { return ovt; }
  /// Set the OVT to t
  void SetOVT(int t) { ovt = t; }
  /// Elapse time by incrementing the OVT by dt
  void elapse(int dt) { ovt += dt; }
  /// Update the OVT and ORT at event start to auto-elapse to event timestamp
  /** If event has timestamp > OVT, OVT elapses to timestamp, otherwise
      there is no change to OVT. ORT updates similarly. */
  void update(int t, double rt);
  /// Called on every object at end of simulation
  virtual void terminus() { 
    //CkPrintf("Object %d terminus at time %d\n", myHandle, ovt);
  }
  /// Timestamps event message, sets priority, and records in spawned list
  virtual void registerTimestamp(int idx, eventMsg *m, unsigned int offset);
  /// Assignment operator
  /** Derived classes must provide assignment */
  virtual rep& operator=(const rep& obj) { 
    ovt = obj.ovt; 
    ort = obj.ort; 
    return *this;
  }
  /// Dump all data fields
  virtual void dump() { CkPrintf("[REP: ovt=%d]\n", ovt); }
  /// Pack/unpack/sizing operator
  /** Derived classes must provide pup */
  virtual void pup(PUP::er &p) { p(ovt); p(myHandle); p(copy); }
};

#endif
