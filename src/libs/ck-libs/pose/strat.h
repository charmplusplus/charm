/// Simulation synchronization strategy base class
/** Protocols such as optimistic and conservative inherit from this. */
#ifndef STRAT_H
#define STRAT_H

/// Strategy types
#define SEQ_T -3
#define INIT_T -2
#define CONS_T -1
#define OPT_T 0
#define OPT2_T 1
#define OPT3_T 2
#define SPEC_T 3
#define ADAPT_T 4
#define ADAPT2_T 5
#define ADAPT3_T 6
#define ADAPT4_T 7
#define ADAPT5_T 8

/// Base synchronization strategy class
class strat
{
 protected:
#if !CMK_TRACE_DISABLED
  localStat *localStats;
#endif
  /// Local PVT branch
  PVT *localPVT;
  /// Pointer to the eventQueue in the poser wrapper
  eventQueue *eq;   
  /// Pointer to the representation object in the poser wrapper
  rep *userObj;     
  /// Pointer to the poser wrapper
  sim *parent;      
 public:
  //  /// Time leash accumulator
  //  long long timeLeashTotal;
  //  /// Number of times Step() is called
  //  int stepCalls;
  /// Type of strategy; see #defines above
  int STRAT_T;      
  /// Target event pointer
  /** Used by strategy to denote point in event queue to 1) rollback to
      2) checkpoint up to etc. */
  Event *targetEvent; 
  /// Current event being executed  
  Event *currentEvent;
  /// Basic Constructor
  strat();
  /// Destructor
  virtual ~strat() { }
  /// Initializer
  void init(eventQueue *q, rep *obj, sim *p, int pIdx);
  /// Initialize synchronization strategy type (optimistic or conservative)
  virtual void initSync() { }
  /// Strategy-specific forward execution step
  /** Code here MUST be overridden, but this gives a basic idea of how
      a forward execution step should go.  Strategies must determine if it is
      safe to execute an event. */
  virtual void Step(); 
  /// Strategy-specific rollback
  virtual void Rollback() { }      
  /// Strategy-specific event cancellation
  virtual void CancelEvents() { }  
  /// Calculate safe time (earliest time at which object can generate events)
  virtual POSE_TimeType SafeTime() { return userObj->OVT(); }
};

#endif
