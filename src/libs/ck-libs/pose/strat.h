// File: strat.h
// Module for basic simulation strategy class for protocols such as 
// optimistic and conservative.
// Last Modified: 06.05.01 by Terry L. Wilmarth

#ifndef STRAT_H
#define STRAT_H

#define INIT_T -2
#define CONS_T -1
#define OPT_T 0
#define OPT2_T 1
#define OPT3_T 2
#define SPEC_T 3
#define ADAPT_T 4
#define ADAPT2_T 5

class strat
{
 protected:
  PVT *localPVT;
#ifdef POSE_STATS_ON
  localStat *localStats;
#endif
  eventQueue *eq;   // pointer to the eventQueue in parent sim obj
  rep *userObj;     // pointer to the user's representation in parent sim obj
  sim *parent;      // pointer to the parent sim object
  int parentIdx;    // array index of parent
 public:
  Event *RBevent;   // current rollback event
  int STRAT_T;      // type of strategy; see defines above
  Event *targetEvent,   // checkpoint state target event
    *currentEvent;  // current event being executed
  int voted;        // if this object's strategy has voted for a GVT run
  strat();          // basic initialization constructor
  virtual ~strat() { }
  void init(eventQueue *q, rep *obj, sim *p, int pIdx);  // init pointers
  virtual void initSync() { }
  virtual void Step(); // Strategy specific forward execution step.
  virtual void Rollback() { }      // strategy specific rollback function
  virtual void CancelEvents() { }  // strategy specific event cancellation
  virtual int SafeTime() { return userObj->OVT(); }  // strategy-specific
  void ResetRBevent(Event *e) { RBevent = e; }
  Event *getCurrentEvent() { return currentEvent; }
  Event *getTargetEvent() { return targetEvent; }
};

#endif
