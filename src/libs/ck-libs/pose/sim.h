/// Sim is the base class for all poser entities
/** This is the wrapper class that encapsulates synchronization strategy,
    representation object and event queue, and manages the orchestration
    of these components. 
    This file also defines the three basic POSE messages: 
    eventMsg, from which all user messages inherit; 
    cancelMsg, for cancelling events; 
    and prioMsg, a null message.
    All three have a priority field which is set to the timestamp; thus all 
    messages with earlier timestamp have higher priority. */
#ifndef SIM_H
#define SIM_H
#include "sim.decl.h"
#include <stdarg.h>

void POSE_prepExit(void *param, void *msg); 
extern CProxy_sim POSE_Objects; 
extern CProxy_sim POSE_Objects_RO; 
extern CkChareID POSE_Coordinator_ID; 
extern POSE_Config pose_config;
class sim; // needed for eventMsg definition below

/// All user event messages inherit from this
/** Adds timestamp and event ID to message, plus other info useful for the
    underlying simulation layer.  Prioritized by default, and given a priority
    based on the timestamp. Events which take no parameters must
    still pass an eventMsg. */
class eventMsg : public CMessage_eventMsg {
public:
  /// The event's timestamp
  POSE_TimeType timestamp;  
  /// The event's globally unique ID
  eventID evID;   
  /// The message size, used for message recycling (currently not used)
  size_t msgSize;    
  /// Pointer to a poser wrapper; used to send the pointer to rep object
  sim *parent;    
  /// Pointer to synchronization strategy; used when creating rep object
  strat *str;
  /// Relative start time: for computing degree of parallelization
  double rst;
  /// Basic Constructor
  eventMsg() { rst = 0.0; parent = NULL; str = NULL; evID.init();}
  /// Destructor
  virtual ~eventMsg() { }
  inline void sanitize() {
    CkAssert(timestamp > -1);
    CkAssert(evID.getPE() > -1);
    CkAssert(evID.getPE() < CkNumPes());
    CkAssert(parent == NULL);
    CkAssert(str == NULL);
    CkAssert(msgSize >= 0);
  }
  /// Timestamps this message and generates a unique event ID
  /** Timestamps this message and generates a unique event ID for the event
      to be invoked on the receiving side.  Sets the priority of this
      message to timestamp - POSE_TimeMax. */
  inline void Timestamp(POSE_TimeType t) { 
    timestamp = t;  
    if (evID.getPE() == -1)
      evID = GetEventID();  
    setPriority(t-POSE_TimeMax); 
    rst = 0.0;
    parent = NULL; str = NULL;
  }
  inline void SetSequenceNumber(int ctrl) { 
    evID = GetEventID();  
    evID.setControl(ctrl);  
  }
  /// Assignment operator: copies priority too
  inline eventMsg& operator=(const eventMsg& obj) {
    timestamp = obj.timestamp;
    evID = obj.evID;
    parent = obj.parent;
    str = obj.str;
    //msgSize = obj.msgSize;
    rst = obj.rst;
    setPriority(timestamp-POSE_TimeMax); 
    return *this;
  }
  /// Allocates event message with space for priority
  /** This can also handle event message recycling (currently off) */
  void *operator new (size_t size) {  
#ifndef SEQUENTIAL_POSE
#ifdef MSG_RECYCLING
    MemoryPool *localPool = (MemoryPool *)CkLocalBranch(MemPoolID);
    if (localPool->CheckPool(size) > 0)
      return localPool->GetBlock(size);
    else {
#endif
#endif
      void *msg = CkAllocMsg(CMessage_eventMsg::__idx, size, 8*sizeof(POSE_TimeType));
      ((eventMsg *)msg)->msgSize = size;
      return msg;
#ifndef SEQUENTIAL_POSE
#ifdef MSG_RECYCLING
    }
#endif
#endif
  }
  inline void operator delete(void *p) { 
#ifndef SEQUENTIAL_POSE
#ifdef MSG_RECYCLING
    MemoryPool *localPool = (MemoryPool *)CkLocalBranch(MemPoolID);
    int ps = localPool->CheckPool(((eventMsg *)p)->msgSize);
    if ((ps < MAX_POOL_SIZE) && (ps > -1)) {
      size_t msgSize = ((eventMsg *)p)->msgSize;
      memset(p, 0, msgSize);
      ((eventMsg *)p)->msgSize = msgSize;
      localPool->PutBlock(msgSize, p);
    }
    else
#endif
#endif
      CkFreeMsg(p);
  }
  /// Set priority field and queuing strategy
  inline void setPriority(POSE_TimeType prio) {
#if USE_LONG_TIMESTAMPS
    memcpy(((POSE_TimeType *)CkPriorityPtr(this)),&prio,sizeof(POSE_TimeType));
    CkSetQueueing(this, CK_QUEUEING_LFIFO);
#else
    *((int*)CkPriorityPtr(this)) = prio;
    CkSetQueueing(this, CK_QUEUEING_IFIFO);
#endif
  }
};

/// Cancellation message
class cancelMsg : public CMessage_cancelMsg {
public:
  /// Event to cancel
  /** Only this is needed to find the event to cancel */
  eventID evID;
  /// Timestamp of event to be cancelled
  /** Providing this makes finding the event faster */
  POSE_TimeType timestamp;          
  /// Allocate cancellation message with priority field
  void *operator new (size_t size) {  
    return CkAllocMsg(CMessage_cancelMsg::__idx, size, 8*sizeof(POSE_TimeType));
  } 
  /// Delete cancellation message
  inline void operator delete(void *p) {  CkFreeMsg(p);  }
  /// Set priority field and queuing strategy
  inline void setPriority(POSE_TimeType prio) {
#if USE_LONG_TIMESTAMPS
    memcpy(((POSE_TimeType *)CkPriorityPtr(this)),&prio,sizeof(POSE_TimeType));
    CkSetQueueing(this, CK_QUEUEING_LFIFO);
#else
    *((int*)CkPriorityPtr(this)) = prio;
    CkSetQueueing(this, CK_QUEUEING_IFIFO);
#endif
  }

};

/// Prioritized null msg; used to sort Step calls
class prioMsg : public CMessage_prioMsg {
public:
  /// Allocate prioritized message with priority field
  void *operator new (size_t size) {
    return CkAllocMsg(CMessage_eventMsg::__idx, size, 8*sizeof(POSE_TimeType));
  }
  /// Delete prioritized message
  inline void operator delete(void *p) {  CkFreeMsg(p);  }
  /// Set priority field and queuing strategy
  inline void setPriority(POSE_TimeType prio) {
#if USE_LONG_TIMESTAMPS
    memcpy(((POSE_TimeType *)CkPriorityPtr(this)),&prio,sizeof(POSE_TimeType));
    CkSetQueueing(this, CK_QUEUEING_LFIFO);
#else
    *((int*)CkPriorityPtr(this)) = prio;
    CkSetQueueing(this, CK_QUEUEING_IFIFO);

#endif
  }
};

/// Used to specify a destination processor to migrate to during load balancing
class destMsg : public CMessage_destMsg {
public:
  int destPE;
};


/// Poser wrapper base class
/** The poser base class: all user posers are translated to classes that
    inherit from this class, and act as wrappers around the actual user 
    object's representation to control the simulation behavior.  These 
    objects are plugged into the POSE_objects array which is of this type. */
class sim : public CBase_sim {
 protected:
  /// Flag to indicate that a Step message is scheduled
  /** Need to re-evaluate the need/function of this... also, how is it used
      during load balancing... */
  int active; 
 public:
  /// This poser's event queue
  eventQueue *eq;
  /// This poser's synchronization strategy   
  strat *myStrat;
  /// This poser's user representation
  rep *objID;       
  /// List of incoming cancellations for this poser
  CancelList cancels;
  /// The local PVT to report to
  PVT *localPVT;    
  /// Unique global ID for this object on PVT branch
  int myPVTidx;
  /// Unique global ID for this object in load balancing data structures
  int myLBidx;
  /// Number of forward execution steps
  /** Is this needed/used? This is load balancing data... */
  int DOs;
  /// Number of undone events
  /** Is this needed/used? This is load balancing data... */
  int UNDOs;
  /// Synchronization strategy type (optimistic or conservative)
  int sync;
  /// Number of sends/recvs per PE
  int *srVector;    
  /// Most recent GVT estimate
  POSE_TimeType lastGVT;
  /// Relative start time, end time, and current time
  /** Used to calculate degree of parallelism */
  double st, et, ct;
#if !CMK_TRACE_DISABLED
  /// The local statistics collector
  localStat *localStats;
  /// Used to manually override the value of evt for DOP calculations
  /* To override the ending virtual time of an entry method when doing
     DOP analysis, add this code to the entry method:
  #if !CMK_TRACE_DISABLED
    if ((pose_config.stats) && (pose_config.dop)) {
      parent->dop_override_evt = ovt + (POSE_TimeType)time_that_would_have_been_elapsed;
    }
  #endif
  */
  POSE_TimeType dop_override_evt;
#endif
  /// Used to count the number of commits (in [0]) and rollbacks (in [1])
  /* These statistics are collected by the poser termination
     reduction at the end of the simulation and printed in one of the
     POSE exit functions */
  long long basicStats[2];
  /// The local load balancer
  LBgroup *localLBG;
  /// Basic Constructor
  sim(void);
  sim(CkMigrateMessage *) {};
  /// Destructor
  virtual ~sim();
  /// Pack/unpack/sizing operator
  void pup(PUP::er &p);
  /// Start a forward execution step on myStrat
  void Step();                 
  /// Start a prioritized forward execution step on myStrat
  void Step(prioMsg *m);    
  /// Start a forward execution step on myStrat after a checkpoint (sequential mode only)
  void CheckpointStep(eventMsg *m);
  /// Report safe time to PVT branch
  inline void Status() { 
    localPVT->objUpdateOVT(myPVTidx, myStrat->SafeTime(), objID->OVT()); 
  }
  /// Commit events based on new GVT estimate
  void Commit();
  /// Commit all possible events before a checkpoint to disk
  void CheckpointCommit();
  /// Add m to cancellation list
  void Cancel(cancelMsg *m);
  /// Report load information to local load balancer
  void ReportLBdata();
  /// Migrate this poser to processor indicated in m
  inline void Migrate(destMsg *m) { migrateMe(m->destPE); }
  /// Terminate this poser, when everyone is terminated we exit 
  inline void Terminate() {
    //#ifndef SEQUENTIAL_POSE
    //    CkPrintf("Step[%d] total=%lld stepCalls=%d avg=%d RBs=%lld\n", 
    //	     objID->myHandle, myStrat->timeLeashTotal, myStrat->stepCalls, 
    //	     myStrat->timeLeashTotal / ((myStrat->stepCalls == 0) ? -1 : myStrat->stepCalls), basicStats[1]);
    //#endif
    objID->terminus();
    contribute(2 * sizeof(long long), basicStats, CkReduction::sum_int, CkCallback(POSE_prepExit, NULL));
  }
  /// In sequential mode, begin checkpoint after reaching quiescence
  void SeqBeginCheckpoint();
  /// In sequential mode, resume after checkpointing or restarting
  void SeqResumeAfterCheckpoint();
  /// Implement this for posers that will need to execute events when POSE reaches quiescence
  void invokeStopEvent() {}
  /// Set simulationStartGVT in the rep
  void setSimulationStartGVT(POSE_TimeType startGVT) {
    objID->setSimulationStartGVT(startGVT);
  }
  /// Return this poser's unique index on PVT branch
  inline int PVTindex() { return myPVTidx; }
  /// Test active flag
  inline int IsActive() { return active; }
  /// Set active flag
  inline void Activate() { active = 1; }
  /// Unset active flag
  inline void Deactivate() { active = 0; }
  /// Invoke an event on this poser according to fnIdx and pass it msg
  /** ResolveFn is generated along with the rest of the wrapper object and
      should handle all possible events on a poser. */
  virtual void ResolveFn(int fnIdx, void *msg) { }
  /// Invoke the commit version of an event to handle special behaviors
  /** This invokes the <fn>_commit method that user provides.  It can be
      used to perform special activities, statistics gathering, output, or
      whatever the user wishes. */
  virtual void ResolveCommitFn(int fnIdx, void *msg) { }
  /// Notify the PVT of a message send
  inline void registerSent(POSE_TimeType timestamp) {
    localPVT->objUpdate(timestamp, SEND);
  }
  /// Used for buffered output
  /** Output is only printed when the event is committed */
  void CommitPrintf(const char *Fmt, ...) {
    va_list ap;
    va_start(ap,Fmt);
    InternalCommitPrintf(Fmt, ap);
    va_end(ap);
  }
  /// Used for buffered output of error messages
  /** Output is only printed when the event is committed */
  void CommitError(const char *Fmt, ...) {
    va_list ap;
    va_start(ap,Fmt);
    InternalCommitPrintf(Fmt, ap);
    va_end(ap);
    myStrat->currentEvent->commitErr = 1;
  }
  void ResumeFromSync(void);
  /// Dump all data fields
  void dump();
 private:
  /// Used by buffered print functions
  void InternalCommitPrintf (const char *Fmt, va_list ap) {
    char *tmp;
    size_t tmplen=myStrat->currentEvent->commitBfrLen + strlen(Fmt) + 1 +512;
    if (!(tmp = (char *)malloc(tmplen * sizeof(char)))) {
      CkPrintf("ERROR: sim::CommitPrintf: OUT OF MEMORY!\n");
      CkExit();
    }
    if (myStrat->currentEvent->commitBfr && myStrat->currentEvent->commitBfrLen) {
      strcpy(tmp, myStrat->currentEvent->commitBfr);
      free(myStrat->currentEvent->commitBfr);
      vsnprintf(tmp+strlen(tmp), tmplen, Fmt, ap); 
    }
    else vsnprintf(tmp, tmplen, Fmt, ap); 
    myStrat->currentEvent->commitBfrLen = strlen(tmp) + 1;  
    myStrat->currentEvent->commitBfr = tmp;
  }
};

#endif









