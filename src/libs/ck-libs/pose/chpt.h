/// Checkpointing representation to be used with optimistic strategy
#ifndef CHPT_H
#define CHPT_H
extern POSE_Config pose_config;

/// Templated checkpointing class derived from rep
/** This class makes it possible for optimistic synchronization strategies
    to automatically checkpoint objects of classes derived from this type. */
template<class StateType> class chpt : public rep {
 public:
  int sinceLast;
  /// Basic Constructor
  chpt() : sinceLast(pose_config.store_rate){ }
  /// Destructor
  virtual ~chpt() { }
  void registerTimestamp(int idx, eventMsg *m, POSE_TimeType offset);
  /// Checkpoint the state
  void checkpoint(StateType *data);          
  /// Restore the state from a checkpoint
  /** Used during a rollback by the Undo method:  if the event being undone is
      not the final destination, we simply remove the checkpointed data; if the
      event is the final target for the rollback, then we restore the data, and
      remove the checkpointed data for it (it will be regenerated when the 
      target event gets re-executed). */
  void restore(StateType *data);
  virtual void pup(PUP::er &p) { rep::pup(p); }
  virtual void dump() { rep::dump(); }
};


/// Timestamps event message, sets priority, and records in spawned list
template<class StateType> 
inline void chpt<StateType>::registerTimestamp(int idx, eventMsg *m, POSE_TimeType offset)
{
  m->Timestamp(ovt+offset);
  m->setPriority(ovt+offset-POSE_TimeMax);
    parent->registerSent(ovt+offset);
  ((opt *)myStrat)->AddSpawnedEvent(idx, m->evID, m->timestamp);
}

/// Checkpoint the state
template<class StateType>
void chpt<StateType>::checkpoint(StateType *data)
{
#if !CMK_TRACE_DISABLED
  localStat *localStats = (localStat *)CkLocalBranch(theLocalStats);
  if(pose_config.stats)
    localStats->SwitchTimer(CP_TIMER);
#endif
  if (usesAntimethods()) {
    myStrat->currentEvent->cpData = new rep;
    *(myStrat->currentEvent->cpData) = *(rep *)data;
  }
  else {
#ifdef MEM_TEMPORAL
    CmiAssert(!(parent->myStrat->currentEvent->serialCPdata));
#else
    CmiAssert(!(parent->myStrat->currentEvent->cpData));
#endif
    //  if ((myStrat->currentEvent->timestamp > 
    //myStrat->currentEvent->prev->timestamp) || (sinceLast == STORE_RATE)) {
    if ((sinceLast == ((opt *)myStrat)->cpRate) || 
	//(CpvAccess(stateRecovery) == 1) || 
	(myStrat->currentEvent->prev == parent->eq->front())) {
#ifdef MEM_TEMPORAL
      PUP::sizer sp; 
      ((StateType *)data)->cpPup(sp);
      myStrat->currentEvent->serialCPdataSz = sp.size();
      myStrat->currentEvent->serialCPdata = localTimePool->tmp_alloc(myStrat->currentEvent->timestamp, myStrat->currentEvent->serialCPdataSz);

      PUP::toMem tp(myStrat->currentEvent->serialCPdata); 
      ((StateType *)data)->cpPup(tp); 
      if (tp.size()!=myStrat->currentEvent->serialCPdataSz)
	CmiAbort("PUP packing size mismatch!");
#else      
      myStrat->currentEvent->cpData = new StateType;
      *((StateType *)myStrat->currentEvent->cpData) = *data;
#endif
      sinceLast = 0;
#if !CMK_TRACE_DISABLED
      //localStats->Checkpoint();
      //localStats->CPbytes(sizeof(StateType));
#endif
    }
    else sinceLast++;
  }
#if !CMK_TRACE_DISABLED
  if(pose_config.stats)
    localStats->SwitchTimer(SIM_TIMER);
#endif
}

/// Restore the state from a checkpoint
template<class StateType> 
void chpt<StateType>::restore(StateType *data) 
{
  if (usesAntimethods()) { 
    *(rep *)data = *(myStrat->currentEvent->cpData);
    delete myStrat->currentEvent->cpData;
    myStrat->currentEvent->cpData = NULL;
  }
  else {
    if (myStrat->currentEvent == myStrat->targetEvent) {
#ifdef MEM_TEMPORAL
      if (myStrat->targetEvent->serialCPdata) {
	// data should be a pre-allocated PUP::able object
        PUP::fromMem fp(myStrat->targetEvent->serialCPdata); 
	((StateType *)data)->cpPup(fp); 
	if (fp.size()!=myStrat->targetEvent->serialCPdataSz) 
	  CmiAbort("PUP unpack size mismatch!");
	localTimePool->tmp_free(myStrat->targetEvent->timestamp, myStrat->targetEvent->serialCPdata);
	myStrat->targetEvent->serialCPdata = NULL;
	myStrat->targetEvent->serialCPdataSz = 0;
      }
#else
      if (myStrat->targetEvent->cpData) {
	*data = *((StateType *)myStrat->targetEvent->cpData);
	delete myStrat->targetEvent->cpData;
	myStrat->targetEvent->cpData = NULL;
      }
#endif
    }
#ifdef MEM_TEMPORAL
    if (myStrat->currentEvent->serialCPdata) {
      localTimePool->tmp_free(myStrat->currentEvent->timestamp, myStrat->currentEvent->serialCPdata);
      myStrat->currentEvent->serialCPdata = NULL;
      myStrat->currentEvent->serialCPdataSz = 0;
    }
#else
    if (myStrat->currentEvent->cpData) {
      delete myStrat->currentEvent->cpData;
      myStrat->currentEvent->cpData = NULL;
    }
#endif
  }
}

#endif
