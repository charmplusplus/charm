// File: chpt.h
// Module for checkpointing representation to be used with optimistic strategy
// Last Modified: 07.31.01 by Terry L. Wilmarth

#ifndef CHPT_H
#define CHPT_H

//----------------------------------------------------------------------------
// rep class for checkpointing
template<class StateType> class chpt : public rep {
 private:
  int store;       // flag specifying store state; if 0, store; otherwise not
  int rate_store;  // store 1 state for every rate_store events
 public:
  chpt() {  store = 0;  rate_store = STORE_RATE;  }
  virtual ~chpt() { }
  // timestamps event message, sets priority, and makes a record of the send
  void registerTimestamp(int idx, eventMsg *m, int offset);
  void checkpoint(StateType *data);          // checkpoint the data
  void restore(StateType *data);             // restore checkpointed data 
  void CheckpointAll() { rate_store = 1; store = 0; }
  void ResetCheckpointRate() { rate_store = STORE_RATE; store = 0; }
  Event *getCommitEvent(Event *e);  // get event to rollback to
  virtual void pup(PUP::er &p);
  virtual void dump(int pdb_level);
};
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// chpt methods defined below
//----------------------------------------------------------------------------


// timestamps event message, sets priority, and makes a record of the send;
// additionally, notes the generated event as a spawn of the current one
template<class StateType> 
void chpt<StateType>::registerTimestamp(int idx, eventMsg *m, int offset)
{
  m->Timestamp(ovt+offset);
  m->setPriority(ovt+offset-INT_MAX);
  parent->registerSent(ovt+offset);
  ((opt *)myStrat)->AddSpawnedEvent(idx, m->evID, m->timestamp);
}

// if the chpt object is in a store state (store == 0), allocate a copy
// int the current event in event queue and copy the current state there
template<class StateType>
void chpt<StateType>::checkpoint(StateType *data)
{
#ifdef POSE_STATS_ON
  localStat *localStats = (localStat *)CkLocalBranch(theLocalStats);
  localStats->SwitchTimer(CP_TIMER);
#endif
  if (myStrat->currentEvent->cpData) {
    CkPrintf("ERROR: chpt::checkpoint: cpData exists already\n");
    CkExit();
  }
  //  if ((store == 0) || (myStrat->currentEvent->prev->fnIdx == -99)) {
    // checkpoint when store is zero or event is first in queue
  /*if (parent->recycCount > 0) {
    parent->recycCount--;
    myStrat->currentEvent->cpData = parent->recyc[parent->recycCount];
  }
  else */
  if (myStrat->currentEvent->timestamp > 
      myStrat->currentEvent->prev->timestamp) {
    myStrat->currentEvent->cpData = new StateType;
    *((StateType *)myStrat->currentEvent->cpData) = *data;
//#ifdef POSE_STATS_ON
//    localStats->Checkpoint();
//#endif
  }
  
    //  }
  store = (store + 1) % rate_store;  // advance the store flag
#ifdef POSE_STATS_ON
  localStats->SwitchTimer(DO_TIMER);
#endif
}

// used during a rollback by the Undo method:  if the event being undone is
// not the final destination, we simply remove the checkpointed data; if the
// event is the final target for the rollback, then we restore the data, and
// remove the checkpointed data for it (it will be regenerated when the 
// target event gets re-executed).
template<class StateType> 
void chpt<StateType>::restore(StateType *data) 
{
  if (myStrat->currentEvent == myStrat->targetEvent) {
    if (myStrat->targetEvent->cpData) {
      *data = *((StateType *)myStrat->targetEvent->cpData);
      delete myStrat->targetEvent->cpData;
      myStrat->targetEvent->cpData = NULL;
    }
  }
  if (myStrat->currentEvent->cpData) {
    delete myStrat->currentEvent->cpData;
    myStrat->currentEvent->cpData = NULL;
  }
  store = 0;
}

// get event to rollback to prior to or including e
template<class StateType>
Event *chpt<StateType>::getCommitEvent(Event *e)
{
  Event *ev = e;
  
  while ((ev != parent->eq->frontPtr) && (!ev->cpData))
    ev = ev->prev;
  if (ev != parent->eq->frontPtr)
    return ev;
  else {
    CkPrintf("[%d] WARNING: chpt::getCommitEvent: event is not checkpointed nor are any events prior to it; returning NULL event\n", CkMyPe());
    return NULL;
  }
}

template<class StateType>
void chpt<StateType>::pup(PUP::er &p)
{
  rep::pup(p);
  p(store); p(rate_store);
}

template<class StateType>
void chpt<StateType>::dump(int pdb_level)
{ 
  rep::dump(pdb_level+1); pdb_indent(pdb_level); 
  CkPrintf("[CHPT: store=%d rate=1/%d]\n", store, rate_store); 
}

#endif
