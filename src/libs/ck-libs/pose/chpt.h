// File: chpt.h
// Module for checkpointing representation to be used with optimistic strategy
// Last Modified: 07.31.01 by Terry L. Wilmarth

#ifndef CHPT_H
#define CHPT_H

//----------------------------------------------------------------------------
// rep class for checkpointing
template<class StateType> class chpt : public rep {
 public:
  chpt() { }
  virtual ~chpt() { }
  // timestamps event message, sets priority, and makes a record of the send
  void registerTimestamp(int idx, eventMsg *m, int offset);
  void checkpoint(StateType *data);          // checkpoint the data
  void restore(StateType *data);             // restore checkpointed data 
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

template<class StateType>
void chpt<StateType>::checkpoint(StateType *data)
{
  static int sinceLast = 0;
#ifdef POSE_STATS_ON
  localStat *localStats = (localStat *)CkLocalBranch(theLocalStats);
  localStats->SwitchTimer(CP_TIMER);
#endif
  if (parent->myStrat->currentEvent->cpData) {
    CkPrintf("ERROR: chpt::checkpoint: cpData exists already\n");
    CkExit();
  }

  if ((myStrat->currentEvent->timestamp > 
       myStrat->currentEvent->prev->timestamp) || (sinceLast == STORE_RATE)) {
    myStrat->currentEvent->cpData = new StateType;
    myStrat->currentEvent->cpData->copy = 1;
    *((StateType *)myStrat->currentEvent->cpData) = *data;
    sinceLast = 0;
  }
  else sinceLast++;
  
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
}

template<class StateType>
void chpt<StateType>::dump(int pdb_level)
{ 
  rep::dump(pdb_level+1); pdb_indent(pdb_level); 
}

#endif
