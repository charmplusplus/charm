// File: opt.h
// Module for optimistic simulation strategy class
// Last Modified: 09.12.01 by Terry L. Wilmarth

#ifndef OPT_H
#define OPT_H

class opt : public strat {
protected:
  int timeLeash;  // time units ahead of GVT an object can progress
  int eventLeash; // # of events w/timestamp > GVT an object can execute
  virtual void Rollback();              // rollback to predetermined RBevent
  virtual void RecoverState(Event *ev); // recover state prior to ev
  virtual void CancelEvents();          // cancel events in cancellation list
  virtual void UndoEvent(Event *e);     // undo single event, cancelling spawn
public:
  opt();
  void initSync() { parent->sync = OPTIMISTIC; }
  virtual void Step();              // single forward execution step
  int SafeTime();
  void AddSpawnedEvent(int AnObjIdx, eventID evID, int ts) { 
    // note spawn in event
    eq->AddSpawnToCurrent(AnObjIdx, evID, ts);
  }
  void CancelSpawn(Event *e) {  
    // send cancel messages to all of event e's spawn
    cancelMsg *m;
    SpawnedEvent *ev = e->spawnedList;
    while (ev) {
      e->spawnedList = ev->next;               // remove a spawn from the list
      ev->next = NULL;
      m = new cancelMsg();                     // build a cancel message
      m->evID = ev->evID;
      m->timestamp = ev->timestamp;
      m->setPriority(m->timestamp - INT_MAX);
      localPVT->objUpdate(ev->timestamp, SEND);
      POSE_Objects[ev->objIdx].Cancel(m);      // send the cancellation
      delete ev;                               // delete the spawn
      ev = e->spawnedList;                     // move on to next in list
    }
  }
};

#endif
