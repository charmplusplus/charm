// File: evq.h
// Defines: eventQueue class and methods
// Last Modified: 06.04.01 by Terry L. Wilmarth

#ifndef EVQ_H
#define EVQ_H

// The event queue: needs a bit of access to the sim that owns it
// Structure: front and back sentinel nodes, doubly linked. 
class eventQueue {
 public:
  EqHeap *eqh;
  Event *frontPtr, *currentPtr, *backPtr, *commitPtr;
  int eqCount;
  
  eventQueue();
  ~eventQueue();
  void InsertEvent(Event *e);      // Insert e in timestamp order.
  void ShiftEvent();               // Moves currentPtr to next event
  void CommitEvents(sim *obj, int ts); // Commit events before target
  void SetCurrentPtr(Event *e);    // Changes what currentPtr points to
  Event *RecomputeRollbackTime();  // Get 1st unexec'ed event before currentPtr
  void DeleteEvent(Event *ev);     // Delete ev and make sure to reconnect
  const eventID& CurrentEventID() { return currentPtr->evID; }
  // Add e and ts as an entry in currentPtr's spawned list
  void AddSpawnToCurrent(int id, eventID e, int ts);
  // Return the first entry in currentPtr's spawned list and remove it
  SpawnedEvent *GetNextCurrentSpawn();
  void dump(int pdb_level);        // Print contents of eventQueue to stdout
  void pup(PUP::er &p);            // Pup the entire event queue
};

#endif
