/// Data types for events and spawned events
#include "pose.h"

/// Basic Constructor
Event::Event() 
{ 
  spawnedList = NULL; msg = NULL; commitBfr = NULL; cpData = NULL;
  next = prev = NULL;
  commitBfrLen = commitErr = done = 0;
  fnIdx = timestamp = -1;
}

/// Destructor
Event::~Event() 
{                                   
  SpawnedEvent *tmp;
  //  double start=CmiWallTimer();
  delete msg;
  //double elapsed=CmiWallTimer()-start;
  //if (elapsed>50e-6)
  //CkPrintf("delete msg(%p) took %.6f\n", (void *)msg,elapsed);
  free(commitBfr);
  while (spawnedList) {  // purge list of spawned events
    tmp = spawnedList;
    spawnedList = spawnedList->next;
    delete tmp;
  }
  // NOTE: cpData is deleted in evq::Commit
}

/// Pack/unpack/sizing operator
void Event::pup(PUP::er &p) 
{
  int msgSize, spawn = 0;
  SpawnedEvent *tmp = NULL;
  
  evID.pup(p); p(fnIdx); p(timestamp); p(done); p(commitBfrLen); p(commitErr);
  
  // commitBfr
  if (p.isUnpacking() && (commitBfrLen > 0)) { // unpack non-empty commitBfr
    commitBfr = new char[commitBfrLen];
    p(commitBfr, commitBfrLen);
  }
  else if (commitBfrLen > 0) // pack/size non-empty commitBfr
    p(commitBfr, commitBfrLen);
  else // unpack empty commitBfr
    commitBfr = NULL;
  
  // msg
  if (p.isUnpacking()) { // unpack msg
    p(msgSize); // unpack msgSize
    if (msgSize > 0) { // if nonzero, unpack msg
      msg = (eventMsg *)CmiAlloc(msgSize); // allocate space
      p(msg, msgSize); // unpack into space
      msg = (eventMsg *)EnvToUsr((envelope *)msg); // reset msg pointer
    }
    else // empty message
      msg = NULL; // set msg to null
  }
  else { // pack msg
    if (msg != NULL) { // msg is not null
      msgSize = (UsrToEnv(msg))->getTotalsize(); // get msg size
      p(msgSize); // pack msg size
      p(UsrToEnv(msg), msgSize); // pack from start of envelope
    }
    else { // msg is null
      msgSize = 0;                           
      p(msgSize); // pack size of zero
    }
  }
  
  // spawnedList
  if (p.isUnpacking()) { // unpack spawnedList
    p(spawn); // unpack spawn count
    if (spawn > 0) spawnedList = tmp = new SpawnedEvent();
    else spawnedList = NULL;
    while (spawn > 0) { // unpack each spawned event record
      tmp->pup(p);
      tmp->next = NULL;
      spawn--;
      if (spawn > 0) {
	tmp->next = new SpawnedEvent();
	tmp = tmp->next;
      }
    }
  }
  else { // pack/size spawnedList
    tmp = spawnedList;
    while (tmp != NULL) { // count the spawn
      spawn++;
      tmp = tmp->next;
    }
    p(spawn); // pack the spawn count
    tmp = spawnedList;
    while (tmp != NULL) { // pack each spawn
      tmp->pup(p);
      tmp = tmp->next;
    }
  }
  
  if (p.isUnpacking())
    cpData = NULL; // to be set later
}

/// Check validity of data fields
void Event::sanitize()
{
  // check that evID is kosher
  // evID->sanitize();   
  // check fnIdx... not clear how
  // check timestamp... not clear how
  /// Execution status: 0=not done, 1=done, 2=executing
  CmiAssert((done == 0) || (done == 1) || (done == 2));
  CmiAssert(commitBfrLen >= 0);
  CmiAssert((commitErr == 0) || (commitErr == 1));
  /// check commitBfr... not sure how
  /// check msg... not sure how
  /// check spawnedList... later
  /// check cpData... not sure how
  /// check links
  CmiAssert(next->prev == this);
  CmiAssert(prev->next == this);
}
