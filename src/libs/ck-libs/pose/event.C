// File: event.C
// Defines: Event and Spawned Event
// Last Modified: 06.04.01 by Terry L. Wilmarth

#include "pose.h"

// initializing constructor
SpawnedEvent::SpawnedEvent() 
{ 
  objIdx = timestamp = -1; next = NULL;
}

// Set constructor
SpawnedEvent::SpawnedEvent(int idx, eventID e, int ts, SpawnedEvent *n) 
{
  objIdx = idx;  evID = e;  timestamp = ts;  next = n;
}

// basic initializing constructor
Event::Event() 
{ 
  spawnedList = NULL; msg = NULL; commitBfr = NULL; 
  cpData = NULL;
  next = prev = NULL;
  commitBfrLen = done = 0;
  fnIdx = timestamp = -1;
}

// basic destructor
Event::~Event() 
{                                   
  SpawnedEvent *tmp;

  if (msg)  delete msg;
  msg = NULL;
  free(commitBfr);
  cpData = NULL;
  while (spawnedList) {  // purge list of spawned events
    tmp = spawnedList;
    spawnedList = spawnedList->next;
    delete tmp;
  }
  spawnedList = NULL;
}

// pup entire event
void Event::pup(PUP::er &p) 
{
  int msgSize, spawn = 0;
  SpawnedEvent *tmp = NULL;
  
  // simple data
  evID.pup(p); p(fnIdx); p(timestamp); p(done); p(commitBfrLen); 
  
  // commitBfr
  if (p.isUnpacking() && (commitBfrLen > 0)) {  // unpack non-empty commitBfr
    commitBfr = new char[commitBfrLen];
    p(commitBfr, commitBfrLen);
  }
  else if (commitBfrLen > 0)                 // pack/size non-empty commitBfr
    p(commitBfr, commitBfrLen);
  else                                       // unpack empty commitBfr
    commitBfr = NULL;
  
  // msg
  if (p.isUnpacking()) {                     // unpack msg
    p(msgSize);                              // unpack msgSize
    if (msgSize > 0) {                       // if nonzero, unpack msg
      msg = (eventMsg *)CmiAlloc(msgSize);   // allocate space
      p(msg, msgSize);                       // unpack into space
      msg = (eventMsg *)EnvToUsr((envelope *)msg);  // reset msg pointer
    }
    else                                     // empty message
      msg = NULL;                            // set msg to null
  }
  else {                                     // pack msg
    if (msg != NULL) {                       // msg is not null
      msgSize = (UsrToEnv(msg))->getTotalsize();  // get msg size
      p(msgSize);                            // pack msg size
      p(UsrToEnv(msg), msgSize);             // pack from start of envelope
    }
    else {                                   // msg is null
      msgSize = 0;                           
      p(msgSize);                            // pack size of zero
    }
  }
  
  // spawnedList
  if (p.isUnpacking()) {                     // unpack spawnedList
    p(spawn);                                // unpack spawn count
    if (spawn > 0)
      spawnedList = tmp = new SpawnedEvent();
    else
      spawnedList = NULL;
    while (spawn > 0) {                   // unpack each spawned event record
      tmp->pup(p);
      tmp->next = NULL;
      spawn--;
      if (spawn > 0) {
	tmp->next = new SpawnedEvent();
	tmp = tmp->next;
      }
    }
  }
  else {                                     // pack/size spawnedList
    tmp = spawnedList;
    while (tmp != NULL) {                    // count the spawn
      spawn++;
      tmp = tmp->next;
    }
    p(spawn);                                // pack the spawn count
    tmp = spawnedList;
    while (tmp != NULL) {                    // pack each spawn
      tmp->pup(p);
      tmp = tmp->next;
    }
  }
  
  if (p.isUnpacking())
    cpData = NULL;      // to be set later
}
