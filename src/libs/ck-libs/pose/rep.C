// File: rep.C
// Module for basic representation class: what the user class becomes. It adds
// minimal  functionality to the user's code, mostly OVT, and links back to 
// the parent sim object.
// Last Modified: 06.05.01 by Terry L. Wilmarth

#include "pose.h"

void rep::init(eventMsg *m)
{ 
  ovt = m->timestamp; 
  parent = m->parent; 
  myStrat = m->str;
  myHandle = parent->thisIndex;
}

// timestamps event message, sets priority, and makes a record of the send
void rep::registerTimestamp(int idx, eventMsg *m, int offset)
{
  m->Timestamp(ovt+offset);
  m->setPriority(ovt+offset-INT_MAX);
  m->fromPE = CkMyPe();  // for comm-based LB
  parent->registerSent(ovt+offset);
}

// get event to rollback to
Event *rep::getCommitEvent(Event *e)
{
  Event *ev = e;
  
  if (ev != parent->eq->frontPtr)
    return ev;
  return NULL;
}

// sets checkpoint rate to 1/1 for chpt
void rep::CheckpointAll() { }

// sets checkpoint rate to default for chpt
void rep::ResetCheckpointRate() { }

// dump the entire rep object: called by objects inheriting from rep
void rep::dump(int pdb_level) 
{ 
  pdb_indent(pdb_level); 
  CkPrintf("[REP: ovt=%d]\n", ovt); 
}
