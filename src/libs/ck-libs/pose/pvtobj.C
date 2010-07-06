/// pvtObjects: a list to hold records of posers registered with a PVT branch.
#include "pose.h"

/// Check validity of data fields
void pvtObjectNode::sanitize() 
{
  if (present) {
    CmiAssert(ovt >= -1);
    CmiAssert(index >= 0);
    CmiAssert((sync == OPTIMISTIC) || (sync == CONSERVATIVE));
    CmiAssert((localObjPtr != NULL) && (localObjPtr->IsActive() < 2));
  }
}

/// Basic Constructor: preallocates space for 100 objects
pvtObjects::pvtObjects() 
{ 
  register int i;
  numObjs = numSpaces = firstEmpty = 0; 
  size = 100;
  if (!(objs = (pvtObjectNode *)malloc(100 * sizeof(pvtObjectNode)))) {
    CkPrintf("ERROR: pvtObjects::pvtObjects: OUT OF MEMORY!\n");
    CkExit();
  }
  for (i=0; i<size; i++) objs[i].set(POSE_UnsetTS, POSE_UnsetTS, false, 0, NULL);
}

/// Insert poser in list
int pvtObjects::Insert(int index, POSE_TimeType ovt, int sync, sim *myPtr)
{
  int idx;
  register int i;
  if (numObjs < size) { // insert in empty space
    idx = firstEmpty;
    if (firstEmpty == numSpaces) // all spaces occupied up to end of list
      numSpaces++;  // use a previously unused space
    objs[idx].set(ovt, index, true, sync, myPtr);
    numObjs++;
    for (i=firstEmpty+1; i<size; i++)  // reset firstEmpty
      if (!objs[i].isPresent()) {
	firstEmpty = i;
	break;
      }
  }
  else { // no free spaces; expand objs
    firstEmpty = size;  // this is where firstEmpty will be after expansion
    size += 50;  // expand by 50
    if (!(objs = 
	  (pvtObjectNode *)realloc(objs, size * sizeof(pvtObjectNode)))) {
      CkPrintf("ERROR: pvtObjects::Insert: OUT OF MEMORY!\n");
      CkExit();
    }
    for (i=firstEmpty; i<size; i++)  // initialize new slots to empty
      objs[i].set(POSE_UnsetTS, POSE_UnsetTS, false, 0, NULL);
    idx = firstEmpty;  // insert new object at firstEmpty
    objs[idx].set(ovt, index, true, sync, myPtr);
    numObjs++;
    numSpaces++;
    firstEmpty++;
  }   
  return idx;
}

void pvtObjects::callAtSync() {
  register int i;
  for (i=0; i<numSpaces; i++)
    if (objs[i].isPresent()) {
      (objs[i].localObjPtr)->AtSync();
    }
}
/// Wake up all posers in list
void pvtObjects::Wake() {
  register int i;
  for (i=0; i<numSpaces; i++)
    if (objs[i].isPresent()) (objs[i].localObjPtr)->Status();
}
/// Call Commit on all posers
void pvtObjects::Commit() {
  register int i;
  for (i=0; i<numSpaces; i++)
    if (objs[i].isPresent()) (objs[i].localObjPtr)->Commit();
}
/// Call CheckpointCommit on all posers
void pvtObjects::CheckpointCommit() {
  register int i;
  for (i=0; i<numSpaces; i++)
    if (objs[i].isPresent()) (objs[i].localObjPtr)->CheckpointCommit();
}

/// Dump data fields
void pvtObjects::dump()
{
  register int i;
  CkPrintf("numObjs=%d numSpaces=%d firstEmpty=%d size=%d\n", 
	   numObjs, numSpaces, firstEmpty, size);
  for (i=0; i<numSpaces; i++) {
    CkPrintf("[%d] ", i);
    objs[i].dump();
    CkPrintf("\n");
  }
}

/// Check validity of data fields
void pvtObjects::sanitize() 
{
  register int i;
  CmiAssert(numObjs >= 0);
  CmiAssert(numSpaces >= 0);
  CmiAssert(size >= 0);
  CmiAssert(firstEmpty >= 0);
  CmiAssert(numObjs <= numSpaces);
  CmiAssert(numSpaces <= size);
  CmiAssert(firstEmpty < numSpaces);
  for (int i=0; i<numSpaces; i++) objs[i].sanitize();
}
