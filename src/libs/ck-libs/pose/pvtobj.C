// File: pvtobj.C
// Defines pvtObjects, a list that holds records of objects registered with 
// a PVT.
// Last Modified: 5.30.01 by Terry L. Wilmarth

#include "pose.h"

// Basic initialization:  preallocates space for 100 objects
pvtObjects::pvtObjects() 
{ 
  numObjs = numSpaces = firstEmpty = 0; 
  size = 100;
  if (!(objs = (pvtObjectNode *)malloc(100 * sizeof(pvtObjectNode)))) {
    CkPrintf("ERROR: pvtObjects::pvtObjects: OUT OF MEMORY!\n");
    CkExit();
  }
  for (int i=0; i<size; i++) {
    objs[i].present = 0;
    objs[i].localObjPtr = NULL;
  }
}

// Set all objects to idle (ovt = -1) in preparation for a PVT cycle
void pvtObjects::SetIdle()
{
  for (int i=0; i<numSpaces; i++)
    if (objs[i].present)
      objs[i].ovt = -1;
}

// Wake all objects up
void pvtObjects::Wake()
{
  for (int i=0; i<numSpaces; i++)
    if (objs[i].present) 
      (objs[i].localObjPtr)->Status();
}
  
// Call commit for all objects
void pvtObjects::Commit()
{
  for (int i=0; i<numSpaces; i++)
    if (objs[i].present)
      (objs[i].localObjPtr)->Commit();
}
  
// Insert an object in the list in the firstEmpty slot, expanding the list
// size if necessary
int pvtObjects::Insert(int index, int ovt, int sync, sim *myPtr)
{
  int idx, i;
  if (numObjs < size) { // insert in empty space
    idx = firstEmpty;
    if (firstEmpty == numSpaces) // all spaces occupied up to end of list
      numSpaces++;  // use a previously unused space
    objs[idx].index = index;
    objs[idx].ovt = ovt;
    objs[idx].present = 1;
    objs[idx].sync = sync;
    objs[idx].localObjPtr = myPtr;
    numObjs++;
    for (i=firstEmpty+1; i<size; i++)  // reset firstEmpty
      if (objs[i].present == 0) {
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
      objs[i].present = 0;
    idx = firstEmpty;  // insert new object at firstEmpty
    objs[idx].index = index;
    objs[idx].ovt = ovt;
    objs[idx].present = 1;
    objs[idx].sync = sync;
    objs[idx].localObjPtr = myPtr;
    numObjs++;
    numSpaces++;
    firstEmpty++;
  }   
  return idx;
}

// Delete an object from the list
void pvtObjects::Delete(int idx)
{
  objs[idx].present = 0;
  objs[idx].localObjPtr = NULL;
  numObjs--;
  if (idx < firstEmpty)  // recalculate firstEmpty
    firstEmpty = idx;
}

// Print out the list contents
void pvtObjects::dump()
{
  CkPrintf("numObjs=%d numSpaces=%d firstEmpty=%d size=%d\n", 
	   numObjs, numSpaces, firstEmpty, size);
  for (int i=0; i<numSpaces; i++) {
    CkPrintf("[%d] ", i);
    objs[i].dump();
    CkPrintf("\n");
  }
}
