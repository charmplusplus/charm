// File: lbObject.C
// Defines lbObjects, a list that holds records of objects registered with 
// a load balancer
// Last Modified: 11/05.01 by Terry L. Wilmarth

#include "pose.h"

// Basic initialization: preallocates space for 100 objects
lbObjects::lbObjects() 
{ 
  numObjs = numSpaces = firstEmpty = 0; 
  size = 100;
  if (!(objs = (lbObjectNode *)malloc(100 * sizeof(lbObjectNode)))) {
    CkPrintf("ERROR: lbObjects::lbObjects: OUT OF MEMORY!\n");
    CkExit();
  }
  for (int i=0; i<size; i++) {
    objs[i].present = 0;
    objs[i].eet = objs[i].ne = objs[i].execPrio = 0;
    objs[i].rbOh = 0.0;
    objs[i].comm = (int *)malloc(CkNumPes()*sizeof(int));
    for (int j=0; j<CkNumPes(); j++) objs[i].comm[j] = 0;
    objs[i].totalComm = objs[i].localComm = objs[i].remoteComm = 
      objs[i].maxComm = 0;
    objs[i].maxCommPE = -1;
  }
}

// Insert an object in the list in the firstEmpty slot, expanding the list
// size if necessary
int lbObjects::Insert(int sync, int index, sim *myPtr)
{
  int idx, i;
  if (numObjs < size) { // insert in empty space
    idx = firstEmpty;
    if (firstEmpty == numSpaces) // all spaces occupied up to end of list
      numSpaces++;  // use a previously unused space
    objs[idx].index = index;
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
	  (lbObjectNode *)realloc(objs, size * sizeof(lbObjectNode)))) {
      CkPrintf("ERROR: lbObjects::Insert: OUT OF MEMORY!\n");
      CkExit();
    }
    for (i=firstEmpty; i<size; i++) {  // initialize new slots to empty
      objs[i].present = 0;
      objs[i].eet = objs[i].ne = objs[i].execPrio = 0;
      objs[i].rbOh = 0.0;
      objs[i].comm = (int *)malloc(CkNumPes()*sizeof(int));
      for (int j=0; j<CkNumPes(); j++) objs[i].comm[j] = 0;
      objs[i].totalComm = objs[i].localComm = objs[i].remoteComm = 
	objs[i].maxComm = 0;
      objs[i].maxCommPE = -1;
    }

    idx = firstEmpty;  // insert new object at firstEmpty
    objs[idx].index = index;
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
void lbObjects::Delete(int idx)
{
  objs[idx].present = 0;
  objs[idx].localObjPtr = NULL;
  numObjs--;
  if (idx < firstEmpty)  // recalculate firstEmpty
    firstEmpty = idx;
}

void lbObjects::UpdateEntry(int idx, POSE_TimeType ovt, POSE_TimeType eet, int ne, double rbOh, 
			    int *srVec)
{
  if (objs[idx].present) {
    objs[idx].ovt = ovt;
    objs[idx].eet = eet;
    objs[idx].ne = ne;
    objs[idx].rbOh = rbOh;
    for (int i=0; i<CkNumPes(); i++) 
      AddComm(idx, i, srVec[i]);
  }
  else CkPrintf("ERROR: lbObjects::UpdateEntry: No such object exists.\n");
}

// add msg s/r to obj idx, t/f pe
void lbObjects::AddComm(int idx, int pe, int sr) 
{
  if (objs[idx].present) {
    objs[idx].comm[pe] += sr;
    objs[idx].totalComm += sr;
    if (pe == CkMyPe())
      objs[idx].localComm += sr;
    else objs[idx].remoteComm += sr;
  }
  else CkPrintf("ERROR: lbObjects::AddComm: No such object exists.\n");
}

// reset comm array entries to 0
void lbObjects::ResetComm()
{
  for (int i=0; i<numSpaces; i++)
    if (objs[i].present) {
      for (int j=0; j<CkNumPes(); j++)
	objs[i].comm[j] = 0;
      objs[i].totalComm = objs[i].localComm = objs[i].remoteComm =
	objs[i].maxComm = 0;
      objs[i].maxCommPE = -1;
    }
}

void lbObjects::RequestReport()
{
  for (int i=0; i<numSpaces; i++)
    if (objs[i].present)
      (objs[i].localObjPtr)->ReportLBdata();
}

// Print out the list contents
void lbObjects::dump()
{
  CkPrintf("numObjs=%d numSpaces=%d firstEmpty=%d size=%d\n", 
	   numObjs, numSpaces, firstEmpty, size);
  for (int i=0; i<numSpaces; i++) {
    CkPrintf("[%d] ", i);
    objs[i].dump();
    CkPrintf("\n");
  }
}
