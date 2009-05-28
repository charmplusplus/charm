/**
   @addtogroup ComlibCharmStrategy
   @{
   
   @file 
*/

#include "MultiRingMulticast.h"


//Unlike ring the source here sends two or more messages while all
//elements along the ring only send one.

void MultiRingMulticastStrategy::createObjectOnSrcPe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *pelist) {

  obj->npes = 0;
  obj->pelist = 0;

  if(npes == 0) return;

  if(npes < 4) {
    // direct sending, take out ourself from the list!
    obj->npes = npes;
    for (int i=0; i<npes; ++i) if (pelist[i].pe == CkMyPe()) obj->npes --;
    obj->pelist = new int[obj->npes];
    for (int i=0, count=0; i<npes; ++i) {
      if (pelist[i].pe != CkMyPe()) {
	obj->pelist[count] = pelist[i].pe;
	count++;
      }
    }
    return;
  }
    
  int myid = -1; // getMyId(pelist, npes, CkMyPe());    
  for (int i=0; i<npes; ++i) {
    if (pelist[i].pe == CkMyPe()) {
      myid = i;
      break;
    }
  }

  int breaking = npes/2; /* 0 : breaking-1    is the first ring
			    breaking : npes-1 is the second ring
			 */

  int next_id = myid + 1;
  // wrap nextpe around the ring
  if(myid < breaking) {
    if (next_id >= breaking) next_id = 0;
  } else {
    if (next_id >= npes) next_id = breaking;
  }
    
  int mid_id;
  if (myid < breaking) {
    mid_id = myid + breaking;
    if (mid_id < breaking) mid_id = breaking;
  } else {
    mid_id = myid - breaking;
    if (mid_id >= breaking) mid_id = 0;
  }
  //mid_pe = getMidPe(pelist, npes, CkMyPe());
    
  if(pelist[next_id].pe != CkMyPe()) {
    obj->pelist = new int[2];
    obj->npes = 2;
        
    obj->pelist[0] = pelist[next_id].pe;
    obj->pelist[1] = pelist[mid_id].pe;
  }
  else {
    CkAbort("Warning Should not be here !!!!!!!!!\n");
  }
  
  //CkPrintf("%d Src = %d Next = %d Mid Pe =%d\n", CkMyPe(), CkMyPe(), pelist[next_id], pelist[mid_id]);
  
  return;
}

void MultiRingMulticastStrategy::createObjectOnIntermediatePe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *counts, int srcpe) {

  obj->pelist = 0;
  obj->npes = 0;

  if(npes < 4) return;

  // where are we inside the list?
  int myid = -1;
  for (int i=0; i<npes; ++i) {
    if (counts[i].pe == CkMyPe()) {
      myid = i;
      break;
    }
  }

  // we must be in the list!
  CkAssert(myid >= 0 && myid < npes);

  int breaking = npes/2;
  int srcid = 0;
  for (int i=0; i<npes; ++i) {
    if (counts[i].pe == srcpe) {
      srcid = i;
      break;
    }
  }

  if (srcid < breaking ^ myid < breaking) {
    // if we are in the two different halves, correct srcid
    if (srcid < breaking) {
      srcid += breaking;
      if (srcid < breaking) srcid = breaking;
    } else {
      srcid -= breaking;
      if (srcid >= breaking) srcid = 0;
    }
  }
  // now srcid is the starting point of this half ring, which could be the
  // original sender itself (0 if the sender is not part of the recipients),
  // or the counterpart in the other ring

  int nextid = myid + 1;
  // wrap nextpe around the ring
  if(myid < breaking) {
    if (nextid >= breaking) nextid = 0;
  }
  else {
    if (nextid >= npes) nextid = breaking;
  }

  if (nextid != srcid) {
    obj->pelist = new int[1];
    obj->npes = 1;
    obj->pelist[0] = counts[nextid].pe;
  }

  return;
}

/*@}*/
