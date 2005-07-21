
#include "MultiRingMulticast.h"

//Array Constructor
MultiRingMulticast::MultiRingMulticast(CkArrayID dest_aid, int flag)
    : DirectMulticastStrategy(dest_aid, flag){
}


void MultiRingMulticast::pup(PUP::er &p){

    DirectMulticastStrategy::pup(p);
}

void MultiRingMulticast::beginProcessing(int  nelements){

    DirectMulticastStrategy::beginProcessing(nelements);
}

/*
inline int getMyId(int *pelist, int npes, int mype) {
    int myid = -1;

    for(int count = 0; count < npes; count ++)
        if(mype == pelist[count])
            myid = count;

    //if(myid == -1)
    //  CkPrintf("Warning myid = -1\n");

    return myid;
}


//Assumes a sorted input. Returns the next processor greater a given
//current pe
inline void getNextPe(int *pelist, int npes, int mype, int &nextpe) {
    
    nextpe = pelist[0];
    
    int count= 0;
    for(count = 0; count < npes; count++)
        if(pelist[count] > mype)
            break;
    
    if(count < npes) 
        nextpe = pelist[count];
    
    return;
}


inline int getMidPe(int *pelist, int npes, int src_pe) {
    
    int my_id = 0;
    int mid_pe = -1;
    
    my_id = getMyId(pelist, npes, src_pe);    
    
    CkAssert(my_id >= 0 && my_id < npes);

    if(my_id < npes/2)
        mid_pe = pelist[npes/2 + my_id];        
    else
        mid_pe = pelist[my_id % (npes/2)];
    
    //if(mid_pe == -1)
    //  CkPrintf("Warning midpe = -1\n");

    return mid_pe;
}
*/

//Unlike ring the source here sends two or more messages while all
//elements along the ring only send one.

ComlibSectionHashObject *MultiRingMulticast::createObjectOnSrcPe
(int nelements, CkArrayIndexMax *elements){

    ComlibSectionHashObject *obj = new ComlibSectionHashObject();

    obj->npes = 0;
    obj->pelist = 0;

    int *pelist;
    int npes;
    sinfo.getPeList(nelements, elements, npes, pelist);
    
    sinfo.getLocalIndices(nelements, elements, obj->indices);

    if(npes == 0)
        return obj;

    if(npes < 4) {
        // direct sending, take out ourself from the list!
	for (int i=0; i<npes; ++i) {
	  if (pelist[i] == CkMyPe()) {
	    pelist[i] = pelist[--npes];
	    break;
	  }
	}
        obj->npes = npes;
        obj->pelist = pelist;
	//CkPrintf("MultiRingMulticast::createObjectOnSrcPe, less than 4 procs\n");

        return obj;
    }
    
    //CkPrintf("MultiRingMulticast::createObjectOnSrcPe, more than 3 procs\n");
    //pelist[npes ++] = CkMyPe();
    qsort(pelist, npes, sizeof(int), intCompare);

    /*
      char dump[2560];
      sprintf(dump, "Section on %d : ", CkMyPe());
      for(int count = 0; count < npes; count ++) {
      sprintf(dump, "%s, %d", dump, pelist[count]);
      }
    
      CkPrintf("%s\n\n", dump);
    */
    
    int myid = -1; // getMyId(pelist, npes, CkMyPe());    
    for (int i=0; i<npes; ++i) {
      if (pelist[i] == CkMyPe()) {
	myid = i;
	break;
      }
    }

    //CkAssert(myid >= 0 && myid < npes);

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
    
    if(pelist[next_id] != CkMyPe()) {
        obj->pelist = new int[2];
        obj->npes = 2;
        
        obj->pelist[0] = pelist[next_id];
        obj->pelist[1] = pelist[mid_id];
    }
    else {
        CkAbort("Warning Should not be here !!!!!!!!!\n");
        //obj->pelist = new int[1];
        //obj->npes = 1;
        
        //obj->pelist[0] = midpe;
    }
    
    delete [] pelist;

    //CkPrintf("%d Src = %d Next = %d Mid Pe =%d\n", CkMyPe(), CkMyPe(), pelist[next_id], pelist[mid_id]);
    
    return obj;
}


ComlibSectionHashObject *MultiRingMulticast::createObjectOnIntermediatePe(int nindices, CkArrayIndexMax *idxlist, int npes, ComlibMulticastIndexCount *counts, int srcpe) {

    ComlibSectionHashObject *obj = new ComlibSectionHashObject();

    //CkPrintf("MultiRingMulticast: creating object on intermediate Pe %d-%d (%d-%d)\n", CkMyPe(),srcpe, npes,nindices);
    //int *pelist;
    //int npes;
    //sinfo.getRemotePelist(nelements, elements, npes, pelist);
    
    obj->pelist = 0;
    obj->npes = 0;
    for (int i=0; i<nindices; ++i) obj->indices.insertAtEnd(idxlist[i]);
    //sinfo.getLocalIndices(nelements, elements, obj->indices);

    //pelist[npes ++] = CkMyPe();

    if(npes < 4)
        return obj;
    
    //qsort(counts, npes, sizeof(ComlibMulticastIndexCount), indexCountCompare);
    
    int myid = -1;
    for (int i=0; i<npes; ++i) {
      if (counts[i].pe == CkMyPe()) {
	myid = i;
	break;
      }
    }
    //getMyId(pelist, npes, CkMyPe());
    
    CkAssert(myid >= 0 && myid < npes);

    int breaking = npes/2;
    int srcid = 0; // = getMyId(pelist, npes, src_pe);
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

    return obj;
}
