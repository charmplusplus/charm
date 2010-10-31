/**
   @addtogroup CharmComlib
   @{ 
   @file
  
   Implementations of classes defined in ComlibSectionInfo.h
*/

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"
#ifdef __MINGW_H 
#include <malloc.h> 
#endif


#define USE_CONTROL_POINTS 0

#if USE_CONTROL_POINTS
#include "controlPoints.h"
#endif

#if CMK_HAS_ALLOCA_H
#include <alloca.h>
#endif



/**
   Create a new multicast message based upon the section info stored inside cmsg.
*/
ComlibMulticastMsg * ComlibSectionInfo::getNewMulticastMessage(CharmMessageHolder *cmsg, int needSort, int instanceID){
    
  cmsg->checkme();

    if(cmsg->sec_id == NULL || cmsg->sec_id->_nElems == 0)
        return NULL;

    void *m = cmsg->getCharmMessage();
    envelope *env = UsrToEnv(m);
        
    // Crate a unique identifier for section id in cmsg->sec_id
    initSectionID(cmsg->sec_id);   

    CkPackMessage(&env);

    const CkArrayID destArrayID(env->getsetArrayMgr());
    int nRemotePes=-1, nRemoteIndices=-1;
    ComlibMulticastIndexCount *indicesCount;
    int *belongingList;


    //  Determine the last known locations of all the destination objects.
    getPeCount(cmsg->sec_id->_nElems, cmsg->sec_id->_elems, destArrayID, nRemotePes, nRemoteIndices, indicesCount, belongingList);

    //     if (nRemotePes == 0) return NULL;

#if 0
    CkPrintf("nRemotePes=%d\n", nRemotePes);
    CkPrintf("nRemoteIndices=%d\n",nRemoteIndices);
    CkPrintf("env->getTotalsize()=%d\n", env->getTotalsize());
    CkPrintf("cmsg->sec_id->_nElems=%d\n", cmsg->sec_id->_nElems);
 #endif

    int sizes[3];
    sizes[0] = nRemotePes;
    sizes[1] = nRemoteIndices; // only those remote ///cmsg->sec_id->_nElems;
    sizes[2] = env->getTotalsize();
    
    ComlibPrintf("Creating new comlib multicast message %d, %d %d\n", sizes[0], sizes[1], sizes[2]);
    
    ComlibMulticastMsg *msg = new(sizes, 0) ComlibMulticastMsg;
    msg->nPes = nRemotePes;
    msg->_cookie.sInfo.cInfo.instId = instanceID;
    msg->_cookie.sInfo.cInfo.id = MaxSectionID - 1;
    msg->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_NEW_SECTION;
    msg->_cookie.type = COMLIB_MULTICAST_MESSAGE;
    msg->_cookie.pe = CkMyPe();

    // fill in the three pointers of the ComlibMulticastMsg
    memcpy(msg->indicesCount, indicesCount, sizes[0] * sizeof(ComlibMulticastIndexCount));
    //memcpy(msg->indices, cmsg->sec_id->_elems, sizes[1] * sizeof(CkArrayIndexMax));

    CkArrayIndexMax **indicesPe = (CkArrayIndexMax**)alloca(nRemotePes * sizeof(CkArrayIndexMax*));

    if (needSort) {
    	// if we are sorting the array, then we need to fix the problem that belongingList
    	// refers to the original ordering! This is done by mapping indicesPe in a way coherent
    	// with the original ordering.
    	int previous, i, j;
    	qsort(msg->indicesCount, sizes[0], sizeof(ComlibMulticastIndexCount), indexCountCompare);

    	for (j=0; j<nRemotePes; ++j) if (indicesCount[j].pe == msg->indicesCount[0].pe) break;
    	indicesPe[j] = msg->indices;
    	previous = j;
    	for (i=1; i<nRemotePes; ++i) {
    		for (j=0; j<nRemotePes; ++j) if (indicesCount[j].pe == msg->indicesCount[i].pe) break;
    		indicesPe[j] = indicesPe[previous] + indicesCount[previous].count;
    		previous = j;
    	}
    } else {
    	indicesPe[0] = msg->indices;
    	for (int i=1; i<nRemotePes; ++i) indicesPe[i] = indicesPe[i-1] + indicesCount[i-1].count;
    }

    for (int i=0; i<cmsg->sec_id->_nElems; ++i) {
    	if (belongingList[i] >= 0) {
    		// If the object is located on a remote PE (-1 is local)
    		*indicesPe[belongingList[i]] = cmsg->sec_id->_elems[i];
    		indicesPe[belongingList[i]]++;
    	}
    }
    memcpy(msg->usrMsg, env, sizes[2] * sizeof(char));
    envelope *newenv = UsrToEnv(msg);
    delete [] indicesCount;
    delete [] belongingList;

    newenv->getsetArrayMgr() = env->getsetArrayMgr();
    newenv->getsetArraySrcPe() = env->getsetArraySrcPe();
    newenv->getsetArrayEp() = env->getsetArrayEp();
    newenv->getsetArrayHops() = env->getsetArrayHops();
    newenv->getsetArrayIndex() = env->getsetArrayIndex();

    // for trace projections
    newenv->setEvent(env->getEvent());
    newenv->setSrcPe(env->getSrcPe());
    
    return (ComlibMulticastMsg *)EnvToUsr(newenv);
}

void ComlibSectionInfo::getPeList(envelope *cb_env, int npes, int *&pelist)
{
    ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)EnvToUsr(cb_env);
    int i;
    
    CkAssert(npes==ccmsg->nPes);
    for (i=0; i<ccmsg->nPes; ++i) {
      pelist[i]=ccmsg->indicesCount[i].pe;
    }

}


void ComlibSectionInfo::unpack(envelope *cb_env,
			       int &nLocalElems,
                               CkArrayIndexMax *&dest_indices, 
                               envelope *&env) {
        
    ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)EnvToUsr(cb_env);
    int i;

    dest_indices = ccmsg->indices;
    for (i=0; i<ccmsg->nPes; ++i) {
      if (ccmsg->indicesCount[i].pe == CkMyPe()) break;
      dest_indices += ccmsg->indicesCount[i].count;
    }

    if(i >= ccmsg->nPes)
      {  //cheap hack for rect bcast
	nLocalElems=0;
	dest_indices=NULL;
      }
    else
      {
    nLocalElems = ccmsg->indicesCount[i].count;

    /*
    ComlibPrintf("Unpacking: %d local elements:",nLocalElems);
    for (int j=0; j<nLocalElems; ++j) ComlibPrintf(" %d",((int*)&dest_indices[j])[1]);
    ComlibPrintf("\n");
    */
    /*
    for(int count = 0; count < ccmsg->nIndices; count++){
        CkArrayIndexMax idx = ccmsg->indices[count];
        
        //This will work because. lastknown always knows if I have the
        //element of not
        int dest_proc = ComlibGetLastKnown(destArrayID, idx);
        //CkArrayID::CkLocalBranch(destArrayID)->lastKnown(idx);
        
        //        if(dest_proc == CkMyPe())
        dest_indices.insertAtEnd(idx);                        
    }
    */
      }
    envelope *usrenv = (envelope *) ccmsg->usrMsg;
    env = (envelope *)CmiAlloc(usrenv->getTotalsize());
    memcpy(env, usrenv, usrenv->getTotalsize());
    env->setEvent(cb_env->getEvent());
}


void ComlibSectionInfo::processOldSectionMessage(CharmMessageHolder *cmsg) {
  cmsg->checkme();

    ComlibPrintf("Process Old Section Message \n");

    int cur_sec_id = cmsg->sec_id->getSectionID();

    //Old section id, send the id with the message
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)cmsg->getCharmMessage();
    cbmsg->_cookie.pe = CkMyPe();
    cbmsg->_cookie.sInfo.cInfo.id = cur_sec_id;
    cbmsg->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_OLD_SECTION;
}

CkMcastBaseMsg *ComlibSectionInfo::getNewDeliveryErrorMsg(CkMcastBaseMsg *base) {
  CkMcastBaseMsg *dest= (CkMcastBaseMsg*)CkAllocMsg(0, sizeof(CkMcastBaseMsg), 0);
  memcpy(dest, base, sizeof(CkMcastBaseMsg));
  dest->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_SECTION_ERROR;
  return dest;
}

void ComlibSectionInfo::getPeList(int _nElems, 
                                  CkArrayIndexMax *_elems,
				  CkArrayID &destArrayID,
                                  int &npes, int *&pelist){

    int length = CkNumPes();
    if(length > _nElems)    //There will not be more processors than
                            //number of elements. This is wastage of
                            //memory as there may be fewer
                            //processors. Fix later.
        length = _nElems;
    
    pelist = new int[length];
    npes = 0;
    
    int count = 0, acount = 0;
    
    CkArray *a = (CkArray *)_localBranch(destArrayID);
    for(acount = 0; acount < _nElems; acount++){
        
      //int p = ComlibGetLastKnown(destArrayID, _elems[acount]);
	int p = a->lastKnown(_elems[acount]);
        
        if(p == -1) CkAbort("Invalid Section\n");        
        for(count = 0; count < npes; count ++)
            if(pelist[count] == p)
                break;
        
        if(count == npes) {
	  pelist[npes ++] = p;
        }
    }   
    
    if(npes == 0) {
      delete [] pelist;
      pelist = NULL;
    }
}



inline int getPErepresentingNodeContainingPE(int pe){

#if 1
    return pe;

#else

#if USE_CONTROL_POINTS
  std::vector<int> v;
  v.push_back(1);
  if(CkNumPes() >= 2)
    v.push_back(2);
  if(CkNumPes() >= 4)
    v.push_back(4);
  if(CkNumPes() >= 8)
    v.push_back(8);
  int pes_per_node = controlPoint("Number of PEs per Node", v);
#else
  int pes_per_node = 1;
#endif


    if(getenv("PE_PER_NODES") != NULL)
    	pes_per_node = CkNumPes()/atoi(getenv("PE_PER_NODES"));
	    
    if( pes_per_node > 1 && pes_per_node <= CkNumPes() ){
        ComlibPrintf("NODE AWARE Sending a message to a representative of the node instead of its real owner\n");
        int newpe = pe - (pe % pes_per_node);
        return newpe;
    } else {
    	return pe;
    }
#endif    
}

/** 
    Determine the last known locations of all the destination objects.

    Create two resulting arrays:
        1) counts -- contains pairs of (pe,count) that describe how many objects were found for each of the pe's
	2) belongs -- belongs[i] points to the owning pe's entry in the "counts" array.
	
*/
void ComlibSectionInfo::getPeCount(int nindices, CkArrayIndexMax *idxlist, 
		      const CkArrayID &destArrayID, int &npes, int &nidx,
		      ComlibMulticastIndexCount *&counts, int *&belongs) {

  int i;
    
  int length = CkNumPes();

  if(length > nindices) length = nindices;
    
  counts = new ComlibMulticastIndexCount[length];
  belongs = new int[nindices];
  npes = 0;
  nidx = 0;

  CkArray *a = (CkArray *)_localBranch(destArrayID);
  for(i=0; i<nindices; ++i){
    int p = a->lastKnown(idxlist[i]);

#define USE_NODE_AWARE 0
#if USE_NODE_AWARE
    //#warning "USING NODE AWARE SECTION INFOs"
    ComlibPrintf("NODE AWARE old p=%d\n", p);
  
    // p = getPErepresentingNodeContainingPE(p);
    
    ComlibPrintf("NODE AWARE new p=%d\n", p);
#endif
    
    if(p == -1) CkAbort("Invalid Section\n");        

    if(p == CkMyPe()) {
      belongs[i] = -1;
      continue;
    }

    //Collect processors
    int count = 0;
    for(count = 0; count < npes; count ++)
      if(counts[count].pe == p)
	break;
    
    if(count == npes) {
      counts[npes].pe = p;
      counts[npes].count = 0;
      ++npes;
    }

    ++nidx;
    counts[count].count++;
    belongs[i] = count;
  }
  //ComlibPrintf("section has %d procs\n",npes);

//   if(npes == 0) {
//     delete [] counts;
//     delete [] belongs;
//     counts = NULL;
//     belongs = NULL;
//   }
}


void ComlibSectionInfo::getRemotePelist(int nindices, 
                                        CkArrayIndexMax *idxlist,
					CkArrayID &destArrayID,
                                        int &npes, int *&pelist) {

	ComlibPrintf("ComlibSectionInfo::getRemotePelist()\n");
	
    int count = 0, acount = 0;
    
    int length = CkNumPes();

    // HACK FOR DEBUGGING
    /*pelist = new int[length-1];
    npes = length-1;
    for (acount=0; acount<length; acount++) {
      if (acount == CkMyPe()) continue;
      pelist[count]=acount;
      count++;
    }
    return;*/
    // END HACK

    if(length > nindices)
        length = nindices;
    
    pelist = new int[length+1];
    npes = 0;

    CkArray *a = (CkArray *)_localBranch(destArrayID);
    for(acount = 0; acount < nindices; acount++){
        
      //int p = ComlibGetLastKnown(destArrayID, idxlist[acount]);
      int p = a->lastKnown(idxlist[acount]);
      if(p == CkMyPe())
            continue;
        
        if(p == -1) CkAbort("Invalid Section\n");        
        
        //Collect remote processors
        for(count = 0; count < npes; count ++)
            if(pelist[count] == p)
                break;
        
        if(count == npes) {
            pelist[npes ++] = p;
        }
    }
    
    if(npes == 0) {
        delete [] pelist;
        pelist = NULL;
    }
}


void ComlibSectionInfo::getLocalIndices(int nindices,
                                        CkArrayIndexMax *idxlist,
					CkArrayID &destArrayID,
                                        CkVec<CkArrayIndexMax> &idx_vec){    
	ComlibPrintf("ComlibSectionInfo::getLocalIndices()\n");
	
	int count = 0, acount = 0;
    idx_vec.resize(0);
    
    CkArray *a = (CkArray *)_localBranch(destArrayID);
    for(acount = 0; acount < nindices; acount++){
        //int p = ComlibGetLastKnown(destArrayID, idxlist[acount]);
        int p = a->lastKnown(idxlist[acount]);
        if(p == CkMyPe()) 
            idx_vec.insertAtEnd(idxlist[acount]);
    }
}



void ComlibSectionInfo::getNodeLocalIndices(int nindices,
                                        CkArrayIndexMax *idxlist,
					CkArrayID &destArrayID,
                                        CkVec<CkArrayIndexMax> &idx_vec){    
    int count = 0, acount = 0;
    idx_vec.resize(0);
    
    CkArray *a = (CkArray *)_localBranch(destArrayID);
    for(acount = 0; acount < nindices; acount++){
        //int p = ComlibGetLastKnown(destArrayID, idxlist[acount]);
        int p = a->lastKnown(idxlist[acount]);
        if(p == CkMyPe()) 
            idx_vec.insertAtEnd(idxlist[acount]);
    }
}


