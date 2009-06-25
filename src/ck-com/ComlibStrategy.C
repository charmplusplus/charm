/**
   @addtogroup CharmComlib
   @{
   @file
   Implementations of ComlibStrategy.h
*/

#include "ComlibStrategy.h"
#include "register.h"


void CharmStrategy::pup(PUP::er &p) {
  //Strategy::pup(p);
    p | nginfo;
    p | ginfo;
    p | ainfo;
    //p | forwardOnMigration;
    p | mflag;
    p | onFinish;
}





/** 
    Deliver a message to a set of indices using the array manager. Indices can be local or remote. 
    
    An optimization for [nokeep] methods is applied: the message is not copied for each invocation.
   
    @return the number of destination objects which were not local (information
    retrieved from the array/location manager)
*/
void CharmStrategy::deliverToIndices(void *msg, int numDestIdxs, const CkArrayIndexMax* indices ){
  int count = 0;
  
  envelope *env = UsrToEnv(msg);
  int ep = env->getsetArrayEp();
  CkUnpackMessage(&env);

  CkArrayID destination_aid = env->getsetArrayMgr();
  CkArray *a=(CkArray *)_localBranch(destination_aid);

  env->setPacked(0);
  env->getsetArrayHops()=1;
  env->setUsed(0);

  //  CkPrintf("Delivering to %d objects\n", numDestIdxs);

  if(numDestIdxs > 0){
        
    // SEND to all destination objects except the last one
    for(int i=0; i<numDestIdxs-1;i++){
      env->getsetArrayIndex() = indices[i];
      
      if(_entryTable[ep]->noKeep)
	// don't make a copy for [nokeep] entry methods
	count += a->deliver((CkArrayMessage *)msg, CkDeliver_inline, CK_MSG_KEEP);
      else {
	void *newmsg = CkCopyMsg(&msg);
	count += a->deliver((CkArrayMessage *)newmsg, CkDeliver_queue);
      }
    }
    
    // SEND to the final destination object
    env->getsetArrayIndex() = indices[numDestIdxs-1];
    
    if(_entryTable[ep]->noKeep){
      count += a->deliver((CkArrayMessage *)msg, CkDeliver_inline, CK_MSG_KEEP);
      CmiFree(env); // runtime frees the [nokeep] messages
    }
    else {
      count += a->deliver((CkArrayMessage *)msg, CkDeliver_queue);
    }
    
  }

  return count;
}








void CharmMessageHolder::pup(PUP::er &p) {

    //    CkPrintf("In CharmMessageHolder::pup \n"); 

    MessageHolder::pup(p);

    //Sec ID depends on the message
    //Currently this pup is only being used for remote messages
    sec_id = NULL;
}

//PUPable_def(CharmStrategy);
PUPable_def(CharmMessageHolder);

ComlibNodeGroupInfo::ComlibNodeGroupInfo() {
    isNodeGroup = 0;
    ngid.setZero();
};

void ComlibNodeGroupInfo::pup(PUP::er &p) {
    p | isNodeGroup;
    p | ngid;
}

ComlibGroupInfo::ComlibGroupInfo() {
    
    isSrcGroup = 0;
    isDestGroup = 0;
    nsrcpes = 0;
    ndestpes = 0;
    srcpelist = NULL;
    destpelist = NULL;
    sgid.setZero();
    dgid.setZero();
};

ComlibGroupInfo::~ComlibGroupInfo() {
    if(nsrcpes > 0 && srcpelist != NULL)
        delete [] srcpelist;

    if(ndestpes > 0 && destpelist != NULL)
        delete [] destpelist;
}

void ComlibGroupInfo::pup(PUP::er &p){

    p | sgid;
    p | dgid;
    p | nsrcpes;
    p | ndestpes;

    p | isSrcGroup;
    p | isDestGroup;

    if(p.isUnpacking()) {
        if(nsrcpes > 0) 
            srcpelist = new int[nsrcpes];

        if(ndestpes > 0) 
            destpelist = new int[ndestpes];
    }

    if(nsrcpes > 0) 
        p(srcpelist, nsrcpes);

    if(ndestpes > 0) 
        p(destpelist, ndestpes);
}

void ComlibGroupInfo::setSourceGroup(CkGroupID gid, int *pelist, 
                                         int npes) {
    this->sgid = gid;
    srcpelist = pelist;
    nsrcpes = npes;
    isSrcGroup = 1;

    if(nsrcpes == 0) {
        nsrcpes = CkNumPes();
        srcpelist = new int[nsrcpes];
        for(int count =0; count < nsrcpes; count ++)
            srcpelist[count] = count;
    }
}

void ComlibGroupInfo::getSourceGroup(CkGroupID &gid, int *&pelist, 
                                         int &npes){
    gid = this->sgid;
    npes = nsrcpes;

    pelist = new int [nsrcpes];
    memcpy(pelist, srcpelist, npes * sizeof(int));
}

void ComlibGroupInfo::getSourceGroup(CkGroupID &gid){
    gid = this->sgid;
}

void ComlibGroupInfo::setDestinationGroup(CkGroupID gid, int *pelist, 
                                         int npes) {
    this->dgid = gid;
    destpelist = pelist;
    ndestpes = npes;
    isDestGroup = 1;

    if(ndestpes == 0) {
        ndestpes = CkNumPes();
        destpelist = new int[ndestpes];
        for(int count =0; count < ndestpes; count ++)
            destpelist[count] = count;
    }
}

void ComlibGroupInfo::getDestinationGroup(CkGroupID &gid, int *&pelist, 
                                         int &npes) {
    gid = this->dgid;
    npes = ndestpes;

    pelist = new int [ndestpes];
    memcpy(pelist, destpelist, npes * sizeof(int));
}

void ComlibGroupInfo::getDestinationGroup(CkGroupID &gid) {
    gid = this->dgid;
}

int *ComlibGroupInfo::getCombinedCountList() {
  int *result = new int[CkNumPes()];
  int i;
  for (i=0; i<CkNumPes(); ++i) result[i] = 0;
  if (nsrcpes != 0) {
    for (i=0; i<nsrcpes; ++i) result[srcpelist[i]] |= 1;
  } else {
    for (i=0; i<CkNumPes(); ++i) result[i] |= 1;
  }
  if (ndestpes != 0) {
    for (i=0; i<ndestpes; ++i) result[destpelist[i]] |= 2;
  } else {
    for (i=0; i<CkNumPes(); ++i) result[i] |= 2;
  }
  return result;
}

/*
void ComlibGroupInfo::getCombinedPeList(int *&pelist, int &npes) {
    int count = 0;        
    pelist = 0;
    npes = 0;

    pelist = new int[CkNumPes()];
    if(nsrcpes == 0 || ndestpes == 0) {
        npes = CkNumPes();        
        for(count = 0; count < CkNumPes(); count ++) 
            pelist[count] = count;                         
    }
    else {        
        npes = ndestpes;
        memcpy(pelist, destpelist, npes * sizeof(int));
        
        //Add source processors to the destination processors
        //already obtained
        for(int count = 0; count < nsrcpes; count++) {
            int p = srcpelist[count];

            for(count = 0; count < npes; count ++)
                if(pelist[count] == p)
                    break;

            if(count == npes)
                pelist[npes ++] = p;
        }                        
    }
}
*/

ComlibArrayInfo::ComlibArrayInfo() {
	
    src_aid.setZero();
    //nSrcIndices = -1;
    //src_elements = NULL;
    isAllSrc = 0;
    totalSrc = 0;
    isSrcArray = 0;

    dest_aid.setZero();
    //nDestIndices = -1;
    //dest_elements = NULL;
    isAllDest = 0;
    totalDest = 0;
    isDestArray = 0;
};

/*
ComlibArrayInfo::~ComlibArrayInfo() {
    //CkPrintf("in comlibarrayinfo destructor\n");

    if(nSrcIndices > 0)
        delete [] src_elements;

    if(nDestIndices > 0)
        delete [] dest_elements;
}
*/

/// Set the  source array used for this strategy. 
/// The list of array indices should be the whole portion of the array involved in the strategy.
/// The non-local array elements will be cleaned up inside purge() at migration of the strategy
void ComlibArrayInfo::setSourceArray(CkArrayID aid, 
                                         CkArrayIndexMax *e, int nind){
    src_aid = aid;
    isSrcArray = 1;
    /*
    nSrcIndices = nind;
    if(nind > 0) {
        src_elements = new CkArrayIndexMax[nind];
        memcpy(src_elements, e, sizeof(CkArrayIndexMax) * nind);
    }
    */
    src_elements.removeAll();
    
    for (int i=0; i<nind; ++i) src_elements.push_back(e[i]);
    
    if (src_elements.size() == 0) 
    	isAllSrc = 1;
    else 
    	isAllSrc = 0;
    
    totalSrc = src_elements.size();
    
}


void ComlibArrayInfo::getSourceArray(CkArrayID &aid, 
                                         CkArrayIndexMax *&e, int &nind){
    aid = src_aid;
    nind = src_elements.length();//nSrcIndices;
    e = src_elements.getVec();//src_elements;
}


void ComlibArrayInfo::setDestinationArray(CkArrayID aid, 
                                          CkArrayIndexMax *e, int nind){
    dest_aid = aid;
    isDestArray = 1;
    /*
    nDestIndices = nind;
    if(nind > 0) {
        dest_elements = new CkArrayIndexMax[nind];
        memcpy(dest_elements, e, sizeof(CkArrayIndexMax) * nind);
    }
    */
    dest_elements.removeAll();
    for (int i=0; i<nind; ++i) dest_elements.push_back(e[i]);

    if (dest_elements.size() == 0) 
    	isAllDest = 1;
    else 
    	isAllDest = 0;
    
    totalDest = dest_elements.size();
    
}


void ComlibArrayInfo::getDestinationArray(CkArrayID &aid, 
                                          CkArrayIndexMax *&e, int &nind){
    aid = dest_aid;
    nind = dest_elements.length();
    e = dest_elements.getVec();
}

/// @TODO fix the pup!
//Each strategy must define his own Pup interface.
void ComlibArrayInfo::pup(PUP::er &p){ 
    p | src_aid;
    //p | nSrcIndices;
    p | isSrcArray;
    p | isAllSrc;
    p | totalSrc;
    p | src_elements;
    
    p | dest_aid;
    //p | nDestIndices;
    p | isDestArray;
    p | isAllDest;
    p | totalDest;
    p | dest_elements;

    if (p.isPacking() || p.isUnpacking()) {
      // calling purge both during packing (at the end) and during unpacking
      // allows this code to be executed both on processor 0 (where the object
      // is created) and on every other processor where it arrives through PUP.
      purge();
    }

    /*    
    if(p.isUnpacking() && nSrcIndices > 0) 
        src_elements = new CkArrayIndexMax[nSrcIndices];
    
    if(p.isUnpacking() && nDestIndices > 0) 
        dest_elements = new CkArrayIndexMax[nDestIndices];        
    
    if(nSrcIndices > 0)
        p((char *)src_elements, nSrcIndices * sizeof(CkArrayIndexMax));    
    else
        src_elements = NULL;

    if(nDestIndices > 0)
        p((char *)dest_elements, nDestIndices * sizeof(CkArrayIndexMax));    
    else
        dest_elements = NULL;

    localDestIndexVec.resize(0);
    */
    
}

void ComlibArrayInfo::newElement(CkArrayID &id, const CkArrayIndex &idx) {
  ComlibPrintf("ComlibArrayInfo::newElement\n");
  if (isAllSrc && id==src_aid) src_elements.push_back(idx);
  if (isAllDest && id==dest_aid) dest_elements.push_back(idx);
}

void ComlibArrayInfo::purge() {
	int i;
	CkArray *a;
//	ComlibPrintf("[%d] ComlibArrayInfo::purge srcArray=%d (%d), destArray=%d (%d)\n",CkMyPe(),isSrcArray,isAllSrc,isDestArray,isAllDest);
	if (isSrcArray) {
		a = (CkArray *)_localBranch(src_aid);
		if (isAllSrc) {
			// gather the information of all the elements that are currenly present here
			ComlibElementIterator iter(&src_elements);
			a->getLocMgr()->iterate(iter);

			// register to ComlibArrayListener for this array id
//			ComlibManagerRegisterArrayListener(src_aid, this);
		} else {
			// delete all the elements of which we are not homePe
			for (i=src_elements.size()-1; i>=0; --i) {
				if (a->homePe(src_elements[i]) != CkMyPe()) { 
					
//					ComlibPrintf("[%d] removing home=%d src element %d  i=%d\n", CkMyPe(),a->homePe(src_elements[i]), src_elements[i].data()[0], i);
					src_elements.remove(i); 
				}
			}
		}
	}
	if (isDestArray) {
		a = (CkArray *)_localBranch(dest_aid);
		if (isAllDest) {
			// gather the information of all the elements that are currenly present here
			ComlibElementIterator iter(&dest_elements);
			a->getLocMgr()->iterate(iter);

			// register to ComlibArrayListener for this array id
//			ComlibManagerRegisterArrayListener(dest_aid, this);
		} else {
			// delete all the elements of which we are not homePe
			for (i=dest_elements.size()-1; i>=0; --i) {
				if (a->homePe(dest_elements[i]) != CkMyPe()) {
//					ComlibPrintf("[%d] removing home=%d dest element %d  i=%d\n", CkMyPe(), a->homePe(dest_elements[i]), dest_elements[i].data()[0], i);
					dest_elements.remove(i); 
				}
			}
		}
	}
}

int *ComlibArrayInfo::getCombinedCountList() {
  int *result = new int[CkNumPes()];
  int i;
  for (i=0; i<CkNumPes(); ++i) result[i] = 0;
  CkArray *a = (CkArray *)_localBranch(src_aid);
  if (src_elements.size() != 0) {
    for (i=0; i<src_elements.size(); ++i) result[a->homePe(src_elements[i])] |= 1;
  } else {
    for (i=0; i<CkNumPes(); ++i) result[i] |= 1;
  }
  a = (CkArray *)_localBranch(dest_aid);
  if (dest_elements.size() != 0) {
    for (i=0; i<dest_elements.size(); ++i) result[a->homePe(dest_elements[i])] |= 2;
  } else {
    for (i=0; i<CkNumPes(); ++i) result[i] |= 2;
  }
  return result;
}

/*
//Get the list of destination processors
void ComlibArrayInfo::getDestinationPeList(int *&destpelist, int &ndestpes) {
    
    int count = 0, acount =0;
    
    //Destination has not been set
    if(nDestIndices < 0) {
        destpelist = 0;
        ndestpes = 0;
        return;
    }

    //Create an array of size CkNumPes()
    //Inefficient in space
    ndestpes = CkNumPes();
    destpelist = new int[ndestpes];

    memset(destpelist, 0, ndestpes * sizeof(int));    

    if(nDestIndices == 0){
        for(count =0; count < CkNumPes(); count ++) 
            destpelist[count] = count;             
        return;
    }

    ndestpes = 0;
    CkArray *amgr = CkArrayID::CkLocalBranch(dest_aid);

    //Find the last known processors of the array elements
    for(acount = 0; acount < nDestIndices; acount++) {

      //int p = ComlibGetLastKnown(dest_aid, dest_elements[acount]); 
        int p = amgr->lastKnown(dest_elements[acount]);
        
        for(count = 0; count < ndestpes; count ++)
            if(destpelist[count] == p)
                break;
        
        if(count == ndestpes) {
            destpelist[ndestpes ++] = p; 
        }       
    }                            
}

void ComlibArrayInfo::getSourcePeList(int *&srcpelist, int &nsrcpes) {
    
    int count = 0, acount =0;

    if(nSrcIndices < 0) {
        srcpelist = 0;
        nsrcpes = 0;
        return;
    }

    nsrcpes = CkNumPes();
    srcpelist = new int[nsrcpes];

    memset(srcpelist, 0, nsrcpes * sizeof(int));    

    if(nSrcIndices == 0){
        for(count =0; count < CkNumPes(); count ++) 
            srcpelist[count] = count;             
        return;
    }

    nsrcpes = 0;
    CkArray *amgr = CkArrayID::CkLocalBranch(src_aid);

    for(acount = 0; acount < nSrcIndices; acount++) {
        
      //int p = ComlibGetLastKnown(src_aid, src_elements[acount]); 
        int p = amgr->lastKnown(src_elements[acount]);
        
        for(count = 0; count < nsrcpes; count ++)
            if(srcpelist[count] == p)
                break;
        
        if(count == nsrcpes) {
            srcpelist[nsrcpes ++] = p; 
        }       
    }                            
}

void ComlibArrayInfo::getCombinedPeList(int *&pelist, int &npes) {

    int count = 0;        
    pelist = 0;
    npes = 0;
    
    //Both arrays empty;
    //Sanity check, this should really not happen
    if(nSrcIndices < 0 && nDestIndices < 0) {
        CkAbort("Arrays have not been set\n");
        return;
    }
    
    //One of them is the entire array Hence set the number of
    //processors to all Currently does not work for the case where
    //number of array elements less than number of processors
    //Will fix it later!
    if(nSrcIndices == 0 || nDestIndices == 0) {
        npes = CkNumPes();        
        pelist = new int[npes];
        for(count = 0; count < CkNumPes(); count ++) 
            pelist[count] = count;                         
    }
    else {
        getDestinationPeList(pelist, npes);
        
        //Destination has not been set
        //Strategy does not care about destination
        //Is an error case
        if(npes == 0)
            pelist = new int[CkNumPes()];
        
	CkArray *amgr = CkArrayID::CkLocalBranch(src_aid);

        //Add source processors to the destination processors
        //already obtained
        for(int acount = 0; acount < nSrcIndices; acount++) {
	  //int p = ComlibGetLastKnown(src_aid, src_elements[acount]);
	    int p = amgr->lastKnown(src_elements[acount]);

            for(count = 0; count < npes; count ++)
                if(pelist[count] == p)
                    break;
            if(count == npes)
                pelist[npes ++] = p;
        }                        
    }
}
*/

/// Broadcast the message to all local elements
void ComlibArrayInfo::localBroadcast(envelope *env) {
  int count = localMulticast(&dest_elements, env);
  ComlibPrintf("[%d] ComlibArrayInfo::localBroadcast to %d elements (%d non local)\n",CmiMyPe(),dest_elements.size(),count);

//  char buf[100000];
//  buf[0] = '\0';
//  for(int i=0;i<dest_elements.size();i++){
//	  sprintf(buf+strlen(buf), " %d", dest_elements[i].data()[0]);
//  }
//  ComlibPrintf("dest_elements = %s\n", buf);

}

/**
  This method multicasts the message to all the indices in vec.  It
  also takes care to check if the entry method is readonly or not. If
  readonly (nokeep) the message is not copied.

  It also makes sure that the entry methods are logged in projections
  and that the array manager is notified about array element
  migrations.  Hence this function should be used extensively in the
  communication library strategies

  This method is more general than just ComlibArrayInfo dest_aid since it takes
  the destination array id directly form the message envelope.

  @return the number of destination objects which were not local (information
  retrieved from the array/location manager)

  @todo Replace this method with calls to CharmStrategy::deliverToIndices, possibly making it a function that is not part of any class

*/

#include "register.h"
int ComlibArrayInfo::localMulticast(CkVec<CkArrayIndexMax>*vec,
                                     envelope *env){
  int count = 0;
    //Multicast the messages to all elements in vec
    int nelements = vec->size();
    if(nelements == 0) {
        CmiFree(env);
        return 0;
    }

    void *msg = EnvToUsr(env);
    int ep = env->getsetArrayEp();
    CkUnpackMessage(&env);

    CkArrayID destination_aid = env->getsetArrayMgr();
    env->setPacked(0);
    env->getsetArrayHops()=1;
    env->setUsed(0);

    CkArrayIndexMax idx;

    //ComlibPrintf("sending to %d elements\n",nelements);
    for(int i = 0; i < nelements-1; i ++){
        idx = (*vec)[i];
        //if(com_debug) idx.print();

        env->getsetArrayIndex() = idx;
        //ComlibPrintf("sending to: "); idx.print();
        
        CkArray *a=(CkArray *)_localBranch(destination_aid);
        if(_entryTable[ep]->noKeep)
            count += a->deliver((CkArrayMessage *)msg, CkDeliver_inline, CK_MSG_KEEP);
        else {
            void *newmsg = CkCopyMsg(&msg);
            count += a->deliver((CkArrayMessage *)newmsg, CkDeliver_queue);
        }

    }

    idx = (*vec)[nelements-1];
    //if(com_debug) idx.print();
    env->getsetArrayIndex() = idx;
    //ComlibPrintf("sending to: "); idx.print();
    
    CkArray *a=(CkArray *)_localBranch(destination_aid);
    if(_entryTable[ep]->noKeep) {
        count += a->deliver((CkArrayMessage *)msg, CkDeliver_inline, CK_MSG_KEEP);
        CmiFree(env);
    }
    else
        count += a->deliver((CkArrayMessage *)msg, CkDeliver_queue);

    return count;
}

/** Delivers a message to an array element, making sure that
    projections is notified */
void ComlibArrayInfo::deliver(envelope *env){
    ComlibPrintf("In ComlibArrayInfo::deliver()\n");
		
    env->setUsed(0);
    env->getsetArrayHops()=1;
    CkUnpackMessage(&env);
    
    CkArray *a=(CkArray *)_localBranch(env->getsetArrayMgr());
    a->deliver((CkArrayMessage *)EnvToUsr(env), CkDeliver_queue);
}


/*@}*/
