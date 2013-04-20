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
    deliver a message to a set of indices using the array manager. Indices can be local or remote. 
    
    An optimization for [nokeep] methods is applied: the message is not copied for each invocation.
   
    @return the number of destination objects which were not local (information
    retrieved from the array/location manager)
*/
int CharmStrategy::deliverToIndices(void *msg, int numDestIdxs, const CkArrayIndex* indices ){
  int count = 0;
  
  envelope *env = UsrToEnv(msg);
  int ep = env->getsetArrayEp();
  CkUnpackMessage(&env);

  CkArrayID destination_aid = env->getArrayMgr();
  CkArray *a=(CkArray *)_localBranch(destination_aid);

  env->setPacked(0);
  env->getsetArrayHops()=1;
  env->setUsed(0);

  //  CkPrintf("Delivering to %d objects\n", numDestIdxs);

  if(numDestIdxs > 0){
        
    // SEND to all destination objects except the last one
    for(int i=0; i<numDestIdxs-1;i++){
      env->getsetArrayIndex() = indices[i];
      
      //      CkPrintf("[%d] in deliverToIndices env->event=%d pe=%d\n", CkMyPe(), (int)env->getEvent(), (int)env->getSrcPe());

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
  else
    CkFreeMsg(msg);

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
PUPable_def(CharmMessageHolder)

ComlibNodeGroupInfo::ComlibNodeGroupInfo() {
    isNodeGroup = 0;
    ngid.setZero();
}

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
}

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


ComlibArrayInfo::ComlibArrayInfo() {
	
    src_aid.setZero();
    isAllSrc = 0;
    totalSrc = 0;
    isSrcArray = 0;

    dest_aid.setZero();
    isAllDest = 0;
    totalDest = 0;
    isDestArray = 0;
}


void ComlibArrayInfo::setSourceArray(CkArrayID aid, CkArrayIndex *e, int nind){
    src_aid = aid;
    isSrcArray = 1;

    src_elements.removeAll();
    for (int i=0; i<nind; ++i){
      CkAssert(e[i].nInts == 1);
      src_elements.push_back(e[i]);
    }
    
    if (nind == 0) 
    	isAllSrc = 1;
    else 
    	isAllSrc = 0;

    totalSrc = nind;

    CkAssert(src_elements.size() == totalSrc);    

}


void ComlibArrayInfo::setDestinationArray(CkArrayID aid, CkArrayIndex *e, int nind){
  ComlibPrintf("[%d] ComlibArrayInfo::setDestinationArray  dest_elements\n", CkMyPe());
    dest_aid = aid;
    isDestArray = 1;
   
    dest_elements.removeAll();
    for (int i=0; i<nind; ++i){
      CkAssert(e[i].nInts > 0);
      dest_elements.push_back(e[i]);
    }

    if (nind == 0) 
    	isAllDest = 1;
    else 
    	isAllDest = 0;
    
    totalDest = nind;
    CkAssert(dest_elements.size() == totalDest);    

}


/// @TODO fix the pup!
//Each strategy must define his own Pup interface.
void ComlibArrayInfo::pup(PUP::er &p){ 
    p | src_aid;
    p | isSrcArray;
    p | isAllSrc;
    p | totalSrc;
    p | src_elements; 
    p | new_src_elements;

    p | dest_aid;
    p | isDestArray;
    p | isAllDest;
    p | totalDest;
    p | dest_elements; 
    p | new_dest_elements;

    if (p.isPacking() || p.isUnpacking()) {
      // calling purge both during packing (at the end) and during unpacking
      // allows this code to be executed both on processor 0 (where the object
      // is created) and on every other processor where it arrives through PUP.
      purge();
    }

    
}


void ComlibArrayInfo::printDestElementList() {
  char buf[100000];
  buf[0] = '\0';
  for(int i=0;i<dest_elements.size();i++){
    sprintf(buf+strlen(buf), " %d", dest_elements[i].data()[0]);
  }
  CkPrintf("[%d] dest_elements = %s\n", CkMyPe(), buf);
}


void ComlibArrayInfo::newElement(CkArrayID &id, const CkArrayIndex &idx) {
  CkAbort("New Comlib implementation does not allow dynamic element insertion yet\n");
  //  CkPrintf("ComlibArrayInfo::newElement dest_elements\n");
  //  if (isAllSrc && id==src_aid) src_elements.push_back(idx);
  //  if (isAllDest && id==dest_aid) dest_elements.push_back(idx);
}

void ComlibArrayInfo::purge() {
  //	ComlibPrintf("[%d] ComlibArrayInfo::purge srcArray=%d (%d), destArray=%d (%d)\n",CkMyPe(),isSrcArray,isAllSrc,isDestArray,isAllDest);

  if (isSrcArray) {
    CkArray *a = (CkArray *)_localBranch(src_aid);

    // delete all the source elements for which we are not homePe
    for (int i=src_elements.size()-1; i>=0; --i) {
      if (a->homePe(src_elements[i]) != CkMyPe()) { 			
	ComlibPrintf("[%d] ComlibArrayInfo::purge removing home=%d src element %d  i=%d\n", CkMyPe(),a->homePe(src_elements[i]), src_elements[i].data()[0], i);
	src_elements.remove(i); 
      }
    }
  }

  if (isDestArray) {
    CkArray *a = (CkArray *)_localBranch(dest_aid);
	
    // delete all the destination elements for which we are not homePe
    for (int i=dest_elements.size()-1; i>=0; --i) {
      if (a->homePe(dest_elements[i]) != CkMyPe()) {
	ComlibPrintf("[%d] ComlibArrayInfo::purge removing home=%d dest element %d  i=%d\n", CkMyPe(), a->homePe(dest_elements[i]), dest_elements[i].data()[0], i);
	dest_elements.remove(i); 
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



/**  Broadcast the message to all local elements (as listed in dest_elements) */
void ComlibArrayInfo::localBroadcast(envelope *env) {
  int count = localMulticast(&dest_elements, env);
  if(com_debug){
    CkPrintf("[%d] ComlibArrayInfo::localBroadcast to %d elements (%d non local)\n",CmiMyPe(),dest_elements.size(),count);
    printDestElementList();
  }

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
int ComlibArrayInfo::localMulticast(CkVec<CkArrayIndex>*vec,
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

    CkArrayID destination_aid = env->getArrayMgr();
    env->setPacked(0);
    env->getsetArrayHops()=1;
    env->setUsed(0);

    CkArrayIndex idx;

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
    
    CkArray *a=(CkArray *)_localBranch(env->getArrayMgr());
    a->deliver((CkArrayMessage *)EnvToUsr(env), CkDeliver_queue);
}


/*@}*/
