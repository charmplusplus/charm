/**
   @addtogroup ComlibCharmStrategy
   @{
   @file

   MulticastStrategy and its
   derivatives, multicast messages to a section of array elements
   created on the fly. The section is invoked by calling a
   section proxy. These strategies can also multicast to a subset
   of processors for groups.
   
   These strategies are non-bracketed. When the first request is
   made a route is dynamically built on the section. The route
   information is stored in

 - Sameer Kumar
 - Heavily revised by Filippo Gioachin 2/2006

*/


#include "MulticastStrategy.h"

CkpvExtern(CkGroupID, cmgrID);


MulticastStrategy::MulticastStrategy()
  : Strategy(), CharmStrategy() {

  ComlibPrintf("MulticastStrategy constructor\n");
  //ainfo.setDestinationArray(aid);
  setType(ARRAY_STRATEGY);
}

//Destroy all old built routes
MulticastStrategy::~MulticastStrategy() {
    
  ComlibPrintf("MulticastStrategy destructor\n");

  if(getLearner() != NULL)
    delete getLearner();
        
  CkHashtableIterator *ht_iterator = sec_ht.iterator();
  ht_iterator->seekStart();
  while(ht_iterator->hasNext()){
    void **data;
    data = (void **)ht_iterator->next();        
    ComlibSectionHashObject *obj = (ComlibSectionHashObject *) (* data);
    if(obj != NULL)
      delete obj;
  }
}

#if 0
void rewritePEs(CharmMessageHolder *cmsg){
  ComlibPrintf("[%d] rewritePEs insertMessage \n",CkMyPe());
    
  CkAssert(cmsg->dest_proc == IS_SECTION_MULTICAST);
    
  void *m = cmsg->getCharmMessage();
  envelope *env = UsrToEnv(m);
    
  ComlibMulticastMsg *msg = (ComlibMulticastMsg *)m;
    
}
#endif

void MulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
    
  ComlibPrintf("[%d] Comlib Section Multicast: insertMessage \n",  CkMyPe());   
  // ComlibPrintf("[%d] Comlib Section Multicast: insertMessage \n",  CkMyPe());   
  
  
  ComlibPrintf("[%d] sec_ht.numObjects() =%d\n", CkMyPe(), sec_ht.numObjects());
  
  
  if(cmsg->dest_proc == IS_SECTION_MULTICAST && cmsg->sec_id != NULL) { 
    ComlibPrintf("[%d] Comlib Section Multicast: looking up cur_sec_id\n",CkMyPe());
    	
    CkSectionID *sid = cmsg->sec_id;

    // This is a sanity check if we only use a tiny chare array
    // if(sid->_nElems > 4 || sid->_nElems<0){
    //       CkPrintf("[%d] Warning!!!!!!!!!!! Section ID in message seems to be screwed up. cmg=%p sid=%p sid->_nElems=%d\n", CkMyPe(), cmsg, sid, (int)sid->_nElems);
    //       CkAbort("");
    //     }
    int cur_sec_id = sid->getSectionID();

    if(cur_sec_id > 0) {
      sinfo.processOldSectionMessage(cmsg);
      ComlibPrintf("Array section id was %d, but now is %d\n", cur_sec_id, sid->getSectionID());
      CkAssert(cur_sec_id == sid->getSectionID());

      ComlibPrintf("[%d] Comlib Section Multicast: insertMessage: cookiePE=%d\n",CkMyPe(),sid->_cookie.get_pe());
      ComlibSectionHashKey key(CkMyPe(), cur_sec_id);
      ComlibSectionHashObject *obj = sec_ht.get(key);

      if(obj == NULL) {
	//CkAbort("Cannot Find Section\n");
	/* The object can be NULL for various reasons:
	 * 1) the user reassociated the section proxy with a different
	 *    multicast strategy, in which case the new one has no idea about
	 *    the previous usage of the proxy, but the proxy has the cur_sec_id
	 *    set by the previous strategy
	 * 2) the proxy migrated to another processor, in which case the
	 *    cur_sec_id is non null, but the CkMyPe changed, so the hashed
	 *    object could not be found (Filippo: I'm not sure if the id will
	 *    be reset upon migration, so if this case if possible)
	 */
      }

      /* In the following if, the check (CkMyPe == sid->_cookie.pe) helps identifying situations
       * where the proxy has migrated from one processor to another. In this situation, the
       * destination processor might find an "obj", created by somebody else. This "obj"
       * is accepted only if the current processor is equal to the processor in which the
       * cookie ID was defined. */
      if (obj != NULL && CkMyPe() == sid->_cookie.get_pe() && !obj->isOld) {
	envelope *env = UsrToEnv(cmsg->getCharmMessage());
	localMulticast(env, obj, (CkMcastBaseMsg*)cmsg->getCharmMessage());
	remoteMulticast(env, obj);

	delete cmsg;
	return;
      }
    }


    // reaching here means the message was not sent as old, either because
    // it is the first for this section or the existing section is old.
    ComlibPrintf("[%d] MulticastStrategy, creating a new multicast path\n", CkMyPe());
        
    //New sec id, so send it along with the message
    ComlibMulticastMsg *newmsg = sinfo.getNewMulticastMessage(cmsg, needSorting(), getInstance());


    ComlibSectionHashObject *obj = NULL;
	    
    //    CkAssert(newmsg!=NULL); // Previously the following code was just not called in this case
    
    if(newmsg !=NULL){
      // Add the section to the hashtable, so we can use it in the future
      ComlibPrintf("[%d] calling insertSectionID\n", CkMyPe());
      ComlibSectionHashObject *obj_inserted = insertSectionID(sid, newmsg->nPes, newmsg->indicesCount);
      
      envelope *newenv = UsrToEnv(newmsg);
      CkPackMessage(&newenv);
    
      ComlibSectionHashKey key(CkMyPe(), sid->_cookie.info.sInfo.cInfo.id);        
    
      obj = sec_ht.get(key);
      ComlibPrintf("[%d] looking up key sid->_cookie.sInfo.cInfo.id=%d. Found obj=%p\n", CkMyPe(), (int)sid->_cookie.info.sInfo.cInfo.id, obj);
      CkAssert(obj_inserted == obj);
    

    
      if(obj == NULL){
	CkPrintf("[%d] WARNING: Cannot Find ComlibRectSectionHashObject object in hash table sec_ht!\n", CkMyPe());
	CkAbort("Cannot Find object. sec_ht.get(key)==NULL");
	// If the number of array elements is fewer than the number of PEs, this happens frequently
      } else {
      
	char *msg = cmsg->getCharmMessage();
	localMulticast(UsrToEnv(msg), obj, (CkMcastBaseMsg*)msg);
	CkFreeMsg(msg);
      
	if (newmsg != NULL) { 
	  remoteMulticast(UsrToEnv(newmsg), obj);
	}
      
      }
    }

  }
  else 
    CkAbort("Section multicast cannot be used without a section proxy");

  delete cmsg;       
}

ComlibSectionHashObject * MulticastStrategy::insertSectionID(CkSectionID *sid, int npes, ComlibMulticastIndexCount* pelist) {

  ComlibPrintf("[%d] MulticastStrategy:insertSectionID\n",CkMyPe());
  ComlibPrintf("[%d] MulticastStrategy:insertSectionID  sid->_cookie.sInfo.cInfo.id=%d \n",CkMyPe(),  (int)sid->_cookie.info.sInfo.cInfo.id);

  //	double StartTime = CmiWallTimer();

  ComlibSectionHashKey key(CkMyPe(), sid->_cookie.info.sInfo.cInfo.id);

  ComlibSectionHashObject *obj = NULL;    
  obj = sec_ht.get(key);
    
  if(obj != NULL) {
    ComlibPrintf("MulticastStrategy:insertSectionID: Deleting old object on proc %d for id %d\n",
		 CkMyPe(), sid->_cookie.info.sInfo.cInfo.id);
    delete obj;
  }

  ComlibPrintf("[%d] Creating new ComlibSectionHashObject in insertSectionID\n", CkMyPe());
  obj = new ComlibSectionHashObject();
  CkArrayID aid(sid->_cookie.get_aid());
  sinfo.getLocalIndices(sid->_nElems, sid->_elems, aid, obj->indices);
    
  createObjectOnSrcPe(obj, npes, pelist);
  sec_ht.put(key) = obj;
  ComlibPrintf("[%d] Inserting object %p into sec_ht\n", CkMyPe(), obj);
  ComlibPrintf("[%d] sec_ht.numObjects() =%d\n", CkMyPe(), sec_ht.numObjects());

  return obj;

  //    traceUserBracketEvent( 2204, StartTime, CmiWallTimer()); 
}


extern void CmiReference(void *);

//Send the multicast message the local array elements. The message is 
//copied and sent if elements exist. 
void MulticastStrategy::localMulticast(envelope *env, 
				       ComlibSectionHashObject *obj,
				       CkMcastBaseMsg *base) {
	
  //	double StartTime = CmiWallTimer();
	
  int nIndices = obj->indices.size();

  if(obj->msg != NULL) {
    CmiFree(obj->msg);
    obj->msg = NULL;
  }

  ComlibPrintf("[%d] localMulticast nIndices=%d\n", CkMyPe(), nIndices);
	
  if(nIndices > 0) {
    void *msg = EnvToUsr(env);
    void *msg1 = msg;

    msg1 = CkCopyMsg(&msg);

    CmiReference(UsrToEnv(msg1));
    obj->msg = (void *)UsrToEnv(msg1);

    int reply = ComlibArrayInfo::localMulticast(&(obj->indices), UsrToEnv(msg1));
    if (reply > 0) {
      // some of the objects were not local, get the update!
      CkMcastBaseMsg *errorMsg = sinfo.getNewDeliveryErrorMsg(base);
      envelope *errorEnv = UsrToEnv(errorMsg);
      CmiSetHandler(errorEnv, CkpvAccess(comlib_handler));
      ((CmiMsgHeaderExt *) errorEnv)->stratid = getInstance();
      CmiSyncSendAndFree(env->getSrcPe(), errorEnv->getTotalsize(), (char*)errorEnv);
    }
  }
	
  //	traceUserBracketEvent( 2200, StartTime, CmiWallTimer()); 
	
}


//Calls default multicast scheme to send the messages. It could 
//also call a converse lower level strategy to do the muiticast.
//For example pipelined multicast
void MulticastStrategy::remoteMulticast(envelope *env, 
					ComlibSectionHashObject *obj) {
    
  //	double StartTime = CmiWallTimer();
	
  int npes = obj->npes;
  int *pelist = obj->pelist;
    
  if(npes == 0) {
    CmiFree(env);
    return;    
  }
    
  //CmiSetHandler(env, handlerId);
  CmiSetHandler(env, CkpvAccess(comlib_handler));

  ((CmiMsgHeaderExt *) env)->stratid = getInstance();

  //Collect Multicast Statistics
  RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
    
  CkPackMessage(&env);
  //Sending a remote multicast
    
  ComlibPrintf("[%d] remoteMulticast Sending to %d PEs: \n", CkMyPe(), npes);
  for(int i=0;i<npes;i++){
    ComlibPrintf("[%d]    %d\n", CkMyPe(), pelist[i]);
  }
    
  CmiSyncListSendAndFree(npes, pelist, env->getTotalsize(), (char*)env);
  //CmiSyncBroadcastAndFree(env->getTotalsize(), (char*)env);

  //	traceUserBracketEvent( 2201, StartTime, CmiWallTimer()); 

}

void MulticastStrategy::pup(PUP::er &p){
  Strategy::pup(p);
  CharmStrategy::pup(p);
}


void MulticastStrategy::handleMessage(void *msg){

	
  //	double StartTime = CmiWallTimer();
	
  envelope *env = (envelope *)msg;
  RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe());

  //Section multicast base message
  CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);
  if (cbmsg->magic != _SECTION_MAGIC) CkAbort("MulticastStrategy received bad message! Did you forget to inherit from CkMcastBaseMsg?\n");
    
  int status = cbmsg->_cookie.info.sInfo.cInfo.status;
  ComlibPrintf("[%d] In handleMulticastMessage %d\n", CkMyPe(), status);
    
  if(status == COMLIB_MULTICAST_NEW_SECTION)
    handleNewMulticastMessage(env);
  else if (status == COMLIB_MULTICAST_SECTION_ERROR) {
    // some objects were not on the correct processor, mark the section as
    // old. next time we try to use it, a new one will be generated with the
    // updated inforamtion in the location manager (since the wrong delivery
    // updated it indirectly.
    ComlibSectionHashKey key(cbmsg->_cookie.get_pe(), 
			     cbmsg->_cookie.info.sInfo.cInfo.id);    
        
    ComlibSectionHashObject *obj;
    obj = sec_ht.get(key);

    if(obj == NULL)
      CkAbort("Destination indices is NULL\n");

    // mark the section as old
    obj->isOld = 1;
  } else if (status == COMLIB_MULTICAST_OLD_SECTION) {
    //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
    ComlibSectionHashKey key(cbmsg->_cookie.get_pe(), 
			     cbmsg->_cookie.info.sInfo.cInfo.id);    
        
    ComlibSectionHashObject *obj;
    obj = sec_ht.get(key);
        
    if(obj == NULL)
      CkAbort("Destination indices is NULL\n");
        
    localMulticast(env, obj, cbmsg);
    remoteMulticast(env, obj);
  } else {
    CkAbort("Multicast message status is zero\n");
  }

  //	traceUserBracketEvent( 2202, StartTime, CmiWallTimer()); 

}


void MulticastStrategy::handleNewMulticastMessage(envelope *env) {
    
  //	double StartTime = CmiWallTimer();

  ComlibPrintf("%d : In handleNewMulticastMessage\n", CkMyPe());
  ComlibPrintf("%d : In handleNewMulticastMessage\n", CkMyPe());

  CkUnpackMessage(&env);

  int localElems;
  envelope *newenv;
  CkArrayIndex *local_idx_list;    
    
  // Extract the list of elements to be delivered locally
  sinfo.unpack(env, localElems, local_idx_list, newenv);

  ComlibMulticastMsg *cbmsg = (ComlibMulticastMsg *)EnvToUsr(env);
  ComlibSectionHashKey key(cbmsg->_cookie.get_pe(), 
			   cbmsg->_cookie.info.sInfo.cInfo.id);
    
  ComlibSectionHashObject *old_obj = NULL;
    
  old_obj = sec_ht.get(key);
  if(old_obj != NULL) {
    delete old_obj;
  }

  /*
    CkArrayIndex *idx_list_array = new CkArrayIndex[idx_list.size()];
    for(int count = 0; count < idx_list.size(); count++)
    idx_list_array[count] = idx_list[count];
  */

  ComlibPrintf("[%d] Creating new ComlibSectionHashObject in handleNewMulticastMessage\n", CkMyPe());
  ComlibSectionHashObject *new_obj = new ComlibSectionHashObject();
  new_obj->indices.resize(0);
  for (int i=0; i<localElems; ++i) new_obj->indices.insertAtEnd(local_idx_list[i]);
    
  createObjectOnIntermediatePe(new_obj, cbmsg->nPes, cbmsg->indicesCount, cbmsg->_cookie.get_pe());

  ComlibPrintf("[%d] Inserting object into sec_ht\n", CkMyPe());
  ComlibPrintf("[%d] sec_ht.numObjects() =%d\n", CkMyPe(), sec_ht.numObjects());

  sec_ht.put(key) = new_obj;

    
  /* local multicast must come before remote multicast because the second can delete
   * the passed env parameter, and cbmsg is part of env!
   */
  //	traceUserBracketEvent( 2203, StartTime, CmiWallTimer()); 

  localMulticast(newenv, new_obj, cbmsg); //local multicast always copies
  remoteMulticast(env, new_obj);
  CmiFree(newenv);   
    
}

/*@}*/
