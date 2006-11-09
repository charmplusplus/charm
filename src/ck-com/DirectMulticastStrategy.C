
/********************************************************
        Section multicast strategy suite. DirectMulticast and its
        derivatives, multicast messages to a section of array elements
        created on the fly. The section is invoked by calling a
        section proxy. These strategies can also multicast to a subset
        of processors for groups.

        These strategies are non-bracketed. When the first request is
        made a route is dynamically built on the section. The route
        information is stored in

 - Sameer Kumar

**********************************************/


#include "DirectMulticastStrategy.h"

CkpvExtern(CkGroupID, cmgrID);

/*
void *DMHandler(void *msg){
    ComlibPrintf("[%d]:In CallbackHandler\n", CkMyPe());
    DirectMulticastStrategy *nm_mgr;    
    
    CkMcastBaseMsg *bmsg = (CkMcastBaseMsg *)EnvToUsr((envelope *)msg);
    int instid = bmsg->_cookie.sInfo.cInfo.instId;
    
    nm_mgr = (DirectMulticastStrategy *) 
        CProxy_ComlibManager(CkpvAccess(cmgrID)).
        ckLocalBranch()->getStrategy(instid);
    
    nm_mgr->handleMulticastMessage(msg);
    return NULL;
}
*/

DirectMulticastStrategy::DirectMulticastStrategy(CkArrayID aid, 
						 int isPersistent)
    :  CharmStrategy() {

    ainfo.setDestinationArray(aid);
    setType(ARRAY_STRATEGY);

    this->isPersistent = isPersistent;
}

//Destroy all old built routes
DirectMulticastStrategy::~DirectMulticastStrategy() {
    
    ComlibPrintf("Calling Distructor\n");

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

void DirectMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
    
    ComlibPrintf("[%d] Comlib Direct Section Multicast: insertMessage \n", 
                 CkMyPe());   

    if(cmsg->dest_proc == IS_SECTION_MULTICAST && cmsg->sec_id != NULL) { 
        CkSectionID *sid = cmsg->sec_id;
        int cur_sec_id = ComlibSectionInfo::getSectionID(*sid);
        
        if(cur_sec_id > 0) {        
	    sinfo.processOldSectionMessage(cmsg);            
            
            ComlibSectionHashKey 
                key(CkMyPe(), sid->_cookie.sInfo.cInfo.id);        
            ComlibSectionHashObject *obj = sec_ht.get(key);

            if(obj == NULL)
                CkAbort("Cannot Find Section\n");

            envelope *env = UsrToEnv(cmsg->getCharmMessage());
            localMulticast(env, obj);
            remoteMulticast(env, obj);
        }
        else {
            //New sec id, so send it along with the message
            void *newmsg = sinfo.getNewMulticastMessage(cmsg, needSorting());
            insertSectionID(sid);

            ComlibSectionHashKey 
                key(CkMyPe(), sid->_cookie.sInfo.cInfo.id);        
            
            ComlibSectionHashObject *obj = sec_ht.get(key);

	    if(obj == NULL)
                CkAbort("Cannot Find Section\n");

            /*
	    CkPrintf("%u: Src = %d dest:", key.hash(), CkMyPe());
	    for (int i=0; i<obj->npes; ++i)
	      CkPrintf(" %d",obj->pelist[i]);
	    CkPrintf(", map:");
	    ComlibMulticastMsg *lll = (ComlibMulticastMsg*)newmsg;
	    envelope *ppp = UsrToEnv(newmsg);
	    CkUnpackMessage(&ppp);
	    int ttt=0;
	    for (int i=0; i<lll->nPes; ++i) {
	      CkPrintf(" %d (",lll->indicesCount[i].pe);
	      for (int j=0; j<lll->indicesCount[i].count; ++j) {
		CkPrintf(" %d",((int*)&(lll->indices[ttt]))[1]);
		ttt++;
	      }
	      CkPrintf(" )");
	    }
	    CkPackMessage(&ppp);
	    CkPrintf("\n");
	    */

            char *msg = cmsg->getCharmMessage();
            localMulticast(UsrToEnv(msg), obj);
            CkFreeMsg(msg);
            
            if (newmsg != NULL) remoteMulticast(UsrToEnv(newmsg), obj);
        }        
    }
    else 
        CkAbort("Section multicast cannot be used without a section proxy");

    delete cmsg;       
}

void DirectMulticastStrategy::insertSectionID(CkSectionID *sid) {
    
    ComlibSectionHashKey 
        key(CkMyPe(), sid->_cookie.sInfo.cInfo.id);

    ComlibSectionHashObject *obj = NULL;    
    obj = sec_ht.get(key);
    
    if(obj != NULL)
        delete obj;
    
    obj = createObjectOnSrcPe(sid->_nElems, sid->_elems);
    sec_ht.put(key) = obj;

}


ComlibSectionHashObject *
DirectMulticastStrategy::createObjectOnSrcPe(int nindices, CkArrayIndexMax *idxlist) {

    ComlibSectionHashObject *obj = new ComlibSectionHashObject();
    
    sinfo.getRemotePelist(nindices, idxlist, obj->npes, obj->pelist);
    sinfo.getLocalIndices(nindices, idxlist, obj->indices);
    
    return obj;
}


ComlibSectionHashObject *
DirectMulticastStrategy::createObjectOnIntermediatePe(int nindices,
						      CkArrayIndexMax *idxlist,
						      int npes,
						      ComlibMulticastIndexCount *counts,
						      int srcpe) {

    ComlibSectionHashObject *obj = new ComlibSectionHashObject();
        
    obj->pelist = 0;
    obj->npes = 0;

    obj->indices.resize(0);
    for (int i=0; i<nindices; ++i) obj->indices.insertAtEnd(idxlist[i]);
    //sinfo.getLocalIndices(nindices, idxlist, obj->indices);

    return obj;
}


void DirectMulticastStrategy::doneInserting(){
    //Do nothing! Its a bracketed strategy
}

extern void CmiReference(void *);

//Send the multicast message the local array elements. The message is 
//copied and sent if elements exist. 
void DirectMulticastStrategy::localMulticast(envelope *env, 
                                             ComlibSectionHashObject *obj) {
    int nIndices = obj->indices.size();
    
    //If the library is set to persistent. 
    //The message is stored in the library. The applications should 
    //use the message as a readonly and it exists till the next one 
    //comes along
    
    if(obj->msg != NULL) {
        CmiFree(obj->msg);
	obj->msg = NULL;
    } 
    
    if(nIndices > 0) {
	void *msg = EnvToUsr(env);
	void *msg1 = msg;
        
        msg1 = CkCopyMsg(&msg);
	
	if(isPersistent) {
	    CmiReference(UsrToEnv(msg1));
	    obj->msg = (void *)UsrToEnv(msg1);
	}
	
        ComlibArrayInfo::localMulticast(&(obj->indices), UsrToEnv(msg1));
    }    
}


//Calls default multicast scheme to send the messages. It could 
//also call a converse lower level strategy to do the muiticast.
//For example pipelined multicast
void DirectMulticastStrategy::remoteMulticast(envelope *env, 
                                              ComlibSectionHashObject *obj) {
    
    int npes = obj->npes;
    int *pelist = obj->pelist;
    
    if(npes == 0) {
        CmiFree(env);
        return;    
    }
    
    //CmiSetHandler(env, handlerId);
    CmiSetHandler(env, CkpvAccess(strategy_handlerid));

    ((CmiMsgHeaderBasic *) env)->stratid = getInstance();

    //Collect Multicast Statistics
    RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), pelist, npes);
    
    CkPackMessage(&env);
    //Sending a remote multicast
    CmiSyncListSendAndFree(npes, pelist, env->getTotalsize(), (char*)env);
    //CmiSyncBroadcastAndFree(env->getTotalsize(), (char*)env);
}

void DirectMulticastStrategy::pup(PUP::er &p){

    CharmStrategy::pup(p);
    p | isPersistent; 
}

void DirectMulticastStrategy::beginProcessing(int numElements){
    
    //handlerId = CkRegisterHandler((CmiHandler)DMHandler);    
    
    CkArrayID dest;
    int nidx;
    CkArrayIndexMax *idx_list;

    ainfo.getDestinationArray(dest, idx_list, nidx);
    sinfo = ComlibSectionInfo(dest, myInstanceID);

    ComlibLearner *learner = new ComlibLearner();
    //setLearner(learner);
}

void DirectMulticastStrategy::handleMessage(void *msg){
    envelope *env = (envelope *)msg;
    RECORD_RECV_STATS(getInstance(), env->getTotalsize(), env->getSrcPe());

    //Section multicast base message
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);
    
    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In handleMulticastMessage %d\n", CkMyPe(), status);
    
    if(status == COMLIB_MULTICAST_NEW_SECTION)
        handleNewMulticastMessage(env);
    else {
        //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);    
        
        ComlibSectionHashObject *obj;
        obj = sec_ht.get(key);
        
        if(obj == NULL)
            CkAbort("Destination indices is NULL\n");
        
        localMulticast(env, obj);
        remoteMulticast(env, obj);
    }
}

#include <string>

void DirectMulticastStrategy::handleNewMulticastMessage(envelope *env) {
    
    ComlibPrintf("%d : In handleNewMulticastMessage\n", CkMyPe());

    CkUnpackMessage(&env);    

    int localElems;
    envelope *newenv;
    CkArrayIndexMax *local_idx_list;    
    
    sinfo.unpack(env, localElems, local_idx_list, newenv);

    ComlibMulticastMsg *cbmsg = (ComlibMulticastMsg *)EnvToUsr(env);
    ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                             cbmsg->_cookie.sInfo.cInfo.id);
    
    ComlibSectionHashObject *old_obj = NULL;
    
    old_obj = sec_ht.get(key);
    if(old_obj != NULL) {
        delete old_obj;
    }

    /*
    CkArrayIndexMax *idx_list_array = new CkArrayIndexMax[idx_list.size()];
    for(int count = 0; count < idx_list.size(); count++)
        idx_list_array[count] = idx_list[count];
    */

    ComlibSectionHashObject *new_obj = createObjectOnIntermediatePe(localElems, local_idx_list, cbmsg->nPes, cbmsg->indicesCount, cbmsg->_cookie.pe);

    /*
    char buf[100];
    std::string tmp = "";
    for (int i=0; i<idx_list.size(); i++) {
      sprintf(buf, " (%d-%d-%d-%d)",((short*)&idx_list_array[i])[2],((short*)&idx_list_array[i])[3],((short*)&idx_list_array[i])[4],((short*)&idx_list_array[i])[5]);
      tmp+=buf;
    }
    ComlibPrintf("[%d] LOCAL MULTICAST: %s\n",key.hash(),tmp.data());
    tmp=""+new_obj->npes;
    for (int i=0; i<new_obj->npes; i++) {
      sprintf(buf, " %d",new_obj->pelist[i]);
      tmp+=buf;
    }
    ComlibPrintf("[%d] REMOTE MULTICAST: %s\n",key.hash(),tmp.data());
    */
    //delete [] idx_list_array;
    
    sec_ht.put(key) = new_obj;

    /*
    CkPrintf("%u: Proc = %d (%d) forward:", key.hash(), CkMyPe(),cbmsg->nPes);
    for (int i=0; i<new_obj->npes; ++i) CkPrintf(" %d",new_obj->pelist[i]);
    CkPrintf(", deliver:");
    for (int i=0; i<new_obj->indices.size(); ++i) CkPrintf(" %d",((int*)&new_obj->indices[i])[1]);
    CkPrintf("\n");
    */

    remoteMulticast(env, new_obj);
    localMulticast(newenv, new_obj); //local multicast always copies
    CmiFree(newenv);                
}
