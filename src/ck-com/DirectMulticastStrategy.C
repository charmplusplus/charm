#include "DirectMulticastStrategy.h"
#include "AAMLearner.h"

CkpvExtern(CkGroupID, cmgrID);

void *DMHandler(void *msg){
    ComlibPrintf("[%d]:In CallbackHandler\n", CkMyPe());
    DirectMulticastStrategy *nm_mgr;    
    
    CkMcastBaseMsg *bmsg = (CkMcastBaseMsg *)EnvToUsr((envelope *)msg);
    int instid = bmsg->_cookie.sInfo.cInfo.instId;
    
    nm_mgr = (DirectMulticastStrategy *) 
        CProxy_ComlibManager(CkpvAccess(cmgrID)).
        ckLocalBranch()->getStrategy(instid);

    envelope *env = (envelope *) msg;
    RECORD_RECV_STATS(instid, env->getTotalsize(), env->getSrcPe());
    nm_mgr->handleMulticastMessage(msg);
    return NULL;
}

//Group Constructor
DirectMulticastStrategy::DirectMulticastStrategy(int ndest, int *pelist)
    : CharmStrategy() {
 
    setType(GROUP_STRATEGY);
    
    ndestpes = ndest;
    destpelist = pelist;

    commonInit();
}

DirectMulticastStrategy::DirectMulticastStrategy(CkArrayID aid)
    :  CharmStrategy() {

    //ainfo.setSourceArray(aid);
    ainfo.setDestinationArray(aid);
    setType(ARRAY_STRATEGY);
    ndestpes = 0;
    destpelist = 0;
    commonInit();
}

DirectMulticastStrategy::DirectMulticastStrategy(CkArrayID said, CkArrayID daid)
    :  CharmStrategy() {

    ainfo.setSourceArray(said);
    ainfo.setDestinationArray(daid);
    setType(ARRAY_STRATEGY);
    ndestpes = 0;
    destpelist = 0;
    commonInit();
}

void DirectMulticastStrategy::commonInit(){

    if(ndestpes == 0) {
        ndestpes = CkNumPes();
        destpelist = new int[CkNumPes()];
        for(int count = 0; count < CkNumPes(); count ++)
            destpelist[count] = count;        
    }
}

DirectMulticastStrategy::~DirectMulticastStrategy() {
    if(ndestpes > 0)
        delete [] destpelist;

    if(getLearner() != NULL)
        delete getLearner();
        
    CkHashtableIterator *ht_iterator = sec_ht.iterator();
    ht_iterator->seekStart();
    while(ht_iterator->hasNext()){
        void **data;
        data = (void **)ht_iterator->next();        
        CkVec<CkArrayIndexMax> *a_vec = (CkVec<CkArrayIndexMax> *) (* data);
        if(a_vec != NULL)
            delete a_vec;
    }
}

void DirectMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
    if(messageBuf == NULL) {
	CkPrintf("ERROR MESSAGE BUF IS NULL\n");
	return;
    }

    ComlibPrintf("[%d] Comlib Direct Multicast: insertMessage \n", 
                 CkMyPe());   
   
    if(cmsg->dest_proc == IS_SECTION_MULTICAST && cmsg->sec_id != NULL) { 
        int cur_sec_id = ComlibSectionInfo::getSectionID(*cmsg->sec_id);

        if(cur_sec_id > 0) {        
            sinfo.processOldSectionMessage(cmsg);
        }
        else {
            CkSectionID *sid = cmsg->sec_id;

            //New sec id, so send it along with the message
            void *newmsg = sinfo.getNewMulticastMessage(cmsg);
            CkFreeMsg(cmsg->getCharmMessage());
            delete cmsg;
            
            sinfo.initSectionID(sid);

            cmsg = new CharmMessageHolder((char *)newmsg, 
                                          IS_SECTION_MULTICAST); 
            cmsg->sec_id = sid;
        }        
    }
   
    messageBuf->enq(cmsg);
    if(!isBracketed())
        doneInserting();
}

void DirectMulticastStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    
    if(messageBuf->length() == 0) {
        return;
    }

    while(!messageBuf->isEmpty()) {
	CharmMessageHolder *cmsg = messageBuf->deq();
        char *msg = cmsg->getCharmMessage();
        	
        if(cmsg->dest_proc == IS_SECTION_MULTICAST || 
           cmsg->dest_proc == IS_BROADCAST) {      

            if(getType() == ARRAY_STRATEGY)
                CmiSetHandler(UsrToEnv(msg), handlerId);
            
            int *cur_map = destpelist;
            int cur_npes = ndestpes;
            if(cmsg->sec_id != NULL && cmsg->sec_id->pelist != NULL) {
                cur_map = cmsg->sec_id->pelist;
                cur_npes = cmsg->sec_id->npes;
            }
            
            //Collect Multicast Statistics
            RECORD_SENDM_STATS(getInstance(), 
                               ((envelope *)cmsg->getMessage())->getTotalsize(), 
                               cur_map, cur_npes);


            ComlibPrintf("[%d] Calling Direct Multicast %d %d %d\n", CkMyPe(),
                         UsrToEnv(msg)->getTotalsize(), cur_npes, 
                         cmsg->dest_proc);

            /*
              for(int i=0; i < cur_npes; i++)
              CkPrintf("[%d] Sending to %d %d\n", CkMyPe(), 
              cur_map[i], cur_npes);
            */

            CmiSyncListSendAndFree(cur_npes, cur_map, 
                                   UsrToEnv(msg)->getTotalsize(), 
                                   (char*)(UsrToEnv(msg)));            
        }
        else {
            //CkPrintf("SHOULD NOT BE HERE\n");
            CmiSyncSendAndFree(cmsg->dest_proc, 
                               UsrToEnv(msg)->getTotalsize(), 
                               (char *)UsrToEnv(msg));
        }        
        
	delete cmsg; 
    }
}

void DirectMulticastStrategy::pup(PUP::er &p){

    CharmStrategy::pup(p);

    p | ndestpes;
    if(p.isUnpacking() && ndestpes > 0)
        destpelist = new int[ndestpes];
    
    p(destpelist, ndestpes);        
    
    if(p.isUnpacking()) {
        CkArrayID src;
        int nidx;
        CkArrayIndexMax *idx_list;     
        ainfo.getSourceArray(src, idx_list, nidx);
        
        if(!src.isZero()) {
            AAMLearner *l = new AAMLearner();
            setLearner(l);
        }
    }
}

void DirectMulticastStrategy::beginProcessing(int numElements){
    
    messageBuf = new CkQ<CharmMessageHolder *>;    
    handlerId = CkRegisterHandler((CmiHandler)DMHandler);    
    
    CkArrayID dest;
    int nidx;
    CkArrayIndexMax *idx_list;

    ainfo.getDestinationArray(dest, idx_list, nidx);
    sinfo = ComlibSectionInfo(dest, myInstanceID);
}

void DirectMulticastStrategy::handleMulticastMessage(void *msg){
    register envelope *env = (envelope *)msg;
    
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);

    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In local multicast %d\n", CkMyPe(), status);
    
    CkVec<CkArrayIndexMax> *dest_indices; 
    if(status == COMLIB_MULTICAST_ALL) {        
        ainfo.localBroadcast(env);
    }   
    else if(status == COMLIB_MULTICAST_NEW_SECTION){        
        CkUnpackMessage(&env);
        envelope *newenv;
        sinfo.unpack(env, dest_indices, newenv);
        ComlibArrayInfo::localMulticast(dest_indices, newenv);

        CkVec<CkArrayIndexMax> *old_dest_indices;
        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);

        old_dest_indices = (CkVec<CkArrayIndexMax> *)sec_ht.get(key);
        if(old_dest_indices != NULL)
            delete old_dest_indices;
        
        sec_ht.put(key) = dest_indices;
        CmiFree(env);                
    }
    else {
        //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);    
        dest_indices = (CkVec<CkArrayIndexMax> *)sec_ht.get(key);
        
        if(dest_indices == NULL)
            CkAbort("Destination indices is NULL\n");
        
        ComlibArrayInfo::localMulticast(dest_indices, env);
    }
}
