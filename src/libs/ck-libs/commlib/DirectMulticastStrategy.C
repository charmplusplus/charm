#include "DirectMulticastStrategy.h"

//ComlibSectionHashKey CODE
int ComlibSectionHashKey::staticCompare(const void *k1,const void *k2,size_t ){
    return ((const ComlibSectionHashKey *)k1)->
                compare(*(const ComlibSectionHashKey *)k2);
}

CkHashCode ComlibSectionHashKey::staticHash(const void *v,size_t){
    return ((const ComlibSectionHashKey *)v)->hash();
}

CpvExtern(CkGroupID, cmgrID);

void *DMHandler(void *msg){
    ComlibPrintf("[%d]:In CallbackHandler\n", CkMyPe());
    DirectMulticastStrategy *nm_mgr;    
    
    CkMcastBaseMsg *bmsg = (CkMcastBaseMsg *)EnvToUsr((envelope *)msg);
    int instid = bmsg->_cookie.sInfo.cInfo.instId;
    
    nm_mgr = (DirectMulticastStrategy *) 
        CProxy_ComlibManager(CpvAccess(cmgrID)).
        ckLocalBranch()->getStrategy(instid);
    
    nm_mgr->handleMulticastMessage(msg);
    return NULL;
}

//Group Constructor
DirectMulticastStrategy::DirectMulticastStrategy(int ndest, int *pelist)
    : Strategy() {
 
    isDestinationArray = 0;
    isDestinationGroup = 1;

    ndestpes = ndest;
    destpelist = pelist;

    commonInit();
}

DirectMulticastStrategy::DirectMulticastStrategy(CkArrayID aid)
    : destArrayID(aid), Strategy() {

    isDestinationArray = 1;
    isDestinationGroup = 0;
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

void DirectMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
    if(messageBuf == NULL) {
	CkPrintf("ERROR MESSAGE BUF IS NULL\n");
	return;
    }

    ComlibPrintf("[%d] Comlib Direct Multicast: insertMessage \n", 
                 CkMyPe());   
   
    if(cmsg->dest_proc == IS_MULTICAST && cmsg->sec_id != NULL) {        
        int cur_sec_id = cmsg->sec_id->_cookie.sInfo.cInfo.id;

        if(cur_sec_id > 0) {        
            //Old section id, send the id with the message
            CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)cmsg->getCharmMessage();
            cbmsg->_cookie.sInfo.cInfo.id = cur_sec_id;
            cbmsg->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_OLD_SECTION;
        }
        else {
            //New sec id, so send it along with the message
            void *newmsg = (void *)getNewMulticastMessage(cmsg);
            CkFreeMsg(cmsg->getCharmMessage());
            CkSectionID *sid = cmsg->sec_id;
            initSectionID(sid);
            delete cmsg;
            
            cmsg = new CharmMessageHolder((char *)newmsg, IS_MULTICAST); 
            cmsg->sec_id = sid;
        }        
    }
    
    messageBuf->enq(cmsg);
    //if(!isBracketed())
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
        	
        if(cmsg->dest_proc == IS_MULTICAST) {      
            if(isDestinationArray)
                CmiSetHandler(UsrToEnv(msg), handlerId);
            
            int *cur_map = destpelist;
            int cur_npes = ndestpes;
            if(cmsg->sec_id != NULL && cmsg->sec_id->pelist != NULL) {
                cur_map = cmsg->sec_id->pelist;
                cur_npes = cmsg->sec_id->npes;
            }
            
            ComlibPrintf("[%d] Calling Direct Multicast %d %d %d\n", CkMyPe(),
                         UsrToEnv(msg)->getTotalsize(), cur_npes, cmsg->dest_proc);

            CmiSyncListSendAndFree(cur_npes, cur_map, 
                                   UsrToEnv(msg)->getTotalsize(), 
                                   UsrToEnv(msg));            
        }
        else {
            CmiSyncSendAndFree(cmsg->dest_proc, UsrToEnv(msg)->getTotalsize(), 
                               (char *)UsrToEnv(msg));
        }        
        
	delete cmsg; 
    }
}

void DirectMulticastStrategy::pup(PUP::er &p){

    Strategy::pup(p);

    p | ndestpes;
    p | destArrayID;
    p | isDestinationArray;
    p | isDestinationGroup;
        
    if(p.isUnpacking())
        destpelist = new int[ndestpes];
    p(destpelist, ndestpes);        
}

void DirectMulticastStrategy::beginProcessing(int numElements){
    
    messageBuf = new CkQ<CharmMessageHolder *>;    
    handlerId = CmiRegisterHandler((CmiHandler)DMHandler);    
    
    if(isDestinationArray) {
        CkArray *dest_array = CkArrayID::CkLocalBranch(destArrayID);
        dest_array->getComlibArrayListener()->getLocalIndices
            (localDestIndices);
    }

    MaxSectionID = 1;
}

void DirectMulticastStrategy::handleMulticastMessage(void *msg){
    register envelope *env = (envelope *)msg;
    
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);

    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In local multicast %d\n", CkMyPe(), status);
    
    CkVec<CkArrayIndexMax> *dest_indices; 
    if(status == COMLIB_MULTICAST_ALL) {        
        //Multicast to all destination elements on current processor        
        ComlibPrintf("[%d] Local multicast sending all %d\n", CkMyPe(), 
                     localDestIndices.size());

        localMulticast(&localDestIndices, env);
    }   
    else if(status == COMLIB_MULTICAST_NEW_SECTION){        
        CkUnpackMessage(&env);
        dest_indices = new CkVec<CkArrayIndexMax>;

        ComlibPrintf("[%d] Received message for new section %d %d\n", 
                     CkMyPe(), cbmsg->_cookie.pe, 
                     cbmsg->_cookie.sInfo.cInfo.id);

        ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)cbmsg;
        for(int count = 0; count < ccmsg->nIndices; count++){
            CkArrayIndexMax idx = ccmsg->indices[count];
            //idx.print();
            int dest_proc =CkArrayID::CkLocalBranch(destArrayID)
                ->lastKnown(idx);
            
            if(dest_proc == CkMyPe())
                dest_indices->insertAtEnd(idx);                        
        }            
        
        envelope *usrenv = (envelope *) ccmsg->usrMsg;
        envelope *newenv = (envelope *)CmiAlloc(usrenv->getTotalsize());
        memcpy(newenv, ccmsg->usrMsg, usrenv->getTotalsize());
        localMulticast(dest_indices, newenv);

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
        
        localMulticast(dest_indices, env);
    }
}

#include "register.h"
void DirectMulticastStrategy::localMulticast(CkVec<CkArrayIndexMax>*vec, 
                                                 envelope *env){
    
    //Multicast the messages to all elements in vec
    void *msg = EnvToUsr(env);
    int nelements = vec->size();

    if(nelements == 0) {
        CmiFree(env);
        return;
    }
    
    int ep = env->array_ep();
    CkUnpackMessage(&env);
    for(int count = 0; count < nelements; count ++){        
        CkArrayIndexMax idx = (*vec)[count];
        
        ComlibPrintf("[%d] Sending multicast message to ", CkMyPe());        
        if(comm_debug) idx.print();     
        
        CProxyElement_ArrayBase ap(destArrayID, idx);
        ArrayElement *elem = ap.ckLocal();
        CkDeliverMessageReadonly(ep, msg, elem);        
    }
    
    CmiFree(env);
}

void DirectMulticastStrategy::initSectionID(CkSectionID *sid){
    
    if(sid->npes > 0) 
        return;
    
    sid->pelist = new int[ndestpes];
    sid->npes = 0;
    
    int count = 0, acount = 0;
    for(acount = 0; acount < sid->_nElems; acount++){
        int p = CkArrayID::CkLocalBranch(destArrayID)->
            lastKnown(sid->_elems[acount]);
        
        for(count = 0; count < sid->npes; count ++)
            if(sid->pelist[count] == p)
                break;
        
        if(count == sid->npes) {
            sid->pelist[sid->npes ++] = p;
        }
    } 
    
}


ComlibMulticastMsg * DirectMulticastStrategy::getNewMulticastMessage
(CharmMessageHolder *cmsg){
    
    if(cmsg->sec_id == NULL || cmsg->sec_id->_nElems == 0)
        return NULL;

    void *m = cmsg->getCharmMessage();
    envelope *env = UsrToEnv(m);
    
    if(cmsg->sec_id->_cookie.sInfo.cInfo.id == 0) {  //New Section ID;
        CkPackMessage(&env);
        int sizes[2];
        sizes[0] = cmsg->sec_id->_nElems;
        sizes[1] = env->getTotalsize();                

        cmsg->sec_id->_cookie.sInfo.cInfo.id = MaxSectionID ++;
        
        ComlibPrintf("Creating new comlib multicast message %d, %d\n", sizes[0], sizes[1]);

        ComlibMulticastMsg *msg = new(sizes, 0) ComlibMulticastMsg;
        msg->nIndices = cmsg->sec_id->_nElems;
        msg->_cookie.sInfo.cInfo.instId = myInstanceID;
        msg->_cookie.sInfo.cInfo.id = MaxSectionID - 1;
        msg->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_NEW_SECTION;
        msg->_cookie.type = COMLIB_MULTICAST_MESSAGE;
        msg->_cookie.pe = CkMyPe();

        memcpy(msg->indices, cmsg->sec_id->_elems, 
               sizes[0] * sizeof(CkArrayIndexMax));
        memcpy(msg->usrMsg, env, sizes[1] * sizeof(char));         
        envelope *newenv = UsrToEnv(msg);
        
        newenv->array_mgr() = env->array_mgr();
        newenv->array_srcPe() = env->array_srcPe();
        newenv->array_ep() = env->array_ep();
        newenv->array_hops() = env->array_hops();
        newenv->array_index() = env->array_index();

        CkPackMessage(&newenv);        
        return (ComlibMulticastMsg *)EnvToUsr(newenv);
    }   

    return NULL;
}

