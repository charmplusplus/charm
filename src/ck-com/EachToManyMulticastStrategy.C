
#include "EachToManyMulticastStrategy.h"
#include "string.h"
#include "routerstrategy.h"

//EachToManyMulticastStrategy CODE
CkpvExtern(int, RecvdummyHandle);
CkpvExtern(CkGroupID, cmgrID);

void *itrDoneHandler(void *msg){

    EachToManyMulticastStrategy *nm_mgr;
    
    DummyMsg *dmsg = (DummyMsg *)msg;
    comID id = dmsg->id;
    int instid = id.instanceID;

    CmiFree(msg);
    ComlibPrintf("[%d] Iteration finished %d\n", CkMyPe(), instid);

    StrategyTableEntry *sentry = 
        CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()
        ->getStrategyTableEntry(instid);
    int nexpected = sentry->numElements;
    
    if(nexpected == 0) {             
        CkPrintf("[%d] Calling Dummy Done Inserting\n", CkMyPe());
        nm_mgr = (EachToManyMulticastStrategy *)sentry->strategy;    
        nm_mgr->doneInserting();
    }
    
    return NULL;
}

void *E2MHandler(void *msg){
    //CkPrintf("[%d]:In EachtoMany CallbackHandler\n", CkMyPe());
    EachToManyMulticastStrategy *nm_mgr;    
    
    CkMcastBaseMsg *bmsg = (CkMcastBaseMsg *)EnvToUsr((envelope *)msg);
    int instid = bmsg->_cookie.sInfo.cInfo.instId;
    
    nm_mgr = (EachToManyMulticastStrategy *) 
        CProxy_ComlibManager(CkpvAccess(cmgrID)).
        ckLocalBranch()->getStrategy(instid);
    
    nm_mgr->localMulticast(msg);
    return NULL;
}

//Group Constructor
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy, 
                                                         int n_srcpes, 
                                                         int *src_pelist,
                                                         int n_destpes, 
                                                         int *dest_pelist) 
    : routerID(substrategy), CharmStrategy() {
    
    setType(GROUP_STRATEGY);

    int count = 0;

    if(n_srcpes == 0) {
        n_srcpes = CkNumPes();
        src_pelist = new int[n_srcpes];
        for(count =0; count < n_srcpes; count ++)
            src_pelist[count] = count;
    }
    
    CkGroupID gid;
    gid.setZero();
    ginfo.setSourceGroup(gid, src_pelist, n_srcpes);    

    if(n_destpes == 0) {
        ndestpes = CkNumPes();
        destpelist = new int[ndestpes];
        for(count =0; count < ndestpes; count ++)
            destpelist[count] = count;
    }
    else {
        ndestpes = n_destpes;
        destpelist = dest_pelist;
    }

    if(n_srcpes == 0){
        pelist = src_pelist;
        npes = n_srcpes;

        commonInit();
        return;
    }

    if(n_destpes == 0) {
        pelist = destpelist;
        npes = ndestpes;
        
        commonInit();
        return;
    }
    
    //source and destination lists are both subsets
    pelist = new int[CkNumPes()];
    npes = n_srcpes;
    memcpy(pelist, src_pelist, n_srcpes * sizeof(int));
    
    for(int dcount = 0; dcount < ndestpes; dcount++) {
        int p = destpelist[dcount];
        
        for(count = 0; count < npes; count ++)
            if(pelist[count] == p)
                break;
        
        if(count == npes)
            pelist[npes++] = p;
    }    

    commonInit();
}

//Array Constructor
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy, 
                                                         CkArrayID src, 
                                                         CkArrayID dest, 
                                                         int nsrc, 
                                                         CkArrayIndexMax 
                                                         *srcelements, 
                                                         int ndest, 
                                                         CkArrayIndexMax 
                                                         *destelements)
    :routerID(substrategy), CharmStrategy() {

    setType(ARRAY_STRATEGY);
    ainfo.setSourceArray(src, srcelements, nsrc);
    ainfo.setDestinationArray(dest, destelements, ndest);

    ainfo.getDestinationPeList(destpelist, ndestpes);
    ainfo.getCombinedPeList(pelist, npes);
    
    //    for(int count = 0; count < npes; count ++){
    //CkPrintf("%d, ", pelist[count]);
    //}    
    //CkPrintf("\n");

    commonInit();
}

extern char *router;
//Common initialization for both group and array constructors
void EachToManyMulticastStrategy::commonInit() {

    setBracketed();

    if(CkMyPe() == 0 && router != NULL){
        if(strcmp(router, "USE_MESH") == 0)
            routerID = USE_MESH;
        else if(strcmp(router, "USE_GRID") == 0)
            routerID = USE_GRID;
        else  if(strcmp(router, "USE_HYPERCUBE") == 0)
            routerID = USE_HYPERCUBE;
        else  if(strcmp(router, "USE_DIRECT") == 0)
            routerID = USE_DIRECT;        
    }
    
    ComlibPrintf("Creating Strategy %d\n", routerID);

    rstrat = NULL;
}


void EachToManyMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){

    ComlibPrintf("[%d] EachToManyMulticast: insertMessage \n", 
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
            delete cmsg;
            
            cmsg = new CharmMessageHolder((char *)newmsg, IS_MULTICAST); 
            cmsg->sec_id = sid;
            initSectionID(cmsg->sec_id);
        }        

        if(cmsg->sec_id != NULL && cmsg->sec_id->pelist != NULL) {
            cmsg->pelist = cmsg->sec_id->pelist;
            cmsg->npes = cmsg->sec_id->npes;
        }
        
        CmiSetHandler(UsrToEnv(cmsg->getCharmMessage()), handlerId);
    }
    
    rstrat->insertMessage(cmsg);
}

void EachToManyMulticastStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    
    rstrat->doneInserting();
}

void EachToManyMulticastStrategy::pup(PUP::er &p){

    int count = 0;
    ComlibPrintf("[%d] Each To many::pup %s\n", CkMyPe(), 
                 ((p.isPacking()==0)?("UnPacking"):("Packing")));

    CharmStrategy::pup(p);

    p | routerID; 
    p | npes; p | ndestpes;     
    
    if(p.isUnpacking()) {
        pelist = new int[npes];    
    }
    p(pelist, npes);

    if(p.isUnpacking()) {
        destpelist = new int[ndestpes];    
    }    

    p(destpelist, ndestpes);

    if(p.isUnpacking()){
	handlerId = CkRegisterHandler((CmiHandler)E2MHandler);
        int handler = CkRegisterHandler((CmiHandler)itrDoneHandler);
        
        rstrat = new RouterStrategy(routerID, handler, npes, pelist);
        setConverseStrategy(rstrat);
        MyPe = rstrat->getProcMap()[CkMyPe()];
    }
    
    ComlibPrintf("[%d] End of pup\n", CkMyPe());
}

void EachToManyMulticastStrategy::beginProcessing(int numElements){

    int expectedDeposits = 0;
    MaxSectionID = 0;

    if(ainfo.isSourceArray()) 
        expectedDeposits = numElements;

    if(getType() == GROUP_STRATEGY) {
        
        CkGroupID gid;
        int *srcpelist;
        int nsrcpes;
        
        ginfo.getSourceGroup(gid, srcpelist, nsrcpes);
        
        for(int count = 0; count < nsrcpes; count ++)
            if(srcpelist[count] == CkMyPe()){
                expectedDeposits = 1;
                break;
            }
        
        StrategyTableEntry *sentry = 
            CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()
            ->getStrategyTableEntry(myInstanceID);
        sentry->numElements = expectedDeposits;
    }
    
    if(expectedDeposits > 0)
        return;
    
    if(expectedDeposits == 0 && MyPe >= 0)
        doneInserting();
}

void EachToManyMulticastStrategy::localMulticast(void *msg){
    register envelope *env = (envelope *)msg;
    CkUnpackMessage(&env);
    
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);

    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In local multicast %d\n", CkMyPe(), status);
        
    if(status == COMLIB_MULTICAST_ALL) {        
        ainfo.localMulticast(env);
        return;
    }   

    CkVec<CkArrayIndexMax> *dest_indices;    
    if(status == COMLIB_MULTICAST_NEW_SECTION){        

        dest_indices = new CkVec<CkArrayIndexMax>;

        //CkPrintf("[%d] Received message for new section\n", CkMyPe());

        CkArrayID destArrayID;
        int nDestElements;
        CkArrayIndexMax *destelements;
        ainfo.getSourceArray(destArrayID, destelements, nDestElements);

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

        ainfo.localMulticast(dest_indices, newenv);

        CkVec<CkArrayIndexMax> *old_dest_indices;
        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);

        old_dest_indices = (CkVec<CkArrayIndexMax> *)sec_ht.get(key);
        if(old_dest_indices != NULL)
            delete old_dest_indices;
        
        sec_ht.put(key) = dest_indices;
        CmiFree(env);
        return;       
    }

    //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
    ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                             cbmsg->_cookie.sInfo.cInfo.id);    
    dest_indices = (CkVec<CkArrayIndexMax> *)sec_ht.get(key);

    if(dest_indices == NULL)
        CkAbort("Destination indices is NULL\n");

    ainfo.localMulticast(dest_indices, env);
}

ComlibMulticastMsg * EachToManyMulticastStrategy::getNewMulticastMessage
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
        msg->_cookie.type = COMLIB_MULTICAST_MESSAGE;
        msg->_cookie.sInfo.cInfo.id = MaxSectionID - 1;
        msg->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_NEW_SECTION;
        msg->_cookie.pe = CkMyPe();

        memcpy(msg->indices, cmsg->sec_id->_elems, 
               sizes[0] * sizeof(CkArrayIndexMax));
        memcpy(msg->usrMsg, env, sizes[1] * sizeof(char));         
        envelope *newenv = UsrToEnv(msg);
        
        newenv->getsetArrayMgr() = env->getsetArrayMgr();
        newenv->getsetArraySrcPe() = env->getsetArraySrcPe();
        newenv->getsetArrayEp() = env->getsetArrayEp();
        newenv->getsetArrayHops() = env->getsetArrayHops();
        newenv->getsetArrayIndex() = env->getsetArrayIndex();
	// for trace projections
        newenv->setEvent(env->getEvent());
        newenv->setSrcPe(env->getSrcPe());

        CkPackMessage(&newenv);        
        return (ComlibMulticastMsg *)EnvToUsr(newenv);
    }   

    return NULL;
}


void EachToManyMulticastStrategy::initSectionID(CkSectionID *sid){

    ainfo.initSectionID(sid);    

    //Convert real processor numbers to virtual processors in the all
    //to all multicast group
    for(int count = 0; count < sid->npes; count ++) {
        sid->pelist[count] = rstrat->getProcMap()[sid->pelist[count]];        
        if(sid->pelist[count] == -1) CkAbort("Invalid Section\n");
    }
}
