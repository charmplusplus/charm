
#include "EachToManyMulticastStrategy.h"
#include "string.h"
#include "routerstrategy.h"

#include "AAPLearner.h"
#include "AAMLearner.h"

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
        ComlibPrintf("[%d] Calling Dummy Done Inserting, %d, %d\n", CkMyPe(), instid, nexpected);
        nm_mgr = (EachToManyMulticastStrategy *)sentry->strategy;    
        nm_mgr->doneInserting();
    }
    
    return NULL;
}

void *E2MHandler(void *msg){
    //CkPrintf("[%d]:In EachtoMany CallbackHandler\n", CkMyPe());
    EachToManyMulticastStrategy *nm_mgr;    

    envelope *env = (envelope *)msg;
    CkMcastBaseMsg *bmsg = (CkMcastBaseMsg *)EnvToUsr(env);
    int instid = bmsg->_cookie.sInfo.cInfo.instId;
    
    nm_mgr = (EachToManyMulticastStrategy *) 
        CProxy_ComlibManager(CkpvAccess(cmgrID)).
        ckLocalBranch()->getStrategy(instid);
    
    RECORD_RECV_STATS(instid, env->getTotalsize(), env->getSrcPe());
    
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

    CkGroupID gid;
    gid.setZero();
    ginfo.setSourceGroup(gid, src_pelist, n_srcpes);    
    ginfo.setDestinationGroup(gid, dest_pelist, n_destpes);

    //Written in this funny way to be symettric with the array case.
    ginfo.getDestinationGroup(gid, destpelist, ndestpes);
    ginfo.getCombinedPeList(pelist, npes);

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
    
    /*
      char dump[1000];
      char sdump[100];
      sprintf(dump, "%d: Each To MANY PELIST :\n", CkMyPe());
      for(int count = 0; count < npes; count ++){
      sprintf(sdump, "%d, ", pelist[count]);
      strcat(dump, sdump);           
      }    
      ComlibPrintf("%s\n", dump);
    */

    commonInit();
}

extern char *router;
//Common initialization for both group and array constructors
void EachToManyMulticastStrategy::commonInit() {

    setBracketed();
    setForwardOnMigration(1);

    mflag = CmiFalse;

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

EachToManyMulticastStrategy::~EachToManyMulticastStrategy() {
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


void EachToManyMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){

    ComlibPrintf("[%d] EachToManyMulticast: insertMessage \n", 
                 CkMyPe());   

    if(cmsg->dest_proc == IS_SECTION_MULTICAST && cmsg->sec_id != NULL) { 
        int cur_sec_id = ComlibSectionInfo::getSectionID(*cmsg->sec_id);

        if(cur_sec_id > 0) {        
            sinfo.processOldSectionMessage(cmsg);
        }
        else {
            //New sec id, so send it along with the message
            void *newmsg = sinfo.getNewMulticastMessage(cmsg);
            CkFreeMsg(cmsg->getCharmMessage());
            CkSectionID *sid = cmsg->sec_id;
            delete cmsg;
            
            cmsg = new CharmMessageHolder((char *)newmsg,
                                          IS_SECTION_MULTICAST); 
            cmsg->sec_id = sid;
            initSectionID(cmsg->sec_id);
        }        

        if(cmsg->sec_id != NULL && cmsg->sec_id->pelist != NULL) {
            cmsg->pelist = cmsg->sec_id->pelist;
            cmsg->npes = cmsg->sec_id->npes;
        }        
    }

    if(cmsg->dest_proc == IS_BROADCAST) {
        cmsg->npes = ndestpes;
        cmsg->pelist = destpelist;
    }

    //For section multicasts and broadcasts
    if(cmsg->dest_proc == IS_SECTION_MULTICAST 
       || cmsg->dest_proc == IS_BROADCAST ) {
        
        //Use Multicast Learner (Foobar will not work for combinations
        //of personalized and multicast messages
        
        if(!mflag) {
            ComlibLearner *l = getLearner();
            if(l != NULL) {
                delete l;
                l = NULL;
            }
            
            AAMLearner *alearner = new AAMLearner();
            setLearner(alearner);
            mflag = CmiTrue;
        }

        CmiSetHandler(UsrToEnv(cmsg->getCharmMessage()), handlerId);

        //Collect Multicast Statistics
        RECORD_SENDM_STATS(getInstance(), 
                           ((envelope *)cmsg->getMessage())->getTotalsize(), 
                           cmsg->pelist, cmsg->npes);
    }
    else {
        //Collect Statistics
        RECORD_SEND_STATS(getInstance(), 
                          ((envelope *)cmsg->getMessage())->getTotalsize(), 
                          cmsg->dest_proc);
    }
    
    rstrat->insertMessage(cmsg);
}

void EachToManyMulticastStrategy::doneInserting(){

    StrategyTableEntry *sentry = 
        CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()
        ->getStrategyTableEntry(getInstance());
    int nexpected = sentry->numElements;

    if(routerID == USE_DIRECT && nexpected == 0)
        return;

    //ComlibPrintf("%d: DoneInserting \n", CkMyPe());    
    rstrat->doneInserting();
}

void EachToManyMulticastStrategy::pup(PUP::er &p){

    int count = 0;
    ComlibPrintf("[%d] Each To many::pup %s\n", CkMyPe(), 
                 ((!p.isUnpacking() == 0)?("UnPacking"):("Packing")));

    CharmStrategy::pup(p);

    p | routerID; 
    p | npes; p | ndestpes;     
    p | mflag;
    
    if(p.isUnpacking() && npes > 0) {
        pelist = new int[npes];    
    }

    if(npes > 0)
        p(pelist, npes);

    if(p.isUnpacking() && ndestpes > 0) {
        destpelist = new int[ndestpes];    
    }    

    if(ndestpes > 0)
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
    
    ComlibPrintf("[%d] Begin processing %d\n", CkMyPe(), numElements);
    
    char dump[1000];
    char sdump[100];
    sprintf(dump, "%d: Each To MANY PELIST :\n", CkMyPe());
    for(int count = 0; count < npes; count ++){
        sprintf(sdump, "%d, ", pelist[count]);
        strcat(dump, sdump);           
    }    
    ComlibPrintf("%s\n", dump);

    int expectedDeposits = 0;

    rstrat->setInstance(getInstance());

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
    
    CkArrayID dest;
    int nidx;
    CkArrayIndexMax *idx_list;
    
    ainfo.getDestinationArray(dest, idx_list, nidx);
    sinfo = ComlibSectionInfo(dest, myInstanceID);
    
    AAPLearner *alearner = new AAPLearner();
    setLearner(alearner);

    if(expectedDeposits > 0)
        return;
    
    if(expectedDeposits == 0 && MyPe >= 0)
        //doneInserting();
        ConvComlibScheduleDoneInserting(myInstanceID);
}

void EachToManyMulticastStrategy::finalizeProcessing() {
    if(npes > 0)
        delete [] pelist;
    
    if(ndestpes > 0)
        delete [] destpelist;

    if(rstrat)
        delete rstrat;

    if(getLearner() != NULL)
        delete getLearner();
}


void EachToManyMulticastStrategy::localMulticast(void *msg){
    register envelope *env = (envelope *)msg;
    CkUnpackMessage(&env);
    
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);

    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In local multicast %d\n", CkMyPe(), status);
        
    if(status == COMLIB_MULTICAST_ALL) {        
        ainfo.localBroadcast(env);
        return;
    }   

    CkVec<CkArrayIndexMax> *dest_indices;    
    if(status == COMLIB_MULTICAST_NEW_SECTION){        
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
        return;       
    }

    //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
    ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                             cbmsg->_cookie.sInfo.cInfo.id);    
    dest_indices = (CkVec<CkArrayIndexMax> *)sec_ht.get(key);

    if(dest_indices == NULL)
        CkAbort("Destination indices is NULL\n");

    ComlibArrayInfo::localMulticast(dest_indices, env);
}

void EachToManyMulticastStrategy::initSectionID(CkSectionID *sid){

    sinfo.initSectionID(sid);    

    //Convert real processor numbers to virtual processors in the all
    //to all multicast group
    for(int count = 0; count < sid->npes; count ++) {
        sid->pelist[count] = rstrat->getProcMap()[sid->pelist[count]]; 
        if(sid->pelist[count] == -1) CkAbort("Invalid Section\n");
    }
}
