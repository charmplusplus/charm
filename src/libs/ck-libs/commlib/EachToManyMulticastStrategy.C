#include "EachToManyMulticastStrategy.h"
#include "commlib.h"
#include "string.h"

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

    StrategyTable *sentry = 
        CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()
        ->getStrategyTableEntry(instid);
    int nexpected = sentry->numElements;
    
    if(nexpected == 0) {             
        ComlibPrintf("[%d] Calling Dummy Done Inserting\n", CkMyPe());
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

void EachToManyMulticastStrategy::setReverseMap(){
    int pcount;
    for(pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;
    
    //All processors not in the domain will point to -1
    for(pcount = 0; pcount < npes; pcount++) 
        procMap[pelist[pcount]] = pcount;
}

//Group Constructor
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy, 
                                                         int n_srcpes, 
                                                         int *src_pelist,
                                                         int n_destpes, 
                                                         int *dest_pelist) 
    : routerID(substrategy), Strategy() {
    
    isGroup = 1;

    //    destArrayID = dummy;  
    nDestElements = 0;  
    destIndices = NULL; 
    
    int count = 0;

    if(n_srcpes == 0) {
        nsrcpes = CkNumPes();
        srcpelist = new int[nsrcpes];
        for(count =0; count < nsrcpes; count ++)
            srcpelist[count] = count;
    }
    else {
        srcpelist = src_pelist;
        nsrcpes = n_srcpes;
    }    
    
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
        pelist = srcpelist;
        npes = nsrcpes;

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
    npes = nsrcpes;
    memcpy(pelist, srcpelist, nsrcpes * sizeof(int));
    
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
    :routerID(substrategy), Strategy() {
    
    isArray = 1; //the source is an array.
    aid = src; //Source array id.
    nIndices = nsrc;  //0 for all array elements
    elements = srcelements; //Null for all array elements

    destArrayID = dest;  
    nDestElements = ndest;  //0 for all array elements
    destIndices = destelements; //Null for all array elements

    if(nsrc > 0 && elements == NULL)
        CkAbort("Invalid parameters to EachToMany Consrtuctor \n");

    if(ndest > 0 && destIndices == NULL)
        CkAbort("Invalid parameters to EachToMany Consrtuctor \n");

    int count = 0, acount =0;
    npes = CkNumPes();
    pelist = new int[npes];    
    ndestpes = CkNumPes();
    destpelist = new int[npes];

    memset(pelist, 0, npes * sizeof(int));
    memset(destpelist, 0, npes * sizeof(int));    

    if(ndest == 0){
        for(count =0; count < CkNumPes(); count ++) {
            pelist[count] = count;                 
            destpelist[count] = count;     
        }    
        commonInit();
        return;
    }

    ndestpes = 0;
    npes = 0;
    for(acount = 0; acount < ndest; acount++) {
        int p = CkArrayID::CkLocalBranch(dest)->
            lastKnown(destelements[acount]);        
        
        for(count = 0; count < ndestpes; count ++)
            if(destpelist[count] == p)
                break;
        if(count == ndestpes) {
            destpelist[ndestpes ++] = p; 
            pelist[npes ++] = p;
        }       
    }                            

    if(nsrc == 0) {
        for(count =0; count < CkNumPes(); count ++) 
            pelist[count] = count;                 
        npes = CkNumPes();
    }
    else {
        npes = 0;
        for(acount = 0; acount < nsrc; acount++) {
            int p = CkArrayID::CkLocalBranch(src)->
                lastKnown(srcelements[acount]);
            
            for(count = 0; count < npes; count ++)
                if(pelist[count] == p)
                    break;
            if(count == npes)
                pelist[npes ++] = p;
        }                        
    }
    
    commonInit();
}

extern char *router;
//Common initialization for both group and array constructors
void EachToManyMulticastStrategy::commonInit() {

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
    
    CkPrintf("Creating Strategy %d\n", routerID);

    messageBuf = 0;

    ComlibPrintf("Before instance\n");
    comid = ComlibInstance(routerID, CkNumPes());
    if(npes < CkNumPes())
	comid = ComlibEstablishGroup(comid, this->npes, pelist);
    ComlibPrintf("After instance\n");    

    destMap = new int[ndestpes];   
    int count = 0;
    if(ndestpes < CkNumPes()){
        for(int dcount = 0; dcount < ndestpes; dcount ++) {            
            for(count = 0; count < npes; count ++)
                if(pelist[count] == destpelist[dcount])
                    break;
            destMap[dcount] = count;
        }
    }
    else
        destMap = pelist;

    procMap = new int[CkNumPes()];
    setReverseMap();
}


void EachToManyMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
    
    if(messageBuf == NULL) {
	CkAbort("ERROR MESSAGE BUF IS NULL\n");
	return;
    }

    ComlibPrintf("[%d] EachToManyMulticast: insertMessage \n", 
                 CkMyPe());   

    if(routerID == USE_DIRECT && cmsg->dest_proc >= 0){
        char *msg = cmsg->getCharmMessage();
        CmiSyncSendAndFree(cmsg->dest_proc, UsrToEnv(msg)->getTotalsize(), 
                           (char *)UsrToEnv(msg));
        delete cmsg;
        return;
    }
   
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
    }
    
    messageBuf->enq(cmsg);
}

void EachToManyMulticastStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    
    if(messageBuf->length() == 0) {
        if(routerID == USE_DIRECT)
            return; 

        ComlibDummyMsg * dummymsg = new ComlibDummyMsg;
        ComlibPrintf("[%d] Creating a dummy message\n", CkMyPe());
        CmiSetHandler(UsrToEnv(dummymsg), 
                      CkpvAccess(RecvdummyHandle));
        
        CharmMessageHolder *cmsg = new CharmMessageHolder((char *)dummymsg, 
                                                          CkMyPe());
        cmsg->isDummy = 1;
        messageBuf->enq(cmsg);
    }

    NumDeposits(comid, messageBuf->length());
    
    while(!messageBuf->isEmpty()) {
	CharmMessageHolder *cmsg = messageBuf->deq();
        char *msg = cmsg->getCharmMessage();

        ComlibPrintf("[%d] Calling EachToMany %d %d %d\n", CkMyPe(),
                     UsrToEnv(msg)->getTotalsize(), 
                     ndestpes, cmsg->dest_proc);
        	
        if(!cmsg->isDummy)  {
            if(cmsg->dest_proc == IS_MULTICAST) {      
                if(isArray)
                    CmiSetHandler(UsrToEnv(msg), handlerId);

                int *cur_map = destMap;
                int cur_npes = ndestpes;
                if(cmsg->sec_id != NULL && cmsg->sec_id->pelist != NULL) {
                    cur_map = cmsg->sec_id->pelist;
                    cur_npes = cmsg->sec_id->npes;
                }
                
                EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
                                    UsrToEnv(msg), cur_npes, cur_map);
            }
            else {
                EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
                                    UsrToEnv(msg), 1, 
                                    &procMap[cmsg->dest_proc]);
            }
        }        
        else
            EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
                                UsrToEnv(msg), 1, &MyPe);
            
	delete cmsg; 
    }
}

void EachToManyMulticastStrategy::pup(PUP::er &p){

    int count = 0;
    ComlibPrintf("[%d] Each To many::pup %s\n", CkMyPe(), 
                 ((p.isPacking()==0)?("UnPacking"):("Packing")));

    Strategy::pup(p);

    p | routerID; p | comid;
    p | npes; p | ndestpes;     
    p | destArrayID; p | nDestElements;

    ComlibPrintf("%d %d %d\n", npes, ndestpes, nDestElements);

    if(p.isUnpacking()) {
        pelist = new int[npes];    
        procMap = new int[CkNumPes()];
    }
    p(pelist, npes);
    p(procMap, CkNumPes());
    
    if(p.isUnpacking()) {
        destpelist = new int[ndestpes];    
        destMap = new int[ndestpes];
    }    
    p(destpelist, ndestpes);
    p(destMap, ndestpes);
        
    if(nDestElements > 0){
        if(p.isUnpacking())
            destIndices = new CkArrayIndexMax[nDestElements];
        p((char *)destIndices, nDestElements * sizeof(CkArrayIndexMax));
        //p(destIndices, nDestElements);
    }
    
    ComlibPrintf("[%d] ndestelements = %d\n", CkMyPe(), nDestElements);
    //destIndices[0].print();
    
    if(p.isUnpacking()){
	messageBuf = new CkQ<CharmMessageHolder *>;
	handlerId = CkRegisterHandler((CmiHandler)E2MHandler);

        MyPe = procMap[CkMyPe()];
        
        if(isArray) {
            CkArray *dest_array = CkArrayID::CkLocalBranch(destArrayID);
            if(nDestElements == 0){            
                dest_array->getComlibArrayListener()->getLocalIndices
                    (localDestIndices);
            }
            else {
                for(count = 0; count < nDestElements; count++) 
                    if(dest_array->lastKnown(destIndices[count]) == CkMyPe())
                        localDestIndices.insertAtEnd(destIndices[count]);
            }
        }
    }

    ComlibPrintf("[%d] End of pup\n", CkMyPe());
}

void EachToManyMulticastStrategy::beginProcessing(int numElements){
    int handler = CkRegisterHandler((CmiHandler)itrDoneHandler);
    ComlibPrintf("[%d]Registering Callback Handler\n", CkMyPe());
    comid.callbackHandler = handler;
    comid.instanceID = myInstanceID;
    
    int expectedDeposits = 0;
    MaxSectionID = 0;
    if(isArray) 
        expectedDeposits = 
            CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()->
            getStrategyTableEntry(myInstanceID)->numElements;        
    
    if(isGroup) {
        for(int count = 0; count < nsrcpes; count ++)
            if(srcpelist[count] == CkMyPe()){
                expectedDeposits = 1;
                break;
            }
        
        StrategyTable *sentry = 
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
        //Multicast to all destination elements on current processor        
        ComlibPrintf("[%d] Local multicast sending all %d\n", CkMyPe(), 
                     localDestIndices.size());

        localMulticast(&localDestIndices, env);
        return;
    }   

    CkVec<CkArrayIndexMax> *dest_indices;    
    if(status == COMLIB_MULTICAST_NEW_SECTION){        

        dest_indices = new CkVec<CkArrayIndexMax>;

        //CkPrintf("[%d] Received message for new section\n", CkMyPe());

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
        return;       
    }

    //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
    ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                             cbmsg->_cookie.sInfo.cInfo.id);    
    dest_indices = (CkVec<CkArrayIndexMax> *)sec_ht.get(key);

    if(dest_indices == NULL)
        CkAbort("Destination indices is NULL\n");

    localMulticast(dest_indices, env);
}

void EachToManyMulticastStrategy::localMulticast(CkVec<CkArrayIndexMax>*vec, 
                                                 envelope *env){
    
    //Multicast the messages to all elements in vec
    void *msg = EnvToUsr(env);
    int nelements = vec->size();

    if(nelements == 0)
        CmiFree(env);

    void *newmsg;
    envelope *newenv;
    for(int count = 0; count < nelements; count ++){

        CkArrayIndexMax idx = (*vec)[count];
        
        //CkPrintf("[%d] Sending multicast message to", CkMyPe());
        //idx.print();     
        
        if(count < nelements - 1) {
            newmsg = CkCopyMsg(&msg); 
            newenv = UsrToEnv(newmsg);
            newenv->setUsed(0);            
        }
        else {
            newmsg = msg;
            newenv = env;
        }
        
        CProxyElement_ArrayBase ap(destArrayID, idx);
        ap.ckSend((CkArrayMessage *)newmsg, newenv->getsetArrayEp());
    }
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
    
    if(sid->npes > 0) 
        return;

    sid->pelist = new int[ndestpes];
    sid->npes = 0;
    
    int count = 0, acount = 0;
    for(acount = 0; acount < sid->_nElems; acount++){
        int p = CkArrayID::CkLocalBranch(destArrayID)->
            lastKnown(sid->_elems[acount]);
                
        p = procMap[p];
        if(p == -1) CkAbort("Invalid Section\n");
        
        for(count = 0; count < sid->npes; count ++)
            if(sid->pelist[count] == p)
                break;
        
        if(count == sid->npes) {
            sid->pelist[sid->npes ++] = p;
        }
    }   
}
