#include "EachToManyMulticastStrategy.h"
#include "commlib.h"

CpvExtern(int, RecvdummyHandle);
CpvExtern(CkGroupID, cmgrID);

void *itrDoneHandler(void *msg){

    EachToManyMulticastStrategy *nm_mgr;
    
    DummyMsg *dmsg = (DummyMsg *)msg;
    comID id = dmsg->id;
    int instid = id.instanceID;

    ComlibPrintf("[%d] Iteration finished %d\n", CkMyPe(), instid);

    StrategyTable *sentry = 
        CProxy_ComlibManager(CpvAccess(cmgrID)).ckLocalBranch()
        ->getStrategyTableEntry(instid);
    int nexpected = sentry->numElements;
    
    if(nexpected == 0) {               
        nm_mgr = (EachToManyMulticastStrategy *)sentry->strategy;    
        nm_mgr->doneInserting();
    }
    
    return NULL;
}

void *E2MHandler(void *msg){
    //ComlibPrintf("[%d]:In EachtoMany CallbackHandler\n", CkMyPe());
    EachToManyMulticastStrategy *nm_mgr;    
    
    CkMcastBaseMsg *bmsg = (CkMcastBaseMsg *)EnvToUsr((envelope *)msg);
    int instid = bmsg->_cookie.sInfo.cInfo.instId;
    
    nm_mgr = (EachToManyMulticastStrategy *) 
        CProxy_ComlibManager(CpvAccess(cmgrID)).
        ckLocalBranch()->getStrategy(instid);
    
    nm_mgr->localMulticast(msg);
    return NULL;
}

void EachToManyMulticastStrategy::setReverseMap(){
    int pcount;
    for(pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;
    
    for(pcount = 0; pcount < npes; pcount++) 
        procMap[pelist[pcount]] = pcount;
}

//Group Constructors
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy, 
                                                         int n_srcpes,
                                                         int *src_pelist)

    : routerID(substrategy) {
    isGroup = 1;
    if (n_srcpes == 0) {
        npes = CkNumPes();
        pelist = new int[npes];
        for(int count =0; count < npes; count ++)
            pelist[count] = count;

        srcpelist = destpelist = pelist;
        nsrcpes = ndestpes = npes;

        init();
        return;
    }

    npes   = nsrcpes   = ndestpes   = n_srcpes;
    pelist = srcpelist = destpelist = src_pelist;
    
    init();
}

//Array Constructor
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy, 
                                                         int n_srcpes, 
                                                         int *src_pelist,
                                                         int n_destpes, 
                                                         int *dest_pelist) 
    : routerID(substrategy) {
    
    isGroup = 1;
    srcpelist = src_pelist;
    nsrcpes = n_srcpes;
    
    int count = 0;
    if(n_destpes == 0 || n_destpes == CkNumPes()) {
        ndestpes = npes = CkNumPes();
        destpelist = pelist = new int[npes];
        for(count =0; count < npes; count ++)
            pelist[count] = count;

        init();
        return;
    }

    ndestpes = n_destpes;
    destpelist = dest_pelist;

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
}


void EachToManyMulticastStrategy::init() {

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

//Array Constructors
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy, 
                                                         CkArrayID src, 
                                                         CkArrayID dest, 
                                                         int nsrc, 
                                                         CkArrayIndexMax 
                                                         *srcelements, 
                                                         int ndest, 
                                                         CkArrayIndexMax 
                                                         *destelements)
    :routerID(substrategy) {
    
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
        init();
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
    
    init();
}


void EachToManyMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
    
    if(messageBuf == NULL) {
	CkPrintf("ERROR MESSAGE BUF IS NULL\n");
	return;
    }
    ComlibPrintf("[%d] EachToManyMulticast: insertMessage, %d\n", 
                 CkMyPe(), cmsg->nIndices);
        
    if(cmsg->nIndices != nDestElements && cmsg->nIndices > 0){
        void *newmsg = (void *)ComlibManager::
            getPackedMulticastMessage(cmsg);
        CkFreeMsg(cmsg->getCharmMessage());
        delete cmsg;
        
        cmsg = new CharmMessageHolder((char *)newmsg, -1);
    }
    
    messageBuf->enq(cmsg);
}

void EachToManyMulticastStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    
    if((messageBuf->length() == 0) && (CkNumPes() > 0)) {
        ComlibDummyMsg * dummymsg = new ComlibDummyMsg;
        
        ComlibPrintf("[%d] Creating a dummy message\n", CkMyPe());
        
        CmiSetHandler(UsrToEnv(dummymsg), 
                      CpvAccess(RecvdummyHandle));
        
        CharmMessageHolder *cmsg = new CharmMessageHolder((char *)dummymsg, 
                                                          CkMyPe());
        cmsg->isDummy = 1;
        messageBuf->enq(cmsg);
    }

    NumDeposits(comid, messageBuf->length());
    
    while(!messageBuf->isEmpty()) {
	CharmMessageHolder *cmsg = messageBuf->deq();
        char *msg = cmsg->getCharmMessage();

        ComlibPrintf("Calling EachToMany %d %d %d\n", 
                     UsrToEnv(msg)->getTotalsize(), CkMyPe(), 
                     ndestpes);
        	
        if(!cmsg->isDummy)  {
            if(cmsg->dest_proc == IS_MULTICAST) {
                CmiSetHandler(UsrToEnv(msg), handlerId);
                EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
                                    UsrToEnv(msg), ndestpes, destMap);
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
    }
    
    ComlibPrintf("[%d] ndestelements = %d\n", CkMyPe(), nDestElements);
    //destIndices[0].print();
    
    if(p.isUnpacking()){
	messageBuf = new CkQ<CharmMessageHolder *>;
	handlerId = CmiRegisterHandler((CmiHandler)E2MHandler);

        MyPe = procMap[CkMyPe()];
        
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

    ComlibPrintf("[%d] End of pup\n", CkMyPe());
}

void EachToManyMulticastStrategy::beginProcessing(int numElements){
    int handler = CmiRegisterHandler((CmiHandler)itrDoneHandler);
    ComlibPrintf("[%d]Registering Callback Handler\n", CkMyPe());
    comid.callbackHandler = handler;
    comid.instanceID = myInstanceID;
    
    int expectedDeposits;
    if(isArray) 
        expectedDeposits = 
            CProxy_ComlibManager(CpvAccess(cmgrID)).ckLocalBranch()->
            getStrategyTableEntry(myInstanceID)->numElements;        
    
    if(isGroup)
        for(int count = 0; count < nsrcpes; count ++)
            if(srcpelist[count] == CkMyPe())
                expectedDeposits = 1;
    
    if(expectedDeposits > 0)
        return;
    
    if(MyPe != -1)
        doneInserting();
}

void EachToManyMulticastStrategy::localMulticast(void *msg){
    register envelope *env = (envelope *)msg;
    CkUnpackMessage(&env);
    
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);

    int nto_send = cbmsg->_cookie.sInfo.cInfo.nIndices;
    ComlibPrintf("[%d] In local multicast %d\n", CkMyPe(), nto_send);
        
    if(nto_send == 0) {        
        //Multicast to all destination elements on current processor        
        ComlibPrintf("[%d] Local multicast sending all %d\n", CkMyPe(), 
                     localDestIndices.size());

        localMulticast(localDestIndices, env);
        return;
    }   

    CkVec<CkArrayIndexMax> dest_indices;
    ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)cbmsg;
    
    for(int count = 0; count < nto_send; count++){
        CkArrayIndexMax idx = ccmsg->indices[count];
        //idx.print();
        int dest_proc =CkArrayID::CkLocalBranch(destArrayID)->lastKnown(idx);
        if(dest_proc == CkMyPe())
            dest_indices.insertAtEnd(idx);
    }
    
    envelope *newenv = (envelope *)CmiAlloc(ccmsg->size);
    memcpy(newenv, ccmsg->usrMsg, ccmsg->size);
    CmiFree(env);
    localMulticast(dest_indices, newenv);
}

void EachToManyMulticastStrategy::localMulticast(CkVec<CkArrayIndexMax> vec, 
                                         envelope *env){
    
    //Multicast the messages to all elements in vec
    void *msg = EnvToUsr(env);
    int nelements = vec.size();

    void *newmsg;
    envelope *newenv;
    for(int count = 0; count < nelements; count ++){

        CkArrayIndexMax idx = vec[count];
        
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
        ap.ckSend((CkArrayMessage *)newmsg, newenv->array_ep());
    }
}
