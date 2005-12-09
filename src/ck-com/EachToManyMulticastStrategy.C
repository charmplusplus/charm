
/*********************************************************
            The EachToManyMulticast Strategy optimizes all-to-all
            communication. It combines messages and sends them along
            virtual topologies 2d mesh, 3d mesh and hypercube.

            For large messages send them directly.

            This is the object level strategy. For processor level
            optimizations routers are called.

  - Sameer Kumar.

**********************************************************/


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
        ComlibPrintf("[%d] Calling Dummy Done Inserting, %d, %d\n", 
                     CkMyPe(), instid, nexpected);
        nm_mgr = (EachToManyMulticastStrategy *)sentry->strategy;    
        nm_mgr->doneInserting();
	
	if (!nm_mgr->getOnFinish().isInvalid()) nm_mgr->getOnFinish().send(0);
    
    }

    return NULL;
}

void *E2MHandler(void *msg){
    //CkPrintf("[%d]:In EachtoMany CallbackHandler\n", CkMyPe());
    EachToManyMulticastStrategy *nm_mgr;    
    
    CmiMsgHeaderExt *conv_header = (CmiMsgHeaderExt *) msg;
    int instid = conv_header->stratid;

    envelope *env = (envelope *)msg;
    
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

    //Written in this funny way to be symmetric with the array case.
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

    if(CkMyPe() == 0 && router != NULL){
        if(strcmp(router, "USE_MESH") == 0)
            routerID = USE_MESH;
        else if(strcmp(router, "USE_GRID") == 0)
            routerID = USE_GRID;
        else  if(strcmp(router, "USE_HYPERCUBE") == 0)
            routerID = USE_HYPERCUBE;
        else  if(strcmp(router, "USE_DIRECT") == 0)
            routerID = USE_DIRECT;        
        else  if(strcmp(router, "USE_PREFIX") == 0)
            routerID = USE_PREFIX;        

        //Just for the first step. After learning the learned
        //strategies will be chosen
        router = NULL;
    }
    
    ComlibPrintf("Creating Strategy %d\n", routerID);

    useLearner = 0;
    rstrat = NULL;
}

EachToManyMulticastStrategy::~EachToManyMulticastStrategy() {
}


void EachToManyMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){

    ComlibPrintf("[%d] EachToManyMulticast: insertMessage \n", 
                 CkMyPe());   

    envelope *env = UsrToEnv(cmsg->getCharmMessage());

    if(cmsg->dest_proc == IS_BROADCAST) {
        //All to all multicast
        
        cmsg->npes = ndestpes;
        cmsg->pelist = destpelist;
        
        //Use Multicast Learner (Foobar will not work for combinations
        //of personalized and multicast messages
        
        CmiSetHandler(env, handlerId);

        //Collect Multicast Statistics
        RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), 
                           cmsg->pelist, cmsg->npes);
    }
    else {
        //All to all personalized

        //Collect Statistics
        RECORD_SEND_STATS(getInstance(), env->getTotalsize(), 
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
    
    if(MyPe < 0)
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
    p | useLearner;

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
        
        if(npes > 0) {
            rstrat = new RouterStrategy(routerID, handler, npes, pelist);
            setConverseStrategy(rstrat);
            MyPe = rstrat->getProcMap()[CkMyPe()];
        }
        else MyPe = -1;
    }
    
    ComlibPrintf("[%d] End of pup\n", CkMyPe());
}

void EachToManyMulticastStrategy::beginProcessing(int numElements){
    
    ComlibPrintf("[%d] Begin processing %d\n", CkMyPe(), numElements);
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
    
    if(useLearner) {
        if(!mflag) 
            setLearner(new AAPLearner());    
        else 
            setLearner(new AAMLearner());                
    }
    
    if(expectedDeposits > 0)
        return;
    
    if(expectedDeposits == 0 && MyPe >= 0)
        ConvComlibScheduleDoneInserting(myInstanceID);
}

void EachToManyMulticastStrategy::finalizeProcessing() {
    if(npes > 0)
        delete [] pelist;
    
    if(ndestpes > 0)
        delete [] destpelist;

    if(rstrat)
        delete rstrat;

    if(useLearner && getLearner() != NULL)
        delete getLearner();
}

void EachToManyMulticastStrategy::localMulticast(void *msg){
    register envelope *env = (envelope *)msg;
    CkUnpackMessage(&env);
    
    ainfo.localBroadcast(env);
}

