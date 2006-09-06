/**
   @addtogroup Comlib
*/
/*@{*/
#include "ComlibManager.h"
#include "EachToManyMulticastStrategy.h"
#include "DirectMulticastStrategy.h"
#include "StreamingStrategy.h"
#include "DummyStrategy.h"
#include "MPIStrategy.h"
#include "NodeMulticast.h"
#include "MsgPacker.h"
#include "RingMulticastStrategy.h"
#include "MultiRingMulticast.h"
#include "PipeBroadcastStrategy.h"
#include "BroadcastStrategy.h"
#include "MeshStreamingStrategy.h"
#include "PrioStreaming.h"

CkpvExtern(int, RecvdummyHandle);

CkpvDeclare(int, RecvmsgHandle);
CkpvDeclare(int, RecvCombinedShortMsgHdlrIdx);
CkpvDeclare(CkGroupID, cmgrID);
CkpvExtern(ConvComlibManager *, conv_com_ptr);

//Handler to receive array messages
void recv_array_msg(void *msg){

    ComlibPrintf("%d:In recv_msg\n", CkMyPe());

    if(msg == NULL)
        return;
    
    register envelope* env = (envelope *)msg;
    env->setUsed(0);
    env->getsetArrayHops()=1;
    CkUnpackMessage(&env);

    int srcPe = env->getSrcPe();
    int sid = ((CmiMsgHeaderExt *) env)->stratid;

    ComlibPrintf("%d: Recording receive %d, %d, %d\n", CkMyPe(), 
             sid, env->getTotalsize(), srcPe);

    RECORD_RECV_STATS(sid, env->getTotalsize(), srcPe);
    
    CkArray *a=(CkArray *)_localBranch(env->getsetArrayMgr());
    //if(!comm_debug)
    a->deliver((CkArrayMessage *)EnvToUsr(env), CkDeliver_queue);

    ComlibPrintf("%d:Out of recv_msg\n", CkMyPe());
    return;
}

void recv_combined_array_msg(void *msg){
    if(msg == NULL)
        return;
    
    ComlibPrintf("%d:In recv_combined_array_msg\n", CkMyPe());

    MsgPacker::deliver((CombinedMessage *)msg);
}

ComlibManager::ComlibManager(){
    init();
    ComlibPrintf("In comlibmanager constructor\n");
}

void ComlibManager::init(){
    
    initComlibManager();

    if (CkMyRank() == 0) {
    PUPable_reg(CharmStrategy);
    PUPable_reg(CharmMessageHolder);
    }
    
    //comm_debug = 1;
    
    numStatsReceived = 0;
    curComlibController = 0;
    clibIteration = 0;
    
    strategyCreated = CmiFalse;

    CkpvInitialize(ClibLocationTableType*, locationTable);
    CkpvAccess(locationTable) = new CkHashtableT <ClibGlobalArrayIndex, int>;

    CkpvInitialize(CkArrayIndexMax, cache_index);
    CkpvInitialize(int, cache_pe);
    CkpvInitialize(CkArrayID, cache_aid);

    CkpvAccess(cache_index).nInts = -1;
    CkpvAccess(cache_aid).setZero();

    CkpvInitialize(int, RecvmsgHandle);
    CkpvAccess(RecvmsgHandle) =CkRegisterHandler((CmiHandler)recv_array_msg);

    bcast_pelist = new int [CkNumPes()];
    for(int brcount = 0; brcount < CkNumPes(); brcount++)
        bcast_pelist[brcount] = brcount;

    CkpvInitialize(int, RecvCombinedShortMsgHdlrIdx);
    CkpvAccess(RecvCombinedShortMsgHdlrIdx) = 
        CkRegisterHandler((CmiHandler)recv_combined_array_msg);
    
    section_send_event = traceRegisterUserEvent("ArraySectionMulticast");
    
    npes = CkNumPes();
    pelist = NULL;

    CkpvInitialize(CkGroupID, cmgrID);
    CkpvAccess(cmgrID) = thisgroup;

    dummyArrayIndex.nInts = 0;

    curStratID = 0;
    prevStratID = -1;
    //prioEndIterationFlag = 1;

    strategyTable = CkpvAccess(conv_com_ptr)->getStrategyTable();
    
    receivedTable = 0;
    setupComplete = 0;

    barrierReached = 0;
    barrier2Reached = 0;

    bcount = b2count = 0;
    lbUpdateReceived = CmiFalse;

    isRemote = 0;
    remotePe = -1;

    CkpvInitialize(int, migrationDoneHandlerID);
    CkpvAccess(migrationDoneHandlerID) = 
        CkRegisterHandler((CmiHandler) ComlibNotifyMigrationDoneHandler);
    
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    cgproxy[curComlibController].barrier();
}

//First barrier makes sure that the communication library group 
//has been created on all processors
void ComlibManager::barrier(){
    ComlibPrintf("In barrier %d\n", bcount);
    if(CkMyPe() == 0) {
        bcount ++;
        if(bcount == CkNumPes()){
            bcount = 0;
            barrierReached = 1;
            barrier2Reached = 0;

            if(strategyCreated)
                broadcastStrategies();
        }
    }
}

//Has finished passing the strategy list to all the processors
void ComlibManager::barrier2(){
    if(CkMyPe() == 0) {
        b2count ++;
        ComlibPrintf("In barrier2 %d\n", bcount);
        if(b2count == CkNumPes()) {
            b2count = 0; 
            CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
            cgproxy.resumeFromBarrier2();
        }
    }
}

//Registers a set of strategies with the communication library
ComlibInstanceHandle ComlibManager::createInstance() {
  
    CkpvAccess(conv_com_ptr)->nstrats++;    
    ComlibInstanceHandle cinst(CkpvAccess(conv_com_ptr)->nstrats -1,
                               CkpvAccess(cmgrID));  
    return cinst;
}

void ComlibManager::registerStrategy(int pos, CharmStrategy *strat) {
    
    strategyCreated = true;

    ListOfStrategies.enq(strat);
    strat->setInstance(pos);
}

//End of registering function, if barriers have been reached send them over
void ComlibManager::broadcastStrategies() {
    if(!barrierReached)
      return;    

    lbUpdateReceived = CmiFalse;
    barrierReached = 0;

    ComlibPrintf("Sending Strategies %d, %d\n", 
                 CkpvAccess(conv_com_ptr)->nstrats, 
                 ListOfStrategies.length());

    StrategyWrapper sw;
    sw.total_nstrats = CkpvAccess(conv_com_ptr)->nstrats;

    if(ListOfStrategies.length() > 0) {
        int len = ListOfStrategies.length();
        sw.s_table = new Strategy* [len];
        sw.nstrats = len;
        
        for (int count=0; count < len; count++)
            sw.s_table[count] = ListOfStrategies.deq();
    }
    else {
        sw.nstrats = 0;
        sw.s_table = 0;
    }

    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    cgproxy.receiveTable(sw, *CkpvAccess(locationTable));
}

//Called when the array/group element starts sending messages
void ComlibManager::beginIteration(){
    //right now does not do anything might need later
    ComlibPrintf("[%d]:In Begin Iteration %d\n", CkMyPe(), (* strategyTable)[0].elementCount);
    //prioEndIterationFlag = 0;
}

void ComlibManager::setInstance(int instID){

    curStratID = instID;
    ComlibPrintf("[%d]:In setInstance\n", CkMyPe(), (* strategyTable)[instID].elementCount);
}

//called when the array elements has finished sending messages
void ComlibManager::endIteration(){
    //    prioEndIterationFlag = 1;
    prevStratID = -1;
    
    ComlibPrintf("[%d]:In End Iteration(%d) %d, %d\n", CkMyPe(), 
                 curStratID, 
                 (* strategyTable)[curStratID].elementCount, 
                 (* strategyTable)[curStratID].numElements);

    if(isRemote) {
        isRemote = 0;
        sendRemote();
        return;
    }

    if(!receivedTable) {
        (* strategyTable)[curStratID].nEndItr++;
        return;
    }        
    
    (* strategyTable)[curStratID].elementCount++;
    int count = 0;
    
    if((* strategyTable)[curStratID].elementCount == (* strategyTable)[curStratID].numElements) {
        
        ComlibPrintf("[%d]:In End Iteration %d\n", CkMyPe(), (* strategyTable)[curStratID].elementCount);
        
        if(barrier2Reached) {	    
	    (* strategyTable)[curStratID].strategy->doneInserting();
        }
	else (* strategyTable)[curStratID].call_doneInserting = 1;
	
        (* strategyTable)[curStratID].elementCount = 0;
    }
    ComlibPrintf("After EndIteration\n");
}

//receive the list of strategies
//Insert the strategies into the strategy table in converse comm lib.
//CkpvAccess(conv_com_ptr) points to the converse commlib instance
void ComlibManager::receiveTable(StrategyWrapper &sw, 
                                 CkHashtableT<ClibGlobalArrayIndex, int> 
                                 &htable)
{

    ComlibPrintf("[%d] In receiveTable %d, ite=%d\n", CkMyPe(), sw.nstrats, 
                 clibIteration);

    receivedTable = 1;

    delete CkpvAccess(locationTable);
    //Delete all records in it too !!!!!!!!!!

    CkpvAccess(locationTable) =  NULL;

    CkpvAccess(locationTable) = new 
        CkHashtableT<ClibGlobalArrayIndex, int>;

    CkHashtableIterator *ht_iterator = htable.iterator();
    ht_iterator->seekStart();
    while(ht_iterator->hasNext()){
        ClibGlobalArrayIndex *idx;
        int *pe;
        pe = (int *)ht_iterator->next((void **)&idx);
        
        ComlibPrintf("[%d] HASH idx %d on %d\n", CkMyPe(), 
                     idx->idx.data()[0], *pe);

        CkpvAccess(locationTable)->put(*idx) = *pe;       
    }
    
    //Reset cached array element index. Location table may have changed
    CkpvAccess(cache_index).nInts = -1;
    CkpvAccess(cache_aid).setZero();

    CkArrayID st_aid;
    int st_nelements;
    CkArrayIndexMax *st_elem;
    int temp_curstratid = curStratID;

    CkpvAccess(conv_com_ptr)->nstrats = sw.total_nstrats;
    clib_stats.setNstrats(sw.total_nstrats);

    //First recreate strategies
    int count = 0;
    for(count = 0; count < sw.nstrats; count ++) {
        CharmStrategy *cur_strategy = (CharmStrategy *)sw.s_table[count];
        
        //set the instance to the current count
        //currently all strategies are being copied to all processors
        //later strategies will be selectively copied
        
        //location of this strategy table entry in the strategy table
        int loc = cur_strategy->getInstance();
        
        if(loc >= MAX_NUM_STRATS)
            CkAbort("Strategy table is full \n");

        CharmStrategy *old_strategy;

        //If this is a learning decision and the old strategy has to
        //be gotten rid of, finalize it here.
        if((old_strategy = 
            (CharmStrategy *)CkpvAccess(conv_com_ptr)->getStrategy(loc)) 
           != NULL) {
            old_strategy->finalizeProcessing();
            
            //Unregister from array listener if array strategy
            if(old_strategy->getType() == ARRAY_STRATEGY) {
                ComlibArrayInfo &as = ((CharmStrategy *)cur_strategy)->ainfo;
                as.getSourceArray(st_aid, st_elem, st_nelements);

                (* strategyTable)[loc].numElements = 0;
                if(!st_aid.isZero()) {
                    ComlibArrayListener *calistener = CkArrayID::
                        CkLocalBranch(st_aid)->getComlibArrayListener();
                    
                    calistener->unregisterStrategy(&((*strategyTable)[loc]));
                }
            }
        }
        
        //Insert strategy, frees an old strategy and sets the
        //strategy_table entry to point to the new one
        CkpvAccess(conv_com_ptr)->insertStrategy(cur_strategy, loc);
        
        ComlibPrintf("[%d] Inserting_strategy \n", CkMyPe());       

        if(cur_strategy->getType() == ARRAY_STRATEGY &&
           cur_strategy->isBracketed()){ 

            ComlibPrintf("Inserting Array Listener\n");
            
            ComlibArrayInfo &as = ((CharmStrategy *)cur_strategy)->ainfo;
            as.getSourceArray(st_aid, st_elem, st_nelements);
            
            (* strategyTable)[loc].numElements = 0;
            if(!st_aid.isZero()) {            
                ComlibArrayListener *calistener = 
                    CkArrayID::CkLocalBranch(st_aid)->getComlibArrayListener();
                
                calistener->registerStrategy(&((* strategyTable)[loc]));
            }
        }                      
        else { //if(cur_strategy->getType() == GROUP_STRATEGY){
	  (* strategyTable)[loc].numElements = 1;
        }
        
        (* strategyTable)[loc].elementCount = 0;
        cur_strategy->beginProcessing((* strategyTable)[loc].numElements); 
    }

    //Resume all end iterarions. Newer strategies may have more 
    //or fewer elements to expect for!!
    for(count = 0; count < CkpvAccess(conv_com_ptr)->nstrats; count++) {
        ComlibPrintf("[%d] endIteration from receiveTable %d, %d\n", 
                     CkMyPe(), count,
                     (* strategyTable)[count].nEndItr);
                         
        curStratID = count;
        for(int itr = 0; itr < (* strategyTable)[count].nEndItr; itr++) 
            endIteration();  
        
        (* strategyTable)[count].nEndItr = 0;        
    }           
    
    curStratID = temp_curstratid;
    ComlibPrintf("receivedTable %d\n", sw.nstrats);
    
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    cgproxy[curComlibController].barrier2();
}

void ComlibManager::resumeFromBarrier2(){

    if(!receivedTable) 
        //Application called atsync inbetween, receiveTable 
        //and resumeFromBarrier2. This will only happen when there
        //is no element on this processor and an element arrived, 
        //leading to a resumeFromSync and hence an AtSync in comlib.
        return; //A new resumeFromBarrier2 is on its way
    
    setupComplete = 1;

    barrier2Reached = 1;
    barrierReached = 0;

    clibIteration ++;

    ComlibPrintf("[%d] Barrier 2 reached nstrats = %d, ite = %d\n", 
                 CkMyPe(), CkpvAccess(conv_com_ptr)->nstrats, clibIteration);

    for (int count = 0; count < CkpvAccess(conv_com_ptr)->nstrats; 
         count ++) {
        if (!(* strategyTable)[count].tmplist.isEmpty()) {
            
            while (!(* strategyTable)[count].tmplist.isEmpty()) {
                CharmMessageHolder *cmsg = (CharmMessageHolder *) 
                    (* strategyTable)[count].tmplist.deq();
                
                if((*strategyTable)[count].strategy->getType() == 
                   ARRAY_STRATEGY) {
                    if(cmsg->dest_proc >= 0) {
                        envelope *env  = UsrToEnv(cmsg->getCharmMessage());
                        cmsg->dest_proc = getLastKnown
                            (env->getsetArrayMgr(),
                             env->getsetArrayIndex());
                    }
                    //else multicast
                }                                
                
                if(cmsg->dest_proc == CkMyPe()) {
                    CmiSyncSendAndFree(CkMyPe(), cmsg->size,
                                       (char *)
                                       UsrToEnv(cmsg->getCharmMessage()));
                    delete cmsg;
                }
                else
                    (* strategyTable)[count].strategy->insertMessage(cmsg);
            }
        }
        
        if ((* strategyTable)[count].call_doneInserting) {
            (* strategyTable)[count].call_doneInserting = 0;
            ComlibPrintf("[%d] Calling done inserting \n", CkMyPe());
            (* strategyTable)[count].strategy->doneInserting();
        }
    }
    
    ComlibPrintf("[%d] After Barrier2\n", CkMyPe());
}

extern int _charmHandlerIdx;

void ComlibManager::ArraySend(CkDelegateData *pd,int ep, void *msg, 
                              const CkArrayIndexMax &idx, CkArrayID a){
    
    if(pd != NULL) {
        ComlibInstanceHandle *ci = (ComlibInstanceHandle *)pd;
        setInstance(ci->_instid);
    }
    
    ComlibPrintf("[%d] In Array Send\n", CkMyPe());
    
    CkArrayIndexMax myidx = idx;
    int dest_proc = getLastKnown(a, myidx); 
    
    //Reading from two hash tables is a big overhead
    //int amgr_destpe = CkArrayID::CkLocalBranch(a)->lastKnown(myidx);
    
    register envelope * env = UsrToEnv(msg);
    
    env->getsetArrayMgr()=a;
    env->getsetArraySrcPe()=CkMyPe();
    env->getsetArrayEp()=ep;
    env->getsetArrayHops()=0;
    env->getsetArrayIndex()=idx;
    env->setUsed(0);
    ((CmiMsgHeaderExt *)env)->stratid = curStratID;

    //RECORD_SEND_STATS(curStratID, env->getTotalsize(), dest_proc);

    if(isRemote) {
        CkPackMessage(&env);        
        CharmMessageHolder *cmsg = new 
            CharmMessageHolder((char *)msg, dest_proc);
        
        remoteQ.enq(cmsg);
        return;
    }
    
    //With migration some array messages may be directly sent Also no
    //message processing should happen before the comlib barriers have
    //gone through
    if(dest_proc == CkMyPe() && setupComplete){  
        CkArray *amgr = (CkArray *)_localBranch(a);
        amgr->deliver((CkArrayMessage *)msg, CkDeliver_queue);
        
        return;
    }

    CkPackMessage(&env);
    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));        
    
    //get rid of the new.
    CharmMessageHolder *cmsg = new 
        CharmMessageHolder((char *)msg, dest_proc);
    
    ComlibPrintf("[%d] Before Insert on strat %d received = %d\n", CkMyPe(), curStratID, setupComplete);
    
    if (setupComplete)        
        (* strategyTable)[curStratID].strategy->insertMessage(cmsg);
    else 
        (* strategyTable)[curStratID].tmplist.enq(cmsg);
    
    //CmiPrintf("After Insert\n");
}


#include "qd.h"
//CkpvExtern(QdState*, _qd);

void ComlibManager::GroupSend(CkDelegateData *pd,int ep, void *msg, int onPE, CkGroupID gid){
    

    if(pd != NULL) {
        ComlibInstanceHandle *ci = (ComlibInstanceHandle *)pd;
        setInstance(ci->_instid);
    }

    int dest_proc = onPE;

    ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, 
                 UsrToEnv(msg)->getTotalsize());

    register envelope * env = UsrToEnv(msg);
    if(dest_proc == CkMyPe() && setupComplete){
        _SET_USED(env, 0);
        CkSendMsgBranch(ep, msg, dest_proc, gid);
        return;
    }
    
    ((CmiMsgHeaderExt *)env)->stratid = curStratID;
    CpvAccess(_qd)->create(1);

    env->setMsgtype(ForBocMsg);
    env->setEpIdx(ep);
    env->setGroupNum(gid);
    env->setSrcPe(CkMyPe());
    env->setUsed(0);

    CkPackMessage(&env);
    CmiSetHandler(env, _charmHandlerIdx);

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.
    
    if(setupComplete)
        (* strategyTable)[curStratID].strategy->insertMessage(cmsg);
    else {
        (* strategyTable)[curStratID].tmplist.enq(cmsg);
    }
}

void ComlibManager::ArrayBroadcast(CkDelegateData *pd,int ep,void *m,CkArrayID a){
    ComlibPrintf("[%d] Array Broadcast \n", CkMyPe());

    if(pd != NULL) {
        ComlibInstanceHandle *ci = (ComlibInstanceHandle *)pd;
        setInstance(ci->_instid);
    }
    
    //Broken, add the processor list here.

    register envelope * env = UsrToEnv(m);
    env->getsetArrayMgr()=a;
    env->getsetArraySrcPe()=CkMyPe();
    env->getsetArrayEp()=ep;
    env->getsetArrayHops()=0;
    env->getsetArrayIndex()= dummyArrayIndex;
    ((CmiMsgHeaderExt *)env)->stratid = curStratID;

    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));
    
    //RECORD_SENDM_STATS(curStratID, env->getTotalsize(), dest_proc);

    CharmMessageHolder *cmsg = new 
        CharmMessageHolder((char *)m, IS_BROADCAST);
    cmsg->npes = 0;
    cmsg->pelist = NULL;
    cmsg->sec_id = NULL;

    multicast(cmsg);
}

void ComlibManager::ArraySectionSend(CkDelegateData *pd,int ep, void *m, 
                                     CkArrayID a, CkSectionID &s, int opts) {

#ifndef CMK_OPTIMIZE
    traceUserEvent(section_send_event);
#endif

    if(pd != NULL) {
        ComlibInstanceHandle *ci = (ComlibInstanceHandle *)pd;
        setInstance(ci->_instid);
    }
    
    ComlibPrintf("[%d] Array Section Send \n", CkMyPe());

    register envelope * env = UsrToEnv(m);
    env->getsetArrayMgr()=a;
    env->getsetArraySrcPe()=CkMyPe();
    env->getsetArrayEp()=ep;
    env->getsetArrayHops()=0;
    env->getsetArrayIndex()= dummyArrayIndex;
    ((CmiMsgHeaderExt *)env)->stratid = curStratID;

    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));
    
    env->setUsed(0);    
    CkPackMessage(&env);
    
    //Provide a dummy dest proc as it does not matter for mulitcast 
    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)m,
                                                      IS_SECTION_MULTICAST);
    cmsg->npes = 0;
    cmsg->sec_id = &s;

    CkSectionInfo minfo;
    minfo.type = COMLIB_MULTICAST_MESSAGE;
    minfo.sInfo.cInfo.instId = curStratID;
    minfo.sInfo.cInfo.status = COMLIB_MULTICAST_ALL;  
    minfo.sInfo.cInfo.id = 0; 
    minfo.pe = CkMyPe();
    ((CkMcastBaseMsg *)m)->_cookie = minfo;    
    //    s.npes = 0;
    //s.pelist = NULL;

    multicast(cmsg);
}

void ComlibManager::GroupBroadcast(CkDelegateData *pd,int ep,void *m,CkGroupID g) {
    
    if(pd != NULL) {
        ComlibInstanceHandle *ci = (ComlibInstanceHandle *)pd;
        setInstance(ci->_instid);
    }
    
    register envelope * env = UsrToEnv(m);
    
    CpvAccess(_qd)->create(1);

    env->setMsgtype(ForBocMsg);
    env->setEpIdx(ep);
    env->setGroupNum(g);
    env->setSrcPe(CkMyPe());
    env->setUsed(0);
    ((CmiMsgHeaderExt *)env)->stratid = curStratID;

    CkPackMessage(&env);
    CmiSetHandler(env, _charmHandlerIdx);
    
    //Provide a dummy dest proc as it does not matter for mulitcast 
    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)m,IS_BROADCAST);
    
    cmsg->npes = 0;
    cmsg->pelist = NULL;

    multicast(cmsg);
}

void ComlibManager::multicast(CharmMessageHolder *cmsg) {

    register envelope * env = UsrToEnv(cmsg->getCharmMessage());    
    ComlibPrintf("[%d]: In multicast\n", CkMyPe());

    env->setUsed(0);    
    CkPackMessage(&env);

    //Will be used to detect multicast message for learning
    
    if (setupComplete)
	(* strategyTable)[curStratID].strategy->insertMessage(cmsg);
    else {
	ComlibPrintf("Enqueuing message in tmplist at %d\n", curStratID);
        (* strategyTable)[curStratID].tmplist.enq(cmsg);
    }

    ComlibPrintf("After multicast\n");
}

//Collect statistics from all the processors, also gets the list of
//array elements on each processor.
void ComlibManager::collectStats(ComlibLocalStats &stat, int pe, 
                                 CkVec<ClibGlobalArrayIndex> &idx_vec) {
    
    ComlibPrintf("%d: Collecting stats %d\n", CkMyPe(), numStatsReceived);

    numStatsReceived ++;
    clib_gstats.updateStats(stat, pe);
    
    for(int count = 0; count < idx_vec.length(); count++) {
        int old_pe = CkpvAccess(locationTable)->get(idx_vec[count]);
        
        ComlibPrintf("Adding idx %d to %d\n", idx_vec[count].idx.data()[0], 
                     pe);
        
        CkpvAccess(locationTable)->put(idx_vec[count]) = pe + CkNumPes();
    }        

    //Reset cached array element index. Location table may have changed
    CkpvAccess(cache_index).nInts = -1;
    CkpvAccess(cache_aid).setZero();

    if(numStatsReceived == CkNumPes()) {
        numStatsReceived = 0;

        for(int count = 0; count < CkpvAccess(conv_com_ptr)->nstrats; 
            count++ ){
            Strategy* strat = CkpvAccess(conv_com_ptr)->getStrategy(count);
            if(strat->getType() > CONVERSE_STRATEGY) {
                CharmStrategy *cstrat = (CharmStrategy *)strat;
                ComlibLearner *learner = cstrat->getLearner();
                CharmStrategy *newstrat = NULL;
                                
                if(learner != NULL) {
                    ComlibPrintf("Calling Learner\n");
                    newstrat = (CharmStrategy *)learner->optimizePattern
                        (strat, clib_gstats);
                    
                    if(newstrat != NULL)
                        ListOfStrategies.enq(newstrat);
                    else
                        ListOfStrategies.enq(cstrat);
                }
            }
        }
        barrierReached = 1;
        
        //if(lbUpdateReceived) {
        //lbUpdateReceived = CmiFalse;
        broadcastStrategies();
        //}
    }
}

void ComlibManager::setRemote(int remote_pe){

    ComlibPrintf("Setting remote flag on\n");

    remotePe = remote_pe;
    isRemote = 1;
}


void ComlibManager::receiveRemoteSend(CkQ<CharmMessageHolder *> &rq, 
                                      int strat_id) {
    setInstance(strat_id);
    
    int nmsgs = rq.length();

    ComlibPrintf("%d: Receiving remote message\n", CkMyPe());

    for(int count = 0; count < nmsgs; count++) {
        char *msg = rq.deq()->getCharmMessage();
        envelope *env = UsrToEnv(msg);
        
        ArraySend(NULL, env->getsetArrayEp(), msg, env->getsetArrayIndex(), 
                  env->getsetArrayMgr());
    }

    endIteration();
}

void ComlibManager::sendRemote(){
    
    int nmsgs = remoteQ.length();

    //if(nmsgs == 0)
    //  return;

    ComlibPrintf("%d: Sending remote message \n", CkMyPe());

    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID)); 
    cgproxy[remotePe].receiveRemoteSend(remoteQ, curStratID);
    
    for(int count = 0; count < nmsgs; count++) {
        CharmMessageHolder *cmsg = remoteQ.deq();
        CkFreeMsg(cmsg->getCharmMessage());
        delete cmsg;
    }
}


void ComlibManager::AtSync() {

    //comm_debug = 1;
    ComlibPrintf("[%d] In ComlibManager::Atsync, controller %d, ite %d\n", CkMyPe(), curComlibController, clibIteration);

    barrier2Reached = 0;
    receivedTable = 0;
    setupComplete = 0;
    barrierReached = 0;
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));

    int pos = 0;

    CkVec<ClibGlobalArrayIndex> gidx_vec;

    CkVec<CkArrayID> tmp_vec;
    for(int count = 0; count < CkpvAccess(conv_com_ptr)->nstrats; 
        count ++) {
        if((* strategyTable)[count].strategy->getType() 
           == ARRAY_STRATEGY) {
            CharmStrategy *cstrat = (CharmStrategy*)
                ((* strategyTable)[count].strategy);
            
            CkArrayID src, dest;
            CkArrayIndexMax *elements;
            int nelem;
            
            cstrat->ainfo.getSourceArray(src, elements, nelem);
            cstrat->ainfo.getDestinationArray(dest, elements, nelem);

            CmiBool srcflag = CmiFalse;
            CmiBool destflag = CmiFalse;
            
            if(src == dest || dest.isZero())
                destflag = CmiTrue;

            if(src.isZero())
                srcflag = CmiTrue;                        

            for(pos = 0; pos < tmp_vec.size(); pos++) {
                if(tmp_vec[pos] == src)
                    srcflag = CmiTrue;

                if(tmp_vec[pos] == dest)
                    destflag = CmiTrue;

                if(srcflag && destflag)
                    break;
            }

            if(!srcflag)
                tmp_vec.insertAtEnd(src);

            if(!destflag)
                tmp_vec.insertAtEnd(dest);
        }
        
        //cant do it here, done in receiveTable
        //if((* strategyTable)[count].strategy->getType() > CONVERSE_STRATEGY)
        //  (* strategyTable)[count].reset();
    }

    for(pos = 0; pos < tmp_vec.size(); pos++) {
        CkArrayID aid = tmp_vec[pos];

        ComlibArrayListener *calistener = 
            CkArrayID::CkLocalBranch(aid)->getComlibArrayListener();

        CkVec<CkArrayIndexMax> idx_vec;
        calistener->getLocalIndices(idx_vec);

        for(int idx_count = 0; idx_count < idx_vec.size(); idx_count++) {
            ClibGlobalArrayIndex gindex;
            gindex.aid = aid;
            gindex.idx = idx_vec[idx_count];

            gidx_vec.insertAtEnd(gindex);
        }
    }

    cgproxy[curComlibController].collectStats(clib_stats, CkMyPe(), gidx_vec);
    clib_stats.reset();
}

#include "lbdb.h"
#include "CentralLB.h"

/******** FOO BAR : NEEDS to be consistent with array manager *******/
void LDObjID2IdxMax (LDObjid ld_id, CkArrayIndexMax &idx) {
    if(OBJ_ID_SZ < CK_ARRAYINDEX_MAXLEN)
        CkAbort("LDB OBJ ID smaller than array index\n");
    
    //values higher than CkArrayIndexMax should be 0
    for(int count = 0; count < CK_ARRAYINDEX_MAXLEN; count ++) {
        idx.data()[count] = ld_id.id[count];
    }
    idx.nInts = 1;
}

void ComlibManager::lbUpdate(LBMigrateMsg *msg) {
    for(int count = 0; count < msg->n_moves; count ++) {
        MigrateInfo m = msg->moves[count];

        CkArrayID aid; CkArrayIndexMax idx;
        aid = CkArrayID(m.obj.omhandle.id.id);
        LDObjID2IdxMax(m.obj.id, idx);

        ClibGlobalArrayIndex cid; 
        cid.aid = aid;
        cid.idx = idx;
        
        int pe = CkpvAccess(locationTable)->get(cid);

        //Value exists in the table, so update it
        if(pe != 0) {
            pe = m.to_pe + CkNumPes();
            CkpvAccess(locationTable)->getRef(cid) = pe;
        }
        //otherwise we dont care about these objects
    }   

    lbUpdateReceived = CmiTrue;
    if(barrierReached) {
        broadcastStrategies();
        barrierReached = 0;
    }

    CkFreeMsg(msg);
}

CkDelegateData* ComlibManager::ckCopyDelegateData(CkDelegateData *data) {
    ComlibInstanceHandle *inst = new ComlibInstanceHandle
        (*((ComlibInstanceHandle *)data));
    return inst;
}


CkDelegateData * ComlibManager::DelegatePointerPup(PUP::er &p,
                                                   CkDelegateData *pd) {

    if(pd == NULL)
        return NULL;

    CmiBool to_pup = CmiFalse;

    ComlibInstanceHandle *inst; 
    if(!p.isUnpacking()) {
        inst = (ComlibInstanceHandle *) pd;       
        if(pd != NULL)
            to_pup = CmiTrue;
    }
    else 
        //Call migrate constructor
        inst = new ComlibInstanceHandle();

    p | to_pup;

    if(to_pup)
        inst->pup(p);
    return inst;
}    

void ComlibDelegateProxy(CProxy *proxy){
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    proxy->ckDelegate(cgproxy.ckLocalBranch(), NULL);
}

void ComlibAssociateProxy(ComlibInstanceHandle *cinst, CProxy &proxy) {
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    proxy.ckDelegate(cgproxy.ckLocalBranch(), cinst);
}

void ComlibAssociateProxy(CharmStrategy *strat, CProxy &proxy) {
    ComlibInstanceHandle *cinst = new ComlibInstanceHandle
        (CkGetComlibInstance());

    cinst->setStrategy(strat);
    
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    proxy.ckDelegate(cgproxy.ckLocalBranch(), cinst);
} 

ComlibInstanceHandle ComlibRegister(CharmStrategy *strat) {
    ComlibInstanceHandle cinst(CkGetComlibInstance());
    cinst.setStrategy(strat);
    return cinst;
}

void ComlibBegin(CProxy &proxy) {
    ComlibInstanceHandle *cinst = (ComlibInstanceHandle *)proxy.ckDelegatedPtr();
    cinst->beginIteration();
} 

void ComlibEnd(CProxy &proxy) {
    ComlibInstanceHandle *cinst = (ComlibInstanceHandle *)proxy.ckDelegatedPtr();
    cinst->endIteration();
}

ComlibInstanceHandle CkCreateComlibInstance(){
    return CkGetComlibInstance();
}

ComlibInstanceHandle CkGetComlibInstance() {
    if(CkMyPe() != 0)
        CkAbort("Comlib Instance can only be created on Processor 0");
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));    
    return (cgproxy.ckLocalBranch())->createInstance();
}

ComlibInstanceHandle CkGetComlibInstance(int id) {
    ComlibInstanceHandle cinst(id, CkpvAccess(cmgrID));
    return cinst;
}

void ComlibDoneCreating(){
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    (cgproxy.ckLocalBranch())->broadcastStrategies();
}

char *router;
int sfactor=0;

class ComlibManagerMain {
public:
    ComlibManagerMain(CkArgMsg *msg) {
        
        if(CkMyPe() == 0 && msg !=  NULL)
            CmiGetArgString(msg->argv, "+strategy", &router);         

        if(CkMyPe() == 0 && msg !=  NULL)
            CmiGetArgInt(msg->argv, "+spanning_factor", &sfactor);
        
        CProxy_ComlibManager::ckNew();
    }
};

//Called by user code
ComlibInstanceHandle::ComlibInstanceHandle() : CkDelegateData() {
    _instid = -1;
    _dmid.setZero();
    _srcPe = -1;
    toForward = 0;
}

//Called by user code
ComlibInstanceHandle::ComlibInstanceHandle(const ComlibInstanceHandle &h) 
    : CkDelegateData() {
    _instid = h._instid;
    _dmid = h._dmid;
    toForward = h.toForward;        

    ComlibPrintf("In Copy Constructor\n");
    _srcPe = h._srcPe;

    reset();
    ref();
}

ComlibInstanceHandle& ComlibInstanceHandle::operator=(const ComlibInstanceHandle &h) {
    _instid = h._instid;
    _dmid = h._dmid;
    toForward = h.toForward;
    _srcPe = h._srcPe;
    
    reset();
    ref();
    return *this;
}

//Called by the communication library
ComlibInstanceHandle::ComlibInstanceHandle(int instid, CkGroupID dmid){
    _instid = instid;
    _dmid   = dmid;
    _srcPe  = -1;
    toForward = 0;
}

void ComlibInstanceHandle::beginIteration() { 
    CProxy_ComlibManager cgproxy(_dmid);

    ComlibPrintf("Instance Handle beginIteration %d, %d, %d\n", CkMyPe(), _srcPe, _instid);

    //User forgot to make the instance handle a readonly or pass it
    //into the constructor of an array and is using it directly from
    //Main :: main
    if(_srcPe == -1) {
        //ComlibPrintf("Warning:Instance Handle needs to be a readonly or a private variable of an array element\n");
        _srcPe = CkMyPe();
    }

    if(_srcPe != CkMyPe() && toForward) {
        (cgproxy.ckLocalBranch())->setRemote(_srcPe);
    }

    (cgproxy.ckLocalBranch())->setInstance(_instid);
    (cgproxy.ckLocalBranch())->beginIteration();   
}

void ComlibInstanceHandle::endIteration() {
    CProxy_ComlibManager cgproxy(_dmid);
    (cgproxy.ckLocalBranch())->endIteration();
}

void ComlibInstanceHandle::setStrategy(CharmStrategy *s) {
    toForward = s->getForwardOnMigration();
    CProxy_ComlibManager cgproxy(_dmid);
    (cgproxy.ckLocalBranch())->registerStrategy(_instid, s);
}

CharmStrategy *ComlibInstanceHandle::getStrategy() {
    if(_instid < 0) 
        return NULL;    
    
    CProxy_ComlibManager cgproxy(_dmid);
    return (cgproxy.ckLocalBranch())->getStrategy(_instid);
}

CkGroupID ComlibInstanceHandle::getComlibManagerID() {return _dmid;}    

void ComlibInitSectionID(CkSectionID &sid){
    
    sid._cookie.type = COMLIB_MULTICAST_MESSAGE;
    sid._cookie.pe = CkMyPe();
    
    sid._cookie.sInfo.cInfo.id = 0;    
    sid.npes = 0;
    sid.pelist = NULL;
}

void ComlibResetSectionProxy(CProxySection_ArrayBase *sproxy) {
    CkSectionID &sid = sproxy->ckGetSectionID();
    ComlibInitSectionID(sid);
    sid._cookie.sInfo.cInfo.status = 0;
}

// for backward compatibility - for old name commlib
void _registercommlib(void)
{
  static int _done = 0; 
  if(_done) 
      return; 
  _done = 1;
  _registercomlib();
}

void ComlibAtSyncHandler(void *msg) {
    CmiFree(msg);
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    ComlibManager *cmgr_ptr = cgproxy.ckLocalBranch();
    if(cmgr_ptr)
        cmgr_ptr->AtSync();    
}

void ComlibNotifyMigrationDoneHandler(void *msg) {
    CmiFree(msg);
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    ComlibManager *cmgr_ptr = cgproxy.ckLocalBranch();
    if(cmgr_ptr)
        cmgr_ptr->AtSync();    
}


void ComlibLBMigrationUpdate(LBMigrateMsg *msg) {
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    (cgproxy.ckLocalBranch())->lbUpdate(msg);
}

#include "comlib.def.h"

/*@}*/
