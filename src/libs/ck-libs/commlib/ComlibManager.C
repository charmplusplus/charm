#include "ComlibManager.h"

#include "EachToManyMulticastStrategy.h"
#include "DirectMulticastStrategy.h"
#include "StreamingStrategy.h"
#include "DummyStrategy.h"
#include "MPIStrategy.h"
#include "NodeMulticast.h"
#include "MsgPacker.h"
#include "RingMulticastStrategy.h"
#include "PipeBroadcastStrategy.h"
#include "BroadcastStrategy.h"
#include "MeshStreamingStrategy.h"
#include "PrioStreaming.h"

CkpvDeclare(int, RecvmsgHandle);
CkpvDeclare(int, RecvCombinedShortMsgHdlrIdx);
CkpvDeclare(int, RecvdummyHandle);
CkpvDeclare(CkGroupID, cmgrID);

//handler to receive array messages
void recv_msg(void *msg){

    if(msg == NULL)
        return;
    
    ComlibPrintf("%d:In recv_msg\n", CkMyPe());

    register envelope* env = (envelope *)msg;
    env->setUsed(0);
    env->getsetArrayHops()=1;
    CkUnpackMessage(&env);
    
    /*
    CProxyElement_ArrayBase ap(env->getsetArrayMgr(), env->getsetArrayIndex());
    ComlibPrintf("%d:Array Base created\n", CkMyPe());
    ap.ckSend((CkArrayMessage *)EnvToUsr(env), env->getsetArrayEp());
    */
    
    CkArray *a=(CkArray *)_localBranch(env->getsetArrayMgr());
    a->deliver((CkArrayMessage *)EnvToUsr(env), CkDeliver_queue, CmiTrue);    

    ComlibPrintf("%d:Out of recv_msg\n", CkMyPe());
    return;
}

void recv_combined_array_msg(void *msg){
    if(msg == NULL)
        return;
    
    ComlibPrintf("%d:In recv_combined_array_msg\n", CkMyPe());

    MsgPacker::deliver((CombinedMessage *)msg);
}

//handler for dummy messages
void recv_dummy(void *msg){
    ComlibPrintf("Received Dummy %d\n", CkMyPe());    
    CmiFree(msg);
}

//An initialization routine which does prelimnary initialization of the 
void initComlibManager(void){
    //comm_debug = 1;
    ComlibInit();
    ComlibPrintf("Init Call\n");

    CkpvInitialize(int, RecvmsgHandle);
    CkpvInitialize(int, RecvCombinedShortMsgHdlrIdx);

    CkpvAccess(RecvmsgHandle) = CkRegisterHandler((CmiHandler)recv_msg);
    CkpvAccess(RecvCombinedShortMsgHdlrIdx) = 
        CkRegisterHandler((CmiHandler)recv_combined_array_msg);
    
    //CmiPrintf("[%d] Registering Handler %d\n", CmiMyPe(), CkpvAccess(RecvmsgHandle));
    
    CkpvInitialize(int, RecvdummyHandle);
    CkpvAccess(RecvdummyHandle) = CkRegisterHandler((CmiHandler)recv_dummy);
    //ComlibPrintf("After Init Call\n");
}

ComlibManager::ComlibManager(){
    init();
    ComlibPrintf("In comlibmanager constructor\n");
}

void ComlibManager::init(){

    //comm_debug = 1;
  
    section_send_event = traceRegisterUserEvent("ArraySectionMulticast");
  
    npes = CkNumPes();
    pelist = NULL;
    nstrats = 0;

    CkpvInitialize(CkGroupID, cmgrID);
    CkpvAccess(cmgrID) = thisgroup;

    dummyArrayIndex.nInts = 0;

    curStratID = 0;
    prevStratID = -1;
    //prioEndIterationFlag = 1;

    //initialize the strategy table.
    //    bzero(strategyTable, MAX_NSTRAT * sizeof(StrategyTable));
    for(int count = 0; count < MAX_NSTRAT; count ++) {
      strategyTable[count].strategy = NULL;
      strategyTable[count].numElements = 0;
      strategyTable[count].elementCount = 0;
      strategyTable[count].call_doneInserting = 0;
      strategyTable[count].nEndItr = 0;
    }
        
    //procMap = new int[CkNumPes()];
    //for(int count = 0; count < CkNumPes(); count++)
    //  procMap[count] = count;
    
    receivedTable = 0;
    flushTable = 0;
    totalMsgCount = 0;
    totalBytes = 0;
    nIterations = 0;
    barrierReached = 0;
    barrier2Reached = 0;

    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    cgproxy[0].barrier();
}

//First barrier makes sure that the communication library group 
//has been created on all processors
void ComlibManager::barrier(){
  static int bcount = 0;
  ComlibPrintf("In barrier %d\n", bcount);
  if(CkMyPe() == 0) {
    bcount ++;
    if(bcount == CkNumPes()){
      barrierReached = 1;
      doneCreating();
    }
  }
}

//Has finished passing the strategy list to all the processors
void ComlibManager::barrier2(){
  static int bcount = 0;
  if(CkMyPe() == 0) {
    bcount ++;
    ComlibPrintf("In barrier2 %d\n", bcount);
    if(bcount == CkNumPes()) {
        CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
        cgproxy.resumeFromBarrier2();
    }
  }
}

//Registers a set of strategies with the communication library
ComlibInstanceHandle ComlibManager::createInstance() {
  
    ListOfStrategies.insertAtEnd(NULL);
    nstrats++;
    
    ComlibInstanceHandle cinst(nstrats - 1, CkpvAccess(cmgrID));  
    return cinst;
}

void ComlibManager::registerStrategy(int pos, Strategy *strat) {
    ListOfStrategies[pos] = strat;
}

//End of registering function, if barriers have been reached send them over
void ComlibManager::doneCreating() {
    if(!barrierReached)
      return;    

    ComlibPrintf("Sending Strategies %d, %d\n", nstrats, 
                 ListOfStrategies.length());

    if(nstrats == 0)
        return;

    StrategyWrapper sw;
    sw.s_table = new Strategy* [nstrats];
    sw.nstrats = nstrats;
    
    for (int count=0; count<nstrats; count++)
        sw.s_table[count] = ListOfStrategies[count];

    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    cgproxy.receiveTable(sw);
}

//Registration now done through array listeners
/*
void ComlibManager::localElement(){
    ComlibPrintf("In Local Element\n");
    strategyTable[0].numElements ++;
}
*/
/*
void ComlibManager::registerElement(int stratID){
    ComlibPrintf("In Register Element\n");
    strategyTable[stratID].numElements ++;
}

void ComlibManager::unRegisterElement(int stratID){
    ComlibPrintf("In Un Register Element\n");
    strategyTable[stratID].numElements --;
}
*/

//Called when the array/group element starts sending messages
void ComlibManager::beginIteration(){
    //right now does not do anything might need later
    ComlibPrintf("[%d]:In Begin Iteration %d\n", CkMyPe(), strategyTable[0].elementCount);
    //prioEndIterationFlag = 0;
}

void ComlibManager::setInstance(int instID){

    curStratID = instID;
    ComlibPrintf("[%d]:In setInstance\n", CkMyPe(), strategyTable[instID].elementCount);
}

//called when the array elements has finished sending messages
void ComlibManager::endIteration(){
    //    prioEndIterationFlag = 1;
    prevStratID = -1;

    if(!receivedTable) {
        strategyTable[curStratID].nEndItr++;
        return;
    }        
    
    ComlibPrintf("[%d]:In End Iteration(%d) %d, %d\n", CkMyPe(), curStratID, 
                 strategyTable[curStratID].elementCount, 
                 strategyTable[curStratID].numElements);
  
    strategyTable[curStratID].elementCount++;
    int count = 0;
    flushTable = 1;

    if(strategyTable[curStratID].elementCount == strategyTable[curStratID].numElements) {
        
        ComlibPrintf("[%d]:In End Iteration %d\n", CkMyPe(), 
                     strategyTable[curStratID].elementCount);
        
        nIterations ++;

        if(nIterations == LEARNING_PERIOD) {
            //CkPrintf("Sending %d, %d\n", totalMsgCount, totalBytes);
            CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
            cgproxy[0].learnPattern(totalMsgCount, totalBytes);
        }
        
        if(barrier2Reached) {	    
	    strategyTable[curStratID].strategy->doneInserting();
        }
	else strategyTable[curStratID].call_doneInserting = 1;
	
        strategyTable[curStratID].elementCount = 0;
    }
    ComlibPrintf("After EndIteration\n");
}

//receive the list of strategies
void ComlibManager::receiveTable(StrategyWrapper sw){
    
    ComlibPrintf("[%d] In receiveTable\n", CkMyPe());

    receivedTable = 1;
    nstrats = sw.nstrats;

    CkArrayID st_aid;
    int st_nelements;
    CkArrayIndexMax *st_elem;

    int count = 0;
    for(count = 0; count < nstrats; count ++) {
        strategyTable[count].strategy = sw.s_table[count];
        strategyTable[count].strategy->setInstance(count);  
        
        if(strategyTable[count].strategy->isSourceArray()){           
            strategyTable[count].strategy->
                getSourceArray(st_aid, st_elem, st_nelements);
            
            //CkPrintf("[%d] Calling Array listener for array\n", CkMyPe());
            ComlibArrayListener *calistener = 
                CkArrayID::CkLocalBranch(st_aid)->getComlibArrayListener();
            
            calistener->registerStrategy(&strategyTable[count]);

            curStratID = count;
            
            ComlibPrintf("[%d] endIteration from receiveTable %d\n", 
                         CkMyPe(), strategyTable[count].nEndItr);
            
            for(int itr = 0; itr < strategyTable[count].nEndItr; itr++) 
                endIteration();            
        }              
  
        if(strategyTable[count].strategy->isSourceGroup()){
            strategyTable[count].numElements = 1;

            for(int itr = 0; itr < strategyTable[count].nEndItr; itr++) 
                endIteration();            
        }
        
        strategyTable[count].strategy->beginProcessing
            (strategyTable[count].numElements);
    }           
    
    ComlibPrintf("receivedTable %d\n", nstrats);
    
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    cgproxy[0].barrier2();
}

void ComlibManager::resumeFromBarrier2(){
    barrier2Reached = 1;
    
    ComlibPrintf("[%d] Barrier 2 reached\n", CkMyPe());

    if(flushTable) {
      for (int count = 0; count < nstrats; count ++) {
          if (!strategyTable[count].tmplist.isEmpty()) {
              CharmMessageHolder *cptr;
              while (!strategyTable[count].tmplist.isEmpty())
                  strategyTable[count].strategy->insertMessage
                      (strategyTable[count].tmplist.deq());
          }
          
          if (strategyTable[count].call_doneInserting)
              strategyTable[count].strategy->doneInserting();
      }
    }
    
    ComlibPrintf("[%d] After Barrier2\n", CkMyPe());
}

extern int _charmHandlerIdx;
//extern int _infoIdx;
//#include "cldb.h"

void ComlibManager::ArraySend(int ep, void *msg, 
                              const CkArrayIndexMax &idx, CkArrayID a){
    
    ComlibPrintf("[%d] In Array Send\n", CkMyPe());
    /*
    if(curStratID != prevStratID && prioEndIterationFlag) {        
        CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
        ComlibPrintf("[%d] Array Send calling prio end iteration\n", 
                     CkMyPe());
        PrioMsg *pmsg = new(8 * sizeof(int)) PrioMsg();
        int mprio = -100;
        *(int *)CkPriorityPtr(pmsg) = mprio;
        pmsg->instID = curStratID;
        CkSetQueueing(pmsg, CK_QUEUEING_BFIFO);
        cgproxy[CkMyPe()].prioEndIteration(pmsg);
        prioEndIterationFlag = 0;
    }        
    prevStratID = curStratID;            
    */

    CkArrayIndexMax myidx = idx;
    int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(myidx);
    
    //ComlibPrintf("Send Data %d %d %d %d\n", CkMyPe(), dest_proc, 
    //	 UsrToEnv(msg)->getTotalsize(), receivedTable);

    if(dest_proc == CkMyPe()){
        //CkArrayID::CkLocalBranch(a)->deliverViaQueue((CkArrayMessage *)msg);

        CProxyElement_ArrayBase ap(a,idx);
        ap.ckSend((CkArrayMessage *)msg, ep);
        return;
    }

    register envelope * env = UsrToEnv(msg);
    
    env->getsetArrayMgr()=a;
    env->getsetArraySrcPe()=CkMyPe();
    env->getsetArrayEp()=ep;
    env->getsetArrayHops()=0;
    env->getsetArrayIndex()=idx;
    env->setUsed(0);
    
    CkPackMessage(&env);
    //    CmiSetHandler(env, _charmHandlerIdx);
    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));
    
    totalMsgCount ++;
    totalBytes += UsrToEnv(msg)->getTotalsize();

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc);
    //get rid of the new.

    ComlibPrintf("Before Insert\n");

    if (receivedTable)
      strategyTable[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
        cmsg->next = NULL;
        strategyTable[curStratID].tmplist.enq(cmsg);
    }

    //CmiPrintf("After Insert\n");
}


#include "qd.h"
//CpvExtern(QdState*, _qd);

void ComlibManager::GroupSend(int ep, void *msg, int onPE, CkGroupID gid){
    
    int dest_proc = onPE;
    /*
    if(curStratID != prevStratID && prioEndIterationFlag) {        
        CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
        ComlibPrintf("[%d] Array Send calling prio end iteration\n", 
                     CkMyPe());
        PrioMsg *pmsg = new(8 * sizeof(int)) PrioMsg;
        *(int *)CkPriorityPtr(pmsg) = -0x7FFFFFFF;
        CkSetQueueing(pmsg, CK_QUEUEING_IFIFO);
        cgproxy[CkMyPe()].prioEndIteration(pmsg);
        prioEndIterationFlag = 0;
    }        
    prevStratID = curStratID;            
    */

    ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, 
                 UsrToEnv(msg)->getTotalsize());

    register envelope * env = UsrToEnv(msg);
    if(dest_proc == CkMyPe()){
        _SET_USED(env, 0);
        CkSendMsgBranch(ep, msg, dest_proc, gid);
        return;
    }
    
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
    
    if(receivedTable)
        strategyTable[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
        cmsg->next = NULL;
        strategyTable[curStratID].tmplist.enq(cmsg);
    }
}

void ComlibManager::ArrayBroadcast(int ep,void *m,CkArrayID a){
    ComlibPrintf("[%d] Array Broadcast \n", CkMyPe());

    register envelope * env = UsrToEnv(m);
    env->getsetArrayMgr()=a;
    env->getsetArraySrcPe()=CkMyPe();
    env->getsetArrayEp()=ep;
    env->getsetArrayHops()=0;
    env->getsetArrayIndex()= dummyArrayIndex;
    
    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));

    CkSectionInfo minfo;
    minfo.type = COMLIB_MULTICAST_MESSAGE;
    minfo.sInfo.cInfo.instId = curStratID;
    minfo.sInfo.cInfo.status = COMLIB_MULTICAST_ALL;  
    minfo.sInfo.cInfo.id = 0; 
    minfo.pe = CkMyPe();
    ((CkMcastBaseMsg *)m)->_cookie = minfo;       

    CharmMessageHolder *cmsg = new 
        CharmMessageHolder((char *)m, IS_MULTICAST);
    cmsg->npes = 0;
    cmsg->pelist = NULL;
    cmsg->sec_id = NULL;

    multicast(cmsg);
}

void ComlibManager::ArraySectionSend(int ep, void *m, CkArrayID a, 
                                     CkSectionID &s) {

#ifndef CMK_OPTIMIZE
    traceUserEvent(section_send_event);
#endif

    ComlibPrintf("[%d] Array Section Send \n", CkMyPe());

    register envelope * env = UsrToEnv(m);
    env->getsetArrayMgr()=a;
    env->getsetArraySrcPe()=CkMyPe();
    env->getsetArrayEp()=ep;
    env->getsetArrayHops()=0;
    env->getsetArrayIndex()= dummyArrayIndex;
    
    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));
    
    env->setUsed(0);    
    CkPackMessage(&env);
    
    totalMsgCount ++;
    totalBytes += env->getTotalsize();

    //Provide a dummy dest proc as it does not matter for mulitcast 
    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)m,IS_MULTICAST);
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

void ComlibManager::GroupBroadcast(int ep,void *m,CkGroupID g) {
    register envelope * env = UsrToEnv(m);

    CpvAccess(_qd)->create(1);

    env->setMsgtype(ForBocMsg);
    env->setEpIdx(ep);
    env->setGroupNum(g);
    env->setSrcPe(CkMyPe());
    env->setUsed(0);

    CkPackMessage(&env);
    CmiSetHandler(env, _charmHandlerIdx);
    
    //Provide a dummy dest proc as it does not matter for mulitcast 
    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)m,IS_MULTICAST);
    
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
    totalMsgCount ++;
    totalBytes += env->getTotalsize();
    
    if (receivedTable)
	strategyTable[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
	cmsg->next = NULL;
	ComlibPrintf("Enqueuing message in tmplist at %d\n", curStratID);
        strategyTable[curStratID].tmplist.enq(cmsg);
    }

    ComlibPrintf("After multicast\n");
}

/*
void ComlibManager::multicast(void *msg, int npes, int *pelist) {
    register envelope * env = UsrToEnv(msg);
    
    ComlibPrintf("[%d]: In multicast\n", CkMyPe());

    env->setUsed(0);    
    CkPackMessage(&env);
    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));
    
    totalMsgCount ++;
    totalBytes += env->getTotalsize();

    CharmMessageHolder *cmsg = new 
    CharmMessageHolder((char *)msg,IS_MULTICAST);
    cmsg->npes = npes;
    cmsg->pelist = pelist;
    //Provide a dummy dest proc as it does not matter for mulitcast 
    //get rid of the new.
    
    if (receivedTable)
	strategyTable[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
	cmsg->next = NULL;
	ComlibPrintf("Enqueuing message in tmplist\n");
        strategyTable[curStratID].tmplist.enq(cmsg);
    }

    ComlibPrintf("After multicast\n");
}
*/


void ComlibManager::learnPattern(int total_msg_count, int total_bytes) {
    static int nrecvd = 0;
    static double avg_message_count = 0;
    static double avg_message_bytes = 0;

    avg_message_count += ((double) total_msg_count) / LEARNING_PERIOD;
    avg_message_bytes += ((double) total_bytes) /  LEARNING_PERIOD;

    nrecvd ++;
    
    if(nrecvd == CkNumPes()) {
        //Number of messages and bytes a processor sends in each iteration
        avg_message_count /= CkNumPes();
        avg_message_bytes /= CkNumPes();
        
        //CkPrintf("STATS = %5.3lf, %5.3lf", avg_message_count,
        //avg_message_bytes);

        //Learning, ignoring contention for now! 
        double cost_dir, cost_mesh, cost_grid, cost_hyp;
	double p=(double)CkNumPes();
        cost_dir = ALPHA * avg_message_count + BETA * avg_message_bytes;
        cost_mesh = ALPHA * 2 * sqrt(p) + BETA * avg_message_bytes * 2;
        cost_grid = ALPHA * 3 * pow(p,1.0/3.0) + BETA * avg_message_bytes * 3;
        cost_hyp =  (log(p)/log(2.0))*(ALPHA  + BETA * avg_message_bytes/2.0);
        
        // Find the one with the minimum cost!
        int min_strat = USE_MESH; 
        double min_cost = cost_mesh;
        if(min_cost > cost_hyp)
            min_strat = USE_HYPERCUBE;
        if(min_cost > cost_grid)
            min_strat = USE_GRID;

        if(min_cost > cost_dir)
            min_strat = USE_DIRECT;

        switchStrategy(min_strat);        
    }
}

void ComlibManager::switchStrategy(int strat){
    //CkPrintf("Switching to %d\n", strat);
}

/*
void ComlibManager::prioEndIteration(PrioMsg *pmsg){
    CkPrintf("[%d] In Prio End Iteration\n", CkMyPe());
    setInstance(pmsg->instID);
    endIteration();
    delete pmsg;
}
*/
/*
ComlibInstanceHandle ComlibCreateInstance(){
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));    
    return (cgproxy.ckLocalBranch())->createInstance();
}

ComlibInstanceHandle ComlibRegisterStrategy(Strategy *s){
    
}

ComlibInstanceHandle ComlibRegisterStrategy(Strategy *s, CkArrayID aid){
    s->setSourceArray(aid);
    ComlibPrintf("Setting aid\n");

    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));    
    ComlibInstanceHandle cinst;
    cinst = (cgproxy.ckLocalBranch())->createInstance(s);
    return cinst;
}

ComlibInstanceHandle ComlibRegisterStrategy(Strategy *s, CkGroupID gid){
    s->setSourceGroup(gid);

    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));    
    ComlibInstanceHandle cinst;
    cinst = (cgproxy.ckLocalBranch())->createInstance(s);
    return cinst;
}
*/

void ComlibDelegateProxy(CProxy *proxy){
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
    proxy->ckDelegate(cgproxy.ckLocalBranch());
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
    (cgproxy.ckLocalBranch())->doneCreating();
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
ComlibInstanceHandle::ComlibInstanceHandle(){
}

void ComlibInstanceHandle::init(){
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));    
    *this = (cgproxy.ckLocalBranch())->createInstance();
}

//Called by the communication library
ComlibInstanceHandle::ComlibInstanceHandle(int instid, CkGroupID dmid){
    _instid = instid;
    _dmid   = dmid;
}

/*
  ComlibInstanceHandle::ComlibInstanceHandle(ComlibInstanceHandle &that){
  _instid = that._instid;
  _dmid   = that._dmid;
  }        
*/

void ComlibInstanceHandle::beginIteration() { 
    CProxy_ComlibManager cgproxy(_dmid);
    (cgproxy.ckLocalBranch())->setInstance(_instid);
    (cgproxy.ckLocalBranch())->beginIteration();
}

void ComlibInstanceHandle::endIteration() {
    CProxy_ComlibManager cgproxy(_dmid);
    (cgproxy.ckLocalBranch())->endIteration();
}

void ComlibInstanceHandle::setStrategy(Strategy *s) {
    CProxy_ComlibManager cgproxy(_dmid);
    (cgproxy.ckLocalBranch())->registerStrategy(_instid, s);
}

CkGroupID ComlibInstanceHandle::getComlibManagerID() {return _dmid;}    

void ComlibInitSectionID(CkSectionID &sid){

    sid._cookie.type = COMLIB_MULTICAST_MESSAGE;
    sid._cookie.pe = CkMyPe();

    sid._cookie.sInfo.cInfo.id = 0;    
    sid.npes = 0;
    sid.pelist = NULL;
}

#include "commlib.def.h"


