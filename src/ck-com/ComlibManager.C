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

CkpvExtern(int, RecvdummyHandle);

CkpvDeclare(int, RecvmsgHandle);
CkpvDeclare(int, RecvCombinedShortMsgHdlrIdx);
CkpvDeclare(CkGroupID, cmgrID);
CkpvExtern(ConvComlibManager *, conv_comm_ptr);

//handler to receive array messages
void recv_array_msg(void *msg){

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

ComlibManager::ComlibManager(){
    init();
    ComlibPrintf("In comlibmanager constructor\n");
}

void ComlibManager::init(){
    
    PUPable_reg(CharmStrategy);
    PUPable_reg(CharmMessageHolder);
    
    //comm_debug = 1;
    
    CkpvAccess(RecvmsgHandle) = CkRegisterHandler((CmiHandler)recv_array_msg);
    CkpvAccess(RecvCombinedShortMsgHdlrIdx) = 
        CkRegisterHandler((CmiHandler)recv_combined_array_msg);
    
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

    strategyTable = CkpvAccess(conv_comm_ptr)->getStrategyTable();
    
    receivedTable = 0;
    flushTable = 0;
    totalMsgCount = 0;
    totalBytes = 0;
    nIterations = 0;
    barrierReached = 0;
    barrier2Reached = 0;

    isRemote = 0;
    remotePe = -1;

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

void ComlibManager::registerStrategy(int pos, CharmStrategy *strat) {
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
    
    ComlibPrintf("[%d]:In End Iteration(%d) %d, %d\n", CkMyPe(), curStratID, 
                 (* strategyTable)[curStratID].elementCount, (* strategyTable)[curStratID].numElements);

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
    flushTable = 1;
    
    if((* strategyTable)[curStratID].elementCount == (* strategyTable)[curStratID].numElements) {
        
        ComlibPrintf("[%d]:In End Iteration %d\n", CkMyPe(), (* strategyTable)[curStratID].elementCount);
        
        nIterations ++;
        
        if(nIterations == LEARNING_PERIOD) {
            //CkPrintf("Sending %d, %d\n", totalMsgCount, totalBytes);
            CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));
            cgproxy[0].learnPattern(totalMsgCount, totalBytes);
        }
        
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
//CpvAccess(conv_comm_ptr) points to the converse commlib instance
void ComlibManager::receiveTable(StrategyWrapper sw){
    
    ComlibPrintf("[%d] In receiveTable %d\n", CkMyPe(), sw.nstrats);

    receivedTable = 1;
    nstrats = sw.nstrats;

    CkArrayID st_aid;
    int st_nelements;
    CkArrayIndexMax *st_elem;

    int count = 0;
    for(count = 0; count < nstrats; count ++) {
        CharmStrategy *cur_strategy = (CharmStrategy *)sw.s_table[count];
        
        //set the instance to the current count
        //currently all strategies are being copied to all processors
        //later strategies will be selectively copied
        cur_strategy->setInstance(count);  
        CkpvAccess(conv_comm_ptr)->insertStrategy(cur_strategy);
        
        ComlibPrintf("[%d] Inserting strategy \n", CkMyPe());       

        if(cur_strategy->getType() == ARRAY_STRATEGY &&
           cur_strategy->isBracketed()){ 

            ComlibPrintf("Inserting Array Listener\n");

            ComlibArrayInfo as = ((CharmStrategy *)cur_strategy)->ainfo;
            as.getSourceArray(st_aid, st_elem, st_nelements);
            
            if(st_aid.isZero())
                CkAbort("Array ID is zero");
            
            ComlibArrayListener *calistener = 
                CkArrayID::CkLocalBranch(st_aid)->getComlibArrayListener();
            
            calistener->registerStrategy(&((* strategyTable)[count]));
        }              
  
        if(cur_strategy->getType() == GROUP_STRATEGY){
            (* strategyTable)[count].numElements = 1;
        }
        
        cur_strategy->beginProcessing((* strategyTable)[count].numElements); 
        
        ComlibPrintf("[%d] endIteration from receiveTable %d, %d\n", 
                     CkMyPe(), count,
                     (* strategyTable)[count].nEndItr);
                         
        curStratID = count;
        for(int itr = 0; itr < (* strategyTable)[count].nEndItr; itr++) 
            endIteration();            
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
          if (!(* strategyTable)[count].tmplist.isEmpty()) {
              CharmMessageHolder *cptr;
              while (!(* strategyTable)[count].tmplist.isEmpty())
                  (* strategyTable)[count].strategy->insertMessage
                      ((* strategyTable)[count].tmplist.deq());
          }
          
          if ((* strategyTable)[count].call_doneInserting) {
              ComlibPrintf("[%d] Calling done inserting \n", CkMyPe());
              (* strategyTable)[count].strategy->doneInserting();
          }
      }
    }
    
    ComlibPrintf("[%d] After Barrier2\n", CkMyPe());
}

extern int _charmHandlerIdx;

void ComlibManager::ArraySend(CkDelegateData *pd,int ep, void *msg, 
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

    register envelope * env = UsrToEnv(msg);
    
    env->getsetArrayMgr()=a;
    env->getsetArraySrcPe()=CkMyPe();
    env->getsetArrayEp()=ep;
    env->getsetArrayHops()=0;
    env->getsetArrayIndex()=idx;
    env->setUsed(0);
    
    CkPackMessage(&env);
    CmiSetHandler(env, CkpvAccess(RecvmsgHandle));
    
    if(isRemote) {
        CharmMessageHolder *cmsg = new 
            CharmMessageHolder((char *)msg, dest_proc);

        remoteQ.enq(cmsg);
        return;
    }

    if(dest_proc == CkMyPe()){
        CProxyElement_ArrayBase ap(a,idx);
        ap.ckSend((CkArrayMessage *)msg, ep);
        return;
    }

    totalMsgCount ++;
    totalBytes += UsrToEnv(msg)->getTotalsize();

    CharmMessageHolder *cmsg = new 
        CharmMessageHolder((char *)msg, dest_proc);
    //get rid of the new.

    ComlibPrintf("Before Insert\n");

    if (receivedTable)
      (* strategyTable)[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
        (* strategyTable)[curStratID].tmplist.enq(cmsg);
    }

    //CmiPrintf("After Insert\n");
}


#include "qd.h"
//CpvExtern(QdState*, _qd);

void ComlibManager::GroupSend(CkDelegateData *pd,int ep, void *msg, int onPE, CkGroupID gid){
    
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
        (* strategyTable)[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
        (* strategyTable)[curStratID].tmplist.enq(cmsg);
    }
}

void ComlibManager::ArrayBroadcast(CkDelegateData *pd,int ep,void *m,CkArrayID a){
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

void ComlibManager::ArraySectionSend(CkDelegateData *pd,int ep, void *m, CkArrayID a, 
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

void ComlibManager::GroupBroadcast(CkDelegateData *pd,int ep,void *m,CkGroupID g) {
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
	(* strategyTable)[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
	ComlibPrintf("Enqueuing message in tmplist at %d\n", curStratID);
        (* strategyTable)[curStratID].tmplist.enq(cmsg);
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
	(* strategyTable)[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
	ComlibPrintf("Enqueuing message in tmplist\n");
        (* strategyTable)[curStratID].tmplist.enq(cmsg);
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

void ComlibManager::setRemote(int remote_pe){

    ComlibPrintf("Setting remote flag on\n");

    remotePe = remote_pe;
    isRemote = 1;
}


void ComlibManager::receiveRemoteSend(CkQ<CharmMessageHolder *> &rq, 
                                      int strat_id) {
    setInstance(strat_id);
    
    int nmsgs = rq.length();

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

    if(nmsgs == 0)
        return;

    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID)); 
    cgproxy[remotePe].receiveRemoteSend(remoteQ, curStratID);
    
    for(int count = 0; count < nmsgs; count++) {
        CharmMessageHolder *cmsg = remoteQ.deq();
        CkFreeMsg(cmsg->getCharmMessage());
        delete cmsg;
    }
}


/*
void ComlibManager::prioEndIteration(PrioMsg *pmsg){
    CkPrintf("[%d] In Prio End Iteration\n", CkMyPe());
    setInstance(pmsg->instID);
    endIteration();
    delete pmsg;
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
    _instid = -1;
    _dmid.setZero();
    _srcPe = -1;
    toForward = 0;
}

//Called by user code
ComlibInstanceHandle::ComlibInstanceHandle(const ComlibInstanceHandle &h){
    _instid = h._instid;
    _dmid = h._dmid;
    toForward = h.toForward;

    ComlibPrintf("In Copy Constructor\n");

    //We DO NOT copy the source processor
    //Source PE is initialized here
    _srcPe = CkMyPe();
}

void ComlibInstanceHandle::init(){
    CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));    
    *this = (cgproxy.ckLocalBranch())->createInstance();
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

    ComlibPrintf("Instance Handle beginIteration %d, %d\n", CkMyPe(), _srcPe);

    //User forgot to make the instance handle a readonly or pass it
    //into the constructor of an array and is using it directly from
    //Main :: main
    if(_srcPe == -1) {
        ComlibPrintf("Warning:Instance Handle needs to be a readonly or a private variable of an array element\n");
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

CkGroupID ComlibInstanceHandle::getComlibManagerID() {return _dmid;}    

void ComlibInitSectionID(CkSectionID &sid){

    sid._cookie.type = COMLIB_MULTICAST_MESSAGE;
    sid._cookie.pe = CkMyPe();

    sid._cookie.sInfo.cInfo.id = 0;    
    sid.npes = 0;
    sid.pelist = NULL;
}

#include "comlib.def.h"


