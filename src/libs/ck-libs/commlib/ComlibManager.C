#include "ComlibManager.h"

#include "EachToManyStrategy.h"
#include "EachToManyMulticastStrategy.h"
#include "StreamingStrategy.h"
#include "DummyStrategy.h"
#include "MPIStrategy.h"
#include "NodeMulticast.h"

CpvDeclare(int, RecvmsgHandle);
CpvDeclare(int, RecvdummyHandle);

//handler to receive array messages
void recv_msg(void *msg){

    if(msg == NULL)
        return;
    
    ComlibPrintf("%d:In recv_msg\n", CkMyPe());

    register envelope* env = (envelope *)msg;
    env->setUsed(0);
    CProxyElement_ArrayBase ap(env->array_mgr(), env->array_index());
    ComlibPrintf("%d:Array Base created\n", CkMyPe());
    ap.ckSend((CkArrayMessage *)EnvToUsr(env), env->array_ep());
    
    ComlibPrintf("%d:Out of recv_msg\n", CkMyPe());
    return;
}

//handler for dummy messages
void recv_dummy(void *msg){
    ComlibPrintf("Received Dummy %d\n", CkMyPe());    
    CmiFree(msg);
}

//Creates a strategy, to write a new strategy add the appropriate constructor call!
//For now the strategy constructor can only receive an int.
Strategy* createStrategy(int s, int n){
    Strategy *strategy;
    switch (s) {
    case USE_MESH: 
    case USE_HYPERCUBE : 
    case USE_GRID : strategy = new EachToManyStrategy(s);
        break;
    case USE_DIRECT: strategy = new DummyStrategy();
        break;
    case USE_STREAMING : strategy = new StreamingStrategy(n);
        break;
    case USE_MPI: strategy = new MPIStrategy();
        break;
    default : CkPrintf("Illegal Strategy\n");
        CkExit();
        break;
    }         
    return strategy;
}

//An initialization routine which does prelimnary initialization of the 
//communications library and registers the strategies with the PUP:able interface.
void initComlibManager(void){
    //comm_debug = 1;
    ComlibInit();
    ComlibPrintf("Init Call\n");
    /*
    //Called once on each processor 
    PUPable_reg(Strategy); 
    PUPable_reg(EachToManyStrategy); 
    PUPable_reg(DummyStrategy); 
    PUPable_reg(MPIStrategy); 
    PUPable_reg(StreamingStrategy);     
    PUPable_reg(NodeMulticast);     
    */

    CpvInitialize(int, RecvmsgHandle);
    CpvAccess(RecvmsgHandle) = CmiRegisterHandler((CmiHandler)recv_msg);
    
    //CmiPrintf("[%d] Registering Handler %d\n", CmiMyPe(), CpvAccess(RecvmsgHandle));
    
    CpvInitialize(int, RecvdummyHandle);
    CpvAccess(RecvdummyHandle) = CmiRegisterHandler((CmiHandler)recv_dummy);
    //ComlibPrintf("After Init Call\n");
}

//ComlibManager Constructor with 1 int the strategy id being passed
//s = Strategy (0 = tree, 1 = tree, 2 = mesh, 3 = hypercube) 
ComlibManager::ComlibManager(int s){
    init();
    strategyID = s;
    Strategy *strat = createStrategy(s, 0);
    createInstance(strat);
}

//ComlibManager Constructor with 2 ints the strategy id and the 
//number of array elements being passed. For Streaming the second 
//int can be used for 
ComlibManager::ComlibManager(int s, int n){
    init();
    strategyID = s;
    if(s == USE_STREAMING) 
      strategyTable[0].numElements = 1;
    else 
      strategyTable[0].numElements = n;  
    
    //receivedTable = 1;
    ComlibPrintf("Strategy %d %d\n", strategyID, strategyTable[0].numElements);

    Strategy *strat = createStrategy(s, n);
    createInstance(strat);
}

ComlibManager::ComlibManager(){
    init();
    ComlibPrintf("In comlibmanager constructor\n");
}

void ComlibManager::init(){

    //comm_debug = 1;
    
    npes = CkNumPes();
    pelist = NULL;
    nstrats = 0;

    cmgrID = thisgroup;

    curStratID = 0;

    //initialize the strategy table.
    //    bzero(strategyTable, MAX_NSTRAT * sizeof(StrategyTable));
    for(int count = 0; count < MAX_NSTRAT; count ++) {
      strategyTable[count].strategy = NULL;
      strategyTable[count].numElements = 0;
      strategyTable[count].elementCount = 0;
      strategyTable[count].call_doneInserting = 0;
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

    CProxy_ComlibManager cgproxy(cmgrID);
    cgproxy[0].barrier();
}

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

void ComlibManager::barrier2(){
  static int bcount = 0;
  if(CkMyPe() == 0) {
    bcount ++;
    ComlibPrintf("In barrier2 %d\n", bcount);
    if(bcount == CkNumPes()) {
      CProxy_ComlibManager cgproxy(cmgrID);
      cgproxy.resumeFromBarrier2();
    }
  }
}


void ComlibManager::createId(){
    doneCreating();
}

void ComlibManager::createId(int *pelist, int npes){
    
    Strategy *strat;
    if(strategyID != USE_MPI) 
        strat = new EachToManyStrategy(strategyID, npes, pelist);
    else 
        strat = new MPIStrategy(npes, pelist);

    //createInstance(strat);        
    ListOfStrategies.insert(0, strat);
    doneCreating();
}

int ComlibManager::createInstance(Strategy *strat) {
  
  ListOfStrategies.enq(strat);
  nstrats++;
  return nstrats - 1;
}

void ComlibManager::doneCreating() {
    if(!barrierReached)
      return;

    ComlibPrintf("Sending Strategies %d, %d\n", nstrats, ListOfStrategies.length());

    StrategyWrapper sw;
    sw.s_table = new Strategy* [nstrats];
    //    Strategy *aStrategy = ListOfStrategies.deq();
    sw.nstrats = nstrats;
    
    for (int count=0; count<nstrats; count++)
        sw.s_table[count] = ListOfStrategies.deq();

    CProxy_ComlibManager cgproxy(cmgrID);
    cgproxy.receiveTable(sw);
}

void ComlibManager::localElement(){
    ComlibPrintf("In Local Element\n");
    strategyTable[0].numElements ++;
}

void ComlibManager::registerElement(int stratID){
    ComlibPrintf("In Register Element\n");
    strategyTable[stratID].numElements ++;
}

void ComlibManager::unRegisterElement(int stratID){
    ComlibPrintf("In Un Register Element\n");
    strategyTable[stratID].numElements --;
}

//Called when the array element starts sending messages
//should rather be a function call ?
void ComlibManager::beginIteration(){
    //right now does not do anything might need later
     
    ComlibPrintf("[%d]:In Begin Iteration %d\n", CkMyPe(), strategyTable[0].elementCount);
    curStratID = 0;
}

//Called when the array element starts sending messages
//should rather be a function call ?
void ComlibManager::beginIteration(int stratID){

    curStratID = stratID;
    ComlibPrintf("[%d]:In Begin Iteration%d\n", CkMyPe(), strategyTable[stratID].elementCount);
}

//called when the array elements has finished sending messages
void ComlibManager::endIteration(){
    
    ComlibPrintf("[%d]:In End Iteration(%d) %d, %d\n", CkMyPe(), curStratID, 
                 strategyTable[curStratID].elementCount, strategyTable[curStratID].numElements);
  
    strategyTable[curStratID].elementCount++;
    int count = 0;
    flushTable = 1;

    if(strategyTable[curStratID].elementCount == strategyTable[curStratID].numElements) {
      
        ComlibPrintf("[%d]:In End Iteration %d\n", CkMyPe(), strategyTable[curStratID].elementCount);
        
        nIterations ++;

        if(nIterations == LEARNING_PERIOD) {
            //CkPrintf("Sending %d, %d\n", totalMsgCount, totalBytes);
            CProxy_ComlibManager cgproxy(cmgrID);
            cgproxy[0].learnPattern(totalMsgCount, totalBytes);
        }
        
        if(receivedTable) {	    
	    strategyTable[curStratID].strategy->doneInserting();
        }
	else strategyTable[curStratID].call_doneInserting = 1;
	
        strategyTable[curStratID].elementCount = 0;
    }
    ComlibPrintf("After EndIteration\n");
}

void ComlibManager::receiveTable(StrategyWrapper sw){
    
    nstrats = sw.nstrats;

    int count = 0;
    for(count = 0; count < nstrats; count ++)
        strategyTable[count].strategy = sw.s_table[count];
    receivedTable = 1;

    ComlibPrintf("receivedTable %d\n", nstrats);

    CProxy_ComlibManager cgproxy(cmgrID);
    cgproxy[0].barrier2();

    /*
    if (flushTable && barrier2Reached) {
      for (count = 0; count < nstrats; count ++) {
	if (!strategyTable[count].tmplist.isEmpty()) {
	  CharmMessageHolder *cptr;
	  while (!strategyTable[count].tmplist.isEmpty())
	    strategyTable[count].strategy->insertMessage
	      (strategyTable[count].tmplist.deq());
	}
	
	if (strategyTable[count].call_doneInserting)
	  strategyTable[count].strategy->doneInserting();
      }
      flushTable = 0;
    }
    */
}

void ComlibManager::resumeFromBarrier2(){
  barrier2Reached = 1;

  ComlibPrintf("[%d] Barrier 2 reached\n", CkMyPe());

  for (int count = 0; count < nstrats; count ++) {
      strategyTable[count].strategy->beginProcessing
          (strategyTable[count].numElements);
  }

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
    
    CkArrayIndexMax myidx = idx;
    int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(myidx);
    
    ComlibPrintf("Send Data %d %d %d %d\n", CkMyPe(), dest_proc, 
		 UsrToEnv(msg)->getTotalsize(), receivedTable);

    if(dest_proc == CkMyPe()){
        //CkArrayID::CkLocalBranch(a)->deliverViaQueue((CkArrayMessage *) msg);
        CProxyElement_ArrayBase ap(a,idx);
        ap.ckSend((CkArrayMessage *)msg, ep);
        return;
    }

    register envelope * env = UsrToEnv(msg);
    
    env->array_mgr()=a;
    env->array_srcPe()=CkMyPe();
    env->array_ep()=ep;
    env->array_hops()=0;
    env->array_index()=idx;
    env->setUsed(0);
    
    CkPackMessage(&env);
    //    CmiSetHandler(env, _charmHandlerIdx);
    CmiSetHandler(env, CpvAccess(RecvmsgHandle));
    
    totalMsgCount ++;
    totalBytes += UsrToEnv(msg)->getTotalsize();

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.

    //CmiPrintf("Before Insert\n");

    if (receivedTable)
      strategyTable[curStratID].strategy->insertMessage(cmsg);
    else {
        flushTable = 1;
        cmsg->next = NULL;
        strategyTable[curStratID].tmplist.enq(cmsg);
    }

    //CmiPrintf("After Insert\n");
}

void ComlibManager::multicast(void *msg) {
    register envelope * env = UsrToEnv(msg);
    
    ComlibPrintf("[%d]: In multicast\n", CkMyPe());

    env->setUsed(0);    
    CkPackMessage(&env);
    CmiSetHandler(env, CpvAccess(RecvmsgHandle));
    
    totalMsgCount ++;
    totalBytes += env->getTotalsize();

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, 0); 
    //Provide a dummy dest proc as it does not matter for mulitcast 
    //get rid of the new.
    cmsg->npes = CkNumPes();
    cmsg->pelist = NULL;

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

void ComlibManager::multicast(void *msg, int npes, int *pelist) {
    register envelope * env = UsrToEnv(msg);
    
    ComlibPrintf("[%d]: In multicast\n", CkMyPe());

    env->setUsed(0);    
    CkPackMessage(&env);
    CmiSetHandler(env, CpvAccess(RecvmsgHandle));
    
    totalMsgCount ++;
    totalBytes += env->getTotalsize();

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, 0);
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


#include "qd.h"
//CpvExtern(QdState*, _qd);

void ComlibManager::GroupSend(int ep, void *msg, int onPE, CkGroupID gid){
    
    int dest_proc = onPE;
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

void ComlibManager::learnPattern(int total_msg_count, int total_bytes) {
    static int nrecvd = 0;
    static double avg_message_count = 0;
    static double avg_message_bytes = 0;

    avg_message_count += ((double) total_msg_count) / LEARNING_PERIOD;
    avg_message_bytes += ((double) total_bytes) /  LEARNING_PERIOD;

    nrecvd ++;
    
    if(nrecvd == CmiNumPes()) {
        //Number of messages and bytes a processor sends in each iteration
        avg_message_count /= CmiNumPes();
        avg_message_bytes /= CmiNumPes();
        
        //CkPrintf("STATS = %5.3lf, %5.3lf", avg_message_count, avg_message_bytes);

        //Learning, ignoring contention for now! 
        double cost_dir, cost_mesh, cost_grid, cost_hyp;
	double p=(double)CmiNumPes();
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

CharmMessageHolder::CharmMessageHolder(char * msg, int proc) {
    data = (char *)UsrToEnv(msg);
    dest_proc = proc;
    isDummy = 0;
}

void CharmMessageHolder::init(char * root_msg) {

    data = (char *)this + sizeof(CharmMessageHolder);
    
    register envelope * env = (envelope *)data;
    CkUnpackMessage(&env);

    setRefcount(root_msg);
}

char * CharmMessageHolder::getCharmMessage(){
    return (char *)EnvToUsr((envelope *) data);
}

void CharmMessageHolder::copy(char *buf){
    *((CharmMessageHolder *)buf) = *this;
    memcpy(buf + sizeof(CharmMessageHolder), data, ((envelope *)data)->getTotalsize());
}

int CharmMessageHolder::getSize(){
    return sizeof(CharmMessageHolder) + ((envelope *)data)->getTotalsize();
}

#define REFFIELD(m) ((int *)((char *)(m)-sizeof(int)))[0]
void CharmMessageHolder::setRefcount(char *msg){
    char *env = (char *)UsrToEnv(msg);
    int pref = REFFIELD(env);
    int ref =  REFFIELD(data);
    
    while (pref < 0) {
        env = env + pref;
        pref =  REFFIELD(env);
    }
    
    pref ++;
    ref = (unsigned int)((char *) env - (char *) data);
    
    REFFIELD(data) = ref;
    REFFIELD(env) = pref;    
}

/*
void ComlibMsg::insert(CharmMessageHolder *msg){
    nmessages ++;
    
    msg->copy(data + curSize);
    curSize += msg->getSize();
    //CkPrintf("CurSize = %d\n", curSize);
}

CharmMessageHolder * ComlibMsg::next(){
    
    if(nmessages == 0){
        CkPrintf("ComlibMsg::next OVEFLOW\n");
        return NULL;
    }

    nmessages --;
    CharmMessageHolder *cmsg = (CharmMessageHolder *)(data + curSize);
    
    cmsg->init((char *) this);
    curSize += cmsg->getSize();
    return cmsg;
}
*/


void StrategyWrapper::pup (PUP::er &p) {

    //CkPrintf("In PUP of StrategyWrapper\n");

    p | nstrats;
    if(p.isUnpacking())
	s_table = new Strategy * [nstrats];
    
    for(int count = 0; count < nstrats; count ++)
        p | s_table[count];
}

#include "ComlibModule.def.h"

/*
void ComlibManager::receiveNamdMessage(ComlibMsg * msg){
    CharmMessageHolder *cmsg = msg->next();
    char *charm_msg = cmsg->getCharmMessage();

    CkPrintf("In receive namd message\n");

    while(charm_msg != NULL){
        register envelope * env = UsrToEnv(charm_msg);
        CkSendMsgBranch(env->getEpIdx(), charm_msg, cmsg->dest_proc, env->getGroupNum());

        cmsg = msg->next();
        charm_msg = cmsg->getCharmMessage();
    }
    
    CkPrintf("After receive namd message\n");

    delete msg;
}
*/

/*
void callNamdStrategy(){
    if(strategy == NAMD_STRAT) {
        // In the Namd Strategy half the messages are sent to a remote 
        //processor and the remaining half are sent directly.
        
        CharmMessageHolder *cmsg = messageBuf;
        envelope *env = (envelope *)cmsg->data;
        for(int count = 0; count < elementCount/2; count ++) {
            CkSendMsgBranch(env->getEpIdx(), cmsg->getCharmMessage(), 
                            cmsg->dest_proc, env->getGroupNum());
            cmsg = cmsg->next;
        }
        
        CharmMessageHolder *tmpbuf = cmsg;
        int tot_size = 0;
        for(int count = elementCount/2; count < elementCount; count ++) {
            tot_size += cmsg->getSize();
            cmsg =  cmsg->next;
        }
        
        ComlibMsg *newmsg = new(&tot_size, 0) ComlibMsg;
        
        cmsg = tmpbuf;
        for(int count = elementCount/2; count < elementCount; count ++) {
            newmsg->insert(cmsg);
            delete tmpbuf;
            cmsg = cmsg->next;
            tmpbuf = cmsg;
        }
        
        CProxy_ComlibManager cmgr(thisgroup);
        cmgr[0].receiveNamdMessage(newmsg); // yet to decide!
    }
}
*/
