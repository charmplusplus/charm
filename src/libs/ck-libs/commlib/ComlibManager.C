#include "ComlibManager.h"

#include "EachToManyStrategy.h"
#include "StreamingStrategy.h"
#include "DummyStrategy.h"
#include "MPIStrategy.h"

CpvDeclare(int, RecvmsgHandle);
CpvDeclare(int, RecvdummyHandle);
int *procMap;

//handler to receive array messages
void recv_msg(void *msg){

    if(msg == NULL)
        return;
    
    ComlibPrintf("%d:In recv_msg\n", CkMyPe());

    register envelope* env = (envelope *)msg;
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
    case USE_DIRECT: strategy = new DummyStrategy(s);
        break;
    case USE_STREAMING : strategy = new StreamingStrategy(n);
        break;
    case USE_MPI: strategy = new MPIStrategy(s);
        break;
    default : CkPrintf("Illegal Strategy\n");
        CkExit();
        break;
    }         
    return strategy;
}

//ComlibManager Constructor with 1 int the strategy id being passed
//s = Strategy (0 = tree, 1 = tree, 2 = mesh, 3 = hypercube) 
ComlibManager::ComlibManager(int s){
    strategyID = s;
    Strategy *str = createStrategy(s, 0);
    init(str);
}

//ComlibManager Constructor with 2 ints the strategy id and the number of array elements being passed. For Streaming the second int can be used for 
ComlibManager::ComlibManager(int s, int n){
    strategyID = s;
    if(s == USE_STREAMING) 
        nelements = 1;
    else 
        nelements = n;  

    ComlibPrintf("Strategy %d %d\n", strategy, nelements);
    Strategy *str = createStrategy(s, n);
    init(str);
}

/*
ComlibManager::ComlibManager(Strategy *strat){
    init(strat);
}
*/

void ComlibManager::init(Strategy *s){
    
    //comm_debug = 1;

    npes = CkNumPes();
    pelist = NULL;

    cmgrID = thisgroup;

    ComlibInit();
    
    strategy = s;
    
    elementCount = 0;
    
    CpvInitialize(int, RecvmsgHandle);
    CpvAccess(RecvmsgHandle) = CmiRegisterHandler((CmiHandler)recv_msg);

    CpvInitialize(int, RecvdummyHandle);
    CpvAccess(RecvdummyHandle) = CmiRegisterHandler((CmiHandler)recv_dummy);
    
    idSet = 0;    
    iterationFinished = 0;

    procMap = new int[CkNumPes()];
    for(int count = 0; count < CkNumPes(); count++)
        procMap[count] = count;
    
    CProxy_ComlibManager cgproxy(cmgrID);
    cgproxy[0].done();

    createDone = 0;
    doneReceived = 0;

    totalMsgCount = 0;
    totalBytes = 0;
    nIterations = 0;
}

void ComlibManager::done(){
    static int nrecvd = 0;

    nrecvd ++;

    if(nrecvd == CkNumPes()) {
      if(createDone){
          CProxy_ComlibManager cgproxy(cmgrID);

          if(npes != CkNumPes()) {
              ComlibPrintf("Calling receive id\n");
              cgproxy.receiveID(npes, pelist, comid);
              
              delete [] this->pelist;
          }
          else
              cgproxy.receiveID(comid);
      }
      doneReceived = 1;
    }
}

void ComlibManager::createId(){
    if(strategyID < USE_GRID) {
	comid = ComlibInstance(strategyID, CkNumPes());
    }
    
    if(doneReceived){
      
	CProxy_ComlibManager cgproxy(cmgrID);
	cgproxy.receiveID(comid);
    }
    createDone = 1;
}

void ComlibManager::createId(int *pelist, int npes){
    if(strategyID < USE_GRID) {
	comid = ComlibInstance(strategyID, npes);
	comid = ComlibEstablishGroup(comid, npes, pelist);
    }

    this->pelist = new int[npes];
    this->npes = npes;

    memcpy(this->pelist, pelist, npes * sizeof(int));

    ComlibPrintf("[%d]Creating comid with %d processors\n", CkMyPe(), npes);
    
    if(doneReceived){
	ComlibPrintf("Calling receive id\n");
	CProxy_ComlibManager cgproxy(cmgrID);
	
	cgproxy.receiveID(npes, pelist, comid);
        delete [] this->pelist;
    }
    createDone = 1;
}

void ComlibManager::setReverseMap(int *pelist, int npes){
    
    for(int pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;
    
    for(int pcount = 0; pcount < npes; pcount++) 
        procMap[pelist[pcount]] = pcount;
}

void ComlibManager::localElement(){
    ComlibPrintf("In Local Element\n");
    nelements ++;
}

//Called when the array element starts sending messages
//should rather be a function call ?
void ComlibManager::beginIteration(){
    //right now does not do anything might need later
    
    ComlibPrintf("[%d]:In Begin Iteration %d\n", CkMyPe(), elementCount);
    iterationFinished = 0;
}

//called when the array elements has finished sending messages
void ComlibManager::endIteration(){
  
    elementCount ++;
    int count = 0;
    if(elementCount == nelements) {
      
        ComlibPrintf("[%d]:In End Iteration %d\n", CkMyPe(), elementCount);
        iterationFinished = 1;
        
        nIterations ++;

        if(nIterations == LEARNING_PERIOD) {
            //CkPrintf("Sending %d, %d\n", totalMsgCount, totalBytes);
            CProxy_ComlibManager cgproxy(cmgrID);
            cgproxy[0].learnPattern(totalMsgCount, totalBytes);
        }
        
        if(idSet) {	    
            strategy->doneInserting();
        }
        elementCount = 0;
    }
}


void ComlibManager::receiveID(comID id){
    receiveID(CkNumPes(), NULL, id);
}

void ComlibManager::receiveID(int npes, int *pelist, comID id){

  if(npes != CkNumPes()) {
    this->npes = npes;
    this->pelist = new int[npes];
    
    memcpy(this->pelist, pelist, sizeof(int) * npes);
  }

#if CHARM_MPI
    if(npes < CkNumPes()){
        PMPI_Comm_group(MPI_COMM_WORLD, &groupWorld);
	PMPI_Group_incl(groupWorld, npes, pelist, &group);
	PMPI_Comm_create(MPI_COMM_WORLD, group, &groupComm);
    }
    else groupComm = MPI_COMM_WORLD;
#endif

    ComlibPrintf("received id in %d, npes = %d\n", CkMyPe(), npes);
    int count = 0;

    if(idSet)
        return;

    if(npes != CkNumPes())
	setReverseMap(pelist, npes);

    comid = id; 
    idSet = 1;

    strategy->setID(comid);

    if(iterationFinished){
        strategy->doneInserting();
    }
}

extern int _charmHandlerIdx;
void ComlibManager::ArraySend(int ep, void *msg, 
                              const CkArrayIndexMax &idx, CkArrayID a){
    
    CkArrayIndexMax myidx = idx;
    int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(myidx);
    
    ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, 
		 UsrToEnv(msg)->getTotalsize());

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
    CkPackMessage(&env);
    //    CmiSetHandler(env, _charmHandlerIdx);
    CmiSetHandler(env, CpvAccess(RecvmsgHandle));
    
    totalMsgCount ++;
    totalBytes += UsrToEnv(msg)->getTotalsize();

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.

    strategy->insertMessage(cmsg);
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
    
    CkPackMessage(&env);
    CmiSetHandler(env, _charmHandlerIdx);

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.
    
    strategy->insertMessage(cmsg);
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
        cost_dir = ALPHA * avg_message_count + BETA * avg_message_bytes;
        cost_mesh = ALPHA * 2 * sqrt(CmiNumPes()) + BETA * avg_message_bytes * 2;
        cost_grid = ALPHA * 3 * cbrt(CmiNumPes()) + BETA * avg_message_bytes * 3;
        cost_hyp =  (log(CmiNumPes())/log(2.0))*(ALPHA  + BETA * avg_message_bytes/2.0);
        
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
        for(int count = 0; count < messageCount/2; count ++) {
            CkSendMsgBranch(env->getEpIdx(), cmsg->getCharmMessage(), 
                            cmsg->dest_proc, env->getGroupNum());
            cmsg = cmsg->next;
        }
        
        CharmMessageHolder *tmpbuf = cmsg;
        int tot_size = 0;
        for(int count = messageCount/2; count < messageCount; count ++) {
            tot_size += cmsg->getSize();
            cmsg =  cmsg->next;
        }
        
        ComlibMsg *newmsg = new(&tot_size, 0) ComlibMsg;
        
        cmsg = tmpbuf;
        for(int count = messageCount/2; count < messageCount; count ++) {
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
