//#include "ck.h"
//#include "ckarray.h"
#include "ComlibManager.h"

CpvDeclare(int, RecvmsgHandle);
CpvDeclare(int, RecvdummyHandle);

//CpvDeclare(int, AllDoneHandle);

CkGroupID cmgrID;
//extern CkGroupID delegateManagerID;

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

void recv_dummy(void *msg){
    ComlibPrintf("Received Dummy%d\n", CkMyPe());    
    CmiFree(msg);
}

ComlibManager::ComlibManager(int s){
    init(s, 0, 1000000, 5000000);
}

//s = Strategy (0 = tree, 1 = mesh, 2 = hypercube) 
ComlibManager::ComlibManager(int s, int n){
    init(s, n, 1000000, 5000000);
}


ComlibManager::ComlibManager(int s, int nmFlush, int bFlush){
    init(s, 0, nmFlush, bFlush);
}

ComlibManager::ComlibManager(int s, int n, int nmFlush, int bFlush){
    init(s, n, nmFlush, bFlush);
}

void ComlibManager::init(int s, int n, int nmFlush, int bFlush){
    
    comm_debug = 1;

    cmgrID = thisgroup;

    ComlibInit();
    
    strategy = s;
    
    if(nmFlush == 0)
        nmFlush = 1000000;
    if(bFlush == 0)
        bFlush = 5000000;
    
    messagesBeforeFlush = nmFlush;
    bytesBeforeFlush = bFlush;

    nelements = n;  //number of elements on that processor, 
    //currently pased by the user. Should discover it.

    elementCount = 0;
    //    elementRecvCount = 0;

    ComlibPrintf("Strategy %d %d %d\n", strategy, nelements, messagesBeforeFlush);
    
    CpvInitialize(int, RecvmsgHandle);
    CpvAccess(RecvmsgHandle) = CmiRegisterHandler((CmiHandler)recv_msg);

    CpvInitialize(int, RecvdummyHandle);
    CpvAccess(RecvdummyHandle) = CmiRegisterHandler((CmiHandler)recv_dummy);
    
    if(CkMyPe() == 0) {
	//printf("before createinstance\n");
        comid = ComlibInstance(strategy, CkNumPes());
	//printf("after createInstance\n");
	//        idSet = 1;
    }
    //else
    idSet = 0;
    
    iterationFinished = 0;

    messageBuf = NULL;
    messageCount = 0;

    procMap = new int[CkNumPes()];
    for(int count = 0; count < CkNumPes(); count++)
        procMap[count] = count;
    
    CProxy_ComlibManager cgproxy(cmgrID);
    cgproxy[0].done();
}

void ComlibManager::done(){
    static int nrecvd = 0;

    nrecvd ++;

    if(nrecvd == CkNumPes()){
        CProxy_ComlibManager cgproxy(cmgrID);
	cgproxy.receiveID(comid);
    }
}

void ComlibManager::localElement(){
    //    ComlibPrintf("In Local Element\n");
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
        
        if((messageCount == 0) && (CkNumPes() > 0)) {
            char *newmsg = (char *) CmiAlloc(CmiMsgHeaderSizeBytes);
            
            ComlibPrintf("Creating a dummy message\n");
            
            CmiSetHandler(UsrToEnv(newmsg), 
                          CpvAccess(RecvdummyHandle));
            
            messageBuf = new CharmMessageHolder((char *)newmsg, CmiMyPe());
            messageCount ++;
        }
        
        if(idSet) {	    
            
            if(strategy != NAMD_STRAT) {
                
                ComlibPrintf("%d:Setting Num Deposit to %d\n", CkMyPe(), messageCount);
                NumDeposits(comid, messageCount);
                
                CharmMessageHolder *cmsg = messageBuf;
                for(count = 0; count < messageCount; count ++) {
                    if(strategy != USE_DIRECT) {
                        char * msg = cmsg->getCharmMessage();
                        ComlibPrintf("Calling EachToMany %d %d %d\n", 
                                     UsrToEnv(msg)->getTotalsize(), CkMyPe(), cmsg->dest_proc);
                        EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), UsrToEnv(msg), 
                                            1, &procMap[cmsg->dest_proc]);
                    }
                    CharmMessageHolder *prev = cmsg;
                    cmsg = cmsg->next;
                    if(prev != NULL)
                        delete prev;                //foobar getrid of the delete
                    
                }
            }
            else{
                // In the Namd Strategy half the messages are sent to a remote 
                //processor and the remaining half are sent directly.
                
                CharmMessageHolder *cmsg = messageBuf;
                envelope *env = (envelope *)cmsg->data;
                for(int count = 0; count < messageCount/2; count ++) {
                    CkSendMsgBranch(env->getEpIdx(), cmsg->getCharmMessage(), cmsg->dest_proc, 
                                    env->getGroupNum());
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
                cmgr[/*yet to decide*/0].receiveNamdMessage(newmsg);
            }
            messageCount = 0;
        }
        elementCount = 0;
    }
}

void ComlibManager::receiveID(comID id){
    
    //    ComlibPrintf("received id in %d\n", CkMyPe());
    
    if(idSet)
        return;

    comid = id; 
    idSet = 1;

    if(iterationFinished){

        if(strategy != NAMD_STRAT) {

            NumDeposits(comid, messageCount);
            ComlibPrintf("Setting Num Deposit to %d\n", messageCount);
            
            CharmMessageHolder *cmsg = messageBuf;
            for(int count = 0; count < messageCount; count ++) {
                if(strategy != USE_DIRECT) {
                    char * msg = cmsg->getCharmMessage();
                    ComlibPrintf("Calling EachToMany %d %d %d\n", UsrToEnv(msg)->getTotalsize(), 
                                 CkMyPe(), cmsg->dest_proc);
                    EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), UsrToEnv(msg), 
                                        1, &procMap[cmsg->dest_proc]);
                }
                CharmMessageHolder *prev = cmsg;
                cmsg = cmsg->next;
                if(prev != NULL)
                    delete prev;                //foobar getrid of the delete
            }
        }
        else {
            // In the Namd Strategy half the messages are sent to a remote 
            //processor and the remaining half are sent directly.

            CharmMessageHolder *cmsg = messageBuf;
            envelope *env = (envelope *)cmsg->data;
            for(int count = 0; count < messageCount/2; count ++) {
                CkSendMsgBranch(env->getEpIdx(), cmsg->getCharmMessage(), cmsg->dest_proc, 
                                env->getGroupNum());
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
            cmgr[/*yet to decide*/0].receiveNamdMessage(newmsg);
        }
        messageCount = 0;
        messageBuf = NULL;
    }
}

extern int _charmHandlerIdx;
void ComlibManager::ArraySend(int ep, void *msg, 
                              const CkArrayIndexMax &idx, CkArrayID a){
    
    CkArrayIndexMax myidx = idx;
    int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(myidx);
    
    ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, UsrToEnv(msg)->getTotalsize());

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
    
    if(strategy == USE_DIRECT) {
        ComlibPrintf("Sending Directly\n");
        CmiSyncSendAndFree(dest_proc, UsrToEnv(msg)->getTotalsize(), (char *)UsrToEnv(msg));
        return;
    }

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.
    
    /*
    cmsg->aid = aid;
    cmsg->idx = idx;
    cmsg->ep = ep;
    */
    messageCount ++;
    cmsg->next = messageBuf;
    messageBuf = cmsg;    
}

void ComlibManager::GroupSend(int ep, void *msg, int onPE, CkGroupID gid){
    
    int dest_proc = onPE;
    ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, UsrToEnv(msg)->getTotalsize());

    if(dest_proc == CkMyPe()){
        CkSendMsgBranch(ep, msg, dest_proc, gid);
        return;
    }
    
    register envelope * env = UsrToEnv(msg);
    env->setMsgtype(ForBocMsg);
    env->setEpIdx(ep);
    env->setGroupNum(gid);
    env->setSrcPe(CkMyPe());
    
    CkPackMessage(&env);
    CmiSetHandler(env, _charmHandlerIdx);

    if(strategy == USE_DIRECT) {
        ComlibPrintf("Sending Directly\n");
        CmiSyncSendAndFree(dest_proc, UsrToEnv(msg)->getTotalsize(), (char *)UsrToEnv(msg));
        return;
    }
    
    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.
    
    /*
    cmsg->gid = gid;
    cmsg->ep = ep;
    */
    messageCount ++;
    cmsg->next = messageBuf;
    messageBuf = cmsg;    
}

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

#include "ComlibModule.def.h"

