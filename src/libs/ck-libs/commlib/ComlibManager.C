//#include "ck.h"
//#include "ckarray.h"
#include "ComlibManager.h"

CpvDeclare(int, RecvmsgHandle);
CpvDeclare(int, AllDoneHandle);

CkGroupID cmgrID;
//extern CkGroupID delegateManagerID;

void recv_msg(void *msg){
    
    //    CkPrintf("Received Message %d\n", CkMyPe());
    
    //    CProxy_ComlibManager cgproxy(cmgrID);
    
    CProxy_ComlibManager cgproxy(cmgrID);
    
    cgproxy.ckLocalBranch()->receiveMessage
        ((ComlibMsg *)EnvToUsr((envelope *)msg));
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
    
    //    CkPrintf("In constructor %d %d\n", CkMyPe(), n);

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

    //CkPrintf("Strategy %d %d %d\n", strategy, nelements, messagesBeforeFlush);

    CpvInitialize(int, RecvmsgHandle);
    CpvAccess(RecvmsgHandle) = CmiRegisterHandler((CmiHandler)recv_msg);
    //    CpvAccess(AllDoneHandle) = CmiRegisterHandler((CmiHandler)all_done);
    
    if(CkMyPe() == 0) {
	//printf("before createinstance\n");
        comid = ComlibInstance(strategy, CkNumPes());
	//printf("after createInstance\n");
	//        idSet = 1;
    }
    //else
    idSet = 0;
    
    iterationFinished = 0;

    messageBuf = new CharmMessageHolder *[CkNumPes()];
    //    receiveBuf = new ComlibMsg *[CkNumPes()];
    
    messageCount = new int[CkNumPes()];
    messageSize = new int[CkNumPes()];
    
    for(int count = 0; count < CkNumPes(); count++){
	messageCount[count] = 0;
	messageSize[count] = 0;
	messageBuf[count] = NULL;
        //        receiveBuf[count] = NULL;
    }    

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
    //    CkPrintf("In Local Element\n");
    nelements ++;
}

//Called when the array element starts sending messages
//should rather be a function call ?
void ComlibManager::beginIteration(){
    //right now does not do anything might need later
    
    iterationFinished = 0;
}

//called when the array elements has finished sending messages
void ComlibManager::endIteration(){
  
    elementCount ++;
    
    if(elementCount == nelements) {
      
        //CkPrintf("[%d]:In End Iteration %d\n", CkMyPe(), elementCount);
        iterationFinished = 1;

        if(idSet) {
            int ndeposit = 0;
	    
	    int count = 0;
            for(count = 0; count < CkNumPes(); count ++)
                if(messageBuf[count] != NULL)
                    ndeposit ++;

            if((ndeposit == 0) && (CkNumPes() > 0))
                for(count = 0; count < CkNumPes(); count ++)
                    if((messageBuf[count] == NULL) && (count != CkMyPe())){
                        
                        int size = 0;
                        ComlibMsg *newmsg = new(&size, 0) ComlibMsg;
                        newmsg->isDummy = 1;
                        
                        //  CkPrintf("Creating a dummy message\n");

                        CmiSetHandler(UsrToEnv(newmsg), 
                                      CpvAccess(RecvmsgHandle));
                        
                        ndeposit ++;
                        break;
                    }
           
	    //CkPrintf("Setting Num Deposit to %d\n", ndeposit);
            NumDeposits(comid, ndeposit);

            for(count = 0; count < CkNumPes(); count ++)
                sendMessage(count);   
        }
        elementCount = 0;
    }
}

/*
  void ComlibManager::setNumMessages(int nmessages){
  CkPrintf("In setNumMessages\n");
  nMessages = nmessages;
  }
*/

void ComlibManager::receiveID(comID id){
    
    //    CkPrintf("received id in %d\n", CkMyPe());
    
    if(idSet)
        return;

    comid = id; 
    idSet = 1;

    if(iterationFinished){
        int ndeposit = 0;
        
	int count = 0;
        for(count = 0; count < CkNumPes(); count ++)
            if(messageBuf[count] != NULL)
                ndeposit ++;

	//	CkPrintf("Setting Num Deposit to %d\n", ndeposit);

        if((ndeposit == 0) && (CkNumPes() > 0))
            for(count = 0; count < CkNumPes(); count ++)
                if((messageBuf[count] == NULL) && (count != CkMyPe())){
                    
                    int size = 0;
                    ComlibMsg *newmsg = new(&size, 0) ComlibMsg;
                    newmsg->isDummy = 1;
                    
                    //  CkPrintf("Creating a dummy message\n");
                    CmiSetHandler(UsrToEnv(newmsg), 
                                  CpvAccess(RecvmsgHandle)); 
                    ndeposit ++;
                    break;
                }       

        NumDeposits(comid, ndeposit);
        //        CkPrintf("Setting Num Deposit to %d\n", ndeposit);
        for(count = 0; count < CkNumPes(); count ++)
            sendMessage(count);
    }
}

void ComlibManager::ArraySend(int ep, void *msg, 
                              const CkArrayIndexMax &idx, CkArrayID a){
    
    CkArrayIndexMax myidx = idx;

    //    int dest_proc = msg->dst % CkNumPes();
    int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(myidx);
    
    //CkPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, UsrToEnv(msg)->getTotalsize());

    if(dest_proc == CkMyPe()){
        //CkArrayID::CkLocalBranch(a)->deliverViaQueue((CkArrayMessage *) msg);
        CProxyElement_ArrayBase ap(a,idx);
        ap.ckSend((CkArrayMessage *)msg, ep);
        return;
    }
    
    //Check later
    CharmMessageHolder *cmsg = new CharmMessageHolder(ep, msg, myidx, a);
    //CharmMessageHolder *cmsg = new CharmMessageHolder(msg);
    
    messageCount[dest_proc] ++;
    messageSize[dest_proc] += cmsg->getSize();
    cmsg->next = messageBuf[dest_proc];
    messageBuf[dest_proc] = cmsg;    

    
    //    CkPrintf("Message Size = %d\n", messageSize[dest_proc]);
    if ((messageCount[dest_proc] >= messagesBeforeFlush)
        /*|| (messageSize[dest_proc] >= bytesBeforeFlush) */) {
        //CkPrintf("Sending to %d after %d messages and %d \n", dest_proc, messageCount[dest_proc], messagesBeforeFlush); 
        sendMessage(dest_proc);
    }
}

void ComlibManager::sendMessage(int dest_proc){

    if(strategy != USE_DIRECT && !idSet)
        return;

    if(messageCount[dest_proc] == 0)
        return;

    //    CkPrintf("Sending data %d\n", messageCount[dest_proc]);

    int sizes[1];
    sizes[0] = messageSize[dest_proc];
    //    sizes[1] = messageCount[dest_proc];
    
    ComlibMsg *newmsg = new(sizes, 0) ComlibMsg();
    newmsg->isDummy = 0;
    
    CharmMessageHolder *cmsg;

    cmsg = messageBuf[dest_proc];
    while(cmsg != NULL) {
        newmsg->insert(cmsg);
        cmsg = cmsg->next;
    }
    
    newmsg->src = CkMyPe();
        
    CharmMessageHolder *dptr = messageBuf[dest_proc], *dprev = NULL;
    while(dptr != NULL){
        if(dprev)
            delete dprev;
        
        dprev = dptr;
        dptr = dptr ->next;
    }
    
    messageCount[dest_proc] = 0;
    messageSize[dest_proc] = 0;
    messageBuf[dest_proc] = NULL;

    CmiSetHandler(UsrToEnv(newmsg), CpvAccess(RecvmsgHandle));

    if(strategy != USE_DIRECT) {
        //        CkPrintf("Calling EachToMany %d %d %d\n", UsrToEnv(newmsg)->getTotalsize(), CkMyPe(), dest_proc);
	EachToManyMulticast(comid, UsrToEnv(newmsg)->getTotalsize(), 
                            UsrToEnv(newmsg), 1, &dest_proc);
    }
    else
        CmiSyncSendAndFree(dest_proc,  UsrToEnv(newmsg)->getTotalsize(), (char *)UsrToEnv(newmsg));
}

void ComlibManager::receiveMessage(ComlibMsg *msg){

    //    CkPrintf("In receiveMessage %d, %d %d\n", msg->curSize, msg->src, CkMyPe());
    
    if(msg->isDummy){

        //        CkPrintf("Received Dummy, ignoring\n");

        delete msg;
        return;
    }
    
    ComlibMsg::unpack(msg);

    //    CkPrintf("After unpack\n");

    //    receiveBuf[msg->src] = msg;
    
    for(int mcount = 0; mcount < msg->nmessages; mcount++){

        CharmMessageHolder *cmsg = msg->next();
        
        cmsg->setRefcount(msg);

        CkArrayID a = cmsg->a;
        CProxyElement_ArrayBase ap(cmsg->a,cmsg->idx);
        ap.ckSend(cmsg->getCharmMessage(), cmsg->ep);
        
        //        CkPrintf("Delivering Message to %d\n", cmsg->ep);
    }
    
    delete msg;
}

CharmMessageHolder::CharmMessageHolder(int ep, void *msg, CkArrayIndexMax 
                                       &idx, CkArrayID a){
    this->ep = ep;
    this->data = (char *)UsrToEnv(msg);

    envelope *env = (envelope *)data;
    CkPackMessage(&env);
    
    //CkPrintf("Charm Message Size %d\n", ((envelope *)data)->getTotalsize());

    this->idx = idx;
    this->a = a;
}

int CharmMessageHolder::getSize(){
    return sizeof(CharmMessageHolder) + ((envelope *)data)->getTotalsize();
}

void CharmMessageHolder::init(){
    data = (char *)this + sizeof(CharmMessageHolder);
}

CkArrayMessage * CharmMessageHolder::getCharmMessage(){
    data = (char *)this + sizeof(CharmMessageHolder);

    //CkPrintf("Charm Message returned size = %d\n", ((envelope *)data)->getTotalsize());

    envelope *env = (envelope *)data;
    CkUnpackMessage(&env);

    return (CkArrayMessage *) EnvToUsr(env);
}

void CharmMessageHolder::copy(char *buf){
    *((CharmMessageHolder *)buf) = *this;
    memcpy(buf + sizeof(CharmMessageHolder), data, 
           ((envelope *)data)->getTotalsize());
}

#define REFFIELD(m) ((int *)((char *)(m)-sizeof(int)))[0]
#define BLKSTART(m) ((char *)m-2*sizeof(int))

void CharmMessageHolder::setRefcount(void *msg){
    char *env = (char *)UsrToEnv(msg);
    int pref = REFFIELD(env);
    int ref =  REFFIELD(data);
    
    while (pref < 0) {

        //        CkPrintf("pref = %d\n", pref);
        
        env = env + pref;
        pref =  REFFIELD(env);
    }
    
    //    CkPrintf("pref = %d\n", pref);

    pref ++;
    ref = (unsigned int)((char *) env - (char *) data);
    
    REFFIELD(data) = ref;
    REFFIELD(env) = pref;

    //    CkPrintf("Setting ref for %u to %u\n", data, BLKSTART(env));
}

void * ComlibMsg::alloc(int mnum, size_t size, int *sizes, int priobits){
    int total_size = size + sizes[0];
    ComlibMsg *msg = (ComlibMsg *)CkAllocMsg(mnum, total_size, priobits);
    msg->nmessages = 0;
    msg->curSize = 0;
    msg->src = CkMyPe();
    msg->data = (char *)msg + size;
    return (void *) msg;
}

void * ComlibMsg::pack(ComlibMsg *msg){
    return (void *) msg;
}

ComlibMsg * ComlibMsg::unpack(void *buf){
    ComlibMsg *msg = (ComlibMsg *) buf;
    msg->curSize = 0;
    msg->data = (char *)msg + sizeof(ComlibMsg);
    
    return msg;
}

void ComlibMsg::insert(CharmMessageHolder *msg){
    nmessages ++;
    msg->copy(data + curSize);
    curSize += msg->getSize();
    //CkPrintf("CurSize = %d\n", curSize);
}

CharmMessageHolder * ComlibMsg::next(){
    
    //    CkPrintf("In ComlibMsg::next()\n");
    
    if(nmessages == 0){
        CkPrintf("ComlibMsg::next OVEFLOW\n");
        return NULL;
    }

    //nmessages --;
    CharmMessageHolder *cmsg = (CharmMessageHolder *)(data + curSize);

    cmsg->init();
    curSize += cmsg->getSize();
    return cmsg;
}

#include "ComlibModule.def.h"
