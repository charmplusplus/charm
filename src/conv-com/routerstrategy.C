
#include "routerstrategy.h"

CkpvDeclare(int, RecvHandle);
CkpvDeclare(int, ProcHandle);
CkpvDeclare(int, DummyHandle);

//Handlers that call the entry funtions of routers 
//Refer to router.h for details on these entry functions

//Correspods to Router::ProcManyMsg
void procManyCombinedMsg(char *msg)
{
    //comID id;
    int instance_id;

    ComlibPrintf("In Recv combined message at %d\n", CkMyPe());
    //memcpy(&id,(msg+CmiReservedHeaderSize+sizeof(int)), sizeof(comID));

    //Comid specific
    memcpy(&instance_id, (char*) msg + CmiReservedHeaderSize + 2*sizeof(int)
           , sizeof(int));

    Strategy *s = ConvComlibGetStrategy(instance_id);
    ((RouterStrategy *)s)->ProcManyMsg(msg);
}

//Correspods to Router::DummyEP
void dummyEP(DummyMsg *m)
{
    Strategy *s = ConvComlibGetStrategy(m->id.instanceID);
    
    ((RouterStrategy *)s)->DummyEP(m);
}

//Correspods to Router::RecvManyMsg
void recvManyCombinedMsg(char *msg)
{
    //comID id;
    int instance_id;
    ComlibPrintf("In Recv combined message at %d\n", CkMyPe());
    //memcpy(&id,(msg+CmiReservedHeaderSize+sizeof(int)), sizeof(comID));
    
    //Comid specific
    memcpy(&instance_id, (char*) msg + CmiReservedHeaderSize + 2*sizeof(int)
           , sizeof(int));

    Strategy *s = ConvComlibGetStrategy(instance_id);
    ((RouterStrategy *)s)->RecvManyMsg(msg);
}


void doneHandler(DummyMsg *m){
    Strategy *s = ConvComlibGetStrategy(m->id.instanceID);
    
    ((RouterStrategy *)s)->Done(m);
}

void RouterStrategy::setReverseMap(){
    int pcount;
    for(pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;

    //All processors not in the domain will point to -1
    for(pcount = 0; pcount < npes; pcount++) {
        if (pelist[pcount] == CkMyPe())
            myPe = pcount;

        procMap[pelist[pcount]] = pcount;
    }
}

RouterStrategy::RouterStrategy(int stratid, int handle, int _npes, 
                               int *_pelist) 
    : Strategy(){
    
    setType(CONVERSE_STRATEGY);

    CkpvInitialize(int, RecvHandle);
    CkpvInitialize(int, ProcHandle);
    CkpvInitialize(int, DummyHandle);

    id.instanceID = 0; //Set later in doneInserting
    
    id.isAllToAll = 0;
    id.refno = 0;

    CkpvAccess(RecvHandle) =
        CkRegisterHandler((CmiHandler)recvManyCombinedMsg);
    CkpvAccess(ProcHandle) =
        CkRegisterHandler((CmiHandler)procManyCombinedMsg);
    CkpvAccess(DummyHandle) = 
        CkRegisterHandler((CmiHandler)dummyEP);    

    myDoneHandle = CkRegisterHandler((CmiHandler)doneHandler);    

    //Array strategy done handle
    doneHandle = handle;

    routerID = stratid;

    npes = _npes;
    //pelist = new int[npes];
    pelist = _pelist;
    //memcpy(pelist, _pelist, sizeof(int) * npes);    

    if(npes <= 1)
        routerID = USE_DIRECT;

    myPe = -1;
    procMap = new int[CkNumPes()];    
    setReverseMap();

    bcast_pemap = NULL;

    ComlibPrintf("Router Strategy : %d, MYPE = %d, NUMPES = %d \n", stratid, 
                 myPe, npes);

    if(myPe < 0) {
        //I am not part of this strategy
        
        doneFlag = 0;
        router = NULL;
        bufferedDoneInserting = 0;
        return;        
    }

    //Start with all iterations done
    doneFlag = 1;
    
    //No Buffered doneInserting at the begining
    bufferedDoneInserting = 0;

    switch(stratid) {
    case USE_TREE: 
        router = new TreeRouter(npes, myPe);
        break;
        
    case USE_MESH:
        router = new GridRouter(npes, myPe);
        break;
        
    case USE_HYPERCUBE:
        router = new DimexRouter(npes, myPe);
        break;
        
    case USE_GRID:
        router = new D3GridRouter(npes, myPe);
        break;

    case USE_DIRECT: router = NULL;
        break;
        
    default: CmiAbort("Unknown Strategy\n");
        break;
    }

    if(router) {
        router->SetMap(pelist);
        router->setDoneHandle(myDoneHandle);
        //router->SetID(id);
    }
}


RouterStrategy::~RouterStrategy() {
    //delete [] pelist;

    if(bcast_pemap)
        delete [] bcast_pemap;
    
    delete [] procMap;
    if(router)
        delete router;
}


void RouterStrategy::insertMessage(MessageHolder *cmsg){

    if(myPe < 0)
        CmiAbort("insertMessage: mype < 0\n");

    int count = 0;
    if(routerID == USE_DIRECT) {
        if(cmsg->dest_proc == IS_BROADCAST) {
            for(count = 0; count < cmsg->npes-1; count ++)
                CmiSyncSend(cmsg->pelist[count], cmsg->size, 
                            cmsg->getMessage());
            if(cmsg->npes > 0)
                CmiSyncSendAndFree(cmsg->pelist[cmsg->npes-1], cmsg->size, 
                                   cmsg->getMessage());
        }
        else
            CmiSyncSendAndFree(cmsg->dest_proc, cmsg->size, 
                               cmsg->getMessage());
        delete cmsg;
    }
    else {
        if(cmsg->dest_proc >= 0) {
            cmsg->pelist = &procMap[cmsg->dest_proc];
            cmsg->npes = 1;
        }
        else if (cmsg->dest_proc == IS_BROADCAST){

            if(bcast_pemap == NULL) {
                bcast_pemap = new int[npes];
                for(count = 0; count < npes; count ++) {
                    bcast_pemap[count] = count;
                }
            }

            cmsg->pelist = bcast_pemap;
            cmsg->npes = npes;
        }
        
        msgQ.push(cmsg);
    }
}

void RouterStrategy::doneInserting(){
    
    if(myPe < 0)
        CmiAbort("insertMessage: mype < 0\n");

    id.instanceID = getInstance();

    //ComlibPrintf("Instance ID = %d\n", getInstance());
    ComlibPrintf("%d: DoneInserting %d \n", CkMyPe(), msgQ.length());
    
    if(doneFlag == 0) {
        ComlibPrintf("%d:Waiting for previous iteration to Finish\n", 
                     CkMyPe());
        bufferedDoneInserting = 1;
        return;
    }
    
    if(routerID == USE_DIRECT) {
        DummyMsg *m = (DummyMsg *)CmiAlloc(sizeof(DummyMsg));
        memset((char *)m, 0, sizeof(DummyMsg)); 
        m->id.instanceID = getInstance();
        
        Done(m);
        return;
    }

    doneFlag = 0;
    bufferedDoneInserting = 0;

    id.refno ++;

    if(msgQ.length() == 0) {
        if(routerID == USE_DIRECT)
            return;

        DummyMsg * dummymsg = (DummyMsg *)CmiAlloc(sizeof(DummyMsg));
        ComlibPrintf("[%d] Creating a dummy message\n", CkMyPe());
        CmiSetHandler(dummymsg, CkpvAccess(RecvdummyHandle));
        
        MessageHolder *cmsg = new MessageHolder((char *)dummymsg, 
                                                     myPe, 
                                                     sizeof(DummyMsg));
        cmsg->isDummy = 1;
        cmsg->pelist = &myPe;
        cmsg->npes = 1;
        msgQ.push(cmsg);
    }

    router->EachToManyMulticastQ(id, msgQ);

    while(!recvQ.isEmpty()) {
        char *msg = recvQ.deq();
        RecvManyMsg(msg);
    }

    while(!procQ.isEmpty()) {
        char *msg = procQ.deq();
        ProcManyMsg(msg);
    }

    while(!dummyQ.isEmpty() > 0) {
        DummyMsg *m = dummyQ.deq();
        router->DummyEP(m->id, m->magic);
        CmiFree(m);
    }
}

void RouterStrategy::Done(DummyMsg *m){

    ComlibPrintf("%d: Finished iteration\n", CkMyPe());

    if(doneHandle > 0) {
        CmiSetHandler(m, doneHandle);
        CmiSyncSendAndFree(CkMyPe(), sizeof(DummyMsg), (char*)m);
    }
    else
        CmiFree(m);

    doneFlag = 1;

    if(bufferedDoneInserting)
        doneInserting();
}


//Implement it later while implementing checkpointing of Comlib
void RouterStrategy::pup(PUP::er &p){}

PUPable_def(RouterStrategy);
