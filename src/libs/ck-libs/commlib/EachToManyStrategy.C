#include "EachToManyStrategy.h"

CpvExtern(int, RecvmsgHandle);
CpvExtern(int, RecvdummyHandle);

void setReverseMap(int *procMap, int *pelist, int npes){
    
    for(int pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;
    
    for(int pcount = 0; pcount < npes; pcount++) 
        procMap[pelist[pcount]] = pcount;
}

EachToManyStrategy::EachToManyStrategy(int substrategy){
    ComlibPrintf("In constructor, %d\n", substrategy);
    routerID = substrategy;
    messageBuf = 0;
    messageCount = 0;

    ComlibPrintf("Before instance\n");
    comid = ComlibInstance(routerID, CkNumPes());
    this->npes = npes;
    ComlibPrintf("After instance\n");

    procMap = new int[CkNumPes()];
    for(int count = 0; count < CkNumPes(); count ++){
        procMap[count] = count;
    }
    ComlibPrintf("After Constructor\n");
}

EachToManyStrategy::EachToManyStrategy(int substrategy, int npes, int *pelist){
    routerID = substrategy;
    messageBuf = NULL;
    messageCount = 0;

    comid = ComlibInstance(routerID, CkNumPes());
    comid = ComlibEstablishGroup(comid, npes, pelist);

    procMap = new int[CkNumPes()];
    setReverseMap(procMap, pelist, npes);
}

void EachToManyStrategy::insertMessage(CharmMessageHolder *cmsg){

    ComlibPrintf("EachToMany: insertMessage\n");

    messageBuf->enq(cmsg);
}

void EachToManyStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    //ComlibPrintf("%d:Setting Num Deposit to %d\n", CkMyPe(), messageCount);

    if((messageCount == 0) && (CkNumPes() > 0)) {
        DummyMsg * dummymsg = new DummyMsg;
        
        ComlibPrintf("Creating a dummy message\n");
        
        CmiSetHandler(UsrToEnv(dummymsg), 
                      CpvAccess(RecvdummyHandle));
        
        messageBuf->enq(new CharmMessageHolder((char *)dummymsg, CkMyPe()));
        messageCount ++;
    }

    NumDeposits(comid, messageCount);
    
    while(!messageBuf->isEmpty()) {
	CharmMessageHolder *cmsg = messageBuf->deq();
        char * msg = cmsg->getCharmMessage();
        ComlibPrintf("Calling EachToMany %d %d %d procMap=%d\n", 
                     UsrToEnv(msg)->getTotalsize(), CkMyPe(), 
                     cmsg->dest_proc, procMap[cmsg->dest_proc]);
        EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
                            UsrToEnv(msg), 1, 
                            &procMap[cmsg->dest_proc]);
        CharmMessageHolder *prev = cmsg;
        cmsg = cmsg->next;
        if(prev != NULL)
            delete prev;                //foobar getrid of the delete
    }
    messageCount = 0;
}

void EachToManyStrategy::pup(PUP::er &p){

  ComlibPrintf("Each To many :: pup\n");

    Strategy::pup(p);
    
    p | messageCount;
    p | routerID;
    p | comid;
    p | npes;
    p | messageCount;
    
    if(p.isUnpacking()) 
        procMap = new int[CkNumPes()];
        
    p | procMap;

    if(p.isUnpacking()){
      messageBuf = new CkQ<CharmMessageHolder *>;
    }
}

PUPable_def(EachToManyStrategy); 
