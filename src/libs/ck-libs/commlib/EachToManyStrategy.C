#include "EachToManyStrategy.h"

extern int * procMap;
CpvExtern(int, RecvmsgHandle);
CpvExtern(int, RecvdummyHandle);

EachToManyStrategy::EachToManyStrategy(int substrategy){
    routerID = substrategy;
    messageBuf = NULL;
    messageCount = 0;

}

void EachToManyStrategy::insertMessage(CharmMessageHolder *cmsg){
    cmsg->next = messageBuf;
    messageBuf = cmsg;    
    messageCount ++;
}

void EachToManyStrategy::doneInserting(){
    ComlibPrintf("%d:Setting Num Deposit to %d\n", CkMyPe(), messageCount);

    if((messageCount == 0) && (CkNumPes() > 0)) {
        DummyMsg * dummymsg = new DummyMsg;
        
        ComlibPrintf("Creating a dummy message\n");
        
        CmiSetHandler(UsrToEnv(dummymsg), 
                      CpvAccess(RecvdummyHandle));
        
        messageBuf = new CharmMessageHolder((char *)dummymsg, CkMyPe());
        messageCount ++;
    }

    NumDeposits(comid, messageCount);
    
    CharmMessageHolder *cmsg = messageBuf;
    for(int count = 0; count < messageCount; count ++) {
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

void EachToManyStrategy::setID(comID id){
    comid = id;
}
