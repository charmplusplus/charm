#include "DummyStrategy.h"

DummyStrategy::DummyStrategy(int substrategy){
}

void DummyStrategy::insertMessage(CharmMessageHolder *cmsg){
    ComlibPrintf("Sending Directly\n");
    char *msg = cmsg->getCharmMessage();
    CmiSyncSendAndFree(cmsg->dest_proc, UsrToEnv(msg)->getTotalsize(), 
                       (char *)UsrToEnv(msg));
}

void DummyStrategy::doneInserting(){
}

