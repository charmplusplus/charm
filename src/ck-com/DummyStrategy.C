// #ifdef filippo

// #include "DummyStrategy.h"

// DummyStrategy::DummyStrategy() : CharmStrategy(){
// }

// void DummyStrategy::insertMessage(CharmMessageHolder *cmsg){
//     ComlibPrintf("Sending Directly\n");
//     char *msg = cmsg->getCharmMessage();
//     CmiSyncSendAndFree(cmsg->dest_proc, UsrToEnv(msg)->getTotalsize(), 
//                        (char *)UsrToEnv(msg));
//     delete cmsg;
// }

// void DummyStrategy::doneInserting(){
// }

// void DummyStrategy::pup(PUP::er &p){
//    CharmStrategy::pup(p);
// }

// //PUPable_def(DummyStrategy);

// #endif
