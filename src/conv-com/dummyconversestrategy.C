// #ifdef filippo

// #include "RouterStrategy.h"

// RouterStrategy::RouterStrategy() : Strategy(){
// }

// void RouterStrategy::insertMessage(MessageHolder *cmsg){
//     ComlibPrintf("Sending Directly\n");
//     char *msg = cmsg->getMessage();
//     CmiSyncSendAndFree(cmsg->dest_proc, cmsg->size, msg);
//     delete cmsg;
// }

// void RouterStrategy::doneInserting(){
// }

// void RouterStrategy::pup(PUP::er &p){}

// #endif
