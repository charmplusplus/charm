#include "PrioStreaming.h"
#include "MsgPacker.h"

PrioStreaming::PrioStreaming(int periodMs,int bufferMax_, int prio)
    : StreamingStrategy(periodMs, bufferMax_), basePriority(prio)
{
}

void PrioStreaming::insertMessage(CharmMessageHolder *cmsg) {

    ComlibPrintf("Prio Straming: InsertMessage %d, %d\n",  
                 PERIOD, bufferMax);

    int pe=cmsg->dest_proc;
    char* msg = cmsg->getCharmMessage();
    envelope *env = UsrToEnv(msg);
    int msg_prio = *(int*)env->getPrioPtr();

    if(streamingMsgCount[pe] == 0) 
        minPrioVec[pe] = msg_prio;
    else if(minPrioVec[pe] > msg_prio)
        minPrioVec[pe] = msg_prio;

    streamingMsgBuf[pe].enq(cmsg);
    streamingMsgCount[pe]++;   

    if(msg_prio <= basePriority)
        flushPE(pe);

    if (streamingMsgCount[pe] > bufferMax) 
        flushPE(pe);
}

void PrioStreaming::pup(PUP::er &p){

    StreamingStrategy::pup(p);
    p | basePriority;

    if(p.isUnpacking())
        minPrioVec.resize(CkNumPes());
}
