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
    streamingMsgBuf[pe].enq(cmsg);
    streamingMsgCount[pe]++;
    if (streamingMsgCount[pe] > bufferMax) {
        flushPE(pe);
        return;
    }
    
    char* msg = cmsg->getCharmMessage();
    envelope *env = UsrToEnv(msg);
    int msg_prio = *(int*)env->getPrioPtr();
    
    if(msg_prio <= basePriority)
        flushPE(pe);
}

void PrioStreaming::pup(PUP::er &p){

    StreamingStrategy::pup(p);

    p | basePriority;

}
