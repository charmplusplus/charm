#include "PrioStreaming.h"
#include "MsgPacker.h"

PrioStreaming::PrioStreaming(int periodMs,int bufferMax_, int prio, 
			     int msgSizeMax_, int bufSizeMax_)
    : StreamingStrategy(periodMs, bufferMax_, msgSizeMax_, bufSizeMax_), basePriority(prio)
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
    bufSize[pe]+=cmsg->getSize();

    if(msg_prio <= basePriority)
        flushPE(pe);

    if (streamingMsgCount[pe] > bufferMax || bufSize[pe] > bufSizeMax) 
        flushPE(pe);
}

void PrioStreaming::pup(PUP::er &p){

    StreamingStrategy::pup(p);
    p | basePriority;

    if(p.isUnpacking())
        minPrioVec.resize(CkNumPes());
}
