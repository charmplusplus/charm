#include "StreamingStrategy.h"

void call_endIteration(void *arg){
    //    CkPrintf("Calling Flush\n");
    ((StreamingStrategy *)arg)->periodicFlush();
    return;
}

StreamingStrategy::StreamingStrategy(int period){
    /*
      streamingMsgBuf = new CharmMessageHolder*[CkNumPes()];
      streamingMsgCount = new int[CkNumPes()];
      for(int count = 0; count < CkNumPes(); count ++){
      streamingMsgBuf[count] = NULL;
      streamingMsgCount[count] = 0;
      }
    */

    PERIOD = period;
}

void StreamingStrategy::insertMessage(CharmMessageHolder *cmsg){
    cmsg->next = streamingMsgBuf[cmsg->dest_proc];
    streamingMsgBuf[cmsg->dest_proc] = cmsg;
    streamingMsgCount[cmsg->dest_proc] ++;
}

void StreamingStrategy::doneInserting(){
    ComlibPrintf("[%d] In Streaming strategy\n", CkMyPe());
    //Do nothing
}

void StreamingStrategy::periodicFlush(){
    CharmMessageHolder *cmsg;
        
    //    CkPrintf("In Periodic Flush\n");

    int buf_size = 0, count = 0;
    for(count = 0; count < CkNumPes(); count ++) {
        //        CkPrintf("Streaming Strategy: Processing proc[%d], %d \n", 
        //        count, streamingMsgCount[count]);
        
        if(streamingMsgCount[count] == 0)
            continue;
        
        cmsg = streamingMsgBuf[count];
        char ** msgComps = new char*[streamingMsgCount[count]];
        int *sizes = new int[streamingMsgCount[count]];
        
        int msg_count = 0;
        while (cmsg != NULL) {
            char * msg = cmsg->getCharmMessage();
            envelope * env = UsrToEnv(msg);
            sizes[msg_count] = env->getTotalsize();
            msgComps[msg_count] = (char *)env;
            
            cmsg = cmsg->next;
            msg_count ++;
        }
        
        CmiMultipleSend(count, streamingMsgCount[count], sizes, msgComps);
        delete [] msgComps;
        delete [] sizes;
        
        cmsg = streamingMsgBuf[count];
        CharmMessageHolder *prev = NULL;
        
        while(cmsg != NULL){
            prev = cmsg;
            cmsg = cmsg->next;
            delete prev;
        }
        
        streamingMsgCount[count] = 0;        
        streamingMsgBuf[count] = NULL;
    }
    
    CcdCallFnAfter(call_endIteration, (void *)this, PERIOD);
}

void StreamingStrategy::pup(PUP::er &p){
    p | PERIOD;
    
    if(p.isUnpacking()) {
        streamingMsgBuf = new CharmMessageHolder*[CkNumPes()];
        streamingMsgCount = new int[CkNumPes()];
        for(int count = 0; count < CkNumPes(); count ++){
            streamingMsgBuf[count] = NULL;
            streamingMsgCount[count] = 0;
        }
        periodicFlush();
    }
}

PUPable_def(StreamingStrategy);
