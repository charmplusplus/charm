#include "StreamingStrategy.h"

void call_endIteration(void *arg){
    ((Strategy *)arg)->doneInserting();
    return;
}

StreamingStrategy::StreamingStrategy(int period){
    streamingMsgBuf = new CharmMessageHolder*[CkNumPes()];
    streamingMsgCount = new int[CkNumPes()];
    for(int count = 0; count < CkNumPes(); count ++){
        streamingMsgBuf[count] = NULL;
        streamingMsgCount[count] = 0;
    }

    PERIOD = period;
    CcdCallFnAfter(call_endIteration, (void *)this, PERIOD);
}

void StreamingStrategy::insertMessage(CharmMessageHolder *cmsg){
    cmsg->next = streamingMsgBuf[cmsg->dest_proc];
    streamingMsgBuf[cmsg->dest_proc] = cmsg;
    streamingMsgCount[cmsg->dest_proc] ++;
}

void StreamingStrategy::doneInserting(){
    ComlibPrintf("[%d] In Streaming strategy\n", CkMyPe());

    CharmMessageHolder *cmsg;
        
    int buf_size = 0, count = 0;
    for(count = 0; count < CkNumPes(); count ++) {
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
    }
    
    for(count = 0; count < CkNumPes(); count ++){
        streamingMsgBuf[count] = NULL;
        streamingMsgCount[count] = 0;
    }
    
    CcdCallFnAfter(call_endIteration, (void *)this, PERIOD);
}
