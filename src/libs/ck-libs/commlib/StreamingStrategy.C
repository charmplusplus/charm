#include "StreamingStrategy.h"
#include "MsgPacker.h"

StreamingStrategy::StreamingStrategy(int periodMs,int bufferMax_)
	: PERIOD(periodMs), bufferMax(bufferMax_)
{
    streamingMsgBuf=NULL;
    streamingMsgCount=NULL;
}

void StreamingStrategy::insertMessage(CharmMessageHolder *cmsg) {

    ComlibPrintf("StramingStrategy: InsertMessage\n");

    int pe=cmsg->dest_proc;
    streamingMsgBuf[pe].enq(cmsg);
    streamingMsgCount[pe]++;
    if (streamingMsgCount[pe]>bufferMax) flushPE(pe);
}

void StreamingStrategy::doneInserting(){
    ComlibPrintf("[%d] In Streaming strategy::doneInserting\n", CkMyPe());
    //Do nothing
}

/// Send off all accumulated messages for this PE:
void StreamingStrategy::flushPE(int pe) {
    if(streamingMsgCount[pe] == 0)
        return; //Nothing to do.

    CharmMessageHolder *cmsg, *toBeDeleted = NULL;
    
    if(shortMsgPackingFlag){
        MsgPacker mpack(streamingMsgBuf[pe], streamingMsgCount[pe]);
        CombinedMessage *msg; 
        int size;
        mpack.getMessage(msg, size);
        
        CmiSyncSendAndFree(pe, size, (char *)msg);
        streamingMsgCount[pe] = 0;
    }
    else {
        // Build a CmiMultipleSend list of messages to be sent off:
        int msg_count=streamingMsgCount[pe], msg_pe=0;
        char **msgComps = new char*[msg_count];
        int *sizes = new int[msg_count];
        while (!streamingMsgBuf[pe].isEmpty()) {
            cmsg = streamingMsgBuf[pe].deq();
            char *msg = cmsg->getCharmMessage();
            envelope *env = UsrToEnv(msg);
            sizes[msg_pe] = env->getTotalsize();
            msgComps[msg_pe] = (char *)env;
            msg_pe++;
            
            // Link cmsg into the toBeDeleted list:
            cmsg->next = toBeDeleted;
            toBeDeleted = cmsg;            
        }
    
        if (msg_count!=msg_pe) 
            CkAbort("streamingMsgCount doesn't match streamingMsgBuf!\n");
    
        CmiMultipleSend(pe, msg_count, sizes, msgComps);
        delete [] msgComps;
        delete [] sizes;
        streamingMsgCount[pe] = 0;
                
        // Traverse the tobeDeleted list:
        cmsg = toBeDeleted;
        while (toBeDeleted) {
            toBeDeleted = toBeDeleted->next;
            CkFreeMsg(cmsg->getCharmMessage());
            delete cmsg;
            cmsg = toBeDeleted;            
        }     
    }
}

void StreamingStrategy::periodicFlush(){
    for (int pe=0; pe<CkNumPes(); pe++) flushPE(pe);
}

/// This routine is called via CcdCallFnAfter to flush all messages:
static void call_delayFlush(void *arg,double curWallTime){
    StreamingStrategy *s=(StreamingStrategy *)arg;
    s->periodicFlush();
    s->registerFlush(); //Set ourselves up to be called again
}

void StreamingStrategy::registerFlush(void) {
    // CkPrintf("[%d] Will call function again every %d ms\n",CkMyPe(),PERIOD);
    CcdCallFnAfter((CcdVoidFn)call_delayFlush, (void *)this, PERIOD);
}

/// This routine is called via CcdCallOnCondition to flush all messages:
static void call_idleFlush(void *arg,double curWallTime){
    StreamingStrategy *s=(StreamingStrategy *)arg;
    s->periodicFlush();
}

// When we're finally ready to go, register for timeout and idle flush.
void StreamingStrategy::beginProcessing(int ignored) {
    registerFlush();
    CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,(CcdVoidFn)call_idleFlush,
                           (void *)this);
}

void StreamingStrategy::pup(PUP::er &p){
    p | PERIOD;
    p | bufferMax;
    p | shortMsgPackingFlag;

    if(p.isUnpacking()) {
        streamingMsgBuf = new CkQ<CharmMessageHolder *>[CkNumPes()];
        streamingMsgCount = new int[CkNumPes()];
        for(int count = 0; count < CkNumPes(); count ++)
            streamingMsgCount[count] = 0;
    }
}

//PUPable_def(StreamingStrategy);
