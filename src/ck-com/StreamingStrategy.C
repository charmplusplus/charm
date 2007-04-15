#include "StreamingStrategy.h"
#include "MsgPacker.h"

void StreamingHandlerFn(void *msg) {
    CombinedMessage hdr;
    
    ComlibPrintf("In streaming handler fn\n");

    PUP_fromCmiAllocMem fp(msg);
    fp | hdr;
    
    for(int count = 0; count < hdr.nmsgs; count ++) {
        char *msg;
        fp.pupCmiAllocBuf((void **)&msg);
        int size = ((envelope *)msg)->getTotalsize(); //SIZEFIELD(msg);
        CmiSyncSendAndFree(CkMyPe(), size, msg);
    }
    CmiFree(msg);
    return;
}

StreamingStrategy::StreamingStrategy(int periodMs, int bufferMax_, 
				     int msgSizeMax_, int bufSizeMax_)
    : PERIOD(periodMs), bufferMax(bufferMax_), msgSizeMax(msgSizeMax_), 
      bufSizeMax(bufSizeMax_), CharmStrategy()
{
    streamingMsgBuf = NULL;
    streamingMsgCount = NULL;
    bufSize = NULL;
    shortMsgPackingFlag = CmiFalse;
    idleFlush = CmiTrue;
    streaming_handler_id = 0;
    setType(ARRAY_STRATEGY);
}

StreamingStrategy::StreamingStrategy(double periodMs, int bufferMax_, 
				     int msgSizeMax_, int bufSizeMax_)
    : PERIOD(periodMs), bufferMax(bufferMax_), msgSizeMax(msgSizeMax_), 
      bufSizeMax(bufSizeMax_), CharmStrategy()
{
    streamingMsgBuf = NULL;
    streamingMsgCount = NULL;
    bufSize = NULL;
    shortMsgPackingFlag = CmiFalse;
    idleFlush = CmiTrue;
    streaming_handler_id = 0;
    setType(ARRAY_STRATEGY);
}

void StreamingStrategy::insertMessage(CharmMessageHolder *cmsg) {

    int pe=cmsg->dest_proc;
    char *msg = cmsg->getCharmMessage();
    envelope *env = UsrToEnv(msg);
    int size = env->getTotalsize();

    if(size > msgSizeMax) {//AVOID COPYING
        ComlibPrintf("StreamingStrategy::inserSessage: direct send\n");
        CmiSyncSendAndFree(pe, size, (char *)env);
        delete cmsg;
        return;
    }

    ComlibPrintf("[%d] StreamingStrategy::insertMessage: buffering t=%g, n=%d, d=%d, s=%d\n",  
                 CkMyPe(), PERIOD, bufferMax, pe, size);
    
    streamingMsgBuf[pe].enq(cmsg);
    streamingMsgCount[pe]++;
    bufSize[pe]+=cmsg->getSize();
    if (streamingMsgCount[pe] > bufferMax || bufSize[pe] > bufSizeMax) {
      flushPE(pe);
    }
}

void StreamingStrategy::doneInserting() {
  ComlibPrintf("[%d] In Streaming strategy::doneInserting\n", CkMyPe());
  //Do nothing

  periodicFlush();
}

/// Send off all accumulated messages for this PE:
void StreamingStrategy::flushPE(int pe) {

  //CkPrintf("Checking %d\n", pe);

  if(streamingMsgCount[pe] == 0)
      return; //Nothing to do.
  
  CharmMessageHolder *cmsg, *toBeDeleted = NULL;
  int size = 0;
  if(shortMsgPackingFlag){
      MsgPacker mpack(streamingMsgBuf[pe], streamingMsgCount[pe]);
      CombinedMessage *msg; 
      mpack.getMessage(msg, size);
      ComlibPrintf("[%d] StreamingStrategy::flushPE: packed %d short messages to %d\n", 
                   CkMyPe(), streamingMsgCount[pe], pe); 
      CmiSyncSendAndFree(pe, size, (char *)msg);
      streamingMsgCount[pe] = 0;
      bufSize[pe] = 0;
  }
  else {
      
    // Build a CmiMultipleSend list of messages to be sent off:
    int msg_count=streamingMsgCount[pe], msg_pe=0;
    if(msg_count == 1) {
        cmsg = streamingMsgBuf[pe].deq();
        char *msg = cmsg->getCharmMessage();
        envelope *env = UsrToEnv(msg);
        int size = env->getTotalsize();
        CmiSyncSendAndFree(pe, size, (char *)env);
        ComlibPrintf("[%d] StreamingStrategy::flushPE: one message to %d\n", 
                     CkMyPe(), pe);            
        delete cmsg;
        streamingMsgCount[pe] = 0;
	bufSize[pe] = 0;
        return;
    }
    /*
    char **msgComps = new char*[msg_count];
    int *sizes = new int[msg_count];
    ComlibPrintf("[%d] StreamingStrategy::flushPE: %d messages to %d\n", 
                 CkMyPe(), msg_count, pe);            
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
    
    ComlibPrintf("--> Sending %d Messages to PE %d\n", msg_count, pe);
    
    CmiMultipleSend(pe, msg_count, sizes, msgComps);
    delete [] msgComps;
    delete [] sizes;
    streamingMsgCount[pe] = 0;
    
    // Traverse the tobeDeleted list:
    cmsg = toBeDeleted;
    while (toBeDeleted) {
        toBeDeleted = (CharmMessageHolder *)toBeDeleted->next;
        CkFreeMsg(cmsg->getCharmMessage());
        delete cmsg;
        cmsg = toBeDeleted;            
    }     
    */
    
    PUP_cmiAllocSizer sp;
    CombinedMessage hdr;
    
    sp | hdr;

    int nmsgs = streamingMsgCount[pe];
    int count;
    for(count = 0; count < nmsgs; count++) {
        cmsg = streamingMsgBuf[pe][count];
        char *msg = cmsg->getCharmMessage();
        envelope *env = UsrToEnv(msg);
        size = env->getTotalsize();
        
        sp.pupCmiAllocBuf((void **)&env, size);
    }
    
    char *newmsg = (char *)CmiAlloc(sp.size());
    PUP_toCmiAllocMem mp(newmsg);
    
    hdr.aid.setZero();
    hdr.srcPE = CkMyPe();
    hdr.nmsgs = nmsgs;
    mp | hdr;
    
    for(count = 0; count < nmsgs; count++) {
        cmsg = streamingMsgBuf[pe][count];
        char *msg = cmsg->getCharmMessage();
        envelope *env = UsrToEnv(msg);
        size = env->getTotalsize();
        
        mp.pupCmiAllocBuf((void **)&env, size);
    }

    for(count = 0; count < nmsgs; count++) {
        cmsg = streamingMsgBuf[pe].deq();
        CkFreeMsg(cmsg->getCharmMessage());
        delete cmsg;
    }    
    
    streamingMsgCount[pe] = 0;
    bufSize[pe] = 0;
    CmiSetHandler(newmsg, streaming_handler_id);
    CmiSyncSendAndFree(pe, sp.size(), newmsg); 
  }
}

void StreamingStrategy::periodicFlush() {
    for (int proc = 0; proc < CkNumPes(); proc++) 
        flushPE(proc);
}

struct MsgStruct {
    char header[CmiReservedHeaderSize];
    void *addr;
};


void testHandler(void *msg) {
    StreamingStrategy *s;

    MsgStruct *mstruct = (MsgStruct *)msg;

    s = (StreamingStrategy *) (mstruct->addr);
    s->periodicFlush();

    CmiSyncSendAndFree(CkMyPe(), sizeof(MsgStruct), (char *)msg);
}

/// This routine is called via CcdCallFnAfter to flush all messages:
static void call_delayFlush(void *arg,double curWallTime) {
    StreamingStrategy *s=(StreamingStrategy *)arg;
    s->periodicFlush();
    s->registerFlush(); //Set ourselves up to be called again
}

void StreamingStrategy::registerFlush(void) {
    //CkPrintf("[%d] Will call function again every %d ms\n",CkMyPe(),PERIOD);
    CcdCallFnAfterOnPE(call_delayFlush, (void *)this, PERIOD, CkMyPe());
}

/// This routine is called via CcdCallOnCondition to flush all messages:
static void call_idleFlush(void *arg,double curWallTime) {
    StreamingStrategy *s=(StreamingStrategy *)arg;
    s->periodicFlush();
}

// When we're finally ready to go, register for timeout and idle flush.
void StreamingStrategy::beginProcessing(int ignored) {
    registerFlush();
    //if(idleFlush)
    //  CcdCallOnConditionKeepOnPE(CcdPROCESSOR_BEGIN_IDLE,
    //                             (CcdVoidFn)call_idleFlush, 
    //                             (void *)this, CkMyPe());
    
    streaming_handler_id = CkRegisterHandler(StreamingHandlerFn);
    
    /*
      int handler = CkRegisterHandler(testHandler);
      
      MsgStruct *msg = (MsgStruct *)CmiAlloc(sizeof(MsgStruct));
      msg->addr = this;
      CmiSetHandler(msg, handler);
      
      CmiSyncSendAndFree(CkMyPe(), sizeof(MsgStruct), (char *)msg);
    */
}

void StreamingStrategy::pup(PUP::er &p){

  CharmStrategy::pup(p);
  p | PERIOD;
  p | bufferMax;
  p | msgSizeMax;
  p | shortMsgPackingFlag;
  p | bufSizeMax;
  p | idleFlush;
  p | streaming_handler_id;

  if(p.isUnpacking()) {
      streamingMsgBuf = new CkQ<CharmMessageHolder *>[CkNumPes()];
      streamingMsgCount = new int[CkNumPes()];
      bufSize = new int[CkNumPes()];
      for(int count = 0; count < CkNumPes(); count ++) {
	streamingMsgCount[count] = 0;
	bufSize[count] = 0;
      }
  }
}

//PUPable_def(StreamingStrategy);
