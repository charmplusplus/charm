/**
   @addtogroup ComlibConverseStrategy
   @{
   @file 
*/

#include "StreamingStrategy.h"
//#include "MsgPacker.h"
#include "pup_cmialloc.h"

/** The handler registerd to StreamingHandlerFn */
CpvDeclare(int, streaming_handler_id);
/**
 * Handler used to receive incoming combined messages, split them into the
 * individual messages and deliver all of them to the application.
 */
void StreamingHandlerFn(void *msg) {
    StreamingMessage hdr;
    
    ComlibPrintf("[%d] In streaming handler fn\n",CmiMyPe());

    PUP_fromCmiAllocMem fp(msg);
    fp | hdr;
    
    for(int count = 0; count < hdr.nmsgs; count ++) {
        char *msg;
        fp.pupCmiAllocBuf((void **)&msg);
        int size = SIZEFIELD(msg);
        CmiSyncSendAndFree(CmiMyPe(), size, msg);
    }
    CmiFree(msg);
    return;
}

StreamingStrategy::StreamingStrategy(int periodMs, int bufferMax_, 
				     int msgSizeMax_, int bufSizeMax_)
    : PERIOD(periodMs), bufferMax(bufferMax_), msgSizeMax(msgSizeMax_), 
      bufSizeMax(bufSizeMax_), Strategy() {
    streamingMsgBuf = NULL;
    streamingMsgCount = NULL;
    bufSize = NULL;
    //shortMsgPackingFlag = CmiFalse;
    idleFlush = CmiTrue;
    //streaming_handler_id = 0;
    setType(CONVERSE_STRATEGY);
}

StreamingStrategy::StreamingStrategy(double periodMs, int bufferMax_, 
				     int msgSizeMax_, int bufSizeMax_)
    : PERIOD(periodMs), bufferMax(bufferMax_), msgSizeMax(msgSizeMax_), 
      bufSizeMax(bufSizeMax_), Strategy() {
    streamingMsgBuf = NULL;
    streamingMsgCount = NULL;
    bufSize = NULL;
    //shortMsgPackingFlag = CmiFalse;
    idleFlush = CmiTrue;
    //streaming_handler_id = 0;
    setType(CONVERSE_STRATEGY);
}

void StreamingStrategy::insertMessage(MessageHolder *cmsg) {

    int pe=cmsg->dest_proc;
    char *msg = cmsg->getMessage();
    //envelope *env = UsrToEnv(msg);
    int size = cmsg->getSize(); // env->getTotalsize();

    if(size > msgSizeMax) {//AVOID COPYING
        ComlibPrintf("[%d] StreamingStrategy::insertMessage: to %d direct send %d\n",CmiMyPe(),pe,size);
        CmiSyncSendAndFree(pe, size, msg);
        delete cmsg;
        return;
    }

    ComlibPrintf("[%d] StreamingStrategy::insertMessage: buffering t=%g, n=%d, d=%d, s=%d\n",
		 CmiMyPe(), PERIOD, bufferMax, pe, size);
    
    streamingMsgBuf[pe].enq(cmsg);
    streamingMsgCount[pe]++;
    bufSize[pe]+=size;
    if (streamingMsgCount[pe] >= bufferMax || bufSize[pe] >= bufSizeMax) flushPE(pe);
}

void StreamingStrategy::doneInserting() {
  ComlibPrintf("[%d] StreamingStrategy::doneInserting\n", CmiMyPe());
  //Do nothing

  periodicFlush();
}

/// Send off all accumulated messages for this PE:
void StreamingStrategy::flushPE(int pe) {

  //CkPrintf("Checking %d\n", pe);

  if(streamingMsgCount[pe] == 0)
      return; //Nothing to do.
  
  MessageHolder *cmsg;
  int size = 0;
 

    // Build a CmiMultipleSend list of messages to be sent off:
    int msg_count=streamingMsgCount[pe];

    // If we have a single message we don't want to copy it
    if(msg_count == 1) {
        cmsg = streamingMsgBuf[pe].deq();
        char *msg = cmsg->getMessage();
        //envelope *env = UsrToEnv(msg);
        int size = cmsg->getSize();
        CmiSyncSendAndFree(pe, size, msg);
        ComlibPrintf("[%d] StreamingStrategy::flushPE: one message to %d\n", 
                     CmiMyPe(), pe);            
        delete cmsg;
        streamingMsgCount[pe] = 0;
	bufSize[pe] = 0;
        return;
    }
   
    
    PUP_cmiAllocSizer sp;
    StreamingMessage hdr;
    
    sp | hdr;

    int nmsgs = streamingMsgCount[pe];
    int count;
    for(count = 0; count < nmsgs; count++) {
        cmsg = streamingMsgBuf[pe][count];
        char *msg = cmsg->getMessage();
        //envelope *env = UsrToEnv(msg);
        size = cmsg->getSize();
        
        sp.pupCmiAllocBuf((void**)&msg, size);
    }
    
    char *newmsg = (char *)CmiAlloc(sp.size());
    PUP_toCmiAllocMem mp(newmsg);
    
    hdr.srcPE = CmiMyPe();
    hdr.nmsgs = nmsgs;
    mp | hdr;
    
    for(count = 0; count < nmsgs; count++) {
        cmsg = streamingMsgBuf[pe][count];
        char *msg = cmsg->getMessage();
        //envelope *env = UsrToEnv(msg);
        size = cmsg->getSize();
        
        mp.pupCmiAllocBuf((void**)&msg, size);
    }

    for(count = 0; count < nmsgs; count++) {
        cmsg = streamingMsgBuf[pe].deq();
        //CkFreeMsg(cmsg->getCharmMessage());
	CmiFree(cmsg->getMessage());
        delete cmsg;
    }    
    
    streamingMsgCount[pe] = 0;
    bufSize[pe] = 0;
    CmiSetHandler(newmsg, CpvAccess(streaming_handler_id));
    CmiSyncSendAndFree(pe, sp.size(), newmsg); 
    //}
}

void StreamingStrategy::periodicFlush() {
    for (int proc = 0; proc < CmiNumPes(); proc++) 
        flushPE(proc);
}

/*
struct MsgStruct {
    char header[CmiReservedHeaderSize];
    void *addr;
};


void testHandler(void *msg) {
    StreamingStrategy *s;

    MsgStruct *mstruct = (MsgStruct *)msg;

    s = (StreamingStrategy *) (mstruct->addr);
    s->periodicFlush();

    CmiSyncSendAndFree(CmiMyPe(), sizeof(MsgStruct), (char *)msg);
}
*/

/// This routine is called via CcdCallFnAfter to flush all messages:
static void call_delayFlush(void *arg,double curWallTime) {
    StreamingStrategy *s=(StreamingStrategy *)arg;
    s->periodicFlush();
    s->registerFlush(); //Set ourselves up to be called again
}

void StreamingStrategy::registerFlush(void) {
    //CkPrintf("[%d] Will call function again every %d ms\n",CmiMyPe(),PERIOD);
    CcdCallFnAfterOnPE(call_delayFlush, (void *)this, PERIOD, CmiMyPe());
}

/// This routine is called via CcdCallOnCondition to flush all messages:
static void call_idleFlush(void *arg,double curWallTime) {
    StreamingStrategy *s=(StreamingStrategy *)arg;
    s->periodicFlush();
}

// When we're finally ready to go, register for timeout and idle flush.
/*
void StreamingStrategy::beginProcessing(int ignored) {
    registerFlush();
    //if(idleFlush)
    //  CcdCallOnConditionKeepOnPE(CcdPROCESSOR_BEGIN_IDLE,
    //                             (CcdVoidFn)call_idleFlush, 
    //                             (void *)this, CmiMyPe());
    
    streaming_handler_id = CkRegisterHandler(StreamingHandlerFn);
    
//       int handler = CkRegisterHandler(testHandler);
      
//       MsgStruct *msg = (MsgStruct *)CmiAlloc(sizeof(MsgStruct));
//       msg->addr = this;
//       CmiSetHandler(msg, handler);
      
//       CmiSyncSendAndFree(CmiMyPe(), sizeof(MsgStruct), (char *)msg);

}
*/

void StreamingStrategy::pup(PUP::er &p){

  Strategy::pup(p);
  p | PERIOD;
  p | bufferMax;
  p | msgSizeMax;
  //p | shortMsgPackingFlag;
  p | bufSizeMax;
  p | idleFlush;
  //p | streaming_handler_id;

  if(p.isPacking() || p.isUnpacking()) {
      streamingMsgBuf = new CkQ<MessageHolder *>[CmiNumPes()];
      streamingMsgCount = new int[CmiNumPes()];
      bufSize = new int[CmiNumPes()];
      for(int count = 0; count < CmiNumPes(); count ++) {
	streamingMsgCount[count] = 0;
	bufSize[count] = 0;
      }
  }

  // packing is done once in processor 0, unpacking is done once in all processors except 0
  if (p.isPacking() || p.isUnpacking()) registerFlush();
}

PUPable_def(StreamingStrategy)

/*@}*/
