#include "StreamingStrategy.h"

void call_endIteration(void *arg){
  ((StreamingStrategy *)arg)->periodicFlush();
  return;
}

StreamingStrategy::StreamingStrategy(int period){
  /*
  streamingMsgBuf = new CkQ<CharmMessageHolder *>[CkNumPes()];
  streamingMsgCount = new int[CkNumPes()];
  for (int count=0; count<CkNumPes(); count++)
    streamingMsgBuf[count] = NULL;
  */
  PERIOD = period;
}

void StreamingStrategy::insertMessage(CharmMessageHolder *cmsg) {
  streamingMsgBuf[cmsg->dest_proc].enq(cmsg);
  streamingMsgCount[cmsg->dest_proc]++;
}

void StreamingStrategy::doneInserting(){
  ComlibPrintf("[%d] In Streaming strategy\n", CkMyPe());
  //Do nothing
}

void StreamingStrategy::periodicFlush(){
  CharmMessageHolder *cmsg, *toBeDeleted = NULL;
  envelope *env;
  char **msgComps, *msg;
  int *sizes, msg_count;

  for (int count=0; count<CkNumPes(); count++) {
    if(streamingMsgCount[count] == 0)
      continue;
    msgComps = new char*[streamingMsgCount[count]];
    sizes = new int[streamingMsgCount[count]];
    msg_count = 0;
    while (!streamingMsgBuf[count].isEmpty()) {
      cmsg = streamingMsgBuf[count].deq();
      msg = cmsg->getCharmMessage();
      env = UsrToEnv(msg);
      sizes[msg_count] = env->getTotalsize();
      msgComps[msg_count] = (char *)env;
      msg_count++;
      cmsg->next = toBeDeleted;
      toBeDeleted = cmsg;
    }
    CmiMultipleSend(count, streamingMsgCount[count], sizes, msgComps);
    delete [] msgComps;
    delete [] sizes;
    streamingMsgCount[count] = 0;        
  }
  cmsg = toBeDeleted;
  while (toBeDeleted) {
    toBeDeleted = toBeDeleted->next;
    delete cmsg;
    cmsg = toBeDeleted;
  }
  CcdCallFnAfter(call_endIteration, (void *)this, PERIOD);
}

void StreamingStrategy::pup(PUP::er &p){
  p | PERIOD;
  if(p.isUnpacking()) {
    streamingMsgBuf = new CkQ<CharmMessageHolder *>[CkNumPes()];
    streamingMsgCount = new int[CkNumPes()];
    for(int count = 0; count < CkNumPes(); count ++)
      streamingMsgCount[count] = 0;
    periodicFlush();
  }
}

//PUPable_def(StreamingStrategy);
