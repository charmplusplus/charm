#include "htram.h"

HTram::HTram(CkCallback cb){
  endCb = cb;
  myPE = CkMyPe();
  msgBuffers = new HTramMessage*[CkNumNodes()];
  for(int i=0;i<CkNumNodes();i++)
    msgBuffers[i] = new HTramMessage();
}

void HTram::setCb(void (*func)(int)) {
  cb = func;
}

HTram::HTram(CkMigrateMessage* msg) {}

//one per node, message, fixed 
//Client inserts
void HTram::insertValue(int value, int dest_pe) {
  int destNode = dest_pe/CkNodeSize(0); //find safer way to find dest node,
  // node size is not always same
  HTramMessage *destMsg = msgBuffers[destNode];
  destMsg->buffer[destMsg->next].payload = value;
  destMsg->buffer[destMsg->next].destPe = dest_pe;
  destMsg->next++;

#ifdef DEBUG
  if(destMsg->next%1000 == 0) CkPrintf("\nPE-%d, BufSize = %d\n", CkMyPe(), destMsg->next);
#endif

  if(destMsg->next == BUFSIZE) {
#ifdef DEBUG
    CkPrintf("\nPE-%d, Flushing", CkMyPe());
#endif
    nodeGrpProxy[destNode].receive(destMsg);
    msgBuffers[destNode] = new HTramMessage();
  }
}

void HTram::tflush() {
  for(int i=0;i<CkNumNodes();i++) {
    nodeGrpProxy[i].receive(msgBuffers[i]);
    msgBuffers[i] = new HTramMessage();
  }
}


HTramRecv::HTramRecv(){
}

HTramRecv::HTramRecv(CkMigrateMessage* msg) {}

void HTramRecv::receive(HTramMessage* agg_message) {
  //broadcast to each PE and decr refcount
  //nodegroup //reference from group
  
  for(int i=CkNodeFirst(CkMyNode()); i < CkNodeFirst(CkMyNode())+CkNodeSize(0);i++) {
    HTramMessage* tmpMsg = new HTramMessage(agg_message->next, agg_message->buffer);
    htramProxy[i].receivePerPE(tmpMsg);
  }
}

void HTram::receivePerPE(HTramMessage* msg) {
  int limit = msg->next;
  for(int i=0;i<limit;i++) {
    if(msg->buffer[i].destPe == myPE) {
      cb(msg->buffer[i].payload);
    }
  }
  contribute(endCb);
}

#include "htram.def.h"

