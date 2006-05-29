/*
 * The emulated version of Get/Put in converse for 
 * messaging hardware that does not have hardware 
 * support for one sided communication
 * Author: Nilesh
 * Date: 05/17/2006
 */
#include "conv-onesided.h"

#ifdef __ONESIDED_IMPL
#ifdef __ONESIDED_NO_HARDWARE

/* This is an object which is kindof a handle to a get or put call.
 * If the machine layer supports a callback, the 'fn' field needs to
 * be populated, and it is called when the get/put request completes.
 * This object is polled in test to verify if the get/put has completed.
 * The ready bit is always set if the operation is complete
 */
struct CmiRMA {
  int completed;
};
typedef struct CmiRMA CmiRMA;

struct CmiRMAMsg {
  char core[CmiMsgHeaderSizeBytes];
  CmiRMA* stat;
};
typedef struct CmiRMAMsg CmiRMAMsg;

struct RMAPutMsg {
  char core[CmiMsgHeaderSizeBytes];
  void *Saddr;
  void *Taddr;
  unsigned int size;
  unsigned int targetId;
  unsigned int sourceId;
  CmiRMA *stat;
};
typedef struct RMAPutMsg RMAPutMsg;

/* Registering a piece of memeory implies making it avaialble(visible) to 
 * all other processors.
 * Each process maintains a data structure for each registered memory,
 * each remote processor that wants to talk to this memory will have
 * a region of its own memory mapped into this.
 */
int CmiRegisterMemory(void *addr, unsigned int size){
  //in the emulated version, this is blank
  return 1;
}

/* Unregister Memory is expensive anyway, so a linked list
 * data structure just means deletion will be slightly 
 * more expensive, won't be too bad, as this list will
 * not be long.
 * Match this address to the entry in the list and delete it
 */
int CmiUnRegisterMemory(void *addr, unsigned int size){
  //in the emulated version, this is blank
  return 1;
}

void handlePutSrc(void *msg) {
  CmiRMA* stat = ((CmiRMAMsg*)msg)->stat;
  stat->completed = 1;
  CmiFree(msg);
  return;
}

void handlePutDest(void *msg) {
  RMAPutMsg *context = (RMAPutMsg*)msg;
  void* putdata = (void*)(((char*)(msg))+sizeof(RMAPutMsg));
  unsigned int sizeRmaStat = sizeof(CmiRMAMsg);
  void *msgRmaStat;
  //copy the message
  memcpy(context->Taddr,putdata,context->size);
  //send the ack
  msgRmaStat = (void*)CmiAlloc(sizeRmaStat);
  ((CmiRMAMsg*)msgRmaStat)->stat = context->stat;
  CmiSetHandler(msgRmaStat,putSrcHandler);
  CmiSyncSendAndFree(context->sourceId,sizeRmaStat,msgRmaStat);

  CmiFree(msg);
  return;
}

void *CmiPut(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size) {
  unsigned int sizeRma = sizeof(RMAPutMsg)+size;
  void *msgRma = (void*)malloc(sizeRma);
  RMAPutMsg *context = (RMAPutMsg*)msgRma;
  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->stat = (CmiRMA*)malloc(sizeof(CmiRMA));
  context->stat->completed = 0;
  void* putdata = (void*)(((char*)(msgRma))+sizeof(RMAPutMsg));
  memcpy(putdata,Saddr,size);
  CmiSetHandler(msgRma,putDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);
  return (void*)(context->stat);
}

void handleGetSrc(void *msg) {
  RMAPutMsg *context = (RMAPutMsg*)msg;
  void* putdata = (void*)(((char*)(msg))+sizeof(RMAPutMsg));
  //copy the message
  memcpy(context->Saddr,putdata,context->size);
  //note the ack
  context->stat->completed = 1;
  CmiFree(msg);
  return;
}

void handleGetDest(void *msg) {
  RMAPutMsg *context1 = (RMAPutMsg*)msg;
  unsigned int sizeRma = sizeof(RMAPutMsg)+context1->size;
  void *msgRma = (void*)malloc(sizeRma);
  RMAPutMsg *context = (RMAPutMsg*)msgRma;
  memcpy(context,context1,sizeof(RMAPutMsg));
  void* putdata = (void*)(((char*)(msgRma))+sizeof(RMAPutMsg));
  memcpy(putdata,context->Taddr,context->size);
  CmiSetHandler(msgRma,getSrcHandler);
  CmiSyncSendAndFree(context->sourceId,sizeRma,msgRma);
  CmiFree(msg);
  return;
}

void *CmiGet(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size) {
  unsigned int sizeRma;
  char *msgRma;
  RMAPutMsg *context;
  sizeRma = sizeof(RMAPutMsg);
  msgRma = (char*)CmiAlloc(sizeRma*sizeof(char));

  context = (RMAPutMsg*)msgRma;
  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->stat = (CmiRMA*)malloc(sizeof(CmiRMA));
  context->stat->completed = 0;

  CmiSetHandler(msgRma,getDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);

  return (void*)(context->stat);
}

int CmiWaitTest(void *obj){
  return ((CmiRMA*)obj)->completed;
}

#endif
#endif

