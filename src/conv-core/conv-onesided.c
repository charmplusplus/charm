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

#ifndef _CONV_ONESIDED_C_
#define _CONV_ONESIDED_C_

struct CmiCb {
  CmiRdmaCallbackFn fn;
  void *param;
};
typedef struct CmiCb CmiCb;

/* This is an object which is kindof a handle to a get or put call.
 * This object is polled in test to verify if the get/put has completed.
 * The completed bit is always set if the operation is complete
 */
struct CmiRMA {
  int type;
  union {
    int completed;
    CmiCb *cb;
  } ready;
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

int CmiRegisterMemory(void *addr, unsigned int size){
  //in the emulated version, this is blank
  return 1;
}

int CmiUnRegisterMemory(void *addr, unsigned int size){
  //in the emulated version, this is blank
  return 1;
}

void handlePutSrc(void *msg) {
  CmiRMA* stat = ((CmiRMAMsg*)msg)->stat;
  if(stat->type==1) {
    stat->ready.completed = 1;
    //the handle is active, and the user will clean it
  }
  else {
    (*(stat->ready.cb->fn))(stat->ready.cb->param);
    CmiFree(stat->ready.cb);
    CmiFree(stat); //clean up the internal handle
  }
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
  void *msgRma = (void*)CmiAlloc(sizeRma);
  RMAPutMsg *context = (RMAPutMsg*)msgRma;

  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 1;
  context->stat->ready.completed = 0;
  void* putdata = (void*)(((char*)(msgRma))+sizeof(RMAPutMsg));
  memcpy(putdata,Saddr,size);
  void *stat = context->stat;

  CmiSetHandler(msgRma,putDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);

  return stat;
}

void CmiPutCb(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size, CmiRdmaCallbackFn fn, void *param) {
  unsigned int sizeRma = sizeof(RMAPutMsg)+size;
  void *msgRma = (void*)CmiAlloc(sizeRma);
  RMAPutMsg *context = (RMAPutMsg*)msgRma;

  context->Saddr = Saddr;
  context->Taddr = Taddr;
  context->size = size;
  context->targetId = targetId;
  context->sourceId = sourceId;
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 0;
  context->stat->ready.cb = (CmiCb*)CmiAlloc(sizeof(CmiCb));
  context->stat->ready.cb->fn = fn;
  context->stat->ready.cb->param = param;
  void* putdata = (void*)(((char*)(msgRma))+sizeof(RMAPutMsg));
  memcpy(putdata,Saddr,size);

  CmiSetHandler(msgRma,putDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);
  return;
}

void handleGetSrc(void *msg) {
  RMAPutMsg *context = (RMAPutMsg*)msg;
  void* putdata = (void*)(((char*)(msg))+sizeof(RMAPutMsg));
  //copy the message
  memcpy(context->Saddr,putdata,context->size);
  //note the ack
  if(context->stat->type==1) {
    context->stat->ready.completed = 1;
    //the handle will be used still, and the user will clean it
  }
  else {
    (*(context->stat->ready.cb->fn))(context->stat->ready.cb->param);
    CmiFree(context->stat->ready.cb);
    CmiFree(context->stat); //clean up the internal handle
  }
  CmiFree(msg);
  return;
}

void handleGetDest(void *msg) {
  RMAPutMsg *context1 = (RMAPutMsg*)msg;
  unsigned int sizeRma = sizeof(RMAPutMsg)+context1->size;
  void *msgRma = (void*)CmiAlloc(sizeRma);
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
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 1;
  context->stat->ready.completed = 0;
  void *stat = context->stat;

  CmiSetHandler(msgRma,getDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);
  return stat;
}

void CmiGetCb(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size, CmiRdmaCallbackFn fn, void *param) {
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
  context->stat = (CmiRMA*)CmiAlloc(sizeof(CmiRMA));
  context->stat->type = 0;
  context->stat->ready.cb = (CmiCb*)CmiAlloc(sizeof(CmiCb));
  context->stat->ready.cb->fn = fn;
  context->stat->ready.cb->param = param;

  CmiSetHandler(msgRma,getDestHandler);
  CmiSyncSendAndFree(targetId,sizeRma,msgRma);
  return;
}


int CmiWaitTest(void *obj){
  CmiRMA *stat = (CmiRMA*)obj;
  return stat->ready.completed;
}

#endif
#endif
#endif

