////////////////////////////////////////////////////
//
//  waitqd.C
//
//
//
//  Author: Michael Lang
//  Created: 7/15/99
//
////////////////////////////////////////////////////

#include "waitqd.h"
#include "fifo.h"

/* readonly */ 
CkChareID waitqd_qdhandle;

extern "C" void CkWaitQD(void) {
  CProxy_waitqd_QDChare qdchareproxy(waitqd_qdhandle);
  qdchareproxy.waitQD();
}
  

waitqd_QDChare::waitqd_QDChare(CkArgMsg *ckam) {
  waitStarted = 0;
  threadList = 0;
  waitqd_qdhandle = thishandle;
}

void waitqd_QDChare::waitQD(void) {
  if (waitStarted == 1) {
    FIFO_EnQueue((FIFO_QUEUE*)threadList, (void *)CthSelf());
  } else {
    waitStarted = 1;
    threadList = (void*) FIFO_Create();
    FIFO_EnQueue((FIFO_QUEUE*) threadList, (void *)CthSelf());
    CkStartQD(EntryIndex(waitqd_QDChare, onQD, CkQdMsg), &thishandle);
  }
  CthSuspend();
}

void waitqd_QDChare::onQD(CkQdMsg *ckqm) {
  CthThread *pthr;
  while(!FIFO_Empty((FIFO_QUEUE*) threadList)) {
    FIFO_DeQueue((FIFO_QUEUE*) threadList, (void**) &pthr);
    CthAwaken(*pthr);
  }
  FIFO_Destroy((FIFO_QUEUE*) threadList);
  threadList = 0;
  waitStarted = 0;
}

#include "waitqd.def.h"
