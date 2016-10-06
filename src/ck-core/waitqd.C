#include "waitqd.h"

/* readonly */ 
CkChareID _waitqd_qdhandle;

extern "C" void CkWaitQD(void) {
  CProxy_waitqd_QDChare qdchareproxy(_waitqd_qdhandle);
  qdchareproxy.waitQD();
}
  
waitqd_QDChare::waitqd_QDChare(CkArgMsg *m) {
  waitStarted = false;
  threadList = 0;
  _waitqd_qdhandle = thishandle;
  delete m;
}

void waitqd_QDChare::waitQD(void) {
  if (waitStarted) {
    CdsFifo_Enqueue((CdsFifo)threadList, (void *)CthSelf());
  } else {
    waitStarted = true;
    threadList = (void*) CdsFifo_Create();
    CdsFifo_Enqueue((CdsFifo) threadList, (void *)CthSelf());
    CkStartQD(CkIndex_waitqd_QDChare::onQD((CkQdMsg*)0), &thishandle);
  }
  CthSuspend();
}

void waitqd_QDChare::onQD(CkQdMsg *ckqm) {
  CthThread pthr;
  while(!CdsFifo_Empty((CdsFifo) threadList)) {
    pthr = (CthThread)CdsFifo_Dequeue((CdsFifo) threadList);
    CthAwaken(pthr);
  }
  CdsFifo_Destroy((CdsFifo) threadList);
  threadList = 0;
  waitStarted = false;
  delete ckqm;
}

#include "waitqd.def.h"
