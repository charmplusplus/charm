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
CkGroupID waitGC_gchandle;

extern "C" void CkWaitQD(void) {
  CProxy_waitqd_QDChare qdchareproxy(waitqd_qdhandle);
  qdchareproxy.waitQD();
}
  
extern "C" CkGroupID CkCreateGroupSync(int cidx, int considx, void *msg)
{
  if(CkMyPe()==0) {
    return CkCreateGroup(cidx, considx, msg, 0, 0);
  } else {
    waitGC_group *local = (waitGC_group *) CkLocalBranch(waitGC_gchandle);
    return local->createGroup(cidx, considx, msg);
  }
}
                                       
extern "C" CkGroupID CkCreateNodeGroupSync(int cidx, int considx, void *msg)
{
  if(CkMyPe()==0) {
    return CkCreateNodeGroup(cidx, considx, msg, 0, 0);
  } else {
    waitGC_group *local = (waitGC_group *) CkLocalBranch(waitGC_gchandle);
    return local->createNodeGroup(cidx, considx, msg);
  }
}
                                       
waitqd_QDChare::waitqd_QDChare(CkArgMsg *m) {
  waitStarted = 0;
  threadList = 0;
  waitqd_qdhandle = thishandle;
  delete m;
  waitGC_gchandle = CProxy_waitGC_group::ckNew();
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
  delete ckqm;
}

#include "waitqd.def.h"
