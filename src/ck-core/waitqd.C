/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "waitqd.h"
#include "fifo.h"

/* readonly */ 
CkChareID waitqd_qdhandle;

extern "C" void CkWaitQD(void) {
  CProxy_waitqd_QDChare qdchareproxy(waitqd_qdhandle);
  qdchareproxy.waitQD();
}
  
extern "C" CkGroupID CkCreateGroupSync(int cidx, int considx, void *msg)
{
  if(CkMyPe()==0) {
    return CkCreateGroup(cidx, considx, msg, 0, 0);
  } else {
    CProxy_waitGC_chare waitChare(CkMyPe());
    ckGroupCreateMsg *inmsg = new ckGroupCreateMsg(cidx, considx, msg);
    ckGroupIDMsg *retmsg = waitChare.createGroup(inmsg);
    CkGroupID gid = retmsg->gid;
    delete retmsg;
    return gid;
  }
}
                                       
extern "C" CkGroupID CkCreateNodeGroupSync(int cidx, int considx, void *msg)
{
  if(CkMyPe()==0) {
    return CkCreateNodeGroup(cidx, considx, msg, 0, 0);
  } else {
    CProxy_waitGC_chare waitChare(CkMyPe());
    ckGroupCreateMsg *inmsg = new ckGroupCreateMsg(cidx, considx, msg);
    ckGroupIDMsg *retmsg = waitChare.createNodeGroup(inmsg);
    CkGroupID gid = retmsg->gid;
    delete retmsg;
    return gid;
  }
}
                                       
waitqd_QDChare::waitqd_QDChare(CkArgMsg *m) {
  waitStarted = 0;
  threadList = 0;
  waitqd_qdhandle = thishandle;
  delete m;
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
  CthThread pthr;
  while(!FIFO_Empty((FIFO_QUEUE*) threadList)) {
    FIFO_DeQueue((FIFO_QUEUE*) threadList, (void**) &pthr);
    CthAwaken(pthr);
  }
  FIFO_Destroy((FIFO_QUEUE*) threadList);
  threadList = 0;
  waitStarted = 0;
  delete ckqm;
}

#include "waitqd.def.h"
