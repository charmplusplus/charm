/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "waitqd.h"

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
    CProxy_waitGC_chare waitChare=CProxy_waitGC_chare::ckNew(CkMyPe());
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
    CProxy_waitGC_chare waitChare=CProxy_waitGC_chare::ckNew(CkMyPe());
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
    CdsFifo_Enqueue((CdsFifo)threadList, (void *)CthSelf());
  } else {
    waitStarted = 1;
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
  waitStarted = 0;
  delete ckqm;
}

#include "waitqd.def.h"
