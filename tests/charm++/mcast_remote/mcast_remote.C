// Element 0 builds a section of all elements and hands the proxy to the last
// element, on another PE, which multicasts through it. That PE must record the
// multicast for the load balancer once, from the sender to every member.
#include "mcast_remote.decl.h"
#include "ckmulticast.h"
#include "LBManager.h"
#include <vector>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numElems;

struct WorkMsg : public CkMcastBaseMsg, public CMessage_WorkMsg {};

class Main : public CBase_Main {
 public:
  Main(CkArgMsg *m) {
    delete m;
    if (CkNumPes() < 2) CkAbort("mcast_remote: needs at least 2 PEs");
    numElems = 2 * CkNumPes();
    mainProxy = thisProxy;
    CProxy_Elem elems = CProxy_Elem::ckNew(numElems);
    elems[0].start();
  }
  void checked() { CkStartQD(CkCallback(CkIndex_Main::done(), thisProxy)); }
  void done() { CkPrintf("mcast_remote PASS\n"); CkExit(); }
};

class Elem : public CBase_Elem {
 public:
  Elem() {}
  void start() {
    std::vector<CkArrayIndex> all;
    for (int i = 0; i < numElems; i++) all.emplace_back(CkArrayIndex1D(i));
    thisProxy[numElems - 1].takeSection(CProxySection_Elem::ckNew(thisArrayID, all));
  }
  void takeSection(CProxySection_Elem sec) {
    if (sec.ckGetSectionInfo().get_pe() == CkMyPe())
      CkAbort("mcast_remote: element %d is on the root's PE %d", thisIndex, CkMyPe());
    sec.work(new WorkMsg);
#if CMK_LBDB_ON
    LBManager *lb = LBManagerObj();
    if (!lb->CollectingCommStats()) CkAbort("mcast_remote: run with +balancer <LB> +LBCommOn");
    CkLocMgr *mgr = CProxy_ArrayBase(thisArrayID).ckLocMgr();
    CmiUInt8 me = mgr->lbObjID(mgr->lookupID(CkArrayIndex1D(thisIndex)));
    std::vector<LDCommData> comm(lb->GetCommDataSz());
    lb->GetCommData(comm.data());
    int found = 0;
    for (LDCommData &c : comm) {
      if (c.recv_type() != LD_OBJLIST_MSG || c.from_proc() || c.sender.objID() != me) continue;
      int len;
      c.receiver.get_destObjs(len);
      if (len != numElems || c.messages != 1)
        CkAbort("mcast_remote: record has %d of %d members, %d messages", len, numElems,
                c.messages);
      found++;
    }
    if (found != 1)
      CkAbort("mcast_remote: PE %d has %d multicast records from element %d", CkMyPe(), found,
              thisIndex);
#endif
    mainProxy.checked();
  }
  void work(WorkMsg *m) { delete m; }
};

#include "mcast_remote.def.h"
