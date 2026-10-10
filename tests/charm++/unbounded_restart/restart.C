#include "restart.decl.h"

// Checkpoint and restart of a chare array whose index is NOT packed into element
// ids: the array is created with no bounds and inserted dynamically, so its ids
// carry a hash key of the index plus a per-process unique number. The ids must
// survive the restart unchanged, and every element must be reachable afterwards,
// also when the restart uses a different number of PEs. (A restart with a
// different number of PROCESSES is rejected until the tranche allocator lands.)
//
// Run: ./restart +pe 4            writes the checkpoint to ckptlog/
//      ./restart +pe 2 +restart ckptlog

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem arrProxy;
static const int nElements = 16;

class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    delete m;
    mainProxy = thisProxy;
    CkArrayOptions opts;  // no bounds: ids are hashed, not packed
    arrProxy = CProxy_Elem::ckNew(opts);
    for (int i = 0; i < nElements; i++) arrProxy[i].insert();
    arrProxy.doneInserting();
  }
  Main(CkMigrateMessage* m) : CBase_Main(m)
  {
    mainProxy = thisProxy;
    CkPrintf("Main restored on %d PEs\n", CkNumPes());
  }
  void ready()
  {
    CkPrintf("All %d elements created; checkpointing\n", nElements);
    CkStartCheckpoint("ckptlog", CkCallback(CkIndex_Main::checkpointed(), thisProxy));
  }
  void checkpointed()
  {
    CkPrintf("Checkpoint done (restarted=%d); pinging every element\n", (int)_restarted);
    arrProxy.ping();
  }
  void done()
  {
    CkPrintf("All done\n");
    CkExit();
  }
  void pup(PUP::er& p) {}
};

class Elem : public CBase_Elem
{
public:
  Elem() { contribute(CkCallback(CkReductionTarget(Main, ready), mainProxy)); }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}
  void ping()
  {
    CkPrintf("Elem %d alive on PE %d\n", thisIndex, CkMyPe());
    contribute(CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
  void pup(PUP::er& p) { CBase_Elem::pup(p); }
};

#include "restart.def.h"
