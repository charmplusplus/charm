#include <cstdio>
#include <map>
#include <set>
#include "pup_stl.h"
#include "restart.decl.h"

// Checkpoint and restart of a chare array whose index is NOT packed into element
// ids: the array is created with no bounds and inserted dynamically, so its ids
// carry a hash key of the index plus a per-process unique number. The ids must
// survive the restart unchanged, and every element must be reachable afterwards,
// also when the restart uses a different number of PEs or processes.
//
// After a restart main inserts 16 NEW elements (indices 100..115 after the first
// restart, 200..215 after a second, and so on). Every process restarts with an
// empty tranche and takes its unique numbers from the allocator
// (doc/objid64-design.md sections 3.1 and 6), so each new id's unique part must
// lie in the allocator region [2^(U-1), 2^U), and all ids must be distinct: a new
// id can never collide with a restored one. Then every element is pinged by
// index; each reply carries the element's id, which must match the id main
// recorded (restored elements: the id from before the checkpoint).
//
// With -c a restarted run checkpoints again (to the same directory) once the pings
// are back, so a chain of restarts can be run, each from the previous run's
// checkpoint. Each restart must cost the allocator's pool only what its insertions
// use: with the grant floor at the launch share (before 2026-10-10) every restart
// spent 1/8 of the pool and the ninth aborted in grantTranche (make testchain).
//
// Run: ./restart +pe 4            writes the checkpoint to ckptlog/
//      ./restart +pe 2 +restart ckptlog
//      ./restart +pe 4 +restart ckptlog -c   (and again, from the new checkpoint)
// Different process count (reconverse): make testprocs

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem arrProxy;
static const int nElements = 16;
static const int newBaseStep = 100;  // restart g inserts indices [100 g, 100 g + 16)

// Unbuffered stdout, so an abort in another process does not discard its output.
void unbufferStdout(void) { setvbuf(stdout, NULL, _IONBF, 0); }

class Main : public CBase_Main
{
  std::map<int, CmiUInt8> ids;  // index -> element id; pupped with the checkpoint
  int gen = 0;                  // restarts so far; pupped: the chain's generation
  int nNew = 0;
  int pongs = 0;
  bool pinged = false;  // this run has pinged: the next checkpointed() ends the run
  static bool chain()  // -c; CmiGetArgFlag removes the flag, so look only once
  {
    static const bool c =
        CmiGetArgFlagDesc(CkGetArgv(), "-c", "checkpoint again after a restart") != 0;
    return c;
  }
  int newBase() const { return newBaseStep * gen; }

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
    CkPrintf("Main restored on %d PEs, %d processes\n", CkNumPes(), CkNumNodes());
  }
  void reportId(int idx, CmiUInt8 id)
  {
    CkEnforceMsg(ids.find(idx) == ids.end(), "element constructed twice");
    ids[idx] = id;
    if (!_restarted)
    {
      if ((int)ids.size() == nElements)
      {
        CkPrintf("All %d elements created; checkpointing\n", nElements);
        CkStartCheckpoint("ckptlog", CkCallback(CkIndex_Main::checkpointed(), thisProxy));
      }
      return;
    }
    CkEnforce(idx >= newBase() && idx < newBase() + nElements);
    if (++nNew == nElements) checkNewIds();
  }
  // Called after every checkpoint, and once at the start of a restarted run.
  void checkpointed()
  {
    if (!_restarted)
    {
      CkPrintf("Checkpoint done (restarted=0); pinging every element\n");
      pingAll();
      return;
    }
    if (pinged)
    {
      CkPrintf("Checkpoint done (restarted=1, generation %d); All done\n", gen);
      CkExit();
      return;
    }
    CkEnforceMsg((int)ids.size() == nElements * (gen + 1),
                 "restored main lost the recorded ids");
    gen++;
    CkPrintf("Restarted (generation %d); inserting %d new elements at %d\n", gen,
             nElements, newBase());
    for (int i = newBase(); i < newBase() + nElements; i++)
      arrProxy[i].insert(i % CkNumPes());
    arrProxy.doneInserting();
  }
  void checkNewIds()
  {
    const int U = ck::objid::getLayout().uniqueBits;
    const CmiUInt8 mask = ((CmiUInt8)1 << U) - 1;
    const CmiUInt8 half = (CmiUInt8)1 << (U - 1);
    std::set<CmiUInt8> seen;
    for (auto& e : ids)
    {
      CkEnforceMsg(seen.insert(e.second).second, "two elements share an id");
      if (e.first >= newBase() && (e.second & mask) < half)
        CkAbort("restart: new element %d has unique part %llu below 2^%d: not from"
                " the allocator\n",
                e.first, (unsigned long long)(e.second & mask), U - 1);
    }
    CkPrintf("All %d ids distinct; the %d new ids came from the allocator; pinging"
             " every element\n",
             (int)ids.size(), nElements);
    pingAll();
  }
  void pingAll()
  {
    pongs = 0;
    pinged = true;
    for (auto& e : ids) arrProxy[e.first].ping();
  }
  void pong(int idx, CmiUInt8 id, int pe)
  {
    auto it = ids.find(idx);
    CkEnforce(it != ids.end());
    CkEnforceMsg(it->second == id, "element id changed across the restart");
    if (++pongs < (int)ids.size()) return;
    if (_restarted && chain())
    {
      CkPrintf("All %d pinged; checkpointing again (generation %d)\n", (int)ids.size(),
               gen);
      CkStartCheckpoint("ckptlog", CkCallback(CkIndex_Main::checkpointed(), thisProxy));
      return;
    }
    CkPrintf("All done\n");
    CkExit();
  }
  void pup(PUP::er& p)
  {
    p | ids;
    p | gen;
  }
};

class Elem : public CBase_Elem
{
public:
  Elem() { mainProxy.reportId(thisIndex, ckGetID().getElementID()); }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}
  void ping()
  {
    CkPrintf("Elem %d alive on PE %d\n", thisIndex, CkMyPe());
    mainProxy.pong(thisIndex, ckGetID().getElementID(), CkMyPe());
  }
  void pup(PUP::er& p) { CBase_Elem::pup(p); }
};

#include "restart.def.h"
