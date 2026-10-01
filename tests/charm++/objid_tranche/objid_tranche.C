// objid_tranche: element-id tranches of an array without bounds (hashed id kind),
// doc/objid64-design.md section 3.1.
//
// An array created with no bounds mints element ids from a per-process tranche of
// unique numbers. At launch each process owns a share of the initial region
// [0, 2^(U-1)); the CkLocMgr branch on PE 0 hands out the allocator region
// [2^(U-1), 2^U) in tranches, requested when half of the current tranche is used.
// If a process runs out before the grant arrives, the insertion is deferred and
// completed when the grant arrives (a one-time "deferred an insertion" note).
//
// What is checked:
//  1. main inserts N elements (-n, default 300) at (0,i) on PE i % CkNumPes() from
//     ONE entry method (local inserts are inline, remote ones go to the target PE),
//     then a group inserts M (-m, default 100) more per PE at (1+pe,k), locally, in
//     one loop. These bulk loops do not return to the scheduler, so with small
//     tranches (+objid_tranche_log2 N caps every tranche at 2^N ids) they exhaust
//     the tranche before the refill arrives and exercise deferral.
//  2. after quiescence exactly N + M*CkNumPes() elements were constructed, each
//     on the PE it was inserted for (also when the insertion was deferred),
//     each index once, and all ids are distinct (tranches are disjoint). The split of
//     unique parts between the initial region and the allocator region is printed
//     (with default tranches nothing comes from the allocator).
//  3. every element is reachable by index (ping), keeps its id across one
//     migration (migrateMe to the next PE), and is reachable again afterwards.
//
// Run: ./objid_tranche +p4 [+objid_tranche_log2 2] [-n N] [-m M]
#include <cstdio>
#include <map>
#include <set>
#include <utility>
#include "objid_tranche.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem arrProxy;
/*readonly*/ CProxy_Checker checkerProxy;
/*readonly*/ int nMain;
/*readonly*/ int nPerPe;

// Unbuffered stdout, so an abort in another process does not discard its output.
void unbufferStdout(void) { setvbuf(stdout, NULL, _IONBF, 0); }

static void watchdogFire(void* arg, double t);
static void abortFire(void* arg, double t)
{
  fflush(stdout);
  CkAbort("objid_tranche: hung");
}

class Main : public CBase_Main
{
  typedef std::pair<int, int> Idx;
  std::map<Idx, CmiUInt8> ids;     // id of every constructed element
  int expected = 0;
  int checkersDone = 0;
  int nRow0 = 0;
  int nullLocalTotal = 0;
  int pongs = 0;
  int phase = 0;  // 0 inserting, 1 first ping, 2 migrating, 3 second ping
  double lastAdvance = 0;
  const char* stage = "inserting";

public:
  Main(CkArgMsg* m)
  {
    nMain = 300;
    nPerPe = 100;
    CmiGetArgInt(m->argv, "-n", &nMain);
    CmiGetArgInt(m->argv, "-m", &nPerPe);
    delete m;
    mainProxy = thisProxy;
    expected = nMain + nPerPe * CkNumPes();
    const ck::objid::Layout& l = ck::objid::getLayout();
    CkPrintf("objid_tranche: %d PEs, %d processes, N=%d, M=%d per PE, uniqueBits=%d,"
             " tranche log2 cap=%d\n",
             CkNumPes(), CkNumNodes(), nMain, nPerPe, l.uniqueBits, l.trancheLog2Cap);
    arrProxy = CProxy_Elem::ckNew(CkArrayOptions());  // no bounds: hashed ids
    checkerProxy = CProxy_Checker::ckNew();
    for (int i = 0; i < nMain; i++) arrProxy(0, i).insert(i % CkNumPes());
    arrProxy.doneInserting();
    if (nMain == 0) startGroupInserts();
    lastAdvance = CkWallTimer();
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }

  void progress() { lastAdvance = CkWallTimer(); }

  void checkProgress()
  {
    if (CkWallTimer() - lastAdvance > 60.0)
    {
      CkPrintf("objid_tranche WATCHDOG: no progress for 60 s in stage '%s':"
               " %zu of %d elements registered, %d checkers done, %d pongs\n",
               stage, ids.size(), expected, checkersDone, pongs);
      CcdCallFnAfter(abortFire, NULL, 1000);
      return;
    }
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }

  void registerElem(int x, int y, CmiUInt8 id, int pe)
  {
    progress();
    Idx k(x, y);
    CkEnforceMsg(ids.find(k) == ids.end(), "element constructed twice");
    ids[k] = id;
    // Inserted with an explicit PE (deferred or not): it must be constructed there.
    const int wantPe = (x == 0) ? y % CkNumPes() : x - 1;
    if (pe != wantPe)
      CkAbort("objid_tranche: (%d,%d) constructed on PE %d, inserted for PE %d\n", x, y, pe,
              wantPe);
    // Once main's own N are all there, let every PE insert its M.
    if (x == 0 && ++nRow0 == nMain) startGroupInserts();
  }

  void startGroupInserts()
  {
    stage = "group inserts";
    checkerProxy.insertMore();
  }

  void checkerDone(int pe, int nullLocal)
  {
    progress();
    nullLocalTotal += nullLocal;
    if (++checkersDone == CkNumPes())
    {
      stage = "quiescence after inserts";
      CkStartQD(CkCallback(CkIndex_Main::insertsQuiet(), thisProxy));
    }
  }

  void insertsQuiet()
  {
    progress();
    CkPrintf("objid_tranche: %zu elements constructed (expected %d); %d of the %d"
             " group inserts had no local element right after insert (deferred)\n",
             ids.size(), expected, nullLocalTotal, nPerPe * CkNumPes());
    CkEnforce((int)ids.size() == expected);
    const int U = ck::objid::getLayout().uniqueBits;
    const CmiUInt8 mask = ((CmiUInt8)1 << U) - 1;
    const CmiUInt8 half = (CmiUInt8)1 << (U - 1);
    std::set<CmiUInt8> seen;
    int fromInitial = 0, fromAllocator = 0;
    for (auto& e : ids)
    {
      CkEnforceMsg(seen.insert(e.second).second, "two elements share an id");
      if ((e.second & mask) >= half) ++fromAllocator;
      else ++fromInitial;
    }
    CkPrintf("objid_tranche: all %zu ids distinct; unique parts: %d from the initial"
             " region, %d from the allocator\n",
             ids.size(), fromInitial, fromAllocator);
    phase = 1;
    stage = "first ping";
    pingAll();
  }

  void pingAll()
  {
    pongs = 0;
    for (auto& e : ids) arrProxy(e.first.first, e.first.second).ping();
  }

  void pong(int x, int y, CmiUInt8 id, int pe)
  {
    progress();
    auto it = ids.find(Idx(x, y));
    CkEnforce(it != ids.end());
    CkEnforceMsg(it->second == id, "element id changed");
    if (++pongs < expected) return;
    if (phase == 1)
    {
      CkPrintf("objid_tranche: all %d elements answered by index; migrating each once\n",
               expected);
      phase = 2;
      stage = "migration";
      for (auto& e : ids) arrProxy(e.first.first, e.first.second).migrateOnce();
      CkStartQD(CkCallback(CkIndex_Main::migratedQuiet(), thisProxy));
    }
    else if (phase == 3)
    {
      CkPrintf("objid_tranche: all %d elements answered after migration; %d elements"
               " total\n",
               expected, (int)ids.size());
      CkPrintf("objid_tranche PASS\n");
      CkExit();
    }
  }

  void migratedQuiet()
  {
    progress();
    phase = 3;
    stage = "second ping";
    pingAll();
  }
};

static void watchdogFire(void* arg, double t) { mainProxy.checkProgress(); }

class Checker : public CBase_Checker
{
public:
  Checker() {}
  void insertMore()
  {
    const int me = CkMyPe();
    int nullLocal = 0;
    for (int k = 0; k < nPerPe; k++)
    {
      arrProxy(1 + me, k).insert(me);
      // A local insert is inline unless the insertion was deferred (no ids left).
      if (arrProxy(1 + me, k).ckLocal() == nullptr) ++nullLocal;
    }
    mainProxy.checkerDone(me, nullLocal);
  }
};

class Elem : public CBase_Elem
{
public:
  Elem()
  {
    mainProxy.registerElem(thisIndex.x, thisIndex.y, ckGetID().getElementID(), CkMyPe());
  }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}
  void ping()
  {
    mainProxy.pong(thisIndex.x, thisIndex.y, ckGetID().getElementID(), CkMyPe());
  }
  void migrateOnce()
  {
    if (CkNumPes() > 1) migrateMe((CkMyPe() + 1) % CkNumPes());
  }
  void pup(PUP::er& p) { CBase_Elem::pup(p); }
};

#include "objid_tranche.def.h"
