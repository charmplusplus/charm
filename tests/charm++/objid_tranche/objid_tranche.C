// objid_tranche: element-id tranches of arrays without bounds (hashed id kind),
// doc/objid64-design.md section 3.1.
//
// An array created with no bounds mints element ids from a per-process tranche of
// unique numbers. At launch each process owns a share of the initial region
// [0, 2^(U-1)); the CkLocMgr branch on PE 0 hands out the allocator region
// [2^(U-1), 2^U) in tranches, requested when half of the current tranche is used.
// If a process runs out before the grant arrives, the insertion is deferred and
// completed when the grant arrives (a one-time "deferred an insertion" note).
// Every array (CkLocMgr) has its own tranche table entry and its own allocator
// pool on PE 0; this test runs A arrays at once so their refills and deferrals
// interleave.
//
// What is checked:
//  1. A (-a, default 3) unbounded arrays of the same Elem type are created. main
//     inserts N elements (-n, default 300) at (0,i) into array i % A, on PE
//     i % CkNumPes(), from ONE entry method (local inserts are inline, remote ones
//     go to the target PE), then a group inserts M (-m, default 100) more per PE at
//     (1+pe,k) into array k % A, locally, in one loop. These bulk loops do not
//     return to the scheduler, so with small tranches (+objid_tranche_log2 N caps
//     every tranche at 2^N ids) they exhaust the tranches before the refills
//     arrive and exercise deferral, on all arrays interleaved.
//  2. after quiescence every array has exactly its expected number of elements
//     (a deferral or refill of one array must not disturb another), each
//     constructed on the PE it was inserted for (also when the insertion was
//     deferred), each index once, and ids are distinct within each array
//     (tranches are disjoint). The split of unique parts between the initial
//     region and the allocator region is printed per array (with default tranches
//     nothing comes from the allocator).
//  3. every element is reachable by index (ping), keeps its id across one
//     migration (migrateMe to the next PE), and is reachable again afterwards.
//  4. demand creation: every PE sends [createhome] revive to D (-d, default 50)
//     never-inserted indices (-1,j) of every array, so each index gets
//     CkNumPes() revives. Each must be created exactly once, at its home PE (also
//     when the creation is deferred there for lack of ids), receive all CkNumPes()
//     revives (one from each PE), and get an id distinct from every other id of
//     its array.
//
// Run: ./objid_tranche +pe 4 [+objid_tranche_log2 2] [-n N] [-m M] [-a A] [-d D]
#include <cstdio>
#include <map>
#include <set>
#include <tuple>
#include <vector>
#include "objid_tranche.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ std::vector<CProxy_Elem> arrProxies;
/*readonly*/ CProxy_Checker checkerProxy;
/*readonly*/ int nMain;
/*readonly*/ int nPerPe;
/*readonly*/ int nArrays;
/*readonly*/ int nRevive;

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
  typedef std::tuple<int, int, int> Key;  // (array, x, y)
  std::map<Key, CmiUInt8> ids;            // id of every constructed element
  std::vector<int> expectedPerArray;
  std::map<Key, std::set<int>> reviveFrom;  // revive senders seen per element
  std::map<Key, int> createdPe;             // PE a demand-created element was built on
  std::map<Key, int> revivePe;              // PE its revives ran on
  int expected = 0;                         // inserted elements, all arrays
  int checkersDone = 0;
  int nRow0 = 0;
  int nullLocalTotal = 0;
  int pongs = 0;
  int nCreated = 0;  // demand-created elements
  int nRevives = 0;
  int phase = 0;  // 0 inserting, 1 first ping, 2 migrating, 3 second ping, 4 revive
  double lastAdvance = 0;
  const char* stage = "inserting";

public:
  Main(CkArgMsg* m)
  {
    nMain = 300;
    nPerPe = 100;
    nArrays = 3;
    nRevive = 50;
    CmiGetArgInt(m->argv, "-n", &nMain);
    CmiGetArgInt(m->argv, "-m", &nPerPe);
    CmiGetArgInt(m->argv, "-a", &nArrays);
    CmiGetArgInt(m->argv, "-d", &nRevive);
    delete m;
    CkEnforceMsg(nArrays >= 1, "-a must be at least 1");
    mainProxy = thisProxy;
    expected = nMain + nPerPe * CkNumPes();
    expectedPerArray.assign(nArrays, 0);
    for (int i = 0; i < nMain; i++) ++expectedPerArray[i % nArrays];
    for (int k = 0; k < nPerPe; k++) expectedPerArray[k % nArrays] += CkNumPes();
    const ck::objid::Layout& l = ck::objid::getLayout();
    CkPrintf("objid_tranche: %d PEs, %d processes, A=%d arrays, N=%d, M=%d per PE,"
             " D=%d revived per array, uniqueBits=%d, tranche log2 cap=%d\n",
             CkNumPes(), CkNumNodes(), nArrays, nMain, nPerPe, nRevive, l.uniqueBits,
             l.trancheLog2Cap);
    for (int a = 0; a < nArrays; a++)
      arrProxies.push_back(CProxy_Elem::ckNew(CkArrayOptions()));  // no bounds: hashed ids
    checkerProxy = CProxy_Checker::ckNew();
    for (int i = 0; i < nMain; i++) arrProxies[i % nArrays](0, i).insert(i % CkNumPes());
    for (int a = 0; a < nArrays; a++) arrProxies[a].doneInserting();
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
               " %zu of %d elements registered, %d checkers done, %d pongs,"
               " %d of %d demand-created, %d of %d revives\n",
               stage, ids.size(), expected + nArrays * nRevive, checkersDone, pongs,
               nCreated, nArrays * nRevive, nRevives, nArrays * nRevive * CkNumPes());
      CcdCallFnAfter(abortFire, NULL, 1000);
      return;
    }
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }

  void registerElem(int a, int x, int y, CmiUInt8 id, int pe, bool atHome)
  {
    progress();
    Key k(a, x, y);
    auto prev = ids.find(k);
    if (prev != ids.end())
    {
      auto cp = createdPe.find(k);
      CkAbort("objid_tranche: array %d element (%d,%d) constructed twice: first id %llu"
              " on PE %d, again id %llu on PE %d (home here: %d)\n",
              a, x, y, (unsigned long long)prev->second,
              cp != createdPe.end() ? cp->second : -1, (unsigned long long)id, pe,
              (int)atHome);
    }
    ids[k] = id;
    if (x == -1)
    {
      // Demand-created: must be at its home PE, and only in the revive phase.
      if (phase != 4)
        CkAbort("objid_tranche: array %d element (-1,%d) created outside the revive phase\n",
                a, y);
      if (!atHome)
        CkAbort("objid_tranche: array %d element (-1,%d) demand-created on PE %d, not its"
                " home\n",
                a, y, pe);
      createdPe[k] = pe;
      ++nCreated;
      return;
    }
    // Inserted with an explicit PE (deferred or not): it must be constructed there.
    const int wantPe = (x == 0) ? y % CkNumPes() : x - 1;
    const int wantArr = y % nArrays;  // i % A for main, k % A for the group
    if (a != wantArr)
      CkAbort("objid_tranche: (%d,%d) constructed in array %d, inserted into array %d\n", x,
              y, a, wantArr);
    if (pe != wantPe)
      CkAbort("objid_tranche: array %d (%d,%d) constructed on PE %d, inserted for PE %d\n", a,
              x, y, pe, wantPe);
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

  // Ids distinct within each array; prints the initial/allocator split per array.
  void checkIds(const char* when)
  {
    const int U = ck::objid::getLayout().uniqueBits;
    const CmiUInt8 mask = ((CmiUInt8)1 << U) - 1;
    const CmiUInt8 half = (CmiUInt8)1 << (U - 1);
    std::vector<std::set<CmiUInt8>> seen(nArrays);
    std::vector<int> fromInitial(nArrays, 0), fromAllocator(nArrays, 0);
    for (auto& e : ids)
    {
      const int a = std::get<0>(e.first);
      if (!seen[a].insert(e.second).second)
        CkAbort("objid_tranche: two elements of array %d share id %llu (%s)\n", a,
                (unsigned long long)e.second, when);
      if ((e.second & mask) >= half) ++fromAllocator[a];
      else ++fromInitial[a];
    }
    for (int a = 0; a < nArrays; a++)
      CkPrintf("objid_tranche: %s: array %d: %zu ids distinct; unique parts: %d from the"
               " initial region, %d from the allocator\n",
               when, a, seen[a].size(), fromInitial[a], fromAllocator[a]);
  }

  void insertsQuiet()
  {
    progress();
    CkPrintf("objid_tranche: %zu elements constructed (expected %d); %d of the %d"
             " group inserts had no local element right after insert (deferred)\n",
             ids.size(), expected, nullLocalTotal, nPerPe * CkNumPes());
    std::vector<int> have(nArrays, 0);
    for (auto& e : ids) ++have[std::get<0>(e.first)];
    for (int a = 0; a < nArrays; a++)
      if (have[a] != expectedPerArray[a])
        CkAbort("objid_tranche: array %d has %d elements after quiescence, expected %d\n", a,
                have[a], expectedPerArray[a]);
    CkEnforce((int)ids.size() == expected);
    checkIds("after inserts");
    if (expected == 0)
    {
      startRevives();  // nothing to ping or migrate
      return;
    }
    phase = 1;
    stage = "first ping";
    pingAll();
  }

  void pingAll()
  {
    pongs = 0;
    for (auto& e : ids)
      arrProxies[std::get<0>(e.first)](std::get<1>(e.first), std::get<2>(e.first)).ping();
  }

  void pong(int a, int x, int y, CmiUInt8 id, int pe)
  {
    progress();
    auto it = ids.find(Key(a, x, y));
    CkEnforce(it != ids.end());
    CkEnforceMsg(it->second == id, "element id changed");
    if (++pongs < expected) return;
    if (phase == 1)
    {
      CkPrintf("objid_tranche: all %d elements answered by index; migrating each once\n",
               expected);
      phase = 2;
      stage = "migration";
      for (auto& e : ids)
        arrProxies[std::get<0>(e.first)](std::get<1>(e.first), std::get<2>(e.first))
            .migrateOnce();
      CkStartQD(CkCallback(CkIndex_Main::migratedQuiet(), thisProxy));
    }
    else if (phase == 3)
    {
      CkPrintf("objid_tranche: all %d elements answered after migration; %d elements"
               " total\n",
               expected, (int)ids.size());
      startRevives();
    }
  }

  void migratedQuiet()
  {
    progress();
    phase = 3;
    stage = "second ping";
    pingAll();
  }

  void startRevives()
  {
    phase = 4;
    stage = "demand creation";
    if (nRevive == 0)
    {
      revivesQuiet();
      return;
    }
    checkerProxy.reviveAll();
    CkStartQD(CkCallback(CkIndex_Main::revivesQuiet(), thisProxy));
  }

  void revived(int a, int x, int y, int from, int pe)
  {
    progress();
    // No check against registration here: the element's registerElem and its revived
    // messages to main are not ordered across processes; revivesQuiet checks all.
    Key k(a, x, y);
    if (x != -1)
      CkAbort("objid_tranche: array %d (%d,%d) got a revive, but was inserted\n", a, x, y);
    auto pit = revivePe.find(k);
    if (pit == revivePe.end()) revivePe[k] = pe;
    else if (pit->second != pe)
      CkAbort("objid_tranche: array %d (-1,%d) got revives on PE %d and on PE %d\n", a, y,
              pit->second, pe);
    if (!reviveFrom[k].insert(from).second)
      CkAbort("objid_tranche: array %d (%d,%d) got the revive from PE %d twice\n", a, x, y,
              from);
    ++nRevives;
  }

  void revivesQuiet()
  {
    progress();
    const int wantCreated = nArrays * nRevive;
    const int wantRevives = wantCreated * CkNumPes();
    CkPrintf("objid_tranche: demand creation: %d of %d elements created, %d of %d"
             " revives delivered\n",
             nCreated, wantCreated, nRevives, wantRevives);
    for (int a = 0; a < nArrays; a++)
      for (int j = 0; j < nRevive; j++)
      {
        Key k(a, -1, j);
        if (ids.find(k) == ids.end())
          CkAbort("objid_tranche: array %d (-1,%d) was never demand-created\n", a, j);
        const size_t got = reviveFrom[k].size();
        if ((int)got != CkNumPes())
          CkAbort("objid_tranche: array %d (-1,%d) got %zu of %d revives\n", a, j, got,
                  CkNumPes());
        if (revivePe[k] != createdPe[k])
          CkAbort("objid_tranche: array %d (-1,%d) created on PE %d, revives ran on PE %d\n",
                  a, j, createdPe[k], revivePe[k]);
      }
    CkEnforce(nCreated == wantCreated);
    CkEnforce(nRevives == wantRevives);
    CkEnforce((int)ids.size() == expected + wantCreated);
    checkIds("after demand creation");  // new ids distinct from all existing ones
    CkPrintf("objid_tranche PASS\n");
    CkExit();
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
      CProxy_Elem& arr = arrProxies[k % nArrays];
      arr(1 + me, k).insert(me);
      // A local insert is inline unless the insertion was deferred (no ids left).
      if (arr(1 + me, k).ckLocal() == nullptr) ++nullLocal;
    }
    mainProxy.checkerDone(me, nullLocal);
  }
  void reviveAll()
  {
    const int me = CkMyPe();
    // Interleave the arrays so their demand creations compete for ids together.
    for (int j = 0; j < nRevive; j++)
      for (int a = 0; a < nArrays; a++) arrProxies[a](-1, j).revive(me);
  }
};

class Elem : public CBase_Elem
{
  int arr = -1;  // which of arrProxies this element belongs to

  int findArray() const
  {
    for (int a = 0; a < (int)arrProxies.size(); a++)
      if (arrProxies[a].ckGetArrayID() == thisArrayID) return a;
    CkAbort("objid_tranche: element of an unknown array\n");
    return -1;
  }

public:
  Elem()
  {
    arr = findArray();
    const bool atHome = arrProxies[arr].ckLocMgr()->homePe(thisIndexMax) == CkMyPe();
    mainProxy.registerElem(arr, thisIndex.x, thisIndex.y, ckGetID().getElementID(), CkMyPe(),
                           atHome);
  }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}
  void ping()
  {
    mainProxy.pong(arr, thisIndex.x, thisIndex.y, ckGetID().getElementID(), CkMyPe());
  }
  void migrateOnce()
  {
    if (CkNumPes() > 1) migrateMe((CkMyPe() + 1) % CkNumPes());
  }
  void revive(int from) { mainProxy.revived(arr, thisIndex.x, thisIndex.y, from, CkMyPe()); }
  void pup(PUP::er& p)
  {
    CBase_Elem::pup(p);
    p | arr;
  }
};

#include "objid_tranche.def.h"
