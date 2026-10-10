// objid_home: checks the invariants of the 64-bit object id redesign
// (charm #3994, doc/objid64-design.md) directly, from every PE, before and
// after migration.
//
// Two arrays, one of each kind (design section 1):
//   Packed  3D, CkArrayOptions::setBounds(64,64,64): 18 index bits, more than
//           the old 16-bit budget (section 2, goal G1). The element id is the
//           packed index, a pure function of the index.
//   Hashed  2D, no bounds: id = indexHashKey(idx) in the top keyBits | a unique
//           number in the low uniqueBits, taken from the creating process's
//           tranche (sections 3.1-3.3).
//
// What is checked, and why:
//   - element ids are distinct within each array (uniqueness, section 3.1);
//   - on EVERY PE: CkLocMgr::homePe(idx) == homePe(id) (goal G2: one home,
//     computable from the index or from the id alone, on any PE);
//   - Packed: lookupID(idx) succeeds on every PE, including PEs that never saw
//     the index, and gives the element's id, which equals the packed index
//     (x<<12 | y<<6 | z) (section 2);
//   - Hashed: if a PE knows the id it is the element's id; indexHashKey(idx)
//     == id >> uniqueBits, and homePe(id) == CkNodeFirst(key % CkNumNodes()),
//     i.e. rank 0 of a process (sections 3.2, 3.3);
//   - all PEs compute the same home for every element (G2);
//   - Hashed ids minted on different processes have distinct unique parts,
//     and each lies in its creating process's initial tranche
//     [node*S, (node+1)*S), S = 2^(U-1-ceil(log2 nodes)) (section 3.1);
//   - after each of two migrations of every element, every id and every home
//     is unchanged (no PE number in the id, section 1 and G2), and each
//     element is where migrateMe sent it.
//
// Run: ./objid_home +pe 4, or multi-process under lcrun. Prints
// "objid_home PASS" on success; aborts with a message on any failure, or after
// 60 s without progress (watchdog).

#include <algorithm>
#include <map>
#include <set>
#include <stdio.h>
#include <vector>
#include "objid_home.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Packed packedProxy;
/*readonly*/ CProxy_Hashed hashedProxy;
/*readonly*/ CProxy_Prober proberProxy;

static const int kPerPe = 8;      // elements each PE inserts into each array
static const int kRounds = 3;     // round 0: before migration; 1, 2: after
static const int kFields = 6;     // table entry: kind, x, y, z, id, pe
static const double kWatchdogSecs = 60.0;
enum { PACKED = 0, HASHED = 1 };

// Distinct pseudo-random in-bounds 3D index for (pe, k): an odd multiplier
// modulo 2^18 is a bijection, so distinct (pe, k) give distinct indices.
static void packedIndex(int pe, int k, int& x, int& y, int& z)
{
  unsigned v = ((unsigned)(pe * kPerPe + k) * 0x2F1B5u + 0x1234u) & 0x3FFFFu;
  x = (v >> 12) & 63;
  y = (v >> 6) & 63;
  z = v & 63;
}

static int ceilLog2(CmiUInt8 n)
{
  int b = 0;
  while (((CmiUInt8)1 << b) < n) b++;
  return b;
}

static void watchdogFire(void*, double) { mainProxy.checkProgress(); }

class Main : public CBase_Main
{
  int round = 0;
  int reports = 0;
  int homeReports = 0;
  int failures = 0;
  double lastAdvance;
  std::vector<CmiInt8> table;               // this round, sorted
  std::vector<CmiInt8> table0;              // round 0
  std::vector<CmiInt8> prevTable;           // previous round
  std::vector<int> homes0;                  // PE 0's home vector of round 0
  std::vector<int> roundHomes;              // first home vector of this round
  int roundHomesPe = -1;

  void progress() { lastAdvance = CkWallTimer(); }

  void fail(const char* what)
  {
    failures++;
    if (failures <= 40) CkPrintf("objid_home FAIL (round %d): %s\n", round, what);
  }

  void abortIfFailed(const char* phase)
  {
    if (failures)
      CkAbort("objid_home FAIL: %d check(s) failed in round %d (%s)\n", failures,
              round, phase);
  }

 public:
  Main(CkArgMsg* m)
  {
    delete m;
    setvbuf(stdout, NULL, _IONBF, 0);
    mainProxy = thisProxy;

    const ck::objid::Layout& l = ck::objid::getLayout();
    CkPrintf("objid_home: %d PEs, %d processes; layout keyBits %d uniqueBits %d "
             "expandFactor %d nodesAtLaunch %d; payload bits %d\n",
             CkNumPes(), CkNumNodes(), l.keyBits, l.uniqueBits, l.expandFactor,
             l.nodesAtLaunch, (int)ck::ObjID::PAYLOAD_BITS);
    if (l.keyBits + l.uniqueBits != (int)ck::ObjID::PAYLOAD_BITS)
      fail("keyBits + uniqueBits != PAYLOAD_BITS");
    if (l.keyBits < 1 || l.uniqueBits < 16) fail("layout out of range");
    if (l.nodesAtLaunch != CkNumNodes()) fail("nodesAtLaunch != CkNumNodes()");
    abortIfFailed("layout");

    CkArrayOptions popts;
    popts.setBounds(64, 64, 64);
    packedProxy = CProxy_Packed::ckNew(0, popts);
    CkArrayOptions hopts;
    hashedProxy = CProxy_Hashed::ckNew(0, hopts);
    proberProxy = CProxy_Prober::ckNew();

    progress();
    CcdCallFnAfter(watchdogFire, NULL, 2000);
    proberProxy.insertAll();
  }

  void checkProgress()
  {
    if (CkWallTimer() - lastAdvance > kWatchdogSecs)
      CkAbort("objid_home WATCHDOG: no progress for %.0f s (round %d, %d/%d "
              "element reports, %d/%d home reports)\n",
              kWatchdogSecs, round, reports, 2 * kPerPe * CkNumPes(), homeReports,
              CkNumPes());
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }

  void insertedAll()
  {
    progress();
    // The inserts may still be in flight; wait until they have all been created.
    CkStartQD(CkCallback(CkIndex_Main::quiet(), mainProxy));
  }

  void quiet()
  {
    progress();
    reports = 0;
    table.clear();
    packedProxy.report();
    hashedProxy.report();
  }

  void report(int kind, int x, int y, int z, CmiUInt8 id, int pe)
  {
    progress();
    CmiInt8 e[kFields] = {kind, x, y, z, (CmiInt8)id, pe};
    table.insert(table.end(), e, e + kFields);
    if (++reports < 2 * kPerPe * CkNumPes()) return;

    // Sort entries by (kind, x, y, z) so rounds compare position by position.
    const int n = (int)table.size() / kFields;
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return std::lexicographical_compare(&table[a * kFields], &table[a * kFields + 4],
                                          &table[b * kFields], &table[b * kFields + 4]);
    });
    std::vector<CmiInt8> sorted;
    for (int i : order)
      sorted.insert(sorted.end(), &table[i * kFields], &table[i * kFields + kFields]);
    table.swap(sorted);

    char buf[256];
    std::set<CmiInt8> ids[2];
    for (int i = 0; i < n; i++)
    {
      const CmiInt8* e = &table[i * kFields];
      if (!ids[e[0]].insert(e[4]).second)
      {
        snprintf(buf, sizeof buf, "duplicate id %llx in %s array", (long long)e[4],
                 e[0] == PACKED ? "Packed" : "Hashed");
        fail(buf);
      }
    }

    if (round == 0)
    {
      // Each element was inserted on its creating PE (insert(onPE = CkMyPe())),
      // so the unique part of a hashed id comes from that process's tranche.
      const ck::objid::Layout& l = ck::objid::getLayout();
      const int shareBits = l.uniqueBits - 1 - ceilLog2((CmiUInt8)l.nodesAtLaunch);
      const CmiUInt8 S = shareBits > 0 ? ((CmiUInt8)1 << shareBits) : 1;
      const CmiUInt8 umask = ((CmiUInt8)1 << l.uniqueBits) - 1;
      std::map<CmiUInt8, int> uniqueNode;
      for (int i = 0; i < n; i++)
      {
        const CmiInt8* e = &table[i * kFields];
        if (e[0] != HASHED) continue;
        const CmiUInt8 u = (CmiUInt8)e[4] & umask;
        const int node = CkNodeOf((int)e[5]);
        auto r = uniqueNode.emplace(u, node);
        if (!r.second)
        {
          snprintf(buf, sizeof buf, "hashed unique part %llx minted on processes %d and %d",
                   (unsigned long long)u, r.first->second, node);
          fail(buf);
        }
        if (u / S != (CmiUInt8)node)
        {
          snprintf(buf, sizeof buf, "hashed (%lld,%lld) unique part %llx not in tranche of "
                   "creating process %d (S = %llu)", (long long)e[1], (long long)e[2],
                   (unsigned long long)u, node, (unsigned long long)S);
          fail(buf);
        }
      }
      table0 = table;
    }
    else
    {
      if (table.size() != table0.size()) fail("element count changed");
      for (int i = 0; i < n && table.size() == table0.size(); i++)
      {
        const CmiInt8* e = &table[i * kFields];
        const CmiInt8* e0 = &table0[i * kFields];
        const CmiInt8* ep = &prevTable[i * kFields];
        if (!std::equal(e, e + 4, e0))
        {
          fail("element set changed");
          break;
        }
        if (e[4] != e0[4])
        {
          snprintf(buf, sizeof buf, "%s (%lld,%lld,%lld) id changed %llx -> %llx",
                   e[0] == PACKED ? "Packed" : "Hashed", (long long)e[1],
                   (long long)e[2], (long long)e[3], (long long)e0[4], (long long)e[4]);
          fail(buf);
        }
        if (e[5] != (ep[5] + 1) % CkNumPes())
        {
          snprintf(buf, sizeof buf, "%s (%lld,%lld,%lld) on PE %lld, expected %lld",
                   e[0] == PACKED ? "Packed" : "Hashed", (long long)e[1],
                   (long long)e[2], (long long)e[3], (long long)e[5],
                   (long long)((ep[5] + 1) % CkNumPes()));
          fail(buf);
        }
      }
    }
    abortIfFailed("element table");
    prevTable = table;

    homeReports = 0;
    roundHomes.clear();
    roundHomesPe = -1;
    proberProxy.check(round, table);
  }

  void homes(int pe, int rnd, std::vector<int> h)
  {
    progress();
    char buf[256];
    if (rnd != round) fail("home report from a different round");
    if (roundHomesPe < 0)
    {
      roundHomes = h;
      roundHomesPe = pe;
    }
    else if (h != roundHomes)
    {
      for (size_t i = 0; i < h.size() && i < roundHomes.size(); i++)
        if (h[i] != roundHomes[i])
        {
          const CmiInt8* e = &table[i * kFields];
          snprintf(buf, sizeof buf, "%s (%lld,%lld,%lld): home %d on PE %d, %d on PE %d",
                   e[0] == PACKED ? "Packed" : "Hashed", (long long)e[1],
                   (long long)e[2], (long long)e[3], h[i], pe, roundHomes[i],
                   roundHomesPe);
          fail(buf);
        }
      if (h.size() != roundHomes.size()) fail("home vectors differ in length");
    }
    if (++homeReports < CkNumPes()) return;

    if (round == 0)
      homes0 = roundHomes;
    else if (roundHomes != homes0)
      fail("home of some element changed after migration");
    abortIfFailed("homes");

    CkPrintf("objid_home: round %d OK (%d elements, %d PEs agree on every home)\n",
             round, (int)table.size() / kFields, CkNumPes());
    if (++round == kRounds)
    {
      CkPrintf("objid_home PASS\n");
      CkExit();
      return;
    }
    packedProxy.hop();
    hashedProxy.hop();
    CkStartQD(CkCallback(CkIndex_Main::quiet(), mainProxy));
  }
};

class Prober : public CBase_Prober
{
 public:
  Prober() {}

  void insertAll()
  {
    const int me = CkMyPe();
    for (int k = 0; k < kPerPe; k++)
    {
      int x, y, z;
      packedIndex(me, k, x, y, z);
      packedProxy(x, y, z).insert(me, me);
      hashedProxy(me * 1000 + k, k * 7).insert(me, me);
    }
    packedProxy.doneInserting();
    hashedProxy.doneInserting();
    contribute(CkCallback(CkReductionTarget(Main, insertedAll), mainProxy));
  }

  void check(int round, std::vector<CmiInt8> table)
  {
    CkLocMgr* pm = packedProxy.ckLocMgr();
    CkLocMgr* hm = hashedProxy.ckLocMgr();
    const ck::objid::Layout& l = ck::objid::getLayout();
    const int n = (int)table.size() / kFields;
    std::vector<int> h(n);
    int bad = 0;
    auto failHere = [&](const CmiInt8* e, const char* what, CmiUInt8 a, CmiUInt8 b) {
      if (++bad <= 20)
        CkPrintf("objid_home FAIL (round %d, PE %d): %s (%lld,%lld,%lld) id %llx: %s "
                 "(%llx vs %llx)\n",
                 round, CkMyPe(), e[0] == PACKED ? "Packed" : "Hashed", (long long)e[1],
                 (long long)e[2], (long long)e[3], (long long)e[4], what,
                 (unsigned long long)a, (unsigned long long)b);
    };
    for (int i = 0; i < n; i++)
    {
      const CmiInt8* e = &table[i * kFields];
      const CmiUInt8 id = (CmiUInt8)e[4];
      CmiUInt8 id2 = 0;
      if (e[0] == PACKED)
      {
        CkArrayIndex3D idx((int)e[1], (int)e[2], (int)e[3]);
        h[i] = pm->homePe(idx);
        if (h[i] != pm->homePe(id)) failHere(e, "homePe(idx) != homePe(id)", h[i], pm->homePe(id));
        if (!pm->lookupID(idx, id2)) failHere(e, "lookupID failed", 0, 0);
        else if (id2 != id) failHere(e, "lookupID gives a different id", id2, id);
        const CmiUInt8 packed = ((CmiUInt8)e[1] << 12) | ((CmiUInt8)e[2] << 6) | (CmiUInt8)e[3];
        if (id != packed) failHere(e, "id != packed index", id, packed);
      }
      else
      {
        CkArrayIndex2D idx((int)e[1], (int)e[2]);
        h[i] = hm->homePe(idx);
        if (h[i] != hm->homePe(id)) failHere(e, "homePe(idx) != homePe(id)", h[i], hm->homePe(id));
        if (hm->lookupID(idx, id2) && id2 != id)
          failHere(e, "lookupID gives a different id", id2, id);
        const CmiUInt8 key = ck::objid::indexHashKey(idx);
        if (key != (id >> l.uniqueBits)) failHere(e, "indexHashKey != id >> uniqueBits", key, id >> l.uniqueBits);
        if (key >> l.keyBits) failHere(e, "indexHashKey has more than keyBits bits", key, l.keyBits);
        const int expect = CkNodeFirst((int)(key % (CmiUInt8)CkNumNodes()));
        if (hm->homePe(id) != expect) failHere(e, "homePe(id) != CkNodeFirst(key % nodes)", hm->homePe(id), expect);
      }
    }
    if (bad)
      CkAbort("objid_home FAIL: %d check(s) failed on PE %d in round %d\n", bad,
              CkMyPe(), round);
    mainProxy.homes(CkMyPe(), round, h);
  }
};

class Packed : public CBase_Packed
{
  int creator;

 public:
  Packed(int c) : creator(c) {}
  Packed(CkMigrateMessage* m) : CBase_Packed(m) {}
  void pup(PUP::er& p) { p | creator; }
  void report()
  {
    mainProxy.report(PACKED, thisIndex.x, thisIndex.y, thisIndex.z,
                     ckGetID().getElementID(), CkMyPe());
  }
  void hop() { migrateMe((CkMyPe() + 1) % CkNumPes()); }
};

class Hashed : public CBase_Hashed
{
  int creator;

 public:
  Hashed(int c) : creator(c) {}
  Hashed(CkMigrateMessage* m) : CBase_Hashed(m) {}
  void pup(PUP::er& p) { p | creator; }
  void report()
  {
    mainProxy.report(HASHED, thisIndex.x, thisIndex.y, 0, ckGetID().getElementID(),
                     CkMyPe());
  }
  void hop() { migrateMe((CkMyPe() + 1) % CkNumPes()); }
};

#include "objid_home.def.h"
