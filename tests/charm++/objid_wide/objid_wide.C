// objid_wide: many chare arrays with the wide-collection element id layout.
//
// What is exercised (src/ck-core/objid.h, doc/objid64-design.md section 12):
//
//   - The 64-bit object id has two layouts: standard (tag 000, 12 collection
//     bits, 49 payload bits) and wide (tag 001, 24 collection bits, 37 payload
//     bits), chosen per array with CkArrayOptions::setWideCollection(). Each
//     array gets a dense collection number on PE 0 at creation: standard ones
//     in [1, STANDARD_LIMIT), wide ones in [WIDE_FIRST, WIDE_LIMIT).
//   - The program creates -wide N wide arrays (default 20000, well past the
//     4095 standard numbers) and -standard M standard arrays (default 3), each
//     a bounded 1D array of 8 elements, so the payload is the packed index.
//   - -bound: one more array per array above, bound to it (bindTo), created
//     with the OPPOSITE layout option: a bound array takes the layout of the
//     array it binds to, whatever its own option says, and still gets its own
//     collection number. Bound arrays follow their manager through every
//     phase: only the manager's elements call migrateMe / AtSync, and every
//     identity pass checks that each bound element sits on its sibling's PE.
//     Array a's bound partner is array a + (N + M).
//
// Phases, driven by Main with counted messages:
//   1. Identity: every element reports (a, i, id, PE). Main checks the layout
//      and range of every collection number, that the 8 elements of an array
//      share it, that numbers are distinct and the wide ones dense, that the
//      payload is the index, and that all id words are distinct.
//   2. Point-to-point by index across arrays: (a, i) pings ((a+1) % total,
//      (i+3) % 8); the receiver checks the sender's id word against the id its
//      own branch of the sender's array computes, and pongs back by index.
//   3. Migration: every element migrateMe's to the next PE (default), or calls
//      AtSync under -lb +balancer X; then phase 2 again.
//   4. -ckpt: Main keeps the phase-1 id words (pupped) and checkpoints to
//      ckptlog. A run with +restart ckptlog re-runs phase 1 (every id word must
//      equal the stored one), phase 2, then creates 10 more wide arrays and 2
//      more standard arrays, whose numbers must exceed every restored number of
//      their kind.
//   5. -abort nobounds / -abort bigbounds: a wide array without bounds (empty
//      CkArrayOptions; CkArrayOptions(n) sets bounds n), and a
//      wide 2D array whose bounds need 40 bits; both must abort at creation.
//
// Run: ./objid_wide +p4 [-wide N] [-standard M] [-bound] [-lb +balancer GreedyLB] [-ckpt]
//      ./objid_wide +p2 +restart ckptlog

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>
#include "pup_stl.h"
#include "objid_wide.decl.h"

// Every process: unbuffered stdout, so an abort does not discard the last lines.
void unbufferStdout(void) { setvbuf(stdout, NULL, _IONBF, 0); }

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ std::vector<CProxy_Elem> arrays;  // wide first, then standard
/*readonly*/ int nWide;
/*readonly*/ int nStd;
/*readonly*/ int lbMode;
/*readonly*/ int boundMode;

static const int nPer = 8;  // elements per array
static const int nNewWide = 10, nNewStd = 2;  // arrays created after a restart

// Arrays [0, nBase) own their location manager: wide first, then standard.
// With -bound, arrays [nBase, 2 nBase) are bound to them, in the same order.
static inline int nBase() { return nWide + nStd; }
static inline int nTotal() { return boundMode ? 2 * nBase() : nBase(); }
static inline bool isWide(int a) { return a % nBase() < nWide; }
static inline bool isBound(int a) { return a >= nBase(); }

// Checks that `id` is the id of element si of array sa (one of `arrays`), using
// this PE's branch of that array.
static void checkPeerId(const char* what, int sa, int si, CmiUInt8 idWord)
{
  ck::ObjID id(idWord);
  const bool wide = isWide(sa);
  ck::ObjID expect = arrays[sa].ckLocalBranch()->makeID(si);
  if (id.isWide() != wide || (int)id.getElementID() != si || id != expect)
    CkAbort("objid_wide: %s: id 0x%llx of array %d element %d on PE %d does not decode"
            " to it: wide %d (expected %d), element %llu, collection %llu; this PE's"
            " branch computes 0x%llx\n",
            what, (unsigned long long)idWord, sa, si, CkMyPe(), (int)id.isWide(),
            (int)wide, (unsigned long long)id.getElementID(),
            (unsigned long long)id.getCollectionNumber(),
            (unsigned long long)expect.getID());
}

class Main : public CBase_Main
{
  std::vector<CmiUInt8> ids;  // phase-1 id word of element (a, i) at a*8+i; pupped
  // Per run, not pupped:
  std::vector<CmiUInt8> cur;  // ids reported in this identity pass
  std::vector<int> pe1;       // PE of each element in phase 1
  std::vector<CProxy_Elem> newArrays;
  std::vector<CmiUInt8> newIds;
  int reported = 0, pongs = 0, resumes = 0, newReported = 0;
  int round = 0;  // ping rounds done
  int moved = 0;
  bool afterMig = false;   // the identity pass after phase 3
  std::vector<int> savedPe;  // PEs before phase 3
  bool ckptMode = false;
  double t0 = 0, tPhase = 0;

  void lap(const char* what)
  {
    double t = CkWallTimer();
    CkPrintf("objid_wide: %s (%.2f s, total %.2f s)\n", what, t - tPhase, t - t0);
    tPhase = t;
  }

public:
  Main(CkArgMsg* m)
  {
    setvbuf(stdout, NULL, _IONBF, 0);
    mainProxy = thisProxy;
    nWide = 20000;
    nStd = 3;
    char* abortCase = nullptr;
    CmiGetArgInt(m->argv, "-wide", &nWide);
    CmiGetArgInt(m->argv, "-standard", &nStd);
    lbMode = CmiGetArgFlag(m->argv, "-lb") ? 1 : 0;
    boundMode = CmiGetArgFlag(m->argv, "-bound") ? 1 : 0;
    ckptMode = CmiGetArgFlag(m->argv, "-ckpt") != 0;
    CmiGetArgString(m->argv, "-abort", &abortCase);
    delete m;
    t0 = tPhase = CkWallTimer();

    if (abortCase)
    {
      if (strcmp(abortCase, "nobounds") == 0)
      {
        // Not CkArrayOptions(8): that constructor sets bounds of 8 as well.
        CkPrintf("objid_wide: abort case: wide array without bounds\n");
        CkArrayOptions opts;
        opts.setWideCollection();
        CProxy_Elem::ckNew(0, opts);
      }
      else if (strcmp(abortCase, "bigbounds") == 0)
      {
        CkPrintf("objid_wide: abort case: wide 2D array with 2^20 x 2^20 bounds\n");
        CkArrayOptions opts;
        opts.setBounds(1 << 20, 1 << 20);
        opts.setWideCollection();
        opts.setNumInitial(1, 1);
        CProxy_Elem::ckNew(0, opts);
      }
      else
        CkAbort("objid_wide: unknown -abort case '%s'\n", abortCase);
      CkPrintf("objid_wide: FAIL: abort case %s did not abort at creation\n", abortCase);
      CkExit(1);
      return;
    }

    CkEnforce(nWide >= 0 && nStd >= 0 && nTotal() >= 1);
    CkEnforce(CkNumPes() >= 2);  // phase 3 migrates
    CkPrintf("objid_wide: %d PEs, %d wide + %d standard arrays of %d elements%s, mode %s%s\n",
             CkNumPes(), nWide, nStd, nPer,
             boundMode ? ", each with a bound partner" : "",
             lbMode ? "load balancer (AtSync)" : "manual (migrateMe)",
             ckptMode ? ", checkpoint at the end" : "");
    arrays.reserve(nTotal());
    for (int a = 0; a < nBase(); a++)
    {
      CkArrayOptions opts(nPer);
      opts.setBounds(nPer);
      if (isWide(a)) opts.setWideCollection();
      arrays.push_back(CProxy_Elem::ckNew(a, opts));
    }
    // Bound partners ask for the opposite layout; the manager's must win.
    for (int a = nBase(); a < nTotal(); a++)
    {
      CkArrayOptions opts(nPer);
      opts.setBounds(nPer);
      if (!isWide(a)) opts.setWideCollection();
      opts.bindTo(arrays[a - nBase()]);
      arrays.push_back(CProxy_Elem::ckNew(a, opts));
    }
    lap("arrays created");
    identifyAll();
  }

  Main(CkMigrateMessage* m) : CBase_Main(m)
  {
    mainProxy = thisProxy;
    CkPrintf("objid_wide: Main restored on %d PEs, %d processes\n", CkNumPes(),
             CkNumNodes());
  }

  // round is pupped too: the elements' pupped ping counters continue from it.
  void pup(PUP::er& p)
  {
    p | ids;
    p | round;
  }

  // ---- phase 1: identity ----
  void identifyAll()
  {
    reported = 0;
    cur.assign((size_t)nTotal() * nPer, 0);
    pe1.assign((size_t)nTotal() * nPer, -1);
    for (auto& ap : arrays) ap.identify();
  }

  void reportId(int a, int i, CmiUInt8 id, int pe)
  {
    CkEnforce(i >= 0 && i < nPer && pe >= 0 && pe < CkNumPes());
    if (a >= nTotal())
    {
      newReport(a, i, id);
      return;
    }
    CkEnforce(a >= 0);
    size_t k = (size_t)a * nPer + i;
    if (pe1[k] != -1) CkAbort("objid_wide: element (%d, %d) reported twice\n", a, i);
    cur[k] = id;
    pe1[k] = pe;
    if (++reported < nTotal() * nPer) return;
    if (afterMig) finishMigrationCheck();
    else checkIdentity();
  }

  // Checks one array's 8 ids; returns its collection number.
  CmiUInt8 checkArray(int a, bool wide, const CmiUInt8* w)
  {
    const CmiUInt8 num = ck::ObjID(w[0]).getCollectionNumber();
    for (int i = 0; i < nPer; i++)
    {
      ck::ObjID id(w[i]);
      bool ok = id.isWide() == wide && id.getCollectionNumber() == num &&
                (int)id.getElementID() == i;
      if (wide)
        ok = ok && num >= ck::ObjID::limits::WIDE_FIRST && num < ck::ObjID::limits::WIDE_LIMIT;
      else
        ok = ok && num >= 1 && num < ck::ObjID::limits::STANDARD_LIMIT;
      if (!ok)
        CkAbort("objid_wide: array %d (%s) element %d: id 0x%llx: wide %d, collection"
                " %llu (element 0 has %llu), element %llu\n",
                a, wide ? "wide" : "standard", i, (unsigned long long)w[i],
                (int)id.isWide(), (unsigned long long)id.getCollectionNumber(),
                (unsigned long long)num, (unsigned long long)id.getElementID());
    }
    return num;
  }

  void checkIdentity()
  {
    std::set<CmiUInt8> numbers;  // (kind, number) keys
    std::set<CmiUInt8> words;
    std::vector<CmiUInt8> wideNums;
    CmiUInt8 maxStd = 0;
    checkBoundPes();
    for (int a = 0; a < nTotal(); a++)
    {
      const bool wide = isWide(a);
      CmiUInt8 num = checkArray(a, wide, &cur[(size_t)a * nPer]);
      if (!numbers.insert(num | (wide ? (1ULL << 40) : 0)).second)
        CkAbort("objid_wide: array %d has collection number %llu, already taken\n", a,
                (unsigned long long)num);
      if (wide) wideNums.push_back(num);
      else maxStd = std::max(maxStd, num);
      for (int i = 0; i < nPer; i++)
        if (!words.insert(cur[(size_t)a * nPer + i]).second)
          CkAbort("objid_wide: id word 0x%llx of (%d, %d) is not unique\n",
                  (unsigned long long)cur[(size_t)a * nPer + i], a, i);
    }
    std::sort(wideNums.begin(), wideNums.end());
    for (size_t j = 1; j < wideNums.size(); j++)
      if (wideNums[j] != wideNums[j - 1] + 1)
        CkAbort("objid_wide: wide collection numbers not dense: %llu then %llu\n",
                (unsigned long long)wideNums[j - 1], (unsigned long long)wideNums[j]);
    CkPrintf("objid_wide: identity OK: %d wide + %d standard arrays%s, %d distinct"
             " collection numbers, %d distinct id words; wide numbers %llu..%llu"
             " (WIDE_FIRST %llu), max standard number %llu\n",
             nWide, nStd, boundMode ? " (half of them bound)" : "", (int)numbers.size(),
             (int)words.size(),
             wideNums.empty() ? 0ULL : (unsigned long long)wideNums.front(),
             wideNums.empty() ? 0ULL : (unsigned long long)wideNums.back(),
             (unsigned long long)ck::ObjID::limits::WIDE_FIRST,
             (unsigned long long)maxStd);
    if (_restarted)
    {
      CkEnforceMsg(ids.size() == cur.size(), "restored Main lost the stored ids");
      for (size_t k = 0; k < ids.size(); k++)
        if (ids[k] != cur[k])
          CkAbort("objid_wide: id of (%d, %d) changed across the restart: 0x%llx before,"
                  " 0x%llx now\n",
                  (int)(k / nPer), (int)(k % nPer), (unsigned long long)ids[k],
                  (unsigned long long)cur[k]);
      CkPrintf("objid_wide: all %d id words equal the ones stored before the checkpoint\n",
               (int)ids.size());
    }
    else
      ids = cur;
    lap("phase 1 (identity) done");
    pingAll();
  }

  // Every bound element is on the PE of the element it is bound to (pe1 holds
  // the PEs of the identity pass just completed).
  void checkBoundPes()
  {
    for (int a = nBase(); a < nTotal(); a++)
      for (int i = 0; i < nPer; i++)
        if (pe1[(size_t)a * nPer + i] != pe1[(size_t)(a - nBase()) * nPer + i])
          CkAbort("objid_wide: bound element (%d, %d) is on PE %d, its sibling (%d, %d)"
                  " on PE %d\n",
                  a, i, pe1[(size_t)a * nPer + i], a - nBase(), i,
                  pe1[(size_t)(a - nBase()) * nPer + i]);
  }

  // ---- phase 2: point-to-point ----
  void pingAll()
  {
    pongs = 0;
    for (auto& ap : arrays) ap.startPing(round);
  }

  void pongDone(int a, int i)
  {
    CkEnforce(a >= 0 && a < nTotal() && i >= 0 && i < nPer);
    if (++pongs < nTotal() * nPer) return;
    char buf[64];
    snprintf(buf, sizeof(buf), "phase 2 round %d: %d pongs", round, pongs);
    lap(buf);
    round++;
    if (_restarted)
    {
      createNew();
      return;
    }
    if (round == 1)
    {
      migrateAll();
      return;
    }
    // round 2 done: after migration
    identifyAfterMigration();
  }

  // ---- phase 3: migration ----
  // Only the arrays that own their location manager migrate or sync; a bound
  // partner moves with its sibling's location record.
  void migrateAll()
  {
    if (lbMode)
    {
      resumes = 0;
      for (int a = 0; a < nBase(); a++) arrays[a].doSync();
      return;
    }
    for (int a = 0; a < nBase(); a++) arrays[a].migrateNow();
    CkStartQD(CkCallback(CkIndex_Main::qdDone(), thisProxy));
  }

  void qdDone()
  {
    lap("phase 3 (migrateMe) quiescent");
    pingAll();
  }

  void resumed(int a, int i)
  {
    CkEnforce(lbMode && a >= 0 && a < nBase() && i >= 0 && i < nPer);
    if (++resumes < nBase() * nPer) return;
    lap("phase 3 (AtSync) all resumed");
    pingAll();
  }

  // After phase 3's ping round, one more identity pass checks where elements are
  // now and that the ids did not change with migration.
  void identifyAfterMigration()
  {
    savedPe = pe1;
    reported = 0;
    pe1.assign(savedPe.size(), -1);
    cur.assign(savedPe.size(), 0);
    afterMig = true;
    for (auto& ap : arrays) ap.identify();
  }

  void finishMigrationCheck()
  {
    checkBoundPes();
    moved = 0;
    for (size_t k = 0; k < cur.size(); k++)
    {
      if (cur[k] != ids[k])
        CkAbort("objid_wide: id of (%d, %d) changed with migration\n", (int)(k / nPer),
                (int)(k % nPer));
      if (pe1[k] != savedPe[k]) moved++;
      if (!lbMode && pe1[k] != (savedPe[k] + 1) % CkNumPes())
        CkAbort("objid_wide: (%d, %d) on PE %d after migrateMe, expected PE %d\n",
                (int)(k / nPer), (int)(k % nPer), pe1[k], (savedPe[k] + 1) % CkNumPes());
    }
    CkPrintf("objid_wide: after phase 3, %d of %d elements are on another PE; ids"
             " unchanged\n",
             moved, (int)cur.size());
    if (ckptMode)
    {
      lap("checkpointing to ckptlog");
      CkStartCheckpoint("ckptlog", CkCallback(CkIndex_Main::checkpointed(), thisProxy));
      return;
    }
    lap("All done");
    CkExit();
  }

  // ---- phase 4: checkpoint / restart ----
  void checkpointed()
  {
    if (!_restarted)
    {
      CkPrintf("objid_wide: Checkpoint written\n");
      CkExit();
      return;
    }
    t0 = tPhase = CkWallTimer();
    CkPrintf("objid_wide: restarted with %d wide + %d standard arrays, %d stored ids\n",
             nWide, nStd, (int)ids.size());
    identifyAll();
  }

  void createNew()
  {
    newReported = 0;
    newIds.assign((size_t)(nNewWide + nNewStd) * nPer, 0);
    for (int k = 0; k < nNewWide + nNewStd; k++)
    {
      CkArrayOptions opts(nPer);
      opts.setBounds(nPer);
      if (k < nNewWide) opts.setWideCollection();
      newArrays.push_back(CProxy_Elem::ckNew(nTotal() + k, opts));
    }
    for (auto& ap : newArrays) ap.identify();
  }

  void newReport(int a, int i, CmiUInt8 id)
  {
    int k = a - nTotal();
    CkEnforce(_restarted && k >= 0 && k < nNewWide + nNewStd);
    CkEnforceMsg(newIds[(size_t)k * nPer + i] == 0, "new element reported twice");
    newIds[(size_t)k * nPer + i] = id;
    if (++newReported < (nNewWide + nNewStd) * nPer) return;

    CmiUInt8 maxW = 0, maxS = 0;
    for (int a2 = 0; a2 < nTotal(); a2++)
    {
      CmiUInt8 n = ck::ObjID(ids[(size_t)a2 * nPer]).getCollectionNumber();
      if (isWide(a2)) maxW = std::max(maxW, n);
      else maxS = std::max(maxS, n);
    }
    std::set<CmiUInt8> words(ids.begin(), ids.end());
    std::set<CmiUInt8> numbers;
    CmiUInt8 minNW = ~0ULL, maxNW = 0, minNS = ~0ULL, maxNS = 0;
    for (int k2 = 0; k2 < nNewWide + nNewStd; k2++)
    {
      const bool wide = k2 < nNewWide;
      CmiUInt8 num = checkArray(nTotal() + k2, wide, &newIds[(size_t)k2 * nPer]);
      if (!numbers.insert(num | (wide ? (1ULL << 40) : 0)).second)
        CkAbort("objid_wide: new array %d reuses collection number %llu\n", k2,
                (unsigned long long)num);
      if (wide ? num <= maxW : num <= maxS)
        CkAbort("objid_wide: new %s array %d got collection number %llu, not above the"
                " largest restored %s number %llu\n",
                wide ? "wide" : "standard", k2, (unsigned long long)num,
                wide ? "wide" : "standard", (unsigned long long)(wide ? maxW : maxS));
      if (wide) { minNW = std::min(minNW, num); maxNW = std::max(maxNW, num); }
      else { minNS = std::min(minNS, num); maxNS = std::max(maxNS, num); }
      for (int i2 = 0; i2 < nPer; i2++)
        if (!words.insert(newIds[(size_t)k2 * nPer + i2]).second)
          CkAbort("objid_wide: new id word 0x%llx collides\n",
                  (unsigned long long)newIds[(size_t)k2 * nPer + i2]);
    }
    CkPrintf("objid_wide: %d new wide arrays got numbers %llu..%llu (restored max %llu),"
             " %d new standard arrays %llu..%llu (restored max %llu); all ids distinct\n",
             nNewWide, (unsigned long long)minNW, (unsigned long long)maxNW,
             (unsigned long long)maxW, nNewStd, (unsigned long long)minNS,
             (unsigned long long)maxNS, (unsigned long long)maxS);
    lap("All done");
    CkExit();
  }
};

class Elem : public CBase_Elem
{
  int myA;
  int lastPing = -1, lastPong = -1;

public:
  Elem(int a) : myA(a) { usesAtSync = lbMode != 0 && !isBound(a); }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}

  void pup(PUP::er& p)
  {
    p | myA;
    p | lastPing;
    p | lastPong;
  }

  void identify() { mainProxy.reportId(myA, thisIndex, ckGetID().getID(), CkMyPe()); }

  void startPing(int round)
  {
    CkEnforce(myA < nTotal());
    int ta = (myA + 1) % nTotal(), ti = (thisIndex + 3) % nPer;
    arrays[ta][ti].ping(round, myA, thisIndex, ckGetID().getID());
  }

  void ping(int round, int sa, int si, CmiUInt8 sid)
  {
    if ((sa + 1) % nTotal() != myA || (si + 3) % nPer != thisIndex || round != lastPing + 1)
      CkAbort("objid_wide: (%d, %d) got ping round %d (expected %d) from (%d, %d)\n", myA,
              thisIndex, round, lastPing + 1, sa, si);
    lastPing = round;
    checkPeerId("ping", sa, si, sid);
    arrays[sa][si].pong(round, myA, thisIndex, ckGetID().getID());
  }

  void pong(int round, int ra, int ri, CmiUInt8 rid)
  {
    if ((myA + 1) % nTotal() != ra || (thisIndex + 3) % nPer != ri || round != lastPong + 1)
      CkAbort("objid_wide: (%d, %d) got pong round %d (expected %d) from (%d, %d)\n", myA,
              thisIndex, round, lastPong + 1, ra, ri);
    lastPong = round;
    checkPeerId("pong", ra, ri, rid);
    mainProxy.pongDone(myA, thisIndex);
  }

  void migrateNow()
  {
    CkEnforce(!lbMode && !isBound(myA));
    migrateMe((CkMyPe() + 1) % CkNumPes());  // last action
  }

  void doSync()
  {
    CkEnforce(lbMode && !isBound(myA));
    AtSync();
  }

  void ResumeFromSync() { mainProxy.resumed(myA, thisIndex); }
};

#include "objid_wide.def.h"
