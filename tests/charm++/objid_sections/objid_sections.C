// Array sections (ckmulticast) on a HASHED-kind array under migration.
//
// Object id redesign (charm #3994, doc/objid64-design.md): an array created
// without bounds has hashed-kind element ids, minted from a per-process
// tranche when the element is inserted (section 3.1); the id is never
// recomputed and carries no PE number, so it is stable for the element's
// lifetime and across migration (sections 3 and 6). ckmulticast section
// cookies and spanning trees hold element ids (mCastEntry ObjKeyList), so a
// section built once must keep reaching its members wherever they move.
//
// What this checks:
//   - section cookies carry element ids: two overlapping sections (even
//     indices; indices 0..15) are built once, before any migration, and
//     reused for every round;
//   - delivery to migrated members by id: every round, after both section
//     reductions complete, every element migrates to a pseudo-random other
//     PE, so each multicast of the next round goes to members that are no
//     longer where the section's tree last saw them (ckmulticast's
//     SimpleSend / rebuild path, which resolves members by id);
//   - each member enforces that it belongs to the section it was reached
//     through (the message carries the section tag) and that rounds arrive
//     exactly once and in order;
//   - section reductions after migration: main checks each section's sum of
//     member indices and member count every round;
//   - hashed-kind ids are stable: the element's id (ckGetID) recorded at
//     insertion is enforced unchanged after every migration.
//
// Failure modes: hang (lost multicast to a migrated member, or lost section
// reduction contribution; the watchdog aborts after 60 s without progress),
// abort on an enforce, or crash.
//
// Run (reconverse): ./objid_sections +pe 4
//   multi-process   lcrun -n 2 ./objid_sections +pe 4

#include "objid_sections.decl.h"
#include "ckmulticast.h"
#include <stdio.h>
#include <vector>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Elem elemProxy;
/*readonly*/ CkGroupID mcastGid;

static const int kElems = 32;
static const int kRounds = 30;
static const int kLowLimit = 16;   // section LOW = indices [0, kLowLimit)
static const double kWatchdogSecs = 60.0;

enum { SEC_EVEN = 0, SEC_LOW = 1 };

static inline bool inSection(int tag, int idx) {
  return tag == SEC_EVEN ? (idx % 2 == 0) : (idx < kLowLimit);
}

static inline unsigned long long mix64(unsigned long long z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

struct WorkMsg : public CkMcastBaseMsg, public CMessage_WorkMsg {
  int round;
  int tag;
  WorkMsg(int r, int t) : round(r), tag(t) {}
};

static void watchdogFire(void*, double) { mainProxy.checkProgress(); }

class Main : public CBase_Main {
  CProxySection_Elem secEven, secLow;
  int round = 0;
  int doneCount = 0;
  bool evenSeen = false, lowSeen = false;
  double lastProgress;

  void checkSum(int tag, int n, int* vals) {
    CkEnforce(n == 3);
    long long expSum = 0, expCount = 0;
    for (int i = 0; i < kElems; i++)
      if (inSection(tag, i)) { expSum += i; expCount++; }
    if (vals[0] != expSum || vals[1] != expCount || vals[2] != expCount * round)
      CkPrintf("objid_sections: round %d section %s MISMATCH: sum %d count %d "
               "roundsum %d, expected %lld %lld %lld\n", round,
               tag == SEC_EVEN ? "even" : "low", vals[0], vals[1], vals[2],
               expSum, expCount, expCount * round);
    CkEnforce(vals[0] == expSum && vals[1] == expCount &&
              vals[2] == expCount * round);
    lastProgress = CkWallTimer();
    if (++doneCount == 2) {
      doneCount = 0; evenSeen = lowSeen = false;
      elemProxy.migrateNow(round);
      CkStartQD(CkCallback(CkIndex_Main::startRound(), thisProxy));
    }
  }

public:
  Main(CkArgMsg* m) {
    delete m;
    setvbuf(stdout, NULL, _IONBF, 0);
    mainProxy = thisProxy;
    CkPrintf("objid_sections: %d PEs, %d processes, %d elements, %d rounds, "
             "unbounded array (hashed ids)\n", CkNumPes(), CkNumNodes(),
             kElems, kRounds);
    elemProxy = CProxy_Elem::ckNew(CkArrayOptions());
    for (int i = 0; i < kElems; i++) elemProxy[i].insert(i % CkNumPes());
    elemProxy.doneInserting();
    mcastGid = CProxy_CkMulticastMgr::ckNew();
    lastProgress = CkWallTimer();
    CcdCallFnAfter(watchdogFire, NULL, 2000);
    // Build the sections once every element exists and the mcast group is up.
    CkStartQD(CkCallback(CkIndex_Main::setup(), thisProxy));
  }

  void setup() {
    std::vector<CkArrayIndex> even, low;
    for (int i = 0; i < kElems; i++) {
      if (inSection(SEC_EVEN, i)) even.emplace_back(CkArrayIndex1D(i));
      if (inSection(SEC_LOW, i)) low.emplace_back(CkArrayIndex1D(i));
    }
    CkMulticastMgr* mg = CProxy_CkMulticastMgr(mcastGid).ckLocalBranch();
    secEven = CProxySection_Elem::ckNew(elemProxy.ckGetArrayID(), even);
    secLow = CProxySection_Elem::ckNew(elemProxy.ckGetArrayID(), low);
    secEven.ckSectionDelegate(mg);
    secLow.ckSectionDelegate(mg);
    // setReductionClient keeps the pointer (mCastEntry::red.storedCallback),
    // so the callbacks must outlive this method.
    mg->setReductionClient(secEven,
        new CkCallback(CkReductionTarget(Main, evenDone), thisProxy));
    mg->setReductionClient(secLow,
        new CkCallback(CkReductionTarget(Main, lowDone), thisProxy));
    round = 0;
    sendRound();
  }

  void sendRound() {
    lastProgress = CkWallTimer();
    secEven.work(new WorkMsg(round, SEC_EVEN));
    secLow.work(new WorkMsg(round, SEC_LOW));
  }

  void startRound() {
    round++;
    if (round == kRounds) {
      CkPrintf("objid_sections PASS\n");
      CkExit();
      return;
    }
    sendRound();
  }

  void evenDone(int n, int* vals) {
    CkEnforce(!evenSeen); evenSeen = true;
    checkSum(SEC_EVEN, n, vals);
  }
  void lowDone(int n, int* vals) {
    CkEnforce(!lowSeen); lowSeen = true;
    checkSum(SEC_LOW, n, vals);
  }

  void checkProgress() {
    if (CkWallTimer() - lastProgress > kWatchdogSecs) {
      CkPrintf("objid_sections WATCHDOG: no progress for %.0f s at round %d "
               "(even %s, low %s)\n", kWatchdogSecs, round,
               evenSeen ? "done" : "pending", lowSeen ? "done" : "pending");
      CkAbort("objid_sections: hung");
    }
    CcdCallFnAfter(watchdogFire, NULL, 2000);
  }
};

class Elem : public CBase_Elem {
  CkSectionInfo cookie[2];
  int nextRound[2];     // next expected round per section
  CmiUInt8 myId;        // element id recorded at insertion
  int migrations;

  void checkId() {
    CmiUInt8 id = ckGetID().getID();
    if (id != myId)
      CkPrintf("objid_sections: elem %d id changed 0x%llx -> 0x%llx after "
               "%d migrations\n", thisIndex, (unsigned long long)myId,
               (unsigned long long)id, migrations);
    CkEnforce(id == myId);
  }

public:
  Elem() : migrations(0) {
    nextRound[0] = nextRound[1] = 0;
    myId = ckGetID().getID();
  }
  Elem(CkMigrateMessage* m) : CBase_Elem(m) {}

  void pup(PUP::er& p) {
    p | cookie[0]; p | cookie[1];
    p | nextRound[0]; p | nextRound[1];
    p | myId; p | migrations;
  }

  void work(WorkMsg* m) {
    checkId();
    int tag = m->tag, r = m->round;
    CkEnforce(tag == SEC_EVEN || tag == SEC_LOW);
    if (!inSection(tag, thisIndex))
      CkPrintf("objid_sections: elem %d reached through section %d it is not "
               "a member of (round %d)\n", thisIndex, tag, r);
    CkEnforce(inSection(tag, thisIndex));
    if (r != nextRound[tag])
      CkPrintf("objid_sections: elem %d section %d got round %d, expected %d\n",
               thisIndex, tag, r, nextRound[tag]);
    CkEnforce(r == nextRound[tag]);
    // One migration per completed round (state carried by pup).
    CkEnforce(CkNumPes() < 2 || migrations == r);
    nextRound[tag]++;
    CkGetSectionInfo(cookie[tag], m);
    delete m;
    int v[3] = { thisIndex, 1, r };
    CkMulticastMgr* mg = CProxy_CkMulticastMgr(mcastGid).ckLocalBranch();
    mg->contribute(sizeof(v), v, CkReduction::sum_int, cookie[tag]);
  }

  void migrateNow(int r) {
    checkId();
    int P = CkNumPes();
    if (P < 2) return;
    int hop = 1 + (int)(mix64(((unsigned long long)(unsigned)r << 32) |
                              (unsigned)thisIndex) % (P - 1));
    int dest = (CkMyPe() + hop) % P;
    migrations++;
    CkEnforce(dest != CkMyPe());
    migrateMe(dest);  // last action
  }
};

#include "objid_sections.def.h"
