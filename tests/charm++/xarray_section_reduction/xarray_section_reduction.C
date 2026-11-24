// Reductions over a cross-array CkMulticast section: a section proxy built
// over elements of two different 1-D arrays (CProxySection_Elem(aids, elems)),
// which CkMulticast reduces per array and then combines on the root
// (XArraySectionReducer).
//
// Each round main multicasts work(round) to the section, and every member
// contributes two section reductions with the cookie from that message, each
// with its own callback and user flag:
//   - max_int of one value; the callback is maxResultA in even rounds and
//     maxResultB in odd rounds, the user flag 100 + round;
//   - sumCount, a reducer registered here that is not streamable and builds
//     a new message, of {value, 1}; callback sumCountResult, user flag
//     200 + round.
// Main checks in each callback the value, the reducer type, getUserFlag(),
// the reference number, and that the callback invoked is the one the members
// passed. The next round starts once both results of a round arrived.
// A watchdog aborts the run after 20 s without progress.

#include "xarray_section_reduction.decl.h"
#include "ckmulticast.h"
#include <stdlib.h>
#include <vector>

/*readonly*/ CProxy_Main mainProxy;

static const int kNumArrays = 2;
static const int kArraySize[kNumArrays] = {7, 5};
static const int kRounds = 3;
static const double kWatchdogSecs = 20.0;

static CkReduction::reducerType sumCountType;

// Members: the even indices of array 0 and the indices >= 1 of array 1.
static inline bool isMember(int arrayNum, int index) {
  return arrayNum == 0 ? index % 2 == 0 : index >= 1;
}

// The value each member contributes; the maximum moves between the arrays
// from round to round.
static inline int valueOf(int arrayNum, int index, int round) {
  return (arrayNum * 31 + index * 17 + round * 7) % 53;
}

static inline int maxFlag(int round) { return 100 + round; }
static inline int sumCountFlag(int round) { return 200 + round; }

// Element-wise sum of int pairs, into a newly built message.
static CkReductionMsg *sumCount(int nMsg, CkReductionMsg **msgs) {
  int out[2] = {0, 0};
  for (int i = 0; i < nMsg; i++) {
    const int *in = (const int *)msgs[i]->getData();
    out[0] += in[0];
    out[1] += in[1];
  }
  return CkReductionMsg::buildNew(sizeof(out), out);
}

void registerSumCount(void) {
  sumCountType = CkReduction::addReducer(sumCount, false, "sumCount");
}

struct WorkMsg : public CkMcastBaseMsg, public CMessage_WorkMsg {
  int round;
  WorkMsg(int r) : round(r) {}
};

static void watchdogFire(void *, double) { mainProxy.checkProgress(); }

class Main : public CBase_Main {
  CProxySection_Elem section;
  int round = 0;
  int resultsThisRound = 0;
  int progress = 0;
  int lastProgress = -1;
  double lastProgressTime = 0;

  void expected(int r, int &max, int &sum, int &count) {
    max = -1; sum = 0; count = 0;
    for (int a = 0; a < kNumArrays; a++)
      for (int i = 0; i < kArraySize[a]; i++)
        if (isMember(a, i)) {
          int v = valueOf(a, i, r);
          if (v > max) max = v;
          sum += v;
          count++;
        }
  }

  void checkMax(CkReductionMsg *m, char which) {
    char want = round % 2 == 0 ? 'A' : 'B';
    if (which != want)
      CkAbort("xarray_section_reduction: round %d max result went to maxResult%c, "
              "the members passed maxResult%c", round, which, want);
    int max, sum, count;
    expected(round, max, sum, count);
    if (m->getSize() != sizeof(int) || *(int *)m->getData() != max)
      CkAbort("xarray_section_reduction: round %d max: got %d, expected %d",
              round, *(int *)m->getData(), max);
    if (m->getReducer() != CkReduction::max_int)
      CkAbort("xarray_section_reduction: round %d max: reducer %d, expected max_int (%d)",
              round, (int)m->getReducer(), (int)CkReduction::max_int);
    if (m->getUserFlag() != maxFlag(round) || CkGetRefNum(m) != maxFlag(round))
      CkAbort("xarray_section_reduction: round %d max: user flag %d, ref num %d, expected %d",
              round, (int)m->getUserFlag(), (int)CkGetRefNum(m), maxFlag(round));
    delete m;
    resultArrived();
  }

  void resultArrived() {
    progress++;
    if (++resultsThisRound < 2) return;
    CkPrintf("xarray_section_reduction: round %d done\n", round);
    resultsThisRound = 0;
    if (++round == kRounds) {
      CkPrintf("xarray_section_reduction: all %d rounds passed\n", kRounds);
      CkExit();
      return;
    }
    section.work(new WorkMsg(round));
  }

public:
  Main(CkArgMsg *m) {
    delete m;
    mainProxy = thisProxy;
    std::vector<CkArrayID> aids(kNumArrays);
    std::vector<std::vector<CkArrayIndex>> elems(kNumArrays);
    for (int a = 0; a < kNumArrays; a++) {
      aids[a] = CProxy_Elem::ckNew(a, kArraySize[a]);
      for (int i = 0; i < kArraySize[a]; i++)
        if (isMember(a, i)) elems[a].push_back(CkArrayIndex1D(i));
    }
    // A section proxy with several array ids is delegated to CkMulticast
    section = CProxySection_Elem(aids, elems);
    lastProgressTime = CkWallTimer();
    CcdCallFnAfter(watchdogFire, NULL, 1000);
    section.work(new WorkMsg(round));
  }

  void maxResultA(CkReductionMsg *m) { checkMax(m, 'A'); }
  void maxResultB(CkReductionMsg *m) { checkMax(m, 'B'); }

  void sumCountResult(CkReductionMsg *m) {
    int max, sum, count;
    expected(round, max, sum, count);
    const int *got = (const int *)m->getData();
    if (m->getSize() != 2 * sizeof(int) || got[0] != sum || got[1] != count)
      CkAbort("xarray_section_reduction: round %d sumCount: got {%d,%d}, expected {%d,%d}",
              round, got[0], got[1], sum, count);
    if (m->getReducer() != sumCountType)
      CkAbort("xarray_section_reduction: round %d sumCount: reducer %d, expected %d",
              round, (int)m->getReducer(), (int)sumCountType);
    if (m->getUserFlag() != sumCountFlag(round) || CkGetRefNum(m) != sumCountFlag(round))
      CkAbort("xarray_section_reduction: round %d sumCount: user flag %d, ref num %d, expected %d",
              round, (int)m->getUserFlag(), (int)CkGetRefNum(m), sumCountFlag(round));
    delete m;
    resultArrived();
  }

  void checkProgress() {
    double now = CkWallTimer();
    if (progress != lastProgress) {
      lastProgress = progress;
      lastProgressTime = now;
    } else if (now - lastProgressTime > kWatchdogSecs) {
      CkAbort("xarray_section_reduction: no progress for %.0f s at round %d "
              "(%d of 2 results)", kWatchdogSecs, round, resultsThisRound);
    }
    CcdCallFnAfter(watchdogFire, NULL, 1000);
  }
};

class Elem : public CBase_Elem {
  int arrayNum;
  // Kept across rounds: it carries the section reduction number
  CkSectionInfo cookie;

public:
  Elem(int a) : arrayNum(a) {}

  void work(WorkMsg *m) {
    if (!isMember(arrayNum, thisIndex))
      CkAbort("xarray_section_reduction: non-member %d of array %d got work",
              thisIndex, arrayNum);
    int r = m->round;
    CkGetSectionInfo(cookie, m);
    delete m;
    int v = valueOf(arrayNum, thisIndex, r);
    CkCallback maxCb(r % 2 == 0 ? CkIndex_Main::maxResultA(NULL)
                                : CkIndex_Main::maxResultB(NULL), mainProxy);
    CProxySection_Elem::contribute(sizeof(int), &v, CkReduction::max_int, cookie,
                                   maxCb, maxFlag(r));
    int pair[2] = {v, 1};
    CProxySection_Elem::contribute(sizeof(pair), pair, sumCountType, cookie,
                                   CkCallback(CkIndex_Main::sumCountResult(NULL), mainProxy),
                                   sumCountFlag(r));
  }
};

#include "xarray_section_reduction.def.h"
