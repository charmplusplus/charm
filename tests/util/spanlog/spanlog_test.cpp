// Host test of the entry-span lane: prune compaction past open spans + incremental copies must give the same
// attribution as a lookup over the never-pruned log, for every time after the prune cutoff.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <atomic>
#include <random>
#include <time.h>
#include <cstring>
static uint64_t g_now = 1000;
static inline uint64_t hapiCuptiSpanNow() { return g_now += 1 + (g_now % 7); }
static inline uint64_t hapiClockNs(clockid_t) { return 0; }
static bool hapiCuptiTimeIt() { return false; }
#include "spans_extracted.h"
struct Ref { std::vector<CuptiEntrySpan> spans; std::vector<uint32_t> open; };   // the never-pruned mirror
int main(int argc, char** argv) {
  const bool bottomOpen = argc > 1 && atoi(argv[1]) != 0;   // a span opened at the start and never closed
  std::mt19937 rng(42);
  CuptiEntryLane ln; CuptiEntryLane ref; CuptiLaneCopy cp;
  uint64_t tok = 1;
  // hapiCuptiLaneOpen reads the clock once per call: run them back to back so both lanes see near-identical stamps;
  // to make the stamps IDENTICAL, drive the clock manually by pinning g_now before each open.
  auto pin = [&](uint64_t v) { g_now = v; };
  uint64_t clk = 1000;
  auto openPinned = [&](uint64_t t) { pin(clk); hapiCuptiLaneOpen(ln, t); pin(clk); hapiCuptiLaneOpen(ref, t); clk += 10; };
  auto closePinned = [&]() { pin(clk); hapiCuptiLaneClose(ln); pin(clk); hapiCuptiLaneClose(ref); clk += 10; };
  if (bottomOpen) openPinned(tok++);
  uint64_t arrivals[3] = {0, 0, 0}; int narr = 0;
  size_t maxLane = 0, maxCopied = 0; uint64_t gens = 0; size_t checks = 0, mismatches = 0;
  const int steps = 400, period = 25, perStep = 40;
  for (int st = 1; st <= steps; st++) {
    for (int e = 0; e < perStep; e++) {
      const int depth = 1 + (int)(rng() % 3);
      for (int d = 0; d < depth; d++) openPinned(tok++);
      for (int d = 0; d < depth; d++) closePinned();
    }
    if (st % period == 0) {   // arrival: prune to two arrivals back (as hapiCuptiArrive), then a drain copies
      const uint64_t now = clk;
      const uint64_t cutoff = narr >= 2 ? arrivals[(narr - 2) % 3] : 0;
      arrivals[narr % 3] = now; narr++;
      const uint32_t genBefore = ln.gen;
      if (cutoff) hapiCuptiLanePrune(ln, cutoff);
      if (ln.gen != genBefore) gens++;
      const size_t before = cp.spans.size();
      hapiCuptiLaneRefresh(ln, cp);
      maxCopied = std::max(maxCopied, cp.spans.size() > before ? cp.spans.size() - before : (size_t)0);
      maxLane = std::max(maxLane, ln.spans.size());
      if (cp.spans.size() != ln.spans.size() || memcmp(cp.spans.data(), ln.spans.data(), ln.spans.size() * sizeof(CuptiEntrySpan)) != 0) { printf("COPY != LANE at arrival %d (%zu vs %zu)\n", narr, cp.spans.size(), ln.spans.size()); mismatches++; }
      // every time from the cutoff to now: the copy must attribute like the full log
      std::vector<CuptiEntrySpan> full(ref.spans);
      const uint64_t lo = cutoff ? cutoff : 1000;
      for (uint64_t t = lo; t <= now; t += 1 + (rng() % 3)) {
        size_t c1 = 0, c2 = 0;
        const CuptiEntrySpan* a = hapiCuptiSpanAt(full, t, c1);
        const CuptiEntrySpan* b = hapiCuptiSpanAt(cp.spans, t, c2);
        const uint64_t ta = a ? a->token : 0, tb = b ? b->token : 0;
        checks++;
        if (ta != tb) { if (mismatches < 5) printf("MISMATCH t=%llu full=%llu copy=%llu (arrival %d)\n", (unsigned long long)t, (unsigned long long)ta, (unsigned long long)tb, narr); mismatches++; }
      }
      // second refresh without changes must copy nothing and change nothing
      const size_t sz = cp.spans.size(); hapiCuptiLaneRefresh(ln, cp);
      if (cp.spans.size() != sz) { printf("REFRESH CHANGED SIZE %zu -> %zu\n", sz, cp.spans.size()); mismatches++; }
    }
  }
  printf("bottomOpen=%d steps=%d spans/step~%d: lane max %zu spans (ref %zu), copy max append %zu, compactions %llu, checks %zu mismatches %zu -> %s\n",
         (int)bottomOpen, steps, perStep * 2, maxLane, ref.spans.size(), maxCopied, (unsigned long long)gens, checks, mismatches, mismatches ? "FAIL" : "PASS");
  return mismatches ? 1 : 0;
}
