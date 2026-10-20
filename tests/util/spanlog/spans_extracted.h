// extracted from ../../../src/arch/cuda/hybridAPI/hapi_impl.cpp by extract.sh -- do not edit
struct CuptiEntrySpan { uint64_t start, end; uint64_t token; uint32_t parent_delta, pad; };
struct CuptiEntryLane {
  std::vector<CuptiEntrySpan> spans;
  std::vector<uint32_t> open;   // indices of the open spans, innermost last (ascending)
  uint64_t pruned = 0;          // spans dropped from the front so far: absolute index = pruned + local
  uint32_t gen = 0;             // bumped by a compacting prune: the copies rebuild
};
static inline void hapiCuptiLaneOpen(CuptiEntryLane& ln, uint64_t token) {   // owner, under the log's mutex
  const uint64_t t = hapiCuptiSpanNow();
  const uint32_t idx = (uint32_t)ln.spans.size();
  const uint32_t pd = ln.open.empty() ? 0 : idx - ln.open.back();
  ln.spans.push_back({t, 0, token, pd, 0});
  ln.open.push_back(idx);
}
static inline void hapiCuptiLaneClose(CuptiEntryLane& ln) {   // owner, under the log's mutex
  if (ln.open.empty()) return;
  const uint32_t idx = ln.open.back();
  ln.open.pop_back();
  if (!ln.open.empty() && idx < ln.spans.size()) ln.spans[idx].end = hapiCuptiSpanNow();
}
static void hapiCuptiLanePrune(CuptiEntryLane& ln, uint64_t cutoff) {
  // Keep from the last span that started before the cutoff (it may still cover times after it).
  size_t k = std::lower_bound(ln.spans.begin(), ln.spans.end(), cutoff,
                              [](const CuptiEntrySpan& sp, uint64_t t) { return sp.start < t; }) - ln.spans.begin();
  if (k > 0) k--;
  // Keep that span's whole nesting context: a record in the gap after a nested span closed is held by
  // an ancestor, so the kept prefix starts at the outermost CLOSED ancestor (an open one is kept by the
  // compaction below; everything a kept span can point at is then kept, by the stack discipline).
  for (;;) {
    const CuptiEntrySpan& sp = ln.spans[k];
    if (sp.parent_delta == 0 || sp.parent_delta > k) break;
    const size_t p = k - sp.parent_delta;
    if (std::binary_search(ln.open.begin(), ln.open.end(), (uint32_t)p)) break;
    k = p;
  }
  if (k == 0) return;
  // Spans below k that are still open are kept, moved to the front in their order (the open stack is
  // ascending); everything else below k goes. Before this the prune stopped at the first open span,
  // so one entry left open for the run (a suspended thread, a lost pop) pinned the whole log and the
  // drains copied all of it every close: the parse of the same records grew 9 -> 38 ms over 1000
  // steps (job 22856533). Compaction renumbers, so the copies rebuild from scratch once (gen).
  size_t nOpenBelow = 0;
  while (nOpenBelow < ln.open.size() && ln.open[nOpenBelow] < k) nOpenBelow++;
  if (nOpenBelow == 0) {
    ln.spans.erase(ln.spans.begin(), ln.spans.begin() + k);
    for (uint32_t& i : ln.open) i -= (uint32_t)k;
    ln.pruned += k;
    return;
  }
  std::vector<CuptiEntrySpan> kept;
  kept.reserve(nOpenBelow + (ln.spans.size() - k));
  for (size_t j = 0; j < nOpenBelow; j++) {
    CuptiEntrySpan sp = ln.spans[ln.open[j]];
    sp.parent_delta = (j == 0) ? 0 : 1;   // an open span's parent is the open span beneath it
    kept.push_back(sp);
  }
  for (size_t i = k; i < ln.spans.size(); i++) {
    CuptiEntrySpan sp = ln.spans[i];
    const size_t newIdx = nOpenBelow + (i - k);
    if (sp.parent_delta != 0) {
      const size_t parentOld = i - sp.parent_delta;
      if (parentOld < k) {
        // A parent below k is one of the kept open spans (a closed parent closes after its children,
        // so a child kept above k with a closed parent below k cannot exist) -- or it was dropped.
        const size_t j = std::lower_bound(ln.open.begin(), ln.open.begin() + nOpenBelow, (uint32_t)parentOld) - ln.open.begin();
        sp.parent_delta = (j < nOpenBelow && ln.open[j] == (uint32_t)parentOld) ? (uint32_t)(newIdx - j) : 0;
      }
    }
    kept.push_back(sp);
  }
  for (size_t j = 0; j < ln.open.size(); j++)
    ln.open[j] = (j < nOpenBelow) ? (uint32_t)j : (uint32_t)(nOpenBelow + (ln.open[j] - k));
  ln.spans.swap(kept);
  ln.pruned += k - nOpenBelow;
  ln.gen++;
}
struct CuptiLaneCopy {
  std::vector<CuptiEntrySpan> spans;
  uint64_t base = 0;                   // absolute index of spans[0]
  uint32_t gen = 0;                    // the lane generation this copy follows
  std::vector<uint32_t> open_at_copy;  // local indices of the spans open at the last copy: the only ones that can change
};
static std::atomic<uint64_t> g_cupti_snap_bytes{0}, g_cupti_snap_ns{0};
static void hapiCuptiLaneRefresh(const CuptiEntryLane& ln, CuptiLaneCopy& cp) {
  const bool timeIt = hapiCuptiTimeIt();
  const uint64_t t0 = timeIt ? hapiClockNs(CLOCK_MONOTONIC) : 0;
  const uint64_t abs_begin = ln.pruned;
  size_t copied = 0;
  if (cp.gen != ln.gen || abs_begin < cp.base) {   // compacted (or a new lane): rebuild
    cp.spans.assign(ln.spans.begin(), ln.spans.end());
    cp.base = abs_begin;
    cp.gen = ln.gen;
    copied = cp.spans.size();
  } else {
    const uint64_t drop = abs_begin - cp.base;
    if (drop >= cp.spans.size()) cp.spans.clear();
    else if (drop > 0) cp.spans.erase(cp.spans.begin(), cp.spans.begin() + drop);
    cp.base = abs_begin;
    // Only a span that was open at the last copy can have changed since (its end stamp); closed
    // spans never change, so the rest of the copy stands and only the new spans are appended.
    for (uint32_t oi : cp.open_at_copy) {
      if (oi < drop) continue;
      const size_t li = oi - drop;
      if (li < cp.spans.size() && li < ln.spans.size()) cp.spans[li].end = ln.spans[li].end;
    }
    const size_t have = cp.spans.size();
    if (have < ln.spans.size()) {
      cp.spans.insert(cp.spans.end(), ln.spans.begin() + have, ln.spans.end());
      copied = ln.spans.size() - have;
    } else if (have > ln.spans.size()) {
      cp.spans.resize(ln.spans.size());
    }
  }
  cp.open_at_copy.assign(ln.open.begin(), ln.open.end());
  if (timeIt) {
    g_cupti_snap_bytes.fetch_add(copied * sizeof(CuptiEntrySpan), std::memory_order_relaxed);
    g_cupti_snap_ns.fetch_add(hapiClockNs(CLOCK_MONOTONIC) - t0, std::memory_order_relaxed);
  }
}
static const CuptiEntrySpan* hapiCuptiSpanAt(const std::vector<CuptiEntrySpan>& snap, uint64_t t, size_t& cursor) {
  if (snap.empty() || t < snap.front().start) return nullptr;
  size_t lo = cursor < snap.size() ? cursor : 0;
  if (snap[lo].start <= t) {
    int steps = 0;
    while (lo + 1 < snap.size() && snap[lo + 1].start <= t && ++steps < 64) lo++;
    if (lo + 1 < snap.size() && snap[lo + 1].start <= t) lo = 0;   // far ahead: search
  } else {
    lo = 0;
  }
  if (lo == 0) {
    size_t hi = snap.size();   // the last span with start <= t
    while (hi - lo > 1) {
      const size_t mid = (lo + hi) / 2;
      if (snap[mid].start <= t) lo = mid; else hi = mid;
    }
  }
  cursor = lo;
  for (;;) {
    const CuptiEntrySpan& sp = snap[lo];
    if (sp.end == 0 || sp.end >= t) return &sp;
    if (sp.parent_delta == 0 || sp.parent_delta > lo) return nullptr;   // closed before t: only an ancestor can hold it
    lo -= sp.parent_delta;
  }
}
