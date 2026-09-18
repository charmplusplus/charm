// Host-only test of the device pool's range heap (hapi_range_heap.h): no CUDA,
// no Charm++, runs anywhere. Exercises best fit, coalescing, in-place growth by
// whole chunks, the no-grow contract, and the sph2d strong shape that broke the
// buddy pool: many 10.8 MiB buffers with 8 MiB landings interleaved.
#include "../../../src/arch/cuda/hybridAPI/hapi_range_heap.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using hapi_pool::RangeHeap;
static const size_t MiB = 1u << 20;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static RangeHeap makeHeap(size_t reserve, size_t chunk, size_t* mappedCalls = nullptr,
                          size_t refuseAfterBytes = (size_t)-1) {
  RangeHeap h(0x100000000ull, reserve, chunk);
  size_t* calls = mappedCalls;
  h.mapper = [calls, refuseAfterBytes, &h](uintptr_t addr, size_t bytes) {
    if (calls) (*calls)++;
    if (h.mappedBytes() + bytes > refuseAfterBytes) return false;
    return addr == h.mappedEnd() && bytes % h.chunkBytes() == 0;
  };
  return h;
}

static void testBasics() {
  size_t calls = 0;
  RangeHeap h = makeHeap(64 * MiB, 8 * MiB, &calls);
  CHECK(h.malloc(100) == nullptr);           // nothing mapped: no growth by itself
  CHECK(h.grow(100));
  CHECK(h.mappedBytes() == 8 * MiB && calls == 1);
  void* a = h.malloc(100);
  CHECK(a != nullptr);
  size_t sz = 0;
  CHECK(h.blockSize(a, &sz) && sz == 256);   // rounded to the alignment
  CHECK(h.freeBytes() == 8 * MiB - 256);
  void* b = h.malloc(3 * MiB);
  void* c = h.malloc(3 * MiB);
  CHECK(b && c && h.malloc(3 * MiB) == nullptr);   // 2 MiB - 256 left: no fit
  CHECK(h.largestFree() == 2 * MiB - 256);
  h.free(b);
  CHECK(h.largestFree() == 3 * MiB);        // b's range, not merged with the tail
  h.free(c);
  CHECK(h.largestFree() == 8 * MiB - 256);  // b + c + tail coalesced
  CHECK(h.freeRanges() == 1);
  h.free(a);
  CHECK(h.freeRanges() == 1 && h.freeBytes() == 8 * MiB && h.liveAllocations() == 0);
}

static void testGrowInPlace() {
  size_t calls = 0;
  RangeHeap h = makeHeap(64 * MiB, 8 * MiB, &calls);
  CHECK(h.grow(1));
  void* a = h.malloc(6 * MiB);
  CHECK(a);
  CHECK(h.malloc(10 * MiB) == nullptr);      // 2 MiB tail
  CHECK(h.grow(10 * MiB));                   // needs 8 more: exactly one chunk
  CHECK(h.mappedBytes() == 16 * MiB && calls == 2);
  void* b = h.malloc(10 * MiB);              // served from tail + new chunk, contiguous
  CHECK(b == (char*)a + 6 * MiB);
  CHECK(h.freeBytes() == 0);
  CHECK(h.grow(64 * MiB) == false);          // past the reserve: refused, unchanged
  CHECK(h.mappedBytes() == 16 * MiB);
  h.free(a); h.free(b);
  CHECK(h.freeRanges() == 1 && h.freeBytes() == 16 * MiB);
}

static void testMapperRefusal() {
  size_t calls = 0;
  RangeHeap h = makeHeap(1024 * MiB, 8 * MiB, &calls, 16 * MiB);
  CHECK(h.grow(8 * MiB) && h.grow(16 * MiB));
  CHECK(h.mappedBytes() == 16 * MiB);
  CHECK(h.grow(17 * MiB) == false);          // the mapper (registration cap) said no
  CHECK(h.mappedBytes() == 16 * MiB && h.freeBytes() == 16 * MiB);
}

// The shape that falsified the byte-reserving gate on the buddy pool: patches
// of 2 x 10.8 MiB + 3 x 8.65 MiB + 17.3 MiB + 4 x 1.35 MiB with 8 MiB landings
// interleaved, then churn. Under buddy each patch held 121 MiB for 71 MiB.
static void testSph2dShape() {
  const size_t pcap = 11337856, ecap = 9070080, small = 1417232;
  size_t calls = 0;
  RangeHeap h = makeHeap(48ull * 1024 * MiB, 256 * MiB, &calls);
  struct Patch { std::vector<void*> b; };
  auto allocPatch = [&](Patch& p) {
    const size_t sizes[] = {pcap, pcap, ecap, ecap, ecap, 2 * ecap, small, small, small, small};
    for (size_t s : sizes) {
      void* q = h.malloc(s);
      if (!q) { CHECK(h.grow(s)); q = h.malloc(s); }
      CHECK(q != nullptr);
      p.b.push_back(q);
    }
  };
  auto freePatch = [&](Patch& p) { for (void* q : p.b) h.free(q); p.b.clear(); };
  std::vector<Patch> patches(272);
  for (auto& p : patches) allocPatch(p);
  const size_t perPatch = 2 * RangeHeap::roundUp(pcap, 256) + 3 * RangeHeap::roundUp(ecap, 256) +
                          RangeHeap::roundUp(2 * ecap, 256) + 4 * RangeHeap::roundUp(small, 256);
  const size_t held = h.mappedBytes() - h.freeBytes();
  CHECK(held == 272 * perPatch);
  printf("  sph2d shape: 272 patches hold %.1f GiB (buddy held 32.3 GiB); mapped %.1f GiB in %zu chunks\n",
         held / (1024.0 * MiB), h.mappedBytes() / (1024.0 * MiB), h.nChunks());
  CHECK(held < 20ull * 1024 * MiB);
  // Migration churn: depart 40 patches, land 40 with an 8 MiB landing each,
  // interleaved in the worst order (landing taken before the departure's
  // state is freed), and make sure the heap did not need a single new chunk
  // beyond one for the landings themselves.
  const size_t chunksBefore = h.nChunks();
  std::vector<void*> landings;
  for (int i = 0; i < 40; i++) {
    void* l = h.malloc(8 * MiB);
    if (!l) { CHECK(h.grow(8 * MiB)); l = h.malloc(8 * MiB); }
    landings.push_back(l);
    freePatch(patches[i * 6]);
    allocPatch(patches[i * 6]);
  }
  for (void* l : landings) h.free(l);
  CHECK(h.nChunks() <= chunksBefore + 2);
  printf("  churn: %zu -> %zu chunks, %zu free ranges, largest free %.1f MiB\n",
         chunksBefore, h.nChunks(), h.freeRanges(), h.largestFree() / (double)MiB);
  for (auto& p : patches) freePatch(p);
  CHECK(h.liveAllocations() == 0 && h.freeRanges() == 1 && h.freeBytes() == h.mappedBytes());
}

static void testRandomChurn() {
  srand(7);
  RangeHeap h = makeHeap(4096 * MiB, 64 * MiB);
  std::vector<std::pair<void*, size_t>> live;
  size_t liveBytes = 0;
  for (int i = 0; i < 20000; i++) {
    if (live.empty() || (rand() % 3 != 0)) {
      const size_t s = 1 + (rand() % (4 * MiB));
      void* p = h.malloc(s);
      if (!p) { if (!h.grow(s)) continue; p = h.malloc(s); }
      CHECK(p != nullptr);
      size_t blk = 0;
      CHECK(h.blockSize(p, &blk) && blk >= s && blk < s + 256);
      // no overlap with any live block
      for (const auto& l : live)
        CHECK((char*)p + blk <= (char*)l.first || (char*)l.first + l.second <= (char*)p);
      live.emplace_back(p, blk);
      liveBytes += blk;
    } else {
      const size_t k = rand() % live.size();
      h.free(live[k].first);
      liveBytes -= live[k].second;
      live[k] = live.back();
      live.pop_back();
    }
    CHECK(h.mappedBytes() - h.freeBytes() == liveBytes);
  }
  for (auto& l : live) h.free(l.first);
  CHECK(h.freeRanges() == 1 && h.freeBytes() == h.mappedBytes());
  printf("  random churn: mapped %.0f MiB, one free range at the end\n", h.mappedBytes() / (double)MiB);
}

int main() {
  testBasics();
  testGrowInPlace();
  testMapperRefusal();
  testSph2dShape();
  testRandomChurn();
  if (fails) { printf("rangeheap_test: %d FAILURE(S)\n", fails); return 1; }
  printf("rangeheap_test: all passed\n");
  return 0;
}
