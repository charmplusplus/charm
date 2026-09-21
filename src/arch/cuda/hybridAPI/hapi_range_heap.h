#ifndef HAPI_RANGE_HEAP_H
#define HAPI_RANGE_HEAP_H

// A best-fit range allocator over one contiguous address range that is backed
// lazily, in chunks, from its start: the allocation half of the device pool's
// virtual-memory backend (+gpupoolalloc vmm). It knows nothing about CUDA; the
// owner supplies `mapper`, which makes [addr, addr+bytes) usable (cuMemCreate +
// cuMemMap on the device, a no-op in a test), and the heap only ever asks it
// for whole chunks appended at the mapped end. Growth is therefore in place:
// the new chunks join the free range that ended at the old mapped end, so a
// request that no free range fits is served by mapping only what is missing,
// never by opening a disjoint arena that nothing can coalesce with.
//
// Sizes round up to `align` bytes (256 by default, what CUDA wants), so the
// internal fragmentation the buddy allocator paid for -- a 10.8 MiB buffer
// holding a 16 MiB block, 40% of sph2d's pool -- is gone. External
// fragmentation remains possible, as it is for any allocator over a fixed
// range; the admission gate handles it by allocating to reserve (cklocation.C)
// and largestFree() says what one request could get right now.
//
// Not thread-safe: the pool holds its own mutex around every call.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <unordered_map>

namespace hapi_pool {

class RangeHeap {
public:
  // [base, base + reserve) is the address range; `chunk` is the mapping
  // granule. Nothing is mapped until grow() is called.
  RangeHeap(uintptr_t base, size_t reserve, size_t chunk, size_t align = 256)
      : base_(base), reserve_(reserve), chunk_(chunk), align_(align),
        mappedEnd_(base) {}

  // Makes [addr, addr + bytes) usable; bytes is a multiple of chunk and addr
  // is the current mapped end. Returns false to refuse (the heap then does
  // not grow).
  std::function<bool(uintptr_t addr, size_t bytes)> mapper;

  // Best fit among the mapped free ranges, and never across a chunk boundary
  // for a block that fits in one chunk: each chunk is its own physical
  // allocation (cuMemCreate), and the fabric accepts a DMA registration that
  // spans two of them but fails the first transfer on it (lci poll_comp Err 5;
  // job 22217038: every 512 MB two-chunk run died, 166 single-chunk runs ran
  // clean). A block larger than a chunk necessarily spans and takes the start
  // of its range. nullptr when nothing fits. Never grows: the pool decides
  // whether growth is allowed (hapiDevPoolMallocNoGrow).
  void* malloc(size_t size) {
    size = roundUp(size ? size : 1, align_);
    for (auto it = bySize_.lower_bound(size); it != bySize_.end(); ++it) {
      const size_t sz = it->first;
      const uintptr_t addr = it->second;
      uintptr_t at;
      if (!placeIn(addr, sz, size, &at)) continue;
      eraseRange(addr, sz);
      if (at > addr) insertRange(addr, at - addr);
      if (addr + sz > at + size) insertRange(at + size, addr + sz - (at + size));
      allocs_[at] = size;
      return (void*)at;
    }
    return nullptr;
  }

  // Where a block of `size` bytes goes in the free range [addr, addr+sz): at
  // its start when it stays inside one chunk (or is bigger than a chunk), else
  // at the next chunk boundary when it still fits there.
  bool placeIn(uintptr_t addr, size_t sz, size_t size, uintptr_t* at) const {
    if (sz < size) return false;
    if (size > chunk_ || chunkOf(addr) == chunkOf(addr + size - 1)) { *at = addr; return true; }
    const uintptr_t next = base_ + (chunkOf(addr) + 1) * chunk_;
    if (next + size <= addr + sz) { *at = next; return true; }
    return false;
  }
  size_t chunkOf(uintptr_t a) const { return (a - base_) / chunk_; }

  // Map enough more of the reserve that a `need`-byte request can be served
  // from the range at the mapped end. Returns false when the reserve is
  // exhausted or the mapper refused; the heap is unchanged then. (A tail of
  // at least `need` bytes always has a placement that respects chunk
  // boundaries: it ends on one, so either it holds a whole chunk or it lies
  // inside one.)
  bool grow(size_t need) {
    need = roundUp(need ? need : 1, align_);
    size_t tail = 0;
    if (!byAddr_.empty()) {
      auto last = std::prev(byAddr_.end());
      if (last->first + last->second == mappedEnd_) tail = last->second;
    }
    if (tail >= need) return true;   // already fits at the end
    const size_t bytes = roundUp(need - tail, chunk_);
    if (mappedEnd_ + bytes > base_ + reserve_) return false;
    if (!mapper || !mapper(mappedEnd_, bytes)) return false;
    insertRange(mappedEnd_, bytes);   // coalesces with the tail range
    mappedEnd_ += bytes;
    return true;
  }

  // Returns the range to the free list, merging with its neighbours. A
  // pointer this heap did not hand out is ignored.
  void free(void* p) {
    auto it = allocs_.find((uintptr_t)p);
    if (it == allocs_.end()) return;
    const size_t size = it->second;
    allocs_.erase(it);
    insertRange((uintptr_t)p, size);
  }

  bool blockSize(const void* p, size_t* size) const {
    auto it = allocs_.find((uintptr_t)p);
    if (it == allocs_.end()) return false;
    *size = it->second;
    return true;
  }

  bool contains(const void* p) const {
    const uintptr_t q = (uintptr_t)p;
    return q >= base_ && q < mappedEnd_;
  }

  size_t freeBytes() const {
    size_t f = 0;
    for (const auto& r : byAddr_) f += r.second;
    return f;
  }
  size_t largestFree() const { return bySize_.empty() ? 0 : bySize_.rbegin()->first; }
  size_t mappedBytes() const { return mappedEnd_ - base_; }
  size_t nChunks() const { return (mappedEnd_ - base_) / chunk_; }
  size_t chunkBytes() const { return chunk_; }
  size_t reserveBytes() const { return reserve_; }
  size_t alignBytes() const { return align_; }
  uintptr_t base() const { return base_; }
  uintptr_t mappedEnd() const { return mappedEnd_; }
  size_t liveAllocations() const { return allocs_.size(); }
  size_t freeRanges() const { return byAddr_.size(); }

  static size_t roundUp(size_t v, size_t to) { return (v + to - 1) / to * to; }

private:
  void eraseRange(uintptr_t addr, size_t size) {
    byAddr_.erase(addr);
    auto r = bySize_.equal_range(size);
    for (auto it = r.first; it != r.second; ++it)
      if (it->second == addr) { bySize_.erase(it); break; }
  }

  void insertRange(uintptr_t addr, size_t size) {
    // Merge with the range ending at addr, and the range starting at addr+size.
    auto next = byAddr_.lower_bound(addr);
    if (next != byAddr_.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == addr) {
        addr = prev->first;
        size += prev->second;
        eraseRange(prev->first, prev->second);
        next = byAddr_.lower_bound(addr);
      }
    }
    if (next != byAddr_.end() && next->first == addr + size) {
      const size_t nsz = next->second;
      eraseRange(next->first, nsz);
      size += nsz;
    }
    byAddr_[addr] = size;
    bySize_.emplace(size, addr);
  }

  uintptr_t base_;
  size_t reserve_, chunk_, align_;
  uintptr_t mappedEnd_;
  std::map<uintptr_t, size_t> byAddr_;          // free ranges by address
  std::multimap<size_t, uintptr_t> bySize_;     // the same, by size (best fit)
  std::unordered_map<uintptr_t, size_t> allocs_;  // live blocks
};

}  // namespace hapi_pool

#endif  // HAPI_RANGE_HEAP_H
