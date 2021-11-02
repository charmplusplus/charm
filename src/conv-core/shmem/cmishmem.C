#if CMI_HAS_XPMEM
#include "cmixpmem.cc"
#else
#include "cmishm.C"
#endif

CpvExtern(int, CthResumeNormalThreadIdx);

inline static CmiIpcBlock* popBlock_(std::atomic<std::uintptr_t>& head,
                                     void* base);
inline static bool pushBlock_(std::atomic<std::uintptr_t>& head,
                              std::uintptr_t value, void* base);
static std::uintptr_t allocBlock_(ipc_shared_* meta, std::size_t size);

void* CmiBlockToMsg(CmiIpcBlock* block, bool init) {
  auto* msg = CmiBlockToMsg(block);
  if (init) {
    // NOTE ( this is identical to code in CmiAlloc )
    CmiAssert(((uintptr_t)msg % ALIGN_BYTES) == 0);
    CmiInitMsgHeader(msg, block->size);
    SIZEFIELD(msg) = block->size;
    REFFIELDSET(msg, 1);
  }
  return msg;
}

static void CmiHandleBlock_(void*, double) {
  auto* block = CmiPopBlock();
  if (block != nullptr) {
    auto* msg = CmiBlockToMsg(block);
#if CMK_SMP
    auto dst = block->dst % CmiMyNodeSize();
    CmiPushPE(dst, msg);
#else
    CmiHandleMessage(msg);
#endif
  }
}

inline static bool metadataReady_(ipc_metadata_ptr_& meta) {
  return meta && meta->shared[meta->mine];
}

CmiIpcBlock* CmiPopBlock(void) {
  auto& meta = CsvAccess(metadata_);
  if (metadataReady_(meta)) {
    auto& shared = meta->shared[meta->mine];
    return popBlock_(shared->queue, shared);
  } else {
    return nullptr;
  }
}

bool CmiPushBlock(CmiIpcBlock* block) {
  auto& meta = CsvAccess(metadata_);
  auto& shared = meta->shared[block->src / CmiMyNodeSize()];
  auto& queue = shared->queue;
#if CMK_SMP
  auto thisPe = CmiInCommThread() ? CmiNodeFirst(CmiMyNode()) : CmiMyPe();
  auto thisNode = CmiPhysicalRank(thisPe);
  CmiAssert(thisNode == block->dst);
#else
  CmiAssert(meta->mine == block->dst);
#endif
  return pushBlock_(queue, block->orig, shared);
}

CmiIpcBlock* CmiAllocBlock(int pe, std::size_t size) {
#if CMK_SMP
  auto myPe = CmiInCommThread() ? CmiNodeFirst(CmiMyNode()) : CmiMyPe();
#else
  auto myPe = CmiMyPe();
#endif
  auto myRank = CmiPhysicalRank(myPe);
  auto myNode = CmiPhysicalNodeID(myPe);
  auto theirRank = CmiPhysicalRank(pe);
  auto theirNode = CmiPhysicalNodeID(pe);
  CmiAssert((myRank != theirRank) && (myNode == theirNode));
  auto& meta = CsvAccess(metadata_);
  auto& shared = meta->shared[theirRank / CmiMyNodeSize()];
  auto bin = whichBin_(size);
  CmiAssert(bin < kNumCutOffPoints);

  auto* block = popBlock_(shared->free[bin], shared);
  if (block == nullptr) {
    auto totalSize = kCutOffPoints[bin];
    auto offset = allocBlock_(shared, totalSize);
    if (offset == cmi::ipc::nil) {
      return nullptr;
    }
    // the block's address is relative to the share
    block = (CmiIpcBlock*)((char*)shared + offset);
    CmiAssert(((std::uintptr_t)block % alignof(CmiIpcBlock)) == 0);
    // construct the block
    new (block) CmiIpcBlock(totalSize, offset);
  }

  block->src = theirRank;
  block->dst = myRank;

  return block;
}

void CmiFreeBlock(CmiIpcBlock* block) {
  auto& meta = CsvAccess(metadata_);
  auto bin = whichBin_(block->size);
  CmiAssertMsg(bin < kNumCutOffPoints);
  auto& shared = meta->shared[block->src / CmiMyNodeSize()];
  auto& free = shared->free[bin];
  while (!pushBlock_(free, block->orig, shared))
    ;
}

CmiIpcBlock* CmiIsBlock(void* addr) {
  auto& meta = CsvAccess(metadata_);
  if (!metadataReady_(meta)) {
    return nullptr;
  }
  auto& shared = meta->shared[meta->mine];
  auto* begin = (char*)shared;
  auto* end = begin + shared->max;
  if (begin < addr && addr < end) {
    return (CmiIpcBlock*)((char*)addr - sizeof(CmiIpcBlock));
  } else {
    return nullptr;
  }
}

static std::uintptr_t allocBlock_(ipc_shared_* meta, std::size_t size) {
  auto res = meta->heap.exchange(cmi::ipc::nil, std::memory_order_acquire);
  if (res == cmi::ipc::nil) {
    return cmi::ipc::nil;
  } else {
    auto next = res + size + sizeof(CmiIpcBlock);
    auto offset = size % alignof(CmiIpcBlock);
    auto status = meta->heap.exchange(next + offset, std::memory_order_release);
    CmiAssert(status == cmi::ipc::nil);
    if (next < meta->max) {
      return res;
    } else {
      return cmi::ipc::nil;
    }
  }
}

// TODO ( find a better way to do this )
inline std::size_t whichBin_(std::size_t size) {
  std::size_t bin;
  for (bin = 0; bin < kNumCutOffPoints; bin++) {
    if (size <= kCutOffPoints[bin]) {
      break;
    }
  }
  return bin;
}

inline static CmiIpcBlock* popBlock_(std::atomic<std::uintptr_t>& head,
                                     void* base) {
  auto prev = head.exchange(cmi::ipc::nil, std::memory_order_acquire);
  if (prev == cmi::ipc::nil) {
    return nullptr;
  } else if (prev == cmi::ipc::max) {
    auto check = head.exchange(prev, std::memory_order_release);
    CmiAssert(check == cmi::ipc::nil);
    return nullptr;
  } else {
    // translate the "home" PE's address into a local one
    CmiAssert(((std::uintptr_t)base % ALIGN_BYTES) == 0);
    auto* xlatd = (CmiIpcBlock*)((char*)base + prev);
    auto check = head.exchange(xlatd->next, std::memory_order_release);
    CmiAssert(check == cmi::ipc::nil);
    return xlatd;
  }
}

inline static bool pushBlock_(std::atomic<std::uintptr_t>& head,
                              std::uintptr_t value, void* base) {
  CmiAssert(value != cmi::ipc::nil);
  auto prev = head.exchange(cmi::ipc::nil, std::memory_order_acquire);
  if (prev == cmi::ipc::nil) {
    return false;
  }
  auto* block = (CmiIpcBlock*)((char*)base + value);
  block->next = prev;
  auto check = head.exchange(value, std::memory_order_release);
  CmiAssert(check == cmi::ipc::nil);
  return true;
}

static void awakenSleepers_(void) {
  auto& sleepers = CsvAccess(sleepers);
  for (auto i = 0; i < sleepers.size(); i++) {
    auto& th = sleepers[i];
    if (i == CmiMyRank()) {
      CthAwaken(th);
    } else {
      auto* token = CthGetToken(th);
      CmiSetHandler(token, CpvAccess(CthResumeNormalThreadIdx));
      CmiPushPE(i, token);
    }
  }
}
