#include "conv-autoconfig.h"

#if CMK_HAS_XPMEM
#include "cmixpmem.C"
#else
#include "cmishm.C"
#endif

#if CMK_SMP
#define CMI_DEST_RANK(msg) ((CmiMsgHeaderBasic*)msg)->rank
#if CMK_NODE_QUEUE_AVAILABLE
extern void CmiPushNode(void* msg);
#endif
#endif

CpvExtern(int, CthResumeNormalThreadIdx);

inline std::size_t whichBin_(std::size_t size);
inline static CmiIpcBlock* popBlock_(std::atomic<std::uintptr_t>& head,
                                     void* base);
inline static bool pushBlock_(std::atomic<std::uintptr_t>& head,
                              std::uintptr_t value, void* base);
static std::uintptr_t allocBlock_(ipc_shared_* meta, std::size_t size);

void* CmiIpcBlockToMsg(CmiIpcBlock* block, bool init) {
  auto* msg = (char*)CmiIpcBlockToMsg(block);
  if (init) {
    // NOTE ( this is identical to code in CmiAlloc )
    CmiAssert(((uintptr_t)msg % ALIGN_BYTES) == 0);
    CmiInitMsgHeader(msg, block->size);
    SIZEFIELD(msg) = block->size;
    REFFIELDSET(msg, 1);
  }
  return msg;
}

CmiIpcBlock* CmiMsgToIpcBlock(CmiIpcManager* manager, char* src, std::size_t len,
                           int node, int rank, int timeout) {
  char* dst;
  CmiIpcBlock* block;
  if ((block = CmiIsIpcBlock(manager, BLKSTART(src), node)) && (node == block->src)) {
    dst = src;
  } else {
    if (timeout > 0) {
      while (--timeout && !(block = CmiAllocIpcBlock(
                                manager, node, len + sizeof(CmiChunkHeader))))
        ;
    } else {
      // don't give up!
      while (
          !(block = CmiAllocIpcBlock(manager, node, len + sizeof(CmiChunkHeader))))
        ;
    }
    if (block == nullptr) {
      return nullptr;
    } else {
      CmiAssertMsg((block->dst == manager->mine) && (manager->mine == CmiMyNode()));
      dst = (char*)CmiIpcBlockToMsg(block, true);
      memcpy(dst, src, len);
      CmiFree(src);
    }
  }
#if CMK_SMP
  CMI_DEST_RANK(dst) = rank;
#endif
  return block;
}

extern void CmiHandleImmediateMessage(void *msg);

void CmiDeliverIpcBlockMsg(CmiIpcBlock* block) {
  auto* msg = CmiIpcBlockToMsg(block);
#if CMK_SMP
  auto& rank = CMI_DEST_RANK(msg);
#if CMK_NODE_QUEUE_AVAILABLE
  if (rank == cmi::ipc::nodeDatagram) {
    CmiPushNode(msg);
  } else
#endif
    CmiPushPE(rank, msg);
#else
#if CMK_IMMEDIATE_MSG
  if (CmiIsImmediate(msg)) {
    CmiHandleImmediateMessage(msg);
  } else
#endif
  {
    CmiHandleMessage(msg);
  }
#endif
}

inline static bool metadataReady_(CmiIpcManager* meta) {
  return meta && meta->shared[meta->mine];
}

CmiIpcBlock* CmiPopIpcBlock(CmiIpcManager* meta) {
  if (metadataReady_(meta)) {
    auto& shared = meta->shared[meta->mine];
    return popBlock_(shared->queue, shared);
  } else {
    return nullptr;
  }
}

bool CmiPushIpcBlock(CmiIpcManager* meta, CmiIpcBlock* block) {
  auto& shared = meta->shared[block->src];
  auto& queue = shared->queue;
  CmiAssert(meta->mine == block->dst);
  return pushBlock_(queue, block->orig, shared);
}

CmiIpcBlock* CmiAllocIpcBlock(CmiIpcManager* meta, int dstProc, std::size_t size) {
  auto dstNode = CmiPhysicalNodeID(CmiNodeFirst(dstProc));
#if CMK_SMP
  auto thisPe = CmiInCommThread() ? CmiNodeFirst(CmiMyNode()) : CmiMyPe();
#else
  auto thisPe = CmiMyPe();
#endif
  auto thisProc = CmiMyNode();
  auto thisNode = CmiPhysicalNodeID(thisPe);
  if ((thisProc == dstProc) || (thisNode != dstNode)) {
    throw std::bad_alloc();
  }

  auto& shared = meta->shared[dstProc];
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

  block->src = dstProc;
  block->dst = thisProc;

  return block;
}

void CmiFreeIpcBlock(CmiIpcManager* meta, CmiIpcBlock* block) {
  auto bin = whichBin_(block->size);
  CmiAssertMsg(bin < kNumCutOffPoints);
  auto& shared = meta->shared[block->src];
  auto& free = shared->free[bin];
  while (!pushBlock_(free, block->orig, shared))
    ;
}

CmiIpcBlock* CmiIsIpcBlock(CmiIpcManager* meta, void* addr, int node) {
  auto* shared = meta ? meta->shared[node] : nullptr;
  if (shared == nullptr) {
    return nullptr;
  }
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
    auto oom = next >= meta->max;
    auto value = oom ? res : (next + offset);
    auto status = meta->heap.exchange(value, std::memory_order_release);
    CmiAssert(status == cmi::ipc::nil);
    if (oom) {
      throw std::bad_alloc();
    } else {
      return res;
    }
  }
}

// NOTE ( there may be a faster way to do this? )
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
