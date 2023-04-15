#ifndef CMI_SHMEM_HH
#define CMI_SHMEM_HH

static_assert(CMK_USE_SHMEM, "enable shmem to use this header");

#include <atomic>
#include <cstdint>
#include <limits>
#include <utility>

#define CMI_IPC_CUTOFF_ARG "ipccutoff"
#define CMI_IPC_CUTOFF_DESC "max message size for cmi-shmem (in bytes)"
#define CMI_IPC_POOL_SIZE_ARG "ipcpoolsize"
#define CMI_IPC_POOL_SIZE_DESC "size of cmi-shmem pool (in bytes)"

#if CMK_BUILD_CHARMRUN
#define CMI_IPC_CUTOFF_ENV_VAR "CmiIpcCutoff"
#define CMI_IPC_POOL_SIZE_ENV_VAR "CmiIpcPoolSize"
#endif

namespace cmi {
namespace ipc {
// recommended cutoff for block sizes
CpvExtern(std::size_t, kRecommendedCutoff);
// used to represent an empty linked list
constexpr auto nil = std::uintptr_t(0);
// used to represent the tail of a linked list
constexpr auto max = std::numeric_limits<std::uintptr_t>::max();
// used to indicate a message bound for a node
constexpr auto nodeDatagram = std::numeric_limits<CmiUInt2>::max();
// default number of attempts to alloc before timing out
constexpr auto defaultTimeout = 4;
}  // namespace ipc
}  // namespace cmi

// alignas is used for padding here, rather than for alignment of the
// CmiIpcBlock itself.
struct alignas(ALIGN_BYTES) CmiIpcBlock {
  // TODO ( find better names than src/dst? )
public:
  // "home" rank of the block
  int src;
  int dst;
  std::uintptr_t orig;
  std::uintptr_t next;
  std::size_t size;

  CmiIpcBlock(std::size_t size_, std::uintptr_t orig_)
      : orig(orig_), next(cmi::ipc::nil), size(size_) {}
};

struct CmiIpcManager;

enum CmiIpcAllocStatus {
  CMI_IPC_OUT_OF_MEMORY,
  CMI_IPC_REMOTE_DESTINATION,
  CMI_IPC_SUCCESS,
  CMI_IPC_TIMEOUT
};

// sets up ipc environment
void CmiIpcInit(char** argv);

// creates an ipc manager, waking the thread when it's done
// ( this must be called in the same order on all pes! )
CmiIpcManager* CmiMakeIpcManager(CthThread th);

// push/pop blocks from the manager's send/recv queue
bool CmiPushIpcBlock(CmiIpcManager*, CmiIpcBlock*);
CmiIpcBlock* CmiPopIpcBlock(CmiIpcManager*);

// tries to allocate a block, returning null if unsucessful
// (fails when other PEs are contending resources)
// second value of pair indicates failure cause
std::pair<CmiIpcBlock*, CmiIpcAllocStatus> CmiAllocIpcBlock(CmiIpcManager*, int node, std::size_t size);

// frees a block -- enabling it to be used again
void CmiFreeIpcBlock(CmiIpcManager*, CmiIpcBlock*);

// currently a no-op but may be eventually usable
// intended to "capture" blocks from remote pes
inline void CmiCacheIpcBlock(CmiIpcBlock*) { return; }

// identifies whether a void* is the payload of a block
// belonging to the given node
CmiIpcBlock* CmiIsIpcBlock(CmiIpcManager*, void*, int node);

// if (init) is true -- initializes the
// memory segment for use as a message
void* CmiIpcBlockToMsg(CmiIpcBlock*, bool init);

// equivalent to calling above with (init = false)
inline void* CmiIpcBlockToMsg(CmiIpcBlock* block) {
  auto res = (char*)block + sizeof(CmiIpcBlock) + sizeof(CmiChunkHeader);
  return (void*)res;
}

inline CmiIpcBlock* CmiMsgToIpcBlock(CmiIpcManager* manager, void* msg) {
  return CmiIsIpcBlock(manager, (char*)msg - sizeof(CmiChunkHeader), CmiMyNode());
}

CmiIpcBlock* CmiMsgToIpcBlock(CmiIpcManager*, char* msg, std::size_t len, int node,
                           int rank = cmi::ipc::nodeDatagram,
                           int timeout = cmi::ipc::defaultTimeout);

// deliver a block as a message
void CmiDeliverIpcBlockMsg(CmiIpcBlock*);

inline const std::size_t& CmiRecommendedIpcBlockCutoff(void) {
  using namespace cmi::ipc;
  return CpvAccess(kRecommendedCutoff);
}

#endif
