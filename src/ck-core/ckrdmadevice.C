/*
 * Direct GPU Messaging
 *
 * Uses host-bypass mechanisms to directly transfer data between GPU devices.
 *
 * 1) Intra-process (intra-node): The sender sends a metadata message to the
 *    receiver containing the pointer to the source GPU buffer. There is
 *    no setup needed on the sender side. The receiver invokes device-to-device
 *    transfer from the source GPU buffer to the destination GPU buffer.
 *
 * 2) Inter-process (intra-node): The pointer to the source GPU buffer will be
 *    invalid on the receiver as it is a different process. Thus CUDA IPC is
 *    used to create a handle to the source GPU buffer, which can be opened on
 *    the receiver side to initiate the data transfer. To mitigate the overheads
 *    of creating and destroying IPC handles, the runtime first allocates a
 *    'device communication buffer' on each GPU device, creates IPC handles
 *    only for these buffers, and then exchanges the handles between processes
 *    on the same physical node. This means each process will have IPC handles
 *    for all device communication buffers on the same host (that are potentially
 *    managed by other processes) and can perform data transfers using these
 *    handles. Each GPU-GPU data transfer invovles requesting a block from the
 *    device communication buffer on the sender, copying the source GPU buffer
 *    to the allocated block, sending a metadata message to the receiver
 *    (that contains the offset of the allocated block), and performing a
 *    transfer from the block on the sender's device communication buffer to
 *    the destination GPU buffer. CUDA Events are used to enforce the correct
 *    ordering between these data transfers. Because multiple PEs can be mapped
 *    to the same GPU and hence concurrently request allocations from the same
 *    device communication buffer, a thread-safe allocator using the buddy
 *    allocation algorithm was implemented. The allocator first calls hapiMalloc
 *    to obtain a relatively large chunk of memory and then services allocation
 *    and deallocation requests from PEs that are mapped to its GPU device.
 *    The buddy algorithm was used to minimize the external fragmentation that
 *    could occur from concurrent manipulations of the device communication
 *    buffer.
 *
 * TODO
 * 3) Inter-node: This currently uses a simple host-staged mechanism to perform
 *    a device-to-host copy of the source GPU buffer to a message, which is sent
 *    to the receiver. The receiver then performs a host-to-device copy to the
 *    destination GPU buffer. This will be updated to use GPUDirect RDMA to
 *    directly performa true device-to-device transfer.
 */

#ifndef _WIN32
#include <pthread.h>
#endif
#include "envelope.h"
#include "charm++.h"
#include "ck.h"
#include "ckrdmadevice.h"
#include <deque>
#include "buddy_allocator.h"
#include <mutex>
#include <map>

#define CMK_GPU_COMM 1

#if CMK_CUDA || CMK_HIP

CmiNcpyModeDevice findTransferModeDevice(int srcPe, int dstPe) {
  CmiEnforce((srcPe >= 0) && (srcPe <= CmiNumPes()));
  CmiEnforce((dstPe >= 0) && (dstPe <= CmiNumPes()));

  if (CmiNodeOf(srcPe) == CmiNodeOf(dstPe)) {
    // Same logical node
    return CmiNcpyModeDevice::MEMCPY;
  } else if (CmiPeOnSamePhysicalNode(srcPe, dstPe)) {
    // Different logical nodes, same physical node
    return CmiNcpyModeDevice::IPC;
  } else {
    // Different physical nodes, requires GPUDirect RDMA
    return CmiNcpyModeDevice::RDMA;
  }
}

#include <atomic>
#include <stdio.h>
#include <unordered_map>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "hapi.h"

// A stream synchronize that also covers work this PE has handed to a submitter
// thread (+gpusubmit): it has to be in the driver before the wait means anything.
static inline hapiError_t ckSubmitDrainThenSync(hapiStream_t stream) {
  hapiSubmitDrain();
  return cudaStreamSynchronize(stream);
}
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);
CpvExtern(int, my_device_id);
static inline hapi_ipc_event_shared* ipcSharedSlot(int device_idx, int event_idx);

// Defined further down, used by both receive completion handlers above it.
void zcRecordRecvTime(int slot, size_t bytes, double seconds);

// void CkRdmaDeviceRecvHandler(void* data)
// {
//   DeviceRdmaOp* op = (DeviceRdmaOp*)data;
//   DeviceRdmaInfo* info = op->info;

//   // Invoke source callbacks
//   if (op->src_cb) {
//     int rank;
//     CkCallback* cb = (CkCallback*)op->src_cb;
//     cb->send();
//     delete cb;
//   }

//   // Update counter (there may be multiple buffers in transit)
//   info->counter++;

//   // Check if all buffers have been received
//   // If so, invoke regular entry method
//   if (info->counter == info->n_ops) {
//     QdCreate(1);

//     enqueueNcpyMessage(op->dest_pe, info->msg);

//     // Free RDMA metadata
//     CmiFree(info);
//   }
// }

// ---- Device registration: pool arenas once, everything else cached --------
//
// Under +gpupool every device buffer the runtime sends or lands lives inside a
// pool arena: one cudaMalloc of CK_GPU_ARENA_MB that is never freed, whatever
// the pool carves out of it and hands back. A network registration of the
// whole arena is therefore valid for the life of the process, and one is all
// an arena ever gets: the first send or receive that touches an arena
// registers [base, base+extent) once, process-wide, and every descriptor for
// a buffer inside it is a copy of that registration with the buffer's own
// pointer and count. The LCI get addresses the remote buffer as a displacement
// from the registered region's base, so a sub-range needs nothing more. Nothing
// is ever deregistered, and nothing needs to be: the arena outlives every
// transfer. That is the whole point of having a pool on the network side --
// without it every inter-node device send was a real fi_mr_reg that nothing
// released, and at 64 PEs the NIC's registration table filled in ~50 steps.
//
// Buffers that are not the pool's (a run without +gpupool, or an application
// allocation outside it) keep the per-buffer path below.
//
// ---- Device registration cache (CHARM_DEVICE_MR_CACHE) --------------------
//
// The device path builds a fresh CmiNcpyBuffer for every send, and constructing
// one registers the buffer with the network. The application reuses the same
// device buffers every iteration, so nearly all of that is re-registration of
// memory that is already registered: 5318 registration calls resolving to 710
// distinct regions in one 60-iteration run. Nothing is ever deregistered
// either, which is why disabling the provider's own MR cache costs 3.3x -- with
// nothing absorbing the repeats, every one becomes a real fi_mr_reg.
//
// Keep the registration instead, keyed by the range it covers, and hand out
// copies that differ only in the per-transfer ref field.
//
// The hard part of any registration cache is knowing when the memory died, and
// this one claims to know exactly one case: an element's device buffers are
// freed when it migrates, and migration is mediated by the runtime, so entries
// are attributed to the element that registered them and dropped when it
// leaves. An application that frees and reallocates a device buffer in place is
// invisible here for the same reason it is invisible to the provider's cache --
// CUDA suballocates, so a free usually raises no address-space event at all.
// That is the gap, and it is why this is off unless CHARM_DEVICE_MR_CACHE is
// set.
//
// Both directions are cached. The send path attributes entries to
// _currentLocRec, the element issuing the send. The receive path runs before
// the entry method, so _currentLocRec is not yet set -- but the envelope names
// the element the receive is addressed to, and CkFindDeviceRecvElement
// resolves it to the same CkLocRec the migration paths retire. A receive with
// no element attached (a migration payload on a group entry) is not cached:
// its buffer is freshly allocated per transfer and freed after, exactly the
// staleness an unowned entry would hide.

struct DeviceMrEntry {
  CmiNcpyBuffer reg;   // registered once; copies differ only in ref and cnt
  CkLocRec* owner;     // whose migration retires this entry
};

// Keyed by pointer alone, not (ptr, cnt): the particle-exchange pattern sends
// from a fixed buffer with a different count every iteration, and a
// count-qualified key turns that into one fresh registration per distinct
// count -- the same unbounded growth this cache exists to stop. A registration
// covers [ptr, ptr+cnt); a request for fewer bytes reuses it with the copy's
// cnt shrunk to the request, and a request for more re-registers the larger
// extent in place.
typedef std::unordered_map<const void*, DeviceMrEntry> DeviceMrCache;

CkpvDeclare(DeviceMrCache*, device_mr_cache);

static bool deviceMrCacheEnabled()
{
  static const bool on = (getenv("CHARM_DEVICE_MR_CACHE") != nullptr);
  return on;
}

// A device receive pins its completion message to this process: the message
// carries pointers into the buffers the element posted here, and they stay
// referenced until the application has consumed the message -- which for an
// SDAG entry can be long after the transfers completed, since a message for a
// future round buffers in the dependency object. Each entry records the
// stand-downs to release when that message is finally freed, which is the
// one event that means nothing references it any more, wherever it happened.
struct DeviceRecvHold { int aid_idx; CmiUInt8 id; };
typedef std::unordered_map<void*, std::vector<DeviceRecvHold>> DeviceRecvHoldMap;
CkpvDeclare(DeviceRecvHoldMap*, device_recv_holds);

// Envelopes held back by admission control while their element is parked,
// keyed by the bare element id. Replayed -- re-injected into delivery --
// when the element unparks (it can consume again) or departs (delivery then
// forwards them to its new home, unpreprocessed and clean).
typedef std::unordered_map<CmiUInt8, std::vector<void*>> DeviceRecvAdmissionMap;
CkpvDeclare(DeviceRecvAdmissionMap*, device_recv_admission);

// Landing buffers of staged parked receives (CkRdmaDeviceStageParked), keyed
// by the landing pointer the rewritten descriptor now carries. `ready` is
// recorded on the staging stream behind the pull; the delivery waits on it.
struct StagedLanding {
  hapiEvent_t ready;
  size_t cnt;
  const void* src;        // CHARM_STAGE_VERIFY: the address the pull read
  unsigned long long sum; // CHARM_STAGE_VERIFY: checksum of the landing after the pull
  bool verified;
};
static inline bool stageVerifyOn()
{
  static const bool on = (getenv("CHARM_STAGE_VERIFY") != nullptr);
  return on;
}
// Synchronous device-to-host checksum, verification only.
static unsigned long long stageChecksum(const void* dptr, size_t cnt)
{
  std::vector<unsigned char> h(cnt);
  if (cudaMemcpy(h.data(), dptr, cnt, cudaMemcpyDefault) != cudaSuccess) {
    cudaGetLastError();
    return ~0ULL;
  }
  unsigned long long sum = 1469598103934665603ULL;
  for (size_t i = 0; i < cnt; i++) sum = (sum ^ h[i]) * 1099511628211ULL;
  return sum;
}
typedef std::unordered_map<const void*, StagedLanding> StagedLandingMap;
// Stamped into comm_offset of a rewritten descriptor (unused when the protocol
// is NONE). A pointer alone does not identify a landing: `ptr` in an incoming
// message is an address in the SENDER's process, and every process lays its
// device arenas out the same way, so a foreign pointer can equal one of this
// PE's landing buffers. Matching on the pointer alone copied a landing buffer
// into a migration payload once in ~5 sph2d runs (2026-09-15, job 22084755).
static const size_t kStagedLandingMark = (size_t)0x53544147454431ULL;
static inline bool isStagedLandingDescriptor(const CkDeviceBuffer& b)
{
  return b.src_pe == CkMyPe() && b.ipc_protocol == CmiIpcProtocol::NONE &&
         b.device_idx == -1 && b.event_idx == -1 && b.sender_prepared &&
         b.comm_offset == kStagedLandingMark;
}
CkpvDeclare(StagedLandingMap*, staged_landings);
CkpvDeclare(hapiStream_t, staging_stream);
// Carries only the producer wait that an inter-node restage put has to observe
// before the NIC may read the source. Its own stream, so that wait is never
// queued behind an unrelated staging copy.
CkpvDeclare(hapiStream_t, restage_wait_stream);
CkpvDeclare(std::vector<hapiEvent_t>*, staging_events);
// The PE's receive stream: every device receive lands its copy on it, and
// nothing is ever issued on it that cannot run at once (DeviceRecvPending).
CkpvDeclare(hapiStream_t, device_recv_stream);
struct DeviceRecvPending;
CkpvDeclare(std::vector<DeviceRecvPending*>*, device_recv_pending);
// Events for the destination-order fallback when the flag ring is full.
CkpvDeclare(std::vector<hapiEvent_t>*, device_recv_events);
static void deviceRecvPendingPoll(void*);

// Device work in flight on an element (hapiDeviceWorkBegin/End, see hapi.h):
// every HAPI callback registered inside an element's entry method raises its
// outstanding count until the callback fires. Without this a Compute went
// "quiet" the moment it consumed its positions, its force kernels still
// running, and 46% of kicks packed an element whose last kernel had not
// finished -- the pack copies then waited on the device and, on the shared
// migration stream, held every later pack behind them (128-190 ms median).
struct CkDeviceWorkToken {
  CkLocMgr* mgr;
  CmiUInt8 id;
};
static void* ckDeviceWorkBegin()
{
  CkLocRec* r = CkpvAccess(_currentLocRec);
  if (r == NULL) return nullptr;
  r->noteDeviceSendPosted();
  return new CkDeviceWorkToken{r->getLocMgr(), r->getID()};
}
static void ckDeviceWorkEnd(void* token)
{
  CkDeviceWorkToken* t = (CkDeviceWorkToken*)token;
  CkLocRec* r = t->mgr->recForDeviceWork(t->id);   // gone if it migrated or died meanwhile
  if (r) r->noteDeviceSendDone();
  delete t;
}

void CkRdmaDeviceRegistrationCacheInit()
{
  hapiDeviceWorkBegin = ckDeviceWorkBegin;
  hapiDeviceWorkEnd = ckDeviceWorkEnd;
  CkpvInitialize(DeviceMrCache*, device_mr_cache);
  CkpvAccess(device_mr_cache) = deviceMrCacheEnabled() ? new DeviceMrCache() : NULL;
  CkpvInitialize(DeviceRecvHoldMap*, device_recv_holds);
  CkpvAccess(device_recv_holds) = new DeviceRecvHoldMap();
  CkpvInitialize(DeviceRecvAdmissionMap*, device_recv_admission);
  CkpvAccess(device_recv_admission) = new DeviceRecvAdmissionMap();
  CkpvInitialize(StagedLandingMap*, staged_landings);
  CkpvAccess(staged_landings) = new StagedLandingMap();
  CkpvInitialize(hapiStream_t, staging_stream);
  CkpvAccess(staging_stream) = nullptr;
  CkpvInitialize(hapiStream_t, restage_wait_stream);
  CkpvAccess(restage_wait_stream) = nullptr;
  CkpvInitialize(std::vector<hapiEvent_t>*, staging_events);
  CkpvAccess(staging_events) = new std::vector<hapiEvent_t>();
  CkpvInitialize(hapiStream_t, device_recv_stream);
  CkpvAccess(device_recv_stream) = nullptr;
  CkpvInitialize(std::vector<DeviceRecvPending*>*, device_recv_pending);
  CkpvAccess(device_recv_pending) = new std::vector<DeviceRecvPending*>();
  CkpvInitialize(std::vector<hapiEvent_t>*, device_recv_events);
  CkpvAccess(device_recv_events) = new std::vector<hapiEvent_t>();
  CcdCallOnConditionKeep(CcdSCHEDLOOP, (CcdCondFn)deviceRecvPendingPoll, NULL);
}

static hapiStream_t stagingStream()
{
  hapiStream_t& s = CkpvAccess(staging_stream);
  if (s == nullptr) hapiCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  return s;
}
static hapiStream_t restageWaitStream()
{
  hapiStream_t& s = CkpvAccess(restage_wait_stream);
  if (s == nullptr) hapiCheck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  return s;
}
static hapiEvent_t stagingEventTake()
{
  std::vector<hapiEvent_t>& pool = *CkpvAccess(staging_events);
  if (pool.empty()) {
    hapiEvent_t e;
    hapiCheck(hapiEventCreateWithFlags(&e, hapiEventDisableTiming));
    return e;
  }
  hapiEvent_t e = pool.back();
  pool.pop_back();
  return e;
}
static void stagingEventGive(hapiEvent_t e) { CkpvAccess(staging_events)->push_back(e); }

// Runs on the posting stream behind the copy out of a landing buffer: the
// buffer goes back to the pool (CkDeviceFree has no stream ordering of its
// own, so it must be called from here, not at issue) and the event to its pool.
struct StagedLandingFree {
  void* ptr;
  hapiEvent_t ready;
};
static void stagedLandingFreeFn(void* param, void*)
{
  StagedLandingFree* f = (StagedLandingFree*)param;
  CkDeviceFree(f->ptr);
  stagingEventGive(f->ready);
  delete f;
}
static inline bool stagedDbg()
{
  static const bool on = (getenv("CHARM_DEBUG_MIGRATE") != nullptr);
  return on;
}

// Called from CkFreeMsg for every message: release whatever this one held.
// The empty() check keeps the common case at one branch.
// Does the running element hold any unconsumed device-receive message? The
// application checks this before parking in AtSyncWait: a parked element runs
// no SDAG and can never consume, so parking un-quiet would deadlock the
// migration interlock.
bool CkDeviceRecvQuiet()
{
  CkLocRec* rec = CkpvAccess(_currentLocRec);
  return rec == NULL || rec->outstandingDeviceSends == 0;
}


void CkDeviceRecvAdmissionBuffer(CmiUInt8 id, void* env)
{
  (*CkpvAccess(device_recv_admission))[id].push_back(env);
  if (getenv("CHARM_DEBUG_MIGRATE"))
    CmiPrintf("[BUFFER %d] id=%llu depth=%zu\n", CkMyPe(),
              (unsigned long long)id,
              (*CkpvAccess(device_recv_admission))[id].size());
}

void CkDeviceRecvAdmissionReplay(CmiUInt8 id)
{
  DeviceRecvAdmissionMap* m = CkpvAccess(device_recv_admission);
  if (m == NULL || m->empty()) return;
  auto it = m->find(id);
  if (it == m->end()) return;
  std::vector<void*> envs;
  envs.swap(it->second);
  m->erase(it);
  if (envs.size() && getenv("CHARM_DEBUG_MIGRATE"))
    CmiPrintf("[REPLAY %d] id=%llu n=%zu\n", CkMyPe(),
              (unsigned long long)id, envs.size());
  for (size_t k = 0; k < envs.size(); k++)
    CmiPushPE(CmiMyRank(), envs[k]);
}

// Receive-side envelope accounting, on only under CHARM_ZC_LEAKDBG. One copy
// is made per device receive and one free is expected per copy.
std::atomic<long> g_zc_copies{0}, g_zc_frees{0};
std::atomic<long> g_zc_meta{0}, g_zc_metafree{0}, g_zc_empty{0}, g_zc_real{0};
// Receives parked for a dependency still running, and the most parked at once.
static long g_recv_parked = 0, g_recv_parked_max = 0;
bool zcLeakDbg() {
  static const bool on = getenv("CHARM_ZC_LEAKDBG") != NULL;
  return on;
}
void zcLeakReport(const char* where) {
  if (!zcLeakDbg()) return;
  const long c = g_zc_copies.load(), f = g_zc_frees.load();
  if (c % 20000 != 0 || c == 0) return;
  CmiPrintf("[ZCLEAK %d] %s: env %ld/%ld live %ld | meta %ld/%ld live %ld | "
            "ops empty %ld real %ld | parked %ld (max %ld at once)\n", CkMyPe(), where, c, f, c - f,
            g_zc_meta.load(), g_zc_metafree.load(),
            g_zc_meta.load() - g_zc_metafree.load(),
            g_zc_empty.load(), g_zc_real.load(), g_recv_parked, g_recv_parked_max);
}

void CkRdmaDeviceMsgFreed(void* env)
{
  DeviceRecvHoldMap* holds = CkpvAccess(device_recv_holds);
  if (holds == NULL || holds->empty()) return;
  auto it = holds->find(env);
  if (it == holds->end()) return;
  if (zcLeakDbg()) g_zc_frees.fetch_add(1);
  // The map is keyed by envelope ADDRESS, and the allocator recycles
  // addresses. A held message freed through a path that does not run this
  // hook -- any raw CmiFree, of which the runtime has many -- leaves its
  // entry behind. The next envelope allocated at that address then releases
  // its predecessor's holds: an element with live transfers reads zero
  // outstanding sends, migrates, and frees the very buffers those transfers
  // are still landing in. That corrupts the heap on the source process and
  // strands every neighbour waiting on the ghosts it owed.
  //
  // A hold is only released against the message it was recorded for, so
  // confirm this envelope still names that recipient. A recycled address is
  // dropped, not released -- the element it names keeps its stand-down and
  // its migration waits for a real completion.
  envelope* e = (envelope*)env;
  const bool for_elt = (e->getMsgtype() == ForArrayEltMsg);
  for (size_t k = 0; k < it->second.size(); k++) {
    if (!for_elt || e->getRecipientID() != it->second[k].id) {
      static int stale_holds = 0;
      if (++stale_holds <= 8 && getenv("CHARM_DEBUG_MIGRATE"))
        CmiPrintf("[STALEHOLD %d] env=%p recycled: recorded id=%llu, now %s "
                  "id=%llu (n=%d)\n", CkMyPe(), env,
                  (unsigned long long)it->second[k].id,
                  for_elt ? "elt" : "non-elt",
                  (unsigned long long)(for_elt ? e->getRecipientID() : 0),
                  stale_holds);
      continue;
    }
    CkGroupID daid; daid.idx = it->second[k].aid_idx;
    CkNoteDeviceRecvComplete(daid, it->second[k].id);
  }
  holds->erase(it);
}

// One registration per pool segment, shared by every PE of the process: the
// arena under the buddy backend, the run of chunks covering the buffer under
// the vmm backend (hapiDevPoolSegmentOf). Keyed by (base, extent); created on
// first use; marked no-dereg so no completion path can ever release it.
static std::mutex g_deviceArenaRegLock;
static std::map<std::pair<uintptr_t, size_t>, CmiNcpyBuffer> g_deviceArenaRegs;

static bool acquireArenaRegistration(const void* ptr, size_t cnt, CmiNcpyBuffer* out)
{
  if (!hapiDevPoolOn()) return false;
  void* base = NULL;
  size_t extent = 0;
  if (!hapiDevPoolSegmentOf(ptr, cnt, &base, &extent)) return false;

  std::lock_guard<std::mutex> lk(g_deviceArenaRegLock);
  const auto key = std::make_pair((uintptr_t)base, extent);
  auto it = g_deviceArenaRegs.find(key);
  if (it == g_deviceArenaRegs.end()) {
    CmiNcpyBuffer reg(base, extent, CMK_BUFFER_REG, CMK_BUFFER_NODEREG);  // the one registration
    it = g_deviceArenaRegs.emplace(key, reg).first;
    CmiPrintf("[%d] device pool: %s %p (%zu MB) registered once for RDMA\n",
              CmiMyPe(), hapiDevPoolIsVmm() ? "chunk run" : "arena", base, extent >> 20);
  }
  *out = it->second;
  // The registration covers the arena; the transfer is the buffer.
  out->ptr = ptr;
  out->cnt = cnt;
  out->pe = CmiMyPe();
  return true;
}

// Registered descriptor for [ptr, ptr+cnt). A pool buffer gets its arena's
// one registration; otherwise registers on a miss, and without the cache
// behaves exactly as constructing one in place did.
static CmiNcpyBuffer acquireDeviceRegistration(const void* ptr, size_t cnt,
                                               CkLocRec* owner)
{
  CmiNcpyBuffer arena;
  if (acquireArenaRegistration(ptr, cnt, &arena)) return arena;

  DeviceMrCache* cache = CkpvAccess(device_mr_cache);
  if (cache == NULL) return CmiNcpyBuffer(ptr, cnt);

  auto it = cache->find(ptr);
  if (it == cache->end()) {
    DeviceMrEntry entry;
    entry.reg = CmiNcpyBuffer(ptr, cnt);   // the one real registration
    entry.owner = owner;
    it = cache->emplace(ptr, entry).first;
  } else {
    if (cnt > it->second.reg.cnt) {
      // A larger transfer from the same buffer: re-register the larger
      // extent, once. Counts only ever grow entries; they never multiply them.
      it->second.reg.deregisterMem();
      it->second.reg = CmiNcpyBuffer(ptr, cnt);
    }
    if (it->second.owner == NULL) {
      // First use came from a context with no element attached; adopt the
      // first owner that does appear, so the entry becomes retirable.
      it->second.owner = owner;
    }
  }
  CmiNcpyBuffer out = it->second.reg;
  // The registration spans at least [ptr, ptr+cnt); the transfer must still
  // be sized by the request, not by the registered extent.
  out.cnt = cnt;
  return out;
}

// Retire everything registered for an element. Called from the migration paths,
// where the element's device buffers are about to be freed.
void CkRdmaDeviceDropRegistrations(CkLocRec* owner)
{
  DeviceMrCache* cache = CkpvAccess(device_mr_cache);
  if (cache == NULL || owner == NULL) return;
  for (auto it = cache->begin(); it != cache->end();) {
    if (it->second.owner == owner) {
      it->second.reg.deregisterMem();
      it = cache->erase(it);
    } else {
      ++it;
    }
  }
}

struct LoopBackMsg {
  char header[CmiMsgHeaderSizeBytes];
  void* msg;
};

extern "C" {
  void* loopback_bridge(void* arg) {
    QdProcess(1);
    LoopBackMsg* recv_msg = (LoopBackMsg*)arg;
    CkRdmaDeviceRecvHandler(recv_msg->msg);
    CmiFree(recv_msg);
    return NULL;
  }
  
  int loopback_handler;
}

// Sender side of a correction put; defined with the rest of the protocol below.
static void notifyDeviceRestagePut(int dest_pe, void* dest_op);

// Neither completion handler below releases the migration stand-downs the
// message's receives took: those release when the completion message itself
// dies (CkRdmaDeviceMsgFreed), i.e. at consumption. Releasing at transfer
// completion is too early -- noteDeviceSendDone fires a pending migrateMe,
// the element packs and frees its buffers before the just-enqueued message
// is consumed, and the message is then forwarded to the new process carrying
// pointers into this one. That was the DiffusionLB-async failure: diffusion
// moves both ends of many send pairs per step, so the window was hit
// constantly.

void CkRdmaDeviceRecvHandler(void* data)
{
  NcpyOperationInfo *ncpy_op_info = (NcpyOperationInfo *)data;

  if (ncpy_op_info->opMode == CMK_DEVICE_RESTAGE_PUT) {
    // A put raises its completion on the initiator only, and for a correction
    // the initiator is the sender. deviceRdmaOpInfo names the receiver's
    // DeviceRdmaOp, which is a pointer into that other process and must not be
    // followed here -- so tell the receiver the payload has landed and let it
    // resolve the op. destPe is a plain int in the ncpyOpInfo, so it is the one
    // piece of the destination that is safe to read from this side.
    notifyDeviceRestagePut(ncpy_op_info->destPe,
                           ncpy_op_info->deviceRdmaOpInfo);
    return;
  }

  DeviceRdmaOp* op = (DeviceRdmaOp*)(ncpy_op_info->deviceRdmaOpInfo);

  if(op->dest_pe != CmiMyPe()) {
        int infoSize = ncpy_op_info->ncpyOpInfoSize;
        NcpyOperationInfo* copy = (NcpyOperationInfo*)CmiAlloc(infoSize);
        memcpy(copy, ncpy_op_info, infoSize);

        LoopBackMsg* conv_msg = (LoopBackMsg*)CmiAlloc(sizeof(LoopBackMsg));
        conv_msg->msg = copy;

        QdCreate(1);
        CmiSetHandler(conv_msg, loopback_handler);
        CmiPushPE(CmiRankOf(op->dest_pe), conv_msg);
        return;
  }

  QdProcess(1);
  DeviceRdmaInfo* info = op->info;

  // Invoke source callbacks
  if (op->src_cb) {
    CkCallback* cb = (CkCallback*)op->src_cb;
    cb->send();
    delete cb;
  }

  // Cross-node tier, per op: this fires for every completed rget, whether or
  // not the receive it belongs to ever completes as a whole.
  if (op->rget_posted > 0.0) {
    zcRecordRecvTime(2, op->size, CkWallTimer() - op->rget_posted);
    op->rget_posted = 0.0;
  }

  // Update counter (there may be multiple buffers in transit)
  info->counter++;

  // Check if all buffers have been received
  // If so, invoke regular entry method
  if (info->counter == info->n_ops) {
    if (info->msg == nullptr) {  // a staged pull for a parked message: nothing to deliver
      CmiFree(info);
      return;
    }
    QdCreate(1);

    enqueueNcpyMessage(op->dest_pe, info->msg);

    // Free RDMA metadata
    // CmiFree(info);
  }
}
static void deviceRecvWatchDrop(DeviceRdmaInfo* info);  // defined by the stall watch below
// Defined just below the zerocopy stats, which live in an anonymous namespace
// this declaration cannot name. The completion path above is where a receive's
// elapsed time becomes known, hence the forward declaration.

// Invoked when a GPU buffer arrives on the receiver
void CkRdmaDeviceRecvHandler(void* data, void* msg)
{
  DeviceRdmaOp* op = (DeviceRdmaOp*)data;
  DeviceRdmaInfo* info = op->info;

  // The copy out of the sender's IPC slot has landed (this runs behind it on
  // the receive stream), so the sender may take the slot back. Raised here,
  // at completion, rather than when the copy was issued: that is what lets
  // reclaimCompletedIpcEvents trust the flag alone and query no event.
  // The sender clears it when it reclaims the slot, so finding it already set
  // means a second receive is signalling the same (device, event) pair before
  // the first was retired -- the sender would then free a block another
  // transfer is still reading.
  if (op->ipc_event_idx >= 0) {
    hapi_ipc_event_shared* sh = ipcSharedSlot(op->ipc_device_idx, op->ipc_event_idx);
    if (sh->dst_flag.exchange(true, std::memory_order_acq_rel)) {
      CmiPrintf("[%d] IPC DUPLICATE dst_flag dev_idx=%d ev_idx=%d srcPe=%d cnt=%zu\n",
                CkMyPe(), op->ipc_device_idx, op->ipc_event_idx, op->src_pe, op->size);
      fflush(stdout);
    }
    op->ipc_event_idx = -1;
  }

  // Invoke source callbacks
  if (op->src_cb) {
    CkCallback* cb = (CkCallback*)op->src_cb;
    cb->send();
    delete cb;
  }

  // Cross-node tier, per op: this fires for every completed rget, whether or
  // not the receive it belongs to ever completes as a whole.
  if (op->rget_posted > 0.0) {
    zcRecordRecvTime(2, op->size, CkWallTimer() - op->rget_posted);
    op->rget_posted = 0.0;
  }

  // Update counter (there may be multiple buffers in transit)
  info->counter++;

  // Check if all buffers have been received
  // If so, invoke regular entry method
  if (info->counter == info->n_ops) {
    if (info->msg == nullptr) {
      // A staged pull for a parked message (CkRdmaDeviceStageParked): the
      // senders are complete, the payload sits in landing buffers, and the
      // message itself is delivered when its element lands.
      if (stageVerifyOn()) {
        StagedLandingMap& landings = *CkpvAccess(staged_landings);
        for (int k = 0; k < info->n_ops; k++) {
          // ops are laid out after the info block, live ones only were counted,
          // so walk by address: every op whose dest is a registered landing
          DeviceRdmaOp* o = (DeviceRdmaOp*)((char*)info + sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * k);
          auto lit = landings.find(o->dest_ptr);
          if (lit == landings.end()) continue;
          const unsigned long long land_sum = stageChecksum(o->dest_ptr, lit->second.cnt);
          const unsigned long long src_sum = lit->second.src ? stageChecksum(lit->second.src, lit->second.cnt) : 0;
          lit->second.sum = land_sum;
          lit->second.verified = true;
          if (lit->second.src && src_sum != land_sum)
            CmiPrintf("[STAGE-VERIFY %d] pull mismatch: landing %p (%zu B) sum %llx != source %p sum %llx\n",
                      CkMyPe(), o->dest_ptr, lit->second.cnt, land_sum, lit->second.src, src_sum);
        }
      }
      deviceRecvWatchDrop(info);
      CmiFree(info);
      return;
    }
    QdCreate(1);
    if (zcLeakDbg()) g_zc_metafree.fetch_add(1);

    // The receive is complete here, which is the moment the application can
    // act on it, so this is the interval worth charging to the transport.
    if (info->zc_posted > 0.0)
      zcRecordRecvTime(info->zc_mode, info->zc_bytes,
                       CkWallTimer() - info->zc_posted);

    enqueueNcpyMessage(op->dest_pe, info->msg);

    // Free RDMA metadata
    deviceRecvWatchDrop(info);
    CmiFree(info);
  }
}

/****************************** Direct (Persistent) API ******************************/

void CkDevicePersistent::init() {
  pe = CkMyPe();
  cb_msg = nullptr;
  ipc_ptr = nullptr;
  ipc_open = false;
}

void CkDevicePersistent::open() {
  // Create a CUDA IPC handle for inter-process communication
  hapiCheck(hapiIpcGetMemHandle(&hapi_ipc_handle, (void*)ptr));
}

void CkDevicePersistent::close() {
  // Close the CUDA IPC handle if it was opened
  hapiCheck(hapiIpcCloseMemHandle(ipc_ptr));
}

void CkDevicePersistent::set_msg(void* msg) {
  cb_msg = msg;
}

void CkDevicePersistent::pup(PUP::er& p) {
  p((char*)&ptr, sizeof(ptr));
  p|cnt;
  p|pe;
  p|cb;
  p((char*)&hapi_ipc_handle, sizeof(hapi_ipc_handle));
}

CkDeviceStatus CkDevicePersistent::get(CkDevicePersistent& src) {
  // Check that the source buffer fits into the destination buffer
  if (cnt < src.cnt) {
    CkAbort("CkDevicePersistent::get: Destination buffer is smaller than source buffer\n");
  }

  CkNcpyModeDevice mode = findTransferModeDevice(src.pe, CkMyPe());

  // Perform get
  if (mode == CkNcpyModeDevice::MEMCPY) {
    hapiMemcpyAsync((void*)ptr, src.ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!src.ipc_open) {
      hapiCheck(hapiIpcOpenMemHandle(&src.ipc_ptr, src.hapi_ipc_handle,
            hapiIpcMemLazyEnablePeerAccess));
      src.ipc_open = true;
    }
    hapiMemcpyAsync((void*)ptr, src.ipc_ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else {
    CkAbort("Persistant GPU messaging is currently not supported for inter-node messages");
  }

  // Set callbacks to be invoked once get is complete
  if (src.cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, src.cb, src.cb_msg);
  }
  if (cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, cb, cb_msg);
  }

  return CkDeviceStatus::incomplete;
}

CkDeviceStatus CkDevicePersistent::put(CkDevicePersistent& dst) {
  // Check that the source buffer fits into the destination buffer
  if (dst.cnt < cnt) {
    CkAbort("CkDevicePersistent::put: Destination buffer is smaller than source buffer\n");
  }

  CkNcpyModeDevice mode = findTransferModeDevice(CkMyPe(), dst.pe);

  // Perform put
  if (mode == CkNcpyModeDevice::MEMCPY) {
    hapiMemcpyAsync((void*)dst.ptr, ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else if (mode == CkNcpyModeDevice::IPC) {
    if (!dst.ipc_open) {
      hapiCheck(hapiIpcOpenMemHandle(&dst.ipc_ptr, dst.hapi_ipc_handle,
            hapiIpcMemLazyEnablePeerAccess));
      dst.ipc_open = true;
    }
    hapiMemcpyAsync(dst.ipc_ptr, ptr, cnt, hapiMemcpyDeviceToDevice, hapi_stream);
  } else {
    CkAbort("Persistant GPU messaging is not yet supported for inter-node messages");
  }

  // Set callbacks to be invoked once get is complete
  if (cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, cb, cb_msg);
  }
  if (dst.cb.type != CkCallback::ignore) {
    hapiAddCallback(hapi_stream, dst.cb, dst.cb_msg);
  }

  return CkDeviceStatus::incomplete;
}

/****************************** Recv Entry Method API ******************************/

// Returns the local rank of the logical node (process) that the given PE belongs to
static inline int CmiNodeRankLocal(int pe) {
  // Logical node index % Number of logical nodes per physical node
  return CmiNodeOf(pe) % (CmiNumNodes() / CmiNumPhysicalNodes());
}

// Returns the local rank of the logical node that I belong to
static inline int CmiMyNodeRankLocal() {
  return CmiNodeRankLocal(CmiMyPe());
}

// Debug/validation knobs on the per-message IPC path, resolved once.
//
// getenv is a linear strncmp walk of the environment and a miss -- which is the
// normal case for these -- scans all of it: measured at 770ns with 93 variables
// and 950ns with 243, the sort of environment a batch scheduler hands out. The
// staged path reads these nine times per message (five ipcDebugSync calls plus
// four direct checks across send and receive), so leaving them uncached spent
// 7-9us per message doing nothing. Read them through these accessors; do not
// call getenv on this path.
static inline bool ipcDebugOn() {
  static const bool on = (getenv("CHARM_DEBUG_IPC_RECV") != nullptr);
  return on;
}

static inline bool zcValidateOn() {
  static const bool on = (getenv("CHARM_ZC_VALIDATE") != nullptr);
  return on;
}

// TEMPORARY (CHARM_DEBUG_IPC_RECV): synchronize after each individual CUDA
// operation on the shm/IPC path and abort naming the exact step that failed.
// Illegal-access errors are sticky and asynchronous, so without this they
// surface at whatever call happens to be checked next -- which is how the same
// fault has been reported from three unrelated lines. With it, the first
// failing operation identifies itself.
static inline void ipcDebugSync(const char* step, hapiStream_t stream) {
  if (!ipcDebugOn()) return;
  hapiSubmitDrain();   // the step being checked may still be in a submitter ring
  hapiError_t err = ckSubmitDrainThenSync(stream);
  if (err != hapiSuccess) {
    CmiPrintf("[%d] IPC step '%s' FAILED: %s\n", CmiMyPe(), step,
              cudaGetErrorString(err));
    fflush(stdout);
    CmiAbort("IPC debug: step '%s' failed", step);
  }
}

// Per-process tally of how device zerocopy receives actually resolved, enabled
// with CHARM_ZC_STATS=1. Says whether a placement decision (e.g. a
// communication-aware load balancer) turned cross-process IPC transfers into
// same-process device-to-device copies, which is the locality question that
// raw timings alone cannot answer. Counting is a relaxed atomic increment on a
// path that already issues a CUDA call, and the env lookup is cached, so an
// instrumented run stays representative.
namespace {
struct ZcModeStats {
  std::atomic<long> memcpy_n{0};
  std::atomic<long> ipc_n{0};
  std::atomic<long> other_n{0};
  // Completed receives, their bytes, and the wall time from posting the receive
  // to its last op completing -- per mode. Counts alone answer "did placement
  // move traffic to a slower transport"; they cannot answer "by how much", and
  // a cost model needs the second question. A pingpong benchmark cannot answer
  // it either: it measures a transfer alone on the machine, whereas what a
  // balancer needs is what the transfer costs amid all the others contending
  // for the same device, link and NIC. Only the application's own traffic has
  // that contention in it.
  //
  // Recorded per receive, not per op: the receive is the unit the application
  // waits on, and its ops complete concurrently.
  std::atomic<long> recv_n[3];
  std::atomic<long> recv_bytes[3];
  // Microseconds, as an integer so the accumulation stays a relaxed atomic add.
  std::atomic<long> recv_us[3];

  ZcModeStats() {
    for (int i = 0; i < 3; i++) { recv_n[i] = 0; recv_bytes[i] = 0; recv_us[i] = 0; }
  }
  static const char* slotName(int s) {
    return (s == 0) ? "MEMCPY" : (s == 1) ? "IPC" : "OTHER";
  }
  ~ZcModeStats() {
    const long m = memcpy_n.load(), i = ipc_n.load(), o = other_n.load();
    const long total = m + i + o;
    if (total == 0) return;
    fprintf(stderr, "[zc-stats] pid=%d MEMCPY=%ld IPC=%ld OTHER=%ld "
                    "(same-process %.1f%% of %ld)\n",
            (int)getpid(), m, i, o, 100.0 * m / total, total);
    for (int s = 0; s < 3; s++) {
      const long n = recv_n[s].load();
      if (n == 0) continue;
      const long us = recv_us[s].load(), by = recv_bytes[s].load();
      // mean us/receive and mean bytes/receive: fit alpha and beta across the
      // three rows and the result already carries the real contention.
      fprintf(stderr, "[zc-time] pid=%d %-6s recvs=%ld bytes=%ld total_us=%ld "
                      "mean_us=%.3f mean_bytes=%.1f\n",
              (int)getpid(), slotName(s), n, by, us, (double)us / n,
              (double)by / n);
    }
    fflush(stderr);
  }
};
ZcModeStats zc_mode_stats;

// Sender-side companion: how often the destination could not be resolved at
// send time. An unresolved destination cannot take the cheap same-process path,
// so a burst of these right after a migration is what turns the first
// post-migration step into a slow one.
struct ZcDestStats {
  std::atomic<long> confirmed{0};
  std::atomic<long> unconfirmed{0};
  ~ZcDestStats() {
    const long c = confirmed.load(), u = unconfirmed.load();
    if (c + u == 0) return;
    fprintf(stderr, "[zc-dest] pid=%d confirmed=%ld unconfirmed=%ld (%.2f%% unresolved)\n",
            (int)getpid(), c, u, 100.0 * u / (c + u));
    fflush(stderr);
  }
};
ZcDestStats zc_dest_stats;

inline void zcDestCount(bool confirmed) {
  static const bool on = (getenv("CHARM_ZC_STATS") != nullptr);
  if (!on) return;
  if (confirmed) zc_dest_stats.confirmed.fetch_add(1, std::memory_order_relaxed);
  else zc_dest_stats.unconfirmed.fetch_add(1, std::memory_order_relaxed);
}

// Slot a mode falls into, matching the count buckets above.
inline int zcModeSlot(CkNcpyModeDevice mode) {
  if (mode == CkNcpyModeDevice::MEMCPY) return 0;
  if (mode == CkNcpyModeDevice::IPC) return 1;
  return 2;
}

inline bool zcStatsOn() {
  static const bool on = (getenv("CHARM_ZC_STATS") != nullptr);
  return on;
}

// Called once per receive, when its last op completes.
inline void zcStatsTime(int slot, size_t bytes, double seconds) {
  if (!zcStatsOn()) return;
  if (slot < 0 || slot > 2) return;
  zc_mode_stats.recv_n[slot].fetch_add(1, std::memory_order_relaxed);
  zc_mode_stats.recv_bytes[slot].fetch_add((long)bytes, std::memory_order_relaxed);
  zc_mode_stats.recv_us[slot].fetch_add((long)(seconds * 1e6),
                                        std::memory_order_relaxed);
}

inline void zcStatsCount(CkNcpyModeDevice mode) {
  static const bool on = (getenv("CHARM_ZC_STATS") != nullptr);
  if (!on) return;
  switch (mode) {
    case CkNcpyModeDevice::MEMCPY: zc_mode_stats.memcpy_n.fetch_add(1, std::memory_order_relaxed); break;
    case CkNcpyModeDevice::IPC:    zc_mode_stats.ipc_n.fetch_add(1, std::memory_order_relaxed); break;
    default:                       zc_mode_stats.other_n.fetch_add(1, std::memory_order_relaxed); break;
  }
}
}  // namespace

// File-scope bridge to the tally above, for the completion path that runs
// earlier in this translation unit.
void zcRecordRecvTime(int slot, size_t bytes, double seconds)
{
  zcStatsTime(slot, bytes, seconds);
}

// Per-PE ring of events used only to order same-process transfers.
//
// A wait enqueued on an event captures the most recent record at enqueue time,
// and the sender always records before the message is sent, so the receiver's
// wait cannot run ahead of the record. Re-recording an event that some earlier
// receiver is still waiting on is therefore harmless: that wait already
// captured the earlier record. The ring only needs to be large enough that
// re-use does not add false dependencies, not for correctness.
static hapiEvent_t* ckDeviceMemcpyEventRing() {
  static const int ring_size = []() {
    const char* s = getenv("CHARM_ZC_MEMCPY_EVENTS");
    const int n = s ? atoi(s) : 1024;
    return n > 0 ? n : 1024;
  }();
  static thread_local hapiEvent_t* ring = nullptr;
  static thread_local int next = 0;
  if (ring == nullptr) {
    ring = new hapiEvent_t[ring_size];
    for (int i = 0; i < ring_size; i++) {
      if (hapiEventCreateWithFlags(&ring[i], hapiEventDisableTiming) != hapiSuccess) {
        delete[] ring;
        ring = nullptr;
        return (hapiEvent_t*)nullptr;
      }
    }
  }
  return ring;
}

void* ckDeviceRecordMemcpyEvent(hapiStream_t stream) {
  hapiSubmitDrain();   // recorded directly and QUERIED by the receiver: it must follow the queued work
  static const int ring_size = []() {
    const char* s = getenv("CHARM_ZC_MEMCPY_EVENTS");
    const int n = s ? atoi(s) : 1024;
    return n > 0 ? n : 1024;
  }();
  static thread_local int next = 0;
  hapiEvent_t* ring = ckDeviceMemcpyEventRing();
  if (ring == nullptr) return NULL;
  hapiEvent_t ev = ring[next];
  next = (next + 1) % ring_size;
  if (hapiEventRecord(ev, stream) != hapiSuccess) {
    // Expected here: the event can belong to another device (see the
    // cross-device notes below), and this is a try-and-fall-back. Clear the
    // sticky error so it is not left for the next cudaPeekAtLastError in
    // application code, which would abort on a failure it did not cause and
    // which was already handled.
    cudaGetLastError();
    return NULL;
  }
  return (void*)ev;
}

// Invoked after post entry method
// The staged/direct CUDA IPC receive, lifted out of CkRdmaDeviceIssueRgets so
// that a transfer deferred by the migration-mismatch correction below can run
// exactly the same code when the sender's retransmit metadata arrives. Pure
// extraction: the caller still owns completion (the hapiAddCallback at the end
// of the loop), and every value this needs is passed in rather than closed over.
static void deviceIpcReceive(CkDeviceBuffer& source, CkDeviceBuffer& dest,
                             hapiStream_t recv_stream, int srcPe,
                             CkNcpyModeDevice mode)
{
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  const bool sender_exported =
      (source.device_idx != -1 && csv_gpu_manager.use_shm);
  const bool sender_direct =
      (sender_exported && source.ipc_protocol == CmiIpcProtocol::DIRECT);
  (void)sender_exported;
      // sender_exported already guarantees device_idx is a real index; guard the
      // upper bound only, since a corrupted index would index the pool out of
      // bounds and surface later as an asynchronous illegal access, far from here.
      if ((size_t)source.device_idx >= csv_gpu_manager.hapi_ipc_device_infos.size()) {
        CkAbort("CkRdmaDeviceIssueRgets: receive on PE %d from PE %d carries an "
                "out-of-range IPC device index %d (pool size %zu).",
                CkMyPe(), srcPe, source.device_idx,
                csv_gpu_manager.hapi_ipc_device_infos.size());
      }
      // Inter-process using shared memory optimizations
      // Use optimiziations with POSIX shared memory
      hapi_ipc_device_info& device_info =
        csv_gpu_manager.hapi_ipc_device_infos[source.device_idx];

      // TEMPORARY: locate the invalid pointer behind the illegal access seen in
      // the first cross-process ghost exchange. Prints the peer mapping and the
      // pool sizes actually backing the indices used just below, so a null peer
      // buffer, an out-of-range index, or a desynchronised event pool is
      // visible directly rather than inferred.
      if (ipcDebugOn()) {
        CmiPrintf("[%d] IPC recv: dev_idx=%d (infos=%zu) ev_idx=%d "
                  "(src_pool=%zu dst_pool=%zu flags=%zu) peer_buf=%p "
                  "off=%zu dest=%p cnt=%zu src_pe=%d\n",
                  CkMyPe(), source.device_idx,
                  csv_gpu_manager.hapi_ipc_device_infos.size(),
                  source.event_idx,
                  device_info.src_event_pool.size(),
                  device_info.dst_event_pool.size(),
                  device_info.event_pool_flags.size(),
                  device_info.buffer, (size_t)source.comm_offset,
                  (void*)dest.ptr, (size_t)dest.cnt, source.src_pe);
        fflush(stdout);
      }

      // 0. Resolve where the bytes actually are.
      //
      // Staged: in the peer's communication buffer, whose mapping every process
      // opened once at startup.
      //
      // Direct: in the peer's own allocation, which has to be mapped now --
      // unless this delivery turned out to be same-process after all, because
      // the target migrated between the send and its arrival. Then the sender's
      // pointer is directly readable and must be used: the driver does not let
      // a process open a handle it exported itself.
      void* imported_base = NULL;
      const void* src_addr;
      if (!sender_direct) {
        src_addr = (const void*)((char*)device_info.buffer + source.comm_offset);
      } else if (mode == CkNcpyModeDevice::MEMCPY) {
        src_addr = source.ptr;
      } else {
        imported_base = hapiIpcImportBuffer(source.ipc_handle,
                                            CmiNodeOf(srcPe), source.ipc_base,
                                            source.ipc_offset + (size_t)dest.cnt);
        if (imported_base == NULL) {
          // Name the CUDA error rather than listing the possibilities.
          // cudaErrorAlreadyMapped means this process still holds a mapping of
          // this memory that the import cache did not find; an invalid-handle
          // error means the exporter's allocation is genuinely gone, so the
          // sender released it while its handle was in flight.
          CkAbort("CkRdmaDeviceIssueRgets: receive on PE %d from PE %d could not "
                  "open the CUDA IPC handle for a %zu-byte direct transfer: "
                  "%s (%d).",
                  CkMyPe(), srcPe, (size_t)dest.cnt,
                  hapiIpcLastImportErrorName(), hapiIpcLastImportError());
        }
        src_addr = (const void*)((char*)imported_base + source.ipc_offset);
      }

      // No wait here: the caller issues this only once the sender's work is
      // known complete on the host (DeviceRecvPending), so nothing on
      // recv_stream can block behind a sender.

      // 2. Invoke hapiMemcpyAsync from the peer's memory to the destination
      //    buffer. This is the only copy a direct transfer makes.
      // Same reason as the same-process copy above: the peer's buffer may be on
      // a different device from ours.
      hapiSubmitMemcpyAsync((void*)dest.ptr, src_addr, dest.cnt, cudaMemcpyDefault, recv_stream);
      ipcDebugSync("recv 2: peer copy -> dest", recv_stream);
      // The slot is released when this copy completes: the caller recorded
      // it in the op, and CkRdmaDeviceRecvHandler raises its dst_flag then.
      // No destination event is recorded any more -- nothing queries it.
}


// Defined below, next to the reclaim scan it drives.
static bool acquireIpcSendSlot(DeviceManager* dm, int cpv_my_device_id,
                               bool is_lb_buffer, bool direct,
                               const void* src_ptr, size_t cnt,
                               void** out_buffer, int* out_event_idx,
                               bool wait = true);

/*************** Host-gated receive issue ***************/
//
// A device receive has two dependencies: the sender's producing work, and
// whatever ran on the posted stream before the post (the buffer's previous
// reader or writer). Putting either into a stream as a wait makes everything
// queued behind it wait too: on the consumer's compute stream that stalled
// other chares' kernels behind an unrelated sender; on a shared receive
// stream it would stall other receives. So neither goes on the GPU. The copy
// is issued only once both are known complete on the host, on the PE's one
// receive stream, where consequently nothing ever blocks. A receive whose
// dependencies are still running is parked and re-checked from the scheduler
// loop -- with plain loads where the sender or the post issued a pinned flag,
// and with an event query no more than once per kRecvQueryPeriod otherwise.

static hapiStream_t deviceRecvStream()
{
  hapiStream_t& s = CkpvAccess(device_recv_stream);
  if (s == nullptr) s = hapiAcquireRuntimeStream(/*highPriority=*/true);
  return s;
}

static hapiEvent_t deviceRecvEventTake()
{
  std::vector<hapiEvent_t>& pool = *CkpvAccess(device_recv_events);
  if (pool.empty()) {
    hapiEvent_t e;
    hapiCheck(hapiEventCreateWithFlags(&e, hapiEventDisableTiming));
    return e;
  }
  hapiEvent_t e = pool.back();
  pool.pop_back();
  return e;
}
static void deviceRecvEventGive(hapiEvent_t e)
{
  CkpvAccess(device_recv_events)->push_back(e);
}

static inline hapi_ipc_event_shared* ipcSharedSlot(int device_idx, int event_idx)
{
  GPUManager& gm = CsvAccess(gpu_manager);
  return (hapi_ipc_event_shared*)((char*)gm.shm_ptr
      + gm.shm_chunk_size * device_idx + sizeof(hapiIpcMemHandle_t)) + event_idx;
}

// The sender's half of the source condition for a cross-process send: once
// the work that produced this slot's payload has completed, raise src_ready
// in the shared slot so the receiving process sees it without a driver call.
// A HAPI callback on the producing stream; reclaimCompletedIpcEvents will not
// free the slot before it has fired.
static void ipcSrcReadyFn(void* param, void*)
{
  const uintptr_t packed = (uintptr_t)param;
  ipcSharedSlot((int)(packed >> 32), (int)(uint32_t)packed)
      ->src_ready.store(true, std::memory_order_release);
}
static void ipcPublishSrcReady(int device_idx, int event_idx, hapiStream_t stream)
{
  const uintptr_t packed = ((uintptr_t)(uint32_t)device_idx << 32) | (uint32_t)event_idx;
  hapiAddCallback(stream, CkCallback(ipcSrcReadyFn, (void*)packed));
}
// For a sender that settled its stream synchronously: ready now.
static void ipcPublishSrcReadyNow(int device_idx, int event_idx)
{
  ipcSharedSlot(device_idx, event_idx)->src_ready.store(true, std::memory_order_release);
}

// Decide at post time what the destination still has to wait for: nothing if
// the poster said the buffer is free or the posted stream is idle; else a
// flag on the posted stream, or an event when the ring has no slot. Kept in
// the op so a receive deferred for correction resumes with the same
// condition, taken before anything later was added to that stream.
static void deviceRecvNoteDestination(DeviceRdmaOp& op, const CkDeviceBufferPost& post)
{
  op.dst_flag_rank = -1;
  op.dst_flag_seq = 0;
  op.dst_event = NULL;
  if (post.buffer_free) return;
  hapiSubmitDrain();   // queued work on the posted stream must be in the driver before it is asked
  const cudaError_t q = cudaStreamQuery(post.hapi_stream);
  if (q == cudaSuccess) return;
  if (q != cudaErrorNotReady) { hapiCheck(q); return; }
  cudaGetLastError();   // NotReady is the answer, not an error to leave behind
  int rank = -1;
  uint32_t seq = 0;
  if (hapiFlagIssue(post.hapi_stream, &rank, &seq)) {
    op.dst_flag_rank = rank;
    op.dst_flag_seq = seq;
    return;
  }
  hapiEvent_t ev = deviceRecvEventTake();
  if (hapiEventRecord(ev, post.hapi_stream) == hapiSuccess) {
    op.dst_event = (void*)ev;
  } else {
    // Another device's stream (a chare that moved between GPUs): our event
    // cannot be recorded on it. Settle it instead; rare and bounded.
    cudaGetLastError();
    deviceRecvEventGive(ev);
    hapiCheck(ckSubmitDrainThenSync(post.hapi_stream));
  }
}

enum class DeviceRecvKind : char { Memcpy, Ipc, Staged };

struct DeviceRecvPending {
  DeviceRdmaOp* op = nullptr;
  CkDeviceBuffer source;
  CkDeviceBuffer dest;
  CkNcpyModeDevice mode = CkNcpyModeDevice::MEMCPY;
  int src_pe = -1;
  DeviceRecvKind kind = DeviceRecvKind::Memcpy;
  StagedLanding sl{};          // Staged only
  CkCallbackFn done = nullptr; // completion handler, called with op
  bool src_ready = false;
  bool dst_ready = false;
  double next_query = 0.0;     // earliest time for the next driver-call check
};

static const double kRecvQueryPeriod = 100e-6;

static inline bool deviceRecvEventDone(hapiEvent_t ev)
{
  const cudaError_t q = cudaEventQuery(ev);
  if (q == cudaSuccess) return true;
  if (q != cudaErrorNotReady) hapiCheck(q);
  cudaGetLastError();
  return false;
}

// Is the sender's producing work done? Plain loads first; a driver query only
// when may_query.
static bool deviceRecvSourceReady(const DeviceRecvPending& p, bool may_query)
{
  GPUManager& gm = CsvAccess(gpu_manager);
  switch (p.kind) {
  case DeviceRecvKind::Memcpy:
    if (p.source.ready_seq != 0)
      return hapiFlagLanded(p.source.ready_rank, p.source.ready_seq);
    if (p.source.memcpy_event == NULL) return true;   // the sender settled synchronously
    return may_query && deviceRecvEventDone((hapiEvent_t)p.source.memcpy_event);   // recorded after a drain, see ckDeviceRecordMemcpyEvent
  case DeviceRecvKind::Ipc:
    if (p.source.event_idx < 0 ||
        (size_t)p.source.device_idx >= gm.hapi_ipc_device_infos.size())
      return true;   // deviceIpcReceive reports a bad index
    if (ipcSharedSlot(p.source.device_idx, p.source.event_idx)
            ->src_ready.load(std::memory_order_acquire))
      return true;
    // Submit mode: the sender's event may still sit in ITS submitter's queue, and
    // an unrecorded event reads as complete. src_ready (a flag) is the only word.
    if (hapiSubmitOn()) return false;
    return may_query &&
        deviceRecvEventDone(gm.hapi_ipc_device_infos[p.source.device_idx]
                                .src_event_pool[p.source.event_idx]);
  case DeviceRecvKind::Staged:
    return may_query && deviceRecvEventDone(p.sl.ready);
  }
  return true;
}

// Has everything issued on the posted stream before the post completed?
static bool deviceRecvDestinationReady(DeviceRecvPending& p, bool may_query)
{
  DeviceRdmaOp& op = *p.op;
  if (op.dst_flag_seq != 0) return hapiFlagLanded(op.dst_flag_rank, op.dst_flag_seq);
  if (op.dst_event == NULL) return true;
  if (!may_query || !deviceRecvEventDone((hapiEvent_t)op.dst_event)) return false;
  deviceRecvEventGive((hapiEvent_t)op.dst_event);
  op.dst_event = NULL;
  return true;
}

static void deviceRecvIssue(DeviceRecvPending& p)
{
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  hapiStream_t rs = deviceRecvStream();
  CkDeviceBuffer& source = p.source;
  CkDeviceBuffer& dest = p.dest;
  switch (p.kind) {
  case DeviceRecvKind::Staged: {
    // The pull's completion was the source condition; copy the landing into
    // the posted buffer and free it behind that copy.
    const StagedLanding& sl = p.sl;
    if (stageVerifyOn()) {
      const unsigned long long now = stageChecksum(source.ptr, sl.cnt);
      if (!sl.verified)
        CmiPrintf("[STAGE-VERIFY %d] delivered before the pull's completion was seen: landing %p\n",
                  CkMyPe(), source.ptr);
      else if (now != sl.sum)
        CkAbort("[STAGE-VERIFY %d] landing %p (%zu B) changed between pull and delivery: %llx -> %llx",
                CkMyPe(), source.ptr, sl.cnt, sl.sum, now);
    }
    hapiSubmitMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt, cudaMemcpyDefault, rs);
    hapiAddCallback(rs, CkCallback(stagedLandingFreeFn,
                                   (void*)new StagedLandingFree{(void*)source.ptr, sl.ready}));
    if (stagedDbg())
      CmiPrintf("[STAGE-USE %d] land=%p cnt=%zu -> posted %p\n", CkMyPe(),
                source.ptr, (size_t)dest.cnt, (void*)dest.ptr);
    break;
  }
  case DeviceRecvKind::Memcpy: {
    // Same process: a device-to-device copy from the sender's buffer. The
    // sender's producing work is already complete (deviceRecvSourceReady),
    // so no stream wait precedes it.
    // A sender that staged IPC state for this transfer anyway -- an
    // unconfirmed destination that turned out to be this process, or a send
    // that left the process and came back -- holds a slot nothing else will
    // release. Its dst_flag is raised when this copy completes
    // (CkRdmaDeviceRecvHandler), exactly as on the IPC path.
    if (source.device_idx >= 0 && source.event_idx >= 0 && csv_gpu_manager.use_shm) {
      p.op->ipc_device_idx = source.device_idx;
      p.op->ipc_event_idx = source.event_idx;
    }
    // cudaMemcpyDefault, not DeviceToDevice: once load balancing has moved
    // chares between GPUs, the source and destination of a same-process
    // transfer can sit on different devices, and an explicit DeviceToDevice
    // kind is rejected for that pair. Default resolves the direction from the
    // pointers themselves and handles the peer case.
    hapiSubmitMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt, cudaMemcpyDefault, rs);

    break;
  }
  case DeviceRecvKind::Ipc:
    // The slot is released from the completion handler; naming it in the op
    // is what arms that.
    if (source.device_idx >= 0 && source.event_idx >= 0) {
      p.op->ipc_device_idx = source.device_idx;
      p.op->ipc_event_idx = source.event_idx;
    }
    deviceIpcReceive(source, dest, rs, p.src_pe, p.mode);
    break;
  }
  hapiAddCallback(rs, CkCallback(p.done, (void*)p.op));
}

// Issue now if both conditions hold, else park for deviceRecvPendingPoll.
static void deviceRecvTryIssue(DeviceRecvPending&& pending)
{
  pending.src_ready = deviceRecvSourceReady(pending, /*may_query=*/true);
  pending.dst_ready = deviceRecvDestinationReady(pending, /*may_query=*/true);
  if (pending.src_ready && pending.dst_ready) {
    deviceRecvIssue(pending);
    return;
  }
  pending.next_query = CkWallTimer() + kRecvQueryPeriod;
  std::vector<DeviceRecvPending*>& list = *CkpvAccess(device_recv_pending);
  list.push_back(new DeviceRecvPending(std::move(pending)));
  g_recv_parked++;
  if ((long)list.size() > g_recv_parked_max) g_recv_parked_max = (long)list.size();
}

static void deviceRecvPendingPoll(void*)
{
  std::vector<DeviceRecvPending*>& list = *CkpvAccess(device_recv_pending);
  if (list.empty()) return;
  const double now = CkWallTimer();
  size_t kept = 0;
  for (size_t i = 0; i < list.size(); i++) {
    DeviceRecvPending* p = list[i];
    const bool may_query = (now >= p->next_query);
    if (may_query) p->next_query = now + kRecvQueryPeriod;
    if (!p->src_ready) p->src_ready = deviceRecvSourceReady(*p, may_query);
    if (!p->dst_ready) p->dst_ready = deviceRecvDestinationReady(*p, may_query);
    if (p->src_ready && p->dst_ready) {
      deviceRecvIssue(*p);
      delete p;
    } else {
      list[kept++] = p;
    }
  }
  list.resize(kept);
}

/*************** Forward-time repair of a memcpy-prepared payload ***************/
//
// CkRdmaDeviceOnSender prepares a send for a co-resident target as MEMCPY: a
// raw device pointer, no export, no IPC event. That is exact at the moment it
// is decided (the process-wide resident table is consulted), but the message
// is then queued, and if the balancer moves the target out of the process
// before the message is consumed, the location manager forwards it to a home
// that cannot read the pointer. Until now that cost a correction round trip
// (requestDeviceRestage): the new home asked the sender for the bytes again.
//
// The forward happens in the process that owns the source, and the source is
// still alive -- a memcpy send ships its completion callback with the message,
// so nothing has released the buffer -- so the payload can be re-prepared
// here as a first-time direct send would have been: export the source, claim
// an IPC event slot, record the event behind the memcpy event that marks the
// data as produced, and rewrite the descriptor in the message. The receiver
// then takes its ordinary direct path and no correction is requested. Costs
// nothing when nothing moves.
//
// A forward off the physical node is repaired the same way, as a first-time
// RDMA send would have been prepared: the bytes have to cross the network
// anyway, and the sender's side of that is a registered descriptor for the
// live source, which in pool mode is the arena's one registration. The
// receiver then takes its ordinary rget path. Before this the inter-node case
// fell back to the correction round trip -- a NACK, a put, and a
// notification -- through a path nothing else exercised.
//
// The descriptors sit at the front of every device-send message, right after
// their count (see the generated _call_ functions), and have one width in
// every protocol (CmiDeviceBuffer::pup), which is what makes the in-place
// rewrite possible without generated code.
// forwarder -> source PE: a message whose device payload only the source's
// process can re-prepare for the network, with the PE it must reach. The
// envelope follows the header verbatim.
struct DeviceForwardRedirect {
  char header[CmiMsgHeaderSizeBytes];
  int newPe;
  int envSize;
};
extern "C" { int device_forward_redirect_handler; }

// An inter-node forward repair re-registers the sender's live buffer so the
// new home can rget it, and the NIC reads that buffer outside every CUDA
// stream -- so, exactly as in issueDeviceRestagePut, the producing kernel has
// to have FINISHED before the descriptor is published, not merely be ordered
// ahead of it. cudaEventSynchronize bought that by stopping the PE until the
// GPU drained, once per repaired payload.
//
// Park the message behind the producer instead. The repair is re-entrant: a
// descriptor it has already rewritten is marked sender_prepared with
// device_idx -1, which the loop below skips as an RDMA descriptor, so resuming
// simply picks up where the wait interrupted it. Each pass repairs at least
// the descriptor it waited on, so the chain terminates.
struct DeviceForwardRepairCtx
{
  envelope* env;
  int newPe;
  int opts;
};

static void deviceForwardRepairReady(void* arg, void*)
{
  DeviceForwardRepairCtx* ctx = (DeviceForwardRepairCtx*)arg;
  // Resume. A further descriptor may park again, or the pass may now redirect;
  // in both cases something else owns the message and this pass must not
  // deliver it.
  if (CkRdmaDeviceRepairForward(ctx->env, ctx->newPe, ctx->opts) ==
      CkDeviceRepairResult::CallerDelivers)
    CkArrayManagerDeliver(ctx->newPe, EnvToUsr(ctx->env), ctx->opts);
  delete ctx;
}

// True when the producer has already finished and the caller may publish the
// descriptor now. False when the message has been parked behind it: the caller
// owns nothing further and must report the message consumed.
static bool forwardRepairReadyOrPark(envelope* env, int newPe, int opts,
                                     hapiEvent_t ev)
{
  if (ev == NULL) return true;
  // The overwhelmingly common case: the payload was produced long before its
  // target moved, so there is nothing to wait for and nothing to park.
  const cudaError_t q = cudaEventQuery(ev);
  if (q == cudaSuccess) return true;
  if (q != cudaErrorNotReady) {
    hapiCheck(q);   // a real failure: report it where it happened
    return true;
  }
  hapiCheck(hapiStreamWaitEvent(restageWaitStream(), ev, 0));
  DeviceForwardRepairCtx* ctx = new DeviceForwardRepairCtx{env, newPe, opts};
  // hapiAddCallback holds quiescence for the whole gap and delivers to this PE.
  hapiAddCallback(restageWaitStream(), CkCallback(deviceForwardRepairReady, ctx));
  return false;
}

CkDeviceRepairResult CkRdmaDeviceRepairForward(envelope* env, int newPe, int opts) {
  if (!CMI_IS_ZC_DEVICE(env)) return CkDeviceRepairResult::CallerDelivers;
  if (env->getMsgtype() != ForArrayEltMsg) return CkDeviceRepairResult::CallerDelivers;
  if (CmiNodeOf(newPe) == CmiMyNode()) return CkDeviceRepairResult::CallerDelivers;       // still readable as is
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  static const bool dbg = (getenv("CHARM_ZC_RESTAGE_DEBUG") != nullptr);

  if (!CmiPeOnSamePhysicalNode(newPe, CkMyPe())) {
    // Inter-node forward: re-prepare as an RDMA send.
    //
    // Only the source's own process can do that: the registration has to be
    // of the live buffer in the address space that owns it. A forward from
    // another process on the same physical node -- the common case with one
    // PE per process, where every cross-process forward is one -- is bounced
    // to the source PE over shared memory with the destination attached; it
    // repairs and delivers off-node from there. One extra same-node hop, in
    // place of the correction round trip nothing else exercised.
    int redirectTo = -1;
    char* buf = ((CkMarshallMsg*)EnvToUsr(env))->msgBuf;
    PUP::fromMem rd(buf);
    int n = 0;
    rd | n;
    for (int i = 0; i < n; i++) {
      const size_t off = rd.size();
      CkDeviceBuffer b;
      rd | b;
      const size_t width = rd.size() - off;
      // Empty: nothing to re-prepare, in any mode (CkRdmaDeviceOnSender).
      if (b.cnt == 0) continue;
      // Two kinds of descriptor can be repaired here, because in both the
      // sender's live buffer is the payload: the memcpy-prepared one (nothing
      // prepared at all) and the direct-IPC one (the live buffer exported by
      // handle, which means nothing off this node). A staged descriptor's
      // bytes are a copy in this node's comm buffer and keep the correction
      // path; so does anything whose source is not in this process.
      const bool direct_prepared =
          b.sender_prepared && b.ipc_protocol == CmiIpcProtocol::DIRECT &&
          b.device_idx >= 0 &&
          (size_t)b.device_idx < csv_gpu_manager.hapi_ipc_device_infos.size();
      // Already an RDMA descriptor (a first-hop inter-node send being
      // forwarded again): readable from anywhere, nothing to do.
      if (b.sender_prepared && b.device_idx < 0) continue;
      const char* skip = NULL;
      if (b.sender_prepared && !direct_prepared) skip = "staged, left to the correction path";
      else if (b.src_mpi_rank != CmiMyNode()) {
        if (CmiPeOnSamePhysicalNode(b.src_pe, CkMyPe())) { redirectTo = b.src_pe; break; }
        skip = "source on another physical node";
      }
      if (skip) {
        if (dbg) {
          CmiPrintf("[%d] ZC FORWARD-SKIP src=%p cnt=%zu srcPe=%d -> PE %d: %s (inter-node)\n",
                    CkMyPe(), b.ptr, (size_t)b.cnt, b.src_pe, newPe, skip);
          fflush(stdout);
        }
        continue;
      }
      // The NIC reads the source outside any CUDA stream, so the production
      // has to be complete, not merely ordered behind. The memcpy prepare marks
      // it with memcpy_event; the direct prepare recorded its slot's src event
      // on the producing stream. A null memcpy event means the send blocked at
      // send time and the data is already there.
      if (direct_prepared) {
        hapi_ipc_device_info& info = csv_gpu_manager.hapi_ipc_device_infos[b.device_idx];
        if (!forwardRepairReadyOrPark(env, newPe, opts, info.src_event_pool[b.event_idx]))
          return CkDeviceRepairResult::Parked;
        // Retire the IPC slot the direct prepare claimed: the receiver it was
        // claimed for is on another node and will never set its flag. A direct
        // slot holds no block and its producer is complete (checked above), so
        // the flag alone lets the owning PE's reclaim scan take it back.
        hapi_ipc_event_shared* slot =
            (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
                + csv_gpu_manager.shm_chunk_size * b.device_idx
                + sizeof(hapiIpcMemHandle_t)) + b.event_idx;
        slot->dst_flag.store(true, std::memory_order_release);
      } else if (b.memcpy_event != NULL) {
        if (!forwardRepairReadyOrPark(env, newPe, opts, (hapiEvent_t)b.memcpy_event))
          return CkDeviceRepairResult::Parked;
      }
      // The source is still owned by the sending element: a memcpy send ships
      // its completion callback with the message, and the receiver fires it
      // once the rget has landed, exactly as it would for a first-time RDMA
      // send. No owner is attributed here (the forwarding context is not the
      // sender); in pool mode the registration is the arena's anyway.
      b.lci_ncpy_buffer = acquireDeviceRegistration(b.ptr, b.cnt, NULL);
      b.sender_prepared = true;
      b.device_idx = -1;                              // not IPC-exported: the rget path
      b.event_idx = -1;
      b.comm_offset = 0;
      b.ipc_protocol = CmiIpcProtocol::NONE;
      b.dest_pe = newPe;
      b.dest_mpi_rank = CmiNodeOf(newPe);
      PUP::toMem wr(buf + off);
      wr | b;
      if (wr.size() != width)
        CkAbort("CkRdmaDeviceRepairForward: descriptor width changed (%zu -> %zu)",
                width, wr.size());
      csv_gpu_manager.ipc_forward_repairs.fetch_add(1, std::memory_order_relaxed);
      if (dbg) {
        CmiPrintf("[%d] ZC FORWARD-REPAIR src=%p cnt=%zu -> PE %d (rdma)\n",
                  CkMyPe(), b.ptr, (size_t)b.cnt, newPe);
        fflush(stdout);
      }
    }
    if (redirectTo < 0) return CkDeviceRepairResult::CallerDelivers;

    // Every descriptor of one message comes from one element, so one source
    // PE serves the whole message. Ship the envelope to it verbatim.
    CkPackMessage(&env);
    const int envSize = env->getTotalsize();
    DeviceForwardRedirect* r = (DeviceForwardRedirect*)CmiAlloc(
        sizeof(DeviceForwardRedirect) + envSize);
    CmiEnforce(r);
    r->newPe = newPe;
    r->envSize = envSize;
    memcpy((char*)r + sizeof(DeviceForwardRedirect), env, envSize);
    if (dbg) {
      CmiPrintf("[%d] ZC FORWARD-REDIRECT %d bytes -> source PE %d, for PE %d\n",
                CkMyPe(), envSize, redirectTo, newPe);
      fflush(stdout);
    }
    CmiFree(env);
    QdCreate(1);
    CmiSetHandler(r, device_forward_redirect_handler);
    CmiSyncSendAndFree(redirectTo, sizeof(DeviceForwardRedirect) + envSize, (char*)r);
    csv_gpu_manager.ipc_forward_repairs.fetch_add(1, std::memory_order_relaxed);
    return CkDeviceRepairResult::Redirected;
  }

  if (!csv_gpu_manager.use_shm || !hapiIpcUseDirect()) return CkDeviceRepairResult::CallerDelivers;
  auto dmit = csv_gpu_manager.device_map.find(CkMyPe());
  if (dmit == csv_gpu_manager.device_map.end()) return CkDeviceRepairResult::CallerDelivers;
  DeviceManager* dm = dmit->second;

  char* buf = ((CkMarshallMsg*)EnvToUsr(env))->msgBuf;
  PUP::fromMem rd(buf);
  int n = 0;
  rd | n;
  for (int i = 0; i < n; i++) {
    const size_t off = rd.size();
    CkDeviceBuffer b;
    rd | b;
    const size_t width = rd.size() - off;
    if (b.cnt == 0) continue;                        // empty: nothing to re-prepare
    if (b.sender_prepared) continue;                 // IPC-prepared: readable node-wide already
    const char* skip = NULL;
    auto srcit = csv_gpu_manager.device_map.find(b.src_pe);
    hapiIpcMemHandle_t handle;
    size_t export_offset = 0;
    void* export_base = NULL;
    if (b.src_mpi_rank != CmiMyNode()) skip = "source not in this process";
    else if (srcit == csv_gpu_manager.device_map.end() || srcit->second != dm)
      skip = "source on another device";
    else if (!hapiIpcExportBuffer(b.ptr, &handle, &export_offset, &export_base))
      skip = "export failed";
    if (skip) {
      if (dbg) {
        CmiPrintf("[%d] ZC FORWARD-SKIP src=%p cnt=%zu srcPe=%d -> PE %d: %s\n",
                  CkMyPe(), b.ptr, (size_t)b.cnt, b.src_pe, newPe, skip);
        fflush(stdout);
      }
      continue;                                      // left to the correction protocol
    }
    const int dev = CpvAccess(my_device_id);
    void* unused = NULL;
    int ev = -1;
    acquireIpcSendSlot(dm, dev, /*is_lb_buffer=*/false, /*direct=*/true,
                       b.ptr, b.cnt, &unused, &ev);
    // The receiver waits on this event before it reads. Order it behind the
    // memcpy event the send recorded on the producing stream (the receiver's
    // memcpy branch would have waited on that same event).
    if (b.memcpy_event != NULL)
      hapiCheck(hapiStreamWaitEvent(hapiStreamPerThread,
                                    (hapiEvent_t)b.memcpy_event, 0));
    const int device_idx =
        csv_gpu_manager.device_count * CmiMyNodeRankLocal() + dev;
    hapi_ipc_device_info& info = csv_gpu_manager.hapi_ipc_device_infos[device_idx];
    hapiCheck(hapiEventRecord(info.src_event_pool[ev], hapiStreamPerThread));
    ipcPublishSrcReady(device_idx, ev, hapiStreamPerThread);

    b.ipc_protocol = CmiIpcProtocol::DIRECT;
    b.ipc_handle = handle;
    b.ipc_offset = export_offset;
    b.ipc_base = export_base;
    b.device_idx = device_idx;
    b.event_idx = ev;
    b.comm_offset = 0;
    b.sender_prepared = true;
    b.dest_pe = newPe;
    b.dest_mpi_rank = CmiNodeOf(newPe);
    PUP::toMem wr(buf + off);
    wr | b;
    if (wr.size() != width)
      CkAbort("CkRdmaDeviceRepairForward: descriptor width changed (%zu -> %zu)",
              width, wr.size());
    csv_gpu_manager.ipc_forward_repairs.fetch_add(1, std::memory_order_relaxed);
    if (getenv("CHARM_ZC_RESTAGE_DEBUG") != nullptr) {
      CmiPrintf("[%d] ZC FORWARD-REPAIR src=%p cnt=%zu -> PE %d (direct, ev %d)\n",
                CkMyPe(), b.ptr, (size_t)b.cnt, newPe, ev);
      fflush(stdout);
    }
  }
  return CkDeviceRepairResult::CallerDelivers;
}

// Source PE: a forward bounced here because only this process can re-prepare
// its payload for the network. Repair in place -- the source is in this
// process now, so the inter-node branch above takes it -- and deliver to the
// PE the forwarder named. The descriptor is an RDMA one from here on, so any
// further forward needs no repair at all.
extern "C" void* device_forward_redirect_bridge(void* arg)
{
  QdProcess(1);
  DeviceForwardRedirect* r = (DeviceForwardRedirect*)arg;
  envelope* env = (envelope*)CmiAlloc(r->envSize);
  CmiEnforce(env);
  memcpy(env, (char*)r + sizeof(DeviceForwardRedirect), r->envSize);
  const int newPe = r->newPe;
  CmiFree(r);
  CkUnpackMessage(&env);
  // A second REDIRECT is fatal: this is already the source process, so there is
  // nowhere left to bounce it. A PARK is not -- the payload's producer simply
  // has not finished yet, and the callback that resumes the repair delivers it.
  // Collapsing the two is what aborted the async arm at the first parked
  // redirect, since every park read as a bounce.
  switch (CkRdmaDeviceRepairForward(env, newPe)) {
    case CkDeviceRepairResult::Redirected:
      CkAbort("[%d] device forward redirect bounced again: the source process "
              "could not repair its own payload", CkMyPe());
      break;
    case CkDeviceRepairResult::Parked:
      return NULL;   // deviceForwardRepairReady delivers it
    case CkDeviceRepairResult::CallerDelivers:
      break;
  }
  CkArrayManagerDeliver(newPe, EnvToUsr(env), 0);
  return NULL;
}

/*************** Migration-mismatch payload correction (NACK + retransmit) ***************/
//
// A device send picks its transfer mode from where the sender believes the
// target lives, but that only becomes true when the message lands. If the
// target migrates across a process boundary in between, the sender will have
// chosen a plain same-process memcpy and prepared nothing readable from
// anywhere else, while the message itself is routed to the target's new home
// perfectly well by the location manager.
//
// So only the payload is missing, not the message. The receiver detects the
// disagreement, defers that buffer's completion, and asks the sender to send
// the bytes again by a route this process can read. Previously this aborted
// (IPC) or issued an unregistered rdmaGet that surfaced as an LCI "Message too
// long" assert far from the cause (inter-node).
//
// Two disagreements reach here, and they differ in what the sender still owns:
//   sender_prepared false -- a plain same-process memcpy was chosen, so nothing
//                 was exported, but the chare's source buffer is still live: a
//                 memcpy send ships its completion callback to the receiver
//                 rather than firing it, so nobody has released that buffer.
//   staged, target off-node -- the sender did export, into its comm buffer, and
//                 a staged send fires its own callback as soon as that copy
//                 lands. The chare's buffer is free from that moment and may
//                 already hold the next iteration, so the staged block is the
//                 only copy that is still this transfer's payload. The request
//                 carries device_idx and comm_offset back for it; those are
//                 node-local, so they mean nothing where the request is built
//                 and are exact on the PE it is sent to.
//
// The sender's stale location cache needs no help from this protocol: the
// forwarded message took more than one hop, so CkArray::deliverToElement calls
// CkLocMgr::multiHop, which pushes the corrected entry back to the sender.
//
// Correction is cheap because it reuses each transport's existing direction:
//   same node  -- the sender stages into its comm buffer and returns the IPC
//                 metadata; the receiver pulls, as it does for any staged send,
//                 through the once-per-device mapping it already holds.
//   inter-node -- the sender registers its source and RDMA-puts straight into
//                 the descriptor the receiver registered for it, then sends a
//                 one-word notification, because an RDMA write raises its
//                 completion on the initiator and leaves the target unaware.
// Putting into an app buffer costs nothing extra inter-node, where the receiver
// registers that buffer on every receive anyway. Same-node it would mean a cold
// cuIpcOpenMemHandle, charged to the one transfer least able to afford it: a
// corrected payload has already taken the extra forwarding hops and waited out
// a request round trip before any of its bytes move. Not that the mapping would
// be wasted under DIRECT -- that protocol is chosen precisely because the
// application reuses its buffers, so later sends to the chare's new location
// would hit it --
// but it does not have to be opened here. The forwarding above repairs the
// sender's cache, so the next ordinary send opens it instead, off the critical
// path of a transfer that is already late. Corrections also come in a burst
// right after a load balancing step, which is the worst moment in the run to
// add hundreds of microseconds apiece.

struct DeviceRestageReq {          // receiver -> sender
  char header[CmiMsgHeaderSizeBytes];
  void* dest_op;                   // receiver's DeviceRdmaOp*, opaque here
  int dest_pe;
  const void* src_ptr;             // the sender's live source, if it still owns one
  bool src_staged;                 // true: read the staged block below instead
  size_t src_comm_offset;
  // What the source read below must be ordered behind. The original send
  // established this ordering and the correction has to re-establish it: the
  // payload is produced by the application's stream (a packing kernel, or the
  // staging copy), and the re-read runs on hapiStreamPerThread, which is
  // ordered against neither.
  const void* src_event;           // memcpy path: the sender's recorded event
  int src_event_idx;               // staged path: the sender's IPC event slot
  size_t cnt;
  bool inter_node;
  CmiNcpyBuffer dest_ncpy;         // inter-node only: registered destination
};

struct DeviceRestageMeta {         // sender -> receiver, same node
  char header[CmiMsgHeaderSizeBytes];
  void* dest_op;
  int device_idx;
  int event_idx;
  size_t comm_offset;              // STAGED: where in the sender's comm buffer
  size_t cnt;
  // DIRECT (+gpupool): the sender's source allocation, exported in place.
  CmiIpcProtocol protocol;
  hapiIpcMemHandle_t ipc_handle;
  void* ipc_base;
  size_t ipc_offset;
};

struct DeviceRestagePutDone {      // sender -> receiver, inter-node put
  char header[CmiMsgHeaderSizeBytes];
  void* dest_op;                   // receiver's DeviceRdmaOp*
};

extern "C" {
  int device_restage_req_handler;
  int device_restage_meta_handler;
  int device_restage_put_done_handler;
}

// Sender side: the write has completed locally, and nothing on the target has
// been told, because an RDMA write raises no completion there. So tell it. This
// is the same shape as LCI's own rendezvous protocol, which posts its puts and
// then, from the local write completion, sends a FIN carrying the receiver's
// context pointer -- ordering the notification behind the payload by waiting
// for the write rather than by trusting the network to order two operations.
static void notifyDeviceRestagePut(int dest_pe, void* dest_op)
{
  auto* m = (DeviceRestagePutDone*)CmiAlloc(sizeof(DeviceRestagePutDone));
  CmiEnforce(m);
  m->dest_op = dest_op;
  // No QdCreate: the count raised before the put is still outstanding and is
  // handed to this message, so quiescence stays blocked across the whole
  // correction rather than reopening between the write and the notification.
  CmiSetHandler(m, device_restage_put_done_handler);
  CmiSyncSendAndFree(dest_pe, sizeof(DeviceRestagePutDone), (char*)m);
}

// Receiver side, inter-node put: the payload is already in the destination
// buffer, so the op only has to be resolved.
extern "C" void* device_restage_put_done_bridge(void* arg)
{
  QdProcess(1);
  auto* m = (DeviceRestagePutDone*)arg;
  // No stand-down release here: it happens when the completion message is
  // freed (CkRdmaDeviceMsgFreed), the same as an uncorrected receive, so a
  // corrected op must not release a second time.
  CkRdmaDeviceRecvHandler(m->dest_op, NULL);
  CmiFree(m);
  return NULL;
}

// Receiver side: defer this buffer and ask the sender to retransmit it.
// What the receiver saw in the descriptor when it asked for a correction;
// only read by the CHARM_ZC_RESTAGE_DEBUG print.
static thread_local bool src_prepared_dbg = false;
static thread_local int src_proto_dbg = 0;
static thread_local int src_dev_idx_dbg = -1;

// An inter-node restage put may only be issued once the kernel that produced
// the source has actually finished: the NIC reads that buffer outside every
// CUDA stream, so a stream wait -- which orders CUDA work against CUDA work --
// cannot hold it back, and a put issued early ships whatever the buffer held
// before. That ordering is required. Blocking the PE on it is not, and
// cudaEventSynchronize did exactly that: the PE stopped scheduling until the
// GPU drained, once per correction, in an LB step that makes hundreds of them.
//
// The wait is enqueued on this PE's restage stream instead, and the put issued
// from that stream's HAPI callback. hapiAddCallback delivers it to the PE that
// registered it (a converse message on light_cb_idx_), so the put is still
// issued by this PE, in the same order relative to its own work -- the PE just
// keeps launching and draining in the meantime. Distinct corrections are
// independent: each carries its own dest_op and its own registered landing
// buffer, and completion is resolved per op by the notification, so letting
// them overlap changes nothing about how any one of them resolves.
struct DeviceRestagePutCtx
{
  DeviceRestageReq* req;
  const void* src_ptr;
};

static void issueDeviceRestagePut(DeviceRestageReq* req, const void* src_ptr)
{
  // Put with notification. The NACK already carried the receiver's registered
  // landing buffer, so the payload goes straight there: one RDMA write from
  // the side that already owns the source registration.
  //
  // CMK_DEVICE_RESTAGE_PUT marks the operation so that its completion --
  // which a write raises on the initiator, here the sender -- is turned into
  // a notification to the target instead of being resolved locally. Without
  // that the receiver is never told, and the run stalls with the bytes
  // already in place and zero load balancing steps.
  CmiSetDirectNcpyAckHandler(CkRdmaDeviceRecvHandler);
  CmiNcpyBuffer src_ncpy(src_ptr, req->cnt);
  NcpyOperationInfo* info = src_ncpy.createNcpyOpInfo(
      src_ncpy, req->dest_ncpy, /*ackSize=*/0, NULL, NULL, /*rootNode=*/-1,
      CMK_DEVICE_RESTAGE_PUT, NULL);
  QdCreate(1);   // released by device_restage_put_done_bridge
  CmiIssueRput(info);
}

// The producer has finished; issue the put and retire the request that was
// held across the wait. Runs on the PE that enqueued the wait.
static void deviceRestagePutReady(void* arg, void*)
{
  DeviceRestagePutCtx* ctx = (DeviceRestagePutCtx*)arg;
  issueDeviceRestagePut(ctx->req, ctx->src_ptr);
  CmiFree(ctx->req);
  delete ctx;
}

static void requestDeviceRestage(int srcPe, void* dest_op, const void* src_ptr,
                                 CkGroupID dest_aid, CmiUInt8 dest_id,
                                 bool src_staged, size_t src_comm_offset,
                                 const void* src_event, int src_event_idx,
                                 size_t cnt, void* dest_ptr, size_t dest_cnt,
                                 bool inter_node)
{
  DeviceRestageReq* req = (DeviceRestageReq*)CmiAlloc(sizeof(DeviceRestageReq));
  CmiEnforce(req);
  req->dest_op = dest_op;
  req->dest_pe = CkMyPe();
  // Stand this element's migration down until the correction lands: the
  // destination below is captured now and written a round trip later.
  // (The stand-down for this op was already taken when its transfers were
  // issued -- every receive is counted now, deferred or not -- so taking
  // another here would need a second release.)
  req->src_ptr = src_ptr;
  // A staged send hands the bytes to the comm buffer and releases the chare's
  // own source buffer, so src_ptr is only trustworthy when the sender staged
  // nothing. Carry the staged location back too: comm_offset is node-local,
  // meaningless where this request is built, and exact on the PE it is being
  // sent to -- the one that staged it.
  //
  // Keyed on the protocol, not on device_idx: a DIRECT send sets device_idx
  // exactly as a staged one does but stages nothing, leaving comm_offset at 0.
  // Reading that would hand back the base of the comm buffer -- some other
  // transfer's bytes.
  req->src_staged = src_staged;
  req->src_comm_offset = src_comm_offset;
  req->src_event = src_event;
  req->src_event_idx = src_event_idx;
  req->cnt = cnt;
  req->inter_node = inter_node;
  if (inter_node) {
    // Register the landing buffer here and hand the sender its descriptor, so
    // the put can be addressed straight at it. The buffer also carries this
    // receive's DeviceRdmaOp, which rides along in the operation info and comes
    // back in the sender's notification as the name of what to resolve.
    req->dest_ncpy = CmiNcpyBuffer(dest_ptr, dest_cnt, dest_op);
  }
  // CHARM_ZC_RESTAGE_DEBUG: a pass on the migration paths only means something
  // if this path actually ran, so make it countable rather than inferred.
  static const bool restage_debug = (getenv("CHARM_ZC_RESTAGE_DEBUG") != nullptr);
  if (restage_debug) {
    static std::atomic<unsigned> n{0};
    CmiPrintf("[%d] ZC RESTAGE #%u srcPe=%d cnt=%zu id=%llu %s "
              "(sender_prepared=%d proto=%d dev_idx=%d src_staged=%d)\n", CkMyPe(),
              n.fetch_add(1, std::memory_order_relaxed) + 1, srcPe, cnt,
              (unsigned long long)dest_id,
              inter_node ? "inter-node put" : "same-node stage",
              (int)src_prepared_dbg, (int)src_proto_dbg, src_dev_idx_dbg,
              (int)src_staged);
    fflush(stdout);
  }

  // Balances the QdProcess in the bridge: the transfer is still outstanding
  // across the round trip, and quiescence must not fire in the middle of it.
  QdCreate(1);
  CmiSetHandler(req, device_restage_req_handler);
  CmiSyncSendAndFree(srcPe, sizeof(DeviceRestageReq), (char*)req);
}

extern "C" void* device_restage_req_bridge(void* arg)
{
  QdProcess(1);
  DeviceRestageReq* req = (DeviceRestageReq*)arg;
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Read back whatever this PE still owns. A staged send copied the payload
  // into the comm buffer and let the chare reuse its own buffer, so where one
  // was staged the comm buffer holds the only copy that is still the payload
  // this transfer promised; src_ptr may already carry the next iteration.
  const void* src_ptr = req->src_ptr;
  if (req->src_staged && csv_gpu_manager.use_shm) {
    DeviceManager* src_dm = csv_gpu_manager.device_map[CkMyPe()];
    src_ptr = (const char*)src_dm->comm_buffer->base_ptr + req->src_comm_offset;
  }

  // What produced these bytes has not necessarily run yet. The original send
  // never read the source itself -- it recorded an event on the application's
  // stream and let the receiver wait on it -- so the payload is only ordered
  // behind the packing kernel (or the staging copy) through that event. The
  // re-read below runs on hapiStreamPerThread, which is ordered against neither
  // the application's stream nor anything else here, so without re-imposing it
  // the correction can copy a slice the kernel has not filled and ship the
  // previous contents of a buffer the application rotates. That lands as a
  // plausible but wrong ghost in one element, far from here.
  //
  // A null event on the unstaged path means the sender blocked at send time
  // (CHARM_ZC_MEMCPY_SYNC, or no event was free), so the data is already there.
  hapiEvent_t src_ready = NULL;
  bool src_needs_full_sync = false;
  if (req->src_staged && csv_gpu_manager.use_shm) {
    if (req->src_event_idx >= 0) {
      const int src_dev_idx = csv_gpu_manager.device_count * CmiMyNodeRankLocal()
                            + CpvAccess(my_device_id);
      src_ready = csv_gpu_manager.hapi_ipc_device_infos[src_dev_idx]
                      .src_event_pool[req->src_event_idx];
    } else {
      src_needs_full_sync = true;
    }
  } else if (req->src_event != NULL) {
    src_ready = (hapiEvent_t)req->src_event;
  }

  if (req->inter_node) {
    // See issueDeviceRestagePut: the production has to be COMPLETE before the
    // write is issued, not merely ordered behind it, because the NIC reads
    // outside every stream. Observe that without stopping the PE -- enqueue the
    // wait and issue the put from the stream's callback. hapiAddCallback holds
    // quiescence for the whole gap, so nothing can complete out from under it.
    if (src_ready) {
      hapiCheck(hapiStreamWaitEvent(restageWaitStream(), src_ready, 0));
      DeviceRestagePutCtx* ctx = new DeviceRestagePutCtx{req, src_ptr};
      hapiAddCallback(restageWaitStream(), CkCallback(deviceRestagePutReady, ctx));
      return NULL;  // ctx owns req until the callback issues the put
    }
    // No event to wait on, and the staged path means something produced these
    // bytes on a stream this PE can no longer name -- the sender found no free
    // IPC event slot. There is nothing to enqueue a wait against, so this one
    // case still drains the device. It fires only under slot exhaustion; the
    // fix for it is a free slot, not a different wait.
    if (src_needs_full_sync) { hapiSubmitDrain(); hapiCheck(cudaDeviceSynchronize()); }
    // Otherwise the sender blocked at send time (CHARM_ZC_MEMCPY_SYNC, or no
    // event was free on the unstaged path), so the bytes are already there.
    issueDeviceRestagePut(req, src_ptr);
  } else if (hapiDevPoolOn()) {
    // +gpupool: nothing to stage into, and no need. The only same-node
    // correction left is the memcpy-chosen send whose target moved to another
    // process, and the sender still owns that source. Export it and let the
    // receiver read it in place, exactly as a first-time direct send would.
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
    const int cpv_my_device_id = CpvAccess(my_device_id);
    hapiIpcMemHandle_t handle;
    size_t export_offset = 0;
    void* export_base = NULL;
    if (!hapiIpcExportBuffer(src_ptr, &handle, &export_offset, &export_base)) {
      CkAbort("PE %d: restage to PE %d could not export the %zu-byte source at "
              "%p for direct CUDA IPC (%s). Under +gpupool every device buffer "
              "sent must be cudaMalloc-backed.",
              CkMyPe(), req->dest_pe, req->cnt, src_ptr,
              hapiIpcLastImportErrorName());
    }
    void* unused = NULL;
    int event_idx = -1;
    acquireIpcSendSlot(dm, cpv_my_device_id, /*is_lb_buffer=*/false,
                       /*direct=*/true, src_ptr, req->cnt, &unused, &event_idx);
    // The receiver waits on this event before it reads; it has to sit behind
    // whatever produced the source, which src_ready marks.
    if (src_ready)
      hapiCheck(hapiStreamWaitEvent(hapiStreamPerThread, src_ready, 0));
    else if (src_needs_full_sync)
      { hapiSubmitDrain(); hapiCheck(cudaDeviceSynchronize()); }
    const int device_idx =
        csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
    hapi_ipc_device_info& my_device_info =
        csv_gpu_manager.hapi_ipc_device_infos[device_idx];
    hapiCheck(hapiEventRecord(my_device_info.src_event_pool[event_idx],
                              hapiStreamPerThread));
    ipcPublishSrcReady(device_idx, event_idx, hapiStreamPerThread);

    DeviceRestageMeta* m = (DeviceRestageMeta*)CmiAlloc(sizeof(DeviceRestageMeta));
    CmiEnforce(m);
    m->dest_op = req->dest_op;
    m->device_idx = device_idx;
    m->event_idx = event_idx;
    m->comm_offset = 0;
    m->cnt = req->cnt;
    m->protocol = CmiIpcProtocol::DIRECT;
    m->ipc_handle = handle;
    m->ipc_base = export_base;
    m->ipc_offset = export_offset;
    QdCreate(1);
    CmiSetHandler(m, device_restage_meta_handler);
    CmiSyncSendAndFree(req->dest_pe, sizeof(DeviceRestageMeta), (char*)m);
  } else {
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
    const int cpv_my_device_id = CpvAccess(my_device_id);
    void* staged = NULL;
    int event_idx = -1;
    acquireIpcSendSlot(dm, cpv_my_device_id, /*is_lb_buffer=*/false,
                       /*direct=*/false, src_ptr, req->cnt, &staged,
                       &event_idx);
    if (src_ready)
      hapiCheck(hapiStreamWaitEvent(hapiStreamPerThread, src_ready, 0));
    else if (src_needs_full_sync)
      { hapiSubmitDrain(); hapiCheck(cudaDeviceSynchronize()); }

    hapiCheck(hapiMemcpyAsync(staged, src_ptr, req->cnt,
                              hapiMemcpyDeviceToDevice, hapiStreamPerThread));

    const int device_idx =
        csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
    hapi_ipc_device_info& my_device_info =
        csv_gpu_manager.hapi_ipc_device_infos[device_idx];
    hapiCheck(hapiEventRecord(my_device_info.src_event_pool[event_idx],
                              hapiStreamPerThread));
    ipcPublishSrcReady(device_idx, event_idx, hapiStreamPerThread);

    // Retire the ORIGINAL send's slot, now that its bytes have been re-read.
    //
    // The deferral could not do this. Where the original send staged, the comm
    // buffer holds the only copy of the payload still owed -- the chare's own
    // buffer may already carry the next iteration -- so the block has to stay
    // intact until the memcpy above has read it. Releasing at the deferral site
    // would hand the block back while it was still the source.
    //
    // Nothing released it afterwards either, so every correction burned one
    // event slot of the per-PE slice for the life of the run, and one comm
    // block with it. Single-digit correction counts kept that invisible until
    // unmappable direct sends started routing through here in volume, at which
    // point the pic2d repro needed +gpuipceventpool 2048 where 16 had done.
    //
    // The original slot's block, if it has one, was written by the sender's
    // own staging copy, whose completion its src_ready publication (a gate on
    // reclaim) attests; the failed delivery never read it. The flag alone
    // releases it.
    if (req->src_event_idx >= 0) {
      hapi_ipc_event_shared* orig_shared =
          (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
              + csv_gpu_manager.shm_chunk_size * device_idx
              + sizeof(hapiIpcMemHandle_t)) + req->src_event_idx;
      orig_shared->dst_flag.store(true, std::memory_order_release);
    }

    // Deliberately no completion callback for the source buffer here. The
    // sender chose memcpy, so it shipped the real CkCallback to the receiver
    // rather than firing it itself as a staged send would; the receiver still
    // holds it in save_op.src_cb and fires it on completion. Firing it here as
    // well would release the application's buffer twice.
    DeviceRestageMeta* m = (DeviceRestageMeta*)CmiAlloc(sizeof(DeviceRestageMeta));
    CmiEnforce(m);
    m->dest_op = req->dest_op;
    m->device_idx = device_idx;
    m->event_idx = event_idx;
    m->comm_offset = (char*)staged - (char*)dm->comm_buffer->base_ptr;
    m->cnt = req->cnt;
    m->protocol = CmiIpcProtocol::STAGED;
    m->ipc_base = NULL;
    m->ipc_offset = 0;
    QdCreate(1);
    CmiSetHandler(m, device_restage_meta_handler);
    CmiSyncSendAndFree(req->dest_pe, sizeof(DeviceRestageMeta), (char*)m);
  }
  CmiFree(req);
  return NULL;
}

// Completion of a same-node corrected receive: the restaged bytes are now in
// the destination buffer. Runs on the destination PE via hapiAddCallback, the
// same context deviceSendReleaseFn already uses. No stand-down release here:
// it happens when the completion message is freed (CkRdmaDeviceMsgFreed), the
// same as an uncorrected receive, so a corrected op must not release twice.
static void deviceRestageRecvDone(void* data, void* msg)
{
  CkRdmaDeviceRecvHandler(data, msg);
}

// Receiver side, same node: the sender has staged the payload, so run the
// ordinary staged-IPC receive against the metadata it just sent.
extern "C" void* device_restage_meta_bridge(void* arg)
{
  QdProcess(1);
  DeviceRestageMeta* m = (DeviceRestageMeta*)arg;
  DeviceRdmaOp* op = (DeviceRdmaOp*)m->dest_op;
  // The stand-down taken at requestDeviceRestage is NOT lifted here. The
  // copy below has not even been issued yet: releasing the element now opens
  // a stream-latency window in which it can emigrate, pack its grid off
  // migration_stream -- which is not ordered against recv_stream -- and leave
  // with the pre-correction bytes while the copy lands in the buffer the
  // departed element abandoned (or in freed device memory). The inter-node
  // path may resolve on arrival of the put-done because the payload is
  // already in place; here that is only true once recv_stream has executed
  // the copy, so the note moves to the completion callback.

  CkDeviceBuffer source;
  source.ptr = NULL;                 // never dereferenced on a staged receive
  source.cnt = m->cnt;
  source.device_idx = m->device_idx;
  source.event_idx = m->event_idx;
  source.comm_offset = m->comm_offset;
  source.ipc_protocol = m->protocol;
  source.ipc_handle = m->ipc_handle;
  source.ipc_base = m->ipc_base;
  source.ipc_offset = m->ipc_offset;
  source.sender_prepared = true;

  CkDeviceBuffer dest(op->dest_ptr, op->size);

  // The same gate as a first-time receive: the retransmit's producer (the
  // sender's staging copy, or its original kernels for a direct export) and
  // the destination condition the original post recorded in the op.
  DeviceRecvPending pr;
  pr.op = op;
  pr.source = source;
  pr.dest = dest;
  pr.mode = CkNcpyModeDevice::IPC;
  pr.src_pe = op->src_pe;
  pr.kind = DeviceRecvKind::Ipc;
  pr.done = deviceRestageRecvDone;
  deviceRecvTryIssue(std::move(pr));

  CmiFree(m);
  return NULL;
}

// ---- Stalled-receive watchdog (CHARM_ZC_STALL_SECS) -----------------------
//
// A device receive releases its message only on counter == n_ops, so a single
// op that never completes wedges that message for good: the entry method never
// runs, the element never advances, and every PE ends up waiting at the load
// balancing barrier with no error anywhere. That is what every stall this code
// has produced looks like from the outside, and it is invisible to a debugger
// attached after the fact -- the stack just shows an idle scheduler.
//
// So make the receive say so itself. Every in-flight DeviceRdmaInfo is
// registered here with the time it was posted; once a second, anything older
// than the threshold is printed with the ops it is still waiting on. A stall
// then names the op that never completed -- which buffer, from which PE, and
// whether it was deferred for correction -- on whatever run happens to hit it,
// with nothing to attach and nothing to reproduce on demand.

struct DeviceRecvWatch {
  DeviceRdmaInfo* info;
  double posted;
  int numops;
  std::vector<int> src_pe;
  std::vector<size_t> size;
  std::vector<char> deferred;   // asked the sender to send it again
  // IPC identity of each op's transfer, when it went through the shm path
  // (-1 otherwise): lets the scan query the actual CUDA events of a stuck op
  // and say WHICH side's stream never fired -- src_event pending means the
  // sender's staging copy never executed; src done with dst pending means the
  // receiver's copy is parked behind something else on its stream. Slots are
  // only reclaimed after dst_event completes, so for a genuinely stuck op the
  // indices still name the live pair.
  std::vector<int> dev_idx;
  std::vector<int> ev_idx;
  bool reported;
};

CkpvDeclare(std::vector<DeviceRecvWatch>*, device_recv_watch);

static double deviceStallSecs()
{
  static const double s = []() {
    const char* e = getenv("CHARM_ZC_STALL_SECS");
    return e ? atof(e) : 0.0;   // 0 disables
  }();
  return s;
}

// Drop a resolved receive's watch entries before its DeviceRdmaInfo is freed.
// The scan otherwise reads the freed struct on its next pass -- observed as
// stall reports with garbage n_ops in the billions. Resolve and scan run on
// the same PE's scheduler, so erasing here fully closes the race.
static void deviceRecvWatchDrop(DeviceRdmaInfo* info)
{
  auto* w = CkpvAccess(device_recv_watch);
  if (w == NULL) return;
  for (auto it = w->begin(); it != w->end();) {
    if (it->info == info) it = w->erase(it);
    else ++it;
  }
}

static void deviceStallScan(void*, double)
{
  auto* w = CkpvAccess(device_recv_watch);
  if (w == NULL) return;
  const double now = CkWallTimer();
  const double limit = deviceStallSecs();
  for (auto it = w->begin(); it != w->end();) {
    if (it->info->counter >= it->info->n_ops) {
      it = w->erase(it);
      continue;
    }
    if (!it->reported && now - it->posted > limit) {
      CmiPrintf("[%d] ZC STALL: receive stuck %.0fs, %d of %d ops complete\n",
                CkMyPe(), now - it->posted, it->info->counter, it->info->n_ops);
      GPUManager& gm = CsvAccess(gpu_manager);
      for (int i = 0; i < it->numops; i++) {
        char ev_state[96] = "";
        const int di = it->dev_idx[i], ei = it->ev_idx[i];
        if (gm.use_shm && di >= 0 && ei >= 0 &&
            (size_t)di < gm.hapi_ipc_device_infos.size()) {
          hapi_ipc_device_info& dinfo = gm.hapi_ipc_device_infos[di];
          if ((size_t)ei < dinfo.src_event_pool.size()) {
            const bool src_done =
                (hapiEventQuery(dinfo.src_event_pool[ei]) == hapiSuccess);
            hapi_ipc_event_shared* sh =
                (hapi_ipc_event_shared*)((char*)gm.shm_ptr
                    + gm.shm_chunk_size * di + sizeof(hapiIpcMemHandle_t)) + ei;
            snprintf(ev_state, sizeof(ev_state),
                     " dev=%d ev=%d src_ev=%s src_ready=%d dst_done=%d", di, ei,
                     src_done ? "DONE" : "PENDING",
                     (int)sh->src_ready.load(std::memory_order_relaxed),
                     (int)sh->dst_flag.load(std::memory_order_relaxed));
          }
        }
        CmiPrintf("[%d]   op %d: srcPe=%d bytes=%zu %s%s\n", CkMyPe(), i,
                  it->src_pe[i], it->size[i],
                  it->deferred[i] ? "DEFERRED for correction" : "ordinary",
                  ev_state);
      }
      fflush(stdout);
      it->reported = true;
    }
    ++it;
  }
}

void CkRdmaDeviceStallWatchInit()
{
  CkpvInitialize(std::vector<DeviceRecvWatch>*, device_recv_watch);
  CkpvAccess(device_recv_watch) = NULL;
  if (deviceStallSecs() <= 0.0) return;
  CkpvAccess(device_recv_watch) = new std::vector<DeviceRecvWatch>();
  CcdCallOnConditionKeep(CcdPERIODIC_1second, (CcdCondFn)deviceStallScan, NULL);
}

void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs) {
  // The element every op of this message is addressed to, or NULL for a
  // message with no element behind it (a migration payload arriving on a
  // group entry -- asking such an envelope for an ArrayID aborts, hence the
  // guard). Resolved once here: admission control, the per-op migration
  // stand-down, and the receive-side registration cache all name the same
  // recipient.
  CkLocRec* recv_elt = (env->getMsgtype() == ForArrayEltMsg)
      ? CkFindDeviceRecvElement(env->getArrayMgr(), env->getRecipientID())
      : NULL;
#if CMK_LBDB_ON
  // Admission control against a departing element. Preprocessing this message
  // pins it to this process: the transfers land in buffers the element just
  // posted here, and the completion message carries pointers into this
  // address space. An element with a migration pending must therefore stop
  // admitting receives -- what it has in flight drains, its step stalls
  // waiting on the receives we hold back, the counter reaches zero, and it
  // leaves. Without this a continuously-receiving element never goes quiet
  // and the deferred migration starves, which wedges any balancer that waits
  // for its moves to finish. The requeued envelope is untouched (the bail is
  // before any mutation), so once the element departs, ordinary delivery
  // forwards it and the new host preprocesses it with local pointers.
  //
  // Only a parked element WITH A MIGRATION PENDING bounces, and both halves
  // of that condition are load bearing. An element with a migration pending
  // that is still running must keep admitting: it needs its current round's
  // receives to consume what it already accepted and to keep feeding its
  // neighbours, and it parks only once quiet -- freezing it at its first
  // quiet moment mid-run stalls its neighbours on missing ghosts and
  // deadlocks the step (measured, not hypothetical). A parked element with
  // no migration decision admits normally: buffering returns before the
  // transfer is issued, so the sender's completion callback never fires, and
  // an application that gates its step on "every zerocopy send I issued has
  // completed" (it must, since the next iteration repacks those buffers)
  // then hangs at that gate forever, never reaches its own park, and nothing
  // ever unparks and replays what was buffered. Measured: one element stuck
  // at its send drain, two neighbours holding 7 of 8 particle messages, and
  // sixty-one elements parked behind them.
  // The pendingMigrateTo condition used to be here too, so a device message
  // arriving at a PARKED element whose destination the strategy had not yet
  // chosen went down the normal delivery path. The element is parked, so its
  // `when` is not ready, and the SDAG buffers the message -- and if the
  // strategy then decides to move that element, _sdag_pup serialises a
  // buffered nocopydevice payload whose pointer is process-local. pic2d's own
  // pup comment states the invariant this violates: "no device-zerocopy
  // message ... is buffered here at migration time". The window is between the
  // park (cklocation.C, AtSyncWait sets deviceRecvParked) and the migration
  // decision (pendingMigrateTo, set when the strategy's move arrives), which
  // is why the wedge is nondeterministic and why it hits sync LB too -- AtSync
  // is AtSyncStart + AtSyncWait and parks just the same.
  //
  // Buffering for the whole park is safe now in a way it was not when this was
  // narrowed: that narrowing predates the switch from REQUEUE to BUFFER below.
  // A requeue spun on the local queue and starved the resume broadcast, so the
  // sender's completion never fired; the buffer path takes a reference and
  // replays on unpark or departure, so the sender completes at replay.
  if (recv_elt != NULL && (recv_elt->deviceRecvParked || recv_elt->migrateKickHeld)) {
    // Buffered, NOT requeued: a requeue spins on the scheduler's local
    // queue, which is popped ahead of the network -- so the resume
    // broadcast that would unpark this element starves behind its own
    // bounced messages, and the wedge sustains itself. Buffer the envelope
    // and replay it when the element unparks or departs.
    //
    // The delivery framework owns this envelope: every marshalled entry is
    // noKeep, so CkDeliverMessageFree frees the message the moment the
    // entry invocation this bail returns into completes -- the normal path
    // below survives that only by copying into new_env. Take a reference
    // so the buffered envelope stays alive until replay re-injects it;
    // redelivery then admits it (a copy is made and this reference is the
    // one the framework frees), bounces it again (another reference is
    // taken here), or forwards it to the element's new home (the send path
    // frees it after transmission).
    CmiReference((void*)env);
    CkDeviceRecvAdmissionBuffer(
        ck::ObjID(env->getRecipientID()).getElementID(), (void*)env);
    return;
  }
#endif
  // Change message header to invoke regular entry method
  CMI_ZC_MSGTYPE(env) = CMK_REG_NO_ZC_MSG;

  // Create a copy of this message for regular entry method invocation
  // FIXME: Reuse the old message instead of creating a new one
  void* old_msg = EnvToUsr(env);
  envelope* new_env = UsrToEnv(CkCopyMsg(&old_msg));
  if (zcLeakDbg()) { g_zc_copies.fetch_add(1); zcLeakReport("issue"); }

  // Retarget the copied message's device buffers to the buffers this receiver
  // posted. The transfers below land in arrPtrs[], but the copy still carries
  // the SENDER's CkDeviceBuffer::ptr, and the entry method delivered from it
  // reads that pointer as its data. Within one process the sender's pointer is
  // a valid local address holding the same bytes, so this went unnoticed;
  // across processes it names memory in another address space and the first
  // kernel touching it faults. Rewriting in place is safe because pupping a
  // CkDeviceBuffer is fixed-width -- only ptr changes value.
  {
    char* new_buf = ((CkMarshallMsg*)EnvToUsr(new_env))->msgBuf;
    PUP::fromMem walk(new_buf);
    int copy_numops;
    walk | copy_numops;
    for (int i = 0; i < numops && i < copy_numops; i++) {
      const size_t field_off = walk.size();
      CkDeviceBuffer db;
      walk | db;
      const size_t field_len = walk.size() - field_off;
      db.ptr = arrPtrs[i];
      PUP::toMem patch(new_buf + field_off);
      patch | db;
      // Rewriting in place is only sound if packing a CkDeviceBuffer produces
      // exactly as many bytes as unpacking consumed. If that ever stops being
      // true, this would silently overwrite the marshalled parameters that
      // follow (ref/dir/n) instead of just the pointer.
      if (patch.size() != field_len)
        CkAbort("CkRdmaDeviceIssueRgets: device buffer pup asymmetry "
                "(read %zu bytes, wrote %zu) -- in-place retarget is unsafe",
                field_len, patch.size());
    }
  }

  // Start unpacking marshalled message
  PUP::fromMem up((void *)((CkMarshallMsg *)EnvToUsr(env))->msgBuf);
  int received_numops;
  up|received_numops;
  CkAssert(numops == received_numops);

  CkDeviceBuffer source;

  // Machine layer does not support GPU-aware communication
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Find which mode of transfer should be used
  // CmiPrintf("[%d] CkRdmaDeviceOnSender: src_pe=%d, dst_pe=%d\n", CkMyPe(), env->getSrcPe(), CkMyPe());
  CkNcpyModeDevice mode = findTransferModeDevice(env->getSrcPe(), CkMyPe());

  // Allocate and fill in metadata for this zerocopy operation
  void* rdma_data = CmiAlloc(sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * numops);
  CmiEnforce(rdma_data);
  if (zcLeakDbg()) g_zc_meta.fetch_add(1);
  DeviceRdmaInfo* rdma_info = (DeviceRdmaInfo*)rdma_data;
  rdma_info->n_ops = numops;
  // Timed tally: zero posted time means "not measuring", so the completion path
  // can tell an untimed receive from one that completed instantly.
  rdma_info->zc_posted = zcStatsOn() ? CkWallTimer() : 0.0;
  rdma_info->zc_bytes = 0;
  rdma_info->zc_mode = 2;
  rdma_info->counter = 0;
  rdma_info->msg = new_env;

  // Watchdog bookkeeping; no cost unless CHARM_ZC_STALL_SECS is set.
  DeviceRecvWatch* watch = NULL;
  if (CkpvAccess(device_recv_watch) != NULL) {
    CkpvAccess(device_recv_watch)->push_back(
        DeviceRecvWatch{rdma_info, CkWallTimer(), numops,
                        std::vector<int>(numops, -1),
                        std::vector<size_t>(numops, 0),
                        std::vector<char>(numops, 0),
                        std::vector<int>(numops, -1),
                        std::vector<int>(numops, -1), false});
    watch = &CkpvAccess(device_recv_watch)->back();
  }

  // Set when the last op of this receive is an empty one, which completes
  // here rather than on a stream callback: nothing else will come back to
  // release the metadata, so this function has to.
  bool completed_inline = false;

  for (int i = 0; i < numops; i++) {
    // Unpack source buffer from sender
    up|source;

    if (arrSizes[i] > source.cnt) {
      CkAbort("CkRdmaDeviceIssueRgets: posted data size is larger than source data size!");
    }

    // Store information about this buffer
    DeviceRdmaOp& save_op = *(DeviceRdmaOp*)((char*)rdma_data
        + sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * i);
    // Use the PE we are actually running on, not the one the sender recorded.
    // If the target chare migrated after the sender posted this transfer, the
    // sender's dest_pe is stale; the message itself has already been routed
    // here by the location manager, so this PE is the one that hosts the chare
    // and that posted arrPtrs below.
    save_op.dest_pe  = CkMyPe();
    save_op.dest_ptr = arrPtrs[i];
    save_op.size = (size_t)arrSizes[i];
    save_op.info = rdma_info;
    // rdma_data comes from CmiAlloc, so this is uninitialised storage: without
    // clearing it, a stale non-zero would be read as an rget timestamp and
    // charge a nonsense interval to the cross-node tier. Only the rget branch
    // sets it.
    save_op.rget_posted = 0.0;
    // DIAGNOSTIC: what buffer did this receive post, for which element, and in
    // which mode. Correlated against the application's own record of which
    // ghost it later reads out of that buffer, this shows whether the runtime
    // ever lands one message's payload in another message's posted buffer.
    save_op.src_pe = env->getSrcPe();
    // Only an array-element message names an element; the migration payload
    // arrives on a group entry, and asking it for an ArrayID aborts. Those
    // receives are never deferred for correction, so -1 simply means "no
    // element to stand down".
    if (env->getMsgtype() == ForArrayEltMsg) {
      save_op.dest_aid_idx = env->getArrayMgr().idx;
      save_op.dest_id = env->getRecipientID();
      // Stand the receiving element's migration down for EVERY in-flight
      // receive, not only the deferred ones. The transfers below land in the
      // buffers this element just posted, and the completion message carries
      // pointers into this process; an element that migrates mid-flight frees
      // those buffers under an active transfer, and the forwarded completion
      // then unpacks a pointer from an address space the new host cannot
      // read. The sender has held its side of every raw-buffer transfer since
      // the interlock was introduced; this is the receiving half.
      if (recv_elt != NULL) {
        recv_elt->noteDeviceSendPosted();
        // Released when the completion message is freed -- consumption, not
        // transfer completion. An SDAG entry can buffer that message for a
        // future round; migrating in that window ships it, still carrying
        // this process's pointers, to a host that cannot read them.
        (*CkpvAccess(device_recv_holds))[(void*)new_env].push_back(
            DeviceRecvHold{save_op.dest_aid_idx, save_op.dest_id});
      }
    } else {
      save_op.dest_aid_idx = -1;
      save_op.dest_id = 0;
    }

    if (watch) {
      watch->src_pe[i] = env->getSrcPe(); watch->size[i] = (size_t)arrSizes[i];
      watch->dev_idx[i] = source.device_idx; watch->ev_idx[i] = source.event_idx;
    }
    save_op.stream = (void*)postStructs[i].hapi_stream;
    save_op.src_cb = (source.cb.type != CkCallback::ignore) ? new CkCallback(source.cb) : nullptr;
    save_op.dst_cb = nullptr;

    // A mismatch here means the target chare migrated between the sender
    // posting this transfer and its arrival. That is expected and safe: the
    // transfer mode above is derived from CkMyPe(), the destination buffers
    // are the ones this PE posted, and the message reached this PE precisely
    // because the location manager knows the chare lives here now. Previously
    // this aborted, which made any migration concurrent with a GPU-direct send
    // fatal.
    if (source.dest_pe != CkMyPe() && _lb_args.debug() > 1) {
      CmiPrintf("[%d] CkRdmaDeviceIssueRgets: sender addressed PE %d (src PE %d); "
                "chare has since migrated here, retargeting.\n",
                CkMyPe(), source.dest_pe, env->getSrcPe());
    }

    // Empty payload: the sender prepared nothing and fired its callback
    // already (CkRdmaDeviceOnSender). Complete the op here, on the host, with
    // no stream wait, copy or event, and deliver once the message's last op
    // is in -- the same count the asynchronous completions drive. There is no
    // QdProcess to balance: nothing was created for this op.
    if (source.cnt == 0) {
      if (zcLeakDbg()) g_zc_empty.fetch_add(1);
      rdma_info->counter++;
      if (rdma_info->counter == rdma_info->n_ops) {
        QdCreate(1);
        enqueueNcpyMessage(CkMyPe(), new_env);
        // The counter can only reach n_ops on the last op there is: every
        // other op has already counted itself, and an op later in this loop
        // has not been unpacked yet. So no save_op below is still live, and
        // the metadata is freed after the loop rather than here only to keep
        // that obvious.
        completed_inline = true;
      }
      continue;
    }

    // The copy lands on the PE's receive stream, issued only once what it
    // depends on is complete (DeviceRecvPending); save_op.stream names it for
    // a transfer that resumes after a correction. What the destination waits
    // for is decided per branch below, before anything later reaches the
    // posted stream.
    save_op.stream = (void*)deviceRecvStream();
    save_op.dst_flag_rank = -1;
    save_op.dst_flag_seq = 0;
    save_op.dst_event = NULL;
    save_op.ipc_device_idx = -1;
    save_op.ipc_event_idx = -1;

    // Destination buffer (on this receiver)
    CkDeviceBuffer dest((const void *)arrPtrs[i], arrSizes[i]);

    // Perform data transfers.
    //
    // `mode` (computed above from the true, always-accurate delivery PEs) is
    // authoritative -- it is never a guess, unlike the sender's transfer_mode
    // decision (see CkRdmaDeviceOnSender), which can be made without knowing
    // the destination. CkRdmaDeviceOnSender's job is to make sure whatever
    // `mode` picks here finds the metadata it needs already staged: it always
    // stages IPC info when it isn't certain MEMCPY suffices and RDMA can be
    // ruled out, so an unconfirmed-destination send still leaves
    // source.device_idx valid for the IPC branch below.
    if (zcLeakDbg()) g_zc_real.fetch_add(1);
    zcStatsCount(mode);
    // Every op of one receive comes from the same sender and so resolves the
    // same way; recording it per op just avoids a second mode computation.
    rdma_info->zc_bytes += (size_t)arrSizes[i];
    rdma_info->zc_mode = zcModeSlot(mode);

    // CHARM_ZC_VALIDATE: check both pointers before handing them to CUDA. An
    // illegal access raised by the copies below is asynchronous and sticky, so
    // it otherwise surfaces at some unrelated later call and says nothing about
    // which side was bad. Migration is what makes this ambiguous: the source
    // buffer can be freed by an object that migrated away, and the destination
    // is whatever the (possibly just-migrated) receiver posted.
    if (zcValidateOn()) {
      cudaPointerAttributes sattr{}, dattr{};
      const cudaError_t serr = cudaPointerGetAttributes(&sattr, source.ptr);
      const cudaError_t derr = cudaPointerGetAttributes(&dattr, (void*)dest.ptr);
      const bool sbad = (serr != cudaSuccess || sattr.type == cudaMemoryTypeUnregistered);
      const bool dbad = (derr != cudaSuccess || dattr.type == cudaMemoryTypeUnregistered);
      // A staged IPC receive reads the sender's comm buffer through an imported
      // handle, never source.ptr, so source.ptr being unmapped here is the normal
      // cross-process case and says nothing. Only report it when this receive
      // would actually dereference it.
      const bool src_will_be_read =
          !(source.device_idx != -1 && csv_gpu_manager.use_shm);
      if ((sbad && src_will_be_read) || dbad) {
        CmiPrintf("[%d] ZC VALIDATE FAIL mode=%d srcPe=%d src=%p(%s type=%d) "
                  "dst=%p(%s type=%d) cnt=%zu dev_idx=%d\n",
                  CkMyPe(), (int)mode, env->getSrcPe(),
                  source.ptr, sbad ? "BAD" : "ok", (int)sattr.type,
                  (void*)dest.ptr, dbad ? "BAD" : "ok", (int)dattr.type,
                  (size_t)dest.cnt, source.device_idx);
        fflush(stdout);
        cudaGetLastError();  // clear so the report is not itself sticky
      }
    }

    // Prefer whatever the sender actually prepared over the mode derived here.
    // The sender exported this buffer one of two ways -- staged into its
    // communication buffer, or named directly by an IPC handle -- and either
    // arrangement is what the metadata in hand describes, so honour it rather
    // than re-deciding from locality.
    //
    // A staged copy additionally holds the bytes outright, so it survives the
    // sending chare migrating away and freeing its source buffer. A direct one
    // does not: it reads the sender's live allocation, which is why such a send
    // must carry a completion callback the sender waits on before reusing or
    // freeing that buffer.
    // device_idx is node-local -- device_count * CmiMyNodeRankLocal() +
    // my_device_id -- so PE 1 on one node and PE 5 on another both produce 1.
    // Indexing this node's hapi_ipc_device_infos with a remote sender's index
    // resolves to an unrelated local staged block, and the receive silently
    // delivers some other chare's ghost. Measured: staged blocks (dev=1,off=0)
    // were referenced by receives naming srcPEs 1 and 5 in the same run.
    //
    // A sender only stages for a destination it believes is in another process
    // on its own node, so an export arriving from another physical node means
    // the target moved after the mode was chosen. Treat it as unprepared and
    // let the correction path re-fetch the payload by a route that works.
    {
      // A staged landing (CkRdmaDeviceStageParked): the payload is already on
      // this device behind the event the pull recorded. Copy it into the
      // posted buffer on the posting stream and free it behind that copy. The
      // sender was completed by the staging pull; the descriptor carries no
      // callback, so src_cb is null and nothing fires twice.
      StagedLandingMap& landings = *CkpvAccess(staged_landings);
      auto lit = isStagedLandingDescriptor(source) ? landings.find(source.ptr) : landings.end();
      if (lit != landings.end() && (size_t)dest.cnt > lit->second.cnt)
        CkAbort("CkRdmaDeviceIssueRgets: staged landing %p holds %zu bytes but %zu were posted",
                source.ptr, lit->second.cnt, (size_t)dest.cnt);
      if (lit != landings.end()) {
        const StagedLanding sl = lit->second;
        landings.erase(lit);
        if (save_op.src_cb) {  // the sender was completed by the staging pull
          delete (CkCallback*)save_op.src_cb;
          save_op.src_cb = nullptr;
        }
        // The pull's completion (sl.ready) is the source condition here;
        // deviceRecvIssue copies the landing into the posted buffer.
        deviceRecvNoteDestination(save_op, postStructs[i]);
        DeviceRecvPending pr;
        pr.op = &save_op;
        pr.source = source;
        pr.dest = dest;
        pr.mode = mode;
        pr.src_pe = env->getSrcPe();
        pr.kind = DeviceRecvKind::Staged;
        pr.sl = sl;
        pr.done = CkRdmaDeviceRecvHandler;
        deviceRecvTryIssue(std::move(pr));
        continue;
      }
    }
    const bool sender_exported =
        (source.ipc_protocol != CmiIpcProtocol::NONE &&
         source.device_idx != -1 && csv_gpu_manager.use_shm &&
         CmiPeOnSamePhysicalNode(env->getSrcPe(), CkMyPe()));
    const bool sender_direct =
        (sender_exported && source.ipc_protocol == CmiIpcProtocol::DIRECT);

    if (mode == CkNcpyModeDevice::MEMCPY && !sender_exported) {
      // Same process: the copy is issued by deviceRecvIssue once the sender's
      // producing work and the destination's earlier work are both complete.
      deviceRecvNoteDestination(save_op, postStructs[i]);
      DeviceRecvPending pr;
      pr.op = &save_op;
      pr.source = source;
      pr.dest = dest;
      pr.mode = mode;
      pr.src_pe = env->getSrcPe();
      pr.kind = DeviceRecvKind::Memcpy;
      pr.done = CkRdmaDeviceRecvHandler;
      deviceRecvTryIssue(std::move(pr));
    } else if (sender_exported) {
      deviceRecvNoteDestination(save_op, postStructs[i]);
      // A direct send can be unmappable here even though nothing is wrong with
      // it. One cudaIpcOpenMemHandle maps the whole suballocator region behind
      // an allocation, so a second allocation carved from that same region
      // cannot be opened at all -- the driver returns cudaErrorAlreadyMapped --
      // and no API says which held mapping already covers it.
      //
      // This is not a migration effect: it reproduces with load balancing
      // switched off entirely, around iteration 15 of pic2d.
      //
      // So take the route that exists for a payload this process cannot read,
      // the same correction the migration mismatch below uses. Importing here
      // rather than inside deviceIpcReceive is what makes that possible: the
      // mapping has to be known to have failed while the send is still
      // identifiable and the buffer can still be asked for again. A successful
      // import is cached, so the one inside deviceIpcReceive hits.
      if (sender_direct && mode != CkNcpyModeDevice::MEMCPY &&
          hapiIpcImportBuffer(source.ipc_handle, CmiNodeOf(env->getSrcPe()),
                              source.ipc_base,
                              source.ipc_offset + (size_t)dest.cnt) == NULL) {
        if (watch) watch->deferred[i] = 1;
        CkGroupID def_aid; def_aid.idx = save_op.dest_aid_idx;
        src_prepared_dbg = source.sender_prepared;
        src_proto_dbg = (int)source.ipc_protocol;
        src_dev_idx_dbg = source.device_idx;
        requestDeviceRestage(env->getSrcPe(), (void*)&save_op, source.ptr,
                             def_aid, save_op.dest_id,
                             source.ipc_protocol == CmiIpcProtocol::STAGED,
                             source.comm_offset,
                             source.memcpy_event, source.event_idx,
                             (size_t)dest.cnt, arrPtrs[i], (size_t)arrSizes[i],
                             mode != CkNcpyModeDevice::IPC);
        continue;  // completion deferred until the retransmit lands
      }
      DeviceRecvPending pr;
      pr.op = &save_op;
      pr.source = source;
      pr.dest = dest;
      pr.mode = mode;
      pr.src_pe = env->getSrcPe();
      pr.kind = DeviceRecvKind::Ipc;
      pr.done = CkRdmaDeviceRecvHandler;
      deviceRecvTryIssue(std::move(pr));
    } else {
      // Nothing staged and not same-process. That is legitimate only for a
      // genuine cross-physical-node transfer, where the sender's else-branch
      // registered source.lci_ncpy_buffer for the rdmaGet below.
      //
      // IPC here means the sender resolved this destination to its own process
      // and staged nothing -- and then the target migrated to another process
      // before the message landed. The transfer mode is chosen at send time from
      // the location the sender knows; it only becomes true at delivery. The
      // source buffer lives in an address space this PE cannot read, and
      // source.lci_ncpy_buffer was never registered, so the rdmaGet below would
      // issue against an unregistered descriptor -- which surfaces as an LCI
      // "Message too long" assert or silent corruption, nowhere near the cause.
      //
      // Fail here, where the send is still identifiable, rather than there.
      // The sender prepared nothing readable from another address space, so
      // the target migrated across a process boundary after the mode was
      // chosen. Defer this buffer and ask for the payload again by a route
      // this process can read; see the protocol notes above. The message is
      // already where it belongs -- only the bytes are missing.
      if (!source.sender_prepared ||
          (source.device_idx != -1 &&
           !CmiPeOnSamePhysicalNode(env->getSrcPe(), CkMyPe()))) {
        if (watch) watch->deferred[i] = 1;
        deviceRecvNoteDestination(save_op, postStructs[i]);
        CkGroupID def_aid; def_aid.idx = save_op.dest_aid_idx;
        src_prepared_dbg = source.sender_prepared;
        src_proto_dbg = (int)source.ipc_protocol;
        src_dev_idx_dbg = source.device_idx;
        requestDeviceRestage(env->getSrcPe(), (void*)&save_op, source.ptr,
                             def_aid, save_op.dest_id,
                             source.ipc_protocol == CmiIpcProtocol::STAGED,
                             source.comm_offset,
                             source.memcpy_event, source.event_idx,
                             (size_t)dest.cnt, arrPtrs[i], (size_t)arrSizes[i],
                             mode != CkNcpyModeDevice::IPC);
        continue;  // completion deferred until the retransmit lands
      }
      // CmiPrintf("it should never be called during intra node\n");
#if CMK_GPU_COMM
      // Machine layer supports GPU-aware communication
      QdCreate(1);
      CmiSetDirectNcpyAckHandler(CkRdmaDeviceRecvHandler);
      // The destination registration is cached against the element this
      // receive is addressed to, whose migration frees the posted buffer and
      // retires the entry. A receive with no element behind it (a migration
      // payload on a group entry) registers per-op as before -- unless the
      // buffer is a pool block, which takes its arena's one registration
      // whoever posted it.
      CmiNcpyBuffer lci_dest_ncpy_buffer;
      void* arena_base_unused = NULL;
      size_t arena_extent_unused = 0;
      const bool in_pool = hapiDevPoolOn() &&
          hapiDevPoolArenaOf(arrPtrs[i], &arena_base_unused, &arena_extent_unused);
      if (recv_elt != NULL || in_pool) {
        lci_dest_ncpy_buffer =
            acquireDeviceRegistration(arrPtrs[i], (size_t)arrSizes[i], recv_elt);
        // What the uncached 3-arg constructor's third argument sets: the
        // completion handler finds this operation's DeviceRdmaOp through it.
        lci_dest_ncpy_buffer.deviceRdmaOpInfo = (void*)(&save_op);
      } else {
        lci_dest_ncpy_buffer =
            CmiNcpyBuffer(arrPtrs[i], (size_t)arrSizes[i], (void*)(&save_op));
      }
      // Stamp the cross-node tier per op; see DeviceRdmaOp::rget_posted.
      save_op.rget_posted = zcStatsOn() ? CkWallTimer() : 0.0;
      lci_dest_ncpy_buffer.rdmaGet(source.lci_ncpy_buffer, 0, nullptr, nullptr);
      continue;
#else
      // Handle all other cases (basic inter-process and inter-node)
      // Transfer the received/unpacked data on host to the destination device buffer
      // FIXME: Print warning that this is slow?
      CkAssert(source.data_stored);
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.data, dest.cnt,
            hapiMemcpyHostToDevice, postStructs[i].hapi_stream));
      hapiAddCallback(postStructs[i].hapi_stream, CkCallback(CkRdmaDeviceRecvHandler, &save_op));
#endif
    }
  }

  // A receive whose every buffer was empty raises no completion: the device
  // was never touched, so no stream callback runs and CkRdmaDeviceRecvHandler,
  // which is what frees this for every other receive, is never reached. Left
  // undone it is one CmiAlloc per message for a stencil that sends to each
  // neighbour every step whether or not it has anything to say -- tens of
  // thousands of messages a second, and the process is killed for memory long
  // before the run ends.
  if (completed_inline) {
    if (zcLeakDbg()) g_zc_metafree.fetch_add(1);
    deviceRecvWatchDrop(rdma_info);
    CmiFree(rdma_data);
  }
}

/*************** Staged receive for a message parked at its destination ***************/
//
// A device message that reaches the PE an element is migrating TO before the
// element has landed is parked there (ckarray.C, bufferedIDMsgs) until it
// lands. A direct device send keeps the SENDER's buffer live and its
// outstanding-send count raised until the receiver pulls, so a parked message
// pins its sender for the whole landing. When that sender must itself move,
// or this PE's landing gate is holding memory for that sender's pack, the wait
// is a cycle and nothing lands: with parking alone, every sph2d async run
// wedged at its first balancing step (2026-09-15).
//
// So the payload is pulled NOW, into landing buffers from the device pool,
// with the same per-protocol code the element's own receive would use, and the
// sender is completed the same way (its callback, its IPC slot flag). The
// parked message's descriptors are rewritten in place to name the landing
// buffers; when the element lands and the message is delivered,
// CkRdmaDeviceIssueRgets recognises a landing buffer, copies it into the
// element's posted buffer on the posting stream, and frees it behind the copy.
//
// Left parked as before, with the sender live: a source this PE cannot pull
// directly -- an unprepared sender, an IPC handle that does not import, any
// off-node source (the rget path completes by ack, not on a stream, and is not
// staged in this version) -- and a payload the pool cannot serve.
bool CkRdmaDeviceStageParked(envelope* env)
{
  if (!CMI_IS_ZC_DEVICE(env) || env->getMsgtype() != ForArrayEltMsg) return false;  // host message
  if (!CkDevicePoolOn()) return false;
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  const int srcPe = env->getSrcPe();
  if (!CmiPeOnSamePhysicalNode(srcPe, CkMyPe())) {
    if (stagedDbg()) CmiPrintf("[STAGE-SKIP %d] source PE %d is off-node\n", CkMyPe(), srcPe);
    return false;
  }
  const CkNcpyModeDevice mode = findTransferModeDevice(srcPe, CkMyPe());
  char* buf = ((CkMarshallMsg*)EnvToUsr(env))->msgBuf;

  // Pass 1: every descriptor, and whether each is pullable from here.
  std::vector<CkDeviceBuffer> src;
  std::vector<size_t> off, width;
  {
    PUP::fromMem rd(buf);
    int n = 0;
    rd | n;
    for (int i = 0; i < n; i++) {
      const size_t o = rd.size();
      CkDeviceBuffer b;
      rd | b;
      src.push_back(b);
      off.push_back(o);
      width.push_back(rd.size() - o);
    }
  }
  StagedLandingMap& landings = *CkpvAccess(staged_landings);
  int live = 0;
  for (size_t i = 0; i < src.size(); i++) {
    const CkDeviceBuffer& b = src[i];
    if (b.cnt == 0) continue;
    if (isStagedLandingDescriptor(b)) {
      if (stagedDbg()) CmiPrintf("[STAGE-SKIP %d] already staged %p\n", CkMyPe(), b.ptr);
      return false;
    }
    const bool sender_exported =
        (b.ipc_protocol != CmiIpcProtocol::NONE && b.device_idx != -1 &&
         csv_gpu_manager.use_shm);
    const bool sender_direct =
        (sender_exported && b.ipc_protocol == CmiIpcProtocol::DIRECT);
    if (mode == CkNcpyModeDevice::MEMCPY && !sender_exported) { live++; continue; }
    if (sender_exported) {
      if (sender_direct && mode != CkNcpyModeDevice::MEMCPY &&
          hapiIpcImportBuffer(b.ipc_handle, CmiNodeOf(srcPe), b.ipc_base,
                              b.ipc_offset + (size_t)b.cnt) == NULL) {
        if (stagedDbg()) CmiPrintf("[STAGE-SKIP %d] IPC handle from PE %d did not import\n", CkMyPe(), srcPe);
        return false;
      }
      live++;
      continue;
    }
    // needs the rget / correction path: leave it to the element
    if (stagedDbg())
      CmiPrintf("[STAGE-SKIP %d] from PE %d: mode=%d proto=%d dev_idx=%d prepared=%d not pullable here\n",
                CkMyPe(), srcPe, (int)mode, (int)b.ipc_protocol, b.device_idx, (int)b.sender_prepared);
    return false;
  }
  if (live == 0) return false;

  // Pass 2: landing buffers, all or nothing.
  std::vector<void*> land(src.size(), nullptr);
  for (size_t i = 0; i < src.size(); i++) {
    if (src[i].cnt == 0) continue;
    land[i] = CkDeviceMalloc(src[i].cnt);
    if (land[i] == nullptr) {
      for (size_t j = 0; j < i; j++) if (land[j]) CkDeviceFree(land[j]);
      if (stagedDbg())
        CmiPrintf("[STAGE-SKIP %d] pool could not serve %zu bytes\n", CkMyPe(),
                  (size_t)src[i].cnt);
      return false;
    }
  }

  // Pass 3: pull, as CkRdmaDeviceIssueRgets would, into the landing buffers.
  const int numops = (int)src.size();
  void* rdma_data = CmiAlloc(sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * numops);
  CmiEnforce(rdma_data);
  DeviceRdmaInfo* info = (DeviceRdmaInfo*)rdma_data;
  info->n_ops = live;
  info->counter = 0;
  info->msg = nullptr;   // the mark of a staged pull: nothing to deliver at completion
  info->zc_posted = 0.0;
  info->zc_bytes = 0;
  info->zc_mode = zcModeSlot(mode);
  hapiStream_t st = stagingStream();
  size_t bytes = 0;
  for (int i = 0; i < numops; i++) {
    CkDeviceBuffer& source = src[i];
    if (source.cnt == 0) continue;
    DeviceRdmaOp& op = *(DeviceRdmaOp*)((char*)rdma_data
        + sizeof(DeviceRdmaInfo) + sizeof(DeviceRdmaOp) * i);
    op.stream = (void*)st;
    op.dest_ptr = land[i];
    op.size = (size_t)source.cnt;
    op.info = info;
    op.src_cb = (source.cb.type != CkCallback::ignore) ? new CkCallback(source.cb) : nullptr;
    op.dst_cb = nullptr;
    op.tag = 0;
    op.dest_pe = CkMyPe();
    op.dest_aid_idx = env->getArrayMgr().idx;
    op.dest_id = env->getRecipientID();
    op.src_pe = srcPe;
    op.src_mpi_rank = source.src_mpi_rank;
    op.dest_mpi_rank = CmiMyNode();
    op.dst_flag_rank = -1;
    op.dst_flag_seq = 0;
    op.dst_event = NULL;
    op.ipc_device_idx = -1;
    op.ipc_event_idx = -1;
    CkDeviceBuffer dest((const void*)land[i], source.cnt);
    bytes += (size_t)source.cnt;
    const bool sender_exported =
        (source.ipc_protocol != CmiIpcProtocol::NONE && source.device_idx != -1 &&
         csv_gpu_manager.use_shm);
    // The sender's slot, if it claimed one, is released when the pull
    // completes: CkRdmaDeviceRecvHandler raises its dst_flag.
    if (source.device_idx >= 0 && source.event_idx >= 0 && csv_gpu_manager.use_shm) {
      op.ipc_device_idx = source.device_idx;
      op.ipc_event_idx = source.event_idx;
    }
    if (mode == CkNcpyModeDevice::MEMCPY && !sender_exported) {
      if (source.memcpy_event != NULL)
        hapiCheck(hapiStreamWaitEvent(st, (hapiEvent_t)source.memcpy_event, 0));
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt, cudaMemcpyDefault, st));
    } else {
      deviceIpcReceive(source, dest, st, srcPe, mode);
    }
    hapiAddCallback(st, CkCallback(CkRdmaDeviceRecvHandler, &op));
    hapiEvent_t ready = stagingEventTake();
    hapiCheck(hapiEventRecord(ready, st));
    const void* src_addr = nullptr;
    if (stageVerifyOn()) {
      if (mode == CkNcpyModeDevice::MEMCPY) src_addr = source.ptr;
      else if (sender_exported && source.ipc_protocol == CmiIpcProtocol::DIRECT) {
        void* base = hapiIpcImportBuffer(source.ipc_handle, CmiNodeOf(srcPe), source.ipc_base,
                                         source.ipc_offset + (size_t)source.cnt);
        if (base) src_addr = (const char*)base + source.ipc_offset;
      }
    }
    landings[land[i]] = StagedLanding{ready, (size_t)source.cnt, src_addr, 0ULL, false};
  }

  // Pass 4: the descriptors now name the landing buffers, as a process-local
  // source with no callback (the sender's fired from the pull above).
  for (int i = 0; i < numops; i++) {
    if (src[i].cnt == 0) continue;
    CkDeviceBuffer nb = src[i];
    nb.ptr = land[i];
    nb.src_pe = CkMyPe();
    nb.src_mpi_rank = CmiMyNode();
    nb.dest_pe = CkMyPe();
    nb.dest_mpi_rank = CmiMyNode();
    nb.device_idx = -1;
    nb.event_idx = -1;
    nb.comm_offset = kStagedLandingMark;
    nb.ipc_protocol = CmiIpcProtocol::NONE;
    nb.ipc_offset = 0;
    nb.ipc_base = NULL;
    nb.memcpy_event = NULL;
    nb.ready_rank = -1;
    nb.ready_seq = 0;
    nb.sender_prepared = true;
    nb.data_stored = false;
    nb.data = NULL;
    // The sender's callback stays in the descriptor: a CkCallback pups a
    // type-dependent number of bytes, so replacing it would change the
    // width of an in-place rewrite. It fired once, from the staging pull
    // above; the landing branch in CkRdmaDeviceIssueRgets drops its copy.
    PUP::toMem wr(buf + off[i]);
    wr | nb;
    if (wr.size() != width[i])
      CkAbort("CkRdmaDeviceStageParked: descriptor width changed (%zu -> %zu)",
              width[i], wr.size());
  }
  if (stagedDbg())
    CmiPrintf("[STAGE %d] id=%llu from PE %d: %d op(s), %zu bytes into landing buffers\n",
              CkMyPe(), (unsigned long long)ck::ObjID(env->getRecipientID()).getElementID(),
              srcPe, live, bytes);
  return true;
}

// Unused, left for future reference
/*
int CkRdmaGetDestPEChare(int dest_pe, void* obj_ptr) {
  // Mechanism extracted from _prepareMsg() in ck.C
  if (dest_pe < 0) {
    int pe = -(dest_pe+1);
    if (pe == CkMyPe()) {
      VidBlock* vblk = CkpvAccess(vidblocks)[(CmiIntPtr)obj_ptr];
      void *objPtr = vblk->getLocalChare();
      dest_pe = objPtr ? pe : vblk->getActualID().onPE;
    } else {
      dest_pe = pe;
    }
  }

  return dest_pe;
}
*/

// Reclaim the IPC events in this PE's slice whose transfers have completed,
// releasing the device comm-buffer block each one was holding. Returns how many
// slots were freed.
//
// Split out of the old findFreeIpcEvent so a sender that finds either resource
// exhausted can keep re-running just this scan while it waits (see
// acquireIpcSendSlot). Nothing here depends on this PE's scheduler: an event
// becomes reclaimable when the *peer* process's receiver records its event and
// sets dst_flag in shared memory, so repeating the scan is what lets a
// momentarily-exhausted pool recover.
//
// max_to_free bounds the sweep: a sender that needs one event slot stops as
// soon as it has one, instead of paying a driver query for every busy slot in
// the slice. Pass 0 for an unbounded sweep, which is what a sender short of
// comm-buffer *bytes* wants -- it cannot know in advance how many blocks it
// must reclaim to fit its request.
//
// Index of this PE among the PEs sharing its device.
//
// The IPC event pool holds one slice per PE *sharing a device* -- its total is
// hapi_ipc_event_pool_size_pe * pes_per_device -- so a slice must be chosen by
// this index, not by the PE's rank in the process. The two differ as soon as a
// process drives more than one GPU: with 8 PEs over 4 devices, CkMyRank()
// reaches 7 and would index 1792 into a 512-entry pool, handing back an event
// that was never created and failing as 'invalid resource handle' when it is
// recorded.
//
// Derived from the device ids rather than from arithmetic on the rank, because
// Block and RoundRobin group PEs onto devices differently and each would
// otherwise need its own formula. Cached: the mapping is fixed after startup.
static int ipcEventPoolSlice(int cpv_my_device_id) {
  static thread_local int cached = -1;
  if (cached < 0) {
    int slice = 0;
    for (int k = 0; k < CkMyRank(); k++)
      if (CpvAccessOther(my_device_id, k) == cpv_my_device_id) slice++;
    cached = slice;
  }
  return cached;
}

// The caller must NOT hold dm->lock. The slice [pool_start, pool_start +
// pool_size) belongs to this PE alone -- pool_start is derived from this PE's
// index among those sharing its device,
// so ranks index disjoint elements -- and the only shared state here is the
// buddy allocator. So the scan runs lock-free and the lock is taken once, at
// the end, around the batch of frees.
static int reclaimCompletedIpcEvents(DeviceManager* dm, int cpv_my_device_id,
                                     int max_to_free) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int pool_size = csv_gpu_manager.hapi_ipc_event_pool_size_pe;
  int pool_start = ipcEventPoolSlice(cpv_my_device_id) * pool_size;
  const int my_device_index =
      csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[my_device_index];
  hapi_ipc_event_shared* my_shm_events =
      (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
          + csv_gpu_manager.shm_chunk_size * my_device_index
          + sizeof(hapiIpcMemHandle_t));

  // Offsets of blocks whose slot has been retired but whose memory has not been
  // handed back yet. Collected here so the whole scan stays outside dm->lock;
  // flushed when full so this stays a fixed-size stack buffer whatever
  // +gpuipceventpool is set to.
  constexpr int kFreeBatch = 64;
  size_t pending_free[kFreeBatch];
  int npending = 0;
  int nfreed = 0;

  auto flush = [&]() {
    if (npending == 0) return;
#if CMK_SMP
    CmiLock(dm->lock);
#endif
    for (int j = 0; j < npending; j++) dm->free_comm_buffer(pending_free[j]);
#if CMK_SMP
    CmiUnlock(dm->lock);
#endif
    npending = 0;
  };

  // Free IPC events that are complete
  for (int i = pool_start; i < pool_start + pool_size; i++) {
    int& event_flag = my_device_info.event_pool_flags[i];
    if (event_flag == 0) continue;  // slot is already free

    // Check in shared memory whether the receiver has invoked the memcpy from
    // the device comm buffer on the sender to the destination buffer. Acquire
    // pairs with the receiver's release store, and is what makes its
    // hapiEventRecord visible before the query below.
    // Raised by the receiver's completion handler once its copy out of this
    // slot has landed (or by a correction path that owns the slot): no event
    // to query.
    if (!my_shm_events[i].dst_flag.load(std::memory_order_acquire)) continue;
    // The publish callback behind this use's source event must have fired
    // too: freed before that, the slot could be claimed and named in a new
    // send, and the late callback would mark that use ready.
    if (!my_shm_events[i].src_ready.load(std::memory_order_acquire)) continue;

    // The receiver has invoked the memcpy, so the sender may query the event.
    // Event completion means the transfer from the source device comm buffer to
    // the destination buffer is done, so the allocated block can go back.
    if (event_flag == 1) {
      pending_free[npending++] = my_device_info.event_pool_buff_offsets[i];
      if (npending == kFreeBatch) flush();
    } else if (event_flag != 2) {
      // 2 is a DIRECT transfer: it never allocated a block, so there is
      // nothing to release beyond the event slot itself. Its completion
      // still matters -- it is what tells the sender the receiver has
      // finished reading its source buffer.
      CkAbort("IPC slot flagged complete but not in use");
    }

    // Mark event as free. Ordered after the offset is read above, so the slot
    // cannot be reclaimed and re-pointed before its old block is recorded.
    event_flag = 0;
    my_shm_events[i].dst_flag.store(false, std::memory_order_release);

    if (++nfreed == max_to_free) break;
  }

  flush();
  return nfreed;
}

// Claim a free IPC event from this PE's slice, or -1 if none is free.
//
// Two events are used per message:
// 1) Recorded by the sender after 'source buffer -> device comm buffer' hapiMemcpy.
//    Can be used by the sender to determine if the sender buffer is free for reuse.
//    It is also used by the receiver to create a dependency for the second hapiMemcpy
//    ('device comm buffer -> dest buffer')
// 2) Recorded by the receiver after 'device comm buffer -> dest buffer' hapiMemcpy.
//    It is used by the sender to determine when the allocated block on
//    device comm buffer and IPC events can be freed.
//
// Needs no lock: the slice is this PE's alone (see reclaimCompletedIpcEvents).
// flag_value distinguishes what the slot is holding: 1 for a staged transfer,
// whose comm-buffer block at comm_offset must be released when the slot
// retires, 2 for a direct one, which holds no block. A sentinel offset would
// not do -- 0 is a legal block offset.
static int claimFreeIpcEvent(int cpv_my_device_id, const size_t comm_offset,
                             int flag_value) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int pool_size = csv_gpu_manager.hapi_ipc_event_pool_size_pe;
  int pool_start = ipcEventPoolSlice(cpv_my_device_id) * pool_size;
  const int my_device_index =
      csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[my_device_index];

  for (int i = pool_start; i < pool_start + pool_size; i++) {
    int& event_flag = my_device_info.event_pool_flags[i];
    size_t& buff_offset = my_device_info.event_pool_buff_offsets[i];
    if (event_flag == 0) {
      event_flag = flag_value;
      buff_offset = comm_offset;
      // Not ready until this use's publish callback says so. The receiver
      // reads it only after the message naming the slot, which follows this.
      if (csv_gpu_manager.use_shm)
        ipcSharedSlot(my_device_index, i)->src_ready.store(false, std::memory_order_release);
      return i;
    }
  }

  return -1;
}

// Acquire the two resources a cross-process IPC send needs -- a block of the
// device comm buffer and an IPC event pair -- waiting for in-flight transfers
// to release them rather than aborting when the pools are momentarily empty.
//
// Both pools are sized for steady state, but a burst can transiently want more
// than they hold. leanmd's first step is the case that exposed this: every cell
// contacts all 26 neighbours at once, and on a 32-PE run that asks for more
// concurrent transfers per PE than a 256-event slice provides. Enlarging the
// pool is not the fix -- every slot costs two cudaEventInterprocess events per
// PE sharing the device, so merely doubling the default already fails device
// allocation at init (hapi_impl.cpp, ipcEventPoolInit). The slots are genuinely
// transient, so the sender waits for one to come back instead.
//
// The wait makes progress only while peers keep running: a slot is released
// once the receiving process handles the transfer and sets dst_flag in shared
// memory. That happens independently of this PE -- the sends holding those
// slots are already in flight -- so the common case drains quickly. It cannot
// happen if every participating PE is parked in this loop simultaneously, so
// the wait is bounded and says so, rather than hanging. Re-entering the
// scheduler (CsdSchedulePoll) would break such a cycle, but is not safe here:
// this runs mid-marshalling inside the caller's entry method, and re-entry
// could deliver another message to the very chare that is partway through a
// send.
//
// A direct transfer needs only the event pair -- it reads the sender's own
// allocation, so there is no block to reserve -- and so cannot be held up by
// comm-buffer exhaustion at all. It still needs the events: src_event orders
// the receiver's copy after the kernels that produced the data, and dst_event
// is how the sender learns the receiver has finished reading a buffer it does
// not own.
//
// Reclaiming is a *failure* path, not part of every send. The scan costs a
// driver query per busy slot in the slice, so running it up front made every
// message pay for garbage collection it usually did not need -- and the cost
// grew with +gpuipceventpool, which is exactly the knob a send-heavy run turns
// up. The invariant that matters is only that no attempt reports exhaustion
// without having reclaimed first (reclaiming solely *after* a failed
// allocation, as the original code did, let comm-buffer exhaustion abort
// without ever running the scan that frees comm buffers). Trying first and
// reclaiming on failure preserves that: the slow path below always reclaims
// before looping, and the timeout can only be reached through it.
//
// wait == false: one try, one reclaim, one more try, then false instead of the
// loop. That is how the deferrable send path asks (ipcPrepareBuffers, the IPC
// slot queue below it): a PE that spins here receives nothing, and receiving
// is what releases every other PE's slots.
static double ipcSlotTimeoutSecs() {
  static const double timeout_s = []() {
    const char* s = getenv("CHARM_IPC_SLOT_TIMEOUT");
    return s ? atof(s) : 60.0;
  }();
  return timeout_s;
}

static bool acquireIpcSendSlot(DeviceManager* dm, int cpv_my_device_id,
                               bool is_lb_buffer, bool direct,
                               const void* src_ptr,
                               size_t cnt, void** out_buffer,
                               int* out_event_idx, bool wait) {
  const double timeout_s = ipcSlotTimeoutSecs();

  // CHARM_IPC_LAZY_RECLAIM=1 takes the reclaim scan off the fast path (try
  // first, reclaim only on failure). Opt-in rather than default: see the
  // discussion at CkRdmaDeviceAllocLbBuffer -- other allocators of this buffer
  // depend on the sweep running often, and the scan is cheap enough now that
  // dst_flag is a plain atomic that removing it from the send path buys much
  // less than it did.
  static const bool lazy = (getenv("CHARM_IPC_LAZY_RECLAIM") != nullptr);

  double wait_start = 0.0;
  bool waiting = false;
  int attempts = 0;
  bool first = lazy;
  // Which resource came up short last time, so the next reclaim knows whether
  // one slot is enough (event pool drained) or it must sweep for bytes (comm
  // buffer drained). Always set by the attempt that precedes each reclaim, so
  // the initializer here is only to keep it defined.
  bool need_slot_only = false;

  for (;;) {
    if (!first) {
      // Bounded when the event pool is what ran dry: one freed slot is all this
      // send can use. Unbounded when it is bytes, since how many blocks must
      // come back to fit cnt is not knowable in advance.
      reclaimCompletedIpcEvents(dm, cpv_my_device_id, need_slot_only ? 1 : 0);
    }
    first = false;

    void* buf = nullptr;
    int ev = -1;

    if (direct) {
      // No block to reserve, so nothing to hand back if the event pool is
      // empty; only the events can hold this up.
      ev = claimFreeIpcEvent(cpv_my_device_id, 0, 2);
      need_slot_only = true;
    } else if (is_lb_buffer) {
      // Already in the comm buffer; nothing to allocate, so only the event pool
      // can hold this up.
      buf = const_cast<void*>(src_ptr);
      const size_t off = (char*)buf - (char*)dm->comm_buffer->base_ptr;
      ev = claimFreeIpcEvent(cpv_my_device_id, off, 1);
      if (ev == -1) buf = nullptr;
      need_slot_only = true;
    } else {
#if CMK_SMP
      CmiLock(dm->lock);
#endif
      buf = dm->alloc_comm_buffer(cnt);
#if CMK_SMP
      CmiUnlock(dm->lock);
#endif
      if (buf == nullptr) {
        need_slot_only = false;  // short of bytes
      } else {
        const size_t off = (char*)buf - (char*)dm->comm_buffer->base_ptr;
        ev = claimFreeIpcEvent(cpv_my_device_id, off, 1);
        if (ev == -1) {
          // Got a block but no event. Hand the block back before waiting:
          // holding half the pair while blocked lets two senders pin each
          // other's missing half indefinitely.
#if CMK_SMP
          CmiLock(dm->lock);
#endif
          dm->free_comm_buffer(off);
#if CMK_SMP
          CmiUnlock(dm->lock);
#endif
          buf = nullptr;
          need_slot_only = true;  // bytes were there, the slot was not
        }
      }
    }

    if (ev != -1) {
      *out_buffer = buf;
      *out_event_idx = ev;
      return true;
    }

    // A caller that can park its send instead: it has had a try and (on the
    // next pass) a reclaim; report the shortage rather than spin.
    if (!wait) {
      if (++attempts >= 2) return false;
      continue;
    }

    if (!waiting) {
      wait_start = CkWallTimer();
      waiting = true;
    } else if (CkWallTimer() - wait_start > timeout_s) {
      if (dm->comm_buffer == nullptr)   // +gpupool: slots are the only resource
        CkAbort("PE %d, device %d: no free CUDA IPC event slot after %.0fs. "
                "If every peer is blocked here too, the transfers that would "
                "release these slots cannot run; reduce concurrent device "
                "sends, or raise +gpuipceventpool.",
                CkMyPe(), dm->global_index, timeout_s);
      CkAbort("PE %d, device %d: no free CUDA IPC event/comm-buffer slot after "
              "%.0fs (comm buffer: %zu bytes free). If every peer is blocked "
              "here too, the transfers that would release these slots cannot "
              "run; reduce concurrent device sends, or raise +gpucommbuffer "
              "(raising +gpuipceventpool costs device memory at init).",
              CkMyPe(), dm->global_index, timeout_s,
              dm->get_comm_buffer_free_size());
    }

    // Back off so the peer processes that release these slots get the core,
    // and so this does not spin on the allocator lock that other PEs sharing
    // this device need.
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 50000;  // 50us
    nanosleep(&ts, nullptr);
  }
}

// Allocate from the device load-balance region, reclaiming and retrying once if
// the first attempt does not fit. See the declaration in ckrdmadevice.h.
//
// Blocks in this region are handed back only by reclaimCompletedIpcEvents: a
// migration's packed payload is released when the receiver acknowledges the
// staged transfer that carried it, through the same slot machinery an ordinary
// cross-process send uses. Nothing on the migration path calls
// acquireIpcSendSlot, so without this the scan would never run on its behalf.
//
// This used to work by accident. The scan sat unconditionally at the top of
// every acquireIpcSendSlot, so an application doing ordinary device sends swept
// the slice often enough that migration always found free blocks waiting.
// Taking that sweep off the send fast path removed the accident and made the
// dependency explicit: the reclaim now happens where the memory is actually
// needed, which is also better timed than sweeping on every unrelated send.
// One reusable event per (device, freeing stream). A block's previous work is
// always on some stream; recording there and waiting on it from the consumer
// is the exact ordering the host barrier was standing in for. Guarded by the
// device manager's own lock, which is what serializes the pool itself.
namespace {
// Keyed by the stream AND the device the entry was made on. A raw
// cudaStream_t is not a stable identity: when an element migrates away its
// stream is destroyed (or pooled), and the next stream created on this PE --
// or an element that arrives and takes the pooled handle -- can carry the very
// same handle value. Keying on the handle alone then returns the departed
// element's event, which was created on whatever device THAT PE had current.
// An event and a stream on different devices is cudaErrorInvalidResourceHandle,
// and because nothing here checks the result the error becomes this thread's
// sticky error and aborts whichever unrelated chare next peeks at it.
// Including the device in the key keeps the two apart; the gate below then
// only waits on events belonging to the consumer's own device, which is the
// only wait that is valid anyway.
struct LbRetireKey {
  cudaStream_t stream;
  int device;
  bool operator==(const LbRetireKey& o) const {
    return stream == o.stream && device == o.device;
  }
};
struct LbRetireKeyHash {
  size_t operator()(const LbRetireKey& k) const {
    return std::hash<void*>()((void*)k.stream) ^ (std::hash<int>()(k.device) << 1);
  }
};
typedef std::unordered_map<LbRetireKey, cudaEvent_t, LbRetireKeyHash> LbRetireEvents;
std::unordered_map<void*, LbRetireEvents> lb_retire_events;
// A std::mutex, not a CmiNodeLock created on first use: "if NULL, create" races
// when two PEs arrive together (see hapi_stream_pool_lock).
std::mutex lb_retire_lock;
}  // namespace


void CkRdmaDeviceNoteLbBufferFreed(void* dm_opaque, cudaStream_t usedBy) {
  if (dm_opaque == NULL) return;
  lb_retire_lock.lock();
  int dev = -1;
  hapiCheck(cudaGetDevice(&dev));
  LbRetireEvents& evs = lb_retire_events[dm_opaque];
  const LbRetireKey key{usedBy, dev};
  auto it = evs.find(key);
  if (it == evs.end()) {
    cudaEvent_t e;
    // Disable timing: this event is only ever waited on, and a timing-enabled
    // event costs more to record.
    hapiCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    it = evs.emplace(key, e).first;
  }
  // Re-recording overwrites the previous capture, which is correct: work on one
  // stream is ordered, so the latest record subsumes every earlier one.
  hapiCheck(cudaEventRecord(it->second, usedBy));
  lb_retire_lock.unlock();
}

void CkRdmaDeviceGateLbBuffer(void* dm_opaque, cudaStream_t consumer) {
  if (dm_opaque == NULL) return;
  lb_retire_lock.lock();
  auto dit = lb_retire_events.find(dm_opaque);
  if (dit != lb_retire_events.end()) {
    int dev = -1;
    hapiCheck(cudaGetDevice(&dev));
    for (auto& kv : dit->second) {
      if (kv.first.device != dev) continue;   // another device's event
      if (kv.first.stream == consumer) continue;  // same stream, already ordered
      hapiCheck(cudaStreamWaitEvent(consumer, kv.second, 0));
    }
  }
  lb_retire_lock.unlock();
}

// ---- Device pool: thin wrappers over hapi's (see hapi.h) ----------------------
void* CkDeviceMalloc(size_t size) {
  return hapiDevPoolMalloc(size, CpvAccess(my_device_id));
}
void* CkDeviceMallocNoGrow(size_t size) {
  return hapiDevPoolMallocNoGrow(size, CpvAccess(my_device_id));
}
void CkDeviceFree(void* ptr) { hapiDevPoolFree(ptr); }
bool CkDevicePoolOn() { return hapiDevPoolOn(); }

static thread_local bool ck_sending_migration_payload = false;
void CkRdmaDeviceMarkMigrationPayload(bool sending) {
  ck_sending_migration_payload = sending;
}

void* CkRdmaDeviceAllocLbBuffer(void* dm_opaque, size_t size) {
  DeviceManager* dm = (DeviceManager*)dm_opaque;

  auto attempt = [&]() {
#if CMK_SMP
    CmiLock(dm->lock);
#endif
    void* p = dm->alloc_comm_buffer(size, false);
#if CMK_SMP
    CmiUnlock(dm->lock);
#endif
    return p;
  };

  void* p = attempt();
  if (p != nullptr) return p;

  if (!CsvAccess(gpu_manager).use_shm) return nullptr;
  if (reclaimCompletedIpcEvents(dm, CpvAccess(my_device_id), 0) == 0) return nullptr;

  return attempt();
}

// Device payload of the zerocopy send currently being marshalled, in bytes.
//
// The load balancer's communication graph weights each edge by
// UsrToEnv(msg)->getTotalsize(), but a device zerocopy message carries only the
// CkDeviceBuffer descriptors in its envelope -- a few hundred bytes standing in
// for a transfer that is routinely megabytes. Weighted that way, GPU-to-GPU
// edges are effectively invisible to any communication-aware strategy.
//
// The real sizes are known here and nowhere later on the send path, so they are
// parked per PE and picked up by CkArray::sendToPe. The generated code between
// this call and that one is straight-line marshalling with no intervening entry
// method, so the value cannot be interleaved with another send on this PE.
// CkArray::sendToPe takes and clears it unconditionally, so a stale value
// cannot outlive one array send.
//
// Limitation: group and nodegroup device sends never reach CkArray::sendToPe,
// so their value is left for the next array send on that PE to discard. The LB
// only builds object-to-object edges from array elements, so this costs
// nothing today, but it is why the take-and-clear must stay unconditional.
static thread_local size_t _ck_pending_device_send_bytes = 0;

// How many of this PE's IPC event slots are currently claimed. A slot is
// released only when the receiving process signals it took the staged bytes,
// so a count that climbs across load balancing rounds and never comes back
// down means transfers staged before a migration are never being acknowledged.
int CkRdmaDeviceBusyIpcSlots() {
  GPUManager& gm = CsvAccess(gpu_manager);
  if (!gm.use_shm) return -1;
  const int pool_size = gm.hapi_ipc_event_pool_size_pe;
  const int pool_start = ipcEventPoolSlice(CpvAccess(my_device_id)) * pool_size;
  const int idx = gm.device_count * CmiMyNodeRankLocal() + CpvAccess(my_device_id);
  if (idx < 0 || (size_t)idx >= gm.hapi_ipc_device_infos.size()) return -1;
  hapi_ipc_device_info& info = gm.hapi_ipc_device_infos[idx];
  if ((size_t)(pool_start + pool_size) > info.event_pool_flags.size()) return -1;
  int busy = 0;
  for (int i = pool_start; i < pool_start + pool_size; i++)
    if (info.event_pool_flags[i] != 0) busy++;
  return busy;
}

size_t CkRdmaDeviceTakePendingSendBytes() {
  const size_t bytes = _ck_pending_device_send_bytes;
  _ck_pending_device_send_bytes = 0;
  return bytes;
}

// Releases the element that issued an inter-node zerocopy send once the runtime
// has finished reading its source buffer, then hands control to the callback the
// application attached. A callCFn callback records the PE that built it and
// CkCallback::send routes back there, so this runs on the sending PE.
struct DeviceSendRelease { CkLocRec* rec; CkCallback app_cb; };

static void deviceSendReleaseFn(void* param, void* msg)
{
  DeviceSendRelease* r = (DeviceSendRelease*)param;
  CkLocRec* rec = r->rec;
  CkCallback cb = r->app_cb;
  delete r;
  if (cb.type != CkCallback::ignore) cb.send(msg);
  else if (msg) CkFreeMsg(msg);
  if (rec) rec->noteDeviceSendDone();  // may start a deferred migration
}

// Only network/fallback sends need a host-side producer completion. Keep the
// descriptors by value: the generated proxy's CkDeviceBuffers are stack locals.
// Nothing in this state is sent over the wire.
struct CkDeviceDeferredSend {
  std::vector<CkDeviceBuffer> buffers;
  CkLocRec* owner;
  Chare* sourceObject;
  int sourcePe;
  size_t bytes = 0;
  size_t descriptorBytes = 0;
  size_t remaining = 0;
  void* msg = nullptr;
  std::function<void()> send;

  // A send parked for a CUDA IPC event slot (see the IPC slot queue below):
  // the destination its prepare is for, whether it is the runtime's own
  // migration payload, one event per buffer marking the producer's position
  // when the send was made, and when it was parked.
  bool slotWait = false;
  int destPe = -1;
  bool migrationPayload = false;
  std::vector<hapiEvent_t> producerEvents;
  double parkedAt = 0.0;
  bool warned = false;

  explicit CkDeviceDeferredSend(CkLocRec* rec)
      : owner(rec), sourceObject(rec ? CkActiveObj() : nullptr), sourcePe(CkMyPe()) {}
};

// What pass one of ipcPrepareBuffers decided and acquired for one buffer.
struct IpcPrep {
  bool live = false;
  bool is_lb_buffer = false;
  bool direct = false;
  hapiIpcMemHandle_t export_handle;
  size_t export_offset = 0;
  void* export_base = nullptr;
  void* block = nullptr;   // staged: the comm-buffer block
  int event_idx = -1;
};

// Hand back a slot (and block) that pass one acquired for a message that is
// not going to be prepared now. Nothing was published for it, so no receiver
// will ever raise its flag: it goes straight back, not through the reclaim.
static void releaseUnusedIpcSlot(DeviceManager* dm, hapi_ipc_device_info& info,
                                 const IpcPrep& p)
{
  if (p.event_idx < 0) return;
  info.event_pool_flags[p.event_idx] = 0;
  info.event_pool_buff_offsets[p.event_idx] = 0;
  if (!p.direct && !p.is_lb_buffer && p.block != nullptr) {
#if CMK_SMP
    CmiLock(dm->lock);
#endif
    dm->free_comm_buffer((size_t)((char*)p.block - (char*)dm->comm_buffer->base_ptr));
#if CMK_SMP
    CmiUnlock(dm->lock);
#endif
  }
}

// The sender side of a cross-process (CUDA IPC) send: for every live buffer
// of one message, its transport, a slot from this PE's slice, the staging
// copy if it stages, the migration hold if it is direct, and the event the
// receiver waits on. Two passes, so that a caller that may not block (wait ==
// false) learns of a slot shortage before anything has been copied, held or
// recorded, and can park the whole message: it returns false then, with
// every slot the message had already taken handed back. With wait set it
// waits for slots as acquireIpcSendSlot always has.
//
// `deferred` is the parked record when this is the retry from the slot
// queue. Its buffers then carry the producer position captured when the send
// was made, and everything in pass two goes on the PE's deferred-send stream
// behind that position, rather than on the application's stream.
static bool ipcPrepareBuffers(int dest_pe, int numops, CkDeviceBuffer** buffers,
                              bool wait, CkDeviceDeferredSend* deferred);

// ---- IPC slot queue ------------------------------------------------------------
//
// A cross-process device send holds one of this PE's CUDA IPC event slots
// until the receiver has read the payload. acquireIpcSendSlot waited for a
// free one by spinning, which is safe only while the PEs that release slots
// keep running -- and they release them by RECEIVING: a receiver claims no
// slot of its own, but its deviceIpcReceive is what raises dst_flag on the
// sender's. A PE spinning inside a send receives nothing. So once every PE
// was short at the same moment nothing could progress, and each aborted 60 s
// later with "no free CUDA IPC event slot". leanmd after MetisLB is the case:
// the repartition makes most of a step's device sends cross-process, several
// hundred per PE against a 256-slot slice, and all 32 PEs went short in the
// first step after the balance -- with every migration window already empty.
//
// A send made through the deferrable proxy (CkRdmaDeviceSendWhenReady) is
// therefore never blocked for a slot. When its slots are not there after one
// reclaim, or other sends are already waiting, it is parked here in order
// and retried from a timer callback, so this PE keeps receiving while it
// waits; that is what lets its peers' sends -- and, through them, its own --
// complete. The producer's position is captured at send time in a private
// event, and the deferred prepare orders the IPC event behind exactly that
// on a dedicated stream, not behind whatever the application queued on its
// own stream afterwards: a stream that goes on to wait for the receiver's
// reply would otherwise put that reply and this read in a cycle on the GPU.
// The sending element is held against migration from the moment it is
// parked, as it is for a deferred network send.
//
// The three-argument prepare (sync entries) and the correction paths keep the
// blocking acquire. CHARM_IPC_SLOT_TIMEOUT (default 60 s) is a warning here,
// since the wait makes progress; CHARM_DEBUG_IPCQ traces each park and send.
struct CkIpcSlotQueue {
  std::deque<CkDeviceDeferredSend*> queue;
  std::vector<hapiEvent_t> spareEvents;
  hapiStream_t stream = nullptr;   // where deferred prepares are ordered
  bool pumpScheduled = false;
  bool reported = false;
  size_t parked = 0;
  size_t maxDepth = 0;
};
static thread_local CkIpcSlotQueue ipc_slot_queue;

static inline bool ipcQueueDbg()
{
  static const bool on = (getenv("CHARM_DEBUG_IPCQ") != nullptr);
  return on;
}

static hapiEvent_t ipcTakeProducerEvent()
{
  CkIpcSlotQueue& q = ipc_slot_queue;
  if (!q.spareEvents.empty()) {
    hapiEvent_t e = q.spareEvents.back();
    q.spareEvents.pop_back();
    return e;
  }
  hapiEvent_t e;
  hapiCheck(hapiEventCreateWithFlags(&e, hapiEventDisableTiming));
  return e;
}

// Non-blocking: the per-thread default stream synchronizes with the legacy
// one, and a landing or restage there has nothing to do with these sends.
static hapiStream_t ipcDeferredStream()
{
  CkIpcSlotQueue& q = ipc_slot_queue;
  if (q.stream == nullptr)
    hapiCheck(cudaStreamCreateWithFlags(&q.stream, cudaStreamNonBlocking));
  return q.stream;
}

static void ipcSlotPump(void*, double);

static void ipcSlotPumpSchedule(double msecs)
{
  if (ipc_slot_queue.pumpScheduled) return;
  ipc_slot_queue.pumpScheduled = true;
  CcdCallFnAfter(ipcSlotPump, nullptr, msecs);
}

// Builds the parked record for a message whose slots were not there. Every
// live buffer's producer position is captured now, and its element held.
static CkDeviceDeferredSend* ipcParkSend(int dest_pe, int numops,
                                         CkDeviceBuffer** buffers, size_t bytes)
{
  CkLocRec* rec = CkpvAccess(_currentLocRec);
  CkDeviceDeferredSend* p = new CkDeviceDeferredSend(rec);
  p->slotWait = true;
  p->destPe = dest_pe;
  p->migrationPayload = ck_sending_migration_payload;
  p->bytes = bytes;
  p->parkedAt = CkWallTimer();
  p->producerEvents.assign(numops, nullptr);
  for (int i = 0; i < numops; i++) {
    CkDeviceBuffer* b = buffers[i];
    if (b->cnt != 0) {
      hapiEvent_t e = ipcTakeProducerEvent();
      if (hapiEventRecord(e, b->hapi_stream) == hapiSuccess) {
        p->producerEvents[i] = e;
      } else {
        // The buffer's stream is on another device (see the cross-device
        // record in ipcPrepareBuffers): settle it here instead, as that does.
        cudaGetLastError();
        ipc_slot_queue.spareEvents.push_back(e);
        hapiCheck(ckSubmitDrainThenSync(b->hapi_stream));
      }
      // Held from now: the buffer must outlive the deferred prepare (staged)
      // or the receiver's read (direct). The deferred prepare runs as the
      // runtime and takes no hold of its own.
      if (rec) {
        rec->noteDeviceSendPosted();
        b->cb = CkCallback(deviceSendReleaseFn,
                           (void*)new DeviceSendRelease{rec, b->cb});
      }
    }
    p->buffers.push_back(*b);
  }
  return p;
}

// Rewrites the descriptor prefix of the held message from the (now prepared)
// buffers, sends it with the original entry's attribution restored, and frees
// the record. Shared by the producer-ready and slot-wait paths.
static void deviceDeferredPublish(CkDeviceDeferredSend* pending)
{
  // Device descriptors occupy a fixed-width prefix of the marshalled message.
  // Fill in the registrations without touching scalar/array arguments after it.
  PUP::sizer size;
  int numops = pending->buffers.size();
  size | numops;
  for (auto& buffer : pending->buffers) size | buffer;
  if (size.size() != pending->descriptorBytes)
    CkAbort("Deferred device send changed its descriptor prefix size");
  PUP::toMem pack(((CkMarshallMsg*)pending->msg)->msgBuf);
  pack | numops;
  for (auto& buffer : pending->buffers) pack | buffer;

  // The original entry method has returned. Restore its attribution just for
  // the send, so LB records an object-to-object edge and the actual device bytes.
  // The outstanding-send references keep this source object stationary/live.
  CkLocRec* savedRec = CkpvAccess(_currentLocRec);
  const size_t savedBytes = _ck_pending_device_send_bytes;
  CkpvAccess(_currentLocRec) = pending->owner;
  _ck_pending_device_send_bytes = pending->bytes;
  if (pending->sourceObject) CkCallstackPush(pending->sourceObject);
  pending->send();
  if (pending->sourceObject) CkCallstackPop(pending->sourceObject);
  _ck_pending_device_send_bytes = savedBytes;
  CkpvAccess(_currentLocRec) = savedRec;
  delete pending;
  // The buffer callbacks, now in the message, release the owner after the
  // network read completes. Producer readiness must not release those holds.
}

// Prepares and sends parked messages, in order, while their slots are there.
static void ipcSlotPump(void*, double)
{
  CkIpcSlotQueue& q = ipc_slot_queue;
  q.pumpScheduled = false;
  while (!q.queue.empty()) {
    CkDeviceDeferredSend* p = q.queue.front();
    std::vector<CkDeviceBuffer*> bufs(p->buffers.size());
    for (size_t i = 0; i < p->buffers.size(); i++) bufs[i] = &p->buffers[i];
    // As the runtime, not the element: the hold was taken when the send was
    // parked and must not be taken twice.
    CkLocRec* savedRec = CkpvAccess(_currentLocRec);
    CkpvAccess(_currentLocRec) = nullptr;
    const bool savedPayload = ck_sending_migration_payload;
    ck_sending_migration_payload = p->migrationPayload;
    const bool ok = ipcPrepareBuffers(p->destPe, (int)bufs.size(), bufs.data(),
                                      /*wait=*/false, p);
    ck_sending_migration_payload = savedPayload;
    CkpvAccess(_currentLocRec) = savedRec;
    if (!ok) {
      const double waited = CkWallTimer() - p->parkedAt;
      if (waited > ipcSlotTimeoutSecs() && !p->warned) {
        p->warned = true;
        CmiPrintf("[%d] WARNING: a %zu-byte device send to PE %d has waited "
                  "%.0f s for a CUDA IPC event slot (%zu parked behind it, %d "
                  "of this PE's slots busy). The receivers of this PE's "
                  "in-flight sends are not reading them.\n",
                  CkMyPe(), p->bytes, p->destPe, waited, q.queue.size() - 1,
                  CkRdmaDeviceBusyIpcSlots());
        fflush(stdout);
      }
      ipcSlotPumpSchedule(0.05);
      return;
    }
    q.queue.pop_front();
    for (hapiEvent_t e : p->producerEvents)
      if (e != nullptr) q.spareEvents.push_back(e);  // its wait is enqueued
    if (ipcQueueDbg())
      CmiPrintf("[IPCQ %d] send %zu bytes to PE %d after %.2f ms; %zu still parked\n",
                CkMyPe(), p->bytes, p->destPe,
                (CkWallTimer() - p->parkedAt) * 1e3, q.queue.size());
    QdProcess(1);
    deviceDeferredPublish(p);
  }
}

static bool ipcPrepareBuffers(int dest_pe, int numops, CkDeviceBuffer** buffers,
                              bool wait, CkDeviceDeferredSend* deferred)
{
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  const int cpv_my_device_id = CpvAccess(my_device_id);
  DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
  const int my_device_idx =
      csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
  hapi_ipc_device_info& my_device_info =
      csv_gpu_manager.hapi_ipc_device_infos[my_device_idx];

  std::vector<IpcPrep> prep(numops);

  // Pass one: the transport of each live buffer, then its slot.
  for (int i = 0; i < numops; i++) {
    if (buffers[i]->cnt == 0) continue;  // empty: completed by the caller
    IpcPrep& p = prep[i];
    p.live = true;
    // Pool mode has no comm buffer, so nothing is "in the LB region".
    p.is_lb_buffer = dm->comm_buffer != nullptr &&
        ( (size_t)((char*)(buffers[i]->ptr) - (char*)(dm->comm_buffer->base_ptr)) < dm->comm_buffer->total_size );

    // Choose the transport for this buffer. Per buffer, not per message: one
    // entry method can legitimately carry payloads on both sides of the
    // threshold.
    //
    // A buffer that already lives in the comm buffer stays staged whatever
    // its size -- staging it costs nothing, since there is no copy to make,
    // and the receiver reads it through the mapping every peer already holds.
    // Otherwise the direct transport applies when the run asked for it,
    // unless the source allocation turns out not to be exportable, in which
    // case hapiIpcExportBuffer says so and this falls back to staging.
    if (!p.is_lb_buffer && hapiIpcUseDirect()) {
      p.direct = hapiIpcExportBuffer(buffers[i]->ptr, &p.export_handle,
                                     &p.export_offset, &p.export_base);
    }
    // Under +gpupool the export is the only route: there is no comm buffer
    // to fall back to. A buffer cudaIpcGetMemHandle will not name -- managed
    // memory, a host allocation, an address the driver does not own -- is a
    // contract violation there, not a slower path.
    if (!p.direct && !p.is_lb_buffer && hapiDevPoolOn()) {
      CkAbort("PE %d: a %zu-byte device buffer at %p could not be exported "
              "for direct CUDA IPC (%s). Under +gpupool every device buffer "
              "sent must be cudaMalloc-backed: from CkDeviceMalloc or "
              "hapiMalloc.",
              CkMyPe(), (size_t)buffers[i]->cnt, buffers[i]->ptr,
              hapiIpcLastImportErrorName());
    }

    // A zero-copy device send may not reuse or free its source buffer until
    // the CkDeviceBuffer's completion callback fires. That is the contract
    // whatever transport carries it, so a send without a callback is already
    // an application bug -- staging merely hides it, since the source happens
    // to be free once the staging copy retires on the sender's own stream.
    // Direct reads the sender's allocation itself, so the same bug shows up
    // as corrupted data instead. Report it once, where the send is still
    // identifiable, rather than let it surface there.
    // The runtime's own migration payload is the one exception: it is
    // released by the receiver's ack, not by a callback.
    if (p.direct && buffers[i]->cb.type == CkCallback::ignore &&
        !ck_sending_migration_payload) {
      static std::atomic<bool> warned{false};
      bool expected = false;
      if (warned.compare_exchange_strong(expected, true)) {
        CmiPrintf("[%d] WARNING: a %zu-byte device buffer is being sent over "
                  "direct CUDA IPC with no completion callback. Every "
                  "zero-copy device send needs one: attach a CkCallback to "
                  "the CkDeviceBuffer and leave the buffer alone until it "
                  "fires. Staging happens to tolerate a missing callback, so "
                  "forcing these sends back to staging would hide this rather "
                  "than fix it.\n",
                  CkMyPe(), (size_t)buffers[i]->cnt);
        fflush(stdout);
      }
    }
  }
  for (int i = 0; i < numops; i++) {
    IpcPrep& p = prep[i];
    if (!p.live) continue;
    // Waits for a slot if the pools are momentarily drained, or reports the
    // shortage when the caller can park instead; takes and releases dm->lock
    // itself.
    if (!acquireIpcSendSlot(dm, cpv_my_device_id, p.is_lb_buffer, p.direct,
                            buffers[i]->ptr, buffers[i]->cnt, &p.block,
                            &p.event_idx, wait)) {
      // Give back what this message had already taken: holding part of a
      // message's slots while parked would let two short PEs pin each
      // other's remainder.
      for (int j = 0; j < i; j++) releaseUnusedIpcSlot(dm, my_device_info, prep[j]);
      return false;
    }
  }

  // Pass two. A deferred prepare orders behind the producer position captured
  // when the send was made (see the IPC slot queue), on the deferred stream.
  if (deferred) {
    hapiStream_t ds = ipcDeferredStream();
    for (int i = 0; i < numops; i++) {
      if (!prep[i].live) continue;
      if (deferred->producerEvents[i] != nullptr)
        hapiCheck(hapiStreamWaitEvent(ds, deferred->producerEvents[i], 0));
      buffers[i]->hapi_stream = ds;
    }
  }
  for (int i = 0; i < numops; i++) {
    IpcPrep& p = prep[i];
    if (!p.live) continue;
    void* alloc_comm_buffer = p.block;
    const int acquired_event_idx = p.event_idx;
    const bool direct = p.direct;
    const bool is_lb_buffer = p.is_lb_buffer;
    if (direct) {
      buffers[i]->ipc_protocol = CmiIpcProtocol::DIRECT;
      buffers[i]->ipc_handle = p.export_handle;
      buffers[i]->ipc_offset = p.export_offset;
      buffers[i]->ipc_base = p.export_base;
      buffers[i]->comm_offset = 0;
      csv_gpu_manager.ipc_direct_sends.fetch_add(1, std::memory_order_relaxed);

      // Direct exports the application's live allocation and ships the peer a
      // handle to it, so the buffer has to outlive the peer's read -- exactly
      // the condition the send interlock exists for, and the same reason the
      // memcpy and inter-node paths register above and below. This path was
      // added later and never did, so emigrate saw no outstanding sends,
      // migrated the element, and freed the allocation whose handle was
      // already on the wire; the peer then opened a dead handle and aborted
      // with "could not open the CUDA IPC handle".
      //
      // Staged needs none of this: it copies into the comm buffer, so the
      // element's own buffer is free the moment that copy retires.
      // A parked send took its hold when it was parked (ipcParkSend), and
      // runs here with no element current.
      CkLocRec* direct_rec = CkpvAccess(_currentLocRec);
      if (direct_rec) {
        direct_rec->noteDeviceSendPosted();
        buffers[i]->cb = CkCallback(deviceSendReleaseFn,
                                    (void*)new DeviceSendRelease{direct_rec,
                                                                 buffers[i]->cb});
      }
    } else {
      buffers[i]->ipc_protocol = CmiIpcProtocol::STAGED;
      buffers[i]->comm_offset = (char*)alloc_comm_buffer - (char*)dm->comm_buffer->base_ptr;
      csv_gpu_manager.ipc_staged_sends.fetch_add(1, std::memory_order_relaxed);
    }
    buffers[i]->device_idx = my_device_idx;
    buffers[i]->event_idx = acquired_event_idx;
    buffers[i]->sender_prepared = true;

    // TEMPORARY: paired with the receive-side print, so the indices and
    // offsets the sender publishes can be compared against what the receiver
    // resolves them to.
    if (ipcDebugOn()) {
      CmiPrintf("[%d] IPC send: dev_idx=%d ev_idx=%d off=%zu cnt=%zu "
                "src_ptr=%p comm_base=%p alloc=%p is_lb=%d dest_pe=%d\n",
                CkMyPe(), buffers[i]->device_idx, buffers[i]->event_idx,
                (size_t)buffers[i]->comm_offset, (size_t)buffers[i]->cnt,
                buffers[i]->ptr,
                dm->comm_buffer ? (void*)dm->comm_buffer->base_ptr : nullptr,
                alloc_comm_buffer, (int)is_lb_buffer, dest_pe);
      fflush(stdout);
    }

    // Initiate transfer from source buffer to device comm buffer. A direct
    // transfer has no comm buffer to fill -- that saved copy is the point of
    // it -- and an LB buffer is already in place.
    if(!is_lb_buffer && !direct) {
      // CHARM_ZC_VALIDATE: the buffer being staged belongs to the sending
      // chare. If that chare has migrated, its device scratch may already be
      // freed (or not yet reallocated on the new PE) while a send referencing
      // it is still being marshalled -- report that here rather than letting
      // it surface asynchronously somewhere unrelated.
      if (zcValidateOn()) {
        cudaPointerAttributes sattr{};
        const cudaError_t serr = cudaPointerGetAttributes(&sattr, buffers[i]->ptr);
        if (serr != cudaSuccess || sattr.type == cudaMemoryTypeUnregistered) {
          CmiPrintf("[%d] ZC SEND VALIDATE FAIL src=%p (type=%d err=%d) cnt=%zu "
                    "dest_pe=%d mode=IPC\n",
                    CkMyPe(), buffers[i]->ptr, (int)sattr.type, (int)serr,
                    (size_t)buffers[i]->cnt, dest_pe);
          fflush(stdout);
          cudaGetLastError();
        }
      }
      hapiCheck(hapiMemcpyAsync(alloc_comm_buffer, buffers[i]->ptr, buffers[i]->cnt,
            hapiMemcpyDeviceToDevice, buffers[i]->hapi_stream));
      ipcDebugSync("send 1: stage src -> comm_buffer", buffers[i]->hapi_stream);

      // The completion callback's contract is "the source buffer is safe to
      // reuse", and for a staged send that is the moment the staging copy
      // above retires -- not when the receiver finishes reading the staged
      // block, a full IPC round later. Fire it here on the sender's stream
      // and ship an ignore callback, so the receiver does not fire it a
      // second time. Every delivery of a STAGED payload reads the staged
      // block rather than the source buffer (including same-process
      // deliveries: a process's own devices are self-mapped in the comm
      // buffer table), so nothing downstream depends on the source after
      // this copy.
      if (buffers[i]->cb.type != CkCallback::ignore) {
        hapiAddCallback(buffers[i]->hapi_stream, buffers[i]->cb);
        buffers[i]->cb = CkCallback(CkCallback::ignore);
      }
    }

    // Record the event the receiver waits on before it reads. Staged, that
    // marks the staging copy as landed; direct, it marks the kernels that
    // produced the source buffer as retired. Either way it is recorded on the
    // application's own stream, so it sits after whatever produced the data.
    // The event comes from THIS PE's pool, so it belongs to this PE's device,
    // and an event can only be recorded on a stream of its own device. The
    // buffer's stream is not always ours: forward-time repair re-prepares a
    // payload on the forwarding PE while the buffer still carries the stream
    // of the PE that built it, and load balancing can put those on different
    // GPUs. Recording across devices fails with invalid-resource-handle.
    //
    // Same remedy the cross-device cases above use: settle that stream on the
    // host and record on ours instead. Stronger ordering than the stream
    // record, and confined to the case where the two devices differ -- which
    // only became reachable once chares started moving between GPUs.
    const int buf_dev = hapiStreamDeviceOf(buffers[i]->hapi_stream);
    if (buf_dev >= 0 && buf_dev != hapiGetDeviceNum()) {
      hapiCheck(ckSubmitDrainThenSync(buffers[i]->hapi_stream));
      hapiCheck(hapiEventRecord(
          my_device_info.src_event_pool[buffers[i]->event_idx], NULL));
      ipcPublishSrcReadyNow(my_device_idx, buffers[i]->event_idx);
    } else {
      hapiCheck(hapiEventRecordNoted(
          my_device_info.src_event_pool[buffers[i]->event_idx],
          buffers[i]->hapi_stream,
          my_device_idx * 100000 + buffers[i]->event_idx));
      ipcPublishSrcReady(my_device_idx, buffers[i]->event_idx, buffers[i]->hapi_stream);
    }
    ipcDebugSync("send 2: record own src_event", buffers[i]->hapi_stream);
  }
  return true;
}

static void deviceSendProducerReady(void* arg, void*) {
  auto* pending = static_cast<CkDeviceDeferredSend*>(arg);
  CkAssert(CkMyPe() == pending->sourcePe);
  CkAssert(pending->remaining > 0);
  if (--pending->remaining != 0) return;

  // HAPI invokes this on the issuing PE, not on the CUDA host-function thread.
  // Registration and metadata publication happen only after every producer.
  for (auto& buffer : pending->buffers) {
    if (buffer.cnt == 0) continue;  // empty: prepared as such by the sender
    buffer.lci_ncpy_buffer =
        acquireDeviceRegistration(buffer.ptr, buffer.cnt, pending->owner);
    buffer.sender_prepared = true;
  }
  deviceDeferredPublish(pending);
}

void CkRdmaDeviceSendWhenReady(CkDeviceDeferredSend* pending, void* msg,
                              const std::function<void()>& send) {
  CkAssert(pending != nullptr && !pending->buffers.empty());
  pending->msg = msg;
  pending->send = send;
  PUP::sizer size;
  int numops = pending->buffers.size();
  size | numops;
  for (auto& buffer : pending->buffers) size | buffer;
  pending->descriptorBytes = size.size();
  _ck_pending_device_send_bytes = 0;  // do not charge the next unrelated send

  if (pending->slotWait) {
    // Parked for a CUDA IPC event slot; the pump prepares and sends it in
    // its turn. Outstanding for quiescence until then.
    CkIpcSlotQueue& q = ipc_slot_queue;
    q.queue.push_back(pending);
    QdCreate(1);
    q.parked++;
    if (q.queue.size() > q.maxDepth) q.maxDepth = q.queue.size();
    if (!q.reported) {
      q.reported = true;
      CmiPrintf("[%d] device sends wait for CUDA IPC event slots: parked in "
                "order and retried from the scheduler (CHARM_DEBUG_IPCQ "
                "traces each)\n", CkMyPe());
      fflush(stdout);
    }
    if (ipcQueueDbg())
      CmiPrintf("[IPCQ %d] park %zu bytes to PE %d; depth %zu, %d slots busy\n",
                CkMyPe(), pending->bytes, pending->destPe, q.queue.size(),
                CkRdmaDeviceBusyIpcSlots());
    ipcSlotPumpSchedule(0.05);
    return;
  }

  std::vector<hapiStream_t> streams;
  for (const auto& buffer : pending->buffers) {
    if (buffer.cnt == 0) continue;  // nothing produced it
    if (std::find(streams.begin(), streams.end(), buffer.hapi_stream) == streams.end())
      streams.push_back(buffer.hapi_stream);
  }
  pending->remaining = streams.size();
  const CkCallback cb(deviceSendProducerReady, pending);
  for (auto stream : streams) hapiAddCallback(stream, cb);
  // HAPI accounts for these callbacks in quiescence detection. The final one
  // publishes the ordinary message before its callback work is retired.
}

// Performs sender-side operations necessary for device zerocopy
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers) {
  CkRdmaDeviceOnSender(dest_pe, numops, buffers, nullptr);
}

void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers,
                          CkDeviceDeferredSend** pending) {
  if (pending) *pending = nullptr;
  // dest_pe == -1 means this PE has never confirmed where the target element
  // actually lives (xi-Parameter.C asks the location manager directly for
  // this, rather than substituting a homePe() guess). Don't decide
  // MEMCPY-vs-IPC from an unconfirmed location. But RDMA is not a safe
  // universal fallback here: findTransferModeDevice (above) only ever
  // returns RDMA for genuinely different physical nodes, so reconverse's
  // CmiIssueRget has never had to handle an RDMA-mode transfer that lands on
  // the sender's own physical node -- and its same-node loopback path does a
  // raw host memcpy between src/dst pointers, which corrupts memory (or
  // segfaults) when those are two different processes' device pointers, as
  // an unconfirmed-destination first contact on a single-node job always is.
  // When the whole job is on one physical node, an unconfirmed destination
  // can only ever truly resolve to MEMCPY or IPC (never RDMA), so stage IPC:
  // if the true destination turns out to be this same process after all,
  // the receiver's MEMCPY branch below uses source.ptr directly and ignores
  // this staging entirely -- the only cost is one unused, never-freed
  // comm-buffer slot and IPC event, a bounded one-time waste per
  // newly-confirmed pair, not a per-step cost. Only for a genuinely
  // multi-physical-node job with an unconfirmed destination do we fall back
  // to RDMA -- correct only if that destination doesn't land back on our own
  // physical node; that residual case isn't handled yet.
  const bool dest_confirmed = (dest_pe != -1);
  zcDestCount(dest_confirmed);
  CkNcpyModeDevice transfer_mode;
  if (dest_confirmed) {
    transfer_mode = findTransferModeDevice(CkMyPe(), dest_pe);
  } else if (CmiNumPhysicalNodes() == 1) {
    transfer_mode = CkNcpyModeDevice::IPC;
  } else {
    transfer_mode = CkNcpyModeDevice::RDMA;
  }

  // The mode above is chosen from where the target is now, but only becomes
  // true when the message lands. A target that migrates across a process
  // boundary in between is handled by the correction protocol, which re-sends
  // the payload by a route the new host can read -- so the mode does not have
  // to be pessimistic here.

  // Store destination PE in the metadata message
  // FIXME: Not necessary? save_op.dest_pe is set to CkMyPe() on the receiver
  size_t device_bytes = 0;
  for (int i = 0; i < numops; i++) {
    buffers[i]->dest_pe = dest_pe;
    buffers[i]->dest_mpi_rank = dest_confirmed ? CmiNodeOf(dest_pe) : -1;
    buffers[i]->src_pe = CmiMyPe();
    buffers[i]->src_mpi_rank = CmiNodeOf(CmiMyPe());
    device_bytes += buffers[i]->cnt;
  }
  _ck_pending_device_send_bytes = device_bytes;

  // Empty payloads. A zero-length buffer has nothing to transfer and nothing to
  // order against, so it takes none of what follows: no event, no IPC slot or
  // export, no registration, no producer wait, and no migration interlock. Its
  // completion callback fires now -- the source is trivially free -- and the
  // descriptor travels marked prepared with no protocol, which the receiver
  // (CkRdmaDeviceIssueRgets) completes without touching the device and the
  // forward repair (CkRdmaDeviceRepairForward) leaves alone. The message keeps
  // its device-zerocopy type, so the post entry method still runs, and a
  // message can mix empty and real payloads: this is per buffer.
  //
  // What it buys: a stencil exchange that sends to every neighbour every step,
  // whether or not it has anything for them, otherwise pays a device handshake
  // per empty message -- on sph2d that was 16 of the 32 device messages a
  // patch sends per step, in a step that is nothing but that protocol.
  //
  // The runtime's own migration payload is exempt: it is released by the
  // receiver's ack, not by a callback, and goes through unchanged.
  int live_ops = 0;
  for (int i = 0; i < numops; i++) {
    if (buffers[i]->cnt != 0 || ck_sending_migration_payload) { live_ops++; continue; }
    buffers[i]->ipc_protocol = CmiIpcProtocol::NONE;
    buffers[i]->device_idx = -1;
    buffers[i]->event_idx = -1;
    buffers[i]->memcpy_event = NULL;
    buffers[i]->ready_rank = -1;
    buffers[i]->ready_seq = 0;
    buffers[i]->sender_prepared = true;
    if (buffers[i]->cb.type != CkCallback::ignore) {
      buffers[i]->cb.send();
      buffers[i]->cb = CkCallback(CkCallback::ignore);
    }
  }
  if (live_ops == 0) return;  // nothing to prepare; the proxy sends at once

  // CHARM_ZC_VALIDATE: check every outgoing source pointer, in every transfer
  // mode, at the moment the send is posted. A device buffer whose owning chare
  // has migrated is either unmapped (freed by the old PE) or resident on a
  // different device than the one now current, and both surface later as an
  // "illegal memory access" inside an async copy with no hint of which send
  // produced it. Report it here, where dest_pe/cnt/mode still identify the send.
  {
    if (zcValidateOn()) {
      int cur_dev = -1;
      cudaGetDevice(&cur_dev);
      for (int i = 0; i < numops; i++) {
        if (buffers[i]->cnt == 0) continue;
        cudaPointerAttributes sattr{};
        const cudaError_t serr = cudaPointerGetAttributes(&sattr, buffers[i]->ptr);
        const bool unmapped = (serr != cudaSuccess || sattr.type == cudaMemoryTypeUnregistered);
        const bool wrong_dev = (!unmapped && sattr.type == cudaMemoryTypeDevice &&
                                sattr.device != cur_dev);
        if (unmapped || wrong_dev) {
          CmiPrintf("[%d] ZC SEND VALIDATE FAIL src=%p %s (type=%d ptr_dev=%d "
                    "cur_dev=%d err=%d) cnt=%zu dest_pe=%d mode=%s\n",
                    CkMyPe(), buffers[i]->ptr,
                    unmapped ? "UNMAPPED" : "WRONG-DEVICE",
                    (int)sattr.type, (int)sattr.device, cur_dev, (int)serr,
                    (size_t)buffers[i]->cnt, dest_pe,
                    transfer_mode == CkNcpyModeDevice::MEMCPY ? "MEMCPY"
                      : (transfer_mode == CkNcpyModeDevice::IPC ? "IPC" : "RDMA"));
          fflush(stdout);
          cudaGetLastError();
        }
      }
    }
  }

  if(transfer_mode == CkNcpyModeDevice::MEMCPY)
  {
    // Same process: the receiver can read buffers[i]->ptr directly, but only
    // once the kernels producing it have finished. Blocking the host here
    // (hapiStreamSynchronize) guaranteed that at the cost of stalling the
    // sender on every such send -- and same-process is the common case, so
    // that stall dominated. Record an event instead and let the receiver's
    // stream wait on it: identical ordering, no host block, no extra copy.
    // Set CHARM_ZC_MEMCPY_SYNC to restore the blocking behaviour.
    // A same-process memcpy send leaves the payload in the application's buffer
    // for the receiver to copy out of, exactly as the inter-node path leaves it
    // for the network to read. Migration frees and reallocates that buffer, so
    // count these against the issuing element too -- this is the common case in
    // a blocked decomposition, where most neighbours share a process, and it
    // was the one path with no protection at all.
    CkLocRec* memcpy_rec = CkpvAccess(_currentLocRec);
    if (memcpy_rec) {
      for (int i = 0; i < numops; i++) {
        if (buffers[i]->cnt == 0) continue;  // completed above, nothing to hold
        memcpy_rec->noteDeviceSendPosted();
        buffers[i]->cb = CkCallback(deviceSendReleaseFn,
                                    (void*)new DeviceSendRelease{memcpy_rec, buffers[i]->cb});
      }
    }

    static const bool force_sync = (getenv("CHARM_ZC_MEMCPY_SYNC") != nullptr);
    for (int i = 0; i < numops; i++) {
      if (buffers[i]->cnt == 0) continue;
      if (force_sync) {
        ckSubmitDrainThenSync(buffers[i]->hapi_stream);
      } else {
        // The receiver gates its copy on this on the host (DeviceRecvPending):
        // a pinned flag it reads with a plain load, or, when this ring has no
        // slot, the event it queries.
        buffers[i]->ready_rank = -1;
        buffers[i]->ready_seq = 0;
        {
          int rank = -1;
          uint32_t seq = 0;
          if (hapiFlagIssue(buffers[i]->hapi_stream, &rank, &seq)) {
            buffers[i]->ready_rank = rank;
            buffers[i]->ready_seq = seq;
          }
        }
        buffers[i]->memcpy_event = ckDeviceRecordMemcpyEvent(buffers[i]->hapi_stream);
        if (buffers[i]->memcpy_event == NULL)  // no event available; fall back
          ckSubmitDrainThenSync(buffers[i]->hapi_stream);
      }
    }
    return;
  }

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  if(transfer_mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
    if (pending) {
      // A deferrable send is never blocked for a slot. It is parked, in
      // order, when its slots are not there or earlier sends are already
      // waiting (see the IPC slot queue above ipcPrepareBuffers).
      if (!ipc_slot_queue.queue.empty() ||
          !ipcPrepareBuffers(dest_pe, numops, buffers, /*wait=*/false, nullptr))
        *pending = ipcParkSend(dest_pe, numops, buffers, device_bytes);
      return;
    }
    // Synchronous entries and the three-argument API keep their blocking
    // contract: the slots are waited for here.
    ipcPrepareBuffers(dest_pe, numops, buffers, /*wait=*/true, nullptr);
    return;
  } else {
#if !CMK_GPU_COMM
    // Use a naive host-staged mechanism
    // Allocate temporary host buffers and copy source buffers
    for (int i = 0; i < numops; i++) {
      if (buffers[i]->cnt == 0) continue;
      buffers[i]->data_stored = true;
      buffers[i]->sender_prepared = true;
      hapiCheck(hapiMallocHost(&buffers[i]->data, buffers[i]->cnt));
      hapiCheck(hapiMemcpyAsync(buffers[i]->data, buffers[i]->ptr, buffers[i]->cnt,
            hapiMemcpyDeviceToHost, buffers[i]->hapi_stream));
    }

    // Wait for the copies to finish
    for (int i = 0; i < numops; i++) {
      if (buffers[i]->cnt == 0) continue;
      hapiCheck(ckSubmitDrainThenSync(buffers[i]->hapi_stream));
    }
#else
  CkLocRec* sender_rec = CkpvAccess(_currentLocRec);
  if (pending && numops > 0) *pending = new CkDeviceDeferredSend(sender_rec);
  for (int i = 0; i < numops; i++) {
    if (buffers[i]->cnt == 0) {
      // Empty: its descriptor keeps its place in the message's fixed-width
      // prefix, but nothing is registered or held.
      if (pending) (*pending)->buffers.push_back(*buffers[i]);
      continue;
    }
    // This registers the application's own buffer and the receiver reads it over
    // the network, so it stays live well past this call. Count it against the
    // issuing element; emigrate stands down while any are outstanding.
    // Same element owns the registration: it is that element's buffer, and its
    // migration is when the buffer dies.
    if (sender_rec) {
      sender_rec->noteDeviceSendPosted();
      buffers[i]->cb = CkCallback(deviceSendReleaseFn,
                                  (void*)new DeviceSendRelease{sender_rec, buffers[i]->cb});
    }
    if (pending) {
      // The proxy marshals an unregistered placeholder. SendWhenReady replaces
      // it after HAPI completion, before publishing any RDMA metadata.
      buffers[i]->lci_ncpy_buffer = CmiNcpyBuffer();
      buffers[i]->lci_ncpy_buffer.deviceRdmaOpInfo = nullptr;
      memset(buffers[i]->lci_ncpy_buffer.layerInfo, 0,
             sizeof(buffers[i]->lci_ncpy_buffer.layerInfo));
      buffers[i]->sender_prepared = false;
      (*pending)->buffers.push_back(*buffers[i]);
      (*pending)->bytes += buffers[i]->cnt;
    } else {
      // Synchronous entry methods and callers using the original prepare API
      // have no deferred continuation and retain their blocking contract.
      hapiCheck(ckSubmitDrainThenSync(buffers[i]->hapi_stream));
      buffers[i]->lci_ncpy_buffer =
          acquireDeviceRegistration(buffers[i]->ptr, buffers[i]->cnt, sender_rec);
      buffers[i]->sender_prepared = true;
    }
  }
#endif
  }
}
#endif // CMK_CUDA
