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
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);
CpvExtern(int, my_device_id);

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
// Only the send path is cached. A receive registers the destination buffer the
// element posted, but CkRdmaDeviceIssueRgets runs before the entry method, so
// _currentLocRec is not yet set and there is no owner to attribute the entry
// to. Caching without an owner would mean never dropping it, which is the
// staleness this design exists to avoid.

struct DeviceMrKey {
  const void* ptr;
  size_t cnt;
  bool operator==(const DeviceMrKey& o) const {
    return ptr == o.ptr && cnt == o.cnt;
  }
};

struct DeviceMrKeyHash {
  size_t operator()(const DeviceMrKey& k) const {
    return std::hash<const void*>()(k.ptr) * 31 + std::hash<size_t>()(k.cnt);
  }
};

struct DeviceMrEntry {
  CmiNcpyBuffer reg;   // registered once; copies differ only in the ref field
  CkLocRec* owner;     // whose migration retires this entry
};

typedef std::unordered_map<DeviceMrKey, DeviceMrEntry, DeviceMrKeyHash>
    DeviceMrCache;

CkpvDeclare(DeviceMrCache*, device_mr_cache);

static bool deviceMrCacheEnabled()
{
  static const bool on = (getenv("CHARM_DEVICE_MR_CACHE") != nullptr);
  return on;
}

void CkRdmaDeviceRegistrationCacheInit()
{
  CkpvInitialize(DeviceMrCache*, device_mr_cache);
  CkpvAccess(device_mr_cache) = deviceMrCacheEnabled() ? new DeviceMrCache() : NULL;
}

// Registered descriptor for [ptr, ptr+cnt). Registers on a miss, and without
// the cache behaves exactly as constructing one in place did.
static CmiNcpyBuffer acquireDeviceRegistration(const void* ptr, size_t cnt,
                                               CkLocRec* owner)
{
  DeviceMrCache* cache = CkpvAccess(device_mr_cache);
  if (cache == NULL) return CmiNcpyBuffer(ptr, cnt);

  DeviceMrKey key{ptr, cnt};
  auto it = cache->find(key);
  if (it == cache->end()) {
    DeviceMrEntry entry;
    entry.reg = CmiNcpyBuffer(ptr, cnt);   // the one real registration
    entry.owner = owner;
    it = cache->emplace(key, entry).first;
  } else if (it->second.owner == NULL) {
    // First use came from a context with no element attached; adopt the first
    // owner that does appear, so the entry becomes retirable.
    it->second.owner = owner;
  }
  return it->second.reg;
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

  // Update counter (there may be multiple buffers in transit)
  info->counter++;

  // Check if all buffers have been received
  // If so, invoke regular entry method
  if (info->counter == info->n_ops) {
    QdCreate(1);

    enqueueNcpyMessage(op->dest_pe, info->msg);

    // Free RDMA metadata
    // CmiFree(info);
  }
}
// Invoked when a GPU buffer arrives on the receiver
void CkRdmaDeviceRecvHandler(void* data, void* msg)
{
  DeviceRdmaOp* op = (DeviceRdmaOp*)data;
  DeviceRdmaInfo* info = op->info;

  // Invoke source callbacks
  if (op->src_cb) {
    CkCallback* cb = (CkCallback*)op->src_cb;
    cb->send();
    delete cb;
  }

  // Update counter (there may be multiple buffers in transit)
  info->counter++;

  // Check if all buffers have been received
  // If so, invoke regular entry method
  if (info->counter == info->n_ops) {
    QdCreate(1);

    enqueueNcpyMessage(op->dest_pe, info->msg);

    // Free RDMA metadata
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
  hapiError_t err = hapiStreamSynchronize(stream);
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
  ~ZcModeStats() {
    const long m = memcpy_n.load(), i = ipc_n.load(), o = other_n.load();
    const long total = m + i + o;
    if (total == 0) return;
    fprintf(stderr, "[zc-stats] pid=%d MEMCPY=%ld IPC=%ld OTHER=%ld "
                    "(same-process %.1f%% of %ld)\n",
            (int)getpid(), m, i, o, 100.0 * m / total, total);
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
  if (hapiEventRecord(ev, stream) != hapiSuccess) return NULL;
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
        imported_base = hapiIpcImportBuffer(source.ipc_handle);
        if (imported_base == NULL) {
          CkAbort("CkRdmaDeviceIssueRgets: receive on PE %d from PE %d could not "
                  "open the CUDA IPC handle for a %zu-byte direct transfer. The "
                  "exporting process may have freed the allocation, or this "
                  "process may already hold a mapping of it that was never "
                  "closed.",
                  CkMyPe(), srcPe, (size_t)dest.cnt);
        }
        src_addr = (const void*)((char*)imported_base + source.ipc_offset);
      }

      // 1. Make user-provided stream wait for IPC event using hapiStreamWaitEvent
      //    (staged: source buffer to device comm buffer on source; direct: the
      //    kernels that produced the source buffer)
      // Same reason as the dst_event_pool record below: across processes
      // ipcHandleOpen imports a peer's events under our own device, but it skips
      // our own process, so a source that is a different device inside this
      // process leaves that device's original event in the pool. Making our
      // stream wait on it is not valid from here. Wait on the host instead --
      // stronger ordering than the stream wait, and confined to this one case,
      // which only became reachable once load balancing moved chares between
      // GPUs within a process.
      {
        const int my_dev_idx = csv_gpu_manager.device_count * CmiMyNodeRankLocal()
                             + CpvAccess(my_device_id);
        if (source.device_idx != my_dev_idx) {
          hapiCheck(cudaEventSynchronize(device_info.src_event_pool[source.event_idx]));
        } else {
          hapiCheck(hapiStreamWaitEvent(recv_stream,
                device_info.src_event_pool[source.event_idx], 0));
        }
      }
      ipcDebugSync("recv 1: wait imported src_event", recv_stream);

      // 2. Invoke hapiMemcpyAsync from the peer's memory to the destination
      //    buffer. This is the only copy a direct transfer makes.
      // Same reason as the same-process copy above: the peer's buffer may be on
      // a different device from ours.
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, src_addr,
            dest.cnt, cudaMemcpyDefault, recv_stream));
      ipcDebugSync("recv 2: peer copy -> dest", recv_stream);

      // 3. Record IPC event so that the sender can query it for freeing
      //    device comm buffer and corresponding pair of CUDA IPC events.
      //    The event belongs to the source's device; when that is a different
      //    device inside this same process it was never imported under ours
      //    (ipcHandleOpen skips our own process), so settle the copy and record
      //    on the owning device rather than on our stream.
      {
        const int my_dev_idx = csv_gpu_manager.device_count * CmiMyNodeRankLocal()
                             + CpvAccess(my_device_id);
        if (source.device_idx != my_dev_idx) {
          hapiCheck(hapiStreamSynchronize(recv_stream));
          const int src_local = source.device_idx % csv_gpu_manager.device_count;
          const int src_global = csv_gpu_manager.device_managers[src_local].global_index;
          int prev_dev = 0;
          hapiCheck(hapiGetDevice(&prev_dev));
          hapiCheck(hapiSetDevice(src_global));
          hapiCheck(hapiEventRecord(device_info.dst_event_pool[source.event_idx], 0));
          hapiCheck(hapiSetDevice(prev_dev));
        } else {
          hapiCheck(hapiEventRecord(device_info.dst_event_pool[source.event_idx],
                recv_stream));
        }
      }
      ipcDebugSync("recv 3: record imported dst_event", recv_stream);

      // 4. Set flag in shared memory so that the sender can start querying
      //    completion of the IPC event
      hapi_ipc_event_shared* shm_event_shared =
        (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
            + csv_gpu_manager.shm_chunk_size * source.device_idx
            + sizeof(hapiIpcMemHandle_t)) + source.event_idx;
      // The sender clears this when it reclaims the slot, so finding it already
      // set means a second receive is signalling the same (device, event) pair
      // before the first was retired. The sender would then free the block
      // belonging to whichever transfer claimed the slot next, while that
      // transfer is still reading it -- a use-after-free inside the comm buffer
      // that surfaces later as an illegal access on an unrelated stream.
      //
      // Release pairs with the sender's acquire load in
      // reclaimCompletedIpcEvents: it is what makes the hapiEventRecord above
      // visible before the sender is allowed to query that event.
      const bool already =
          shm_event_shared->dst_flag.exchange(true, std::memory_order_acq_rel);
      if (already) {
        CmiPrintf("[%d] IPC DUPLICATE dst_flag dev_idx=%d ev_idx=%d srcPe=%d "
                  "off=%zu cnt=%zu\n",
                  CkMyPe(), source.device_idx, source.event_idx,
                  srcPe, (size_t)source.comm_offset, (size_t)dest.cnt);
        fflush(stdout);
      }

}


// Defined below, next to the reclaim scan it drives.
static void acquireIpcSendSlot(DeviceManager* dm, int cpv_my_device_id,
                               bool is_lb_buffer, bool direct,
                               const void* src_ptr, size_t cnt,
                               void** out_buffer, int* out_event_idx);

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
  size_t cnt;
  bool inter_node;
  CmiNcpyBuffer dest_ncpy;         // inter-node only: registered destination
};

struct DeviceRestageMeta {         // sender -> receiver, same node
  char header[CmiMsgHeaderSizeBytes];
  void* dest_op;
  int device_idx;
  int event_idx;
  size_t comm_offset;
  size_t cnt;
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
  CkRdmaDeviceRecvHandler(m->dest_op, NULL);
  CmiFree(m);
  return NULL;
}

// Receiver side: defer this buffer and ask the sender to retransmit it.
static void requestDeviceRestage(int srcPe, void* dest_op, const void* src_ptr,
                                 bool src_staged, size_t src_comm_offset,
                                 size_t cnt, void* dest_ptr, size_t dest_cnt,
                                 bool inter_node)
{
  DeviceRestageReq* req = (DeviceRestageReq*)CmiAlloc(sizeof(DeviceRestageReq));
  CmiEnforce(req);
  req->dest_op = dest_op;
  req->dest_pe = CkMyPe();
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
    CmiPrintf("[%d] ZC RESTAGE #%u srcPe=%d cnt=%zu %s\n", CkMyPe(),
              n.fetch_add(1, std::memory_order_relaxed) + 1, srcPe, cnt,
              inter_node ? "inter-node put" : "same-node stage");
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

  if (req->inter_node) {
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
  } else {
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
    const int cpv_my_device_id = CpvAccess(my_device_id);
    void* staged = NULL;
    int event_idx = -1;
    acquireIpcSendSlot(dm, cpv_my_device_id, /*is_lb_buffer=*/false,
                       /*direct=*/false, src_ptr, req->cnt, &staged,
                       &event_idx);
    hapiCheck(hapiMemcpyAsync(staged, src_ptr, req->cnt,
                              hapiMemcpyDeviceToDevice, hapiStreamPerThread));

    const int device_idx =
        csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id;
    hapi_ipc_device_info& my_device_info =
        csv_gpu_manager.hapi_ipc_device_infos[device_idx];
    hapiCheck(hapiEventRecord(my_device_info.src_event_pool[event_idx],
                              hapiStreamPerThread));

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
    QdCreate(1);
    CmiSetHandler(m, device_restage_meta_handler);
    CmiSyncSendAndFree(req->dest_pe, sizeof(DeviceRestageMeta), (char*)m);
  }
  CmiFree(req);
  return NULL;
}

// Receiver side, same node: the sender has staged the payload, so run the
// ordinary staged-IPC receive against the metadata it just sent.
extern "C" void* device_restage_meta_bridge(void* arg)
{
  QdProcess(1);
  DeviceRestageMeta* m = (DeviceRestageMeta*)arg;
  DeviceRdmaOp* op = (DeviceRdmaOp*)m->dest_op;

  CkDeviceBuffer source;
  source.ptr = NULL;                 // never dereferenced on a staged receive
  source.cnt = m->cnt;
  source.device_idx = m->device_idx;
  source.event_idx = m->event_idx;
  source.comm_offset = m->comm_offset;
  source.ipc_protocol = CmiIpcProtocol::STAGED;
  source.sender_prepared = true;

  CkDeviceBuffer dest(op->dest_ptr, op->size);
  hapiStream_t recv_stream = (hapiStream_t)op->stream;

  deviceIpcReceive(source, dest, recv_stream, op->src_pe,
                   CkNcpyModeDevice::IPC);
  hapiAddCallback(recv_stream, CkCallback(CkRdmaDeviceRecvHandler, op));

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
      for (int i = 0; i < it->numops; i++)
        CmiPrintf("[%d]   op %d: srcPe=%d bytes=%zu %s\n", CkMyPe(), i,
                  it->src_pe[i], it->size[i],
                  it->deferred[i] ? "DEFERRED for correction" : "ordinary");
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
  // Change message header to invoke regular entry method
  CMI_ZC_MSGTYPE(env) = CMK_REG_NO_ZC_MSG;

  // Create a copy of this message for regular entry method invocation
  // FIXME: Reuse the old message instead of creating a new one
  void* old_msg = EnvToUsr(env);
  envelope* new_env = UsrToEnv(CkCopyMsg(&old_msg));

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
  DeviceRdmaInfo* rdma_info = (DeviceRdmaInfo*)rdma_data;
  rdma_info->n_ops = numops;
  rdma_info->counter = 0;
  rdma_info->msg = new_env;

  // Watchdog bookkeeping; no cost unless CHARM_ZC_STALL_SECS is set.
  DeviceRecvWatch* watch = NULL;
  if (CkpvAccess(device_recv_watch) != NULL) {
    CkpvAccess(device_recv_watch)->push_back(
        DeviceRecvWatch{rdma_info, CkWallTimer(), numops,
                        std::vector<int>(numops, -1),
                        std::vector<size_t>(numops, 0),
                        std::vector<char>(numops, 0), false});
    watch = &CkpvAccess(device_recv_watch)->back();
  }

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
    // DIAGNOSTIC: what buffer did this receive post, for which element, and in
    // which mode. Correlated against the application's own record of which
    // ghost it later reads out of that buffer, this shows whether the runtime
    // ever lands one message's payload in another message's posted buffer.
    save_op.src_pe = env->getSrcPe();
    if (watch) { watch->src_pe[i] = env->getSrcPe(); watch->size[i] = (size_t)arrSizes[i]; }
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
    zcStatsCount(mode);

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
    const bool sender_exported =
        (source.ipc_protocol != CmiIpcProtocol::NONE &&
         source.device_idx != -1 && csv_gpu_manager.use_shm &&
         CmiPeOnSamePhysicalNode(env->getSrcPe(), CkMyPe()));
    const bool sender_direct =
        (sender_exported && source.ipc_protocol == CmiIpcProtocol::DIRECT);

    if (mode == CkNcpyModeDevice::MEMCPY && !sender_exported) {
      // Source and destination PEs are in the same process (logical node)
      // Directly invoke memcpy from source buffer to destination buffer.
      // Order against the sender's stream first: without this the copy could
      // run before the kernels that produced the source data. A null event
      // means the sender blocked instead, so the data is already there.
      if (source.memcpy_event != NULL) {
        hapiCheck(hapiStreamWaitEvent(postStructs[i].hapi_stream,
              (hapiEvent_t)source.memcpy_event, 0));
      }
      // cudaMemcpyDefault, not DeviceToDevice: once load balancing has moved
      // chares between GPUs, the source and destination of a same-process
      // transfer can sit on different devices, and an explicit DeviceToDevice
      // kind is rejected for that pair. Default resolves the direction from the
      // pointers themselves and handles the peer case.
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.ptr, dest.cnt,
            cudaMemcpyDefault, postStructs[i].hapi_stream));

      // The sender may have staged IPC info for this transfer anyway: an
      // unconfirmed destination that turned out to be this same process
      // (see CkRdmaDeviceOnSender). That staging allocated a device
      // comm-buffer slot and claimed a CUDA IPC event that nothing will
      // ever free unless we tell the sender it's unused -- the free-up
      // logic in reclaimCompletedIpcEvents only runs once it sees the dst_flag
      // this receiver would have set had it taken the IPC branch below.
      // Signal that now (skip the actual comm-buffer copy in steps 1-2,
      // since the real data already moved via source.ptr above; just record
      // the completion event and flag so the sender can reclaim the slot).
      if (source.device_idx != -1 && csv_gpu_manager.use_shm) {
        hapi_ipc_device_info& device_info =
          csv_gpu_manager.hapi_ipc_device_infos[source.device_idx];

        // The event belongs to the sender's device, and cudaEventRecord
        // requires the event and the stream to be on the same one. Across
        // processes that is never a problem: ipcHandleOpen imports a peer's
        // events under our own device. It skips our own process, though -- a
        // process cannot open a handle it exported itself -- so when the source
        // is a different device inside this same process, what sits in the pool
        // is that device's original event, and recording it on our stream fails
        // with 'invalid argument'. This only became reachable once load
        // balancing started moving chares between GPUs in one process.
        //
        // Settle the copy, then record on the owning device. Synchronous, but
        // it applies only to this one case: a same-process transfer that
        // crossed devices and whose sender staged IPC state that has to be
        // released.
        const int my_dev_idx =
            csv_gpu_manager.device_count * CmiMyNodeRankLocal() + CpvAccess(my_device_id);
        if (source.device_idx != my_dev_idx) {
          hapiCheck(hapiStreamSynchronize(postStructs[i].hapi_stream));
          const int src_local = source.device_idx % csv_gpu_manager.device_count;
          const int src_global =
              csv_gpu_manager.device_managers[src_local].global_index;
          int prev_dev = 0;
          hapiCheck(hapiGetDevice(&prev_dev));
          hapiCheck(hapiSetDevice(src_global));
          hapiCheck(hapiEventRecord(device_info.dst_event_pool[source.event_idx], 0));
          hapiCheck(hapiSetDevice(prev_dev));
        } else {
          hapiCheck(hapiEventRecord(device_info.dst_event_pool[source.event_idx],
                postStructs[i].hapi_stream));
        }
        hapi_ipc_event_shared* shm_event_shared =
          (hapi_ipc_event_shared*)((char*)csv_gpu_manager.shm_ptr
              + csv_gpu_manager.shm_chunk_size * source.device_idx
              + sizeof(hapiIpcMemHandle_t)) + source.event_idx;
        shm_event_shared->dst_flag.store(true, std::memory_order_release);
      }
    } else if (sender_exported) {
      deviceIpcReceive(source, dest, postStructs[i].hapi_stream,
                       env->getSrcPe(), mode);
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
        requestDeviceRestage(env->getSrcPe(), (void*)&save_op, source.ptr,
                             source.ipc_protocol == CmiIpcProtocol::STAGED,
                             source.comm_offset,
                             (size_t)dest.cnt, arrPtrs[i], (size_t)arrSizes[i],
                             mode != CkNcpyModeDevice::IPC);
        continue;  // completion deferred until the retransmit lands
      }
      // CmiPrintf("it should never be called during intra node\n");
#if CMK_GPU_COMM
      // Machine layer supports GPU-aware communication
      QdCreate(1);
      CmiSetDirectNcpyAckHandler(CkRdmaDeviceRecvHandler);
      CmiNcpyBuffer lci_dest_ncpy_buffer(arrPtrs[i], (size_t)arrSizes[i], (void*)(&save_op));
      lci_dest_ncpy_buffer.rdmaGet(source.lci_ncpy_buffer, 0, nullptr, nullptr);
      continue;
#else
      // Handle all other cases (basic inter-process and inter-node)
      // Transfer the received/unpacked data on host to the destination device buffer
      // FIXME: Print warning that this is slow?
      CkAssert(source.data_stored);
      hapiCheck(hapiMemcpyAsync((void*)dest.ptr, source.data, dest.cnt,
            hapiMemcpyHostToDevice, postStructs[i].hapi_stream));
#endif
    }

    // Add source callback for polling, so that it can be invoked once the transfer is complete
    hapiAddCallback(postStructs[i].hapi_stream, CkCallback(CkRdmaDeviceRecvHandler, &save_op));
  }
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
    if (!my_shm_events[i].dst_flag.load(std::memory_order_acquire)) continue;

    // The receiver has invoked the memcpy, so the sender may query the event.
    if (hapiEventQuery(my_device_info.dst_event_pool[i]) != hapiSuccess) continue;

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
      CkAbort("Retrieved hapiSuccess for a free IPC event");
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
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id];

  for (int i = pool_start; i < pool_start + pool_size; i++) {
    int& event_flag = my_device_info.event_pool_flags[i];
    size_t& buff_offset = my_device_info.event_pool_buff_offsets[i];
    if (event_flag == 0) {
      event_flag = flag_value;
      buff_offset = comm_offset;
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
static void acquireIpcSendSlot(DeviceManager* dm, int cpv_my_device_id,
                               bool is_lb_buffer, bool direct,
                               const void* src_ptr,
                               size_t cnt, void** out_buffer,
                               int* out_event_idx) {
  static const double timeout_s = []() {
    const char* s = getenv("CHARM_IPC_SLOT_TIMEOUT");
    return s ? atof(s) : 60.0;
  }();

  // CHARM_IPC_LAZY_RECLAIM=1 takes the reclaim scan off the fast path (try
  // first, reclaim only on failure). Opt-in rather than default: see the
  // discussion at CkRdmaDeviceAllocLbBuffer -- other allocators of this buffer
  // depend on the sweep running often, and the scan is cheap enough now that
  // dst_flag is a plain atomic that removing it from the send path buys much
  // less than it did.
  static const bool lazy = (getenv("CHARM_IPC_LAZY_RECLAIM") != nullptr);

  double wait_start = 0.0;
  bool waiting = false;
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
      return;
    }

    if (!waiting) {
      wait_start = CkWallTimer();
      waiting = true;
    } else if (CkWallTimer() - wait_start > timeout_s) {
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
  const int pool_start = CkMyRank() * pool_size;
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

// Performs sender-side operations necessary for device zerocopy
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers) {
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
        memcpy_rec->noteDeviceSendPosted();
        buffers[i]->cb = CkCallback(deviceSendReleaseFn,
                                    (void*)new DeviceSendRelease{memcpy_rec, buffers[i]->cb});
      }
    }

    static const bool force_sync = (getenv("CHARM_ZC_MEMCPY_SYNC") != nullptr);
    for (int i = 0; i < numops; i++) {
      if (force_sync) {
        hapiStreamSynchronize(buffers[i]->hapi_stream);
      } else {
        buffers[i]->memcpy_event = ckDeviceRecordMemcpyEvent(buffers[i]->hapi_stream);
        if (buffers[i]->memcpy_event == NULL)  // no event available; fall back
          hapiStreamSynchronize(buffers[i]->hapi_stream);
      }
    }
    return;
  }

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  //int cpv_my_device_id = CmiMyRank() % csv_gpu_manager.device_count;
  int cpv_my_device_id = CpvAccess(my_device_id);

  if(transfer_mode == CkNcpyModeDevice::IPC && csv_gpu_manager.use_shm) {
    // Use optimizations with POSIX shaerd memory
    // Allocate blocks on device comm buffer
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];

    for (int i = 0; i < numops; i++) {
      bool is_lb_buffer = ( (size_t)((char*)(buffers[i]->ptr) - (char*)(dm->comm_buffer->base_ptr)) < dm->comm_buffer->total_size );

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
      hapiIpcMemHandle_t export_handle;
      size_t export_offset = 0;
      bool direct = false;
      if (!is_lb_buffer && hapiIpcUseDirect()) {
        direct = hapiIpcExportBuffer(buffers[i]->ptr, &export_handle,
                                     &export_offset);
      }

      // A zero-copy device send may not reuse or free its source buffer until
      // the CkDeviceBuffer's completion callback fires. That is the contract
      // whatever transport carries it, so a send without a callback is already
      // an application bug -- staging merely hides it, since the source happens
      // to be free once the staging copy retires on the sender's own stream.
      // Direct reads the sender's allocation itself, so the same bug shows up
      // as corrupted data instead. Report it once, where the send is still
      // identifiable, rather than let it surface there.
      if (direct && buffers[i]->cb.type == CkCallback::ignore) {
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

      // Waits for a slot if the pools are momentarily drained rather than
      // aborting; takes and releases dm->lock itself.
      void* alloc_comm_buffer;
      int acquired_event_idx;
      acquireIpcSendSlot(dm, cpv_my_device_id, is_lb_buffer, direct,
                         buffers[i]->ptr, buffers[i]->cnt, &alloc_comm_buffer,
                         &acquired_event_idx);
      if (direct) {
        buffers[i]->ipc_protocol = CmiIpcProtocol::DIRECT;
        buffers[i]->ipc_handle = export_handle;
        buffers[i]->ipc_offset = export_offset;
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
      buffers[i]->device_idx = (csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id);
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
                  buffers[i]->ptr, dm->comm_buffer->base_ptr,
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
      hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[(csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id)];
      hapiCheck(hapiEventRecord(my_device_info.src_event_pool[buffers[i]->event_idx], buffers[i]->hapi_stream));
      ipcDebugSync("send 2: record own src_event", buffers[i]->hapi_stream);
    }
  } else {
#if !CMK_GPU_COMM
    // Use a naive host-staged mechanism
    // Allocate temporary host buffers and copy source buffers
    for (int i = 0; i < numops; i++) {
      buffers[i]->data_stored = true;
      buffers[i]->sender_prepared = true;
      hapiCheck(hapiMallocHost(&buffers[i]->data, buffers[i]->cnt));
      hapiCheck(hapiMemcpyAsync(buffers[i]->data, buffers[i]->ptr, buffers[i]->cnt,
            hapiMemcpyDeviceToHost, buffers[i]->hapi_stream));
    }

    // Wait for the copies to finish
    for (int i = 0; i < numops; i++) {
      hapiCheck(hapiStreamSynchronize(buffers[i]->hapi_stream));
    }
#else
  for (int i = 0; i < numops; i++) {
    cudaStreamSynchronize(buffers[i]->hapi_stream);
    // This registers the application's own buffer and the receiver reads it over
    // the network, so it stays live well past this call. Count it against the
    // issuing element; emigrate stands down while any are outstanding.
    CkLocRec* sender_rec = CkpvAccess(_currentLocRec);
    // Same element owns the registration: it is that element's buffer, and its
    // migration is when the buffer dies.
    buffers[i]->lci_ncpy_buffer =
        acquireDeviceRegistration(buffers[i]->ptr, buffers[i]->cnt, sender_rec);
    buffers[i]->sender_prepared = true;
    if (sender_rec) {
      sender_rec->noteDeviceSendPosted();
      buffers[i]->cb = CkCallback(deviceSendReleaseFn,
                                  (void*)new DeviceSendRelease{sender_rec, buffers[i]->cb});
    }
  }
#endif
  }
}
#endif // CMK_CUDA
