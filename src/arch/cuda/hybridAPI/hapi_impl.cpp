#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <queue>
#include <atomic>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sched.h>

#define hapi_API_PER_THREAD_DEFAULT_STREAM

#include "hapi_portable.h"
#include "converse.h"
#include "conv-mach-opt.h" /* for CMK_CUDA / CMK_HIP */
#include "charm++.h"

#include "hapi.h"
#include "hapi_impl.h"
#include "gpumanager.h"
#ifdef HAPI_NVTX_PROFILE
#include "hapi_nvtx.h"
#endif

#if CMK_CUDA && CMK_LBDB_ON
// lbdb.h, not LBManager.h: this file needs the LB *types* (LDObjKey,
// GpuObjectTokenTable, LBKernelRecord), which are header-only, and must not
// acquire a link dependency on ck-ldb globals such as _lb_args. See the note on
// cuptiDebugLevel below.
#include "lbdb.h"
#include "cklocation.h"
#include <algorithm>
#include <climits>

// Defined in ck.C. Forward-declared rather than reached through ck.h, which
// this file does not otherwise need. Gives the entry-method correlation hook
// the running element's full LB identity.
CkLocRec* CkActiveLocRec(void);

// Verbosity for the CUPTI accounting below, from the environment rather than
// from _lb_args.debug().
//
// This object is only ever pulled into a link that also has libck -- hapi is
// reached from ck-core/init.C -- but nothing enforces that, and when it is
// violated the failure is a wall of undefined references in unrelated
// pure-Converse programs rather than anything pointing here. Reading a global
// that lives in ck-ldb would add one more way to trip it, for two diagnostic
// printfs. The other diagnostic in this file (CHARM_LB_CUPTI_TIME) is already
// environment-driven, so this matches it.
static int cuptiDebugLevel()
{
  static const int level = []() {
    const char* s = getenv("CHARM_LB_CUPTI_DEBUG");
    return s != nullptr ? atoi(s) : 0;
  }();
  return level;
}

#ifdef HAPI_CUPTI_LB
#include <cupti.h>

#define CUPTI_SAFE_CALL(call)                                              \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      CmiPrintf("HAPI CUPTI error: %s at %s:%d\n", errstr, __FILE__, __LINE__); \
    }                                                          \
  } while (0)
#endif
#endif

#define STREAM_BUF_SIZE 1024

#if defined HAPI_TRACE || defined HAPI_INSTRUMENT_WRS
// extern "C" double CmiWallTimer();
#endif

extern int Cmi_isOldProcess;

extern int CmiSetCPUAffinityLogical(int core);

static void createPool(int *nbuffers, int n_slots, std::vector<BufferPool> &pools);
static void releasePool(std::vector<BufferPool> &pools);

#ifdef HAPI_CUDA_CALLBACK
struct hapiCallbackMessage {
  char header[CmiMsgHeaderSizeBytes];
  int rank;
  CkCallback cb;
  void* cb_msg;
};
#endif

#ifndef HAPI_CUDA_CALLBACK
typedef struct hapiEvent {
  hapiEvent_t event;
  CkCallback cb;
  void* cb_msg;
  hapiWorkRequest* wr; // if this is not NULL, buffers and request itself are deallocated

  hapiEvent(hapiEvent_t event_, const CkCallback& cb_, void* cb_msg_, hapiWorkRequest* wr_ = NULL)
            : event(event_), cb(cb_), cb_msg(cb_msg_), wr(wr_) {}
} hapiEvent;

CpvDeclare(std::queue<hapiEvent>, hapi_event_queue);
CpvDeclare(std::queue<hapiEvent_t>, hapi_event_pool);
#endif // HAPI_CUDA_CALLBACK
CpvDeclare(int, n_hapi_events);

int firstRankForDevice = 0; // First rank for each device, used for mapping

// Managing memory state in server
int hapiAllocId = 0; // Global allocation ID for HAPI

// Used to invoke user's Charm++ callback function
void (*hapiInvokeCallback)(void*, void*) = NULL;

// Functions used to support quiescence detection.
void (*hapiQdCreate)(int) = NULL;
void (*hapiQdProcess)(int) = NULL;

#define MAX_PINNED_REQ 64
#define MAX_DELAYED_FREE_REQS 64

// Declare GPU Manager as a process-shared object.
CsvDeclare(GPUManager, gpu_manager);

CpvDeclare(int, my_device); // GPU device that this thread is mapped to
CpvDeclare(int, my_device_id); // index to the deviceManager that stores info about the device
CpvDeclare(bool, device_rep); // Is this PE a device representative thread? (1 per device)

// See gpumanager.h for what this indexes and why it is cached.
CpvDeclare(int, my_node_rank_local);

void hapiRefreshTopologyCache() {
#if CMK_RECONVERSE
  CpvAccess(my_node_rank_local) = CmiNodeRankOnPhysicalNode(CmiMyNode());
#else
  // Classic Converse has no equivalent query and the line is frozen, so it
  // keeps the arithmetic it has always used. That form assumes processes per
  // host are uniform and that logical node numbering is contiguous and
  // host-ordered; reconverse asks the machine layer instead (reconverse#211).
  CpvAccess(my_node_rank_local) =
      CmiNodeOf(CmiMyPe()) % (CmiNumNodes() / CmiNumPhysicalNodes());
#endif
}

// HAPI internal function declarations
static void hapiInitCsv(char** argv);
static void hapiInitCpv();
static void hapiExitCsv();

static void hapiMapping(char** argv);
static void hapiRegisterCallbacks();

// hapi IPC related functions
static void shmInit();
static void shmSetup();
static void shmCreate();
static void shmOpen();
static void shmMap();
static void shmAbort();
static void shmCleanup();
static void ipcHandleCreate();
static void ipcHandleOpen();

#if CMK_CUDA && CMK_LBDB_ON

// Sentinel external-correlation ID meaning "no owning migratable object".
// Must not collide with a real chare ID -- 0 is a perfectly valid one, which
// is why this is UINT64_MAX rather than the obvious choice.
static constexpr uint64_t HAPI_CUPTI_NO_OBJECT =
    GpuObjectTokenTable::noObjectToken();

#ifdef HAPI_CUPTI_LB

static void CUPTIAPI cuptiBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  // CUPTI writes activity records straight into this buffer and has no way to
  // tell us the allocation failed: hand it NULL and it writes through a null
  // pointer, and the NULL comes back through cuptiBufferCompleted to be parsed
  // later, so the fault surfaces inside cuptiActivityGetNextRecord with nothing
  // left to say where it came from. Step down to a smaller buffer before giving
  // up, and if even that fails, say so here.
  static const size_t sizes[] = {5*1024*1024, 1024*1024, 256*1024};
  for (size_t s : sizes) {
    *buffer = (uint8_t *)malloc(s);
    if (*buffer != NULL) {
      *size = s;
      *maxNumRecords = 0;
      return;
    }
  }
  CmiAbort("HAPI: could not allocate a CUPTI activity buffer (tried down to "
           "%zu bytes). GPU load instrumentation cannot continue.", sizes[2]);
}

static void CUPTIAPI cuptiBufferCompleted(CUcontext ctx, uint32_t streamId,
                                          uint8_t *buffer, size_t size, size_t validSize) {
  GPUManager& gm = CsvAccess(gpu_manager);

  std::lock_guard<std::mutex> lk(gm.cupti_queue_lock_);
  gm.cupti_buffer_queue_.push({buffer, validSize});
}

// Populate DeviceManager with the device attributes needed to compute per-kernel
// SM usage from CUPTI records. Queried once per local device, lazily, because
// device_managers is not populated when the GPUManager is constructed.
static void hapiPopulateDeviceProps(GPUManager& gm) {
  for (DeviceManager& dm : gm.device_managers) {
    if (dm.props_initialized) continue;
    int dev = dm.global_index;
    cudaDeviceProp props;
    hapiCheck(cudaGetDeviceProperties(&props, dev));

    dm.multi_processor_count = props.multiProcessorCount;
    dm.max_threads_per_sm = props.maxThreadsPerMultiProcessor;
#ifdef cudaDevAttrMaxBlocksPerMultiprocessor
    hapiCheck(cudaDeviceGetAttribute(&dm.max_blocks_per_sm,
                                     cudaDevAttrMaxBlocksPerMultiprocessor, dev));
#else
    dm.max_blocks_per_sm = 0;
#endif
    dm.max_registers_per_sm = props.regsPerMultiprocessor;
    dm.max_shared_mem_per_sm = static_cast<int>(props.sharedMemPerMultiprocessor);
    dm.warp_size = props.warpSize;
    dm.props_initialized = true;
  }
}

// Kept for callers that want tracing up before the balancer asks for it;
// hapiCuptiStartTracing attaches on its own, so this is not needed at startup.
void hapiCuptiInit() { hapiCuptiStartTracing(); }

// Attaching CUPTI to the process is NOT free even when no activity kind is
// enabled -- measured at ~1.3 ms per step on a 4-PE run, which is most of the
// cost that remains once tracing itself is windowed. So attach here and stop in
// hapiCuptiStopTracing, rather than staying enabled for the whole run.
// Enabling an activity kind is separately what makes records flow.
//
// Per-PE view of the tracing switch. LBDatabase::TurnStatsOn/Off is a PE-local
// switch, but CUPTI tracing is process-wide: a PE that switched its own
// instrumentation off would otherwise switch tracing off for every PE in the
// process, and whichever PE was still finishing its iteration would lose every
// kernel it launched after that instant -- one whole PE per process reading
// zero GPU load at every step. So the process traces while ANY PE wants
// instrumentation: tracing starts with the first PE to switch on and stops with
// the last to switch off, counted per PE so repeated switches do not skew the
// count.
static thread_local bool cupti_pe_tracing = false;

void hapiCuptiStartTracing() {
  GPUManager& gm = CsvAccess(gpu_manager);
  // Every PE thread reaches this through its own LBDatabase::TurnStatsOn, so
  // the check and the enable must be one atomic step -- otherwise several
  // threads each enable the same activity kinds.
  std::lock_guard<std::mutex> lk(gm.cupti_tracing_lock_);
  if (!cupti_pe_tracing) {
    cupti_pe_tracing = true;
    gm.cupti_tracing_users_++;
  }
  if (gm.cupti_tracing_active_.load(std::memory_order_relaxed)) return;

  if (!gm.cupti_initialized_) {
    cudaDeviceSynchronize();
    CUPTI_SAFE_CALL(
        cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));
    gm.cupti_initialized_ = true;
  }

  // RUNTIME must stay enabled alongside the kernel records even though nothing
  // consumes its records directly: EXTERNAL_CORRELATION records are only
  // emitted for correlation IDs generated by runtime-API tracking, so without
  // it every kernel arrives unattributed and the balancer sees zero GPU load.
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));

  gm.cupti_tracing_active_.store(true, std::memory_order_relaxed);
}

void hapiCuptiStopTracing() {
  GPUManager& gm = CsvAccess(gpu_manager);
  std::lock_guard<std::mutex> lk(gm.cupti_tracing_lock_);
  if (cupti_pe_tracing) {
    cupti_pe_tracing = false;
    if (gm.cupti_tracing_users_ > 0) gm.cupti_tracing_users_--;
  }
  // Other PEs of this process still have their instrumentation on: their
  // kernels are still being launched and must keep being recorded.
  if (gm.cupti_tracing_users_ > 0) return;
  if (!gm.cupti_initialized_ ||
      !gm.cupti_tracing_active_.load(std::memory_order_relaxed))
    return;

  CUPTI_SAFE_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_SAFE_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_SAFE_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));

  // Clear the flag before flushing so the buffers handed back by the flush are
  // the last ones, and no further correlation pushes race with them. Flush on
  // the way out so records buffered before the stop are not lost when the
  // application switches instrumentation off around its own AtSync. The flush
  // drives the buffer-completed callback, which takes cupti_queue_lock_ -- a
  // different mutex from the one held here, so this cannot deadlock.
  gm.cupti_tracing_active_.store(false, std::memory_order_relaxed);
  CUPTI_SAFE_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

  // Deliberately NOT detaching with cuptiFinalize(). Staying attached costs
  // ~1.3 ms per step even with every kind disabled, and detaching measured
  // cheaper (~2.3 ms per window against 1.3 ms per step) -- but it cannot be
  // done safely from here. The entry-method hooks check cupti_tracing_active_
  // and then call into CUPTI without holding this lock, so a thread that has
  // already passed that check can be inside cuptiActivityPushExternalCorrelationId
  // while this one finalizes underneath it, which corrupts CUPTI's allocator
  // and surfaces later as heap corruption in unrelated allocations. Reclaiming
  // that 1.3 ms needs the hooks made safe against detach first.
}

bool hapiCuptiTracingActive() {
  return CsvAccess(gpu_manager).cupti_tracing_active_.load(
      std::memory_order_relaxed);
}

void hapiCuptiFinalize() {
  GPUManager& gm = CsvAccess(gpu_manager);
  if (!gm.cupti_initialized_) return;
  cudaDeviceSynchronize(); // Ensure all activity records are flushed
  gm.cupti_initialized_ = false;
  gm.cupti_tracing_active_.store(false, std::memory_order_relaxed);
  ++gm.cupti_generation_;

  CUPTI_SAFE_CALL(cuptiFinalize());
}

#else /* !HAPI_CUPTI_LB: no CUPTI in this build */

void hapiCuptiInit() {}
void hapiCuptiFinalize() {}
void hapiCuptiStartTracing() {}
void hapiCuptiStopTracing() {}
bool hapiCuptiTracingActive() { return false; }

#endif /* HAPI_CUPTI_LB */
#endif /* CMK_CUDA && CMK_LBDB_ON */

#ifndef HAPI_CUDA_CALLBACK
#if CSD_NO_SCHEDLOOP
#  error please disable CSD_NO_SCHEDLOOP to use HAPI
#endif
#endif

// Called by all PEs in Charm++ layer init
void hapiInit(char** argv) {
  // Before hapiInitCsv: device-manager creation already indexes by it.
  CpvInitialize(int, my_node_rank_local);
  hapiRefreshTopologyCache();
  if (!CmiInCommThread()) {
    if (CmiMyRank() == 0) {
      hapiInitCsv(argv); // Initialize per-process variables (GPUManager)
    }
    hapiInitCpv(); // Initialize per-PE variables

    CmiNodeBarrier(); // Ensure hapiInitCsv is done for all PEs within a logical node

    hapiMapping(argv); // Perform PE-device mapping

  /* GPU checkpoint/restart (memory daemon) path removed here; returns
   * with the shrink/expand series (plan item 10) if not superseded. */
    int& cpv_my_device = CpvAccess(my_device);
    hapiCheck(hapiSetDevice(cpv_my_device));

#ifndef HAPI_CUDA_CALLBACK
    // Register polling function to be invoked at every scheduler loop
    CcdCallOnConditionKeep(CcdSCHEDLOOP, (CcdCondFn)hapiPollEvents, NULL);
#endif
  }

  CmiNodeAllBarrier();

  if (CmiInCommThread()) {
    // FIXME: Comm. thread sets its device to be the same as worker thread 0
    hapiSetDevice(CsvAccess(gpu_manager).comm_thread_device);
  }

  shmInit();

  hapiRegisterCallbacks(); // Register callback functions
}


void hapiExit() {
  // Ensure all PEs have finished GPU work
  CmiNodeBarrier();

  /* GPU checkpoint/restart (memory daemon) path removed here; returns
   * with the shrink/expand series (plan item 10) if not superseded. */

  if (CmiMyRank() == 0) {
    shmCleanup();

    hapiExitCsv();
  }
}

// Initialize per-process variables
static void hapiInitCsv(char** argv) {
  // Create and initialize GPU Manager object
  CsvInitialize(GPUManager, gpu_manager);
  CsvAccess(gpu_manager).init();
  // CUPTI is attached lazily by hapiCuptiStartTracing, which the balancer
  // reaches through LBDatabase::TurnStatsOn. Attaching here instead would pay
  // the attach cost for the whole run even when the application only wants
  // instrumentation around its load-balancing steps.
}


#if CMK_CUDA && CMK_LBDB_ON
#ifdef HAPI_CUPTI_LB

// Find the DeviceManager that matches this kernel's device id.
static DeviceManager* findDeviceManager(GPUManager& gm, uint32_t device_id) {
  for (DeviceManager& dm : gm.device_managers) {
    if ((uint32_t)dm.global_index == device_id) return &dm;
  }
  return nullptr;
}

// Compute the number of SMs this kernel occupies while running.
// Uses the CUDA occupancy model: theoretical max_active_blocks_per_sm is
// limited by (a) max blocks per SM, (b) warp count, (c) register pressure,
// (d) shared memory. Then:
//   sms_used = min(num_sms, ceil(total_blocks / max_active_blocks_per_sm))
static int computeKernelSMs(const DeviceManager& dm,
                            const CUpti_ActivityKernel4* k) {
  if (!dm.props_initialized || dm.multi_processor_count <= 0) return 1;

  uint64_t threads_per_block =
      (uint64_t)k->blockX * (uint64_t)k->blockY * (uint64_t)k->blockZ;
  uint64_t total_blocks =
      (uint64_t)k->gridX * (uint64_t)k->gridY * (uint64_t)k->gridZ;
  if (threads_per_block == 0 || total_blocks == 0) return 1;

  // Warp-count limit: maxThreadsPerSM / threadsPerBlock (rounded down).
  int limit_warps =
      dm.max_threads_per_sm > 0
          ? (int)(dm.max_threads_per_sm / threads_per_block)
          : INT_MAX;
  if (limit_warps <= 0) limit_warps = 1;

  // Block-count limit (CUDA 11+; 0 means not available -> use a large value).
  int limit_blocks = dm.max_blocks_per_sm > 0 ? dm.max_blocks_per_sm : INT_MAX;

  // Register-pressure limit.
  uint64_t regs_per_block = (uint64_t)k->registersPerThread * threads_per_block;
  int limit_regs = INT_MAX;
  if (regs_per_block > 0 && dm.max_registers_per_sm > 0) {
    uint64_t r = (uint64_t)dm.max_registers_per_sm / regs_per_block;
    limit_regs = r > INT_MAX ? INT_MAX : (int)r;
    if (limit_regs <= 0) limit_regs = 1;
  }

  // Shared-memory limit.
  uint64_t smem_per_block =
      (uint64_t)k->staticSharedMemory + (uint64_t)k->dynamicSharedMemory;
  int limit_smem = INT_MAX;
  if (smem_per_block > 0 && dm.max_shared_mem_per_sm > 0) {
    uint64_t s = (uint64_t)dm.max_shared_mem_per_sm / smem_per_block;
    limit_smem = s > INT_MAX ? INT_MAX : (int)s;
    if (limit_smem <= 0) limit_smem = 1;
  }

  int max_active_blocks_per_sm =
      std::min(std::min(limit_blocks, limit_warps),
               std::min(limit_regs, limit_smem));
  if (max_active_blocks_per_sm < 1) max_active_blocks_per_sm = 1;

  uint64_t sms_needed =
      (total_blocks + max_active_blocks_per_sm - 1) / max_active_blocks_per_sm;
  int sms_used = (int)std::min<uint64_t>(sms_needed,
                                         (uint64_t)dm.multi_processor_count);
  if (sms_used < 1) sms_used = 1;
  return sms_used;
}

void hapiProcessCuptiBuffers() {
  GPUManager& gm = CsvAccess(gpu_manager);
  hapiPopulateDeviceProps(gm);  // lazy: device_managers is ready by now

  // A kernel record can be parsed before the correlation record that names it:
  // correlation and kernel records are queued at different points in the
  // launch's life and land in buffers that complete independently. Only those
  // kernels are parked for a second pass; one whose correlation is already
  // known is filed immediately, so the common case never holds two copies of
  // every record.
  struct PendingKernel {
    uint32_t       correlation_id;
    LBKernelRecord rec;
  };
  std::vector<PendingKernel> pending;
  pending.reserve(gm.cupti_pending_hint_);

  uint32_t kernel_count = 0;
  uint32_t object_corr_count = 0;
  uint32_t invalid_duration_count = 0;
  uint32_t attributed = 0;
  uint32_t unattributed = 0;
  uint32_t unresolved_token = 0;
  uint32_t deferred = 0;

  // Resolve each distinct token once rather than once per kernel record. A
  // round holds far more kernels than objects, and this lock is the same one
  // every entry method needs, so taking it per record would both dominate the
  // drain and stall PEs that are still running.
  struct ResolvedToken {
    LDObjKey key{};
    bool valid = false;
  };
  std::unordered_map<uint64_t, ResolvedToken> resolved_tokens;

  auto fileKernel = [&](const LBKernelRecord& rec, uint64_t object_token) {
    if (object_token == HAPI_CUPTI_NO_OBJECT) {
      gm.cupti_unattributed_kernels_.push_back(rec);
      unattributed++;
      return;
    }
    auto memo = resolved_tokens.find(object_token);
    if (memo == resolved_tokens.end()) {
      ResolvedToken entry;
      {
        std::lock_guard<std::mutex> token_lock(gm.cupti_object_token_lock_);
        entry.valid = gm.cupti_object_tokens_.resolve(object_token, entry.key);
      }
      memo = resolved_tokens.emplace(object_token, entry).first;
    }
    if (!memo->second.valid) {
      gm.cupti_unattributed_kernels_.push_back(rec);
      unattributed++;
      unresolved_token++;
      return;
    }
    gm.cupti_obj_kernel_records_[memo->second.key].push_back(rec);
    attributed++;
  };

  while (true) {
    CuptiBufferItem item;

    // Pop one buffer from the queue
    {
      std::lock_guard<std::mutex> lk(gm.cupti_queue_lock_);
      if (gm.cupti_buffer_queue_.empty()) break;
      item = gm.cupti_buffer_queue_.front();
      gm.cupti_buffer_queue_.pop();
    }

    // A buffer CUPTI never wrote to (or one it handed back empty) has nothing
    // to parse, and passing it on would dereference whatever came back.
    if (item.buffer == NULL || item.validSize == 0) {
      free(item.buffer);
      continue;
    }

    // Parse records in this buffer
    CUpti_Activity *record = NULL;
    while (cuptiActivityGetNextRecord(item.buffer, item.validSize, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        CUpti_ActivityExternalCorrelation *corr = (CUpti_ActivityExternalCorrelation *)record;
        if (corr->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN) {
          object_corr_count++;
          gm.cupti_object_correlation_db_[corr->correlationId] = corr->externalId;
        }
      }
      else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
               record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        kernel_count++;
        CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;

        DeviceManager* dm = findDeviceManager(gm, kernel->deviceId);

        LBKernelRecord rec{};
        rec.start_ns  = kernel->start;
        rec.end_ns    = kernel->end;
        rec.device_id = kernel->deviceId;
        rec.sms_used  = dm ? computeKernelSMs(*dm, kernel) : 1;
        if (rec.end_ns <= rec.start_ns) invalid_duration_count++;

        auto object = gm.cupti_object_correlation_db_.find(kernel->correlationId);
        if (object != gm.cupti_object_correlation_db_.end()) {
          const uint64_t object_token = object->second;
          gm.cupti_object_correlation_db_.erase(object);
          fileKernel(rec, object_token);
        } else {
          pending.push_back({kernel->correlationId, rec});
          deferred++;
        }
      }
    }

    free(item.buffer);
  }

  // Second pass: correlations that arrived in a later buffer are now known.
  for (PendingKernel& pending_kernel : pending) {
    auto object =
        gm.cupti_object_correlation_db_.find(pending_kernel.correlation_id);
    if (object == gm.cupti_object_correlation_db_.end()) {
      gm.cupti_unattributed_kernels_.push_back(pending_kernel.rec);
      unattributed++;
      continue;
    }

    const uint64_t object_token = object->second;
    gm.cupti_object_correlation_db_.erase(object);
    fileKernel(pending_kernel.rec, object_token);
  }

  // Size next round's parked vector from this one: the straggler count is a
  // property of how CUPTI is batching buffers, which changes slowly.
  gm.cupti_pending_hint_ = deferred;

  // Every entry method pushes a correlation ID, and CUPTI emits a record for
  // each runtime call made under it -- memcpys, syncs and so on, not just
  // kernel launches. Those never match a kernel record. We flush before
  // draining, so any kernel that was going to arrive has arrived; whatever is
  // left over is non-kernel traffic and would otherwise accumulate without
  // bound (entry-method correlation emits on the order of 10^5 records a step).
  size_t object_corr_dropped = gm.cupti_object_correlation_db_.size();
  gm.cupti_object_correlation_db_.clear();

  if (cuptiDebugLevel() > 1) {
    CmiPrintf("HAPI[pe=%d]: hapiProcessCuptiBuffers kernels=%u "
              "object_correlations=%u attributed=%u unattributed=%u "
              "deferred=%u invalid_durations=%u unresolved_tokens=%u "
              "objects=%zu object_corr_dropped=%zu\n",
              CmiMyPe(), kernel_count, object_corr_count, attributed,
              unattributed, deferred, invalid_duration_count, unresolved_token,
              gm.cupti_obj_kernel_records_.size(), object_corr_dropped);
  }
}

// Convert this process's raw kernel timeline into one SM-utilization-
// normalized load per object, in seconds of whole-device occupancy.
//
// Per-device sweep-line over all kernel intervals. At each event (kernel start
// or end) the interval's device time is split among the kernels then running,
// in proportion to the SMs each occupies.
//
// Because the result is device-seconds of demand rather than elapsed time, it
// is (to first order) invariant to how the objects happen to be placed right
// now, which is what makes it usable as a load estimate for a *different*
// placement.
//
// This runs in the process that produced the records, before the stats leave
// for the central LB: every PE bound to a device is in the same process, so the
// timeline for that device is already complete here. If several processes share
// one GPU, each sees only its own kernels and the contention between processes
// is not modelled.
void hapiNormalizeCuptiLoads() {
  GPUManager& gm = CsvAccess(gpu_manager);
  gm.cupti_obj_norm_load_.clear();

  struct SweepKernel {
    LDObjKey obj_key;
    uint64_t start_ns;
    uint64_t end_ns;
    int      sms_used;
    bool     attributed;   // false => consumes SMs but earns no load
    // Whole-device occupancy this kernel earned, filled in by the sweep.
    double   demand;
  };
  std::unordered_map<uint32_t, std::vector<SweepKernel>> byDevice;
  for (const auto& kv : gm.cupti_obj_kernel_records_) {
    for (const LBKernelRecord& k : kv.second) {
      if (k.end_ns <= k.start_ns) continue;
      byDevice[k.device_id].push_back({kv.first, k.start_ns, k.end_ns,
                                       k.sms_used, true, 0.0});
    }
  }
  for (const LBKernelRecord& k : gm.cupti_unattributed_kernels_) {
    if (k.end_ns <= k.start_ns) continue;
    byDevice[k.device_id].push_back({LDObjKey{}, k.start_ns, k.end_ns,
                                     k.sms_used, false, 0.0});
  }

  size_t total_kernels = 0, devices_normalized = 0;
  for (auto& kv : byDevice) {
    std::vector<SweepKernel>& kernels = kv.second;
    if (kernels.empty()) continue;

    DeviceManager* dm = findDeviceManager(gm, kv.first);
    int total_sms = (dm != nullptr) ? dm->multi_processor_count : 0;
    if (total_sms <= 0) continue;  // unknown device size -- cannot normalize
    total_kernels += kernels.size();
    devices_normalized++;

    // Two events per kernel. END sorts before START on a tie so a kernel
    // ending at instant t does not briefly count alongside one starting at t.
    struct Event { uint64_t time; int kind; int kidx; };  // kind: 0=END, 1=START
    std::vector<Event> events;
    events.reserve(2 * kernels.size());
    for (int ki = 0; ki < (int)kernels.size(); ++ki) {
      events.push_back({kernels[ki].start_ns, 1, ki});
      events.push_back({kernels[ki].end_ns,   0, ki});
    }
    std::sort(events.begin(), events.end(),
              [](const Event& a, const Event& b) {
                if (a.time != b.time) return a.time < b.time;
                return a.kind < b.kind;
              });

    // Kernels holding the device right now. Order is irrelevant: the two loops
    // below sum sms_used and then credit each kernel its own share, and both
    // give the same answer in any order. (An earlier version handed SMs out
    // FIFO by submission, which is what the proportional split above replaced.)
    // A vector with swap-and-pop erase beats a node-based container here --
    // these sets are small, and every event does one insert or one erase.
    std::vector<int> active;
    active.reserve(kernels.size());

    uint64_t t_prev = events.front().time;
    for (const auto& ev : events) {
      if (ev.time > t_prev && !active.empty()) {
        double dt_s = (double)(ev.time - t_prev) / 1.0e9;

        // Split this interval's occupancy in proportion to what each active
        // kernel asked for.
        //
        // Handing SMs out FIFO by submission and stopping at the first kernel
        // that finds the pool empty does not work: every kernel behind that
        // point earns nothing for the interval, so whether an object is
        // credited depends on where its kernels land in the launch order
        // relative to its neighbours' -- which reshuffles whenever placement
        // changes. That makes the metric unstable in the one direction that
        // matters: a device running more objects oversubscribes harder, so more
        // of its kernels fall past the cut-off and its objects measure cheaper
        // than they are, and a balancer reading that sends the busiest node
        // still more work. Proportional sharing is also the better model of the
        // hardware: kernels that oversubscribe an SM pool time-share it rather
        // than running strictly in submission order.
        long want = 0;
        for (int ki : active) {
          if (kernels[ki].sms_used > 0) want += kernels[ki].sms_used;
        }
        if (want > 0) {
          // Split the interval's DEVICE TIME among the kernels holding the
          // device, in proportion to the SMs each asked for. The weights decide
          // how concurrent work is divided; they must not decide how much there
          // is to divide.
          //
          // Charging dt * sms_used/total_sms instead -- SM-occupancy seconds --
          // records a kernel that held the device for an interval at low
          // occupancy as almost no load. That is the right measure of
          // throughput and the wrong one for balancing: an object whose kernels
          // tie up the GPU for 10ms costs its PE 10ms whether they fill 4 SMs
          // or 80. The error is also not uniform, which is what makes it
          // dangerous: an imbalanced distribution gives lightly-loaded objects
          // small grids, so a node holding many of them runs many low-occupancy
          // kernels -- busy the whole interval, yet reporting almost nothing --
          // and the balancer reads it as idle and sends it still more work.
          //
          // Weighting by sms_used keeps a big kernel worth more than a small
          // one running beside it, while the interval total stays dt: summed
          // over a device, attributed demand equals its busy time.
          for (int ki : active) {
            const int used = kernels[ki].sms_used;
            if (used <= 0) continue;
            // Computed for unowned kernels too; they are still not credited to
            // any object below, but leaving their share uncomputed is what
            // makes such a loss invisible to the audit.
            kernels[ki].demand += dt_s * ((double)used / (double)want);
          }
        }
      }
      if (ev.kind == 1) {
        active.push_back(ev.kidx);
      } else {
        // Swap-and-pop: nothing here depends on the order of the remainder.
        auto it = std::find(active.begin(), active.end(), ev.kidx);
        if (it != active.end()) { *it = active.back(); active.pop_back(); }
      }
      t_prev = ev.time;
    }

    for (const SweepKernel& k : kernels) {
      if (k.demand <= 0.0) continue;
      if (!k.attributed) continue;
      gm.cupti_obj_norm_load_[k.obj_key] += k.demand;
    }
  }

  if (cuptiDebugLevel() > 1) {
    CmiPrintf("HAPI[pe=%d]: hapiNormalizeCuptiLoads  %zu kernels across %zu "
              "device(s) -> %zu objects\n",
              CmiMyPe(), total_kernels, devices_normalized,
              gm.cupti_obj_norm_load_.size());
  }

}

void hapiPrepareCuptiLoads(uint64_t epoch) {
  GPUManager& gm = CsvAccess(gpu_manager);
  std::lock_guard<std::mutex> lk(gm.cupti_prepare_lock_);
  // Already built for this epoch or a later one.
  if (gm.cupti_loads_ready_ && epoch <= gm.cupti_loads_epoch_) return;

  const bool timeIt = (getenv("CHARM_LB_CUPTI_TIME") != nullptr);
  const double t0 = timeIt ? CmiWallTimer() : 0.0;
  // Only flush while CUPTI is attached: an application driving its own
  // instrumentation window may already have switched tracing off, and its stop
  // path flushed on the way out, so there is nothing left to pull.
  //
  // FLUSH_FORCED rather than a plain flush: without it CUPTI hands back only
  // the buffers it considers complete, and the kernel records still sitting in
  // a partly-filled buffer are read a round late or not at all.
  if (hapiCuptiTracingActive())
    CUPTI_SAFE_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  const double t1 = timeIt ? CmiWallTimer() : 0.0;
  hapiProcessCuptiBuffers();
  const double t2 = timeIt ? CmiWallTimer() : 0.0;
  hapiNormalizeCuptiLoads();
  // Close the measurement window exactly where it was read. These records were
  // just consumed; anything recorded from here on belongs to the next round.
  //
  // Dropping them at MigrationDone instead -- after the strategy has run and
  // the migrations have executed -- means kernels running through all of that
  // are recorded and then discarded, belonging to no round at all, and the
  // amount discarded depends on how long that round's LB step took. That is a
  // feedback loop between migration count and measured load.
  gm.cupti_obj_kernel_records_.clear();
  gm.cupti_unattributed_kernels_.clear();
  if (timeIt) {
    const double t3 = CmiWallTimer();
    CmiPrintf("[LBCUPTI pe=%d] flush=%.3fs process=%.3fs normalize=%.3fs total=%.3fs\n",
              CmiMyPe(), t1 - t0, t2 - t1, t3 - t2, t3 - t0);
    fflush(stdout);
  }

  gm.cupti_loads_ready_ = true;
  gm.cupti_loads_epoch_ = epoch;
}

void hapiClearCuptiData() {
  GPUManager& gm = CsvAccess(gpu_manager);
  // Same lock as hapiPrepareCuptiLoads: this drops the maps that function
  // builds and that every PE reads, so it must not run underneath either.
  std::lock_guard<std::mutex> lk(gm.cupti_prepare_lock_);
  gm.cupti_loads_ready_ = false;
  gm.cupti_loads_epoch_ = 0;

  gm.cupti_obj_kernel_records_.clear();
  gm.cupti_unattributed_kernels_.clear();
  gm.cupti_obj_norm_load_.clear();
  // The correlation map is drained alongside the CUPTI buffers in
  // hapiProcessCuptiBuffers. Do not clear the object-token table: later epochs
  // must reuse the same token for the same full LB identity, and the per-PE
  // caches in front of it have no invalidation protocol.
}

#else /* !HAPI_CUPTI_LB */

void hapiProcessCuptiBuffers() {}
void hapiNormalizeCuptiLoads() {}
void hapiPrepareCuptiLoads(uint64_t epoch) {}
void hapiClearCuptiData() {}

#endif /* HAPI_CUPTI_LB */

// Build this round's per-object GPU loads exactly once per round, however many
// PE threads call in. A load balancer's per-PE barrier fires when THAT PE's
// objects are at AtSync, but the records are shared by the whole process, so
// the drain has to wait for the last PE rather than run on the first.
bool hapiCuptiArrive(uint64_t epoch, int expected) {
#ifdef HAPI_CUPTI_LB
  GPUManager& gm = CsvAccess(gpu_manager);
  std::lock_guard<std::mutex> lk(gm.cupti_arrive_lock_);
  // A new round resets the count. Rounds are strictly sequential -- a step
  // completes on a job-wide barrier before the next can start -- so a stale
  // count from a previous epoch can only mean that epoch is over.
  if (gm.cupti_arrive_epoch_ != epoch) {
    gm.cupti_arrive_epoch_ = epoch;
    gm.cupti_arrive_count_ = 0;
  }
  gm.cupti_arrive_count_++;
  if (gm.cupti_arrive_count_ < expected) return false;
  gm.cupti_arrive_count_ = 0;
  return true;
#else
  // Nothing to build, so nothing to wait for: every caller proceeds and reads
  // an empty load map.
  return true;
#endif
}

#endif /* CMK_CUDA && CMK_LBDB_ON */


// Initialize per-PE variables
static void hapiInitCpv() {
  // HAPI event-related
#ifndef HAPI_CUDA_CALLBACK
  CpvInitialize(std::queue<hapiEvent>, hapi_event_queue);
  CpvInitialize(std::queue<hapiEvent_t>, hapi_event_pool);
  // for(int i = 0; i < 8; i++) {
  //   hapiEvent_t ev;
  //   hapiEventCreateWithFlags(&ev, hapiEventDisableTiming);
  //   CpvAccess(hapi_event_pool).push(ev);
  // }
#endif
  CpvInitialize(int, n_hapi_events);
  CpvAccess(n_hapi_events) = 0;

  // Device mapping
  CpvInitialize(int, my_device);
  CpvInitialize(int, my_device_id);
  CpvAccess(my_device_id) = 0;
  CpvAccess(my_device) = 0;
  CpvInitialize(bool, device_rep);
  CpvAccess(device_rep) = false;
}

// Clean up per-process data
static void hapiExitCsv() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Destroy GPU Manager object
  csv_gpu_manager.destroy();

  // Release memory pool
  if (csv_gpu_manager.mempool_initialized_) {
    releasePool(csv_gpu_manager.mempool_free_bufs_);
  }
#ifndef HAPI_CUDA_CALLBACK
  auto& hapi_event_pool_ = CpvAccess(hapi_event_pool);
  while(!hapi_event_pool_.empty()) {
    hapiEventDestroy(hapi_event_pool_.front());
    hapi_event_pool_.pop();
  }
#endif
}

// Set up PE to GPU mapping, invoked from all PEs
// TODO: Support custom mappings
static void hapiMapping(char** argv) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  Mapping map_type = Mapping::RoundRobin; // Default is round robin
  char* gpumap = NULL;

  // Process +gpumap
  if (CmiGetArgStringDesc(argv, "+gpumap", &gpumap,
        "define pe to gpu device mapping")) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> PE-GPU mapping: %s\n", gpumap);
    }

    if (strcmp(gpumap, "none") == 0) {
      map_type = Mapping::None;
    } else if (strcmp(gpumap, "block") == 0) {
      map_type = Mapping::Block;
    } else if (strcmp(gpumap, "roundrobin") == 0) {
      map_type = Mapping::RoundRobin;
    } else {
      CmiAbort("Unsupported mapping type: %s, use one of \"none\", \"block\", "
          "\"roundrobin\"", gpumap);
    }
  }

  // No mapping specified, user assumes responsibility
  if (map_type == Mapping::None) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> User should explicitly select devices for PEs/chares\n");
    }
    return;
  }

  CmiAssert(map_type != Mapping::None);

  if (CmiMyRank() == 0) {
    // Count number of GPU devices used by each process
    int visible_device_count;
    hapiCheck(hapiGetDeviceCount(&visible_device_count));
    if (visible_device_count <= 0) {
      CmiAbort("Unable to perform PE-GPU mapping, no GPUs found!");
    }

    int& device_count = csv_gpu_manager.device_count;
    device_count = visible_device_count / (CmiNumNodes() / CmiNumPhysicalNodes());//?????

    // Handle the case where the number of GPUs per process are larger than
    // the number of PEs per process. This is needed because we currently don't
    // support each PE using more than one device.
    if (device_count > CmiNodeSize(CmiMyNode())) {
      if (CmiMyPe() == 0) {
        CmiPrintf("HAPI> Found more GPU devices (%d) than PEs (%d) per process, "
            "limiting to %d device(s) per process\n", device_count,
            CmiNodeSize(CmiMyNode()), CmiNodeSize(CmiMyNode()));
      }
      device_count = CmiNodeSize(CmiMyNode());
    }

    // We also need to handle the case where the number of GPUs are less than the 
    // number of processes launched on a physical node. Thus multiple processes can
    // share a GPU. In this case device_count would be 0, but instead, we will assign
    // at least one gpu to each process
    if(device_count == 0) {
      device_count = 1;
    }
    // Count number of PEs per device
    csv_gpu_manager.pes_per_device = CmiNodeSize(CmiMyNode()) / device_count;

    // Count number of devices on a physical node
    csv_gpu_manager.device_count_on_physical_node = visible_device_count;

    // Create a DeviceManager per GPU device
    std::vector<DeviceManager>& device_managers = csv_gpu_manager.device_managers;
    if(map_type == Mapping::RoundRobin) {
      for (int i = 0; i < device_count; i++) {
        device_managers.emplace_back(i, (device_count * CmiMyNodeRankLocal() + i) % visible_device_count);
      }
    }
    else if(map_type == Mapping::Block)
    {
      for (int i = 0; i < device_count; i++) {
        device_managers.emplace_back(i, (CmiMyNodeRankLocal() * visible_device_count + i)/(CmiNumNodes() / CmiNumPhysicalNodes()));
      }
    }
    else
    {
      CmiAbort("Unsupported mapping type!");
    }
  }

  if (CmiMyPe() == 0) {
    CmiPrintf("HAPI> Config: %d device(s) per process, %d PE(s) per device, %d device(s) per host\n",
        csv_gpu_manager.device_count, csv_gpu_manager.pes_per_device,
        csv_gpu_manager.device_count_on_physical_node);
  }

  CmiNodeBarrier();

  // Perform mapping and set device representative PE
  int my_rank = CmiMyRank();
  int& cpv_my_device = CpvAccess(my_device);
  int& cpv_my_device_id = CpvAccess(my_device_id);
  bool& cpv_device_rep = CpvAccess(device_rep);

  switch (map_type) {
    case Mapping::Block:{
      cpv_my_device_id   = (my_rank*csv_gpu_manager.device_count) / CmiNodeSize(CmiMyNode());
      cpv_my_device      = csv_gpu_manager.device_managers[cpv_my_device_id].global_index;
      if (my_rank < csv_gpu_manager.device_count) cpv_device_rep = true;
      firstRankForDevice = cpv_my_device;
    }
      break;
    case Mapping::RoundRobin: {
      cpv_my_device_id   = my_rank % csv_gpu_manager.device_count;
      cpv_my_device      = csv_gpu_manager.device_managers[cpv_my_device_id].global_index;
      if (my_rank < csv_gpu_manager.device_count) cpv_device_rep = true;
      firstRankForDevice = cpv_my_device;
    }
      break;
    default:  
      CmiAbort("Unsupported mapping type!");
  }
  
  hapiCheck(hapiSetDevice(cpv_my_device));
#if CMK_SMP
  CmiLock(csv_gpu_manager.device_mapping_lock);
#endif
  csv_gpu_manager.device_map.emplace(CmiMyPe(), &(csv_gpu_manager.device_managers[cpv_my_device_id]));
#if CMK_SMP
  CmiUnlock(csv_gpu_manager.device_mapping_lock);
#endif

  // Comm. thread will set its device to the same one as worker thread 0
  if (CmiMyRank() == 0) csv_gpu_manager.comm_thread_device = cpv_my_device;

  // Check if user opted in to POSIX shared memory optimizations for
  // inter-process GPU messaging
  bool use_shm = false;
  if (CmiGetArgFlagDesc(argv, "+gpushm",
        "enable shared memory optimizations for inter-process GPU messaging")) {
    use_shm = true;
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> Enabled POSIX shared memory optimizations for inter-process GPU messaging\n");
    }
  }

  if (CmiMyRank() == 0) {
    if (use_shm) {
      csv_gpu_manager.use_shm = true;
    }
    // csv_gpu_manager.test_field = true;
  }

  CmiNodeBarrier();

  /* These three are consumed unconditionally, even though only the
   * shared-memory path acts on them. CmiGetArgIntDesc also *removes* the
   * argument from argv, so leaving the calls inside the use_shm branch meant
   * that passing any of them without +gpushm left a stray '+arg' behind, and
   * the RTS then reported "not parsed by the RTS ... you may need to recompile
   * Charm++ with different options" -- pointing at a build problem that does
   * not exist, for a flag that was simply inapplicable. Parse them always and
   * say plainly when they do not apply. */
  int input_comm_buffer_size = 0;
  const bool given_comm_buffer = CmiGetArgIntDesc(argv, "+gpucommbuffer",
      &input_comm_buffer_size, "GPU communication buffer size (in MB); requires +gpushm");

  int input_lb_buffer_size = 0;
  const bool given_lb_buffer = CmiGetArgIntDesc(argv, "+gpulbbuffer",
      &input_lb_buffer_size, "GPU load balancing buffer size (in MB); requires +gpushm");

  int input_hapi_ipc_event_pool_size = 16;
  const bool given_event_pool = CmiGetArgIntDesc(argv, "+gpuipceventpool",
      &input_hapi_ipc_event_pool_size, "GPU IPC event pool size per PE; requires +gpushm");

  if (!csv_gpu_manager.use_shm && CmiMyPe() == 0 &&
      (given_comm_buffer || given_lb_buffer || given_event_pool)) {
    CmiPrintf("HAPI> Ignoring +gpucommbuffer/+gpulbbuffer/+gpuipceventpool: "
        "these configure the shared-memory inter-process path, which is off "
        "unless +gpushm is also given\n");
  }

  if (csv_gpu_manager.use_shm) {
    if (given_comm_buffer) {
      if (CmiMyRank() == 0) {
        // Round up size to the closest power of 2
        size_t comm_buffer_size = (size_t)input_comm_buffer_size * 1024 * 1024;
        int size_log2 = std::ceil(std::log2((double)comm_buffer_size));
        csv_gpu_manager.comm_buffer_size = (size_t)std::pow(2, size_log2);
      }
    }

    if (given_lb_buffer) {
      if (CmiMyRank() == 0) {
        csv_gpu_manager.lb_buffer_size =  (size_t)input_lb_buffer_size * 1024 * 1024;
      }
    }

    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> GPU communication buffer size: %zu MB "
          "(rounded up to the nearest power of two)\n",
          csv_gpu_manager.comm_buffer_size / (1024 * 1024));

      CmiPrintf("HAPI> GPU load balancing buffer size: %zu MB "
          "\n",
          csv_gpu_manager.lb_buffer_size / (1024 * 1024));
    }

    CmiNodeBarrier(); // Ensure device communication buffer size is set

    // Create device communication buffers
    // Should only be done by device representative threads
    if (cpv_device_rep) {
      DeviceManager* dm = csv_gpu_manager.device_map[CmiMyPe()];
#if CMK_SMP
      CmiLock(dm->lock);
#endif
      dm->create_comm_buffer(csv_gpu_manager.comm_buffer_size + csv_gpu_manager.lb_buffer_size, csv_gpu_manager.comm_buffer_size);
#if CMK_SMP
      CmiUnlock(dm->lock);
#endif
    }

    if (CmiMyRank() == 0) {
      csv_gpu_manager.hapi_ipc_event_pool_size_pe = input_hapi_ipc_event_pool_size;
      csv_gpu_manager.hapi_ipc_event_pool_size_total = input_hapi_ipc_event_pool_size * csv_gpu_manager.pes_per_device;
    }

    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> hapi IPC event pool size - %d per PE, %d per device\n",
          csv_gpu_manager.hapi_ipc_event_pool_size_pe, csv_gpu_manager.hapi_ipc_event_pool_size_total);
    }
  }

  // Check if P2P access should be enabled
  bool enable_peer = true; // Enabled by default
  if (CmiGetArgFlagDesc(argv, "+gpunopeer",
        "do not enable P2P access between visible GPU pairs")) {
    enable_peer = false;
  }

  // Enable P2P access to other visible devices
  // (only useful for multiple devices per process)
  // Should only be done by device representative threads
  if (enable_peer) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> Enabling P2P access between devices\n");
    }
    if (cpv_device_rep) {
      for (int i = 0; i < csv_gpu_manager.device_count; i++) {
        if (i != cpv_my_device) {
          int can_access_peer;

          hapiCheck(hapiDeviceCanAccessPeer(&can_access_peer, cpv_my_device, i));
          if (can_access_peer) {
            hapiDeviceEnablePeerAccess(i, 0);
          }
        }
      }
    }
  } else {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> P2P access between devices not enabled\n");
    }
  }
}

#ifndef HAPI_CUDA_CALLBACK
void recordEvent(hapiStream_t stream, const CkCallback& cb, void* cb_msg, hapiWorkRequest* wr = NULL) {
  // create hapi event / get hapi event from the pool and insert into stream
  hapiEvent_t ev;
  auto& hapi_event_pool_local = CpvAccess(hapi_event_pool);
  if(hapi_event_pool_local.size() == 0) {
    hapiEventCreateWithFlags(&ev, hapiEventDisableTiming);
  } else {
    ev = hapi_event_pool_local.front();
    hapi_event_pool_local.pop();
  }
  hapiEventRecord(ev, stream);

  hapiEvent hev(ev, cb, cb_msg, wr);

  // push event information in queue
  CpvAccess(hapi_event_queue).push(hev);

  // increase count so that scheduler can poll the queue
  CpvAccess(n_hapi_events)++;
}
#endif

inline static void hapiWorkRequestCleanup(hapiWorkRequest* wr) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.progress_lock_);
#endif

  // free device buffers
  csv_gpu_manager.freeBuffers(wr);

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.progress_lock_);
#endif

  // free hapiWorkRequest
  delete wr;
}

#ifdef HAPI_CUDA_CALLBACK
// Invokes user's host-to-device callback.
static void* hostToDeviceCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("hostToDeviceCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));
  wr->host_to_device_cb.send();

  // inform QD that the host-to-device transfer is complete
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

  return NULL;
}

// Invokes user's kernel execution callback.
static void* kernelCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("kernelCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));
  wr->kernel_cb.send();

  // inform QD that the kernel is complete
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

  return NULL;
}

// Frees device buffers and invokes user's device-to-host callback.
// Invoked regardless of the availability of the user's callback.
static void* deviceToHostCallback(void* arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("deviceToHostCallback", NVTXColor::Asbestos);
#endif
  hapiWorkRequest* wr = *((hapiWorkRequest**)((char*)arg + CmiMsgHeaderSizeBytes + sizeof(int)));
  wr->device_to_host_cb.send();

  hapiWorkRequestCleanup(wr);

  // inform QD that device-to-host transfer is complete
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

  return NULL;
}

// Used by lightweight HAPI.
static void* lightCallback(void *arg) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("lightCallback", NVTXColor::Asbestos);
#endif

  hapiCallbackMessage* conv_msg = (hapiCallbackMessage*)arg;

  // invoke user callback
  conv_msg->cb.send(conv_msg->cb_msg);

  // notify process to QD
  CmiAssert(hapiQdProcess);
  hapiQdProcess(1);

  return NULL;
}
#endif // HAPI_CUDA_CALLBACK

// Register callback functions. All PEs need to call this.
static void hapiRegisterCallbacks() {
#ifdef HAPI_CUDA_CALLBACK
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // FIXME: Potential race condition on assignments, but CmiAssignOnce
  // causes a hang at startup.
  csv_gpu_manager.host_to_device_cb_idx_
    = CmiRegisterHandler((CmiHandler)hostToDeviceCallback);
  csv_gpu_manager.kernel_cb_idx_
    = CmiRegisterHandler((CmiHandler)kernelCallback);
  csv_gpu_manager.device_to_host_cb_idx_
    = CmiRegisterHandler((CmiHandler)deviceToHostCallback);
  csv_gpu_manager.light_cb_idx_
    = CmiRegisterHandler((CmiHandler)lightCallback);
#endif
}

#ifdef HAPI_CUDA_CALLBACK
// Callback function invoked by the hapi runtime certain parts of GPU work are
// complete. It sends a converse message to the original PE to free the relevant
// device memory and invoke the user's callback. The reason for this method is
// that a thread created by the hapi runtime does not have access to any of the
// CpvDeclare'd variables as it is not one of the threads created by the Charm++
// runtime.
static void hapiCallback(void *data) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("hapiCallback", NVTXColor::Silver);
#endif

  // send message to the original PE
  char *conv_msg = (char*)data;
  int dstRank = *((int *)(conv_msg + CmiMsgHeaderSizeBytes));
  CmiPushPE(dstRank, conv_msg);
}

enum CallbackStage {
  AfterHostToDevice,
  AfterKernel,
  AfterDeviceToHost
};

static void addCallback(hapiWorkRequest *wr, CallbackStage stage) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // create converse message to be delivered to this PE after hapi callback
  char *conv_msg = (char *)CmiAlloc(CmiMsgHeaderSizeBytes + sizeof(int) +
                                  sizeof(hapiWorkRequest *)); // FIXME memory leak?
  *((int *)(conv_msg + CmiMsgHeaderSizeBytes)) = CmiMyRank();
  *((hapiWorkRequest **)(conv_msg + CmiMsgHeaderSizeBytes + sizeof(int))) = wr;

  int handlerIdx;
  switch (stage) {
    case AfterHostToDevice:
      handlerIdx = csv_gpu_manager.host_to_device_cb_idx_;
      break;
    case AfterKernel:
      handlerIdx = csv_gpu_manager.kernel_cb_idx_;
      break;
    case AfterDeviceToHost:
      handlerIdx = csv_gpu_manager.device_to_host_cb_idx_;
      break;
    default: // wrong type
      CmiFree(conv_msg);
      return;
  }
  CmiSetHandler(conv_msg, handlerIdx);

  // add callback into hapi stream
  hapiCheck(hapiLaunchHostFunc(wr->stream, hapiCallback, (void*)conv_msg));
}
#endif // HAPI_CUDA_CALLBACK

/******************** DEPRECATED ********************/
// User calls this function to offload work to the GPU.
void hapiEnqueue(hapiWorkRequest* wr) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("enqueue", NVTXColor::Pomegranate);
#endif

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.progress_lock_);
#endif

  // allocate device memory
  csv_gpu_manager.allocateBuffers(wr);

  // transfer data to device
  csv_gpu_manager.hostToDeviceTransfer(wr);

  // add host-to-device transfer callback
  if (wr->host_to_device_cb_set) {
    // while there is an ongoing workrequest, quiescence should not be detected
    // even if all PEs seem idle
    CmiAssert(hapiQdCreate);
    hapiQdCreate(1);

#ifdef HAPI_CUDA_CALLBACK
    addCallback(wr, AfterHostToDevice);
#else
    recordEvent(wr->stream, wr->host_to_device_cb, NULL);
#endif
  }

  // run kernel
  csv_gpu_manager.runKernel(wr);

  // add kernel callback
  if (wr->kernel_cb_set) {
    CmiAssert(hapiQdCreate);
    hapiQdCreate(1);

#ifdef HAPI_CUDA_CALLBACK
    addCallback(wr, AfterKernel);
#else
    recordEvent(wr->stream, wr->kernel_cb, NULL);
#endif
  }

  // transfer data to host
  csv_gpu_manager.deviceToHostTransfer(wr);

  // add device-to-host transfer callback
  CmiAssert(hapiQdCreate);
  hapiQdCreate(1);
#ifdef HAPI_CUDA_CALLBACK
  // always invoked to free memory
  addCallback(wr, AfterDeviceToHost);
#else
  if (wr->device_to_host_cb_set) {
    recordEvent(wr->stream, wr->device_to_host_cb, NULL, wr);
  }
  else {
    recordEvent(wr->stream, CkCallback::ignore, NULL, wr);
  }
#endif

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.progress_lock_);
#endif
}

/******************** DEPRECATED ********************/
// Creates a hapiWorkRequest object on the heap and returns it to the user.
hapiWorkRequest* hapiCreateWorkRequest() {
  return (new hapiWorkRequest);
}

hapiWorkRequest::hapiWorkRequest() :
    grid_dim(0), block_dim(0), shared_mem(0), runKernel(NULL), state(0),
    user_data(NULL), free_user_data(false)
{
#ifdef HAPI_TRACE
  trace_name = "";
#endif
#ifdef HAPI_INSTRUMENT_WRS
  chare_index = -1;
#endif

  // Use hapi per-thread default stream
  stream = hapiStreamPerThread;

  // Charm++ callbacks are not set by default
  host_to_device_cb = CkCallback(CkCallback::ignore);
  host_to_device_cb_set = false;
  kernel_cb = CkCallback(CkCallback::ignore);
  kernel_cb_set = false;
  device_to_host_cb = CkCallback(CkCallback::ignore);
  device_to_host_cb_set = false;
}

void hapiWorkRequestSetCallback(hapiWorkRequest* wr, void* cb) {
  wr->setCallback(*(CkCallback*)cb);
}

static void shmInit() {
  if (!CsvAccess(gpu_manager).use_shm) return;

  if (CmiMyRank() == 0) {
    if (!CmiInCommThread()) shmSetup();
    if (CmiMyNodeRankLocal() == 0) {
      if (!CmiInCommThread()) shmCreate(); // Create a per-host shared memory region
      CmiBarrier(); // FIXME: Only needs to be a host-wide barrier
    } else {
      CmiBarrier();
      if (!CmiInCommThread()) shmOpen(); // Open the shared memory region created by local logical node 0
    }
    if (!CmiInCommThread()) shmMap(); // Map the shared memory file into memory
  } else {
    CmiBarrier();
  }

  if (!CmiInCommThread()) CmiNodeBarrier(); // Ensure shared memory has been mapped into the logical node

  if (!CmiInCommThread()) ipcHandleCreate(); // Create hapi IPC handles

  // Ensure hapi IPC handles are available for all processes
  // Note: Causes a hang when this barrier is placed after CPU topology initialization
  // FIXME: This only needs to be a host-wide synchronization
  CmiBarrier();

  if (CmiMyRank() == 0) {
    if (!CmiInCommThread()) ipcHandleOpen(); // Open hapi IPC handles for accessing other processes' device memory
  }
}

static void shmSetup() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Set up shared memory file name
  csv_gpu_manager.shm_name.assign("charm-hapi-host");
  int host_id = CmiPhysicalNodeID(CmiMyPe());
  csv_gpu_manager.shm_name.append(std::to_string(host_id));
  const char* shm_name = csv_gpu_manager.shm_name.c_str();

  // Calculate shared memory region size
  csv_gpu_manager.shm_chunk_size = sizeof(hapiIpcMemHandle_t) +
      sizeof(hapi_ipc_event_shared) * csv_gpu_manager.hapi_ipc_event_pool_size_total;
  csv_gpu_manager.shm_size = csv_gpu_manager.shm_chunk_size *
    csv_gpu_manager.device_count * ((CmiNumNodes() / CmiNumPhysicalNodes()));
}

// Create POSIX shared memory region accessible to all processes on the same host
// Invoked by PE rank 0 of local logical node 0 (1 PE per host)
static void shmCreate() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Remove the shared memory file if it exists (could be left over from a
  // previous run that exited abnormally)
  struct stat stat_result;
  std::string stat_path("/dev/shm/");
  stat_path.append(csv_gpu_manager.shm_name);
  if (stat(stat_path.c_str(), &stat_result) == 0) {
    if (remove(stat_path.c_str())) {
      CmiAbort("Failure during shared memory file removal");
    }
  }

  // Create the shared memory file
  csv_gpu_manager.shm_file = shm_open(csv_gpu_manager.shm_name.c_str(),
      O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (csv_gpu_manager.shm_file < 0) {
    CmiError("Failure at shm_open");
    shmAbort();
  }

  // Set it to the appropriate size
  if (ftruncate(csv_gpu_manager.shm_file, 0) != 0) {
    CmiError("Failure at ftruncate");
    shmAbort();
  }
  if (ftruncate(csv_gpu_manager.shm_file, csv_gpu_manager.shm_size) != 0) {
    CmiError("Failure at ftruncate");
    shmAbort();
  }

  // Busywait until file is properly sized
  struct stat shm_file_stat;
  do {
    if (fstat(csv_gpu_manager.shm_file, &shm_file_stat) != 0) {
      CmiError("Failure at fstat");
      shmAbort();
    }
  } while (shm_file_stat.st_size != csv_gpu_manager.shm_size);
}

// Open POSIX shared memory region
// Invoked by logical nodes other than local rank 0
static void shmOpen() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Open the shared memory file
  csv_gpu_manager.shm_file = shm_open(csv_gpu_manager.shm_name.c_str(),
      O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (csv_gpu_manager.shm_file < 0) {
    CmiError("Failure at shm_open");
    shmAbort();
  }

  // Busywait until file is properly sized
  struct stat shm_file_stat;
  do {
    if (fstat(csv_gpu_manager.shm_file, &shm_file_stat) != 0) {
      CmiError("Failure at fstat");
      shmAbort();
    }
  } while (shm_file_stat.st_size != csv_gpu_manager.shm_size);
}

static void shmMap() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Map shared memory file into memory
  csv_gpu_manager.shm_ptr = mmap(NULL, csv_gpu_manager.shm_size,
      PROT_READ | PROT_WRITE, MAP_SHARED, csv_gpu_manager.shm_file, 0);
  if (csv_gpu_manager.shm_ptr == (void*)-1) {
    CmiError("Failure at mmap");
    shmAbort();
  }

  // Store pointer to my process' portion of the shared memory region
  csv_gpu_manager.shm_my_ptr = (void*)((char*)csv_gpu_manager.shm_ptr +
      csv_gpu_manager.shm_chunk_size * (csv_gpu_manager.device_count *
      CmiMyNodeRankLocal()));

  // Allocate memory for local storage
  for (int i = 0; i < csv_gpu_manager.device_count * ((CmiNumNodes() / CmiNumPhysicalNodes())); i++) {
    csv_gpu_manager.hapi_ipc_device_infos.emplace_back();
  }
}

static void shmAbort() {
  shmCleanup();
  CmiAbort("Failure in shared memory initialization");
}

// Clean up shared memory region
// Invoked by PE rank 0 of each process
static void shmCleanup() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  if (!csv_gpu_manager.use_shm) return;

  if (csv_gpu_manager.shm_ptr != NULL) {
    munmap(csv_gpu_manager.shm_ptr, csv_gpu_manager.shm_size);
  }

  if (csv_gpu_manager.shm_file != -1) {
    close(csv_gpu_manager.shm_file);
  }

  if (!csv_gpu_manager.shm_name.empty()) {
    shm_unlink(csv_gpu_manager.shm_name.c_str());
    csv_gpu_manager.shm_name.clear();
  }
}

// Create hapi IPC handles and populate shared memory region
// Invoked by all PEs
static void ipcHandleCreate() {
  // Only device reps should continue to perform the following operations
  // so that they are done only once per device
  if (!CpvAccess(device_rep)) return;

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int& cpv_my_device_id = CpvAccess(my_device_id);

  // Create hapi IPC memory handle in shared memory
  auto it = csv_gpu_manager.device_map.find(CmiMyPe());
  if (it == csv_gpu_manager.device_map.end()) {
    CmiAbort("PE not found in device_map during ipcHandleCreate");
  }
  DeviceManager& my_dm = *(it->second);
  auto comm_buffer = my_dm.get_comm_buffer();
  CmiAssert(comm_buffer);

  // Use local device index (0 to device_count-1) for shm_mem_handle offset
  // int local_device_idx = my_dm.local_index;
  hapiIpcMemHandle_t* shm_mem_handle = (hapiIpcMemHandle_t*)((char*)csv_gpu_manager.shm_my_ptr +
      csv_gpu_manager.shm_chunk_size * cpv_my_device_id);

  void* device_ptr = comm_buffer->base_ptr;
  hapiCheck(hapiIpcGetMemHandle(shm_mem_handle, device_ptr));

  // Create hapi IPC events and store them locally (in hapi_ipc_device_info),
  // and create corresponding IPC handles in shared memory
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id];
  hapi_ipc_event_shared* shm_event_shared = (hapi_ipc_event_shared*)((char*)shm_mem_handle + sizeof(hapiIpcMemHandle_t));

  for (int i = 0; i < csv_gpu_manager.hapi_ipc_event_pool_size_total; i++) {
    hapi_ipc_event_shared* cur_shm_event_shared = shm_event_shared + i;

    my_device_info.event_pool_flags.push_back(0);
    my_device_info.event_pool_buff_offsets.push_back(0);
    my_device_info.src_event_pool.emplace_back();
    my_device_info.dst_event_pool.emplace_back();
    hapiCheck(hapiEventCreateWithFlags(&my_device_info.src_event_pool[i],
          hapiEventDisableTiming | hapiEventInterprocess));
    hapiCheck(hapiEventCreateWithFlags(&my_device_info.dst_event_pool[i],
          hapiEventDisableTiming | hapiEventInterprocess));
    hapiCheck(hapiIpcGetEventHandle(&cur_shm_event_shared->src_event_handle,
          my_device_info.src_event_pool[i]));
    hapiCheck(hapiIpcGetEventHandle(&cur_shm_event_shared->dst_event_handle,
          my_device_info.dst_event_pool[i]));
  }

  // Store device comm buffer ptr in local info (just in case)
  my_device_info.buffer = device_ptr;
}

// Open hapi IPC handles created by other processes
// Invoked by PE rank 0 of each process
static void ipcHandleOpen() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Loop through all processes on this host
  for (int i = 0; i < CmiNumNodes() / CmiNumPhysicalNodes(); i++) {
    if (i == CmiMyNodeRankLocal()) continue;

    // Loop through GPU devices per process
    for (int j = 0; j < csv_gpu_manager.device_count; j++) {
      int device_index = csv_gpu_manager.device_count * i + j;
      hapi_ipc_device_info& cur_device_info = csv_gpu_manager.hapi_ipc_device_infos[device_index];

      // Open memory handle
      hapiIpcMemHandle_t* shm_mem_handle =
        (hapiIpcMemHandle_t*)((char*)csv_gpu_manager.shm_ptr
            + csv_gpu_manager.shm_chunk_size * device_index);
      hapiCheck(hapiIpcOpenMemHandle(&cur_device_info.buffer, *shm_mem_handle,
            hapiIpcMemLazyEnablePeerAccess));

      // Open event handles
      hapi_ipc_event_shared* shm_event_shared =
        (hapi_ipc_event_shared*)((char*)shm_mem_handle + sizeof(hapiIpcMemHandle_t));

      cur_device_info.event_pool_flags.clear();
      cur_device_info.event_pool_buff_offsets.clear();

      for (int k = 0; k < csv_gpu_manager.hapi_ipc_event_pool_size_total; k++) {
        hapi_ipc_event_shared* cur_shm_event_shared = shm_event_shared + k;

        cur_device_info.src_event_pool.emplace_back();
        cur_device_info.dst_event_pool.emplace_back();
        hapiCheck(hapiIpcOpenEventHandle(&cur_device_info.src_event_pool[k],
              cur_shm_event_shared->src_event_handle));
        hapiCheck(hapiIpcOpenEventHandle(&cur_device_info.dst_event_pool[k],
              cur_shm_event_shared->dst_event_handle));
      }
    }
  }
}

/******************** DEPRECATED ********************/
// Need to be updated with the Tracing API.
static inline void gpuEventStart(hapiWorkRequest* wr, int* index,
                                 WorkRequestStage event, ProfilingStage stage) {
#ifdef HAPI_TRACE
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  gpuEventTimer* shared_gpu_events_ = csv_gpu_manager.gpu_events_;
  int shared_time_idx_ = csv_gpu_manager.time_idx_++;
  // shared_gpu_events_[shared_time_idx_].cmi_start_time = CmiWallTimer();
  shared_gpu_events_[shared_time_idx_].event_type = event;
  shared_gpu_events_[shared_time_idx_].trace_name = wr->trace_name;
  *index = shared_time_idx_;
  shared_gpu_events_[shared_time_idx_].stage = stage;
#ifdef HAPI_DEBUG
  CmiPrintf("[HAPI] start event %d of WR %s, profiling stage %d\n",
         event, wr->trace_name, stage);
#endif
#endif // HAPI_TRACE
}

/******************** DEPRECATED ********************/
// Need to be updated with the Tracing API.
static inline void gpuEventEnd(int index) {
#ifdef HAPI_TRACE
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  // csv_gpu_manager.gpu_events_[index].cmi_end_time = CmiWallTimer();
  traceUserBracketEvent(csv_gpu_manager.gpu_events_[index].stage,
                        csv_gpu_manager.gpu_events_[index].cmi_start_time,
                        csv_gpu_manager.gpu_events_[index].cmi_end_time);
#ifdef HAPI_DEBUG
  Cmiprintf("[HAPI] end event %d of WR %s, profiling stage %d\n",
          csv_gpu_manager.gpu_events_[index].event_type,
          csv_gpu_manager.gpu_events_[index].trace_name,
          csv_gpu_manager.gpu_events_[index].stage);
#endif
#endif // HAPI_TRACE
}

static inline void hapiWorkRequestStartTime(hapiWorkRequest* wr) {
#ifdef HAPI_INSTRUMENT_WRS
  // wr->phase_start_time = CmiWallTimer();
#endif
}

static inline void profileWorkRequestEvent(hapiWorkRequest* wr,
                                           WorkRequestStage event) {
#ifdef HAPI_INSTRUMENT_WRS
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.inst_lock_);
#endif

  if (csv_gpu_manager.init_instr_) {
    // double tt = CmiWallTimer() - (wr->phase_start_time);
    int index = wr->chare_index;
    char type = wr->comp_type;
    char phase = wr->comp_phase;

    std::vector<hapiRequestTimeInfo> &vec = csv_gpu_manager.avg_times_[index][type];
    if (vec.size() <= phase) {
      vec.resize(phase+1);
    }
    switch (event) {
      case DataSetup:
        vec[phase].transfer_time += tt;
        break;
      case KernelExecution:
        vec[phase].kernel_time += tt;
        break;
      case DataCleanup:
        vec[phase].cleanup_time += tt;
        vec[phase].n++;
        break;
      default:
        CmiPrintf("[HAPI] invalid event during profileWorkRequestEvent\n");
    }
  }
  else {
    CmiPrintf("[HAPI] instrumentation not initialized!\n");
  }

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.inst_lock_);
#endif
#endif // HAPI_INSTRUMENT_WRS
}

// Create a pool with n_slots slots.
// There are n_buffers[i] buffers for each buffer size corresponding to entry i.
// TODO list the alignment/fragmentation issues with either of two allocation schemes:
// if single, large buffer is allocated for each subpool
// if multiple, smaller buffers are allocated for each subpool
static void createPool(int *n_buffers, int n_slots, std::vector<BufferPool> &pools){
  std::vector<size_t>& mempool_boundaries = CsvAccess(gpu_manager).mempool_boundaries_;

  // initialize pools
  pools.resize(n_slots);
  for (int i = 0; i < n_slots; i++) {
    pools[i].size = mempool_boundaries[i];
    pools[i].head = NULL;
  }

  int device;
  hapiDeviceProp device_prop;
  hapiCheck(hapiGetDevice(&device));
  hapiCheck(hapiGetDeviceProperties(&device_prop, device));

  // divide by # of PEs on physical node and multiply by # of PEs in logical node
  size_t available_memory = device_prop.totalGlobalMem /
                           CmiNumPesOnPhysicalNode(CmiPhysicalNodeID(CmiMyPe()))
                           * CmiMyNodeSize() * HAPI_MEMPOOL_SCALE;

  // pre-calculate memory per size
  int max_buffers = *std::max_element(n_buffers, n_buffers + n_slots);
  int n_buffers_to_allocate[n_slots];
  memset(n_buffers_to_allocate, 0, sizeof(n_buffers_to_allocate));
  size_t buf_size;
  while (available_memory >= mempool_boundaries[0] + sizeof(BufferPoolHeader)) {
    for (int i = 0; i < max_buffers; i++) {
      for (int j = n_slots - 1; j >= 0; j--) {
        buf_size = mempool_boundaries[j] + sizeof(BufferPoolHeader);
        if (i < n_buffers[j] && buf_size <= available_memory) {
          n_buffers_to_allocate[j]++;
          available_memory -= buf_size;
        }
      }
    }
  }

  // pin the host memory
  for (int i = 0; i < n_slots; i++) {
    buf_size = mempool_boundaries[i] + sizeof(BufferPoolHeader);
    int num_buffers = n_buffers_to_allocate[i];

    BufferPoolHeader* hd;
    BufferPoolHeader* previous = NULL;

    // pin host memory in a contiguous block for a slot
    void* pinned_chunk;
    hapiCheck(hapiMallocHost(&pinned_chunk, buf_size * num_buffers));

    // initialize header structs
    for (int j = num_buffers - 1; j >= 0; j--) {
      hd = reinterpret_cast<BufferPoolHeader*>(reinterpret_cast<unsigned char*>(pinned_chunk)
                                     + buf_size * j);
      hd->slot = i;
      hd->next = previous;
      previous = hd;
    }

    pools[i].head = previous;
    pools[i].chunk = pinned_chunk;
#ifdef HAPI_MEMPOOL_DEBUG
    pools[i].num = num_buffers;
#endif
  }
}

static void releasePool(std::vector<BufferPool> &pools){
  int device;
  hapiCheck(hapiGetDevice(&device));
  for (int i = 0; i < pools.size(); i++) {
    void* chunk = pools[i].chunk;
    if (chunk != NULL) {
      hapiCheck(hapiFreeHost(chunk));
    }
  }
  pools.clear();
}

static int findPool(size_t size){
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int boundary_array_len = csv_gpu_manager.mempool_boundaries_.size();
  if (size <= csv_gpu_manager.mempool_boundaries_[0]) {
    return 0;
  }
  else if (size > csv_gpu_manager.mempool_boundaries_[boundary_array_len-1]) {
    // create new slot
    csv_gpu_manager.mempool_boundaries_.push_back(size);

    BufferPool newpool;
    hapiCheck(hapiMallocHost((void**)&newpool.head, size + sizeof(BufferPoolHeader)));
    if (newpool.head == NULL) {
      CmiPrintf("[HAPI (%d)] findPool: failed to allocate newpool %d head, size %zu\n",
             CmiMyPe(), boundary_array_len, size);
      return -1;
    }
    newpool.size = size;
    newpool.chunk = (void *)newpool.head;
#ifdef HAPI_MEMPOOL_DEBUG
    newpool.num = 1;
#endif
    csv_gpu_manager.mempool_free_bufs_.push_back(newpool);

    BufferPoolHeader* hd = newpool.head;
    hd->next = NULL;
    hd->slot = boundary_array_len;

    return boundary_array_len;
  }
  for (int i = 0; i < csv_gpu_manager.mempool_boundaries_.size()-1; i++) {
    if (csv_gpu_manager.mempool_boundaries_[i] < size &&
        size <= csv_gpu_manager.mempool_boundaries_[i+1]) {
      return (i + 1);
    }
  }
  return -1;
}

static void* getBufferFromPool(int pool, size_t size){
  BufferPoolHeader* ret;
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  if (pool < 0 || pool >= csv_gpu_manager.mempool_free_bufs_.size()) {
    CmiPrintf("[HAPI (%d)] getBufferFromPool, pool: %d, size: %zu invalid pool\n",
           CmiMyPe(), pool, size);
#ifdef HAPI_MEMPOOL_DEBUG
    CmiPrintf("[HAPI (%d)] num: %d\n", CmiMyPe(),
           csv_gpu_manager.mempool_free_bufs_[pool].num);
#endif
    CmiAbort("[HAPI] exiting after invalid pool");
  }
  else if (csv_gpu_manager.mempool_free_bufs_[pool].head == NULL) {
    BufferPoolHeader* hd;
    hapiCheck(hapiMallocHost((void**)&hd, sizeof(BufferPoolHeader) +
                             csv_gpu_manager.mempool_free_bufs_[pool].size));
#ifdef HAPI_MEMPOOL_DEBUG
    CmiPrintf("[HAPI (%d)] getBufferFromPool, pool: %d, size: %zu expand by 1\n",
           CmiMyPe(), pool, size);
#endif
    if (hd == NULL) {
      CmiAbort("[HAPI] exiting after NULL hd from pool");
    }
    hd->slot = pool;
    return (void*)(hd + 1);
  }
  else {
    ret = csv_gpu_manager.mempool_free_bufs_[pool].head;
    csv_gpu_manager.mempool_free_bufs_[pool].head = ret->next;
#ifdef HAPI_MEMPOOL_DEBUG
    ret->size = size;
    csv_gpu_manager.mempool_free_bufs_[pool].num--;
#endif
    return (void*)(ret + 1);
  }
  return NULL;
}

static void returnBufferToPool(int pool, BufferPoolHeader* hd) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  hd->next = csv_gpu_manager.mempool_free_bufs_[pool].head;
  csv_gpu_manager.mempool_free_bufs_[pool].head = hd;
#ifdef HAPI_MEMPOOL_DEBUG
  csv_gpu_manager.mempool_free_bufs_[pool].num++;
#endif
}

hapiError_t hapiPoolMalloc(void** ptr, size_t size) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.mempool_lock_);
#endif

  if (!csv_gpu_manager.mempool_initialized_) {
    // create pool of page-locked memory
    int sizes[HAPI_MEMPOOL_NUM_SLOTS];
          /*256*/ sizes[0]  =  4;
          /*512*/ sizes[1]  =  2;
         /*1024*/ sizes[2]  =  2;
         /*2048*/ sizes[3]  =  4;
         /*4096*/ sizes[4]  =  2;
         /*8192*/ sizes[5]  =  6;
        /*16384*/ sizes[6]  =  5;
        /*32768*/ sizes[7]  =  2;
        /*65536*/ sizes[8]  =  1;
       /*131072*/ sizes[9]  =  1;
       /*262144*/ sizes[10] =  1;
       /*524288*/ sizes[11] =  1;
      /*1048576*/ sizes[12] =  1;
      /*2097152*/ sizes[13] =  2;
      /*4194304*/ sizes[14] =  2;
      /*8388608*/ sizes[15] =  2;
     /*16777216*/ sizes[16] =  2;
     /*33554432*/ sizes[17] =  1;
     /*67108864*/ sizes[18] =  1;
    /*134217728*/ sizes[19] =  7;
    createPool(sizes, HAPI_MEMPOOL_NUM_SLOTS, csv_gpu_manager.mempool_free_bufs_);
    csv_gpu_manager.mempool_initialized_ = true;

#ifdef HAPI_MEMPOOL_DEBUG
    CmiPrintf("[HAPI (%d)] done creating buffer pool\n", CmiMyPe());
#endif
  }

  int pool = findPool(size);
  if (pool < 0) {
    *ptr = nullptr;

#if CMK_SMP
    CmiUnlock(csv_gpu_manager.mempool_lock_);
#endif

    return hapiErrorMemoryAllocation;
  }
  *ptr = getBufferFromPool(pool, size);

#ifdef HAPI_MEMPOOL_DEBUG
  CmiPrintf("[HAPI (%d)] hapiPoolMalloc size %zu pool %d left %d\n",
      CmiMyPe(), size, pool, csv_gpu_manager.mempool_free_bufs_[pool].num);
#endif

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.mempool_lock_);
#endif

  return hapiSuccess;
}

hapiError_t hapiPoolFree(void* ptr) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  // Check if mempool was initialized
  if (!csv_gpu_manager.mempool_initialized_)
    return hapiErrorInitializationError;

  BufferPoolHeader* hd = ((BufferPoolHeader*)ptr) - 1;
  int pool = hd->slot;

#ifdef HAPI_MEMPOOL_DEBUG
  size_t size = hd->size;
#endif

#if CMK_SMP
  CmiLock(csv_gpu_manager.mempool_lock_);
#endif

  returnBufferToPool(pool, hd);

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.mempool_lock_);
#endif

#ifdef HAPI_MEMPOOL_DEBUG
  CmiPrintf("[HAPI (%d)] hapiPoolFree size %zu pool %d left %d\n",
         CmiMyPe(), size, pool,
         csv_gpu_manager.mempool_free_bufs_[pool].num);
#endif

  return hapiSuccess;
}

#ifdef HAPI_INSTRUMENT_WRS
void hapiInitInstrument(int n_chares, int n_types) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.inst_lock_);
#endif

  if (!csv_gpu_manager.init_instr_) {
    csv_gpu_manager.avg_times_.resize(n_chares);
    for (int i = 0; i < n_chares; i++) {
      csv_gpu_manager.avg_times_[i].resize(n_types);
    }
    csv_gpu_manager.init_instr_ = true;
  }

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.inst_lock_);
#endif
}

hapiRequestTimeInfo* hapiQueryInstrument(int chare, char type, char phase) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.inst_lock_);
#endif

  if (phase < csv_gpu_manager.avg_times_[chare][type].size()) {
    return &csv_gpu_manager.avg_times_[chare][type][phase];
  }
  else {
    return NULL;
  }

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.inst_lock_);
#endif
}

void hapiClearInstrument() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.inst_lock_);
#endif

  for (int chare = 0; chare < csv_gpu_manager.avg_times_.size(); chare++) {
    for (char type = 0; type < csv_gpu_manager.avg_times_[chare].size(); type++) {
      csv_gpu_manager.avg_times_[chare][type].clear();
    }
    csv_gpu_manager.avg_times_[chare].clear();
  }
  csv_gpu_manager.avg_times_.clear();
  csv_gpu_manager.init_instr_ = false;

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.inst_lock_);
#endif
}
#endif // HAPI_INSTRUMENT_WRS

// Poll HAPI events stored in the PE's queue. Current strategy is to process
// all successive completed events in the queue starting from the front.
// TODO Maybe we should make one pass of all events in the queue instead,
// since there might be completed events later in the queue.
void hapiPollEvents(void* param) {
#ifndef HAPI_CUDA_CALLBACK
  if (CpvAccess(n_hapi_events) <= 0) return;

  std::queue<hapiEvent>& queue = CpvAccess(hapi_event_queue);
  while (!queue.empty()) {
    hapiEvent hev = queue.front();
    if (hapiEventQuery(hev.event) == hapiSuccess) {
      queue.pop(); // TODO: investigate possible race condition with charm4py futures - temporarily resolved by popping here

      // invoke Charm++ callback if one was given
      hev.cb.send(hev.cb_msg);

      // clean up hapiWorkRequest
      if (hev.wr) {
        hapiWorkRequestCleanup(hev.wr);
      }
      CpvAccess(hapi_event_pool).push(hev.event);
      CpvAccess(n_hapi_events)--;

      // inform QD that an event was processed
      CmiAssert(hapiQdProcess);
      hapiQdProcess(1);
    }
    else {
      // stop going through the queue once we encounter a non-successful event
      break;
    }
  }
#endif
}

int hapiCreateStreams() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.stream_lock_);
#endif

  int ret = csv_gpu_manager.createStreams();

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.stream_lock_);
#endif

  return ret;
}

hapiStream_t hapiGetStream() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

#if CMK_SMP
  CmiLock(csv_gpu_manager.stream_lock_);
#endif

  hapiStream_t ret = csv_gpu_manager.getNextStream();

#if CMK_SMP
  CmiUnlock(csv_gpu_manager.stream_lock_);
#endif

  return ret;
}
#if CMK_CUDA && CMK_LBDB_ON
#ifdef HAPI_CUPTI_LB

// How many external-correlation IDs this PE has actually pushed and not yet
// popped. Tracing is switched on and off from inside entry methods, so a push
// can be skipped while its matching pop still runs (or the reverse). Pairing
// the pop against this count rather than against the tracing flag keeps
// CUPTI's stack balanced across those transitions; without it the pop reports
// CUPTI_ERROR_QUEUE_EMPTY and attribution drifts. Each Charm++ PE is its own
// thread, so thread_local is per-PE.
static thread_local int cupti_pushed_depth = 0;
// The detach generation this PE last observed; see GPUManager::cupti_generation_.
static thread_local uint64_t cupti_seen_generation = 0;

// This PE's view of the process-wide token table. Every entry method on a
// migratable chare needs its object's token, and the table behind it is shared
// by every PE in the process, so consulting it under the node-wide lock would
// serialize the whole process on one mutex for the length of the run.
//
// The table is append-only for the lifetime of the process, which is what makes
// this cache safe without any invalidation protocol: an entry, once correct,
// stays correct, including across migration (the destination PE simply misses
// once and interns the same token the source PE already has). Anything that
// gains the ability to clear or renumber GpuObjectTokenTable must also
// invalidate these caches.
static thread_local std::unordered_map<LDObjKey, uint64_t, LDObjKeyHash>
    cupti_local_object_tokens;

// Drop this PE's outstanding push count if CUPTI has been detached since we
// last looked -- the stack those pushes referred to no longer exists, so
// popping against it would report CUPTI_ERROR_QUEUE_EMPTY.
static inline void hapiCuptiSyncGeneration(GPUManager& gm) {
  if (cupti_seen_generation != gm.cupti_generation_) {
    cupti_seen_generation = gm.cupti_generation_;
    cupti_pushed_depth = 0;
  }
}

uint64_t hapiCuptiPushObjCorrelation() {
  GPUManager& gm = CsvAccess(gpu_manager);
  // Gated on tracing rather than initialization: this runs on every entry
  // method, so when tracing is off the whole body -- the active-object lookup
  // and two CUPTI calls -- must be skipped, not just wasted.
  if (!gm.cupti_tracing_active_.load(std::memory_order_relaxed)) return 0;
  hapiCuptiSyncGeneration(gm);

  // The CUPTI external ID is a process-local token for the complete LB object
  // key. Using CkMigratable::ckGetID() here loses the object-manager identity
  // and aliases equal element IDs from different chare arrays.
  uint64_t object_token = HAPI_CUPTI_NO_OBJECT;
  if (CkLocRec* active = CkActiveLocRec()) {
    const LDObjHandle& handle = active->getLdHandle();
    LDObjKey key;
    key.omID() = handle.omID();
    key.objID() = handle.objID();

    // Steady state is a PE-local hit: the shared lock is taken only the first
    // time this PE runs a given object, so it costs O(objects that ever run
    // here) acquisitions rather than one per entry method.
    auto cached = cupti_local_object_tokens.find(key);
    if (cached != cupti_local_object_tokens.end()) {
      object_token = cached->second;
    } else {
      {
        std::lock_guard<std::mutex> token_lock(gm.cupti_object_token_lock_);
        if (!gm.cupti_object_tokens_.intern(key, object_token))
          CmiAbort("HAPI: exhausted CUPTI object-correlation tokens");
      }
      cupti_local_object_tokens.emplace(key, object_token);
    }
  }

  // Always push, even with the sentinel, so that the matching pop always has
  // something to remove; an unbalanced stack would mis-attribute every
  // subsequent kernel.
  CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, object_token));
  ++cupti_pushed_depth;

  return object_token;
}

void hapiCuptiPopObjCorrelation() {
  // Runs the generation check even when tracing is off: a detach may have
  // happened between this entry method's push and its pop, and the stale count
  // has to be cleared here rather than on the next push.
  GPUManager& gm = CsvAccess(gpu_manager);
  hapiCuptiSyncGeneration(gm);

  // Pop exactly what was pushed. Checking the tracing flag here instead would
  // pop entries this PE never pushed, once tracing is switched on part-way
  // through an entry method.
  if (cupti_pushed_depth == 0 || !gm.cupti_initialized_) return;
  --cupti_pushed_depth;

  uint64_t tag;
  CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &tag));
}

#else /* !HAPI_CUPTI_LB */

uint64_t hapiCuptiPushObjCorrelation() { return 0; }
void hapiCuptiPopObjCorrelation() {}

#endif /* HAPI_CUPTI_LB */
#endif /* CMK_CUDA && CMK_LBDB_ON */

// Lightweight HAPI, to be invoked after data transfer or kernel execution.
void hapiAddCallback(hapiStream_t stream, const CkCallback& cb, void* cb_msg) {
#ifndef HAPI_CUDA_CALLBACK
  // record hapi event
  recordEvent(stream, cb, cb_msg);
#else
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  /* FIXME works for now (faster too), but CmiAlloc might not be thread-safe
#if CMK_SMP
  CmiLock(csv_gpu_manager.queue_lock_);
#endif
*/

  // create converse message to be delivered to this PE after hapi callback
  hapiCallbackMessage* conv_msg = (hapiCallbackMessage*)CmiAlloc(sizeof(hapiCallbackMessage)); // FIXME memory leak?
  conv_msg->rank = CmiMyRank();
  conv_msg->cb = cb;
  conv_msg->cb_msg = cb_msg;
  CmiSetHandler(conv_msg, csv_gpu_manager.light_cb_idx_);

  // push into hapi stream
  hapiCheck(hapiLaunchHostFunc(stream, hapiCallback, (void*)conv_msg));

  /*
#if CMK_SMP
  CmiUnlock(csv_gpu_manager.queue_lock_);
#endif
*/
#endif

  // while there is an ongoing workrequest, quiescence should not be detected
  // even if all PEs seem idle
  CmiAssert(hapiQdCreate);
  hapiQdCreate(1);
}

void hapiAddCallback(hapiStream_t stream, void* cb, void* cb_msg) {
  hapiAddCallback(stream, *(CkCallback*)cb, cb_msg);
}

// hapiError_t hapiMemcpyAsync(void* dst, const void* src, size_t count, hapiMemcpyKind kind, hapiStream_t stream = 0) {
//   hapiError_t err;
// #if CMK_LBDB_ON
//   hapiEvent_t start;

//   hapiEventCreate(&start);
//   hapiEventRecord(start, stream);
// #endif

//   err = hapiMemcpyAsync(dst, src, count, kind, stream);
// #if CMK_LBDB_ON
//   hapiRecordTime(stream, start);  
// #endif
//   return err;
// }

// hapiError_t hapiMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hapiMemcpyKind kind, hapiStream_t stream = 0) {
//   hapiError_t err;
// #if CMK_LBDB_ON
//   hapiEvent_t start;

//   hapiEventCreate(&start);
//   hapiEventRecord(start, stream);
// #endif
//   err = hapiMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
// #if CMK_LBDB_ON
//   hapiRecordTime(stream, start);
// #endif
//   return err;
// }


void hapiErrorDie(hapiError_t retCode, const char* code, const char* file, int line) {
  if (retCode != hapiSuccess) {
    fprintf(stderr, "Fatal hapi Error [%d] %s at %s:%d\n", retCode, hapiGetErrorString(retCode), file, line);
    CmiAbort("Exit due to hapi error");
  }
}

uint64_t hapiMyDevice() {
  int physical_node_id = CmiPhysicalNodeID(CmiMyPe());
  int my_device = CpvAccess(my_device);
  return (static_cast<uint64_t>(physical_node_id) << 32) | my_device;
}

int hapiMyDeviceTotalSMs() {
  GPUManager& gm = CsvAccess(gpu_manager);
  int local_id = CpvAccess(my_device_id);
  if (local_id < 0 || local_id >= (int)gm.device_managers.size()) return 0;
  DeviceManager& dm = gm.device_managers[local_id];
  if (!dm.props_initialized) {
    // The lazy population in hapiProcessCuptiBuffers has not run yet (or CUPTI
    // is not in this build at all), so ask for just this one attribute.
    int count = 0;
    hapiCheck(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount,
                                     dm.global_index));
    return count;
  }
  return dm.multi_processor_count;
}

