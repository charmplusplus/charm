#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <queue>
#include <atomic>
#include <vector>
#include <set>
#include <map>
#include <mutex>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sched.h>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include <cuda.h>

#include "hapi_portable.h"
#include "converse.h"
#include "conv-mach-opt.h" /* for CMK_CUDA */
#include "ckrescale.h"
#include "charm++.h"

#include "hapi.h"
#include "hapi_impl.h"
#include "gpumanager.h"
#ifdef HAPI_NVTX_PROFILE
#include "hapi_nvtx.h"
#endif

#if CMK_LBDB_ON
#include <cupti.h>
#include "LBManager.h"
#include "ck.h"
#include "cklocrec.h"

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

#define SERVER_FIFO_TEMPLATE "/tmp/server_pipe_%ld"
#define CLIENT_FIFO_TEMPLATE "/tmp/client_pipe_%ld"
#define BUFFER_SIZE 256
#define STREAM_BUF_SIZE 1024

#if defined HAPI_TRACE || defined HAPI_INSTRUMENT_WRS
extern "C" double CmiWallTimer();
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

void hapiSendMemoryRequest(char* msg, int size);

// Returns the local rank of the logical node (process) that the given PE belongs to
static inline int CmiNodeRankLocal(int pe) {
  // Logical node index % Number of logical nodes per physical node
  return CmiNodeOf(pe) % (CmiNumNodes() / CmiNumPhysicalNodes());
}

// Returns the local rank of the logical node that I belong to
static inline int CmiMyNodeRankLocal() {
  return CmiNodeRankLocal(CmiMyPe());
}

// HAPI internal function declarations
static void hapiInitCsv(char** argv);
static void hapiInitCpv();
static void hapiExitCsv();

static void hapiMapping(char** argv);
static void hapiRegisterCallbacks();

// CUDA IPC related functions
static void shmInit();
static void shmSetup();
static void shmCreate();
static void shmOpen();
static void shmMap();
static void shmAbort();
static void shmCleanup();
static void ipcHandleCreate();
static void ipcHandleOpen();

// Parse a byte count that may carry a K/M/G suffix, so a size threshold can be
// written the way people say it ("256K") in an environment variable. Anything
// unparseable yields SIZE_MAX, which for the direct IPC threshold means "never"
// -- the safe reading of a typo.
static size_t hapiParseByteSize(const char* s) {
  char* end = NULL;
  const unsigned long long value = strtoull(s, &end, 10);
  if (end == s) return SIZE_MAX;
  switch (*end) {
    case 'k': case 'K': return (size_t)value << 10;
    case 'm': case 'M': return (size_t)value << 20;
    case 'g': case 'G': return (size_t)value << 30;
    default:            return (size_t)value;
  }
}

#ifdef CMK_LBDB_ON
// Sentinel external-correlation ID meaning "no owning migratable object".
// Must not collide with a real chare ID -- 0 is a perfectly valid one.
static constexpr uint64_t HAPI_CUPTI_NO_OBJECT =
    GpuObjectTokenTable::noObjectToken();

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

// Populate DeviceManager with device attributes needed to compute per-kernel
// SM usage from CUPTI records. Queried once per local device.
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

    GpuDeviceDescriptor& descriptor = dm.descriptor;
    descriptor.instanceId =
        (static_cast<uint64_t>(static_cast<uint32_t>(CmiPhysicalNodeID(CmiMyPe())))
         << 32) |
        static_cast<uint32_t>(dm.global_index);
    descriptor.smCount = static_cast<uint32_t>(std::max(props.multiProcessorCount, 0));
    descriptor.computeMajor = static_cast<uint32_t>(std::max(props.major, 0));
    descriptor.computeMinor = static_cast<uint32_t>(std::max(props.minor, 0));
    // CUDA 13 dropped cudaDeviceProp::clockRate, so the device attribute is
    // the only way left to ask. It can legitimately fail (on a MIG instance,
    // for one); a zero here is what selects the Unknown-source path below.
    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev) !=
        cudaSuccess) {
      clock_khz = 0;
      cudaGetLastError();  // don't leave the failure for the next hapiCheck
    }
    descriptor.maxClockKHz = static_cast<uint32_t>(std::max(clock_khz, 0));
    descriptor.totalMemory = static_cast<uint64_t>(props.totalGlobalMem);
    descriptor.typeId = gpuStableDeviceType(
        props.name, descriptor.smCount, descriptor.computeMajor,
        descriptor.computeMinor, descriptor.maxClockKHz, descriptor.totalMemory);
    gpuDerivePeakRateScore(descriptor.smCount, descriptor.computeMajor,
                           descriptor.computeMinor, descriptor.maxClockKHz,
                           descriptor.peakRateScore, descriptor.peakRateSource);

    if (_lb_args.gpuScaling() &&
        gm.cupti_logged_device_types_.insert(descriptor.typeId).second) {
      CmiPrintf("HAPI GPU scaling: device type=%" PRIu64
                " name=%s cc=%u.%u sms=%u clock_khz=%u memory=%" PRIu64
                " rate=%.0f source=%s\n",
                descriptor.typeId, props.name, descriptor.computeMajor,
                descriptor.computeMinor, descriptor.smCount,
                descriptor.maxClockKHz, descriptor.totalMemory,
                descriptor.peakRateScore,
                gpuPeakRateSourceName(descriptor.peakRateSource));
      // An Unknown source means a property the prior needs was missing and a
      // nominal value stood in for it. The relative ordering against a fully
      // described device is then only as good as that guess, so say so once.
      if (descriptor.peakRateSource == GpuPeakRateSource::Unknown)
        CmiPrintf("HAPI GPU scaling: device type=%" PRIu64
                  " reported no usable clock rate; the cross-GPU prior for it "
                  "assumes %u kHz and will be corrected by observation\n",
                  descriptor.typeId, gpuNominalClockKHz());
    }
    dm.props_initialized = true;
  }
}

// Kept for callers that want tracing up before the balancer asks for it;
// hapiCuptiStartTracing attaches on its own, so this is not needed at startup.
void hapiCuptiInit() { hapiCuptiStartTracing(); }

// Attaching CUPTI to the process is NOT free even when no activity kind is
// enabled -- measured at ~1.3 ms per step on a 4-PE pic2d run, which is most of
// the cost that remains once tracing itself is windowed. So attach here and
// detach in hapiCuptiStopTracing, rather than staying attached for the whole
// run. Enabling an activity kind is separately what makes records flow.
void hapiCuptiStartTracing() {
  GPUManager& gm = CsvAccess(gpu_manager);
  // Every PE thread reaches this through its own LBDatabase::TurnStatsOn, so
  // the check and the enable must be one atomic step -- otherwise several
  // threads each enable the same activity kinds.
  std::lock_guard<std::mutex> lk(gm.cupti_tracing_lock_);
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
  CmiPrintf("HAPI: Finalizing CUPTI...\n");
  cudaDeviceSynchronize(); // Ensure all activity records are flushed
  GPUManager& gm = CsvAccess(gpu_manager);
  if(gm.cupti_initialized_== false) return;
  gm.cupti_initialized_ = false;
  gm.cupti_tracing_active_.store(false, std::memory_order_relaxed);
  ++gm.cupti_generation_;

  CUPTI_SAFE_CALL(cuptiFinalize());
}
#endif

#ifndef HAPI_CUDA_CALLBACK
#if CSD_NO_SCHEDLOOP
#  error please disable CSD_NO_SCHEDLOOP to use HAPI
#endif
#endif

// Called by all PEs in Charm++ layer init
void hapiInit(char** argv) {
  if (!CmiInCommThread()) {
    if (CmiMyRank() == 0) {
      hapiInitCsv(argv); // Initialize per-process variables (GPUManager)
    }
    hapiInitCpv(); // Initialize per-PE variables

    CmiNodeBarrier(); // Ensure hapiInitCsv is done for all PEs within a logical node

    hapiMapping(argv); // Perform PE-device mapping

#if CMK_SHRINK_EXPAND
    hapiStartMemoryDaemon(argv);
#else
    int& cpv_my_device = CpvAccess(my_device);
    hapiCheck(cudaSetDevice(cpv_my_device));
#endif

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


void hapiStartMemoryDaemon(char** argv)
{
#if CMK_SHRINK_EXPAND
  // start client FIFO
  long pid = getpid();
  char client_fifo_path[BUFFER_SIZE];
  sprintf(client_fifo_path, CLIENT_FIFO_TEMPLATE, pid);
  std::remove(client_fifo_path);
  mkfifo(client_fifo_path, 0666);

  int& cpv_my_device = CpvAccess(my_device);
  CkPrintf("Device = %i\n", cpv_my_device);
  hapiCheck(cudaSetDevice(cpv_my_device));

  if (CmiPhysicalRank(CmiMyPe()) != firstRankForDevice)
  {
    CmiBarrier();
    return;
  }

  char server_fifo_path[BUFFER_SIZE];
  sprintf(server_fifo_path, SERVER_FIFO_TEMPLATE, cpv_my_device);

  // Create a ready signal FIFO for synchronization
  if (!CmiGetArgFlagDesc(argv,"+shrinkexpand","Restarting of already running prcoess")) {
    char ready_fifo_path[BUFFER_SIZE];
    sprintf(ready_fifo_path, "/tmp/daemon_ready_%d", cpv_my_device);

    CmiPrintf("Parent: Waiting for daemon to be ready...\n");
    
    int ready_fd = open(ready_fifo_path, O_RDONLY);
    if (ready_fd == -1) {
      perror("Parent: open ready FIFO");
      CmiAbort("Failed to open ready FIFO");
    }
  
    char ready_signal;
    read(ready_fd, &ready_signal, 1);
    close(ready_fd);
    unlink(ready_fifo_path);  // Clean up
    
    CmiPrintf("Parent: Daemon is ready!\n");
  }
  
  CmiBarrier();
  return;
#endif
}

int hapiCheckpoint(void* devPtr, int size) {
  pid_t pid = getpid();

  char client_fifo_path[BUFFER_SIZE];
  sprintf(client_fifo_path, CLIENT_FIFO_TEMPLATE, pid);

  cudaIpcMemHandle_t ipc_handle;
  hapiCheck(cudaIpcGetMemHandle(&ipc_handle, devPtr));

  char msg_buf[BUFFER_SIZE];
  int offset = sprintf(msg_buf, "CKPT:%ld:%d:%d:", pid, CkMyPe(), size);
  memcpy(msg_buf + offset, &ipc_handle, sizeof(cudaIpcMemHandle_t));
  int total_size = offset + sizeof(cudaIpcMemHandle_t);

  hapiSendMemoryRequest(msg_buf, total_size);

  int client_fd = open(client_fifo_path, O_RDONLY);
  int alloc_id;
  read(client_fd, &alloc_id, sizeof(int));
  close(client_fd);

  return alloc_id;
}

void hapiRestore(void* devPtr, int size, int alloc_id) {
  pid_t pid = getpid();

  char client_fifo_path[BUFFER_SIZE];
  sprintf(client_fifo_path, CLIENT_FIFO_TEMPLATE, pid);

  char msg_buf[BUFFER_SIZE];
  sprintf(msg_buf, "GET:%ld:%d", pid, alloc_id);

  hapiSendMemoryRequest(msg_buf, strlen(msg_buf) + 1);

  int client_fd = open(client_fifo_path, O_RDONLY);
  cudaIpcMemHandle_t ipc_handle;
  read(client_fd, &ipc_handle, sizeof(cudaIpcMemHandle_t));
  close(client_fd);

  void* srcPtr;
  hapiCheck(cudaIpcOpenMemHandle(&srcPtr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));
  hapiCheck(cudaMemcpy(devPtr, srcPtr, size, cudaMemcpyDeviceToDevice));
  hapiCheck(cudaIpcCloseMemHandle(srcPtr));

  char free_msg[BUFFER_SIZE];
  sprintf(free_msg, "FREE:%ld:%d", pid, alloc_id);
  hapiSendMemoryRequest(free_msg, strlen(free_msg) + 1);

  client_fd = open(client_fifo_path, O_RDONLY);
  char status;
  read(client_fd, &status, sizeof(char));
  close(client_fd);
}

void hapiExit() {
  // Ensure all PEs have finished GPU work
  CmiPrintf("Exit called on PE %d\n", CmiMyPe());
  CmiNodeBarrier();

#if CMK_SHRINK_EXPAND
  char client_fifo_path[BUFFER_SIZE];
  sprintf(client_fifo_path, CLIENT_FIFO_TEMPLATE, getpid());

  if (!get_shrinkexpand_exit() && CmiPhysicalRank(CmiMyPe()) == firstRankForDevice)
  {
    char msg_buf[BUFFER_SIZE];
    sprintf(msg_buf, "KILL:%ld:0", getpid());
    hapiSendMemoryRequest(msg_buf, strlen(msg_buf) + 1);

    int client_fd = open(client_fifo_path, O_RDONLY);
    char status;
    read(client_fd, &status, sizeof(char));
    close(client_fd);
  }

  if (!get_shrinkexpand_exit())
  {
    // Attempt to delete the file
    if (std::remove(client_fifo_path) == 0) {
        CmiPrintf("File '%s' deleted successfully.\n", client_fifo_path);
    } else {
        CmiPrintf("Error deleting file '%s': %s\n", client_fifo_path, strerror(errno));
    }
  }
#endif

  if (CmiMyRank() == 0) {
    if (getenv("CHARM_ZC_STATS") != NULL) hapiIpcReportStats();

    // Safe to close peer mappings here and nowhere cheaper: the node barrier
    // above has already quiesced GPU work, and closing a mapping some copy is
    // still reading is an illegal access. See the note on hapiIpcFlushImportCache.
    hapiIpcFlushImportCache();

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


#ifdef CMK_LBDB_ON
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

  const bool scaling = _lb_args.gpuScaling();

  // A kernel record can be parsed before the correlation records that name it:
  // correlation and kernel records are queued at different points in the
  // launch's life and land in buffers that complete independently. Only those
  // kernels are parked for a second pass; one whose correlations are already
  // known is filed immediately, so the common case never holds two copies of
  // every record. The launch signature rides along with a parked kernel because
  // it is needed to rebucket if a work tag turns up, but it is deliberately not
  // stored in LBKernelRecord: there is one of those per launch.
  struct PendingKernel {
    uint32_t           correlation_id;
    GpuLaunchSignature launch;
    LBKernelRecord     rec;
  };
  std::vector<PendingKernel> pending;
  pending.reserve(gm.cupti_pending_hint_);

  uint32_t kernel_count = 0;
  uint32_t object_corr_count = 0;
  uint32_t work_tag_count = 0;
  uint32_t invalid_duration_count = 0;
  uint32_t hash_collision_count = 0;
  uint32_t attributed = 0;
  uint32_t unattributed = 0;
  uint32_t unresolved_token = 0;
  uint32_t deferred = 0;
  uint32_t lost_work_tags = 0;

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

  // Applies the work tag for this launch, if one has been parsed. Returns false
  // when the tag may still be in an unparsed buffer, in which case the bucket
  // is not settled and the kernel must be parked until the whole drain is done.
  auto applyWorkTag = [&](uint32_t correlation_id, const GpuLaunchSignature& launch,
                          LBKernelRecord& rec, bool last_chance) {
    // Already resolved during the parse loop. Without this the second pass
    // would look the tag up again, fail to find the entry it consumed itself,
    // and condemn a correctly tagged kernel as unmodelable.
    if (rec.has_explicit_work_tag) return true;

    auto work_tag = gm.cupti_work_tag_correlation_db_.find(correlation_id);
    if (work_tag != gm.cupti_work_tag_correlation_db_.end()) {
      rec.has_explicit_work_tag = true;
      rec.kernel_key.workBucket =
          gpuStableWorkBucket(launch, true, work_tag->second);
      gm.cupti_tagged_kernel_classes_.insert(rec.kernel_key.kernelClass);
      gm.cupti_work_tag_correlation_db_.erase(work_tag);
      return true;
    }

    // No tag found yet. Whether that is final depends on what is known about
    // this kernel class, because a tag record can be in a buffer this drain has
    // not parsed. Deciding per class rather than per application matters: an
    // application that tags one kernel usually leaves its others untagged, and
    // parking all of them would give back the fast path this drain exists to
    // preserve.
    const uint64_t kernelClass = rec.kernel_key.kernelClass;
    if (!last_chance) {
      // Confirmed untagged by a previous full drain: nothing to wait for.
      if (gm.cupti_untagged_kernel_classes_.count(kernelClass) != 0) return true;
      // Either known-tagged (so this is a missing tag, or one still unparsed)
      // or never seen before (so its first instance must not be committed to
      // the untagged bucket on incomplete information). Both need the full
      // drain first.
      return false;
    }

    // Every buffer has been parsed, so the absence is real. A legitimately
    // untagged class keeps the automatic launch-signature bucket and is
    // remembered so its later instances stay on the fast path.
    if (gm.cupti_tagged_kernel_classes_.count(kernelClass) == 0) {
      gm.cupti_untagged_kernel_classes_.insert(kernelClass);
      return true;
    }

    // The class has been seen tagged, so the tag went missing. The automatic
    // bucket is guaranteed by the hasExplicitTag discriminator to be a
    // different key from the one its tagged siblings use, and filing it there
    // would blend unrelated work sizes into a bucket the estimator treats as
    // one comparable population. Mark it unmodelable instead.
    rec.unmodelable = true;
    lost_work_tags++;
    return true;
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
        else if (_lb_args.gpuScaling() &&
                 corr->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
          work_tag_count++;
          gm.cupti_work_tag_correlation_db_[corr->correlationId] = corr->externalId;
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

        GpuLaunchSignature launch;
        bool bucket_settled = true;
        if (scaling) {
          launch.gridX = kernel->gridX;
          launch.gridY = kernel->gridY;
          launch.gridZ = kernel->gridZ;
          launch.blockX = kernel->blockX;
          launch.blockY = kernel->blockY;
          launch.blockZ = kernel->blockZ;
          launch.staticSharedMemory = kernel->staticSharedMemory;
          launch.dynamicSharedMemory = kernel->dynamicSharedMemory;
          rec.kernel_key.kernelClass = gpuStableKernelClass(kernel->name);
          rec.kernel_key.workBucket = gpuStableWorkBucket(launch, false, 0);

          if (_lb_args.debug() > 0) {
            const char* kernel_name = kernel->name != nullptr ? kernel->name : "";
            auto inserted = gm.cupti_kernel_names_.emplace(
                rec.kernel_key.kernelClass, kernel_name);
            if (!inserted.second && inserted.first->second != kernel_name)
              hash_collision_count++;
          }

          bucket_settled =
              applyWorkTag(kernel->correlationId, launch, rec, /*last_chance=*/false);
        }

        auto object = gm.cupti_object_correlation_db_.find(kernel->correlationId);
        if (bucket_settled && object != gm.cupti_object_correlation_db_.end()) {
          const uint64_t object_token = object->second;
          gm.cupti_object_correlation_db_.erase(object);
          fileKernel(rec, object_token);
        } else {
          pending.push_back({kernel->correlationId, launch, rec});
          deferred++;
        }
      }
    }

    free(item.buffer);
  }

  // Second pass: correlations that arrived in a later buffer are now known.
  for (PendingKernel& pending_kernel : pending) {
    if (scaling)
      applyWorkTag(pending_kernel.correlation_id, pending_kernel.launch,
                   pending_kernel.rec, /*last_chance=*/true);

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
  // kernel launches. Those never match a kernel record. We device-sync and
  // force-flush before draining, so any kernel that was going to arrive has
  // arrived; whatever is left over is non-kernel traffic and would otherwise
  // accumulate without bound.
  size_t object_corr_dropped = gm.cupti_object_correlation_db_.size();
  size_t work_tags_dropped = gm.cupti_work_tag_correlation_db_.size();
  gm.cupti_object_correlation_db_.clear();
  gm.cupti_work_tag_correlation_db_.clear();

  // A lost work tag is not just a lost sample: it would otherwise file the
  // kernel under a bucket its tagged siblings never use, so report it at the
  // first debug level rather than burying it with the bookkeeping counters.
  if (_lb_args.debug() > 0 && lost_work_tags > 0) {
    CmiPrintf("HAPI[pe=%d]: %u kernel record(s) of a tagged class arrived with "
              "no work tag and were excluded from the scaling model\n",
              CmiMyPe(), lost_work_tags);
  }

  if (_lb_args.debug() > 1) {
    CmiPrintf("HAPI[pe=%d]: hapiProcessCuptiBuffers kernels=%u "
              "object_correlations=%u work_tags=%u attributed=%u "
              "unattributed=%u deferred=%u invalid_durations=%u "
              "unresolved_tokens=%u hash_collisions=%u lost_work_tags=%u "
              "objects=%zu object_corr_dropped=%zu work_tags_dropped=%zu\n",
              CmiMyPe(), kernel_count, object_corr_count, work_tag_count,
              attributed, unattributed, deferred, invalid_duration_count,
              unresolved_token, hash_collision_count, lost_work_tags,
              gm.cupti_obj_kernel_records_.size(), object_corr_dropped,
              work_tags_dropped);
  }
}

// Convert this process's raw kernel timeline into one SM-utilization-
// normalized load per object, in seconds of whole-device occupancy.
//
// Per-device FIFO sweep-line over all kernel intervals. At each event (kernel
// start or end) the active kernels are walked in submission order (earliest
// start_ns first) and each is granted min(its sms_used, remaining capacity);
// its share of the device over the segment is integrated into the owning
// object's load. This matches the GPU's block-level FIFO scheduler: a running
// kernel keeps its allocation and a later arrival gets only what is left.
//
// Because the result is SM-seconds of demand rather than elapsed time, it is
// (to first order) invariant to how the objects happen to be placed right now,
// which is what makes it usable as a load estimate for a *different* placement.
//
// This runs in the process that produced the records, on rank 0, before the
// stats leave for the central LB: every PE bound to a device is in the same
// process, so the timeline for that device is already complete here. If
// several processes share one GPU, each sees only its own kernels and the
// contention between processes is not modelled.
void hapiNormalizeCuptiLoads() {
  GPUManager& gm = CsvAccess(gpu_manager);
  gm.cupti_obj_norm_load_.clear();
  gm.cupti_obj_epoch_costs_.clear();

  const bool scaling = _lb_args.gpuScaling();

  struct SweepKernel {
    LDObjKey obj_key;
    uint64_t start_ns;
    uint64_t end_ns;
    int      sms_used;
    bool     attributed;   // false => consumes SMs but earns no load
    GpuKernelKey kernel_key;
    bool     unmodelable;
    // Whole-device occupancy this kernel earned, filled in by the sweep. Held
    // per kernel rather than summed straight into the object so the same
    // numbers can also be aggregated per kernel identity.
    double   demand;
  };
  std::unordered_map<uint32_t, std::vector<SweepKernel>> byDevice;
  for (const auto& kv : gm.cupti_obj_kernel_records_) {
    for (const LBKernelRecord& k : kv.second) {
      if (k.end_ns <= k.start_ns) continue;
      byDevice[k.device_id].push_back({kv.first, k.start_ns, k.end_ns, k.sms_used,
                                       true, k.kernel_key, k.unmodelable, 0.0});
    }
  }
  for (const LBKernelRecord& k : gm.cupti_unattributed_kernels_) {
    if (k.end_ns <= k.start_ns) continue;
    byDevice[k.device_id].push_back({LDObjKey{}, k.start_ns, k.end_ns, k.sms_used,
                                     false, GpuKernelKey{}, true, 0.0});
  }

  // Per-object accumulators. Components are gathered in a hash map and only
  // flattened into the wire vector once, after the cap is applied.
  struct ObjectAccumulator {
    std::unordered_map<GpuKernelKey, GpuKernelEpochCost, GpuKernelKeyHash> components;
    double unmodeled = 0.0;
    // An object's kernels are almost always all on one device, but a migration
    // mid-round can split them. The device that did most of the work is the one
    // a destination prediction should scale from.
    double sourceDemand = 0.0;
    uint64_t sourceInstanceId = 0;
    uint64_t sourceTypeId = 0;
  };
  std::unordered_map<LDObjKey, ObjectAccumulator, LDObjKeyHash> accumulators;

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

    // Active set ordered by start_ns (then index, for stability) so that
    // iteration order is FIFO by submission.
    auto cmpActive = [&kernels](int a, int b) {
      if (kernels[a].start_ns != kernels[b].start_ns)
        return kernels[a].start_ns < kernels[b].start_ns;
      return a < b;
    };
    std::set<int, decltype(cmpActive)> active(cmpActive);

    uint64_t t_prev = events.front().time;
    for (const auto& ev : events) {
      if (ev.time > t_prev && !active.empty()) {
        double dt_s = (double)(ev.time - t_prev) / 1.0e9;
        int remaining = total_sms;
        for (int ki : active) {       // FIFO iteration
          if (remaining <= 0) break;
          int eff = std::min(kernels[ki].sms_used, remaining);
          if (eff <= 0) continue;
          remaining -= eff;
          if (!kernels[ki].attributed) continue;  // contention only, no owner
          kernels[ki].demand += dt_s * ((double)eff / (double)total_sms);
        }
      }
      if (ev.kind == 1) active.insert(ev.kidx);
      else              active.erase(ev.kidx);
      t_prev = ev.time;
    }

    for (const SweepKernel& k : kernels) {
      if (!k.attributed || k.demand <= 0.0) continue;
      gm.cupti_obj_norm_load_[k.obj_key] += k.demand;
    }

    if (!scaling) continue;

    const uint64_t instanceId = dm->descriptor.instanceId;
    const uint64_t typeId = dm->descriptor.typeId;
    std::unordered_map<LDObjKey, double, LDObjKeyHash> deviceDemand;
    for (const SweepKernel& k : kernels) {
      if (!k.attributed || k.demand <= 0.0) continue;
      deviceDemand[k.obj_key] += k.demand;

      ObjectAccumulator& acc = accumulators[k.obj_key];
      if (k.unmodelable) {
        acc.unmodeled += k.demand;
        continue;
      }
      GpuKernelEpochCost& component = acc.components[k.kernel_key];
      component.key = k.kernel_key;
      const double duration_s = (double)(k.end_ns - k.start_ns) / 1.0e9;
      // A rejected sample still has to be accounted for somewhere, or the
      // per-object total stops reconciling with the scalar load.
      if (!component.observe(k.demand, duration_s)) acc.unmodeled += k.demand;
    }
    for (const auto& objectDemand : deviceDemand) {
      ObjectAccumulator& acc = accumulators[objectDemand.first];
      if (objectDemand.second > acc.sourceDemand) {
        acc.sourceDemand = objectDemand.second;
        acc.sourceInstanceId = instanceId;
        acc.sourceTypeId = typeId;
      }
    }
  }

  if (scaling) {
    const std::size_t cap =
        static_cast<std::size_t>(_lb_args.gpuScalingMaxComponents());
    size_t capped_objects = 0;
    for (auto& kv : accumulators) {
      GpuObjectEpochCosts costs;
      costs.sourceInstanceId = kv.second.sourceInstanceId;
      costs.sourceTypeId = kv.second.sourceTypeId;
      costs.unmodeledGpuTime = kv.second.unmodeled;
      costs.components.reserve(kv.second.components.size());
      for (const auto& component : kv.second.components)
        costs.components.push_back(component.second);

      if (costs.components.size() > cap) capped_objects++;
      costs.enforceComponentCap(cap);
      // Canonical wire order, independent of hash iteration and of whether the
      // cap reordered anything, so every replica sees the same bytes.
      std::sort(costs.components.begin(), costs.components.end(),
                [](const GpuKernelEpochCost& left, const GpuKernelEpochCost& right) {
                  return left.key < right.key;
                });

      if (_lb_args.debug() > 0) {
        auto scalar = gm.cupti_obj_norm_load_.find(kv.first);
        const double expected =
            scalar == gm.cupti_obj_norm_load_.end() ? 0.0 : scalar->second;
        const double tolerance = 1.0e-9 + 1.0e-6 * expected;
        if (std::fabs(expected - costs.totalDemand()) > tolerance) {
          CmiPrintf("HAPI[pe=%d]: GPU summary does not reconcile: scalar=%.9f "
                    "components+residual=%.9f\n",
                    CmiMyPe(), expected, costs.totalDemand());
        }
      }

      gm.cupti_obj_epoch_costs_[kv.first] = std::move(costs);
    }

    if (_lb_args.debug() > 1) {
      CmiPrintf("HAPI[pe=%d]: GPU summaries objects=%zu capped=%zu cap=%zu\n",
                CmiMyPe(), gm.cupti_obj_epoch_costs_.size(), capped_objects, cap);
    }
  }

  if (_lb_args.debug() > 1) {
    CmiPrintf("HAPI[pe=%d]: hapiNormalizeCuptiLoads  %zu kernels across %zu "
              "device(s) -> %zu objects\n",
              CmiMyPe(), total_kernels, devices_normalized,
              gm.cupti_obj_norm_load_.size());
  }
}

// Build this round's per-object GPU loads, exactly once per round however many
// PE threads call in. Every PE needs cupti_obj_norm_load_ populated before it
// reads its own objects out of it, and the work itself must be done by one
// thread; holding the lock across both gives the readers their ordering without
// a barrier that every PE has to reach.
void hapiPrepareCuptiLoads() {
  GPUManager& gm = CsvAccess(gpu_manager);
  std::lock_guard<std::mutex> lk(gm.cupti_prepare_lock_);
  if (gm.cupti_loads_ready_) return;

  const bool timeIt = (getenv("CHARM_LB_CUPTI_TIME") != nullptr);
  const double t0 = timeIt ? CmiWallTimer() : 0.0;
  // Only flush while CUPTI is attached: an application driving its own
  // instrumentation window may already have switched tracing off, and its stop
  // path flushed on the way out, so there is nothing left to pull.
  if (hapiCuptiTracingActive())
    CUPTI_SAFE_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  const double t1 = timeIt ? CmiWallTimer() : 0.0;
  hapiProcessCuptiBuffers();
  const double t2 = timeIt ? CmiWallTimer() : 0.0;
  hapiNormalizeCuptiLoads();
  if (timeIt) {
    const double t3 = CmiWallTimer();
    CmiPrintf("[LBCUPTI pe=%d] flush=%.3fs process=%.3fs normalize=%.3fs total=%.3fs\n",
              CmiMyPe(), t1 - t0, t2 - t1, t3 - t2, t3 - t0);
    fflush(stdout);
  }

  gm.cupti_loads_ready_ = true;
}

void hapiClearCuptiData() {
  GPUManager& gm = CsvAccess(gpu_manager);
  // Same lock as hapiPrepareCuptiLoads: this drops the maps that function
  // builds and that every PE reads, so it must not run underneath either.
  std::lock_guard<std::mutex> lk(gm.cupti_prepare_lock_);
  gm.cupti_loads_ready_ = false;

  gm.cupti_obj_kernel_records_.clear();
  gm.cupti_unattributed_kernels_.clear();
  gm.cupti_obj_norm_load_.clear();
  gm.cupti_obj_epoch_costs_.clear();
  // Correlation maps are drained alongside the CUPTI buffers in
  // hapiProcessCuptiBuffers. Do not clear the object-token table: later epochs
  // must reuse the same token for the same full LB identity.
}

#endif


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
  csv_gpu_manager.map_type = map_type;

  if (map_type == Mapping::None) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> User should explicitly select devices for PEs/chares\n");
    }
    return;
  }

  CmiAssert(map_type != Mapping::None);

  if (CmiMyRank() == 0) {
    printf("number of physical nodes is %d\n", CmiNumPhysicalNodes());
    printf("number of nodes is %d\n", CmiNumNodes());
    printf("my rank is %d\n", CmiMyRank());
    // Count number of GPU devices used by each process
    int visible_device_count;
    hapiCheck(hapiGetDeviceCount(&visible_device_count));
    if (visible_device_count <= 0) {
      CmiAbort("Unable to perform PE-GPU mapping, no GPUs found!");
    }

    int& device_count = csv_gpu_manager.device_count;
    device_count = visible_device_count / (CmiNumNodes() / CmiNumPhysicalNodes());//?????
    ckout<<"device count "<<device_count<<endl;

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

  if (csv_gpu_manager.use_shm) {
    // Process device communication buffer parameters (in MB)
    int input_comm_buffer_size = 0;
    if (CmiGetArgIntDesc(argv, "+gpucommbuffer", &input_comm_buffer_size,
          "GPU communication buffer size (in MB)")) {
      if (CmiMyRank() == 0 && input_comm_buffer_size > 0) {
        // Round up to the next power of two, in integer arithmetic.
        //
        // This was pow(): `(size_t)std::pow(2, ceil(log2(bytes)))`. pow returns
        // a double and is not exact here -- on this toolchain pow(2,29) and
        // pow(2,31) both come back a hair under the true value, and the cast
        // truncated them to 2^29-1 and 2^31-1. Requests of 64, 128, 256 and
        // 1024 MB happened to survive; 512 and 2048 MB did not.
        //
        // A size one byte short of a power of two is not a rounding nuisance,
        // it silently corrupts data: the buddy allocator seeds its top bucket
        // with this whole region and reaches the smaller buckets by halving,
        // and halving 2^k-1 truncates to 2^(k-1)-1 all the way down. Every
        // bucket then hands out blocks one byte short of their nominal size,
        // so consecutive blocks overlap by a byte and a request of exactly 2^n
        // gets a 2^n-1 block. The staged copy writes its last byte into the
        // next block, and the next sender's first byte overwrites it -- one
        // wrong trailing byte per message, only for power-of-two payloads,
        // only with more than one send in flight.
        size_t requested_bytes = (size_t)input_comm_buffer_size * 1024 * 1024;
        size_t rounded = 1;
        while (rounded < requested_bytes) rounded <<= 1;
        csv_gpu_manager.comm_buffer_size = rounded;
      }
    }

    // Process device communication buffer parameters (in MB)
    int input_lb_buffer_size = 0;
    if (CmiGetArgIntDesc(argv, "+gpulbbuffer", &input_lb_buffer_size,
          "GPU load balancing buffer size (in MB)")) {
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

    // Process custom size for CUDA IPC event pool
    int input_hapi_ipc_event_pool_size;
    if (!CmiGetArgIntDesc(argv, "+gpuipceventpool", &input_hapi_ipc_event_pool_size,
          "GPU IPC event pool size per PE")) {
      input_hapi_ipc_event_pool_size = 16;
    }

    if (CmiMyRank() == 0) {
      csv_gpu_manager.hapi_ipc_event_pool_size_pe = input_hapi_ipc_event_pool_size;
      csv_gpu_manager.hapi_ipc_event_pool_size_total = input_hapi_ipc_event_pool_size * csv_gpu_manager.pes_per_device;
    }

    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> CUDA IPC event pool size - %d per PE, %d per device\n",
          csv_gpu_manager.hapi_ipc_event_pool_size_pe, csv_gpu_manager.hapi_ipc_event_pool_size_total);
    }

    // Payload size at which a cross-process send switches from staging through
    // the device communication buffer to exporting its source allocation
    // directly. Unset means never, so a run that does not ask for this behaves
    // exactly as it did before. CHARM_GPU_IPC_THRESHOLD overrides
    // +gpuipcthreshold, so a sweep can vary it without rewriting command lines.
    int input_ipc_threshold = 0;
    const bool have_arg = CmiGetArgIntDesc(argv, "+gpuipcthreshold",
        &input_ipc_threshold,
        "device payload size (bytes) at or above which cross-process sends "
        "use direct CUDA IPC instead of staging");
    if (CmiMyRank() == 0) {
      size_t threshold = have_arg && input_ipc_threshold >= 0
                             ? (size_t)input_ipc_threshold
                             : SIZE_MAX;
      const char* env = getenv("CHARM_GPU_IPC_THRESHOLD");
      if (env != NULL) threshold = hapiParseByteSize(env);
      csv_gpu_manager.ipc_direct_threshold = threshold;

      const char* cache_env = getenv("CHARM_GPU_IPC_CACHE");
      csv_gpu_manager.ipc_cache_imports = !(cache_env && atoi(cache_env) == 0);
    }

    CmiNodeBarrier(); // Ensure the threshold is set before any send reads it

    if (CmiMyPe() == 0) {
      if (csv_gpu_manager.ipc_direct_threshold == SIZE_MAX) {
        CmiPrintf("HAPI> Direct CUDA IPC transport: disabled (all "
                  "cross-process sends staged)\n");
      } else {
        CmiPrintf("HAPI> Direct CUDA IPC transport: payloads >= %zu bytes, "
                  "import cache %s\n",
                  csv_gpu_manager.ipc_direct_threshold,
                  csv_gpu_manager.ipc_cache_imports ? "on" : "OFF (measurement "
                  "mode: each receive opens, synchronizes and closes)");
      }
    }
  }

  // Check if P2P access should be enabled
  bool enable_peer = true; // Enabled by default
  if (CmiGetArgFlagDesc(argv, "+gpunopeer",
        "do not enable P2P access between visible GPU pairs")) {
    enable_peer = false;
  }

  // Enable P2P access to every other device on this host, not just the ones
  // this process owns. Inter-process GPU messaging opens an IPC handle to a
  // buffer on a peer process's device, and the resulting pointer is only usable
  // once peer access to that device has been enabled here. Ranging over
  // device_count (a per-process count) instead left peer access off entirely in
  // the common one-GPU-per-process layout, so the first copy touching a peer
  // buffer faulted with an illegal access -- reported at whichever CUDA call
  // checked next, since error 700 is sticky and asynchronous.
  //
  // Note cpv_my_device and the loop index are both global device indices;
  // device_count_on_physical_node counts devices on the host.
  //
  // Should only be done by device representative threads
  if (enable_peer) {
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> Enabling P2P access between devices\n");
    }
    if (cpv_device_rep) {
      for (int i = 0; i < csv_gpu_manager.device_count_on_physical_node; i++) {
        if (i != cpv_my_device) {
          int can_access_peer;

          hapiCheck(hapiDeviceCanAccessPeer(&can_access_peer, cpv_my_device, i));
          if (can_access_peer) {
            // Returns hapiErrorPeerAccessAlreadyEnabled when already on, which
            // is benign -- deliberately not wrapped in hapiCheck.
            hapiDeviceEnablePeerAccess(i, 0);
          } else if (_lb_args.debug() > 0 && CmiMyPe() == 0) {
            CmiPrintf("HAPI> No P2P access from device %d to device %d\n",
                      cpv_my_device, i);
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
void recordEvent(cudaStream_t stream, const CkCallback& cb, void* cb_msg, hapiWorkRequest* wr = NULL) {
  // if(obj!=NULL)
  //   CmiAbort("non null without HAPI CUDA CALLBACK");
  // create CUDA event / get CUDA event from the pool and insert into stream
  hapiEvent_t ev;
  auto& hapi_event_pool_local = CpvAccess(hapi_event_pool);
  if(hapi_event_pool_local.size() == 0) {
  #if CMK_LBDB_ON
    hapiEventCreateWithFlags(&ev, hapiEventDefault);
  #else
    hapiEventCreateWithFlags(&ev, hapiEventDisableTiming);
  #endif
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
// Callback function invoked by the CUDA runtime certain parts of GPU work are
// complete. It sends a converse message to the original PE to free the relevant
// device memory and invoke the user's callback. The reason for this method is
// that a thread created by the CUDA runtime does not have access to any of the
// CpvDeclare'd variables as it is not one of the threads created by the Charm++
// runtime.
static void CUDACallback(void *data) {
#ifdef HAPI_NVTX_PROFILE
  NVTXTracer nvtx_range("CUDACallback", NVTXColor::Silver);
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

  // create converse message to be delivered to this PE after CUDA callback
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

  // add callback into CUDA stream
  hapiCheck(hapiLaunchHostFunc(wr->stream, CUDACallback, (void*)conv_msg));
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

  // Use CUDA per-thread default stream
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

  if (!CmiInCommThread()) ipcHandleCreate(); // Create CUDA IPC handles

  // Ensure CUDA IPC handles are available for all processes
  // Note: Causes a hang when this barrier is placed after CPU topology initialization
  // FIXME: This only needs to be a host-wide synchronization
  CmiBarrier();

  if (CmiMyRank() == 0) {
    if (!CmiInCommThread()) ipcHandleOpen(); // Open CUDA IPC handles for accessing other processes' device memory
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

// Create CUDA IPC handles and populate shared memory region
// Invoked by all PEs
static void ipcHandleCreate() {
  // Only device reps should continue to perform the following operations
  // so that they are done only once per device
  if (!CpvAccess(device_rep)) return;

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  int& cpv_my_device_id = CpvAccess(my_device_id);

  // Create CUDA IPC memory handle in shared memory
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

  // Create CUDA IPC events and store them locally (in hapi_ipc_device_info),
  // and create corresponding IPC handles in shared memory
  hapi_ipc_device_info& my_device_info = csv_gpu_manager.hapi_ipc_device_infos[csv_gpu_manager.device_count * CmiMyNodeRankLocal() + cpv_my_device_id];
  hapi_ipc_event_shared* shm_event_shared = (hapi_ipc_event_shared*)((char*)shm_mem_handle + sizeof(hapiIpcMemHandle_t));

  // Each slot carries a pthread mutex that lives in the shared-memory region
  // and is locked by BOTH the owning process and whichever peer receives from
  // it. A pthread mutex is process-private unless it is explicitly created
  // with PTHREAD_PROCESS_SHARED, and mmap'd memory merely starts zeroed --
  // which is not a valid initialized mutex. Locking it from the peer process
  // was undefined behavior and segfaulted on the first cross-process transfer.
  // Only the slot's owner initializes it, before any peer can reach it: the
  // CmiBarrier between ipcHandleCreate and ipcHandleOpen orders that.
  pthread_mutexattr_t shared_attr;
  pthread_mutexattr_init(&shared_attr);
  pthread_mutexattr_setpshared(&shared_attr, PTHREAD_PROCESS_SHARED);

  for (int i = 0; i < csv_gpu_manager.hapi_ipc_event_pool_size_total; i++) {
    hapi_ipc_event_shared* cur_shm_event_shared = shm_event_shared + i;

    // CHARM_NO_IPC_MUTEX_INIT restores the original (uninitialized) state for
    // bisecting.
    if (getenv("CHARM_NO_IPC_MUTEX_INIT") == nullptr) {
      pthread_mutex_init(&cur_shm_event_shared->lock, &shared_attr);
      cur_shm_event_shared->src_flag = false;
      cur_shm_event_shared->dst_flag = false;
    }

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

  pthread_mutexattr_destroy(&shared_attr);

  // Store device comm buffer ptr in local info (just in case)
  my_device_info.buffer = device_ptr;
}

// Open CUDA IPC handles created by other processes
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

      // Open the peer's event handles under OUR current device. Do NOT switch to
      // the exporter's device around this: that was tried and it breaks all
      // cross-process transfers (git bisect, commit bec910fef). cudaIpcOpen*
      // is defined in terms of the caller's context, and the imported event is
      // usable from here without any device change.
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

/*** Direct CUDA IPC transport: export and import caches ***/

size_t hapiIpcDirectThreshold() {
  return CsvAccess(gpu_manager).ipc_direct_threshold;
}

bool hapiIpcCacheImports() {
  return CsvAccess(gpu_manager).ipc_cache_imports;
}

bool hapiIpcExportBuffer(const void* ptr, hapiIpcMemHandle_t* handle,
                         size_t* offset) {
  // cudaIpcGetMemHandle names an allocation, and cudaIpcOpenMemHandle hands the
  // peer that allocation's base -- so an interior pointer has to be split into
  // (base, offset) here and reassembled on the far side.
  void* base = NULL;
  size_t alloc_size = 0;
  if (!hapiMemGetAddressRange(&base, &alloc_size, ptr)) return false;

  const void* base_ptr = (const void*)base;
  *offset = (size_t)((const char*)ptr - (const char*)base_ptr);

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
#if CMK_SMP
  CmiLock(csv_gpu_manager.ipc_cache_lock);
#endif
  auto it = csv_gpu_manager.ipc_export_cache.find(base_ptr);
  if (it != csv_gpu_manager.ipc_export_cache.end()) {
    *handle = it->second;
#if CMK_SMP
    CmiUnlock(csv_gpu_manager.ipc_cache_lock);
#endif
    return true;
  }

  hapiIpcMemHandle_t new_handle;
  const hapiError_t err = hapiIpcGetMemHandle(&new_handle, (void*)base_ptr);
  if (err != hapiSuccess) {
    // Not an exportable allocation (managed or host-registered memory, for
    // instance). Clear the sticky error and let the caller stage instead.
    cudaGetLastError();
#if CMK_SMP
    CmiUnlock(csv_gpu_manager.ipc_cache_lock);
#endif
    return false;
  }

  csv_gpu_manager.ipc_export_cache.emplace(base_ptr, new_handle);
  *handle = new_handle;
#if CMK_SMP
  CmiUnlock(csv_gpu_manager.ipc_cache_lock);
#endif
  return true;
}

void* hapiIpcImportBuffer(const hapiIpcMemHandle_t& handle) {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
#if CMK_SMP
  CmiLock(csv_gpu_manager.ipc_cache_lock);
#endif
  if (csv_gpu_manager.ipc_cache_imports) {
    auto it = csv_gpu_manager.ipc_import_cache.find(handle);
    if (it != csv_gpu_manager.ipc_import_cache.end()) {
      void* ptr = it->second;
      csv_gpu_manager.ipc_import_hits.fetch_add(1, std::memory_order_relaxed);
#if CMK_SMP
      CmiUnlock(csv_gpu_manager.ipc_cache_lock);
#endif
      return ptr;
    }
  }

  // Open under the caller's current device, as ipcHandleOpen does for the comm
  // buffers. Switching devices around cudaIpcOpen* breaks cross-process
  // transfers (commit bec910fef) -- the call is defined in terms of the
  // caller's context and the mapping is usable from here because P2P access is
  // enabled between every pair of devices on the host.
  void* mapped = NULL;
  const hapiError_t err =
      hapiIpcOpenMemHandle(&mapped, handle, hapiIpcMemLazyEnablePeerAccess);
  if (err != hapiSuccess) {
    cudaGetLastError();
#if CMK_SMP
    CmiUnlock(csv_gpu_manager.ipc_cache_lock);
#endif
    return NULL;
  }

  csv_gpu_manager.ipc_import_misses.fetch_add(1, std::memory_order_relaxed);
  if (csv_gpu_manager.ipc_cache_imports)
    csv_gpu_manager.ipc_import_cache.emplace(handle, mapped);
#if CMK_SMP
  CmiUnlock(csv_gpu_manager.ipc_cache_lock);
#endif
  return mapped;
}

void hapiIpcFlushImportCache() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
#if CMK_SMP
  CmiLock(csv_gpu_manager.ipc_cache_lock);
#endif
  for (auto& entry : csv_gpu_manager.ipc_import_cache) {
    if (hapiIpcCloseMemHandle(entry.second) != hapiSuccess) cudaGetLastError();
  }
  csv_gpu_manager.ipc_import_cache.clear();
  // Exports name this process's own allocations, which migration also frees and
  // reallocates -- and cudaMalloc reuses addresses, so a stale entry would hand
  // out a handle for memory that is no longer there.
  csv_gpu_manager.ipc_export_cache.clear();
#if CMK_SMP
  CmiUnlock(csv_gpu_manager.ipc_cache_lock);
#endif
}

void hapiIpcReportStats() {
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  const long staged = csv_gpu_manager.ipc_staged_sends.load();
  const long direct = csv_gpu_manager.ipc_direct_sends.load();
  const long hits = csv_gpu_manager.ipc_import_hits.load();
  const long misses = csv_gpu_manager.ipc_import_misses.load();
  if (staged + direct + hits + misses == 0) return;
  CmiPrintf("[ipc-stats] pid=%d staged=%ld direct=%ld import_hits=%ld "
            "import_misses=%ld threshold=%zu cache=%d\n",
            (int)getpid(), staged, direct, hits, misses,
            csv_gpu_manager.ipc_direct_threshold,
            (int)csv_gpu_manager.ipc_cache_imports);
}

/*** Migration arena registry (see hapi.h) ***/

namespace {
struct HapiArenaRec {
  size_t extent;
  size_t live;
};
// Ordered by base so an interior pointer resolves with upper_bound. Shared by
// every PE in the process (SMP threads included), hence the mutex.
std::map<char*, HapiArenaRec> hapi_arena_registry;
std::mutex hapi_arena_registry_lock;
}

void hapiArenaRegister(void* base, size_t extent, size_t liveBuffers) {
  std::lock_guard<std::mutex> g(hapi_arena_registry_lock);
  hapi_arena_registry[(char*)base] = HapiArenaRec{extent, liveBuffers};
}

void hapiFreeMigratable(void* ptr) {
  if (ptr == NULL) return;
  void* arena_to_free = NULL;
  {
    std::lock_guard<std::mutex> g(hapi_arena_registry_lock);
    if (!hapi_arena_registry.empty()) {
      auto it = hapi_arena_registry.upper_bound((char*)ptr);
      if (it != hapi_arena_registry.begin()) {
        --it;
        if ((char*)ptr >= it->first &&
            (char*)ptr < it->first + it->second.extent) {
          if (--it->second.live == 0) {
            arena_to_free = it->first;
            hapi_arena_registry.erase(it);
          }
          // Interior pointer handled: either the arena still has live
          // buffers (nothing to free yet) or it is freed below.
          if (arena_to_free == NULL) return;
        }
      }
    }
  }
  if (arena_to_free != NULL) {
    hapiCheck(hapiFree(arena_to_free));
    return;
  }
  // Not arena-interior: an ordinary allocation.
  hapiCheck(hapiFree(ptr));
}

/*** Per-chare device footprint tracking (see hapi_portable.h) ***/
//
// The producer behind the LB memory contract: every hapiMalloc/hapiFree made
// while a chare's entry method is running is attributed to that chare's LB
// identity, giving the balancer the per-object resident footprint its
// final-placement feasibility check needs. Identity comes from the same
// source as CUPTI kernel attribution -- the active location record -- and is
// stable across migration within a process. Allocations made outside entry
// methods (startup, comm buffers, arenas) are deliberately unattributed; the
// consumer floors the footprint at the serialized size, so missing
// attribution errs toward refusing a move, never toward approving one.

#if CMK_LBDB_ON

namespace {
std::unordered_map<LDObjKey, size_t, LDObjKeyHash> gpu_obj_footprint;
std::unordered_map<void*, std::pair<LDObjKey, size_t>> gpu_ptr_owner;
std::mutex gpu_footprint_lock;

bool hapiActiveObjKey(LDObjKey& key) {
  CkLocRec* active = CkActiveLocRec();
  if (active == NULL) return false;
  const LDObjHandle& handle = active->getLdHandle();
  key.omID() = handle.omID();
  key.objID() = handle.objID();
  return true;
}
}

void hapiRecordAlloc(void* ptr, size_t size) {
  if (ptr == NULL) return;
  LDObjKey key{};
  if (!hapiActiveObjKey(key)) return;  // runtime allocation: unattributed
  std::lock_guard<std::mutex> g(gpu_footprint_lock);
  gpu_obj_footprint[key] += size;
  gpu_ptr_owner[ptr] = std::make_pair(key, size);
}

void hapiRecordFree(void* ptr) {
  if (ptr == NULL) return;
  std::lock_guard<std::mutex> g(gpu_footprint_lock);
  auto it = gpu_ptr_owner.find(ptr);
  if (it == gpu_ptr_owner.end()) return;  // was not attributed at allocation
  auto owner = gpu_obj_footprint.find(it->second.first);
  if (owner != gpu_obj_footprint.end()) {
    owner->second -= (owner->second >= it->second.second)
                         ? it->second.second : owner->second;
  }
  gpu_ptr_owner.erase(it);
}

// The running chare's attributed live bytes; called from the AtSync path,
// where the active object is the chare itself.
size_t hapiCurrentObjectFootprint() {
  LDObjKey key{};
  if (!hapiActiveObjKey(key)) return 0;
  std::lock_guard<std::mutex> g(gpu_footprint_lock);
  auto it = gpu_obj_footprint.find(key);
  return (it == gpu_obj_footprint.end()) ? 0 : it->second;
}

#else

void hapiRecordAlloc(void* ptr, size_t size) { (void)ptr; (void)size; }
void hapiRecordFree(void* ptr) { (void)ptr; }
size_t hapiCurrentObjectFootprint() { return 0; }

#endif  // CMK_LBDB_ON

/******************** DEPRECATED ********************/
// Need to be updated with the Tracing API.
static inline void gpuEventStart(hapiWorkRequest* wr, int* index,
                                 WorkRequestStage event, ProfilingStage stage) {
#ifdef HAPI_TRACE
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  gpuEventTimer* shared_gpu_events_ = csv_gpu_manager.gpu_events_;
  int shared_time_idx_ = csv_gpu_manager.time_idx_++;
  shared_gpu_events_[shared_time_idx_].cmi_start_time = CmiWallTimer();
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
  csv_gpu_manager.gpu_events_[index].cmi_end_time = CmiWallTimer();
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
  wr->phase_start_time = CmiWallTimer();
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
    double tt = CmiWallTimer() - (wr->phase_start_time);
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
#if CMK_LBDB_ON
// Lightweight HAPI, to be invoked after data transfer or kernel execution.
#endif

// How many external-correlation IDs this PE has actually pushed and not yet
// popped. Tracing is switched on and off from inside entry methods, so a push
// can be skipped while its matching pop still runs (or the reverse). Pairing
// the pop against this count rather than against the tracing flag keeps
// CUPTI's stack balanced across those transitions; without it the pop reports
// CUPTI_ERROR_QUEUE_EMPTY and attribution drifts. Each Charm++ PE is its own
// thread, so thread_local is per-PE.
static thread_local int cupti_pushed_depth = 0;
static thread_local int cupti_tag_pushed_depth = 0;
// The detach generation this PE last observed; see GPUManager::cupti_generation_.
static thread_local uint64_t cupti_seen_generation = 0;

// This PE's view of the process-wide token table. Every entry method on a
// migratable chare needs its object's token, and the table behind it is shared
// by every PE in the process, so consulting it under the node-wide lock would
// serialize the whole process on one mutex for the length of the run.
//
// The table is append-only for the lifetime of the process: a token is never
// reassigned, reused for a different identity, or dropped -- hapiClearCuptiData
// deliberately keeps it so that later epochs reuse the same token for the same
// LB identity. That invariant is what makes this cache safe without any
// invalidation protocol: an entry, once correct, stays correct, including
// across migration (the destination PE simply misses once and interns the same
// token the source PE already has). Anything that gains the ability to clear or
// renumber GpuObjectTokenTable must also invalidate these caches.
static thread_local std::unordered_map<LDObjKey, uint64_t, LDObjKeyHash>
    cupti_local_object_tokens;

// Drop this PE's outstanding push count if CUPTI has been detached since we
// last looked -- the stack those pushes referred to no longer exists, so
// popping against it would report CUPTI_ERROR_QUEUE_EMPTY.
static inline void hapiCuptiSyncGeneration(GPUManager& gm) {
  if (cupti_seen_generation != gm.cupti_generation_) {
    cupti_seen_generation = gm.cupti_generation_;
    cupti_pushed_depth = 0;
    cupti_tag_pushed_depth = 0;
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

bool hapiCuptiPushKernelTag(uint64_t workTag) {
  GPUManager& gm = CsvAccess(gpu_manager);
  if (!_lb_args.gpuScaling() ||
      !gm.cupti_tracing_active_.load(std::memory_order_relaxed))
    return false;
  hapiCuptiSyncGeneration(gm);

  CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, workTag));
  ++cupti_tag_pushed_depth;
  return true;
}

void hapiCuptiPopKernelTag() {
  GPUManager& gm = CsvAccess(gpu_manager);
  hapiCuptiSyncGeneration(gm);
  if (cupti_tag_pushed_depth == 0 || !gm.cupti_initialized_) return;
  --cupti_tag_pushed_depth;

  uint64_t tag;
  CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &tag));
}

// Lightweight HAPI, to be invoked after data transfer or kernel execution.
void hapiAddCallback(hapiStream_t stream, const CkCallback& cb, void* cb_msg) {
#ifndef HAPI_CUDA_CALLBACK
  // record CUDA event
  recordEvent(stream, cb, cb_msg);
#else
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);

  /* FIXME works for now (faster too), but CmiAlloc might not be thread-safe
#if CMK_SMP
  CmiLock(csv_gpu_manager.queue_lock_);
#endif
*/

  // create converse message to be delivered to this PE after CUDA callback
  hapiCallbackMessage* conv_msg = (hapiCallbackMessage*)CmiAlloc(sizeof(hapiCallbackMessage)); // FIXME memory leak?
  conv_msg->rank = CmiMyRank();
  conv_msg->cb = cb;
  conv_msg->cb_msg = cb_msg;
  CmiSetHandler(conv_msg, csv_gpu_manager.light_cb_idx_);

  // push into CUDA stream
  hapiCheck(hapiLaunchHostFunc(stream, CUDACallback, (void*)conv_msg));

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

void hapiSendMemoryRequest(char* msg, int size)
{
    int cpv_my_device = CpvAccess(my_device);
    
    char server_fifo[BUFFER_SIZE];
    sprintf(server_fifo, SERVER_FIFO_TEMPLATE, cpv_my_device);
    CmiPrintf("Sending request to %s\n", server_fifo);
    
    int server_fd = open(server_fifo, O_WRONLY | O_NONBLOCK);
    if (server_fd == -1) {
        perror("open server FIFO for writing");
        return;
    }

    ssize_t written = write(server_fd, msg, size);
    if (written == -1) {
        perror("write to server FIFO");
    } else {
        //CmiPrintf("Successfully wrote %zd bytes to server FIFO\n", written);
    }
    
    close(server_fd);
}


// hapiError_t hapiMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
//   hapiError_t err;
// #if CMK_LBDB_ON
//   hapiEvent_t start;

//   cudaEventCreate(&start);
//   cudaEventRecord(start, stream);
// #endif

//   err = cudaMemcpyAsync(dst, src, count, kind, stream);
// #if CMK_LBDB_ON
//   hapiRecordTime(stream, start);  
// #endif
//   return err;
// }

// cudaError_t hapiMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0) {
//   cudaError_t err;
// #if CMK_LBDB_ON
//   cudaEvent_t start;

//   cudaEventCreate(&start);
//   cudaEventRecord(start, stream);
// #endif
//   err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
// #if CMK_LBDB_ON
//   hapiRecordTime(stream, start);
// #endif
//   return err;
// }


void hapiErrorDie(cudaError_t retCode, const char* code, const char* file, int line) {
  if (retCode != cudaSuccess) {
    fprintf(stderr, "Fatal CUDA Error [%d] %s at %s:%d\n", retCode, cudaGetErrorString(retCode), file, line);
    CmiAbort("Exit due to CUDA error");
  }
}

uint64_t hapiMyDevice() {
  int physical_node_id = CmiPhysicalNodeID(CmiMyPe());
  int my_device = CpvAccess(my_device);
  return (static_cast<uint64_t>(physical_node_id) << 32) | my_device;
}

int hapiDeviceForPe(int pe) {
  return CpvAccessOther(my_device, CmiRankOf(pe));
}

#ifdef CMK_LBDB_ON
const GpuDeviceDescriptor& hapiMyDeviceDescriptor() {
  static const GpuDeviceDescriptor kNoDevice;
  GPUManager& gm = CsvAccess(gpu_manager);
  int local_id = CpvAccess(my_device_id);
  if (local_id < 0 || local_id >= (int)gm.device_managers.size()) return kNoDevice;
  // Same lock the CUPTI drain uses, because that is the other caller of the
  // lazy discovery this may have to run.
  {
    std::lock_guard<std::mutex> lk(gm.cupti_prepare_lock_);
    hapiPopulateDeviceProps(gm);
  }
  return gm.device_managers[local_id].descriptor;
}
#endif

int hapiMyDeviceTotalSMs() {
#ifdef CMK_LBDB_ON
  GPUManager& gm = CsvAccess(gpu_manager);
  int local_id = CpvAccess(my_device_id);
  if (local_id < 0 || local_id >= (int)gm.device_managers.size()) return 0;
  DeviceManager& dm = gm.device_managers[local_id];
  if (!dm.props_initialized) {
    // Fallback: query directly for just this device if the lazy population in
    // hapiProcessCuptiBuffers hasn't run yet.
    int count = 0;
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount,
                           dm.global_index);
    return count;
  }
  return dm.multi_processor_count;
#else
  return 0;
#endif
}
