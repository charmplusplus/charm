#include <stdio.h>
#include <stdlib.h>
#include <climits>
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

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include <cuda.h>

#include "hapi_portable.h"
#include "converse.h"
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
  CkMigratable* obj; // pointer to the object whose load we want to set
  hapiEvent_t start_ev; // event to record the start time

  hapiEvent(hapiEvent_t event_, const CkCallback& cb_, void* cb_msg_, hapiWorkRequest* wr_ = NULL, CkMigratable* obj_ = NULL, hapiEvent_t start_ev_ = NULL)
            : event(event_), cb(cb_), cb_msg(cb_msg_), wr(wr_), obj(obj_), start_ev(start_ev_) {}
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

#ifdef CMK_LBDB_ON
static void CUPTIAPI cuptiBufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  *size = 5*1024 * 1024;  // 5MB per buffer
  *buffer = (uint8_t *)malloc(*size);
  *maxNumRecords = 0;
}

//TODO: handle SMP mode
static void CUPTIAPI cuptiBufferCompleted(CUcontext ctx, uint32_t streamId,
                                          uint8_t *buffer, size_t size, size_t validSize) {
  GPUManager& gm = CsvAccess(gpu_manager);

  gm.cupti_buffer_queue_.push({buffer, validSize});
}

// Populate DeviceManager with device attributes needed to compute per-kernel
// SM usage from CUPTI records. Queried once per local device.
static void hapiPopulateDeviceProps(GPUManager& gm) {
  for (DeviceManager& dm : gm.device_managers) {
    if (dm.props_initialized) continue;
    int dev = dm.global_index;
    cudaDeviceGetAttribute(&dm.multi_processor_count,
                           cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&dm.max_threads_per_sm,
                           cudaDevAttrMaxThreadsPerMultiProcessor, dev);
#ifdef cudaDevAttrMaxBlocksPerMultiprocessor
    cudaDeviceGetAttribute(&dm.max_blocks_per_sm,
                           cudaDevAttrMaxBlocksPerMultiprocessor, dev);
#else
    dm.max_blocks_per_sm = 0;
#endif
    cudaDeviceGetAttribute(&dm.max_registers_per_sm,
                           cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&dm.max_shared_mem_per_sm,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&dm.warp_size, cudaDevAttrWarpSize, dev);
    dm.props_initialized = true;
    CmiPrintf("HAPI: device %d props: %d SMs, %d warp_size, %d max_threads/SM, "
              "%d max_blocks/SM, %d max_regs/SM, %d max_smem/SM\n",
              dev, dm.multi_processor_count, dm.warp_size,
              dm.max_threads_per_sm, dm.max_blocks_per_sm,
              dm.max_registers_per_sm, dm.max_shared_mem_per_sm);
  }
}

// Initialize CUPTI activity tracing — called once per process
void hapiCuptiInit() {
  CmiPrintf("HAPI: Initializing CUPTI...\n");
  cudaDeviceSynchronize();
  GPUManager& gm = CsvAccess(gpu_manager);
  if (gm.cupti_initialized_) return;

  CUPTI_SAFE_CALL(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));

  // Note: device_managers is populated later in hapiMapping, so device
  // properties are queried lazily on first use in hapiProcessCuptiBuffers.

  gm.cupti_initialized_ = true;
}

void hapiCuptiFinalize() {
  CmiPrintf("HAPI: Finalizing CUPTI...\n");
  cudaDeviceSynchronize(); // Ensure all activity records are flushed
  GPUManager& gm = CsvAccess(gpu_manager);
  if(gm.cupti_initialized_== false) return;
  gm.cupti_initialized_ = false;

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
    shmCleanup();

    hapiExitCsv();
  }
}

// Initialize per-process variables
static void hapiInitCsv(char** argv) {
  // Create and initialize GPU Manager object
  CsvInitialize(GPUManager, gpu_manager);
  CsvAccess(gpu_manager).init();
  #if CMK_LBDB_ON
    CmiPrintf("HAPI: seeing _lb_args.statsOn() = %d\n", _lb_args.statsOn());
    if (LBHasBalancersRegistered() && _lb_args.statsOn())
      hapiCuptiInit();
  #endif
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

  int warp_size = dm.warp_size > 0 ? dm.warp_size : 32;
  int warps_per_block = (int)((threads_per_block + warp_size - 1) / warp_size);

  // Warp-count limit: maxThreadsPerSM / threadsPerBlock (rounded down).
  int limit_warps =
      dm.max_threads_per_sm > 0 && threads_per_block > 0
          ? (int)(dm.max_threads_per_sm / threads_per_block)
          : INT_MAX;
  if (limit_warps <= 0) limit_warps = 1;

  // Block-count limit (CUDA 11+; 0 means not available → use a large value).
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

  uint32_t kernel_count = 0;
  uint32_t corr_count = 0;
  while (true) {
    uint32_t record_count = 0;
    CuptiBufferItem item;

    // Pop one buffer from the queue
    if (gm.cupti_buffer_queue_.empty()) {
      break;
    }
    item = gm.cupti_buffer_queue_.front();
    gm.cupti_buffer_queue_.pop();

    // Parse records in this buffer
    CUpti_Activity *record = NULL;
    while (cuptiActivityGetNextRecord(item.buffer, item.validSize, &record) == CUPTI_SUCCESS) {
      ++record_count;
      if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        CUpti_ActivityExternalCorrelation *corr = (CUpti_ActivityExternalCorrelation *)record;
        corr_count++;
        // Record the object ID that maps to this kernel's correlationId.
        // If the kernel record already arrived out-of-order, it parked a
        // pending-kernel sentinel under this ID; clear it (we cannot
        // retroactively attribute that kernel — drop it to keep the flow
        // simple).
        gm.cupti_correlation_db_[corr->correlationId] = corr->externalId;
      }
      else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
               record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        kernel_count++;
        CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;

        auto it = gm.cupti_correlation_db_.find(kernel->correlationId);
        if (it == gm.cupti_correlation_db_.end()) {
          // External correlation record hasn't arrived yet — skip.
          // (In practice external correlation records precede kernels.)
          continue;
        }
        uint64_t obj_id = it->second;
        gm.cupti_correlation_db_.erase(it);

        DeviceManager* dm = findDeviceManager(gm, kernel->deviceId);
        int sms_used = dm ? computeKernelSMs(*dm, kernel) : 1;

        LBKernelRecord rec;
        rec.start_ns = kernel->start;
        rec.end_ns   = kernel->end;
        rec.sms_used = sms_used;
        gm.cupti_obj_kernel_records_[obj_id].push_back(rec);
      }
    }

    free(item.buffer);
  }
  CmiPrintf("size of correlation DB is: %zu\n", gm.cupti_correlation_db_.size());
  CmiPrintf("size of obj_kernel_records_ map is: %zu\n", gm.cupti_obj_kernel_records_.size());
  CmiPrintf("number of kernel records processed: %u\n", kernel_count);
  CmiPrintf("number of correlation records processed: %u\n", corr_count);

  if (!gm.cupti_obj_kernel_records_.empty()) {
    CkPrintf("[PE %d] CUPTI: %zu objects with kernel records:\n",
             CmiMyPe(), gm.cupti_obj_kernel_records_.size());
    for (auto& kv : gm.cupti_obj_kernel_records_) {
      uint64_t total_ns = 0;
      for (auto& r : kv.second) total_ns += (r.end_ns - r.start_ns);
      CkPrintf("[PE %d]   objID=%lu  kernels=%zu  total_ns=%lu (%.6f s)\n",
               CmiMyPe(), kv.first, kv.second.size(), total_ns, total_ns / 1.0e9);
    }
  } else {
    CkPrintf("[PE %d] CUPTI: no obj kernel records recorded (map empty)\n", CmiMyPe());
  }
}

//TODO: safely handle SMP mode
void hapiClearCuptiData() {
  GPUManager& gm = CsvAccess(gpu_manager);

  gm.cupti_obj_kernel_records_.clear();
  gm.cupti_correlation_db_.clear();
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
    int input_comm_buffer_size;
    if (CmiGetArgIntDesc(argv, "+gpucommbuffer", &input_comm_buffer_size,
          "GPU communication buffer size (in MB)")) {
      if (CmiMyRank() == 0) {
        // Round up size to the closest power of 2
        size_t comm_buffer_size = (size_t)input_comm_buffer_size * 1024 * 1024;
        int size_log2 = std::ceil(std::log2((double)comm_buffer_size));
        csv_gpu_manager.comm_buffer_size = (size_t)std::pow(2, size_log2);
      }
    }

    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> GPU communication buffer size: %zu MB "
          "(rounded up to the nearest power of two)\n",
          csv_gpu_manager.comm_buffer_size / (1024 * 1024));
    }

    CmiNodeBarrier(); // Ensure device communication buffer size is set

    // Create device communication buffers
    // Should only be done by device representative threads
    if (cpv_device_rep) {
      DeviceManager* dm = csv_gpu_manager.device_map[CmiMyPe()];
#if CMK_SMP
      CmiLock(dm->lock);
#endif
      dm->create_comm_buffer(csv_gpu_manager.comm_buffer_size);
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
void recordEvent(cudaStream_t stream, const CkCallback& cb, void* cb_msg, hapiWorkRequest* wr = NULL, CkMigratable* obj = NULL, cudaEvent_t start_ev = NULL) {
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

  hapiEvent hev(ev, cb, cb_msg, wr, obj, start_ev);

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

#if CMK_LBDB_ON
      if (hev.obj) {
        // CmiPrintf("should not be printed w/o hapi cuda callback \n");
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, hev.start_ev, hev.event);
        // cudaEventElapsedTime returns ms, convert to seconds to match wallTime units
        double gpu_time_s = gpu_time / 1000.0;
        hev.obj->setObjGPUTime(gpu_time_s + hev.obj->getObjGPUTime());
        cudaEventDestroy(hev.start_ev);
      } else 
#endif        
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
void hapiRecordTime(cudaStream_t stream, cudaEvent_t start) {
  Chare* obj = CkActiveObj();
  if (obj && dynamic_cast<CkMigratable*>(obj)) {

  #ifndef HAPI_CUDA_CALLBACK
  // record CUDA event
    recordEvent(stream, NULL, NULL, NULL, dynamic_cast<CkMigratable*>(obj), start);
#else
  #error hapi record time with hapi_cuda_callback not supported
#endif

    // while there is an ongoing workrequest, quiescence should not be detected
    // even if all PEs seem idle
    CmiAssert(hapiQdCreate);
    hapiQdCreate(1);
  }
}
#endif

uint64_t hapiCuptiPushObjCorrelation() {
  if (!CsvAccess(gpu_manager).cupti_initialized_) return 0;

  // Always push (possibly with id=0 when there's no active migratable chare)
  // so the paired pop in _ckStopTiming always has a match. The chare may not
  // be ckInitialized yet at push time but become initialized before the
  // matching pop, which would otherwise underflow CUPTI's stack.
  uint64_t obj_id = 0;
  Chare* chare = CkActiveObj();
  if (chare) {
    if (CkMigratable* mig = dynamic_cast<CkMigratable*>(chare))
      obj_id = (uint64_t)mig->ckGetID();
  }

  CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, obj_id));

  return obj_id;
}

void hapiCuptiPopObjCorrelation() {
  if (!CsvAccess(gpu_manager).cupti_initialized_) return;

  uint64_t tag;
  CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &tag));
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

int hapiMyDeviceTotalSMs() {
#ifdef CMK_LBDB_ON
  GPUManager& gm = CsvAccess(gpu_manager);
  int local_id = CpvAccess(my_device_id);
  if (local_id < 0 || local_id >= (int)gm.device_managers.size()) return 0;
  DeviceManager& dm = gm.device_managers[local_id];
  if (!dm.props_initialized) {
    // Fallback: query directly for just this device if CUPTI init hasn't
    // populated props yet.
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

