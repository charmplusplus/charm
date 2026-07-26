#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdint>
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

// Must be CUDA_API_PER_THREAD_DEFAULT_STREAM (the standard CUDA macro) for
// cuda_runtime.h to redirect cudaXxx() -> cudaXxx_ptsz(). Without per-thread
// default streams, every PE in an SMP process shares the per-context driver
// launch lock; with 8 PEs/process that serializes every CUDA call. A rename
// to hapi_API_PER_THREAD_DEFAULT_STREAM in the reconverse port left this
// silently broken — nsys profile showed cudaEventQuery at 2.5 us median in
// reconverse vs 0.91 us in classic, and the symbol was plain cudaEventRecord
// instead of cudaEventRecord_ptsz.
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "hapi_portable.h"
#include "converse.h"
#include "conv-mach-opt.h" /* for CMK_hapi */
#include "ckrescale.h"
#include "charm++.h"

#include "hapi.h"
#include "hapi_impl.h"
#include "gpumanager.h"
#ifdef HAPI_NVTX_PROFILE
#include "hapi_nvtx.h"
#endif

#if CMK_LBDB_ON
#if CMK_CUDA
#include <cupti.h>
#endif
#include "LBManager.h"

#if CMK_CUDA
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

#define SERVER_FIFO_TEMPLATE "/tmp/server_pipe_%ld"
#define CLIENT_FIFO_TEMPLATE "/tmp/client_pipe_%ld"
#define BUFFER_SIZE 256
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
  hapiEvent_t event; // NULL marks a pinned-flag entry (see flag_seq)
  CkCallback cb;
  void* cb_msg;
  hapiWorkRequest* wr; // if this is not NULL, buffers and request itself are deallocated
  CkMigratable* obj; // pointer to the object whose load we want to set
  hapiEvent_t start_ev; // event to record the start time
  uint32_t flag_seq; // pinned-flag entries: sequence number the slot must reach

  hapiEvent(hapiEvent_t event_, const CkCallback& cb_, void* cb_msg_, hapiWorkRequest* wr_ = NULL, CkMigratable* obj_ = NULL, hapiEvent_t start_ev_ = NULL)
            : event(event_), cb(cb_), cb_msg(cb_msg_), wr(wr_), obj(obj_), start_ev(start_ev_), flag_seq(0) {}
} hapiEvent;

CpvDeclare(std::queue<hapiEvent>, hapi_event_queue);
CpvDeclare(std::queue<hapiEvent_t>, hapi_event_pool);

// Pinned-flag completion detection (+gpuflagpoll). Motivation: on AMD an
// event query against a not-yet-complete event costs ~10 us and is
// serialized process-wide inside the runtime, so a scheduler that polls
// events steals ~10 us from launches/messaging per in-flight chare per
// sweep. Instead, completion is detected by enqueueing a stream-ordered
// 32-bit write of a per-PE sequence number into a pinned slot
// (hip/cuStreamWriteValue32); the scheduler then checks completion with a
// plain load, which takes no lock and costs nanoseconds. Slots live in a
// per-PE ring; a slot is reassigned only after HAPI_FLAG_SLOTS newer
// callbacks, and recordEvent falls back to the event path when that many
// are already in flight, so a pending entry's slot is never overwritten.
// The LB instrumentation path keeps events (it needs event timestamps).
#define HAPI_FLAG_SLOTS 256  // power of two; max in-flight flag entries per PE
#define HAPI_FLAG_STRIDE 16  // uint32s per slot: one cache line, no false sharing
static bool hapi_use_flag_poll = false;
CpvDeclare(uint32_t*, hapi_flag_slots);   // pinned host ring
CpvDeclare(void*, hapi_flag_slots_dev);   // device alias of the ring
CpvDeclare(uint32_t, hapi_flag_seq);      // last assigned sequence number
CpvDeclare(int, hapi_flag_inflight);      // flag entries currently queued
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

#ifdef CMK_LBDB_ON

#if CMK_CUDA
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
#endif

// Initialize CUPTI activity tracing — called once per process
void hapiCuptiInit() {
#if CMK_CUDA
  CmiPrintf("HAPI: Initializing CUPTI...\n");
  hapiDeviceSynchronize(); 
  GPUManager& gm = CsvAccess(gpu_manager);
  if (gm.cupti_initialized_) return;

  CUPTI_SAFE_CALL(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_SAFE_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));

  gm.cupti_initialized_ = true;
#endif
}

void hapiCuptiFinalize() {
  CmiPrintf("HAPI: Finalizing CUPTI...\n");
  hapiDeviceSynchronize(); // Ensure all activity records are flushed
  GPUManager& gm = CsvAccess(gpu_manager);
  if(gm.cupti_initialized_== false) return;
  gm.cupti_initialized_ = false;
#if CMK_CUDA
  CUPTI_SAFE_CALL(cuptiFinalize());
#endif
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
    hapiCheck(hapiSetDevice(cpv_my_device));
#endif

#ifndef HAPI_CUDA_CALLBACK
    // Pre-warm the per-PE event pool so steady-state recordEvent never hits
    // cudaEventCreate on the critical path. Must run after hapiSetDevice so
    // events bind to the right CUDA context. 64 covers the deepest expected
    // in-flight depth for a single PE under task-bench-style workloads.
    {
      auto& pool = CpvAccess(hapi_event_pool);
      for (int i = 0; i < 64; i++) {
        hapiEvent_t ev;
        hapiCheck(hapiEventCreateWithFlags(&ev, hapiEventDisableTiming));
        pool.push(ev);
      }
    }

    // Allocate the per-PE pinned-flag ring (must run after hapiSetDevice so
    // the pinned allocation and its device alias belong to this device).
    if (hapi_use_flag_poll) {
      uint32_t*& slots = CpvAccess(hapi_flag_slots);
      size_t bytes = (size_t)HAPI_FLAG_SLOTS * HAPI_FLAG_STRIDE * sizeof(uint32_t);
      hapiCheck(hapiMallocHost((void**)&slots, bytes));
      memset((void*)slots, 0, bytes);
      hapiCheck(hapiHostGetDevicePointer(&CpvAccess(hapi_flag_slots_dev),
                                         (void*)slots, 0));
    }

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
  hapiCheck(hapiSetDevice(cpv_my_device));

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

  hapiIpcMemHandle_t ipc_handle;
  hapiCheck(hapiIpcGetMemHandle(&ipc_handle, devPtr));

  char msg_buf[BUFFER_SIZE];
  int offset = sprintf(msg_buf, "CKPT:%ld:%d:%d:", pid, CkMyPe(), size);
  memcpy(msg_buf + offset, &ipc_handle, sizeof(hapiIpcMemHandle_t));
  int total_size = offset + sizeof(hapiIpcMemHandle_t);

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
  hapiIpcMemHandle_t ipc_handle;
  read(client_fd, &ipc_handle, sizeof(hapiIpcMemHandle_t));
  close(client_fd);

  void* srcPtr;
  hapiCheck(hapiIpcOpenMemHandle(&srcPtr, ipc_handle, hapiIpcMemLazyEnablePeerAccess));
  hapiCheck(hapiMemcpy(devPtr, srcPtr, size, hapiMemcpyDeviceToDevice));
  hapiCheck(hapiIpcCloseMemHandle(srcPtr));

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

void hapiProcessCuptiBuffers() {
  #if CMK_CUDA
  GPUManager& gm = CsvAccess(gpu_manager);
  
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
    // ckout<<"valid size for the CUPTI buffer: "<<item.validSize<<" bytes"<<endl;
    while (cuptiActivityGetNextRecord(item.buffer, item.validSize, &record) == CUPTI_SUCCESS) {
      ++record_count;
      if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        CUpti_ActivityExternalCorrelation *corr = (CUpti_ActivityExternalCorrelation *)record;
        corr_count++;
        if(gm.cupti_correlation_db_.find(corr->correlationId)!=gm.cupti_correlation_db_.end())
        {
          //out of order block 
          uint64_t curr_kernel_time = gm.cupti_correlation_db_[corr->correlationId];
          gm.cupti_obj_gpu_times_[corr->externalId] += curr_kernel_time;
          gm.cupti_correlation_db_.erase(corr->correlationId); // Remove correlation ID after processing
        }
        else 
        {
          gm.cupti_correlation_db_[corr->correlationId] = corr->externalId;
        }
      }
      else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
               record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        kernel_count++;
        CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
        uint64_t duration_ns = kernel->end - kernel->start;
        // ckout<<"the current kernel's duration is "<<duration_ns<<" ns "<<endl;

        auto it = gm.cupti_correlation_db_.find(kernel->correlationId);
        if (it != gm.cupti_correlation_db_.end()) {
          uint64_t obj_id = it->second;
          gm.cupti_obj_gpu_times_[obj_id] += duration_ns;
          gm.cupti_correlation_db_.erase(it); // Remove correlation ID after processing
        }
        else 
        {
          // CmiPrintf("found an out of order entry\n");
          gm.cupti_correlation_db_[kernel->correlationId] = duration_ns;
        }
      }
    }
    
    // ckout<<"number of CUPTI records in this buffer: "<<record_count<<endl;
    
    free(item.buffer);
  }
  //final state of gm.cupti_correlation_db_ and gm.cupti_obj_gpu_times_ 
  // CmiPrintf("size of correlation DB is: %zu\n", gm.cupti_correlation_db_.size());
  // CmiPrintf("size of obj_gpu_times_ map is: %zu\n", gm.cupti_obj_gpu_times_.size());
  // CmiPrintf("number of kernel records processed: %u\n", kernel_count);
  // CmiPrintf("number of correlation records processed: %u\n", corr_count);
  
  // DEBUG: print CUPTI obj-gpu-time map summary
  // if (!gm.cupti_obj_gpu_times_.empty()) {
    //   CkPrintf("[PE %d] CUPTI: %zu objects with GPU times:\n", CmiMyPe(), gm.cupti_obj_gpu_times_.size());
    //   for (auto& kv : gm.cupti_obj_gpu_times_)
    //     CkPrintf("[PE %d]   objID=%lu  gpu_ns=%lu (%.6f s)\n", CmiMyPe(), kv.first, kv.second, kv.second / 1.0e9);
    // } else {
      //   CkPrintf("[PE %d] CUPTI: no obj GPU times recorded (map empty)\n", CmiMyPe());
      // }
      #endif
    }
    
    
//TODO: safely handle SMP mode
void hapiClearCuptiData() {
  GPUManager& gm = CsvAccess(gpu_manager);

  gm.cupti_obj_gpu_times_.clear();
  gm.cupti_correlation_db_.clear();
}

#endif


// Initialize per-PE variables
static void hapiInitCpv() {
  // HAPI event-related
#ifndef HAPI_CUDA_CALLBACK
  CpvInitialize(std::queue<hapiEvent>, hapi_event_queue);
  CpvInitialize(std::queue<hapiEvent_t>, hapi_event_pool);
  // The event pool is pre-warmed in hapiInit() after hapiSetDevice, since
  // cudaEventCreate binds the event to the calling thread's current device.
  CpvInitialize(uint32_t*, hapi_flag_slots);
  CpvInitialize(void*, hapi_flag_slots_dev);
  CpvInitialize(uint32_t, hapi_flag_seq);
  CpvInitialize(int, hapi_flag_inflight);
  CpvAccess(hapi_flag_slots) = NULL;
  CpvAccess(hapi_flag_slots_dev) = NULL;
  CpvAccess(hapi_flag_seq) = 0;
  CpvAccess(hapi_flag_inflight) = 0;
  // The flag ring itself is allocated in hapiInit() after hapiSetDevice.
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
  if (CpvAccess(hapi_flag_slots)) {
    hapiFreeHost((void*)CpvAccess(hapi_flag_slots));
    CpvAccess(hapi_flag_slots) = NULL;
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
    int input_comm_buffer_size = 0;
    if (CmiGetArgIntDesc(argv, "+gpucommbuffer", &input_comm_buffer_size,
          "GPU communication buffer size (in MB)")) {
      if (CmiMyRank() == 0) {
        // Round up size to the closest power of 2
        size_t comm_buffer_size = (size_t)input_comm_buffer_size * 1024 * 1024;
        int size_log2 = std::ceil(std::log2((double)comm_buffer_size));
        csv_gpu_manager.comm_buffer_size = (size_t)std::pow(2, size_log2);
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

    // Process custom size for hapi IPC event pool
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
      CmiPrintf("HAPI> hapi IPC event pool size - %d per PE, %d per device\n",
          csv_gpu_manager.hapi_ipc_event_pool_size_pe, csv_gpu_manager.hapi_ipc_event_pool_size_total);
    }
  }

#ifndef HAPI_CUDA_CALLBACK
  // Check if user opted in to pinned-flag completion detection
  if (CmiGetArgFlagDesc(argv, "+gpuflagpoll",
        "detect kernel completion via pinned-flag writes instead of event polling")) {
    hapi_use_flag_poll = true;
    if (CmiMyPe() == 0) {
      CmiPrintf("HAPI> Pinned-flag completion detection enabled "
                "(%d slots per PE)\n", HAPI_FLAG_SLOTS);
    }
  }
#endif

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
// Enqueue a stream-ordered write of seq into flag slot idx: executes only
// after all prior work on the stream, so the slot reaching seq means that
// work is complete.
static inline void hapiEnqueueFlagWrite(hapiStream_t stream, uint32_t idx,
                                        uint32_t seq) {
  uint32_t* slot_dev =
      (uint32_t*)CpvAccess(hapi_flag_slots_dev) + (size_t)idx * HAPI_FLAG_STRIDE;
#ifdef CMK_HIP
  hapiCheck(hipStreamWriteValue32(stream, (void*)slot_dev, seq, 0));
#else
  // Runtime API has no stream write; use the driver API (cuda.h is included
  // by hapi_portable.h and the runtime's primary context is current).
  CUresult res =
      cuStreamWriteValue32((CUstream)stream, (CUdeviceptr)slot_dev, seq, 0);
  if (res != CUDA_SUCCESS)
    CmiAbort("HAPI> cuStreamWriteValue32 failed; "
             "+gpuflagpoll is not supported on this system");
#endif
}

void recordEvent(hapiStream_t stream, const CkCallback& cb, void* cb_msg, hapiWorkRequest* wr = NULL, CkMigratable* obj = NULL, hapiEvent_t start_ev = NULL) {
  // Pinned-flag path: no event object, no event query later. Excluded for
  // LB instrumentation entries (they need event timestamps) and when the
  // ring is full (falling back to events preserves correctness; a slot may
  // otherwise be reassigned while still pending).
  if (hapi_use_flag_poll && obj == NULL && start_ev == NULL &&
      CpvAccess(hapi_flag_inflight) < HAPI_FLAG_SLOTS) {
    uint32_t& seq_counter = CpvAccess(hapi_flag_seq);
    if (++seq_counter == 0) ++seq_counter; // 0 means "slot never written"
    uint32_t seq = seq_counter;
    uint32_t idx = seq & (HAPI_FLAG_SLOTS - 1);
    hapiEnqueueFlagWrite(stream, idx, seq);

    hapiEvent hev(NULL, cb, cb_msg, wr);
    hev.flag_seq = seq;
    CpvAccess(hapi_flag_inflight)++;
    CpvAccess(hapi_event_queue).push(hev);
    CpvAccess(n_hapi_events)++;
    return;
  }

  // if(obj!=NULL)
  //   CmiAbort("non null without HAPI hapi CALLBACK");
  // create hapi event / get hapi event from the pool and insert into stream
  hapiEvent_t ev;
  auto& hapi_event_pool_local = CpvAccess(hapi_event_pool);
  if(hapi_event_pool_local.size() == 0) {
    // Always disable timing. The elapsed-time path in hapiPollEvents only
    // fires when hev.obj != NULL (LB instrumentation), and recording a
    // timing-enabled event is measurably heavier on the driver.
    hapiEventCreateWithFlags(&ev, hapiEventDisableTiming);
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
    bool complete;
    if (hev.event == NULL) {
      // Pinned-flag entry: a plain load, no runtime call, no lock.
      volatile uint32_t* slot = CpvAccess(hapi_flag_slots) +
          (size_t)(hev.flag_seq & (HAPI_FLAG_SLOTS - 1)) * HAPI_FLAG_STRIDE;
      complete = (*slot == hev.flag_seq);
    } else {
      complete = (hapiEventQuery(hev.event) == hapiSuccess);
    }
    if (complete) {
      queue.pop(); // TODO: investigate possible race condition with charm4py futures - temporarily resolved by popping here

#if CMK_LBDB_ON
      // hev.obj is only set when the caller passes a CkMigratable* (LB
      // instrumentation overload). Tell the compiler the common case is
      // null so the elapsed-time + destroy path stays off the hot trace.
      if (__builtin_expect(hev.obj != nullptr, 0)) {
        float gpu_time;
        hapiEventElapsedTime(&gpu_time, hev.start_ev, hev.event);
        // hapiEventElapsedTime returns ms, convert to seconds to match wallTime units
        double gpu_time_s = gpu_time / 1000.0;
        hev.obj->setObjGPUTime(gpu_time_s + hev.obj->getObjGPUTime());
        hapiEventDestroy(hev.start_ev);
      } else
#endif
      // invoke Charm++ callback if one was given
      hev.cb.send(hev.cb_msg);

      // clean up hapiWorkRequest
      if (hev.wr) {
        hapiWorkRequestCleanup(hev.wr);
      }
      if (hev.event == NULL) {
        CpvAccess(hapi_flag_inflight)--; // slot may now be reassigned
      } else {
        CpvAccess(hapi_event_pool).push(hev.event);
      }
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
void hapiRecordTime(hapiStream_t stream, hapiEvent_t start) {
  Chare* obj = CkActiveObj();
  if (obj && dynamic_cast<CkMigratable*>(obj)) {

  #ifndef HAPI_CUDA_CALLBACK
  // record hapi event
    recordEvent(stream, CkCallback(), NULL, NULL, dynamic_cast<CkMigratable*>(obj), start);
#else
  #error hapi record time with HAPI_CUDA_CALLBACK not supported
#endif

    // while there is an ongoing workrequest, quiescence should not be detected
    // even if all PEs seem idle
    CmiAssert(hapiQdCreate);
    hapiQdCreate(1);
  }
}
#endif

uint64_t hapiCuptiPushObjCorrelation() {
  // printf("seeing CsvAccess(gpu_manager).cupti_initialized_ as %d\n", CsvAccess(gpu_manager).cupti_initialized_);
  if (!CsvAccess(gpu_manager).cupti_initialized_) return 0;

  // Get the active Charm++ object
  Chare* chare = CkActiveObj();
  if (!chare)
    CmiAbort("hapiCuptiPushObjCorrelation call without active object is not possible");

  CkMigratable* mig = dynamic_cast<CkMigratable*>(chare);
  // printf("mig %p\n", mig);
  if (!mig) return 0;

  // Use the raw element ID as the external correlation ID
  // CmiUInt8 is a 64-bit unique object identifier
  uint64_t obj_id = (uint64_t)mig->ckGetID();
#if CMK_CUDA
  CUPTI_SAFE_CALL(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, obj_id));
#endif
  // printf("pushed corr id\n");

  return obj_id;
}

void hapiCuptiPopObjCorrelation() {
  if (!CsvAccess(gpu_manager).cupti_initialized_) return;

  // printf("popped corr id\n");
  uint64_t tag;
#if CMK_CUDA
  CUPTI_SAFE_CALL(cuptiActivityPopExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &tag));
#endif
}

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

