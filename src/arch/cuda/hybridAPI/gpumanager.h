#ifndef __GPUMANAGER_H_
#define __GPUMANAGER_H_

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "hapi_portable.h"
#include "converse.h"
#include "hapi.h"
#include "hapi_impl.h"
#include "devicemanager.h"

#include <cupti.h>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>

#if CMK_LBDB_ON
#include "lbdb.h"  // for LBKernelRecord
#endif

// Initial size of the user-addressed portion of host/device buffer arrays;
// the system-addressed portion of host/device buffer arrays (used when there
// is no need to share buffers between work requests) will be equivalant in size.
// FIXME hard-coded maximum
#if CMK_SMP
#define NUM_BUFFERS 4096
#else
#define NUM_BUFFERS 256
#endif

// CUDA IPC Event related struct, stored in host-wide shared memory.
// One object is used for each interaction/message between sender and receiver.
// The number of these objects per device will be equal to the CUDA IPC event pool size.
struct hapi_ipc_event_shared {
  hapiIpcEventHandle_t src_event_handle;
  hapiIpcEventHandle_t dst_event_handle;
  bool src_flag; // Unused for now
  // Set by the receiving process once it has recorded dst_event; read and
  // cleared by the owning sender when it reclaims the slot.
  //
  // A lock-free atomic rather than the process-shared pthread mutex this used
  // to carry. The handshake is a single bit with no invariant spanning any
  // other field, so acquire/release on it gives exactly the ordering that
  // matters -- the receiver's hapiEventRecord happens-before the sender's
  // hapiEventQuery -- at a fraction of the cost. That cost is on the hot path:
  // the sender's reclaim scan touches every slot in its slice on the send
  // path, so a lock/unlock pair per slot was paid per message.
  //
  // This object lives in POSIX shared memory and is written by two processes,
  // so it must be address-free; the assertion below is what guarantees the
  // implementation does not hide a lock inside it (which would sit in one
  // process's private memory and silently fail to synchronize the other).
  std::atomic<bool> dst_flag;
};
static_assert(ATOMIC_BOOL_LOCK_FREE == 2,
              "hapi_ipc_event_shared::dst_flag lives in shared memory and is "
              "accessed from two processes, so it must be lock-free");

#if CMK_LBDB_ON
struct CuptiBufferItem {
  uint8_t* buffer;
  size_t validSize;
};
#endif

// Per-device struct containing data for CUDA IPC.
// Use SMP lock in DeviceManager if needed.
struct hapi_ipc_device_info {
  std::vector<hapiEvent_t> src_event_pool;
  std::vector<hapiEvent_t> dst_event_pool;
  // Flag per event pair (0: free, 1: used with a comm buffer block to release,
  // 2: used by a DIRECT transfer, which holds no block)
  std::vector<int> event_pool_flags;
  // Offset in device comm buffer (per event)
  std::vector<size_t> event_pool_buff_offsets;
  void* buffer;
};

// Hash and equality over the raw bytes of a CUDA IPC memory handle, so
// imported peer allocations can be looked up by the handle that named them.
// The handle is an opaque fixed-size blob with no accessors, so byte identity
// is all there is to go on -- which is exactly right, since the driver hands
// out the same bytes for the same allocation.
struct hapiIpcMemHandleHash {
  size_t operator()(const hapiIpcMemHandle_t& h) const {
    const unsigned char* bytes = (const unsigned char*)&h;
    size_t hash = 1469598103934665603ULL;  // FNV-1a
    for (size_t i = 0; i < sizeof(h); i++) {
      hash ^= bytes[i];
      hash *= 1099511628211ULL;
    }
    return hash;
  }
};

struct hapiIpcMemHandleEq {
  bool operator()(const hapiIpcMemHandle_t& a, const hapiIpcMemHandle_t& b) const {
    return memcmp(&a, &b, sizeof(a)) == 0;
  }
};

#ifdef HAPI_TRACE
#define QUEUE_SIZE_INIT 128
extern "C" int traceRegisterUserEvent(const char* x, int e);
extern "C" void traceUserBracketEvent(int e, double beginT, double endT);

typedef struct gpuEventTimer {
  int stage;
  double cmi_start_time;
  double cmi_end_time;
  int event_type;
  const char* trace_name;
} gpuEventTimer;
#endif

// Event stages used for profiling.
enum WorkRequestStage{
  DataSetup        = 1,
  KernelExecution  = 2,
  DataCleanup      = 3
};

enum ProfilingStage{
  GpuMemSetup   = 8800,
  GpuKernelExec = 8801,
  GpuMemCleanup = 8802
};

// Contains per-process data and methods needed by HAPI.
#ifdef CMK_LBDB_ON
// Hardware identity of the GPU this PE is bound to. Forces the lazy property
// discovery if it has not run yet, so this is safe to call before the first
// CUPTI drain. Returns a zeroed descriptor when this PE has no device.
const GpuDeviceDescriptor& hapiMyDeviceDescriptor();
#endif

struct GPUManager {
  std::vector<BufferPool> mempool_free_bufs_;
  std::vector<size_t> mempool_boundaries_;
  bool mempool_initialized_;

  // The runtime system keeps track of all allocated buffers on the GPU.
  // The following arrays contain pointers to host (CPU) data and the
  // corresponding data on the device (GPU).
  void **host_buffers_;
  void **device_buffers_;

  // Used to assign buffer IDs automatically by the system if the user
  // specifies an invalid buffer ID.
  int next_buffer_;

  hapiStream_t *streams_;
  int n_streams_;
  int last_stream_id_;

#ifdef HAPI_CUDA_CALLBACK
  int host_to_device_cb_idx_;
  int kernel_cb_idx_;
  int device_to_host_cb_idx_;
  int light_cb_idx_; // for lightweight version
#endif

  int running_kernel_idx_;
  int data_setup_idx_;
  int data_cleanup_idx_;

#ifdef HAPI_TRACE
  gpuEventTimer gpu_events_[QUEUE_SIZE_INIT * 3];
  std::atomic<int> time_idx_;
#endif

#ifdef HAPI_INSTRUMENT_WRS
  std::vector<std::vector<std::vector<hapiRequestTimeInfo>>> avg_times_;
  bool init_instr_;
#endif

#if CMK_SMP
  CmiNodeLock queue_lock_;
  CmiNodeLock progress_lock_;
  CmiNodeLock stream_lock_;
  CmiNodeLock mempool_lock_;
  CmiNodeLock inst_lock_;
  CmiNodeLock device_mapping_lock;
#endif

  int device_count; // GPU devices usable by this process (could be less than the number of visible devices)
  int device_count_on_physical_node;
  int pes_per_device;
  std::vector<DeviceManager> device_managers;
  std::unordered_map<int, DeviceManager*> device_map;
  int comm_thread_device;

  // Device communication buffer
  size_t comm_buffer_size;

  // Device load-balancing buffer
  size_t lb_buffer_size;

  // POSIX shared memory for sharing CUDA IPC handles between processes on the same host
  bool use_shm;
  bool test_field;
  void* shm_ptr;
  std::string shm_name;
  int shm_file;
  size_t shm_chunk_size;
  size_t shm_size;
  void* shm_my_ptr;

  // CUDA IPC event pool
  int hapi_ipc_event_pool_size_pe;
  int hapi_ipc_event_pool_size_total;

  // CUDA IPC handles opened for processes on the same node
  // Vector size is equal to the number of devices on the physical node
  std::vector<hapi_ipc_device_info> hapi_ipc_device_infos;

  // Direct CUDA IPC transport (see CmiIpcProtocol). Payloads at or above the
  // threshold skip the device communication buffer and are read out of the
  // sender's allocation directly; SIZE_MAX (the default) means never.
  size_t ipc_direct_threshold;

  // Whether imported peer mappings are kept. Keeping them is what makes the
  // direct transport worth using at all -- cudaIpcOpenMemHandle costs hundreds
  // of microseconds, which no realistic payload amortizes -- and it is also
  // required for correctness, since the driver refuses to open a handle a
  // second time in a process that has not closed it. Turning it off
  // (CHARM_GPU_IPC_CACHE=0) prices the handle operations for a sweep and is a
  // measurement aid only, not a transport option.
  bool ipc_cache_imports;

  // Peer allocation base, keyed by the handle that exported it. Process-wide,
  // like the comm buffer mappings ipcHandleOpen creates, and used from every PE
  // regardless of which device it drives -- P2P access is enabled between all
  // devices on the host.
  std::unordered_map<hapiIpcMemHandle_t, void*, hapiIpcMemHandleHash,
                     hapiIpcMemHandleEq> ipc_import_cache;

  // Allocation base -> handle exporting it, so a repeated send from the same
  // application buffer does not repeat cuMemGetAddressRange/IpcGetMemHandle.
  std::unordered_map<const void*, hapiIpcMemHandle_t> ipc_export_cache;

#if CMK_SMP
  CmiNodeLock ipc_cache_lock;
#endif

  std::atomic<long> ipc_import_hits;
  std::atomic<long> ipc_import_misses;
  std::atomic<long> ipc_staged_sends;
  std::atomic<long> ipc_direct_sends;

  //CUPTI load balancing
#ifdef CMK_LBDB_ON
  // Runtime correlation ID -> process-local full-object token. CUSTOM0 work
  // tags use a separate namespace and must never overwrite object ownership.
  std::unordered_map<uint32_t, uint64_t> cupti_object_correlation_db_;
  std::unordered_map<uint32_t, uint64_t> cupti_work_tag_correlation_db_;

  GpuObjectTokenTable cupti_object_tokens_;
  std::mutex cupti_object_token_lock_;

  // Full LB object identity -> attributed kernel records.
  std::unordered_map<LDObjKey, std::vector<LBKernelRecord>, LDObjKeyHash>
      cupti_obj_kernel_records_;

  // Kernels that could not be attributed to any object (launched outside a
  // migratable entry method, or with no correlation record). They occupy SMs
  // and so take part in the sweep-line as contention, but receive no load.
  // Kept separate rather than under a sentinel object ID, because 0 is a
  // perfectly valid chare element ID.
  std::vector<LBKernelRecord> cupti_unattributed_kernels_;

  // Full object identity -> SM-utilization-normalized GPU load in seconds.
  std::unordered_map<LDObjKey, double, LDObjKeyHash> cupti_obj_norm_load_;

  // Full object identity -> this epoch's per-kernel summary. Built alongside
  // cupti_obj_norm_load_ and read by every PE in the process, so like that map
  // it is const once hapiPrepareCuptiLoads has run.
  std::unordered_map<LDObjKey, GpuObjectEpochCosts, LDObjKeyHash>
      cupti_obj_epoch_costs_;

  // Kernel classes that have been observed carrying an explicit work tag. An
  // untagged instance of such a class means the tag went missing, which is not
  // the same thing as a kernel that is legitimately untagged.
  std::unordered_set<uint64_t> cupti_tagged_kernel_classes_;

  // Kernel classes that have completed a full drain with no tag, and so are
  // known to be legitimately untagged. Their later instances need no second
  // look and can be filed as soon as the object correlation is known.
  std::unordered_set<uint64_t> cupti_untagged_kernel_classes_;

  // Previous round's count of kernels whose correlations had not been parsed
  // yet, used to size the parked vector.
  uint32_t cupti_pending_hint_ = 0;

  // Diagnostic name table for detecting deterministic kernel-hash collisions.
  // Names are copied before CUPTI releases its activity buffer.
  std::unordered_map<uint64_t, std::string> cupti_kernel_names_;
  std::unordered_set<uint64_t> cupti_logged_device_types_;

  // Written by CUPTI's buffer-completed callback, which may run on a
  // CUPTI-owned thread. That thread exists in non-SMP builds too, where the
  // Converse locks compile out, so this needs a real mutex rather than a
  // CmiNodeLock.
  std::queue<CuptiBufferItem> cupti_buffer_queue_;
  std::mutex cupti_queue_lock_;

  // PE-to-device mapping scheme, kept so later code (IPC handle exchange) can
  // recompute which global device another process's slot lives on.
  Mapping map_type = Mapping::RoundRobin;

  bool cupti_initialized_;
  // Whether activity tracing is currently running. Separate from
  // cupti_initialized_: the buffer callbacks are registered once, but tracing
  // itself is switched on and off as the application asks for it.
  //
  // Atomic because the entry-method hooks read it on every invocation from
  // every PE thread while another thread may be switching tracing on or off.
  std::atomic<bool> cupti_tracing_active_{false};
  // Serializes hapiCuptiStartTracing/hapiCuptiStopTracing. This state lives in
  // the node-wide GPUManager, but the switch is reached per-PE through
  // LBDatabase::TurnStatsOn/Off, so every PE thread calls in. Without this,
  // several threads enable or disable the same CUPTI activity kinds and flush
  // concurrently, which corrupts CUPTI's internal buffer bookkeeping and shows
  // up later as heap corruption in an unrelated allocation. Must NOT be the
  // same mutex as cupti_queue_lock_: the flush in the stop path invokes the
  // buffer-completed callback, which takes that one.
  std::mutex cupti_tracing_lock_;
  // Bumped every time CUPTI is detached. Detaching clears CUPTI's
  // external-correlation stack for every PE, but the counters that keep the
  // entry-method push/pop hooks paired are per-PE, and only the PE that ran the
  // detach could reset its own. Each PE compares this against the generation it
  // last saw and zeroes its counter when they differ.
  uint64_t cupti_generation_ = 0;
  // Serializes turning this round's raw CUPTI records into cupti_obj_norm_load_,
  // and makes that work happen exactly once per LB round no matter how many PE
  // threads ask for it. The load balancers used to do this with "rank 0 does the
  // work between two CmiNodeBarrier calls", but both barriers sat behind
  // #if CMK_SMP, which is 0 in the multicore build even though a process really
  // does run many PE threads -- so the barriers vanished and the other ranks read
  // cupti_obj_norm_load_ while rank 0 was rebuilding it. A lock also cannot
  // deadlock the way a spin barrier can: a PE waiting here waits on a PE that is
  // running, not on one that has yet to arrive.
  std::mutex cupti_prepare_lock_;
  // Set once this round's loads are built; cleared by hapiClearCuptiData.
  bool cupti_loads_ready_ = false;
#endif

  void init() {
    next_buffer_ = NUM_BUFFERS;
    streams_ = NULL;
    n_streams_ = 0;
    last_stream_id_ = -1;
    running_kernel_idx_ = 0;
    data_setup_idx_ = 0;
    data_cleanup_idx_ = 0;

#ifdef CMK_LBDB_ON
    cupti_initialized_ = false;
    cupti_tracing_active_.store(false, std::memory_order_relaxed);
    cupti_generation_ = 0;
    cupti_loads_ready_ = false;
#endif

#if CMK_SMP
    // Create mutex locks
    queue_lock_ = CmiCreateLock();
    progress_lock_ = CmiCreateLock();
    stream_lock_ = CmiCreateLock();
    mempool_lock_ = CmiCreateLock();
    inst_lock_ = CmiCreateLock();
    device_mapping_lock = CmiCreateLock();
#endif

#ifdef HAPI_TRACE
    time_idx_ = 0;
#endif

    // Number of PEs mapped to each device
    pes_per_device = -1;

    // Device communication buffer
    comm_buffer_size = 1 << 26; // 64MB by default

    // Device load-balancing buffer. Zero unless +gpulbbuffer asks for one:
    // create_comm_buffer allocates comm_buffer_size + lb_buffer_size on the
    // device, so leaving this uninitialized made the size of that allocation
    // whatever happened to be on the stack. Device migration needs the flag;
    // everything else now gets a defined size without it.
    lb_buffer_size = 0;

    // Shared memory region for CUDA IPC
    use_shm = false;
    shm_ptr = NULL;
    shm_file = -1;
    shm_chunk_size = 0;
    shm_size = 0;
    shm_my_ptr = NULL;

    // Number of CUDA IPC events per PE
    hapi_ipc_event_pool_size_pe = -1;
    hapi_ipc_event_pool_size_total = -1;

    // Direct CUDA IPC transport: off until a threshold is asked for, so an
    // unconfigured run behaves exactly as it did before.
    ipc_direct_threshold = SIZE_MAX;
    ipc_cache_imports = true;
#if CMK_SMP
    ipc_cache_lock = CmiCreateLock();
#endif
    ipc_import_hits = 0;
    ipc_import_misses = 0;
    ipc_staged_sends = 0;
    ipc_direct_sends = 0;

    // Allocate host/device buffers array (both user and system-addressed)
    host_buffers_ = new void*[NUM_BUFFERS*2];
    device_buffers_ = new void*[NUM_BUFFERS*2];

    // Initialize device array to NULL
    for (int i = 0; i < NUM_BUFFERS*2; i++) {
      device_buffers_[i] = NULL;
    }

#ifdef HAPI_TRACE
    traceRegisterUserEvent("GPU Memory Setup", GpuMemSetup);
    traceRegisterUserEvent("GPU Kernel Execution", GpuKernelExec);
    traceRegisterUserEvent("GPU Memory Cleanup", GpuMemCleanup);
#endif

    // set up mempool metadata
    mempool_initialized_ = false;
    mempool_boundaries_.resize(HAPI_MEMPOOL_NUM_SLOTS);

    size_t buf_size = HAPI_MEMPOOL_MIN_BUFFER_SIZE;
    for(int i = 0; i < HAPI_MEMPOOL_NUM_SLOTS; i++){
      mempool_boundaries_[i] = buf_size;
      buf_size = buf_size << 1;
    }

#ifdef HAPI_INSTRUMENT_WRS
    init_instr_ = false;
#endif
  }

  void destroy() {
#if CMK_SMP
    // Destroy mutex locks
    CmiDestroyLock(queue_lock_);
    CmiDestroyLock(progress_lock_);
    CmiDestroyLock(stream_lock_);
    CmiDestroyLock(mempool_lock_);
    CmiDestroyLock(inst_lock_);
#endif

    // Delete data structures
    delete[] host_buffers_;
    delete[] device_buffers_;

    // Destroy device managers
    for (DeviceManager& dm : device_managers) {
      dm.destroy();
    }
    device_managers.clear();

    // Destroy streams
    if (streams_) {
      for (int i = 0; i < n_streams_; i++) {
        hapiCheck(hapiStreamDestroy(streams_[i]));
      }
    }

#ifdef HAPI_TRACE
    // Print traced GPU events
    for (int i = 0; i < time_idx_; i++) {
      switch (gpu_events_[i].event_type) {
        case DataSetup:
          CmiPrintf("[HAPI] kernel %s data setup\n", gpu_events_[i].trace_name);
          break;
        case DataCleanup:
          CmiPrintf("[HAPI] kernel %s data cleanup\n", gpu_events_[i].trace_name);
          break;
        case KernelExecution:
          CmiPrintf("[HAPI] kernel %s execution\n", gpu_events_[i].trace_name);
          break;
        default:
          CmiPrintf("[HAPI] invalid timer identifier\n");
      }
      CmiPrintf("[HAPI] %.2f:%.2f\n",
          gpu_events_[i].cmi_start_time - gpu_events_[0].cmi_start_time,
          gpu_events_[i].cmi_end_time - gpu_events_[0].cmi_start_time);
    }
#endif
  }

  // Creates streams equal to the maximum number of concurrent kernels,
  // which depends on the compute capability of the device.
  // Returns the number of created streams.
  int createStreams() {
    int device;
    hapiDeviceProp device_prop;
    hapiCheck(hapiGetDevice(&device));
    hapiCheck(hapiGetDeviceProperties(&device_prop, device));

    int new_n_streams = 0;

    if (device_prop.major == 3) {
      if (device_prop.minor == 0)
        new_n_streams = 16;
      else if (device_prop.minor == 2)
        new_n_streams = 4;
      else // 3.5, 3.7 or unknown 3.x
        new_n_streams = 32;
    }
    else if (device_prop.major == 5) {
      if (device_prop.minor == 3)
        new_n_streams = 16;
      else // 5.0, 5.2 or unknown 5.x
        new_n_streams = 32;
    }
    else if (device_prop.major == 6) {
      if (device_prop.minor == 1)
        new_n_streams = 32;
      else if (device_prop.minor == 2)
        new_n_streams = 16;
      else // 6.0 or unknown 6.x
        new_n_streams = 128;
    }
    else // unknown (future) compute capability
      new_n_streams = 128;
#if !CMK_SMP
    // Allocate total physical streams between GPU managers sharing a device...
    // i.e. PEs / num devices
    int device_count;
    hapiCheck(hapiGetDeviceCount(&device_count));
    int pes_per_device = CmiNumPesOnPhysicalNode(0) / device_count;
    pes_per_device = pes_per_device > 0 ? pes_per_device : 1;
    new_n_streams =  (new_n_streams + pes_per_device - 1) / pes_per_device;
#endif

    int total_n_streams = createNStreams(new_n_streams);

    return total_n_streams;
  }

  int createNStreams(int new_n_streams) {
    if (new_n_streams <= n_streams_) {
      return n_streams_;
    }

    hapiStream_t* old_streams = streams_;

    streams_ = new hapiStream_t[new_n_streams];

    int i = 0;
    // Copy old streams
    for (; i < n_streams_; i++) {
      // TODO alt. use memcpy?
      streams_[i] = old_streams[i];
    }

    // Create new streams
    for (; i < new_n_streams; i++) {
      hapiCheck(hapiStreamCreate(&streams_[i]));
    }

    // Update
    n_streams_ = new_n_streams;
    delete [] old_streams;

    return n_streams_;
  }

  hapiStream_t getNextStream() {
    if (streams_ == NULL)
      return NULL;

    last_stream_id_ = (++last_stream_id_) % n_streams_;
    return streams_[last_stream_id_];
  }

  hapiStream_t getStream(int i) {
    if (streams_ == NULL)
      return NULL;

    if (i < 0 || i >= n_streams_)
      CmiAbort("[HAPI] invalid stream ID");
    return streams_[i];
  }

  int getNStreams() {
    if (!streams_) // NULL - default stream
      return 1;

    return n_streams_;
  }

  // Allocates device buffers.
  void allocateBuffers(hapiWorkRequest* wr) {
    for (int i = 0; i < wr->getBufferCount(); i++) {
      hapiBufferInfo& bi = wr->buffers[i];
      int index = bi.id;
      size_t size = bi.size;

      // if index value is invalid, use an available ID
      if (index < 0 || index >= NUM_BUFFERS) {
        bool is_found = false;
        for (int j = next_buffer_; j < NUM_BUFFERS*2; j++) {
          if (device_buffers_[j] == NULL) {
            index = j;
            is_found = true;
            break;
          }
        }

        // if no index was found, try to search for a value at the
        // beginning of the system addressed space
        if (!is_found) {
          for (int j = NUM_BUFFERS; j < next_buffer_; j++) {
            if (device_buffers_[j] == NULL) {
              index = j;
              is_found = true;
              break;
            }
          }
        }

        if (!is_found) {
          CmiAbort("[HAPI] ran out of device buffer indices");
        }

        next_buffer_ = index + 1;
        if (next_buffer_ == NUM_BUFFERS*2) {
          next_buffer_ = NUM_BUFFERS;
        }

        bi.id = index;
      }

      if (device_buffers_[index] == NULL) {
        // allocate device memory
        hapiCheck(hapiMalloc((void **)&device_buffers_[index], size));

#ifdef HAPI_DEBUG
        CmiPrintf("[HAPI] allocated buffer %d at %p, time: %.2f, size: %zu\n",
               index, device_buffers_[index], cutGetTimerValue(timerHandle),
               size);
#endif
      }
    }
  }

  // Initiates host-to-device data transfer.
  void hostToDeviceTransfer(hapiWorkRequest* wr) {
    for (int i = 0; i < wr->getBufferCount(); i++) {
      hapiBufferInfo& bi = wr->buffers[i];
      int index = bi.id;
      size_t size = bi.size;
      host_buffers_[index] = bi.host_buffer;

      if (bi.transfer_to_device) {
        hapiCheck(hapiMemcpyAsync(device_buffers_[index], host_buffers_[index], size,
                                  hapiMemcpyHostToDevice, wr->stream));

#ifdef HAPI_DEBUG
        CmiPrintf("[HAPI] transferring buffer %d from host to device, time: %.2f, "
               "size: %zu\n", index, cutGetTimerValue(timerHandle), size);
#endif
      }
    }
  }

  // Initiates device-to-host data transfer.
  void deviceToHostTransfer(hapiWorkRequest* wr) {
    for (int i = 0; i < wr->getBufferCount(); i++) {
      hapiBufferInfo& bi = wr->buffers[i];
      int index = bi.id;
      size_t size = bi.size;

      if (bi.transfer_to_host) {
        hapiCheck(hapiMemcpyAsync(host_buffers_[index], device_buffers_[index], size,
                                  hapiMemcpyDeviceToHost, wr->stream));

#ifdef HAPI_DEBUG
        CmiPrintf("[HAPI] transferring buffer %d from device to host, time %.2f, "
               "size: %zu\n", index, cutGetTimerValue(timerHandle), size);
#endif
      }
    }
  }

  // Frees device buffers.
  void freeBuffers(hapiWorkRequest* wr) {
    for (int i = 0; i < wr->getBufferCount(); i++) {
      hapiBufferInfo& bi = wr->buffers[i];
      int index = bi.id;

      if (bi.need_free) {
        hapiCheck(hapiFree(device_buffers_[index]));
        device_buffers_[index] = NULL;

#ifdef HAPI_DEBUG
        CmiPrintf("[HAPI] freed buffer %d, time %.2f\n",
               index, cutGetTimerValue(timerHandle));
#endif
      }
    }
  }

  // Run the user's kernel for the given work request.
  // This used to be a switch statement defined by the user to allow the runtime
  // to execute the correct kernel.
  void runKernel(hapiWorkRequest* wr) {
    if (wr->runKernel) {
      wr->runKernel(wr, wr->stream, device_buffers_);
    }
    // else, might be only for data transfer (or might be a bug?)
  }
};

#endif // __GPUMANAGER_H_
