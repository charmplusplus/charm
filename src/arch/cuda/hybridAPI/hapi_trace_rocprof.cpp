// The HIP trace backend: rocprofiler-sdk's buffered tracing, delivering the
// neutral records of hapi_trace_backend.h that the balancer's load
// attribution in hapi_impl.cpp consumes. The CUDA counterpart (CUPTI) lives
// in hapi_impl.cpp.
//
// Shape: a tool registered with rocprofiler before the HIP runtime initialises
// (hapiTraceBackendEarlyInit, from hapiPresetEnvironment) creates one context
// with kernel-dispatch and HIP-runtime-API buffer tracing and a second with
// memory-copy tracing (started only when the +gputrace sink wants copies),
// plus a code-object callback that learns kernel names and register counts.
// The buffer callback converts each delivery into a HapiTraceBatch and queues
// it (hapiTraceBackendDeliver); the parse frees the batch. Polling calls
// (names containing "Query", the error peeks and hipGetDevice) are excluded
// from the API service as they are on CUDA.
#include "hapi_portable.h"
#include "converse.h"   // CMK_LBDB_ON comes from the machine configuration behind it
#if CMK_HIP && CMK_LBDB_ON

#include "hapi_trace_backend.h"
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>
#include <atomic>
#include <mutex>
#include <string>
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>

void hapiTraceBackendDeliver(HapiTraceBatch* batch, size_t approxBytes);   // hapi_impl.cpp

namespace {

struct KernelSymbol {
  std::string name;
  uint32_t vgprs = 0;
  uint32_t group_segment = 0;
};

struct RocprofState {
  bool configured = false;      // rocprofiler_configure ran
  bool usable = false;          // contexts and buffers exist
  bool started = false;
  bool copies_started = false;
  rocprofiler_context_id_t ctx{0}, copy_ctx{0};
  rocprofiler_buffer_id_t buf{0}, copy_buf{0};
  std::mutex symbols_lock;
  std::unordered_map<uint64_t, KernelSymbol> symbols;   // kernel_id -> symbol (node-stable: records keep c_str())
  std::unordered_map<uint64_t, int> agent_device;       // agent id handle -> HIP device ordinal
  std::vector<std::string> api_names;                   // by operation id
  std::vector<rocprofiler_tracing_operation_t> api_ops; // the operations traced (no polls)
};
RocprofState g_rp;

int deviceOfAgent(rocprofiler_agent_id_t a) {
  auto it = g_rp.agent_device.find(a.handle);
  return it != g_rp.agent_device.end() ? it->second : 0;
}

// Kernel names and register counts arrive with the code object, before any dispatch of them.
void codeObjectCallback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
      record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
      record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
    return;
  auto* d = (rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*)record.payload;
  std::lock_guard<std::mutex> lk(g_rp.symbols_lock);
  KernelSymbol& s = g_rp.symbols[d->kernel_id];
  s.name = d->kernel_name ? d->kernel_name : "?";
  s.vgprs = d->arch_vgpr_count + d->accum_vgpr_count;
  s.group_segment = d->group_segment_size;
}

// One buffer delivery -> one batch for the parse.
void bufferCallback(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers,
                    size_t n, void*, uint64_t) {
  HapiTraceBatch* b = new HapiTraceBatch;
  size_t bytes = 0;
  for (size_t i = 0; i < n; i++) {
    rocprofiler_record_header_t* h = headers[i];
    if (h == nullptr || h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
    if (h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      auto* r = (rocprofiler_buffer_tracing_kernel_dispatch_record_t*)h->payload;
      const rocprofiler_kernel_dispatch_info_t& di = r->dispatch_info;
      HapiTraceKernelRec k;
      memset(&k, 0, sizeof k);
      k.start = r->start_timestamp; k.end = r->end_timestamp;
      k.deviceId = (uint32_t)deviceOfAgent(di.agent_id);
      k.streamId = (uint32_t)di.queue_id.handle;
      k.correlationId = (uint32_t)r->correlation_id.internal;
      // rocprofiler reports the grid in work-items; the parse wants blocks.
      k.blockX = di.workgroup_size.x ? di.workgroup_size.x : 1;
      k.blockY = di.workgroup_size.y ? di.workgroup_size.y : 1;
      k.blockZ = di.workgroup_size.z ? di.workgroup_size.z : 1;
      k.gridX = (di.grid_size.x + k.blockX - 1) / k.blockX;
      k.gridY = (di.grid_size.y + k.blockY - 1) / k.blockY;
      k.gridZ = (di.grid_size.z + k.blockZ - 1) / k.blockZ;
      k.staticSharedMemory = (int32_t)di.group_segment_size;
      k.dynamicSharedMemory = 0;
      {
        std::lock_guard<std::mutex> lk(g_rp.symbols_lock);
        auto it = g_rp.symbols.find(di.kernel_id);
        if (it != g_rp.symbols.end()) { k.name = it->second.name.c_str(); k.registersPerThread = it->second.vgprs; }
        else k.name = "?";
      }
      // The HIP runtime's own blit kernels (__amd_rocclr_fillBuffer*, __amd_rocclr_copyBuffer*): on AMD a
      // memset or a device-to-device copy is a kernel dispatch, so every completion flag and channel put
      // arrived here as a kernel nobody owns (unowned share 0.8 against the A40's 0.66, Frontier
      // 2026-10-22). CUDA reports them as copy-engine records; they go the same way here, 'S' for a fill,
      // 'M' for a copy, their size unknown.
      if (strncmp(k.name, "__amd_rocclr_", 13) == 0) {
        HapiTraceCopyRec c{strstr(k.name, "fill") ? 'S' : 'M', k.deviceId, k.streamId, k.start, k.end, 0, 0};
        b->copies.push_back(c);
        bytes += sizeof(*r);
        continue;
      }
      b->kernels.push_back(k);
      bytes += sizeof(*r);
    } else if (h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API) {
      auto* r = (rocprofiler_buffer_tracing_hip_api_record_t*)h->payload;
      HapiTraceApiRec a{r->start_timestamp, r->end_timestamp, (uint32_t)r->thread_id,
                        (uint32_t)r->correlation_id.internal, (uint32_t)r->operation};
      b->apis.push_back(a);
      bytes += sizeof(*r);
    } else if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
      auto* r = (rocprofiler_buffer_tracing_memory_copy_record_t*)h->payload;
      HapiTraceCopyRec c{'M', (uint32_t)deviceOfAgent(r->dst_agent_id), 0, r->start_timestamp, r->end_timestamp,
                         r->bytes, (int)r->operation};
      b->copies.push_back(c);
      bytes += sizeof(*r);
    }
  }
  if (b->kernels.empty() && b->apis.empty() && b->copies.empty()) { delete b; return; }
  hapiTraceBackendDeliver(b, bytes);
}

rocprofiler_status_t agentsCallback(rocprofiler_agent_version_t, const void** agents, size_t n, void*) {
  int gpu = 0;
  for (size_t i = 0; i < n; i++) {
    const auto* a = (const rocprofiler_agent_v0_t*)agents[i];
    if (a->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
    // logical_node_type_id is the index among the GPU agents, the HIP ordinal
    // when no device is hidden by the environment.
    g_rp.agent_device[a->id.handle] = a->logical_node_type_id >= 0 ? a->logical_node_type_id : gpu;
    gpu++;
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

bool isPollName(const char* name) {
  return strstr(name, "Query") != nullptr || strcmp(name, "hipGetLastError") == 0 ||
         strcmp(name, "hipPeekAtLastError") == 0 || strcmp(name, "hipGetDevice") == 0;
}

int toolInit(rocprofiler_client_finalize_t, void*) {
  rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, agentsCallback,
                                     sizeof(rocprofiler_agent_t), nullptr);
  // The traced HIP API operations: every runtime call but the polls.
  const uint32_t nops = (uint32_t)ROCPROFILER_HIP_RUNTIME_API_ID_LAST;
  g_rp.api_names.assign(nops + 1, std::string());
  for (uint32_t op = 0; op < nops; op++) {
    const char* name = nullptr; uint64_t len = 0;
    if (rocprofiler_query_buffer_tracing_kind_operation_name(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                                             (rocprofiler_tracing_operation_t)op, &name, &len)
            != ROCPROFILER_STATUS_SUCCESS || name == nullptr)
      continue;
    g_rp.api_names[op] = name;
    // Only the calls that dispatch work carry the correlation the attribution needs (the launching call's
    // thread and entry-method span name the kernel's object); event records, stream waits, queries and
    // device lookups were intercepted for nothing at 20-35 us each under seven PEs (Frontier 2026-10-22,
    // ~1M flag-write event records per process per run). CHARM_TRACE_ALL_API=1 restores every call.
    static const bool all_api = getenv("CHARM_TRACE_ALL_API") != nullptr;
    const bool dispatches = strncmp(name, "hipLaunch", 9) == 0 || strncmp(name, "hipExtLaunch", 12) == 0 ||
                            strncmp(name, "hipModuleLaunch", 15) == 0 || strncmp(name, "hipGraphLaunch", 14) == 0 ||
                            strncmp(name, "hipMemcpy", 9) == 0 || strncmp(name, "hipMemset", 9) == 0 ||
                            strncmp(name, "hipDrvMemcpy", 12) == 0 || strncmp(name, "hipExtModuleLaunch", 18) == 0;
    if (!isPollName(name) && (all_api || dispatches)) g_rp.api_ops.push_back((rocprofiler_tracing_operation_t)op);
  }
  fprintf(stderr, "[hapi-trace] HIP API operations traced: %zu of %u (%s)\n", g_rp.api_ops.size(), (unsigned)nops,
          getenv("CHARM_TRACE_ALL_API") ? "every call but the polls" : "dispatching calls only; CHARM_TRACE_ALL_API=1 for every call");
  if (rocprofiler_create_context(&g_rp.ctx) != ROCPROFILER_STATUS_SUCCESS) return -1;
  if (rocprofiler_create_context(&g_rp.copy_ctx) != ROCPROFILER_STATUS_SUCCESS) return -1;
  // 1 MB buffers, delivered at 3/4 full and at every flush.
  const size_t buf_bytes = 1 << 20;
  if (rocprofiler_create_buffer(g_rp.ctx, buf_bytes, buf_bytes * 3 / 4, ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                bufferCallback, nullptr, &g_rp.buf) != ROCPROFILER_STATUS_SUCCESS) return -1;
  if (rocprofiler_create_buffer(g_rp.copy_ctx, buf_bytes, buf_bytes * 3 / 4, ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                bufferCallback, nullptr, &g_rp.copy_buf) != ROCPROFILER_STATUS_SUCCESS) return -1;
  if (rocprofiler_configure_buffer_tracing_service(g_rp.ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
                                                   g_rp.buf) != ROCPROFILER_STATUS_SUCCESS) return -1;
  if (rocprofiler_configure_buffer_tracing_service(g_rp.ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                                   g_rp.api_ops.data(), g_rp.api_ops.size(), g_rp.buf)
          != ROCPROFILER_STATUS_SUCCESS) return -1;
  if (rocprofiler_configure_callback_tracing_service(g_rp.ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                                                     codeObjectCallback, nullptr) != ROCPROFILER_STATUS_SUCCESS)
    return -1;
  if (rocprofiler_configure_buffer_tracing_service(g_rp.copy_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0,
                                                   g_rp.copy_buf) != ROCPROFILER_STATUS_SUCCESS) return -1;
  g_rp.usable = true;
  return 0;
}
void toolFini(void*) {}

rocprofiler_tool_configure_result_t* toolConfigure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
  id->name = "charm-hapi";
  static rocprofiler_tool_configure_result_t result{sizeof(rocprofiler_tool_configure_result_t), toolInit, toolFini, nullptr};
  g_rp.configured = true;
  return &result;
}

}  // namespace

void hapiTraceBackendEarlyInit() {
  static const bool no_trace = (getenv("CHARM_LB_NO_CUPTI") != nullptr);   // the same switch as on CUDA
  if (no_trace || g_rp.configured) return;
  // rocprofiler-sdk's inline queue interposition (1.3, ROCm 7.14) parks one of its pool threads in an
  // active wait per in-flight kernel (ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS, default 4), and a barrier
  // on another process's IPC event then starves the pool: every multi-process leanmd run hung within four
  // steps on Frontier, 2026-10-22, with the PEs blocked on the queue's gate lock. The legacy intercept
  // queue traces the same dispatches without it (60 steps, under 3% over no tracing); chosen here unless
  // the environment already says otherwise.
  setenv("ROCPROFILER_QUEUE_INTERPOSITION", "false", 0);
  const rocprofiler_status_t st = rocprofiler_force_configure(toolConfigure);
  if (st != ROCPROFILER_STATUS_SUCCESS)
    fprintf(stderr, "HAPI> rocprofiler-sdk registration failed (%d): the balancer will see no device load\n", (int)st);
}

bool hapiTraceBackendAttach() {
  if (!g_rp.configured) hapiTraceBackendEarlyInit();
  if (!g_rp.usable) {
    static bool said = false;
    if (!said) { said = true; CmiPrintf("HAPI> rocprofiler-sdk is not available in this process: no GPU load tracing\n"); }
    return false;
  }
  return true;
}

void hapiTraceBackendEnable(bool trace_copies) {
  if (!g_rp.usable) return;
  if (!g_rp.started) {
    if (rocprofiler_start_context(g_rp.ctx) == ROCPROFILER_STATUS_SUCCESS) g_rp.started = true;
    else CmiPrintf("HAPI> rocprofiler-sdk: could not start the tracing context\n");
  }
  if (trace_copies && !g_rp.copies_started) {
    if (rocprofiler_start_context(g_rp.copy_ctx) == ROCPROFILER_STATUS_SUCCESS) g_rp.copies_started = true;
  }
}

void hapiTraceBackendFlush(bool) {
  if (!g_rp.started) return;
  rocprofiler_flush_buffer(g_rp.buf);
  if (g_rp.copies_started) rocprofiler_flush_buffer(g_rp.copy_buf);
}

uint64_t hapiTraceBackendTimestamp() {
  rocprofiler_timestamp_t t = 0;
  rocprofiler_get_timestamp(&t);
  return (uint64_t)t;
}

const char* hapiTraceBackendCallbackName(uint32_t cbid) {
  if (cbid >= g_rp.api_names.size() || g_rp.api_names[cbid].empty()) return nullptr;
  return g_rp.api_names[cbid].c_str();
}

uint32_t hapiTraceBackendCallbackIdCount() { return (uint32_t)ROCPROFILER_HIP_RUNTIME_API_ID_LAST; }

void hapiTraceBackendFinalize() {
  if (g_rp.started) { rocprofiler_stop_context(g_rp.ctx); g_rp.started = false; }
  if (g_rp.copies_started) { rocprofiler_stop_context(g_rp.copy_ctx); g_rp.copies_started = false; }
}

#endif  // CMK_HIP && CMK_LBDB_ON
