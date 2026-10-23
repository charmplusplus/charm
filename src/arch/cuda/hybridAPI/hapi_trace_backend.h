#ifndef __HAPI_TRACE_BACKEND_H_
#define __HAPI_TRACE_BACKEND_H_

// The GPU trace backend behind the balancer's per-object device loads.
//
// The load attribution in hapi_impl.cpp (entry spans, correlation tables,
// shards, the SM-share model, the +gputrace sink) works on the three record
// kinds below. Each vendor's tracer fills them: CUPTI's activity records on
// CUDA (the walker in hapi_impl.cpp), rocprofiler-sdk's buffer records on HIP
// (hapi_trace_rocprof.cpp). The field names follow CUPTI's so the parse code
// reads the same on both.
#include <stdint.h>
#include <stddef.h>
#include <vector>

struct HapiTraceKernelRec {
  uint64_t start, end;              // device timestamps, ns, the backend's clock
  uint32_t deviceId, streamId, correlationId;
  uint32_t threadId;                // the OS thread that enqueued the dispatch (0 if the backend has none)
  uint32_t gridX, gridY, gridZ;     // blocks
  uint32_t blockX, blockY, blockZ;  // threads per block
  uint32_t registersPerThread;
  int32_t staticSharedMemory, dynamicSharedMemory;
  const char* name;                 // owned by the backend; valid for the run
};
struct HapiTraceApiRec {
  uint64_t start, end;              // host timestamps on the backend's clock
  uint32_t threadId;                // the OS thread id (gettid) of the caller
  uint32_t correlationId;           // shared with the kernel the call launched
  uint32_t cbid;                    // the API function, hapiTraceBackendCallbackName names it
};
struct HapiTraceCopyRec {
  char kind;                        // 'M' memcpy, 'S' memset
  uint32_t deviceId, streamId;
  uint64_t start, end, bytes;
  int copyKind;
};

// A HIP backend delivers its records already converted, one batch per buffer
// callback; a CUDA backend delivers raw CUPTI buffers and the walker converts
// as it reads. CuptiBufferItem (gpumanager.h) carries either.
struct HapiTraceBatch {
  std::vector<HapiTraceKernelRec> kernels;
  std::vector<HapiTraceApiRec> apis;
  std::vector<HapiTraceCopyRec> copies;
};

// The backend's operations, each implemented per vendor.
// Before the first vendor call of the process (hapiPresetEnvironment): a
// tracer that must register with the runtime before it initialises does so
// here; nothing happens on CUDA.
void hapiTraceBackendEarlyInit();
// Attach once per process (register the tracer); returns false if there is
// no tracer to attach (CHARM_LB_NO_CUPTI, or no rocprofiler).
bool hapiTraceBackendAttach();
// Enable the record kinds: kernels and API calls always, copies on request.
void hapiTraceBackendEnable(bool trace_copies);
// Deliver the records produced so far to the buffer queue (a forced flush
// also takes buffers whose last record is incomplete).
void hapiTraceBackendFlush(bool forced);
// The clock the records are stamped with, in ns.
uint64_t hapiTraceBackendTimestamp();
// The API function behind a cbid, or null.
const char* hapiTraceBackendCallbackName(uint32_t cbid);
// One more than the largest cbid the backend can report.
uint32_t hapiTraceBackendCallbackIdCount();
// Detach for good.
void hapiTraceBackendFinalize();

#endif // __HAPI_TRACE_BACKEND_H_
