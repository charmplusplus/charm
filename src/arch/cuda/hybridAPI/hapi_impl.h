#ifndef __HAPI_IMPL_H_
#define __HAPI_IMPL_H_

#ifdef __cplusplus
extern "C" {
#endif

// Mempool macros
// Update for new row, again this shouldn't be hard coded!
#define HAPI_MEMPOOL_NUM_SLOTS 20
// Pre-allocated buffers will be at least this big (in bytes).
#define HAPI_MEMPOOL_MIN_BUFFER_SIZE 256
// Scale the amount of memory each node pins.
#define HAPI_MEMPOOL_SCALE 1.0

// Init & exit functions
void hapiInitCsv();
void hapiInitCpv();
void hapiExitCsv();

// Maps PEs to devices
void hapiMapping(char** argv);

// Registers callback handler functions.
void hapiRegisterCallbacks();

// Polls for GPU work completion. Does not do anything if HAPI_CUDA_CALLBACK is defined.
void hapiPollEvents();

// BufferPool constructs for mempool implementation.
// Data and metadata reside in same chunk of memory.
typedef struct _bufferPoolHeader {
  struct _bufferPoolHeader *next;
  int slot;
#ifdef HAPI_MEMPOOL_DEBUG
  size_t size;
#endif
} BufferPoolHeader;

typedef struct _bufferPool {
  BufferPoolHeader *head;
  size_t size;
#ifdef HAPI_MEMPOOL_DEBUG
  int num;
#endif
} BufferPool;

// PE-GPU mapping types
enum class Mapping {
  None, // Mapping is explicitly performed by the user
  Block,
  RoundRobin
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
#endif // HAPI_TRACE

#ifdef __cplusplus
}
#endif

#endif // __HAPI_IMPL_H_
