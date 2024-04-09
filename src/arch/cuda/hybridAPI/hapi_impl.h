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

// HAPI init & exit functions
void hapiInit(char** argv);
void hapiExit();

// Polls for GPU work completion. Does not do anything if HAPI_CUDA_CALLBACK is defined.
void hapiPollEvents(void* param);

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
  void *chunk;
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

#ifdef __cplusplus
}
#endif

#endif // __HAPI_IMPL_H_
