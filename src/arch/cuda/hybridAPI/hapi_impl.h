#ifndef __HAPI_SRC_H_
#define __HAPI_SRC_H_

#ifdef __cplusplus
extern "C" {
#endif

// Initialize & exit hybrid API.
void initHybridAPI();
void exitHybridAPI();

// Initializes event queues used for polling.
void initEventQueues();

// Registers callback handler functions.
void registerCallbacks();

// Polls for GPU work completion. Does not do anything if HAPI_CUDA_CALLBACK is defined.
void hapiPoll();

#ifdef HAPI_MEMPOOL
// data and metadata reside in same chunk of memory
typedef struct _header {
  struct _header *next;
  int slot;
#ifdef HAPI_MEMPOOL_DEBUG
  int size;
#endif
} Header;

typedef struct _bufferPool {
  Header *head;
  size_t size;
#ifdef HAPI_MEMPOOL_DEBUG
  int num;
#endif
} BufferPool;
#endif // HAPI_MEMPOOL

#ifdef __cplusplus
}
#endif

#endif // __HAPI_SRC_H_
