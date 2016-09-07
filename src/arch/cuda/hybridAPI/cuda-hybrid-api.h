/*
 * cuda-hybrid-api.h
 *
 * by Lukasz Wesolowski
 * 04.01.2008
 *
 * an interface for execution on the GPU
 *
 * description:
 * -user enqueues one or more work requests to the work
 * request queue (wrQueue) to be executed on the GPU
 * - a converse function (gpuProgressFn) executes periodically to
 * offload work requests to the GPU one at a time
 *
 */

#ifndef __CUDA_HYBRID_API_H__
#define __CUDA_HYBRID_API_H__


#ifdef __cplusplus
extern "C" {
#endif

/* initHybridAPI
   initializes the work request queue
*/
void initHybridAPI();

/* gpuProgressFn
   called periodically to check if the current kernel has completed,
   and invoke subsequent kernel */
void gpuProgressFn();

/* exitHybridAPI
   cleans up and deletes memory allocated for the queue
*/
void exitHybridAPI();


#ifdef GPU_MEMPOOL
// data and metadata reside in same chunk of memory
typedef struct _header {
  struct _header *next;
  int slot;
#ifdef GPU_MEMPOOL_DEBUG
  int size;
#endif
} Header;

typedef struct _bufferPool {
  Header *head;
  size_t size;
#ifdef GPU_MEMPOOL_DEBUG
  int num;
#endif
} BufferPool;

#endif // GPU_MEMPOOL

extern void cudaErrorDie(int err, const char* code, const char* file, int line);

#define cudaChk(code)                                                  \
  do { int e = (code); if (cudaSuccess != e) {                         \
    cudaErrorDie(e, #code, __FILE__, __LINE__); } } while (0)

#ifdef __cplusplus
}
#endif

#endif // __CUDA_HYBRID_API_H__
