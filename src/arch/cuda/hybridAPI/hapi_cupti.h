#ifndef __HAPI_CUPTI_H_
#define __HAPI_CUPTI_H_

// Use NVIDIA's CUPTI to get timers for GPU operations.
#include <cupti.h>

#define CUPTI_CALL(call)                                                    \
  do {                                                                      \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
          __FILE__, __LINE__, #call, errstr);                               \
      exit(-1);                                                             \
    }                                                                       \
  } while (0)

#define CUPTI_BUF_SIZE (32 * 1024)
#define CUPTI_ALIGN_SIZE (8)
#define CUPTI_ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

void cuptiInit();
void CUPTIAPI cuptiBufferRequested(uint8_t **buffer, size_t *size, size_t *max_num_records);
void CUPTIAPI cuptiBufferCompleted(CUcontext ctx, uint32_t stream_id, uint8_t *buffer, size_t size, size_t valid_size);

#endif // __HAPI_CUPTI_H_
