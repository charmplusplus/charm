#include "hapi.h"

__global__ void busyKernel(long long clock_count) {
  long long start_clock = clock64();
  long long clock_offset = 0;

  while (clock_offset < clock_count) {
    long long end_clock = clock64();

    clock_offset = end_clock - start_clock;
  }
}

#ifdef USE_WR
void runEmpty(struct hapiWorkRequest* wr, cudaStream_t kernel_stream, void** deviceBuffers) {
  busyKernel<<<wr->grid_dim, wr->block_dim, wr->shared_mem, kernel_stream>>>(*(long long*)wr->getUserData());
  hapiCheck(cudaPeekAtLastError());
}
#endif

void kernelSetup(char* h_A, char* h_B, char* d_A, char* d_B, size_t data_size,
                 long long clock_count, cudaStream_t stream, void* h2d_cb,
                 void* kernel_cb, void* d2h_cb) {
#ifdef USE_WR
  // DEPRECATED
  // No data transfers with this API
  hapiWorkRequest* wr = hapiCreateWorkRequest();
  wr->setExecParams(dim3(1, 1), dim3(1, 1));
  wr->setStream(stream);
  wr->setCallback(cb);
#ifdef HAPI_TRACE
  wr->setTraceName("qdtest");
#endif
  wr->setRunKernel(runEmpty);
  wr->copyUserData(&clock_count, sizeof(long long));

  hapiEnqueue(wr);
#else
  if (data_size > 0) {
    hapiCheck(cudaMemcpyAsync(d_A, h_A, data_size, cudaMemcpyHostToDevice, stream));
    hapiAddCallback(stream, h2d_cb);
  }

  busyKernel<<<dim3(1, 1), dim3(1, 1), 0, stream>>>(clock_count);
  hapiCheck(cudaPeekAtLastError());
  hapiAddCallback(stream, kernel_cb);

  if (data_size > 0) {
    hapiCheck(cudaMemcpyAsync(h_B, d_B, data_size, cudaMemcpyDeviceToHost, stream));
    hapiAddCallback(stream, d2h_cb);
  }
#endif
}
