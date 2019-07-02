#include <stdio.h>
#include "hapi.h"

__global__ void helloKernel() {

}

#if USE_WR
void run_hello(hapiWorkRequest* wr, cudaStream_t kernel_stream, void** devBuffers) {
  printf("Calling kernel...\n");
  helloKernel<<<wr->grid_dim, wr->block_dim, wr->shared_mem, kernel_stream>>>();
}

hapiWorkRequest* setupWorkRequest(cudaStream_t stream) {
  hapiWorkRequest* wr = hapiCreateWorkRequest();
  wr->setExecParams(dim3(1), dim3(1));
  wr->setRunKernel(run_hello);
  wr->setStream(stream);

  return wr;
}
#else
void invokeKernel(cudaStream_t stream) {
  dim3 grid_dim(1, 1);
  dim3 block_dim(16, 16);

  printf("Calling kernel...\n");
  helloKernel<<<grid_dim, block_dim, 0, stream>>>();
}
#endif
