#include <stdio.h>
#include <stdlib.h>
#include "hapi.h"

__global__ void helloKernel() {}

void runHello(struct hapiWorkRequest* wr, cudaStream_t kernel_stream,
              void** deviceBuffers) {
  helloKernel<<<wr->grid_dim, wr->block_dim, wr->shared_mem, kernel_stream>>>();
}

void kernelSetup(cudaStream_t stream, const CkCallback& cb) {
#ifdef USE_WR
  // DEPRECATED
  hapiWorkRequest* wr = hapiCreateWorkRequest();
  wr->setExecParams(dim3(1, 1), dim3(1, 1));
  wr->setStream(stream);
  wr->setCallback(cb);
#ifdef HAPI_TRACE
  wr->setTraceName("hello");
#endif
  wr->setRunKernel(runHello);

  hapiEnqueue(wr);
#else
  helloKernel<<<dim3(1, 1), dim3(1, 1), 0, stream>>>();
  hapiAddCallback(stream, cb);
#endif
}
