#include <stdio.h>
#include <stdlib.h>
#include "hapi.h"

__global__ void helloKernel() {}

void kernelSetup(cudaStream_t stream, void* cb) {
  helloKernel<<<dim3(1, 1), dim3(1, 1), 0, stream>>>();

  hapiAddCallback(stream, cb);
}
