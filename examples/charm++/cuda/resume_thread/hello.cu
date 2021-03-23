#include <stdio.h>
#include <stdlib.h>
#include "hapi.h"

__global__ void helloKernel() {}

void invokeKernel(cudaStream_t stream) {
  helloKernel<<<dim3(1, 1), dim3(1, 1), 0, stream>>>();
}
