#include <stdio.h>
#include <stdlib.h>
#include "hapi.h"

__global__ void helloKernel() {
  printf("Hello from HAPI kernel!\n");
}

void kernelSetup(hapiStream_t stream, const CkCallback& cb) {
  helloKernel<<<dim3(1, 1), dim3(1, 1), 0, stream>>>();
  CkPrintf("stream ptr=%p, sizeof(hapiStream_t)=%zu\n", (void*)stream, sizeof(stream));
  hapiAddCallback(stream, cb, nullptr);
}
