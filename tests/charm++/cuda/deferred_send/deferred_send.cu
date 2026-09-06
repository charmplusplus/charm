#include <cuda_runtime.h>

__global__ void fillAfterDelay(int* a, int* c, int n, unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {}
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    a[i] = 11;
    if (c) c[i] = 33;
  }
}

void delayedFill(int* a, int* c, int n, unsigned long long cycles, cudaStream_t stream) {
  fillAfterDelay<<<1, 128, 0, stream>>>(a, c, n, cycles);
}
