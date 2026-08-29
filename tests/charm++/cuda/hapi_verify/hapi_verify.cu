#include <cuda_runtime.h>

__global__ void fillKernel(int* buf, int n, int val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = val + i;
}

#include <cstdio>
#include <cstdlib>

extern "C" void launchFill(int* dbuf, int n, int val, cudaStream_t stream) {
  int block = 128;
  int grid = (n + block - 1) / block;
  fillKernel<<<grid, block, 0, stream>>>(dbuf, n, val);
  /* A failed launch (e.g. no matching cubin and JIT unavailable) is
   * otherwise silent and leaves buffers untouched -- defect #6 of this
   * test, found only because its verification caught all-zero output. */
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launchFill: kernel launch failed: %s\n",
            cudaGetErrorString(err));
    abort();
  }
}
