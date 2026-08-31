// Device-side half of d2dtest. Kept separate from d2dtest.C so the test
// program itself is compiled by charmc (a plain host compiler), the way user
// code actually sees the headers; only the kernel needs nvcc/hipcc.
//
// Includes hapi_portable.h rather than hapi.h so the same source builds under
// both CUDA and HIP without a single #ifdef here.
#include "hapi_portable.h"

#define D2D_BLOCK 256

// Fills d[j] with base + (j % PERIOD). Every value is a small integer scaled
// by powers of ten, so it is exact in a double and the receiver can compare
// with ==; there is no arithmetic here for the compiler to contract.
__global__ void fillKernel(double* d, int n, double base, int period) {
  int ti = blockDim.x * blockIdx.x + threadIdx.x;
  if (ti < n) {
    d[ti] = base + (double)(ti % period);
  }
}

// Returns the launch error rather than checking it here: hapiCheck lives in
// hapi.h, which drags in the Charm++ C++ runtime headers that this file is
// kept away from. d2dtest.C wraps the call in hapiCheck instead.
hapiError_t invokeFillKernel(double* d, int n, double base, int period,
                             hapiStream_t stream) {
  dim3 block_dim(D2D_BLOCK);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x);
  fillKernel<<<grid_dim, block_dim, 0, stream>>>(d, n, base, period);
  return hapiPeekAtLastError();
}
