// Device half of tests/charm++/cuda/hapitest. Deliberately tiny: everything
// under test lives in hapitest.C, which is compiled by charmc (a plain host
// compiler) precisely so that the portability header is exercised the way user
// code sees it. This file only has to supply a kernel and a launch site.
//
// It includes hapi_portable.h rather than hapi.h so that the same source
// compiles under nvcc and hipcc without dragging in the C++ runtime headers.

#include "hapi_portable.h"

__global__ void scaleAddKernel(const double* in, double* out, int n, double addend) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0 + addend;
}

void hapitestLaunch(hapiStream_t stream, const double* in, double* out, int n,
                    double addend) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scaleAddKernel<<<blocks, threads, 0, stream>>>(in, out, n, addend);
}
