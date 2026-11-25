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

// Check 6: one thread spins until the host sets hold[0], or until `ticks` of
// the device's wall clock have passed, and records which ended it in hold[1]
// (1 released, 2 timed out). `hold` is mapped pinned host memory.
__device__ __forceinline__ long long hapitestNow() {
#if defined(__HIP_PLATFORM_AMD__)
  return (long long)wall_clock64();
#else
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));   // nanoseconds
  return (long long)t;
#endif
}

__global__ void holdKernel(unsigned int* hold, long long ticks) {
  const long long t0 = hapitestNow();
  unsigned int outcome = 2;
  for (;;) {
#if defined(__HIP_PLATFORM_AMD__)
    const unsigned int released = __hip_atomic_load(&hold[0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    const unsigned int released = *(volatile unsigned int*)&hold[0];
#endif
    if (released != 0) { outcome = 1; break; }
    if (hapitestNow() - t0 > ticks) break;
  }
  hold[1] = outcome;
  __threadfence_system();
}

void hapitestHold(hapiStream_t stream, unsigned int* hold_dev, int timeout_ms) {
#if defined(__HIP_PLATFORM_AMD__)
  int dev = 0, khz = 0;
  if (hipGetDevice(&dev) != hipSuccess ||
      hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || khz <= 0)
    khz = 100000;   // 100 MHz, the MI200 series' wall clock
  const long long ticks = (long long)khz * timeout_ms;
#else
  const long long ticks = (long long)timeout_ms * 1000000LL;
#endif
  holdKernel<<<1, 1, 0, stream>>>(hold_dev, ticks);
}
