// Device kernels of the HAPI runtime. Compiled by the vendor's device compiler
// (nvcc, or hipcc as HIP) into libhybridapi beside hapi_impl.cpp, which the
// C++ compiler builds; neither sees the Charm++ headers, so the interface is
// extern "C" with the stream as the only vendor type, named per vendor here.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define hapiStream_t hipStream_t
#define hapiMemsetAsync hipMemsetAsync
// Wavefronts are 64 lanes wide, but the reductions below fold 32 lanes at a
// time and index the per-warp scratch by threadIdx >> 5, which is correct on
// either width: the XOR offsets stay inside a 32-lane half.
#define HAPI_SHFL_XOR(v, o) __shfl_xor((v), (o))
#else
#include <cuda_runtime.h>
#define hapiStream_t cudaStream_t
#define hapiMemsetAsync cudaMemsetAsync
#define HAPI_SHFL_XOR(v, o) __shfl_xor_sync(0xffffffffu, (v), (o))
#endif
#include <stdint.h>
#include <stddef.h>

namespace {

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

// Content hash of [p, p+n): every 8-byte word is mixed with its position and
// the mixes are combined by XOR, so the result does not depend on which
// thread saw which word but does change if any word changes or two words
// swap. An unaligned range is walked by bytes. The tail (< 8 bytes) is folded
// by one thread. Each block XORs its partial into *out, which the launcher
// zeroes first.
__global__ void hapiHashKernel(const unsigned char* p, size_t n, unsigned long long* out) {
  unsigned long long h = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if ((((uintptr_t)p) & 7) == 0) {
    const unsigned long long* w = (const unsigned long long*)p;
    const size_t nw = n / 8;
    for (size_t i = tid; i < nw; i += stride)
      h ^= mix64(w[i] + 0x9e3779b97f4a7c15ULL * (unsigned long long)(i + 1));
    if (tid == 0) {
      unsigned long long t = 0;
      for (size_t i = nw * 8, k = 0; i < n; i++, k++) t |= (unsigned long long)p[i] << (8 * k);
      if (nw * 8 != n) h ^= mix64(t + 0x1234567ULL);
    }
  } else {
    for (size_t i = tid; i < n; i += stride)
      h ^= mix64((unsigned long long)p[i] + 0x9e3779b97f4a7c15ULL * (unsigned long long)(i + 1));
  }
  // Block reduction by XOR: warp shuffles, then one word per warp.
  for (int o = 16; o > 0; o >>= 1) h ^= HAPI_SHFL_XOR(h, o);
  __shared__ unsigned long long s[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) s[wid] = h;
  __syncthreads();
  if (wid == 0) {
    const int nwarps = (blockDim.x + 31) / 32;
    h = (lane < nwarps) ? s[lane] : 0ULL;
    for (int o = 16; o > 0; o >>= 1) h ^= HAPI_SHFL_XOR(h, o);
    if (lane == 0) atomicXor(out, h);
  }
}

// Compares two arrays of hashes; *flag becomes 1 + the first differing index
// (0 when they agree). The launcher zeroes *flag first.
__global__ void hapiHashCompareKernel(const unsigned long long* a, const unsigned long long* b,
                                      size_t count, unsigned long long* flag) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count && a[i] != b[i]) atomicMin(flag, (unsigned long long)(i + 1));
}

}  // namespace

extern "C" void hapiHashAsyncImpl(const void* p, size_t n, unsigned long long* d_out,
                                  hapiStream_t stream) {
  hapiMemsetAsync(d_out, 0, sizeof(unsigned long long), stream);
  if (n == 0) return;
  const int threads = 256;
  size_t units = ((((uintptr_t)p) & 7) == 0) ? (n + 7) / 8 : n;
  size_t blocks = (units + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  if (blocks == 0) blocks = 1;
  hapiHashKernel<<<(unsigned)blocks, threads, 0, stream>>>((const unsigned char*)p, n, d_out);
}

extern "C" void hapiHashCompareImpl(const unsigned long long* a, const unsigned long long* b,
                                    size_t count, unsigned long long* d_flag, hapiStream_t stream) {
  // The flag starts at the maximum; a mismatch stores 1 + its index and the
  // minimum wins, so the host reads ~0 as "all equal".
  hapiMemsetAsync(d_flag, 0xFF, sizeof(unsigned long long), stream);
  if (count == 0) return;
  const int threads = 256;
  const unsigned blocks = (unsigned)((count + threads - 1) / threads);
  hapiHashCompareKernel<<<blocks, threads, 0, stream>>>(a, b, count, d_flag);
}

// A completion flag written by a one-thread kernel: the system-scope fence is the release that
// hipMemsetD32Async lacks (the event record ahead of it in hapiHipMemsetD32Async), in one launch instead
// of an event record and a blit (Frontier 2026-10-22, +gpuflagkernel).
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
__global__ void hapiFlagKernel(unsigned int* p, unsigned int v) {
  __threadfence_system();
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
extern "C" int hapiFlagKernelImpl(void* p, unsigned int v, void* stream) {
  hapiFlagKernel<<<1, 1, 0, (hipStream_t)stream>>>((unsigned int*)p, v);
  return (int)hipGetLastError();
}
#endif
