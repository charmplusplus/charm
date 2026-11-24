// Device kernels of the HAPI runtime, compiled into libhybridapi beside
// hapi_impl.cpp: as HIP in HIP builds, as CUDA in CUDA builds
// (CMakeLists.txt, which defines HAPI_FLAG_KERNEL for hapi_impl.cpp when it
// does). Neither sees the Charm++ headers, so the interface is extern "C"
// with the stream passed as void*.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
typedef hipStream_t hapiKernelStream;
#define hapiKernelLastError hipGetLastError
#else
#include <cuda_runtime.h>
typedef cudaStream_t hapiKernelStream;
#define hapiKernelLastError cudaGetLastError
#endif

// A pinned completion flag (hapi_impl.cpp, HAPI_FLAG_SLOTS), stored by one
// thread after a system-scope fence. Both vendors document the fence for the
// calling thread's own writes; the stream orders its earlier work before this
// kernel. hipMemsetD32Async releases only its own write, which is why HIP
// writes the flag this way by default.
__global__ void hapiFlagKernel(unsigned int* p, unsigned int v) {
  __threadfence_system();
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP__)
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  *(volatile unsigned int*)p = v;
#endif
}

extern "C" int hapiFlagKernelImpl(void* p, unsigned int v, void* stream) {
  hapiFlagKernel<<<1, 1, 0, (hapiKernelStream)stream>>>((unsigned int*)p, v);
  return (int)hapiKernelLastError();
}
