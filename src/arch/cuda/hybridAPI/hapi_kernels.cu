// Device kernels of the HAPI runtime, compiled by hipcc as HIP into
// libhybridapi beside hapi_impl.cpp (CMakeLists.txt, HIP builds only). Neither
// sees the Charm++ headers, so the interface is extern "C" with the stream
// passed as void*.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>

// A pinned completion flag (hapi_impl.cpp, HAPI_FLAG_SLOTS), written by one
// thread after a system-scope fence: the fence releases the stream's earlier
// work to every device and process that reads it on the flag, which
// hipMemsetD32Async alone does not.
__global__ void hapiFlagKernel(unsigned int* p, unsigned int v) {
  __threadfence_system();
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" int hapiFlagKernelImpl(void* p, unsigned int v, void* stream) {
  hapiFlagKernel<<<1, 1, 0, (hipStream_t)stream>>>((unsigned int*)p, v);
  return (int)hipGetLastError();
}
#endif
