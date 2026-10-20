// Standalone probe of what the device pool's vmm backend needs from the
// driver, one process, no Charm++: for each visible device, the VMM
// attributes, the allocation granularity, cuMemCreate of one chunk under each
// property combination the pool tries (RDMA-capable + POSIX fd, fd only, RDMA
// only, plain), then the full IPC path against itself -- reserve a range, map
// the chunk, export its fd, duplicate the fd through pidfd_getfd exactly as a
// peer would, import it, map it into a mirror range, and check that a write
// through one mapping reads back through the other.
//
//   nvcc -o vmm_probe vmm_probe.cu -lcuda
//   srun -n 1 ./vmm_probe [chunk_mb]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>

static const char* name(CUresult r) { const char* s = "?"; cuGetErrorName(r, &s); return s; }
#define CU(call) do { CUresult _r = (call); if (_r != CUDA_SUCCESS) { printf("    FAIL %s -> %s\n", #call, name(_r)); return false; } } while (0)

static bool tryCreate(int dev, size_t chunk, bool fd, bool rdma, CUmemGenericAllocationHandle* out,
                      CUmemAllocationProp* used) {
  CUmemAllocationProp prop; memset(&prop, 0, sizeof prop);
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  if (fd) prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (rdma) prop.allocFlags.gpuDirectRDMACapable = 1;
  size_t gmin = 0, grec = 0;
  CUresult g1 = cuMemGetAllocationGranularity(&gmin, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  CUresult g2 = cuMemGetAllocationGranularity(&grec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  size_t sz = chunk;
  if (g1 == CUDA_SUCCESS && gmin) sz = (chunk + gmin - 1) / gmin * gmin;
  CUresult r = cuMemCreate(out, sz, &prop, 0);
  printf("  fd=%d rdma=%d: granularity min %s/%zu KB rec %s/%zu KB; cuMemCreate(%zu MB) -> %s\n",
         (int)fd, (int)rdma, name(g1), gmin >> 10, name(g2), grec >> 10, sz >> 20, name(r));
  if (r == CUDA_SUCCESS) *used = prop;
  return r == CUDA_SUCCESS;
}

static bool ipcRoundTrip(int dev, size_t chunk, CUmemGenericAllocationHandle h) {
  CUdeviceptr base = 0, mirror = 0;
  CU(cuMemAddressReserve(&base, chunk * 4, 0, 0, 0));
  CU(cuMemMap(base, chunk, 0, h, 0));
  CUmemAccessDesc acc; memset(&acc, 0, sizeof acc);
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc.location.id = dev;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU(cuMemSetAccess(base, chunk, &acc, 1));
  int fd = -1;
  CU(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  printf("    exported chunk as fd %d\n", fd);
#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
  int pidfd = (int)syscall(SYS_pidfd_open, getpid(), 0);
  if (pidfd < 0) { printf("    FAIL pidfd_open: %s\n", strerror(errno)); return false; }
  int dup = (int)syscall(SYS_pidfd_getfd, pidfd, fd, 0);
  if (dup < 0) { printf("    FAIL pidfd_getfd: %s (kernel.yama.ptrace_scope?)\n", strerror(errno)); return false; }
  printf("    pidfd_getfd duplicated it as fd %d\n", dup);
#else
  int dup = fd;
  printf("    (no pidfd syscalls in this libc; importing the exporter's own fd)\n");
#endif
  CUmemGenericAllocationHandle h2;
  CU(cuMemImportFromShareableHandle(&h2, (void*)(uintptr_t)dup, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  CU(cuMemAddressReserve(&mirror, chunk * 4, 0, 0, 0));
  CU(cuMemMap(mirror, chunk, 0, h2, 0));
  CU(cuMemSetAccess(mirror, chunk, &acc, 1));
  const unsigned pattern = 0xC0FFEE01u;
  if (cudaMemset((void*)base, 0, 4096) != cudaSuccess) { printf("    FAIL memset\n"); return false; }
  CU(cuMemsetD32(base + 1024, pattern, 8));
  unsigned back[8] = {0};
  CU(cuMemcpyDtoH(back, mirror + 1024, sizeof back));
  bool ok = true;
  for (int i = 0; i < 8; i++) ok = ok && back[i] == pattern;
  printf("    write through the heap, read through the mirror: %s\n", ok ? "MATCH" : "MISMATCH");
  return ok;
}

int main(int argc, char** argv) {
  const size_t chunk = (size_t)(argc > 1 ? atol(argv[1]) : 256) << 20;
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) { printf("cuInit -> %s\n", name(r)); return 1; }
  int n = 0; cuDeviceGetCount(&n);
  printf("vmm_probe: %d device(s), chunk %zu MB, pid %d\n", n, chunk >> 20, (int)getpid());
  int failures = 0;
  for (int dev = 0; dev < n; dev++) {
    CUdevice cudev; cuDeviceGet(&cudev, dev);
    char dn[128] = {0}; cuDeviceGetName(dn, sizeof dn, cudev);
    int vmm = 0, fdok = 0, rdma = 0;
    cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cudev);
    cuDeviceGetAttribute(&fdok, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, cudev);
    cuDeviceGetAttribute(&rdma, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cudev);
    cudaSetDevice(dev);
    size_t freeB = 0, totalB = 0; cudaMemGetInfo(&freeB, &totalB);
    printf("device %d %s: vmm=%d posix_fd=%d rdma_with_vmm=%d, %zu of %zu MB free\n",
           dev, dn, vmm, fdok, rdma, freeB >> 20, totalB >> 20);
    CUmemGenericAllocationHandle h; CUmemAllocationProp used;
    bool got = false;
    const bool combos[4][2] = {{true, true}, {true, false}, {false, true}, {false, false}};
    for (int c = 0; c < 4 && !got; c++) got = tryCreate(dev, chunk, combos[c][0], combos[c][1], &h, &used);
    if (!got) { printf("  no property combination works on device %d\n", dev); failures++; continue; }
    if (used.requestedHandleTypes == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      if (!ipcRoundTrip(dev, chunk, h)) failures++;
    } else {
      printf("  (fd export not available: the pool's peer mapping cannot work on this device)\n");
      failures++;
    }
  }
  printf("vmm_probe: %s\n", failures ? "FAILURES" : "all passed");
  return failures ? 1 : 0;
}
