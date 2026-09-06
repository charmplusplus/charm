#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
#define NEV 64
struct Handles { cudaIpcMemHandle_t mem; cudaIpcEventHandle_t src; cudaIpcEventHandle_t dst[NEV]; };
int main() {
  int p2c[2], c2p[2]; pipe(p2c); pipe(c2p); const size_t sz = (size_t)1 << 20; const size_t big = (size_t)NEV * sz;
  pid_t pid = fork();
  if (pid == 0) {
    Handles h; read(p2c[0], &h, sizeof h);
    CK(cudaSetDevice(1)); void* src; CK(cudaIpcOpenMemHandle(&src, h.mem, cudaIpcMemLazyEnablePeerAccess));
    cudaEvent_t sev; CK(cudaIpcOpenEventHandle(&sev, h.src)); cudaEvent_t dev[NEV]; for (int i = 0; i < NEV; i++) CK(cudaIpcOpenEventHandle(&dev[i], h.dst[i]));
    cudaEvent_t loc[NEV]; for (int i = 0; i < NEV; i++) CK(cudaEventCreateWithFlags(&loc[i], cudaEventDisableTiming));
    void* dst; CK(cudaMalloc(&dst, big)); cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    auto bench = [&](const char* name, bool waitsrc, bool recdst, bool reclocal) {
      CK(cudaStreamSynchronize(s)); double t0 = now();
      for (int i = 0; i < NEV; i++) {
        if (waitsrc) CK(cudaStreamWaitEvent(s, sev, 0));
        CK(cudaMemcpyAsync((char*)dst + i * sz, (char*)src + i * sz, sz, cudaMemcpyDefault, s));
        if (recdst) CK(cudaEventRecord(dev[i], s));
        if (reclocal) CK(cudaEventRecord(loc[i], s));
      }
      double t1 = now(); CK(cudaStreamSynchronize(s)); double t2 = now();
      printf("child %-44s issue %.2f ms, done %.2f ms -> %.3f ms per 1 MB copy (%.1f GB/s)\n", name, (t1 - t0) * 1e3, (t2 - t0) * 1e3, (t2 - t0) * 1e3 / NEV, big / (t2 - t0) / 1e9);
      return 0; };
    bench("64 copies only", false, false, false);
    bench("+ local event record per copy", false, false, true);
    bench("+ wait on imported src event per copy", true, false, false);
    bench("+ record imported dst event per copy", false, true, false);
    bench("runtime pattern: wait src + record dst", true, true, false);
    bench("runtime pattern + local record", true, true, true);
    int ok = 1; write(c2p[1], &ok, sizeof ok); return 0;
  }
  CK(cudaSetDevice(0)); void* buf; CK(cudaMalloc(&buf, big)); CK(cudaMemset(buf, 1, big));
  Handles h; CK(cudaIpcGetMemHandle(&h.mem, buf));
  cudaEvent_t sev; CK(cudaEventCreateWithFlags(&sev, cudaEventInterprocess | cudaEventDisableTiming)); CK(cudaEventRecord(sev, 0)); CK(cudaEventSynchronize(sev)); CK(cudaIpcGetEventHandle(&h.src, sev));
  for (int i = 0; i < NEV; i++) { cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventInterprocess | cudaEventDisableTiming)); CK(cudaIpcGetEventHandle(&h.dst[i], ev)); }
  write(p2c[1], &h, sizeof h); int ok = 0; read(c2p[0], &ok, sizeof ok); waitpid(pid, NULL, 0); return 0; }
