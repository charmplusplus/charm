#include <cstdio>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
int main() {
  int p2c[2], c2p[2]; pipe(p2c); pipe(c2p); const size_t big = (size_t)256 << 20;
  pid_t pid = fork();
  if (pid == 0) {  // child: device 1, importer
    cudaIpcMemHandle_t h; read(p2c[0], &h, sizeof h);
    CK(cudaSetDevice(1)); void* src = NULL;
    for (int mode = 0; mode < 2; mode++) {
      if (mode == 1) { cudaError_t e = cudaDeviceEnablePeerAccess(0, 0); printf("child: explicit cudaDeviceEnablePeerAccess(0): %s\n", cudaGetErrorString(e)); }
      if (src) CK(cudaIpcCloseMemHandle(src));
      CK(cudaIpcOpenMemHandle(&src, h, cudaIpcMemLazyEnablePeerAccess));
      void* dst; CK(cudaMalloc(&dst, big)); cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      size_t sizes[] = {(size_t)1 << 20, (size_t)8 << 20, (size_t)64 << 20, (size_t)128 << 20};
      for (size_t sz : sizes) {
        CK(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDefault, s)); CK(cudaStreamSynchronize(s));
        double t0 = now(); for (int r = 0; r < 4; r++) CK(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDefault, s)); double t1 = now(); CK(cudaStreamSynchronize(s)); double t2 = now();
        double t3 = now(); for (int r = 0; r < 4; r++) CK(cudaMemcpyPeerAsync(dst, 1, src, 0, sz, s)); double t4 = now(); CK(cudaStreamSynchronize(s)); double t5 = now();
        printf("child mode%d %4zu MB: memcpyAsync(Default) issue %.2f ms, total %.2f ms = %.2f GB/s | memcpyPeerAsync issue %.2f ms, total %.2f ms = %.2f GB/s\n", mode, sz >> 20,
          (t1 - t0) * 1e3, (t2 - t0) * 1e3, 4.0 * sz / (t2 - t0) / 1e9, (t4 - t3) * 1e3, (t5 - t3) * 1e3, 4.0 * sz / (t5 - t3) / 1e9);
      }
      CK(cudaFree(dst));
    }
    int ok = 1; write(c2p[1], &ok, sizeof ok); return 0;
  }
  CK(cudaSetDevice(0)); void* buf; CK(cudaMalloc(&buf, big)); CK(cudaMemset(buf, 1, big)); cudaIpcMemHandle_t h; CK(cudaIpcGetMemHandle(&h, buf));
  write(p2c[1], &h, sizeof h); int ok = 0; read(c2p[0], &ok, sizeof ok); waitpid(pid, NULL, 0); return 0;
}
