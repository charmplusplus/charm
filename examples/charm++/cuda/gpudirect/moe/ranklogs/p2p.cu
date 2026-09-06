#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int n = 0; cudaGetDeviceCount(&n); const size_t sz = (size_t)128 << 20;
  void* buf[8]; void* host = NULL;
  for (int d = 0; d < n; d++) { cudaSetDevice(d); cudaMalloc(&buf[d], sz);
    for (int e = 0; e < n; e++) if (e != d) { int can = 0; cudaDeviceCanAccessPeer(&can, d, e); if (can) cudaDeviceEnablePeerAccess(e, 0); else printf("no P2P %d->%d\n", d, e); } }
  cudaSetDevice(0); cudaMallocHost(&host, sz);
  printf("128 MB copies, GB/s (row = src device, col = dst device; diag = D2D same device)\n");
  for (int s = 0; s < n; s++) { cudaSetDevice(s); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    printf("src %d:", s);
    for (int d = 0; d < n; d++) {
      cudaMemcpyPeerAsync(buf[d], d, buf[s], s, sz, 0); cudaDeviceSynchronize();
      cudaEventRecord(a, 0); for (int r = 0; r < 4; r++) cudaMemcpyPeerAsync(buf[d], d, buf[s], s, sz, 0); cudaEventRecord(b, 0); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); printf(" %6.2f", 4.0 * sz / (ms * 1e-3) / 1e9); }
    cudaEventRecord(a, 0); for (int r = 0; r < 4; r++) cudaMemcpyAsync(host, buf[s], sz, cudaMemcpyDeviceToHost, 0); cudaEventRecord(b, 0); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    printf("   D2H %6.2f", 4.0 * sz / (ms * 1e-3) / 1e9);
    cudaEventRecord(a, 0); for (int r = 0; r < 4; r++) cudaMemcpyAsync(buf[s], host, sz, cudaMemcpyHostToDevice, 0); cudaEventRecord(b, 0); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    printf(" H2D %6.2f\n", 4.0 * sz / (ms * 1e-3) / 1e9); }
  // two concurrent peer copies from different sources into different destinations
  if (n >= 4) { cudaSetDevice(0); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaStream_t s0; cudaStreamCreate(&s0); cudaSetDevice(2); cudaStream_t s2; cudaStreamCreate(&s2); cudaSetDevice(0);
    cudaEventRecord(a, s0); for (int r = 0; r < 4; r++) { cudaMemcpyPeerAsync(buf[1], 1, buf[0], 0, sz, s0); cudaMemcpyPeerAsync(buf[3], 3, buf[2], 2, sz, s2); } cudaEventRecord(b, s0); cudaSetDevice(2); cudaStreamSynchronize(s2); cudaSetDevice(0); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    printf("concurrent 0->1 and 2->3: %.2f GB/s aggregate\n", 8.0 * sz / (ms * 1e-3) / 1e9); }
  return 0; }
