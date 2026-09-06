#include <cstdio>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  int n = 0; cudaGetDeviceCount(&n); const size_t sz = (size_t)32 << 20; const int reps = 4;
  std::vector<void*> src(n), dst(n * n); std::vector<cudaStream_t> st(n * n);
  for (int d = 0; d < n; d++) { cudaSetDevice(d); cudaMalloc(&src[d], sz); for (int e = 0; e < n; e++) { if (e != d) cudaDeviceEnablePeerAccess(e, 0); cudaMalloc(&dst[d * n + e], sz); cudaStreamCreateWithFlags(&st[d * n + e], cudaStreamNonBlocking); } }
  auto run = [&](const char* name, std::vector<std::pair<int,int>> flows) {  // (dst, src) pulls, dst issues
    for (auto f : flows) { cudaSetDevice(f.first); cudaMemcpyPeerAsync(dst[f.first * n + f.second], f.first, src[f.second], f.second, sz, st[f.first * n + f.second]); }
    for (int d = 0; d < n; d++) { cudaSetDevice(d); cudaDeviceSynchronize(); }
    double t0 = now();
    for (int r = 0; r < reps; r++) for (auto f : flows) { cudaSetDevice(f.first); cudaMemcpyPeerAsync(dst[f.first * n + f.second], f.first, src[f.second], f.second, sz, st[f.first * n + f.second]); }
    for (int d = 0; d < n; d++) { cudaSetDevice(d); cudaDeviceSynchronize(); }
    double dt = now() - t0; double bytes = (double)reps * flows.size() * sz;
    printf("%-34s %2zu flows: aggregate %6.1f GB/s, per flow %5.1f GB/s\n", name, flows.size(), bytes / dt / 1e9, bytes / dt / 1e9 / flows.size());
  };
  run("single 1<-0", {{1,0}});
  run("fan-in 0<-1,2,3", {{0,1},{0,2},{0,3}});
  run("fan-out 1,2,3<-0", {{1,0},{2,0},{3,0}});
  std::vector<std::pair<int,int>> all; for (int d = 0; d < n; d++) for (int s = 0; s < n; s++) if (d != s) all.push_back({d, s});
  run("all-to-all (12 pulls)", all);
  return 0; }
