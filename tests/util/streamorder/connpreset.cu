// Does a CUDA_DEVICE_MAX_CONNECTIONS value that setenv() sets INSIDE the process
// reach the driver, and how late may it be set? The Charm++ runtime sets 32
// from charm_main (hapiPresetEnvironment, 2026-10-20), before ConverseInit,
// because the machine layer's LCI calls cuInit(0) during ConverseInit. This
// program measures the effective queue count after one of four placements:
//   none     no setenv: whatever the environment says (driver default 8)
//   before   setenv 32, then cuInit(0)                   [charm_main]
//   after    cuInit(0), then setenv 32                   [hapiInit, had LCI gone first]
//   context  cuInit(0), cudaFree(0) (primary context), then setenv 32   [too late by construction]
// The effective count is read off the head-of-line penalty (tests/util/streamorder,
// "queues"): Q streams x 2 dependent 200 us kernels, issued depth-first, cost
// ~(Q/C + 1) x T on C queues (driver 595.71.05 on 2026-10-20 assigns queues
// dynamically, so a two-stream pair never shares one; a static index map does
// not exist any more). implied C = Q / (ratio - 1), reported for Q = 64.
//   nvcc -O2 -arch=sm_86 -o connpreset connpreset.cu -lcuda
//   env -u CUDA_DEVICE_MAX_CONNECTIONS ./connpreset before
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call)                                                              \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      printf("CUDA %s:%d %s -> %s\n", __FILE__, __LINE__, #call,              \
             cudaGetErrorString(_e));                                         \
      exit(1);                                                                \
    }                                                                         \
  } while (0)
#define CU(call)                                                              \
  do {                                                                        \
    CUresult _r = (call);                                                     \
    if (_r != CUDA_SUCCESS) {                                                 \
      printf("CU %s:%d %s -> %d\n", __FILE__, __LINE__, #call, (int)_r);      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

__global__ void spin(long long cycles, unsigned long long* sink) {
  const long long start = clock64();
  long long now;
  do { now = clock64(); } while (now - start < cycles);
  if (now == start - 1) *sink = (unsigned long long)now;
}

static unsigned long long* d_sink = nullptr;

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// Median over reps of: `perQ` dependent kernels on each stream, issued
// depth-first (all of stream 0's, then all of stream 1's), then a device sync.
static double depthFirst(const std::vector<cudaStream_t>& ss, long long cycles, int perQ, int reps) {
  std::vector<double> t;
  for (int r = 0; r < reps; r++) {
    CK(cudaDeviceSynchronize());
    const double t0 = now_us();
    for (cudaStream_t s : ss)
      for (int i = 0; i < perQ; i++) spin<<<4, 128, 0, s>>>(cycles, d_sink);
    CK(cudaDeviceSynchronize());
    t.push_back(now_us() - t0);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "none";
  const char* pre = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  printf("# mode %s; CUDA_DEVICE_MAX_CONNECTIONS at entry: %s\n", mode, pre ? pre : "(unset)");
  if (!strcmp(mode, "before")) {
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
    CU(cuInit(0));
  } else if (!strcmp(mode, "after")) {
    CU(cuInit(0));
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
  } else if (!strcmp(mode, "context")) {
    CU(cuInit(0));
    CK(cudaFree(0));
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
  } else if (strcmp(mode, "none") != 0) {
    printf("usage: connpreset none|before|after|context\n");
    return 2;
  }
  const char* post = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  printf("# CUDA_DEVICE_MAX_CONNECTIONS now: %s\n", post ? post : "(unset)");

  int dev = 0;
  if (getenv("STREAMORDER_DEV")) dev = atoi(getenv("STREAMORDER_DEV"));
  CK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  printf("# device %d %s\n", dev, p.name);
  CK(cudaMalloc(&d_sink, sizeof(unsigned long long)));

  const int QMAX = 64;
  std::vector<cudaStream_t> S(QMAX);
  for (int i = 0; i < QMAX; i++) CK(cudaStreamCreateWithFlags(&S[i], cudaStreamNonBlocking));

  double cyc_per_us;
  {
    const long long cal = 200LL * 1000 * 1000;
    CK(cudaDeviceSynchronize());
    const double t0 = now_us();
    spin<<<1, 32, 0, S[0]>>>(cal, d_sink);
    CK(cudaDeviceSynchronize());
    cyc_per_us = cal / (now_us() - t0);
  }
  const long long cycles = (long long)(200.0 * cyc_per_us);   // 200 us kernels
  const int reps = 11;
  std::vector<cudaStream_t> one{S[0]};
  const double t1 = depthFirst(one, cycles, 1, reps);
  printf("# T1=%.1f us; Q streams x 2 dependent kernels depth-first: ~(Q/C + 1) x T1 on C queues\n", t1);
  printf("tag\tQ\tus\tratio\timplied_C\n");
  double implied = 0;
  for (int Q = 8; Q <= QMAX; Q *= 2) {
    std::vector<cudaStream_t> qs(S.begin(), S.begin() + Q);
    const double t = depthFirst(qs, cycles, 2, reps);
    const double r = t / t1;
    const double c = r > 1.05 ? Q / (r - 1.0) : 0;
    printf("queues\t%d\t%.1f\t%.2f\t%.1f\n", Q, t, r, c);
    if (Q == QMAX) implied = c;
    fflush(stdout);
  }
  const int effective = implied >= 24 ? 32 : implied <= 12 ? 8 : (int)(implied + 0.5);
  printf("RESULT mode=%s effective_queues=%d (implied %.1f from Q=64)\n", mode, effective, implied);
  return 0;
}
