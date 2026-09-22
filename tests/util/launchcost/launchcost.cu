// launchcost: what does one CUDA driver call cost the HOST, from one thread and
// from T threads at once on the same GPU, with CUPTI activity tracing off and on?
//
// The go/no-go number for a per-GPU submitter thread (one thread issuing every
// driver call a process makes, PEs feeding it through queues). leanmd at 960
// atoms/cell measured 9.2 us per cudaLaunchKernel and 9.3 us per cudaMemcpyAsync
// with 8 PEs calling at once under the runtime's CUPTI tracing (2026-09-21). If a
// lone thread pays much less, the difference is contention a submitter removes;
// and the lone-thread rate says whether ONE core can carry a GPU's calls at all.
//
//   launchcost [device=0] [calls_per_burst=2000] [bursts=20]
//
// Each config: T threads (pinned to cores base..base+T-1), each issuing bursts of
// trivial kernel launches (or 24 KB D2D cudaMemcpyAsync) on its OWN stream; only
// the issuing loop is timed, the stream is drained between bursts. With CUPTI on,
// the runtime's activity kinds are enabled and every call is wrapped in an
// external-correlation push/pop, as hapi does for load attribution.
// "1 thread x 8 streams" is the submitter's shape: same streams, one caller.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void nop(int* sink) { if (sink == nullptr) return; }

static void CUPTIAPI bufRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *size = 5 * 1024 * 1024; *buffer = (uint8_t*)malloc(*size); *maxNumRecords = 0;
}
static std::atomic<unsigned long long> g_records{0};
static void CUPTIAPI bufCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  g_records += validSize;   // consumed, not parsed: parsing is the LB step's cost, not the call's
  free(buffer);
}
static void cuptiOn() {
  cuptiActivityRegisterCallbacks(bufRequested, bufCompleted);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
}

enum Op { LAUNCH, MEMCPY, FLAG, EVENT };
struct Result { double us_per_call; double calls_per_s; };

// background > 0: only thread 0 is measured; the other `threads-1` are untimed callers issuing a
// launch and an event record in a loop until thread 0 finishes -- the submitter's real situation in
// stage 1, where the PEs still call the driver for their own kernels and event records.
static Result run(int threads, int streamsPerThread, Op op, bool cupti, int perBurst, int bursts,
                  int baseCore, int dev, bool background = false) {
  static std::atomic<bool> bgStop; bgStop.store(false);
  std::vector<double> busy(threads, 0.0);
  std::atomic<int> ready{0}; std::atomic<bool> go{false};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) pool.emplace_back([&, t]() {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(baseCore + t, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    CK(cudaSetDevice(dev));
    std::vector<cudaStream_t> S(streamsPerThread);
    for (auto& s : S) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    int* sink = nullptr; char* a = nullptr; char* b = nullptr;
    CK(cudaMalloc(&sink, sizeof(int)));
    CK(cudaMalloc(&a, 24 * 1024)); CK(cudaMalloc(&b, 24 * 1024));
    // The runtime's completion flag: a stream-ordered 32-bit write into pinned,
    // mapped host memory (cuMemsetD32Async), and its cross-process event record.
    unsigned int* h_flag = nullptr; void* p_flag = nullptr;
    CK(cudaHostAlloc((void**)&h_flag, 64 * sizeof(unsigned int), cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&p_flag, h_flag, 0));
    cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    for (auto& s : S) { nop<<<1, 1, 0, s>>>(sink); CK(cudaStreamSynchronize(s)); }   // warm
    ready++; while (!go.load()) {}
    if (background && t > 0) {
      unsigned long n = 0;
      while (!bgStop.load(std::memory_order_relaxed)) {
        nop<<<1, 1, 0, S[0]>>>(sink); cudaEventRecord(ev, S[0]);
        if ((++n & 1023) == 0) cudaStreamSynchronize(S[0]);
      }
      cudaStreamSynchronize(S[0]);
      for (auto& s : S) cudaStreamDestroy(s);
      return;
    }
    double mine = 0.0;
    for (int r = 0; r < bursts; r++) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < perBurst; i++) {
        cudaStream_t s = S[i % streamsPerThread];
        if (cupti) cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, (uint64_t)(t * 1000003 + i));
        if (op == LAUNCH) nop<<<1, 1, 0, s>>>(sink);
        else if (op == MEMCPY) cudaMemcpyAsync(b, a, 24 * 1024, cudaMemcpyDeviceToDevice, s);
        else if (op == FLAG) cuMemsetD32Async((CUdeviceptr)p_flag, (unsigned)i + 1, 1, (CUstream)s);
        else cudaEventRecord(ev, s);
        if (cupti) { uint64_t id; cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id); }
      }
      mine += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      for (auto& s : S) CK(cudaStreamSynchronize(s));
    }
    busy[t] = mine;
    if (background) bgStop.store(true);
    for (auto& s : S) cudaStreamDestroy(s);
    cudaFree(sink); cudaFree(a); cudaFree(b); cudaFreeHost(h_flag); cudaEventDestroy(ev);
  });
  while (ready.load() < threads) {}
  go.store(true);
  for (auto& th : pool) th.join();
  double sum = 0.0, mx = 0.0;
  for (double x : busy) { sum += x; if (x > mx) mx = x; }
  const double calls = (double)perBurst * bursts;
  Result res;
  if (background) { res.us_per_call = 1e6 * busy[0] / calls; res.calls_per_s = calls / busy[0]; return res; }
  res.us_per_call = 1e6 * (sum / threads) / calls;            // mean per-thread cost of a call
  res.calls_per_s = threads * calls / mx;                     // aggregate issue rate while issuing
  return res;
}

int main(int argc, char** argv) {
  const int dev = argc > 1 ? atoi(argv[1]) : 0;
  const int perBurst = argc > 2 ? atoi(argv[2]) : 2000;
  const int bursts = argc > 3 ? atoi(argv[3]) : 20;
  const int baseCore = getenv("LAUNCHCOST_BASE_CORE") ? atoi(getenv("LAUNCHCOST_BASE_CORE")) : sched_getcpu();
  CK(cudaSetDevice(dev)); CK(cudaFree(0));
  printf("# device %d, %d calls x %d bursts per thread, cores from %d, CUDA_DEVICE_MAX_CONNECTIONS=%s\n",
         dev, perBurst, bursts, baseCore, getenv("CUDA_DEVICE_MAX_CONNECTIONS") ? getenv("CUDA_DEVICE_MAX_CONNECTIONS") : "(default 8)");
  printf("%-8s %-7s %-22s %12s %16s\n", "cupti", "op", "shape", "us/call", "calls/s (all)");
  for (int pass = 0; pass < 2; pass++) {
    const bool cupti = pass == 1;
    if (cupti) cuptiOn();
    for (Op op : {LAUNCH, MEMCPY, FLAG, EVENT}) {
      struct { int t, s; const char* name; } shapes[] = {
        {1, 1, "1 thread x 1 stream"}, {1, 8, "1 thread x 8 streams"},
        {2, 1, "2 threads x 1 stream"}, {4, 1, "4 threads x 1 stream"}, {8, 1, "8 threads x 1 stream"}};
      for (auto& sh : shapes) {
        const Result r = run(sh.t, sh.s, op, cupti, perBurst, bursts, baseCore, dev);
        printf("%-8s %-7s %-22s %12.2f %16.0f\n", cupti ? "on" : "off", op == LAUNCH ? "launch" : op == MEMCPY ? "memcpy" : op == FLAG ? "flag" : "event",
               sh.name, r.us_per_call, r.calls_per_s);
        fflush(stdout);
      }
    }
  }
  // CUPTI is on from the second pass: one measured caller among K background callers.
  printf("# measured thread 0 only; the others issue launch + event record in a loop, untimed\n");
  for (Op op : {MEMCPY, FLAG})
    for (int bg : {0, 1, 2, 4, 7}) {
      const Result r = run(1 + bg, 1, op, true, perBurst, bursts, baseCore, dev, bg > 0);
      char name[64]; snprintf(name, sizeof(name), "1 timed + %d background", bg);
      printf("%-8s %-7s %-22s %12.2f %16.0f\n", "on", op == MEMCPY ? "memcpy" : "flag", name, r.us_per_call, r.calls_per_s);
      fflush(stdout);
    }
  cuptiActivityFlushAll(0);
  printf("# cupti bytes delivered: %llu\n", g_records.load());
  return 0;
}
