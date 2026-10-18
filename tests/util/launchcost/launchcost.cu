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
// devices > 1: thread t drives device dev + (t % devices) -- several GPUs in ONE process, the
// (1,4,4) layout's shape, against the same thread count on one device and against separate
// processes (run several copies with CUDA_VISIBLE_DEVICES).
// cupti: 0 off, 1 activity records only (RUNTIME + CONCURRENT_KERNEL), 2 records + the per-call external
// correlation push/pop (hapi's per-entry attribution).
static Result run(int threads, int streamsPerThread, Op op, int cupti, int perBurst, int bursts,
                  int baseCore, int dev, bool background = false, int devices = 1) {
  static std::atomic<bool> bgStop; bgStop.store(false);
  std::vector<double> busy(threads, 0.0);
  std::atomic<int> ready{0}; std::atomic<bool> go{false};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) pool.emplace_back([&, t]() {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(baseCore + t, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    CK(cudaSetDevice(dev + (devices > 1 ? t % devices : 0)));
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
        if (cupti == 2) cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, (uint64_t)(t * 1000003 + i));
        if (op == LAUNCH) nop<<<1, 1, 0, s>>>(sink);
        else if (op == MEMCPY) cudaMemcpyAsync(b, a, 24 * 1024, cudaMemcpyDeviceToDevice, s);
        else if (op == FLAG) cuMemsetD32Async((CUdeviceptr)p_flag, (unsigned)i + 1, 1, (CUstream)s);
        else cudaEventRecord(ev, s);
        if (cupti == 2) { uint64_t id; cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id); }
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
  const int devices = argc > 4 ? atoi(argv[4]) : 1;   // > 1: the multi-device shapes (one process, D GPUs)
  const int baseCore = getenv("LAUNCHCOST_BASE_CORE") ? atoi(getenv("LAUNCHCOST_BASE_CORE")) : sched_getcpu();
  for (int d = 0; d < devices; d++) { CK(cudaSetDevice(dev + d)); CK(cudaFree(0)); }   // contexts up front
  CK(cudaSetDevice(dev));
  printf("# device %d (%d device(s)), %d calls x %d bursts per thread, cores from %d, CUDA_DEVICE_MAX_CONNECTIONS=%s\n",
         dev, devices, perBurst, bursts, baseCore, getenv("CUDA_DEVICE_MAX_CONNECTIONS") ? getenv("CUDA_DEVICE_MAX_CONNECTIONS") : "(default 8)");
  printf("%-8s %-7s %-34s %12s %16s\n", "cupti", "op", "shape", "us/call", "calls/s (all)");
  const int passes = devices > 1 ? 3 : 2;   // multi-device: off, records only, records + push/pop
  for (int pass = 0; pass < passes; pass++) {
    const int cupti = devices > 1 ? pass : (pass == 1 ? 2 : 0);
    if (pass == 1) cuptiOn();
    for (Op op : {LAUNCH, MEMCPY, FLAG, EVENT}) {
      struct Shape { int t, s, d; const char* name; };
      const Shape one[] = {
        {1, 1, 1, "1 thread x 1 stream"}, {1, 8, 1, "1 thread x 8 streams"},
        {2, 1, 1, "2 threads x 1 stream"}, {4, 1, 1, "4 threads x 1 stream"}, {8, 1, 1, "8 threads x 1 stream"}};
      const Shape many[] = {   // the process/GPU split: 4 PEs per device, 1 or D devices in the process
        {4, 1, 1, "4 threads x 1 device"}, {4 * devices, 1, 1, "4D threads x 1 device"},
        {devices, 1, devices, "D threads x D devices (1 each)"}, {4 * devices, 1, devices, "4D threads x D devices (4 each)"}};
      const Shape* shapes = devices > 1 ? many : one;
      const int nShapes = devices > 1 ? 4 : 5;
      for (int i = 0; i < nShapes; i++) {
        const Shape& sh = shapes[i];
        const Result r = run(sh.t, sh.s, op, cupti, perBurst, bursts, baseCore, dev, false, sh.d);
        printf("%-8s %-7s %-34s %12.2f %16.0f\n", cupti == 0 ? "off" : cupti == 1 ? "rec" : "on", op == LAUNCH ? "launch" : op == MEMCPY ? "memcpy" : op == FLAG ? "flag" : "event",
               sh.name, r.us_per_call, r.calls_per_s);
        fflush(stdout);
      }
    }
  }
  if (devices > 1) { cuptiActivityFlushAll(0); printf("# cupti bytes delivered: %llu\n", g_records.load()); return 0; }
  // CUPTI is on from the second pass: one measured caller among K background callers.
  printf("# measured thread 0 only; the others issue launch + event record in a loop, untimed\n");
  for (Op op : {MEMCPY, FLAG})
    for (int bg : {0, 1, 2, 4, 7}) {
      const Result r = run(1 + bg, 1, op, 2, perBurst, bursts, baseCore, dev, bg > 0);
      char name[64]; snprintf(name, sizeof(name), "1 timed + %d background", bg);
      printf("%-8s %-7s %-22s %12.2f %16.0f\n", "on", op == MEMCPY ? "memcpy" : "flag", name, r.us_per_call, r.calls_per_s);
      fflush(stdout);
    }
  cuptiActivityFlushAll(0);
  printf("# cupti bytes delivered: %llu\n", g_records.load());
  return 0;
}
