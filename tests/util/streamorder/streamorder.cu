// Does host launch ORDER across CUDA streams change wall time?
//
// CUDA streams are software queues; the driver round-robins them onto a fixed
// number of hardware channels (CUDA_DEVICE_MAX_CONNECTIONS, default 8, max 32).
// Two streams that land on the same channel share one in-order front-end FIFO,
// so an item that has to WAIT at the head of that FIFO blocks every item behind
// it -- including items from the other stream. Two kernels from the SAME stream
// are dependent by stream semantics, so issuing
//
//     A1 A2 B1 B2   (depth-first)  puts A2's wait ahead of B1  -> ~4T
//     A1 B1 A2 B2   (breadth-first) puts nothing waiting ahead  -> ~2T
//
// if and only if the two streams share a channel. This program measures that,
// and three things the runtime design needs:
//
//   map      which stream indices actually alias onto stream 0's channel
//   order    depth vs breadth, swept over kernel duration and grid size
//   queues   the same with Q>2 queues on one channel (round-robin at depth)
//   bubble   the cost of instead holding A2 on the host until A1 completes
//            (the price of software interleaving when the partner is late)
//
//   nvcc -O2 -arch=sm_86 -o streamorder streamorder.cu
//   CUDA_DEVICE_MAX_CONNECTIONS=8 ./streamorder
//
// Every number is a median of REPS timed launch bursts, reported both in
// microseconds and as a ratio to T1 -- one kernel of the same shape run alone.
// Ratios are what matter: they are immune to clock boost drift.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(call)                                                             \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      printf("CUDA %s:%d %s -> %s\n", __FILE__, __LINE__, #call,             \
             cudaGetErrorString(_e));                                        \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// Busy-wait for `cycles` of the SM clock. clock64() is a volatile intrinsic, so
// the loop survives -O2; the store is unreachable and only anchors `now`.
__global__ void spin(long long cycles, unsigned long long* sink) {
  const long long start = clock64();
  long long now;
  do { now = clock64(); } while (now - start < cycles);
  if (now == start - 1) *sink = (unsigned long long)now;
}

static unsigned long long* d_sink = nullptr;
static int REPS = 21;

struct Cfg {
  long long cycles;
  int blocks;
  int threads;
  const char* grid;   // label
};

static inline void launchOn(cudaStream_t s, const Cfg& c) {
  spin<<<c.blocks, c.threads, 0, s>>>(c.cycles, d_sink);
}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch())
      .count();
}

// Median wall time of `f` (a burst of launches), device-synchronized either
// side so the burst is measured end to end.
template <typename F>
static double timeMedian(F&& f) {
  for (int i = 0; i < 3; i++) { f(); CK(cudaDeviceSynchronize()); }
  std::vector<double> v;
  v.reserve(REPS);
  for (int i = 0; i < REPS; i++) {
    CK(cudaDeviceSynchronize());
    const double t0 = now_us();
    f();
    CK(cudaDeviceSynchronize());
    v.push_back(now_us() - t0);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Launch orders. perQ items per queue, every item on its queue's own stream.
static void depthFirst(const std::vector<cudaStream_t>& ss, const Cfg& c,
                       int perQ) {
  for (cudaStream_t s : ss)
    for (int i = 0; i < perQ; i++) launchOn(s, c);
}
static void breadthFirst(const std::vector<cudaStream_t>& ss, const Cfg& c,
                         int perQ) {
  for (int i = 0; i < perQ; i++)
    for (cudaStream_t s : ss) launchOn(s, c);
}

int main(int argc, char** argv) {
  const char* envc = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  const int C = envc ? atoi(envc) : 8;
  if (getenv("STREAMORDER_REPS")) REPS = atoi(getenv("STREAMORDER_REPS"));
  const int QMAX = 64;

  int dev = 0;
  if (getenv("STREAMORDER_DEV")) dev = atoi(getenv("STREAMORDER_DEV"));
  CK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  printf("# device %d %s: %d SMs, cc %d.%d\n", dev, p.name,
         p.multiProcessorCount, p.major, p.minor);
  printf("# CUDA_DEVICE_MAX_CONNECTIONS=%s (using C=%d), reps=%d\n",
         envc ? envc : "(unset, driver default)", C, REPS);

  CK(cudaMalloc(&d_sink, sizeof(unsigned long long)));

  // One stream per (queue, channel) slot we might want, created back to back so
  // the driver's round-robin is over exactly this array: index k is expected to
  // land on channel k % C.
  const int nStreams = QMAX * C + 2;
  std::vector<cudaStream_t> S(nStreams);
  for (int i = 0; i < nStreams; i++)
    CK(cudaStreamCreateWithFlags(&S[i], cudaStreamNonBlocking));
  printf("# %d streams created\n", nStreams);

  // Kernel shapes. `small` leaves the GPU almost empty, so concurrency is
  // limited by channels alone; `full` puts one 1024-thread block on every SM,
  // so a second kernel cannot start until SMs free regardless of channels.
  const Cfg grids[2] = {{0, 4, 128, "small"},
                        {0, p.multiProcessorCount, 1024, "full"}};
  const double durs_us[] = {5, 10, 20, 50, 100, 200, 500, 1000};
  const int nd = sizeof(durs_us) / sizeof(durs_us[0]);
  // Calibrate the SM clock by timing a long spin rather than trusting a
  // nominal rate: cudaDeviceProp::clockRate is gone in CUDA 13, and the boost
  // clock under load is what actually sets a kernel's duration.
  double cyc_per_us = 0;
  {
    const long long cal = 200LL * 1000 * 1000;
    CK(cudaDeviceSynchronize());
    const double t0 = now_us();
    spin<<<1, 32, 0, S[0]>>>(cal, d_sink);
    CK(cudaDeviceSynchronize());
    cyc_per_us = cal / (now_us() - t0);
    printf("# SM clock calibrated at %.0f MHz\n", cyc_per_us);
  }

  // ---- map: which stream index shares stream 0's channel? -----------------
  // Depth-first over a pair is ~4T when the two streams share a channel and
  // ~2T when they do not, so this prints the aliasing map directly.
  {
    Cfg c = grids[0];
    c.cycles = (long long)(200.0 * cyc_per_us);
    std::vector<cudaStream_t> one{S[0]};
    const double t1 = timeMedian([&] { depthFirst(one, c, 1); });
    printf("\n# map: depth-first on streams {0,j}, ~2*T1 = distinct channel, "
           "~4*T1 = shared. T1=%.1f us\n", t1);
    printf("tag\tj\tus\tratio\tverdict\n");
    for (int j = 1; j <= 2 * C + 1; j++) {
      std::vector<cudaStream_t> pr{S[0], S[j]};
      const double t = timeMedian([&] { depthFirst(pr, c, 2); });
      const double r = t / t1;
      printf("map\t%d\t%.1f\t%.2f\t%s\n", j, t, r,
             r > 3.0 ? "SHARED" : (r < 2.6 ? "distinct" : "?"));
      fflush(stdout);
    }
  }

  // ---- order: depth vs breadth, aliased vs distinct, by duration and grid --
  printf("\n# order: 2 queues x 2 dependent kernels. pairA = streams {0,C}, "
         "pairB = streams {0,1}. At C=1 both pairs share the only channel.\n");
  printf("# ratios are to T1, one kernel alone. 4.0 = fully serial, "
         "2.0 = both pairs overlapped.\n");
  printf("tag\tgrid\tus_target\tT1_us\tA_depth\tA_breadth\tB_depth"
         "\tB_breadth\tAd_r\tAb_r\tBd_r\tBb_r\n");
  for (int g = 0; g < 2; g++) {
    for (int i = 0; i < nd; i++) {
      Cfg c = grids[g];
      c.cycles = (long long)(durs_us[i] * cyc_per_us);
      std::vector<cudaStream_t> one{S[0]};
      std::vector<cudaStream_t> alias{S[0], S[C]};
      std::vector<cudaStream_t> dist{S[0], S[1]};
      const double t1 = timeMedian([&] { depthFirst(one, c, 1); });
      const double ad = timeMedian([&] { depthFirst(alias, c, 2); });
      const double ab = timeMedian([&] { breadthFirst(alias, c, 2); });
      const double dd = timeMedian([&] { depthFirst(dist, c, 2); });
      const double db = timeMedian([&] { breadthFirst(dist, c, 2); });
      printf("order\t%s\t%.0f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.2f\t%.2f\t%.2f"
             "\t%.2f\n",
             c.grid, durs_us[i], t1, ad, ab, dd, db, ad / t1, ab / t1, dd / t1,
             db / t1);
      fflush(stdout);
    }
  }

  // ---- queues: Q queues all on ONE channel, round-robin at depth ----------
  // Streams 0, C, 2C, ... all alias to channel 0. Depth-first should cost
  // ~2*Q*T; breadth-first ~2*T until the SMs run out.
  printf("\n# queues: Q queues x 2 dependent kernels, ALL on channel 0 "
         "(streams 0, C, 2C, ...)\n");
  printf("tag\tgrid\tus_target\tQ\tT1_us\tdepth_us\tbreadth_us\tdepth_r"
         "\tbreadth_r\n");
  for (int g = 0; g < 2; g++) {
    for (double d : {20.0, 200.0}) {
      Cfg c = grids[g];
      c.cycles = (long long)(d * cyc_per_us);
      std::vector<cudaStream_t> one{S[0]};
      const double t1 = timeMedian([&] { depthFirst(one, c, 1); });
      for (int Q : {2, 4, 8, 16, 32, 64}) {
        std::vector<cudaStream_t> ss;
        for (int q = 0; q < Q; q++) ss.push_back(S[q * C]);
        const double td = timeMedian([&] { depthFirst(ss, c, 2); });
        const double tb = timeMedian([&] { breadthFirst(ss, c, 2); });
        printf("queues\t%s\t%.0f\t%d\t%.1f\t%.1f\t%.1f\t%.2f\t%.2f\n", c.grid,
               d, Q, t1, td, tb, td / t1, tb / t1);
        fflush(stdout);
      }
    }
  }

  // ---- indep: Q INDEPENDENT kernels, one per stream, no dependencies ------
  // The control the first run lacked. At C=1 every stream is on the one
  // channel. If Q kernels still cost ~1*T the channel is not a concurrency
  // limiter at all, so no launch order can matter and the order arms below are
  // vacuous; if they cost ~Q*T the channel serializes and the order arms mean
  // something. This distinguishes "connections ignored" from "connections
  // honoured but no head-of-line penalty".
  printf("\n# indep: Q independent kernels, one per stream, no intra-queue "
         "dependency. ~1.0 = the channel is not a limiter, ~Q = it serializes\n");
  printf("tag\tgrid\tus_target\tQ\tT1_us\ttotal_us\tratio\n");
  for (int g = 0; g < 2; g++) {
    for (double d : {20.0, 200.0}) {
      Cfg c = grids[g];
      c.cycles = (long long)(d * cyc_per_us);
      std::vector<cudaStream_t> one{S[0]};
      const double t1 = timeMedian([&] { depthFirst(one, c, 1); });
      for (int Q : {2, 4, 8, 16, 32, 64}) {
        std::vector<cudaStream_t> ss;
        for (int q = 0; q < Q; q++) ss.push_back(S[q]);
        const double t = timeMedian([&] { breadthFirst(ss, c, 1); });
        printf("indep\t%s\t%.0f\t%d\t%.1f\t%.1f\t%.2f\n", c.grid, d, Q, t1, t,
               t / t1);
        fflush(stdout);
      }
    }
  }

  // ---- bubble: deferring the dependent kernel to the host, two ways ------
  // back2back  A1,A2 on one stream: the front end resolves the dependency, ~0 gap.
  // eventgap   A2 launched after the host polls cudaEventQuery -- a driver call
  //            on every poll.
  // flaggap    A2 launched after the host polls a PINNED HOST WORD written by a
  //            stream-ordered cuMemsetD32Async. That is the completion path this
  //            runtime already uses (+gpuflagpoll, hapiFlagIssue); polling it is
  //            a plain load with no driver call, so it should be the cheaper of
  //            the two and is the honest floor for a software work queue.
  printf("\n# bubble: deferring the dependent kernel to a host poll, two ways\n");
  printf("tag\tgrid\tus_target\tT1_us\tback2back_us\tevent_us\tflag_us"
         "\tevent_bubble_us\tflag_bubble_us\n");
  cudaEvent_t ev;
  CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  unsigned int* h_flag = nullptr;
  CK(cudaHostAlloc((void**)&h_flag, sizeof(unsigned int), cudaHostAllocMapped));
  *h_flag = 0;
  void* p_flag = nullptr;
  CK(cudaHostGetDevicePointer(&p_flag, h_flag, 0));
  const CUdeviceptr d_flag = (CUdeviceptr)p_flag;
  volatile unsigned int* vflag = h_flag;
  unsigned int seq = 0;
  {  // probe once: if the driver-API write is unavailable, say so, never hang
    const CUresult r = cuMemsetD32Async(d_flag, 0xABCDu, 1, (CUstream)S[0]);
    CK(cudaStreamSynchronize(S[0]));
    if (r != CUDA_SUCCESS || *vflag != 0xABCDu) {
      printf("# flag path unavailable (cuMemsetD32Async -> %d, flag reads %u); "
             "reporting the event path only\n", (int)r, *vflag);
      vflag = nullptr;
    }
  }
  for (int g = 0; g < 2; g++) {
    for (int i = 0; i < nd; i++) {
      Cfg c = grids[g];
      c.cycles = (long long)(durs_us[i] * cyc_per_us);
      std::vector<cudaStream_t> one{S[0]};
      const double t1 = timeMedian([&] { depthFirst(one, c, 1); });
      const double b2b = timeMedian([&] { depthFirst(one, c, 2); });
      const double eg = timeMedian([&] {
        launchOn(S[0], c);
        CK(cudaEventRecord(ev, S[0]));
        while (cudaEventQuery(ev) == cudaErrorNotReady) {}
        launchOn(S[0], c);
      });
      double fg = 0;
      if (vflag) {
        fg = timeMedian([&] {
          launchOn(S[0], c);
          ++seq;
          cuMemsetD32Async(d_flag, seq, 1, (CUstream)S[0]);
          while (*vflag != seq) {}
          launchOn(S[0], c);
        });
      }
      printf("bubble\t%s\t%.0f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n", c.grid,
             durs_us[i], t1, b2b, eg, fg, eg - b2b, vflag ? fg - b2b : 0.0);
      fflush(stdout);
    }
  }

  printf("\n# done\n");
  return 0;
}
