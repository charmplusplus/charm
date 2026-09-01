// Transfer-cost calibration for DiffusionLB's cost model.
//
// Measures alpha (per message) and beta (per byte) for each of the four tiers a
// device transfer can resolve to, and writes them to a table that
// DiffusionLB reads at runtime via +LBCostConfig.
//
//   intra_process   two PEs in one process        device-to-device memcpy
//   ipc_same_gpu    two processes, one device     CUDA IPC
//   ipc_cross_gpu   two processes, one host       CUDA IPC across devices
//   inter_node      two hosts                     network RDMA
//
// Two things this is careful about, because getting either wrong produces
// numbers that look plausible and are badly misleading:
//
// 1. PE-to-GPU ratio. On a shared device a large part of the measured latency is
//    contention between the processes sharing it, not the transport. Calibrate
//    at the ratio the application will run at (--pes-per-gpu on the launch, the
//    same +pemap as production) or the shared-device tiers absorb a contention
//    cost that the model will then charge to every IPC byte.
//
// 2. Latency is not the same as marginal cost. These numbers are pingpong
//    latencies: what a transfer costs when nothing overlaps it. In a
//    bulk-synchronous application the cost that matters is the contribution to
//    the critical path, which can be far larger once a slow transfer gates the
//    next iteration. Treat this table as the *shape* -- the ratios between
//    tiers -- and expect the absolute scale to need an end-to-end correction.
//    DiffusionLB's own predicted-vs-actual residual is the intended source of
//    that correction.

#include "lbcalib.decl.h"
#include "hapi.h"
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Calib calib_proxy;
/* readonly */ size_t max_size;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;

enum Tier { TIER_INTRA_PROCESS = 0, TIER_IPC_SAME_GPU, TIER_IPC_CROSS_GPU,
            TIER_INTER_NODE, TIER_COUNT };

static const char* tierName(int t) {
  switch (t) {
    case TIER_INTRA_PROCESS: return "intra_process";
    case TIER_IPC_SAME_GPU:  return "ipc_same_gpu";
    case TIER_IPC_CROSS_GPU: return "ipc_cross_gpu";
    case TIER_INTER_NODE:    return "inter_node";
  }
  return "unknown";
}

struct Placement {
  int host, process, device;
  Placement() : host(-1), process(-1), device(-1) {}
  Placement(int h, int p, int d) : host(h), process(p), device(d) {}
};

class Main : public CBase_Main {
  std::vector<Placement> place;
  int placementsIn = 0;

  // Message sizes to sweep, and the mean one-way time measured at each.
  std::vector<size_t> sizes;
  std::vector<double> timesForTier;

  int tier = 0;
  int sizeIdx = 0;
  int pairA[TIER_COUNT], pairB[TIER_COUNT];
  double alpha[TIER_COUNT], beta[TIER_COUNT], rsq[TIER_COUNT];
  bool haveTier[TIER_COUNT];
  std::string outPath;
  int devicesPerHost = 0;

 public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    max_size = 4194304;
    n_iters = 200;
    warmup_iters = 20;
    outPath = "lbcost.conf";
    size_t min_size = 64;

    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:i:w:o:")) != -1) {
      switch (c) {
        case 's': min_size = atol(optarg); break;
        case 'x': max_size = atol(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'w': warmup_iters = atoi(optarg); break;
        case 'o': outPath = optarg; break;
        default: CkAbort("usage: lbcalib [-s min] [-x max] [-i iters] [-w warmup] [-o out.conf]\n");
      }
    }
    delete m;

    // Geometric sweep. A linear sweep would put nearly every sample in the
    // bandwidth-bound regime and leave alpha determined by one point.
    for (size_t s = min_size; s <= max_size; s *= 2) sizes.push_back(s);

    for (int t = 0; t < TIER_COUNT; t++) {
      haveTier[t] = false; pairA[t] = pairB[t] = -1;
      alpha[t] = beta[t] = rsq[t] = 0.0;
    }

    place.resize(CkNumPes());
    calib_proxy = CProxy_Calib::ckNew();
    calib_proxy.reportPlacement();
  }

  void reportPlacement(int pe, int host, int process, int device) {
    place[pe] = Placement(host, process, device);
    if (++placementsIn < CkNumPes()) return;

    // How many distinct devices the first host exposes -- written into the
    // table so the balancer can map a PE to a device without asking CUDA about
    // a PE it is not running on.
    {
      std::vector<int> devs;
      const int h0 = place[0].host;
      for (int pe2 = 0; pe2 < CkNumPes(); pe2++)
        if (place[pe2].host == h0 &&
            std::find(devs.begin(), devs.end(), place[pe2].device) == devs.end())
          devs.push_back(place[pe2].device);
      devicesPerHost = (int)devs.size();
    }

    choosePairs();

    CkPrintf("\n=== lbcalib: %d PEs, %d devices/host, sizes %zu..%zu ===\n",
             CkNumPes(), devicesPerHost, sizes.front(), sizes.back());
    for (int t = 0; t < TIER_COUNT; t++) {
      if (haveTier[t])
        CkPrintf("  %-14s PE %d <-> PE %d\n", tierName(t), pairA[t], pairB[t]);
      else
        CkPrintf("  %-14s UNAVAILABLE in this launch -- see the note at exit\n",
                 tierName(t));
    }
    CkPrintf("\n");

    calib_proxy.init();
  }

  // One representative PE pair per tier, if the launch exposes one.
  void choosePairs() {
    const int n = CkNumPes();
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        const Placement &a = place[i], &b = place[j];
        int t;
        if (a.process == b.process)      t = TIER_INTRA_PROCESS;
        else if (a.host != b.host)       t = TIER_INTER_NODE;
        else if (a.device == b.device)   t = TIER_IPC_SAME_GPU;
        else                             t = TIER_IPC_CROSS_GPU;
        if (!haveTier[t]) { haveTier[t] = true; pairA[t] = i; pairB[t] = j; }
      }
    }
  }

  void initDone() { runNextTier(); }

  void runNextTier() {
    while (tier < TIER_COUNT && !haveTier[tier]) tier++;
    if (tier >= TIER_COUNT) { writeConfig(); return; }
    sizeIdx = 0;
    timesForTier.assign(sizes.size(), 0.0);
    calib_proxy.setPair(pairA[tier], pairB[tier]);
  }

  void pairReady() { runNextSize(); }

  void runNextSize() {
    if (sizeIdx >= (int)sizes.size()) { fitTier(); tier++; runNextTier(); return; }
    calib_proxy[pairA[tier]].startSweep(sizes[sizeIdx]);
  }

  void sampleDone(double seconds) {
    timesForTier[sizeIdx] = seconds;
    CkPrintf("  %-14s %9zu B  %10.3f us\n", tierName(tier), sizes[sizeIdx],
             seconds * 1e6);
    sizeIdx++;
    thisProxy.runNextSize();
  }

  // Ordinary least squares of t = alpha + beta*n over the sweep.
  void fitTier() {
    const int N = (int)sizes.size();
    double sn = 0, st = 0, snt = 0, snn = 0;
    for (int i = 0; i < N; i++) {
      const double x = (double)sizes[i], y = timesForTier[i];
      sn += x; st += y; snt += x * y; snn += x * x;
    }
    const double denom = N * snn - sn * sn;
    double b = (denom != 0.0) ? (N * snt - sn * st) / denom : 0.0;
    double a = (st - b * sn) / N;
    // A negative intercept or slope is physically meaningless and would make the
    // model pay a node to move work. Clamp and say so rather than write it out.
    if (b < 0) { CkPrintf("  WARNING: %s beta fitted negative (%.3e); clamped to 0\n", tierName(tier), b); b = 0; }
    if (a < 0) { CkPrintf("  WARNING: %s alpha fitted negative (%.3e); clamped to 0\n", tierName(tier), a); a = 0; }

    double ssTot = 0, ssRes = 0; const double mean = st / N;
    for (int i = 0; i < N; i++) {
      const double pred = a + b * (double)sizes[i];
      ssRes += (timesForTier[i] - pred) * (timesForTier[i] - pred);
      ssTot += (timesForTier[i] - mean) * (timesForTier[i] - mean);
    }
    alpha[tier] = a; beta[tier] = b;
    rsq[tier] = (ssTot > 0) ? 1.0 - ssRes / ssTot : 1.0;
    CkPrintf("  --> %-14s alpha=%.6es beta=%.6es/B  R2=%.4f\n\n",
             tierName(tier), a, b, rsq[tier]);
  }

  void writeConfig() {
    // Any tier the launch could not expose is filled from the next more
    // expensive one that was measured. That direction is deliberate: an
    // over-priced tier makes the balancer leave objects alone, which is the
    // safe failure. Under-pricing is what causes the regression this model
    // exists to prevent.
    for (int t = TIER_COUNT - 2; t >= 0; t--) {
      if (!haveTier[t] && haveTier[t + 1]) {
        alpha[t] = alpha[t + 1]; beta[t] = beta[t + 1];
      }
    }
    for (int t = 1; t < TIER_COUNT; t++) {
      if (!haveTier[t] && haveTier[t - 1]) {
        alpha[t] = alpha[t - 1]; beta[t] = beta[t - 1];
      }
    }

    FILE* f = fopen(outPath.c_str(), "w");
    if (!f) { CkPrintf("lbcalib: cannot write '%s'\n", outPath.c_str()); CkExit(1); }

    fprintf(f, "# DiffusionLB transfer-cost table, written by lbcalib.\n");
    fprintf(f, "# Pass to an application with +LBCostConfig %s\n#\n", outPath.c_str());
    fprintf(f, "# %d PEs, %d device(s) per host, message sizes %zu..%zu bytes,\n",
            CkNumPes(), devicesPerHost, sizes.front(), sizes.back());
    fprintf(f, "# %d timed iterations after %d warmup.\n#\n", n_iters, warmup_iters);
    fprintf(f, "# alpha is seconds per message, beta seconds per byte.\n#\n");
    for (int t = 0; t < TIER_COUNT; t++)
      fprintf(f, "# %-14s %s (R2=%.4f)\n", tierName(t),
              haveTier[t] ? "measured" : "NOT MEASURED - copied from an adjacent tier",
              haveTier[t] ? rsq[t] : 0.0);
    fprintf(f, "\ndevices_per_host = %d\n\n", devicesPerHost);
    for (int t = 0; t < TIER_COUNT; t++) {
      fprintf(f, "%s_alpha = %.9e\n", tierName(t), alpha[t]);
      fprintf(f, "%s_beta  = %.9e\n", tierName(t), beta[t]);
    }

    // Migration cost. NOT measured here: a migration is not a pingpong -- it
    // packs the object, ships host and device state through the migration
    // path, and unpacks. These are estimates from the transport numbers, marked
    // so nobody mistakes them for measurements, and are the obvious next thing
    // to measure directly (time a migration of an object of known footprint).
    fprintf(f, "\n# Estimated, not measured -- see the note in lbcalib.C.\n");
    fprintf(f, "migrate_alpha       = %.9e\n", alpha[TIER_INTER_NODE] * 4.0);
    fprintf(f, "migrate_beta_host   = %.9e\n", beta[TIER_INTER_NODE]);
    fprintf(f, "migrate_beta_device = %.9e\n", beta[TIER_INTER_NODE]);
    fprintf(f, "\n# How many balancer intervals a placement is assumed to last,\n");
    fprintf(f, "# used to amortise the one-off migration cost.\n");
    fprintf(f, "placement_lifetime_intervals = 4\n");
    fclose(f);

    CkPrintf("lbcalib: wrote %s\n", outPath.c_str());
    bool anyMissing = false;
    for (int t = 0; t < TIER_COUNT; t++) if (!haveTier[t]) anyMissing = true;
    if (anyMissing) {
      CkPrintf("\nSome tiers were not exercised by this launch. To measure them all:\n"
               "  intra_process  needs >1 PE per process\n"
               "  ipc_same_gpu   needs >1 process sharing one device\n"
               "  ipc_cross_gpu  needs >1 device per host, 1 process each\n"
               "  inter_node     needs >1 host\n"
               "Unmeasured tiers were filled from an adjacent one, which biases\n"
               "the model toward leaving objects where they are.\n");
    }
    CkExit();
  }
};

class Calib : public CBase_Calib {
  char *d_local = nullptr, *d_remote = nullptr;
  cudaStream_t stream;
  bool allocated = false, streamCreated = false;
  CkDeviceBuffer sendBuf;
  int peer = -1;
  bool initiator = false;
  size_t curSize = 0;
  int iter = 0;
  double startTime = 0.0, accum = 0.0;

 public:
  Calib() {}

  void reportPlacement() {
    int device = -1;
#if CMK_CUDA
    device = (int)hapiMyDevice();
#endif
    main_proxy.reportPlacement(CkMyPe(), CmiPhysicalNodeID(CkMyPe()),
                               CmiNodeOf(CkMyPe()), device);
  }

  void init() {
    if (!allocated) {
      hapiCheck(cudaMalloc(&d_local, max_size));
      hapiCheck(cudaMalloc(&d_remote, max_size));
      hapiCheck(cudaMemset(d_local, 'a', max_size));
      hapiCheck(cudaMemset(d_remote, 'b', max_size));
      allocated = true;
    }
    if (!streamCreated) { cudaStreamCreate(&stream); streamCreated = true; }
    sendBuf = CkDeviceBuffer(d_local);
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  void setPair(int a, int b) {
    initiator = (CkMyPe() == a);
    if (CkMyPe() == a) peer = b;
    else if (CkMyPe() == b) peer = a;
    else peer = -1;
    contribute(CkCallback(CkReductionTarget(Main, pairReady), main_proxy));
  }

  void startSweep(size_t size) {
    curSize = size; iter = 0; accum = 0.0;
    startTime = CkWallTimer();
    thisProxy[peer].ping(size, sendBuf);
  }

  // Responder side: bounce it straight back.
  void ping(size_t& size, char*& data, CkDeviceBufferPost* post) {
    data = d_remote; post[0].hapi_stream = stream;
  }
  void ping(size_t size, char* data) {
    thisProxy[peer].pong(size, sendBuf);
  }

  // Initiator side: one round trip complete.
  void pong(size_t& size, char*& data, CkDeviceBufferPost* post) {
    data = d_remote; post[0].hapi_stream = stream;
  }
  void pong(size_t size, char* data) {
    const double rt = CkWallTimer() - startTime;
    iter++;
    if (iter > warmup_iters) accum += rt / 2.0;   // one way
    if (iter >= warmup_iters + n_iters) {
      main_proxy.sampleDone(accum / n_iters);
      return;
    }
    startTime = CkWallTimer();
    thisProxy[peer].ping(curSize, sendBuf);
  }
};

#include "lbcalib.def.h"
