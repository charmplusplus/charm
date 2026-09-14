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
//
// 3. Migration is measured, not derived. After the transfer sweeps a batch of
//    objects with a known host and device footprint migrates between the PEs
//    of the most expensive same-host tier the launch exposes (cross-GPU IPC,
//    else intra-process) through the same pack / stage / land / unpack path a
//    balancer's move takes, and the batch time per object is fitted as
//    alpha + beta_host * hostBytes + beta_device * devBytes. The batch is what
//    a balancing step pays, so the per-object figure is the batch time divided
//    by the batch, which is what the cost model sums over a step's moves.

#include "lbcalib.decl.h"
#include "hapi.h"
#include "ckrdmadevice.h"
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

  // Migration sweep. Each point is (hostBytes, devBytes) for a batch of
  // migBatch objects that migrate src -> dst -> src, migRounds round trips;
  // the sample is the mean batch time per object.
  struct MigPoint { size_t hostBytes, devBytes; double perObject; };
  std::vector<MigPoint> migPoints;
  int migIdx = 0, migBatch = 32, migRounds = 3;
  int migTier = -1, migSrc = -1, migDst = -1;
  int migReady = 0, migArrived = 0, migLeg = 0;
  double migStart = 0.0, migAccum = 0.0;
  CProxy_Mover movers;
  bool haveMigrate = false;
  double migAlpha = 0.0, migBetaHost = 0.0, migBetaDevice = 0.0, migRsq = 0.0;

 public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    max_size = 4194304;
    n_iters = 200;
    warmup_iters = 20;
    outPath = "lbcost.conf";
    size_t min_size = 64;

    int c;
    while ((c = getopt(m->argc, m->argv, "s:x:i:w:o:b:r:")) != -1) {
      switch (c) {
        case 's': min_size = atol(optarg); break;
        case 'x': max_size = atol(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'w': warmup_iters = atoi(optarg); break;
        case 'o': outPath = optarg; break;
        case 'b': migBatch = atoi(optarg); break;
        case 'r': migRounds = atoi(optarg); break;
        default: CkAbort("usage: lbcalib [-s min] [-x max] [-i iters] [-w warmup] [-o out.conf] "
                         "[-b migration batch] [-r migration round trips]\n");
      }
    }
    // The migration points: a host sweep with no device state, then a device
    // sweep with a small host part, so the two slopes separate.
    for (size_t h = 4096; h <= (size_t)4 << 20; h *= 4) migPoints.push_back(MigPoint{h, 0, 0.0});
    for (size_t d = (size_t)256 << 10; d <= (size_t)16 << 20; d *= 4) migPoints.push_back(MigPoint{4096, d, 0.0});
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
    if (tier >= TIER_COUNT) { runMigration(); return; }
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

  // ---- migration sweep --------------------------------------------------
  //
  // The pair: the costliest same-host tier the launch has, since that is the
  // move a balancer makes across GPUs; intra-process if that is all there is
  // (a handoff, which the runtime does not stage). Inter-node is not used even
  // when present: a table is per host type, and the network move is priced by
  // the inter_node transfer tier.
  void runMigration() {
    for (int t : {TIER_IPC_CROSS_GPU, TIER_IPC_SAME_GPU, TIER_INTRA_PROCESS})
      if (haveTier[t]) { migTier = t; migSrc = pairA[t]; migDst = pairB[t]; break; }
    if (migTier < 0) {
      CkPrintf("lbcalib: no same-host pair to migrate between; migration cost stays estimated\n");
      writeConfig();
      return;
    }
    CkPrintf("=== migration: %d objects per batch, %d round trip(s), PE %d <-> PE %d (%s)\n",
             migBatch, migRounds, migSrc, migDst, tierName(migTier));
    migIdx = 0;
    startMigPoint();
  }

  void startMigPoint() {
    const MigPoint& pt = migPoints[migIdx];
    migReady = 0; migArrived = 0; migLeg = 0; migAccum = 0.0;
    CkArrayOptions opts;
    movers = CProxy_Mover::ckNew(opts);
    for (int i = 0; i < migBatch; i++) movers[i].insert(pt.hostBytes, pt.devBytes, migSrc);
    movers.doneInserting();
  }

  void moverReady() {
    if (++migReady < migBatch) return;
    startLeg();
  }

  void startLeg() {
    migArrived = 0;
    migStart = CkWallTimer();
    movers.moveTo((migLeg % 2 == 0) ? migDst : migSrc);
  }

  void moverArrived() {
    if (++migArrived < migBatch) return;
    const double leg = CkWallTimer() - migStart;
    // The first leg also pays for the destination's first look at the source
    // arena (the IPC import); every later one is the steady case a balancer
    // sees, so only those count.
    if (migLeg > 0) migAccum += leg;
    if (++migLeg < 2 * migRounds) { startLeg(); return; }
    MigPoint& pt = migPoints[migIdx];
    pt.perObject = migAccum / (double)(2 * migRounds - 1) / (double)migBatch;
    CkPrintf("  migrate %8zu B host + %9zu B device: %10.3f us per object (batch of %d)\n",
             pt.hostBytes, pt.devBytes, pt.perObject * 1e6, migBatch);
    movers.ckDestroy();
    migIdx++;
    thisProxy.runNextMigPoint();
  }

  void runNextMigPoint() {
    if (migIdx >= (int)migPoints.size()) { fitMigration(); writeConfig(); return; }
    startMigPoint();
  }

  // Least squares of t = alpha + bh*host + bd*device over the points: the
  // normal equations of a 3-parameter fit, solved directly.
  void fitMigration() {
    const int N = (int)migPoints.size();
    double A[3][3] = {{0,0,0},{0,0,0},{0,0,0}}, B[3] = {0,0,0};
    for (int i = 0; i < N; i++) {
      const double x[3] = {1.0, (double)migPoints[i].hostBytes, (double)migPoints[i].devBytes};
      const double y = migPoints[i].perObject;
      for (int r = 0; r < 3; r++) { B[r] += x[r] * y; for (int c = 0; c < 3; c++) A[r][c] += x[r] * x[c]; }
    }
    // Gaussian elimination with partial pivoting on the 3x3 system.
    double M[3][4];
    for (int r = 0; r < 3; r++) { for (int c = 0; c < 3; c++) M[r][c] = A[r][c]; M[r][3] = B[r]; }
    for (int col = 0; col < 3; col++) {
      int piv = col;
      for (int r = col + 1; r < 3; r++) if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
      for (int c = 0; c < 4; c++) std::swap(M[col][c], M[piv][c]);
      if (std::fabs(M[col][col]) < 1e-300) { CkPrintf("  WARNING: migration fit is singular\n"); return; }
      for (int r = 0; r < 3; r++) {
        if (r == col) continue;
        const double f = M[r][col] / M[col][col];
        for (int c = 0; c < 4; c++) M[r][c] -= f * M[col][c];
      }
    }
    double a = M[0][3] / M[0][0], bh = M[1][3] / M[1][1], bd = M[2][3] / M[2][2];
    if (a < 0) { CkPrintf("  WARNING: migrate_alpha fitted negative (%.3e); clamped to 0\n", a); a = 0; }
    if (bh < 0) { CkPrintf("  WARNING: migrate_beta_host fitted negative (%.3e); clamped to 0\n", bh); bh = 0; }
    if (bd < 0) { CkPrintf("  WARNING: migrate_beta_device fitted negative (%.3e); clamped to 0\n", bd); bd = 0; }
    double mean = 0, ssTot = 0, ssRes = 0;
    for (int i = 0; i < N; i++) mean += migPoints[i].perObject;
    mean /= N;
    for (int i = 0; i < N; i++) {
      const double pred = a + bh * migPoints[i].hostBytes + bd * migPoints[i].devBytes;
      ssRes += (migPoints[i].perObject - pred) * (migPoints[i].perObject - pred);
      ssTot += (migPoints[i].perObject - mean) * (migPoints[i].perObject - mean);
    }
    migAlpha = a; migBetaHost = bh; migBetaDevice = bd;
    migRsq = (ssTot > 0) ? 1.0 - ssRes / ssTot : 1.0;
    haveMigrate = true;
    CkPrintf("  --> migration alpha=%.6es beta_host=%.6es/B beta_device=%.6es/B  R2=%.4f\n\n",
             a, bh, bd, migRsq);
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

    // Migration cost: measured by migrating batches of objects of known
    // footprint (runMigration) when the launch had a same-host pair; otherwise
    // the old estimate from the transport numbers, marked as such.
    if (haveMigrate) {
      fprintf(f, "\n# Migration: measured (R2=%.4f), batches of %d objects migrating\n"
                 "# PE %d <-> PE %d (%s), per-object time = batch time / batch.\n",
              migRsq, migBatch, migSrc, migDst, tierName(migTier));
      fprintf(f, "migrate_alpha       = %.9e\n", migAlpha);
      fprintf(f, "migrate_beta_host   = %.9e\n", migBetaHost);
      fprintf(f, "migrate_beta_device = %.9e\n", migBetaDevice);
    } else {
      fprintf(f, "\n# Estimated, not measured -- see the note in lbcalib.C.\n");
      fprintf(f, "migrate_alpha       = %.9e\n", alpha[TIER_INTER_NODE] * 4.0);
      fprintf(f, "migrate_beta_host   = %.9e\n", beta[TIER_INTER_NODE]);
      fprintf(f, "migrate_beta_device = %.9e\n", beta[TIER_INTER_NODE]);
    }
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

// An object of known footprint that migrates on request. Its device buffer
// comes from the device pool when that is on, as an application's would, and
// travels through pup_buffer_device; after a migration it is arena-bound and
// is released through hapiFreeMigratable, as the applications do (leanmd's
// md_alloc.h).
class Mover : public CBase_Mover {
  size_t hostBytes = 0, devBytes = 0;
  std::vector<char> host;
  char* dev = nullptr;
  bool devFromPool = false, migrated = false;

 public:
  Mover(size_t h, size_t d) : hostBytes(h), devBytes(d) {
    host.assign(h, 'x');
    if (d > 0) {
      if (CkDevicePoolOn()) { dev = (char*)CkDeviceMalloc(d); devFromPool = true; }
      else hapiCheck(hapiMalloc((void**)&dev, d));
      if (dev == nullptr) CkAbort("lbcalib: could not allocate %zu device bytes for a mover\n", d);
      hapiCheck(cudaMemset(dev, 'y', d));
    }
    main_proxy.moverReady();
  }
  Mover(CkMigrateMessage* m) : CBase_Mover(m) { migrated = true; }
  ~Mover() {
    if (dev == nullptr) return;
    if (migrated) hapiFreeMigratable(dev);
    else if (devFromPool) CkDeviceFree(dev);
    else hapiCheck(hapiFree(dev));
  }
  void pup(PUP::er& p) {
    CBase_Mover::pup(p);
    p | hostBytes; p | devBytes; p | host;
    if (devBytes > 0) p.pup_buffer_device(dev, devBytes);
  }
  void ckJustMigrated() override {
    CBase_Mover::ckJustMigrated();
    main_proxy.moverArrived();
  }
  void moveTo(int pe) { ckMigrate(pe); }
};

#include "lbcalib.def.h"
