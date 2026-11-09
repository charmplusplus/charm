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
// 3. A copy's exchange is measured at its fan-in, not derived. A replicated
//    object's copies each send one buffer to every other copy and land every
//    other's at the same time (ShedLB's multiplicity, doc/internals/
//    lb-multiplicity.md); on a PCIe-attached host the k-th copy costs far more
//    than the (k-1)-th, the receives serialising on one stream and the fabric
//    shared (four copies of a 134 MB block sum: 100 ms a peer where one
//    transfer alone takes 5). The fan sweep (-F) exchanges a buffer among k
//    PEs on one host, one per device, for k = 2.. and fits what one peer costs
//    a copy, alpha + beta * bytes, at each fan; the same across hosts, one PE
//    per host, as far as the launch has hosts. The table's xchg_fan<k>_host_*
//    and xchg_fan<k>_inter_* entries price a copy before it has exchanged.
//    The buffers go by device channel puts, each its own batch, since
//    2026-11-06: what the MoE example's copies exchange their block sums by
//    (the entry-method device sends measured before then, -e, serialised
//    their landings on the receiver's stream and collapsed at fan three on
//    PCIe: 18, 1.5 and 0.9 GB/s a peer at two, three and four copies).
//
// 4. Migration is measured, not derived. After the transfer sweeps a batch of
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
#include <map>
#include <string>
#include <cstdio>
#include <cmath>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Calib calib_proxy;
/* readonly */ size_t max_size;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;

enum Tier { TIER_INTRA_PROCESS = 0, TIER_INTRA_PROCESS_PEER, TIER_IPC_SAME_GPU,
            TIER_IPC_CROSS_GPU, TIER_INTER_NODE, TIER_COUNT };

static const char* tierName(int t) {
  switch (t) {
    case TIER_INTRA_PROCESS: return "intra_process";
    case TIER_INTRA_PROCESS_PEER: return "intra_process_peer";
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
  // -M <tier>: force the migration sweep onto a tier instead of picking the
  // costliest same-host one. -M 3 measures an OFF-NODE move, which the default
  // deliberately skips (see runMigration); the resulting migrate_* is the
  // network move, not the local pack/land work, and the table says which.
  int migTierForce = -1;
  int migReady = 0, migArrived = 0, migLeg = 0;
  double migStart = 0.0, migAccum = 0.0;
  CProxy_Mover movers;
  bool haveMigrate = false;
  double migAlpha = 0.0, migBetaHost = 0.0, migBetaDevice = 0.0, migRsq = 0.0;

  // The exchange sweep: fan 2..fanMax among PEs of one host (class 0) and of
  // distinct hosts (class 1), buffers of xsizes, fanIters timed rounds after
  // fanWarm; the sample is the round time per peer, fitted per fan as
  // alpha + beta * bytes.
  static const int FAN_MAX = 8;
  int fanMax = 4, fanIters = 10, fanWarm = 2;
  // The fan sweep exchanges by device channel puts (what a replicated object
  // exchanges its block sums by since 2026-11-06); -e sends the buffers as
  // entry-method device parameters instead (the path measured until then).
  int fanByEntry = 0;
  size_t xMax = (size_t)128 << 20;
  std::vector<size_t> xsizes;
  std::vector<double> xtimes;
  int fanCls = 0, fan = 2, xIdx = 0;
  std::vector<int> fanMembers;
  double xalpha[2][FAN_MAX + 1] = {}, xbeta[2][FAN_MAX + 1] = {}, xrsq[2][FAN_MAX + 1] = {};
  bool xhave[2][FAN_MAX + 1] = {};
  int roundDoneCount = 0, roundIter = 0;
  double roundStart = 0.0, roundAccum = 0.0;

 public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    max_size = 4194304;
    n_iters = 200;
    warmup_iters = 20;
    outPath = "lbcost.conf";
    size_t min_size = 64;

    int c;
    size_t migHostMax = (size_t)4 << 20, migDevMax = (size_t)4 << 20;
    while ((c = getopt(m->argc, m->argv, "s:x:i:w:o:b:r:H:D:M:F:X:N:e")) != -1) {
      switch (c) {
        case 'F': fanMax = std::min(FAN_MAX, std::max(1, atoi(optarg))); break;
        case 'e': fanByEntry = 1; break;
        case 'X': xMax = (size_t)atol(optarg) << 20; break;
        case 'N': fanIters = atoi(optarg); break;
        case 's': min_size = atol(optarg); break;
        case 'x': max_size = atol(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'w': warmup_iters = atoi(optarg); break;
        case 'o': outPath = optarg; break;
        case 'b': migBatch = atoi(optarg); break;
        case 'r': migRounds = atoi(optarg); break;
        case 'M': migTierForce = atoi(optarg); break;
        case 'H': migHostMax = (size_t)atol(optarg) << 20; break;
        case 'D': migDevMax = (size_t)atol(optarg) << 20; break;
        default: CkAbort("usage: lbcalib [-s min] [-x max] [-i iters] [-w warmup] [-o out.conf] "
                         "[-b migration batch] [-r migration round trips] "
                         "[-M migration tier 0..4] "
                         "[-H max host MB per object] [-D max device MB per object] "
                         "[-F max exchange fan (4; 1 = no fan sweep)] [-X max exchange MB (128)] [-N fan rounds (10)]\n");
      }
    }
    // The migration points: a host sweep with no device state, then a device
    // sweep with a small host part, so the two slopes separate. The largest
    // device point times the batch is what the pool has to land at once; the
    // 4 MB default keeps a batch of 32 inside one 256 MB arena. (A 16 MB
    // point, 512 MB per batch, made both processes grow to four arenas and
    // the destination died in the landing: arenas added after startup are
    // not pre-opened on the peers the way the first one is.)
    for (size_t h = 4096; h <= migHostMax; h *= 4) migPoints.push_back(MigPoint{h, 0, 0.0});
    for (size_t d = (size_t)256 << 10; d <= migDevMax; d *= 2) migPoints.push_back(MigPoint{4096, d, 0.0});
    delete m;

    // Geometric sweep. A linear sweep would put nearly every sample in the
    // bandwidth-bound regime and leave alpha determined by one point.
    for (size_t s = min_size; s <= max_size; s *= 2) sizes.push_back(s);
    // The exchange sizes: 4 MB up, the block sums a copy exchanges being tens
    // to hundreds of MB.
    for (size_t s = (size_t)4 << 20; s <= xMax; s *= 2) xsizes.push_back(s);

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
        if (a.process == b.process)      t = (a.device == b.device) ? TIER_INTRA_PROCESS : TIER_INTRA_PROCESS_PEER;
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
    if (tier >= TIER_COUNT) { runFanSweeps(); return; }
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

  // ---- exchange sweep ---------------------------------------------------
  void runFanSweeps() {
    if (fanMax < 2 || xsizes.empty()) { runMigration(); return; }
    CkPrintf("=== exchange sweep by %s\n", fanByEntry ? "entry-method device sends" : "device channel puts");
    if (fanByEntry) { fanCls = 0; fan = 2; thisProxy.nextFan(); return; }
    setUpXChannels();
  }
  // The members of a group of `fan` in class `fanCls`: one PE per device on
  // the first host, or one PE per host.
  bool groupFor(int cls, int k, std::vector<int>& out) {
    out.clear();
    if (cls == 0) {
      std::vector<int> devs;
      for (int pe = 0; pe < CkNumPes() && (int)out.size() < k; pe++)
        if (place[pe].host == place[0].host &&
            std::find(devs.begin(), devs.end(), place[pe].device) == devs.end()) {
          devs.push_back(place[pe].device); out.push_back(pe);
        }
    } else {
      std::vector<int> hosts;
      for (int pe = 0; pe < CkNumPes() && (int)out.size() < k; pe++)
        if (std::find(hosts.begin(), hosts.end(), place[pe].host) == hosts.end()) {
          hosts.push_back(place[pe].host); out.push_back(pe);
        }
    }
    return (int)out.size() == k;
  }
  bool chooseFan() { return groupFor(fanCls, fan, fanMembers); }
  // Channels: every PE that will put to another in any group gets that PE's
  // channel over a landing buffer of its own, once, before the sweeps.
  void setUpXChannels() {
    std::vector<std::vector<int>> peers(CkNumPes());
    std::vector<int> g;
    for (int cls = 0; cls < 2; cls++)
      for (int k = 2; k <= fanMax; k++) {
        if (!groupFor(cls, k, g)) continue;
        for (int a : g)
          for (int b : g)
            if (a != b && std::find(peers[a].begin(), peers[a].end(), b) == peers[a].end()) peers[a].push_back(b);
      }
    for (int pe = 0; pe < CkNumPes(); pe++) calib_proxy[pe].setXPeers(peers[pe], xMax);
  }
  void xChannelsReady() { fanCls = 0; fan = 2; thisProxy.nextFan(); }
  void xHoldersDropped() { calib_proxy.closeXChannels(); }
  void xChannelsClosed() { runMigration(); }
  void finishFans() {
    if (fanByEntry) { runMigration(); return; }
    calib_proxy.dropXHolders();
  }
  void nextFan() {
    for (;;) {
      if (fan > fanMax) { fanCls++; fan = 2; }
      if (fanCls > 1) { finishFans(); return; }
      if (chooseFan()) break;
      // No group of this size in this class: the larger ones are out too.
      CkPrintf("  exchange fan %d %s: no %d %s in this launch\n", fan, fanCls == 0 ? "host" : "inter",
               fan, fanCls == 0 ? "devices on one host" : "hosts");
      fanCls++; fan = 2;
    }
    CkPrintf("=== exchange at fan %d (%s): PEs", fan, fanCls == 0 ? "one host, one PE per device" : "one PE per host");
    for (int pe : fanMembers) CkPrintf(" %d", pe);
    CkPrintf(", %zu..%zu B, %d rounds after %d\n", xsizes.front(), xsizes.back(), fanIters, fanWarm);
    xIdx = 0;
    xtimes.assign(xsizes.size(), 0.0);
    calib_proxy.setGroup(fanMembers, xMax, fanByEntry ? fanMax : 0);
  }
  void groupReady() { thisProxy.nextXSize(); }
  void nextXSize() {
    if (xIdx >= (int)xsizes.size()) { fitFan(); fan++; thisProxy.nextFan(); return; }
    roundIter = 0; roundAccum = 0.0;
    startRound();
  }
  void startRound() {
    roundDoneCount = 0;
    roundStart = CkWallTimer();
    for (int pe : fanMembers) calib_proxy[pe].round(xsizes[xIdx]);
  }
  // A member has every other member's buffer; the round ends with the last.
  void memberDone() {
    if (++roundDoneCount < fan) return;
    const double dt = CkWallTimer() - roundStart;
    roundIter++;
    if (roundIter > fanWarm) roundAccum += dt;
    if (roundIter < fanWarm + fanIters) { startRound(); return; }
    xtimes[xIdx] = roundAccum / fanIters / (fan - 1);   // per peer
    CkPrintf("  fan %d %-5s %10zu B  %10.3f ms per round, %8.3f ms per peer\n", fan, fanCls == 0 ? "host" : "inter",
             xsizes[xIdx], roundAccum / fanIters * 1e3, xtimes[xIdx] * 1e3);
    xIdx++;
    thisProxy.nextXSize();
  }
  void fitFan() {
    const int N = (int)xsizes.size();
    double sn = 0, st = 0, snt = 0, snn = 0;
    for (int i = 0; i < N; i++) {
      const double x = (double)xsizes[i], y = xtimes[i];
      sn += x; st += y; snt += x * y; snn += x * x;
    }
    const double denom = N * snn - sn * sn;
    double b = (denom != 0.0) ? (N * snt - sn * st) / denom : 0.0;
    double a = (st - b * sn) / N;
    if (b < 0) { CkPrintf("  WARNING: fan %d beta fitted negative (%.3e); clamped to 0\n", fan, b); b = 0; }
    if (a < 0) a = 0;   // a negative intercept at these sizes is the fit's slack, not a cost
    double ssTot = 0, ssRes = 0; const double mean = st / N;
    for (int i = 0; i < N; i++) {
      const double pred = a + b * (double)xsizes[i];
      ssRes += (xtimes[i] - pred) * (xtimes[i] - pred);
      ssTot += (xtimes[i] - mean) * (xtimes[i] - mean);
    }
    xalpha[fanCls][fan] = a; xbeta[fanCls][fan] = b;
    xrsq[fanCls][fan] = (ssTot > 0) ? 1.0 - ssRes / ssTot : 1.0;
    xhave[fanCls][fan] = true;
    CkPrintf("  --> fan %d %-5s alpha=%.6es beta=%.6es/B per peer (%.1f GB/s)  R2=%.4f\n\n", fan,
             fanCls == 0 ? "host" : "inter", a, b, b > 0 ? 1e-9 / b : 0.0, xrsq[fanCls][fan]);
  }

  // ---- migration sweep --------------------------------------------------
  //
  // The pair: the costliest same-host tier the launch has, since that is the
  // move a balancer makes across GPUs; intra-process if that is all there is
  // (a handoff, which the runtime does not stage). Inter-node is not used even
  // when present: a table is per host type, and the network move is priced by
  // the inter_node transfer tier.
  void runMigration() {
    if (migTierForce >= 0 && migTierForce < TIER_COUNT) {
      if (!haveTier[migTierForce]) {
        CkPrintf("lbcalib: -M %d (%s) has no pair in this launch; migration cost stays estimated\n",
                 migTierForce, tierName(migTierForce));
        writeConfig();
        return;
      }
      migTier = migTierForce; migSrc = pairA[migTier]; migDst = pairB[migTier];
    } else {
      for (int t : {TIER_IPC_CROSS_GPU, TIER_IPC_SAME_GPU, TIER_INTRA_PROCESS})
        if (haveTier[t]) { migTier = t; migSrc = pairA[t]; migDst = pairB[t]; break; }
    }
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

    // The exchange at each fan: what one peer costs a copy.
    bool anyFan = false;
    for (int c = 0; c < 2; c++) for (int k = 2; k <= FAN_MAX; k++) if (xhave[c][k]) anyFan = true;
    if (anyFan) {
      fprintf(f, "\n# A copy's exchange with one peer in a group of k copies exchanging all at once\n"
                 "# (by %s;\n"
                 "# xchg_fan<k>: k PEs, one per device on one host / one per host, each sending\n"
                 "# %zu..%zu B to every other), alpha seconds per message, beta seconds per byte,\n"
                 "# per peer; %d rounds after %d. A balancer prices a copy's k-1 peers with these.\n",
              fanByEntry ? "entry-method device sends" : "device channel puts, each its own batch",
              xsizes.front(), xsizes.back(), fanIters, fanWarm);
      for (int c = 0; c < 2; c++)
        for (int k = 2; k <= FAN_MAX; k++) {
          if (!xhave[c][k]) continue;
          fprintf(f, "xchg_fan%d_%s_alpha = %.9e   # R2=%.4f\n", k, c == 0 ? "host" : "inter", xalpha[c][k], xrsq[c][k]);
          fprintf(f, "xchg_fan%d_%s_beta  = %.9e\n", k, c == 0 ? "host" : "inter", xbeta[c][k]);
        }
    }

    // Migration cost: measured by migrating batches of objects of known
    // footprint (runMigration) when the launch had a same-host pair; otherwise
    // the old estimate from the transport numbers, marked as such.
    if (haveMigrate) {
      fprintf(f, "\n# Migration: measured (R2=%.4f), batches of %d objects migrating\n"
                 "# PE %d <-> PE %d (%s), per-object time = batch time / batch.\n",
              migRsq, migBatch, migSrc, migDst, tierName(migTier));
      if (migTier == TIER_INTER_NODE)
        fprintf(f, "# NOTE: this migrate_* is an OFF-NODE move (-M 3). It already contains the\n"
                   "# network transfer, so do NOT also charge the inter_node transfer tier for\n"
                   "# a migration priced from it.\n");
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
  hapiStream_t stream;
  bool allocated = false, streamCreated = false;
  CkDeviceBuffer sendBuf;
  int peer = -1;
  bool initiator = false;
  size_t curSize = 0;
  int iter = 0;
  double startTime = 0.0, accum = 0.0;
  // The exchange sweep: this PE's place in the group, its send buffer and one
  // landing buffer per peer slot (allocated once, at the largest size).
  std::vector<int> grp;
  int grpIdx = -1;
  char* d_xsend = nullptr;
  std::vector<char*> d_xland;
  // Arrivals of this group's puts, counted across rounds: a peer's put for
  // the next round can land before this PE's round() entry runs (job
  // 23625084: Main's round message lost to the put by a few hundred us), so
  // the count is never reset by a round, only consumed by the report.
  int landed = 0;
  // Channels: a landing buffer and an owned channel per PE that puts here,
  // handed to it; holder copies of the channels of the PEs this one puts to.
  std::map<int, char*> xlandOf;
  std::map<int, CkDeviceChannel> xown;
  std::map<int, CkDeviceChannel*> xhold;
  int xExpect = -1, xHave = 0;
  uint64_t xTag = 0;

 public:
  Calib() {}

  void reportPlacement() {
    int device = -1;
#if CMK_CUDA || CMK_HIP   // the HIP build (CMK_HIP) drives devices too; without this every PE reported device -1 (one device per host, 2026-10-22)
    device = (int)hapiMyDevice();
#endif
    main_proxy.reportPlacement(CkMyPe(), CmiPhysicalNodeID(CkMyPe()),
                               CmiNodeOf(CkMyPe()), device);
  }

  void init() {
    if (!allocated) {
      hapiCheck(hapiMalloc(&d_local, max_size));
      hapiCheck(hapiMalloc(&d_remote, max_size));
      hapiCheck(hapiMemset(d_local, 'a', max_size));
      hapiCheck(hapiMemset(d_remote, 'b', max_size));
      allocated = true;
    }
    if (!streamCreated) { hapiStreamCreate(&stream); streamCreated = true; }
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

  // Before the sweeps: the landing buffer and channel for each PE that will
  // put here, the channel sent to it; this PE is ready once it holds every
  // channel it will put into.
  void setXPeers(const std::vector<int>& peers, size_t maxBytes) {
    if (d_xsend == nullptr) {
      hapiCheck(hapiMalloc((void**)&d_xsend, maxBytes));
      hapiCheck(hapiMemset(d_xsend, 'x', maxBytes));
    }
    for (int p : peers) {
      char* land = nullptr;
      hapiCheck(hapiMalloc((void**)&land, maxBytes));
      xlandOf[p] = land;
      xown[p] = CkDeviceChannel(land, maxBytes, CkCallback(CkIndex_Calib::xarrived(NULL), thisProxy[CkMyPe()]));
      xown[p].open();
      thisProxy[p].takeXChannel(CkMyPe(), xown[p]);
    }
    xExpect = (int)peers.size();
    if (xHave == xExpect) contribute(CkCallback(CkReductionTarget(Main, xChannelsReady), main_proxy));
  }
  void takeXChannel(int owner, CkDeviceChannel c) {
    delete xhold[owner];
    xhold[owner] = new CkDeviceChannel(c);
    xHave++;
    if (xExpect >= 0 && xHave == xExpect) contribute(CkCallback(CkReductionTarget(Main, xChannelsReady), main_proxy));
  }
  void xarrived(CkDataMsg* m) {
    CkDeviceChannelEvent ev;
    memcpy(&ev, m->getData(), sizeof(ev));
    CkDeviceChannel* c = nullptr;
    for (auto& kv : xown) if (kv.second.id() == ev.channel) { c = &kv.second; break; }
    if (c == nullptr) CkAbort("lbcalib: a put note for channel %llx, not one of PE %d's\n", (unsigned long long)ev.channel, CkMyPe());
    const CkDeviceChannelEvent* e = c->event(m);
    delete m;
    if (e == NULL) return;
    c->release();   // the bytes are not read; the sender may put again
    oneLanded();
  }
  // Every other member's buffer is in: this member's part of the round is
  // done, whichever round the puts belong to.
  void oneLanded() {
    if (++landed < (int)grp.size() - 1) return;
    landed = 0;
    main_proxy.memberDone();
  }
  void xsent() {}
  void dropXHolders() {
    for (auto& kv : xhold) delete kv.second;
    xhold.clear();
    contribute(CkCallback(CkReductionTarget(Main, xHoldersDropped), main_proxy));
  }
  void closeXChannels() {
    for (auto& kv : xown) kv.second.close();
    xown.clear();
    contribute(CkCallback(CkReductionTarget(Main, xChannelsClosed), main_proxy));
  }

  void setGroup(const std::vector<int>& members, size_t maxBytes, int fanMax) {
    grp = members;
    grpIdx = -1;
    landed = 0;   // no put of the previous group is in flight: Main moved on after its last round
    for (int i = 0; i < (int)members.size(); i++) if (members[i] == CkMyPe()) grpIdx = i;
    if (fanMax > 0 && grpIdx >= 0 && d_xsend == nullptr) {
      hapiCheck(hapiMalloc((void**)&d_xsend, maxBytes));
      hapiCheck(hapiMemset(d_xsend, 'x', maxBytes));
      d_xland.assign(fanMax - 1, nullptr);
      for (int i = 0; i < fanMax - 1; i++) hapiCheck(hapiMalloc((void**)&d_xland[i], maxBytes));
    }
    contribute(CkCallback(CkReductionTarget(Main, groupReady), main_proxy));
  }
  // One round: my buffer to every other member, into the slot that is mine
  // at each (members below me keep my index, those above me my index less one).
  void round(size_t size) {
    for (int j = 0; j < (int)grp.size(); j++) {
      if (j == grpIdx) continue;
      if (!xhold.empty()) {
        // The channel path: one put per peer, each its own batch (what the
        // MoE example's copies do with their block sums).
        CkDeviceChannel* ch = xhold[grp[j]];
        if (ch == nullptr) CkAbort("lbcalib: PE %d holds no channel of PE %d\n", CkMyPe(), grp[j]);
        ch->put(d_xsend, size, stream, CkCallback(CkIndex_Calib::xsent(), thisProxy[CkMyPe()]), xTag++);
        CkRdmaDeviceChannelBatchFlush();
        continue;
      }
      const int slotAtPeer = grpIdx < j ? grpIdx : grpIdx - 1;
      thisProxy[grp[j]].xpart(size, slotAtPeer, CkDeviceBuffer(d_xsend));
    }
    if (grp.size() == 1) main_proxy.memberDone();
  }
  void xpart(size_t& size, int& slot, char*& data, CkDeviceBufferPost* post) {
    data = d_xland[slot]; post[0].hapi_stream = stream;
  }
  void xpart(size_t size, int slot, char* data) { oneLanded(); }

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
      hapiCheck(hapiMemset(dev, 'y', d));
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
