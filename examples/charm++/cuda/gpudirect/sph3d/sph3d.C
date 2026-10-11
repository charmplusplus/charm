#include "hapi.h"
#include "sph3d.decl.h"
#include "sph3d.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <climits>
#include <malloc.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <time.h>
#include <atomic>

// The runtime's +gputrace hook for application message flights (ck-core/cknvtx.h,
// which the install does not ship): an L record per message when the trace is on.
extern std::atomic<bool> _nvtxOn;
extern void _nvtxTraceMsg(unsigned long long t_send, unsigned long long t_env, unsigned long long t_land, int kind, int tier);

// A 3D port of sph2d.C. The runtime machinery -- submitter queue, stream-less
// sends from completion callbacks, pinned and event pools, direct halo landing,
// device-to-device pup, the async-LB windows and the integer checks -- is the
// same code; see the comments in sph2d.C for the reasoning behind each piece.
// What differs is geometry: a 3D patch array, 26 exchange partners indexed as
// in sph3d.h, per-direction exchange slots of three sizes, and 3D cell lists.
//
// Two things GPUSPH does that this code can now do as well (sph3d.h):
//
// NEIGHBOUR-LIST WINDOW (-k N). Step numbers 1, N, 2N, ... are REBUILD steps:
// the locals are sorted into cell order, the halo is packed by position and
// the per-direction index lists of what was sent are recorded, the ghosts land
// by arrival order and that layout (ghost_off/ghost_cnt) is recorded, and the
// cell list is built over locals + ghosts with each particle's cell kept
// (d_cell_of). Steps N-1, 2N-1, ... are MIGRATE steps: the compaction runs at
// their end, so the next step rebuilds from a freshly migrated set. In between
// nothing permutes the arrays: every halo exchange re-sends the same particles
// by index into the same ghost slots, the forces use the stale cell list, and
// the migration messages go out empty (the activity advertisement rides on
// them, so they are still sent). With N = 1 every step is both.
//
// PREDICTOR-CORRECTOR (-p). Between the two force evaluations the predicted
// state (d_parts[1-cur], same layout) is exchanged once more, by index, into
// its own ghost slots; the corrector then integrates the n-state in place.
// The second exchange has its own entry methods, send and staging buffers: a
// partner can post its second exchange before this patch has consumed its
// first, and the first exchange's send buffer may still be being read.
//
// CONCURRENT EXCHANGE (-c, Euler-Cromer with -k 1). The two rounds of a step,
// the halo before the forces and the migration after the integration, are
// each a barrier on 26 partners; with one patch kernel at a time per GPU the
// 2000-step timeline (job 22796600) showed every GPU idle 18-20% of the time
// in 1-5 ms waits at those boundaries, with the partner GPUs busy. Here the
// migration of step t and the halo of step t+1 leave together, at the end of
// step t, so a step waits once for both. What makes that possible:
//   - the halo of step t+1 is packed from the survivors plus this patch's own
//     leavers (positions at t+1), not from the migrants about to arrive: a
//     leaver stays this patch's ghost for one step (the new owner cannot send
//     it back before it has it) and is sent to every other partner within
//     support of it, but not to the partner whose box it is in, which gets it
//     as a migrant;
//   - the migrants are consumed first (their advertisements say which halos to
//     expect), then the leaver ghosts are placed behind them, then the halos
//     land behind those (directly once every migration envelope of the step is
//     known, otherwise staged in a per-parity slot and copied);
//   - the activity advertised for step t+1 is decided before the incoming
//     migrants are known: ACTIVE is exact (a neighbour with fluid at t is the
//     only source of them), FLUID is widened by that same condition, so the
//     active region is one patch ring wider at the front;
//   - the quiet-edge bookkeeping (xchg_step, xchg_min, edgeSkipsC) is the same
//     proof shifted by one step: a message for step s describes step s.
// Step 1's exchange is a prologue pack of the initial state. Every exchange
// message carries the sender's clock for the latency summary (Main::latDone)
// and the +gputrace L records.

// The message timestamps on the same clock as the +gputrace records (CUPTI's
// timestamp is CLOCK_REALTIME in ns on Delta; the merge aligns by the header
// pair anyway), without a CUPTI call: cuptiGetTimestamp blocked for 66-86 ms
// in receive hooks while the helper thread flushed CUPTI (job 22798528
// asyncH trace), and every send and receive calls this.
static inline long long sphNowNs() {
  struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline void sphSubmitMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t s) {
  hapiSubmit(s, [=]() { hapiCheck(cudaMemcpyAsync(dst, src, n, k, s)); });
}
static inline void sphSubmitWaitEvent(cudaStream_t s, cudaEvent_t ev) {
  hapiSubmit(s, [=]() { hapiCheck(hapiStreamWaitEvent(s, ev, 0)); });
}
static inline cudaError_t sphDrainSync(cudaStream_t s) {
  hapiSubmitDrain();
  return cudaStreamSynchronize(s);
}

// Every device send is made from the completion callback of the kernel that
// packed the buffer, so the buffer is complete when the send is issued.
// SPH_SEND_STREAM=1 restores the stream-tagged contract for A/B.
static inline CkDeviceBuffer sphSendBuffer(const void* ptr, const CkCallback& cb, cudaStream_t s) {
  static const bool tag = getenv("SPH_SEND_STREAM") != nullptr;
  return tag ? CkDeviceBuffer(ptr, cb, s) : CkDeviceBuffer(ptr, cb);
}

// SPH_SEND_WARN_MS=N: a send loop (26 or 52 device sends) that takes longer
// than N ms prints which send took the time -- the kernels-only async trace
// (job 22798305) had one 41 ms leaversPacked entry on a PE, i.e. a send call
// blocking inside the runtime.
static const double send_warn_ms = getenv("SPH_SEND_WARN_MS") ? atof(getenv("SPH_SEND_WARN_MS")) : 0.0;
struct SendTimer {
  double t0 = 0.0, worst = 0.0; int worst_d = -1, n = 0;
  void begin() { if (send_warn_ms > 0) t0 = CkWallTimer(); }
  void end(int d) {
    if (send_warn_ms <= 0) return;
    const double dt = (CkWallTimer() - t0) * 1000.0; n++;
    if (dt > worst) { worst = dt; worst_d = d; }
  }
  void report(const char* what, int px, int py, int pz, int it, double total_ms, int tier) const {
    if (send_warn_ms > 0 && total_ms > send_warn_ms)
      CkPrintf("[SENDSTALL] pe %d patch (%d,%d,%d) step %d: %s took %.1f ms for %d sends, worst %.1f ms to direction %d (tier %d: 0 same process, 1 same node, 2 other node)\n",
               CkMyPe(), px, py, pz, it, what, total_ms, n, worst, worst_d, tier);
  }
};

// Pinned landing pads and ordering events are recycled, never returned to the
// driver (cudaMallocHost/cudaFreeHost and event create/destroy synchronize the
// device; see sph2d.C). Process-wide pools, one lock each.
namespace {
class PinnedBlockPool {
 public:
  explicit PinnedBlockPool(size_t bytes)
      : bytes_((bytes + 63) & ~size_t(63)), lock_(CmiCreateLock()) {}
  void* take() {
    CmiLock(lock_);
    if (free_.empty()) grow();
    void* p = free_.back();
    free_.pop_back();
    CmiUnlock(lock_);
    return p;
  }
  void give(void* p) {
    if (p == NULL) return;
    CmiLock(lock_);
    free_.push_back(p);
    CmiUnlock(lock_);
  }
 private:
  void grow() {
    const size_t n = 1024;
    char* slab = NULL;
    hapiCheck(hapiMallocHost((void**)&slab, bytes_ * n));
    free_.reserve(free_.size() + n);
    for (size_t i = 0; i < n; i++) free_.push_back(slab + i * bytes_);
  }
  size_t bytes_;
  std::vector<void*> free_;
  CmiNodeLock lock_;
};
PinnedBlockPool& pinnedPool(size_t bytes) {
  static CmiNodeLock lock = CmiCreateLock();
  static std::map<size_t, PinnedBlockPool*> pools;
  CmiLock(lock);
  PinnedBlockPool*& p = pools[bytes];
  if (p == NULL) p = new PinnedBlockPool(bytes);
  CmiUnlock(lock);
  return *p;
}
template <typename T>
inline T* takePinned(size_t count) {
  return (T*)pinnedPool(sizeof(T) * count).take();
}
template <typename T>
inline void givePinned(T* p, size_t count) {
  pinnedPool(sizeof(T) * count).give((void*)p);
}

class EventPool {
 public:
  EventPool() : lock_(CmiCreateLock()) {}
  cudaEvent_t take() {
    CmiLock(lock_);
    cudaEvent_t e;
    if (free_.empty()) {
      hapiCheck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    } else {
      e = free_.back();
      free_.pop_back();
    }
    CmiUnlock(lock_);
    return e;
  }
  void give(cudaEvent_t e) {
    CmiLock(lock_);
    free_.push_back(e);
    CmiUnlock(lock_);
  }
 private:
  std::vector<cudaEvent_t> free_;
  CmiNodeLock lock_;
};
EventPool& eventPool() {
  static EventPool s;
  return s;
}
}  // namespace

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Patch patch_proxy;
/* readonly */ CProxy_HostProfGroup prof_proxy;
/* readonly */ RealType dom_lx;
/* readonly */ RealType dom_ly;
/* readonly */ RealType dom_lz;
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_chares_z;
/* readonly */ RealType spacing;
/* readonly */ RealType smooth_h;
/* readonly */ RealType support;
/* readonly */ RealType pmass;
/* readonly */ RealType rho0;
/* readonly */ RealType sound_c0;
/* readonly */ RealType gravity;
/* readonly */ RealType sim_dt;
/* readonly */ RealType wall_t;
/* readonly */ RealType col_w;
/* readonly */ RealType col_h;
/* readonly */ RealType init_vx;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ int first_lb;
/* readonly */ int lb_freq;
/* readonly */ int async_lb;
/* readonly */ int lb_wait_lag;
/* readonly */ int lattice_nx;
/* readonly */ int lattice_ny;
/* readonly */ int part_capacity;
/* readonly */ int exch_capacity;
/* readonly */ int stats_freq;
/* readonly */ int nl_freq;
/* readonly */ int integrator;
/* readonly */ RealType verlet_margin;
/* readonly */ RealType cell_size;
/* readonly */ int concurrent_x;

extern size_t sphScanTempBytes(int);
extern void invokeCellBuild(const Particle*, int, RealType, RealType, RealType,
    RealType, int, int, int, int, int*, int*, int*, int*, int*, void*, size_t,
    cudaStream_t);
extern void invokeGather(const Particle*, const int*, int, Particle*, cudaStream_t);
extern void invokeLayout(const Particle*, const int*, int, float4*, float4*,
    unsigned char*, cudaStream_t);
extern void invokeEOS(Particle*, int, RealType, RealType, int*, cudaStream_t);
extern void invokeForces(int, int, int, int, int, int, const int*, const int*,
    const int*, const int*, const float4*, const float4*, const unsigned char*,
    RealType, RealType, RealType, RealType, RealType*, RealType*, RealType*,
    RealType*, cudaStream_t);
extern void invokeIntegrate(Particle*, int, RealType, RealType, RealType, RealType*,
    RealType*, RealType*, RealType*, int*, cudaStream_t);
extern void invokeIntegrateLeavers(bool, const Particle*, int, RealType, RealType,
    RealType, RealType*, RealType*, RealType*, RealType*, RealType, RealType,
    RealType, RealType, RealType, RealType, Particle**, Particle*, int*, int,
    cudaStream_t);
extern void sphSetInteriorLayers(int);
extern size_t sphSelectTempBytes(int);
extern void invokeForcesInteriorRows(int, int, int, int, int, const int*, const int*, const int*, const int*,
    const float4*, const float4*, const unsigned char*, RealType, RealType, RealType, RealType, RealType*,
    RealType*, RealType*, RealType*, int*, int*, cudaStream_t);
extern void invokeForcesTwoListRows(int, int, int, int, int, int, int, const int*, const int*, const int*,
    const int*, const int*, const int*, const int*, const int*, const float4*, const float4*,
    const unsigned char*, RealType, RealType, RealType, RealType, RealType*, RealType*, RealType*,
    RealType*, int*, int*, cudaStream_t);
extern void sphIota(int, int*, cudaStream_t);
extern void invokeForcesInteriorSel(int, int, int, int, const int*, const int*, const int*, const int*,
    const float4*, const float4*, const unsigned char*, RealType, RealType, RealType, RealType, RealType*,
    RealType*, RealType*, RealType*, const int*, unsigned char*, int*, int*, void*, size_t, cudaStream_t);
extern void invokeForcesTwoList(int, int, int, int, int, int, const int*, const int*, const int*,
    const int*, const int*, const int*, const int*, const int*, const float4*, const float4*,
    const unsigned char*, RealType, RealType, RealType, RealType, RealType*, RealType*, RealType*,
    RealType*, const int*, unsigned char*, int*, int*, void*, size_t, cudaStream_t);
extern void invokePredict(const Particle*, Particle*, int, RealType, RealType,
    RealType, const RealType*, const RealType*, const RealType*, const RealType*,
    cudaStream_t);
extern void invokeCorrect(Particle*, int, RealType, RealType, RealType,
    const RealType*, const RealType*, const RealType*, const RealType*, int*,
    cudaStream_t);
extern void invokePackHalo(const Particle*, int, RealType, RealType, RealType,
    RealType, RealType, RealType, RealType, Particle**, int**, int*, int, int,
    bool, cudaStream_t);
extern void invokePackHaloByIndex(const Particle*, Particle**, const int*,
    const int*, int, int, cudaStream_t);
extern void invokeMarkLeavers(const Particle*, int, RealType, RealType, RealType,
    RealType, RealType, RealType, Particle**, Particle*, int*, int, bool,
    cudaStream_t);
extern void invokeGatherLeaverGhosts(Particle**, const int*, Particle*, Particle*, int, int,
    int, cudaStream_t);
extern void invokePackHaloConcurrent(const Particle*, const int*, int, RealType, RealType,
    RealType, RealType, RealType, RealType, RealType, Particle**, int**, int*, int, int,
    cudaStream_t);
extern void invokeIdBuckets(const Particle*, int, unsigned long long*, unsigned long long*,
    unsigned long long*, cudaStream_t);
extern void invokeFindId(const Particle*, int, int, int*, cudaStream_t);
#define ID_AUDIT_BUCKETS 4096
static const bool id_audit_on = (getenv("SPH_ID_AUDIT") != nullptr);
extern void invokeStats(const Particle*, int, RealType*, cudaStream_t);
extern void invokeCheck(const Particle*, int, RealType, RealType, RealType,
    RealType, RealType, RealType, RealType, RealType, RealType,
    unsigned long long*, cudaStream_t, int);

// Exchange slot layout: the 26 directions laid end to end, each with its own
// size (dirCap), slot STAY skipped. Offsets in particles. The index lists use
// the same layout.
static size_t dirOffset(int d, int cap_face) {
  size_t off = 0;
  for (int k = 0; k < d; k++) if (k != STAY) off += (size_t)dirCap(k, cap_face);
  return off;
}
static size_t exchSlots(int cap_face) { return dirOffset(NUM_DIRS, cap_face); }

// ---- per-PE host-time profile (SPH_HOST_PROFILE=1) ---------------------------
// Every entry body is wrapped in a PhaseScope; a PE is a thread, so the
// accumulators are thread_local. Messages are counted by the tier of the
// partner's PE: same process, same physical node, other node. One line per PE
// at the end (HostProfGroup::report), so the hot PE's time can be split by
// phase and set against its message mix in a single run.
enum HostPhase { PH_ITERATE, PH_START_HALO, PH_SEND_HALO, PH_RECV_HALO, PH_APPEND_GHOSTS,
                 PH_COMPUTE, PH_FINISH, PH_SEND_LEAVERS, PH_RECV_PARTS, PH_APPEND_PARTS,
                 PH_DRAIN, PH_SEND_DONE, PH_END_STEP, PH_PC, PH_N };
static const char* const host_phase_name[PH_N] = {
  "iterate", "startHalo", "sendHalo", "recvHalo", "appendGhosts", "compute", "finish",
  "sendLeavers", "recvParts", "appendParts", "drain", "sendDone", "endOfStep", "pc" };
struct HostProf {
  double t[PH_N] = {}; long n[PH_N] = {};
  long sent[3] = {}, recv[3] = {}; long sent_bytes[3] = {}, recv_bytes[3] = {};
};
static thread_local HostProf g_hprof;
static const bool host_profile_on = (getenv("SPH_HOST_PROFILE") != nullptr);
struct PhaseScope {
  int ph; double t0;
  PhaseScope(int p) : ph(p), t0(host_profile_on ? CkWallTimer() : 0.0) {}
  ~PhaseScope() { if (host_profile_on) { g_hprof.t[ph] += CkWallTimer() - t0; g_hprof.n[ph]++; } }
};
static inline int peTier(int pe) {
  if (pe < 0) return 2;
  if (CmiNodeOf(pe) == CmiMyNode()) return 0;
  return CmiPeOnSamePhysicalNode(pe, CkMyPe()) ? 1 : 2;
}

// ---- message latency ---------------------------------------------------------
// Every exchange message carries the sender's hapiTraceTimestamp(); the
// receiver notes the envelope's arrival (the post hook that picks the landing
// buffer) and the moment the payload is usable (the SDAG body). Per PE, kind
// (0 halo, 1 migration, 2 predicted halo) and tier of the partner's PE (same
// process / same node / other node): count, sums, max and a quarter-octave
// log histogram of microseconds, reduced at the end into Main::latDone's
// percentiles. With +gputrace every message is also an L record of the PE's
// trace file. Cross-node clocks agree to NTP, so tier 2 carries that error.
#define LAT_KINDS 3
#define LAT_BINS 96   // 2^(96/4) us = 16.8 s
struct LatStats {
  int hist[LAT_KINDS][3][LAT_BINS] = {};    // payload usable
  int ehist[LAT_KINDS][3][LAT_BINS] = {};   // envelope arrived (the transport alone)
  long n[LAT_KINDS][3] = {};
  double sum[LAT_KINDS][3] = {}, esum[LAT_KINDS][3] = {}, mx[LAT_KINDS][3] = {};
};
static thread_local LatStats g_lat;
static inline int latBin(double us) {
  if (us < 1.0) return 0;
  const int b = (int)(4.0 * std::log2(us));
  return b < 0 ? 0 : b >= LAT_BINS ? LAT_BINS - 1 : b;
}
static inline double latBinUpperMs(int b) { return std::exp2((b + 1) / 4.0) / 1000.0; }

class Main : public CBase_Main {
  double start_time;
  double last_report_t = 0.0;
  int stat_count;
  long n_total_particles;
  unsigned long long ref_id_sum, ref_bnd_sum, ref_n_fluid, ref_n_bound;
  bool conservation_void;
public:
  Main(CkArgMsg* m) : stat_count(0), conservation_void(false) {
    // Tank: length x (the collapse direction), depth y, height z. The default
    // is the sph2d tank extruded to 0.5 m of depth, with a water column one
    // metre wide and two tall that spans the full depth between the walls.
    dom_lx = 1.5f; dom_ly = 0.5f; dom_lz = 2.5f;
    n_chares_x = 6; n_chares_y = 2; n_chares_z = 10;
    spacing = 0.0088f;
    col_w = 1.0f; col_h = 2.0f;
    init_vx = 0.0f;
    rho0 = 1000.0f;
    gravity = 9.81f;
    n_iters = 1000; warmup_iters = 10;
    first_lb = 99999; lb_freq = 0; async_lb = 0; lb_wait_lag = 3;
    stats_freq = 200;
    nl_freq = 1; integrator = 0; verlet_margin = 0.0f;
    concurrent_x = 0;
    RealType headroom = 2.0f;
    RealType exch_frac = 0.1f;

    int c;
    while ((c = getopt(m->argc, m->argv, "X:Y:Z:x:y:z:s:w:t:V:i:u:f:b:l:ar:e:S:k:pm:c")) != -1) {
      switch (c) {
        case 'X': dom_lx = atof(optarg); break;
        case 'Y': dom_ly = atof(optarg); break;
        case 'Z': dom_lz = atof(optarg); break;
        case 'x': n_chares_x = atoi(optarg); break;
        case 'y': n_chares_y = atoi(optarg); break;
        case 'z': n_chares_z = atoi(optarg); break;
        case 's': spacing = atof(optarg); break;
        case 'w': col_w = atof(optarg); break;
        case 't': col_h = atof(optarg); break;
        case 'V': init_vx = atof(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'u': warmup_iters = atoi(optarg); break;
        case 'f': first_lb = atoi(optarg); break;
        case 'b': lb_freq = atoi(optarg); break;
        case 'l': lb_wait_lag = atoi(optarg); break;
        case 'a': async_lb = 1; break;
        case 'r': headroom = atof(optarg); break;
        case 'e': exch_frac = atof(optarg); break;
        case 'S': stats_freq = atoi(optarg); break;
        case 'k': nl_freq = atoi(optarg); break;
        case 'p': integrator = 1; break;
        case 'm': verlet_margin = atof(optarg); break;
        case 'c': concurrent_x = 1; break;
        default:
          CkPrintf("Usage: sph3d [-X tank length] [-Y tank depth] [-Z tank height]\n"
                   "  -x -y -z [chares along each axis] -s [particle spacing]\n"
                   "  -w [column width, x] -t [column height, z]\n"
                   "  -V [initial fluid speed, +x, m/s]\n"
                   "  -i [iterations] -u [warmup] -S [stats every N steps]\n"
                   "  -f [first LB iter] -b [LB period]\n"
                   "  -a (async LB, needs +LBAsync) -l [wait lag]\n"
                   "  -r [capacity headroom] -e [face exchange buffer fraction]\n"
                   "  -p (predictor-corrector, two force evaluations per step)\n"
                   "  -k [neighbour-list window, steps] -m [Verlet margin, in h]\n"
                   "  -c (concurrent exchange: halo and migration sent together; needs -k 1, Euler-Cromer)\n");
          CkExit();
      }
    }
    delete m;

    if (nl_freq < 1) CkAbort("-k (%d) must be at least 1\n", nl_freq);
    if (verlet_margin < 0.0f) CkAbort("-m (%g) must not be negative\n", verlet_margin);
    if (concurrent_x && (nl_freq != 1 || integrator != 0))
      CkAbort("-c (concurrent exchange) needs -k 1 and the Euler-Cromer integrator (no -p)\n");

    wall_t = 3.0f * spacing;
    smooth_h = 1.3f * spacing;
    support = KERNEL_SUPPORT * smooth_h;
    cell_size = support + verlet_margin * smooth_h;
    pmass = rho0 * spacing * spacing * spacing;

    // c0 = 10 x the fastest expected flow (free fall over the column height,
    // plus any imposed start velocity); acoustic CFL and body-force limits.
    const RealType v_free = std::sqrt(2.0f * gravity * col_h);
    const RealType v_max = std::sqrt(v_free * v_free + init_vx * init_vx);
    sound_c0 = 10.0f * v_max;
    const RealType dt_c = 0.15f * smooth_h / sound_c0;
    const RealType dt_g = 0.25f * std::sqrt(smooth_h / gravity);
    sim_dt = std::min(dt_c, dt_g);

    if (async_lb && !_lb_args.lbAsync()) {
      CkPrintf("[WARN] -a ignored: async LB needs +LBAsync on the command "
               "line. Running the unsplit AtSync barrier instead.\n");
      async_lb = 0;
    }
    // The lag must close the window before the NEXT trigger (see sph2d.C).
    if (async_lb && lb_freq > 0) {
      int gap_after_first = lb_freq - (first_lb % lb_freq);
      int min_gap = gap_after_first < lb_freq ? gap_after_first : lb_freq;
      if (lb_wait_lag < 1 || lb_wait_lag >= min_gap)
        CkAbort("-l (%d) must be in [1, %d): the wait must land before the next "
                "LB trigger (first LB %d, period %d).\n",
                lb_wait_lag, min_gap, first_lb, lb_freq);
    } else if (async_lb && lb_wait_lag < 1) {
      CkAbort("-l (%d) must be at least 1\n", lb_wait_lag);
    }

    // Capacity: a patch's share of the lattice times the headroom. The face
    // exchange slot is a fraction of it; edges and corners get a quarter and
    // a sixteenth of the face slot (dirCap).
    const double patch_vol = (double)(dom_lx / n_chares_x) * (dom_ly / n_chares_y) *
                             (dom_lz / n_chares_z);
    const double per_patch = patch_vol / ((double)spacing * spacing * spacing);
    if (per_patch * headroom > (double)INT_MAX / 4)
      CkAbort("A patch would hold %.3g lattice sites; use more chares or a "
              "coarser spacing\n", per_patch);
    part_capacity = std::max(4096, (int)(per_patch * headroom));
    exch_capacity = std::max(1024, (int)(part_capacity * exch_frac));
    // Keep the slot sizes multiples of 16 so the corner slots are whole.
    exch_capacity = (exch_capacity + 15) & ~15;

    // Global particle id = lattice index (iz*ny + iy)*nx + ix, unique without
    // communication. It has to fit an int.
    lattice_nx = (int)(dom_lx / spacing);
    lattice_ny = (int)(dom_ly / spacing);
    const long lattice_x = lattice_nx, lattice_y = lattice_ny;
    const long lattice_z = (long)(dom_lz / spacing);
    if (lattice_x * lattice_y * lattice_z >= (long)INT_MAX)
      CkAbort("Lattice of %ld sites does not fit a 32-bit particle id; coarsen "
              "the spacing\n", lattice_x * lattice_y * lattice_z);

    main_proxy = thisProxy;
    CkPrintf("SPH 3D dam break: tank %.3f x %.3f x %.3f m (length, depth, height), "
             "spacing %.5f m, h %.5f m\n", dom_lx, dom_ly, dom_lz, spacing, smooth_h);
    CkPrintf("  column %.2f wide x %.2f tall (full depth), rho0 %.0f, c0 %.2f m/s, "
             "dt %.3e s\n", col_w, col_h, rho0, sound_c0, sim_dt);
    const RealType pw = dom_lx / n_chares_x;
    const double cross = init_vx > 0.0f
        ? pw / (init_vx * sim_dt)
        : std::sqrt(2.0 * pw / gravity) / sim_dt;
    CkPrintf("  fluid starts at %.2f m/s; the load crosses one patch (%.3f m) "
             "every ~%.0f steps\n", init_vx, pw, cross);
    CkPrintf("  lattice %ld x %ld x %ld, %d x %d x %d patches, capacity/patch %d, "
             "face exchange %d (edges /4, corners /16)\n", lattice_x, lattice_y,
             lattice_z, n_chares_x, n_chares_y, n_chares_z, part_capacity,
             exch_capacity);
    CkPrintf("  integrator %s (%d force evaluation%s per step); neighbour list, "
             "halo membership and migration every %d step%s; cell size %.3f h "
             "(Verlet margin %.2f h)\n",
             integrator ? "predictor-corrector" : "Euler-Cromer",
             integrator ? 2 : 1, integrator ? "s" : "", nl_freq,
             nl_freq == 1 ? "" : "s", cell_size / smooth_h, verlet_margin);
    CkPrintf("  iterations %d (+%d warmup), first LB %d, LB period %d%s\n",
             n_iters, warmup_iters, first_lb, lb_freq,
             async_lb ? ", async" : "");
    if (concurrent_x)
      CkPrintf("  concurrent exchange: a step's halo and the previous step's migration "
               "leave together, one wait per step\n");

    patch_proxy = CProxy_Patch::ckNew(n_chares_x, n_chares_y, n_chares_z);
    prof_proxy = CProxy_HostProfGroup::ckNew();
    patch_proxy.init();
  }

  // sum np | id sum | boundary sum | count fluid | count boundary
  void initDone(CkReductionMsg* msg) {
    int n;
    CkReduction::tupleElement* t;
    msg->toTuple(&t, &n);
    n_total_particles = *(long*)t[0].data;
    ref_id_sum        = *(unsigned long long*)t[1].data;
    ref_bnd_sum       = *(unsigned long long*)t[2].data;
    ref_n_fluid       = *(unsigned long long*)t[3].data;
    ref_n_bound       = *(unsigned long long*)t[4].data;
    delete[] t;
    delete msg;

    CkPrintf("Init: %ld particles (%llu fluid, %llu boundary)\n"
             "  reference checksums: id %016llx  boundary state %016llx\n",
             n_total_particles, ref_n_fluid, ref_n_bound, ref_id_sum,
             ref_bnd_sum);
    start_time = CkWallTimer();
    patch_proxy.iterate();
    // SPH_STUCK_SECONDS=N: if no stats report arrives for N s, every patch
    // prints the state of its step (a hang diagnostic; one line per patch).
    static const int stuck_s = getenv("SPH_STUCK_SECONDS") ? atoi(getenv("SPH_STUCK_SECONDS")) : 0;
    if (stuck_s > 0) thisProxy.watchdog();
  }

  // SPH_ID_AUDIT: per-bucket (count, sum id, sum id^2) of every local particle,
  // reduced at each stats step. Against the first audit's reference, a bucket
  // whose count moved by +1/-1 names the duplicated / lost id by its sum
  // difference; a bucket with the count unchanged and the sums changed holds
  // both: x - y = dsum, x^2 - y^2 = dsum2, so x + y = dsum2 / dsum. The holders
  // then print themselves (Patch::dumpId).
  std::vector<unsigned long long> ref_cnt, ref_sum, ref_sum2;
  int corrupt_step = -1;   // SPH_ID_AUDIT: checkState defers its abort to the audit of that step
  void idAudit(CkReductionMsg* msg) {
    int nt; CkReduction::tupleElement* t; msg->toTuple(&t, &nt);
    const unsigned long long* cnt = (const unsigned long long*)t[0].data;
    const unsigned long long* sum = (const unsigned long long*)t[1].data;
    const unsigned long long* sum2 = (const unsigned long long*)t[2].data;
    const int step = *(const int*)t[3].data;
    if (ref_cnt.empty()) {
      ref_cnt.assign(cnt, cnt + ID_AUDIT_BUCKETS); ref_sum.assign(sum, sum + ID_AUDIT_BUCKETS);
      ref_sum2.assign(sum2, sum2 + ID_AUDIT_BUCKETS);
      CkPrintf("[IDAUDIT] reference taken at step %d\n", step);
    } else {
      int bad = 0;
      for (int b = 0; b < ID_AUDIT_BUCKETS; b++) {
        if (cnt[b] == ref_cnt[b] && sum[b] == ref_sum[b] && sum2[b] == ref_sum2[b]) continue;
        bad++;
        const long long dc = (long long)(cnt[b] - ref_cnt[b]);
        const long long ds = (long long)(sum[b] - ref_sum[b]);
        const long long ds2 = (long long)(sum2[b] - ref_sum2[b]);
        if (dc == 1) { CkPrintf("[IDAUDIT] step %d bucket %d: one extra particle, id %lld (duplicated)\n", step, b, ds); patch_proxy.dumpId((int)ds); }
        else if (dc == -1) { CkPrintf("[IDAUDIT] step %d bucket %d: one particle missing, id %lld (lost)\n", step, b, -ds); patch_proxy.dumpId((int)-ds); }
        else if (dc == 0 && ds != 0 && ds2 % ds == 0) {
          const long long xy = ds2 / ds, x = (xy + ds) / 2, y = (xy - ds) / 2;
          CkPrintf("[IDAUDIT] step %d bucket %d: id %lld duplicated, id %lld lost\n", step, b, x, y);
          patch_proxy.dumpId((int)x); patch_proxy.dumpId((int)y);
        } else CkPrintf("[IDAUDIT] step %d bucket %d: dcount %lld dsum %lld dsum2 %lld (more than one change)\n", step, b, dc, ds, ds2);
        if (bad >= 8) break;
      }
      if (bad == 0) CkPrintf("[IDAUDIT] step %d: identical to the reference\n", step);
      if (corrupt_step == step) {
        CkPrintf("[IDAUDIT] the identity check failed at step %d; exiting in 5 s after the holders printed\n", step);
        CcdCallFnAfter([](void*, double) { CkAbort("CORRUPTION (audited above)\n"); }, nullptr, 5000.0);
      }
    }
    delete[] t; delete msg;
  }

  int watchdog_seen = -1;
  void watchdog() {
    static const int stuck_s = getenv("SPH_STUCK_SECONDS") ? atoi(getenv("SPH_STUCK_SECONDS")) : 0;
    if (watchdog_seen == stat_count && stat_count >= 0 && CkWallTimer() - start_time > stuck_s) {
      CkPrintf("[STUCK] no stats report for %d s after step %d: dumping every patch's state\n", stuck_s, stat_count * stats_freq);
      patch_proxy.dumpState();
      return;   // once
    }
    watchdog_seen = stat_count;
    CcdCallFnAfter([](void* m, double) { ((Main*)m)->thisProxy.watchdog(); }, this, stuck_s * 1000.0);
  }

  void stepStats(CkReductionMsg* msg) {
    int n;
    CkReduction::tupleElement* t;
    msg->toTuple(&t, &n);
    const long sum_np = *(long*)t[0].data;
    const long max_np = *(long*)t[1].data;
    const double sum_rho = *(double*)t[2].data;
    const double sum_ke = *(double*)t[3].data;
    const double n_fluid = *(double*)t[4].data;
    const double vmax = *(double*)t[5].data;
    const double front = *(double*)t[6].data;
    const unsigned long long id_sum  = *(unsigned long long*)t[7].data;
    const unsigned long long bnd_sum = *(unsigned long long*)t[8].data;
    const unsigned long long n_fl    = *(unsigned long long*)t[9].data;
    const unsigned long long n_bd    = *(unsigned long long*)t[10].data;
    const long escaped = *(long*)t[11].data;
    const long n_active = *(long*)t[12].data;
    const long n_skipped = *(long*)t[13].data;
    delete[] t;
    delete msg;

    const int step = (stat_count + 1) * stats_freq;
    checkState(step, sum_np, id_sum, bnd_sum, n_fl, n_bd, escaped);

    if (n_fluid > 0) {
      const long n_patches = (long)n_chares_x * n_chares_y * n_chares_z;
      const double imb = (double)max_np * n_patches / (double)sum_np;
      if (getenv("CHARM_SPH_MEM")) {
        long rss_kb = 0;
        if (FILE* f = fopen("/proc/self/statm", "r")) {
          long sz = 0, res = 0;
          if (fscanf(f, "%ld %ld", &sz, &res) == 2) rss_kb = res * 4;
          fclose(f);
        }
        struct mallinfo2 mi = mallinfo2();
        CkPrintf("  mem: rss %ld MB  heap-in-use %ld MB  mmap %ld MB\n",
                 rss_kb / 1024, (long)(mi.uordblks >> 20), (long)(mi.hblkhd >> 20));
      }
      // Wall time since the start and the mean of the LAST interval, so a
      // trend across the run (or the effect of a balance step) is visible.
      const double now = CkWallTimer() - start_time;
      CkPrintf("  step %6d: rho %.2f  KE %.4e  |v|max %.3f  front x %.3f  "
               "particle imbalance (max/avg) %.2f  active %ld/%ld  [check ok]  "
               "wall %.1f s, %.1f ms/step over the last %d  quiet-skipped %ld msgs\n",
               step, sum_rho / n_fluid, sum_ke * pmass, vmax, front, imb,
               n_active, n_patches, now, (now - last_report_t) * 1000.0 / stats_freq,
               stats_freq, n_skipped);
      last_report_t = now;
    }
    stat_count++;
  }

  void checkState(int step, long sum_np, unsigned long long id_sum,
      unsigned long long bnd_sum, unsigned long long n_fl,
      unsigned long long n_bd, long escaped) {
    if (bnd_sum != ref_bnd_sum || n_bd != ref_n_bound)
      CkAbort("CORRUPTION at step %d: the boundary particles changed. "
              "checksum %016llx (expected %016llx), count %llu (expected "
              "%llu). Boundary particles never move, so this is state "
              "damaged in transit -- the halo/migration path or a chare "
              "migration's device pup.\n",
              step, bnd_sum, ref_bnd_sum, n_bd, ref_n_bound);

    if (escaped > 0) {
      if (!conservation_void) {
        conservation_void = true;
        CkPrintf("[WARN] step %d: %ld particle(s) have left the domain. The "
                 "run is at or past the edge of its validity; particle "
                 "conservation and the identity checksum are no longer "
                 "enforced from here on (the boundary check still is).\n",
                 step, escaped);
      }
      return;
    }

    if (sum_np != n_total_particles || n_fl != ref_n_fluid)
      CkAbort("CORRUPTION at step %d: particles were lost or duplicated. "
              "total %ld (expected %ld), fluid %llu (expected %llu), and "
              "none left the domain.\n",
              step, sum_np, n_total_particles, n_fl, ref_n_fluid);

    if (id_sum != ref_id_sum) {
      if (id_audit_on && corrupt_step < 0) {
        corrupt_step = step;
        CkPrintf("CORRUPTION at step %d: the particle identity checksum changed, %016llx (expected %016llx), "
                 "with the count right (%ld); waiting for the identity audit of this step\n",
                 step, id_sum, ref_id_sum, sum_np);
        return;
      }
      CkAbort("CORRUPTION at step %d: the particle identity checksum changed, "
              "%016llx (expected %016llx), with the count right (%ld).\n",
              step, id_sum, ref_id_sum, sum_np);
    }
  }

  void allDone() {
    const double elapsed = CkWallTimer() - start_time;
    const double per_step = elapsed / n_iters * 1000;
    const int evals = integrator ? 2 : 1;
    CkPrintf("Average iteration time: %.3f ms\n", per_step);
    CkPrintf("  %d force evaluation%s per step: %.3f ms per force evaluation\n",
             evals, evals > 1 ? "s" : "", per_step / evals);
    prof_proxy.report(CkCallback(CkReductionTarget(Main, profDone), main_proxy));
  }
  void profDone() { CkExit(); }

  // hist | n | sum | envelope sum | max, over every PE (HostProfGroup::report).
  void latDone(CkReductionMsg* msg) {
    int nt;
    CkReduction::tupleElement* t;
    msg->toTuple(&t, &nt);
    const int* hist = (const int*)t[0].data;
    const int* ehist = (const int*)t[1].data;
    const long* cnt = (const long*)t[2].data;
    const double* sum = (const double*)t[3].data;
    const double* esum = (const double*)t[4].data;
    const double* mx = (const double*)t[5].data;
    static const char* const kind_name[LAT_KINDS] = {"halo", "migration", "halo2"};
    static const char* const tier_name[3] = {"same process", "same node", "other node"};
    CkPrintf("Message latency, ms: envelope = sent -> arrived on the receiver's PE (the transport);"
             " usable = sent -> payload consumed by the step (includes the receiver's own wait)\n");
    for (int k = 0; k < LAT_KINDS; k++) {
      for (int tier = 0; tier < 3; tier++) {
        const int i = k * 3 + tier;
        if (cnt[i] == 0) continue;
        double pct[3] = {0, 0, 0}, ep[3] = {0, 0, 0};
        const double want[3] = {0.5, 0.9, 0.99};
        long acc = 0, eacc = 0; int w = 0, ew = 0;
        for (int b = 0; b < LAT_BINS; b++) {
          acc += hist[i * LAT_BINS + b];
          eacc += ehist[i * LAT_BINS + b];
          while (w < 3 && acc >= want[w] * cnt[i]) pct[w++] = latBinUpperMs(b);
          while (ew < 3 && eacc >= want[ew] * cnt[i]) ep[ew++] = latBinUpperMs(b);
        }
        CkPrintf("  %-9s %-12s n=%9ld  envelope mean %6.3f p50 %6.3f p90 %6.3f p99 %7.3f | usable mean %7.3f p50 %7.3f p90 %7.3f p99 %8.3f max %8.3f\n",
                 kind_name[k], tier_name[tier], cnt[i], esum[i] / cnt[i] / 1000.0, ep[0], ep[1], ep[2],
                 sum[i] / cnt[i] / 1000.0, pct[0], pct[1], pct[2], mx[i] / 1000.0);
      }
    }
    delete[] t;
    delete msg;
    CkExit();
  }
};

class HostProfGroup : public CBase_HostProfGroup {
 public:
  HostProfGroup() {}
  void report(CkCallback cb) {
    if (host_profile_on) {
      const HostProf& h = g_hprof;
      double total = 0.0;
      for (int i = 0; i < PH_N; i++) total += h.t[i];
      char line[2048]; int o = 0;
      o += snprintf(line + o, sizeof(line) - o, "HOSTPROF pe=%d proc=%d objs=%u total=%.3f |",
                    CkMyPe(), CmiMyNode(), patch_proxy.ckLocMgr()->numLocalElements(), total);
      for (int i = 0; i < PH_N; i++)
        o += snprintf(line + o, sizeof(line) - o, " %s=%.3f/%ld", host_phase_name[i], h.t[i], h.n[i]);
      o += snprintf(line + o, sizeof(line) - o,
                    " | sent intra/node/xnode=%ld/%ld/%ld (MB %.0f/%.0f/%.0f) recv=%ld/%ld/%ld (MB %.0f/%.0f/%.0f)\n",
                    h.sent[0], h.sent[1], h.sent[2], h.sent_bytes[0] / 1e6, h.sent_bytes[1] / 1e6,
                    h.sent_bytes[2] / 1e6, h.recv[0], h.recv[1], h.recv[2], h.recv_bytes[0] / 1e6,
                    h.recv_bytes[1] / 1e6, h.recv_bytes[2] / 1e6);
      CkPrintf("%s", line);
    }
    LatStats& L = g_lat;
    CkReduction::tupleElement tuple[] = {
        CkReduction::tupleElement(sizeof(int) * LAT_KINDS * 3 * LAT_BINS, &L.hist[0][0][0], CkReduction::sum_int),
        CkReduction::tupleElement(sizeof(int) * LAT_KINDS * 3 * LAT_BINS, &L.ehist[0][0][0], CkReduction::sum_int),
        CkReduction::tupleElement(sizeof(long) * LAT_KINDS * 3, &L.n[0][0], CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(double) * LAT_KINDS * 3, &L.sum[0][0], CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double) * LAT_KINDS * 3, &L.esum[0][0], CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double) * LAT_KINDS * 3, &L.mx[0][0], CkReduction::max_double)};
    CkReductionMsg* m = CkReductionMsg::buildFromTuple(tuple, 6);
    m->setCallback(CkCallback(CkIndex_Main::latDone(NULL), main_proxy));
    contribute(m);
  }
};

class Patch : public CBase_Patch {
  Patch_SDAG_CODE

  int x, y, z;                          // my index
  RealType x0, y0, z0, x1, y1, z1;      // my box
  int nbr_x[NUM_DIRS], nbr_y[NUM_DIRS], nbr_z[NUM_DIRS];
  bool valid_dir[NUM_DIRS];             // false for STAY and beyond the tank
  int valid_mask;                       // the same as a bitmask, for the pack

  int np;                         // local particles
  int n_ghost;                    // ghosts appended after them (this window)
  int cur;                        // which particle buffer is live
  int my_iter;
  int recv_count;
  int outstanding_sends;
  bool draining;
  bool halo_pending, halo_pending2, mig_pending;
  bool lb_waiting;
  int lb_start_iter;
  double lb_t0;
  long escaped;

  // ---- activity (see sph2d.C) ---------------------------------------------
  int n_fluid;
  int my_adv, my_adv_next;
  int nbr_adv[NUM_DIRS][ADV_RING];  // advertisement for step t in slot t % ADV_RING
  int xchg_step[NUM_DIRS];          // last step the migration message crossed this edge
  int xchg_min[NUM_DIRS];           // min quiet level of its two ends for the step after
  int my_level_next;                // quiet level advertised for the next step
  long n_skipped;                   // migration messages not sent over quiet edges
  bool step_active;
  bool halo_peer[NUM_DIRS];
  int n_valid;
  int n_expect;
  int n_expect_mig;
  std::map<int, int> halo_arrived, halo_arrived2, parts_arrived;
  int ghost_alloc;
  bool halo_open;                  // ghost slots of d_parts[cur] may be landed in
  bool pc_open;                    // ghost slots of d_parts[1-cur] (predicted) likewise
  std::map<int, int> halo_direct, halo_direct2;   // dir -> slot already landed

  // ---- neighbour-list window (see the top of the file) ---------------------
  bool rebuild_step;               // this step rebuilds lists, layout and cells
  bool migrate_step;               // this step ends with the compaction
  bool pc_step;                    // this step runs the second force evaluation
  bool pc_pending;                 // predicted state in d_parts[1-cur] is live
  bool cells_valid;                // the cell list matches the current layout
  int ghost_off[NUM_DIRS];         // ghost slot of each direction (this window)
  int ghost_cnt[NUM_DIRS];         // its size; -1 until the rebuild assigns it
  bool rebuild_peer[NUM_DIRS];     // the peer set the lists were built for
  int exch_slots;

  // cell list geometry
  int ncx, ncy, ncz, ncells;
  RealType inv_csize;

  Particle* d_parts[2];
  Particle* d_send_halo;          // exchSlots(exch_capacity) particles
  Particle* d_send_halo2;         // the predicted state's exchange
  Particle* d_recv_halo;
  Particle* d_recv_halo2;
  Particle* d_send_mig;
  Particle* d_recv_mig;           // twice that: slot by the sender's step parity
  Particle** d_halo_ptrs;         // device array of the per-direction send bases
  Particle** d_halo_ptrs2;
  Particle** d_mig_ptrs;
  int* d_halo_idx;                // per-direction lists of what the rebuild sent
  int** d_halo_idx_ptrs;
  int* d_halo_cnt;                // their lengths, on the device
  int* d_counts;
  int* d_cell_cnt;
  int* d_cell_off;
  int* d_cursor;
  int* d_cell_parts;
  int* d_cell_of;
  float4* d_pos4;                 // neighbour data in cell order (layoutKernel)
  float4* d_vel4;
  unsigned char* d_type8;
  void* d_scan_tmp;               // CUB scan temporaries
  size_t scan_tmp_bytes;
  RealType *d_drho, *d_ax, *d_ay, *d_az, *d_stats;
  unsigned long long* d_check;
  int* h_counts;
  int* h_halo_cnt;                // the halo counts of the current window
  RealType* h_stats;
  unsigned long long* h_check;

  cudaStream_t compute_stream, comm_stream;
  cudaEvent_t halo_done;

  // ---- interior-first forces (SPH_INTERIOR_FIRST=1, old protocol) ---------
  // At the end of a step the survivors' interior forces (cells 3..nc-2, no
  // dependency on any neighbour) are computed over a locals-only cell list
  // while the migration round and the next halo are in flight; the next
  // step's forces then run the boundary layers only, over the full list.
  bool interior_done;
  int interior_np;                // the survivors the locals' list L was built over
  // The interior pass runs on its own normal-priority stream while the
  // pack / boundary / integrate chain runs on compute_stream at the device's
  // highest priority: a neighbour's halo pack queued behind its 0.85 ms
  // interior kernel arrived late enough to cancel the overlap (job 22801886).
  // v4: the forces (interior, then boundary) and the integrate stay serialized on
  // compute_stream, one patch kernel at a time on the GPU as in the plain
  // schedule (v3 ran interior and boundary kernels of different patches
  // concurrently, 1.57 in flight, and the cache-bound forces lost what the
  // idle gave); only the halo pack goes on pack_stream at the device's highest
  // priority, so it is never queued behind an interior kernel. The integrate
  // waits for the pack's read of the positions (pack_ev).
  cudaStream_t pack_stream;
  cudaEvent_t pack_ev;            // the pack has read the positions
  cudaEvent_t compact_ev;         // the compaction is complete (the pack waits on it)
  int *d_cell_cnt2, *d_cell_off2, *d_cursor2, *d_cell_parts2, *d_cell_of2;   // the extras list E
  int* d_iota; unsigned char* d_sel_flags; int* d_sel; int* d_sel_num; void* d_sel_tmp; size_t sel_tmp_bytes;   // target selection
  int *d_item_beg, *d_item_pre;   // the rows form's table

  // ---- concurrent exchange (-c; see the top of the file) --------------------
  int* d_counts_mig;              // the compaction's counters (d_counts is the halo pack's)
  int* h_counts_mig;
  Particle* d_leaver_ghosts;      // this step's leavers, this patch's ghosts next step
  int n_leaver_ghosts;
  bool lg_pending;                // they are only in the side buffer so far
  int open_ref;                   // the step whose exchange may land now
  int np_base;                    // survivors before that step's migrants
  bool mig_known;                 // every migration envelope of that step has arrived
  int ghost_base;                 // where that step's ghosts start (np_base + its migrants), frozen when mig_known:
                                  // the per-step totals are erased at the merge, and a halo landing directly after
                                  // that computed its base from a zero and overwrote the migrants (job 22799507)
  bool prologue;                  // step 1's exchange is being packed from the initial state
  bool xpack_pending;             // compaction + pack submitted, exchange not yet sent
  int n_expect_mig_cur;           // this step's expectation (n_expect_mig moves on at the send)
  bool xsend_pending;             // exchange packed and the step opened, sends deferred past a park
  int xsend_ref, xsend_adv;       // what the deferred sends carry
  bool xsend_halo;
  unsigned long long pup_id_sum, pup_n;   // SPH_PUP_CHECK: this patch's locals before the pack, re-checked after the unpack
  bool pup_check_due;                     // landed: compare at the first step boundary (the device unpack is asynchronous)
  unsigned long long *d_aud_cnt, *d_aud_sum, *d_aud_sum2, *h_aud;   // SPH_ID_AUDIT buffers
  int* d_find;
  std::map<int, int> mig_env_n, mig_env_total;             // per step: envelopes, particles
  std::map<int, int> mig_landed, halo_landed;              // per step: directions whose payload has landed (the plain
                                                           // entry bodies set them; a replayed body is a no-op)
  std::map<int, int> done_sent;                            // per step: bit 0 migrantsDone sent, bit 1 halosDone sent
  bool halo_expect_known;                                  // this step's halo expectation is set (after the migrants)
  bool parked_midstep, sync_park_due;                      // the LB park sits after the receives, before the forces
  std::map<int, int> mig_consumed, halo_consumed;          // (kept for the pup layout; unused by the signal scheme)
                                                           // a replayed message is a second message object next to the
                                                           // one pupped with the element (job 22799507, step 625: six
                                                           // migrants appended twice, the real message left unconsumed)
  std::map<int, int> mig_seen, halo_seen;                  // per step: directions whose hook already ran (bitmask):
                                                           // the runtime re-invokes a landing hook when it bounces and
                                                           // replays a device receive around a migration (job 22799507:
                                                           // 42 particles announced, 21 appended, step 701)
  std::map<int, std::map<int, int>> halo_staged, mig_staged;   // per step: dir -> staged count
  bool prio_dirty, prio_now;      // SPH_PRIO_BORDER: border patches at high stream priority
  long long env_t[LAT_KINDS][2][NUM_DIRS];   // envelope arrival: kind, step parity, direction

public:
  Patch() {
    usesAtSync = true;
    setupGeometry();
    setupState();
    createCudaEntities();
    allocDevice();
  }

  Patch(CkMigrateMessage* m) : CBase_Patch(m) {
    usesAtSync = true;
    setupState();
    createCudaEntities();
  }

  ~Patch() {
    freeDevice();
    hapiReleaseStream(compute_stream);
    hapiReleaseStream(comm_stream);
    if (pack_stream != NULL) { hapiReleaseStream(pack_stream); eventPool().give(pack_ev); eventPool().give(compact_ev); }
    eventPool().give(halo_done);
  }

  // Everything derived from the array index; recomputed on the destination
  // after a migration (thisIndex is not reliable in the migration ctor).
  void setupGeometry() {
    x = thisIndex.x; y = thisIndex.y; z = thisIndex.z;
    const RealType px = dom_lx / n_chares_x, py = dom_ly / n_chares_y,
                   pz = dom_lz / n_chares_z;
    x0 = x * px; x1 = (x + 1) * px;
    y0 = y * py; y1 = (y + 1) * py;
    z0 = z * pz; z1 = (z + 1) * pz;

    for (int d = 0; d < NUM_DIRS; d++) {
      const int nx = x + dirSX(d), ny = y + dirSY(d), nz = z + dirSZ(d);
      // Wrapped so every direction names a partner; one across the tank wall
      // (or STAY) is invalid and is sent nothing.
      valid_dir[d] = d != STAY && nx >= 0 && nx < n_chares_x &&
                     ny >= 0 && ny < n_chares_y && nz >= 0 && nz < n_chares_z;
      nbr_x[d] = (nx + n_chares_x) % n_chares_x;
      nbr_y[d] = (ny + n_chares_y) % n_chares_y;
      nbr_z[d] = (nz + n_chares_z) % n_chares_z;
    }
    n_valid = 0; valid_mask = 0;
    for (int d = 0; d < NUM_DIRS; d++)
      if (valid_dir[d]) { n_valid++; valid_mask |= 1 << d; }

    // Cells of cell_size (the support plus the Verlet margin): a particle's
    // neighbours lie in its own cell and the 26 around it.
    ncx = std::max(1, (int)std::ceil((x1 - x0) / cell_size));
    ncy = std::max(1, (int)std::ceil((y1 - y0) / cell_size));
    ncz = std::max(1, (int)std::ceil((z1 - z0) / cell_size));
    ncells = (ncx + 2) * (ncy + 2) * (ncz + 2);
    inv_csize = 1.0f / cell_size;
    exch_slots = (int)exchSlots(exch_capacity);
  }

  void setupState() {
    np = n_ghost = 0; cur = 0; my_iter = 0;
    outstanding_sends = 0; draining = false;
    halo_pending = halo_pending2 = mig_pending = false;
    lb_waiting = false; lb_start_iter = 0; escaped = 0;
    n_fluid = 0; step_active = false; n_expect = 0; n_expect_mig = 0;
    my_adv = my_adv_next = 0;
    my_level_next = 0; n_skipped = 0;
    for (int d = 0; d < NUM_DIRS; d++) { xchg_step[d] = 0; xchg_min[d] = 0; }
    ghost_alloc = 0; halo_open = pc_open = false;
    rebuild_step = migrate_step = pc_step = pc_pending = false;
    cells_valid = false;
    for (int d = 0; d < NUM_DIRS; d++) {
      nbr_adv[d][0] = nbr_adv[d][1] = 0;
      halo_peer[d] = false;
      rebuild_peer[d] = false;
      ghost_off[d] = 0; ghost_cnt[d] = -1;
    }
    for (int i = 0; i < 2; i++) d_parts[i] = NULL;
    d_send_halo = d_send_halo2 = d_recv_halo = d_recv_halo2 = NULL;
    d_send_mig = d_recv_mig = NULL;
    d_halo_ptrs = d_halo_ptrs2 = d_mig_ptrs = NULL;
    d_halo_idx = NULL; d_halo_idx_ptrs = NULL; d_halo_cnt = NULL;
    d_counts = d_cell_cnt = d_cell_off = d_cursor = d_cell_parts = d_cell_of = NULL;
    d_pos4 = d_vel4 = NULL; d_type8 = NULL; d_scan_tmp = NULL; scan_tmp_bytes = 0;
    d_drho = d_ax = d_ay = d_az = d_stats = NULL;
    d_check = NULL;
    h_counts = NULL; h_halo_cnt = NULL; h_stats = NULL; h_check = NULL;
    d_counts_mig = NULL; h_counts_mig = NULL; d_leaver_ghosts = NULL;
    n_leaver_ghosts = 0; lg_pending = false; open_ref = 0; np_base = 0;
    mig_known = false; ghost_base = 0; prologue = false; xpack_pending = false; n_expect_mig_cur = 0;
    xsend_pending = false; xsend_ref = 0; xsend_adv = 0; xsend_halo = false;
    halo_expect_known = false; parked_midstep = false; sync_park_due = false;
    pup_id_sum = 0; pup_n = 0; pup_check_due = false;
    d_aud_cnt = d_aud_sum = d_aud_sum2 = NULL; h_aud = NULL; d_find = NULL;
    prio_dirty = true; prio_now = false;
    interior_done = false; interior_np = 0;
    d_cell_cnt2 = d_cell_off2 = d_cursor2 = d_cell_parts2 = d_cell_of2 = NULL;
    d_iota = NULL; d_sel_flags = NULL; d_sel = NULL; d_sel_num = NULL; d_sel_tmp = NULL; sel_tmp_bytes = 0;
    d_item_beg = NULL; d_item_pre = NULL;
    for (int k = 0; k < LAT_KINDS; k++)
      for (int par = 0; par < 2; par++)
        for (int d = 0; d < NUM_DIRS; d++) env_t[k][par][d] = 0;
  }

  static int interiorLayers() {
    static const int layers = getenv("SPH_INTERIOR_LAYERS") ? std::max(1, atoi(getenv("SPH_INTERIOR_LAYERS"))) : 2;
    return layers;
  }
  // SPH_INTERIOR_ROWS=1: the passes map threads to targets through a row table (no flags, no select).
  static bool interiorRows() {
    static const bool rows = (getenv("SPH_INTERIOR_ROWS") != nullptr);
    return rows;
  }
  static bool interiorFirstOn() {
    static const bool on = (getenv("SPH_INTERIOR_FIRST") != nullptr);
    static bool layers_set = false;
    if (on && !layers_set) {
      layers_set = true;
      const int layers = interiorLayers();
      sphSetInteriorLayers(layers);
      if (CkMyPe() == 0 && interiorRows()) CkPrintf("  interior-first: rows form (row tables, no select)\n");
      if (CkMyPe() == 0) CkPrintf("  interior-first: %d boundary layer(s) per side%s\n", layers, layers == 1 ? " (TIMING EXPERIMENT: cell-2 targets miss the migrants)" : "");
    }
    return on;
  }
  void createCudaEntities() {
    compute_stream = hapiAcquireStream();
    if (interiorFirstOn()) {
      pack_stream = hapiAcquirePriorityStream();
      pack_ev = eventPool().take();
      compact_ev = eventPool().take();
    } else {
      pack_stream = NULL; pack_ev = compact_ev = NULL;
    }
    comm_stream = hapiAcquireStream();
    halo_done = eventPool().take();
  }

  void allocDevice() {
    const size_t pcap = sizeof(Particle) * (size_t)part_capacity;
    const size_t ecap = sizeof(Particle) * (size_t)exch_slots;
    hapiCheck(hapiMalloc((void**)&d_parts[0], pcap));
    hapiCheck(hapiMalloc((void**)&d_parts[1], pcap));
    hapiCheck(hapiMalloc((void**)&d_send_halo, ecap));
    hapiCheck(hapiMalloc((void**)&d_send_halo2, ecap));
    hapiCheck(hapiMalloc((void**)&d_recv_halo, 2 * ecap));   // two parities (-c)
    hapiCheck(hapiMalloc((void**)&d_recv_halo2, ecap));
    hapiCheck(hapiMalloc((void**)&d_leaver_ghosts, ecap));
    hapiCheck(hapiMalloc((void**)&d_counts_mig, sizeof(int) * NUM_COUNTERS));
    if (id_audit_on) {
      hapiCheck(hapiMalloc((void**)&d_aud_cnt, sizeof(unsigned long long) * ID_AUDIT_BUCKETS));
      hapiCheck(hapiMalloc((void**)&d_aud_sum, sizeof(unsigned long long) * ID_AUDIT_BUCKETS));
      hapiCheck(hapiMalloc((void**)&d_aud_sum2, sizeof(unsigned long long) * ID_AUDIT_BUCKETS));
      hapiCheck(hapiMalloc((void**)&d_find, sizeof(int)));
      h_aud = takePinned<unsigned long long>(3 * ID_AUDIT_BUCKETS);
    }
    hapiCheck(hapiMalloc((void**)&d_send_mig, ecap));
    hapiCheck(hapiMalloc((void**)&d_recv_mig, 2 * ecap));
    hapiCheck(hapiMalloc((void**)&d_halo_ptrs, sizeof(Particle*) * NUM_DIRS));
    hapiCheck(hapiMalloc((void**)&d_halo_ptrs2, sizeof(Particle*) * NUM_DIRS));
    hapiCheck(hapiMalloc((void**)&d_mig_ptrs, sizeof(Particle*) * NUM_DIRS));
    hapiCheck(hapiMalloc((void**)&d_halo_idx, sizeof(int) * (size_t)exch_slots));
    hapiCheck(hapiMalloc((void**)&d_halo_idx_ptrs, sizeof(int*) * NUM_DIRS));
    hapiCheck(hapiMalloc((void**)&d_halo_cnt, sizeof(int) * NUM_DIRS));
    hapiCheck(hapiMalloc((void**)&d_counts, sizeof(int) * NUM_COUNTERS));
    hapiCheck(hapiMalloc((void**)&d_cell_cnt, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cell_off, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cursor, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cell_parts, sizeof(int) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_cell_of, sizeof(int) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_cell_cnt2, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cell_off2, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cursor2, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cell_parts2, sizeof(int) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_cell_of2, sizeof(int) * part_capacity));
    if (interiorFirstOn()) {
      hapiCheck(hapiMalloc((void**)&d_iota, sizeof(int) * part_capacity));
      hapiCheck(hapiMalloc((void**)&d_sel_flags, (size_t)part_capacity));
      hapiCheck(hapiMalloc((void**)&d_sel, sizeof(int) * part_capacity));
      hapiCheck(hapiMalloc((void**)&d_sel_num, sizeof(int)));
      sel_tmp_bytes = sphSelectTempBytes(part_capacity);
      hapiCheck(hapiMalloc((void**)&d_sel_tmp, std::max(sel_tmp_bytes, (size_t)256)));
      sphIota(part_capacity, d_iota, compute_stream);
      const int max_items = 3 * (ncy + 2) * (ncz + 2) + 2;
      hapiCheck(hapiMalloc((void**)&d_item_beg, sizeof(int) * max_items));
      hapiCheck(hapiMalloc((void**)&d_item_pre, sizeof(int) * max_items));
    }
    hapiCheck(hapiMalloc((void**)&d_pos4, sizeof(float4) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_vel4, sizeof(float4) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_type8, (size_t)part_capacity));
    scan_tmp_bytes = sphScanTempBytes(ncells);
    hapiCheck(hapiMalloc((void**)&d_scan_tmp, std::max(scan_tmp_bytes, (size_t)256)));
    hapiCheck(hapiMalloc((void**)&d_drho, sizeof(RealType) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_ax, sizeof(RealType) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_ay, sizeof(RealType) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_az, sizeof(RealType) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_stats, sizeof(RealType) * 8));
    hapiCheck(hapiMalloc((void**)&d_check,
        sizeof(unsigned long long) * NUM_CHECKS));
    h_counts = takePinned<int>(NUM_COUNTERS);
    h_counts_mig = takePinned<int>(NUM_COUNTERS);
    h_halo_cnt = takePinned<int>(NUM_DIRS);
    h_stats = takePinned<RealType>(8);
    h_check = takePinned<unsigned long long>(NUM_CHECKS);
    for (int i = 0; i < 8; i++) h_stats[i] = 0;
    for (int i = 0; i < NUM_CHECKS; i++) h_check[i] = 0;
    for (int i = 0; i < NUM_COUNTERS; i++) { h_counts[i] = 0; h_counts_mig[i] = 0; }
    for (int i = 0; i < NUM_DIRS; i++) h_halo_cnt[i] = 0;

    // Per-direction bases (STAY's entry is unused). Written on compute_stream,
    // which is where the pack kernels read them.
    Particle* hp[NUM_DIRS];
    for (int d = 0; d < NUM_DIRS; d++)
      hp[d] = d == STAY ? NULL : d_send_halo + dirOffset(d, exch_capacity);
    sphSubmitMemcpy(d_halo_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice,
        compute_stream);
    for (int d = 0; d < NUM_DIRS; d++)
      hp[d] = d == STAY ? NULL : d_send_halo2 + dirOffset(d, exch_capacity);
    sphSubmitMemcpy(d_halo_ptrs2, hp, sizeof(hp), cudaMemcpyHostToDevice,
        compute_stream);
    for (int d = 0; d < NUM_DIRS; d++)
      hp[d] = d == STAY ? NULL : d_send_mig + dirOffset(d, exch_capacity);
    sphSubmitMemcpy(d_mig_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice,
        compute_stream);
    int* hi[NUM_DIRS];
    for (int d = 0; d < NUM_DIRS; d++)
      hi[d] = d == STAY ? NULL : d_halo_idx + dirOffset(d, exch_capacity);
    sphSubmitMemcpy(d_halo_idx_ptrs, hi, sizeof(hi), cudaMemcpyHostToDevice,
        compute_stream);
    sphSubmitMemcpy(d_halo_cnt, h_halo_cnt, sizeof(int) * NUM_DIRS,
        cudaMemcpyHostToDevice, compute_stream);
    hapiCheck(sphDrainSync(compute_stream));
  }

  void freeDevice() {
    hapiFree(d_parts[0]); hapiFree(d_parts[1]);
    hapiFree(d_send_halo); hapiFree(d_send_halo2);
    hapiFree(d_recv_halo); hapiFree(d_recv_halo2);
    hapiFree(d_send_mig); hapiFree(d_recv_mig);
    hapiFree(d_halo_ptrs); hapiFree(d_halo_ptrs2); hapiFree(d_mig_ptrs);
    hapiFree(d_halo_idx); hapiFree(d_halo_idx_ptrs); hapiFree(d_halo_cnt);
    hapiFree(d_counts); hapiFree(d_cell_cnt); hapiFree(d_cell_off);
    hapiFree(d_cursor); hapiFree(d_cell_parts); hapiFree(d_cell_of);
    hapiFree(d_cell_cnt2); hapiFree(d_cell_off2); hapiFree(d_cursor2); hapiFree(d_cell_parts2); hapiFree(d_cell_of2);
    d_cell_cnt2 = d_cell_off2 = d_cursor2 = d_cell_parts2 = d_cell_of2 = NULL;
    if (d_sel_flags) { hapiFree(d_iota); hapiFree(d_sel_flags); hapiFree(d_sel); hapiFree(d_sel_num); hapiFree(d_sel_tmp); hapiFree(d_item_beg); hapiFree(d_item_pre); }
    d_iota = NULL; d_sel_flags = NULL; d_sel = NULL; d_sel_num = NULL; d_sel_tmp = NULL; d_item_beg = NULL; d_item_pre = NULL;
    hapiFree(d_pos4); hapiFree(d_vel4); hapiFree(d_type8); hapiFree(d_scan_tmp);
    hapiFree(d_drho); hapiFree(d_ax); hapiFree(d_ay); hapiFree(d_az);
    hapiFree(d_stats); hapiFree(d_check);
    hapiFree(d_leaver_ghosts); hapiFree(d_counts_mig);
    d_leaver_ghosts = NULL; d_counts_mig = NULL;
    if (id_audit_on) {
      hapiFree(d_aud_cnt); hapiFree(d_aud_sum); hapiFree(d_aud_sum2); hapiFree(d_find);
      givePinned(h_aud, 3 * ID_AUDIT_BUCKETS);
      d_aud_cnt = d_aud_sum = d_aud_sum2 = NULL; d_find = NULL; h_aud = NULL;
    }
    givePinned(h_counts, NUM_COUNTERS);
    givePinned(h_counts_mig, NUM_COUNTERS); h_counts_mig = NULL;
    givePinned(h_halo_cnt, NUM_DIRS);
    givePinned(h_stats, 8);
    givePinned(h_check, NUM_CHECKS);
    for (int i = 0; i < 2; i++) d_parts[i] = NULL;
    d_send_halo = d_send_halo2 = d_recv_halo = d_recv_halo2 = NULL;
    d_send_mig = d_recv_mig = NULL;
    d_halo_ptrs = d_halo_ptrs2 = d_mig_ptrs = NULL;
    d_halo_idx = NULL; d_halo_idx_ptrs = NULL; d_halo_cnt = NULL;
    d_counts = d_cell_cnt = d_cell_off = d_cursor = d_cell_parts = d_cell_of = NULL;
    d_pos4 = d_vel4 = NULL; d_type8 = NULL; d_scan_tmp = NULL; scan_tmp_bytes = 0;
    d_drho = d_ax = d_ay = d_az = d_stats = NULL;
    d_check = NULL;
    h_counts = NULL; h_halo_cnt = NULL; h_stats = NULL; h_check = NULL;
  }

  // The halo counts that go with the index lists: at the boundary between a
  // rebuild step's pack and its send they are still only in h_counts (sendHalo
  // copies them into h_halo_cnt), everywhere else h_halo_cnt.
  const int* haloCounts() const {
    return (rebuild_step && halo_pending) ? h_counts : h_halo_cnt;
  }

  // Migration: what is live at the worst entry-method boundary (see sph2d.C).
  // New with the window and the predictor-corrector: the index lists and the
  // ghost layout (needed until the next rebuild), the predicted buffer while
  // it is live, the second exchange's unsent pack. The cell list is NOT
  // carried: it is rebuilt locally from the current positions, which is a
  // valid (symmetric) list for the same layout.
  void pup(PUP::er& p) {
    CBase_Patch::pup(p);
    p | my_iter; p | np; p | cur; p | lb_waiting; p | lb_start_iter; p | lb_t0;
    p | escaped;
    p | n_fluid; p | step_active; p | n_valid; p | n_expect; p | n_expect_mig;
    p | my_adv; p | my_adv_next;
    for (int d = 0; d < NUM_DIRS; d++) PUParray(p, nbr_adv[d], ADV_RING);
    PUParray(p, xchg_step, NUM_DIRS);
    PUParray(p, xchg_min, NUM_DIRS);
    p | my_level_next; p | n_skipped;
    PUParray(p, halo_peer, NUM_DIRS);
    PUParray(p, rebuild_peer, NUM_DIRS);
    PUParray(p, ghost_off, NUM_DIRS);
    PUParray(p, ghost_cnt, NUM_DIRS);
    p | halo_arrived; p | halo_arrived2; p | parts_arrived;
    p | n_ghost;
    p | ghost_alloc; p | halo_open; p | pc_open; p | halo_direct; p | halo_direct2;
    p | rebuild_step; p | migrate_step; p | pc_step; p | pc_pending;
    p | n_leaver_ghosts; p | lg_pending; p | open_ref; p | np_base; p | mig_known; p | ghost_base;
    p | prologue; p | xpack_pending; p | n_expect_mig_cur; p | mig_env_n; p | mig_env_total;
    p | xsend_pending; p | xsend_ref; p | xsend_adv; p | xsend_halo;
    p | interior_done; p | interior_np;
    p | pup_check_due;
    p | halo_staged; p | mig_staged; p | mig_seen; p | halo_seen; p | mig_consumed; p | halo_consumed;
    p | mig_landed; p | halo_landed; p | done_sent; p | halo_expect_known; p | parked_midstep; p | sync_park_due;
    if (p.isUnpacking()) { prio_dirty = true; interior_done = false; }
    _sdag_pup(p);
    p | recv_count;
    p | outstanding_sends;
    p | draining;
    p | halo_pending; p | halo_pending2; p | mig_pending;

    static const bool pup_check = (getenv("SPH_PUP_CHECK") != nullptr);
    if (!p.isUnpacking()) {
      sphDrainSync(compute_stream);
      sphDrainSync(comm_stream);
      if (pack_stream != NULL) sphDrainSync(pack_stream);
      if (pup_check) {
        // The locals' identity before the pack (ghosts excluded), carried
        // across and re-checked right after the unpack: a corruption inside
        // the migration path is then caught at the patch it happened to.
        check_ghosts = false;
        runCheck(1e30f);
        hapiCheck(sphDrainSync(compute_stream));
        pup_id_sum = h_check[CHK_ID_SUM];
        pup_n = h_check[CHK_N_FLUID] + h_check[CHK_N_BOUND];
      }
    }
    p | pup_id_sum; p | pup_n;
    if (p.isUnpacking()) {
      setupGeometry();
      allocDevice();
      cells_valid = false;
    }

    PUParray(p, h_counts, NUM_COUNTERS);
    PUParray(p, h_counts_mig, NUM_COUNTERS);
    PUParray(p, h_halo_cnt, NUM_DIRS);
    PUParray(p, h_stats, 8);
    PUParray(p, h_check, NUM_CHECKS);
    if (p.isUnpacking())
      sphSubmitMemcpy(d_halo_cnt, h_halo_cnt, sizeof(int) * NUM_DIRS,
          cudaMemcpyHostToDevice, compute_stream);

    const size_t per = sizeof(Particle) / sizeof(RealType);
    if (mig_pending) {
      // The compaction has already run: the live particles are the survivors
      // in the other buffer, h_counts[STAY] of them.
      p((RealType*)d_parts[1 - cur],
        (size_t)std::max(h_counts[STAY], 0) * per, PUP::PUPMode::DEVICE);
    } else if (concurrent_x) {
      // Locals so far, the slots the rest of this step's migrants will fill,
      // the leaver ghosts and the halos landed directly behind them.
      // (find, not operator[]: an insertion here would make the sizing and
      // packing passes disagree by one map entry)
      auto mt = mig_env_total.find(open_ref);
      const int live = std::max(np, np_base + (mt == mig_env_total.end() ? 0 : mt->second)) + ghost_alloc;
      p((RealType*)d_parts[cur], (size_t)live * per, PUP::PUPMode::DEVICE);
    } else {
      p((RealType*)d_parts[cur], (size_t)(np + std::max(n_ghost, ghost_alloc)) * per,
        PUP::PUPMode::DEVICE);
      // The predicted state, locals and the ghost slots (some landed, the
      // rest about to be), between the predictor and the corrector.
      if (pc_pending)
        p((RealType*)d_parts[1 - cur], (size_t)(np + n_ghost) * per,
          PUP::PUPMode::DEVICE);
    }

    // The index lists, for the rest of the window (and the second exchange).
    if (nl_freq > 1 || integrator == 1) {
      const int* hc = haloCounts();
      for (int d = 0; d < NUM_DIRS; d++) {
        if (d == STAY) continue;
        const size_t cnt = (size_t)std::max(hc[d], 0);
        if (cnt == 0) continue;
        p(d_halo_idx + dirOffset(d, exch_capacity), cnt, PUP::PUPMode::DEVICE);
      }
    }

    if (concurrent_x) {
      // Staged exchange slots not yet consumed: a partner's messages that
      // landed before their step opened or before its migrants were known.
      for (auto& st : halo_staged)
        for (auto& dn : st.second)
          if (dn.second > 0)
            p((RealType*)(d_recv_halo + (size_t)(st.first & 1) * exch_slots + dirOffset(dn.first, exch_capacity)),
              (size_t)dn.second * per, PUP::PUPMode::DEVICE);
      for (auto& st : mig_staged)
        for (auto& dn : st.second)
          if (dn.second > 0)
            p((RealType*)(d_recv_mig + (size_t)(st.first & 1) * exch_slots + dirOffset(dn.first, exch_capacity)),
              (size_t)dn.second * per, PUP::PUPMode::DEVICE);
      if (lg_pending && n_leaver_ghosts > 0)
        p((RealType*)d_leaver_ghosts, (size_t)n_leaver_ghosts * per, PUP::PUPMode::DEVICE);
      if (xpack_pending || xsend_pending) {
        // Compaction and pack done, exchange not sent: the survivors and the
        // leaver ghosts behind them (not swapped in yet), and the send slots.
        // Deferred sends (xsend_pending): the swap has happened, the live
        // range above has the survivors; only the send slots are owed.
        int L = 0;
        for (int d = 0; d < NUM_DIRS; d++) if (d != STAY && valid_dir[d]) L += h_counts_mig[d];
        if (xpack_pending && !prologue) {
          const int S = h_counts_mig[STAY];
          p((RealType*)d_parts[1 - cur], (size_t)(S + L) * per, PUP::PUPMode::DEVICE);
          // The side copy too (the gather wrote both): restoring it with a
          // D2D copy on compute_stream could run before the runtime's device
          // unpack of the line above has landed.
          if (L > 0) p((RealType*)d_leaver_ghosts, (size_t)L * per, PUP::PUPMode::DEVICE);
        }
        for (int d = 0; d < NUM_DIRS; d++) {
          if (d == STAY || !valid_dir[d]) continue;
          if (h_counts_mig[d] > 0)
            p((RealType*)(d_send_mig + dirOffset(d, exch_capacity)), (size_t)h_counts_mig[d] * per, PUP::PUPMode::DEVICE);
          if (h_counts[d] > 0)
            p((RealType*)(d_send_halo + dirOffset(d, exch_capacity)), (size_t)h_counts[d] * per, PUP::PUPMode::DEVICE);
        }
      }
    }

    // A packed-but-unsent send buffer, each direction's filled prefix only.
    if (halo_pending || halo_pending2 || mig_pending) {
      Particle* base = halo_pending ? d_send_halo
                     : halo_pending2 ? d_send_halo2 : d_send_mig;
      const int* hc = mig_pending ? h_counts
                    : halo_pending ? haloCounts() : h_halo_cnt;
      for (int d = 0; d < NUM_DIRS; d++) {
        if (d == STAY) continue;
        const size_t cnt = (size_t)std::max(hc[d], 0);
        if (cnt == 0) continue;
        p((RealType*)(base + dirOffset(d, exch_capacity)), cnt * per,
          PUP::PUPMode::DEVICE);
      }
    }
    // The device unpack lands asynchronously; the comparison runs at the first
    // step boundary on the destination (pupCheckLanded), not here.
    if (pup_check && p.isUnpacking() && !mig_pending && !xpack_pending) pup_check_due = true;
  }

  void pupCheckLanded(const char* where) {
    if (!pup_check_due) return;
    pup_check_due = false;
    hapiSubmitDrain();
    hapiCheck(cudaDeviceSynchronize());
    check_ghosts = false;
    runCheck(1e30f);
    hapiCheck(sphDrainSync(compute_stream));
    const unsigned long long n_now = h_check[CHK_N_FLUID] + h_check[CHK_N_BOUND];
    if (h_check[CHK_ID_SUM] != pup_id_sum || n_now != pup_n)
      CkAbort("PUP CORRUPTION: patch (%d,%d,%d) on PE %d at %s step %d: id sum %016llx / %llu locals, "
              "packed with %016llx / %llu (np %d base %d ghost_alloc %d n_ghost %d open_ref %d mig_known %d "
              "xpack %d xsend %d lg_pending %d prologue %d staged halos %zu migs %zu direct %zu)\n",
              x, y, z, CkMyPe(), where, my_iter, h_check[CHK_ID_SUM], n_now, pup_id_sum, pup_n, np, np_base,
              ghost_alloc, n_ghost, open_ref, (int)mig_known, (int)xpack_pending, (int)xsend_pending,
              (int)lg_pending, (int)prologue, halo_staged.size(), mig_staged.size(), halo_direct.size());
  }

  // Lattice sites at (i + 1/2) spacing; wall, column or air by position.
  static inline bool siteIsWall(RealType px, RealType py, RealType pz) {
    return (px < wall_t) || (px > dom_lx - wall_t) ||
           (py < wall_t) || (py > dom_ly - wall_t) || (pz < wall_t);
  }
  static inline bool siteInColumn(RealType px, RealType py, RealType pz) {
    return !siteIsWall(px, py, pz) && px >= wall_t && px < wall_t + col_w &&
           pz >= wall_t && pz < wall_t + col_h;
  }

  // Lay down the lattice, keeping only what falls in my box. O(local).
  void init() {
    const RealType s = spacing;
    const int ix_lo = std::max(0, (int)std::floor(x0 / s) - 1);
    const int ix_hi = (int)std::ceil(x1 / s) + 1;
    const int iy_lo = std::max(0, (int)std::floor(y0 / s) - 1);
    const int iy_hi = (int)std::ceil(y1 / s) + 1;
    const int iz_lo = std::max(0, (int)std::floor(z0 / s) - 1);
    const int iz_hi = (int)std::ceil(z1 / s) + 1;

    std::vector<Particle> mine;
    for (int iz = iz_lo; iz <= iz_hi; iz++) {
      const RealType pz = (iz + 0.5f) * s;
      if (pz >= dom_lz || pz < z0 || pz >= z1) continue;
      for (int iy = iy_lo; iy <= iy_hi; iy++) {
        const RealType py = (iy + 0.5f) * s;
        if (py >= dom_ly || py < y0 || py >= y1) continue;
        for (int ix = ix_lo; ix <= ix_hi; ix++) {
          const RealType px = (ix + 0.5f) * s;
          if (px >= dom_lx || px < x0 || px >= x1) continue;

          const bool is_wall = siteIsWall(px, py, pz);
          const bool in_col = siteInColumn(px, py, pz);
          if (!is_wall && !in_col) continue;   // air

          Particle q;
          q.x = px; q.y = py; q.z = pz;
          q.vx = is_wall ? 0.0f : init_vx; q.vy = 0.0f; q.vz = 0.0f;
          q.rho = rho0; q.p = 0.0f;
          q.type = is_wall ? PTYPE_BOUND : PTYPE_FLUID;
          q.id = (iz * lattice_ny + iy) * lattice_nx + ix;
          mine.push_back(q);
        }
      }
    }

    np = (int)mine.size();
    if (np > part_capacity)
      CkAbort("Patch (%d,%d,%d): %d initial particles exceed capacity %d; "
              "increase headroom (-r)\n", x, y, z, np, part_capacity);
    n_fluid = 0;
    for (const Particle& q : mine) if (q.type == PTYPE_FLUID) n_fluid++;
    seedActivity();
    if (np > 0)
      sphSubmitMemcpy(d_parts[cur], mine.data(),
          sizeof(Particle) * np, cudaMemcpyHostToDevice, compute_stream);
    // Pressures once here; from now on every integrator sets them.
    invokeEOS(d_parts[cur], np, rho0, sound_c0, NULL, compute_stream);

    runCheck(0.0f);
    hapiCheck(sphDrainSync(compute_stream));
    abortOnBadParticles(0);

    long n = np;
    CkReduction::tupleElement tuple[] = {
        CkReduction::tupleElement(sizeof(long), &n, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(unsigned long long),
            &h_check[CHK_ID_SUM], CkReduction::sum_ulong_long),
        CkReduction::tupleElement(sizeof(unsigned long long),
            &h_check[CHK_BND_SUM], CkReduction::sum_ulong_long),
        CkReduction::tupleElement(sizeof(unsigned long long),
            &h_check[CHK_N_FLUID], CkReduction::sum_ulong_long),
        CkReduction::tupleElement(sizeof(unsigned long long),
            &h_check[CHK_N_BOUND], CkReduction::sum_ulong_long)};
    CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple, 5);
    msg->setCallback(CkCallback(CkIndex_Main::initDone(NULL), main_proxy));
    contribute(msg);
  }

  bool check_ghosts = false;
  // tol: see checkKernel. Zero on a rebuild step (the previous step migrated)
  // and at init; one support inside a window.
  void runCheck(RealType tol) {
    invokeCheck(d_parts[cur], np, x0, y0, z0, x1, y1, z1, tol, rho0, sound_c0,
        d_check, compute_stream, check_ghosts ? n_ghost : 0);
    sphSubmitMemcpy(h_check, d_check,
        sizeof(unsigned long long) * NUM_CHECKS, cudaMemcpyDeviceToHost,
        compute_stream);
  }
  RealType stepCheckTol() const { return rebuild_step ? 0.0f : support; }

  void abortOnBadParticles(int step) {
    const unsigned long long mask = h_check[CHK_BAD_MASK];
    if (!mask) return;
    CkAbort("CORRUPTION at step %d, patch (%d,%d,%d) on PE %d: %llu of %d local "
            "particle(s) are invalid (ghosts: %llu of %d non-finite, first at %lld) --%s%s%s%s%s\n",
            step, x, y, z, CkMyPe(), h_check[CHK_BAD_COUNT], np,
            h_check[6], n_ghost, (long long)(h_check[7] == ~0ull ? -1 : (long long)h_check[7]),
            (mask & CHK_BAD_NAN)     ? " non-finite state;" : "",
            (mask & CHK_BAD_RHO)     ? " density far outside the weakly-"
                                       "compressible band;" : "",
            (mask & CHK_BAD_SPEED)   ? " speed past Mach 0.5;" : "",
            (mask & CHK_BAD_TYPE)    ? " type field neither fluid nor "
                                       "boundary;" : "",
            (mask & CHK_BAD_OUTSIDE) ? " particle outside this patch's own "
                                       "box, i.e. an exchange delivered it to "
                                       "the wrong neighbour;" : "");
  }

  bool isLBIter(int it) const {
    return it == first_lb || (it != 0 && lb_freq > 0 && it % lb_freq == 0);
  }
  bool isStatsIter() const { return stats_freq > 0 && (my_iter % stats_freq) == 0; }

  // ---- activity ------------------------------------------------------------
  // What patch (px,py,pz) starts with, by the rule init() uses, so every patch
  // computes the same answer about every patch. Bit 0: holds particles at
  // all. Bit 1: some of them are fluid.
  static int siteMask(int px, int py, int pz) {
    if (px < 0 || px >= n_chares_x || py < 0 || py >= n_chares_y ||
        pz < 0 || pz >= n_chares_z) return 0;
    const RealType pw = dom_lx / n_chares_x, pd = dom_ly / n_chares_y,
                   ph = dom_lz / n_chares_z;
    const RealType ax0 = px * pw, ax1 = (px + 1) * pw;
    const RealType ay0 = py * pd, ay1 = (py + 1) * pd;
    const RealType az0 = pz * ph, az1 = (pz + 1) * ph;
    const RealType s = spacing;
    const int ix_lo = std::max(0, (int)std::floor(ax0 / s) - 1);
    const int ix_hi = (int)std::ceil(ax1 / s) + 1;
    const int iy_lo = std::max(0, (int)std::floor(ay0 / s) - 1);
    const int iy_hi = (int)std::ceil(ay1 / s) + 1;
    const int iz_lo = std::max(0, (int)std::floor(az0 / s) - 1);
    const int iz_hi = (int)std::ceil(az1 / s) + 1;
    int mask = 0;
    for (int iz = iz_lo; iz <= iz_hi && mask != 3; iz++) {
      const RealType qz = (iz + 0.5f) * s;
      if (qz >= dom_lz || qz < az0 || qz >= az1) continue;
      for (int iy = iy_lo; iy <= iy_hi && mask != 3; iy++) {
        const RealType qy = (iy + 0.5f) * s;
        if (qy >= dom_ly || qy < ay0 || qy >= ay1) continue;
        for (int ix = ix_lo; ix <= ix_hi; ix++) {
          const RealType qx = (ix + 0.5f) * s;
          if (qx >= dom_lx || qx < ax0 || qx >= ax1) continue;
          if (siteIsWall(qx, qy, qz)) mask |= ADV_ANY;
          else if (siteInColumn(qx, qy, qz)) mask |= ADV_ANY | ADV_FLUID;
        }
      }
    }
    return mask;
  }
  static bool activeAtStart(int px, int py, int pz) {
    if (!(siteMask(px, py, pz) & ADV_ANY)) return false;
    if (siteMask(px, py, pz) & ADV_FLUID) return true;
    for (int d = 0; d < NUM_DIRS; d++) {
      if (d == STAY) continue;
      if (siteMask(px + dirSX(d), py + dirSY(d), pz + dirSZ(d)) & ADV_FLUID) return true;
    }
    return false;
  }
  void seedActivity() {
    for (int d = 0; d < NUM_DIRS; d++) {
      const int nx = x + dirSX(d), ny = y + dirSY(d), nz = z + dirSZ(d);
      const int seed = !valid_dir[d] ? 0
          : ((activeAtStart(nx, ny, nz) ? ADV_ACTIVE : 0) |
             (siteMask(nx, ny, nz) & ADV_FLUID));
      for (int s = 0; s < ADV_RING; s++) nbr_adv[d][s] = seed;
    }
    my_adv = my_adv_next =
        (activeAtStart(x, y, z) ? ADV_ACTIVE : 0) | (n_fluid > 0 ? ADV_FLUID : 0);
  }
  // The advertisement a neighbour made for step t. On an edge that skipped
  // step t-1 none was sent: both ends had proven they are inactive at t and
  // t+1 (quiet levels), which is exactly an all-zero advertisement.
  int nbrAdv(int d, int t) const {
    return (xchg_step[d] == t - 1) ? nbr_adv[d][t % ADV_RING] : 0;
  }
  // A lower bound on the neighbour's quiet level for step t: its own word
  // when one arrived, otherwise the edge's level at the last exchange, less
  // the steps since (a guarantee shortens by one step per step).
  int nbrLevel(int d, int t) const {
    return (xchg_step[d] == t - 1) ? ADV_LEVEL(nbr_adv[d][t % ADV_RING])
                                   : std::max(0, xchg_min[d] - (t - xchg_step[d] - 1));
  }
  bool anyNbrAdvActive(int t) const {
    for (int d = 0; d < NUM_DIRS; d++)
      if (valid_dir[d] && (nbrAdv(d, t) & ADV_ACTIVE)) return true;
    return false;
  }
  // Both ends computed the same level at their last exchange, so both skip
  // the same steps; the message resumes one step past the guarantee.
  bool edgeSkips(int d, int t) const {
    return valid_dir[d] && t > xchg_step[d] && t - xchg_step[d] <= xchg_min[d];
  }
  bool anyNbrAdvFluid(int t) const {
    for (int d = 0; d < NUM_DIRS; d++)
      if (valid_dir[d] && (nbrAdv(d, t) & ADV_FLUID)) return true;
    return false;
  }
  bool isActive() const { return step_active; }

  void checkArrivals(std::map<int, int>& arrived, const char* what, int n_expect) {
    auto it = arrived.find(my_iter);
    const int got = it == arrived.end() ? 0 : it->second;
    if (got != n_expect)
      CkAbort("Patch (%d,%d,%d) step %d: %d %s message(s) arrived but %d were "
              "expected -- a partner sent a step's worth of messages that this "
              "patch did not consume\n", x, y, z, my_iter, got, what, n_expect);
    if (it != arrived.end()) arrived.erase(it);
  }

  // The next step, in whichever exchange mode; between steps is where the
  // priority of the compute stream can change (nothing is queued on it).
  void startStep() {
    pupCheckLanded("startStep");
    if (prio_dirty) updatePriority();
    if (concurrent_x) thisProxy[thisIndex].runStepC();
    else thisProxy[thisIndex].runStep();
  }

  // SPH_PRIO_BORDER=1: a patch with a partner on another GPU runs its kernels
  // on a stream at the device's highest priority, so the halos other GPUs
  // wait for are produced before the interior patches' work. Re-evaluated
  // after every migration (the partner map changes).
  void updatePriority() {
    prio_dirty = false;
    static const bool on = (getenv("SPH_PRIO_BORDER") != nullptr);
    if (!on) return;
    bool border = false;
    for (int d = 0; d < NUM_DIRS && !border; d++)
      if (valid_dir[d] &&
          peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[d], nbr_y[d], nbr_z[d]))) != 0)
        border = true;
    if (border == prio_now) return;
    cudaStream_t ns = border ? hapiAcquirePriorityStream() : hapiAcquireStream();
    hapiSubmitEventRecord((void*)halo_done, compute_stream);
    sphSubmitWaitEvent(ns, halo_done);
    hapiReleaseStream(compute_stream);
    compute_stream = ns;
    prio_now = border;
  }

  void iterate() {
    PhaseScope _ps(PH_ITERATE);
    if (concurrent_x) {
      // The park (AtSync, or AtSyncWait at the lag) happens inside the next
      // step, after its receives and before its forces (parkOrComputeC):
      // nothing of this patch's is then in flight and nothing is buffered.
      if (isLBIter(my_iter) && !lb_waiting && !sync_park_due) {
        lb_start_iter = my_iter;
        lb_t0 = CkWallTimer();
        if (!async_lb) sync_park_due = true;
        else { AtSyncStart(); lb_waiting = true; }
      }
      startStep();
      return;
    }
    if (isLBIter(my_iter) && !lb_waiting) {
      lb_start_iter = my_iter;
      lb_t0 = CkWallTimer();
      if (!async_lb) {
        AtSync();
        return;
      }
      AtSyncStart();
      lb_waiting = true;
      startStep();
    } else if (lb_waiting && my_iter >= lb_start_iter + lb_wait_lag) {
      lb_waiting = false;
      if (x == 0 && y == 0 && z == 0)
        CkPrintf("  LB at step %d: AtSyncWait entered %.3f s after AtSyncStart\n",
                 lb_start_iter, CkWallTimer() - lb_t0);
      AtSyncWait();
    } else {
      startStep();
    }
  }

  void ResumeFromSync() {
    if (x == 0 && y == 0 && z == 0)
      CkPrintf("  LB at step %d: resumed %.3f s after %s; patch (0,0,0) now on PE %d\n",
               lb_start_iter, CkWallTimer() - lb_t0,
               async_lb ? "AtSyncStart" : "AtSync", CkMyPe());
    prio_dirty = true;
    if (parked_midstep) {   // -c: resume the step where it parked (parkOrComputeC)
      parked_midstep = false;
      if (prio_dirty) updatePriority();
      computeForcesC();
      return;
    }
    if (xsend_pending) sendExchangeMsgs();   // deferred past the park (see sendExchange)
    startStep();
  }

  int stallTier(int worst_d) const {
    if (send_warn_ms <= 0 || worst_d < 0) return -1;
    const int d = worst_d % 100;
    return peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[d], nbr_y[d], nbr_z[d])));
  }

  // The latency of one exchange message, at the moment its payload is usable.
  void noteLanded(int kind, int ref, int dir, long long ts) {
    const long long now = sphNowNs();
    const long long te = env_t[kind][ref & 1][dir];
    const int tier = peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[dir], nbr_y[dir], nbr_z[dir])));
    const double us = (now - ts) / 1000.0, eus = (te - ts) / 1000.0;
    LatStats& L = g_lat;
    L.n[kind][tier]++; L.sum[kind][tier] += us; L.esum[kind][tier] += eus;
    if (us > L.mx[kind][tier]) L.mx[kind][tier] = us;
    L.hist[kind][tier][latBin(us)]++;
    L.ehist[kind][tier][latBin(eus)]++;
    if (_nvtxOn.load(std::memory_order_relaxed))
      _nvtxTraceMsg((unsigned long long)ts, (unsigned long long)te, (unsigned long long)now, kind, tier);
  }

  // ---- phase 1: pressure, then halo ----------------------------------------
  void startHalo() {
    PhaseScope _ps(PH_START_HALO);
    // Step 1 and every nl_freq-th step rebuild; the step before a rebuild
    // migrates. With nl_freq = 1 every step does both (the original scheme).
    rebuild_step = (my_iter == 1) || (my_iter % nl_freq == 0);
    migrate_step = ((my_iter + 1) % nl_freq == 0);
    my_adv = my_adv_next;
    step_active = (my_adv & ADV_ACTIVE) != 0;
    pc_step = (integrator == 1) && step_active;
    n_expect = 0;
    n_expect_mig = 0;
    for (int d = 0; d < NUM_DIRS; d++)
      if (valid_dir[d] && !edgeSkips(d, my_iter)) n_expect_mig++;
    for (int d = 0; d < NUM_DIRS; d++) {
      halo_peer[d] = valid_dir[d] && step_active &&
          (nbrAdv(d, my_iter) & ADV_ACTIVE);
      if (halo_peer[d]) n_expect++;
    }
    if (rebuild_step) {
      n_ghost = 0; ghost_alloc = 0;
      for (int d = 0; d < NUM_DIRS; d++) { ghost_off[d] = 0; ghost_cnt[d] = -1; }
      cells_valid = false;
    }
    halo_direct.clear(); halo_direct2.clear();
    halo_open = true;
    pc_open = pc_step;
    if (!step_active) {
      thisProxy[thisIndex].haloPacked();
      return;
    }
    static const bool batch_halo = (getenv("SPH_NO_BATCH_HALO") == nullptr);
    // SPH_SORT=1 sorts the locals into cell order at every rebuild. Off by
    // default: the forces kernel iterates in cell order by itself now, and
    // the sort's cell build and gather cost more than the random reads of the
    // layout pass they used to prevent (73.2 vs 79.1 ms/step at 25M, 1 GPU).
    // SPH_SORT_EVERY=N sorts at every N-th step's rebuild instead: the layout
    // pass reads particles in cell order, and over a long run the unsorted
    // array mixes across cells as the fluid moves (98M, 8 GPUs, 2000 steps:
    // the busiest GPU's kernel time grew 23% at a flat host load; TNL-SPH,
    // which reorders every step, stayed flat), so a sort every 100 steps
    // keeps the reads coherent for a cost that does not register.
    static const bool sort_locals = (getenv("SPH_SORT") != nullptr);
    static const int sort_every =
        getenv("SPH_SORT_EVERY") ? atoi(getenv("SPH_SORT_EVERY")) : (sort_locals ? 1 : 0);
    if (batch_halo) hapiSubmitBatchBegin();
    // The migrants appended last step landed on comm_stream; everything below
    // reads them on compute_stream -- or, with the interior pass queued ahead
    // on compute_stream, the pack reads them on pack_stream (highest priority)
    // after the compaction, so it is not delayed behind the interior kernel.
    const bool pack_prio = interior_done && pack_stream != NULL;
    cudaStream_t ps = pack_prio ? pack_stream : compute_stream;
    hapiSubmitEventRecord((void*)halo_done, comm_stream);
    sphSubmitWaitEvent(ps, halo_done);
    if (pack_prio) { hapiSubmitEventRecord((void*)compact_ev, compute_stream); sphSubmitWaitEvent(ps, compact_ev); }
    if (rebuild_step) {
      // Optional sort of the locals into cell order (see above), BEFORE the
      // pack records its index lists, so the lists survive until the
      // compaction at the end of the window. The swap is inside this serial
      // block, so cur names the live buffer at every entry-method boundary.
      if (sort_every > 0 && np > 0 && (my_iter % sort_every) == 0) {
        invokeCellBuild(d_parts[cur], np, x0, y0, z0, inv_csize, ncx, ncy, ncz,
            ncells, d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, d_cell_of,
            d_scan_tmp, scan_tmp_bytes, compute_stream);
        invokeGather(d_parts[cur], d_cell_parts, np, d_parts[1 - cur], compute_stream);
        cur = 1 - cur;
      }
      invokePackHalo(d_parts[cur], np, x0, y0, z0, x1, y1, z1, cell_size,
          d_halo_ptrs, d_halo_idx_ptrs, d_counts, exch_capacity, valid_mask,
          /*counts_zeroed=*/false, ps);
      sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
          cudaMemcpyDeviceToHost, ps);
    } else {
      invokePackHaloByIndex(d_parts[cur], d_halo_ptrs, d_halo_idx, d_halo_cnt,
          exch_capacity, exch_slots, ps);
    }
    if (pack_prio) hapiSubmitEventRecord((void*)pack_ev, ps);
    halo_pending = true;
    hapiAddCallback(ps,
        CkCallback(CkIndex_Patch::haloPacked(), thisProxy[thisIndex]));
    if (batch_halo) hapiSubmitBatchEnd();
  }

  void sendHalo() {
    PhaseScope _ps(PH_SEND_HALO);
    if (!step_active) return;
    halo_pending = false;
    if (rebuild_step) {
      if (h_counts[ERR_COUNTER] > 0)
        CkAbort("Patch (%d,%d,%d): halo exchange overflowed a per-direction buffer "
                "(face slot %d) at step %d; increase -e\n", x, y, z, exch_capacity, my_iter);
      for (int d = 0; d < NUM_DIRS; d++) {
        h_halo_cnt[d] = d == STAY ? 0 : h_counts[d];
        rebuild_peer[d] = halo_peer[d];
      }
      // The by-index packs of the rest of the window read the lengths from
      // the device; compute_stream order puts this copy ahead of them.
      sphSubmitMemcpy(d_halo_cnt, h_halo_cnt, sizeof(int) * NUM_DIRS,
          cudaMemcpyHostToDevice, compute_stream);
    } else {
      for (int d = 0; d < NUM_DIRS; d++)
        if (halo_peer[d] != rebuild_peer[d])
          CkAbort("Patch (%d,%d,%d) step %d: the halo peer set changed inside a "
                  "neighbour-list window (direction %d) -- activity can only "
                  "change at a migration, which only happens at the window's "
                  "end\n", x, y, z, my_iter, d);
    }
    const long long ts = sphNowNs();
    SendTimer st; const double tloop = send_warn_ms > 0 ? CkWallTimer() : 0.0;
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!halo_peer[d]) continue;
      const int cnt = h_halo_cnt[d];
      if (host_profile_on) {
        const int tier = peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[d], nbr_y[d], nbr_z[d])));
        g_hprof.sent[tier]++; g_hprof.sent_bytes[tier] += (long)cnt * sizeof(Particle);
      }
      st.begin();
      thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveHalo(my_iter, flipDir(d), cnt, ts, cnt,
          (outstanding_sends++, sphSendBuffer(
           d_send_halo + dirOffset(d, exch_capacity),
           sendDoneCb(),
           comm_stream)));
      st.end(d);
    }
    st.report("sendHalo", x, y, z, my_iter, send_warn_ms > 0 ? (CkWallTimer() - tloop) * 1000.0 : 0.0, stallTier(st.worst_d));
  }

  // A halo for THIS step lands straight in its ghost slot when that slot is
  // known and free (see sph2d.C receiveHalo). On a rebuild step the slot is
  // handed out in arrival order and recorded; inside a window it is the
  // recorded one, and the count has to match.
  void receiveHalo(int ref, int dir, int n, long long ts, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    PhaseScope _ps(PH_RECV_HALO);
    env_t[0][ref & 1][dir] = sphNowNs();
    if (host_profile_on) {
      const int tier = peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[dir], nbr_y[dir], nbr_z[dir])));
      g_hprof.recv[tier]++; g_hprof.recv_bytes[tier] += (long)n * sizeof(Particle);
    }
    static const bool direct_ok = (getenv("SPH_NO_DIRECT_HALO") == nullptr);
    static const bool ordered = (getenv("SPH_HALO_ORDERED") != nullptr);
    parts = d_recv_halo + dirOffset(dir, exch_capacity);
    if (direct_ok && halo_open && ref == my_iter && n > 0 &&
        halo_direct.find(dir) == halo_direct.end()) {
      if (rebuild_step) {
        if (ghost_cnt[dir] < 0 && np + ghost_alloc + n <= part_capacity) {
          parts = d_parts[cur] + np + ghost_alloc;
          ghost_off[dir] = ghost_alloc; ghost_cnt[dir] = n;
          halo_direct[dir] = ghost_alloc;
          ghost_alloc += n;
          devicePost[0].buffer_free = !ordered;
        }
      } else if (ghost_cnt[dir] == n) {
        parts = d_parts[cur] + np + ghost_off[dir];
        halo_direct[dir] = ghost_off[dir];
        devicePost[0].buffer_free = !ordered;
      }
    }
    devicePost[0].hapi_stream = comm_stream;
  }

  // ---- concurrent exchange: the receive hooks and bodies ---------------------
  // Hook (landing buffer): straight into the ghost region once this step's
  // migrants are all known (their total fixes where the ghosts start),
  // otherwise into the parity slot. Idempotent per (step, direction): the
  // runtime re-invokes the hook when it replays a buffered receive.
  void receiveHaloC(int ref, int dir, int n, long long ts, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    PhaseScope _ps(PH_RECV_HALO);
    env_t[0][ref & 1][dir] = sphNowNs();
    static const bool direct_ok = (getenv("SPH_NO_DIRECT_HALO") == nullptr);
    static const bool ordered = (getenv("SPH_HALO_ORDERED") != nullptr);
    parts = d_recv_halo + (size_t)(ref & 1) * exch_slots + dirOffset(dir, exch_capacity);
    const int bit = 1 << dir;
    if (halo_seen[ref] & bit) {
      auto landed = halo_direct.find(dir);
      if (ref == open_ref && landed != halo_direct.end()) {
        parts = d_parts[cur] + ghost_base + landed->second;
        devicePost[0].buffer_free = !ordered;
      }
      devicePost[0].hapi_stream = comm_stream;
      return;
    }
    halo_seen[ref] |= bit;
    halo_staged[ref][dir] = n;   // every direction: the merge walks this
    if (direct_ok && halo_open && ref == open_ref && mig_known && n > 0 &&
        ghost_cnt[dir] < 0 && halo_direct.find(dir) == halo_direct.end()) {
      const int base = ghost_base;
      if (base + ghost_alloc + n <= part_capacity) {
        parts = d_parts[cur] + base + ghost_alloc;
        ghost_off[dir] = ghost_alloc; ghost_cnt[dir] = n;
        halo_direct[dir] = ghost_alloc;
        ghost_alloc += n;
        devicePost[0].buffer_free = !ordered;
      }
    }
    devicePost[0].hapi_stream = comm_stream;
  }
  // Body (payload usable): record it; the last one the step waits for sends halosDone.
  void receiveHaloC(int ref, int dir, int n, long long ts, int m, Particle* parts) {
    const int bit = 1 << dir;
    if (halo_landed[ref] & bit) return;   // a replayed delivery
    halo_landed[ref] |= bit;
    noteLanded(0, ref, dir, ts);
    if (ref == my_iter && halo_expect_known) checkHalosDone(ref);
  }
  void receiveParticlesC(int ref, int dir, int adv, int n, long long ts, int& m,
      Particle*& parts, CkDeviceBufferPost* devicePost) {
    PhaseScope _ps(PH_RECV_PARTS);
    env_t[1][ref & 1][dir] = sphNowNs();
    if (ref > my_iter + ADV_LEVEL_CAP)
      CkAbort("Patch (%d,%d,%d) at step %d: migration from step %d -- a partner ran more than %d steps ahead\n",
              x, y, z, my_iter, ref, ADV_LEVEL_CAP);
    nbr_adv[dir][ref % ADV_RING] = adv;
    const int bit = 1 << dir;
    if (!(mig_seen[ref] & bit)) {
      mig_seen[ref] |= bit;
      mig_env_n[ref]++;
      mig_env_total[ref] += n;
      mig_staged[ref][dir] = n;
      if (ref == open_ref && mig_env_n[ref] == n_expect_mig) { mig_known = true; ghost_base = np_base + mig_env_total[ref]; }
    }
    parts = d_recv_mig + (size_t)(ref & 1) * exchSlots(exch_capacity) + dirOffset(dir, exch_capacity);
    devicePost[0].hapi_stream = comm_stream;
  }
  void receiveParticlesC(int ref, int dir, int adv, int n, long long ts, int m, Particle* parts) {
    const int bit = 1 << dir;
    if (mig_landed[ref] & bit) return;
    mig_landed[ref] |= bit;
    noteLanded(1, ref, dir, ts);
    if (ref == open_ref) checkMigrantsDone(ref);
  }
  static int popcount27(int v) { return __builtin_popcount((unsigned)v); }
  void checkMigrantsDone(int ref) {
    if (ref != open_ref || (done_sent[ref] & 1)) return;
    if (popcount27(mig_landed[ref]) != n_expect_mig) return;
    done_sent[ref] |= 1;
    thisProxy[thisIndex].migrantsDone(ref);
  }
  void checkHalosDone(int ref) {
    if (ref != my_iter || !halo_expect_known || (done_sent[ref] & 2)) return;
    if (popcount27(halo_landed[ref]) != n_expect) return;
    done_sent[ref] |= 2;
    thisProxy[thisIndex].halosDone(ref);
  }
  // Every migrant of this step has landed: append them, in direction order,
  // place the leaver ghosts behind them, and set the halo expectation.
  void mergeMigrantsC() {
    PhaseScope _ps(PH_APPEND_PARTS);
    auto st = mig_staged.find(my_iter);
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!(mig_landed[my_iter] & (1 << d))) continue;
      const int n = (st != mig_staged.end() && st->second.count(d)) ? st->second[d] : 0;
      parts_arrived[my_iter]++;
      xchg_step[d] = my_iter - 1;
      xchg_min[d] = std::min(my_level_next, ADV_LEVEL(nbr_adv[d][my_iter % ADV_RING]));
      if (n == 0) continue;
      if (np + n > part_capacity)
        CkAbort("Patch (%d,%d,%d): %d particles exceed capacity %d at step %d; increase headroom (-r)\n",
                x, y, z, np + n, part_capacity, my_iter);
      sphSubmitMemcpy(d_parts[cur] + np,
          d_recv_mig + (size_t)(my_iter & 1) * exchSlots(exch_capacity) + dirOffset(d, exch_capacity),
          sizeof(Particle) * n, cudaMemcpyDeviceToDevice, comm_stream);
      np += n;
      n_fluid += n;
    }
    mig_staged.erase(my_iter); mig_landed.erase(my_iter);
    expectHalos();
    halo_expect_known = true;
    checkHalosDone(my_iter);
  }
  // Every halo of this step has landed: copy the staged ones behind the
  // leaver ghosts (the direct ones are in place).
  void mergeHalosC() {
    PhaseScope _ps(PH_APPEND_GHOSTS);
    auto st = halo_staged.find(my_iter);
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!(halo_landed[my_iter] & (1 << d))) continue;
      halo_arrived[my_iter]++;
      const int n = (st != halo_staged.end() && st->second.count(d)) ? st->second[d] : 0;
      auto landed = halo_direct.find(d);
      if (landed != halo_direct.end()) { halo_direct.erase(landed); if (step_active) n_ghost += n; continue; }
      if (n == 0) { ghost_off[d] = ghost_alloc; ghost_cnt[d] = 0; continue; }
      if (!step_active) continue;   // an active partner's boundary; nothing here moves
      if (np + ghost_alloc + n > part_capacity)
        CkAbort("Patch (%d,%d,%d): %d particles+ghosts exceed capacity %d at step %d; increase headroom (-r)\n",
                x, y, z, np + ghost_alloc + n, part_capacity, my_iter);
      sphSubmitMemcpy(d_parts[cur] + np + ghost_alloc,
          d_recv_halo + (size_t)(my_iter & 1) * exch_slots + dirOffset(d, exch_capacity),
          sizeof(Particle) * n, cudaMemcpyDeviceToDevice, comm_stream);
      ghost_off[d] = ghost_alloc; ghost_cnt[d] = n;
      ghost_alloc += n;
      n_ghost += n;
    }
    halo_staged.erase(my_iter); halo_landed.erase(my_iter); halo_seen.erase(my_iter);
    done_sent.erase(my_iter);
    halo_expect_known = false;
  }
  // The LB park, after the receives and before the forces: this patch has
  // nothing in flight and nothing buffered (its own exchange goes out only
  // after the forces), so a move here is the case the runtime handles.
  void parkOrComputeC() {
    if (async_lb && lb_waiting && my_iter - 1 >= lb_start_iter + lb_wait_lag) {
      lb_waiting = false;
      if (x == 0 && y == 0 && z == 0)
        CkPrintf("  LB at step %d: AtSyncWait entered %.3f s after AtSyncStart\n", lb_start_iter, CkWallTimer() - lb_t0);
      parked_midstep = true;
      AtSyncWait();
      return;
    }
    if (!async_lb && sync_park_due) {
      sync_park_due = false;
      parked_midstep = true;
      AtSync();
      return;
    }
    computeForcesC();
  }

  void appendGhosts(int dir, int n) {
    PhaseScope _ps(PH_APPEND_GHOSTS);
    halo_arrived[my_iter]++;
    if (rebuild_step) {
      if (n == 0) { ghost_off[dir] = ghost_alloc; ghost_cnt[dir] = 0; return; }
      if (np + n_ghost + n > part_capacity)
        CkAbort("Patch (%d,%d,%d): %d particles+ghosts exceed capacity %d at step "
                "%d; increase headroom (-r)\n", x, y, z, np + n_ghost + n,
                part_capacity, my_iter);
      auto landed = halo_direct.find(dir);
      if (landed != halo_direct.end()) {
        halo_direct.erase(landed);
      } else {
        if (np + ghost_alloc + n > part_capacity)
          CkAbort("Patch (%d,%d,%d): %d particles+ghosts exceed capacity %d at step %d; increase headroom (-r)\n",
                  x, y, z, np + ghost_alloc + n, part_capacity, my_iter);
        sphSubmitMemcpy(d_parts[cur] + np + ghost_alloc,
            d_recv_halo + dirOffset(dir, exch_capacity), sizeof(Particle) * n,
            cudaMemcpyDeviceToDevice, comm_stream);
        ghost_off[dir] = ghost_alloc; ghost_cnt[dir] = n;
        ghost_alloc += n;
      }
      n_ghost += n;
    } else {
      if (n != ghost_cnt[dir])
        CkAbort("Patch (%d,%d,%d) step %d: %d halo particles from direction %d "
                "but the window's rebuild recorded %d -- the halo membership "
                "changed inside a neighbour-list window\n",
                x, y, z, my_iter, n, dir, ghost_cnt[dir]);
      if (n == 0) return;
      auto landed = halo_direct.find(dir);
      if (landed != halo_direct.end()) {
        halo_direct.erase(landed);
      } else {
        sphSubmitMemcpy(d_parts[cur] + np + ghost_off[dir],
            d_recv_halo + dirOffset(dir, exch_capacity), sizeof(Particle) * n,
            cudaMemcpyDeviceToDevice, comm_stream);
      }
    }
  }

  // ---- phase 2: neighbours, forces, then integrate or predict --------------
  void computeForces() {
    PhaseScope _ps(PH_COMPUTE);
    halo_open = false;
    if (!isActive()) {
      if (isStatsIter()) {
        hapiSubmitEventRecord((void*)halo_done, comm_stream);
        sphSubmitWaitEvent(compute_stream, halo_done);
        runCheck(stepCheckTol());
        invokeStats(d_parts[cur], np, d_stats, compute_stream);
        sphSubmitMemcpy(h_stats, d_stats, sizeof(RealType) * 8,
            cudaMemcpyDeviceToHost, compute_stream);
        hapiAddCallback(compute_stream,
            CkCallback(CkIndex_Patch::leaversPacked(), thisProxy[thisIndex]));
      } else {
        thisProxy[thisIndex].leaversPacked();
      }
      return;
    }
    static const bool batch_compute = (getenv("SPH_NO_BATCH_COMPUTE") == nullptr);
    static const bool fold_zero = (getenv("SPH_NO_FOLD_ZERO") == nullptr);
    if (batch_compute) hapiSubmitBatchBegin();
    hapiSubmitEventRecord((void*)halo_done, comm_stream);
    sphSubmitWaitEvent(compute_stream, halo_done);

    if (isStatsIter()) { check_ghosts = true; runCheck(stepCheckTol()); check_ghosts = false; idAuditPass(); }

    const int ntot = np + n_ghost;
    if (interior_done) {
      // The interior pass (end of the previous step) left the locals' list L
      // and their cell-ordered streams in [0, interior_np). Build the EXTRAS
      // list E over what was appended since (migrants, then ghosts) into
      // [interior_np, ntot) of the same streams, and run the boundary layers
      // over both lists. No full rebuild.
      const int n0 = interior_np, nx = ntot - n0;
      sphSubmitWaitEvent(compute_stream, pack_ev);   // the pack's read of the positions is done before the integrate below
      invokeCellBuild(d_parts[cur] + n0, nx, x0, y0, z0, inv_csize, ncx, ncy, ncz,
          ncells, d_cell_cnt2, d_cell_off2, d_cursor2, d_cell_parts2, d_cell_of2,
          d_scan_tmp, scan_tmp_bytes, compute_stream);
      invokeLayout(d_parts[cur] + n0, d_cell_parts2, nx, d_pos4 + n0, d_vel4 + n0, d_type8 + n0, compute_stream);
      if (interiorRows())
        invokeForcesTwoListRows(ntot, np, n0, ncx, ncy, ncz, interiorLayers(), d_cell_parts, d_cell_of, d_cell_off,
            d_cell_cnt, d_cell_parts2, d_cell_of2, d_cell_off2, d_cell_cnt2, d_pos4, d_vel4, d_type8,
            smooth_h, pmass, sound_c0, gravity, d_drho, d_ax, d_ay, d_az, d_item_beg, d_item_pre, compute_stream);
      else
        invokeForcesTwoList(ntot, np, n0, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_cell_off, d_cell_cnt,
            d_cell_parts2, d_cell_of2, d_cell_off2, d_cell_cnt2, d_pos4, d_vel4, d_type8,
            smooth_h, pmass, sound_c0, gravity, d_drho, d_ax, d_ay, d_az,
            d_iota, d_sel_flags, d_sel, d_sel_num, d_sel_tmp, sel_tmp_bytes, compute_stream);
      splitStats("boundary", ntot);
      cells_valid = false;   // L alone is not the full list
      interior_done = false;
    } else {
      // A rebuild step builds the window's cell list over the sorted locals and
      // the ghosts; a chare that migrated inside a window rebuilds it locally.
      if (rebuild_step || !cells_valid) {
        invokeCellBuild(d_parts[cur], ntot, x0, y0, z0, inv_csize, ncx, ncy, ncz,
            ncells, d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, d_cell_of,
            d_scan_tmp, scan_tmp_bytes, compute_stream);
        cells_valid = true;
      }
      invokeLayout(d_parts[cur], d_cell_parts, ntot, d_pos4, d_vel4, d_type8, compute_stream);
      invokeForces(0, ntot, np, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_cell_off,
          d_cell_cnt, d_pos4, d_vel4, d_type8, smooth_h, pmass, sound_c0, gravity,
          d_drho, d_ax, d_ay, d_az, compute_stream);
    }

    if (!pc_step) {
      // On a migration step that is not reporting, the integration and the
      // compaction are one pass (finishCompute then only copies the counts).
      const bool fused = migrate_step && !isStatsIter();
      if (fused)
        invokeIntegrateLeavers(false, d_parts[cur], np, sim_dt, rho0, sound_c0,
            d_drho, d_ax, d_ay, d_az, x0, y0, z0, x1, y1, z1, d_mig_ptrs,
            d_parts[1 - cur], d_counts, exch_capacity, compute_stream);
      else
        invokeIntegrate(d_parts[cur], np, sim_dt, rho0, sound_c0, d_drho, d_ax,
            d_ay, d_az, fold_zero ? d_counts : NULL, compute_stream);
      finishCompute(fused);
    } else {
      // Half step into the other buffer (with its pressure), and the second
      // exchange of exactly the particles the first one sent.
      invokePredict(d_parts[cur], d_parts[1 - cur], np, 0.5f * sim_dt, rho0,
          sound_c0, d_drho, d_ax, d_ay, d_az, compute_stream);
      invokePackHaloByIndex(d_parts[1 - cur], d_halo_ptrs2, d_halo_idx,
          d_halo_cnt, exch_capacity, exch_slots, compute_stream);
      pc_pending = true;
      halo_pending2 = true;
      hapiAddCallback(compute_stream,
          CkCallback(CkIndex_Patch::haloPacked2(), thisProxy[thisIndex]));
    }
    if (batch_compute) hapiSubmitBatchEnd();
  }

  // The tail of the compute phase on compute_stream: stats on a reporting
  // step, the compaction on a migrate step, and the callback that ends it.
  // leavers_done: the integrator already compacted (invokeIntegrateLeavers),
  // so only the counts are still to come.
  void finishCompute(bool leavers_done) {
    PhaseScope _ps(PH_FINISH);
    static const bool fold_zero = (getenv("SPH_NO_FOLD_ZERO") == nullptr);
    if (isStatsIter()) {
      invokeStats(d_parts[cur], np, d_stats, compute_stream);
      sphSubmitMemcpy(h_stats, d_stats, sizeof(RealType) * 8,
          cudaMemcpyDeviceToHost, compute_stream);
    }
    if (migrate_step) {
      if (!leavers_done)
        invokeMarkLeavers(d_parts[cur], np, x0, y0, z0, x1, y1, z1, d_mig_ptrs,
            d_parts[1 - cur], d_counts, exch_capacity, fold_zero, compute_stream);
      sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
          cudaMemcpyDeviceToHost, compute_stream);
      mig_pending = true;
    }
    hapiAddCallback(compute_stream,
        CkCallback(CkIndex_Patch::leaversPacked(), thisProxy[thisIndex]));
  }

  // ---- predictor-corrector: second exchange, second evaluation -------------
  void sendHalo2() {
    PhaseScope _ps(PH_PC);
    halo_pending2 = false;
    const long long ts = sphNowNs();
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!halo_peer[d]) continue;
      const int cnt = h_halo_cnt[d];
      thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveHalo2(my_iter, flipDir(d), cnt, ts, cnt,
          (outstanding_sends++, sphSendBuffer(
           d_send_halo2 + dirOffset(d, exch_capacity),
           sendDoneCb(),
           comm_stream)));
    }
  }

  // The predicted ghosts land in the same slots of the OTHER buffer. The slot
  // is known once this step's first exchange from that direction has been
  // posted; before that (or for a later step) the message is staged.
  void receiveHalo2(int ref, int dir, int n, long long ts, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    PhaseScope _ps(PH_PC);
    env_t[2][ref & 1][dir] = sphNowNs();
    parts = d_recv_halo2 + dirOffset(dir, exch_capacity);
    static const bool direct_ok = (getenv("SPH_NO_DIRECT_HALO") == nullptr);
    static const bool ordered = (getenv("SPH_HALO_ORDERED") != nullptr);
    if (direct_ok && pc_open && ref == my_iter && n > 0 && ghost_cnt[dir] == n &&
        halo_direct2.find(dir) == halo_direct2.end()) {
      parts = d_parts[1 - cur] + np + ghost_off[dir];
      halo_direct2[dir] = ghost_off[dir];
      devicePost[0].buffer_free = !ordered;
    }
    devicePost[0].hapi_stream = comm_stream;
  }

  void appendGhosts2(int dir, int n) {
    PhaseScope _ps(PH_PC);
    halo_arrived2[my_iter]++;
    if (n != ghost_cnt[dir])
      CkAbort("Patch (%d,%d,%d) step %d: %d predicted halo particles from "
              "direction %d but the first exchange brought %d\n",
              x, y, z, my_iter, n, dir, ghost_cnt[dir]);
    if (n == 0) return;
    auto landed = halo_direct2.find(dir);
    if (landed != halo_direct2.end()) {
      halo_direct2.erase(landed);
    } else {
      sphSubmitMemcpy(d_parts[1 - cur] + np + ghost_off[dir],
          d_recv_halo2 + dirOffset(dir, exch_capacity), sizeof(Particle) * n,
          cudaMemcpyDeviceToDevice, comm_stream);
    }
  }

  void correct() {
    PhaseScope _ps(PH_PC);
    pc_open = false;
    static const bool batch_compute = (getenv("SPH_NO_BATCH_COMPUTE") == nullptr);
    static const bool fold_zero = (getenv("SPH_NO_FOLD_ZERO") == nullptr);
    if (batch_compute) hapiSubmitBatchBegin();
    hapiSubmitEventRecord((void*)halo_done, comm_stream);
    sphSubmitWaitEvent(compute_stream, halo_done);
    const int ntot = np + n_ghost;
    if (!cells_valid) {   // moved here between the two evaluations
      invokeCellBuild(d_parts[1 - cur], ntot, x0, y0, z0, inv_csize, ncx, ncy, ncz,
          ncells, d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, d_cell_of,
          d_scan_tmp, scan_tmp_bytes, compute_stream);
      cells_valid = true;
    }
    invokeLayout(d_parts[1 - cur], d_cell_parts, ntot, d_pos4, d_vel4, d_type8, compute_stream);
    invokeForces(0, ntot, np, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_cell_off,
        d_cell_cnt, d_pos4, d_vel4, d_type8, smooth_h, pmass, sound_c0, gravity,
        d_drho, d_ax, d_ay, d_az, compute_stream);
    const bool fused = migrate_step && !isStatsIter();
    if (fused)
      invokeIntegrateLeavers(true, d_parts[cur], np, sim_dt, rho0, sound_c0,
          d_drho, d_ax, d_ay, d_az, x0, y0, z0, x1, y1, z1, d_mig_ptrs,
          d_parts[1 - cur], d_counts, exch_capacity, compute_stream);
    else
      invokeCorrect(d_parts[cur], np, sim_dt, rho0, sound_c0, d_drho, d_ax, d_ay,
          d_az, fold_zero ? d_counts : NULL, compute_stream);
    pc_pending = false;
    finishCompute(fused);
    if (batch_compute) hapiSubmitBatchEnd();
  }

  // ---- migration -----------------------------------------------------------
  // The interior forces of the next step, now: the survivors' interior cells
  // need no neighbour. Not before a step that sorts the locals (the sort
  // permutes the indices the results are stored by), not with -c (its own
  // schedule), not for the predictor-corrector (two evaluations).
  // SPH_SPLIT_STATS=1: the targets of each pass (rows form) for a few patches at the stats steps.
  void splitStats(const char* pass, int n_entries) {
    static const bool on = (getenv("SPH_SPLIT_STATS") != nullptr);
    if (!on || !interiorRows() || !isStatsIter() || ((x + y + z) % 5) != 0) return;
    const int L = interiorLayers();
    const int nitems = (pass[0] == 'i') ? (ncy - 2 * L) * (ncz - 2 * L) : 3 * ncy * ncz;
    int total = 0;
    hapiCheck(cudaStreamSynchronize(compute_stream));
    hapiCheck(cudaMemcpy(&total, d_item_pre + nitems, sizeof(int), cudaMemcpyDeviceToHost));
    CkPrintf("[SPLIT] patch (%d,%d,%d) step %d %s: %d targets of %d entries (np %d, cells %dx%dx%d)\n",
        x, y, z, my_iter, pass, total, n_entries, np, ncx, ncy, ncz);
  }
  void interiorPass() {
    static const bool on = (getenv("SPH_INTERIOR_FIRST") != nullptr);
    if (!on || concurrent_x || integrator != 0 || nl_freq != 1 || !step_active || np <= 0) return;
    static const int sort_every =
        getenv("SPH_SORT_EVERY") ? atoi(getenv("SPH_SORT_EVERY")) : (getenv("SPH_SORT") ? 1 : 0);
    const int next = my_iter + 1;
    if (sort_every > 0 && (next % sort_every) == 0) return;
    // Not into the LB window close either: the balancer's forced CUPTI flush
    // returns only once every queued kernel has completed and blocks the
    // process's CUDA calls meanwhile, so a deep queue there drains and starves
    // the GPU (135-250 ms per window, job 22799507). A shallow queue at that
    // moment costs one step of overlap per window.
    if (isLBIter(my_iter) || isLBIter(next)) return;
    if (ncx < 6 || ncy < 6 || ncz < 6) return;   // no interior cell at all
    static const bool batch_compute = (getenv("SPH_NO_BATCH_COMPUTE") == nullptr);
    if (batch_compute) hapiSubmitBatchBegin();
    invokeCellBuild(d_parts[cur], np, x0, y0, z0, inv_csize, ncx, ncy, ncz,
        ncells, d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, d_cell_of,
        d_scan_tmp, scan_tmp_bytes, compute_stream);
    invokeLayout(d_parts[cur], d_cell_parts, np, d_pos4, d_vel4, d_type8, compute_stream);
    if (interiorRows())
      invokeForcesInteriorRows(np, ncx, ncy, ncz, interiorLayers(), d_cell_parts, d_cell_of, d_cell_off, d_cell_cnt,
          d_pos4, d_vel4, d_type8, smooth_h, pmass, sound_c0, gravity, d_drho, d_ax, d_ay, d_az,
          d_item_beg, d_item_pre, compute_stream);
    else
      invokeForcesInteriorSel(np, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_cell_off, d_cell_cnt,
          d_pos4, d_vel4, d_type8, smooth_h, pmass, sound_c0, gravity, d_drho, d_ax, d_ay, d_az,
          d_iota, d_sel_flags, d_sel, d_sel_num, d_sel_tmp, sel_tmp_bytes, compute_stream);
    if (batch_compute) hapiSubmitBatchEnd();
    splitStats("interior", np);
    cells_valid = false;
    interior_done = true;
    interior_np = np;
  }

  void sendLeavers() {
    PhaseScope _ps(PH_SEND_LEAVERS);
    const bool moved = isActive() && migrate_step;
    if (moved) {
      mig_pending = false;
      if (h_counts[ERR_COUNTER] > 0)
        CkAbort("Patch (%d,%d,%d): migration overflowed a per-direction buffer "
                "(face slot %d) at step %d; increase -e\n", x, y, z, exch_capacity, my_iter);
      cur = 1 - cur;
      np = h_counts[STAY];
      interiorPass();
    }
    // The ghost layout lives until the window ends; a migrate step ends it.
    if (migrate_step) { n_ghost = 0; ghost_alloc = 0; }
    my_adv_next = (n_fluid > 0 ? ADV_FLUID : 0) |
        ((np > 0 && (n_fluid > 0 || anyNbrAdvFluid(my_iter))) ? ADV_ACTIVE : 0);
    // Quiet level for the next step. Particles only leave an active patch,
    // and a patch is active only when it holds fluid or a neighbour held
    // fluid the step before; so a patch that is inactive next step with no
    // active neighbour now stays inactive for one step more than the least
    // quiet of its neighbours. Over such a step the only thing a migration
    // message would carry is an all-zero advertisement, and both ends know
    // it, so neither sends (edgeSkips). SPH_NO_QUIET_SKIP=1 restores the
    // message on every edge every step.
    static const bool quiet_skip = (getenv("SPH_NO_QUIET_SKIP") == nullptr);
    my_level_next = 0;
    if (quiet_skip && !(my_adv_next & ADV_ACTIVE) && !anyNbrAdvActive(my_iter)) {
      int lo = ADV_LEVEL_CAP;
      for (int d = 0; d < NUM_DIRS; d++)
        if (valid_dir[d]) lo = std::min(lo, nbrLevel(d, my_iter));
      my_level_next = std::min(ADV_LEVEL_CAP, 1 + lo);
    }
    const int adv_out = my_adv_next | (my_level_next << ADV_LEVEL_SHIFT);
    const long long ts = sphNowNs();
    SendTimer st; const double tloop = send_warn_ms > 0 ? CkWallTimer() : 0.0;

    for (int d = 0; d < NUM_DIRS; d++) {
      if (d == STAY) continue;
      int cnt = moved ? h_counts[d] : 0;
      if (!valid_dir[d]) {
        // Beyond the tank wall: a particle heading there has tunnelled through
        // the boundary. Drop it and keep count.
        if (cnt > 0) escaped += cnt;
        cnt = 0;
        continue;
      }
      if (edgeSkips(d, my_iter)) {
        if (cnt > 0)
          CkAbort("Patch (%d,%d,%d) step %d: %d particle(s) bound for a quiet "
                  "edge (direction %d); the quiet-level proof is broken\n",
                  x, y, z, my_iter, cnt, d);
        n_skipped++;
        continue;
      }
      n_fluid -= cnt;   // every leaver is fluid
      // Sent every step the edge is not quiet, empty inside a window: the
      // advertisement rides on it.
      if (host_profile_on) {
        const int tier = peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[d], nbr_y[d], nbr_z[d])));
        g_hprof.sent[tier]++; g_hprof.sent_bytes[tier] += (long)cnt * sizeof(Particle);
      }
      st.begin();
      thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveParticles(my_iter, flipDir(d),
          adv_out, cnt, ts, cnt,
          (outstanding_sends++, sphSendBuffer(
           d_send_mig + dirOffset(d, exch_capacity),
           sendDoneCb(),
           comm_stream)));
      st.end(d);
    }
    st.report("sendLeavers", x, y, z, my_iter, send_warn_ms > 0 ? (CkWallTimer() - tloop) * 1000.0 : 0.0, stallTier(st.worst_d));
  }

  void receiveParticles(int ref, int dir, int adv, int n, long long ts, int& m,
      Particle*& parts, CkDeviceBufferPost* devicePost) {
    PhaseScope _ps(PH_RECV_PARTS);
    env_t[1][ref & 1][dir] = sphNowNs();
    if (host_profile_on) {
      const int tier = peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[dir], nbr_y[dir], nbr_z[dir])));
      g_hprof.recv[tier]++; g_hprof.recv_bytes[tier] += (long)n * sizeof(Particle);
    }
    // A partner can run ahead only over a quiet edge, by at most its level;
    // such a message is empty, and its advertisement has a slot of its own.
    if (ref > my_iter + ADV_LEVEL_CAP)
      CkAbort("Patch (%d,%d,%d) at step %d: migration from step %d -- a partner "
              "ran more than %d steps ahead, beyond the advertisement ring\n",
              x, y, z, my_iter, ref, ADV_LEVEL_CAP);
    nbr_adv[dir][(ref + 1) % ADV_RING] = adv;
    // Slot by the SENDER's step parity (see sph2d.C receiveParticles).
    parts = d_recv_mig + (size_t)(ref & 1) * exchSlots(exch_capacity) +
            dirOffset(dir, exch_capacity);
    devicePost[0].hapi_stream = comm_stream;
  }

  void appendParticles(int ref, int dir, int n) {
    PhaseScope _ps(PH_APPEND_PARTS);
    parts_arrived[my_iter]++;
    // Both ends now hold each other's advertisement for the next step and
    // compute the same skip count for this edge (see edgeSkips).
    xchg_step[dir] = my_iter;
    xchg_min[dir] = std::min(my_level_next,
                             ADV_LEVEL(nbr_adv[dir][(my_iter + 1) % ADV_RING]));
    if (n == 0) return;
    if (np + n > part_capacity)
      CkAbort("Patch (%d,%d,%d): %d particles exceed capacity %d at step %d; "
              "increase headroom (-r)\n", x, y, z, np + n, part_capacity, my_iter);
    sphSubmitMemcpy(d_parts[cur] + np,
        d_recv_mig + (size_t)(ref & 1) * exchSlots(exch_capacity) +
            dirOffset(dir, exch_capacity),
        sizeof(Particle) * n, cudaMemcpyDeviceToDevice, comm_stream);
    np += n;
    n_fluid += n;
  }

  // ---- concurrent exchange (-c) --------------------------------------------
  // The quiet-edge proof shifted by one step: a message for step s describes
  // s, so the last exchange over this edge (ref xchg_step+1) covered
  // xchg_min steps after itself.
  bool edgeSkipsC(int d, int s) const {
    return valid_dir[d] && s > xchg_step[d] + 1 && s - xchg_step[d] - 1 <= xchg_min[d];
  }
  int expectMigC(int s) const {
    int n = 0;
    for (int d = 0; d < NUM_DIRS; d++)
      if (valid_dir[d] && !edgeSkipsC(d, s)) n++;
    return n;
  }

  void beginStepC() {
    PhaseScope _ps(PH_START_HALO);
    rebuild_step = true; migrate_step = true; pc_step = false;
    my_adv = my_adv_next;
    step_active = (my_adv & ADV_ACTIVE) != 0;
    cells_valid = false;
    n_expect = 0;
    if (my_iter == 1) {
      // Prologue: open step 1 and pack its exchange from the initial state.
      open_ref = 1; np_base = np;
      n_leaver_ghosts = 0; lg_pending = false; ghost_alloc = 0; n_ghost = 0;
      for (int d = 0; d < NUM_DIRS; d++) { ghost_off[d] = 0; ghost_cnt[d] = -1; }
      n_expect_mig = expectMigC(1);
      mig_known = (mig_env_n[1] == n_expect_mig);
      if (mig_known) ghost_base = np_base + mig_env_total[1];
      halo_open = step_active;
      prologue = true;
    }
    n_expect_mig_cur = n_expect_mig;
    halo_expect_known = false;
    if (my_iter != 1) checkMigrantsDone(my_iter);
    if (my_iter == 1) {
      xpack_pending = true;
      for (int i = 0; i < NUM_COUNTERS; i++) h_counts_mig[i] = 0;
      h_counts_mig[STAY] = np;
      if (np > 0) {
        invokePackHalo(d_parts[cur], np, x0, y0, z0, x1, y1, z1, cell_size,
            d_halo_ptrs, d_halo_idx_ptrs, d_counts, exch_capacity, valid_mask,
            /*counts_zeroed=*/false, compute_stream);
        sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
            cudaMemcpyDeviceToHost, compute_stream);
        hapiAddCallback(compute_stream,
            CkCallback(CkIndex_Patch::exchangePacked(), thisProxy[thisIndex]));
      } else {
        for (int i = 0; i < NUM_COUNTERS; i++) h_counts[i] = 0;
        thisProxy[thisIndex].exchangePacked();
      }
    }
  }

  // Every migrant of this step is in (np = np_base + their total): place the
  // leaver ghosts behind them, and decide which halos to expect from the
  // advertisements they carried.
  void expectHalos() {
    PhaseScope _ps(PH_START_HALO);
    if (!mig_known)
      CkAbort("Patch (%d,%d,%d) step %d: migrants consumed but envelopes not all known\n", x, y, z, my_iter);
    if (np != np_base + mig_env_total[my_iter] || np != ghost_base)
      CkAbort("Patch (%d,%d,%d) step %d: %d locals but %d + %d expected (ghost base %d)\n", x, y, z, my_iter,
              np, np_base, mig_env_total[my_iter], ghost_base);
    if (n_leaver_ghosts > 0) {
      if (np + n_leaver_ghosts > part_capacity)
        CkAbort("Patch (%d,%d,%d): %d particles+ghosts exceed capacity %d at step %d; increase headroom (-r)\n",
                x, y, z, np + n_leaver_ghosts, part_capacity, my_iter);
      sphSubmitMemcpy(d_parts[cur] + np, d_leaver_ghosts, sizeof(Particle) * n_leaver_ghosts,
          cudaMemcpyDeviceToDevice, comm_stream);
    }
    lg_pending = false;
    mig_env_n.erase(my_iter); mig_env_total.erase(my_iter); mig_seen.erase(my_iter); mig_consumed.erase(my_iter);
    done_sent[my_iter] |= 1;
    n_expect = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
      halo_peer[d] = valid_dir[d] && !edgeSkipsC(d, my_iter) && (nbrAdv(d, my_iter) & ADV_ACTIVE);
      if (halo_peer[d]) n_expect++;
    }
  }

  // Forces and integration, then the compaction, the leaver ghosts, the pack
  // of the next step's halo and the counts, all on compute_stream; the
  // callback is exchangePacked.
  void computeForcesC() {
    PhaseScope _ps(PH_COMPUTE);
    halo_open = false;
    xpack_pending = true;
    static const bool batch_compute = (getenv("SPH_NO_BATCH_COMPUTE") == nullptr);
    if (batch_compute) hapiSubmitBatchBegin();
    hapiSubmitEventRecord((void*)halo_done, comm_stream);
    sphSubmitWaitEvent(compute_stream, halo_done);
    if (!step_active) {
      // Nothing moves: no leavers, the exchange is the current boundary.
      if (isStatsIter()) {
        runCheck(0.0f);
        idAuditPass();
        invokeStats(d_parts[cur], np, d_stats, compute_stream);
        sphSubmitMemcpy(h_stats, d_stats, sizeof(RealType) * 8, cudaMemcpyDeviceToHost, compute_stream);
      }
      for (int i = 0; i < NUM_COUNTERS; i++) h_counts_mig[i] = 0;
      h_counts_mig[STAY] = np;
      if (np > 0) {
        invokePackHalo(d_parts[cur], np, x0, y0, z0, x1, y1, z1, cell_size,
            d_halo_ptrs, d_halo_idx_ptrs, d_counts, exch_capacity, valid_mask,
            /*counts_zeroed=*/false, compute_stream);
        sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
            cudaMemcpyDeviceToHost, compute_stream);
        hapiAddCallback(compute_stream,
            CkCallback(CkIndex_Patch::exchangePacked(), thisProxy[thisIndex]));
      } else {
        for (int i = 0; i < NUM_COUNTERS; i++) h_counts[i] = 0;
        if (isStatsIter())
          hapiAddCallback(compute_stream,
              CkCallback(CkIndex_Patch::exchangePacked(), thisProxy[thisIndex]));
        else
          thisProxy[thisIndex].exchangePacked();
      }
      if (batch_compute) hapiSubmitBatchEnd();
      return;
    }
    if (isStatsIter()) { check_ghosts = true; runCheck(0.0f); check_ghosts = false; idAuditPass(); }
    const int ntot = np + n_ghost;
    invokeCellBuild(d_parts[cur], ntot, x0, y0, z0, inv_csize, ncx, ncy, ncz,
        ncells, d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, d_cell_of,
        d_scan_tmp, scan_tmp_bytes, compute_stream);
    cells_valid = true;
    invokeLayout(d_parts[cur], d_cell_parts, ntot, d_pos4, d_vel4, d_type8, compute_stream);
    invokeForces(0, ntot, np, ncx, ncy, ncz, d_cell_parts, d_cell_of, d_cell_off,
        d_cell_cnt, d_pos4, d_vel4, d_type8, smooth_h, pmass, sound_c0, gravity,
        d_drho, d_ax, d_ay, d_az, compute_stream);
    if (!isStatsIter()) {
      invokeIntegrateLeavers(false, d_parts[cur], np, sim_dt, rho0, sound_c0,
          d_drho, d_ax, d_ay, d_az, x0, y0, z0, x1, y1, z1, d_mig_ptrs,
          d_parts[1 - cur], d_counts_mig, exch_capacity, compute_stream);
    } else {
      invokeIntegrate(d_parts[cur], np, sim_dt, rho0, sound_c0, d_drho, d_ax,
          d_ay, d_az, NULL, compute_stream);
      invokeStats(d_parts[cur], np, d_stats, compute_stream);
      sphSubmitMemcpy(h_stats, d_stats, sizeof(RealType) * 8, cudaMemcpyDeviceToHost, compute_stream);
      invokeMarkLeavers(d_parts[cur], np, x0, y0, z0, x1, y1, z1, d_mig_ptrs,
          d_parts[1 - cur], d_counts_mig, exch_capacity, /*counts_zeroed=*/false, compute_stream);
    }
    invokeGatherLeaverGhosts(d_mig_ptrs, d_counts_mig, d_parts[1 - cur], d_leaver_ghosts,
        exch_capacity, exch_slots, valid_mask, compute_stream);
    invokePackHaloConcurrent(d_parts[1 - cur], d_counts_mig, np, x0, y0, z0, x1, y1, z1,
        cell_size, d_halo_ptrs, d_halo_idx_ptrs, d_counts, exch_capacity, valid_mask,
        compute_stream);
    sphSubmitMemcpy(h_counts_mig, d_counts_mig, sizeof(int) * NUM_COUNTERS,
        cudaMemcpyDeviceToHost, compute_stream);
    sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
        cudaMemcpyDeviceToHost, compute_stream);
    hapiAddCallback(compute_stream,
        CkCallback(CkIndex_Patch::exchangePacked(), thisProxy[thisIndex]));
    if (batch_compute) hapiSubmitBatchEnd();
  }

  // The exchange for the next step (or, in the prologue, for step 1): swap in
  // the survivors, decide the activity and the quiet level it advertises,
  // open the next step's landings, and send the migrants and the halo on
  // every edge that is not quiet.
  void sendExchange() {
    PhaseScope _ps(PH_SEND_LEAVERS);
    xpack_pending = false;
    const bool pro = prologue;   // step 1's exchange: the step is already open (beginStepC)
    const int ref = pro ? my_iter : my_iter + 1;
    if (h_counts[ERR_COUNTER] > 0 || h_counts_mig[ERR_COUNTER] > 0)
      CkAbort("Patch (%d,%d,%d): the exchange overflowed a per-direction buffer "
              "(face slot %d) at step %d; increase -e\n", x, y, z, exch_capacity, my_iter);
    int leavers_valid = 0, leavers_lost = 0;
    if (!pro && step_active) {
      cur = 1 - cur;
      np = h_counts_mig[STAY];
      for (int d = 0; d < NUM_DIRS; d++) {
        if (d == STAY) continue;
        if (valid_dir[d]) leavers_valid += h_counts_mig[d];
        else leavers_lost += h_counts_mig[d];   // tunnelled through the tank wall
      }
      n_fluid -= leavers_valid + leavers_lost;   // every leaver is fluid
      escaped += leavers_lost;
    }
    prologue = false;
    n_leaver_ghosts = leavers_valid;
    lg_pending = n_leaver_ghosts > 0;

    // Next step's activity, before its migrants are known (see the top). In
    // the prologue the step-1 activity is the seed that beginStepC already
    // applied: recomputing it here from neighbours' advertisements that may
    // or may not have arrived yet made an inactive patch open direct landings
    // it then refused to consume (job 22798305, sync smoke).
    static const bool quiet_skip = (getenv("SPH_NO_QUIET_SKIP") == nullptr);
    if (!pro) {
      const bool nbr_fluid = anyNbrAdvFluid(my_iter);
      my_adv_next = ((n_fluid > 0 || nbr_fluid) ? ADV_FLUID : 0) |
          ((np > 0 && (n_fluid > 0 || nbr_fluid)) ? ADV_ACTIVE : 0);
      my_level_next = 0;
      if (quiet_skip && !(my_adv_next & ADV_ACTIVE) && !anyNbrAdvActive(my_iter)) {
        int lo = ADV_LEVEL_CAP;
        for (int d = 0; d < NUM_DIRS; d++)
          if (valid_dir[d]) lo = std::min(lo, nbrLevel(d, my_iter));
        my_level_next = std::min(ADV_LEVEL_CAP, 1 + lo);
      }
    }
    const int adv_out = my_adv_next | (my_level_next << ADV_LEVEL_SHIFT);

    // Open the next step's landings (in the prologue, step 1 is open already
    // and its halos may have landed directly during the pack).
    if (!pro) {
      if (!halo_direct.empty())
        CkAbort("Patch (%d,%d,%d) step %d: a directly landed halo was never consumed\n", x, y, z, my_iter);
      open_ref = ref; np_base = np;
      ghost_alloc = n_ghost = n_leaver_ghosts;
      for (int d = 0; d < NUM_DIRS; d++) { ghost_off[d] = 0; ghost_cnt[d] = -1; }
      n_expect_mig = expectMigC(ref);
      mig_known = (mig_env_n[ref] == n_expect_mig);
      if (mig_known) ghost_base = np_base + mig_env_total[ref];
    }
    halo_open = (my_adv_next & ADV_ACTIVE) != 0;
    if (pro) { n_expect_mig = expectMigC(ref); mig_known = (mig_env_n[ref] == n_expect_mig); if (mig_known) ghost_base = np_base + mig_env_total[ref]; }
    checkMigrantsDone(ref);

    xsend_ref = ref; xsend_adv = adv_out; xsend_halo = halo_open;
    // The step before a park (AtSyncWait at the lag, or the sync AtSync) sends
    // nothing: a parked element with a move pending bounces device receives
    // until it has moved, and it only moves once nothing is in flight into it,
    // so exchanges sent to parked neighbours deadlocked the first LB (job
    // 22798305: every patch either parked with its step-36 exchange received
    // or draining 2-9 sends to parked ones). ResumeFromSync sends them.
    sendExchangeMsgs();   // (the park is inside the step now: parkOrComputeC)
  }

  void sendExchangeMsgs() {
    PhaseScope _ps(PH_SEND_LEAVERS);
    xsend_pending = false;
    const int ref = xsend_ref, adv_out = xsend_adv;
    const bool send_halo = xsend_halo;
    const long long ts = sphNowNs();
    SendTimer st; const double tloop = send_warn_ms > 0 ? CkWallTimer() : 0.0;
    for (int d = 0; d < NUM_DIRS; d++) {
      if (d == STAY || !valid_dir[d]) continue;
      if (edgeSkipsC(d, ref)) {
        if (h_counts_mig[d] > 0)
          CkAbort("Patch (%d,%d,%d) step %d: %d particle(s) bound for a quiet edge "
                  "(direction %d); the quiet-level proof is broken\n", x, y, z, my_iter, h_counts_mig[d], d);
        n_skipped++;
        continue;
      }
      const int cm = h_counts_mig[d];
      const int ch = send_halo ? h_counts[d] : 0;
      if (host_profile_on) {
        const int tier = peTier(thisProxy.ckLocMgr()->whichPe(CkArrayIndex3D(nbr_x[d], nbr_y[d], nbr_z[d])));
        g_hprof.sent[tier] += send_halo ? 2 : 1;
        g_hprof.sent_bytes[tier] += (long)(cm + ch) * sizeof(Particle);
      }
      st.begin();
      thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveParticlesC(ref, flipDir(d), adv_out, cm, ts, cm,
          (outstanding_sends++, sphSendBuffer(d_send_mig + dirOffset(d, exch_capacity), sendDoneCb(), comm_stream)));
      st.end(d);
      if (send_halo) {
        st.begin();
        thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveHaloC(ref, flipDir(d), ch, ts, ch,
            (outstanding_sends++, sphSendBuffer(d_send_halo + dirOffset(d, exch_capacity), sendDoneCb(), comm_stream)));
        st.end(d + 100);
      }
    }
    st.report("sendExchange", x, y, z, my_iter, send_warn_ms > 0 ? (CkWallTimer() - tloop) * 1000.0 : 0.0, stallTier(st.worst_d));
  }

  // The audit's bucket pass over this step's locals, at the point of the
  // official check; the sums reach the host by endOfStep, which contributes them.
  void idAuditPass() {
    if (!id_audit_on) return;
    invokeIdBuckets(d_parts[cur], np, d_aud_cnt, d_aud_sum, d_aud_sum2, compute_stream);
    sphSubmitMemcpy(h_aud, d_aud_cnt, sizeof(unsigned long long) * ID_AUDIT_BUCKETS, cudaMemcpyDeviceToHost, compute_stream);
    sphSubmitMemcpy(h_aud + ID_AUDIT_BUCKETS, d_aud_sum, sizeof(unsigned long long) * ID_AUDIT_BUCKETS, cudaMemcpyDeviceToHost, compute_stream);
    sphSubmitMemcpy(h_aud + 2 * ID_AUDIT_BUCKETS, d_aud_sum2, sizeof(unsigned long long) * ID_AUDIT_BUCKETS, cudaMemcpyDeviceToHost, compute_stream);
  }

  void dumpId(int id) {
    if (!id_audit_on) return;
    const int n = np + n_ghost;
    invokeFindId(d_parts[cur], n, id, d_find, compute_stream);
    int* h = takePinned<int>(1);
    sphSubmitMemcpy(h, d_find, sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
    hapiCheck(sphDrainSync(compute_stream));
    const int idx = *h;
    if (idx < n) {
      Particle q;
      sphSubmitMemcpy(&q, d_parts[cur] + idx, sizeof(Particle), cudaMemcpyDeviceToHost, compute_stream);
      hapiCheck(sphDrainSync(compute_stream));
      CkPrintf("[IDAUDIT] id %d: patch (%d,%d,%d) PE %d step %d index %d of np %d (%s) pos %.5f %.5f %.5f box [%.4f,%.4f)x[%.4f,%.4f)x[%.4f,%.4f) type %d\n",
               id, x, y, z, CkMyPe(), my_iter, idx, np, idx < np ? "LOCAL" : "ghost", q.x, q.y, q.z, x0, x1, y0, y1, z0, z1, q.type);
    }
    givePinned(h, 1);
  }

  void dumpState() {
    int staged_h = 0, staged_m = 0;
    for (auto& st : halo_staged) staged_h += (int)st.second.size();
    for (auto& st : mig_staged) staged_m += (int)st.second.size();
    int got_h = 0, got_m = 0;
    { auto it = halo_landed.find(my_iter); if (it != halo_landed.end()) got_h = popcount27(it->second); }
    { auto it = mig_landed.find(my_iter); if (it != mig_landed.end()) got_m = popcount27(it->second); }
    int env_n = 0, env_tot = 0;
    { auto it = mig_env_n.find(open_ref); if (it != mig_env_n.end()) env_n = it->second; }
    { auto it = mig_env_total.find(open_ref); if (it != mig_env_total.end()) env_tot = it->second; }
    CkPrintf("[STUCK] patch (%d,%d,%d) pe %d step %d active %d open_ref %d np %d base %d | halos got %d of %d (direct %d, staged %d) "
             "| migrants got %d of %d (envelopes %d, %d parts, known %d, staged %d) | sends out %d draining %d xpack %d xsend_deferred %d prologue %d lb_wait %d\n",
             x, y, z, CkMyPe(), my_iter, (int)step_active, open_ref, np, np_base, got_h, n_expect, (int)halo_direct.size(), staged_h,
             got_m, concurrent_x ? n_expect_mig_cur : n_expect_mig, env_n, env_tot, (int)mig_known, staged_m,
             outstanding_sends, (int)draining, (int)xpack_pending, (int)xsend_pending, (int)prologue, (int)lb_waiting);
  }

  // ---- end of step ---------------------------------------------------------
  void gateStepDrain() {
    PhaseScope _ps(PH_DRAIN);
    if (outstanding_sends == 0) {
      thisProxy[thisIndex].sendsDrained();
    } else {
      draining = true;
    }
  }

  // A send's completion runs this on the sending PE, in the scheduler between
  // entry methods, instead of posting a sendDone message the scheduler would
  // deliver later. The patch is found by index: it may have migrated with the
  // send in flight (the async window packs mid-step), and then the message
  // route is kept.
  static void* sendDoneParam(int px, int py, int pz) {
    return (void*)(uintptr_t)(((uint64_t)px << 40) | ((uint64_t)py << 20) | (uint64_t)pz);
  }
  static void sendDoneFn(void* param, void*) {
    const uint64_t v = (uint64_t)(uintptr_t)param;
    const int px = (int)(v >> 40), py = (int)((v >> 20) & 0xfffff), pz = (int)(v & 0xfffff);
    if (Patch* p = patch_proxy(px, py, pz).ckLocal()) p->sendDone();
    else patch_proxy(px, py, pz).sendDone();
  }
  CkCallback sendDoneCb() const {
    static const bool by_msg = (getenv("SPH_SENDDONE_MSG") != nullptr);  // A/B: the old route
    if (by_msg) return CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]);
    return CkCallback(&Patch::sendDoneFn, sendDoneParam(x, y, z));
  }

  void sendDone() {
    PhaseScope _ps(PH_SEND_DONE);
    outstanding_sends--;
    if (draining && outstanding_sends == 0) {
      draining = false;
      thisProxy[thisIndex].sendsDrained();
    }
  }

  void endOfStep() {
    PhaseScope _ps(PH_END_STEP);
    checkArrivals(halo_arrived, "halo", n_expect);
    checkArrivals(halo_arrived2, "predicted halo", pc_step ? n_expect : 0);
    checkArrivals(parts_arrived, "migration", concurrent_x ? n_expect_mig_cur : n_expect_mig);
    const bool reporting = isStatsIter();
    long sum_np = 0, max_np = 0, esc = escaped, n_active = isActive() ? 1 : 0;
    double sum_rho = 0, sum_ke = 0, n_fluid_d = 0, vmax = 0, front = 0;
    unsigned long long id_sum = 0, bnd_sum = 0, n_fl = 0, n_bd = 0;
    if (reporting) {
      sum_rho = h_stats[0];
      sum_ke = h_stats[1];
      n_fluid_d = h_stats[2];
      vmax = h_stats[3];
      front = h_stats[4];

      abortOnBadParticles(my_iter);
      id_sum = h_check[CHK_ID_SUM];
      bnd_sum = h_check[CHK_BND_SUM];
      n_fl = h_check[CHK_N_FLUID];
      n_bd = h_check[CHK_N_BOUND];
      sum_np = max_np = (long)(n_fl + n_bd);
    }

    CkReduction::tupleElement tuple[] = {
        CkReduction::tupleElement(sizeof(long), &sum_np, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(long), &max_np, CkReduction::max_long),
        CkReduction::tupleElement(sizeof(double), &sum_rho, CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double), &sum_ke, CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double), &n_fluid_d, CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double), &vmax, CkReduction::max_double),
        CkReduction::tupleElement(sizeof(double), &front, CkReduction::max_double),
        CkReduction::tupleElement(sizeof(unsigned long long), &id_sum,
            CkReduction::sum_ulong_long),
        CkReduction::tupleElement(sizeof(unsigned long long), &bnd_sum,
            CkReduction::sum_ulong_long),
        CkReduction::tupleElement(sizeof(unsigned long long), &n_fl,
            CkReduction::sum_ulong_long),
        CkReduction::tupleElement(sizeof(unsigned long long), &n_bd,
            CkReduction::sum_ulong_long),
        CkReduction::tupleElement(sizeof(long), &esc, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(long), &n_active, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(long), &n_skipped, CkReduction::sum_long)};
    if (reporting) {
      CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple, 14);
      msg->setCallback(CkCallback(CkIndex_Main::stepStats(NULL), main_proxy));
      contribute(msg);
      if (id_audit_on) {
        // (the bucket pass ran at the official check; its D2H completed with the step's stream work)
        hapiCheck(sphDrainSync(compute_stream));
        int st = my_iter;
        CkReduction::tupleElement at[] = {
            CkReduction::tupleElement(sizeof(unsigned long long) * ID_AUDIT_BUCKETS, h_aud, CkReduction::sum_ulong_long),
            CkReduction::tupleElement(sizeof(unsigned long long) * ID_AUDIT_BUCKETS, h_aud + ID_AUDIT_BUCKETS, CkReduction::sum_ulong_long),
            CkReduction::tupleElement(sizeof(unsigned long long) * ID_AUDIT_BUCKETS, h_aud + 2 * ID_AUDIT_BUCKETS, CkReduction::sum_ulong_long),
            CkReduction::tupleElement(sizeof(int), &st, CkReduction::max_int)};
        CkReductionMsg* am = CkReductionMsg::buildFromTuple(at, 4);
        am->setCallback(CkCallback(CkIndex_Main::idAudit(NULL), main_proxy));
        contribute(am);
      }
    }

    if (my_iter < warmup_iters + n_iters) {
      thisProxy[thisIndex].iterate();
    } else {
      if (escaped > 0)
        CkPrintf("[WARN] patch (%d,%d,%d) lost %ld particle(s) out of the tank "
                 "-- the run went past the edge of its validity\n", x, y, z,
                 escaped);
      contribute(CkCallback(CkReductionTarget(Main, allDone), main_proxy));
    }
  }
};

#include "sph3d.def.h"
