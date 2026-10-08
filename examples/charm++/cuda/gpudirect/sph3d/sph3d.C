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
#include <unistd.h>

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

extern size_t sphScanTempBytes(int);
extern void invokeCellBuild(const Particle*, int, RealType, RealType, RealType,
    RealType, int, int, int, int, int*, int*, int*, int*, int*, void*, size_t,
    cudaStream_t);
extern void invokeGather(const Particle*, const int*, int, Particle*, cudaStream_t);
extern void invokeLayout(const Particle*, const int*, int, float4*, float4*, cudaStream_t);
extern void invokeEOS(Particle*, int, RealType, RealType, int*, cudaStream_t);
extern void invokeForces(const Particle*, int, int, int, int, const int*,
    const int*, const int*, const float4*, const float4*, RealType, RealType,
    RealType, RealType, RealType*, RealType*, RealType*, RealType*, cudaStream_t);
extern void invokeIntegrate(Particle*, int, RealType, RealType, RealType, RealType*,
    RealType*, RealType*, RealType*, int*, cudaStream_t);
extern void invokeIntegrateLeavers(bool, const Particle*, int, RealType, RealType,
    RealType, RealType*, RealType*, RealType*, RealType*, RealType, RealType,
    RealType, RealType, RealType, RealType, Particle**, Particle*, int*, int,
    cudaStream_t);
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
    RealType headroom = 2.0f;
    RealType exch_frac = 0.1f;

    int c;
    while ((c = getopt(m->argc, m->argv, "X:Y:Z:x:y:z:s:w:t:V:i:u:f:b:l:ar:e:S:k:pm:")) != -1) {
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
                   "  -k [neighbour-list window, steps] -m [Verlet margin, in h]\n");
          CkExit();
      }
    }
    delete m;

    if (nl_freq < 1) CkAbort("-k (%d) must be at least 1\n", nl_freq);
    if (verlet_margin < 0.0f) CkAbort("-m (%g) must not be negative\n", verlet_margin);

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

    patch_proxy = CProxy_Patch::ckNew(n_chares_x, n_chares_y, n_chares_z);
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
               "wall %.1f s, %.1f ms/step over the last %d\n",
               step, sum_rho / n_fluid, sum_ke * pmass, vmax, front, imb,
               n_active, n_patches, now, (now - last_report_t) * 1000.0 / stats_freq,
               stats_freq);
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

    if (id_sum != ref_id_sum)
      CkAbort("CORRUPTION at step %d: the particle identity checksum changed, "
              "%016llx (expected %016llx), with the count right (%ld).\n",
              step, id_sum, ref_id_sum, sum_np);
  }

  void allDone() {
    const double elapsed = CkWallTimer() - start_time;
    const double per_step = elapsed / n_iters * 1000;
    const int evals = integrator ? 2 : 1;
    CkPrintf("Average iteration time: %.3f ms\n", per_step);
    CkPrintf("  %d force evaluation%s per step: %.3f ms per force evaluation\n",
             evals, evals > 1 ? "s" : "", per_step / evals);
    CkExit();
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
  int nbr_adv[NUM_DIRS][2];
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
    d_pos4 = d_vel4 = NULL; d_scan_tmp = NULL; scan_tmp_bytes = 0;
    d_drho = d_ax = d_ay = d_az = d_stats = NULL;
    d_check = NULL;
    h_counts = NULL; h_halo_cnt = NULL; h_stats = NULL; h_check = NULL;
  }

  void createCudaEntities() {
    compute_stream = hapiAcquireStream();
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
    hapiCheck(hapiMalloc((void**)&d_recv_halo, ecap));
    hapiCheck(hapiMalloc((void**)&d_recv_halo2, ecap));
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
    hapiCheck(hapiMalloc((void**)&d_pos4, sizeof(float4) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_vel4, sizeof(float4) * part_capacity));
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
    h_halo_cnt = takePinned<int>(NUM_DIRS);
    h_stats = takePinned<RealType>(8);
    h_check = takePinned<unsigned long long>(NUM_CHECKS);
    for (int i = 0; i < 8; i++) h_stats[i] = 0;
    for (int i = 0; i < NUM_CHECKS; i++) h_check[i] = 0;
    for (int i = 0; i < NUM_COUNTERS; i++) h_counts[i] = 0;
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
    hapiFree(d_pos4); hapiFree(d_vel4); hapiFree(d_scan_tmp);
    hapiFree(d_drho); hapiFree(d_ax); hapiFree(d_ay); hapiFree(d_az);
    hapiFree(d_stats); hapiFree(d_check);
    givePinned(h_counts, NUM_COUNTERS);
    givePinned(h_halo_cnt, NUM_DIRS);
    givePinned(h_stats, 8);
    givePinned(h_check, NUM_CHECKS);
    for (int i = 0; i < 2; i++) d_parts[i] = NULL;
    d_send_halo = d_send_halo2 = d_recv_halo = d_recv_halo2 = NULL;
    d_send_mig = d_recv_mig = NULL;
    d_halo_ptrs = d_halo_ptrs2 = d_mig_ptrs = NULL;
    d_halo_idx = NULL; d_halo_idx_ptrs = NULL; d_halo_cnt = NULL;
    d_counts = d_cell_cnt = d_cell_off = d_cursor = d_cell_parts = d_cell_of = NULL;
    d_pos4 = d_vel4 = NULL; d_scan_tmp = NULL; scan_tmp_bytes = 0;
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
    for (int d = 0; d < NUM_DIRS; d++) PUParray(p, nbr_adv[d], 2);
    PUParray(p, halo_peer, NUM_DIRS);
    PUParray(p, rebuild_peer, NUM_DIRS);
    PUParray(p, ghost_off, NUM_DIRS);
    PUParray(p, ghost_cnt, NUM_DIRS);
    p | halo_arrived; p | halo_arrived2; p | parts_arrived;
    p | n_ghost;
    p | ghost_alloc; p | halo_open; p | pc_open; p | halo_direct; p | halo_direct2;
    p | rebuild_step; p | migrate_step; p | pc_step; p | pc_pending;
    _sdag_pup(p);
    p | recv_count;
    p | outstanding_sends;
    p | draining;
    p | halo_pending; p | halo_pending2; p | mig_pending;

    if (!p.isUnpacking()) {
      sphDrainSync(compute_stream);
      sphDrainSync(comm_stream);
    }
    if (p.isUnpacking()) {
      setupGeometry();
      allocDevice();
      cells_valid = false;
    }

    PUParray(p, h_counts, NUM_COUNTERS);
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
      nbr_adv[d][0] = nbr_adv[d][1] = !valid_dir[d] ? 0
          : ((activeAtStart(nx, ny, nz) ? ADV_ACTIVE : 0) |
             (siteMask(nx, ny, nz) & ADV_FLUID));
    }
    my_adv = my_adv_next =
        (activeAtStart(x, y, z) ? ADV_ACTIVE : 0) | (n_fluid > 0 ? ADV_FLUID : 0);
  }
  int nbrAdv(int d, int t) const { return nbr_adv[d][t & 1]; }
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

  void iterate() {
    if (isLBIter(my_iter) && !lb_waiting) {
      lb_start_iter = my_iter;
      lb_t0 = CkWallTimer();
      if (!async_lb) {
        AtSync();
        return;
      }
      AtSyncStart();
      lb_waiting = true;
      thisProxy[thisIndex].runStep();
    } else if (lb_waiting && my_iter >= lb_start_iter + lb_wait_lag) {
      lb_waiting = false;
      if (x == 0 && y == 0 && z == 0)
        CkPrintf("  LB at step %d: AtSyncWait entered %.3f s after AtSyncStart\n",
                 lb_start_iter, CkWallTimer() - lb_t0);
      AtSyncWait();
    } else {
      thisProxy[thisIndex].runStep();
    }
  }

  void ResumeFromSync() {
    if (x == 0 && y == 0 && z == 0)
      CkPrintf("  LB at step %d: resumed %.3f s after %s; patch (0,0,0) now on PE %d\n",
               lb_start_iter, CkWallTimer() - lb_t0,
               async_lb ? "AtSyncStart" : "AtSync", CkMyPe());
    thisProxy[thisIndex].runStep();
  }

  // ---- phase 1: pressure, then halo ----------------------------------------
  void startHalo() {
    // Step 1 and every nl_freq-th step rebuild; the step before a rebuild
    // migrates. With nl_freq = 1 every step does both (the original scheme).
    rebuild_step = (my_iter == 1) || (my_iter % nl_freq == 0);
    migrate_step = ((my_iter + 1) % nl_freq == 0);
    my_adv = my_adv_next;
    step_active = (my_adv & ADV_ACTIVE) != 0;
    pc_step = (integrator == 1) && step_active;
    n_expect = 0;
    n_expect_mig = n_valid;
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
    static const bool sort_locals = (getenv("SPH_NO_SORT") == nullptr);
    if (batch_halo) hapiSubmitBatchBegin();
    // The migrants appended last step landed on comm_stream; everything below
    // reads them on compute_stream.
    hapiSubmitEventRecord((void*)halo_done, comm_stream);
    sphSubmitWaitEvent(compute_stream, halo_done);
    if (rebuild_step) {
      // Locals into cell order (gatherKernel in sph3d.cu), BEFORE the pack
      // records its index lists, so the lists survive until the compaction
      // at the end of the window. The swap is inside this serial block, so
      // cur names the live buffer at every entry-method boundary.
      if (sort_locals && np > 0) {
        invokeCellBuild(d_parts[cur], np, x0, y0, z0, inv_csize, ncx, ncy, ncz,
            ncells, d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, d_cell_of,
            d_scan_tmp, scan_tmp_bytes, compute_stream);
        invokeGather(d_parts[cur], d_cell_parts, np, d_parts[1 - cur], compute_stream);
        cur = 1 - cur;
      }
      invokePackHalo(d_parts[cur], np, x0, y0, z0, x1, y1, z1, cell_size,
          d_halo_ptrs, d_halo_idx_ptrs, d_counts, exch_capacity, valid_mask,
          /*counts_zeroed=*/false, compute_stream);
      sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
          cudaMemcpyDeviceToHost, compute_stream);
    } else {
      invokePackHaloByIndex(d_parts[cur], d_halo_ptrs, d_halo_idx, d_halo_cnt,
          exch_capacity, exch_slots, compute_stream);
    }
    halo_pending = true;
    hapiAddCallback(compute_stream,
        CkCallback(CkIndex_Patch::haloPacked(), thisProxy[thisIndex]));
    if (batch_halo) hapiSubmitBatchEnd();
  }

  void sendHalo() {
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
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!halo_peer[d]) continue;
      const int cnt = h_halo_cnt[d];
      thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveHalo(my_iter, flipDir(d), cnt, cnt,
          (outstanding_sends++, sphSendBuffer(
           d_send_halo + dirOffset(d, exch_capacity),
           CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
           comm_stream)));
    }
  }

  // A halo for THIS step lands straight in its ghost slot when that slot is
  // known and free (see sph2d.C receiveHalo). On a rebuild step the slot is
  // handed out in arrival order and recorded; inside a window it is the
  // recorded one, and the count has to match.
  void receiveHalo(int ref, int dir, int n, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    parts = d_recv_halo + dirOffset(dir, exch_capacity);
    static const bool direct_ok = (getenv("SPH_NO_DIRECT_HALO") == nullptr);
    static const bool ordered = (getenv("SPH_HALO_ORDERED") != nullptr);
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

  void appendGhosts(int dir, int n) {
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

    if (isStatsIter()) { check_ghosts = true; runCheck(stepCheckTol()); check_ghosts = false; }

    const int ntot = np + n_ghost;
    // A rebuild step builds the window's cell list over the sorted locals and
    // the ghosts; a chare that migrated inside a window rebuilds it locally.
    if (rebuild_step || !cells_valid) {
      invokeCellBuild(d_parts[cur], ntot, x0, y0, z0, inv_csize, ncx, ncy, ncz,
          ncells, d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, d_cell_of,
          d_scan_tmp, scan_tmp_bytes, compute_stream);
      cells_valid = true;
    }
    invokeLayout(d_parts[cur], d_cell_parts, ntot, d_pos4, d_vel4, compute_stream);
    invokeForces(d_parts[cur], np, ncx, ncy, ncz, d_cell_of, d_cell_off,
        d_cell_cnt, d_pos4, d_vel4, smooth_h, pmass, sound_c0, gravity,
        d_drho, d_ax, d_ay, d_az, compute_stream);

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
    halo_pending2 = false;
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!halo_peer[d]) continue;
      const int cnt = h_halo_cnt[d];
      thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveHalo2(my_iter, flipDir(d), cnt, cnt,
          (outstanding_sends++, sphSendBuffer(
           d_send_halo2 + dirOffset(d, exch_capacity),
           CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
           comm_stream)));
    }
  }

  // The predicted ghosts land in the same slots of the OTHER buffer. The slot
  // is known once this step's first exchange from that direction has been
  // posted; before that (or for a later step) the message is staged.
  void receiveHalo2(int ref, int dir, int n, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
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
    invokeLayout(d_parts[1 - cur], d_cell_parts, ntot, d_pos4, d_vel4, compute_stream);
    invokeForces(d_parts[1 - cur], np, ncx, ncy, ncz, d_cell_of, d_cell_off,
        d_cell_cnt, d_pos4, d_vel4, smooth_h, pmass, sound_c0, gravity,
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
  void sendLeavers() {
    const bool moved = isActive() && migrate_step;
    if (moved) {
      mig_pending = false;
      if (h_counts[ERR_COUNTER] > 0)
        CkAbort("Patch (%d,%d,%d): migration overflowed a per-direction buffer "
                "(face slot %d) at step %d; increase -e\n", x, y, z, exch_capacity, my_iter);
      cur = 1 - cur;
      np = h_counts[STAY];
    }
    // The ghost layout lives until the window ends; a migrate step ends it.
    if (migrate_step) { n_ghost = 0; ghost_alloc = 0; }
    my_adv_next = (n_fluid > 0 ? ADV_FLUID : 0) |
        ((np > 0 && (n_fluid > 0 || anyNbrAdvFluid(my_iter))) ? ADV_ACTIVE : 0);

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
      n_fluid -= cnt;   // every leaver is fluid
      // Sent every step, empty inside a window: the advertisement rides on it.
      thisProxy(nbr_x[d], nbr_y[d], nbr_z[d]).receiveParticles(my_iter, flipDir(d),
          my_adv_next, cnt, cnt,
          (outstanding_sends++, sphSendBuffer(
           d_send_mig + dirOffset(d, exch_capacity),
           CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
           comm_stream)));
    }
  }

  void receiveParticles(int ref, int dir, int adv, int n, int& m,
      Particle*& parts, CkDeviceBufferPost* devicePost) {
    if (ref > my_iter + 1)
      CkAbort("Patch (%d,%d,%d) at step %d: migration from step %d -- a partner "
              "ran more than one step ahead, which the two advertisement slots "
              "assume cannot happen\n", x, y, z, my_iter, ref);
    nbr_adv[dir][(ref + 1) & 1] = adv;
    // Slot by the SENDER's step parity (see sph2d.C receiveParticles).
    parts = d_recv_mig + (size_t)(ref & 1) * exchSlots(exch_capacity) +
            dirOffset(dir, exch_capacity);
    devicePost[0].hapi_stream = comm_stream;
  }

  void appendParticles(int ref, int dir, int n) {
    parts_arrived[my_iter]++;
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

  // ---- end of step ---------------------------------------------------------
  void gateStepDrain() {
    if (outstanding_sends == 0) {
      thisProxy[thisIndex].sendsDrained();
    } else {
      draining = true;
    }
  }

  void sendDone() {
    outstanding_sends--;
    if (draining && outstanding_sends == 0) {
      draining = false;
      thisProxy[thisIndex].sendsDrained();
    }
  }

  void endOfStep() {
    checkArrivals(halo_arrived, "halo", n_expect);
    checkArrivals(halo_arrived2, "predicted halo", pc_step ? n_expect : 0);
    checkArrivals(parts_arrived, "migration", n_expect_mig);
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
        CkReduction::tupleElement(sizeof(long), &n_active, CkReduction::sum_long)};
    if (reporting) {
      CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple, 13);
      msg->setCallback(CkCallback(CkIndex_Main::stepStats(NULL), main_proxy));
      contribute(msg);
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
