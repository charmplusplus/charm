#include "hapi.h"
#include "sph2d.decl.h"
#include "sph2d.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <climits>
#include <malloc.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// +gpusubmit (see sph2d.cu): copies, event records, stream waits and synchronizes
// on a patch's streams go through the submitter's queue. Arguments are evaluated
// at the call and captured by value. NOTE: the comm->compute dependency is an
// event record on one stream and a wait on the other; their order is the queue's
// order, which holds with ONE submitter (the default). With more than one the two
// streams can land on different submitters -- not supported here.
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

// Device buffers come from hapiMalloc/hapiFree, which are the device pool
// under +gpupool (an arena the peers have already opened, no driver call, no
// device sync) and cudaMalloc/cudaFree without it. The runtime owns the
// switch; the application does not see it.

// Pinned landing pads and the ordering event are recycled, never returned to
// the driver. cudaMallocHost / cudaFreeHost synchronize the device and take
// the driver lock, and cudaEventCreate / cudaEventDestroy take the lock: on a
// busy device each is tens of ms, and every one blocks the PE. A patch paid
// three pad frees and the event destroy at departure and three pad mallocs
// and the event create at arrival, so a move mid-step (async balancing) cost
// what a move at an idle barrier (sync balancing) never did. Measured on
// leanmd's one pinned double: 23 ms mean, 351 ms max per move (job 22081612).
// The pools are process-wide, one lock each; the PEs of a process share one
// device, so an event or a pad from any of them is valid on all.
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
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
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
/* readonly */ int part_capacity;
/* readonly */ int exch_capacity;
/* readonly */ int stats_freq;

extern void invokeCellBuild(const Particle*, int, RealType, RealType, RealType,
    int, int, int, int*, int*, int*, int*, cudaStream_t);
extern void invokeEOS(Particle*, int, RealType, RealType, int*, cudaStream_t);
extern void invokeForces(const Particle*, int, RealType, RealType, RealType,
    int, int, const int*, const int*, const int*, RealType, RealType, RealType,
    RealType, RealType*, RealType*, RealType*, cudaStream_t);
extern void invokeIntegrate(Particle*, int, RealType, RealType, RealType*,
    RealType*, RealType*, int*, cudaStream_t);
extern void invokePackHalo(const Particle*, int, RealType, RealType, RealType,
    RealType, RealType, Particle**, int*, int, bool, cudaStream_t);
extern void invokeMarkLeavers(const Particle*, int, RealType, RealType, RealType,
    RealType, Particle**, Particle*, int*, int, bool, cudaStream_t);
extern void invokeStats(const Particle*, int, RealType*, cudaStream_t);
extern void invokeCheck(const Particle*, int, RealType, RealType, RealType,
    RealType, RealType, RealType, unsigned long long*, cudaStream_t);

// The direction tag a message carries is the side of the RECEIVING patch it
// arrived on, so the sender flips its own direction on the way out.
static inline int flipDir(int d) {
  static const int f[NUM_DIRS] = {RIGHT, LEFT, BOTTOM, TOP, BR, BL, TR, TL};
  return f[d];
}
static const int DIR_DX[NUM_DIRS] = {-1, 1, 0, 0, -1, 1, -1, 1};
static const int DIR_DY[NUM_DIRS] = {0, 0, 1, -1, 1, 1, -1, -1};

class Main : public CBase_Main {
  double start_time;
  int stat_count;
  // Reference state, latched from the initial lattice. Every later report has
  // to reproduce these exactly; see the checks block in sph2d.h for why they
  // are integers.
  long n_total_particles;
  unsigned long long ref_id_sum, ref_bnd_sum, ref_n_fluid, ref_n_bound;
  bool conservation_void;   // set once a particle has legitimately left the domain
public:
  Main(CkArgMsg* m) : stat_count(0), conservation_void(false) {
    // Tank and fluid geometry are in metres; the default is the Koshizuka &
    // Oka dam break shape (column twice as tall as it is wide) scaled up so a
    // reasonable particle count fits.
    dom_lx = 4.0f; dom_ly = 3.0f;
    n_chares_x = 8; n_chares_y = 4;
    spacing = 0.02f;
    col_w = 1.0f; col_h = 2.0f;
    init_vx = 0.0f;
    rho0 = 1000.0f;
    gravity = 9.81f;
    // The acoustic CFL makes the timestep small: at the default spacing dt is
    // ~6e-5 s, and the load needs about 5000 steps to move one patch width --
    // the header prints that number for whatever geometry is asked for. Fluid
    // released from rest starts quadratically, so the first patch width is the
    // expensive one; -V starts it already moving, which turns the cost of a
    // patch crossing into a constant and is the way to get a load that keeps
    // moving inside a run of a few tens of thousands of steps.
    n_iters = 4000; warmup_iters = 10;
    first_lb = 99999; lb_freq = 0; async_lb = 0; lb_wait_lag = 3;
    stats_freq = 200;
    RealType headroom = 4.0f;
    RealType exch_frac = 0.25f;

    int c;
    while ((c = getopt(m->argc, m->argv, "X:Y:x:y:s:w:t:V:i:u:f:b:l:ar:e:S:")) != -1) {
      switch (c) {
        case 'X': dom_lx = atof(optarg); break;
        case 'Y': dom_ly = atof(optarg); break;
        case 'x': n_chares_x = atoi(optarg); break;
        case 'y': n_chares_y = atoi(optarg); break;
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
        default:
          CkPrintf("Usage: sph2d [-X domain width] [-Y domain height]\n"
                   "  -x [chares in x] -y [chares in y] -s [particle spacing]\n"
                   "  -w [column width] -t [column height]\n"
                   "  -V [initial fluid speed, +x, m/s]\n"
                   "  -i [iterations] -u [warmup] -S [stats every N steps]\n"
                   "  -f [first LB iter] -b [LB period]\n"
                   "  -a (async LB, needs +LBAsync) -l [wait lag]\n"
                   "  -r [capacity headroom] -e [exchange buffer fraction]\n");
          CkExit();
      }
    }
    delete m;

    // Wall thickness: three layers, which is what a Wendland support of 2h
    // with h = 1.3 dx needs to keep a fluid particle from seeing through the
    // boundary into the empty outside.
    wall_t = 3.0f * spacing;
    smooth_h = 1.3f * spacing;
    support = KERNEL_SUPPORT * smooth_h;
    pmass = rho0 * spacing * spacing;

    // Weakly compressible: the artificial sound speed is set to ten times the
    // fastest expected flow, which keeps the density variation near 1%. The
    // fastest flow is a particle that starts at init_vx and falls the height
    // of the column, so an imposed velocity has to be carried here too -- and
    // it is why a faster start does not buy proportionally faster physics:
    // raising init_vx raises c0, which shrinks the timestep by the same factor.
    const RealType v_free = std::sqrt(2.0f * gravity * col_h);
    const RealType v_max = std::sqrt(v_free * v_free + init_vx * init_vx);
    sound_c0 = 10.0f * v_max;
    // Acoustic CFL, plus the body-force limit. The acoustic one dominates here.
    const RealType dt_c = 0.15f * smooth_h / sound_c0;
    const RealType dt_g = 0.25f * std::sqrt(smooth_h / gravity);
    sim_dt = std::min(dt_c, dt_g);

    if (async_lb && !_lb_args.lbAsync()) {
      CkPrintf("[WARN] -a ignored: async LB needs +LBAsync on the command "
               "line. Running the unsplit AtSync barrier instead.\n");
      async_lb = 0;
    }
    // The lag must close the window before the NEXT trigger, and the first LB
    // need not sit on a period boundary: with -f 1000 -b 2000 the triggers are
    // 1000, 2000, 4000, 6000, so the shortest gap is 1000, not lb_freq. A lag
    // that straddles a trigger does not delay it -- iterate() drops it, because
    // it only starts a step while !lb_waiting -- so the async arm silently runs
    // FEWER balancing rounds than the sync arm and the two stop being
    // comparable. Checking against lb_freq alone let -l 1800 through and cost
    // async half its placements on every weak run.
    if (async_lb && lb_freq > 0) {
      int gap_after_first = lb_freq - (first_lb % lb_freq);
      int min_gap = gap_after_first < lb_freq ? gap_after_first : lb_freq;
      if (lb_wait_lag < 1 || lb_wait_lag >= min_gap)
        CkAbort("-l (%d) must be in [1, %d): the wait must land before the next "
                "LB trigger. First LB %d, period %d, so the triggers are %d, "
                "%d, %d, ... and the shortest gap is %d.\n",
                lb_wait_lag, min_gap, first_lb, lb_freq, first_lb,
                first_lb + gap_after_first, first_lb + gap_after_first + lb_freq,
                min_gap);
    } else if (async_lb && lb_wait_lag < 1) {
      CkAbort("-l (%d) must be at least 1\n", lb_wait_lag);
    }

    // Capacity. A patch's share of the fluid at rest is what it starts with;
    // the surge concentrates far more than that into the patches it runs
    // through, which is the whole point, so the headroom is generous.
    const RealType patch_area = (dom_lx / n_chares_x) * (dom_ly / n_chares_y);
    const int per_patch = (int)(patch_area / (spacing * spacing));
    part_capacity = std::max(4096, (int)(per_patch * headroom));
    exch_capacity = std::max(1024, (int)(part_capacity * exch_frac));

    // Also the stride of the global particle id, which is the lattice index
    // iy*lattice_nx + ix. Each lattice site falls in exactly one patch, so the
    // ids are unique without any communication at init.
    lattice_nx = (int)(dom_lx / spacing);
    const long lattice_x = lattice_nx;
    const long lattice_y = (long)(dom_ly / spacing);

    main_proxy = thisProxy;
    CkPrintf("SPH dam break: tank %.2f x %.2f m, spacing %.4f m, h %.4f m\n",
             dom_lx, dom_ly, spacing, smooth_h);
    CkPrintf("  column %.2f x %.2f m, rho0 %.0f, c0 %.2f m/s, dt %.3e s\n",
             col_w, col_h, rho0, sound_c0, sim_dt);
    // What the decomposition actually sees. From rest the fluid starts
    // quadratically, so the first patch width is the expensive one and every
    // later one is cheaper; with -V the fluid translates and each patch costs
    // the same. Either way the answer is fixed by patch width over spacing --
    // see the note on c0 above -- so this is the number that says whether a
    // run is long enough for the load to have moved at all.
    const RealType pw = dom_lx / n_chares_x;
    const double cross = init_vx > 0.0f
        ? pw / (init_vx * sim_dt)
        : std::sqrt(2.0 * pw / gravity) / sim_dt;
    CkPrintf("  fluid starts at %.2f m/s; the load crosses one patch (%.3f m) "
             "every ~%.0f steps\n", init_vx, pw, cross);
    CkPrintf("  lattice %ld x %ld, %d x %d patches, capacity/patch %d, "
             "exchange %d\n", lattice_x, lattice_y, n_chares_x, n_chares_y,
             part_capacity, exch_capacity);
    CkPrintf("  iterations %d (+%d warmup), first LB %d, LB period %d%s\n",
             n_iters, warmup_iters, first_lb, lb_freq,
             async_lb ? ", async" : "");

    patch_proxy = CProxy_Patch::ckNew(n_chares_x, n_chares_y);
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

  // sum np | max np | sum rho | sum KE | count fluid | max speed | front x
  //   | id sum | boundary sum | count fluid | count boundary | escaped
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
      // max/avg particles per patch is the imbalance a balancer can act on.
      const double imb = (double)max_np * (n_chares_x * n_chares_y) / (double)sum_np;
      // CHARM_SPH_MEM: where the memory is going, VmRSS against the C heap in
      // use. They move together here, which is what says the growth is malloc
      // and not pinned or device-mapped memory.
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
      CkPrintf("  step %6d: rho %.2f  KE %.4e  |v|max %.3f  front x %.3f  "
               "particle imbalance (max/avg) %.2f  active %ld/%d  [check ok]\n",
               step, sum_rho / n_fluid, sum_ke * pmass, vmax, front, imb,
               n_active, n_chares_x * n_chares_y);
    }
    stat_count++;
  }

  // The correctness check proper. Everything here is exact integer arithmetic
  // on order-invariant quantities, so it holds identically under no LB, sync
  // LB and async LB -- which is the point: any difference between those three
  // is a defect, not float noise.
  void checkState(int step, long sum_np, unsigned long long id_sum,
      unsigned long long bnd_sum, unsigned long long n_fl,
      unsigned long long n_bd, long escaped) {
    // The boundary particles are fixed geometry and never leave, so this one
    // holds for the whole run whatever the fluid does. It is also the check
    // that a migration's device-to-device pup is byte-exact, since boundary
    // and fluid particles are interleaved throughout the local array.
    if (bnd_sum != ref_bnd_sum || n_bd != ref_n_bound)
      CkAbort("CORRUPTION at step %d: the boundary particles changed. "
              "checksum %016llx (expected %016llx), count %llu (expected "
              "%llu). Boundary particles never move, so this is state "
              "damaged in transit -- the halo/migration path or a chare "
              "migration's device pup.\n",
              step, bnd_sum, ref_bnd_sum, n_bd, ref_n_bound);

    if (escaped > 0) {
      // Nothing below can hold once particles have genuinely left: the tank
      // has no lid, so a splash that clears dom_ly is gone from the
      // simulation. Say so once and stop claiming conservation, rather than
      // reporting a physics event as a runtime defect.
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
              "none left the domain. A particle only ever moves through the "
              "halo/migration exchange, so this is a dropped, duplicated or "
              "misrouted transfer.\n",
              step, sum_np, n_total_particles, n_fl, ref_n_fluid);

    if (id_sum != ref_id_sum)
      CkAbort("CORRUPTION at step %d: the particle identity checksum changed, "
              "%016llx (expected %016llx), with the count right (%ld) and "
              "nothing lost. One particle was substituted for another, or an "
              "id field was overwritten.\n",
              step, id_sum, ref_id_sum, sum_np);
  }

  void allDone() {
    const double elapsed = CkWallTimer() - start_time;
    CkPrintf("Average iteration time: %.3f ms\n", elapsed / n_iters * 1000);
    CkExit();
  }
};

class Patch : public CBase_Patch {
  Patch_SDAG_CODE

  int x, y;                       // my index
  RealType x0, y0, x1, y1;        // my rectangle
  int nbr_x[NUM_DIRS], nbr_y[NUM_DIRS];
  bool valid_dir[NUM_DIRS];       // false where the neighbour is outside the tank

  int np;                         // local particles
  int n_ghost;                    // ghosts appended after them this step
  int cur;                        // which particle buffer is live
  int my_iter;
  int recv_count;
  int outstanding_sends;
  bool draining;
  // A pack kernel has filled a send buffer but the send has not been issued
  // yet -- the window between the pack and its hapiAddCallback. An element
  // that moves inside it must carry the packed bytes, or the destination
  // sends whatever its freshly allocated buffer happens to hold.
  bool halo_pending, mig_pending;
  bool lb_waiting;
  int lb_start_iter;
  double lb_t0;             // wall time of this patch's last AtSync/AtSyncStart
  long escaped;                   // particles that left the tank (instability)

  // ---- activity ------------------------------------------------------------
  // Five sixths of the tank is air, and a patch with nothing in it used to pay
  // the whole step anyway: memsets, the scan kernel, two copies, sixteen empty
  // messages and their callbacks -- 0.3-0.4 ms of host time against 0.55 for a
  // patch holding 60k particles (pinned, 12 Sep 2026). That flat cost is why
  // no balancer could find anything to move at that size. What is dropped here
  // is everything on that list except the sixteen messages, which is the part
  // that cannot be dropped without breaking the step -- see below.
  //
  // A patch is ACTIVE for a step if it holds fluid or a neighbour does; wall
  // particles never move, so a patch of wall alone next to air is not. Being
  // inactive means skipping the DEVICE WORK only -- the kernels, the packs and
  // the copies, which is all of the cost. The exchange itself is unconditional:
  // every patch sends a halo and a migration message to each of its eight
  // partners every step and waits for all eight of each, carrying a count of
  // zero when it has nothing. That is what makes the step sound. A patch with
  // no fluid can still be handed particles by a neighbour, and if it were free
  // to run on it would receive them at a step it had already left behind; an
  // agreed activation step cannot fix that, it can only bound how far ahead it
  // gets. So no patch ever runs ahead of its neighbours, and activity is then
  // a purely local decision that needs no agreement at all.
  //
  // Each halo message carries whether its sender holds fluid, so a patch knows
  // its neighbourhood one step in arrears. Fluid enters a patch only from a
  // neighbour that has held fluid for the thousands of steps it took to cross
  // its own patch, and that neighbour's ring is already active, so a step of
  // lag never costs a ghost that was within a support radius of anything.
  int n_fluid;                    // fluid particles here, tracked exactly
  // Advertisement, carried on every migration message. ADV_ACTIVE is the
  // sender's own answer for its next step -- its neighbours cannot work that
  // out, since it turns on ITS neighbours -- and ADV_FLUID is what a patch
  // needs to work out its own.
  // An advertisement sent with step s applies to step s+1, and it is held in
  // the slot for that step. Without the slots a neighbour that has nothing to
  // do races a step ahead -- it waits on no halo -- and overwrites the value
  // with its NEXT one before this patch has latched against it. The two then
  // disagree about whether a halo is coming and the step never completes.
  // One step of skew is all the migration exchange allows, so two slots are
  // enough; the run aborts below if that assumption ever fails.
  int my_adv, my_adv_next;
  int nbr_adv[NUM_DIRS][2];
  bool step_active;               // latched at startHalo, fixed for the step
  bool halo_peer[NUM_DIRS];       // latched with it: who exchanges ghosts
  int n_valid;                    // partners inside the tank
  int n_expect;                   // halo partners this step
  int n_expect_mig;               // == n_valid, every step, for all time
  std::map<int, int> halo_arrived, parts_arrived;   // per step, for the check
  // Halos of the CURRENT iteration land straight in the particle array's tail
  // (receiveHalo), so appendGhosts has no copy to issue for them. One running
  // offset serves both those and the slot-and-copy fallback, so the ghost region
  // stays contiguous. SPH_NO_DIRECT_HALO=1 keeps every halo on the old path.
  int ghost_alloc = 0;
  bool halo_open = false;
  std::map<int, int> halo_direct;   // dir -> offset of a halo already in place

  // cell list geometry
  int ncx, ncy, ncells;
  RealType inv_csize;

  Particle* d_parts[2];
  Particle* d_send_halo;          // NUM_DIRS * exch_capacity
  Particle* d_recv_halo;
  Particle* d_send_mig;
  Particle* d_recv_mig;
  Particle** d_halo_ptrs;         // device array of the 8 send-buffer bases
  Particle** d_mig_ptrs;
  int* d_counts;
  int* d_cell_cnt;
  int* d_cell_off;
  int* d_cursor;
  int* d_cell_parts;
  RealType *d_drho, *d_ax, *d_ay, *d_stats;
  unsigned long long* d_check;
  int* h_counts;
  RealType* h_stats;
  unsigned long long* h_check;

  cudaStream_t compute_stream, comm_stream;
  // Orders the physics on compute_stream behind the ghost copies that
  // landed on comm_stream.
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
    // Streams and the ordering event belong to the object's lifetime, created
    // here and destroyed in ~Patch. pup must NOT destroy them: Charm++ runs
    // the source object's destructor after packing it, and a destructor that
    // synchronizes an already-destroyed stream leaves cudaErrorInvalidResource
    // Handle as this thread's sticky error -- which then aborts whichever
    // unrelated chare next calls cudaPeekAtLastError.
  }

  ~Patch() {
    // No drain here. Settling the streams is pup's job, and only on the
    // packing pass -- that is the one moment the device state has to be
    // quiesced, and only the elements that actually move pay for it.
    //
    // Nor is there a "leak the buffers if sends are still outstanding" path,
    // and no assertion on outstanding_sends either.
    //
    // outstanding_sends is NOT the migration-safety condition. It counts this
    // element's sendDone callbacks, and deviceSendReleaseFn delivers those with
    // cb.send() -- an asynchronous message -- while decrementing the runtime's
    // own outstandingDeviceSends synchronously at transport completion. So the
    // runtime can correctly conclude the element is device-quiet and move it
    // while our count is still non-zero, purely because our notification is
    // still in the queue. The buffers really are free at that point.
    //
    // Migration safety is the runtime's gate (outstandingDeviceSends == 0), not
    // ours. Our counter's only job is the step's drain, so that the next step
    // does not repack a send buffer whose callback has not come back yet.
    freeDevice();
    // Hand the streams back so they are recycled rather than leaked or
    // destroyed; the runtime returns each to its own device's free list, which
    // is what makes a chare that migrates between GPUs safe.
    hapiReleaseStream(compute_stream);
    hapiReleaseStream(comm_stream);
    eventPool().give(halo_done);
  }

  // Everything derived from the array index. Recomputed on the destination
  // after a migration rather than trusted from the migration constructor,
  // where thisIndex is not reliably established yet -- a garbage ncells sizes
  // the cell arrays wrongly and a garbage x0 sends cellOf out of range, which
  // shows up as an out-of-bounds write in cellScatterKernel and then as an
  // invalid-handle error on some later launch.
  void setupGeometry() {
    x = thisIndex.x; y = thisIndex.y;
    const RealType px = dom_lx / n_chares_x, py = dom_ly / n_chares_y;
    x0 = x * px; x1 = (x + 1) * px;
    y0 = y * py; y1 = (y + 1) * py;

    for (int d = 0; d < NUM_DIRS; d++) {
      const int nx = x + DIR_DX[d], ny = y + DIR_DY[d];
      // The array index is wrapped so every patch has eight partners and the
      // step always consumes eight messages; a partner across the tank wall is
      // marked invalid and is sent nothing. Wrapping without this would inject
      // ghosts from the far side of the tank.
      valid_dir[d] = (nx >= 0 && nx < n_chares_x && ny >= 0 && ny < n_chares_y);
      nbr_x[d] = (nx + n_chares_x) % n_chares_x;
      nbr_y[d] = (ny + n_chares_y) % n_chares_y;
    }
    n_valid = 0;
    for (int d = 0; d < NUM_DIRS; d++) if (valid_dir[d]) n_valid++;

    // Cells are one kernel support wide, so neighbours are in the 3x3 block
    // around a particle's own cell.
    ncx = std::max(1, (int)std::ceil((x1 - x0) / support));
    ncy = std::max(1, (int)std::ceil((y1 - y0) / support));
    ncells = (ncx + 2) * (ncy + 2);
    inv_csize = 1.0f / support;
  }

  void setupState() {
    np = n_ghost = 0; cur = 0; my_iter = 0;
    outstanding_sends = 0; draining = false;
    halo_pending = mig_pending = false;
    lb_waiting = false; lb_start_iter = 0; escaped = 0;
    n_fluid = 0; step_active = false; n_expect = 0; n_expect_mig = 0;
    my_adv = my_adv_next = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
      nbr_adv[d][0] = nbr_adv[d][1] = 0;
      halo_peer[d] = false;
    }
    for (int i = 0; i < 2; i++) d_parts[i] = NULL;
    d_send_halo = d_recv_halo = d_send_mig = d_recv_mig = NULL;
    d_halo_ptrs = d_mig_ptrs = NULL;
    d_counts = d_cell_cnt = d_cell_off = d_cursor = d_cell_parts = NULL;
    d_drho = d_ax = d_ay = d_stats = NULL;
    d_check = NULL;
    h_counts = NULL; h_stats = NULL; h_check = NULL;
  }

  void createCudaEntities() {
    // Streams come from the runtime's per-device pool: it hands back one that
    // belongs to this PE's device, and takes it back on migration. The event is
    // ours, created here on the same device.
    compute_stream = hapiAcquireStream();
    comm_stream = hapiAcquireStream();
    halo_done = eventPool().take();
  }

  void allocDevice() {
    const size_t pcap = sizeof(Particle) * (size_t)part_capacity;
    const size_t ecap = sizeof(Particle) * (size_t)NUM_DIRS * exch_capacity;
    hapiCheck(hapiMalloc((void**)&d_parts[0], pcap));
    hapiCheck(hapiMalloc((void**)&d_parts[1], pcap));
    hapiCheck(hapiMalloc((void**)&d_send_halo, ecap));
    hapiCheck(hapiMalloc((void**)&d_recv_halo, ecap));
    hapiCheck(hapiMalloc((void**)&d_send_mig, ecap));
    // Twice the directions: the migration receive slot is indexed by the
    // SENDING step's parity. See receiveParticles for why it has to be, and
    // why d_recv_halo above does not.
    hapiCheck(hapiMalloc((void**)&d_recv_mig, 2 * ecap));
    hapiCheck(hapiMalloc((void**)&d_halo_ptrs, sizeof(Particle*) * NUM_DIRS));
    hapiCheck(hapiMalloc((void**)&d_mig_ptrs, sizeof(Particle*) * NUM_DIRS));
    hapiCheck(hapiMalloc((void**)&d_counts, sizeof(int) * NUM_COUNTERS));
    hapiCheck(hapiMalloc((void**)&d_cell_cnt, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cell_off, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cursor, sizeof(int) * ncells));
    hapiCheck(hapiMalloc((void**)&d_cell_parts, sizeof(int) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_drho, sizeof(RealType) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_ax, sizeof(RealType) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_ay, sizeof(RealType) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_stats, sizeof(RealType) * 8));
    hapiCheck(hapiMalloc((void**)&d_check,
        sizeof(unsigned long long) * NUM_CHECKS));
    h_counts = takePinned<int>(NUM_COUNTERS);
    h_stats = takePinned<RealType>(8);
    h_check = takePinned<unsigned long long>(NUM_CHECKS);
    // An inactive patch never runs the stats or check kernels; it reports
    // whatever these last held, which for one that never ran is these zeros.
    for (int i = 0; i < 8; i++) h_stats[i] = 0;
    for (int i = 0; i < NUM_CHECKS; i++) h_check[i] = 0;
    for (int i = 0; i < NUM_COUNTERS; i++) h_counts[i] = 0;

    // The per-direction base pointers the pack kernels scatter into.
    // Same ordering argument as the lattice upload in init(): these are read by
    // the pack kernels on compute_stream, so they are written there too. hp is
    // on the stack, hence the sync before it goes out of scope.
    Particle* hp[NUM_DIRS];
    for (int d = 0; d < NUM_DIRS; d++) hp[d] = d_send_halo + (size_t)d * exch_capacity;
    sphSubmitMemcpy(d_halo_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice,
        compute_stream);
    for (int d = 0; d < NUM_DIRS; d++) hp[d] = d_send_mig + (size_t)d * exch_capacity;
    sphSubmitMemcpy(d_mig_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice,
        compute_stream);
    hapiCheck(sphDrainSync(compute_stream));
  }

  void freeDevice() {
    hapiFree(d_parts[0]); hapiFree(d_parts[1]);
    hapiFree(d_send_halo); hapiFree(d_recv_halo);
    hapiFree(d_send_mig); hapiFree(d_recv_mig);
    hapiFree(d_halo_ptrs); hapiFree(d_mig_ptrs);
    hapiFree(d_counts); hapiFree(d_cell_cnt); hapiFree(d_cell_off);
    hapiFree(d_cursor); hapiFree(d_cell_parts);
    hapiFree(d_drho); hapiFree(d_ax); hapiFree(d_ay); hapiFree(d_stats);
    hapiFree(d_check);
    givePinned(h_counts, NUM_COUNTERS);
    givePinned(h_stats, 8);
    givePinned(h_check, NUM_CHECKS);
    for (int i = 0; i < 2; i++) d_parts[i] = NULL;
    d_send_halo = d_recv_halo = d_send_mig = d_recv_mig = NULL;
    d_halo_ptrs = d_mig_ptrs = NULL;
    d_counts = d_cell_cnt = d_cell_off = d_cursor = d_cell_parts = NULL;
    d_drho = d_ax = d_ay = d_stats = NULL;
    d_check = NULL;
    h_counts = NULL; h_stats = NULL; h_check = NULL;
  }

  // Migration: the only state worth moving is the particles. The scratch
  // arrays and the cell list are rebuilt from scratch every step anyway.
  // What travels is whatever is live at the WORST entry-method boundary, not
  // what is live at the end of a step: under async LB the element can be moved
  // at any of them, and _sdag_pup brings the continuation with it, so it
  // resumes wherever it left off. Everything below is state some continuation
  // reads after a boundary it could have moved across.
  void pup(PUP::er& p) {
    CBase_Patch::pup(p);
    p | my_iter; p | np; p | cur; p | lb_waiting; p | lb_start_iter; p | lb_t0;
    p | escaped;
    p | n_fluid; p | step_active; p | n_valid; p | n_expect; p | n_expect_mig;
    p | my_adv; p | my_adv_next;
    for (int d = 0; d < NUM_DIRS; d++) PUParray(p, nbr_adv[d], 2);
    PUParray(p, halo_peer, NUM_DIRS);
    p | halo_arrived; p | parts_arrived;
    // n_ghost, not reset: a move partway through the ghost-receive loop leaves
    // ghosts already appended above np, and the remaining receives have to
    // append after them.
    p | n_ghost;
    // The tail handed out to halos landing DIRECTLY in the particle array
    // (receiveHalo), which n_ghost does not count until appendGhosts runs: a
    // move between landing and append must carry those particles too, and the
    // per-direction offsets so the appends on the new PE skip their copies.
    p | ghost_alloc; p | halo_open; p | halo_direct;
    // Which whens are outstanding and any buffered ordinary messages: under
    // async LB an element can be moved mid-step and its continuations must
    // follow it.
    _sdag_pup(p);
    p | recv_count;
    p | outstanding_sends;
    // draining, not reset either. The runtime may move an element whose
    // transport is complete but whose sendDone callbacks are still queued;
    // those follow it here. Clearing the flag loses the fact that the step is
    // waiting on them, so the last decrement fires nothing and the step never
    // ends.
    p | draining;
    p | halo_pending; p | mig_pending;

    // Particles migrate DEVICE TO DEVICE. Staging them through the host --
    // D2H, PUParray, H2D -- works but pays a full host round trip per particle
    // and, worse, never exercises the device migration path this example exists
    // to benchmark. The device overload takes flat fundamental arrays, so the
    // particles go as floats.
    if (!p.isUnpacking()) {
      // Settle the device before ANY size is read, not only before packing:
      // the sizes below come from h_counts, which a device-to-host copy on
      // compute_stream fills. A sizer that ran ahead of that copy would
      // report different lengths than the packer and trip the pup direction
      // mismatch check.
      sphDrainSync(compute_stream);
      sphDrainSync(comm_stream);
    }
    if (p.isUnpacking()) {
      setupGeometry();          // thisIndex is valid here, unlike in the ctor
      allocDevice();
    }

    // The pinned landing pads for the device-to-host copies. The copies were
    // enqueued before the callback that resumes the step, so at a migration in
    // between they hold results the destination is about to read -- the
    // per-direction counts sendHalo/sendLeavers act on above all. These are
    // host allocations, so nothing carries them but this.
    PUParray(p, h_counts, NUM_COUNTERS);
    PUParray(p, h_stats, 8);
    PUParray(p, h_check, NUM_CHECKS);

    const size_t per = sizeof(Particle) / sizeof(RealType);
    if (mig_pending) {
      // The compaction has ALREADY run at this boundary: markLeavers wrote the
      // survivors into d_parts[1-cur], and the sendLeavers that is about to
      // run on the destination promotes that buffer to current and discards
      // d_parts[cur]. So the live particles are the ones in the other buffer,
      // h_counts[STAY] of them. Carrying d_parts[cur] here instead ships the
      // array that is about to be thrown away and leaves the destination
      // promoting an allocation it never wrote -- which reads back as tens of
      // thousands of nonsense particles and blows up as an exchange overflow a
      // few steps later.
      p((RealType*)d_parts[1 - cur],
        (size_t)std::max(h_counts[STAY], 0) * per, PUP::PUPMode::DEVICE);
    } else {
      // Locals AND the ghosts after them, for the reason above -- the whole
      // tail handed out so far (ghost_alloc >= n_ghost: it also counts halos
      // that have landed directly but are not yet appended).
      p((RealType*)d_parts[cur], (size_t)(np + std::max(n_ghost, ghost_alloc)) * per,
        PUP::PUPMode::DEVICE);
    }

    // A packed-but-unsent send buffer, if the move landed in that window.
    // Only the prefix each direction actually filled: the buffer is eight
    // slots of exch_capacity and carrying it whole would cost more than the
    // particles do. h_counts is unpacked above, so both sides agree on the
    // lengths.
    if (halo_pending || mig_pending) {
      Particle* base = halo_pending ? d_send_halo : d_send_mig;
      for (int d = 0; d < NUM_DIRS; d++) {
        const size_t cnt = (size_t)std::max(h_counts[d], 0);
        if (cnt == 0) continue;
        p((RealType*)(base + (size_t)d * exch_capacity), cnt * per,
          PUP::PUPMode::DEVICE);
      }
    }
  }

  // Lay down the lattice, keeping only what falls in my rectangle. Each patch
  // walks only the lattice indices that overlap it, so this is O(local).
  void init() {
    const RealType s = spacing;
    const int ix_lo = std::max(0, (int)std::floor(x0 / s) - 1);
    const int ix_hi = (int)std::ceil(x1 / s) + 1;
    const int iy_lo = std::max(0, (int)std::floor(y0 / s) - 1);
    const int iy_hi = (int)std::ceil(y1 / s) + 1;

    std::vector<Particle> mine;
    for (int iy = iy_lo; iy <= iy_hi; iy++) {
      const RealType py = (iy + 0.5f) * s;
      if (py >= dom_ly) continue;
      for (int ix = ix_lo; ix <= ix_hi; ix++) {
        const RealType px = (ix + 0.5f) * s;
        if (px >= dom_lx) continue;
        if (px < x0 || px >= x1 || py < y0 || py >= y1) continue;

        const bool is_wall = (px < wall_t) || (px > dom_lx - wall_t) ||
                             (py < wall_t);
        const bool in_col = !is_wall && px >= wall_t && px < wall_t + col_w &&
                            py >= wall_t && py < wall_t + col_h;
        if (!is_wall && !in_col) continue;   // air

        Particle q;
        q.x = px; q.y = py;
        q.vx = is_wall ? 0.0f : init_vx; q.vy = 0.0f;
        q.rho = rho0; q.p = 0.0f;
        q.type = is_wall ? PTYPE_BOUND : PTYPE_FLUID;
        // Global lattice id. Unique without communication, because a lattice
        // site is created by exactly the one patch whose rectangle contains
        // it, and fixed for the run -- it is what the identity checksum and
        // the boundary checksum are built on.
        q.id = iy * lattice_nx + ix;
        mine.push_back(q);
      }
    }

    np = (int)mine.size();
    if (np > part_capacity)
      CkAbort("Patch (%d,%d): %d initial particles exceed capacity %d; "
              "increase headroom (-r)\n", x, y, np, part_capacity);
    n_fluid = 0;
    for (const Particle& q : mine) if (q.type == PTYPE_FLUID) n_fluid++;
    seedActivity();
    // On compute_stream, NOT the null stream. The runtime's pool hands out
    // cudaStreamNonBlocking streams, so null-stream work no longer implicitly
    // orders against them -- and a pageable host-to-device cudaMemcpy returns
    // once the buffer is staged, with the DMA still in flight. Uploading there
    // and then launching the first kernel on compute_stream is a race that
    // stays invisible while a patch holds a few hundred particles and starts
    // handing out uninitialised particles once the transfer is megabytes.
    if (np > 0)
      sphSubmitMemcpy(d_parts[cur], mine.data(),
          sizeof(Particle) * np, cudaMemcpyHostToDevice, compute_stream);

    // The reference values come from the initial lattice on the device, by the
    // same kernel that will recompute them every stats step -- so the check is
    // comparing like with like, and covers the run from step 0 rather than
    // from wherever the first report happens to fall. A blocking sync is fine
    // here; this runs once, before any timing starts.
    runCheck();
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

  // Enqueued on compute_stream; the result lands in h_check.
  void runCheck() {
    invokeCheck(d_parts[cur], np, x0, y0, x1, y1, rho0, sound_c0, d_check,
        compute_stream);
    sphSubmitMemcpy(h_check, d_check,
        sizeof(unsigned long long) * NUM_CHECKS, cudaMemcpyDeviceToHost,
        compute_stream);
  }

  // Per-particle validity. Aborted here rather than through the reduction so
  // the message names the patch that actually holds the bad particles, which
  // is what a corruption hunt needs first.
  void abortOnBadParticles(int step) {
    const unsigned long long mask = h_check[CHK_BAD_MASK];
    if (!mask) return;
    CkAbort("CORRUPTION at step %d, patch (%d,%d) on PE %d: %llu of %d local "
            "particle(s) are invalid --%s%s%s%s%s\n",
            step, x, y, CkMyPe(), h_check[CHK_BAD_COUNT], np,
            (mask & CHK_BAD_NAN)     ? " non-finite state;" : "",
            (mask & CHK_BAD_RHO)     ? " density far outside the weakly-"
                                       "compressible band;" : "",
            (mask & CHK_BAD_SPEED)   ? " speed past Mach 0.5;" : "",
            (mask & CHK_BAD_TYPE)    ? " type field neither fluid nor "
                                       "boundary;" : "",
            (mask & CHK_BAD_OUTSIDE) ? " particle outside this patch's own "
                                       "rectangle, i.e. an exchange delivered "
                                       "it to the wrong neighbour;" : "");
  }

  bool isLBIter(int it) const {
    return it == first_lb || (it != 0 && lb_freq > 0 && it % lb_freq == 0);
  }

  // ---- activity ------------------------------------------------------------
  // What patch (px,py) starts with, by the rule init() lays the lattice down
  // with, so every patch computes the same answer about every patch.
  // Bit 0: it holds particles at all. Bit 1: some of them are fluid.
  static int siteMask(int px, int py) {
    if (px < 0 || px >= n_chares_x || py < 0 || py >= n_chares_y) return 0;
    const RealType pw = dom_lx / n_chares_x, ph = dom_ly / n_chares_y;
    const RealType ax0 = px * pw, ax1 = (px + 1) * pw;
    const RealType ay0 = py * ph, ay1 = (py + 1) * ph;
    const RealType s = spacing;
    const int ix_lo = std::max(0, (int)std::floor(ax0 / s) - 1);
    const int ix_hi = (int)std::ceil(ax1 / s) + 1;
    const int iy_lo = std::max(0, (int)std::floor(ay0 / s) - 1);
    const int iy_hi = (int)std::ceil(ay1 / s) + 1;
    int mask = 0;
    for (int iy = iy_lo; iy <= iy_hi && mask != 3; iy++) {
      const RealType qy = (iy + 0.5f) * s;
      if (qy >= dom_ly || qy < ay0 || qy >= ay1) continue;
      for (int ix = ix_lo; ix <= ix_hi; ix++) {
        const RealType qx = (ix + 0.5f) * s;
        if (qx >= dom_lx || qx < ax0 || qx >= ax1) continue;
        const bool is_wall = (qx < wall_t) || (qx > dom_lx - wall_t) || (qy < wall_t);
        const bool in_col = !is_wall && qx >= wall_t && qx < wall_t + col_w &&
                            qy >= wall_t && qy < wall_t + col_h;
        if (is_wall) mask |= ADV_ANY;
        else if (in_col) mask |= ADV_ANY | ADV_FLUID;
      }
    }
    return mask;
  }
  // A patch takes part if it holds particles and there is fluid in reach of
  // them -- its own or a neighbour's. Computed over the 5x5 block at the
  // start so every patch's opening view of a neighbour is that neighbour's
  // own; after that each patch advertises the answer itself.
  static bool activeAtStart(int px, int py) {
    if (!(siteMask(px, py) & ADV_ANY)) return false;
    if (siteMask(px, py) & ADV_FLUID) return true;
    for (int d = 0; d < NUM_DIRS; d++)
      if (siteMask(px + DIR_DX[d], py + DIR_DY[d]) & ADV_FLUID) return true;
    return false;
  }
  // Seeded from the lattice so the first step already knows where the fluid
  // is; from then on it is all carried on the migration messages.
  void seedActivity() {
    for (int d = 0; d < NUM_DIRS; d++) {
      const int nx = x + DIR_DX[d], ny = y + DIR_DY[d];
      nbr_adv[d][0] = nbr_adv[d][1] = !valid_dir[d] ? 0
          : ((activeAtStart(nx, ny) ? ADV_ACTIVE : 0) |
             (siteMask(nx, ny) & ADV_FLUID));
    }
    my_adv = my_adv_next =
        (activeAtStart(x, y) ? ADV_ACTIVE : 0) | (n_fluid > 0 ? ADV_FLUID : 0);
  }
  // The advertisements that apply to step t.
  int nbrAdv(int d, int t) const { return nbr_adv[d][t & 1]; }
  bool anyNbrAdvFluid(int t) const {
    for (int d = 0; d < NUM_DIRS; d++)
      if (valid_dir[d] && (nbrAdv(d, t) & ADV_FLUID)) return true;
    return false;
  }
  // Latched once per step. n_fluid moves within a step -- leavers out, arrivals
  // in -- and the phases after the pack have to agree with the phase that did
  // it about whether there is anything to send.
  bool isActive() const { return step_active; }

  // Every message this patch was sent for a step must have been one it
  // expected, else the step consumed fewer than arrived and the rest sit in
  // the SDAG buffer forever, with the physics having run without them. The
  // count is the same every step for the life of the run, which is the whole
  // point: nothing about it depends on who is doing work.
  void checkArrivals(std::map<int, int>& arrived, const char* what, int n_expect) {
    auto it = arrived.find(my_iter);
    const int got = it == arrived.end() ? 0 : it->second;
    if (got != n_expect)
      CkAbort("Patch (%d,%d) step %d: %d %s message(s) arrived but %d were "
              "expected -- a partner sent a step's worth of messages that this "
              "patch did not consume\n", x, y, my_iter, got, what, n_expect);
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
      // Split barrier: join the step and keep integrating while the strategy
      // runs and other patches migrate.
      AtSyncStart();
      lb_waiting = true;
      thisProxy[thisIndex].runStep();
    } else if (lb_waiting && my_iter >= lb_start_iter + lb_wait_lag) {
      lb_waiting = false;
      // One patch reports the event timeline: whether the wait lands before
      // or after the step has finished is the whole question for the lag.
      if (x == 0 && y == 0)
        CkPrintf("  LB at step %d: AtSyncWait entered %.3f s after AtSyncStart\n",
                 lb_start_iter, CkWallTimer() - lb_t0);
      AtSyncWait();
    } else {
      thisProxy[thisIndex].runStep();
    }
  }

  void ResumeFromSync() {
    if (x == 0 && y == 0)
      CkPrintf("  LB at step %d: resumed %.3f s after %s; patch (0,0) now on PE %d\n",
               lb_start_iter, CkWallTimer() - lb_t0,
               async_lb ? "AtSyncStart" : "AtSync", CkMyPe());
    thisProxy[thisIndex].runStep();
  }

  // ---- phase 1: pressure, then halo ----------------------------------------
  void startHalo() {
    n_ghost = 0;
    ghost_alloc = 0; halo_direct.clear(); halo_open = true;
    // Both fixed for the step from here on. step_active is what this patch
    // told its neighbours one step ago, not a fresh answer: the two sides of
    // every halo have to agree about it, and the advertisement is the only
    // value they both hold.
    my_adv = my_adv_next;
    step_active = (my_adv & ADV_ACTIVE) != 0;
    n_expect = 0;
    n_expect_mig = n_valid;
    for (int d = 0; d < NUM_DIRS; d++) {
      halo_peer[d] = valid_dir[d] && step_active &&
          (nbrAdv(d, my_iter) & ADV_ACTIVE);
      if (halo_peer[d]) n_expect++;
    }
    if (!step_active) {
      // No device work this step: no fluid here or next door. The step's
      // continuations still have to fire, so hand them their messages; the
      // messages themselves still go out, empty, from sendHalo.
      thisProxy[thisIndex].haloPacked();
      return;
    }
    // The kernel in front of each counter kernel clears d_counts for it, on the same
    // stream (SPH_NO_FOLD_ZERO=1 keeps the separate cudaMemsetAsync).
    static const bool fold_zero = (getenv("SPH_NO_FOLD_ZERO") == nullptr);
    hapiSubmitBatchBegin();   // one handoff for the phase (see computeAndIntegrate)
    invokeEOS(d_parts[cur], np, rho0, sound_c0, fold_zero ? d_counts : NULL, compute_stream);
    invokePackHalo(d_parts[cur], np, x0, y0, x1, y1, support, d_halo_ptrs,
        d_counts, exch_capacity, fold_zero, compute_stream);
    sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
        cudaMemcpyDeviceToHost, compute_stream);
    halo_pending = true;
    hapiAddCallback(compute_stream,
        CkCallback(CkIndex_Patch::haloPacked(), thisProxy[thisIndex]));
    hapiSubmitBatchEnd();
  }

  void sendHalo() {
    if (!step_active) return;         // halo_peer is empty; nothing to send
    halo_pending = false;
    if (h_counts[ERR_COUNTER] > 0)
      CkAbort("Patch (%d,%d): halo exchange overflowed the %d-particle buffer "
              "at step %d; increase -e\n", x, y, exch_capacity, my_iter);
    // The count goes as it is, zero included. An empty direction then costs a
    // plain message: the runtime completes a zero-length device buffer with
    // no event, no IPC slot and no stream wait on the receiver, and fires the
    // callback at once (CkRdmaDeviceOnSender). Empty is the common case: a
    // patch with no fluid has nothing to send at all, and fluid released from
    // rest needs a few thousand steps at a CFL-limited dt to cross its first
    // spacing, so early on nearly every migration message below is empty too.
    // Once the flow is running -- later, or from step 0 under -V -- they carry
    // real counts.
    for (int d = 0; d < NUM_DIRS; d++) {
      if (!halo_peer[d]) continue;
      const int cnt = h_counts[d];
      thisProxy(nbr_x[d], nbr_y[d]).receiveHalo(my_iter, flipDir(d), cnt, cnt,
          (outstanding_sends++,
           CkDeviceBuffer(d_send_halo + (size_t)d * exch_capacity,
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  void receiveHalo(int ref, int dir, int n, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    halo_arrived[ref]++;
    parts = d_recv_halo + (size_t)dir * exch_capacity;
    // Direct landing only when it cannot be wrong: the halo is for THIS iteration
    // (a later one can arrive early -- that is what the per-direction slots are
    // for), the patch is collecting ghosts (np is final, the tail is free), it
    // fits. A mid-step move (async LB) between landing and appendGhosts is
    // covered: pup ships np + ghost_alloc particles and the halo_direct map.
    static const bool direct_ok = (getenv("SPH_NO_DIRECT_HALO") == nullptr);
    if (direct_ok && halo_open && ref == my_iter && n > 0 &&
        halo_direct.find(dir) == halo_direct.end() &&
        np + ghost_alloc + n <= part_capacity) {
      parts = d_parts[cur] + np + ghost_alloc;
      halo_direct[dir] = ghost_alloc;
      ghost_alloc += n;
      // Nothing in flight touches this piece of the tail: everything issued so
      // far this step works on [0, np), the previous step's ghosts and migrants
      // sit below np or in the other buffer, and the region is handed out once.
      // So the runtime need not wait for comm_stream's earlier work before it
      // lands the copy (CkDeviceBufferPost::buffer_free) -- which also skips the
      // stream-idle check, and under +gpusubmit the drain in front of it.
      // SPH_HALO_ORDERED=1 keeps the ordered receive for comparison.
      static const bool ordered = (getenv("SPH_HALO_ORDERED") != nullptr);
      devicePost[0].buffer_free = !ordered;
    }
    devicePost[0].hapi_stream = comm_stream;
  }

  void appendGhosts(int dir, int n) {
    if (n == 0) return;
    if (np + n_ghost + n > part_capacity)
      CkAbort("Patch (%d,%d): %d particles+ghosts exceed capacity %d at step "
              "%d; increase headroom (-r)\n", x, y, np + n_ghost + n,
              part_capacity, my_iter);
    auto landed = halo_direct.find(dir);
    if (landed != halo_direct.end()) {
      halo_direct.erase(landed);          // already in place: nothing to issue
    } else {
      if (np + ghost_alloc + n > part_capacity)
        CkAbort("Patch (%d,%d): %d particles+ghosts exceed capacity %d at step %d; increase headroom (-r)\n",
                x, y, np + ghost_alloc + n, part_capacity, my_iter);
      sphSubmitMemcpy(d_parts[cur] + np + ghost_alloc,
          d_recv_halo + (size_t)dir * exch_capacity, sizeof(Particle) * n,
          cudaMemcpyDeviceToDevice, comm_stream);
      ghost_alloc += n;
    }
    n_ghost += n;
  }

  // ---- phase 2: neighbours, forces, integrate, migrate ---------------------
  void computeAndIntegrate() {
    halo_open = false;   // the ghost region is final; nothing may land in it now
    if (!isActive()) {
      // No physics, but particles CAN have arrived here -- migration reaches
      // every neighbour now, working or not -- so on a reporting step the
      // checksums have to be taken again rather than reported from the last
      // time this patch ran. One kernel every stats period, on 1 step in
      // thousands.
      if (stats_freq > 0 && (my_iter % stats_freq) == 0) {
        // Particles that arrived here landed on comm_stream, in earlier steps
        // and possibly in this one; the check reads them on compute_stream.
        hapiSubmitEventRecord((void*)halo_done, comm_stream);
        sphSubmitWaitEvent(compute_stream, halo_done);
        runCheck();
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
    // Also binds here, not only in startHalo: under async LB the element can
    // migrate MID-step, and _sdag_pup brings the continuation with it, so it
    // resumes at this phase without passing through startHalo again. The
    // migration constructor left the handles null.
    // The ghosts landed on comm_stream; the physics runs on compute_stream.
    // The whole phase is one submitter handoff (+gpusubmit): the event record
    // and wait, the cell build, forces, integrate, the leaver mark and the
    // counts copy -- ~10 driver calls -- go over as one queue entry.
    hapiSubmitBatchBegin();
    hapiSubmitEventRecord((void*)halo_done, comm_stream);
    sphSubmitWaitEvent(compute_stream, halo_done);

    // Before the physics: this is exactly what the previous step's halo
    // exchange and migration produced, and (unlike the state after integrate)
    // every local particle is required to be inside this patch's rectangle,
    // which is what makes the routing check meaningful.
    if (stats_freq > 0 && (my_iter % stats_freq) == 0) runCheck();

    const int ntot = np + n_ghost;
    invokeCellBuild(d_parts[cur], ntot, x0, y0, inv_csize, ncx, ncy, ncells,
        d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, compute_stream);
    invokeForces(d_parts[cur], np, x0, y0, inv_csize, ncx, ncy, d_cell_off,
        d_cell_cnt, d_cell_parts, smooth_h, pmass, sound_c0, gravity,
        d_drho, d_ax, d_ay, compute_stream);
    static const bool fold_zero = (getenv("SPH_NO_FOLD_ZERO") == nullptr);
    invokeIntegrate(d_parts[cur], np, sim_dt, rho0, d_drho, d_ax, d_ay,
        fold_zero ? d_counts : NULL, compute_stream);

    if (stats_freq > 0 && (my_iter % stats_freq) == 0) {
      invokeStats(d_parts[cur], np, d_stats, compute_stream);
      sphSubmitMemcpy(h_stats, d_stats, sizeof(RealType) * 8,
          cudaMemcpyDeviceToHost, compute_stream);
    }

    invokeMarkLeavers(d_parts[cur], np, x0, y0, x1, y1, d_mig_ptrs,
        d_parts[1 - cur], d_counts, exch_capacity, fold_zero, compute_stream);
    sphSubmitMemcpy(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
        cudaMemcpyDeviceToHost, compute_stream);
    mig_pending = true;
    hapiAddCallback(compute_stream,
        CkCallback(CkIndex_Patch::leaversPacked(), thisProxy[thisIndex]));
    hapiSubmitBatchEnd();
  }

  void sendLeavers() {
    const bool act = isActive();
    if (act) {
      mig_pending = false;
      if (h_counts[ERR_COUNTER] > 0)
        CkAbort("Patch (%d,%d): migration overflowed the %d-particle buffer at "
                "step %d; increase -e\n", x, y, exch_capacity, my_iter);
      cur = 1 - cur;
      np = h_counts[STAY];
    }
    // Dropped either way: this step's ghosts are this step's only. So is the
    // tail handed to direct-landed halos: a pup after this point ships
    // np + max(n_ghost, ghost_alloc), and a stale ghost_alloc here would ship
    // last step's ghost region, which the compaction into the other buffer
    // just left behind, past a NEW np -- more particles than the buffer holds.
    n_ghost = 0; ghost_alloc = 0; halo_direct.clear();
    // What every neighbour is told, and the only thing they have to go on.
    // Worked out here because the migration message is the one message that is
    // always sent -- it is what keeps a patch with nothing in it from running
    // on past the step at which a neighbour hands it particles.
    my_adv_next = (n_fluid > 0 ? ADV_FLUID : 0) |
        ((np > 0 && (n_fluid > 0 || anyNbrAdvFluid(my_iter))) ? ADV_ACTIVE : 0);

    for (int d = 0; d < NUM_DIRS; d++) {
      int cnt = act ? h_counts[d] : 0;
      if (!valid_dir[d]) {
        // Nothing is beyond the tank wall. A particle heading that way has
        // tunnelled through the boundary, which means the run has gone
        // unstable -- drop it and keep count rather than wrapping it around.
        if (cnt > 0) escaped += cnt;
        cnt = 0;
        continue;
      }
      // Every leaver is fluid: wall particles never move.
      n_fluid -= cnt;
      thisProxy(nbr_x[d], nbr_y[d]).receiveParticles(my_iter, flipDir(d),
          my_adv_next, cnt, cnt,
          (outstanding_sends++,
           CkDeviceBuffer(d_send_mig + (size_t)d * exch_capacity,
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  void receiveParticles(int ref, int dir, int adv, int n, int& m,
      Particle*& parts, CkDeviceBufferPost* devicePost) {
    parts_arrived[ref]++;
    // Sent with step ref, so it is the answer for ref+1. A message two steps
    // ahead would land in the slot this patch has yet to read; the migration
    // exchange makes that impossible, and this says so out loud.
    if (ref > my_iter + 1)
      CkAbort("Patch (%d,%d) at step %d: migration from step %d -- a partner "
              "ran more than one step ahead, which the two advertisement slots "
              "assume cannot happen\n", x, y, my_iter, ref);
    nbr_adv[dir][(ref + 1) & 1] = adv;
    // The payload slot needs the same parity the advertisement has. This
    // handler runs when the message ARRIVES, not when the step consumes it,
    // and a partner one step ahead is expected, not exceptional (see the
    // abort above). A patch takes its migration in phase 6, AFTER the phase-5
    // send that is all a partner needs to finish its own step -- so a partner
    // can complete that step, run phases 1-5 of the next one, and post a
    // second payload into this direction while the first is still sitting
    // here unread. Stream order does not save it: the rgets and the copy in
    // appendParticles all run on comm_stream, but the HOST enqueues them, and
    // the second rget is enqueued at arrival while the first copy waits for
    // the SDAG to reach its serial block. The count travels in the message,
    // so nothing downstream notices -- checkArrivals balances and the
    // capacity check passes, and the step consumes the wrong generation of
    // particles in silence.
    //
    // The halo has no such window: it is consumed in phase 3, before the
    // phase-5 send that lets a partner advance, so d_recv_halo is safe with
    // one slot per direction. That is an invariant of the phase order in
    // sph2d.ci, not an accident -- moving the migration receive earlier, or
    // the halo receive later, would change which of these needs doubling.
    parts = d_recv_mig + (size_t)((ref & 1) * NUM_DIRS + dir) * exch_capacity;
    devicePost[0].hapi_stream = comm_stream;
  }

  // ref, not my_iter: the slot is the one the sender's step parity chose in
  // receiveParticles, and a message from step my_iter+1 is legitimate.
  void appendParticles(int ref, int dir, int n) {
    if (n == 0) return;
    if (np + n > part_capacity)
      CkAbort("Patch (%d,%d): %d particles exceed capacity %d at step %d; "
              "increase headroom (-r)\n", x, y, np + n, part_capacity, my_iter);
    sphSubmitMemcpy(d_parts[cur] + np,
        d_recv_mig + (size_t)((ref & 1) * NUM_DIRS + dir) * exch_capacity,
        sizeof(Particle) * n,
        cudaMemcpyDeviceToDevice, comm_stream);
    np += n;
    n_fluid += n;   // only fluid migrates
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
    checkArrivals(parts_arrived, "migration", n_expect_mig);
    const bool reporting = (stats_freq > 0 && (my_iter % stats_freq) == 0);
    // The count reduced is the one the check kernel actually saw, not the
    // current np: this step's leavers have already been subtracted by
    // sendLeavers and its arrivals added by appendParticles, so np here is the
    // NEXT step's state. Mixing the two would make the conservation check
    // compare a checksum and a count taken at different instants.
    long sum_np = 0, max_np = 0, esc = escaped, n_active = isActive() ? 1 : 0;
    double sum_rho = 0, sum_ke = 0, n_fluid_d = 0, vmax = 0, front = 0;
    unsigned long long id_sum = 0, bnd_sum = 0, n_fl = 0, n_bd = 0;
    if (reporting) {
      // No sync. Both copies were enqueued on compute_stream ahead of the
      // migration pack, and leaversPacked is a hapiAddCallback on that same
      // stream -- it cannot fire until they have landed. endOfStep runs
      // strictly after leaversPacked, so h_stats and h_check are already valid.
      // An inactive patch reports what these last held: nothing in it has
      // changed since, so the checksums still describe its particles.
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
        CkPrintf("[WARN] patch (%d,%d) lost %ld particle(s) out of the tank "
                 "-- the run went past the edge of its validity\n", x, y,
                 escaped);
      contribute(CkCallback(CkReductionTarget(Main, allDone), main_proxy));
    }
  }
};

#include "sph2d.def.h"
