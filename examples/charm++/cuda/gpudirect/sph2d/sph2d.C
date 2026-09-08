#include "hapi.h"
#include "sph2d.decl.h"
#include "sph2d.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <unistd.h>

// Device allocation following the runtime's choice: under +gpupool every buffer
// comes from CkDeviceMalloc (an arena the peers have already opened, no driver
// call, no device sync); without it, from hapiMalloc.
inline hapiError_t pcMalloc(void** p, size_t n) {
  static const bool pool = CkDevicePoolOn();
  if (!pool) return hapiMalloc(p, n);
  *p = CkDeviceMalloc(n);
  return (*p != NULL) ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline hapiError_t pcFree(void* p) {
  static const bool pool = CkDevicePoolOn();
  if (p == NULL) return cudaSuccess;
  if (pool) { CkDeviceFree(p); return cudaSuccess; }
  return hapiFree(p);
}

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
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ int first_lb;
/* readonly */ int lb_freq;
/* readonly */ int async_lb;
/* readonly */ int lb_wait_lag;
/* readonly */ int part_capacity;
/* readonly */ int exch_capacity;
/* readonly */ int stats_freq;

extern void invokeCellBuild(const Particle*, int, RealType, RealType, RealType,
    int, int, int, int*, int*, int*, int*, cudaStream_t);
extern void invokeEOS(Particle*, int, RealType, RealType, cudaStream_t);
extern void invokeForces(const Particle*, int, RealType, RealType, RealType,
    int, int, const int*, const int*, const int*, RealType, RealType, RealType,
    RealType, RealType*, RealType*, RealType*, cudaStream_t);
extern void invokeIntegrate(Particle*, int, RealType, RealType, RealType*,
    RealType*, RealType*, cudaStream_t);
extern void invokePackHalo(const Particle*, int, RealType, RealType, RealType,
    RealType, RealType, Particle**, int*, int, cudaStream_t);
extern void invokeMarkLeavers(const Particle*, int, RealType, RealType, RealType,
    RealType, Particle**, Particle*, int*, int, cudaStream_t);
extern void invokeStats(const Particle*, int, RealType*, cudaStream_t);

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
public:
  Main(CkArgMsg* m) : stat_count(0) {
    // Tank and fluid geometry are in metres; the default is the Koshizuka &
    // Oka dam break shape (column twice as tall as it is wide) scaled up so a
    // reasonable particle count fits.
    dom_lx = 4.0f; dom_ly = 3.0f;
    n_chares_x = 8; n_chares_y = 4;
    spacing = 0.02f;
    col_w = 1.0f; col_h = 2.0f;
    rho0 = 1000.0f;
    gravity = 9.81f;
    // The acoustic CFL makes the timestep small: at the default spacing dt is
    // ~1e-4 s, the column takes ~0.2 s to collapse and the surge needs ~1.4 s
    // (about 13000 steps) to reach the far wall. 4000 steps is enough for the
    // fluid to leave its starting patches and the load to have visibly moved,
    // which is the shortest run that exercises a balancer at all.
    n_iters = 4000; warmup_iters = 10;
    first_lb = 99999; lb_freq = 0; async_lb = 0; lb_wait_lag = 3;
    stats_freq = 200;
    RealType headroom = 4.0f;
    RealType exch_frac = 0.25f;

    int c;
    while ((c = getopt(m->argc, m->argv, "X:Y:x:y:s:w:t:i:u:f:b:l:ar:e:S:")) != -1) {
      switch (c) {
        case 'X': dom_lx = atof(optarg); break;
        case 'Y': dom_ly = atof(optarg); break;
        case 'x': n_chares_x = atoi(optarg); break;
        case 'y': n_chares_y = atoi(optarg); break;
        case 's': spacing = atof(optarg); break;
        case 'w': col_w = atof(optarg); break;
        case 't': col_h = atof(optarg); break;
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
    // fastest expected flow, which keeps the density variation near 1%.
    const RealType v_max = std::sqrt(2.0f * gravity * col_h);
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
    if (async_lb && (lb_wait_lag < 1 || (lb_freq > 0 && lb_wait_lag >= lb_freq))) {
      CkAbort("-l (%d) must be in [1, lb_freq): the wait must land before the "
              "next AtSyncStart\n", lb_wait_lag);
    }

    // Capacity. A patch's share of the fluid at rest is what it starts with;
    // the surge concentrates far more than that into the patches it runs
    // through, which is the whole point, so the headroom is generous.
    const RealType patch_area = (dom_lx / n_chares_x) * (dom_ly / n_chares_y);
    const int per_patch = (int)(patch_area / (spacing * spacing));
    part_capacity = std::max(4096, (int)(per_patch * headroom));
    exch_capacity = std::max(1024, (int)(part_capacity * exch_frac));

    const long lattice_x = (long)(dom_lx / spacing);
    const long lattice_y = (long)(dom_ly / spacing);

    main_proxy = thisProxy;
    CkPrintf("SPH dam break: tank %.2f x %.2f m, spacing %.4f m, h %.4f m\n",
             dom_lx, dom_ly, spacing, smooth_h);
    CkPrintf("  column %.2f x %.2f m, rho0 %.0f, c0 %.2f m/s, dt %.3e s\n",
             col_w, col_h, rho0, sound_c0, sim_dt);
    CkPrintf("  lattice %ld x %ld, %d x %d patches, capacity/patch %d, "
             "exchange %d\n", lattice_x, lattice_y, n_chares_x, n_chares_y,
             part_capacity, exch_capacity);
    CkPrintf("  iterations %d (+%d warmup), first LB %d, LB period %d%s\n",
             n_iters, warmup_iters, first_lb, lb_freq,
             async_lb ? ", async" : "");

    patch_proxy = CProxy_Patch::ckNew(n_chares_x, n_chares_y);
    patch_proxy.init();
  }

  void initDone(long total_np) {
    CkPrintf("Init: %ld particles\n", total_np);
    start_time = CkWallTimer();
    patch_proxy.iterate();
  }

  // sum np | max np | sum rho | sum KE | count fluid | max speed | front x
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
    delete[] t;
    delete msg;

    if (n_fluid > 0) {
      // max/avg particles per patch is the imbalance a balancer can act on.
      const double imb = (double)max_np * (n_chares_x * n_chares_y) / (double)sum_np;
      CkPrintf("  step %6d: rho %.2f  KE %.4e  |v|max %.3f  front x %.3f  "
               "particle imbalance (max/avg) %.2f\n",
               (stat_count + 1) * stats_freq, sum_rho / n_fluid, sum_ke * pmass,
               vmax, front, imb);
    }
    stat_count++;
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
  bool lb_waiting;
  int lb_start_iter;
  long escaped;                   // particles that left the tank (instability)

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
  int* h_counts;
  RealType* h_stats;
  std::vector<Particle> h_parts;  // staging for migration only

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

  // Acquired on first use, not in a constructor. A constructor does not
  // necessarily run on the PE worker thread that will execute this element --
  // measured: an element ended up holding a stream whose owning device was 3
  // while running on a PE whose device was 1 -- and a stream belongs to the
  // device that was current when it was created. Acquiring here, from the
  // entry method that launches the kernels, binds it to the right GPU by
  // construction. Cheap: one null check per step.
  void bindStreams() {
    if (compute_stream == NULL) {
      createCudaEntities();
    }
  }
  Patch(CkMigrateMessage* m) : CBase_Patch(m) {
    usesAtSync = true;
    setupState();      // leaves the streams unbound; bindStreams() takes them
    // Streams and the ordering event belong to the object's lifetime, created
    // here and destroyed in ~Patch. pup must NOT destroy them: Charm++ runs
    // the source object's destructor after packing it, and a destructor that
    // synchronizes an already-destroyed stream leaves cudaErrorInvalidResource
    // Handle as this thread's sticky error -- which then aborts whichever
    // unrelated chare next calls cudaPeekAtLastError.
  }

  ~Patch() {
    if (compute_stream != NULL) cudaStreamSynchronize(compute_stream);
    if (comm_stream != NULL) cudaStreamSynchronize(comm_stream);
    freeDevice();
    // Hand the streams back so they are recycled rather than leaked or
    // destroyed; the runtime returns each to its own device's free list, which
    // is what makes a chare that migrates between GPUs safe.
    hapiReleaseStream(compute_stream);
    hapiReleaseStream(comm_stream);
    if (halo_done != NULL) cudaEventDestroy(halo_done);
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

    // Cells are one kernel support wide, so neighbours are in the 3x3 block
    // around a particle's own cell.
    ncx = std::max(1, (int)std::ceil((x1 - x0) / support));
    ncy = std::max(1, (int)std::ceil((y1 - y0) / support));
    ncells = (ncx + 2) * (ncy + 2);
    inv_csize = 1.0f / support;
  }

  void setupState() {
    compute_stream = NULL; comm_stream = NULL; halo_done = NULL;
    np = n_ghost = 0; cur = 0; my_iter = 0;
    outstanding_sends = 0; draining = false;
    lb_waiting = false; lb_start_iter = 0; escaped = 0;
    for (int i = 0; i < 2; i++) d_parts[i] = NULL;
    d_send_halo = d_recv_halo = d_send_mig = d_recv_mig = NULL;
    d_halo_ptrs = d_mig_ptrs = NULL;
    d_counts = d_cell_cnt = d_cell_off = d_cursor = d_cell_parts = NULL;
    d_drho = d_ax = d_ay = d_stats = NULL;
    h_counts = NULL; h_stats = NULL;
  }

  void createCudaEntities() {
    // Streams come from the runtime's per-device pool: it hands back one that
    // belongs to this PE's device, and takes it back on migration. The event is
    // ours, created here on the same device.
    compute_stream = hapiAcquireStream();
    comm_stream = hapiAcquireStream();
    hapiCheck(cudaEventCreateWithFlags(&halo_done, cudaEventDisableTiming));
  }

  void allocDevice() {
    const size_t pcap = sizeof(Particle) * (size_t)part_capacity;
    const size_t ecap = sizeof(Particle) * (size_t)NUM_DIRS * exch_capacity;
    hapiCheck(pcMalloc((void**)&d_parts[0], pcap));
    hapiCheck(pcMalloc((void**)&d_parts[1], pcap));
    hapiCheck(pcMalloc((void**)&d_send_halo, ecap));
    hapiCheck(pcMalloc((void**)&d_recv_halo, ecap));
    hapiCheck(pcMalloc((void**)&d_send_mig, ecap));
    hapiCheck(pcMalloc((void**)&d_recv_mig, ecap));
    hapiCheck(pcMalloc((void**)&d_halo_ptrs, sizeof(Particle*) * NUM_DIRS));
    hapiCheck(pcMalloc((void**)&d_mig_ptrs, sizeof(Particle*) * NUM_DIRS));
    hapiCheck(pcMalloc((void**)&d_counts, sizeof(int) * NUM_COUNTERS));
    hapiCheck(pcMalloc((void**)&d_cell_cnt, sizeof(int) * ncells));
    hapiCheck(pcMalloc((void**)&d_cell_off, sizeof(int) * ncells));
    hapiCheck(pcMalloc((void**)&d_cursor, sizeof(int) * ncells));
    hapiCheck(pcMalloc((void**)&d_cell_parts, sizeof(int) * part_capacity));
    hapiCheck(pcMalloc((void**)&d_drho, sizeof(RealType) * part_capacity));
    hapiCheck(pcMalloc((void**)&d_ax, sizeof(RealType) * part_capacity));
    hapiCheck(pcMalloc((void**)&d_ay, sizeof(RealType) * part_capacity));
    hapiCheck(pcMalloc((void**)&d_stats, sizeof(RealType) * 8));
    hapiCheck(hapiMallocHost((void**)&h_counts, sizeof(int) * NUM_COUNTERS));
    hapiCheck(hapiMallocHost((void**)&h_stats, sizeof(RealType) * 8));

    // The per-direction base pointers the pack kernels scatter into.
    Particle* hp[NUM_DIRS];
    for (int d = 0; d < NUM_DIRS; d++) hp[d] = d_send_halo + (size_t)d * exch_capacity;
    hapiCheck(cudaMemcpy(d_halo_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice));
    for (int d = 0; d < NUM_DIRS; d++) hp[d] = d_send_mig + (size_t)d * exch_capacity;
    hapiCheck(cudaMemcpy(d_mig_ptrs, hp, sizeof(hp), cudaMemcpyHostToDevice));
  }

  void freeDevice() {
    pcFree(d_parts[0]); pcFree(d_parts[1]);
    pcFree(d_send_halo); pcFree(d_recv_halo);
    pcFree(d_send_mig); pcFree(d_recv_mig);
    pcFree(d_halo_ptrs); pcFree(d_mig_ptrs);
    pcFree(d_counts); pcFree(d_cell_cnt); pcFree(d_cell_off);
    pcFree(d_cursor); pcFree(d_cell_parts);
    pcFree(d_drho); pcFree(d_ax); pcFree(d_ay); pcFree(d_stats);
    if (h_counts) hapiFreeHost(h_counts);
    if (h_stats) hapiFreeHost(h_stats);
    for (int i = 0; i < 2; i++) d_parts[i] = NULL;
    d_send_halo = d_recv_halo = d_send_mig = d_recv_mig = NULL;
    d_halo_ptrs = d_mig_ptrs = NULL;
    d_counts = d_cell_cnt = d_cell_off = d_cursor = d_cell_parts = NULL;
    d_drho = d_ax = d_ay = d_stats = NULL;
    h_counts = NULL; h_stats = NULL;
  }

  // Migration: the only state worth moving is the particles. The scratch
  // arrays and the cell list are rebuilt from scratch every step anyway.
  void pup(PUP::er& p) {
    CBase_Patch::pup(p);
    p | my_iter; p | cur; p | lb_waiting; p | lb_start_iter; p | escaped;
    // Which whens are outstanding and any buffered ordinary messages: under
    // async LB an element can be moved mid-step and its continuations must
    // follow it.
    _sdag_pup(p);
    p | recv_count;
    p | outstanding_sends;

    if (p.isPacking()) {
      // The only moment the device state has to be settled.
      cudaStreamSynchronize(compute_stream);
      cudaStreamSynchronize(comm_stream);
      h_parts.resize(np);
      if (np > 0)
        hapiCheck(cudaMemcpy(h_parts.data(), d_parts[cur],
            sizeof(Particle) * np, cudaMemcpyDeviceToHost));
      freeDevice();
    }

    int n = (int)(p.isPacking() ? h_parts.size() : 0);
    p | n;
    if (p.isUnpacking()) h_parts.resize(n);
    if (n > 0) PUParray(p, h_parts.data(), n);

    if (p.isUnpacking()) {
      setupGeometry();          // thisIndex is valid here, unlike in the ctor
      np = n; n_ghost = 0; cur = 0;
      draining = false;
      allocDevice();
      if (np > 0)
        hapiCheck(cudaMemcpy(d_parts[0], h_parts.data(),
            sizeof(Particle) * np, cudaMemcpyHostToDevice));
      h_parts.clear();
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
        q.x = px; q.y = py; q.vx = 0.0f; q.vy = 0.0f;
        q.rho = rho0; q.p = 0.0f;
        q.type = is_wall ? PTYPE_BOUND : PTYPE_FLUID;
        q.pad = 0;
        mine.push_back(q);
      }
    }

    np = (int)mine.size();
    if (np > part_capacity)
      CkAbort("Patch (%d,%d): %d initial particles exceed capacity %d; "
              "increase headroom (-r)\n", x, y, np, part_capacity);
    if (np > 0)
      hapiCheck(cudaMemcpy(d_parts[cur], mine.data(), sizeof(Particle) * np,
          cudaMemcpyHostToDevice));

    long n = np;
    contribute(sizeof(long), &n, CkReduction::sum_long,
        CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  bool isLBIter(int it) const {
    return it == first_lb || (it != 0 && lb_freq > 0 && it % lb_freq == 0);
  }

  void iterate() {
    if (isLBIter(my_iter) && !lb_waiting) {
      if (!async_lb) {
        AtSync();
        return;
      }
      // Split barrier: join the step and keep integrating while the strategy
      // runs and other patches migrate.
      AtSyncStart();
      lb_start_iter = my_iter;
      lb_waiting = true;
      thisProxy[thisIndex].runStep();
    } else if (lb_waiting && my_iter >= lb_start_iter + lb_wait_lag) {
      lb_waiting = false;
      AtSyncWait();
    } else {
      thisProxy[thisIndex].runStep();
    }
  }

  void ResumeFromSync() {
    thisProxy[thisIndex].runStep();
  }

  // ---- phase 1: pressure, then halo ----------------------------------------
  void startHalo() {
    bindStreams();
    n_ghost = 0;
    invokeEOS(d_parts[cur], np, rho0, sound_c0, compute_stream);
    invokePackHalo(d_parts[cur], np, x0, y0, x1, y1, support, d_halo_ptrs,
        d_counts, exch_capacity, compute_stream);
    hapiCheck(cudaMemcpyAsync(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
        cudaMemcpyDeviceToHost, compute_stream));
    hapiAddCallback(compute_stream,
        CkCallback(CkIndex_Patch::haloPacked(), thisProxy[thisIndex]));
  }

  void sendHalo() {
    if (h_counts[ERR_COUNTER] > 0)
      CkAbort("Patch (%d,%d): halo exchange overflowed the %d-particle buffer "
              "at step %d; increase -e\n", x, y, exch_capacity, my_iter);
    for (int d = 0; d < NUM_DIRS; d++) {
      const int cnt = valid_dir[d] ? h_counts[d] : 0;
      thisProxy(nbr_x[d], nbr_y[d]).receiveHalo(my_iter, flipDir(d), cnt,
          std::max(cnt, 1),
          (outstanding_sends++,
           CkDeviceBuffer(d_send_halo + (size_t)d * exch_capacity,
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  void receiveHalo(int ref, int dir, int n, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    parts = d_recv_halo + (size_t)dir * exch_capacity;
    devicePost[0].hapi_stream = comm_stream;
  }

  void appendGhosts(int dir, int n) {
    if (n == 0) return;
    if (np + n_ghost + n > part_capacity)
      CkAbort("Patch (%d,%d): %d particles+ghosts exceed capacity %d at step "
              "%d; increase headroom (-r)\n", x, y, np + n_ghost + n,
              part_capacity, my_iter);
    hapiCheck(cudaMemcpyAsync(d_parts[cur] + np + n_ghost,
        d_recv_halo + (size_t)dir * exch_capacity, sizeof(Particle) * n,
        cudaMemcpyDeviceToDevice, comm_stream));
    n_ghost += n;
  }

  // ---- phase 2: neighbours, forces, integrate, migrate ---------------------
  void computeAndIntegrate() {
    // The ghosts landed on comm_stream; the physics runs on compute_stream.
    hapiCheck(cudaEventRecord(halo_done, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, halo_done, 0));

    const int ntot = np + n_ghost;
    invokeCellBuild(d_parts[cur], ntot, x0, y0, inv_csize, ncx, ncy, ncells,
        d_cell_cnt, d_cell_off, d_cursor, d_cell_parts, compute_stream);
    invokeForces(d_parts[cur], np, x0, y0, inv_csize, ncx, ncy, d_cell_off,
        d_cell_cnt, d_cell_parts, smooth_h, pmass, sound_c0, gravity,
        d_drho, d_ax, d_ay, compute_stream);
    invokeIntegrate(d_parts[cur], np, sim_dt, rho0, d_drho, d_ax, d_ay,
        compute_stream);

    if (stats_freq > 0 && (my_iter % stats_freq) == 0) {
      invokeStats(d_parts[cur], np, d_stats, compute_stream);
      hapiCheck(cudaMemcpyAsync(h_stats, d_stats, sizeof(RealType) * 8,
          cudaMemcpyDeviceToHost, compute_stream));
    }

    invokeMarkLeavers(d_parts[cur], np, x0, y0, x1, y1, d_mig_ptrs,
        d_parts[1 - cur], d_counts, exch_capacity, compute_stream);
    hapiCheck(cudaMemcpyAsync(h_counts, d_counts, sizeof(int) * NUM_COUNTERS,
        cudaMemcpyDeviceToHost, compute_stream));
    hapiAddCallback(compute_stream,
        CkCallback(CkIndex_Patch::leaversPacked(), thisProxy[thisIndex]));
  }

  void sendLeavers() {
    if (h_counts[ERR_COUNTER] > 0)
      CkAbort("Patch (%d,%d): migration overflowed the %d-particle buffer at "
              "step %d; increase -e\n", x, y, exch_capacity, my_iter);

    const int stay = h_counts[STAY];
    cur = 1 - cur;
    np = stay;
    n_ghost = 0;

    for (int d = 0; d < NUM_DIRS; d++) {
      int cnt = h_counts[d];
      if (!valid_dir[d]) {
        // Nothing is beyond the tank wall. A particle heading that way has
        // tunnelled through the boundary, which means the run has gone
        // unstable -- drop it and keep count rather than wrapping it around.
        if (cnt > 0) escaped += cnt;
        cnt = 0;
      }
      thisProxy(nbr_x[d], nbr_y[d]).receiveParticles(my_iter, flipDir(d), cnt,
          std::max(cnt, 1),
          (outstanding_sends++,
           CkDeviceBuffer(d_send_mig + (size_t)d * exch_capacity,
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  void receiveParticles(int ref, int dir, int n, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    parts = d_recv_mig + (size_t)dir * exch_capacity;
    devicePost[0].hapi_stream = comm_stream;
  }

  void appendParticles(int dir, int n) {
    if (n == 0) return;
    if (np + n > part_capacity)
      CkAbort("Patch (%d,%d): %d particles exceed capacity %d at step %d; "
              "increase headroom (-r)\n", x, y, np + n, part_capacity, my_iter);
    hapiCheck(cudaMemcpyAsync(d_parts[cur] + np,
        d_recv_mig + (size_t)dir * exch_capacity, sizeof(Particle) * n,
        cudaMemcpyDeviceToDevice, comm_stream));
    np += n;
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
    long sum_np = np, max_np = np;
    double sum_rho = 0, sum_ke = 0, n_fluid = 0, vmax = 0, front = 0;
    if (stats_freq > 0 && (my_iter % stats_freq) == 0) {
      // No sync. The stats copy was enqueued on compute_stream ahead of the
      // migration pack, and leaversPacked is a hapiAddCallback on that same
      // stream -- it cannot fire until the copy has landed. endOfStep runs
      // strictly after leaversPacked, so h_stats is already valid.
      sum_rho = h_stats[0];
      sum_ke = h_stats[1];
      n_fluid = h_stats[2];
      vmax = h_stats[3];
      front = h_stats[4];
    }

    CkReduction::tupleElement tuple[] = {
        CkReduction::tupleElement(sizeof(long), &sum_np, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(long), &max_np, CkReduction::max_long),
        CkReduction::tupleElement(sizeof(double), &sum_rho, CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double), &sum_ke, CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double), &n_fluid, CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(double), &vmax, CkReduction::max_double),
        CkReduction::tupleElement(sizeof(double), &front, CkReduction::max_double)};
    if (stats_freq > 0 && (my_iter % stats_freq) == 0) {
      CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple, 7);
      msg->setCallback(CkCallback(CkIndex_Main::stepStats(NULL), main_proxy));
      contribute(msg);
    }

    if (my_iter < warmup_iters + n_iters) {
      thisProxy[thisIndex].iterate();
    } else {
      if (escaped > 0)
        CkPrintf("[WARN] patch (%d,%d) lost %ld particle(s) through the tank "
                 "wall -- the run went unstable\n", x, y, escaped);
      contribute(CkCallback(CkReductionTarget(Main, allDone), main_proxy));
    }
  }
};

#include "sph2d.def.h"
