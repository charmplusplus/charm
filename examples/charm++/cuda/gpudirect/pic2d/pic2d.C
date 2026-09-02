#include "hapi.h"
#include "pic2d.decl.h"
#include "pic2d.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Patch patch_proxy;
/* readonly */ int grid_width;
/* readonly */ int grid_height;
/* readonly */ int block_width;
/* readonly */ int block_height;
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ int jacobi_iters;
// Phi exchanges per timestep: one per block of PHI_HALO sweeps. The solve used
// one exchange per sweep, which on a latency-bound run is the whole cost.
/* readonly */ int jacobi_rounds;
/* readonly */ int ppc;
/* readonly */ int first_lb;
/* readonly */ int async_lb;
/* readonly */ int lb_wait_lag;
/* readonly */ int lb_freq;

// How many steps before an AtSync step to start gathering load measurements.
// Instrumentation is expensive enough to be worth confining to a window, and a
// few steps are enough to characterise a chare's load.
#define LB_INSTRUMENT_WINDOW 3
/* readonly */ int dist_type;
/* readonly */ long n_total_particles;
/* readonly */ double bunch_frac;
/* readonly */ double bunch_sigma;
/* readonly */ double drift_vx;
/* readonly */ double vth;
/* readonly */ double sim_dt;
/* readonly */ double coupling;
/* readonly */ int part_capacity;
/* readonly */ int exch_capacity;
/* readonly */ int stats_freq;
/* readonly */ bool print_parts;

extern void invokeInitParticlesKernel(Particle* d_parts, int* d_np,
    long n_total, long n_bunch, int dist_type, float bunch_cx, float bunch_cy,
    float sigma, float drift, float thermal, float Lx, float Ly, float px0,
    float py0, float px1, float py1, int capacity, int* d_err,
    cudaStream_t stream);
extern void invokeDepositKernel(const Particle* d_parts, int n,
    RealType* d_rho, float weight, float px0, float py0, int block_width,
    int block_height, cudaStream_t stream);
extern void invokePackRhoHaloKernel(const RealType* d_rho, RealType* d_slab,
    int block_width, int block_height, cudaStream_t stream);
extern void invokeUnpackRhoHaloKernel(RealType* d_rho, const RealType* d_buf,
    int dir, int block_width, int block_height, cudaStream_t stream);
extern void invokePackChargeGhostsKernel(const RealType* d_rho,
    RealType* d_slab, int block_width, int block_height, cudaStream_t stream);
extern void invokeAccumChargeGhostKernel(RealType* d_rho,
    const RealType* d_buf, int dir, int block_width, int block_height,
    cudaStream_t stream);
extern void invokePackPhiKernel(const RealType* d_phi, RealType* d_slab,
    int block_width, int block_height, cudaStream_t stream);
extern void invokeUnpackPhiKernel(RealType* d_phi, const RealType* d_buf,
    int dir, int block_width, int block_height, cudaStream_t stream);
extern void invokeJacobiPhiKernel(const RealType* d_phi, RealType* d_phi_new,
    const RealType* d_rho, float rho_bar, int block_width, int block_height,
    int margin, cudaStream_t stream);
extern void invokeEFieldKernel(const RealType* d_phi, float2* d_efield,
    int block_width, int block_height, cudaStream_t stream);
extern void invokePackEGhostsKernel(const float2* d_efield, float2* d_slab,
    int block_width, int block_height, cudaStream_t stream);
extern void invokeUnpackEGhostKernel(float2* d_efield, const float2* d_buf,
    int dir, int block_width, int block_height, cudaStream_t stream);
extern void invokePushAndMarkKernel(const Particle* d_parts, int n,
    const float2* d_efield, Particle* d_stay, Particle* d_sendslab,
    int exch_cap, int* d_counts, float dt, float qm, float px0, float py0,
    float Lx, float Ly, int block_width, int block_height,
    cudaStream_t stream);

// Direction helpers; see the Dir enum in pic2d.h. Directions in messages are
// always from the receiver's perspective, so senders flip them.
static const int DIR_DX[NUM_DIRS] = {-1, 1, 0, 0, -1, 1, -1, 1};
static const int DIR_DY[NUM_DIRS] = {0, 0, -1, 1, -1, -1, 1, 1};

static inline int flipDir(int d) {
  static const int f[NUM_DIRS] = {RIGHT, LEFT, BOTTOM, TOP, BR, BL, TR, TL};
  return f[d];
}

// Strip lengths and offsets in the shared 8-way slab layout
// (see packChargeGhostsKernel in pic2d.cu)
static inline int stripLen(int d) {
  if (d == LEFT || d == RIGHT) return block_height;
  if (d == TOP || d == BOTTOM) return block_width;
  return 1;
}

static inline int stripOff(int d) {
  int W = block_width, H = block_height;
  switch (d) {
    case LEFT:   return 0;
    case RIGHT:  return H;
    case TOP:    return 2*H;
    case BOTTOM: return 2*H + W;
    default:     return 2*H + 2*W + (d - TL);
  }
}

// Phi exchanges PHI_HALO layers per side, so its strips are that much longer
// than the single-layer charge/E strips stripLen() describes.
static inline int phiStripLen(int d) {
  if (d == LEFT || d == RIGHT || d == TOP || d == BOTTOM)
    return PHI_HALO * stripLen(d);
  return PHI_CORNER;   // diagonal: a square block, zero at PHI_HALO 1
}

// Final-rho halo: D layers per side with D*D corner blocks, the same shape as
// the phi slab one depth shallower.
static inline int rhoHaloLen(int d) {
  const int D = RHO_HALO_D;
  if (d == LEFT || d == RIGHT) return D * block_height;
  if (d == TOP || d == BOTTOM) return D * block_width;
  return D * D;
}

static inline int rhoHaloOff(int d) {
  const int D = RHO_HALO_D, W = block_width, H = block_height;
  switch (d) {
    case LEFT:   return 0;
    case RIGHT:  return D*H;
    case TOP:    return 2*D*H;
    case BOTTOM: return 2*D*H + D*W;
    default:     return 2*D*(H + W) + (d - TL) * D * D;
  }
}

static inline int rhoHaloSlab() {
  const int D = RHO_HALO_D;
  return 2*D*(block_width + block_height) + 4*D*D;
}

// Total 4-way phi slab, one parity.
static inline int phiSlab4() {
  return 2 * PHI_HALO * (block_width + block_height) + 4 * PHI_CORNER;
}

// Offsets in the 4-way phi slab (see packPhiLRKernel in pic2d.cu)
static inline int phiOff(int d) {
  int W = block_width, H = block_height;
  switch (d) {
    case LEFT:   return 0;
    case RIGHT:  return PHI_HALO*H;
    case TOP:    return 2*PHI_HALO*H;
    case BOTTOM: return 2*PHI_HALO*H + PHI_HALO*W;
    // Diagonals, in TL, TR, BL, BR order to match packPhiKernel.
    default:     return 2*PHI_HALO*(H + W) + (d - TL) * PHI_CORNER;
  }
}

class Main : public CBase_Main {
  double init_start_time;
  double start_time;
  double window_start_time;
  int stats_iter;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;

    // Default configuration
    grid_width = 1024;
    grid_height = 1024;
    block_width = 128;
    block_height = 128;
    ppc = 32;
    n_iters = 100;
    warmup_iters = 10;
    jacobi_iters = 20;
    dist_type = 1;
    bunch_frac = 0.5;
    double sigma_frac = 0.1;
    drift_vx = 20.0;
    vth = 1.0;
    sim_dt = 0.0;  // derived below unless overridden with -T
    coupling = 1e-3;
    double headroom = 12.0;
    double exch_frac = 0.05;
    first_lb = 10;
    lb_freq = 9999;
    async_lb = 0;
    lb_wait_lag = 3;
    stats_freq = 1;
    print_parts = false;
    stats_iter = 0;

    int c;
    while ((c = getopt(m->argc, m->argv, "W:H:w:h:p:i:u:j:d:F:s:v:t:T:q:f:b:r:x:c:Pal:")) != -1) {
      switch (c) {
        case 'W': grid_width = atoi(optarg); break;
        case 'H': grid_height = atoi(optarg); break;
        case 'w': block_width = atoi(optarg); break;
        case 'h': block_height = atoi(optarg); break;
        case 'p': ppc = atoi(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'u': warmup_iters = atoi(optarg); break;
        case 'j': jacobi_iters = atoi(optarg); break;
        case 'd': dist_type = atoi(optarg); break;
        case 'F': bunch_frac = atof(optarg); break;
        case 's': sigma_frac = atof(optarg); break;
        case 'v': drift_vx = atof(optarg); break;
        case 't': vth = atof(optarg); break;
        case 'T': sim_dt = atof(optarg); break;
        case 'q': coupling = atof(optarg); break;
        case 'f': first_lb = atoi(optarg); break;
        case 'b': lb_freq = atoi(optarg); break;
        case 'r': headroom = atof(optarg); break;
        case 'x': exch_frac = atof(optarg); break;
        case 'c': stats_freq = atoi(optarg); break;
        case 'P': print_parts = true; break;
        case 'a': async_lb = 1; break;
        case 'l': lb_wait_lag = atoi(optarg); break;
        default:
          CkPrintf(
              "Usage: %s -W [grid width] -H [grid height] -w [patch width] -h [patch height]\n"
              "  -p [particles per cell] -i [iterations] -u [warmup iterations]\n"
              "  -j [jacobi iterations per step]\n"
              "  -d [distribution: 0 uniform, 1 gaussian bunch, 2 two-stream]\n"
              "  -F [bunch fraction] -s [bunch sigma, fraction of min(W,H)]\n"
              "  -v [drift velocity, cells/time]\n"
              "  -t [thermal velocity] -T [dt override] -q [field coupling strength]\n"
              "  -f [first LB iteration] -b [LB frequency]\n"
              "  -a (async LB: overlap the step; needs +LBAsync) -l [wait lag]\n"
              "  -r [particle capacity headroom] -x [exchange buffer fraction]\n"
              "  -c [stats frequency, 0 disables] -P (print final particle counts)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (grid_width % block_width != 0 || grid_height % block_height != 0) {
      CkAbort("Invalid grid & patch configuration\n");
    }
    // One exchange per PHI_HALO sweeps, plus the trailing exchange-only round.
    jacobi_rounds = (jacobi_iters + PHI_HALO - 1) / PHI_HALO;

    if (dist_type < 0 || dist_type > 2) {
      CkAbort("Invalid distribution type %d (-d): 0 uniform, 1 gaussian "
          "bunch, 2 two-stream\n", dist_type);
    }
    // Only safe under +LBAsync, and not as a preference: with the flag off,
    // an AtSyncStart() with no step due calls ResumeFromSync() inline *and*
    // returns, so a patch written to the split pattern would drive runStep
    // twice. Fall back rather than let that happen.
    if (async_lb && !_lb_args.lbAsync()) {
      CkPrintf("[WARN] -a ignored: async LB needs +LBAsync on the command "
               "line. Running the unsplit AtSync barrier instead.\n");
      async_lb = 0;
    }
    if (async_lb && (lb_wait_lag < 1 || lb_wait_lag >= lb_freq)) {
      CkAbort("-l (%d) must be in [1, lb_freq): the wait must land before "
              "the next AtSyncStart\n", lb_wait_lag);
    }

    n_chares_x = grid_width / block_width;
    n_chares_y = grid_height / block_height;

    n_total_particles = (long)ppc * grid_width * grid_height;
    bunch_sigma = sigma_frac * std::min(grid_width, grid_height);

    if (sim_dt <= 0.0) {
      // Cap dt so that the fastest particles cannot cross a whole patch in
      // one step (the push kernel flags >1-patch hops as an error)
      double vmax = std::max(1.0, std::abs(drift_vx) + 4.0 * vth);
      sim_dt = std::min(0.1, 0.25 * std::min(block_width, block_height) / vmax);
    }

    // A patch can never hold more than every particle in the simulation
    long cap = (long)((double)ppc * block_width * block_height * headroom);
    cap = std::min(cap, n_total_particles);
    if (cap > 2000000000L) cap = 2000000000L;
    part_capacity = (int)cap;
    exch_capacity = std::max(1024, (int)(part_capacity * exch_frac));

    CkPrintf("\n[CUDA 2D electrostatic PIC]\n");
    CkPrintf("Grid: %d x %d, Patch: %d x %d, Chares: %d x %d\n",
        grid_width, grid_height, block_width, block_height, n_chares_x,
        n_chares_y);
    CkPrintf("Particles: %ld total (%d per cell), capacity/patch: %d, "
        "exchange capacity: %d\n",
        n_total_particles, ppc, part_capacity, exch_capacity);
    if (dist_type == 1) {
      CkPrintf("Distribution: gaussian bunch, bunch fraction: %.2lf, "
          "sigma: %.1lf cells, drift: %.2lf, vth: %.2lf\n",
          bunch_frac, bunch_sigma, drift_vx, vth);
    } else {
      CkPrintf("Distribution: %s, drift: %.2lf, vth: %.2lf\n",
          dist_type == 0 ? "uniform" : "two-stream", drift_vx, vth);
    }
    CkPrintf("dt: %.4lf, coupling: %.2e, jacobi iterations: %d\n",
        sim_dt, coupling, jacobi_iters);
    CkPrintf("Iterations: %d (+%d warmup), first LB: %d, LB frequency: %d\n\n",
        n_iters, warmup_iters, first_lb, lb_freq);

    patch_proxy = CProxy_Patch::ckNew(n_chares_x, n_chares_y);
    init_start_time = CkWallTimer();
    patch_proxy.init();
  }

  void initDone(long total_np) {
    if (total_np != n_total_particles) {
      CkAbort("Particle init mismatch: created %ld, expected %ld\n",
          total_np, n_total_particles);
    }
    CkPrintf("Init time: %.3lf s (%ld particles)\n",
        CkWallTimer() - init_start_time, total_np);

    start_time = CkWallTimer();
    window_start_time = start_time;
    patch_proxy.iterate();
  }

  void stepStats(CkReductionMsg* msg) {
    CkReduction::tupleElement* results = nullptr;
    int num_elems = 0;
    msg->toTuple(&results, &num_elems);
    long total_np = *(long*)results[0].data;
    long max_np = *(long*)results[1].data;
    // Sum of phi^2 over every interior cell, when CHARM_PIC2D_CHECKSUM is set.
    // The particle-imbalance figure this used to be checked against is an
    // aggregate printed to two decimals, and two runs of the same build differ
    // at that precision -- far too coarse to tell a correct field from a
    // slightly wrong one. Squares rather than values so nothing cancels.
    double phi_sum = (num_elems > 2) ? *(double*)results[2].data : 0.0;
    delete msg;
    delete [] results;

    if (total_np != n_total_particles) {
      CkAbort("Particle count not conserved at iteration %d: %ld of %ld\n",
          stats_iter + 1, total_np, n_total_particles);
    }

    stats_iter++;
    if (stats_iter == warmup_iters) {
      start_time = CkWallTimer();
      window_start_time = start_time;
    }

    if (stats_freq > 0 && stats_iter > warmup_iters &&
        (stats_iter - warmup_iters) % stats_freq == 0) {
      double now = CkWallTimer();
      double avg_np = (double)n_total_particles / (n_chares_x * n_chares_y);
      // %.15e on the checksum: a build-to-build comparison needs every digit,
      // since the point is to catch a field error too small for the imbalance
      // figure beside it to register.
      if (phi_sum != 0.0)
        CkPrintf("Iter %d: %.3lf ms/iter, particle imbalance (max/avg): %.2lf, "
            "phi2 %.15e\n",
            stats_iter, (now - window_start_time) / stats_freq * 1e3,
            (double)max_np / avg_np, phi_sum);
      else
        CkPrintf("Iter %d: %.3lf ms/iter, particle imbalance (max/avg): %.2lf\n",
            stats_iter, (now - window_start_time) / stats_freq * 1e3,
            (double)max_np / avg_np);
      window_start_time = now;
    }
  }

  void allDone() {
    double total_time = CkWallTimer() - start_time;
    CkPrintf("\nTotal time: %.3lf s\nAverage iteration time: %.3lf ms\n",
        total_time, total_time / n_iters * 1e3);

    if (print_parts) {
      patch_proxy(0, 0).print();
    } else {
      CkExit();
    }
  }

  void printDone() {
    CkExit();
  }
};

class Patch : public CBase_Patch {
  Patch_SDAG_CODE

 public:
  int my_iter;
  bool instrumenting = true;  // matches the runtime default at startup

  // Zerocopy send-completion accounting. A send's source buffer is live until
  // its completion callback fires; these counters are what the two gates in
  // runStep wait on. The phi counters are per round parity, matching the slab
  // double-buffering. All travel: sendDone() entries follow a migrated patch.
  int outstanding_sends = 0;   // charge + E + particle sends of this step
  int phi_out[2] = {0, 0};     // phi sends, by round parity
  bool drain_pending = false;  // gateStepDrain is waiting
  bool phi_gate_pending = false;  // gatePhiRound is waiting on cur parity

  // Async LB state (see iterate): the step was joined at lb_start_iter and
  // AtSyncWait is owed lb_wait_lag iterations later.
  bool lb_waiting = false;
  int lb_start_iter = 0;
  int park_skips = 0;
  int phi_iter;
  int recv_count;
  int jacobi_k;
  int jacobi_done;   // sweeps completed this timestep
  int np;
  int cur;  // which of d_parts[2] holds the live particles
  int x, y;
  double px0, py0;  // patch origin in global cells
  int nbr_x[NUM_DIRS], nbr_y[NUM_DIRS];

  RealType* d_rho;
  RealType* d_phi;
  RealType* d_phi_new;
  float2* d_efield;
  Particle* d_parts[2];
  Particle* d_send_parts;  // 8 segments of exch_capacity
  Particle* d_recv_parts;  // 8 segments of exch_capacity
  RealType* d_send_rho;    // 8-way slab, 2*(W+H)+4
  RealType* d_recv_rho;
  // Second 8-way slab, for shipping FINAL rho outward once the charge
  // accumulation above has settled. Separate buffers because that exchange is
  // still in flight -- its sends are only retired by sendDone.
  RealType* d_send_rhoh;
  RealType* d_recv_rhoh;
  float2* d_send_e;        // 8-way slab, 2*(W+H)+4
  float2* d_recv_e;
  // Phi slabs are double-buffered by Jacobi round parity: a neighbor may run
  // one exchange round ahead, so round k+1 must not reuse round k's buffers
  // while they may still be in flight
  RealType* d_send_phi;    // 2 x 4-way slab, 2*(W+H) each
  RealType* d_recv_phi;
  int* d_counts;
  int* h_counts;           // pinned

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;
  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  Patch() {
    usesAtSync = true;
    jacobi_done = 0;
  }

  Patch(CkMigrateMessage* m) {
    usesAtSync = true;
    createCudaEntities();
  }

  ~Patch() {
    // Under async LB this destructor runs mid-step on a migrating element:
    // kernels and the staging copies of its own zerocopy sends may still be
    // in flight on these streams. Settle them before freeing what they read.
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(comm_stream);
    if (outstanding_sends != 0 || phi_out[0] != 0 || phi_out[1] != 0) {
      // A transport may still be reading a send buffer (direct IPC reads the
      // source allocation itself). Leak rather than pull memory out from
      // under it -- the jacobi2d-imbalance convention.
      if (getenv("CHARM_DEBUG_MIGRATE"))
        CkPrintf("[%d] (%d,%d) leaking device buffers at destruction: "
                 "%d+%d+%d sends outstanding\n", CkMyPe(), thisIndex.x,
                 thisIndex.y, outstanding_sends, phi_out[0], phi_out[1]);
      hapiCheck(cudaStreamDestroy(compute_stream));
      hapiCheck(cudaStreamDestroy(comm_stream));
      hapiCheck(cudaEventDestroy(compute_event));
      hapiCheck(cudaEventDestroy(comm_event));
      return;
    }
    hapiCheck(hapiFree(d_rho));
    hapiCheck(hapiFree(d_phi));
    hapiCheck(hapiFree(d_phi_new));
    hapiCheck(hapiFree(d_efield));
    hapiCheck(hapiFree(d_parts[0]));
    hapiCheck(hapiFree(d_parts[1]));
    hapiCheck(hapiFree(d_send_parts));
    hapiCheck(hapiFree(d_recv_parts));
    hapiCheck(hapiFree(d_send_rho));
    hapiCheck(hapiFree(d_recv_rho));
    hapiCheck(hapiFree(d_send_rhoh));
    hapiCheck(hapiFree(d_recv_rhoh));
    hapiCheck(hapiFree(d_send_e));
    hapiCheck(hapiFree(d_recv_e));
    hapiCheck(hapiFree(d_send_phi));
    hapiCheck(hapiFree(d_recv_phi));
    hapiCheck(hapiFree(d_counts));
    hapiCheck(hapiFreeHost(h_counts));

    hapiCheck(cudaStreamDestroy(compute_stream));
    hapiCheck(cudaStreamDestroy(comm_stream));
    hapiCheck(cudaEventDestroy(compute_event));
    hapiCheck(cudaEventDestroy(comm_event));
  }

  void createCudaEntities() {
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream,
        cudaStreamNonBlocking, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream,
        cudaStreamNonBlocking, -1));
    hapiCheck(cudaEventCreateWithFlags(&compute_event,
        cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));
  }

  void computeNeighbors() {
    x = thisIndex.x;
    y = thisIndex.y;
    px0 = (double)x * block_width;
    py0 = (double)y * block_height;
    for (int d = 0; d < NUM_DIRS; d++) {
      nbr_x[d] = (x + DIR_DX[d] + n_chares_x) % n_chares_x;
      nbr_y[d] = (y + DIR_DY[d] + n_chares_y) % n_chares_y;
    }
  }

  void allocateDeviceBuffers() {
    size_t field_size = sizeof(RealType) * FIELD_W * FIELD_H;
    int slab8 = 2 * (block_width + block_height) + 4;
    int slab4 = phiSlab4();

    hapiCheck(hapiMalloc((void**)&d_rho, field_size));
    hapiCheck(hapiMalloc((void**)&d_phi, field_size));
    hapiCheck(hapiMalloc((void**)&d_phi_new, field_size));
    hapiCheck(hapiMalloc((void**)&d_efield,
        sizeof(float2) * FIELD_W * FIELD_H));
    hapiCheck(hapiMalloc((void**)&d_parts[0],
        sizeof(Particle) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_parts[1],
        sizeof(Particle) * part_capacity));
    hapiCheck(hapiMalloc((void**)&d_send_parts,
        sizeof(Particle) * (size_t)NUM_DIRS * exch_capacity));
    hapiCheck(hapiMalloc((void**)&d_recv_parts,
        sizeof(Particle) * (size_t)NUM_DIRS * exch_capacity));
    hapiCheck(hapiMalloc((void**)&d_send_rho, sizeof(RealType) * slab8));
    hapiCheck(hapiMalloc((void**)&d_recv_rho, sizeof(RealType) * slab8));
    // Zero-sized when no sweep reads outside the interior; hapiMalloc(0) is
    // not worth relying on, so keep one element.
    const int rslab = rhoHaloSlab() > 0 ? rhoHaloSlab() : 1;
    hapiCheck(hapiMalloc((void**)&d_send_rhoh, sizeof(RealType) * rslab));
    hapiCheck(hapiMalloc((void**)&d_recv_rhoh, sizeof(RealType) * rslab));
    hapiCheck(hapiMalloc((void**)&d_send_e, sizeof(float2) * slab8));
    hapiCheck(hapiMalloc((void**)&d_recv_e, sizeof(float2) * slab8));
    hapiCheck(hapiMalloc((void**)&d_send_phi, sizeof(RealType) * 2 * slab4));
    hapiCheck(hapiMalloc((void**)&d_recv_phi, sizeof(RealType) * 2 * slab4));
    hapiCheck(hapiMalloc((void**)&d_counts, sizeof(int) * NUM_COUNTERS));
    hapiCheck(hapiMallocHost((void**)&h_counts, sizeof(int) * NUM_COUNTERS));
  }

  void pup(PUP::er& p) {
    // Migration copies phi and the particles straight off the device on a
    // stream of its own that is not ordered against ours. Under async LB this
    // patch is still computing when that happens, so settle our own work
    // first, or the pack copies out a half-written grid -- and the frees that
    // follow pull buffers out from under in-flight kernels and staging
    // copies. (jacobi2d-imbalance does the same; sync-mode migration never
    // needed it because AtSync ran with both streams drained.)
    if (p.isPacking()) {
      cudaStreamSynchronize(compute_stream);
      cudaStreamSynchronize(comm_stream);
    }
    p | my_iter;
    p | phi_iter;
    // The SDAG dependency state travels too: admission control can migrate a
    // patch mid-step, and the continuations (which whens are outstanding,
    // buffered ordinary messages) must follow it. Safe only because the
    // consumption-holds guarantee no device-zerocopy message -- whose payload
    // pointers cannot cross a process -- is buffered here at migration time.
    _sdag_pup(p);
    p | recv_count;
    p | jacobi_k;
    p | jacobi_done;
    p | np;
    p | cur;
    p | outstanding_sends;
    p | phi_out[0];
    p | phi_out[1];
    p | lb_waiting;
    p | lb_start_iter;

    if (p.isUnpacking()) {
      computeNeighbors();
      allocateDeviceBuffers();
    }

    // Only the live particles and the potential (warm start for the next
    // solve) migrate; rho, E and all exchange buffers are recomputed or
    // overwritten every step. Particles are pup'ed as a flat RealType array
    // since the device pup overload only accepts fundamental types.
    p(d_phi, FIELD_W * FIELD_H, PUP::PUPMode::DEVICE);
    p((RealType*)d_parts[cur], (size_t)np * (sizeof(Particle) / sizeof(RealType)),
        PUP::PUPMode::DEVICE);
  }

  void init() {
    my_iter = 0;
    phi_iter = 0;
    np = 0;
    cur = 0;

    computeNeighbors();
    allocateDeviceBuffers();
    createCudaEntities();

    size_t field_size = sizeof(RealType) * FIELD_W * FIELD_H;
    hapiCheck(cudaMemsetAsync(d_phi, 0, field_size, compute_stream));
    hapiCheck(cudaMemsetAsync(d_phi_new, 0, field_size, compute_stream));
    hapiCheck(cudaMemsetAsync(d_efield, 0,
        sizeof(float2) * FIELD_W * FIELD_H,
        compute_stream));
    hapiCheck(cudaMemsetAsync(d_counts, 0, sizeof(int) * NUM_COUNTERS,
        compute_stream));

    long n_bunch = (dist_type == 1) ?
        (long)(bunch_frac * n_total_particles) : 0;
    invokeInitParticlesKernel(d_parts[0], &d_counts[0], n_total_particles,
        n_bunch, dist_type, 0.5f * grid_width, 0.5f * grid_height,
        (float)bunch_sigma, (float)drift_vx, (float)vth, (float)grid_width,
        (float)grid_height, (float)px0, (float)py0,
        (float)(px0 + block_width), (float)(py0 + block_height),
        part_capacity, &d_counts[ERR_COUNTER], compute_stream);
    hapiCheck(hapiMemcpyAsync(h_counts, d_counts,
        sizeof(int) * NUM_COUNTERS, cudaMemcpyDeviceToHost, compute_stream));

    CkCallback* cb = new CkCallback(CkIndex_Patch::initDone(),
        thisProxy[thisIndex]);
    hapiAddCallback(compute_stream, cb);
  }

  void initDone() {
    if (h_counts[ERR_COUNTER]) {
      CkAbort("Patch (%d,%d): initial particles exceed capacity %d; "
          "increase headroom (-r)\n", x, y, part_capacity);
    }
    np = h_counts[0];

    long npl = np;
    contribute(sizeof(long), &npl, CkReduction::sum_long,
        CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  // Load-balancing instrumentation -- GPU kernel tracing especially -- costs a
  // noticeable fraction of each step, but the balancer only ever reads the
  // loads it gathers at an AtSync step. So keep it off and switch it on for the
  // few steps leading up to one, which is enough to characterise the load.
  bool isLBIter(int it) const {
    return it == first_lb || (it != 0 && lb_freq > 0 && it % lb_freq == 0);
  }

  // The next iteration at which this chare will call AtSync.
  int nextLBIter(int it) const {
    if (it < first_lb) return first_lb;
    if (lb_freq <= 0) return INT_MAX;
    return ((it / lb_freq) + 1) * lb_freq;
  }

  void iterate() {
    // Drive instrumentation off the distance to the next AtSync, rather than
    // switching it on at one particular iteration -- with closely spaced
    // load-balancing steps the latter can leave a step with no measurements at
    // all, and the balancer then decides on zero load. Toggle only on a
    // transition so the common case costs a comparison.
    const bool want = (nextLBIter(my_iter) - my_iter) <= LB_INSTRUMENT_WINDOW;
    if (want != instrumenting) {
      instrumenting = want;
      if (want) LBTurnInstrumentOn(); else LBTurnInstrumentOff();
    }

    if (isLBIter(my_iter) && !lb_waiting) {
      // The !lb_waiting guard: one step at a time per element. If the wait
      // for the previous step is still owed -- the quiet check can delay the
      // park -- starting another AtSyncStart aborts by contract; the missed
      // step is simply skipped and the cadence resumes at the next one.
      cudaStreamSynchronize(comm_stream);
      cudaStreamSynchronize(compute_stream);
      if (!async_lb) {
        // The patch stops here; the next thing it hears (ResumeFromSync) is
        // that the whole step -- strategy and migrations -- is over.
        AtSync();
        return;
      }
      // Split barrier: join the step and keep iterating while the strategy
      // runs and other patches migrate. Fixed cadence and no MetaBalancer, so
      // every AtSyncStart starts a step; the runStep gates guarantee no
      // zerocopy send was in flight across the join.
      AtSyncStart();
      lb_start_iter = my_iter;
      lb_waiting = true;
      thisProxy[thisIndex].runStep();
    } else if (lb_waiting && my_iter >= lb_start_iter + lb_wait_lag) {
      // Park unconditionally, even with device sends still draining. Every
      // patch parks here before running this iteration's step, so no ghost
      // for it can have been sent yet -- nothing unconsumed can be buffered.
      // A patch that skips the park to wait out its drain runs a step its
      // parked neighbors never feed, and wedges the whole app; the runtime
      // already defers an actual migration until the drain completes.
      // The other half of the split, a few iterations later: the patch has to
      // be quiesced before it can be pupped, and AtSyncWait is where the step
      // is collected. Resumes inline if the step is already over.
      lb_waiting = false;
      cudaStreamSynchronize(comm_stream);
      cudaStreamSynchronize(compute_stream);
      AtSyncWait();
    } else {
      thisProxy[thisIndex].runStep();
    }
  }

  void ResumeFromSync() {
    // iterate() switches instrumentation back off once the next AtSync is far
    // enough away; leaving it to that keeps the decision in one place.
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[RESUME %d] (%d,%d) iter=%d\n", CkMyPe(), thisIndex.x,
               thisIndex.y, my_iter);
    thisProxy[thisIndex].runStep();
  }

  // ---- Phase 1: charge deposition ----

  void startDeposit() {
    // Particles appended on the comm stream last step must be visible
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    hapiCheck(cudaMemsetAsync(d_rho, 0,
        sizeof(RealType) * FIELD_W * FIELD_H,
        compute_stream));
    float weight = (float)(-coupling / ppc);
    invokeDepositKernel(d_parts[cur], np, d_rho, weight, (float)px0,
        (float)py0, block_width, block_height, compute_stream);

    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));
    invokePackChargeGhostsKernel(d_rho, d_send_rho, block_width, block_height,
        comm_stream);

    CkCallback* cb = new CkCallback(CkIndex_Patch::chargeGhostsPacked(),
        thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
  }

  // Only reached when the sweeps actually need it; at PHI_HALO 1 the sweep
  // runs at margin 0 and never looks outside the interior.
  static bool rhoHaloNeeded() { return RHO_HALO_D > 0; }

  void packRhoHalo() {
    // rho was last written by accumChargeGhost on the comm stream.
    invokePackRhoHaloKernel(d_rho, d_send_rhoh, block_width, block_height,
        comm_stream);
    CkCallback* cb = new CkCallback(CkIndex_Patch::rhoHaloPacked(),
        thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
  }

  void sendRhoHalo() {
    for (int d = 0; d < NUM_DIRS; d++) {
      thisProxy(nbr_x[d], nbr_y[d]).receiveRhoHalo(my_iter, flipDir(d),
          rhoHaloLen(d),
          (outstanding_sends++,
           CkDeviceBuffer(d_send_rhoh + rhoHaloOff(d),
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  void receiveRhoHalo(int ref, int dir, int& n, RealType*& buf,
      CkDeviceBufferPost* devicePost) {
    buf = d_recv_rhoh + rhoHaloOff(dir);
    devicePost[0].hapi_stream = comm_stream;
  }

  void unpackRhoHalo(int dir, int n, RealType* buf) {
    invokeUnpackRhoHaloKernel(d_rho, buf, dir, block_width, block_height,
        comm_stream);
  }

  void sendChargeGhosts() {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[PH1] (%d,%d) iter=%d\n", thisIndex.x, thisIndex.y, my_iter);
    for (int d = 0; d < NUM_DIRS; d++) {
      thisProxy(nbr_x[d], nbr_y[d]).receiveChargeGhosts(my_iter, flipDir(d),
          stripLen(d),
          (outstanding_sends++,
           CkDeviceBuffer(d_send_rho + stripOff(d),
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  // Post entry method: places the incoming strip in the matching segment of
  // the receive slab; ordered behind its consumers via the comm stream
  void receiveChargeGhosts(int ref, int dir, int& n, RealType*& buf,
      CkDeviceBufferPost* devicePost) {
    buf = d_recv_rho + stripOff(dir);
    devicePost[0].hapi_stream = comm_stream;
    if (getenv("CHARM_DEBUG_IPC_RECV")) {
      const int slab8 = 2 * (block_width + block_height) + 4;
      CkPrintf("[%d] post (%d,%d): ref=%d dir=%d n=%d off=%d slab8=%d "
               "recv_base=%p buf=%p%s\n",
               CkMyPe(), thisIndex.x, thisIndex.y, ref, dir, n, stripOff(dir),
               slab8, (void*)d_recv_rho, (void*)buf,
               (dir < 0 || dir >= NUM_DIRS || stripOff(dir) + n > slab8)
                 ? "  <-- OUT OF RANGE" : "");
    }
  }

  void accumChargeGhost(int dir, int n, RealType* buf) {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[CRECV] (%d,%d) iter=%d dir=%d\n", thisIndex.x, thisIndex.y,
               my_iter, dir);
    if (getenv("CHARM_DEBUG_IPC_RECV")) {
      const int slab8 = 2 * (block_width + block_height) + 4;
      CkPrintf("[%d] accum (%d,%d): dir=%d n=%d buf=%p recv_base=%p "
               "delta=%ld rho=%p%s\n",
               CkMyPe(), thisIndex.x, thisIndex.y, dir, n, (void*)buf,
               (void*)d_recv_rho, (long)(buf - d_recv_rho), (void*)d_rho,
               (dir < 0 || dir >= NUM_DIRS ||
                buf < d_recv_rho || buf - d_recv_rho >= slab8)
                 ? "  <-- BAD" : "");
    }
    invokeAccumChargeGhostKernel(d_rho, buf, dir, block_width, block_height,
        comm_stream);
  }

  // ---- Phase 2: Jacobi solve of the Poisson equation ----

  void packPhiGhosts() {
    // phi was last written on the compute stream (previous Jacobi update)
    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));

    RealType* slab = d_send_phi + (size_t)(phi_iter & 1) * phiSlab4();
    // One launch for all four directions; top and bottom used to be
    // 2*PHI_HALO separate device-to-device copies on top of the LR kernel.
    invokePackPhiKernel(d_phi, slab, block_width, block_height, comm_stream);

    CkCallback* cb = new CkCallback(CkIndex_Patch::phiGhostsPacked(),
        thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
  }

  void sendPhiGhosts() {
    RealType* slab = d_send_phi + (size_t)(phi_iter & 1) * phiSlab4();
    for (int d = 0; d < PHI_DIRS; d++) {
      thisProxy(nbr_x[d], nbr_y[d]).receivePhiGhosts(phi_iter, flipDir(d),
          phiStripLen(d),
          (phi_out[phi_iter & 1]++,
           CkDeviceBuffer(slab + phiOff(d),
               CkCallback((phi_iter & 1) ? CkIndex_Patch::phiSendDoneOdd()
                                         : CkIndex_Patch::phiSendDoneEven(),
                          thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  void receivePhiGhosts(int ref, int dir, int& n, RealType*& buf,
      CkDeviceBufferPost* devicePost) {
    // ref is the sender's round number; pick the matching parity buffer
    // phiSlab4(), not the old 2*(W+H): that was the depth-1 slab size, so at
    // any deeper halo the odd-parity landing region overlapped the even one
    // and consecutive rounds scribbled on each other.
    buf = d_recv_phi + (size_t)(ref & 1) * phiSlab4() + phiOff(dir);
    devicePost[0].hapi_stream = comm_stream;
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] postPHI (%d,%d): ref=%d dir=%d n=%d base=%p buf=%p\n",
               CkMyPe(), thisIndex.x, thisIndex.y, ref, dir, n,
               (void*)d_recv_phi, (void*)buf);
  }

  // Forensic probe: is a received device pointer inside the posted buffer it
  // is supposed to be? A foreign pointer means the message was preprocessed
  // in another process (retargeted to ITS posted buffers) and delivered here
  // after a migration. Report and skip instead of copying: a poisoned async
  // copy would take the whole context down and hide every later event.
  bool foreignBuf(const char* what, const RealType* buf, const RealType* base,
                  size_t span, int dir, int n) {
    if (buf >= base && buf < base + span) return false;
    CkPrintf("[FOREIGN] pe=%d (%d,%d) %s iter=%d phi_iter=%d dir=%d n=%d "
             "buf=%p base=%p\n", CkMyPe(), thisIndex.x, thisIndex.y, what,
             my_iter, phi_iter, dir, n, (void*)buf, (void*)base);
    fflush(stdout);
    return true;
  }

  void unpackPhiGhost(int dir, int n, RealType* buf) {
    if (foreignBuf("phi", buf, d_recv_phi, (size_t)2 * phiSlab4(), dir, n))
      return;
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] usePHI  (%d,%d): dir=%d n=%d buf=%p base=%p delta=%ld\n",
               CkMyPe(), thisIndex.x, thisIndex.y, dir, n, (void*)buf,
               (void*)d_recv_phi, (long)(buf - d_recv_phi));
    // One kernel for every direction now, corners included; top and bottom
    // used to be per-layer device-to-device copies.
    invokeUnpackPhiKernel(d_phi, buf, dir, block_width, block_height,
        comm_stream);
  }

  void jacobiUpdate() {
    // Fresh ghosts (and, on the first iteration, accumulated ghost charge)
    // were written on the comm stream
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    // One exchange buys PHI_HALO sweeps. Each consumes a halo layer, so the
    // swept margin shrinks to zero on the last of them; `remaining` stops the
    // block short when fewer sweeps are left than the halo would allow.
    float rho_bar = (float)(-coupling);
    const int remaining = jacobi_iters - jacobi_done;
    const int sweeps = (remaining < PHI_HALO) ? remaining : PHI_HALO;
    for (int s = 0; s < sweeps; s++) {
      invokeJacobiPhiKernel(d_phi, d_phi_new, d_rho, rho_bar, block_width,
          block_height, sweeps - 1 - s, compute_stream);
      std::swap(d_phi, d_phi_new);
    }
    jacobi_done += sweeps;
  }

  // ---- Phase 3: electric field ----

  void computeEField() {
    // The final phi ghost exchange was written on the comm stream
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));
    invokeEFieldKernel(d_phi, d_efield, block_width, block_height,
        compute_stream);

    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));
    invokePackEGhostsKernel(d_efield, d_send_e, block_width, block_height,
        comm_stream);

    CkCallback* cb = new CkCallback(CkIndex_Patch::eGhostsPacked(),
        thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
  }

  void sendEGhosts() {
    for (int d = 0; d < NUM_DIRS; d++) {
      thisProxy(nbr_x[d], nbr_y[d]).receiveEGhosts(my_iter, flipDir(d),
          2 * stripLen(d),
          (outstanding_sends++,
           CkDeviceBuffer((RealType*)(d_send_e + stripOff(d)),
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               comm_stream)));
    }
  }

  void receiveEGhosts(int ref, int dir, int& n, RealType*& buf,
      CkDeviceBufferPost* devicePost) {
    buf = (RealType*)(d_recv_e + stripOff(dir));
    devicePost[0].hapi_stream = comm_stream;
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] postE   (%d,%d): ref=%d dir=%d n=%d base=%p buf=%p\n",
               CkMyPe(), thisIndex.x, thisIndex.y, ref, dir, n,
               (void*)d_recv_e, (void*)buf);
  }

  void unpackEGhost(int dir, int n, RealType* buf) {
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] useE    (%d,%d): dir=%d n=%d buf=%p base=%p delta=%ld\n",
               CkMyPe(), thisIndex.x, thisIndex.y, dir, n, (void*)buf,
               (void*)d_recv_e, (long)((float2*)buf - d_recv_e));
    invokeUnpackEGhostKernel(d_efield, (const float2*)buf, dir, block_width,
        block_height, comm_stream);
  }

  // ---- Phase 4: particle push and exchange ----

  void pushParticles() {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[PH4] (%d,%d) iter=%d\n", thisIndex.x, thisIndex.y, my_iter);
    // E ghosts were unpacked on the comm stream
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    hapiCheck(cudaMemsetAsync(d_counts, 0, sizeof(int) * NUM_COUNTERS,
        compute_stream));
    invokePushAndMarkKernel(d_parts[cur], np, d_efield, d_parts[1 - cur],
        d_send_parts, exch_capacity, d_counts, (float)sim_dt, -1.0f,
        (float)px0, (float)py0, (float)grid_width, (float)grid_height,
        block_width, block_height, compute_stream);
    hapiCheck(hapiMemcpyAsync(h_counts, d_counts,
        sizeof(int) * NUM_COUNTERS, cudaMemcpyDeviceToHost, compute_stream));

    CkCallback* cb = new CkCallback(CkIndex_Patch::pushDone(),
        thisProxy[thisIndex]);
    hapiAddCallback(compute_stream, cb);
  }

  void finishPush() {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[PUSHD] (%d,%d) iter=%d\n", thisIndex.x, thisIndex.y, my_iter);
    if (h_counts[ERR_COUNTER]) {
      CkAbort("Patch (%d,%d): particle moved more than one patch in one "
          "step at iteration %d; reduce dt (-T) or drift (-v)\n",
          x, y, my_iter);
    }
    for (int d = 0; d < NUM_DIRS; d++) {
      if (h_counts[d] > exch_capacity) {
        CkAbort("Patch (%d,%d): %d outgoing particles exceed exchange "
            "capacity %d; increase -x or -r\n",
            x, y, h_counts[d], exch_capacity);
      }
    }

    np = h_counts[STAY];
    cur = 1 - cur;

    for (int d = 0; d < NUM_DIRS; d++) {
      int cnt = h_counts[d];
      thisProxy(nbr_x[d], nbr_y[d]).receiveParticles(my_iter, flipDir(d), cnt,
          std::max(cnt, 1),
          (outstanding_sends++,
           CkDeviceBuffer(d_send_parts + (size_t)d * exch_capacity,
               CkCallback(CkIndex_Patch::sendDone(), thisProxy[thisIndex]),
               compute_stream)));
    }
  }

  void receiveParticles(int ref, int dir, int n, int& m, Particle*& parts,
      CkDeviceBufferPost* devicePost) {
    parts = d_recv_parts + (size_t)dir * exch_capacity;
    devicePost[0].hapi_stream = comm_stream;
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] postPART(%d,%d): ref=%d dir=%d n=%d m=%d cap=%d base=%p parts=%p\n",
               CkMyPe(), thisIndex.x, thisIndex.y, ref, dir, n, m,
               exch_capacity, (void*)d_recv_parts, (void*)parts);
  }

  void appendParticles(int dir, int n) {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[PRECV] (%d,%d) iter=%d\n", thisIndex.x, thisIndex.y, my_iter);
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] usePART (%d,%d): dir=%d n=%d np=%d cap=%d%s\n",
               CkMyPe(), thisIndex.x, thisIndex.y, dir, n, np, part_capacity,
               (dir < 0 || dir >= NUM_DIRS || n < 0 || n > exch_capacity)
                 ? "  <-- BAD" : "");
    if (n == 0) return;
    if (np + n > part_capacity) {
      CkAbort("Patch (%d,%d): %d particles exceed capacity %d at iteration "
          "%d; increase headroom (-r)\n",
          x, y, np + n, part_capacity, my_iter);
    }
    hapiCheck(hapiMemcpyAsync(d_parts[cur] + np,
        d_recv_parts + (size_t)dir * exch_capacity, sizeof(Particle) * n,
        cudaMemcpyDeviceToDevice, comm_stream));
    np += n;
  }

  void sendDone() {
    outstanding_sends--;
    maybeReleaseGates();
  }
  void phiSendDoneEven() { phi_out[0]--; maybeReleaseGates(); }
  void phiSendDoneOdd()  { phi_out[1]--; maybeReleaseGates(); }

  // Round k+2's pack reuses round k's parity slab, so it waits for that
  // parity's sends. In steady state two rounds of neighbour latency have
  // passed and this releases immediately.
  void gatePhiRound() {
    if (phi_out[phi_iter & 1] == 0) {
      thisProxy[thisIndex].phiRoundFree();
    } else {
      phi_gate_pending = true;
    }
  }

  // The next iteration repacks every send buffer, and under async LB a
  // migration's pup frees them; neither may happen over a live transport read.
  void gateStepDrain() {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[GATE] (%d,%d) iter=%d\n", thisIndex.x, thisIndex.y, my_iter);
    if (outstanding_sends == 0 && phi_out[0] == 0 && phi_out[1] == 0) {
      thisProxy[thisIndex].sendsDrained();
    } else {
      drain_pending = true;
    }
  }

  void maybeReleaseGates() {
    if (phi_gate_pending && phi_out[phi_iter & 1] == 0) {
      phi_gate_pending = false;
      thisProxy[thisIndex].phiRoundFree();
    }
    if (drain_pending && outstanding_sends == 0 && phi_out[0] == 0 &&
        phi_out[1] == 0) {
      drain_pending = false;
      thisProxy[thisIndex].sendsDrained();
    }
  }

  static bool checksumOn() {
    static const bool on = (getenv("CHARM_PIC2D_CHECKSUM") != nullptr);
    return on;
  }

  // Deterministic on purpose: the field comes back to the host and is summed
  // in a fixed order in double. A device-side reduction with atomics would
  // reorder between runs and blur the low bits, which is the opposite of what
  // a build-to-build comparison needs.
  double phiCheckSum() {
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(comm_stream);
    std::vector<RealType> h((size_t)FIELD_W * FIELD_H);
    hapiCheck(cudaMemcpy(h.data(), d_phi,
        sizeof(RealType) * FIELD_W * FIELD_H, cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int j = 1; j <= block_height; j++)
      for (int i = 1; i <= block_width; i++) {
        const double v = (double)h[IDX(i, j)];
        s += v * v;
      }
    return s;
  }

  void endOfStep() {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[EOS] pe=%d (%d,%d) iter=%d\n", CkMyPe(), thisIndex.x,
               thisIndex.y, my_iter);
    long sum_np = np;
    long max_np = np;
    // Off unless asked for: it drains both streams and pulls the field back to
    // the host, which would distort exactly the timings this app is used to
    // measure. On, it is what makes a field regression visible.
    double phi_sq = checksumOn() ? phiCheckSum() : 0.0;
    CkReduction::tupleElement tuple[] = {
        CkReduction::tupleElement(sizeof(long), &sum_np, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(long), &max_np, CkReduction::max_long),
        CkReduction::tupleElement(sizeof(double), &phi_sq, CkReduction::sum_double)};
    CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple, 3);
    msg->setCallback(CkCallback(CkIndex_Main::stepStats(NULL), main_proxy));
    contribute(msg);

    if (my_iter < warmup_iters + n_iters) {
      thisProxy[thisIndex].iterate();
    } else {
      contribute(CkCallback(CkReductionTarget(Main, allDone), main_proxy));
    }
  }

  void print() {
    CkPrintf("Patch (%d,%d) on PE %d: %d particles\n", x, y, CkMyPe(), np);

    if (!(x == n_chares_x - 1 && y == n_chares_y - 1)) {
      if (x == n_chares_x - 1) {
        thisProxy(0, y + 1).print();
      } else {
        thisProxy(x + 1, y).print();
      }
    } else {
      main_proxy.printDone();
    }
  }
};

#include "pic2d.def.h"
