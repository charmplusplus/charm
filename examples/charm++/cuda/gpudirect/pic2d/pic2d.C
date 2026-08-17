#include "hapi.h"
#include "pic2d.decl.h"
#include "pic2d.h"
#include <algorithm>
#include <cmath>
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
/* readonly */ int ppc;
/* readonly */ int first_lb;
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
extern void invokePackChargeGhostsKernel(const RealType* d_rho,
    RealType* d_slab, int block_width, int block_height, cudaStream_t stream);
extern void invokeAccumChargeGhostKernel(RealType* d_rho,
    const RealType* d_buf, int dir, int block_width, int block_height,
    cudaStream_t stream);
extern void invokePackPhiLRKernel(const RealType* d_phi, RealType* d_slab,
    int block_width, int block_height, cudaStream_t stream);
extern void invokeUnpackPhiLRKernel(RealType* d_phi, const RealType* d_buf,
    int dir, int block_width, int block_height, cudaStream_t stream);
extern void invokeJacobiPhiKernel(const RealType* d_phi, RealType* d_phi_new,
    const RealType* d_rho, float rho_bar, int block_width, int block_height,
    cudaStream_t stream);
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

// Offsets in the 4-way phi slab (see packPhiLRKernel in pic2d.cu)
static inline int phiOff(int d) {
  int W = block_width, H = block_height;
  switch (d) {
    case LEFT:   return 0;
    case RIGHT:  return H;
    case TOP:    return 2*H;
    default:     return 2*H + W;  // BOTTOM
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
    stats_freq = 1;
    print_parts = false;
    stats_iter = 0;

    int c;
    while ((c = getopt(m->argc, m->argv, "W:H:w:h:p:i:u:j:d:F:s:v:t:T:q:f:b:r:x:c:P")) != -1) {
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
    if (dist_type < 0 || dist_type > 2) {
      CkAbort("Invalid distribution type %d (-d): 0 uniform, 1 gaussian "
          "bunch, 2 two-stream\n", dist_type);
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
  int phi_iter;
  int recv_count;
  int jacobi_k;
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
  }

  Patch(CkMigrateMessage* m) {
    usesAtSync = true;
    createCudaEntities();
  }

  ~Patch() {
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
    size_t field_size = sizeof(RealType) * (block_width + 2) * (block_height + 2);
    int slab8 = 2 * (block_width + block_height) + 4;
    int slab4 = 2 * (block_width + block_height);

    hapiCheck(hapiMalloc((void**)&d_rho, field_size));
    hapiCheck(hapiMalloc((void**)&d_phi, field_size));
    hapiCheck(hapiMalloc((void**)&d_phi_new, field_size));
    hapiCheck(hapiMalloc((void**)&d_efield,
        sizeof(float2) * (block_width + 2) * (block_height + 2)));
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
    hapiCheck(hapiMalloc((void**)&d_send_e, sizeof(float2) * slab8));
    hapiCheck(hapiMalloc((void**)&d_recv_e, sizeof(float2) * slab8));
    hapiCheck(hapiMalloc((void**)&d_send_phi, sizeof(RealType) * 2 * slab4));
    hapiCheck(hapiMalloc((void**)&d_recv_phi, sizeof(RealType) * 2 * slab4));
    hapiCheck(hapiMalloc((void**)&d_counts, sizeof(int) * NUM_COUNTERS));
    hapiCheck(hapiMallocHost((void**)&h_counts, sizeof(int) * NUM_COUNTERS));
  }

  void pup(PUP::er& p) {
    p | my_iter;
    p | phi_iter;
    p | recv_count;
    p | jacobi_k;
    p | np;
    p | cur;

    if (p.isUnpacking()) {
      computeNeighbors();
      allocateDeviceBuffers();
    }

    // Only the live particles and the potential (warm start for the next
    // solve) migrate; rho, E and all exchange buffers are recomputed or
    // overwritten every step. Particles are pup'ed as a flat RealType array
    // since the device pup overload only accepts fundamental types.
    p(d_phi, (block_width + 2) * (block_height + 2), PUP::PUPMode::DEVICE);
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

    size_t field_size = sizeof(RealType) * (block_width + 2) * (block_height + 2);
    hapiCheck(cudaMemsetAsync(d_phi, 0, field_size, compute_stream));
    hapiCheck(cudaMemsetAsync(d_phi_new, 0, field_size, compute_stream));
    hapiCheck(cudaMemsetAsync(d_efield, 0,
        sizeof(float2) * (block_width + 2) * (block_height + 2),
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

    if (isLBIter(my_iter)) {
      cudaStreamSynchronize(comm_stream);
      cudaStreamSynchronize(compute_stream);
      AtSync();
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
        sizeof(RealType) * (block_width + 2) * (block_height + 2),
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

  void sendChargeGhosts() {
    for (int d = 0; d < NUM_DIRS; d++) {
      thisProxy(nbr_x[d], nbr_y[d]).receiveChargeGhosts(my_iter, flipDir(d),
          stripLen(d), CkDeviceBuffer(d_send_rho + stripOff(d), comm_stream));
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

    RealType* slab = d_send_phi +
        (size_t)(phi_iter & 1) * 2 * (block_width + block_height);
    invokePackPhiLRKernel(d_phi, slab, block_width, block_height,
        comm_stream);
    // Top and bottom interior rows are contiguous
    hapiCheck(hapiMemcpyAsync(slab + phiOff(TOP), d_phi + IDX(1, 1),
        sizeof(RealType) * block_width, cudaMemcpyDeviceToDevice,
        comm_stream));
    hapiCheck(hapiMemcpyAsync(slab + phiOff(BOTTOM),
        d_phi + IDX(1, block_height), sizeof(RealType) * block_width,
        cudaMemcpyDeviceToDevice, comm_stream));

    CkCallback* cb = new CkCallback(CkIndex_Patch::phiGhostsPacked(),
        thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
  }

  void sendPhiGhosts() {
    RealType* slab = d_send_phi +
        (size_t)(phi_iter & 1) * 2 * (block_width + block_height);
    for (int d = 0; d < 4; d++) {
      thisProxy(nbr_x[d], nbr_y[d]).receivePhiGhosts(phi_iter, flipDir(d),
          stripLen(d), CkDeviceBuffer(slab + phiOff(d), comm_stream));
    }
  }

  void receivePhiGhosts(int ref, int dir, int& n, RealType*& buf,
      CkDeviceBufferPost* devicePost) {
    // ref is the sender's round number; pick the matching parity buffer
    buf = d_recv_phi + (size_t)(ref & 1) * 2 * (block_width + block_height) +
        phiOff(dir);
    devicePost[0].hapi_stream = comm_stream;
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] postPHI (%d,%d): ref=%d dir=%d n=%d base=%p buf=%p\n",
               CkMyPe(), thisIndex.x, thisIndex.y, ref, dir, n,
               (void*)d_recv_phi, (void*)buf);
  }

  void unpackPhiGhost(int dir, int n, RealType* buf) {
    if (getenv("CHARM_DEBUG_IPC_RECV"))
      CkPrintf("[%d] usePHI  (%d,%d): dir=%d n=%d buf=%p base=%p delta=%ld\n",
               CkMyPe(), thisIndex.x, thisIndex.y, dir, n, (void*)buf,
               (void*)d_recv_phi, (long)(buf - d_recv_phi));
    if (dir == LEFT || dir == RIGHT) {
      invokeUnpackPhiLRKernel(d_phi, buf, dir, block_width, block_height,
          comm_stream);
    } else if (dir == TOP) {
      hapiCheck(hapiMemcpyAsync(d_phi + IDX(1, 0), buf,
          sizeof(RealType) * block_width, cudaMemcpyDeviceToDevice,
          comm_stream));
    } else {  // BOTTOM
      hapiCheck(hapiMemcpyAsync(d_phi + IDX(1, block_height + 1), buf,
          sizeof(RealType) * block_width, cudaMemcpyDeviceToDevice,
          comm_stream));
    }
  }

  void jacobiUpdate() {
    // Fresh ghosts (and, on the first iteration, accumulated ghost charge)
    // were written on the comm stream
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    float rho_bar = (float)(-coupling);
    invokeJacobiPhiKernel(d_phi, d_phi_new, d_rho, rho_bar, block_width,
        block_height, compute_stream);
    std::swap(d_phi, d_phi_new);
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
          CkDeviceBuffer((RealType*)(d_send_e + stripOff(d)), comm_stream));
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
          CkDeviceBuffer(d_send_parts + (size_t)d * exch_capacity,
              compute_stream));
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

  void endOfStep() {
    long sum_np = np;
    long max_np = np;
    CkReduction::tupleElement tuple[] = {
        CkReduction::tupleElement(sizeof(long), &sum_np, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(long), &max_np, CkReduction::max_long)};
    CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple, 2);
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
