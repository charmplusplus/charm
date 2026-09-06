#include "hapi.h"
#include "moe.decl.h"
#include "moe.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>
#include <sys/time.h>
#include <unistd.h>

// CHARM_MOE_TRACE=1: timestamped LB, migration and dispatcher-step events per
// PE, for reconstructing what a load-balancing step costs and where; =2 adds
// the experts' per-step events. Wall-clock seconds so lines from different
// processes on a node line up.
static inline double tnow() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
static inline int traceLevel() {
  static const int l = getenv("CHARM_MOE_TRACE") ? atoi(getenv("CHARM_MOE_TRACE")) : 0;
  return l;
}
// CHARM_MOE_HOSTSYNC=1: the original migration behaviour, for A/B runs --
// the pup and the destructor stop the host until the shared stream drains.
static inline bool hostSyncMode() {
  static const bool on = (getenv("CHARM_MOE_HOSTSYNC") != nullptr);
  return on;
}
#define TRACE(lvl, fmt, ...) do { \
    if (traceLevel() >= (lvl)) \
      CkPrintf("[T %.6f pe%d] " fmt "\n", tnow(), CkMyPe(), ##__VA_ARGS__); \
  } while (0)

// Device allocation following the runtime's choice: under +gpupool every
// buffer comes from CkDeviceMalloc (an arena the peers have already opened, no
// driver call, no device sync); without it, from hapiMalloc. Same convention
// as pic2d.
inline hapiError_t mmMalloc(void** p, size_t n) {
  static const bool pool = CkDevicePoolOn();
  if (!pool) return hapiMalloc(p, n);
  *p = CkDeviceMalloc(n);
  return (*p != NULL) ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline hapiError_t mmFree(void* p) {
  static const bool pool = CkDevicePoolOn();
  if (p == NULL) return cudaSuccess;
  if (pool) { CkDeviceFree(p); return cudaSuccess; }
  return hapiFree(p);
}

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Dispatcher disp_proxy;
/* readonly */ CProxy_Expert expert_proxy;
/* readonly */ int n_experts;
/* readonly */ int d_model;
/* readonly */ int d_ff;
/* readonly */ int n_tokens;
/* readonly */ int top_k;
/* readonly */ int n_steps;
/* readonly */ int warmup_steps;
/* readonly */ int chunk_tokens;
/* readonly */ int n_lanes;
/* readonly */ int cap_src;
/* readonly */ int n_disp;
/* readonly */ int drift_period;
/* readonly */ int first_lb;
/* readonly */ int lb_freq;
/* readonly */ int async_lb;
/* readonly */ int lb_wait_lag;
/* readonly */ int checksum_freq;
/* readonly */ int stats_freq;
/* readonly */ int use_adam;
/* readonly */ int use_tf32;
/* readonly */ int use_instrument;
/* readonly */ int fuse_sources;
/* readonly */ int print_place;
/* readonly */ double zipf_z;
/* readonly */ double learning_rate;
/* readonly */ long moe_seed;

// With -I (CUPTI-measured loads instead of the experts' own estimates): how
// many steps before an AtSync step to switch instrumentation on, as in pic2d.
#define LB_INSTRUMENT_WINDOW 3

// ---- host hashing: routing is a pure function of (seed, step, PE, token,
// slot), so every run with the same arguments routes identically whatever the
// placement, and the checksums can be compared across runs.

static inline uint64_t sm64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
static inline uint64_t hmix(uint64_t a, uint64_t b) {
  return sm64(a ^ (b * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL));
}
static inline double u01(uint64_t h) {
  return ((double)(h >> 11) + 0.5) / 9007199254740992.0;
}

static inline bool isLBStep(int s) {
  return s == first_lb || (s != 0 && lb_freq > 0 && s % lb_freq == 0);
}
static inline int nextLBStep(int s) {
  if (s < first_lb) return first_lb;
  if (lb_freq <= 0) return INT_MAX;
  return ((s / lb_freq) + 1) * lb_freq;
}

// Per-step figures from one of the two reductions, held until the other
// reduction of the same step arrives so one line reports both.
struct DispRec { double out2; double t; };
struct ExpRec { double tokratio, gpuratio, gmax_ms, expratio, w2; };

class Main : public CBase_Main {
  double init_start_time;
  double start_time;
  double window_start_time;
  int disp_step;   // steps whose dispatcher reduction has arrived
  int exp_step;    // steps whose expert reduction has arrived
  int done_count;
  std::deque<DispRec> disp_q;
  std::deque<ExpRec> exp_q;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;

    n_experts = 64;
    d_model = 2048;
    d_ff = 8192;
    n_tokens = 8192;
    top_k = 1;
    n_steps = 50;
    warmup_steps = 5;
    zipf_z = 1.0;
    drift_period = 10;
    chunk_tokens = 1024;
    n_lanes = 4;
    double headroom = 0.0;
    learning_rate = 1e-4;
    checksum_freq = 5;
    stats_freq = 1;
    first_lb = 10;
    lb_freq = 9999;
    lb_wait_lag = 3;
    moe_seed = 12345;
    async_lb = 0;
    use_adam = 0;
    use_tf32 = 0;
    use_instrument = 0;
    fuse_sources = 1;
    print_place = 0;
    disp_step = exp_step = done_count = 0;

    int c;
    while ((c = getopt(m->argc, m->argv, "e:m:h:t:k:i:u:z:p:c:r:L:C:s:f:b:l:S:n:aAUTIP")) != -1) {
      switch (c) {
        case 'e': n_experts = atoi(optarg); break;
        case 'm': d_model = atoi(optarg); break;
        case 'h': d_ff = atoi(optarg); break;
        case 't': n_tokens = atoi(optarg); break;
        case 'k': top_k = atoi(optarg); break;
        case 'i': n_steps = atoi(optarg); break;
        case 'u': warmup_steps = atoi(optarg); break;
        case 'z': zipf_z = atof(optarg); break;
        case 'p': drift_period = atoi(optarg); break;
        case 'c': chunk_tokens = atoi(optarg); break;
        case 'n': n_lanes = atoi(optarg); break;
        case 'r': headroom = atof(optarg); break;
        case 'L': learning_rate = atof(optarg); break;
        case 'C': checksum_freq = atoi(optarg); break;
        case 's': stats_freq = atoi(optarg); break;
        case 'f': first_lb = atoi(optarg); break;
        case 'b': lb_freq = atoi(optarg); break;
        case 'l': lb_wait_lag = atoi(optarg); break;
        case 'S': moe_seed = atol(optarg); break;
        case 'a': async_lb = 1; break;
        case 'A': use_adam = 1; break;
        case 'T': use_tf32 = 1; break;
        case 'I': use_instrument = 1; break;
        case 'U': fuse_sources = 0; break;
        case 'P': print_place = 1; break;
        default:
          CkPrintf(
              "Usage: %s -e [experts] -m [d_model] -h [d_ff] -t [tokens per PE per step]\n"
              "  -k [top-k] -i [steps] -u [warmup steps]\n"
              "  -z [zipf exponent of expert popularity, 0 = uniform]\n"
              "  -p [drift period: steps between reshuffles of the hot set, 0 = static]\n"
              "  -c [chunk: tokens per GEMM] -r [capacity headroom, 0 = auto from zipf]\n"
              "  -n [compute lanes per PE: streams experts are spread over]\n"
              "  -L [learning rate] -A (Adam: 3x payload) -T (TF32 GEMMs)\n"
              "  -U (one chunked pass per source instead of one per expert: many more\n"
              "      weight updates, ~12%% slower, kept for comparison. Either way the\n"
              "      chunk boundaries follow the routing alone, so the checksums are\n"
              "      placement-independent; the two modes' values differ.)\n"
              "  -C [checksum frequency, 0 disables] -s [stats frequency, 0 disables]\n"
              "  -f [first LB step] -b [LB frequency]\n"
              "  -a (async LB: overlap the step; needs +LBAsync) -l [wait lag]\n"
              "  -I (CUPTI-measured loads in a window before each LB step, instead of\n"
              "      the experts' token-count estimates)\n"
              "  -S [seed] -P (print final placement)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (top_k < 1 || top_k > MOE_MAX_TOPK || top_k > n_experts)
      CkAbort("-k must be in [1, min(%d, experts)]\n", MOE_MAX_TOPK);
    if (chunk_tokens < 1) CkAbort("-c must be positive\n");
    if (n_lanes < 1 || n_lanes > MOE_MAX_LANES)
      CkAbort("-n must be in [1, %d]\n", MOE_MAX_LANES);
    if (async_lb && !_lb_args.lbAsync()) {
      CkPrintf("[WARN] -a ignored: async LB needs +LBAsync on the command "
               "line. Running the unsplit AtSync barrier instead.\n");
      async_lb = 0;
    }
    if (async_lb && (lb_wait_lag < 1 || lb_wait_lag >= lb_freq)) {
      CkAbort("-l (%d) must be in [1, lb_freq): the wait must land before "
              "the next AtSyncStart\n", lb_wait_lag);
    }
    n_disp = CkNumPes();

    // Per-(source, expert) slab capacity. The hottest expert draws a share
    // p_max of every dispatcher's tokens; auto sizes for 1.5x that plus slack.
    double p_max;
    if (zipf_z > 0.0) {
      double H = 0.0;
      for (int r = 0; r < n_experts; r++) H += pow((double)(r + 1), -zipf_z);
      p_max = 1.0 / H;
    } else {
      p_max = 1.0 / n_experts;
    }
    const double avg_src = (double)top_k * n_tokens / n_experts;
    double cap = (headroom > 0.0) ? headroom * avg_src
                                  : 1.5 * p_max * top_k * n_tokens + 64.0;
    cap = std::min(cap, (double)top_k * n_tokens);
    cap_src = std::max(64, (int)((ceil(cap) + 31.0) / 32.0) * 32);

    const size_t wbytes = sizeof(float) * (size_t)d_model * d_ff * 2 *
                          (use_adam ? 3 : 1);
    const size_t slab_bytes = sizeof(float) * (size_t)n_disp * cap_src *
                              d_model * 2;
    const double gflop = 10.0 * avg_src * n_disp * d_model * d_ff * 1e-9;

    CkPrintf("\n[CUDA mixture-of-experts layer]\n");
    CkPrintf("Experts: %d, d_model %d, d_ff %d, %s%s\n", n_experts, d_model,
        d_ff, use_adam ? "Adam" : "SGD", use_tf32 ? ", TF32" : "");
    CkPrintf("Tokens: %d per PE per step on %d PEs, top-%d, chunk %d, "
        "slab capacity %d per (PE, expert), %d compute lanes per PE%s\n",
        n_tokens, n_disp, top_k, chunk_tokens, cap_src, n_lanes,
        fuse_sources ? "" : ", one pass per source");
    CkPrintf("Routing: zipf %.2f (hottest expert %.1fx average), hot set "
        "reshuffled every %d steps, seed %ld\n", zipf_z, p_max * n_experts,
        drift_period, moe_seed);
    CkPrintf("Per expert: %.1f MB migratable state, %.1f MB slabs, "
        "%.1f GFLOP per step at average load\n", wbytes / 1048576.0,
        slab_bytes / 1048576.0, gflop);
    CkPrintf("Steps: %d (+%d warmup), first LB: %d, LB frequency: %d%s, "
        "loads: %s\n\n", n_steps, warmup_steps, first_lb, lb_freq,
        async_lb ? ", async" : "",
        use_instrument ? "CUPTI-measured" : "token-count estimates");

    disp_proxy = CProxy_Dispatcher::ckNew();
    expert_proxy = CProxy_Expert::ckNew(n_experts);
    init_start_time = CkWallTimer();
    disp_proxy.init();
  }

  void dispReady() { expert_proxy.init(); }

  void expertsReady() {
    CkPrintf("Init time: %.3lf s\n", CkWallTimer() - init_start_time);
    start_time = CkWallTimer();
    window_start_time = start_time;
    disp_proxy.runStep();
    expert_proxy.runStep();
  }

  void stepStatsDisp(CkReductionMsg* msg) {
    const double* outpe = (const double*)msg->getData();
    double out2 = 0.0;
    for (int p = 0; p < n_disp; p++) out2 += outpe[p];
    delete msg;
    disp_step++;
    const double now = CkWallTimer();
    if (disp_step == warmup_steps) {
      start_time = now;
      window_start_time = now;
    }
    DispRec r;
    r.out2 = out2;
    r.t = now;
    disp_q.push_back(r);
    tryPrint();
  }

  void stepStatsExpert(CkReductionMsg* msg) {
    CkReduction::tupleElement* results = nullptr;
    int num_elems = 0;
    msg->toTuple(&results, &num_elems);
    const long sum_tok = *(long*)results[0].data;
    const long max_tok = *(long*)results[1].data;
    const double* w2e = (const double*)results[2].data;
    double w2 = 0.0;
    for (int e = 0; e < n_experts; e++) w2 += w2e[e];
    const long* tokpe = (const long*)results[3].data;
    const double* gpupe = (const double*)results[4].data;
    exp_step++;

    const long expect = (long)top_k * n_tokens * n_disp;
    if (sum_tok != expect)
      CkAbort("Token count not conserved at step %d: experts saw %ld of %ld\n",
          exp_step, sum_tok, expect);

    long max_pe = 0;
    double gsum = 0.0, gmax = 0.0;
    for (int p = 0; p < n_disp; p++) {
      max_pe = std::max(max_pe, tokpe[p]);
      gsum += gpupe[p];
      gmax = std::max(gmax, gpupe[p]);
    }
    ExpRec r;
    r.tokratio = (double)max_pe / ((double)sum_tok / n_disp);
    r.gpuratio = gsum > 0.0 ? gmax / (gsum / n_disp) : 0.0;
    r.gmax_ms = gmax * 1e3;
    r.expratio = (double)max_tok / ((double)sum_tok / n_experts);
    r.w2 = w2;
    exp_q.push_back(r);
    delete msg;
    delete [] results;
    tryPrint();
  }

  void tryPrint() {
    while (!disp_q.empty() && !exp_q.empty()) {
      const DispRec d = disp_q.front();
      const ExpRec e = exp_q.front();
      disp_q.pop_front();
      exp_q.pop_front();
      // The step both records describe: as many as have been popped so far.
      static int step = 0;
      step++;
      if (stats_freq > 0 && step > warmup_steps &&
          (step - warmup_steps) % stats_freq == 0) {
        const double ms = (d.t - window_start_time) / stats_freq * 1e3;
        window_start_time = d.t;
        // %.15e on the checksums: a placement-to-placement comparison needs
        // every digit.
        if (checksum_freq > 0 && step % checksum_freq == 0)
          CkPrintf("Step %d: %.3lf ms/step, tokens/PE max/avg %.2lf, "
              "gpu/PE max %.1lf ms max/avg %.2lf (prev), expert max/avg %.2lf, "
              "out2 %.15e, w2 %.15e\n", step, ms, e.tokratio, e.gmax_ms,
              e.gpuratio, e.expratio, d.out2, e.w2);
        else
          CkPrintf("Step %d: %.3lf ms/step, tokens/PE max/avg %.2lf, "
              "gpu/PE max %.1lf ms max/avg %.2lf (prev), expert max/avg "
              "%.2lf\n", step, ms, e.tokratio, e.gmax_ms, e.gpuratio,
              e.expratio);
      }
    }
  }

  void dispDone() { finish(); }
  void expertsDone() { finish(); }

  void finish() {
    if (++done_count < 2) return;
    const double total_time = CkWallTimer() - start_time;
    CkPrintf("\nTotal time: %.3lf s\nAverage step time: %.3lf ms\n",
        total_time, total_time / n_steps * 1e3);
    CkExit();
  }
};

// ---------------------------------------------------------------------------

class Dispatcher : public CBase_Dispatcher {
  Dispatcher_SDAG_CODE

 public:
  MoeCtx gpu;
  int my_step = 0;
  int recv_count = 0;
  int outstanding_sends = 0;
  bool drain_pending = false;
  bool chk_due = false;
  int cur_phase = -1;
  std::vector<int> perm;        // expert at each popularity rank, this phase
  std::vector<double> cdf;      // cumulative popularity by rank
  std::vector<int> counts;      // tokens per expert, this step
  std::vector<int> offs;        // slab offsets per expert (n_experts + 1)
  std::vector<int> fill_;
  std::vector<int> slot_expert; // expert chosen for each (token, j)

  float* d_x;         // [n_tokens x d_model], fixed for the run
  float* d_send;      // [(n_tokens*k + 1) x d_model], tokens sorted by expert
  float* d_recv;      // same layout, expert outputs; +1 row of slack for n = 0
  float* d_y;         // [n_tokens x d_model], combined outputs
  int* d_send_idx;    // token for each slot of d_send
  int* d_recv_pos;    // slot of d_recv for each (token, j)
  int* h_send_idx;    // pinned copies
  int* h_recv_pos;
  double* d_chk;
  double* h_chk;
  // A send is tagged with a stream, and the runtime records "data ready" as
  // that stream's tail at send time. The comm stream's tail may be a landing
  // copy waiting on a remote PE, so the sends carry a stream of their own that
  // waits only on the event recorded right after the gather.
  cudaStream_t send_stream;
  cudaEvent_t ev_gather;

  Dispatcher() {}

  size_t slots() const { return (size_t)n_tokens * top_k; }

  void init() {
    // The experts declare their own loads (token count x measured seconds per
    // token); with instrumentation on, the balancer would overwrite them with
    // CUPTI's measurement, and pay CUPTI's cost every step. -I keeps it.
    if (!use_instrument) LBTurnInstrumentOff();

    memset(&gpu, 0, sizeof(gpu));
    gpu.n_lanes = n_lanes;
    // See MoeCtx in moe.h for why communication has its own stream.
    hapiCheck(cudaStreamCreateWithPriority(&gpu.comm.stream,
        cudaStreamNonBlocking, -1));
    hapiCheck(mmMalloc((void**)&gpu.comm.partials,
        sizeof(double) * MOE_RED_BLOCKS));
    hapiCheck(cudaStreamCreateWithPriority(&send_stream, cudaStreamNonBlocking,
        -1));
    hapiCheck(cudaEventCreateWithFlags(&ev_gather, cudaEventDisableTiming));
    const size_t wsz = (size_t)d_model * d_ff;
    for (int l = 0; l < n_lanes; l++) {
      MoeGpu& g = gpu.lanes[l];
      hapiCheck(cudaStreamCreateWithPriority(&g.stream, cudaStreamNonBlocking,
          0));
      hapiCheck(mmMalloc((void**)&g.h,
          sizeof(float) * (size_t)chunk_tokens * d_ff));
      hapiCheck(mmMalloc((void**)&g.dh,
          sizeof(float) * (size_t)chunk_tokens * d_ff));
      if (use_adam) {
        hapiCheck(mmMalloc((void**)&g.dW1, sizeof(float) * wsz));
        hapiCheck(mmMalloc((void**)&g.dW2, sizeof(float) * wsz));
      }
      if (fuse_sources) {
        // Bounded by what one expert can be sent in a step, the same bound its
        // own slabs use. Sized once: growing it would have to free a block the
        // lane's queued kernels may still be reading.
        const size_t frows = (size_t)n_disp * cap_src;
        hapiCheck(mmMalloc((void**)&g.fx, sizeof(float) * frows * d_model));
        hapiCheck(mmMalloc((void**)&g.fy, sizeof(float) * frows * d_model));
      }
      g.workspace_bytes = (size_t)32 << 20;
      hapiCheck(mmMalloc(&g.workspace, g.workspace_bytes));
      hapiCheck(mmMalloc((void**)&g.partials,
          sizeof(double) * MOE_RED_BLOCKS));
      if (moeBlasCreate(&g, use_tf32 != 0))
        CkAbort("cublasCreate failed on PE %d lane %d\n", CkMyPe(), l);
    }

    const size_t TK = slots();
    hapiCheck(mmMalloc((void**)&d_x, sizeof(float) * n_tokens * d_model));
    hapiCheck(mmMalloc((void**)&d_send, sizeof(float) * (TK + 1) * d_model));
    hapiCheck(mmMalloc((void**)&d_recv, sizeof(float) * (TK + 1) * d_model));
    hapiCheck(mmMalloc((void**)&d_y, sizeof(float) * n_tokens * d_model));
    hapiCheck(mmMalloc((void**)&d_send_idx, sizeof(int) * TK));
    hapiCheck(mmMalloc((void**)&d_recv_pos, sizeof(int) * TK));
    hapiCheck(mmMalloc((void**)&d_chk, sizeof(double)));
    hapiCheck(hapiMallocHost((void**)&h_send_idx, sizeof(int) * TK));
    hapiCheck(hapiMallocHost((void**)&h_recv_pos, sizeof(int) * TK));
    hapiCheck(hapiMallocHost((void**)&h_chk, sizeof(double)));

    counts.assign(n_experts, 0);
    offs.assign(n_experts + 1, 0);
    fill_.assign(n_experts, 0);
    slot_expert.assign(TK, 0);

    // Unit-variance inputs, a pure function of (seed, PE, element)
    moeInitUniform(d_x, (size_t)n_tokens * d_model,
        hmix(hmix(moe_seed, 1000), CkMyPe()), sqrtf(3.0f), gpu.comm.stream);
    hapiCheck(cudaStreamSynchronize(gpu.comm.stream));
    contribute(CkCallback(CkReductionTarget(Main, dispReady), main_proxy));
  }

  // The hot set of a phase: a hashed permutation of the experts, with rank r
  // drawing a share proportional to (r+1)^-z.
  void buildPhase(int phase) {
    perm.resize(n_experts);
    for (int i = 0; i < n_experts; i++) perm[i] = i;
    for (int i = n_experts - 1; i > 0; i--) {
      const uint64_t h = hmix(hmix(moe_seed, 77), (uint64_t)phase * n_experts + i);
      const int j = (int)(h % (uint64_t)(i + 1));
      std::swap(perm[i], perm[j]);
    }
    cdf.resize(n_experts);
    double s = 0.0;
    for (int r = 0; r < n_experts; r++) {
      s += (zipf_z > 0.0) ? pow((double)(r + 1), -zipf_z) : 1.0;
      cdf[r] = s;
    }
    for (int r = 0; r < n_experts; r++) cdf[r] /= s;
  }

  // Route this step's tokens: counts per expert, slab offsets, and the two
  // index maps the gather and combine kernels use. Tokens land in an expert's
  // slab in (token, j) order, which is what makes the expert's reduction order
  // independent of anything but the routing.
  void route() {
    const int E = n_experts, T = n_tokens, K = top_k;
    const int phase = drift_period > 0 ? (my_step - 1) / drift_period : 0;
    if (phase != cur_phase) {
      cur_phase = phase;
      buildPhase(phase);
    }
    std::fill(counts.begin(), counts.end(), 0);
    for (int t = 0; t < T; t++) {
      int chosen[MOE_MAX_TOPK];
      for (int j = 0; j < K; j++) {
        int e = -1;
        for (int attempt = 0; attempt < 16 && e < 0; attempt++) {
          const uint64_t h = hmix(hmix(hmix(hmix(hmix(moe_seed, my_step),
              CkMyPe()), t), j), attempt);
          int r = (int)(std::upper_bound(cdf.begin(), cdf.end(), u01(h)) -
                        cdf.begin());
          if (r >= E) r = E - 1;
          const int cand = perm[r];
          bool dup = false;
          for (int q = 0; q < j; q++) if (chosen[q] == cand) dup = true;
          if (!dup) e = cand;
        }
        if (e < 0) {
          // Sixteen collisions: walk to the next expert not yet chosen.
          e = (chosen[j - 1] + 1) % E;
          for (;;) {
            bool dup = false;
            for (int q = 0; q < j; q++) if (chosen[q] == e) dup = true;
            if (!dup) break;
            e = (e + 1) % E;
          }
        }
        chosen[j] = e;
        slot_expert[(size_t)t * K + j] = e;
        counts[e]++;
      }
    }
    offs[0] = 0;
    for (int e = 0; e < E; e++) {
      offs[e + 1] = offs[e] + counts[e];
      if (counts[e] > cap_src)
        CkAbort("PE %d step %d: %d tokens for expert %d exceed the slab "
                "capacity %d; increase -r\n", CkMyPe(), my_step, counts[e], e,
                cap_src);
    }
    std::fill(fill_.begin(), fill_.end(), 0);
    const size_t TK = slots();
    for (size_t i = 0; i < TK; i++) {
      const int e = slot_expert[i];
      const int pos = offs[e] + fill_[e]++;
      h_send_idx[pos] = (int)(i / K);
      h_recv_pos[i] = pos;
    }
  }

  void beginStep() {
    my_step++;
    recv_count = 0;
    TRACE(1, "disp step %d begin", my_step);
    route();
    const size_t TK = slots();
    // The gather depends on nothing that lands on the comm stream (x is
    // fixed, the send slab is quiet once the previous step's sends drained),
    // so it runs on the send stream: the sends are then tagged with the very
    // stream that produced their data, and the gather never queues behind a
    // landing. d_recv_pos is read by the combine on the comm stream, which
    // runs only after every output of this step has landed there.
    hapiCheck(hapiMemcpyAsync(d_send_idx, h_send_idx, sizeof(int) * TK,
        cudaMemcpyHostToDevice, send_stream));
    hapiCheck(hapiMemcpyAsync(d_recv_pos, h_recv_pos, sizeof(int) * TK,
        cudaMemcpyHostToDevice, send_stream));
    moeGather(d_x, d_send_idx, d_send, (int)TK, d_model, send_stream);
    hapiCheck(cudaEventRecord(ev_gather, send_stream));
    TRACE(2, "disp step %d gather issued", my_step);
    CkCallback* cb = new CkCallback(CkIndex_Dispatcher::gathered(),
        thisProxy[CkMyPe()]);
    hapiAddCallback(send_stream, cb);
  }

  // One slab per expert, empty ones included (the expert's step waits for one
  // message per dispatcher). len = max(n,1) rows so an empty message still
  // carries a valid buffer; the expert lands it in the unused slab.
  void sendTokens() {
    const double ts0 = tnow();
    hapiCheck(cudaStreamWaitEvent(send_stream, ev_gather, 0));
    for (int e = 0; e < n_experts; e++) {
      const int n = counts[e];
      const int len = std::max(n, 1) * d_model;
      expert_proxy[e].receiveTokens(my_step, CkMyPe(), n, len,
          (outstanding_sends++,
           CkDeviceBuffer(d_send + (size_t)offs[e] * d_model,
               CkCallback(CkIndex_Dispatcher::sendDone(), thisProxy[CkMyPe()]),
               send_stream)));
    }
    TRACE(1, "disp step %d tokens sent", my_step);
    TRACE(2, "disp step %d issued %d sends in %.1f ms", my_step, n_experts,
        (tnow() - ts0) * 1e3);
  }

  // Post entry method: an expert's outputs land in its segment of the receive
  // slab. An empty message lands in the slack row past the end.
  void receiveOutput(int ref, int e, int n, int& len, float*& buf,
      CkDeviceBufferPost* devicePost) {
    if (ref != my_step)
      CkAbort("PE %d: outputs for step %d arrived during step %d\n", CkMyPe(),
          ref, my_step);
    buf = (n > 0) ? d_recv + (size_t)offs[e] * d_model
                  : d_recv + slots() * d_model;
    devicePost[0].hapi_stream = gpu.comm.stream;
    TRACE(3, "disp step %d output expert %d n=%d posted", ref, e, n);
  }

  void noteOutput(int e, int n) {
    TRACE(3, "disp step %d output expert %d n=%d consumed", my_step, e, n);
    if (n != counts[e])
      CkAbort("PE %d step %d: expert %d returned %d rows for %d tokens\n",
          CkMyPe(), my_step, e, n, counts[e]);
  }

  void combine() {
    TRACE(1, "disp step %d outputs in", my_step);
    moeCombine(d_recv, d_recv_pos, d_y, n_tokens, top_k, d_model,
        1.0f / top_k, gpu.comm.stream);
    chk_due = checksum_freq > 0 && (my_step % checksum_freq == 0);
    if (chk_due) {
      hapiCheck(cudaMemsetAsync(d_chk, 0, sizeof(double), gpu.comm.stream));
      moeSumSqAdd(d_y, (size_t)n_tokens * d_model, &gpu.comm, d_chk, gpu.comm.stream);
      hapiCheck(hapiMemcpyAsync(h_chk, d_chk, sizeof(double),
          cudaMemcpyDeviceToHost, gpu.comm.stream));
    }
    CkCallback* cb = new CkCallback(CkIndex_Dispatcher::combined(),
        thisProxy[CkMyPe()]);
    hapiAddCallback(gpu.comm.stream, cb);
  }

  // The next step's gather rewrites the send slab; every send must have
  // confirmed completion first.
  void gateDrain() {
    if (outstanding_sends == 0) thisProxy[CkMyPe()].sendsDrained();
    else drain_pending = true;
  }

  void sendDone() {
    outstanding_sends--;
    if (drain_pending && outstanding_sends == 0) {
      drain_pending = false;
      thisProxy[CkMyPe()].sendsDrained();
    }
  }

  void endStep() {
    TRACE(1, "disp step %d end", my_step);
    // One slot per PE, summed elementwise: exact whatever order the reduction
    // combines contributions in, so the checksum is the same bits on every
    // run. Main adds the slots in PE order. (A plain sum_double of the values
    // differed in the last digit from run to run, with zero migrations.)
    std::vector<double> outpe(n_disp, 0.0);
    outpe[CkMyPe()] = chk_due ? *h_chk : 0.0;
    contribute(sizeof(double) * n_disp, outpe.data(), CkReduction::sum_double,
        CkCallback(CkIndex_Main::stepStatsDisp(NULL), main_proxy));
    if (my_step < warmup_steps + n_steps) {
      thisProxy[CkMyPe()].runStep();
    } else {
      contribute(CkCallback(CkReductionTarget(Main, dispDone), main_proxy));
    }
  }
};

// ---------------------------------------------------------------------------

class Expert : public CBase_Expert {
  Expert_SDAG_CODE

 public:
  int my_step = 0;
  int recv_count = 0;
  // 0: receiving slabs (x segments live); 1: kernels issued (y segments
  // live); 2: outputs sent (nothing live but the weights)
  int phase = 0;
  std::vector<int> n_src;   // rows received from each dispatcher, -1 = none
  int n_tot = 0;
  int outstanding_sends = 0;
  bool drain_pending = false;
  bool chk_due = false;

  // Async LB state: the step was joined at lb_start_step and AtSyncWait is
  // owed lb_wait_lag steps later.
  bool lb_waiting = false;
  int lb_start_step = 0;
  bool instrumenting = true;   // matches the runtime default at startup

  // Load estimate: seconds per token, an EMA of the measured step time over
  // its token count. Rate-aware for free: a slower device shows up here.
  double spt = 0.0;
  double gpu_last = 0.0;   // measured device time of the last sampled step
  int n_last = 0;
  int step_last = 0;       // the step those events were recorded in
  int adam_t = 0;
  bool ev_pending = false;

  float* W1;   // [d_model x d_ff]
  float* W2;   // [d_ff x d_model]
  float* m1;   // Adam moments, -A only
  float* v1;
  float* m2;
  float* v2;
  float* x_seg;   // [n_disp x cap_src x d_model], one segment per dispatcher
  float* y_seg;   // same layout, outputs
  double* d_chk;
  double* h_chk;
  cudaEvent_t ev_start;
  cudaEvent_t ev_end;
  // Recorded after the checksum's device-to-host copy. A pack reads h_chk on
  // the host, which the copy may still be writing; pup_device_order only
  // orders DEVICE-mode copies and leaves that host read racing. Losing it
  // costs one expert's slot in the w2 reduction, seen once in eight async
  // runs at Llama size as a deficit of exactly 1/64.
  cudaEvent_t ev_chk;
  // Same reason as the dispatcher's: the lane's tail at send time includes
  // other experts' queued GEMMs, and a landing that waited on it stalled the
  // receiver's comm stream for a remote expert's compute. This stream waits
  // only on ev_end, recorded after this expert's own kernels.
  cudaStream_t send_stream;

  Expert() {
    usesAtSync = true;
    n_src.assign(n_disp, -1);
    W1 = W2 = m1 = v1 = m2 = v2 = x_seg = y_seg = NULL;
    d_chk = NULL;
    h_chk = NULL;
  }

  Expert(CkMigrateMessage* m) {
    usesAtSync = true;
    W1 = W2 = m1 = v1 = m2 = v2 = x_seg = y_seg = NULL;
    d_chk = NULL;
    h_chk = NULL;
  }

  ~Expert() {
    // Under async LB this runs mid-step on a migrating element: kernels may
    // still be in flight on the shared stream. Settle them before freeing.
    const double t0 = tnow();
    if (hostSyncMode() || !CkDevicePoolOn()) {
      hapiCheck(cudaStreamSynchronize(lane()->stream));
      TRACE(1, "expert %d dtor step %d phase %d synced after %.3f ms",
          thisIndex, my_step, phase, (tnow() - t0) * 1e3);
    } else {
      // Stream-ordered frees instead of a host wait: each block is noted as
      // read on this stream, and the pool parks it until the work queued
      // here retires. The pack copies' own note (migration stream) stays
      // alongside; the pool honours every stream a block was noted on.
      float* bufs[] = {W1, W2, m1, v1, m2, v2, x_seg, y_seg};
      for (float* b : bufs)
        if (b) hapiDevPoolNoteRead(b, lane()->stream);
      if (d_chk) hapiDevPoolNoteRead(d_chk, lane()->stream);
      TRACE(1, "expert %d dtor step %d phase %d stream-ordered", thisIndex,
          my_step, phase);
    }
    hapiCheck(cudaEventDestroy(ev_start));
    hapiCheck(cudaEventDestroy(ev_end));
    hapiCheck(cudaEventDestroy(ev_chk));
    hapiCheck(cudaStreamDestroy(send_stream));
    if (outstanding_sends != 0) {
      // A transport may still be reading a send buffer. Leak rather than pull
      // memory out from under it -- the pic2d convention.
      if (getenv("CHARM_DEBUG_MIGRATE"))
        CkPrintf("[%d] expert %d leaking device buffers at destruction: %d "
                 "sends outstanding\n", CkMyPe(), thisIndex, outstanding_sends);
      return;
    }
    hapiCheck(mmFree(W1));
    hapiCheck(mmFree(W2));
    hapiCheck(mmFree(m1));
    hapiCheck(mmFree(v1));
    hapiCheck(mmFree(m2));
    hapiCheck(mmFree(v2));
    hapiCheck(mmFree(x_seg));
    hapiCheck(mmFree(y_seg));
    hapiCheck(mmFree(d_chk));
    hapiCheck(hapiFreeHost(h_chk));
  }

  // Never cached: a different object on every PE this expert lands on.
  MoeCtx* ctx() { return &disp_proxy.ckLocalBranch()->gpu; }
  MoeGpu* lane() { return &ctx()->lanes[thisIndex % n_lanes]; }
  cudaStream_t commStream() { return ctx()->comm.stream; }
  size_t segStride() const { return (size_t)cap_src * d_model; }
  size_t wsz() const { return (size_t)d_model * d_ff; }

  void allocate() {
    const size_t seg = segStride() * n_disp;
    hapiCheck(mmMalloc((void**)&W1, sizeof(float) * wsz()));
    hapiCheck(mmMalloc((void**)&W2, sizeof(float) * wsz()));
    if (use_adam) {
      hapiCheck(mmMalloc((void**)&m1, sizeof(float) * wsz()));
      hapiCheck(mmMalloc((void**)&v1, sizeof(float) * wsz()));
      hapiCheck(mmMalloc((void**)&m2, sizeof(float) * wsz()));
      hapiCheck(mmMalloc((void**)&v2, sizeof(float) * wsz()));
    }
    hapiCheck(mmMalloc((void**)&x_seg, sizeof(float) * seg));
    hapiCheck(mmMalloc((void**)&y_seg, sizeof(float) * seg));
    hapiCheck(mmMalloc((void**)&d_chk, sizeof(double)));
    hapiCheck(hapiMallocHost((void**)&h_chk, sizeof(double)));
  }

  void createEvents() {
    hapiCheck(cudaEventCreate(&ev_start));
    hapiCheck(cudaEventCreate(&ev_end));
    hapiCheck(cudaEventCreateWithFlags(&ev_chk, cudaEventDisableTiming));
    hapiCheck(cudaStreamCreateWithPriority(&send_stream, cudaStreamNonBlocking,
        -1));
    ev_pending = false;
  }

  void init() {
    allocate();
    createEvents();
    MoeGpu* g = lane();
    moeInitUniform(W1, wsz(), hmix(hmix(moe_seed, 1), thisIndex),
        1.0f / sqrtf((float)d_model), g->stream);
    moeInitUniform(W2, wsz(), hmix(hmix(moe_seed, 2), thisIndex),
        1.0f / sqrtf((float)d_ff), g->stream);
    if (use_adam) {
      hapiCheck(cudaMemsetAsync(m1, 0, sizeof(float) * wsz(), g->stream));
      hapiCheck(cudaMemsetAsync(v1, 0, sizeof(float) * wsz(), g->stream));
      hapiCheck(cudaMemsetAsync(m2, 0, sizeof(float) * wsz(), g->stream));
      hapiCheck(cudaMemsetAsync(v2, 0, sizeof(float) * wsz(), g->stream));
    }
    CkCallback* cb = new CkCallback(CkIndex_Expert::initDone(),
        thisProxy[thisIndex]);
    hapiAddCallback(g->stream, cb);
  }

  void initDone() {
    contribute(CkCallback(CkReductionTarget(Main, expertsReady), main_proxy));
  }

  void pup(PUP::er& p) {
    // The migration copies weights and slabs off the device on a stream of
    // its own; settle this PE's stream first so nothing half-written travels
    // and no in-flight kernel loses its buffers to the frees that follow.
    const double t0 = tnow();
    if (p.isPacking()) {
      if (hostSyncMode()) {
        // The original way: stop the host until every kernel queued on the
        // shared stream -- other experts' too -- has finished, so the pack
        // copies read final weights. Up to a full step of GEMMs, host-blocked.
        hapiCheck(cudaStreamSynchronize(lane()->stream));
        TRACE(1, "expert %d pack step %d phase %d synced after %.3f ms",
            thisIndex, my_step, phase, (tnow() - t0) * 1e3);
      } else {
        // Order the migration stream's copies behind this stream's queued
        // work on the device instead; the host goes on. Safe because the
        // runtime notes every pool source as read on the migration stream,
        // so the destructor's frees park until the copies retire.
        p.pup_device_order((void*)lane()->stream);
      }
    }
    p | my_step;
    p | recv_count;
    p | phase;
    p | n_src;
    p | n_tot;
    // SDAG state: which whens are outstanding, buffered ordinary messages.
    _sdag_pup(p);
    p | outstanding_sends;
    p | drain_pending;
    p | chk_due;
    p | lb_waiting;
    p | lb_start_step;
    p | instrumenting;
    p | spt;
    p | gpu_last;
    p | n_last;
    p | step_last;
    p | adam_t;

    if (p.isUnpacking()) {
      allocate();
      createEvents();   // a pending timing sample is lost with the old events
    }
    // The checksum's copy into h_chk is asynchronous; wait for it before the
    // host reads it, or a migration here packs whatever was there before.
    if (p.isPacking() && h_chk && chk_due)
      hapiCheck(cudaEventSynchronize(ev_chk));
    double chk = (h_chk && chk_due) ? *h_chk : 0.0;
    p | chk;
    if (p.isUnpacking()) *h_chk = chk;

    p(W1, wsz(), PUP::PUPMode::DEVICE);
    p(W2, wsz(), PUP::PUPMode::DEVICE);
    if (use_adam) {
      p(m1, wsz(), PUP::PUPMode::DEVICE);
      p(v1, wsz(), PUP::PUPMode::DEVICE);
      p(m2, wsz(), PUP::PUPMode::DEVICE);
      p(v2, wsz(), PUP::PUPMode::DEVICE);
    }
    // What else is live depends on where in the step the move happens. The
    // runtime holds a migration until every device receive addressed to this
    // element has been consumed and every send it issued has completed, so:
    //   phase 0: the slabs consumed so far, which compute will read;
    //   phase 1: the outputs the kernels (now complete) produced, which
    //            sendOutputs will send from the destination;
    //   phase 2: nothing -- the outputs have gone.
    if (phase == 0) {
      for (int s = 0; s < n_disp; s++)
        if (n_src[s] > 0)
          p(x_seg + s * segStride(), (size_t)n_src[s] * d_model,
              PUP::PUPMode::DEVICE);
    } else if (phase == 1) {
      for (int s = 0; s < n_disp; s++)
        if (n_src[s] > 0)
          p(y_seg + s * segStride(), (size_t)n_src[s] * d_model,
              PUP::PUPMode::DEVICE);
    }
    if (p.isPacking())
      TRACE(1, "expert %d pack end %.3f ms", thisIndex, (tnow() - t0) * 1e3);
    else if (p.isUnpacking())
      TRACE(1, "expert %d unpack step %d phase %d %.3f ms", thisIndex, my_step,
          phase, (tnow() - t0) * 1e3);
  }

  void beginStep() {
    my_step++;
    recv_count = 0;
    phase = 0;
    n_tot = 0;
    std::fill(n_src.begin(), n_src.end(), -1);
  }

  // Post entry method: a dispatcher's slab lands in its own segment. An empty
  // message still lands one row there, which nothing reads.
  void receiveTokens(int ref, int src, int n, int& len, float*& buf,
      CkDeviceBufferPost* devicePost) {
    if (ref < my_step || ref > my_step + 1)
      CkAbort("Expert %d on PE %d: slab for step %d arrived during step %d\n",
          thisIndex, CkMyPe(), ref, my_step);
    buf = x_seg + (size_t)src * segStride();
    devicePost[0].hapi_stream = commStream();
    TRACE(3, "expert %d step %d slab src %d n=%d posted", thisIndex, ref, src,
        n);
  }

  void noteTokens(int src, int n) {
    n_src[src] = n;
    TRACE(3, "expert %d step %d slab src %d n=%d consumed", thisIndex, my_step,
        src, n);
  }

  // The load-balancing join point; see runStep in moe.ci. Every path ends in
  // exactly one resumed() for this step: sent here directly, or from
  // ResumeFromSync when the runtime releases the element.
  void lbPoint() {
    // Declare the load this step will cost before joining: every slab is in,
    // so the token count is exact, and the rate comes from the steps before.
    sampleTiming();
    n_tot = 0;
    for (int s = 0; s < n_disp; s++) n_tot += std::max(n_src[s], 0);
    // The PE's pooled figure, so an expert that has not run yet (just
    // migrated, or idle last step) still declares a load on the same scale.
    const MoeCtx* c = ctx();
    double s_tok = (c->spt_tok > 0.0) ? c->spt_gpu / c->spt_tok : spt;
    // Diagnostic: a fixed per-token cost, identical on every PE. Every expert
    // here does the same work per token, so this is the exact load ratio; it
    // separates a balancer that cannot use a good signal from a bad signal.
    static const bool flat = getenv("CHARM_MOE_FLATLOAD") != NULL;
    if (flat) s_tok = 1e-5;
    setObjGPUTime(s_tok * n_tot);

    if (use_instrument) {
      const bool want = (nextLBStep(my_step) - my_step) <= LB_INSTRUMENT_WINDOW;
      if (want != instrumenting) {
        instrumenting = want;
        if (want) LBTurnInstrumentOn(); else LBTurnInstrumentOff();
      }
    }
    if (isLBStep(my_step) && !lb_waiting) {
      if (!async_lb) {
        // Stops here; ResumeFromSync means the strategy and the migrations
        // are over.
        TRACE(1, "expert %d step %d atsync", thisIndex, my_step);
        AtSync();
        return;
      }
      // Split barrier: join the step and keep computing while the strategy
      // runs and other experts migrate.
      TRACE(1, "expert %d step %d atsyncstart", thisIndex, my_step);
      AtSyncStart();
      lb_waiting = true;
      lb_start_step = my_step;
      thisProxy[thisIndex].resumed();
      return;
    }
    if (lb_waiting && my_step >= lb_start_step + lb_wait_lag) {
      // The other half of the split: park until the step is collected.
      // Resumes inline if it already is.
      lb_waiting = false;
      TRACE(1, "expert %d step %d atsyncwait", thisIndex, my_step);
      AtSyncWait();
      return;
    }
    thisProxy[thisIndex].resumed();
  }

  void ResumeFromSync() {
    TRACE(1, "expert %d step %d resume", thisIndex, my_step);
    thisProxy[thisIndex].resumed();
  }

  // Fold the last recorded step's device time into the per-token rate. Its
  // kernels are complete whenever this runs after that step's outputs were
  // sent. Step 1 is skipped: its time is cuBLAS loading its kernels, and one
  // such sample made every expert's first estimate the same 90 ms whatever
  // its token count.
  void sampleTiming() {
    if (!ev_pending || cudaEventQuery(ev_end) != cudaSuccess) return;
    float ms = 0.0f;
    hapiCheck(cudaEventElapsedTime(&ms, ev_start, ev_end));
    ev_pending = false;
    gpu_last = ms * 1e-3;
    if (n_last > 0 && step_last > 1) {
      // Pool the sample into the PE's figure rather than keeping a per-expert
      // one; see MoeCtx. The decay makes it track without a window.
      MoeCtx* c = ctx();
      c->spt_gpu = 0.75 * c->spt_gpu + gpu_last;
      c->spt_tok = 0.75 * c->spt_tok + n_last;
      spt = c->spt_gpu / c->spt_tok;
    }
  }

  void compute() {
    phase = 1;
    MoeGpu* g = lane();
    n_tot = 0;
    for (int s = 0; s < n_disp; s++) n_tot += std::max(n_src[s], 0);

    sampleTiming();

    chk_due = checksum_freq > 0 && (my_step % checksum_freq == 0);
    TRACE(2, "expert %d step %d compute n=%d", thisIndex, my_step, n_tot);
    if (n_tot > 0) {
      const double ti0 = tnow();
      int chunks = 0;
      hapiCheck(cudaEventRecord(ev_start, g->stream));
      if (use_adam) moeZeroGrad(g, d_model, d_ff);
      if (fuse_sources) {
        // One pass over all this expert's tokens instead of one per source.
        // The two weight-update GEMMs of a chunk read and rewrite both weight
        // matrices whatever the chunk holds, so a step's cost is set by the
        // number of chunks, and unfused that is at least one per (source,
        // expert) pair: a typical expert gets a few hundred tokens from each
        // dispatcher and pays a full weight update for each. Concatenating in
        // dispatcher order keeps the chunk boundaries a function of the
        // routing alone, so the reduction order is still placement-independent
        // -- the values differ from the unfused ones, the property does not.
        size_t off = 0;
        for (int s = 0; s < n_disp; s++) {
          if (n_src[s] <= 0) continue;
          const size_t n = (size_t)n_src[s] * d_model;
          hapiCheck(cudaMemcpyAsync(g->fx + off, x_seg + s * segStride(),
              sizeof(float) * n, cudaMemcpyDeviceToDevice, g->stream));
          off += n;
        }
        chunks = (n_tot + chunk_tokens - 1) / chunk_tokens;
        moeExpertChunks(g, W1, W2, g->fx, g->fy, n_tot, d_model, d_ff,
            chunk_tokens, (float)learning_rate, 1.0f / n_tot, use_adam != 0);
        // Back into the per-source segments, so the sends and both pup phases
        // are unchanged. Queued before ev_end, which orders a pack behind it.
        off = 0;
        for (int s = 0; s < n_disp; s++) {
          if (n_src[s] <= 0) continue;
          const size_t n = (size_t)n_src[s] * d_model;
          hapiCheck(cudaMemcpyAsync(y_seg + s * segStride(), g->fy + off,
              sizeof(float) * n, cudaMemcpyDeviceToDevice, g->stream));
          off += n;
        }
      } else {
        for (int s = 0; s < n_disp; s++) {
          if (n_src[s] <= 0) continue;
          chunks += (n_src[s] + chunk_tokens - 1) / chunk_tokens;
          moeExpertChunks(g, W1, W2, x_seg + s * segStride(),
              y_seg + s * segStride(), n_src[s], d_model, d_ff, chunk_tokens,
              (float)learning_rate, 1.0f / n_tot, use_adam != 0);
        }
      }
      TRACE(2, "expert %d step %d issued %d chunks in %.1f ms", thisIndex,
          my_step, chunks, (tnow() - ti0) * 1e3);
      if (use_adam)
        moeAdamApply(g, W1, m1, v1, W2, m2, v2, d_model, d_ff,
            (float)learning_rate, ++adam_t);
      hapiCheck(cudaEventRecord(ev_end, g->stream));
      ev_pending = true;
      n_last = n_tot;
      step_last = my_step;
    }
    if (chk_due) {
      hapiCheck(cudaMemsetAsync(d_chk, 0, sizeof(double), g->stream));
      moeSumSqAdd(W1, wsz(), g, d_chk, g->stream);
      moeSumSqAdd(W2, wsz(), g, d_chk, g->stream);
      hapiCheck(hapiMemcpyAsync(h_chk, d_chk, sizeof(double),
          cudaMemcpyDeviceToHost, g->stream));
      hapiCheck(cudaEventRecord(ev_chk, g->stream));
    }
    if (n_tot > 0 || chk_due) {
      CkCallback* cb = new CkCallback(CkIndex_Expert::computeDone(),
          thisProxy[thisIndex]);
      hapiAddCallback(g->stream, cb);
    } else {
      thisProxy[thisIndex].computeDone();
    }
  }

  void sendOutputs() {
    phase = 2;
    const double ts0 = tnow();
    if (n_tot > 0) hapiCheck(cudaStreamWaitEvent(send_stream, ev_end, 0));
    for (int s = 0; s < n_disp; s++) {
      const int n = std::max(n_src[s], 0);
      const int len = std::max(n, 1) * d_model;
      disp_proxy[s].receiveOutput(my_step, thisIndex, n, len,
          (outstanding_sends++,
           CkDeviceBuffer(y_seg + s * segStride(),
               CkCallback(CkIndex_Expert::sendDone(), thisProxy[thisIndex]),
               send_stream)));
    }
    TRACE(2, "expert %d step %d outputs sent", thisIndex, my_step);
    TRACE(2, "expert %d step %d issued %d sends in %.1f ms", thisIndex,
        my_step, n_disp, (tnow() - ts0) * 1e3);
    gateDrain();
  }

  // The next step's kernels rewrite the output segments, and under async LB a
  // migration's pup frees them; neither may happen over a live transport read.
  void gateDrain() {
    if (outstanding_sends == 0) thisProxy[thisIndex].sendsDrained();
    else drain_pending = true;
  }

  void sendDone() {
    outstanding_sends--;
    if (drain_pending && outstanding_sends == 0) {
      drain_pending = false;
      thisProxy[thisIndex].sendsDrained();
    }
  }

  void endStep() {
    long sum_tok = n_tot;
    long max_tok = n_tot;
    // One slot per expert for the weight checksum, for the same reason as
    // the dispatcher's out2 slots: an order-independent exact sum.
    std::vector<double> w2e(n_experts, 0.0);
    w2e[thisIndex] = chk_due ? *h_chk : 0.0;
    std::vector<long> tokpe(n_disp, 0);
    std::vector<double> gpupe(n_disp, 0.0);
    tokpe[CkMyPe()] = n_tot;
    gpupe[CkMyPe()] = gpu_last;
    CkReduction::tupleElement tuple[] = {
        CkReduction::tupleElement(sizeof(long), &sum_tok, CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(long), &max_tok, CkReduction::max_long),
        CkReduction::tupleElement(sizeof(double) * n_experts, w2e.data(),
            CkReduction::sum_double),
        CkReduction::tupleElement(sizeof(long) * n_disp, tokpe.data(),
            CkReduction::sum_long),
        CkReduction::tupleElement(sizeof(double) * n_disp, gpupe.data(),
            CkReduction::sum_double)};
    CkReductionMsg* msg = CkReductionMsg::buildFromTuple(tuple, 5);
    msg->setCallback(CkCallback(CkIndex_Main::stepStatsExpert(NULL),
        main_proxy));
    contribute(msg);

    if (my_step < warmup_steps + n_steps) {
      thisProxy[thisIndex].runStep();
    } else {
      if (print_place)
        CkPrintf("Expert %d on PE %d: %d tokens in the last step\n", thisIndex,
            CkMyPe(), n_tot);
      contribute(CkCallback(CkReductionTarget(Main, expertsDone), main_proxy));
    }
  }
};

#include "moe.def.h"
