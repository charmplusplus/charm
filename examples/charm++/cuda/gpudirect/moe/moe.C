#include "hapi.h"
#include "moe.decl.h"
#include "moe.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <string>
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

// Device buffers come from hapiMalloc/hapiFree, which are the device pool
// under +gpupool (an arena the peers have already opened, no driver call, no
// device sync) and hapiMalloc/hapiFree without it. The runtime owns the switch.

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
/* readonly */ int cap_src;            // the largest expert's capacity (cap_e): the fused-run scratch
/* readonly */ std::vector<int> cap_e;   // per expert: rows of one dispatcher's slab, from its own peak in the trace
/* readonly */ int n_disp;
/* readonly */ int drift_period;
/* readonly */ int first_lb;
/* readonly */ int lb_freq;
/* readonly */ int async_lb;
/* readonly */ int lb_wait_lag;
static const int LB_PARK_MAX = 512;   // steps after the join an expert parks at whatever the round's state
/* readonly */ int checksum_freq;
/* readonly */ int stats_freq;
/* readonly */ int use_adam;
/* readonly */ int use_tf32;
/* readonly */ int use_bf16;
/* readonly */ int serve_only;
/* readonly */ int use_instrument;
// One device-busy pool per GPU of this process (MoeDevPool, moe.h), indexed by
// the runtime's device index of the calling PE.
static MoeDevPool g_dev_pools[64];
static MoeDevPool* moeDevPool() {
  const int d = hapiMyDeviceIndex();
  return &g_dev_pools[(d >= 0 && d < 64) ? d : 0];
}
/* readonly */ double capacity_factor;
/* readonly */ int fuse_sources;
/* readonly */ int stream_outputs;
/* readonly */ int send_ahead;
/* readonly */ int tok_channels;
/* readonly */ int print_place;
/* readonly */ std::vector<int> disp_pes;   // the PE of each dispatcher: the first PE the runtime maps to each GPU
/* readonly */ int out_channels;   // outputs return through device channels the dispatcher owns (MOE_OUT_ENTRY=1: entry-method sends, the path before 2026-11-06)
/* readonly */ int flight_log;
/* readonly */ int rep_init_expert;
/* readonly */ double zipf_z;
/* readonly */ std::string trace_path;   // -E: routing replayed from a recorded trace ("" = hashed Zipf)
/* readonly */ int use_attn;              // -w: an attention sublayer (32 heads, sequences of 1024) before the MoE
/* readonly */ double learning_rate;
/* readonly */ long moe_seed;

// -F: every slab and output message carries its sender's clock and PE; the
// receiver notes the envelope (the post hook that picks the landing buffer)
// and the payload (the SDAG body), and each PE prints its records after the
// last step. CLOCK_REALTIME in ns, the clock the +gputrace records use, so the
// records line up with a trace; across nodes the clocks agree only to NTP.
static inline long long moeNowNs() {
  struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
// The PE that holds dispatcher d (Main: the first PE of each GPU).
static inline int dispPe(int d) { return disp_pes[d]; }
// The dispatcher on this PE's GPU: the first PE the runtime mapped to it.
static inline int localDispPe() {
  for (int d = 0; d < (int)disp_pes.size(); d++) {
    const int pe = disp_pes[d];
    if (CkNodeOf(pe) == CkMyNode() && hapiDeviceIndexOfPe(pe) == hapiDeviceIndexOfPe(CkMyPe())) return pe;
  }
  return disp_pes.empty() ? 0 : disp_pes[0];
}

struct MoeFlight { char kind; int step, spe, dpe, expert; long bytes; long long t_send, t_env, t_land; };
static thread_local std::vector<MoeFlight> g_flights;
// A token put's completion (its source-done): the record's send time is the issue, env and land the completion.
struct TokFlight { void* disp; int step, expert, dpe; long bytes; long long t_issue; };
static void moeTokSentFn(void* param, void* msg);
static void moeFlightDump() {
  for (const MoeFlight& f : g_flights)
    CkPrintf("[FLIGHT] %c step %d src %d dst %d expert %d bytes %ld send %lld env %lld land %lld\n", f.kind,
        f.step, f.spe, f.dpe, f.expert, f.bytes, f.t_send, f.t_env, f.t_land);
  g_flights.clear();
}

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

// The hot set of a phase: a hashed permutation of the experts; rank r draws a
// share proportional to (r+1)^-z, so perm[0] is the hottest expert.
// A routing trace (ref/trace_routing.py): header int32[4] {0x4d4f4554, n_experts, k, n_tokens},
// then int32 ids[n_tokens][k], the top-k experts of each token of a text in text order. Token t of
// dispatcher d at step s replays text token ((s - 1) * n_disp + d) * n_tokens + t, modulo the trace
// length; the PyTorch harnesses apply the same rule, so every implementation routes the same tokens.
static void loadTrace(const std::string& path, int E, int K, std::vector<int32_t>& ids) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) CkAbort("-E %s: cannot open the routing trace\n", path.c_str());
  int32_t hdr[4];
  if (fread(hdr, sizeof(int32_t), 4, f) != 4 || hdr[0] != 0x4d4f4554)
    CkAbort("-E %s: not a routing trace\n", path.c_str());
  if (hdr[1] != E || hdr[2] != K)
    CkAbort("-E %s: the trace has %d experts and top-%d, the run %d and top-%d\n", path.c_str(),
        hdr[1], hdr[2], E, K);
  ids.resize((size_t)hdr[3] * K);
  if (fread(ids.data(), sizeof(int32_t), ids.size(), f) != ids.size())
    CkAbort("-E %s: short read\n", path.c_str());
  fclose(f);
}

static void hotSetPermutation(int phase, std::vector<int>& perm) {
  perm.resize(n_experts);
  for (int i = 0; i < n_experts; i++) perm[i] = i;
  for (int i = n_experts - 1; i > 0; i--) {
    const uint64_t h = hmix(hmix(moe_seed, 77), (uint64_t)phase * n_experts + i);
    const int j = (int)(h % (uint64_t)(i + 1));
    std::swap(perm[i], perm[j]);
  }
}

// Token channels per (expert, dispatcher): one per step parity when the next
// step's tokens can land while this step computes, else one.
static inline int tokParities() { return send_ahead ? 2 : 1; }

static inline bool isLBStep(int s) {
  return s == first_lb || (s != 0 && lb_freq > 0 && s % lb_freq == 0);
}
static inline int nextLBStep(int s) {
  if (s < first_lb) return first_lb;
  if (lb_freq <= 0) return INT_MAX;
  return ((s / lb_freq) + 1) * lb_freq;
}
// A step inside a balancing window: a balancing step, or one of the lag steps
// after it under the split barrier. The dispatchers send no tokens ahead
// across such a step (Dispatcher::sendAhead), so its outputs are in before
// the next step is routed, and a change of an expert's copies announced in
// them takes effect cleanly at the next step.
static inline bool inLBWindow(int s) {
  const int lag = async_lb ? lb_wait_lag : 0;
  for (int L = s - lag; L <= s; L++)
    if (L >= 1 && isLBStep(L)) return true;
  return false;
}
// Steps from the previous balancing step to step s: the span the balancer's
// own host-side loads accumulate over, and so the span a declared device
// load has to cover to be commensurate with them (LBLoadDim.h). Before the
// first balancing step, the run so far.
static inline int stepsSinceLastLB(int s) {
  if (s <= first_lb) return s > 0 ? s : 1;
  int prev = (lb_freq > 0) ? (s / lb_freq) * lb_freq : 0;
  if (prev == s) prev -= lb_freq;
  if (prev < first_lb) prev = first_lb;
  return s - prev > 0 ? s - prev : 1;
}

// Replication: an expert may exist as k copies (e, 0) .. (e, k-1). The
// dispatchers are in K_PART fixed partitions of contiguous dispatchers,
// p = d K_PART / n_disp, whatever k is (contiguous so that a partition's
// dispatchers share a host when the dispatchers are laid out host by host:
// with four per host, partitions 0 and 1 are host 0's, 2 and 3 host 1's);
// copy r of k serves the contiguous block of partitions [K_PART r / k,
// K_PART (r+1) / k), a single partition or an aligned pair. Each dispatcher's
// PE declares its partition to the balancer (LBManager::setPartition), which
// places a copy with the host that feeds its block when the copies have no
// exchange to pay for (serving). The step's
// gradient is always the fixed tree ((P0 + P1) + (P2 + P3)) over the
// partitions' accumulators (Expert::treeNodes): a copy computes its block's
// leaves, folds them to its block sum, exchanges block sums with the other
// copies and assembles the two tree nodes; an unreplicated expert computes
// all four leaves itself and folds the same way. Every copy then applies the
// same update, so their weights stay identical, and out2 and w2 are the same
// bits whatever the balancer did (doc/internals/lb-multiplicity.md).
// The partitions and the most copies an expert may have are structure, not choices (2026-11-08): in serving every
// dispatcher is its own partition, so the balancer may give an expert any count up to the number of dispatchers (a
// copy serves at least one); training's update is the fixed tree over four partitions above, so four there. Set by
// Main once the dispatchers are known, before any element exists (readonlies).
/* readonly */ int K_PART;
/* readonly */ int K_MAX_COPIES;
// Serving: every dispatcher divides each expert's tokens between all of the expert's copies, by the runtime's split
// of the step's job-wide demand (CkReplicaSplit; 2026-11-09). MOE_BLOCK_ROUTE=1: the copies serve fixed blocks of
// dispatchers instead (CkReplicaRoute), as before.
/* readonly */ int split_route;
static inline int partitionOf(int disp) { return (disp * K_PART) / n_disp; }
static inline int partitionSize(int p) {
  int n = 0;
  for (int d = 0; d < n_disp; d++) if (partitionOf(d) == p) n++;
  return n;
}
static inline int blockLo(int r, int k) { return (K_PART * r) / k; }
static inline int blockHi(int r, int k) { return (K_PART * (r + 1)) / k; }
// Which copy of an expert with `mult` copies serves dispatcher d.
static inline int copyFor(int disp, int mult) {
  const int p = partitionOf(disp);
  for (int r = 0; r < mult; r++) if (p < blockHi(r, mult)) return r;
  return mult - 1;
}

// Bytes per element of a token, an activation or an output: bf16 under -B
// (moe.h), else fp32. The master weights, the gradient accumulators and the
// checksums are fp32 either way.
static inline size_t actEsz() { return use_bf16 ? sizeof(moe_bf16) : sizeof(float); }

// Per-step figures from one of the two report streams, held until the other
// stream's record of the same step arrives so one line reports both.
struct DispRec { double out2; double drop; double t; };
struct ExpRec { double tokratio, gpuratio, gmax_ms, expratio, w2; long sum_tok; };

// Expert (e, 0)'s home PE. The default map blocks the experts over the PEs,
// which at one PE per GPU is a block over the GPUs: SmartMoE's 32 experts two
// per GPU on 16, Mixtral's eight one per GPU on the first host, OLMoE's 64
// four per GPU. At +ppn 4 the same rule over 64 PEs puts SmartMoE's experts
// four per GPU on the first host's eight alone, Mixtral's four on each of two
// GPUs (past the pool's 16 GB on Frontier, where the run aborts), and the
// balancer spends its first round undoing the layout (measured 2026-11-07).
// This map keeps the default's block rule but applies it to the GPUs (the
// devices in PE order, as Main numbers the dispatchers), so the GPU holding
// each expert is the one-PE run's at any +ppn, and spreads a GPU's experts
// over its PEs in turn, the first on the dispatcher's PE.
class ExpertMap : public CkArrayMap {
  std::vector<int> home;
 public:
  ExpertMap(int n_experts) {
    // The devices in PE order, each with its PEs (hapiDeviceIndexOfPe: +ppn
    // PEs over +gpudevs devices).
    std::vector<std::vector<int>> dev;
    for (int p = 0; p < CkNumPes(); p++) {
      int d = -1;
      for (size_t i = 0; i < dev.size() && d < 0; i++) {
        const int q = dev[i][0];
        if (CkNodeOf(q) == CkNodeOf(p) && hapiDeviceIndexOfPe(q) == hapiDeviceIndexOfPe(p)) d = (int)i;
      }
      if (d < 0) { d = (int)dev.size(); dev.emplace_back(); }
      dev[d].push_back(p);
    }
    // The default map's block rule (cklocation.C, DefaultArrayMap::procNum)
    // over D bins: the first rem bins take per + 1 experts, the rest per.
    const int D = (int)dev.size(), per = n_experts / D, rem = n_experts % D, first_set = rem * (per + 1);
    home.resize(n_experts);
    for (int e = 0; e < n_experts; e++) {
      int d, k;
      if (e < first_set) { d = e / (per + 1); k = e % (per + 1); }
      else { d = rem + (e - first_set) / per; k = (e - first_set) % per; }
      home[e] = dev[d][k % dev[d].size()];
    }
    if (CkMyPe() == 0 && CkNumPes() > D) {
      std::string s;
      for (int e = 0; e < n_experts; e++) s += (e ? "," : "") + std::to_string(home[e]);
      CkPrintf("Experts: %d over %d GPUs by the block rule, a GPU's over its %d PEs; expert e on PE %s\n",
          n_experts, D, (int)dev[0].size(), s.c_str());
    }
  }
  ExpertMap(CkMigrateMessage* m) : CkArrayMap(m) {}
  int procNum(int, const CkArrayIndex& idx) override {
    const int e = idx.data()[0];
    return (e >= 0 && e < (int)home.size()) ? home[e] : 0;
  }
};

class Main : public CBase_Main {
  double init_start_time;
  double start_time;
  double window_start_time;
  double first_lb_done_time = -1.0;   // when every dispatcher had finished the first balancing step
  double last_step_time = 0.0;        // ... and the last step
  int last_step = 0;
  int disp_step;   // steps whose dispatcher reports have all arrived
  int exp_step;    // steps whose expert reports have all arrived
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
    use_attn = 0;
    drift_period = 10;
    chunk_tokens = 1024;
    n_lanes = 0;   // resolved below: -n, else four per GPU shared by its PEs
    double headroom = 0.0;
    learning_rate = 1e-4;
    checksum_freq = 5;
    stats_freq = 1;
    first_lb = 10;
    lb_freq = 9999;
    // The round's decision applies at the first step boundary after the runtime has finished the round: an
    // expert keeps stepping until then (it parks only once the round is over, LB_PARK_MAX), so a longer lag only
    // delays the copies (it was 3 until 2026-11-08). -l still overrides.
    lb_wait_lag = 1;
    moe_seed = 12345;
    async_lb = 0;
    use_adam = 0;
    use_tf32 = 0;
    use_bf16 = 0;
    serve_only = 0;
    use_instrument = 0;
    capacity_factor = 0.0;
    fuse_sources = 1;
    stream_outputs = 1;
    send_ahead = 1;
    tok_channels = 1;
    print_place = 0;
    flight_log = getenv("MOE_FLIGHT_LOG") != NULL;   // -F, or the environment for a run whose command line is fixed
    out_channels = getenv("MOE_OUT_ENTRY") == NULL;
    rep_init_expert = -1;
    bool rep_init = false;
    disp_step = exp_step = done_count = 0;

    int c;
    while ((c = getopt(m->argc, m->argv, "e:m:h:t:k:i:u:z:p:c:r:L:C:s:f:b:l:S:n:D:E:aAUTIPFOYMXBGw")) != -1) {
      switch (c) {
        case 'e': n_experts = atoi(optarg); break;
        case 'm': d_model = atoi(optarg); break;
        case 'h': d_ff = atoi(optarg); break;
        case 't': n_tokens = atoi(optarg); break;
        case 'k': top_k = atoi(optarg); break;
        case 'i': n_steps = atoi(optarg); break;
        case 'u': warmup_steps = atoi(optarg); break;
        case 'z': zipf_z = atof(optarg); break;
        case 'E': trace_path = optarg; break;
        case 'w': use_attn = 1; break;
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
        case 'X': rep_init = true; break;
        case 'a': async_lb = 1; break;
        case 'A': use_adam = 1; break;
        case 'T': use_tf32 = 1; break;
        case 'B': use_bf16 = 1; break;
        case 'G': serve_only = 1; break;
        case 'I': use_instrument = 1; break;
        case 'D': capacity_factor = atof(optarg); break;
        case 'U': fuse_sources = 0; break;
        case 'O': stream_outputs = 0; break;
        case 'Y': send_ahead = 0; break;
        case 'M': tok_channels = 0; break;
        case 'P': print_place = 1; break;
        case 'F': flight_log = 1; break;
        default:
          CkPrintf(
              "Usage: %s -e [experts] -m [d_model] -h [d_ff] -t [tokens per process per step]\n"
              "  -k [top-k] -i [steps] -u [warmup steps]\n"
              "  -z [zipf exponent of expert popularity, 0 = uniform]\n"
              "  -E [routing trace to replay (ref/trace_routing.py) instead of the hashed Zipf]\n"
              "  -w [a transformer block: an attention sublayer (32 heads, sequences of 1024) before the MoE]\n"
              "  -p [drift period: steps between reshuffles of the hot set, 0 = static]\n"
              "  -c [chunk: tokens per GEMM] -r [capacity headroom, 0 = auto from zipf]\n"
              "  -n [compute lanes per PE: streams experts are spread over;\n"
              "      default four per GPU divided among the process's PEs]\n"
              "  With several PEs per process (+ppn N, MOE_PPN=N through the\n"
              "  wrapper) the process still has one dispatcher and -t tokens, on\n"
              "  its first PE; the experts spread over all of its PEs.\n"
              "  -L [learning rate] -A (Adam: 3x payload) -T (TF32 GEMMs)\n"
              "  -B (bf16 GEMM operands: the tokens, activations and outputs, and a bf16\n"
              "      copy of each weight the GEMMs read; fp32 accumulation, fp32 master\n"
              "      weights, moments and gradient accumulators. Halves the exchange bytes.\n"
              "      The checksums differ from fp32's and are placement-independent as before.)\n"
              "  -G (serving: the forward alone, no gradient, no update, and no exchange\n"
              "      between an expert's copies, which then cost one state send each.\n"
              "      w2 stays its initial value; out2 is still the oracle across placements.)\n"
              "  -D [capacity factor: cap each expert at f x the mean tokens per\n"
              "      expert and DROP the overflow, the GShard/Switch policy. Bounds\n"
              "      the imbalance without moving anything; the dropped tokens get a\n"
              "      zero output, so the checksums differ from every other mode and\n"
              "      the drop rate is reported.]\n"
              "  -U (one chunked pass per source instead of one per expert: many more\n"
              "      chunks, kept for comparison. Either way the chunk boundaries follow\n"
              "      the routing alone, so the checksums are placement-independent; the\n"
              "      two modes' values differ.)\n"
              "  -X (start with the hottest expert of the first hot set replicated,\n"
              "      its second copy on the first PE of process 1 mod processes: the\n"
              "      replicated arithmetic with no balancer, same checksums as without)\n"
              "  -C [checksum frequency, 0 disables] -s [stats frequency, 0 disables]\n"
              "  -f [first LB step] -b [LB frequency]\n"
              "  -a (async LB: overlap the step; needs +LBAsync) -l [wait lag]\n"
              "  -I (CUPTI-measured loads in a window before each LB step, instead of\n"
              "      the experts' token-count estimates)\n"
              "  -S [seed] -P (print final placement)\n"
              "  -F (print every slab and output message: bytes, sent, envelope, landed)\n"
              "  -O (send an expert's outputs after its whole step, not each dispatcher's\n"
              "      as soon as its tokens are done)\n"
              "  -Y (send a step's tokens only after the previous step's outputs are back,\n"
              "      not while the experts still compute it)\n"
              "  -M (send tokens as device messages the expert reads, not as channel puts\n"
              "      the dispatcher writes)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (top_k < 1 || top_k > MOE_MAX_TOPK || top_k > n_experts)
      CkAbort("-k must be in [1, min(%d, experts)]\n", MOE_MAX_TOPK);
    if (chunk_tokens < 1) CkAbort("-c must be positive\n");
    if (use_attn && (n_tokens % 1024 != 0 || d_model % 32 != 0))
      CkAbort("-w needs -t a multiple of 1024 and -m a multiple of 32\n");
    if (use_bf16 && use_tf32) CkAbort("-B and -T exclude each other\n");
    // Several PEs per process (+ppn) share one GPU, and each PE's dispatcher
    // owns its own lanes -- a stream, a cuBLAS handle with a fixed workspace,
    // and fused-source scratch sized by the dispatcher count. Lanes exist to
    // overlap experts on one device; PEs already do that, so the default
    // keeps about four lanes per GPU rather than four per PE, which at eight
    // PEs would mean 32 handles and gigabytes of scratch on one card.
    // Four lanes per GPU: the runtime says how many PEs share one (+ppn PEs over
    // +gpudevs devices, hapiPesPerDevice), one process per node driving four
    // GPUs being one PE per device.
    const int pes_per_device = std::max(1, hapiPesPerDevice());
    if (n_lanes == 0) n_lanes = std::max(1, 4 / pes_per_device);
    if (n_lanes < 1 || n_lanes > MOE_MAX_LANES)
      CkAbort("-n must be in [1, %d]\n", MOE_MAX_LANES);
    // The split barrier is the application's choice (-a), declared to the
    // runtime here as sph3d, pic2d and leanmd do; no +LBAsync on the command
    // line. (Until 2026-10-31 this asked for the flag and fell back to the
    // unsplit barrier without it, so every "-a" run whose script did not
    // pass +LBAsync ran synchronously.)
    if (async_lb) CkLBSetAsync(true);
    if (async_lb && (lb_wait_lag < 1 || lb_wait_lag >= lb_freq)) {
      CkAbort("-l (%d) must be in [1, lb_freq): the wait must land before "
              "the next AtSyncStart\n", lb_wait_lag);
    }
    // One dispatcher per GPU, on the first PE the runtime maps to it
    // (hapiDeviceIndexOfPe: +ppn PEs over +gpudevs devices, block by default).
    // PEs sharing a GPU get one dispatcher between them on the first; its
    // other PEs hold no tokens, carry experts and provide their compute
    // context, so the work per GPU, the slabs and the routing are the same at
    // any +ppn and the checksums match the one-PE-per-process run bit for
    // bit. One process per node driving four GPUs is one dispatcher per PE,
    // the puts between a node's GPUs then plain copies in one address space.
    disp_pes.clear();
    for (int p = 0; p < CkNumPes(); p++) {
      bool first = true;
      for (int q = CkNodeFirst(CkNodeOf(p)); q < p && first; q++)
        if (hapiDeviceIndexOfPe(q) == hapiDeviceIndexOfPe(p)) first = false;
      if (first) disp_pes.push_back(p);
    }
    n_disp = (int)disp_pes.size();
    K_PART = serve_only ? n_disp : 4;
    split_route = serve_only && getenv("MOE_BLOCK_ROUTE") == NULL;
    K_MAX_COPIES = K_PART;
    if (CkNumPes() > CkNumNodes())
      CkPrintf("MoE: %d dispatchers on %d PEs in %d processes (%d PE%s per GPU)\n", n_disp, CkNumPes(), CkNumNodes(),
          pes_per_device, pes_per_device == 1 ? "" : "s");
    // A replayed trace: the hottest expert and the slab capacity come from the
    // trace itself, p_max being the largest share any expert takes of one
    // dispatcher's tokens in one step (one chunk of n_tokens text tokens).
    std::vector<int32_t> trace_ids;
    std::vector<long> trace_emax(n_experts, 0);   // per expert: its most tokens from one dispatcher in one step
    double trace_share_max = 0.0, trace_step_max = 0.0;
    int trace_hot = -1;
    if (!trace_path.empty()) {
      loadTrace(trace_path, n_experts, top_k, trace_ids);
      main_trace = trace_ids;   // kept: the conservation check names the expert short of its tokens
      const size_t n = trace_ids.size() / top_k;
      std::vector<long> cnt(n_experts, 0), ccnt(n_experts);
      for (size_t i = 0; i < trace_ids.size(); i++) cnt[trace_ids[i]]++;
      for (int e = 0; e < n_experts; e++) if (trace_hot < 0 || cnt[e] > cnt[trace_hot]) trace_hot = e;
      trace_share_max = (double)cnt[trace_hot] / (double)trace_ids.size();
      for (size_t c = 0; c + (size_t)n_tokens <= n; c += n_tokens) {
        std::fill(ccnt.begin(), ccnt.end(), 0);
        for (size_t i = c * top_k; i < (c + n_tokens) * top_k; i++) ccnt[trace_ids[i]]++;
        for (int e = 0; e < n_experts; e++) {
          trace_step_max = std::max(trace_step_max, (double)ccnt[e] / ((double)top_k * n_tokens));
          trace_emax[e] = std::max(trace_emax[e], ccnt[e]);
        }
      }
    }
    if (rep_init) {
      if (n_disp < 2)
        CkAbort("-X needs at least two processes: a copy serves the second half of the dispatcher partitions\n");
      std::vector<int> perm;
      hotSetPermutation(0, perm);
      rep_init_expert = trace_path.empty() ? perm[0] : trace_hot;
    }

    // Per-(source, expert) slab capacity. The hottest expert draws a share
    // p_max of every dispatcher's tokens; auto sizes for 1.5x that plus slack.
    double p_max;
    if (!trace_path.empty()) {
      p_max = trace_step_max;
    } else if (zipf_z > 0.0) {
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
    // Each expert its own capacity, by the same rule from its own peak (2026-11-08): sized for the hottest, every
    // expert and every copy held 1.2 GB of SmartMoE slabs, and the balancer could not place the copies the hot
    // expert needed ("the bin could not hold the object"). -r keeps one capacity for all.
    cap_e.assign(n_experts, cap_src);
    if (!trace_path.empty() && headroom <= 0.0) {
      int cmax = 64, cmin = INT_MAX;
      for (int e = 0; e < n_experts; e++) {
        const double c = std::min(1.5 * trace_emax[e] + 64.0, (double)top_k * n_tokens);
        cap_e[e] = std::max(64, (int)((ceil(c) + 31.0) / 32.0) * 32);
        cmax = std::max(cmax, cap_e[e]);
        cmin = std::min(cmin, cap_e[e]);
      }
      cap_src = cmax;
      CkPrintf("Slab capacity per (dispatcher, expert): %d to %d rows, each expert from its own peak in the trace\n",
          cmin, cmax);
    }

    const size_t wbytes = sizeof(float) * (size_t)d_model * d_ff * 2 *
                          (use_adam ? 3 : 1) +
                          (use_bf16 ? sizeof(moe_bf16) * (size_t)d_model * d_ff * 2 : 0);
    // Inputs and outputs, and a second input slab when the next step's tokens
    // can land while this step computes (send_ahead).
    const size_t slab_bytes = actEsz() * (size_t)n_disp * cap_src *
                              d_model * (send_ahead ? 3 : 2);
    const double gflop = 10.0 * avg_src * n_disp * d_model * d_ff * 1e-9;

    CkPrintf("\n[CUDA mixture-of-experts layer]\n");
    CkPrintf("Experts: %d, d_model %d, d_ff %d, %s%s, up to %d cop%s each\n", n_experts, d_model,
        d_ff, serve_only ? "serving (forward only)" : use_adam ? "Adam" : "SGD",
        use_bf16 ? ", bf16 operands with fp32 master weights" : use_tf32 ? ", TF32" : "", K_MAX_COPIES,
        K_MAX_COPIES == 1 ? "y" : "ies");
    CkPrintf("Tokens: %d per dispatcher per step on %d dispatchers (one per GPU, %d PE%s each), "
        "top-%d, chunk %d, slab capacity %d per (dispatcher, expert), %d compute "
        "lanes per PE%s\n",
        n_tokens, n_disp, pes_per_device, pes_per_device == 1 ? "" : "s", top_k,
        chunk_tokens, cap_src, n_lanes,
        fuse_sources ? "" : ", one pass per source");
    CkPrintf("Outputs return %s\n", out_channels ? "through the dispatchers' device channels (one segment per expert)"
                                                  : "as entry-method device sends (MOE_OUT_ENTRY)");
    if (capacity_factor > 0.0)
      CkPrintf("Capacity factor: %.2f (GShard/Switch token dropping; no state moves)\n",
          capacity_factor);
    if (!trace_path.empty())
      CkPrintf("Routing: trace %s (%zu tokens; hottest expert %.1fx average over the text, "
          "%.1fx in its hottest step; replayed in text order, a chunk of %d tokens per "
          "dispatcher and step)\n", trace_path.c_str(), trace_ids.size() / top_k,
          trace_share_max * n_experts, trace_step_max * n_experts, n_tokens);
    else
      CkPrintf("Routing: zipf %.2f (hottest expert %.1fx average), hot set "
          "reshuffled every %d steps, seed %ld\n", zipf_z, p_max * n_experts,
          drift_period, moe_seed);
    if (use_attn)
      CkPrintf("Block: an attention sublayer (q/k/v/o %d x %d, 32 heads, causal, sequences of 1024, %s) runs on each "
          "dispatcher's tokens before the MoE, trained on its own loss%s\n", d_model, d_model,
          moeAttnKernels(use_bf16 != 0), serve_only ? " (forward only)" : "");
    if (rep_init_expert >= 0)
      CkPrintf("Expert %d starts as two copies (-X): copy 1 on PE %d serves dispatcher partitions 2 and 3 of 4\n",
          rep_init_expert, CkNodeFirst(1 % CkNumNodes()));
    CkPrintf("Per expert: %.1f MB migratable state, %.1f MB slabs, "
        "%.1f GFLOP per step at average load\n", wbytes / 1048576.0,
        slab_bytes / 1048576.0, gflop);
    CkPrintf("Steps: %d (+%d warmup), first LB: %d, LB frequency: %d%s, "
        "loads: %s; next step's tokens sent %s, %s\n\n", n_steps, warmup_steps, first_lb, lb_freq,
        async_lb ? ", async" : "",
        use_instrument ? "CUPTI-measured" : "token-count estimates",
        send_ahead ? "while this step computes" : "after its outputs are back",
        tok_channels ? "as channel puts" : "as device messages");

    disp_proxy = CProxy_Dispatcher::ckNew();
    // (e, 0) for every expert; (e, 1) is inserted when a copy is made. The
    // bounds give the second index room in the element id -- created as
    // (n, 1) alone, the id would have no bit for it and (e, 1) would take
    // another element's id.
    CkArrayOptions opts(n_experts, 1);
    opts.setBounds(n_experts, K_MAX_COPIES);
    opts.setMap(CProxy_ExpertMap::ckNew(n_experts));   // by GPU, not by PE (ExpertMap)
    expert_proxy = CProxy_Expert::ckNew(opts);
    if (rep_init_expert >= 0)
      expert_proxy(rep_init_expert, 1).insert(CkNodeFirst(1 % CkNumNodes()));
    init_start_time = CkWallTimer();
    disp_proxy.init();
  }

  // The experts initialise once every dispatcher is ready. Each expert
  // declares to the balancer what a copy of it would send its peers (bytes
  // and messages per interval) and, once it has copies, what the exchange
  // measured; the runtime's calibrated transfer-cost table prices the rest.
  void dispReady() {
    expert_proxy.init();
  }

  // The steps start once the experts are initialised and, with token channels,
  // every dispatcher holds every expert's channels.
  bool experts_ready = false;
  int chan_ready_n = 0;
  void expertsReady() { experts_ready = true; maybeStart(); }
  void dispChannelsReady() { chan_ready_n++; maybeStart(); }

  void maybeStart() {
    if (!experts_ready || (tok_channels && chan_ready_n < n_disp)) return;
    CkPrintf("Init time: %.3lf s\n", CkWallTimer() - init_start_time);
    start_time = CkWallTimer();
    window_start_time = start_time;
    // Only the dispatcher branch on each process's first PE runs steps.
    for (int d = 0; d < n_disp; d++) disp_proxy[dispPe(d)].runStep();
    expert_proxy.runStep();
  }

  // One report per dispatcher per step. A step is in hand once all n_disp
  // have reported; their values are then added in dispatcher order, the same
  // fixed order the old slot reduction summed in, so the checksum is the same
  // bits on every run and every placement. Dispatchers run at most a step
  // apart, so the map holds one or two steps.
  std::map<int, std::vector<double>> pend_out2, pend_drop;
  std::map<int, int> pend_n;
  int disp_done_n = 0;

  void dispStep(int disp, int step, double out2_d, double drop_d) {
    std::vector<double>& o = pend_out2[step];
    std::vector<double>& d = pend_drop[step];
    if (o.empty()) { o.assign(n_disp, 0.0); d.assign(n_disp, 0.0); }
    o[disp] = out2_d;
    d[disp] = drop_d;
    if (++pend_n[step] < n_disp) return;
    double out2 = 0.0, drop = 0.0;
    for (int p = 0; p < n_disp; p++) out2 += o[p];
    for (int p = 0; p < n_disp; p++) drop += d[p];
    pend_out2.erase(step);
    pend_drop.erase(step);
    pend_n.erase(step);
    disp_step++;
    const double now = CkWallTimer();
    if (disp_step == warmup_steps) {
      start_time = now;
      window_start_time = now;
    }
    // The first balancing step's completion: the average reported after it covers everything from there on (the
    // first round's transition, every later round, its copies and moves), on the same clock as the per-step lines.
    if (disp_step == first_lb) first_lb_done_time = now;
    last_step_time = now;
    last_step = disp_step;
    DispRec r;
    r.out2 = out2;
    r.drop = drop;
    r.t = now;
    disp_q.push_back(r);
    tryPrint();
  }

  // One report per expert per step, from its rank-0 copy, which folds in what
  // its other copy saw. Held per step until all n_experts are in; the weight
  // checksums are added in expert order, a fixed order, as the dispatchers'
  // are. Experts run at most a step or two apart.
  struct ExpPend {
    int n = 0;
    long sum_tok = 0, max_tok = 0;
    std::vector<double> w2e;
    std::vector<long> ne;   // per expert: its report (copies included)
    std::vector<int> ce;    // per expert: the reports in (serving: one per copy)
    std::vector<long> tokpe;
    std::vector<double> gpupe;
    std::vector<double> pool_busy, pool_tok;   // per PE, the latest report's figure
  };
  std::map<int, ExpPend> pend_exp;
  std::vector<double> w2_seen;   // serving: each expert's last weight checksum
  std::vector<int32_t> main_trace;
  int exp_done_n = 0;
  double rate_sent = 0.0;

  // One report per expert and step (ncopies 1: the primary's, its copies' figures in pes/ns/gs), or in serving one
  // per copy (ncopies: the expert's copies that step), the expert's step complete when all of them are in.
  void expertStep(int e, int step, long n_tot, double w2, int npe, int* pes, long* ns, double* gs,
                  double pool_busy, double pool_tok, int ncopies, int rank) {
    ExpPend& p = pend_exp[step];
    if (p.w2e.empty()) {
      p.w2e.assign(n_experts, 0.0);
      p.tokpe.assign(CkNumPes(), 0);
      p.gpupe.assign(CkNumPes(), 0.0);
      p.pool_busy.assign(CkNumPes(), 0.0);
      p.pool_tok.assign(CkNumPes(), 0.0);
      p.ce.assign(n_experts, 0);
      p.ne.assign(n_experts, 0);
    }
    if (rank == 0) {
      p.w2e[e] = w2;
      // Serving changes no weight: an expert's checksum that moves names the copy whose masters went bad (2026-11-09:
      // one run's sum jumped from 1.0923e5 to 4.1150e5 between steps 25 and 30, after the round at 20, and stayed).
      if (serve_only && w2 != 0.0) {
        if (w2_seen.empty()) w2_seen.assign(n_experts, 0.0);
        if (w2_seen[e] != 0.0 && w2 != w2_seen[e])
          CkPrintf("[W2] expert %d: weight checksum %.17g at step %d on PE %d, %.17g before\n", e, w2, step,
              npe > 0 ? pes[0] : -1, w2_seen[e]);
        w2_seen[e] = w2;
      }
    }
    p.ne[e] += n_tot;   // per expert, for the conservation report
    for (int i = 0; i < npe; i++) {
      if (pes[i] < 0) continue;
      p.tokpe[pes[i]] += ns[i];
      p.gpupe[pes[i]] += gs[i];
      if (i == 0) { p.pool_busy[pes[i]] = pool_busy; p.pool_tok[pes[i]] = pool_tok; }
    }
    if (++p.ce[e] < ncopies) return;
    p.sum_tok += p.ne[e];
    p.max_tok = std::max(p.max_tok, p.ne[e]);
    if (++p.n < n_experts) return;
    // The job-wide rate: every GPU's busy time over every GPU's tokens. One
    // scale for every expert's declared load (MoeCtx::global_spt).
    {
      double busy = 0.0, tok = 0.0;
      for (int q = 0; q < CkNumPes(); q++) { busy += p.pool_busy[q]; tok += p.pool_tok[q]; }
      if (tok > 0.0 && busy > 0.0) {
        const double rate = busy / tok;
        if (rate_sent == 0.0 || std::fabs(rate - rate_sent) > 0.01 * rate_sent) {
          rate_sent = rate;
          disp_proxy.setRate(rate);
        }
      }
    }
    const long sum_tok = p.sum_tok;
    const long max_tok = p.max_tok;
    double w2sum = 0.0;
    for (int i = 0; i < n_experts; i++) w2sum += p.w2e[i];
    exp_step++;

    // With a capacity factor the experts legitimately see fewer tokens than
    // were routed; the pair is checked in tryPrint, where the dispatchers'
    // drop count for the same step is also in hand.
    const long expect = (long)top_k * n_tokens * n_disp;
    if (capacity_factor == 0.0 && sum_tok != expect) {
      std::string per;
      for (int i = 0; i < n_experts; i++) per += " " + std::to_string(i) + ":" + std::to_string(p.ne[i]);
      CkPrintf("Token report of step %d (report step %d), expert:tokens (its copies' included):%s\n", exp_step, step,
          per.c_str());
      if (!main_trace.empty()) {
        // What the trace routed to each expert in this step, over every dispatcher's chunk (Dispatcher::route).
        std::vector<long> want(n_experts, 0);
        const size_t n = main_trace.size() / top_k;
        for (int d = 0; d < n_disp; d++) {
          const size_t g0 = ((size_t)(step - 1) * n_disp + d) * n_tokens;
          for (int t = 0; t < n_tokens; t++)
            for (int j = 0; j < top_k; j++) want[main_trace[((g0 + t) % n) * top_k + j]]++;
        }
        for (int i = 0; i < n_experts; i++)
          if (p.ne[i] != want[i])
            CkPrintf("Token report of step %d: expert %d saw %ld of the %ld the trace routed to it\n", step, i,
                p.ne[i], want[i]);
      }
      CkAbort("Token count not conserved at step %d: experts saw %ld of %ld\n",
          exp_step, sum_tok, expect);
    }

    // Per PE, whatever the process count: the experts' spread over PEs.
    const int npes = CkNumPes();
    long max_pe = 0;
    double gsum = 0.0, gmax = 0.0;
    for (int q = 0; q < npes; q++) {
      max_pe = std::max(max_pe, p.tokpe[q]);
      gsum += p.gpupe[q];
      gmax = std::max(gmax, p.gpupe[q]);
    }
    ExpRec r;
    r.tokratio = (double)max_pe / ((double)sum_tok / npes);
    r.gpuratio = gsum > 0.0 ? gmax / (gsum / npes) : 0.0;
    r.gmax_ms = gmax * 1e3;
    r.expratio = (double)max_tok / ((double)sum_tok / n_experts);
    r.w2 = w2sum;
    r.sum_tok = sum_tok;
    exp_q.push_back(r);
    pend_exp.erase(step);
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
      // Conservation, now that both halves of the step are in hand: every
      // routed slot was either seen by an expert or dropped for capacity.
      const long routed = (long)top_k * n_tokens * n_disp;
      if (e.sum_tok + (long)llround(d.drop) != routed)
        CkAbort("Step %d: %ld tokens seen + %ld dropped != %ld routed\n",
            step, e.sum_tok, (long)llround(d.drop), routed);
      if (stats_freq > 0 && step > warmup_steps &&
          (step - warmup_steps) % stats_freq == 0) {
        const double ms = (d.t - window_start_time) / stats_freq * 1e3;
        window_start_time = d.t;
        char dropbuf[64];
        dropbuf[0] = '\0';
        if (capacity_factor > 0.0)
          snprintf(dropbuf, sizeof(dropbuf), ", dropped %.2lf%%",
              100.0 * d.drop / ((double)top_k * n_tokens * n_disp));
        // %.15e on the checksums: a placement-to-placement comparison needs
        // every digit.
        if (checksum_freq > 0 && step % checksum_freq == 0)
          CkPrintf("Step %d: %.3lf ms/step, tokens/PE max/avg %.2lf, "
              "gpu/PE max %.1lf ms max/avg %.2lf (prev), expert max/avg %.2lf%s, "
              "out2 %.15e, w2 %.15e\n", step, ms, e.tokratio, e.gmax_ms,
              e.gpuratio, e.expratio, dropbuf, d.out2, e.w2);
        else
          CkPrintf("Step %d: %.3lf ms/step, tokens/PE max/avg %.2lf, "
              "gpu/PE max %.1lf ms max/avg %.2lf (prev), expert max/avg "
              "%.2lf%s\n", step, ms, e.tokratio, e.gmax_ms, e.gpuratio,
              e.expratio, dropbuf);
      }
    }
  }

  void dispDone() {
    if (++disp_done_n < n_disp) return;
    finish();
  }
  void expertDone(int e) {
    if (++exp_done_n < n_experts) return;
    finish();
  }

  void finish() {
    if (++done_count < 2) return;
    const double total_time = CkWallTimer() - start_time;
    CkPrintf("\nTotal time: %.3lf s\nAverage step time: %.3lf ms\n",
        total_time, total_time / n_steps * 1e3);
    // Every step after the first balancing step, rounds and all: from the completion of step first_lb to that of
    // the last step, over the steps in between (the per-step lines' sum over the same steps).
    if (first_lb_done_time >= 0.0 && last_step > first_lb)
      CkPrintf("Average step time after the first balancing step (steps %d-%d): %.3lf ms\n", first_lb + 1, last_step,
          (last_step_time - first_lb_done_time) / (last_step - first_lb) * 1e3);
    else
      CkPrintf("Average step time after the first balancing step: none (no balancing step in the run)\n");
    CkExit();
  }
};

// ---------------------------------------------------------------------------

// Device events, per PE, reused (2026-11-09): hipEventCreate costs ~175 us on ROCm 7.14, and an expert's ~20 events
// (one per dispatcher's output, four of its own) held its arrival's unpack on the host 3.55 ms of 3.7, on every
// clone and move, so a PE that received nine experts in a round stalled 32 ms and the job's chain with it. An event
// is handed on whatever it last recorded: its next owner records it before any query or wait, and a wait already
// queued holds the earlier record. [0] timing, [1] not.
static thread_local std::vector<hapiEvent_t> moe_ev_pool[2];
static bool moeEvPoolOff() { static const bool off = getenv("MOE_NO_EVPOOL") != NULL; return off; }
static hapiEvent_t moeEventGet(bool timing) {
  std::vector<hapiEvent_t>& v = moe_ev_pool[timing ? 0 : 1];
  if (!moeEvPoolOff() && !v.empty()) { hapiEvent_t e = v.back(); v.pop_back(); return e; }
  hapiEvent_t e;
  if (timing) hapiCheck(hapiEventCreate(&e));
  else hapiCheck(hapiEventCreateWithFlags(&e, hapiEventDisableTiming));
  return e;
}
static void moeEventPut(hapiEvent_t e, bool timing) {
  if (!e) return;
  if (moeEvPoolOff()) { hapiCheck(hapiEventDestroy(e)); return; }
  moe_ev_pool[timing ? 0 : 1].push_back(e);
}
// At start-up, before the first step: enough for `experts` experts on this PE.
static void moeEventWarm(int experts, int per_expert_plain) {
  // MOE_STREAMWARM=1: the runtime's stream pool too. An expert takes two streams on arrival (createEvents), and an
  // empty pool makes each with hipStreamCreate, which holds the arrival ~3.4 ms on the host. Off by default: with
  // the pool made at start-up, so that an arriving expert ran at once, outputs came out wrong at some steps (one run
  // in three to six with MOE_SPLIT_MIN=256; none with the streams made on arrival, 0 of 12, 2026-11-09). A race in
  // what an expert's arrival does next, which the creation time hides; not found yet (the unpack's device copies
  // are waited for, the payload and the stable region are taken as landed only on the landing stream).
  static const bool stream_warm = getenv("MOE_STREAMWARM") != NULL;
  if (stream_warm) {
    std::vector<hapiStream_t> st;
    for (int i = 0; i < 2 * experts; i++) st.push_back(hapiAcquireStream());
    for (hapiStream_t x : st) hapiReleaseStream(x);
  }
  if (moeEvPoolOff()) return;
  std::vector<hapiEvent_t> t, n;
  for (int i = 0; i < 2 * experts; i++) t.push_back(moeEventGet(true));
  for (int i = 0; i < per_expert_plain * experts; i++) n.push_back(moeEventGet(false));
  for (hapiEvent_t e : t) moeEventPut(e, true);
  for (hapiEvent_t e : n) moeEventPut(e, false);
}

// Split routing's plan of one step (2026-11-09), the same on every PE: each dispatcher's tokens per expert (c, expert
// major) and each copy's target (x, from the runtime's split of the step's job-wide demand). An expert's tokens fill its
// copies in rank order, dispatcher by dispatcher, so a dispatcher sends an expert's tokens to one copy, or two where
// a copy's target ends inside its run: n_disp + k - 1 transfers an expert instead of n_disp k, and no empty ones (with
// every dispatcher sending to every copy, the host spent 3.7-6.6 ms a 13 ms step issuing puts and per-source kernels).
// The walk is host by host on both sides (2026-11-09): the copies of an expert in the order of their hosts (ord, from
// where the runtime last heard each copy is, CkReplicaWhere) and the dispatchers in the order of theirs, so a host's
// dispatchers fill the copies on that host first and only the difference between a host's demand and its copies'
// targets crosses between hosts. Walked in rank and dispatcher order, half the tokens crossed, and a put across the
// two nodes took 6.5-7.3 ms from send to landing (median, any size) against 0.2 ms within a node.
struct MoeStepPlan {
  std::vector<int> c, k, xo, x, ord;   // ord[xo[e] + j]: the copy at walk position j
};
static std::vector<int> moe_disp_walk;   // the dispatchers in host order
static std::vector<int> moe_disp_at;     // each dispatcher's place in it
static void moeDispWalk() {
  if (!moe_disp_walk.empty()) return;
  const int nd = (int)disp_pes.size();
  moe_disp_walk.resize(nd);
  for (int d = 0; d < nd; d++) moe_disp_walk[d] = d;
  std::stable_sort(moe_disp_walk.begin(), moe_disp_walk.end(),
      [](int a, int b) { return CmiPhysicalNodeID(disp_pes[a]) < CmiPhysicalNodeID(disp_pes[b]); });
  moe_disp_at.resize(nd);
  for (int i = 0; i < nd; i++) moe_disp_at[moe_disp_walk[i]] = i;
}
// Where each walk position's range starts in dispatcher d's run of expert e's tokens: so[0..k], so[k] = its count;
// the copy at position j is P.ord[P.xo[e] + j].
static void moePlanOffsets(const MoeStepPlan& P, int nd, int e, int d, int* so) {
  moeDispWalk();
  const int k = P.k[e];
  const int* x = &P.x[P.xo[e]];
  const int* ord = &P.ord[P.xo[e]];
  const int* c = &P.c[(size_t)e * nd];
  const long inf = LONG_MAX / 4;
  int r = 0;
  long room = (k == 1) ? inf : x[ord[0]];
  auto adv = [&]() { r++; room = (r == k - 1) ? inf : x[ord[r]]; };
  for (int i = 0; i < moe_disp_at[d]; i++) {
    long n = c[moe_disp_walk[i]];
    while (n > 0) {
      if (room == 0) { adv(); continue; }
      const long t = std::min(n, room);
      n -= t; room -= t;
    }
  }
  for (int i = 0; i <= k; i++) so[i] = 0;
  long n = c[d];
  int pos = 0;
  while (n > 0) {
    if (room == 0) { adv(); so[r] = pos; continue; }
    const long t = std::min(n, room);
    n -= t; room -= t; pos += (int)t;
  }
  for (int i = r + 1; i <= k; i++) so[i] = pos;
}

class Dispatcher : public CBase_Dispatcher {
  Dispatcher_SDAG_CODE

 public:
  MoeCtx gpu;
  int my_step = 0;
  int recv_count = 0;
  bool drain_pending = false;
  bool chk_due = false;
  int cur_phase = -1;
  std::vector<int> perm;        // expert at each popularity rank, this phase
  std::vector<double> cdf;      // cumulative popularity by rank
  std::vector<int> fill_;
  std::vector<int> slot_expert; // expert chosen for each (token, j)
  std::vector<int32_t> trace_ids; // -E: the replayed trace, loaded on first use
  MoeAttn attn;                 // -w: the attention sublayer on this dispatcher's tokens
  hapiEvent_t ev_attn;
  std::vector<long long> env_t;   // -F: envelope arrival per (step parity, expert)

  // One step's routing and the buffers that carry it. Two alternate by step
  // parity: with send_ahead, step s+1 is routed, gathered and sent while step
  // s's outputs are still landing and before s is combined.
  // A step's routing (RouteBuf) rotates over three steps, its slabs (StepBuf) over two (2026-11-09): the step after
  // next can be routed and gathered while this step's outputs come in and its combine reads this step's receive
  // positions, and only its sends wait for this step's last output (Dispatcher::tryPreGather).
  struct RouteBuf {
    std::vector<int> counts;    // tokens per expert
    std::vector<int> offs;      // slab offsets per expert (n_experts + 1)
    long dropped = 0;           // tokens the capacity factor refused
    int* d_send_idx = nullptr;  // token for each slot of d_send
    int* d_recv_pos = nullptr;  // slot of d_recv for each (token, j)
    int* h_send_idx = nullptr;  // pinned copies
    int* h_recv_pos = nullptr;
    // Split routing: per expert its copies this step and where each copy's tokens start in the expert's run of the
    // send slab (soff[e * (K_MAX_COPIES + 1) + r], r = 0..k); the outputs the step waits for, one per copy.
    std::vector<int> nk, soff, sord;   // sord[e * K_MAX_COPIES + j]: the copy at walk position j
    int outputs_expected = 0;
  };
  RouteBuf rb[3];
  RouteBuf& routeBuf(int step) { return rb[step % 3]; }
  const RouteBuf& routeBuf(int step) const { return rb[step % 3]; }
  int gathered_step = 0;   // the last step routed and gathered (issueGather); its sends may still be to go
  struct StepBuf {
    void* d_send = nullptr;     // [(n_tokens*k + 1) x d_model], tokens sorted by expert; fp32 or bf16 (actEsz)
    void* d_recv = nullptr;     // same layout, expert outputs; +2 rows, see init() (the first chunk)
    // The receive slab in chunks of rpc rows, whole expert segments each (see init(): a NIC-registered pool
    // arena holds at most 2048 MB on Frontier); the chunks' bases on the device for the combine kernel.
    std::vector<void*> d_recv_chunks;
    void** d_recv_bases = nullptr;
    int rpc = 0;
    int outstanding = 0;        // this step's token sends not yet complete
    void** h_recv_bases = nullptr;   // pinned mirror of d_recv_bases (split routing: segments are added as copies appear)
  };
  StepBuf sb[2];
  StepBuf& stepBuf(int step) { return sb[step & 1]; }
  int send_step = 0;      // the step issueGather routed; sendTokens sends it
  int sent_through = 0;   // the last step whose tokens are out
  // How many copies of each expert serve this step, and the change an
  // expert announced for the next step (0: none). The copy this dispatcher
  // sends expert e's slab to is copyFor(dispId, mult[e]).
  std::vector<int> mult;
  std::vector<int> pend_mult;
  std::vector<int> pend_from;   // the first step pend_mult serves: two after the step whose outputs announced it
  // Sender copies of the expert copies' token channels, by (expert, copy,
  // step parity); NULL where no copy of that rank sends here. Pointers, so a
  // copy's channels can be destroyed when it retires (an assignment over a
  // channel would leave its registration behind).
  std::vector<CkDeviceChannel*> tok_ch;
  // Output channels (out_channels): this dispatcher owns one per (expert, copy
  // rank, step parity) over the expert's fixed segment of the receive slab of
  // that parity, hands them to the copy that serves it, takes each put's note
  // (outputArrived) and releases the segment once the step is combined. The
  // 64 outputs a step then cost the dispatcher no landing copy, no post entry
  // and no acknowledgement (measured 2026-11-06: an output waited 16 to 30 ms
  // between its envelope and its consumption on that path).
  std::vector<CkDeviceChannel> out_ch;
  std::vector<char> out_open;
  std::vector<int> out_from;   // per (parity, expert): the copy whose put landed this step, -1 none
  std::vector<int> out_landed;   // split routing: per (expert, copy, parity) channel, the step + 1 of the put landed (0: none)
  // Split routing: the job-wide demand of each step, summed by dispatcher 0 from every dispatcher's counts
  // (countsIn) and sent back (demandIn); counts go out two steps ahead, so a step's split is ready before it is routed.
  std::map<int, std::vector<int>> dem_acc;   // dispatcher 0: the counts matrix of a step being collected
  std::map<int, int> dem_cnt;
  std::set<int> plan_pending;               // dispatcher 0: steps whose counts are in, waiting for an agreement
  std::map<int, MoeStepPlan> plans;         // every PE: the plans of the steps around this one
  std::vector<CkCallback> plan_waiters;     // experts here waiting for a plan
  const MoeStepPlan* planOf(int u) const { auto it = plans.find(u); return it == plans.end() ? nullptr : &it->second; }
  void waitPlan(const CkCallback& cb) { plan_waiters.push_back(cb); }
  int counts_sent_through = 0;
  std::vector<int> cnt_slots, cnt_tmp;
  int chans_in = 0;          // rank-0 channel sets held (init handshake)
  int chans_new_in = 0;      // rank > 0 channel sets taken by runStep so far
  int chans_new_needed = 0;  // ... and how many steps so far have needed one

  void* d_x;          // [n_tokens x d_model], fixed for the run; fp32 or bf16 (-B)
  float* d_y;         // [n_tokens x d_model], combined outputs
  double* d_chk;
  double* h_chk;
  // A send is tagged with a stream, and the runtime records "data ready" as
  // that stream's tail at send time. The comm stream's tail may be a landing
  // copy waiting on a remote PE, so the sends carry a stream of their own that
  // waits only on the event recorded right after the gather.
  hapiStream_t send_stream;
  hapiEvent_t ev_gather;
  hapiEvent_t ev_comb;   // after a step's combine: the step after next's gather reuses its buffers (sendEarly)
  // Which process's batch this branch owns, and whether it owns one at all:
  // only the first PE of a process dispatches. The other branches exist for
  // the compute context (lanes) the experts on their PE run in.
  int dispId = -1;
  bool active = false;
  // The last balancing round decided nothing (LBManager::LastRoundDecisions):
  // nothing migrates or changes its copies, so the tokens of the steps in its
  // window may go ahead after all. Reset at every balancing step.
  bool lb_round_idle = false;
  bool lb_round_open = false;       // a balancing round in flight: no lookahead
  int lb_round_close_step = 0;      // lookahead again from this step

  // A fixed load balancing object on this PE: the dispatcher's host time and
  // kernels are its own, not unattributed background that DiffusionLB ignores
  // and MetisLB reads as PE overhead.
  Dispatcher() { ckRegisterWithLB(); }

  size_t slots() const { return (size_t)n_tokens * top_k; }
  static int chIdx(int e, int r, int q) { return (e * K_MAX_COPIES + r) * 2 + q; }
  // Rows of the receive slab: with output channels every expert has a fixed
  // segment of cap_src rows (a channel's buffer is fixed), else the tokens'.
  size_t recvRows() const { return split_route ? (size_t)split_slack * split_rpc : out_channels ? recv_extra : slots(); }
  // Split routing: one segment per (expert, copy) and parity, a chunk of its own of up to split_rpc rows (an
  // output channel lands at its buffer's start, so copies of one expert cannot share a segment), allocated as the
  // copy is first routed to; chunk split_slack holds the slack and zero rows.
  size_t split_rpc = 0;
  int split_slack = 0;
  static int segChunk(int e, int r) { return e * K_MAX_COPIES + r; }
  void ensureRecvSeg(int e, int r) {
    const int c = segChunk(e, r);
    for (int q = 0; q < 2; q++) {
      StepBuf& b = sb[q];
      if (b.d_recv_chunks[c] != nullptr) continue;
      hapiCheck(hapiMalloc(&b.d_recv_chunks[c], actEsz() * (size_t)cap_e[e] * d_model));
      b.h_recv_bases[c] = b.d_recv_chunks[c];
      // The table only gains entries, so a copy that reads a later state of the mirror is as good; behind every
      // combine of this parity so far, ahead of the first that reads the new segment.
      hapiCheck(hapiMemcpyAsync(b.d_recv_bases, b.h_recv_bases, sizeof(void*) * b.d_recv_chunks.size(),
          hapiMemcpyHostToDevice, gpu.comm.stream));
    }
  }
  // With output channels every expert has its own segment of cap_e[e] rows, placed whole in a chunk of recv_rpc
  // rows; recv_extra is the first of the two extra rows (the n=0 slack and the zero row), also in one chunk.
  std::vector<size_t> recv_off;
  size_t recv_rpc = 0, recv_extra = 0;
  void layoutRecv(size_t esz) {
    if (!out_channels) { recv_rpc = slots() + 2; return; }
    recv_rpc = std::max((size_t)cap_src, ((size_t)1024 << 20) / ((size_t)d_model * esz));
    size_t row = 0;
    auto place = [&](size_t n) {
      if (row % recv_rpc + n > recv_rpc) row = (row / recv_rpc + 1) * recv_rpc;
      const size_t at = row;
      row += n;
      return at;
    };
    recv_off.assign(n_experts, 0);
    for (int e = 0; e < n_experts; e++) recv_off[e] = place((size_t)cap_e[e]);
    recv_extra = place(2);
  }
  void* recvRowPtr(StepBuf& b, size_t row) { return (char*)b.d_recv_chunks[row / b.rpc] + (row % b.rpc) * d_model * actEsz(); }
  void* recvSeg(int q, int e) { return recvRowPtr(sb[q], recv_off[e]); }
  void openOutChannelsFor(int e, int r, int late) {
    if (split_route) ensureRecvSeg(e, r);
    for (int q = 0; q < 2; q++) {
      CkDeviceChannel& c = out_ch[chIdx(e, r, q)];
      c = CkDeviceChannel(split_route ? sb[q].d_recv_chunks[segChunk(e, r)] : recvSeg(q, e),
                          (size_t)cap_e[e] * d_model * actEsz(),
                          CkCallback(CkIndex_Dispatcher::outputArrived(NULL), thisProxy[CkMyPe()]));
      c.open();
      out_open[chIdx(e, r, q)] = 1;
    }
    expert_proxy(e, r).takeOutChannels(dispId, late, out_ch[chIdx(e, r, 0)], out_ch[chIdx(e, r, 1)]);
    TRACE(2, "disp %d: output channels for expert %d copy %d opened%s", dispId, e, r, late ? " (late)" : "");
  }
  // A put's note: which channel, hence which expert and copy; the tag carries
  // the step and the copies the sender announces for the next one.
  void outputArrived(CkDataMsg* m) {
    CkDeviceChannelEvent ev;
    memcpy(&ev, m->getData(), sizeof(ev));
    int idx = -1;
    for (int i = 0; i < (int)out_ch.size(); i++)
      if (out_open[i] && out_ch[i].id() == ev.channel) { idx = i; break; }
    if (idx < 0)
      CkAbort("PE %d: an output note for channel %llx, which is not one of its dispatcher's\n", CkMyPe(),
          (unsigned long long)ev.channel);
    const CkDeviceChannelEvent* e = out_ch[idx].event(m);
    delete m;
    if (e == NULL) return;
    const int step = (int)(e->tag >> 8), mult_next = (int)(e->tag & 255);   // 8 bits of copy count (Expert::sendOutputs)
    const int ex = idx / (K_MAX_COPIES * 2), r = (idx / 2) % K_MAX_COPIES, q = idx & 1;
    if (q != (step & 1))
      CkAbort("PE %d: expert %d copy %d's output for step %d landed on the channel of parity %d\n", CkMyPe(), ex, r,
          step, q);
    if (flight_log) { env_t.resize(2 * n_experts); env_t[q * n_experts + ex] = moeNowNs(); }
    out_from[q * n_experts + ex] = r;
    // Which step's put: with the step after next's tokens out before this step is combined (sendEarly), a copy that
    // sent nothing here this step can land that step's output on the channel of this step's parity before this
    // step's release, and releasing it with this step's (by parity alone) let the copy overwrite rows not yet
    // combined (2026-11-09: two runs' output checksums off, one by 1%).
    if (split_route) out_landed[idx] = step + 1;
    thisProxy[CkMyPe()].outputIn(step, ex, (int)(e->bytes / (actEsz() * d_model)), mult_next, r);
  }
  // This step's segments were combined: the copies may put the step after next.
  void releaseOutputs() {
    const int q = my_step & 1;
    if (split_route) {
      for (int i = q; i < (int)out_landed.size(); i += 2)
        if (out_landed[i] == my_step + 1) { out_ch[i].release(); out_landed[i] = 0; }
      return;
    }
    for (int e = 0; e < n_experts; e++) {
      const int r = out_from[q * n_experts + e];
      if (r < 0) continue;
      out_ch[chIdx(e, r, q)].release();
      out_from[q * n_experts + e] = -1;
    }
  }
  // A retired copy has destroyed its holder copies: the channels close.
  void outChannelsFree(int e, int r) {
    for (int q = 0; q < 2; q++) {
      const int i = chIdx(e, r, q);
      if (!out_open[i]) continue;
      out_ch[i].close();
      out_open[i] = 0;
    }
    TRACE(2, "disp %d: output channels of expert %d copy %d closed", dispId, e, r);
  }

  void init() {
    // The experts declare their own loads (token count x measured seconds per
    // token); with instrumentation on, the balancer would overwrite them with
    // CUPTI's measurement, and pay CUPTI's cost every step. -I keeps it.

    dispId = -1;
    for (int d = 0; d < n_disp; d++) if (disp_pes[d] == CkMyPe()) dispId = d;
    active = dispId >= 0;
    if (active) LBManagerObj()->setPartition(partitionOf(dispId));   // this PE feeds that partition of every expert

    memset(&gpu, 0, sizeof(gpu));
    gpu.n_lanes = n_lanes;
    // See MoeCtx in moe.h for why communication has its own stream.
    // Normal priority, as every stream here (2026-11-08): with the communication streams at the device's greatest
    // priority, Frontier's GPUs stalled at the first balancing rounds -- high-priority queues held barriers on
    // events whose producers sat behind a copy a normal-priority queue never launched (job 5888388, rocgdb).
    hapiCheck(hapiStreamCreateWithPriority(&gpu.comm.stream,
        hapiStreamNonBlocking, 0));
    hapiCheck(hapiEventCreate(&gpu.ref_ev));
    hapiCheck(hapiEventRecord(gpu.ref_ev, gpu.comm.stream));
    {   // the device's busy pool takes the first PE's reference event as its clock origin
      MoeDevPool* pool = moeDevPool();
      std::lock_guard<std::mutex> lk(pool->m);
      if (pool->ref_ev == nullptr) { hapiCheck(hapiEventSynchronize(gpu.ref_ev)); pool->ref_ev = gpu.ref_ev; }
    }
    hapiCheck(hapiMalloc((void**)&gpu.comm.partials,
        sizeof(double) * MOE_RED_BLOCKS));
    hapiCheck(hapiStreamCreateWithPriority(&send_stream, hapiStreamNonBlocking,
        0));   // normal priority (gpu.comm.stream)
    hapiCheck(hapiEventCreateWithFlags(&ev_gather, hapiEventDisableTiming));
    hapiCheck(hapiEventCreateWithFlags(&ev_comb, hapiEventDisableTiming));
    const size_t wsz = (size_t)d_model * d_ff;
    // A lane's buffers are scratch of this process's compute: never sent, never migrated. They come from plain
    // device memory, not the registered pool: with 2+ nodes every pool arena is registered with the fabric and a
    // process may hold at most 4096 MB of them, which the default size (-e 64 -m 2048 -h 8192 -t 8192) exceeded by
    // ~1.8 GB of lane scratch (4 lanes x ~450 MB) on top of its experts' 3.7 GB (2026-10-25).
    for (int l = 0; l < n_lanes; l++) {
      MoeGpu& g = gpu.lanes[l];
      hapiCheck(hapiStreamCreateWithPriority(&g.stream, hapiStreamNonBlocking,
          0));
      g.bf16 = use_bf16 != 0;
      hapiCheck(hapiMallocUnattributed((void**)&g.h,
          actEsz() * (size_t)chunk_tokens * d_ff));
      hapiCheck(hapiMallocUnattributed((void**)&g.dh,
          actEsz() * (size_t)chunk_tokens * d_ff));
      // Three accumulator pairs (moe.h): the four partitions' leaves, folded
      // into the two tree nodes as they are computed (Expert::treeNodes).
      for (int p = 0; p < 3; p++) {
        hapiCheck(hapiMallocUnattributed((void**)&g.acc1[p], sizeof(float) * wsz));
        hapiCheck(hapiMallocUnattributed((void**)&g.acc2[p], sizeof(float) * wsz));
      }
      if (fuse_sources) {
        // Bounded by what one expert can be sent in a step, the same bound its
        // own slabs use. Sized once: growing it would have to free a block the
        // lane's queued kernels may still be reading.
        const size_t frows = (size_t)n_disp * cap_src;
        hapiCheck(hapiMallocUnattributed((void**)&g.fx, actEsz() * frows * d_model));
        hapiCheck(hapiMallocUnattributed((void**)&g.fy, actEsz() * frows * d_model));
      }
      g.workspace_bytes = (size_t)32 << 20;
      hapiCheck(hapiMallocUnattributed(&g.workspace, g.workspace_bytes));
      hapiCheck(hapiMallocUnattributed((void**)&g.partials,
          sizeof(double) * MOE_RED_BLOCKS));
      if (moeBlasCreate(&g, use_tf32 != 0))
        CkAbort("hapiBlasCreate failed on PE %d lane %d\n", CkMyPe(), l);
    }

    // The experts' events, made now rather than on each arrival (moeEventGet): about three copies an expert, spread
    // over the PEs, twice over for a round's arrivals.
    moeEventWarm(std::max(8, 2 * (3 * n_experts + CkNumPes() - 1) / CkNumPes() + 4), n_disp + 2);
    if (!active) {
      // Context only; no batch, no steps. Still a member of the group's
      // init reduction.
      contribute(CkCallback(CkReductionTarget(Main, dispReady), main_proxy));
      return;
    }

    const size_t TK = slots();
    const size_t esz = actEsz();
    hapiCheck(hapiMalloc((void**)&d_x, esz * n_tokens * d_model));
    for (StepBuf& b : sb) {
      hapiCheck(hapiMalloc((void**)&b.d_send, esz * (TK + 1) * d_model));
      // TK+2 rows: [TK] absorbs the one-row dummy a zero-token expert sends,
      // [TK+1] is held at zero for tokens dropped by the capacity factor.
      // The receive slab in chunks. With output channels it holds a fixed segment of cap_src rows per
      // expert (OLMoE on 16 GCDs: 64 x 11424 rows x 4 KB = 3 GB a parity), and a NIC-registered pool arena
      // on Frontier holds at most 2048 MB (FRONTIER_START.md; the pool refused the allocation, 2026-11-07),
      // so the slab is allocated in chunks of whole segments of at most 1024 MB and a row is mapped to its
      // chunk (recvRowPtr, the combine kernel). One chunk, the layout as before, when it all fits.
      if (split_route) {
        if (split_rpc == 0) {
          for (int e = 0; e < n_experts; e++) split_rpc = std::max(split_rpc, (size_t)cap_e[e]);
          split_slack = n_experts * K_MAX_COPIES;
        }
        b.rpc = (int)split_rpc;
        const int nch = split_slack + 1;
        b.d_recv_chunks.assign(nch, nullptr);
        hapiCheck(hapiMalloc(&b.d_recv_chunks[split_slack], esz * 2 * d_model));
        hapiCheck(hapiMallocHost((void**)&b.h_recv_bases, sizeof(void*) * nch));
        for (int c = 0; c < nch; c++) b.h_recv_bases[c] = b.d_recv_chunks[c];
        hapiCheck(hapiMalloc((void**)&b.d_recv_bases, sizeof(void*) * nch));
        hapiCheck(hapiMemcpyAsync(b.d_recv_bases, b.h_recv_bases, sizeof(void*) * nch, hapiMemcpyHostToDevice, 0));
        hapiCheck(hapiStreamSynchronize(0));
        b.d_recv = b.d_recv_chunks[split_slack];
      } else {
        if (recv_rpc == 0) layoutRecv(esz);
        const size_t rows = recvRows() + 2;
        b.rpc = (int)std::min(rows, recv_rpc);
        const int nch = (int)((rows + b.rpc - 1) / b.rpc);
        b.d_recv_chunks.assign(nch, nullptr);
        for (int c = 0; c < nch; c++) {
          const size_t r = std::min((size_t)b.rpc, rows - (size_t)c * b.rpc);
          hapiCheck(hapiMalloc(&b.d_recv_chunks[c], esz * r * d_model));
        }
        b.d_recv = b.d_recv_chunks[0];
        hapiCheck(hapiMalloc((void**)&b.d_recv_bases, sizeof(void*) * nch));
        hapiCheck(hapiMemcpyAsync(b.d_recv_bases, b.d_recv_chunks.data(), sizeof(void*) * nch,
            hapiMemcpyHostToDevice, 0));
        hapiCheck(hapiStreamSynchronize(0));
        if (nch > 1 && dispId == 0 && &b == &sb[0])
          CkPrintf("Receive slab: %zu rows in %d chunks of %d rows (%.0f MB each)\n", rows, nch, b.rpc,
                   (double)b.rpc * d_model * esz / (1 << 20));
      }
      // The zero row, recvRows() + 1 (the last; TK + 1 without output channels), for the tokens a
      // capacity factor drops.
      hapiCheck(hapiMemset(recvRowPtr(b, recvRows() + 1), 0, esz * d_model));
    }
    for (RouteBuf& r : rb) {
      hapiCheck(hapiMalloc((void**)&r.d_send_idx, sizeof(int) * TK));
      hapiCheck(hapiMalloc((void**)&r.d_recv_pos, sizeof(int) * TK));
      hapiCheck(hapiMallocHost((void**)&r.h_send_idx, sizeof(int) * TK));
      hapiCheck(hapiMallocHost((void**)&r.h_recv_pos, sizeof(int) * TK));
      r.counts.assign(n_experts, 0);
      r.offs.assign(n_experts + 1, 0);
      r.nk.assign(n_experts, 1);
      r.soff.assign((size_t)n_experts * (K_MAX_COPIES + 1), 0);
      r.sord.assign((size_t)n_experts * K_MAX_COPIES, 0);
    }
    hapiCheck(hapiMalloc((void**)&d_y, sizeof(float) * n_tokens * d_model));
    hapiCheck(hapiMalloc((void**)&d_chk, sizeof(double)));
    hapiCheck(hapiMallocHost((void**)&h_chk, sizeof(double)));

    fill_.assign(n_experts, 0);
    slot_expert.assign(TK, 0);
    mult.assign(n_experts, 1);
    pend_mult.assign(n_experts, 0);
    pend_from.assign(n_experts, 0);
    // -X: the copy exists from the start; it takes effect at step 1 like any
    // announced change, so the channel wait below covers it too.
    if (rep_init_expert >= 0) pend_mult[rep_init_expert] = 2;
    tok_ch.assign(chIdx(n_experts, 0, 0), nullptr);
    if (out_channels && active) {
      out_ch.resize(chIdx(n_experts, 0, 0));
      out_open.assign(out_ch.size(), 0);
      out_from.assign(2 * n_experts, -1);
      out_landed.assign(out_ch.size(), 0);
      for (int e = 0; e < n_experts; e++)
        openOutChannelsFor(e, copyFor(dispId, e == rep_init_expert ? 2 : 1), 0);
    }
    LBManagerObj()->AddMigrationDoneFn(this, &Dispatcher::roundDone);

    // Unit-variance inputs, a pure function of (seed, PE, element); under -B
    // drawn in fp32 and rounded once to bf16, so the bf16 run's tokens are the
    // fp32 run's rounded.
    if (use_bf16) {
      float* tmp;
      hapiCheck(hapiMalloc((void**)&tmp, sizeof(float) * (size_t)n_tokens * d_model));
      moeInitUniform(tmp, (size_t)n_tokens * d_model,
          hmix(hmix(moe_seed, 1000), dispId), sqrtf(3.0f), gpu.comm.stream);
      moeCastBf16(tmp, (moe_bf16*)d_x, (size_t)n_tokens * d_model, gpu.comm.stream);
      hapiCheck(hapiStreamSynchronize(gpu.comm.stream));
      hapiCheck(hapiFree(tmp));
    } else {
      moeInitUniform((float*)d_x, (size_t)n_tokens * d_model,
          hmix(hmix(moe_seed, 1000), dispId), sqrtf(3.0f), gpu.comm.stream);
      hapiCheck(hapiStreamSynchronize(gpu.comm.stream));
    }
    if (use_attn) {
      // Lane 0 has the BLAS handle; the gather (send stream) waits for the sublayer through ev_attn.
      moeAttnInit(&gpu.lanes[0], &attn, n_tokens, d_model, 32, 1024, hmix(hmix(moe_seed, 2000), dispId));
      hapiCheck(hapiEventCreateWithFlags(&ev_attn, hapiEventDisableTiming));
    }
    contribute(CkCallback(CkReductionTarget(Main, dispReady), main_proxy));
  }

  // The hot set of a phase: a hashed permutation of the experts, with rank r
  // drawing a share proportional to (r+1)^-z.
  void buildPhase(int phase) {
    hotSetPermutation(phase, perm);
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
  // Each token's experts at `step` into sl (token-major, top_k a token) and the tokens per expert into counts,
  // before the capacity factor: a pure function of the step and this dispatcher.
  void selectExperts(int step, std::vector<int>& sl, std::vector<int>& counts) {
    const int E = n_experts, T = n_tokens, K = top_k;
    const int phase = drift_period > 0 ? (step - 1) / drift_period : 0;
    if (phase != cur_phase) {
      cur_phase = phase;
      buildPhase(phase);
    }
    std::fill(counts.begin(), counts.end(), 0);
    if (!trace_path.empty()) {
      if (trace_ids.empty()) loadTrace(trace_path, E, K, trace_ids);
      const size_t n = trace_ids.size() / K;
      const size_t g0 = ((size_t)(step - 1) * n_disp + dispId) * T;
      for (int t = 0; t < T; t++) {
        const int32_t* row = &trace_ids[((g0 + t) % n) * K];
        for (int j = 0; j < K; j++) {
          sl[(size_t)t * K + j] = row[j];
          counts[row[j]]++;
        }
      }
    } else
    for (int t = 0; t < T; t++) {
      int chosen[MOE_MAX_TOPK];
      for (int j = 0; j < K; j++) {
        int e = -1;
        for (int attempt = 0; attempt < 16 && e < 0; attempt++) {
          const uint64_t h = hmix(hmix(hmix(hmix(hmix(moe_seed, step),
              dispId), t), j), attempt);
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
        sl[(size_t)t * K + j] = e;
        counts[e]++;
      }
    }
  }
  // This dispatcher's tokens per expert at `step`, the capacity factor applied (what route() will send).
  void countStep(int step, std::vector<int>& c) {
    cnt_slots.resize(slots());
    c.assign(n_experts, 0);
    selectExperts(step, cnt_slots, c);
    if (capacity_factor > 0.0) {
      const int cap = std::max(1, (int)ceil(capacity_factor * (double)top_k * (double)n_tokens / (double)n_experts));
      for (int e = 0; e < n_experts; e++) c[e] = std::min(c[e], cap);
    }
  }
  // Split routing: this dispatcher's counts for the steps up to `through`, to dispatcher 0.
  void sendCounts(int through) {
    through = std::min(through, warmup_steps + n_steps);
    for (int u = counts_sent_through + 1; u <= through; u++) {
      countStep(u, cnt_tmp);
      // With the counts, what a token costs on this GPU lately: its experts' device-busy seconds over their tokens
      // (MoeDevPool), for the split to level time rather than tokens (CkReplicaSplit's peCost).
      double busy = 0.0, tok = 0.0;   // decayed sums (MoeDevPool): device-busy seconds and the tokens they covered
      {
        MoeDevPool* pool = moeDevPool();
        std::lock_guard<std::mutex> lk(pool->m);
        busy = pool->spt_gpu;
        tok = pool->spt_tok;
      }
      thisProxy[dispPe(0)].countsIn(u, dispId, n_experts, cnt_tmp.data(), busy, tok);
    }
    counts_sent_through = std::max(counts_sent_through, through);
  }
  // Dispatcher 0: one dispatcher's counts for step u; once every dispatcher's are in, the sums go to all.
  std::vector<double> disp_rate;   // dispatcher 0: each dispatcher's GPU's latest cost of a token
  std::vector<double> disp_busy, disp_tok;   // ... and its decayed device-busy seconds and tokens
  void countsIn(int u, int d, int n, int* c, double busy, double tok) {
    if (disp_rate.empty()) { disp_rate.assign(n_disp, 0.0); disp_busy.assign(n_disp, 0.0); disp_tok.assign(n_disp, 0.0); }
    if (tok > 0.0) { disp_rate[d] = busy / tok; disp_busy[d] = busy; disp_tok[d] = tok; }
    std::vector<int>& m = dem_acc[u];
    if (m.empty()) m.assign((size_t)n * n_disp, 0);
    for (int e = 0; e < n; e++) m[(size_t)e * n_disp + d] = c[e];
    if (++dem_cnt[u] < n_disp) return;
    dem_cnt.erase(u);
    tryPlan(u);
  }
  // Dispatcher 0: the plan of step u, once every expert's copies at u are agreed (the split is over the copies that
  // serve u); broadcast to every PE, for the dispatchers' routing and the copies' expectations.
  void tryPlan(int u) {
    const CkArrayID aid = expert_proxy.ckGetArrayID();
    for (int e = 0; e < n_experts; e++)
      if (CkReplicaSendCount(aid, e, u) < 0) {
        plan_pending.insert(u);
        CkReplicaWhenAgreed(aid, e, CkCallback(CkIndex_Dispatcher::planRetry(), thisProxy[CkMyPe()]));
        return;
      }
    plan_pending.erase(u);
    std::vector<int>& m = dem_acc[u];
    std::vector<long> tot(n_experts, 0);
    for (int e = 0; e < n_experts; e++)
      for (int d = 0; d < n_disp; d++) tot[e] += m[(size_t)e * n_disp + d];
    std::vector<std::vector<float>> frac;
    // MOE_SPLIT_COST=1: each GPU's measured cost of a token, on its PEs, for the split to level time. Off: it made
    // the split chase a lagging average that folds each copy's fixed cost into its tokens (2026-11-09, 16 GPUs: 67%
    // busy against 79% splitting tokens alone).
    static const bool by_tokens = getenv("MOE_SPLIT_COST") == NULL;
    std::vector<double> peCost(CkNumPes(), 0.0);
    bool any = false;
    if (!by_tokens && !disp_rate.empty())
      for (int p = 0; p < CkNumPes(); p++)
        for (int d = 0; d < n_disp; d++)
          if (hapiDeviceIndexOfPe(p) == hapiDeviceIndexOfPe(dispPe(d)) && CkNodeOf(p) == CkNodeOf(dispPe(d)) && disp_rate[d] > 0.0) {
            peCost[p] = disp_rate[d];
            any = true;
          }
    // A copy's fixed cost in tokens (MOE_SPLIT_PERCOPY, default 200: ~0.17 ms of a GEMM chain's setup and its
    // partial chunk, at ~0.85 us a token on an MI250X GCD in bf16; 2026-11-09).
    // Default 0: 200 and 500 lowered the steady state to 79% and 74% busy against 86% (2026-11-09).
    static const double per_copy = getenv("MOE_SPLIT_PERCOPY") ? atof(getenv("MOE_SPLIT_PERCOPY")) : 0.0;
    // MOE_SPLIT_BASE=1: each GPU's fixed time a step, in tokens, as its base in the split: the measured busy time
    // less what its tokens cost at the cheapest GPU's rate (the marginal cost; the rest is the fixed cost of its
    // copies' products and pieces), half of it (damping). At equal tokens the GPUs were 57-95% busy (2026-11-09).
    static const bool by_base = getenv("MOE_SPLIT_BASE") != NULL;
    std::vector<double> peBase;
    if (by_base && !disp_rate.empty()) {
      double a = 0.0;
      for (int d = 0; d < n_disp; d++) if (disp_rate[d] > 0.0 && (a == 0.0 || disp_rate[d] < a)) a = disp_rate[d];
      if (a > 0.0) {
        peBase.assign(CkNumPes(), 0.0);
        for (int d = 0; d < n_disp; d++)
          if (disp_tok[d] > 0.0) peBase[dispPe(d)] = 0.5 * std::max(0.0, disp_busy[d] / a - disp_tok[d]) * 0.25;
      }
    }
    // MOE_SPLIT_MIN (tokens, default 0): no copy gets a share below it (CkReplicaSplit's minShare).
    static const double min_share = getenv("MOE_SPLIT_MIN") ? atof(getenv("MOE_SPLIT_MIN")) : 0.0;
    CkReplicaSplit(aid, u, tot, frac, any ? &peCost : nullptr, per_copy, peBase.empty() ? nullptr : &peBase, min_share);
    std::vector<int> k(n_experts), x;
    for (int e = 0; e < n_experts; e++) {
      const int ke = (int)frac[e].size();
      k[e] = ke;
      // Integer targets that sum to the demand: floors, then the largest remainders.
      std::vector<std::pair<double, int>> rem;
      long left = tot[e];
      const size_t base = x.size();
      for (int r = 0; r < ke; r++) {
        const double want = frac[e][r] * (double)tot[e];
        const int f = (int)std::floor(want);
        x.push_back(f);
        left -= f;
        rem.push_back(std::make_pair(-(want - f), r));
      }
      std::sort(rem.begin(), rem.end());
      for (int i = 0; left > 0; i = (i + 1) % ke, left--) x[base + rem[i].second]++;
    }
    // Each expert's copies in the order of their hosts (unknown last), for the walk.
    std::vector<int> ord;
    for (int e = 0; e < n_experts; e++) {
      std::vector<std::pair<int, int>> h;
      for (int r = 0; r < k[e]; r++) {
        const int pe = CkReplicaWhere(aid, e, r);
        h.push_back(std::make_pair(pe >= 0 ? CmiPhysicalNodeID(pe) : INT_MAX, r));
      }
      std::stable_sort(h.begin(), h.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.first < b.first; });
      for (auto& hr : h) ord.push_back(hr.second);
    }
    thisProxy.planIn(u, n_experts, n_disp, (int)m.size(), m.data(), n_experts, k.data(), (int)x.size(), x.data(),
        (int)ord.size(), ord.data());
    dem_acc.erase(u);
  }
  void planRetry() {
    std::vector<int> us(plan_pending.begin(), plan_pending.end());
    for (int u : us) if (plan_pending.count(u)) tryPlan(u);
  }
  // Every PE: a step's plan.
  void planIn(int u, int ne, int nd, int nc, int* c, int nk, int* k, int nx, int* x, int no, int* ord) {
    MoeStepPlan& P = plans[u];
    P.c.assign(c, c + nc);
    P.k.assign(k, k + nk);
    P.x.assign(x, x + nx);
    P.ord.assign(ord, ord + no);
    P.xo.assign(ne + 1, 0);
    for (int e = 0; e < ne; e++) P.xo[e + 1] = P.xo[e] + P.k[e];
    while (!plans.empty() && plans.begin()->first < u - 32) plans.erase(plans.begin());
    std::vector<CkCallback> w;
    w.swap(plan_waiters);
    for (const CkCallback& cb : w) cb.send();
    if (active) thisProxy[CkMyPe()].chanArrived();   // the routing loop's wake-up (runStep)
    if (active) trySendEarly();
    if (active) tryPreGather();
  }
  void route(int step) {
    RouteBuf& sbuf = routeBuf(step);
    std::vector<int>& counts = sbuf.counts;
    std::vector<int>& offs = sbuf.offs;
    const int E = n_experts, T = n_tokens, K = top_k;
    (void)T;
    selectExperts(step, slot_expert, counts);
    // Capacity factor (-D), the GShard/Switch policy: an expert accepts at
    // most cap tokens from this dispatcher and the rest are dropped, which
    // bounds the imbalance without moving any state. Dropped tokens keep their
    // routing but take no slab slot; the combine reads them from a row held at
    // zero, so their contribution to the layer output is nothing. Tokens are
    // admitted in token order, so which ones are dropped is still a pure
    // function of the routing and independent of placement.
    long& dropped = sbuf.dropped;
    dropped = 0;
    if (capacity_factor > 0.0) {
      const int cap = std::max(1, (int)ceil(capacity_factor * (double)K *
                                            (double)T / (double)E));
      for (int e = 0; e < E; e++) {
        if (counts[e] > cap) { dropped += counts[e] - cap; counts[e] = cap; }
      }
    }
    offs[0] = 0;
    for (int e = 0; e < E; e++) {
      offs[e + 1] = offs[e] + counts[e];
      if (counts[e] > cap_e[e])
        CkAbort("PE %d step %d: %d tokens for expert %d exceed the slab "
                "capacity %d; increase -r\n", CkMyPe(), step, counts[e], e,
                cap_e[e]);
    }
    std::fill(fill_.begin(), fill_.end(), 0);
    const size_t TK = slots();
    const int zero_row = (int)recvRows() + 1;   // d_recv row kept at zero; recvRows() is the n=0 slack
    if (split_route) {
      // Each expert's run of the send slab divided between its copies by the runtime's split of the step's
      // job-wide demand; copy r takes [soff[r], soff[r + 1]) of the run, and its outputs land in its own segment.
      const MoeStepPlan* P = planOf(step);
      if (P == nullptr) CkAbort("PE %d: step %d routed before its plan was in\n", CkMyPe(), step);
      sbuf.outputs_expected = 0;
      for (int e = 0; e < E; e++) {
        const int k = P->k[e];
        if (k < 1 || k > K_MAX_COPIES) CkAbort("PE %d step %d: expert %d split over %d copies\n", CkMyPe(), step, e, k);
        if (P->c[(size_t)e * n_disp + dispId] != counts[e])
          CkAbort("PE %d step %d: expert %d has %d tokens here, the plan %d\n", CkMyPe(), step, e, counts[e],
              P->c[(size_t)e * n_disp + dispId]);
        int* so = &sbuf.soff[(size_t)e * (K_MAX_COPIES + 1)];
        moePlanOffsets(*P, n_disp, e, dispId, so);
        sbuf.nk[e] = k;
        for (int j = 0; j < k; j++) {
          sbuf.sord[(size_t)e * K_MAX_COPIES + j] = P->ord[P->xo[e] + j];
          if (so[j + 1] > so[j]) sbuf.outputs_expected++;
        }
      }
      for (size_t i = 0; i < TK; i++) {
        const int e = slot_expert[i];
        if (fill_[e] >= counts[e]) { sbuf.h_recv_pos[i] = zero_row; continue; }
        const int f = fill_[e]++;
        sbuf.h_send_idx[offs[e] + f] = (int)(i / K);
        const int* so = &sbuf.soff[(size_t)e * (K_MAX_COPIES + 1)];
        int j = 0;
        while (f >= so[j + 1]) j++;
        sbuf.h_recv_pos[i] = (int)((size_t)segChunk(e, sbuf.sord[(size_t)e * K_MAX_COPIES + j]) * split_rpc) + (f - so[j]);
      }
      return;
    }
    for (size_t i = 0; i < TK; i++) {
      const int e = slot_expert[i];
      if (fill_[e] >= counts[e]) { sbuf.h_recv_pos[i] = zero_row; continue; }
      const int pos = offs[e] + fill_[e];   // the send slab: compact, by expert
      sbuf.h_send_idx[pos] = (int)(i / K);
      // The receive slab: the same row of the expert's segment (fixed with
      // output channels, the send layout without).
      sbuf.h_recv_pos[i] = (out_channels ? (int)recv_off[e] : offs[e]) + fill_[e];
      fill_[e]++;
    }
  }

  // The copies announced for this step take effect: a new copy that serves
  // this dispatcher must send its channels before the step is routed
  // (chans_new_needed, waited for in runStep); a copy that has gone loses
  // its channels here and is told so, which lets it close them and leave.
  void beginStep() {
    my_step++;
    if (my_step == warmup_steps + 1) {
      // The balancer's first interval would otherwise span the start-up (the pool, the handles, the first
      // launches: on two A100 nodes a PE's host time over steps 1-20 read 0.10 s against 0.02 over steps 20-40,
      // and the host cap refused 43 placements on it, job 23646171). Cleared once per PE as the timed steps
      // begin, as sph3d does; the runtime clears them at the end of every round after that.
      static thread_local bool lb_loads_cleared = false;
      if (!lb_loads_cleared) { lb_loads_cleared = true; LBClearLoads(); }
    }
    recv_count = 0;
    early_wanted = false;   // the step after next is this step's lookahead now (runStep)
    // This fixed object's device load, declared like the experts' (none: its gather and combine are the same on
    // every GPU), so that a round with every object's load declared need not wait for the kernels in flight
    // (CentralLB::CallLB, CHARM_LB_DECLARED_NOWAIT).
    if (!use_instrument) ckSetGPUTime(0.0);
    // Counts go out plan_ahead steps early: a step's plan needs every dispatcher's. Further ahead delays a round's new
    // copies by as many steps (the plan is made over the copies agreed for its step) and gained nothing (2026-11-09:
    // four ahead, steady 69% busy against 80% at two; the copies that wait for a plan are those with no tokens).
    static const int plan_ahead = getenv("MOE_PLAN_AHEAD") ? atoi(getenv("MOE_PLAN_AHEAD")) : 2;
    if (split_route) sendCounts(my_step + plan_ahead);
    if (isLBStep(my_step)) { lb_round_idle = false; lb_round_open = true; }
    for (int e = 0; e < n_experts; e++) {
      if (pend_mult[e] == 0 || my_step < pend_from[e]) continue;
      const int old = mult[e];
      mult[e] = pend_mult[e];
      pend_mult[e] = 0;
      const int cOld = copyFor(dispId, old), cNew = copyFor(dispId, mult[e]);
      if (cOld == cNew) continue;   // the same copy serves this dispatcher either way
      // The copy now serving this dispatcher hands it channels, or a note
      // that it already has them (Expert::refreshChannels): one arrival per
      // change, waited for before this step is routed.
      if (tok_channels) chans_new_needed++;
      // The copy now serving this dispatcher gets its output channels: opened if they are not, else the open ones
      // again, since the copy of that rank may be a new one (2026-11-09: expert 29's copy 1, retired at step 25 and
      // made again at 110, waited at step 113 for output channels this dispatcher still held open for the old one).
      if (out_channels) {
        if (!out_open[chIdx(e, cNew, 0)]) openOutChannelsFor(e, cNew, 1);
        else expert_proxy(e, cNew).takeOutChannels(dispId, 1, out_ch[chIdx(e, cNew, 0)], out_ch[chIdx(e, cNew, 1)]);
      }
      TRACE(1, "disp step %d: expert %d has %d copies from this step; copy %d serves this dispatcher",
          my_step, e, mult[e], cNew);
      if (mult[e] < old && cOld == old - 1) {
        // The highest-ranked copy retires: its channels go before it does.
        for (int q = 0; q < 2; q++) {
          delete tok_ch[chIdx(e, cOld, q)];
          tok_ch[chIdx(e, cOld, q)] = nullptr;
        }
        expert_proxy(e, cOld).channelsDropped();
        TRACE(1, "disp step %d: expert %d: copy %d's channels dropped", my_step, e, cOld);
      }
    }
    TRACE(1, "disp step %d begin", my_step);
  }

  // Send the next step's tokens now, before this step's outputs are back.
  // They depend on nothing this step computes -- x is fixed and the routing
  // is a function of the step -- so their exchange overlaps the experts'
  // compute of this step instead of every GPU idling through it at the start
  // of the next (z 0.5, 2 nodes: 21-28 ms of a 98 ms step, GPUs 64% busy).
  // Through balancing rounds too (below): until 2026-11-08 it was off from
  // the balancing step to two steps past the release, idling every GPU
  // through the exchange of those steps.
  bool sendAhead() {
    const int u = my_step + 1;
    if (!send_ahead || u > warmup_steps + n_steps) return false;
    // On through balancing rounds (2026-11-08). A move takes an expert with tokens in flight to it through the
    // runtime's channel protocol (ckrdmadevice.h, "Migration, without application re-exchange": the senders' puts
    // land or are staged and forwarded) and its pup carries a later step's landed slab. Off only in the one step
    // whose successor changes an expert's copies: that successor is routed after runStep has the new copy's
    // channels (announced two steps ahead, noteOutput).
    for (int e = 0; e < n_experts; e++)
      if (pend_mult[e] != 0 && pend_from[e] == u) return false;
    if (!routable(u)) return false;
    return true;
  }
  // 1: send the next step now; 2: serving, the next step is not routable yet, send it when it is; 0: not at all.
  int ahead_mode = 0;
  int aheadMode() {
    if (sent_through >= my_step + 1) return 0;   // out already (sendEarly)
    if (sendAhead()) return 1;
    if (serve_only && send_ahead && my_step + 1 <= warmup_steps + n_steps) return 2;
    return 0;
  }
  // Serving under split routing (2026-11-09): a step's tokens are put right behind their gather on the send stream,
  // which is what the runtime takes as the data's readiness, with no host round trip through the gather's
  // completion; and the step after next goes out as soon as this step's last output is in (sendEarly). After a
  // step's last output the dispatcher took 2.1 ms (median; 6.4 at the 90th percentile) to put the tokens of the
  // step after next, through the combine's completion, the step's end and begin, and the gather's completion, each
  // kernel queued behind the experts' GEMMs on the device. Tokens are sent no further ahead than before: the step
  // after next, once every output of this one is in. MOE_NO_QUICK=1: as before.
  bool quickSend() const {
    static const bool off = getenv("MOE_NO_QUICK") != NULL;
    return !off && serve_only && split_route && tok_channels && !flight_log;
  }
  bool early_sent = false;   // the step after next went out from this step (its slab's sends are that step's)
  bool early_wanted = false; // ... and will, as soon as it can: its plan is in and this step's sends are done
  double dropped_report = 0.0;
  void sendEarly() {
    early_sent = false;
    early_wanted = false;
    if (!quickSend() || !send_ahead) return;
    const int u = my_step + 2;
    if (u > warmup_steps + n_steps || sent_through >= u || sent_through < my_step + 1) return;
    early_wanted = true;
    trySendEarly();
  }
  // From the combine on, until the next step begins: when the plan of the step after next arrives (planIn) or this
  // step's last send is confirmed (sendDone), rather than at the next step's begin, which waits for the combine to
  // come back from a device busy with the experts' products (2026-11-09: 4.6 ms on one GPU; a third of the
  // dispatchers missed the early send for a plan or a send still out at the combine).
  void trySendEarly() {
    if (!early_wanted) return;
    const int u = my_step + 2;
    if (sent_through >= u) { early_wanted = false; return; }
    if (stepBuf(my_step).outstanding != 0) return;   // this step's sends still read the slab u gathers into
    if (!routable(u)) return;
    early_wanted = false;
    issueGather(u);   // nothing to do if it was gathered ahead (tryPreGather)
    sendTokens();
    early_sent = true;
    TRACE(1, "disp step %d tokens of step %d sent with the combine", my_step, u);
  }
  // The step after next routed and gathered as soon as this step's own sends are done and its plan is in, so that
  // at this step's last output only its puts remain: they had waited for that gather, queued on a device busy with
  // the experts' products (2026-11-09: 2.6 ms from put to landing, median, against 0.2-0.6 before the puts went
  // straight behind their gather). Nothing is sent earlier than before.
  void tryPreGather() {
    static const bool off = getenv("MOE_NO_PREGATHER") != NULL;
    if (off || !quickSend() || !send_ahead) return;
    const int u = my_step + 2;
    if (u > warmup_steps + n_steps || sent_through != my_step + 1 || gathered_step >= u) return;
    if (stepBuf(my_step).outstanding != 0) return;   // this step's sends still read the slab u gathers into
    if (!routable(u)) return;
    issueGather(u);
    TRACE(2, "disp step %d step %d gathered ahead", my_step, u);
  }
  // From an expert on this GPU, resumed by the balancer: the round is over.
  void roundResumed() {
    if (!lb_round_open) return;
    lb_round_open = false;
    lb_round_close_step = my_step + 2;
    TRACE(1, "disp step %d: balancing round over, tokens go ahead from step %d", my_step, lb_round_close_step);
  }
  // Every PE's migration-done callback: the round is over here. With no
  // decision anywhere, the window's remaining steps need no pause.
  void roundDone() {
    const int n = LBManagerObj()->LastRoundDecisions();
    if (n == 0) lb_round_idle = true;
    TRACE(1, "disp step %d: balancing round done, %d decisions job-wide%s", my_step, n,
        n == 0 ? "; tokens go ahead again" : "");
  }

  // Route step `step` into its parity's buffers and gather its tokens. The
  // buffers were last used by step - 2, which was combined and whose sends
  // all completed (gateDrain) before step - 1 began.
  void issueGather(int step) {
    send_step = step;
    if (gathered_step == step) return;   // routed and gathered ahead (tryPreGather): the sends go behind it
    gathered_step = step;
    route(step);
    StepBuf& b = stepBuf(step);
    RouteBuf& rt = routeBuf(step);
    const size_t TK = slots();
    // The gather depends on nothing that lands on the comm stream (x is
    // fixed, the send slab is quiet once its previous step's sends drained),
    // so it runs on the send stream: the sends are then tagged with the very
    // stream that produced their data, and the gather never queues behind a
    // landing. d_recv_pos is read by the combine on the comm stream, which
    // runs only after every output of its step has landed there.
    hapiCheck(hapiMemcpyAsync(rt.d_send_idx, rt.h_send_idx, sizeof(int) * TK,
        hapiMemcpyHostToDevice, send_stream));
    hapiCheck(hapiMemcpyAsync(rt.d_recv_pos, rt.h_recv_pos, sizeof(int) * TK,
        hapiMemcpyHostToDevice, send_stream));
    const void* x_src = d_x;
    if (use_attn) {
      // The step's attention sublayer, on lane 0, before its tokens are gathered: trained on its own
      // loss unless serving; its output is what the MoE routes.
      moeAttnStep(&gpu.lanes[0], &attn, d_x, serve_only == 0, (float)(learning_rate / n_tokens));
      hapiCheck(hapiEventRecord(ev_attn, gpu.lanes[0].stream));
      hapiCheck(hapiStreamWaitEvent(send_stream, ev_attn, 0));
      x_src = attn.h;
    }
    moeGather(x_src, rt.d_send_idx, b.d_send, (int)TK, d_model, use_bf16 != 0, send_stream);
    hapiCheck(hapiEventRecord(ev_gather, send_stream));
    TRACE(2, "disp step %d gather of step %d issued", my_step, step);
    if (quickSend()) return;   // the sends go right behind it on this stream
    CkCallback* cb = new CkCallback(CkIndex_Dispatcher::gathered(),
        thisProxy[CkMyPe()]);
    hapiAddCallback(send_stream, cb);
  }

  // One slab per expert, empty ones included (the expert's step waits for one
  // message per dispatcher). len = max(n,1) rows so an empty message still
  // carries a valid buffer; the expert lands it in the unused slab. The slab
  // goes to the copy that serves this dispatcher.
  // The copy of expert e that serves this dispatcher at `step`: the runtime's in serving (2026-11-08), known once
  // routable(step) has said so.
  int routeOf(int e, int step) const {
    if (!serve_only) return copyFor(dispId, mult[e]);
    return CkReplicaRoute(expert_proxy.ckGetArrayID(), e, partitionOf(dispId), K_PART, step);
  }
  // Serving: every expert's route at `step` is agreed and its copy's channels are here (else a wake-up comes as
  // chanArrived: the agreement's callback, or the copy's channels); the output channels of a copy routed to for the
  // first time go out now.
  bool routable(int step) {
    if (!serve_only) {
      // Training: every copy this step routes to has handed its channels over. The count of arrivals against
      // changes (chans_new_in / chans_new_needed, runStep) is shared by all experts and was met by another copy's
      // arrival while a re-made copy's were still on their way (2026-11-09: expert 29's copy 1, retired at the
      // round at 20 and made again at 110, was routed to at step 113 with no channel, "no channel to copy 1").
      if (!tok_channels) return true;
      const int q = step % tokParities();
      for (int e = 0; e < n_experts; e++)
        if (tok_ch[chIdx(e, routeOf(e, step), q)] == nullptr) return false;
      return true;
    }
    const CkArrayID aid = expert_proxy.ckGetArrayID();
    const int q = step % tokParities();
    if (split_route) {
      // The step's demand (demandIn wakes the loop), and every copy of every expert agreed and reachable.
      if (step > counts_sent_through) sendCounts(step);
      const MoeStepPlan* P = planOf(step);
      if (P == nullptr) return false;   // planIn wakes the loop
      (void)aid;
      for (int e = 0; e < n_experts; e++) {
        const int k = P->k[e];
        for (int r = 0; r < k; r++) {
          if (tok_channels && tok_ch[chIdx(e, r, q)] == nullptr) return false;
          if (out_channels && !out_open[chIdx(e, r, 0)]) openOutChannelsFor(e, r, 1);
        }
      }
      return true;
    }
    for (int e = 0; e < n_experts; e++) {
      const int r = CkReplicaRoute(aid, e, partitionOf(dispId), K_PART, step);
      if (r < 0) {
        CkReplicaWhenAgreed(aid, e, CkCallback(CkIndex_Dispatcher::chanArrived(), thisProxy[CkMyPe()]));
        return false;
      }
      if (tok_channels && tok_ch[chIdx(e, r, q)] == nullptr) return false;
      if (out_channels && !out_open[chIdx(e, r, 0)]) openOutChannelsFor(e, r, 1);
    }
    return true;
  }
  void sendTokens() {
    const double ts0 = tnow();
    StepBuf& b = stepBuf(send_step);
    const RouteBuf& rt = routeBuf(send_step);
    const CkCallback done = (send_step & 1)
        ? CkCallback(CkIndex_Dispatcher::sendDoneOdd(), thisProxy[CkMyPe()])
        : CkCallback(CkIndex_Dispatcher::sendDoneEven(), thisProxy[CkMyPe()]);
    hapiCheck(hapiStreamWaitEvent(send_stream, ev_gather, 0));
    if (tok_channels) {
      // Written into the expert's slab by this GPU (same node) or one-sided
      // across nodes: on the node every GPU reading its peers at once shares
      // ~20 GB/s, writing ~50 GB/s (A40 x4, no NVLink). The put's size is the
      // token count; an empty one is a note alone. The tag is the step.
      const int q = send_step % tokParities();
      // MOE_TOK_FLUSH=1: each token put its own batch (the default lets the
      // runtime gather this entry's same-host puts into one copy).
      static const bool tok_flush = getenv("MOE_TOK_FLUSH") != NULL;
      if (split_route) {
        for (int e = 0; e < n_experts; e++) {
          const int* so = &rt.soff[(size_t)e * (K_MAX_COPIES + 1)];
          for (int j = 0; j < rt.nk[e]; j++) {
            if (so[j + 1] == so[j]) continue;
            const int r = rt.sord[(size_t)e * K_MAX_COPIES + j];
            CkDeviceChannel* ch = tok_ch[chIdx(e, r, q)];
            if (ch == nullptr) CkAbort("PE %d step %d: no channel to copy %d of expert %d\n", CkMyPe(), send_step, r, e);
            const int n = so[j + 1] - so[j];
            if (n == 0) continue;   // the copy knows from the plan that nothing comes from here
            TRACE(3, "disp %d step %d sends %d tokens of expert %d to copy %d", dispId, send_step, n, e, r);
            b.outstanding++;
            ch->put((char*)b.d_send + (size_t)(rt.offs[e] + so[j]) * d_model * actEsz(), actEsz() * (size_t)n * d_model,
                send_stream, done, (uint64_t)send_step);
            if (tok_flush) CkRdmaDeviceChannelBatchFlush();
          }
        }
      } else
      for (int e = 0; e < n_experts; e++) {
        const int n = rt.counts[e];
        const int r = routeOf(e, send_step);
        CkDeviceChannel* ch = tok_ch[chIdx(e, r, q)];
        TRACE(3, "disp %d step %d sends %d tokens of expert %d to copy %d", dispId, send_step, n, e, r);
        if (ch == nullptr)
          CkAbort("PE %d step %d: no channel to copy %d of expert %d\n", CkMyPe(), send_step, r, e);
        b.outstanding++;
        const size_t bytes = actEsz() * (size_t)n * d_model;
        if (flight_log) {
          // The destination PE is where copy r of expert e lives; recorded as the expert and the bytes.
          TokFlight* f = new TokFlight{this, send_step, e, -1, (long)bytes, moeNowNs()};
          ch->put((char*)b.d_send + (size_t)rt.offs[e] * d_model * actEsz(), bytes, send_stream,
              CkCallback(moeTokSentFn, (void*)f), (uint64_t)send_step);
        } else {
          ch->put((char*)b.d_send + (size_t)rt.offs[e] * d_model * actEsz(), bytes, send_stream, done,
              (uint64_t)send_step);
        }
        if (tok_flush) CkRdmaDeviceChannelBatchFlush();
      }
    } else {
      for (int e = 0; e < n_experts; e++) {
        const int n = rt.counts[e];
        const int len = (int)(std::max(n, 1) * d_model * actEsz());   // bytes
        expert_proxy(e, routeOf(e, send_step)).receiveTokens(send_step, dispId, n, moeNowNs(),
            CkMyPe(), len,
            (b.outstanding++,
             CkDeviceBuffer((char*)b.d_send + (size_t)rt.offs[e] * d_model * actEsz(), done,
                 send_stream)));
      }
    }
    sent_through = send_step;
    // CHARM_MOE_SLABPROBE=1 (fp32, the first two steps): the sum of squares of each
    // expert's slab as sent, to match against what the expert finds in its segments
    // (the diagnostic that found the truncated off-node landings of 2026-10-31).
    static const bool slab_probe = getenv("CHARM_MOE_SLABPROBE") != NULL;
    if (slab_probe && !use_bf16 && send_step <= 2) {
      for (int e = 0; e < n_experts; e++) {
        const int n = rt.counts[e];
        if (n <= 0) continue;
        hapiCheck(hapiMemsetAsync(d_chk, 0, sizeof(double), send_stream));
        moeSumSqAdd((const float*)((char*)b.d_send + (size_t)rt.offs[e] * d_model * actEsz()),
            (size_t)n * d_model, &gpu.comm, d_chk, send_stream);
        hapiCheck(hapiMemcpyAsync(h_chk, d_chk, sizeof(double), hapiMemcpyDeviceToHost, send_stream));
        hapiCheck(hapiStreamSynchronize(send_stream));
        CkPrintf("[SLAB] disp %d step %d expert %d n=%d sent sumsq %.17g\n", dispId, send_step, e, n, *h_chk);
      }
    }
    TRACE(1, "disp step %d tokens sent%s", my_step,
        send_step > my_step ? " for the next step" : "");
    TRACE(2, "disp step %d issued %d sends in %.1f ms", my_step, n_experts,
        (tnow() - ts0) * 1e3);
    tryPreGather();
  }

  // Post entry method: an expert's outputs land in its segment of the receive
  // slab of their step's parity. An empty message lands in the slack row past
  // the end. Outputs of the next step can arrive before this one is combined:
  // that step's tokens went out early, and an expert can finish it first.
  void receiveOutput(int ref, int e, int n, long long ts, int spe, int mult_next, int& len, char*& buf,
      CkDeviceBufferPost* devicePost) {
    if (flight_log) { env_t.resize(2 * n_experts); env_t[(ref & 1) * n_experts + e] = moeNowNs(); }
    if (ref < my_step || ref > sent_through)
      CkAbort("PE %d: outputs for step %d arrived during step %d (tokens sent "
          "through step %d)\n", CkMyPe(), ref, my_step, sent_through);
    StepBuf& b = stepBuf(ref);
    buf = (char*)recvRowPtr(b, n > 0 ? (size_t)routeBuf(ref).offs[e] : slots());
    devicePost[0].hapi_stream = gpu.comm.stream;
    TRACE(3, "disp step %d output expert %d n=%d posted", ref, e, n);
  }

  void setRate(double spt) { gpu.global_spt = spt; }

  void takeChannels(int e, int r, int late, CkDeviceChannel c0, CkDeviceChannel c1) {
    if (tok_ch.empty()) tok_ch.assign(chIdx(n_experts, 0, 0), nullptr);
    for (int q = 0; q < 2; q++) {
      delete tok_ch[chIdx(e, r, q)];
      tok_ch[chIdx(e, r, q)] = new CkDeviceChannel(q == 0 ? c0 : c1);
    }
    // One serving copy per expert hands this dispatcher its channels at
    // start-up, whichever copy that is (-X: copy 1 for its dispatchers); a
    // set that arrives later (a new copy's, or an existing copy's block that
    // grew) is what runStep waits for before routing the first step it
    // serves. -X's copy 1 counts as both.
    if (!late && ++chans_in == n_experts) main_proxy.dispChannelsReady();
    if (late || r > 0) thisProxy[CkMyPe()].chanArrived();
  }

  int outputsExpected() const { return split_route ? routeBuf(my_step).outputs_expected : n_experts; }
  void noteOutput(int e, int n, long long ts, int spe, int mult_next, int r = -1) {
    if (split_route) {
      const RouteBuf& sbn = routeBuf(my_step);
      const int* so = &sbn.soff[(size_t)e * (K_MAX_COPIES + 1)];
      int j = 0;
      while (j < sbn.nk[e] && sbn.sord[(size_t)e * K_MAX_COPIES + j] != r) j++;
      if (r < 0 || j >= sbn.nk[e] || n != so[j + 1] - so[j])
        CkAbort("PE %d step %d: expert %d copy %d returned %d rows\n", CkMyPe(), my_step, e, r, n);
      TRACE(3, "disp step %d output expert %d copy %d n=%d consumed", my_step, e, r, n);
      return;
    }
    if (flight_log)
      g_flights.push_back({'O', my_step, spe, CkMyPe(), e, (long)std::max(n, 1) * d_model * (long)actEsz(),
                           ts ? ts : env_t[(my_step & 1) * n_experts + e],   // a channel note carries no send time
                           env_t[(my_step & 1) * n_experts + e], moeNowNs()});
    TRACE(3, "disp step %d output expert %d n=%d consumed", my_step, e, n);
    if (n != routeBuf(my_step).counts[e])
      CkAbort("PE %d step %d: expert %d returned %d rows for %d tokens\n",
          CkMyPe(), my_step, e, n, routeBuf(my_step).counts[e]);
    // A count announced in step s's outputs serves from step s + 2 (Expert::computeStep, 2026-11-08): step s + 1's
    // tokens may already be out under the old count, which still serves it; beginStep switches at s + 2, and
    // sendAhead holds the one send that would route s + 2 before the new copy's channels are in.
    const int known = pend_mult[e] ? pend_mult[e] : mult[e];
    if (!serve_only && mult_next != known) {
      pend_mult[e] = mult_next;
      pend_from[e] = my_step + 2;
    }
  }

  void combine() {
    TRACE(1, "disp step %d outputs in", my_step);
    StepBuf& b = stepBuf(my_step);
    dropped_report = (double)routeBuf(my_step).dropped;
    moeCombine((const void* const*)b.d_recv_bases, b.rpc, routeBuf(my_step).d_recv_pos, d_y, n_tokens, top_k, d_model,
        1.0f / top_k, use_bf16 != 0, gpu.comm.stream);
    chk_due = checksum_freq > 0 && (my_step % checksum_freq == 0);
    if (chk_due) {
      hapiCheck(hapiMemsetAsync(d_chk, 0, sizeof(double), gpu.comm.stream));
      moeSumSqAdd(d_y, (size_t)n_tokens * d_model, &gpu.comm, d_chk, gpu.comm.stream);
      hapiCheck(hapiMemcpyAsync(h_chk, d_chk, sizeof(double),
          hapiMemcpyDeviceToHost, gpu.comm.stream));
    }
    CkCallback* cb = new CkCallback(CkIndex_Dispatcher::combined(),
        thisProxy[CkMyPe()]);
    hapiAddCallback(gpu.comm.stream, cb);
  }

  // This step's send slab is gathered into again two steps on; every send
  // from it must have confirmed completion first.
  void gateDrain() {
    if (out_channels) releaseOutputs();
    if (early_sent || stepBuf(my_step).outstanding == 0) thisProxy[CkMyPe()].sendsDrained();
    else drain_pending = true;
  }

  void sendDoneEven() { sendDone(0); }
  void sendDoneOdd() { sendDone(1); }
  void tokSent(TokFlight* f) {
    g_flights.push_back({'T', f->step, CkMyPe(), f->dpe, f->expert, f->bytes, f->t_issue, moeNowNs(), moeNowNs()});
    sendDone(f->step & 1);
    delete f;
  }
  void sendDone(int parity) {
    sb[parity].outstanding--;
    if (drain_pending && (my_step & 1) == parity && sb[parity].outstanding == 0) {
      drain_pending = false;
      thisProxy[CkMyPe()].sendsDrained();
    }
    if (early_wanted && (my_step & 1) == parity && sb[parity].outstanding == 0) trySendEarly();
    if ((my_step & 1) == parity && sb[parity].outstanding == 0) tryPreGather();
  }

  void endStep() {
    TRACE(1, "disp step %d end", my_step);
    if (chk_due) TRACE(2, "disp %d step %d out2 %.17g", dispId, my_step, *h_chk);
    // One slot per PE, summed elementwise: exact whatever order the reduction
    // combines contributions in, so the checksum is the same bits on every
    // run. Main adds the slots in PE order. (A plain sum_double of the values
    // differed in the last digit from run to run, with zero migrations.)
    // Reported point-to-point rather than through a group reduction: only the
    // active branch of each process runs steps, and a group reduction would
    // wait on the branches that never do. Main sums the reports in dispatcher
    // order, which is the same fixed-order sum the reduction's slots gave.
    main_proxy.dispStep(dispId, my_step, chk_due ? *h_chk : 0.0, dropped_report);
    if (my_step < warmup_steps + n_steps) {
      thisProxy[CkMyPe()].runStep();
    } else {
      if (flight_log) moeFlightDump();
      main_proxy.dispDone();
    }
  }
};

// ---------------------------------------------------------------------------

static void moeTokSentFn(void* param, void* msg) {
  TokFlight* f = (TokFlight*)param;
  ((Dispatcher*)f->disp)->tokSent(f);
  if (msg) CkFreeMsg(msg);
}

// An expert's pinned 8-byte checksum buffer, kept per PE for the whole run. hipHostFree (cudaFreeHost) synchronizes
// the whole device, and since 2026-11-08 the runtime moves an expert mid-step (a copy switches when its Stable buffers
// have landed), so its destructor runs while this GPU holds work that waits on other processes, which wait on this
// PE's host: a host blocked in that sync deadlocked the job (Frontier, job 5888388, PE 3 in hipHostFree under
// ~Expert from _deferredMigrateHandler). A returned buffer may still be the target of the expert's last checksum copy,
// so it is reused only once an event recorded behind that copy has completed. (hipHostMalloc is no sync, but it pins
// pages; reuse also takes it off every arrival.)
struct MoePinnedParked { double* p; hapiEvent_t ev; };
static thread_local std::vector<MoePinnedParked> moe_pinned_parked;
static thread_local std::vector<double*> moe_pinned_free;
static double* moePinnedChkGet() {
  for (size_t i = 0; i < moe_pinned_parked.size();) {
    if (hapiEventQuery(moe_pinned_parked[i].ev) == hapiSuccess) {
      moeEventPut(moe_pinned_parked[i].ev, false);
      moe_pinned_free.push_back(moe_pinned_parked[i].p);
      moe_pinned_parked[i] = moe_pinned_parked.back();
      moe_pinned_parked.pop_back();
    } else {
      i++;
    }
  }
  if (!moe_pinned_free.empty()) { double* p = moe_pinned_free.back(); moe_pinned_free.pop_back(); return p; }
  double* p = nullptr;
  hapiCheck(hapiMallocHost((void**)&p, sizeof(double)));
  return p;
}
static void moePinnedChkPut(double* p, hapiStream_t behind) {
  if (p == nullptr) return;
  hapiEvent_t ev = moeEventGet(false);
  hapiCheck(hapiEventRecord(ev, behind));
  moe_pinned_parked.push_back(MoePinnedParked{p, ev});
}

class Expert : public CBase_Expert {
  Expert_SDAG_CODE

 public:
  int my_step = 0;
  int recv_count = 0;
  int n_expect = 0;   // slabs this step waits for: one per dispatcher this copy serves
  // 0: receiving slabs (x segments live); 1: kernels issued (y segments
  // live); 2: outputs sent (nothing live but the weights)
  int phase = 0;
  std::vector<int> n_src;   // rows received from each dispatcher, -1 = none
  std::vector<long long> env_t;   // -F: envelope arrival per (step mod 3, dispatcher) (not migrated)
  int n_tot = 0;            // tokens this copy received this step
  int outstanding_sends = 0;
  bool drain_pending = false;
  bool chk_due = false;
  // The outputs went out from compute(), each dispatcher's behind the chunk
  // that finished its tokens (stream_outputs); sendOutputs only drains.
  bool outputs_sent = false;
  int cur_lane = -1;      // see pickLane; not migrated: a new PE picks afresh
  long lane_claim = 0;    // tokens this expert holds against cur_lane
  std::vector<hapiEvent_t> ev_seg;   // per dispatcher: its outputs are in y_seg
  // Receive channels on the input slab segments, [parity * n_disp + dispatcher]
  // (tok_channels), opened for the dispatchers this copy serves. Released once
  // the step's kernels are done with the slab.
  std::vector<CkDeviceChannel> tok_ch;
  std::vector<char> ch_open;
  // Holder copies of the dispatchers' output channels (out_channels), per
  // (step parity, dispatcher): the step's outputs are put into them. Pointers,
  // destroyed when the copy retires or dies (never assigned over); they travel
  // with the copy and re-resolve after a migration.
  std::vector<CkDeviceChannel*> out_ch;
  std::vector<char> out_have;
  bool out_by_pup = false;
  // The block-sum exchange over channels (out_channels, training): this copy
  // owns a channel per (peer slot, step parity) over its landing pair
  // dWpeer[slot][q] (one contiguous buffer of two blocks) and hands it to
  // that peer; it holds copies of the peers' channels and puts its own pair
  // into them. The peer's figures (tokens, PE, device time, retiring) go as
  // the small peerStats message serving already uses.
  std::vector<CkDeviceChannel> grad_ch;     // (slot, parity), owner side
  std::vector<char> grad_open;
  std::vector<CkDeviceChannel*> grad_hold;  // (peer rank, parity), holder copies
  std::vector<char> grad_have;              // per peer rank
  std::vector<char> grad_holder_alive;      // per peer rank: the peer still holds copies of this copy's channels
  bool grad_by_pup = false;
  // Per channel: a put has landed and its note was taken, but the step loop
  // has not consumed it (its slabIn is queued for a later step). Its bytes are
  // in the slab only, so a migration has to carry that segment: the runtime
  // holds a migration for a device message until it is consumed, but slabIn
  // carries no device buffer.
  std::vector<char> slab_held;
  bool arrived_by_pup = false;   // the channels reopen in ckJustMigrated

  // Async LB state: the step was joined at lb_start_step and AtSyncWait is
  // owed lb_wait_lag steps later.
  bool lb_waiting = false;
  int lb_start_step = 0;
  // AtSync or AtSyncWait was called and ResumeFromSync is owed; a resume the
  // element did not ask for (a copy inserted during a step's resume) is
  // ignored.
  bool awaiting_resume = false;
  // This step's resumed() came from the balancer releasing the element: the
  // step in which a clone or retire decision takes effect.
  bool resumed_by_lb = false;

  // Load estimate: seconds per token, an EMA of the measured step time over
  // its token count. Rate-aware for free: a slower device shows up here.
  double spt = 0.0;
  double gpu_last = 0.0;   // measured device time of the last sampled step
  int n_last = 0;
  int step_last = 0;       // the step those events were recorded in
  int adam_t = 0;
  bool ev_pending = false;

  // ---- replication (see partitionOf above) ----
  int nc = 1;                   // copies of this expert now, this one included
  bool replicated = false;      // nc > 1: the other copies serve their blocks of partitions
  int nc_next = 1;              // copies from step nc_from (mult_next, announced in the outputs)
  int nc_from = 0;              // the first step nc_next serves: two after the step that decided it
  bool retire_me = false;       // this copy goes when nc_next takes effect
  std::vector<int> res_at = std::vector<int>(K_MAX_COPIES, -1);   // the step copy r tagged its exchange with its resume
  int gp_in = 0;                // the SDAG loop over the other copies' block sums
  float* dWown1 = NULL;         // this copy's block sum of the step's gradient (its leaves folded)
  float* dWown2 = NULL;
  float* dWownS1 = NULL;        // scratch for the block's second leaf
  float* dWownS2 = NULL;
  // The other copies' block sums, by peer slot (slotOf) and step parity: the
  // next step's may land while this step's is read.
  std::vector<std::array<float*, 2>> dWpeer1 = std::vector<std::array<float*, 2>>(K_MAX_COPIES - 1, {{nullptr, nullptr}});
  std::vector<std::array<float*, 2>> dWpeer2 = std::vector<std::array<float*, 2>>(K_MAX_COPIES - 1, {{nullptr, nullptr}});
  bool grad_in_flight = false;  // the block sum's sends have not all completed
  int grad_bufs_done = 0;
  int grad_bufs_expected = 0;
  // The balancer's decisions for this round, known to every copy of the group
  // (ckClone / ckRetire on the copy concerned, the notices on the others):
  // copies of the ranks marked in decided_new are made (clone_decided), or
  // the copy of rank decided_rank retired (retire_decided), at the end of
  // the step the round releases us in. A round may add several copies at
  // once; their ranks run on from nc.
  bool clone_decided = false;
  bool retire_decided = false;
  int decided_rank = -1;
  std::vector<char> decided_new = std::vector<char>(K_MAX_COPIES, 0);
  bool clone_in_flight = false; // the clones' state sends have not all completed
  int clone_bufs_done = 0;
  int clone_bufs_expected = 0;
  int n_clone = 0;              // clone decisions this copy sources, waiting for their step
  std::vector<int> clone_to_pe = std::vector<int>(K_MAX_COPIES, 0);
  std::vector<int> clone_rank = std::vector<int>(K_MAX_COPIES, 0);
  bool retire_pending = false;  // a retire decision waiting for its step
  bool retiring = false;        // this step is this copy's last
  int mult_next = 1;            // copies serving the next step, told to the dispatchers
  // What the exchange costs this copy, measured: the wait from its compute's
  // end to the last of the other copies' block sums, summed over the steps
  // since the last balancing point (declared per peer at the next one; the
  // runtime's calibrated table prices a copy that has not exchanged yet).
  double xchg_wait_s = 0.0;
  int xchg_waits = 0;
  double t_compute_done = 0.0;
  // The other copies' step, by rank: tokens, PE, device time, whether it retires.
  std::vector<long> peer_n = std::vector<long>(K_MAX_COPIES, 0);
  std::vector<int> peer_pe = std::vector<int>(K_MAX_COPIES, -1);
  std::vector<double> peer_gpu = std::vector<double>(K_MAX_COPIES, 0.0);
  std::vector<char> peer_ret = std::vector<char>(K_MAX_COPIES, 0);
  bool exchange_free_due = false;   // alone again: the exchange buffers go once the last partial's send is done
  int chans_dropped = 0;        // dispatchers that dropped their channels to this retiring copy
  bool step_ended = false;      // retiring: the last step is over (teardown waits for both)
  bool started = false;         // init() ran (an original) or the state is in (a new copy)

  float* W1;   // [d_model x d_ff], the master weights (fp32 always)
  float* W2;   // [d_ff x d_model]
  // -B: the bf16 copies the GEMMs read, rewritten by every update in its own
  // pass; wb_stale says the master weights are new on this device (init, a
  // migration, a clone's state) and the copies are remade before the next
  // compute. Not migrated: 4 bytes of transfer saved per 2 of cast.
  moe_bf16* W1b;
  moe_bf16* W2b;
  bool wb_stale = false;
  float* m1;   // Adam moments, -A only
  float* v1;
  float* m2;
  float* v2;
  char* x_seg;    // [n_disp x cap_src x d_model] elements of actEsz() bytes, one
                  // segment per dispatcher; two such slabs by step parity with
                  // send_ahead (xSeg)
  char* y_seg;    // same layout, outputs
  double* d_chk;
  double* h_chk;
  hapiEvent_t ev_start;
  hapiEvent_t ev_end;
  // Recorded after the checksum's device-to-host copy. A pack reads h_chk on
  // the host, which the copy may still be writing; pup_device_order only
  // orders DEVICE-mode copies and leaves that host read racing. Losing it
  // costs one expert's slot in the w2 reduction, seen once in eight async
  // runs at Llama size as a deficit of exactly 1/64.
  hapiEvent_t ev_chk;
  hapiEvent_t ev_grad;   // the partial is complete on the lane (replicated)
  // A replicated copy's exchange and update run here, not on the lane it computed
  // on: the lane is released at computeDone and another expert's chunks may be
  // queued on it by the time the peers' block sums are in (A100, job 23601306:
  // an apply waited 20 ms behind them). Stream and checksum partials only.
  MoeGpu xg;
  // Same reason as the dispatcher's: the lane's tail at send time includes
  // other experts' queued GEMMs, and a landing that waited on it stalled the
  // receiver's comm stream for a remote expert's compute. This stream waits
  // only on ev_end, recorded after this expert's own kernels.
  hapiStream_t send_stream;

  Expert() {
    usesAtSync = true;
    setReplicaManaged(serve_only != 0);   // serving: the runtime makes, switches and retires the copies
    n_src.assign(n_disp, -1);
    W1 = W2 = m1 = v1 = m2 = v2 = NULL;
    W1b = W2b = NULL;
    x_seg = y_seg = NULL;
    d_chk = NULL;
    h_chk = NULL;
    memset(&xg, 0, sizeof(xg));
  }

  Expert(CkMigrateMessage* m) {
    usesAtSync = true;
    setReplicaManaged(serve_only != 0);
    W1 = W2 = m1 = v1 = m2 = v2 = NULL;
    W1b = W2b = NULL;
    x_seg = y_seg = NULL;
    d_chk = NULL;
    h_chk = NULL;
    memset(&xg, 0, sizeof(xg));
  }

  ~Expert() {
    if (W1 == NULL) return;   // a copy destroyed before its state arrived
    // Under async LB this runs mid-step on a migrating element: kernels may
    // still be in flight on the shared stream. Settle them before freeing.
    const double t0 = tnow();
    std::vector<void*> bufs = {W1, W2, W1b, W2b, m1, v1, m2, v2, x_seg, y_seg, dWown1, dWownS1};   // the pairs' second halves are the firsts' tails
    for (int sl = 0; sl < K_MAX_COPIES - 1; sl++)
      for (int q = 0; q < 2; q++) bufs.push_back(dWpeer1[sl][q]);
    if (hostSyncMode() || !CkDevicePoolOn()) {
      hapiCheck(hapiStreamSynchronize(lane()->stream));
      TRACE(1, "expert %d.%d dtor step %d phase %d synced after %.3f ms",
          expert(), rank(), my_step, phase, (tnow() - t0) * 1e3);
    } else {
      // Stream-ordered frees instead of a host wait: each block is noted as
      // read on this stream, and the pool parks it until the work queued
      // here retires. The pack copies' own note (migration stream) stays
      // alongside; the pool honours every stream a block was noted on.
      for (void* b : bufs)
        if (b) hapiDevPoolNoteRead(b, lane()->stream);
      if (d_chk) hapiDevPoolNoteRead(d_chk, lane()->stream);
      TRACE(1, "expert %d.%d dtor step %d phase %d stream-ordered", expert(), rank(),
          my_step, phase);
    }
    moeEventPut(ev_start, true);
    moeEventPut(ev_end, true);
    moeEventPut(ev_chk, false);
    moeEventPut(ev_grad, false);
    for (hapiEvent_t e : ev_seg) moeEventPut(e, false);
    ev_seg.clear();
    // Nothing here waits for the device (moePinnedChkGet): the streams go back to the runtime's pool with whatever
    // is queued on them, the pinned buffer is parked behind the send stream, and the exchange scratch is freed
    // stream-ordered. Only without the pool (or in the host-sync mode) does the scratch wait for its stream.
    // Behind the lane: the checksum's copy into it is on the lane stream (applyUpdate), not the send stream, and a
    // buffer handed on while that copy was still to land took it into its next owner's checksum.
    moePinnedChkPut(h_chk, lane()->stream);
    h_chk = NULL;
    if (xg.partials) {
      if (hostSyncMode() || !CkDevicePoolOn()) hapiCheck(hapiStreamSynchronize(xg.stream));
      else hapiDevPoolNoteRead(xg.partials, xg.stream);
      hapiCheck(hapiFree(xg.partials));
      xg.partials = NULL;
    }
    hapiReleaseStream(send_stream);
    send_stream = NULL;
    if (xg.stream) { hapiReleaseStream(xg.stream); xg.stream = NULL; }
    if (outstanding_sends != 0) {
      // A transport may still be reading a send buffer. Leak rather than pull
      // memory out from under it -- the pic2d convention.
      if (getenv("CHARM_DEBUG_MIGRATE"))
        CkPrintf("[%d] expert %d.%d leaking device buffers at destruction: %d "
                 "sends outstanding\n", CkMyPe(), expert(), rank(), outstanding_sends);
      return;
    }
    for (void* b : bufs)
      if (b) hapiCheck(hapiFree(b));
    hapiCheck(hapiFree(d_chk));
    for (CkDeviceChannel* c : out_ch) delete c;
    out_ch.clear();
    for (CkDeviceChannel* c : grad_hold) delete c;
    grad_hold.clear();
  }

  int expert() const { return thisIndex.x; }
  int rank() const { return thisIndex.y; }
  void ensureOutCh() {
    if (out_ch.empty()) { out_ch.assign(2 * n_disp, nullptr); out_have.assign(n_disp, 0); }
  }
  static int gIdx(int slot, int q) { return slot * 2 + q; }
  static int hIdx(int r, int q) { return r * 2 + q; }
  int rankOfSlot(int slot) const { return slot < rank() ? slot : slot + 1; }
  void ensureGradCh() {
    if (grad_ch.empty()) {
      grad_ch.resize((K_MAX_COPIES - 1) * 2);
      grad_open.assign(grad_ch.size(), 0);
      grad_hold.assign(K_MAX_COPIES * 2, nullptr);
      grad_have.assign(K_MAX_COPIES, 0);
      grad_holder_alive.assign(K_MAX_COPIES, 0);
    }
  }
  bool gradChannelsOn() const { return out_channels && !serve_only; }
  // This copy's channels for peer r, over its landing pair for r, handed to r.
  void openGradChannelsFor(int r) {
    ensureGradCh();
    const int sl = slotOf(r);
    for (int q = 0; q < 2; q++) {
      CkDeviceChannel& c = grad_ch[gIdx(sl, q)];
      c = CkDeviceChannel(dWpeer1[sl][q], 2 * sizeof(float) * wsz(),
                          CkCallback(CkIndex_Expert::gradArrived(NULL), self()));
      c.open();
      grad_open[gIdx(sl, q)] = 1;
    }
    grad_holder_alive[r] = 1;
    peerAt(r).takeGradChannels(rank(), grad_ch[gIdx(sl, 0)], grad_ch[gIdx(sl, 1)]);
    TRACE(2, "expert %d.%d step %d: exchange channels for copy %d opened", expert(), rank(), my_step, r);
  }
  void storeGradChannels(int from, const CkDeviceChannel& c0, const CkDeviceChannel& c1) {
    ensureGradCh();
    delete grad_hold[hIdx(from, 0)];
    delete grad_hold[hIdx(from, 1)];
    grad_hold[hIdx(from, 0)] = new CkDeviceChannel(c0);
    grad_hold[hIdx(from, 1)] = new CkDeviceChannel(c1);
    grad_have[from] = 1;
    TRACE(2, "expert %d.%d step %d: copy %d's exchange channels held", expert(), rank(), my_step, from);
  }
  // Every peer of this step's group has handed this copy channels.
  bool gradChannelsReady() {
    if (!gradChannelsOn() || !replicated) return true;
    ensureGradCh();
    for (int r = 0; r < nc; r++)
      if (r != rank() && !grad_have[r]) return false;
    return true;
  }
  // This copy destroys its holder copies of peer r's channels and says so.
  int grad_drop_after = -1;   // a retired peer's holders, dropped once this copy's put into them is done
  void dropGradAfterSend() {
    if (grad_drop_after < 0) return;
    const int r = grad_drop_after;
    grad_drop_after = -1;
    dropGradHolders(r);
  }
  void dropGradHolders(int r) {
    if (grad_hold.empty() || !grad_have[r]) return;
    delete grad_hold[hIdx(r, 0)];
    delete grad_hold[hIdx(r, 1)];
    grad_hold[hIdx(r, 0)] = grad_hold[hIdx(r, 1)] = nullptr;
    grad_have[r] = 0;
    peerAt(r).gradChannelsDropped(rank());
  }
  // Peer `from` destroyed its holder copies of this copy's channels: they close.
  void gradChannelsDropped(int from) {
    ensureGradCh();
    const int sl = slotOf(from);
    for (int q = 0; q < 2; q++)
      if (grad_open[gIdx(sl, q)]) { grad_ch[gIdx(sl, q)].close(); grad_open[gIdx(sl, q)] = 0; }
    grad_holder_alive[from] = 0;
    TRACE(2, "expert %d.%d step %d: copy %d dropped this copy's exchange channels", expert(), rank(), my_step, from);
    if (retiring) maybeFinishRetire();
    else if (exchange_free_due && !grad_in_flight && noGradHolders()) { freeExchange(); exchange_free_due = false; }
  }
  bool noGradHolders() const {
    for (char a : grad_holder_alive) if (a) return false;
    return true;
  }
  // A peer's block-sum pair landed: which peer, which step.
  void gradArrived(CkDataMsg* m) {
    CkDeviceChannelEvent ev;
    memcpy(&ev, m->getData(), sizeof(ev));
    int idx = -1;
    for (int i = 0; i < (int)grad_ch.size(); i++)
      if (grad_open[i] && grad_ch[i].id() == ev.channel) { idx = i; break; }
    if (idx < 0)
      CkAbort("Expert %d.%d on PE %d: a block-sum note for channel %llx, which is not one of its own\n", expert(),
          rank(), CkMyPe(), (unsigned long long)ev.channel);
    const CkDeviceChannelEvent* e = grad_ch[idx].event(m);
    delete m;
    if (e == NULL) return;
    const int step = (int)e->tag, sl = idx / 2, q = idx & 1;
    if (q != (step & 1))
      CkAbort("Expert %d.%d on PE %d: a block sum for step %d landed on the channel of parity %d\n", expert(), rank(),
          CkMyPe(), step, q);
    self().gradIn(step, rankOfSlot(sl));
  }
  // The step's landed pairs were applied: the peers may put the step after next.
  void releaseGrads() {
    const int q = my_step & 1;
    for (int r = 0; r < nc; r++)
      if (r != rank() && grad_open[gIdx(slotOf(r), q)]) grad_ch[gIdx(slotOf(r), q)].release();
  }
  void storeOutChannels(int d, const CkDeviceChannel& c0, const CkDeviceChannel& c1) {
    ensureOutCh();
    delete out_ch[d];
    delete out_ch[n_disp + d];
    out_ch[d] = new CkDeviceChannel(c0);
    out_ch[n_disp + d] = new CkDeviceChannel(c1);
    out_have[d] = 1;
    TRACE(2, "expert %d.%d step %d: output channels of dispatcher %d held", expert(), rank(), my_step, d);
  }
  // Every dispatcher this copy serves this step has handed it channels.
  bool outChannelsReady() {
    if (!out_channels) return true;
    ensureOutCh();
    for (int d = 0; d < n_disp; d++)
      if (serves(d) && !out_have[d]) return false;
    return true;
  }
  void dropOutChannels() {
    for (int d = 0; d < n_disp; d++) {
      if (out_ch.empty() || !out_have[d]) continue;
      delete out_ch[d];
      delete out_ch[n_disp + d];
      out_ch[d] = out_ch[n_disp + d] = nullptr;
      out_have[d] = 0;
      disp_proxy[dispPe(d)].outChannelsFree(expert(), rank());
    }
  }
  // Whether this copy serves dispatcher d this step.
  int ncopies() const { return nc; }
  // The landing slot of copy r's block sum (the copies other than this one, in rank order).
  int slotOf(int r) const { return r < rank() ? r : r - 1; }
  // The block (copy) that holds partition p under k copies.
  static int blockOfPartition(int p, int k) {
    for (int r = 0; r < k; r++) if (p < blockHi(r, k)) return r;
    return k - 1;
  }
  void setCopies(int k) {
    nc = k;
    replicated = k > 1;
    if (replicated) allocateExchange(k);
    if (replicated && gradChannelsOn()) {
      ensureGradCh();
      for (int r = 0; r < k; r++)
        if (r != rank() && !grad_open[gIdx(slotOf(r), 0)]) openGradChannelsFor(r);
    }
  }
  long peerTokens() const {
    long n = 0;
    for (int r = 0; r < nc; r++) if (r != rank()) n += peer_n[r];
    return n;
  }
  // Split routing: the dispatchers that send this copy tokens this step, from the step's plan (beginStep).
  std::vector<int> plan_n;
  bool serves(int d) const {
    if (split_route) return d < (int)plan_n.size() && plan_n[d] > 0;
    return copyFor(d, ncopies()) == rank();
  }
  int servedCount() const {
    int n = 0;
    for (int d = 0; d < n_disp; d++) if (serves(d)) n++;
    return n;
  }
  CProxyElement_Expert peerAt(int r) { return expert_proxy(expert(), r); }
  CProxyElement_Expert self() { return thisProxy[thisIndex]; }

  // Never cached: a different object on every PE this expert lands on.
  MoeCtx* ctx() { return &disp_proxy.ckLocalBranch()->gpu; }
  MoeDevPool* devPool() { return moeDevPool(); }
  // The lane this expert's current (or last) step was issued on.
  MoeGpu* lane() { return &ctx()->lanes[cur_lane >= 0 ? cur_lane : expert() % n_lanes]; }

  // A step's whole chain goes on one lane, behind whatever that lane already
  // holds; with lanes[thisIndex % n_lanes] an expert that happened to share
  // the hot expert's lane queued behind its entire step, so its outputs --
  // which every dispatcher needs -- left at the end and all eight dispatchers
  // started their next step together (large MoE, 2 nodes: the hot expert then
  // waited ~30 ms for its slabs, vs 1-9 ms when its neighbours were on other
  // lanes; job 23192278). The lane with the fewest tokens still in flight goes
  // first. Safe to switch: compute() runs only after the previous step's
  // kernels completed (computeDone), and the results do not depend on lanes.
  void pickLane() {
    MoeCtx* c = ctx();
    int best = 0;
    for (int l = 1; l < n_lanes; l++)
      if (c->lane_pending[l] < c->lane_pending[best]) best = l;
    cur_lane = best;
    lane_claim = std::max(n_tot, 1);
    c->lane_pending[best] += lane_claim;
  }
  void releaseLane() {
    if (lane_claim == 0) return;
    ctx()->lane_pending[cur_lane] -= lane_claim;
    lane_claim = 0;
  }
  hapiStream_t commStream() { return ctx()->comm.stream; }
  size_t segStride() const { return (size_t)cap_e[expert()] * d_model; }   // elements: this expert's own capacity
  size_t segBytes() const { return segStride() * actEsz(); }
  // The weights the GEMMs read: the bf16 copies under -B, else the masters.
  const void* wRead1() const { return use_bf16 ? (const void*)W1b : (const void*)W1; }
  const void* wRead2() const { return use_bf16 ? (const void*)W2b : (const void*)W2; }
  size_t wsz() const { return (size_t)d_model * d_ff; }
  // What one copy holds beyond the expert's own state: its accumulator and
  // the two landing buffers for the other copy's (moe.h, LBObjData::
  // replicaExtraBytes).
  // A copy's device state beyond the expert: its own block sum and scratch
  // (replicaExtraBytes), and two pairs of landing buffers per peer
  // (replicaExtraBytesPerPeer); nothing in serving, which exchanges nothing.
  size_t replicaExtraBytes() const { return serve_only ? 0 : sizeof(float) * wsz() * 4; }
  size_t replicaExtraBytesPerPeer() const { return serve_only ? 0 : sizeof(float) * wsz() * 4; }
  // A step's input slab. With send_ahead a dispatcher sends step s+1 before
  // it has combined step s, so slabs of s+1 and s+2 can arrive while this
  // expert is still in step s; s+2 lands in s's slab, which is free by then:
  // a dispatcher sends s+2 only once it has this expert's step-s output,
  // which leaves behind the copies of s's slab into the lane (fused) or
  // after the whole step (unfused, or -O).
  char* xSeg(int step) {
    return x_seg + (send_ahead ? (size_t)(step & 1) * segBytes() * n_disp : 0);
  }

  void allocate() {
    const size_t seg = segStride() * n_disp;
    hapiCheck(hapiMalloc((void**)&W1, sizeof(float) * wsz()));
    hapiCheck(hapiMalloc((void**)&W2, sizeof(float) * wsz()));
    if (use_adam) {
      hapiCheck(hapiMalloc((void**)&m1, sizeof(float) * wsz()));
      hapiCheck(hapiMalloc((void**)&v1, sizeof(float) * wsz()));
      hapiCheck(hapiMalloc((void**)&m2, sizeof(float) * wsz()));
      hapiCheck(hapiMalloc((void**)&v2, sizeof(float) * wsz()));
    }
    if (use_bf16) {
      hapiCheck(hapiMalloc((void**)&W1b, sizeof(moe_bf16) * wsz()));
      hapiCheck(hapiMalloc((void**)&W2b, sizeof(moe_bf16) * wsz()));
      wb_stale = true;
    }
    hapiCheck(hapiMalloc((void**)&x_seg, actEsz() * seg * (send_ahead ? 2 : 1)));
    hapiCheck(hapiMalloc((void**)&y_seg, actEsz() * seg));
    hapiCheck(hapiMalloc((void**)&d_chk, sizeof(double)));
    h_chk = moePinnedChkGet();   // per PE, never freed mid-run (moePinnedChkGet)
  }

  // The exchange buffers of a replicated copy. Freed, stream-ordered, when the
  // expert is alone again.
  // The exchange state for a group of k copies: this copy's block sum and
  // scratch once, and the landing pairs for k - 1 peers, added as the group
  // grows (a copy pays for the peers it has, not for the most it could have).
  void allocateExchange(int k) {
    if (serve_only) return;   // serving: nothing is exchanged
    // Each pair (W1-shaped, W2-shaped) is one allocation: a block-sum pair is
    // one channel put. The second pointer is the first plus wsz; only the
    // first is freed.
    if (!dWown1) {
      hapiCheck(hapiMalloc((void**)&dWown1, 2 * sizeof(float) * wsz()));
      dWown2 = dWown1 + wsz();
      hapiCheck(hapiMalloc((void**)&dWownS1, 2 * sizeof(float) * wsz()));
      dWownS2 = dWownS1 + wsz();
    }
    for (int sl = 0; sl < k - 1; sl++)
      for (int q = 0; q < 2; q++) {
        if (dWpeer1[sl][q]) continue;
        hapiCheck(hapiMalloc((void**)&dWpeer1[sl][q], 2 * sizeof(float) * wsz()));
        dWpeer2[sl][q] = dWpeer1[sl][q] + wsz();
      }
  }
  void freeExchange() {
    // Any channel still open over a landing pair closes first (its holder
    // dropped its copies before this ran: noGradHolders).
    for (size_t i = 0; i < grad_ch.size(); i++)
      if (grad_open[i]) { grad_ch[i].close(); grad_open[i] = 0; }
    std::vector<float*> bufs = {dWown1, dWownS1};
    for (int sl = 0; sl < K_MAX_COPIES - 1; sl++)
      for (int q = 0; q < 2; q++) bufs.push_back(dWpeer1[sl][q]);
    for (float* b : bufs) {
      if (!b) continue;
      if (CkDevicePoolOn() && !hostSyncMode()) { hapiDevPoolNoteRead(b, xg.stream); hapiDevPoolNoteRead(b, send_stream); }
      else { hapiCheck(hapiStreamSynchronize(xg.stream)); hapiCheck(hapiStreamSynchronize(send_stream)); }
      hapiCheck(hapiFree(b));
    }
    dWown1 = dWown2 = dWownS1 = dWownS2 = NULL;
    for (int sl = 0; sl < K_MAX_COPIES - 1; sl++)
      for (int q = 0; q < 2; q++) dWpeer1[sl][q] = dWpeer2[sl][q] = NULL;
  }

  void createEvents() {
    ev_start = moeEventGet(true);
    ev_end = moeEventGet(true);
    ev_chk = moeEventGet(false);
    ev_grad = moeEventGet(false);
    // From the runtime's stream pool (normal priority; they were created at the greatest, -1, until 2026-11-08), and returned
    // to it by the destructor: hipStreamCreate costs ~13 ms a stream on ROCm 7.14, paid twice by every arrival, and
    // hipStreamDestroy drains the stream, a host wait the destructor cannot afford mid-step (moePinnedChkGet).
    send_stream = hapiAcquireStream();   // normal priority (Dispatcher: gpu.comm.stream)
    memset(&xg, 0, sizeof(xg));
    xg.bf16 = use_bf16 != 0;
    xg.stream = hapiAcquireStream();
    hapiCheck(hapiMalloc((void**)&xg.partials, sizeof(double) * MOE_RED_BLOCKS));
    ev_seg.resize(n_disp);
    for (hapiEvent_t& e : ev_seg) e = moeEventGet(false);
    ev_pending = false;
  }

  // Token channels for the dispatchers this copy serves, handed to them:
  // at start (late 0, the init handshake) and, when the copies change and
  // this copy's block grows, for the dispatchers it newly serves (late 1,
  // which the dispatcher waits for before routing the step).
  void openChannelsFor(int d, int late) {
    const int np = tokParities();
    for (int q = 0; q < np; q++) {
      CkDeviceChannel& c = tok_ch[q * n_disp + d];
      c = CkDeviceChannel(xSeg(q) + (size_t)d * segBytes(), segBytes(),
                          CkCallback(CkIndex_Expert::slabArrived(NULL), self()));
      c.open();
      ch_open[q * n_disp + d] = 1;
    }
    disp_proxy[dispPe(d)].takeChannels(expert(), rank(), late, tok_ch[d],
                                            tok_ch[(np - 1) * n_disp + d]);
  }
  void openChannels(int late = 0) {
    if (!tok_channels) return;
    const int np = tokParities();
    tok_ch.resize(np * n_disp);
    ch_open.assign(np * n_disp, 0);
    slab_held.assign(np * n_disp, 0);
    // Serving: every copy has channels from every dispatcher, which the runtime routes to whichever copy serves it.
    for (int d = 0; d < n_disp; d++)
      if (serve_only || serves(d)) openChannelsFor(d, late);
  }
  // When the copies change: every dispatcher this copy newly serves gets
  // its channels, or a note when it has them already from an earlier
  // block (channels are never dropped except by a retiring copy); the
  // dispatcher counts one arrival per change.
  void refreshChannels(int k_old, int k_new) {
    if (!tok_channels) return;
    for (int d = 0; d < n_disp; d++) {
      if (copyFor(d, k_new) != rank() || copyFor(d, k_old) == rank()) continue;
      if (!ch_open[d]) openChannelsFor(d, 1);
      else disp_proxy[dispPe(d)].chanArrived();
    }
  }

  void init() {
    started = true;
    allocate();
    createEvents();
    // -X: both copies of the chosen expert exist from the start.
    if (expert() == rep_init_expert) setCopies(2);
    openChannels();
    MoeGpu* g = lane();
    // The same seed for both copies: identical weights without a transfer.
    moeInitUniform(W1, wsz(), hmix(hmix(moe_seed, 1), expert()),
        1.0f / sqrtf((float)d_model), g->stream);
    moeInitUniform(W2, wsz(), hmix(hmix(moe_seed, 2), expert()),
        1.0f / sqrtf((float)d_ff), g->stream);
    if (use_adam) {
      hapiCheck(hapiMemsetAsync(m1, 0, sizeof(float) * wsz(), g->stream));
      hapiCheck(hapiMemsetAsync(v1, 0, sizeof(float) * wsz(), g->stream));
      hapiCheck(hapiMemsetAsync(m2, 0, sizeof(float) * wsz(), g->stream));
      hapiCheck(hapiMemsetAsync(v2, 0, sizeof(float) * wsz(), g->stream));
    }
    CkCallback* cb = new CkCallback(CkIndex_Expert::initDone(), self());
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
        hapiCheck(hapiStreamSynchronize(lane()->stream));
        TRACE(1, "expert %d.%d pack step %d phase %d synced after %.3f ms",
            expert(), rank(), my_step, phase, (tnow() - t0) * 1e3);
      } else if (!(serve_only && p.isCloning())) {
        // Order the migration stream's copies behind this stream's queued
        // work on the device instead; the host goes on. Safe because the
        // runtime notes every pool source as read on the migration stream,
        // so the destructor's frees park until the copies retire. (A copy in
        // serving reads only the weights, which nothing writes: no order.)
        p.pup_device_order((void*)lane()->stream);
      }
      if (!p.isCloning()) releaseLane();   // this PE's lane count must not keep a departed expert's tokens
    }
    p | my_step;
    p | recv_count;
    p | n_expect;
    p | phase;
    p | n_src;
    p | plan_n;
    p | src_sent;
    if (p.isUnpacking()) eager_started = false;   // split routing: whom this step's slabs come from and its outputs go to (beginStep)
    // The other copies' figures for this step (their notes), which a move mid-step must not lose: since 2026-11-08
    // an expert moves while it steps, and a note already in left its original reporting without that copy's tokens
    // (expert 25 at step 41: 2164 of 4477, the token check aborted). A copy starts with none (beginStep).
    if (!p.isCloning()) {
      p | peer_n;
      p | peer_pe;
      p | peer_gpu;
      p | peer_ret;
    }
    p | n_tot;
    // SDAG state: which whens are outstanding, buffered ordinary messages. A
    // copy (p.isCloning(), the runtime's clone) starts with none of it: the
    // migration constructor's fresh state stands.
    if (!p.isCloning()) _sdag_pup(p);
    p | outstanding_sends;
    p | outputs_sent;
    p | drain_pending;
    p | chk_due;
    p | lb_waiting;
    p | lb_start_step;
    p | awaiting_resume;
    p | spt;
    p | gpu_last;
    p | n_last;
    p | step_last;
    p | adam_t;
    // Replication state. The exchange buffers hold nothing live at the join
    // point (the previous step's partial was consumed before this step
    // began, and this step's is produced after the join), so only their
    // existence travels; a partial's send still in flight is held by the
    // runtime (outstanding device sends) like any other.
    p | started;
    // A copy's start, across a move between its making and its start (the copy made on its source's PE and moved).
    {
      int src = p.isCloning() ? rank() : clone_source;
      p | src;
      if (p.isUnpacking()) clone_source = src;
    }
    if (!p.isCloning()) {
      p | state_whole;
      p | start_pending;
      PUParray(p, pend_start, 3);
      p | pend_spt;
      p | clone_start_step;
      p | clone_start_adam;
      p | clone_start_nc;
      p | clone_start_spt;
      p | clone_whole;
    }
    p | replicated;
    p | nc;
    p | nc_next;
    p | nc_from;
    p | retire_me;
    p | res_at;
    p | grad_in_flight;
    p | grad_drop_after;
    p | grad_bufs_done;
    p | clone_in_flight;
    p | clone_bufs_done;
    p | clone_bufs_expected;
    p | clone_decided;
    p | retire_decided;
    p | decided_rank;
    p | decided_new;
    p | n_clone;
    p | clone_to_pe;
    p | clone_rank;
    p | retire_pending;
    p | exchange_free_due;
    p | mult_next;
    // The token channels travel with the element and reopen on its new slab.
    // A put that landed after the last note taken (unseen), or whose note was
    // taken but not yet consumed (slab_held), has its bytes in the slab only,
    // so that segment goes with the pup.
    std::vector<char> carry;
    if (!p.isCloning()) {
      p | tok_ch;
      p | ch_open;
      p | slab_held;
      carry.assign(tok_ch.size(), 0);
      if (!p.isUnpacking())   // the sizing pass must count what the packing pass emits
        for (size_t i = 0; i < tok_ch.size(); i++)
          carry[i] = ch_open[i] && (tok_ch[i].unseenBytes() != 0 || slab_held[i]);
      p | carry;
    }

    if (p.isUnpacking() && p.isCloning()) {
      // A copy: its own buffers, its own events, none of the source's step,
      // channels or decisions. It serves nothing until startCopy.
      const double ta = tnow();
      allocate();
      const double tb = tnow();
      createEvents();
      TRACE(1, "expert %d.%d clone unpack: allocate %.3f ms, events %.3f ms", expert(), rank(), (tb - ta) * 1e3, (tnow() - tb) * 1e3);
      resetAsCopy();
    } else if (p.isUnpacking()) {
      const double ta = tnow();
      allocate();
      const double tb = tnow();
      createEvents();   // a pending timing sample is lost with the old events
      TRACE(1, "expert %d.%d unpack: allocate %.3f ms, events %.3f ms", expert(), rank(), (tb - ta) * 1e3, (tnow() - tb) * 1e3);
      if (replicated) allocateExchange(nc);
      // Control events queue until the channels reopen, in ckJustMigrated: an
      // ordinary entry context once the element's state is in place, not the
      // unpack, which can run inside the device poll's own callback (leanmd's
      // Compute does the same).
      for (size_t i = 0; i < tok_ch.size(); i++)
        if (ch_open[i]) tok_ch[i].arriving();
      arrived_by_pup = !tok_ch.empty();
    }
    if (!p.isCloning()) {
      // The exchange channels: the owner side reopens on the new landing pairs
      // (allocated above on unpack), the holder copies re-resolve.
      p | grad_ch;
      p | grad_open;
      p | grad_holder_alive;
      int ng = (int)grad_hold.size();
      p | ng;
      if (p.isUnpacking()) { grad_hold.assign(ng, nullptr); grad_by_pup = ng > 0; }
      for (int i = 0; i < ng; i++) {
        bool has = grad_hold[i] != nullptr;
        p | has;
        if (!has) continue;
        if (p.isUnpacking()) grad_hold[i] = new CkDeviceChannel();
        grad_hold[i]->pup(p);
      }
      p | grad_have;
      if (p.isUnpacking())
        for (size_t i = 0; i < grad_ch.size(); i++)
          if (grad_open[i]) grad_ch[i].arriving();
    }
    if (!p.isCloning()) {
      int no = (int)out_ch.size();
      p | no;
      if (p.isUnpacking()) { out_ch.assign(no, nullptr); out_by_pup = no > 0; }
      for (int i = 0; i < no; i++) {
        bool has = out_ch[i] != nullptr;
        p | has;
        if (!has) continue;
        if (p.isUnpacking()) out_ch[i] = new CkDeviceChannel();
        out_ch[i]->pup(p);
      }
      p | out_have;
    }
    for (size_t i = 0; i < carry.size(); i++) {
      if (!carry[i]) continue;
      TRACE(2, "expert %d.%d step %d phase %d %s channel %zu (step parity %zu, dispatcher %zu): %s",
          expert(), rank(), my_step, phase, p.isUnpacking() ? "unpacks" : "packs", i, i / n_disp,
          i % n_disp, slab_held[i] ? "held" : "unseen");
      p(xSeg((int)(i / n_disp)) + (i % n_disp) * segBytes(), segBytes(),
          PUP::PUPMode::DEVICE);
    }
    // The checksum's copy into h_chk is asynchronous; wait for it before the
    // host reads it, or a migration here packs whatever was there before.
    if (!p.isCloning()) {
      if (p.isPacking() && h_chk && chk_due)
        hapiCheck(hapiEventSynchronize(ev_chk));
      double chk = (h_chk && chk_due) ? *h_chk : 0.0;
      p | chk;
      if (p.isUnpacking()) *h_chk = chk;
    }

    // In serving nothing writes the weights, so they are Stable: the runtime
    // ships them at the balancer's decision while this copy keeps stepping,
    // holds the move until they have landed, and the pack carries the rest
    // (a migration then stops the expert for the host state and the
    // channels, not for 200 MB over the link: two A100 nodes, 2026-11-07, a
    // round moving 16 experts and cloning 30 stopped every step for 245 ms).
    // In training the update rewrites them every step: Mutable, copied at
    // the pack as before.
    // A copy in bf16 serving computes from the bf16 operands alone, and only the primary (rank 0, never a clone)
    // reads the masters, for the weight checksum: a clone, and any copy but the primary when it moves, carries the
    // bf16 weights, half the bytes (2026-11-09: the first round's clones of the hot expert across nodes landed 150 ms
    // after those within its node, and the copies serve in rank order).
    static const bool fp32_state = getenv("MOE_FP32_STATE") != NULL;   // MOE_FP32_STATE=1: the masters, as before
    const bool bf16State = !fp32_state && serve_only && use_bf16 && (p.isCloning() || rank() > 0);
    if (bf16State) {
      p.pup_device(W1b, wsz(), PUP::DeviceLife::Stable);
      p.pup_device(W2b, wsz(), PUP::DeviceLife::Stable);
      if (p.isUnpacking()) wb_stale = false;   // the operands came whole; the masters here are unused
    } else {
      const PUP::DeviceLife life = serve_only ? PUP::DeviceLife::Stable : PUP::DeviceLife::Mutable;
      p.pup_device(W1, wsz(), life);
      p.pup_device(W2, wsz(), life);
    }
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
    if (p.isCloning()) {
      // A copy takes the weights (and moments) alone.
    } else if (phase == 0) {
      for (int s = 0; s < n_disp; s++)
        if (n_src[s] > 0)
          p(xSeg(my_step) + s * segBytes(), (size_t)n_src[s] * d_model * actEsz(),
              PUP::PUPMode::DEVICE);
    } else if (phase == 1) {
      for (int s = 0; s < n_disp; s++)
        if (n_src[s] > 0)
          p(y_seg + s * segBytes(), (size_t)n_src[s] * d_model * actEsz(),
              PUP::PUPMode::DEVICE);
    }
    if (p.isPacking())
      TRACE(1, "expert %d.%d pack end %.3f ms", expert(), rank(), (tnow() - t0) * 1e3);
    else if (p.isUnpacking())
      TRACE(1, "expert %d.%d unpack step %d phase %d %.3f ms", expert(), rank(), my_step,
          phase, (tnow() - t0) * 1e3);
  }

  void beginStep() {
    my_step++;
    if (my_step == warmup_steps + 1) {
      // The balancer's first interval would otherwise span the start-up (the pool, the handles, the first
      // launches: on two A100 nodes a PE's host time over steps 1-20 read 0.10 s against 0.02 over steps 20-40,
      // and the host cap refused 43 placements on it, job 23646171). Cleared once per PE as the timed steps
      // begin, as sph3d does; the runtime clears them at the end of every round after that.
      static thread_local bool lb_loads_cleared = false;
      if (!lb_loads_cleared) { lb_loads_cleared = true; LBClearLoads(); }
    }
    recv_count = 0;
    phase = 0;
    n_tot = 0;
    outputs_sent = false;
    resumed_by_lb = false;
    for (int r = 0; r < K_MAX_COPIES; r++) { peer_n[r] = 0; peer_pe[r] = -1; peer_gpu[r] = 0.0; peer_ret[r] = false; }
    if (serve_only) {
      const int k = CkReplicaCount(expert_proxy.ckGetArrayID(), expert(), my_step);   // known: copiesKnown
      if (k != nc && rank() < k) setCopies(k);
      if (rank() >= k) {   // past the count: this copy's last step was the one before
        replica_gone = true;
        n_expect = 0;
        return;
      }
    }
    if (split_route) {
      const MoeStepPlan* P = disp_proxy.ckLocalBranch()->planOf(my_step);
      if (P == nullptr) CkAbort("Expert %d.%d on PE %d: step %d began before its plan was here\n", expert(), rank(), CkMyPe(), my_step);
      plan_n.assign(n_disp, 0);
      std::vector<int> so(P->k[expert()] + 1);
      if (rank() < P->k[expert()]) {
        int j = 0;
        while (P->ord[P->xo[expert()] + j] != rank()) j++;
        for (int d = 0; d < n_disp; d++) {
          moePlanOffsets(*P, n_disp, expert(), d, so.data());
          plan_n[d] = so[j + 1] - so[j];
        }
      }
    }
    n_expect = servedCount();
    std::fill(n_src.begin(), n_src.end(), -1);
    src_sent.assign(n_disp, 0);
    eager_started = false;
  }

  // Post entry method: a dispatcher's slab lands in its own segment. An empty
  // message still lands one row there, which nothing reads.
  void receiveTokens(int ref, int src, int n, long long ts, int spe, int& len, char*& buf,
      CkDeviceBufferPost* devicePost) {
    if (flight_log) { env_t.resize(3 * n_disp); env_t[(ref % 3) * n_disp + src] = moeNowNs(); }
    if (ref < my_step || ref > my_step + (send_ahead ? 2 : 1))
      CkAbort("Expert %d.%d on PE %d: slab for step %d arrived during step %d\n",
          expert(), rank(), CkMyPe(), ref, my_step);
    buf = xSeg(ref) + (size_t)src * segBytes();
    devicePost[0].hapi_stream = commStream();
    TRACE(3, "expert %d.%d step %d slab src %d n=%d posted", expert(), rank(), ref, src,
        n);
  }

  // A note on one of the token channels: find the channel it names and let it
  // take the note. A landed put goes on to the step loop as slabIn, matched by
  // its step; anything else (a resolve, an announce) the channel handles.
  void slabArrived(CkDataMsg* m) {
    CkDeviceChannelEvent ev;
    memcpy(&ev, m->getData(), sizeof(ev));
    int idx = -1;
    for (int i = 0; i < (int)tok_ch.size(); i++)
      if (ch_open[i] && tok_ch[i].id() == ev.channel) { idx = i; break; }
    if (idx < 0)
      CkAbort("Expert %d.%d on PE %d: a note for channel %llx, which is not one of its own\n",
          expert(), rank(), CkMyPe(), (unsigned long long)ev.channel);
    const CkDeviceChannelEvent* e = tok_ch[idx].event(m);
    delete m;
    if (e == NULL) return;
    const int ref = (int)e->tag;
    if (ref % tokParities() != idx / n_disp)
      CkAbort("Expert %d.%d on PE %d: step %d's slab arrived on the channel of parity %d\n",
          expert(), rank(), CkMyPe(), ref, idx / n_disp);
    slab_held[idx] = 1;
    TRACE(2, "expert %d.%d step %d slab for step %d from dispatcher %d landed, %zu bytes",
        expert(), rank(), my_step, ref, idx % n_disp, e->bytes);
    self().slabIn(ref, idx % n_disp, (int)(e->bytes / (actEsz() * d_model)));
  }

  void ckJustMigrated() {
    CBase_Expert::ckJustMigrated();
    if (out_by_pup) {
      out_by_pup = false;
      for (CkDeviceChannel* c : out_ch)
        if (c) c->resolve();
    }
    if (grad_by_pup) {
      grad_by_pup = false;
      for (CkDeviceChannel* c : grad_hold)
        if (c) c->resolve();
      for (size_t i = 0; i < grad_ch.size(); i++)
        if (grad_open[i]) grad_ch[i].reopen(dWpeer1[i / 2][i & 1]);
    }
    if (!arrived_by_pup) return;   // a handoff within the process: the channels are intact
    arrived_by_pup = false;
    int n = 0;
    for (size_t i = 0; i < tok_ch.size(); i++) {
      if (!ch_open[i]) continue;
      tok_ch[i].reopen(xSeg((int)(i / n_disp)) + (i % n_disp) * segBytes());
      n++;
    }
    TRACE(2, "expert %d.%d step %d phase %d reopened %d channels", expert(), rank(), my_step,
        phase, n);
  }

  void noteTokens(int src, int n, long long ts, int spe) {
    TRACE(2, "expert %d.%d step %d consumes slab from dispatcher %d n=%d", expert(), rank(),
        my_step, src, n);
    if (!serves(src))
      CkAbort("Expert %d.%d on PE %d: step %d slab from dispatcher %d, which its other copy serves\n",
          expert(), rank(), CkMyPe(), my_step, src);
    if (tok_channels) slab_held[(my_step % tokParities()) * n_disp + src] = 0;
    if (flight_log && !tok_channels)
      g_flights.push_back({'T', my_step, spe, CkMyPe(), expert(), (long)std::max(n, 1) * d_model * (long)actEsz(),
                           ts, env_t[(my_step % 3) * n_disp + src], moeNowNs()});
    n_src[src] = n;
    TRACE(3, "expert %d.%d step %d slab src %d n=%d consumed", expert(), rank(), my_step,
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
    // The job-wide rate (MoeCtx::global_spt), so every expert on every GPU
    // declares tokens on one scale; this PE's pooled figure until Main has
    // broadcast one, and the expert's own until this PE has measured.
    const MoeCtx* c = ctx();
    const MoeDevPool* pool = devPool();
    double s_tok = (c->global_spt > 0.0) ? c->global_spt
                 : (pool->spt_tok > 0.0) ? pool->spt_gpu / pool->spt_tok : spt;
    // Diagnostic: a fixed per-token cost, identical on every PE. Every expert
    // here does the same work per token, so this is the exact load ratio; it
    // separates a balancer that cannot use a good signal from a bad signal.
    static const bool flat = getenv("CHARM_MOE_FLATLOAD") != NULL;
    if (flat) s_tok = 1e-5;
    // No rate measured yet anywhere (a balancing step this early, 2026-11-08: the round at step 5 read most experts
    // as idle, the experts of one host at 0 s, and piled seven of them on each of two GPUs): the tokens at one rate
    // for every expert, the load ratio the balancer needs; the measured rate takes over as soon as there is one.
    if (!(s_tok > 0.0)) s_tok = 1e-6;
    // Declared over the balancing interval, not the step. The balancer holds
    // this against host time it accumulated since the previous balancing
    // step and against that interval's wall time, so a per-step figure is
    // short by the interval length: measured 12 Sep 2026, it read as 17% of
    // the interval when the device was 68-77% busy (nvidia-smi). This step's
    // tokens stand for the interval; every expert scales by the same factor,
    // so the balance is unchanged and only the units are right.
    //
    // A declaration now takes precedence over the CUPTI attribution for the
    // interval (LBObj::gpuDeclared), so under -I, which asks for the measured
    // loads, nothing is declared.
    if (!use_instrument) setObjGPUTime(s_tok * n_tot * stepsSinceLastLB(my_step));
    // This copy's place in its expert: the balancer may ask for a second copy
    // of an expert hotter than a GPU's share, or take one back (ShedLB).
    // What the balancer prices a copy with: the bytes and messages it sends
    // each peer over the interval (two fp32 block sums a step), priced by the
    // runtime's calibrated transfer-cost table at the group's size; and, once
    // this copy has exchanged, what the exchange took here, its wait per
    // step and peer, which stands in for the table's figure. The capability
    // is the four-partition tree's: up to four copies.
    const double xsteps = serve_only ? 0.0 : (double)stepsSinceLastLB(my_step);
    double measured = 0.0;
    if (!serve_only && replicated && xchg_waits > 0) measured = xchg_wait_s / xchg_waits / (nc - 1) * xsteps;
    setLBReplica(expert(), rank(), nc, K_MAX_COPIES, replicaExtraBytes(), measured, measured, split_route ? -1 : K_PART,
        serve_only ? 0 : (size_t)(2.0 * sizeof(float) * wsz() * xsteps), 2.0 * xsteps, replicaExtraBytesPerPeer());
    if (isLBStep(my_step)) {
      if (replicated && xchg_waits > 0)
        TRACE(1, "expert %d.%d step %d: exchange wait %.2f ms per step over %d steps, %.2f ms per peer and step declared",
            expert(), rank(), my_step, xchg_wait_s / xchg_waits * 1e3, xchg_waits, xchg_wait_s / xchg_waits / (nc - 1) * 1e3);
      xchg_wait_s = 0.0;
      xchg_waits = 0;
    }
    if (isLBStep(my_step))
      TRACE(1, "expert %d.%d step %d declares %.6f s: %d tokens x %.4e s/token x %d steps; job rate %.4e; PE pool "
          "%.4f s / %.0f tokens; own last sample %.4f s for %d tokens", expert(), rank(), my_step,
          s_tok * n_tot * stepsSinceLastLB(my_step),
          n_tot, s_tok, stepsSinceLastLB(my_step), c->global_spt, pool->spt_gpu, pool->spt_tok, gpu_last, n_last);

    if (isLBStep(my_step) && !lb_waiting) {
      if (!async_lb) {
        // Stops here; ResumeFromSync means the strategy and the migrations
        // are over.
        TRACE(1, "expert %d.%d step %d atsync", expert(), rank(), my_step);
        awaiting_resume = true;
        AtSync();
        return;
      }
      // Split barrier: join the step and keep computing while the strategy
      // runs and other experts migrate.
      TRACE(1, "expert %d.%d step %d atsyncstart", expert(), rank(), my_step);
      AtSyncStart();
      lb_waiting = true;
      lb_start_step = my_step;
      self().resumed();
      return;
    }
    if (lb_waiting && my_step >= lb_start_step + lb_wait_lag &&
        (!AtSyncStepInFlight() || my_step >= lb_start_step + LB_PARK_MAX)) {
      // The other half of the split: park until the step is collected.
      // Resumes inline if it already is -- and that is the normal case now:
      // an expert keeps stepping while the round's moves ship (a move waits
      // for its Stable buffers to land, then takes the expert at a quiet
      // entry boundary), and parks only once the round is over, where the
      // park is a formality. A parked expert serves no tokens, so parking at
      // the lag while another expert's 200 MB crossed the link stopped every
      // dispatcher for the transfer. LB_PARK_MAX bounds the wait.
      lb_waiting = false;
      TRACE(1, "expert %d.%d step %d atsyncwait", expert(), rank(), my_step);
      awaiting_resume = true;
      AtSyncWait();
      return;
    }
    self().resumed();
  }

  void ResumeFromSync() {
    if (!awaiting_resume) {
      // A copy inserted while the balancer's step was resuming on this PE
      // gets the resume too; it asked for nothing.
      TRACE(1, "expert %d.%d step %d ignores a resume it did not ask for", expert(), rank(), my_step);
      return;
    }
    awaiting_resume = false;
    resumed_by_lb = true;
    TRACE(1, "expert %d.%d step %d resume", expert(), rank(), my_step);
    // The round is over everywhere (the resume epoch is global): this GPU's
    // dispatcher may send ahead again, from the step after the next.
    disp_proxy[localDispPe()].roundResumed();
    self().resumed();
  }

  // ---- the balancer's replication decisions (CkMigratable) ----
  // Recorded here and acted on in the step the balancer releases this copy
  // in (resumed_by_lb), after that step's update: the dispatchers learn the
  // change from that step's outputs and route the next step accordingly.
  // Serving (2026-11-08): the copies are the runtime's (CkMigratable::setReplicaManaged) -- it makes them, agrees
  // the step from which they serve on every PE, and tells each copy (ckReplicaServing); the dispatchers ask it which
  // copy serves them (Dispatcher::routeOf). This copy only names a copy's index and opens a new copy's channels.
  CkArrayIndex replicaIndex(int r) const override { return CkArrayIndex2D(thisIndex.x, r); }
  void ckReplicaServing(int fromEpoch, int k) override {
    TRACE(1, "expert %d.%d step %d: %d copies from step %d (the runtime's)", expert(), rank(), my_step, k, fromEpoch);
    if (!started && rank() > 0 && rank() < k && state_whole) {
      my_step = fromEpoch - 1;   // beginStep makes it fromEpoch
      started = true;
      self().runStep();
    }
  }
  void ckClone(int toPe, int rank_) {
    if (serve_only) { CkMigratable::ckClone(toPe, rank_); return; }
    if (rank_ < nc || rank_ >= K_MAX_COPIES || decided_new[rank_])
      CkAbort("Expert %d.%d on PE %d: asked for copy %d on PE %d while %d of %d copies exist\n",
          expert(), rank(), CkMyPe(), rank_, toPe, nc, K_MAX_COPIES);
    clone_to_pe[n_clone] = toPe;
    clone_rank[n_clone] = rank_;
    n_clone++;
    decided_new[rank_] = true;
    clone_decided = true;
    TRACE(1, "expert %d.%d step %d: clone to PE %d as copy %d decided (the runtime copies the state%s)",
        expert(), rank(), my_step, toPe, rank_, serve_only ? " now" : " after this step's update");
    // Serving: the weights never change, so the copy's state goes now, while
    // this copy keeps stepping; by the step it is released in, the copy is
    // there, and it is told its first step then.
    if (serve_only) ckCloneTo(toPe, CkArrayIndex2D(expert(), rank_));
  }
  void ckCloneNotice(int toPe, int rank_) {
    if (serve_only) return;
    if (rank_ < nc || rank_ >= K_MAX_COPIES || decided_new[rank_])
      CkAbort("Expert %d.%d on PE %d: told of copy %d while %d of %d copies exist\n", expert(), rank(),
          CkMyPe(), rank_, nc, K_MAX_COPIES);
    decided_new[rank_] = true;
    clone_decided = true;
    TRACE(1, "expert %d.%d step %d: copy %d on PE %d decided (notice)", expert(), rank(), my_step, rank_, toPe);
  }
  // The copies this round adds: their ranks run on from nc without a gap,
  // as the block mapping assumes (the balancer numbers them so, and
  // CentralLB keeps the contiguous prefix of what the memory verifier let
  // through).
  int newCopies() const {
    int m = 0;
    while (nc + m < K_MAX_COPIES && decided_new[nc + m]) m++;
    for (int r = nc + m; r < K_MAX_COPIES; r++)
      if (decided_new[r])
        CkAbort("Expert %d.%d on PE %d: copy %d decided while copy %d was not (%d copies exist)\n",
            expert(), rank(), CkMyPe(), r, nc + m, nc);
    return m;
  }
  void ckRetire() {
    if (serve_only) { CkMigratable::ckRetire(); return; }
    if (!replicated || rank() != nc - 1)
      CkAbort("Expert %d.%d on PE %d: asked to retire but it is not the highest of %d copies\n",
          expert(), rank(), CkMyPe(), nc);
    retire_pending = true;
    retire_decided = true;
    decided_rank = rank();
    TRACE(1, "expert %d.%d step %d: retire decided", expert(), rank(), my_step);
  }
  void ckRetireNotice(int rank_) {
    if (serve_only) return;
    if (rank_ != nc - 1)
      CkAbort("Expert %d.%d on PE %d: told copy %d retires while %d copies exist\n", expert(), rank(),
          CkMyPe(), rank_, nc);
    retire_decided = true;
    decided_rank = rank_;
    TRACE(1, "expert %d.%d step %d: copy %d retires (notice)", expert(), rank(), my_step, rank_);
  }

  // Fold the last recorded step's device time into the per-token rate. Its
  // kernels are complete whenever this runs after that step's outputs were
  // sent. Step 1 is skipped: its time is cuBLAS loading its kernels, and one
  // such sample made every expert's first estimate the same 90 ms whatever
  // its token count.
  void sampleTiming() {
    if (!ev_pending || hapiEventQuery(ev_end) != hapiSuccess) return;
    float ms = 0.0f;
    hapiCheck(hapiEventElapsedTime(&ms, ev_start, ev_end));
    ev_pending = false;
    gpu_last = ms * 1e-3;
    if (n_last > 0 && step_last > 1) {
      // Pool the sample into the PE's figure rather than keeping a per-expert
      // one; see MoeCtx. The decay advances once per step, so the order in
      // which a step's experts report does not weight them differently.
      MoeDevPool* pool = devPool();
      std::lock_guard<std::mutex> lk(pool->m);
      float s_ms = 0.0f, e_ms = 0.0f;
      hapiCheck(hapiEventElapsedTime(&s_ms, pool->ref_ev, ev_start));
      hapiCheck(hapiEventElapsedTime(&e_ms, pool->ref_ev, ev_end));
      if (step_last > pool->spt_step) {
        pool->spt_gpu *= 0.75;
        pool->spt_tok *= 0.75;
        pool->spt_step = step_last;
      }
      pool->spt_gpu += 1e-3 * addBusy(pool, s_ms, e_ms);
      pool->spt_tok += n_last;
      spt = pool->spt_gpu / pool->spt_tok;
    }
  }

  // Adds [s, e] (ms on the device clock) to the PE's busy set and returns the
  // part no earlier span -- of any step -- covered. One set across steps: the
  // experts' steps overlap in time under lookahead, and per-step sets counted
  // the same device time once per step. The set is kept disjoint; intervals
  // older than the horizon are dropped.
  static float addBusy(MoeDevPool* c, float s, float e) {
    float fresh = e - s;
    for (int i = 0; i < c->n_iv; i++)
      fresh -= std::max(0.0f, std::min(e, c->iv_e[i]) - std::max(s, c->iv_s[i]));
    // Merge: the overlapping intervals fold into [s, e]; the rest stay unless
    // they are older than the horizon.
    int k = 0;
    for (int i = 0; i < c->n_iv; i++) {
      if (c->iv_e[i] < e - MOE_BUSY_HORIZON_MS) continue;
      if (c->iv_e[i] < s || c->iv_s[i] > e) { c->iv_s[k] = c->iv_s[i]; c->iv_e[k] = c->iv_e[i]; k++; }
      else { s = std::min(s, c->iv_s[i]); e = std::max(e, c->iv_e[i]); }
    }
    if (k < MOE_BUSY_IVS) { c->iv_s[k] = s; c->iv_e[k] = e; k++; }
    c->n_iv = k;
    return std::max(fresh, 0.0f);
  }

  // Tokens this copy holds for partition p this step.
  int partitionTokens(int p) const {
    int n = 0;
    for (int s = 0; s < n_disp; s++)
      if (p < 0 || partitionOf(s) == p) n += std::max(n_src[s], 0);   // p < 0: every dispatcher
    return n;
  }

  // One run: the slabs of partition p's dispatchers, in dispatcher order,
  // through the chunked step into the accumulators dW1/dW2 (the run's first
  // chunk overwrites them). The outputs go back into each dispatcher's
  // segment and, with stream_outputs, each dispatcher's leave as soon as its
  // tokens are done. Returns the chunks issued.
  // One chunked run: the forward alone when serving, else the forward,
  // backward and gradient into dW1/dW2.
  void chunksOf(MoeGpu* g, const void* x, void* y, int n, float* dW1, float* dW2, bool zeroFirst) {
    if (serve_only)
      moeExpertForward(g, wRead1(), wRead2(), x, y, n, d_model, d_ff, chunk_tokens);
    else
      moeExpertChunks(g, wRead1(), wRead2(), x, y, n, d_model, d_ff, chunk_tokens, dW1, dW2,
          zeroFirst);
  }

  int runPartition(MoeGpu* g, int p, float* dW1, float* dW2) {
    const int n_run = partitionTokens(p);
    if (n_run <= 0) return 0;
    int chunks = 0;
    const size_t esz = actEsz();
    char* fx = (char*)g->fx;
    char* fy = (char*)g->fy;
    if (fuse_sources) {
      // One pass over the partition's tokens instead of one per source.
      // The gradient GEMMs of a chunk read and rewrite both accumulators
      // whatever the chunk holds, so a run's cost is set by the number of
      // chunks, and unfused that is at least one per (source, expert) pair:
      // a typical expert gets a few hundred tokens from each dispatcher.
      // Concatenating in dispatcher order keeps the chunk boundaries a
      // function of the routing alone, so the reduction order is still
      // placement-independent.
      size_t off = 0;
      for (int s = 0; s < n_disp; s++) {
        if ((p >= 0 && partitionOf(s) != p) || n_src[s] <= 0) continue;
        const size_t n = (size_t)n_src[s] * d_model;
        hapiCheck(hapiMemcpyAsync(fx + off * esz, xSeg(my_step) + s * segBytes(),
            esz * n, hapiMemcpyDeviceToDevice, g->stream));
        off += n;
      }
      chunks = (n_run + chunk_tokens - 1) / chunk_tokens;
      if (stream_outputs) {
        // The same chunks, issued up to the one that holds each dispatcher's
        // last token; that dispatcher's outputs are final there, so they go
        // back into its segment and its send leaves while the rest of the
        // run computes. A chunk range that starts at a multiple of the chunk
        // size is the same sequence of kernels as one call over all n_run,
        // so the results are bit for bit those of the single call. Sending
        // all of them after the last chunk put every dispatcher's next step
        // behind the busiest expert's whole step plus the slowest of its
        // eight 14 MB sends at once (large MoE, 2 nodes: the busiest GPU idle
        // ~70 ms of a 190 ms step, job 23175260).
        int done = 0, tok = 0;
        for (int s = 0; s < n_disp; s++) {
          if (p >= 0 && partitionOf(s) != p) continue;
          const int ns = std::max(n_src[s], 0);
          if (ns > 0) {
            const int end = tok + ns;
            const int upto = (end >= n_run) ? n_run
                : std::min(n_run, (end + chunk_tokens - 1) / chunk_tokens * chunk_tokens);
            if (upto > done) {
              chunksOf(g, fx + (size_t)done * d_model * esz, fy + (size_t)done * d_model * esz,
                  upto - done, dW1, dW2, done == 0);
              done = upto;
            }
            hapiCheck(hapiMemcpyAsync(y_seg + s * segBytes(),
                fy + (size_t)tok * d_model * esz, esz * (size_t)ns * d_model,
                hapiMemcpyDeviceToDevice, g->stream));
            tok = end;
          }
          hapiCheck(hapiEventRecord(ev_seg[s], g->stream));
        }
      } else {
        chunksOf(g, fx, fy, n_run, dW1, dW2, true);
        // Back into the per-source segments, so the sends and both pup phases
        // are unchanged. Queued before ev_end, which orders a pack behind it.
        off = 0;
        for (int s = 0; s < n_disp; s++) {
          if ((p >= 0 && partitionOf(s) != p) || n_src[s] <= 0) continue;
          const size_t n = (size_t)n_src[s] * d_model;
          hapiCheck(hapiMemcpyAsync(y_seg + s * segBytes(), fy + off * esz,
              esz * n, hapiMemcpyDeviceToDevice, g->stream));
          off += n;
        }
      }
    } else {
      bool first = true;
      for (int s = 0; s < n_disp; s++) {
        if ((p >= 0 && partitionOf(s) != p) || n_src[s] <= 0) continue;
        chunks += (n_src[s] + chunk_tokens - 1) / chunk_tokens;
        chunksOf(g, xSeg(my_step) + s * segBytes(), y_seg + s * segBytes(), n_src[s], dW1, dW2,
            first);
        first = false;
      }
    }
    return chunks;
  }

  // One node of the gradient's tree: partitions p0 and p0 + 1 (an aligned
  // pair; a block of one partition when k = 4) into the accumulator pair
  // (A1, A2), with (S1, S2) as the second leaf's scratch, their sum folded
  // into (A1, A2). o1/o2: the node, or NULL when neither partition had
  // tokens. A copy with a block of one partition (k = 4) folds nothing.
  int treeNode(MoeGpu* g, int p0, float* A1, float* A2, float* S1, float* S2, float*& o1, float*& o2) {
    const bool pair = (p0 + 1 < K_PART) && (blockHi(rank(), ncopies()) - blockLo(rank(), ncopies()) >= 2 || !replicated);
    const int na = partitionTokens(p0), nb = pair ? partitionTokens(p0 + 1) : 0;
    int chunks = 0;
    o1 = o2 = NULL;
    if (na > 0) {
      chunks += runPartition(g, p0, A1, A2);
      o1 = A1; o2 = A2;
    }
    if (nb > 0) {
      float* l1 = na > 0 ? S1 : A1;
      float* l2 = na > 0 ? S2 : A2;
      chunks += runPartition(g, p0 + 1, l1, l2);
      if (na > 0) { moeAddInto(g, A1, S1, wsz()); moeAddInto(g, A2, S2, wsz()); }
      o1 = A1; o2 = A2;
    }
    return chunks;
  }

  // The step's update from the two tree nodes (NULL: no tokens), then the
  // weight checksum when it is due.
  void applyUpdate(MoeGpu* g, const float* A1, const float* A2, const float* B1,
      const float* B2, int n_all) {
    if (n_all > 0) {
      const float inv_n = 1.0f / n_all;
      if (use_adam)
        moeAdamApply(g, W1, m1, v1, W2, m2, v2, W1b, W2b, A1, A2, B1, B2, d_model, d_ff,
            (float)learning_rate, inv_n, ++adam_t);
      else
        moeApplySGD(g, W1, W2, W1b, W2b, A1, A2, B1, B2, d_model, d_ff,
            (float)learning_rate * inv_n);
    }
    if (chk_due) {
      hapiCheck(hapiMemsetAsync(d_chk, 0, sizeof(double), g->stream));
      moeSumSqAdd(W1, wsz(), g, d_chk, g->stream);
      moeSumSqAdd(W2, wsz(), g, d_chk, g->stream);
      hapiCheck(hapiMemcpyAsync(h_chk, d_chk, sizeof(double),
          hapiMemcpyDeviceToHost, g->stream));
      hapiCheck(hapiEventRecord(ev_chk, g->stream));
    }
  }

  // Eager sources (2026-11-09): serving under split routing computes each dispatcher's tokens as its slab is consumed
  // and sends that dispatcher's output behind its own chunk, instead of after the step's last slab. A copy's slabs
  // of a step arrived over 6.7 ms (median) as the dispatchers began their steps at different times, and every one of
  // its outputs waited for the last; now a dispatcher's output depends on its own slab. src_sent marks a source
  // done, pupped, so a copy that moves mid-step neither recomputes nor resends it.
  std::vector<char> src_sent;
  bool eager_started = false;   // this step's lane, timing start and bf16 operands are set (not pupped: redone after a move)
  // Only a piece of at least eagerMin() tokens goes on arrival; the smaller ones are fused into one pass after the
  // step's last slab (fusedRemaining). Every piece on arrival (MOE_EAGER_MIN=1) cost each (dispatcher, copy) piece a
  // GEMM chain of its own, and the GPUs whose copies had many small pieces ran 3x the kernels of others at the same
  // token count (2026-11-09, SmartMoE on 16 GPUs: the fullest at 99% busy, 74% mean, against 82% computing after the
  // last slab). MOE_NO_EAGER=1: everything after the last slab.
  int eagerMin() const {
    static const int m = getenv("MOE_EAGER_MIN") ? atoi(getenv("MOE_EAGER_MIN")) : chunk_tokens;
    return std::max(1, m);
  }
  bool eagerOn() const {
    static const bool on = getenv("MOE_NO_EAGER") == NULL;
    return on && serve_only && split_route && tok_channels && out_channels && fuse_sources && stream_outputs;
  }
  void eagerStart() {
    if (eager_started) return;
    eager_started = true;
    // The previous step's timing sample, before its start event is reused; dropped if its kernels are still going.
    sampleTiming();
    ev_pending = false;
    n_tot = 0;
    for (int d = 0; d < (int)plan_n.size(); d++) n_tot += plan_n[d];
    pickLane();
    MoeGpu* g = lane();
    if (use_bf16 && wb_stale) {
      moeCastBf16(W1, W1b, wsz(), g->stream);
      moeCastBf16(W2, W2b, wsz(), g->stream);
      wb_stale = false;
    }
    hapiCheck(hapiEventRecord(ev_start, g->stream));
  }
  void eagerSource(int s, bool any = false) {
    if (!eagerOn() || s < 0 || s >= (int)src_sent.size() || src_sent[s] || n_src[s] <= 0) return;
    if (!any && n_src[s] < eagerMin()) return;
    ensureOutCh();
    if (!out_have[s]) return;   // compute() does it, once the dispatcher's output channel is here
    eagerStart();
    MoeGpu* g = lane();
    chunksOf(g, xSeg(my_step) + (size_t)s * segBytes(), y_seg + (size_t)s * segBytes(), n_src[s], NULL, NULL, true);
    hapiCheck(hapiEventRecord(ev_seg[s], g->stream));
    hapiCheck(hapiStreamWaitEvent(send_stream, ev_seg[s], 0));
    sendOutput(s);
    src_sent[s] = 1;
  }
  // The small pieces, in one pass: gathered into the lane's scratch in dispatcher order, one product, each
  // dispatcher's rows back into its output segment and sent behind them.
  void fusedRemaining(MoeGpu* g) {
    const size_t esz = actEsz();
    char* fx = (char*)g->fx;
    char* fy = (char*)g->fy;
    size_t off = 0;
    int n_run = 0;
    for (int s = 0; s < n_disp; s++) {
      if (n_src[s] <= 0 || src_sent[s]) continue;
      const size_t n = (size_t)n_src[s] * d_model;
      hapiCheck(hapiMemcpyAsync(fx + off * esz, xSeg(my_step) + (size_t)s * segBytes(), esz * n,
          hapiMemcpyDeviceToDevice, g->stream));
      off += n;
      n_run += n_src[s];
    }
    if (n_run == 0) return;
    chunksOf(g, fx, fy, n_run, NULL, NULL, true);
    size_t tok = 0;
    for (int s = 0; s < n_disp; s++) {
      if (n_src[s] <= 0 || src_sent[s]) continue;
      const size_t n = (size_t)n_src[s];
      hapiCheck(hapiMemcpyAsync(y_seg + (size_t)s * segBytes(), fy + tok * d_model * esz, esz * n * d_model,
          hapiMemcpyDeviceToDevice, g->stream));
      tok += n;
      hapiCheck(hapiEventRecord(ev_seg[s], g->stream));
      hapiCheck(hapiStreamWaitEvent(send_stream, ev_seg[s], 0));
      sendOutput(s);
      src_sent[s] = 1;
    }
  }
  // The step's end under eager sources: whatever source is not done yet, the checksum, and the completion.
  void computeEager() {
    phase = 1;
    eagerStart();
    n_tot = 0;
    for (int s = 0; s < n_disp; s++) n_tot += std::max(n_src[s], 0);
    nc_next = nc;
    mult_next = nc_next;
    chk_due = checksum_freq > 0 && (my_step % checksum_freq == 0) && (rank() == 0 || traceLevel() >= 2);
    TRACE(2, "expert %d.%d step %d compute n=%d (eager)", expert(), rank(), my_step, n_tot);
    for (int s = 0; s < n_disp; s++) eagerSource(s);   // a large piece whose output channel came late
    MoeGpu* g = lane();
    fusedRemaining(g);
    applyUpdate(g, NULL, NULL, NULL, NULL, 0);
    hapiCheck(hapiEventRecord(ev_end, g->stream));
    if (n_tot > 0) {
      ev_pending = true;
      n_last = n_tot;
      step_last = my_step;
    }
    if (n_tot > 0 || chk_due) {
      CkCallback* cb = new CkCallback(CkIndex_Expert::computeDone(), self());
      hapiAddCallback(g->stream, cb);
    } else {
      self().computeDone();
    }
    outputs_sent = true;
    phase = 2;   // a migration is held until every send has completed
  }

  void compute() {
    if (eagerOn()) { computeEager(); return; }
    phase = 1;
    n_tot = 0;
    for (int s = 0; s < n_disp; s++) n_tot += std::max(n_src[s], 0);
    pickLane();
    MoeGpu* g = lane();

    sampleTiming();
    if (exchange_free_due && !grad_in_flight && noGradHolders()) {
      freeExchange();
      exchange_free_due = false;
    }

    // The copies serving the next step. A clone decision takes effect at the
    // first step inside the balancing window after it arrived, after this
    // step's update (executeClone, from endStep): under the split barrier
    // that is up to lag steps before the round's resume, and the new copy
    // registers into the next round (its PE's barrier for this one has
    // fired, or the strategy could not have run). A retire takes effect in
    // the step the balancer released this copy in, at its end.
    // Every copy of the group knows the round's decision (ckClone / ckRetire
    // on the copy concerned, the notices on the others) before the round
    // releases it, so all announce the same count in this step's outputs and
    // act at its end: the new copy starts, or the retired one ends, at the
    // next step.
    if (!serve_only && resumed_by_lb) res_at[rank()] = my_step;   // tagged in this step's exchange
    if (serve_only) {
      nc_next = nc;   // the runtime's count (beginStep)
    } else if (nc_from <= my_step) {   // no change in flight: this step may decide one
      nc_next = nc;
      // The round's decisions, at the step every copy of the expert agrees on (resumedEverywhere).
      const bool lb_agreed = resumedEverywhere();
      if (lb_agreed)
        for (int r = 0; r < K_MAX_COPIES; r++) res_at[r] = -1;
      if (lb_agreed && clone_decided) nc_next = nc + newCopies();
      if (lb_agreed && retire_decided) {
        nc_next = nc - 1;
        retire_me = (rank() == decided_rank);
      }
      // Announced in this step's outputs and the next one's, in effect from the step after: the dispatchers send
      // tokens a step ahead, so the step after this one is routed before they could know (Dispatcher::noteOutput).
      if (nc_next != nc) nc_from = my_step + 2;
    }
    // The last step under the old count: a retiring copy says so in its exchange and goes at its end.
    if (retire_me && my_step + 1 == nc_from) { retiring = true; retire_me = false; }
    mult_next = nc_next;

    // Only the primary copy reports the weight checksum; every copy has the
    // bits, and at trace level 2 each prints its own, so they can be compared.
    chk_due = checksum_freq > 0 && (my_step % checksum_freq == 0) && (rank() == 0 || traceLevel() >= 2);
    TRACE(2, "expert %d.%d step %d compute n=%d%s", expert(), rank(), my_step, n_tot,
        replicated ? " (one of two copies)" : "");
    const double ti0 = tnow();
    int chunks = 0;
    if (use_bf16 && wb_stale) {
      // The master weights are new on this device (init, a migration, a
      // clone's state, all complete before this entry runs): remake the bf16
      // copies on the lane, ahead of the GEMMs that read them. Every update
      // from here on rewrites the copies in its own pass.
      moeCastBf16(W1, W1b, wsz(), g->stream);
      moeCastBf16(W2, W2b, wsz(), g->stream);
      wb_stale = false;
    }
    static const bool slab_probe = getenv("CHARM_MOE_SLABPROBE") != NULL;
    if (slab_probe && !use_bf16 && my_step <= 2 && send_ahead) {
      for (int sd = 0; sd < n_disp; sd++) {
        if (n_src[sd] <= 0) continue;
        double sums[2];
        for (int q = 0; q < 2; q++) {
          hapiCheck(hapiMemsetAsync(d_chk, 0, sizeof(double), g->stream));
          moeSumSqAdd((const float*)(x_seg + (size_t)q * segBytes() * n_disp + (size_t)sd * segBytes()),
              (size_t)n_src[sd] * d_model, g, d_chk, g->stream);
          hapiCheck(hapiMemcpyAsync(h_chk, d_chk, sizeof(double), hapiMemcpyDeviceToHost, g->stream));
          hapiCheck(hapiStreamSynchronize(g->stream));
          sums[q] = *h_chk;
        }
        const uintptr_t a1 = (uintptr_t)(x_seg + segBytes() * n_disp + (size_t)sd * segBytes());
        const uintptr_t unit = (uintptr_t)256 << 20;
        CkPrintf("[SLAB] expert %d step %d disp %d n=%d parity0 sumsq %.17g parity1 sumsq %.17g (reads parity %d) "
                 "parity1 seg at %#lx, %zu MiB to the next 256 MiB boundary, %zu MiB long\n",
            expert(), my_step, sd, n_src[sd], sums[0], sums[1], my_step & 1, (unsigned long)a1,
            (size_t)((unit - (a1 % unit)) >> 20), (size_t)(((size_t)n_src[sd] * d_model * actEsz()) >> 20));
      }
    }
    hapiCheck(hapiEventRecord(ev_start, g->stream));
    if (serve_only) {
      // Serving: the forward of every partition here, or of this copy's
      // block of them; no gradient, no update, no exchange. The weight
      // checksum is what it was. (A copy's dispatchers are its block's, so
      // the other partitions hold no tokens.)
      // One run over all of them (p = -1): with a partition per dispatcher, a run per partition would split an
      // unreplicated expert's tokens into n_disp small products.
      chunks += runPartition(g, -1, NULL, NULL);
      applyUpdate(g, NULL, NULL, NULL, NULL, 0);
    } else if (!replicated) {
      // Both partitions here, each into its own lane accumulator, and the
      // update from the two: the same operands and the same add as two
      // copies exchanging partials, so the bits agree either way.
      // The fixed tree over the four partitions: node (P0 + P1) into pair 0
      // (pair 1 as the second leaf's scratch), node (P2 + P3) into pair 1
      // (pair 2 as scratch); the root add is the SGD kernel's.
      float *a1, *a2, *b1, *b2;
      chunks += treeNode(g, 0, g->acc1[0], g->acc2[0], g->acc1[1], g->acc2[1], a1, a2);
      chunks += treeNode(g, 2, g->acc1[1], g->acc2[1], g->acc1[2], g->acc2[2], b1, b2);
      applyUpdate(g, a1, a2, b1, b2, n_tot);
    } else {
      // This copy's block of partitions, folded into its own accumulator; the
      // update follows the exchange of block sums (applyReplicated).
      float *o1, *o2;
      chunks += treeNode(g, blockLo(rank(), ncopies()), dWown1, dWown2, dWownS1, dWownS2, o1, o2);
      hapiCheck(hapiEventRecord(ev_grad, g->stream));
    }
    TRACE(2, "expert %d.%d step %d issued %d chunks in %.1f ms", expert(), rank(),
        my_step, chunks, (tnow() - ti0) * 1e3);
    hapiCheck(hapiEventRecord(ev_end, g->stream));
    if (n_tot > 0) {
      ev_pending = true;
      n_last = n_tot;
      step_last = my_step;
    }
    if (n_tot > 0 || chk_due) {
      CkCallback* cb = new CkCallback(CkIndex_Expert::computeDone(), self());
      hapiAddCallback(g->stream, cb);
    } else {
      self().computeDone();
    }
    // Each send is tagged with the send stream after a wait on its own
    // dispatcher's event, so the runtime takes that point as "data ready":
    // across nodes it holds the envelope until then, on the node the receiver
    // polls the flag. Nothing waits on the host.
    if (n_tot > 0 && fuse_sources && stream_outputs) {
      phase = 2;   // a migration is held until every send has completed
      for (int s = 0; s < n_disp; s++) {
        if (!serves(s)) continue;
        hapiCheck(hapiStreamWaitEvent(send_stream, ev_seg[s], 0));
        sendOutput(s);
      }
      outputs_sent = true;
      TRACE(2, "expert %d.%d step %d outputs queued behind their chunks", expert(), rank(), my_step);
    }
    // Serving: nothing between the copies (2026-11-09). Each reports its own step to Main (endStep); the notes they
    // traded made every copy of an expert wait each step for the slowest of them, and under split routing a copy
    // with no tokens in a step deadlocked its expert (SmartMoE, 16 GPUs: expert 5's nine copies at step 23).
    if (replicated && !serve_only) sendGradPartial();
  }

  // Serving: the other copy gets this step's figures alone (tokens, PE,
  // device time, whether this copy retires); nothing is in flight after.
  void sendPeerStats() {
    for (int r = 0; r < nc; r++)
      if (r != rank()) peerAt(r).peerStats(my_step, rank(), n_tot, CkMyPe(), gpu_last, exchangeFlags());
  }

  // This copy's partial to the other copy, behind its own kernels on the
  // device; dWown is rewritten only after gradSent (runStep). Sent whole even
  // when this copy saw no tokens -- n_mine 0 tells the other copy to add
  // nothing -- so the message shape never changes.
  void sendGradPartial() {
    hapiCheck(hapiStreamWaitEvent(send_stream, ev_grad, 0));
    grad_in_flight = true;
    grad_bufs_done = 0;
    const CkCallback done(CkIndex_Expert::gradBufDone(), self());
    if (gradChannelsOn()) {
      // One put of the pair into each peer's landing channel, each its own
      // batch; the figures go as peerStats, as in serving.
      grad_bufs_expected = nc - 1;
      for (int r = 0; r < nc; r++) {
        if (r == rank()) continue;
        CkDeviceChannel* ch = grad_hold[hIdx(r, my_step & 1)];
        if (ch == nullptr)
          CkAbort("Expert %d.%d on PE %d step %d: no exchange channel of copy %d\n", expert(), rank(), CkMyPe(), my_step, r);
        ch->put(dWown1, 2 * sizeof(float) * wsz(), send_stream, done, (uint64_t)my_step);
        CkRdmaDeviceChannelBatchFlush();
      }
      sendPeerStats();
      TRACE(2, "expert %d.%d step %d block-sum pair of %d tokens put to %d other cop%s%s", expert(), rank(),
          my_step, n_tot, nc - 1, nc - 1 == 1 ? "y" : "ies", retiring ? " (retiring)" : "");
      return;
    }
    grad_bufs_expected = 2 * (nc - 1);
    for (int r = 0; r < nc; r++) {
      if (r == rank()) continue;
      peerAt(r).gradPartial(my_step, rank(), n_tot, CkMyPe(), gpu_last, exchangeFlags(), (int)wsz(),
          CkDeviceBuffer(dWown1, done, send_stream), CkDeviceBuffer(dWown2, done, send_stream));
    }
    TRACE(2, "expert %d.%d step %d block sum of %d tokens sent to %d other cop%s%s", expert(), rank(),
        my_step, n_tot, nc - 1, nc - 1 == 1 ? "y" : "ies", retiring ? " (retiring)" : "");
  }
  void gradBufDone() {
    if (++grad_bufs_done < grad_bufs_expected) return;
    if (retiring) {
      // No next step will wait for gradSent; the teardown does instead.
      grad_in_flight = false;
      maybeFinishRetire();
    } else {
      self().gradSent();
    }
  }

  // Post entry method: the other copy's partial lands in this step parity's
  // landing buffers on the comm stream.
  void gradPartial(int ref, int from_rank, int n_mine, int pe, double gpu_s, int retiring_, int& n,
      float*& g1, float*& g2, CkDeviceBufferPost* devicePost) {
    // A new copy: the other copies' block sums for its first step can land before its state
    // (their step ends while the state is still in flight from another PE; on Frontier the
    // partial beat a 384 MB state). The landing buffers do not depend on the copy count, so
    // they are made here; the count is set when the copy starts (startCopy), and the SDAG
    // loop takes the partial when its step comes. Nothing else can arrive early: a copy's
    // next block sum waits for this copy's.
    if (!started) {
      if (W1 == NULL) { allocate(); createEvents(); }
      allocateExchange(nc_next > 1 ? nc_next : K_MAX_COPIES);   // the count is not yet known before the state lands
    }
    if ((started && !replicated) || from_rank == rank() || from_rank >= K_MAX_COPIES ||
        dWpeer1[slotOf(from_rank)][ref & 1] == NULL)
      CkAbort("Expert %d.%d on PE %d: a block sum from copy %d for step %d but this copy has %d copies\n",
          expert(), rank(), CkMyPe(), from_rank, ref, nc);
    if (started && (ref < my_step || ref > my_step + 1))
      CkAbort("Expert %d.%d on PE %d: block sum for step %d arrived during step %d\n",
          expert(), rank(), CkMyPe(), ref, my_step);
    g1 = dWpeer1[slotOf(from_rank)][ref & 1];
    g2 = dWpeer2[slotOf(from_rank)][ref & 1];
    devicePost[0].hapi_stream = commStream();
    devicePost[1].hapi_stream = commStream();
    TRACE(3, "expert %d.%d step %d partial for step %d posted", expert(), rank(), my_step, ref);
  }

  void noteGrad(int from_rank, int n_mine, int pe, double gpu_s, int retiring_) {
    peer_n[from_rank] = n_mine;
    peer_pe[from_rank] = pe;
    peer_gpu[from_rank] = gpu_s;
    peer_ret[from_rank] = (retiring_ & 1) != 0;
    if (retiring_ & 2) res_at[from_rank] = my_step;
    TRACE(2, "expert %d.%d step %d block sum of %d tokens from copy %d in%s%s", expert(), rank(), my_step,
        n_mine, from_rank, peer_ret[from_rank] ? " (it retires)" : "", (retiring_ & 2) ? " (resumed)" : "");
  }
  // Bit 0: this copy retires at this step's end; bit 1: the balancer released this copy in this step.
  int exchangeFlags() const { return (retiring ? 1 : 0) | (res_at[rank()] == my_step ? 2 : 0); }
  // Training: the step all copies of the expert act on a round's decisions in. Each copy parks for the round at a
  // step of its own (the split barrier's park waits for this PE's moves, AtSyncStepInFlight), so the resume comes
  // at different steps on different copies; acting at the resume, copy 0 of expert 10 went to one copy from step 113
  // and copy 1 from step 114 (2026-11-09, training on 16 GPUs, rounds every 10 steps), and copy 0 waited for slabs
  // that dispatchers 8-15 sent to copy 1. Each copy tags the exchange of the step it resumed in (exchangeFlags); all
  // act in the first step after the last tag. Every copy has read every tag by then (the update before that step
  // waited for all block sums of the step before), and none counts a tag of that step or later, even one in early.
  // A lone copy acts at its resume.
  bool resumedEverywhere() const {
    if (nc == 1) return res_at[0] >= 0;
    for (int r = 0; r < nc; r++)
      if (res_at[r] < 0 || res_at[r] >= my_step) return false;
    return true;
  }

  // Copy r's block sum this step: its own, or the landed one; NULL when it saw no tokens.
  float* blockSum1(int r) { return r == rank() ? (n_tot > 0 ? dWown1 : NULL) : (peer_n[r] > 0 ? dWpeer1[slotOf(r)][my_step & 1] : NULL); }
  float* blockSum2(int r) { return r == rank() ? (n_tot > 0 ? dWown2 : NULL) : (peer_n[r] > 0 ? dWpeer2[slotOf(r)][my_step & 1] : NULL); }
  // One node of the tree, partitions pA and pA + 1, from the block sums: the
  // one block that holds both, or the two blocks' sums added. The add goes
  // into a landed buffer, never this copy's own (a send may still read it);
  // the order of the two operands does not matter, fp32 addition is
  // commutative, and the unreplicated expert adds the same two leaves.
  void treeNodeFromBlocks(MoeGpu* g, int pA, float*& o1, float*& o2) {
    const int rA = blockOfPartition(pA, nc), rB = blockOfPartition(pA + 1, nc);
    float *a1 = blockSum1(rA), *a2 = blockSum2(rA);
    if (rA == rB) { o1 = a1; o2 = a2; return; }
    float *b1 = blockSum1(rB), *b2 = blockSum2(rB);
    if (!a1) { o1 = b1; o2 = b2; return; }
    if (!b1) { o1 = a1; o2 = a2; return; }
    float *d1 = a1, *d2 = a2, *s1 = b1, *s2 = b2;
    if (rA == rank()) { d1 = b1; d2 = b2; s1 = a1; s2 = a2; }
    moeAddInto(g, d1, s1, wsz());
    moeAddInto(g, d2, s2, wsz());
    o1 = d1; o2 = d2;
  }

  // The update of a replicated copy: the two tree nodes assembled from its
  // own block sum and the landed ones, then the same SGD as every copy's.
  void applyReplicated() {
    if (t_compute_done > 0.0) {
      xchg_wait_s += tnow() - t_compute_done;
      xchg_waits++;
    }
    // On the copy's own exchange stream, behind its block sum (ev_grad, on the
    // lane); the landed ones are complete, the runtime delivered their entries
    // after the landings.
    MoeGpu* g = &xg;
    hapiCheck(hapiStreamWaitEvent(g->stream, ev_grad, 0));
    float *a1, *a2, *b1, *b2;
    treeNodeFromBlocks(g, 0, a1, a2);
    treeNodeFromBlocks(g, 2, b1, b2);
    applyUpdate(g, a1, a2, b1, b2, (int)(n_tot + peerTokens()));
    hapiCheck(hapiEventRecord(ev_end, g->stream));   // the clone's state send orders behind the update
    CkCallback* cb = new CkCallback(CkIndex_Expert::applyDone(), self());
    hapiAddCallback(g->stream, cb);
  }

  void sendOutput(int s) {
    const int n = std::max(n_src[s], 0);
    if (out_channels) {
      // Into the dispatcher's segment for this expert, the step's parity; an
      // empty output is a note alone. The tag: the step and the copies
      // announced for the next one.
      CkDeviceChannel* ch = out_ch[(my_step & 1) * n_disp + s];
      if (ch == nullptr)
        CkAbort("Expert %d.%d on PE %d step %d: no output channel of dispatcher %d\n", expert(), rank(), CkMyPe(),
            my_step, s);
      outstanding_sends++;
      ch->put(y_seg + s * segBytes(), (size_t)n * d_model * actEsz(), send_stream,
          CkCallback(CkIndex_Expert::sendDone(), self()), ((uint64_t)my_step << 8) | (uint64_t)mult_next);
      // Its own batch, now: the runtime gathers an entry method's puts into one
      // copy flushed when the method ends, which would hold every dispatcher's
      // output until the run's last chunk (stream_outputs issues them as each
      // dispatcher's chunk completes; batched, ShedLB training on the A40 nodes
      // went from 50 to 63 ms a step, job 23612036).
      CkRdmaDeviceChannelBatchFlush();
      return;
    }
    const int len = (int)(std::max(n, 1) * d_model * actEsz());   // bytes
    disp_proxy[dispPe(s)].receiveOutput(my_step, expert(), n, moeNowNs(), CkMyPe(), mult_next, len,
        (outstanding_sends++,
         CkDeviceBuffer(y_seg + s * segBytes(),
             CkCallback(CkIndex_Expert::sendDone(), self()),
             send_stream)));
  }

  void sendOutputs() {
    t_compute_done = tnow();
    releaseLane();   // the step's kernels are complete (computeDone)
    if (tok_channels) {
      // The kernels that read this step's slab are done: the dispatchers may
      // put into it again (the step after next, or the next without send_ahead).
      // Only the channels that were put into this step: the other copy's
      // dispatchers never sent here.
      const int q = my_step % tokParities();
      for (int d = 0; d < n_disp; d++)
        if (n_src[d] >= 0) tok_ch[q * n_disp + d].release();
    }
    if (!outputs_sent) {
      phase = 2;
      const double ts0 = tnow();
      if (n_tot > 0) hapiCheck(hapiStreamWaitEvent(send_stream, ev_end, 0));
      for (int s = 0; s < n_disp; s++)
        if (serves(s)) sendOutput(s);
      TRACE(2, "expert %d.%d step %d outputs sent", expert(), rank(), my_step);
      TRACE(2, "expert %d.%d step %d issued %d sends in %.1f ms", expert(), rank(),
          my_step, servedCount(), (tnow() - ts0) * 1e3);
    }
    gateDrain();
  }

  // The next step's kernels rewrite the output segments, and under async LB a
  // migration's pup frees them; neither may happen over a live transport read.
  void gateDrain() {
    if (outstanding_sends == 0) self().sendsDrained();
    else drain_pending = true;
  }

  void sendDone() {
    outstanding_sends--;
    if (drain_pending && outstanding_sends == 0) {
      drain_pending = false;
      self().sendsDrained();
    }
  }

  // ---- making a copy ----
  // At the end of the step the balancer released this copy in: in training
  // the copies' state goes now, after this step's update, through the
  // runtime (ckCloneTo), and the next step's compute waits for the pack to
  // have read the weights (ckCloneSent -> cloneSent); in serving it went at
  // the decision. Either way each copy is then told its first step, a
  // message that waits for the copy if its state has not landed yet.
  void executeClone() {
    if (!serve_only) {
      clone_in_flight = true;
      clone_bufs_done = 0;
      clone_bufs_expected = n_clone;
      for (int i = 0; i < n_clone; i++) ckCloneTo(clone_to_pe[i], CkArrayIndex2D(expert(), clone_rank[i]));
    }
    for (int i = 0; i < n_clone; i++) {
      TRACE(1, "expert %d.%d step %d: copy %d of %d on PE %d starts at step %d", expert(), rank(), my_step,
          clone_rank[i], nc_next, clone_to_pe[i], my_step + 1);
      // Training: told only once the copy exists (copyWhole). Sent here, right after the clone was begun, the start
      // could reach no element (2026-11-09: expert 29's copy 1, made again at the round at 110 after an earlier
      // copy 1 had retired, never started, and its dispatchers routed to it without its channels at step 113).
      const int r = clone_rank[i];
      if (!serve_only) {
        if (clone_start_step.empty()) clearCloneStarts();
        clone_start_step[r] = my_step + 1;
        clone_start_adam[r] = adam_t;
        clone_start_spt[r] = spt;
        clone_start_nc[r] = nc_next;
        if (clone_whole[r]) sendCloneStart(r);
      } else {
        expert_proxy(expert(), r).startCopy(my_step + 1, adam_t, spt, nc_next);
      }
    }
    n_clone = 0;
  }
  // The source's side of a new training copy's start: its step and figures (executeClone) and whether the copy has
  // reported its state in (copyWhole); the start goes when both are known.
  std::vector<int> clone_start_step, clone_start_adam, clone_start_nc;
  std::vector<double> clone_start_spt;
  std::vector<char> clone_whole;
  int clone_source = -1;   // a copy: the rank it was made from
  void clearCloneStarts() {
    clone_start_step.assign(K_MAX_COPIES, -1);
    clone_start_adam.assign(K_MAX_COPIES, 0);
    clone_start_nc.assign(K_MAX_COPIES, 0);
    clone_start_spt.assign(K_MAX_COPIES, 0.0);
    clone_whole.assign(K_MAX_COPIES, 0);
  }
  void sendCloneStart(int r) {
    expert_proxy(expert(), r).startCopy(clone_start_step[r], clone_start_adam[r], clone_start_spt[r], clone_start_nc[r]);
    clone_start_step[r] = -1;
    clone_whole[r] = 0;
  }
  void copyWhole(int r) {
    if (clone_start_step.empty()) clearCloneStarts();
    if (clone_start_step[r] >= 0) sendCloneStart(r);
    else clone_whole[r] = 1;
  }
  // The runtime: the pack of one copy has finished reading this copy's buffers.
  void ckCloneSent() {
    if (!clone_in_flight) return;   // serving: nothing waits
    if (++clone_bufs_done == clone_bufs_expected) self().cloneSent();
  }

  // A copy made by the runtime (ckCloneTo): the unpack gave it the weights
  // (and moments); everything of the source's step, channels and decisions
  // it must not inherit is cleared here (called from the pup's cloning
  // branch after allocate). It serves nothing until startCopy.
  void resetAsCopy() {
    recv_count = 0; n_expect = 0; phase = 0;
    n_src.assign(n_src.size(), 0);
    n_tot = 0;
    outstanding_sends = 0; outputs_sent = 0; drain_pending = false; chk_due = false;
    lb_waiting = false; lb_start_step = 0; awaiting_resume = false; resumed_by_lb = false;
    started = false; replicated = false; nc = 1;
    grad_in_flight = false; grad_bufs_done = 0;
    clone_in_flight = false; clone_bufs_done = 0; clone_bufs_expected = 0;
    clone_decided = false; retire_decided = false; decided_rank = -1;
    for (int r = 0; r < K_MAX_COPIES; r++) decided_new[r] = false;
    n_clone = 0; retire_pending = false; retiring = false; exchange_free_due = false;
    nc_from = 0; retire_me = false;
    for (int r = 0; r < K_MAX_COPIES; r++) res_at[r] = -1;
    state_whole = false; start_pending = false;   // until ckJustCloned
    mult_next = 1;
    tok_ch.clear(); ch_open.clear(); slab_held.clear();
    arrived_by_pup = grad_by_pup = out_by_pup = false;
    // The bf16 operands are made from the masters at the first compute, unless they came as the state (bf16 serving).
    static const bool fp32_state = getenv("MOE_FP32_STATE") != NULL;
    wb_stale = use_bf16 != 0 && (!serve_only || fp32_state);
  }
  // The runtime made this copy (on its source's PE, from where it moves to
  // the balancer's PE when its Stable weights have landed there, serving
  // wherever it is meanwhile).
  void ckJustCloned() {
    TRACE(1, "expert %d.%d: a copy, state in; waiting for its first step", expert(), rank());
    // Made on its source's PE, a copy hears this from a callback once its unpack has completed on the device
    // (CkLocMgr::cloneUnpackedFor, 2026-11-08): a start told before then waits for it.
    state_whole = true;
    if (serve_only) {
      // Opened in an entry method of this copy (openCopyChannels): the runtime attributes a channel's owner from
      // the entry that opens it, and only then closes it to its senders when the copy moves (2026-11-08: opened
      // here, a copy the balancer then moved left its senders waiting for credits).
      self().openCopyChannels();
      return;
    }
    if (start_pending) { start_pending = false; startCopy(pend_start[0], pend_start[1], pend_spt, pend_start[2]); }
    else if (clone_source >= 0 && !started) peerAt(clone_source).copyWhole(rank());   // the source sends the start now
  }
  bool state_whole = true;      // a copy: its state has been unpacked (ckJustCloned)
  bool replica_gone = false;    // serving: past the runtime's count, this copy ends
  void openCopyChannels() {
    openChannels(1);   // to every dispatcher; the runtime routes a dispatcher here from an agreed step
    ckReplicaReady(rank());
  }
  // Serving: the runtime knows the copy count of the step about to begin (a change being agreed may not be).
  bool copiesKnown() {
    if (!serve_only) return true;
    if (split_route && disp_proxy.ckLocalBranch()->planOf(my_step + 1) == nullptr) {
      disp_proxy.ckLocalBranch()->waitPlan(CkCallback(CkIndex_Expert::replicaAgreed(), self()));
      return false;
    }
    const CkArrayID aid = expert_proxy.ckGetArrayID();
    if (CkReplicaCount(aid, expert(), my_step + 1) >= 0) return true;
    CkReplicaWhenAgreed(aid, expert(), CkCallback(CkIndex_Expert::replicaAgreed(), self()));
    return false;
  }
  void replicaGone() {
    TRACE(1, "expert %d.%d step %d: past the runtime's count, retires", expert(), rank(), my_step);
    for (size_t i = 0; i < tok_ch.size(); i++)
      if (i < ch_open.size() && ch_open[i]) tok_ch[i].close();
    if (out_channels) dropOutChannels();
    self().die();
  }
  bool start_pending = false;   // startCopy arrived first
  int pend_start[3] = {0, 0, 0};
  double pend_spt = 0.0;
  // From the source, at the end of the step the balancer released it in: the
  // copy serves its block from `step`.
  void startCopy(int step, int adam_t_, double spt_, int ncopies_) {
    if (!state_whole) {
      start_pending = true;
      pend_start[0] = step; pend_start[1] = adam_t_; pend_start[2] = ncopies_; pend_spt = spt_;
      return;
    }
    my_step = step - 1;   // beginStep makes it `step`
    adam_t = adam_t_;
    spt = spt_;
    nc_next = ncopies_;
    TRACE(1, "expert %d.%d: starts as a copy at step %d of %d copies", expert(), rank(), step, ncopies_);
    started = true;
    setCopies(nc_next);
    openChannels();
    self().runStep();
  }

  // ---- retiring ----
  // The dispatchers this copy served drop their channels at the start of the
  // next step and say so; once the last step is over and every send is done,
  // the channels close and the copy goes.
  void channelsDropped() {
    chans_dropped++;
    TRACE(2, "expert %d.%d: %d of %d dispatchers dropped their channels", expert(), rank(),
        chans_dropped, servedCount());
    maybeFinishRetire();
  }
  void maybeFinishRetire() {
    if (!retiring || !step_ended || grad_in_flight || chans_dropped < servedCount()) return;
    if (gradChannelsOn()) {
      // This copy's holder copies of the peers' channels go first (they are
      // told); its own channels close once every peer has dropped its copies.
      for (int r = 0; r < nc; r++)
        if (r != rank()) dropGradHolders(r);
      if (!noGradHolders()) return;
      for (size_t i = 0; i < grad_ch.size(); i++)
        if (grad_open[i]) { grad_ch[i].close(); grad_open[i] = 0; }
    }
    TRACE(1, "expert %d.%d step %d: retired", expert(), rank(), my_step);
    for (size_t i = 0; i < tok_ch.size(); i++)
      if (ch_open[i]) tok_ch[i].close();
    if (out_channels) dropOutChannels();
    self().die();   // off this entry method: a message to itself
  }
  void die() { if (serve_only) ckReplicaDone(); else ckDestroy(); }

  void endStep() {
    if (!retiring) {
      // One report per expert per step, from the primary copy: both copies'
      // tokens and device time (the other's rode in with its partial), and
      // the weight checksum.
      if (chk_due) TRACE(2, "expert %d.%d step %d w2 %.17g", expert(), rank(), my_step, *h_chk);
      if (serve_only) {
        // Every copy its own report: Main completes the expert's step at its copy count.
        const double w2 = (chk_due && rank() == 0) ? *h_chk : 0.0;
        int pe = CkMyPe();
        long n = n_tot;
        double g = gpu_last;
        main_proxy.expertStep(expert(), my_step, (long)n_tot, w2, 1, &pe, &n, &g, devPool()->spt_gpu, devPool()->spt_tok,
            nc, rank());
      } else if (rank() == 0) {
        const double w2 = chk_due ? *h_chk : 0.0;
        const MoeCtx* c = ctx();
        std::vector<int> pes(K_MAX_COPIES); std::vector<long> ns(K_MAX_COPIES); std::vector<double> gs(K_MAX_COPIES);
        int np = 0;
        pes[np] = CkMyPe(); ns[np] = n_tot; gs[np] = gpu_last; np++;
        for (int r = 1; r < nc; r++) { pes[np] = peer_pe[r]; ns[np] = peer_n[r]; gs[np] = peer_gpu[r]; np++; }
        main_proxy.expertStep(expert(), my_step, (long)n_tot + peerTokens(), w2, np, pes.data(), ns.data(), gs.data(),
            devPool()->spt_gpu, devPool()->spt_tok, 1, 0);
      }
      if (nc_next != nc && my_step + 1 >= nc_from) {
        // The round's decision takes effect: a copy more (this copy sends the
        // state when the balancer named it) or one fewer (the highest rank,
        // which is not this copy here). Channels for a grown block go out
        // now; a copy left alone frees its exchange buffers once this step's
        // block sum has been read by the transport (gradSent).
        if (nc_next > nc && n_clone > 0) executeClone();
        TRACE(1, "expert %d.%d step %d: %d copies from step %d", expert(), rank(), my_step, nc_next, my_step + 1);
        const int k_old = nc;
        setCopies(nc_next);
        refreshChannels(k_old, nc);
        // One fewer: the highest rank retires; its channels are of no use.
        // Not while this copy's last block sum is still being put into it: the put's completion (gradBufDone, then
        // gradSent, which the next step waits for) goes through the holder, and deleting it first lost it (2026-11-09,
        // training on 16 GPUs: the remaining copy of three experts whose copy 1 retired at step 24 never began
        // step 25). Dropped once gradSent is in (dropGradAfterSend).
        if (gradChannelsOn() && nc_next < k_old) {
          if (grad_in_flight) grad_drop_after = k_old - 1;
          else dropGradHolders(k_old - 1);
        }
        if (nc == 1) exchange_free_due = true;
        clone_decided = retire_decided = false;
        decided_rank = -1;
        for (int r = 0; r < K_MAX_COPIES; r++) decided_new[r] = false;
        n_clone = 0;
      }
    }
    if (retiring) {
      step_ended = true;
      maybeFinishRetire();
    } else if (my_step < warmup_steps + n_steps) {
      self().runStep();
    } else {
      if (print_place)
        CkPrintf("Expert %d.%d on PE %d: %d tokens in the last step\n", expert(), rank(),
            CkMyPe(), n_tot);
      if (flight_log) moeFlightDump();
      if (rank() == 0) main_proxy.expertDone(expert());
    }
  }
};

#include "moe.def.h"
