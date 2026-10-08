// SM-aware GPU load measurement, in a controlled setting.
//
// jacobi2d with one knob added: how many SMs each block's stencil kernel
// occupies. The kernel is launched as (sms x blocks-per-SM) blocks of 1024
// threads that stride over the tile, so the runtime's occupancy model
// (hapi computeKernelSMs: min(SMs, ceil(blocks / max active blocks per SM)))
// reads exactly `sms` for it, and the work per kernel is the whole tile
// whatever the knob. A kernel on s SMs therefore takes about SMs/s times
// longer than one on the full device: a duration-based load would scale the
// same way, while the SM-seconds the runtime charges -- duration x s / SMs
// when the device is not oversubscribed (hapiNormalizeCuptiLoads) -- should
// not move.
//
// Every block times its own stencil kernels with CUDA events over each
// balancing interval and, when the step resumes it, asks the runtime what
// load that step read for it (getObjLastGPUTime). The ratio
//     measured / (duration x sms / SMs)
// is what this benchmark checks: 1, within -t, for every block and every
// mix of SM counts, as long as the blocks sharing a device ask for no more
// SMs than it has. Above that the runtime splits each busy interval among
// the kernels holding the device and the isolated expectation no longer
// applies; the benchmark says so instead of judging.
//
// Knobs: -s lo[:hi] SM counts and -P how they are assigned to blocks
// (uniform, checker, split), -m stencil repeats per cell (the duration),
// -F pin every block in place (measurement only: a balancer sees the loads
// but can move nothing), -t the verdict's tolerance, -v a line per block per
// step. Without -F the same runs show what a balancer does with the loads:
// a checkerboard of 8- and 84-SM blocks is balanced by SM-seconds and
// nothing should move; under +gpuloadbusy (duration split) the 8-SM blocks
// look ten times heavier and the balancer moves them.

#include "hapi.h"
#include "smload.decl.h"
#include "smload.h"
#include <utility>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int grid_width;
/* readonly */ int grid_height;
/* readonly */ int block_width;
/* readonly */ int block_height;
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ int lb_freq;
/* readonly */ int first_lb;
/* readonly */ int work_iters;
/* readonly */ int sms_lo;
/* readonly */ int sms_hi;
/* readonly */ int sm_pattern;   // 0 uniform, 1 checker, 2 split
/* readonly */ int verbose;
/* readonly */ bool fixed_placement;
/* readonly */ double tolerance;

extern void invokeInitKernel(DataType* d_temperature, int block_width,
    int block_height, cudaStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, bool left_bound, bool right_bound, bool top_bound,
    bool bottom_bound, cudaStream_t stream);
extern void invokeJacobiSM(const DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int iter, int nblocks, cudaStream_t stream);
extern int smloadBlocksPerSM();
extern void invokePackingKernels(const DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, bool left_bound, bool right_bound, int block_width,
    int block_height, cudaStream_t stream);
extern void invokeUnpackingKernel(DataType* d_temperature, const DataType* d_ghost,
    bool is_left, int block_width, int block_height, cudaStream_t stream);

enum Direction { LEFT = 1, RIGHT, TOP, BOTTOM };

// The SM count a block launches with, from its index.
static int smsFor(int x, int y) {
  if (sms_hi <= 0) return sms_lo;
  switch (sm_pattern) {
    case 1: return ((x + y) & 1) ? sms_hi : sms_lo;       // checker: every PE gets both
    case 2: return (x < n_chares_x / 2) ? sms_lo : sms_hi;  // split: left half lo, right half hi
    default: return sms_lo;
  }
}

static const char* patternName(int p) {
  return p == 1 ? "checker" : p == 2 ? "split" : "uniform";
}

class Main : public CBase_Main {
  double init_start_time;
  double start_time;
  int total_sms;          // of device 0; the verdict's subscription estimate
  double subscription;    // worst-case sum of SMs asked per device / SMs
  int lb_reports;
  int evaluated;
  double worst_min, worst_max;
  double sum_moved;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    grid_width = 8192;
    grid_height = 4096;
    block_width = 1024;
    block_height = 1024;
    n_iters = 60;
    warmup_iters = 5;
    first_lb = 10;
    lb_freq = 10;
    work_iters = 16;
    sms_lo = 84;
    sms_hi = 0;
    sm_pattern = 0;
    verbose = 0;
    fixed_placement = false;
    tolerance = 0.10;
    lb_reports = 0;
    evaluated = 0;
    worst_min = 1e300;
    worst_max = -1e300;
    sum_moved = 0.0;

    int c;
    while ((c = getopt(m->argc, m->argv, "W:H:w:h:i:u:f:b:m:s:P:t:Fv")) != -1) {
      switch (c) {
        case 'W': grid_width = atoi(optarg); break;
        case 'H': grid_height = atoi(optarg); break;
        case 'w': block_width = atoi(optarg); break;
        case 'h': block_height = atoi(optarg); break;
        case 'i': n_iters = atoi(optarg); break;
        case 'u': warmup_iters = atoi(optarg); break;
        case 'f': first_lb = atoi(optarg); break;
        case 'b': lb_freq = atoi(optarg); break;
        case 'm': work_iters = atoi(optarg); break;
        case 's': {
          sms_lo = atoi(optarg);
          const char* colon = strchr(optarg, ':');
          sms_hi = colon ? atoi(colon + 1) : 0;
          break;
        }
        case 'P':
          if (strcmp(optarg, "uniform") == 0) sm_pattern = 0;
          else if (strcmp(optarg, "checker") == 0) sm_pattern = 1;
          else if (strcmp(optarg, "split") == 0) sm_pattern = 2;
          else { CkPrintf("Unknown -P %s (uniform, checker, split)\n", optarg); CkExit(); }
          break;
        case 't': tolerance = atof(optarg); break;
        case 'F': fixed_placement = true; break;
        case 'v': verbose = 1; break;
        default:
          CkPrintf(
              "Usage: %s -W [grid width] -H [grid height] -w [block width] -h [block height]\n"
              "  -i [iterations] -u [warmup] -f [first LB iteration] -b [LB period]\n"
              "  -m [stencil repeats per cell] -s lo[:hi] (SMs per kernel)\n"
              "  -P uniform|checker|split (how lo/hi are assigned to blocks)\n"
              "  -F (pin every block: measure only) -t [tolerance, 0.10] -v (per-block lines)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (grid_width % block_width != 0 || grid_height % block_height != 0)
      CkAbort("Invalid grid & block configuration\n");
    if (sms_lo <= 0 || (sms_hi < 0))
      CkAbort("-s needs positive SM counts\n");
    if (first_lb <= warmup_iters)
      CkPrintf("Note: -f %d is inside the warmup (-u %d); the first interval "
               "is measured anyway\n", first_lb, warmup_iters);

    n_chares_x = grid_width / block_width;
    n_chares_y = grid_height / block_height;

    hapiCheck(cudaDeviceGetAttribute(&total_sms, cudaDevAttrMultiProcessorCount, 0));
    const int blocks_per_sm = smloadBlocksPerSM();
    const int sms_max = sms_hi > sms_lo ? sms_hi : sms_lo;
    const int n_chares = n_chares_x * n_chares_y;
    const int per_gpu = (n_chares + CkNumNodes() - 1) / CkNumNodes();  // one GPU per process
    subscription = (double)per_gpu * sms_max / total_sms;

    CkPrintf("\n[smload] SM-aware GPU load benchmark (jacobi2d)\n");
    CkPrintf("[smload] grid %d x %d, block %d x %d, blocks %d x %d = %d (%d per process, %d processes)\n",
             grid_width, grid_height, block_width, block_height, n_chares_x, n_chares_y,
             n_chares, per_gpu, CkNumNodes());
    CkPrintf("[smload] iterations %d + %d warmup, LB at %d then every %d, %d stencil repeats per cell\n",
             n_iters, warmup_iters, first_lb, lb_freq, work_iters);
    if (sms_hi > 0)
      CkPrintf("[smload] SMs per kernel %d:%d, pattern %s\n", sms_lo, sms_hi, patternName(sm_pattern));
    else
      CkPrintf("[smload] SMs per kernel %d, uniform\n", sms_lo);
    CkPrintf("[smload] device 0: %d SMs, %d stencil block(s) per SM at 1024 threads, "
             "so a kernel on s SMs launches %d x s blocks\n", total_sms, blocks_per_sm, blocks_per_sm);
    CkPrintf("[smload] worst-case subscription %.2f (blocks per device x max SMs / SMs)%s\n",
             subscription, subscription > 1.0 ? " -- OVERSUBSCRIBED, no verdict" : "");
    CkPrintf("[smload] placement %s, tolerance %.0f%%\n\n",
             fixed_placement ? "FIXED (no block may move)" : "free (the balancer may move blocks)",
             tolerance * 100.0);
    if (sms_max > total_sms)
      CkPrintf("[smload] WARNING: -s %d exceeds the device's %d SMs; the runtime caps at %d\n",
               sms_max, total_sms, total_sms);

    block_proxy = CProxy_Block::ckNew(n_chares_x, n_chares_y);
    init_start_time = CkWallTimer();
    start_time = init_start_time;
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("[smload] init %.3f s\n", CkWallTimer() - init_start_time);
    block_proxy.exchangeGhosts();
  }

  void warmupDone() { start_time = CkWallTimer(); }

  void allDone() {
    const double total_time = CkWallTimer() - start_time;
    CkPrintf("[smload] total %.3f s, %.3f ms per iteration\n", total_time,
             total_time / n_iters * 1e3);
    block_proxy.checksum();
  }

  // Once per balancing step, after every block resumed: what the step read
  // against what the blocks timed.
  void lbReport(CkReductionMsg* m) {
    CkReduction::tupleElement* res = NULL;
    int n = 0;
    m->toTuple(&res, &n);
    const double* s = (const double*)res[0].data;
    const double mn = *(const double*)res[1].data;
    const double mx = *(const double*)res[2].data;
    lb_reports++;
    const double objs = s[3], moved = s[5];
    const double mean = objs > 0 ? s[4] / objs : 0.0;
    sum_moved += moved;
    CkPrintf("[smload] LB step %d: blocks %.0f (moved %.0f)  measured %.4f s  "
             "expected(SM) %.4f s  ratio mean %.3f min %.3f max %.3f  "
             "time-only %.4f s  host %.4f s\n",
             lb_reports, objs, moved, s[0], s[1], mean, objs > 0 ? mn : 0.0,
             objs > 0 ? mx : 0.0, s[2], s[6]);
    fflush(stdout);
    // The first interval also holds the init kernels and the warmup, which
    // the blocks do not time; judged from the second step on.
    if (lb_reports >= 2 && objs > 0) {
      evaluated++;
      if (mn < worst_min) worst_min = mn;
      if (mx > worst_max) worst_max = mx;
    }
    delete[] res;
    delete m;
  }

  void checksumDone(unsigned long long sum) {
    CkPrintf("[smload] checksum %016llx\n", sum);
    CkPrintf("[smload] moves over the run: %.0f\n", sum_moved);
    if (subscription > 1.0)
      CkPrintf("SMLOAD NO-VERDICT: the device is oversubscribed (%.2f x its SMs); "
               "the isolated expectation does not apply, expect ratios near 1/%.2f "
               "when every block's kernels overlap\n", subscription, subscription);
    else if (evaluated == 0)
      CkPrintf("SMLOAD NO-VERDICT: fewer than two balancing steps were reported\n");
    else if (worst_min >= 1.0 - tolerance && worst_max <= 1.0 + tolerance)
      CkPrintf("SMLOAD PASS: over %d step(s) every block's measured/expected ratio "
               "stayed in [%.3f, %.3f] (tolerance %.0f%%)\n",
               evaluated, worst_min, worst_max, tolerance * 100.0);
    else
      CkPrintf("SMLOAD FAIL: over %d step(s) the measured/expected ratio ranged "
               "[%.3f, %.3f], outside %.0f%% of 1\n",
               evaluated, worst_min, worst_max, tolerance * 100.0);
    fflush(stdout);
    CkExit();
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

 public:
  int my_iter;
  int neighbors;
  int remote_count;
  int x, y;
  int my_sms;
  int blocks_per_sm;
  int total_sms;

  // The interval's own accounting: stencil kernel time by CUDA events, and the
  // PE the block joined the step from (a block that moved reports nothing:
  // its object on the new PE has no last load).
  double win_dur_s;
  int win_kernels;
  int pe_at_sync;
  int lb_steps_seen;

  static const int RING = 8;
  cudaEvent_t t_begin[RING], t_end[RING];
  int ring_head, ring_tail;   // pending pairs are [tail, head)

  DataType* h_temperature;
  DataType* d_temperature;
  DataType* d_new_temperature;
  DataType* h_left_ghost;
  DataType* h_right_ghost;
  DataType* h_top_ghost;
  DataType* h_bottom_ghost;
  DataType* d_left_ghost;
  DataType* d_right_ghost;

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;
  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  bool left_bound, right_bound, top_bound, bottom_bound;

  Block() : win_dur_s(0.0), win_kernels(0), pe_at_sync(-1), lb_steps_seen(0),
            ring_head(0), ring_tail(0) {
    usesAtSync = true;
  }

  Block(CkMigrateMessage* m) : win_dur_s(0.0), win_kernels(0), pe_at_sync(-1),
                               lb_steps_seen(0), ring_head(0), ring_tail(0) {
    usesAtSync = true;
    createCudaState();
  }

  void createCudaState() {
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));
    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));
    for (int i = 0; i < RING; i++) {
      hapiCheck(cudaEventCreate(&t_begin[i]));   // timing enabled
      hapiCheck(cudaEventCreate(&t_end[i]));
    }
  }

  void allocate() {
    const size_t n = (size_t)(block_width + 2) * (block_height + 2);
    hapiCheck(hapiMallocHost((void**)&h_temperature, sizeof(DataType) * n));
    hapiCheck(hapiMalloc((void**)&d_temperature, sizeof(DataType) * n));
    hapiCheck(hapiMalloc((void**)&d_new_temperature, sizeof(DataType) * n));
    hapiCheck(hapiMallocHost((void**)&h_left_ghost, sizeof(DataType) * block_height));
    hapiCheck(hapiMallocHost((void**)&h_right_ghost, sizeof(DataType) * block_height));
    hapiCheck(hapiMallocHost((void**)&h_top_ghost, sizeof(DataType) * block_width));
    hapiCheck(hapiMallocHost((void**)&h_bottom_ghost, sizeof(DataType) * block_width));
    hapiCheck(hapiMalloc((void**)&d_left_ghost, sizeof(DataType) * block_height));
    hapiCheck(hapiMalloc((void**)&d_right_ghost, sizeof(DataType) * block_height));
  }

  ~Block() {
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(cudaFree(d_temperature));
    hapiCheck(cudaFree(d_new_temperature));
    hapiCheck(cudaFreeHost(h_left_ghost));
    hapiCheck(cudaFreeHost(h_right_ghost));
    hapiCheck(cudaFreeHost(h_top_ghost));
    hapiCheck(cudaFreeHost(h_bottom_ghost));
    hapiCheck(cudaFree(d_left_ghost));
    hapiCheck(cudaFree(d_right_ghost));
    hapiCheck(cudaStreamDestroy(compute_stream));
    hapiCheck(cudaStreamDestroy(comm_stream));
    hapiCheck(cudaEventDestroy(compute_event));
    hapiCheck(cudaEventDestroy(comm_event));
    for (int i = 0; i < RING; i++) {
      hapiCheck(cudaEventDestroy(t_begin[i]));
      hapiCheck(cudaEventDestroy(t_end[i]));
    }
  }

  void pup(PUP::er& p) {
    // The migration copies the grids off the device on a stream of its own;
    // settle ours first, and bank the kernel timings while the events are
    // still this copy's.
    if (p.isPacking()) {
      cudaStreamSynchronize(compute_stream);
      cudaStreamSynchronize(comm_stream);
      flushTimings(true);
    }
    p | my_iter;
    p | neighbors;
    p | remote_count;
    p | x;
    p | y;
    p | my_sms;
    p | blocks_per_sm;
    p | total_sms;
    p | left_bound;
    p | right_bound;
    p | top_bound;
    p | bottom_bound;
    p | win_dur_s;
    p | win_kernels;
    p | pe_at_sync;
    p | lb_steps_seen;

    if (p.isUnpacking()) allocate();

    // The outgoing ghosts are live from packGhosts() until sendGhosts() reads
    // them, and a step can move this block inside that window.
    PUParray(p, h_left_ghost, block_height);
    PUParray(p, h_right_ghost, block_height);
    PUParray(p, h_top_ghost, block_width);
    PUParray(p, h_bottom_ghost, block_width);

    p(d_temperature, (block_width + 2) * (block_height + 2), PUP::PUPMode::DEVICE);
    p(d_new_temperature, (block_width + 2) * (block_height + 2), PUP::PUPMode::DEVICE);
  }

  void init() {
    my_iter = 0;
    neighbors = 0;
    x = thisIndex.x;
    y = thisIndex.y;
    my_sms = smsFor(x, y);

    left_bound = right_bound = top_bound = bottom_bound = false;
    if (thisIndex.x == 0) left_bound = true; else neighbors++;
    if (thisIndex.x == n_chares_x - 1) right_bound = true; else neighbors++;
    if (thisIndex.y == 0) top_bound = true; else neighbors++;
    if (thisIndex.y == n_chares_y - 1) bottom_bound = true; else neighbors++;

    int dev = 0;
    hapiCheck(cudaGetDevice(&dev));
    hapiCheck(cudaDeviceGetAttribute(&total_sms, cudaDevAttrMultiProcessorCount, dev));
    blocks_per_sm = smloadBlocksPerSM();
    if (my_sms > total_sms) my_sms = total_sms;

    if (fixed_placement) setMigratable(false);

    allocate();
    createCudaState();

    if (verbose)
      CkPrintf("[smload] block (%d,%d) pe=%d id=%llu sms=%d launch=%d blocks\n",
               x, y, CkMyPe(), (unsigned long long)ckGetID().getID(), my_sms,
               my_sms * blocks_per_sm);

    invokeInitKernel(d_temperature, block_width, block_height, compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, compute_stream);
    invokeBoundaryKernels(d_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);
    invokeBoundaryKernels(d_new_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);

    hapiAddCallback(compute_stream, CkCallback(CkIndex_Block::initDone(), thisProxy[thisIndex]));
  }

  void initDone() {
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  void checksum() {
    const size_t n = (size_t)(block_width + 2) * (block_height + 2);
    std::vector<DataType> host(n);
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(comm_stream);
    hapiCheck(cudaMemcpy(host.data(), d_temperature, sizeof(DataType) * n,
                         cudaMemcpyDeviceToHost));
    uint64_t h = 1469598103934665603ULL;  // FNV-1a over the interior
    for (int j = 1; j <= block_height; j++) {
      const unsigned char* row =
          (const unsigned char*)(host.data() + (block_width + 2) * j + 1);
      for (size_t b = 0; b < sizeof(DataType) * (size_t)block_width; b++) {
        h ^= row[b];
        h *= 1099511628211ULL;
      }
    }
    contribute(sizeof(uint64_t), &h, CkReduction::sum_ulong_long,
               CkCallback(CkReductionTarget(Main, checksumDone), main_proxy));
  }

  // Bank every completed stencil timing; with `all`, wait for the rest.
  void flushTimings(bool all) {
    while (ring_tail < ring_head) {
      const int i = ring_tail % RING;
      if (all) {
        hapiCheck(cudaEventSynchronize(t_end[i]));
      } else if (cudaEventQuery(t_end[i]) != cudaSuccess) {
        break;
      }
      float ms = 0.0f;
      hapiCheck(cudaEventElapsedTime(&ms, t_begin[i], t_end[i]));
      win_dur_s += ms * 1e-3;
      win_kernels++;
      ring_tail++;
    }
  }

  void iterate() {
    const bool lb_now = (my_iter == first_lb) ||
        (my_iter > first_lb && lb_freq > 0 && (my_iter - first_lb) % lb_freq == 0);
    if (lb_now) {
      // Nothing of this block's may be in flight at the join: the step reads
      // the interval's loads here, and the block's own timings must cover the
      // same kernels.
      cudaStreamSynchronize(comm_stream);
      cudaStreamSynchronize(compute_stream);
      flushTimings(true);
      pe_at_sync = CkMyPe();
      AtSync();
    } else {
      thisProxy[thisIndex].exchangeGhosts();
    }
  }

  // The step that read this block's loads is over. Compare what it read with
  // what the block timed, report, and go on.
  void ResumeFromSync() {
    lb_steps_seen++;
    const double meas = getObjLastGPUTime();
    const double host = getObjLastTime();
    const double expected = win_dur_s * (double)my_sms / (double)total_sms;
    const bool moved = (CkMyPe() != pe_at_sync);
    const double ratio = (!moved && expected > 0.0) ? meas / expected : 0.0;
    if (verbose)
      CkPrintf("[smload] block (%d,%d) pe=%d step=%d sms=%d kernels=%d dur=%.5f s "
               "measured=%.5f expected=%.5f ratio=%.3f host=%.5f%s\n",
               x, y, CkMyPe(), lb_steps_seen, my_sms, win_kernels, win_dur_s,
               meas, expected, ratio, host, moved ? " MOVED" : "");
    double sums[7] = {
      moved ? 0.0 : meas, moved ? 0.0 : expected, moved ? 0.0 : win_dur_s,
      moved ? 0.0 : 1.0, ratio, moved ? 1.0 : 0.0, moved ? 0.0 : host };
    double mn = moved ? 1e300 : ratio;
    double mx = moved ? -1e300 : ratio;
    CkReduction::tupleElement t[3] = {
      CkReduction::tupleElement(sizeof(sums), sums, CkReduction::sum_double),
      CkReduction::tupleElement(sizeof(double), &mn, CkReduction::min_double),
      CkReduction::tupleElement(sizeof(double), &mx, CkReduction::max_double) };
    CkReductionMsg* msg = CkReductionMsg::buildFromTuple(t, 3);
    msg->setCallback(CkCallback(CkIndex_Main::lbReport(NULL), main_proxy));
    contribute(msg);

    win_dur_s = 0.0;
    win_kernels = 0;
    thisProxy[thisIndex].exchangeGhosts();
  }

  void update() {
    // The stencil runs behind the ghost unpacks of the comm stream.
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    if (ring_head - ring_tail == RING) flushTimings(true);
    hapiCheck(cudaEventRecord(t_begin[ring_head % RING], compute_stream));
    invokeJacobiSM(d_temperature, d_new_temperature, block_width, block_height,
        work_iters, my_sms * blocks_per_sm, compute_stream);
    hapiCheck(cudaEventRecord(t_end[ring_head % RING], compute_stream));
    ring_head++;
    flushTimings(false);

    // The pack and the ghost copies run behind the stencil.
    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));
  }

  void packGhosts() {
    invokePackingKernels(d_new_temperature, d_left_ghost, d_right_ghost,
        left_bound, right_bound, block_width, block_height, comm_stream);
    if (!left_bound)
      hapiCheck(hapiMemcpyAsync(h_left_ghost, d_left_ghost, block_height * sizeof(DataType),
            cudaMemcpyDeviceToHost, comm_stream));
    if (!right_bound)
      hapiCheck(hapiMemcpyAsync(h_right_ghost, d_right_ghost, block_height * sizeof(DataType),
            cudaMemcpyDeviceToHost, comm_stream));
    if (!top_bound)
      hapiCheck(hapiMemcpyAsync(h_top_ghost, d_new_temperature + (block_width + 2) + 1,
            block_width * sizeof(DataType), cudaMemcpyDeviceToHost, comm_stream));
    if (!bottom_bound)
      hapiCheck(hapiMemcpyAsync(h_bottom_ghost, d_new_temperature + (block_width + 2) * block_height + 1,
            block_width * sizeof(DataType), cudaMemcpyDeviceToHost, comm_stream));
    hapiAddCallback(comm_stream, CkCallback(CkIndex_Block::packGhostsDone(), thisProxy[thisIndex]));
  }

  void sendGhosts() {
    if (!left_bound)
      thisProxy(x - 1, y).receiveGhosts(my_iter, RIGHT, block_height, h_left_ghost);
    if (!right_bound)
      thisProxy(x + 1, y).receiveGhosts(my_iter, LEFT, block_height, h_right_ghost);
    if (!top_bound)
      thisProxy(x, y - 1).receiveGhosts(my_iter, BOTTOM, block_width, h_top_ghost);
    if (!bottom_bound)
      thisProxy(x, y + 1).receiveGhosts(my_iter, TOP, block_width, h_bottom_ghost);
  }

  void processGhosts(int dir, int size, DataType* gh) {
    switch (dir) {
      case LEFT:
        memcpy(h_left_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_left_ghost, h_left_ghost,
              block_height * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
        invokeUnpackingKernel(d_temperature, d_left_ghost, true, block_width,
            block_height, comm_stream);
        break;
      case RIGHT:
        memcpy(h_right_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_right_ghost, h_right_ghost,
              block_height * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
        invokeUnpackingKernel(d_temperature, d_right_ghost, false, block_width,
            block_height, comm_stream);
        break;
      case TOP:
        memcpy(h_top_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_temperature + 1, h_top_ghost,
              block_width * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
        break;
      case BOTTOM:
        memcpy(h_bottom_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_temperature + (block_width + 2) * (block_height + 1) + 1,
              h_bottom_ghost, block_width * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }
};

#include "smload.def.h"
