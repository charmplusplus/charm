#include "hapi.h"
#include "hapi_nvtx.h"
#include "jacobi2d.decl.h"
#include "jacobi2d.h"
#include <utility>
#include <sstream>

#define COMM_ONLY 0
#define CUDA_SYNC 0

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
/* readonly */ bool sync_ver;
/* readonly */ bool use_zerocopy;
/* readonly */ bool print_elements;
/* readonly */ int lb_freq;
/* readonly */ int first_lb;
/* readonly */ int imbalance;
// MetaBalancer mode: instead of balancing on a fixed schedule, sample the load
// every sample_freq iterations and let MetaBalancer decide when to balance. The
// two rates are separate on purpose -- sampling is the expensive half, and a
// step must be able to start on any iteration, not just a sampling one.
/* readonly */ bool metalb_mode;
/* readonly */ int sample_freq;
// Async mode (-A, needs +LBAsync): AtSyncStart() returns without waiting, the
// block keeps iterating, and AtSyncWait() comes wait_lag iterations before the
// next AtSyncSample(). The
// gap between the two is the overlap.
/* readonly */ bool async_mode;
/* readonly */ int wait_lag;

extern void invokeInitKernel(DataType* d_temperature, int block_width,
    int block_height, cudaStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, bool left_bound, bool right_bound, bool top_bound,
    bool bottom_bound, cudaStream_t stream);
extern void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int iter, cudaStream_t stream);
extern void invokePackingKernels(DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, bool left_bound, bool right_bound, int block_width,
    int block_height, cudaStream_t stream);
extern void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost,
    bool is_left, int block_width, int block_height, cudaStream_t stream);

enum Direction { LEFT = 1, RIGHT, TOP, BOTTOM };

class Main : public CBase_Main {
  int my_iter;
  double init_start_time;
  double start_time;
  double comm_start_time;
  double comm_agg_time;
  double update_start_time;
  double update_agg_time;

public:
  Main(CkArgMsg* m) {
    // Set default values
    main_proxy = thisProxy;
    grid_width = 8192;
    grid_height = 8192;
    block_width = 2048;
    block_height = 2048;
    n_iters = 100;
    warmup_iters = 10;
    use_zerocopy = false;
    print_elements = false;
    sync_ver = false;
    my_iter = 0;
    first_lb = 10;
    lb_freq = 100;
    imbalance = 5;  // Max extra iterations for load imbalance
    metalb_mode = false;
    sample_freq = 5;
    async_mode = false;
    wait_lag = 3;

    // Initialize aggregate timers
    update_agg_time = 0.0;
    comm_agg_time = 0.0;

    // Process arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "W:H:w:h:i:b:f:m:u:s:l:yzpMA")) != -1) {
      switch (c) {
        case 'W':
          grid_width = atoi(optarg);
          break;
        case 'H':
          grid_height = atoi(optarg);
          break;
        case 'w':
          block_width = atoi(optarg);
          break;
        case 'h':
          block_height = atoi(optarg);
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 'b':
          lb_freq = atoi(optarg);
          break;
        case 'f':
          first_lb = atoi(optarg);
          break;
        case 'm':
          imbalance = atoi(optarg);
          break;
        case 'u':
          warmup_iters = atoi(optarg);
          break;
        case 's':
          sample_freq = atoi(optarg);
          break;
        case 'l':
          wait_lag = atoi(optarg);
          break;
        case 'M':
          metalb_mode = true;
          break;
        case 'A':
          metalb_mode = true;
          async_mode = true;
          break;
        case 'y':
          sync_ver = true;
          break;
        case 'z':
          use_zerocopy = true;
          break;
        case 'p':
          print_elements = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -W [grid width] -H [grid height] -w [block width] -h [block height]"
              "-b [lb frequency] -f [first lb] -m [max imbalance] "
              "-M (let MetaBalancer decide when to balance) -s [sample frequency, -M only] "
              "-A (async LB: implies -M, needs +LBAsync) "
              "-l [instrumentation window: iterations before each sample at which "
              "AtSyncWait is called, -A only] "
              "-i [iterations] -u [warmup] -y (use sync version) -z (use GPU zerocopy) -p (print blocks)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (grid_width % block_width != 0 || grid_height % block_height != 0) {
      CkAbort("Invalid grid & block configuration\n");
    }

    // Number of chares per dimension
    if (async_mode && (wait_lag <= 0 || wait_lag >= sample_freq)) {
      CkPrintf("Error: -l (%d) must be in 1..%d, i.e. inside one sampling "
               "period (-s). The instrumentation window runs from AtSyncWait to "
               "the next sample; a window as long as the period leaves no "
               "overlap, and a longer one inverts it.\n", wait_lag, sample_freq - 1);
      CkExit();
    }

    n_chares_x = grid_width / block_width;
    n_chares_y = grid_height / block_height;

    // Print configuration
    CkPrintf("\n[CUDA 2D Jacobi example]\n");
    CkPrintf("Grid: %d x %d, Block: %d x %d, Chares: %d x %d, Iterations: %d, "
        "Warm-up: %d, Bulk-synchronous: %d, Zerocopy: %d, Print: %d\n\n",
        grid_width, grid_height, block_width, block_height, n_chares_x, n_chares_y,
        n_iters, warmup_iters, sync_ver, use_zerocopy, print_elements);

    // Create blocks and start iteration
    block_proxy = CProxy_Block::ckNew(n_chares_x, n_chares_y);
    init_start_time = CkWallTimer();
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("Init time: %.3lf s\n", CkWallTimer() - init_start_time);

    startIter();
  }

  void startIter() {
    if (my_iter++ == warmup_iters) start_time = CkWallTimer();
    update_start_time = CkWallTimer();

    block_proxy.exchangeGhosts();
  }

  void updateDone() {
    if (my_iter > warmup_iters) update_agg_time += CkWallTimer() - update_start_time;
    comm_start_time = CkWallTimer();

    block_proxy.packGhosts();
  }

  void commDone() {
    if (my_iter > warmup_iters) comm_agg_time += CkWallTimer() - comm_start_time;

    if (my_iter == warmup_iters + n_iters) {
      allDone();
    } else {
      startIter();
    }
  }

  void allDone() {
    double total_time = CkWallTimer() - start_time;
    CkPrintf("Total time: %.3lf s\nAverage iteration time: %.3lf us\n",
        total_time, (total_time / n_iters) * 1e6);
    if (sync_ver) {
      CkPrintf("Comm time per iteration: %.3lf us\nUpdate time per iteration: %.3lf us\n",
          (comm_agg_time / n_iters) * 1e6, (update_agg_time / n_iters) * 1e6);
    }

    // Sum the whole grid. Jacobi is deterministic for a fixed iteration count,
    // so this value must match between a run that balances and one that does
    // not, and between sync and async load balancing -- a message dropped or
    // delivered twice across a migration shows up here.
    block_proxy.checksum();
  }

  void checksumDone(unsigned long long sum) {
    CkPrintf("Final checksum: %016llx\n", sum);
    if (print_elements) {
      sleep(1);
      block_proxy(0,0).print();
    } else {
      CkExit();
    }
  }

  void printDone() {
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
  int load_iters;

  // Async mode bookkeeping: the iteration at which this block's current LB step
  // began, and whether it still owes that step an AtSyncWait.
  int lb_start_iter = 0;
  bool lb_waiting = false;

  DataType* __restrict__ h_temperature;
  DataType* __restrict__ d_temperature;
  DataType* __restrict__ d_new_temperature;
  DataType* __restrict__ h_left_ghost;
  DataType* __restrict__ h_right_ghost;
  DataType* __restrict__ h_top_ghost;
  DataType* __restrict__ h_bottom_ghost;
  DataType* __restrict__ d_left_ghost;
  DataType* __restrict__ d_right_ghost;
  DataType* __restrict__ d_send_left_ghost;
  DataType* __restrict__ d_send_right_ghost;
  DataType* __restrict__ d_send_top_ghost;
  DataType* __restrict__ d_send_bottom_ghost;
  DataType* __restrict__ d_recv_left_ghost;
  DataType* __restrict__ d_recv_right_ghost;
  // Top and bottom ghosts land here rather than straight into d_temperature's
  // halo. Posting the halo directly is a race: the post hook runs whenever the
  // message arrives, which may be before this block has swapped d_temperature
  // with d_new_temperature for the iteration, so the data lands in whichever
  // array the pointer happened to name. Staging and copying in
  // processGhostsZC puts the write after the swap, where it is deterministic.
  DataType* __restrict__ d_recv_top_ghost;
  DataType* __restrict__ d_recv_bottom_ghost;

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;

  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  bool left_bound, right_bound, top_bound, bottom_bound;

  Block() {
    usesAtSync = true;
  }

  Block(CkMigrateMessage* m) {
    if (getenv("CHARM_DEBUG_MIGRATE") != NULL)
      CkPrintf("[APP] block ctor-migrated pe=%d obj=%p\n", CkMyPe(), (void*)this);
    usesAtSync = true;
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));

    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));
  }

  ~Block() {
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(cudaFree(d_temperature));
    hapiCheck(cudaFree(d_new_temperature));
    hapiCheck(cudaFreeHost(h_left_ghost));
    hapiCheck(cudaFreeHost(h_right_ghost));
    hapiCheck(cudaFreeHost(h_top_ghost));
    hapiCheck(cudaFreeHost(h_bottom_ghost));
    if (!use_zerocopy) {
      hapiCheck(cudaFree(d_left_ghost));
      hapiCheck(cudaFree(d_right_ghost));
    } else {
      // Deliberately not freed. A direct device-zerocopy send reads this
      // block's live allocation -- the runtime's own contract says such a send
      // must carry a completion callback the sender waits on before freeing.
      // These ghost sends carry none, so under +LBAsync a step can destroy this
      // block while a neighbour is still copying out of these buffers, and the
      // copy then faults on freed memory. They are four short vectors; leaking
      // them per migration is the price of not having that callback.
      //
      // The proper fix is a completion callback on the send, with the free
      // deferred until it fires.
      hapiCheck(cudaFree(d_recv_left_ghost));
      hapiCheck(cudaFree(d_recv_right_ghost));
      hapiCheck(cudaFree(d_recv_top_ghost));
      hapiCheck(cudaFree(d_recv_bottom_ghost));
    }

    hapiCheck(cudaStreamDestroy(compute_stream));
    hapiCheck(cudaStreamDestroy(comm_stream));

    hapiCheck(cudaEventDestroy(compute_event));
    hapiCheck(cudaEventDestroy(comm_event));
  }

  void pup(PUP::er& p) {
    // Migration copies d_temperature and d_new_temperature straight off the
    // device, on a stream of its own that is not ordered against ours. Under
    // async load balancing this block is still computing when that happens, so
    // settle our own work first, or the pack copies out a half-written grid.
    // The packing pass specifically: it is the one that issues those copies,
    // and the sizing pass also runs from AtSyncSample's size measurement, where
    // draining every sample would cost for nothing.
    if (p.isPacking()) {
      cudaStreamSynchronize(compute_stream);
      cudaStreamSynchronize(comm_stream);
    }

    p | my_iter;
    p | neighbors;
    p | remote_count;
    p | x;
    p | y;
    p | left_bound;
    p | right_bound;
    p | top_bound;
    p | bottom_bound;
    p | load_iters;
    p | lb_start_iter;
    p | lb_waiting;

    if (p.isUnpacking()) {
      hapiCheck(hapiMallocHost((void**)&h_temperature,
            sizeof(DataType) * (block_width + 2) * (block_height + 2)));
      hapiCheck(hapiMalloc((void**)&d_temperature,
            sizeof(DataType) * (block_width + 2) * (block_height + 2)));
      hapiCheck(hapiMalloc((void**)&d_new_temperature,
            sizeof(DataType) * (block_width + 2) * (block_height + 2)));
      hapiCheck(hapiMallocHost((void**)&h_left_ghost, sizeof(DataType) * block_height));
      hapiCheck(hapiMallocHost((void**)&h_right_ghost, sizeof(DataType) * block_height));
      hapiCheck(hapiMallocHost((void**)&h_top_ghost, sizeof(DataType) * block_width));
      hapiCheck(hapiMallocHost((void**)&h_bottom_ghost, sizeof(DataType) * block_width));
      if (!use_zerocopy) {
        hapiCheck(hapiMalloc((void**)&d_left_ghost, sizeof(DataType) * block_height));
        hapiCheck(hapiMalloc((void**)&d_right_ghost, sizeof(DataType) * block_height));
      } else {
        hapiCheck(hapiMalloc((void**)&d_send_left_ghost, sizeof(DataType) * block_height * 2));
        hapiCheck(hapiMalloc((void**)&d_send_right_ghost, sizeof(DataType) * block_height * 2));
        hapiCheck(hapiMalloc((void**)&d_send_top_ghost, sizeof(DataType) * block_width * 2));
        hapiCheck(hapiMalloc((void**)&d_send_bottom_ghost, sizeof(DataType) * block_width * 2));
        hapiCheck(hapiMalloc((void**)&d_recv_left_ghost, sizeof(DataType) * block_height * 2));
        hapiCheck(hapiMalloc((void**)&d_recv_right_ghost, sizeof(DataType) * block_height * 2));
        hapiCheck(hapiMalloc((void**)&d_recv_top_ghost, sizeof(DataType) * block_width * 2));
        hapiCheck(hapiMalloc((void**)&d_recv_bottom_ghost, sizeof(DataType) * block_width * 2));
      }
    }
      
    // The outgoing ghosts are live from packGhosts() until sendGhosts() reads
    // them, and under +LBAsync a step can move this block inside that window.
    // Carrying only the two grids left the destination's sendGhosts() reading
    // freshly allocated buffers, so its neighbours got garbage halos.
    PUParray(p, h_left_ghost, block_height);
    PUParray(p, h_right_ghost, block_height);
    PUParray(p, h_top_ghost, block_width);
    PUParray(p, h_bottom_ghost, block_width);

    // The landing buffers travel with the element. A ghost rget issued before a
    // step moves this block has already landed by the time this runs -- the
    // copy is issued on comm_stream and the packing pass above drains it -- but
    // the entry method that consumes it has not run yet, and will run on the
    // copy that resumes. So the bytes have to come along.
    if (use_zerocopy) {
      p(d_recv_left_ghost, block_height * 2, PUP::PUPMode::DEVICE);
      p(d_recv_right_ghost, block_height * 2, PUP::PUPMode::DEVICE);
      p(d_recv_top_ghost, block_width * 2, PUP::PUPMode::DEVICE);
      p(d_recv_bottom_ghost, block_width * 2, PUP::PUPMode::DEVICE);
    }

    p(d_temperature, (block_width + 2) * (block_height + 2), PUP::PUPMode::DEVICE);
    p(d_new_temperature, (block_width + 2) * (block_height + 2), PUP::PUPMode::DEVICE);
  }

  void init() {
    // Initialize values
    my_iter = 0;
    neighbors = 0;
    x = thisIndex.x;
    y = thisIndex.y;

    load_iters = (((float) (x + y)) / (n_chares_x + n_chares_y)) * imbalance;
    //CkPrintf("Block (%d,%d) load iters: %d\n", x, y, load_iters);

    std::ostringstream os;
    os << "Init (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Turquoise);

    // Check bounds and set number of valid neighbors
    left_bound = right_bound = top_bound = bottom_bound = false;
    if (thisIndex.x == 0)
      left_bound = true;
    else
      neighbors++;
    if (thisIndex.x == n_chares_x - 1)
      right_bound = true;
    else
      neighbors++;
    if (thisIndex.y == 0)
      top_bound = true;
    else
      neighbors++;
    if (thisIndex.y == n_chares_y - 1)
      bottom_bound = true;
    else
      neighbors++;

    // Allocate memory and create CUDA entities
    hapiCheck(hapiMallocHost((void**)&h_temperature,
          sizeof(DataType) * (block_width + 2) * (block_height + 2)));
    hapiCheck(hapiMalloc((void**)&d_temperature,
          sizeof(DataType) * (block_width + 2) * (block_height + 2)));
    hapiCheck(hapiMalloc((void**)&d_new_temperature,
          sizeof(DataType) * (block_width + 2) * (block_height + 2)));
    hapiCheck(hapiMallocHost((void**)&h_left_ghost, sizeof(DataType) * block_height));
    hapiCheck(hapiMallocHost((void**)&h_right_ghost, sizeof(DataType) * block_height));
    hapiCheck(hapiMallocHost((void**)&h_top_ghost, sizeof(DataType) * block_width));
    hapiCheck(hapiMallocHost((void**)&h_bottom_ghost, sizeof(DataType) * block_width));
    if (!use_zerocopy) {
      hapiCheck(hapiMalloc((void**)&d_left_ghost, sizeof(DataType) * block_height));
      hapiCheck(hapiMalloc((void**)&d_right_ghost, sizeof(DataType) * block_height));
    } else {
      hapiCheck(hapiMalloc((void**)&d_send_left_ghost, sizeof(DataType) * block_height * 2));
      hapiCheck(hapiMalloc((void**)&d_send_right_ghost, sizeof(DataType) * block_height * 2));
      hapiCheck(hapiMalloc((void**)&d_send_top_ghost, sizeof(DataType) * block_width * 2));
      hapiCheck(hapiMalloc((void**)&d_send_bottom_ghost, sizeof(DataType) * block_width * 2));
      hapiCheck(hapiMalloc((void**)&d_recv_left_ghost, sizeof(DataType) * block_height * 2));
      hapiCheck(hapiMalloc((void**)&d_recv_right_ghost, sizeof(DataType) * block_height * 2));
      hapiCheck(hapiMalloc((void**)&d_recv_top_ghost, sizeof(DataType) * block_width * 2));
      hapiCheck(hapiMalloc((void**)&d_recv_bottom_ghost, sizeof(DataType) * block_width * 2));
    }

    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));

    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_width, block_height, compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, compute_stream);

    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);
    invokeBoundaryKernels(d_new_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);

#if CUDA_SYNC
    cudaStreamSynchronize(compute_stream);
    thisProxy[thisIndex].initDone();
#else
    // TODO: Support reduction callback in hapiAddCallback
    CkCallback* cb = new CkCallback(CkIndex_Block::initDone(), thisProxy[thisIndex]);
    hapiAddCallback(compute_stream, cb);
#endif
  }

  void initDone() {
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  // Reports the iteration a block is moved at, which is how the overlap is
  // observed: under async LB the move should land inside the window between
  // "LB step starting" and "waiting for LB", not bunched at the wait.
  //
  // Chaining to ArrayElement is not optional. Its version notifies the array
  // listeners that this element is leaving, and an override that skips it lets
  // every migration complete and resume and then stalls the application.
  void ckAboutToMigrate() override {
    if (getenv("CHARM_DEBUG_MIGRATE") != NULL)
      CkPrintf("[APP] block (%d,%d) pe=%d obj=%p migrating at iteration %d\n",
               thisIndex.x, thisIndex.y, CkMyPe(), (void*)this, my_iter);
    ArrayElement::ckAboutToMigrate();
  }

  // Runs once at the end of the run, off the hot path. Uses its own host buffer
  // rather than h_temperature, which the unpacking path does not reallocate.
  //
  // A bitwise hash rather than a numeric sum: the field legitimately holds
  // non-finite values here, which a sum collapses to nan. Migration does not
  // change the arithmetic any cell sees, so a correct run is bit-identical
  // whatever the object-to-PE mapping was, and a hash says so crisply. Summed
  // across blocks so the reduction does not depend on their order.
  // Hash of this block's interior, for locating the first point at which an
  // async run diverges from a no-LB reference. Drains both streams, so only
  // used under CHARM_DEBUG_HASH.
  uint64_t gridHash() {
    const size_t n = (size_t)(block_width + 2) * (block_height + 2);
    std::vector<DataType> host(n);
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(comm_stream);
    hapiCheck(cudaMemcpy(host.data(), d_temperature, sizeof(DataType) * n,
                         cudaMemcpyDeviceToHost));
    uint64_t h = 1469598103934665603ULL;
    for (int j = 1; j <= block_height; j++) {
      const unsigned char* row =
          (const unsigned char*)(host.data() + (block_width + 2) * j + 1);
      for (size_t b = 0; b < sizeof(DataType) * (size_t)block_width; b++) {
        h ^= row[b];
        h *= 1099511628211ULL;
      }
    }
    return h;
  }

  void checksum() {
    const size_t n = (size_t)(block_width + 2) * (block_height + 2);
    std::vector<DataType> host(n);
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(comm_stream);
    hapiCheck(cudaMemcpy(host.data(), d_temperature, sizeof(DataType) * n,
                         cudaMemcpyDeviceToHost));

    uint64_t h = 1469598103934665603ULL;  // FNV-1a
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

  void iterate() {
    {
      static const bool hashDbg = (getenv("CHARM_DEBUG_HASH") != NULL);
      if (hashDbg)
        CkPrintf("[HASH] (%d,%d) iter=%d h=%016llx\n", x, y, my_iter,
                 (unsigned long long)gridHash());
    }
    if (getenv("CHARM_DEBUG_GHOSTV") != NULL)
      CkPrintf("[GH] (%d,%d) pe=%d ITER iter=%d\n", x, y, CkMyPe(), my_iter);
    if (metalb_mode) {
      // Sampling is throttled; starting a step is not. AtSyncStart runs every
      // iteration and costs an integer compare unless MetaBalancer has asked
      // for a step, so a step begins the iteration after the imbalance is seen
      // however rarely we sample.
      // Sampling is unconditional: MetaBalancer's sample stream expects every
      // element to keep contributing at the same cadence, including while a
      // step is in flight, or the PE's bucket for that sample never completes.
      const bool sampling = (my_iter != 0 && my_iter % sample_freq == 0);
      // Do not start another step while this block still owes one a wait --
      // it would lose track of the first. A step requested in that window is
      // picked up on a later iteration, once the wait is done.
      const bool starting = !lb_waiting && AtSyncPending();
      // The wait is placed relative to the *sampling* cadence, not to the start:
      // the decision is made at the sample, and instrumentation runs for the
      // wait_lag iterations leading up to it. So wait at the first iteration of
      // that window, which leaves the overlap running from the join all the way
      // to here -- most of a sampling period.
      //
      // Deliberately not conditioned on having started a step: a block that
      // blocked at the tentative count is joined by the runtime rather than by
      // its own AtSyncStart(), so it cannot know it started one. AtSyncWait()
      // resumes inline when there is nothing to wait for, so calling it on the
      // cadence is always safe.
      const bool finishing = (my_iter % sample_freq) >= (sample_freq - wait_lag);
      if (sampling || starting || finishing) {
        cudaStreamSynchronize(comm_stream);
        cudaStreamSynchronize(compute_stream);
      }
      if (sampling) AtSyncSample();

      if (!async_mode) {
        // Called every iteration -- that cadence is what makes the runtime's
        // step counter a clock every block agrees on. It joins only at the
        // agreed count and resumes inline otherwise, so there is nothing left
        // to do here either way.
        AtSyncStart();
        return;
      }

      // Same cadence in async, except while this block still owes a wait: the
      // contract is one step at a time per element, and every block starts and
      // waits at the same iterations, so they all skip the same calls and their
      // counters stay level.
      // The runtime knows whether this block still owes a wait; the application
      // cannot, for the reason above.
      if (!AtSyncStepInFlight()) {
        const CkMigratable::AtSyncStatus st = AtSyncStart();
        // Stopped at the tentative count while the step's iteration is agreed.
        // Do nothing at all: the runtime resumes this block, or joins it, once
        // the count is settled. Iterating on would be running past the count
        // that is about to be agreed, which is what the stop exists to prevent.
        if (st == CkMigratable::AtSyncStatus::Blocked) return;
        if (st == CkMigratable::AtSyncStatus::Started) {
          lb_start_iter = my_iter;
          if (thisIndex.x == 0 && thisIndex.y == 0)
            CkPrintf("[APP] LB step starting at iteration %d\n", my_iter);
        }
      }

      if (finishing) {
        if (thisIndex.x == 0 && thisIndex.y == 0)
          CkPrintf("[APP] waiting for LB at iteration %d (%d iterations overlapped)\n",
                   my_iter, my_iter - lb_start_iter);
        // The element has to be quiesced before it can be pupped, and
        // AtSyncWait is where it becomes migratable.
        cudaStreamSynchronize(comm_stream);
        cudaStreamSynchronize(compute_stream);
        AtSyncWait();
        return;
      }
      // Nothing to wait for: keep iterating while the strategy runs and other
      // blocks migrate. This is the overlap the split buys.
      thisProxy[thisIndex].exchangeGhosts();
    } else if (my_iter == first_lb || (my_iter != 0 && my_iter % lb_freq == 0)) {
      cudaStreamSynchronize(comm_stream);
      cudaStreamSynchronize(compute_stream);
      AtSync();
    } else {
      thisProxy[thisIndex].exchangeGhosts();
    }
  }

  void ResumeFromSync() {
    thisProxy[thisIndex].exchangeGhosts();
  }

  void update() {
    if (getenv("CHARM_DEBUG_GHOSTV") != NULL)
      CkPrintf("[GH] (%d,%d) pe=%d UPD iter=%d\n", x, y, CkMyPe(), my_iter);
    std::ostringstream os;
    os << "update (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::WetAsphalt);

    // Operations in compute stream should only be executed when
    // operations in communication stream (transfers and unpacking) complete
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

#if !COMM_ONLY
    // Invoke GPU kernel for Jacobi computation.
    //
    // Every chare launches this with identical grid and block dimensions and
    // differs only in load_iters, which CUPTI cannot see. Without the tag the
    // automatic launch-signature bucket merges all of them into one identity,
    // and the estimator then averages incomparable amounts of work. The tag
    // gives each loop count its own bucket.
    //
    // Set JACOBI_NO_WORK_TAG=1 to launch untagged instead. This benchmark is
    // the case the tag exists for, so being able to A/B it is what makes the
    // difference measurable rather than asserted.
    static const bool tag_work =
        (getenv("JACOBI_NO_WORK_TAG") == nullptr);
    if (tag_work) {
      hapiCuptiKernelTagScope work_tag(load_iters);
      invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height, load_iters,
          compute_stream);
    } else {
      invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height, load_iters,
          compute_stream);
    }
#endif

    // Operations in communication stream (packing and transfers) should
    // only be executed when operations in compute stream complete
    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));

    // Copy final temperature data back to host
    if (print_elements && (my_iter == warmup_iters + n_iters)) {
      hapiCheck(hapiMemcpyAsync(h_temperature, d_new_temperature,
            sizeof(DataType) * (block_width + 2) * (block_height + 2),
            cudaMemcpyDeviceToHost, comm_stream));
    }

    if (sync_ver) {
#if CUDA_SYNC
      cudaStreamSynchronize(compute_stream);
      thisProxy[thisIndex].updateDone();
#else
      CkCallback* cb = new CkCallback(CkIndex_Block::updateDone(), thisProxy[thisIndex]);
      hapiAddCallback(compute_stream, cb);
#endif
    }
  }

  void updateDone() {
    contribute(CkCallback(CkReductionTarget(Main, updateDone), main_proxy));
  }

  // The receiver pulls straight out of these buffers, asynchronously, and
  // nothing tells this block when that pull has finished. Reusing one buffer
  // every iteration therefore lets a block overwrite ghosts a neighbour has not
  // read yet -- which is why -z gave a different answer every run even with no
  // load balancing at all. Alternating halves gives the neighbour a full
  // iteration to pull: this block cannot reach iteration N+2 until it has the
  // ghosts its neighbour sent at N+1, which it could only have sent after
  // reading this block's N.
  DataType* sendSlice(DataType* base, int n) const { return base + (my_iter & 1) * n; }

  void packGhosts() {
    if (getenv("CHARM_DEBUG_GHOSTV") != NULL)
      CkPrintf("[GH] (%d,%d) pe=%d PACK iter=%d\n", x, y, CkMyPe(), my_iter);
    std::ostringstream os;
    os << "packGhosts (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Emerald);

    if (use_zerocopy) {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, sendSlice(d_send_left_ghost, block_height),
          sendSlice(d_send_right_ghost, block_height),
          left_bound, right_bound, block_width, block_height, comm_stream);
#endif

      // Copy top and bottom ghosts to send buffers
      if (!top_bound)
        hapiCheck(hapiMemcpyAsync(sendSlice(d_send_top_ghost, block_width), d_new_temperature + (block_width + 2) + 1,
              block_width * sizeof(DataType), cudaMemcpyDeviceToDevice, comm_stream));
      if (!bottom_bound)
        hapiCheck(hapiMemcpyAsync(sendSlice(d_send_bottom_ghost, block_width), d_new_temperature + (block_width + 2) * block_height + 1,
              block_width * sizeof(DataType), cudaMemcpyDeviceToDevice, comm_stream));
    } else {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_left_ghost, d_right_ghost,
          left_bound, right_bound, block_width, block_height, comm_stream);
#endif

      // Transfer ghosts from device to host
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
    }

#if CUDA_SYNC
    cudaStreamSynchronize(comm_stream);
    thisProxy[thisIndex].packGhostsDone();
#else
    // Add asynchronous callback to be invoked when packing kernels and
    // ghost transfers are complete
    CkCallback* cb = new CkCallback(CkIndex_Block::packGhostsDone(), thisProxy[thisIndex]);
    if (getenv("CHARM_DEBUG_GHOST") != NULL)
      CkPrintf("[GH] (%d,%d) pe=%d PACKARM iter=%d\n", x, y, CkMyPe(), my_iter);
    hapiAddCallback(comm_stream, cb);
#endif
  }

  void sendGhosts() {
    std::ostringstream os;
    os << "sendGhosts (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::PeterRiver);

    // Send ghosts to neighboring chares
    if (use_zerocopy) {
      if (!left_bound)
        thisProxy(x - 1, y).receiveGhostsZC(my_iter, RIGHT, block_height,
            CkDeviceBuffer(sendSlice(d_send_left_ghost, block_height), comm_stream));
      if (!right_bound)
        thisProxy(x + 1, y).receiveGhostsZC(my_iter, LEFT, block_height,
            CkDeviceBuffer(sendSlice(d_send_right_ghost, block_height), comm_stream));
      if (!top_bound)
        thisProxy(x, y - 1).receiveGhostsZC(my_iter, BOTTOM, block_width,
            CkDeviceBuffer(sendSlice(d_send_top_ghost, block_width), comm_stream));
      if (!bottom_bound)
        thisProxy(x, y + 1).receiveGhostsZC(my_iter, TOP, block_width,
            CkDeviceBuffer(sendSlice(d_send_bottom_ghost, block_width), comm_stream));
    } else {
      if (getenv("CHARM_DEBUG_GHOST") != NULL) {
        CkPrintf("[GH] (%d,%d) pe=%d obj=%p SEND iter=%d\n", x, y, CkMyPe(), (void*)this, my_iter);
        if (!left_bound)   CkPrintf("[TX] %d,%d -> %d,%d iter=%d\n", x, y, x-1, y, my_iter);
        if (!right_bound)  CkPrintf("[TX] %d,%d -> %d,%d iter=%d\n", x, y, x+1, y, my_iter);
        if (!top_bound)    CkPrintf("[TX] %d,%d -> %d,%d iter=%d\n", x, y, x, y-1, my_iter);
        if (!bottom_bound) CkPrintf("[TX] %d,%d -> %d,%d iter=%d\n", x, y, x, y+1, my_iter);
      }
      if (!left_bound)
        thisProxy(x - 1, y).receiveGhostsReg(my_iter, RIGHT, block_height, h_left_ghost);
      if (!right_bound)
        thisProxy(x + 1, y).receiveGhostsReg(my_iter, LEFT, block_height, h_right_ghost);
      if (!top_bound)
        thisProxy(x, y - 1).receiveGhostsReg(my_iter, BOTTOM, block_width, h_top_ghost);
      if (!bottom_bound)
        thisProxy(x, y + 1).receiveGhostsReg(my_iter, TOP, block_width, h_bottom_ghost);
    }
  }

  // This is the post entry method, the regular entry method is defined as a
  // SDAG entry method in the .ci file
  void receiveGhostsZC(int ref, int dir, int &size, DataType *&buf, CkDeviceBufferPost *devicePost) {
    switch (dir) {
      // Alternate halves by the sender's iteration tag. A landing buffer is
      // still being read by the unpack kernel queued on comm_stream for the
      // previous iteration, and the runtime's write is not ordered against
      // that stream, so reusing one buffer every iteration lets an arrival
      // overwrite ghosts that have not been consumed yet.
      case LEFT:
        buf = d_recv_left_ghost + (ref & 1) * block_height;
        break;
      case RIGHT:
        buf = d_recv_right_ghost + (ref & 1) * block_height;
        break;
      case TOP:
        buf = d_recv_top_ghost + (ref & 1) * block_width;
        break;
      case BOTTOM:
        buf = d_recv_bottom_ghost + (ref & 1) * block_width;
        break;
      default:
        CkAbort("Error: invalid direction");
    }
    devicePost[0].hapi_stream = comm_stream;
  }

  void processGhostsZC(int dir, int size, DataType* gh) {
    std::ostringstream os;
    os << "processGhostsZC (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Amethyst);

    switch (dir) {
      // gh is the pointer the runtime actually landed into -- the correct half
      // of the double buffer. Reading the buffer base instead would defeat the
      // alternation done in the post hook.
      // This copy's own buffer, not gh. gh is the address owned by the copy that
      // posted the rget; if a step moved this block since, that address is on
      // another device. The contents came across in pup, and the half is the
      // sender's iteration tag, which the SDAG matched against my_iter.
      case LEFT:
        invokeUnpackingKernel(d_temperature,
            d_recv_left_ghost + (my_iter & 1) * block_height, true, block_width,
            block_height, comm_stream);
        break;
      case RIGHT:
        invokeUnpackingKernel(d_temperature,
            d_recv_right_ghost + (my_iter & 1) * block_height, false, block_width,
            block_height, comm_stream);
        break;
      case TOP:
        hapiCheck(cudaMemcpyAsync(d_temperature + 1,
            d_recv_top_ghost + (my_iter & 1) * block_width,
            block_width * sizeof(DataType), cudaMemcpyDeviceToDevice, comm_stream));
        break;
      case BOTTOM:
        hapiCheck(cudaMemcpyAsync(d_temperature + (block_width + 2) * (block_height + 1) + 1,
            d_recv_bottom_ghost + (my_iter & 1) * block_width,
            block_width * sizeof(DataType), cudaMemcpyDeviceToDevice, comm_stream));
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void processGhostsReg(int dir, int size, DataType* gh) {
    if (getenv("CHARM_DEBUG_GHOSTV") != NULL)
      CkPrintf("[GH] (%d,%d) pe=%d RECV iter=%d dir=%d count=%d/%d\n", x, y,
               CkMyPe(), my_iter, dir, remote_count + 1, neighbors);
    else if (getenv("CHARM_DEBUG_GHOST") != NULL) {
      CkPrintf("[RX] %d,%d iter=%d dir=%d\n", x, y, my_iter, dir);
    }
    if (getenv("CHARM_DEBUG_GHOST") != NULL && remote_count + 1 == neighbors)
      CkPrintf("[GH] (%d,%d) pe=%d obj=%p RECVDONE iter=%d\n", x, y, CkMyPe(), (void*)this, my_iter);
    std::ostringstream os;
    os << "processGhostsReg (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Amethyst);

    switch (dir) {
      case LEFT:
        memcpy(h_left_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_left_ghost, h_left_ghost,
              block_height * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_left_ghost, true, block_width,
            block_height, comm_stream);
#endif
        break;
      case RIGHT:
        memcpy(h_right_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_right_ghost, h_right_ghost,
              block_height * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_right_ghost, false, block_width,
            block_height, comm_stream);
#endif
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

  void print() {
    CkPrintf("[%d,%d]\n", thisIndex.x, thisIndex.y);
    for (int j = 0; j < block_height + 2; j++) {
      for (int i = 0; i < block_width + 2; i++) {
#ifdef TEST_CORRECTNESS
        CkPrintf("%d ", h_temperature[(block_width + 2) * j + i]);
#else
        CkPrintf("%.6lf ", h_temperature[(block_width + 2) * j + i]);
#endif
      }
      CkPrintf("\n");
    }

    if (!(thisIndex.x == n_chares_x-1 && thisIndex.y == n_chares_y-1)) {
      if (thisIndex.x == n_chares_x-1) {
        thisProxy(0,thisIndex.y+1).print();
      } else {
        thisProxy(thisIndex.x+1,thisIndex.y).print();
      }
    } else {
      main_proxy.printDone();
    }
  }
};

#include "jacobi2d.def.h"
