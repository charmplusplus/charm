#include "hapi.h"
#include "jacobi2d.decl.h"
#include "jacobi2d.h"
#include <utility>
#include <sstream>

/* NVTX ranges are a CUDA-only profiling facility: hapi_nvtx.h pulls in
 * nvtx3/nvToolsExt.h, which only the CUDA toolkit ships. Follow the runtime's
 * own convention (hapi_impl.cpp) and compile them in only under
 * -DHAPI_NVTX_PROFILE, so this example builds against either GPU backend.
 *
 * NVTX_RANGE also names the tracer. The ranges here used to be unnamed
 * temporaries -- NVTXTracer(os.str(), ...); -- which are destroyed at the end
 * of the full expression, so every range closed the instant it opened and
 * timed nothing. A named object lives to the end of the enclosing scope,
 * which is the region the range is meant to cover. */
#ifdef HAPI_NVTX_PROFILE
#include "hapi_nvtx.h"
#define NVTX_RANGE(label, color)                                    \
  std::ostringstream nvtx_os_;                                      \
  nvtx_os_ << label << " (" << std::to_string(x) << ","             \
           << std::to_string(y) << ")";                             \
  NVTXTracer nvtx_range_(nvtx_os_.str(), NVTXColor::color)
#else
#define NVTX_RANGE(label, color) ((void)0)
#endif

/* CkDeviceBufferPost names its stream field hapi_stream on the reconverse line
 * and cuda_stream on the classic one (ckrdmadevice.h splits at CMK_RECONVERSE),
 * so a single copy of this example has to spell it both ways until the classic
 * line is removed. Same shim as tests/charm++/cuda/d2dtest. */
#if CMK_RECONVERSE
#define POST_STREAM(p) ((p).hapi_stream)
#else
#define POST_STREAM(p) ((p).cuda_stream)
#endif

#define COMM_ONLY 0
#define GPU_SYNC 0

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

extern void invokeInitKernel(DataType* d_temperature, int block_width,
    int block_height, hapiStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, bool left_bound, bool right_bound, bool top_bound,
    bool bottom_bound, hapiStream_t stream);
extern void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int reps, hapiStream_t stream);
extern void invokePackingKernels(DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, bool left_bound, bool right_bound, int block_width,
    int block_height, hapiStream_t stream);
extern void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost,
    bool is_left, int block_width, int block_height, hapiStream_t stream);

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
    // A smaller default grid than plain jacobi2d: this variant exists to be
    // migrated, and 16384/4096 is only 16 chares to spread over the PEs.
    grid_width = 8192;
    grid_height = 8192;
    block_width = 2048;
    block_height = 2048;
    n_iters = 100;
    warmup_iters = 10;
    use_zerocopy = false;
    print_elements = false;
    sync_ver = false;
    // 0 disables load balancing; -f/-l turn it on.
    first_lb = 0;
    lb_freq = 0;
    // Extra stencil passes the most-loaded block does relative to the
    // least-loaded one. This is the whole point of the variant.
    imbalance = 5;
    my_iter = 0;

    // Initialize aggregate timers
    update_agg_time = 0.0;
    comm_agg_time = 0.0;

    // Process arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "W:H:w:h:i:u:f:l:m:yzp")) != -1) {
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
        case 'u':
          warmup_iters = atoi(optarg);
          break;
        case 'f':
          first_lb = atoi(optarg);
          break;
        case 'l':
          lb_freq = atoi(optarg);
          break;
        case 'm':
          imbalance = atoi(optarg);
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
              "-i [iterations] -u [warmup] -f [first LB iteration] -l [LB interval] "
              "-m [max extra stencil passes] "
              "-y (use sync version) -z (use GPU zerocopy) -p (print blocks)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (grid_width % block_width != 0 || grid_height % block_height != 0) {
      CkAbort("Invalid grid & block configuration\n");
    }

    // Number of chares per dimension
    n_chares_x = grid_width / block_width;
    n_chares_y = grid_height / block_height;

    // Print configuration
    CkPrintf("\n[GPU 2D Jacobi example, load imbalance variant]\n");
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
  DataType* __restrict__ d_recv_top_ghost;
  DataType* __restrict__ d_recv_bottom_ghost;

  hapiStream_t compute_stream;
  hapiStream_t comm_stream;

  hapiEvent_t compute_event;
  hapiEvent_t comm_event;

  bool left_bound, right_bound, top_bound, bottom_bound;

  Block() { usesAtSync = true; }

  // Migration constructor. The streams and events are recreated here; the
  // buffers are allocated from pup(), which is also where their contents
  // arrive.
  Block(CkMigrateMessage* m) : CBase_Block(m) {
    usesAtSync = true;
    createStreams();
  }

  void allocate() {
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
      hapiCheck(hapiMalloc((void**)&d_send_left_ghost, sizeof(DataType) * block_height));
      hapiCheck(hapiMalloc((void**)&d_send_right_ghost, sizeof(DataType) * block_height));
      hapiCheck(hapiMalloc((void**)&d_send_top_ghost, sizeof(DataType) * block_width));
      hapiCheck(hapiMalloc((void**)&d_send_bottom_ghost, sizeof(DataType) * block_width));
      hapiCheck(hapiMalloc((void**)&d_recv_left_ghost, sizeof(DataType) * block_height));
      hapiCheck(hapiMalloc((void**)&d_recv_right_ghost, sizeof(DataType) * block_height));
      hapiCheck(hapiMalloc((void**)&d_recv_top_ghost, sizeof(DataType) * block_width));
      hapiCheck(hapiMalloc((void**)&d_recv_bottom_ghost, sizeof(DataType) * block_width));
    }
  }

  void createStreams() {
    hapiCheck(hapiStreamCreateWithPriority(&compute_stream, hapiStreamDefault, 0));
    hapiCheck(hapiStreamCreateWithPriority(&comm_stream, hapiStreamDefault, -1));

    hapiCheck(hapiEventCreateWithFlags(&compute_event, hapiEventDisableTiming));
    hapiCheck(hapiEventCreateWithFlags(&comm_event, hapiEventDisableTiming));
  }

  /* Migration support.
   *
   * The device data is staged through the pinned host buffer by hand,
   * because PUP on this branch has no device mode: a chare that owns GPU
   * memory has to move it itself. Once GPU-aware load balancing lands this
   * whole staging dance collapses into a single
   *   p(d_temperature, n_elems, PUP::PUPMode::DEVICE);
   * and the host round trip goes away.
   *
   * Only d_temperature is carried across. d_new_temperature is scratch that
   * update() overwrites before anything reads it, and the ghost buffers are
   * refilled every iteration -- but iterate() only reaches AtSync() on an
   * iteration boundary with both streams quiesced, so nothing in flight is
   * being dropped here. */
  void pup(PUP::er& p) {
    // Chains to the generated pup, which carries the SDAG state. Skipping
    // it leaves a migrated chare with no continuation to resume into.
    CBase_Block::pup(p);

    p | my_iter;
    p | neighbors;
    p | remote_count;
    p | x;
    p | y;
    p | load_iters;
    p | left_bound;
    p | right_bound;
    p | top_bound;
    p | bottom_bound;

    const size_t n_elems = (size_t)(block_width + 2) * (block_height + 2);

    if (p.isUnpacking()) allocate();

    if (p.isPacking()) {
      hapiCheck(hapiMemcpy(h_temperature, d_temperature,
            sizeof(DataType) * n_elems, hapiMemcpyDeviceToHost));
    }
    PUParray(p, h_temperature, n_elems);
    if (p.isUnpacking()) {
      hapiCheck(hapiMemcpy(d_temperature, h_temperature,
            sizeof(DataType) * n_elems, hapiMemcpyHostToDevice));
    }
  }

  ~Block() {
    hapiCheck(hapiFreeHost(h_temperature));
    hapiCheck(hapiFree(d_temperature));
    hapiCheck(hapiFree(d_new_temperature));
    hapiCheck(hapiFreeHost(h_left_ghost));
    hapiCheck(hapiFreeHost(h_right_ghost));
    hapiCheck(hapiFreeHost(h_top_ghost));
    hapiCheck(hapiFreeHost(h_bottom_ghost));
    if (!use_zerocopy) {
      hapiCheck(hapiFree(d_left_ghost));
      hapiCheck(hapiFree(d_right_ghost));
    } else {
      hapiCheck(hapiFree(d_send_left_ghost));
      hapiCheck(hapiFree(d_send_right_ghost));
      hapiCheck(hapiFree(d_send_top_ghost));
      hapiCheck(hapiFree(d_send_bottom_ghost));
      hapiCheck(hapiFree(d_recv_left_ghost));
      hapiCheck(hapiFree(d_recv_right_ghost));
      hapiCheck(hapiFree(d_recv_top_ghost));
      hapiCheck(hapiFree(d_recv_bottom_ghost));
    }

    hapiCheck(hapiStreamDestroy(compute_stream));
    hapiCheck(hapiStreamDestroy(comm_stream));

    hapiCheck(hapiEventDestroy(compute_event));
    hapiCheck(hapiEventDestroy(comm_event));
  }

  void init() {
    // Initialize values
    my_iter = 0;
    neighbors = 0;
    x = thisIndex.x;
    y = thisIndex.y;

    // Graded load: blocks toward the bottom-right of the grid do up to
    // `imbalance` extra stencil passes, blocks at the top-left do none. A
    // block-mapped decomposition therefore leaves the last PEs overloaded,
    // which is what gives the load balancer something to move.
    load_iters = (int)(((float)(x + y) / (n_chares_x + n_chares_y)) * imbalance);

    NVTX_RANGE("Init", Turquoise);

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

    allocate();
    createStreams();

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_width, block_height, compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, compute_stream);

    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);
    invokeBoundaryKernels(d_new_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);

#if GPU_SYNC
    hapiStreamSynchronize(compute_stream);
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

  /* Iteration boundary in the asynchronous version, and the only place a
   * migration can happen. This is a separate entry method rather than an
   * inline check at the tail of exchangeGhosts so that the SDAG method has
   * fully returned before AtSync() can migrate the chare out from under it.
   *
   * Load balancing is off unless -l is given, so the default run is
   * unchanged: iterate() just re-enters exchangeGhosts. */
  void iterate() {
    if (lb_freq > 0 && my_iter >= first_lb && (my_iter - first_lb) % lb_freq == 0) {
      // pup() copies straight off the device, so drain both streams first.
      hapiCheck(hapiStreamSynchronize(comm_stream));
      hapiCheck(hapiStreamSynchronize(compute_stream));
      AtSync();
    } else {
      thisProxy[thisIndex].exchangeGhosts();
    }
  }

  void ResumeFromSync() {
    thisProxy[thisIndex].exchangeGhosts();
  }

  void update() {
    NVTX_RANGE("update", WetAsphalt);

    // Operations in compute stream should only be executed when
    // operations in communication stream (transfers and unpacking) complete
    hapiCheck(hapiEventRecord(comm_event, comm_stream));
    hapiCheck(hapiStreamWaitEvent(compute_stream, comm_event, 0));

#if !COMM_ONLY
    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height,
        1 + load_iters, compute_stream);
#endif

    // Operations in communication stream (packing and transfers) should
    // only be executed when operations in compute stream complete
    hapiCheck(hapiEventRecord(compute_event, compute_stream));
    hapiCheck(hapiStreamWaitEvent(comm_stream, compute_event, 0));

    // Copy final temperature data back to host
    if (print_elements && (my_iter == warmup_iters + n_iters)) {
      hapiCheck(hapiMemcpyAsync(h_temperature, d_new_temperature,
            sizeof(DataType) * (block_width + 2) * (block_height + 2),
            hapiMemcpyDeviceToHost, comm_stream));
    }

    if (sync_ver) {
#if GPU_SYNC
      hapiStreamSynchronize(compute_stream);
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

  void packGhosts() {
    NVTX_RANGE("packGhosts", Emerald);

    if (use_zerocopy) {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_send_left_ghost, d_send_right_ghost,
          left_bound, right_bound, block_width, block_height, comm_stream);
#endif

      // Copy top and bottom ghosts to send buffers
      if (!top_bound)
        hapiCheck(hapiMemcpyAsync(d_send_top_ghost, d_new_temperature + (block_width + 2) + 1,
              block_width * sizeof(DataType), hapiMemcpyDeviceToDevice, comm_stream));
      if (!bottom_bound)
        hapiCheck(hapiMemcpyAsync(d_send_bottom_ghost, d_new_temperature + (block_width + 2) * block_height + 1,
              block_width * sizeof(DataType), hapiMemcpyDeviceToDevice, comm_stream));
    } else {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_left_ghost, d_right_ghost,
          left_bound, right_bound, block_width, block_height, comm_stream);
#endif

      // Transfer ghosts from device to host
      if (!left_bound)
        hapiCheck(hapiMemcpyAsync(h_left_ghost, d_left_ghost, block_height * sizeof(DataType),
              hapiMemcpyDeviceToHost, comm_stream));
      if (!right_bound)
        hapiCheck(hapiMemcpyAsync(h_right_ghost, d_right_ghost, block_height * sizeof(DataType),
              hapiMemcpyDeviceToHost, comm_stream));
      if (!top_bound)
        hapiCheck(hapiMemcpyAsync(h_top_ghost, d_new_temperature + (block_width + 2) + 1,
              block_width * sizeof(DataType), hapiMemcpyDeviceToHost, comm_stream));
      if (!bottom_bound)
        hapiCheck(hapiMemcpyAsync(h_bottom_ghost, d_new_temperature + (block_width + 2) * block_height + 1,
              block_width * sizeof(DataType), hapiMemcpyDeviceToHost, comm_stream));
    }

#if GPU_SYNC
    hapiStreamSynchronize(comm_stream);
    thisProxy[thisIndex].packGhostsDone();
#else
    // Add asynchronous callback to be invoked when packing kernels and
    // ghost transfers are complete
    CkCallback* cb = new CkCallback(CkIndex_Block::packGhostsDone(), thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
#endif
  }

  void sendGhosts() {
    NVTX_RANGE("sendGhosts", PeterRiver);

    // Send ghosts to neighboring chares
    if (use_zerocopy) {
      if (!left_bound)
        thisProxy(x - 1, y).receiveGhostsZC(my_iter, RIGHT, block_height,
            CkDeviceBuffer(d_send_left_ghost,
              CkCallback(CkIndex_Block::d_send_left_ghost_done(), thisProxy[thisIndex]),
              comm_stream));
      if (!right_bound)
        thisProxy(x + 1, y).receiveGhostsZC(my_iter, LEFT, block_height,
            CkDeviceBuffer(d_send_right_ghost,
              CkCallback(CkIndex_Block::d_send_right_ghost_done(), thisProxy[thisIndex]),
              comm_stream));
      if (!top_bound)
        thisProxy(x, y - 1).receiveGhostsZC(my_iter, BOTTOM, block_width,
            CkDeviceBuffer(d_send_top_ghost,
              CkCallback(CkIndex_Block::d_send_top_ghost_done(), thisProxy[thisIndex]),
              comm_stream));
      if (!bottom_bound)
        thisProxy(x, y + 1).receiveGhostsZC(my_iter, TOP, block_width,
            CkDeviceBuffer(d_send_bottom_ghost,
              CkCallback(CkIndex_Block::d_send_bottom_ghost_done(), thisProxy[thisIndex]),
              comm_stream));
    } else {
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
      case LEFT:
        buf = d_recv_left_ghost;
        break;
      case RIGHT:
        buf = d_recv_right_ghost;
        break;
      case TOP:
        buf = d_recv_top_ghost;
        break;
      case BOTTOM:
        buf = d_recv_bottom_ghost;
        break;
      default:
        CkAbort("Error: invalid direction");
    }
    POST_STREAM(devicePost[0]) = comm_stream;
  }

  void processGhostsZC(int dir, int size, DataType* gh) {
    NVTX_RANGE("processGhostsZC", Amethyst);

    switch (dir) {
      case LEFT:
        invokeUnpackingKernel(d_temperature, d_recv_left_ghost, true, block_width,
            block_height, comm_stream);
        break;
      case RIGHT:
        invokeUnpackingKernel(d_temperature, d_recv_right_ghost, false, block_width,
            block_height, comm_stream);
        break;
      case TOP:
        // The top and bottom ghosts are contiguous rows, so they need no
        // unpacking kernel -- but they still land in their own buffer rather
        // than straight into the halo, so that all four directions present the
        // runtime with a whole allocation to write into. A device-to-device
        // copy on the communication stream then moves the row into place.
        hapiCheck(hapiMemcpyAsync(d_temperature + 1, d_recv_top_ghost,
              block_width * sizeof(DataType), hapiMemcpyDeviceToDevice, comm_stream));
        break;
      case BOTTOM:
        hapiCheck(hapiMemcpyAsync(d_temperature + (block_width + 2) * (block_height + 1) + 1,
              d_recv_bottom_ghost, block_width * sizeof(DataType),
              hapiMemcpyDeviceToDevice, comm_stream));
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void processGhostsReg(int dir, int size, DataType* gh) {
    NVTX_RANGE("processGhostsReg", Amethyst);

    switch (dir) {
      case LEFT:
        memcpy(h_left_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_left_ghost, h_left_ghost,
              block_height * sizeof(DataType), hapiMemcpyHostToDevice, comm_stream));
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_left_ghost, true, block_width,
            block_height, comm_stream);
#endif
        break;
      case RIGHT:
        memcpy(h_right_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_right_ghost, h_right_ghost,
              block_height * sizeof(DataType), hapiMemcpyHostToDevice, comm_stream));
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_right_ghost, false, block_width,
            block_height, comm_stream);
#endif
        break;
      case TOP:
        memcpy(h_top_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_temperature + 1, h_top_ghost,
              block_width * sizeof(DataType), hapiMemcpyHostToDevice, comm_stream));
        break;
      case BOTTOM:
        memcpy(h_bottom_ghost, gh, size * sizeof(DataType));
        hapiCheck(hapiMemcpyAsync(d_temperature + (block_width + 2) * (block_height + 1) + 1,
              h_bottom_ghost, block_width * sizeof(DataType), hapiMemcpyHostToDevice, comm_stream));
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
