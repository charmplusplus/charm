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
/* readonly */ int grid_size;
/* readonly */ int block_size;
/* readonly */ int n_chares;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ bool sync_ver;
/* readonly */ bool use_zerocopy;
/* readonly */ bool print;

extern void invokeInitKernel(DataType* d_temperature, int block_size, cudaStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_size, bool left_bound,
    bool right_bound, bool top_bound, bool bottom_bound, cudaStream_t stream);
extern void invokePackingKernels(DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, bool left_bound, bool right_bound, int block_size,
    cudaStream_t stream);
extern void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost,
    bool is_left, int block_size, cudaStream_t stream);
extern void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_size, cudaStream_t stream);

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
    grid_size = 16384;
    block_size = 4096;
    n_iters = 100;
    warmup_iters = 10;
    use_zerocopy = false;
    print = false;
    sync_ver = false;
    my_iter = 0;

    // Initialize aggregate timers
    update_agg_time = 0.0;
    comm_agg_time = 0.0;

    // Process arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:b:i:w:yzp")) != -1) {
      switch (c) {
        case 's':
          grid_size = atoi(optarg);
          break;
        case 'b':
          block_size = atoi(optarg);
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 'w':
          warmup_iters = atoi(optarg);
          break;
        case 'y':
          sync_ver = true;
          break;
        case 'z':
          use_zerocopy = true;
          break;
        case 'p':
          print = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -s [grid size] -b [block size] -i [iterations] -w [warmup]"
              "-y (use sync version) -z (use GPU zerocopy) -p (print blocks)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (grid_size < block_size || grid_size % block_size != 0) {
      CkAbort("Invalid grid & block configuration\n");
    }

    // Number of chares per dimension
    n_chares = grid_size / block_size;

    // Print configuration
    CkPrintf("\n[CUDA 2D Jacobi example]\n");
    CkPrintf("Grid: %d x %d, Block: %d x %d, Chares: %d x %d, Iterations: %d, "
        "Warm-up: %d, Bulk-synchronous: %d, Zerocopy: %d, Print: %d\n\n",
        grid_size, grid_size, block_size, block_size, n_chares, n_chares,
        n_iters, warmup_iters, sync_ver, use_zerocopy, print);

    // Create blocks and start iteration
    block_proxy = CProxy_Block::ckNew(n_chares, n_chares);
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

    if (print) {
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

  DataType* __restrict__ h_temperature;
  DataType* __restrict__ d_temperature;
  DataType* __restrict__ d_new_temperature;
  DataType* __restrict__ h_left_ghost;
  DataType* __restrict__ h_right_ghost;
  DataType* __restrict__ h_bottom_ghost;
  DataType* __restrict__ h_top_ghost;
  DataType* __restrict__ d_left_ghost;
  DataType* __restrict__ d_right_ghost;

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;

  cudaEvent_t compute_event;

  bool left_bound, right_bound, top_bound, bottom_bound;

  Block() {}

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
  }

  void init() {
    // Initialize values
    my_iter = 0;
    neighbors = 0;
    x = thisIndex.x;
    y = thisIndex.y;

    std::ostringstream os;
    os << "Init (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Turquoise);

    // Check bounds and set number of valid neighbors
    left_bound = right_bound = top_bound = bottom_bound = false;
    if (thisIndex.x == 0)
      left_bound = true;
    else
      neighbors++;
    if (thisIndex.x == n_chares - 1)
      right_bound = true;
    else
      neighbors++;
    if (thisIndex.y == 0)
      top_bound = true;
    else
      neighbors++;
    if (thisIndex.y == n_chares - 1)
      bottom_bound = true;
    else
      neighbors++;

    // Allocate memory and create CUDA entities
    hapiCheck(cudaMallocHost((void**)&h_temperature,
          sizeof(DataType) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMalloc((void**)&d_temperature,
          sizeof(DataType) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMalloc((void**)&d_new_temperature,
          sizeof(DataType) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMallocHost((void**)&h_left_ghost, sizeof(DataType) * block_size));
    hapiCheck(cudaMallocHost((void**)&h_right_ghost, sizeof(DataType) * block_size));
    hapiCheck(cudaMallocHost((void**)&h_bottom_ghost, sizeof(DataType) * block_size));
    hapiCheck(cudaMallocHost((void**)&h_top_ghost, sizeof(DataType) * block_size));
    hapiCheck(cudaMalloc((void**)&d_left_ghost, sizeof(DataType) * block_size));
    hapiCheck(cudaMalloc((void**)&d_right_ghost, sizeof(DataType) * block_size));
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));
    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_size, compute_stream);

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

  void update() {
    std::ostringstream os;
    os << "update (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::WetAsphalt);

#if !COMM_ONLY
    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_size, left_bound, right_bound,
        top_bound, bottom_bound, compute_stream);

    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_size, compute_stream);
#endif

    // Record event to create dependency for communication stream
    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));

    // Copy final temperature data back to host
    if (my_iter == warmup_iters + n_iters) {
      hapiCheck(cudaMemcpyAsync(h_temperature, d_new_temperature,
            sizeof(DataType) * (block_size + 2) * (block_size + 2),
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

  void packGhosts() {
    std::ostringstream os;
    os << "packGhosts (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Emerald);

#if !COMM_ONLY
    // Pack non-contiguous ghosts to temporary contiguous buffers on device
    invokePackingKernels(d_temperature, d_left_ghost, d_right_ghost, left_bound,
        right_bound, block_size, comm_stream);
#endif

    if (!use_zerocopy) {
      // Transfer ghosts from device to host
      if (!left_bound)
        hapiCheck(cudaMemcpyAsync(h_left_ghost, d_left_ghost, block_size * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
      if (!right_bound)
        hapiCheck(cudaMemcpyAsync(h_right_ghost, d_right_ghost, block_size * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
      if (!top_bound)
        hapiCheck(cudaMemcpyAsync(h_top_ghost, d_temperature + (block_size + 2) + 1,
              block_size * sizeof(DataType), cudaMemcpyDeviceToHost, comm_stream));
      if (!bottom_bound)
        hapiCheck(cudaMemcpyAsync(h_bottom_ghost, d_temperature + (block_size + 2) * block_size + 1,
              block_size * sizeof(DataType), cudaMemcpyDeviceToHost, comm_stream));

#if CUDA_SYNC
      cudaStreamSynchronize(comm_stream);
      thisProxy[thisIndex].packGhostsDone();
#else
      // Add asynchronous callback to be invoked when packing and device-to-host
      // transfers are complete
      CkCallback* cb = new CkCallback(CkIndex_Block::packGhostsDone(), thisProxy[thisIndex]);
      hapiAddCallback(comm_stream, cb);
#endif
    }
  }

  void sendGhosts() {
    std::ostringstream os;
    os << "sendGhosts (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::PeterRiver);

    // Send ghosts to neighboring chares
    if (use_zerocopy) {
      CkCallback cb(CkIndex_Block::sendGhostDone(), thisProxy[thisIndex]);

      if (!left_bound)
        thisProxy(x - 1, y).receiveGhostsZC(my_iter, RIGHT, block_size,
            CkDeviceBuffer(d_left_ghost, cb, comm_stream));
      if (!right_bound)
        thisProxy(x + 1, y).receiveGhostsZC(my_iter, LEFT, block_size,
            CkDeviceBuffer(d_right_ghost, cb, comm_stream));
      if (!top_bound)
        thisProxy(x, y - 1).receiveGhostsZC(my_iter, BOTTOM, block_size,
            CkDeviceBuffer(d_temperature + (block_size + 2) + 1, cb, comm_stream));
      if (!bottom_bound)
        thisProxy(x, y + 1).receiveGhostsZC(my_iter, TOP, block_size,
            CkDeviceBuffer(d_temperature + (block_size + 2) * block_size + 1, cb, comm_stream));
    } else {
      if (!left_bound)
        thisProxy(x - 1, y).receiveGhostsReg(my_iter, RIGHT, block_size, h_left_ghost);
      if (!right_bound)
        thisProxy(x + 1, y).receiveGhostsReg(my_iter, LEFT, block_size, h_right_ghost);
      if (!top_bound)
        thisProxy(x, y - 1).receiveGhostsReg(my_iter, BOTTOM, block_size, h_top_ghost);
      if (!bottom_bound)
        thisProxy(x, y + 1).receiveGhostsReg(my_iter, TOP, block_size, h_bottom_ghost);
    }
  }

  // This is the post entry method, the regular entry method is defined as a
  // SDAG entry method in the .ci file
  void receiveGhostsZC(int ref, int dir, int &w, DataType *&buf, CkDeviceBufferPost *devicePost) {
    switch (dir) {
      case LEFT:
        buf = d_left_ghost;
        break;
      case RIGHT:
        buf = d_right_ghost;
        break;
      case TOP:
        buf = d_temperature + (block_size + 2) + 1;
        break;
      case BOTTOM:
        buf = d_temperature + (block_size + 2) * block_size + 1;
        break;
      default:
        CkAbort("Error: invalid direction");
    }
    devicePost[0].cuda_stream = comm_stream;
  }

  void processGhostsZC(int dir, int width, DataType* gh) {
    std::ostringstream os;
    os << "processGhostsZC (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Amethyst);

    switch (dir) {
      case LEFT:
        invokeUnpackingKernel(d_temperature, d_left_ghost, true, block_size, comm_stream);
        break;
      case RIGHT:
        invokeUnpackingKernel(d_temperature, d_right_ghost, false, block_size, comm_stream);
        break;
      case TOP:
      case BOTTOM:
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void processGhostsReg(int dir, int width, DataType* gh) {
    std::ostringstream os;
    os << "processGhostsReg (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Amethyst);

    switch (dir) {
      case LEFT:
        memcpy(h_left_ghost, gh, width * sizeof(DataType));
        hapiCheck(cudaMemcpyAsync(d_left_ghost, h_left_ghost, block_size * sizeof(DataType),
              cudaMemcpyHostToDevice, comm_stream));
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_left_ghost, true, block_size, comm_stream);
#endif
        break;
      case RIGHT:
        memcpy(h_right_ghost, gh, width * sizeof(DataType));
        hapiCheck(cudaMemcpyAsync(d_right_ghost, h_right_ghost, block_size * sizeof(DataType),
              cudaMemcpyHostToDevice, comm_stream));
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_right_ghost, false, block_size, comm_stream);
#endif
        break;
      case TOP:
        memcpy(h_top_ghost, gh, width * sizeof(DataType));
        hapiCheck(cudaMemcpyAsync(d_temperature + (block_size + 2) + 1, h_top_ghost,
              block_size * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
        break;
      case BOTTOM:
        memcpy(h_bottom_ghost, gh, width * sizeof(DataType));
        hapiCheck(cudaMemcpyAsync(d_temperature + (block_size + 2) * block_size + 1,
              h_bottom_ghost, block_size * sizeof(DataType), cudaMemcpyHostToDevice, comm_stream));
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void print() {
    CkPrintf("[%d,%d]\n", thisIndex.x, thisIndex.y);
    for (int j = 0; j < block_size + 2; j++) {
      for (int i = 0; i < block_size + 2; i++) {
#ifdef TEST_CORRECTNESS
        CkPrintf("%d ", h_temperature[(block_size + 2) * j + i]);
#else
        CkPrintf("%.6lf ", h_temperature[(block_size + 2) * j + i]);
#endif
      }
      CkPrintf("\n");
    }

    if (!(thisIndex.x == n_chares-1 && thisIndex.y == n_chares-1)) {
      if (thisIndex.x == n_chares-1) {
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
