#include "hapi.h"
#include "jacobi2d.decl.h"
#include <utility>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int grid_size;
/* readonly */ int block_size;
/* readonly */ int n_chares;
/* readonly */ int n_iters;
/* readonly */ bool sync_ver;
/* readonly */ bool print;

extern void invokeInitKernel(double* d_temperature, int block_size, cudaStream_t stream);
extern void invokeBoundaryKernels(double* d_temperature, int block_size, bool left_bound,
    bool right_bound, bool top_bound, bool bottom_bound, cudaStream_t stream);
extern void invokePackingKernels(double* d_temperature, double* d_left_ghost,
    double* d_right_ghost, bool left_bound, bool right_bound, int block_size,
    cudaStream_t stream);
extern void invokeUnpackingKernel(double* d_temperature, double* d_ghost,
    bool is_left, int block_size, cudaStream_t stream);
extern void invokeJacobiKernel(double* d_temperature, double* d_new_temperature,
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
    print = false;
    sync_ver = false;

    // Process arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:b:i:yp")) != -1) {
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
        case 'y':
          sync_ver = true;
          break;
        case 'p':
          print = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -s [grid size] -b [block size] -i [iterations] "
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
        "Bulk-synchronous: %d, Print: %d\n", grid_size, grid_size, block_size,
        block_size, n_chares, n_chares, n_iters, sync_ver, print);

    // Create blocks and start iteration
    block_proxy = CProxy_Block::ckNew(n_chares, n_chares);
    init_start_time = CkWallTimer();
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("Init time: %.3lf us\n", CkWallTimer() - init_start_time);
    my_iter = 1;
    start_time = CkWallTimer();
    if (sync_ver) comm_start_time = CkWallTimer();
    block_proxy.exchangeGhosts();
  }

  void commDone() {
    comm_agg_time += CkWallTimer() - comm_start_time;
    update_start_time = CkWallTimer();
    block_proxy.update();
  }

  void updateDone() {
    update_agg_time += CkWallTimer() - update_start_time;

    if (my_iter++ == n_iters) {
      thisProxy.done();
    }
    else {
      comm_start_time = CkWallTimer();
      block_proxy.exchangeGhosts();
    }
  }

  void done() {
    double total_time = CkWallTimer() - start_time;
    CkPrintf("Finished due to max iterations %d, total time %lf seconds\n",
             n_iters, total_time);
    if (sync_ver) {
      CkPrintf("Comm time: %.3lf us\nUpdate time: %.3lf us\n",
          (comm_agg_time / n_iters) * 1000000, (update_agg_time / n_iters) * 1000000);
    }
    CkPrintf("Iteration time: %.3lf us\n", (total_time / n_iters) * 1000000);

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

  double* __restrict__ h_temperature;
  double* __restrict__ d_temperature;
  double* __restrict__ d_new_temperature;
  double* __restrict__ h_left_ghost;
  double* __restrict__ h_right_ghost;
  double* __restrict__ h_bottom_ghost;
  double* __restrict__ h_top_ghost;
  double* __restrict__ d_left_ghost;
  double* __restrict__ d_right_ghost;

  cudaStream_t stream;

  bool left_bound, right_bound, top_bound, bottom_bound;

  Block() {}

  ~Block() {
    // Free memory and destroy CUDA stream
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(cudaFree(d_temperature));
    hapiCheck(cudaFree(d_new_temperature));
    hapiCheck(cudaFreeHost(h_left_ghost));
    hapiCheck(cudaFreeHost(h_right_ghost));
    hapiCheck(cudaFreeHost(h_top_ghost));
    hapiCheck(cudaFreeHost(h_bottom_ghost));
    hapiCheck(cudaFree(d_left_ghost));
    hapiCheck(cudaFree(d_right_ghost));
    cudaStreamDestroy(stream);
  }

  void init() {
    // Initialize values
    my_iter = 1;
    neighbors = 0;
    x = thisIndex.x;
    y = thisIndex.y;

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

    // Allocate memory and create CUDA stream
    hapiCheck(cudaMallocHost((void**)&h_temperature,
          sizeof(double) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMalloc((void**)&d_temperature,
          sizeof(double) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMalloc((void**)&d_new_temperature,
          sizeof(double) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMallocHost((void**)&h_left_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost((void**)&h_right_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost((void**)&h_bottom_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost((void**)&h_top_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMalloc((void**)&d_left_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMalloc((void**)&d_right_ghost, sizeof(double) * block_size));
    cudaStreamCreate(&stream);

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_size, stream);

    // TODO: Support reduction callback in hapiAddCallback
    CkCallback* cb = new CkCallback(CkIndex_Block::initDone(), thisProxy[thisIndex]);
    hapiAddCallback(stream, cb);
  }

  void initDone() {
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  void sendGhosts(void) {
    // Pack non-contiguous ghosts to temporary contiguous buffers on device
    invokePackingKernels(d_temperature, d_left_ghost, d_right_ghost, left_bound,
        right_bound, block_size, stream);

    // Transfer ghosts from device to host
    hapiCheck(cudaMemcpyAsync(h_left_ghost, d_left_ghost, block_size * sizeof(double),
          cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaMemcpyAsync(h_right_ghost, d_right_ghost, block_size * sizeof(double),
          cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaMemcpyAsync(h_top_ghost, d_temperature + (block_size + 2) + 1,
          block_size * sizeof(double), cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaMemcpyAsync(h_bottom_ghost, d_temperature + (block_size + 2) * block_size + 1,
          block_size * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Send ghosts to neighboring chares
    if (!left_bound)
      thisProxy(x - 1, y).receiveGhosts(my_iter, RIGHT, block_size, h_left_ghost);
    if (!right_bound)
      thisProxy(x + 1, y).receiveGhosts(my_iter, LEFT, block_size, h_right_ghost);
    if (!top_bound)
      thisProxy(x, y - 1).receiveGhosts(my_iter, BOTTOM, block_size, h_top_ghost);
    if (!bottom_bound)
      thisProxy(x, y + 1).receiveGhosts(my_iter, TOP, block_size, h_bottom_ghost);
  }

  void processGhosts(int dir, int width, double* gh) {
    switch (dir) {
      case LEFT:
        memcpy(h_left_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpyAsync(d_left_ghost, h_left_ghost, block_size * sizeof(double),
              cudaMemcpyHostToDevice, stream));
        invokeUnpackingKernel(d_temperature, d_left_ghost, true, block_size, stream);
        break;
      case RIGHT:
        memcpy(h_right_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpyAsync(d_right_ghost, h_right_ghost, block_size * sizeof(double),
              cudaMemcpyHostToDevice, stream));
        invokeUnpackingKernel(d_temperature, d_right_ghost, false, block_size, stream);
        break;
      case TOP:
        memcpy(h_top_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpyAsync(d_temperature + (block_size + 2) + 1, h_top_ghost,
              block_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        break;
      case BOTTOM:
        memcpy(h_bottom_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpyAsync(d_temperature + (block_size + 2) * block_size + 1,
              h_bottom_ghost, block_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void update() {
    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_size, left_bound, right_bound,
        top_bound, bottom_bound, stream);

    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_size, stream);

    // Copy final temperature data back to host
    if (my_iter == n_iters) {
      hapiCheck(cudaMemcpyAsync(h_temperature, d_new_temperature,
            sizeof(double) * (block_size + 2) * (block_size + 2),
            cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);

    // Swap pointers
    std::swap(d_temperature, d_new_temperature);

    if (sync_ver) {
      my_iter++;
      CkCallback cb(CkReductionTarget(Main, updateDone), main_proxy);
      contribute(cb);
    } else {
      if (my_iter++ < n_iters) {
        thisProxy[thisIndex].exchangeGhosts();
      } else {
        CkCallback cb(CkReductionTarget(Main, done), main_proxy);
        contribute(cb);
      }
    }
  }

  void print() {
    CkPrintf("[%d,%d]\n", thisIndex.x, thisIndex.y);
    for (int j = 0; j < block_size + 2; j++) {
      for (int i = 0; i < block_size + 2; i++) {
        CkPrintf("%.3lf ", h_temperature[(block_size + 2) * j + i]);
      }
      CkPrintf("\n");
    }

    if (!(thisIndex.x == n_chares-1 && thisIndex.y == n_chares-1)) {
      if (thisIndex.x == n_chares-1) {
        thisProxy(0,thisIndex.y+1).print();
      }
      else {
        thisProxy(thisIndex.x+1,thisIndex.y).print();
      }
    }
    else {
      main_proxy.printDone();
    }
  }
};

#include "jacobi2d.def.h"
