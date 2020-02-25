#include "hapi.h"
#include "jacobi2d.decl.h"
#include <string>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int grid_size;
/* readonly */ int block_size;
/* readonly */ int n_chares;
/* readonly */ int n_iters;
/* readonly */ int thread_size;
/* readonly */ bool gpu_prio;

extern void invokeKernel(cudaStream_t stream, double* d_temperature,
    double* d_new_temperature, int block_size, int thread_size);

enum Direction { LEFT = 1, RIGHT, TOP, BOTTOM };

// Used to specify LIFO ordering on callbacks
class CallbackMsg : public CMessage_CallbackMsg {
public:
  CallbackMsg() {}
};

class Main : public CBase_Main {
  double init_start_time;
  double start_time;

public:
  Main(CkArgMsg* m) {
    // Set default values
    main_proxy = thisProxy;
    grid_size = 1024;
    block_size = 128;
    n_iters = 100;
    thread_size = 1;
    gpu_prio = false;

    // Process arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:b:i:t:p")) != -1) {
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
        case 't':
          thread_size = atoi(optarg);
          break;
        case 'p':
          gpu_prio = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -s [grid size] -b [block size] -i [iterations] "
              "-t [thread coarsening factor] -p (higher priority for GPU)\n",
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
    CkPrintf("Grid: %d x %d, Block: %d x %d, Chares: %d x %d\n", grid_size, grid_size,
        block_size, block_size, n_chares, n_chares);
    CkPrintf("Iterations: %d\n", n_iters);
    CkPrintf("Thread coarsening size: %d x %d\n", thread_size, thread_size);
    CkPrintf("Higher priority for GPU methods and callbacks: %s\n\n",
             (gpu_prio) ? "ON" : "OFF");

    // Create blocks and start iteration
    block_proxy = CProxy_Block::ckNew(n_chares, n_chares);
    init_start_time = CkWallTimer();
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("Chare array initialization time: %lf seconds\n", CkWallTimer() - init_start_time);

    start_time = CkWallTimer();
    block_proxy.iterate();
  }

  void done(double time) {
    CkPrintf("\nAverage time per iteration: %lf\n",
             time / ((n_chares * n_chares) * n_iters));
    CkPrintf("Finished due to max iterations %d, total time %lf seconds\n",
             n_iters, CkWallTimer() - start_time);
    CkExit();
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

 public:
  int my_iter;
  int neighbors;
  int remote_count;

  double* __restrict__ h_temperature;
  double* __restrict__ d_temperature;
  double* __restrict__ d_new_temperature;
  double* __restrict__ left_ghost;
  double* __restrict__ right_ghost;
  double* __restrict__ bottom_ghost;
  double* __restrict__ top_ghost;

  cudaStream_t stream;

  bool left_bound, right_bound, top_bound, bottom_bound;
  double iter_start_time;
  double agg_time;

  Block() {}

  ~Block() {
    // Free memory and destroy CUDA stream
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(cudaFree(d_temperature));
    hapiCheck(cudaFree(d_new_temperature));
    hapiCheck(cudaFreeHost(left_ghost));
    hapiCheck(cudaFreeHost(right_ghost));
    hapiCheck(cudaFreeHost(top_ghost));
    hapiCheck(cudaFreeHost(bottom_ghost));
    cudaStreamDestroy(stream);
  }

  void init() {
    // Initialize values
    my_iter = 1;
    agg_time = 0.0;
    neighbors = 0;

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
      bottom_bound = true;
    else
      neighbors++;
    if (thisIndex.y == n_chares - 1)
      top_bound = true;
    else
      neighbors++;

    // Allocate memory and create CUDA stream
    hapiCheck(cudaMallocHost((void**)&h_temperature,
          sizeof(double) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMalloc((void**)&d_temperature,
          sizeof(double) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMalloc((void**)&d_new_temperature,
          sizeof(double) * (block_size + 2) * (block_size + 2)));
    hapiCheck(cudaMallocHost((void**)&left_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost((void**)&right_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost((void**)&bottom_ghost, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost((void**)&top_ghost, sizeof(double) * block_size));
    cudaStreamCreate(&stream);

    // Initialize temperature data
    // FIXME: Do this on the device
    for (int j = 0; j < block_size + 2; j++) {
      for (int i = 0; i < block_size + 2; i++) {
        h_temperature[(block_size + 2) * j + i] = 0.0;
      }
    }

    // Enforce boundary conditions
    constrainBC();

    CkCallback cb(CkReductionTarget(Main, initDone), main_proxy);
    contribute(cb);
  }

  void sendGhosts(void) {
    // Copy temperature data to the GPU on first iteration
    if (my_iter == 1) {
      hapiCheck(cudaMemcpyAsync(d_temperature, h_temperature,
                          sizeof(double) * (block_size + 2) * (block_size + 2),
                          cudaMemcpyHostToDevice, stream));
    }

    int x = thisIndex.x, y = thisIndex.y;
    if (!left_bound)
      thisProxy(x - 1, y).receiveGhosts(my_iter, RIGHT, block_size, left_ghost);
    if (!right_bound)
      thisProxy(x + 1, y).receiveGhosts(my_iter, LEFT, block_size, right_ghost);
    if (!top_bound)
      thisProxy(x, y + 1).receiveGhosts(my_iter, BOTTOM, block_size, top_ghost);
    if (!bottom_bound)
      thisProxy(x, y - 1).receiveGhosts(my_iter, TOP, block_size, bottom_ghost);
  }

  void processGhosts(int dir, int width, double* gh) {
    // TODO: Don't use cudaMemcpy2DAsync, use kernel instead
    switch (dir) {
      case LEFT:
        memcpy(left_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpy2DAsync(
            d_temperature + (block_size + 2), (block_size + 2) * sizeof(double),
            left_ghost, sizeof(double), sizeof(double), block_size,
            cudaMemcpyHostToDevice, stream));
        break;
      case RIGHT:
        memcpy(right_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpy2DAsync(
            d_temperature + (block_size + 2) + (block_size + 1),
            (block_size + 2) * sizeof(double), right_ghost, sizeof(double),
            sizeof(double), block_size, cudaMemcpyHostToDevice, stream));
        break;
      case BOTTOM:
        memcpy(bottom_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpyAsync(d_temperature + 1, bottom_ghost,
              block_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        break;
      case TOP:
        memcpy(top_ghost, gh, width * sizeof(double));
        hapiCheck(cudaMemcpyAsync(d_temperature + (block_size + 2) * (block_size + 1) + 1,
              top_ghost, block_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void update() {
    // Invoke GPU kernel
    invokeKernel(stream, d_temperature, d_new_temperature, block_size, thread_size);

    // Transfer left ghost
    hapiCheck(cudaMemcpy2DAsync(left_ghost, sizeof(double), d_new_temperature + (block_size + 2),
          (block_size + 2) * sizeof(double), sizeof(double), block_size, cudaMemcpyDeviceToHost, stream));

    // Transfer right ghost
    hapiCheck(cudaMemcpy2DAsync(right_ghost, sizeof(double), d_new_temperature + (block_size + 2) + (block_size + 1),
          (block_size + 2) * sizeof(double), sizeof(double), block_size, cudaMemcpyDeviceToHost, stream));

    // Transfer bottom ghost
    hapiCheck(cudaMemcpyAsync(bottom_ghost, d_new_temperature + 1, block_size * sizeof(double),
          cudaMemcpyDeviceToHost, stream));

    // Transfer top ghost
    hapiCheck(cudaMemcpyAsync(top_ghost, d_new_temperature + (block_size + 2) * (block_size + 1) + 1,
          block_size * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // Copy final temperature data back to host
    if (my_iter == n_iters) {
      hapiCheck(cudaMemcpyAsync(h_temperature, d_new_temperature, sizeof(double) * (block_size + 2) * (block_size + 2),
            cudaMemcpyDeviceToHost, stream));
    }

    cudaStreamSynchronize(stream);

    thisProxy(thisIndex.x, thisIndex.y).iterate();

    /*
    CallbackMsg* m = new CallbackMsg();
    CkArrayIndex2D myIndex = CkArrayIndex2D(thisIndex);
    CkCallback* cb =
        new CkCallback(CkIndex_Stencil::iterate(NULL), myIndex, thisProxy);
    if (gpu_prio)
      CkSetQueueing(m, CK_QUEUEING_LIFO);
    hapiAddCallback(stream, cb, m);
    */
  }

  void constrainBC() {
    // FIXME: Do this on the device
    if (left_bound) {
      for (int j = 0; j < block_size + 2; ++j) {
        h_temperature[j * (block_size + 2)] = 1.0;
      }
    }
    if (right_bound) {
      for (int j = 0; j < block_size + 2; ++j) {
        h_temperature[j * (block_size + 2) + (block_size + 1)] = 1.0;
      }
    }
    if (top_bound) {
      for (int i = 0; i < block_size + 2; ++i) {
        h_temperature[(block_size + 1) * (block_size + 2) + i] = 1.0;
      }
    }
    if (bottom_bound) {
      for (int i = 0; i < block_size + 2; ++i) {
        h_temperature[i] = 1.0;
      }
    }
  }
};

#include "jacobi2d.def.h"
