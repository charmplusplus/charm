#include "hapi.h"
#include "stencil2d.decl.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif
#include <string>

#define CPU_MODE 10
#define CUDA_MODE 11
#define HAPI_MODE 12

#define LEFT 1
#define RIGHT 2
#define TOP 3
#define BOTTOM 4
#define DIVIDEBY5 0.2

#define USE_CUSTOM_MAP 1 // Should be set to 1 to use GPU handler PEs

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int grid_x;
/* readonly */ int grid_y;
/* readonly */ int block_x;
/* readonly */ int block_y;
/* readonly */ int num_chares_x;
/* readonly */ int num_chares_y;
/* readonly */ int num_iters;
/* readonly */ int global_exec_mode;
/* readonly */ int thread_size;
/* readonly */ float offload_ratio;
/* readonly */ bool gpu_prio;
/* readonly */ int gpu_pes;

extern void invokeKernel(cudaStream_t stream, double* d_temperature,
                         double* d_new_temperature, int block_x, int block_y,
                         int thread_size);

// Calculate the number of digits.
int numDigits(int n) {
  int digits = 0;
  if (n < 0) digits = 1;
  while (n) {
    n /= 10;
    digits++;
  }
  return digits;
}

class CustomMap : public CkArrayMap {
  public:
    CustomMap() {}

    int registerArray(CkArrayIndex& numElements, CkArrayID aid) {
      return 0;
    }

    int procNum(int, const CkArrayIndex &idx) {
      int x_index = ((int*)idx.data())[0];
      int y_index = ((int*)idx.data())[1];
      int elem = num_chares_y * x_index + y_index;
      int penum;
      int gpu_chares = num_chares_x * num_chares_y * offload_ratio;

      if (gpu_pes > 0) {
        if (elem < gpu_chares) {
          // GPU chares
          penum = elem % gpu_pes + (CkNumPes() - gpu_pes);
        }
        else {
          // CPU chares
          if (gpu_pes != CkNumPes()) {
            penum = elem % (CkNumPes() - gpu_pes);
          }
          else {
            penum = 0; // No normal PEs; place all CPU chares on PE 0
          }
        }
      }
      else {
        // No GPU PE
        penum = elem % CkNumPes();
      }

      return penum;
    }
};

// Used to specify LIFO ordering on callbacks.
class CallbackMsg : public CMessage_CallbackMsg {
 public:
  CallbackMsg() {}
};

class Main : public CBase_Main {
  double init_start_time;
  double start_time;

 public:
  CProxy_Stencil stencils;

  Main(CkArgMsg* m) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::Main", NVTXColor::Turquoise);
#endif
    // Set default values
    mainProxy = thisProxy;
    grid_x = grid_y = 1024;
    block_x = block_y = 128;
    num_iters = 10;
    global_exec_mode = CPU_MODE;
    thread_size = 1;
    offload_ratio = 0.0;
    gpu_prio = false;
    gpu_pes = 0;

    // Process arguments
    int c;
    bool sFlag = false;
    bool bFlag = false;
    while ((c = getopt(m->argc, m->argv, "s:b:i:uyt:r:pg:")) != -1) {
      switch (c) {
        case 's':
          grid_x = grid_y = atoi(optarg);
          sFlag = true;
          break;
        case 'b':
          block_x = block_y = atoi(optarg);
          bFlag = true;
          break;
        case 'i':
          num_iters = atoi(optarg);
          break;
        case 'u':
          global_exec_mode = CUDA_MODE;
          break;
        case 'y':
          global_exec_mode = HAPI_MODE;
          break;
        case 'r':
          offload_ratio = atof(optarg);
          break;
        case 't':
          thread_size = atoi(optarg);
          break;
        case 'p':
          gpu_prio = true;
          break;
        case 'g':
          gpu_pes = atoi(optarg);
          break;
        default:
          CkPrintf(
              "Usage: %s -s [grid size] -b [block size] -i [iterations] -u/y: "
              "CUDA/HAPI -r [offload ratio] -t [thread size] -p: higher"
              "priority for GPU -g [GPU handler PEs]\n",
              m->argv[0]);
          CkExit();
      }
    }

    if (sFlag && !bFlag) block_x = block_y = grid_x;
    if (grid_x < block_x || grid_x % block_x != 0)
      CkAbort("array_size_X %% block_size_X != 0!");
    if (grid_y < block_y || block_y % block_y != 0)
      CkAbort("array_size_Y %% block_size_Y != 0!");
    if (offload_ratio < 0.0f || offload_ratio > 1.0f)
      CkAbort("offload_ratio should be between 0 and 1!");
    if (offload_ratio > 0.0f && global_exec_mode == CPU_MODE) {
      CkPrintf("Offload ratio set higher than 0 but GPU mode not set!\n"
               "Reverting offload ratio to 0...\n");
      offload_ratio = 0.0f;
    }
    if (gpu_pes > CkNumPes()) {
      CkPrintf("More GPU handler PEs than total number of PEs!\n"
               "Setting it to equal the total number of PEs...\n");
      gpu_pes = CkNumPes();
    }

    num_chares_x = grid_x / block_x;
    num_chares_y = grid_y / block_y;

    // Print info
    CkPrintf("\n[CUDA 2D stencil example]\n");
    CkPrintf("Execution mode: %s\n",
             ((global_exec_mode == CPU_MODE)
                  ? "CPU only"
                  : ((global_exec_mode == CUDA_MODE) ? "CPU + CUDA"
                                                     : "CPU + HAPI")));
    CkPrintf("Chares: %d x %d\n", num_chares_x, num_chares_y);
    CkPrintf("Grid dimensions: %d x %d\n", grid_x, grid_y);
    CkPrintf("Block dimensions: %d x %d\n", block_x, block_y);
    CkPrintf("Iterations: %d\n", num_iters);
    CkPrintf("Offload ratio: %.2f\n", offload_ratio);
    CkPrintf("Thread coarsening size: %d x %d\n", thread_size, thread_size);
    CkPrintf("Higher priority for GPU methods and callbacks: %s\n",
             (gpu_prio) ? "ON" : "OFF");
    CkPrintf("GPU handler PEs: %d\n\n", gpu_pes);
    delete m;

    // Create 2D chare array
#if USE_CUSTOM_MAP
    CkArrayOptions opts(num_chares_x, num_chares_y);
    CProxy_CustomMap cmap = CProxy_CustomMap::ckNew();
    opts.setMap(cmap);
    stencils = CProxy_Stencil::ckNew(opts);
#else
    stencils = CProxy_Stencil::ckNew(num_chares_x, num_chares_y);
#endif

    // Start measuring initialization time
    init_start_time = CkWallTimer();

    // Initialize workers
    stencils.init();
  }

  void initDone() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::initDone", NVTXColor::Emerald);
#endif
    CkPrintf("\nChare array initialization time: %lf seconds\n\n",
             CkWallTimer() - init_start_time);

    // Start measuring total execution time
    start_time = CkWallTimer();

    // Start stencil iterations
    CallbackMsg* m = new CallbackMsg();
    stencils.iterate(m);
  }

  void done(double time) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::done", NVTXColor::Emerald);
#endif
    CkPrintf("\nAverage time per iteration: %lf\n",
             time / ((num_chares_x * num_chares_y) * num_iters));
    CkPrintf("Finished due to max iterations %d, total time %lf seconds\n",
             num_iters, CkWallTimer() - start_time);
    CkExit();
  }
};

class Stencil : public CBase_Stencil {
  Stencil_SDAG_CODE

 public:
  int n_digits;
  int my_iter;
  int neighbors;
  int remote_count;

  double* __restrict__ temperature;
  double* __restrict__ new_temperature;
  double* __restrict__ d_temperature;
  double* __restrict__ d_new_temperature;
  double* __restrict__ left_ghost;
  double* __restrict__ right_ghost;
  double* __restrict__ bottom_ghost;
  double* __restrict__ top_ghost;

  cudaStream_t stream;

  int thisFlatIndex;
  int local_exec_mode;

  bool left_bound, right_bound, top_bound, bottom_bound;
  double iter_start_time;
  double agg_time;

  Stencil() {}

  ~Stencil() {
    if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
      hapiCheck(cudaFreeHost(temperature));
      hapiCheck(cudaFree(d_temperature));
      hapiCheck(cudaFree(d_new_temperature));
      hapiCheck(cudaFreeHost(left_ghost));
      hapiCheck(cudaFreeHost(right_ghost));
      hapiCheck(cudaFreeHost(top_ghost));
      hapiCheck(cudaFreeHost(bottom_ghost));

      cudaStreamDestroy(stream);
    } else { // CPU_MODE
      delete temperature;
      delete new_temperature;
      delete left_ghost;
      delete right_ghost;
      delete top_ghost;
      delete bottom_ghost;
    }
  }

  void init() {
    thisFlatIndex = num_chares_y * thisIndex.x + thisIndex.y;

#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Stencil::initialize", NVTXColor::SunFlower);
#endif

    // Determine execution mode
#if USE_CUSTOM_MAP
    local_exec_mode = global_exec_mode;
    if (thisFlatIndex >= num_chares_x * num_chares_y * offload_ratio) {
      local_exec_mode = CPU_MODE;
    }
#else
    int num_chares_pe = (num_chares_x * num_chares_y) / CkNumPes();
    int leftover = (num_chares_x * num_chares_y) % CkNumPes();
    int start_index = num_chares_pe * CkMyPe();
    if (CkMyPe() < leftover) {
      num_chares_pe++;
      start_index += CkMyPe();
    } else {
      start_index += leftover;
    }
    int this_rank = thisFlatIndex - start_index;

    local_exec_mode = global_exec_mode;
    if (this_rank >= num_chares_pe * offload_ratio) {
      local_exec_mode = CPU_MODE;
    }
#endif

    // Print execution mode and PE
    n_digits = numDigits(num_chares_x * num_chares_y);
    std::string mode_string;
    switch (local_exec_mode) {
      case CPU_MODE:
        mode_string = "CPU";
        break;
      case CUDA_MODE:
        mode_string = "CUDA";
        break;
      case HAPI_MODE:
        mode_string = "HAPI";
        break;
    }
    CkPrintf("[%*d] Mode: %s, PE: %d\n", n_digits, thisFlatIndex, mode_string.c_str(), CkMyPe());

    // Initialize values
    my_iter = 0;
    agg_time = 0.0;
    neighbors = 0;

    // Check bounds and set number of valid neighbors
    left_bound = right_bound = top_bound = bottom_bound = false;
    if (thisIndex.x == 0)
      left_bound = true;
    else
      neighbors++;
    if (thisIndex.x == num_chares_x - 1)
      right_bound = true;
    else
      neighbors++;
    if (thisIndex.y == 0)
      bottom_bound = true;
    else
      neighbors++;
    if (thisIndex.y == num_chares_y - 1)
      top_bound = true;
    else
      neighbors++;

    // Allocate memory and create CUDA stream
    if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
      hapiCheck(
          cudaMallocHost((void**)&temperature,
                         sizeof(double) * (block_x + 2) * (block_y + 2)));
      hapiCheck(cudaMalloc((void**)&d_temperature,
                           sizeof(double) * (block_x + 2) * (block_y + 2)));
      hapiCheck(cudaMalloc((void**)&d_new_temperature,
                           sizeof(double) * (block_x + 2) * (block_y + 2)));
      hapiCheck(
          cudaMallocHost((void**)&left_ghost, sizeof(double) * block_y));
      hapiCheck(
          cudaMallocHost((void**)&right_ghost, sizeof(double) * block_y));
      hapiCheck(
          cudaMallocHost((void**)&bottom_ghost, sizeof(double) * block_x));
      hapiCheck(cudaMallocHost((void**)&top_ghost, sizeof(double) * block_x));

      cudaStreamCreate(&stream);
    } else {  // CPU_MODE
      temperature = new double[(block_x + 2) * (block_y + 2)];
      new_temperature = new double[(block_x + 2) * (block_y + 2)];
      left_ghost = new double[block_y];
      right_ghost = new double[block_y];
      top_ghost = new double[block_x];
      bottom_ghost = new double[block_x];
    }

    // Initialize temperature data
    for (int j = 0; j < block_y + 2; j++) {
      for (int i = 0; i < block_x + 2; i++) {
        temperature[(block_x + 2) * j + i] = 0.0;
        if (local_exec_mode == CPU_MODE) {
          new_temperature[(block_x + 2) * j + i] = 0.0;
        }
      }
    }

    // Enforce boundary conditions
    constrainBC();

    CkCallback cb(CkReductionTarget(Main, initDone), mainProxy);
    contribute(cb);
  }

  void sendGhosts(void) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Stencil::sendGhosts", NVTXColor::PeterRiver);
#endif
    // Copy temperature data to the GPU on first iteration
    if ((local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) &&
        my_iter == 0) {
      hapiCheck(
          cudaMemcpyAsync(d_temperature, temperature,
                          sizeof(double) * (block_x + 2) * (block_y + 2),
                          cudaMemcpyHostToDevice, stream));
    }

    // Copy different faces into messages.
    // For GPU modes, the ghost data gets filled directly via cudaMemcpy.
    if (local_exec_mode == CPU_MODE) {
      for (int j = 0; j < block_y; j++) {
        left_ghost[j] = temperature[(block_x + 2) * (1 + j)];
        right_ghost[j] =
            temperature[(block_x + 2) * (1 + j) + (block_x + 1)];
      }

      for (int i = 0; i < block_x; i++) {
        bottom_ghost[i] = temperature[1 + i];
        top_ghost[i] = temperature[(block_x + 2) * (block_y + 1) + (1 + i)];
      }
    }

    int x = thisIndex.x, y = thisIndex.y;
    if (!left_bound)
      thisProxy(x - 1, y).receiveGhosts(my_iter, RIGHT, block_y, left_ghost);
    if (!right_bound)
      thisProxy(x + 1, y).receiveGhosts(my_iter, LEFT, block_y, right_ghost);
    if (!top_bound)
      thisProxy(x, y + 1).receiveGhosts(my_iter, BOTTOM, block_x, top_ghost);
    if (!bottom_bound)
      thisProxy(x, y - 1).receiveGhosts(my_iter, TOP, block_x, bottom_ghost);
  }

  void processGhosts(int dir, int width, double* gh) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Stencil::processGhosts", NVTXColor::WetAsphalt);
#endif
    switch (dir) {
      case LEFT:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(left_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpy2DAsync(
              d_temperature + (block_x + 2), (block_x + 2) * sizeof(double),
              left_ghost, sizeof(double), sizeof(double), block_y,
              cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (1 + j)] = gh[j];
          }
        }
        break;
      case RIGHT:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(right_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpy2DAsync(
              d_temperature + (block_x + 2) + (block_x + 1),
              (block_x + 2) * sizeof(double), right_ghost, sizeof(double),
              sizeof(double), block_y, cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (1 + j) + (block_x + 1)] = gh[j];
          }
        }
        break;
      case BOTTOM:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(bottom_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpyAsync(d_temperature + 1, bottom_ghost,
                                    block_x * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[1 + j] = gh[j];
          }
        }
        break;
      case TOP:
        if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
          memcpy(top_ghost, gh, width * sizeof(double));
          hapiCheck(cudaMemcpyAsync(
              d_temperature + (block_x + 2) * (block_y + 1) + 1, top_ghost,
              block_x * sizeof(double), cudaMemcpyHostToDevice, stream));
        } else {
          for (int j = 0; j < width; j++) {
            temperature[(block_x + 2) * (block_y + 1) + (1 + j)] = gh[j];
          }
        }
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  // Updates local data with stencil computation.
  void update() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range(std::to_string(thisFlatIndex) + " Stencil::update", NVTXColor::Amethyst);
#endif

    CallbackMsg* m = new CallbackMsg();
    if (local_exec_mode == CUDA_MODE || local_exec_mode == HAPI_MODE) {
      // Invoke 2D stencil kernel
      invokeKernel(stream, d_temperature, d_new_temperature, block_x, block_y,
                   thread_size);

      // Transfer left ghost
      hapiCheck(cudaMemcpy2DAsync(left_ghost, sizeof(double),
            d_new_temperature + (block_x + 2),
            (block_x + 2) * sizeof(double), sizeof(double),
            block_y, cudaMemcpyDeviceToHost, stream));

      // Transfer right ghost
      hapiCheck(
          cudaMemcpy2DAsync(right_ghost, sizeof(double),
            d_new_temperature + (block_x + 2) + (block_x + 1),
            (block_x + 2) * sizeof(double), sizeof(double),
            block_y, cudaMemcpyDeviceToHost, stream));

      // Transfer bottom ghost
      hapiCheck(cudaMemcpyAsync(bottom_ghost, d_new_temperature + 1,
            block_x * sizeof(double), cudaMemcpyDeviceToHost,
            stream));

      // Transfer top ghost
      hapiCheck(cudaMemcpyAsync(
            top_ghost, d_new_temperature + (block_x + 2) * (block_y + 1) + 1,
            block_x * sizeof(double), cudaMemcpyDeviceToHost, stream));

      // Copy final temperature data back to host (on last iteration)
      if (my_iter == num_iters - 1) {
        hapiCheck(
            cudaMemcpyAsync(temperature, d_new_temperature,
                            sizeof(double) * (block_x + 2) * (block_y + 2),
                            cudaMemcpyDeviceToHost, stream));
      }

      if (local_exec_mode == CUDA_MODE) {
        cudaStreamSynchronize(stream);

        thisProxy(thisIndex.x, thisIndex.y).iterate(m);
      } else {
        CkArrayIndex2D myIndex = CkArrayIndex2D(thisIndex);
        CkCallback* cb =
            new CkCallback(CkIndex_Stencil::iterate(NULL), myIndex, thisProxy);
        if (gpu_prio)
          CkSetQueueing(m, CK_QUEUEING_LIFO);
        hapiAddCallback(stream, cb, m);
      }
    } else {  // CPU_MODE
      for (int i = 1; i <= block_x; ++i) {
        for (int j = 1; j <= block_y; ++j) {
          // Update my value based on the surrounding values
          new_temperature[j * (block_x + 2) + i] =
              (temperature[j * (block_x + 2) + (i - 1)] +
               temperature[j * (block_x + 2) + (i + 1)] +
               temperature[(j - 1) * (block_x + 2) + i] +
               temperature[(j + 1) * (block_x + 2) + i] +
               temperature[j * (block_x + 2) + i]) *
              DIVIDEBY5;
        }
      }
      double* tmp;
      tmp = temperature;
      temperature = new_temperature;
      new_temperature = tmp;

      thisProxy(thisIndex.x, thisIndex.y).iterate(m);
    }
  }

  void constrainBC() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Stencil::constrainBC", NVTXColor::Carrot);
#endif
    if (left_bound) {
      for (int j = 0; j < block_y + 2; ++j) {
        temperature[j * (block_x + 2)] = 1.0;
        if (local_exec_mode == CPU_MODE) {
          new_temperature[j * (block_x + 2)] = 1.0;
        }
      }
    }
    if (right_bound) {
      for (int j = 0; j < block_y + 2; ++j) {
        temperature[j * (block_x + 2) + (block_x + 1)] = 1.0;
        if (local_exec_mode == CPU_MODE) {
          new_temperature[j * (block_x + 2) + (block_x + 1)] = 1.0;
        }
      }
    }
    if (top_bound) {
      for (int i = 0; i < block_x + 2; ++i) {
        temperature[(block_y + 1) * (block_x + 2) + i] = 1.0;
        if (local_exec_mode == CPU_MODE) {
          new_temperature[(block_y + 1) * (block_x + 2) + i] = 1.0;
        }
      }
    }
    if (bottom_bound) {
      for (int i = 0; i < block_x + 2; ++i) {
        temperature[i] = 1.0;
        if (local_exec_mode == CPU_MODE) {
          new_temperature[i] = 1.0;
        }
      }
    }
  }
};

#include "stencil2d.def.h"
