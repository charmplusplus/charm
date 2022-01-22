#include "hapi.h"
#include "hapi_nvtx.h"
#include "jacobi3d.decl.h"
#include "jacobi3d.h"
#include <utility>
#include <string>
#include <sstream>

#define CUDA_SYNC 0

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int num_chares;
/* readonly */ int grid_width;
/* readonly */ int grid_height;
/* readonly */ int grid_depth;
/* readonly */ int block_width;
/* readonly */ int block_height;
/* readonly */ int block_depth;
/* readonly */ int x_surf_count;
/* readonly */ int y_surf_count;
/* readonly */ int z_surf_count;
/* readonly */ size_t x_surf_size;
/* readonly */ size_t y_surf_size;
/* readonly */ size_t z_surf_size;
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_chares_z;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ bool print_elements;

extern void invokeInitKernel(DataType* d_temperature, int block_width,
    int block_height, int block_depth, cudaStream_t stream);
extern void invokeGhostInitKernels(const std::vector<DataType*>& ghosts,
    const std::vector<int>& ghost_counts, cudaStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool bounds[], cudaStream_t stream);
extern void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int block_depth, cudaStream_t stream);
extern void invokePackingKernels(DataType* d_temperature, DataType* d_ghosts[],
    bool bounds[], int block_width, int block_height, int block_depth,
    cudaStream_t stream);
extern void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost,
    int dir, int block_width, int block_height, int block_depth,
    cudaStream_t stream);

class Main : public CBase_Main {
  int my_iter;
  double init_start_time;
  double start_time;

public:
  Main(CkArgMsg* m) {
    NVTXTracer nvtx_range("Main::Main", NVTXColor::Turquoise);

    // Set default values
    main_proxy = thisProxy;
    num_chares = CkNumPes();
    grid_width = 512;
    grid_height = 512;
    grid_depth = 512;
    n_iters = 100;
    warmup_iters = 10;
    print_elements = false;
    my_iter = 0;

    // Process arguments
    int c;
    bool dims[3] = {false, false, false};
    while ((c = getopt(m->argc, m->argv, "c:x:y:z:i:w:dsp")) != -1) {
      switch (c) {
        case 'c':
          num_chares = atoi(optarg);
          break;
        case 'x':
          grid_width = atoi(optarg);
          dims[0] = true;
          break;
        case 'y':
          grid_height = atoi(optarg);
          dims[1] = true;
          break;
        case 'z':
          grid_depth = atoi(optarg);
          dims[2] = true;
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 'w':
          warmup_iters = atoi(optarg);
          break;
        case 'p':
          print_elements = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -x [grid width] -y [grid height] -z [grid depth] "
              "-c [number of chares] -i [iterations] -w [warmup iterations] "
              "-p (print blocks)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // If only the X dimension is given, use it for Y and Z as well
    if (dims[0] && !dims[1] && !dims[2]) grid_height = grid_depth = grid_width;

    // Setup 3D grid of chares
    double area[3];
    int ipx, ipy, ipz, nremain;
    double surf, bestsurf;
    area[0] = grid_width * grid_height;
    area[1] = grid_width * grid_depth;
    area[2] = grid_height * grid_depth;
    bestsurf = 2.0 * (area[0] + area[1] + area[2]);
    ipx = 1;
    while (ipx <= num_chares) {
      if (num_chares % ipx == 0) {
        nremain = num_chares / ipx;
        ipy = 1;

        while (ipy <= nremain) {
          if (nremain % ipy == 0) {
            ipz = nremain / ipy;
            surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz;

            if (surf < bestsurf) {
              bestsurf = surf;
              n_chares_x = ipx;
              n_chares_y = ipy;
              n_chares_z = ipz;
            }
          }
          ipy++;
        }
      }
      ipx++;
    }

    if (n_chares_x * n_chares_y * n_chares_z != num_chares) {
      CkPrintf("ERROR: Bad grid of chares: %d x %d x %d != %d\n",
          n_chares_x, n_chares_y, n_chares_z, num_chares);
      CkExit(-1);
    }

    // Calculate block size
    block_width = grid_width / n_chares_x;
    block_height = grid_height / n_chares_y;
    block_depth = grid_depth / n_chares_z;

    // Calculate surface count and sizes
    x_surf_count = block_height * block_depth;
    y_surf_count = block_width * block_depth;
    z_surf_count = block_width * block_height;
    x_surf_size = x_surf_count * sizeof(DataType);
    y_surf_size = y_surf_count * sizeof(DataType);
    z_surf_size = z_surf_count * sizeof(DataType);

    // Print configuration
    CkPrintf("\n[CUDA 3D Jacobi example]\n");
    CkPrintf("Grid: %d x %d x %d, Block: %d x %d x %d, Chares: %d x %d x %d, "
        "Iterations: %d, Warm-up: %d, Print: %d\n\n",
        grid_width, grid_height, grid_depth, block_width, block_height, block_depth,
        n_chares_x, n_chares_y, n_chares_z, n_iters, warmup_iters, print_elements);

    // Create blocks and start iteration
    block_proxy = CProxy_Block::ckNew(n_chares_x, n_chares_y, n_chares_z);
    init_start_time = CkWallTimer();
    block_proxy.init();
  }

  void initDone() {
    NVTXTracer nvtx_range("Main::initDone", NVTXColor::Turquoise);

    CkPrintf("Init time: %.3lf s\n", CkWallTimer() - init_start_time);

    startIter();
  }

  void startIter() {
    if (my_iter++ == warmup_iters) start_time = CkWallTimer();

    block_proxy.run();
  }

  void warmupDone() {
    NVTXTracer nvtx_range("Main::warmupDone", NVTXColor::Turquoise);

    startIter();
  }

  void allDone() {
    NVTXTracer nvtx_range("Main::allDone", NVTXColor::Turquoise);

    double total_time = CkWallTimer() - start_time;
    CkPrintf("Total time: %.3lf s\nAverage iteration time: %.3lf us\n",
        total_time, (total_time / n_iters) * 1e6);
    CkExit();
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

 public:
  int my_iter;
  int neighbors;
  int remote_count;
  int x, y, z;
  std::string index_str;

  DataType* h_temperature;
  DataType* d_temperature;
  DataType* d_new_temperature;

  DataType* h_ghosts[DIR_COUNT];
  DataType* d_ghosts[DIR_COUNT];
  DataType* d_send_ghosts[DIR_COUNT];
  DataType* d_recv_ghosts[DIR_COUNT];

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;
  cudaStream_t h2d_stream;
  cudaStream_t d2h_stream;

  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  bool bounds[DIR_COUNT];

  Block() {}

  ~Block() {
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(cudaFree(d_temperature));
    hapiCheck(cudaFree(d_new_temperature));
    for (int i = 0; i < DIR_COUNT; i++) {
      hapiCheck(cudaFreeHost(h_ghosts[i]));
      hapiCheck(cudaFree(d_ghosts[i]));
    }

    hapiCheck(cudaStreamDestroy(compute_stream));
    hapiCheck(cudaStreamDestroy(comm_stream));
    hapiCheck(cudaStreamDestroy(h2d_stream));
    hapiCheck(cudaStreamDestroy(d2h_stream));

    hapiCheck(cudaEventDestroy(compute_event));
    hapiCheck(cudaEventDestroy(comm_event));
  }

  void init() {
    // Initialize values
    my_iter = 0;
    neighbors = 0;
    x = thisIndex.x;
    y = thisIndex.y;
    z = thisIndex.z;
    index_str = "[" + std::to_string(x) + "," + std::to_string(y)
      + "," + std::to_string(z) + "]";

    // Check bounds and set number of valid neighbors
    for (int i = 0; i < DIR_COUNT; i++) bounds[i] = false;

    if (x == 0)            bounds[LEFT] = true;
    else                   neighbors++;
    if (x == n_chares_x-1) bounds[RIGHT]= true;
    else                   neighbors++;
    if (y == 0)            bounds[TOP] = true;
    else                   neighbors++;
    if (y == n_chares_y-1) bounds[BOTTOM] = true;
    else                   neighbors++;
    if (z == 0)            bounds[FRONT] = true;
    else                   neighbors++;
    if (z == n_chares_z-1) bounds[BACK] = true;
    else                   neighbors++;

    // Allocate memory and create CUDA entities
    hapiCheck(cudaMallocHost((void**)&h_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(cudaMalloc((void**)&d_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(cudaMalloc((void**)&d_new_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    std::vector<size_t> ghost_sizes = {x_surf_size, x_surf_size, y_surf_size,
      y_surf_size, z_surf_size, z_surf_size};
    for (int i = 0; i < DIR_COUNT; i++) {
      hapiCheck(cudaMallocHost((void**)&h_ghosts[i], ghost_sizes[i]));
      hapiCheck(cudaMalloc((void**)&d_ghosts[i], ghost_sizes[i]));
    }

    // Create CUDA streams and events
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));
    hapiCheck(cudaStreamCreateWithPriority(&h2d_stream, cudaStreamDefault, -1));
    hapiCheck(cudaStreamCreateWithPriority(&d2h_stream, cudaStreamDefault, -1));

    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_width, block_height, block_depth, compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, block_depth, compute_stream);

    // Initialize ghost data
    std::vector<int> ghost_counts = {x_surf_count, x_surf_count, y_surf_count,
      y_surf_count, z_surf_count, z_surf_count};
    std::vector<DataType*> ghosts;
    for (int i = 0; i < DIR_COUNT; i++) {
      ghosts.push_back(d_ghosts[i]);
    }
    invokeGhostInitKernels(ghosts, ghost_counts, compute_stream);

    for (int i = 0; i < DIR_COUNT; i++) {
      int ghost_count = ghost_counts[i];
      for (int j = 0; j < ghost_count; j++) {
        h_ghosts[i][j] = 0;
      }
    }

    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_width, block_height, block_depth,
        bounds, compute_stream);
    invokeBoundaryKernels(d_new_temperature, block_width, block_height, block_depth,
        bounds, compute_stream);

#if CUDA_SYNC
    cudaStreamSynchronize(compute_stream);
    thisProxy[thisIndex].initDone();
#else
    // TODO: Support reduction callback in hapiAddCallback
    CkCallback* cb = new CkCallback(CkIndex_Block::initDone(), thisProxy[thisIndex]);
    hapiAddCallback(compute_stream, cb);
#endif
  }

  void packGhosts() {
    NVTXTracer nvtx_range(index_str + " packGhosts", NVTXColor::PeterRiver);

    // Pack non-contiguous ghosts to temporary contiguous buffers on device
    invokePackingKernels(d_temperature, d_ghosts, bounds,
        block_width, block_height, block_depth, comm_stream);

    // Transfer ghosts from device to host
    std::vector<size_t> ghost_sizes = {x_surf_size, x_surf_size, y_surf_size,
      y_surf_size, z_surf_size, z_surf_size};
    for (int i = 0; i < DIR_COUNT; i++) {
      if (!bounds[i]) {
        hapiCheck(cudaMemcpyAsync(h_ghosts[i], d_ghosts[i], ghost_sizes[i],
              cudaMemcpyDeviceToHost, comm_stream));
      }
    }

#if CUDA_SYNC
    cudaStreamSynchronize(comm_stream);
    thisProxy[thisIndex].packGhostsDone();
#else
    // Add asynchronous callback to be invoked when packing kernels and
    // ghost transfers are complete
    CkCallback* cb = new CkCallback(CkIndex_Block::packGhostsDone(), thisProxy[thisIndex]);
    hapiAddCallback(comm_stream, cb);
#endif
  }

  void sendGhosts() {
    NVTXTracer nvtx_range(index_str + " sendGhosts", NVTXColor::WetAsphalt);

    if (!bounds[LEFT])
      thisProxy(x-1, y, z).recvGhostReg(my_iter, RIGHT,
          x_surf_count, h_ghosts[LEFT]);
    if (!bounds[RIGHT])
      thisProxy(x+1, y, z).recvGhostReg(my_iter, LEFT,
          x_surf_count, h_ghosts[RIGHT]);
    if (!bounds[TOP])
      thisProxy(x, y-1, z).recvGhostReg(my_iter, BOTTOM,
          y_surf_count, h_ghosts[TOP]);
    if (!bounds[BOTTOM])
      thisProxy(x, y+1, z).recvGhostReg(my_iter, TOP,
          y_surf_count, h_ghosts[BOTTOM]);
    if (!bounds[FRONT])
      thisProxy(x, y, z-1).recvGhostReg(my_iter, BACK,
          z_surf_count, h_ghosts[FRONT]);
    if (!bounds[BACK])
      thisProxy(x, y, z+1).recvGhostReg(my_iter, FRONT,
          z_surf_count, h_ghosts[BACK]);
  }

  void processGhostReg(int dir, int size, DataType* gh) {
    NVTXTracer nvtx_range(index_str + " processGhostReg " + std::to_string(dir), NVTXColor::Carrot);

    DataType* h_ghost = nullptr; DataType* d_ghost = nullptr;
    CkAssert(dir >= 0 && dir < DIR_COUNT);
    h_ghost = h_ghosts[dir];
    d_ghost = d_ghosts[dir];

    memcpy(h_ghost, gh, size * sizeof(DataType));
    hapiCheck(cudaMemcpyAsync(d_ghost, h_ghost, size * sizeof(DataType),
          cudaMemcpyHostToDevice, comm_stream));
    invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height,
        block_depth, comm_stream);
  }

  void update() {
    NVTXTracer nvtx_range(index_str + " update", NVTXColor::BelizeHole);

    // Operations in compute stream should only be executed when
    // operations in communication stream (transfers and unpacking) complete
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height,
        block_depth, compute_stream);

    CkCallback update_cb(CkIndex_Block::updateDone(), thisProxy[thisIndex]);
    hapiAddCallback(compute_stream, update_cb);
  }

  void prepNextIter() {
    NVTXTracer nvtx_range(index_str + " prepNextIter", NVTXColor::GreenSea);

    std::swap(d_temperature, d_new_temperature);
    my_iter++;
    if (my_iter <= warmup_iters) {
      contribute(CkCallback(CkReductionTarget(Main, warmupDone), main_proxy));
    } else {
      if (my_iter < warmup_iters + n_iters) {
        thisProxy[thisIndex].run();
      } else {
        contribute(CkCallback(CkReductionTarget(Main, allDone), main_proxy));
      }
    }
  }

  void print() {
    hapiCheck(cudaMemcpyAsync(h_temperature, d_temperature,
          sizeof(DataType) * (block_width+2)*(block_height+2)*(block_depth+2),
          cudaMemcpyDeviceToHost, comm_stream));
    cudaStreamSynchronize(comm_stream);

    CkPrintf("[%d,%d,%d]\n", x, y, z);
    for (int k = 0; k < block_depth+2; k++) {
      for (int j = 0; j < block_height+2; j++) {
        for (int i = 0; i < block_width+2; i++) {
#ifdef TEST_CORRECTNESS
          CkPrintf("%d ", h_temperature[IDX(i,j,k)]);
#else
          CkPrintf("%.6lf ", h_temperature[IDX(i,j,k)]);
#endif
        }
        CkPrintf("\n");
      }
      CkPrintf("\n");
    }

    if (x == n_chares_x-1 && y == n_chares_y-1 && z == n_chares_z-1) {
      thisProxy.printDone();
    } else {
      if (x == n_chares_x-1 && y == n_chares_y-1) {
        thisProxy(0,0,z+1).print();
      } else {
        if (x == n_chares_x-1) {
          thisProxy(0,y+1,z).print();
        } else {
          thisProxy(x+1,y,z).print();
        }
      }
    }
  }
};

#include "jacobi3d.def.h"
