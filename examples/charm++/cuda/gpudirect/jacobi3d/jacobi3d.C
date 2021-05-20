#include "hapi.h"
#include "jacobi3d.decl.h"
#include "jacobi3d.h"
#include <utility>
#include <sstream>

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Manager manager_proxy;
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
/* readonly */ bool use_zerocopy;
/* readonly */ bool print_elements;

extern void invokeInitKernel(DataType* d_temperature, int block_width,
    int block_height, int block_depth, cudaStream_t stream);
extern void invokeGhostInitKernels(const std::vector<DataType*>& ghosts,
    const std::vector<int>& ghost_counts, cudaStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool bounds[], cudaStream_t stream);
extern void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int block_depth, cudaStream_t stream);
extern void invokePackingKernel(DataType* d_temperature, DataType* d_ghost,
    int dir, int block_width, int block_height, int block_depth,
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
    // Set default values
    main_proxy = thisProxy;
    num_chares = CkNumPes();
    grid_width = 512;
    grid_height = 512;
    grid_depth = 512;
    n_iters = 100;
    warmup_iters = 10;
    use_zerocopy = false;
    print_elements = false;
    my_iter = 0;

    // Process arguments
    int c;
    bool dims[3] = {false, false, false};
    while ((c = getopt(m->argc, m->argv, "c:x:y:z:i:w:dp")) != -1) {
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
        case 'd':
          use_zerocopy = true;
          break;
        case 'p':
          print_elements = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -x [grid width] -y [grid height] -z [grid depth] "
              "-c [number of chares] -i [iterations] -w [warmup iterations] "
              "-d (use GPU zerocopy) -p (print blocks)\n",
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
        "Iterations: %d, Warm-up: %d, Zerocopy: %d, Print: %d\n\n",
        grid_width, grid_height, grid_depth, block_width, block_height, block_depth,
        n_chares_x, n_chares_y, n_chares_z, n_iters, warmup_iters, use_zerocopy,
        print_elements);

    // Create blocks and start iteration
    block_proxy = CProxy_Block::ckNew(n_chares_x, n_chares_y, n_chares_z);
    manager_proxy = CProxy_Manager::ckNew();
    init_start_time = CkWallTimer();
  }

  void managerInitDone() {
    CkPrintf("Managers initialized\n");
    block_proxy.init();
  }

  void initDone() {
    CkPrintf("Init time: %.3lf s\n", CkWallTimer() - init_start_time);

    startIter();
  }

  void startIter() {
    if (my_iter++ == warmup_iters) start_time = CkWallTimer();

    block_proxy.run();
  }

  void warmupDone() {
    startIter();
  }

  void allDone() {
    double total_time = CkWallTimer() - start_time;
    CkPrintf("Total time: %.3lf s\nAverage iteration time: %.3lf ms\n",
        total_time, (total_time / n_iters) * 1e3);
    CkExit();
  }
};

class Manager : public CBase_Manager {
public:
  cudaStream_t compute_stream;
  cudaStream_t h2d_stream;
  cudaStream_t d2h_stream;
  cudaStream_t unpack_stream;
  cudaStream_t pack_stream;

  Manager() {
    // Create CUDA streams
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&h2d_stream, cudaStreamDefault, -1));
    hapiCheck(cudaStreamCreateWithPriority(&d2h_stream, cudaStreamDefault, -1));
    hapiCheck(cudaStreamCreateWithPriority(&unpack_stream, cudaStreamDefault, -1));
    hapiCheck(cudaStreamCreateWithPriority(&pack_stream, cudaStreamDefault, -1));

    contribute(CkCallback(CkReductionTarget(Main, managerInitDone), main_proxy));
  }

  ~Manager() {
    // Destroy CUDA streams
    hapiCheck(cudaStreamDestroy(compute_stream));
    hapiCheck(cudaStreamDestroy(h2d_stream));
    hapiCheck(cudaStreamDestroy(d2h_stream));
    hapiCheck(cudaStreamDestroy(unpack_stream));
    hapiCheck(cudaStreamDestroy(pack_stream));
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

 public:
  int my_iter;
  int neighbors;
  int remote_count;
  int x, y, z;

  DataType* h_temperature;
  DataType* d_temperature;
  DataType* d_new_temperature;

  DataType* h_ghosts[DIR_COUNT];
  DataType* d_ghosts[DIR_COUNT];
  DataType* d_send_ghosts[DIR_COUNT];
  DataType* d_recv_ghosts[DIR_COUNT];
  CkDeviceBuffer send_ghosts_buf[DIR_COUNT];

  Manager* manager;
  cudaEvent_t compute_pack_event;
  cudaEvent_t pack_d2h_events[DIR_COUNT];
  cudaEvent_t h2d_unpack_events[DIR_COUNT];
  cudaEvent_t unpack_compute_event;

  bool bounds[DIR_COUNT];
  double comm_start_time;
  double comm_time;

  Block() {}

  ~Block() {
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(cudaFree(d_temperature));
    hapiCheck(cudaFree(d_new_temperature));
    if (use_zerocopy) {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(cudaFree(d_send_ghosts[i]));
        hapiCheck(cudaFree(d_recv_ghosts[i]));
      }
    } else {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(cudaFreeHost(h_ghosts[i]));
        hapiCheck(cudaFree(d_ghosts[i]));
      }
    }

    hapiCheck(cudaEventDestroy(compute_pack_event));
    for (int i = 0; i < DIR_COUNT; i++) {
      hapiCheck(cudaEventDestroy(pack_d2h_events[i]));
      hapiCheck(cudaEventDestroy(h2d_unpack_events[i]));
    }
    hapiCheck(cudaEventDestroy(unpack_compute_event));
  }

  void init() {
    // Initialize values
    my_iter = 0;
    neighbors = 0;
    x = thisIndex.x;
    y = thisIndex.y;
    z = thisIndex.z;
    comm_time = 0;

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
    if (use_zerocopy) {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(cudaMalloc((void**)&d_send_ghosts[i], ghost_sizes[i]));
        hapiCheck(cudaMalloc((void**)&d_recv_ghosts[i], ghost_sizes[i]));
      }

      // Create CkDeviceBuffers
      send_ghosts_buf[LEFT] = CkDeviceBuffer(d_send_ghosts[LEFT], x_surf_count);
      send_ghosts_buf[RIGHT] = CkDeviceBuffer(d_send_ghosts[RIGHT], x_surf_count);
      send_ghosts_buf[TOP] = CkDeviceBuffer(d_send_ghosts[TOP], y_surf_count);
      send_ghosts_buf[BOTTOM] = CkDeviceBuffer(d_send_ghosts[BOTTOM], y_surf_count);
      send_ghosts_buf[FRONT] = CkDeviceBuffer(d_send_ghosts[FRONT], z_surf_count);
      send_ghosts_buf[BACK] = CkDeviceBuffer(d_send_ghosts[BACK], z_surf_count);
    } else {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(cudaMallocHost((void**)&h_ghosts[i], ghost_sizes[i]));
        hapiCheck(cudaMalloc((void**)&d_ghosts[i], ghost_sizes[i]));
      }
    }

    // Create CUDA events for enforcing dependencies
    manager = manager_proxy.ckLocalBranch();
    hapiCheck(cudaEventCreateWithFlags(&compute_pack_event, cudaEventDisableTiming));
    for (int i = 0; i < DIR_COUNT; i++) {
      hapiCheck(cudaEventCreateWithFlags(&pack_d2h_events[i], cudaEventDisableTiming));
      hapiCheck(cudaEventCreateWithFlags(&h2d_unpack_events[i], cudaEventDisableTiming));
    }
    hapiCheck(cudaEventCreateWithFlags(&unpack_compute_event, cudaEventDisableTiming));

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_width, block_height, block_depth, manager->compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, block_depth, manager->compute_stream);

    // Initialize ghost data
    std::vector<int> ghost_counts = {x_surf_count, x_surf_count, y_surf_count,
      y_surf_count, z_surf_count, z_surf_count};
    if (use_zerocopy) {
      std::vector<DataType*> send_ghosts;
      std::vector<DataType*> recv_ghosts;
      for (int i = 0; i < DIR_COUNT; i++) {
        send_ghosts.push_back(d_send_ghosts[i]);
        recv_ghosts.push_back(d_recv_ghosts[i]);
      }
      invokeGhostInitKernels(send_ghosts, ghost_counts, manager->compute_stream);
      invokeGhostInitKernels(recv_ghosts, ghost_counts, manager->compute_stream);
    } else {
      std::vector<DataType*> ghosts;
      for (int i = 0; i < DIR_COUNT; i++) {
        ghosts.push_back(d_ghosts[i]);
      }
      invokeGhostInitKernels(ghosts, ghost_counts, manager->compute_stream);

      for (int i = 0; i < DIR_COUNT; i++) {
        int ghost_count = ghost_counts[i];
        for (int j = 0; j < ghost_count; j++) {
          h_ghosts[i][j] = 0;
        }
      }
    }

    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_width, block_height, block_depth,
        bounds, manager->compute_stream);
    invokeBoundaryKernels(d_new_temperature, block_width, block_height, block_depth,
        bounds, manager->compute_stream);

    // TODO: Support reduction callback in hapiAddCallback
    CkCallback* cb = new CkCallback(CkIndex_Block::initDone(), thisProxy[thisIndex]);
    hapiAddCallback(manager->compute_stream, cb);
  }

  void initDone() {
    contribute(CkCallback(CkReductionTarget(Main, initDone), main_proxy));
  }

  void updateAndPack() {
    // Enforce unpack -> compute dependency
    hapiCheck(cudaEventRecord(unpack_compute_event, manager->unpack_stream));
    hapiCheck(cudaStreamWaitEvent(manager->compute_stream, unpack_compute_event, 0));

    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height,
        block_depth, manager->compute_stream);
    hapiCheck(cudaEventRecord(compute_pack_event, manager->compute_stream));

    std::vector<size_t> ghost_sizes = {x_surf_size, x_surf_size, y_surf_size,
      y_surf_size, z_surf_size, z_surf_size};

    for (int i = 0; i < DIR_COUNT; i++) {
      if (!bounds[i]) {
        // Enforce compute -> pack dependency
        hapiCheck(cudaStreamWaitEvent(manager->pack_stream, compute_pack_event, 0));

        // Pack
        DataType* pack_ghost = use_zerocopy ? d_send_ghosts[i] : d_ghosts[i];
        invokePackingKernel(d_temperature, pack_ghost, i, block_width, block_height,
            block_depth, manager->pack_stream);

        if (!use_zerocopy) {
          // Enforce pack -> d2h dependency
          hapiCheck(cudaEventRecord(pack_d2h_events[i], manager->pack_stream));
          hapiCheck(cudaStreamWaitEvent(manager->d2h_stream, pack_d2h_events[i], 0));

          // Transfer ghosts from device to host when packing kernel completes
          hapiCheck(cudaMemcpyAsync(h_ghosts[i], d_ghosts[i], ghost_sizes[i],
                cudaMemcpyDeviceToHost, manager->d2h_stream));
        }
      }
    }

    CkCallback* cb = new CkCallback(CkIndex_Block::packDone(), thisProxy[thisIndex]);
    if (use_zerocopy) {
      hapiAddCallback(manager->pack_stream, cb);
    } else {
      hapiAddCallback(manager->d2h_stream, cb);
    }
  }

  void sendGhosts() {
    if (my_iter > warmup_iters) comm_start_time = CkWallTimer();

    // Send ghosts to neighboring chares
    if (use_zerocopy) {
      if (!bounds[LEFT])
        thisProxy(x-1, y, z).recvGhostZC(my_iter, RIGHT, x_surf_count,
            send_ghosts_buf[LEFT]);
      if (!bounds[RIGHT])
        thisProxy(x+1, y, z).recvGhostZC(my_iter, LEFT, x_surf_count,
            send_ghosts_buf[RIGHT]);
      if (!bounds[TOP])
        thisProxy(x, y-1, z).recvGhostZC(my_iter, BOTTOM, y_surf_count,
            send_ghosts_buf[TOP]);
      if (!bounds[BOTTOM])
        thisProxy(x, y+1, z).recvGhostZC(my_iter, TOP, y_surf_count,
            send_ghosts_buf[BOTTOM]);
      if (!bounds[FRONT])
        thisProxy(x, y, z-1).recvGhostZC(my_iter, BACK, z_surf_count,
            send_ghosts_buf[FRONT]);
      if (!bounds[BACK])
        thisProxy(x, y, z+1).recvGhostZC(my_iter, FRONT, z_surf_count,
            send_ghosts_buf[BACK]);
    } else {
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
  }

  // This is the post entry method, the regular entry method is defined as a
  // SDAG entry method in the .ci file
  void recvGhostZC(int ref, int dir, int &count, DataType *&buf, CkDeviceBufferPost *devicePost) {
    CkAssert(dir >= 0 && dir < DIR_COUNT);
    buf = d_recv_ghosts[dir];
    if (dir == LEFT || dir == RIGHT) count = x_surf_count;
    else if (dir == TOP || dir == BOTTOM) count = y_surf_count;
    else if (dir == FRONT || dir == BACK) count = z_surf_count;
    //devicePost[0].cuda_stream = comm_stream;
  }

  void processGhostZC(int dir, int count, DataType* gh) {
    CkAssert(dir >= 0 && dir < DIR_COUNT);
    // XXX: d_recv_ghosts[dir] should be used instead of gh
    invokeUnpackingKernel(d_temperature, d_recv_ghosts[dir], dir, block_width, block_height,
        block_depth, manager->unpack_stream);
  }

  void processGhostReg(int dir, int size, DataType* gh) {
    CkAssert(dir >= 0 && dir < DIR_COUNT);
    DataType* h_ghost = h_ghosts[dir];
    DataType* d_ghost = d_ghosts[dir];

    // Copy ghost from host to device buffer
    memcpy(h_ghost, gh, size * sizeof(DataType));
    hapiCheck(cudaMemcpyAsync(d_ghost, h_ghost, size * sizeof(DataType),
          cudaMemcpyHostToDevice, manager->h2d_stream));

    // Enforce h2d -> unpack dependency
    hapiCheck(cudaEventRecord(h2d_unpack_events[dir], manager->h2d_stream));
    hapiCheck(cudaStreamWaitEvent(manager->unpack_stream, h2d_unpack_events[dir], 0));

    // Unpack
    invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height,
        block_depth, manager->unpack_stream);
  }

  void proceed() {
    if (my_iter <= warmup_iters) {
      contribute(CkCallback(CkReductionTarget(Main, warmupDone), main_proxy));
    } else {
      comm_time += CkWallTimer() - comm_start_time;
      if (my_iter < warmup_iters + n_iters) {
        thisProxy[thisIndex].run();
      } else {
        if (x == 0 && y == 0 && z == 0) {
          CkPrintf("Chare 0 comm time: %.3lf s (avg %.3lf ms, only valid without overdecomposition)\n",
            comm_time, comm_time / n_iters * 1e3);
        }
        contribute(CkCallback(CkReductionTarget(Main, allDone), main_proxy));
      }
    }
  }

  void print() {
    hapiCheck(cudaMemcpyAsync(h_temperature, d_temperature,
          sizeof(DataType) * (block_width+2)*(block_height+2)*(block_depth+2),
          cudaMemcpyDeviceToHost, manager->d2h_stream));
    cudaStreamSynchronize(manager->d2h_stream);

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
