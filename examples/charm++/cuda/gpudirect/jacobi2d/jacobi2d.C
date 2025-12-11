#include "hapi.h"
#include "hapi_nvtx.h"
#include "jacobi2d.decl.h"
#include "jacobi2d.h"
#include <utility>
#include <sstream>
#include <Kokkos_Core.hpp>

#define COMM_ONLY 0
#define hapi_SYNC 0

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
    Kokkos::initialize();
    // Set default values
    main_proxy = thisProxy;
    grid_width = 16384;
    grid_height = 16384;
    block_width = 4096;
    block_height = 4096;
    n_iters = 100;
    warmup_iters = 10;
    use_zerocopy = false;
    print_elements = false;
    sync_ver = false;
    my_iter = 0;

    // Initialize aggregate timers
    update_agg_time = 0.0;
    comm_agg_time = 0.0;

    // Process arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "W:H:w:h:i:u:yzp")) != -1) {
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
    n_chares_x = grid_width / block_width;
    n_chares_y = grid_height / block_height;

    // Print configuration
    CkPrintf("\n[hapi 2D Jacobi example]\n");
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

using ExecSpace = Kokkos::DefaultExecutionSpace;
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;
using MDRangePolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>;
using HostMemSpace = Kokkos::HostSpace;
using DeviceMemSpace = ExecSpace::memory_space;

void invokeInitKernel(Kokkos::View<DataType*, DeviceMemSpace> d_temperature, int block_width, int block_height, hapiStream_t stream) {
  Kokkos::parallel_for(
      "invokeInitKernel",
      MDRangePolicy(ExecSpace(stream),
          {0, 0}, {block_width + 2, block_height + 2}),
      KOKKOS_LAMBDA(int i, int j) { d_temperature(IDX(i, j)) = 0; });

  hapiCheck(cudaPeekAtLastError());
}

void invokeBoundaryKernels(Kokkos::View<DataType*, DeviceMemSpace> d_temperature, int block_width,
    int block_height, bool left_bound, bool right_bound, bool top_bound,
    bool bottom_bound, hapiStream_t stream) {
  if (left_bound) {
    Kokkos::parallel_for(
        "leftBoundaryKernel",
        RangePolicy(ExecSpace(stream), 0, block_height), KOKKOS_LAMBDA(int i) { d_temperature(IDX(0, 1 + i)) = 1; });
  }
  if (right_bound) {
    Kokkos::parallel_for(
        "rightBoundaryKernel",
        RangePolicy(ExecSpace(stream), 0, block_height), KOKKOS_LAMBDA(int i) { d_temperature(IDX(block_width + 1, 1 + i)) = 1; });
  }

  if (top_bound) {
    Kokkos::parallel_for(
        "topBoundaryKernel",
        RangePolicy(ExecSpace(stream), 0, block_width), KOKKOS_LAMBDA(int i) { d_temperature(IDX(1 + i, 0)) = 1; });
  }
  if (bottom_bound) {
    Kokkos::parallel_for(
        "bottomBoundaryKernel",
        RangePolicy(ExecSpace(stream), 0, block_width), KOKKOS_LAMBDA(int i) { d_temperature(IDX(1 + i, block_height + 1)) = 1; });
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeJacobiKernel(Kokkos::View<DataType*, DeviceMemSpace> d_temperature, Kokkos::View<DataType*, DeviceMemSpace> d_new_temperature,
    int block_width, int block_height, hapiStream_t stream) {
  Kokkos::parallel_for(
      "invokeJacobiKernel",
      MDRangePolicy(ExecSpace(stream),
          {1, 1}, {block_width + 1, block_height + 1}),
      KOKKOS_LAMBDA(int i, int j) {
        d_new_temperature(IDX(i, j)) = (d_temperature(IDX(i - 1, j)) + d_temperature(IDX(i + 1, j)) +
          d_temperature(IDX(i, j - 1)) + d_temperature(IDX(i, j + 1)) + d_temperature(IDX(i, j))) *
          0.2;
      });
  
  hapiCheck(cudaPeekAtLastError());
}

void invokePackingKernels(Kokkos::View<DataType*, DeviceMemSpace> d_temperature, Kokkos::View<DataType*, 
                          DeviceMemSpace> d_left_ghost, Kokkos::View<DataType*, DeviceMemSpace> d_right_ghost, 
                          bool left_bound, bool right_bound, int block_width, int block_height, hapiStream_t stream) {
  if(!left_bound) {
    Kokkos::parallel_for(
        "leftPackingKernel",
        RangePolicy(ExecSpace(stream), 0, block_height),
        KOKKOS_LAMBDA(int j) {
          d_left_ghost(j) = d_temperature(IDX(1, 1 + j));
        });
  }
  if(!right_bound) {
    Kokkos::parallel_for(
        "rightPackingKernel",
        RangePolicy(ExecSpace(stream), 0, block_height),
        KOKKOS_LAMBDA(int j) {
          d_right_ghost(j) = d_temperature(IDX(block_width, 1 + j));
        });
  }
  hapiCheck(cudaPeekAtLastError());
}

void invokeUnpackingKernel(Kokkos::View<DataType*, DeviceMemSpace> d_temperature, Kokkos::View<DataType*, DeviceMemSpace> d_ghost,
                           bool is_left, int block_width, int block_height, hapiStream_t stream) {
  if (is_left) {
    Kokkos::parallel_for(
        "leftUnpackingKernel",
        RangePolicy(ExecSpace(stream), 0, block_height),
        KOKKOS_LAMBDA(int j) {
          d_temperature(IDX(0, 1 + j)) = d_ghost(j);
        });
  } else {
    Kokkos::parallel_for(
        "rightUnpackingKernel",
        RangePolicy(ExecSpace(stream), 0, block_height),
        KOKKOS_LAMBDA(int j) {
          d_temperature(IDX(block_width + 1, 1 + j)) = d_ghost(j);
        });
  }
  hapiCheck(cudaPeekAtLastError());
}

class Block : public CBase_Block {
  Block_SDAG_CODE

 public:
  int my_iter;
  int neighbors;
  int remote_count;
  int x, y;

  Kokkos::View<DataType*, HostMemSpace> h_temperature;
  Kokkos::View<DataType*, HostMemSpace> h_left_ghost;
  Kokkos::View<DataType*, HostMemSpace> h_right_ghost;
  Kokkos::View<DataType*, HostMemSpace> h_top_ghost;
  Kokkos::View<DataType*, HostMemSpace> h_bottom_ghost;

  Kokkos::View<DataType*, DeviceMemSpace> d_temperature;
  Kokkos::View<DataType*, DeviceMemSpace> d_new_temperature;
  Kokkos::View<DataType*, DeviceMemSpace> d_left_ghost;
  Kokkos::View<DataType*, DeviceMemSpace> d_right_ghost;
  Kokkos::View<DataType*, DeviceMemSpace> d_send_left_ghost;
  Kokkos::View<DataType*, DeviceMemSpace> d_send_right_ghost;
  Kokkos::View<DataType*, DeviceMemSpace> d_send_top_ghost;
  Kokkos::View<DataType*, DeviceMemSpace> d_send_bottom_ghost;
  Kokkos::View<DataType*, DeviceMemSpace> d_recv_left_ghost;
  Kokkos::View<DataType*, DeviceMemSpace> d_recv_right_ghost;

  hapiStream_t compute_stream;
  hapiStream_t comm_stream;

  hapiEvent_t compute_event;
  hapiEvent_t comm_event;

  bool left_bound, right_bound, top_bound, bottom_bound;

  Block() {}

  ~Block() {
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

    h_temperature =
        Kokkos::View<DataType*, HostMemSpace>(
            "h_temperature",
            (block_width + 2) * (block_height + 2));

    h_left_ghost = Kokkos::View<DataType*, HostMemSpace>("h_left_ghost", block_height);

    h_right_ghost = Kokkos::View<DataType*, HostMemSpace>("h_right_ghost", block_height);

    h_top_ghost = Kokkos::View<DataType*, HostMemSpace>("h_top_ghost", block_width);

    h_bottom_ghost = Kokkos::View<DataType*, HostMemSpace>("h_bottom_ghost", block_width);


    // ---- Device allocations (GPU/ExecSpace device) ----
    d_temperature =
        Kokkos::View<DataType*, DeviceMemSpace>(
            "d_temperature",
            (block_width + 2) * (block_height + 2));

    d_new_temperature =
        Kokkos::View<DataType*, DeviceMemSpace>(
            "d_new_temperature",
            (block_width + 2) * (block_height + 2));


    // ---- Conditional ghost-layer allocations ----
    if (!use_zerocopy)
    {
        d_left_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_left_ghost", block_height);

        d_right_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_right_ghost", block_height);
    }
    else
    {
        d_send_left_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_send_left_ghost", block_height);

        d_send_right_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_send_right_ghost", block_height);

        d_send_top_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_send_top_ghost", block_width);

        d_send_bottom_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_send_bottom_ghost", block_width);

        d_recv_left_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_recv_left_ghost", block_height);

        d_recv_right_ghost =
            Kokkos::View<DataType*, DeviceMemSpace>("d_recv_right_ghost", block_height);
    }

    hapiCheck(hapiStreamCreateWithPriority(&compute_stream, hapiStreamDefault, 0));
    hapiCheck(hapiStreamCreateWithPriority(&comm_stream, hapiStreamDefault, -1));

    hapiCheck(hapiEventCreateWithFlags(&compute_event, hapiEventDisableTiming));
    hapiCheck(hapiEventCreateWithFlags(&comm_event, hapiEventDisableTiming));

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_width, block_height, compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, compute_stream);

    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);
    invokeBoundaryKernels(d_new_temperature, block_width, block_height, left_bound,
        right_bound, top_bound, bottom_bound, compute_stream);

#if hapi_SYNC
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

  void update() {
    std::ostringstream os;
    os << "update (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::WetAsphalt);

    // Operations in compute stream should only be executed when
    // operations in communication stream (transfers and unpacking) complete
    hapiCheck(hapiEventRecord(comm_event, comm_stream));
    hapiCheck(hapiStreamWaitEvent(compute_stream, comm_event, 0));

#if !COMM_ONLY
    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height,
        compute_stream);
#endif

    // Operations in communication stream (packing and transfers) should
    // only be executed when operations in compute stream complete
    hapiCheck(hapiEventRecord(compute_event, compute_stream));
    hapiCheck(hapiStreamWaitEvent(comm_stream, compute_event, 0));

    // Copy final temperature data back to host
    if (print_elements && (my_iter == warmup_iters + n_iters)) {
      Kokkos::deep_copy(ExecSpace(comm_stream), h_temperature, d_new_temperature);
    }

    if (sync_ver) {
#if hapi_SYNC
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
    std::ostringstream os;
    os << "packGhosts (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Emerald);

    if (use_zerocopy) {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_send_left_ghost, d_send_right_ghost,
          left_bound, right_bound, block_width, block_height, comm_stream);
#endif

      const size_t pitch = block_width + 2;

      if(!top_bound) {
        auto top_src = Kokkos::subview(d_new_temperature,
                                      std::make_pair(pitch + 1,
                                                      pitch + 1 + block_width));
        Kokkos::deep_copy(ExecSpace(comm_stream), d_send_top_ghost, top_src);
      }

      if (!bottom_bound) {
        auto bottom_src = Kokkos::subview(d_new_temperature,
                                        std::make_pair(pitch * block_height + 1,
                                                      pitch * block_height + 1 + block_width));
        Kokkos::deep_copy(ExecSpace(comm_stream), d_send_bottom_ghost, bottom_src);
      }
    } else {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_left_ghost, d_right_ghost,
          left_bound, right_bound, block_width, block_height, comm_stream);
#endif

      size_t pitch = block_width + 2;

      // Device → Host: left ghost
      if (!left_bound) {
          Kokkos::deep_copy(ExecSpace(comm_stream), h_left_ghost, d_left_ghost);
      }

      // Device → Host: right ghost
      if (!right_bound) {
          Kokkos::deep_copy(ExecSpace(comm_stream), h_right_ghost, d_right_ghost);
      }

      // Device → Host: top ghost
      if (!top_bound) {
          auto top_src = Kokkos::subview(
              d_new_temperature,
              std::make_pair(pitch + 1,
                            pitch + 1 + block_width)
          );
          Kokkos::deep_copy(ExecSpace(comm_stream), h_top_ghost, top_src);
      }

      // Device → Host: bottom ghost
      if (!bottom_bound) {
          auto bottom_src = Kokkos::subview(
              d_new_temperature,
              std::make_pair(pitch * block_height + 1,
                            pitch * block_height + 1 + block_width)
          );
          Kokkos::deep_copy(ExecSpace(comm_stream), h_bottom_ghost, bottom_src);
      }
    }

#if hapi_SYNC
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
    std::ostringstream os;
    os << "sendGhosts (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::PeterRiver);

    // Send ghosts to neighboring chares
    if (use_zerocopy) {
      if (!left_bound) 
        thisProxy(x - 1, y).receiveGhostsZC(my_iter, RIGHT, block_height,
            CkDeviceBuffer(d_send_left_ghost.data(), comm_stream));
      if (!right_bound)
        thisProxy(x + 1, y).receiveGhostsZC(my_iter, LEFT, block_height,
            CkDeviceBuffer(d_send_right_ghost.data(), comm_stream));
      if (!top_bound)
        thisProxy(x, y - 1).receiveGhostsZC(my_iter, BOTTOM, block_width,
            CkDeviceBuffer(d_send_top_ghost.data(), comm_stream));
      if (!bottom_bound)
        thisProxy(x, y + 1).receiveGhostsZC(my_iter, TOP, block_width,
            CkDeviceBuffer(d_send_bottom_ghost.data(), comm_stream));
    } else {
      if (!left_bound)
        thisProxy(x - 1, y).receiveGhostsReg(my_iter, RIGHT, block_height, h_left_ghost.data());
      if (!right_bound)
        thisProxy(x + 1, y).receiveGhostsReg(my_iter, LEFT, block_height, h_right_ghost.data());
      if (!top_bound)
        thisProxy(x, y - 1).receiveGhostsReg(my_iter, BOTTOM, block_width, h_top_ghost.data());
      if (!bottom_bound)
        thisProxy(x, y + 1).receiveGhostsReg(my_iter, TOP, block_width, h_bottom_ghost.data());
    }
  }

  // This is the post entry method, the regular entry method is defined as a
  // SDAG entry method in the .ci file
  void receiveGhostsZC(int ref, int dir, int &size, DataType *&buf, CkDeviceBufferPost *devicePost) {
    switch (dir) {
      case LEFT:
        buf = d_recv_left_ghost.data();
        break;
      case RIGHT:
        buf = d_recv_right_ghost.data();
        break;
      case TOP:
        buf = d_temperature.data() + 1;
        break;
      case BOTTOM:
        buf = d_temperature.data() + (block_width + 2) * (block_height + 1) + 1;
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
      case LEFT:
        invokeUnpackingKernel(d_temperature, d_recv_left_ghost, true, block_width,
            block_height, comm_stream);
        break;
      case RIGHT:
        invokeUnpackingKernel(d_temperature, d_recv_right_ghost, false, block_width,
            block_height, comm_stream);
        break;
      case TOP:
      case BOTTOM:
        break;
      default:
        CkAbort("Error: invalid direction");
    }
  }

  void processGhostsReg(int dir, int size, DataType* gh) {
    std::ostringstream os;
    os << "processGhostsReg (" << std::to_string(x) << "," << std::to_string(y) << ")";
    NVTXTracer(os.str(), NVTXColor::Amethyst);

    switch (dir) {
      case LEFT:
        memcpy(h_left_ghost.data(), gh, size * sizeof(DataType));
        Kokkos::deep_copy(ExecSpace(comm_stream), d_left_ghost, h_left_ghost);
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_left_ghost, true, block_width,
            block_height, comm_stream);
#endif
        break;
      case RIGHT:
        memcpy(h_right_ghost.data(), gh, size * sizeof(DataType));
        Kokkos::deep_copy(ExecSpace(comm_stream), d_right_ghost, h_right_ghost);
#if !COMM_ONLY
        invokeUnpackingKernel(d_temperature, d_right_ghost, false, block_width,
            block_height, comm_stream);
#endif
        break;
      case TOP: {
        memcpy(h_top_ghost.data(), gh, size * sizeof(DataType));
          auto dst = Kokkos::subview(
              d_temperature,
              std::make_pair(1, 1 + block_width)   // d_temperature + 1
          );

          Kokkos::deep_copy(ExecSpace(comm_stream), dst, h_top_ghost);
      } break;
      case BOTTOM: {
        memcpy(h_bottom_ghost.data(), gh, size * sizeof(DataType));
        size_t pitch = block_width + 2;
        size_t start = pitch * (block_height + 1) + 1;

        auto dst = Kokkos::subview(
            d_temperature,
            std::make_pair(start, start + block_width)
        );

        Kokkos::deep_copy(ExecSpace(comm_stream), dst, h_bottom_ghost);
      }
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
