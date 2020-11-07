#include "hapi.h"
#include "hapi_nvtx.h"
#include "jacobi3d.decl.h"
#include "jacobi3d.h"
#include <utility>
#include <sstream>

#define COMM_ONLY 0
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
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_chares_z;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ bool sync_ver;
/* readonly */ bool use_zerocopy;
/* readonly */ bool print_elements;

extern void invokeInitKernel(DataType* d_temperature, int block_width,
    int block_height, int block_depth, cudaStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool left_bound, bool right_bound,
    bool top_bound, bool bottom_bound, bool front_bound, bool back_bound,
    cudaStream_t stream);
extern void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int block_depth, cudaStream_t stream);
extern void invokePackingKernels(DataType* d_temperature, DataType* d_left_ghost,
    DataType* d_right_ghost, DataType* d_top_ghost, DataType* d_bottom_ghost,
    DataType* d_front_ghost, DataType* d_back_ghost, bool left_bound,
    bool right_bound, bool top_bound, bool bottom_bound, bool front_bound,
    bool back_bound, int block_width, int block_height, int block_depth,
    cudaStream_t stream);
extern void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost,
    int dir, int block_width, int block_height, int block_depth,
    cudaStream_t stream);

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
    num_chares = CkNumPes();
    grid_width = 512;
    grid_height = 512;
    grid_depth = 512;
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
    bool dims[3] = {false, false, false};
    while ((c = getopt(m->argc, m->argv, "c:x:y:z:i:w:sdp")) != -1) {
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
        case 's':
          sync_ver = true;
          break;
        case 'd':
          use_zerocopy = true;
          break;
        case 'p':
          print_elements = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -W [grid width] -H [grid height] -D [grid depth] "
              "-w [block width] -h [block height] -d [block depth] "
              "-i [iterations] -u [warmup] -y (use sync version) "
              "-z (use GPU zerocopy) -p (print blocks)\n",
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

    // Print configuration
    CkPrintf("\n[CUDA 3D Jacobi example]\n");
    CkPrintf("Grid: %d x %d x %d, Block: %d x %d x %d, Chares: %d x %d x %d, "
        "Iterations: %d, Warm-up: %d, Bulk-synchronous: %d, Zerocopy: %d, Print: %d\n\n",
        grid_width, grid_height, grid_depth, block_width, block_height, block_depth,
        n_chares_x, n_chares_y, n_chares_z, n_iters, warmup_iters, sync_ver,
        use_zerocopy, print_elements);

    // Create blocks and start iteration
    block_proxy = CProxy_Block::ckNew(n_chares_x, n_chares_y, n_chares_z);
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
      block_proxy(0,0,0).print();
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
  int x, y, z;

  DataType* __restrict__ h_temperature;
  DataType* __restrict__ d_temperature;
  DataType* __restrict__ d_new_temperature;

  DataType* __restrict__ h_left_ghost;
  DataType* __restrict__ h_right_ghost;
  DataType* __restrict__ h_top_ghost;
  DataType* __restrict__ h_bottom_ghost;
  DataType* __restrict__ h_front_ghost;
  DataType* __restrict__ h_back_ghost;

  DataType* __restrict__ d_left_ghost;
  DataType* __restrict__ d_right_ghost;
  DataType* __restrict__ d_top_ghost;
  DataType* __restrict__ d_bottom_ghost;
  DataType* __restrict__ d_front_ghost;
  DataType* __restrict__ d_back_ghost;

  DataType* __restrict__ d_send_left_ghost;
  DataType* __restrict__ d_send_right_ghost;
  DataType* __restrict__ d_send_top_ghost;
  DataType* __restrict__ d_send_bottom_ghost;
  DataType* __restrict__ d_send_front_ghost;
  DataType* __restrict__ d_send_back_ghost;

  DataType* __restrict__ d_recv_left_ghost;
  DataType* __restrict__ d_recv_right_ghost;
  DataType* __restrict__ d_recv_top_ghost;
  DataType* __restrict__ d_recv_bottom_ghost;
  DataType* __restrict__ d_recv_front_ghost;
  DataType* __restrict__ d_recv_back_ghost;

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;

  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  bool left_bound, right_bound, top_bound, bottom_bound, front_bound, back_bound;

  Block() {}

  ~Block() {
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(cudaFree(d_temperature));
    hapiCheck(cudaFree(d_new_temperature));
    hapiCheck(cudaFreeHost(h_left_ghost));
    hapiCheck(cudaFreeHost(h_right_ghost));
    hapiCheck(cudaFreeHost(h_top_ghost));
    hapiCheck(cudaFreeHost(h_bottom_ghost));
    hapiCheck(cudaFreeHost(h_front_ghost));
    hapiCheck(cudaFreeHost(h_back_ghost));
    if (!use_zerocopy) {
      hapiCheck(cudaFree(d_left_ghost));
      hapiCheck(cudaFree(d_right_ghost));
      hapiCheck(cudaFree(d_top_ghost));
      hapiCheck(cudaFree(d_bottom_ghost));
      hapiCheck(cudaFree(d_front_ghost));
      hapiCheck(cudaFree(d_back_ghost));
    } else {
      hapiCheck(cudaFree(d_send_left_ghost));
      hapiCheck(cudaFree(d_send_right_ghost));
      hapiCheck(cudaFree(d_send_top_ghost));
      hapiCheck(cudaFree(d_send_bottom_ghost));
      hapiCheck(cudaFree(d_send_front_ghost));
      hapiCheck(cudaFree(d_send_back_ghost));
      hapiCheck(cudaFree(d_recv_left_ghost));
      hapiCheck(cudaFree(d_recv_right_ghost));
      hapiCheck(cudaFree(d_recv_top_ghost));
      hapiCheck(cudaFree(d_recv_bottom_ghost));
      hapiCheck(cudaFree(d_recv_front_ghost));
      hapiCheck(cudaFree(d_recv_back_ghost));
    }

    hapiCheck(cudaStreamDestroy(compute_stream));
    hapiCheck(cudaStreamDestroy(comm_stream));

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

    std::ostringstream os;
    os << "Init (" << std::to_string(x) << "," << std::to_string(y) << "," <<
      std::to_string(z) << ")";
    NVTXTracer(os.str(), NVTXColor::Turquoise);

    // Check bounds and set number of valid neighbors
    left_bound = right_bound = top_bound = bottom_bound = front_bound = back_bound = false;
    if (thisIndex.x == 0)
      left_bound = true;
    else
      neighbors++;
    if (thisIndex.x == n_chares_x-1)
      right_bound = true;
    else
      neighbors++;
    if (thisIndex.y == 0)
      top_bound = true;
    else
      neighbors++;
    if (thisIndex.y == n_chares_y-1)
      bottom_bound = true;
    else
      neighbors++;
    if (thisIndex.z == 0)
      front_bound = true;
    else
      neighbors++;
    if (thisIndex.z == n_chares_z-1)
      back_bound = true;
    else
      neighbors++;

    // Allocate memory and create CUDA entities
    hapiCheck(cudaMallocHost((void**)&h_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(cudaMalloc((void**)&d_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(cudaMalloc((void**)&d_new_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(cudaMallocHost((void**)&h_left_ghost, sizeof(DataType) * block_height * block_depth));
    hapiCheck(cudaMallocHost((void**)&h_right_ghost, sizeof(DataType) * block_height * block_depth));
    hapiCheck(cudaMallocHost((void**)&h_top_ghost, sizeof(DataType) * block_width * block_depth));
    hapiCheck(cudaMallocHost((void**)&h_bottom_ghost, sizeof(DataType) * block_width * block_depth));
    hapiCheck(cudaMallocHost((void**)&h_front_ghost, sizeof(DataType) * block_width * block_height));
    hapiCheck(cudaMallocHost((void**)&h_back_ghost, sizeof(DataType) * block_width * block_height));
    if (!use_zerocopy) {
      hapiCheck(cudaMalloc((void**)&d_left_ghost, sizeof(DataType) * block_height * block_depth));
      hapiCheck(cudaMalloc((void**)&d_right_ghost, sizeof(DataType) * block_height * block_depth));
      hapiCheck(cudaMalloc((void**)&d_top_ghost, sizeof(DataType) * block_width * block_depth));
      hapiCheck(cudaMalloc((void**)&d_bottom_ghost, sizeof(DataType) * block_width * block_depth));
      hapiCheck(cudaMalloc((void**)&d_front_ghost, sizeof(DataType) * block_width * block_height));
      hapiCheck(cudaMalloc((void**)&d_back_ghost, sizeof(DataType) * block_width * block_height));
    } else {
      hapiCheck(cudaMalloc((void**)&d_send_left_ghost, sizeof(DataType) * block_height * block_depth));
      hapiCheck(cudaMalloc((void**)&d_send_right_ghost, sizeof(DataType) * block_height * block_depth));
      hapiCheck(cudaMalloc((void**)&d_send_top_ghost, sizeof(DataType) * block_width * block_depth));
      hapiCheck(cudaMalloc((void**)&d_send_bottom_ghost, sizeof(DataType) * block_width * block_depth));
      hapiCheck(cudaMalloc((void**)&d_send_front_ghost, sizeof(DataType) * block_width * block_height));
      hapiCheck(cudaMalloc((void**)&d_send_back_ghost, sizeof(DataType) * block_width * block_height));
      hapiCheck(cudaMalloc((void**)&d_recv_left_ghost, sizeof(DataType) * block_height * block_depth));
      hapiCheck(cudaMalloc((void**)&d_recv_right_ghost, sizeof(DataType) * block_height * block_depth));
      hapiCheck(cudaMalloc((void**)&d_recv_top_ghost, sizeof(DataType) * block_width * block_depth));
      hapiCheck(cudaMalloc((void**)&d_recv_bottom_ghost, sizeof(DataType) * block_width * block_depth));
      hapiCheck(cudaMalloc((void**)&d_recv_front_ghost, sizeof(DataType) * block_width * block_height));
      hapiCheck(cudaMalloc((void**)&d_recv_back_ghost, sizeof(DataType) * block_width * block_height));
    }

    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));

    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_width, block_height, block_depth, compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, block_depth, compute_stream);

    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, block_width, block_height, block_depth,
        left_bound, right_bound, top_bound, bottom_bound, front_bound, back_bound,
        compute_stream);
    invokeBoundaryKernels(d_new_temperature, block_width, block_height, block_depth,
        left_bound, right_bound, top_bound, bottom_bound, front_bound, back_bound,
        compute_stream);

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
    os << "update (" << std::to_string(x) << "," << std::to_string(y) << "," <<
      std::to_string(z) << ")";
    NVTXTracer(os.str(), NVTXColor::WetAsphalt);

    // Operations in compute stream should only be executed when
    // operations in communication stream (transfers and unpacking) complete
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

#if !COMM_ONLY
    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height,
        block_depth, compute_stream);
#endif

    // Operations in communication stream (packing and transfers) should
    // only be executed when operations in compute stream complete
    hapiCheck(cudaEventRecord(compute_event, compute_stream));
    hapiCheck(cudaStreamWaitEvent(comm_stream, compute_event, 0));

    // Copy final temperature data back to host
    if (print_elements && (my_iter == warmup_iters + n_iters)) {
      hapiCheck(cudaMemcpyAsync(h_temperature, d_new_temperature,
            sizeof(DataType) * (block_width+2)*(block_height+2)*(block_depth+2),
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
    os << "packGhosts (" << std::to_string(x) << "," << std::to_string(y) <<
      "," << std::to_string(z) << ")";
    NVTXTracer(os.str(), NVTXColor::Emerald);

    if (use_zerocopy) {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_send_left_ghost,
          d_send_right_ghost, d_send_top_ghost, d_send_bottom_ghost,
          d_send_front_ghost, d_send_back_ghost, left_bound, right_bound,
          top_bound, bottom_bound, front_bound, back_bound, block_width,
          block_height, block_depth, comm_stream);
#endif
    } else {
#if !COMM_ONLY
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_left_ghost, d_right_ghost,
          d_top_ghost, d_bottom_ghost, d_front_ghost, d_back_ghost, left_bound,
          right_bound, top_bound, bottom_bound, front_bound, back_bound,
          block_width, block_height, block_depth, comm_stream);
#endif

      // Transfer ghosts from device to host
      if (!left_bound)
        hapiCheck(cudaMemcpyAsync(h_left_ghost, d_left_ghost,
              block_height * block_depth * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
      if (!right_bound)
        hapiCheck(cudaMemcpyAsync(h_right_ghost, d_right_ghost,
              block_height * block_depth * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
      if (!top_bound)
        hapiCheck(cudaMemcpyAsync(h_top_ghost, d_top_ghost,
              block_width * block_depth * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
      if (!bottom_bound)
        hapiCheck(cudaMemcpyAsync(h_bottom_ghost, d_bottom_ghost,
              block_width * block_depth * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
      if (!front_bound)
        hapiCheck(cudaMemcpyAsync(h_front_ghost, d_front_ghost,
              block_width * block_height * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
      if (!back_bound)
        hapiCheck(cudaMemcpyAsync(h_back_ghost, d_back_ghost,
              block_width * block_height * sizeof(DataType),
              cudaMemcpyDeviceToHost, comm_stream));
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
    std::ostringstream os;
    os << "sendGhosts (" << std::to_string(x) << "," << std::to_string(y) <<
      "," << std::to_string(z) << ")";
    NVTXTracer(os.str(), NVTXColor::PeterRiver);

    // Send ghosts to neighboring chares
    if (use_zerocopy) {
      if (!left_bound)
        thisProxy(x-1, y, z).receiveGhostsZC(my_iter, RIGHT, block_height * block_depth,
            CkDeviceBuffer(d_send_left_ghost, comm_stream));
      if (!right_bound)
        thisProxy(x+1, y, z).receiveGhostsZC(my_iter, LEFT, block_height * block_depth,
            CkDeviceBuffer(d_send_right_ghost, comm_stream));
      if (!top_bound)
        thisProxy(x, y-1, z).receiveGhostsZC(my_iter, BOTTOM, block_width * block_depth,
            CkDeviceBuffer(d_send_top_ghost, comm_stream));
      if (!bottom_bound)
        thisProxy(x, y+1, z).receiveGhostsZC(my_iter, TOP, block_width * block_depth,
            CkDeviceBuffer(d_send_bottom_ghost, comm_stream));
      if (!front_bound)
        thisProxy(x, y, z-1).receiveGhostsZC(my_iter, BACK, block_width * block_height,
            CkDeviceBuffer(d_send_front_ghost, comm_stream));
      if (!back_bound)
        thisProxy(x, y, z+1).receiveGhostsZC(my_iter, FRONT, block_width * block_height,
            CkDeviceBuffer(d_send_back_ghost, comm_stream));
    } else {
      if (!left_bound)
        thisProxy(x-1, y, z).receiveGhostsReg(my_iter, RIGHT,
            block_height * block_depth, h_left_ghost);
      if (!right_bound)
        thisProxy(x+1, y, z).receiveGhostsReg(my_iter, LEFT,
            block_height * block_depth, h_right_ghost);
      if (!top_bound)
        thisProxy(x, y-1, z).receiveGhostsReg(my_iter, BOTTOM,
            block_width * block_depth, h_top_ghost);
      if (!bottom_bound)
        thisProxy(x, y+1, z).receiveGhostsReg(my_iter, TOP,
            block_width * block_depth, h_bottom_ghost);
      if (!front_bound)
        thisProxy(x, y, z-1).receiveGhostsReg(my_iter, BACK,
            block_width * block_height, h_front_ghost);
      if (!back_bound)
        thisProxy(x, y, z+1).receiveGhostsReg(my_iter, FRONT,
            block_width * block_height, h_back_ghost);
    }
  }

  // This is the post entry method, the regular entry method is defined as a
  // SDAG entry method in the .ci file
  void receiveGhostsZC(int ref, int dir, int &size, DataType *&buf, CkDeviceBufferPost *devicePost) {
    switch (dir) {
      case LEFT:   buf = d_recv_left_ghost;   break;
      case RIGHT:  buf = d_recv_right_ghost;  break;
      case TOP:    buf = d_recv_top_ghost;    break;
      case BOTTOM: buf = d_recv_bottom_ghost; break;
      case FRONT:  buf = d_recv_front_ghost;  break;
      case BACK:   buf = d_recv_back_ghost;   break;
      default: CkAbort("Error: invalid direction");
    }
    devicePost[0].cuda_stream = comm_stream;
  }

  void processGhostsZC(int dir, int size, DataType* gh) {
    std::ostringstream os;
    os << "processGhostsZC (" << std::to_string(x) << "," << std::to_string(y) <<
      "," << std::to_string(z) << ")";
    NVTXTracer(os.str(), NVTXColor::Amethyst);

#if !COMM_ONLY
    invokeUnpackingKernel(d_temperature, gh, dir, block_width, block_height,
        block_depth, comm_stream);
#endif
  }

  void processGhostsReg(int dir, int size, DataType* gh) {
    std::ostringstream os;
    os << "processGhostsReg (" << std::to_string(x) << "," << std::to_string(y) <<
      "," << std::to_string(z) << ")";
    NVTXTracer(os.str(), NVTXColor::Amethyst);

    DataType* h_ghost = nullptr; DataType* d_ghost = nullptr;
    switch (dir) {
      case LEFT:   h_ghost = h_left_ghost; d_ghost = d_left_ghost;     break;
      case RIGHT:  h_ghost = h_right_ghost; d_ghost = d_right_ghost;   break;
      case TOP:    h_ghost = h_top_ghost; d_ghost = d_top_ghost;       break;
      case BOTTOM: h_ghost = h_bottom_ghost; d_ghost = d_bottom_ghost; break;
      case FRONT:  h_ghost = h_front_ghost; d_ghost = d_front_ghost;   break;
      case BACK:   h_ghost = h_back_ghost; d_ghost = d_back_ghost; break;
      default: CkAbort("Error: invalid direction");
    }

    memcpy(h_ghost, gh, size * sizeof(DataType));
    hapiCheck(cudaMemcpyAsync(d_ghost, h_ghost, size * sizeof(DataType),
          cudaMemcpyHostToDevice, comm_stream));
#if !COMM_ONLY
    invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height,
        block_depth, comm_stream);
#endif
  }

  void print() {
    CkPrintf("[%d,%d,%d]\n", thisIndex.x, thisIndex.y, thisIndex.z);
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

    if (thisIndex.x == n_chares_x-1 && thisIndex.y == n_chares_y-1 &&
        thisIndex.z == n_chares_z-1) {
      main_proxy.printDone();
    } else {
      if (thisIndex.x == n_chares_x-1 && thisIndex.y == n_chares_y-1) {
        thisProxy(0, 0, thisIndex.z+1).print();
      } else {
        if (thisIndex.x == n_chares_x-1) {
          thisProxy(0, thisIndex.y+1, thisIndex.z).print();
        } else {
          thisProxy(thisIndex.x+1, thisIndex.y, thisIndex.z).print();
        }
      }
    }
  }
};

#include "jacobi3d.def.h"
