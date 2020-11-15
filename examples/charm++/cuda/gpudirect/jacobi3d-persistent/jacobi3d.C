#include "hapi.h"
#include "jacobi3d.decl.h"
#include "jacobi3d.h"
#include <utility>
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
/* readonly */ size_t x_surf_size;
/* readonly */ size_t y_surf_size;
/* readonly */ size_t z_surf_size;
/* readonly */ int n_chares_x;
/* readonly */ int n_chares_y;
/* readonly */ int n_chares_z;
/* readonly */ int n_iters;
/* readonly */ int warmup_iters;
/* readonly */ bool use_zerocopy;
/* readonly */ bool use_persistent;
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

class PersistentMsg : public CMessage_PersistentMsg {
public:
  int dir;

  PersistentMsg(int dir_) : dir(dir_) {}
};

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
    use_persistent = false;
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
        case 'd':
          use_zerocopy = true;
          break;
        case 's':
          use_persistent = true;
          break;
        case 'p':
          print_elements = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -W [grid width] -H [grid height] -D [grid depth] "
              "-w [block width] -h [block height] -d [block depth] "
              "-i [iterations] -u [warmup] -z (use GPU zerocopy) "
              "-s (use persistent) -p (print blocks)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // Zerocopy and persistent cannot be used together
    if (use_zerocopy && use_persistent) {
      CkPrintf("Zerocopy and persistent cannot be used together!\n");
      CkExit(-1);
    }

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

    // Calculate surface sizes
    x_surf_size = block_height * block_depth * sizeof(DataType);
    y_surf_size = block_width * block_depth * sizeof(DataType);
    z_surf_size = block_width * block_height * sizeof(DataType);

    // Print configuration
    CkPrintf("\n[CUDA 3D Jacobi example]\n");
    CkPrintf("Grid: %d x %d x %d, Block: %d x %d x %d, Chares: %d x %d x %d, "
        "Iterations: %d, Warm-up: %d, Zerocopy: %d, Persistent: %d, Print: %d\n\n",
        grid_width, grid_height, grid_depth, block_width, block_height, block_depth,
        n_chares_x, n_chares_y, n_chares_z, n_iters, warmup_iters, use_zerocopy,
        use_persistent, print_elements);

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

    block_proxy.exchangeGhosts();
  }

  void warmupDone() {
    startIter();
  }

  void allDone() {
    double total_time = CkWallTimer() - start_time;
    CkPrintf("Total time: %.3lf s\nAverage iteration time: %.3lf us\n",
        total_time, (total_time / n_iters) * 1e6);

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

  std::vector<CkDevicePersistent> p_send_bufs;
  std::vector<CkDevicePersistent> p_recv_bufs;
  std::vector<CkDevicePersistent> p_neighbor_bufs;

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
    if (use_zerocopy || use_persistent) {
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
    } else {
      hapiCheck(cudaFreeHost(h_left_ghost));
      hapiCheck(cudaFreeHost(h_right_ghost));
      hapiCheck(cudaFreeHost(h_top_ghost));
      hapiCheck(cudaFreeHost(h_bottom_ghost));
      hapiCheck(cudaFreeHost(h_front_ghost));
      hapiCheck(cudaFreeHost(h_back_ghost));
      hapiCheck(cudaFree(d_left_ghost));
      hapiCheck(cudaFree(d_right_ghost));
      hapiCheck(cudaFree(d_top_ghost));
      hapiCheck(cudaFree(d_bottom_ghost));
      hapiCheck(cudaFree(d_front_ghost));
      hapiCheck(cudaFree(d_back_ghost));
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
    if (use_zerocopy || use_persistent) {
      hapiCheck(cudaMalloc((void**)&d_send_left_ghost, x_surf_size));
      hapiCheck(cudaMalloc((void**)&d_send_right_ghost, x_surf_size));
      hapiCheck(cudaMalloc((void**)&d_send_top_ghost, y_surf_size));
      hapiCheck(cudaMalloc((void**)&d_send_bottom_ghost, y_surf_size));
      hapiCheck(cudaMalloc((void**)&d_send_front_ghost, z_surf_size));
      hapiCheck(cudaMalloc((void**)&d_send_back_ghost, z_surf_size));
      hapiCheck(cudaMalloc((void**)&d_recv_left_ghost, x_surf_size));
      hapiCheck(cudaMalloc((void**)&d_recv_right_ghost, x_surf_size));
      hapiCheck(cudaMalloc((void**)&d_recv_top_ghost, y_surf_size));
      hapiCheck(cudaMalloc((void**)&d_recv_bottom_ghost, y_surf_size));
      hapiCheck(cudaMalloc((void**)&d_recv_front_ghost, z_surf_size));
      hapiCheck(cudaMalloc((void**)&d_recv_back_ghost, z_surf_size));
    } else {
      hapiCheck(cudaMallocHost((void**)&h_left_ghost, x_surf_size));
      hapiCheck(cudaMallocHost((void**)&h_right_ghost, x_surf_size));
      hapiCheck(cudaMallocHost((void**)&h_top_ghost, y_surf_size));
      hapiCheck(cudaMallocHost((void**)&h_bottom_ghost, y_surf_size));
      hapiCheck(cudaMallocHost((void**)&h_front_ghost, z_surf_size));
      hapiCheck(cudaMallocHost((void**)&h_back_ghost, z_surf_size));
      hapiCheck(cudaMalloc((void**)&d_left_ghost, x_surf_size));
      hapiCheck(cudaMalloc((void**)&d_right_ghost, x_surf_size));
      hapiCheck(cudaMalloc((void**)&d_top_ghost, y_surf_size));
      hapiCheck(cudaMalloc((void**)&d_bottom_ghost, y_surf_size));
      hapiCheck(cudaMalloc((void**)&d_front_ghost, z_surf_size));
      hapiCheck(cudaMalloc((void**)&d_back_ghost, z_surf_size));
    }

    // Create CUDA streams and events
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));

    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));

    if (use_persistent) {
      CkCallback send_cb = CkCallback(CkIndex_Block::sendGhostP(),
          thisProxy[thisIndex]);
      CkCallback recv_cb = CkCallback(CkIndex_Block::recvGhostP(nullptr),
          thisProxy[thisIndex]);

      p_send_bufs.reserve(DIR_COUNT);
      p_recv_bufs.reserve(DIR_COUNT);
      p_neighbor_bufs.resize(DIR_COUNT);

      // Create persistent buffers
      p_send_bufs.emplace_back(d_send_left_ghost,   x_surf_size, send_cb, comm_stream);
      p_send_bufs.emplace_back(d_send_right_ghost,  x_surf_size, send_cb, comm_stream);
      p_send_bufs.emplace_back(d_send_top_ghost,    y_surf_size, send_cb, comm_stream);
      p_send_bufs.emplace_back(d_send_bottom_ghost, y_surf_size, send_cb, comm_stream);
      p_send_bufs.emplace_back(d_send_front_ghost,  z_surf_size, send_cb, comm_stream);
      p_send_bufs.emplace_back(d_send_back_ghost,   z_surf_size, send_cb, comm_stream);
      p_recv_bufs.emplace_back(d_recv_left_ghost,   x_surf_size, recv_cb, comm_stream);
      p_recv_bufs.emplace_back(d_recv_right_ghost,  x_surf_size, recv_cb, comm_stream);
      p_recv_bufs.emplace_back(d_recv_top_ghost,    y_surf_size, recv_cb, comm_stream);
      p_recv_bufs.emplace_back(d_recv_bottom_ghost, y_surf_size, recv_cb, comm_stream);
      p_recv_bufs.emplace_back(d_recv_front_ghost,  z_surf_size, recv_cb, comm_stream);
      p_recv_bufs.emplace_back(d_recv_back_ghost,   z_surf_size, recv_cb, comm_stream);

      // Open persistent buffers that will be sent to neighbors
      for (int i = 0; i < DIR_COUNT; i++) {
        p_recv_bufs[i].open();
      }

      // Send persistent buffer info to neighbors
      if (!left_bound)   thisProxy(x-1, y, z).initRecv(RIGHT, p_recv_bufs[LEFT]);
      if (!right_bound)  thisProxy(x+1, y, z).initRecv(LEFT, p_recv_bufs[RIGHT]);
      if (!top_bound)    thisProxy(x, y-1, z).initRecv(BOTTOM, p_recv_bufs[TOP]);
      if (!bottom_bound) thisProxy(x, y+1, z).initRecv(TOP, p_recv_bufs[BOTTOM]);
      if (!front_bound)  thisProxy(x, y, z-1).initRecv(BACK, p_recv_bufs[FRONT]);
      if (!back_bound)   thisProxy(x, y, z+1).initRecv(FRONT, p_recv_bufs[BACK]);
    }

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

  void update() {
    // Operations in compute stream should only be executed when
    // operations in communication stream (transfers and unpacking) complete
    hapiCheck(cudaEventRecord(comm_event, comm_stream));
    hapiCheck(cudaStreamWaitEvent(compute_stream, comm_event, 0));

    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, block_width, block_height,
        block_depth, compute_stream);

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
  }

  void packGhosts() {
    if (use_persistent || use_zerocopy) {
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_send_left_ghost,
          d_send_right_ghost, d_send_top_ghost, d_send_bottom_ghost,
          d_send_front_ghost, d_send_back_ghost, left_bound, right_bound,
          top_bound, bottom_bound, front_bound, back_bound, block_width,
          block_height, block_depth, comm_stream);
    } else {
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_new_temperature, d_left_ghost, d_right_ghost,
          d_top_ghost, d_bottom_ghost, d_front_ghost, d_back_ghost, left_bound,
          right_bound, top_bound, bottom_bound, front_bound, back_bound,
          block_width, block_height, block_depth, comm_stream);

      // Transfer ghosts from device to host
      if (!left_bound)
        hapiCheck(cudaMemcpyAsync(h_left_ghost, d_left_ghost,
              x_surf_size, cudaMemcpyDeviceToHost, comm_stream));
      if (!right_bound)
        hapiCheck(cudaMemcpyAsync(h_right_ghost, d_right_ghost,
              x_surf_size, cudaMemcpyDeviceToHost, comm_stream));
      if (!top_bound)
        hapiCheck(cudaMemcpyAsync(h_top_ghost, d_top_ghost,
              y_surf_size, cudaMemcpyDeviceToHost, comm_stream));
      if (!bottom_bound)
        hapiCheck(cudaMemcpyAsync(h_bottom_ghost, d_bottom_ghost,
              y_surf_size, cudaMemcpyDeviceToHost, comm_stream));
      if (!front_bound)
        hapiCheck(cudaMemcpyAsync(h_front_ghost, d_front_ghost,
              z_surf_size, cudaMemcpyDeviceToHost, comm_stream));
      if (!back_bound)
        hapiCheck(cudaMemcpyAsync(h_back_ghost, d_back_ghost,
              z_surf_size, cudaMemcpyDeviceToHost, comm_stream));
    }

    if (use_persistent) {
      thisProxy[thisIndex].packGhostsDone();
    } else {
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
  }

  void sendGhosts() {
    // Send ghosts to neighboring chares
    // PersistentMsg is used to store the direction
    if (use_persistent) {
      PersistentMsg* msg;
      if (!left_bound) {
        msg = new PersistentMsg(RIGHT);
        p_neighbor_bufs[LEFT].set_msg(msg);
        p_neighbor_bufs[LEFT].cb.setRefNum(my_iter);
        p_send_bufs[LEFT].cb.setRefNum(my_iter);
        p_send_bufs[LEFT].put(p_neighbor_bufs[LEFT]);
      }
      if (!right_bound) {
        msg = new PersistentMsg(LEFT);
        p_neighbor_bufs[RIGHT].set_msg(msg);
        p_neighbor_bufs[RIGHT].cb.setRefNum(my_iter);
        p_send_bufs[RIGHT].cb.setRefNum(my_iter);
        p_send_bufs[RIGHT].put(p_neighbor_bufs[RIGHT]);
      }
      if (!top_bound) {
        msg = new PersistentMsg(BOTTOM);
        p_neighbor_bufs[TOP].set_msg(msg);
        p_neighbor_bufs[TOP].cb.setRefNum(my_iter);
        p_send_bufs[TOP].cb.setRefNum(my_iter);
        p_send_bufs[TOP].put(p_neighbor_bufs[TOP]);
      }
      if (!bottom_bound) {
        msg = new PersistentMsg(TOP);
        p_neighbor_bufs[BOTTOM].set_msg(msg);
        p_neighbor_bufs[BOTTOM].cb.setRefNum(my_iter);
        p_send_bufs[BOTTOM].cb.setRefNum(my_iter);
        p_send_bufs[BOTTOM].put(p_neighbor_bufs[BOTTOM]);
      }
      if (!front_bound) {
        msg = new PersistentMsg(BACK);
        p_neighbor_bufs[FRONT].set_msg(msg);
        p_neighbor_bufs[FRONT].cb.setRefNum(my_iter);
        p_send_bufs[FRONT].cb.setRefNum(my_iter);
        p_send_bufs[FRONT].put(p_neighbor_bufs[FRONT]);
        CkPrintf("Chare %d,%d,%d, iter %d: sendGhosts to dir %d\n", thisIndex.x, thisIndex.y, thisIndex.z, my_iter, FRONT);
      }
      if (!back_bound) {
        msg = new PersistentMsg(FRONT);
        p_neighbor_bufs[BACK].set_msg(msg);
        p_neighbor_bufs[BACK].cb.setRefNum(my_iter);
        p_send_bufs[BACK].cb.setRefNum(my_iter);
        p_send_bufs[BACK].put(p_neighbor_bufs[BACK]);
        CkPrintf("Chare %d,%d,%d, iter %d: sendGhosts to dir %d\n", thisIndex.x, thisIndex.y, thisIndex.z, my_iter, BACK);
      }
    } else if (use_zerocopy) {
      if (!left_bound)
        thisProxy(x-1, y, z).recvGhostZC(my_iter, RIGHT, block_height * block_depth,
            CkDeviceBuffer(d_send_left_ghost, comm_stream));
      if (!right_bound)
        thisProxy(x+1, y, z).recvGhostZC(my_iter, LEFT, block_height * block_depth,
            CkDeviceBuffer(d_send_right_ghost, comm_stream));
      if (!top_bound)
        thisProxy(x, y-1, z).recvGhostZC(my_iter, BOTTOM, block_width * block_depth,
            CkDeviceBuffer(d_send_top_ghost, comm_stream));
      if (!bottom_bound)
        thisProxy(x, y+1, z).recvGhostZC(my_iter, TOP, block_width * block_depth,
            CkDeviceBuffer(d_send_bottom_ghost, comm_stream));
      if (!front_bound)
        thisProxy(x, y, z-1).recvGhostZC(my_iter, BACK, block_width * block_height,
            CkDeviceBuffer(d_send_front_ghost, comm_stream));
      if (!back_bound)
        thisProxy(x, y, z+1).recvGhostZC(my_iter, FRONT, block_width * block_height,
            CkDeviceBuffer(d_send_back_ghost, comm_stream));
    } else {
      if (!left_bound)
        thisProxy(x-1, y, z).recvGhostReg(my_iter, RIGHT,
            block_height * block_depth, h_left_ghost);
      if (!right_bound)
        thisProxy(x+1, y, z).recvGhostReg(my_iter, LEFT,
            block_height * block_depth, h_right_ghost);
      if (!top_bound)
        thisProxy(x, y-1, z).recvGhostReg(my_iter, BOTTOM,
            block_width * block_depth, h_top_ghost);
      if (!bottom_bound)
        thisProxy(x, y+1, z).recvGhostReg(my_iter, TOP,
            block_width * block_depth, h_bottom_ghost);
      if (!front_bound)
        thisProxy(x, y, z-1).recvGhostReg(my_iter, BACK,
            block_width * block_height, h_front_ghost);
      if (!back_bound)
        thisProxy(x, y, z+1).recvGhostReg(my_iter, FRONT,
            block_width * block_height, h_back_ghost);
    }
  }

  // This is the post entry method, the regular entry method is defined as a
  // SDAG entry method in the .ci file
  void recvGhostZC(int ref, int dir, int &size, DataType *&buf, CkDeviceBufferPost *devicePost) {
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

  void processGhostZC(int dir, int size, DataType* gh) {
    invokeUnpackingKernel(d_temperature, gh, dir, block_width, block_height,
        block_depth, comm_stream);
  }

  void processGhostP(PersistentMsg* msg) {
    DataType* d_ghost = nullptr;
    int dir = msg->dir;
    switch (dir) {
      case LEFT:   d_ghost = d_recv_left_ghost;   break;
      case RIGHT:  d_ghost = d_recv_right_ghost;  break;
      case TOP:    d_ghost = d_recv_top_ghost;    break;
      case BOTTOM: d_ghost = d_recv_bottom_ghost; break;
      case FRONT:  d_ghost = d_recv_front_ghost;  break;
      case BACK:   d_ghost = d_recv_back_ghost;   break;
      default: CkAbort("Error: invalid direction");
    }

    invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height,
        block_depth, comm_stream);

    delete msg;
  }

  void processGhostReg(int dir, int size, DataType* gh) {
    DataType* h_ghost = nullptr; DataType* d_ghost = nullptr;
    switch (dir) {
      case LEFT:   h_ghost = h_left_ghost; d_ghost = d_left_ghost;     break;
      case RIGHT:  h_ghost = h_right_ghost; d_ghost = d_right_ghost;   break;
      case TOP:    h_ghost = h_top_ghost; d_ghost = d_top_ghost;       break;
      case BOTTOM: h_ghost = h_bottom_ghost; d_ghost = d_bottom_ghost; break;
      case FRONT:  h_ghost = h_front_ghost; d_ghost = d_front_ghost;   break;
      case BACK:   h_ghost = h_back_ghost; d_ghost = d_back_ghost;     break;
      default: CkAbort("Error: invalid direction");
    }

    memcpy(h_ghost, gh, size * sizeof(DataType));
    hapiCheck(cudaMemcpyAsync(d_ghost, h_ghost, size * sizeof(DataType),
          cudaMemcpyHostToDevice, comm_stream));
    invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height,
        block_depth, comm_stream);
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
