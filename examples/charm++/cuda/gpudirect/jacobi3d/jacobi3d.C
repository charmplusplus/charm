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
/* readonly */ bool use_persistent;
/* readonly */ bool use_channels;
/* readonly */ bool print_elements;
/* readonly */ bool verbose;

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

// The source callback of a send: re-issued to the block as sendDone(iter) with
// the iteration as the message's reference number, which the SDAG `when`
// matches. (A parameterless entry reached through a callback loses the
// reference number in the generated closure.)
struct SendDoneCtx { CkArrayID aid; CkArrayIndex3D idx; int iter; };
static void sendDoneFn(void* param, void* msg) {
  SendDoneCtx* c = (SendDoneCtx*)param;
  CkDataMsg* m = CkDataMsg::buildNew((int)sizeof(int), &c->iter);
  CkSetRefNum(m, (CMK_REFNUM_TYPE)c->iter);
  CProxy_Block(c->aid)(c->idx.index[0], c->idx.index[1], c->idx.index[2]).sendDone(m);
  if (msg) CkFreeMsg(msg);
  delete c;
}

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
    use_channels = false;
    print_elements = false;
    verbose = false;
    my_iter = 0;

    // Process arguments
    int c;
    bool dims[3] = {false, false, false};
    while ((c = getopt(m->argc, m->argv, "c:x:y:z:i:w:dskpv")) != -1) {
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
        case 'k':
          use_channels = true;
          break;
        case 'p':
          print_elements = true;
          break;
        case 'v':
          verbose = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -x [grid width] -y [grid height] -z [grid depth] "
              "-c [number of chares] -i [iterations] -w [warmup iterations] "
              "-d (use GPU zerocopy) -s (use persistent) -k (use device channels) -p (print blocks)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // Zerocopy and persistent cannot be used together
    if ((int)use_zerocopy + (int)use_persistent + (int)use_channels > 1) {
      CkPrintf("Zerocopy, persistent and channels are exclusive!\n");
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
        "Iterations: %d, Warm-up: %d, Zerocopy: %d, Persistent: %d, Channels: %d, Print: %d\n\n",
        grid_width, grid_height, grid_depth, block_width, block_height, block_depth,
        n_chares_x, n_chares_y, n_chares_z, n_iters, warmup_iters, use_zerocopy,
        use_persistent, use_channels, print_elements);

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

    block_proxy.run();
  }

  void warmupDone() {
    startIter();
  }

  void allDone(double checksum) {
    double total_time = CkWallTimer() - start_time;
    CkPrintf("Total time: %.3lf s\nAverage iteration time: %.3lf us\n",
        total_time, (total_time / n_iters) * 1e6);
    // The sum of every block's interior after the last iteration: identical
    // across the transfer modes when the exchange is correct (same blocks,
    // same reduction tree).
    CkPrintf("Checksum: %.12e\n", checksum);
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

  DataType* h_temperature;
  DataType* d_temperature;
  DataType* d_new_temperature;

  DataType* h_ghosts[DIR_COUNT];
  DataType* d_ghosts[DIR_COUNT];
  DataType* d_send_ghosts[DIR_COUNT];
  DataType* d_recv_ghosts[DIR_COUNT];
  // -k: a second set of ghost buffers and the channels on them, by iteration parity.
  DataType* d_send_ghosts_b[DIR_COUNT];
  DataType* d_recv_ghosts_b[DIR_COUNT];
  CkDeviceChannel ch_recv[2][DIR_COUNT];   // mine: a neighbour's face lands here
  CkDeviceChannel ch_send[2][DIR_COUNT];   // the neighbours': my face goes there
  DataType* sendBuf(int p, int dir) { return p ? d_send_ghosts_b[dir] : d_send_ghosts[dir]; }
  DataType* recvBuf(int p, int dir) { return p ? d_recv_ghosts_b[dir] : d_recv_ghosts[dir]; }

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;

  cudaEvent_t compute_event;
  cudaEvent_t comm_event;

  bool bounds[DIR_COUNT];

  Block() {}

  ~Block() {
    hapiCheck(cudaFreeHost(h_temperature));
    hapiCheck(hapiFree(d_temperature));
    hapiCheck(hapiFree(d_new_temperature));
    if (use_zerocopy || use_persistent || use_channels) {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(hapiFree(d_send_ghosts[i]));
        hapiCheck(hapiFree(d_recv_ghosts[i]));
        if (use_channels) hapiCheck(hapiFree(d_send_ghosts_b[i]));
        if (use_channels || use_zerocopy) hapiCheck(hapiFree(d_recv_ghosts_b[i]));
      }
    } else {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(cudaFreeHost(h_ghosts[i]));
        hapiCheck(hapiFree(d_ghosts[i]));
      }
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

    // Allocate memory and create CUDA entities. Device buffers come from the
    // runtime's pool (+gpupool): one registration and one IPC export for the
    // whole arena, instead of one per buffer on first use.
    hapiCheck(cudaMallocHost((void**)&h_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(hapiMalloc((void**)&d_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(hapiMalloc((void**)&d_new_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    std::vector<size_t> ghost_sizes = {x_surf_size, x_surf_size, y_surf_size,
      y_surf_size, z_surf_size, z_surf_size};
    if (use_zerocopy || use_persistent || use_channels) {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(hapiMalloc((void**)&d_send_ghosts[i], ghost_sizes[i]));
        hapiCheck(hapiMalloc((void**)&d_recv_ghosts[i], ghost_sizes[i]));
        // Zero-copy lands a neighbour's face at message arrival, which can be
        // before this block has unpacked the previous one from the same
        // buffer: receive by iteration parity. Channels double-buffer both.
        if (use_channels) hapiCheck(hapiMalloc((void**)&d_send_ghosts_b[i], ghost_sizes[i]));
        if (use_channels || use_zerocopy) hapiCheck(hapiMalloc((void**)&d_recv_ghosts_b[i], ghost_sizes[i]));
      }
    } else {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(cudaMallocHost((void**)&h_ghosts[i], ghost_sizes[i]));
        hapiCheck(hapiMalloc((void**)&d_ghosts[i], ghost_sizes[i]));
      }
    }

    // Create CUDA streams and events
    hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));

    hapiCheck(cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming));
    hapiCheck(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));

    // Create persistent buffers
    if (use_persistent) {
      CkCallback recv_cb = CkCallback(CkIndex_Block::recvGhostP(nullptr), thisProxy[thisIndex]);

      p_send_bufs.reserve(DIR_COUNT);
      p_recv_bufs.reserve(DIR_COUNT);
      p_neighbor_bufs.resize(DIR_COUNT);

      for (int i = 0; i < DIR_COUNT; i++) {
        p_send_bufs.emplace_back(d_send_ghosts[i], ghost_sizes[i], CkCallback::ignore, comm_stream);
        p_recv_bufs.emplace_back(d_recv_ghosts[i], ghost_sizes[i], recv_cb, comm_stream);

        // Open buffers that will be sent to neighbors
        p_recv_bufs[i].open();
      }

      // Send persistent buffer info to neighbors
      if (!bounds[LEFT])   thisProxy(x-1, y, z).initRecv(RIGHT, p_recv_bufs[LEFT]);
      if (!bounds[RIGHT])  thisProxy(x+1, y, z).initRecv(LEFT, p_recv_bufs[RIGHT]);
      if (!bounds[TOP])    thisProxy(x, y-1, z).initRecv(BOTTOM, p_recv_bufs[TOP]);
      if (!bounds[BOTTOM]) thisProxy(x, y+1, z).initRecv(TOP, p_recv_bufs[BOTTOM]);
      if (!bounds[FRONT])  thisProxy(x, y, z-1).initRecv(BACK, p_recv_bufs[FRONT]);
      if (!bounds[BACK])   thisProxy(x, y, z+1).initRecv(FRONT, p_recv_bufs[BACK]);
    }

    // Device channels: one per neighbour and parity, owned here, landing a
    // neighbour's face in my receive buffer; the copy the neighbour puts into.
    if (use_channels) {
      CkCallback arrived(CkIndex_Block::recvGhostCh(nullptr), thisProxy[thisIndex]);
      for (int p = 0; p < 2; p++) {
        for (int i = 0; i < DIR_COUNT; i++) {
          if (bounds[i]) continue;
          ch_recv[p][i] = CkDeviceChannel(recvBuf(p, i), ghost_sizes[i], arrived);
          ch_recv[p][i].open();
        }
        if (!bounds[LEFT])   thisProxy(x-1, y, z).initRecvCh(p, RIGHT, ch_recv[p][LEFT]);
        if (!bounds[RIGHT])  thisProxy(x+1, y, z).initRecvCh(p, LEFT, ch_recv[p][RIGHT]);
        if (!bounds[TOP])    thisProxy(x, y-1, z).initRecvCh(p, BOTTOM, ch_recv[p][TOP]);
        if (!bounds[BOTTOM]) thisProxy(x, y+1, z).initRecvCh(p, TOP, ch_recv[p][BOTTOM]);
        if (!bounds[FRONT])  thisProxy(x, y, z-1).initRecvCh(p, BACK, ch_recv[p][FRONT]);
        if (!bounds[BACK])   thisProxy(x, y, z+1).initRecvCh(p, FRONT, ch_recv[p][BACK]);
      }
    }

    // Initialize temperature data
    invokeInitKernel(d_temperature, block_width, block_height, block_depth, compute_stream);
    invokeInitKernel(d_new_temperature, block_width, block_height, block_depth, compute_stream);

    // Initialize ghost data
    std::vector<int> ghost_counts = {x_surf_count, x_surf_count, y_surf_count,
      y_surf_count, z_surf_count, z_surf_count};
    if (use_zerocopy || use_persistent || use_channels) {
      std::vector<DataType*> send_ghosts;
      std::vector<DataType*> recv_ghosts;
      for (int i = 0; i < DIR_COUNT; i++) {
        send_ghosts.push_back(d_send_ghosts[i]);
        recv_ghosts.push_back(d_recv_ghosts[i]);
      }
      invokeGhostInitKernels(send_ghosts, ghost_counts, compute_stream);
      invokeGhostInitKernels(recv_ghosts, ghost_counts, compute_stream);
      if (use_channels || use_zerocopy) {
        std::vector<DataType*> recv_b;
        for (int i = 0; i < DIR_COUNT; i++) recv_b.push_back(d_recv_ghosts_b[i]);
        invokeGhostInitKernels(recv_b, ghost_counts, compute_stream);
      }
      if (use_channels) {
        std::vector<DataType*> send_b;
        for (int i = 0; i < DIR_COUNT; i++) send_b.push_back(d_send_ghosts_b[i]);
        invokeGhostInitKernels(send_b, ghost_counts, compute_stream);
      }
    } else {
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
    if (use_channels) {
      // This iteration's parity set of send buffers (the other set may still
      // be read by the previous iteration's puts).
      invokePackingKernels(d_temperature, (my_iter & 1) ? d_send_ghosts_b : d_send_ghosts, bounds,
          block_width, block_height, block_depth, comm_stream);
    } else if (use_persistent || use_zerocopy) {
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_temperature, d_send_ghosts, bounds,
          block_width, block_height, block_depth, comm_stream);
    } else {
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

  // One source callback per send, carrying this iteration (see sendDoneFn).
  CkCallback srcDone() {
    return CkCallback(sendDoneFn, (void*)new SendDoneCtx{thisProxy.ckGetArrayID(), thisIndex, my_iter});
  }

  // -v: block (0,0,0) prints where it is, each iteration.
  void trace(const char* what) {
    if (verbose && x == 0 && y == 0 && z == 0) CkPrintf("[j3d %d] iter %d: %s\n", CkMyPe(), my_iter, what);
  }

  void sendGhosts() {
    // Send ghosts to neighboring chares
    if (use_persistent) {
      // PersistentMsg is used to store the direction
      PersistentMsg* msg;
      for (int dir = 0; dir < DIR_COUNT; dir++) {
        int rev_dir = (dir % 2 == 0) ? (dir+1) : (dir-1);
        if (!bounds[dir]) {
          msg = new PersistentMsg(rev_dir);
          p_neighbor_bufs[dir].set_msg(msg);
          p_neighbor_bufs[dir].cb.setRefNum(my_iter);
          p_send_bufs[dir].put(p_neighbor_bufs[dir]);
        }
      }
    } else if (use_channels) {
      // The put is issued behind the packing kernels on comm_stream; the
      // receiver's event carries my_iter as its reference number.
      const int p = my_iter & 1;
      std::vector<size_t> ghost_sizes = {x_surf_size, x_surf_size, y_surf_size,
        y_surf_size, z_surf_size, z_surf_size};
      for (int dir = 0; dir < DIR_COUNT; dir++)
        if (!bounds[dir])
          ch_send[p][dir].put(sendBuf(p, dir), ghost_sizes[dir], comm_stream, srcDone(), (uint64_t)my_iter);
    } else if (use_zerocopy) {
      // The source callback says when the neighbour has read the face; the
      // next iteration's packing waits for all of them (run()).
      if (!bounds[LEFT])
        thisProxy(x-1, y, z).recvGhostZC(my_iter, RIGHT, x_surf_count,
            CkDeviceBuffer(d_send_ghosts[LEFT], x_surf_count, srcDone(), comm_stream));
      if (!bounds[RIGHT])
        thisProxy(x+1, y, z).recvGhostZC(my_iter, LEFT, x_surf_count,
            CkDeviceBuffer(d_send_ghosts[RIGHT], x_surf_count, srcDone(), comm_stream));
      if (!bounds[TOP])
        thisProxy(x, y-1, z).recvGhostZC(my_iter, BOTTOM, y_surf_count,
            CkDeviceBuffer(d_send_ghosts[TOP], y_surf_count, srcDone(), comm_stream));
      if (!bounds[BOTTOM])
        thisProxy(x, y+1, z).recvGhostZC(my_iter, TOP, y_surf_count,
            CkDeviceBuffer(d_send_ghosts[BOTTOM], y_surf_count, srcDone(), comm_stream));
      if (!bounds[FRONT])
        thisProxy(x, y, z-1).recvGhostZC(my_iter, BACK, z_surf_count,
            CkDeviceBuffer(d_send_ghosts[FRONT], z_surf_count, srcDone(), comm_stream));
      if (!bounds[BACK])
        thisProxy(x, y, z+1).recvGhostZC(my_iter, FRONT, z_surf_count,
            CkDeviceBuffer(d_send_ghosts[BACK], z_surf_count, srcDone(), comm_stream));
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
    buf = recvBuf(ref & 1, dir);   // by iteration parity: see init()
    if (dir == LEFT || dir == RIGHT) count = x_surf_count;
    else if (dir == TOP || dir == BOTTOM) count = y_surf_count;
    else if (dir == FRONT || dir == BACK) count = z_surf_count;
    devicePost[0].hapi_stream = comm_stream;   // held behind the previous unpack on this stream (buffer_free stays false)
  }

  void processGhostZC(int dir, int count, DataType* gh) {
    trace("ghost in (zero-copy)");
    // gh is the buffer the post entry chose for this iteration's parity.
    invokeUnpackingKernel(d_temperature, gh, dir, block_width, block_height,
        block_depth, comm_stream);
  }

  void processGhostP(PersistentMsg* msg) {
    int dir = msg->dir;
    CkAssert(dir >= 0 && dir < DIR_COUNT);
    DataType* d_ghost = d_recv_ghosts[dir];

    invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height,
        block_depth, comm_stream);

    delete msg;
  }

  // A neighbour's face landed in one of my channel buffers: the event names
  // the channel; unpack from that buffer.
  void processGhostCh(CkDataMsg* m) {
    const CkDeviceChannelEvent* peek = (const CkDeviceChannelEvent*)m->getData();
    int p = -1, dir = -1;
    for (int q = 0; q < 2 && p < 0; q++)
      for (int d = 0; d < DIR_COUNT; d++)
        if (!bounds[d] && ch_recv[q][d].id() == peek->channel) { p = q; dir = d; break; }
    if (p < 0) CkAbort("jacobi3d: a channel event for a channel that is not mine\n");
    const CkDeviceChannelEvent* e = ch_recv[p][dir].event(m);
    delete m;
    if (e == NULL) CkAbort("jacobi3d: a control event reached recvGhostCh\n");
    trace("ghost in (channel)");
    invokeUnpackingKernel(d_temperature, recvBuf(p, dir), dir, block_width, block_height,
        block_depth, comm_stream);
  }

  // After the update: the unpacking kernels that read this iteration's
  // buffers have completed (the update waited on comm_stream), so the
  // neighbours may put into them again.
  void releaseGhosts() {
    const int p = my_iter & 1;
    for (int dir = 0; dir < DIR_COUNT; dir++)
      if (!bounds[dir]) ch_recv[p][dir].release();
  }

  void processGhostReg(int dir, int size, DataType* gh) {
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
    std::swap(d_temperature, d_new_temperature);
    my_iter++;
    if (my_iter <= warmup_iters) {
      contribute(CkCallback(CkReductionTarget(Main, warmupDone), main_proxy));
    } else {
      if (my_iter < warmup_iters + n_iters) {
        thisProxy[thisIndex].run();
      } else {
        // Correctness: the interior summed on the host (the end only; the
        // cost is not timed).
        const size_t n = (size_t)(block_width+2) * (block_height+2) * (block_depth+2);
        hapiCheck(cudaMemcpy(h_temperature, d_temperature, sizeof(DataType) * n, cudaMemcpyDeviceToHost));
        double sum = 0.0;
        for (int k = 1; k <= block_depth; k++)
          for (int j = 1; j <= block_height; j++)
            for (int i = 1; i <= block_width; i++) sum += (double)h_temperature[IDX(i,j,k)];
        contribute(sizeof(double), &sum, CkReduction::sum_double,
                   CkCallback(CkReductionTarget(Main, allDone), main_proxy));
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
