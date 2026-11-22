#include "hapi.h"
#include "jacobi3d.decl.h"
#include "jacobi3d.h"
#include <utility>
#include <sstream>

/* CkDeviceBufferPost names its stream field hapi_stream on the reconverse line
 * and cuda_stream on the classic one (ckrdmadevice.h splits at CMK_RECONVERSE),
 * so a single copy of this example has to spell it both ways until the classic
 * line is removed. Same shim as tests/charm++/cuda/d2dtest. */
#if CMK_RECONVERSE
#define POST_STREAM(p) ((p).hapi_stream)
#else
#define POST_STREAM(p) ((p).cuda_stream)
#endif

#define GPU_SYNC 0

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
/* readonly */ bool use_channels;
/* readonly */ bool print_elements;

extern void invokeInitKernel(DataType* d_temperature, int block_width,
    int block_height, int block_depth, hapiStream_t stream);
extern void invokeGhostInitKernels(const std::vector<DataType*>& ghosts,
    const std::vector<int>& ghost_counts, hapiStream_t stream);
extern void invokeBoundaryKernels(DataType* d_temperature, int block_width,
    int block_height, int block_depth, bool bounds[], hapiStream_t stream);
extern void invokeJacobiKernel(DataType* d_temperature, DataType* d_new_temperature,
    int block_width, int block_height, int block_depth, hapiStream_t stream);
extern void invokePackingKernels(DataType* d_temperature, DataType* d_ghosts[],
    bool bounds[], int block_width, int block_height, int block_depth,
    hapiStream_t stream);
extern void invokeUnpackingKernel(DataType* d_temperature, DataType* d_ghost,
    int dir, int block_width, int block_height, int block_depth,
    hapiStream_t stream);

// The source callback of a channel put: re-issued to the block as
// sendDone(iter) with the iteration as the message's reference number, which
// the SDAG `when` matches.
struct SendDoneCtx { CkArrayID aid; CkArrayIndex3D idx; int iter; };
static void sendDoneFn(void* param, void* msg) {
  SendDoneCtx* c = (SendDoneCtx*)param;
  CkDataMsg* m = CkDataMsg::buildNew((int)sizeof(int), &c->iter);
  CkSetRefNum(m, (CMK_REFNUM_TYPE)c->iter);
  CProxy_Block(c->aid)(c->idx.index[0], c->idx.index[1], c->idx.index[2]).sendDone(m);
  if (msg) CkFreeMsg(msg);
  delete c;
}

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
    use_channels = false;
    print_elements = false;
    my_iter = 0;

    // Process arguments
    int c;
    bool dims[3] = {false, false, false};
    while ((c = getopt(m->argc, m->argv, "c:x:y:z:i:w:dkp")) != -1) {
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
        case 'k':
          use_channels = true;
          break;
        case 'p':
          print_elements = true;
          break;
        default:
          CkPrintf(
              "Usage: %s -x [grid width] -y [grid height] -z [grid depth] "
              "-c [number of chares] -i [iterations] -w [warmup iterations] "
              "-d (use GPU zerocopy) -k (use device channels) -p (print blocks)\n",
              m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // Zerocopy and channels cannot be used together
    if (use_zerocopy && use_channels) {
      CkPrintf("Zerocopy and channels cannot be used together!\n");
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
    CkPrintf("\n[GPU 3D Jacobi example]\n");
    CkPrintf("Grid: %d x %d x %d, Block: %d x %d x %d, Chares: %d x %d x %d, "
        "Iterations: %d, Warm-up: %d, Zerocopy: %d, Channels: %d, Print: %d\n\n",
        grid_width, grid_height, grid_depth, block_width, block_height, block_depth,
        n_chares_x, n_chares_y, n_chares_z, n_iters, warmup_iters, use_zerocopy,
        use_channels, print_elements);

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

  // The timed region ends at itersDone, which every block contributes to after
  // its last iteration and before the checksum (a copy and a sum on the host).
  double end_time = 0.0, checksum_ = 0.0;
  bool iters_done = false, all_done = false;
  void itersDone() { end_time = CkWallTimer(); iters_done = true; if (all_done) finish(); }
  void allDone(double checksum) { checksum_ = checksum; all_done = true; if (iters_done) finish(); }
  void finish() {
    double checksum = checksum_;
    double total_time = end_time - start_time;
    CkPrintf("Total time: %.3lf s\nAverage iteration time: %.3lf us\n",
        total_time, (total_time / n_iters) * 1e6);
    // The sum of every block's interior after the last iteration: identical
    // across the transfer modes when the exchange is correct.
    CkPrintf("Checksum: %.12e\n", checksum);
    CkExit();
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

  // -k: one channel per neighbour and iteration parity. Mine are opened on my
  // receive buffers; the neighbours' are the ones my faces are put into.
  CkDeviceChannel ch_recv[2][DIR_COUNT];
  CkDeviceChannel ch_send[2][DIR_COUNT];

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
  /* Two send-buffer sets, selected per iteration by curSendGhosts(), in both
   * the zerocopy and the channel paths. */
  DataType* d_send_ghosts[2][DIR_COUNT];
  /* Receive side, same shape. Zerocopy: the post entry method runs when a
   * message is delivered to this object, before the SDAG when-clause matches
   * it, so a neighbour's iteration i+1 ghost can be fetched into a receive
   * buffer while iteration i's contents are still waiting to be unpacked; the
   * set is selected by the message's iteration parity. Channels: one channel
   * per set, so a neighbour can put iteration i+1 while this block still reads
   * iteration i. */
  DataType* d_recv_ghosts[2][DIR_COUNT];

  hapiStream_t compute_stream;
  hapiStream_t comm_stream;

  hapiEvent_t compute_event;
  hapiEvent_t comm_event;

  bool bounds[DIR_COUNT];

  /* The send-buffer set this iteration packs into and sends from.
   *
   * Alternating by iteration parity is sufficient on its own, with no
   * acknowledgement from the receiver: a neighbour issues its get of our
   * iteration i buffer while processing our iteration i metadata, and it only
   * sends its own iteration i+1 ghost after that get has completed and it has
   * finished iteration i. We cannot start packing iteration i+2 -- the next
   * use of this same set -- until that i+1 message has arrived, so by then
   * every neighbour has finished reading the set. A channel put read by its
   * receiver (another host) is read outside every stream, so the channel path
   * also waits for the iteration-i puts' source callbacks before packing i+2
   * (run()). */
  DataType** curSendGhosts() {
    return d_send_ghosts[my_iter & 1];
  }

  Block() {}

  ~Block() {
    hapiCheck(hapiFreeHost(h_temperature));
    hapiCheck(hapiFree(d_temperature));
    hapiCheck(hapiFree(d_new_temperature));
    if (use_zerocopy || use_channels) {
      for (int p = 0; p < 2; p++) {
        for (int i = 0; i < DIR_COUNT; i++) {
          if (use_channels && !bounds[i]) ch_recv[p][i].close();
          hapiCheck(hapiFree(d_send_ghosts[p][i]));
          hapiCheck(hapiFree(d_recv_ghosts[p][i]));
        }
      }
    } else {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(hapiFreeHost(h_ghosts[i]));
        hapiCheck(hapiFree(d_ghosts[i]));
      }
    }

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

    // Allocate memory and create GPU entities
    hapiCheck(hapiMallocHost((void**)&h_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(hapiMalloc((void**)&d_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    hapiCheck(hapiMalloc((void**)&d_new_temperature,
          sizeof(DataType) * (block_width+2) * (block_height+2) * (block_depth+2)));
    std::vector<size_t> ghost_sizes = {x_surf_size, x_surf_size, y_surf_size,
      y_surf_size, z_surf_size, z_surf_size};
    if (use_zerocopy || use_channels) {
      for (int p = 0; p < 2; p++) {
        for (int i = 0; i < DIR_COUNT; i++) {
          hapiCheck(hapiMalloc((void**)&d_send_ghosts[p][i], ghost_sizes[i]));
          hapiCheck(hapiMalloc((void**)&d_recv_ghosts[p][i], ghost_sizes[i]));
        }
      }
    } else {
      for (int i = 0; i < DIR_COUNT; i++) {
        hapiCheck(hapiMallocHost((void**)&h_ghosts[i], ghost_sizes[i]));
        hapiCheck(hapiMalloc((void**)&d_ghosts[i], ghost_sizes[i]));
      }
    }

    // Create GPU streams and events
    hapiCheck(hapiStreamCreateWithPriority(&compute_stream, hapiStreamDefault, 0));
    hapiCheck(hapiStreamCreateWithPriority(&comm_stream, hapiStreamDefault, -1));

    hapiCheck(hapiEventCreateWithFlags(&compute_event, hapiEventDisableTiming));
    hapiCheck(hapiEventCreateWithFlags(&comm_event, hapiEventDisableTiming));

    // Device channels: one per neighbour and parity, opened on my receive
    // buffers and sent to the neighbour that puts into them.
    if (use_channels) {
      CkCallback arrived(CkIndex_Block::recvGhostCh(nullptr), thisProxy[thisIndex]);
      for (int p = 0; p < 2; p++) {
        for (int i = 0; i < DIR_COUNT; i++) {
          if (bounds[i]) continue;
          ch_recv[p][i] = CkDeviceChannel(d_recv_ghosts[p][i], ghost_sizes[i], arrived);
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
    if (use_zerocopy || use_channels) {
      for (int p = 0; p < 2; p++) {
        std::vector<DataType*> send_ghosts;
        std::vector<DataType*> recv_ghosts;
        for (int i = 0; i < DIR_COUNT; i++) {
          send_ghosts.push_back(d_send_ghosts[p][i]);
          recv_ghosts.push_back(d_recv_ghosts[p][i]);
        }
        invokeGhostInitKernels(send_ghosts, ghost_counts, compute_stream);
        invokeGhostInitKernels(recv_ghosts, ghost_counts, compute_stream);
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

#if GPU_SYNC
    hapiStreamSynchronize(compute_stream);
    thisProxy[thisIndex].initDone();
#else
    // TODO: Support reduction callback in hapiAddCallback
    CkCallback* cb = new CkCallback(CkIndex_Block::initDone(), thisProxy[thisIndex]);
    hapiAddCallback(compute_stream, cb);
#endif
  }

  void packGhosts() {
    if (use_channels || use_zerocopy) {
      // Pack non-contiguous ghosts to temporary contiguous buffers on device
      invokePackingKernels(d_temperature, curSendGhosts(), bounds,
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
          hapiCheck(hapiMemcpyAsync(h_ghosts[i], d_ghosts[i], ghost_sizes[i],
                hapiMemcpyDeviceToHost, comm_stream));
        }
      }
    }

    if (use_channels) {
      // A put is ordered behind the packing kernels on comm_stream by the runtime.
      thisProxy[thisIndex].packGhostsDone();
    } else {
#if GPU_SYNC
      hapiStreamSynchronize(comm_stream);
      thisProxy[thisIndex].packGhostsDone();
#else
      // Add asynchronous callback to be invoked when packing kernels and
      // ghost transfers are complete
      CkCallback* cb = new CkCallback(CkIndex_Block::packGhostsDone(), thisProxy[thisIndex]);
      hapiAddCallback(comm_stream, cb);
#endif
    }
  }

  // One source callback per put, carrying this iteration (see sendDoneFn).
  CkCallback srcDone() {
    return CkCallback(sendDoneFn, (void*)new SendDoneCtx{thisProxy.ckGetArrayID(), thisIndex, my_iter});
  }

  void sendGhosts() {
    // Send ghosts to neighboring chares
    if (use_channels) {
      // Into this iteration's parity channel of each neighbour; the event the
      // neighbour receives carries my_iter as its reference number.
      const int p = my_iter & 1;
      DataType** d_send = curSendGhosts();
      std::vector<size_t> ghost_sizes = {x_surf_size, x_surf_size, y_surf_size,
        y_surf_size, z_surf_size, z_surf_size};
      for (int dir = 0; dir < DIR_COUNT; dir++)
        if (!bounds[dir])
          ch_send[p][dir].put(d_send[dir], ghost_sizes[dir], comm_stream, srcDone(), (uint64_t)my_iter);
    } else if (use_zerocopy) {
      // Same set packGhosts() just filled -- see curSendGhosts() for why
      // alternating it by parity keeps a neighbour's get from racing the
      // repack two iterations later.
      DataType** d_send = curSendGhosts();

      if (!bounds[LEFT])
        thisProxy(x-1, y, z).recvGhostZC(my_iter, RIGHT, x_surf_count,
            CkDeviceBuffer(d_send[LEFT], x_surf_count, comm_stream));
      if (!bounds[RIGHT])
        thisProxy(x+1, y, z).recvGhostZC(my_iter, LEFT, x_surf_count,
            CkDeviceBuffer(d_send[RIGHT], x_surf_count, comm_stream));
      if (!bounds[TOP])
        thisProxy(x, y-1, z).recvGhostZC(my_iter, BOTTOM, y_surf_count,
            CkDeviceBuffer(d_send[TOP], y_surf_count, comm_stream));
      if (!bounds[BOTTOM])
        thisProxy(x, y+1, z).recvGhostZC(my_iter, TOP, y_surf_count,
            CkDeviceBuffer(d_send[BOTTOM], y_surf_count, comm_stream));
      if (!bounds[FRONT])
        thisProxy(x, y, z-1).recvGhostZC(my_iter, BACK, z_surf_count,
            CkDeviceBuffer(d_send[FRONT], z_surf_count, comm_stream));
      if (!bounds[BACK])
        thisProxy(x, y, z+1).recvGhostZC(my_iter, FRONT, z_surf_count,
            CkDeviceBuffer(d_send[BACK], z_surf_count, comm_stream));
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
    // ref is the sender's iteration; it equals my_iter for the message the
    // SDAG loop is waiting on and my_iter+1 for one that arrived early.
    buf = d_recv_ghosts[ref & 1][dir];
    if (dir == LEFT || dir == RIGHT) count = x_surf_count;
    else if (dir == TOP || dir == BOTTOM) count = y_surf_count;
    else if (dir == FRONT || dir == BACK) count = z_surf_count;
    POST_STREAM(devicePost[0]) = comm_stream;
  }

  void processGhostZC(int ref, int dir, int count, DataType* gh) {
    // Do not unpack from gh. On the inter-process (RDMA) path the pointer
    // delivered with the regular entry method is the SENDER's device address,
    // copied out of the metadata message, not the buffer we posted; reading it
    // faults the kernel and the context, and the run hangs with no message.
    // Index our own receive set by the message's iteration, exactly as the
    // post entry method did when it handed the buffer out.
    invokeUnpackingKernel(d_temperature, d_recv_ghosts[ref & 1][dir], dir,
        block_width, block_height, block_depth, comm_stream);
  }

  // A neighbour's face is in one of my channel buffers: the event names the
  // channel, which names the buffer to unpack from.
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
    invokeUnpackingKernel(d_temperature, d_recv_ghosts[p][dir], dir, block_width, block_height,
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
    hapiCheck(hapiMemcpyAsync(d_ghost, h_ghost, size * sizeof(DataType),
          hapiMemcpyHostToDevice, comm_stream));
    invokeUnpackingKernel(d_temperature, d_ghost, dir, block_width, block_height,
        block_depth, comm_stream);
  }

  void update() {
    // Operations in compute stream should only be executed when
    // operations in communication stream (transfers and unpacking) complete
    hapiCheck(hapiEventRecord(comm_event, comm_stream));
    hapiCheck(hapiStreamWaitEvent(compute_stream, comm_event, 0));

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
        contribute(CkCallback(CkReductionTarget(Main, itersDone), main_proxy));
        // The interior summed on the host, at the end only (not timed).
        const size_t n = (size_t)(block_width+2) * (block_height+2) * (block_depth+2);
        hapiCheck(hapiMemcpy(h_temperature, d_temperature, sizeof(DataType) * n, hapiMemcpyDeviceToHost));
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
    hapiCheck(hapiMemcpyAsync(h_temperature, d_temperature,
          sizeof(DataType) * (block_width+2)*(block_height+2)*(block_depth+2),
          hapiMemcpyDeviceToHost, comm_stream));
    hapiStreamSynchronize(comm_stream);

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
