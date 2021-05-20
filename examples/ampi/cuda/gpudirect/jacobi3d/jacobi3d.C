#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <cuda_runtime.h>
#include "jacobi3d.h"

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

// Program parameters
struct Param {
  int n_procs;
  int rank;
  MPI_Comm cart_comm;

  int n_chares;
  int n_chares_x;
  int n_chares_y;
  int n_chares_z;
  int grid_width;
  int grid_height;
  int grid_depth;
  int block_width;
  int block_height;
  int block_depth;
  int x_surf_count;
  int y_surf_count;
  int z_surf_count;
  size_t x_surf_size;
  size_t y_surf_size;
  size_t z_surf_size;
  int n_iters;
  int warmup_iters;
  bool use_zerocopy;
  bool print_elements;

  int my_iter;
  int x, y, z;

  Param(int n_procs_, int rank_) : n_procs(n_procs_), rank(rank_) {
    n_chares = n_procs;
    grid_width = grid_height = grid_depth = 512;
    n_iters = 100;
    warmup_iters = 10;
    use_zerocopy = false;
    print_elements = false;
  }
};

// Main block object
struct Block {
  Param param;
  int my_iter;
  int neighbors;
  int neighbor_ranks[DIR_COUNT];
  int x, y, z;

  cudaStream_t compute_stream;
  cudaStream_t h2d_stream;
  cudaStream_t d2h_stream;
  cudaStream_t unpack_stream;
  cudaStream_t pack_stream;

  DataType* h_temperature;
  DataType* d_temperature;
  DataType* d_new_temperature;

  DataType* h_send_ghosts[DIR_COUNT];
  DataType* h_recv_ghosts[DIR_COUNT];
  DataType* d_send_ghosts[DIR_COUNT];
  DataType* d_recv_ghosts[DIR_COUNT];

  cudaEvent_t compute_pack_event;
  cudaEvent_t pack_d2h_events[DIR_COUNT];
  cudaEvent_t h2d_unpack_events[DIR_COUNT];
  cudaEvent_t unpack_compute_event;

  bool bounds[DIR_COUNT];
  std::vector<size_t> ghost_sizes;

  Block(Param param_, int x_, int y_, int z_)
    : param(param_), x(x_), y(y_), z(z_), my_iter(0), neighbors(0) {
    // Check bounds and set number of valid neighbors
    for (int i = 0; i < DIR_COUNT; i++) bounds[i] = false;
    if (x == 0)                  bounds[LEFT] = true;
    else                         neighbors++;
    if (x == param.n_chares_x-1) bounds[RIGHT]= true;
    else                         neighbors++;
    if (y == 0)                  bounds[TOP] = true;
    else                         neighbors++;
    if (y == param.n_chares_y-1) bounds[BOTTOM] = true;
    else                         neighbors++;
    if (z == 0)                  bounds[FRONT] = true;
    else                         neighbors++;
    if (z == param.n_chares_z-1) bounds[BACK] = true;
    else                         neighbors++;

    // Determine neighbor ranks
    for (int i = 0; i < DIR_COUNT; i++) {
      neighbor_ranks[i] = MPI_PROC_NULL;
    }
    if (!bounds[LEFT]) {
      int coords[NDIMS] = {x-1, y, z};
      MPI_Cart_rank(param.cart_comm, coords, &neighbor_ranks[LEFT]);
    }
    if (!bounds[RIGHT]) {
      int coords[NDIMS] = {x+1, y, z};
      MPI_Cart_rank(param.cart_comm, coords, &neighbor_ranks[RIGHT]);
    }
    if (!bounds[TOP]) {
      int coords[NDIMS] = {x, y-1, z};
      MPI_Cart_rank(param.cart_comm, coords, &neighbor_ranks[TOP]);
    }
    if (!bounds[BOTTOM]) {
      int coords[NDIMS] = {x, y+1, z};
      MPI_Cart_rank(param.cart_comm, coords, &neighbor_ranks[BOTTOM]);
    }
    if (!bounds[FRONT]) {
      int coords[NDIMS] = {x, y, z-1};
      MPI_Cart_rank(param.cart_comm, coords, &neighbor_ranks[FRONT]);
    }
    if (!bounds[BACK]) {
      int coords[NDIMS] = {x, y, z+1};
      MPI_Cart_rank(param.cart_comm, coords, &neighbor_ranks[BACK]);
    }

    // Create CUDA streams
    cudaCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
    cudaCheck(cudaStreamCreateWithPriority(&h2d_stream, cudaStreamDefault, -1));
    cudaCheck(cudaStreamCreateWithPriority(&d2h_stream, cudaStreamDefault, -1));
    cudaCheck(cudaStreamCreateWithPriority(&unpack_stream, cudaStreamDefault, -1));
    cudaCheck(cudaStreamCreateWithPriority(&pack_stream, cudaStreamDefault, -1));

    // Allocate memory
    cudaCheck(cudaMallocHost((void**)&h_temperature,
          sizeof(DataType) * (param.block_width+2) * (param.block_height+2)
          * (param.block_depth+2)));
    cudaCheck(cudaMalloc((void**)&d_temperature,
          sizeof(DataType) * (param.block_width+2) * (param.block_height+2)
          * (param.block_depth+2)));
    cudaCheck(cudaMalloc((void**)&d_new_temperature,
          sizeof(DataType) * (param.block_width+2) * (param.block_height+2)
          * (param.block_depth+2)));
    ghost_sizes = {param.x_surf_size, param.x_surf_size, param.y_surf_size,
                   param.y_surf_size, param.z_surf_size, param.z_surf_size};
    for (int i = 0; i < DIR_COUNT; i++) {
      cudaCheck(cudaMalloc((void**)&d_send_ghosts[i], ghost_sizes[i]));
      cudaCheck(cudaMalloc((void**)&d_recv_ghosts[i], ghost_sizes[i]));
      if (!param.use_zerocopy) {
        cudaCheck(cudaMallocHost((void**)&h_send_ghosts[i], ghost_sizes[i]));
        cudaCheck(cudaMallocHost((void**)&h_recv_ghosts[i], ghost_sizes[i]));
      }
    }

    // Create CUDA events for enforcing dependencies
    cudaCheck(cudaEventCreateWithFlags(&compute_pack_event, cudaEventDisableTiming));
    for (int i = 0; i < DIR_COUNT; i++) {
      cudaCheck(cudaEventCreateWithFlags(&pack_d2h_events[i], cudaEventDisableTiming));
      cudaCheck(cudaEventCreateWithFlags(&h2d_unpack_events[i], cudaEventDisableTiming));
    }
    cudaCheck(cudaEventCreateWithFlags(&unpack_compute_event, cudaEventDisableTiming));
  }

  void init() {
    // Initialize temperature data
    invokeInitKernel(d_temperature, param.block_width, param.block_height,
        param.block_depth, compute_stream);
    invokeInitKernel(d_new_temperature, param.block_width, param.block_height,
        param.block_depth, compute_stream);

    // Initialize ghost data
    std::vector<int> ghost_counts
      = {param.x_surf_count, param.x_surf_count, param.y_surf_count,
         param.y_surf_count, param.z_surf_count, param.z_surf_count};
    std::vector<DataType*> send_ghosts;
    std::vector<DataType*> recv_ghosts;
    for (int i = 0; i < DIR_COUNT; i++) {
      send_ghosts.push_back(d_send_ghosts[i]);
      recv_ghosts.push_back(d_recv_ghosts[i]);
    }
    invokeGhostInitKernels(send_ghosts, ghost_counts, compute_stream);
    invokeGhostInitKernels(recv_ghosts, ghost_counts, compute_stream);
    if (!param.use_zerocopy) {
      for (int i = 0; i < DIR_COUNT; i++) {
        int ghost_count = ghost_counts[i];
        for (int j = 0; j < ghost_count; j++) {
          h_send_ghosts[i][j] = 0;
          h_recv_ghosts[i][j] = 0;
        }
      }
    }

    // Enforce boundary conditions
    invokeBoundaryKernels(d_temperature, param.block_width, param.block_height,
        param.block_depth, bounds, compute_stream);
    invokeBoundaryKernels(d_new_temperature, param.block_width, param.block_height,
        param.block_depth, bounds, compute_stream);

    // TODO: Use HAPI for AMPI
    cudaCheck(cudaStreamSynchronize(compute_stream));
  }

  ~Block() {
    // Destroy CUDA events
    cudaCheck(cudaEventDestroy(compute_pack_event));
    for (int i = 0; i < DIR_COUNT; i++) {
      cudaCheck(cudaEventDestroy(pack_d2h_events[i]));
      cudaCheck(cudaEventDestroy(h2d_unpack_events[i]));
    }
    cudaCheck(cudaEventDestroy(unpack_compute_event));

    // Free allocated memory
    cudaCheck(cudaFreeHost(h_temperature));
    cudaCheck(cudaFree(d_temperature));
    cudaCheck(cudaFree(d_new_temperature));
    for (int i = 0; i < DIR_COUNT; i++) {
      cudaCheck(cudaFree(d_send_ghosts[i]));
      cudaCheck(cudaFree(d_recv_ghosts[i]));
      if (!param.use_zerocopy) {
        cudaCheck(cudaFreeHost(h_send_ghosts[i]));
        cudaCheck(cudaFreeHost(h_recv_ghosts[i]));
      }
    }

    // Destroy CUDA streams
    cudaCheck(cudaStreamDestroy(compute_stream));
    cudaCheck(cudaStreamDestroy(h2d_stream));
    cudaCheck(cudaStreamDestroy(d2h_stream));
    cudaCheck(cudaStreamDestroy(unpack_stream));
    cudaCheck(cudaStreamDestroy(pack_stream));
  }

  void updateAndPack() {
    // Enforce unpack -> compute dependency
    cudaCheck(cudaEventRecord(unpack_compute_event, unpack_stream));
    cudaCheck(cudaStreamWaitEvent(compute_stream, unpack_compute_event, 0));

    // Invoke GPU kernel for Jacobi computation
    invokeJacobiKernel(d_temperature, d_new_temperature, param.block_width,
        param.block_height, param.block_depth, compute_stream);
    cudaCheck(cudaEventRecord(compute_pack_event, compute_stream));

    for (int i = 0; i < DIR_COUNT; i++) {
      if (!bounds[i]) {
        // Enforce compute -> pack dependency
        cudaCheck(cudaStreamWaitEvent(pack_stream, compute_pack_event, 0));

        // Pack
        invokePackingKernel(d_temperature, d_send_ghosts[i], i, param.block_width,
            param.block_height, param.block_depth, pack_stream);

        if (!param.use_zerocopy) {
          // Enforce pack -> d2h dependency
          cudaCheck(cudaEventRecord(pack_d2h_events[i], pack_stream));
          cudaCheck(cudaStreamWaitEvent(d2h_stream, pack_d2h_events[i], 0));

          // Transfer ghosts from device to host when packing kernel completes
          cudaCheck(cudaMemcpyAsync(h_send_ghosts[i], d_send_ghosts[i], ghost_sizes[i],
                cudaMemcpyDeviceToHost, d2h_stream));
        }
      }
    }

    // TODO: Use HAPI for AMPI
    if (param.use_zerocopy) {
      cudaStreamSynchronize(pack_stream);
    } else {
      cudaStreamSynchronize(d2h_stream);
    }
  }

  void exchangeGhosts() {
    // Swap pointers and advance iteration number
    std::swap(d_temperature, d_new_temperature);
    my_iter++;

    // Data sizes
    size_t data_sizes[DIR_COUNT]
      = {param.x_surf_count * sizeof(DataType), param.x_surf_count * sizeof(DataType),
         param.y_surf_count * sizeof(DataType), param.y_surf_count * sizeof(DataType),
         param.z_surf_count * sizeof(DataType), param.z_surf_count * sizeof(DataType)};

    // Send ghosts to neighbors
    MPI_Request send_requests[DIR_COUNT];
    int send_count = 0;
    for (int dir = 0; dir < DIR_COUNT; dir++) {
      DataType* send_ghost = param.use_zerocopy ? d_send_ghosts[dir] : h_send_ghosts[dir];
      int rev_dir = (dir % 2 == 0) ? (dir + 1) : (dir - 1);
      if (!bounds[dir])
        MPI_Isend(send_ghost, data_sizes[dir], MPI_CHAR, neighbor_ranks[dir],
            my_iter * DIR_COUNT + rev_dir, param.cart_comm, &send_requests[send_count++]);
    }

    // Receive ghosts from neighbors
    MPI_Request recv_requests[DIR_COUNT];
    int recv_count = 0;
    int recv_dirs[DIR_COUNT];
    for (int dir = 0; dir < DIR_COUNT; dir++) {
      DataType* recv_ghost = param.use_zerocopy ? d_recv_ghosts[dir] : h_recv_ghosts[dir];
      if (!bounds[dir]) {
        MPI_Irecv(recv_ghost, data_sizes[dir], MPI_CHAR, neighbor_ranks[dir],
            my_iter * DIR_COUNT + dir, param.cart_comm, &recv_requests[recv_count]);
        recv_dirs[recv_count++] = dir;
      }
    }

    // Invoke unpacking kernel for each received ghost
    MPI_Status recv_statuses[DIR_COUNT];
    for (int i = 0; i < recv_count; i++) {
      int index;
      // Wait for a ghost to arrive
      MPI_Waitany(recv_count, recv_requests, &index, recv_statuses);
      int dir = recv_dirs[index];

      if (!param.use_zerocopy) {
        // Copy ghost from host to device buffer
        cudaCheck(cudaMemcpyAsync(d_recv_ghosts[dir], h_recv_ghosts[dir],
              data_sizes[dir], cudaMemcpyHostToDevice, h2d_stream));

        // Enforce h2d -> unpack dependency
        cudaCheck(cudaEventRecord(h2d_unpack_events[dir], h2d_stream));
        cudaCheck(cudaStreamWaitEvent(unpack_stream, h2d_unpack_events[dir], 0));
      }

      // Unpack
      invokeUnpackingKernel(d_temperature, d_recv_ghosts[dir], dir,
          param.block_width, param.block_height, param.block_depth,
          unpack_stream);
    }

    // Wait for sends to complete
    MPI_Status send_statuses[DIR_COUNT];
    MPI_Waitall(send_count, send_requests, send_statuses);
  }
};

int main (int argc, char *argv[]) {
  int n_procs;
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Param param(n_procs, rank);

  // Process arguments
  int c;
  bool dims[3] = {false, false, false};
  while ((c = getopt(argc, argv, "c:x:y:z:i:w:dp")) != -1) {
    switch (c) {
      case 'c':
        param.n_chares = atoi(optarg);
        break;
      case 'x':
        param.grid_width = atoi(optarg);
        dims[0] = true;
        break;
      case 'y':
        param.grid_height = atoi(optarg);
        dims[1] = true;
        break;
      case 'z':
        param.grid_depth = atoi(optarg);
        dims[2] = true;
        break;
      case 'i':
        param.n_iters = atoi(optarg);
        break;
      case 'w':
        param.warmup_iters = atoi(optarg);
        break;
      case 'd':
        param.use_zerocopy = true;
        break;
      case 'p':
        param.print_elements = true;
        break;
      default:
        if (rank == 0) {
          fprintf(stderr,
              "Usage: %s -x [grid width] -y [grid height] -z [grid depth] "
              "-c [number of chares] -i [iterations] -w [warmup iterations] "
              "-d (use GPU zerocopy) -p (print blocks)\n", argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  // If only the X dimension is given, use it for Y and Z as well
  if (dims[0] && !dims[1] && !dims[2]) {
    param.grid_height = param.grid_depth = param.grid_width;
  }

  // Setup 3D grid of chares
  double area[3];
  int ipx, ipy, ipz, nremain;
  double surf, bestsurf;
  area[0] = param.grid_width * param.grid_height;
  area[1] = param.grid_width * param.grid_depth;
  area[2] = param.grid_height * param.grid_depth;
  bestsurf = 2.0 * (area[0] + area[1] + area[2]);
  ipx = 1;
  while (ipx <= param.n_chares) {
    if (param.n_chares % ipx == 0) {
      nremain = param.n_chares / ipx;
      ipy = 1;

      while (ipy <= nremain) {
        if (nremain % ipy == 0) {
          ipz = nremain / ipy;
          surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz;

          if (surf < bestsurf) {
            bestsurf = surf;
            param.n_chares_x = ipx;
            param.n_chares_y = ipy;
            param.n_chares_z = ipz;
          }
        }
        ipy++;
      }
    }
    ipx++;
  }

  if (param.n_chares_x * param.n_chares_y * param.n_chares_z != param.n_chares) {
    if (rank == 0) {
      fprintf(stderr, "ERROR: Bad grid of chares: %d x %d x %d != %d\n",
          param.n_chares_x, param.n_chares_y, param.n_chares_z, param.n_chares);
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // Calculate block size
  param.block_width  = param.grid_width  / param.n_chares_x;
  param.block_height = param.grid_height / param.n_chares_y;
  param.block_depth  = param.grid_depth  / param.n_chares_z;

  // Calculate surface count and sizes
  param.x_surf_count = param.block_height * param.block_depth;
  param.y_surf_count = param.block_width  * param.block_depth;
  param.z_surf_count = param.block_width  * param.block_height;
  param.x_surf_size  = param.x_surf_count * sizeof(DataType);
  param.y_surf_size  = param.y_surf_count * sizeof(DataType);
  param.z_surf_size  = param.z_surf_count * sizeof(DataType);

  // Print configuration
  if (rank == 0) {
    printf("\n[CUDA 3D Jacobi example]\n");
    printf("Grid: %d x %d x %d, Block: %d x %d x %d, Chares: %d x %d x %d, "
        "Iterations: %d, Warm-up: %d, Zerocopy: %d, Print: %d\n\n",
        param.grid_width, param.grid_height, param.grid_depth, param.block_width,
        param.block_height, param.block_depth, param.n_chares_x, param.n_chares_y,
        param.n_chares_z, param.n_iters, param.warmup_iters, param.use_zerocopy,
        param.print_elements);
  }

  // Create 3D Cartesian topology and Block object
  int proc_dims[NDIMS] = {param.n_chares_x, param.n_chares_y, param.n_chares_z};
  int periods[NDIMS] = {0, 0, 0};
  int coords[NDIMS];
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, proc_dims, periods, 0, &param.cart_comm);
  MPI_Cart_coords(param.cart_comm, rank, NDIMS, coords);
  Block block(param, coords[0], coords[1], coords[2]);

  // Initialize block
  if (rank == 0) printf("Initializing blocks...\n");

  block.init();

  if (rank == 0) printf("Blocks initialized\n");

  MPI_Barrier(MPI_COMM_WORLD);

  // Main iteration loop
  if (rank == 0) printf("Running iterations...\n");
  double start_time;
  double comm_time;
  double comm_start_time;
  for (int i = 0; i < param.n_iters + param.warmup_iters; i++) {
    if (i == param.warmup_iters) start_time = MPI_Wtime();

    block.updateAndPack();

    comm_start_time = MPI_Wtime();

    block.exchangeGhosts();

    if (i >= param.warmup_iters) comm_time += MPI_Wtime() - comm_start_time;
  }
  double total_time = MPI_Wtime() - start_time;

  // Finalize
  if (rank == 0) {
    printf("Completed %d iterations (%d warmup), "
        "total time: %.3lf s (avg %.3lf ms), "
        "comm time: %.3lf s (avg %.3lf ms)\n",
        param.n_iters, param.warmup_iters,
        total_time, total_time / param.n_iters * 1e3,
        comm_time, comm_time / param.n_iters * 1e3);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
