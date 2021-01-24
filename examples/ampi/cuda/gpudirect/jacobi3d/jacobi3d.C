#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "jacobi3d.h"

int main (int argc, char *argv[]) {
  // Set default values
  int n_chares = 1;
  int n_chares_x = 1;
  int n_chares_y = 1;
  int n_chares_z = 1;
  int grid_width = 512;
  int grid_height = 512;
  int grid_depth = 512;
  int block_width = -1;
  int block_height = -1;
  int block_depth = -1;
  int x_surf_count;
  int y_surf_count;
  int z_surf_count;
  size_t x_surf_size;
  size_t y_surf_size;
  size_t z_surf_size;
  int n_iters = 100;
  int warmup_iters = 10;
  int use_zerocopy = false;
  bool print_elements = false;
  int my_iter = 0;

  // Process arguments
  int c;
  bool dims[3] = {false, false, false};
  while ((c = getopt(argc, argv, "c:x:y:z:i:w:dp")) != -1) {
    switch (c) {
      case 'c':
        n_chares = atoi(optarg);
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
        fprintf(stderr,
            "Usage: %s -x [grid width] -y [grid height] -z [grid depth] "
            "-c [number of chares] -i [iterations] -w [warmup iterations] "
            "-d (use GPU zerocopy) -p (print blocks)\n", argv[0]);
        exit(EXIT_FAILURE);
    }
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
  while (ipx <= n_chares) {
    if (n_chares % ipx == 0) {
      nremain = n_chares / ipx;
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

  if (n_chares_x * n_chares_y * n_chares_z != n_chares) {
    fprintf(stderr, "ERROR: Bad grid of chares: %d x %d x %d != %d\n",
        n_chares_x, n_chares_y, n_chares_z, n_chares);
    exit(EXIT_FAILURE);
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

  int n_procs;
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Print configuration
  if (rank == 0) {
    printf("\n[CUDA 3D Jacobi example]\n");
    printf("Grid: %d x %d x %d, Block: %d x %d x %d, Chares: %d x %d x %d, "
        "Iterations: %d, Warm-up: %d, Zerocopy: %d, Print: %d\n\n",
        grid_width, grid_height, grid_depth, block_width, block_height, block_depth,
        n_chares_x, n_chares_y, n_chares_z, n_iters, warmup_iters, use_zerocopy,
        print_elements);
  }

  if (n_procs != n_chares) {
    if (rank == 0) {
      fprintf(stderr, "Number of chares should be the same as the number of processes!\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
