/* Modified from the OSU Latency Benchmark */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuda_runtime.h>

#define LARGE_MESSAGE_SIZE 8192
#define FIELD_WIDTH 20
#define FLOAT_PRECISION 2

int main (int argc, char *argv[]) {
  int myid, numprocs, i, j;
  int size;
  char *s_buf, *r_buf;
  char *s_buf_host, *r_buf_host;
  double t_start = 0.0, t_end = 0.0, t = 0.0;
  int po_ret = 0;

  size_t min_size = 1;
  size_t max_size = 4194304;
  int n_iters_reg = 1000;
  int n_iters_large = 100;
  int warmup_iters = 10;
  int cuda_aware = 0;
  int window_size = 64;

  // Process command line arguments
  int c;
  while ((c = getopt(argc, argv, "s:x:i:l:w:gd:")) != -1) {
    switch (c) {
      case 's':
        min_size = atoi(optarg);
        break;
      case 'x':
        max_size = atoi(optarg);
        break;
      case 'i':
        n_iters_reg = atoi(optarg);
        break;
      case 'l':
        n_iters_large = atoi(optarg);
        break;
      case 'w':
        warmup_iters = atoi(optarg);
        break;
      case 'g':
        cuda_aware = 1;
        break;
      case 'd':
        window_size = atoi(optarg);
        break;
      default:
        fprintf(stderr, "Unknown command line argument detected\n");
        exit(EXIT_FAILURE);
    }
  }

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (numprocs != 2) {
    if (myid == 0) {
      fprintf(stderr, "This test requires exactly two processes\n");
    }

    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  if (myid == 0) {
    fprintf(stdout, "# AMPI OSU Bandwidth Benchmark\n"
        "# Message sizes: %lu - %lu bytes\n"
        "# Iterations: %d regular, %d large\n"
        "# Warmup: %d\n"
        "# CUDA-aware: %d\n"
        "# Window size: %d\n",
        min_size, max_size, n_iters_reg, n_iters_large, warmup_iters, cuda_aware, window_size);
    fprintf(stdout, "%-*s%*s\n", FIELD_WIDTH, "Message size", FIELD_WIDTH, "Bandwidth (MB/s)");
  }

  cudaMalloc((void**)&s_buf, max_size);
  cudaMalloc((void**)&r_buf, max_size);
  if (!cuda_aware) {
    cudaMallocHost((void**)&s_buf_host, max_size);
    cudaMallocHost((void**)&r_buf_host, max_size);
  }
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  MPI_Request request[window_size];
  MPI_Status reqstat[window_size];

  for (size = min_size; size <= max_size; size = (size ? size * 2 : 1)) {
    cudaMemset(s_buf, 'a', size);
    cudaMemset(r_buf, 'b', size);

    int n_iters = (size > LARGE_MESSAGE_SIZE) ? n_iters_large : n_iters_reg;

    MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0) {
      for (i = 0; i < n_iters + warmup_iters; i++) {
        if (i == warmup_iters) {
          t_start = MPI_Wtime();
        }

        if (cuda_aware) {
          for (j = 0; j < window_size; j++) {
            MPI_Isend(s_buf, size, MPI_CHAR, 1, 100, MPI_COMM_WORLD, request + j);
          }

          MPI_Waitall(window_size, request, reqstat);

          MPI_Recv(r_buf, 4, MPI_CHAR, 1, 101, MPI_COMM_WORLD, &reqstat[0]);
        } else {
          for (j = 0; j < window_size; j++) {
            cudaMemcpyAsync(s_buf_host, s_buf, size, cudaMemcpyDeviceToHost, stream);
          }
          cudaStreamSynchronize(stream);

          for (j = 0; j < window_size; j++) {
            MPI_Isend(s_buf_host, size, MPI_CHAR, 1, 100, MPI_COMM_WORLD, request + j);
          }

          MPI_Waitall(window_size, request, reqstat);

          MPI_Recv(r_buf_host, 4, MPI_CHAR, 1, 101, MPI_COMM_WORLD, &reqstat[0]);
          cudaMemcpy(r_buf, r_buf_host, 4, cudaMemcpyHostToDevice);
        }
      }

      t_end = MPI_Wtime();
      t = t_end - t_start;
    } else if (myid == 1) {
      for (i = 0; i < n_iters + warmup_iters; i++) {
        if (cuda_aware) {
          for (j = 0; j < window_size; j++) {
            MPI_Irecv(r_buf, size, MPI_CHAR, 0, 100, MPI_COMM_WORLD, request + j);
          }

          MPI_Waitall(window_size, request, reqstat);

          MPI_Send(s_buf, 4, MPI_CHAR, 0, 101, MPI_COMM_WORLD);
        } else {
          for (j = 0; j < window_size; j++) {
            MPI_Irecv(r_buf_host, size, MPI_CHAR, 0, 100, MPI_COMM_WORLD, request + j);
          }

          MPI_Waitall(window_size, request, reqstat);

          for (j = 0; j < window_size; j++) {
            cudaMemcpyAsync(r_buf, r_buf_host, size, cudaMemcpyHostToDevice, stream);
          }
          cudaStreamSynchronize(stream);

          cudaMemcpy(s_buf_host, s_buf, 4, cudaMemcpyDeviceToHost);
          MPI_Send(s_buf_host, 4, MPI_CHAR, 0, 101, MPI_COMM_WORLD);
        }
      }
    }

    if (myid == 0) {
      double tmp = size / 1e6 * n_iters * window_size;

      fprintf(stdout, "%-*d%*.*f\n", FIELD_WIDTH, size, FIELD_WIDTH, FLOAT_PRECISION, tmp / t);
      fflush(stdout);
    }
  }

  cudaFree(s_buf);
  cudaFree(r_buf);
  if (!cuda_aware) {
    cudaFreeHost(s_buf_host);
    cudaFreeHost(r_buf_host);
  }
  cudaStreamDestroy(stream);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
