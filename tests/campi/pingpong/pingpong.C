// campi test: blocking point-to-point between ranks 0 and 1 (verified
// payloads, several sizes), then an Allreduce and a Barrier over all ranks.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "mpi.h"
#include "campi.h"

CAMPI_MAIN(argc, argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    if (rank == 0) printf("campi pingpong: needs at least 2 ranks\n");
    MPI_Finalize();
    co_return 1;
  }

  const int iters = 50;
  const int sizes[] = {1, 16, 1024, 65536, 1 << 20}; // ints; the largest takes AMPI's rendezvous path
  long errors = 0;
  for (int n : sizes) {
    std::vector<int> buf(n);
    for (int it = 0; it < iters; it++) {
      if (rank == 0) {
        for (int i = 0; i < n; i++) buf[i] = it * 7 + i;
        MPI_Send(buf.data(), n, MPI_INT, 1, it, MPI_COMM_WORLD);
        MPI_Status sts;
        MPI_Recv(buf.data(), n, MPI_INT, 1, it, MPI_COMM_WORLD, &sts);
        for (int i = 0; i < n; i++)
          if (buf[i] != it * 7 + i + 1) { errors++; break; }
        if (sts.MPI_SOURCE != 1 || sts.MPI_TAG != it) errors++;
      } else if (rank == 1) {
        MPI_Status sts;
        MPI_Recv(buf.data(), n, MPI_INT, 0, it, MPI_COMM_WORLD, &sts);
        for (int i = 0; i < n; i++)
          if (buf[i] != it * 7 + i) { errors++; break; }
        for (int i = 0; i < n; i++) buf[i] += 1;
        MPI_Send(buf.data(), n, MPI_INT, 0, it, MPI_COMM_WORLD);
      }
    }
  }

  long total = 0;
  MPI_Allreduce(&errors, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  int ranksum = 0;
  MPI_Allreduce(&rank, &ranksum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (ranksum != size * (size - 1) / 2) total++;
  if (rank == 0)
    printf("campi pingpong: %d ranks, %d sizes x %d iterations, errors=%ld: %s\n",
           size, (int)(sizeof(sizes) / sizeof(sizes[0])), iters, total,
           total == 0 ? "PASSED" : "FAILED");
  MPI_Finalize();
  co_return total == 0 ? 0 : 1;
}
