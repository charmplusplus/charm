/*
 * The campi version of lbimbalance.C: the same ring, load patterns, arguments
 * and output, with each rank a campi coroutine instead of a thread.
 *
 * Everything a rank needs after the AMPI_Migrate site is in its State object,
 * because a rank that moves to another PE is rebuilt there from State and
 * re-enters main at that site.
 *
 * Usage: ./lbimbalance-campi +vp 64 +pe 4 +mapping BLOCK_MAP
 *          [+balancer GreedyLB] [lbimbalance.C's options]
 */

#include <mpi.h>
#include <cstdio>
#include <vector>
#include "lbimbalance.h"
#include "campi.h"

struct State {
  Params p;
  int rank = 0, n = 0, npes = 1;
  int step = 0, first = 0;
  int errors = 0;
  int pe = 0;                // PE before the AMPI_Migrate site
  double intervalLoad = 0.0; // microseconds spun since the last report
  double t0 = 0.0, t1 = 0.0, totalTime = 0.0;
  std::vector<double> peLoadSum;

  void pup(PUP::er &er) {
    er((char *)&p, sizeof(p));
    er | rank; er | n; er | npes;
    er | step; er | first;
    er | errors; er | pe;
    er | intervalLoad;
    er | t0; er | t1; er | totalTime;
    er | peLoadSum;
  }
};

CAMPI_MAIN_STATE(State, argc, argv, s)
{
  // Scratch values within one step; none is live across the migrate site.
  int out[2], fromLeft[2], fromRight[2], left, right, moved, movedSum, errSum;
  MPI_Request req[4];
  std::vector<double> peLoad;

  CAMPI_MIGRATABLE {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &s.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.n);
    s.p = parseArgs(argc, argv);
    s.npes = numPes();
    s.peLoadSum.resize(s.npes);

    if (s.rank == 0)
      printf("lbimbalance-campi: %d ranks on %d PEs, pattern %s, %d steps, "
             "LB every %d, a=%g us\n",
             s.n, s.npes, s.p.spike ? "spike" : s.p.pivot ? "pivot" : "tri",
             s.p.steps, s.p.lbperiod, s.p.a);

    MPI_Barrier(MPI_COMM_WORLD);
    s.t0 = MPI_Wtime();

    for (s.step = 0; s.step < s.p.steps; s.step++) {
      left = (s.rank + s.n - 1) % s.n;
      right = (s.rank + 1) % s.n;
      out[0] = s.rank;
      out[1] = s.step;
      MPI_Irecv(fromLeft, 2, MPI_INT, left, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(fromRight, 2, MPI_INT, right, 1, MPI_COMM_WORLD, &req[1]);
      MPI_Isend(out, 2, MPI_INT, right, 0, MPI_COMM_WORLD, &req[2]);
      MPI_Isend(out, 2, MPI_INT, left, 1, MPI_COMM_WORLD, &req[3]);
      MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
      if (fromLeft[0] != left || fromLeft[1] != s.step ||
          fromRight[0] != right || fromRight[1] != s.step)
        s.errors++;

      {
        double l = loadOf(s.p, s.rank, s.n, s.step);
        spin(l);
        s.intervalLoad += l;
      }

      if ((s.step + 1) % s.p.lbperiod == 0 || s.step + 1 == s.p.steps) {
        MPI_Barrier(MPI_COMM_WORLD);
        s.t1 = MPI_Wtime();
        s.pe = myPe();
        peLoad.assign(s.npes, 0.0);
        peLoad[s.pe] = s.intervalLoad;
        MPI_Reduce(peLoad.data(), s.peLoadSum.data(), s.npes, MPI_DOUBLE,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        CAMPI_MIGRATE(AMPI_INFO_LB_SYNC);

        moved = myPe() != s.pe;
        movedSum = 0;
        MPI_Reduce(&moved, &movedSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (s.rank == 0) {
          int nsteps = s.step + 1 - s.first;
          double sum = 0, mx = 0;
          for (double v : s.peLoadSum) {
            sum += v;
            if (v > mx) mx = v;
          }
          // t0 and t1 were both read by rank 0 between two migrate sites,
          // so from the clock of one process.
          s.totalTime += s.t1 - s.t0;
          printf("steps %4d-%4d: %8.3f ms/step  ideal %8.3f  max/avg PE load "
                 "%5.2f  moved %d\n",
                 s.first, s.step, (s.t1 - s.t0) / nsteps * 1e3,
                 sum / s.npes / nsteps * 1e-3, mx / (sum / s.npes), movedSum);
          fflush(stdout);
        }
        s.intervalLoad = 0.0;
        s.first = s.step + 1;
        MPI_Barrier(MPI_COMM_WORLD);
        s.t0 = MPI_Wtime();
      }
    }

    errSum = 0;
    MPI_Reduce(&s.errors, &errSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (s.rank == 0) {
      printf("total stepping time %.3f s, %s\n", s.totalTime,
             errSum ? "MESSAGE ERRORS" : "messages correct");
      if (errSum) printf("%d wrong neighbor messages\n", errSum);
    }
    MPI_Finalize();
    co_return errSum != 0;
  }
  co_return 0;
}
