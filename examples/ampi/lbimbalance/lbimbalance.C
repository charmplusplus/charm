/*
 * Load imbalance with AMPI_Migrate and a measurement-based balancer.
 *
 * Each rank is one element of a 1D array of n ranks, in a ring. Every step it
 * exchanges a small message with its left and right neighbors and then spins
 * for its load (in microseconds). Every -lbperiod steps it calls
 * AMPI_Migrate(AMPI_INFO_LB_SYNC); with "+balancer <LB>" the runtime moves
 * ranks between PEs, and without one the call returns at once.
 *
 * Two load patterns:
 *
 *  -pattern tri    (slowly varying) rank i has load a + a*m*i/n. The slope m
 *                  goes linearly from -k to +k over the run. For m < 0 the
 *                  heavy end is rank 0: load a + a*|m|*(n-1-i)/n, so loads
 *                  stay positive for any k. At m = 0 the load is even.
 *
 *  -pattern pivot  (slowly varying, constant total) rank i has load
 *                  c + a*m*(i - (n-1)/2)/n with c = a*(1 + k/2): the line turns
 *                  around the center rank as m goes from -k to +k, the total
 *                  load stays n*c, and the lightest rank has load a at |m| = k.
 *
 *  -pattern spike  (adaptive refinement) every -spikeperiod steps a new random
 *                  subset of -spikefrac of the ranks gets -spikefactor times
 *                  the base load; the others have load a.
 *
 * Start with a block mapping (+mapping BLOCK_MAP), so neighbors share a PE.
 *
 * After each AMPI_Migrate rank 0 prints, for the steps since the previous one:
 * the time per step, the ideal time per step (total load / number of PEs), the
 * largest PE load divided by the average PE load, and the number of ranks that
 * moved to another PE.
 *
 * Usage: ./lbimbalance +vp 64 +pe 4 +mapping BLOCK_MAP [+balancer GreedyLB]
 *          [-pattern tri|pivot|spike] [-steps 400] [-lbperiod 20] [-a 1000] [-k 3]
 *          [-spikefrac 0.1] [-spikefactor 4] [-spikeperiod 100]
 */

#include <mpi.h>
#include <cstdio>
#include <vector>
#include "lbimbalance.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, n;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  Params p = parseArgs(argc, argv);
  int npes = numPes();
  int left = (rank + n - 1) % n, right = (rank + 1) % n;

  if (rank == 0)
    printf("lbimbalance: %d ranks on %d PEs, pattern %s, %d steps, LB every %d, "
           "a=%g us, %s\n",
           n, npes, p.spike ? "spike" : p.pivot ? "pivot" : "tri", p.steps,
           p.lbperiod, p.a, p.spike ? "spikes" : "triangle");
  if (rank == 0 && p.spike)
    printf("  spikefrac %g, spikefactor %g, spikeperiod %d\n", p.spikefrac,
           p.spikefactor, p.spikeperiod);
  if (rank == 0 && !p.spike) printf("  k %g\n", p.k);

  int errors = 0;
  double intervalLoad = 0.0; // microseconds spun since the last report
  std::vector<double> peLoad(npes), peLoadSum(npes);
  double totalTime = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  int first = 0;

  for (int step = 0; step < p.steps; step++) {
    int out[2] = {rank, step}, fromLeft[2], fromRight[2];
    MPI_Request req[4];
    MPI_Irecv(fromLeft, 2, MPI_INT, left, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(fromRight, 2, MPI_INT, right, 1, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(out, 2, MPI_INT, right, 0, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(out, 2, MPI_INT, left, 1, MPI_COMM_WORLD, &req[3]);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    if (fromLeft[0] != left || fromLeft[1] != step || fromRight[0] != right ||
        fromRight[1] != step)
      errors++;

    double l = loadOf(p, rank, n, step);
    spin(l);
    intervalLoad += l;

    if ((step + 1) % p.lbperiod == 0 || step + 1 == p.steps) {
      // Time for this interval: all ranks finish their steps, then measure.
      MPI_Barrier(MPI_COMM_WORLD);
      double t1 = MPI_Wtime();
      int pe = myPe();
      std::fill(peLoad.begin(), peLoad.end(), 0.0);
      peLoad[pe] = intervalLoad;
      MPI_Reduce(peLoad.data(), peLoadSum.data(), npes, MPI_DOUBLE, MPI_SUM, 0,
                 MPI_COMM_WORLD);

      AMPI_Migrate(AMPI_INFO_LB_SYNC);

      int moved = myPe() != pe, movedSum = 0;
      MPI_Reduce(&moved, &movedSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        int nsteps = step + 1 - first;
        double sum = 0, mx = 0;
        for (double v : peLoadSum) {
          sum += v;
          if (v > mx) mx = v;
        }
        double per = (t1 - t0) / nsteps;
        totalTime += t1 - t0;
        printf("steps %4d-%4d: %8.3f ms/step  ideal %8.3f  max/avg PE load "
               "%5.2f  moved %d\n",
               first, step, per * 1e3, sum / npes / nsteps * 1e-3,
               mx / (sum / npes), movedSum);
        fflush(stdout);
      }
      intervalLoad = 0.0;
      first = step + 1;
      MPI_Barrier(MPI_COMM_WORLD);
      t0 = MPI_Wtime();
    }
  }

  int errSum = 0;
  MPI_Reduce(&errors, &errSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("total stepping time %.3f s, %s\n", totalTime,
           errSum ? "MESSAGE ERRORS" : "messages correct");
    if (errSum) printf("%d wrong neighbor messages\n", errSum);
  }
  MPI_Finalize();
  return errSum != 0;
}
