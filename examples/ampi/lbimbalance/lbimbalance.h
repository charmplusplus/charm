// Load patterns and helpers shared by lbimbalance.C (thread AMPI) and
// lbimbalance-campi.C (campi). See lbimbalance.C for the patterns.
#ifndef LBIMBALANCE_H
#define LBIMBALANCE_H

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Params {
  bool spike = false;
  bool pivot = false;
  int steps = 400;
  int lbperiod = 20;
  double a = 1000.0; // base load, microseconds
  double k = 3.0;
  double spikefrac = 0.1;
  double spikefactor = 4.0;
  int spikeperiod = 100;
};

static Params parseArgs(int argc, char **argv) {
  Params p;
  for (int i = 1; i + 1 < argc; i += 2) {
    const char *f = argv[i], *v = argv[i + 1];
    if (!strcmp(f, "-pattern")) {
      p.spike = !strcmp(v, "spike");
      p.pivot = !strcmp(v, "pivot");
    }
    else if (!strcmp(f, "-steps")) p.steps = atoi(v);
    else if (!strcmp(f, "-lbperiod")) p.lbperiod = atoi(v);
    else if (!strcmp(f, "-a")) p.a = atof(v);
    else if (!strcmp(f, "-k")) p.k = atof(v);
    else if (!strcmp(f, "-spikefrac")) p.spikefrac = atof(v);
    else if (!strcmp(f, "-spikefactor")) p.spikefactor = atof(v);
    else if (!strcmp(f, "-spikeperiod")) p.spikeperiod = atoi(v);
  }
  return p;
}

// A small random generator that gives every rank the same sequence for the
// same seed, so all ranks agree on the spiked subset without communicating.
static unsigned long long mix(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static double loadOf(const Params &p, int rank, int n, int step) {
  if (!p.spike) {
    double m = p.steps > 1 ? -p.k + 2.0 * p.k * step / (p.steps - 1) : 0.0;
    if (p.pivot) return p.a * (1 + p.k / 2) + p.a * m * (rank - (n - 1) / 2.0) / n;
    double pos = m >= 0 ? rank : (n - 1 - rank);
    return p.a + p.a * std::fabs(m) * pos / n;
  }
  unsigned long long epoch = step / p.spikeperiod;
  double u = (mix(epoch * 1000003ULL + rank) >> 11) * (1.0 / 9007199254740992.0);
  return u < p.spikefrac ? p.a * p.spikefactor : p.a;
}

static void spin(double us) {
  double end = MPI_Wtime() + us * 1e-6;
  while (MPI_Wtime() < end) {
  }
}

static int myPe() {
  int *v, flag;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_MY_WTH, &v, &flag);
  return flag ? *v : 0;
}

static int numPes() {
  int *v, flag;
  MPI_Comm_get_attr(MPI_COMM_WORLD, AMPI_NUM_WTHS, &v, &flag);
  return flag ? *v : 1;
}

#endif
