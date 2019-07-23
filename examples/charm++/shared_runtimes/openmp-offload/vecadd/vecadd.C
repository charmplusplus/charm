#include <stdio.h>
#include "vecadd.decl.h"
#include <omp.h>
#include <typeinfo>

#if !defined _OPENMP || ! _OPENMP >= 201307
#error This file requires compiler support for OpenMP 4.0+.
#endif

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Process processProxy;
/* readonly */ int device_cnt;
/* readonly */ long n;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    n = 128 * 1024 * 1024; // 128 M doubles by default

    // Command line parsing
    int c;
    while ((c = getopt(m->argc, m->argv, "n")) != -1) {
      switch (c) {
        case 'n':
          n = atoi(optarg);
          break;
        default:
          CkPrintf("Unknown argument '%c', exiting.", c);
          CkExit(1);
      }
    }

    CkPrintf("\n[OpenMP offloading + Charm++ Vector Addition]\n");
    CkPrintf("Vector size: %lu doubles\n", n);

    int device_cnt = omp_get_num_devices();
    if (device_cnt <= 0) {
      CkPrintf("No OpenMP offloading-capable device found, exiting.");
      CkExit(2);
    }

    // Create nodegroup and run
    processProxy = CProxy_Process::ckNew();
    processProxy.run();
  };

  void done()
  {
    CkPrintf("All done.\n");
    CkExit();
  };
};


class Process : public CBase_Process {
public:
  Process() {
    // Figure out which GPU this process should be mapped to in round-robin.
    int processes_per_node = CkNumNodes() / CmiNumPhysicalNodes();
    int local_pid = CkMyNode() % processes_per_node;
    int my_gpu = local_pid % device_cnt;

    omp_set_default_device(my_gpu);
    CkPrintf("[PE %d] Using device %d.\n", CkMyPe(), my_gpu);
  }

  void run() {
    double *A = (double*) malloc(n * sizeof(double));
    double *B = (double*) malloc(n * sizeof(double));

    // Initialize vectors
    for (long i=0; i<n; i++) {
      A[i] = 1.0;
      B[i] = 2.0;
    }

    // Run vector addition
    #pragma omp target teams distribute parallel for map(tofrom: A[:n]) map(to: B[:n])
    for (long i=0; i<n; i++) {
      A[i] += B[i];
    }

    // Check that results are correct on the host
    for (long i=0; i<n; i++) {
      if (abs(A[i] - 3.0) > 1e-15)
        CkPrintf("Incorrect value calculated: A[%ld] = %lf.\n", i, A[i]);
    }

    free(A);
    free(B);

    // Reduce to Main to end the program
    CkCallback cb(CkReductionTarget(Main, done), mainProxy);
    contribute(cb);
  }
};

#include "vecadd.def.h"
