#include <stdio.h>
#include "hello.decl.h"
#include <omp.h>
#include <typeinfo>

/* readonly */ CProxy_Main mainProxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    int n = 128 * 1024 * 1024; // 128 M doubles by default

    double *A = (double*) calloc(n, sizeof(double));
    double *B = (double*) calloc(n, sizeof(double));

    for (long i=0; i<n; i++) {
      A[i] = 1.0;
      B[i] = 2.0;
    }

    CkPrintf("Hello World with OpenMP offloading. %d device(s) available.\n", omp_get_num_devices());

    #pragma omp target teams distribute parallel for map(tofrom: A[:n]) map(to: B[:n])
    for (long i=0; i<n; i++) {
      A[i] += B[i];
    }

    // Check that results are correct on the host
    for (long i=0; i<n; i++) {
      if (A[i] != 3.0)
        CkPrintf("Incorrect value calculated: A[%ld] = %lf.\n", i, A[i]);
    }
   
    free(A);
    free(B);

    done();
  };

  void done()
  {
    CkPrintf("All done.\n");
    CkExit();
  };
};

#include "hello.def.h"
