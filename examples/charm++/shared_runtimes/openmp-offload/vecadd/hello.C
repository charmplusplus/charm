#include <stdio.h>
#include "hello.decl.h"
#include <omp.h>
#include <typeinfo>

/* readonly */ CProxy_Main mainProxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {

    int *A = (int*) calloc(1024, sizeof(int));

    CkPrintf("Hello World with OpenMP offloading. %d device(s) available.\n", omp_get_num_devices());

    #pragma omp target teams distribute parallel for map(tofrom: A[:1024])
    for (int i=0; i<1024; i++) {
      A[i] = 42;
    }

    // Check that results are correct on the host
    for (int i=0; i<1024; i++) {
      if (A[i] != 42)
        CkPrintf("Incorrect value calculated: %d.\n", A[i]);
    }
   
    free(A);

    done();
  };

  void done()
  {
    CkPrintf("All done.\n");
    CkExit();
  };
};

#include "hello.def.h"
