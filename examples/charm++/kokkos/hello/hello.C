#include <stdio.h>
#include "hello.decl.h"
#include <Kokkos_Core.hpp>
#include <typeinfo>

/* readonly */ CProxy_Main mainProxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    // Initialize Kokkos
    Kokkos::initialize(m->argc, m->argv);

    CkPrintf("Hello World on Kokkos execution space %s\n",
             typeid(Kokkos::DefaultExecutionSpace).name());

    // Parallel execution with Kokkos
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::parallel_for(16, KOKKOS_LAMBDA (const int i) {
      printf("Hello from i = %i\n", i);
    });
#endif

    done();
  };

  void done()
  {
    CkPrintf("All done\n");

    // Finalize Kokkos and exit
    Kokkos::finalize();
    CkExit();
  };
};

#include "hello.def.h"
