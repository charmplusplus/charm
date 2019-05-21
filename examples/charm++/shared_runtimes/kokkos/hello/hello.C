#include <stdio.h>
#include "hello.decl.h"
#include "pup_stl.h"
#include <vector>
#include <string>

#include <Kokkos_Core.hpp>
#include <typeinfo>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Hello helloProxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    // Pack arguments (optional)
    std::vector<std::string> args;
    for (int i = 0; i < m->argc; i++) {
      args.push_back(std::string(m->argv[i]));
    }

    // Create nodegroup and run
    helloProxy = CProxy_Hello::ckNew(m->argc, args);
    helloProxy.run();
  };

  void done() {
    CkPrintf("All done\n");

    CkExit();
  };
};

class Hello : public CBase_Hello {
public:
  Hello(int argc, std::vector<std::string> args) {
    char* argv[argc];
    for (int i = 0; i < argc; i++) {
      argv[i] = const_cast<char*>(args[i].c_str());
    }

    // Initialize Kokkos. Needs to be done on every process
    Kokkos::initialize(argc, argv);
  }

  void run() {
    CkPrintf("Hello World on Kokkos execution space %s\n",
             typeid(Kokkos::DefaultExecutionSpace).name());

    // Parallel execution with Kokkos
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::parallel_for(16, KOKKOS_LAMBDA (const int i) {
      printf("Hello from i = %i\n", i);
    });
#endif

    // Finialize Kokkos. Needs to be done on every process
    Kokkos::finalize();

    // Reduce to Main to end the program
    CkCallback cb(CkReductionTarget(Main, done), mainProxy);
    contribute(cb);
  }
};

#include "hello.def.h"
