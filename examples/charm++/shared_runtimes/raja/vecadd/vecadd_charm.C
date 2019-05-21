#include "vecadd.decl.h"
#include "pup_stl.h"
#include "vecadd.h"
#include <unistd.h>
#include <cuda_runtime.h>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Process processProxy;
/* readonly */ uint64_t n;
/* readonly */ bool use_gpu;
/* readonly */ int device_cnt;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    n = 128 * 1024 * 1024; // 128 M doubles by default
    use_gpu = false;

    // Command line parsing
    int c;
    while ((c = getopt(m->argc, m->argv, "n:g")) != -1) {
      switch (c) {
        case 'n':
          n = atoi(optarg);
          break;
        case 'g':
          use_gpu = true;
          break;
        default:
          CkExit();
      }
    }

    CkPrintf("\n[Raja + Charm++ Vector Addition]\n");
    CkPrintf("Vector size: %lu doubles\n", n);
    CkPrintf("Use GPU: %s\n\n", use_gpu ? "Yes" : "No");

    // Check for GPUs
    cudaGetDeviceCount(&device_cnt);
    if (use_gpu && device_cnt <= 0) {
      CkPrintf("CUDA capable devices not found, exiting...\n");
      CkExit();
    }

    // Create nodegroup and run
    processProxy = CProxy_Process::ckNew();
    processProxy.run();
  };

  void done() {
    CkPrintf("\nAll done\n");

    CkExit();
  };
};

class Process : public CBase_Process {
public:
  Process() {}

  void run() {
    // Run vector addition
    vecadd(n, CkMyNode(), use_gpu);

    // Reduce to Main to end the program
    CkCallback cb(CkReductionTarget(Main, done), mainProxy);
    contribute(cb);
  }
};

#include "vecadd.def.h"
