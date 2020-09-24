#include <stdio.h>
#include "hapi.h"
#include "cudacallback.decl.h"

#define WARMUP_ITERS 1

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Worker workers;
/* readonly */ int iterations;
/* readonly */ int data_count;
/* readonly */ int offload_pe;
/* readonly */ bool offload;

extern void kernelSetup(cudaStream_t stream, void* cb);

class Main : public CBase_Main {
 public:
  Main(CkArgMsg* m) {
    // Default values
    mainProxy = thisProxy;
    iterations = 1000;
    data_count = 1048576;
    offload_pe = 0;
    offload = false;

    // Process arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "i:n:p:g")) != -1) {
      switch (c) {
        case 'i':
          iterations = atoi(optarg);
          break;
        case 'n':
          data_count = atoi(optarg);
          break;
        case 'p':
          offload_pe = atoi(optarg);
          break;
        case 'g':
          offload = true;
          break;
        default:
          CkPrintf("Usage: %s -i [iterations] -n [data count] -g (GPU offload)\n", m->argv[0]);
          CkExit();
      }
    }
    delete m;

    if (offload_pe >= CkNumPes()) {
      CkPrintf("Error: invalid PE (%d) used for GPU offloading\n", offload_pe);
      CkExit(1);
    }

    CkPrintf("\n[CUDA Callback Overhead]\n"
        "Iterations: %d, data count: %d, offload: PE %d (%s)\n\n",
        iterations, data_count, offload_pe, offload ? "yes" : "no");

    workers = CProxy_Worker::ckNew();
    workers.run();
  };

  void done() {
    CkPrintf("\nAll done\n");
    CkExit();
  }
};

class Worker : public CBase_Worker {
  int iteration;
  cudaStream_t stream;
  double* data;
  double start_time;

 public:
  Worker() {
    iteration = 0;
    hapiCheck(cudaStreamCreate(&stream));
    data = (double*)malloc(sizeof(double) * data_count);
    if (data == nullptr) {
      CkPrintf("Error: malloc failure\n");
      CkExit(1);
    }
  }

  ~Worker() {
    hapiCheck(cudaStreamDestroy(stream));
    free(data);
  }

  void run() {
    if (iteration == WARMUP_ITERS) {
      start_time = CkWallTimer();
    }

    if (thisIndex == offload_pe) {
      // Selected PE offloads GPU work
      if (offload) {
        CkCallback* cb = new CkCallback(CkIndex_Worker::kernelDone(), thisProxy[thisIndex]);
        kernelSetup(stream, (void*)cb);
      } else {
        kernelDone();
      }
    } else {
      // Other PEs perform CPU computation
      for (int i = 0; i < data_count; i++) {
        data[i] = (double)i / 11;
      }

      kernelDone();
    }
  }

  void kernelDone() {
    // Continue next iteration or return to Main
    if (iteration++ < iterations) {
      thisProxy[thisIndex].run();
    } else {
      double total_time = CkWallTimer() - start_time;
      CkPrintf("PE %d: %.3lf us average\n", CkMyPe(), (total_time * 1e6) / (iterations - WARMUP_ITERS));
      contribute(CkCallback(CkReductionTarget(Main, done), mainProxy));
    }
  }
};

#include "cudacallback.def.h"
