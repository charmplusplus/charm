#include <stdio.h>
#include "hapi.h"
#include "qdtest.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_QD qdProxy;
/* readonly */ int n_chares;
/* readonly */ int n_iters;
/* readonly */ float busy_time;
/* readonly */ long long kernel_clock_count;
/* readonly */ size_t data_size;
/* reaodnly */ bool qd_off;

extern void kernelSetup(char* h_A, char* h_B, char* d_A, char* d_B, size_t data_size,
                        long long clock_count, cudaStream_t stream, void* h2d_cb,
                        void* kernel_cb, void* d2h_cb);

/* mainchare */
class Main : public CBase_Main {
  int cur_iter;

public:
  Main(CkArgMsg* m) {
    // Default values
    mainProxy = thisProxy;
    n_chares = 5;
    n_iters = 3;
    cur_iter = 1;
    busy_time = 1.0f;
    data_size = 0;
    qd_off = false;

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "c:i:t:d:n")) != -1) {
      switch (c) {
        case 'c':
          n_chares = atoi(optarg);
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 't':
          busy_time = atof(optarg);
          break;
        case 'd':
          data_size = (size_t)atoi(optarg);
          break;
        case 'n':
          qd_off = true;
          break;
        default:
          CkPrintf("Usage: %s -c [chares] -i [iterations] -t [busy time]"
                   "-d [data size]\n", m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // Calculate kernel clock count
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&cuda_device);
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    kernel_clock_count = busy_time * deviceProp.clockRate * 1000;

    // Print configuration
    CkPrintf("\n[CUDA qdtest example]\n");
    CkPrintf("Chares: %d\n", n_chares);
    CkPrintf("Iterations: %d\n", n_iters);
    CkPrintf("Busy time: %f s\n", busy_time);
    CkPrintf("Kernel clock count: %lld\n", kernel_clock_count);
    CkPrintf("Data size: %llu\n", data_size);
    CkPrintf("QD off: %s\n\n", qd_off ? "true" : "false");

    // Create 1D chare array and start
    qdProxy = CProxy_QD::ckNew(n_chares);

    // Ensure all chares are fully created before running them
    CkStartQD(CkCallback(CkIndex_Main::start(), thisProxy));
  }

  void start() {
    qdProxy.run();
    CkPrintf("Begin iteration %d\n", cur_iter);

    if (!qd_off) {
      // Start quiescence detection
      CkCallback cb(CkIndex_Main::done(), thisProxy);
      CkStartQD(cb);
    }
  };

  void done() {
    CkPrintf("End iteration %d\n", cur_iter);
    if (++cur_iter <= n_iters) {
      // Run next iteration and wait for QD
      qdProxy.run();
      CkPrintf("\nBegin iteration %d\n", cur_iter);

      if (!qd_off) {
        CkCallback cb(CkIndex_Main::done(), thisProxy);
        CkStartQD(cb);
      }
    }
    else {
      CkPrintf("\nAll done\n");
      CkExit();
    }
  }
};

/* array [1D] */
class QD : public CBase_QD {
  cudaStream_t stream;
  char* h_A;
  char* h_B;
  char* d_A;
  char* d_B;
  CkCallback* h2d_cb;
  CkCallback* kernel_cb;
  CkCallback* d2h_cb;

 public:
  QD() {
    cudaMallocHost(&h_A, data_size);
    cudaMallocHost(&h_B, data_size);
    cudaMalloc(&d_A, data_size);
    cudaMalloc(&d_B, data_size);
    hapiCheck(cudaStreamCreate(&stream));

    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
    h2d_cb = new CkCallback(CkIndex_QD::h2dDone(), myIndex, thisArrayID);
    kernel_cb = new CkCallback(CkIndex_QD::kernelDone(), myIndex, thisArrayID);
    d2h_cb = new CkCallback(CkIndex_QD::d2hDone(), myIndex, thisArrayID);
  }

  ~QD() {
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    hapiCheck(cudaStreamDestroy(stream));

    delete h2d_cb;
    delete kernel_cb;
    delete d2h_cb;
  }

  void run() {
    kernelSetup(h_A, h_B, d_A, d_B, data_size, kernel_clock_count, stream,
                (void*)h2d_cb, (void*)kernel_cb, (void*)d2h_cb);
  }

  void h2dDone() {
    CkPrintf("Chare %d h2dDone\n", thisIndex);
  }

  void kernelDone() {
    CkPrintf("Chare %d kernelDone\n", thisIndex);
    if (data_size <= 0)
      allDone();
  }

  void d2hDone() {
    CkPrintf("Chare %d d2hDone\n", thisIndex);
    allDone();
  }

  void allDone() {
    if (qd_off) {
      CkCallback cb(CkReductionTarget(Main, done), mainProxy);
      contribute(cb);
    }
  }
};

#include "qdtest.def.h"
