#include <time.h>
#include "mempool.decl.h"
#include "hapi.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif

#define N_SIZES 6

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int n_chares;
/* readonly */ int iterations;
/* readonly */ bool use_mempool;

extern void cudaVecAdd(int n_floats, size_t size, float* h_A, float* h_B, float* h_C,
    float* d_A, float* d_B, float* d_C, cudaStream_t stream);

class Main : public CBase_Main {
 private:
  CProxy_Workers workers;
  double start_time;

 public:
  Main(CkArgMsg* m) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::Main", NVTXColor::Turquoise);
#endif

    // default values
    mainProxy = thisProxy;
    iterations = 100;
    n_chares = CkNumPes();
    use_mempool = true;

    // handle arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "i:n:x")) != -1) {
      switch (c) {
        case 'n':
          n_chares = atoi(optarg);
          break;
        case 'i':
          iterations = atoi(optarg);
          break;
        case 'x':
          use_mempool = false;
          break;
        default:
          CkPrintf("Usage: %s -n [chares] -i [iterations]\n", m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // print configuration
    CkPrintf("\n[CUDA mempool example]\n");
    CkPrintf("Chares: %d\n", n_chares);
    CkPrintf("Iterations: %d\n", iterations);
    CkPrintf("Using mempool: %s\n", use_mempool ? "True" : "False");

    // create 1D chare array
    workers = CProxy_Workers::ckNew(n_chares);

    // start measuring execution time
    start_time = CkWallTimer();

    // fire off all chares in array
    workers.startIter();
  }

  void done() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::done", NVTXColor::Turquoise);
#endif

    CkPrintf("\nElapsed time: %f s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class Workers : public CBase_Workers {
  private:
    int cur_iter;
    int n_floats[N_SIZES] = {16, 256, 4096, 65536, 1048576, 16777216};
    size_t sizes[N_SIZES] = {64, 1024, 16384, 262144, 4194304, 67108864};
    float* h_A[N_SIZES];
    float* h_B[N_SIZES];
    float* h_C[N_SIZES];
    float* d_A[N_SIZES];
    float* d_B[N_SIZES];
    float* d_C[N_SIZES];
    cudaStream_t stream;

  public:
    Workers() {
#ifdef USE_NVTX
      NVTXTracer nvtx_range("Workers::Workers", NVTXColor::WetAsphalt);
#endif

      cur_iter = 0;

      for (int i = 0; i < N_SIZES; i++) {
        hapiCheck(cudaMalloc(&d_A[i], sizes[i]));
        hapiCheck(cudaMalloc(&d_B[i], sizes[i]));
        hapiCheck(cudaMalloc(&d_C[i], sizes[i]));
      }

      hapiCheck(cudaStreamCreate(&stream));
    }

    ~Workers() {
#ifdef USE_NVTX
      NVTXTracer nvtx_range("Workers::~Workers", NVTXColor::WetAsphalt);
#endif

      for (int i = 0; i < N_SIZES; i++) {
        hapiCheck(cudaFree(d_A[i]));
        hapiCheck(cudaFree(d_B[i]));
        hapiCheck(cudaFree(d_C[i]));
      }

      hapiCheck(cudaStreamDestroy(stream));
    }

    void startIter() {
#ifdef USE_NVTX
      NVTXTracer nvtx_range("Workers::startIter", NVTXColor::Carrot);
#endif

      for (int i = 0; i < N_SIZES; i++) {
        // dynamically allocate host memory to observe performance implications
        // of mempool
        hapiCheck(hapiMallocHost((void**)&h_A[i], sizes[i], use_mempool));
        hapiCheck(hapiMallocHost((void**)&h_B[i], sizes[i], use_mempool));
        hapiCheck(hapiMallocHost((void**)&h_C[i], sizes[i], use_mempool));

        cudaVecAdd(n_floats[i], sizes[i], h_A[i], h_B[i], h_C[i],
            d_A[i], d_B[i], d_C[i], stream);
      }

      CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
      CkCallback* cb =
          new CkCallback(CkIndex_Workers::endIter(), myIndex, thisArrayID);
      hapiAddCallback(stream, cb);
  }

  void endIter() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Workers::endIter", NVTXColor::Clouds);
#endif

    for (int i = 0; i < N_SIZES; i++) {
      hapiCheck(hapiFreeHost(h_A[i], use_mempool));
      hapiCheck(hapiFreeHost(h_B[i], use_mempool));
      hapiCheck(hapiFreeHost(h_C[i], use_mempool));
    }

    if (++cur_iter < iterations) {
      thisProxy[thisIndex].startIter();
    }
    else {
      contribute(CkCallback(CkIndex_Main::done(), mainProxy));
    }
  }
};

#include "mempool.def.h"
