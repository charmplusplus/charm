#include <time.h>
#include "vecadd.decl.h"
#include "hapi.h"
#ifdef USE_NVTX
#include "hapi_nvtx.h"
#endif

/* readonly */ CProxy_Main mainProxy;

#ifdef USE_WR
extern void cudaVecAdd(int, float*, float*, float*, cudaStream_t, void*);
#else
extern void cudaVecAdd(int vectorSize, float* h_A, float* d_A);
#endif

void randomInit(float* data, int size) {
#ifdef USE_NVTX
  NVTXTracer nvtx_range("randomInit", NVTXColor::PeterRiver);
#endif
  for (int i = 0; i < size; ++i) {
    data[i] = 10;
  }
}

class Main : public CBase_Main {
 private:
  CProxy_Workers workers;
  int numChares;
  double startTime;

 public:
  Main(CkArgMsg* m) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::Main", NVTXColor::Turquoise);
#endif

    // default values
    mainProxy = thisProxy;
    numChares = 4;

    // handle arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "c:s:")) != -1) {
      switch (c) {
        case 'c':
          numChares = atoi(optarg);
          break;
        default:
          CkPrintf("Usage: %s -c [chares] -s [vector size]\n", m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // print configuration
    CkPrintf("\n[CUDA vecadd example]\n");
    CkPrintf("Chares: %d\n", 1);

    // create 1D chare array
    workers = CProxy_Workers::ckNew(1024, 4);

    // start measuring execution time
    startTime = CkWallTimer();

    // fire off all chares in array
    workers.begin();
  }

  void done() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Main::done", NVTXColor::Turquoise);
#endif

    CkPrintf("\nElapsed time: %f s\n", CkWallTimer() - startTime);
    CkExit();
  }
};

class Workers : public CBase_Workers {
 private:
  int vectorSize;
  float* h_A;
#ifndef USE_WR
  float* d_A;
  float* d_B;
#endif
  cudaStream_t stream;

 public:
  Workers(int size) : vectorSize(size) {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Workers::Workers", NVTXColor::WetAsphalt);
#endif

    int dataSize = sizeof(float) * vectorSize;
    hapiCheck(cudaMallocHost(&h_A, dataSize));
    hapiCheck(cudaStreamCreate(&stream));
#ifndef USE_WR
    hapiCheck(hapiMalloc((void**) &d_A, dataSize));
    hapiCheck(hapiMalloc((void**) &d_B, dataSize));
    CkPrintf("[%d] d_A pointer: %p\n", CkMyPe(), d_A);
    CkPrintf("[%d] d_B pointer: %p\n", CkMyPe(), d_B);
#endif

    srand(time(NULL));
    randomInit(h_A, vectorSize);
  }

  Workers(CkMigrateMessage* m) : CBase_Workers(m) 
  {
    hapiCheck(cudaStreamCreate(&stream));    
  }

  ~Workers() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Workers::~Workers", NVTXColor::WetAsphalt);
#endif

    hapiCheck(cudaFreeHost(h_A));
    hapiCheck(cudaStreamDestroy(stream));
#ifndef USE_WR
    hapiCheck(cudaFree(d_A));
#endif
  }

  void pup(PUP::er& p) {
    p | vectorSize;
    if (p.isUnpacking())
    {
      cudaMallocHost(&h_A, vectorSize * sizeof(float));
      hapiMalloc((void**) &d_A, vectorSize * sizeof(float));
      CkPrintf("[%d] d_A pointer: %p\n", CkMyPe(), d_A);
    }
    //p(h_A, vectorSize);
    p(d_A, vectorSize, PUP::PUPMode::DEVICE);
  }

  void begin() {
#ifdef USE_NVTX
    NVTXTracer nvtx_range("Workers::begin", NVTXColor::Carrot);
#endif

    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);

    cudaVecAdd(vectorSize, h_A, d_A);
    complete();

    // CkPrintf("[%d] h_A array (size=%d):\n", CkMyPe(), vectorSize);
    // for (int i = 0; i < vectorSize; ++i) {
    //   CkPrintf("%.2f ", h_A[i]);
    // }
    // CkPrintf("\n");

    //if (thisIndex == 0)
    //  migrateMe(1);
    //CkExit();
  }

  void ckJustMigrated()
  {
    //CProxy_Workers self(thisArrayID);
    complete();
  }

  void complete() {
    cudaMemcpy(h_A, d_A, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

    CkPrintf("[%d] d_A pointer: %p\n", CkMyPe(), d_A);
    // CkPrintf("[%d] h_A array (size=%d):\n", CkMyPe(), vectorSize);
    // for (int i = 0; i < vectorSize; ++i) {
    //   CkPrintf("%.2f ", h_A[i]);
    // }
    // CkPrintf("\n");

    //CkExit();

#ifdef USE_NVTX
    NVTXTracer nvtx_range("Workers::complete", NVTXColor::Clouds);
#endif

#ifdef DEBUG
    CkPrintf("[%d] A\n", thisIndex);
    for (int i = 0; i < vectorSize; i++) {
      CkPrintf("%.2f ", h_A[i]);
    }
    CkPrintf("\n");

    CkPrintf("[%d] B\n", thisIndex);
    for (int i = 0; i < vectorSize; i++) {
      CkPrintf("%.2f ", h_B[i]);
    }
    CkPrintf("\n");

    CkPrintf("[%d] C\n", thisIndex);
    for (int i = 0; i < vectorSize; i++) {
      CkPrintf("%.2f ", h_C[i]);
    }
    CkPrintf("\n");

    CkPrintf("[%d] C-gold\n", thisIndex);
    for (int j = 0; j < vectorSize; j++) {
      h_C[j] = h_A[j] + h_B[j];
      CkPrintf("%.2f ", h_C[j]);
    }
    CkPrintf("\n");
#endif

    hapiCheck(hapiFree(d_A));
    hapiCheck(hapiFree(d_B));
    contribute(CkCallback(CkIndex_Main::done(), mainProxy));
  }
};

#include "vecadd.def.h"
