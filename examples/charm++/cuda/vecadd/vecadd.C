#include <time.h>
#include "hapi.h"
#include "vecadd.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int vectorSize;

#ifdef USE_WR
extern void cudaVecAdd(int, float*, float*, float*, cudaStream_t, void*);
#else
extern void cudaVecAdd(int, float*, float*, float*, float*, float*, float*,
                       cudaStream_t, void*);
#endif

void randomInit(float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

class Main : public CBase_Main {
 private:
  CProxy_Workers workers;
  int numChares;
  double startTime;

 public:
  Main(CkArgMsg* m) {
    // default values
    mainProxy = thisProxy;
    numChares = 4;
    vectorSize = 1024;

    // handle arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "c:s:")) != -1) {
      switch (c) {
        case 'c':
          numChares = atoi(optarg);
          break;
        case 's':
          vectorSize = atoi(optarg);
          break;
        default:
          CkPrintf("Usage: %s -c [chares] -s [vector size]\n", m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // print configuration
    CkPrintf("\n[CUDA vecadd example]\n");
    CkPrintf("Chares: %d\n", numChares);
    CkPrintf("Vector size: %d\n", vectorSize);

    // create 1D chare array
    workers = CProxy_Workers::ckNew(numChares);

    // start measuring execution time
    startTime = CkWallTimer();

    // fire off all chares in array
    workers.begin();
  }

  void done() {
    CkPrintf("\nElapsed time: %f s\n", CkWallTimer() - startTime);
    CkExit();
  }
};

class Workers : public CBase_Workers {
 private:
  float* h_A;
  float* h_B;
  float* h_C;
#ifndef USE_WR
  float* d_A;
  float* d_B;
  float* d_C;
#endif
  cudaStream_t stream;

 public:
  Workers() {
    int size = sizeof(float) * vectorSize;
    hapiCheck(cudaMallocHost(&h_A, size));
    hapiCheck(cudaMallocHost(&h_B, size));
    hapiCheck(cudaMallocHost(&h_C, size));
    hapiCheck(cudaStreamCreate(&stream));
#ifndef USE_WR
    hapiCheck(cudaMalloc(&d_A, size));
    hapiCheck(cudaMalloc(&d_B, size));
    hapiCheck(cudaMalloc(&d_C, size));
#endif

    srand(time(NULL));
    randomInit(h_A, vectorSize);
    randomInit(h_B, vectorSize);
  }

  ~Workers() {
    hapiFreeHost(h_A);
    hapiFreeHost(h_B);
    hapiFreeHost(h_C);
    hapiCheck(cudaStreamDestroy(stream));
#ifndef USE_WR
    hapiFree(d_A);
    hapiFree(d_B);
    hapiFree(d_C);
#endif
  }

  void begin() {
    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
    CkCallback* cb =
        new CkCallback(CkIndex_Workers::complete(), myIndex, thisArrayID);
#ifdef USE_WR
    cudaVecAdd(vectorSize, h_A, h_B, h_C, stream, (void*)cb);
#else
    cudaVecAdd(vectorSize, h_A, h_B, h_C, d_A, d_B, d_C, stream, (void*)cb);
#endif
  }

  void complete() {
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

    contribute(CkCallback(CkIndex_Main::done(), mainProxy));
  }
};

#include "vecadd.def.h"
