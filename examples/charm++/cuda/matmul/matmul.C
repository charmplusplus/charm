#include <time.h>
#include "cublas_v2.h"
#include "hapi.h"
#include "matmul.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int matrixSize;
/* readonly */ bool useCublas;

#ifdef USE_WR
extern void cudaMatMul(int, float*, float*, float*, cudaStream_t, void*);
#else
extern void cudaMatMul(int, float*, float*, float*, float*, float*, float*,
                       cudaStream_t, cublasHandle_t, void*);
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
    matrixSize = 8;
    useCublas = false;

    // handle arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "c:s:b")) != -1) {
      switch (c) {
        case 'c':
          numChares = atoi(optarg);
          break;
        case 's':
          matrixSize = atoi(optarg);
          break;
        case 'b':
          useCublas = true;
          break;
        default:
          CkPrintf("Usage: %s -c [chares] -s [matrix size] -b: use CuBLAS\n",
                   m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // print configuration
    CkPrintf("\n[CUDA matmul example]\n");
    CkPrintf("Chares: %d\n", numChares);
    CkPrintf("Matrix size: %d x %d\n", matrixSize, matrixSize);
    CkPrintf("Use CuBLAS: %s\n", useCublas ? "true" : "false");

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
  cublasHandle_t handle;

 public:
  Workers() {
    int size = sizeof(float) * matrixSize * matrixSize;
    hapiCheck(cudaMallocHost(&h_A, size));
    hapiCheck(cudaMallocHost(&h_B, size));
    hapiCheck(cudaMallocHost(&h_C, size));
    hapiCheck(cudaStreamCreate(&stream));
#ifndef USE_WR
    hapiCheck(cudaMalloc(&d_A, size));
    hapiCheck(cudaMalloc(&d_B, size));
    hapiCheck(cudaMalloc(&d_C, size));
    if (useCublas) {
      cublasCreate(&handle);
      cublasSetStream(handle, stream);
    }
#endif

    srand(time(NULL));
    randomInit(h_A, matrixSize * matrixSize);
    randomInit(h_B, matrixSize * matrixSize);
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

    if (useCublas) cublasDestroy(handle);
#endif
  }

  void begin() {
    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
    CkCallback* cb =
        new CkCallback(CkIndex_Workers::complete(), myIndex, thisArrayID);
#ifdef USE_WR
    cudaMatMul(matrixSize, h_A, h_B, h_C, stream, (void*)cb);
#else
    cudaMatMul(matrixSize, h_A, h_B, h_C, d_A, d_B, d_C, stream, handle,
               (void*)cb);
#endif
  }

  void complete() {
#ifdef DEBUG
    CkPrintf("[%d] A\n", thisIndex);
    for (int i = 0; i < matrixSize; i++) {
      CkPrintf("[%d] ", thisIndex);
      for (int j = 0; j < matrixSize; j++) {
        CkPrintf("%.2f ", h_A[i * matrixSize + j]);
      }
      CkPrintf("\n");
    }
    CkPrintf("[%d] B\n", thisIndex);
    for (int i = 0; i < matrixSize; i++) {
      CkPrintf("[%d] ", thisIndex);
      for (int j = 0; j < matrixSize; j++) {
        CkPrintf("%.2f ", h_B[i * matrixSize + j]);
      }
      CkPrintf("\n");
    }
    CkPrintf("[%d] C\n", thisIndex);
    for (int i = 0; i < matrixSize; i++) {
      CkPrintf("[%d] ", thisIndex);
      for (int j = 0; j < matrixSize; j++) {
        CkPrintf("%.2f ", h_C[i * matrixSize + j]);
      }
      CkPrintf("\n");
    }
    CkPrintf("[%d] C-gold\n", thisIndex);
    for (int i = 0; i < matrixSize; i++) {
      CkPrintf("[%d] ", thisIndex);
      for (int j = 0; j < matrixSize; j++) {
        C[i * matrixSize + j] = 0;
        for (int k = 0; k < matrixSize; k++) {
          C[i * matrixSize + j] +=
              A[i * matrixSize + k] * B[k * matrixSize + j];
        }
        CkPrintf("%.2f ", C[i * matrixSize + j]);
      }
      CkPrintf("\n");
    }
#endif

    contribute(CkCallback(CkIndex_Main::done(), mainProxy));
  }
};

#include "matmul.def.h"
