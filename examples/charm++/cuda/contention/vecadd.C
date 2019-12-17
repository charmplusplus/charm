#include "vecadd.decl.h"
#include "hapi.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int vector_size;
/* readonly */ int chunk;
/* readonly */ bool multi_threaded;

extern void cudaVecAdd(int, double*, double*, double*, double*, double*, double*,
                       cudaStream_t);

class Main : public CBase_Main {
 private:
  CProxy_Chunk chunks;
  double program_start_time;
  double kernel_start_time;

 public:
  Main(CkArgMsg* m) {
    mainProxy = thisProxy;
    chunk = 128 * 1024;
    multi_threaded = true;

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "n:x")) != -1) {
      switch (c) {
        case 'n':
          chunk = atoi(optarg);
          break;
        case 'x':
          multi_threaded = false;
          break;
        default:
          CkAbort("Unknown argument");
      }
    }
    delete m;

    CkPrintf("PEs: %d, chunk: %d, multi-threaded: %d\n", CkNumPes(), chunk, multi_threaded);

    program_start_time = CkWallTimer();

    // Create chunk chares and initiate H2D data transfers
    if (multi_threaded) {
      chunks = CProxy_Chunk::ckNew(CkNumPes());
    }
    else {
      chunks = CProxy_Chunk::ckNew(1);
    }

    chunks.h2d();
  }

  void h2d_complete() {
    kernel_start_time = CkWallTimer();

    chunks.kernel();
  }

  void kernel_complete() {
    CkPrintf("\nKernel time: %.6lf s\n", CkWallTimer() - kernel_start_time);

    chunks.d2h();
  }

  void d2h_complete() {
    CkPrintf("Program time: %.6lf s\n", CkWallTimer() - program_start_time);

    CkExit();
  }
};

class Chunk : public CBase_Chunk {
 private:
  int weight;
  double** h_A;
  double** h_B;
  double** h_C;
  double** d_A;
  double** d_B;
  double** d_C;
  size_t size;
  cudaStream_t stream;

 public:
  Chunk() {
    weight = multi_threaded ? 1 : CkNumPes();
    h_A = (double**)malloc(sizeof(double*) * weight);
    h_B = (double**)malloc(sizeof(double*) * weight);
    h_C = (double**)malloc(sizeof(double*) * weight);
    d_A = (double**)malloc(sizeof(double*) * weight);
    d_B = (double**)malloc(sizeof(double*) * weight);
    d_C = (double**)malloc(sizeof(double*) * weight);

    size = chunk * sizeof(double);

    for (int i = 0; i < weight; i++) {
      hapiCheck(cudaMallocHost(&h_A[i], size));
      hapiCheck(cudaMallocHost(&h_B[i], size));
      hapiCheck(cudaMallocHost(&h_C[i], size));
      hapiCheck(cudaMalloc(&d_A[i], size));
      hapiCheck(cudaMalloc(&d_B[i], size));
      hapiCheck(cudaMalloc(&d_C[i], size));

      for (int j = 0; j < chunk; j++) {
        h_A[i][j] = (double)j;
        h_B[i][j] = (double)j;
      }
    }

    hapiCheck(cudaStreamCreate(&stream));
  }

  ~Chunk() {
    for (int i = 0; i < weight; i++) {
      hapiCheck(cudaFreeHost(h_A[i]));
      hapiCheck(cudaFreeHost(h_B[i]));
      hapiCheck(cudaFreeHost(h_C[i]));
      hapiCheck(cudaFree(d_A[i]));
      hapiCheck(cudaFree(d_B[i]));
      hapiCheck(cudaFree(d_C[i]));
    }

    hapiCheck(cudaStreamDestroy(stream));
  }

  void h2d() {
    double memcpy_start_time = CkWallTimer();

    // Copy input vectors to device
    for (int i = 0; i < weight; i++) {
      hapiCheck(cudaMemcpyAsync(d_A[i], h_A[i], size, cudaMemcpyHostToDevice, stream));
      hapiCheck(cudaMemcpyAsync(d_B[i], h_B[i], size, cudaMemcpyHostToDevice, stream));
    }

    CkPrintf("[Chare %d] H2D memcpy API time: %.3lf us\n", thisIndex,
        (CkWallTimer() - memcpy_start_time) * 1000000);

    // Set up callback
    CkCallback* cb = new CkCallback(CkIndex_Chunk::h2d_done(), thisProxy[thisIndex]);
    hapiAddCallback(stream, cb);
  }

  void h2d_done() {
    // Synchronize
    contribute(CkCallback(CkReductionTarget(Main, h2d_complete), mainProxy));
  }

  void kernel() {
    double launch_start_time = CkWallTimer();

    // Invoke kernel
    for (int i = 0; i < weight; i++) {
      cudaVecAdd(chunk, h_A[i], h_B[i], h_C[i], d_A[i], d_B[i], d_C[i], stream);
    }

    CkPrintf("[Chare %d] launch API time: %.3lf us\n", thisIndex,
        (CkWallTimer() - launch_start_time) * 1000000);

    // Set up callback
    CkCallback* cb = new CkCallback(CkIndex_Chunk::kernel_done(), thisProxy[thisIndex]);
    hapiAddCallback(stream, cb);
  }

  void kernel_done() {
    // Synchronize
    contribute(CkCallback(CkReductionTarget(Main, kernel_complete), mainProxy));
  }

  void d2h() {
    double memcpy_start_time = CkWallTimer();

    // Copy output vector to host
    for (int i = 0; i < weight; i++) {
      hapiCheck(cudaMemcpyAsync(h_C[i], d_C[i], size, cudaMemcpyDeviceToHost, stream));
    }

    CkPrintf("[Chare %d] D2H memcpy API time: %.3lf us\n", thisIndex,
        (CkWallTimer() - memcpy_start_time) * 1000000);

    // Set up callback
    CkCallback* cb = new CkCallback(CkIndex_Chunk::d2h_done(), thisProxy[thisIndex]);
    hapiAddCallback(stream, cb);
  }

  void d2h_done() {
    contribute(CkCallback(CkReductionTarget(Main, d2h_complete), mainProxy));
  }
};

#include "vecadd.def.h"
