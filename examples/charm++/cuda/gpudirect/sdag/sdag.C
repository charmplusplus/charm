#include "sdag.decl.h"
#include <string>
#include "hapi.h"

#define ERROR_TOLERANCE 1e-6

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Block block_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;

extern void invokeInitKernel(double*, int, double, cudaStream_t);

class Main : public CBase_Main {
  double start_time;

 public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    block_size = 128;
    n_iters = 1;

    // Check if number of PEs is even
    if (CkNumPes() % 2 != 0) {
      CkAbort("Number of PEs has to be even!");
    }

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:i:")) != -1) {
      switch (c) {
        case 's':
          block_size = atoi(optarg);
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        default:
          CkAbort("Unknown command line argument detected");
      }
    }
    delete m;

    // Print info
    CkPrintf("PEs/Chares: %d, Block size: %d, Iters: %d\n",
        CkNumPes(), block_size, n_iters);

    // Create block group chare
    block_proxy = CProxy_Block::ckNew();
    start_time = CkWallTimer();
    block_proxy.init();
  }

  void terminate() {
    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

 public:
  int iter;
  int peer;

  double* h_local_data;
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;
  int* reg_local_data;
  int* reg_remote_data;

  cudaStream_t stream;

  Block() {}

  ~Block() {
    // Free memory and destroy CUDA stream
    hapiCheck(cudaFreeHost(h_local_data));
    hapiCheck(cudaFreeHost(h_remote_data));
    hapiCheck(cudaFree(d_local_data));
    hapiCheck(cudaFree(d_remote_data));
    free(reg_local_data);
    free(reg_remote_data);
    cudaStreamDestroy(stream);
  }

  void init() {
    iter = 1;

    // Determine the peer index
    peer = (thisIndex < CkNumPes() / 2) ? (thisIndex + CkNumPes() / 2) :
      (thisIndex - CkNumPes() / 2);

    // Allocate memory and create CUDA stream
    hapiCheck(cudaMallocHost(&h_local_data, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost(&h_remote_data, sizeof(double) * block_size));
    hapiCheck(cudaMalloc(&d_local_data, sizeof(double) * block_size));
    hapiCheck(cudaMalloc(&d_remote_data, sizeof(double) * block_size));
    reg_local_data = (int*)malloc(sizeof(int) * block_size);
    reg_remote_data = (int*)malloc(sizeof(int) * block_size);
    cudaStreamCreate(&stream);

    // Initialize data
    invokeInitKernel(d_local_data, block_size, (double)thisIndex, stream);
    invokeInitKernel(d_remote_data, block_size, (double)thisIndex, stream);
    for (int i = 0; i < block_size; i++) reg_local_data[i] = thisIndex;

    // Start iterating once data is initialized
    CkCallback* cb = new CkCallback(CkIndex_Block::iterate(), thisProxy[thisIndex]);
    hapiAddCallback(stream, cb);
  }

  void receive(int ref, int &size1, double *&arr1, int size2, int *arr2,
      CkDeviceBufferPost *devicePost) {
    // Inform the runtime where the incoming data should be stored
    // and which CUDA stream should be used for the transfer
    arr1 = d_remote_data;
    devicePost[0].cuda_stream = stream;

    // Last array should be available here as it is not RDMA
    // Copy it over for validation
    CkAssert(size2 == block_size);
    memcpy(reg_remote_data, arr2, sizeof(int) * block_size);
  }

  void validateData() {
    // Move the data to the host for validation
    hapiCheck(cudaMemcpyAsync(h_remote_data, d_remote_data,
          sizeof(double) * block_size, cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaStreamSynchronize(stream));

    // Validate data
    bool validated = true;
    for (int i = 0; i < block_size; i++) {
      if (fabs(h_remote_data[i] - (double)peer) > ERROR_TOLERANCE) {
        CkPrintf("h_remote_data[%d] = %lf invalid! Expected %lf\n", i,
            h_remote_data[i], (double)peer);
        validated = false;
      }
      if (fabs(reg_remote_data[i] - (double)peer) > ERROR_TOLERANCE) {
        CkPrintf("reg_remote_data[%d] = %d invalid! Expected %d\n", i,
            reg_remote_data[i], peer);
        validated = false;
      }
    }

    if (validated) {
      CkPrintf("PE %d: Validation success\n", CkMyPe());
    } else {
      CkAbort("PE %d: Validation failed\n", CkMyPe());
    }
  }
};

#include "sdag.def.h"
