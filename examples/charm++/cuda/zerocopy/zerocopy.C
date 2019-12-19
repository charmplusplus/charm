#include "zerocopy.decl.h"
#include <string>
#include "hapi.h"

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
    n_iters = 100;

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
    CkPrintf("PEs/Chares: %d, Block size: %d, Iters: %d\n", CkNumPes(),
        block_size, n_iters);

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

  cudaStream_t stream;

  Block() {}

  ~Block() {
    // Free memory and destroy CUDA stream
    hapiCheck(cudaFreeHost(h_local_data));
    hapiCheck(cudaFreeHost(h_remote_data));
    hapiCheck(cudaFree(d_local_data));
    hapiCheck(cudaFree(d_remote_data));
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
    cudaStreamCreate(&stream);

    // Initialize data
    invokeInitKernel(d_local_data, block_size, (double)thisIndex, stream);

    // Start iterating once data is initialized
    CkCallback* cb = new CkCallback(CkIndex_Block::iterate(), thisProxy[thisIndex]);
    hapiAddCallback(stream, cb);
  }

  void receive(int ref, int &size1, double *&arr1, int size2, double *arr2, CkNcpyBufferPost *ncpyPost) {
    CkPrintf("PE %d: receive function with CkNcpyBufferPost\n", CkMyPe());
    // Inform the runtime where the incoming data should be stored
    arr1 = d_remote_data;

    // Set flag in CkNcpyBufferPost to let the runtime know that it is
    // a device-to-device transfer
    ncpyPost->device = true;
  }

  void validateData() {
    // Move the data to the host for validation
    hapiCheck(cudaMemcpyAsync(h_remote_data, d_remote_data,
          sizeof(double) * block_size, cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaStreamSynchronize(stream));

    // Validate data
    bool validated = true;
    for (int i = 0; i < block_size; i++) {
      if (h_remote_data[i] != (double)peer) validated = false;
    }

    if (!validated) {
      CkPrintf("PE %d: Validation failed", CkMyPe());
    }
  }
};

#include "zerocopy.def.h"
