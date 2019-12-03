#include "matmul.decl.h"
#include "hapi.h"
#include "rand48_replacement.h"

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_GPUHandler gpuhandler_proxy;
/* readonly */ CProxy_Block a, b, c;
/* readonly */ double alpha;
/* readonly */ int block_size;
/* readonly */ int n_blocks;
/* readonly */ bool direct;
/* readonly */ bool print_block;

void invokeDgemm(int M, int N, int K, double alpha,
                   double *A, double *B, double *C) {
}

class Main : public CBase_Main {
  double start_time;

public:
  Main(CkArgMsg* m) {
    alpha = 1.0;
    block_size = 1024;
    n_blocks = 2;
    direct = false;
    print_block = false;

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:n:dp")) != -1) {
      switch (c) {
        case 's':
          block_size = atoi(optarg);
          break;
        case 'n':
          n_blocks = atoi(optarg);
          break;
        case 'd':
          direct = true;
          break;
        case 'p':
          print_block = true;
          break;
        default:
          CkAbort("Unknown command line argument detected");
      }
    }
    delete m;

    // Print info
    CkPrintf("Block size: %d x %d, Grid: %d x %d\n", block_size, block_size,
        n_blocks, n_blocks);
    CkPrintf("Direct: %d\n", direct);

    main_proxy = thisProxy;

    // Override GPU settings set by HAPI
    gpuhandler_proxy = CProxy_GPUHandler::ckNew();
    gpuhandler_proxy.setGPU();
  }

  void ready() {
    a = CProxy_Block::ckNew(n_blocks, n_blocks);
    b = CProxy_Block::ckNew(n_blocks, n_blocks);
    c = CProxy_Block::ckNew(n_blocks, n_blocks);

    a.init(true);
    b.init(true);
    c.init(false);
  }

  void initDone() {
    start_time = CkWallTimer();

    a.sendInput(true);
    b.sendInput(false);
    c.run(CkCallback(CkReductionTarget(Main, done), thisProxy));
  }

  void done() {
    CkPrintf("Elapsed: %.6lf s", CkWallTimer() - start_time);
    CkExit();
  }
};

class GPUHandler : public CBase_GPUHandler {
public:
  int device_count;
  int pes_per_process;
  int local_pe_id;
  int pes_per_gpu;
  int gpu_id;
  bool gpu_pe_handler;

  GPUHandler() {
    device_count = 0;
    pes_per_process = 0;
    local_pe_id = -1;
    pes_per_gpu = 0;
    gpu_id = -1;
    gpu_pe_handler = false;
  }

  // WARNING: Assumes this code was run with jsrun, where the number of GPUs
  // accessible to each process are explicitly specified.
  void setGPU() {
    // Get number of accessible GPUs from this PE/process
    hapiCheck(cudaGetDeviceCount(&device_count));
    CkAssert(device_count > 0);

    // Block mapping of PEs to GPUs
    pes_per_process = CkNumPes() / CkNumNodes();
    local_pe_id = CkMyPe() % pes_per_process;
    pes_per_gpu = pes_per_process / device_count;
    gpu_id = local_pe_id / pes_per_gpu;
    hapiCheck(cudaSetDevice(gpu_id));

    CkPrintf("[PE %d, LPE %d] Set CUDA device to %d\n", CkMyPe(), local_pe_id, gpu_id);

    // Assign a GPU handler PE for each GPU, by choosing the first PE among
    // the PEs mapped to a GPU
    gpu_pe_handler = (local_pe_id % pes_per_gpu == 0);

    // Following code is executed by a single PE per GPU (GPU handler)
    if (gpu_pe_handler) {
      CkPrintf("[PE %d, LPE %d] I'm handler for GPU %d\n", CkMyPe(), local_pe_id, gpu_id);
      // Check if other GPUs accessible from the process can be peer-accessed,
      // and enable peer access if so
      int can_access_peer = 0;
      for (int i = 0; i < device_count; i++) {
        if (i != gpu_id) {
          hapiCheck(cudaDeviceCanAccessPeer(&can_access_peer, gpu_id, i));

          CkPrintf("Peer access from GPU %d to GPU %d: %d\n", gpu_id, i, can_access_peer);

          if (can_access_peer) {
            hapiCheck(cudaDeviceEnablePeerAccess(i, 0));
          }
        }
      }
    }

    contribute(CkCallback(CkReductionTarget(Main, ready), main_proxy));
  }
};

class Block : public CBase_Block {
  Block_SDAG_CODE

  int block;
  double* d_data;
  double* h_data;
  cudaStream_t stream;

public:
  Block() {}

  void init(rand_init) {
    // Allocate memory
    hapiCheck(cudaMalloc(&d_data, sizeof(double) * block_size * block_size));
    if (!direct) {
      hapiCheck(cudaMallocHost(&h_data, sizeof(double) * block_size * block_size));
    }

    cudaStreamCreate(&stream);

    // TODO: Initialize data

    // TODO: Reduce back to initDone()
  }

  ~Block() {
    hapiCheck(cudaFree(d_data));
    if (!direct) {
      hapiCheck(cudaFreeHost(h_data));
    }

    cudaStreamDestroy(stream);
  }

  void sendInput(bool is_a) {
    if (is_a) {
      c((thisIndex.x - thisIndex.y + n_blocks) % n_blocks, thisIndex.y).inputA(
          0, data, block_size, block_size);
    }
    else {
      c(thisIndex.x, (thisIndex.y - thisIndex.x + n_blocks) % n_blocks).inputB(
          0, data, block_size, block_size);
    }
  }
};

#include "matmul.def.h"
