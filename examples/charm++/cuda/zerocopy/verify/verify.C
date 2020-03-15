#include "verify.decl.h"
#include <string>
#include "hapi.h"

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_VerifyChare send_proxy;
/* readonly */ CProxy_VerifyChare recv_proxy;
/* readonly */ CProxy_VerifyArray array_proxy;
/* readonly */ CProxy_VerifyGroup group_proxy;
/* readonly */ CProxy_VerifyNodeGroup nodegroup_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;

extern void invokeInitKernel(double*, int, double, cudaStream_t);

struct Container {
  double* h_local_data;
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;
  cudaStream_t stream;

  Container() : h_local_data(nullptr), h_remote_data(nullptr),
    d_local_data(nullptr), d_remote_data(nullptr) {}

  inline void wait() {
    cudaStreamSynchronize(stream);
  }

  void init(double val) {
    hapiCheck(cudaMallocHost(&h_local_data, sizeof(double) * block_size));
    hapiCheck(cudaMallocHost(&h_remote_data, sizeof(double) * block_size));
    hapiCheck(cudaMalloc(&d_local_data, sizeof(double) * block_size));
    hapiCheck(cudaMalloc(&d_remote_data, sizeof(double) * block_size));
    cudaStreamCreate(&stream);

    invokeInitKernel(d_local_data, block_size, val, stream);
    invokeInitKernel(d_remote_data, block_size, val, stream);

    wait();
  }

  void clear() {
    hapiCheck(cudaFreeHost(h_local_data));
    hapiCheck(cudaFreeHost(h_remote_data));
    hapiCheck(cudaFree(d_local_data));
    hapiCheck(cudaFree(d_remote_data));
    cudaStreamDestroy(stream);
  }

  void verify(double val) {
    hapiCheck(cudaMemcpyAsync(h_remote_data, d_remote_data,
          sizeof(double) * block_size, cudaMemcpyDeviceToHost, stream));
    wait();

    for (int i = 0; i < block_size; i++) {
      if (h_remote_data[i] != val) {
        CkAbort("Validation failure at data index %d: expected %.3lf, got %.3lf\n",
            i, val, h_remote_data[i]);
      }
    }
  }
};

class Main : public CBase_Main {
  bool test_nodegroup;
  double start_time;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    block_size = 128;
    n_iters = 1;
    test_nodegroup = true;

    // Check if there are 2 PEs
    if (CkNumPes() != 2) {
      CkAbort("Should be run with 2 PEs");
    }

    // Don't do nodegroup test if run with 1 process
    if (CmiNumNodes() == 1) {
      test_nodegroup = false;
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
    CkPrintf("[CUDA Zerocopy Verification Test]\n"
        "Block size: %d, Iters: %d, Nodegroup %s\n",
        block_size, n_iters, test_nodegroup ? "true" : "false");

    // Create chares
    send_proxy = CProxy_VerifyChare::ckNew(true, 0);
    recv_proxy = CProxy_VerifyChare::ckNew(false, 1);
    array_proxy = CProxy_VerifyArray::ckNew(CkNumPes());
    group_proxy = CProxy_VerifyGroup::ckNew();
    nodegroup_proxy = CProxy_VerifyNodeGroup::ckNew();

    // Begin testing
    thisProxy.test();
  }

  void test() {
    start_time = CkWallTimer();

    /*
    // TODO: Singleton chares are currently not supported
    CkPrintf("Testing singleton chares... ");
    for (int i = 0; i < n_iters; i++) {
      send_proxy.send();
      CkWaitQD();
    }
    CkPrintf("PASS\n");
    */

    CkPrintf("Testing chare array... ");
    for (int i = 0; i < n_iters; i++) {
      array_proxy[0].send();
      CkWaitQD();
    }
    CkPrintf("PASS\n");

    CkPrintf("Testing chare group... ");
    for (int i = 0; i < n_iters; i++) {
      group_proxy[0].send();
      CkWaitQD();
    }
    CkPrintf("PASS\n");

    if (test_nodegroup) {
      CkPrintf("Testing chare nodegroup... ");
      for (int i = 0; i < n_iters; i++) {
        nodegroup_proxy[0].send();
        CkWaitQD();
      }
      CkPrintf("PASS\n");
    }

    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class VerifyChare : public CBase_VerifyChare {
  bool is_send;
  Container container;

public:
  VerifyChare(bool is_send_) : is_send(is_send_) {
    container.init(is_send ? 1 : 2);
  }

  ~VerifyChare() {
    container.clear();
  }

  void send() {
    recv_proxy.recv(block_size, CkSendBuffer(container.d_local_data, container.stream));
  }

  void recv(int& size, double*& data, CkNcpyBufferPost* post) {
    data = container.d_remote_data;
    post[0].cuda_stream = container.stream;
  }

  void recv(int size, double* data) {
    container.verify(1);
  }
};

class VerifyArray : public CBase_VerifyArray {
  Container container;

public:
  VerifyArray() {
    container.init((thisIndex == 0) ? 1 : 2);
  }

  ~VerifyArray() {
    container.clear();
  }

  void send() {
    thisProxy[1].recv(block_size, CkSendBuffer(container.d_local_data, container.stream));
  }

  void recv(int& size, double*& data, CkNcpyBufferPost* post) {
    data = container.d_remote_data;
    post[0].cuda_stream = container.stream;
  }

  void recv(int size, double* data) {
    container.verify(1);
  }
};

class VerifyGroup : public CBase_VerifyGroup {
  Container container;

public:
  VerifyGroup() {
    container.init((thisIndex == 0) ? 1 : 2);
  }

  ~VerifyGroup() {
    container.clear();
  }

  void send() {
    thisProxy[1].recv(block_size, CkSendBuffer(container.d_local_data, container.stream));
  }

  void recv(int& size, double*& data, CkNcpyBufferPost* post) {
    data = container.d_remote_data;
    post[0].cuda_stream = container.stream;
  }

  void recv(int size, double* data) {
    container.verify(1);
  }
};

class VerifyNodeGroup : public CBase_VerifyNodeGroup {
  Container container;

public:
  VerifyNodeGroup() {
    container.init((thisIndex == 0) ? 1 : 2);
  }

  ~VerifyNodeGroup() {
    container.clear();
  }

  void send() {
    thisProxy[1].recv(block_size, CkSendBuffer(container.d_local_data, container.stream));
  }

  void recv(int& size, double*& data, CkNcpyBufferPost* post) {
    data = container.d_remote_data;
    post[0].cuda_stream = container.stream;
  }

  void recv(int size, double* data) {
    container.verify(1);
  }
};

#include "verify.def.h"
