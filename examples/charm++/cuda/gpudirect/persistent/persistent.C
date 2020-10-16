#include "persistent.decl.h"
#include <string>
#include "hapi.h"

#define ERROR_TOLERANCE 1e-6

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_PersistentArray array_proxy;
/* readonly */ CProxy_PersistentGroup group_proxy;
/* readonly */ //CProxy_PersistentNodeGroup nodegroup_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;
/* readonly */ bool lb_test;

extern void invokeFillKernel(double*, int, double, cudaStream_t);

struct Container {
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;
  cudaStream_t stream;

  Container() : h_remote_data(nullptr), d_local_data(nullptr),
    d_remote_data(nullptr) {}

  ~Container() {
    hapiCheck(cudaFreeHost(h_remote_data));
    hapiCheck(cudaFree(d_local_data));
    hapiCheck(cudaFree(d_remote_data));
    hapiCheck(cudaStreamDestroy(stream));
  }

  void init() {
    hapiCheck(cudaMallocHost(&h_remote_data, sizeof(double) * block_size));
    hapiCheck(cudaMalloc(&d_local_data, sizeof(double) * block_size));
    hapiCheck(cudaMalloc(&d_remote_data, sizeof(double) * block_size));
    hapiCheck(cudaStreamCreate(&stream));
  }

  void fill(double val) {
    invokeFillKernel(d_local_data, block_size, val, stream);

    hapiCheck(cudaStreamSynchronize(stream));
  }

  void verify(double val) {
    hapiCheck(cudaMemcpyAsync(h_remote_data, d_remote_data,
          sizeof(double) * block_size, cudaMemcpyDeviceToHost, stream));
    hapiCheck(cudaStreamSynchronize(stream));

    for (int i = 0; i < block_size; i++) {
      if (fabs(h_remote_data[i] - val) > ERROR_TOLERANCE) {
        CkAbort("Validation failure at data index %d: expected %.6lf, got %.6lf\n",
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
    n_iters = 10;
    test_nodegroup = true;
    lb_test = false;

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
    while ((c = getopt(m->argc, m->argv, "s:i:l")) != -1) {
      switch (c) {
        case 's':
          block_size = atoi(optarg);
          break;
        case 'i':
          n_iters = atoi(optarg);
          break;
        case 'l':
          lb_test = true;
          break;
        default:
          CkAbort("Unknown command line argument detected");
      }
    }
    delete m;

    // Print info
    CkPrintf("[CUDA Zerocopy Verification Test]\n"
        "Block size: %d, Iters: %d, Nodegroup: %s, LB test: %s\n",
        block_size, n_iters, test_nodegroup ? "true" : "false",
        lb_test ? "true" : "false");

    // Create chares
    array_proxy = CProxy_PersistentArray::ckNew(CkNumPes());
    group_proxy = CProxy_PersistentGroup::ckNew();
    //nodegroup_proxy = CProxy_PersistentNodeGroup::ckNew();

    // Begin testing
    thisProxy.test();
  }

  void test() {
    start_time = CkWallTimer();

    CkPrintf("Testing chare array...\n");
    array_proxy.initSend();
    CkWaitQD();
    for (int i = 0; i < n_iters; i++) {
      array_proxy.testGet(i);
      CkWaitQD();
    }
    for (int i = 0; i < n_iters; i++) {
      array_proxy.testPut(i);
      CkWaitQD();
    }
    CkPrintf("PASS\n");

    CkPrintf("Testing chare group...\n");
    group_proxy.initSend();
    CkWaitQD();
    for (int i = 0; i < n_iters; i++) {
      group_proxy.testGet(i);
      CkWaitQD();
    }
    for (int i = 0; i < n_iters; i++) {
      group_proxy.testPut(i);
      CkWaitQD();
    }
    CkPrintf("PASS\n");

    /*
    if (test_nodegroup) {
      CkPrintf("Testing chare nodegroup...\n");
      for (int i = 0; i < n_iters; i++) {
        nodegroup_proxy[0].send();
        CkWaitQD();
      }
      CkPrintf("PASS\n");
    }
    */

    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class PersistentArray : public CBase_PersistentArray {
  PersistentArray_SDAG_CODE

  Container container;
  CkDeviceBuffer my_send_buf;
  CkDeviceBuffer my_recv_buf;
  CkDeviceBuffer peer_send_buf;
  CkDeviceBuffer peer_recv_buf;
  int me;
  int peer;

public:
  PersistentArray() {
    me = CkMyPe();
    peer = (CkMyPe() == 0) ? 1 : 0;
    usesAtSync = true;
    container.init();
  }

  PersistentArray(CkMigrateMessage* m) {
    container.init();
  }

  void pup(PUP::er& p) {}

  void initSend() {
    container.fill(0);

    // Initialize and send my metadata to peer
    my_send_buf = CkDeviceBuffer(container.d_local_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentArray::callback(), thisProxy[thisIndex]),
        container.stream);
    my_recv_buf = CkDeviceBuffer(container.d_remote_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentArray::callback(), thisProxy[thisIndex]),
        container.stream);
    thisProxy[peer].initRecv(my_send_buf, my_recv_buf);
  }

  void initRecv(CkDeviceBuffer send_buf, CkDeviceBuffer recv_buf) {
    peer_send_buf = send_buf;
    peer_recv_buf = recv_buf;
  }

  void ResumeFromSync() {}
};

class PersistentGroup : public CBase_PersistentGroup {
  PersistentGroup_SDAG_CODE

  Container container;
  CkDeviceBuffer my_send_buf;
  CkDeviceBuffer my_recv_buf;
  CkDeviceBuffer peer_send_buf;
  CkDeviceBuffer peer_recv_buf;
  int me;
  int peer;

public:
  PersistentGroup() {
    me = CkMyPe();
    peer = (CkMyPe() == 0) ? 1 : 0;
    container.init();
  }

  void initSend() {
    container.fill(0);

    // Initialize and send my metadata to peer
    my_send_buf = CkDeviceBuffer(container.d_local_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentArray::callback(), thisProxy[thisIndex]),
        container.stream);
    my_recv_buf = CkDeviceBuffer(container.d_remote_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentArray::callback(), thisProxy[thisIndex]),
        container.stream);
    thisProxy[peer].initRecv(my_send_buf, my_recv_buf);
  }

  void initRecv(CkDeviceBuffer send_buf, CkDeviceBuffer recv_buf) {
    peer_send_buf = send_buf;
    peer_recv_buf = recv_buf;
  }
};

/*
class PersistentNodeGroup : public CBase_PersistentNodeGroup {
  Container container;
  CkDeviceBuffer buf;

public:
  PersistentNodeGroup() {
    container.init((thisIndex == 0) ? 1 : 2);
  }

  void send() {
    buf = CkDeviceBuffer(container.d_local_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentNodeGroup::srcCb(), thisProxy[thisIndex]),
        container.stream);
    thisProxy[1].recv(buf);
  }

  void recv(CkDeviceBuffer src_buf) {
    buf = CkDeviceBuffer(container.d_remote_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentNodeGroup::dstCb(), thisProxy[thisIndex]),
        container.stream);
    buf.get(src_buf);
  }

  void srcCb() { CkPrintf("PersistentNodeGroup %d, srcCb\n", thisIndex); }

  void dstCb() {
    CkPrintf("PersistentNodeGroup %d, dstCb\n", thisIndex);
    container.verify(1);
  }
};
*/

#include "persistent.def.h"
