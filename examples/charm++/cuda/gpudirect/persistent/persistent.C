#include "persistent.decl.h"
#include <string>
#include "hapi.h"

#define ERROR_TOLERANCE 1e-6

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_PersistentArray array_proxy;
/* readonly */ CProxy_PersistentGroup group_proxy;
/* readonly */ CProxy_PersistentNodeGroup nodegroup_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;
/* readonly */ bool lb_test;
/* readonly */ int lb_period;

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

  void pup(PUP::er& p) {
    if (p.isUnpacking()) {
      init();
    }
    PUParray(p, h_remote_data, block_size);
    // Data on GPU device do not migrate
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
  double start_time;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    block_size = 128;
    n_iters = 10;
    lb_test = false;
    lb_period = 3;

    // Check if there are 2 PEs
    if (CkNumPes() != 2) {
      CkAbort("Should be run with 2 PEs");
    }

    // Process command line arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "s:i:lp:")) != -1) {
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
        case 'p':
          lb_period = atoi(optarg);
          break;
        default:
          CkAbort("Unknown command line argument detected");
      }
    }
    delete m;

    // Print info
    CkPrintf("[CUDA Zerocopy Verification Test]\n"
        "Block size: %d, Iters: %d, LB: %s (period: %d)\n",
        block_size, n_iters, lb_test ? "true" : "false", lb_period);

    // Create chares
    array_proxy = CProxy_PersistentArray::ckNew(CkNumPes());
    group_proxy = CProxy_PersistentGroup::ckNew();
    nodegroup_proxy = CProxy_PersistentNodeGroup::ckNew();

    // Begin testing
    thisProxy.test();
  }

  void test() {
    start_time = CkWallTimer();

    CkPrintf("Testing chare array...\n");
    // Exchange CkDevicePersistents
    array_proxy.initSend();
    CkWaitQD();
    for (int i = 0; i < n_iters; i++) {
      array_proxy[1].fill(i);
      CkWaitQD();
      array_proxy.testGet(i);
      CkWaitQD();
      if (lb_test && ((i+1) % lb_period == 0)) {
        CkPrintf("LB step (iter %d), calling initSend\n", i);
        array_proxy.initSend();
        CkWaitQD();
      }
    }
    for (int i = 0; i < n_iters; i++) {
      array_proxy[0].fill(i);
      CkWaitQD();
      array_proxy.testPut(i);
      CkWaitQD();
    }
    CkPrintf("PASS\n");

    CkPrintf("Testing chare group...\n");
    group_proxy.initSend();
    CkWaitQD();
    for (int i = 0; i < n_iters; i++) {
      group_proxy[1].fill(i);
      CkWaitQD();
      group_proxy.testGet(i);
      CkWaitQD();
    }
    for (int i = 0; i < n_iters; i++) {
      group_proxy[0].fill(i);
      CkWaitQD();
      group_proxy.testPut(i);
      CkWaitQD();
    }
    CkPrintf("PASS\n");

    if (CmiNumNodes() != 1) {
      CkPrintf("Testing chare nodegroup...\n");
      nodegroup_proxy.initSend();
      for (int i = 0; i < n_iters; i++) {
        nodegroup_proxy[1].fill(i);
        CkWaitQD();
        nodegroup_proxy.testGet(i);
        CkWaitQD();
      }
      for (int i = 0; i < n_iters; i++) {
        nodegroup_proxy[0].fill(i);
        CkWaitQD();
        nodegroup_proxy.testPut(i);
        CkWaitQD();
      }
      CkPrintf("PASS\n");
    }

    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    CkExit();
  }
};

class PersistentArray : public CBase_PersistentArray {
  PersistentArray_SDAG_CODE

  Container container;
  CkDevicePersistent my_send_buf;
  CkDevicePersistent my_recv_buf;
  CkDevicePersistent peer_send_buf;
  CkDevicePersistent peer_recv_buf;
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
    usesAtSync = true;
  }

  void pup(PUP::er& p) {
    p|me;
    p|peer;
    p|container;
  }

  void initSend() {
    // Initialize and send my metadata to peer
    my_send_buf = CkDevicePersistent(container.d_local_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentArray::callback(), thisProxy[thisIndex]),
        container.stream);
    my_send_buf.open();
    my_recv_buf = CkDevicePersistent(container.d_remote_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentArray::callback(), thisProxy[thisIndex]),
        container.stream);
    my_recv_buf.open();
    thisProxy[peer].initRecv(my_send_buf, my_recv_buf);
  }

  void initRecv(CkDevicePersistent send_buf, CkDevicePersistent recv_buf) {
    peer_send_buf = send_buf;
    peer_recv_buf = recv_buf;
  }

  void fill(int iter) {
    container.fill(iter);
  }

  void ckAboutToMigrate() {
    peer_send_buf.close();
    peer_recv_buf.close();
  }
  void ResumeFromSync() {}
};

class PersistentGroup : public CBase_PersistentGroup {
  PersistentGroup_SDAG_CODE

  Container container;
  CkDevicePersistent my_send_buf;
  CkDevicePersistent my_recv_buf;
  CkDevicePersistent peer_send_buf;
  CkDevicePersistent peer_recv_buf;
  int me;
  int peer;

public:
  PersistentGroup() {
    me = CkMyPe();
    peer = (CkMyPe() == 0) ? 1 : 0;
    container.init();
  }

  void initSend() {
    // Initialize and send my metadata to peer
    my_send_buf = CkDevicePersistent(container.d_local_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentGroup::callback(), thisProxy[thisIndex]),
        container.stream);
    my_send_buf.open();
    my_recv_buf = CkDevicePersistent(container.d_remote_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentGroup::callback(), thisProxy[thisIndex]),
        container.stream);
    my_recv_buf.open();
    thisProxy[peer].initRecv(my_send_buf, my_recv_buf);
  }

  void initRecv(CkDevicePersistent send_buf, CkDevicePersistent recv_buf) {
    peer_send_buf = send_buf;
    peer_recv_buf = recv_buf;
  }

  void fill(int iter) {
    container.fill(iter);
  }
};

class PersistentNodeGroup : public CBase_PersistentNodeGroup {
  PersistentNodeGroup_SDAG_CODE

  Container container;
  CkDevicePersistent my_send_buf;
  CkDevicePersistent my_recv_buf;
  CkDevicePersistent peer_send_buf;
  CkDevicePersistent peer_recv_buf;
  int me;
  int peer;

public:
  PersistentNodeGroup() {
    me = CkMyNode();
    peer = (CkMyNode() == 0) ? 1 : 0;
    container.init();
  }

  void initSend() {
    // Initialize and send my metadata to peer
    my_send_buf = CkDevicePersistent(container.d_local_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentNodeGroup::callback(), thisProxy[thisIndex]),
        container.stream);
    my_send_buf.open();
    my_recv_buf = CkDevicePersistent(container.d_remote_data, sizeof(double) * block_size,
        CkCallback(CkIndex_PersistentNodeGroup::callback(), thisProxy[thisIndex]),
        container.stream);
    my_recv_buf.open();
    thisProxy[peer].initRecv(my_send_buf, my_recv_buf);
  }

  void initRecv(CkDevicePersistent send_buf, CkDevicePersistent recv_buf) {
    peer_send_buf = send_buf;
    peer_recv_buf = recv_buf;
  }

  void fill(int iter) {
    container.fill(iter);
  }
};

#include "persistent.def.h"
