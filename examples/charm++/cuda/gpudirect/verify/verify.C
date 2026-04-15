#include "verify.decl.h"
#include <string>
#include "hapi.h"

#define ERROR_TOLERANCE 1e-6

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_VerifyArray array_proxy;
/* readonly */ CProxy_VerifyGroup group_proxy;
/* readonly */ CProxy_VerifyNodeGroup nodegroup_proxy;
/* readonly */ int block_size;
/* readonly */ int n_iters;
/* readonly */ int n_warpup_iters;
/* readonly */ bool lb_test;

extern void invokeInitKernel(double*, int, double, hapiStream_t);

struct Container {
  double* h_local_data;
  double* h_remote_data;
  double* d_local_data;
  double* d_remote_data;
  hapiStream_t stream;

  Container() : h_local_data(nullptr), h_remote_data(nullptr),
    d_local_data(nullptr), d_remote_data(nullptr) {}

  ~Container() {
    hapiCheck(hapiFreeHost(h_local_data));
    hapiCheck(hapiFreeHost(h_remote_data));
    hapiCheck(hapiFree(d_local_data));
    hapiCheck(hapiFree(d_remote_data));
    hapiCheck(hapiStreamDestroy(stream));
  }

  void init(double val) {
    hapiCheck(hapiMallocHost(&h_local_data, sizeof(double) * block_size));
    hapiCheck(hapiMallocHost(&h_remote_data, sizeof(double) * block_size));
    hapiCheck(hapiMalloc(&d_local_data, sizeof(double) * block_size));
    hapiCheck(hapiMalloc(&d_remote_data, sizeof(double) * block_size));
    hapiCheck(hapiStreamCreate(&stream));

    for (int i = 0; i < block_size; i++) {
      h_local_data[i] = val;
    }
    invokeInitKernel(d_local_data, block_size, val, stream);
    invokeInitKernel(d_remote_data, block_size, val, stream);

    hapiCheck(hapiStreamSynchronize(stream));
  }

  void verify(double val) {
    hapiCheck(hapiMemcpyAsync(h_remote_data, d_remote_data,
          sizeof(double) * block_size, hapiMemcpyDeviceToHost, stream));
    hapiCheck(hapiStreamSynchronize(stream));

    for (int i = 0; i < block_size; i++) {
      if (fabs(h_remote_data[i] - val) > ERROR_TOLERANCE) {
        CkAbort("Validation failure at data index %d: expected %.6lf, got %.6lf\n",
            i, val, h_remote_data[i]);
      }
    }

    CmiPrintf("Data verified, looks OK!\n");
  }
};

class Main : public CBase_Main {
  bool test_nodegroup;
  double start_time;

public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    block_size = 1024*1024;
    n_iters = 100;
    n_warpup_iters = 3;
    test_nodegroup = true;
    lb_test = false;

    // Check if there are 2 PEs
    // if (CkNumPes() != 2) {
    //   CkAbort("Should be run with 2 PEs");
    // }

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
    CkPrintf("[hapi Zerocopy Verification Test]\n"
        "Block size: %d, Iters: %d, Nodegroup: %s, LB test: %s\n",
        block_size, n_iters, test_nodegroup ? "true" : "false",
        lb_test ? "true" : "false");

    // Create chares
    array_proxy = CProxy_VerifyArray::ckNew(CkNumPes());
    group_proxy = CProxy_VerifyGroup::ckNew();
    nodegroup_proxy = CProxy_VerifyNodeGroup::ckNew();

    // Begin testing
    thisProxy.test();
  }

  void test() {
    // warm up
    for (int i = 0; i < n_warpup_iters; i++) {
      array_proxy[0].send();
      // CkWaitQD();
      printf("[ITER] %d DONE!", i);
      fflush(stdout);
    }
    start_time = CkWallTimer();
    
    CkPrintf("Testing chare array... ");
    for (int i = 0; i < n_iters; i++) {
      array_proxy[0].send();
      printf("[ITER] %d DONE!", i);
      fflush(stdout);
    }
    CkWaitQD();
    CkPrintf("PASS\n");

    CkPrintf("Testing chare group... ");
    for (int i = 0; i < n_iters; i++) {
      group_proxy[0].send();
    }
    CkWaitQD();
    CkPrintf("PASS\n");

    // if (test_nodegroup) {
    //   CkPrintf("Testing chare nodegroup... ");
    //   for (int i = 0; i < 1; i++) {
    //     nodegroup_proxy[0].send();
    //   }
    //   CkWaitQD();
    //   CkPrintf("PASS\n");
    // }

    // sleep(3);

    CkPrintf("Elapsed: %.6lf s\n", CkWallTimer() - start_time);
    fflush(stdout);
    CkExit();
  }
};

class VerifyArray : public CBase_VerifyArray {
  Container container;
  int pe;

public:
  VerifyArray() {
    usesAtSync = true;
    container.init((thisIndex == 0) ? 1 : 2);
  }

  VerifyArray(CkMigrateMessage* m) {
    container.init((thisIndex == 0) ? 1 : 2);
  }

  void pup(PUP::er& p) {
    p|pe;
  }

  void send() {
    thisProxy[2].recv(block_size, CkDeviceBuffer(container.d_local_data,
          CkCallback(CkIndex_VerifyArray::reuse(), thisProxy[thisIndex]),
          container.stream));
    if (lb_test) {
      pe = CkMyPe();
      AtSync();
    }
  }

  void recv(int& size, double*& data, CkDeviceBufferPost* post) {
    data = container.d_remote_data;
    post[0].hapi_stream = container.stream;
  }

  void recv(int size, double* data) {
    container.verify(1);
    if (lb_test) {
      pe = CkMyPe();
      AtSync();
    }
    printf("[VERIFY] data received on PE: %d, Process: %d\n", CmiMyPe(), CmiMyNode());
    fflush(stdout);
  }

  void reuse() {
    printf("[VERIFY] source callback called on PE: %d, Process: %d\n", CmiMyPe(), CmiMyNode());
    fflush(stdout);
  }

  void ResumeFromSync() {}
};

class VerifyGroup : public CBase_VerifyGroup {
  Container container;

public:
  VerifyGroup() {
    container.init((thisIndex == 0) ? 1 : 2);
  }

  void send() {
    thisProxy[2].recv(block_size, CkDeviceBuffer(container.d_local_data, container.stream));
  }

  void recv(int& size, double*& data, CkDeviceBufferPost* post) {
    data = container.d_remote_data;
    post[0].hapi_stream = container.stream;
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

  void send() {
    thisProxy[1].recv(block_size, CkDeviceBuffer(container.d_local_data, container.stream));
  }

  void recv(int& size, double*& data, CkDeviceBufferPost* post) {
    data = container.d_remote_data;
    post[0].hapi_stream = container.stream;
  }

  void recv(int size, double* data) {
    container.verify(1);
  }
};

#include "verify.def.h"
