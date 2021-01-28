#include <stdio.h>
#include "hapi.h"
#include "hello.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int nElements;
/* readonly */ CProxy_Hello arr;

extern void invokeKernel(cudaStream_t stream);

class Main : public CBase_Main {
 public:
  Main(CkArgMsg* m) {
    // Default values
    mainProxy = thisProxy;
    nElements = 4;

    // Handle arguments
    int c;
    while ((c = getopt(m->argc, m->argv, "c:")) != -1) {
      switch (c) {
        case 'c':
          nElements = atoi(optarg);
          break;
        default:
          CkPrintf("Usage: %s -c [chares]\n", m->argv[0]);
          CkExit();
      }
    }
    delete m;

    // Print configuration
    CkPrintf("\n[CUDA resume thread example]\n");
    CkPrintf("PEs: %d, Chares: %d\n", CkNumPes(), nElements);

    // create 1D chare array
    arr = CProxy_Hello::ckNew(nElements);

    arr.greet();
  };

  void done() {
    CkPrintf("\nAll done\n");
    CkExit();
  }
};

/* array [1D] */
class Hello : public CBase_Hello {
  cudaStream_t stream;
  CkCallbackResumeThread* resume_cb;

 public:
  Hello() { hapiCheck(cudaStreamCreate(&stream)); }

  ~Hello() { hapiCheck(cudaStreamDestroy(stream)); }

  void greet() {
    // Invoke GPU kernel
    invokeKernel(stream);

    // Suspend thread which will be resumed when GPU kernel completes
    resume_cb = new CkCallbackResumeThread();
    hapiAddCallback(stream, resume_cb);
    delete resume_cb;

    // GPU kernel complete, resuming thread
    CkPrintf("[%d] Kernel complete, resuming thread\n", thisIndex);

    // Thread resumed, return to Main
    contribute(CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
};

#include "hello.def.h"
