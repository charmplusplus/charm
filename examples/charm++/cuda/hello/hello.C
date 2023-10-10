#include <stdio.h>
#include "hapi.h"
#include "hello.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int nElements;
/* readonly */ CProxy_Hello arr;

extern void kernelSetup(cudaStream_t stream, const CkCallback& cb);

/* mainchare */
class Main : public CBase_Main {
 public:
  Main(CkArgMsg* m) {
    // default values
    mainProxy = thisProxy;
    nElements = 5;

    // handle arguments
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

    // print configuration
    CkPrintf("\n[CUDA hello example]\n");
    CkPrintf("PEs: %d, Chares: %d\n", CkNumPes(), nElements);

    // create 1D chare array
    arr = CProxy_Hello::ckNew(nElements);

    // start by triggering first chare element
    arr[0].greet();
  };

  void done() {
    CkPrintf("\nAll done\n");
    CkExit();
  }
};

/* array [1D] */
class Hello : public CBase_Hello {
  cudaStream_t stream;

 public:
  Hello() { hapiCheck(cudaStreamCreate(&stream)); }

  ~Hello() { hapiCheck(cudaStreamDestroy(stream)); }

  void greet() {
    int device;
    hapiCheck(cudaGetDevice(&device));
    cudaDeviceProp prop;
    hapiCheck(cudaGetDeviceProperties(&prop, device));

    CkPrintf("Hello, I'm chare %d, on PE %d using GPU #%d %s\n",
        thisIndex, CkMyPe(), device, prop.name);

    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
    CkCallback cb(CkIndex_Hello::pass(), myIndex, thisArrayID);

    kernelSetup(stream, cb);
  }

  void pass() {
    if (thisIndex == nElements - 1) {
      // we've been around once, we're done
      mainProxy.done();
    } else {
      // pass the hello on
      thisProxy[thisIndex + 1].greet();
    }
  }
};

#include "hello.def.h"
