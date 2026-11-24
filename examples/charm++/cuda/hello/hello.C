#include <stdio.h>
#include "hapi.h"
#include "hello.h"
#include "hello.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int nElements;
/* readonly */ CProxy_Hello arr;

extern void kernelSetup(cudaStream_t stream, const CkCallback& cb,
                        int* h_out, int* d_out, int val);

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
    CkPrintf("\nAll done -- every chare verified its kernel output\n");
    CkExit();
  }
};

/* array [1D] */
class Hello : public CBase_Hello {
  cudaStream_t stream;
  // The greeting kernel's output, and the value it is supposed to contain.
  int* h_out;
  int* d_out;
  int expected;

 public:
  Hello() {
    hapiCheck(cudaStreamCreate(&stream));
    hapiCheck(cudaMallocHost(&h_out, HELLO_ELEMS * sizeof(int)));
    hapiCheck(cudaMalloc(&d_out, HELLO_ELEMS * sizeof(int)));
    // Non-zero and distinct per chare, so neither a zeroed buffer nor another
    // chare's result can pass for this one's.
    expected = thisIndex + 1;
  }

  ~Hello() {
    hapiCheck(cudaStreamDestroy(stream));
    hapiCheck(cudaFreeHost(h_out));
    hapiCheck(cudaFree(d_out));
  }

  void greet() {
    int device;
    hapiCheck(cudaGetDevice(&device));
    cudaDeviceProp prop;
    hapiCheck(cudaGetDeviceProperties(&prop, device));

    CkPrintf("Hello, I'm chare %d, on PE %d using GPU #%d %s\n",
        thisIndex, CkMyPe(), device, prop.name);

    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
    CkCallback cb(CkIndex_Hello::pass(), myIndex, thisArrayID);

    kernelSetup(stream, cb, h_out, d_out, expected);
  }

  void pass() {
    // Check what the kernel actually wrote. Without this the example cannot
    // tell a working GPU from one where every launch silently did nothing --
    // which is exactly what happens when the object was built without a cubin
    // or PTX for this device's architecture (see CUDA_ARCH in the Makefile).
    for (int i = 0; i < HELLO_ELEMS; i++) {
      if (h_out[i] != expected) {
        CkAbort("chare %d: kernel output[%d] is %d, expected %d -- the kernel "
                "did not run correctly on this device\n",
                thisIndex, i, h_out[i], expected);
      }
    }

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
