#include <stdio.h>
#include "hapi.h"
#include "hello.decl.h"

/* readonly */ CProxy_Main main_proxy;
/* readonly */ CProxy_Hello hello_proxy;
/* readonly */ int n_elem;

extern void invokeKernel(cudaStream_t stream, void* cb);

class Main : public CBase_Main {
 public:
  Main(CkArgMsg* m) {
    main_proxy = thisProxy;
    n_elem = 4;

    int c;
    while ((c = getopt(m->argc, m->argv, "n:")) != -1) {
      switch (c) {
        case 'n':
          n_elem = atoi(optarg);
          break;
        default:
          CkPrintf("Usage: %s -n [chares]\n", m->argv[0]);
          CkExit();
      }
    }
    delete m;

    CkPrintf("\n[CUDA hello example]\n");
    CkPrintf("Chares: %d\n", n_elem);

    hapi_init();

    hello_proxy = CProxy_Hello::ckNew(n_elem);

    hello_proxy[0].greet();
  };

  void done() {
    CkPrintf("\nAll done\n");
    CkExit();
  }
};

class Hello : public CBase_Hello {
  cudaStream_t stream;

 public:
  Hello() { cudaStreamCreate(&stream); }

  ~Hello() { cudaStreamDestroy(stream); }

  void greet() {
    CkArrayIndex1D myIndex = CkArrayIndex1D(thisIndex);
    CkCallback* cb =
        new CkCallback(CkIndex_Hello::pass(), myIndex, thisArrayID);

    CkPrintf("Hello, I'm chare %d!\n", thisIndex);
    if (thisIndex < n_elem - 1) {
      invokeKernel(stream, (void*)cb);
      pass();
    }
    else {
      main_proxy.done();
    }
  }

  void pass() {
    thisProxy[thisIndex + 1].greet();
  }
};

#include "hello.def.h"
