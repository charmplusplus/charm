#include "hello.decl.h"
#include "hello.h"
#include "pup_stl.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <typeinfo>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Hello helloProxy;

class Main : public CBase_Main {
public:
  Main(CkArgMsg* m) {
    // Create nodegroup and run
    helloProxy = CProxy_Hello::ckNew();
    helloProxy.run();
  };

  void done() {
    CkPrintf("All done\n");

    CkExit();
  };
};

class Hello : public CBase_Hello {
public:
  Hello() {}

  void run() {
    // Parallel execution with Raja
    hello(16, CmiMyNode());

    // Reduce to Main to end the program
    CkCallback cb(CkReductionTarget(Main, done), mainProxy);
    contribute(cb);
  }
};

#include "hello.def.h"
