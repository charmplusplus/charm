#include "tester.decl.h"

class Test : public CBase_Test {
 public:
  Test(void) = default;

  void sayHi(void) {
    CkPrintf("%d> hi!\n", thisIndex);
    this->contribute(CkCallback(CkCallback::ckExit));
  }
};

class Main : public CBase_Main {
  CProxy_Test testProxy;
  int end;

 public:
  Main(CkArgMsg* msg) : end(CkNumPes() * 4) {
    CkArrayOptions opts;
    testProxy = CProxy_Test::ckNew(end);
    for (auto i = 1; i < end; i += 2) {
      // expected use-case for ( forward ) is after element deletion
      testProxy[i].ckDestroy();
    }
    CkStartQD(CkCallback(CkIndex_Main::run(), thisProxy));
  }

  void run(void) {
    CkPrintf("main> array elements [0:2:%d) remain.\n", end);
    for (auto i = 1; i < end; i += 2) {
      CkPrintf("main> forwarding messages from %d to %d.\n", i, i - 1);
      CkArrayIndex1D src(i), dst(i - 1);
      testProxy.ckLocMgr()->forward(src, dst);
      testProxy[i].sayHi();
    }
  }
};

#include "tester.def.h"
