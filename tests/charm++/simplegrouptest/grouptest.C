#include "grouptest.decl.h"

/*readonly*/ CProxy_Main mainProxy;

class Main : public CBase_Main {
private:
  Main_SDAG_CODE

  int test_num;
  CProxy_TestGroup groupProxy;
  CProxy_TestNodeGroup nodeGroupProxy;

public:
  Main(CkArgMsg* m) : test_num(0) {
    delete m;

    groupProxy = CProxy_TestGroup::ckNew();
    nodeGroupProxy = CProxy_TestNodeGroup::ckNew();

    thisProxy.run_tests();

    CkStartQD(CkCallback(CkIndex_Main::all_complete(), thisProxy));
  };

  void all_complete() {
    // Make sure all tests completed, then exit
    CkAssert(test_num == 2);
    CkExit();
  }
};

class TestGroup : public CBase_TestGroup {
public:
  TestGroup() {}

  void p2p(int idx) {
    CkAssert(idx == thisIndex);
    if (thisIndex < CkNumPes() - 1) {
      thisProxy[thisIndex + 1].p2p(thisIndex + 1);
    }
    CkCallback cb(CkReductionTarget(Main, reduction), mainProxy);
    contribute(sizeof(thisIndex), &thisIndex, CkReduction::sum_int, cb);
  }

  void bcast() {
    CkCallback cb(CkReductionTarget(Main, reduction), mainProxy);
    contribute(sizeof(thisIndex), &thisIndex, CkReduction::sum_int, cb);
  }
};

class TestNodeGroup : public CBase_TestNodeGroup {
public:
  TestNodeGroup() {}

  void p2p(int idx) {
    CkAssert(idx == thisIndex);
    if (thisIndex < CkNumNodes() - 1) {
      thisProxy[thisIndex + 1].p2p(thisIndex + 1);
    }
    CkCallback cb(CkReductionTarget(Main, reduction), mainProxy);
    contribute(sizeof(thisIndex), &thisIndex, CkReduction::sum_int, cb);
  }

  void bcast() {
    CkCallback cb(CkReductionTarget(Main, reduction), mainProxy);
    contribute(sizeof(thisIndex), &thisIndex, CkReduction::sum_int, cb);
  }
};

#include "grouptest.def.h"
