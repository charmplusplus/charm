#include "arraytest.decl.h"

/*readonly*/ CProxy_Main mainProxy;

class Main : public CBase_Main {
private:
  Main_SDAG_CODE

  int test_num;
  std::vector<int> sizes;
  std::vector<CProxy_TestArray> arrays;

public:
  Main(CkArgMsg* m) : test_num(0) {
    delete m;

    // Test with 1 element per PE
    int size = CkNumPes();
    sizes.push_back(size);
    arrays.push_back(CProxy_TestArray::ckNew(size, size));

    // Test with many elements per PE
    size = CkNumPes() * 4;
    sizes.push_back(size);
    arrays.push_back(CProxy_TestArray::ckNew(size, size));

    // Test with fewer elements than PEs
    size = CkNumPes() / 2;
    if (size > 1) {
      sizes.push_back(size);
      arrays.push_back(CProxy_TestArray::ckNew(size, size));
    }

    // Test with singleton chare array
    size = 1;
    sizes.push_back(size);
    arrays.push_back(CProxy_TestArray::ckNew(size, size));

    // Defined in the ci file to do p2p and bcast sends to each array and
    // check the resulting sum reduction for correctness.
    thisProxy.run_tests();

    CkStartQD(CkCallback(CkIndex_Main::all_complete(), thisProxy));
  };

  void all_complete() {
    // Make sure all tests completed, then exit
    CkAssert(test_num == arrays.size());
    CkExit();
  }
};

class TestArray : public CBase_TestArray {
private:
  int size;

public:
  TestArray(int s) : size(s) {}

  void p2p(int idx) {
    CkAssert(idx == thisIndex);
    if (thisIndex < size - 1) {
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

#include "arraytest.def.h"
