#include "comp.decl.h"
#include "completion.h"

struct Main : public CBase_Main {
  CProxy_CompletionDetector detector;
  const int num;

  Main(CkArgMsg* msg)
    : num(10) {
    delete msg;
    detector = CProxy_CompletionDetector::ckNew();
    detector.start_detection(num,
                             CkCallback(CkIndex_Main::startTest(), thisProxy),
                             CkCallback(),
                             CkCallback(CkIndex_Main::finishTest(), thisProxy), 0);
  }

  void startTest() {
    CkPrintf("completion module initialized\n");
    CProxy_completion_array::ckNew(detector, num, num);
  }

  void finishTest() {
    CkPrintf("completion completed successfully\n");
    CkExit();
  }
};

struct completion_array : public CBase_completion_array {
  completion_array(CProxy_CompletionDetector det, int n) {
    CkPrintf("Array element %d producing %d elements\n", thisIndex, thisIndex + 1);
    det.ckLocalBranch()->produce(thisIndex + 1);
    det.ckLocalBranch()->done();
    CkPrintf("Array element %d consuming %d elements\n", thisIndex, n - thisIndex);
    det.ckLocalBranch()->consume(n - thisIndex);
  }
  completion_array(CkMigrateMessage *) {}
};

#include "comp.def.h"
