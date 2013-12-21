#include "forallStateVar.decl.h"

struct Main : CBase_Main {
  Main_SDAG_CODE

  Main(CkArgMsg*) {
    thisProxy.testMethod();
    thisProxy.recvTest(10);
  }
};

#include "forallStateVar.def.h"
