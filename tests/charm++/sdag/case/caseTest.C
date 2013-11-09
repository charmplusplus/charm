#include "caseTest.decl.h"

struct Main : public CBase_Main {
  Main_SDAG_CODE;

  Main(CkArgMsg *m) {
    delete m;
    thisProxy.run();
  }
};

#include "caseTest.def.h"
