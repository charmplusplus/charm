#include "caseTest.decl.h"

struct Main : public CBase_Main {
  Main_SDAG_CODE;

  Main(CkArgMsg *m) {
    delete m;
#if defined(_WIN32)
    CkExit();
#else
    thisProxy.run();
#endif
  }
};

#include "caseTest.def.h"
