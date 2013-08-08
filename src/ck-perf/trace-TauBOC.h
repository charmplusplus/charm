#include "TraceTau.decl.h"

extern CkGroupID traceTauGID;

// We typically declare parallel object classes here for the purposes of
// performing parallel operations for the trace module after the main
// application has completed execution (and calls CkExit()).
//
// TraceTauInit is an initialization class.
//
// TraceTauBOC is a one-per-processor object (defined in the .ci file as
//    a "group" instead of a "chare") which hosts the methods for the
//    parallel operations. In this case, there are no methods defined.
//    Otherwise, one may write any Charm++ code here.

class TraceTauInit : public Chare {
  public:
  TraceTauInit(CkArgMsg *m) {
    delete m;
    traceTauGID = CProxy_TraceTauBOC::ckNew();
    CProxy_TraceTauBOC tauProxy(traceTauGID);
  }
  TraceTauInit(CkMigrateMessage *m):Chare(m) {}
};

class TraceTauBOC : public CBase_TraceTauBOC {
public:
  TraceTauBOC(void) {};
  TraceTauBOC(CkMigrateMessage *m) {};
};


