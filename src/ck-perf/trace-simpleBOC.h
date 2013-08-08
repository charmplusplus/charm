#include "TraceSimple.decl.h"

extern CkGroupID traceSimpleGID;

// We typically declare parallel object classes here for the purposes of
// performing parallel operations for the trace module after the main
// application has completed execution (and calls CkExit()).
//
// TraceSimpleInit is an initialization class.
//
// TraceSimpleBOC is a one-per-processor object (defined in the .ci file as
//    a "group" instead of a "chare") which hosts the methods for the
//    parallel operations. In this case, there are no methods defined.
//    Otherwise, one may write any Charm++ code here.

class TraceSimpleInit : public Chare {
  public:
  TraceSimpleInit(CkArgMsg *m) {
    delete m;
    traceSimpleGID = CProxy_TraceSimpleBOC::ckNew();
    CProxy_TraceSimpleBOC simpleProxy(traceSimpleGID);
  }
  TraceSimpleInit(CkMigrateMessage *m):Chare(m) {}
};

class TraceSimpleBOC : public CBase_TraceSimpleBOC {
public:
  TraceSimpleBOC(void) {};
  TraceSimpleBOC(CkMigrateMessage *m) {};
};


