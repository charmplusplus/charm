#include "TraceControlPoints.decl.h"

extern CkGroupID traceControlPointsGID;

// We typically declare parallel object classes here for the purposes of
// performing parallel operations for the trace module after the main
// application has completed execution (and calls CkExit()).
//
// TraceControlPointsInit is an initialization class.
//
// TraceControlPointsBOC is a one-per-processor object (defined in the .ci file as
//    a "group" instead of a "chare") which hosts the methods for the
//    parallel operations. In this case, there are no methods defined.
//    Otherwise, one may write any Charm++ code here.

class TraceControlPointsInit : public Chare {
  public:
  TraceControlPointsInit(CkArgMsg*) {
    traceControlPointsGID = CProxy_TraceControlPointsBOC::ckNew();
    CProxy_TraceControlPointsBOC controlPointsProxy(traceControlPointsGID);
  }
  TraceControlPointsInit(CkMigrateMessage *m):Chare(m) {}
};

class TraceControlPointsBOC : public CBase_TraceControlPointsBOC {
public:
  TraceControlPointsBOC(void) {};
  TraceControlPointsBOC(CkMigrateMessage *m) {};
};


