#include "TraceControlPoints.decl.h"

extern CkGroupID traceControlPointsGID;

#ifdef CMK_BLUEGENEP
void initBGP_UPC_Counters(void);
#endif

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
    CkPrintf("Initializing counters on pe %d\n", CkMyPe());
   
  }
  TraceControlPointsInit(CkMigrateMessage *m):Chare(m) {}
};

class TraceControlPointsBOC : public CBase_TraceControlPointsBOC {
public:
  TraceControlPointsBOC(void) {
#if 0
#ifdef CMK_BLUEGENEP
      initBGP_UPC_Counters();
#endif
#endif
  };


  void pup(PUP::er &p)
  {
    CBase_TraceControlPointsBOC::pup(p);
    if(p.isUnpacking()){
      CkPrintf("Group TraceControlPointsBOC is not yet capable of migration.\n");
    }
  }

 TraceControlPointsBOC(CkMigrateMessage *m) : CBase_TraceControlPointsBOC(m) {};

  void printBGP_UPC_CountersBOC(void);

};


