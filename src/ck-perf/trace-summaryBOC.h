
#include "TraceSummary.decl.h"

extern CkGroupID traceSummaryGID;

class TraceSummaryInit : public Chare {
  public:
  TraceSummaryInit(CkArgMsg*) {
    traceSummaryGID = CProxy_TraceSummaryBOC::ckNew();
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
    CkCallback *cb = new CkCallback(CkIndex_TraceSummaryBOC::sendSummaryBOC(NULL), 0, sumProxy);
    CProxy_TraceSummaryBOC(traceSummaryGID).ckSetReductionClient(cb);
  }
  TraceSummaryInit(CkMigrateMessage *m):Chare(m) {}
};

class TraceSummaryBOC : public CBase_TraceSummaryBOC {
private:
  int count;
  BinEntry *bins;
  int  nBins;
  int nTracedPEs;
public:
  TraceSummaryBOC(void): count(0), bins(NULL), nBins(0), nTracedPEs(0) {};
  TraceSummaryBOC(CkMigrateMessage *m):CBase_TraceSummaryBOC(m) {};
  void startSumOnly();
  void askSummary(int size);
  void sendSummaryBOC(CkReductionMsg *);
private:
  void write();
};


