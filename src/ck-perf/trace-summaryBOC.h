
#include "TraceSummary.decl.h"

extern CkGroupID traceSummaryGID;

class TraceSummaryInit : public Chare {
  public:
  TraceSummaryInit(CkArgMsg*) {
    traceSummaryGID = CProxy_TraceSummaryBOC::ckNew();
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
  void askSummary();
  void sendSummaryBOC(int traced, int n, BinEntry *b);
private:
  void write();
};


