
#include "TraceSummary.decl.h"

extern CkGroupID traceSummaryGID;

class TraceSummaryInit : public Chare {
  public:
  TraceSummaryInit(CkArgMsg*) {
    traceSummaryGID = CProxy_TraceSummaryBOC::ckNew();
  }
  TraceSummaryInit(CkMigrateMessage *m) {}
};

class TraceSummaryBOC : public CBase_TraceSummaryBOC {
private:
  int count;
  BinEntry *bins;
  int  nBins;
public:
  TraceSummaryBOC(void): count(0), bins(NULL), nBins(0) {};
  TraceSummaryBOC(CkMigrateMessage *m) {};
  void askSummary();
  void sendSummaryBOC(int n, BinEntry *b);
private:
  void write();
};


