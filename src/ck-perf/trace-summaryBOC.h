#include "ckcallback-ccs.h"
#include "TraceSummary.decl.h"
#include <deque>

extern CkGroupID traceSummaryGID;
extern bool summaryCcsStreaming;

class TraceSummaryInit : public Chare {
 public:
  TraceSummaryInit(CkArgMsg *m) {
    delete m;
    traceSummaryGID = CProxy_TraceSummaryBOC::ckNew();
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);

    CkCallback *cb = new CkCallback(CkIndex_TraceSummaryBOC::sendSummaryBOC(NULL), 0, sumProxy);
    CProxy_TraceSummaryBOC(traceSummaryGID).ckSetReductionClient(cb);

    // No CCS Streaming support until user-code requires it.
    summaryCcsStreaming = false;
  }
  TraceSummaryInit(CkMigrateMessage *m):Chare(m) {}
};

class TraceSummaryBOC : public CBase_TraceSummaryBOC {
private:
  int count;
  BinEntry *bins;
  int  nBins;
  int nTracedPEs;

  bool firstTime; // used to make sure traceEnableCCS only has an effect the first time.
  double _maxBinSize; //the max bin size collected from all processors
public:
  /* CCS support variables */
  int lastRequestedIndexBlock;
  int indicesPerBlock;
  double collectionGranularity; /* time in seconds */
  int nBufferedBins;
  CkVec<double> *ccsBufferedData;
  int nextBinIndexCcs;

public:
  TraceSummaryBOC(void): count(0), bins(NULL), nBins(0), 
    nTracedPEs(0), nextBinIndexCcs(0), firstTime(true) {};
  TraceSummaryBOC(CkMigrateMessage *m):CBase_TraceSummaryBOC(m) {};
  void startSumOnly();
  void askSummary(int size);
  void sendSummaryBOC(CkReductionMsg *);

  /* CCS support methods/entry methods */
  void initCCS();
  void ccsRequestSummaryDouble(CkCcsRequestMsg *m);
  void ccsRequestSummaryUnsignedChar(CkCcsRequestMsg *m);

  void collectSummaryData(double startTime, double binSize, int numBins);
  void summaryDataCollected(CkReductionMsg *);

  void traceSummaryParallelShutdown(int pe);
  void maxBinSize(CkReductionMsg *msg);
  void shrink(double _maxBinSize);

  void sumData(CkReductionMsg *msg);

private:
  void write();
};

void startCollectData(void *data, double currT);
