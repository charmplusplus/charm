#include "ckcallback-ccs.h"
#include "TraceSummary.decl.h"

extern CkGroupID traceSummaryGID;
extern CProxy_TraceSummaryInit initProxy;
extern bool summaryCcsStreaming;

void masterCollectData(void *data, double currT);

class TraceSummaryInit : public Chare {
 public:
  int lastRequestedIndexBlock;
  int indicesPerBlock;
  double collectionGranularity;
  CkVec<double> *ccsBufferedData;
  int nBufferedBins;
 public:
  TraceSummaryInit(CkArgMsg *m) {
    lastRequestedIndexBlock = 0;
    indicesPerBlock = 1000;
    collectionGranularity = 0.001; // time in seconds
    nBufferedBins = 0;
    traceSummaryGID = CProxy_TraceSummaryBOC::ckNew();
    CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
    initProxy = thishandle;
    CkCallback *cb = new CkCallback(CkIndex_TraceSummaryBOC::sendSummaryBOC(NULL), 0, sumProxy);
    CProxy_TraceSummaryBOC(traceSummaryGID).ckSetReductionClient(cb);
    if (CmiGetArgFlagDesc(m->argv,"+sumCCS",
			  "CCS client connected to Trace Summary")) {
      ccsBufferedData = new CkVec<double>();
      CkPrintf("Trace Summary listening in for CCS Client\n");
      CcsRegisterHandler("CkPerfSummaryCcsClientCB", 
			 CkCallback(CkIndex_TraceSummaryInit::ccsClientRequest(NULL), thishandle));
      CcdCallOnConditionKeep(CcdPERIODIC_1second, masterCollectData, 
			     (void *)this);
      summaryCcsStreaming = CmiTrue;
    } else {
      summaryCcsStreaming = CmiFalse;
    }
  }
  TraceSummaryInit(CkMigrateMessage *m):Chare(m) {}
  void dataCollected(CkReductionMsg *);  
  void ccsClientRequest(CkCcsRequestMsg *m);
};

void masterCollectData(void *data, double currT) {
  // CkPrintf("collectData called\n");
  TraceSummaryInit *sumInitObj = (TraceSummaryInit *)data;

  double startTime = sumInitObj->lastRequestedIndexBlock*
    sumInitObj->collectionGranularity*sumInitObj->indicesPerBlock;
  int numIndicesToGet = (int)floor((currT - startTime)/
				   sumInitObj->collectionGranularity);
  int numBlocksToGet = numIndicesToGet/sumInitObj->indicesPerBlock;
  // **TODO** consider limiting the total number of blocks each collection
  //   request will pick up. This is to limit the amount of perturbation
  //   if it proves to be a problem.
  CProxy_TraceSummaryBOC sumProxy(traceSummaryGID);
  sumProxy.collectData(startTime, 
		       sumInitObj->collectionGranularity,
		       numBlocksToGet*sumInitObj->indicesPerBlock);
  // assume success
  sumInitObj->lastRequestedIndexBlock += numBlocksToGet; 
}

class TraceSummaryBOC : public CBase_TraceSummaryBOC {
private:
  int count;
  BinEntry *bins;
  int  nBins;
  int nTracedPEs;

  int nextBinIndexCcs;
public:
  TraceSummaryBOC(void): count(0), bins(NULL), nBins(0), 
    nTracedPEs(0), nextBinIndexCcs(0) {};
  TraceSummaryBOC(CkMigrateMessage *m):CBase_TraceSummaryBOC(m) {};
  void startSumOnly();
  void askSummary(int size);
  void sendSummaryBOC(CkReductionMsg *);

  void collectData(double startTime, double binSize, int numBins);
private:
  void write();
};


