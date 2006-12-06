
#include "TraceProjections.decl.h"

extern CkGroupID traceProjectionsGID;

class TraceProjectionsInit : public Chare {
  public:
  TraceProjectionsInit(CkArgMsg*) {
    traceProjectionsGID = CProxy_TraceProjectionsBOC::ckNew();
  }
  TraceProjectionsInit(CkMigrateMessage *m):Chare(m) {}
};

class OutlierStatsMessage : public CMessage_OutlierStatsMessage {
 public:
  double *stats;
};

class OutlierWeightMessage : public CMessage_OutlierWeightMessage {
 public:
  int sourcePe;
  double weight;
};

class OutlierThresholdMessage : public CMessage_OutlierThresholdMessage {
 public:
  double threshold;
};

class TraceProjectionsBOC : public CBase_TraceProjectionsBOC {
private:
  double dummy;
  double endTime;
  double analysisStartTime;
  double *execTimes;
  double weight;
  int encounteredWeights;
  double *weightArray;
  int *mapArray;

  FILE *outlierfp;
  
public:
  TraceProjectionsBOC(void) {};
  TraceProjectionsBOC(CkMigrateMessage *m):CBase_TraceProjectionsBOC(m) {};
  void startOutlierAnalysis();
  void outlierAverageReduction(CkReductionMsg *);
  void calculateWeights(OutlierStatsMessage *);
  void determineOutliers(OutlierWeightMessage *);
  void setOutliers(OutlierThresholdMessage *);
  void startEndTimeAnalysis();
  void endTimeReduction(CkReductionMsg *);
  void finalReduction(CkReductionMsg *);
  void shutdownAnalysis(void);
  void closeTrace(void);
};


