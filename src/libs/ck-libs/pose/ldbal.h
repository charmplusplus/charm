#ifndef LDBAL_H
#define LDBAL_H

#include "ldbal.decl.h"

extern CProxy_LBgroup TheLBG;
extern CProxy_LBstrategy TheLBstrategy;

class LoadReport : public CMessage_LoadReport {
public:
  int PE, peLoad;
};

typedef struct 
{
  int PE, peLoad, startPEidx, endPEidx;
} balanceData;

class BalanceSpecs : public CMessage_BalanceSpecs {
public:

  balanceData sortArray[128];
  int indexArray[128];
  int avgLoad, totalLoad;
};

class LBgroup : public Group {  // Gathers load info locally on a single PE
  // Sends load info to LBstrategy; LBstrategy sends a global report back; 
  // Uses this report to redistribute local objects
#ifdef POSE_STATS_ON
  localStat *localStats;
#endif
  int reportTo, peLoad, busy;
  lbObjects objs;
 public:
  LBgroup(void);
  LBgroup(CkMigrateMessage *) { };
  // local methods
  int computeObjectLoad(POSE_TimeType ovt, POSE_TimeType eet, double rbOh, int sync, POSE_TimeType gvt);
  int computePeLoad();
  int findHeaviestUnder(int loadDiff, int prioLoad, int **mvObjs,
			int pe, int *contrib);
  int objRegister(int arrayIdx, int sync, sim *myPtr);
  void objRemove(int arrayIdx);
  void objUpdate(int ldIdx, POSE_TimeType ovt, POSE_TimeType eet, int ne, double rbOh, int *srVec);
  // entry methods
  void calculateLocalLoad(void);
  void balance(BalanceSpecs *);
};

class LBstrategy : public Group { // Gathers summarized load info from all PEs
  // Combines data in a single report given back to each LBgroup
#ifdef POSE_STATS_ON
  localStat *localStats;
#endif
  int *peLoads;
 public:
  LBstrategy(void);
  LBstrategy(CkMigrateMessage *) { };
  // local methods
  void computeLoadMap(int avgLd, int ttlLd);
  int findMinPE();
  // entry methods
  void recvLoadReport(LoadReport *);
};

#endif
