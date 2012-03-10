/* Declaration of the class that represents the results of the simulation process
 */
#ifndef SIMRESULTS_H
#define SIMRESULTS_H

#include <charm++.h>
#include "CentralLB.h"

class LBSimulation
{
public:
  static int doSimulation;
  static char* dumpFile;
  static int dumpStep;
  static int dumpStepSize;
  static int simStep;
  static int simStepSize;
  static int simProcs;
  static int procsChanged;

  static int showDecisionsOnly;
public:
  LBSimulation(int numPes_);
  ~LBSimulation();
  void reset();
  void SetProcessorLoad(int pe, double load, double bgload);
  void PrintSimulationResults();
  void PrintDecisions(LBMigrateMsg *m, char *simFileName, int peCount);
  void PrintDifferences(LBSimulation *realSim, BaseLB::LDStats *stats);
private:
  LBInfo  lbinfo;
  int numPes;
  friend class CentralLB;   // so that we don't have to provide little get/put functions
};

#endif /* SIMRESULTS_H */
