/* Declaration of the class that represents the results of the simulation process
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 */
#ifndef SIMRESULTS_H
#define SIMRESULTS_H

#include <charm++.h>
#include "CentralLB.h"

class CentralLB;

class LBSimulation
{
public:
  static int doSimulation;
  static char* dumpFile;
  static int dumpStep;
  static int simProcs;
public:
  LBSimulation(int numPes_);
  ~LBSimulation();
  void SetProcessorLoad(int pe, double load, double bgload);
  void PrintSimulationResults();
private:
  double* peLoads;
  double* bgLoads;
  int numPes;
  friend class CentralLB;   // so that we don't have to provide little get/put functions
};

#endif /* SIMRESULTS_H */
