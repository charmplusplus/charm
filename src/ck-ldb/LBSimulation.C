/* Implementation of the CLBSimResults class
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 */
#include "LBSimulation.h"

/*****************************************************************************
		Sequentail Simulation 
*****************************************************************************/

int LBSimulation::dumpStep = -1;  	     /// first step number to dump
int LBSimulation::dumpStepSize = 1;          /// number of steps to dump 
char* LBSimulation::dumpFile = (char*)"lbdata.dat"; /// dump file name
int LBSimulation::doSimulation = 0; 	     /// flag if do simulation
int LBSimulation::simStep = -1;              /// first step number to simulate
int LBSimulation::simStepSize = 1;           /// number of steps to simulate
int LBSimulation::simProcs = 0; 	     /// simulation target procs
int LBSimulation::procsChanged = 0;          /// flag if the number of procs has been changed

LBSimulation::LBSimulation(int numPes_) : numPes(numPes_)
{
	peLoads = new double [numPes];
	bgLoads = new double [numPes];
	for(int i = 0; i < numPes; i++)
		peLoads[i] = bgLoads[i] = 0.0;
}

LBSimulation::~LBSimulation()
{
 	delete [] peLoads;
 	delete [] bgLoads;
}

void LBSimulation::reset()
{
  for(int i = 0; i < numPes; i++)
    peLoads[i] = bgLoads[i] = 0.0;
}

void LBSimulation::SetProcessorLoad(int pe, double load, double bgload)
{
	CkAssert(0 <= pe && pe < numPes);
	peLoads[pe] = load;
	bgLoads[pe] = bgload;
}

void LBSimulation::PrintSimulationResults()
{
  int i;
  double minLoad, maxLoad, sum, average;
  sum = .0;
  sum = minLoad = maxLoad = peLoads[0];
  for (i = 1; i < numPes; i++) {
    if (peLoads[i]>maxLoad) maxLoad=peLoads[i];
    else if (peLoads[i]<minLoad) minLoad=peLoads[i];
    sum += peLoads[i];
  }
  average = sum/numPes;
  CmiPrintf("The processor loads are: \n");
  CmiPrintf("PE   (Total Load) (BG Load)\n");
  for(i = 0; i < numPes; i++) {
    CmiPrintf("%-4d %10f %10f", i, peLoads[i], bgLoads[i]);
    CmiPrintf("\n");
  }
  CmiPrintf("Min : %f	Max : %f	Average: %f\n", minLoad, maxLoad, average);
  CmiPrintf("MinObj : %f	MaxObj : %f\n", minObjLoad, maxObjLoad, average);
}

void LBSimulation::PrintDifferences(LBSimulation *realSim, CentralLB::LDStats *stats)
{
  // the number of procs during the simulation and the real execution must be checked by the caller!
  int i;
  // here to print the differences between the predicted (this) and the real (real)
  CmiPrintf("Differences between predicted and real balance:\n");
  CmiPrintf("PE   (Predicted Load) (Real Predicted)  (Difference)  (Real CPU)  (Prediction Error)\n");
  for(i = 0; i < numPes; ++i) {
    CmiPrintf("%-4d %13f %16f %15f %12f %14f\n", i, peLoads[i], realSim->peLoads[i], peLoads[i]-realSim->peLoads[i],
	      stats->procs[i].total_walltime-stats->procs[i].idletime, realSim->peLoads[i]-(stats->procs[i].total_walltime-stats->procs[i].idletime));
  }
}
