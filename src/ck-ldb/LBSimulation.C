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

/*****************************************************************************
		LBInfo: evaluation information for LBStats  
*****************************************************************************/

LBInfo::LBInfo(int count): minObjLoad(0.0), maxObjLoad(0.0), numPes(count)
{
  peLoads = new double[numPes]; 
  objLoads = new double[numPes]; 
  comLoads = new double[numPes]; 
  bgLoads = new double[numPes]; 
  clear();
}

LBInfo::~LBInfo()
{
  // only free when it is allocated in the constructor
  if (peLoads && bgLoads) {
    delete [] bgLoads;
    delete [] comLoads;
    delete [] objLoads;
    delete [] peLoads;
  }
}

void LBInfo::clear()
{
  for (int i=0; i<numPes; i++) {
    peLoads[i] = 0.0;
    if (objLoads) objLoads[i] = 0.0;
    if (comLoads) comLoads[i] = 0.0;
    if (bgLoads)  bgLoads[i] = 0.0;
  }
  minObjLoad = 0.0;
  maxObjLoad = 0.0;
}

void LBInfo::print()
{
  int i;
  double minLoad, maxLoad, maxObjLoad, maxComLoad, sum, average;
  sum = .0;
  sum = minLoad = maxLoad = peLoads[0];
  maxObjLoad = objLoads[0];
  maxComLoad = comLoads[0];
  for (i = 1; i < numPes; i++) {
    double load = peLoads[i];
    if (load>maxLoad) maxLoad=load;
    else if (peLoads[i]<minLoad) minLoad=load;
    if (objLoads[i]>maxObjLoad) maxObjLoad = objLoads[i];
    if (comLoads[i]>maxComLoad) maxComLoad = comLoads[i];
    sum += load;
  }
  average = sum/numPes;
  CmiPrintf("The processor loads are: \n");
  CmiPrintf("PE   (Total Load) (Obj Load) (Comm Load) (BG Load)\n");
  for(i = 0; i < numPes; i++) {
    CmiPrintf("%-4d %10f %10f %10f %10f\n", i, peLoads[i], objLoads[i], comLoads[i], bgLoads[i]);
  }
  CmiPrintf("max: %10f %10f %10f\n", maxLoad, maxObjLoad, maxComLoad);
  CmiPrintf("Min : %f	Max : %f	Average: %f\n", minLoad, maxLoad, average);
  CmiPrintf("MinObj : %f	MaxObj : %f\n", minObjLoad, maxObjLoad, average);
}

////////////////////////////////////////////////////////////////////////////

LBSimulation::LBSimulation(int numPes_) : numPes(numPes_), lbinfo(numPes_)
{
}

LBSimulation::~LBSimulation()
{
}

void LBSimulation::reset()
{
  lbinfo.clear();
}

void LBSimulation::SetProcessorLoad(int pe, double load, double bgload)
{
	CkAssert(0 <= pe && pe < numPes);
	lbinfo.peLoads[pe] = load;
	lbinfo.bgLoads[pe] = bgload;
}

void LBSimulation::PrintSimulationResults()
{
  lbinfo.print();
}

void LBSimulation::PrintDifferences(LBSimulation *realSim, CentralLB::LDStats *stats)
{
  double *peLoads = lbinfo.peLoads;
  double *realPeLoads = realSim->lbinfo.peLoads;

  // the number of procs during the simulation and the real execution must be checked by the caller!
  int i;
  // here to print the differences between the predicted (this) and the real (real)
  CmiPrintf("Differences between predicted and real balance:\n");
  CmiPrintf("PE   (Predicted Load) (Real Predicted)  (Difference)  (Real CPU)  (Prediction Error)\n");
  for(i = 0; i < numPes; ++i) {
    CmiPrintf("%-4d %13f %16f %15f %12f %14f\n", i, peLoads[i], realPeLoads[i], peLoads[i]-realPeLoads[i],
	      stats->procs[i].total_walltime-stats->procs[i].idletime, realPeLoads[i]-(stats->procs[i].total_walltime-stats->procs[i].idletime));
  }
}
