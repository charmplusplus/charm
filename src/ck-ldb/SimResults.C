/* Implementation of the CLBSimResults class
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 */
#include "SimResults.h"

CLBSimResults::CLBSimResults(int numPes_) : numPes(numPes_)
{
	peLoads = new double [numPes];
	bgLoads = new double [numPes];
	for(int i = 0; i < numPes; i++)
		peLoads[i] = bgLoads[i] = 0.0;
}

CLBSimResults::~CLBSimResults()
{
 	delete [] peLoads;
 	delete [] bgLoads;
}

void CLBSimResults::SetProcessorLoad(int pe, double load, double bgload)
{
	CkAssert(0 <= pe && pe < numPes);
	peLoads[pe] = load;
	bgLoads[pe] = bgload;
}

void CLBSimResults::PrintSimulationResults()
{
  ckout << "The processor loads are " << endl;
  CmiPrintf("PE   (Total Load) (BG Load)\n");
  for(int i = 0; i < numPes; i++) {
    CmiPrintf("%-4d %10f %10f", i, peLoads[i], bgLoads[i]);
    CmiPrintf("\n");
  }
}


