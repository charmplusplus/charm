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
	for(int i = 0; i < numPes; i++)
		peLoads[i] = 0.0;
}

void CLBSimResults::SetProcessorLoad(int pe, double load)
{
	CkAssert(0 <= pe && pe < numPes);
	peLoads[pe] = load;
}

void CLBSimResults::PrintSimulationResults()
{
	ckout << "The processor loads are " << endl;
	for(int i = 0; i < numPes; i++)
		ckout << peLoads[i] << " ";
	ckout << endl;
}


