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

class CLBSimResults
{
public:
	CLBSimResults(int numPes_);
	void SetProcessorLoad(int pe, double load);
	void PrintSimulationResults();
private:
	double* peLoads;
	int numPes;
	friend class CentralLB;		// so that we don't have to provide little get/put functions
};

#endif /* SIMRESULTS_H */
