/** \file ckgraph.C
 *  Author: Abhinav S Bhatele
 *  Date Created: October 29th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#include "ckgraphTemp.h"


void ProcArrayTemp::convertToInsts(BaseLB::LDStats *stats)
{
  int numPes = stats->nprocs();

	// loop through all pes and convert the exe time to num of instructions each PE is executing.
	// Of course its not correct to actually that we are calculating instructions!
  avgLoad = 0.0;
  for(int pe = 0; pe < numPes; pe++) {
    procs[pe].totalLoad() = procs[pe].totalLoad() * procFreq[pe];
    avgLoad += procs[pe].totalLoad();
//    CkPrintf("PE%d overhead:%f totalLoad:%f \n",pe,procs[pe].overhead(),procs[pe].totalLoad());
  }
  avgLoad /= numPes;
}

void ObjGraphTemp::convertToInsts(BaseLB::LDStats *stats)
{
  for(int vert = 0; vert < stats->n_objs; vert++) {
    vertices[vert].compLoad   = vertices[vert].compLoad * procFreq[vertices[vert].currPe];
  } // end for

}


/*@}*/

