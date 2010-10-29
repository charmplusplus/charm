/** \file GraphPartLB.C
 *  Author: Abhinav S Bhatele
 *  Date Created: September 3rd, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#include "GraphPartLB.h"
#include "ckgraph.h"

CreateLBFunc_Def(GraphPartLB, "Algorithm which uses graph partitioning for communication aware load balancing")

GraphPartLB::GraphPartLB(const CkLBOptions &opt) : CentralLB(opt) {
  lbname = "GraphPartLB";
  if(CkMyPe() == 0)
    CkPrintf("GraphPartLB created\n");
}

CmiBool GraphPartLB::QueryBalanceNow(int _step) {
  return CmiTrue;
}

void GraphPartLB::work(LDStats *stats) {

}

#include "GraphPartLB.def.h"

/*@}*/
