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

GraphPartLB::GraphPartLB(const CkLBOptions &opt) : CBase_GraphPartLB(opt) {
  lbname = "GraphPartLB";
  if(CkMyPe() == 0)
    CkPrintf("GraphPartLB created\n");
}

bool GraphPartLB::QueryBalanceNow(int _step) {
  return true;
}

void GraphPartLB::work(LDStats *stats) {
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
}

#include "GraphPartLB.def.h"

/*@}*/
