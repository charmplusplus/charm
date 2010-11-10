/** \file GraphBFTLB.C
 *  Author: Abhinav S Bhatele
 *  Date Created: November 10th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#include "GraphBFTLB.h"
#include "ckgraph.h"

CreateLBFunc_Def(GraphBFTLB, "Algorithm which does breadth first traversal for communication aware load balancing")

GraphBFTLB::GraphBFTLB(const CkLBOptions &opt) : CentralLB(opt) {
  lbname = "GraphBFTLB";
  if(CkMyPe() == 0)
    CkPrintf("GraphBFTLB created\n");
}

CmiBool GraphBFTLB::QueryBalanceNow(int _step) {
  return CmiTrue;
}

void GraphBFTLB::work(LDStats *stats) {
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
}

#include "GraphBFTLB.def.h"

/*@}*/
