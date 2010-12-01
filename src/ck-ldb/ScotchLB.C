/** \file ScotchLB.C
 *  Authors: Abhinav S Bhatele (bhatele@illinois.edu)
 *           Sebastien Fourestier (fouresti@labri.fr)
 *  Date Created: November 25th, 2010
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#include "ScotchLB.h"
#include "ckgraph.h"
#include "scotch.h"

CreateLBFunc_Def(ScotchLB, "Load balancing using the Scotch graph partitioning library")

ScotchLB::ScotchLB(const CkLBOptions &opt) : CentralLB(opt) {
  lbname = "ScotchLB";
  if(CkMyPe() == 0)
    CkPrintf("ScotchLB created\n");
}

CmiBool ScotchLB::QueryBalanceNow(int _step) {
  return CmiTrue;
}

void ScotchLB::work(LDStats *stats) {
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */
  // convert ObjGraph to the Scotch graph
  SCOTCH_Num baseval = 0;			// starting index of vertices
  SCOTCH_Num vertnbr = ogr->vertices.size();	// number of vertices
  SCOTCH_Num edgenbr = 0;			// number of edges

  double maxLoad = 0.0;
  int maxBytes = 0, i, j;

  /** both object load and number of bytes exchanged are normalized to an
   *  integer between 0 and 256 */
  for(i = baseval; i < vertnbr; i++) {
    if(ogr->vertices[i].getVertexLoad() > maxLoad)
      maxLoad = ogr->vertices[i].getVertexLoad();
    edgenbr += ogr->vertices[i].edgeList.size();
    for(j = 0; j < ogr->vertices[i].edgeList.size(); j++) {
      if(ogr->vertices[i].edgeList[j].getNumBytes() > maxBytes)
	maxBytes = ogr->vertices[i].edgeList[j].getNumBytes();
    }
  }

  /* adjacency list */
  SCOTCH_Num *verttab = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * (vertnbr+1));
  /* loads of vertices */
  SCOTCH_Num *velotab = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * vertnbr);
  /* id of the neighbors */
  SCOTCH_Num *edgetab = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * edgenbr);
  /* number of bytes exchanged */
  SCOTCH_Num *edlotab = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * edgenbr);

  int edgeNum = 0;

  for(i = baseval; i < vertnbr; i++) {
    verttab[i] = edgeNum;
    velotab[i] = (int)( (ogr->vertices[i].getVertexLoad() * 128) /maxLoad );
    for(j = 0; j < ogr->vertices[i].edgeList.size(); j++) {
      edgetab[edgeNum] = ogr->vertices[i].edgeList[j].getNeighborId();
      edlotab[edgeNum] = (int)( (ogr->vertices[i].edgeList[j].getNumBytes() * 128) / maxBytes );
      edgeNum++;
    }
  }
  verttab[i] = edgeNum;
  CkAssert(edgeNum == edgenbr);

  SCOTCH_Graph graph;		// Graph to partition
  SCOTCH_Strat strat;		// Strategy to achieve partitioning

  /* Initialize data structures */
  SCOTCH_graphInit (&graph);
  SCOTCH_stratInit (&strat);

  SCOTCH_graphBuild (&graph, baseval, vertnbr, verttab, NULL, velotab, NULL, edgenbr, edgetab, edlotab); 
  SCOTCH_graphCheck (&graph);

  // SCOTCH_stratGraphMap (&strat, "m{type=h,vert=80,low=r{job=t,map=t,poli=S,sep=h{pass=10}},asc=b{bnd=d{dif=1,rem=1,pass=40},org=}f{bal=0.01,move=80}}");

  SCOTCH_stratGraphMap (&strat, "r{job=t,map=t,poli=S,sep=(m{type=h,vert=80,low=h{pass=10}f{bal=0.001,move=80},asc=b{bnd=1f{bal=0.001,move=80},org=f{bal=0.001,move=80}}}|m{type=h,vert=80,low=h{pass=10}f{bal=0.001,move=80},asc=b{bnd=1f{bal=0.001,move=80},org=f{bal=0.001,move=80}}})}");

  SCOTCH_Num *pemap = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * vertnbr);

  SCOTCH_graphPart(&graph, parr->procs.size(), &strat, pemap);

  SCOTCH_graphExit (&graph);
  SCOTCH_stratExit (&strat);

  for(i = baseval; i < vertnbr; i++) {
    if(pemap[i] != ogr->vertices[i].getCurrentPe())
      ogr->vertices[i].setNewPe(pemap[i]);
  }

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
}

#include "ScotchLB.def.h"

/*@}*/
