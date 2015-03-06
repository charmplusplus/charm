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
#include <scotch.h>

CreateLBFunc_Def(ScotchLB, "Load balancing using the Scotch graph partitioning library")

ScotchLB::ScotchLB(const CkLBOptions &opt) : CBase_ScotchLB(opt) {
  lbname = "ScotchLB";
  if(CkMyPe() == 0)
    CkPrintf("ScotchLB created\n");
}

bool ScotchLB::QueryBalanceNow(int _step) {
  return true;
}

void ScotchLB::work(LDStats *stats) {
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);
  double start_time = CmiWallTimer();

  /** ============================= STRATEGY ================================ */
  // convert ObjGraph to the Scotch graph
  SCOTCH_Num baseval = 0;			// starting index of vertices
  SCOTCH_Num vertnbr = ogr->vertices.size();	// number of vertices
  SCOTCH_Num edgenbr = 0;			// number of edges

  double maxLoad = 0.0;
  double minLoad = 0.0;
  if (vertnbr > 0) {
    minLoad = ogr->vertices[baseval].getVertexLoad();
  }

  long maxBytes = 1;
  int i, j, k, vert;


  /** remove duplicate edges from recvFrom */
  for(i = baseval; i < vertnbr; i++) {
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      vert = ogr->vertices[i].sendToList[j].getNeighborId();
      for(k = 0; k < ogr->vertices[i].recvFromList.size(); k++) {
        if(ogr->vertices[i].recvFromList[k].getNeighborId() == vert) {
          ogr->vertices[i].sendToList[j].setNumBytes(
              ogr->vertices[i].sendToList[j].getNumBytes() +
              ogr->vertices[i].recvFromList[k].getNumBytes());
          ogr->vertices[i].recvFromList.erase(ogr->vertices[i].recvFromList.begin() + k);
        }
      }
    }
  }
  /** the object load is normalized to an integer between 0 and 256 */
  for(i = baseval; i < vertnbr; i++) {
    if(ogr->vertices[i].getVertexLoad() > maxLoad)
      maxLoad = ogr->vertices[i].getVertexLoad();

    if (ogr->vertices[i].getVertexLoad() < minLoad) {
      minLoad = ogr->vertices[i].getVertexLoad();
    }
    edgenbr += ogr->vertices[i].sendToList.size() + ogr->vertices[i].recvFromList.size();
  }

  for(i = baseval; i < vertnbr; i++) {
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      if (ogr->vertices[i].sendToList[j].getNumBytes() > maxBytes) {
        maxBytes = ogr->vertices[i].sendToList[j].getNumBytes();
      }
    }
    for(j = 0; j < ogr->vertices[i].recvFromList.size(); j++) {
      if (ogr->vertices[i].recvFromList[j].getNumBytes() > maxBytes) {
        maxBytes = ogr->vertices[i].recvFromList[j].getNumBytes();
      }
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
  double ratio = 256.0/maxLoad;
  double byteRatio = 1024.0/maxBytes;

  for(i = baseval; i < vertnbr; i++) {
    verttab[i] = edgeNum;
    velotab[i] = (SCOTCH_Num) ceil(ogr->vertices[i].getVertexLoad() * ratio);
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      edgetab[edgeNum] = ogr->vertices[i].sendToList[j].getNeighborId();
      edlotab[edgeNum] = (int) ceil(ogr->vertices[i].sendToList[j].getNumBytes() * byteRatio);
      edgeNum++;
    }
    for(j = 0; j < ogr->vertices[i].recvFromList.size(); j++) {
      edgetab[edgeNum] = ogr->vertices[i].recvFromList[j].getNeighborId();
      edlotab[edgeNum] = (int) ceil(ogr->vertices[i].recvFromList[j].getNumBytes() * byteRatio);
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

  SCOTCH_stratGraphMapBuild (&strat, SCOTCH_STRATBALANCE, parr->procs.size (), 0.01);
  SCOTCH_Num *pemap = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * vertnbr);

  SCOTCH_graphPart(&graph, parr->procs.size(), &strat, pemap);


  SCOTCH_graphExit (&graph);
  SCOTCH_stratExit (&strat);

  free(verttab);
  free(velotab);
  free(edgetab);
  free(edlotab);
  for(i = baseval; i < vertnbr; i++) {
    if(pemap[i] != ogr->vertices[i].getCurrentPe()) {
      ogr->vertices[i].setNewPe(pemap[i]);
    }
  }

  free(pemap);
  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
  delete parr;
  delete ogr;
}

#include "ScotchLB.def.h"

/*@}*/
