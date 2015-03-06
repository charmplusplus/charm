/** \file ScotchRefineLB.C
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#include "ScotchRefineLB.h"
#include "ckgraph.h"
#include <scotch.h>

CreateLBFunc_Def(ScotchRefineLB, "Load balancing using the Scotch graph partitioning library")

ScotchRefineLB::ScotchRefineLB(const CkLBOptions &opt) : CBase_ScotchRefineLB(opt) {
  lbname = "ScotchRefineLB";
  if(CkMyPe() == 0)
    CkPrintf("ScotchRefineLB created\n");
}

bool ScotchRefineLB::QueryBalanceNow(int _step) {
  return true;
}

void ScotchRefineLB::work(LDStats *stats) {
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);
  int cost_array[10] = {64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

  /** ============================= STRATEGY ================================ */
  // convert ObjGraph to the Scotch graph
  SCOTCH_Num baseval = 0;			// starting index of vertices
  SCOTCH_Num vertnbr = ogr->vertices.size();	// number of vertices
  SCOTCH_Num edgenbr = 0;			// number of edges

  SCOTCH_Num *oldpemap = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * vertnbr);

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
          ogr->vertices[i].sendToList[j].setNumBytes(ogr->vertices[i].sendToList[j].getNumBytes() + ogr->vertices[i].recvFromList[k].getNumBytes());
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
    oldpemap[i] = ogr->vertices[i].getCurrentPe();
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
    velotab[i] = (int)ceil(ogr->vertices[i].getVertexLoad() * ratio);
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      edgetab[edgeNum] = ogr->vertices[i].sendToList[j].getNeighborId();
      edlotab[edgeNum] = (int) ceil(ogr->vertices[i].sendToList[j].getNumBytes()
          * byteRatio);
      edgeNum++;
    }
    for(j = 0; j < ogr->vertices[i].recvFromList.size(); j++) {
      edgetab[edgeNum] = ogr->vertices[i].recvFromList[j].getNeighborId();
      edlotab[edgeNum] = (int)
          ceil(ogr->vertices[i].recvFromList[j].getNumBytes() * byteRatio);
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

  double migration_cost = 1024.0;

    if (step() == 0) {
      SCOTCH_stratGraphMapBuild (&strat, SCOTCH_STRATBALANCE, parr->procs.size (), 0.01);
    } else {
      SCOTCH_stratGraphMapBuild (&strat, SCOTCH_STRATBALANCE | SCOTCH_STRATREMAP, parr->procs.size (), 0.01);
    }

  SCOTCH_Num *pemap = (SCOTCH_Num *)malloc(sizeof(SCOTCH_Num) * vertnbr);

  // Takes as input the graph, arch graph, strategy, migration cost in
  // double, old mapping and new mapping

  if (step() == 0) {
    SCOTCH_graphPart(&graph, parr->procs.size(), &strat, pemap);
  } else {
    SCOTCH_graphRepart(&graph, parr->procs.size(), oldpemap, migration_cost, NULL, &strat, pemap);
  }
  
  SCOTCH_graphExit (&graph);
  SCOTCH_stratExit (&strat);

  free(verttab);
  free(velotab);
  free(edgetab);
  free(edlotab);

  for(i = baseval; i < vertnbr; i++) {
    if(pemap[i] != ogr->vertices[i].getCurrentPe())
      ogr->vertices[i].setNewPe(pemap[i]);
  }

  free(pemap);
  free(oldpemap);
  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
  delete parr;
  delete ogr;
}

#include "ScotchRefineLB.def.h"

/*@}*/
