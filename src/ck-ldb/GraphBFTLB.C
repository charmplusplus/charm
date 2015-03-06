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
#include <queue>

CreateLBFunc_Def(GraphBFTLB, "Algorithm which does breadth first traversal for communication aware load balancing")

GraphBFTLB::GraphBFTLB(const CkLBOptions &opt) : CBase_GraphBFTLB(opt) {
  lbname = "GraphBFTLB";
  if(CkMyPe() == 0)
    CkPrintf("GraphBFTLB created\n");
}

bool GraphBFTLB::QueryBalanceNow(int _step) {
  return true;
}

void GraphBFTLB::work(LDStats *stats) {
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);	// Processor Array
  ObjGraph *ogr = new ObjGraph(stats);		// Object Graph

  /** ============================= STRATEGY ================================ */
  double avgLoad = parr->getAverageLoad();
  int numPes = parr->procs.size();

  // CkPrintf("Average Load %g\n\n", avgLoad);
  // for(int i=0; i<numPes; i++)
  //  CkPrintf("PE [%d] %g %g\n", i, parr->procs[i].getTotalLoad(), parr->procs[i].getOverhead());
  parr->resetTotalLoad();

  int start = 0, nextPe = 0;
  std::queue<int> vertexq;

  // start at vertex with id 0
  vertexq.push(start);
  if(parr->procs[nextPe].getTotalLoad() + ogr->vertices[start].getVertexLoad() > avgLoad) {
    nextPe++;
    avgLoad += (avgLoad - parr->procs[nextPe].getTotalLoad())/(numPes-nextPe);
  }
  ogr->vertices[start].setNewPe(nextPe);
  // CkPrintf("[%d] %d %d %g %g %g\n", start, ogr->vertices[start].getCurrentPe(), ogr->vertices[start].getNewPe(), parr->procs[nextPe].getTotalLoad(), ogr->vertices[start].getVertexLoad(), parr->procs[nextPe].getTotalLoad() + ogr->vertices[start].getVertexLoad());
  parr->procs[nextPe].totalLoad() += ogr->vertices[start].getVertexLoad();

  int i, nbr;
  // breadth first traversal
  while(!vertexq.empty()) {
    start = vertexq.front();
    vertexq.pop();

    for(i = 0; i < ogr->vertices[start].sendToList.size(); i++) {
      // look at all neighbors of a node in the queue and map them while
      // inserting them in the queue (so we can look at their neighbors next)
      nbr = ogr->vertices[start].sendToList[i].getNeighborId();
      if(ogr->vertices[nbr].getNewPe() == -1) {
	vertexq.push(nbr);

	if(parr->procs[nextPe].getTotalLoad() + ogr->vertices[nbr].getVertexLoad() > avgLoad) {
	  nextPe++;
	  avgLoad += (avgLoad - parr->procs[nextPe].getTotalLoad())/(numPes-nextPe);
	}
	ogr->vertices[nbr].setNewPe(nextPe);
	// CkPrintf("[%d] %d %d %g %g %g\n", nbr, ogr->vertices[nbr].getCurrentPe(), ogr->vertices[nbr].getNewPe(), parr->procs[nextPe].getTotalLoad(), ogr->vertices[start].getVertexLoad(), parr->procs[nextPe].getTotalLoad() + ogr->vertices[start].getVertexLoad());
	parr->procs[nextPe].totalLoad() += ogr->vertices[nbr].getVertexLoad();
      }
    } // end of for loop
  } // end of while loop

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);		// Send decisions back to LDStats
}

#include "GraphBFTLB.def.h"

/*@}*/
