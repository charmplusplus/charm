/** \file MetisLB.C
 *
 *  Updated by Abhinav Bhatele, 2010-11-26 to use ckgraph
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "MetisLB.h"
#include "ckgraph.h"
#include <metis.h>


CreateLBFunc_Def(MetisLB, "Use Metis(tm) to partition object graph")

MetisLB::MetisLB(const CkLBOptions &opt): CBase_MetisLB(opt)
{
  lbname = "MetisLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] MetisLB created\n",CkMyPe());
}

void MetisLB::work(LDStats* stats)
{
  /** ========================== INITIALIZATION ============================= */
  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */
  if (_lb_args.debug() >= 2) {
    CkPrintf("[%d] In MetisLB Strategy...\n", CkMyPe());
  }

  // convert ObjGraph to the adjacency structure
  int numVertices = ogr->vertices.size();	// number of vertices
  int numEdges = 0;				// number of edges

  double maxLoad = 0.0;
  int i, j, k, vert;

  /** remove duplicate edges from recvFrom */
  for(i = 0; i < numVertices; i++) {
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
  for(i = 0; i < numVertices; i++) {
    if(ogr->vertices[i].getVertexLoad() > maxLoad)
      maxLoad = ogr->vertices[i].getVertexLoad();
    numEdges = numEdges + ogr->vertices[i].sendToList.size() + ogr->vertices[i].recvFromList.size();
  }

  /* adjacency list */
  idx_t *xadj = new idx_t[numVertices + 1];
  /* id of the neighbors */
  idx_t *adjncy = new idx_t[numEdges];
  /* weights of the vertices */
  idx_t *vwgt = new idx_t[numVertices];
  /* weights of the edges */
  idx_t *adjwgt = new idx_t[numEdges];

  int edgeNum = 0;
  double ratio = 256.0/maxLoad;

  for(i = 0; i < numVertices; i++) {
    xadj[i] = edgeNum;
    vwgt[i] = (int)ceil(ogr->vertices[i].getVertexLoad() * ratio);
    for(j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      adjncy[edgeNum] = ogr->vertices[i].sendToList[j].getNeighborId();
      adjwgt[edgeNum] = ogr->vertices[i].sendToList[j].getNumBytes();
      edgeNum++;
    }
    for(j = 0; j < ogr->vertices[i].recvFromList.size(); j++) {
      adjncy[edgeNum] = ogr->vertices[i].recvFromList[j].getNeighborId();
      adjwgt[edgeNum] = ogr->vertices[i].recvFromList[j].getNumBytes();
      edgeNum++;
    }
  }
  xadj[i] = edgeNum;
  CkAssert(edgeNum == numEdges);

  idx_t edgecut;		// number of edges cut by the partitioning
  idx_t *pemap;

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  //options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;
  // C style numbering
  options[METIS_OPTION_NUMBERING] = 0;

  // number of constrains
  idx_t ncon = 1;
  // number of partitions
  idx_t numPes = parr->procs.size();
  real_t ubvec[ncon];
  // allow 10% imbalance
  ubvec[0] = 1.1;

  // mapping of objs to partitions
  pemap = new idx_t[numVertices];

  // Specifies size of vertices for computing the total communication volume
  idx_t *vsize = NULL;
  // This array of size nparts specifies the desired weight for each partition
  // and setting it to NULL indicates graph should be equally divided among
  // partitions
  real_t *tpwgts = NULL;

  int option = 0;
  if (WEIGHTED == option) {
    // set up the different weights between 0 and 1
    tpwgts = new real_t[numPes];
    for (i = 0; i < numPes; i++) {
      tpwgts[i] = 1.0/(real_t)numPes;
    }
  } else if (MULTI_CONSTRAINT == option) {
    CkAbort("Multiple constraints not implemented.\n");
  }

  // numVertices: num vertices in the graph; ncon: num balancing constrains
  // xadj, adjncy: of size n+1 and adjncy of 2m, adjncy[xadj[i]] through and
  // including adjncy[xadj[i+1]-1];
  // vwgt: weight of the vertices; vsize: amt of data that needs to be sent
  // for ith vertex is vsize[i]
  // adjwght: the weight of edges; numPes: total parts
  // tpwghts: target partition weight, can pass NULL to equally divide
  // ubvec: of size ncon to indicate allowed load imbalance tolerance (> 1.0)
  // options: array of options; edgecut: stores the edgecut; pemap: mapping
  METIS_PartGraphRecursive(&numVertices, &ncon,  xadj, adjncy, vwgt, vsize, adjwgt,
      &numPes, tpwgts, ubvec, options, &edgecut, pemap);

  delete[] xadj;
  delete[] adjncy;
  delete[] vwgt;
  delete[] adjwgt;
  delete[] vsize;
  delete[] tpwgts;

  if (_lb_args.debug() >= 1) {
   CkPrintf("[%d] MetisLB done! \n", CkMyPe());
  }

  for(i = 0; i < numVertices; i++) {
    if(pemap[i] != ogr->vertices[i].getCurrentPe())
      ogr->vertices[i].setNewPe(pemap[i]);
  }

  delete[] pemap;

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
  delete parr;
  delete ogr;
}

#include "MetisLB.def.h"

/*@}*/
