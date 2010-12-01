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
#include "metis.h"

/*extern "C" void METIS_PartGraphRecursive(int*, int*, int*, int*, int*,
			      int*, int*, int*, int*, int*, int*);
extern "C" void METIS_PartGraphKway(int*, int*, int*, int*, int*,
                              int*, int*, int*, int*, int*, int*);
extern "C" void METIS_PartGraphVKway(int*, int*, int*, int*, int*,
			      int*, int*, int*, int*, int*, int*);

// the following are to compute a partitioning with a given partition weights
// "W" means giving weights
extern "C" void METIS_WPartGraphRecursive(int*, int*, int*, int*, int*,
			      int*, int*, int*, float*, int*, int*, int*);
extern "C" void METIS_WPartGraphKway(int*, int*, int*, int*, int*,
			      int*, int*, int*, float*, int*, int*, int*);

// the following are for multiple constraint partition "mC"
extern "C" void METIS_mCPartGraphRecursive(int*, int*, int*, int*, int*, int*,
			      int*, int*, int*, int*, int*, int*);
extern "C" void METIS_mCPartGraphKway(int*, int*, int*, int*, int*, int*,
                              int*, int*, int*, int*, int*, int*, int*);
*/

CreateLBFunc_Def(MetisLB, "Use Metis(tm) to partition object graph")

MetisLB::MetisLB(const CkLBOptions &opt): CentralLB(opt)
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
  int maxBytes = 0, i, j;

  /** both object load and number of bytes exchanged are normalized to an
   *  integer between 0 and 256 */
  for(i = 0; i < numVertices; i++) {
    if(ogr->vertices[i].getVertexLoad() > maxLoad)
      maxLoad = ogr->vertices[i].getVertexLoad();
    numEdges += ogr->vertices[i].edgeList.size();
    for(j = 0; j < ogr->vertices[i].edgeList.size(); j++) {
      if(ogr->vertices[i].edgeList[j].getNumBytes() > maxBytes)
        maxBytes = ogr->vertices[i].edgeList[j].getNumBytes();
    }
  }

  /* adjacency list */
  idxtype *xadj = new idxtype[numVertices + 1];
  /* id of the neighbors */
  idxtype *adjncy = new idxtype[numEdges];
  /* weights of the vertices */
  idxtype *vwgt = new idxtype[numVertices];
  /* weights of the edges */
  idxtype *adjwgt = new idxtype[numEdges];

  int edgeNum = 0;

  for(i = 0; i < numVertices; i++) {
    xadj[i] = edgeNum;
    vwgt[i] = (int)( (ogr->vertices[i].getVertexLoad() * 128) /maxLoad );
    for(j = 0; j < ogr->vertices[i].edgeList.size(); j++) {
      adjncy[edgeNum] = ogr->vertices[i].edgeList[j].getNeighborId();
      adjwgt[edgeNum] = (int)( (ogr->vertices[i].edgeList[j].getNumBytes() * 128) / maxBytes );
      edgeNum++;
    }
  }
  xadj[i] = edgeNum;
  CkAssert(edgeNum == numEdges);

  int wgtflag = 3;	// weights both on vertices and edges
  int numflag = 0;	// C Style numbering
  int options[5];
  options[0] = 0;	// use default values
  int edgecut;		// number of edges cut by the partitioning
  idxtype *pemap;

  int option = 0;
  int numPes = parr->procs.size();
  pemap = new idxtype[numVertices];

  if (0 == option) {
    /** I intended to follow the instruction in the Metis 4.0 manual
     *  which said that METIS_PartGraphKway is preferable to
     *  METIS_PartGraphRecursive, when nparts > 8. However, it turned out that
     *  there is a bug in METIS_PartGraphKway, and the function seg faulted when
     *  nparts = 4 or 9. So right now I just comment that function out and
     *  always use the other one.
     */

    /* if (n_pes > 8)
      METIS_PartGraphKway(&numobjs, xadj, adjncy, objwt, edgewt,
		&wgtflag, &numflag, &n_pes, options, &edgecut, newmap);
    else
      METIS_PartGraphRecursive(&numVertices, xadj, adjncy, vwgt, adjwgt,
		&wgtflag, &numflag, &numPes, options, &edgecut, pemap); */

    METIS_PartGraphRecursive(&numVertices, xadj, adjncy, vwgt, adjwgt,
		&wgtflag, &numflag, &numPes, options, &edgecut, pemap);
  } else if (WEIGHTED == option) {
    // set up the different weights between 0 and 1
    float *tpwgts = new float[numPes];
    for (i = 0; i < numPes; i++) {
      tpwgts[i] = 1.0/(float)numPes;
    }

    if (numPes > 8)
      METIS_WPartGraphKway(&numVertices, xadj, adjncy, vwgt, adjwgt,
	      &wgtflag, &numflag, &numPes, tpwgts, options, &edgecut, pemap);
    else
      METIS_WPartGraphRecursive(&numVertices, xadj, adjncy, vwgt, adjwgt,
	      &wgtflag, &numflag, &numPes, tpwgts, options, &edgecut, pemap);
    delete[] tpwgts;
  } else if (MULTI_CONSTRAINT == option) {
    CkPrintf("Metis load balance strategy: ");
    CkPrintf("multiple constraints not implemented yet.\n");
  }

  delete[] xadj;
  delete[] adjncy;
  delete[] vwgt;
  delete[] adjwgt;

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
}

#include "MetisLB.def.h"

/*@}*/
