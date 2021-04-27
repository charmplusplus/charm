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
#include <algorithm>
#include <cstddef>
#include <metis.h>

extern int quietModeRequested;

static void lbinit()
{
  LBRegisterBalancer<MetisLB>("MetisLB", "Use Metis(tm) to partition object graph");
}

MetisLB::MetisLB(const CkLBOptions& opt) : CBase_MetisLB(opt)
{
  lbname = "MetisLB";
  if (CkMyPe() == 0 && !quietModeRequested)
    CkPrintf("CharmLB> MetisLB created.\n");
}

void MetisLB::work(LDStats* stats)
{
  /** ========================== INITIALIZATION ============================= */
  ProcArray* parr = new ProcArray(stats);
  ObjGraph* ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */
  if (_lb_args.debug() >= 2)
  {
    CkPrintf("[%d] In MetisLB Strategy...\n", CkMyPe());
  }

  // convert ObjGraph to the adjacency structure
  idx_t numVertices = ogr->vertices.size();
  size_t numEdges = 0;
  double maxLoad = 0.0;

  /** remove duplicate edges from recvFrom */
  for (auto& vertex : ogr->vertices)
  {
    for (auto& outEdge : vertex.sendToList)
    {
      const auto nId = outEdge.getNeighborId();
      auto& inList = vertex.recvFromList;

      // Partition the incoming edges into {not from vertex nId}, {from vertex nId}
      const auto it = std::partition(inList.begin(), inList.end(), [nId](const Edge& e) {
        return e.getNeighborId() != nId;
      });
      // Add the bytes received from vertex nId to the outgoing edge to nId, and then
      // remove those incoming edges
      std::for_each(it, inList.end(), [&outEdge](const Edge& e) {
        outEdge.setNumBytes(outEdge.getNumBytes() + e.getNumBytes());
      });
      inList.erase(it, inList.end());
    }
  }

  /** the object load is normalized to an integer between 0 and 256 */
  for (const auto& vertex : ogr->vertices)
  {
    maxLoad = std::max(maxLoad, vertex.getVertexLoad());
    numEdges += vertex.sendToList.size() + vertex.recvFromList.size();
  }

  /* adjacency list */
  std::vector<idx_t> xadj(numVertices + 1);
  /* id of the neighbors */
  std::vector<idx_t> adjncy(numEdges);
  /* weights of the vertices */
  std::vector<idx_t> vwgt(numVertices);
  /* weights of the edges */
  std::vector<idx_t> adjwgt(numEdges);

  int edgeNum = 0;
  const double ratio = 256.0 / maxLoad;

  for (int i = 0; i < numVertices; i++)
  {
    xadj[i] = edgeNum;
    vwgt[i] = (int)ceil(ogr->vertices[i].getVertexLoad() * ratio);
    for (const auto& outEdge : ogr->vertices[i].sendToList)
    {
      adjncy[edgeNum] = outEdge.getNeighborId();
      adjwgt[edgeNum] = outEdge.getNumBytes();
      edgeNum++;
    }
    for (const auto& inEdge : ogr->vertices[i].recvFromList)
    {
      adjncy[edgeNum] = inEdge.getNeighborId();
      adjwgt[edgeNum] = inEdge.getNumBytes();
      edgeNum++;
    }
  }

  xadj[numVertices] = edgeNum;
  CkAssert(edgeNum == numEdges);

  std::array<idx_t, METIS_NOPTIONS> options;
  METIS_SetDefaultOptions(options.data());
  // C style numbering
  options[METIS_OPTION_NUMBERING] = 0;
  // options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;

  // number of constraints
  constexpr idx_t numConstraints = 1;
  idx_t ncon = numConstraints;
  // number of partitions
  idx_t numPes = parr->procs.size();
  // allow 10% imbalance
  std::array<real_t, numConstraints> ubvec = {1.1};

  // Specifies size of vertices for computing the total communication volume
  constexpr idx_t* vsize = nullptr;
  // This array of size nparts specifies the desired weight for each partition
  // and setting it to NULL indicates graph should be equally divided among
  // partitions
  constexpr real_t* tpwgts = nullptr;

  // Output fields:
  // number of edges cut by the partitioning
  idx_t edgecut;
  // mapping of objs to partitions
  std::vector<idx_t> pemap(numVertices);

  // METIS always looks at the zeroth element of these, even when there are no edges, so
  // create dummy elements when there are no edges
  if (adjncy.data() == nullptr)
    adjncy = {0};
  if (adjwgt.data() == nullptr)
    adjwgt = {0};

  // numVertices: num vertices in the graph; ncon: num balancing constrains
  // xadj, adjncy: of size n+1 and adjncy of 2m, adjncy[xadj[i]] through and
  // including adjncy[xadj[i+1]-1];
  // vwgt: weight of the vertices; vsize: amt of data that needs to be sent
  // for ith vertex is vsize[i]
  // adjwght: the weight of edges; numPes: total parts
  // tpwghts: target partition weight, can pass NULL to equally divide
  // ubvec: of size ncon to indicate allowed load imbalance tolerance (> 1.0)
  // options: array of options; edgecut: stores the edgecut; pemap: mapping
  METIS_PartGraphRecursive(&numVertices, &ncon, xadj.data(), adjncy.data(), vwgt.data(),
                           vsize, adjwgt.data(), &numPes, tpwgts, ubvec.data(),
                           options.data(), &edgecut, pemap.data());

  if (_lb_args.debug() >= 1)
  {
    CkPrintf("[%d] MetisLB done! \n", CkMyPe());
  }

  for (int i = 0; i < numVertices; i++)
  {
    if (pemap[i] != ogr->vertices[i].getCurrentPe())
      ogr->vertices[i].setNewPe(pemap[i]);
  }

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
  delete parr;
  delete ogr;
}

#include "MetisLB.def.h"

/*@}*/
