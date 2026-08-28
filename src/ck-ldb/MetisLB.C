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
#include "LBMemoryContract.h"
#include <algorithm>
#include <cstddef>
#include <metis.h>

extern int quietModeRequested;

static void lbinit()
{
  LBRegisterBalancer<MetisLB>("MetisLB", "Use Metis(tm) to partition object graph");
  LBTurnCommOn();
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
      const auto it = std::partition(inList.begin(), inList.end(), [nId](const CkEdge& e) {
        return e.getNeighborId() != nId;
      });
      // Add the bytes received from vertex nId to the outgoing edge to nId, and then
      // remove those incoming edges
      std::for_each(it, inList.end(), [&outEdge](const CkEdge& e) {
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

  // Memory contract: when per-object device footprints exist, memory becomes
  // a second METIS balancing constraint, so the partitioner spreads resident
  // device bytes as well as load. The verifier in CentralLB::Strategy then
  // hardens METIS's soft tolerance into hard capacity compliance.
  int nConstraints = 1;
#if CMK_CUDA
  LBMemoryModel memModel;
  memModel.build(stats);
  bool memAware = false;
  if (memModel.numDevices() > 0)
    for (int i = 0; i < numVertices && !memAware; i++)
      if (memModel.footprint(ogr->vertices[i].getVertexId()) > 0) memAware = true;
  if (memAware) nConstraints = 2;
#endif

  /* adjacency list */
  std::vector<idx_t> xadj(numVertices + 1);
  /* id of the neighbors */
  std::vector<idx_t> adjncy(numEdges);
  /* weights of the vertices (interleaved when nConstraints > 1) */
  std::vector<idx_t> vwgt((size_t)numVertices * nConstraints);
  /* weights of the edges */
  std::vector<idx_t> adjwgt(numEdges);

  int edgeNum = 0;
  double ratio;
  if (maxLoad == 0)
    ratio = 0;
  else
    ratio = 256.0 / maxLoad;

  for (int i = 0; i < numVertices; i++)
  {
    xadj[i] = edgeNum;
    idx_t* w = &vwgt[(size_t)i * nConstraints];
    if (ogr->vertices[i].getVertexLoad() == 0 && ratio == 0)
      w[0] = 1;
    else
      w[0] = (int)ceil(ogr->vertices[i].getVertexLoad() * ratio);
#if CMK_CUDA
    if (nConstraints > 1) {
      // Footprint in MB, floored at 1 so every object has nonzero weight in
      // the memory dimension (METIS requires positive weights to balance).
      size_t fp = memModel.footprint(ogr->vertices[i].getVertexId());
      w[1] = (idx_t)(fp >> 20) + 1;
    }
#endif
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
  CkPrintf("Metis partitioning in %i partitions\n", parr->availProcSize);
  
  if (parr->availProcSize > 1)
    METIS_PartGraphRecursive(&numVertices, &ncon, xadj.data(), adjncy.data(), vwgt.data(),
                            vsize, adjwgt.data(), &parr->availProcSize, tpwgts, ubvec.data(),
                            options.data(), &edgecut, pemap.data());
  else
    pemap.resize(numVertices, 0);
  
  parr->reassignPeMapToAvailable(pemap);

  if (_lb_args.debug() >= 1)
  {
    CkPrintf("[%d] MetisLB done! \n", CkMyPe());
  }

  for (int i = 0; i < numVertices; i++)
  {
    // Objects that declared themselves non-migratable (setMigratable(false))
    // stay put. Metis partitions the whole graph without any notion of pinned
    // vertices, so without this the partition is applied to them as well and
    // the runtime migrates chares whose own code has assumed it never would --
    // e.g. one that leaves device-resident state out of its pup because it is
    // documented as non-migratable. The partition was computed as if these
    // were free to move, so balance is a little worse than Metis intended;
    // that is the cost of honouring the constraint at all.
    if (!ogr->vertices[i].isMigratable()) continue;
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
