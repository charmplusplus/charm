/** \file ckgraph.C
 *  Author: Abhinav S Bhatele
 *  Date Created: October 29th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#include "ckgraph.h"
#include "LBLoadDim.h"

ProcArray::ProcArray(BaseLB::LDStats *stats) {
  const int numPes = stats->procs.size();

  // fill the processor array
  procs.resize(numPes);
  availPeMap.clear();
  availPeMap.reserve(numPes);

  // Loop through the LDStats structure, copying data into this array and calculating
  //   the average 'totalLoad' of all the PEs
  avgLoad = 0.0;
  for(int pe = 0; pe < numPes; pe++) {
    procs[pe].id        = stats->procs[pe].pe;
    procs[pe].setOverhead(stats->procs[pe].bg_walltime);
    procs[pe].setTotalLoad(stats->procs[pe].total_walltime - stats->procs[pe].idletime);
    procs[pe].available = stats->procs[pe].available;
    //CkPrintf("%i avail = %d\n", pe, procs[pe].available);
    avgLoad += procs[pe].getTotalLoad();
    // availPeMap[k] is the PE that the k-th available processor is. A graph
    // partitioner numbers its parts 0..availProcSize-1 over the available PEs
    // only, and reassignPeMapToAvailable turns a part number back into a real
    // PE by indexing this. It used to be filled the other way round -- PE to
    // running index -- with a second stray increment on every unavailable PE,
    // and then truncated to availProcSize. With every PE available that is the
    // identity and works; with even one unavailable it hands back the wrong PE
    // or the -1 fill, and -1 goes on to setNewPe as a migration destination.
    if (procs[pe].available) availPeMap.push_back(pe);
//		CkPrintf("PE%d overhead:%f totalLoad:%f \n",pe,procs[pe].overhead(),procs[pe].totalLoad());
  }
  availProcSize = availPeMap.size();
  avgLoad /= numPes;
}

void ProcArray::reassignPeMapToAvailable(std::vector<int32_t> &pemap) {
  for (int i = 0; i < pemap.size(); i++)
    pemap[i] = availPeMap[pemap[i]];
}

void ProcArray::resetTotalLoad() {
  for(int pe = 0; pe < procs.size(); pe++)
    procs[pe].setTotalLoad(procs[pe].getOverhead());
}

ObjGraph::ObjGraph(BaseLB::LDStats *stats) {
  // fill the vertex list
  vertices.resize(stats->objData.size());

  // Which resource the graph strategies balance: the dimension that binds the
  // step, measured from these stats (LBLoadDim.h), unless +LBDiffusionGpuDim
  // or +LBDiffusionHostDim says otherwise. On a GPU-resident application the
  // host side only enqueues kernels and returns, so wallTime is launch
  // overhead and partitioning on it balances nothing real; on a host-bound one
  // the reverse holds, and partitioning on device occupancy spends every move
  // on the resource nobody waits for. Set here rather than in each strategy so
  // every ckgraph consumer (MetisLB, Scotch*, RecBipartLB, ZoltanLB) agrees on
  // what a vertex weight means.
  //
  // gpuTime is only populated for strategies whose base class collects the
  // CUPTI loads (CentralLB::CallLB, DistBaseLB::barrierDone); a strategy that
  // does neither sees zeros here, the host binds, and nothing changes.
  deviceDim = lbResolveDeviceDim(lbCriticalityOf(stats));
  const bool useGpuDim = deviceDim;

  for(int vert = 0; vert < stats->objData.size(); vert++) {
    vertices[vert].id         = vert;
#if CMK_CUDA
    vertices[vert].compLoad   = useGpuDim ? stats->objData[vert].gpuTime
                                          : stats->objData[vert].wallTime;
#else
    vertices[vert].compLoad   = stats->objData[vert].wallTime;
#endif
    vertices[vert].migratable = stats->objData[vert].migratable;
    vertices[vert].currPe     = stats->from_proc[vert];
    vertices[vert].newPe      = -1;
    vertices[vert].pupSize    = pup_decodeSize(stats->objData[vert].pupSize);
  } // end for

  // fill the edge list for each vertex
  stats->makeCommHash();

  int from, to;

  for(auto& commData : stats->commData) {
    // ensure that the message is not from a processor but from an object
    // and that the type is an object to object message
    if( (!commData.from_proc()) && (commData.recv_type()==LD_OBJ_MSG) ) {
      from = stats->getHash(commData.sender);
      to = stats->getHash(commData.receiver.get_destObj());

      vertices[from].sendToList.emplace_back(to, commData.messages, commData.bytes);
      vertices[to].recvFromList.emplace_back(from, commData.messages, commData.bytes);
    } //else if a multicast list
    else if((!commData.from_proc()) && (commData.recv_type() == LD_OBJLIST_MSG)) {
      int nobjs, offset;
      const LDObjKey *objs = commData.receiver.get_destObjs(nobjs);
      McastSrc sender(commData.messages, commData.bytes);

      from = stats->getHash(commData.sender);
      offset = vertices[from].mcastToList.size();

      for(int i = 0; i < nobjs; i++) {
        int idx = stats->getHash(objs[i]);
        CmiAssert(idx != -1);
        vertices[idx].mcastFromList.push_back(McastDest(from, offset,
        commData.messages, commData.bytes));
        sender.destList.push_back(idx);
      }
      vertices[from].mcastToList.push_back(sender);
    }
  } // end for
}

void ObjGraph::convertDecisions(BaseLB::LDStats *stats) {
  for(const auto& vertex : vertices) {
    if(vertex.newPe != -1) {
      stats->to_proc[vertex.id] = vertex.newPe;
    }
  }
}

/*@}*/

