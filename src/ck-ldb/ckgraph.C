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

ProcArray::ProcArray(BaseLB::LDStats *stats) {
  const int numPes = stats->procs.size();

  // fill the processor array
  procs.resize(numPes);
  availPeMap.resize(numPes);
  std::fill(availPeMap.data(), availPeMap.data() + numPes, -1);

  // Loop through the LDStats structure, copying data into this array and calculating
  //   the average 'totalLoad' of all the PEs
  availProcSize = 0;
  avgLoad = 0.0;
  int currAvailPe = 0;
  for(int pe = 0; pe < numPes; pe++) {
    procs[pe].id        = stats->procs[pe].pe;
    procs[pe].setOverhead(stats->procs[pe].bg_walltime);
    procs[pe].setTotalLoad(stats->procs[pe].total_walltime - stats->procs[pe].idletime);
    procs[pe].available = stats->procs[pe].available;
    availProcSize += (procs[pe].available ? 1 : 0);
    avgLoad += procs[pe].getTotalLoad();
    if (!procs[pe].available)
      currAvailPe++;
    if (currAvailPe < numPes)
      availPeMap[pe] = currAvailPe++;
//		CkPrintf("PE%d overhead:%f totalLoad:%f \n",pe,procs[pe].overhead(),procs[pe].totalLoad());
  }
  availPeMap.resize(availProcSize);
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

  for(int vert = 0; vert < stats->objData.size(); vert++) {
    vertices[vert].id         = vert;
    vertices[vert].compLoad   = stats->objData[vert].wallTime;
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

