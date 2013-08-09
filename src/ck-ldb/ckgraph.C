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
  int numPes = stats->nprocs();
  // fill the processor array
  procs.resize(numPes);

  // Loop through the LDStats structure, copying data into this array and calculating
  //   the average 'totalLoad' of all the PEs
  avgLoad = 0.0;
  for(int pe = 0; pe < numPes; pe++) {
    procs[pe].id        = stats->procs[pe].pe;
    procs[pe].overhead()  = stats->procs[pe].bg_walltime;
    procs[pe].totalLoad() = stats->procs[pe].total_walltime - stats->procs[pe].idletime;
    procs[pe].available = stats->procs[pe].available;
    avgLoad += procs[pe].totalLoad();
//		CkPrintf("PE%d overhead:%f totalLoad:%f \n",pe,procs[pe].overhead(),procs[pe].totalLoad());
  }
  avgLoad /= numPes;
}

void ProcArray::resetTotalLoad() {
  for(int pe = 0; pe < procs.size(); pe++)
    procs[pe].totalLoad() = procs[pe].overhead();
}

ObjGraph::ObjGraph(BaseLB::LDStats *stats) {
  // fill the vertex list
  vertices.resize(stats->n_objs);

  for(int vert = 0; vert < stats->n_objs; vert++) {
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

  for(int edge = 0; edge < stats->n_comm; edge++) {
    LDCommData &commData = stats->commData[edge];

    // ensure that the message is not from a processor but from an object
    // and that the type is an object to object message
    if( (!commData.from_proc()) && (commData.recv_type()==LD_OBJ_MSG) ) {
      from = stats->getHash(commData.sender);
      to = stats->getHash(commData.receiver.get_destObj());

      vertices[from].sendToList.push_back(Edge(to, commData.messages, commData.bytes));
      vertices[to].recvFromList.push_back(Edge(from, commData.messages, commData.bytes));
    } //else if a multicast list
    else if((!commData.from_proc()) && (commData.recv_type() == LD_OBJLIST_MSG)) {
      int nobjs, offset;
      LDObjKey *objs = commData.receiver.get_destObjs(nobjs);
      McastSrc sender(nobjs, commData.messages, commData.bytes);

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
  for(int vert = 0; vert < stats->n_objs; vert++) {
    if(vertices[vert].newPe != -1) {
      stats->to_proc[vertices[vert].id] = vertices[vert].newPe;
    }
  }
}

/*@}*/

