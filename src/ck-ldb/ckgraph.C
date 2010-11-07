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
  // fill the processor array
  procs.resize(stats->nprocs());

  for(int pe = 0; pe < stats->nprocs(); pe++) {
    procs[pe].id        = stats->procs[pe].pe;
    procs[pe].overhead  = stats->procs[pe].bg_walltime;
    procs[pe].available = stats->procs[pe].available;
  }
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

      vertices[from].edgeList.push_back(Edge(to, commData.messages, commData.bytes));
    }
  } // end for
}

void ObjGraph::convertDecisions(BaseLB::LDStats *stats) {
  for(int vert = 0; vert < stats->n_objs; vert++) {
    if(vertices[vert].newPe != -1) {
      stats->to_proc[vert] = vertices[vert].newPe;
    }
  }
}

/*@}*/

