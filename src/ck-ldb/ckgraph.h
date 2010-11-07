/** \file ckgraph.h
 *  Author: Abhinav S Bhatele
 *  Date Created: October 29th, 2010
 *  E-mail: bhatele@illinois.edu
 *
 */

/**
 *  \addtogroup CkLdb
 */

/*@{*/

#ifndef _CKGRAPH_H_
#define _CKGRAPH_H_

#include <vector>
#include "BaseLB.h"

class ProcInfo {
  friend class ProcArray;

  private:
    int id;		// CkMyPe of the processor
    double overhead;	// previously called background load (bg_walltime)
    bool available;	// if the processor is available
};

class ProcArray {
  public:
    ProcArray(BaseLB::LDStats *stats);
    ~ProcArray() { }

  private:
    // vector containing the list of processors
    std::vector<ProcInfo> procs;
};

class Edge {
  friend class ObjGraph;

  public:
    Edge(int _id, int _msgs, double _bytes) : id(_id), msgs(_msgs),
      bytes(_bytes) {
    }
    ~Edge() { }

  private:
    int id;		// id of the neighbor
    int msgs;		// number of messages exchanged
    double bytes;	// total number of bytes exchanged
};


class Vertex {
  friend class ObjGraph;

  private:
    int id;		// index in the LDStats array
    double compLoad;	// computational load (walltime in LDStats)
    bool migratable;	// migratable or non-migratable
    int currPe;		// current processor assignment
    int newPe;		// new processor assignment after load balancing

    // undirected edges from this vertex to other vertices
    std::vector<Edge> edgeList;
};


class ObjGraph {
  public:
    ObjGraph(BaseLB::LDStats *stats);
    ~ObjGraph() { }

    void convertDecisions(BaseLB::LDStats *stats);
  private:
    // all vertices in the graph. Each vertex corresponds to a chare
    std::vector<Vertex> vertices;
};

#endif // _CKGRAPH_H_

/*@}*/

