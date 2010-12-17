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

  public:
    inline int getProcId() { return id; }
    inline double getTotalLoad() { return totalLoad; }
    inline double getOverhead() { return overhead; }
    inline void setTotalLoad(double _tload) { totalLoad = _tload; }
    inline bool isAvailable() { return available; }

  private:
    int id;		// CkMyPe of the processor
    double overhead;	// previously called background load (bg_walltime)
    double totalLoad;	// includes object_load + overhead
    bool available;	// if the processor is available
};

class ProcArray {
  public:
    ProcArray(BaseLB::LDStats *stats);
    ~ProcArray() { }
    inline double getAverageLoad() { return avgLoad; }
    void resetTotalLoad();

    // vector containing the list of processors
    std::vector<ProcInfo> procs;

  private:
    double avgLoad;
};

class Edge {
  friend class ObjGraph;

  public:
    Edge(int _id, int _msgs, int _bytes) : id(_id), msgs(_msgs),
      bytes(_bytes) {
    }
    ~Edge() { }
    inline int getNeighborId() { return id; }
    inline int getNumMsgs() { return msgs; }
    inline int getNumBytes() { return bytes; }

  private:
    int id;		// id of the neighbor = index of the neighbor vertex
			// in the vector 'vertices'
    int msgs;		// number of messages exchanged
    int bytes;		// total number of bytes exchanged
};


class Vertex {
  friend class ObjGraph;

  public:
    inline int getVertexId() { return id; }
    inline double getVertexLoad() { return compLoad; }
    inline int getCurrentPe() { return currPe; }
    inline int getNewPe() { return newPe; }
    inline void setNewPe(int _newpe) { newPe = _newpe; }
    inline bool isMigratable() { return migratable; }

    // undirected edges from this vertex to other vertices
    std::vector<Edge> sendToList;

  private:
    int id;		// index in the LDStats array
    double compLoad;	// computational load (walltime in LDStats)
    bool migratable;	// migratable or non-migratable
    int currPe;		// current processor assignment
    int newPe;		// new processor assignment after load balancing
};


class ObjGraph {
  public:
    ObjGraph(BaseLB::LDStats *stats);
    ~ObjGraph() { }
    void convertDecisions(BaseLB::LDStats *stats);

    // all vertices in the graph. Each vertex corresponds to a chare
    std::vector<Vertex> vertices;
};

#endif // _CKGRAPH_H_

/*@}*/

