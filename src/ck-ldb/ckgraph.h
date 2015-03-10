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
    ProcInfo() {}
    ProcInfo(int i, double ov, double tl, double sp, bool avail): id(i), _overhead(ov), _totalLoad(tl), _pe_speed(sp), available(avail) {}
    inline int getProcId() { return id; }
    inline void setProcId(int _id) { id = _id; }
    inline double getTotalLoad() const { return _totalLoad; }
//    inline void setTotalLoad(double load) { totalLoad = load; }
//    inline double getOverhead() { return overhead; }
//    inline void setOverhead(double oh) { overhead = oh; }
    inline double &overhead() { return _overhead; }
    inline double &totalLoad() { return _totalLoad; }
    inline double &pe_speed() { return _pe_speed; }
    inline bool isAvailable() { return available; }

  protected:
    int id;		// CkMyPe of the processor
    double _overhead;	// previously called background load (bg_walltime)
    double _totalLoad;	// includes object_load + overhead
    double _pe_speed;	// CPU speed
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

  protected:
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
    inline void setNumBytes(int _bytes) { bytes = _bytes; }

  private:
    int id;		// id of the neighbor = index of the neighbor vertex
			// in the vector 'vertices'
    int msgs;		// number of messages exchanged
    int bytes;		// total number of bytes exchanged
};

class McastSrc {
  friend class ObjGraph;

  public:
    McastSrc(int _numDest, int _msgs, int _bytes) : numDest(_numDest), msgs(_msgs),
    bytes(_bytes) {
    }

    ~McastSrc() { }

    inline int getNumMsgs() { return msgs; }
    inline int getNumBytes() { return bytes; }
    inline void setNumBytes(int _bytes) { bytes = _bytes; }

    std::vector<int> destList;

  private:
    int numDest; //number of destination for this multicast
    int msgs; // number of messages exchanged
    int bytes; // total number of bytes exchanged
};

class McastDest {
  friend class ObjGraph;

  public:
    McastDest(int _src, int _offset, int _msgs, int _bytes) : src(_src),
    offset(_offset), msgs(_msgs), bytes(_bytes) {
    }

    ~McastDest() { }

    inline int getSrc() { return src; }
    inline int getOffset() { return offset; }
    inline int getNumMsgs() { return msgs; }
    inline int getNumBytes() { return bytes; }
    inline void setNumBytes(int _bytes) { bytes = _bytes; }

  private:
    int src; // src of multicast being received
    int offset; //multicast list which this message belongs to
    int msgs; // number of messages exchanged
    int bytes; // total number of bytes exchanged
};

class Vertex {
  friend class ObjGraph;

  public:
    Vertex() {}
    Vertex(int i, double cl, bool mig, int curpe, int newpe=-1, size_t pupsize=0):
        id(i), compLoad(cl), migratable(mig), currPe(curpe), newPe(newpe),
        pupSize(pupsize)  {}
    inline int getVertexId() { return id; }
    inline double getVertexLoad() const { return compLoad; }
    inline int getCurrentPe() { return currPe; }
    inline int getNewPe() { return newPe; }
    inline void setNewPe(int _newpe) { newPe = _newpe; }
    inline bool isMigratable() { return migratable; }

    // list of vertices this vertex sends messages to and receives from
    std::vector<Edge> sendToList;
    std::vector<Edge> recvFromList;
    std::vector<McastSrc> mcastToList;
    std::vector<McastDest> mcastFromList;
		double getCompLoad() {return compLoad;}
		void setCompLoad(double s) {compLoad = s;}
    int getCurrPe() {return currPe;}
    void setCurrPe(int s) {currPe = s;}


  private:
    int id;		// index in the LDStats array
    double compLoad;	// computational load (walltime in LDStats)
    bool migratable;	// migratable or non-migratable
    int currPe;		// current processor assignment
    int newPe;		// new processor assignment after load balancing
    size_t pupSize;
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

