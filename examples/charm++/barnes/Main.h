#ifndef __MAIN_H__
#define __MAIN_H__

#include "Parameters.h"
#include "defines.h"
#include "OrientedBox.h"

#include "barnes.decl.h"

class Main : public CBase_Main {
  Parameters params;
  int numQuiescenceRecvd;

  void getNumParticles();
  void setParameters(CkArgMsg *m);
  void usage();

  public:
  Main(CkArgMsg *msg);
  void commence();
  void niceExit();

  void quiescence();
  void quiescenceExit();
};

// Block cyclic placement for the tree pieces.
//
// -p is a budget, not a count: the decomposition uses whatever it needs, 652
// of 2048 in a 500K run. That breaks both stock maps. A block map hands the
// used prefix to the first PEs and leaves the rest idle -- measured as two of
// four PEs doing zero walk work. A round robin map balances whatever prefix is
// used, but tree piece indices follow SFC key order, so it scatters
// neighbouring regions across PEs and almost everything becomes remote: 4557
// remote round trips per iteration against 1118 for a block map that happened
// to be sized right.
//
// Chunks of `chunk` consecutive indices, dealt round robin, gets both: the
// prefix spreads over every PE whatever its length, and each PE still holds
// runs of neighbouring key ranges.
class BlockCyclicMap : public CkArrayMap {
  int chunk;
 public:
  BlockCyclicMap(int c) : chunk(c > 0 ? c : 1) {}
  BlockCyclicMap(CkMigrateMessage *m) : CkArrayMap(m), chunk(1) {}
  // Two levels, because the cost of a hop is not uniform. Chunks are dealt
  // round robin across *nodes* so a node holds runs of neighbouring key
  // ranges, and a node's chunks are then dealt across its own PEs, where
  // sharing is cheap. Dealing straight across PEs, as a flat block cyclic map
  // does, scatters neighbouring regions over the network instead.
  int procNum(int, const CkArrayIndex &idx){
    const int i = idx.data()[0];
    const int c = i / chunk;
    const int numNodes = CkNumNodes();
    const int node = c % numNodes;
    const int nth = c / numNodes;          // which of this node's chunks
    const int sz = CkNodeSize(node);
    return CkNodeFirst(node) + (sz > 0 ? (nth % sz) : 0);
  }
};

#endif
