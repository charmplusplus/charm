/**
 *                 !!!!!!!!!!!!!!!!! DEFUNCT !!!!!!!!!!!!!!!!!!!
 *
 *  This file is not compiled anymore and its uses should be replaced by the
 *  class of the same name (ObjGraph) in ckgraph.h
 *
 *                 !!!!!!!!!!!!!!!!! DEFUNCT !!!!!!!!!!!!!!!!!!!
 */

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _OBJGRAPH_H_
#define _OBJGRAPH_H_

#include "lbdb.h"
#include "CentralLB.h"

class ObjGraph {
public:
  class Edge {
    friend class ObjGraph;
  public:
    int edge_index;
    int index;
    int from_node;
    int to_node;
    Edge* next_from() { return nxt_out; };
    Edge* next_to() { return nxt_in; };
  private:
    Edge* nxt_out;
    Edge* nxt_in;
  };

  class Node {
    friend class ObjGraph;
  public:
    int node_index;
    int proc;
    int index;
    int n_out;
    int n_in;
    Edge* edges_from() { return outEdge; };
    Edge* edges_to() { return inEdge; };

  private:
    Edge* outEdge;
    Edge* inEdge;
    Node* nxt_hash;
  };

  ObjGraph(int count, BaseLB::LDStats* stats);
  ~ObjGraph();

  int NodeCount() { return n_objs; };
  int EdgeCount() { return n_edges; };
  Node* Start() { return nodelist; };
  Node GraphNode(int i) { return nodelist[i]; };

  double LoadOf(int i) {
    const Node n = GraphNode(i);
    const int index = n.index;
    return stats->objData[index].wallTime;
  };

  double EdgeWeight(Edge* e);

private:
  enum { hash_max = 256 };

  int calc_hashval(LDOMid, LDObjid);
  Node* find_node(const LDObjKey &);

  Edge* edgelist;
  Node* node_table[hash_max];

  int n_objs;
  int n_edges;
  Node* nodelist;
  BaseLB::LDStats* stats;
};

#endif

/*@}*/
