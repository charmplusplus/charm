#ifndef _OBJGRAPH_H_
#define _OBJGRAPH_H_

#include "lbdb.h"
#include "CentralLB.h"

class ObjGraph {
public:
  class Edge;
  class Node;
  
  ObjGraph(int count, CentralLB::LDStats* stats);
  ~ObjGraph();
  int NodeCount() { return n_objs; };
  Node* Start() { return nodelist; };

  class Edge {
  friend class ObjGraph;
  public:
    int proc;
    int index;
    Edge* next_from() { return nxt_out; };
    Edge* next_to() { return nxt_in; };
  private:
    Edge* nxt_out;
    Edge* nxt_in;
  };

  class Node {
  friend class ObjGraph;
  public:
    int proc;
    int index;
    int n_out;
    int n_in;
    Node* next() { return nxt; };
    Edge* edges_from() { return outEdge; };
    Edge* edges_to() { return inEdge; };

  private:
    Edge* outEdge;
    Edge* inEdge;
    Node* nxt;
    Node* nxt_hash;
  };

private:
  enum { hash_max = 256 };

  int calc_hashval(LDOMid, LDObjid);
  Node* find_node(LDOMid, LDObjid);

  CentralLB::LDStats* stats;
  Node* nodelist;
  int n_objs;
  Node* node_table[hash_max];
};

#endif
