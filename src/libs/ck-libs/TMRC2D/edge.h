#ifndef EDGE_H
#define EDGE_H

#include "charm++.h"
#include "node.h"
#include "ref.h"
#include "refine.decl.h"

class edge {
 public:
  int pending;
  elemRef waitingFor;
  nodeRef *newNodeRef, incidentNode;
  edgeRef *newEdgeRef;
  chunk *C;
  nodeRef nodes[2];     // the nodes that define the edge
  elemRef elements[2];  // the elements on either side of the edge
  edge() { pending = 0; }
  edge(chunk *myChk) { pending = 0; set(myChk); }
  edge(nodeRef n1, nodeRef n2, elemRef e1, elemRef e2, int p) {
    nodes[0] = n1;  nodes[1] = n2;  elements[0] = e1;  elements[1] = e2;
    pending = p; 
  }
  edge(nodeRef n1, nodeRef n2, elemRef e1, elemRef e2) {
    nodes[0] = n1;  nodes[1] = n2;  elements[0] = e1;  elements[1] = e2;
    pending = 0; 
  }
  edge(const edge& e) {
    for (int i=0; i<2; i++) {
      nodes[i] = e.nodes[i];
      elements[i] = e.elements[i];
    }
    pending = e.pending;
    C = e.C;
    waitingFor = e.waitingFor;
    newNodeRef = e.newNodeRef;
    incidentNode = e.incidentNode;
    newEdgeRef = e.newEdgeRef;
  }
  void set(chunk *myChk)  { C = myChk; }
  void set(nodeRef n1, nodeRef n2, elemRef e1, elemRef e2) {
    nodes[0] = n1;  nodes[1] = n2;  elements[0] = e1;  elements[1] = e2;
  }
  void set(nodeRef *n) {
    nodes[0] = n[0]; nodes[1] = n[1];
  }
  void set(nodeRef *n, elemRef *e) {
    nodes[0] = n[0]; nodes[1] = n[1]; elements[0] = e[0]; elements[1] = e[1];
  }
  void update(nodeRef oldval, nodeRef newval) {
    CkAssert((nodes[0] == oldval) || (nodes[1] == oldval));
    if (nodes[0] == oldval) nodes[0] = newval;
    else if (nodes[1] == oldval) nodes[1] = newval;
  }
  void updateSilent(nodeRef oldval, nodeRef newval) {
    if (nodes[0] == oldval) nodes[0] = newval;
    else if (nodes[1] == oldval) nodes[1] = newval;
  }
  void update(elemRef oldval, elemRef newval);
  nodeRef& get(int idx) { 
    if ((idx < 0) || (idx > 1)) 
      CkPrintf("ERROR: nodeRef& edge::get(int idx):\n ---> Invalid idx.\n");
    return nodes[idx];
  }
  elemRef& getElement(int idx) { 
    if ((idx < 0) || (idx > 1)) 
      CkPrintf("ERROR: elemRef& edge::get(int idx):\n ---> Invalid idx.\n");
    return elements[idx];
  }
  nodeRef& getNot(nodeRef nr) {
    if (nodes[0] == nr) return nodes[1];
    else if (nodes[1] == nr) return nodes[0];
    else {
      CkPrintf("WARNING: nodeRef *edge::getNot(nodeRef nr):\n ---> No match for nr.\n");
      return nodes[0];
    }
  }
  elemRef& getNot(elemRef er){
    if (elements[0] == er) return elements[1];
    else if (elements[1] == er) return elements[0];
    else {
      CkPrintf("WARNING: elemRef *edge::getNot(elemRef er):\n ---> No match for er.\n");
      return elements[0];
    }
  }
  double length() {             // accesses node coords to compute edge length
    node n[2];
    double result;
    
    n[0] = nodes[0].get();
    n[1] = nodes[1].get();
    result = n[0].distance(n[1]); // find distance between two nodes
    return result;
  }

  void midpoint(node& result) {  // compute edge midpoint and place in result
    node n[2];
    
    n[0] = nodes[0].get();
    n[1] = nodes[1].get();
    n[0].midpoint(n[1], result); // find midpoint between two nodes
  }
  void setPending() { pending = 1; }
  void unsetPending() { pending = 0; }
  int isPending() { return pending; }
  void checkPending(elemRef e);
  void checkPending(elemRef e, elemRef ne);
  edge& operator=(const edge& e) { 
    for (int i=0; i<2; i++) { 
      nodes[i] = e.nodes[i]; elements[i] = e.elements[i];
    }
    pending = e.pending;
    C = e.C;
    waitingFor = e.waitingFor;
    newNodeRef = e.newNodeRef;
    incidentNode = e.incidentNode;
    newEdgeRef = e.newEdgeRef;
    return *this; 
  }
  int split(nodeRef *m, edgeRef *e_prime, nodeRef othernode, elemRef eRef);
  void sanityCheck(chunk *c,edgeRef shouldRef);
};

#endif
