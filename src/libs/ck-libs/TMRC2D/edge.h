#ifndef EDGE_H
#define EDGE_H

#include "charm++.h"
#include "node.h"
#include "ref.h"
#include "refine.decl.h"

class edge {
 public:
  int pending, newNodeIdx;
  elemRef waitingFor;
  node newNode, incidentNode, fixNode;
  edgeRef newEdgeRef; // half of this edge: from newNode to incidentNode
  chunk *C;
  edgeRef myRef;
  elemRef elements[2];  // the elements on either side of the edge
  int present;  // indicates this is an edge present in the mesh
  edge() { unsetPending(); }
  edge(int idx, int cid, chunk *myChk) { 
    unsetPending(); myRef.set(cid, idx); C = myChk; 
  }
  edge(elemRef e1, elemRef e2, int p) {
    elements[0] = e1;  elements[1] = e2;  pending = p; 
  }
  edge(elemRef e1, elemRef e2) {
    elements[0] = e1;  elements[1] = e2;  unsetPending();
  }
  edge(const edge& e) {
    for (int i=0; i<2; i++)  elements[i] = e.elements[i];
    pending = e.pending;
    newNodeIdx = e.newNodeIdx;
    C = e.C;
    waitingFor = e.waitingFor;
    newNode = e.newNode;
    incidentNode = e.incidentNode;
    fixNode = e.fixNode;
    newEdgeRef = e.newEdgeRef;
    myRef = e.myRef;
  }
  void set(int idx, int cid, chunk *myChk)  { 
    unsetPending(); myRef.set(cid, idx); C = myChk; 
  }
  void set(elemRef e1, elemRef e2) { elements[0] = e1;  elements[1] = e2; }
  void set(elemRef *e) { elements[0] = e[0]; elements[1] = e[1]; }
  void reset();
  edge& operator=(const edge& e) { 
    for (int i=0; i<2; i++)  elements[i] = e.elements[i];
    pending = e.pending;
    newNodeIdx = e.newNodeIdx;
    C = e.C;
    waitingFor = e.waitingFor;
    newNode = e.newNode;
    incidentNode = e.incidentNode;
    fixNode = e.fixNode;
    newEdgeRef = e.newEdgeRef;
    myRef = e.myRef;
    return *this; 
  }
  int isPresent() { return present; }
  void update(elemRef oldval, elemRef newval) {
    CkAssert((elements[0] == oldval) || (elements[1] == oldval));
    if (elements[0] == oldval)  elements[0] = newval;
    else /* (elements[1] == oldval) */ elements[1] = newval;
  }
  elemRef& getElement(int idx) {
    CkAssert((idx==0) || (idx==1));
    return elements[idx];
  }
  elemRef& getNot(elemRef er) {
    CkAssert((elements[0] == er) || (elements[1] == er));
    if (elements[0] == er) return elements[1];
    else return elements[0];
  }
  void setPending() { pending = 1; }
  void unsetPending() { pending = 0; }
  void checkPending(elemRef e);
  void checkPending(elemRef e, elemRef ne);
  int split(int *m, edgeRef *e_prime, node iNode, node fNode,
	    elemRef requester, int *local, int *first, int *nullNbr);
  int collapse(node *m, elemRef requester, node kNode, node dNode);
  void sanityCheck(chunk *c, edgeRef shouldRef);
};

#endif
