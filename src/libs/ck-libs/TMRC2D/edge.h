#ifndef EDGE_H
#define EDGE_H

#include "charm++.h"
#include "node.h"
#include "ref.h"
#include "refine.decl.h"

class edge {
 public:
  int pending, newNodeIdx;
  double length;
  elemRef waitingFor, delNbr, keepNbr;
  node newNode, incidentNode, fixNode, opnode;
  edgeRef newEdgeRef; // half of this edge: from newNode to incidentNode
  chunk *C;
  edgeRef myRef, keepEdge, delEdge;
  elemRef elements[2];  // the elements on either side of the edge
  int present;  // indicates this is an edge present in the mesh
  edge() { unsetPending(); present = 0; }
  edge(int idx, int cid, chunk *myChk) { 
    unsetPending(); myRef.set(cid, idx); C = myChk; present = 1;
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
    keepNbr = e.keepNbr;
    delNbr = e.delNbr;
    keepEdge = e.keepEdge;
    delEdge = e.delEdge;
    newNode = e.newNode;
    opnode = e.opnode;
    incidentNode = e.incidentNode;
    fixNode = e.fixNode;
    newEdgeRef = e.newEdgeRef;
    myRef = e.myRef;
    present = e.present;
    length = e.length;
  }
  void set(int idx, int cid, chunk *myChk)  { 
    unsetPending(); myRef.set(cid, idx); C = myChk; present = 1;
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
    keepNbr = e.keepNbr;
    delNbr = e.delNbr;
    keepEdge = e.keepEdge;
    delEdge = e.delEdge;
    newNode = e.newNode;
    opnode = e.opnode;
    incidentNode = e.incidentNode;
    fixNode = e.fixNode;
    newEdgeRef = e.newEdgeRef;
    myRef = e.myRef;
    present = e.present;
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
  int isPending(elemRef e);
  void checkPending(elemRef e);
  void checkPending(elemRef e, elemRef ne);
  int split(int *m, edgeRef *e_prime, node iNode, node fNode,
	    elemRef requester, int *local, int *first, int *nullNbr);
  int collapse(elemRef requester, node kNode, node dNode, elemRef kNbr,
	       elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node oNode,
	       int *local, int *first);
  void sanityCheck(chunk *c, edgeRef shouldRef);
  int nodeLockup(node n, edgeRef start, elemRef from, elemRef end, double l);
  int nodeUpdate(node n, elemRef from, elemRef end, node newNode);
  int nodeDelete(node n, elemRef from, elemRef end, node ndReplace);
};

#endif
