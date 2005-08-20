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
  elemRef waitingFor;
  node newNode;
  int incidentNode, fixNode;
  int boundary;
  edgeRef newEdgeRef; // half of this edge: from newNode to incidentNode
  chunk *C;
  edgeRef myRef;
  elemRef elements[2];  // the elements on either side of the edge
  int nodes[2];  // the nodes on either end of the edge on the edge's chunk
  int present;  // indicates this is an edge present in the mesh
  edge() { unsetPending(); present = 0; }
  edge(int idx, int cid, chunk *myChk) { 
    unsetPending(); myRef.set(cid, idx); C = myChk; present = 1;
    nodes[0] = nodes[1] = -1;
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
    present = e.present;
    length = e.length;
    boundary = e.boundary;
    nodes[0] = e.nodes[0];
    nodes[1] = e.nodes[1];
  }
  void set(int idx, int cid, chunk *myChk)  { 
    unsetPending(); myRef.set(cid, idx); C = myChk; present = 1;
  }
  void set(elemRef e1, elemRef e2) { elements[0] = e1;  elements[1] = e2; }
  void set(elemRef *e) { elements[0] = e[0]; elements[1] = e[1]; }
  void setNodes(int n1, int n2) { nodes[0] = n1; nodes[1] = n2; CkAssert(n1!=n2);}
  void setBoundary(int b) { boundary = b;}
  int getBoundary() { return boundary;}
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
    present = e.present;
    boundary = e.boundary;
    nodes[0] = e.nodes[0];
    nodes[1] = e.nodes[1];
    return *this; 
  }
  int isPresent() { return present; }
  void update(elemRef oldval, elemRef newval) {
    CkAssert((elements[0] == oldval) || (elements[1] == oldval));
    if (elements[0] == oldval)  elements[0] = newval;
    else /* (elements[1] == oldval) */ elements[1] = newval;
  }
  void updateNode(int oldval, int newval) {
    CkAssert((nodes[0] == oldval) || (nodes[1] == oldval));
    if (nodes[0] == oldval)  nodes[0] = newval;
    else /* (nodes[1] == oldval) */ nodes[1] = newval;
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
  int split(int *m, edgeRef *e_prime, int oIdx, int fIdx,
	    elemRef requester, int *local, int *first, int *nullNbr);
  void collapse(elemRef requester, int kIdx, int dIdx, elemRef kNbr,
		elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node newN, 
		double frac);
  int flipPrevent(elemRef requester, int kIdx, int dIdx, elemRef kNbr,
	       elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node newN);
  int existsOn(FEM_Comm_Rec *cl, int chunkID) {
    int count = cl->getShared();
    for (int i=0; i<count; i++) {
      if (chunkID == cl->getChk(i)) return i;
    }
    return -1;
  }
  void translateSharedNodeIDs(int *kIdx, int *dIdx, elemRef req);
  void unlockCloudRemoveEdge(int dIdxlShared, int kIdxlShared, 
			     FEM_Comm_Rec *dNodeRec, FEM_Comm_Rec *kNodeRec);
  void localCollapse(int kIdx, int dIdx, elemRef *req, node *newNode, 
		     double frac, elemRef *keepNbr, elemRef *delNbr, 
		     edgeRef *kEdge, edgeRef *dEdge, int local, int first);
  void updateCloud(int kIdx, int dIdx, node newNode, int *dIdxl, int *kIdxl,
		   FEM_Comm_Rec **dNodeRec, FEM_Comm_Rec **kNodeRec);
  int buildLockingCloud(int kIdx, int dIdx, elemRef *req, elemRef *nbr);
  void sanityCheck(chunk *c, edgeRef shouldRef);
  void sanityCheck(int node1, int node2, int eIdx);
};

#endif
