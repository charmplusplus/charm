#include "tri.h"
#include "edge.h"

void edge::reset() 
{ 
  newNodeIdx = -1;
  if (!(newEdgeRef == nullRef))
    C->theEdges[newEdgeRef.idx].reset();
  unsetPending(); waitingFor.reset(); newEdgeRef.reset(); 
  newNode.reset(); incidentNode.reset(); fixNode.reset(); 
}

void edge::checkPending(elemRef e) 
{
  elemRef nullRef;
  if (pending && (waitingFor == e) && !(e == nullRef))
    mesh[e.cid].refineElement(e.idx, e.getArea());
}

void edge::checkPending(elemRef e, elemRef ne) 
{
  elemRef nullRef;
  if (pending && (waitingFor == e) && !(e == nullRef)) {
    waitingFor = ne;
    mesh[ne.cid].refineElement(ne.idx, ne.getArea());
  }
}

int edge::split(int *m, edgeRef *e_prime, node iNode, node fNode,
		elemRef requester, int *local, int *first, int *nullNbr)
{
  // element requester has asked this edge to split and give back a new node
  // and new edgeRef on iNode; return value is 1 if successful, 0 if
  // e_prime is NOT incident on iNode, -1 if another split is pending
  // on this edge; local is set if this edge is not a boundary between chunks
  // and first is set if this was the first split request on this edge
  intMsg *im;
  elemRef nbr = getNot(requester), nullRef;
  nullRef.reset();
  if (pending && (waitingFor == requester)) { 
    // already split; waiting for requester
  CkPrintf("TMRC2D: edge::split: ** PART 2! ** On edge=%d on chunk=%d, requester=(%d,%d) with nbr=(%d,%d)\n", myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);
    *m = newNodeIdx;
    *e_prime = newEdgeRef;
    *first = 0;
    *local = 1;
    if (nbr.cid != requester.cid) { 
      *local = 0;
      im = mesh[requester.cid].addNode(newNode);
      *m = im->anInt;
      CkFreeMsg(im);
      CkPrintf("TMRC2D: New node (%f,%f) added at index %d on chunk %d\n",
	       newNode.X(), newNode.Y(), *m, myRef.cid);
    }
    if (iNode == incidentNode) return 1; // incidence as planned
    else return 0; // incidence is on fNode
  }
  else if (pending) { // can't split a second time yet; waiting for nbr elem
  CkPrintf("TMRC2D: edge::split: ** Pending on (%d,%d)! ** On edge=%d on chunk=%d, requester=%d on chunk=%d\n", waitingFor.cid, waitingFor.idx, myRef.idx, myRef.cid, requester.idx, requester.cid);
    return -1;
  }
  else { // Need to do the split
  CkPrintf("TMRC2D: edge::split: ** PART 1! ** On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d)\n", myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);
    setPending();
    iNode.midpoint(fNode, newNode);
    im = mesh[requester.cid].addNode(newNode);
    newNodeIdx = im->anInt;
    CkFreeMsg(im);
    CkPrintf("TMRC2D: New node (%f,%f) added at index %d on chunk %d\n", 
	     newNode.X(), newNode.Y(), newNodeIdx, myRef.cid);
    newEdgeRef = C->addEdge();
    CkPrintf("TMRC2D: New edge (%d,%d) added between nodes (%f,%f) and newNode\n", 
	     newEdgeRef.cid, newEdgeRef.idx, iNode.X(), iNode.Y());
    incidentNode = iNode;
    fixNode = fNode;
    *m = newNodeIdx;
    *e_prime = newEdgeRef;
    *first = 1;
    if ((nbr.cid == requester.cid) || (nbr.cid == -1)) *local = 1;
    else *local = 0;
    C->theEdges[newEdgeRef.idx].setPending();
    *nullNbr = 0;
    if (nbr == nullRef) *nullNbr = 1;
    if (*nullNbr) CkPrintf("TMRC2D: on edge, nbr is null\n");
    if (nbr.cid != -1) {
      waitingFor = nbr;
      double nbrArea = nbr.getArea();
      mesh[nbr.cid].refineElement(nbr.idx, nbrArea);
    }
    return 1;
  }
}

void edge::sanityCheck(chunk *c, edgeRef shouldRef) 
{
  int nonNullElements=0;
  for (int i=0;i<2;i++) {
    if (!elements[i].isNull()) {
      elements[i].sanityCheck();
      nonNullElements++;
    }
  }
  CkAssert(nonNullElements > 0);
}
