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

int edge::collapse(elemRef requester, node kNode, node dNode)
{
  // element requester has asked this edge to collapse and give back new node
  // coordinates resulting node; return value is 1 if successful, 0 if
  // dNode is the node that is kept
  intMsg *im;
  elemRef nbr = getNot(requester), nullRef;
  nullRef.reset();
  if (pending && (waitingFor == requester)) { // collapsed; awaiting requester
    CkPrintf("TMRC2D: edge::collapse: ** PART 2! ** On edge=%d on chunk=%d, requester=(%d,%d) with nbr=(%d,%d)\n", myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);
    CkPrintf("TMRC2D: removing edge %d on %d\n", myRef.idx, myRef.cid);
    C->removeEdge(myRef.idx);
    if (dNode == incidentNode) { // incidence as planned
      CkPrintf("TMRC2D: moving node %f,%f to %f,%f\n", kNode.X(), kNode.Y(), 
	       newNode.X(), newNode.Y());
      im = mesh[requester.cid].nodeUpdate(requester.idx,kNode,myRef,nbr,newNode);
      if (im->anInt == -1)
	im = mesh[nbr.cid].nodeUpdate(nbr.idx,kNode,myRef,requester,newNode);
      CkPrintf("TMRC2D: deleting node %f,%f\n", dNode.X(), dNode.Y());
      im = mesh[requester.cid].nodeDelete(requester.idx,dNode,myRef,nbr);
      if (im->anInt == -1)
	im = mesh[nbr.cid].nodeDelete(nbr.idx,dNode,myRef,requester);
      return 1; 
    }
    else { // incidence is on kNode
      CkPrintf("TMRC2D: moving node %f,%f to %f,%f\n", dNode.X(), dNode.Y(), 
	       newNode.X(), newNode.Y());
      im = mesh[requester.cid].nodeUpdate(requester.idx,dNode,myRef,nbr,newNode);
      if (im->anInt == -1)
	im = mesh[nbr.cid].nodeUpdate(nbr.idx,dNode,myRef,requester,newNode);
      CkPrintf("TMRC2D: deleting node %f,%f\n", kNode.X(), kNode.Y());
      im = mesh[requester.cid].nodeDelete(requester.idx,kNode,myRef,nbr);
      if (im->anInt == -1)
	im = mesh[nbr.cid].nodeDelete(nbr.idx,kNode,myRef,requester);
      return 0; 
    }
  }
  else if (pending) { // can't collapse a second time yet; waiting for nbr elem
  CkPrintf("TMRC2D: edge::collapse: ** Pending on (%d,%d)! ** On edge=%d on chunk=%d, requester=%d on chunk=%d\n", waitingFor.cid, waitingFor.idx, myRef.idx, myRef.cid, requester.idx, requester.cid);
    return -1;
  }
  else { // Need to do the collapse
    // need to lock adjacent nodes
    CkPrintf("TMRC2D: edge::collapse: ** PART 1! ** On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d)\n", myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);
    length = kNode.distance(dNode);
    intMsg *im = mesh[requester.cid].nodeLockup(requester.idx, kNode, myRef, myRef,
						nbr, length);
    if (im->anInt == -1) 
      im = mesh[nbr.cid].nodeLockup(nbr.idx, kNode, myRef, myRef, requester, 
				    length);
    if (!im->anInt) return -1;
    else {
      im = mesh[requester.cid].nodeLockup(requester.idx, dNode, myRef, myRef,
					  nbr, length);
      if (im->anInt == -1) 
	im = mesh[nbr.cid].nodeLockup(nbr.idx, kNode, myRef, myRef, requester, 
				      length);
      if (!im->anInt) return -1;
    }
    // both nodes locked
    CkPrintf("TMRC2D: edge::collapse: LOCKS obtained... On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d)\n", myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);

    setPending();
    kNode.midpoint(dNode, newNode);
    incidentNode = dNode;
    fixNode = kNode;
    if (nbr.cid != -1) {
      waitingFor = nbr;
      double nbrArea = nbr.getArea();
      mesh[nbr.cid].coarsenElement(nbr.idx, nbrArea);
    }
    else {
      CkPrintf("TMRC2D: removing edge %d on %d\n", myRef.idx, myRef.cid);
      C->removeEdge(myRef.idx);
      CkPrintf("TMRC2D: moving node %f,%f to %f,%f\n", kNode.X(), kNode.Y(), 
	       newNode.X(), newNode.Y());
      im = mesh[requester.cid].nodeUpdate(requester.idx,kNode,myRef,nbr,newNode);
      if (im->anInt == -1)
	im = mesh[nbr.cid].nodeUpdate(nbr.idx,kNode,myRef,requester,newNode);
      CkPrintf("TMRC2D: deleting node %f,%f\n", dNode.X(), dNode.Y());
      im = mesh[requester.cid].nodeDelete(requester.idx,dNode,myRef,nbr);
      if (im->anInt == -1)
	im = mesh[nbr.cid].nodeDelete(nbr.idx,dNode,myRef,requester);
    }
    return 1;
  }
}

int edge::nodeLockup(node n, edgeRef start, elemRef from, elemRef end, 
		     double l)
{
  elemRef next = getNot(from);
  if (next.cid == -1) return -1;
  intMsg *im = mesh[next.cid].nodeLockup(next.idx, n, myRef, start, end, l);
  return im->anInt;
}

int edge::nodeUpdate(node n, elemRef from, elemRef end, node newNode)
{
  elemRef next = getNot(from);
  if (next.cid == -1) return -1;
  intMsg *im = mesh[next.cid].nodeUpdate(next.idx, n, myRef, end, newNode);
  return im->anInt;
}

int edge::nodeDelete(node n, elemRef from, elemRef end)
{
  elemRef next = getNot(from);
  if (next.cid == -1) return -1;
  intMsg *im = mesh[next.cid].nodeDelete(next.idx, n, myRef, end);
  return im->anInt;
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
