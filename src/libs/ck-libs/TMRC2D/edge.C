#include "tri.h"
#include "edge.h"

void edge::update(elemRef oldval, elemRef newval)
{
  CkAssert((elements[0] == oldval) || (elements[1] == oldval) || 
	   (elements[0] == newval) || (elements[1] == newval));
  if ((elements[0] == oldval) && !(elements[1] == newval))  
    elements[0] = newval;
  else if ((elements[1] == oldval) && !(elements[0] == newval))
    elements[1] = newval;
}

void edge::checkPending(elemRef e) 
{
  elemRef nullRef;

  if (pending && (waitingFor == e) && !(e == nullRef)) {
    refineMsg *rm = new refineMsg;
    rm->idx = e.idx;
    rm->area = e.getArea();
    mesh[e.cid].refineElement(rm);
  }
}

void edge::checkPending(elemRef e, elemRef ne) 
{
  elemRef nullRef;

  if (pending && (waitingFor == e) && !(e == nullRef)) {
    refineMsg *rm = new refineMsg;
    waitingFor = ne;
    rm->idx = ne.idx;
    rm->area = ne.getArea();
    mesh[ne.cid].refineElement(rm);
  }
}

int edge::split(nodeRef *m, edgeRef *e_prime, nodeRef othernode, elemRef eRef)
{
  // element eRef has asked this edge to split and give back a new nodeRef
  // and new edgeRef on othernode; return value is 1 if successful, 0 if
  // e_prime is NOT incident on othernode
  if (pending && (waitingFor == eRef)) { // already split; waiting for eRef
    *m = *newNodeRef;
    *e_prime = *newEdgeRef;
    update(incidentNode, *newNodeRef);
    pending = 0;
    newEdgeRef->unsetPending();
    waitingFor.reset();
    newNodeRef->reset();
    newEdgeRef->reset();
    if (othernode == incidentNode) return 1;
    else return 0;
  }
  else if (pending) { // can't split a second time yet; waiting for nbr elem
    return -1;
  }
  else { // Need to do the split
    node newNode;
    elemRef nbr, nullRef;
    
    CkAssert((nodes[0] == othernode) || (nodes[1] == othernode));
    nbr = getNot(eRef);
    midpoint(newNode);
    newNodeRef = C->addNode(newNode);
    if ((elements[0] == nullRef) || (elements[1] == nullRef))
      C->theNodes[newNodeRef->idx].setBorder();
    newEdgeRef = C->addEdge(othernode, *newNodeRef);
    incidentNode = othernode;
    *m = *newNodeRef;
    *e_prime = *newEdgeRef;
    if (!(nbr == nullRef)) { // refine the nbr
      pending = 1;
      newEdgeRef->setPending();
      waitingFor = nbr;
      double nbrArea = nbr.getArea();
      refineMsg *rm = new refineMsg;
      rm->idx = nbr.idx;
      rm->area = nbrArea;
      mesh[nbr.cid].refineElement(rm);
    }
    else {
      update(othernode, *newNodeRef);
      waitingFor.reset();
      newNodeRef->reset();
      newEdgeRef->reset();
    }
    return 1;
  }
}

void edge::sanityCheck(chunk *c,edgeRef shouldRef) 
{
  //CkAssert(myRef == shouldRef);
  int nonNullElements=0;
  for (int i=0;i<2;i++) {
    CkAssert(nodes[i].idx < C->numNodes);
    nodes[i].sanityCheck();
    if (!elements[i].isNull()) {
      elements[i].sanityCheck();
      nonNullElements++;
    }
  }
  CkAssert(nonNullElements > 0);
}
