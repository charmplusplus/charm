// Triangular Mesh Refinement Framework - 2D (TMR)
// Created by: Terry L. Wilmarth

#include <stdlib.h>
#include <stdio.h>
#include "tri.h"

//readonlys
CProxy_chunk mesh;
CtvDeclare(chunk *, _refineChunk);

void refineChunkInit(void) {
  CtvInitialize(chunk *, _refineChunk);
}

static elemRef nullRef(-1,-1);

// edgeRef methods
void edgeRef::updateElement(chunk *C, elemRef oldval, elemRef newval)
{
  if (cid == C->cid) // the edge is local; update its elemRef on this chunk
    C->theEdges[idx].updateElement(oldval, newval);
  else // the edge is remote; send elemRef update to remote chunk
    mesh[cid].updateElement(idx, oldval, newval);
}

int edgeRef::lock(chunk *C)
{
  // fails if edge is already locked
  if (cid == C->cid) { // the edge is local; lock it on this chunk
    if (C->theEdges[idx].locked())
      return 0;
    else { 
      C->theEdges[idx].lock();
      return 1;
    }
  }
  else { // the edge is remote; tell remote chunk to lock it
    intMsg *im2;
    int result;
    im2 = mesh[cid].lock(idx);
    result = im2->anInt;
    CkFreeMsg(im2);
    return result;
  }
}

void edgeRef::unlock(chunk *C)
{
  if (cid == C->cid) // the edge is local; unlock it on this chunk
    C->theEdges[idx].unlock();
  else // the edge is remote; tell remote chunk to unlock it
    mesh[cid].unlock(idx);
}

int edgeRef::locked(chunk *C) const
{
  if (cid == C->cid) // the edge is local; query its lock on this chunk
    return C->theEdges[idx].locked();
  else { // the edge is remote; query its lock on the remote chunk
    intMsg *im2;
    int result;
    im2 = mesh[cid].locked(idx);
    result = im2->anInt;
    CkFreeMsg(im2);
    return result;
  }
}

// elemRef methods
double elemRef::getArea(chunk *C)
{
  if (cid == C->cid) // the element is local; get the area on this chunk
    return C->theElements[idx].getArea();
  else { // the element is remote; get the area from the remote chunk
    doubleMsg *dm;
    double result;
    dm = mesh[cid].getArea(idx);
    result = dm->aDouble;
    CkFreeMsg(dm);
    return result;
  }
}

int elemRef::checkIfLongEdge(chunk *C, edgeRef e)
{
  if (cid == C->cid) // the element is local; check e == longEdge on this chunk
    return C->theElements[idx].checkIfLongEdge(e);
  else { // the element is remote; check e == longEdge on remote chunk
    intMsg *im;
    int result;
    im = mesh[cid].checkElement(e, idx);
    result = im->anInt;
    CkFreeMsg(im);
    return result;
  }
}

void elemRef::updateEdges(chunk *C, edgeRef e0, edgeRef e1, 
			  edgeRef e2)
{
  if (cid == C->cid) // the element is local; updateEdges on this chunk
    C->theElements[idx].updateEdges(e0, e1, e2);
  else // the element is remote; update edges on remote chunk
    mesh[cid].updateEdges(idx, e0, e1, e2);
}

void elemRef::setDependent(chunk *C, int anIdx, int aCid)
{
  if (cid == C->cid) { // the element is local; setDependent on this chunk
    C->theElements[idx].setDependent(aCid, anIdx);
    C->setModified(); // this modifies the set of elements needing refinement
    if (!C->isRefining()) { // if the refinement loop isn't already running
      C->setRefining();
      mesh[cid].refiningElements(); // start it up now
    }
  }
  else // the element is remote; setDependent on remote chunk
    mesh[cid].setDependent(objRef(aCid, anIdx), idx);
}

void elemRef::unsetDependency(chunk *C)
{
  if (cid == C->cid) { // the element is local; unsetDependency on this chunk
    C->theElements[idx].unsetDependency();
    C->setModified(); // this modifies the set of elements needing refinement
    if (!C->isRefining()) { // if the refinement loop isn't already running
      C->setRefining();
      mesh[cid].refiningElements(); // start it up now
    }
  }
  else // the element is remote; unsetDependency on remote chunk
    mesh[cid].unsetDependency(idx);
}

int elemRef::hasDependent(chunk *C)
{
  if (cid == C->cid) // the element is local; query dependent on this chunk
    return (C->theElements[idx].hasDependent());
  else { // the element is remote; query dependent on remote chunk
    intMsg *im2;
    int result;
    im2 = mesh[cid].hasDependent(idx);
    result = im2->anInt;
    CkFreeMsg(im2);
    return result;
  }
}

void elemRef::setTargetArea(chunk *C, double ta)
{
  if (cid == C->cid) // the element is local; setTargetArea on this chunk
    C->theElements[idx].setTargetArea(ta);
  else // the element is remote; setTargetArea dependent on remote chunk
    mesh[cid].setTargetArea(idx, ta);
}

// edge methods
void edge::init(int i, chunk *cPtr) 
{ 
  C = cPtr;  myRef.init(C->cid,i);  theLock = 0;
}

void edge::init(elemRef *e, int i, chunk *cPtr) 
{
  elements[0] = e[0];  elements[1] = e[1];
  C = cPtr;  myRef.init(C->cid,i);  theLock = 0;
}

void edge::updateElement(elemRef oldval, elemRef newval)
{ // find which element matches oldval and replace it with newval
  if (elements[0] == oldval)
    elements[0] = newval;
  else if (elements[1] == oldval)
    elements[1] = newval;
  else 
    CkAbort("ERROR: edge::updateElement: no match for oldval\n");
}

// element methods
element::element()
{
  targetArea = currentArea = -1.0;
  specialRequest = pendingRequest = requestResponse = depend = 0;
  unsetDependent();
  specialRequester.init();
  newNode.init();
  otherNode.init();
  newLongEdgeRef.init();
  C = NULL;
}

void element::init()
{
  targetArea = currentArea = -1.0;
  specialRequest = pendingRequest = requestResponse = depend = 0;
  unsetDependent();
}

void element::init(int *n, int index, chunk *chk)
{
  for (int i=0; i<3; i++) {
    nodes[i] = n[i];
    edges[i].init();
  }
  C = chk;
  myRef.init(C->cid, index);
  targetArea = currentArea = -1.0;
  specialRequest = pendingRequest = requestResponse = depend = 0;
  unsetDependent();
}

void element::init(int *n, edgeRef *e, int index, chunk *chk)
{
  for (int i=0; i<3; i++) {
    nodes[i] = n[i];
    edges[i] = e[i];
  }
  C = chk;
  myRef.init(C->cid, index);
  targetArea = currentArea = -1.0;
  specialRequest = pendingRequest = requestResponse = depend = 0;
  unsetDependent();
}

double element::getArea()
{ 
  calculateArea();
  return currentArea;
}

void element::calculateArea()
{ // calulate area of triangle using Heron's formula:
  // Let a, b, c be the lengths of the three sides.
  // Area=SQRT(s(s-a)(s-b)(s-c)), where s=(a+b+c)/2 or perimeter/2.

  node n[3];
  double s, perimeter, len[3];

  for (int i=0; i<3; i++) // get node coordinates
    n[i] = C->theNodes[nodes[i]];
  // fine lengths of sides
  len[0] = n[0].distance(n[1]);
  len[1] = n[1].distance(n[2]);
  len[2] = n[2].distance(n[0]);
  
  // apply Heron's formula
  perimeter = len[0] + len[1] + len[2];
  s = perimeter / 2.0;
  // cache the result in currentArea
  currentArea = sqrt(s * (s - len[0]) * (s - len[1]) * (s - len[2]));
}

int element::findLongestEdge()
{
  int i, longEdge;
  node n[3];
  double maxlen = 0.0, len[3];

  for (i=0; i<3; i++) // get node coordinates
    n[i] = C->theNodes[nodes[i]];
  // fine lengths of sides
  len[0] = n[0].distance(n[1]);
  len[1] = n[1].distance(n[2]);
  len[2] = n[2].distance(n[0]);

  for (i=0; i<3; i++) // find max length of a side
    if (len[i] > maxlen) {
      longEdge = i;
      maxlen = len[i];
    }
  return longEdge;
}

elemRef element::getNeighbor(int e) const
{
  if (edges[e].cid == myRef.cid) { 
    // if edges[e] is local, look up neighboring elemRef on this chunk
    return C->theEdges[edges[e].idx].getNbrRef(myRef);
  }
  else { // edges[e] is not local; getNeighbor on remote chunk
    refMsg *result = mesh[edges[e].cid].getNeighbor(myRef, edges[e].idx);
    elemRef ret=*(elemRef *)&result->aRef;
    CkFreeMsg(result);
    return ret;
  }
}

int element::checkNeighbor(int longEdge)
{
  elemRef nbr = getNeighbor(longEdge);

  if (nbr.idx == -1) { // no neighbor; on border
    return -1; 
  }
  else { // neighbor exists                         
    int result;
    result = nbr.checkIfLongEdge(C, edges[longEdge]); // is longEdge shared?
    return result;
  }
}

int element::checkIfLongEdge(edgeRef e)
{
  int longEdge = findLongestEdge();
  return (edges[longEdge] == e);
}

void element::refine()
{
  int longEdge, test; 
  
  longEdge = findLongestEdge();

  if (requestResponse) {
    // this element sent a special request and has received a response
    // from its neighbor; the neighbor performed half the refinement
    splitHelp(longEdge); // now finish the other half
    return;
  }
  if (specialRequest) { // my neighbor sent me a special request to refine
    if (getArea() < targetArea) {  // no refinement necessary; accept request
      splitResponse(longEdge);  // do my half of the refine
      return;
    }
    else { // refinement necessary
      if ((myRef.idx > specialRequester.idx) || 
	  ((myRef.idx == specialRequester.idx) && (myRef.cid > specialRequester.cid))) {
	// I have precedence; accept request
	splitResponse(longEdge);
	return;
      }
      else // requester has precedence; will send it special request shortly
	specialRequest = pendingRequest = 0;
    }
  }
  if (pendingRequest) // still awaiting a response; skip me
    return;

  // test longEdge relationship
  test = checkNeighbor(longEdge);
  if (test == -1) // long edge is on border
    splitBorder(longEdge);
  else if (test == 1) { // long edge is longest of both elements
    if (!edges[longEdge].locked(C))
      splitNeighbors(longEdge);
  }
  else // long edge is not neighbor's long edge
    refineNeighbor(longEdge);
}

void element::splitBorder(int longEdge)
{ // split a triangle with longest edge on border
  int opnode, othernode, modEdge, otherEdge;
  edgeRef modEdgeRef;

  /*
                      opnode
                        /@\  
                       /   \
                      /     \
          otherEdge  /       \  modEdge
                    /         \
                   /           \
                  @_____________@ othernode
                      longEdge
  */
  

  // initializations of shortcuts to affected parts of element
  opnode = (longEdge + 2) % 3;
  othernode = longEdge;
  modEdge = opnode;
  otherEdge = (longEdge + 1) % 3;
  modEdgeRef = edges[modEdge];  

  // lock the perimeter to prevent access by other refinements
  if (edges[modEdge].lock(C)) { // if modEdge not locked already
    if (edges[otherEdge].lock(C)) { // if otherEdge not locked already
      splitBorderLocal(longEdge, opnode, othernode, modEdge); // do the split
      edges[otherEdge].unlock(C); // unlock otherEdge
    }
    modEdgeRef.unlock(C); // unlock modEdge if otherEdge locked
  }
}

void element::splitBorderLocal(int longEdge, int opnode, int othernode, 
			       int modEdge)
{ // split a triangle with longest edge on border
  // newElem nodes are numbered as in parens below
  /*
                      opnode(0)
                        /@\ 
                       / | \
                      /  |  \
          otherEdge  /   |   \  modEdge
                    /   n|    \
                   /    e|     \
                  /     w|      \
                 /      E|       \
                /       d|        \
               /        g|         \
              /         e| newElem  \
             /           |           \
            @____________@____________@ othernode(1)
              longEdge  m(2)  newLong
  */
  int fixnode = 3 - opnode - othernode;
  // find midpoint on longest edge
  node m; // the new node
  C->theNodes[nodes[othernode]].midpoint(C->theNodes[nodes[fixnode]], &m);

  // add new components to local chunk and get refs to them
  int mIdx = C->addNode(m);
  edgeRef newEdgeRef = C->addEdge();
  edgeRef newLongRef = C->addEdge();
  elemRef newElemRef;
  if (opnode == 0) {
    if (othernode == 1)
      newElemRef = C->addElement(nodes[0], nodes[1], mIdx, edges[modEdge],
				 newLongRef, newEdgeRef);
    else // othernode == 2
      newElemRef = C->addElement(nodes[0], mIdx, nodes[2], newEdgeRef, 
				 newLongRef, edges[modEdge]);
  }
  else if (opnode == 1) {
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], nodes[1], mIdx, edges[modEdge], 
				 newEdgeRef, newLongRef);
    else // othernode == 2
      newElemRef = C->addElement(mIdx, nodes[1], nodes[2], newEdgeRef, 
				 edges[modEdge], newLongRef);
  }
  else { // opnode is 2
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], mIdx, nodes[2], newLongRef, 
				 newEdgeRef, edges[modEdge]);
    else // othernode == 1
      newElemRef = C->addElement(mIdx, nodes[1], nodes[2], newLongRef, 
				 edges[modEdge], newEdgeRef);
  }

  edges[modEdge].updateElement(C, myRef, newElemRef);
  C->theEdges[newEdgeRef.idx].updateElement(nullRef, myRef);
  C->theEdges[newEdgeRef.idx].updateElement(nullRef, newElemRef);
  C->theEdges[newLongRef.idx].updateElement(nullRef, newElemRef);
  
  nodes[othernode] = mIdx;
  edges[modEdge] = newEdgeRef;
  C->theElements[newElemRef.idx].setTargetArea(targetArea);
  
  // tell the world outside about the split
  if (C->theClient) C->theClient->split(myRef.idx, longEdge, othernode, 0.5, 
					LOCAL_FIRST);

  tellDepend();  // tell dependent it can go now

  calculateArea(); // update cached area of original element
  C->theElements[newElemRef.idx].calculateArea(); // and of new element
}

void element::splitNeighbors(int longEdge)
{
  int opnode, othernode, modEdge, otherEdge;
  int nbrLongEdge, nbrOpnode, nbrOthernode, nbrModEdge, nbrOtherEdge;
  elemRef nbr = getNeighbor(longEdge);
  edgeRef modEdgeRef, nbrModEdgeRef;
  
  // initializations of shortcuts to affected parts of element
  opnode = (longEdge + 2) % 3;
  othernode = longEdge;
  modEdge = opnode;
  otherEdge = (longEdge + 1) % 3;

  if (nbr.cid == myRef.cid) { // neighbor is local -> longEdge is local
    // initializations of shortcuts to affected parts of neighbor element
    nbrLongEdge = C->theElements[nbr.idx].findLongestEdge();
    nbrOpnode = (nbrLongEdge + 2) % 3;
    nbrOthernode = (nbrOpnode+1) % 3;
    nbrModEdge = nbrOpnode;
    nbrOtherEdge = (nbrLongEdge + 1) % 3;
    if (!(C->theElements[nbr.idx].nodes[nbrOthernode] == nodes[othernode])) {
      nbrOthernode = (nbrOpnode+2) % 3;
      nbrModEdge = nbrOthernode;
      nbrOtherEdge = (nbrLongEdge + 2) % 3;
    }
    
    modEdgeRef = edges[modEdge];
    nbrModEdgeRef = C->theElements[nbr.idx].edges[nbrModEdge];
    
    // lock the perimeter
    if (edges[modEdge].lock(C)) {
      if (edges[otherEdge].lock(C)) {
	if (C->theElements[nbr.idx].edges[nbrModEdge].lock(C)) {
	  if (C->theElements[nbr.idx].edges[nbrOtherEdge].lock(C)) {
	    // perimeter locked; split both elements
	    splitNeighborsLocal(longEdge, opnode, othernode, modEdge, 
				nbrLongEdge, nbrOpnode, nbrOthernode, 
				nbrModEdge, nbr);
	    // and now for the tedious unlocking process
	    C->theElements[nbr.idx].edges[nbrOtherEdge].unlock(C);
	  }
	  nbrModEdgeRef.unlock(C);
	}
	edges[otherEdge].unlock(C);
      }
      modEdgeRef.unlock(C);
    }
  }
  else { // neighbor is not local; send a special request to it
    pendingRequest = 1; // indicates that this element is awaiting a response
    mesh[nbr.cid].specialRequest(nbr.idx, myRef);
  }
}

void element::splitNeighborsLocal(int longEdge, int opnode, int othernode, 
				  int modEdge, int nbrLongEdge, int nbrOpnode,
				  int nbrOthernode, int nbrModEdge, 
				  const elemRef &nbr)
{
  // find the new node along longEdge
  int fixnode = 3 - opnode - othernode;
  node m;
  C->theNodes[nodes[othernode]].midpoint(C->theNodes[nodes[fixnode]], &m);

  // add new components to local chunk and get refs to them
  int mIdx = C->addNode(m);
  edgeRef newEdgeRef = C->addEdge();
  edgeRef newLongRef = C->addEdge();
  edgeRef newNbrEdgeRef = C->addEdge();
  elemRef newElemRef;
  if (opnode == 0) {
    if (othernode == 1)
      newElemRef = C->addElement(nodes[0], nodes[1], mIdx, edges[modEdge],
				 newLongRef, newEdgeRef);
    else // othernode == 2
      newElemRef = C->addElement(nodes[0], mIdx, nodes[2], newEdgeRef, 
				 newLongRef, edges[modEdge]);
  }
  else if (opnode == 1) {
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], nodes[1], mIdx, edges[modEdge], 
				 newEdgeRef, newLongRef);
    else // othernode == 2
      newElemRef = C->addElement(mIdx, nodes[1], nodes[2], newEdgeRef, 
				 edges[modEdge], newLongRef);
  }
  else { // opnode is 2
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], mIdx, nodes[2], newLongRef, 
				 newEdgeRef, edges[modEdge]);
    else // othernode == 1
      newElemRef = C->addElement(mIdx, nodes[1], nodes[2], newLongRef, 
				 edges[modEdge], newEdgeRef);
  }
  elemRef newNbrRef;
  if (nbrOpnode == 0) {
    if (nbrOthernode == 1)
      newNbrRef = C->addElement(C->theElements[nbr.idx].nodes[0], 
				 C->theElements[nbr.idx].nodes[1], mIdx, 
				 C->theElements[nbr.idx].edges[nbrModEdge], 
				 newLongRef, newNbrEdgeRef);
    else // nbrOthernode == 2
      newNbrRef = C->addElement(C->theElements[nbr.idx].nodes[0], mIdx, 
				 C->theElements[nbr.idx].nodes[2], 
				 newNbrEdgeRef, newLongRef, 
				 C->theElements[nbr.idx].edges[nbrModEdge]);
  }
  else if (nbrOpnode == 1) {
    if (nbrOthernode == 0)
      newNbrRef = C->addElement(C->theElements[nbr.idx].nodes[0], 
				 C->theElements[nbr.idx].nodes[1], mIdx, 
				 C->theElements[nbr.idx].edges[nbrModEdge], 
				 newNbrEdgeRef, newLongRef);
    else // nbrOthernode == 2
      newNbrRef = C->addElement(mIdx, C->theElements[nbr.idx].nodes[1], 
				 C->theElements[nbr.idx].nodes[2], 
				 newNbrEdgeRef, 
				 C->theElements[nbr.idx].edges[nbrModEdge], 
				 newLongRef);
  }
  else { // nbrOpnode is 2
    if (nbrOthernode == 0)
      newNbrRef = C->addElement(C->theElements[nbr.idx].nodes[0], mIdx, 
				 C->theElements[nbr.idx].nodes[2], newLongRef, 
				 newNbrEdgeRef, 
				 C->theElements[nbr.idx].edges[nbrModEdge]);
    else // nbrOthernode == 1
      newNbrRef = C->addElement(mIdx, C->theElements[nbr.idx].nodes[1], 
				 C->theElements[nbr.idx].nodes[2], newLongRef, 
				 C->theElements[nbr.idx].edges[nbrModEdge], 
				 newNbrEdgeRef);
  }

  // link everything together properly
  edges[modEdge].updateElement(C, myRef, newElemRef);
  C->theEdges[newEdgeRef.idx].updateElement(nullRef, myRef);
  C->theEdges[newEdgeRef.idx].updateElement(nullRef, newElemRef);
  C->theEdges[newLongRef.idx].updateElement(nullRef, newElemRef);
  C->theEdges[newLongRef.idx].updateElement(nullRef, newNbrRef);
  C->theEdges[newNbrEdgeRef.idx].updateElement(nullRef, nbr);
  C->theEdges[newNbrEdgeRef.idx].updateElement(nullRef, newNbrRef);
  C->theElements[nbr.idx].edges[nbrModEdge].updateElement(C, nbr, newNbrRef);
  C->theElements[newElemRef.idx].setTargetArea(targetArea);
  C->theElements[newNbrRef.idx].setTargetArea(C->theElements[nbr.idx].getTargetArea());
  nodes[othernode] = mIdx;
  edges[modEdge] = newEdgeRef;
  C->theElements[nbr.idx].nodes[nbrOthernode] = mIdx;
  C->theElements[nbr.idx].edges[nbrModEdge] = newNbrEdgeRef;

  // tell the world outside about the split
  if (C->theClient) C->theClient->split(myRef.idx, longEdge, othernode, 0.5,
					LOCAL_FIRST);
  if (C->theClient) C->theClient->split(nbr.idx, nbrLongEdge, nbrOthernode, 
					0.5, LOCAL_SECOND);

  // tell dependents they can go now
  tellDepend();
  C->theElements[nbr.idx].tellDepend();
    
  // calculate new areas for the original two elements and the two new
  // ones and cache the results
  calculateArea();
  C->theElements[newElemRef.idx].calculateArea();
  C->theElements[newNbrRef.idx].calculateArea();
  C->theElements[nbr.idx].calculateArea();  
}

void element::splitHelp(int longEdge)
{
  int othernode, opnode, modEdge, otherEdge;
  int newNodeIdx;

  // neighbor has refined: newNode, otherNode and newLongEdgeRef sent and
  // stored locally by requestResponse

  // initializations of shortcuts to affected parts of element
  opnode = (longEdge + 2) % 3;
  othernode = longEdge;
  modEdge = opnode;
  otherEdge = (longEdge + 1) % 3;
  if (!(otherNode == C->theNodes[nodes[othernode]])) {
    othernode = (longEdge + 1) % 3;
    modEdge = othernode;
    otherEdge = opnode;
  }

  edgeRef modEdgeRef = edges[modEdge];

  // lock perimeter
  if (!edges[modEdge].lock(C))
    return;
  if (!edges[otherEdge].lock(C)) {
    edges[modEdge].unlock(C);
    return;
  }

  // add new components and make proper connections
  newNodeIdx = C->addNode(newNode);
  edgeRef newEdgeRef = C->addEdge();
  elemRef newElemRef;
  if (opnode == 0) {
    if (othernode == 1)
      newElemRef = C->addElement(nodes[0], nodes[1], newNodeIdx, 
				 edges[modEdge], newLongEdgeRef, newEdgeRef);
    else // othernode == 2
      newElemRef = C->addElement(nodes[0], newNodeIdx, nodes[2], newEdgeRef, 
				 newLongEdgeRef, edges[modEdge]);
  }
  else if (opnode == 1) {
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], nodes[1], newNodeIdx, 
				 edges[modEdge], newEdgeRef, newLongEdgeRef);
    else // othernode == 2
      newElemRef = C->addElement(newNodeIdx, nodes[1], nodes[2], newEdgeRef, 
				 edges[modEdge], newLongEdgeRef);
  }
  else { // opnode is 2
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], newNodeIdx, nodes[2], 
				 newLongEdgeRef, newEdgeRef, edges[modEdge]);
    else // othernode == 1
      newElemRef = C->addElement(newNodeIdx, nodes[1], nodes[2], 
				 newLongEdgeRef, edges[modEdge], newEdgeRef);
  }

  C->theEdges[newEdgeRef.idx].updateElement(nullRef, myRef);
  C->theEdges[newEdgeRef.idx].updateElement(nullRef, newElemRef);
  newLongEdgeRef.updateElement(C, nullRef, newElemRef);
  edges[modEdge].updateElement(C, myRef, newElemRef);
  C->theElements[newElemRef.idx].setTargetArea(targetArea);
  nodes[othernode] = newNodeIdx;
  edges[modEdge] = newEdgeRef;

  // tell the world outside about the split
  if (C->theClient) C->theClient->split(myRef.idx, longEdge, othernode, 0.5,
					BOUND_SECOND);

  tellDepend();  // tell dependent it can go now
  specialRequest = pendingRequest = requestResponse = 0; // reset flags

  // unlock perimeter
  edges[longEdge].unlock(C);
  newLongEdgeRef.unlock(C);
  edges[otherEdge].unlock(C);
  modEdgeRef.unlock(C);

  // calculate areas of original and new element and cache results
  calculateArea();
  C->theElements[newElemRef.idx].calculateArea();
}

void element::splitResponse(int longEdge)
{
  int opnode, othernode, modEdge, otherEdge;

  // this element is first to refine of a pair of elements on differing chunks

  // initializations of shortcuts to affected parts of element
  opnode = (longEdge + 2) % 3;
  othernode = longEdge;
  modEdge = opnode;
  otherEdge = (longEdge + 1) % 3;
  edgeRef modEdgeRef = edges[modEdge];

  // lock perimeter
  if (!edges[longEdge].lock(C)) return;
  if (!edges[modEdge].lock(C)) {
    edges[longEdge].unlock(C);
    return;
  }
  if (!edges[otherEdge].lock(C)) {
    edges[modEdge].unlock(C);
    edges[longEdge].unlock(C);
    return;
  }

  // find midpoint on longest edge
  int fixnode = 3 - opnode - othernode;
  node m;
  C->theNodes[nodes[othernode]].midpoint(C->theNodes[nodes[fixnode]], &m);

  // add new components to local chunk and get refs to them
  int mIdx = C->addNode(m);
  edgeRef newEdgeRef = C->addEdge();
  edgeRef newLongRef = C->addEdge();
  elemRef newElemRef;
  if (opnode == 0) {
    if (othernode == 1)
      newElemRef = C->addElement(nodes[0], nodes[1], mIdx, edges[modEdge],
				 newLongRef, newEdgeRef);
    else // othernode == 2
      newElemRef = C->addElement(nodes[0], mIdx, nodes[2], newEdgeRef, 
				 newLongRef, edges[modEdge]);
  }
  else if (opnode == 1) {
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], nodes[1], mIdx, edges[modEdge], 
				 newEdgeRef, newLongRef);
    else // othernode == 2
      newElemRef = C->addElement(mIdx, nodes[1], nodes[2], newEdgeRef, 
				 edges[modEdge], newLongRef);
  }
  else { // opnode is 2
    if (othernode == 0)
      newElemRef = C->addElement(nodes[0], mIdx, nodes[2], newLongRef, 
				 newEdgeRef, edges[modEdge]);
    else // othernode == 1
      newElemRef = C->addElement(mIdx, nodes[1], nodes[2], newLongRef, 
				 edges[modEdge], newEdgeRef);
  }

  edges[modEdge].updateElement(C, myRef, newElemRef);
  C->theEdges[newEdgeRef.idx].updateElement(nullRef, myRef);
  C->theEdges[newEdgeRef.idx].updateElement(nullRef, newElemRef);
  C->theEdges[newLongRef.idx].updateElement(nullRef, newElemRef);

  if (!newLongRef.lock(C)) CkAbort("ERROR locking new long edge.\n");
  // tell other half of pair that it can proceed with refinement
  mesh[specialRequester.cid].specialRequestResponse(specialRequester.idx,
    m.X(), m.Y(), C->theNodes[nodes[othernode]].X(), 
    C->theNodes[nodes[othernode]].Y(), newLongRef);

  nodes[othernode] = mIdx;
  edges[modEdge] = newEdgeRef;
  C->theElements[newElemRef.idx].setTargetArea(targetArea);

  // tell the world outside about the split
  if (C->theClient) C->theClient->split(myRef.idx, longEdge, othernode, 0.5,
					BOUND_FIRST);

  specialRequest = pendingRequest = requestResponse = 0; // reset flags
  tellDepend();  // tell dependent it can go now

  // unlock perimeter
  edges[otherEdge].unlock(C);
  modEdgeRef.unlock(C);

  // calculate aeras of original and new elements and cache results
  calculateArea();
  C->theElements[newElemRef.idx].calculateArea();
}

void element::refineNeighbor(int longEdge)
{
  elemRef nbr = getNeighbor(longEdge);
  // this element and the neighbor on its long edge do not share the
  // same longEdge
  if (!nbr.hasDependent(C)) {
    double nbrArea = nbr.getArea(C);
    nbr.setDependent(C, myRef.idx, myRef.cid); // set nbr's dependent to this
    depend = 1;  // flag this element as dependent on another
    if (nbr.cid == myRef.cid) // nbr is local
      // force at least one refinement level on nbr
      C->theElements[nbr.idx].setTargetArea(nbrArea); 
    else // nbr not local; tell nbr's chunk to refine nbr element
      mesh[nbr.cid].refineElement(nbr.idx, nbrArea);
  }
  // Note: if neighbor already has a dependent, only that dependent
  // will be notified when it refines. So this element must not be
  // labelled as dependent on another.  Instead, it keeps attempting
  // to refine until either its neighbor has refined, or it is able
  // the make itself dependent on the neighbor.
}

void element::updateEdges(edgeRef e0, edgeRef e1, edgeRef e2)
{
  edges[0] = e0;
  edges[1] = e1;
  edges[2] = e2;
}

/**********************  chunk methods  ***********************/
chunk::chunk(chunkMsg *m)
  : TCharmClient1D(m->myThreads), numElements(0), numEdges(0), numNodes(0), 
    sizeElements(0), sizeEdges(0), sizeNodes(0),
    debug_counter(0), refineInProgress(0), modified(0),
    meshLock(0), meshExpandFlag(0), theClient(NULL)
{
  refineResultsStorage=NULL;
  cid = thisIndex;
  numChunks = m->nChunks;
  CkFreeMsg(m);
  tcharmClientInit();
  thread->resume();
}

void chunk::refineElement(int i, double area)
{
  // we indicate a need for refinement by reducing an element's targetArea
  accessLock();
  theElements[i].setTargetArea(area);
  releaseLock();
  modified = 1;  // flag a change in one of the chunk's elements
  if (!refineInProgress) { // if refine loop not running
    refineInProgress = 1;
    mesh[cid].refiningElements(); // start it up
  }
}

void chunk::refiningElements()
{
  int i;
  
  CkPrintf("[tmr] Chunk %d: refiningElements\n", cid);
  while (modified) { 
    // continue trying to refine elements until nothing changes during
    // a refinement cycle
    i = 0;
    modified = 0;
    CkPrintf("[tmr] Chunk %d: Entering internal refinement loop...\n", cid);
    while (i < numElements) { // loop through the elements
      if (theElements[i].getCachedArea() < 0.0) // no cached area yet
      	theElements[i].calculateArea();
      if ((!theElements[i].hasDependency() &&
	  (((theElements[i].getTargetArea() <= theElements[i].getCachedArea()) 
	    && (theElements[i].getTargetArea() >= 0.0)) 
	   || theElements[i].isSpecialRequest() 
	   || theElements[i].isPendingRequest()))
	  || theElements[i].isRequestResponse()) {
	// the element either needs refining or has been asked to
	// refine or has asked someone else to refine
	//CkPrintf("[tmr] Chunk %d: Element %d: hasdep? %c target=%f current=%f spcReq? %c pend? %c reqResp? %c\n", cid, i, (theElements[i].hasDependency() ? 'y' : 'n'), theElements[i].getTargetArea(), theElements[i].getCachedArea(), (theElements[i].isSpecialRequest() ? 'y' : 'n'), (theElements[i].isPendingRequest() ? 'y' : 'n'), (theElements[i].isRequestResponse() ? 'y' : 'n'));
	modified = 1; // something's bound to change
	theElements[i].refine(); // refine the element
      }
      i++;
    }
    //if (CkMyPe() == 0) for (int j=0; j<numChunks; j++) mesh[j].print();
    adjustMesh();
    CthYield(); // give other chunks on the same PE a chance
  }
  // nothing is in need of refinement; turn refine loop off
  refineInProgress = 0;  
  CkPrintf("[tmr] Chunk %d: DONE refiningElements\n", cid);
}


void chunk::updateElement(int i, objRef oldval, objRef newval)
{
  accessLock();
  theEdges[i].updateElement(elemRef(oldval.cid, oldval.idx), 
			    elemRef(newval.cid, newval.idx));
  releaseLock();
}

// special requests need to wake things up on the local chunk
void chunk::specialRequest(int requestee, elemRef requester)
{
  accessLock();
  theElements[requestee].setSpecialRequest(requester);
  releaseLock();
  //  CkPrintf("Element %d on chunk %d received special request from element %d on chunk %d.\n", m->requestee, cid, m->requester.idx, m->requester.cid);
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

void chunk::specialRequestResponse(int i, double newNodeX, double newNodeY, 
				   double otherNodeX, double otherNodeY, 
				   edgeRef newLongEdgeRef)
{
  node newNode(newNodeX, newNodeY);
  node otherNode(otherNodeX, otherNodeY);
  accessLock();
  theElements[i].setRequestResponse(newNode, otherNode, newLongEdgeRef);
  releaseLock();
  //  CkPrintf("Element %d on chunk %d received special request RESPONSE.\n", m->idx, cid);
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

doubleMsg *chunk::getArea(int i)
{
  doubleMsg *dm = new doubleMsg;
  accessLock();
  dm->aDouble = theElements[i].getArea();
  releaseLock();
  return dm;
}

intMsg *chunk::lock(int i)
{
  intMsg *rm = new intMsg;
  accessLock();
  if (theEdges[i].locked())
    rm->anInt = 0;
  else {
    theEdges[i].lock();
    rm->anInt = 1;
  }
  releaseLock();
  return rm;
}

void chunk::unlock(int i)
{
  accessLock();
  theEdges[i].unlock();
  releaseLock();
}


intMsg *chunk::locked(int i)
{
  intMsg *rm = new intMsg;
  accessLock();
  rm->anInt = theEdges[i].locked();
  releaseLock();
  return rm;
}

intMsg *chunk::checkElement(objRef oR, int i)
{
  intMsg *im = new intMsg;
  edgeRef eRef(oR.cid, oR.idx);

  accessLock();
  im->anInt = theElements[i].checkIfLongEdge(eRef);
  releaseLock();
  return im;
}

refMsg *chunk::getNeighbor(objRef oR, int i)
{
  refMsg *rm = new refMsg;
  elemRef er, ar(oR.cid, oR.idx);
  accessLock();
  er = theEdges[i].getNbrRef(ar);
  releaseLock();
  rm->aRef = er;
  return rm;
}

void chunk::setTargetArea(int i, double area)
{
  accessLock();
  theElements[i].setTargetArea(area);
  releaseLock();
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

void chunk::updateEdges(int i, edgeRef e0, edgeRef e1, edgeRef e2)
{
  accessLock();
  theElements[i].updateEdges(e0, e1, e2);
  releaseLock();
}

void chunk::setDependent(objRef oR, int i)
{
  accessLock();
  theElements[i].setDependent(oR.cid, oR.idx);
  releaseLock();
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

intMsg *chunk::hasDependent(int i)
{
  intMsg *result = new intMsg;
  accessLock();
  result->anInt = theElements[i].hasDependent();
  releaseLock();
  return result;
}

void chunk::unsetDependency(int i)
{
  accessLock();
  theElements[i].unsetDependency();
  releaseLock();
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

// the following methods are for run-time additions and modifications
// to the chunk components
void chunk::accessLock()
{
  while (meshExpandFlag)
    CthYield();
  meshLock--;
}

void chunk::releaseLock()
{
  meshLock++;
}

void chunk::adjustFlag()
{
  meshExpandFlag = 1;
}

void chunk::adjustLock()
{
  while (meshLock != 0)
    CthYield();
  meshLock = 1;
}

void chunk::adjustRelease()
{
  meshLock = meshExpandFlag = 0;
}

void chunk::adjustMesh()
{
  if ((sizeElements < numElements) || (sizeNodes < numNodes) || (sizeEdges < numEdges))
    CkPrintf("[tmr] WARNING!\n\n     Mesh overrun! Data is probably corrupted!!!\n");
  if (sizeElements <= numElements*4) {
    adjustFlag();
    adjustLock();
    CkPrintf("[tmr] [%d] Adjusting mesh size...\n", cid);
    sizeElements = numElements*4;
    sizeEdges = numEdges*12;
    sizeNodes = numNodes*12;
    theElements.resize(sizeElements);
    theEdges.resize(sizeEdges);
    theNodes.resize(sizeNodes);
    CkPrintf("[tmr] [%d] Done adjusting mesh size...\n", cid);
    adjustRelease();
  }
}

int chunk::addNode(node n)
{
  theNodes[numNodes] = n;
  theNodes[numNodes].init(this);
  numNodes++;
  return numNodes-1;
}


edgeRef chunk::addEdge()
{
  theEdges[numEdges].init(numEdges, this);
  edgeRef eRef(cid, numEdges);
  numEdges++;
  return eRef;
}

elemRef chunk::addElement(int n1, int n2, int n3)
{
  int n[3] = {n1, n2, n3};
  elemRef eRef(cid, numElements);
  theElements[numElements].init(n, numElements, this);
  theElements[numElements].calculateArea();
  numElements++;
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
  return eRef;
}

elemRef chunk::addElement(int n1, int n2, int n3,
			  edgeRef er1, edgeRef er2, edgeRef er3)
{
  int n[3] = {n1, n2, n3};
  edgeRef e[3] = {er1, er2, er3}; 

  elemRef eRef(cid, numElements);
  theElements[numElements].init(n, e, numElements, this);
  theElements[numElements].calculateArea();
  numElements++;
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
  return eRef;
}

// these two functions produce debugging output by printing somewhat
// sychronized versions of the entire mesh to files readable by tkplotter
void chunk::print()
{
  accessLock();
  debug_print(debug_counter);
  debug_counter++;
  releaseLock();
}

void chunk::debug_print(int c)
{
  FILE *fp;
  char filename[30];
  int i, j;

  memset(filename, 0, 30);
  sprintf(filename, "mesh_debug_%d.%d", cid, c);
  fp = fopen(filename, "w");

  fprintf(fp, "%d %d\n", cid, numElements);
  for (i=0; i<numElements; i++) {
    for (j=0; j<3; j++)
      fprintf(fp, "%f %f   ", theNodes[theElements[i].nodes[j]].X(), 
	      theNodes[theElements[i].nodes[j]].Y());
    fprintf(fp, "%d %f\n", i, theElements[i].getTargetArea());
    for (j=0; j<3; j++) {
      fprintf(fp, "%d  ", ((theElements[i].getEdge(j).locked(this))?2:0));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void chunk::out_print()
{
  FILE *fp;
  char filename[30];
  int i;

  memset(filename, 0, 30);
  sprintf(filename, "mesh.out");
  fp = fopen(filename, "a");

  if (cid == 0)
    fprintf(fp, "%d\n", numChunks);
  fprintf(fp, " %d %d %d %d\n", cid, numNodes, numEdges, numElements);
  for (i=0; i<numNodes; i++)
    fprintf(fp, "    %f %f\n", theNodes[i].X(), theNodes[i].Y());
  for (i=0; i<numEdges; i++) {
    fprintf(fp, "   ");
    fprintf(fp, " %d %d ", theEdges[i].elements[0].idx, theEdges[i].elements[0].cid);
    fprintf(fp, " %d %d\n", theEdges[i].elements[1].idx, theEdges[i].elements[1].cid);
  }
  for (i=0; i<numElements; i++) {
    fprintf(fp, " %d %d ", theElements[i].nodes[0], cid);
    fprintf(fp, " %d %d ", theElements[i].nodes[1], cid);
    fprintf(fp, " %d %d ", theElements[i].nodes[2], cid);
    fprintf(fp, "   ");
    fprintf(fp, " %d %d ", theElements[i].edges[0].idx, theElements[i].edges[0].cid);
    fprintf(fp, " %d %d ", theElements[i].edges[1].idx, theElements[i].edges[1].cid);
    fprintf(fp, " %d %d\n", theElements[i].edges[2].idx, theElements[i].edges[2].cid);
  }
  fprintf(fp, "\n");
  fclose(fp);
}

void chunk::updateNodeCoords(int nNode, double *coord, int nEl)
{
  int i;

#if 1
  // do some error checking
  if (nEl != numElements || nNode != numNodes) {
    CkPrintf("ERROR: inconsistency in REFINE2D's updateNodeCoords on chunk %d:\n"
       "  your nEl (%d); my numElements (%d)\n"
       "  your nNode (%d); my numNodes (%d)\n",
       cid, nEl, numElements, nNode,numNodes);
    CkAbort("User code/library numbering inconsistency in REFINE2D");
  }
#endif  
  
  // update node coordinates from coord
  for (i=0; i<numNodes; i++){
    theNodes[i].init(coord[2*i], coord[2*i + 1]);
    CkPrintf("[tmr] Chunk %d Node %d %.8lf %.8lf \n",cid,i,theNodes[i].X(),theNodes[i].Y());
  }  
    
  // recalculate and cache new areas for each element
  for (i=0; i<numElements; i++){
    theElements[i].calculateArea();
    CkPrintf("[tmr] Chunk %d Element %d area = %.8lf \n",cid,i, theElements[i].getArea());
  }	
  sanityCheck();
}

void chunk::multipleRefine(double *desiredArea, refineClient *client)
{
  int i;
  theClient = client; // initialize refine client associated with this chunk

  CkPrintf("[tmr] Chunk %d: multipleRefine\n", cid);
  // set desired areas for elements
  for (i=0; i<numElements; i++)
    if (desiredArea[i] < theElements[i].getArea()) {
      theElements[i].setTargetArea(desiredArea[i]);
      CkPrintf("[tmr] Chunk %d: Element %d to be refined from %.8lf to below %.8lf \n",
	       cid, i, theElements[i].getArea(), desiredArea[i]);
    }
  
  if (CkMyPe() == 0)
    for (i=0; i<numChunks; i++) 
      mesh[i].out_print();

  // start the refinement loop
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

/**************** Sanity Checking **********************/

void chunk::sanityCheck(void)
{
  int i;
  if (numElements<0 || (int)theElements.size()<numElements)
  	CkAbort("REFINE2D chunk elements insane!");
  if (numEdges<0 || (int)theEdges.size()<numEdges)
  	CkAbort("REFINE2D chunk edges insane!");
  if (numNodes<0 || (int)theNodes.size()<numNodes)
  	CkAbort("REFINE2D chunk nodes insane!");
  for (i=0;i<numElements;i++) {
    theElements[i].sanityCheck(this,elemRef(cid,i));
  }
  for (i=0;i<numEdges;i++) {
    theEdges[i].sanityCheck(this,edgeRef(cid,i));
  }
}

void objRef::sanityCheck(chunk *C)
{
  if (isNull()) 
    CkAbort("REFINE2D objRef is unexpectedly null");
}
void element::sanityCheck(chunk *c,elemRef shouldRef)
{
  if (myRef!=shouldRef)
    CkAbort("REFINE2D elem has wrong ref");
  
  for (int i=0;i<3;i++) {
    // nodes[i].sanityCheck(c);
    edges[i].sanityCheck(c);
  }
}
void edge::sanityCheck(chunk *c,edgeRef shouldRef)
{
  if (myRef!=shouldRef)
    CkAbort("REFINE2D edge has wrong ref");
  
  int nonNullElements=0;
  for (int i=0;i<2;i++) {
    // nodes[i].sanityCheck(c);
    if (!elements[i].isNull()) {
      elements[i].sanityCheck(c);
      nonNullElements++;
    }
  }
  if (!(nonNullElements==1 || nonNullElements==2))
    CkAbort("REFINE2D edge has unexpected number of neighbors!");
}
  

/****************** Initialization ******************/

void chunk::allocMesh(int nEl)
{
  sizeElements = nEl * 2;
  sizeNodes = sizeEdges = sizeElements * 3;
  theElements.resize(sizeElements);
  theNodes.resize(sizeNodes);
  theEdges.resize(sizeEdges);
  for (int i=0; i<sizeElements; i++) {
    theElements[i].init(); 
    theEdges[i].init();
  }
}

void chunk::newMesh(int nEl, int nGhost, const int *conn_, const int *gid_, int idxOffset)
{
  int i, j;

  CkPrintf("[tmr] newMesh on chunk %d...\n", cid);
  numElements = nEl;
  numGhosts = nGhost;
  allocMesh(nEl);
  int *conn = new int[3*numGhosts];
  int *gid = new int[2*numGhosts];
  
  // add elements to chunk
  for (i=0; i<numElements; i++) {
    int nodes[3];
    edgeRef edges[3];
    for (j=0; j<3; j++) {
      int c=conn_[i*3+j]-idxOffset;
      conn[i*3 + j]=c;
      nodes[j] = c;
      edges[j].init();
    }
    theElements[i].init(nodes, edges, i, this);
    gid[i*2] = cid;
    gid[i*2 + 1] = i;
  }

  // add ghost elements to chunk
  for (i=nEl; i<nGhost; i++) {
    for (j=0; j<3; j++)
      conn[i*3+j] = conn_[i*3+j]-idxOffset;
    gid[i*2+0] = gid_[i*2]-idxOffset;
    gid[i*2+1] = gid_[i*2 + 1]-idxOffset;
  }

/***** derive edges from elements on this chunk ****/
  
  // need to add edges to the chunk, and update all edgeRefs on all elements
  // also need to add nodes to the chunk
  int n1localIdx, n2localIdx, newEdge;

  deriveNodes(); // now numNodes and theNodes have values
  
  for (i=0; i<numElements; i++) {
    elemRef myRef(cid,i);
    for (j=0; j<3; j++) {
      n1localIdx = j;
      n2localIdx = (j+1) % 3;

      // look for edge
      if (theElements[i].edges[j] == nullRef) { // the edge doesn't exist yet
	// get nbr ref
	elemRef nbrRef;
	int edgeIdx = getNbrRefOnEdge(theElements[i].nodes[n1localIdx], 
				      theElements[i].nodes[n2localIdx], 
				      conn, numGhosts, gid, i, &nbrRef); 
	if (edgeLocal(myRef, nbrRef)) { // make edge here
	  newEdge = addNewEdge();
	  // point edge to the two neighboring elements
	  theEdges[newEdge].updateElement(nullRef, myRef);
	  theEdges[newEdge].updateElement(nullRef, nbrRef);
	  // point elem i's edge j at the edge
	  theElements[i].updateEdge(j, theEdges[newEdge].getRef());
	  // point nbrRef at the edge
	  if (nbrRef.cid==cid) // Local neighbor
	    theElements[nbrRef.idx].updateEdge(edgeIdx,
					       theEdges[newEdge].getRef());
	  else if (nbrRef.cid != -1) // Remote neighbor
	    mesh[nbrRef.cid].addRemoteEdge(nbrRef.idx, edgeIdx, 
					   theEdges[newEdge].getRef());
	}
	// else edge will be made on a different chunk
      }
    }
  }
  delete[] conn;
  delete[] gid;
  CkPrintf("[tmr] Finished newMesh on chunk %d...\n", cid);
}

void chunk::deriveNodes()
{
  int i, j;
  int aNode;

  numNodes = 0;
  for (i=0; i<numElements; i++) {
    for (j=0; j<3; j++) {
      aNode = theElements[i].nodes[j];
      if (aNode >= 0) {
	theNodes[aNode].init(this);
	if ((aNode + 1) > numNodes)
	  numNodes = aNode + 1;
      }
      else CkAbort("[tmr] ERROR: negative node id found in conn...\n");
    }
  }
}

void chunk::addRemoteEdge(int elem, int localEdge, edgeRef er)
{
  while (sizeElements == 0) CthYield();
  theElements[elem].updateEdge(localEdge, er);
}

int chunk::edgeLocal(elemRef e1, elemRef e2)
{
  if (e1.cid==-1 || e2.cid==-1) 
    return 1; //Edge lies on external boundary-- one of its elements is missing
  if (e1.cid == e2.cid)
    return 1; //Edge is completely internal-- both elements on my chunk
  if (e1.idx > e2.idx)
    return 1;
  else if ((e1.idx == e2.idx) && (e1.cid > e2.cid))
    return 1;
  else return 0;
}

int chunk::addNewEdge()
{
  theEdges[numEdges].init(numEdges, this);
  numEdges++;
  return numEdges-1;
}

int chunk::getNbrRefOnEdge(int n1, int n2, int *conn, int nGhost, int *gid, 
			   int idx, elemRef *er)
{
  int i, e;
  er->init();
  for (i=idx+1; i<nGhost; i++)
    if ((e = hasEdge(n1, n2, conn, i)) != -1) {
      er->init(gid[i*2], gid[i*2+1]);
      return e;
    }
  return -1;
}

int chunk::hasEdge(int n1, int n2, int *conn, int idx) 
{
  for (int i=0; i<3; i++) {
    int a=i;
    int b=(i+1)%3;
    if (((conn[idx*3+a] == n1) && (conn[idx*3+b] == n2)) ||
	((conn[idx*3+b] == n1) && (conn[idx*3+a] == n2)))
      return i;
  }
  return -1;
	
}

#include "refine.def.h"
