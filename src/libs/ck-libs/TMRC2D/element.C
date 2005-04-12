#include "element.h"
#include "tri.h"

int element::lockOpNode(edgeRef e, double l) 
{
  int edgeIdx = getEdgeIdx(e);
  int opNode = (edgeIdx + 2) % 3;
  return C->theNodes[nodes[opNode]].lock(l, e);
}

void element::unlockOpNode(edgeRef e) 
{
  int edgeIdx = getEdgeIdx(e);
  int opNode = (edgeIdx + 2) % 3;
  C->theNodes[nodes[opNode]].unlock();
}

void element::calculateArea()
{ // calulate area of triangle using Heron's formula:
  // Let a, b, c be the lengths of the three sides.
  // Area=SQRT(s(s-a)(s-b)(s-c)), where s=(a+b+c)/2 or perimeter/2.
  int i;
  node n[3];
  double s, perimeter, len[3];
  for (i=0; i<3; i++) n[i] = C->theNodes[nodes[i]];
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

void element::refine()
{
  int longEdge = findLongestEdge();
  split(longEdge);
}

void element::split(int longEdge)
{ /*                  opnode
                         @
                        / \  
                       /   \
           otherEdge  /     \  modEdge
                     /       \
                    /         \
           fixnode @___________@ othernode
                      longEdge                         */
  int opnode, othernode, fixnode, modEdge, otherEdge, result, local, first;
  edgeRef e_prime, newEdge;
  int m = -2, nullNbr=0;
  elemRef newElem, nullRef;

  // initializations of shortcuts to affected parts of element
  opnode = (longEdge + 2) % 3;
  othernode = longEdge;
  modEdge = opnode;
  otherEdge = (longEdge + 1) % 3;
  fixnode = otherEdge;

  int fIdx, oIdx;
  if (edges[longEdge].cid == myRef.cid) {
    fIdx = nodes[fixnode];
    oIdx = nodes[othernode];
  }
  else {
    FEM_Node *theNodes = &(C->meshPtr->node);
    FEM_Comm_Rec *oNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[othernode]));
    FEM_Comm_Rec *fNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[fixnode]));
    edge e;
    fIdx = fNodeRec->getIdx(e.existsOn(fNodeRec, edges[longEdge].cid));
    oIdx = oNodeRec->getIdx(e.existsOn(oNodeRec, edges[longEdge].cid));
    CkAssert(fIdx > -1);
    CkAssert(oIdx > -1);
    CkAssert(oIdx != fIdx);
  }

  if ((result=edges[longEdge].split(&m, &e_prime,oIdx, fIdx,
				    myRef, &local, &first, &nullNbr)) == 1) {
    // e_prime successfully created incident on othernode
    DEBUGREF(CkPrintf("TMRC2D: Refining element %d, opnode=%d ^othernode=%d fixnode=%d longEdge=%d modEdge=%d otherEdge=%d\n", myRef.idx, nodes[opnode], nodes[othernode], nodes[fixnode], edges[longEdge].idx, edges[modEdge].idx, edges[otherEdge].idx);)
    DEBUGREF(CkPrintf("TMRC2D: to FEM: element=%d local=%d first=%d between nodes %d and %d\n", myRef.idx, local, first, nodes[othernode], nodes[fixnode]);)
    newEdge = C->addEdge();
    DEBUGREF(CkPrintf("TMRC2D: New edge (%d,%d) added between nodes %d and %d\n", newEdge.cid, newEdge.idx, m, nodes[opnode]);)
    // add new element to preserve orientation
    if (opnode == 0) {
      if (othernode == 1)
	newElem = C->addElement(nodes[0], nodes[1], m, 
				edges[modEdge], e_prime, newEdge);
      else // othernode == 2
	newElem = C->addElement(nodes[0], m, nodes[2],
				newEdge, e_prime, edges[modEdge]);
    }
    else if (opnode == 1) {
      if (othernode == 0)
	newElem = C->addElement(nodes[0], nodes[1], m,
				edges[modEdge], newEdge, e_prime);
      else // othernode == 2
	newElem = C->addElement(m, nodes[1], nodes[2],
				newEdge, edges[modEdge], e_prime);
    }
    else { // opnode == 2
      if (othernode == 0)
	newElem = C->addElement(nodes[0], m, nodes[2],
				e_prime, newEdge, edges[modEdge]);
      else // othernode == 1
	newElem = C->addElement(m, nodes[1], nodes[2],
				e_prime, edges[modEdge], newEdge);
    }
    edges[modEdge].update(myRef, newElem);
    C->theEdges[newEdge.idx].update(nullRef, myRef);
    C->theEdges[newEdge.idx].update(nullRef, newElem);
    e_prime.update(nullRef, newElem);
    nodes[othernode] = m;
    edges[modEdge].checkPending(myRef, newElem);
    edges[modEdge] = newEdge;
    edges[otherEdge].checkPending(myRef);
    C->theElements[newElem.idx].setTargetArea(targetArea);
    calculateArea(); // update cached area of original element
    C->theElements[newElem.idx].calculateArea(); // and of new element
    // tell the world outside about the split
    int flag;
    if (local && first) flag = LOCAL_FIRST;
    if (local && !first) flag = LOCAL_SECOND;
    if (!local && first) flag = BOUND_FIRST;
    if (!local && !first) flag = BOUND_SECOND;
    if(C->theClient)C->theClient->split(myRef.idx,longEdge,othernode,0.5,flag);
    if (nullNbr){ DEBUGREF(CkPrintf("TMRC2D: nbr is null\n");)}
    if (!first || nullNbr) {
      if (!first) {
	DEBUGREF(CkPrintf("TMRC2D: Resetting pending edges, second split complete.\n");)
      }
      else if (nullNbr) {
	DEBUGREF(CkPrintf("TMRC2D: Resetting pending edges, neighbor NULL.\n");)      }
      edges[longEdge].resetEdge();
    }
  }
  else if (result == 0) { 
    // e_prime already incident on fixnode
  DEBUGREF(CkPrintf("TMRC2D: Refining element %d, opnode=%d othernode=%d ^fixnode=%d longEdge=%d modEdge=%d otherEdge=%d\n", myRef.idx, nodes[opnode], nodes[othernode], nodes[fixnode], edges[longEdge].idx, edges[modEdge].idx, edges[otherEdge].idx);)
      DEBUGREF(CkPrintf("TMRC2D: to FEM: element=%d local=%d first=%d between nodes %d and %d\n", myRef.idx, local, first, nodes[fixnode], nodes[othernode]);)
    newEdge = C->addEdge();
    DEBUGREF(CkPrintf("TMRC2D: New edge (%d,%d) added between nodes %d and %d\n", newEdge.cid, newEdge.idx, m, nodes[opnode]);)
    // add new element to preserve orientation
    if (opnode == 0) {
      if (fixnode == 1)
	newElem = C->addElement(nodes[0], nodes[1], m, 
				edges[otherEdge], e_prime, newEdge);
      else // fixnode == 2
	newElem = C->addElement(nodes[0], m, nodes[2],
				newEdge, e_prime, edges[otherEdge]);
    }
    else if (opnode == 1) {
      if (fixnode == 0)
	newElem = C->addElement(nodes[0], nodes[1], m,
				edges[otherEdge], newEdge, e_prime);
      else // fixnode == 2
	newElem = C->addElement(m, nodes[1], nodes[2],
				newEdge, edges[otherEdge], e_prime);
    }
    else { // opnode == 2
      if (fixnode == 0)
	newElem = C->addElement(nodes[0], m, nodes[2],
				e_prime, newEdge, edges[otherEdge]);
      else // fixnode == 1
	newElem = C->addElement(m, nodes[1], nodes[2],
				e_prime, edges[otherEdge], newEdge);
    }
    edges[otherEdge].update(myRef, newElem);
    C->theEdges[newEdge.idx].update(nullRef, myRef);
    C->theEdges[newEdge.idx].update(nullRef, newElem);
    e_prime.update(nullRef, newElem);
    nodes[fixnode] = m;
    edges[otherEdge].checkPending(myRef, newElem);
    edges[otherEdge] = newEdge;
    edges[modEdge].checkPending(myRef);
    C->theElements[newElem.idx].setTargetArea(targetArea);
    calculateArea(); // update cached area of original element
    C->theElements[newElem.idx].calculateArea(); // and of new element
    // tell the world outside about the split
    int flag;
    CkAssert(!first);
    if (local) flag = LOCAL_SECOND;
    else  flag = BOUND_SECOND;
    if (C->theClient) C->theClient->split(myRef.idx,longEdge,fixnode,0.5,flag);
    DEBUGREF(CkPrintf("TMRC2D: Resetting pending edges, second split complete.\n");)
    edges[longEdge].resetEdge();
  }
  else { // longEdge still trying to complete previous split; try later
    // do nothing for now
    DEBUGREF(CkPrintf("TMRC2D: Can't bisect element %d, longEdge %d pending\n", myRef.idx, edges[longEdge].idx);)
  }
}

void element::coarsen()
{
  int shortEdge = findShortestEdge();
  DEBUGREF(CkPrintf("TMRC2D: [%d] ...Coarsen element %d on edge %d\n", myRef.cid, myRef.idx, shortEdge);)
  collapse(shortEdge);
}


void element::collapse(int shortEdge)
{
/*                    opnode
                         @
                        / \  
       keepNbr         /   \        delNbr
              keepEdge/     \delEdge
                     /       \
                    /         \
           keepNode@_____m_____@delNode
                     shortEdge            
             
                        nbr                      */  

  int opnode, delNode, keepNode, delEdge, keepEdge, result;
  elemRef keepNbr, delNbr, nbr;
  int local, first, flag, kBound, dBound, kFixed, dFixed;

  // check if a different edge from the shortEdge is pending for coarsening
  if (edges[shortEdge].isPending(myRef)) {
  }
  else if (edges[(shortEdge+1)%3].isPending(myRef)) {
    shortEdge = (shortEdge+1)%3;
  }
  else if (edges[(shortEdge+2)%3].isPending(myRef)) {
    shortEdge = (shortEdge+2)%3;
  }

  // set up all the variables
  opnode = (shortEdge + 2) % 3;
  delNode = shortEdge;
  delEdge = opnode;
  keepEdge = (shortEdge + 1) % 3;
  keepNode = keepEdge;
  keepNbr = edges[keepEdge].getNbr(myRef);
  delNbr = edges[delEdge].getNbr(myRef);
  nbr = edges[shortEdge].getNbr(myRef);
  // get the boundary flags for the nodes on the edge to collapse
  kBound = C->theNodes[nodes[keepNode]].boundary;
  dBound = C->theNodes[nodes[delNode]].boundary;
  kFixed = C->theNodes[nodes[keepNode]].fixed;
  dFixed = C->theNodes[nodes[delNode]].fixed;

  CkAssert(!(nbr == myRef));
  CkAssert(!(keepNbr == myRef));
  CkAssert(!(delNbr == myRef));
  // find coords of node to collapse to based on boundary conditions
  node newNode;
  if ((kBound == 0) && (dBound == 0)) { // both interior; collapse to midpoint
    if (!kFixed && !dFixed) {
      newNode=C->theNodes[nodes[keepNode]].midpoint(C->theNodes[nodes[delNode]]);
      newNode.boundary = 0;
    }
    else if (dFixed && kFixed) return;
    else if (dFixed) {
      newNode = C->theNodes[nodes[delNode]];
    }
    else {
      newNode = C->theNodes[nodes[delNode]];
    }
  }
  else if ((kBound == 0) || (dBound == 0)) { // only one on boundary
    // collapse edge to boundary node
    if (kBound && !dFixed) {
      newNode = C->theNodes[nodes[keepNode]];
    }
    else if (dBound && !kFixed) {
      newNode = C->theNodes[nodes[delNode]];
    }
    else return;
  }
  else if (kBound == dBound) { // both on same boundary
    // check fixed status of both nodes
    if (kFixed && dFixed) return; // if both fixeds don't refine
    else if (kFixed || dFixed) { // if one fixed, collapse edge to fixed
      if (kFixed) {
	newNode = C->theNodes[nodes[keepNode]];
      }
      else {
	newNode = C->theNodes[nodes[delNode]];
      }
    }
    else { // neither are fixeds, collapse edge to midpoint
      newNode=C->theNodes[nodes[keepNode]].midpoint(C->theNodes[nodes[delNode]]);
      newNode.boundary = kBound;
    }
  }
  else { // nodes on different boundary
    if (nbr.cid >= 0) return; // edge is internal; don't coarsen
    else { // if it isn't check if lower boundary node is a fixed
      if (dBound > kBound) { // dBound is numbered higher
	if (kFixed) return; // if it is, don't coarsen
	else { // if it isn't, collapse edge to larger boundary node
	  newNode = C->theNodes[nodes[delNode]];
	}
      }
      else { // kBound is numbered higher
	if (dFixed) return; // if it is, don't coarsen
	else { // if it isn't, collapse edge to larger boundary node
	  newNode = C->theNodes[nodes[keepNode]];
	}
      }
    }
  }

  int kIdx, dIdx;
  if (edges[shortEdge].cid == myRef.cid) {
    kIdx = nodes[keepNode];
    dIdx = nodes[delNode];
  }
  else {
    FEM_Node *theNodes = &(C->meshPtr->node);
    FEM_Comm_Rec *dNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[delNode]));
    FEM_Comm_Rec *kNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[keepNode]));
    edge e;
    kIdx = kNodeRec->getIdx(e.existsOn(kNodeRec, edges[shortEdge].cid));
    dIdx = dNodeRec->getIdx(e.existsOn(dNodeRec, edges[shortEdge].cid));
    CkAssert(kIdx > -1);
    CkAssert(dIdx > -1);
    CkAssert(dIdx != kIdx);
  }
  // collapse the edge; takes care of neighbor element
  DEBUGREF(CkPrintf("TMRC2D: [%d] ...kIdx=%d dIdx=%d\n", myRef.cid, kIdx, dIdx);)
  result = edges[shortEdge].collapse(myRef, kIdx, dIdx, keepNbr, delNbr, 
				     edges[keepEdge], edges[delEdge], 
				     C->theNodes[nodes[opnode]], 
				     &local, &first, newNode);

  // clean up based on result of edge collapse
  if (result == 1) {
    // collapse successful; keepNode is node to keep
    DEBUGREF(CkPrintf("TMRC2D: [%d] ...In collapse[%d](a) shortEdge=%d delEdge=%d keepEdge=%d opnode=%d delNode=%d keepNode=%d delNbr=%d keepNbr=%d\n", myRef.cid, myRef.idx, edges[shortEdge].idx, edges[delEdge].idx, edges[keepEdge].idx, nodes[opnode], dIdx, kIdx, delNbr.idx, keepNbr.idx);)
    // tell delNbr to replace delEdge with keepEdge
    if (delNbr.cid != -1)
      mesh[delNbr.cid].updateElementEdge(delNbr.idx, edges[delEdge], 
					 edges[keepEdge]);
    // tell keepEdge to replace myRef with delNbr
    edges[keepEdge].update(myRef, delNbr);
    // remove delEdge
    edges[delEdge].remove();
    // edge[shortEdge] handles removal of delNode and shortEdge, as well as
    // update of keepNode; so nothing else to do here
    // Notify FEM client of the collapse
    if (local && first) flag = LOCAL_FIRST;
    if (local && !first) flag = LOCAL_SECOND;
    if (!local && first) flag = BOUND_FIRST;
    if (!local && !first) flag = BOUND_SECOND;
    C->theClient->collapse(myRef.idx, kIdx, dIdx, newNode.X(), newNode.Y(), 
			   flag);
    DEBUGREF(CkPrintf("TMRC2D: [%d] theClient->collapse(%d, %d, %d, %2.10f, %2.10f\n", myRef.cid, myRef.idx, kIdx, dIdx, newNode.X(), newNode.Y());)
    // remove self
    C->removeElement(myRef.idx);
  }
  else if (result == 0) {
    // collapse successful, but first half of collapse decided to keep delNode
    // remap for sanity
    keepNode = shortEdge;
    keepEdge = opnode;
    delEdge = (shortEdge + 1) % 3;
    delNode = delEdge;
    // tell delNbr to replace delEdge with keepEdge
    keepNbr = edges[keepEdge].getNbr(myRef);
    delNbr = edges[delEdge].getNbr(myRef);
    int mytmp = dIdx;
    dIdx = kIdx;    
    kIdx = mytmp;
    DEBUGREF(CkPrintf("TMRC2D: [%d] ...In collapse[%d](b) shortEdge=%d delEdge=%d keepEdge=%d opnode=%d delNode=%d keepNode=%d delNbr=%d keepNbr=%d\n", myRef.cid, myRef.idx, edges[shortEdge].idx, edges[delEdge].idx, edges[keepEdge].idx, nodes[opnode], dIdx, kIdx, delNbr.idx, keepNbr.idx);)
    if (delNbr.cid != -1)
      mesh[delNbr.cid].updateElementEdge(delNbr.idx, edges[delEdge], 
					 edges[keepEdge]);
    // tell keepEdge to replace myRef with delNbr
    edges[keepEdge].update(myRef, delNbr);
    // remove delEdge
    edges[delEdge].remove();
    // edge[shortEdge] handles removal of delNode and shortEdge, as well as
    // update of keepNode; so nothing else to do here
    // Notify FEM client of the collapse
    if (local && first) flag = LOCAL_FIRST;
    if (local && !first) flag = LOCAL_SECOND;
    if (!local && first) flag = BOUND_FIRST;
    if (!local && !first) flag = BOUND_SECOND;
    DEBUGREF(CkPrintf("TMRC2D: [%d] theClient->collapse(%d, %d, %d, %2.10f, %2.10f)\n", myRef.cid, myRef.idx, kIdx, dIdx, newNode.X(), newNode.Y());)
    C->theClient->collapse(myRef.idx, kIdx, dIdx,
			   newNode.X(), newNode.Y(), flag);
    // remove self
    C->removeElement(myRef.idx);
  }
  // else coarsen is pending
}

int element::findLongestEdge()
{
  int i, longEdge;
  double maxlen = 0.0, len[3];
  // fine lengths of sides
  len[0] = C->theNodes[nodes[0]].distance(C->theNodes[nodes[1]]);
  len[1] = C->theNodes[nodes[1]].distance(C->theNodes[nodes[2]]);
  len[2] = C->theNodes[nodes[2]].distance(C->theNodes[nodes[0]]);
  for (i=0; i<3; i++) // find max length of a side
    if (len[i] > maxlen) {
      longEdge = i;
      maxlen = len[i];
    }
  return longEdge;
}

int element::findShortestEdge()
{
  int i, shortEdge = 0;
  double minlen, len[3];
  // fine lengths of sides
  minlen = len[0] = C->theNodes[nodes[0]].distance(C->theNodes[nodes[1]]);
  len[1] = C->theNodes[nodes[1]].distance(C->theNodes[nodes[2]]);
  len[2] = C->theNodes[nodes[2]].distance(C->theNodes[nodes[0]]);
  for (i=1; i<3; i++) // find min length of a side
    if (len[i] < minlen) {
      shortEdge = i;
      minlen = len[i];
    }
  return shortEdge;
}

int element::isLongestEdge(edgeRef& e)
{
  int longEdge = findLongestEdge();
  return (edges[longEdge] == e);
}

/*
void element::tweakNodes()
{
  node n[3], tn[3];
  int i;

  for (i=0; i<3; i++)
    n[i] = nodes[i].get();
  for (i=0; i<3; i++)
    tn[i] = tweak(n, i);
  for (i=0; i<3; i++) {
    nodes[i].reportPos(tn[i]);
    nodes[i].reportPos(n[(i+1)%3]);
    nodes[i].reportPos(n[(i+2)%3]);
  }
}

node element::tweak(node n[3], int i)
{
  int pal=(i+1)%3, acq=(i+2)%3;
  double iLoc, result1Loc, result2Loc;
  double L, d;
  node mid, result1, result2;

  L = (n[pal].distance(n[acq]))/2.0;
  d = getArea() / L;
  mid = n[pal].midpoint(n[acq]);
  
  result1.set(mid.X() + (d * (n[pal].Y() - mid.Y()))/L,
	      mid.Y() - (d * (n[pal].X() - mid.X()))/L);
  result2.set(mid.X() - (d * (n[pal].Y() - mid.Y()))/L,
	      mid.Y() + (d * (n[pal].X() - mid.X()))/L);

  iLoc = n[i].Y() - n[acq].Y() - 
    ((n[pal].Y() - n[acq].Y())/(n[pal].X() - n[acq].X()) *
     (n[i].X() - n[acq].X()));
  result1Loc = result1.Y() - n[acq].Y() - 
    ((n[pal].Y() - n[acq].Y())/(n[pal].X() - n[acq].X()) *
     (result1.X() - n[acq].X()));
  result2Loc = result2.Y() - n[acq].Y() - 
    ((n[pal].Y() - n[acq].Y())/(n[pal].X() - n[acq].X()) *
     (result2.X() - n[acq].X()));

  if ((iLoc > 0.0) && (result1Loc > 0.0)) return result1;
  else if ((iLoc < 0.0) && (result1Loc < 0.0)) return result1;
  else if ((iLoc > 0.0) && (result2Loc > 0.0)) return result2;
  else if ((iLoc < 0.0) && (result2Loc < 0.0)) return result2;
  else {
    CkPrintf("ERROR: tweak: can't find where point lies: %f %f %f\n",
	     iLoc, result1Loc, result2Loc);
    CkPrintf("Original: [%f,%f]  Result 1: [%f,%f]  Result 2: [%f,%f]\n", n[i].X(), n[i].Y(), result1.X(), result1.Y(), result2.X(), result2.Y());
    return mid;
  }
}
*/

void element::sanityCheck(chunk *c, elemRef shouldRef, int n) 
{
  CkAssert(myRef == shouldRef);
  CkAssert(C == c);
  for (int i=0;i<3;i++) {
    CkAssert((nodes[i] < n) && (nodes[i] > -1));
    CkAssert(!(edges[i].isNull()));
    edges[i].sanityCheck();
  }
}
