#include "element.h"
#include "tri.h"

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

  if ((result=edges[longEdge].split(&m, &e_prime,C->theNodes[nodes[othernode]],
				    C->theNodes[nodes[fixnode]],
				    myRef, &local, &first, &nullNbr)) == 1) {
    // e_prime successfully created incident on othernode
    CkPrintf("TMRC2D: Refining element %d, opnode=%d ^othernode=%d fixnode=%d longEdge=%d modEdge=%d otherEdge=%d\n", myRef.idx, nodes[opnode], nodes[othernode], nodes[fixnode], edges[longEdge].idx, edges[modEdge].idx, edges[otherEdge].idx);
    CkPrintf("TMRC2D: to FEM: element=%d local=%d first=%d between nodes %d and %d\n", myRef.idx, local, first, nodes[othernode], nodes[fixnode]);
    newEdge = C->addEdge();
    CkPrintf("TMRC2D: New edge (%d,%d) added between nodes %d and %d\n", 
	     newEdge.cid, newEdge.idx, m, nodes[opnode]);
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
    if (nullNbr) CkPrintf("TMRC2D: nbr is null\n");
    if (!first || nullNbr) {
      if (!first)
	CkPrintf("TMRC2D: Resetting pending edges, second split complete.\n");
      else if (nullNbr)
	CkPrintf("TMRC2D: Resetting pending edges, neighbor NULL.\n");
      edges[longEdge].resetEdge();
    }
  }
  else if (result == 0) { 
    // e_prime already incident on fixnode
  CkPrintf("TMRC2D: Refining element %d, opnode=%d othernode=%d ^fixnode=%d longEdge=%d modEdge=%d otherEdge=%d\n", myRef.idx, nodes[opnode], nodes[othernode], nodes[fixnode], edges[longEdge].idx, edges[modEdge].idx, edges[otherEdge].idx);
      CkPrintf("TMRC2D: to FEM: element=%d local=%d first=%d between nodes %d and %d\n", myRef.idx, local, first, nodes[fixnode], nodes[othernode]);
    newEdge = C->addEdge();
    CkPrintf("TMRC2D: New edge (%d,%d) added between nodes %d and %d\n", 
	     newEdge.cid, newEdge.idx, m, nodes[opnode]);
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
    CkPrintf("TMRC2D: Resetting pending edges, second split complete.\n");
    edges[longEdge].resetEdge();
  }
  else { // longEdge still trying to complete previous split; try later
    // do nothing for now
    CkPrintf("TMRC2D: Can't bisect element %d, longEdge %d pending\n", myRef.idx, edges[longEdge].idx);
  }
}

void element::coarsen()
{
  CkAbort("NOT IMPLEMENTED! element::coarsen()\n");
/*
  int shortEdge = findShortestEdge();
  int n1, n2, e1, e2;

  n1 = (4-shortEdge)%3;
  n2 = (3-shortEdge)%3;
  e1 = (shortEdge+2)%3;
  e2 = (shortEdge+1)%3;

  if (nodes[n1].lock()) {
    if (nodes[n2].lock()) {
      collapse(shortEdge, n1, n2, e1, e2);
      nodes[n2].unlock();
    }
    nodes[n1].unlock();
  }
*/
}


//void element::collapse(int shortEdge, int n1, int n2, int e1, int e2)
//{
/*                       @n3
                        / \  
                       /   \
                    e1/     \e2
                     /       \
                    /         \
                 n1@_____m_____@n2
                     shortEdge                         */  

/*
  node m;  // midpoint on edge to collapse;
  elemRef elem2, opElem;
  updateMsg *um = new updateMsg;
  int n3 = 3 - (n1 + n2), s1, s2;

  edges[shortEdge].midpoint(m); // find midpoint on shortest edge

  // Before we do anything, we need to look at the two nodes whose
  // positions will change: if a node is on the border, we cannot move
  // it; if a node causes a pair of incident edges to flip their order
  // of incidence on the node when it moves, we cannot move it
  s1 = nodes[n1].safeToMove(m, myRef, edges[shortEdge], edges[e1], nodes[n1], nodes[n2], nodes[n3]);
  s2 = nodes[n2].safeToMove(m, myRef, edges[shortEdge], edges[e1], nodes[n2], nodes[n2], nodes[n3]);
  if (!s1 && !s2) { // can't move either node
    targetArea = -1.0;  // don't bother to coarsen
    return; // no border movement or flipping elements allowed
  }
  else if (s1 && !s2) {
    int tmp = shortEdge;
    shortEdge = e1; 
    e1 = tmp;
    tmp = n2;
    n2 = n3; 
    n3 = tmp;
    edges[shortEdge].midpoint(m); // find midpoint on shortest edge
    s2 = nodes[n2].safeToMove(m, myRef, edges[shortEdge], edges[e1], nodes[n2], nodes[n1], nodes[n3]);
    if (!s2) { // can't move two of the nodes
      targetArea = -1.0;  // don't bother to coarsen
      return; // no border movement or flipping elements allowed
    }
  }
  else if (s2 && !s1) {
    int tmp = shortEdge;
    shortEdge = e2; 
    e2 = tmp;
    tmp = n1;
    n1 = n3; 
    n3 = tmp;
    edges[shortEdge].midpoint(m); // find midpoint on shortest edge
    s1 = nodes[n1].safeToMove(m, myRef, edges[shortEdge], edges[e1], nodes[n1], nodes[n2], nodes[n3]);
    if (!s1) { // can't move two of the nodes
      targetArea = -1.0;  // don't bother to coarsen
      return; // no border movement or flipping elements allowed
    }
  }
  // end of border/edge flip tests
  
  elem2 = edges[e2].get(myRef);
  opElem = edges[shortEdge].get(myRef);

  nodes[n1].update(m);
  edges[e1].update(myRef, elem2);
  if (elem2.idx != -1)  elem2.update(edges[e2], edges[e1]);
  if (opElem.idx != -1) { // need to collapse opElem too
    opElem.collapseHelp(edges[shortEdge], nodes[n1], nodes[n2]);
  }

  um->oldval = nodes[n2];
  um->newval = nodes[n1];
  mesh.updateReferences(um);
  nodes[n2].remove();
  edges[e2].remove();
  edges[shortEdge].remove();
  myRef.remove();
}
*/

//void element::collapseHelp(edgeRef shortEdgeRef, nodeRef n1ref, nodeRef n2ref)
//{
/*                       @
                        / \  
                       /   \
                    e1/     \e2
                     /       \
                    /         \
                 n1@_____m_____@n2
                     shortEdge                         */ 
/*
  int shortEdge = getEdgeIdx(shortEdgeRef), 
    n1 = getNodeIdx(n1ref), n2 = getNodeIdx(n2ref), e1, e2;
  elemRef elem2;
  e1 = n1 - shortEdge + 1;
  e2 = n2 - shortEdge + 1;
  // calculate the above ints
  elem2 = edges[e2].get(myRef);
  edges[e1].update(myRef, elem2);
  if (elem2.idx != -1)  elem2.update(edges[e2], edges[e1]);
  edges[e2].remove();
  myRef.remove();
}
*/
int element::findLongestEdge()
{
  int i, longEdge;
  node n[3];
  double maxlen = 0.0, len[3];

  for (i=0; i<3; i++)  n[i] = C->theNodes[nodes[i]];
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
/*
int element::findShortestEdge()
{
  int i, shortEdge = 0;
  node n[3];
  double minlen, len[3];

  for (i=0; i<3; i++)  n[i] = nodes[i].get();
  // fine lengths of sides
  minlen = len[0] = n[0].distance(n[1]);
  len[1] = n[0].distance(n[2]);
  len[2] = n[1].distance(n[2]);

  for (i=1; i<3; i++) // find min length of a side
    if (len[i] < minlen) {
      shortEdge = i;
      minlen = len[i];
    }
  return shortEdge;
}
*/
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

void element::sanityCheck(chunk *c, elemRef shouldRef) 
{
  CkAssert(myRef == shouldRef);
  CkAssert(C == c);
  for (int i=0;i<3;i++) {
    CkAssert((nodes[i] < C->numNodes) && (nodes[i] > -1));
    CkAssert(!(edges[i].isNull()));
    edges[i].sanityCheck();
  }
}
