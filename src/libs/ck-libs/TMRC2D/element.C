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
  int shortEdge = findShortestEdge();
  CkPrintf("TMRC2D: Coarsen element %d\n", myRef.idx);
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
  elemRef keepNbr, delNbr;
  int local, first, flag;

  if (edges[shortEdge].isPending(myRef)) {
  }
  else if (edges[(shortEdge+1)%3].isPending(myRef)) {
    shortEdge = (shortEdge+1)%3;
  }
  else if (edges[(shortEdge+2)%3].isPending(myRef)) {
    shortEdge = (shortEdge+2)%3;
  }

  opnode = (shortEdge + 2) % 3;
  delNode = shortEdge;
  delEdge = opnode;
  keepEdge = (shortEdge + 1) % 3;
  keepNode = keepEdge;
  keepNbr = edges[keepEdge].getNbr(myRef);
  delNbr = edges[delEdge].getNbr(myRef);
  
  double length = 
    C->theNodes[nodes[keepNode]].distance(C->theNodes[nodes[delNode]]);
  CkPrintf("TMRC2D: LOCKing opnode=%d\n", nodes[opnode]);
  int aResult = nodeLockup(C->theNodes[nodes[opnode]], edges[keepEdge], edges[keepEdge], keepNbr, length);
  if (aResult == 0) return;
  if ((aResult == -1) && (keepNbr.cid != -1)) {
    intMsg *im = mesh[keepNbr.cid].nodeLockup(keepNbr.idx, C->theNodes[nodes[opnode]], edges[keepEdge], edges[keepEdge], delNbr, length);
    if (im->anInt == 0) {
      int junkResult = nodeUpdate(C->theNodes[nodes[opnode]],edges[keepEdge],keepNbr,C->theNodes[nodes[opnode]]);
      return;
    }
  }

  node newNode;
  result = edges[shortEdge].collapse(myRef, C->theNodes[nodes[keepNode]],
				     C->theNodes[nodes[delNode]], keepNbr,
				     delNbr, edges[keepEdge], edges[delEdge], 
				     C->theNodes[nodes[opnode]], 
				     &local, &first);
  if (result == 1) {
    // collapse successful; keepNode is node to keep
    // tell delNbr to replace delEdge with keepEdge
    newNode = 
      C->theNodes[nodes[keepNode]].midpoint(C->theNodes[nodes[delNode]]);
    CkPrintf("In collapse[%d](a) shortEdge=%d delEdge=%d keepEdge=%d opnode=%d delNode=%d keepNode=%d delNbr=%d keepNbr=%d\n", myRef.idx, edges[shortEdge].idx, edges[delEdge].idx, edges[keepEdge].idx, nodes[opnode], nodes[delNode], nodes[keepNode], delNbr.idx, keepNbr.idx);
    if (delNbr.cid != -1)
      mesh[delNbr.cid].updateElementEdge(delNbr.idx, edges[delEdge], 
					 edges[keepEdge]);
    // tell keepEdge to replace myRef with delNbr
    edges[keepEdge].update(myRef, delNbr);
    // remove self
    C->removeElement(myRef.idx);
    // remove delEdge
    edges[delEdge].remove();
    // edge[shortEdge] handles removal of delNode and shortEdge, as well as
    // update of keepNode
    if (local && first) flag = LOCAL_FIRST;
    if (local && !first) flag = LOCAL_SECOND;
    if (!local && first) flag = BOUND_FIRST;
    if (!local && !first) flag = BOUND_SECOND;
    C->theClient->collapse(myRef.idx, shortEdge, keepNode, newNode.X(), 
			   newNode.Y(), flag);
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
    newNode = 
      C->theNodes[nodes[keepNode]].midpoint(C->theNodes[nodes[delNode]]);
    CkPrintf("In collapse[%d](b) shortEdge=%d delEdge=%d keepEdge=%d opnode=%d delNode=%d keepNode=%d delNbr=%d keepNbr=%d\n", myRef.idx, edges[shortEdge].idx, edges[delEdge].idx, edges[keepEdge].idx, nodes[opnode], nodes[delNode], nodes[keepNode], delNbr.idx, keepNbr.idx);
    if (delNbr.cid != -1)
      mesh[delNbr.cid].updateElementEdge(delNbr.idx, edges[delEdge], 
					 edges[keepEdge]);
    // tell keepEdge to replace myRef with delNbr
    edges[keepEdge].update(myRef, delNbr);
    // remove self
    C->removeElement(myRef.idx);
    // remove delEdge
    edges[delEdge].remove();
    // edge[shortEdge] handles removal of delNode and shortEdge, as well as
    // update of keepNode
    if (local && first) flag = LOCAL_FIRST;
    if (local && !first) flag = LOCAL_SECOND;
    if (!local && first) flag = BOUND_FIRST;
    if (!local && !first) flag = BOUND_SECOND;
    C->theClient->collapse(myRef.idx, shortEdge, keepNode, newNode.X(), 
			   newNode.Y(), flag);
  }
}

int element::nodeLockup(node n, edgeRef from, edgeRef start, elemRef end, 
			double l)
{
  int nIdx, fIdx, nextIdx;
  for (int i=0; i<3; i++) {
    if (n == C->theNodes[nodes[i]]) nIdx = i;
    if (from == edges[i]) fIdx = i;
  }
  CkAssert((nIdx > -1) && (nIdx < 3));
  CkAssert((fIdx > -1) && (fIdx < 3));
  int lockResult = C->theNodes[nodes[nIdx]].lock(l, start);
  if (!lockResult) return 0;
  if (myRef == end) return 1;
  if (nIdx == fIdx) nextIdx = (nIdx + 2) % 3;
  else nextIdx = nIdx;
  //CkPrintf("TMRC2D: In element[%d]::nodeLockup: from=%d nodes[nIdx]=%d fIdx=%d nextIdx=%d\n", myRef.idx, from.idx, nodes[nIdx], fIdx, nextIdx);
  edgeRef nextRef = edges[nextIdx];
  intMsg *im = mesh[nextRef.cid].nodeLockupER(nextRef.idx, n, start, myRef, 
					      end, l);
  if (im->anInt == 0) C->theNodes[nodes[nIdx]].unlock();
  return im->anInt;
}

int element::nodeUpdate(node n, edgeRef from, elemRef end, node newNode)
{
  int nIdx, fIdx, nextIdx;
  for (int i=0; i<3; i++) {
    if (n == C->theNodes[nodes[i]]) nIdx = i;
    else if (newNode == C->theNodes[nodes[i]]) nIdx = i;
    if (from == edges[i]) fIdx = i;
  }
  CkAssert((nIdx > -1) && (nIdx < 3));
  CkAssert((fIdx > -1) && (fIdx < 3));
  C->theNodes[nodes[nIdx]].unlock();
  CkPrintf("TMRC2D: about to set node %d to [%f,%f]\n", nodes[nIdx], newNode.X(), 
	   newNode.Y());
  C->theNodes[nodes[nIdx]].set(newNode.X(), newNode.Y());
  if (myRef == end) return 1;
  if (nIdx == fIdx) nextIdx = (nIdx + 2) % 3;
  else nextIdx = nIdx;
  //CkPrintf("TMRC2D: In element[%d]::nodeUpdate: from=%d nodes[nIdx]=%d fIdx=%d nextIdx=%d\n", myRef.idx, from.idx, nodes[nIdx], fIdx, nextIdx);
  edgeRef nextRef = edges[nextIdx];
  intMsg *im = mesh[nextRef.cid].nodeUpdateER(nextRef.idx, n, myRef, end, 
					      newNode);
  return im->anInt;
}

int element::nodeDelete(node n, edgeRef from, elemRef end, node ndReplace)
{
  int nIdx = -1, fIdx, nextIdx;
  for (int i=0; i<3; i++) {
    if (n == C->theNodes[nodes[i]]) nIdx = i;
    if (from == edges[i]) fIdx = i;
  }
  int found = 0;
  for (int j=0; j<C->nodeSlots; j++)
    if (C->theNodes[j] == ndReplace) {
      if (nIdx == -1) {
	if (nodes[0] == j) nIdx = 0;
	else if (nodes[1] == j) nIdx = 1;
	else if (nodes[2] == j) nIdx = 2;
      }
      CkAssert((nIdx > -1) && (nIdx < 3));
      found = 1;
      //CkPrintf("TMRC2D: about to remove node %d\n", nodes[nIdx]);
      C->removeNode(nodes[nIdx]);
      nodes[nIdx] = j;
      break;
    }
  CkAssert((nIdx > -1) && (nIdx < 3));      
  CkAssert((fIdx > -1) && (fIdx < 3));
  C->theNodes[nodes[nIdx]].unlock();
  if (!found) {
    //CkPrintf("TMRC2D: about to replace node %d\n", nodes[nIdx]);
    C->theNodes[nodes[nIdx]] = ndReplace;
    C->theNodes[nodes[nIdx]].present = 1;
  }
  if (myRef == end) return 1;
  if (nIdx == fIdx) nextIdx = (nIdx + 2) % 3;
  else nextIdx = nIdx;
  //CkPrintf("TMRC2D: In element[%d]::nodeDelete: from=%d nodes[nIdx]=%d fIdx=%d nextIdx=%d\n", myRef.idx, from.idx, nodes[nIdx], fIdx, nextIdx);
  edgeRef nextRef = edges[nextIdx];
  intMsg *im = mesh[nextRef.cid].nodeDeleteER(nextRef.idx, n, myRef, end, ndReplace);
  return im->anInt;
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
