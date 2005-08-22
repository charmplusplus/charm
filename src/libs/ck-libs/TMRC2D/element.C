#include "element.h"
#include "tri.h"

#define ZEROAREA 1.0e-15

int myisnan(double x) { return (x!=x); }

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
  double s, perimeter, len[3];
  // fine lengths of sides
  len[0] = (C->theNodes[nodes[0]]).distance(C->theNodes[nodes[1]]);
  len[1] = (C->theNodes[nodes[1]]).distance(C->theNodes[nodes[2]]);
  len[2] = (C->theNodes[nodes[2]]).distance(C->theNodes[nodes[0]]);
  CkAssert(len[0] > 0.0);
  CkAssert(len[1] > 0.0);
  CkAssert(len[2] > 0.0);
  // apply Heron's formula
  perimeter = len[0] + len[1] + len[2];
  s = perimeter / 2.0;
  // cache the result in currentArea
  currentArea = sqrt(s * (s - len[0]) * (s - len[1]) * (s - len[2]));
  if (myisnan(currentArea)) currentArea = 0.0;
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
	int oldothernode;
  int m = -2, nullNbr=0;
  elemRef newElem, nullRef;

  // initializations of shortcuts to affected parts of element
  opnode = (longEdge + 2) % 3;
  othernode = longEdge;
  modEdge = opnode;
  otherEdge = (longEdge + 1) % 3;
  fixnode = otherEdge;

  int fIdx, oIdx;
#ifdef TDEBUG3
  CkPrintf("TMRC2D: [%d] oIdx=%d fIdx=%d ", myRef.cid, nodes[othernode], 
	   nodes[fixnode]);
  CkPrintf("node[oIdx]="); C->theNodes[nodes[othernode]].dump();
  CkPrintf(" node[fIdx]="); C->theNodes[nodes[fixnode]].dump();
  CkPrintf("\n");
#endif

  if (edges[longEdge].cid == myRef.cid) {
    oIdx = nodes[othernode];
    fIdx = nodes[fixnode];
  }
  else {
    FEM_Node *theNodes = &(C->meshPtr->node);
    FEM_Comm_Rec *oNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[othernode]));
    FEM_Comm_Rec *fNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[fixnode]));
    edge e;
    oIdx = oNodeRec->getIdx(e.existsOn(oNodeRec, edges[longEdge].cid));
    fIdx = fNodeRec->getIdx(e.existsOn(fNodeRec, edges[longEdge].cid));
#ifdef TDEBUG3
    CkPrintf("TMRC2D: [%d] oIdx=%d locally, %d on longEdge's chunk\n", 
	     myRef.cid, nodes[othernode], oIdx);
    CkPrintf("TMRC2D: [%d] fIdx=%d locally, %d on longEdge's chunk\n", 
	     myRef.cid, nodes[fixnode], fIdx);
    CkPrintf("oNodeRec: chk=%d idx=%d\n", oNodeRec->getChk(0), oNodeRec->getIdx(0));
    CkPrintf("fNodeRec: chk=%d idx=%d\n", fNodeRec->getChk(0), fNodeRec->getIdx(0));
#endif
  }
  CkAssert(fIdx > -1);
  CkAssert(oIdx > -1);
  CkAssert(oIdx != fIdx);
  
  if ((result=edges[longEdge].split(&m, &e_prime,oIdx, fIdx,
				    myRef, &local, &first, &nullNbr)) == 1) {
    // e_prime successfully created incident on othernode
#ifdef TDEBUG2
    CkPrintf("TMRC2D: [%d] Refining element %d, opnode=%d ^othernode=%d fixnode=%d longEdge=%d modEdge=%d otherEdge=%d\n", myRef.cid, myRef.idx, nodes[opnode], nodes[othernode], nodes[fixnode], edges[longEdge].idx, edges[modEdge].idx, edges[otherEdge].idx);
#endif
    newEdge = C->addEdge(m, nodes[opnode], 0);
#ifdef TDEBUG2
    CkPrintf("TMRC2D: [%d] New edge (%d,%d) added between nodes %d and %d\n", myRef.cid, newEdge.cid, newEdge.idx, m, nodes[opnode]);
#endif
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
    edges[modEdge].update(myRef, newElem, 0);
    C->theEdges[newEdge.idx].update(nullRef, myRef);
    C->theEdges[newEdge.idx].update(nullRef, newElem);
    e_prime.update(nullRef, newElem, 0);
    oldothernode = nodes[othernode];
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
    int b = 0;
    if (nullNbr) b = edges[longEdge].getBoundary();
    if(C->theClient) {
      C->theClient->split(myRef.idx, oldothernode,nodes[fixnode],nodes[opnode], m, newElem.idx, 0.5, flag,b, 0, b);
#ifdef TDEBUG1
      CkPrintf("TMRC2D: [%d] theClient->split(elem=%d, edge=%d, newNode=%d, newElem=%d, bound=%d)\n", myRef.cid, myRef.idx, longEdge, m, newElem.idx, b);
#endif
    }

    if (!first || nullNbr) {
#ifdef TDEBUG3
      if (!first)
	CkPrintf("TMRC2D: [%d] Resetting pending edges, second split complete.\n", myRef.cid);
      else if (nullNbr)
	CkPrintf("TMRC2D: [%d] Resetting pending edges, neighbor NULL.\n", 
		 myRef.cid);
#endif
      edges[longEdge].resetEdge();
    }
  }
  else if (result == 0) { 
    // e_prime already incident on fixnode
#ifdef TDEBUG2
    CkPrintf("TMRC2D: [%d] Refining element %d, opnode=%d othernode=%d ^fixnode=%d longEdge=%d modEdge=%d otherEdge=%d\n", myRef.cid, myRef.idx, nodes[opnode], nodes[othernode], nodes[fixnode], edges[longEdge].idx, edges[modEdge].idx, edges[otherEdge].idx);
#endif
    newEdge = C->addEdge(m, nodes[opnode], 0);
#ifdef TDEBUG2
    CkPrintf("TMRC2D: [%d] New edge (%d,%d) added between nodes %d and %d\n", myRef.cid, newEdge.cid, newEdge.idx, m, nodes[opnode]);
#endif
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
    edges[otherEdge].update(myRef, newElem, 0);
    C->theEdges[newEdge.idx].update(nullRef, myRef);
    C->theEdges[newEdge.idx].update(nullRef, newElem);
    e_prime.update(nullRef, newElem, 0);
		int oldfixnode = nodes[fixnode];
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
    int b = 0;
    if (nullNbr) b = edges[longEdge].getBoundary();
    if (C->theClient) {
      C->theClient->split(myRef.idx,oldfixnode,nodes[othernode],nodes[opnode], m, newElem.idx, 0.5,flag,b, 0, b);
#ifdef TDEBUG1
      CkPrintf("TMRC2D: [%d] theClient->split(elem=%d, edge=%d, newNode=%d, newElem=%d, bound=%d)\n", myRef.cid, myRef.idx, longEdge, m, newElem.idx, b);
#endif
    }
#ifdef TDEBUG3
    CkPrintf("TMRC2D: [%d] Resetting pending edges, second split complete.\n", 
	     myRef.cid);
#endif
    edges[longEdge].resetEdge();
  }
#ifdef TDEBUG2
  else { // longEdge still trying to complete previous split; try later
    // do nothing for now
    CkPrintf("TMRC2D: [%d] Can't bisect element %d, longEdge %d pending\n", 
	     myRef.cid, myRef.idx, edges[longEdge].idx);
  }
#endif
}

void element::coarsen()
{
  int shortEdge = findShortestEdge();
  // check if a different edge from the shortEdge is pending for coarsening
  if (!(edges[shortEdge].isPending(myRef))) {
    if (edges[(shortEdge+1)%3].isPending(myRef)) {
      shortEdge = (shortEdge+1)%3;
    }
    else if (edges[(shortEdge+2)%3].isPending(myRef)) {
      shortEdge = (shortEdge+2)%3;
    }
  }
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
  int opnode, delNode, keepNode, delEdge, keepEdge;
  int kBound, dBound, kFixed, dFixed;
  elemRef nbr, delNbr, keepNbr;
  double frac;

  opnode = (shortEdge+2)%3;  delNode = shortEdge;  keepNode = (shortEdge+1)%3;
  delEdge = opnode;  keepEdge = keepNode;
  // get the boundary flags for the nodes on the edge to collapse
  kBound = C->theNodes[nodes[keepNode]].boundary;
  dBound = C->theNodes[nodes[delNode]].boundary;
  kFixed = C->theNodes[nodes[keepNode]].fixed; 
  dFixed = C->theNodes[nodes[delNode]].fixed;
  keepNbr = edges[keepEdge].getNbr(myRef);
  delNbr = edges[delEdge].getNbr(myRef);
  nbr = edges[shortEdge].getNbr(myRef);

  if (!safeToCoarsen(&nonCoarsenCount, shortEdge, delNbr, keepNbr,nbr)) return;

  // find coords of node to collapse to based on boundary conditions
  node newNode;
  if (!findNewNodeDetails(&newNode, &frac, kBound, dBound, kFixed, dFixed, 
			  &keepNode, &delNode, &nonCoarsenCount, &keepEdge,
			  &delEdge, &keepNbr, &delNbr, &nbr))
    return;

  // translate node ids the shared ids
  int kIdx, dIdx;
  translateNodeIDs(&kIdx, &dIdx, shortEdge, keepNode, delNode);

#ifdef FLIPPREVENT
  if (edges[shortEdge].flipPrevent(myRef, kIdx, dIdx, keepNbr, delNbr, 
				   edges[keepEdge], edges[delEdge], newNode)
      == -1) {
    nonCoarsenCount++;
    return;
  }
#endif

  // collapse the edge; takes care of neighbor element
  present = 0;
  edges[shortEdge].collapse(myRef, kIdx, dIdx, keepNbr, delNbr, 
			    edges[keepEdge], edges[delEdge], newNode, frac);
  present = 1;
  C->removeElement(myRef.idx);     // remove self
}

int element::findLongestEdge()
{
  int i, longEdge;
  double maxlen = 0.0, len[3];
  // find lengths of sides
  len[0] = C->theNodes[nodes[0]].distance(C->theNodes[nodes[1]]);
  len[1] = C->theNodes[nodes[1]].distance(C->theNodes[nodes[2]]);
  len[2] = C->theNodes[nodes[2]].distance(C->theNodes[nodes[0]]);
  for (i=0; i<3; i++) // find max length of a side
    if (len[i] > maxlen) {
      longEdge = i;
      maxlen = len[i];
    }
  CkAssert(longEdge > -1);
  CkAssert(longEdge < 3);
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

double element::getShortestEdge(double *angle)
{
  double minlen, len[3];
  int shortest=0;
  // fine lengths of sides
  minlen = len[0] = C->theNodes[nodes[0]].distance(C->theNodes[nodes[1]]);
  len[1] = C->theNodes[nodes[1]].distance(C->theNodes[nodes[2]]);
  len[2] = C->theNodes[nodes[2]].distance(C->theNodes[nodes[0]]);
  for (int i=1; i<3; i++) // find min length of a side
    if (len[i] < minlen) {
      shortest = i;
      minlen = len[i];
    }
  double A, B, C;
  C = len[shortest];
  A = len[(shortest+1)%3];
  B = len[(shortest+2)%3];
  (*angle) = acos((C*C - A*A - B*B)/(-2*A*B));
  return minlen;
}

double element::getAreaQuality()
{
  double f, q, len[3];
  len[0] = C->theNodes[nodes[0]].distance(C->theNodes[nodes[1]]);
  len[1] = C->theNodes[nodes[1]].distance(C->theNodes[nodes[2]]);
  len[2] = C->theNodes[nodes[2]].distance(C->theNodes[nodes[0]]);
  f = 4.0*sqrt(3.0); //proportionality constant
  q = (f*currentArea)/(len[0]*len[0]+len[1]*len[1]+len[2]*len[2]);  
  return q;
}

double element::getLargestEdge(double *angle)
{
  double maxlen, len[3];
  int largest=0;
  // fine lengths of sides
  maxlen = len[0] = C->theNodes[nodes[0]].distance(C->theNodes[nodes[1]]);
  len[1] = C->theNodes[nodes[1]].distance(C->theNodes[nodes[2]]);
  len[2] = C->theNodes[nodes[2]].distance(C->theNodes[nodes[0]]);
  for (int i=1; i<3; i++) // find max length of a side
    if (len[i] > maxlen) {
      largest = i;
      maxlen = len[i];
    }
  double A, B, C;
  C = len[largest];
  A = len[(largest+1)%3];
  B = len[(largest+2)%3];
  (*angle) = acos((C*C - A*A - B*B)/(-2*A*B));
  return maxlen;
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
    if (edges[i].cid == myRef.cid)
      C->theEdges[edges[i].idx].sanityCheck(nodes[i],nodes[(i+1)%3],myRef.idx);
  }
  CkAssert(nodes[0] != nodes[1]);
  CkAssert(nodes[0] != nodes[2]);
  CkAssert(nodes[2] != nodes[1]);
}

void element::incnonCoarsen() {
  nonCoarsenCount++;
  return;
}

bool element::flipTest(node* oldnode, node* newnode) {
  //oldnode is C->thenodes[nodes[delnode]]  
}

bool element::flipInverseTest(node* oldnode, node* newnode) {
  double x0,x1,x2,x3,y0,y1,y2,y3;
  int i=0;

  for(i=0; i<3; i++) {
    if( oldnode->X() == C->theNodes[nodes[i]].X() && oldnode->Y() == C->theNodes[nodes[i]].Y() ) {
      break;
    }
  }
  x0 = C->theNodes[nodes[(i+1)%3]].X();
  x1 = oldnode->X();
  x2 = C->theNodes[nodes[(i+2)%3]].X();
  x3 = newnode->X();
  y0 = C->theNodes[nodes[(i+1)%3]].Y();
  y1 = oldnode->Y();
  y2 = C->theNodes[nodes[(i+2)%3]].Y();
  y3 = newnode->Y();

  //vector-product (axby - aybx)
  double res1 = (x0-x1)*(y2-y1) - (y0-y1)*(x2-x1);
  double res2 = (x0-x3)*(y2-y3) - (y0-y3)*(x2-x3);

  //zero area is sometimes giving bad results because zero can be represented as a 
  //negative small number or a positive small number
  if((res1>0 && res2>0)||(res1<=0 && res2<=0)) return false;
  else if(/*fabs(res1) < ZEROAREA || */fabs(res2) < ZEROAREA) {
    CkPrintf("Zero area: Chunk %d Elem %d (%lf,%lf);(%lf,%lf)--(%lf,%lf)--(%lf,%lf)\n",myRef.cid,myRef.idx,x0,y0,x2,y2,x1,y1,x3,y3);
    return true;
  }
  else {
    CkPrintf("Flip: Chunk %d Elem %d -- (%lf,%lf);(%lf,%lf)--(%lf,%lf)--(%lf,%lf)\n",myRef.cid,myRef.idx,x0,y0,x2,y2,x1,y1,x3,y3);
    return true; //the two vector products are opposite in sign, so there is a flip
  }
}


int element::findNewNodeDetails(node *newNode, double *frac, int kBc, int dBc,
				int kFx, int dFx, int *kNd, int *dNd, 
				short *nonCC, int *kEg, int *dEg, 
				elemRef *kNbr, elemRef *dNbr, elemRef *nbr)
{
  int tmpMap;
  elemRef tmpRef;
  if ((kBc == 0) && (dBc == 0)) { // both interior; collapse to midpoint
    if (!kFx && !dFx) {
      (*newNode)=C->theNodes[nodes[(*kNd)]].midpoint(C->theNodes[nodes[(*dNd)]]);
      (*newNode).boundary = 0;
      (*frac) = 0.5;
      return 1;
    }
    else if (dFx && kFx)  (*nonCC)++;
    else if (dFx) {
      (*newNode) = C->theNodes[nodes[(*dNd)]];
      tmpMap = (*dNd); (*dNd) = (*kNd); (*kNd) = tmpMap;
      tmpMap = (*dEg); (*dEg) = (*kEg); (*kEg) = tmpMap;
      tmpRef = (*dNbr); (*dNbr) = (*kNbr); (*kNbr) = tmpRef;
      (*frac) = 1.0;
      return 1;
    }
    else {
      (*newNode) = C->theNodes[nodes[(*kNd)]];
      (*frac) = 1.0;
      return 1;
    }
  }
  else if ((kBc == 0) || (dBc == 0)) { // only one on boundary
    // collapse edge to boundary node
    if (kBc && !dFx) {
      (*newNode) = C->theNodes[nodes[(*kNd)]];
      (*frac) = 1.0;
      return 1;
    }
    else if (dBc && !kFx) {
      (*newNode) = C->theNodes[nodes[(*dNd)]];
      tmpMap = (*dNd); (*dNd) = (*kNd); (*kNd) = tmpMap;
      tmpMap = (*dEg); (*dEg) = (*kEg); (*kEg) = tmpMap;
      tmpRef = (*dNbr); (*dNbr) = (*kNbr); (*kNbr) = tmpRef;
      (*frac) = 1.0;
      return 1;
    }
    else (*nonCC)++;
  }
  else if (kBc == dBc) { // both on same boundary
    if (nbr->cid >= 0) (*nonCC)++; // edge is internal; don't coarsen
    // check fixed status of both nodes
    else if (kFx && dFx) (*nonCC)++;  // if both fixeds don't coarsen
    else if (kFx || dFx) { // if one fixed, collapse edge to fixed
      if (kFx) {
	(*newNode) = C->theNodes[nodes[(*kNd)]];
	(*frac) = 1.0;
	return 1;
      }
      else {
	(*newNode) = C->theNodes[nodes[(*dNd)]];
	tmpMap = (*dNd); (*dNd) = (*kNd); (*kNd) = tmpMap;
	tmpMap = (*dEg); (*dEg) = (*kEg); (*kEg) = tmpMap;
	tmpRef = (*dNbr); (*dNbr) = (*kNbr); (*kNbr) = tmpRef;
	(*frac) = 1.0;
	return 1;
      }
    }
    else { // neither are fixeds, collapse edge to midpoint
      (*newNode)=C->theNodes[nodes[(*kNd)]].midpoint(C->theNodes[nodes[(*dNd)]]);
      (*newNode).boundary = kBc;
      (*frac) = 0.5;
      return 1;
    }
  }
  else { // nodes on different boundary
    if (nbr->cid >= 0) (*nonCC)++; // edge is internal; don't coarsen
    else { // if it isn't check if lower boundary node is a fixed
      if (dBc > kBc) { // dBc is numbered higher
	if (kFx) (*nonCC)++;  // if it is, don't coarsen
	else { // if it isn't, collapse edge to larger boundary node
	  (*newNode) = C->theNodes[nodes[(*dNd)]];
	  tmpMap = (*dNd); (*dNd) = (*kNd); (*kNd) = tmpMap;
	  tmpMap = (*dEg); (*dEg) = (*kEg); (*kEg) = tmpMap;
	  tmpRef = (*dNbr); (*dNbr) = (*kNbr); (*kNbr) = tmpRef;
	  (*frac) = 1.0;
	  return 1;
	}
      }
      else { // kBc is numbered higher
	if (dFx) (*nonCC)++;  // if it is, don't coarsen
	else { // if it isn't, collapse edge to larger boundary node
	  (*newNode) = C->theNodes[nodes[(*kNd)]];
	  (*frac) = 1.0;
	  return 1;
	}
      }
    }
  }
  return 0;
}

void element::translateNodeIDs(int *kIdx, int *dIdx, int sEg, int kNd, int dNd)
{
  if (edges[sEg].cid == myRef.cid) {
    (*kIdx) = nodes[kNd];
    (*dIdx) = nodes[dNd];
  }
  else {
    FEM_Node *theNodes = &(C->meshPtr->node);
    FEM_Comm_Rec *dNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[dNd]));
    FEM_Comm_Rec *kNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nodes[kNd]));
    edge e;
    (*kIdx) = kNodeRec->getIdx(e.existsOn(kNodeRec, edges[sEg].cid));
    (*dIdx) = dNodeRec->getIdx(e.existsOn(dNodeRec, edges[sEg].cid));
  }
}

int element::safeToCoarsen(short *nonCC, int sEg, elemRef dNbr, elemRef kNbr,
			   elemRef nbr)
{
  if (!edges[sEg].isPending(myRef)) {
    if ((dNbr == kNbr) && (dNbr.cid != -1)) { // wackiness has ensued
      (*nonCC)++;
      return 0;
    }
    else if ((dNbr.cid != -1) && (kNbr.cid != -1) && 
	     (neighboring(dNbr, kNbr))) {
      (*nonCC)++;
      return 0;
    }
    else if (nbr.cid != -1) {
      intMsg *im;
      im = mesh[nbr.cid].safeToCoarsen(nbr.idx, edges[sEg]);
      if (im->anInt == 0) (*nonCC)++;
      return im->anInt;
    }
  }
  return 1;
}

int element::safeToCoarsen(edgeRef ser) {
  elemRef nbr1, nbr2;
  for (int i=0; i<3; i++) {
    if (edges[i] == ser) {
      nbr1 = edges[(i+1)%3].getNbr(myRef);
      nbr2 = edges[(i+2)%3].getNbr(myRef);
      break;
    }
  }
  if (nbr1 == nbr2) return 0;
  if ((nbr1.cid == -1) || (nbr2.cid == -1)) return 1;
  return (!neighboring(nbr1, nbr2));
}

int element::neighboring(elemRef e1, elemRef e2) {
  intMsg *im;
  if ((e1.cid == -1) || (e2.cid == -1)) return 0;
  im = mesh[e1.cid].neighboring(e1.idx, e2);
  return im->anInt;
}

int element::neighboring(elemRef e) {
  return ((edges[0].getNbr(myRef) == e) || (edges[1].getNbr(myRef) == e) || 
	  (edges[2].getNbr(myRef) == e));
}
