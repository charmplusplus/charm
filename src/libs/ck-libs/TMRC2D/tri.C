// Triangular Mesh Refinement Framework - 2D (TMR)
// Created by: Terry L. Wilmarth

#include <stdlib.h>
#include <stdio.h>
#include "tri.h"

// I will keep this code until I die!
//#define accessLock() {printf("%s : %d calling accessLock \n",__FILE__,__LINE__); accessLock1();}
//readonlys
CProxy_chunk mesh;
CtvDeclare(chunk *, _refineChunk);

void refineChunkInit(void) {
  CtvInitialize(chunk *, _refineChunk);
}

chunk::chunk(chunkMsg *m)
  : TCharmClient1D(m->myThreads), sizeElements(0), sizeEdges(0), sizeNodes(0),
    firstFreeElement(0), firstFreeEdge(0), firstFreeNode(0),
    edgesSent(0), edgesRecvd(0), first(0),
    coarsenElements(NULL), refineElements(NULL), refineStack(NULL),
    refineHeapSize(0), coarsenHeapSize(0), refineTop(0),
    additions(0), debug_counter(0), refineInProgress(0), coarsenInProgress(0),
    meshLock(0), meshExpandFlag(0), 
    numElements(0), numEdges(0), numNodes(0), numGhosts(0), theClient(NULL),
    elementSlots(0), edgeSlots(0), nodeSlots(0), lock(0), lockCount(0),
    lockHolderIdx(-1), lockHolderCid(-1), lockPrio(-1.0), lockList(NULL)
{
  refineResultsStorage=NULL;
  cid = thisIndex;
  numChunks = m->nChunks;
  CkFreeMsg(m);
  tcharmClientInit();
  thread->resume();
}

void chunk::addRemoteEdge(int elem, int localEdge, edgeRef er)
{
  accessLock();
  CkAssert(localEdge >=0);
  CkAssert(localEdge < 3);
  CkAssert(er.cid > -1);
  CkAssert(er.idx > -1);
  if ((theElements[elem].edges[localEdge].cid > -1) &&
      (theElements[elem].edges[localEdge].idx > -1)){
    CkPrintf("TMRC2D: [%d] WARNING: addRemoteEdge replacing non-null edgeRef!\n", cid);
  }	
  theElements[elem].set(localEdge, er);
#ifdef TDEBUG3
  CkPrintf("TMRC2D: [%d] addRemoteEdge on element %d", cid, elem);
#endif
  edgesRecvd++;
  releaseLock();
}

void chunk::refineElement(int idx, double area)
{ // Reduce element's targetArea to indicate need for refinement
  if (!theElements[idx].isPresent()) return;
  accessLock();
  theElements[idx].resetTargetArea(area);
  refineStack[refineTop].elID = idx;
  refineStack[refineTop].len = -1.0;
  refineTop++;
  int pos = refineHeapSize;
  while (pos >= 1) {
    if (refineElements[pos].elID == idx)
      refineElements[pos].elID = -1;
    pos--;
  }
  releaseLock();
  if (!refineInProgress) { // if refine loop not running
    refineInProgress = 1;
    mesh[cid].refiningElements(); // start it up
  }
}

void chunk::refiningElements()
{
  int i;
  while (refineHeapSize>0 || refineTop > 0) { // loop through the elements
    if (refineTop>0) {
      refineTop--;
      i=refineStack[refineTop].elID;
    }
    else
      i=Delete_Min(0);
    if ((i != -1) && theElements[i].isPresent()) {
      // element i has a lower target area -- needs to refine
      theElements[i].refine(); // refine the element
      adjustMesh();
    }
    CthYield(); // give other chunks on the same PE a chance
  }
  sanityCheck(); // quietly make sure mesh is in shape
  refineInProgress = 0; // nothing needs refinement; turn refine loop off
  //    if (CkMyPe() == 0) for (int j=0; j<5; j++) mesh[j].print();
}


void chunk::coarsenElement(int idx, double area)
{ 
  if (!theElements[idx].isPresent()) return;
  double theArea;
  accessLock();
  theArea = theElements[idx].getArea();
  if (area > theElements[idx].getTargetArea())
    theElements[idx].resetTargetArea(area);
  Insert(idx, -1.0, 1);
  int pos = coarsenHeapSize;
  while (pos > 1) {
    if (coarsenElements[pos].elID == idx)
      coarsenElements[pos].elID = -1;
    pos--;
  }
  releaseLock();
  if (!coarsenInProgress) { // if coarsen loop not running
    coarsenInProgress = 1;
    mesh[cid].coarseningElements(); // start it up
  }
}

void chunk::coarseningElements()
{
  int i;
  double area, qFactor, targetArea;
  while (coarsenHeapSize > 0) { // loop through the elements
    i=Delete_Min(1);
    CkPrintf("Loop at element %d\n", i);
    if ((i != -1) && theElements[i].isPresent()) {
      CkPrintf("Checking element %d\n", i);
      area = theElements[i].getArea();
      targetArea = theElements[i].getTargetArea();
      qFactor = theElements[i].getAreaQuality();
      if ((theElements[i].getTargetArea()*COARSEN_PRECISION > area) || 
	  (area == 0.0) || (qFactor < QUALITY_MIN)) { 
	CkPrintf("Coarsening element %d\n", i);
	theElements[i].coarsen(); // coarsen the element
      }
    }
    CthYield(); // give other chunks on the same PE a chance
  }
  coarsenInProgress = 0;  // turn coarsen loop off
  if (coarsenElements) delete [] coarsenElements;
  coarsenElements = new elemHeap[numElements+1];
  coarsenElements[0].elID=-1;
  coarsenElements[0].len=-2.0;
  for (i=0; i<elementSlots; i++) {
    if (theElements[i].isPresent() && (theElements[i].nonCoarsenCount<1)) {
      area = theElements[i].getArea();
      if (area == 0.0) {
	CkPrintf("Element[%d] has area %1.10e!\n", i, area);
      }
      targetArea = theElements[i].getTargetArea();
      double shortEdgeLen, angle;
      shortEdgeLen=theElements[i].getShortestEdge(&angle);
      qFactor = theElements[i].getAreaQuality();
      if ((targetArea*COARSEN_PRECISION > area) || (qFactor < QUALITY_MIN)) {
	CkPrintf("Element[%d] has area %1.10e target %1.10e qFactor %1.10e\n", i, area, targetArea, qFactor);
	Insert(i, shortEdgeLen*qFactor, 1);
      }
    }
  }
  sanityCheck(); // quietly make sure mesh is in shape
  if ((coarsenHeapSize>0) && !coarsenInProgress) {
    coarsenInProgress = 1;
    mesh[cid].coarseningElements();
  }
}

// many remote access methods follow
intMsg *chunk::safeToMoveNode(int idx, double x, double y)
{
  node foo(x, y);
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theNodes[idx].safeToMove(foo);
  releaseLock();
  return im;
}

splitOutMsg *chunk::split(int idx, elemRef e, int oIdx, int fIdx)
{
  splitOutMsg *som = new splitOutMsg;
  accessLock();
  som->result = theEdges[idx].split(&(som->n), &(som->e), oIdx, fIdx, e, 
				    &(som->local), &(som->first), 
				    &(som->nullNbr));
  releaseLock();
  return som;
}

void chunk::collapse(int idx, elemRef e, int kIdx, int dIdx, elemRef kNbr, 
		     elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node newN, 
		     double frac)
{
  accessLock();
  theEdges[idx].collapse(e, kIdx, dIdx, kNbr, dNbr, kEdge, dEdge, newN, frac);
  releaseLock();
}

splitOutMsg *chunk::flipPreventE(int idx, elemRef e, int kIdx, int dIdx,
			     elemRef kNbr, elemRef dNbr, edgeRef kEdge, 
			     edgeRef dEdge, node newN)
{
  splitOutMsg *som = new splitOutMsg;
  accessLock();
  som->result = theEdges[idx].flipPrevent(e, kIdx, dIdx, kNbr, dNbr, kEdge, 
					  dEdge, newN);
  releaseLock();
  return som;
}

void chunk::nodeReplaceDelete(int kIdx, int dIdx, node nn, int shared, 
			      int *chk, int *idx)
{
  int *jChk, *jIdx;
  int jShared;
  accessLock();
  if (dIdx == -1) { 
    if (kIdx != -1) {
      theNodes[kIdx].set(nn.X(), nn.Y());
      theNodes[kIdx].boundary = nn.boundary;
      theNodes[kIdx].fixed = nn.fixed;
      jShared = joinCommLists(kIdx, shared, chk, idx, jChk, jIdx);
      theClient->nodeUpdate(kIdx, nn.X(), nn.Y(), nn.boundary, jShared, jChk, 
			    jIdx);
#ifdef TDEBUG1
      CkPrintf("TMRC2D: [%d] (a)theClient->nodeUpdate(%d, %2.10f, %2.10f, %d)\n", cid, kIdx, nn.X(), nn.Y(), nn.boundary);
#endif
    }
    return;
  }
  else if (kIdx == -1) {
    theNodes[dIdx].set(nn.X(), nn.Y());
    theNodes[dIdx].boundary = nn.boundary;
    theNodes[dIdx].fixed = nn.fixed;
    jShared = joinCommLists(dIdx, shared, chk, idx, jChk, jIdx);
    theClient->nodeUpdate(dIdx, nn.X(), nn.Y(), nn.boundary, jShared, jChk, 
			  jIdx);
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] (b)theClient->nodeUpdate(%d, %2.10f, %2.10f, %d)\n", cid, dIdx, nn.X(), nn.Y(), nn.boundary);
#endif
  }
  else {
    removeNode(dIdx);
    theNodes[kIdx].set(nn.X(), nn.Y());
    theNodes[kIdx].boundary = nn.boundary;
    theNodes[kIdx].fixed = nn.fixed;
    jShared = joinCommLists(kIdx, shared, chk, idx, jChk, jIdx);
    theClient->nodeUpdate(kIdx, nn.X(), nn.Y(), nn.boundary, jShared, jChk, 
			  jIdx);
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] (c)theClient->nodeUpdate(%d, %2.10f, %2.10f, %d)\n", cid, kIdx, nn.X(), nn.Y(), nn.boundary);
#endif
    for (int j=0; j<elementSlots; j++) {
      if (theElements[j].isPresent()) {
	CkAssert(theElements[j].nodes[0] != theElements[j].nodes[1]);
	CkAssert(theElements[j].nodes[0] != theElements[j].nodes[2]);
	CkAssert(theElements[j].nodes[2] != theElements[j].nodes[1]);
	for (int k=0; k<3; k++) {
	  if (theElements[j].nodes[k] == dIdx) {
#ifdef FLIPTEST
	    if(theElements[j].nodes[(k+1)%3] == kIdx || theElements[j].nodes[(k+2)%3] == kIdx) break; //don't worry abt this element, it will be deleted
	    if(theElements[j].flipInverseTest(&(theNodes[dIdx]),&nn)) {
	      //CkPrintf("Chunk %d, Elem %d is flipping!!\n",cid,j);
	    }
#endif
	    theElements[j].nonCoarsenCount = 0;
	    theElements[j].nodes[k] = kIdx;
	    theClient->nodeReplaceDelete(j, k, dIdx, kIdx);
#ifdef TDEBUG1
	    CkPrintf("TMRC2D: [%d] theClient->nodeReplaceDelete(%d, %d, %d, %d)\n", cid, j, k, dIdx, kIdx);
#endif
	  }
	  if (theElements[j].nodes[k] == kIdx) {
	    theElements[j].nonCoarsenCount = 0;
	  }
	}
	CkAssert(theElements[j].nodes[0] != theElements[j].nodes[1]);
	CkAssert(theElements[j].nodes[0] != theElements[j].nodes[2]);
	CkAssert(theElements[j].nodes[2] != theElements[j].nodes[1]);
      }
    }
    for (int j=0; j<edgeSlots; j++) {
      if (theEdges[j].isPresent()) {
	for (int k=0; k<2; k++) {
	  if (theEdges[j].nodes[k] == dIdx) {
	    theEdges[j].nodes[k] = kIdx;
#ifdef TDEBUG1
	    CkPrintf("TMRC2D: [%d] Edge %d node %d replaced with %d\n", cid, j,
		     dIdx, kIdx);
#endif
	  }
	}
      }
    }
  }
  releaseLock();
}

boolMsg *chunk::flipPrevent(int kIdx, int dIdx, node nn, int shared, int *chk, int *idx)
{
  boolMsg *bm = new boolMsg;
  bm->aBool = false;
  for (int j=0; j<elementSlots; j++) {
    if (theElements[j].isPresent()) {
      CkAssert(theElements[j].nodes[0] != theElements[j].nodes[1]);
      CkAssert(theElements[j].nodes[1] != theElements[j].nodes[2]);
      CkAssert(theElements[j].nodes[2] != theElements[j].nodes[0]);
      for (int k=0; k<3; k++) {
	if (theElements[j].nodes[k] == dIdx) {
	  if(theElements[j].nodes[(k+1)%3] == kIdx || theElements[j].nodes[(k+2)%3] == kIdx) break; //don't worry abt this element, it will be deleted
	  if(theElements[j].flipInverseTest(&(theNodes[dIdx]),&nn)) {
	    //CkPrintf("Chunk %d, Elem %d is flipping!!\n",cid,j);
	    bm->aBool = true;
	    //mesh[*chk].incnonCoarsen(*idx);
	    return bm;
	  }
	}
	if (theElements[j].nodes[k] == kIdx) {
	  if(theElements[j].nodes[(k+1)%3] == dIdx || theElements[j].nodes[(k+2)%3] == dIdx) break; //don't worry abt this element, it will be deleted
	  if(theElements[j].flipInverseTest(&(theNodes[kIdx]),&nn)) {
	    //CkPrintf("Chunk %d, Elem %d is flipping!!\n",cid,j);
	    bm->aBool = true;
	    //mesh[*chk].incnonCoarsen(*idx);
	    return bm;
	  }
	}
      }
    }
  }
  return bm;
}

void chunk::incnonCoarsen(int idx) {
  accessLock();
  theElements[idx].incnonCoarsen();
  releaseLock();
  return;
}

void chunk::resetnonCoarsen(int idx) {
  accessLock();
  theElements[idx].resetnonCoarsen();
  releaseLock();
  return;
}

intMsg *chunk::isPending(int idx, objRef e)
{
  intMsg *im = new intMsg;
  elemRef eR(e.cid, e.idx);
  accessLock();
  im->anInt = theEdges[idx].isPending(eR);
  releaseLock();
  return im;
}

void chunk::checkPending(int idx, objRef aRef)
{
  elemRef eRef;
  eRef.idx = aRef.idx; eRef.cid = aRef.cid;
  accessLock();
  theEdges[idx].checkPending(eRef);
  releaseLock();
}

void chunk::checkPending(int idx, objRef aRef1, objRef aRef2)
{
  elemRef eRef1, eRef2;
  eRef1.idx = aRef1.idx; eRef1.cid = aRef1.cid;
  eRef2.idx = aRef2.idx; eRef2.cid = aRef2.cid;
  accessLock();
  theEdges[idx].checkPending(eRef1, eRef2);
  releaseLock();
}

void chunk::updateElement(int idx, objRef oldval, objRef newval, int b)
{
  elemRef ov, nv;
  ov.idx = oldval.idx;   ov.cid = oldval.cid; 
  nv.idx = newval.idx;   nv.cid = newval.cid; 
  accessLock();
  theEdges[idx].update(ov, nv);
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] Update edge %d replaced element %d with %d\n", cid,
	     idx, ov.idx, nv.idx);
#endif
  if (theEdges[idx].getBoundary() < b)
    theEdges[idx].setBoundary(b);
  if ((theEdges[idx].getBoundary() > 0) && (b > 0))
    CkPrintf("TMRC2D: [%d] WARNING! chunk::updateElement collapsed two different boundaries together on one edge.\n", cid);
  releaseLock();
}

void chunk::updateElementEdge(int idx, objRef oldval, objRef newval)
{
  edgeRef ov, nv;
  ov.idx = oldval.idx;   ov.cid = oldval.cid; 
  nv.idx = newval.idx;   nv.cid = newval.cid; 
  accessLock();
  theElements[idx].update(ov, nv);
  releaseLock();
}

void chunk::updateReferences(int idx, objRef oldval, objRef newval)
{
  CkPrintf("TMRC2D: [%d] WARNING! chunk::updateReferences called but not implemented!\n", cid);
  /*
  int i;
  nodeRef ov, nv;
  ov.idx = oldval.idx;   ov.cid = oldval.cid; 
  nv.idx = newval.idx;   nv.cid = newval.cid;
  for (i=0; i<numElements; i++)
    theElements[i].update(ov, nv);
  for (i=0; i<numEdges; i++)
    theEdges[i].updateSilent(ov, nv);
  */
}

doubleMsg *chunk::getArea(int n)
{
  doubleMsg *dm = new doubleMsg;
  accessLock();
  dm->aDouble = theElements[n].getArea();
  releaseLock();
  return dm;
}

void chunk::resetEdge(int n)
{
  accessLock();
  theEdges[n].reset();
  releaseLock();
}

refMsg *chunk::getNbr(int idx, objRef aRef)
{
  refMsg *rm = new refMsg;
  elemRef er, ar;
  ar.cid = aRef.cid; ar.idx = aRef.idx;
  accessLock();
  er = theEdges[idx].getNot(ar);
  releaseLock();
  rm->aRef = er;
  return rm;
}

void chunk::setTargetArea(int idx, double aDouble)
{
  accessLock();
  theElements[idx].setTargetArea(aDouble);
  releaseLock();
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

void chunk::resetTargetArea(int idx, double aDouble)
{
  accessLock();
  theElements[idx].resetTargetArea(aDouble);
  releaseLock();
}

void chunk::reportPos(int idx, double x, double y)
{
  node z(x, y);
  accessLock();
  theNodes[idx].reportPos(z);
  releaseLock();
}

// the following methods are for run-time additions and modifications
// to the chunk components
void chunk::accessLock()
{
  while (meshExpandFlag && (meshLock==0))
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

void chunk::allocMesh(int nEl)
{
  int i;
  sizeElements = nEl * 2;
  sizeNodes = sizeEdges = sizeElements * 3;
  elementSlots = nEl;
  firstFreeElement = nEl;
  theElements.resize(sizeElements);
  theNodes.resize(sizeNodes);
  theEdges.resize(sizeEdges);
  for (i=0; i<sizeElements; i++)
    theElements[i].set(); 
  for (i=0; i<sizeNodes; i++) {
    theNodes[i].present = 0; 
    theEdges[i].present = 0; 
  }
  conn = new int[3*numGhosts];
  gid = new int[2*numGhosts];
}

void chunk::adjustMesh()
{
  int i;
  if (sizeElements <= elementSlots+100) {
    adjustFlag();
    adjustLock();
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] Adjusting mesh size...\n", cid);
#endif
    sizeElements += 100;
    sizeEdges += 300;
    sizeNodes += 300;
    theElements.resize(sizeElements);
    theEdges.resize(sizeEdges);
    theNodes.resize(sizeNodes);
    for (i=elementSlots; i<sizeElements; i++)
      theElements[i].present = 0;
    for (i=nodeSlots; i<sizeNodes; i++)
      theNodes[i].present = 0;
    for (i=edgeSlots; i<sizeEdges; i++)
      theEdges[i].present = 0;
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] Done adjusting mesh size...\n", cid);
#endif
    adjustRelease();
  }
}

intMsg *chunk::addNode(node n, int b1, int b2, int internal)
{
  intMsg *im = new intMsg;
  im->anInt = firstFreeNode;
  theNodes[firstFreeNode] = n;
  theNodes[firstFreeNode].present = 1;
  if ((b1 == 0) || (b2 == 0) || internal) theNodes[firstFreeNode].boundary = 0;
  else if (b1 < b2) theNodes[firstFreeNode].boundary = b1; 
  else theNodes[firstFreeNode].boundary = b2;
  numNodes++;
  firstFreeNode++;
  if (firstFreeNode-1 == nodeSlots)  nodeSlots++;
  else  while (theNodes[firstFreeNode].isPresent()) firstFreeNode++;
  return im;
}

edgeRef chunk::addEdge(int n1, int n2, int b)
{
#ifdef TDEBUG1  
  CkPrintf("TMRC2D: [%d] Adding edge %d between nodes %d and %d\n", 
	   cid, numEdges, n1, n2);
#endif
  edgeRef eRef(cid, firstFreeEdge);
  theEdges[firstFreeEdge].set(firstFreeEdge, cid, this);
  theEdges[firstFreeEdge].reset();
  theEdges[firstFreeEdge].setNodes(n1, n2);
  theEdges[firstFreeEdge].setBoundary(b);
  numEdges++;
  firstFreeEdge++;
  if (firstFreeEdge-1 == edgeSlots)  edgeSlots++;
  else  while (theEdges[firstFreeEdge].isPresent()) firstFreeEdge++;
  return eRef;
}

edgeRef chunk::addEdge(int n1, int n2, const int *edgeBounds, const int *edgeConn, int nEdges)
{
  int i=0;
#ifdef TDEBUG1  
  CkPrintf("TMRC2D: [%d] Adding edge %d between nodes %d and %d\n", 
	   cid, nEdges, n1, n2);
#endif
  while (i<nEdges && !(n1==edgeConn[2*i] && n2==edgeConn[2*i+1]) || (n2==edgeConn[2*i] && n1==edgeConn[2*i+1]))
    i++;
  theEdges[i].set(i, cid, this);
  theEdges[i].reset();
  theEdges[i].setNodes(n1, n2);
  theEdges[i].setBoundary(edgeBounds[i]);
  edgeRef eRef(cid, i);
  numEdges++;
  while (theEdges[firstFreeEdge].isPresent()) firstFreeEdge++;
  return eRef;
}


elemRef chunk::addElement(int n1, int n2, int n3)
{
  elemRef eRef(cid, firstFreeElement);
  theElements[firstFreeElement].set(cid, firstFreeElement, this);
  theElements[firstFreeElement].set(n1, n2, n3);
  theElements[firstFreeElement].calculateArea();
  numElements++;
  firstFreeElement++;
  if (firstFreeElement-1 == elementSlots)  elementSlots++;
  else  while (theElements[firstFreeElement].isPresent()) firstFreeElement++;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
  return eRef;
}

elemRef chunk::addElement(int n1, int n2, int n3,
			  edgeRef er1, edgeRef er2, edgeRef er3)
{
#ifdef TDEBUG1
  CkPrintf("TMRC2D: [%d] New element %d added with nodes %d, %d and %d\n", 
	   cid, firstFreeElement, n1, n2, n3);
#endif
  elemRef eRef(cid, firstFreeElement);
  theElements[firstFreeElement].set(cid, firstFreeElement, this);
  theElements[firstFreeElement].set(n1, n2, n3, er1, er2, er3);
  theElements[firstFreeElement].calculateArea();
  numElements++;
  firstFreeElement++;
  if (firstFreeElement-1 == elementSlots)  elementSlots++;
  else  while (theElements[firstFreeElement].isPresent()) firstFreeElement++;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
  return eRef;
}

void chunk::removeNode(int n)
{
#ifdef TDEBUG1
  CkPrintf("TMRC2D: [%d] Removing node %d\n", cid, n);
#endif
  if (theNodes[n].present) {
    theNodes[n].present = 0;
    theNodes[n].reset();
    if (n < firstFreeNode) firstFreeNode = n;
    else if ((n == firstFreeNode) && (firstFreeNode == nodeSlots)) {
      firstFreeNode--; nodeSlots--;
    }
    numNodes--;
  }
}

void chunk::removeEdge(int n)
{
#ifdef TDEBUG1
  CkPrintf("TMRC2D: [%d] Removing edge %d\n", cid, n);
#endif
  if (theEdges[n].present) {
    theEdges[n].reset();
    theEdges[n].present = 0;
    theEdges[n].elements[0].reset();
    theEdges[n].elements[1].reset();
    theEdges[n].nodes[0] = theEdges[n].nodes[1] = -1;
    if (n < firstFreeEdge) firstFreeEdge = n;
    else if ((n == firstFreeEdge) && (firstFreeEdge == edgeSlots)) {
      firstFreeEdge--; edgeSlots--;
    }
    numEdges--;
  }
  else {
    CkPrintf("TMRC2D: [%d] WARNING: chunk::removeEdge(%d): edge not present\n", cid, n);
  }	
}

void chunk::removeElement(int n)
{
#ifdef TDEBUG1
  CkPrintf("TMRC2D: [%d] Removing element %d\n", cid, n);
#endif
  if (theElements[n].present) {
    theElements[n].present = 0;
    theElements[n].clear();
    if (n < firstFreeElement) firstFreeElement = n;
    else if ((n == firstFreeElement) && (firstFreeElement == elementSlots)) {
      firstFreeElement--; elementSlots--;
    }
    numElements--;
  }
  else {
    CkPrintf("TMRC2D: [%d] WARNING: chunk::removeElement(%d): element not present\n", cid, n);
  }	
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
  node n;

  memset(filename, 0, 30);
  sprintf(filename, "mesh_debug_%d.%d", cid, c);
  fp = fopen(filename, "w");

  fprintf(fp, "%d %d\n", cid, numElements);
  for (i=0; i<elementSlots; i++) {
    if (theElements[i].isPresent()) {
      for (j=0; j<3; j++) {
	n = theNodes[theElements[i].getNode(j)];
	fprintf(fp, "%f %f   ", n.X(), n.Y());
      }
      fprintf(fp, "%d %f\n", i, theElements[i].getTargetArea());
      for (j=0; j<3; j++) {
	fprintf(fp, "0  ");
      }
      fprintf(fp, "\n");
    }
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
    fprintf(fp, " %d %d ", theEdges[i].elements[0].idx, theEdges[i].elements[0].cid);
    fprintf(fp, " %d %d\n", theEdges[i].elements[1].idx, theEdges[i].elements[1].cid);
  }
  for (i=0; i<numElements; i++) {
    if (theElements[i].isPresent()) {
      fprintf(fp, " %d ", theElements[i].nodes[0]);
      fprintf(fp, " %d ", theElements[i].nodes[1]);
      fprintf(fp, " %d ", theElements[i].nodes[2]);
      fprintf(fp, "   ");
      fprintf(fp, " %d %d ", theElements[i].edges[0].idx, theElements[i].edges[0].cid);
      fprintf(fp, " %d %d ", theElements[i].edges[1].idx, theElements[i].edges[1].cid);
      fprintf(fp, " %d %d\n", theElements[i].edges[2].idx, theElements[i].edges[2].cid);
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
}

void chunk::updateNodeCoords(int nNode, double *coord, int nEl)
{
  int i;
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] updateNodeCoords...\n", cid);
#endif
  if (first) {
    CkWaitQD();
    first = 0;
  }
#ifdef TDEBUG3
  CkPrintf("TMRC2D: [%d] In updateNodeCoords after CkWaitQD: edges sent=%d edge recvd=%d\n", cid, edgesSent, edgesRecvd);
#endif
#ifdef TDEBUG1
  sanityCheck(); // quietly make sure mesh is in shape
  validCheck();
#endif
  // do some error checking
  if (nEl != numElements) {
    CkPrintf("TMRC2D: [%d] ERROR: nEl=%d passed in updateNodeCoords does not match TMRC2D numElements=%d!\n", cid, nEl, numElements);
    CkAssert(nEl == numElements);
  }
  if (nNode != numNodes) {
    CkPrintf("TMRC2D: [%d] ERROR: nNode=%d passed in updateNodeCoords does not match TMRC2D numNodes=%d!\n", cid, nNode, numNodes);
    CkAssert(nNode == numNodes);
  }	
  // update node coordinates from coord
  for (i=0; i<nodeSlots; i++)
    if (theNodes[i].isPresent()) {
      theNodes[i].set(coord[2*i], coord[2*i + 1]);
#ifdef TDEBUG3      
      if (theNodes[i].boundary) {
	CkPrintf("TMRC2D: [%d] Node %d on boundary!\n", cid, i);
      }	
#endif
    }
  // recalculate and cache new areas for each element
  for (i=0; i<elementSlots; i++) 
    if (theElements[i].isPresent())  theElements[i].calculateArea();
#ifdef TDEBUG1
  sanityCheck(); // quietly make sure mesh is in shape
  dump();
#endif
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] updateNodeCoords DONE.\n", cid);
#endif
}

void chunk::multipleRefine(double *desiredArea, refineClient *client)
{
  int i;
  double area;
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] multipleRefine....\n", cid);
#endif
#ifdef TDEBUG1
  sanityCheck(); // quietly make sure mesh is in shape
#endif
  CmiMemoryCheck();
  theClient = client; // initialize refine client associated with this chunk
  
  if (refineStack) delete [] refineStack;
  refineStack=new elemHeap[numElements];
  if (refineElements) delete [] refineElements;
  refineElements = new elemHeap[numElements+1];
  for (i=0; i<elementSlots; i++) { // set desired areas for elements
    if (theElements[i].isPresent()) {
      area = theElements[i].getArea();
      if (area == 0.0) {
	CkPrintf("Element[%d] has area %1.10e!\n", i, area);
      }
      //precThrshld = area * 1e-8;
      if (desiredArea[i] < REFINE_PRECISION*area) {
	theElements[i].resetTargetArea(desiredArea[i]);
	double qFactor=theElements[i].getAreaQuality();
	Insert(i, qFactor, 0);
	CkPrintf("Element[%d] has area %1.10e target %1.10e qFactor %1.10e\n", i, area, desiredArea[i], qFactor);
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Setting target on element %d to %1.10e with largeEdge %1.10e\n", cid, i, desiredArea[i], qFactor);
#endif
      }
    }
  }
  // start refinement
  if (!refineInProgress) {
   refineInProgress = 1;
   mesh[cid].refiningElements();
  }
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] multipleRefine DONE.\n", cid);
#endif
  CkWaitQD();
  CmiMemoryCheck();
}

void chunk::multipleCoarsen(double *desiredArea, refineClient *client)
{
  int i;
  double precThrshld, area, qFactor;
  CmiMemoryCheck();
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] multipleCoarsen....\n", cid);
#endif
#ifdef TDEBUG1
  sanityCheck(); // quietly make sure mesh is in shape
#endif
  theClient = client; // initialize refine client associated with this chunk
  if (coarsenElements) delete [] coarsenElements;
  coarsenElements = new elemHeap[numElements+1];
  coarsenElements[0].len=-2.0;
  coarsenElements[0].elID=-1;
  for (i=0; i<elementSlots; i++) { // set desired areas for elements
    if (theElements[i].isPresent()) {
      area = theElements[i].getArea();
      if (area == 0.0) {
	CkPrintf("Element[%d] has area %1.10e!\n", i, area);
      }
      double shortEdgeLen, angle;
      shortEdgeLen = theElements[i].getShortestEdge(&angle);
      qFactor = theElements[i].getAreaQuality();
      theElements[i].nonCoarsenCount = 0;
      //precThrshld = area * 1e-8;
      theElements[i].resetTargetArea(desiredArea[i]);
      if ((desiredArea[i]*COARSEN_PRECISION > area) || (area == 0.0) || 
	  (qFactor<QUALITY_MIN)) {
	Insert(i, shortEdgeLen*qFactor, 1);
	CkPrintf("Element[%d] has area %1.10e target %1.10e qFactor %1.10e\n", i, area, desiredArea[i], qFactor);
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Setting target on element %d to %1.10e with shortEdge %1.10e\n", cid, i, desiredArea[i], qFactor);
#endif
      }
    }
  }
  
  // start coarsening
  if (!coarsenInProgress) {
   coarsenInProgress = 1;
   mesh[cid].coarseningElements();
  }
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] multipleCoarsen DONE.\n", cid);
#endif
  CkWaitQD();
/*
  for (i=0; i<elementSlots; i++) { // check desired areas for elements
    if (theElements[i].isPresent()) {
      area = theElements[i].getArea();
      if (desiredArea[i] > area) {
	CkPrintf("TMRC2D: [%d] WARNING: element %d area is %1.10e but target was %1.10e\n", cid, i, area, desiredArea[i]);
      }
    }
  }
*/
  CmiMemoryCheck();
}

void chunk::newMesh(int meshID_,int nEl, int nGhost, const int *conn_, 
		    const int *gid_, int nnodes, const int *boundaries, int nEdges, 
		    const int *edgeConn, const int *edgeBounds, int idxOffset)
{
  int i, j;
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] In newMesh...\n", cid);
#endif
  meshID = meshID_;

  meshPtr = FEM_Mesh_lookup(meshID, "chunk::newMesh");
  numElements=nEl;
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
      conn[i*3 + j] = c;
      nodes[j] = c;
      edges[j].reset();
     }
    theElements[i].set(cid, i, this);
    theElements[i].set(nodes, edges);
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

  MPI_Barrier(MPI_COMM_WORLD);
  // derive edges from elements on this chunk
  if (boundaries) {
    numNodes = nnodes;
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] Received non-NULL boundaries... adding...\n", cid);
#endif
    for (i=0; i<numNodes; i++) {
      theNodes[i].boundary = boundaries[i];
#ifdef TDEBUG3
      if (theNodes[i].boundary) 
	CkPrintf("TMRC2D: [%d] Node %d has boundary %d.\n", cid, i, theNodes[i].boundary);
#endif
    }
    deriveEdges(conn, gid, edgeBounds, edgeConn, nEdges);
    CkAssert(nnodes == numNodes);
  }
  else {
    deriveEdges(conn, gid, edgeBounds, edgeConn, nEdges);
    CkAssert(nnodes == numNodes);
    deriveBoundaries(conn, gid);
  }
  delete[] conn;
  delete[] gid;
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] newMesh DONE; chunk created with %d elements.\n", 
	   cid, numElements);
#endif
}

void chunk::deriveEdges(int *conn, int *gid, const int *edgeBounds, const int *edgeConn, int nEdges)
{
  // need to add edges to the chunk, and update all edgeRefs on all elements
  // also need to add nodes to the chunk
  int i, j, n1localIdx, n2localIdx;
  edgeRef newEdge;

  deriveNodes(); // now numNodes and theNodes have values
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] Deriving edges...\n", cid);
#endif
  for (i=0; i<elementSlots; i++) {
#ifdef TDEBUG3
    CkPrintf("TMRC2D: [%d] Deriving edges for element %d...\n", cid, i);
#endif
    elemRef myRef(cid,i);
    for (j=0; j<3; j++) {
      n1localIdx = j;
      n2localIdx = (j+1) % 3;
#ifdef TDEBUG3
    CkPrintf("TMRC2D: [%d] Deriving edges for element %d between %d,%d (real nodes %d,%d)...\n", cid, i, n1localIdx, n2localIdx, theElements[i].nodes[n1localIdx], theElements[i].nodes[n2localIdx]);
#endif
      // look for edge
      if (theElements[i].edges[j] == nullRef) { // the edge doesn't exist yet
	// get nbr ref
#ifdef TDEBUG3
	CkPrintf("TMRC2D: [%d] Edge between nodes %d,%d doesn't exist yet...\n", cid, theElements[i].nodes[n1localIdx], theElements[i].nodes[n2localIdx]);
#endif
	elemRef nbrRef;
	int edgeIdx = getNbrRefOnEdge(theElements[i].nodes[n1localIdx], 
				      theElements[i].nodes[n2localIdx], 
				      conn, numGhosts, gid, i, &nbrRef);
	FEM_Node *mNodes = &(meshPtr->node);
	int nIdx1 = theElements[i].nodes[n1localIdx];
	int nIdx2 = theElements[i].nodes[n2localIdx];
	if ((theNodes[nIdx1].boundary < theNodes[nIdx2].boundary) && 
	    (theNodes[nIdx1].boundary != 0))	{
	  theNodes[nIdx2].fixed = 1;
#ifdef TDEBUG2
	  CkPrintf("TMRC2D: [%d] Corner detected, fixing node %d\n", 
		   cid, nIdx2);
#endif
	  FEM_Comm_Rec *nRec = (FEM_Comm_Rec*)(mNodes->shared.getRec(nIdx2));
	  if (nRec) {
	    int count = nRec->getShared();
	    for (i=0; i<count; i++) {
	      mesh[nRec->getChk(i)].fixNode(nRec->getIdx(i), cid);
	    }
	  }
	}
	if ((theNodes[nIdx2].boundary < theNodes[nIdx1].boundary) && 
	    (theNodes[nIdx2].boundary != 0)) {
	  theNodes[nIdx1].fixed = 1;
#ifdef TDEBUG2
	  CkPrintf("TMRC2D: [%d] Corner detected, fixing node %d\n", 
		   cid, nIdx1);
#endif
	  FEM_Comm_Rec *nRec = (FEM_Comm_Rec*)(mNodes->shared.getRec(nIdx1));
	  if (nRec) {
	    int count = nRec->getShared();
	    for (i=0; i<count; i++) {
	      mesh[nRec->getChk(i)].fixNode(nRec->getIdx(i), cid);
	    }
	  }
	}
	if (edgeLocal(myRef, nbrRef)) { // make edge here
	  if (!edgeBounds)
	    newEdge = addEdge(nIdx1, nIdx2, 0);
	  else
	    newEdge = addEdge(nIdx1, nIdx2, edgeBounds, edgeConn, nEdges);
#ifdef TDEBUG3
	  CkPrintf("TMRC2D: [%d] New edge (%d,%d) added between nodes %d and %d and elements %d and %d on chunk %d\n", cid, newEdge.cid, newEdge.idx, nIdx1, nIdx2, i, nbrRef.idx, nbrRef.cid);
#endif
	  // point edge to the two neighboring elements
	  theEdges[newEdge.idx].update(nullRef, myRef);
	  theEdges[newEdge.idx].update(nullRef, nbrRef);
	  // point elem i's edge j at the edge
	  theElements[i].set(j, newEdge);
	  // point nbrRef at the edge
	  if (nbrRef.cid==cid) // Local neighbor
	    theElements[nbrRef.idx].set(edgeIdx, newEdge);
	  else if (nbrRef.cid != -1) { // Remote neighbor
	    mesh[nbrRef.cid].addRemoteEdge(nbrRef.idx, edgeIdx, newEdge);
	    edgesSent++;
	  }
	}
        else { // else edge will be made on a different chunk
#ifdef TDEBUG3
	  CkPrintf("TMRC2D: [%d] Edge between nodes %d,%d to be created elsewhere...\n", cid, theElements[i].nodes[n1localIdx], theElements[i].nodes[n2localIdx]);
#endif
	  // mark the edge as non-boundary; will be filled later
	  theElements[i].edges[j].idx = theElements[i].edges[j].cid = -2;
	}
      }
#ifdef TDEBUG3
      else { 
	CkPrintf("TMRC2D: [%d] Edge between nodes %d,%d exists at %d\n", cid, theElements[i].nodes[n1localIdx], theElements[i].nodes[n2localIdx], theElements[i].edges[j].idx);
      }
#endif
    }
  }
  nodeSlots = numNodes;
  firstFreeNode = numNodes;
  edgeSlots = numEdges;
  firstFreeEdge = numEdges;
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] Done deriving edges...\n", cid);
#endif
}

void chunk::deriveNodes()
{
  int i, j;
  int aNode;

#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] Deriving nodes...\n", cid);
#endif
  numNodes = 0;
  for (i=0; i<elementSlots; i++) {
    for (j=0; j<3; j++) {
      aNode = theElements[i].nodes[j];
      CkAssert(aNode > -1);
      if ((aNode + 1) > numNodes)
	numNodes = aNode + 1;
      theNodes[aNode].present = 1;
    }
  }
#ifdef TDEBUG3
  CkPrintf("TMRC2D: [%d] NumNodes = %d; max node idx = %d\n", 
	   cid, numNodes, numNodes-1);
#endif
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] Done deriving nodes.\n", cid);
#endif
}

int chunk::edgeLocal(elemRef e1, elemRef e2)
{
  return ((e1.cid==-1 || e2.cid==-1) || (e1.cid > e2.cid) ||
	  ((e1.cid == e2.cid) && (e1.idx < e2.idx)));
}

int chunk::getNbrRefOnEdge(int n1, int n2, int *conn, int nGhost, int *gid, 
			   int idx, elemRef *er)
{
  int i, e;
  er->set(-1, -1);
  for (i=0; i<nGhost; i++) {
    if (i != idx) {
      if ((e = hasEdge(n1, n2, conn, i)) != -1) {
	er->set(gid[i*2], gid[i*2+1]);
	return e;
      }
    }
  }
  return -1;
}

int chunk::hasEdge(int n1, int n2, int *conn, int idx) 
{
  int i, j;
  for (i=0; i<3; i++) {
    j = (i+1) % 3;
    if (((conn[idx*3+i] == n1) && (conn[idx*3+j] == n2)) ||
	((conn[idx*3+j] == n1) && (conn[idx*3+i] == n2)))
      return i;
  }
  return -1;
}

void chunk::deriveBoundaries(int *conn, int *gid)
{
  int edgeIdx;
  elemRef nbrRef;
  CkPrintf("TMRC2D: [%d] WARNING! Null list of boundary flags passed to newMesh...\n ...I hope you didn't want coarsening to work!\n", cid);
  CkPrintf("TMRC2D: [%d] WARNING! Attempting to derive single boundary information -- corners will be ignored!\n", cid);
  for (int i=0; i<numElements; i++) {
    for (int j=0; j<3; j++) {
      edgeIdx = getNbrRefOnEdge(theElements[i].nodes[j], 
				theElements[i].nodes[(j+1)%3], 
				conn, numGhosts, gid, i, &nbrRef); 
      if ((nbrRef.idx == -1) && (nbrRef.cid == -1)) {
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Marking node %d as a boundary.\n", 
		 cid, theElements[i].nodes[j]);
	CkPrintf("TMRC2D: [%d] Marking node %d as a boundary.\n", 
		 cid, theElements[i].nodes[(j+1)%3]);
#endif
	theNodes[theElements[i].nodes[j]].boundary = 1;
	theNodes[theElements[i].nodes[(j+1)%3]].boundary = 1;
      }
    }
  }
}

void chunk::tweakMesh()
{
  CkPrintf("TMRC2D: [%d] WARNING! chunk::tweakMesh called but not implemented!\n", cid);
  /*
  for (int i=0; i<numElements; i++) 
    theElements[i].tweakNodes();
  */
}

void chunk::improveChunk()
{
  CkPrintf("TMRC2D: [%d] WARNING! chunk::improveChunk called but not implemented!\n", cid);
  /*
  for (int i=0; i<numNodes; i++) 
    if (!theNodes[i].border) {
      theNodes[i].improvePos();
    }
    else CkPrintf("Not adjusting node %d on chunk %d\n", i, thisIndex);
  */
}

void chunk::improve()
{
  CkPrintf("TMRC2D: [%d] WARNING! chunk::improve called but not implemented!\n", cid);
  /*
  for (int i=0; i<20; i++) {
    mesh.tweakMesh();
    mesh.improveChunk();
  }
  */
}

void chunk::sanityCheck(void)
{
  CkPrintf("TMRC2D: [%d] running sanity check...\n", cid);
  int i;
  if (numElements<0 || (int)theElements.size()<numElements)
        CkAbort("sc-> TMRC2D: numElements or vector size insane!");
  if (numEdges<0 || (int)theEdges.size()<numEdges)
        CkAbort("sc-> TMRC2D: numEdges or vector size insane!");
  if (numNodes<0 || (int)theNodes.size()<numNodes)
        CkAbort("sc-> TMRC2D: numNodes or vector size insane!");
  i=0; 
  while (theElements[i].isPresent()) i++;
  CkAssert(firstFreeElement == i);
  i=0; 
  while (theEdges[i].isPresent()) i++;
  CkAssert(firstFreeEdge == i);
  i=0; 
  while (theNodes[i].isPresent()) i++;
  CkAssert(firstFreeNode == i);
  for (i=0; i<elementSlots; i++) 
    if (theElements[i].isPresent())
      theElements[i].sanityCheck(this, elemRef(cid,i), nodeSlots);
  for (i=0; i<edgeSlots; i++)
    if (theEdges[i].isPresent())
      theEdges[i].sanityCheck(this, edgeRef(cid,i));
  for (i=0; i<nodeSlots; i++)
    if (theNodes[i].isPresent())
      theNodes[i].sanityCheck(cid, i);
  CkPrintf("TMRC2D: [%d] sanity check PASSED.\n", cid);
}

void chunk::dump()
{
  int i;
  CkPrintf("TMRC2D: [%d] has %d elements, %d nodes and %d edges:\n", 
	   cid, numElements, numNodes, numEdges);
  for (i=0; i<elementSlots; i++) 
    if (theElements[i].isPresent()){
      CkPrintf("TMRC2D: [%d] Element %d nodes %d %d %d edges [%d,%d] [%d,%d] [%d,%d]\n", cid, i, theElements[i].nodes[0], theElements[i].nodes[1], theElements[i].nodes[2], theElements[i].edges[0].cid, theElements[i].edges[0].idx, theElements[i].edges[1].cid, theElements[i].edges[1].idx, theElements[i].edges[2].cid, theElements[i].edges[2].idx);
		}	
  for (i=0; i<nodeSlots; i++)
    if (theNodes[i].isPresent()) {
      CkPrintf("TMRC2D: [%d] Node %d has coordinates (%f,%f)\n", 
	       cid, i, theNodes[i].X(), theNodes[i].Y());
    }		 
  for (i=0; i<edgeSlots; i++)
    if (theEdges[i].isPresent()) {
      CkPrintf("TMRC2D: [%d] Edge %d has nodes %d %d elements [%d,%d] [%d,%d]\n", cid, 
	       i, theEdges[i].nodes[0], theEdges[i].nodes[1], theEdges[i].elements[0].cid, theEdges[i].elements[0].idx, 
	       theEdges[i].elements[1].cid, theEdges[i].elements[1].idx);
    }		 
}

intMsg *chunk::lockChunk(int lhc, int lhi, double prio)
{
  intMsg *im = new intMsg;
  im->anInt = lockLocalChunk(lhc, lhi, prio);
  return im;
}

int chunk::lockLocalChunk(int lhc, int lhi, double prio)
{ // priority is given to the lower prio, lower lh
  // each chunk can have at most one lock reservation
  // if a chunk holds a lock, it cannot have a reservation
  if (!lock) { // this chunk is UNLOCKED
    // remove previous lh reservation; insert this reservation
    // then check first reservation
    removeLock(lhc, lhi);
    insertLock(lhc, lhi, prio);
    if ((lockList->prio == prio) && (lockList->holderIdx == lhi)
	&& (lockList->holderCid == lhc)) {
#ifdef TDEBUG3
      CkPrintf("TMRC2D: [%d] LOCK chunk %d by %d on %d prio %.10f LOCKED(was %d[%don%d:%.10f])\n", CkMyPe(), cid, lhi, lhc, prio, lock, lockHolderIdx, lockHolderCid, lockPrio);
#endif
      lock = 1; lockHolderCid = lhc; lockHolderIdx = lhi; lockPrio = prio; 
      //lockCount = 1;
      lockList = lockList->next;
      return 1;
    }
    removeLock(lhc, lhi);
#ifdef TDEBUG3
    CkPrintf("TMRC2D: [%d] LOCK chunk %d by %d on %d prio %.10f REFUSED (was %d[%don%d:%.10f])\n", CkMyPe(), cid, lhi, lhc, prio, lock, lockHolderIdx, lockHolderCid, lockPrio);
#endif
    return 0;
  }
  else { // this chunk is LOCKED
    if ((prio == lockPrio) && (lhi == lockHolderIdx) && (lhc == lockHolderCid)) {
      //lockCount++;
#ifdef TDEBUG3
      CkPrintf("TMRC2D: [%d] LOCK chunk %d by %d on %d prio %.10f HELD (was %d[%don%d:%.10f])\n", CkMyPe(), cid, lhi, lhc, prio, lock, lockHolderIdx, lockHolderCid, lockPrio);
#endif
      return 1;
    }
    if ((lhi == lockHolderIdx) && (lhc == lockHolderCid)) {
      CkPrintf("ERROR: lockholder %d on %d trying to relock with different prio! prio=%f newprio=%f\n", lhi, lhc, lockPrio, prio);
      CmiAssert(!((lhi == lockHolderIdx) && (lhc == lockHolderCid)));
    }
    /*
    removeLock(lhc, lhi);
    insertLock(lhc, lhi, prio);
    if ((lockList->prio == prio) && (lockList->holderIdx == lhi) && (lockList->holderCid == lhc)) {
      if ((prio < lockPrio) && (lhc != lockHolderCid)) { // lh might be next
	CkPrintf("[%d]LOCK chunk %d by %d on %d prio %.10f SPIN (was %d[%don%d:%.10f])\n", CkMyPe(), cid, lhi, lhc, prio, lock, lockHolderIdx, lockHolderCid, lockPrio);
	return 0; 
      }
    }
    removeLock(lhc, lhi);
    */
#ifdef TDEBUG3
    CkPrintf("TMRC2D: [%d] LOCK chunk %d by %d on %d prio %.10f REFUSED (was %d[%don%d:%.10f])\n", CkMyPe(), cid, lhi, lhc, prio, lock, lockHolderIdx, lockHolderCid, lockPrio);
#endif
    return 0;
  }
}

void chunk::insertLock(int lhc, int lhi, double prio)
{
  prioLockStruct *newLock, *tmp;
  
  newLock = (prioLockStruct *)malloc(sizeof(prioLockStruct));
  newLock->prio = prio;
  newLock->holderIdx = lhi;
  newLock->holderCid = lhc;
  newLock->next = NULL;
  if (!lockList) lockList = newLock;
  else {
    if ((prio < lockList->prio) || 
	((prio == lockList->prio) && (lhc < lockList->holderCid)) ||
	((prio == lockList->prio) && (lhc == lockList->holderCid) && 
	 (lhi < lockList->holderIdx))) {
      // insert before first element in lockList
      newLock->next = lockList;
      lockList = newLock;
    }
    else { 
      tmp = lockList;
      while (tmp->next) {
	if ((prio > tmp->next->prio) || 
	    ((prio == tmp->next->prio) && (lhc > tmp->next->holderCid)) ||
	    ((prio == tmp->next->prio) && (lhc == tmp->next->holderCid) &&
	     (lhi > tmp->next->holderIdx))) 
	  tmp = tmp->next;
	else break;
      }
      newLock->next = tmp->next;
      tmp->next = newLock;
    }
  }
}

void chunk::removeLock(int lhc, int lhi) 
{
  prioLockStruct *tmp = lockList, *del;
  if (!lockList) return;
  if ((tmp->holderCid == lhc) && (tmp->holderIdx == lhi)) {
    lockList = lockList->next;
    free(tmp);
  }
  else {
    while (tmp->next && 
	   ((tmp->next->holderCid != lhc) || (tmp->next->holderIdx != lhi)))
      tmp = tmp->next;
    if (tmp->next) {
      del = tmp->next;
      tmp->next = del->next;
      free(del);
    }
  }
}

void chunk::unlockChunk(int lhc, int lhi)
{
  unlockLocalChunk(lhc, lhi);
}

void chunk::unlockLocalChunk(int lhc, int lhi)
{
  if ((lockHolderIdx == lhi) && (lockHolderCid == lhc)) {
    //    lockCount--;
    //    if (lockCount == 0) {
      lock = 0; 
      lockHolderIdx = -1;
      lockHolderCid = -1;
      lockPrio = -1.0;
#ifdef TDEBUG3
      CkPrintf("TMRC2D: [%d] UNLOCK chunk %d by holder %d on %d\n", 
	       CkMyPe(), cid, lhi, lhc);
#endif
	//    }
  }
  /*  else { 
    CkPrintf("TMRC2D: [%d] ERROR: UNLOCK chunk %d by holder %d on %d -- THIS WAS NOT THE HOLDER!!! %d on %d was!\n", CkMyPe(), cid, lhi, lhc, lockHolderIdx, lockHolderCid);
    CkAbort(0);
    }*/
}

void chunk::fixNode(int nIdx, int chkid)
{
  FEM_Node *nodesPtr = &(meshPtr->node);
  const FEM_Comm_List *sharedList = &(nodesPtr->shared.getList(chkid));
  nIdx = (*sharedList)[nIdx];

  theNodes[nIdx].fixed = 1;
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] Corner detected, fixing node %d\n", cid, nIdx);
#endif
}

int chunk::joinCommLists(int nIdx, int shd, int *chk, int *idx, int *rChk,
			 int *rIdx)
{
  FEM_Node *theNodes = &(meshPtr->node);
  FEM_Comm_Rec *nodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(nIdx));
  int nShared, count, chunk, index, found, i, j;
  if (nodeRec) nShared = nodeRec->getShared();
  else { 
    rChk = chk;
    rIdx = idx;
    return shd;
  }
  rChk = (int *)malloc((shd+nShared)*sizeof(int));
  rIdx = (int *)malloc((shd+nShared)*sizeof(int));
  for (i=0; i<shd; i++) {
    rChk[i] = chk[i];
    rIdx[i] = idx[i];
  }
  count = shd;
  for (i=0; i<nShared; i++) {
    chunk = nodeRec->getChk(i);
    for (j=0; j<shd; j++) {
      if (rChk[j] == chunk) { // already present
	found = 1;
	break;
      }
    }
    if (!found) {
      index = nodeRec->getIdx(i-shd);
      rChk[count] = chunk;
      rIdx[count] = index;
      count++;
    }
  }
  free(rChk);
  free(rIdx);
  return count;
}

void chunk::Insert(int eIdx, double len, int cFlag)
{
  int i;
  if (cFlag) {
    i = ++coarsenHeapSize; 
    while ((coarsenElements[i/2].len>=len) && (i != 1))
    {
      coarsenElements[i].len=coarsenElements[i/2].len;
      coarsenElements[i].elID=coarsenElements[i/2].elID;  // swap bubble with number above  
      i/=2;                     
    }
    coarsenElements[i].elID=eIdx;
    coarsenElements[i].len=len; 
  }
  else {
    i = ++refineHeapSize; 
    while ((refineElements[i/2].len>=len) && (i != 1))
    {
      refineElements[i].len=refineElements[i/2].len;
      refineElements[i].elID=refineElements[i/2].elID;  // swap bubble with number above  
      i/=2;                     
    }
    refineElements[i].elID=eIdx;
    refineElements[i].len=len; 
  }
}

// removes and returns the minimum element of the heap 
int chunk::Delete_Min(int cflag)
{
  int Child, i, Min_ID; 

  if (cflag) {
    Min_ID=coarsenElements[1].elID;
    for (i=1; i*2 <= coarsenHeapSize-1; i=Child)
    {
      // Find smaller child
      Child = i*2;       // child is left child  
      if (Child !=coarsenHeapSize)  // If this is not == Size, then right child exists
	if (coarsenElements[Child+1].len < coarsenElements[Child].len)
	  Child++; 

      // Percolate one level
      if (coarsenElements[coarsenHeapSize].len >= coarsenElements[Child].len){  
	coarsenElements[i].elID = coarsenElements[Child].elID;   
	coarsenElements[i].len = coarsenElements[Child].len;
      }
      else
         break; 
    }
    coarsenElements[i].elID = coarsenElements[coarsenHeapSize].elID;   // Place holding value into the bubble
    coarsenElements[i].len = coarsenElements[coarsenHeapSize].len; 
    coarsenHeapSize--;
    return Min_ID; 
  }
  else {
    Min_ID=refineElements[1].elID;
    for (i=1; i*2 <= refineHeapSize-1; i=Child)
    {
      // Find smaller child
      Child = i*2;       // child is left child  
      if (Child !=refineHeapSize)  // If this is not == Size, then right child exists
	if (refineElements[Child+1].len < refineElements[Child].len)
	  Child++; 

      // Percolate one level
      if (refineElements[refineHeapSize].len >= refineElements[Child].len){  
	refineElements[i].elID = refineElements[Child].elID;   
	refineElements[i].len = refineElements[Child].len;
      }
      else
         break; 
    }
    refineElements[i].elID = refineElements[refineHeapSize].elID;   // Place holding value into the bubble
    refineElements[i].len = refineElements[refineHeapSize].len; 
    refineHeapSize--;
    return Min_ID; 
  }
}



intMsg *chunk::getBoundary(int edgeIdx)
{
  intMsg *im = new intMsg;
  im->anInt = theEdges[edgeIdx].getBoundary();
  return im;
}


void chunk::validCheck()
{
  int i;
  FEM_Entity *fem_node = meshPtr->lookup(FEM_NODE,"validCheck");
  FEM_DataAttribute *validNode = (FEM_DataAttribute *)fem_node->lookup(FEM_ATTRIB_TAG_MAX-1,"validCheck");
  AllocTable2d<int> &validNodeTable = validNode->getInt();
  /*
  for (i=0; i<elementSlots; i++) {
    //    CkAssert(theElements[i].isPresent()==);
  }
  */
  for (i=0; i<nodeSlots; i++) {
    CkAssert(theNodes[i].isPresent()==validNodeTable[i][0]);
  }
}

intMsg *chunk::neighboring(int idx, elemRef e)
{
  intMsg *m= new intMsg;
  m->anInt = theElements[idx].neighboring(e);
  return m;
}

intMsg *chunk::safeToCoarsen(int idx, edgeRef e)
{
  intMsg *m= new intMsg;
  m->anInt = theElements[idx].safeToCoarsen(e);
  return m;
}

#include "refine.def.h"

