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
    elementSlots(0), edgeSlots(0), nodeSlots(0),
    additions(0), debug_counter(0), refineInProgress(0), coarsenInProgress(0),
    modified(0), meshLock(0), meshExpandFlag(0), 
    numElements(0), numEdges(0), numNodes(0), numGhosts(0), theClient(NULL)
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
  theElements[elem].set(localEdge, er);
  releaseLock();
}

void chunk::refineElement(int idx, double area)
{ // Reduce element's targetArea to indicate need for refinement
  if (!theElements[idx].isPresent()) return;
  accessLock();
  theElements[idx].setTargetArea(area);
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
  while (modified) { // Refine elements until no changes occur
    i = 0;
    modified = 0;
    while (i < elementSlots) { // loop through the elements
      if (theElements[i].isPresent() && 
	  (theElements[i].getTargetArea() <= theElements[i].getArea()) && 
	  (theElements[i].getTargetArea() >= 0.0)) {
	// element i has a lower target area -- needs to refine
	modified = 1; // something's bound to change
	theElements[i].refine(); // refine the element
	adjustMesh();
      }
      i++;
    }
    CthYield(); // give other chunks on the same PE a chance
  }
  refineInProgress = 0; // nothing needs refinement; turn refine loop off
  //    if (CkMyPe() == 0) for (int j=0; j<5; j++) mesh[j].print();
}


void chunk::coarsenElement(int idx, double area)
{ // Increase element's targetArea to indicate need for coarsening
  if (!theElements[idx].isPresent()) return;
  accessLock();
  theElements[idx].resetTargetArea(area);
  releaseLock();
  modified = 1;  // flag a change in one of the chunk's elements
  if (!coarsenInProgress) { // if coarsen loop not running
    coarsenInProgress = 1;
    mesh[cid].coarseningElements(); // start it up
  }
}

void chunk::coarseningElements()
{
  int i;
  while (modified) { // try to coarsen elements until no changes occur
    i = 0;
    modified = 0;
    while (i < elementSlots) { // loop through the elements
      if (theElements[i].isPresent())
	CkPrintf("TMRC2D: Coarsen element %d: area=%f target=%f\n", i, 
		 theElements[i].getArea(), theElements[i].getTargetArea());
      if (theElements[i].isPresent() && 
	  (theElements[i].getTargetArea() > theElements[i].getArea()) && 
	  (theElements[i].getTargetArea() >= 0.0)) {
	// element i has higher target area -- needs coarsening
	modified = 1; // something's bound to change
	theElements[i].coarsen(); // coarsen the element
      }
      i++;
    }
    CthYield(); // give other chunks on the same PE a chance
  }
  coarsenInProgress = 0;  // turn coarsen loop off
  sanityCheck(); // quietly make sure mesh is in shape
}

// many remote access methods follow
void chunk::setBorder(int n)
{
  accessLock();
  theNodes[n].setBorder();
  releaseLock();
}

intMsg *chunk::safeToMoveNode(int idx, double x, double y)
{
  node foo(x, y);
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theNodes[idx].safeToMove(foo);
  releaseLock();
  return im;
}

splitOutMsg *chunk::split(int idx, elemRef e, node in, node fn)
{
  splitOutMsg *som = new splitOutMsg;
  accessLock();
  som->result = theEdges[idx].split(&(som->n), &(som->e), in, fn, e, 
				    &(som->local), &(som->first), 
				    &(som->nullNbr));
  releaseLock();
  return som;
}

intMsg *chunk::collapse(int idx, elemRef e, node kn, node dn)
{
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theEdges[idx].collapse(e, kn, dn);
  releaseLock();
  return im;
}

void chunk::collapseHelp(int idx, edgeRef er, node n1, node n2)
{
  CkPrintf("TMRC2D: WARNING! chunk::collapseHelp called but not implemented!\n");
  //  theElements[idx].collapseHelp(er, n1, n2);
}

intMsg *chunk::nodeLockup(int idx, node n, edgeRef from, edgeRef start, 
			  elemRef end, double l)
{
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theElements[idx].nodeLockup(n, from, start, end, l);
  releaseLock();
  return im;
}

intMsg *chunk::nodeLockupER(int idx, node n, edgeRef start, elemRef from, 
			    elemRef end, double l)
{
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theEdges[idx].nodeLockup(n, start, from, end, l);
  releaseLock();
  return im;
}

intMsg *chunk::nodeUpdate(int idx, node n, edgeRef from, elemRef end, 
			  node newNode)
{
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theElements[idx].nodeUpdate(n, from, end, newNode);
  releaseLock();
  return im;
}

intMsg *chunk::nodeUpdateER(int idx, node n, elemRef from, elemRef end, 
			    node newNode)
{
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theEdges[idx].nodeUpdate(n, from, end, newNode);
  releaseLock();
  return im;
}

intMsg *chunk::nodeDelete(int idx, node n, edgeRef from, elemRef end)
{
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theElements[idx].nodeDelete(n, from, end);
  releaseLock();
  return im;
}

intMsg *chunk::nodeDeleteER(int idx, node n, elemRef from, elemRef end)
{
  intMsg *im = new intMsg;
  accessLock();
  im->anInt = theEdges[idx].nodeDelete(n, from, end);
  releaseLock();
  return im;
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

void chunk::updateElement(int idx, objRef oldval, objRef newval)
{
  elemRef ov, nv;
  ov.idx = oldval.idx;   ov.cid = oldval.cid; 
  nv.idx = newval.idx;   nv.cid = newval.cid; 
  accessLock();
  theEdges[idx].update(ov, nv);
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
  CkPrintf("TMRC2D: WARNING! chunk::updateReferences called but not implemented!\n");
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
  modified = 1;
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
  modified = 1;
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
    CkPrintf("TMRC2D: [%d] Adjusting mesh size...\n", cid);
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
    CkPrintf("TMRC2D: [%d] Done adjusting mesh size...\n", cid);
    adjustRelease();
  }
}

intMsg *chunk::addNode(node n)
{
  intMsg *im = new intMsg;
  im->anInt = firstFreeNode;
  theNodes[firstFreeNode] = n;
  theNodes[firstFreeNode].present = 1;
  numNodes++;
  firstFreeNode++;
  if (firstFreeNode-1 == nodeSlots)  nodeSlots++;
  else  while (theNodes[firstFreeNode].isPresent()) firstFreeNode++;
  return im;
}

edgeRef chunk::addEdge()
{
  CkPrintf("TMRC2D: Adding edge %d to chunk %d\n", numEdges, cid);
  edgeRef eRef(cid, firstFreeEdge);
  theEdges[firstFreeEdge].set(firstFreeEdge, cid, this);
  theEdges[firstFreeEdge].reset();
  numEdges++;
  firstFreeEdge++;
  if (firstFreeEdge-1 == edgeSlots)  edgeSlots++;
  else  while (theEdges[firstFreeEdge].isPresent()) firstFreeEdge++;
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
  CkPrintf("TMRC2D: New element %d added with nodes %d, %d and %d\n", 
	   firstFreeElement, n1, n2, n3);
  elemRef eRef(cid, firstFreeElement);
  theElements[firstFreeElement].set(cid, firstFreeElement, this);
  theElements[firstFreeElement].set(n1, n2, n3, er1, er2, er3);
  theElements[firstFreeElement].calculateArea();
  numElements++;
  firstFreeElement++;
  if (firstFreeElement-1 == elementSlots)  elementSlots++;
  else  while (theElements[firstFreeElement].isPresent()) firstFreeElement++;
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
  return eRef;
}

void chunk::removeNode(int n)
{
  theNodes[n].present = 0;
  theNodes[n].reset();
  if (n < firstFreeNode) firstFreeNode = n;
  else if ((n == firstFreeNode) && (firstFreeNode == nodeSlots)) {
    firstFreeNode--; nodeSlots--;
  }
  numNodes--;
}

void chunk::removeEdge(int n)
{
  theEdges[n].present = 0;
  theEdges[n].reset();
  if (n < firstFreeEdge) firstFreeEdge = n;
  else if ((n == firstFreeEdge) && (firstFreeEdge == edgeSlots)) {
    firstFreeEdge--; edgeSlots--;
  }
  numEdges--;
}

void chunk::removeElement(int n)
{
  theElements[n].present = 0;
  theElements[n].clear();
  if (n < firstFreeElement) firstFreeElement = n;
  else if ((n == firstFreeElement) && (firstFreeElement == elementSlots)) {
    firstFreeElement--; elementSlots--;
  }
  numElements--;
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
  for (i=0; i<numElements; i++) {
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
  CkPrintf("TMRC2D: updateNodeCoords...\n");
  // do some error checking
  CkAssert(nEl == numElements);
  CkAssert(nNode == numNodes);
  // update node coordinates from coord
  for (i=0; i<nodeSlots; i++)
    if (theNodes[i].isPresent()) {
      if ((theNodes[i].X() != coord[2*i]) || (theNodes[i].Y() != coord[2*i+1]))
	CkPrintf("TMRC2D: updateNodeCoords WARNING: coords changed for node %d on chunk %d: Were %f,%f; now %f,%f\n", i, cid, theNodes[i].X(), theNodes[i].Y(), coord[2*i], coord[2*i + 1]);
      theNodes[i].set(coord[2*i], coord[2*i + 1]);
    }
  // recalculate and cache new areas for each element
  for (i=0; i<elementSlots; i++) 
    if (theElements[i].isPresent())  theElements[i].calculateArea();
  CkPrintf("TMRC2D: updateNodeCoords DONE.\n");
}

void chunk::multipleRefine(double *desiredArea, refineClient *client)
{
  int i;
  CkPrintf("TMRC2D: multipleRefine....\n");
  theClient = client; // initialize refine client associated with this chunk
  //Uncomment this dump call to see TMRC2D's mesh config
  //dump();
  sanityCheck(); // quietly make sure mesh is in shape

  for (i=0; i<elementSlots; i++)  // set desired areas for elements
    if ((theElements[i].isPresent()) && 
	(desiredArea[i] < theElements[i].getArea())){
      theElements[i].setTargetArea(desiredArea[i]);
	}
  // start refinement
  modified = 1;
  if (!refineInProgress) {
   refineInProgress = 1;
   mesh[cid].refiningElements();
  }
  CkPrintf("TMRC2D: multipleRefine DONE.\n");
}

void chunk::multipleCoarsen(double *desiredArea, refineClient *client)
{
  int i;
  CkPrintf("TMRC2D: multipleCoarsen....\n");
  theClient = client; // initialize refine client associated with this chunk
  //Uncomment this dump call to see TMRC2D's mesh config
  //dump();
  sanityCheck(); // quietly make sure mesh is in shape

  for (i=0; i<elementSlots; i++) { // set desired areas for elements
    CkPrintf("TMRC2D: desiredArea[%d]=%f present? %d area=%f\n", i, desiredArea[i], theElements[i].isPresent(), theElements[i].getArea());
    if ((theElements[i].isPresent()) &&
	(desiredArea[i] > theElements[i].getArea())) {
      theElements[i].resetTargetArea(desiredArea[i]);
      CkPrintf("TMRC2D: Setting target on element %d to %f\n", i, desiredArea[i]);
    }
  }

  // start coarsening
  modified = 1;
  if (!coarsenInProgress) {
   coarsenInProgress = 1;
   mesh[cid].coarseningElements();
  }
  CkPrintf("TMRC2D: multipleCoarsen DONE.\n");
}

void chunk::newMesh(int nEl, int nGhost, const int *conn_, const int *gid_, 
		    int idxOffset)
{
  int i, j;
  CkPrintf("TMRC2D: newMesh on chunk %d...\n", cid);
  numElements=nEl;
  numGhosts = nGhost;
  allocMesh(nEl);
  CkWaitQD();
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

  // derive edges from elements on this chunk
  deriveEdges(conn, gid);
  delete[] conn;
  delete[] gid;
  CkPrintf("TMRC2D: newMesh DONE; chunk created with %d elements.\n", 
	   numElements);
}

void chunk::deriveEdges(int *conn, int *gid)
{
  // need to add edges to the chunk, and update all edgeRefs on all elements
  // also need to add nodes to the chunk
  int i, j, n1localIdx, n2localIdx;
  edgeRef newEdge;

  deriveNodes(); // now numNodes and theNodes have values
  CkPrintf("TMRC2D: Deriving edges...\n");
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
	  newEdge = addEdge();
	  CkPrintf("TMRC2D: New edge (%d,%d) added between nodes %d and %d\n",
		   newEdge.cid, newEdge.idx, theElements[i].nodes[n1localIdx], 
		   theElements[i].nodes[n2localIdx]);
	  // point edge to the two neighboring elements
	  theEdges[newEdge.idx].update(nullRef, myRef);
	  theEdges[newEdge.idx].update(nullRef, nbrRef);
	  // point elem i's edge j at the edge
	  theElements[i].set(j, newEdge);
	  // point nbrRef at the edge
	  if (nbrRef.cid==cid) // Local neighbor
	    theElements[nbrRef.idx].set(edgeIdx, newEdge);
	  else if (nbrRef.cid != -1) // Remote neighbor
	    mesh[nbrRef.cid].addRemoteEdge(nbrRef.idx, edgeIdx, newEdge);
	}
	// else edge will be made on a different chunk
      }
    }
  }
  nodeSlots = numNodes;
  firstFreeNode = numNodes;
  edgeSlots = numEdges;
  firstFreeEdge = numEdges;
  CkPrintf("TMRC2D: Done deriving edges...\n");
}

void chunk::deriveNodes()
{
  int i, j;
  int aNode;

  CkPrintf("TMRC2D: Deriving nodes...\n");
  numNodes = 0;
  for (i=0; i<numElements; i++) {
    for (j=0; j<3; j++) {
      aNode = theElements[i].nodes[j];
      CkAssert(aNode > -1);
      if ((aNode + 1) > numNodes)
	numNodes = aNode + 1;
      theNodes[aNode].present = 1;
    }
  }
  CkPrintf("TMRC2D: NumNodes = %d; max node idx = %d\n", numNodes, numNodes-1);
  CkPrintf("TMRC2D: Done deriving nodes.\n");
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
  for (i=idx+1; i<nGhost; i++)
    if ((e = hasEdge(n1, n2, conn, i)) != -1) {
      er->set(gid[i*2], gid[i*2+1]);
      return e;
    }
  return -1;
}

int chunk::hasEdge(int n1, int n2, int *conn, int idx) 
{
  int i, j;
  for (i=0; i<3; i++)
    for (j=i+1; j<3; j++) 
      if (((conn[idx*3+i] == n1) && (conn[idx*3+j] == n2)) ||
	  ((conn[idx*3+j] == n1) && (conn[idx*3+i] == n2)))
	return i+j-1;
  return -1;
}

void chunk::deriveBorderNodes()
{
  CkPrintf("TMRC2D: WARNING! chunk::deriveBorderNodes called but not implemented!\n");
  /*
  elemRef nullRef;
  for (int i=0; i<numEdges; i++) {
    if ((theEdges[i].elements[0] == nullRef) || 
	(theEdges[i].elements[1] == nullRef)) {
      theEdges[i].nodes[0].setBorder();
      theEdges[i].nodes[1].setBorder();
    }
  }
  */
}

void chunk::tweakMesh()
{
  CkPrintf("TMRC2D: WARNING! chunk::tweakMesh called but not implemented!\n");
  /*
  for (int i=0; i<numElements; i++) 
    theElements[i].tweakNodes();
  */
}

void chunk::improveChunk()
{
  CkPrintf("TMRC2D: WARNING! chunk::improveChunk called but not implemented!\n");
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
  CkPrintf("TMRC2D: WARNING! chunk::improve called but not implemented!\n");
  /*
  for (int i=0; i<20; i++) {
    mesh.tweakMesh();
    CkWaitQD();
    mesh.improveChunk();
    CkWaitQD();
  }
  */
}

void chunk::sanityCheck(void)
{
  CkPrintf("TMRC2D: running sanity check...\n");
  int i;
  if (numElements<0 || (int)theElements.size()<numElements)
        CkAbort("-> TMRC2D: numElements or vector size insane!");
  if (numEdges<0 || (int)theEdges.size()<numEdges)
        CkAbort("-> TMRC2D: numEdges or vector size insane!");
  if (numNodes<0 || (int)theNodes.size()<numNodes)
        CkAbort("-> TMRC2D: numNodes or vector size insane!");
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
  CkPrintf("TMRC2D: sanity check PASSED.\n");
}

void chunk::dump()
{
  int i;
  CkPrintf("TMRC2D: Chunk %d, with %d elements, %d nodes and %d edges:\n", cid, 
	   numElements, numNodes, numEdges);
  for (i=0; i<elementSlots; i++) 
    if (theElements[i].isPresent())
      CkPrintf("TMRC2D: Element %d nodes %d %d %d edges [%d,%d] [%d,%d] [%d,%d]\n", i, theElements[i].nodes[0], theElements[i].nodes[1], theElements[i].nodes[2], theElements[i].edges[0].cid, theElements[i].edges[0].idx, theElements[i].edges[1].cid, theElements[i].edges[1].idx, theElements[i].edges[2].cid, theElements[i].edges[2].idx);
  for (i=0; i<numNodes; i++)
    if (theNodes[i].isPresent())
      CkPrintf("TMRC2D: Node %d has coordinates (%f,%f)\n", i, theNodes[i].X(),
	       theNodes[i].Y());
  for (i=0; i<numEdges; i++)
    if (theEdges[i].isPresent())
      CkPrintf("TMRC2D: Edge %d has elements [%d,%d] and [%d,%d]\n", i, 
	       theEdges[i].elements[0].cid, theEdges[i].elements[0].idx, 
	       theEdges[i].elements[1].cid, theEdges[i].elements[1].idx);
}

#include "refine.def.h"
