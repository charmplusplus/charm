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

chunk::chunk(chunkMsg *m)
  : TCharmClient1D(m->myThreads), numElements(0), numEdges(0), numNodes(0), 
    numGhosts(0), sizeElements(0), sizeEdges(0), sizeNodes(0),
    additions(0), debug_counter(0), refineInProgress(0), coarsenInProgress(0),
    modified(0), meshLock(0), meshExpandFlag(0), theClient(NULL)
{
  refineResultsStorage=NULL;
  cid = thisIndex;
  numChunks = m->nChunks;
  CkFreeMsg(m);
  tcharmClientInit();
  thread->resume();
}

void chunk::addNode(nodeMsg *m)
{
  theNodes[numNodes].set(m->x, m->y);
  CkFreeMsg(m);
  numNodes++;
}

void chunk::addEdge(edgeMsg *m)
{
  theEdges[numEdges].set(this);
  theEdges[numEdges].set(m->nodes[0], m->nodes[1], m->elements[0], m->elements[1]);
  CkFreeMsg(m);
  numEdges++;
}

void chunk::addRemoteEdge(remoteEdgeMsg *m)
{
  theElements[m->elem].set(m->localEdge, m->er);
  CkFreeMsg(m);
}

void chunk::addElement(elementMsg *m)
{
  theElements[numElements].set(cid, numElements, this);
  theElements[numElements].set(m->nodes, m->edges);
  CkFreeMsg(m);
  numElements++;
}

void chunk::refineElement(refineMsg *m)
{
  // we indicate a need for refinement by reducing an element's targetArea
  theElements[m->idx].setTargetArea(m->area);
  CkFreeMsg(m);
  modified = 1;  // flag a change in one of the chunk's elements
  if (!refineInProgress) { // if refine loop not running
    refineInProgress = 1;
    mesh[cid].refiningElements(); // start it up
  }
}

void chunk::refiningElements()
{
  int i;

  CkPrintf("Refining...\n");
  while (modified) { 
    // continue trying to refine elements until nothing changes during
    // a refinement cycle
    i = 0;
    modified = 0;
    while (i < numElements) { // loop through the elements
      if ((theElements[i].getTargetArea() <= theElements[i].getArea()) 
	    && (theElements[i].getTargetArea() >= 0.0)) {
	// the element either needs refining or has been asked to
	// refine or has asked someone else to refine
	modified = 1; // something's bound to change
	theElements[i].refine(); // refine the element
	adjustMesh();
      }
      i++;
    }
    //    if (CkMyPe() == 0) for (int j=0; j<5; j++) mesh[j].print();
    CthYield(); // give other chunks on the same PE a chance
  }
  // nothing is in need of refinement; turn refine loop off
  refineInProgress = 0;  
  CkPrintf("Refining done.\n");
}


// This initiates a coarsening for a single element
void chunk::coarsenElement(coarsenMsg *m)
{
  // we indicate a need for coarsening by increasing an element's targetArea
  if (!theElements[m->idx].isPresent()) return;
  theElements[m->idx].resetTargetArea(m->area);
  CkFreeMsg(m);
  modified = 1;  // flag a change in one of the chunk's elements
  if (!coarsenInProgress) { // if coarsen loop not running
    coarsenInProgress = 1;
    mesh[cid].coarseningElements(); // start it up
  }
}

// This loops through all elements performing coarsenings as needed
void chunk::coarseningElements()
{
  int i;

  CkPrintf("Coarsening...\n");
  while (modified) { // try to coarsen elements until no changes occur
    i = 0;
    modified = 0;
    while (i < numElements) { // loop through the elements
      if (theElements[i].isPresent() && 
	  ((theElements[i].getTargetArea() > theElements[i].getArea()) 
	   && (theElements[i].getTargetArea() >= 0.0))) {
	// element i needs coarsening
	modified = 1; // something's bound to change
	theElements[i].coarsen(); // coarsen the element
	//if (CkMyPe() == 0) for (int j=0; j<5; j++) mesh[j].print();
      }
      i++;
    }
    CthYield(); // give other chunks on the same PE a chance
  }
  coarsenInProgress = 0;  // turn coarsen loop off
  CkPrintf("Coarsening done.\n");
}

// many remote access methods follow
nodeMsg *chunk::getNode(intMsg *im)
{
  nodeMsg *nm = new nodeMsg;
  nm->x = theNodes[im->anInt].X();
  nm->y = theNodes[im->anInt].Y();
  CkFreeMsg(im);
  return nm;
}

refMsg *chunk::getEdge(collapseMsg *cm)
{
  refMsg *rm = new refMsg;
  rm->aRef = theElements[cm->idx].getEdge(cm->er, cm->nr1);
  CkFreeMsg(cm);
  return rm;
}

void chunk::setBorder(intMsg *im)
{
  theNodes[im->anInt].setBorder();
  CkFreeMsg(im);
}

intMsg *chunk::safeToMoveNode(nodeMsg *nm)
{
  node foo(nm->x, nm->y);
  intMsg *im = new intMsg;
  im->anInt = theNodes[nm->idx].safeToMove(foo);
  CkFreeMsg(nm);
  return im;
}

splitOutMsg *chunk::split(splitInMsg *sim)
{
  splitOutMsg *som = new splitOutMsg;
  som->result = theEdges[sim->idx].split(&(som->n), &(som->e), sim->n, 
					   sim->e);
  CkFreeMsg(sim);
  return som;
}

void chunk::collapseHelp(collapseMsg *cm)
{
  theElements[cm->idx].collapseHelp(cm->er, cm->nr1, cm->nr2);
  CkFreeMsg(cm);
}

void chunk::checkPending(refMsg *rm)
{
  elemRef eRef;
  eRef.idx = rm->aRef.idx; eRef.cid = rm->aRef.cid;
  theEdges[rm->idx].checkPending(eRef);
  CkFreeMsg(rm);
}

void chunk::checkPending(drefMsg *rm)
{
  elemRef eRef1, eRef2;
  eRef1.idx = rm->aRef1.idx; eRef1.cid = rm->aRef1.cid;
  eRef2.idx = rm->aRef2.idx; eRef2.cid = rm->aRef2.cid;
  theEdges[rm->idx].checkPending(eRef1, eRef2);
  CkFreeMsg(rm);
}

void chunk::updateNode(updateMsg *um)
{
  nodeRef ov, nv;
  ov.idx = um->oldval.idx;   ov.cid = um->oldval.cid; 
  nv.idx = um->newval.idx;   nv.cid = um->newval.cid; 
  theEdges[um->idx].update(ov, nv);
  CkFreeMsg(um);
}

void chunk::updateElement(updateMsg *um)
{
  elemRef ov, nv;
  ov.idx = um->oldval.idx;   ov.cid = um->oldval.cid; 
  nv.idx = um->newval.idx;   nv.cid = um->newval.cid; 
  theEdges[um->idx].update(ov, nv);
  CkFreeMsg(um);
}

void chunk::updateElementEdge(updateMsg *um)
{
  edgeRef ov, nv;
  ov.idx = um->oldval.idx;   ov.cid = um->oldval.cid; 
  nv.idx = um->newval.idx;   nv.cid = um->newval.cid; 
  theElements[um->idx].update(ov, nv);
  CkFreeMsg(um);
}

void chunk::updateReferences(updateMsg *um)
{
  int i;
  nodeRef ov, nv;
  ov.idx = um->oldval.idx;   ov.cid = um->oldval.cid; 
  nv.idx = um->newval.idx;   nv.cid = um->newval.cid;
  for (i=0; i<numElements; i++)
    theElements[i].update(ov, nv);
  for (i=0; i<numEdges; i++)
    theEdges[i].updateSilent(ov, nv);
  CkFreeMsg(um);
}

doubleMsg *chunk::getArea(intMsg *im)
{
  doubleMsg *dm = new doubleMsg;
  accessLock();
  dm->aDouble = theElements[im->anInt].getArea();
  CkFreeMsg(im);
  releaseLock();
  return dm;
}

nodeMsg *chunk::midpoint(intMsg *im)
{
  nodeMsg *nm = new nodeMsg;
  node result;
  accessLock();
  theEdges[im->anInt].midpoint(result);
  CkFreeMsg(im);
  releaseLock();
  nm->x = result.X();
  nm->y = result.Y();
  return nm;
}

intMsg *chunk::setPending(intMsg *im)
{
  intMsg *rm = new intMsg;
  if (theEdges[im->anInt].isPending())
    rm->anInt = 0;
  else {
    theEdges[im->anInt].setPending();
    rm->anInt = 1;
  }
  CkFreeMsg(im);
  return rm;
}

void chunk::unsetPending(intMsg *im)
{
  theEdges[im->anInt].unsetPending();
  CkFreeMsg(im);
}


intMsg *chunk::isPending(intMsg *im)
{
  intMsg *rm = new intMsg;
  rm->anInt = theEdges[im->anInt].isPending();
  CkFreeMsg(im);
  return rm;
}

intMsg *chunk::lockNode(intMsg *im)
{
  intMsg *rm = new intMsg;
  rm->anInt = theNodes[im->anInt].lock();
  CkFreeMsg(im);
  return rm;
}

void chunk::unlockNode(intMsg *im)
{
  theNodes[im->anInt].unlock();
  CkFreeMsg(im);
}

intMsg *chunk::isLongestEdge(refMsg *rm)
{
  intMsg *im = new intMsg;
  edgeRef eRef;

  accessLock();
  eRef.idx = rm->aRef.idx; eRef.cid = rm->aRef.cid;
  im->anInt = theElements[rm->idx].isLongestEdge(eRef);
  CkFreeMsg(rm);
  releaseLock();
  return im;
}

refMsg *chunk::getNeighbor(refMsg *gm)
{
  refMsg *rm = new refMsg;
  elemRef er, ar;
  ar.cid = gm->aRef.cid; ar.idx = gm->aRef.idx;
  er = theEdges[gm->idx].getNot(ar);
  CkFreeMsg(gm);
  rm->aRef = er;
  return rm;
}

refMsg *chunk::getNotNode(refMsg *gm)
{
  refMsg *rm = new refMsg;
  nodeRef er, ar;
  ar.cid = gm->aRef.cid; ar.idx = gm->aRef.idx;
  er = theEdges[gm->idx].getNot(ar);
  CkFreeMsg(gm);
  rm->aRef = er;
  return rm;
}

refMsg *chunk::getNotElem(refMsg *gm)
{
  refMsg *rm = new refMsg;
  elemRef er, ar;
  ar.cid = gm->aRef.cid; ar.idx = gm->aRef.idx;
  er = theEdges[gm->idx].getNot(ar);
  CkFreeMsg(gm);
  rm->aRef = er;
  return rm;
}

void chunk::setTargetArea(doubleMsg *dm)
{
  theElements[dm->idx].setTargetArea(dm->aDouble);
  CkFreeMsg(dm);
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

void chunk::resetTargetArea(doubleMsg *dm)
{
  theElements[dm->idx].resetTargetArea(dm->aDouble);
  CkFreeMsg(dm);
  modified = 1;
}

refMsg *chunk::getOpposingNode(refMsg *m)
{
  refMsg *rm = new refMsg;
  nodeRef nRef;
  edgeRef eRef;
  eRef.idx = m->aRef.idx;  eRef.cid = m->aRef.cid;
  nRef = theElements[m->idx].getOpnode(eRef);
  CkFreeMsg(m);
  rm->aRef = nRef;
  return rm;
}

void chunk::updateEdges(edgeUpdateMsg *em)
{
  theElements[em->idx].set(em->e0, em->e1, em->e2);
  CkFreeMsg(em);
}

void chunk::updateNodeCoords(nodeMsg *m)
{
  theNodes[m->idx].set(m->x, m->y);
  CkFreeMsg(m);
}

void chunk::reportPos(nodeMsg *m)
{
  node z(m->x, m->y);
  theNodes[m->idx].reportPos(z);
  CkFreeMsg(m);
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

void chunk::allocMesh(int nEl)
{
  sizeElements = nEl * 2;
  sizeNodes = sizeEdges = sizeElements * 3;
  theElements.resize(sizeElements);
  theNodes.resize(sizeNodes);
  theEdges.resize(sizeEdges);
  for (int i=0; i<sizeElements; i++)
    theElements[i].set(); 
  conn = new int[3*numGhosts];
  gid = new int[2*numGhosts];
}

void chunk::adjustMesh()
{
  if (sizeElements <= numElements+100) {
    adjustFlag();
    adjustLock();
    CkPrintf("[%d] Adjusting mesh size...\n", cid);
    sizeElements += 100;
    sizeEdges += 300;
    sizeNodes += 300;
    theElements.resize(sizeElements);
    theEdges.resize(sizeEdges);
    theNodes.resize(sizeNodes);
    CkPrintf("[%d] Done adjusting mesh size...\n", cid);
    adjustRelease();
  }
}

nodeRef *chunk::addNode(node& n)
{
  nodeRef *nRef = new nodeRef(cid, numNodes);
  theNodes[numNodes] = n;
  numNodes++;
  return nRef;
}


edgeRef *chunk::addEdge(nodeRef& nr1, nodeRef& nr2)
{
  edgeRef *eRef = new edgeRef(cid, numEdges);
  nodeRef n[2] = {nr1, nr2};
  elemRef e[2];

  theEdges[numEdges].set(this);
  theEdges[numEdges].set(n, e);
  numEdges++;
  return eRef;
}

elemRef *chunk::addElement(nodeRef& nr1, nodeRef& nr2, nodeRef& nr3)
{
  elemRef *eRef = new elemRef(cid, numElements);
  nodeRef n[3] = {nr1, nr2, nr3};

  theElements[numElements].set(cid, numElements, this);
  theElements[numElements].set(n);
  theElements[numElements].calculateArea();
  numElements++;
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
  return eRef;
}

elemRef *chunk::addElement(nodeRef& nr1, nodeRef& nr2, nodeRef& nr3,
			   edgeRef& er1, edgeRef& er2, edgeRef& er3)
{
  elemRef *eRef = new elemRef(cid, numElements);
  nodeRef n[3] = {nr1, nr2, nr3};
  edgeRef e[3] = {er1, er2, er3}; 

  theElements[numElements].set(cid, numElements, this);
  theElements[numElements].set(n, e);
  theElements[numElements].calculateArea();
  numElements++;
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
  return eRef;
}

void chunk::removeNode(intMsg *im)
{
  theNodes[im->anInt].reset();
  CkFreeMsg(im);
}

void chunk::removeEdge(intMsg *im)
{
  CkFreeMsg(im);
}

void chunk::removeElement(intMsg *im)
{
  theElements[im->anInt].clear();
  CkFreeMsg(im);
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
	n = theElements[i].getNode(j).get();
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
    fprintf(fp, " %d %d ", theEdges[i].nodes[0].idx, theEdges[i].nodes[0].cid);
    fprintf(fp, " %d %d ", theEdges[i].nodes[1].idx, theEdges[i].nodes[1].cid);
    fprintf(fp, "   ");
    fprintf(fp, " %d %d ", theEdges[i].elements[0].idx, theEdges[i].elements[0].cid);
    fprintf(fp, " %d %d\n", theEdges[i].elements[1].idx, theEdges[i].elements[1].cid);
  }
  for (i=0; i<numElements; i++) {
    if (theElements[i].isPresent()) {
      fprintf(fp, " %d %d ", theElements[i].nodes[0].idx, theElements[i].nodes[0].cid);
      fprintf(fp, " %d %d ", theElements[i].nodes[1].idx, theElements[i].nodes[1].cid);
      fprintf(fp, " %d %d ", theElements[i].nodes[2].idx, theElements[i].nodes[2].cid);
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
  sanityCheck();
  
  // update node coordinates from coord
  for (i=0; i<numNodes; i++)
    theNodes[i].set(coord[2*i], coord[2*i + 1]);
    
  // recalculate and cache new areas for each element
  for (i=0; i<numElements; i++) theElements[i].calculateArea();
  CkPrintf("TMRC2D: updateNodeCoords DONE.\n");
}

void chunk::multipleRefine(double *desiredArea, refineClient *client)
{
  int i;
  CkPrintf("TMRC2D: multipleRefine....\n");
  theClient = client; // initialize refine client associated with this chunk
  sanityCheck();

  // set desired areas for elements
  for (i=0; i<numElements; i++)
    if (desiredArea[i] < theElements[i].getArea())
      theElements[i].setTargetArea(desiredArea[i]);
  
  // start the refinement loop
  modified = 1;
  refineInProgress = 1;
  mesh[cid].refiningElements();
  CkPrintf("TMRC2D: multipleRefine DONE.\n");
}


// check this for kosherness....
void chunk::newMesh(int nEl, int nGhost, const int *conn_, const int *gid_, int idxOffset)
{
  int i, j;
  numElements=nEl;
  numGhosts = nGhost;
  allocMesh(nEl);
  int *conn = new int[3*numGhosts];
  int *gid = new int[2*numGhosts];

  CkPrintf("TMRC2D: newMesh...\n");
  // add elements to chunk
  for (i=0; i<numElements; i++) {
    nodeRef nodes[3];
    for (j=0; j<3; j++) {
      int c=conn_[i*3+j]-idxOffset;
      conn[i*3 + j]=c;
      nodes[j].set(cid, c);
    }
    theElements[i].set(cid, i, this);
    theElements[i].set(nodes);
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
  sanityCheck();
  delete[] conn;
  delete[] gid;
  CkPrintf("TMRC2D: newMesh DONE.\n");
}

void chunk::deriveEdges(int *conn, int *gid)
{
  // need to add edges to the chunk, and update all edgeRefs on all elements
  // also need to add nodes to the chunk
  int i, j, n1, n2, e, newEdge;
  elemRef nullRef, myRef, nbrRef;
  edgeRef er;

  deriveNodes(); // now numNodes and theNodes have values
  for (i=0; i<numElements; i++) {
    myRef.set(cid, i);
    for (j=0; j<3; j++) {
      // get endpoints n1 and n2 for edge j
      if (j != 2) n1 = conn[i*3];
      else n1 = conn[i*3 + 1];
      if (j != 0) n2 = conn[i*3 + 2];
      else n2 = conn[i*3 + 1];
      
      CkAssert(n1 >-1);
      CkAssert(n2 >-1);

      // look for edge (n1, n2) 
      if ((e = findEdge(n1, n2)) != -1) { 
	// an edge between n1 & n2 has been added already at theEdges[e]
	theEdges[e].update(nullRef, myRef); // point edge at elem i
	// point elem i's edge j at the edge
	er.cid = cid; er.idx = e;
	theElements[i].set(j, er);
      }
      else { // no local edge yet exists
	// get nbr ref
	int edgeIdx = getNbrRefOnEdge(n1, n2, conn, numGhosts, gid, i, &nbrRef); 
	if (edgeLocal(myRef, nbrRef)) { // make edge here
	  newEdge = addNewEdge(n1, n2);
	  // point edge to the two neighboring elements
	  theEdges[newEdge].update(nullRef, myRef);
	  theEdges[newEdge].update(nullRef, nbrRef);
	  // point elem i's edge j at the edge
	  er.cid = cid; er.idx = newEdge;
	  theElements[i].set(j, er);
	  // if not on border, point nbrRef at the edge
	  if (nbrRef.cid != -1) {
	    remoteEdgeMsg *rem = new remoteEdgeMsg;
	    rem->elem = nbrRef.cid;
	    rem->er = er;
	    rem->localEdge = edgeIdx;
	    mesh[nbrRef.cid].addRemoteEdge(rem);
	  }
	}
	// else edge will be made on a different chunk
      }
    }
  }
}

void chunk::deriveNodes()
{
  int i, j;
  nodeRef nr;

  numNodes = 0;
  for (i=0; i<numElements; i++) {
    for (j=0; j<3; j++) {
      nr = theElements[i].getNode(j);
      if ((nr.idx + 1) > numNodes)
	numNodes = nr.idx + 1;
    }
  }
  CkPrintf("NumNodes = %d; max node idx = %d\n", numNodes, numNodes-1);
}

int chunk::edgeLocal(elemRef e1, elemRef e2)
{
  return ((e1.cid == e2.cid) || (e1.idx >= e2.idx));
}

int chunk::findEdge(int n1, int n2)
{
  int i;
  nodeRef nr[2];

  for (i=0; i<numEdges; i++) {
    nr[0] = theEdges[i].get(0);
    nr[1] = theEdges[i].get(1);
    if (((nr[0].idx == n1) && (nr[1].idx == n2)) ||
	((nr[1].idx == n1) && (nr[0].idx == n2)))
      return i;
  }
  return -1;
}

int chunk::addNewEdge(int n1, int n2)
{
  nodeRef n[2];
  n[0].set(cid, n1);   n[1].set(cid, n2);
  theEdges[numEdges].set(this);
  theEdges[numEdges].set(n);
  numEdges++;
  return numEdges-1;
}

int chunk::getNbrRefOnEdge(int n1, int n2, int *conn, int nGhost, int *gid, 
			   int idx, elemRef *er)
{
  int i, e;
  
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

void chunk::freshen()
{
  for (int i=0; i<numElements; i++) theElements[i].resetTargetArea(-1.0);
}

void chunk::deriveBorderNodes()
{
  elemRef nullRef;

  for (int i=0; i<numEdges; i++) {
    if ((theEdges[i].elements[0] == nullRef) || 
	(theEdges[i].elements[1] == nullRef)) {
      CkPrintf("Edge %d on chunk %d is on border; node %d on chunk %d and node %d on chunk %d also on border\n", i, thisIndex, theEdges[i].nodes[0].idx, theEdges[i].nodes[0].cid, theEdges[i].nodes[1].idx, theEdges[i].nodes[1].cid);
      theEdges[i].nodes[0].setBorder();
      theEdges[i].nodes[1].setBorder();
    }
  }
}

void chunk::tweakMesh()
{
  for (int i=0; i<numElements; i++) 
    theElements[i].tweakNodes();
}

void chunk::improveChunk()
{
  for (int i=0; i<numNodes; i++) 
    if (!theNodes[i].border) {
      theNodes[i].improvePos();
    }
    else CkPrintf("Not adjusting node %d on chunk %d\n", i, thisIndex);

}

void chunk::improve()
{
  for (int i=0; i<20; i++) {
    mesh.tweakMesh();
    CkWaitQD();
    mesh.improveChunk();
    CkWaitQD();
  }
}

void chunk::sanityCheck(void)
{
  int i;
  CkPrintf("TMRC2D: running sanity check...\n");
  if (numElements<0 || (int)theElements.size()<numElements)
        CkAbort("-> TMRC2D: numElements or vector size insane!");
  if (numEdges<0 || (int)theEdges.size()<numEdges)
        CkAbort("-> TMRC2D: numEdges or vector size insane!");
  if (numNodes<0 || (int)theNodes.size()<numNodes)
        CkAbort("-> TMRC2D: numNodes or vector size insane!");
  for (i=0;i<numElements;i++)
    theElements[i].sanityCheck(this,elemRef(cid,i));
  for (i=0;i<numEdges;i++)
    theEdges[i].sanityCheck(this,edgeRef(cid,i));
  CkPrintf("TMRC2D: sanity check PASSED.\n");
}

#include "refine.def.h"
