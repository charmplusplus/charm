// Parallel Mesh Adaptivity Framework - 3D
// Created by: Terry L. Wilmarth

#include <stdlib.h>
#include <stdio.h>
#include "chunk.h"

//readonlys
CProxy_chunk mesh;
CtvDeclare(chunk *, _refineChunk);

void refineChunkInit(void) {
  CtvInitialize(chunk *, _refineChunk);
}

static elemRef nullRef(-1,-1);

// chunk methods
chunk::chunk(int nChunks)
  : numElements(0), numNodes(0), sizeElements(0), sizeNodes(0),
    coordsRecvd(0), debug_counter(0), refineInProgress(0),
    modified(0), accessLock(0), adjustLock(0), lock(0), lockCount(0),
    lockHolder(-1), lockPrio(-1.0), smoothness(0.25), lockList(NULL), theClient(NULL)
  //TCharmClient1D(m->myThreads), 
{
  refineResultsStorage=NULL;
  cid = thisIndex;
  numChunks = nChunks;
  //tcharmClientInit();
  //thread->ready();
}

void chunk::refineElement(int idx, double volume)
{
  // we indicate a need for refinement by reducing an element's targetVolume
  CmiAssert((idx < numElements) && (idx >= 0));
  CmiAssert((volume >= 0.0) || (volume == -1.0));

  theElements[idx].setTargetVolume(volume);
  modified = 1;  // flag a change in one of the chunk's elements
  if (!refineInProgress) { // if refine loop not running
    refineInProgress = 1;
    CkPrintf("Chunk %d activating...\n", cid);
    mesh[cid].refiningElements(); // start it up
  }
}

void chunk::refineElement(int idx)
{
  double vol = theElements[idx].getTargetVolume();
  CmiAssert((idx < numElements) && (idx >= 0));
  if ((vol == -1.0) || (vol >= theElements[idx].getVolume()))
    vol = theElements[idx].getVolume();
  theElements[idx].setTargetVolume(vol);
  modified = 1;  // flag a change in one of the chunk's elements
  if (!refineInProgress) { // if refine loop not running
    refineInProgress = 1;
    CkPrintf("Chunk %d activating...\n", cid);
    mesh[cid].refiningElements(); // start it up
  }
}

void chunk::refiningElements()
{
  int i;

  while (modified) { 
    // continue trying to refine elements until nothing changes
    i = 0;
    modified = 0;
    //CkPrintf("Chunk %d in refiningElements loop\n", cid);
    while (i < numElements) { // loop through the elements
      if ((theElements[i].getTargetVolume() <= theElements[i].getVolume()) 
	  && (theElements[i].getTargetVolume() >= 0.0)) {
	// the element needs refining
	modified = 1; // something's bound to change
	if (theElements[i].LEtest())
	  theElements[i].refineLE(); // refine the element
        else if (theElements[i].LFtest())
	  theElements[i].refineLF(); // refine the element
	else if (theElements[i].CPtest())
	  theElements[i].refineCP(); // refine the element
	else theElements[i].refineLE(); // refine the element
      }
      i++;
      adjustMesh();
    }
    CthYield(); // give other chunks on the same PE a chance
  }
  // nothing is in need of refinement; turn refine loop off
  if (CkMyPe() == 0) for (int j=0; j<numChunks; j++) mesh[j].print();
  refineInProgress = 0;  
  CkPrintf("Chunk %d going inactive...\n", cid);
}


// This initiates a coarsening for a single element
void chunk::coarsenElement(int idx, double volume)
{
  // we indicate a need for coarsening by increasing an element's targetVolume
  CmiAssert((idx < numElements) && (idx >= 0));
  CmiAssert((volume >= 0.0) || (volume == -1.0));
  if (!theElements[idx].isPresent()) return;
  theElements[idx].resetTargetVolume(volume);
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

  while (modified) { // try to coarsen elements until no changes occur
    i = 0;
    modified = 0;
    while (i < numElements) { // loop through the elements
      if (theElements[i].isPresent() && 
	  ((theElements[i].getTargetVolume() > theElements[i].getVolume()) 
	   && (theElements[i].getTargetVolume() >= 0.0))) {
	// element i needs coarsening
	modified = 1; // something's bound to change
	theElements[i].coarsen(); // coarsen the element
	if (CkMyPe() == 0) for (int j=0; j<5; j++) mesh[j].print();
      }
      i++;
    }
    CthYield(); // give other chunks on the same PE a chance
  }
  coarsenInProgress = 0;  // turn coarsen loop off
}

void chunk::improveMesh()
{
  for (int i=0; i<numElements; i++)
    theElements[i].improveElement();
}

void chunk::relocatePoints() 
{
  for (int i=0; i<numNodes; i++)
    theNodes[i].relocateNode();
}

void chunk::flippingElements() 
{
  int edge[2] = {2, 3};
  //int face[3] = {0, 1, 2};
  if (cid == 0) 
    theElements[0].flip32(edge);
    //theElements[0].flip23(face);
}

intMsg *chunk::lockChunk(int lh, double prio)
{
  intMsg *im = new intMsg;
  im->anInt = lockLocalChunk(lh, prio);
  return im;
}

int chunk::lockLocalChunk(int lh, double prio)
{ // priority is given to the higher prio, lower lh
  // each chunk can have at most one lock reservation
  // if a chunk holds a lock, it cannot have a reservation
  if (!lock) { // this chunk is UNLOCKED
    // remove previous lh reservation; insert this reservation
    // then check first reservation
    removeLock(lh);
    insertLock(lh, prio);
    if ((lockList->prio == prio) && (lockList->holder == lh)) {
      prioLockStruct *tmp = lockList;
      //CkPrintf("[%d]LOCK chunk %d by %d prio %.10f LOCKED(was %d[%d:%.10f])\n",
      //	       CkMyPe(), cid, lh, prio, lock, lockHolder, lockPrio);
      lock = 1; lockHolder = lh; lockPrio = prio; lockCount = 1;
      lockList = lockList->next;
      free(tmp);
      return 1;
    }
    removeLock(lh);
    //    CkPrintf("[%d]LOCK chunk %d by %d prio %.10f REFUSED (was %d[%d:%.10f])\n",
    //	     CkMyPe(), cid, lh, prio, lock, lockHolder, lockPrio);
    return 0;
  }
  else { // this chunk is LOCKED
    if ((prio == lockPrio) && (lh == lockHolder)) {
      lockCount++;
      return 1;
    }
    if (lh == lockHolder) {
      CkPrintf("ERROR: chunk holding lock trying to relock with different prio! lockHolder=%d prio=%f newprio=%f\n", lockHolder, lockPrio, prio);
      CmiAssert(lh != lockHolder);
    }
    removeLock(lh);
    insertLock(lh, prio);
    if ((lockList->prio == prio) && (lockList->holder == lh)) {
      if ((prio > lockPrio) ||
	  ((prio == lockPrio) && (lh < lockHolder))) { // lh might be next
	//	CkPrintf("[%d]LOCK chunk %d by %d prio %.10f RESERVED (was %d[%d:%.10f])\n",
	//		 CkMyPe(), cid, lh, prio, lock, lockHolder, lockPrio);
	return -1; 
      }
    }
    removeLock(lh);
    //    CkPrintf("[%d]LOCK chunk %d by %d prio %.10f REFUSED (was %d[%d:%.10f])\n",
    //	     CkMyPe(), cid, lh, prio, lock, lockHolder, lockPrio);
    return 0;
  }
}

void chunk::insertLock(int lh, double prio)
{
  prioLockStruct *newLock, *tmp;
  
  newLock = (prioLockStruct *)malloc(sizeof(prioLockStruct));
  newLock->prio = prio;
  newLock->holder = lh;
  newLock->next = NULL;
  if (!lockList) lockList = newLock;
  else {
    if ((prio > lockList->prio) || 
	(prio == lockList->prio) && (lh < lockList->holder)) {
      // insert before first element in lockList
      newLock->next = lockList;
      lockList = newLock;
    }
    else { 
      tmp = lockList;
      while (tmp->next) {
	if ((prio < tmp->next->prio) || ((prio == tmp->next->prio) &&
					 (lh > tmp->next->holder)))
	  tmp = tmp->next;
	else break;
      }
      newLock->next = tmp->next;
      tmp->next = newLock;
    }
  }
}

void chunk::removeLock(int lh) 
{
  prioLockStruct *tmp = lockList, *del;
  if (!lockList) return;
  if (tmp->holder == lh) {
    lockList = lockList->next;
    free(tmp);
  }
  else {
    while (tmp->next && (tmp->next->holder != lh))
      tmp = tmp->next;
    if (tmp->next) {
      del = tmp->next;
      tmp->next = del->next;
      free(del);
    }
  }
}

void chunk::unlockChunk(int lh)
{
  unlockLocalChunk(lh);
}

void chunk::unlockLocalChunk(int lh)
{
  if (lockHolder == lh) {
    lockCount--;
    if (lockCount == 0) {
      lock = 0; 
      lockHolder = -1;
      lockPrio = -1.0;
      //CkPrintf("[%d]UNLOCK chunk %d by holder %d\n", CkMyPe(), cid, lh);
    }
  }
}

// these two functions produce debugging output by printing somewhat
// sychronized versions of the entire mesh to files readable by tkplotter
void chunk::print()
{
  getAccessLock();
  debug_print(debug_counter);
  debug_counter++;
  releaseAccessLock();
}

void chunk::out_print()
{ // THIS FUNCTION IS OUT-OF-DATE AND MUST BE REWRITTEN
  FILE *fp;
  char filename[30];
  int i;

  memset(filename, 0, 30);
  sprintf(filename, "mesh.out");
  fp = fopen(filename, "a");

  if (cid == 0)
    fprintf(fp, "%d\n", numChunks);
  fprintf(fp, " %d %d %d\n", cid, numNodes, numElements);
  for (i=0; i<numNodes; i++)
    fprintf(fp, "    %f %f %f\n", theNodes[i].getCoord(0), theNodes[i].getCoord(1), theNodes[i].getCoord(2));
  for (i=0; i<numElements; i++) {
    if (theElements[i].isPresent()) {
      fprintf(fp, " %d %d ", theElements[i].nodes[0].idx, theElements[i].nodes[0].cid);
      fprintf(fp, " %d %d ", theElements[i].nodes[1].idx, theElements[i].nodes[1].cid);
      fprintf(fp, " %d %d ", theElements[i].nodes[2].idx, theElements[i].nodes[2].cid);
      fprintf(fp, " %d %d ", theElements[i].nodes[3].idx, theElements[i].nodes[3].cid);
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
}

// entries to node data
nodeMsg *chunk::getNode(int n)
{
  CmiAssert((n < numNodes) && (n >= 0));
  nodeMsg *nm = new nodeMsg;
  nm->coord[0] = theNodes[n].getCoord(0);
  nm->coord[1] = theNodes[n].getCoord(1);
  nm->coord[2] = theNodes[n].getCoord(2);
  return nm;
}

void chunk::updateNodeCoord(nodeMsg *m)
{
  CmiAssert((m->idx < numNodes) && (m->idx >= 0));
  theNodes[m->idx].set(m->coord[0], m->coord[1], m->coord[2]);
  CkFreeMsg(m);
}

void chunk::relocationVote(nodeVoteMsg *m)
{
  node oldNode(m->oldCoord[0], m->oldCoord[1], m->oldCoord[2]);
  node newNode(m->newCoord[0], m->newCoord[1], m->newCoord[2]);
  
  for (int i=0; i<numNodes; i++) 
    if (theNodes[i] == oldNode) {
      theNodes[i].relocationVote(newNode);
      return;
    }
  CkFreeMsg(m);
}

// entries to element data
doubleMsg *chunk::getVolume(intMsg *im)
{
  CmiAssert((im->anInt < numElements) && (im->anInt >= 0));
  doubleMsg *dm = new doubleMsg;
  dm->aDouble = theElements[im->anInt].getVolume();
  CkFreeMsg(im);
  return dm;
}

void chunk::setTargetVolume(doubleMsg *dm)
{
  CmiAssert((dm->idx < numElements) && (dm->idx >= 0));
  CmiAssert((dm->aDouble >= 0.0) || (dm->aDouble == -1.0));
  theElements[dm->idx].setTargetVolume(dm->aDouble);
  CkFreeMsg(dm);
  modified = 1;
  if (!refineInProgress) {
    refineInProgress = 1;
    mesh[cid].refiningElements();
  }
}

void chunk::resetTargetVolume(doubleMsg *dm)
{
  CmiAssert((dm->idx < numElements) && (dm->idx >= 0));
  CmiAssert((dm->aDouble >= 0.0) || (dm->aDouble == -1.0));
  theElements[dm->idx].resetTargetVolume(dm->aDouble);
  CkFreeMsg(dm);
  modified = 1;
}

flip23response *chunk::flip23remote(flip23request *fr)
{
  CmiAssert((fr->requestee < numElements) && (fr->requestee >= 0));
  flip23response *f23r;
  getAccessLock();
  f23r = theElements[fr->requestee].flip23remote(fr);
  releaseAccessLock();
  return f23r;
}

flip32response *chunk::flip32remote(flip32request *fr)
{
  CmiAssert((fr->requestee < numElements) && (fr->requestee >= 0));
  flip32response *f32r;
  getAccessLock();
  f32r = theElements[fr->requestee].flip32remote(fr);
  releaseAccessLock();
  return f32r;
}

flip32response *chunk::remove32element(flip32request *fr)
{
  CmiAssert((fr->requestee < numElements) && (fr->requestee >= 0));
  flip32response *f32r;
  getAccessLock();
  f32r = theElements[fr->requestee].remove32element(fr);
  releaseAccessLock();
  return f32r;
}

intMsg *chunk::checkFace(int idx, elemRef face)
{
  intMsg *result = new intMsg;
  getAccessLock();
  result->anInt = theElements[idx].hasFace(face);
  releaseAccessLock();
  return result;
}

intMsg *chunk::checkFace(int idx, node n1, node n2, node n3, elemRef nbr)
{
  intMsg *im = new intMsg;
  getAccessLock();
  im->anInt = theElements[idx].checkFace(n1, n2, n3, nbr);
  releaseAccessLock();
  return im;
}

// local methods

void chunk::debug_print(int c)
{
  FILE *fp;
  char filename[30];
  int i;

  memset(filename, 0, 30);
  sprintf(filename, "dbg_msh%d.%d", cid, c);
  fp = fopen(filename, "w");


  for (i=0; i<numNodes; i++) {
    fprintf(fp, "%lf %lf %lf  ", theNodes[i].getCoord(0), 
	    theNodes[i].getCoord(1), theNodes[i].getCoord(2));
    /*    if (theNodes[i].isFixed())
      fprintf(fp, "1.0 0.5 0.75\n");
    else if (theNodes[i].onSurface())
      fprintf(fp, "1.0 1.0 0.5\n");
    else fprintf(fp, "0.5 0.75 1.0\n");
    */
    if (i % 3 == 0)
      fprintf(fp, "0.0 0.0 0.1\n");
    else if (i % 3 == 1)
      fprintf(fp, "0.0 0.0 0.1\n");
    else fprintf(fp, "0.0 0.0 0.1\n");
  }
  fprintf(fp, "foo\n");
  for (i=0; i<numElements; i++) {
    if (theElements[i].present)
      fprintf(fp, "%d %d %d  %d %d %d  %d %d %d  %d %d %d\n", 
	      theElements[i].nodes[0].idx, theElements[i].nodes[1].idx, 
	      theElements[i].nodes[2].idx,
	      theElements[i].nodes[0].idx, theElements[i].nodes[1].idx, 
	      theElements[i].nodes[3].idx,
	      theElements[i].nodes[0].idx, theElements[i].nodes[2].idx, 
	      theElements[i].nodes[3].idx,
	      theElements[i].nodes[1].idx, theElements[i].nodes[2].idx, 
	      theElements[i].nodes[3].idx);
  }
  fclose(fp);
}

// helpers
elemRef chunk::findNeighbor(nodeRef nr1, nodeRef nr2, nodeRef nr3, int lidx)
{
  int i;
  refMsg *aResult;
  elemRef theResult(-1, -1);
	   
  for (i=0; i<numElements; i++)
    if (i != lidx)
      if (theElements[i].hasNodes(nr1, nr2, nr3)) {
	theResult.cid = cid;
	theResult.idx = i;
	return theResult;
      }
  // nothing found locally; check remote chunks
  for (i=0; i<numChunks; i++)
    if (i != cid) {
      threeNodeMsg *tnm = new threeNodeMsg;
      tnm->coords[0][0] = theNodes[nr1.idx].getCoord(0);
      tnm->coords[0][1] = theNodes[nr1.idx].getCoord(1);
      tnm->coords[0][2] = theNodes[nr1.idx].getCoord(2);
      tnm->coords[1][0] = theNodes[nr2.idx].getCoord(0);
      tnm->coords[1][1] = theNodes[nr2.idx].getCoord(1);
      tnm->coords[1][2] = theNodes[nr2.idx].getCoord(2);
      tnm->coords[2][0] = theNodes[nr3.idx].getCoord(0);
      tnm->coords[2][1] = theNodes[nr3.idx].getCoord(1);
      tnm->coords[2][2] = theNodes[nr3.idx].getCoord(2);
      aResult = mesh[i].findRemoteNeighbor(tnm);
      if (aResult->idx >= 0) {
	theResult.idx = aResult->idx;
	theResult.cid = aResult->cid;
	return theResult;
      }
    }
  if (!faceOnSurface(nr1.idx, nr2.idx, nr3.idx))
    CkPrintf("ERROR: Search for neighbor of non-surface face FAILED.\n");
  return theResult;
}

refMsg *chunk::findRemoteNeighbor(threeNodeMsg *tnm)
{
  refMsg *theResult = new refMsg;
  node n1(tnm->coords[0][0], tnm->coords[0][1], tnm->coords[0][2]),
    n2(tnm->coords[1][0], tnm->coords[1][1], tnm->coords[1][2]),
    n3(tnm->coords[2][0], tnm->coords[2][1], tnm->coords[2][2]);
  
  CkFreeMsg(tnm);
  for (int i=0; i<numElements; i++)
    if (theElements[i].hasNode(n1) && theElements[i].hasNode(n2) &&
	theElements[i].hasNode(n3)) {
      theResult->idx = i; 
      theResult->cid = cid;
      return theResult;
    }
  theResult->idx = theResult->cid = -1; 
  return theResult;
}

nodeRef chunk::findNode(node n)
{
  nodeRef foo;
  foo.cid = cid;
  for (int i=0; i<numNodes; i++)
    if (theNodes[i] == n) {
      foo.idx = i;
      return foo;
    }
  foo.idx = -1;
  return foo;
}

intMsg *chunk::lockLF(int idx, node n1, node n2, node n3, node n4, 
		      elemRef requester, double prio)
{
  forcedGetAccessLock();
  intMsg *result = new intMsg;
  result->anInt = theElements[idx].lockLF(n1, n2, n3, n4, requester, prio);
  releaseAccessLock();
  return result;
}

splitResponse *chunk::splitLF(int idx, node in1, node in2, node in3, node in4,
			      elemRef requester)
{
  CmiAssert((idx < numElements) && (idx >= 0));
  getAccessLock();
  splitResponse *sr = theElements[idx].splitLF(in1, in2, in3, in4, requester);
  releaseAccessLock();
  return sr;
}

LEsplitResult *chunk::LEsplit(LEsplitMsg *lsm)
{
  forcedGetAccessLock();
  LEsplitResult *lsr = theElements[lsm->idx].LEsplit(lsm->root, lsm->parent, 
    lsm->newNodeRef, lsm->newNode, lsm->newRootElem, lsm->newElem, 
    lsm->targetElem, lsm->targetVol, lsm->a, lsm->b);
  CkFreeMsg(lsm);
  releaseAccessLock();
  return lsr;
}

lockResult *chunk::lockArc(lockArcMsg *lm)
{
  forcedGetAccessLock();
  lockResult *lr = 
    theElements[lm->idx].lockArc(lm->prioRef, lm->parentRef, lm->prio, 
				 lm->destRef, lm->a, lm->b);
  CkFreeMsg(lm);
  releaseAccessLock();
  return lr;
}

void chunk::unlockArc1(int idx, int prio, elemRef parentRef, elemRef destRef, 
		       node aNode, node bNode)
{
  forcedGetAccessLock();
  theElements[idx].unlockArc1(prio, parentRef, destRef, aNode, bNode);
  releaseAccessLock();
}

void chunk::unlockArc2(int idx, int prio, elemRef parentRef, elemRef destRef, 
		       node aNode, node bNode)
{
  forcedGetAccessLock();
  theElements[idx].unlockArc2(prio, parentRef, destRef, aNode, bNode);
  releaseAccessLock();
}

void chunk::updateFace(int idx, int rcid, int ridx)
{
  CmiAssert((idx < numElements) && (idx >= 0));
  theElements[idx].updateFace(rcid, ridx);
}

void chunk::updateFace(int idx, elemRef oldElem, elemRef newElem)
{
  CmiAssert((idx < numElements) && (idx >= 0));
  theElements[idx].updateFace(oldElem, newElem);
}

// surface maintenance
int chunk::nodeOnSurface(int n)
{
  return (!theSurface[n].empty());
}

int chunk::edgeOnSurface(int n1, int n2)
{
  for (unsigned int i=0; i<theSurface[n1].size(); i++) {
    if (theSurface[n1][i] == n2)
      return 1;
  }
  return 0;
}

int chunk::faceOnSurface(int n1, int n2, int n3)
{
  for (unsigned int i=0; i<theSurface[n1].size(); i+=2) {
    if (((theSurface[n1][i] == n2) && (theSurface[n1][i+1] == n3))
	|| ((theSurface[n1][i] == n3) && (theSurface[n1][i+1] == n2)))
      return 1;
  }
  return 0;
}

void chunk::updateFace(int n1, int n2, int n3, int oldNode, int newNode)
{
  if (n1 == oldNode) {
    // remove n2,n3 pair from n1's list
    simpleRemoveFace(n1, n2, n3);
    // add n2,n3 pair to newNode's list
    simpleAddFace(newNode, n2, n3);
    // modify n2's n1,n3 pair to be newNode,n3
    simpleUpdateFace(n2, n1, n3, newNode);
    // modify n3's n1,n2 pair to be newNode,n2
    simpleUpdateFace(n3, n1, n2, newNode);
  }
  else if (n2 == oldNode) {
    simpleRemoveFace(n2, n1, n3);
    simpleAddFace(newNode, n1, n3);
    simpleUpdateFace(n1, n2, n3, newNode);
    simpleUpdateFace(n3, n2, n1, newNode);
  }
  else if (n3 == oldNode) {
    simpleRemoveFace(n3, n1, n2);
    simpleAddFace(newNode, n1, n2);
    simpleUpdateFace(n1, n3, n2, newNode);
    simpleUpdateFace(n2, n3, n1, newNode);
  }
  else CkPrintf("ERROR: updateFace: oldNode not on input face!\n");
}

void chunk::simpleUpdateFace(int n1, int n2, int n3, int newNode)
{
  for (unsigned int i=0; i<theSurface[n1].size(); i+=2) {
    if ((theSurface[n1][i] == n2) && (theSurface[n1][i+1] == n3))
      theSurface[n1][i] = newNode;
    else if ((theSurface[n1][i] == n3) && (theSurface[n1][i+1] == n2))
      theSurface[n1][i+1] = newNode;
  }
}

void chunk::simpleRemoveFace(int n1, int n2, int n3)
{
  for (unsigned int i=0; i<theSurface[n1].size(); i+=2) {
    if (((theSurface[n1][i] == n2) && (theSurface[n1][i+1] == n3))
	|| ((theSurface[n1][i] == n3) && (theSurface[n1][i+1] == n2))) {
      theSurface[n1][i] = theSurface[n1][theSurface[n1].size()-2];
      theSurface[n1][i+1] = theSurface[n1][theSurface[n1].size()-1];
      theSurface[n1].pop_back();
      theSurface[n1].pop_back();
    }
  }
}

void chunk::addFace(int n1, int n2, int n3)
{
  simpleAddFace(n1, n2, n3);
  simpleAddFace(n2, n1, n3);
  simpleAddFace(n3, n1, n2);
}

void chunk::removeFace(int n1, int n2, int n3)
{
  simpleRemoveFace(n1, n2, n3);
  simpleRemoveFace(n2, n1, n3);
  simpleRemoveFace(n3, n1, n2);
}

void chunk::simpleAddFace(int n1, int n2, int n3)
{
  theSurface[n1].push_back(n2);
  theSurface[n1].push_back(n3);
}

void chunk::printSurface()
{
  for (unsigned int i=0; i<theSurface.size(); i++)
    if (theSurface[i].size() > 0) {
      CkPrintf("Node %d: ", i);
      for (unsigned int j=0; j<theSurface[i].size(); j++)
	CkPrintf(" %d", theSurface[i][j]);
      CkPrintf("\n");
    }
}

// meshLock methods: the following methods are for run-time additions
// and modifications to the chunk components
void chunk::getAccessLock()
{
  while (adjustLock)
    CthYield();
  accessLock++;
}

void chunk::forcedGetAccessLock()
{
  if ((accessLock > 0) || (adjustLock == 0))
    accessLock++;
  else getAccessLock();
}

void chunk::releaseAccessLock()
{
  accessLock--;
}

void chunk::getAdjustLock()
{
  adjustLock = 1;
  while (accessLock)
    CthYield();
}

void chunk::releaseAdjustLock()
{
  adjustLock = 0;
}

// these methods allow for run-time additions/modifications to the chunk
void chunk::allocMesh(int nEl)
{
  sizeElements = nEl + 10000;
  sizeNodes = sizeElements * 3;
  theElements.resize(sizeElements);
  theNodes.resize(sizeNodes);
  theSurface.resize(sizeNodes);
  for (int i=0; i<sizeElements; i++)
    theElements[i].set(cid, i, this); 
}

void chunk::adjustMesh()
{
  if (3*numElements >= sizeElements) {
    getAdjustLock();
    CkPrintf("[%d] Adjusting # elements...\n", cid);
    sizeElements += 3*numElements;
    theElements.resize(sizeElements);
    CkPrintf("[%d] Done adjusting # elements...\n", cid);
    for (int i=numElements; i<sizeElements; i++)
      theElements[i].set(cid, i, this); 
    releaseAdjustLock();
  }
  if (3*numElements >= sizeNodes) {
    getAdjustLock();
    CkPrintf("[%d] Adjusting # nodes...\n", cid);
    sizeNodes += 3*numElements;
    theNodes.resize(sizeNodes);
    theSurface.resize(sizeNodes);
    CkPrintf("[%d] Done adjusting # nodes...\n", cid);
    releaseAdjustLock();
  }
}

nodeRef chunk::addNode(node& n)
{
  nodeRef nRef(cid, numNodes);

  theNodes[numNodes] = n;
  if (n.isFixed() < 0) 
    CkPrintf("ERROR: chunk::addNode: must initialize fixed status of node!\n");
  if (n.onSurface() < 0) 
    CkPrintf("ERROR: chunk::addNode: must initialize surface status of node!\n");
  numNodes++;
  return nRef;
}

elemRef chunk::addElement(nodeRef& nr1, nodeRef& nr2, nodeRef& nr3, nodeRef& nr4)
{
  elemRef eRef(cid, numElements);
  nodeRef n[4] = {nr1, nr2, nr3, nr4};

  CmiAssert(numElements < sizeElements);
  CmiAssert((nr1.idx >= 0) && (nr1.idx < numNodes));
  CmiAssert((nr2.idx >= 0) && (nr2.idx < numNodes));
  CmiAssert((nr3.idx >= 0) && (nr3.idx < numNodes));
  CmiAssert((nr4.idx >= 0) && (nr4.idx < numNodes));
  theElements[numElements].set(cid, numElements, this);
  theElements[numElements].set(n);
  numElements++;
  return eRef;
}

void chunk::removeNode(intMsg *im)
{
  CmiAssert((im->anInt < numNodes) && (im->anInt >= 0));
  theNodes[im->anInt].reset();
  CkFreeMsg(im);
}

void chunk::removeElement(intMsg *im)
{
  CmiAssert((im->anInt < numElements) && (im->anInt >= 0));
  theElements[im->anInt].clear();
  CkFreeMsg(im);
}

// FEM interface methods
void chunk::newMesh(int nEl, int nGhost,const int *conn_,const int *gid_, 
		    int *surface, int nSurFaces, int idxOffset)
{ // make elements for this chunk; derive nodes and edge cids
  int i, j;

  numElements=nEl;
  allocMesh(nEl);
  
  // add elements to chunk
  for (i=0; i<numElements; i++) {
    nodeRef nodes[4];
    for (j=0; j<4; j++) {
      nodes[j].idx = conn_[i*4+j]-idxOffset;
      nodes[j].cid = cid;
    }
    theElements[i].set(cid, i, this);
    theElements[i].set(nodes);
  }

  // derive nodes from elements on this chunk
  int aNode;
  numNodes = 0;
  for (i=0; i<numElements; i++) {
    for (j=0; j<4; j++) {
      aNode = theElements[i].nodes[j].idx;
      theNodes[aNode].set(cid, aNode, this);
      theNodes[aNode].notFixed();
      theNodes[aNode].notSurface();
      if ((aNode + 1) > numNodes)
        numNodes = aNode + 1;
    }
  }

  // build surface table
  for (i=0; i<nSurFaces; i++) {
    addFace(surface[3*i], surface[3*i+1], surface[3*i+2]);
    theNodes[surface[3*i]].setSurface();
    theNodes[surface[3*i+1]].setSurface();
    theNodes[surface[3*i+2]].setSurface();
  }
  printSurface();

  for (i=0; i<numNodes; i++) 
    if (nodeOnSurface(i))
      theNodes[i].setSurface();
}

void chunk::deriveFaces()
{
  int f, i,j,k,l;
  nodeRef nr1, nr2, nr3;
  for (i=0; i<numElements; i++) {
    f = 0;
    CkPrintf("Deriving faces for element %d on chunk %d\n", i, cid);
    for (j=0; j<4; j++)
      for (k=j+1; k<4; k++)
	for (l=k+1; l<4; l++) {
	  CkPrintf("Looking for face %d from nodes (%d,%d,%d)\n",f,j,k,l);
	  nr1 = theElements[i].getNode(j);
	  nr2 = theElements[i].getNode(k);
	  nr3 = theElements[i].getNode(l);
	  theElements[i].faceElements[f] = findNeighbor(nr1, nr2, nr3, i);
	  f++;
	}
  }
  CkPrintf("Done deriving faces!\n");
}

void chunk::updateNodeCoords(int nNode, double *coord, int nEl, int nFx,
			     int *fixed)
{
  int i;
  // do some error checking
  if (nEl != numElements) 
    CkPrintf("ERROR: updateNodeCoords: nEl(%d) != numElements(%d) on chunk %d\n", nEl, numElements, cid);
  if (nNode != numNodes)
    CkPrintf("ERROR: updateNodeCoords: nNode(%d) != numNodes(%d) on chunk %d\n", nNode, numNodes, cid);
  
  // update node coordinates from coord
  for (i=0; i<numNodes; i++)
    theNodes[i].set(coord[3*i], coord[3*i + 1], coord[3*i + 2]);
  
  for (i=0; i<nFx; i++) 
    theNodes[fixed[i]].fix();
}

void chunk::refine(double *desiredVolume, refineClient *client)
{
  int i;
  theClient = client; // initialize refine client associated with this chunk

  // set desired volumes for elements
  for (i=0; i<numElements; i++)
    if (desiredVolume[i] < theElements[i].getVolume())
      theElements[i].setTargetVolume(desiredVolume[i]);
  
  // start the refinement loop
  modified = 1;
  refineInProgress = 1;
  mesh[cid].refiningElements();
}

void chunk::coarsen(double *desiredVolume, refineClient *client)
{
}

void chunk::improve(refineClient *client)
{
}

// entries to get data in mesh in stand-alone mode
void chunk::newMesh(meshMsg *mm)
{
  newMesh(mm->numElements, mm->numGhosts, mm->conn, mm->gid, mm->surface, 
	  mm->numSurFaces, mm->idxOffset);
  CkFreeMsg(mm);
}

void chunk::updateNodeCoords(coordMsg *cm)
{
  updateNodeCoords(cm->numNodes, cm->coords, cm->numElements, cm->numFixed,
		   cm->fixedNodes);
  CkFreeMsg(cm);
}

void chunk::refine()
{
  // set a target volume for some elements

  /*
  if (cid == 0) 
    mesh[cid].flippingElements();
  */

  // this happens on all chunks!
  if (cid == 0) {
  //for (int i=0; i<numElements; i++) {
    theElements[0].setTargetVolume(theElements[0].getVolume()/3000.0);
  }
  //  }
}

void chunk::start()
{
  modified = 1;  // flag a change in one of the chunk's elements
  if (!refineInProgress) { // if refine loop not running
    refineInProgress = 1;
    mesh[cid].refiningElements(); // start it up
  }
}

void chunk::improve()
{
  mesh[cid].improveMesh();
}

void chunk::finalizeImprovements()
{
  mesh[cid].relocatePoints();
}

void chunk::checkRefine()
{
  double vol, tvol;
  for (int i=0; i<numElements; i++) {
    vol = theElements[i].getVolume();
    if (vol >= 0.000416) {
      CkPrintf("WARNING: On chunk %d element %d is not adequately refined: volume=%f\n",
	       cid, i, vol);
    }
  }
}

#include "PMAF.def.h"
