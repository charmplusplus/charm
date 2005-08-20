#include "tri.h"
#include "edge.h"

void edge::reset() 
{ 
  newNodeIdx = incidentNode = fixNode = -1;
  if (!(newEdgeRef == nullRef))
    C->theEdges[newEdgeRef.idx].reset();
  unsetPending(); waitingFor.reset(); newEdgeRef.reset(); 
  newNode.reset(); 
}

int edge::isPending(elemRef e)
{
  return (pending && (waitingFor == e));
}

void edge::checkPending(elemRef e) 
{
  elemRef nullRef;
  if (pending && (waitingFor == e) && !(e == nullRef))
    mesh[e.cid].refineElement(e.idx, e.getArea());
}

void edge::checkPending(elemRef e, elemRef ne) 
{
  elemRef nullRef;
  if (pending && (waitingFor == e) && !(e == nullRef)) {
    waitingFor = ne;
    mesh[ne.cid].refineElement(ne.idx, ne.getArea());
  }
}

int edge::split(int *m, edgeRef *e_prime, int oIdx, int fIdx,
		elemRef requester, int *local, int *first, int *nullNbr)
{
  // element requester has asked this edge to split and give back a new node
  // and new edgeRef on oIdx; return value is 1 if successful, 0 if
  // e_prime is NOT incident on oIdx, -1 if another split is pending
  // on this edge; local is set if this edge is not a boundary between chunks
  // and first is set if this was the first split request on this edge
  intMsg *im;
  elemRef nbr = getNot(requester), nullRef;
  nullRef.reset();

  if (requester.cid != myRef.cid) {
    FEM_Node *theNodes = &(C->meshPtr->node);
    const FEM_Comm_List *sharedList = &(theNodes->shared.getList(requester.cid));
    oIdx = (*sharedList)[oIdx];
    fIdx = (*sharedList)[fIdx];
  }
#ifdef TDEBUG3
  CkPrintf("TMRC2D: [%d] oIdx=%d fIdx=%d ", myRef.cid, oIdx, fIdx);
  CkPrintf("\ntheNodes[oIdx]="); C->theNodes[oIdx].dump();
  CkPrintf("\ntheNodes[fIdx]="); C->theNodes[fIdx].dump();
  CkPrintf("\n");
  CkPrintf("TMRC2D: [%d] node[0]=%d node[1]=%d ",myRef.cid,nodes[0],nodes[1]);
  CkPrintf("\ncoords of node[0]="); C->theNodes[nodes[0]].dump();
  CkPrintf("\ncoords of node[1]="); C->theNodes[nodes[1]].dump();
  CkPrintf("\n");
  CkAssert((oIdx == nodes[0]) || (oIdx == nodes[1]));
  CkAssert((fIdx == nodes[0]) || (fIdx == nodes[1]));
#endif
  if (pending && (waitingFor == requester)) { 
    // already split; waiting for requester
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] edge::split: ** PART 2! ** On edge=%d on chunk=%d, requester=(%d,%d) with nbr=(%d,%d)\n", myRef.cid, myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);
#endif
    *m = newNodeIdx;
    *e_prime = newEdgeRef;
    *first = 0;
    *local = 1;
    if (nbr.cid != requester.cid) { 
      *local = 0;
      im = mesh[requester.cid].addNode(newNode, C->theNodes[nodes[0]].boundary,
				       C->theNodes[nodes[1]].boundary, 1);
      *m = im->anInt;
      CkFreeMsg(im);
#ifdef TDEBUG2
      CkPrintf("TMRC2D: [%d] New node (%f,%f) added at index %d on chunk %d\n", myRef.cid, newNode.X(), newNode.Y(), *m, myRef.cid);
#endif
    }
    int nLoc = newNodeIdx;
    if (requester.cid == myRef.cid) { 
      nLoc = *m;
      C->theEdges[newEdgeRef.idx].updateNode(newNodeIdx, nLoc);
    }
    if (oIdx == incidentNode) { // incidence as planned
      if (nodes[0] == oIdx) {
	nodes[0] = nLoc;
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Edge %d node[0] updated to %d\n", myRef.cid, 
		 myRef.idx, nLoc);
#endif
      }
      else if (nodes[1] == oIdx) {
	nodes[1] = nLoc;
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Edge %d node[1] updated to %d\n", myRef.cid, 
		 myRef.idx, nLoc);
#endif
      }
      else CkAbort("ERROR: incident node not found on edge\n");
      return 1; 
    }
    else { // incidence is on fIdx
      CkAssert(fIdx == incidentNode);
      if (nodes[0] == fIdx) {
	nodes[0] = nLoc;
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Edge %d node[0] updated to %d\n", myRef.cid, 
		 myRef.idx, nLoc);
#endif
      }
      else if (nodes[1] == fIdx) {
	nodes[1] = nLoc;
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Edge %d node[1] updated to %d\n", myRef.cid, 
		 myRef.idx, nLoc);
#endif
      }
      else CkAbort("ERROR: incident node not found on edge\n");
      return 0;
    }
  }
  else if (pending) { // can't split a second time yet; waiting for nbr elem
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] edge::split: ** Pending on (%d,%d)! ** On edge=%d on chunk=%d, requester=%d on chunk=%d\n", myRef.cid, waitingFor.cid, waitingFor.idx, myRef.idx, myRef.cid, requester.idx, requester.cid);
#endif
    return -1;
  }
  else { // Need to do the split
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] edge::split: ** PART 1! ** On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d)\n", myRef.cid, myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);
#endif
    setPending();
    C->theNodes[oIdx].midpoint(C->theNodes[fIdx], newNode);
    im = mesh[requester.cid].addNode(newNode, C->theNodes[nodes[0]].boundary, 
				     C->theNodes[nodes[1]].boundary, (nbr.cid != -1));
    newNodeIdx = im->anInt;
    CkFreeMsg(im);
#ifdef TDEBUG2
    CkPrintf("TMRC2D: [%d] New node (%f,%f) added at index %d on chunk %d\n", myRef.cid, newNode.X(), newNode.Y(), newNodeIdx, requester.cid);
#endif
    newEdgeRef = C->addEdge(newNodeIdx, oIdx, boundary);
#ifdef TDEBUG2
    CkPrintf("TMRC2D: [%d] New edge (%d,%d) added between nodes (%f,%f) and newNode\n", myRef.cid, newEdgeRef.cid, newEdgeRef.idx, C->theNodes[oIdx].X(), C->theNodes[oIdx].Y());
#endif
    incidentNode = oIdx;
    fixNode = fIdx;
    *m = newNodeIdx;
    *e_prime = newEdgeRef;
    *first = 1;
    if ((nbr.cid == requester.cid) || (nbr.cid == -1)) *local = 1;
    else *local = 0;
    C->theEdges[newEdgeRef.idx].setPending();
    *nullNbr = 0;
    if (nbr == nullRef) *nullNbr = 1;
    if (nbr.cid != -1) {
      waitingFor = nbr;
      double nbrArea = nbr.getArea();
      mesh[nbr.cid].refineElement(nbr.idx, nbrArea);
    }
    else {
      if (nodes[0] == oIdx) {
	nodes[0] = newNodeIdx;
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Edge %d node[0] updated to %d\n", myRef.cid, 
		 myRef.idx, newNodeIdx);
#endif
      }
      else if (nodes[1] == oIdx) {
	nodes[1] = newNodeIdx;
#ifdef TDEBUG2
	CkPrintf("TMRC2D: [%d] Edge %d node[1] updated to %d\n", myRef.cid, 
		 myRef.idx, newNodeIdx);
#endif
      }
      else CkAbort("ERROR: incident node not found on edge\n");
    }
    return 1;
  }
}

void edge::collapse(elemRef requester, int kIdx, int dIdx, elemRef kNbr,
		    elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node newN, 
		    double frac)
{
  int local, first, dIdxlShared, kIdxlShared;
  elemRef nbr = getNot(requester);
  FEM_Comm_Rec *dNodeRec, *kNodeRec;

  translateSharedNodeIDs(&kIdx, &dIdx, requester);

  local = 0;
  if ((nbr.cid == -1) || (nbr.cid == requester.cid)) local = 1;
  if (pending && (waitingFor == requester)) { // collapsed; awaiting requester
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] edge::collapse: PART 2: On edge=%d on chunk=%d, requester=(%d,%d) with nbr=(%d,%d) dIdx=%d kIdx=%d dNbr=%d kNbr=%d dEdge=%d kEdge=%d\n", myRef.cid, myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx, dIdx, kIdx, dNbr.idx, kNbr.idx, dEdge.idx, kEdge.idx);
#endif
    first = 0;
    if (dIdx == incidentNode) { // incidence as planned
      localCollapse(kIdx, dIdx, &requester, &newNode, frac, &kNbr, &dNbr, 
		    &kEdge, &dEdge, local, first);
      updateCloud(kIdx, dIdx, newNode, &dIdxlShared, &kIdxlShared, &dNodeRec,
		  &kNodeRec);
      unlockCloudRemoveEdge(dIdxlShared, kIdxlShared, dNodeRec, kNodeRec);
    }
    else { // incidence is on kNode
      localCollapse(dIdx, kIdx, &requester, &newNode, frac, &dNbr, &kNbr, 
		    &dEdge, &kEdge, local, first);
      updateCloud(dIdx, kIdx, newNode, &dIdxlShared, &kIdxlShared, &dNodeRec, 
		  &kNodeRec);
      unlockCloudRemoveEdge(dIdxlShared, kIdxlShared, dNodeRec, kNodeRec);
    }
  }
  else if (pending) { // can't collapse a second time yet; waiting for nbr elem
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] edge::collapse: Pending on (%d,%d): On edge=%d on chunk=%d, requester=%d on chunk=%d\n", myRef.cid, waitingFor.cid, waitingFor.idx, myRef.idx, myRef.cid, requester.idx, requester.cid);
#endif
  }
  else { // Need to do the collapse
    first = 1;
    if (!buildLockingCloud(kIdx, dIdx, &requester, &nbr)) return;
#ifdef TDEBUG1
    CkPrintf("TMRC2D: [%d] edge::collapse: PART 1: On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d) dIdx=%d kIdx=%d dNbr=%d kNbr=%d dEdge=%d kEdge=%d\n", myRef.cid, myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx, dIdx, kIdx, dNbr.idx, kNbr.idx, dEdge.idx, kEdge.idx);
#endif
    setPending();
    incidentNode = dIdx;  fixNode = kIdx;
    newNode = newN;
    localCollapse(kIdx, dIdx, &requester, &newNode, frac, &kNbr, &dNbr, 
		  &kEdge, &dEdge, local, first);
    if (nbr.cid > -1) {
      waitingFor = nbr;
      double nbrArea = nbr.getArea();
      CkPrintf("Calling coarsen on neighbor element[%d] with area=%1.10e\n", 
	       nbr.idx, 2.0*nbrArea);
      mesh[nbr.cid].coarsenElement(nbr.idx, 2.0*nbrArea);
    }
    else {
      updateCloud(kIdx, dIdx, newNode, &dIdxlShared, &kIdxlShared, &dNodeRec,
		  &kNodeRec);
      unlockCloudRemoveEdge(dIdxlShared, kIdxlShared, dNodeRec, kNodeRec);
    }
  }
}

int edge::flipPrevent(elemRef requester, int kIdx, int dIdx, elemRef kNbr,
		   elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node newN)
{
  int i,j, lk, chunk, dIdxlShared, kIdxlShared;
  int *dIdxlChk, *dIdxlIdx, *kIdxlChk, *kIdxlIdx;
  boolMsg *ret;

  length = (C->theNodes[kIdx]).distance(C->theNodes[dIdx]);
  FEM_Node *theNodes = &(C->meshPtr->node);
  FEM_Comm_Rec *dNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(dIdx));
  FEM_Comm_Rec *kNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(kIdx));
  intMsg *im;
  lk = C->lockLocalChunk(myRef.cid, myRef.idx, length);
  if (dNodeRec) dIdxlShared = dNodeRec->getShared();
  else dIdxlShared = 0;
  if (kNodeRec) kIdxlShared = kNodeRec->getShared();
  else kIdxlShared = 0;
  if (!(C->lockLocalChunk(myRef.cid, myRef.idx, length)))
    return -1;
  for (i=0; i<dIdxlShared; i++) {
    chunk = dNodeRec->getChk(i);
    im = mesh[chunk].lockChunk(myRef.cid, myRef.idx, length);
    if (im->anInt == 0) { 
      CkFreeMsg(im); 
      C->unlockLocalChunk(myRef.cid, myRef.idx);
      for (j=0; j<i; j++) {
	chunk = dNodeRec->getChk(j);
	mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
      }
      return -1; 
    }
  }
  for (i=0; i<kIdxlShared; i++) {
    chunk = kNodeRec->getChk(i);
    im = mesh[chunk].lockChunk(myRef.cid, myRef.idx, length);
    if (im->anInt == 0) { 
      CkFreeMsg(im); 
      C->unlockLocalChunk(myRef.cid, myRef.idx);
      for (j=0; j<dIdxlShared; j++) {
	chunk = dNodeRec->getChk(j);
	mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
      }
      for (j=0; j<i; j++) {
	chunk = kNodeRec->getChk(j);
	mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
      }
      return -1; 
    }
  }
  fixNode = kIdx;
  newNode = newN;

  theNodes = &(C->meshPtr->node);
  dNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(dIdx));
  kNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(kIdx));
  // Replace dNode with kNode and delete dNode everywhere in the mesh
  if (dNodeRec) dIdxlShared = dNodeRec->getShared();
  else dIdxlShared = 0;
  if (kNodeRec) kIdxlShared = kNodeRec->getShared();
  else kIdxlShared = 0;
  dIdxlChk = (int *)malloc((dIdxlShared+1)*sizeof(int));
  kIdxlChk = (int *)malloc((kIdxlShared+1)*sizeof(int));
  dIdxlIdx = (int *)malloc((dIdxlShared+1)*sizeof(int));
  kIdxlIdx = (int *)malloc((kIdxlShared+1)*sizeof(int));
  for (i=0; i<dIdxlShared; i++) {
    dIdxlIdx[i] = dNodeRec->getIdx(i);
    dIdxlChk[i] = dNodeRec->getChk(i);
  }
  dIdxlIdx[dIdxlShared] = dIdx;
  dIdxlChk[dIdxlShared] = myRef.cid;
  for (i=0; i<kIdxlShared; i++) {
    kIdxlIdx[i] = kNodeRec->getIdx(i);
    kIdxlChk[i] = kNodeRec->getChk(i);
  }
  kIdxlIdx[kIdxlShared] = kIdx;
  kIdxlChk[kIdxlShared] = myRef.cid;
  ret = C->flipPrevent(kIdx, dIdx, newNode, dIdxlShared, dIdxlChk, dIdxlIdx);
  for (i=0; i<dIdxlShared; i++) {
    chunk = dNodeRec->getChk(i);
    if (kNodeRec && ((j=existsOn(kNodeRec, chunk)) >= 0)) {
      ret = mesh[chunk].flipPrevent(kNodeRec->getIdx(j), dNodeRec->getIdx(i), newNode, dIdxlShared, dIdxlChk, dIdxlIdx);
    }
    else {
      ret = mesh[chunk].flipPrevent(-1, dNodeRec->getIdx(i), newNode, kIdxlShared, kIdxlChk, kIdxlIdx);
    }
  }
  for (i=0; i<kIdxlShared; i++) {
    chunk = kNodeRec->getChk(i);
    if (!dNodeRec || (existsOn(dNodeRec, chunk) == -1)) {
      ret = mesh[chunk].flipPrevent(kNodeRec->getIdx(i), -1, newNode, dIdxlShared, dIdxlChk, dIdxlIdx);
    }
  }
  // unlock the cloud
  C->unlockLocalChunk(myRef.cid, myRef.idx);
  for (i=0; i<dIdxlShared; i++) {
    chunk = dNodeRec->getChk(i);
    mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
  }
  for (i=0; i<kIdxlShared; i++) {
    chunk = kNodeRec->getChk(i);
    mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
  }
  if(ret->aBool) return -1;
  return 1;  
}

void edge::translateSharedNodeIDs(int *kIdx, int *dIdx, elemRef req)
{
  if (req.cid != myRef.cid) { 
    FEM_Node *theNodes = &(C->meshPtr->node);
    const FEM_Comm_List *sharedList = &(theNodes->shared.getList(req.cid));
    (*kIdx) = (*sharedList)[(*kIdx)];
    (*dIdx) = (*sharedList)[(*dIdx)];
  }
#ifdef TDEBUG3
  CkPrintf("TMRC2D: [%d] kIdx=%d dIdx=%d\n", myRef.cid, *kIdx, *dIdx);
#endif
}

void edge::unlockCloudRemoveEdge(int dIdxlShared, int kIdxlShared, 
			 FEM_Comm_Rec *dNodeRec, FEM_Comm_Rec *kNodeRec)
{
  int chunk;
  C->unlockLocalChunk(myRef.cid, myRef.idx);
  for (int i=0; i<dIdxlShared; i++) {
    chunk = dNodeRec->getChk(i);
    mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
  }
  for (int i=0; i<kIdxlShared; i++) {
    chunk = kNodeRec->getChk(i);
    mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
  }
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] ......removing edge %d on %d\n", myRef.cid, 
	   myRef.idx, myRef.cid);
#endif
  C->removeEdge(myRef.idx);
}

void edge::localCollapse(int kIdx, int dIdx, elemRef *req, node *newNode, 
			 double frac, elemRef *keepNbr, elemRef *delNbr, 
			 edgeRef *kEdge, edgeRef *dEdge, int local, int first)
{
  int b = 0, flag;
  // tell delNbr to replace dEdge with kEdge
  if (delNbr->cid != -1) {
    CkPrintf("Telling delNbr %d to replace dEdge %d with kEdge %d\n", 
	     delNbr->idx, dEdge->idx, kEdge->idx);
    mesh[delNbr->cid].updateElementEdge(delNbr->idx, *dEdge, *kEdge);
    b = dEdge->getBoundary();
  }
  // tell kEdge to replace myRef with delNbr
  CkPrintf("Telling keepEdge %d to replace requester %d with delNbr %d\n", 
	   kEdge->idx, req->idx, delNbr->idx);
  kEdge->update(*req, *delNbr, b);
  // remove delEdge
  dEdge->remove();
  if ((delNbr->cid == -1) && (keepNbr->cid == -1))
    kEdge->remove();
  // Notify FEM client of the collapse
  if (local && first) flag = LOCAL_FIRST;
  if (local && !first) flag = LOCAL_SECOND;
  if (!local && first) flag = BOUND_FIRST;
  if (!local && !first) flag = BOUND_SECOND;
  C->theClient->collapse(req->idx, kIdx, dIdx, newNode->X(), newNode->Y(),
			 flag, b, frac);
#ifdef TDEBUG1
  CkPrintf("TMRC2D: [%d] theClient->collapse(%d, %d, %d, %2.10f, %2.10f, (flag), %d, %1.1f\n", req->cid, req->idx, kIdx, dIdx, newNode->X(), newNode->Y(), b, frac);
#endif
}

int edge::buildLockingCloud(int kIdx, int dIdx, elemRef *req, elemRef *nbr)
{
  // Lock the cloud of chunks around dNode and kNode
  double length;
  int dIdxlShared, kIdxlShared, chunk;
  length = (C->theNodes[kIdx]).distance(C->theNodes[dIdx]);
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] ......Building lock cloud... edge=%d requester=%d nbr=%d\n", myRef.cid, myRef.idx, req->idx, nbr->idx);
#endif 
  FEM_Node *theNodes = &(C->meshPtr->node);
  FEM_Comm_Rec *dNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(dIdx));
  FEM_Comm_Rec *kNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(kIdx));
  intMsg *im;
  if (dNodeRec) dIdxlShared = dNodeRec->getShared();
  else dIdxlShared = 0;
  if (kNodeRec) kIdxlShared = kNodeRec->getShared();
  else kIdxlShared = 0;
  if (!(C->lockLocalChunk(myRef.cid, myRef.idx, length)))
    return 0;
  for (int i=0; i<dIdxlShared; i++) {
    chunk = dNodeRec->getChk(i);
    im = mesh[chunk].lockChunk(myRef.cid, myRef.idx, length);
    if (im->anInt == 0) { 
      CkFreeMsg(im); 
      C->unlockLocalChunk(myRef.cid, myRef.idx);
      for (int j=0; j<i; j++) {
	chunk = dNodeRec->getChk(j);
	mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
      }
      return 0; 
    }
  }
  for (int i=0; i<kIdxlShared; i++) {
    chunk = kNodeRec->getChk(i);
    im = mesh[chunk].lockChunk(myRef.cid, myRef.idx, length);
    if (im->anInt == 0) { 
      CkFreeMsg(im); 
      C->unlockLocalChunk(myRef.cid, myRef.idx);
      for (int j=0; j<dIdxlShared; j++) {
	chunk = dNodeRec->getChk(j);
	mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
      }
      for (int j=0; j<i; j++) {
	chunk = kNodeRec->getChk(j);
	mesh[chunk].unlockChunk(myRef.cid, myRef.idx);
      }
      return 0; 
    }
  }
#ifdef TDEBUG2
  CkPrintf("TMRC2D: [%d] ......edge::collapse: LOCKS obtained... On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d)\n", myRef.cid, myRef.idx, myRef.cid, req->cid, req->idx, nbr->cid, nbr->idx);
#endif
  return 1;
}

void edge::updateCloud(int kIdx, int dIdx, node newNode, int *dIdxl,int *kIdxl,
		       FEM_Comm_Rec **dNodeRec, FEM_Comm_Rec **kNodeRec)
{
  int chunk, j;
  int *dIdxlChk, *dIdxlIdx, *kIdxlChk, *kIdxlIdx;

  FEM_Node *theNodes = &(C->meshPtr->node);
  (*dNodeRec) = (FEM_Comm_Rec *)(theNodes->shared.getRec(dIdx));
  (*kNodeRec) = (FEM_Comm_Rec *)(theNodes->shared.getRec(kIdx));
  // Replace dNode with kNode and delete dNode everywhere in the mesh
  if ((*dNodeRec)) (*dIdxl) = (*dNodeRec)->getShared();
  else (*dIdxl) = 0;
  if ((*kNodeRec)) (*kIdxl) = (*kNodeRec)->getShared();
  else (*kIdxl) = 0;
  dIdxlChk = (int *)malloc(((*dIdxl)+1)*sizeof(int));
  kIdxlChk = (int *)malloc(((*kIdxl)+1)*sizeof(int));
  dIdxlIdx = (int *)malloc(((*dIdxl)+1)*sizeof(int));
  kIdxlIdx = (int *)malloc(((*kIdxl)+1)*sizeof(int));
  for (int i=0; i<(*dIdxl); i++) {
    dIdxlIdx[i] = (*dNodeRec)->getIdx(i);
    dIdxlChk[i] = (*dNodeRec)->getChk(i);
  }
  dIdxlIdx[(*dIdxl)] = dIdx;
  dIdxlChk[(*dIdxl)] = myRef.cid;
  for (int i=0; i<(*kIdxl); i++) {
    kIdxlIdx[i] = (*kNodeRec)->getIdx(i);
    kIdxlChk[i] = (*kNodeRec)->getChk(i);
  }
  kIdxlIdx[(*kIdxl)] = kIdx;
  kIdxlChk[(*kIdxl)] = myRef.cid;
  C->nodeReplaceDelete(kIdx, dIdx, newNode, (*dIdxl), dIdxlChk, 
		       dIdxlIdx);
  for (int i=0; i<(*dIdxl); i++) {
    chunk = (*dNodeRec)->getChk(i);
    if ((*kNodeRec) && ((j=existsOn((*kNodeRec), chunk)) >= 0)) {
      mesh[chunk].nodeReplaceDelete((*kNodeRec)->getIdx(j), 
				    (*dNodeRec)->getIdx(i), newNode, 
				    (*dIdxl), dIdxlChk, dIdxlIdx);
    }
    else {
      mesh[chunk].nodeReplaceDelete(-1, (*dNodeRec)->getIdx(i), newNode,
				    (*kIdxl), kIdxlChk, kIdxlIdx);
    }
  }
  for (int i=0; i<(*kIdxl); i++) {
    chunk = (*kNodeRec)->getChk(i);
    if (!(*dNodeRec) || (existsOn((*dNodeRec), chunk) == -1)) {
      mesh[chunk].nodeReplaceDelete((*kNodeRec)->getIdx(i), -1, newNode, 
				    (*dIdxl), dIdxlChk, dIdxlIdx);
    }
  }
}

void edge::sanityCheck(chunk *c, edgeRef shouldRef) 
{
  int nonNullElements=0;
  for (int i=0;i<2;i++) {
    if (!elements[i].isNull()) {
      elements[i].sanityCheck();
      nonNullElements++;
    }
  }
  if (nonNullElements == 0)
    CkPrintf("TMRC2D: WARNING: Dangling edge found!\n");
}

void edge::sanityCheck(int node1, int node2, int eIdx)
{
  CkAssert((node1 == nodes[0]) || (node1 == nodes[1]));
  CkAssert((node2 == nodes[0]) || (node2 == nodes[1]));
  CkAssert((elements[0].idx == eIdx) || (elements[1].idx == eIdx));
}
