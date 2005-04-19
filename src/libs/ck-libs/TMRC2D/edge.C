#include "tri.h"
#include "edge.h"

void edge::reset() 
{ 
  newNodeIdx = incidentNode = fixNode = -1;
  if (!(newEdgeRef == nullRef))
    C->theEdges[newEdgeRef.idx].reset();
  unsetPending(); waitingFor.reset(); newEdgeRef.reset(); 
  keepNbr.reset(); delNbr.reset();
  keepEdge.reset(); delEdge.reset();
  newNode.reset(); opnode.reset();
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
  if (pending && (waitingFor == requester)) { 
    // already split; waiting for requester
    DEBUGREF(CkPrintf("TMRC2D: edge::split: ** PART 2! ** On edge=%d on chunk=%d, requester=(%d,%d) with nbr=(%d,%d)\n", myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);)
  CkAssert((oIdx == nodes[0]) || (oIdx == nodes[1]));
  CkAssert((fIdx == nodes[0]) || (fIdx == nodes[1]));
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
      DEBUGREF(CkPrintf("TMRC2D: New node (%f,%f) added at index %d on chunk %d\n", newNode.X(), newNode.Y(), *m, myRef.cid);)
    }
    int nLoc = newNodeIdx;
    if (requester.cid == myRef.cid) nLoc = *m;
    if (oIdx == incidentNode) { // incidence as planned
      if (nodes[0] == oIdx) nodes[0] = nLoc;
      else if (nodes[1] == oIdx) nodes[1] = nLoc;
      else CkAbort("ERROR: incident node not found on edge\n");
      return 1; 
    }
    else { // incidence is on fIdx
      if (nodes[0] == fIdx) nodes[0] = nLoc;
      else if (nodes[1] == fIdx) nodes[1] = nLoc;
      else CkAbort("ERROR: incident node not found on edge\n");
      return 0;
    }
    
  }
  else if (pending) { // can't split a second time yet; waiting for nbr elem
    DEBUGREF(CkPrintf("TMRC2D: edge::split: ** Pending on (%d,%d)! ** On edge=%d on chunk=%d, requester=%d on chunk=%d\n", waitingFor.cid, waitingFor.idx, myRef.idx, myRef.cid, requester.idx, requester.cid);)
    return -1;
  }
  else { // Need to do the split
    DEBUGREF(CkPrintf("TMRC2D: edge::split: ** PART 1! ** On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d)\n", myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);)
    setPending();
    C->theNodes[oIdx].midpoint(C->theNodes[fIdx], newNode);
    im = mesh[requester.cid].addNode(newNode, C->theNodes[nodes[0]].boundary, 
				     C->theNodes[nodes[1]].boundary, (nbr.cid != -1));
    newNodeIdx = im->anInt;
    CkFreeMsg(im);
    DEBUGREF(CkPrintf("TMRC2D: New node (%f,%f) added at index %d on chunk %d\n", newNode.X(), newNode.Y(), newNodeIdx, myRef.cid);)
    newEdgeRef = C->addEdge(newNodeIdx, oIdx);
    DEBUGREF(CkPrintf("TMRC2D: New edge (%d,%d) added between nodes (%f,%f) and newNode\n", newEdgeRef.cid, newEdgeRef.idx, C->theNodes[oIdx].X(), C->theNodes[oIdx].Y());)
    incidentNode = oIdx;
    fixNode = fIdx;
    *m = newNodeIdx;
    *e_prime = newEdgeRef;
    *first = 1;
  CkAssert((oIdx == nodes[0]) || (oIdx == nodes[1]));
  CkAssert((fIdx == nodes[0]) || (fIdx == nodes[1]));
    if ((nbr.cid == requester.cid) || (nbr.cid == -1)) *local = 1;
    else *local = 0;
    C->theEdges[newEdgeRef.idx].setPending();
    *nullNbr = 0;
    if (nbr == nullRef) *nullNbr = 1;
    if (*nullNbr) {
      DEBUGREF(CkPrintf("TMRC2D: on edge, nbr is null\n");)
    }
  CkAssert((oIdx == nodes[0]) || (oIdx == nodes[1]));
  CkAssert((fIdx == nodes[0]) || (fIdx == nodes[1]));
    if (nbr.cid != -1) {
      waitingFor = nbr;
      double nbrArea = nbr.getArea();
      mesh[nbr.cid].refineElement(nbr.idx, nbrArea);
    }
    else {
      if (nodes[0] == oIdx) nodes[0] = newNodeIdx;
      else if (nodes[1] == oIdx) nodes[1] = newNodeIdx;
      else CkAbort("ERROR: incident node not found on edge\n");
    }
    return 1;
  }
}

int edge::collapse(elemRef requester, int kIdx, int dIdx, elemRef kNbr,
		   elemRef dNbr, edgeRef kEdge, edgeRef dEdge, node oNode,
		   int *local, int *first, node newN)
{
  // element requester has asked this edge to collapse and give back new node
  // coordinates resulting node; return value is 1 if successful, 0 if
  // dNode is the node that is kept, -1 if fail
  int i, j, lk, chunk, dIdxlShared, kIdxlShared;
  int *dIdxlChk, *dIdxlIdx, *kIdxlChk, *kIdxlIdx;
  elemRef nbr = getNot(requester), nullRef;
  nullRef.reset();
  *local = 0;
  if ((nbr.cid == -1) || (nbr.cid == requester.cid)) *local = 1;
  if (pending && (waitingFor == requester)) { // collapsed; awaiting requester
    DEBUGREF(CkPrintf("TMRC2D: [%d] ......edge::collapse: ** PART 2! ** On edge=%d on chunk=%d, requester=(%d,%d) with nbr=(%d,%d) dIdx=%d kIdx=%d\n", myRef.cid, myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx, dIdx, kIdx);)
  CkAssert((dIdx == nodes[0]) || (dIdx == nodes[1]));
  CkAssert((kIdx == nodes[0]) || (kIdx == nodes[1]));
    *first = 0;
    if (dIdx == incidentNode) { // incidence as planned
      FEM_Node *theNodes = &(C->meshPtr->node);
      FEM_Comm_Rec *dNodeRec = (FEM_Comm_Rec *)(theNodes->shared.getRec(dIdx));
      FEM_Comm_Rec *kNodeRec = (FEM_Comm_Rec *)(theNodes->shared.getRec(kIdx));
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
      C->nodeReplaceDelete(kIdx, dIdx, newNode, dIdxlShared, dIdxlChk, 
			   dIdxlIdx);
      for (i=0; i<dIdxlShared; i++) {
	chunk = dNodeRec->getChk(i);
	if (kNodeRec && ((j=existsOn(kNodeRec, chunk)) >= 0)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(j), 
					dNodeRec->getIdx(i), newNode, 
					dIdxlShared, dIdxlChk, dIdxlIdx);
	}
	else {
	  mesh[chunk].nodeReplaceDelete(-1, dNodeRec->getIdx(i), newNode,
					kIdxlShared, kIdxlChk, kIdxlIdx);
	}
      }
      for (i=0; i<kIdxlShared; i++) {
	chunk = kNodeRec->getChk(i);
	if (!dNodeRec || (existsOn(dNodeRec, chunk) == -1)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(i), -1, newNode, 
					dIdxlShared, dIdxlChk, dIdxlIdx);
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
      //DEBUGREF(CkPrintf("TMRC2D: [%d] ......removing edge %d on %d\n", myRef.cid, myRef.idx, myRef.cid);)
      C->removeEdge(myRef.idx);
      return 1; 
    }
    else { // incidence is on kNode
      int tmp = dIdx; 
      dIdx = kIdx;
      kIdx = tmp;
      FEM_Node *theNodes = &(C->meshPtr->node);
      FEM_Comm_Rec *dNodeRec = (FEM_Comm_Rec *)(theNodes->shared.getRec(dIdx));
      FEM_Comm_Rec *kNodeRec = (FEM_Comm_Rec *)(theNodes->shared.getRec(kIdx));
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
      C->nodeReplaceDelete(kIdx, dIdx, newNode, dIdxlShared, dIdxlChk, 
			   dIdxlIdx);
      for (i=0; i<dIdxlShared; i++) {
	chunk = dNodeRec->getChk(i);
	if (kNodeRec && ((j=existsOn(kNodeRec, chunk)) >= 0)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(j), 
					dNodeRec->getIdx(i), newNode, 
					dIdxlShared, dIdxlChk, dIdxlIdx);
	}
	else {
	  mesh[chunk].nodeReplaceDelete(-1, dNodeRec->getIdx(i), newNode,
					kIdxlShared, kIdxlChk, kIdxlIdx);
	}
      }
      for (i=0; i<kIdxlShared; i++) {
	chunk = kNodeRec->getChk(i);
	if (!dNodeRec || (existsOn(dNodeRec, chunk) == -1)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(i), -1, newNode, 
					dIdxlShared, dIdxlChk, dIdxlIdx);
	}
      }
      for (i=0; i<dIdxlShared; i++) {
	chunk = dNodeRec->getChk(i);
	if (kNodeRec && ((j=existsOn(kNodeRec, chunk)) >= 0)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(j), 
					dNodeRec->getIdx(i), newNode,
					dIdxlShared, dIdxlChk, dIdxlIdx);
	}
	else {
	  mesh[chunk].nodeReplaceDelete(-1, dNodeRec->getIdx(i), newNode,
					kIdxlShared, kIdxlChk, kIdxlIdx);
	}
      }
      for (i=0; i<kIdxlShared; i++) {
	chunk = kNodeRec->getChk(i);
	if (!dNodeRec || (existsOn(dNodeRec, chunk) == -1)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(i), -1, newNode,
					dIdxlShared, dIdxlChk, dIdxlIdx);
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
      //DEBUGREF(CkPrintf("TMRC2D: [%d] ......removing edge %d on %d\n", myRef.cid, myRef.idx, myRef.cid);)
      C->removeEdge(myRef.idx);
      return 0; 
    }
  }
  else if (pending) { // can't collapse a second time yet; waiting for nbr elem
    DEBUGREF(CkPrintf("TMRC2D: [%d] ......edge::collapse: ** Pending on (%d,%d)! ** On edge=%d on chunk=%d, requester=%d on chunk=%d\n", myRef.cid, waitingFor.cid, waitingFor.idx, myRef.idx, myRef.cid, requester.idx, requester.cid);)
    return -1;
  }
  else { // Need to do the collapse
    // Lock the cloud of chunks around dNode and kNode
    length = (C->theNodes[kIdx]).distance(C->theNodes[dIdx]);
    *first = 1;
    //DEBUGREF(CkPrintf("TMRC2D: [%d] ......Building lock cloud... edge=%d requester=%d nbr=%d\n", myRef.cid, myRef.idx, requester.idx, nbr.idx);)
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
    //DEBUGREF(CkPrintf("TMRC2D: [%d] ......edge::collapse: LOCKS obtained... On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d)\n", myRef.cid, myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx);)
    DEBUGREF(CkPrintf("TMRC2D: [%d] ......edge::collapse: ** PART 1! ** On edge=%d on chunk=%d, requester==(%d,%d) with nbr=(%d,%d) dIdx=%d kIdx=%d\n", myRef.cid, myRef.idx, myRef.cid, requester.cid, requester.idx, nbr.cid, nbr.idx, dIdx, kIdx);)
    CkAssert((dIdx == nodes[0]) || (dIdx == nodes[1]));
    CkAssert((kIdx == nodes[0]) || (kIdx == nodes[1]));
    setPending();
    incidentNode = dIdx;
    fixNode = kIdx;
    opnode = oNode;
    newNode = newN;
    keepNbr = kNbr;
    delNbr = dNbr;
    keepEdge = kEdge;
    delEdge = dEdge;
    if (nbr.cid > -1) {
      waitingFor = nbr;
      double nbrArea = nbr.getArea();
      mesh[nbr.cid].coarsenElement(nbr.idx, nbrArea*1.01+0.000000000000000001);
    }
    else {
      FEM_Node *theNodes = &(C->meshPtr->node);
      FEM_Comm_Rec *dNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(dIdx));
      FEM_Comm_Rec *kNodeRec=(FEM_Comm_Rec *)(theNodes->shared.getRec(kIdx));
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
      C->nodeReplaceDelete(kIdx, dIdx, newNode, dIdxlShared, dIdxlChk, 
			   dIdxlIdx);
      for (i=0; i<dIdxlShared; i++) {
	chunk = dNodeRec->getChk(i);
	if (kNodeRec && ((j=existsOn(kNodeRec, chunk)) >= 0)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(j), 
					dNodeRec->getIdx(i), newNode, 
					dIdxlShared, dIdxlChk, dIdxlIdx);
	}
	else {
	  mesh[chunk].nodeReplaceDelete(-1, dNodeRec->getIdx(i), newNode, 
					kIdxlShared, kIdxlChk, kIdxlIdx);
	}
      }
      for (i=0; i<kIdxlShared; i++) {
	chunk = kNodeRec->getChk(i);
	if (!dNodeRec || (existsOn(dNodeRec, chunk) == -1)) {
	  mesh[chunk].nodeReplaceDelete(kNodeRec->getIdx(i), -1, newNode, 
					dIdxlShared, dIdxlChk, dIdxlIdx);
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
      //DEBUGREF(CkPrintf("TMRC2D: [%d] ......removing edge %d\n", myRef.cid, myRef.idx);)
      C->removeEdge(myRef.idx);
    }
    return 1;
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
  CkAssert(nonNullElements > 0);
}
