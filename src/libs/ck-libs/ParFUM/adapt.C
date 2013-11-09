/* File: fem_adapt_new.C
 * Authors: Nilesh Choudhury, Terry Wilmarth
 *
 */

#include "ParFUM.h"
#include "ParFUM_internals.h"

//#define DEBUG_1

/** Makes use of ordering of nodes in e1 to check is e3 is on the same side
    of the path of edges (n1, n) and (n, n2) */
int FEM_Adapt::check_orientation(int e1, int e3, int n, int n1, int n2)
{
  int e1_n = find_local_node_index(e1, n);
  int e1_n1 = find_local_node_index(e1, n1);
  int e3_n = find_local_node_index(e3, n);
  int e3_n2 = find_local_node_index(e3, n2);
  
  if (((e1_n1 == (e1_n+1)%3) && (e3_n == (e3_n2+1)%3)) ||
      ((e1_n == (e1_n1+1)%3) && (e3_n2 == (e3_n+1)%3)))
    return 1;
  else return 0;
}

/** Given two element-local node numberings (i.e. 0, 1, 2 for triangular 
    elements), calculate an element-local edge numbering (also 0, 1, or 2
    for triangular elements)
*/
int FEM_Adapt::get_edge_index(int local_node1, int local_node2) 
{
  int sum = local_node1 + local_node2;
  CkAssert(local_node1 != local_node2);
  if (sum == 1) return 0;
  else if (sum == 3) return 1;
  else if (sum == 2) return 2;
  else {
    CkPrintf("ERROR: local node pair is strange: [%d,%d]\n", local_node1,
	    local_node2);
    CkAbort("ERROR: local node pair is strange\n");
    return -1;
  }
}

/** Given a chunk-local element number e and a chunk-local node number n,
    determine the element-local node numbering for node n on element e **/
int FEM_Adapt::find_local_node_index(int e, int n) {
  int result = theMesh->e2n_getIndex(e, n);
  if (result < 0) {
    CkPrintf("ERROR: node %d not found on element %d\n", n, e);
    CkAbort("ERROR: node not found\n");
  }
  return result;
}



/** Extract elements adjacent to edge [n1,n2] along with element-local node
    numberings and nodes opposite input edge **/
int FEM_Adapt::findAdjData(int n1, int n2, int *e1, int *e2, int *e1n1, 
			    int *e1n2, int *e1n3, int *e2n1, int *e2n2, 
			    int *e2n3, int *n3, int *n4)
{
  // Set some default values in case e1 is not there
  (*e1n1) = (*e1n2) = (*e1n3) = (*n3) = -1;

  //if n1,n2 is not an edge return 
  if(n1<0 || n2<0) return -1;
  if(theMesh->node.is_valid(n1)==0 || theMesh->node.is_valid(n2)==0) return -1;
  if(theMesh->n2n_exists(n1,n2)!=1 || theMesh->n2n_exists(n2,n1)!=1) return -1; 

  (*e1) = theMesh->getElementOnEdge(n1, n2); // assumed to return local element
  if ((*e1) == -1) {
    CkPrintf("[%d]Warning: No Element on edge %d->%d\n",theMod->idx,n1,n2);
    return -1;
  }
  (*e1n1) = find_local_node_index((*e1), n1);
  (*e1n2) = find_local_node_index((*e1), n2);
  (*e1n3) = 3 - (*e1n1) - (*e1n2);
  (*n3) = theMesh->e2n_getNode((*e1), (*e1n3));
  (*e2) = theMesh->e2e_getNbr((*e1), get_edge_index((*e1n1), (*e1n2)));
  // Set some default values in case e2 is not there
  (*e2n1) = (*e2n2) = (*e2n3) = (*n4) = -1;
  if ((*e2) != -1) { // e2 exists
    (*e2n1) = find_local_node_index((*e2), n1);
    (*e2n2) = find_local_node_index((*e2), n2);
    (*e2n3) = 3 - (*e2n1) - (*e2n2);
    //if ((*e2) > -1) { // if e2 is a ghost, there is no e2n data
    (*n4) = theMesh->e2n_getNode((*e2), (*e2n3));
    //}
  }
  if(*n3 == *n4) {
    CkPrintf("[%d]Warning: Identical elements %d:(%d,%d,%d) & %d:(%d,%d,%d)\n",theMod->idx,*e1,n1,n2,*n3,*e2,n1,n2,*n4);
    return -1;
  }
  return 1;
}

int FEM_Adapt::e2n_getNot(int e, int n1, int n2) {
  int eConn[3];
  theMesh->e2n_getAll(e, eConn);
  for (int i=0; i<3; i++) 
    if ((eConn[i] != n2) && (eConn[i] != n1)) return eConn[i];
  return -1; //should never come here
}

int FEM_Adapt::n2e_exists(int n, int e) {
  int *nConn, nSz;
  theMesh->n2e_getAll(n, nConn, nSz);
  for (int i=0; i<nSz; i++) {
    if (nConn[i] == e) {
      if(nSz!=0) free(nConn);
      return 1;
    }
  }
  if(nSz!=0) free(nConn);
  return 0;
}

int FEM_Adapt::findElementWithNodes(int n1, int n2, int n3) {
  int *nConn, nSz;
  int ret = -1;
  theMesh->n2e_getAll(n1, nConn, nSz);
  for (int i=0; i<nSz; i++) {
    if ((n2e_exists(n2, nConn[i])) && (n2e_exists(n3, nConn[i]))) {
      ret = nConn[i];
      break;
    }
  }
  if(nSz!=0) free(nConn);
  return ret; //should never come here
}



int FEM_Adapt::getSharedNodeIdxl(int n, int chk) {
  return theMod->getfmUtil()->exists_in_IDXL(theMesh, n, chk, 0, -1);
}

int FEM_Adapt::getGhostNodeIdxl(int n, int chk) { 
  return theMod->getfmUtil()->exists_in_IDXL(theMesh, n, chk, 2, -1);
}

int FEM_Adapt::getGhostElementIdxl(int e, int chk) { 
  return theMod->getfmUtil()->exists_in_IDXL(theMesh, e, chk, 4, 0);
}



void FEM_Adapt::printAdjacencies(int *nodes, int numNodes, int *elems, int numElems) {

  for(int i=0; i<numNodes; i++) {
    if(nodes[i] == -1) continue;
    theMod->getfmUtil()->FEM_Print_n2e(theMesh, nodes[i]);
    theMod->getfmUtil()->FEM_Print_n2n(theMesh, nodes[i]);
  }
  for(int i=0; i<numElems; i++) {
    if(elems[i] == -1) continue;
    theMod->getfmUtil()->FEM_Print_e2n(theMesh, elems[i]);
    theMod->getfmUtil()->FEM_Print_e2e(theMesh, elems[i]);
  }

  return;
}



bool FEM_Adapt::isFixedNode(int n1) {
  for(int i=0; i<theMod->fmfixedNodes.size(); i++) {
    if(theMod->fmfixedNodes[i]==n1) return true;
  }
  return false;
}

/** A node is a corner if it is connected to two different boundaries and it is 
    on the boundary
*/
bool FEM_Adapt::isCorner(int n1) {
  //if it has at least two adjacent nodes on different boundaries and the edges are boundaries
  int *n1AdjNodes;
  int n1NumNodes=0;
  int n1_bound, n2_bound;
  FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n1_bound, n1, 1 , FEM_INT, 1);
  if(n1_bound==0) return false; //it is internal
  theMesh->n2n_getAll(n1, n1AdjNodes, n1NumNodes);
  for (int i=0; i<n1NumNodes; i++) {
    int n2 = n1AdjNodes[i];
    if(FEM_Is_ghost_index(n2)) {
      int numchunks;
      IDXL_Share **chunks1;
      theMod->fmUtil->getChunkNos(0,n2,&numchunks,&chunks1);
      int index = theMod->idx;
      CkAssert(numchunks>0);
      int chk = chunks1[0]->chk;
      int ghostidx = theMod->fmUtil->exists_in_IDXL(theMesh,n2,chk,2);
      intMsg *im = meshMod[chk].getRemoteBound(index,ghostidx);
      n2_bound = im->i;
      for(int j=0; j<numchunks; j++) {
	delete chunks1[j];
      }
      delete im;
      if(numchunks>0) free(chunks1);
    }
    else {
      FEM_Mesh_dataP(theMesh, FEM_NODE, FEM_BOUNDARY, &n2_bound, n2, 1 , FEM_INT, 1);
    }
    if(n2_bound == 0) continue;
    if(n1_bound != n2_bound) {
      if(isEdgeBoundary(n1,n2) && abs(n1_bound)>abs(n2_bound)) {
	if(n1NumNodes!=0) delete[] n1AdjNodes;
	return true;
      }
    }
  }
  if(n1NumNodes!=0) delete[] n1AdjNodes;
  return false;
}

bool FEM_Adapt::isEdgeBoundary(int n1, int n2) {
  int *n1AdjElems, *n2AdjElems;
  int n1NumElems=0, n2NumElems=0;
  int ret = 0;
  //find the number of elements this edge belongs to
  theMesh->n2e_getAll(n1, n1AdjElems, n1NumElems);
  theMesh->n2e_getAll(n2, n2AdjElems, n2NumElems);
  for(int k=0; k<n1NumElems; k++) {
    for (int j=0; j<n2NumElems; j++) {
      if (n1AdjElems[k] == n2AdjElems[j]) {
	if(n1AdjElems[k] != -1) {
	  ret++;
	}
      }
    }
  }
  if(n1NumElems!=0) delete[] n1AdjElems;
  if(n2NumElems!=0) delete[] n2AdjElems;
  if(ret==1) return true;
  return false;
}



// ======================  BEGIN edge_flip  =================================
/** Perform a Delaunay flip of the edge (n1, n2) returning 1 if successful, 0 if
   not (likely due to the edge being on a boundary). The convexity of the 
   quadrilateral formed by two faces incident to edge (n1, n2) is assumed. n1 
   and n2 are assumed to be local to this chunk.  An adjacency test is 
   performed on n1 and n2 by searching for an element with edge [n1,n2].

       n3                 n3
        o                  o
       / \                /|\
      /   \              / | \
     /     \            /  |  \
    /       \          /   |   \
n1 o---------o n2  n1 o    |    o n2
    \       /          \   |   / 
     \     /            \  |  /
      \   /              \ | /
       \ /                \|/
        o                  o
       n4                 n4
*/
int FEM_Adapt::edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			      int e1_n2, int e1_n3, int n3, int n4, int *locknodes) 
{
  int numNodes = 4;
  int numElems = 2;
  int lockelems[2];
  int elemConn[3];
  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  lockelems[0] = e1;
  lockelems[1] = e2;
  if(n1 < 0 || n2 < 0) {
    return -1;
  }
  int index = theMod->getIdx();
  bool flag = theMod->fmAdaptAlgs->controlQualityF(n1,n2,n3,n4);
  if(flag) return -1;
  int e1Topurge = e1;
  int e2Topurge = e2;

#ifdef DEBUG_1
  CkPrintf("Flipping edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
#endif
  //FEM_Modify_Lock(theMesh, locknodes, numNodes, lockelems, numElems);
  //if any of the two elements is remote, eat those
  if(n3 < 0) {
    e1Topurge = theMod->fmUtil->eatIntoElement(e1);
    theMesh->e2n_getAll(e1Topurge,elemConn);
    for(int i=0; i<3; i++) {
      if(elemConn[i]!=n1 && elemConn[i]!=n2) {
	n3 = elemConn[i];
      }
    }
    locknodes[2] = n3;
  }
  if(n4 < 0) {
    e2Topurge = theMod->fmUtil->eatIntoElement(e2);
    theMesh->e2n_getAll(e2Topurge,elemConn);
    for(int i=0; i<3; i++) {
      if(elemConn[i]!=n1 && elemConn[i]!=n2) {
	n4 = elemConn[i];
      }
    }
    locknodes[3] = n4;
  }

  FEM_remove_element(theMesh,e1Topurge,0,0);
  FEM_remove_element(theMesh,e2Topurge,0,0);
  // add n1, n3, n4
  elemConn[e1_n1] = n1;  elemConn[e1_n2] = n4;  elemConn[e1_n3] = n3;
  lockelems[0] = FEM_add_element(theMesh, elemConn, 3, 0, index);
  //the attributes should really be interpolated, i.e. on both new elems,
  //the values should be an average of the previous two elements
  theMod->fmUtil->copyElemData(0,e1Topurge,lockelems[0]);
  // add n2, n3, n4
  elemConn[e1_n1] = n4;  elemConn[e1_n2] = n2;  elemConn[e1_n3] = n3;
  lockelems[1] = FEM_add_element(theMesh, elemConn, 3, 0, index);
  theMod->fmUtil->copyElemData(0,e1Topurge,lockelems[1]); //both of the new elements copy from one element
  //purge the two elements
  FEM_purge_element(theMesh,e1Topurge,0);
  FEM_purge_element(theMesh,e2Topurge,0);

  //get rid of some unnecessary ghost node sends
  for(int i=0; i<4;i++) {
    int nodeToUpdate = -1;
    if(i==0) nodeToUpdate = n1;
    else if(i==1) nodeToUpdate = n2;
    else if(i==2) nodeToUpdate = n3;
    else if(i==3) nodeToUpdate = n4;
    //if any of the chunks sharing this node sends this as a ghost, then all of them have to
    //so find out the set of chunks I need to send this as a ghost to
    //collect info from each of the shared chunks, do a union of all these chunks
    //send this updated list to everyone.
    //if anyone needs to add or delete some ghosts, they will
    int *chkl, numchkl=0;
    CkVec<int> finalchkl;
    theMod->fmUtil->findGhostSend(nodeToUpdate, chkl, numchkl);
    for(int j=0; j<numchkl; j++) {
      finalchkl.push_back(chkl[j]);
    }
    if(numchkl>0) delete[] chkl;

    const IDXL_Rec *irec=theMesh->node.shared.getRec(nodeToUpdate);
    int numchunks=0;
    int *chunks1, *inds1;
    if(irec) {
      numchunks = irec->getShared();
      chunks1 = new int[numchunks];
      inds1 = new int[numchunks];
      for(int j=0; j<numchunks; j++) {
	chunks1[j] = irec->getChk(j);
	inds1[j] = irec->getIdx(j);
      }
    }
    for(int j=0; j<numchunks; j++) {
      findgsMsg *fmsg = meshMod[chunks1[j]].findghostsend(index,inds1[j]);
      if(fmsg->numchks>0) {
	for(int k=0; k<fmsg->numchks; k++) {
	  bool shouldbeadded = true;
	  for(int l=0; l<finalchkl.size(); l++) {
	    if(fmsg->chunks[k]==finalchkl[l]) {
	      shouldbeadded = false;
	      break;
	    }
	  }
	  if(shouldbeadded) finalchkl.push_back(fmsg->chunks[k]);
	}
      }
      delete fmsg;
    }

    int *finall=NULL, numfinall=finalchkl.size();
    if(numfinall>0) finall = new int[numfinall];
    for(int j=0; j<numfinall; j++) finall[j] = finalchkl[j];
    finalchkl.free();

    theMod->fmUtil->UpdateGhostSend(nodeToUpdate, finall, numfinall);
    for(int j=0; j<numchunks; j++) {
      verifyghostsendMsg *vmsg = new(numfinall)verifyghostsendMsg();
      vmsg->fromChk = index;
      vmsg->sharedIdx = inds1[j];
      vmsg->numchks = numfinall;
      for(int k=0; k<numfinall; k++) vmsg->chunks[k] = finall[k];
      meshMod[chunks1[j]].updateghostsend(vmsg);
    }
    if(numfinall>0) delete[] finall;
    if(numchunks>0) {
      delete[] chunks1;
      delete[] inds1;
    }
  }
  //make sure that it always comes here, don't return with unlocking
  return 1; //return newNode;
}
// ======================  END edge_flip  ===================================


// ======================  BEGIN edge_bisect  ===============================
/** Given edge e:(n1, n2), remove the two elements (n1,n2,n3) and 
   (n2,n1,n4) adjacent to e, and bisect e by adding node 
   n5. Add elements (n1,n5,n3), (n5,n2,n3), (n5,n1,n4) and (n2,n5,n4); 
   returns new node n5.

       n3                 n3
        o                  o
       / \                /|\
      /   \              / | \
     /     \            /  |  \
    /       \          /   |n5 \
n1 o---------o n2  n1 o----o----o n2
    \       /          \   |   / 
     \     /            \  |  /
      \   /              \ | /
       \ /                \|/
        o                  o
       n4                 n4
*/
int FEM_Adapt::edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
				int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
				int e2_n3, int n3, int n4)
{
  int n5;
  int numNodes = 4;
  int numElems = 2;
  int numNodesNew = 5;
  int numElemsNew = 4;
  int locknodes[5];
  int lockelems[4];
  int elemConn[3];

  locknodes[0] = n1;
  locknodes[1] = n2;
  locknodes[2] = n3;
  locknodes[3] = n4;
  locknodes[4] = -1;
  lockelems[0] = e1;
  lockelems[1] = e2;
  lockelems[2] = -1;
  lockelems[3] = -1;

  //FEM_Modify_Lock(theMesh, locknodes, numNodes, lockelems, numElems);

  int e1chunk=-1, e2chunk=-1, e3chunk=-1, e4chunk=-1, n5chunk=-1;
  int index = theMod->getIdx();

#ifdef DEBUG_1
  CkPrintf("Bisect edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
#endif
  //verify quality
  bool flag = theMod->fmAdaptAlgs->controlQualityR(n1,n2,n3,n4);
  if(flag) return -1;

  //add node
  if(e1==-1) e1chunk=-1;
  else if(e1>=0) e1chunk=index;
  else {
    int ghostid = FEM_To_ghost_index(e1);
    const IDXL_Rec *irec = theMesh->elem[0].ghost->ghostRecv.getRec(ghostid);
    CkAssert(irec->getShared()==1);
    e1chunk = irec->getChk(0);
  }
  if(e2==-1) e2chunk=-1;
  else if(e2>=0) e2chunk=index;
  else {
    int ghostid = FEM_To_ghost_index(e2);
    const IDXL_Rec *irec = theMesh->elem[0].ghost->ghostRecv.getRec(ghostid);
    CkAssert(irec->getShared()==1);
    e2chunk = irec->getChk(0);
  }
  int adjnodes[2];
  adjnodes[0] = n1;
  adjnodes[1] = n2;
  int *chunks;
  int numChunks=0;
  int forceshared = 0;
  if(e1chunk==e2chunk || (e1chunk==-1 || e2chunk==-1)) {
    forceshared = -1;
    numChunks = 1;
    chunks = new int[1];
    if(e1chunk!=-1) chunks[0] = e1chunk;
    else chunks[0] = e2chunk;
  }
  else {
    numChunks = 2;
    chunks = new int[2];
    chunks[0] = e1chunk;
    chunks[1] = e2chunk;
  }
  n5 = FEM_add_node(theMesh,adjnodes,2,chunks,numChunks,forceshared);
  delete[] chunks;
  //lock this node immediately
  FEM_Modify_LockN(theMesh, n5, 0);

  //remove elements
  e1chunk = FEM_remove_element(theMesh, e1, 0); 
  e2chunk = FEM_remove_element(theMesh, e2, 0);  // assumes intelligent behavior when no e2 exists

  // hmm... if e2 is a ghost and we remove it and create all the new elements
  // locally, then we don't really need to add a *shared* node
  //but we are not moving chunk boundaries for bisect
  if(e1chunk==-1 || e2chunk==-1) {
    //it is fine, let it continue
    e4chunk = e2chunk;
    e3chunk = e1chunk;
  }
  else if(e1chunk==e2chunk && e1chunk!=index) {
    n5chunk = e1chunk;
    e4chunk = e2chunk;
    e3chunk = e1chunk;
  }
  else {
    //there can be a lot of conditions, but we do not have to do aything special now
    n5chunk = -1;
    e4chunk = e2chunk;
    e3chunk = e1chunk;
  }

  // add n1, n5, n3
  elemConn[e1_n1] = n1;  elemConn[e1_n2] = n5;  elemConn[e1_n3] = n3;
  lockelems[0] = FEM_add_element(theMesh, elemConn, 3, 0, e1chunk);
  theMod->fmUtil->copyElemData(0,e1,lockelems[0]);
  // add n2, n5, n3
  elemConn[e1_n1] = n5;  elemConn[e1_n2] = n2;  elemConn[e1_n3] = n3;
  lockelems[1] = FEM_add_element(theMesh, elemConn, 3, 0, e3chunk);
  theMod->fmUtil->copyElemData(0,e1,lockelems[1]);
  if (e2 != -1) { // e2 exists
    // add n1, n5, n4
    elemConn[e2_n1] = n1;  elemConn[e2_n2] = n5;  elemConn[e2_n3] = n4;
    lockelems[2] = FEM_add_element(theMesh, elemConn, 3, 0, e2chunk);
    theMod->fmUtil->copyElemData(0,e2,lockelems[2]);
    // add n2, n5, n4
    elemConn[e2_n1] = n5;  elemConn[e2_n2] = n2;  elemConn[e2_n3] = n4;
    lockelems[3] = FEM_add_element(theMesh, elemConn, 3, 0, e4chunk);
    theMod->fmUtil->copyElemData(0,e2,lockelems[3]);
  }

  FEM_purge_element(theMesh,e1,0);
  FEM_purge_element(theMesh,e2,0);

  //get rid of some unnecessary ghost node sends
  for(int i=0; i<4;i++) {
    int nodeToUpdate = -1;
    if(i==0) nodeToUpdate = n1;
    else if(i==1) nodeToUpdate = n2;
    else if(i==2) nodeToUpdate = n3;
    else if(i==3) nodeToUpdate = n4;
    //if any of the chunks sharing this node sends this as a ghost, then all of them have to
    //so find out the set of chunks I need to send this as a ghost to
    //collect info from each of the shared chunks, do a union of all these chunks
    //send this updated list to everyone.
    //if anyone needs to add or delete some ghosts, they will
    int *chkl, numchkl=0;
    CkVec<int> finalchkl;
    theMod->fmUtil->findGhostSend(nodeToUpdate, chkl, numchkl);
    for(int j=0; j<numchkl; j++) {
      finalchkl.push_back(chkl[j]);
    }
    if(numchkl>0) delete[] chkl;

    const IDXL_Rec *irec=theMesh->node.shared.getRec(nodeToUpdate);
    int numchunks=0;
    int *chunks1, *inds1;
    if(irec) {
      numchunks = irec->getShared();
      chunks1 = new int[numchunks];
      inds1 = new int[numchunks];
      for(int j=0; j<numchunks; j++) {
	chunks1[j] = irec->getChk(j);
	inds1[j] = irec->getIdx(j);
      }
    }
    for(int j=0; j<numchunks; j++) {
      findgsMsg *fmsg = meshMod[chunks1[j]].findghostsend(index,inds1[j]);
      if(fmsg->numchks>0) {
	for(int k=0; k<fmsg->numchks; k++) {
	  bool shouldbeadded = true;
	  for(int l=0; l<finalchkl.size(); l++) {
	    if(fmsg->chunks[k]==finalchkl[l]) {
	      shouldbeadded = false;
	      break;
	    }
	  }
	  if(shouldbeadded) finalchkl.push_back(fmsg->chunks[k]);
	}
      }
      delete fmsg;
    }

    int *finall=NULL, numfinall=finalchkl.size();
    if(numfinall>0) finall = new int[numfinall];
    for(int j=0; j<numfinall; j++) finall[j] = finalchkl[j];
    finalchkl.free();

    theMod->fmUtil->UpdateGhostSend(nodeToUpdate, finall, numfinall);
    for(int j=0; j<numchunks; j++) {
      verifyghostsendMsg *vmsg = new(numfinall)verifyghostsendMsg();
      vmsg->fromChk = index;
      vmsg->sharedIdx = inds1[j];
      vmsg->numchks = numfinall;
      for(int k=0; k<numfinall; k++) vmsg->chunks[k] = finall[k];
      meshMod[chunks1[j]].updateghostsend(vmsg);
    }
    if(numfinall>0) delete[] finall;
    if(numchunks>0) {
      delete[] chunks1;
      delete[] inds1;
    }
  }

  FEM_Modify_UnlockN(theMesh, n5, 0);
  return n5;
}
// ======================  END edge_bisect  ================================


// ======================  BEGIN vertex_remove  ============================
/** Inverse of edge bisect, this removes a degree 4 vertex n1 and 2 of its
   adjacent elements.  n2 indicates that the two elements removed are
   adjacent to edge [n1,n2]. This could be performed with edge_contraction,
   but this is a simpler operation. 

         n3	             n3        
          o	              o        
         /|\	             / \       
        / | \	            /   \      
       /  |  \	           /     \     
      /   |n1 \           /       \    
  n5 o----o----o n2   n5 o---------o n2
      \   |   /           \       /    
       \  |  /	           \     /     
        \ | /	            \   /      
         \|/	             \ /       
          o	              o        
         n4                  n4        
*/
int FEM_Adapt::vertex_remove_help(int e1, int e2, int n1, int n2, int e1_n1, 
				  int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
				  int e2_n3, int n3, int n4, int n5)
{
  int numNodes = 5;
  int numElems = 4;
  int numNodesNew = 4;
  int numElemsNew = 2;
  int locknodes[5];
  int lockelems[4];
  int elemConn[3];

  locknodes[0] = n2;
  locknodes[1] = n3;
  locknodes[2] = n4;
  locknodes[3] = n5;
  locknodes[4] = n1;
  lockelems[0] = e1;
  lockelems[1] = e2;
  lockelems[2] = -1;
  lockelems[3] = -1;

  int e3 = theMesh->e2e_getNbr(e1, get_edge_index(e1_n1, e1_n3));
  int e4 = -1;
  lockelems[2] = e3;
  if (e3 != -1) {
    if (e2 != -1) {
      e4 = theMesh->e2e_getNbr(e2, get_edge_index(e2_n1, e2_n3));
      lockelems[3] = e4;
      if(e4 == -1 ) {
	return 0;
      }
    }
    //FEM_Modify_Lock(theMesh, locknodes, numNodes, lockelems, numElems);

    int e1chunk=-1, e2chunk=-1, e3chunk=-1, e4chunk=-1;
    int index = theMod->getIdx();

#ifdef DEBUG_1
    CkPrintf("Vertex Remove edge %d->%d on chunk %d\n", n1, n2, theMod->getfmUtil()->getIdx());
#endif
    
    e1chunk = FEM_remove_element(theMesh, e1, 0);
    e3chunk = FEM_remove_element(theMesh, e3, 0);
    if (e2 != -1) {
      e2chunk = FEM_remove_element(theMesh, e2, 0);
      e4chunk = FEM_remove_element(theMesh, e4, 0);
    }
    FEM_remove_node(theMesh, n1);
    
    // add n2, n5, n3
    elemConn[e1_n1] = n2;  elemConn[e1_n2] = n3;  elemConn[e1_n3] = n5;
    lockelems[0] = FEM_add_element(theMesh, elemConn, 3, 0, e1chunk);
    if (e2 != -1) {
      // add n2, n5, n4
      elemConn[e2_n1] = n5;  elemConn[e2_n2] = n4;  elemConn[e2_n3] = n2;
      lockelems[1] = FEM_add_element(theMesh, elemConn, 3, 0, e2chunk);
    }

    return 1;
  }

  return 0;
}
// ======================  END vertex_remove  ==============================
  
// ======================  BEGIN vertex_split =================================
/** Given a node n and two adjacent nodes n1 and n2, split n into two nodes n 
   and np such that the edges to the neighbors n1 and n2 expand into two new 
   elements (n, np, n1) and (np, n, n2); return the id of the newly created 
   node np

    n1	            n1             
     o	             o             
     |	            / \            
     |             /   \           
   \ | /      \   /     \   /      
    \|/        \ /       \ /       
     o n      n o---------o np
    /|\        / \       / \       
   / | \      /   \     /   \      
     |             \   /           
     | 	            \ /            
     o	             o             
    n2              n2             
*/
/* This function has some undefined characteristics.. please do not use it as of now*/
int FEM_Adapt::vertex_split(int n, int n1, int n2) 
{
  if ((n < 0) || ((n1 <= -1) && (n2 <= -1)))
    CkAbort("FEM_Adapt::vertex_split: n and at least one of its neighbor must be local to this chunk; n1 and n2 must both exist\n");
  int locknodes[2];
  locknodes[0] = n1; locknodes[1] = n2;
  FEM_Modify_Lock(theMesh, locknodes, 2, locknodes, 0);
  int e1 = theMesh->getElementOnEdge(n, n1);
  if (e1 == -1) {
    FEM_Modify_Unlock(theMesh);
    return -1;	     
  }
  int e3 = theMesh->getElementOnEdge(n, n2);
  if (e3 == -1) {
    FEM_Modify_Unlock(theMesh);
    return -1;	     
  }
  int ret = vertex_split_help(n, n1, n2, e1, e3);
  FEM_Modify_Unlock(theMesh);
  return ret;
}

int FEM_Adapt::vertex_split_help(int n, int n1, int n2, int e1, int e3)
{
  int e1_n = find_local_node_index(e1, n);
  int e1_n1 = find_local_node_index(e1, n1);
  int e2 = theMesh->e2e_getNbr(e1, get_edge_index(e1_n, e1_n1));
  int e3_n = find_local_node_index(e3, n);
  int e3_n2 = find_local_node_index(e3, n2);
  int e4 = theMesh->e2e_getNbr(e3, get_edge_index(e3_n, e3_n2));
  if (!check_orientation(e1, e3, n, n1, n2)) {
    int tmp = e3;
    e3 = e4;
    e4 = tmp;
    e3_n = find_local_node_index(e3, n);
    e3_n2 = find_local_node_index(e3, n2);
  }

  int locknodes[4];
  locknodes[0] = n1;
  locknodes[1] = n;
  locknodes[2] = n2;
  locknodes[3] = -1;
  int lockelems[6];
  lockelems[0] = e1;
  lockelems[1] = e2;
  lockelems[2] = e3;
  lockelems[3] = e4;
  lockelems[4] = -1;
  lockelems[5] = -1;
  int elemConn[3];

  //FEM_Modify_Lock(theMesh, locknodes, 4, lockelems, 6);
#ifdef DEBUG_1
  CkPrintf("Vertex Split, %d-%d-%d on chunk %d\n", n1, n, n2, theMod->getfmUtil()->getIdx());
#endif
  int adjnodes[2];
  adjnodes[0] = n; //looks like it will never be shared, since according to later code, all n1, n & n2 should be local.. appears to be not correct
  //the new node will be shared to wahtever the old node was shared to, we'll do this later
  int *chunks = NULL;
  int numChunks = 0;
  int np = FEM_add_node(theMesh,adjnodes,1,chunks,numChunks,0);
  locknodes[3] = np;

  int current, next, nt, nl, eknp, eknt, eknl;
  // traverse elements on one side of n starting with e2
  current = e2;
  nt = n1;
  while ((current != e3) && (current != -1)) { 
    eknp = find_local_node_index(current, n);
    eknt = find_local_node_index(current, nt);
    eknl = 3 - eknp - eknt;
    next = theMesh->e2e_getNbr(current, get_edge_index(eknp, eknl));
    nl = theMesh->e2n_getNode(current, eknl);
    FEM_remove_element(theMesh, current, 0);
    // add nl, nt, np
    elemConn[eknp] = np; elemConn[eknt] = nt; elemConn[eknl] = nl;
    int newelem = FEM_add_element(theMesh, elemConn, 3, 0);
    nt = nl;
    current = next;
  }
  if (current == -1) { // didn't make it all the way around
    // traverse elements on one side of n starting with e4
    current = e4;
    nt = n2;
    while ((current != e1) && (current != -1)) {
      eknp = find_local_node_index(current, n);
      eknt = find_local_node_index(current, nt);
      eknl = 3 - eknp - eknt;
      next = theMesh->e2e_getNbr(current, get_edge_index(eknp, eknl));
      nl = theMesh->e2n_getNode(current, eknl);
      FEM_remove_element(theMesh, current, 0);
      // add nl, nt, np
      elemConn[eknp] = np; elemConn[eknt] = nt; elemConn[eknl] = nl;
      int newelem = FEM_add_element(theMesh, elemConn, 3, 0);
      nt = nl;
      current = next;
    }
  }

  // add n, n1, np
  elemConn[e1_n] = n; elemConn[e1_n1] = n1; elemConn[3 - e1_n - e1_n1] = np;
  lockelems[4] = FEM_add_element(theMesh, elemConn, 3, 0);
  // add n, n2, np
  elemConn[e3_n] = n; elemConn[e3_n2] = n2; elemConn[3 - e3_n - e3_n2] = np;
  lockelems[5] = FEM_add_element(theMesh, elemConn, 3, 0);

  return np;
}
// ======================  END vertex_split ===================


int FEM_AdaptL::eatIntoElement(int e, bool aggressive_node_removal)
{
    return theMod->fmUtil->eatIntoElement(e, aggressive_node_removal);
}


void FEM_AdaptL::residualLockTest()
{
    theMod->fmUtil->residualLockTest(theMod->fmMesh);
}


void FEM_AdaptL::unlockAll() {
    theMod->fmUtil->unlockAll(theMod->fmMesh);
}


void FEM_AdaptL::structureTest()
{
    theMod->fmUtil->StructureTest(theMod->fmMesh);
}
