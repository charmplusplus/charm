/* File: fem_util.C
 * Authors: Nilesh Choudhury
 * 
 */


#include "fem_util.h"
#include "fem_mesh_modify.h"
#include <vector>

extern void splitEntity(IDXL_Side &c, int localIdx, int nBetween, int *between, int idxbase);

FEM_MUtil::FEM_MUtil(int i, femMeshModify *m) {
  idx = i;
  mmod = m;
}

FEM_MUtil::~FEM_MUtil() {
}

void FEM_MUtil::getChunkNos(int entType, int entNo, int *numChunks, IDXL_Share ***chunks, int elemType) {
  int type = 0; //0 - local, 1 - shared, 2 - ghost.

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(entType == 0) { //nodes
    //only nodes can be shared
    if(FEM_Is_ghost_index(entNo)) type = 2;
    else if(isShared(entNo)) type = 1;
    else type = 0;

    if(type == 2) {
      int ghostid = FEM_To_ghost_index(entNo);
      int noShared = 0;
      const IDXL_Rec *irec = mmod->fmMesh->node.ghost->ghostRecv.getRec(ghostid);
      if(irec) {
	noShared = irec->getShared(); //check this value!!
	//CkAssert(noShared > 0);
	*numChunks = noShared;
	*chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
	int i=0;
	for(i=0; i<*numChunks; i++) {
	  int chk = irec->getChk(i); 
	  int index = -1; // no need to have these, I never use it anyway
	  (*chunks)[i] = new IDXL_Share(chk, index);
	}
	//(*chunks)[i] = new IDXL_Share(chunk, -1);
      }
      else { //might be that it does not exist anymore in the ghost list
	*numChunks = 0;
      }
    }
    else if(type == 1) {
      const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(entNo);
      if(irec) {
	*numChunks = irec->getShared() + 1; //add myself to the list
	*chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
	int i=0;
	for(i=0; i<*numChunks-1; i++) {
	  int chk = irec->getChk(i);
	  int index = irec->getIdx(i);
	  (*chunks)[i] = new IDXL_Share(chk, index);
	}
	(*chunks)[i] = new IDXL_Share(idx, -1);
      }
      else { //might be that it does not exist anymore in the ghost list
	*numChunks = 0;
      }
    }
    else if(type == 0) {
      *numChunks = 0;
      *chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = idx; //index of this chunk
	int index = entNo;
	(*chunks)[i] = new IDXL_Share(chk, index);
      }
    }
  }
  else if(entType == 1) { //elems
    //elements cannot be shared
    if(FEM_Is_ghost_index(entNo)) type = 2;
    else type = 0;

    if(type == 2) {
      int ghostid = FEM_To_ghost_index(entNo);
      const IDXL_Rec *irec = mmod->fmMesh->elem[elemType].ghost->ghostRecv.getRec(ghostid);
      if(irec) {
	*numChunks = irec->getShared(); //should be 1
	*chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
	for(int i=0; i<*numChunks; i++) {
	  int chk = irec->getChk(i);
	  int index = irec->getIdx(i);
	  (*chunks)[i] = new IDXL_Share(chk, index);
	}
      }
      else { //might be that it does not exist anymore in the ghost list
	*numChunks = 0;
      }
    }
    else if(type == 0) {
      *numChunks = 0;
      *chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = idx; //index of this chunk
	int index = entNo;
	(*chunks)[i] = new IDXL_Share(chk, index);
      }
    }
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

bool FEM_MUtil::isShared(int index) {
  //this function will be only called for a shared list
  //have to figure out if node.shared is kept up to date
  const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(index);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  //if an entry exists in the shared idxl lists, then it is a shared node
  if(irec != NULL) return true;
  return false;
}

//An IDXL helper function which is same as splitEntity, but instead of just adding to this chunk's
//idxl list, it will add to the idxl lists of all chunks.
//degenerate now, unless we find some use for it in 3D
void FEM_MUtil::splitEntityAll(FEM_Mesh *m, int localIdx, int nBetween, int *between)
{
  //Find the commRecs for the surrounding nodes
  IDXL_Side *c = &(m->node.shared);
  const IDXL_Rec **tween = (const IDXL_Rec **)malloc(nBetween*sizeof(IDXL_Rec *));
  
  //Make a new commRec as the interesection of the surrounding entities--
  // we loop over the first entity's comm. list
  tween[0] = c->getRec(between[0]);
  for (int zs=tween[0]->getShared()-1; zs>=0; zs--) {
    for (int w1=0; w1<nBetween; w1++) {
      tween[w1] = c->getRec(between[w1]);
    }
    int chk = tween[0]->getChk(zs);
#ifdef DEBUG 
    CmiMemoryCheck(); 
#endif

    //Make sure this processor shares all our entities
    int w = 0;
    for (w=0; w<nBetween; w++) {
      if (!tween[w]->hasChk(chk)) {
	break;
      }
    }

    if (w == nBetween) {//The new node is shared with chk
      //idxllock(m,chk,0);
      idxllock(m,chk,0); //coarser lock
      c->addNode(localIdx,chk); //add in the shared entry of this chunk

      //generate the shared node numbers with chk from the local indices
      int *sharedIndices = (int *)malloc(nBetween*sizeof(int));
      const IDXL_List ll = m->node.shared.addList(chk);
      for(int w1=0; w1<nBetween; w1++) {
	for(int w2=0; w2<ll.size(); w2++) {
	  if(ll[w2] == between[w1]) {
	    sharedIndices[w1] = w2;
	    break;
	  }
	}
      }
      sharedNodeMsg *fm = new (nBetween, 0) sharedNodeMsg;
      fm->chk = mmod->idx;
      fm->nBetween = nBetween;
      for(int j=0; j<nBetween; j++) {
	fm->between[j] = sharedIndices[j];
      }
      meshMod[chk].addSharedNodeRemote(fm);
      idxlunlock(m,chk,0); //coarser lock
      free(sharedIndices);
    }
  }
  free(tween);
  return;
}

//adds this new shared node to chunks only which share this edge(n=2) or face(n=3)
void FEM_MUtil::splitEntitySharing(FEM_Mesh *m, int localIdx, int nBetween, int *between, int numChunks, int *chunks)
{
  for(int i=0; i<numChunks; i++) {
    int chk = chunks[i];
    if(chk==idx) continue;
    sharedNodeMsg *fm = new (nBetween, 0) sharedNodeMsg;
    fm->chk = idx;
    fm->nBetween = nBetween;
    for(int j=0; j<nBetween; j++) {
      fm->between[j] = exists_in_IDXL(m,between[j],chk,0);
      CkAssert(fm->between[j]!=-1);
    }
    idxllock(m,chk,0); //coarser lock
    m->node.shared.addNode(localIdx,chk); //add in the shared entry of this chunk
    meshMod[chk].addSharedNodeRemote(fm);
    idxlunlock(m,chk,0); //coarser lock
  }
  return;
}

void FEM_MUtil::splitEntityRemote(FEM_Mesh *m, int chk, int localIdx, int nBetween, int *between)
{
  //convert the shared indices to local indices
  int *localIndices = (int *)malloc(nBetween*sizeof(int));
  const IDXL_List ll = m->node.shared.addList(chk);
  for(int i=0; i<nBetween; i++) {
    localIndices[i] = ll[between[i]];
  }

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  FEM_Interpolate *inp = m->getfmMM()->getfmInp();
  FEM_Interpolate::NodalArgs nm;
  nm.n = localIdx;
  for(int i=0; i<nBetween; i++) {
    nm.nodes[i] = localIndices[i];
  }
  nm.frac = 0.5;
  nm.addNode = true;
  inp->FEM_InterpolateNodeOnEdge(nm);

  m->node.shared.addNode(localIdx,chk);
  free(localIndices);
  return;
}

void FEM_MUtil::removeNodeAll(FEM_Mesh *m, int localIdx)
{
  IDXL_Side *c = &(m->node.shared);
  const IDXL_Rec *tween = c->getRec(localIdx);
  int size = 0;
  if(tween) 
    size = tween->getShared();
  if(size>0) {
    int *schunks = (int*)malloc(size*sizeof(int));
    int *sidx = (int*)malloc(size*sizeof(int));
    for(int i=0; i<size; i++) {
      schunks[i] = tween->getChk(i);
      sidx[i] = tween->getIdx(i);
    }
    for(int i=0; i<size; i++) {
      removeSharedNodeMsg *fm = new removeSharedNodeMsg;
      fm->chk = mmod->idx;
      fm->index = sidx[i];
      meshMod[schunks[i]].removeSharedNodeRemote(fm);
      m->node.shared.removeNode(localIdx, schunks[i]);      
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
    }
    free(schunks);
    free(sidx);
  }
  //remove it from this chunk
  FEM_remove_node_local(m,localIdx);
  return;
}

//if this index is shared with that chunk, it returns the shared index on the list, otherwise returns -1
int FEM_MUtil::exists_in_IDXL(FEM_Mesh *m, int localIdx, int chk, int type, int elemType) {
  int exists  = -1;
  IDXL_List ll;
  if(type == 0) { //shared node
    ll = m->node.shared.addList(chk);
  }
  else if(type == 1) { //ghost node send 
    ll = m->node.ghostSend.addList(chk);
  }
  else if(type == 2) { //ghost node recv 
    ll = m->node.ghost->ghostRecv.addList(chk);
    localIdx = FEM_To_ghost_index(localIdx);
  }
  else if(type == 3) { //ghost elem send 
    ll = m->elem[elemType].ghostSend.addList(chk);
  }
  else if(type == 4) { //ghost elem recv 
    ll = m->elem[elemType].ghost->ghostRecv.addList(chk);
    localIdx = FEM_To_ghost_index(localIdx);
  }
#ifdef DEBUG 
  CmiMemoryCheck();
#endif
  for(int w2=0; w2<ll.size(); w2++) {
    if(ll[w2] == localIdx) {
      exists = w2;
      break;
    }
  }
  return exists;
}

void FEM_MUtil::removeNodeRemote(FEM_Mesh *m, int chk, int sharedIdx) {
  int localIdx;
  const IDXL_List ll = m->node.shared.addList(chk);
  localIdx = ll[sharedIdx];
  m->node.shared.removeNode(localIdx, chk);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  FEM_remove_node_local(m,localIdx);
  return;
}

void FEM_MUtil::removeGhostNodeRemote(FEM_Mesh *m, int fromChk, int sharedIdx) {
  int localIdx = lookup_in_IDXL(m,sharedIdx,fromChk,2); //look in the ghostrecv list
  if(localIdx >= 0) {
    m->node.ghost->ghostRecv.removeNode(localIdx, fromChk);
    if(m->node.ghost->ghostRecv.getRec(localIdx)==NULL) {
      int ghostid = FEM_To_ghost_index(localIdx);
      int numAdjNodes, numAdjElts;
      int *adjNodes, *adjElts;
      m->n2n_getAll(ghostid, &adjNodes, &numAdjNodes);
      m->n2e_getAll(ghostid, &adjElts, &numAdjElts);
      
      // mark node as deleted/invalid
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
      if(!((numAdjNodes==0) && (numAdjElts==0))) {
	CkPrintf("Error: Node %d cannot be removed, it is connected to :\n",ghostid);
	FEM_Print_n2e(m,ghostid);
	FEM_Print_n2n(m,ghostid);
      }
      //CkAssert((numAdjNodes==0) && (numAdjElts==0));
      m->node.ghost->set_invalid(localIdx,true);
    }
    //else, it still comes as a ghost from some other chunk. That chunk should call a remove on this and it should be deleted then.
  }
  return;
}

void FEM_MUtil::addGhostElementRemote(FEM_Mesh *m, int chk, int elemType, int numGhostIndices, int *ghostIndices, int numSharedIndices, int *sharedIndices, int connSize) {
  int numNewGhostIndices = connSize - (numGhostIndices + numSharedIndices);
  int *conn = (int *)malloc(connSize*sizeof(int));
  for(int i=0; i<numNewGhostIndices; i++) {
    //DOES THIS WORK for 2 new ghost additions.. make sure!
    int newGhostNode = FEM_add_node_local(m, 1);
    m->node.ghost->ghostRecv.addNode(newGhostNode,chk);
    //make this node come as ghost from all chunks that share the node
    int sharedIdx = exists_in_IDXL(m,FEM_To_ghost_index(newGhostNode),chk,2);
    int2Msg *i2 = new int2Msg(idx, sharedIdx);
    chunkListMsg *clm = meshMod[chk].getChunksSharingGhostNode(i2);
    conn[i] = FEM_To_ghost_index(newGhostNode);
    for(int j=0; j<clm->numChunkList; j++) {
      if(clm->chunkList[j]==idx) continue;
      int chk1 = clm->chunkList[j];
      int sharedIdx1 = clm->indexList[j];
      idxllock(m,chk1,0);
      m->node.ghost->ghostRecv.addNode(newGhostNode,chk1);
      //on 'chk1' find the local index on shared list with 'chk' & add that to ghostsend with 'index'
      meshMod[chk1].addTransIDXLRemote(idx,sharedIdx1,chk);
      idxlunlock(m,chk1,0);
    }
    FEM_Ghost_Essential_attributes(m, mmod->fmAdaptAlgs->coord_attr, FEM_BOUNDARY, conn[i]);
  }
  //convert existing remote ghost indices to local ghost indices 
  const IDXL_List ll1 = m->node.ghost->ghostRecv.addList(chk);
  for(int i=0; i<numGhostIndices; i++) {
    conn[i+numNewGhostIndices] = FEM_To_ghost_index(ll1[ghostIndices[i]]);
  }
  //convert sharedIndices to localIndices
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  const IDXL_List ll2 = m->node.shared.addList(chk);
  for(int i=0; i<numSharedIndices; i++) {
    conn[i+numNewGhostIndices+numGhostIndices] = ll2[sharedIndices[i]];
  }
  int newGhostElement = FEM_add_element_local(m, conn, connSize, elemType, 1);
  m->elem[elemType].ghost->ghostRecv.addNode(FEM_To_ghost_index(newGhostElement),chk);
  free(conn);
  return;
}

chunkListMsg *FEM_MUtil::getChunksSharingGhostNodeRemote(FEM_Mesh *m, int chk, int sharedIdx) {
  const IDXL_List ll = m->node.ghostSend.addList(chk);
  int localIdx = ll[sharedIdx];
  int numChunkList = 0;
  const IDXL_Rec *tween = m->node.shared.getRec(localIdx);
  if(tween) {
    numChunkList = tween->getShared();
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  chunkListMsg *clm = new (numChunkList, numChunkList, 0) chunkListMsg;
  clm->numChunkList = numChunkList;
  for(int i=0; i<numChunkList; i++) {
    clm->chunkList[i] = tween->getChk(i);
    clm->indexList[i] = tween->getIdx(i);
  }
  //return a sorted list, small to large
  for(int i=0; i<numChunkList; i++) {
    for(int j=i+1; j<numChunkList; j++) {
      if(clm->chunkList[i]> clm->chunkList[j]) {
	int tmp = clm->chunkList[i];
	clm->chunkList[i] = clm->chunkList[j];
	clm->chunkList[j] = tmp;
	tmp = clm->indexList[i];
	clm->indexList[i] = clm->indexList[j];
	clm->indexList[j] = tmp;
      }
    }
  }
  return clm;
}

void FEM_MUtil::buildChunkToNodeTable(int *nodetype, int sharedcount, int ghostcount, int localcount, int *conn, int connSize, CkVec<int> ***allShared, int *numSharedChunks, CkVec<int> **allChunks, int ***sharedConn) {
  if((sharedcount > 0 && ghostcount == 0) || (ghostcount > 0 && localcount == 0)) { 
    *allShared = (CkVec<int> **)malloc(connSize*sizeof(CkVec<int> *));
    for(int i=0; i<connSize; i++) {
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
      (*allShared)[i] = new CkVec<int>; //list of chunks shared with node
      int numchunks;
      IDXL_Share **chunks1;
      if(nodetype[i] == 1) { //it is a shared node, figure out all chunks where the ghosts need to be added
	getChunkNos(0,conn[i],&numchunks,&chunks1);
      }
      else if(nodetype[i] == 2) { //it is a ghost node, figure out all chunks where the ghosts need to be added
	getChunkNos(0,conn[i],&numchunks,&chunks1);
      }
      else if(nodetype[i] == 0) {
	numchunks = 1;
	(*allShared)[i]->push_back(getIdx());
      }
      if((nodetype[i] == 1) || (nodetype[i] == 2)) {
	for(int j=0; j<numchunks; j++) {
	  (*allShared)[i]->push_back(chunks1[j]->chk);
	}
      }
      if(nodetype[i]==1 || nodetype[i]==2) {
	for(int j=0; j<numchunks; j++) {
	  delete chunks1[j];
	}
	if(numchunks!=0) free(chunks1);
      }
    }
    //translate the information in a reverse data structure -- which chunk has which nodes as shared
    *allChunks = new CkVec<int>;
    for(int i=0; i<connSize; i++) {
      for(int j=0; j<(*allShared)[i]->size(); j++) {
	int exists = 0;
	for(int k=0; k<(*allChunks)->size(); k++) {
	  if((*(*allChunks))[k]==(*(*allShared)[i])[j]) {
	    exists = 1;
	    break;
	  }
	}
	if(!exists) {
	  (*allChunks)->push_back((*(*allShared)[i])[j]);
	  (*numSharedChunks)++;
	}
      }
    }
    *sharedConn = (int **)malloc((*numSharedChunks)*sizeof(int *));
    for(int j=0; j<*numSharedChunks; j++) {
      (*sharedConn)[j] = (int *)malloc(connSize*sizeof(int));
    }
    int index = getIdx();
    for(int i=0; i<connSize; i++) {
      //(*sharedConn)[i] = (int*)malloc((*numSharedChunks)*sizeof(int));
      int chkindex = -1;
      if((nodetype[i] == 1) || (nodetype[i] == 2)) {
	for(int j=0; j<(*numSharedChunks); j++) {//initialize
	  (*sharedConn)[j][i] = -1;
	}
	for(int j=0; j<(*allShared)[i]->size(); j++) {
	  for(int k=0; k<*numSharedChunks; k++) {
	    if((*(*allShared)[i])[j] == (*(*allChunks))[k]) chkindex = k;
	  }
	  (*sharedConn)[chkindex][i] = 1; 
	}
	if(nodetype[i] == 2) {
 	  for(int k=0; k<*numSharedChunks; k++) {
	    if(index == (*(*allChunks))[k]) chkindex = k;
	  }
	  (*sharedConn)[chkindex][i] = 2;
	  if((*allShared)[i]->size()==1) {
	    for(int k=0; k<*numSharedChunks; k++) {
	      if((*(*allShared)[i])[0] == (*(*allChunks))[k]) chkindex = k;
	    }
	    (*sharedConn)[chkindex][i] = 0;
	  }
	}
      }
      else {
	//node 'i' is local hence not shared with any chunk
	for(int j=0; j<(*numSharedChunks); j++) {
	  (*sharedConn)[j][i] = -1; 
	}
	for(int k=0; k<*numSharedChunks; k++) {
	  if(index == (*(*allChunks))[k]) chkindex = k;
	}
	(*sharedConn)[chkindex][i] = 0;
      }
    }
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  CkAssert(*numSharedChunks>0);
  return;
}

void FEM_MUtil::addElemRemote(FEM_Mesh *m, int chk, int elemtype, int connSize, int *conn, int numGhostIndex, int *ghostIndices) {
  //translate all the coordinates to local coordinates
  //chk is the chunk who send this message
  //convert sharedIndices to localIndices & ghost to local indices
  const IDXL_List ll1 = m->node.ghostSend.addList(chk);
  const IDXL_List ll2 = m->node.shared.addList(chk);

  int *localIndices = (int *)malloc(connSize*sizeof(int));
  int j=0;
  int ghostsRemaining = numGhostIndex;
  for(int i=0; i<connSize; i++) {
    if(ghostsRemaining > 0) {
      if(ghostIndices[j] == i) {
	localIndices[i] = ll1[conn[i]];
	ghostsRemaining--;
	j++;
      }
      else {
	localIndices[i] = ll2[conn[i]];
      }
    }
    else {
      localIndices[i] = ll2[conn[i]];
    }
  }

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  FEM_add_element(m, localIndices, connSize, elemtype, idx);
  free(localIndices);
  return;
}


void FEM_MUtil::removeGhostElementRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int numGhostIndex, int *ghostIndices, int numGhostRNIndex, int *ghostRNIndices, int numGhostREIndex, int *ghostREIndices, int numSharedIndex, int *sharedIndices) {
  //translate all ghost node coordinates to local coordinates and delete those ghost nodes on chk
  //remove ghost element elementid on chk
  int localIdx;
  
  const IDXL_List lgre = m->elem[elemtype].ghost->ghostRecv.addList(chk);
  localIdx = lgre[elementid];
  if(localIdx == -1) {
    CkPrintf("Ghost element at shared index %d, already removed\n",elementid);
    return;
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  //purge should do this now
  //m->elem[elemtype].ghost->ghostRecv.removeNode(localIdx, chk);
  FEM_remove_element_local(m, FEM_To_ghost_index(localIdx), elemtype);

  //convert existing remote ghost indices to local ghost indices 
  if(numGhostIndex > 0) {
    const IDXL_List lgrn = m->node.ghost->ghostRecv.addList(chk);
    for(int i=0; i<numGhostIndex; i++) {
      localIdx = lgrn[ghostIndices[i]];
      m->node.ghost->ghostRecv.removeNode(localIdx, chk); //do not delete ghost nodes
      if(m->node.ghost->ghostRecv.getRec(localIdx)) {
	//it is a ghost because of other chunks
      }
      else {
	FEM_remove_node_local(m, FEM_To_ghost_index(localIdx));
      }
    }
  }

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(numGhostREIndex > 0) {
    const IDXL_List lgse = m->elem[elemtype].ghostSend.addList(chk);
    for(int i=0; i<numGhostREIndex; i++) {
      localIdx = lgse[ghostREIndices[i]];
      m->elem[elemtype].ghostSend.removeNode(localIdx, chk); 
    }
  }

  if(numGhostRNIndex > 0) {
    const IDXL_List lgsn = m->node.ghostSend.addList(chk);
    for(int i=0; i<numGhostRNIndex; i++) {
      localIdx = lgsn[ghostRNIndices[i]];
      m->node.ghostSend.removeNode(localIdx, chk); 
    }
  }

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(numSharedIndex > 0) { //a shared node or corner became ghost
    const IDXL_List lsn = m->node.shared.addList(chk);
    for(int i=0; i<numSharedIndex; i++) {
      bool flag1 = true;
      /*if(sharedIndices[i]<-500000000) {
	//a corner, local node, became ghost
	const IDXL_List lgrn = m->node.ghost->ghostRecv.addList(chk);
	if(sharedIndices[i]<-1000000000) {
	  sharedIndices[i] += 500000000;
	  flag1 = false;
	}
	sharedIndices[i] += 1000000000;
	localIdx = lgrn[sharedIndices[i]];
	if(flag1) m->node.ghostSend.addNode(localIdx, chk);
	//ghostrecv remove will be done in addtosharedlist
	//will need to add a new node on this chunk
      }
      else */{
	if(sharedIndices[i]<0 && sharedIndices[i]>=-500000000) {
	  sharedIndices[i] += 500000000;
	  flag1 = false;
	}
	localIdx = lsn[sharedIndices[i]];
	if(flag1) m->node.ghostSend.addNode(localIdx, chk);
	m->node.shared.removeNode(localIdx, chk);
      }
    }
  }

  return;
}

void FEM_MUtil::removeElemRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int permanent) {

  const IDXL_List ll = m->elem[elemtype].ghostSend.addList(chk);
  int localIdx = ll[elementid];
  FEM_remove_element(m, localIdx, elemtype, permanent);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

int FEM_MUtil::lookup_in_IDXL(FEM_Mesh *m, int sharedIdx, int chk, int type, int elemType) {
  int localIdx  = -1;
  IDXL_List ll;
  if(type == 0) { //shared node
    ll = m->node.shared.addList(chk);
  }
  else if(type == 1) { //ghost node send 
    ll = m->node.ghostSend.addList(chk);
  }
  else if(type == 2) { //ghost node recv 
    ll = m->node.ghost->ghostRecv.addList(chk);
  }
  else if(type == 3) { //ghost node recv 
    ll = m->elem[elemType].ghostSend.addList(chk);
  }
  else if(type == 4) { //ghost node recv 
    ll = m->elem[elemType].ghost->ghostRecv.addList(chk);
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  CkAssert(sharedIdx < ll.size());
  localIdx = ll[sharedIdx];
  return localIdx;
}

int FEM_MUtil::getRemoteIdx(FEM_Mesh *m, int elementid, int elemtype) {
  CkAssert(elementid < -1);
  int ghostid = FEM_To_ghost_index(elementid);
  const IDXL_Rec *irec = m->elem[elemtype].ghost->ghostRecv.getRec(ghostid);
  int size = irec->getShared();
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  CkAssert(size == 1);
  int remoteChunk = irec->getChk(0);
  int sharedIdx = irec->getIdx(0);
  return remoteChunk;
}

int FEM_MUtil::Replace_node_local(FEM_Mesh *m, int oldIdx, int newIdx) {
  //removes oldIdx and copies all its attributes to newIdx
  if(newIdx==-1) {
    newIdx = m->node.size();
    m->node.setLength(newIdx+1); // lengthen node attributes
    m->node.set_valid(newIdx,true);   // set new node as valid
    m->n2e_removeAll(newIdx);    // initialize element adjacencies
    m->n2n_removeAll(newIdx);    // initialize node adjacencies
    mmod->fmLockN.push_back(new FEM_lockN(newIdx,mmod));
    mmod->fmLockN[newIdx]->wlock(idx); //lock it anyway, will unlock if needed in lockupdate
  }

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  FEM_Interpolate *inp = mmod->getfmInp();
  inp->FEM_InterpolateCopyAttributes(oldIdx,newIdx);
  
  //update the conectivity of neighboring nodes and elements
  int *nnbrs;
  int nsize;
  m->n2n_getAll(oldIdx, &nnbrs, &nsize);
  m->n2n_removeAll(newIdx);
  for(int i=0; i<nsize; i++) {
    m->n2n_add(newIdx, nnbrs[i]);
    m->n2n_replace(nnbrs[i], oldIdx, newIdx);
  }
  int *enbrs;
  int esize;
  m->n2e_getAll(oldIdx, &enbrs, &esize);
  m->n2e_removeAll(newIdx);
  for(int i=0; i<esize; i++) {
    m->n2e_add(newIdx, enbrs[i]);
    m->e2n_replace(enbrs[i], oldIdx, newIdx, 0);
  }

  //get rid of all connections of the older node
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  m->n2n_removeAll(oldIdx);
  m->n2e_removeAll(oldIdx);

  if(nsize>0) delete[] nnbrs;
  if(esize>0) delete[] enbrs;
  return newIdx;  // return a new index
}

void FEM_MUtil::addToSharedList(FEM_Mesh *m, int fromChk, int sharedIdx) {
  int elemType = 0;
  int connSize = m->elem[elemType].getConn().width();
  int localIdx = mmod->fmUtil->lookup_in_IDXL(m,sharedIdx,fromChk,1); //look in the ghostsend list

  //fix the idxl lists
  m->node.shared.addNode(localIdx, fromChk);
  m->node.ghostSend.removeNode(localIdx, fromChk);

  //find all chunks that receive this node as a ghost
  const IDXL_Rec *irec2 = m->node.ghostSend.getRec(localIdx);
  if(irec2!=NULL) {
    for(int i=0; i<irec2->getShared(); i++) {
      //tell that chunk to verify if it needs to add this node as a ghost
      int pchk = irec2->getChk(i);
      int shidx = irec2->getIdx(i);
      meshMod[pchk].addghostsendr(idx,shidx,fromChk,exists_in_IDXL(m,localIdx,fromChk,0));
    }
  }

  int *enbrs;
  int esize;
  m->n2e_getAll(localIdx, &enbrs, &esize);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  for(int i=0; i<esize; i++) {
    if(enbrs[i] >= 0) { //if it is a local element
      //if it exists in the ghostsend list already
      int exists = mmod->fmUtil->exists_in_IDXL(m, enbrs[i], fromChk, 3);
      if(exists == -1) {
	//idxllock(m, fromChk, 1);
	//idxllock(m, fromChk, 3);
	m->elem[elemType].ghostSend.addNode(enbrs[i], fromChk);
	//ghost nodes should be added only if they were not already present as ghosts on that chunk.
	int numNodesToAdd = 0;
	int numSharedGhosts = 0;
	int numSharedNodes = 0;
	int *sharedGhosts = (int *)malloc(connSize*sizeof(int));
	int *sharedNodes = (int *)malloc(connSize*sizeof(int));
	int *nnbrs = (int*)malloc(connSize*sizeof(int));
	int *nodesToAdd = (int*)malloc(connSize*sizeof(int));
	m->e2n_getAll(enbrs[i], nnbrs, elemType);
	for(int j=0; j<connSize; j++) {
#ifdef DEBUG 
	  CmiMemoryCheck(); 
#endif
	  int sharedNode = mmod->fmUtil->exists_in_IDXL(m,nnbrs[j],fromChk,0);
	  if(sharedNode == -1) {
	    //node 'j' is a ghost on chunk 'i'
	    int sharedGhost = mmod->fmUtil->exists_in_IDXL(m,nnbrs[j],fromChk,1);
	    if( sharedGhost == -1) {
	      //this might be a new ghost, figure out if any of the chunks sharing
	      //this node has created this as a ghost on 'fromChk'
	      const IDXL_Rec *irec = m->node.shared.getRec(nnbrs[j]);
	      if(irec) {
		int noShared = irec->getShared();
		for(int sharedck=0; sharedck<noShared; sharedck++) {
		  int ckshared = irec->getChk(sharedck);
		  int idxshared = irec->getIdx(sharedck);
		  if(ckshared == fromChk) continue;
		  CkAssert(fromChk!=idx && fromChk!=ckshared && ckshared!=idx);
		  int idxghostsend = meshMod[ckshared].getIdxGhostSend(idx,idxshared,fromChk)->i;
		  if(idxghostsend != -1) {
		    m->node.ghostSend.addNode(nnbrs[j],fromChk);
		    meshMod[fromChk].updateIdxlList(idx,idxghostsend,ckshared);
		    sharedGhost = exists_in_IDXL(m,nnbrs[j],fromChk,1);
		    CkAssert(sharedGhost != -1);
		    break; //found a chunk that sends it out, update my tables
		  }
		  //Chunk 'ckshared' does not send this to Chunk 'fromChk' as ghost
		}
	      }
	      //else it is a new ghost
	    }
	    if( sharedGhost == -1) {
	      //it is a new ghost
	      nodesToAdd[numNodesToAdd] = nnbrs[j];
	      numNodesToAdd++;
	    }
	    else {
	      //it is a shared ghost
	      sharedGhosts[numSharedGhosts] = sharedGhost;
	      numSharedGhosts++;
	    }
	  }
	  else {
	    //it is a shared node
	    sharedNodes[numSharedNodes] = sharedNode;
	    numSharedNodes++;
	  }
	}
	//add this element as a ghost on fromChk
	addGhostElemMsg *fm = new (numSharedGhosts, numSharedNodes, 0)addGhostElemMsg;
	fm->chk = getIdx();
	fm->elemType = elemType;
	fm->numGhostIndex = numSharedGhosts;
	for(int j=0; j<numSharedGhosts; j++) {
	  fm->ghostIndices[j] = sharedGhosts[j];
	}
	fm->numSharedIndex = numSharedNodes;
	for(int j=0; j<numSharedNodes; j++) {
	  fm->sharedIndices[j] = sharedNodes[j];
	}
	for(int j=0; j<numNodesToAdd; j++) {
	  m->node.ghostSend.addNode(nodesToAdd[j],fromChk);
	}
	fm->connSize = connSize;
	meshMod[fromChk].addGhostElem(fm); 

	for(int j=0; j<numNodesToAdd; j++) {
	  //make the chunks which share this node also add this node as a ghost node
	  //if it is not already sending it
	  const IDXL_Rec *irec1 = m->node.shared.getRec(nodesToAdd[j]);
	  if(irec1!=NULL) {
	    for(int sh=0; sh<irec1->getShared(); sh++) {
	      int transIdx = exists_in_IDXL(m,nodesToAdd[j],fromChk,1);
	      meshMod[irec1->getChk(sh)].addghostsendl(idx,irec1->getIdx(sh),fromChk,transIdx);
	    }
	  }
	}

	//idxlunlock(m, fromChk, 1);
	//idxlunlock(m, fromChk, 3);
	free(nodesToAdd);
	free(sharedGhosts);
	free(sharedNodes);
	free(nnbrs);
      }
    }
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  delete[] enbrs;
  return;
}

//find the set of chunks I need to send this node as a ghost to
void FEM_MUtil::findGhostSend(int nodeId, int **chunkl, int *numchunkl) {
  if(nodeId<0) return;
  CkVec<int> chkl;
  int *adjnds, numadjnds=0;
  mmod->fmMesh->n2n_getAll(nodeId, &adjnds, &numadjnds);
  for(int j=0; j<numadjnds; j++) {
    int node1 = adjnds[j];
    if(node1>=0) {
      const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(node1);
      if(irec) {
	for(int k=0; k<irec->getShared(); k++) {
	  int pchk = irec->getChk(k);
	  //add this to ckl, if it does not exist already
	  //if nodeId is shared on pchk, continue
	  if(exists_in_IDXL(mmod->fmMesh,nodeId,pchk,0)!=-1) continue;
	  bool shouldadd=true;
	  for(int l=0; l<chkl.size(); l++) {
	    if(chkl[l]==pchk) {
	      shouldadd=false;
	      break;
	    }
	  }
	  if(shouldadd) chkl.push_back(pchk);
	}
      }
    }
  }
  if(numadjnds>0) delete[] adjnds;
  *numchunkl = chkl.size();
  if(numchunkl>0) 
    *chunkl = new int[*numchunkl];
  for(int i=0;i<*numchunkl;i++) {
    (*chunkl)[i] = chkl[i];
  }
  //delete the vector
  chkl.free();
  return;
}

//get rid of unnecessary ghostsends..
void FEM_MUtil::UpdateGhostSend(int nodeId, int *chunkl, int numchunkl) {
  if(nodeId<0) return;
  const IDXL_Rec *irec = mmod->fmMesh->node.ghostSend.getRec(nodeId);
  int numchunks=0;
  int *chunks1, *inds1;
  if(irec) {
    numchunks = irec->getShared();
    chunks1 = new int[numchunks];
    inds1 = new int[numchunks];
    for(int i=0; i<numchunks; i++) {
      chunks1[i] = irec->getChk(i);
      inds1[i] = irec->getIdx(i);
    }
  }
  for(int i=0; i<numchunks; i++) {
    bool shouldbeSent = false;
    for(int j=0; j<numchunkl; j++) { //if it is not in this list
      if(chunks1[i]==chunkl[j]) {
	shouldbeSent=true;
	break;
      }
    }
    if(!shouldbeSent) {
      meshMod[chunks1[i]].removeGhostNode(idx, inds1[i]);
      mmod->fmMesh->node.ghostSend.removeNode(nodeId,chunks1[i]);
    }
  }

  if(numchunks>0) {
    delete[] chunks1;
    delete[] inds1;
  }
  return;
}

//should only be called on a ghost element
int FEM_MUtil::eatIntoElement(int localIdx) {
  CkAssert(FEM_Is_ghost_index(localIdx));
  int nodesPerEl = mmod->fmMesh->elem[0].getConn().width();
  int *adjnodes = new int[nodesPerEl];
  int *oldnodes = new int[nodesPerEl];
  mmod->fmMesh->e2n_getAll(localIdx, adjnodes, 0);
#ifdef DEBUG_1
  CkPrintf("Chunk %d eating elem %d(%d,%d,%d)\n",idx,localIdx,adjnodes[0],adjnodes[1],adjnodes[2]);
#endif
  int remChk = mmod->fmMesh->elem[0].ghost->ghostRecv.getRec(FEM_From_ghost_index(localIdx))->getChk(0);
  //a ghost elem should be coming from only one chunk
  for(int i=0; i<nodesPerEl; i++) {
    if(FEM_Is_ghost_index(adjnodes[i])) { 
      //this will be a new node on this chunk, lock it on all shared chunks
      int sharedIdx = exists_in_IDXL(mmod->fmMesh,adjnodes[i],remChk,2);
      meshMod[remChk].modifyLockAll(idx,sharedIdx);
    }
  }
  FEM_remove_element(mmod->fmMesh,localIdx,0,idx);
  for(int j=0; j<nodesPerEl; j++) oldnodes[j] = adjnodes[j];
  int newEl = FEM_add_element(mmod->fmMesh, adjnodes, nodesPerEl, 0, idx);
  copyElemData(0,localIdx,newEl); //special copy across chunk
  FEM_purge_element(mmod->fmMesh,localIdx,0);
  for(int i=0; i<nodesPerEl; i++) {
    if(adjnodes[i]!=oldnodes[i]) {
      //correct the lock
      FEM_Modify_LockUpdate(mmod->fmMesh,adjnodes[i]);
    }
  }
  delete [] adjnodes;
  delete [] oldnodes;
  return newEl;
}

int FEM_MUtil::getLockOwner(int nodeId) {
  int owner = -1;
  if(nodeId>=0) {//it is local/shared
    const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(nodeId);
    //find the minchunk
    int minchunk = MAX_CHUNK;
    if(irec) {
      for(int i=0; i<irec->getShared(); i++) {
	int pchk = irec->getChk(i);
	if(pchk<minchunk) minchunk = pchk;
      }
    }
    else minchunk = idx;
    if(minchunk == idx) owner = mmod->fmMesh->getfmMM()->getfmLockN(nodeId)->lockOwner();
    else {
      CkAssert(minchunk!=MAX_CHUNK);
      int sharedIdx = mmod->getfmUtil()->exists_in_IDXL(mmod->fmMesh,nodeId,minchunk,0);
      owner = meshMod[minchunk].hasLockRemoteNode(sharedIdx, idx, 0)->i;
    }
  }
  else {
    int otherchk = mmod->fmMesh->node.ghost->ghostRecv.getRec(FEM_From_ghost_index(nodeId))->getChk(0);
    int sharedIdx = mmod->fmMesh->node.ghost->ghostRecv.getRec(FEM_From_ghost_index(nodeId))->getIdx(0);
    owner = meshMod[otherchk].getLockOwner(idx, sharedIdx)->i;
  }
  //CkAssert(owner != -1);
  return owner;
}

bool FEM_MUtil::knowsAbtNode(int chk, int nodeId) {
  if(nodeId >= 0) {
    const IDXL_Rec *irec = mmod->fmMesh->node.ghostSend.getRec(nodeId);
    if(irec) {
      for(int i=0; i<irec->getShared(); i++) {
	if(irec->getChk(i) == chk) return true;
      }
    }
    irec = mmod->fmMesh->node.shared.getRec(nodeId);
    if(irec) {
      for(int i=0; i<irec->getShared(); i++) {
	if(irec->getChk(i) == chk) return true;
      }
    }
  }
  else { // it is a ghost on this chunk
    //find any chunk that owns it
    int owner = mmod->fmMesh->node.ghost->ghostRecv.getRec(FEM_From_ghost_index(nodeId))->getChk(0);
    int sharedIdx = mmod->fmMesh->node.ghost->ghostRecv.getRec(FEM_From_ghost_index(nodeId))->getIdx(0);
    //does 'chk' know abt 'nodeId' in owner
    return meshMod[owner].knowsAbtNode(idx, chk, sharedIdx)->b;
  }
  return false;
}

void FEM_MUtil::StructureTest(FEM_Mesh *m) {
  int noNodes = m->node.size();
  int noEle = m->elem[0].size();
  int noGhostEle = m->elem[0].ghost->size();
  int noGhostNodes = m->node.ghost->size();

  int wdt = m->elem[0].getConn().width();
  int *e2n = (int*)malloc(wdt*sizeof(int));

  for(int i=0; i<noEle; i++) {
    if(m->elem[0].is_valid(i)) {
      m->e2n_getAll(i,e2n,0);
      //must have all different connections
      if(e2n[0]==e2n[1] || e2n[1]==e2n[2] || e2n[2]==e2n[0]) {
	CkPrintf("ERROR: element %d, has connectivity (%d,%d,%d)\n",i,e2n[0],e2n[1],e2n[2]);
	CkAssert(false);
      }
      //local elem must have all local node connectivity 
      if(e2n[0]<0 || e2n[1]<0 || e2n[2]<0) {
	CkPrintf("ERROR: element %d, has connectivity (%d,%d,%d)\n",i,e2n[0],e2n[1],e2n[2]);
	CkAssert(false);
      }
      for(int j=0; j<3; j++) {
	//all nodes must be valid
	CkAssert(m->node.is_valid(e2n[j]));
      }
    }
  }
  for(int i=0; i<noGhostEle; i++) {
    if(m->elem[0].ghost->is_valid(i)) {
      int ghostIndex = FEM_To_ghost_index(i);
      m->e2n_getAll(ghostIndex,e2n,0);
      if(e2n[0]==e2n[1] || e2n[1]==e2n[2] || e2n[2]==e2n[0]) {
	//must have all different connections
	CkPrintf("ERROR: element %d, has connectivity (%d,%d,%d)\n",ghostIndex,e2n[0],e2n[1],e2n[2]);
	CkAssert(false);
      }
      if(!(e2n[0]>=0 || e2n[1]>=0 || e2n[2]>=0)) {
	//must have at least one local node
	CkPrintf("ERROR: element %d, has connectivity (%d,%d,%d)\n",ghostIndex,e2n[0],e2n[1],e2n[2]);
	CkAssert(false);
      }
      for(int j=0; j<3; j++) {
	//all nodes must be valid
	if(e2n[j]>=0) CkAssert(m->node.is_valid(e2n[j]));
	else CkAssert(m->node.ghost->is_valid(FEM_From_ghost_index(e2n[j])));
      }
    }
  }

  int *n2e, n2esize=0;
  int *n2n, n2nsize=0;

  for(int i=0; i<noNodes; i++) {
    if(m->node.is_valid(i)) {
      m->n2e_getAll(i,&n2e,&n2esize);
      m->n2n_getAll(i,&n2n,&n2nsize);
      if(n2esize!=n2nsize && n2nsize!=(n2esize+1)) {
	FEM_Print_coords(m,i);
	FEM_Print_n2e(m,i);
	FEM_Print_n2n(m,i);
	CkPrintf("ERROR: local node %d, with inconsistent adjacency list\n",i);
	CkAssert(false);
      }
    }
    else {
      continue;
    }
    if(n2esize > 0) {
      for(int j=0; j<n2esize; j++) {
	CkAssert(n2e[j]!=-1);
	if(FEM_Is_ghost_index(n2e[j])) CkAssert(m->elem[0].ghost->is_valid(FEM_From_ghost_index(n2e[j]))==1);
	else CkAssert(m->elem[0].is_valid(n2e[j])==1);
      }

      m->e2n_getAll(n2e[0],e2n,0);

      //any local/shared node should have at least one local element connected to it
      bool done = false;
      for(int j=0; j<n2esize; j++) {
	if(n2e[j] >= 0) {
	  done = true; 
	  break;
	}
      }
      if(!done) {
	FEM_Print_coords(m,i);
	FEM_Print_n2e(m,i);
	FEM_Print_n2n(m,i);
	CkPrintf("ERROR: isolated local node %d, with no local element connectivity\n",i);
	CkAssert(false);
      }

      //ensure that there is a cloud of connectivity, no disconnected elements, other than boundaries
      int testnode = i;
      int startnode = (e2n[0]==testnode) ? e2n[1] : e2n[0];
      int othernode = (e2n[2]==testnode) ? e2n[1] : e2n[2];
      int previousnode = startnode;
      int nextnode = -1;
      int numdeadends = 0;
      int numunused = n2esize-1;
      n2e[0] = -1;
      for(int j=0; j<n2esize-1; j++) {
	nextnode = -1;
	for(int k=1; k<n2esize; k++) {
	  if(n2e[k]==-1) continue;
	  m->e2n_getAll(n2e[k],e2n,0);
	  if(e2n[0]==previousnode || e2n[1]==previousnode || e2n[2]==previousnode) {
	    nextnode = (e2n[0]==previousnode) ? ((e2n[1]==testnode)? e2n[2]:e2n[1]) : ((e2n[1]==previousnode)? ((e2n[0]==testnode)? e2n[2]:e2n[0]):((e2n[1]==testnode)? e2n[0]:e2n[1]));
	    previousnode = nextnode;
	    n2e[k] = -1;
	    numunused--;
	  }
	}
	if(nextnode==othernode && othernode!=-1) {
	  //it has reached a full circle
	  break;
	}
	else if(nextnode==-1) {
	  //this is one edge, start travelling along the other end
	  numdeadends++;
	  previousnode = othernode;
	  othernode = -1;
	}
	if(numdeadends>=2 && numunused!=0) {
	  FEM_Print_coords(m,i);
	  FEM_Print_n2e(m,i);
	  FEM_Print_n2n(m,i);
	  CkPrintf("ERROR: cloud connectivity of node %d is discontinuous\n",i);
	  CkAssert(false);
	}
      }
      //reconstruct n2n from n2e & e2n
      int n2n1size = 0;
      int *n2n1 = (int*)malloc(n2nsize*sizeof(int));
      int n2n1Count = 0;
      m->n2e_getAll(i,&n2e,&n2esize);
      for(int j=0; j<n2esize; j++) {
	CkAssert(n2e[j]!=-1);
	//each of these elems should have me in its e2n
	int e2n1[3];
	m->e2n_getAll(n2e[j],e2n1,0);
	if(e2n1[0]!=i && e2n1[1]!=i && e2n1[2]!=i) {
	  FEM_Print_coords(m,i);
	  FEM_Print_n2e(m,i);
	  FEM_Print_e2n(m,n2e[j]);
	  CkPrintf("ERROR: ghost elem %d & ghost node %d have inconsistent adjacency list\n",n2e[j],i);
	  CkAssert(false);
	}
	for(int k=0; k<3;k++) {
	  if(e2n1[k] == i) continue;
	  bool flag1 = true;
	  for(int l=0; l<n2n1Count; l++) {
	    if(e2n1[k] == n2n1[l]) flag1 = false;
	  }
	  if(flag1 && n2n1Count<n2nsize) { //this is not in the list
	    n2n1[n2n1Count] = e2n1[k];
	    n2n1Count++;
	  }
	}
      }
      
      //verify if n2n1 has the same nodes as n2n
      bool flag2 = true;
      if(n2n1Count!=n2nsize) flag2 = false;
      for(int j=0; j<n2n1Count; j++) {
	bool flag1 = false;
	for(int k=0; k<n2nsize; k++) {
	  if(n2n[k]==n2n1[j]) flag1 = true;
	}
	if(!flag1) {
	  flag2 = false;
	  break;
	}
      }
      if(!flag2) {
	FEM_Print_coords(m,i);
	FEM_Print_n2n(m,i);
	for(int l=0; l<n2esize; l++) FEM_Print_e2n(m,n2e[l]);
	CkPrintf("ERROR: ghost node %d has inconsistent adjacency list\n",i);
	CkAssert(false);
      }
      
      free(n2n1);
      delete [] n2e;
      if(n2nsize>0) delete [] n2n;
    }
  }
  for(int i=0; i<noGhostNodes; i++) {
    int ghostidx = FEM_To_ghost_index(i);
    if(m->node.ghost->is_valid(i)) {
      m->n2e_getAll(ghostidx,&n2e,&n2esize);
      m->n2n_getAll(ghostidx,&n2n,&n2nsize);
      bool done = false;
      for(int k=0;k<n2nsize;k++) {
	if(n2n[k]>=0) {
	  done = true;
	  break;
	}
      }
      if(n2esize>n2nsize || !done) {
	FEM_Print_coords(m,ghostidx);
	FEM_Print_n2e(m,ghostidx);
	FEM_Print_n2n(m,ghostidx);
	CkPrintf("ERROR: ghost node %d, with inconsistent adjacency list\n",ghostidx);
	CkAssert(false);
      }
      if(n2esize > 0) {
	//reconstruct n2n from n2e & e2n
	int n2n1size = 0;
	int *n2n1 = (int*)malloc(n2nsize*sizeof(int));
	int n2n1Count = 0;
	for(int j=0; j<n2esize; j++) {
	  CkAssert(n2e[j]!=-1);
	  if(FEM_Is_ghost_index(n2e[j])) CkAssert(m->elem[0].ghost->is_valid(FEM_From_ghost_index(n2e[j]))==1);
	  else CkAssert(m->elem[0].is_valid(n2e[j])==1);
	  //each of these elems should have me in its e2n
	  int e2n1[3];
	  m->e2n_getAll(n2e[j],e2n1,0);
	  if(e2n1[0]!=ghostidx && e2n1[1]!=ghostidx && e2n1[2]!=ghostidx) {
	    FEM_Print_coords(m,ghostidx);
	    FEM_Print_n2e(m,ghostidx);
	    FEM_Print_e2n(m,n2e[j]);
	    CkPrintf("ERROR: ghost elem %d & ghost node %d have inconsistent adjacency list\n",n2e[j],ghostidx);
	    CkAssert(false);
	  }
	  for(int k=0; k<3;k++) {
	    if(e2n1[k] == ghostidx) continue;
	    bool flag1 = true;
	    for(int l=0; l<n2n1Count; l++) {
	      if(e2n1[k] == n2n1[l]) flag1 = false;
	    }
	    if(flag1 && n2n1Count<n2nsize) { //this is not in the list
	      n2n1[n2n1Count] = e2n1[k];
	      n2n1Count++;
	    }
	  }
	}

	//verify if n2n1 has the same nodes as n2n
	bool flag2 = true;
	if(n2n1Count!=n2nsize) flag2 = false;
	for(int j=0; j<n2n1Count; j++) {
	  bool flag1 = false;
	  for(int k=0; k<n2nsize; k++) {
	    if(n2n[k]==n2n1[j]) flag1 = true;
	  }
	  if(!flag1) {
	    flag2 = false;
	    break;
	  }
	}
	if(!flag2) {
	  FEM_Print_coords(m,ghostidx);
	  FEM_Print_n2n(m,ghostidx);
	  for(int l=0; l<n2esize; l++) FEM_Print_e2n(m,n2e[l]);
	  CkPrintf("ERROR: ghost node %d has inconsistent adjacency list\n",ghostidx);
	  CkAssert(false);
	}
	free(n2n1);
	delete [] n2e;
      }
      if(n2nsize > 0) {
	for(int j=0; j<n2nsize; j++) {
	  CkAssert(n2n[j]!=-1);
	}
	delete [] n2n;
      }
      //verify that it is coming correctly from all chunks as a ghost
      const IDXL_Rec *irec = m->node.ghost->ghostRecv.getRec(i);
      //go to a chunk which owns it & verify that these are the only chunks that own it
      int remChk = irec->getChk(0);
      int sharedIdx = irec->getIdx(0);
      int numsh = irec->getShared();
      verifyghostsendMsg *vmsg = new(numsh,0)verifyghostsendMsg();
      vmsg->fromChk = idx;
      vmsg->sharedIdx = sharedIdx;
      vmsg->numchks = numsh;
      for(int k=0; k<numsh; k++) vmsg->chunks[k] = irec->getChk(k);
      meshMod[remChk].verifyghostsend(vmsg);
    }
    else {
      continue;
    }
  }

  free(e2n);
  return;
}

int FEM_MUtil::AreaTest(FEM_Mesh *m) {
  int noEle = m->elem[0].size();
  int wdt = m->elem[0].getConn().width();
  int *con = (int*)malloc(wdt*sizeof(int));

  for(int i=0; i<noEle; i++) {
    if(m->elem[0].is_valid(i)) {
      m->e2n_getAll(i,con,0);
      double area = mmod->fmAdaptAlgs->getArea(con[0],con[1],con[2]);
      if(fabs(area) < SLIVERAREA) {
	CkAssert(false);
	free(con);
	return -1;
      }
    }
  }
  free(con);
  return 1;
}

int FEM_MUtil::IdxlListTest(FEM_Mesh *m) {
  for(int type=0; type <5; type++) {
    int listsize = 0;
    if(type==0) listsize = m->node.shared.size();
    else if(type==1) listsize = m->node.ghostSend.size();
    else if(type==2) listsize = m->node.ghost->ghostRecv.size();
    else if(type==3) listsize = m->elem[0].ghostSend.size();
    else if(type==4) listsize = m->elem[0].ghost->ghostRecv.size();
    
    for(int i=0; i<listsize; i++) {
      IDXL_List il;
      if(type==0) il = m->node.shared.getLocalList(i);
      else if(type==1) il = m->node.ghostSend.getLocalList(i);
      else if(type==2) il = m->node.ghost->ghostRecv.getLocalList(i);
      else if(type==3) il = m->elem[0].ghostSend.getLocalList(i);
      else if(type==4) il = m->elem[0].ghost->ghostRecv.getLocalList(i);
      int shck = il.getDest();
      int size = il.size();
      //verify that chunk 'shck' also has an idxl list with me of 'size'
      meshMod[shck].verifyIdxlList(idx,size,type);
      //verify that all entries are positive
      for(int j=0; j<size; j++) {
	CkAssert(il[j] >= -1);
      }
    }
  }
  
  return 1;
}

int FEM_MUtil::residualLockTest(FEM_Mesh *m) {
  int noNodes = m->node.size();
  for(int i=0; i<noNodes; i++) {
    if(m->node.is_valid(i)) {
      CkAssert(!mmod->fmLockN[i]->haslocks());
    }
  }
  for(int i=0; i<mmod->numChunks*5; i++) {
    CkAssert(mmod->fmIdxlLock[i]==false);
  }
  return 1;
}

void FEM_MUtil::verifyIdxlListRemote(FEM_Mesh *m, int fromChk, int fsize, int type) {
  IDXL_Side is;

  IDXL_List il;
  if(type==0) il = m->node.shared.addList(fromChk);
  else if(type==1) il = m->node.ghost->ghostRecv.addList(fromChk);
  else if(type==2) il = m->node.ghostSend.addList(fromChk);
  else if(type==3) il = m->elem[0].ghost->ghostRecv.addList(fromChk);
  else if(type==4) il = m->elem[0].ghostSend.addList(fromChk);
  int size = il.size();
  CkAssert(fsize == size);
  //verify that all entries are positive
  for(int j=0; j<size; j++) {
    CkAssert(il[j] >= -1);
  }
  
  return;
}

void FEM_MUtil::FEM_Print_n2n(FEM_Mesh *m, int nodeid){
  CkPrintf("node %d is adjacent to nodes:", nodeid);
  int *adjnodes;
  int sz;
  m->n2n_getAll(nodeid, &adjnodes, &sz); 
  for(int i=0;i<sz;i++)
    CkPrintf(" %d", adjnodes[i]);
  if(sz!=0) delete[] adjnodes;  
  CkPrintf("\n");
}

void FEM_MUtil::FEM_Print_n2e(FEM_Mesh *m, int eid){
  CkPrintf("node %d is adjacent to elements:", eid);
  int *adjes;
  int sz;
  m->n2e_getAll(eid, &adjes, &sz);
  for(int i=0;i<sz;i++)
    CkPrintf(" %d", adjes[i]);
  if(sz!=0) delete[] adjes;
  CkPrintf("\n");
}

void FEM_MUtil::FEM_Print_e2n(FEM_Mesh *m, int eid){
  CkPrintf("element %d is adjacent to nodes:", eid);
  int consz = m->elem[0].getConn().width();
  int *adjns = new int[consz];
  m->e2n_getAll(eid, adjns, 0); 
  for(int i=0;i<consz;i++)
    CkPrintf(" %d", adjns[i]);
  CkPrintf("\n");
  delete [] adjns;
}

void FEM_MUtil::FEM_Print_e2e(FEM_Mesh *m, int eid){
  CkPrintf("element %d is adjacent to elements:", eid);
  int consz = m->elem[0].getConn().width();
  int *adjes = new int[consz];
  m->e2e_getAll(eid, adjes, 0); 
  for(int i=0;i<consz;i++)
    CkPrintf(" %d", adjes[i]);
  CkPrintf("\n");
  delete [] adjes;
}

void FEM_MUtil::FEM_Print_coords(FEM_Mesh *m, int nodeid) {
  double crds[2];
  int bound;
  if(!FEM_Is_ghost_index(nodeid)) {
    FEM_Mesh_dataP(m, FEM_NODE, mmod->fmAdaptAlgs->coord_attr, (void *)crds, nodeid, 1, FEM_DOUBLE, 2);
    FEM_Mesh_dataP(m, FEM_NODE, FEM_BOUNDARY, (void *)&bound, nodeid, 1, FEM_INT, 1);
  }
  else {
    int numchunks;
    IDXL_Share **chunks1;
    getChunkNos(0,nodeid,&numchunks,&chunks1);
    int index = mmod->idx;
    for(int j=0; j<numchunks; j++) {
      int chk = chunks1[j]->chk;
      if(chk==index) continue;
      int ghostidx = exists_in_IDXL(m,nodeid,chk,2);
      double2Msg *d = meshMod[chk].getRemoteCoord(index,ghostidx);
      intMsg *im = meshMod[chk].getRemoteBound(index,ghostidx);
      crds[0] = d->i;
      crds[1] = d->j;
      bound = im->i;
      for(int j=0; j<numchunks; j++) {
	delete chunks1[j];
      }
      if(numchunks != 0) free(chunks1);
      break;
    }
  }
  CkPrintf("node %d (%f,%f) and boundary %d\n",nodeid,crds[0],crds[1],bound);
}

void FEM_MUtil::idxllock(FEM_Mesh *m, int chk, int type) {
#ifdef DEBUG 
CmiMemoryCheck();
#endif
  if(idx < chk) {
    idxllockLocal(m,chk,type);
  } else {
    meshMod[chk].idxllockRemote(idx,type);
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void FEM_MUtil::idxlunlock(FEM_Mesh *m, int chk, int type) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(idx < chk) {
    //meshMod[chk].idxlunlockRemote(idx,type);
    idxlunlockLocal(m,chk,type);
  } else {
    //idxlunlockLocal(m,chk,type);
    meshMod[chk].idxlunlockRemote(idx,type);
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void FEM_MUtil::idxllockLocal(FEM_Mesh *m, int toChk, int type) {
  CkAssert(toChk>=0 && toChk<mmod->numChunks && toChk!=idx && type>=0 && type<5);
  while(mmod->fmIdxlLock[toChk*5 + type] == true) {
    //block by looping,
    CthYield();
  }
  //CkPrintf("%d locking idxl list %d: type %d\n",idx,toChk,type);
  mmod->fmIdxlLock[toChk*5 + type] = true;
#ifdef DEBUG_IDXLLock
  CkPrintf("[%d]locked idxl lock %d\n",idx,toChk*5+type);
#endif
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void FEM_MUtil::idxlunlockLocal(FEM_Mesh *m, int toChk, int type) {
  CkAssert(toChk>=0 && toChk<mmod->numChunks && toChk!=idx && type>=0 && type<5);
  CkAssert(mmod->fmIdxlLock[toChk*5 + type] == true);
  //CkPrintf("%d unlocking idxl list %d: type %d\n",idx,toChk,type);
  mmod->fmIdxlLock[toChk*5 + type] = false;
#ifdef DEBUG_IDXLLock
  CkPrintf("[%d]unlocked idxl lock %d\n",idx,toChk*5+type);
#endif
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

//copies the elem data from elemid to newEl
void FEM_MUtil::copyElemData(int etype, int elemid, int newEl) {
  FEM_Interpolate *inp = mmod->getfmInp();
  FEM_Interpolate::ElementArgs em;
  em.e = newEl;
  em.oldElement = elemid;
  em.elType = etype;
  inp->FEM_InterpolateElementCopy(em);
}

