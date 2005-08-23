#include "fem_util.h"
#include "fem_mesh_modify.h"


extern void splitEntity(IDXL_Side &c, int localIdx, int nBetween, int *between, int idxbase);

FEM_MUtil::FEM_MUtil(int i, femMeshModify *m) {
  idx = i;
  mmod = m;
}

FEM_MUtil::~FEM_MUtil() {
}

void FEM_MUtil::getChunkNos(int entType, int entNo, int *numChunks, IDXL_Share ***chunks, int elemType) {
  int type = 0; //0 - local, 1 - shared, 2 - ghost.

  if(entType == 0) { //nodes
    //only nodes can be shared
    if(FEM_Is_ghost_index(entNo)) type = 2;
    else if(isShared(entNo)) type = 1;
    else type = 0;

    if(type == 2) {
      int ghostid = FEM_To_ghost_index(entNo);
      int noShared = 0;  //I think a ghost node in the receiver table is entered as only coming from the one chunk that was first to add it.
      const IDXL_Rec *irec = mmod->fmMesh->node.ghost->ghostRecv.getRec(ghostid);
      if(irec) {
	noShared = irec->getShared(); //check this value!!
	//CkAssert(noShared > 0);
	int chunk = irec->getChk(0);
	int sharedIdx = exists_in_IDXL(mmod->fmMesh,ghostid,chunk,2);
	int2Msg *i2 = new int2Msg(idx, sharedIdx);
	chunkListMsg *clm = meshMod[chunk].getChunksSharingGhostNode(i2);
	*numChunks = clm->numChunkList + 1; //add chunk to the list
	*chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
	int i=0;
	for(i=0; i<*numChunks - 1; i++) {
	  int chk = clm->chunkList[i];
	  int index = -1; // no need to have these, I never use it anyway
	  (*chunks)[i] = new IDXL_Share(chk, index);
	}
	(*chunks)[i] = new IDXL_Share(chunk, -1);
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
  return;
}

bool FEM_MUtil::isShared(int index) {
  //this function will be only called for a shared list
  //have to figure out if node.shared is kept up to date
  const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(index);
  //if an entry exists in the shared idxl lists, then it is a shared node
  if(irec != NULL) return true;
  return false;
}

//An IDXL helper function which is same as splitEntity, but instead of just adding to this chunk's
//idxl list, it will add to the idxl lists of all chunks.
void FEM_MUtil::splitEntityAll(FEM_Mesh *m, int localIdx, int nBetween, int *between, int idxbase)
{
  //Find the commRecs for the surrounding nodes
  IDXL_Side *c = &(m->node.shared);
  const IDXL_Rec **tween = (const IDXL_Rec **)malloc(nBetween*sizeof(IDXL_Rec *));
  
  //Make a new commRec as the interesection of the surrounding entities--
  // we loop over the first entity's comm. list
  tween[0] = c->getRec(between[0] - idxbase);
  for (int zs=tween[0]->getShared()-1; zs>=0; zs--) {
    for (int w1=0; w1<nBetween; w1++) {
      tween[w1] = c->getRec(between[w1]-idxbase);
    }
    int chk = tween[0]->getChk(zs);

    //Make sure this processor shares all our entities
    int w = 0;
    for (w=0; w<nBetween; w++) {
      if (!tween[w]->hasChk(chk)) {
	break;
      }
    }

    if (w == nBetween) {//The new node is shared with chk
      c->addNode(localIdx,chk); //add in the shared entry of this chunk

      //generate the shared node numbers with chk from the local indices
      int *sharedIndices = (int *)malloc(nBetween*sizeof(int));
      const IDXL_List ll = m->node.shared.getList(chk);
      for(int w1=0; w1<nBetween; w1++) {
	for(int w2=0; w2<ll.size(); w2++) {
	  if(ll[w2] == between[w1]) {
	    sharedIndices[w1] = w2;
	    break;
	  }
	}
      }
      sharedNodeMsg *fm = new (2, 0) sharedNodeMsg;
      fm->chk = mmod->idx;
      fm->nBetween = nBetween;
      for(int j=0; j<nBetween; j++) {
	fm->between[j] = sharedIndices[j];
      }
      meshMod[chk].addSharedNodeRemote(fm);
      free(sharedIndices);
      //break;
    }
  }
  free(tween);
  return;
}

void FEM_MUtil::splitEntityRemote(FEM_Mesh *m, int chk, int localIdx, int nBetween, int *between, int idxbase)
{
  //convert the shared indices to local indices
  int *localIndices = (int *)malloc(nBetween*sizeof(int));
  const IDXL_List ll = m->node.shared.getList(chk);
  for(int i=0; i<nBetween; i++) {
    localIndices[i] = ll[between[i]];
  }

  FEM_Interpolate *inp = m->getfmMM()->getfmInp();
  FEM_Interpolate::NodalArgs nm;
  nm.n = localIdx;
  for(int i=0; i<nBetween; i++) {
    nm.nodes[i] = localIndices[i];
  }
  nm.frac = 0.5;
  inp->FEM_InterpolateNodeOnEdge(nm);

  splitEntity(m->node.shared, localIdx, nBetween, localIndices, idxbase);
  free(localIndices);
  return;
}

void FEM_MUtil::removeNodeAll(FEM_Mesh *m, int localIdx)
{
  IDXL_Side *c = &(m->node.shared);
  const IDXL_Rec *tween = c->getRec(localIdx);
  for(int i=0; i<tween->getShared(); i++) {
    removeSharedNodeMsg *fm = new removeSharedNodeMsg;
    fm->chk = mmod->idx;
    fm->index = tween->getIdx(i);
    meshMod[tween->getChk(i)].removeSharedNodeRemote(fm);
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
    ll = m->node.shared.getList(chk);
  }
  else if(type == 1) { //ghost node send 
    ll = m->node.ghostSend.getList(chk);
  }
  else if(type == 2) { //ghost node recv 
    ll = m->node.ghost->ghostRecv.getList(chk);
    localIdx = FEM_To_ghost_index(localIdx);
  }
  else if(type == 3) { //ghost elem send 
    ll = m->elem[elemType].ghostSend.getList(chk);
  }
  else if(type == 4) { //ghost elem recv 
    ll = m->elem[elemType].ghost->ghostRecv.getList(chk);
    localIdx = FEM_To_ghost_index(localIdx);
  }
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
  const IDXL_List ll = m->node.shared.getList(chk);
  localIdx = ll[sharedIdx];
  FEM_remove_node_local(m,localIdx);
  return;
}

void FEM_MUtil::addGhostElementRemote(FEM_Mesh *m, int chk, int elemType, int numGhostIndices, int *ghostIndices, int numSharedIndices, int *sharedIndices, int connSize) {

  int numNewGhostIndices = connSize - (numGhostIndices + numSharedIndices);
  int *conn = (int *)malloc(connSize*sizeof(int));
  for(int i=0; i<numNewGhostIndices; i++) {
    int newGhostNode = FEM_add_node_local(m, 1);
    m->node.ghost->ghostRecv.addNode(newGhostNode,chk);
    conn[i] = FEM_To_ghost_index(newGhostNode);
  }

  //convert existing remote ghost indices to local ghost indices 
  const IDXL_List ll1 = m->node.ghost->ghostRecv.getList(chk);
  for(int i=0; i<numGhostIndices; i++) {
    conn[i+numNewGhostIndices] = FEM_To_ghost_index(ll1[ghostIndices[i]]);
  }

  //convert sharedIndices to localIndices
  const IDXL_List ll2 = m->node.shared.getList(chk);
  for(int i=0; i<numSharedIndices; i++) {
    conn[i+numNewGhostIndices+numGhostIndices] = ll2[sharedIndices[i]];
  }

  int newGhostElement = FEM_add_element_local(m, conn, connSize, elemType, 1);
  m->elem[elemType].ghost->ghostRecv.addNode(FEM_To_ghost_index(newGhostElement),chk);
  free(conn);
  return;
}

chunkListMsg *FEM_MUtil::getChunksSharingGhostNodeRemote(FEM_Mesh *m, int chk, int sharedIdx) {
  const IDXL_List ll = m->node.ghostSend.getList(chk);
  int localIdx = ll[sharedIdx];
  int numChunkList = 0;
  const IDXL_Rec *tween = m->node.shared.getRec(localIdx);
  if(tween) {
    int numChunkList = tween->getShared();
  }
  chunkListMsg *clm = new (numChunkList, 0) chunkListMsg;
  clm->numChunkList = numChunkList;
  for(int i=0; i<numChunkList; i++) {
    clm->chunkList[i] = tween->getChk(i);
  }
  return clm;
}

void FEM_MUtil::buildChunkToNodeTable(int *nodetype, int sharedcount, int ghostcount, int localcount, int *conn, int connSize, CkVec<int> ***allShared, int *numSharedChunks, CkVec<int> **allChunks, int ***sharedConn) {
  if((sharedcount > 0 && ghostcount == 0) || (ghostcount > 0 && localcount == 0)) { 
    *allShared = (CkVec<int> **)malloc(connSize*sizeof(CkVec<int> *));
    for(int i=0; i<connSize; i++) {
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
  CkAssert(*numSharedChunks>0);
  return;
}

void FEM_MUtil::addElemRemote(FEM_Mesh *m, int chk, int elemtype, int connSize, int *conn, int numGhostIndex, int *ghostIndices) {
  //translate all the coordinates to local coordinates
  //chk is the chunk who send this message
  //convert sharedIndices to localIndices & ghost to local indices
  const IDXL_List ll1 = m->node.ghostSend.getList(chk);
  const IDXL_List ll2 = m->node.shared.getList(chk);

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

  FEM_add_element(m, localIndices, connSize, elemtype);
  free(localIndices);
  return;
}


void FEM_MUtil::removeGhostElementRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int numGhostIndex, int *ghostIndices, int numGhostRNIndex, int *ghostRNIndices, int numGhostREIndex, int *ghostREIndices, int numSharedIndex, int *sharedIndices) {
  //translate all ghost node coordinates to local coordinates and delete those ghost nodes on chk
  //remove ghost element elementid on chk
  int localIdx;
  
  const IDXL_List lgre = m->elem[elemtype].ghost->ghostRecv.getList(chk);
  localIdx = lgre[elementid];
  m->elem[elemtype].ghost->ghostRecv.removeNode(localIdx, chk); 
  FEM_remove_element_local(m, FEM_To_ghost_index(localIdx), elemtype);

  //convert existing remote ghost indices to local ghost indices 
  if(numGhostIndex > 0) {
    const IDXL_List lgrn = m->node.ghost->ghostRecv.getList(chk);
    for(int i=0; i<numGhostIndex; i++) {
      localIdx = lgrn[ghostIndices[i]];
      m->node.ghost->ghostRecv.removeNode(localIdx, chk); //do not delete ghost nodes
      FEM_remove_node_local(m, FEM_To_ghost_index(localIdx));
    }
  }

  if(numGhostREIndex > 0) {
    const IDXL_List lgse = m->elem[elemtype].ghostSend.getList(chk);
    for(int i=0; i<numGhostREIndex; i++) {
      localIdx = lgse[ghostREIndices[i]];
      m->elem[elemtype].ghostSend.removeNode(localIdx, chk); 
    }
  }

  if(numGhostRNIndex > 0) {
    const IDXL_List lgsn = m->node.ghostSend.getList(chk);
    for(int i=0; i<numGhostRNIndex; i++) {
      localIdx = lgsn[ghostRNIndices[i]];
      m->node.ghostSend.removeNode(localIdx, chk); 
    }
  }

  if(numSharedIndex > 0) { //a shared node became ghost
    const IDXL_List lsn = m->node.shared.getList(chk);
    for(int i=0; i<numSharedIndex; i++) {
      localIdx = lsn[sharedIndices[i]];
      m->node.shared.removeNode(localIdx, chk);
      m->node.ghostSend.addNode(localIdx, chk);
    }
  }

  return;
}

void FEM_MUtil::removeElemRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int permanent) {

  const IDXL_List ll = m->elem[elemtype].ghostSend.getList(chk);
  int localIdx = ll[elementid];
  FEM_remove_element(m, localIdx, elemtype, permanent);
  return;
}

int FEM_MUtil::lookup_in_IDXL(FEM_Mesh *m, int sharedIdx, int chk, int type, int elemType) {
  int localIdx  = -1;
  IDXL_List ll;
  if(type == 0) { //shared node
    ll = m->node.shared.getList(chk);
  }
  else if(type == 1) { //ghost node send 
    ll = m->node.ghostSend.getList(chk);
  }
  else if(type == 2) { //ghost node recv 
    ll = m->node.ghost->ghostRecv.getList(chk);
  }
  else if(type == 3) { //ghost node recv 
    ll = m->elem[elemType].ghostSend.getList(chk);
  }
  else if(type == 4) { //ghost node recv 
    ll = m->elem[elemType].ghost->ghostRecv.getList(chk);
  }
  localIdx = ll[sharedIdx];
  return localIdx;
}

int FEM_MUtil::getRemoteIdx(FEM_Mesh *m, int elementid, int elemtype) {
  CkAssert(elementid < -1);
  int ghostid = FEM_To_ghost_index(elementid);
  const IDXL_Rec *irec = m->elem[elemtype].ghost->ghostRecv.getRec(ghostid);
  int size = irec->getShared();
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
    m->node.set_valid(newIdx);   // set new node as valid
    m->n2e_removeAll(newIdx);    // initialize element adjacencies
    m->n2n_removeAll(newIdx);    // initialize node adjacencies
  }

  CkVec<FEM_Attribute *>*attrs;
  CkVec<FEM_Attribute *>*nattrs;
  if(FEM_Is_ghost_index(oldIdx)){
    attrs =  m->node.ghost->getAttrVec();
    //oldIdx = FEM_To_ghost_index(oldIdx);
  }
  else{
    attrs =  m->node.getAttrVec();
  }
  nattrs = m->node.getAttrVec();

  for(int j=0; j<attrs->size(); j++){
    FEM_Attribute *att = (FEM_Attribute *)(*attrs)[j];
    FEM_Attribute *natt = (FEM_Attribute *)(*nattrs)[j];
    if(att->getAttr() < FEM_ATTRIB_TAG_MAX){ 
      FEM_DataAttribute *d = (FEM_DataAttribute *)natt;
      if(FEM_Is_ghost_index(oldIdx)){
	d->copyEntity(newIdx, *att, FEM_To_ghost_index(oldIdx));
      }
      else {
	d->copyEntity(newIdx, *att, oldIdx);
      }
    }
  }

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
  m->n2n_removeAll(oldIdx);
  m->n2e_removeAll(oldIdx);

  delete[] nnbrs;
  delete[] enbrs;
  return newIdx;  // return a new index
}

void FEM_MUtil::addToSharedList(FEM_Mesh *m, int fromChk, int sharedIdx) {
  int elemType = 0;
  int connSize = m->elem[elemType].getConn().width();
  int localIdx = mmod->fmUtil->lookup_in_IDXL(m,sharedIdx,fromChk,1); //look in the ghostsend list

  //fix the idxl lists
  m->node.shared.addNode(localIdx, fromChk);
  m->node.ghostSend.removeNode(localIdx, fromChk);

  int *enbrs;
  int esize;
  m->n2e_getAll(localIdx, &enbrs, &esize);
  for(int i=0; i<esize; i++) {
    if(enbrs[i] >= 0) { //if it is a local element
      //if it exists in the ghostsend list already
      int exists = mmod->fmUtil->exists_in_IDXL(m, enbrs[i], fromChk, 3);
      if(exists == -1) {
	m->elem[elemType].ghostSend.addNode(enbrs[i], fromChk);
	//ghost nodes should be added only if they were not already present as ghosts on that chunk.
	int numNodesToAdd = 0;
	int numSharedGhosts = 0;
	int numSharedNodes = 0;
	int *sharedGhosts = (int *)malloc(connSize*sizeof(int));
	int *sharedNodes = (int *)malloc(connSize*sizeof(int));
	int *nnbrs = (int*)malloc(connSize*sizeof(int));
	m->e2n_getAll(enbrs[i], nnbrs, elemType);
	for(int j=0; j<connSize; j++) {
	  int sharedNode = mmod->fmUtil->exists_in_IDXL(m,nnbrs[j],fromChk,0);
	  if(sharedNode == -1) {
	    //node 'j' is a ghost on chunk 'i'
	    int sharedGhost = mmod->fmUtil->exists_in_IDXL(m,nnbrs[j],fromChk,1);
	    if( sharedGhost == -1) {
	      //it is a new ghost
	      m->node.ghostSend.addNode(nnbrs[j],fromChk);
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
	fm->connSize = connSize;
	meshMod[fromChk].addGhostElem(fm); 
	free(sharedGhosts);
	free(sharedNodes);
	free(nnbrs);
      }
    }
  }
  delete[] enbrs;
  return;
}

