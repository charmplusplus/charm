/* File: fem_mesh_modify.C
 * Authors: Isaac Dooley, Nilesh Choudhury
 * 
 * This file contains a set of functions, which allow primitive operations upon meshes in parallel.
 *
 * See the assumptions listed in fem_mesh_modify.h before using these functions.
 *
 */

#include "fem.h"
#include "fem_impl.h"
#include "fem_mesh_modify.h"

extern void splitEntity(IDXL_Side &c, int localIdx, int nBetween, int *between, int idxbase);


CProxy_femMeshModify meshMod;



// A wrapper to simplify the lookup to whether a node is shared
inline int is_shared(FEM_Mesh *m, int node){
  return m->getfmMM()->getfmUtil()->isShared(node);
}


CDECL void FEM_Print_Mesh_Summary(int mesh){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  
  // Print Element info
  
  CkPrintf("Element Types: %d\n", m->elem.size());
  for (int t=0;t<m->elem.size();t++) // for each element type t
	if (m->elem.has(t)) {
	  const int numElements = m->elem[t].size();
	  const int numGhostElements = ((FEM_Elem*)(m->elem[t].getGhost()))->size();
	  unsigned long numValidEl=0;
	  unsigned long numValidElGhosts=0;
	  unsigned char *validData;
	  FEM_DataAttribute *validAttr;
	  
	  validAttr = (FEM_DataAttribute *)m->elem[t].lookup(FEM_VALID,"FEM_Print_Mesh_Summary");
	  validData = validAttr->getChar().getData();	
	  CkAssert(validData);
	  for(int i=0;i<numElements;i++){
		if (validData[i])
		  numValidEl++;
	  }
	  
	  validAttr = (FEM_DataAttribute *)m->elem[t].getGhost()->lookup(FEM_VALID,"FEM_Print_Mesh_Summary");
	  validData = validAttr->getChar().getData();
	  CkAssert(validData);
	  for(int i=0;i<numElements;i++){
		if (validData[i])
		  numValidElGhosts++;
	  }
	  
	  CkPrintf("Element type %d contains %d/%d elements and %d/%d ghosts\n", t, numValidEl, numElements, numValidElGhosts, numGhostElements);
	}
}



int FEM_add_node_local(FEM_Mesh *m){
  // lengthen node attributes
  int oldLength = m->node.size();
  m->node.setLength(oldLength+1);
  const int newNode = oldLength;

  // set new node as valid
  FEM_DataAttribute *validAttr = (FEM_DataAttribute*)m->node.lookup(FEM_VALID,"FEM_add_node_local");
  unsigned char *validData = validAttr->getChar().getData();
  validData[newNode]=1;
  
  // return a new index
  return newNode;
}


int FEM_add_node(FEM_Mesh *m, int* adjacentNodes, int numAdjacentNodes, int upcall){
  // add local node
  int newNode = FEM_add_node_local(m);
  int sharedCount = 0;

  // for each adjacent node, if the node is shared
  for(int i=0;i<numAdjacentNodes;i++){ //a newly added node is shared only if all the 
    //nodes between which it is added are shared
    if(is_shared(m, adjacentNodes[i]))
      {
		sharedCount++;
        // lookup adjacent_nodes[i] in IDXL, to find all remote chunks which share this node
        // call_shared_node_remote() on all chunks for which the shared node exists
        // we must make sure that we only call the remote entry method once for each remote chunk
      }

  }

  //this is the entry in the IDXL.
  //since we are locking all chunks that are participating in an operation,
  //so, two different operations cannot happen on the same chunk, hence, the
  //entry in the IDXL will be correct
  //besides, do we have a basic operation, where two add_nodes are done within the same lock?
  //if so, we will have to ensure lock steps, to ensure correct idxl entries
  if(sharedCount == numAdjacentNodes) {
    m->getfmMM()->getfmUtil()->splitEntityAll(m, newNode, numAdjacentNodes, adjacentNodes, 0);
  }

  return newNode;
}


// The function called by the entry method on the remote chunk
void FEM_add_shared_node_remote(FEM_Mesh *m, int chk, int nBetween, int *between){
  // create local node
  int newnode = FEM_add_node_local(m);
  
  // must negotiate the common IDXL number for the new node, 
  // and store it in appropriate IDXL tables

  //note that these are the shared indices, local indices need to be calculated
  m->getfmMM()->getfmUtil()->splitEntityRemote(m, chk, newnode, nBetween, between, 0);
}




// remove a local or shared node, but NOT a ghost node
// Should probably be able to handle ghosts somday, but I cannot 
// remember the reasoning for not allowing them
void FEM_remove_node(FEM_Mesh *m, int node){

  if(FEM_Is_ghost_index(node))
    CkAbort("Cannot call FEM_remove_node on a ghost node\n");
  
  // if node is shared:
  if(is_shared(m, node)){
    // verify it is not adjacent to any elements locally
    int numAdjNodes, numAdjElts;
    int **adjNodes, **adjElts;
    m->n2n_getAll(node, adjNodes, &numAdjNodes);
    m->n2e_getAll(node, adjElts, &numAdjElts);
    CkAssert((numAdjNodes==0) && (numAdjElts==0)); // we shouldn't be removing a node away that is connected to anything
  
    
    
    // verify it is not adjacent to any elements on any of the associated chunks
    
	// mark node as deleted/invalid locally
	FEM_DataAttribute *validAttr = (FEM_DataAttribute*)m->node.lookup(FEM_VALID,"FEM_remove_node");
	unsigned char *validData = validAttr->getChar().getData();
	validData[node]=0;

    // delete it on remote chunks(shared and ghost), update IDXL tables
    

  }
  else {
    // if node is local:
    int numAdjNodes, numAdjElts;
    int **adjNodes, **adjElts;
    m->n2n_getAll(node, adjNodes, &numAdjNodes);
    m->n2e_getAll(node, adjElts, &numAdjElts);
    CkAssert((numAdjNodes==0) && (numAdjElts==0)); // we shouldn't be removing a node away that is connected to anything
    
    // mark node as deleted/invalid
	FEM_DataAttribute *validAttr = (FEM_DataAttribute*)m->node.lookup(FEM_VALID,"FEM_remove_node");
	unsigned char *validData = validAttr->getChar().getData();
	validData[node]=0;

  }
}


// remove a local element from the adjacency tables as well as the element list
void FEM_remove_element_local(FEM_Mesh *m, int element, int etype){

  // find adjacent nodes
  int width = m->elem[etype].getConn().size(); // should be the number of nodes that can be adjacent to this element
  int *adjnodes = new int[width];
  m->e2n_getAll(element, adjnodes, etype);
  
  // replace me in their adjacencies with -1
  for(int i=0;i<width;i++){
    m->n2e_replace(adjnodes[i],element,-1);
  }

  // find adjacent elements
  width = m->elem[etype].getConn().size();
  int *adjelts = new int[width];
  m->e2e_getAll(element, adjelts, etype);
  
  // replace me in their adjacencies with -1
  for(int i=0;i<width;i++){
    m->e2e_replace(adjelts[i],element,-1);
  }

  // delete element by marking invalid
  // mark node as deleted/invalid
  if(FEM_Is_ghost_index(element)){
	FEM_DataAttribute *validAttr = (FEM_DataAttribute*)m->elem[etype].getGhost()->lookup(FEM_VALID,"FEM_remove_element_local");
	unsigned char *validData = validAttr->getChar().getData();
	validData[FEM_To_ghost_index(element)]=0;
  }
  else {
	FEM_DataAttribute *validAttr = (FEM_DataAttribute*)m->elem[etype].lookup(FEM_VALID,"FEM_remove_element_local");
	unsigned char *validData = validAttr->getChar().getData();
	validData[element]=0;
  }

  delete[] adjnodes;
}

// Can be called on local or ghost elements
void FEM_remove_element(FEM_Mesh *m, int element, int elemType){

  if(FEM_Is_ghost_index(element)){
    // remove local ghost element
    FEM_remove_element_local(m, element, elemType);
    
    // call FEM_remove_element_remote on other chunk which owns the element   
  }
  else {
    // remove local element
    FEM_remove_element_local(m, element, elemType);

    // call FEM_remove_element_remote on any other chunk for which this is a ghost
  }
}

void FEM_remove_element_remote(FEM_Mesh *m, int element, int elemType){
  // remove local element from elem[elemType] table
}


// A helper function for FEM_add_element_local below
// Will only work with the same element type as the one given, may crash otherwise
void update_new_element_e2e(FEM_Mesh *m, int newEl, int elemType){
  CkAssert(!FEM_Is_ghost_index(newEl)); // TODO: fix this function to handle ghosts
  // Create tuple table
  FEM_ElemAdj_Layer *g = m->getElemAdjLayer();
  CkAssert(g->initialized);
  const int nodesPerTuple = g->nodesPerTuple;
  tupleTable table(nodesPerTuple);
  FEM_Symmetries_t allSym;

  // insert the new element
  const int tuplesPerElem = g->elem[elemType].tuplesPerElem;
  int tuple[tupleTable::MAX_TUPLE];
  const int *conn=m->elem[elemType].connFor(newEl);
  for (int u=0;u<tuplesPerElem;u++)
    for (int i=0;i<nodesPerTuple;i++) {
      int eidx=g->elem[elemType].elem2tuple[i+u*g->nodesPerTuple];
      if (eidx==-1)  //"not-there" node--
        tuple[i]=-1; //Don't map via connectivity
      else           //Ordinary node
        tuple[i]=conn[eidx]; 
    
      table.addTuple(tuple,new elemList(0,newEl,elemType,allSym,u)); 
    }

  // insert all elements adjacent to the nodes adjacent to the new element
  int *adjnodes = new int[tuplesPerElem];

  m->e2n_getAll(newEl, adjnodes, elemType);
  
  for(int i=0;i<tuplesPerElem;i++){
    int sz;
    int **adjelements;
    m->n2e_getAll(adjnodes[i], adjelements, &sz);
    
    for(int j=0;j<sz;j++){
      int elementToAdd = *adjelements[j];
      const int tuplesPerElem = g->elem[elemType].tuplesPerElem;
      int tuple[tupleTable::MAX_TUPLE];
      const int *conn=m->elem[elemType].connFor(elementToAdd);
      for (int u=0;u<tuplesPerElem;u++)
        for (int i=0;i<nodesPerTuple;i++) {
          int eidx=g->elem[elemType].elem2tuple[i+u*g->nodesPerTuple];
          if (eidx==-1)  //"not-there" node--
            tuple[i]=-1; //Don't map via connectivity
          else           //Ordinary node
            tuple[i]=conn[eidx]; 
          
          table.addTuple(tuple,new elemList(0,elementToAdd,elemType,allSym,u)); 
        }
      
    }
  }
  delete[] adjnodes;

  
  // extract adjacencies from table and update all e2e tables for both newEl and the others
  
    
  // look through each elemList that is returned by the tuple table
  elemList *l;
  FEM_IndexAttribute *elemAdjTypesAttr = (FEM_IndexAttribute *)m->elem[elemType].lookup(FEM_ELEM_ELEM_ADJ_TYPES,"update_new_element_e2e");
  FEM_IndexAttribute *elemAdjAttr = (FEM_IndexAttribute *)m->elem[elemType].lookup(FEM_ELEM_ELEM_ADJACENCY,"update_new_element_e2e");
  FEM_IndexAttribute *elemAdjTypesAttrGhost = (FEM_IndexAttribute *)m->elem[elemType].getGhost()->lookup(FEM_ELEM_ELEM_ADJ_TYPES,"update_new_element_e2e");
  FEM_IndexAttribute *elemAdjAttrGhost = (FEM_IndexAttribute *)m->elem[elemType].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"update_new_element_e2e");

  AllocTable2d<int> &adjTable = elemAdjAttr->get();
  int *adjs = adjTable.getData();
  AllocTable2d<int> &adjTypesTable = elemAdjTypesAttr->get();
  int *adjTypes = adjTypesTable.getData();
  AllocTable2d<int> &adjTableGhost = elemAdjAttrGhost->get();
  int *adjsGhost = adjTableGhost.getData();
  AllocTable2d<int> &adjTypesTableGhost = elemAdjTypesAttrGhost->get();
  int *adjTypesGhost = adjTypesTableGhost.getData();
  
  while (NULL!=(l=table.lookupNext())) {
    if (l->next==NULL) { 
      // One-entry list: must be a symmetry
      // UNHANDLED CASE: not sure exactly what this means
    }
    else { /* Several elements in list: normal case */
      // for each a,b from the list
      for (const elemList *a=l;a!=NULL;a=a->next){
        for (const elemList *b=l;b!=NULL;b=b->next){
          // if a and b are different elements
          if((a->localNo != b->localNo) || (a->type != b->type)){
            int j;
            if(FEM_Is_ghost_index(a->localNo))
              j = FEM_To_ghost_index(a->localNo)*tuplesPerElem + a->tupleNo;
            else
              j = a->localNo*tuplesPerElem + a->tupleNo;
            
            if(a->type == elemType){ // only update the entries for element type t
              adjs[j] = b->localNo;
              adjTypes[j] = b->type;              
            }
          }
        }
      }
    }
  } 
}


// A helper function that adds the local element, and updates adjacencies
int FEM_add_element_local(FEM_Mesh *m, const int *conn, int connSize, int elemType){
  // lengthen element attributes
  int oldLength = m->elem[elemType].size();
  m->elem[elemType].setLength(oldLength+1);
  const int newEl = oldLength;

  // Mark new element as valid
  FEM_DataAttribute *validAttr = (FEM_DataAttribute*)m->elem[elemType].lookup(FEM_VALID,"FEM_add_element_local");
  unsigned char *validData = validAttr->getChar().getData();
  validData[newEl]=1;
  
  // update element's conn, i.e. e2n table
  m->elem[elemType].connIs(newEl,conn);
  
  // add to corresponding inverse, the n2e and n2n table
  for(int i=0;i<connSize;i++){
    m->n2e_add(conn[i],newEl);
    for(int j=i+1;j<connSize;j++){
      if(! m->n2n_exists(i,j))
        m->n2n_add(i,j);
      if(! m->n2n_exists(j,i))
        m->n2n_add(j,i);
    }
  }

  // update e2e table -- too complicated, so it gets is own function
  update_new_element_e2e(m,newEl,elemType);
  
  return newEl;
}


int FEM_add_element(FEM_Mesh *m, int* conn, int connSize, int elemType){
  
  int sharedcount=0;
  int ghostcount=0;
  for(int i=0;i<connSize;i++){
    if(is_shared(m,conn[i])) sharedcount++;
    if(FEM_Is_ghost_index(conn[i])) ghostcount++;
  }

  if(sharedcount==0 && ghostcount==0){// no ghost or shared nodes in conn
    return FEM_add_element_local(m,conn,connSize,elemType);
  }
  else if(ghostcount==0){// else if any shared nodes but no ghosts in conn
    int newEl = FEM_add_element_local(m,conn,connSize,elemType);
    
    //   make this element ghost on all others, updating all IDXL's
    //   also in same remote entry method, update adjacencies on all others
    //   grow local element and attribute tables if needed
    //   add to local elem[elemType] table, and update IDXL if needed
    //   update local adjacencies
    //   return the new element id
  }
  else if(ghostcount !=0){// else if any ghosts in conn
   
    //   promote ghosts to shared on others, requesting new ghosts
    //   grow local element and attribute tables if needed
    //   add to local elem[elemType] table, and update IDXL if needed
    //   update remote adjacencies
    //   update local adjacencies
  }

  return 0;
}


int FEM_add_element_remote(){
  // promote ghosts to shared

  // find new ghosts for remote calling chunk by looking at new shared nodes
  // send these new ghosts to the remote calling chunk.

  // update my adjacencies

  return 0;
}


void FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes, int numAffectedNodes, int* affectedElts, int numAffectedElts) {
  m->getfmMM()->getfmLock()->lock(numAffectedNodes, affectedNodes, numAffectedElts, affectedElts);
  return;
}

void FEM_Modify_Unlock(FEM_Mesh *m) {
  m->getfmMM()->getfmLock()->unlock();
  return;
}


CDECL void FEM_REF_INIT(int mesh) {
  CkArrayID femRefId;
  int cid;
  int size;
  TCharm *tc=TCharm::get();

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&cid);
  MPI_Comm_size(comm,&size);
  if(cid==0) {
    CkArrayOptions opts;
    opts.bindTo(tc->getProxy()); //bind to the current proxy
    femMeshModMsg *fm = new femMeshModMsg;
    femRefId = CProxy_femMeshModify::ckNew(fm, opts);
  }
  MPI_Bcast(&femRefId, sizeof(CkArrayID), MPI_BYTE, 0, comm);
  meshMod = femRefId;

  femMeshModMsg *fm = new femMeshModMsg(size,cid);
  meshMod[cid].insert(fm);


  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_REF_INIT");
  FEMMeshMsg *msg = new FEMMeshMsg(m); 
  meshMod[cid].setFemMesh(msg);

  return;
}


FEM_lock::FEM_lock(int i, femMeshModify *m) {
  idx = i;
  owner = -1;
  isOwner = false;
  isLocked = false;
  hasLocks = false;
  mmod = m;
}

FEM_lock::~FEM_lock() {
  //before deleting it, ensure that it is not holding any locks
  if(hasLocks) {
    unlock();
  }
  delete &lockedChunks;
}

bool FEM_lock::existsChunk(int index) {
  for(int i=0; i<lockedChunks.size(); i++) {
    if(lockedChunks[i] == index) return true;
  }
  return false;
}

//will only return if it gets the locks
//the set of nodes or elems that a chunk can ask to be locked can either be 
//shared, ghosts or local nodes/elements.
int FEM_lock::lock(int numNodes, int *nodes, int numElems, int* elems) {
  bool done = false;
  int ret = 0;
  while(!done) {
    if(!isLocked || (isLocked && isOwner)) {
      for(int i=0; i<numNodes; i++) {
	//which chunk does this belong to
	//add that chunk to the lock list, if it does not exist already.
	int numchunks;
	IDXL_Share **chunks;
	mmod->fmUtil->getChunkNos(0,nodes[i],&numchunks,chunks);
	for(int j=0; j<numchunks; j++) {
	  if(!existsChunk(chunks[j]->chk)) {
	    lockedChunks.push_back(chunks[j]->chk);
	  }
	}
      }
      for(int i=0; i<numElems; i++) {
	//which chunk does this belong to
	//add that chunk to the lock list, if not already in it.
	int numchunks;
	IDXL_Share **chunks;
	mmod->fmUtil->getChunkNos(1,elems[i],&numchunks,chunks);
	for(int j=0; j<numchunks; j++) {
	  if(!existsChunk(chunks[j]->chk)) {
	    lockedChunks.push_back(chunks[j]->chk);
	  }
	}
      }

      //sort the elements in ascending order
      int tmp;
      int numLocks = lockedChunks.size();
      for(int i=0; i<numLocks; i++) {
	for(int j=i+1; j<numLocks; j++) {
	  if(lockedChunks[i] > lockedChunks[j]) {
	    tmp = lockedChunks[i];
	    lockedChunks[i] = lockedChunks[j];
	    lockedChunks[j] = tmp;
	  }
	}
      }

      //lock them
      for(int i=0; i<numLocks; i++) {
	ret = lock(lockedChunks[i],idx);
	if(ret != 1) return -1;
      }
      done = true;
    }
    else {
      CthYield();
      //block
    }
  }
  return 1;
}

int FEM_lock::unlock() {
  bool done = false;
  int ret = 0;
  while(!done) {
    if(!isLocked || (isLocked && isOwner)) {
      //get rid of the locks
      if(hasLocks) {
	for(int i=0; i<lockedChunks.size(); i++) {
	  ret = unlock(lockedChunks[i],idx);
	  if(ret != 1) return -1;
	}
      }
      hasLocks = false;
    }
    else {
      CthYield();
      //block
    }
  }
  return 1;
}

int FEM_lock::lock(int chunkNo, int own) {
  intMsg *ret = new intMsg(0);
  while(true) {
    if(!isLocked || (chunkNo != idx)) {
      if(chunkNo == idx) {
	isLocked = true;
	owner = own;
	if(owner == idx) isOwner = true;
	else isOwner = false;
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].lockRemoteChunk(imsg);
	if(ret->i != 1) return -1;
      }
      break;
    }
    else {
      CthYield();
    }
  }
  return 1;
}

//for sanity, only the owner should unlock it
int FEM_lock::unlock(int chunkNo, int own) {
  intMsg *ret = new intMsg(0);
  while(true) {
    if(!isLocked && (chunkNo == idx)) {
      CkError("%d trying to unlock %d which is not locked!!\n",own,chunkNo);
      return -1;
    }
    else if(isLocked && (chunkNo == idx) && (owner != own)) {
      CkError("%d trying to unlock %d which is locked by %d!!\n",own,chunkNo,owner);
      return -1;
    }
    else if(isLocked || (chunkNo != idx)) {
      if(chunkNo == idx) {
	isLocked = false;
	owner = -1;
	isOwner = false;
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].lockRemoteChunk(imsg);
	if(ret->i != 1) return -1;
      }
      break;
    }
    else {
      CthYield();
    }
  }
  return 1;
}


FEM_MUtil::FEM_MUtil(int i, femMeshModify *m) {
  idx = i;
  mmod = m;
}

FEM_MUtil::~FEM_MUtil() {
}

void FEM_MUtil::getChunkNos(int entType, int entNo, int *numChunks, IDXL_Share **chunks) {
  int type = 0; //0 - local, 1 - shared, 2 - ghost.

  if(entType == 0) { //nodes
    //only nodes can be shared
    if(FEM_Is_ghost_index(entNo)) type = 2;
    else if(isShared(entNo)) type = 1;
    else type = 0;

    if(type == 2) {
      int ghostid = FEM_To_ghost_index(entNo);
      const IDXL_Rec *irec = mmod->fmMesh->node.getGhostRecv().getRec(ghostid);
      *numChunks = irec->getShared(); //check this value!!
      chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = irec->getChk(i);
	int index = irec->getIdx(i);
	chunks[i] = new IDXL_Share(chk, index);
      }
    }
    else if(type == 1) {
      const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(entNo);
      *numChunks = irec->getShared();
      chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = irec->getChk(i);
	int index = irec->getIdx(i);
	chunks[i] = new IDXL_Share(chk, index);
      }
    }
    else if(type == 0) {
      *numChunks = 1;
      chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = idx; //index of this chunk
	int index = entNo;
	chunks[i] = new IDXL_Share(chk, index);
      }
    }
  }
  else if(entType == 1) { //elems
    //elements cannot be shared
    if(FEM_Is_ghost_index(entNo)) type = 2;
    else type = 0;

    if(type == 2) {
      int ghostid = FEM_To_ghost_index(entNo);
      const IDXL_Rec *irec = mmod->fmMesh->node.getGhostRecv().getRec(ghostid);
      *numChunks = irec->getShared(); //should be 1
      chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = irec->getChk(i);
	int index = irec->getIdx(i);
	chunks[i] = new IDXL_Share(chk, index);
      }
    }
    else if(type == 0) {
      *numChunks = 1;
      chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = idx; //index of this chunk
	int index = entNo;
	chunks[i] = new IDXL_Share(chk, index);
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
      delete &ll; //clean up
      sharedNodeMsg *fm = new sharedNodeMsg(mmod->idx,nBetween,sharedIndices);
      meshMod[chk].addSharedNodeRemote(fm);
      //break;
    }
  }
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
  splitEntity(m->node.shared, localIdx, nBetween, localIndices, idxbase);
  return;
}


femMeshModify::femMeshModify(femMeshModMsg *fm) {
  numChunks = fm->numChunks;
  idx = fm->myChunk;
  fmLock = new FEM_lock(idx, this);
  fmUtil = new FEM_MUtil(idx, this);
  fmMesh = NULL;
}

femMeshModify::~femMeshModify() {
  if(fmLock != NULL) {
    delete fmLock;
  }
  if(fmUtil != NULL) {
    delete fmUtil;
  }
}

void femMeshModify::setFemMesh(FEMMeshMsg *fm) {
  fmMesh = fm->m;
  fmMesh->setFemMeshModify(this);
  return;
}

intMsg *femMeshModify::lockRemoteChunk(int2Msg *msg) {
  intMsg *imsg = new intMsg(0);
  int ret = fmLock->lock(msg->i, msg->j);
  imsg->i = ret;
  return imsg;
}

intMsg *femMeshModify::unlockRemoteChunk(int2Msg *msg) {
  intMsg *imsg = new intMsg(0);
  int ret = fmLock->unlock(msg->i, msg->j);
  imsg->i = ret;
  return imsg;
}

void femMeshModify::addSharedNodeRemote(sharedNodeMsg *fm) {

  FEM_add_shared_node_remote(fmMesh, fm->chk, fm->nBetween, fm->between);
  return;
}


#include "FEMMeshModify.def.h"
