/* File: fem_mesh_modify.C
 * Authors: Isaac Dooley, Nilesh Choudhury
 * 
 * This file contains a set of functions, which allow primitive operations upon meshes in parallel.
 *
 * See the assumptions listed in fem_mesh_modify.h before using these functions.
 *
 */

#include "charm.h"
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
  CkPrintf("---------------FEM_Print_Mesh_Summary-------------\n");
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");

  // Print Node information
  CkPrintf("     Nodes: %d/%d and ghost nodes: %d/%d\n", m->node.count_valid(), m->node.size(),m->node.getGhost()->count_valid(),m->node.getGhost()->size());

  // Print Element information
  CkPrintf("     Element Types: %d\n", m->elem.size());
  for (int t=0;t<m->elem.size();t++) // for each element type t
	if (m->elem.has(t)) {
	  unsigned int numEl = m->elem[t].size();
	  unsigned int numElG = m->elem[t].getGhost()->size();
	  unsigned int numValidEl = m->elem[t].count_valid();
	  unsigned int numValidElG = m->elem[t].getGhost()->count_valid();
	  CkPrintf("     Element type %d contains %d/%d elements and %d/%d ghosts\n", t, numValidEl, numEl, numValidElG, numElG);
	  
	}

  CkPrintf("\n");
}


int FEM_add_node_local(FEM_Mesh *m){
  const int newNode = m->node.size();
  m->node.setLength(newNode+1); // lengthen node attributes
  m->node.set_valid(newNode);   // set new node as valid
  m->n2e_removeAll(newNode);    // initialize element adjacencies
  m->n2n_removeAll(newNode);    // initialize node adjacencies
  return newNode;  // return a new index
}

int FEM_add_ghost_node_local(FEM_Mesh *m) {
  const int newNode = m->node.getGhost()->size();
  m->node.getGhost()->setLength(newNode+1); // lengthen node attributes
  m->node.getGhost()->set_valid(newNode);   // set new node as valid
  //I guess ghosts do not have adjacency information
  //m->n2e_removeAll(newNode);    // initialize element adjacencies
  //m->n2n_removeAll(newNode);    // initialize node adjacencies
  return newNode;  // return a new index
}

int FEM_add_ghost_elem_local(FEM_Mesh *m, int elemType) {
  const int newNode = m->elem[elemType].getGhost()->size();
  m->elem[elemType].getGhost()->setLength(newNode+1); // lengthen node attributes
  m->elem[elemType].getGhost()->set_valid(newNode);   // set new node as valid
  //I guess ghosts do not have adjacency information
  //m->n2e_removeAll(newNode);    // initialize element adjacencies
  //m->n2n_removeAll(newNode);    // initialize node adjacencies
  return newNode;  // return a new index
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
  if(sharedCount==numAdjacentNodes && numAdjacentNodes!=0) {
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


void FEM_remove_node_local(FEM_Mesh *m, int node) {
    // if node is local:
    int numAdjNodes, numAdjElts;
    int *adjNodes, *adjElts;
    m->n2n_getAll(node, &adjNodes, &numAdjNodes);
    m->n2e_getAll(node, &adjElts, &numAdjElts);
    CkAssert((numAdjNodes==0) && (numAdjElts==0)); // we shouldn't be removing a node away that is connected to anything
    
    // mark node as deleted/invalid
	m->node.set_invalid(node);  
}


// remove a local or shared node, but NOT a ghost node
// Should probably be able to handle ghosts someday, but I cannot 
// remember the reasoning for not allowing them
void FEM_remove_node(FEM_Mesh *m, int node){

  if(FEM_Is_ghost_index(node))
    CkAbort("Cannot call FEM_remove_node on a ghost node\n");
  
  CkAssert(m->node.is_valid(node)); // make sure the node is still there
  
  // if node is shared:
  if(is_shared(m, node)){
    // verify it is not adjacent to any elements locally
    int numAdjNodes, numAdjElts;
    int *adjNodes, *adjElts;
    m->n2n_getAll(node, &adjNodes, &numAdjNodes);
    m->n2e_getAll(node, &adjElts, &numAdjElts);
    CkAssert((numAdjNodes==0) && (numAdjElts==0)); // we shouldn't be removing a node away that is connected to anything
  
    
    
    // verify it is not adjacent to any elements on any of the associated chunks
    
	// mark node as deleted/invalid locally
	m->node.set_invalid(node);

    // delete it on remote chunks(shared and ghost), update IDXL tables
	m->getfmMM()->getfmUtil()->removeNodeAll(m, node);

  }
  else {
    FEM_remove_node_local(m,node);
  }
}


// remove a local element from the adjacency tables as well as the element list
void FEM_remove_element_local(FEM_Mesh *m, int element, int etype){

  // replace this element with -1 in adjacent nodes' adjacencies
  const int nodesPerEl = m->elem[etype].getConn().width(); // should be the number of nodes that can be adjacent to this element
  int *adjnodes = new int[nodesPerEl];
  m->e2n_getAll(element, adjnodes, etype);
  for(int i=0;i<nodesPerEl;i++)
    if(adjnodes[i] != -1) //if an element is local, then an adjacent node should not be -1
	  m->n2e_remove(adjnodes[i],element);
  
  // replace this element with -1 in adjacent elements' adjacencies
  const int numAdjElts = nodesPerEl;    // FIXME: hopefully there will be at most as many faces on an element as vertices
  int *adjelts = new int[numAdjElts]; 
  m->e2e_getAll(element, adjelts, etype);
  for(int i=0;i<numAdjElts;i++){
    m->e2e_replace(adjelts[i],element,-1);
  }
  
  // delete element by marking invalid
  if(FEM_Is_ghost_index(element)){
	m->elem[etype].getGhost()->set_invalid(FEM_To_ghost_index(element));
  }
  else {
	m->elem[etype].set_invalid(element);
  }
  
  // We must now remove any n2n adjacencies which existed because of the 
  // element that is now being removed. This is done by removing all 
  // n2n adjacencies for the nodes of this element, and recreating those
  // that still exist by using the neighboring elements.
  //FIXME: I think, if we just consider on which all faces (for a 2D, a face is an edge),
  //we have valid neighboring elements, then none of the edges (n2n conn) on that
  //face should go off, this would be more efficient.
  for(int i=0;i<nodesPerEl;i++)
	for(int j=i+1;j<nodesPerEl;j++){
	  m->n2n_remove(adjnodes[i],adjnodes[j]);
	  m->n2n_remove(adjnodes[j],adjnodes[i]);
	}

  for(int i=0;i<numAdjElts;i++){ // for each neighboring element
	if(adjelts[i] != -1){
	  int *adjnodes2 = new int[nodesPerEl];
	  m->e2n_getAll(adjelts[i], adjnodes2, etype);
	
	  for(int j=0;j<nodesPerEl;j++){     // for each j,k pair of nodes adjacent to the neighboring element
		for(int k=j+1;k<nodesPerEl;k++){   
		  if(!m->n2n_exists(adjnodes2[j],adjnodes2[k]))
			m->n2n_add(adjnodes2[j],adjnodes2[k]);
		  if(!m->n2n_exists(adjnodes2[k],adjnodes2[j]))
			m->n2n_add(adjnodes2[k],adjnodes2[j]);
		}
	  }
	 
	  delete[] adjnodes2;
	}
  }
 
  delete[] adjelts;
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
  //  CkPrintf("nodesPerTuple=%d\n", nodesPerTuple);
  tupleTable table(nodesPerTuple);
  FEM_Symmetries_t allSym;


  // insert all elements adjacent to the nodes adjacent to the new 
  // element, including ghosts, and the new element itself

  const int tuplesPerElem = g->elem[elemType].tuplesPerElem;
  int *adjnodes = new int[tuplesPerElem];
  CkVec<int> elist;
  m->e2n_getAll(newEl, adjnodes, elemType);
  for(int i=0;i<tuplesPerElem;i++){
    int sz;
    int *adjelements;
    m->n2e_getAll(adjnodes[i], &adjelements, &sz);
    for(int j=0;j<sz;j++){
	  int found=0;
	  // only insert if it is not already in the list
	  for(int i=0;i<elist.length();i++)// we use a slow linear scan of the vector
		if(elist[i] == adjelements[j])
		  found=1;
	  if(!found){
		elist.push_back(adjelements[j]);
		//	CkPrintf("Adding element %d to list\n", adjelements[j]);
	  }
	}
	delete[] adjelements;
  }
  delete[] adjnodes;
  
  
  for(int i=0;i<elist.length();i++){
	int nextElem = elist[i];
	//CkPrintf("Adding elem %d to tuple table\n", nextElem);
	int tuple[tupleTable::MAX_TUPLE];
	const int *conn=m->elem[elemType].connFor(nextElem);
	//CkPrintf("tuplesPerElem=%d\n", tuplesPerElem);
	for (int u=0;u<tuplesPerElem;u++) {
	  for (int i=0;i<nodesPerTuple;i++) {
		int eidx=g->elem[elemType].elem2tuple[i+u*g->nodesPerTuple];
		if (eidx==-1)  //"not-there" node--
		  tuple[i]=-1; //Don't map via connectivity
		else           //Ordinary node
		  tuple[i]=conn[eidx]; 
	  }
	  
	  // CkPrintf("tuple=%d,%d\n", tuple[0], tuple[1]);
	  table.addTuple(tuple,new elemList(0,nextElem,elemType,allSym,u)); 
	}
  }

  
  // extract adjacencies from table and update all e2e tables for both newEl and the others
  table.beginLookup();
    
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
    if (l->next==NULL) {} // One-entry list: must be a symmetry
    else { /* Several elements in list: normal case */
      // for each a,b pair of adjacent edges
      for (const elemList *a=l;a!=NULL;a=a->next){
        for (const elemList *b=l;b!=NULL;b=b->next){
          // if a and b are different elements
          if((a->localNo != b->localNo) || (a->type != b->type)){
            int j;
	    //	CkPrintf("%d:%d:%d adj to %d:%d:%d\n", a->type, a->localNo, a->tupleNo, b->type, b->localNo, b->tupleNo);
	    // Put b in a's adjacency list
	    if(FEM_Is_ghost_index(a->localNo)){
	      j = FEM_To_ghost_index(a->localNo)*tuplesPerElem + a->tupleNo;
	      adjsGhost[j] = b->localNo;
	      adjTypesGhost[j] = b->type;
	    }
	    else{
	      j= a->localNo*tuplesPerElem + a->tupleNo;
	      adjs[j] = b->localNo;
	      adjTypes[j] = b->type;
	    }

	    // Put a in b's adjacency list
	    if(FEM_Is_ghost_index(b->localNo)){
	      j = FEM_To_ghost_index(b->localNo)*tuplesPerElem + b->tupleNo;
	      adjsGhost[j] = a->localNo;
	      adjTypesGhost[j] = a->type;
	    }
	    else{
	      j= b->localNo*tuplesPerElem + b->tupleNo;
	      adjs[j] = a->localNo;
	      adjTypes[j] = a->type;
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

  m->e2n_removeAll(newEl);
  m->e2e_removeAll(newEl);

  // Mark new element as valid
  m->elem[elemType].set_valid(newEl);
  
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
  
  int newEl = -1;
  int sharedcount=0;
  int ghostcount=0;
  int localcount=0;
  int *nodetype = (int *)malloc(connSize *sizeof(int)); //0 -- local, 1 -- shared, 2--ghost
  for(int i=0;i<connSize;i++){
    if(is_shared(m,conn[i])) {
      nodetype[i] = 1;
      sharedcount++;
    }
    else if(FEM_Is_ghost_index(conn[i])) {
      nodetype[i] = 2;
      ghostcount++;
    }
    else {
      nodetype[i] = 0;
    }
  }
  localcount = connSize - (sharedcount + ghostcount);

  //build a mapping of all shared chunks to all nodes in this element
  //this should probably be a separate function! -- start here
  CkVec<int> **allShared;
  int numSharedChunks = 0;
  CkVec<int> *allChunks;
  int **sharedConn;
  if((sharedcount > 0 && ghostcount == 0) ||(ghostcount > 0 && localcount > 0 && sharedcount > 0)) { //this is a local element or it will be a local element
    allShared = (CkVec<int> **)malloc(connSize*sizeof(CkVec<int> *));
    for(int i=0; i<connSize; i++) {
      if(nodetype[i] == 1) { //it is a shared node, figure out all chunks where the ghosts need to be added
	allShared[i] = new CkVec<int>; //list of chunks shared with node
	int numchunks;
	IDXL_Share **chunks1;
	m->getfmMM()->getfmUtil()->getChunkNos(0,conn[i],&numchunks,&chunks1);
	for(int j=0; j<numchunks; j++) {
	  allShared[i]->push_back(chunks1[j]->chk);
 	}
      }
    }
    //translate the information in a reverse data structure -- which chunk has which nodes as shared
    allChunks = new CkVec<int>;
    for(int i=0; i<connSize; i++) {
      if(allShared[i] != NULL) {
	for(int j=0; j<allShared[i]->size(); j++) {
	  int exists = 0;
	  for(int k=0; k<allShared[i]->size(); k++) {
	    if((*allShared[i])[k]==(*allShared[i])[j]) {
	      exists = 1;
	      break;
	    }
	  }
	  if(!exists) {
	    allChunks->push_back((*allShared[i])[j]);
	    numSharedChunks++;
	  }
	}
      }
      else {
	//node 'i' is local hence not shared with any chunk
      }
    }
    sharedConn = (int **)malloc(connSize*sizeof(int *));
    for(int i=0; i<connSize; i++) {
      if(allShared[i] != NULL) {
	sharedConn[i] = (int*)malloc(numSharedChunks*sizeof(int));
	for(int j=0; j<numSharedChunks; j++) {//initialize
	  sharedConn[i][j] = 0;
	}
	for(int j=0; j<allShared[i]->size(); j++) {
	  sharedConn[i][(*allShared[i])[j]] = 1; 
	}
      }
      else {
	//node 'i' is local hence not shared with any chunk
	sharedConn[i] = (int*)malloc(numSharedChunks*sizeof(int));
	for(int j=0; j<numSharedChunks; j++) {
	  sharedConn[i][j] = 0; 
	}
      }
    }
  }
  //function -- ends here

  if(sharedcount==0 && ghostcount==0){// add a local elem with all local nodes
    newEl = FEM_add_element_local(m,conn,connSize,elemType);
    //no modifications required for ghostsend or ghostrecv of nodes or elements
  }
  else if(ghostcount==0 && sharedcount > 0){// a local elem with some shared nodes
    newEl = FEM_add_element_local(m,conn,connSize,elemType);
    
    //   make this element ghost on all others, updating all IDXL's
    
    //add all the local nodes in this element to the ghost list, if they did not exist already
    for(int i=0; i<numSharedChunks; i++) {
      int chk = (*allChunks)[i];
      const FEM_Comm &gesend = m->elem[elemType].getGhostSend();
      //gesend.addNode(newEl,chk);
      addGhostElemMsg *fm = new addGhostElemMsg;
      fm->chk = m->getfmMM()->getfmUtil()->getIdx();
      fm->index = newEl;
      fm->elemType = elemType;
      meshMod[chk].addGhostElem(fm); //newEl, m->fmMM->idx;
      //ghost nodes should be added only if they were not already present as ghosts on that chunk.
      for(int j=0; j<connSize; j++) {
	if(sharedConn[j][i] == 0) {
	  //is it a ghost on that chunk
	  if(m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[j],chk) != 1) {
	    const FEM_Comm &gnsend = m->node.getGhostSend();
	    //gnsend.addNode(conn[j],chk);
	    addGhostNodeMsg *fm = new addGhostNodeMsg;
	    fm->chk = m->getfmMM()->getfmUtil()->getIdx();
	    fm->index = conn[i];
	    meshMod[chk].addGhostNode(fm); //conn[i], m->fmMM->idx;
	  }
	}
      }
    }

    //   also in same remote entry method, update adjacencies on all others
    //   grow local element and attribute tables if needed
    //   add to local elem[elemType] table, and update IDXL if needed
    //   update local adjacencies
    //   return the new element id

  }
  else if(ghostcount > 0 && localcount == 0 && sharedcount > 0) { // it is remote elem with some shared nodes
    //figure out which chunk it is local to
  }
  else if(ghostcount > 0 && localcount == 0 && sharedcount == 0) { // it is a remote elem with no shared nodes
    //I guess such a situation will never occur
    //figure out which chunk it is local to
  }
  else if(ghostcount > 0 && localcount > 0 && sharedcount > 0){// it is a flip operation
   
    //   promote ghosts to shared on others, requesting new ghosts
    //   grow local element and attribute tables if needed
    //   add to local elem[elemType] table, and update IDXL if needed
    //   update remote adjacencies
    //   update local adjacencies
  }
  else if(ghostcount > 0 && localcount > 0 && sharedcount == 0) { //this is an impossible case
    //bogus case
  }

  return newEl;
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
	if(i==0) { //lock myself
	  if(!existsChunk(idx)) {
	    lockedChunks.push_back(idx);
	  }
	}
	//which chunk does this belong to
	//add that chunk to the lock list, if it does not exist already.
	int numchunks;
	IDXL_Share **chunks1;
	mmod->fmUtil->getChunkNos(0,nodes[i],&numchunks,&chunks1);
	for(int j=0; j<numchunks; j++) {
	  if(!existsChunk(chunks1[j]->chk)) {
	    lockedChunks.push_back(chunks1[j]->chk);
	  }
	}
      }
      for(int i=0; i<numElems; i++) {
	//which chunk does this belong to
	//add that chunk to the lock list, if not already in it.
	int numchunks;
	IDXL_Share **chunks1;
	mmod->fmUtil->getChunkNos(1,elems[i],&numchunks,&chunks1);
	for(int j=0; j<numchunks; j++) {
	  if(!existsChunk(chunks1[j]->chk)) {
	    lockedChunks.push_back(chunks1[j]->chk);
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
      hasLocks = true;
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
      done = true;
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
	if(owner == idx) {
	  isOwner = true;
	  hasLocks = true;
	}
	else {
	  isOwner = false;
	}
	CkPrintf("Chunk %d locked by chunk %d\n",chunkNo, own);
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].lockRemoteChunk(imsg);
	if(ret->i != 1) return -1;
	else {
	  hasLocks = true;
	}
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
	CkPrintf("Chunk %d unlocked by chunk %d\n",chunkNo, own);
      }
      else {
	int2Msg *imsg = new int2Msg(chunkNo,own);
	ret = meshMod[chunkNo].unlockRemoteChunk(imsg);
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

void FEM_MUtil::getChunkNos(int entType, int entNo, int *numChunks, IDXL_Share ***chunks) {
  int type = 0; //0 - local, 1 - shared, 2 - ghost.

  if(entType == 0) { //nodes
    //only nodes can be shared
    if(FEM_Is_ghost_index(entNo)) type = 2;
    else if(isShared(entNo)) type = 1;
    else type = 0;

    if(type == 2) {
      //not very sure if it is any use ever to lock ghost nodes..
      //cannot think of a situation
      int ghostid = FEM_To_ghost_index(entNo);
      const IDXL_Rec *irec = mmod->fmMesh->node.getGhostRecv().getRec(ghostid);
      *numChunks = irec->getShared(); //check this value!!
      *chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = irec->getChk(i);
	int index = irec->getIdx(i);
	(*chunks)[i] = new IDXL_Share(chk, index);
      }
    }
    else if(type == 1) {
      const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(entNo);
      *numChunks = irec->getShared();
      *chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = irec->getChk(i);
	int index = irec->getIdx(i);
	(*chunks)[i] = new IDXL_Share(chk, index);
      }
    }
    else if(type == 0) {
      *numChunks = 1;
      (*chunks) = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
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
      const IDXL_Rec *irec = mmod->fmMesh->elem[0].getGhostRecv().getRec(ghostid);
      *numChunks = irec->getShared(); //should be 1
      *chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = irec->getChk(i);
	int index = irec->getIdx(i);
	(*chunks)[i] = new IDXL_Share(chk, index);
      }
    }
    else if(type == 0) {
      *numChunks = 1;
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
      //delete &ll; //clean up
      sharedNodeMsg *fm = new (2, 0) sharedNodeMsg;
      fm->chk = mmod->idx;
      fm->nBetween = nBetween;
      for(int j=0; j<nBetween; j++) {
	fm->between[j] = sharedIndices[j];
      }
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
  return;
}

int FEM_MUtil::exists_in_IDXL(FEM_Mesh *m, int localIdx, int chk) {
  int exists  = 0;
  const IDXL_Side &c = m->node.getGhost()->getGhostSend();
  const IDXL_Rec *tween = c.getRec(localIdx);
  for(int i=0; i<tween->getShared(); i++) {
    if(tween->getChk(i) == chk) {
      exists = 1;
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

void femMeshModify::removeSharedNodeRemote(removeSharedNodeMsg *fm) {
  fmUtil->removeNodeRemote(fmMesh, fm->chk, fm->index);
}

void femMeshModify::addGhostNode(addGhostNodeMsg *fm) {
}

void femMeshModify::addGhostElem(addGhostElemMsg *fm) {
}

#include "FEMMeshModify.def.h"
