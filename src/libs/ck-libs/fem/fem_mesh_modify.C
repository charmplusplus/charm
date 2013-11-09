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

CProxy_femMeshModify meshMod;

CDECL int FEM_add_node(int mesh, int* adjacent_nodes, int num_adjacent_nodes, int *chunks, int numChunks, int forceShared, int upcall){
  return FEM_add_node(FEM_Mesh_lookup(mesh,"FEM_add_node"), adjacent_nodes, num_adjacent_nodes, chunks, numChunks, forceShared, upcall);
}
CDECL void FEM_remove_node(int mesh,int node){
  return FEM_remove_node(FEM_Mesh_lookup(mesh,"FEM_remove_node"), node);
}
CDECL int FEM_remove_element(int mesh, int element, int elem_type, int permanent){
  return FEM_remove_element(FEM_Mesh_lookup(mesh,"FEM_remove_element"), element, elem_type, permanent);
}
CDECL int FEM_purge_element(int mesh, int element, int elem_type) {
  return FEM_purge_element(FEM_Mesh_lookup(mesh,"FEM_remove_element"), element, elem_type);
}
CDECL int FEM_add_element(int mesh, int* conn, int conn_size, int elem_type, int chunkNo){
  return FEM_add_element(FEM_Mesh_lookup(mesh,"FEM_add_element"), conn, conn_size, elem_type, chunkNo);
}
CDECL int FEM_Modify_Lock(int mesh, int* affectedNodes, int numAffectedNodes, int* affectedElts, int numAffectedElts, int elemtype){
  return FEM_Modify_Lock(FEM_Mesh_lookup(mesh,"FEM_Modify_Lock"), affectedNodes, numAffectedNodes, affectedElts, numAffectedElts, elemtype);
}
CDECL int FEM_Modify_Unlock(int mesh){
  return FEM_Modify_Unlock(FEM_Mesh_lookup(mesh,"FEM_Modify_Unlock"));
}
CDECL int FEM_Modify_LockN(int mesh, int nodeId, int readLock) {
  return FEM_Modify_LockN(FEM_Mesh_lookup(mesh,"FEM_Modify_LockN"),nodeId, readLock);
}
CDECL int FEM_Modify_UnlockN(int mesh, int nodeId, int readLock) {
  return FEM_Modify_UnlockN(FEM_Mesh_lookup(mesh,"FEM_Modify_UnlockN"),nodeId, readLock);
}


void FEM_Mesh_dataP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem,int length, int datatype,int width) {
  IDXL_Layout lo(datatype,width);
  FEM_Mesh_data_layoutP(fem_mesh,entity,attr,data,firstItem,length,lo);
}

void FEM_Mesh_data_layoutP(FEM_Mesh *fem_mesh,int entity,int attr,void *data, int firstItem,int length, IDXL_Layout_t layout) {
  const char *caller="FEM_Mesh_data_layout";
  FEM_Mesh_data_layoutP(fem_mesh,entity,attr,data,firstItem,length,
			IDXL_Layout_List::get().get(layout,caller));
}

void FEM_Mesh_data_layoutP(FEM_Mesh *m,int entity,int attr,void *data, int firstItem,int length, const IDXL_Layout &layout) {
  const char *caller="FEM_Mesh_data";
  //FEMAPI(caller);
  //FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
  FEM_Attribute *a=m->lookup(entity,caller)->lookup(attr,caller);
  
  if (m->isSetting()) 
    a->set(data,firstItem,length,layout,caller);
  else /* m->isGetting()*/
    a->get(data,firstItem,length,layout,caller);
}


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


CDECL void FEM_Print_n2n(int mesh, int nodeid){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  m->getfmMM()->getfmUtil()->FEM_Print_n2n(m, nodeid);
}

CDECL void FEM_Print_n2e(int mesh, int eid){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  m->getfmMM()->getfmUtil()->FEM_Print_n2e(m, eid);
}


// WARNING THESE TWO FUNCTIONS ONLY WORK ON TRIANGULAR ELEMENTS...
CDECL void FEM_Print_e2n(int mesh, int eid){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  m->getfmMM()->getfmUtil()->FEM_Print_e2n(m, eid);
}

CDECL void FEM_Print_e2e(int mesh, int eid){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  m->getfmMM()->getfmUtil()->FEM_Print_e2e(m, eid);
}


int FEM_add_node_local(FEM_Mesh *m, int addGhost){
  int newNode;
  if(addGhost){
    newNode = m->node.getGhost()->get_next_invalid(m,true,true); // find a place for new node in tables, reusing old invalid nodes if possible
    m->n2e_removeAll(FEM_To_ghost_index(newNode));    // initialize element adjacencies
    m->n2n_removeAll(FEM_To_ghost_index(newNode));    // initialize node adjacencies
  }
  else{
    newNode = m->node.get_next_invalid(m,true,false);
    m->n2e_removeAll(newNode);    // initialize element adjacencies
    m->n2n_removeAll(newNode);    // initialize node adjacencies
    //add a lock
    if(newNode >= m->getfmMM()->fmLockN.size()) {
      m->getfmMM()->fmLockN.push_back(new FEM_lockN(newNode,m->getfmMM()));
    }
    else {
      m->getfmMM()->fmLockN[newNode]->reset(newNode,m->getfmMM());
    }
  }
  return newNode;  // return a new index
}

//usually we know where to add this node.. use this info
int FEM_add_node(FEM_Mesh *m, int* adjacentNodes, int numAdjacentNodes, int*chunks, int numChunks, int forceShared, int upcall){
  // add local node
  //should be used only when all the adjacentnodes are shared but you know that the new node should not
  //be shared, but should belong to only one of the chunks sharing the face/edge.
  //usually it is safe to use -1, except in some weird cases.
  int index = m->getfmMM()->getIdx();

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(numChunks==1 && chunks[0]!=index) {
    //translate the indices.. the new node will be a ghost
    int chunkNo = chunks[0];
    addNodeMsg *am = new (numAdjacentNodes,numChunks,0) addNodeMsg;
    am->chk = index;
    am->nBetween = numAdjacentNodes;
    for(int i=0; i<numAdjacentNodes; i++) {
      am->between[i] = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,adjacentNodes[i],chunkNo,0);
      CkAssert(am->between[i]!=-1);
    }
    am->chunks[0] = chunkNo; //there should be only 1 chunk
    am->numChunks = numChunks;
    am->upcall = upcall;
    intMsg *imsg = new intMsg(-1);
    //m->getfmMM()->getfmUtil()->idxllock(m,chunkNo,2);
    //coarser lock
    m->getfmMM()->getfmUtil()->idxllock(m,chunkNo,0);
    imsg = meshMod[chunkNo].addNodeRemote(am);
    int newghost = FEM_add_node_local(m,1);
    m->node.ghost->ghostRecv.addNode(newghost,chunkNo);
    //m->getfmMM()->getfmUtil()->idxlunlock(m,chunkNo,2);
    //coarser lock
    m->getfmMM()->getfmUtil()->idxlunlock(m,chunkNo,0);
    //this is the ghostsend index on that chunk, translate it from ghostrecv idxl table
    //return FEM_To_ghost_index(m->getfmMM()->getfmUtil()->lookup_in_IDXL(m,imsg->i,chunkNo,2));
    //rather this is same as
    return FEM_To_ghost_index(newghost);
  }
  int newNode = FEM_add_node_local(m, 0);
  int sharedCount = 0;

  FEM_Interpolate *inp = m->getfmMM()->getfmInp();
  FEM_Interpolate::NodalArgs nm;
  nm.n = newNode;
  for(int i=0; i<numAdjacentNodes; i++) {
    nm.nodes[i] = adjacentNodes[i];
  }
  nm.frac = 0.5;
  nm.addNode = true;
  inp->FEM_InterpolateNodeOnEdge(nm);

  if(numChunks>1) {
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
    //just a sanity check, not required really
    if((sharedCount==numAdjacentNodes && numAdjacentNodes!=0) && forceShared!=-1 ) {
      //m->getfmMM()->getfmUtil()->splitEntityAll(m, newNode, numAdjacentNodes, adjacentNodes, 0);
      m->getfmMM()->getfmUtil()->splitEntitySharing(m, newNode, numAdjacentNodes, adjacentNodes, numChunks, chunks);
    }
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return newNode;
}


// The function called by the entry method on the remote chunk
void FEM_add_shared_node_remote(FEM_Mesh *m, int chk, int nBetween, int *between){
  // create local node
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int newnode = FEM_add_node_local(m, 0);
  
  // must negotiate the common IDXL number for the new node, 
  // and store it in appropriate IDXL tables

  //note that these are the shared indices, local indices need to be calculated
  m->getfmMM()->getfmUtil()->splitEntityRemote(m, chk, newnode, nBetween, between);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
}


void FEM_remove_node_local(FEM_Mesh *m, int node) {
  // if node is local:
  int numAdjNodes, numAdjElts;
  int *adjNodes, *adjElts;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  m->n2n_getAll(node, &adjNodes, &numAdjNodes);
  m->n2e_getAll(node, &adjElts, &numAdjElts);
  
  // mark node as deleted/invalid
  if(FEM_Is_ghost_index(node)){
    CkAssert((numAdjNodes==0) && (numAdjElts==0));
    //otherwise this ghost node is connected to some element in another chunk, which the chunk that just informed us doesn't know abt
    //look up the ghostrecv idxl list & clean up all instances
    const IDXL_Rec *irec = m->node.ghost->ghostRecv.getRec(FEM_To_ghost_index(node));
    //this ghost node might be sent from a chunk without any elems
    int size = 0;
    if(irec) size = irec->getShared();
    int *chunks1, *inds1;
    if(size>0) {
      chunks1 = new int[size];
      inds1 = new int[size];
    }
    for(int i=0; i<size; i++) {
      chunks1[i] = irec->getChk(i);
      inds1[i] = irec->getIdx(i);
    }
    int index = m->getfmMM()->getIdx();
    for(int i=0; i<size; i++) {
      int chk = chunks1[i];
      int sharedIdx = inds1[i];
      m->node.ghost->ghostRecv.removeNode(FEM_To_ghost_index(node),chk);
      meshMod[chk].removeIDXLRemote(index,sharedIdx,1);
    }
    if(size>0) {
      delete [] chunks1;
      delete [] inds1;
    }
    m->node.ghost->set_invalid(FEM_To_ghost_index(node),true);
  }
  else {
    //look it up on the idxl list and delete any instances in it
    const IDXL_Rec *irec = m->node.ghostSend.getRec(node);
    int size = 0;
    if(irec) size = irec->getShared();
    if(size > 0) {
      int *chknos = (int*)malloc(size*sizeof(int));
      int *sharedIndices = (int*)malloc(size*sizeof(int));
      for(int i=0; i<size; i++) {
	chknos[i] = irec->getChk(i);
	sharedIndices[i] = irec->getIdx(i);
      }
      for(int chkno=0; chkno<size; chkno++) {
	int remoteChunk = chknos[chkno];
	int sharedIdx = sharedIndices[chkno];
	//go to that remote chunk & delete this idxl list and this ghost node
	meshMod[remoteChunk].removeGhostNode(m->getfmMM()->getIdx(), sharedIdx);
	m->node.ghostSend.removeNode(node, remoteChunk);
      }
      free(chknos);
      free(sharedIndices);
    }
    
    /*if(!((numAdjNodes==0) && (numAdjElts==0))) {
      CkPrintf("Error: Node %d cannot be removed, it is connected to :\n",node);
      m->getfmMM()->fmUtil->FEM_Print_n2e(m,node);
      m->getfmMM()->fmUtil->FEM_Print_n2n(m,node);
      }*/
    CkAssert((numAdjNodes==0) && (numAdjElts==0)); // we shouldn't be removing a node away that is connected to something

    m->node.set_invalid(node,true);
    m->getfmMM()->fmLockN[node]->reset(node,m->getfmMM()); //should be done for the locked node only, i.e. the node on the smallest chunk no
  }
  if(numAdjNodes != 0) delete[] adjNodes;
  if(numAdjElts != 0) delete[] adjElts;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}


// remove a local or shared node, but NOT a ghost node
// Should probably be able to handle ghosts someday, but I cannot 
// remember the reasoning for not allowing them
void FEM_remove_node(FEM_Mesh *m, int node){

  CkAssert(node >= 0);

  //someone might actually want to remove a ghost node... when there is a ghost edge with both end nodes shared
  //we will have to intercept such a situation and call it on the remotechunk
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(FEM_Is_ghost_index(node)) {
    //this is interpreted as a chunk trying to delete this ghost node, this is just to get rid of its version of the node entity
  }
  
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

    // delete it on remote chunks(shared and ghost), update IDXL tables
    m->getfmMM()->getfmUtil()->removeNodeAll(m, node);
    
    // mark node as deleted/invalid locally
    //FEM_remove_node_local(m,node);
    //m->node.set_invalid(node,true);
    if(numAdjNodes!=0) delete[] adjNodes;
    if(numAdjElts!=0) delete[] adjElts;
  }
  else {
    FEM_remove_node_local(m,node);
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
}


// remove a local element from the adjacency tables as well as the element list
void FEM_remove_element_local(FEM_Mesh *m, int element, int etype){

  // replace this element with -1 in adjacent nodes' adjacencies
  const int nodesPerEl = m->elem[etype].getConn().width(); // should be the number of nodes that can be adjacent to this element
  int *adjnodes = new int[nodesPerEl];
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  m->e2n_getAll(element, adjnodes, etype);
  for(int i=0;i<nodesPerEl;i++) {
    //if(adjnodes[i] != -1) //if an element is local, then an adjacent node should not be -1
    m->n2e_remove(adjnodes[i],element);
  }
  
  // replace this element with -1 in adjacent elements' adjacencies
  const int numAdjElts = nodesPerEl;    // FIXME: hopefully there will be at most as many faces on an element as vertices
  int *adjelts = new int[numAdjElts]; 
  m->e2e_getAll(element, adjelts, etype);
  for(int i=0;i<numAdjElts;i++){
    m->e2e_replace(adjelts[i],element,-1);
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
  
  //done in purge now
  /*if(FEM_Is_ghost_index(element)){
    m->elem[etype].getGhost()->set_invalid(FEM_To_ghost_index(element),false);
  }
  else {
    m->elem[etype].set_invalid(element,false);
    }*/
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
 
  delete[] adjelts;
  delete[] adjnodes;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

// Can be called on local or ghost elements
int FEM_remove_element(FEM_Mesh *m, int elementid, int elemtype, int permanent){

  //CkAssert(elementid != -1);
  if(elementid == -1) return -1;
  int index = m->getfmMM()->getfmUtil()->getIdx();

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(FEM_Is_ghost_index(elementid)){
    //an element can come as a ghost from only one chunk, so just convert args and call it on that chunk
    int ghostid = FEM_To_ghost_index(elementid);
    const IDXL_Rec *irec = m->elem[elemtype].ghost->ghostRecv.getRec(ghostid);
    int size = irec->getShared();
    CkAssert(size == 1);
    int remoteChunk = irec->getChk(0);
    int sharedIdx = irec->getIdx(0);
    
    removeElemMsg *rm = new removeElemMsg;
    rm->chk = index;
    rm->elementid = sharedIdx;
    rm->elemtype = elemtype;
    rm->permanent = permanent;
    meshMod[remoteChunk].removeElementRemote(rm);
    //another possible deadlock

    // remove local ghost element
    //FEM_remove_element_local(m, elementid, elemtype);
    return remoteChunk;
  }
  else {
    //if this is a ghost element on some other chunks, then for all ghost nodes in this element,
    //we should individually go to each ghost and figure out if it has any connectivity in
    //the ghost layer after the removal. If not, then it will be removed from their ghost layers.
    //This is something similar to an add_element, where we do a similar analaysis on each 
    //involved chunk neighbor and add ghost nodes.
    
    //The bottomline is: A CHUNK CANOT OPERATE ON ITSELF AND GET RID OF ITS GHOSTS!!
    //If it operates on remote elements it sure can.
    
    //get a list of chunks for which this element is a ghost
    const IDXL_Rec *irec = m->elem[elemtype].ghostSend.getRec(elementid);
    if(irec){
      int numSharedChunks = irec->getShared();
      int connSize = m->elem[elemtype].getConn().width();
      if(numSharedChunks>0) {
	int *chknos = (int*)malloc(numSharedChunks*sizeof(int));
	int *inds = (int*)malloc(numSharedChunks*sizeof(int));
	for(int i=0; i<numSharedChunks; i++) {
	  chknos[i] = irec->getChk(i);
	  inds[i] = irec->getIdx(i);
	}
	int *newghost, *ghostidx, *losingThisNode, *nodes, *willBeGhost;
	if(permanent>=0) {
	  newghost = new int[connSize];
	  ghostidx = new int[connSize]; //need to create new ghosts
	  losingThisNode = new int[connSize];
	  nodes = new int[connSize];
	  willBeGhost = new int[connSize];
	  int *elems;
	  int numElems;
	  m->e2n_getAll(elementid,nodes,elemtype);
	  for(int i=0; i<connSize; i++) {
	    newghost[i] = -1;
	    ghostidx[i] = -1;
	    //am I losing this node?
	    losingThisNode[i] = 1;
	    m->n2e_getAll(nodes[i], &elems, &numElems);
	    for(int k=0; k<numElems; k++) {
	      if((!FEM_Is_ghost_index(elems[k])) && (elems[k]!=elementid)) {
		losingThisNode[i] = 0;
		break;
	      }
	    }
	    if(losingThisNode[i]) {
	      willBeGhost[i] = 1;
	    }
	    free(elems);
	  }
	  if(losingThisNode[0]==1 && losingThisNode[1]==1 && losingThisNode[2]==1) {
	    //if it is connected to the rest of the chunk, it will be a ghost
	    //otherwise, it will not even be a ghost
	    for(int i=0; i<connSize; i++) {
	      int *ndnbrs;
	      int numndnbrs;
	      m->n2n_getAll(nodes[i], &ndnbrs, &numndnbrs);
	      willBeGhost[i]=0;
	      for(int n1=0; n1<numndnbrs; n1++) {
		if(ndnbrs[n1]>=0 && ndnbrs[n1]!=nodes[(i+1)%connSize] && ndnbrs[n1]!=nodes[(i+2)%connSize]) {
		  willBeGhost[i]=1;
		}
	      }
	    }
	  }
	  for(int i=0; i<connSize; i++) {
	    if(losingThisNode[i]==1 && willBeGhost[i]==1) {
	      newghost[i] = FEM_add_node_local(m,1);
	      ghostidx[i] = FEM_To_ghost_index(newghost[i]);
	      //lock it on the min chunk other than index, correct this lock later
	      FEM_Modify_LockAll(m,nodes[i],false);
	    }
	  }
	}
	/*//a correction to deal with physical corners
	//if losing a node which is a corner, then upgrade that ghostnode on the chunk to a shared node
	//before performing this entire operation
	//I can only think of this happening for 1 chunk
	for(int i=0; i<connSize; i++) {
	  if(losingThisNode[i]==1 && m->node.shared.getRec(nodes[i]==NULL)) {
	  }
	  }*/
	for(int i=0; i<numSharedChunks; i++) {
#ifdef DEBUG 
	  CmiMemoryCheck(); 
#endif
	  bool lockedRemoteIDXL = false;
	  int chk = chknos[i];
	  int sharedIdx = inds[i];
	  int numGhostNodes = 0;
	  int *ghostIndices = (int*)malloc(connSize*sizeof(int));
	  int numGhostRN = 0;
	  CkVec<int> ghostRNIndices;
	  int numGhostRE = 0;
	  CkVec<int> ghostREIndices;
	  int numSharedNodes = 0;
	  int *sharedIndices = (int*)malloc(connSize*sizeof(int));
	  //purge will do this now
	  //m->elem[elemtype].ghostSend.removeNode(elementid, chk);
	  if(permanent>=0) {
	    //get the list of n2e for all nodes of this element. If any node has only this element in its list.
	    //it no longer should be a ghost on chk
	    //the next step is to figure out all the ghost elements & nodes that it no longer needs to have
	    CkVec<int> testelems;
	    const IDXL_List ll = m->elem[elemtype].ghostSend.addList(chk);
	    int size = ll.size();
	    const IDXL_List ln = m->node.ghostSend.addList(chk);
	    int sizeN = ln.size();
	    const IDXL_List lre = m->elem[elemtype].ghost->ghostRecv.addList(chk);
	    int lresize = lre.size();
	    const IDXL_List lrn = m->node.ghost->ghostRecv.addList(chk);
	    int lrnsize = lrn.size();
	  
	    for(int j=0; j<connSize; j++) {
	      int *elems;
	      int numElems;
#ifdef DEBUG 
	      CmiMemoryCheck(); 
#endif
	      m->n2e_getAll(nodes[j], &elems, &numElems);

	      if(chk != permanent) { //do not delete ghost nodes on the eating chunk
		//if any of these elems is a ghost on chk then do not delete this ghost node
		bool shouldBeDeleted = true;
		for(int k=0; k<numElems; k++) {
		  if(elems[k]==elementid) continue;
		  if(elems[k]>0) {
		    if(m->getfmMM()->fmUtil->exists_in_IDXL(m,elems[k],chk,3)!=-1) {
		      shouldBeDeleted = false;
		      break;
		    }
		  }
		}

		//find out what the other shared chunks think abt it
		//if every shared chunk believes it should be deleted, get rid of it
		if(shouldBeDeleted) {
		  const IDXL_Rec *irecsh = m->node.shared.getRec(nodes[j]);
		  if(irecsh!=NULL) {
		    for(int k=0; k<irecsh->getShared(); k++) {
		      if(shouldBeDeleted) {
			shouldBeDeleted = meshMod[irecsh->getChk(k)].shouldLoseGhost(index,irecsh->getIdx(k),chk)->b;
		      }
		    }
		  }
		}
		
		//add this to the list of ghost nodes to be deleted on the remote chunk
		if(shouldBeDeleted) {
		  //convert this local index to a shared index
		  int shidx = m->getfmMM()->fmUtil->exists_in_IDXL(m,nodes[j],chk,1);
		  if(shidx!=-1) {
		    m->node.ghostSend.removeNode(nodes[j], chk);
		    ghostIndices[numGhostNodes] = shidx;
		    numGhostNodes++;
		  }
		}
	      }

	      //if I am losing this node, then only will I lose some ghosts
	      if(losingThisNode[j]) {
		//was a shared node, but now it should be a ghost node
		for(int k=0; k<numElems; k++) {
		  int *nds = (int*)malloc(m->elem[elemtype].getConn().width()*sizeof(int));
		  if(FEM_Is_ghost_index(elems[k]) && m->getfmMM()->getfmUtil()->exists_in_IDXL(m,elems[k],chk,4,elemtype)!=-1) {
		    m->e2n_getAll(elems[k], nds, elemtype);
		    int geShouldBeDeleted = 1;
		    for(int l=0; l<connSize; l++) {
		      if(!FEM_Is_ghost_index(nds[l]) && (nodes[j]!=nds[l])) {
			if(losingThisNode[(j+1)%connSize]==1 && nds[l]==nodes[(j+1)%connSize]) continue;
			else if(losingThisNode[(j+2)%connSize]==1 && nds[l]==nodes[(j+2)%connSize]) continue;
			geShouldBeDeleted = 0;
		      }
		    }
		    if(geShouldBeDeleted) {
		      int sge = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,elems[k],chk,4,elemtype);
		      m->elem[elemtype].ghost->ghostRecv.removeNode(FEM_To_ghost_index(elems[k]), chk);
		      FEM_remove_element_local(m,elems[k],elemtype);
		      m->elem[elemtype].ghost->set_invalid(FEM_From_ghost_index(elems[k]),false);
		      ghostREIndices.push_back(sge);
		      numGhostRE++;

		      //find out if any nodes need to be removed
		      for(int l=0; l<connSize; l++) {
			int *elts;
			int numElts;
#ifdef DEBUG 
			CmiMemoryCheck(); 
#endif
			m->n2e_getAll(nds[l], &elts, &numElts);
			//if this is no longer connected to this chunk
			if(nds[l]<-1) { //it is a ghost
			  bool removeflag = true;
			  for(int lm=0; lm<numElts; lm++) {
			    //if(elts[lm]>=0 || m->getfmMM()->getfmUtil()->exists_in_IDXL(m,elts[lm],chk,4)>=0)
			    if(elts[lm]!=elems[k]) {
			      removeflag = false;
			    }
			  }
			  if(removeflag) {
			    //remove this ghost node on this chunk
			    int sgn = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nds[l],chk,2);
			    m->node.ghost->ghostRecv.removeNode(FEM_To_ghost_index(nds[l]), chk);
			    if(numElts==0) FEM_remove_node_local(m,nds[l]);
			    ghostRNIndices.push_back(sgn);
			    numGhostRN++;
			  }
			}
			if(numElts!=0) delete[] elts;
		      }
		    }
		  }
		  free(nds);
		}
		//if it is shared, tell that chunk, it no longer exists on me
		int ssn = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodes[j],chk,0);
		//m->getfmMM()->getfmUtil()->idxllock(m,chk,2);
		//add a new coarse idxl lock, should lock a remote idxl only once
		//even if it loses more than one nodes
		if(!lockedRemoteIDXL) m->getfmMM()->getfmUtil()->idxllock(m,chk,0);
	        lockedRemoteIDXL = true;
	        bool losingacorner = false;
		/*if(ssn==-1) {
		  losingacorner=m->getfmMM()->fmAdapt->isFixedNode(nodes[j]);
		  //this is a special case, when it is a physical corner
		  if(losingacorner) {
		    ssn = -1000000000;
		    ssn += m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodes[j],chk,1);
		    //ghostsend remove will be done in addtosharedlist
		  }
		  }*/
		if(ssn!=-1) {
		  if(willBeGhost[j]==1) {
		    m->node.ghost->ghostRecv.addNode(newghost[j],chk);
		  }
		  else {
		    ssn -= 500000000; //to tell the other chunk, not to add it in ghostsend
		  }
		  if(!losingacorner) {
		    m->node.shared.removeNode(nodes[j], chk);
		  }
		  sharedIndices[numSharedNodes] = ssn;
		  numSharedNodes++;
		}
	        if(i==numSharedChunks-1) {
		  //add only to the elems which still exist
		  for(int k=0; k<numElems; k++) {
		    bool flagElem = false;
		    if(FEM_Is_ghost_index(elems[k])) {
		      if(m->elem[0].ghost->is_valid(FEM_From_ghost_index(elems[k]))) flagElem = true;
		    }
		    else {
		      if(m->elem[0].is_valid(elems[k])) flagElem = true;
		    }
		    if(flagElem) {
		      m->e2n_replace(elems[k],nodes[j],ghostidx[j],elemtype);
		      m->n2e_remove(nodes[j],elems[k]);
		      m->n2e_add(ghostidx[j],elems[k]);
		      testelems.push_back(elems[k]);
		    }
		  }
		  int *n2ns;
		  int numn2ns;
		  m->n2n_getAll(nodes[j],&n2ns,&numn2ns);
		  for(int k=0; k<numn2ns; k++) {
		    m->n2n_replace(n2ns[k],nodes[j],ghostidx[j]);
		    m->n2n_remove(nodes[j],n2ns[k]);
		    m->n2n_add(ghostidx[j],n2ns[k]);
		  }
		  if(numn2ns!=0) delete[] n2ns;
		  const IDXL_Rec *irec1 = m->node.shared.getRec(nodes[j]);
		  if(!irec1) {
		    if(!losingacorner) FEM_remove_node_local(m,nodes[j]);
		    //else it is a corner, we do not want to remove it until the other chunk has added it
		    //it is the responsibility of the other chunk to delete this
		  }
		}
	      }
	      if(numElems!=0) delete[] elems;
	    }
	    //now that all ghost nodes to be removed have been decided, we add the elem & call the entry method
	    //test if all the elems added are valid
	    for(int testelemsc=0; testelemsc <testelems.size(); testelemsc++) {
	      int el = testelems[testelemsc];
	      if(FEM_Is_ghost_index(el)) {
		CkAssert(m->elem[0].ghost->is_valid(FEM_From_ghost_index(el))==1);
	      }
	      else {
		CkAssert(m->elem[0].is_valid(el)==1);
	      }
	    }
	  }
#ifdef DEBUG 
	  CmiMemoryCheck(); 
#endif
	  removeGhostElemMsg *rm = new (numGhostNodes, numGhostRN, numGhostRE, numSharedNodes, 0) removeGhostElemMsg;
	  rm->chk = index;
	  rm->elemtype = elemtype;
	  rm->elementid = sharedIdx;
	  rm->numGhostIndex = numGhostNodes;
	  for(int j=0; j<numGhostNodes; j++) {
	    rm->ghostIndices[j] = ghostIndices[j];
	  }
	  rm->numGhostRNIndex = numGhostRN;
	  for(int j=0; j<numGhostRN; j++) {
	    rm->ghostRNIndices[j] = ghostRNIndices[j];
	  }
	  rm->numGhostREIndex = numGhostRE;
	  for(int j=0; j<numGhostRE; j++) {
	    rm->ghostREIndices[j] = ghostREIndices[j];
	  }
	  rm->numSharedIndex = numSharedNodes;
	  for(int j=0; j<numSharedNodes; j++) {
	    rm->sharedIndices[j] = sharedIndices[j];
	  }
	  meshMod[chk].removeGhostElem(rm);  //update the ghosts on all shared chunks
	  //if(lockedRemoteIDXL) m->getfmMM()->getfmUtil()->idxlunlock(m,chk,2);
	  if(lockedRemoteIDXL) m->getfmMM()->getfmUtil()->idxlunlock(m,chk,0); //unlock this coarse lock
	  free(ghostIndices);
	  free(sharedIndices);
	}
	free(chknos);
	free(inds);
	if(permanent>=0) {
	  delete [] newghost;
	  delete [] ghostidx;
	  delete [] losingThisNode;
	  delete [] nodes;
	  delete [] willBeGhost;
	}
      }
    }
    // remove local element
    FEM_remove_element_local(m, elementid, elemtype);
#ifdef DEBUG 
    CmiMemoryCheck(); 
#endif
  }
  return index;
}

void FEM_remove_element_remote(FEM_Mesh *m, int element, int elemtype){
  // remove local element from elem[elemType] table
}

int FEM_purge_element(FEM_Mesh *m, int elementid, int elemtype) {
  if(elementid==-1) return 1;
  int index = m->getfmMM()->idx;
  if(FEM_Is_ghost_index(elementid)) {
    const IDXL_Rec *irec1 = m->elem[elemtype].ghost->ghostRecv.getRec(FEM_To_ghost_index(elementid));
    int remotechk = irec1->getChk(0);
    int sharedIdx = irec1->getIdx(0);
    meshMod[remotechk].purgeElement(index,sharedIdx);
  }
  else {
    const IDXL_Rec *irec = m->elem[elemtype].ghostSend.getRec(elementid);
    if(irec){
      int numSharedChunks = irec->getShared();
      int connSize = m->elem[elemtype].getConn().width();
      int *chknos, *inds;
      if(numSharedChunks>0) {
	chknos = (int*)malloc(numSharedChunks*sizeof(int));
	inds = (int*)malloc(numSharedChunks*sizeof(int));
	for(int i=0; i<numSharedChunks; i++) {
	  chknos[i] = irec->getChk(i);
	  inds[i] = irec->getIdx(i);
	}
      }
      for(int i=0; i<numSharedChunks; i++) {
	meshMod[chknos[i]].cleanupIDXL(index,inds[i]);
	m->elem[elemtype].ghostSend.removeNode(elementid, chknos[i]);
      }
      if(numSharedChunks>0) {
	free(chknos);
	free(inds);
      }
    }
  }
  // delete element by marking invalid
  if(!FEM_Is_ghost_index(elementid)){
    m->elem[elemtype].set_invalid(elementid,false);
  }
  return 1;
}


// A helper function for FEM_add_element_local below
// Will only work with the same element type as the one given, may crash otherwise
void update_new_element_e2e(FEM_Mesh *m, int newEl, int elemType){
  CkAssert(elemType==0); // this function most definitely will not yet work with mixed element types.

  // Create tuple table
  FEM_ElemAdj_Layer *g = m->getElemAdjLayer();
  CkAssert(g->initialized);
  const int nodesPerTuple = g->nodesPerTuple;
  const int tuplesPerElem = g->elem[elemType].tuplesPerElem;
  tupleTable table(nodesPerTuple);
  FEM_Symmetries_t allSym;


  // locate all elements adjacent to the nodes adjacent to the new 
  // element, including ghosts, and the new element itself
  const int nodesPerElem = m->elem[elemType].getNodesPer();
  int *adjnodes = new int[nodesPerElem];
  CkVec<int> elist;
  m->e2n_getAll(newEl, adjnodes, elemType);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  for(int i=0;i<nodesPerElem;i++){
    int sz;
    int *adjelements=0;
    m->n2e_getAll(adjnodes[i], &adjelements, &sz);
    for(int j=0;j<sz;j++){
      int found=0;
      // only insert if it is not already in the list
      for(int i=0;i<elist.length();i++)// we use a slow linear scan of the CkVec
	if(elist[i] == adjelements[j])
	  found=1;
      if(!found){
	elist.push_back(adjelements[j]);
      }
    }
    if(sz!=0) delete[] adjelements;
  }
  delete[] adjnodes;
  

  // Add all the potentially adjacent elements to the tuple table
  for(int i=0;i<elist.length();i++){
    int nextElem = elist[i];
    int tuple[tupleTable::MAX_TUPLE];
    int *conn;
    if(FEM_Is_ghost_index(nextElem))
      conn=((FEM_Elem*)m->elem[elemType].getGhost())->connFor(FEM_To_ghost_index(nextElem));
    else
      conn=m->elem[elemType].connFor(nextElem);
    for (int u=0;u<tuplesPerElem;u++) {
      for (int i=0;i<nodesPerTuple;i++) {
	int eidx=g->elem[elemType].elem2tuple[i+u*g->nodesPerTuple];
	if (eidx==-1)  //"not-there" node--
	  tuple[i]=-1; //Don't map via connectivity
	else           //Ordinary node
	  tuple[i]=conn[eidx]; 
      }
      table.addTuple(tuple,new elemList(0,nextElem,elemType,allSym,u)); 
    }
  }


  
  // extract true adjacencies from table and update all e2e tables for both newEl and the others
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
        for (const elemList *b=a->next;b!=NULL;b=b->next){
          if((a->localNo != b->localNo) || (a->type != b->type)){ // if a and b are different elements
	    //	CkPrintf("%d:%d:%d adj to %d:%d:%d\n", a->type, a->localNo, a->tupleNo, b->type, b->localNo, b->tupleNo);
	    // Put b in a's adjacency list
	    if(FEM_Is_ghost_index(a->localNo)){
	      const int j = FEM_To_ghost_index(a->localNo)*tuplesPerElem + a->tupleNo;
	      adjsGhost[j] = b->localNo;
	      adjTypesGhost[j] = b->type;
	    }
	    else{
	      const int j= a->localNo*tuplesPerElem + a->tupleNo;
	      adjs[j] = b->localNo;
	      adjTypes[j] = b->type;
	    }
			
	    // Put a in b's adjacency list
	    if(FEM_Is_ghost_index(b->localNo)){
	      const int j = FEM_To_ghost_index(b->localNo)*tuplesPerElem + b->tupleNo;
	      adjsGhost[j] = a->localNo;
	      adjTypesGhost[j] = a->type;
	    }
	    else{
	      const int j= b->localNo*tuplesPerElem + b->tupleNo;
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
int FEM_add_element_local(FEM_Mesh *m, const int *conn, int connSize, int elemType, int addGhost){
  // lengthen element attributes
  int newEl;
  if(addGhost){
    newEl = m->elem[elemType].getGhost()->get_next_invalid(m,false,true); // find a place in the array for the new el
    ((FEM_Elem*)m->elem[elemType].getGhost())->connIs(newEl,conn);// update element's conn, i.e. e2n table
    newEl = FEM_From_ghost_index(newEl); // return the signed ghost value
  }
  else{
    newEl = m->elem[elemType].get_next_invalid(m,false,false);
    m->elem[elemType].connIs(newEl,conn);  // update element's conn, i.e. e2n table
  }
  
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  // add to corresponding inverse, the n2e and n2n table
  for(int i=0;i<connSize;i++){
    m->n2e_add(conn[i],newEl);
    for(int j=i+1;j<connSize;j++){
      if(! m->n2n_exists(conn[i],conn[j]))
        m->n2n_add(conn[i],conn[j]);
      if(! m->n2n_exists(conn[j],conn[i]))
        m->n2n_add(conn[j],conn[i]);
    }
  }

  // update e2e table -- too complicated, so it gets is own function
  m->e2e_removeAll(newEl);
  update_new_element_e2e(m,newEl,elemType);

  int *adjes = new int[connSize];
  m->e2e_getAll(newEl, adjes, 0);
  CkAssert(!((adjes[0]==adjes[1] && adjes[0]!=-1) || (adjes[1]==adjes[2] && adjes[1]!=-1) || (adjes[2]==adjes[0] && adjes[2]!=-1)));
  delete[] adjes;
  return newEl;
}


int FEM_add_element(FEM_Mesh *m, int* conn, int connSize, int elemType, int chunkNo){
  int newEl = -1;
  int index = m->getfmMM()->getIdx();
  int buildGhosts = 0;
  int sharedcount=0;
  int ghostcount=0;
  int localcount=0;
  CkAssert(conn[0]!=conn[1] &&conn[1]!=conn[2] &&conn[2]!=conn[0]);
  int *nodetype = (int *)malloc(connSize *sizeof(int)); //0 -- local, 1 -- shared, 2--ghost
  for(int i=0;i<connSize;i++){
    CkAssert(conn[i]!=-1);
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
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  localcount = connSize - (sharedcount + ghostcount);
  if(sharedcount==0 && ghostcount==0){// add a local elem with all local nodes
    newEl = FEM_add_element_local(m,conn,connSize,elemType,0);
    //no modifications required for ghostsend or ghostrecv of nodes or elements
  }
  else if(ghostcount==0 && sharedcount > 0 && localcount > 0){// a local elem with not ALL shared nodes
    newEl = FEM_add_element_local(m,conn,connSize,elemType,0);
    buildGhosts = 1;
  }
  else if(ghostcount==0 && sharedcount > 0 && localcount == 0){// a local elem with ALL shared nodes
    //it is interesting to note that such a situation can only occur between only two chunks
    //in any number of dimensions.
    //So, the solution is to figure out, where it belongs to...
    if(!(chunkNo==index || chunkNo==-1)) {
      addElemMsg *am = new (connSize, 0, 0) addElemMsg;
      int chk = index;
      am->chk = chk;
      am->elemtype = elemType;
      am->connSize = connSize;
      am->numGhostIndex = 0;
      for(int i=0; i<connSize; i++) {
	CkAssert(nodetype[i] == 1);
	am->conn[i] = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[i],chunkNo,0);
	CkAssert(am->conn[i] >= 0);
      }
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
      meshMod[chunkNo].addElementRemote(am);
      //this might be a source of a deadlock, because I am sure the remote chunk will call a sync entry method on this chunk for updating my ghosts....

      //pick up the last entry from the element ghostrecv IDXL with remoteChunk, that is the index of the last element added.
      const IDXL_List ilist = m->elem[elemType].ghost->ghostRecv.addList(chunkNo);
      int size = ilist.size();
      newEl = ilist[size-1];
      newEl = FEM_To_ghost_index(newEl);
      free(nodetype);
      return newEl;
    }
    newEl = FEM_add_element_local(m,conn,connSize,elemType,0);
    buildGhosts = 1;
  }
  //I will assume that this is a correct call.. the caller never gives me junk ghost elements
  else if(ghostcount > 0 /*&& localcount == 0 && sharedcount > 0*/) { // it is remote elem with some shared nodes
    //this is the part when it eats a ghost element
    if((chunkNo!=-1) && (chunkNo==index)) { //this is because the chunk doing this operation is supposed to eat into anyone else.
      //change all ghost nodes to shared nodes
      //this also involves going to allchunks that it was local/shared on and make it shared and add this chunk to the list of chunks it is shared to.
      for(int i=0; i<connSize; i++) {
	if(nodetype[i]==2) {
	  //build up the list of chunks it is shared/local to
#ifdef DEBUG 
	  CmiMemoryCheck(); 
#endif
	  int numchunks;
	  IDXL_Share **chunks1;
	  m->getfmMM()->getfmUtil()->getChunkNos(0,conn[i],&numchunks,&chunks1);
	  //add a new node with the same attributes as this ghost node, do not remove the ghost node yet
	  int newN = m->getfmMM()->getfmUtil()->Replace_node_local(m, conn[i], -1);
	  //add index to the shared list of this node on all the chunks
	  for(int j=0; j<numchunks; j++) {
	    int chk = chunks1[j]->chk;
	    if(chk==index) continue;
	    //m->getfmMM()->getfmUtil()->idxllock(m, chk, 0);
	    //coarser lock
	    m->getfmMM()->getfmUtil()->idxllock(m,chk,0);
	    m->node.shared.addNode(newN,chk);
	    //find out what chk calls this node (from the ghostrecv idxl list)
	    int idx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m, conn[i],  chk, 2);
	    m->node.ghost->ghostRecv.removeNode(FEM_To_ghost_index(conn[i]), chk);
	    meshMod[chk].addToSharedList(index, idx); //when doing this add, make all idxl lists consistent, and return me the list of n2e connections for this node, which are local to this chunk, and also a list of the connectivity
	    //m->getfmMM()->getfmUtil()->idxlunlock(m, chk, 0);
	    //coarser lock
	    m->getfmMM()->getfmUtil()->idxlunlock(m, chk, 0);
	  }
	  nodetype[i] = 1;
	  sharedcount++;
	  ghostcount--;
	  //remove the ghost node
	  FEM_remove_node_local(m,conn[i]);
	  conn[i] = newN;
	  //lock the newly formed node, if it needs to be
	  //FEM_Modify_LockUpdate(m,conn[i]);
	  for(int j=0; j<numchunks; j++) {
	    delete chunks1[j];
	  }
	  if(numchunks != 0) free(chunks1);
	}
      }
      newEl = FEM_add_element_local(m,conn,connSize,elemType,0);
      buildGhosts = 1;
    }
    else {
      int numSharedChunks = 0;
      int remoteChunk = -1;
      if(chunkNo==-1) {
	CkAssert(false); //shouldn't be here
	//figure out which chunk it is local to
	//among all chunks who share some of the nodes or from whom this chunk receives ghost nodes
	//if there is any chunk which owns a ghost node which is not shared, then that is a local node 
	//to that chunk and that chunk owns that element. However, only that chunk knows abt it.
	//So, just go to the owner of every ghost node and figure out who all share that node.
	//Build up this table of nodes owned by which all chunks.
	//The chunk that is in the table corresponding to all nodes wins the element.
	CkVec<int> **allShared;
	CkVec<int> *allChunks;
	int **sharedConn; 
	m->getfmMM()->getfmUtil()->buildChunkToNodeTable(nodetype, sharedcount, ghostcount, localcount, conn, connSize, &allShared, &numSharedChunks, &allChunks, &sharedConn);   
	//we are looking for a chunk which does not have a ghost node
	for(int i=0; i<numSharedChunks; i++) {
#ifdef DEBUG 
	  CmiMemoryCheck(); 
#endif
	  remoteChunk = i;
	  for(int j=0; j<connSize; j++) {
	    if(sharedConn[i][j] == -1 || sharedConn[i][j] == 2) {
	      remoteChunk = -1;
	      break; //this chunk has a ghost node
	    }
	    //if(sharedConn[i][j] == 0) {
	    //  break; //this is a local node, hence it is the remotechunk
	    //}
	  }
	  if(remoteChunk == i) break;
	  else remoteChunk = -1;
	}
	if(remoteChunk==-1) {
	  //every chunk has a ghost node.
	  if(chunkNo != -1) {
	    //upgrade the ghost node to a shared node on chunkNo
	    remoteChunk = chunkNo;
	    for(int k=0; k<numSharedChunks; k++) {
	      if(chunkNo == (*allChunks)[k]) {
		for(int l=0; l<connSize; l++) {
		  if(sharedConn[k][l]==-1 || sharedConn[k][l]==2) {
		    //FIXME: upgrade this ghost node
		  }
		}
	      }
	    }
	  }
	  else {
	    CkPrintf("[%d]ERROR: Can not derive where (%d,%d,%d) belongs to\n",index,conn[0],conn[1],conn[2]);
	    CkAssert(false);
	  }
	}
	remoteChunk = (*allChunks)[remoteChunk];
	//convert all connections to the shared IDXL indices. We should also tell which are ghost indices
	CkPrintf("[%d]Error: I derived it should go to chunk %d, which is not %d\n",index,remoteChunk,chunkNo);
	for(int k=0; k<numSharedChunks; k++) {
	  free(sharedConn[k]);
	}
	if(numSharedChunks!=0) free(sharedConn);
	for(int k=0; k<connSize; k++) {
	  delete allShared[k];
	}
	free(allShared);
      }
      else {
	remoteChunk=chunkNo;
      }
      int numGhostNodes = 0;
      for(int i=0; i<connSize; i++) {
	if(nodetype[i] == 2) { //a ghost node
	  numGhostNodes++;
	}
      }
      CkAssert(numGhostNodes > 0);
      addElemMsg *am = new (connSize, numGhostNodes, 0) addElemMsg;
      int chk = index;
      am->chk = chk;
      am->elemtype = elemType;
      am->connSize = connSize;
      am->numGhostIndex = numGhostNodes;
      int j = 0;
      for(int i=0; i<connSize; i++) {
#ifdef DEBUG 
	CmiMemoryCheck(); 
#endif
	if(nodetype[i] == 1) {
	  am->conn[i] = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[i],remoteChunk,0);
	  CkAssert(am->conn[i] >= 0);
	}
	else if(nodetype[i] == 2) {
	  am->conn[i] = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[i],remoteChunk,2);
	  CkAssert(am->conn[i] >= 0);
	  am->ghostIndices[j] = i;
	  j++;
	}
      }
      meshMod[remoteChunk].addElementRemote(am);
      //this might be a source of a deadlock, because I am sure the remote chunk will call a sync entry method on this chunk for updating my ghosts....
      
      //pick up the last entry from the element ghostrecv IDXL with remoteChunk, that is the index of the last element added.
      const IDXL_List ilist = m->elem[elemType].ghost->ghostRecv.addList(remoteChunk);
      int size = ilist.size();
      newEl = ilist[size-1];
      newEl = FEM_To_ghost_index(newEl);
    }
  }
  else if(ghostcount > 0 && localcount == 0 && sharedcount == 0) { // it is a remote elem with no shared nodes
    //I guess such a situation will never occur
    //figure out which chunk it is local to -- this would be really difficult to do in such a case.
    //so, for now, we do not allow a chunk to add an element for which it does not share even a single node.

    // almost the code in the preceeding else cacse will work for this, but its better not to allow this
  }
  else if(ghostcount > 0 && localcount > 0 && sharedcount > 0){// it is a flip operation
    //actually this can be generalized as an operation which moves the boundary, to make the ghost node shared
    //if one uses FEM_elem_acquire, then this condition gets distributed across other conditions already done.
   
    //   promote ghosts to shared on others, requesting new ghosts
    //   grow local element and attribute tables if needed
    //   add to local elem[elemType] table, and update IDXL if needed
    //   update remote adjacencies
    //   update local adjacencies
  }
  else if(ghostcount > 0 && localcount > 0 && sharedcount == 0) { //this is an impossible case
    //bogus case
  }

  if(buildGhosts==1) {
    //   make this element ghost on all others, updating all IDXL's
    //   also in same remote entry method, update adjacencies on all others
    //   grow local element and attribute tables if needed
    //   add to local elem[elemType] table, and update IDXL if needed
    //   update local adjacencies
    //   return the new element id

    //build a mapping of all shared chunks to all nodes in this element    
    CkVec<int> **allShared;
    int numSharedChunks = 0;
    CkVec<int> *allChunks;
    int **sharedConn; 
    m->getfmMM()->getfmUtil()->buildChunkToNodeTable(nodetype, sharedcount, ghostcount, localcount, conn, connSize, &allShared, &numSharedChunks, &allChunks, &sharedConn);   
    //add all the local nodes in this element to the ghost list, if they did not exist already
    for(int i=0; i<numSharedChunks; i++) {
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
      int chk = (*allChunks)[i];
      if(chk == index) continue; //it is this chunk
      //it is a new element so it could not have existed as a ghost on that chunk. Just add it.
      //m->getfmMM()->getfmUtil()->idxllock(m, chk, 3);
      //m->getfmMM()->getfmUtil()->idxllock(m, chk, 1);
      //coarser lock
      m->getfmMM()->getfmUtil()->idxllock(m, chk, 0);
      m->elem[elemType].ghostSend.addNode(newEl,chk);
      //ghost nodes should be added only if they were not already present as ghosts on that chunk.
      int numNodesToAdd = 0;
      int numSharedGhosts = 0;
      int numSharedNodes = 0;
      int *sharedGhosts = (int *)malloc((connSize-1)*sizeof(int));
      int *sharedNodes = (int *)malloc((connSize)*sizeof(int));
      int *nodesToAdd = (int *)malloc((connSize)*sizeof(int));
      for(int j=0; j<connSize; j++) {
	int sharedNode = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[j],chk,0);
	if(sharedNode == -1) {
	  //node 'j' is a ghost on chunk 'i'
	  int sharedGhost = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[j],chk,1);
	  if( sharedGhost == -1) {
	    //this might be a new ghost, figure out if any of the chunks sharing
	    //this node has created this as a ghost on 'chk'
	    const IDXL_Rec *irec = m->node.shared.getRec(conn[j]);
	    if(irec) {
	      int noShared = irec->getShared();
	      for(int sharedck=0; sharedck<noShared; sharedck++) {
		int ckshared = irec->getChk(sharedck);
		int idxshared = irec->getIdx(sharedck);
		if(ckshared == chk) continue;
		CkAssert(chk!=index && chk!=ckshared && ckshared!=index);
		int idxghostsend = meshMod[ckshared].getIdxGhostSend(index,idxshared,chk)->i;
		if(idxghostsend != -1) {
		  m->node.ghostSend.addNode(conn[j],chk);
		  meshMod[chk].updateIdxlList(index,idxghostsend,ckshared);
		  sharedGhost = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[j],chk,1);
		  CkAssert(sharedGhost != -1);
		  break; //found a chunk that sends it out, update my tables
		}
		//Chunk 'ckshared' does not send this to Chunk 'chk' as ghost
	      }
	    }
	    //else it is a new ghost
	  }
	  if(sharedGhost == -1) {
	    nodesToAdd[numNodesToAdd] = conn[j];
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
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
      addGhostElemMsg *fm = new (numSharedGhosts, numSharedNodes, 0)addGhostElemMsg;
      fm->chk = index;
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
	m->node.ghostSend.addNode(nodesToAdd[j],chk);
      }
      free(nodesToAdd);
      fm->connSize = connSize;
      meshMod[chk].addGhostElem(fm); //newEl, m->fmMM->idx, elemType;
      //m->getfmMM()->getfmUtil()->idxlunlock(m, chk, 3);
      //m->getfmMM()->getfmUtil()->idxlunlock(m, chk, 1);
      //coarser lock
      m->getfmMM()->getfmUtil()->idxlunlock(m, chk, 0);
      free(sharedGhosts);
      free(sharedNodes);
    }
    for(int k=0; k<numSharedChunks; k++) {
      free(sharedConn[k]);
    }
    if(numSharedChunks!=0) free(sharedConn);
    for(int k=0; k<connSize; k++) {
      delete allShared[k];
    }
    free(allShared);
  }

  free(nodetype);
  return newEl;
}


int FEM_add_element_remote(){
  // promote ghosts to shared

  // find new ghosts for remote calling chunk by looking at new shared nodes
  // send these new ghosts to the remote calling chunk.

  // update my adjacencies

  return 0;
}


int FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes, int numAffectedNodes, int* affectedElts, int numAffectedElts, int elemtype) {
  return m->getfmMM()->getfmLock()->lock(numAffectedNodes, affectedNodes, numAffectedElts, affectedElts, elemtype);
}

int FEM_Modify_Unlock(FEM_Mesh *m) {
  return m->getfmMM()->getfmLock()->unlock();
}

int FEM_Modify_LockN(FEM_Mesh *m, int nodeId, int readlock) {
  int ret = -1;
  if(is_shared(m,nodeId)) {
    //find the index of the least chunk it is shared on and try to lock it
    int index = m->getfmMM()->getIdx();
    int numchunks;
    IDXL_Share **chunks1;
    m->getfmMM()->getfmUtil()->getChunkNos(0,nodeId,&numchunks,&chunks1);
    int minChunk = MAX_CHUNK;
    for(int j=0; j<numchunks; j++) {
      int pchk = chunks1[j]->chk;
      if(pchk < minChunk) minChunk = pchk;
    }
    for(int j=0; j<numchunks; j++) {
      delete chunks1[j];
    }
    if(numchunks!=0) free(chunks1);
    CkAssert(minChunk!=MAX_CHUNK);
    if(minChunk==index) {
      if(readlock) {
	ret = m->getfmMM()->getfmLockN(nodeId)->rlock();
      } else {
	ret = m->getfmMM()->getfmLockN(nodeId)->wlock(index);
      }
      if(ret==1) {
	m->getfmMM()->getfmLockN(nodeId)->verifyLock();
      }
      return ret;
    }
    else {
      CkAssert(minChunk!=MAX_CHUNK);
      int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,minChunk,0);
      if(sharedIdx < 0) return -1;
      if(readlock) {
	ret = meshMod[minChunk].lockRemoteNode(sharedIdx, index, 0, 1)->i;
      } else {
	ret = meshMod[minChunk].lockRemoteNode(sharedIdx, index, 0, 0)->i;
      }
      if(ret==1) {
	meshMod[minChunk].verifyLock(index,sharedIdx,0);
      }
      return ret;
    }
  }
  else if(FEM_Is_ghost_index(nodeId)) {
    //find the index of the least chunk that owns it & try to lock it
#ifdef DEBUG 
    CmiMemoryCheck(); 
#endif
    int index = m->getfmMM()->getIdx();
    int numchunks;
    IDXL_Share **chunks1;
    m->getfmMM()->getfmUtil()->getChunkNos(0,nodeId,&numchunks,&chunks1);
    int minChunk = MAX_CHUNK;
    for(int j=0; j<numchunks; j++) {
      int pchk = chunks1[j]->chk;
      if(pchk == index) continue;
      if(pchk < minChunk) minChunk = pchk;
    }
    for(int j=0; j<numchunks; j++) {
      delete chunks1[j];
    }
    if(numchunks!=0) free(chunks1);
    if(minChunk==MAX_CHUNK) return -1;
    int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,minChunk,2);
    if(sharedIdx < 0) return -1;
    if(readlock) {
      ret = meshMod[minChunk].lockRemoteNode(sharedIdx, index, 1, 1)->i;
    } else {
      ret = meshMod[minChunk].lockRemoteNode(sharedIdx, index, 1, 0)->i;
    }
    /*if(ret==1) {
      if(nodeId==-8 && index==4) {
	CkPrintf("Locking node %d on chunk %d\n",nodeId, minChunk);
      }
      meshMod[minChunk].verifyLock(index, sharedIdx, 1);
      }*/
    return ret;
  }
  else {
    if(readlock) {
      ret = m->getfmMM()->getfmLockN(nodeId)->rlock();
    } else {
      int index = m->getfmMM()->getIdx();
      ret = m->getfmMM()->getfmLockN(nodeId)->wlock(index);
    }
    /*if(ret==1) {
      m->getfmMM()->getfmLockN(nodeId)->verifyLock();
      }*/
    return ret;
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return -1; //should not reach here
}

int FEM_Modify_UnlockN(FEM_Mesh *m, int nodeId, int readlock) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  if(is_shared(m,nodeId)) {
    //find the index of the least chunk it is shared on and try to unlock it
    int index = m->getfmMM()->getIdx();
    int numchunks;
    IDXL_Share **chunks1;
    m->getfmMM()->getfmUtil()->getChunkNos(0,nodeId,&numchunks,&chunks1);
    int minChunk = MAX_CHUNK;
    for(int j=0; j<numchunks; j++) {
      int pchk = chunks1[j]->chk;
      if(pchk < minChunk) minChunk = pchk;
    }
    for(int j=0; j<numchunks; j++) {
      delete chunks1[j];
    }
    if(numchunks!=0) free(chunks1);
    if(minChunk==index) {
      if(readlock) {
	return m->getfmMM()->getfmLockN(nodeId)->runlock();
      } else {
	return m->getfmMM()->getfmLockN(nodeId)->wunlock(index);
      }
    }
    else {
      CkAssert(minChunk!=MAX_CHUNK);
      int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,minChunk,0);
      if(readlock) {
	return meshMod[minChunk].unlockRemoteNode(sharedIdx, index, 0, 1)->i;
      } else {
	return meshMod[minChunk].unlockRemoteNode(sharedIdx, index, 0, 0)->i;
      }
    }
  }
  else if(FEM_Is_ghost_index(nodeId)) {
    //find the index of the least chunk that owns it & try to unlock it
    int index = m->getfmMM()->getIdx();
    int numchunks;
    IDXL_Share **chunks1;
    m->getfmMM()->getfmUtil()->getChunkNos(0,nodeId,&numchunks,&chunks1);
    int minChunk = MAX_CHUNK;
    for(int j=0; j<numchunks; j++) {
      int pchk = chunks1[j]->chk;
      if(pchk == index) continue;
      if(pchk < minChunk) minChunk = pchk;
    }
    for(int j=0; j<numchunks; j++) {
      delete chunks1[j];
    }
    if(numchunks!=0) free(chunks1);
    if(minChunk==MAX_CHUNK) return -1;
    int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,minChunk,2);
    if(readlock) {
      return meshMod[minChunk].unlockRemoteNode(sharedIdx, index, 1, 1)->i;
    } else {
      return meshMod[minChunk].unlockRemoteNode(sharedIdx, index, 1, 0)->i;
    }
  }
  else {
    if(readlock) {
      return m->getfmMM()->getfmLockN(nodeId)->runlock();
    } else {
      int index = m->getfmMM()->getIdx();
      return m->getfmMM()->getfmLockN(nodeId)->wunlock(index);
    }
  }
  return -1; //should not reach here
}

//lock it on the minchunk other than 'this chunk'
//should only be called when 'this chunk' is losing the node as local/shared
void FEM_Modify_LockAll(FEM_Mesh*m, int nodeId, bool lockall) {
  int index = m->getfmMM()->getIdx();
  int numchunks;
  const IDXL_Rec *irec = m->node.shared.getRec(nodeId);
  //if it is a corner node, it might not have been a shared node, so can't lock it on anything else
  if(irec) {
    numchunks = irec->getShared();
    if(!lockall) {
      int minchunk=MAX_CHUNK;
      int sharedIdx = -1;
      for(int i=0; i<numchunks; i++) {
	int pchk = irec->getChk(i); 
	if(pchk<minchunk) {
	  minchunk = pchk;
	  sharedIdx = irec->getIdx(i);
	}
      }
      CkAssert(minchunk!=MAX_CHUNK && sharedIdx!=-1);
      //lock it on this chunk, if not already locked
      int done = meshMod[minchunk].lockRemoteNode(sharedIdx, index, 0, 0)->i;
      //if done=-1, then it is already locked, otherwise we just locked it
    }
    else {
      for(int i=0; i<numchunks; i++) {
	int pchk = irec->getChk(i); 
	int sharedIdx = irec->getIdx(i);
	int done = meshMod[pchk].lockRemoteNode(sharedIdx, index, 0, 0)->i;
      }
      m->getfmMM()->getfmLockN(nodeId)->wlock(index);
    }
  }
  return;
}

//must be a local node, lock it & then unlock it if needed
void FEM_Modify_LockUpdate(FEM_Mesh*m, int nodeId, bool lockall) {
  int index = m->getfmMM()->getIdx();
  int numchunks;
  const IDXL_Rec *irec = m->node.shared.getRec(nodeId);
  //if it is a corner, the new node might not be shared
  //nothing was locked, hence nothing needs to be unlocked too
  if(irec) {
    numchunks = irec->getShared();
    int minchunk=MAX_CHUNK;
    int minI=-1;
    for(int i=0; i<numchunks; i++) {
      int pchk = irec->getChk(i); 
      if(pchk<minchunk) {
	minchunk = pchk;
	minI=i;
      }
    }
    if(!lockall) {
      if(minchunk>index) {
	int prevminchunk=minchunk;
	minchunk=index;
	int sharedIdx = irec->getIdx(minI);
	CkAssert(prevminchunk!=MAX_CHUNK && sharedIdx!=-1);
	meshMod[prevminchunk].unlockRemoteNode(sharedIdx, index, 0, 0);
      }
      else if(minchunk < index) {
	//unlock the previously acquired lock
	int sharedIdx = irec->getIdx(minI);
	meshMod[minchunk].lockRemoteNode(sharedIdx, index, 0, 0);
	m->getfmMM()->getfmLockN(nodeId)->wunlock(index);
      }
    }
    else {
      if(minchunk>index) minchunk=index;
      if(minchunk!=index) {
	m->getfmMM()->getfmLockN(nodeId)->wunlock(index);
      }
      for(int i=0; i<numchunks; i++) {
	int pchk = irec->getChk(i);
	if(pchk!=minchunk) {
	  int sharedIdx = irec->getIdx(i);
	  meshMod[pchk].unlockRemoteNode(sharedIdx, index, 0, 0);
	}
      }
    }
  }
  return;
}


//should always be called on the new node to correct the locks
//should make sure that it is called before unlocking, i.e. before the entire operation completes.
//gets rid of the extra safety lock & makes the locks consistent
void FEM_Modify_correctLockN(FEM_Mesh *m, int nodeId) {
  int index = m->getfmMM()->getIdx();
  int numchunks;
  IDXL_Share **chunks1;
  m->getfmMM()->getfmUtil()->getChunkNos(0,nodeId,&numchunks,&chunks1);
  int minChunk = MAX_CHUNK;
  int owner = -1;
  for(int j=0; j<numchunks; j++) {
    int pchk = chunks1[j]->chk;
    if(pchk < minChunk) minChunk = pchk;
  }
  if(is_shared(m,nodeId)) {
    for(int j=0; j<numchunks; j++) {
      int pchk = chunks1[j]->chk;
      if(pchk == index) owner = m->getfmMM()->getfmLockN(nodeId)->lockOwner();
      else {
	int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,pchk,0);
	owner = meshMod[pchk].hasLockRemoteNode(sharedIdx, index, 0)->i;
      }
      if(owner != -1) { //this node is locked
	if(pchk == minChunk) {
	  //the lock is good
	  break;
	}
	else {
	  //unlock the node on pchk & lock it on minChunk
	  int locknodes = nodeId;
	  int gotlocks = 1;
	  int done = -1;
	  if(pchk==index) m->getfmMM()->getfmLockN(nodeId)->wunlock(index);
	  else {
	    int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,pchk,0);
	    meshMod[pchk].unlockRemoteNode(sharedIdx, index, 0, 0);
	  }
	  if(minChunk==index) done = m->getfmMM()->getfmLockN(nodeId)->wlock(index);
	  else {
	    CkAssert(minChunk!=MAX_CHUNK);
	    int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,minChunk,0);
	    done = meshMod[minChunk].lockRemoteNode(sharedIdx, index, 0, 0)->i;
	  }
	  break;
	}
      }
    }
  }
  else if(FEM_Is_ghost_index(nodeId)) {
    //someone must have the lock
    for(int j=0; j<numchunks; j++) {
      int pchk = chunks1[j]->chk;
      int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,pchk,2);
      owner = meshMod[pchk].hasLockRemoteNode(sharedIdx, index, 1)->i;
      if(owner != -1) { //this node is locked
	if(pchk == minChunk) {
	  //the lock is good
	  break;
	}
	else {
	  //unlock the node on pchk & lock it on minChunk
	  int locknodes = nodeId;
	  int gotlocks = 1;
	  int done = -1;
	  int sharedIdx = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,nodeId,pchk,2);
	  meshMod[pchk].unlockRemoteNode(sharedIdx, index, 1, 0);
	  gotlocks=-1;
	  while(done==-1) {
	    done = m->getfmMM()->fmAdaptL->lockNodes(&gotlocks, &locknodes, 0, &locknodes, 1);
	  }
	  break;
	}
      }
    }
    if(owner==-1) {
      int locknodes = nodeId;
      int done = -1;
      int gotlocks=-1;
      while(done==-1) {
	done = m->getfmMM()->fmAdaptL->lockNodes(&gotlocks, &locknodes, 0, &locknodes, 1);
      }
    }
  }
  for(int j=0; j<numchunks; j++) {
    delete chunks1[j];
  }
  if(numchunks!=0) free(chunks1);
  return;
}

CDECL void FEM_REF_INIT(int mesh, int dim) {
  CkArrayID femRefId;
  int cid;
  int size;
  TCharm *tc=TCharm::get();

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
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

#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_REF_INIT");
  FEMMeshMsg *msg = new FEMMeshMsg(m,dim); 
  meshMod[cid].setFemMesh(msg);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif

  return;
}

//this would be only done for attributes of a node, currently for coords & boundary
void FEM_Ghost_Essential_attributes(FEM_Mesh *m, int coord_attr, int bc_attr, int nodeid) {
  femMeshModify *theMod = m->getfmMM();
  FEM_MUtil *util = theMod->getfmUtil();
  int index = theMod->idx;

  if(FEM_Is_ghost_index(nodeid)) {
    //build up the list of chunks it is ghost to
    int numchunks;
    IDXL_Share **chunks1;
    util->getChunkNos(0,nodeid,&numchunks,&chunks1);
    for(int j=0; j<numchunks; j++) {
      int chk = chunks1[j]->chk;
      if(chk==index) continue;
      int ghostidx = util->exists_in_IDXL(m,nodeid,chk,2);
      double2Msg *d = meshMod[chk].getRemoteCoord(index,ghostidx);
      intMsg *im = meshMod[chk].getRemoteBound(index,ghostidx);
      double *coord = new double[2];
      coord[0] = d->i; coord[1] = d->j;
      int bound = im->i;
      CkVec<FEM_Attribute *>*attrs = m->node.ghost->getAttrVec();
      for (int i=0; i<attrs->size(); i++) {
	FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
	if (a->getAttr() == theMod->fmAdaptAlgs->coord_attr) {
	  FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	  d->getDouble().setRow(FEM_From_ghost_index(nodeid),coord,0);
	}
	else if(a->getAttr() == FEM_BOUNDARY) {
	  FEM_DataAttribute *d = (FEM_DataAttribute *)a;
	  d->getInt().setRow(FEM_From_ghost_index(nodeid),bound);
	}
      }
      delete [] coord;
      for(int j=0; j<numchunks; j++) {
	delete chunks1[j];
      }
      if(numchunks != 0) free(chunks1);
      break;
    }
  }

  return;
}


femMeshModify::femMeshModify(femMeshModMsg *fm) {
  numChunks = fm->numChunks;
  idx = fm->myChunk;
  fmLock = new FEM_lock(idx, this);
  fmUtil = new FEM_MUtil(idx, this);
  fmAdapt = NULL;
  fmAdaptL = NULL;
  fmAdaptAlgs = NULL;
  fmInp = NULL;
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
  fmAdapt = new FEM_Adapt(fmMesh, this);
  fmAdaptL = new FEM_AdaptL(fmMesh, this);
  int dim = fm->dimn;
  fmAdaptAlgs = new FEM_Adapt_Algs(fmMesh, this, dim);
  fmInp = new FEM_Interpolate(fmMesh, this);
  //populate the node locks
  int nsize = fmMesh->node.size();
  for(int i=0; i<nsize; i++) {
    fmLockN.push_back(new FEM_lockN(i,this));
  }
  /*int gsize = fmMesh->node.ghost->size();
    for(int i=0; i<gsize; i++) {
    fmgLockN.push_back(new FEM_lockN(FEM_To_ghost_index(i),this));
    }*/
  for(int i=0; i<numChunks*5; i++) {
    fmIdxlLock.push_back(false);
  }
  //compute all the fixed nodes
  for(int i=0; i<nsize; i++) {
    if(fmAdaptL->isCorner(i)) {
      fmfixedNodes.push_back(i);
    }
  }
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
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

intMsg *femMeshModify::lockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx;
  if(isGhost) {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 1);
  } else {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  }
  //CkAssert(localIdx != -1);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  intMsg *imsg = new intMsg(0);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int ret;
  if(localIdx == -1) {
    ret = -1;
  }
  else {
    if(readLock) {
      ret = fmLockN[localIdx]->rlock();
    } else {
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
      ret = fmLockN[localIdx]->wlock(fromChk);
#ifdef DEBUG 
      CmiMemoryCheck(); 
#endif
    }
  }
  imsg->i = ret;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return imsg;
}

intMsg *femMeshModify::unlockRemoteNode(int sharedIdx, int fromChk, int isGhost, int readLock) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx;
  if(isGhost) {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 1);
  } else {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  }
  CkAssert(localIdx != -1);
  intMsg *imsg = new intMsg(0);
  int ret;
  if(readLock) {
    ret = fmLockN[localIdx]->runlock();
  } else {
    ret = fmLockN[localIdx]->wunlock(fromChk);
  }
  imsg->i = ret;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return imsg;
}

intMsg *femMeshModify::hasLockRemoteNode(int sharedIdx, int fromChk, int isGhost) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx;
  if(isGhost) {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 1);
  } else {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  }
  CkAssert(localIdx != -1);
  intMsg *imsg = new intMsg(0);
  int ret = fmLockN[localIdx]->lockOwner();
  imsg->i = ret;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return imsg;
}

void femMeshModify::modifyLockAll(int fromChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 1);
  FEM_Modify_LockAll(fmMesh, localIdx);
  return;
}

intMsg *femMeshModify::addNodeRemote(addNodeMsg *msg) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  intMsg *imsg = new intMsg(-1);
  //translate the indices
  int *localIndices = (int*)malloc(msg->nBetween*sizeof(int));
  int *chunks = (int*)malloc(msg->numChunks*sizeof(int));
  CkAssert(msg->numChunks==1);
  chunks[0] = msg->chunks[0];
  for(int i=0; i<msg->nBetween; i++) {
    localIndices[i] = fmUtil->lookup_in_IDXL(fmMesh, msg->between[i], chunks[0], 0);
    CkAssert(localIndices[i] != -1);
  }
  int ret = FEM_add_node(fmMesh, localIndices, msg->nBetween, chunks, msg->numChunks, msg->forceShared, msg->upcall);
  //this is a ghost on that chunk,
  //add it to the idxl & update that guys idxl list
  fmMesh->node.ghostSend.addNode(ret,chunks[0]);
  imsg->i = fmUtil->exists_in_IDXL(fmMesh,ret,chunks[0],1);
  free(localIndices);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return imsg;
}

void femMeshModify::addSharedNodeRemote(sharedNodeMsg *fm) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  FEM_add_shared_node_remote(fmMesh, fm->chk, fm->nBetween, fm->between);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::removeSharedNodeRemote(removeSharedNodeMsg *fm) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->removeNodeRemote(fmMesh, fm->chk, fm->index);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::addGhostElem(addGhostElemMsg *fm) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->addGhostElementRemote(fmMesh, fm->chk, fm->elemType, fm->numGhostIndex, fm->ghostIndices, fm->numSharedIndex, fm->sharedIndices, fm->connSize);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

chunkListMsg *femMeshModify::getChunksSharingGhostNode(int2Msg *i2m) {
  chunkListMsg *clm;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  clm = fmUtil->getChunksSharingGhostNodeRemote(fmMesh, i2m->i, i2m->j);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return clm;
}

void femMeshModify::addElementRemote(addElemMsg *fm) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->addElemRemote(fmMesh, fm->chk, fm->elemtype, fm->connSize, fm->conn, fm->numGhostIndex, fm->ghostIndices);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::removeGhostElem(removeGhostElemMsg *fm) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->removeGhostElementRemote(fmMesh, fm->chk, fm->elementid, fm->elemtype, fm->numGhostIndex, fm->ghostIndices, fm->numGhostRNIndex, fm->ghostRNIndices, fm->numGhostREIndex, fm->ghostREIndices, fm->numSharedIndex, fm->sharedIndices);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::removeElementRemote(removeElemMsg *fm) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->removeElemRemote(fmMesh, fm->chk, fm->elementid, fm->elemtype, fm->permanent);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::removeGhostNode(int fromChk, int sharedIdx) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->removeGhostNodeRemote(fmMesh, fromChk, sharedIdx);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

intMsg *femMeshModify::eatIntoElement(int fromChk, int sharedIdx) {
  intMsg *imsg = new intMsg(-1);
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 4);
  if(localIdx==-1) return imsg;
  int newEl = fmUtil->eatIntoElement(FEM_To_ghost_index(localIdx));
  int returnIdx = fmUtil->exists_in_IDXL(fmMesh, newEl, fromChk, 3);
  //CkAssert(returnIdx!=-1); //returnIdx=-1 means that it was a discontinuous part
  if(returnIdx==-1) {
    //the other node has lost the entire region, unlock all the nodes involved
    //if there is a node 'fromChk' doesn't know about, but it is holding a lock on it, unlock it
    const int nodesPerEl = fmMesh->elem[0].getConn().width(); // should be the number of nodes that can be adjacent to this element
    int *adjnodes = new int[nodesPerEl];
    int numLocks = nodesPerEl + 1; //for 2D.. will be different for 3D
    int *lockednodes = new int[numLocks];
    fmMesh->e2n_getAll(newEl, adjnodes, 0);
    int newNode = -1;
    bool foundNewNode = false;
    //get the other node, which will be on an element that is an e2e of newEl
    //and which fromChk doesn't know abt, but has a lock on
    /*int *adjelems = new int[nodesPerEl];
    fmMesh->e2e_getAll(newEl, adjelems, 0);
    int *nnds = new int[nodesPerEl];
    for(int i=0; i<nodesPerEl; i++) {
      if(adjelems[i]!=-1) {
	fmMesh->e2n_getAll(adjelems[i], nnds, 0);
	for(int j=0; j<nodesPerEl; j++) {
	  bool istarget = true;
	  for(int k=0; k<nodesPerEl; k++) {
	    if(nnds[j]==adjnodes[k]) {
	      istarget = false;
	    }
	  }
	  //get the owner of the lock
	  if(istarget) {
	    int lockowner = fmUtil->getLockOwner(nnds[j]);
	    if(lockowner==fromChk) {
	      bool knows = fmUtil->knowsAbtNode(fromChk,nnds[j]);
	      //fromChk doesn't know abt this node
	      if(!knows) {
		newNode = nnds[j];
		foundNewNode = true;
		break;
	      }
	    }
	  }
	  if(foundNewNode) break;
	}
      }
      if(foundNewNode) break;
    }
    delete[] adjelems;
    delete[] nnds;
    */
    int *gotlocks = new int[numLocks];
    for(int i=0; i<nodesPerEl; i++) {
      lockednodes[i] = adjnodes[i];
      gotlocks[i] = 1;
    }
    if(foundNewNode) {
      lockednodes[nodesPerEl] = newNode;
      gotlocks[nodesPerEl] = 1;
    }
    else numLocks--;
    fmAdaptL->unlockNodes(gotlocks, lockednodes, 0, lockednodes, numLocks);
    delete[] lockednodes;
    delete[] gotlocks;
    delete[] adjnodes;
  }
  imsg->i = returnIdx;
  return imsg;
}

intMsg *femMeshModify::getLockOwner(int fromChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  int ret = fmUtil->getLockOwner(localIdx);
  intMsg *imsg = new intMsg(ret);
  return imsg;
}

boolMsg *femMeshModify::knowsAbtNode(int fromChk, int toChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 1);
  bool ret = fmUtil->knowsAbtNode(toChk,localIdx);
  boolMsg *bmsg = new boolMsg(ret);
  return bmsg;
}


void femMeshModify::refine_flip_element_leb(int fromChk, int propElemT, 
					    int propNodeT, int newNodeT, 
					    int nbrOpNodeT, int nbrghost, double longEdgeLen)
{
#ifdef DEBUG 
  CmiMemoryCheck();
#endif
  int propElem, propNode, newNode, nbrOpNode;
  propElem = getfmUtil()->lookup_in_IDXL(fmMesh, propElemT, fromChk, 3, 0);
  propNode = getfmUtil()->lookup_in_IDXL(fmMesh, propNodeT, fromChk, 0, -1);
  newNode = getfmUtil()->lookup_in_IDXL(fmMesh, newNodeT, fromChk, 0, -1);
  if(nbrghost) {
    nbrOpNode = getfmUtil()->lookup_in_IDXL(fmMesh, nbrOpNodeT, fromChk, 1,-1); 
  }
  else {
    nbrOpNode = getfmUtil()->lookup_in_IDXL(fmMesh, nbrOpNodeT, fromChk, 0,-1); 
  }
  fmAdaptAlgs->refine_flip_element_leb(propElem, propNode, newNode, nbrOpNode, longEdgeLen);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::addToSharedList(int fromChk, int sharedIdx) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->addToSharedList(fmMesh, fromChk, sharedIdx);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
}

void femMeshModify::updateNodeAttrs(int fromChk, int sharedIdx, double coordX, double coordY, int bound, bool isGhost) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx = -1;
  if(!isGhost) {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  }
  else localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 2);
  double *coord = new double[2];
  coord[0] = coordX; coord[1] = coordY;
  CkVec<FEM_Attribute *>*attrs = (fmMesh->node).getAttrVec();
  for (int i=0; i<attrs->size(); i++) {
    FEM_Attribute *a = (FEM_Attribute *)(*attrs)[i];
    if (a->getAttr() == fmAdaptAlgs->coord_attr) {
      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
      d->getDouble().setRow(localIdx,coord,0);
    }
    else if(a->getAttr() == FEM_BOUNDARY) {
      FEM_DataAttribute *d = (FEM_DataAttribute *)a;
      d->getInt().setRow(localIdx,bound);
    }
  }
  delete [] coord;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

double2Msg *femMeshModify::getRemoteCoord(int fromChk, int ghostIdx) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, ghostIdx, fromChk, 1);
  double coord[2];
  FEM_Mesh_dataP(fmMesh, FEM_NODE, fmAdaptAlgs->coord_attr, coord, localIdx, 1, FEM_DOUBLE, 2);
  double2Msg *d = new double2Msg(coord[0], coord[1]);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return d;
}

intMsg *femMeshModify::getRemoteBound(int fromChk, int ghostIdx) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, ghostIdx, fromChk, 1);
  int bound;
  FEM_Mesh_dataP(fmMesh, FEM_NODE, FEM_BOUNDARY, &bound, localIdx, 1, FEM_INT, 1);
  intMsg *d = new intMsg(bound);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return d;
}

intMsg *femMeshModify::getIdxGhostSend(int fromChk, int idxshared, int toChk) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, idxshared, fromChk, 0);
  int idxghostsend = -1;
  if(localIdx != -1) { 
    const IDXL_Rec *irec = fmMesh->node.ghostSend.getRec(localIdx);
    if(irec) {
      for(int i=0; i<irec->getShared(); i++) {
	if(irec->getChk(i) == toChk) {
	  idxghostsend = fmUtil->exists_in_IDXL(fmMesh, localIdx, toChk, 1);
	  break;
	}
      }
    }
  }
  intMsg *d = new intMsg(idxghostsend);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return d;
}

void femMeshModify::updateIdxlList(int fromChk, int idxTrans, int transChk) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int idxghostrecv = fmUtil->lookup_in_IDXL(fmMesh, idxTrans, transChk, 2);
  CkAssert(idxghostrecv != -1);
  fmMesh->node.ghost->ghostRecv.addNode(idxghostrecv,fromChk);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::addTransIDXLRemote(int fromChk, int sharedIdx, int transChk) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, transChk, 0);
  CkAssert(localIdx != -1);
  fmMesh->node.ghostSend.addNode(localIdx,fromChk);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::removeIDXLRemote(int fromChk, int sharedIdx, int type) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, type);
  CkAssert(localIdx != -1);
  fmMesh->node.ghostSend.removeNode(localIdx,fromChk);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::verifyIdxlList(int fromChk, int size, int type) {
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->verifyIdxlListRemote(fmMesh, fromChk, size, type);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::idxllockRemote(int fromChk, int type) {
  if(type==1) type = 2;
  else if(type ==2) type = 1;
  else if(type ==3) type = 4;
  else if(type ==4) type = 3;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->idxllockLocal(fmMesh, fromChk, type);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

void femMeshModify::idxlunlockRemote(int fromChk, int type) {
  if(type==1) type = 2;
  else if(type ==2) type = 1;
  else if(type ==3) type = 4;
  else if(type ==4) type = 3;
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  fmUtil->idxlunlockLocal(fmMesh, fromChk, type);
#ifdef DEBUG 
  CmiMemoryCheck(); 
#endif
  return;
}

boolMsg *femMeshModify::verifyLock(int fromChk, int sharedIdx, int isGhost) {
  int localIdx;
  if(isGhost) {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 1);
  } else {
    localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  }
  //CkAssert(localIdx != -1);
  boolMsg *bmsg = new boolMsg(0);
  bool ret;
  if(localIdx == -1) {
    ret = false;
  }
  else {
    ret = fmLockN[localIdx]->verifyLock();
  }
  bmsg->b = ret;
  return bmsg;
}

void femMeshModify::verifyghostsend(verifyghostsendMsg *vmsg) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, vmsg->sharedIdx, vmsg->fromChk, 1);
  const IDXL_Rec *irec = fmMesh->node.shared.getRec(localIdx);
  if (irec!=NULL) {
    int numsh = irec->getShared();
    CkAssert(numsh==vmsg->numchks-1);
    for(int i=0; i<numsh; i++) {
      int ckl = irec->getChk(i);
      bool found = false;
      for(int j=0; j<numsh+1; j++) {
	if(vmsg->chunks[j]==ckl) {
	  found = true; break;
	}
      }
      CkAssert(found);
    }
  }
}

findgsMsg *femMeshModify::findghostsend(int fromChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  int *chkl, numchkl=0;
  fmUtil->findGhostSend(localIdx, &chkl, &numchkl);
  findgsMsg *fmsg = new(numchkl)findgsMsg();
  fmsg->numchks = numchkl;
  for(int i=0; i<numchkl; i++) fmsg->chunks[i] = chkl[i];
  if(numchkl>0) delete[] chkl;
  return fmsg;
}

void femMeshModify::updateghostsend(verifyghostsendMsg *vmsg) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, vmsg->sharedIdx, vmsg->fromChk, 0);
  fmUtil->UpdateGhostSend(localIdx, vmsg->chunks, vmsg->numchks);
  delete vmsg;
}

boolMsg *femMeshModify::shouldLoseGhost(int fromChk, int sharedIdx, int toChk) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  int *elems, numElems;
  fmMesh->n2e_getAll(localIdx, &elems, &numElems);
  bool shouldBeDeleted = true;
  for(int k=0; k<numElems; k++) {
    if(elems[k]>=0) {
      if(fmMesh->getfmMM()->fmUtil->exists_in_IDXL(fmMesh,elems[k],toChk,3)!=-1) {
	shouldBeDeleted = false;
	break;
      }
    }
  }
  if(numElems>0) delete[] elems;
  boolMsg *bmsg = new boolMsg(shouldBeDeleted);
  return bmsg;
}

void femMeshModify::addghostsendl(int fromChk, int sharedIdx, int toChk, int transIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 0);
  int sharedghost = fmUtil->exists_in_IDXL(fmMesh,localIdx,toChk,1);
  if(sharedghost==-1) { //it needs to be added from this chunk
    //lock idxl
    fmMesh->node.ghostSend.addNode(localIdx,toChk);
    meshMod[toChk].addghostsendl1(idx,fromChk,transIdx);
    //unlock idxl
  }
}

void femMeshModify::addghostsendl1(int fromChk, int transChk, int transIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, transIdx, transChk, 2);
  fmMesh->node.ghost->ghostRecv.addNode(localIdx,fromChk);
}

void femMeshModify::addghostsendr(int fromChk, int sharedIdx, int toChk, int transIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 2);
  int sharedghost = fmUtil->exists_in_IDXL(fmMesh,FEM_To_ghost_index(localIdx),toChk,2);
  if(sharedghost==-1) { //it needs to be added from this chunk
    //lock idxl
    fmMesh->node.ghost->ghostRecv.addNode(localIdx,toChk);
    meshMod[toChk].addghostsendr1(idx,fromChk,transIdx);
    //unlock idxl
  }
}

void femMeshModify::addghostsendr1(int fromChk, int transChk, int transIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, transIdx, transChk, 0);
  fmMesh->node.ghostSend.addNode(localIdx,fromChk);
}

boolMsg *femMeshModify::willItLose(int fromChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 3);
  int nnbrs[3];
  fmMesh->e2n_getAll(localIdx,nnbrs,0);
  //if it loses any node if it loses this element, then it should let fromChk, acquire this elem
  bool willlose = true;
  for(int i=0; i<3; i++) {
    int *enbrs, numenbrs;
    fmMesh->n2e_getAll(nnbrs[i],&enbrs,&numenbrs);
    willlose = true;
    for(int j=0; j<numenbrs; j++) {
      if(enbrs[j]>=0 && enbrs[j]!=localIdx) {
	willlose = false;
	break;
      }
    }
    if(numenbrs>0) delete [] enbrs;
    if(willlose) break;
  }
  boolMsg *bmsg = new boolMsg(willlose);
  return bmsg;
}

void femMeshModify::interpolateElemCopy(int fromChk, int sharedIdx1, int sharedIdx2) {
  int localIdx1 = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx1, fromChk, 3);
  int localIdx2 = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx2, fromChk, 3);
  CkAssert(localIdx1!=-1 && localIdx2!=-1);
  fmUtil->copyElemData(0,localIdx1,localIdx2);
}

void femMeshModify::cleanupIDXL(int fromChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 4);
  CkAssert(fmUtil->exists_in_IDXL(fmMesh,FEM_To_ghost_index(localIdx),fromChk,4)!=-1);
  fmMesh->elem[0].ghost->ghostRecv.removeNode(localIdx, fromChk);
  fmMesh->elem[0].getGhost()->set_invalid(localIdx,false);
}

void femMeshModify::purgeElement(int fromChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 3);
  CkAssert(localIdx!=-1);
  FEM_purge_element(fmMesh,localIdx,0);
}


elemDataMsg *femMeshModify::packElemData(int fromChk, int sharedIdx) {
  int localIdx = fmUtil->lookup_in_IDXL(fmMesh, sharedIdx, fromChk, 3);
  CkAssert(localIdx!=-1);
  CkVec<FEM_Attribute *>*elemattrs = (fmMesh->elem[0]).getAttrVec();
  int count = 0;
  PUP::sizer psizer;
  for(int j=0;j<elemattrs->size();j++){
    FEM_Attribute *elattr = (FEM_Attribute *)(*elemattrs)[j];
    if(elattr->getAttr() < FEM_ATTRIB_FIRST){ 
      elattr->pupSingle(psizer, localIdx);
      count++;
    }
  }
  elemDataMsg *edm = new (psizer.size()) elemDataMsg(count);

  PUP::toMem pmem(edm->data);
  for(int j=0;j<elemattrs->size();j++){
    FEM_Attribute *elattr = (FEM_Attribute *)(*elemattrs)[j];
    if(elattr->getAttr() < FEM_ATTRIB_FIRST){ 
      elattr->pupSingle(pmem, localIdx);
    }
  }
  return edm;
}

#include "FEMMeshModify.def.h"
