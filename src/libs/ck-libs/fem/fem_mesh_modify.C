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


CDECL int FEM_add_node(int mesh, int* adjacent_nodes, int num_adjacent_nodes, int upcall){
  return FEM_add_node(FEM_Mesh_lookup(mesh,"FEM_add_node"), adjacent_nodes, num_adjacent_nodes, upcall);}
CDECL void FEM_remove_node(int mesh,int node){
  FEM_remove_node(FEM_Mesh_lookup(mesh,"FEM_remove_node"), node);}
CDECL void FEM_remove_element(int mesh, int element, int elem_type){
  FEM_remove_element(FEM_Mesh_lookup(mesh,"FEM_remove_element"), element, elem_type);}
CDECL int FEM_add_element(int mesh, int* conn, int conn_size, int elem_type){
  return FEM_add_element(FEM_Mesh_lookup(mesh,"FEM_add_element"), conn, conn_size, elem_type);}
CDECL void FEM_Modify_Lock(int mesh, int* affectedNodes, int numAffectedNodes, int* affectedElts, int numAffectedElts, int elemtype){
  FEM_Modify_Lock(FEM_Mesh_lookup(mesh,"FEM_Modify_Lock"), affectedNodes, numAffectedNodes, affectedElts, numAffectedElts, elemtype);}
CDECL void FEM_Modify_Unlock(int mesh){
  FEM_Modify_Unlock(FEM_Mesh_lookup(mesh,"FEM_Modify_Unlock"));}



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
  CkPrintf("node %d is adjacent to nodes:", nodeid);
  int *adjnodes;
  int sz;
  m->n2n_getAll(nodeid, &adjnodes, &sz); 
  for(int i=0;i<sz;i++)
	CkPrintf(" %d", adjnodes[i]);
  if(sz!=0) delete[] adjnodes;  
  CkPrintf("\n");
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

CDECL void FEM_Print_n2e(int mesh, int eid){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  CkPrintf("node %d is adjacent to elements:", eid);
  int *adjes;
  int sz;
  m->n2e_getAll(eid, &adjes, &sz);
  for(int i=0;i<sz;i++)
	CkPrintf(" %d", adjes[i]);
  if(sz!=0) delete[] adjes;
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


// WARNING THESE TWO FUNCTIONS ONLY WORK ON TRIANGULAR ELEMENTS...
CDECL void FEM_Print_e2n(int mesh, int eid){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  CkPrintf("element %d is adjacent to nodes:", eid);
  int adjns[3];
  m->e2n_getAll(eid, adjns, 0); 
  for(int i=0;i<3;i++)
	CkPrintf(" %d", adjns[i]);
  CkPrintf("\n");
}

void FEM_MUtil::FEM_Print_e2n(FEM_Mesh *m, int eid){
  CkPrintf("element %d is adjacent to nodes:", eid);
  int adjns[3];
  m->e2n_getAll(eid, adjns, 0); 
  for(int i=0;i<3;i++)
	CkPrintf(" %d", adjns[i]);
  CkPrintf("\n");
}

CDECL void FEM_Print_e2e(int mesh, int eid){
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
  CkPrintf("element %d is adjacent to elements:", eid);
  int adjes[3];
  m->e2e_getAll(eid, adjes, 0); 
  for(int i=0;i<3;i++)
	CkPrintf(" %d", adjes[i]);
  CkPrintf("\n");
}

void FEM_MUtil::FEM_Print_e2e(FEM_Mesh *m, int eid){
  CkPrintf("element %d is adjacent to elements:", eid);
  int adjes[3];
  m->e2e_getAll(eid, adjes, 0); 
  for(int i=0;i<3;i++)
	CkPrintf(" %d", adjes[i]);
  CkPrintf("\n");
}


int FEM_add_node_local(FEM_Mesh *m, int addGhost){
  int newNode;
  if(addGhost){
	newNode = m->node.getGhost()->size();
	m->node.getGhost()->setLength(newNode+1); // lengthen node attributes
	m->node.getGhost()->set_valid(newNode);   // set new node as valid
  }
  else{
	newNode = m->node.size();
	m->node.setLength(newNode+1); // lengthen node attributes
	m->node.set_valid(newNode);   // set new node as valid
  }
  m->n2e_removeAll(newNode);    // initialize element adjacencies
  m->n2n_removeAll(newNode);    // initialize node adjacencies
  return newNode;  // return a new index
}

int FEM_add_node(FEM_Mesh *m, int* adjacentNodes, int numAdjacentNodes, int upcall){
  // add local node
  int newNode = FEM_add_node_local(m, 0);
  int sharedCount = 0;

  FEM_Interpolate *inp = m->getfmMM()->getfmInp();
  FEM_Interpolate::NodalArgs nm;
  nm.n = newNode;
  for(int i=0; i<numAdjacentNodes; i++) {
    nm.nodes[i] = adjacentNodes[i];
  }
  nm.frac = 0.5;
  inp->FEM_InterpolateNodeOnEdge(nm);

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
  int newnode = FEM_add_node_local(m, 0);
  
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
    
    // mark node as deleted/invalid
  if(FEM_Is_ghost_index(node)){
    if((numAdjNodes==0) && (numAdjElts==0)) {
      m->node.ghost->set_invalid(FEM_To_ghost_index(node));
    } //otherwise this ghost node is connected to some element in another chunk, which the chunk that just informed us doesn't know abt
  }
  else {
    CkAssert((numAdjNodes==0) && (numAdjElts==0)); // we shouldn't be removing a node away that is connected to anything
    m->node.set_invalid(node);
  }
}


// remove a local or shared node, but NOT a ghost node
// Should probably be able to handle ghosts someday, but I cannot 
// remember the reasoning for not allowing them
void FEM_remove_node(FEM_Mesh *m, int node){

  if(node == -1) return; // -1 is not even a valid ghost number

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
void FEM_remove_element(FEM_Mesh *m, int elementid, int elemtype){

  if(elementid == -1) return; // -1 is not even a valid ghost number

  if(FEM_Is_ghost_index(elementid)){
    // remove local ghost element
    //FEM_remove_element_local(m, elementid, elemtype);
    
    //an element can come as a ghost from only one chunk, so just convert args and call it on that chunk
    int ghostid = FEM_To_ghost_index(elementid);
    const IDXL_Rec *irec = m->elem[elemtype].ghost->ghostRecv.getRec(ghostid);
    int size = irec->getShared();
    CkAssert(size == 1);
    int remoteChunk = irec->getChk(0);
    int sharedIdx = irec->getIdx(0);
    
    removeElemMsg *rm = new removeElemMsg;
    rm->chk = m->getfmMM()->getfmUtil()->getIdx();
    rm->elementid = sharedIdx;
    rm->elemtype = elemtype;
    meshMod[remoteChunk].removeElementRemote(rm);
    //another possible deadlock
  }
  else {
    // remove local element
    FEM_remove_element_local(m, elementid, elemtype);

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
      for(int i=0; i<numSharedChunks; i++) {
	irec = m->elem[elemtype].ghostSend.getRec(elementid);
	int chk = irec->getChk(0);
	int sharedIdx = irec->getIdx(0);
	//get the list of n2e for all nodes of this element. If any node has only this element in its list.
	//it no longer should be a ghost on chk
	m->elem[elemtype].ghostSend.removeNode(elementid, chk);
	const IDXL_List ll = m->elem[elemtype].ghostSend.getList(chk);
	int size = ll.size();
	int connSize = m->elem[elemtype].getConn().width();
	int *nodes = (int*)malloc(connSize*sizeof(int));
	int numGhostNodes = 0;
	int *ghostIndices = (int*)malloc(connSize*sizeof(int));
	m->e2n_getAll(elementid,nodes,elemtype);
	
	const IDXL_List ln = m->node.ghostSend.getList(chk);
	int sizeN = ln.size();
	
	for(int j=0; j<connSize; j++) {
	  int *elems;
	  int numElems;
	  m->n2e_getAll(nodes[j], &elems, &numElems);
	  
	  //if any of these elems is a ghost on chk then do not delete this ghost node
	  int shouldBeDeleted = 1;
	  for(int k=0; k<numElems; k++) {
	    for(int l=0; l<size; l++) {
	      if(elems[k] == ll[l]) {
		shouldBeDeleted = 0; 
		break;
	      }
	    }
	    if(shouldBeDeleted == 0) break;
	  }
	  
	  //add this to the list of ghost nodes to be deleted on the remote chunk
	  if(shouldBeDeleted == 1) {
	    //convert this local index to a shared index
	    for(int k=0; k<sizeN; k++) {
	      if(nodes[j] == ln[k]) {
		m->node.ghostSend.removeNode(nodes[j], chk);
		ghostIndices[numGhostNodes] = k;
		numGhostNodes++;
	      }
	    }
	  }
	}
	//now that all ghost nodes to be removed have been decided, we add the elem & call the entry method
	removeGhostElemMsg *rm = new (numGhostNodes, 0) removeGhostElemMsg;
	rm->chk = m->getfmMM()->getfmUtil()->getIdx();
	rm->elemtype = elemtype;
	rm->elementid = sharedIdx;
	rm->numGhostIndex = numGhostNodes;
	for(int j=0; j<numGhostNodes; j++) {
	  rm->ghostIndices[j] = ghostIndices[j];
	}
	meshMod[chk].removeGhostElem(rm);  //update the ghosts on all shared chunks
      }
    }
  }
  return;
}

void FEM_remove_element_remote(FEM_Mesh *m, int element, int elemtype){
  // remove local element from elem[elemType] table
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
	int oldLength = m->elem[elemType].getGhost()->size();
	m->elem[elemType].getGhost()->setLength(oldLength+1);
	m->elem[elemType].getGhost()->set_valid(oldLength);// Mark new element as valid
	((FEM_Elem*)m->elem[elemType].getGhost())->connIs(oldLength,conn);// update element's conn, i.e. e2n table
	newEl = FEM_From_ghost_index(oldLength);
}
  else{
	int oldLength = m->elem[elemType].size();
	m->elem[elemType].setLength(oldLength+1);
	newEl = oldLength;
	m->elem[elemType].set_valid(newEl);  // Mark new element as valid
	m->elem[elemType].connIs(newEl,conn);  // update element's conn, i.e. e2n table
  }
  
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

  if(sharedcount==0 && ghostcount==0){// add a local elem with all local nodes
    newEl = FEM_add_element_local(m,conn,connSize,elemType,0);
    //no modifications required for ghostsend or ghostrecv of nodes or elements
  }
  else if(ghostcount==0 && sharedcount > 0){// a local elem with some or ALL shared nodes
    newEl = FEM_add_element_local(m,conn,connSize,elemType,0);
    
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
      int chk = (*allChunks)[i];
      if(chk == m->getfmMM()->getfmUtil()->getIdx()) continue; //it is this chunk
      //it is a new element so it could not have existed as a ghost on that chunk. Just add it.
      m->elem[elemType].ghostSend.addNode(newEl,chk);
      //ghost nodes should be added only if they were not already present as ghosts on that chunk.
      int numNodesToAdd = 0;
      int numSharedGhosts = 0;
      int numSharedNodes = 0;
      int *sharedGhosts = (int *)malloc((connSize-1)*sizeof(int));
      int *sharedNodes = (int *)malloc((connSize)*sizeof(int));
      for(int j=0; j<connSize; j++) {
	int sharedNode = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[j],chk,0);
	if(sharedNode == -1) {
	  //node 'j' is a ghost on chunk 'i'
	  int sharedGhost = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[j],chk,1);
	  if( sharedGhost == -1) {
	    //it is a new ghost
	    m->node.ghostSend.addNode(FEM_To_ghost_index(conn[j]),chk);
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
      addGhostElemMsg *fm = new (numSharedGhosts, numSharedNodes, 0)addGhostElemMsg;
      fm->chk = m->getfmMM()->getfmUtil()->getIdx();
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
      meshMod[chk].addGhostElem(fm); //newEl, m->fmMM->idx, elemType;
    }
  }
  else if(ghostcount > 0 && localcount == 0 && sharedcount > 0) { // it is remote elem with some shared nodes
    //figure out which chunk it is local to
    //among all chunks who share some of the nodes or from whom this chunk receives ghost nodes
    //if there is any chunk which owns a ghost node which is not shared, then that is a local node 
    //to that chunk and that chunk owns that element. However, only that chunk knows abt it.
    //So, just go to the owner of every ghost node and figure out who all share that node.
    //Build up this table of nodes owned by which all chunks.
    //The chunk that is in the table corresponding to all nodes wins the element.

    CkVec<int> **allShared;
    int numSharedChunks = 0;
    CkVec<int> *allChunks;
    int **sharedConn; 
    m->getfmMM()->getfmUtil()->buildChunkToNodeTable(nodetype, sharedcount, ghostcount, localcount, conn, connSize, &allShared, &numSharedChunks, &allChunks, &sharedConn);   
    
    //we are looking for any chunk which does not have a ghost node
    int remoteChunk = -1;
    for(int i=0; i<numSharedChunks; i++) {
      remoteChunk = i;
      for(int j=0; j<connSize; j++) {
	if(sharedConn[i][j] == -1) {
	  remoteChunk = -1;
	  break; //this chunk has a ghost node
	}
	if(sharedConn[i][j] == 0) {
	  break; //this is a local node, hence it is the remotechunk
	}
      }
      if(remoteChunk == i) break;
      else remoteChunk = -1;
    }
    CkAssert(remoteChunk != -1);

    remoteChunk = (*allChunks)[remoteChunk];
    //convert all connections to the shared IDXL indices. We should also tell which are ghost indices
    int numGhostNodes = 0;
    for(int i=0; i<connSize; i++) {
      if(nodetype[i] == 2) { //a ghost node
	numGhostNodes++;
      }
    }
    CkAssert(numGhostNodes > 0);
    addElemMsg *am = new (connSize, numGhostNodes, 0) addElemMsg;
    int chk = m->getfmMM()->getfmUtil()->getIdx();
    am->chk = chk;
    am->elemtype = elemType;
    am->connSize = connSize;
    am->numGhostIndex = numGhostNodes;
    int j = 0;
    for(int i=0; i<connSize; i++) {
      if(nodetype[i] == 1) {
	am->conn[i] = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[i],remoteChunk,0);
      }
      else if(nodetype[i] == 2) {
	am->conn[i] = m->getfmMM()->getfmUtil()->exists_in_IDXL(m,conn[i],remoteChunk,2);
	am->ghostIndices[j] = i;
	j++;
      }
    }
    meshMod[remoteChunk].addElementRemote(am);
    //not sure what to return, it is not a local element anyway, so lets just return -1;
    //this might be a source of a deadlock, because I am sure the remote chunk will call a sync entry method on this chunk for updating my ghosts....

    //pick up the last entry from the element ghostrecv IDXL with remoteChunk, that is the index of the last element added.
    const IDXL_List ilist = m->elem[elemType].ghost->ghostRecv.getList(remoteChunk);
    int size = ilist.size();
    newEl = ilist[size-1];
    newEl = FEM_To_ghost_index(newEl);
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

  return newEl;
}


int FEM_add_element_remote(){
  // promote ghosts to shared

  // find new ghosts for remote calling chunk by looking at new shared nodes
  // send these new ghosts to the remote calling chunk.

  // update my adjacencies

  return 0;
}


void FEM_Modify_Lock(FEM_Mesh *m, int* affectedNodes, int numAffectedNodes, int* affectedElts, int numAffectedElts, int elemtype) {
  m->getfmMM()->getfmLock()->lock(numAffectedNodes, affectedNodes, numAffectedElts, affectedElts, elemtype);
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
int FEM_lock::lock(int numNodes, int *nodes, int numElems, int* elems, int elemType) {
  bool done = false;
  int ret = 0;
  while(!done) {
    if(!isLocked || (isLocked && isOwner)) {
      for(int i=0; i<numNodes; i++) {
	if(i==0) { //lock myself, just once
	  if(!existsChunk(idx)) {
	    lockedChunks.push_back(idx);
	  }
	}
	//which chunk does this belong to
	//add that chunk to the lock list, if it does not exist already.
	if(nodes[i] != -1) {
	  int numchunks;
	  IDXL_Share **chunks1;
	  mmod->fmUtil->getChunkNos(0,nodes[i],&numchunks,&chunks1);
	  for(int j=0; j<numchunks; j++) {
	    if(!existsChunk(chunks1[j]->chk)) {
	      lockedChunks.push_back(chunks1[j]->chk);
	    }
	  }
	}
      }
      for(int i=0; i<numElems; i++) {
	//which chunk does this belong to
	//add that chunk to the lock list, if not already in it.
	if(elems[i] != -1) {
	  int numchunks;
	  IDXL_Share **chunks1;
	  mmod->fmUtil->getChunkNos(1,elems[i],&numchunks,&chunks1,elemType);
	  for(int j=0; j<numchunks; j++) {
	    if(!existsChunk(chunks1[j]->chk)) {
	      lockedChunks.push_back(chunks1[j]->chk);
	    }
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
      noShared = irec->getShared(); //check this value!!
      CkAssert(noShared > 0);
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
    else if(type == 1) {
      const IDXL_Rec *irec = mmod->fmMesh->node.shared.getRec(entNo);
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
      *numChunks = irec->getShared(); //should be 1
      *chunks = (IDXL_Share**)malloc((*numChunks)*sizeof(IDXL_Share*));
      for(int i=0; i<*numChunks; i++) {
	int chk = irec->getChk(i);
	int index = irec->getIdx(i);
	(*chunks)[i] = new IDXL_Share(chk, index);
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

  FEM_Interpolate *inp = m->getfmMM()->getfmInp();
  FEM_Interpolate::NodalArgs nm;
  nm.n = localIdx;
  for(int i=0; i<nBetween; i++) {
    nm.nodes[i] = localIndices[i];
  }
  nm.frac = 0.5;
  inp->FEM_InterpolateNodeOnEdge(nm);

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
  else if(type == 3) { //ghost node recv 
    ll = m->elem[elemType].ghostSend.getList(chk);
  }
  else if(type == 4) { //ghost node recv 
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
  for(int i=0; i<connSize; i++) {
    if(ghostIndices[j] == i) {
      localIndices[i] = ll1[conn[i]];
      j++;
    }
    else {
      localIndices[i] = ll2[conn[i]];
    }
  }

  FEM_add_element(m, localIndices, connSize, elemtype);
  return;
}


void FEM_MUtil::removeGhostElementRemote(FEM_Mesh *m, int chk, int elementid, int elemtype, int numGhostIndex, int *ghostIndices) {
  //translate all ghost node coordinates to local coordinates and delete those ghost nodes on chk
  //remove ghost element elementid on chk

  const IDXL_List ll2 = m->elem[elemtype].ghost->ghostRecv.getList(chk);
  int localIdx = ll2[elementid];
  m->elem[elemtype].ghost->ghostRecv.removeNode(localIdx, chk);
  FEM_remove_element_local(m, FEM_To_ghost_index(localIdx), elemtype);

  //convert existing remote ghost indices to local ghost indices 
  const IDXL_List ll1 = m->node.ghost->ghostRecv.getList(chk);
  for(int i=0; i<numGhostIndex; i++) {
    int localIdx = ll1[ghostIndices[i]];
    m->node.ghost->ghostRecv.removeNode(localIdx, chk);
    FEM_remove_node_local(m, FEM_To_ghost_index(localIdx));
  }

  return;
}

void FEM_MUtil::removeElemRemote(FEM_Mesh *m, int chk, int elementid, int elemtype) {

  const IDXL_List ll = m->elem[elemtype].ghostSend.getList(chk);
  int localIdx = ll[elementid];
  FEM_remove_element(m, localIdx, elemtype);
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
    sharedIdx = FEM_To_ghost_index(sharedIdx);
  }
  else if(type == 3) { //ghost node recv 
    ll = m->elem[elemType].ghostSend.getList(chk);
  }
  else if(type == 4) { //ghost node recv 
    ll = m->elem[elemType].ghost->ghostRecv.getList(chk);
    sharedIdx = FEM_To_ghost_index(sharedIdx);
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

femMeshModify::femMeshModify(femMeshModMsg *fm) {
  numChunks = fm->numChunks;
  idx = fm->myChunk;
  fmLock = new FEM_lock(idx, this);
  fmUtil = new FEM_MUtil(idx, this);
  fmAdapt = NULL;
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
  fmInp = new FEM_Interpolate(fmMesh, this);
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
  return;
}

void femMeshModify::addGhostElem(addGhostElemMsg *fm) {
  fmUtil->addGhostElementRemote(fmMesh, fm->chk, fm->elemType, fm->numGhostIndex, fm->ghostIndices, fm->numSharedIndex, fm->sharedIndices, fm->connSize);
  return;
}

chunkListMsg *femMeshModify::getChunksSharingGhostNode(int2Msg *i2m) {
  chunkListMsg *clm;
  clm = fmUtil->getChunksSharingGhostNodeRemote(fmMesh, i2m->i, i2m->j);
  return clm;
}

void femMeshModify::addElementRemote(addElemMsg *fm) {
  fmUtil->addElemRemote(fmMesh, fm->chk, fm->elemtype, fm->connSize, fm->conn, fm->numGhostIndex, fm->ghostIndices);
  return;
}

void femMeshModify::removeGhostElem(removeGhostElemMsg *fm) {
  fmUtil->removeGhostElementRemote(fmMesh, fm->chk, fm->elementid, fm->elemtype, fm->numGhostIndex, fm->ghostIndices);
  return;
}

void femMeshModify::removeElementRemote(removeElemMsg *fm) {
  fmUtil->removeElemRemote(fmMesh, fm->chk, fm->elementid, fm->elemtype);
  return;
}

void femMeshModify::refine_flip_element_leb(int fromChk, int propElemT, 
					    int propNodeT, int newNodeT, 
					    int nbrOpNodeT, double longEdgeLen)
{
  if (fromChk == getfmUtil()->getIdx()) { // no translation necessary
    fmAdapt->refine_flip_element_leb(propElemT, propNodeT, newNodeT, 
				     nbrOpNodeT, longEdgeLen);
  }
  else {
    int propElem, propNode, newNode, nbrOpNode;
    propElem = getfmUtil()->lookup_in_IDXL(fmMesh, propElemT, fromChk, 3, 0);
    propNode = getfmUtil()->lookup_in_IDXL(fmMesh, propElemT, fromChk, 0, -1);
    newNode = getfmUtil()->lookup_in_IDXL(fmMesh, newNodeT, fromChk, 0, -1);
    nbrOpNode = getfmUtil()->lookup_in_IDXL(fmMesh, nbrOpNodeT, fromChk, 1,-1);
    fmAdapt->refine_flip_element_leb(propElem, propNode, newNode, nbrOpNode, 
				     longEdgeLen);
  }
}

#include "FEMMeshModify.def.h"
