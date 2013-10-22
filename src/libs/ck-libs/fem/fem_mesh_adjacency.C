/***************************************************
 * fem_mesh_adjacency.C
 *
 * All the adjacency creation and update functions.
 * The first half of the file is for initial creation
 * of all needed tables. The second half provides 
 * easy accessor and modifier functions.
 *
 * Authors include: Isaac, Sayantan, Terry, Nilesh
 */

#include "fem.h"
#include "fem_impl.h"
#include "charm-api.h" /*for CDECL, FTN_NAME*/



//#define DEBUG


#ifdef DEBUG
FORTRAN_AS_C(CMIMEMORYCHECK,
             CmiMemoryCheck,
             cmimemorycheck, 
             (void),  () )
#endif




CDECL void 
FEM_Mesh_create_node_elem_adjacency(int fem_mesh){
	const char *caller="FEM_Mesh_create_node_elem_adjacency"; FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	m->createNodeElemAdj();
}
FORTRAN_AS_C(FEM_MESH_CREATE_NODE_ELEM_ADJACENCY,
             FEM_Mesh_create_node_elem_adjacency,
             fem_mesh_create_node_elem_adjacency, 
             (int *fem_mesh),  (*fem_mesh) )


CDECL void 
FEM_Mesh_create_node_node_adjacency(int fem_mesh){
	const char *caller="FEM_Mesh_create_node_node_adjacency"; FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	m->createNodeNodeAdj();
}
FORTRAN_AS_C(FEM_MESH_CREATE_NODE_NODE_ADJACENCY,
             FEM_Mesh_create_node_node_adjacency,
             fem_mesh_create_node_node_adjacency, 
             (int *fem_mesh),  (*fem_mesh) )


CDECL void 
FEM_Mesh_create_elem_elem_adjacency(int fem_mesh){
	const char *caller="FEM_Mesh_create_elem_elem_adjacency"; FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	m->createElemElemAdj();
}
FORTRAN_AS_C(FEM_MESH_CREATE_ELEM_ELEM_ADJACENCY,
             FEM_Mesh_create_elem_elem_adjacency,
             fem_mesh_create_elem_elem_adjacency, 
             (int *fem_mesh),  (*fem_mesh) )



CDECL void 
FEM_Mesh_create_elem_node_adjacency(int fem_mesh){
  CkPrintf("WARNING: Do Not Call FEM_Mesh_create_elem_node_adjacency(), as the connectivity table should already exist\n");
}


CDECL void 
FEM_Mesh_get2ElementsOnEdge(int fem_mesh, int n1, int n2, int *e1, int *e2){
	const char *caller="FEM_Mesh_get2ElementsOnEdge"; FEMAPI(caller);
	FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
	m->get2ElementsOnEdge(n1, n2, e1, e2);
}
FORTRAN_AS_C(FEM_MESH_GET2ELEMENTSONEDGE,
             FEM_Mesh_get2ElementsOnEdge,
             fem_mesh_get2elementsonedge, 
             (int *fem_mesh, int *n1, int *n2, int *e1, int *e2),  
             (*fem_mesh,*n1,*n2,e1,e2) )



void FEM_Node::allocateElemAdjacency(){
#ifdef DEBUG
  CmiMemoryCheck();
#endif
	if(elemAdjacency){
		delete elemAdjacency;
	}
	elemAdjacency = new FEM_VarIndexAttribute(this,FEM_NODE_ELEM_ADJACENCY);
	add(elemAdjacency);
}

void FEM_Node::allocateNodeAdjacency(){
#ifdef DEBUG
  CmiMemoryCheck();
#endif
	if(nodeAdjacency){
		delete nodeAdjacency;
	}
	nodeAdjacency = new FEM_VarIndexAttribute(this,FEM_NODE_NODE_ADJACENCY);
	add(nodeAdjacency);
}


//  Fill the node to element adjacency table for both this element and its corresponding ghosts
void FEM_Node::setElemAdjacency(int type, const FEM_Elem &elem){
	int nodesPerElem = elem.getNodesPer();
	FEM_VarIndexAttribute *adjacencyAttr = elemAdjacency;
	CkVec<CkVec<var_id> > &adjacencyTable = elemAdjacency->get();
	FEM_VarIndexAttribute *ghostAdjacencyAttr = ((FEM_Node *)getGhost())->elemAdjacency;
	CkVec<CkVec<var_id> > &ghostAdjacencyTable = ghostAdjacencyAttr->get();

	// Scan through elements
	for(int i=0;i<elem.size();i++){
	  const int *conn = elem.connFor(i);
	  for(int j=0;j<nodesPerElem;j++){
		int node = conn[j];
		if (node!=-1){
		  if(FEM_Is_ghost_index(node)){
			int idx = ghostAdjacencyAttr->findInRow(FEM_To_ghost_index(node),var_id(type,i));
			if(idx == -1) {// If not currently in the adjacency list, push onto list
			  ghostAdjacencyTable[FEM_To_ghost_index(node)].push_back(var_id(type,i));
			}
		  }
		  else{
			int idx = adjacencyAttr->findInRow(node,var_id(type,i));
			if(idx == -1) {// If not currently in the adjacency list, push onto list
			  adjacencyTable[node].push_back(var_id(type,i));
			}
		  }
		}
	  }
	}
	
	// Scan through ghost elements
	if(elem.getGhost()){
	  for(int i=0;i<((FEM_Elem*)elem.getGhost())->size();i++){
		const int *conn = ((FEM_Elem*)elem.getGhost())->connFor(i);
		for(int j=0;j<nodesPerElem;j++){
		  int node = conn[j];
		  if (node!=-1){
			if(FEM_Is_ghost_index(node)){
			  int idx = ghostAdjacencyAttr->findInRow(FEM_To_ghost_index(node),var_id(type,FEM_From_ghost_index(i)));
			  if(idx == -1){ // If not currently in the adjacency list, push onto list
				ghostAdjacencyTable[FEM_To_ghost_index(node)].push_back(var_id(type,FEM_From_ghost_index(i)));
			  }
			}
			
			else{
			  int idx = adjacencyAttr->findInRow(node,var_id(type,FEM_From_ghost_index(i)));
			  if(idx == -1){// If not currently in the adjacency list, push onto list
				adjacencyTable[node].push_back(var_id(type,FEM_From_ghost_index(i)));
			  }
			}
		  }
		}
	  }
	}
}


//  Populate the entire node to node adjacency table
//  Two nodes are considered adjacent if they both are in the connectivity table for a common element.
//  This choice for definition of adjacent nodes does not take into account what are edges 
//  of the element, but it does simplify the computation. It will work fine for 
//  triangles and tetrahedra, but may not make as much sense for more complicated
//  element types where all nodes are not directly connected by edges.
void FEM_Node::setNodeAdjacency(const FEM_Elem &elem){
  //CkPrintf("In FEM_Node::setNodeAdjacency()\n");
  int nodesPerElem = elem.getNodesPer();
  CkVec<CkVec<var_id> > &adjacencyTable = nodeAdjacency->get();
  FEM_VarIndexAttribute *ghostAdjacencyAttr = ((FEM_Node *)getGhost())->nodeAdjacency;
  CkVec<CkVec<var_id> > &ghostAdjacencyTable = ghostAdjacencyAttr->get();
  
#ifdef DEBUG
  CmiMemoryCheck();
#endif

  // Add the adjacencies defined by the non-ghost elements
  for(int i=0;i<elem.size();i++) {        // for each element of the given type
	const int *conn = elem.connFor(i);
	for(int j=0;j<nodesPerElem;j++){   // for each node adjacent to the element
	  const int nodej = conn[j];
	  if (nodej!=-1){
		if(FEM_Is_ghost_index(nodej)) { // A ghost node
		  for(int k=0;k<nodesPerElem;k++){
			const int nodek=conn[k];
			if(nodek != nodej){
			  var_id nodeID = var_id::createNodeID(1,nodek);
			  int idx = ghostAdjacencyAttr->findInRow(FEM_To_ghost_index(nodej),nodeID);
			  if(idx == -1){
				//if(nodej==-5|| nodek==-5) CkPrintf("G %d->%d not found adding\n", nodej, nodek);
				ghostAdjacencyTable[FEM_To_ghost_index(nodej)].push_back(nodeID);
			  }
			  {
				//if(nodej==-5|| nodek==-5) CkPrintf("G %d->%d found already\n", nodej, nodek);
			  }
			}
		  }
		}
		else { // A non-ghost node, almost same as for ghost nodes
		  for(int k=0;k<nodesPerElem;k++){
			const int nodek=conn[k];
			if(nodek != nodej){
			  var_id nodeID = var_id::createNodeID(1,nodek);
			  int idx = nodeAdjacency->findInRow(nodej,nodeID);
			  if(idx == -1){
				//if(nodej==-5|| nodek==-5) CkPrintf("NG %d->%d not found--adding\n", nodej, nodek);
				adjacencyTable[nodej].push_back(nodeID);
			  }
			  {
				//if(nodej==-5 || nodek==-5) CkPrintf("NG %d->%d found already\n", nodej, nodek);
			  }
			}
		  }
		}
	  }
	}
  }


  for(int i=0;i<((FEM_Elem*)elem.getGhost())->size();i++) {        // for each element of the given type
	const int *conn = ((FEM_Elem*)elem.getGhost())->connFor(i);
	for(int j=0;j<nodesPerElem;j++){   // for each node adjacent to the element
	  const int nodej = conn[j];
	  if (nodej!=-1){
		if(FEM_Is_ghost_index(nodej)) { // A ghost node
		  for(int k=0;k<nodesPerElem;k++){
			const int nodek=conn[k];
			if(nodek != nodej){
			  var_id nodeID = var_id::createNodeID(1,nodek);
			  int idx = ghostAdjacencyAttr->findInRow(FEM_To_ghost_index(nodej),nodeID);
			  if(idx == -1){
				//if(nodej==-5|| nodek==-5) CkPrintf("G-G %d->%d not found adding\n", nodej, nodek);
				ghostAdjacencyTable[FEM_To_ghost_index(nodej)].push_back(nodeID);
			  }
			  {
				//if(nodej==-5|| nodek==-5) CkPrintf("G-G %d->%d found already\n", nodej, nodek);
			  }
			}
		  }
		}
		else { // A non-ghost node, almost same as for ghost nodes
		  for(int k=0;k<nodesPerElem;k++){
			const int nodek=conn[k];
			if(nodek != nodej){
			  var_id nodeID = var_id::createNodeID(1,nodek);
			  int idx = nodeAdjacency->findInRow(nodej,nodeID);
			  if(idx == -1){
				//if(nodej==-5|| nodek==-5) CkPrintf("G-NG %d->%d not found--adding\n", nodej, nodek);
				adjacencyTable[nodej].push_back(nodeID);
			  }
			  {
				//if (nodej==-5 || nodek==-5) CkPrintf("G-NG %d->%d found already\n", nodej, nodek);
			  }
			}
		  }
		}
	  }
	}
  }


#ifdef DEBUG
  CmiMemoryCheck();
#endif


}



// Allocate both the FEM_ELEM_ELEM_ADJACENCY and FEM_ELEM_ELEM_ADJ_TYPES attributes
// The values in these two attributes will be generated by 
void FEM_Elem::allocateElemAdjacency(){

#ifdef DEBUG
  CmiMemoryCheck();
#endif

  if(elemAdjacency){
	CkPrintf("FEM> WARNING: Deleting previously allocated(?) elemAdjacency array. allocateElemAdjacency() should probably only be called once.\n");
    delete elemAdjacency;
  }
  if(elemAdjacencyTypes){
   	CkPrintf("FEM> WARNING: Deleting previously allocated(?) elemAdjacencyTypes array. allocateElemAdjacency() should probably only be called once.\n");
	delete elemAdjacencyTypes;
  }

  elemAdjacency = new FEM_IndexAttribute(this,FEM_ELEM_ELEM_ADJACENCY);
  elemAdjacencyTypes = new FEM_IndexAttribute(this,FEM_ELEM_ELEM_ADJ_TYPES);

  elemAdjacency->setLength(size());
  elemAdjacency->setWidth(conn->getWidth());
  elemAdjacencyTypes->setLength(size());
  elemAdjacencyTypes->setWidth(conn->getWidth());
	
  add(elemAdjacency);
  add(elemAdjacencyTypes);

#ifdef DEBUG
  CmiMemoryCheck();
#endif

}


void FEM_Mesh::createNodeElemAdj(){
	node.lookup(FEM_NODE_ELEM_ADJACENCY,"FEM_Mesh::createElemNodeAdj");
	for(int i=0;i<elem.size();i++){
		node.setElemAdjacency(i,elem[i]);
	}
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

void FEM_Mesh::createNodeNodeAdj(){
	node.lookup(FEM_NODE_NODE_ADJACENCY,"FEM_Mesh::createNodeNodeAdj");
	for(int i=0;i<elem.size();i++){
		node.setNodeAdjacency(elem[i]);
	}
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}


/* A created on demand data structure that stores
 * the number of tuples(or faces) for each type of element
 * along with the number of nodes in each tuple.
 * The number of nodes in each tuple depends upon how many
 * nodes are in a shared face(an edge will have 2, a triangle 3)
 */
FEM_ElemAdj_Layer* FEM_Mesh::getElemAdjLayer(void) {
  if (! lastElemAdjLayer) 
	lastElemAdjLayer=new FEM_ElemAdj_Layer();
  return lastElemAdjLayer;
}


/* createElemElemAdj() is similar to splitter::addTuple()
 * It will scan through all tuples/faces for each element
 * and register them with a tupleTable which will find 
 * matching tuples between different elements. The matches  
 * are then extracted from the table and the resulting
 * adjacencies are marked in the FEM_ELEM_ELEM_ADJACENCY
 * and FEM_ELEM_ELEM_ADJ_TYPES attribute fields. The attributes
 * store the adjacent element's local id and type respectively.
 * The ordering in the attribute tables is by local element id,
 * as usual, and is then ordered by the tuple ordering specified
 * in the last parameter to FEM_Add_elem2face_tuples, which 
 * gets stored in the FEM_ElemAdj_Layer data structure
 * created/returned by getElemAdjLayer().
 *
 * The element id's are NOT indices into the conn array in 
 * the case of ghost elements. Ghost elements will have the
 * same element type as their corresponding real elements.
 * Their id, which gets stored in FEM_ELEM_ELEM_ADJACENCY
 * will be a negative number which can be converted to an index
 * with FEM_To_ghost_index(). Thus the user MUST use 
 * FEM_Is_ghost_index(i) on the values, before accessing 
 * them in the conn array, especially since the ghosts
 * will have a negative id.
 *
 * The function assumes that the FEM_ELEM_ELEM_ADJACENCY
 * and FEM_ELEM_ELEM_ADJ_TYPES attribute fields already exist.
 *
 * TODO:
 *  
 *   Verify the tuple table does not need to be 
 *   explicitly deleted.
 */
void FEM_Mesh::createElemElemAdj()
{
  FEM_ElemAdj_Layer *g = getElemAdjLayer();
  if(! g->initialized)
	CkAbort("FEM> Cannot call FEM_Mesh_create_elem_elem_adjacency() before registering tuples with FEM_Add_elem2face_tuples()\n");

  const int nodesPerTuple = g->nodesPerTuple;
  tupleTable table(nodesPerTuple);

#ifdef DEBUG
  CmiMemoryCheck();
#endif

  // Put tuples into table
  for (int t=0;t<elem.size();t++){ // for each element type
      if(elem.has(t)) {
          const int tuplesPerElem = g->elem[t].tuplesPerElem;
          const int numElements = elem[t].size();
          // for every element of  type t:
          for (int elemIndex=0;elemIndex<numElements;elemIndex++)	{
              // insert element into the tuple table
              const int *conn=elem[t].connFor(elemIndex);
              int tuple[tupleTable::MAX_TUPLE];
              FEM_Symmetries_t allSym;
              // copy node numbers into tuple
              for (int u=0;u<tuplesPerElem;u++) {
                  for (int i=0;i<nodesPerTuple;i++) {
                    int eidx=g->elem[t].elem2tuple[i+u*g->nodesPerTuple];
                    if (eidx==-1)    // "not-there" node
                      tuple[i]=-1;   // Don't map via connectivity
                    else             // Ordinary node
                      tuple[i]=conn[eidx]; 
                  }
                  // add tuple to table
                  table.addTuple(tuple,new elemList(0,elemIndex,t,allSym,u)); 
              }
          }
      
          // Put corresponding ghost elements into tuple table
          if(elem[t].getGhost() != NULL){
              FEM_Elem *ghostElem = (FEM_Elem *)elem[t].getGhost();
              const int numElements = ghostElem->size();
              // for every element of  type t:
              for (int elemIndex=0;elemIndex<numElements;elemIndex++)	{
                  // insert element into the tuple table
                  const int *conn=ghostElem->connFor(elemIndex);
                  int tuple[tupleTable::MAX_TUPLE];
                  FEM_Symmetries_t allSym;
                  // copy node numbers into tuple
                  for (int u=0;u<tuplesPerElem;u++) {
                      for (int i=0;i<nodesPerTuple;i++) {
                       
                          int eidx=g->elem[t].elem2tuple[i+u*g->nodesPerTuple];
                          if (eidx==-1) { //"not-there" node--
                              tuple[i]=-1; //Don't map via connectivity
                          } else { //Ordinary node
                              int n=conn[eidx];
                              tuple[i]=n; 
                          }
                      }
                      // add tuple to table
                      table.addTuple(tuple,new elemList(0,FEM_From_ghost_index(elemIndex),t,allSym,u)); 
                  }
              }
          }
          
      }
  }
  

#ifdef DEBUG
  CmiMemoryCheck();
#endif


  /* Extract adjacencies from table and store into 
   * FEM_ELEM_ELEM_ADJACENCY and FEM_ELEM_ELEM_ADJ_TYPES 
   * attribute fields
   */
  for (int t=0;t<elem.size();t++) // for each element type t
      if (elem.has(t)) {
        elemList *l;
        const int tuplesPerElem = g->elem[t].tuplesPerElem;
        const int numElements = elem[t].size();
        const int numGhostElements = ((FEM_Elem*)(elem[t].getGhost()))->size();
        table.beginLookup();
        
        // directly modify the element adjacency table for element type t
        FEM_IndexAttribute *elemAdjTypesAttr = (FEM_IndexAttribute *)elem[t].lookup(FEM_ELEM_ELEM_ADJ_TYPES,"createElemElemAdj");
        FEM_IndexAttribute *elemAdjAttr = (FEM_IndexAttribute *)elem[t].lookup(FEM_ELEM_ELEM_ADJACENCY,"createElemElemAdj");
        FEM_IndexAttribute *elemAdjTypesAttrGhost = (FEM_IndexAttribute *)elem[t].getGhost()->lookup(FEM_ELEM_ELEM_ADJ_TYPES,"createElemElemAdj");
        FEM_IndexAttribute *elemAdjAttrGhost = (FEM_IndexAttribute *)elem[t].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"createElemElemAdj");
        CkAssert(elemAdjTypesAttr && elemAdjAttr);
        
        AllocTable2d<int> &adjTable = elemAdjAttr->get();
        int *adjs = adjTable.getData();
        AllocTable2d<int> &adjTypesTable = elemAdjTypesAttr->get();
        int *adjTypes = adjTypesTable.getData();
        
        AllocTable2d<int> &adjTableGhost = elemAdjAttrGhost->get();
        int *adjsGhost = adjTableGhost.getData();
        AllocTable2d<int> &adjTypesTableGhost = elemAdjTypesAttrGhost->get();
        int *adjTypesGhost = adjTypesTableGhost.getData();
        
        // initialize tables
        for(int i=0;i<numElements*tuplesPerElem;i++){
          adjs[i]=-1;
          adjTypes[i]=0;
        }
        for(int i=0;i<numGhostElements*tuplesPerElem;i++){
          adjsGhost[i]=-1;
          adjTypesGhost[i]=0;
        }
        
        // look through each elemList that is returned by the tuple table
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
                  if(a->type == t){ // only update the entries for element type t
                    
                    if(FEM_Is_ghost_index(a->localNo)){
                      j = FEM_To_ghost_index(a->localNo)*tuplesPerElem + a->tupleNo;
                      CkAssert(j<numGhostElements*tuplesPerElem);
                      adjsGhost[j] = b->localNo;
                      adjTypesGhost[j] = b->type;
                    }
                    else {
                      j = a->localNo*tuplesPerElem + a->tupleNo;
                      CkAssert(j<numElements*tuplesPerElem);
                      adjs[j] = b->localNo;
                      adjTypes[j] = b->type;
                    }
                  }
                  
                }
              }
            }
          }
        }
      }

#ifdef DEBUG
  CmiMemoryCheck();
#endif

  delete g;
}



//  ------- Element-to-element: preserve initial ordering relative to nodes
/// Place all of element e's adjacent elements in neighbors; assumes
/// neighbors allocated to correct size
void FEM_Mesh::e2e_getAll(int e, int *neighbors, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return; // non existent element
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e)){
    eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getAll");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      neighbors[i] = eAdjs[FEM_To_ghost_index(e)][i];
    }
  }
  else {
    eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getAll");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      neighbors[i] = eAdjs[e][i];
    }
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Given id of element e of type etype, return the id of the idx-th adjacent element
int FEM_Mesh::e2e_getNbr(int e, short idx, int etype) 
{     
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return -1;
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e)){
    eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getNbr");
    AllocTable2d<int> &eAdjs = eAdj->get();
    return eAdjs[FEM_To_ghost_index(e)][idx];
  }
  else {
    eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getNbr");
    AllocTable2d<int> &eAdjs = eAdj->get();
    return eAdjs[e][idx];
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Given id of element e and id of another element nbr, return i such that
/// nbr is the i-th element adjacent to e
int FEM_Mesh::e2e_getIndex(int e, int nbr, int etype) 
{ 
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return -1;
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e)){
    eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getIndex");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      if (eAdjs[FEM_To_ghost_index(e)][i] == nbr) {
        return i;
      }
    }
  }
  else{
    eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getIndex");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      if (eAdjs[e][i] == nbr) {
        return i;
      }
    }
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  return -1;
}

/// Set the element adjacencies of element e to neighbors; assumes neighbors 
/// has the correct size
void FEM_Mesh::e2e_setAll(int e, int *neighbors, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e)){
    eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_setAll");
    AllocTable2d<int> &eAdjs = eAdj->get();
    //CkPrintf("e2e_setAll: Setting element %d's neighbors to %d,%d,%d\n", e, 
    //neighbors[0], neighbors[1], neighbors[2]);
    for (int i=0; i<eAdjs.width(); i++) {
      eAdjs[FEM_To_ghost_index(e)][i] = neighbors[i];
    }
  }
  else{
    eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_setAll");
    AllocTable2d<int> &eAdjs = eAdj->get();
    //CkPrintf("e2e_setAll: Setting element %d's neighbors to %d,%d,%d\n", e, 
    //neighbors[0], neighbors[1], neighbors[2]);
    for (int i=0; i<eAdjs.width(); i++) {
      eAdjs[e][i] = neighbors[i];
    }
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  
}

/// Set the idx-th element adjacent to e to be newElem
void FEM_Mesh::e2e_setIndex(int e, short idx, int newElem, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e)){
    eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_setIndex");
    AllocTable2d<int> &eAdjs = eAdj->get();
    eAdjs[FEM_To_ghost_index(e)][idx] = newElem;
  }
  else {
    eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_setIndex");    
    AllocTable2d<int> &eAdjs = eAdj->get();
    eAdjs[e][idx] = newElem;
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Find element oldNbr in e's adjacent elements and replace with newNbr
void FEM_Mesh::e2e_replace(int e, int oldNbr, int newNbr, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e)){
    eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_replace");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      if (eAdjs[FEM_To_ghost_index(e)][i] == oldNbr) {
        eAdjs[FEM_To_ghost_index(e)][i] = newNbr;
        break;
      }
    }
  }
  else{
    eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_replace");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      if (eAdjs[e][i] == oldNbr) {
        eAdjs[e][i] = newNbr;
        break;
      }
    }
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Remove all neighboring elements in adjacency
void FEM_Mesh::e2e_removeAll(int e, int etype)
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e)) {
    eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_removeAll");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      eAdjs[FEM_To_ghost_index(e)][i] = -1;
    }
  }
  else {
    eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_removeAll");
    AllocTable2d<int> &eAdjs = eAdj->get();
    for (int i=0; i<eAdjs.width(); i++) {
      eAdjs[e][i] = -1;
    }
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

//  ------- Element-to-node: preserve initial ordering
/// Place all of element e's adjacent nodes in adjnodes; assumes
/// adjnodes allocated to correct size
void FEM_Mesh::e2n_getAll(int e, int *adjnodes, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  if(FEM_Is_ghost_index(e)){
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++) {
      adjnodes[i] = conn[FEM_To_ghost_index(e)][i];
    }
  }
  else{
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++) {
      adjnodes[i] = conn[e][i];
    }
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Given id of element e, return the id of the idx-th adjacent node
int FEM_Mesh::e2n_getNode(int e, short idx, int etype) 
{ 
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return -1;
  if(FEM_Is_ghost_index(e)){
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    return conn[FEM_To_ghost_index(e)][idx];
  }
  else{
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    return conn[e][idx];
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Given id of element e and id of a node n, return i such that
/// n is the i-th node adjacent to e
short FEM_Mesh::e2n_getIndex(int e, int n, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return -1;
  if(FEM_Is_ghost_index(e)){
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++)
      if (conn[FEM_To_ghost_index(e)][i] == n) 
        return i;
  }
  else{
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++)
      if (conn[e][i] == n)
        return i;
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  return -1;
}

/// Set the node adjacencies of element e to adjnodes; assumes adjnodes 
/// has the correct size
void FEM_Mesh::e2n_setAll(int e, int *adjnodes, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  if(FEM_Is_ghost_index(e)){
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++) {
      conn[FEM_To_ghost_index(e)][i] = adjnodes[i];
    }
  }
  else{
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++) {
      conn[e][i] = adjnodes[i];
    }
  }  
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Set the idx-th node adjacent to e to be newNode
void FEM_Mesh::e2n_setIndex(int e, short idx, int newNode, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  if(FEM_Is_ghost_index(e)){
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    conn[FEM_To_ghost_index(e)][idx] = newNode;
  }
  else{
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    conn[e][idx] = newNode;
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Find node oldNode in e's adjacent ndoes and replace with newNode
void FEM_Mesh::e2n_replace(int e, int oldNode, int newNode, int etype) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  if(FEM_Is_ghost_index(e)){
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++) {
      if (conn[FEM_To_ghost_index(e)][i] == oldNode) {
        conn[FEM_To_ghost_index(e)][i] = newNode;
        break;
      }
    }
  }
  else{
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++) {
      if (conn[e][i] == oldNode) {
        conn[e][i] = newNode;
        break;
      }
    }
  }  
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

void FEM_Mesh::e2n_removeAll(int e, int etype)
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (e == -1) return;
  if(FEM_Is_ghost_index(e)){
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++)
	  conn[FEM_To_ghost_index(e)][i] = -1;
  }
  else{
    FEM_IndexAttribute *eConn = (FEM_IndexAttribute *)elem[etype].lookup(FEM_CONN,"e2n_getAll");
    AllocTable2d<int> &conn = eConn->get();
    for (int i=0; i<conn.width(); i++)
	  conn[e][i] = -1;
  }  
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}




//  ------- Node-to-node
/// Place all of node n's adjacent nodes in adjnodes and the resulting 
/// length of adjnodes in sz; assumes adjnodes is not allocated, but sz is
void FEM_Mesh::n2n_getAll(int n, int **adjnodes, int *sz) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) {
    *sz = 0;
    return;
  }
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_getAll");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[FEM_To_ghost_index(n)];
	*sz = nsVec.length();
	if(*sz != 0) (*adjnodes) = new int[*sz];
	for (int i=0; i<(*sz); i++) {
	  (*adjnodes)[i] = nsVec[i].getSignedId();
	}
  }
  else{
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_getAll");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[n];
	*sz = nsVec.length();
	if(*sz != 0) (*adjnodes) = new int[*sz];
	for (int i=0; i<(*sz); i++) {
	  (*adjnodes)[i] = nsVec[i].getSignedId();
	}
  }
  
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}
 
/// Adds newNode to node n's node adjacency list
void FEM_Mesh::n2n_add(int n, int newNode) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_add");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	FEM_VarIndexAttribute::ID nn(0, newNode);
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[FEM_To_ghost_index(n)];
	nsVec.push_back(nn);
  }
  else{
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_add");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	FEM_VarIndexAttribute::ID nn(0, newNode);
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[n];
	nsVec.push_back(nn);
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}




/// Removes oldNode from n's node adjacency list
void FEM_Mesh::n2n_remove(int n, int oldNode) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_remove");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec.remove(i);
		break;
	  }
	}
  }
  else {
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_remove");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec.remove(i);
		break;
	  }
	}
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Is queryNode in node n's adjacency vector?
int FEM_Mesh::n2n_exists(int n, int queryNode) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return 0;
  if(FEM_Is_ghost_index(n)){
	CkAssert(node.getGhost());
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_exists");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++)
	  if (nsVec[i].getSignedId() == queryNode) 
		return 1;
  }
  else {
    FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_exists");
    CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
    CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[n];
    for (int i=0; i<nsVec.length(); i++)
      if (nsVec[i].getSignedId() == queryNode) 
	return 1;
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  return 0;
}

/// Finds oldNode in n's node adjacency list, and replaces it with newNode
void FEM_Mesh::n2n_replace(int n, int oldNode, int newNode) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_replace");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec[i] = FEM_VarIndexAttribute::ID(0,newNode);
		break;
	  }
	}
  }
  else {
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_replace");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec[i] = FEM_VarIndexAttribute::ID(0,newNode);
		break;
	  }
	}

  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Remove all nodes from n's node adjacency list
void FEM_Mesh::n2n_removeAll(int n)
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_removeAll");  
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[FEM_To_ghost_index(n)];
	nsVec.free();
  }
  else{
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_removeAll");  
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &nVec = nAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = nVec[n];
	nsVec.free();
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

//  ------- Node-to-element
/// Place all of node n's adjacent elements in adjelements and the resulting 
/// length of adjelements in sz; assumes adjelements is not allocated, 
/// but sz is
void FEM_Mesh::n2e_getAll(int n, int **adjelements, int *sz) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif


  if (n == -1) {
    *sz = 0;
    return;
  }
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getAll");  
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[FEM_To_ghost_index(n)];
	*sz = nsVec.length();
	if(*sz !=0) (*adjelements) = new int[*sz];
	for (int i=0; i<(*sz); i++) {
	  (*adjelements)[i] = nsVec[i].getSignedId();
	}
  }
  else {
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getAll");  
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[n];
  int len = nsVec.length();
	*sz = len;
	if(*sz !=0) (*adjelements) = new int[*sz];
	for (int i=0; i<(*sz); i++) {
	  (*adjelements)[i] = nsVec[i].getSignedId();
	}
  }

#ifdef DEBUG
  CmiMemoryCheck();
#endif

}
 
/// Adds newElem to node n's element adjacency list
void FEM_Mesh::n2e_add(int n, int newElem) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif

  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_add");     
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[FEM_To_ghost_index(n)];
	FEM_VarIndexAttribute::ID ne(0, newElem);
	nsVec.push_back(ne);
	int *testn2e, testn2ec;
	n2e_getAll(n,&testn2e,&testn2ec);
	for(int i=0; i<testn2ec; i++) {
	  if(FEM_Is_ghost_index(testn2e[i]))
	    CkAssert(elem[0].ghost->is_valid(FEM_From_ghost_index(testn2e[i])));
	  else 
	    CkAssert(elem[0].is_valid(testn2e[i]));
	}
	if(testn2ec!=0) delete[] testn2e;
  }
  else {
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_add");     
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[n];
	FEM_VarIndexAttribute::ID ne(0, newElem);
	nsVec.push_back(ne);
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif

}

/// Removes oldElem from n's element adjacency list
void FEM_Mesh::n2e_remove(int n, int oldElem) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_remove");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec.remove(i);
		break;
	  }
	}
  }
  else {
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_remove");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec.remove(i);
		break;
	  }
	}
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}
 
/// Finds oldElem in n's element adjacency list, and replaces it with newElem
void FEM_Mesh::n2e_replace(int n, int oldElem, int newElem) 
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_replace");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec[i] = FEM_VarIndexAttribute::ID(0,newElem);
		break;
	  }
	}
	int *testn2e, testn2ec;
	n2e_getAll(n,&testn2e,&testn2ec);
	for(int i=0; i<testn2ec; i++) {
	  if(FEM_Is_ghost_index(testn2e[i]))
	    CkAssert(elem[0].ghost->is_valid(FEM_From_ghost_index(testn2e[i])));
	  else 
	    CkAssert(elem[0].is_valid(testn2e[i]));
	}
	if(testn2ec!=0) delete[] testn2e;
  }
  else{
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_replace");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec[i] = FEM_VarIndexAttribute::ID(0,newElem);
		break;
	  }
	}
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Remove all elements from n's element adjacency list
void FEM_Mesh::n2e_removeAll(int n)
{
#ifdef DEBUG
  CmiMemoryCheck();
#endif
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_removeAll");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[FEM_To_ghost_index(n)];
	nsVec.free();
  }
  else{
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_removeAll");
	CkVec<CkVec<FEM_VarIndexAttribute::ID> > &eVec = eAdj->get();
	CkVec<FEM_VarIndexAttribute::ID> &nsVec = eVec[n];
	nsVec.free();
  }
#ifdef DEBUG
  CmiMemoryCheck();
#endif
}

/// Get an element on edge (n1, n2) where n1, n2 are chunk-local
/// node numberings; return -1 in case of failure

// I think the break statement may cause some unexpected behaviour here
// The break will break out of the inner for loop back to the outer for loop
// I am not sure exactly what the symantics of this function should be
//   -- Isaac

int FEM_Mesh::getElementOnEdge(int n1, int n2) 
{
  int *n1AdjElems, *n2AdjElems;
  int n1NumElems, n2NumElems;
  n2e_getAll(n1, &n1AdjElems, &n1NumElems);
  n2e_getAll(n2, &n2AdjElems, &n2NumElems);
  int ret = -1;
  //CkPrintf("%d has %d neighboring elements, %d has %d\n", n1, n1NumElems, n2, n2NumElems);

#ifdef DEBUG
  CmiMemoryCheck();
#endif 

  for (int i=0; i<n1NumElems; i++) {
    for (int j=0; j<n2NumElems; j++) {
      if (n1AdjElems[i] == n2AdjElems[j]) {
        if(n1AdjElems[i] >= 0) {
          ret = n1AdjElems[i];
          break;
        }
        else {
          ret = n1AdjElems[i];
        }
      }
    }
  }
  delete[] n1AdjElems;
  delete[] n2AdjElems;
  return ret; //preferably return a local element, otherwise return a ghost 
}


/// Get 2 elements on edge (n1, n2) where n1, n2 are chunk-local
/// node numberings; return the edges in result_e1 and result_e2
/// No preference is given to ghosts over local elements
void FEM_Mesh::get2ElementsOnEdge(int n1, int n2, int *result_e1, int *result_e2) 
{
  int *n1AdjElems=0, *n2AdjElems=0;
  int n1NumElems, n2NumElems;

#ifdef DEBUG
  CmiMemoryCheck();
#endif

  n2e_getAll(n1, &n1AdjElems, &n1NumElems);
  n2e_getAll(n2, &n2AdjElems, &n2NumElems);
  CkAssert(n1AdjElems!=0);
  CkAssert(n2AdjElems!=0);
  int found=0;

#ifdef DEBUG
  CmiMemoryCheck();
#endif

  *result_e1=-1;
  *result_e2=-1;

  for (int i=0; i<n1NumElems; i++) {
    for (int j=0; j<n2NumElems; j++) {
      if (n1AdjElems[i] == n2AdjElems[j]) {
        if(found==0){
          //          CkPrintf("found element1 %d\n", n1AdjElems[i]);
          *result_e1 = n1AdjElems[i];
          found++;
        }
        else if(found==1){
          //   CkPrintf("found element2 %d\n", n1AdjElems[i]);
          *result_e2 = n1AdjElems[i];
          found++;
        }
        else {
#ifdef DEBUG
          CmiMemoryCheck();
#endif          
          CkPrintf("ERROR: Found a third element(%d) on edge %d,%d \n", n1AdjElems[i], n1, n2);
           CkExit();
        }
      }
    }
  }


#ifdef DEBUG
  CmiMemoryCheck();
#endif 
  delete[] n1AdjElems;
  delete[] n2AdjElems;
}

