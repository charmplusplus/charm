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

#include "ParFUM.h"
#include "ParFUM_internals.h"


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


CDECL void
FEM_Mesh_get2ElementsOnEdgeSorted(int fem_mesh, int n1, int n2, int *e1, int *e2){
  const char *caller="FEM_Mesh_get2ElementsOnEdge"; FEMAPI(caller);
  FEM_Mesh *m=FEM_Mesh_lookup(fem_mesh,caller);
  m->get2ElementsOnEdgeSorted(n1, n2, e1, e2);
}
FORTRAN_AS_C(FEM_MESH_GET2ELEMENTSONEDGESORTED,
             FEM_Mesh_get2ElementsOnEdgeSorted,
             fem_mesh_get2elementsonedgesorted,
             (int *fem_mesh, int *n1, int *n2, int *e1, int *e2),
             (*fem_mesh,*n1,*n2,e1,e2) )


void FEM_Node::allocateElemAdjacency(){
	if(elemAdjacency){
		delete elemAdjacency;
	}
	elemAdjacency = new FEM_VarIndexAttribute(this,FEM_NODE_ELEM_ADJACENCY);
	add(elemAdjacency);

}


void FEM_Node::allocateNodeAdjacency(){
	if(nodeAdjacency){
		delete nodeAdjacency;
	}
	nodeAdjacency = new FEM_VarIndexAttribute(this,FEM_NODE_NODE_ADJACENCY);
	add(nodeAdjacency);
}


/** Fill the node to element adjacency table for both 
    this element and its corresponding ghosts
 */
void FEM_Node::setElemAdjacency(int type, const FEM_Elem &elem){
	int nodesPerElem = elem.getNodesPer();
	FEM_VarIndexAttribute *adjacencyAttr = elemAdjacency;
	CkVec<CkVec<ElemID> > &adjacencyTable = elemAdjacency->get();
	FEM_VarIndexAttribute *ghostAdjacencyAttr = ((FEM_Node *)getGhost())->elemAdjacency;
	CkVec<CkVec<ElemID> > &ghostAdjacencyTable = ghostAdjacencyAttr->get();

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


/**  Populate the entire node to node adjacency table
     Two nodes are considered adjacent if they both are 
     in the connectivity table for a common element.
     This choice for definition of adjacent nodes 
     does not take into account what are edges 
     of the element, but it does simplify the computation.
     It will work fine for triangles and tetrahedra, but 
     may not make as much sense for more complicated
     element types where all nodes are not directly connected by edges.
*/
void FEM_Node::setNodeAdjacency(const FEM_Elem &elem){
  //CkPrintf("In FEM_Node::setNodeAdjacency()\n");
  int nodesPerElem = elem.getNodesPer();
  CkVec<CkVec<var_id> > &adjacencyTable = nodeAdjacency->get();
  FEM_VarIndexAttribute *ghostAdjacencyAttr = ((FEM_Node *)getGhost())->nodeAdjacency;
  CkVec<CkVec<var_id> > &ghostAdjacencyTable = ghostAdjacencyAttr->get();
  
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

}



/** Allocate both the FEM_ELEM_ELEM_ADJACENCY and FEM_ELEM_ELEM_ADJ_TYPES attributes
    The values in these two attributes will be generated by 
*/
void FEM_Elem::allocateElemAdjacency(){
	
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

}


void FEM_Mesh::createNodeElemAdj(){
	node.lookup(FEM_NODE_ELEM_ADJACENCY,"FEM_Mesh::createNodeElemAdj");
	for(int i=0;i<elem.size();i++){
		node.setElemAdjacency(i,elem[i]);
	}
}

void FEM_Mesh::createNodeNodeAdj(){
	node.lookup(FEM_NODE_NODE_ADJACENCY,"FEM_Mesh::createNodeNodeAdj");
	for(int i=0;i<elem.size();i++){
		node.setNodeAdjacency(elem[i]);
	}
}


/** A created on demand data structure that stores
 * the number of tuples(or faces) for each type of element
 * along with the number of nodes in each tuple.
 * The number of nodes in each tuple depends upon how many
 * nodes are in a shared face(an edge will have 2, a triangle 3)
 */
FEM_ElemAdj_Layer* FEM_Mesh::getElemAdjLayer(void) {
  if (! lastElemAdjLayer) { 
	lastElemAdjLayer=new FEM_ElemAdj_Layer();
	lastLayerSet = true;
  }
  return lastElemAdjLayer;
}


/** createElemElemAdj() is similar to splitter::addTuple()
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
    //CkPrintf("createElemElemAdj()\n");
	
  FEM_ElemAdj_Layer *g = getElemAdjLayer();
  if(! g->initialized)
	CkAbort("FEM> Cannot call FEM_Mesh_create_elem_elem_adjacency() before registering tuples with FEM_Add_elem2face_tuples()\n");

  
  int nodesPerTuple = g->nodesPerTuple;
  tupleTable table(nodesPerTuple);

  // Put tuples into table
  for (int t=0;t<elem.size();t++){ // for each element type
	  if(elem.hasNonEmpty(t)) {
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

  /* Extract adjacencies from table and store into 
   * FEM_ELEM_ELEM_ADJACENCY and FEM_ELEM_ELEM_ADJ_TYPES 
   * attribute fields
   */
  for (int t=0;t<elem.size();t++) { // for each element type t
	  if (elem.hasNonEmpty(t)) {
		  elemList *l;
		  const int tuplesPerElem = g->elem[t].tuplesPerElem;
		  const int numElements = elem[t].size();
		  const int numGhostElements = ((FEM_Elem*)(elem[t].getGhost()))->size();

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
			  adjTypes[i]=-1;
		  }
		  for(int i=0;i<numGhostElements*tuplesPerElem;i++){
			  adjsGhost[i]=-1;
			  adjTypesGhost[i]=-1;
		  }

		  // look through each elemList that is returned by the tuple table
		  table.beginLookup();
		  while (NULL!=(l=table.lookupNext())) {
			  if (l->next==NULL) { 
				  // One-entry list: must be a symmetry
				  // UNHANDLED CASE: not sure exactly what this means
			  }
			  else { /* Several elements in list: normal case */
				  //CkPrintf("Found in table list: ");
				  //for(const elemList *c=l;c!=NULL;c=c->next){
                                  //	  CkPrintf("     %d,%d", c->type, c->localNo);
				  //}
				  //CkPrintf("\n");
				  
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
									  //CkPrintf("Recording that %d,%d at position %d has neighbor %d,%d \n", a->type,a->localNo, a->tupleNo, b->type, b->localNo);
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
  }
  delete g;
}



/**  ------- Element-to-element: preserve initial ordering relative to nodes
     Place all of element e's adjacent elements in neighbors; assumes
     neighbors allocated to correct size
*/
void FEM_Mesh::e2e_getAll(int e, int *neighbors, int etype) 
{
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
}

/** Given id of element e of type etype, return the id of the idx-th adjacent element
 */
int FEM_Mesh::e2e_getNbr(int e, short idx, int etype) 
{     
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
}


ElemID FEM_Mesh::e2e_getElem(ElemID elem, int idx){
	return e2e_getElem(elem.id, idx, elem.type);
}

ElemID FEM_Mesh::e2e_getElem(ElemID *elem, int idx){
	return e2e_getElem(elem->id, idx, elem->type);
}



/** Get the idx'th element adjacent to element (e,etype) */
ElemID FEM_Mesh::e2e_getElem(int e, int idx, int etype){
  if (e == -1){
    ElemID ele(-1,-1);    
    return ele;
  }
  
  if(FEM_Is_ghost_index(e)){

	  FEM_IndexAttribute *eAdj;
	  FEM_IndexAttribute *eAdjType;

	  eAdj = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getElem");
	  AllocTable2d<int> &eAdjs = eAdj->get();
	  eAdjType = (FEM_IndexAttribute *)elem[etype].getGhost()->lookup(FEM_ELEM_ELEM_ADJ_TYPES,"e2e_getElem");
	  AllocTable2d<int> &eAdjTypes = eAdjType->get();

	  int t =  eAdjTypes[FEM_To_ghost_index(e)][idx];
	  int id =  eAdjs[FEM_To_ghost_index(e)][idx];
	  ElemID ele(t,id);

	  return ele;
  }
  else {

	  CkAssert(elem.has(etype) && e < elem[etype].size());

	  FEM_IndexAttribute *eAdj;
	  FEM_IndexAttribute *eAdjType;

	  eAdj = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_getElem");
	  AllocTable2d<int> &eAdjs = eAdj->get();
	  eAdjType = (FEM_IndexAttribute *)elem[etype].lookup(FEM_ELEM_ELEM_ADJ_TYPES,"e2e_getElem");
	  AllocTable2d<int> &eAdjTypes = eAdjType->get();

	  int t =  eAdjTypes[e][idx];
	  int id =  eAdjs[e][idx];
	  ElemID ele(t,id);

	  return ele;
  }
  
}


/** Given id of element e and id of another element nbr, return i such that
    nbr is the i-th element adjacent to e
*/
int FEM_Mesh::e2e_getIndex(int e, int nbr, int etype) 
{ 
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
  return -1;
}

/** Set the element adjacencies of element e to neighbors; assumes neighbors 
    has the correct size
*/
void FEM_Mesh::e2e_setAll(int e, int *neighbors, int etype) 
{
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
}

/** Set the idx-th element adjacent to e to be newElem
 */
void FEM_Mesh::e2e_setIndex(int e, short idx, int newElem, int etype) 
{
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
}

/** Find element oldNbr in e's adjacent elements and replace with newNbr
 */
void FEM_Mesh::e2e_replace(int e, int oldNbr, int newNbr, int etype) 
{
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
}



/** Find element oldNbr in e's adjacent elements and replace with newNbr
 */
void FEM_Mesh::e2e_replace(ElemID e, ElemID oldNbr, ElemID newNbr) 
{
  if (e.id == -1) return;

//	CkPrintf("replacing %d,%d with %d,%d\n", oldNbr.type, oldNbr.id, newNbr.type, newNbr.id);

  
  FEM_IndexAttribute *eAdj;
  if(FEM_Is_ghost_index(e.id)){
	  FEM_IndexAttribute *elemAdjTypesAttrGhost = (FEM_IndexAttribute *)elem[e.type].getGhost()->lookup(FEM_ELEM_ELEM_ADJ_TYPES,"e2e_replace");
	  FEM_IndexAttribute *elemAdjAttrGhost = (FEM_IndexAttribute *)elem[e.type].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_replace");
   
	  AllocTable2d<int> &eAdjs = elemAdjAttrGhost->get();
	  AllocTable2d<int> &eAdjTypes = elemAdjTypesAttrGhost->get();
	  
	  for (int i=0; i<eAdjs.width(); i++) {
		  if (eAdjs[FEM_To_ghost_index(e.id)][i] == oldNbr.id && eAdjTypes[FEM_To_ghost_index(e.id)][i] == oldNbr.type ) {
			  eAdjs[FEM_To_ghost_index(e.id)][i] = newNbr.id;
			  eAdjTypes[FEM_To_ghost_index(e.id)][i] = newNbr.type;
			  break;
		  }
	  }
  }
  else{
    eAdj = (FEM_IndexAttribute *)elem[e.type].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_replace");
  
    FEM_IndexAttribute *elemAdjTypesAttr = (FEM_IndexAttribute *)elem[e.type].lookup(FEM_ELEM_ELEM_ADJ_TYPES,"e2e_replace");
    FEM_IndexAttribute *elemAdjAttr = (FEM_IndexAttribute *)elem[e.type].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_replace");
    
    AllocTable2d<int> &eAdjs = elemAdjAttr->get();
    AllocTable2d<int> &eAdjTypes = elemAdjTypesAttr->get();

    for (int i=0; i<eAdjs.width(); i++) {
    	if (eAdjs[e.id][i] == oldNbr.id && eAdjTypes[e.id][i] == oldNbr.type ) {
//    		CkPrintf("eAdjTypes[e.id][i] was %d eAdjs[e.id][i] was %d\n", eAdjTypes[e.id][i], eAdjs[e.id][i]);
    		eAdjs[e.id][i] = newNbr.id;
    		eAdjTypes[e.id][i] = newNbr.type;
//    		CkPrintf("newNbr.type = %d\n", newNbr.type);
//    		CkPrintf("eAdjTypes[e.id][i] is %d eAdjs[e.id][i] is %d\n", eAdjTypes[e.id][i], eAdjs[e.id][i]);
    		break;
    	}
    }
    
  }
}


/** Remove all neighboring elements in adjacency
 */
void FEM_Mesh::e2e_removeAll(int e, int etype)
{
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
}


/**  Print all the elements adjacent to the given element.
*/
void FEM_Mesh::e2e_printAll(ElemID e) 
{
	if (e.id == -1) return; // non existent element
	
	FEM_IndexAttribute *eAdj;
	FEM_IndexAttribute *eAdjType;
	
	if(FEM_Is_ghost_index(e.id)){
		eAdj = (FEM_IndexAttribute *)elem[e.type].getGhost()->lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_printAll");
		eAdjType = (FEM_IndexAttribute *)elem[e.type].getGhost()->lookup(FEM_ELEM_ELEM_ADJ_TYPES,"e2e_printAll");
	}
	else {
		eAdj = (FEM_IndexAttribute *)elem[e.type].lookup(FEM_ELEM_ELEM_ADJACENCY,"e2e_printAll");
		eAdjType = (FEM_IndexAttribute *)elem[e.type].lookup(FEM_ELEM_ELEM_ADJ_TYPES,"e2e_printAll");
	}
	
	
	AllocTable2d<int> &eAdjs = eAdj->get();
	AllocTable2d<int> &eAdjTypes = eAdjType->get();
	CkAssert(eAdjs.width() == eAdjTypes.width());

	CkAssert(e.getSignedId()>=0);

	for (int i=0; i<eAdjs.width(); i++) {
		CkPrintf("Element %d,%d is adjacent to %d,%d\n", e.type, e.id, eAdjTypes[e.getSignedId()][i], eAdjs[e.getSignedId()][i]);

	}

}



/**  ------- Element-to-node: preserve initial ordering
     Place all of element e's adjacent nodes in adjnodes; assumes
     adjnodes allocated to correct size
*/
void FEM_Mesh::e2n_getAll(int e, int *adjnodes, int etype) 
{
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
}

/** Given id of element e, return the id of the idx-th adjacent node
 */
int FEM_Mesh::e2n_getNode(int e, short idx, int etype) 
{ 
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
}

/** Given id of element e and id of a node n, return i such that
    n is the i-th node adjacent to e
*/
short FEM_Mesh::e2n_getIndex(int e, int n, int etype) 
{
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
  return -1;
}

/** Set the node adjacencies of element e to adjnodes; assumes adjnodes 
    has the correct size
*/
void FEM_Mesh::e2n_setAll(int e, int *adjnodes, int etype) 
{
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
}

/** Set the idx-th node adjacent to e to be newNode
 */
void FEM_Mesh::e2n_setIndex(int e, short idx, int newNode, int etype) 
{
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
}

/** Find node oldNode in e's adjacent ndoes and replace with newNode
 */
void FEM_Mesh::e2n_replace(int e, int oldNode, int newNode, int etype) 
{
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
}

void FEM_Mesh::e2n_removeAll(int e, int etype)
{
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
}




/**  ------- Node-to-node */
int FEM_Mesh::n2n_getLength(int n) {
  if (n == -1) {
    return 0;
  }
  FEM_VarIndexAttribute *eAdj;
  if(FEM_Is_ghost_index(n)){
    eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_getLength");
    n = FEM_To_ghost_index(n);
  }
  else {
    eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_getLength");
  }
  CkVec<CkVec<ElemID> > &eVec = eAdj->get();
  CkVec<ElemID> &nsVec = eVec[n];
  return nsVec.length();
}

/** Place all of node n's adjacent nodes in adjnodes and the resulting 
    length of adjnodes in sz; assumes adjnodes is not allocated, but sz is
*/
void FEM_Mesh::n2n_getAll(int n, int *&adjnodes, int &sz) 
{
  if (n == -1) {
    sz = 0;
    return;
  }
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_getAll");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[FEM_To_ghost_index(n)];
	sz = nsVec.length();
	if(sz > 0) adjnodes = new int[sz];
	for (int i=0; i<sz; i++) {
	  adjnodes[i] = nsVec[i].getSignedId();
	}
  }
  else{
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_getAll");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[n];
	sz = nsVec.length();
	if(sz > 0) adjnodes = new int[sz];
	for (int i=0; i<sz; i++) {
	  adjnodes[i] = nsVec[i].getSignedId();
	}
  }
}
 
/** Adds newNode to node n's node adjacency list
 */
void FEM_Mesh::n2n_add(int n, int newNode) 
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_add");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	ElemID nn(0, newNode);
	CkVec<ElemID> &nsVec = nVec[FEM_To_ghost_index(n)];
	nsVec.push_back(nn);
  }
  else{
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_add");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	ElemID nn(0, newNode);
	CkVec<ElemID> &nsVec = nVec[n];
	nsVec.push_back(nn);
  }
}




/** Removes oldNode from n's node adjacency list
 */
void FEM_Mesh::n2n_remove(int n, int oldNode) 
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_remove");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec.remove(i);
		break;
	  }
	}
  }
  else {
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_remove");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec.remove(i);
		break;
	  }
	}
  }
}

/** Is queryNode in node n's adjacency vector?
 */
int FEM_Mesh::n2n_exists(int n, int queryNode) 
{
  if (n == -1) return 0;
  if(FEM_Is_ghost_index(n)){
	CkAssert(node.getGhost());
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_exists");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++)
	  if (nsVec[i].getSignedId() == queryNode) 
		return 1;
  }
  else {
    FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_exists");
    CkVec<CkVec<ElemID> > &nVec = nAdj->get();
    CkVec<ElemID> &nsVec = nVec[n];
    for (int i=0; i<nsVec.length(); i++)
      if (nsVec[i].getSignedId() == queryNode) 
	return 1;
  }
  return 0;
}

/** Finds oldNode in n's node adjacency list, and replaces it with newNode
 */
void FEM_Mesh::n2n_replace(int n, int oldNode, int newNode) 
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_replace");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec[i] = ElemID(0,newNode);
		break;
	  }
	}
  }
  else {
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_replace");
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldNode) {
		nsVec[i] = ElemID(0,newNode);
		break;
	  }
	}

  }
}

/** Remove all nodes from n's node adjacency list
 */
void FEM_Mesh::n2n_removeAll(int n)
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_NODE_ADJACENCY,"n2n_removeAll");  
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[FEM_To_ghost_index(n)];
	nsVec.free();
  }
  else{
	FEM_VarIndexAttribute *nAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_NODE_ADJACENCY,"n2n_removeAll");  
	CkVec<CkVec<ElemID> > &nVec = nAdj->get();
	CkVec<ElemID> &nsVec = nVec[n];
	nsVec.free();
  }
}

/**  ------- Node-to-element */
int FEM_Mesh::n2e_getLength(int n) {
  if (n == -1) {
    return 0;
  }
  FEM_VarIndexAttribute *eAdj;
  if(FEM_Is_ghost_index(n)){
    eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getLength");
    n = FEM_To_ghost_index(n);
  }
  else {
    eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getLength");
  }
  CkVec<CkVec<ElemID> > &eVec = eAdj->get();
  CkVec<ElemID> &nsVec = eVec[n];
  return nsVec.length();
}

/** Return one of node n's adjacent elements 
*/

ElemID FEM_Mesh::n2e_getElem(int n, int whichIdx){

  if(FEM_Is_ghost_index(n)){
    FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getElem");
    CkVec<CkVec<ElemID> > &eVec = eAdj->get();
    CkVec<ElemID> &nsVec = eVec[FEM_To_ghost_index(n)];
    assert(whichIdx < nsVec.length());
    return  nsVec[whichIdx];
    
  }
  else {
    FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getAll");
    CkVec<CkVec<ElemID> > &eVec = eAdj->get();
    CkVec<ElemID> &nsVec = eVec[n];
    assert(whichIdx < nsVec.length());
    return  nsVec[whichIdx];
  }
  
}



/** Place all of node n's adjacent elements in adjelements and the resulting 
    length of adjelements in sz; assumes adjelements is not allocated, 
    but sz is
*/
void FEM_Mesh::n2e_getAll(int n, int *&adjelements, int &sz) 
{
  if (n == -1) {
    sz = 0;
    return;
  }
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getAll");  
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[FEM_To_ghost_index(n)];
	sz = nsVec.length();
	if(sz > 0) adjelements = new int[sz];
	for (int i=0; i<sz; i++) {
	  adjelements[i] = nsVec[i].getSignedId();
	}
  }
  else {
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getAll");  
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[n];
	int len = nsVec.length();
	sz = len;
	if(sz > 0) adjelements = new int[sz];
	for (int i=0; i<sz; i++) {
	  adjelements[i] = nsVec[i].getSignedId();
	}
  }
}
 

/** Place all of node n's adjacent elements in adjelements and the resulting 
    length of adjelements in sz; assumes adjelements is not allocated, 
    but sz is.
    
    This function returns elements and their associated types.
*/
const CkVec<ElemID> &  FEM_Mesh::n2e_getAll(int n) 
{
	assert(n!=-1); 

	if(FEM_Is_ghost_index(n)){
		FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getAll");  
		CkVec<CkVec<ElemID> > &eVec = eAdj->get();
		return eVec[FEM_To_ghost_index(n)];
	}
	else {
		FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_getAll");  
		CkVec<CkVec<ElemID> > &eVec = eAdj->get();
		return eVec[n];
	}
}
 


/** Adds newElem to node n's element adjacency list
 */
void FEM_Mesh::n2e_add(int n, int newElem) 
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_add");     
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[FEM_To_ghost_index(n)];
	ElemID ne(0, newElem);
	nsVec.push_back(ne);
	int *testn2e, testn2ec;
	n2e_getAll(n,testn2e,testn2ec);
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
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[n];
	ElemID ne(0, newElem);
	nsVec.push_back(ne);
  }
}

/** Removes oldElem from n's element adjacency list
 */
void FEM_Mesh::n2e_remove(int n, int oldElem) 
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_remove");
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec.remove(i);
		break;
	  }
	}
  }
  else {
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_remove");
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec.remove(i);
		break;
	  }
	}
  }
}
 
/** Finds oldElem in n's element adjacency list, and replaces it with newElem
 */
void FEM_Mesh::n2e_replace(int n, int oldElem, int newElem) 
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_replace");
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[FEM_To_ghost_index(n)];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec[i] = ElemID(0,newElem);
		break;
	  }
	}
	int *testn2e, testn2ec;
	n2e_getAll(n,testn2e,testn2ec);
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
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[n];
	for (int i=0; i<nsVec.length(); i++) {
	  if (nsVec[i].getSignedId() == oldElem) {
		nsVec[i] = ElemID(0,newElem);
		break;
	  }
	}
  }
}

/** Remove all elements from n's element adjacency list
 */
void FEM_Mesh::n2e_removeAll(int n)
{
  if (n == -1) return;
  if(FEM_Is_ghost_index(n)){
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.getGhost()->lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_removeAll");
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[FEM_To_ghost_index(n)];
	nsVec.free();
  }
  else{
	FEM_VarIndexAttribute *eAdj = (FEM_VarIndexAttribute *)node.lookup(FEM_NODE_ELEM_ADJACENCY,"n2e_removeAll");
	CkVec<CkVec<ElemID> > &eVec = eAdj->get();
	CkVec<ElemID> &nsVec = eVec[n];
	nsVec.free();
  }
}

/** Get an element on edge (n1, n2) where n1, n2 are chunk-local
    node numberings; return -1 in case of failure
    I think the break statement may cause some unexpected behaviour here
    The break will break out of the inner for loop back to the outer for loop
    I am not sure exactly what the symantics of this function should be
    -- Isaac
*/

int FEM_Mesh::getElementOnEdge(int n1, int n2) 
{
  int *n1AdjElems, *n2AdjElems;
  int n1NumElems, n2NumElems;
  n2e_getAll(n1, n1AdjElems, n1NumElems);
  n2e_getAll(n2, n2AdjElems, n2NumElems);
  int ret = -1;
  //CkPrintf("%d has %d neighboring elements, %d has %d\n", n1, n1NumElems, n2, n2NumElems);

  bool flag = false;
  for (int i=0; i<n1NumElems; i++) {
    for (int j=0; j<n2NumElems; j++) {
      if (n1AdjElems[i] == n2AdjElems[j]) {
        if(n1AdjElems[i] >= 0) {
          ret = n1AdjElems[i];
	  flag = true;
          break;
        }
        else {
          ret = n1AdjElems[i];
        }
      }
    }
    if(flag) break;
  }
  delete[] n1AdjElems;
  delete[] n2AdjElems;
  return ret; //preferably return a local element, otherwise return a ghost 
}


/** Get 2 elements on edge (n1, n2) where n1, n2 are chunk-local
    node numberings; return the edges in result_e1 and result_e2
    No preference is given to ghosts over local elements
*/
void FEM_Mesh::get2ElementsOnEdge(int n1, int n2, int *result_e1, int *result_e2) 
{
  int *n1AdjElems=0, *n2AdjElems=0;
  int n1NumElems, n2NumElems;

  if(n1==n2){
    CkPrintf("ERROR: You called get2ElementsOnEdge() with two identical nodes %d, and %d \n", n1, n2);
    CkExit();
  }

  n2e_getAll(n1, n1AdjElems, n1NumElems);
  n2e_getAll(n2, n2AdjElems, n2NumElems);
  CkAssert(n1AdjElems!=0);
  CkAssert(n2AdjElems!=0);
  int found=0;

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
          CkPrintf("ERROR: Found a third element(%d) on edge %d,%d \n", n1AdjElems[i], n1, n2);
           CkExit();
        }
      }
    }
  }
  delete[] n1AdjElems;
  delete[] n2AdjElems;
}

/** 
    Get an id out of FEM_DATA+3 for an element of type 0.
 */
int getID(FEM_Mesh *mesh, int elem){
    int id = 0;

    if(elem<0){
      AllocTable2d<int> &idTable = ((FEM_DataAttribute *)mesh->elem[0].getGhost()->lookup(FEM_DATA+3,"getID"))->getInt();
      id = idTable(0,FEM_To_ghost_index(elem));
    } else {
      AllocTable2d<int> &idTable = ((FEM_DataAttribute *)mesh->elem[0].lookup(FEM_DATA+3,"getID"))->getInt();
      id = idTable(0,elem);
    }
    
  return id;
}


/** Get 2 elements on edge (n1, n2) where n1, n2 are chunk-local
    node numberings; return the edges in result_e1 and result_e2

    Elements will be sorted by their integer stored in FEM_DATA+3.
    Only works for element type 0

*/
void FEM_Mesh::get2ElementsOnEdgeSorted(int n1, int n2, int *result_e1, int *result_e2)
{
  int *n1AdjElems=0, *n2AdjElems=0;
  int n1NumElems, n2NumElems;
  
  get2ElementsOnEdge(n1, n2, result_e1, result_e2);

  //   Swap the two elements if the first has greater id than the second 
    if(getID(this,*result_e1) > getID(this,*result_e2)){
    int temp = *result_e1;
    *result_e1 = *result_e2;
    *result_e2 = temp;
  }
}




/** Count the number of elements on edge (n1, n2) */
int FEM_Mesh::countElementsOnEdge(int n1, int n2) {
  if (n1==n2) {
    CkPrintf("ERROR: You called countElementsOnEdge() with two identical nodes %d, and %d \n", n1, n2);
    CkExit();
  }
  
  int *n1AdjElems=0, *n2AdjElems=0;
  int n1NumElems, n2NumElems;

  CkAssert(node.is_valid_any_idx(n1));
  CkAssert(node.is_valid_any_idx(n2));

  n2e_getAll(n1, n1AdjElems, n1NumElems);
  n2e_getAll(n2, n2AdjElems, n2NumElems);
  CkAssert(n1AdjElems!=0);
  CkAssert(n2AdjElems!=0);
  int count=0;


  for (int i=0; i<n1NumElems; i++) {
    for (int j=0; j<n2NumElems; j++) {
      if (n1AdjElems[i] == n2AdjElems[j]) {
        count++;
      }
    }
  }
  delete[] n1AdjElems;
  delete[] n2AdjElems;

  return count;
}


