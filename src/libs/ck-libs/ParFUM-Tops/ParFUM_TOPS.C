/**

   @file
   @brief Implementation of ParFUM-TOPS layer, except for Iterators
   
   @author Isaac Dooley

   @todo add code to generate ghost layers
   @todo Support multiple models

   @note FEM_DATA+0 holds the elemAttr or nodeAtt data
   @note FEM_DATA+1 holds the id 
   @note FEM_CONN holds the element connectivity
   @note FEM_COORD holds the nodal coordinates

*/

#include "ParFUM_TOPS.h"
#include "ParFUM.decl.h"
#include "ParFUM_internals.h"

int elem_attr_size, node_attr_size;

TopModel* topModel_Create_Init(int elem_attr_sz, int node_attr_sz){
  CkAssert(elem_attr_sz > 0);
  CkAssert(node_attr_sz > 0);
  elem_attr_size = elem_attr_sz;
  node_attr_size = node_attr_sz;

  // This only uses a single mesh, so better not create multiple ones of these
  int which_mesh=FEM_Mesh_default_write();
  FEM_Mesh *mesh = FEM_Mesh_lookup(which_mesh,"TopModel::TopModel()");

  char* temp_array = new char[16]; // just some junk array to use below
  
  // Allocate element connectivity
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_CONN,temp_array, 0, 0, FEM_INDEX_0, 4);
  // Allocate element attributes
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+0,temp_array, 0, 0, FEM_BYTE, elem_attr_size);
  // Allocate element Id array
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+1,temp_array, 0, 0, FEM_INT, 1);

  // Allocate node coords
  FEM_Mesh_data(which_mesh,FEM_NODE,FEM_COORD,temp_array, 0, 0, FEM_DOUBLE, 3);
  // Allocate node attributes
  FEM_Mesh_data(which_mesh,FEM_NODE+0,FEM_DATA+0,temp_array, 0, 0, FEM_BYTE, node_attr_size);
  // Allocate node Id array
  FEM_Mesh_data(which_mesh,FEM_NODE+0,FEM_DATA+1,temp_array, 0, 0, FEM_INT, 1);

  delete[] temp_array;

  // Don't Allocate the Global Number attribute for the elements and nodes  
  // It will be automatically created upon calls to void FEM_Entity::setGlobalno(int r,int g) {

  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_NODE);
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_ELEM+0);
  

  // Setup ghost layers, assume Tet4
  // const int tet4vertices[4] = {0,1,2,3};
  //  FEM_Add_ghost_layer(1,1);
  // FEM_Add_ghost_elem(0,4,tet4vertices);

  return mesh;
}

TopModel* topModel_Create_Driver(int elem_attr_sz, int node_attr_sz){
  // This only uses a single mesh, so don't create multiple TopModels of these
  CkAssert(elem_attr_sz > 0);
  CkAssert(node_attr_sz > 0);
  elem_attr_size = elem_attr_sz;
  node_attr_size = node_attr_sz;
  int which_mesh=FEM_Mesh_default_read();
  FEM_Mesh *mesh = FEM_Mesh_lookup(which_mesh,"TopModel::TopModel()");
  return mesh;
}

void topModel_Destroy(TopModel* m){
}


TopNode topModel_InsertNode(TopModel* m, double x, double y, double z){
  int newNode = FEM_add_node_local(m,false,false,false);
  m->node.set_coord(newNode,x,y,z);
  return newNode;
}


/** Set id of a node 
@todo Make this work with ghosts
*/
void topNode_SetId(TopModel* m, TopNode n, TopID id){
  CkAssert(n>=0);
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+1,"topNode_SetId");
  at->getInt()(n,0)=id;
}

/** Insert an element */
TopElement topModel_InsertElem(TopModel*m, TopElemType type, TopNode* nodes){
  CkAssert(type ==  TOP_ELEMENT_TET4);
  int conn[4];
  conn[0] = nodes[0];
  conn[1] = nodes[1];
  conn[2] = nodes[2];
  conn[3] = nodes[3];
  int newEl = FEM_add_element_local(m, conn, 4, 0, 0, 0);
  return newEl;
}

/** Set id of an element 
@todo Make this work with ghosts
*/
void topElement_SetId(TopModel* m, TopElement e, TopID id){
  CkAssert(e>=0);
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+1,"topElement_SetId");
  at->getInt()(e,0)=id;
}



/** 
	@brief Set attribute of a node 
	
	The attribute passed in must be a contiguous data structure with size equal to the value node_attr_sz passed into topModel_Create_Driver() and topModel_Create_Init() 

	The supplied attribute will be copied into the ParFUM attribute array "FEM_DATA+0". Then ParFUM will own this data. The function topNode_GetAttrib() will return a pointer to the copy owned by ParFUM. If a single material parameter attribute is used for multiple nodes, each node will get a separate copy of the array. Any subsequent modifications to the data will only be reflected at a single node. 

	The user is responsible for deallocating parameter d passed into this function.

*/
void topNode_SetAttrib(TopModel* m, TopNode n, void* d){
  CkAssert(n>=0);
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+0,"topNode_SetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  memcpy(data + n*node_attr_size, d, node_attr_size);
}

/** @brief Set attribute of an element 
See topNode_SetAttrib() for description
*/
void topElement_SetAttrib(TopModel* m, TopElement e, void* d){
  CkAssert(e>=0);
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+0,"topElem_SetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  memcpy(data + e*elem_attr_size, d, elem_attr_size);
}


/** @brief Get elem attribute 
See topNode_SetAttrib() for description
*/
void* topElement_GetAttrib(TopModel* m, TopElement e){
  if(! m->elem[0].is_valid_any_idx(e))
	return NULL;

  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+0,"topElem_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (data + e*elem_attr_size);
}

/** @brief Get nodal attribute 
See topNode_SetAttrib() for description
*/
void* topNode_GetAttrib(TopModel* m, TopNode n){
  if(! m->node.is_valid_any_idx(n))
	return NULL;

  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+0,"topNode_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (data + n*node_attr_size);
}



/** 
	Get node via id 
	@todo Re-implement this function with some hash to make it fast. 
	@note In the explicit FEA example, this is just used during initialization, so speed is not too important.
	@todo Does not work with ghosts yet.
*/
TopNode topModel_GetNodeAtId(TopModel* m, TopID id){
  // lookup node via global ID
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+1,"topModel_GetNodeAtId");
  for(int i=0;i<at->getInt().size();++i){
	if(at->getInt()(i,0)==id){
	  return i;
	}
  }
  return -1;
}

/** 
	Get elem via id
	@todo Implement this function
	@note Is this function even supposed to exist?
 */
TopElement topModel_GetElemAtId(TopModel*m,TopID id){
  CkPrintf("Error: topModel_GetElemAtId() not yet implemented");
  CkExit();
}



TopNode topElement_GetNode(TopModel* m,TopElement e,int idx){
  CkAssert(e>=0);
  const AllocTable2d<int> &conn = m->elem[0].getConn();
  CkAssert(idx>=0 && idx<conn.width());
  CkAssert(idx<conn.size());

  int node = conn(e,idx);

  return conn(e,idx);
}

int topNode_GetId(TopModel* m, TopNode n){
  CkAssert(n>=0);
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+1,"topNode_SetId");
  return at->getInt()(n,0);
}


/** @todo handle ghost nodes as appropriate */
int topModel_GetNNodes(TopModel *model){
  return model->node.count_valid();
}

/** @todo Fix to return the width of the conn array */
int topElement_GetNNodes(TopModel* model, TopElement elem){
  return 4;
}

/** @todo make sure we are in a getting mesh */
void topNode_GetPosition(TopModel*model, TopNode node,double*x,double*y,double*z){
  CkAssert(node>=0);
  double coord[3]={7.01,7.01,7.01};

  // lookup node via global ID
  FEM_DataAttribute * at = (FEM_DataAttribute*) model->node.lookup(FEM_COORD,"topModel_GetNodeAtId");
  CkAssert(at->getDouble().width()==3);
  //  CkPrintf("node=%d, size=%d\n",node,at->getDouble().size() );
  CkAssert(node<at->getDouble().size());
  *x = at->getDouble()(node,0);
  *y = at->getDouble()(node,1);
  *z = at->getDouble()(node,2);
}

void topModel_Sync(TopModel*m){
  MPI_Barrier(MPI_COMM_WORLD);

 
  //  CkPrintf("%d: %d local, %d ghost elements\n", FEM_My_partition(), m->elem[0].size(),m->elem[0].ghost->size() );
  //  CkPrintf("%d: %d local, %d ghost valid elements\n", FEM_My_partition(), m->elem[0].count_valid(),m->elem[0].ghost->count_valid() );

}

void topModel_TestIterators(TopModel*m){
  CkAssert(m->elem[0].ghost!=NULL);
  CkAssert(m->node.ghost!=NULL);

  int expected_elem_count = m->elem[0].count_valid() + m->elem[0].ghost->count_valid(); 
  int iterated_elem_count = 0;

  int expected_node_count = m->node.count_valid() + m->node.ghost->count_valid(); 
  int iterated_node_count = 0;

  int myId = FEM_My_partition();


  TopNodeItr* itr = topModel_CreateNodeItr(m);
  for(topNodeItr_Begin(itr);topNodeItr_IsValid(itr);topNodeItr_Next(itr)){
	iterated_node_count++;
	TopNode node = topNodeItr_GetCurr(itr);
	void* na = topNode_GetAttrib(m,node);
	CkAssert(na != NULL);
  }
    
  TopElemItr* e_itr = topModel_CreateElemItr(m);
  for(topElemItr_Begin(e_itr);topElemItr_IsValid(e_itr);topElemItr_Next(e_itr)){
	iterated_elem_count++;
	TopElement elem = topElemItr_GetCurr(e_itr);
	void* ea = topElement_GetAttrib(m,elem);
	CkAssert(ea != NULL);
  }
  
  CkAssert(iterated_node_count == expected_node_count);
  CkAssert(iterated_elem_count==expected_elem_count);
  
}


#include "ParFUM_TOPS.def.h"
