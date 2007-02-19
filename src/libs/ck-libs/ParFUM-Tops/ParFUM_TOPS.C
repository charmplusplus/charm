/**

   @file
   @brief Implementation of ParFUM-TOPS layer, except for Iterators
   
   @author Isaac Dooley

   @todo add code to generate ghost layers
   @todo Support multiple models

*/

#include "ParFUM_TOPS.h"
#include "ParFUM.decl.h"

int elem_attr_size, node_attr_size;

TopModel* topModel_Create_Init(int elem_attr_sz, int node_attr_sz){

  elem_attr_size = elem_attr_sz;
  node_attr_size = node_attr_sz;

  // This only uses a single mesh, so better not create multiple ones of these
  int which_mesh=FEM_Mesh_default_write();
  FEM_Mesh *mesh = FEM_Mesh_lookup(which_mesh,"TopModel::TopModel()");

  char* temp_array = new char[16]; // just some junk array to use below
  
  // Allocate element connectivity
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_CONN,temp_array, 0, 0, FEM_INDEX_0, 3);
  // Allocate node coords
  FEM_Mesh_data(which_mesh,FEM_NODE,FEM_COORD,temp_array, 0, 0, FEM_DOUBLE, 3);
  // Allocate element attributes
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+0,temp_array, 0, 0, FEM_BYTE, elem_attr_size);
  // Allocate node attributes
  FEM_Mesh_data(which_mesh,FEM_NODE+0,FEM_DATA+0,temp_array, 0, 0, FEM_BYTE, node_attr_size);

  delete[] temp_array;

  // Don't Allocate the Global Number attribute for the elements and nodes  
  // It will be automatically created upon calls to void FEM_Entity::setGlobalno(int r,int g) {
  
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_NODE);
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_ELEM+0);
  
  return mesh;
}

TopModel* topModel_Create_Driver(int elem_attr_sz, int node_attr_sz){
  // This only uses a single mesh, so don't create multiple TopModels of these
  elem_attr_size = elem_attr_sz;
  node_attr_size = node_attr_sz;
  int which_mesh=FEM_Mesh_default_read();
  FEM_Mesh *mesh = FEM_Mesh_lookup(which_mesh,"TopModel::TopModel()");
  return mesh;
}

TopNode topModel_InsertNode(TopModel* m, double x, double y, double z){
  int newNode = FEM_add_node_local(m,false,false,false);
  m->node.set_coord(newNode,x,y,z);
  return newNode;
}


/** Set id of a node */
void topNode_SetId(TopModel* m, TopNode n, TopID id){
	// just set the nodal id attribute to id
  m->node.setGlobalno(n,id);
}

 /** Insert an element */
TopElement topModel_InsertElem(TopModel*m, TopElemType type, TopNode* nodes){
  assert(type == FEM_TRIANGULAR);
  int conn[3];
  conn[0] = nodes[0];
  conn[1] = nodes[1];
  conn[2] = nodes[2];
  int newEl = FEM_add_element_local(m, conn, 3, 0, 0, 0);
  return newEl;
}

/** Set id of an element */
void topElement_SetId(TopModel* m, TopElement e, TopID id){
  m->elem[0].setGlobalno(e,id);
}



/** 
	@brief Set attribute of a node 
	
	The attribute passed in must be a contiguous data structure with size equal to the value node_attr_sz passed into topModel_Create_Driver() and topModel_Create_Init() 

	The supplied attribute will be copied into the ParFUM attribute array "FEM_DATA+0". Then ParFUM will own this data. The function topNode_GetAttrib() will return a pointer to the copy owned by ParFUM. If a single material parameter attribute is used for multiple nodes, each node will get a separate copy of the array. Any subsequent modifications to the data will only be reflected at a single node. 

	The user is responsible for deallocating parameter d passed into this function.

*/
void topNode_SetAttrib(TopModel* m, TopNode n, NodeAtt* d){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+0,"topNode_SetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  memcpy(data + n*node_attr_size, d, node_attr_size);
}

/** @brief Set attribute of an element 
See topNode_SetAttrib() for description
*/
void topElement_SetAttrib(TopModel* m, TopElement e, ElemAtt* d){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+0,"topElem_SetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  memcpy(data + e*elem_attr_size, d, elem_attr_size);
}


/** @brief Get elem attribute 
See topNode_SetAttrib() for description
*/
ElemAtt* topElem_GetAttrib(TopModel* m, TopElement e){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+0,"topElem_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (ElemAtt*)(data + e*elem_attr_size);
}

/** @brief Get nodal attribute 
See topNode_SetAttrib() for description
*/
NodeAtt* topNode_GetAttrib(TopModel* m, TopNode n){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+0,"topNode_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (NodeAtt*)(data + n*node_attr_size);
}



/** 
	Get node via id 
	@todo Implement this function
*/
TopNode topModel_GetNodeAtId(TopModel*,TopID){
  // lookup node via global ID
  assert(0);
}

/** 
	Get elem via id
	@todo Implement this function
 */
TopElement topModel_GetElemAtId(TopModel*,TopID){
  assert(0);
}





#include "ParFUM_TOPS.def.h"
