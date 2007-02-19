/**
*	A ParFUM TOPS compatibility layer
*
*      Author: Isaac Dooley



Assumptions: 

TopNode is just the index into the nodes
TopElem is just the signed index into the elements negatives are ghosts


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
  // This only uses a single mesh, so better not create multiple ones of these
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

/** Set attribute of a node */
void topNode_SetAttrib(TopModel* m, TopNode n, NodeAtt* d){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+0,"topNode_SetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  memcpy(data + n*node_attr_size, d, node_attr_size);
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

/** Set attribute of an element */
void topElement_SetAttrib(TopModel* m, TopElement e, ElemAtt* d){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+0,"topElem_SetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  memcpy(data + e*elem_attr_size, d, elem_attr_size);
}


/** Get elem attribute */
ElemAtt* topElem_GetAttrib(TopModel* m, TopElement e){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+0,"topElem_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (ElemAtt*)(data + e*elem_attr_size);
}

// /** Get nodal attribute */
NodeAtt* topNode_GetAttrib(TopModel* m, TopNode n){
  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+0,"topNode_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (NodeAtt*)(data + n*node_attr_size);
}



/** Get node via id */
TopNode topModel_GetNodeAtId(TopModel*,TopID){
  // lookup node via global ID
  assert(0);
}

/** Get elem via id */
TopElement topModel_GetElemAtId(TopModel*,TopID){
  assert(0);
}





#include "ParFUM_TOPS.def.h"
