/**
*	A ParFUM TOPS compatibility layer
*
*      Author: Isaac Dooley



Assumptions: 

TopNode is just the index into the nodes
TopElem is just the signed index into the elements negatives are ghosts





Notes:


mesh_modify.C:  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_Print_Mesh_Summary");
mesh_modify.C:  return FEM_add_node(FEM_Mesh_lookup(mesh,"FEM_add_node"), adjacent_nodes, num_adjacent_nodes, chunks, numChunks, forceShared);
mesh_modify.C:  return FEM_remove_node(FEM_Mesh_lookup(mesh,"FEM_remove_node"), node);
mesh_modify.C:  return FEM_add_element(FEM_Mesh_lookup(mesh,"FEM_add_element"), conn, conn_size, elem_type, chunkNo);
mesh_modify.C:  return FEM_remove_element(FEM_Mesh_lookup(mesh,"FEM_remove_element"), element, elem_type, permanent);
mesh_modify.C:  return FEM_purge_element(FEM_Mesh_lookup(mesh,"FEM_remove_element"), element, elem_type);

*/


#include "ParFUM_TOPS.h"
#include "ParFUM.decl.h"

TopModel* topModel_Create(){
  // This only uses a single mesh, so better not create multiple ones of these
  int which_mesh=FEM_Mesh_default_read();
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_NODE);
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_ELEM+0);
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_ELEM+1);
  FEM_Mesh *mesh = FEM_Mesh_lookup(which_mesh,"TopModel::TopModel()");

  // Don't Allocate the Global Number attribute for the elements and nodes  
  // It will be automatically created upon calls to void FEM_Entity::setGlobalno(int r,int g) {

  return mesh;
}

TopNode topModel_InsertNode(TopModel* m, double x, double y, double z){
  int newNode = FEM_add_node_local_nolock(m);
  m->node.set_coord(newNode,x,y,z);
  return newNode;
}


/** Set id of a node */
void topNode_SetId(TopModel* m, TopNode n, TopID id){
	// just set the nodal id attribute to id
  m->node.setGlobalno(n,id);
}

/** Set attribute of a node */
void topNode_SetAttrib(TopModel*, TopNode, NodeAtt*){

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
void topElement_SetAttrib(TopModel*, TopElement, ElemAtt*){

}

/** Get node via id */
TopNode topModel_GetNodeAtId(TopModel*,TopID);

/** Get elem via id */
TopElement topModel_GetElemAtId(TopModel*,TopID);




#include "ParFUM_TOPS.def.h"
