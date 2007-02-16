/**
*	A ParFUM TOPS compatibility layer
*
*      Author: Isaac Dooley

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
  
  // Allocate an ID attribute for the elements and nodes
  
  //	FEM_GLOBALNO
  
  return FEM_Mesh_lookup(which_mesh,"TopModel::TopModel()");
}

TopNode topModel_InsertNode(TopModel* m, double x, double y, double z){
  // TODO : insert a node here
  int which = FEM_add_node_local_nolock(m);
  m->node.set_coord(which,x,y,z);

  TopNode a;
  return a;
}


/** Set id of a node */
void topNode_SetId(TopModel*, TopNode, TopID id){
	// just set the nodal id attribute to id
}

/** Set attribute of a node */
void topNode_SetAttrib(TopModel*, TopNode, NodeAtt*){

}

/** Insert an element */
TopElement topModel_InsertElem(TopModel*, TopElemType, TopNode*){

}

/** Set id of an element */
void topElement_SetId(TopModel*, TopElement, TopID id){

}

/** Set attribute of an element */
void topElement_SetAttrib(TopModel*, TopElement, ElemAtt*){


}

/** Get node via id */
TopNode topModel_GetNodeAtId(TopModel*,TopID);

/** Get elem via id */
TopElement topModel_GetElemAtId(TopModel*,TopID);




#include "ParFUM_TOPS.def.h"
