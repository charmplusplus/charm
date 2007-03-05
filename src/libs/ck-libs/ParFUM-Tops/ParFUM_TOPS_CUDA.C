/**

   @file
   @brief Implementation of ParFUM-TOPS layer, except for Iterators

   @author Isaac Dooley, Aaron Becker

*/

#include "ParFUM_TOPS.h"
#include "ParFUM.decl.h"
#include "ParFUM_internals.h"


__device__ void* topElement_D_GetAttrib(TopModel* m, TopElement e){
  if(! m->elem[0].is_valid_any_idx(e))
	return NULL;

  FEM_DataAttribute * at = (FEM_DataAttribute*) m->elem[0].lookup(FEM_DATA+0,"topElem_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (data + e*elem_attr_size);
}


__device__ void* topNode_D_GetAttrib(TopModel* m, TopNode n){
  if(! m->node.is_valid_any_idx(n))
	return NULL;

  FEM_DataAttribute * at = (FEM_DataAttribute*) m->node.lookup(FEM_DATA+0,"topNode_GetAttrib");
  AllocTable2d<unsigned char> &dataTable  = at->getChar();
  unsigned char *data = dataTable.getData();
  return (data + n*node_attr_size);
}


__device__ TopNode topElement_D_GetNode(TopModel* m,TopElement e,int idx){
  CkAssert(e>=0);
  const AllocTable2d<int> &conn = m->elem[0].getConn();
  CkAssert(idx>=0 && idx<conn.width());
  CkAssert(idx<conn.size());

  int node = conn(e,idx);

  return conn(e,idx);
}

#include "ParFUM_TOPS.def.h"
