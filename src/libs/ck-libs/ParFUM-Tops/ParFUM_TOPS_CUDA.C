/**

   @file
   @brief Implementation of ParFUM-TOPS layer, except for Iterators

   @author Isaac Dooley, Aaron Becker

*/

#include "ParFUM_TOPS.h"
#include "ParFUM.decl.h"
#include "ParFUM_internals.h"


__device__ void* topElement_D_GetAttrib(TopModel* m, TopElement e){
  return (m->ElemDataDevice + e*m->elem_attr_size);
}


__device__ void* topNode_D_GetAttrib(TopModel* m, TopNode n){
  return (m->NodeDataDevice + n*m->node_attr_size);
}


#include "ParFUM_TOPS.def.h"
