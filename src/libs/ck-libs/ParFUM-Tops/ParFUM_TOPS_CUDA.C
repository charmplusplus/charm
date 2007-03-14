/**

   @file
   @brief Implementation of ParFUM-TOPS layer, except for Iterators

   @author Isaac Dooley, Aaron Becker

*/

#include "ParFUM_TOPS_CUDA.h"


#ifdef CUDA
__device__ void* topElement_D_GetAttrib(TopModelDevice* m, TopElement e){
  return (m->ElemDataDevice + e*m->elem_attr_size);
}


__device__ void* topNode_D_GetAttrib(TopModelDevice* m, TopNode n){
  return (m->NodeDataDevice + n*m->node_attr_size);
}
#endif
