
/**
  @file
  @brief A ParFUM Iterators layer for CUDA 

  @author Isaac Dooley
  @author Aaron Becker

  ParFUM-Iterators provides iterators for ParFUM meshes that work on
  a variety of platforms. This is the CUDA implementation.

*/

#ifndef PARFUM_ITERATORS_CUDA_H
#define PARFUM_ITERATORS_CUDA_H

#include "ParFUM_Iterators_Types.h"


#ifdef CUDA

/** A MeshModelDevice contains structures for use by CUDA kernels */
typedef struct {
    unsigned node_attr_size;
    unsigned elem_attr_size;
    unsigned model_attr_size;

    unsigned num_local_elem;
    unsigned num_local_node;

    /** Device pointers to the goods */
    void *mAttDevice;
    void *ElemDataDevice;
    void *NodeDataDevice;
    int *ElemConnDevice;
    int *n2eConnDevice;
} MeshModelDevice;


#define meshElement_D_GetAttrib(m, e) (((char*)(m)->ElemDataDevice) + (e)*(m)->elem_attr_size)
#define meshNode_D_GetAttrib(m, n) (((char*)(m)->NodeDataDevice) + (n)*(m)->node_attr_size)
#define meshElement_D_GetNode(m, e, idx) (((m)->ElemConnDevice)[e*4 + idx])

#endif
#endif
