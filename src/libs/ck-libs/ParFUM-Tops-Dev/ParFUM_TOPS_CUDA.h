/**
   @file
   @brief A ParFUM "Tops" compatibility layer API Definition

   @author Isaac Dooley and Aaron

   ParFUM-TOPS provides a Tops-like API for ParFUM, meant to run on CUDA NVIDIA system.

*/

#ifndef __PARFUM_TOPS_CUDA___H
#define __PARFUM_TOPS_CUDA___H

#include "ParFUM_TOPS_Types.h"


#ifdef CUDA

#include <cutil.h>

/** A TopModelDevice contains structures for use by CUDA kernels */
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
} TopModelDevice;


#define topElement_D_GetAttrib(m, e) (((char*)(m)->ElemDataDevice) + (e.idx)*(m)->elem_attr_size)

#define topNode_D_GetAttrib(m, n) (((char*)(m)->NodeDataDevice) + n*(m)->node_attr_size)

#define topElement_D_GetNode(m, e, i) (((m)->ElemConnDevice)[(e.idx) * 4 + i])

#endif
#endif
