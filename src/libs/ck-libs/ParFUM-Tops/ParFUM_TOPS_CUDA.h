/**
   @file
   @brief A ParFUM "Tops" compatibility layer API Definition

   @author Isaac Dooley and Aaron

   ParFUM-TOPS provides a Tops-like API for ParFUM, meant to run on CUDA NVIDIA system.

*/

#ifndef __PARFUM_TOPS_CUDA___H
#define __PARFUM_TOPS_CUDA___H
#ifdef CUDA

#include "ParFUM_Tops_Types.h"

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
} TopModelDevice;


void* topElement_D_GetAttrib(TopModelDevice* m, TopElement e);
void* topNode_D_GetAttrib(TopModelDevice* m, TopNode n);
TopNode topElement_D_GetNode(TopModelDevice* m,TopElement e,int idx);

#endif
#endif
