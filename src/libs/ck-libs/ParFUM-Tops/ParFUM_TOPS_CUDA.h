/**
   @file
   @brief A ParFUM "Tops" compatibility layer API Definition

   @author Isaac Dooley and Aaron

   ParFUM-TOPS provides a Tops-like API for ParFUM, meant to run on CUDA NVIDIA system.

*/

#ifndef __PARFUM_TOPS_CUDA___H
#define __PARFUM_TOPS_CUDA___H

#include <ParFUM.h>
#include <ParFUM_internals.h>
#include "ParFUM_TOPS.h"


void* topElement_D_GetAttrib(TopModel* m, TopElement e);

void* topNode_D_GetAttrib(TopModel* m, TopNode n);

TopNode topElement_D_GetNode(TopModel* m,TopElement e,int idx);

#endif
