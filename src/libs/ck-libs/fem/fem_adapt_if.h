#ifndef __CHARM_FEM_ADAPT_IF_H
#define __CHARM_FEM_ADAPT_IF_H

#include "charm-api.h"
#include "fem_adapt_algs.h"
#include "fem_mesh_modify.h"

extern void _registerFEMMeshModify(void);

void FEM_ADAPT_Init(int meshID);
FDECL void FTN_NAME(FEM_ADAPT_INIT,fem_adapt_init)(int *meshID);


void FEM_ADAPT_Refine(int meshID, int qm, int method, double factor, double *sizes);
FDECL void FTN_NAME(FEM_ADAPT_REFINE,fem_adapt_refine)(int* meshID, 
        int *qm, int *method, double *factor, double *sizes);


void FEM_ADAPT_Coarsen(int meshID, int qm, int method, double factor, 
        double *sizes);
FDECL void FTN_NAME(FEM_ADAPT_COARSEN,fem_adapt_coarsen)(int* meshID, 
        int *qm, int *method, double *factor, double *sizes);

void FEM_ADAPT_AdaptMesh(int meshID, int qm, int method, double factor, double *sizes);
FDECL void FTN_NAME(FEM_ADAPT_ADAPTMESH,fem_adapt_adaptmesh)(int* meshID, 
        int *qm, int *method, double *factor, double *sizes);

void FEM_ADAPT_SetElementSizeField(int meshID, int elem, double size);
FDECL void FTN_NAME(FEM_ADAPT_SETELEMENTSIZEFIELD,fem_adapt_setelementsizefield)(int *meshID, int *elem, double *size);


void FEM_ADAPT_SetElementsSizeField(int meshID, double *sizes);
FDECL void FTN_NAME(FEM_ADAPT_SETELEMENTSSIZEFIELD,fem_adapt_setelementssizefield)(int *meshID, double *sizes);


void FEM_ADAPT_SetReferenceMesh(int meshID);
FDECL void FTN_NAME(FEM_ADAPT_SETREFERENCEMESH, fem_adapt_setreferencemesh)(int* meshID);


void FEM_ADAPT_GradateMesh(int meshID, double smoothness);
FDECL void FTN_NAME(FEM_ADAPT_GRADATEMESH, fem_adapt_gradatemesh)(int* meshID, double* smoothness);

#endif
