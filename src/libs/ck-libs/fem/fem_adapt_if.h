#ifndef __CHARM_FEM_ADAPT_IF_H
#define __CHARM_FEM_ADAPT_IF_H

#include "charm-api.h"
#include "fem_mesh_modify.h"
#include "fem_adapt_algs.h"

extern void _registerFEMMeshModify(void);

void FEM_ADAPT_Init(int meshID) {
  _registerFEMMeshModify();
  FEM_REF_INIT(meshID, 2);  // dim=2
  FEM_Mesh *meshP = FEM_Mesh_lookup(meshID, "FEM_ADAPT_Init");
  CtvInitialize(FEM_Adapt_Algs *, _adaptAlgs);
  CtvAccess(_adaptAlgs) = meshP->getfmMM()->getfmAdaptAlgs();
  CtvAccess(_adaptAlgs)->FEM_Adapt_Algs_Init(FEM_DATA+0, FEM_BOUNDARY);
}
FDECL void FTN_NAME(FEM_ADAPT_INIT,fem_adapt_init)(int *meshID)
{
  FEM_ADAPT_Init(*meshID);
}


void FEM_ADAPT_Refine(int qm, int method, double factor, double *sizes) {
  CtvAccess(_adaptAlgs)->FEM_Refine(qm, method, factor, sizes);
}
FDECL void FTN_NAME(FEM_ADAPT_REFINE,fem_adapt_refine)(int *qm, int *method, double *factor, double *sizes)
{
  FEM_ADAPT_Refine(*qm, *method, *factor, sizes);
}


void FEM_ADAPT_Coarsen(int qm, int method, double factor, double *sizes) {
  CtvAccess(_adaptAlgs)->FEM_Coarsen(qm, method, factor, sizes);
}
FDECL void FTN_NAME(FEM_ADAPT_COARSEN,fem_adapt_coarsen)(int *qm, int *method, double *factor, double *sizes)
{
  FEM_ADAPT_Coarsen(*qm, *method, *factor, sizes);
}


void FEM_ADAPT_SetElementSizeField(int meshID, int elem, double size) {
  FEM_Mesh *meshP = FEM_Mesh_lookup(meshID, "FEM_ADAPT_Init");
  meshP->elem[0].setMeshSizing(elem, size);
}
FDECL void FTN_NAME(FEM_ADAPT_SETELEMENTSIZEFIELD,fem_adapt_setelementsizefield)(int *meshID, int *elem, double *size)
{
  FEM_ADAPT_SetElementSizeField(*meshID, *elem, *size);
}


void FEM_ADAPT_SetElementsSizeField(int meshID, double *sizes) {
  FEM_Mesh *meshP = FEM_Mesh_lookup(meshID, "FEM_ADAPT_Init");
  int numElements = meshP->elem[0].size();
  for (int i=0; i<numElements; i++) {
    meshP->elem[0].setMeshSizing(i, sizes[i]);
  }
}
FDECL void FTN_NAME(FEM_ADAPT_SETELEMENTSSIZEFIELD,fem_adapt_setelementssizefield)(int *meshID, double *sizes)
{
  FEM_ADAPT_SetElementsSizeField(*meshID, sizes);
}


void FEM_ADAPT_SetReferenceMesh(int meshID) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_Init");

    // for each element, set its size to its average edge length
    // TODO: do we need to run this loop for element types other than 0?
    
    double avgLength;
    int width = mesh->elem[0].getConn().width();
    int* eConn = (int*)malloc(width*sizeof(int));
    int numElements = mesh->elem[0].size();
    
    for (int i=0; i<numElements; ++i, avgLength=0) {
        mesh->e2n_getAll(i, eConn);
        
        for (int j=0; j<width-1; ++j) {
            avgLength += CtvAccess(_adaptAlgs)->length(eConn[j], eConn[j+1]);
        }
        avgLength += CtvAccess(_adaptAlgs)->length(eConn[0], eConn[width-1]);
        avgLength /= width;
        mesh->elem[0].setMeshSizing(i, avgLength);      
    }
    free(eConn);
}
FDECL void FTN_NAME(FEM_ADAPT_SETREFERENCEMESH, fem_adapt_setreferencemesh)(int* meshID)
{
    FEM_ADAPT_SetReferenceMesh(*meshID);
}

#endif
