#include "fem_adapt_if.h"


void FEM_ADAPT_Init(int meshID) {
  const int triangleFaces[6] = {0,1,1,2,2,0};
  FEM_Add_elem2face_tuples(meshID, 0, 2, 3, triangleFaces);
  FEM_Mesh_create_elem_elem_adjacency(meshID);
  FEM_Mesh_create_node_elem_adjacency(meshID);
  FEM_Mesh_create_node_node_adjacency(meshID);
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


void FEM_ADAPT_Refine(int meshID, int qm, int method, double factor,
        double *sizes) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_Refine");
    mesh->getfmMM()->getfmAdaptAlgs()->FEM_Refine(qm, method, factor, sizes);
}
FDECL void FTN_NAME(FEM_ADAPT_REFINE,fem_adapt_refine)(int* meshID, 
        int *qm, int *method, double *factor, double *sizes)
{
  FEM_ADAPT_Refine(*meshID, *qm, *method, *factor, sizes);
}


 void FEM_ADAPT_Coarsen(int meshID, int qm, int method, double factor, 
        double *sizes) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_Coarsen");
    mesh->getfmMM()->getfmAdaptAlgs()->FEM_Coarsen(qm, method, factor, sizes);
}
FDECL  void FTN_NAME(FEM_ADAPT_COARSEN,fem_adapt_coarsen)(int* meshID, 
        int *qm, int *method, double *factor, double *sizes)
{
  FEM_ADAPT_Coarsen(*meshID, *qm, *method, *factor, sizes);
}

void FEM_ADAPT_AdaptMesh(int meshID, int qm, int method, double factor,
        double *sizes) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_AdaptMesh");
    mesh->getfmMM()->getfmAdaptAlgs()->FEM_AdaptMesh(qm, method, factor, sizes);
}
FDECL void FTN_NAME(FEM_ADAPT_ADAPTMESH,fem_adapt_adaptmesh)(int* meshID, 
        int *qm, int *method, double *factor, double *sizes)
{
  FEM_ADAPT_AdaptMesh(*meshID, *qm, *method, *factor, sizes);
}


 void FEM_ADAPT_SetElementSizeField(int meshID, int elem, double size) {
  FEM_Mesh *meshP = FEM_Mesh_lookup(meshID, "FEM_ADAPT_SetElementSizeField");
  meshP->elem[0].setMeshSizing(elem, size);
}
FDECL  void FTN_NAME(FEM_ADAPT_SETELEMENTSIZEFIELD,fem_adapt_setelementsizefield)(int *meshID, int *elem, double *size)
{
  FEM_ADAPT_SetElementSizeField(*meshID, *elem, *size);
}


 void FEM_ADAPT_SetElementsSizeField(int meshID, double *sizes) {
  FEM_Mesh *meshP = FEM_Mesh_lookup(meshID, "FEM_ADAPT_SetElementsSizeField");
  int numElements = meshP->elem[0].size();
  for (int i=0; i<numElements; i++) {
    meshP->elem[0].setMeshSizing(i, sizes[i]);
  }
}
FDECL  void FTN_NAME(FEM_ADAPT_SETELEMENTSSIZEFIELD,fem_adapt_setelementssizefield)(int *meshID, double *sizes)
{
  FEM_ADAPT_SetElementsSizeField(*meshID, sizes);
}


void FEM_ADAPT_SetReferenceMesh(int meshID) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_SetReferenceMesh");
    mesh->getfmMM()->getfmAdaptAlgs()->SetReferenceMesh();
}
FDECL void FTN_NAME(FEM_ADAPT_SETREFERENCEMESH, fem_adapt_setreferencemesh)(int* meshID)
{
    FEM_ADAPT_SetReferenceMesh(*meshID);
}


void FEM_ADAPT_GradateMesh(int meshID, double smoothness)
{
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_GradateMesh");
    mesh->getfmMM()->getfmAdaptAlgs()->GradateMesh(smoothness);
}
FDECL void FTN_NAME(FEM_ADAPT_GRADATEMESH, fem_adapt_gradatemesh)(int* meshID, double* smoothness)
{
    FEM_ADAPT_GradateMesh(*meshID, *smoothness);
}

