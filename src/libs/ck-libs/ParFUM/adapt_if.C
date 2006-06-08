/*
 * The user visible adaptivity interface
 *
 */


#include "ParFUM.h"
#include "ParFUM_internals.h"


CDECL void FEM_REF_INIT(int mesh) {
  CkArrayID femRefId;
  int cid;
  int size;
  TCharm *tc=TCharm::get();
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&cid);
  MPI_Comm_size(comm,&size);
  if(cid==0) {
    CkArrayOptions opts;
    opts.bindTo(tc->getProxy()); //bind to the current proxy
    femMeshModMsg *fm = new femMeshModMsg;
    femRefId = CProxy_femMeshModify::ckNew(fm, opts);
  }
  MPI_Bcast(&femRefId, sizeof(CkArrayID), MPI_BYTE, 0, comm);
  meshMod = femRefId;
  femMeshModMsg *fm = new femMeshModMsg(size,cid);
  meshMod[cid].insert(fm);
  FEM_Mesh *m=FEM_Mesh_lookup(mesh,"FEM_REF_INIT");
  FEMMeshMsg *msg = new FEMMeshMsg(m,tc); 
  meshMod[cid].setFemMesh(msg);
  return;
}

void FEM_ADAPT_Init(int meshID) {
  const int triangleFaces[6] = {0,1,1,2,2,0};
  FEM_Add_elem2face_tuples(meshID, 0, 2, 3, triangleFaces);
  FEM_Mesh_allocate_valid_attr(meshID, FEM_ELEM+0);
  FEM_Mesh_allocate_valid_attr(meshID, FEM_NODE);
  FEM_Mesh_create_elem_elem_adjacency(meshID);
  FEM_Mesh_create_node_elem_adjacency(meshID);
  FEM_Mesh_create_node_node_adjacency(meshID);
  //_registerParFUM();
  FEM_REF_INIT(meshID);
  FEM_Mesh *meshP = FEM_Mesh_lookup(meshID, "FEM_ADAPT_Init");
  CtvInitialize(FEM_Adapt_Algs *, _adaptAlgs);
  CtvAccess(_adaptAlgs) = meshP->getfmMM()->getfmAdaptAlgs();
  CtvAccess(_adaptAlgs)->FEM_Adapt_Algs_Init(FEM_DATA+0, FEM_BOUNDARY,2); //dimension=2
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

void FEM_ADAPT_TestMesh(int meshID) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_GradateMesh");
    mesh->getfmMM()->getfmAdaptAlgs()->tests(true);
}
FDECL void FTN_NAME(FEM_ADAPT_TESTMESH, fem_adapt_testmesh)(int* meshID)
{
    FEM_ADAPT_TestMesh(*meshID);
}

int FEM_ADAPT_SimpleRefineMesh(int meshID, double targetA, double xmin, double ymin, double xmax, double ymax) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_GradateMesh");
    return mesh->getfmMM()->getfmAdaptAlgs()->simple_refine(targetA,xmin,ymin,xmax,ymax);
}
FDECL void FTN_NAME(FEM_ADAPT_SIMPLEREFINEMESH, fem_adapt_simplerefinemesh)(int* meshID, double* targetA, double* xmin, double* ymin, double* xmax, double* ymax)
{
    FEM_ADAPT_SimpleRefineMesh(*meshID,*targetA,*xmin,*ymin,*xmax,*ymax);
}

int FEM_ADAPT_SimpleCoarsenMesh(int meshID, double targetA, double xmin, double ymin, double xmax, double ymax) {
    FEM_Mesh* mesh = FEM_Mesh_lookup(meshID, "FEM_ADAPT_GradateMesh");
    return mesh->getfmMM()->getfmAdaptAlgs()->simple_coarsen(targetA,xmin,ymin,xmax,ymax);
}
FDECL void FTN_NAME(FEM_ADAPT_SIMPLECOARSENMESH, fem_adapt_simplecoarsenmesh)(int* meshID, double* targetA, double* xmin, double* ymin, double* xmax, double* ymax)
{
    FEM_ADAPT_SimpleCoarsenMesh(*meshID,*targetA,*xmin,*ymin,*xmax,*ymax);
}
