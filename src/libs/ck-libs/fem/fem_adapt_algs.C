// This module implements high level mesh adaptivity algorithms that make use 
// of the primitive mesh adaptivity operations provided by fem_adapt(_new).
// Ask: TLW
#include "fem_adapt_algs.h"
#include "fem_mesh_modify.h"

FEM_Adapt_Algs::FEM_Adapt_Algs(FEM_Mesh *m, femMeshModify *fm, int dimension) 
{ 
  theMesh = m; theMod = fm; dim = dimension; theAdaptor = theMod->fmAdapt;
}

/* Perform refinements on a mesh.  Tries to maintain/improve element quality as
   specified by a quality measure qm; if method = 0, refine areas with size 
   larger than factor down to factor; if method = 1, refine elements down to 
   sizes specified in sizes array; Negative entries in size array indicate no 
   refinement. */
void FEM_Adapt_Algs::FEM_Refine(int qm, int method, double factor, 
				double *sizes, double *coord)
{
  CkPrintf("WARNING: FEM_Refine: Under construction.\n");
  Adapt_Init(coord);
  (void)Refine(qm, method, factor, sizes);
}

/* Performs refinement; returns number of modifications */
int FEM_Adapt_Algs::Refine(int qm, int method, double factor, double *sizes)
{
  SetMeshSize(method, factor, sizes);
  return 0;
}

/* Perform coarsening on a mesh.  Tries to maintain/improve element quality as 
   specified by a quality measure qm; if method = 0, coarsen areas with size 
   smaller than factor up to factor; if method = 1, coarsen elements up to 
   sizes specified in sizes array; Negative entries in size array indicate no 
   coarsening. */
void FEM_Adapt_Algs::FEM_Coarsen(int qm, int method, double factor, 
				 double *sizes, double *coord)
{
  CkPrintf("WARNING: FEM_Coarsen: Under construction.\n");
  Adapt_Init(coord);
  (void)Coarsen(qm, method, factor, sizes);
}

/* Performs coarsening; returns number of modifications */
int FEM_Adapt_Algs::Coarsen(int qm, int method, double factor, double *sizes)
{
  SetMeshSize(method, factor, sizes);
  return 0;
}

/* Smooth the mesh using method according to some quality measure qm */
void FEM_Adapt_Algs::FEM_Smooth(int qm, int method, double *coord)
{
  CkPrintf("WARNING: FEM_Smooth: Not yet implemented.\n");
  Adapt_Init(coord);
}

/* Repair the mesh according to some quality measure qm */
void FEM_Adapt_Algs::FEM_Repair(int qm, double *coord)
{
  CkPrintf("WARNING: FEM_Repair: Not yet implemented.\n");
  Adapt_Init(coord);
}

/* Remesh entire mesh according to quality measure qm. If method = 0, set 
   entire mesh size to factor; if method = 1, use sizes array; if method = 2, 
   uses existing regional sizes and scale by factor*/
void FEM_Adapt_Algs::FEM_Remesh(int qm, int method, double factor, 
				double *sizes, double *coord)
{
  CkPrintf("WARNING: FEM_Remesh: Under construction.\n");
  Adapt_Init(coord);
}

/* Initialize numNodes, numElements and coords */
void FEM_Adapt_Algs::Adapt_Init(double *coord)
{
  nodeCoords = coord;
  numNodes = theMesh->node.size();
  numElements = theMesh->elem[0].size();
}

/* Initialize numNodes, numElements and coords */
void FEM_Adapt_Algs::Adapt_Init(int m, int a)
{
  mesh = m;
  coord_attr = a;
  numNodes = theMesh->node.size();
  numElements = theMesh->elem[0].size();
}

/* Set sizes on elements throughout the mesh; note: size is edge length */
void FEM_Adapt_Algs::SetMeshSize(int method, double factor, double *sizes)
{
  // assumes Adapt_Init has been called for the current algorithm
  if (method == 0) {
    regional_sizes = (double *)malloc(numElements*sizeof(double));
    for (int i=0; i<numElements; i++) {
      regional_sizes[i] = factor;
    }
  }
  else if (method == 1) {
    regional_sizes = (double *)malloc(numElements*sizeof(double));
    for (int i=0; i<numElements; i++) {
      regional_sizes[i] = sizes[i];
    }
  }
  else if (method == 2) {
    regional_sizes = (double *)malloc(numElements*sizeof(double));
    double avgEdgeLength = 0.0;
    for (int i=0; i<numElements; i++) {
      int eConn[4];
      theMesh->e2n_getAll(i, eConn);
      avgEdgeLength = length(eConn[0], eConn[1]) + length(eConn[1], eConn[2]) +
	length(eConn[2], eConn[0]);
      if (dim == 3) {
	avgEdgeLength += length(eConn[0], eConn[3]) + length(eConn[1], eConn[3]) +
	  length(eConn[2], eConn[3]);
	avgEdgeLength /= 6.0;
      }
      else {
	avgEdgeLength /= 3.0;
      }
      regional_sizes[i] = factor * avgEdgeLength;
    }
  }
}

// =====================  BEGIN refine_element_leb ========================= 
/* Given an element e, if e's longest edge f is also the longest edge
   of e's neighbor across f, g, split f by adding a new node in the 
   center of f, and splitting both e and g into two elements.  If g
   does not have f as it's longest edge, recursively call refine_element_leb 
   on g, and start over. */ 
int FEM_Adapt_Algs::refine_element_leb(int e)
{
  int *eConn = (int*)malloc(3*sizeof(int));
  int fixNode, otherNode, opNode, longEdge, nbr; 
  double eLens[3], longEdgeLen = 0.0;
  theMesh->e2n_getAll(e, eConn);
  eLens[0] = length(eConn[0], eConn[1]);
  eLens[1] = length(eConn[1], eConn[2]);
  eLens[2] = length(eConn[2], eConn[0]);
  for (int i=0; i<3; i++)
    if (eLens[i] > longEdgeLen) {
      longEdgeLen = eLens[i];
      longEdge = i;
      fixNode = eConn[i];
      otherNode = eConn[(i+1)%3];
      opNode = eConn[(i+2)%3];
    }
  nbr = theMesh->e2e_getNbr(e, longEdge);
  if (nbr == -1) // e's longEdge is on physical boundary
    return theAdaptor->edge_bisect(fixNode, otherNode);
  int nbrOpNode = theAdaptor->e2n_getNot(nbr, fixNode, otherNode);
  double fixEdgeLen = length(fixNode, nbrOpNode);
  double otherEdgeLen = length(otherNode, nbrOpNode);
  if ((fixEdgeLen > longEdgeLen) || (otherEdgeLen > longEdgeLen)) { 
    // longEdge is not nbr's longest edge
    int newNode = theAdaptor->edge_bisect(fixNode, otherNode);
    int propElem, propNode; // get the element to propagate on
    if (fixEdgeLen > otherEdgeLen) {
      propElem = theAdaptor->findElementWithNodes(newNode, fixNode, nbrOpNode);
      propNode = fixNode;
    }
    else {
      propElem = theAdaptor->findElementWithNodes(newNode, otherNode, nbrOpNode);
      propNode = otherNode;
    }

    //if propElem is a ghost, then it is propagating in a neighboring chunk, otherwise not
    if(!FEM_Is_ghost_index(propElem)) {
      refine_flip_element_leb(propElem, propNode, newNode, nbrOpNode, longEdgeLen);
    }
    else {
      int localChk, nbrChk;
      localChk = theMod->getfmUtil()->getIdx();
      nbrChk = theMod->getfmUtil()->getRemoteIdx(theMesh,propElem,0);
      int propNodeT = theAdaptor->getSharedNodeIdxl(propNode, nbrChk);
      int newNodeT = theAdaptor->getSharedNodeIdxl(newNode, nbrChk);
      int nbrOpNodeT = (nbrOpNode>=0)?(theAdaptor->getSharedNodeIdxl(nbrOpNode, nbrChk)):(theAdaptor->getGhostNodeIdxl(nbrOpNode, nbrChk));
      int propElemT = theAdaptor->getGhostElementIdxl(propElem, nbrChk);
      meshMod[nbrChk].refine_flip_element_leb(localChk, propElemT, propNodeT, newNodeT,nbrOpNodeT,longEdgeLen);
    }
    return newNode;
  }
  else return theAdaptor->edge_bisect(fixNode, otherNode); // longEdge is nbr's long edge
}
void FEM_Adapt_Algs::refine_flip_element_leb(int e, int p, int n1, int n2, double le) 
{
  int newNode = refine_element_leb(e);
  (void) theAdaptor->edge_flip(n1, n2);
  if (length(p, newNode) > le) {
    int localChk = theMod->getfmUtil()->getIdx();
    int newElem = theAdaptor->findElementWithNodes(newNode, n1, p);
    refine_flip_element_leb(newElem, p, n1, newNode, le);
  }
}
// ========================  END refine_element_leb ========================

double FEM_Adapt_Algs::length(int n1, int n2)
{
  //not the correct way to grab coordinates... what abt new nodes???
  //double *n1_coord = &(nodeCoords[n1*dim]), *n2_coord = &(nodeCoords[n2*dim]);
  double *n1_coord = (double*)malloc(2*sizeof(double));
  double *n2_coord = (double*)malloc(2*sizeof(double));

  if(!FEM_Is_ghost_index(n1)) {
    FEM_Mesh_get_data(mesh, FEM_NODE, coord_attr, (void *)n1_coord, n1, 1, FEM_DOUBLE, 2);
  }
  else {
    int ghostidx = FEM_To_ghost_index(n1);
    FEM_Mesh_get_data(mesh, FEM_NODE + FEM_GHOST, coord_attr, (void *)n1_coord, ghostidx, 1, FEM_DOUBLE, 2);
  }
  if(!FEM_Is_ghost_index(n2)) {
    FEM_Mesh_get_data(mesh, FEM_NODE, coord_attr, (void *)n2_coord, n2, 1, FEM_DOUBLE, 2);
  }
  else {
    int ghostidx = FEM_To_ghost_index(n2);
    FEM_Mesh_get_data(mesh, FEM_NODE + FEM_GHOST, coord_attr, (void *)n2_coord, ghostidx, 1, FEM_DOUBLE, 2);
  }
  
  double d, ds_sum=0.0;
  for (int i=0; i<dim; i++) {
    d = n1_coord[i] - n2_coord[i];
    ds_sum += d*d;
  }
  return (sqrt(ds_sum));
}
