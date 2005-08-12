// This module implements high level mesh adaptivity algorithms that make use 
// of the primitive mesh adaptivity operations provided by fem_adapt(_new).
// Ask: TLW
#ifndef __CHARM_FEM_ADAPT_ALGS_H
#define __CHARM_FEM_ADAPT_ALGS_H

#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"

class femMeshModify;
class FEM_Adapt;
class FEM_AdaptL;

class FEM_Adapt_Algs {
  FEM_Mesh *theMesh;
  femMeshModify *theMod;
  //FEM_Adapt *theAdaptor;
  FEM_AdaptL *theAdaptor;
  int numElements, numNodes, dim;
  double *regional_sizes;
  double *nodeCoords;
  int coord_attr;

 public:
  FEM_Adapt_Algs() {
    theMesh = NULL; theMod = NULL; theAdaptor = NULL;
  }
  /// Initialize FEM_Adapt_Algs with a chunk of the mesh
  FEM_Adapt_Algs(FEM_Mesh *m, femMeshModify *fm, int dimension);
  /// Perform refinements on a mesh
  /** Perform refinements on a mesh.  Tries to maintain/improve element quality
      as specified by a quality measure qm;
      if method = 0, refine areas with size larger than factor down to factor
      if method = 1, refine elements down to sizes specified in sizes array
      Negative entries in size array indicate no refinement. */
  void FEM_Refine(int qm, int method, double factor, double *sizes, 
		  double *coord);
  /// Perform coarsening on a mesh
  /** Perform coarsening on a mesh.  Tries to maintain/improve element quality
      as specified by a quality measure qm;
      if method = 0, coarsen areas with size smaller than factor up to factor
      if method = 1, coarsen elements up to sizes specified in sizes array
      Negative entries in size array indicate no coarsening. */
  void FEM_Coarsen(int qm, int method, double factor, double *sizes, 
		   double *coord);
  /// Smooth the mesh using method according to some quality measure qm
  void FEM_Smooth(int qm, int method, double *coord);
  /// Repair the mesh according to some quality measure qm
  void FEM_Repair(int qm, double *coord);
  /// Remesh entire mesh
  /** Remesh entire mesh according to quality measure qm
      if method = 0, set entire mesh size to factor
      if method = 1, keep regional mesh sizes, and scale by factor
      if method = 2, uses sizes to size mesh by regions */
  void FEM_Remesh(int qm, int method, double factor, double *sizes, 
		  double *coord);
 private:
  // Helper methods
  /// Performs refinement; returns number of modifications
  int Refine(int qm, int method, double factor, double *sizes);
  /// Performs coarsening; returns number of modifications
  int Coarsen(int qm, int method, double factor, double *sizes);
  /// Set sizes on elements throughout the mesh; note: size is edge length
  void SetMeshSize(int method, double factor, double *sizes);
  /// Initiate instance of Longest Edge Bisection on an element
  /** Initiate instance of Longest Edge Bisection on element e.  Propagates
      throughout the mesh to maintain the requirement that only longest edges
      are bisected; returns 1 if successful, 0 if not **/
 public:
  /// Initialize numNodes, numElements and coords
  void Adapt_Init(double *coord);
  void Adapt_Init(int a); //this should be the correct interface
  virtual int refine_element_leb(int e);
  virtual void refine_flip_element_leb(int e, int p, int n1, int n2, 
				       double le);

  int simple_refine(double targetA);
  int simple_coarsen(double targetA);
  double length(int n1, int n2);
  double getArea(int n1, int n2, int n3);
  double length(double *n1_coord, double *n2_coord);
  double getArea(double *n1_coord, double *n2_coord, double *n3_coord);
  int getCoord(int n1, double *crds);
  int getShortestEdge(int n1, int n2, int n3, int* shortestEdge);
};


#endif
