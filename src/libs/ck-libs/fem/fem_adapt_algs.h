/* File: fem_adapt_algs.h
 * Authors: Terry Wilmarth, Nilesh Choudhury
 * 
 */

// This module implements high level mesh adaptivity algorithms that make use 
// of the primitive mesh adaptivity operations provided by fem_adapt(_new).
// Ask: TLW
#ifndef __CHARM_FEM_ADAPT_ALGS_H
#define __CHARM_FEM_ADAPT_ALGS_H

#include "charm++.h"
#include "tcharm.h"
#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"
#include "vector2d.h"  // **CW** tentative

#define SLIVERAREA 1.0e-18
#define REFINE_TOL 1.1  // Refine elements with average edge length > 
                        // REFINE_TOL*desiredEdgeLength
#define COARSEN_TOL 0.8 // Coarsen element with average edge length <
                        // COARSEN_TOL*desiredEdgeLength
#define QUALITY_MIN 0.6

class FEM_Adapt_Algs;
CtvExtern(FEM_Adapt_Algs *, _adaptAlgs);

class femMeshModify;
class FEM_Adapt;
class FEM_AdaptL;

class FEM_Adapt_Algs {
  friend class FEM_AdaptL;
  friend class FEM_Adapt;
  friend class femMeshModify;
  friend class FEM_Interpolate;
  friend class FEM_MUtil;

 public:
  int coord_attr;
  int bc_attr;

 protected: 
  FEM_Mesh *theMesh;
  femMeshModify *theMod;
  //FEM_Adapt *theAdaptor;
  FEM_AdaptL *theAdaptor;
  int numNodes, numElements, dim;
  // These are for element sorting
  typedef struct {
    int elID;
    double len;
  } elemHeap;
  elemHeap *coarsenElements;
  elemHeap *refineElements;
  elemHeap *refineStack;
  int refineTop, refineHeapSize, coarsenHeapSize;

 public:
  FEM_Adapt_Algs() {
    theMesh = NULL; theMod = NULL; theAdaptor = NULL;
  }
  /// Initialize FEM_Adapt_Algs with a chunk of the mesh
  FEM_Adapt_Algs(FEM_Mesh *m, femMeshModify *fm, int dimension);
  void FEM_Adapt_Algs_Init(int coord_at, int bc_at) {
    coord_attr = coord_at;
    bc_attr = bc_at;
  }
  /// Perform refinements on a mesh
  /** Perform refinements on a mesh.  Tries to maintain/improve element quality
      as specified by a quality measure qm;
      if method = 0, refine areas with size larger than factor down to factor
      if method = 1, refine elements down to sizes specified in sizes array
      Negative entries in size array indicate no refinement. */
  void FEM_Refine(int qm, int method, double factor, double *sizes);
  /// Perform coarsening on a mesh
  /** Perform coarsening on a mesh.  Tries to maintain/improve element quality
      as specified by a quality measure qm;
      if method = 0, coarsen areas with size smaller than factor up to factor
      if method = 1, coarsen elements up to sizes specified in sizes array
      Negative entries in size array indicate no coarsening. */
  void FEM_Coarsen(int qm, int method, double factor, double *sizes);
  /// Perform refinement/coarsening on a mesh
    /** Same as above */
  void FEM_AdaptMesh(int qm, int method, double factor, double *sizes);
  /// Smooth the mesh using method according to some quality measure qm
  void FEM_Smooth(int qm, int method);
  /// Repair the mesh according to some quality measure qm

  // FEM_Mesh_mooth
  //	Inputs	: meshP - a pointer to the FEM_Mesh object to smooth
  //		: nodes - an array of local node numbers to be smoothed.  Send
  //			  NULL pointer to smooth all nodes.
  //		: nNodes - the size of the nodes array
  //		: attrNo - the attribute number where the coords are registered
  //	Shifts nodes around to improve mesh quality.  FEM_BOUNDARY attribute
  //	and interpolator function must be registered by user to maintain 
  //	boundary information.
  void FEM_mesh_smooth(FEM_Mesh *meshP, int *nodes, int nNodes, int attrNo);

  void FEM_Repair(int qm);
  /// Remesh entire mesh
  /** Remesh entire mesh according to quality measure qm
      if method = 0, set entire mesh size to factor
      if method = 1, keep regional mesh sizes, and scale by factor
      if method = 2, uses sizes to size mesh by regions */
  void FEM_Remesh(int qm, int method, double factor, double *sizes);
  
  /// Set sizes on mesh elements based on their average edge length
  void SetReferenceMesh();
  /// Adjust sizes on mesh elements to avoid sharp discontinuities
  void GradateMesh(double smoothness);
 private:
  // Helper methods
  /// Performs refinement; returns number of modifications
  int Refine(int qm, int method, double factor, double *sizes);
  /// Performs coarsening; returns number of modifications
  int Coarsen(int qm, int method, double factor, double *sizes);
  /// Set sizes on elements throughout the mesh; note: size is edge length
  void SetMeshSize(int method, double factor, double *sizes);
  /// Insert element to be refined/coarsened
  void Insert(int elID, double len, int cFlag);
  /// Get next element to be refined/coarsened
  int Delete_Min(int cFlag);
 public:
  /// Initiate instance of Longest Edge Bisection on an element
  /** Initiate instance of Longest Edge Bisection on element e.  Propagates
      throughout the mesh to maintain the requirement that only longest edges
      are bisected; returns 1 if successful, 0 if not **/
  virtual int refine_element_leb(int e);
  virtual void refine_flip_element_leb(int e, int p, int n1, int n2, 
				       double le);

  int simple_refine(double targetA, double xmin=0.0, double ymin=0.0, double xmax=1.0, double ymax=1.0);
  int simple_coarsen(double targetA, double xmin=0.0, double ymin=0.0, double xmax=1.0, double ymax=1.0);
  double length(int n1, int n2);
  double getArea(int n1, int n2, int n3);
  double length(double *n1_coord, double *n2_coord);
  double getArea(double *n1_coord, double *n2_coord, double *n3_coord);
  int getCoord(int n1, double *crds);
  int getShortestEdge(int n1, int n2, int n3, int* shortestEdge);
  double getAreaQuality(int elem);
  bool didItFlip(int n1, int n2, int n3, double *n4_coord);
  bool didItFlip(double *n1_coord, double *n2_coord, double *n3_coord, double *n4_coord);
  double getSignedArea(double *n1_coord, double *n2_coord, double *n3_coord);
  void tests(void);
};


#endif
