

/* File: adapt_algs.h
 * Authors: Terry Wilmarth, Nilesh Choudhury
 * 
 */

#ifndef __ParFUM_Adapt_Algs_H
#define __ParFUM_Adapt_Algs_H

#define SLIVERAREA 1.0e-18
#define REFINE_TOL 1.1  // Refine elements with average edge length > 
                        // REFINE_TOL*desiredEdgeLength
#define COARSEN_TOL 0.8 // Coarsen element with average edge length <
                        // COARSEN_TOL*desiredEdgeLength
#define QUALITY_MIN 0.7

class FEM_Adapt_Algs;
CtvExtern(FEM_Adapt_Algs *, _adaptAlgs);

class femMeshModify;
class FEM_Adapt;
class FEM_AdaptL;

/// Provides high level adaptivity operations (by calling the primitive operations repetitively)
/** This module implements high level mesh adaptivity algorithms that make use 
 * of the primitive mesh adaptivity operations provided by ParFUM_Adapt.
 */
class FEM_Adapt_Algs {
  friend class FEM_AdaptL;
  friend class FEM_Adapt;
  friend class femMeshModify;
  friend class FEM_Interpolate;
  friend class FEM_MUtil;

 public:
  /// attribute index for coordinates
  int coord_attr;
  /// attribute index for boundaries
  int bc_attr;

 protected: 
  ///cross-pointer to theMesh object on this chunk
  FEM_Mesh *theMesh;
  ///cross-pointer to the femMeshModify object on this chunk
  femMeshModify *theMod;
  ///cross-pointer to the FEM_AdaptL object for this chunk
  FEM_AdaptL *theAdaptor;
  /// Number of nodes on this chunk
  int numNodes;
  /// Number of elements on this chunk
  int numElements;
  /// The number of dimensions of this mesh (adaptivity works only for 2D)
  int dim;
  /// This is a heap data structure used to sort elements
  typedef struct {
    int elID;
    double len;
  } elemHeap;
  /// A heap pointer created while coarsening (to order from smallest to largest length)
  elemHeap *coarsenElements;
  /// A heap pointer created while refining (to order from largest to smallest length)
  elemHeap *refineElements;
  /// A stack of elements used for refine
  elemHeap *refineStack;
  /// The number of entries in the coarsen heap
  int coarsenHeapSize;
  /// The number of entries in the refine heap
  int refineHeapSize;
  /// The number of entries in refinestack
  int refineTop;
  /// Used for populating the e2n adjacency connectivity for an element (avoids multiple memory allocations)
  int *elemConn;
  /// Coordinates of the three nodes which form the e2n of an element (avoids multiple memory allocations)
  double *coordsn1, *coordsn2, *coordsn3;

 public:
  ///default constructor
  FEM_Adapt_Algs() {
    theMesh = NULL; theMod = NULL; theAdaptor = NULL;
  }
  /// Initialize FEM_Adapt_Algs with a chunk of the mesh
  FEM_Adapt_Algs(FEM_Mesh *m, femMeshModify *fm);
  /// Initialize FEM_Adapt with the femMeshModify object
  FEM_Adapt_Algs(femMeshModify *fm);
  /// Initialize FEM_Adapt with the FEM_Mesh object
  void FEM_AdaptAlgsSetMesh(FEM_Mesh *m) {theMesh = m;}
  /// Initialize the coord_attr and boundary attr and number of dimensions for this mesh
  void FEM_Adapt_Algs_Init(int coord_at, int bc_at, int dimension) {
    coord_attr = coord_at;
    bc_attr = bc_at;
    dim = dimension;
  }
  /// default destructor
  ~FEM_Adapt_Algs();
  /// pup for this object 
  void pup(PUP::er &p) {
    p|coord_attr;
    p|bc_attr;
    p|dim;
  }

  /// Perform refinements on a mesh
  void FEM_Refine(int qm, int method, double factor, double *sizes);
  /// Perform coarsening on a mesh
  void FEM_Coarsen(int qm, int method, double factor, double *sizes);
  /// Perform refinement/coarsening on a mesh
  void FEM_AdaptMesh(int qm, int method, double factor, double *sizes);
  /// Smooth the mesh using method according to some quality measure qm
  void FEM_Smooth(int qm, int method);
  /// Repair the mesh according to some quality measure qm
  void FEM_mesh_smooth(FEM_Mesh *meshP, int *nodes, int nNodes, int attrNo);
  /// Repair the bad quality elements of a mesh
  void FEM_Repair(int qm);
  /// Remesh entire mesh
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
  /// Get next element to be refined/coarsenen
  int Delete_Min(int cFlag);

 public:
  /// Initiate instance of Longest Edge Bisection on an element
  int refine_element_leb(int e);
  /// The propagating function for Longest Edge Bisection
  void refine_flip_element_leb(int e, int p, int n1, int n2, 
				       double le);

  /// A simple refine algorithm that refines all elements in a region to areas larger than targetA
  int simple_refine(double targetA, double xmin=0.0, double ymin=0.0, double xmax=1.0, double ymax=1.0);
  /// A simple coarsen algorithm that coarsen all elements in a region to areas smaller than targetA
  int simple_coarsen(double targetA, double xmin=0.0, double ymin=0.0, double xmax=1.0, double ymax=1.0);

  /// Returns the length of the edge formed by nodes n1, n2
  double length(int n1, int n2);
  /// Returns the length between these two points
  double length(double *n1_coord, double *n2_coord);
  /// Returns the area of the triangle formed by n1, n2 aand n3
  double getArea(int n1, int n2, int n3);
  /// Returns the area between these three points
  double getArea(double *n1_coord, double *n2_coord, double *n3_coord);
  /// Returns the signed area of the triangle formed by the nodes in that order
  double getSignedArea(int n1, int n2, int n3);
  /// Returns the signed area of the triangle formed by the point coordinates in that order
  double getSignedArea(double *n1_coord, double *n2_coord, double *n3_coord);
  /// Populates crds with the coordinates of the node n1
  int getCoord(int n1, double *crds);
  /// Retuns the shortest edge for the triangle formed by n1, n2, n3
  int getShortestEdge(int n1, int n2, int n3, int* shortestEdge);

  /// Retuens the quality metric for this element
  double getAreaQuality(int elem);
  /// Ensure the quality of the triangle formed by these three nodes
  void ensureQuality(int n1, int n2, int n3);
  /// Verify if flip(n1,n2) will create bad quality elements
  bool controlQualityF(int n1, int n2, int n3, int n4);
  /// Verify if bisect(n1,n2) will create bad quality elements
  bool controlQualityR(int n1, int n2, int n3, int n4);
  /// Same as above; instead of node indices, it takes point coordinates as input
  bool controlQualityR(double *n1_coord, double *n2_coord, double *n3_coord);
  /// Returns true if the quality will become bad if element (n1,n2,n3) changes to (n1,n2,n4)
  bool controlQualityC(int n1, int n2, int n3, double *n4_coord);
  /// same as above only takes in point coordinates instead of node indices
  bool controlQualityC(double *n1_coord, double *n2_coord, double *n3_coord, double *n4_coord);

  /// Decide based on quality metrics if a flip or bisect is good for this element
  bool flipOrBisect(int elId, int n1, int n2, int maxEdgeIdx, double maxlen);
  /// A series of tests to maintain validity of mesh structure, area, IDXL lists, etc
  void tests(bool b);
};

// End Adapt Algs

#endif
