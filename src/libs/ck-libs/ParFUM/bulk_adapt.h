/* File: bulk_adapt.h
 * Author: Terry Wilmarth
 */
#ifndef __ParFUM_Bulk_Adapt_H
#define __ParFUM_Bulk_Adapt_H

#define SLIVERAREA 1.0e-18
#define REFINE_TOL 1.1  // Refine elements with average edge length > 
                        // REFINE_TOL*desiredEdgeLength
#define COARSEN_TOL 0.8 // Coarsen element with average edge length <
                        // COARSEN_TOL*desiredEdgeLength
#define QUALITY_MIN 0.7

// Refine Methods
#define SCALED_SIZING 0
#define ABSOLUTE_SIZING 1

class Bulk_Adapt;
CtvExtern(Bulk_Adapt *, _bulkAdapt);

/// Type for sorted element storage while awaiting adaptivity.
/** Elements are sorted by some real valued attribute associated with
 *  them.  This is typically edge length or element quality.
 *  Iteration through the data structure should not produce repeats in
 *  a single iteration through elements to be refined.  Thus, visited
 *  elements that are reinserted are buffered in a stack until the end of an
 *  iteration.
*/
class Element_Bucket {
 public:
  /// This is a heap data structure used to sort elements
  typedef struct {
    int elID;
    double len;
  } elemHeap;
  /// Elements to coarsen (ordered smallest to largest)
  elemHeap *coarsenElements;
  /// Elements to refine (ordered largest to smallest)
  elemHeap *refineElements;
  /// A stack of elements used for refine
  elemHeap *refineStack;
  /// The number of entries in the coarsen heap
  int coarsenHeapSize;
  /// The number of entries in the refine heap
  int refineHeapSize;
  /// The number of entries in refinestack
  int refineTop;
  Element_Bucket() { 
    coarsenHeapSize = refineHeapSize = refineTop = 0;
    coarsenElements = refineElements = refineStack = NULL;
  }
  void Reset(int numElements) {
    if (refineStack) delete [] refineStack;
    refineStack = new elemHeap[numElements];
    if (refineElements) delete [] refineElements;
    refineElements = new elemHeap[numElements+1];
  }
  bool RefineEmpty() {
    return((refineHeapSize==0) && (refineTop==0));
  }
  int Delete_Top() {
    if (refineTop > 0) {
      refineTop--;
      return refineStack[refineTop].elID;
    }
    return -1;
  }
  /// Insert element to be refined/coarsened
  void Insert(int elID, double len, int cFlag);
  /// Get next element to be refined/coarsenen
    int Delete_Min(int cFlag, double *len);
};


/// Provides high level adaptivity operations
/** This module implements high level mesh adaptivity algorithms that make use 
    of the bulk mesh adaptivity operations provided by bulk_adapt_ops.
*/
class Bulk_Adapt {
  friend class FEM_Interpolate;
 protected: 
  ///cross-pointer to theMesh object on this chunk
  FEM_Mesh *theMesh; 
  int theMeshID;
  /// The number of dimensions of this mesh (adaptivity works only for 2D)
    int dim, elemType;
 public:
  ///default constructor
  Bulk_Adapt() { theMesh = NULL; }
  /// Initialize Bulk_Adapt_Algs with a partition of the mesh
  Bulk_Adapt(FEM_Mesh *m, int d) { theMesh = m; dim=d; }
  /// Initialize FEM_Adapt with the FEM_Mesh object
  void Bulk_Adapt_SetMesh(FEM_Mesh *m) { theMesh = m; }
  /// Initialize the coord_attr and boundary attr and number of dimensions for this mesh
  void Bulk_Adapt_SetDim(int dimension) { dim = dimension; }
  void Bulk_Adapt_SetElemType(int et) { elemType = et; }
  /// default destructor
  ~Bulk_Adapt() {}
  /// pup for this object 
  void pup(PUP::er &p) {
    p|dim;
  }

  void Bulk_Adapt_Init(int mid, FEM_Mesh *mp, int d) { 
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    theMesh = mp;
    theMeshID = mid;
    dim = d;
    CkPrintf("[%d] Adapt init...\n", rank);
    FEM_Mesh_allocate_valid_attr(theMeshID,FEM_NODE);
    FEM_Mesh_allocate_valid_attr(theMeshID,FEM_ELEM);
    MPI_Barrier(MPI_COMM_WORLD);
    CkPrintf("[%d] Creating adapt adjacencies...\n", rank);
    CreateAdaptAdjacencies(theMeshID, 0);
    ParFUM_SA_Init(theMeshID);  
    MPI_Barrier(MPI_COMM_WORLD);
    CkPrintf("[%d] End Adapt init...\n", rank);
  }

  /// Perform refinements on a mesh
  void ParFUM_Refine(int qm, int method, double factor, double *sizes);
  /// Perform coarsening on a mesh
  void ParFUM_Coarsen(int qm, int method, double factor, double *sizes);
  /// Perform refinement/coarsening on a mesh
  void ParFUM_AdaptMesh(int qm, int method, double factor, double *sizes);
  /// Smooth the mesh using method according to some quality measure qm
  void ParFUM_Smooth(int qm, int method);
  /// Repair the bad quality elements of a mesh
  void ParFUM_Repair(int qm);
  /// Remesh entire mesh
  void ParFUM_Remesh(int qm, int method, double factor, double *sizes);
  
  /// Set sizes on mesh elements based on their average edge length 
  void ParFUM_SetReferenceMesh();
  /// Adjust sizes on mesh elements to avoid sharp discontinuities 
  void ParFUM_GradateMesh(double smoothness);
  void findEdgeLengths(int elemID, double *avgEdgeLength, double *maxEdgeLength, int *maxEdge, 
		       double *minEdgeLength, int *minEdge);

 private:
  // Helper methods
  /// Performs refinement; returns number of modifications
  int Refine_h(int qm, int method, double factor, double *sizes);
  /// Performs coarsening; returns number of modifications
  int Coarsen_h(int qm, int method, double factor, double *sizes);

  /// Set sizes on elements throughout the mesh; note: size is edge length
  void SetMeshSize(int method, double factor, double *sizes);

  /// Returns the length of the edge formed by nodes n1, n2
  double length(int n1, int n2);
  double length3D(int n1, int n2);
  /// Returns the length between these two points
  double length(double *n1_coord, double *n2_coord);
  double length3D(double *n1_coord, double *n2_coord);

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

  /// Returns the quality metric for this element
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
  bool flipOrBisect(int elId, int n1, int n2, int maxEdgeIdx, double maxlen);
};

#endif
