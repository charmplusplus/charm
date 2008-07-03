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
  /// This is a bucket data structure used to sort elements
  typedef struct {
    int elID;     // the element ID
    double len;   // the sorting criteria
    int leftIdx;  // lesser elements
    int rightIdx; // greater elements
  } elemEntry;
  /// Element bucket
  elemEntry *elements;
  /// The number of entries in the bucket
  int numElements;
  /// The number of spaces in the bucket
  int bucketSz;
  /// index of the first element in the bucket
  int root;
  Element_Bucket() { 
    root = -1;
    numElements = bucketSz = 0;
    elements = NULL;
  }
  void Alloc(int n) {
    elements = new elemEntry[n];
    for (int i=0; i<n; i++) {
      elements[i].elID = -1;
      elements[i].len = -1.0;
      elements[i].leftIdx = -1;
      elements[i].rightIdx = -1;
    }
    bucketSz = n;
    numElements = 0;
    root = -1;
  }
  void Resize(int n) {
    elemEntry *newBkt = new elemEntry[n];
    for (int i=0; i<n; i++) {
      newBkt[i].elID = -1;
      newBkt[i].len = -1.0;
      newBkt[i].leftIdx = -1;
      newBkt[i].rightIdx = -1;
    }
    for (int i=0; i<bucketSz; i++) {
      newBkt[i].elID = elements[i].elID;
      newBkt[i].len = elements[i].len;
      newBkt[i].leftIdx = elements[i].leftIdx;
      newBkt[i].rightIdx = elements[i].rightIdx;
    }
    bucketSz = n;
    if (elements) delete[] elements;
    elements = newBkt;
  }
  void Clear() {
    if (elements) delete[] elements;
    bucketSz = numElements = 0;
    root = -1;
  }
  bool IsBucketEmpty() { return(numElements == 0); }
  /// Insert element to be refined/coarsened
  void Insert(int elID, double len) {
    CkAssert(numElements <= bucketSz);
    if (numElements == bucketSz) {
      Resize(bucketSz+100);
    }
    CkAssert(elID < bucketSz);
    if (root==-1) {
      root = 0;
      elements[0].elID = elID;
      elements[0].len = len;
      elements[0].leftIdx = elements[0].rightIdx = -1;
      numElements++;
    }
    else {
      int pos = findEmptySlot();
      elements[pos].elID = elID;
      elements[pos].len = len;
      elements[pos].leftIdx = elements[pos].rightIdx = -1;
      if ((len < elements[root].len) || ((len == elements[root].len) && (elID < elements[root].elID))) { 
	// go left
	if (elements[root].leftIdx == -1) {
	  elements[root].leftIdx = pos;
	}
	else {
	  Insert_help(elements[root].leftIdx, pos);
	}
	numElements++;
      }
      else if ((len > elements[root].len) || ((len == elements[root].len) && (elID >= elements[root].elID))) { 
	// go right
	if (elements[root].rightIdx == -1) {
	  elements[root].rightIdx = pos;
	}
	else {
	  Insert_help(elements[root].rightIdx, pos);
	}
	numElements++;
      }
      else { // invalid case
	CkAbort("ERROR: Element_Bucket::Insert: unreachable case!\n");
      }
    }
    sanity_check();
  }

  void Insert_help(int subtree, int pos) {
    if ((elements[pos].len < elements[subtree].len) || 
	((elements[pos].len == elements[subtree].len) && (elements[pos].elID < elements[subtree].elID))) { 
      // go left
      if (elements[subtree].leftIdx == -1) {
	elements[subtree].leftIdx = pos;
      }
      else {
	Insert_help(elements[subtree].leftIdx, pos);
      }
    }
    else if ((elements[pos].len > elements[subtree].len) || 
	     ((elements[pos].len == elements[subtree].len) && (elements[pos].elID >=elements[subtree].elID))) { 
      // go right
      if (elements[subtree].rightIdx == -1) {
	elements[subtree].rightIdx = pos;
      }
      else {
	Insert_help(elements[subtree].rightIdx, pos);
      }
    }
    else { // invalid case
      CkAbort("ERROR: Element_Bucket::Insert_help: unreachable case!\n");
    }
  }

  /// Get next element to be refined/coarsened
  int Remove(double *len) {
    // ASSERT: bucket is not empty; however, will return -1 in this case
    int elID, leftIdx, rightIdx;
    if (numElements == 0) {
      return -1;
    }
    int idx = root;
    int parent = root;
    while (elements[idx].leftIdx != -1) {
      if (parent != idx) parent = idx;
      idx = elements[idx].leftIdx;
    }
    elID = elements[idx].elID;
    CkAssert((elID < bucketSz) && (elID >=0));
    (*len) = elements[idx].len;
    if (elements[idx].rightIdx == -1) { // no children
      elements[idx].elID = -1;
      elements[idx].len = -1.0;
      elements[parent].leftIdx = -1;
    }
    else { // there is a right child
      elements[idx].elID = -1;
      elements[idx].len = -1.0;
      if (parent == idx) {
	root = elements[idx].rightIdx;
      }
      else {
	elements[parent].leftIdx = elements[idx].rightIdx;
      }
      elements[idx].rightIdx = -1;
    }
    numElements--;
    if (numElements==0) {
      root=-1;
    }
    CkPrintf("In Element_Bucket::Remove: the elId to be returned is: %d\n", elID);
    sanity_check();
    return elID;
  }

  int findEmptySlot() {
    for (int i=0; i<numElements+1; i++) {
      if (elements[i].elID == -1) {
	return i;
      }
    }
    CkAbort("ERROR: Element_Bucket::findEmptySlot: more than numElements solts appear to be full!\n");
  }

  void sanity_check() {
    int idx = root;
    int count = 0;
    CkAssert((root == -1) || ((root < bucketSz) && (root >=0)));
    if (root == -1) {
      CkAssert(numElements == 0);
    }
    else {
      sanity_check_helper(idx, &count);
      CkAssert(count == numElements);
    }
  }
  
  void sanity_check_helper(int idx, int *count) {
    (*count)++;
    CkAssert((elements[idx].elID < bucketSz) && (elements[idx].elID >= 0));
    CkAssert(elements[idx].len > 0.0);
    int leftIdx, rightIdx;
    leftIdx = elements[idx].leftIdx;
    CkAssert((leftIdx == -1) || ((leftIdx < bucketSz) && (leftIdx >=0)));
    rightIdx = elements[idx].rightIdx;
    CkAssert((rightIdx == -1) || ((rightIdx < bucketSz) && (rightIdx >=0)));
    if (leftIdx != -1) {
      sanity_check_helper(leftIdx, count);
    }
    if (rightIdx != -1) {
      sanity_check_helper(rightIdx, count);
    }
  }
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
