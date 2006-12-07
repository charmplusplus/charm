/** Bulk Adapt Operations: An array class that shadows the mesh
    partitions and performs bulk adaptivity operations.  
    Created: 6 Dec 2006 by Terry L. Wilmarth */
#ifndef __BULK_ADAPT_OPS_H
#define __BULK_ADAPT_OPS_H

#include "charm++.h"
#include "ParFUM.h"
#include "ParFUM_internals.h"
#include "ParFUM_SA.h"

/// This shadow array is attached to a partition to perform bulk adaptivity
/** This is a shadow array for performing all bulk adaptivity operations, 
    including cross-partition operations. */
class BulkAdapt {
 private:
  /// The index of the partition this array element is attached to
  int partitionID;
	int meshID;
  /// Pointer to the local mesh partition associated with this array
  FEM_Mesh *meshPtr;
  /// Proxy to ParFUM shadow array attached to each mesh partition
  CProxy_ParFUMShadowArray shadowProxy;
  /// Local element of shadow array
  ParFUMShadowArray *localShadow;

 public:
  /// Construct array to be attached to the partitions of mesh mId
  BulkAdapt(int meshid, FEM_Mesh *mPtr, int partID, CProxy_ParFUMShadowArray sa_proxy);
  /// Destructor
  ~BulkAdapt();

  /// Pack/Unpack this array element
  void pup(PUP::er &p);

  /* BASIC DATA ACCESS FUNCTIONS */
  
  /// Return this partition's ID
  int getPartition() { return partitionID; }
  /// Return a pointer to the local mesh partition
  FEM_Mesh *getMeshPtr() { return meshPtr; }

  /* BULK MESH OPERATIONS: These are all called locally, but may invoke
     remote operations. */

  /// Perform an edge bisection (2D and 3D).
  /** Locks mesh and any affected IDXL lists, performs operation,
      updates adapt adjacencies, and unlocks mesh & IDXL
      lists. Returns zero if the lock fails, positive if the operation
      suceeds, and negative if the operations fails for some other
      reason. */
  int edge_bisect(int elemID, int elemType, int edgeID);

  /// Perform an edge flip (2D)
  /** Locks mesh and any affected IDXL lists, performs operation,
      updates adapt adjacencies, and unlocks mesh & IDXL
      lists. Returns zero if the lock fails, positive if the operation
      suceeds, and negative if the operations fails for some other
      reason. */
  int edge_flip(int elemID, int edgeID);

  /// Perform a Delaunay 2-3 flip (3D)
  /** Locks mesh and any affected IDXL lists, performs operation,
      updates adapt adjacencies, and unlocks mesh & IDXL
      lists. Returns zero if the lock fails, positive if the operation
      suceeds, and negative if the operations fails for some other
      reason. */
  int flip_23(int elemID, int faceID);

  /// Perform a Delaunay 3-2 flip (3D)
  /** Locks mesh and any affected IDXL lists, performs operation,
      updates adapt adjacencies, and unlocks mesh & IDXL
      lists. Returns zero if the lock fails, positive if the operation
      suceeds, and negative if the operations fails for some other
      reason. */
  int flip_32(int elemID, int edgeID);

  /// Perform an edge collapse (2D and 3D)
  /** Locks mesh and any affected IDXL lists, performs operation,
      updates adapt adjacencies, and unlocks mesh & IDXL
      lists. Returns zero if the lock fails, positive if the operation
      suceeds, and negative if the operations fails for some other
      reason. */
  int edge_collapse(int elemID, int edgeID);

  /* COMMUNICATION HELPERS FOR BULK ADAPTIVITY OPERATIONS */
  
  /* LOCAL HELPERS FOR BULK ADAPTIVITY OPERATIONS */
};
#endif
