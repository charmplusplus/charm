/** Bulk Adapt Operations: An array class that shadows the mesh
    partitions and performs bulk adaptivity operations.  
    Created: 6 Dec 2006 by Terry L. Wilmarth */

#include "bulk_adapt_ops.h"

/// Construct array to be attached to the partitions of mesh mId
BulkAdapt::BulkAdapt(int meshid,FEM_Mesh *mPtr, int partID, 
		     CProxy_ParFUMShadowArray sa_proxy)
{
	meshID = meshid;
  meshPtr = mPtr;
  partitionID = partID;
  shadowProxy = sa_proxy;
  localShadow = meshPtr->parfumSA;
}

/// Destructor
BulkAdapt::~BulkAdapt()
{
}

/// Pack/Unpack this array element
void BulkAdapt::pup(PUP::er &p)
{
}

// MIGRATION NOTES:
// * will need to fix meshPtr when this partition migrates


/* BULK MESH OPERATIONS: These are all called locally, but may invoke
   remote operations. */

/// Perform an edge bisection (2D and 3D).
/** Locks mesh and any affected IDXL lists, performs operation,
    updates adapt adjacencies, and unlocks mesh & IDXL
    lists. Returns zero if the lock fails, positive if the operation
    suceeds, and negative if the operations fails for some other
    reason. */
int BulkAdapt::edge_bisect(int elemID, int elemType, int edgeID)
{
  CkPrintf("BulkAdapt::edge_bisect not yet implemented.\n");
  adaptAdj region[2];
  region[0].localID = elemID;
  region[0].partID = partitionID;
  region[0].elemType = elemType;
  region[1] = *GetAdaptAdj(meshID,elemID, elemType, edgeID);
  RegionID regionID;
  bool success;
  if (success = (localShadow->lockRegion(2, region, &regionID))) {
    CkPrintf("Lock obtained.\n");
    localShadow->unlockRegion(regionID);
  }
  else {
    CkPrintf("Lock not obtained.\n");
  }
  return success;
}

/// Perform an edge flip (2D)
/** Locks mesh and any affected IDXL lists, performs operation,
    updates adapt adjacencies, and unlocks mesh & IDXL
    lists. Returns zero if the lock fails, positive if the operation
    suceeds, and negative if the operations fails for some other
    reason. */
int BulkAdapt::edge_flip(int elemID, int edgeID)
{
  CkPrintf("BulkAdapt::edge_flip not yet implemented.\n");
  return 0;
}

/// Perform a Delaunay 2-3 flip (3D)
/** Locks mesh and any affected IDXL lists, performs operation,
    updates adapt adjacencies, and unlocks mesh & IDXL
    lists. Returns zero if the lock fails, positive if the operation
    suceeds, and negative if the operations fails for some other
    reason. */
int BulkAdapt::flip_23(int elemID, int faceID)
{
  CkPrintf("BulkAdapt::flip_23 not yet implemented.\n");
  return 0;
}

/// Perform a Delaunay 3-2 flip (3D)
/** Locks mesh and any affected IDXL lists, performs operation,
    updates adapt adjacencies, and unlocks mesh & IDXL
    lists. Returns zero if the lock fails, positive if the operation
    suceeds, and negative if the operations fails for some other
    reason. */
int BulkAdapt::flip_32(int elemID, int edgeID)
{
  CkPrintf("BulkAdapt::flip_32 not yet implemented.\n");
  return 0;
}

/// Perform an edge collapse (2D and 3D)
/** Locks mesh and any affected IDXL lists, performs operation,
    updates adapt adjacencies, and unlocks mesh & IDXL
    lists. Returns zero if the lock fails, positive if the operation
    suceeds, and negative if the operations fails for some other
    reason. */
int BulkAdapt::edge_collapse(int elemID, int edgeID)
{
  CkPrintf("BulkAdapt::edge_collapse not yet implemented.\n");
  return 0;
}

/* COMMUNICATION HELPERS FOR BULK ADAPTIVITY OPERATIONS 
   ARE LOCATED IN ParFUM_SA. */

/* LOCAL HELPERS FOR BULK ADAPTIVITY OPERATIONS */

/** Add a new element to the mesh. 
	 * Update its connectivity
	 * Return index of new element
	 * */
	int BulkAdapt::add_element(int elemType,int nodesPerElem,int *conn){};

	/** Update the conn of an element*/
	void BulkAdapt::update_element_conn(int elemType,int elemID,int nodesPerElem,int *conn){};



	/**
	 * Add a new node to the mesh
	 * update its co-ordinates 
	 * Return index of new node
	 * */
	int BulkAdapt::add_node(int dim,double *coords){};
	
	/** Update the co-ordimates of the given node
	*/
	void BulkAdapt::update_node_coord(int nodeID,int dim,double *coords){};

	void BulkAdapt::make_node_shared(int nodeID,int numSharedChunks,int *sharedChunks){};

