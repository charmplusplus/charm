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
  region[1] = *GetAdaptAdj(meshID, elemID, elemType, edgeID);
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

/*
int BulkAdapt::edge_bisect_2D(int elemID, int elemType, int edgeID)
{
  // lock partitions for the two involved elements
  
  // ******** LOCAL OPS *********
  // get elemID's conn.
  // let edgeID have node indices n1, n2.
  // let e2 (adaptAdj) be the element across edgeID.
  // get n1_coord and n2_coord. find midpoint: n5_coord.
  // add node at n5_coord to get node n5.
  // duplicate conn in conn2.  
  // in conn, replace n2 with n5.  in conn2, replace n1 with n5.
  // update elemID's with new conn.
  // add element e3 with conn2. copy elemID's adapt adj for all edges.
  // update adapt adj of elemID for edge (edgeID+1)%3 to local e3.
  // update adapt adj of e3 for edge (edgeID+2)%3 to local elemID.
  // interpolate nodal data, copy elemID data to e3

  // if e2 exists and is local...
  // find edgeID2, the edge with n1 and n2 on e2
  // get e2's conn as conn3. duplicate to conn4.
  // in conn3, replace n2 with n5. in conn4, replace n1 with n5.
  // update e2 with new conn3.
  // add element e4 with conn4.  copy e2's adapt adj for all edges.
  // update adapt adj of e2 for edge (edgeID2+2)%3 to local e4.
  // update adapt adj of e4 for edge (edgeID2+1)%3 to local e2.
  // update adapt adj of e4 for edge edgeID2 to local e3.
  // copy e2 data to e4

  // if e2 exists and is remote...
  // make n5 shared with e2's partition, get n5_idxl.
  // get n1_idxl and n2_idxl.
  // set up adaptAdj info for e3: e3_data.
  // SEND: e2, n5_idxl, n1_idxl, n2_idxl, e3_data, partID, to e2's partition

  // on remote side...
  // RECEIVE: e2, n5_idxl, n1_idxl, n2_idxl, e3_data, remote_partID
  // get e2's conn3.
  // get n1 and n2 from IDXL
  // find edgeID2 from n1, n2 and e2's conn3
  // get n1_coord and n2_coord. find midpoint: n5_coord.
  // add node at n5_coord to get node n5.
  // make n5 shared with partID at n5_idxl (n5_idxl is for error checking)
  // duplicate conn3 to conn4.
  // in conn3, replace n2 with n5. in conn4, replace n1 with n5.
  // update e2 with conn3. 
  // add element e4 with conn4.  copy e2's adapt adj for all edge.
  // update adapt adj of e2 for edge (edgeID2+2)%3 to local e4.
  // update adapt adj of e4 for edge (edgeID2+1)%3 to local e2.
  // update adapt adj of e4 for edge edgeID2 to e3_data.
  // interpolate nodal data, copy e2 data to e4
  // set up adaptAdj info for e4: e4_data.
  // SEND: e4_data, partID to remote_partID

  // locally...
  // RECEIVE: e4_data, remote_partID
  // update adapt adj of e3 for edge edgeID to e4_data

  // unlock the two partitions
}

int BulkAdapt::edge_bisect_3D(int elemID, int elemType, int edgeID)
{
  // ASSERT: An edge can only be on one surface.
  // get elemID's conn.
  // let edgeID have node indices n1, n2.
  // Union the set of partitions that share n1 and n2.
  // Lock the partition set.

  // ******** LOCAL OPS *********
  // let the other nodes of elemID be n3 and n4
  // let elemNext be the adaptAdj on face of (n1, n2, n3)
  // let nPrev = n3
  // if there is no neighbor there, let elemNext be across face (n1, n2, n4),
  //   and let nPrev = n4
  // get n1_coord and n2_coord. find midpoint: n5_coord.
  // add node at n5_coord to get node n5.
  // duplicate conn in conn2.  
  // in conn, replace n2 with n5.  in conn2, replace n1 with n5.
  // update elemID's with new conn.
  // add element e3 with conn2. copy elemID's adapt adj for all faces.
  // update adapt adj of elemID for face (n5,n3,n4) (any order) to local e3.
  // update adapt adj of e3 for face (n5,n3,n4) (any order) to local elemID.
  // interpolate nodal data, copy elemID data to e3

  // if elemNext is local...
  // let nNext be the node that is not n1, n2 or nPrev
  // find edgeID2, the edge with n1 and n2 on elemNext
  // get elemNext's conn as conn3. duplicate to conn4.
  // in conn3, replace n2 with n5. in conn4, replace n1 with n5.
  // update elemNext with new conn3.
  // add element elemNextSplit with conn4.  copy elemNext's adaptadj
  // update adapt adj of elemNext for face (n5,n3,n4) to local elemNextSplit.
  // update adapt adj of elemNextSplit for face (n5,n3,n4) to local elemNext.
  // update adapt adj of elemNext for face (n1,n5,nPrev) to local elemID.
  // update adapt adj of elemNextSplit for face (n2,n5,nPrev) to local e3.
  // copy elemNext data to elemNextSplit
  // update adaptAdj of e3 for face (n2,n5,nPrev) to local elemNextSplit
  // find element across face n1, n2, nNext and recurse until elemID
  //   is reached or boundary is reached.

  // if elemNext is remote...

  // Unlock the partition set.
}
*/

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
	int BulkAdapt::add_element(int elemType,int nodesPerElem,int *conn){ 
		FEM_Elem &elem = meshPtr->elem[elemType];
		int newElem = elem.get_next_invalid();
		elem.connIs(newElem,conn);
		return newElem;
	}

	/** Update the conn of an element*/
	void BulkAdapt::update_element_conn(int elemType,int elemID,int nodesPerElem,int *conn){
		FEM_Elem &elem = meshPtr->elem[elemType];
		elem.connIs(elemID,conn);
	};



	/**
	 * Add a new node to the mesh
	 * update its co-ordinates 
	 * Return index of new node
	 * */
	int BulkAdapt::add_node(int dim,double *coords){ 
		int newNode = meshPtr->node.get_next_invalid();
		FEM_DataAttribute *coord = meshPtr->node.getCoord();
		(coord->getDouble()).setRow(newNode,coords);
		return newNode;
	}
	
	/** Update the co-ordimates of the given node
	*/
	void BulkAdapt::update_node_coord(int nodeID,int dim,double *coords){
		FEM_DataAttribute *coord = meshPtr->node.getCoord();
		(coord->getDouble()).setRow(nodeID,coords);
	};

	void BulkAdapt::make_node_shared(int nodeID,int numSharedChunks,int *sharedChunks){
	};

