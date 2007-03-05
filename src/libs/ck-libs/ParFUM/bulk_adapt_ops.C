/** Bulk Adapt Operations: An array class that shadows the mesh
    partitions and performs bulk adaptivity operations.  
    Created: 6 Dec 2006 by Terry L. Wilmarth */

#include "bulk_adapt_ops.h"

#define BULK_DEBUG(x) x

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
int BulkAdapt::edge_bisect(int elemID, int elemType, int edgeID, int dim)
{
  if (dim == 2) {
    return edge_bisect_2D(elemID, elemType, edgeID);
  }
  else if (dim == 3) {
    return edge_bisect_3D(elemID, elemType, edgeID);
  }
}


int BulkAdapt::edge_bisect_2D(int elemID, int elemType, int edgeID)
{
  CkPrintf("BulkAdapt::edge_bisect_2D not yet implemented.\n");
  // lock partitions for the two involved elements
  adaptAdj elemsToLock[2];
  adaptAdj startElem(partitionID, elemID, elemType);
  adaptAdj nbrElem = *GetAdaptAdj(meshID, elemID, elemType, edgeID);
  elemsToLock[0] = startElem;
  elemsToLock[1] = nbrElem;
  RegionID lockRegionID;
  bool success;
  if (success = (localShadow->lockRegion(2, elemsToLock, &lockRegionID))) {
    BULK_DEBUG(CkPrintf("Lock obtained.\n"););
  }
  else {
    BULK_DEBUG(CkPrintf("Lock not obtained.\n"););
    return success;
  }

  // ******** LOCAL OPS *********

  adaptAdj *startElemAdaptAdj;
  adaptAdj *splitElemAdaptAdj;
  int node1idx, node2idx;
  adaptAdj splitElem;

  one_side_split_2D(startElem, splitElem, edgeID, &node1idx, &node2idx, &startElemAdaptAdj, &splitElemAdaptAdj, true);

  if ((nbrElem.partID > -1) && (nbrElem.partID == partitionID)) { // if e2 exists and is local...
    // PRE: neighbor-side operations
    FEM_Elem &elem = meshPtr->elem[elemType]; // elem is local elements
    int *nbrConn = elem.connFor(nbrElem.localID);
    int relNode1 = getRelNode(node1idx, nbrConn, 3);
    int relNode2 = getRelNode(node2idx, nbrConn, 3);
    int nbrEdgeID = getEdgeID(relNode1, relNode2, 3, 2);

    int nbrNode1, nbrNode2;
    adaptAdj *nbrElemAdaptAdj, *nbrSplitElemAdaptAdj;
    adaptAdj nbrSplitElem;
    
    // nugget :)
    one_side_split_2D(nbrElem, nbrSplitElem, nbrEdgeID, &nbrNode1, &nbrNode2, 
		      &nbrElemAdaptAdj, &nbrSplitElemAdaptAdj, false);

    nbrSplitElemAdaptAdj[nbrEdgeID] = splitElem;
    // POST: start-side operations
    splitElemAdaptAdj[edgeID] = nbrSplitElem;
  }

  /*
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
  */

  // unlock the two partitions
  localShadow->unlockRegion(lockRegionID);
  return 1;
}

int BulkAdapt::edge_bisect_3D(int elemID, int elemType, int edgeID)
{
  CkPrintf("BulkAdapt::edge_bisect_3D not yet implemented.\n");
  /*
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
*/
  return 0;
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

void BulkAdapt::one_side_split_2D(adaptAdj &startElem, adaptAdj &splitElem, int edgeID,
				  int *node1idx, int *node2idx,
				  adaptAdj **startElemAdaptAdj, adaptAdj **splitElemAdaptAdj,
				  bool startSide)
{
  // get startElem.localID's conn.
  FEM_Elem &elem = meshPtr->elem[startElem.elemType]; // elem is local elements
  int *startConn = elem.connFor(startElem.localID); // startConn points at startElem.localID's ACTUAL data!
  // let edgeID have element-relative node indices relNode1, relNode2.
  int relNode1 = edgeID, relNode2 = (edgeID+1)%3;
  *node1idx = startConn[relNode1];
  *node2idx = startConn[relNode2];
  // get node1coords and node2Coords. find midpoint: bisectCoords
  FEM_DataAttribute *coord = meshPtr->node.getCoord(); // entire local coords
  double *node1coords = (coord->getDouble()).getRow(*node1idx); // ptrs to ACTUAL coords!
  double *node2coords = (coord->getDouble()).getRow(*node2idx); // ptrs to ACTUAL coords!
  double bisectCoords[2];
  midpoint(node1coords, node2coords, 2, &bisectCoords[0]);
  // add node at bisectCoords to get bisectNode
  int bisectNode = add_node(2, &bisectCoords[0]);
  // duplicate conn in conn2.  
  int splitConn[3];
  memcpy(&splitConn[0], startConn, 3*sizeof(int));
  // in startConn, replace node2idx with bisectNode
  startConn[relNode2] = bisectNode; // ACTUAL data was changed!
  // in splitConn, replace node1idx with bisectNode
  splitConn[relNode1] = bisectNode;
  // add element split with splitConn
  int splitElemID = add_element(startElem.elemType, 3, &splitConn[0]);
  // copy startElem.localID's adapt adj for all edges.
  *startElemAdaptAdj = GetAdaptAdj(meshID, startElem.localID, startElem.elemType, 0);
  *splitElemAdaptAdj = GetAdaptAdj(meshID, splitElemID, startElem.elemType, 0);
  splitElem = adaptAdj(partitionID, splitElemID, startElem.elemType);
  memcpy(*splitElemAdaptAdj, *startElemAdaptAdj, 3*sizeof(adaptAdj));
  if (startSide) {
    // update startElemAdaptAdj for edge (edgeID+1)%3 to local splitElem
    (*startElemAdaptAdj)[(edgeID+1)%3] = splitElem;
    // update splitElemAdaptAdj for edge (edgeID+2)%3 to local startElem
    (*splitElemAdaptAdj)[(edgeID+2)%3] = startElem;
  }
  else {
    // update startElemAdaptAdj for edge (edgeID+1)%3 to local splitElem
    (*startElemAdaptAdj)[(edgeID+2)%3] = splitElem;
    // update splitElemAdaptAdj for edge (edgeID+2)%3 to local startElem
    (*splitElemAdaptAdj)[(edgeID+1)%3] = startElem;
  }
  // interpolate nodal data, copy startElem data to splitElem
  CkPrintf("WARNING: Data transfer not yet implemented.\n");
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
		for(int i=0;i<numSharedChunks;i++){
			IDXL_List &sharedList = meshPtr->node.shared.addList(sharedChunks[i]);
			sharedList.push_back(nodeID);
		}
		meshPtr->node.shared.flushMap();
	};





void midpoint(double *n1, double *n2, int dim, double *result) {
	for(int i=0;i<dim;i++){
		result[i] = (n1[i]+n2[i])/2.0;
	}
}

int getRelNode(int nodeIdx, int *conn, int nodesPerElem) {
	for(int i=0;i<nodesPerElem;i++){
		if(conn[i] == nodeIdx){
			return i;
		}
	}
	CkAbort("Could not find node in given connectivity");
}

int getEdgeID(int node1, int node2, int nodesPerElem, int dim) {
	switch(dim){
		case 2:
		switch(nodesPerElem){
			case 3:
				{
					static int edgeFromNode[3][3]={-1,0,2,0,-1,1,2,1,-1};
					return edgeFromNode[node1][node2];
				}
				break;
			default:
				CkAbort("This shape is not yet implemented");
		};
		break;
		default:
		 CkAbort("This dimension not yet implemented");
	};
  return 0;
}
