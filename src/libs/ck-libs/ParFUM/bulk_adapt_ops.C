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

/// Perform a 2D edge bisection on a triangle
/*
               o                                     o	 
              /|\		                    /|\	 
 startElem   / | \   nbrElem	       startElem   / | \   nbrElem	 
            /  |  \		                  /  |  \	 
           /   |   \		                 /   |   \	 
          /    |    \		                /    |    \	 
         o     |     o		               o-----o-----o	 
          \    |    /		                \    |    /	 
           \   |   /		                 \   |   /	 
            \  |  /		       splitElem  \  |  /  nbrSplitElem
             \ | /		                   \ | /	 
              \|/		                    \|/	 
               o         	                     o         
 */
int BulkAdapt::edge_bisect_2D(int elemID, int elemType, int edgeID)
{
  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::edge_bisect_2D starts elemID %d \n",partitionID,elemID));
  // lock partitions for the two involved elements
  adaptAdj elemsToLock[2];
  adaptAdj startElem(partitionID, elemID, elemType);
  adaptAdj nbrElem = *GetAdaptAdj(meshID, elemID, elemType, edgeID);
  BULK_DEBUG(printf("[%d] neighbor of elem %d is elem (%d,%d) \n",partitionID,elemID,nbrElem.partID,nbrElem.localID);)
  elemsToLock[0] = startElem;
  elemsToLock[1] = nbrElem;
  RegionID lockRegionID;
  bool success;
  if (success = (localShadow->lockRegion(2, elemsToLock, &lockRegionID))) {
    BULK_DEBUG(CkPrintf("[%d] Lock obtained.\n",partitionID););
  }
  else {
    BULK_DEBUG(CkPrintf("[%d] Lock not obtained.\n",partitionID););
    return success;
  }

  // ******** LOCAL OPS *********
  int node1idx, node2idx, newNodeID;
  adaptAdj splitElem;
  // split the local element, i.e. the first "side"
  one_side_split_2D(startElem, splitElem, edgeID, &node1idx, &node2idx, &newNodeID, true);

  if ((nbrElem.partID > -1) && (nbrElem.partID == partitionID)) { // if nbrElem exists and is local...
    // PRE: neighbor-side operations
    FEM_Elem &elem = meshPtr->elem[elemType]; // elem is local elements
    int *nbrConn = elem.connFor(nbrElem.localID);
    int relNode1 = getRelNode(node1idx, nbrConn, 3);
    int relNode2 = getRelNode(node2idx, nbrConn, 3);
    int nbrEdgeID = getEdgeID(relNode1, relNode2, 3, 2);

    int nbrNode1, nbrNode2;
    adaptAdj nbrSplitElem;
    // split the local neighbor element, i.e. the second "side"
    one_side_split_2D(nbrElem, nbrSplitElem, nbrEdgeID, &nbrNode1, &nbrNode2, &newNodeID, false);

    // now fix the adjacencies across the new edge to the two new elements
    adaptAdj *splitElemAdaptAdj = GetAdaptAdj(meshID, splitElem.localID, splitElem.elemType, 0);
    adaptAdj *nbrSplitElemAdaptAdj = GetAdaptAdj(meshID, nbrSplitElem.localID, nbrSplitElem.elemType, 0);
    nbrSplitElemAdaptAdj[nbrEdgeID] = splitElem;
    BULK_DEBUG(printf("[%d] For nbrSplitElem %d set adjacency to %d across splitEdge\n",partitionID,nbrSplitElem.localID,splitElem.localID);)
    // POST: start-side operations
    splitElemAdaptAdj[edgeID] = nbrSplitElem;
    BULK_DEBUG(printf("[%d] For splitElem %d set adjacency to %d across splitEdge\n",partitionID,splitElem.localID,nbrSplitElem.localID);)
  }
  else if (nbrElem.partID == -1) { // startElem's edgeID is on domain boundary
    adaptAdj *splitElemAdaptAdj = GetAdaptAdj(meshID, splitElem.localID, splitElem.elemType, 0);
    splitElemAdaptAdj[edgeID] = adaptAdj(-1, -1, -1);
    BULK_DEBUG(printf("[%d] For splitElem %d splitEdge is on the domain boundary.\n",partitionID,splitElem.localID);)
  }
  else { // nbrElem exists and is remote
    int chunks[1] = {nbrElem.partID};
    make_node_shared(newNodeID, 1, &chunks[0]);
    int new_idxl, n1_idxl, n2_idxl;
    new_idxl = get_idxl_for_node(newNodeID, nbrElem.partID);
    n1_idxl = get_idxl_for_node(node1idx, nbrElem.partID);
    n2_idxl = get_idxl_for_node(node2idx, nbrElem.partID);

    // make sync call on partition nbrElem.partID
    // SEND: nbrElem, splitElem, new_idxl, n1_idxl, n2_idxl, partitionID
    // RETURNS: nbrSplitElem
    adaptAdjMsg *am = shadowProxy[nbrElem.partID].remote_bulk_edge_bisect_2D(nbrElem, splitElem, new_idxl, n1_idxl, n2_idxl, partitionID);
    
    adaptAdj nbrSplitElem = am->elem;
    // now fix the adjacencies across the new edge to remote new element
    adaptAdj *splitElemAdaptAdj = GetAdaptAdj(meshID, splitElem.localID, splitElem.elemType, 0);
    splitElemAdaptAdj[edgeID] = nbrSplitElem;
    BULK_DEBUG(printf("[%d] For splitElem %d set adjacency to %d across splitEdge\n",partitionID,splitElem.localID,nbrSplitElem.localID);)
  }    

  // unlock the two partitions
  localShadow->unlockRegion(lockRegionID);
  dumpAdaptAdjacencies(GetAdaptAdj(meshID, 0, elemType, 0),meshPtr->nElems(),3,partitionID);	
  return 1;
}

adaptAdj BulkAdapt::remote_edge_bisect_2D(adaptAdj nbrElem, adaptAdj splitElem, int new_idxl, int n1_idxl, int n2_idxl, int remotePartID)
{
  int node1idx, node2idx, newNodeID;
  node1idx = get_node_from_idxl(n1_idxl, remotePartID);
  node2idx = get_node_from_idxl(n2_idxl, remotePartID);
  
  FEM_Elem &elem = meshPtr->elem[nbrElem.elemType]; // elem is local elements
  int *nbrConn = elem.connFor(nbrElem.localID);
  int relNode1 = getRelNode(node1idx, nbrConn, 3);
  int relNode2 = getRelNode(node2idx, nbrConn, 3);
  int nbrEdgeID = getEdgeID(relNode1, relNode2, 3, 2);

  FEM_DataAttribute *coord = meshPtr->node.getCoord(); // entire local coords
  double *node1coords = (coord->getDouble()).getRow(node1idx); // ptrs to ACTUAL coords!
  double *node2coords = (coord->getDouble()).getRow(node2idx); // ptrs to ACTUAL coords!
  double bisectCoords[2];
  midpoint(node1coords, node2coords, 2, &bisectCoords[0]);
  newNodeID = add_node(2, &bisectCoords[0]);
  int chunks[2] = {nbrElem.partID, remotePartID};
  make_node_shared(newNodeID, 2, chunks);
  int local_new_idxl = get_idxl_for_node(newNodeID, remotePartID);
  if (new_idxl != local_new_idxl)
    printf("ERROR: Partition %d added shared node at different idxl index %d than other copy at %d on partition %d!", 
	   nbrElem.partID, local_new_idxl, new_idxl, remotePartID);

  int nbrNode1, nbrNode2;
  adaptAdj nbrSplitElem;
  // split the local neighbor element, i.e. the second "side"
  one_side_split_2D(nbrElem, nbrSplitElem, nbrEdgeID, &nbrNode1, &nbrNode2, &newNodeID, false);

  adaptAdj *nbrSplitElemAdaptAdj = GetAdaptAdj(meshPtr, nbrSplitElem.localID, nbrSplitElem.elemType, 0);
  nbrSplitElemAdaptAdj[nbrEdgeID] = splitElem;
  BULK_DEBUG(printf("[%d] For nbrSplitElem %d set adjacency to %d across splitEdge\n",partitionID,nbrSplitElem.localID,splitElem.localID);)
  
  return nbrSplitElem;
}

/// Perform a 3D edge bisection on a tetrahedron
int BulkAdapt::edge_bisect_3D(int elemID, int elemType, int edgeID)
{ // ASSERT: An edge can only be on one surface.
  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::edge_bisect_3D starts at elemID %d \n",partitionID,elemID));
  int numElemsToLock = 0;
  adaptAdj *elemsToLock;
  adaptAdj startElem(partitionID, elemID, elemType);
  get_elemsToLock(startElem, elemsToLock, &numElemsToLock);
  RegionID lockRegionID;
  bool success;
  if (success = (localShadow->lockRegion(numElemsToLock, elemsToLock, &lockRegionID))) {
    BULK_DEBUG(CkPrintf("[%d] Lock obtained.\n",partitionID););
  }
  else {
    BULK_DEBUG(CkPrintf("[%d] Lock not obtained.\n",partitionID););
    return success;
  }

  // ******** LOCAL OPS *********
  int nodeIDs[4], newNodeID;
  adaptAdj splitElem;
  // split the local element, i.e. the first "side"
  one_side_split_3D(startElem, splitElem, startElem, startElem, edgeID, &(nodeIDs[0]), &(nodeIDs[1]), &newNodeID, true);

  int relNode[4];
  FEM_Elem &elem = meshPtr->elem[elemType]; // elem is local elements
  int *startConn = elem.connFor(startElem.localID);
  relNode[0] = getRelNode(nodeIDs[0], startConn, 4);
  relNode[1] = getRelNode(nodeIDs[1], startConn, 4);
  fillNodes(&(relNode[0]), &(nodeIDs[0]), startConn);
  int face[2]; // faces on which the two potentially splittable neighbors lie
  face[0] = 3 - (relNode[0] + relNode[1] + relNode[2]);
  face[1] = 3 - (relNode[0] + relNode[1] + relNode[3]);
  adaptAdj neighbors[2]; // startElem's neighbors
  neighbors[0] = *GetAdaptAdj(meshID, elemID, elemType, face[0]);
  neighbors[1] = *GetAdaptAdj(meshID, elemID, elemType, face[1]);

  /*
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
*/

  // unlock the partitions
  localShadow->unlockRegion(lockRegionID);
  return 1;
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

void BulkAdapt::one_side_split_2D(adaptAdj &startElem, adaptAdj &splitElem, 
				  int edgeID, int *node1idx, int *node2idx, 
				  int *newNodeID, bool startSide)
{
  // get startElem.localID's conn.
  FEM_Elem &elem = meshPtr->elem[startElem.elemType]; // elem is local elements
  int *startConn = elem.connFor(startElem.localID); // startConn points at startElem.localID's ACTUAL data!
  // let edgeID have element-relative node indices relNode1, relNode2.
  int relNode1 = edgeID, relNode2 = (edgeID+1)%3;
  if (!startSide) {
    relNode1 = (edgeID+1)%3;
    relNode2 = edgeID;
  }
  *node1idx = startConn[relNode1];
  *node2idx = startConn[relNode2];


  BULK_DEBUG(printf("[%d] one_side_split_2D called for elem %d edge %d nodes %d %d \n",partitionID,startElem.localID,edgeID,*node1idx,*node2idx);)
  // get node1coords and node2Coords. find midpoint: bisectCoords
  FEM_DataAttribute *coord = meshPtr->node.getCoord(); // entire local coords
  double *node1coords = (coord->getDouble()).getRow(*node1idx); // ptrs to ACTUAL coords!
  double *node2coords = (coord->getDouble()).getRow(*node2idx); // ptrs to ACTUAL coords!
  double bisectCoords[2];
  midpoint(node1coords, node2coords, 2, &bisectCoords[0]);
  // add node at bisectCoords to get bisectNode
  int bisectNode;
  if (startSide) { // For local ops, startSide creates the node, nbr uses it
    bisectNode = add_node(2, &bisectCoords[0]);
    *newNodeID = bisectNode;
  }
  else { // local nbr just uses node created on startSide
    // if neighbor is remote, the newNodeID is set in remote_edge_bisect_2D before calling this function
    bisectNode = *newNodeID;
  }
  // duplicate conn in conn2.  
  int splitConn[3];
  memcpy(&splitConn[0], startConn, 3*sizeof(int));
  // in startConn, replace node2idx with bisectNode
  startConn[relNode2] = bisectNode; // ACTUAL data was changed!
  BULK_DEBUG(printf("[%d] conn[%d] of elem %d changed to node %d\n",partitionID,relNode2,startElem.localID,bisectNode);)
  // in splitConn, replace node1idx with bisectNode
  splitConn[relNode1] = bisectNode;
  // add element split with splitConn
  int splitElemID = add_element(startElem.elemType, 3, &splitConn[0]);
  BULK_DEBUG(printf("[%d] new element %d with conn %d %d %d added \n", partitionID, splitElemID, splitConn[0], splitConn[1], splitConn[2]);)
  // copy startElem.localID's adapt adj for all edges.
  adaptAdj *startElemAdaptAdj = GetAdaptAdj(meshPtr, startElem.localID, 
					    startElem.elemType, 0);
  adaptAdj *splitElemAdaptAdj = GetAdaptAdj(meshPtr, splitElemID, 
					    startElem.elemType, 0);
  splitElem = adaptAdj(partitionID, splitElemID, startElem.elemType);
  memcpy(splitElemAdaptAdj, startElemAdaptAdj, 3*sizeof(adaptAdj));
  adaptAdj startElemNbr;  // startElem's original nbr on the edge that will now border with splitElem
  if (startSide) {
    // update startElemAdaptAdj for edge (edgeID+1)%3 to local splitElem
    startElemNbr = startElemAdaptAdj[(edgeID+1)%3];
    startElemAdaptAdj[(edgeID+1)%3] = splitElem;
    // update splitElemAdaptAdj for edge (edgeID+2)%3 to local startElem
    splitElemAdaptAdj[(edgeID+2)%3] = startElem;
    BULK_DEBUG(printf("[%d] For startElem %d edge %d is now set to %d\n",partitionID,startElem.localID,(edgeID+1)%3,splitElem.localID));
    BULK_DEBUG(printf("[%d] For splitElem %d edge %d is now set to %d\n",partitionID,splitElem.localID,(edgeID+2)%3,startElem.localID));
  }
  else {
    // update startElemAdaptAdj for edge (edgeID+1)%3 to local splitElem
    startElemNbr = startElemAdaptAdj[(edgeID+2)%3];
    startElemAdaptAdj[(edgeID+2)%3] = splitElem;
    // update splitElemAdaptAdj for edge (edgeID+2)%3 to local startElem
    splitElemAdaptAdj[(edgeID+1)%3] = startElem;
    BULK_DEBUG(printf("[%d] For startElem %d edge %d is now set to %d\n",partitionID,startElem.localID,(edgeID+2)%3,splitElem.localID));
    BULK_DEBUG(printf("[%d] For splitElem %d edge %d is now set to %d\n",partitionID,splitElem.localID,(edgeID+1)%3,startElem.localID));
  }
  if (startElemNbr.partID == startElem.partID) {
    ReplaceAdaptAdj(meshPtr, startElemNbr.localID, startElemNbr.elemType, startElem, splitElem);
    BULK_DEBUG(printf("[%d] For startElemNbr %d replaced startElem %d with splitElem %d\n",partitionID,startElemNbr.localID,startElem.localID,splitElem.localID);)
  }
  else if (startElemNbr.partID != -1) { // startElemNbr exists and is remote
    // need to call ReplaceAdaptAdj on startElemNbr.partID
    shadowProxy[startElemNbr.partID].remote_adaptAdj_replace(startElemNbr, startElem, splitElem); 
  }
  // interpolate nodal data, copy startElem data to splitElem
  CkPrintf("WARNING: Data transfer not yet implemented.\n");
}


void BulkAdapt::one_side_split_3D(adaptAdj &startElem, adaptAdj &splitElem, 
				  adaptAdj &firstElem, adaptAdj &fromElem,
				  int edgeID, int *node1idx, int *node2idx, 
				  int *newNodeID, bool startSide)
{
  CkPrintf("ERROR: one_side_split_3D not yet implemented!\n");
}




/* COMMUNICATION HELPERS FOR BULK ADAPTIVITY OPERATIONS 
   ARE LOCATED IN ParFUM_SA. */

void BulkAdapt::remote_adaptAdj_replace(adaptAdj elem, adaptAdj oldElem, adaptAdj newElem)
{
  ReplaceAdaptAdj(meshPtr, elem.localID, elem.elemType, oldElem, newElem);
}

/* LOCAL HELPERS FOR BULK ADAPTIVITY OPERATIONS */

/** Add a new element to the mesh. 
 * Update its connectivity
 * Return index of new element
 * */
int BulkAdapt::add_element(int elemType,int nodesPerElem,int *conn){ 
  //since the element array might be resized we need to set the correct thread
  //context before we call get_next_invalid
  localShadow->setRunningTCharm();
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
  //since the node array might be resized we need to set the correct thread
  //context before we call get_next_invalid
  localShadow->setRunningTCharm();
  int newNode = meshPtr->node.get_next_invalid();
  FEM_DataAttribute *coord = meshPtr->node.getCoord();
  (coord->getDouble()).setRow(newNode,coords);
  AllocTable2d<int> &lockTable = ((FEM_DataAttribute *)(meshPtr->node.lookup(FEM_ADAPT_LOCK,"lockRegion")))->getInt();
  
  lockTable[newNode][0] = -1;
  lockTable[newNode][1] = -1;
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

int BulkAdapt::get_idxl_for_node(int nodeID, int partID) 
{
  IDXL_List *list = meshPtr->node.shared.getIdxlListN(partID);
  CkAssert(list!=NULL);
  for (int i=0; i<list->size(); i++) {
    if ((*list)[i] == nodeID) {
      return i;
    }
  }
  CkAssert(0);
}

int BulkAdapt::get_node_from_idxl(int node_idxl, int partID)
{
  IDXL_List *list = meshPtr->node.shared.getIdxlListN(partID);
  CkAssert(list!=NULL);
  CkAssert(list->size()>node_idxl);
  return (*list)[node_idxl];
}


void BulkAdapt::get_elemsToLock(adaptAdj startElem, adaptAdj *elemsToLock, int *count)
{
  CkPrintf("ERROR: get_elemsToLock not yet implemented!\n");
}



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

void fillNodes(int *relNode, int *nodeIDs, int *conn)
{ // ASSERT: this is only for tets.  Assumes positions 0 and 1 are filled.
  if ((relNode[0] != 0) && (relNode[1] != 0))
    relNode[2] = 0;
  else if ((relNode[0] != 1) && (relNode[1] != 1))
    relNode[2] = 1;
  else relNode[2] = 2;
  relNode[3] = 6 - relNode[0] - relNode[1] - relNode[2];
  nodeIDs[2] = conn[relNode[2]];
  nodeIDs[3] = conn[relNode[3]];
}
