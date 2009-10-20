/** Bulk Adapt Operations: An array class that shadows the mesh
    partitions and performs bulk adaptivity operations.  
    Created: 6 Dec 2006 by Terry L. Wilmarth */

#include "bulk_adapt_ops.h"
#include "import.h"

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
  for (int i=0; i<10; i++) { freeTable[i] = 1; }
  firstFree = 0;
}

/// Destructor
BulkAdapt::~BulkAdapt()
{
}

/// Pack/Unpack this array element
void BulkAdapt::pup(PUP::er &p)
{
// MIGRATION NOTES:
// * will need to fix meshPtr when this partition migrates
// * will need to fix localShadow too.  shadowProxy? should be fine?
}

/* BULK MESH OPERATIONS: These are all called locally, but may invoke
   remote operations. */

/// Perform an edge bisection (2D and 3D).
/** Locks mesh and any affected IDXL lists, performs operation,
    updates adapt adjacencies, and unlocks mesh & IDXL
    lists. Returns zero if the lock fails, positive if the operation
    suceeds, and negative if the operations fails for some other
    reason. */
bool BulkAdapt::edge_bisect(int elemID, int elemType, int edgeID, int dim, RegionID lockRegionID)
{
  if (dim == 2) {
    return edge_bisect_2D(elemID, elemType, edgeID);
  }
  else if (dim == 3) {
    return edge_bisect_3D(elemID, elemType, edgeID, lockRegionID);
  }
  else return false;
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
bool BulkAdapt::edge_bisect_2D(int elemID, int elemType, int edgeID)
{
  //BULK_DEBUG(CkPrintf("[%d] BulkAdapt::edge_bisect_2D starts elemID %d \n",partitionID,elemID));
  // lock partitions for the two involved elements
  adaptAdj elemsToLock[2];
  adaptAdj startElem(partitionID, elemID, elemType);
  adaptAdj nbrElem = *getAdaptAdj(meshID, elemID, elemType, edgeID);
  //BULK_DEBUG(printf("[%d] neighbor of elem %d is elem (%d,%d) \n",partitionID,elemID,nbrElem.partID,nbrElem.localID);)
  elemsToLock[0] = startElem;
  elemsToLock[1] = nbrElem;
  RegionID lockRegionID;
  if (localShadow->lockRegion(2, elemsToLock, &lockRegionID,0.0)) {
    //BULK_DEBUG(CkPrintf("[%d] Lock obtained.\n",partitionID););
  }
  else {
    //BULK_DEBUG(CkPrintf("[%d] Lock not obtained.\n",partitionID););
    return false;
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
    adaptAdj *splitElemAdaptAdj = getAdaptAdj(meshID, splitElem.localID, splitElem.elemType, 0);
    adaptAdj *nbrSplitElemAdaptAdj = getAdaptAdj(meshID, nbrSplitElem.localID, nbrSplitElem.elemType, 0);
    nbrSplitElemAdaptAdj[nbrEdgeID] = splitElem;
    //BULK_DEBUG(printf("[%d] For nbrSplitElem %d set adjacency to %d across splitEdge\n",partitionID,nbrSplitElem.localID,splitElem.localID);)
    // POST: start-side operations
    splitElemAdaptAdj[edgeID] = nbrSplitElem;
    //BULK_DEBUG(printf("[%d] For splitElem %d set adjacency to %d across splitEdge\n",partitionID,splitElem.localID,nbrSplitElem.localID);)
  }
  else if (nbrElem.partID == -1) { // startElem's edgeID is on domain boundary
    adaptAdj *splitElemAdaptAdj = getAdaptAdj(meshID, splitElem.localID, splitElem.elemType, 0);
    splitElemAdaptAdj[edgeID] = adaptAdj(-1, -1, 0);
    //BULK_DEBUG(printf("[%d] For splitElem %d splitEdge is on the domain boundary.\n",partitionID,splitElem.localID);)
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
    adaptAdj *splitElemAdaptAdj = getAdaptAdj(meshID, splitElem.localID, splitElem.elemType, 0);
    splitElemAdaptAdj[edgeID] = nbrSplitElem;
    //BULK_DEBUG(printf("[%d] For splitElem %d set adjacency to %d across splitEdge\n",partitionID,splitElem.localID,nbrSplitElem.localID);)
  }    

  // unlock the two partitions
  localShadow->unlockRegion(lockRegionID);
  getAndDumpAdaptAdjacencies(meshID, meshPtr->nElems(), elemType, partitionID);
  return true;
}

void BulkAdapt::dumpConn()
{
  FEM_Elem &elem = meshPtr->elem[0]; // elem is local elements
  int nelems=FEM_Mesh_get_length(meshID, FEM_ELEM+0); // Get number of elements
  for (int i=0; i<nelems; i++) {
    int *conn = elem.connFor(i);
    CkPrintf("[%d] %d: %d %d %d %d\n", partitionID, i, conn[0], conn[1], conn[2], conn[3]);
  }
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

  adaptAdj *nbrSplitElemAdaptAdj = getAdaptAdj(meshPtr, nbrSplitElem.localID, nbrSplitElem.elemType, 0);
  nbrSplitElemAdaptAdj[nbrEdgeID] = splitElem;
  //BULK_DEBUG(printf("[%d] For nbrSplitElem %d set adjacency to %d across splitEdge\n",partitionID,nbrSplitElem.localID,splitElem.localID);)
  
  return nbrSplitElem;
}

int BulkAdapt::lock_3D_region(int elemID, int elemType, int edgeID, double prio, RegionID *lockRegionID)
{ // Lock region defined by all elements surrounding edgeID, i.e. the "star"
  int numElemsToLock=0, success=-1;
  adaptAdj *elemsToLock;
  adaptAdj startElem(partitionID, elemID, elemType);

  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::lock_3D_region acquiring list of elements to build locked region.\n",
		      partitionID));
  get_elemsToLock(startElem, &elemsToLock, edgeID, &numElemsToLock);

  success = localShadow->lockRegion(numElemsToLock, elemsToLock, lockRegionID, prio);
  free(elemsToLock);
  if (success==2) { // got the lock free and clear; refine the element
    BULK_DEBUG(CkPrintf("[%d] BulkAdapt::lock_3D_region: Lock (%d,%d,%6.4f) obtained.\n", partitionID,
			lockRegionID->chunkID, lockRegionID->localID, lockRegionID->prio));
  }
  else if (success==1) { // lock is pending
    BULK_DEBUG(CkPrintf("[%d] BulkAdapt::lock_3D_region: Lock (%d,%d,%6.4f) pending.\n", partitionID,
			lockRegionID->chunkID, lockRegionID->localID, lockRegionID->prio));
  }
  else if (success==0) { // lock failed
    BULK_DEBUG(CkPrintf("[%d] BulkAdapt::edge_bisect_3D: Lock (%d,%d,%6.4f) not obtained.\n", partitionID,
			lockRegionID->chunkID, lockRegionID->localID, lockRegionID->prio));
  }
  else {
    CkPrintf("Lock status=%d\n", success);
    CkAbort("Invalid lock return status!\n");
  }
  return success;
}

void BulkAdapt::unlock_3D_region(RegionID lockRegionID)
{ // unlock the region
  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::unlock_3D_region: Unlocking lock (%d,%d,%6.4f).\n", partitionID,
		      lockRegionID.chunkID, lockRegionID.localID, lockRegionID.prio));
  localShadow->unlockRegion(lockRegionID);
}

void BulkAdapt::unpend_3D_region(RegionID lockRegionID)
{ // unlock the region
  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::unpend_3D_region: Unlocking lock (%d,%d,%6.4f).\n", partitionID,
		      lockRegionID.chunkID, lockRegionID.localID, lockRegionID.prio));
  localShadow->unpendRegion(lockRegionID);
}


/// Perform a 3D edge bisection on a tetrahedron
bool BulkAdapt::edge_bisect_3D(int elemID, int elemType, int edgeID, RegionID lockRegionID)
{ // ASSERT: An edge can only be on one surface. Region around edge is locked, along with appropriate 
  // partitions and IDXL lists.
  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::edge_bisect_3D starts at elemID %d \n",partitionID,elemID));
  //dumpConn();
  //getAndDumpAdaptAdjacencies(meshID, meshPtr->nElems(), elemType, partitionID);

  int numElemsToLock=0;
  adaptAdj *elemsToLock;
  adaptAdj startElem(partitionID, elemID, elemType);

  CkAssert(lockRegionID == localShadow->holdingLock);

  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::edge_bisect_3D acquiring list of elements surrounding edge.\n",
		      partitionID));
  get_elemsToLock(startElem, &elemsToLock, edgeID, &numElemsToLock);

  // Find the nodes on the edge to be bisected
  int localNodes[3], localRelNodes[2]; 
  double nodeCoords[9];
  FEM_Elem &elems = meshPtr->elem[elemType]; // elems is all local elements
  FEM_DataAttribute *coord = meshPtr->node.getCoord(); // all local coords
  int *conn = elems.connFor(elemID); // conn points at elemID's ACTUAL data!

  // edgeID has element-relative node indices localRelNodes[0], localRelNodes[1]
  getRelNodes(edgeID, 4, &(localRelNodes[0]), &(localRelNodes[1]));
  localNodes[0] = conn[localRelNodes[0]];
  localNodes[1] = conn[localRelNodes[1]];
  double *n0co = (coord->getDouble()).getRow(localNodes[0]);
  double *n1co = (coord->getDouble()).getRow(localNodes[1]);
  nodeCoords[0] = n0co[0];
  nodeCoords[1] = n0co[1];
  nodeCoords[2] = n0co[2];
  nodeCoords[3] = n1co[0];
  nodeCoords[4] = n1co[1];
  nodeCoords[5] = n1co[2];

  // Create the new shared node that will be added
  midpoint(&(nodeCoords[0]), &(nodeCoords[3]), 3, &(nodeCoords[6]));
  localNodes[2] = add_node(3, &(nodeCoords[6]));
  int *chunks = (int *)malloc(numElemsToLock*sizeof(int));
  int numParts=0;

  for (int i=0; i<numElemsToLock; i++) {
    chunks[i] = -1;
    int j;
    for (j=0; j<numParts; j++) {
      if (chunks[j] == elemsToLock[i].partID) {
	break;
      }
    }
    chunks[j] = elemsToLock[i].partID;
    if (j==numParts) numParts++;
  }

  if (numParts > 1)
    make_node_shared(localNodes[2], numParts, &chunks[0]);

  free(chunks);

  // Perform the splits on all affected elements
  int numRemote=0;
  int tableID = getTableID();
  elemPairs[tableID] = (adaptAdj *)malloc(2*numElemsToLock*sizeof(adaptAdj));

  for (int i=0; i<numElemsToLock; i++) {
    if (elemsToLock[i].partID != partitionID) {
      elemPairs[tableID][2*i] = elemsToLock[i];
      numRemote++;
      shadowProxy[elemsToLock[i].partID].
	handle_split_3D(partitionID, i, tableID, elemsToLock[i], lockRegionID,
			get_idxl_for_node(localNodes[0], elemsToLock[i].partID), 
			get_idxl_for_node(localNodes[1], elemsToLock[i].partID), 
			get_idxl_for_node(localNodes[2], elemsToLock[i].partID));
    }
  }

  for (int i=0; i<numElemsToLock; i++) {
    if (elemsToLock[i].partID == partitionID) {
      elemPairs[tableID][2*i] = elemsToLock[i];
      adaptAdj *splitElem = local_split_3D(elemsToLock[i], localNodes[0], 
					   localNodes[1], localNodes[2]);
      elemPairs[tableID][2*i+1] = (*splitElem);
      delete splitElem;
    }
  }

  if (numRemote > 0) {
    localShadow->recv_splits(tableID, numRemote);
  }

  // Now we need to send out the elemPairs and make sure all
  // adjacencies are updated
  for (int i=0; i<numElemsToLock; i++) {
    if (elemsToLock[i].partID != partitionID) {
      shadowProxy[elemsToLock[i].partID].
	update_asterisk_3D(partitionID, i, elemsToLock[i], numElemsToLock, 
			   elemPairs[tableID], lockRegionID,
			   get_idxl_for_node(localNodes[0], elemsToLock[i].partID), 
			   get_idxl_for_node(localNodes[1], elemsToLock[i].partID),
			   get_idxl_for_node(localNodes[2], elemsToLock[i].partID));
    }
  }
  for (int i=0; i<numElemsToLock; i++) {
    if (elemsToLock[i].partID == partitionID) {
      local_update_asterisk_3D(i, elemsToLock[i], numElemsToLock, elemPairs[tableID],
			       localNodes[0], localNodes[1], localNodes[2]);
    }
  }

  // clean up elemPairs
  freeTableID(tableID);
  free(elemsToLock);

  //dumpConn();
  //getAndDumpAdaptAdjacencies(meshID, meshPtr->nElems(), elemType, partitionID);

  BULK_DEBUG(CkPrintf("[%d] BulkAdapt::edge_bisect_3D of elem %d successful.\n",partitionID, elemID));  
  return true;
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


  //BULK_DEBUG(printf("[%d] one_side_split_2D called for elem %d edge %d nodes %d %d \n",partitionID,startElem.localID,edgeID,*node1idx,*node2idx);)
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
  //BULK_DEBUG(printf("[%d] conn[%d] of elem %d changed to node %d\n",partitionID,relNode2,startElem.localID,bisectNode);)
  // in splitConn, replace node1idx with bisectNode
  splitConn[relNode1] = bisectNode;
  // add element split with splitConn
  int splitElemID = add_element(startElem.elemType, 3, &splitConn[0], elem.getMeshSizing(startElem.localID));
  //BULK_DEBUG(printf("[%d] new element %d with conn %d %d %d added \n", partitionID, splitElemID, splitConn[0], splitConn[1], splitConn[2]);)
  // copy startElem.localID's adapt adj for all edges.
  splitElem = adaptAdj(partitionID, splitElemID, startElem.elemType);
  adaptAdj *startElemAdaptAdj = getAdaptAdj(meshPtr, startElem.localID, startElem.elemType, 0);
  adaptAdj *splitElemAdaptAdj = getAdaptAdj(meshPtr, splitElem.localID, startElem.elemType, 0);
  memcpy(splitElemAdaptAdj, startElemAdaptAdj, 3*sizeof(adaptAdj));
  adaptAdj startElemNbr;  // startElem's original nbr on the edge that will now border with splitElem
  if (startSide) {
    // update startElemAdaptAdj for edge (edgeID+1)%3 to local splitElem
    startElemNbr = startElemAdaptAdj[(edgeID+1)%3];
    startElemAdaptAdj[(edgeID+1)%3] = splitElem;
    // update splitElemAdaptAdj for edge (edgeID+2)%3 to local startElem
    splitElemAdaptAdj[(edgeID+2)%3] = startElem;
    //BULK_DEBUG(printf("[%d] For startElem %d edge %d is now set to %d\n",partitionID,startElem.localID,(edgeID+1)%3,splitElem.localID));
    //BULK_DEBUG(printf("[%d] For splitElem %d edge %d is now set to %d\n",partitionID,splitElem.localID,(edgeID+2)%3,startElem.localID));
  }
  else {
    // update startElemAdaptAdj for edge (edgeID+1)%3 to local splitElem
    startElemNbr = startElemAdaptAdj[(edgeID+2)%3];
    startElemAdaptAdj[(edgeID+2)%3] = splitElem;
    // update splitElemAdaptAdj for edge (edgeID+2)%3 to local startElem
    splitElemAdaptAdj[(edgeID+1)%3] = startElem;
    //BULK_DEBUG(printf("[%d] For startElem %d edge %d is now set to %d\n",partitionID,startElem.localID,(edgeID+2)%3,splitElem.localID));
    //BULK_DEBUG(printf("[%d] For splitElem %d edge %d is now set to %d\n",partitionID,splitElem.localID,(edgeID+1)%3,startElem.localID));
  }
  if (startElemNbr.partID == startElem.partID) {
    replaceAdaptAdj(meshPtr, startElemNbr, startElem, splitElem);
    //BULK_DEBUG(printf("[%d] For startElemNbr %d replaced startElem %d with splitElem %d\n",partitionID,startElemNbr.localID,startElem.localID,splitElem.localID);)
  }
  else if (startElemNbr.partID != -1) { // startElemNbr exists and is remote
    // need to call replaceAdaptAdj on startElemNbr.partID
    shadowProxy[startElemNbr.partID].remote_adaptAdj_replace(startElemNbr, startElem, splitElem); 
  }
  // interpolate nodal data, copy startElem data to splitElem
  CkPrintf("WARNING: Data transfer not yet implemented.\n");
}



/* COMMUNICATION HELPERS FOR BULK ADAPTIVITY OPERATIONS 
   ARE LOCATED IN ParFUM_SA. */

void BulkAdapt::remote_adaptAdj_replace(adaptAdj elem, adaptAdj oldElem, adaptAdj newElem)
{
  replaceAdaptAdj(meshPtr, elem, oldElem, newElem);
}

void BulkAdapt::remote_edgeAdj_replace(int remotePartID, adaptAdj elem, adaptAdj oldElem, adaptAdj newElem, int n1_idxl, int n2_idxl)
{
  int n1 = get_node_from_idxl(n1_idxl, remotePartID), 
    n2 = get_node_from_idxl(n2_idxl, remotePartID);
  // Find the relative nodes on the edge to be bisected
  int relNodes[4]; 
  FEM_Elem &elems = meshPtr->elem[elem.elemType]; // elems is all local elements
  int *conn = elems.connFor(elem.localID); // conn points at elemID's ACTUAL data!
  fillNodes(relNodes, n1, n2, conn);
  // find which edgeID is bisected
  int edgeID = getEdgeID(relNodes[0], relNodes[1], 4, 3);

  replaceAdaptAdjOnEdge(meshPtr, elem, oldElem, newElem, edgeID);
}

void BulkAdapt::remote_edgeAdj_add(int remotePartID, adaptAdj adj, adaptAdj splitElem, int n1_idxl, int n2_idxl)
{
  int n1 = get_node_from_idxl(n1_idxl, remotePartID), 
    n2 = get_node_from_idxl(n2_idxl, remotePartID);
  // Find the relative nodes on the edge to be bisected
  int relNodes[4]; 
  FEM_Elem &elems = meshPtr->elem[adj.elemType]; // elems is all local elements
  int *conn = elems.connFor(adj.localID); // conn points at elemID's ACTUAL data!
  fillNodes(relNodes, n1, n2, conn);
  // find which edgeID is bisected
  int edgeID = getEdgeID(relNodes[0], relNodes[1], 4, 3);

  addToAdaptAdj(meshPtr, adj, edgeID, splitElem);
}

void BulkAdapt::handle_split_3D(int remotePartID, int pos, int tableID, adaptAdj elem, 
				RegionID lockRegionID, int n1_idxl, int n2_idxl, int n5_idxl)
{
  int n1 = get_node_from_idxl(n1_idxl, remotePartID), 
    n2 = get_node_from_idxl(n2_idxl, remotePartID), 
    n5;

  if (is_node_in_idxl(n5_idxl, remotePartID)) {
    n5 = get_node_from_idxl(n5_idxl, remotePartID);
  }
  else {
    FEM_DataAttribute *coord = meshPtr->node.getCoord(); // all local coords
    double *n0co = (coord->getDouble()).getRow(n1);
    double *n1co = (coord->getDouble()).getRow(n2);
    double nodeCoords[9];
    nodeCoords[0] = n0co[0];
    nodeCoords[1] = n0co[1];
    nodeCoords[2] = n0co[2];
    nodeCoords[3] = n1co[0];
    nodeCoords[4] = n1co[1];
    nodeCoords[5] = n1co[2];
    midpoint(&(nodeCoords[0]), &(nodeCoords[3]), 3, &(nodeCoords[6]));
    n5 = add_node(3, &(nodeCoords[6]));  
    
    // Find the relative nodes on the edge to be bisected
    int relNodes[4]; 
    FEM_Elem &elems = meshPtr->elem[elem.elemType]; // elems is all local elements
    int *conn = elems.connFor(elem.localID); // conn points at elemID's ACTUAL data!
    
    fillNodes(relNodes, n1, n2, conn);
    // find which edgeID is bisected
    int edgeID = getEdgeID(relNodes[0], relNodes[1], 4, 3);
    
    // find elements on edge to be bisected
    int numElemsToLock = 0;
    adaptAdj *elemsToLock;
    get_elemsToLock(elem, &elemsToLock, edgeID, &numElemsToLock);
    
    // find chunks that share the edge to be bisected
    int *chunks = (int *)malloc(numElemsToLock*sizeof(int));
    int numParts=0;
    for (int i=0; i<numElemsToLock; i++) {
      chunks[i] = -1;
      int j;
      for (j=0; j<numParts; j++) {
	if (chunks[j] == elemsToLock[i].partID) {
	  break;
	}
      }
      chunks[j] = elemsToLock[i].partID;
      if (j==numParts) numParts++;
    }
    
    if (numParts > 1)
      make_node_shared(n5, numParts, &chunks[0]);
    
    free(chunks);
    free(elemsToLock);
  }

  adaptAdj *splitElem = local_split_3D(elem, n1, n2, n5);
  shadowProxy[remotePartID].recv_split_3D(pos, tableID, elem, *splitElem);
  delete splitElem;
}

void BulkAdapt::recv_split_3D(int pos, int tableID, adaptAdj elem,
			      adaptAdj splitElem) 
{
  assert(elemPairs[tableID][2*pos] == elem);
  elemPairs[tableID][2*pos+1] = splitElem;
  numGathered[tableID]++;
}

bool BulkAdapt::all_splits_received(int tableID, int expectedSplits)
{
  return (expectedSplits == numGathered[tableID]);
}

void BulkAdapt::update_asterisk_3D(int remotePartID, int i, adaptAdj elem, 
				   int numElemPairs, adaptAdj *elemPairs, RegionID lockRegionID,
				   int n1_idxl, int n2_idxl, int n5_idxl)
{
  CkAssert(remotePartID != elem.partID);
  int n1 = get_node_from_idxl(n1_idxl, remotePartID), 
    n2 = get_node_from_idxl(n2_idxl, remotePartID), 
    n5 = get_node_from_idxl(n5_idxl, remotePartID);
  //printf("[%d] update_asterisk_3D(from %d): elem %d n1=%d n2=%d n5=%d\n",
  //elem.partID, remotePartID, elem.localID, n1, n2, n5);
  local_update_asterisk_3D(i, elem, numElemPairs, elemPairs, n1, n2, n5);
}


/* LOCAL HELPERS FOR BULK ADAPTIVITY OPERATIONS */

/** Add a new element to the mesh. 
 * Update its connectivity
 * Return index of new element
 * */
int BulkAdapt::add_element(int elemType,int nodesPerElem,int *conn, double sizing){ 
  //since the element array might be resized we need to set the correct thread
  //context before we call get_next_invalid
  localShadow->setRunningTCharm();
  FEM_Elem &elem = meshPtr->elem[elemType];
  int newElem = elem.get_next_invalid();
  elem.connIs(newElem,conn);
  elem.setMeshSizing(newElem, sizing);

  int nAdj;
  adaptAdj* adaptAdjacencies = lookupAdaptAdjacencies(
        meshPtr, elemType, &nAdj);
  for (int a = 0; a<nAdj; ++a) {
    //adaptAdjacencies[newElem*nAdj + a].partID = TCHARM_Element();
    adaptAdjacencies[newElem*nAdj + a].partID = -1;
    adaptAdjacencies[newElem*nAdj + a].localID = -1;
    adaptAdjacencies[newElem*nAdj + a].elemType = elemType;
  }
  CkVec<adaptAdj>** adaptEdgeAdjacencies = 
    lookupEdgeAdaptAdjacencies(meshPtr,elemType,&nAdj);
  for (int a=0; a<nAdj; ++a) {
    adaptEdgeAdjacencies[newElem*nAdj + a] = new CkVec<adaptAdj>;
  }
  return newElem;
}

/** Update the conn of an element*/
void BulkAdapt::update_element_conn(int elemType,int elemID,int nodesPerElem,int *conn){
  FEM_Elem &elem = meshPtr->elem[elemType];
  elem.connIs(elemID,conn);
}

bool BulkAdapt::isLongest(int elem, int elemType, double len) {
  FEM_Elem &elems = meshPtr->elem[elemType];
  int *conn = elems.connFor(elem);
  if ((len < length(conn[0], conn[1], 3)) || (len < length(conn[0], conn[2], 3)) || (len < length(conn[0], conn[3], 3)) || (len < length(conn[1], conn[2], 3)) || (len < length(conn[1], conn[3], 3)) || (len < length(conn[2], conn[3], 3))) {
    return false;
  }
  return true;
}



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


double BulkAdapt::length(int n1, int n2, int dim) {
  FEM_DataAttribute *coord = meshPtr->node.getCoord();
  double *coordsn1 = (coord->getDouble()).getRow(n1);
  double *coordsn2 = (coord->getDouble()).getRow(n2);
  return length(coordsn1, coordsn2, dim);
}

double BulkAdapt::length(double *n1, double *n2, int dim) {
  double d, ds_sum=0.0;
  for (int i=0; i<dim; i++) {
    if(n1[i]<-1.0 || n2[i]<-1.0) return -2.0;
    d = n1[i] - n2[i];
    ds_sum += d*d;
  }
  return (sqrt(ds_sum));
}



/** Update the coordinates of the given node
*/
void BulkAdapt::update_node_coord(int nodeID,int dim,double *coords){
  FEM_DataAttribute *coord = meshPtr->node.getCoord();
  (coord->getDouble()).setRow(nodeID,coords);
}

void BulkAdapt::make_node_shared(int nodeID,int numSharedChunks,int *sharedChunks){
  for(int i=0;i<numSharedChunks;i++){
    IDXL_List &sharedList = meshPtr->node.shared.addList(sharedChunks[i]);
    sharedList.push_back(nodeID);
  }
  meshPtr->node.shared.flushMap();
}

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
  return -1;
}

int BulkAdapt::get_node_from_idxl(int node_idxl, int partID)
{
  IDXL_List *list = meshPtr->node.shared.getIdxlListN(partID);
  CkAssert(list!=NULL);
  CkAssert(list->size()>node_idxl);
  return (*list)[node_idxl];
}

bool BulkAdapt::is_node_in_idxl(int node_idxl, int partID)
{
  IDXL_List *list = meshPtr->node.shared.getIdxlListN(partID);
  return(list->size()>node_idxl);
}

void BulkAdapt::get_elemsToLock(adaptAdj startElem, adaptAdj **elemsToLock, int edgeID, int *count)
{
  CkVec<adaptAdj>* nbrElems;
  // find the elements adjacent to startElem along the edge edgeID
  //BULK_DEBUG(CkPrintf("[%d] BulkAdapt::get_elemsToLock: calling getEdgeAdaptAdj on elem %d\n",partitionID,startElem.localID));

  nbrElems = getEdgeAdaptAdj(meshID, startElem.localID, startElem.elemType, 
			     edgeID);
  // extract adjacencies from CkVec into array needed by the locking code 
  (*count) = nbrElems->size();
  (*elemsToLock) = (adaptAdj *)malloc((*count + 1) * sizeof(adaptAdj));

  for (int i=0; i<*count; i++) {
    (*elemsToLock)[i] = (*nbrElems)[i];
  }
  // add the start element
  (*elemsToLock)[*count] = startElem;
  (*count)++;

  /*
  printf("Elems to lock: ");
  for (int i=0; i<*count; i++) {
    printf("(%d, %d, %d) ", (*elemsToLock)[i].partID, (*elemsToLock)[i].localID,
	   (*elemsToLock)[i].elemType);
  }
  printf("\n");
  */
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
  return -1;
}


void getRelNodes(int edgeID, int nodesPerElem, int *r1, int *r2)
{
  if (nodesPerElem == 3) {
    (*r1) = edgeID;
    (*r2) = (edgeID + 1) % 3;
  }
  else if (nodesPerElem == 4) {
    if (edgeID < 3) {
      (*r1) = 0;
      (*r2) = edgeID+1;
    }
    else if (edgeID < 5) {
      (*r1) = 1;
      (*r2) = edgeID-1;
    }
    else {
      (*r1) = 2; 
      (*r2) = 3;
    }
  }
}

int getEdgeID(int node1, int node2, int nodesPerElem, int dim) {
  switch(dim){
  case 2: {
    switch(nodesPerElem){
    case 3:
      {
	static int edgeFromNode[3][3]={{-1,0,2},{0,-1,1},{2,1,-1}};
	return edgeFromNode[node1][node2];
      }
      break;
    default:
      CkAbort("This shape is not yet implemented");
    };
    break;
  }
  case 3: {
    switch(nodesPerElem){
    case 4:
      {
	static int edgeFromNode[4][4]={{-1,0,1,2},{0,-1,3,4},{1,3,-1,5},{2,4,5,-1}};
	return edgeFromNode[node1][node2];
      }
      break;
    default:
      CkAbort("This shape is not yet implemented");
    };
    break;
  }
  default:
    CkAbort("This dimension not yet implemented");
  };
  return -1;
}


int getFaceID(int node1, int node2, int node3, int nodesPerElem)
{
  switch(nodesPerElem){
  case 4: {
    static int faceFromNode[4][4][4]=
      {{{-1,-1,-1,-1},  {-1,-1, 0, 1},  {-1, 0,-1, 2},  {-1, 1, 2,-1}}, 
       {{-1,-1, 0, 1},  {-1,-1,-1,-1},   {0,-1,-1, 3},   {1,-1, 3,-1}}, 
       {{-1, 0,-1, 2},   {0,-1,-1, 3},  {-1,-1,-1,-1},   {2, 3,-1,-1}},
       {{-1, 1, 2,-1},   {1,-1, 3,-1},   {2, 3,-1,-1},  {-1,-1,-1,-1}}};
    return faceFromNode[node1][node2][node3];
    break;
  }
  default:
    CkAbort("This shape is not yet implemented");
  };
  return -1;
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

void fillNodes(int *relNode, int n1, int n2, int *conn)
{ // Given real node IDs n1 and n2 representing, respectively, the
  // unchanged and the changed ends of the edge to bisect, and given
  // the conn of a tet with edge (n1, n2), fill out a set of relative
  // node numberings that put n1 and n2 in the first two relNode
  // positions, respectively, and the remaining nodes in the remaining
  // positions. i.e.:
  // conn[relNode[0]] = n1, conn[relNode[1] = n2, and the rest map to 
  // either of the remaining nodes.
  for (int i=0; i<4; i++) {
    if (conn[i] == n1)
      relNode[0] = i;
    else if (conn[i] == n2)
      relNode[1] = i;
  }
  if ((relNode[0] != 0) && (relNode[1] != 0))
    relNode[2] = 0;
  else if ((relNode[0] != 1) && (relNode[1] != 1))
    relNode[2] = 1;
  else relNode[2] = 2;
  relNode[3] = 6 - relNode[0] - relNode[1] - relNode[2];
}

/** Perform all local mesh mods and updates for a local tet split */
adaptAdj *BulkAdapt::local_split_3D(adaptAdj elem, int n1, int n2, int n5)
{
  CkPrintf("[%d] BEGIN local_split_3D\n", partitionID);
  int nElems = meshPtr->lookup(FEM_ELEM+elem.elemType,"BulkAdapt::local_split_3D")->size();  // Get number of elements
  int nNodes=meshPtr->lookup(FEM_NODE,"BulkAdapt::local_split_3D")->size();    // Get number of nodes
  CkAssert(elem.localID < nElems);
  CkAssert((n1 < nNodes) && (n1 >= 0));
  CkAssert((n2 < nNodes) && (n2 >= 0));
  CkAssert((n5 < nNodes) && (n5 >= 0));

  // get elem's conn
  FEM_Elem &elems = meshPtr->elem[elem.elemType];
  int *conn = elems.connFor(elem.localID);
  // make splitElem's conn and init with elem's conn
  int splitConn[4];
  memcpy(&splitConn[0], conn, 4*sizeof(int));
  // find relative node numbers for n1 and n2
  int rel_n1 = -1, rel_n2 = -1;
  for (int i=0; i<4; i++) {
    if (conn[i] == n1) rel_n1 = i;
    if (conn[i] == n2) rel_n2 = i;
  }
  CkAssert((rel_n1 != -1) && (rel_n2 != -1));
  // adjust elem's conn to reflect the split
  CkPrintf("[%d] elem %d conn was %d,%d,%d,%d\n", partitionID, elem.localID, conn[0], conn[1], conn[2], conn[3]);
  CkPrintf("[%d] old edge is %d,%d, inserting node %d in between\n", partitionID, n1, n2, n5);
  conn[rel_n2] = n5;
  CkPrintf("[%d] modifying elem %d to be %d,%d,%d,%d\n", partitionID, elem.localID, conn[0], conn[1], conn[2], conn[3]);
  // adjust splitElem's conn to reflect the split
  splitConn[rel_n1] = n5;
  // add splitElem
  int splitElemID = add_element(elem.elemType, 4, &splitConn[0], elems.getMeshSizing(elem.localID));
  CkPrintf("[%d] Adding elem %d with conn %d,%d,%d,%d\n", partitionID, splitElemID, splitConn[0], splitConn[1], splitConn[2], splitConn[3]); 
  adaptAdj *splitElem = new adaptAdj(partitionID, splitElemID, elem.elemType);
  // call updates here
  update_local_face_adj(elem, *splitElem, n1, n2, n5);
  update_local_edge_adj(elem, *splitElem, n1, n2, n5);
  CkPrintf("[%d] END local_split_3D\n", partitionID);
  return splitElem;
}

void BulkAdapt::local_update_asterisk_3D(int i, adaptAdj elem, int numElemPairs,
					 adaptAdj *elemPairs, 
					 int n1, int n2, int n5)
{
  FEM_Elem &elems = meshPtr->elem[elem.elemType];
  int *conn = elems.connFor(elem.localID); 
  int n3=-1, n4=-1;
  for (int i=0; i<4; i++) {
    if ((conn[i] != n1) && (conn[i] != n5) && (n3 == -1))
      n3 = conn[i]; 
    else if ((conn[i] != n1) && (conn[i] != n5))
      n4 = conn[i];
  }
  // derive relative nodes
  int relNode[4];
  relNode[0] = getRelNode(n1, conn, 4);
  relNode[1] = getRelNode(n5, conn, 4);
  relNode[2] = getRelNode(n3, conn, 4);
  relNode[3] = getRelNode(n4, conn, 4);
  int face[4]; // face[0] and [1] are split; others are not
  face[0] = (relNode[0] + relNode[1] + relNode[2]) - 3;
  face[1] = (relNode[0] + relNode[1] + relNode[3]) - 3;
  face[2] = (relNode[0] + relNode[3] + relNode[2]) - 3;
  face[3] = (relNode[1] + relNode[3] + relNode[2]) - 3;
  adaptAdj neighbors[4]; // elem's neighbors
  neighbors[0] = *getFaceAdaptAdj(meshPtr, elem.localID, elem.elemType, face[0]);
  neighbors[1] = *getFaceAdaptAdj(meshPtr, elem.localID, elem.elemType, face[1]);
  neighbors[2] = *getFaceAdaptAdj(meshPtr, elem.localID, elem.elemType, face[2]);
  neighbors[3] = *getFaceAdaptAdj(meshPtr, elem.localID, elem.elemType, face[3]);

  adaptAdj splitElem = neighbors[3], nbr1 = neighbors[0], nbr2 = neighbors[1];
  adaptAdj nbr1split(-1,-1,0), nbr2split(-1,-1,0);
  int found=0;
  if (nbr1.localID != -1) found++;
  if (nbr2.localID != -1) found++;
  for (int i=0; i<numElemPairs; i++) {
    if (elemPairs[2*i] == nbr1) {
      nbr1split = elemPairs[2*i+1];
      found--;
    }
    else if (elemPairs[2*i] == nbr2) {
      nbr2split = elemPairs[2*i+1];
      found--;
    }
    if (found == 0) break;
  }
  int *splitConn = elems.connFor(splitElem.localID); 
  int splitRelNode[4];
  splitRelNode[0] = getRelNode(n5, splitConn, 4);
  splitRelNode[1] = getRelNode(n2, splitConn, 4);
  splitRelNode[2] = getRelNode(n3, splitConn, 4);
  splitRelNode[3] = getRelNode(n4, splitConn, 4);
  int splitFace[4]; // splitFace[0] and [1] are split; others are not
  splitFace[0] = (splitRelNode[0] + splitRelNode[1] + splitRelNode[2]) - 3;
  splitFace[1] = (splitRelNode[0] + splitRelNode[1] + splitRelNode[3]) - 3;
  splitFace[2] = (splitRelNode[0] + splitRelNode[3] + splitRelNode[2]) - 3;
  splitFace[3] = (splitRelNode[1] + splitRelNode[3] + splitRelNode[2]) - 3;

  // Update faces first
  replaceAdaptAdj(meshPtr, splitElem, nbr1, nbr1split);
  replaceAdaptAdj(meshPtr, splitElem, nbr2, nbr2split);
  // Now, update edges
  // Here are the edges to update for elem:
  int n3_n5, n4_n5;
  // Here are the edges to update for splitElem:
  int n5_n2, n5_n3, n5_n4, n2_n3, n2_n4;
  // set the edge IDs
  n3_n5 = getEdgeID(relNode[2], relNode[1], 4, 3);
  n4_n5 = getEdgeID(relNode[3], relNode[1], 4, 3);
  n5_n2 = getEdgeID(splitRelNode[0], splitRelNode[1], 4, 3);
  n5_n3 = getEdgeID(splitRelNode[0], splitRelNode[2], 4, 3);
  n5_n4 = getEdgeID(splitRelNode[0], splitRelNode[3], 4, 3);
  n2_n3 = getEdgeID(splitRelNode[1], splitRelNode[2], 4, 3);
  n2_n4 = getEdgeID(splitRelNode[1], splitRelNode[3], 4, 3);

  clearEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n3_n5);  
  clearEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n4_n5);  
  clearEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n2);  
  clearEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n3);  
  clearEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n4);  

  addEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n3_n5, splitElem);
  if (nbr1.localID != -1) {
    addEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n3_n5, nbr1);
    addEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n3_n5, nbr1split);
  }

  addEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n4_n5, splitElem);
  if (nbr2.localID != -1) {
    addEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n4_n5, nbr2);
    addEdgeAdjacency(meshPtr, elem.localID, elem.elemType, n4_n5, nbr2split);
  }

  addEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n3, elem);
  if (nbr1.localID != -1) {
    addEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n3, nbr1);
    addEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n3, nbr1split);
  }

  addEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n4, elem);
  if (nbr2.localID != -1) {
    addEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n4, nbr2);
    addEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n4, nbr2split);
  }

  for (int i=0; i<numElemPairs; i++) {
    if (splitElem != elemPairs[2*i+1]) {
      addEdgeAdjacency(meshPtr, splitElem.localID, splitElem.elemType, n5_n2,
		       elemPairs[2*i+1]);
    }
  }

  if(nbr1.localID != -1)
    replaceAdaptAdjOnEdge(meshPtr, splitElem, nbr1, nbr1split, n2_n3);
  if(nbr2.localID != -1)
    replaceAdaptAdjOnEdge(meshPtr, splitElem, nbr2, nbr2split, n2_n4);
}


/** Perform local face adjacency updates associated with a split */
void BulkAdapt::update_local_face_adj(adaptAdj elem, adaptAdj splitElem, 
				      int n1, int n2, int n5)
{
  // init splitElem's face and edge adjacencies to elem's
  CmiMemoryCheck();
  copyAdaptAdj(meshPtr, &elem, &splitElem);
  // get n3 and n4 from conn
  FEM_Elem &elems = meshPtr->elem[elem.elemType];
  int *conn = elems.connFor(elem.localID); 
  int n3=-1, n4=-1;
  for (int i=0; i<4; i++) {
    if ((conn[i] != n1) && (conn[i] != n5) && (n3 == -1))
      n3 = conn[i]; 
    else if ((conn[i] != n1) && (conn[i] != n5))
      n4 = conn[i];
  }
  // derive relative nodes
  int relNode[4];
  relNode[0] = getRelNode(n1, conn, 4);
  relNode[1] = getRelNode(n5, conn, 4);
  relNode[2] = getRelNode(n3, conn, 4);
  relNode[3] = getRelNode(n4, conn, 4);
  int face[4]; // face[0] and [1] are split; others are not
  face[0] = (relNode[0] + relNode[1] + relNode[2]) - 3;
  face[1] = (relNode[0] + relNode[1] + relNode[3]) - 3;
  face[2] = (relNode[0] + relNode[3] + relNode[2]) - 3;
  face[3] = (relNode[1] + relNode[3] + relNode[2]) - 3;
  adaptAdj neighbors[4]; // elem's neighbors
  neighbors[0] = *getFaceAdaptAdj(meshPtr,elem.localID, elem.elemType, face[0]);
  neighbors[1] = *getFaceAdaptAdj(meshPtr,elem.localID, elem.elemType, face[1]);
  neighbors[2] = *getFaceAdaptAdj(meshPtr,elem.localID, elem.elemType, face[2]);
  neighbors[3] = *getFaceAdaptAdj(meshPtr,elem.localID, elem.elemType, face[3]);
  // elem's non-split faces have neighbors[2] and [3]
  // update elem's new nbr from neighbors[3] to splitElem
  setAdaptAdj(meshPtr, elem, face[3], splitElem);
  //replaceAdaptAdj(meshPtr, elem, neighbors[3], splitElem);
  // update splitElem's neighbor from neighbors[2] to elem
  setAdaptAdj(meshPtr, splitElem, face[2], elem);
  //replaceAdaptAdj(meshPtr, splitElem, neighbors[2], elem);
  // update elem's nbr's back-adjacency from elem to splitElem
  if (neighbors[3].partID == elem.partID) {
    replaceAdaptAdj(meshPtr, neighbors[3], elem, splitElem);
  }
  else if (neighbors[3].partID != -1) { // elem's nbr exists and is remote
    shadowProxy[neighbors[3].partID].remote_adaptAdj_replace(neighbors[3], elem, splitElem); 
  }
}

/** Perform local edge adjacency updates associated with a split */
void BulkAdapt::update_local_edge_adj(adaptAdj elem, adaptAdj splitElem, 
				      int n1, int n2, int n5)
{
  copyEdgeAdaptAdj(meshPtr, &elem, &splitElem);
  // first, extract the modifiable data structures in question
  CkVec<adaptAdj> *elemEdgeAdj[6];
  CkVec<adaptAdj> *splitElemEdgeAdj[6];
  for (int i=0; i<6; i++) {
    elemEdgeAdj[i] = getEdgeAdaptAdj(meshPtr, elem.localID, elem.elemType, i);
    splitElemEdgeAdj[i] = getEdgeAdaptAdj(meshPtr, splitElem.localID, splitElem.elemType, i);
  }
  // the edges modified in this operation are the old edges of elem,
  // now on splitElem: (n2,n3) and (n2,n4), the old edge (n3,n4) which
  // was on elem, but is now on both elem and splitElem, and the new
  // edges that get added incident on n5: (n3,n5) and (n4,n5).  The
  // new edge (n2,n5) must be updated in a different context.
  
  // Here's how each of these needs to be modified:
  // (n2,n3): originally on elem, for splitElem, it will be identical, but we
  // need to replace elem with splitElem on all the other elements along that 
  // edge.
  // (n2,n4): identical to (n2,n3)
  // (n3,n4): elem needs to take it's original list and add splitElem to it; 
  // splitElem takes elem's original list and adds element to it; then
  // all the elements in that original list need to add splitElem.
  // (n3,n5): start with an empty list on both elem and splitElem.
  // elem adds splitElem, splitElem adds elem.  Other elements
  // surrounding the split edge (n1,n2) must be added outside of this
  // operation
  // (n4,n5): identical to (n3,n5)

  // Start with splitElem: get relNodes and edgeIDs
  FEM_Elem &elems = meshPtr->elem[splitElem.elemType];
  int *conn = elems.connFor(splitElem.localID); 
  int n3=-1, n4=-1;
  for (int i=0; i<4; i++) {
    if ((conn[i] != n5) && (conn[i] != n2) && (n3 == -1))
      n3 = conn[i]; 
    else if ((conn[i] != n5) && (conn[i] != n2))
      n4 = conn[i];
  }
  // derive relative nodes
  int relNode[4];
  relNode[0] = getRelNode(n5, conn, 4);
  relNode[1] = getRelNode(n2, conn, 4);
  relNode[2] = getRelNode(n3, conn, 4);
  relNode[3] = getRelNode(n4, conn, 4);
  
  // edgeIDs on splitElem
  int n2_n3 = getEdgeID(relNode[1], relNode[2], 4, 3);
  int n2_n4 = getEdgeID(relNode[1], relNode[3], 4, 3);
  int n3_n4 = getEdgeID(relNode[2], relNode[3], 4, 3);
  int n3_n5 = getEdgeID(relNode[2], relNode[0], 4, 3);
  int n4_n5 = getEdgeID(relNode[3], relNode[0], 4, 3);

  // Get similar info for elem
  int *elemConn = elems.connFor(elem.localID); 
  // derive relative nodes
  int elem_relNode[4];
  elem_relNode[0] = getRelNode(n1, elemConn, 4);
  elem_relNode[1] = getRelNode(n5, elemConn, 4);
  elem_relNode[2] = getRelNode(n3, elemConn, 4);
  elem_relNode[3] = getRelNode(n4, elemConn, 4);

  // edgeIDs on elem
  int elem_n3_n4 = getEdgeID(relNode[2], relNode[3], 4, 3);
  int elem_n3_n5 = getEdgeID(relNode[2], relNode[1], 4, 3);
  int elem_n4_n5 = getEdgeID(relNode[3], relNode[1], 4, 3);

  // split face IDs on elem
  int face[4]; // face[0] and [1] are split; others are not
  face[0] = (elem_relNode[0] + elem_relNode[1] + elem_relNode[2]) - 3;
  face[1] = (elem_relNode[0] + elem_relNode[1] + elem_relNode[3]) - 3;
  // split neighbors on elem
  adaptAdj elem_neighbors[4]; // elem's neighbors
  elem_neighbors[0] = *getFaceAdaptAdj(meshPtr,elem.localID, elem.elemType, face[0]);
  elem_neighbors[1] = *getFaceAdaptAdj(meshPtr,elem.localID, elem.elemType, face[1]);

  // Now we're ready to go to town
  // n2_n3
  for (int i=0; i<elemEdgeAdj[elem_n3_n5]->size(); i++) {
    adaptAdj adj = (*elemEdgeAdj[elem_n3_n5])[i];
    if ((adj != elem_neighbors[0]) && (adj != elem_neighbors[1])) {
      if (adj.partID == splitElem.partID) { // do the local replace on adj
	int *adjConn = elems.connFor(adj.localID); 
	// derive relative nodes
	int r2, r3;
	r2 = getRelNode(n2, adjConn, 4);
	r3 = getRelNode(n3, adjConn, 4);
	// edgeID on adj
	int edgeID = getEdgeID(r2, r3, 4, 3);
	replaceAdaptAdjOnEdge(meshPtr, adj, elem, splitElem, edgeID);
      }
      else if (adj.partID != -1) { // call remote replacement
	int n2_idxl = get_idxl_for_node(n2, adj.partID);
	int n3_idxl = get_idxl_for_node(n3, adj.partID);
	//printf("[%d] A:calling remote edgeAdj replace on %don%d, to replace %don%d with %don%d, with edge at %d,%d thinking elem's forbidden face neighbors are %don%d and %don%d\n", partitionID, adj.localID, adj.partID, elem.localID, elem.partID, splitElem.localID, splitElem.partID, n2, n3, elem_neighbors[0].localID, elem_neighbors[0].partID, elem_neighbors[1].localID, elem_neighbors[1].partID);
	shadowProxy[adj.partID].remote_edgeAdj_replace(partitionID, adj, elem, 
						       splitElem, n2_idxl, n3_idxl);
      }
    }
  }
  // wow!  all that was just for the (n2,n3) edge!  This sucks!
  // n2_n4 -- easy, just like the previous one
  for (int i=0; i<elemEdgeAdj[elem_n4_n5]->size(); i++) {
    adaptAdj adj = (*elemEdgeAdj[elem_n4_n5])[i];
    if ((adj != elem_neighbors[0]) && (adj != elem_neighbors[1])) {
      if (adj.partID == splitElem.partID) { // do the local replace on adj
	int *adjConn = elems.connFor(adj.localID); 
	// derive relative nodes
	int r2, r4;
	r2 = getRelNode(n2, adjConn, 4);
	r4 = getRelNode(n4, adjConn, 4);
	// edgeID on adj
	int edgeID = getEdgeID(r2, r4, 4, 3);
	replaceAdaptAdjOnEdge(meshPtr, adj, elem, splitElem, edgeID);
      }
      else if (adj.partID != -1) { // call remote replacement
	int n2_idxl = get_idxl_for_node(n2, adj.partID);
	int n4_idxl = get_idxl_for_node(n4, adj.partID);
	//printf("[%d] B:calling remote edgeAdj replace on %don%d, to replace %don%d with %don%d, with edge at %d,%d, thinking elem's forbidden face neighbors are %don%d and %don%d\n", partitionID, adj.localID, adj.partID, elem.localID, elem.partID, splitElem.localID, splitElem.partID, n2, n4, elem_neighbors[0].localID, elem_neighbors[0].partID, elem_neighbors[1].localID, elem_neighbors[1].partID);
	shadowProxy[adj.partID].remote_edgeAdj_replace(partitionID, adj, elem, 
						       splitElem, n2_idxl, n4_idxl);
      }
    }
  }
  // n3_n4: elem needs to take it's original list and add splitElem to it; 
  // splitElem takes elem's original list and adds element to it; then
  // all the elements in that original list need to add splitElem.
  // This is *similar* to the previous case, with the exceptions that
  // we are *adding* instead of replacing on the other elems on the
  // edge, and the addition of elem and splitElem to each other's edgeAdj.
  for (int i=0; i<elemEdgeAdj[elem_n3_n4]->size(); i++) {
    adaptAdj adj = (*elemEdgeAdj[elem_n3_n4])[i];
    if (adj.partID == splitElem.partID) { // do the local replace on adj
      int *adjConn = elems.connFor(adj.localID); 
      // derive relative nodes
      int r3, r4;
      r3 = getRelNode(n3, adjConn, 4);
      r4 = getRelNode(n4, adjConn, 4);
      // edgeID on adj
      int edgeID = getEdgeID(r3, r4, 4, 3);
      addToAdaptAdj(meshPtr, adj, edgeID, splitElem);
    }
    else if (adj.partID != -1) { // call remote replacement
      int n3_idxl = get_idxl_for_node(n3, adj.partID);
      int n4_idxl = get_idxl_for_node(n4, adj.partID);
      shadowProxy[adj.partID].remote_edgeAdj_add(partitionID, adj, splitElem, 
						 n3_idxl, n4_idxl);
    }
  }
  addToAdaptAdj(meshPtr, splitElem, n3_n4, elem);
  addToAdaptAdj(meshPtr, elem, elem_n3_n4, splitElem);
  // The last two cases are performed elsewhere
}

