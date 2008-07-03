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

  /// Data structure to gather elem-split pairs
  adaptAdj *elemPairs[10];
  int freeTable[10];
  int numGathered[10];
  int firstFree;
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

  int getTableID() { 
    int x = firstFree; 
    freeTable[x]=0; 
    numGathered[x] = 0;
    firstFree++; 
    while (!freeTable[firstFree]) {
      firstFree++;
      if (firstFree == 10) {
	CkPrintf("ERROR: elemPairs table is full!\n");
	break;
      }
    }
    return x; 
  }
  void freeTableID(int x) { 
    freeTable[x]=1; 
    free(elemPairs[x]);
    if (x<firstFree) firstFree = x;
  }

  /* BULK MESH OPERATIONS: These are all called locally, but may invoke
     remote operations. */

  /// Perform an edge bisection (2D and 3D).
  /** Locks mesh and any affected IDXL lists, performs operation,
      updates adapt adjacencies, and unlocks mesh & IDXL
      lists. Returns zero if the lock fails, positive if the operation
      suceeds, and negative if the operations fails for some other
      reason. */
    bool edge_bisect(int elemID, int elemType, int edgeID, int dim, RegionID lockRegionID);
  bool edge_bisect_2D(int elemID, int elemType, int edgeID);
  bool edge_bisect_3D(int elemID, int elemType, int edgeID, RegionID lockRegionID);

  //TODO: add elemType to the prototype of all the following mesh-modification functions
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

  /// Perform a single side of an edge_bisect operation
  void one_side_split_2D(adaptAdj &startElem, adaptAdj &splitElem, int edgeID,
			 int *node1idx, int *node2idx, int *newNodeID,
			 bool startSide);

  /* COMMUNICATION HELPERS FOR BULK ADAPTIVITY OPERATIONS */
  adaptAdj remote_edge_bisect_2D(adaptAdj nbrElem, adaptAdj splitElem, 
				 int new_idxl, int n1_idxl, int n2_idxl, 
				 int remotePartID);

  void remote_adaptAdj_replace(adaptAdj elem, adaptAdj oldElem, 
			       adaptAdj newElem);
  void remote_edgeAdj_replace(int remotePartID, adaptAdj adj, adaptAdj elem, 
			      adaptAdj splitElem, int n1_idxl, int n2_idxl);
  void remote_edgeAdj_add(int remotePartID, adaptAdj adj, adaptAdj splitElem,
			  int n1_idxl, int n2_idxl);

  adaptAdj remote_edge_bisect_3D(adaptAdj nbrElem, adaptAdj splitElem, 
				 int new_idxl, int n1_idxl, int n2_idxl, 
				 int remotePartID);

  void handle_split_3D(int remotePartID, int pos, int tableID, adaptAdj elem, RegionID lockRegionID,
		       int n1_idxl, int n2_idxl, int n5_idxl);
  void recv_split_3D(int pos, int tableID, adaptAdj elem, adaptAdj splitElem);
  bool all_splits_received(int tableID, int expectedSplits);
  void update_asterisk_3D(int remotePartID, int i, adaptAdj elem, 
			  int numElemPairs, adaptAdj *elemPairs, RegionID lockRegionID,
			  int n1_idxl, int n2_idxl, int n5_idxl);

  bool isLongest(int elem, int elemType, double len);

  /* LOCAL HELPERS FOR BULK ADAPTIVITY OPERATIONS */

  int lock_3D_region(int elemID, int elemType, int edgeID, double prio, RegionID *lockRegionID);
  void unlock_3D_region(RegionID lockRegionID);
  void unpend_3D_region(RegionID lockRegionID);
	
  /** Add a new element to the mesh. 
   * Update its connectivity
   * Return index of new element
   * */
  int add_element(int elemType,int nodesPerElem,int *conn,double sizing);
  
  /** Update the conn of an element*/
  void update_element_conn(int elemType,int elemID,int nodesPerElem,int *conn);
  
  /** Add a new node to the mesh
   * update its co-ordinates 
   * Return index of new node
   * */
  int add_node(int dim,double *coords);
  
  /** Update the co-ordinates of the given node */
  void update_node_coord(int nodeID,int dim,double *coords);
  
  void make_node_shared(int nodeID,int numSharedChunks,int *sharedChunks);
  
  int get_idxl_for_node(int nodeID, int partID);
  int get_node_from_idxl(int node_idxl, int partID);
  bool is_node_in_idxl(int node_idxl, int partID);

  /** Find all elements adjacent to an edge, for locking purposes */
  void get_elemsToLock(adaptAdj startElem, adaptAdj **elemsToLock, int edgeID, int *count);

  /** Perform all local mesh mods and updates for a local tet split */
  adaptAdj *local_split_3D(const adaptAdj elem, int n1, int n2, int n5);

  void local_update_asterisk_3D(int i, adaptAdj elem, int numElemPairs, 
				adaptAdj *elemPairs, int n1, 
				int n2, int n5);
  /** Perform local face adjacency updates associated with a split */
  void update_local_face_adj(const adaptAdj elem, const adaptAdj splitElem, int n1, int n2, int n5);
  /** Perform local edge adjacency updates associated with a split */
  void update_local_edge_adj(const adaptAdj elem, const adaptAdj splitElem, int n1, int n2, int n5);
  double length(int n1, int n2, int dim);
  double length(double *n1, double *n2, int dim);
  void dumpConn();
};

// GENERAL HELPER FUNCTIONS

/** Find the midpoint between two nodes; store in result. */
void midpoint(double *n1, double *n2, int dim, double *result);

int getRelNode(int nodeIdx, int *conn, int nodesPerElem);
void getRelNodes(int edgeID, int nodesPerElem, int *r1, int *r2);
int getEdgeID(int node1, int node2, int nodePerElem, int dim);
int getFaceID(int node1, int node2, int node3, int nodesPerElem);

/** Fill out the nodes and relative numberings for a tet */
void fillNodes(int *relNode, int *nodeIDs, int *conn);
void fillNodes(int *relNode, int n1, int n2, int *conn);

#endif
