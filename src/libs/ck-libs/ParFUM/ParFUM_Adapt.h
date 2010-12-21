
/* This module contains the class which presents the 
 * edge_bisect, edge_contract, edge_flip functions.
 *
 * File: ParFUM_Adapt.h
 * Authors: Terry Wilmarth, Nilesh Choudhury
 */

#ifndef __ParFUM_Adapt_H
#define __ParFUM_Adapt_H

class femMeshModify;

///Provides primitive mesh modification functions
/** An adaptivity class which provides the primitive incremental mesh modification operations.
 * These operations are edge_bisect, edge_flip and edge_contract.
 */
class FEM_Adapt {
 protected:
  ///cross-pointer to theMesh object on this chunk
  FEM_Mesh *theMesh;
  ///cross-pointer to the femMeshModify object on this chunk
  femMeshModify *theMod;
  // Helper methods: see bottom of this file
  /// Check if e1 and e3 are on the same side of edge path (n1, n, n2)
  int check_orientation(int e1, int e3, int n, int n1, int n2);

 public:
  ///default constructor
  FEM_Adapt() {
    theMesh = NULL; theMod = NULL;
  }
  /// Initialize FEM_Adapt with a chunk of the mesh
  FEM_Adapt(FEM_Mesh *m, femMeshModify *fm) { 
    theMesh = m; theMod = fm; 
  }
  /// Initialize FEM_Adapt with the femMeshModify object
  FEM_Adapt(femMeshModify *fm) { theMesh = NULL; theMod = fm; }
  /// Initialize FEM_Adapt with the FEM_Mesh object
  void FEM_AdaptSetMesh(FEM_Mesh *m) {theMesh = m;}
  /// pup for this object 
  void pup(PUP::er &p) {
  }

  /// Map a pair of element-local node numberings to an element-local edge numbering
  int get_edge_index(int local_node1, int local_node2);
  /// Find an element-local node numbering for a chunk-local node
  int find_local_node_index(int e, int n);

  /// Extract adjacency data relative to edge [n1,n2]
  int findAdjData(int n1, int n2, int *e1, int *e2, int *e1n1, int *e1n2, 
		   int *e1n3, int *e2n1, int *e2n2, int *e2n3, int *n3, 
		   int *n4);
  /// Get the other node connectivity on this element
  int e2n_getNot(int e, int n1, int n2);
  /// Verifies if n is a node connectivity of this element
  int n2e_exists(int n, int e);
  /// Find the element with connectivity n1, n2, n3
  int findElementWithNodes(int n1, int n2, int n3);

  /// Return the shared index for this node on this chunk
  int getSharedNodeIdxl(int n, int chk);
  /// Return the ghost index for this node on this chunk
  int getGhostNodeIdxl(int n, int chk);
  /// Return the ghost index for this element on this chunk
  int getGhostElementIdxl(int e, int chk);

  /// Print all the adjacencies of these nodes and elements
  void printAdjacencies(int *nodes, int numNodes, int *elems, int numElems);

  /// Is Node 'n1' a fixed node (i.e. defines the shape)
  bool isFixedNode(int n1);
  /// Is node 'n1' a corner
  bool isCorner(int n1);
  /// Is the edge defined by (n1,n2) on the boundary of the mesh
  bool isEdgeBoundary(int n1, int n2);

  /// Helper function to perform a Delaunay flip of edge (n1, n2)
  /** Perform a Delaunay flip of edge (n1, n2) returning 1 if successful, 0 if 
      not (likely due to the edge being on a boundary).  The convexity of the 
      quadrilateral formed by two faces incident to edge (n1, n2) is verified. 
      n1 and n2 are assumed to be local/shared to this chunk. An adjacency test is
      performed on n1 and n2 by searching for an element with edge [n1,n2]. **/
  int edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			     int e1_n2, int e1_n3, int n3, int n4,int *locknodes);
  
  /// Helper function to Bisect edge (n1, n2) and the two adjacent elements
  /** Given edge e:(n1, n2), remove the two elements (n1,n2,n3) and (n2,n1,n4) 
      adjacent to e, and bisect e by adding node n5. Add elements (n1,n5,n3), 
      (n5,n2,n3), (n5,n1,n4) and (n2,n5,n4); returns new node n5. **/
  int edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
			       int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
			       int e2_n3, int n3, int n4);

  /// Helper function to remove the degree 4 vertex n1 without modifying degree of adj n2
  /** Inverse of edge bisect, this removes a degree 4 vertex n1 and 2 of its
      adjacent elements.  n2 indicates that the two elements removed are
      adjacent to edge [n1,n2]. This could be performed with edge_contraction,
      but this is a simpler operation.
      Warning: This function was never tested!! It definitely needs work! */
  int vertex_remove_help(int e1, int e2, int n1, int n2, int e1_n1, 
				 int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
				 int e2_n3, int n3, int n4, int n5);

  /// Split a node n into two nodes with an edge in between
  /** Given a node n and two adjacent nodes n1 and n2, split n into two nodes
      n and np such that the edges to the neighbors n1 and n2
      expand into two new elements (n, np, n1) and (np, n, n2);
      return the id of the newly created node 'np'.
      Warning: This function was never tested!! It definitely needs work! */
  int vertex_split(int n, int n1, int n2);
  /// Helper function to split a node n into two nodes with an edge in between
  int vertex_split_help(int n, int n1, int n2, int e1, int e3);
};


///Provides primitive mesh modification functions (involves atomic locking/unlocking)
/** An adaptivity class which provides the primitive incremental mesh modification operations.
 * These operations are edge_bisect, edge_flip and edge_contract.
 * It provides a lock for all the nodes involved in the operation, so
 * that it does not interfere with any other operation.
 */
class FEM_AdaptL : public FEM_Adapt {
 public:
  /// default constructor
  FEM_AdaptL() {
    theMesh = NULL; theMod = NULL;
  }
  /// Initialize FEM_Adapt with a chunk of the mesh
  FEM_AdaptL(FEM_Mesh *m, femMeshModify *fm) { theMesh = m; theMod = fm; }
  /// Initialize FEM_Adapt with the femMeshModify object
  FEM_AdaptL(femMeshModify *fm) { theMesh = NULL; theMod = fm; }
  /// Initialize FEM_Adapt with the FEM_Mesh object
  void FEM_AdaptLSetMesh(FEM_Mesh *m) {theMesh = m;}
  /// Pup code for this class
  void pup(PUP::er &p) {
  }

  /// Lock the following set of read and write nodes
  int lockNodes(int *gotlocks, int *lockrnodes, int numRNodes, int *lockwnodes, int numWNodes);
  /// Lock the following set of read and write nodes
  int unlockNodes(int *gotlocks, int *lockrnodes, int numRNodes, int *lockwnodes, int numWNodes);

  /// Perform a Delaunay flip of edge (n1, n2)
  int edge_flip(int n1, int n2);
  /// Bisect edge (n1, n2) and the two adjacent elements
  int edge_bisect(int n1, int n2);
  /// Remove the degree 4 vertex n1 without modifying degree of adj n2
  int vertex_remove(int n1, int n2);
  /// Contract edge (n1, n2) and the two adjacent elements
  /** Given and edge e:(n1, n2), determine the two adjacent elements (n1,n2,n3)
      and (n1,n2,n4). Contract edge e by creating node n5, removing all 
      elements incident on n1 xor n2 and reinserting with incidence on n5, 
      removing the two elements (n1,n2,n3) and (n1,n2,n4) adjacent to e, and 
      finally removing nodes n1 and n2; return 1 if successful, 0 if not **/
  int edge_contraction(int n1, int n2);
  /// Helper function for contract edge (n1, n2) and the two adjacent elements
  int edge_contraction_help(int *e1P, int *e2P, int n1, int n2, int e1_n1, 
				    int e1_n2, int e1_n3, int e2_n1, int e2_n2,
				    int e2_n3, int n3, int n4);

  /// Acquire an element in our ghost layer, turning it into a local element
  int eatIntoElement(int e, bool aggressive_node_removal=false);
  /// Test the adaptivity system to see if any nodes are locked
  void residualLockTest();
  /// Release all currently held locks on this partition
  void unlockAll();
  /// Test the mesh for corruption in connectivity/adjacency
  void structureTest();
};

#endif

