#ifndef __CHARM_FEM_ADAPT_H
#define __CHARM_FEM_ADAPT_H

#include "fem_impl.h"

class FEM_Adapt {
  FEM_Mesh *theMesh;
  // Helper methods
  /// Map a pair of element-local node numberings to an element-local edge 
  /// numbering
  /** Given two element-local node numberings (i.e. 0, 1, 2 for triangular 
      elements), calculate an element-local edge numbering (also 0, 1, or 2
      for triangular elements) **/
  int get_edge_index(int local_node1, int local_node2);
  /// Find an element-local node numbering for a chunk-local node
  /** Given a chunk-local element number e and a chunk-local node number n,
      determine the element-local node numbering for node n on element e **/
  int find_local_node_index(int e, int n);
  /// Check if e1 and e3 are on the same side of edge path (n1, n, n2)
  /** Makes use of ordering of nodes in e1 to check is e3 is on the same side
      of the path of edges (n1, n) and (n, n2) **/
  int check_orientation(int e1, int e3, int n, int n1, int n2);
  /// Build adjacency lists for vertex_split operation
  /** When splitting a node, the adjacency lists for the old and new
      nodes need to be reconstructed from scratch. In case of gap in
      mesh, traversal is attempted from both sides of the arc.  Arc
      goes around node n starting at node startNode and element
      startElem, and ends at stopElem.  Resulting counts for nodes and
      elements are stored in nn and ne respectively, and actual nodes
      and elements are stored in nodeList and elemList
      respectively **/
  void adj_traverse(int n, int startNode, int stopNode, int startElem, 
		    int stopElem, int *nn, int *ne, int *nodeList,
		    int *elemList);

 public:
  /// Initialize FEM_Adapt with a chunk of the mesh
  FEM_Adapt(FEM_Mesh *m) {
    theMesh = m;
  }
  /// Perform a Delaunay flip of edge (n1, n2)
  /** Perform a Delaunay flip of the edge (n1, n2) returning 1 if successful,
      0 if not.  The convexity of the quadrilateral formed by two faces
      incident to edge (n1, n2) is assumed. **/
  virtual int edge_flip(int n1, int n2);
  /// Helper method to int edge_flip(int n1, int n2)
  virtual int edge_flip(int e1, int n1, int n2);
  /// Helper method to int edge_flip(int e1, int n1, int n2)
  virtual int edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			     int e1_n2);
  
  /// Bisect edge (n1, n2) and the two adjacent elements
  /** Given edge e:(n1, n2), determing the two elements e1:(n1,n2,n3) and 
      e2:(n2,n1,n4) adjacent to e, and bisect e, e1 and e2 by adding a node 
      n5 to edge e. e1 becomes (n1,n5,n3) and we add e3:(n5,n2,n3), e2 becomes 
      (n5,n1,n4) and we add e4:(n2,n5,n4); returns new node n5 **/
  virtual int edge_bisect(int n1, int n2);
  /// Helper method to int edge_bisect(int n1, int n2);
  virtual int edge_bisect(int e1, int n1, int n2);
  /// Helper method to int edge_bisect(int e1, int n1, int n2);
  virtual int edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
			       int e1_n2);
  
  /// Contract edge (n1, n2) and the two adjacent elements
  /** Given and edge e:(n1, n2), determine the two adjacent elements
      e1:(n1,n2,n3) and e2:(n1,n2,n4). Contract edge e by collapsing node n2 
      to n1, removing edge e, node n2, elements e1 & e2; return 1 if 
      successful, 0 if not **/
  virtual int edge_contraction(int n1, int n2);
  virtual int edge_contraction(int e1, int n1, int n2);
  virtual int edge_contraction_help(int e1, int e2, int n1, int n2, int e1_n1,
				    int e1_n2);

  /// Split a node n into two nodes with an edge in between
  /** Given a node n and two adjacent nodes n1 and n2, split n into two nodes
      n and newNode such that the edges to the neighbors n1 and n2
      expand into two new elements (n, newNode, n1) and (newNode, n, n2);
      return the id of the newly created node newNode **/
  virtual int vertex_split(int n, int n1, int n2);
  virtual int vertex_split(int n, int n1, int n2, int e1, int e3);
  
  // SINCE THIS USES THE MARKED VERTEX BISECTION ALGORITHM, IT IS A SPECIAL
  // PURPOSE METHOD FOR CPSD.  THIS SHOULD MOVE UP INTO THE CPSD INTERFACE
  // AS SHOULD THE MARKING OPERATIONS IN FEM_Mesh.
  /// Perform a propagating bisection of element e1 according to marked nodes
  /** Given and element e1, determined n, e1's marked node, and bisect
      the edge opposite to n, propagating and bisecting the
      neighboring element in a similar fashion **/
  virtual void element_bisect(int e1);
};

#endif
