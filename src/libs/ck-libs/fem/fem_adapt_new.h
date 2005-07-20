// This will eventually take the place of the previous version when Nilesh and 
// Isaac's fem_mesh_modify module is complete.
#ifndef __CHARM_FEM_ADAPT_H
#define __CHARM_FEM_ADAPT_H

#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"
#include "fem_mesh_modify.h"

class FEM_Adapt {
  FEM_Mesh *theMesh;

  // Helper methods: see bottom of this file
  /// Check if e1 and e3 are on the same side of edge path (n1, n, n2)
  /** Makes use of ordering of nodes in e1 to check is e3 is on the same side
      of the path of edges (n1, n) and (n, n2) **/
  int check_orientation(int e1, int e3, int n, int n1, int n2);
 public:
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
  /// Extract adjacency data relative to edge [n1,n2]
  /** Extract elements adjacent to edge [n1,n2] along with element-local node
      numberings and nodes opposite input edge **/
  void findAdjData(int n1, int n2, int *e1, int *e2, int *e1n1, int *e1n2, 
		   int *e1n3, int *e2n1, int *e2n2, int *e2n3, int *n3, 
		   int *n4);

  /// Initialize FEM_Adapt with a chunk of the mesh
  FEM_Adapt(FEM_Mesh *m) { theMesh = m; }
  /// Perform a Delaunay flip of edge (n1, n2)
  /** Perform a Delaunay flip of edge (n1, n2) returning 1 if successful, 0 if 
      not (likely due to the edge being on a boundary).  The convexity of the 
      quadrilateral formed by two faces incident to edge (n1, n2) is assumed. 
      n1 and n2 are assumed to be local to this chunk. An adjacency test is
      performed on n1 and n2 by searching for an element with edge [n1,n2]. **/
  virtual int edge_flip(int n1, int n2);
  virtual int edge_flip_help(int e1, int e2, int n1, int n2, int e1_n1, 
			     int e1_n2, int e1_n3, int n3, int n4);
  
  /// Bisect edge (n1, n2) and the two adjacent elements
  /** Given edge e:(n1, n2), remove the two elements (n1,n2,n3) and (n2,n1,n4) 
      adjacent to e, and bisect e by adding node n5. Add elements (n1,n5,n3), 
      (n5,n2,n3), (n5,n1,n4) and (n2,n5,n4); returns new node n5. **/
  virtual int edge_bisect(int n1, int n2);
  virtual int edge_bisect_help(int e1, int e2, int n1, int n2, int e1_n1, 
			       int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
			       int e2_n3, int n3, int n4);

  /// Remove the degree 4 vertex n1 without modifying degree of adj n2
  /** Inverse of edge bisect, this removes a degree 4 vertex n1 and 2 of its
      adjacent elements.  n2 indicates that the two elements removed are
      adjacent to edge [n1,n2]. This could be performed with edge_contraction,
      but this is a simpler operation. **/
  virtual int vertex_remove(int n1, int n2);
  virtual int vertex_remove_help(int e1, int e2, int n1, int n2, int e1_n1, 
				 int e1_n2, int e1_n3, int e2_n1, int e2_n2, 
				 int e2_n3, int n3, int n4, int n5);

  /// Contract edge (n1, n2) and the two adjacent elements
  /** Given and edge e:(n1, n2), determine the two adjacent elements (n1,n2,n3)
      and (n1,n2,n4). Contract edge e by creating node n5, removing all 
      elements incident on n1 xor n2 and reinserting with incidence on n5, 
      removing the two elements (n1,n2,n3) and (n1,n2,n4) adjacent to e, and 
      finally removing nodes n1 and n2; return 1 if successful, 0 if not **/
  virtual int edge_contraction(int n1, int n2);
  virtual int edge_contraction_help(int e1, int e2, int n1, int n2, int e1_n1, 
				    int e1_n2, int e1_n3, int e2_n1, int e2_n2,
				    int e2_n3, int n3, int n4);

  /// Split a node n into two nodes with an edge in between
  /** Given a node n and two adjacent nodes n1 and n2, split n into two nodes
      n and np such that the edges to the neighbors n1 and n2
      expand into two new elements (n, np, n1) and (np, n, n2);
      return the id of the newly created node np **/
  virtual int vertex_split(int n, int n1, int n2);
  virtual int vertex_split(int n, int n1, int n2, int e1, int e3);
  /// Initiate instance of Longest Edge Bisection on an element
  /** Initiate instance of Longest Edge Bisection on element e.  Propagates
      throughout the mesh to maintain the requirement that only longest edges
      are bisected; returns 1 if successful, 0 if not **/
  virtual int refine_element_leb(int e);
};


#endif
