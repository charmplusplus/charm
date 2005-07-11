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
};

// Helpers
int FEM_Adapt::get_edge_index(int local_node1, int local_node2) 
{
  int sum = local_node1 + local_node2;
  CkAssert(local_node1 != local_node2);
  if (sum == 1) return 0;
  else if (sum == 3) return 1;
  else if (sum == 2) return 2;
  else {
    CkPrintf("ERROR: local node pair is strange: [%d,%d]\n", local_node1,
	    local_node2);
    CkAbort("ERROR: local node pair is strange\n");
    return -1;
  }
}

int FEM_Adapt::find_local_node_index(int e, int n) {
  int result = theMesh->e2n_getIndex(e, n);
  if (result < 0) {
    CkPrintf("ERROR: node %d not found on element %d\n", n, e);
    CkAbort("ERROR: node not found\n");
  }
  return result;
}

int FEM_Adapt::check_orientation(int e1, int e3, int n, int n1, int n2)
{
  int e1_n = find_local_node_index(e1, n);
  int e1_n1 = find_local_node_index(e1, n1);
  int e3_n = find_local_node_index(e3, n);
  int e3_n2 = find_local_node_index(e3, n2);
  
  if (((e1_n1 == (e1_n+1)%3) && (e3_n == (e3_n2+1)%3)) ||
      ((e1_n == (e1_n1+1)%3) && (e3_n2 == (e3_n+1)%3)))
    return 1;
  else return 0;
}

void FEM_Adapt::findAdjData(int n1, int n2, int *e1, int *e2, int *e1n1, 
			    int *e1n2, int *e1n3, int *e2n1, int *e2n2, 
			    int *e2n3, int *n3, int *n4)
{
  // Set some default values in case e1 is not there
  (*e1n1) = (*e1n2) = (*e1n3) = (*n3) = -1;
  (*e1) = theMesh->getElementOnEdge(n1, n2); // assumed to return local element
  if ((*e1) == -1) return;
  (*e1n1) = find_local_node_index((*e1), n1);
  (*e1n2) = find_local_node_index((*e1), n2);
  (*e1n3) = 3 - (*e1n1) - (*e1n2);
  (*n3) = theMesh->e2n_getNode((*e1), (*e1n3));
  (*e2) = theMesh->e2e_getNbr((*e1), get_edge_index((*e1n1), (*e1n2)));
  // Set some default values in case e2 is not there
  (*e2n1) = (*e2n2) = (*e2n3) = (*n4) = -1;
  if ((*e2) != -1) { // e2 exists
    (*e2n1) = find_local_node_index((*e2), n1);
    (*e2n2) = find_local_node_index((*e2), n2);
    (*e2n3) = 3 - (*e2n1) - (*e2n2);
    if ((*e2) > -1) { // if e2 is a ghost, there is no e2n data
      (*n4) = theMesh->e2n_getNode((*e2), (*e2n3));
    }
  }
}

#endif
