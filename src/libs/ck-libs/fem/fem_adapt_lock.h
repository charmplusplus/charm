/* File: fem_adapt_lock.h
 * Authors: Nilesh Choudhury, Terry Wilmarth
 *
 */

#ifndef __CHARM_FEM_ADAPT_LOCK_H
#define __CHARM_FEM_ADAPT_LOCK_H

#include "charm-api.h"
#include "ckvector3d.h"
#include "fem.h"
#include "fem_mesh.h"
#include "fem_adapt_new.h"

class femMeshModify;

class FEM_AdaptL : public FEM_Adapt {
 public:
  FEM_AdaptL() {
    theMesh = NULL; theMod = NULL;
  }
  
  /// Initialize FEM_Adapt with a chunk of the mesh
  FEM_AdaptL(FEM_Mesh *m, femMeshModify *fm) { theMesh = m; theMod = fm; }

  int lockNodes(int *, int *, int, int *, int);
  int unlockNodes(int *, int *, int, int *, int);
  int edge_flip(int n1, int n2);
  int edge_bisect(int n1, int n2);
  int vertex_remove(int n1, int n2);
  int edge_contraction(int n1, int n2);
  int edge_contraction_help(int *e1P, int *e2P, int n1, int n2, int e1_n1, 
				    int e1_n2, int e1_n3, int e2_n1, int e2_n2,
				    int e2_n3, int n3, int n4);
};

#endif
