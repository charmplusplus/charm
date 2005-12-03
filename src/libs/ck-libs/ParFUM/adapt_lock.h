/* File: ParFUM_adapt_lock.h
 * Authors: Nilesh Choudhury, Terry Wilmarth
 *
 */

#ifndef __CHARM_ParFUM_adapt_lock_h
#define __CHARM_ParFUM_adapt_lock_h

#include "charm-api.h"
#include "ckvector3d.h"
#include "ParFUM.h"
#include "mesh.h"
#include "ParFUM_adapt.h"

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
