/**
 * Mesh interfacing routines for the FEM framework.
 * 
 * Orion Sky Lawlor, olawlor@acm.org, 3/24/2003
 */
#include "tetmesh.h"
#include "fem.h"
#include <assert.h>
 
/// Read this mesh from the FEM framework's mesh m
void readFEM(int m,TetMesh &t) {
	int nNode=FEM_Mesh_get_length(m,FEM_NODE);
	int ngNode=FEM_Mesh_get_length(m,FEM_NODE+FEM_GHOST);
	int nTet=FEM_Mesh_get_length(m,FEM_ELEM+0);
	int ngTet=FEM_Mesh_get_length(m,FEM_ELEM+0+FEM_GHOST);
	t.allocate(nTet+ngTet,nNode+ngNode);
	FEM_Mesh_data(m,FEM_NODE,FEM_COORD,t.getPointArray(), 0,nNode, FEM_DOUBLE,3);
	FEM_Mesh_data(m,FEM_NODE+FEM_GHOST,FEM_COORD,t.getPoint(nNode), 0,ngNode, FEM_DOUBLE,3);
	FEM_Mesh_data(m,FEM_ELEM+0,FEM_CONN,t.getTetConn(), 0,nTet, FEM_INDEX_0,4);
	FEM_Mesh_data(m,FEM_ELEM+0+FEM_GHOST,FEM_CONN,t.getTet(nTet), 0,ngTet, FEM_INDEX_0,4);
	t.nonGhostPt = nNode;
	t.nonGhostTet = nTet;
	int *tconn;
	for (int i=nTet; i<ngTet+nTet; i++) {
	  tconn = t.getTet(i);
	  for (int j=0; j<4; j++) {
	    if (tconn[j] < -1) tconn[j] = (tconn[j]*-1)-2+nNode;
	    assert(tconn[j] < nNode+ngNode);
	  }
	}
}

/// Write this mesh to the FEM framework's mesh m
void writeFEM(int m,TetMesh &t) {
	int nNode=t.getPoints();
	int nTet=t.getTets();
	FEM_Mesh_data(m,FEM_NODE,FEM_COORD,t.getPointArray(), 0,nNode, FEM_DOUBLE,3);
	FEM_Mesh_data(m,FEM_ELEM+0,FEM_CONN,t.getTetConn(), 0,nTet, FEM_INDEX_0,4);
}

