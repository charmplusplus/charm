/**
 * Mesh interfacing routines for the FEM framework.
 * 
 * Orion Sky Lawlor, olawlor@acm.org, 3/24/2003
 */
#include "tetmesh.h"
#include "fem.h"

/// Read this mesh from the FEM framework's mesh m
void readFEM(int m,TetMesh &t) {
	int nNode=FEM_Mesh_get_length(m,FEM_NODE);
	int nTet=FEM_Mesh_get_length(m,FEM_ELEM+0);
	t.allocate(nTet,nNode);
	FEM_Mesh_data(m,FEM_NODE,FEM_COORD,t.getPointArray(), 0,nNode, FEM_DOUBLE,3);
	FEM_Mesh_data(m,FEM_ELEM+0,FEM_CONN,t.getTetConn(), 0,nTet, FEM_INDEX_0,4);
}

/// Write this mesh to the FEM framework's mesh m
void writeFEM(int m,TetMesh &t) {
	int nNode=t.getPoints();
	int nTet=t.getTets();
	FEM_Mesh_data(m,FEM_NODE,FEM_COORD,t.getPointArray(), 0,nNode, FEM_DOUBLE,3);
	FEM_Mesh_data(m,FEM_ELEM+0,FEM_CONN,t.getTetConn(), 0,nTet, FEM_INDEX_0,4);
}

