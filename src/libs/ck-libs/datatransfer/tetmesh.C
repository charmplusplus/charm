/**
 * A very basic tetrahedral mesh.
 * 
 * Orion Sky Lawlor, olawlor@acm.org, 3/11/2003
 */
#include <stdio.h>
#include <stdlib.h>
#include "tetmesh.h"
#include "bbox.h"
#include "charm.h" /* for CkAbort */

#if OSL_TETMESH_DEBUG /* Slow bounds checks */
void TetMesh::ct(int t) const {
	if (t<0 || t>=tet.size())
		CkAbort("TetMesh::ct> Tet index out of bounds");
}
void TetMesh::cp(int p) const {
	if (p<0 || p>=pts.size())
		CkAbort("TetMesh::cp> Point index out of bounds");
	
}

#endif

/// Create a new empty mesh.
TetMesh::TetMesh() {
}
/// Create a new mesh with this many tets and points.
TetMesh::TetMesh(int nt,int np) {
	allocate(nt,np);
}
TetMesh::~TetMesh() {
}

/// Set the size of this mesh to be nt tets and np points.
///  Throws away the previous mesh.
void TetMesh::allocate(int nt,int np) {
	tet.resize(nt);
	pts.resize(np);
}

// declaring these inline confuses the Intel C++ 7.1 compiler...
CkVector3d *TetMesh::getPointArray(void) {
	return &(pts[0]);
}
const CkVector3d *TetMesh::getPointArray(void) const {
	return &(pts[0]);
}

/**
 * Return the volume of the tetrahedron with these vertices.
 */
double tetVolume(const CkVector3d &A,const CkVector3d &B,
		const CkVector3d &C,const CkVector3d &D) 
{
	const static double oneSixth=1.0/6.0;
	return oneSixth*(B-A).dot((D-A).cross(C-A));
}

// Compute the volume of cell s
double TetMesh::getTetVolume(int s) const {
	const int *sc=getTet(s);
	return fabs(tetVolume(getPoint(sc[0]),getPoint(sc[1]),
	                 getPoint(sc[2]),getPoint(sc[3])));
}

// Return the average edge length on this mesh.
double averageEdgeLength(const TetMesh &m) {
	double edgeSum=0;
	int nEdge=0;
	for (int t=0;t<m.getTets();t++) {
		const int *conn=m.getTet(t);
		typedef int edgeID[2];
		const int nID=6;
		const static edgeID edgeIDs[nID]={
			{0,1},{0,2},{0,3},{1,2},{2,3},{1,3}
		};
		for (int i=0;i<nID;i++)
			edgeSum+=m.getPoint(conn[edgeIDs[i][0]]).
			    dist(m.getPoint(conn[edgeIDs[i][1]]));
		nEdge+=nID;
	}
	return edgeSum/nEdge;
}

/// Print a debugging representation to stdout
void print(const TetMesh &tet) {
	printSize(tet);
	
	int maxT=20; //Maximum number of tets to print out
	int nTp=tet.getTets();  if (nTp>maxT) nTp=maxT;
	for (int t=0;t<nTp;t++) {
		const int *c=tet.getTet(t);
		printf(" tet %d: %d %d %d %d\n",t,
			c[0],c[1],c[2],c[3]);
	}
	if (nTp<tet.getTets()) printf("   <more tets>\n");
	
	int maxP=20; //Maximum number of points to print out
	int nPp=tet.getPoints();  if (nPp>maxP) nPp=maxP;
	for (int p=0;p<nPp;p++) {
		printf(" point %d: ",p);
		print(tet.getPoint(p));
		printf("\n");
	}
	if (nPp<tet.getPoints()) printf("   <more points>\n");
}

/// Print a debugging representation of the mesh size, to stdout
void printSize(const TetMesh &tet) {
	printf("Tetrahedral mesh: %d tets, %d points\n",
		tet.getTets(),tet.getPoints());
	bbox3d box;box.empty();
	for (int p=0;p<tet.getPoints();p++)
		box.add(tet.getPoint(p));
	printf("Coordinate range: ");
	// box.print();
	printf("\n");
}

/// Print a debugging representation of this tet to stdout
void printTet(const TetMesh &m,int t) {
	const int *c=m.getTet(t);
	printf(" tet %d: %d %d %d %d\n",t,
			c[0],c[1],c[2],c[3]);
	for (int i=0;i<4;i++) print(m.getPoint(c[i]));
	printf("\n");
}

void print(const CkVector3d &p) {
	printf(" (%.3f, %.3f, %.3f)",p.x,p.y,p.z);
}
