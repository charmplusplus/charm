/**
Mesh data transfer routines.

Originally written by Mike Campbell, 2003.
Interface modified for Charm integration by Orion Lawlor, 2004.
*/
#ifndef CHARM_TRANSFER_H
#define CHARM_TRANSFER_H

#include "GenericElement.h"

#include "tetmesh.h"
#include "prismMesh.h"
#include "triSurfMesh.h"

/// Compute the volume shared by elements A and B, which must be tets.
double getSharedVolumeTets(const ConcreteElement &A,const ConcreteElement &B); 

/// get the overlap area between A, a prism, and B, a 3D triangle.
double getSharedArea(const ConcreteElement &A,const ConcreteElement &B); 

/// Compute the volume shared by cell s of srcMesh
///   and cell d of destMesh.
double getSharedVolume(int s,const TetMesh &srcMesh,
	int d,const TetMesh &destMesh);

/**
 * Conservatively, accurately transfer 
 *   srcVals, tet-centered values on srcMesh
 * to
 *   destVals, tet-centered values on destMesh
 * WARNING: uses O(srcMesh * destMesh) time!
 */
void transferCells(int valsPerTet,
	double *srcVals,const TetMesh &srcMesh,
	double *destVals,const TetMesh &destMesh);
	
/**
 Provides ConcreteElement interface for one element of the TetMesh class.
*/
class TetMeshElement : public ConcreteElement {
	int s; const TetMesh &srcMesh;
	const int *conn;
public:
	TetMeshElement(int s_,const TetMesh &srcMesh_)
		:s(s_), srcMesh(srcMesh_) { conn=srcMesh.getTet(s); }
	
	/** Return the location of the i'th node of this element. */
	virtual CPoint getNodeLocation(int i) const {
		return srcMesh.getPoint(conn[i]);
	}
};

/**
 Provides ConcreteElement interface for one element of the TriangularSurfaceMesh class.
*/
class Triangle3DElement : public ConcreteElement {
	int s; const TriangleSurfaceMesh &srcMesh;
	const int *conn;
public:
	Triangle3DElement(int s_,const TriangleSurfaceMesh &srcMesh_)
		:s(s_), srcMesh(srcMesh_) { conn=srcMesh.getTriangle(s); }
	
	/** Return the location of the i'th node of this element. */
	virtual CPoint getNodeLocation(int i) const {
		return srcMesh.getPoint(conn[i]);
	}
};

#endif
