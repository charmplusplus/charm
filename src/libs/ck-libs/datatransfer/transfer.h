/**
Mesh data transfer routines.

Originally written by Mike Campbell, 2003.
Interface modified for Charm integration by Orion Lawlor, 2004.
*/
#ifndef CHARM_TRANSFER_H
#define CHARM_TRANSFER_H

#include "GenericElement.h"

/// Compute the volume shared by elements A and B, which must be tets.
double getSharedVolumeTets(const ConcreteElement &A,const ConcreteElement &B); 


#include "tetmesh.h"

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


#endif
