/**
 * A tetrahedral mesh.
 * 
 * Orion Sky Lawlor, olawlor@acm.org, 3/12/2003
 */
#ifndef __UIUC_CHARM_TETMESH_H
#define __UIUC_CHARM_TETMESH_H

#include <stdio.h> // For FILE *
#include "ckvector3d.h"

#define OSL_TETMESH_DEBUG 1

/**
 * A 3d tetrahedral mesh.  Contains the connectivity only--no data.
 */
class TetMesh {
	int nTet; //< Number of tets in the mesh.
	int *conn; //< Connectivity: 4 x nTet 0-based node indices.
	int nPts; //< Number of points (vertices, nodes) in the mesh.
	CkVector3d *pts; //< nPts 3d node locations.
	
	///Check these indices for in-range
#if OSL_TETMESH_DEBUG /* Slow bounds checks */
	void ct(int t) const;
	void cp(int p) const;
#else /* Fast unchecked version for production code */
	inline void ct(int t) const {}
	inline void cp(int p) const {}
#endif
public:
	/// Create a new empty mesh.
	TetMesh();
	/// Create a new mesh with this many tets and points.
	TetMesh(int nt,int np);
	virtual ~TetMesh();
	
	/// Set the size of this mesh to be nt tets and np points.
	///  Throws away the previous mesh.
	virtual void allocate(int nt,int np);
	
	/// Return the number of tets in the mesh
	inline int getTets(void) const {return nTet;}
	/// Return the t'th tetrahedra's 0-based node indices
	inline int *getTet(int t) {ct(t); return &conn[4*t];}
	inline const int *getTet(int t) const {ct(t); return &conn[4*t];}
	inline int *getTetConn(void) {return conn;}
	inline const int *getTetConn(void) const {return conn;}
	double getTetVolume(int t) const;
	
	/// Return the number of points (vertices, nodes) in the mesh
	inline int getPoints(void) const {return nPts;}
	/// Return the p'th vertex (0..getPoints()-1)
	inline CkVector3d &getPoint(int p) {cp(p); return pts[p];}
	inline const CkVector3d &getPoint(int p) const {cp(p); return pts[p];}
	inline CkVector3d *getPointArray(void) {return pts;}
	inline const CkVector3d *getPointArray(void) const {return pts;}
	
private:
	void deallocate(void);
	void justAllocate(int nt,int np);
};

/// Print a debugging representation of this mesh, to stdout.
void print(const TetMesh &t);

/// Print a debugging representation of this mesh's size, to stdout.
void printSize(const TetMesh &t);

/// Print a debugging representation of this tet to stdout
void printTet(const TetMesh &m,int t);

/// Print a debugging point
void print(const CkVector3d &p);

/// Read a TetMesh (ghs3d) ".noboite" mesh description file.
/// Aborts on errors.
void readNoboite(FILE *f,TetMesh &t);

/// Read this mesh from the FEM framework's mesh m
void readFEM(int m,TetMesh &t);

/// Write this mesh to the FEM framework's mesh m
void writeFEM(int m,TetMesh &t);

namespace cg3d { class Planar3dDest; };

/**
 * Compute the volume of the intersection of these two cells.
 */
double getSharedVolume(int s,const TetMesh &srcMesh,
	int d,const TetMesh &destMesh,cg3d::Planar3dDest *dest=NULL);

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
 * Return the average edge length on this mesh.
 */
double averageEdgeLength(const TetMesh &m);

#endif
