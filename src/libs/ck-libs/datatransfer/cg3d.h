/*  
Volume calculations for complicated shapes, that
are the intersection of simpler shapes.

Orion Sky Lawlor, olawlor@acm.org, 2003/2/26
*/
#ifndef __UIUC_CHARM_CG3D_H
#define __UIUC_CHARM_CG3D_H

#include "ckvector3d.h"
#include "charm.h" /* for "ckabort" */

#define OSL_CG3D_DEBUG 0

namespace cg3d {

/// Small roundoff-control value:
extern double epsilon;

/**
 * A dynamic pointset, with each point marked with the
 * halfspaces it contains.
 */
class PointSet3d {
	int nPts; enum {maxPts=32, firstPt=1000000000};
	CkVector3d pts[maxPts];
	int saved_nPts;
	
	int nHalf; enum {maxHalfs=12, firstHalf=2000000000};
	CkHalfspace3d half[maxHalfs];
	int setPts; //Halfspace points to maintain halfset for
	
	/**
	 * A "halfset" is a bitvector per point that indicates
	 *   where the point belongs for each halfspace.
	 */
	typedef unsigned int halfset_t;
	halfset_t inHalf[maxHalfs]; 
	halfset_t outHalf[maxHalfs]; 
	void addHalf(const CkVector3d &pt,int p);
public:
	/// Create an empty pointset. 
	PointSet3d();
	
	/// Add a halfspace to our list
	int addHalfspace(const CkHalfspace3d &h);
	inline int getHalfpaces(void) const {return nHalf;}
	inline const CkHalfspace3d &getHalfspace(int f) const {
#if OSL_CG3D_DEBUG
		if ((f-firstHalf)<0 || (f-firstHalf)>=nHalf)
			CkAbort("PointSet3d::halfspace index out of range!");
#endif
		return half[f-firstHalf];
	}
	
	/// Add a point to our list
	int addPoint(const CkVector3d &p);
	inline int getPoints(void) const {return nPts;}
	inline const CkVector3d &getPoint(int p) const {
#if OSL_CG3D_DEBUG
		if ((p-firstPt)<0 || (p-firstPt)>=nPts)
			CkAbort("PointSet3d::point index out of range!");
#endif
		return pts[p-firstPt];
	}
	
	/// Return true if point p is completely inside halfspace h
	inline bool isInside(int p,int h) const 
	{
		if (p<setPts)
			return inHalf[h-firstHalf]&(1<<(p-firstPt));
		else
			return isInside(getPoint(p),h);
	}
	inline bool isInside(const CkVector3d &p,int h) const 
		{ return getHalfspace(h).side(p)>epsilon; }
	
	/// Return true if point p is completely outside halfspace h
	inline bool isOutside(int p,int h) const 
	{ 
		if (p<setPts)
			return outHalf[h-firstHalf]&(1<<(p-firstPt));
		else
			return isOutside(getPoint(p),h);
	}
	inline bool isOutside(const CkVector3d &p,int h) const 
		{ return getHalfspace(h).side(p)<-epsilon; }
	
	/// Cache halfspace insideness for existing points
	void calculateHalfspaces(void);
	
	/// Save the number of points (before adding temporary points)
	inline void pushPoints(void) { saved_nPts=nPts; }
	/// Restore to the last saved number of points (throwing away later points)
	inline void popPoints(void) { nPts=saved_nPts; }
};

/**
 * A convex planar polyhedron, with its vertices scattered in 3D.
 * This typically represents the face of some shape.
 *
 * The representation is a chain of boundary points, which are 
 * split and queried as each halfspace is intersected.
 */
class Planar3d {
	PointSet3d *ps;
	int nPts; enum {maxPts=4+2*6};//< Maximum number of boundary points=start+2*halfspace
	int pts[maxPts];//Circular list of point locations, in order.
public:
	Planar3d(PointSet3d *ps_);
	
	inline int getPoints(void) const {return nPts;}
	inline int getPointIndex(int p) const {return pts[p];}
	inline const CkVector3d &getPoint(int p) const 
		{return ps->getPoint(pts[p]);}
	
	/// Add a new point along our boundary.  Points must be presented in order.
	inline void addPoint(int ptIdx) {
		pts[nPts++]=ptIdx;
	}
	inline void addPoint(int pt0,int pt1,int pt2) {
		pts[nPts+0]=pt0;
		pts[nPts+1]=pt1;
		pts[nPts+2]=pt2;
		nPts+=3;
	}
	inline void addPoint(int pt0,int pt1,int pt2,int pt3) {
		pts[nPts+0]=pt0;
		pts[nPts+1]=pt1;
		pts[nPts+2]=pt2;
		pts[nPts+3]=pt3;
		nPts+=4;
	}
	
	/// Clip this shape to lie within this halfspace.
	///  Returns false if we are made empty by the intersection.
	bool addConstraint(int halfspace);
};


/**
 * A convex 3D shape, described either as a set of convex
 * planar faces or the intersection of a set of halfspaces.
 */
class Shape3d {
protected:
	PointSet3d *ps;
private:
	int nFaces; int nPoints;
	const int *halfspaces; //Pass to gs->getHalfspace
	const int *points; //Pass to gs->getPoint
public:
	Shape3d(PointSet3d *ps_,int nFaces_,int nPoints_,
		const int *halfspaces_,const int *points_)
		:ps(ps_), nFaces(nFaces_), nPoints(nPoints_),
		halfspaces(halfspaces_), points(points_) {}
	virtual ~Shape3d();
	
	inline PointSet3d *getSet(void) const {return ps;}
	
	inline int getFaces(void) const {return nFaces;}
	inline int getPoints(void) const {return nPoints;}
	
	/**
	 * Get this point on the boundary of the shape.
	 * The return value is suitable for passing to PointSet3d::getPoint.
	 */
	inline int getPointIndex(int p) const {return points[p];}
	inline const CkVector3d &getPoint(int p) const 
		{return ps->getPoint(getPointIndex(p));}
	
	/** 
	 * Extract the halfspace of face f of this shape.
	 * The halfspace normal should point into the shape's interior. 
	 * The return value is suitable for passing to PointSet3d::getHalfspace.
	 */
	inline int getHalfspaceIndex(int f) const {return halfspaces[f];}
	inline const CkHalfspace3d &getHalfspace(int f) const 
		{return ps->getHalfspace(getHalfspaceIndex(f));}
	
	/** 
	 * Extract the boundary of face f of this shape.
	 * Points in face should spiral right-handed around plane normal. 
	 */
	virtual void getFace(int f, Planar3d &face) const =0;
	

	/// Return true if this shape fully contains this point.
	/// Works by enumerating the shape's halfspaces.
	bool contains(const CkVector3d &pt) const;
};

/// Test this shape for validity.  Aborts if the shape doesn't
///  satisfy its invariants.
void testShape(const Shape3d &s);


/**
 * A 4-node tetrahedron.
 */
class Tet3d : public Shape3d {
	int p[4]; //Indices of oriented vertices of tetrahedron
	int h[4]; //Indices of oriented halfspaces
	void operator=(const Tet3d &t); //<- don't use this
public:
	Tet3d(PointSet3d *ps_,const CkVector3d &A_,const CkVector3d &B_,
		const CkVector3d &C_,const CkVector3d &D_);
	/// Copy constructor
	Tet3d(const Tet3d &t) 
		:Shape3d(t.ps,4,4,h,p)
	{
		for (int i=0;i<4;i++) { h[i]=t.h[i]; p[i]=t.p[i]; }
	}
	
	virtual void getFace(int f, Planar3d &face) const;
};

/** Send clipped segments of a face to here. */
class Planar3dDest {
public:
	virtual void addFace(const Planar3d &f,int src) =0;
};

/// Under OSL_CG3D_DEBUG, this exception is thrown by ~Volume3dDest:
class NonManifoldException {
public:
	double a; //Volume swept from a vertex
	double b; //Volume swept from an outside point
	NonManifoldException(double a_,double b_) :a(a_), b(b_) {}
};

/** Compute the volume of the shape bounded by these faces */
class Volume3dDest : public Planar3dDest {
	bool hasOrigin;
	CkVector3d origin;
	double volume;
#if OSL_CG3D_DEBUG
	Volume3dDest *subVolume; //Second origin, for manifold checking
#endif
public:
	Volume3dDest();
	/// Debugging-only constructor: specify sweep origin
	Volume3dDest(const CkVector3d &origin_);
#if OSL_CG3D_DEBUG
	~Volume3dDest(); //Check volume at second origin
#endif
	virtual void addFace(const Planar3d &f,int src);
	inline double getVolume(void) const { return volume; }
};


/**
 * Return the volume of the tetrahedron with these vertices.
 */
double tetVolume(const CkVector3d &A,const CkVector3d &B,
		const CkVector3d &C,const CkVector3d &D);

/**
 * Compute the surface of the intersection of these two shapes,
 * sending the resulting face fragments to faceDest.
 */
void intersect(PointSet3d *ps,const Shape3d &shape0, const Shape3d &shape1, 
	Planar3dDest &faceDest);
	
/**
 * Debugging version of "intersect".  Writes out a tecplot file
 *  if the intersection is non-manifold.
 */
double intersectDebug(PointSet3d *ps,const Tet3d &S,const Tet3d &D);

};


#endif
