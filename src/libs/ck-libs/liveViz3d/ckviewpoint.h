/*
Describes a viewpoint in 3d space: a projection matrix.
  (Copied out of Orion's Standard Library)

Orion Sky Lawlor, olawlor@acm.org, 2003/3/28
*/

#ifndef __UIUC_PPL_CHARM_VIEWPOINT_H
#define __UIUC_PPL_CHARM_VIEWPOINT_H

#include "ckvector3d.h"
#include "pup.h"

/**
 * A start point and a direction
 */
class CkRay {
public:
	CkVector3d pos,dir;

	CkRay() {}
	CkRay(const CkVector3d &s,const CkVector3d &d)
		:pos(s), dir(d) {}
	
	void pup(PUP::er &p) {p|pos; p|dir;}
};

/**
 * A 4x4 matrix.  This is only used for projection, below, so it's 
 * a trivial class.
 */
class CkMatrix3d {
	double data[4][4];
public:
	inline double &operator() (int r,int c) {return data[r][c];}
	inline const double &operator() (int r,int c) const {return data[r][c];}
	
	// Scale this row by this scalar
	inline void scaleRow(int r,double s) {
		for (int c=0;c<4;c++) data[r][c]*=s;
	}
	// Add this scaling of this row to this destination row
	inline void addRow(int rSrc,double s,int rDest) {
		for (int c=0;c<4;c++) data[rDest][c]+=s*data[rSrc][c];
	}
	
	void pup(PUP::er &p) {
		p(&data[0][0],16);
	}
};

/// Horrific hack: we don't have 2d vectors, so ignore z part of a 3d vector:
typedef CkVector3d CkVector2d;

/**
 * Describes a method for converting universe locations to screen pixels
 * and back again: a projection matrix.
 */
class CkViewpoint {
	CkVector3d E; //< Eye point (PHIGS Projection Reference Point)
	CkVector3d R; //< Projection plane origin (PHIGS View Reference Point)
	CkVector3d X,Y; //< Pixel-length axes of projection plane (Y is PHIGS View Up Vector)
	CkVector3d Z; //< Arbitrary-length normal to projection plane (PHIGS View Plane Normal)
	CkMatrix3d m; //< 4x4 projection matrix: universe to screen pixels
	int wid,ht; //< Width and height of view plane, in pixels
	
	/// Fill our projection matrix m with values from E, R, X, Y, Z
	///   Assert: X, Y, and Z are orthogonal
	void buildM(void);
public:
	/// Build a camera at eye point E pointing toward R, with up vector Y.
	///   The up vector is not allowed to be parallel to E-R.
	/// This is an easy-to-use, but restricted (no off-axis) routine.
	/// It is normally followed by discretize or discretizeFlip.
	CkViewpoint(const CkVector3d &E_,const CkVector3d &R_,CkVector3d Y_=CkVector3d(0,1,1.0e-8));
	
	/// Make this camera, fresh-built with the above constructor, have
	///  this X and Y resolution and horizontal full field-of-view (degrees).
	/// This routine rescales X and Y to have the appropriate length for
	///  the field of view, and shifts the projection origin by (-w/2,-h/2). 
	void discretize(int w,int h,double hFOV);
	
	/// Like discretize, but flips the Y axis (for typical raster viewing)
	void discretizeFlip(int w,int h,double hFOV);
	
	
	/// Build a camera at eye point E for view plane with origin R
	///  and X and Y as pixel sizes.  Note X and Y must be orthogonal.
	/// This is a difficult-to-use, but completely general routine.
	CkViewpoint(const CkVector3d &E_,const CkVector3d &R_,
		const CkVector3d &X_,const CkVector3d &Y_,int w,int h);
	
	/// For use by pup:
	CkViewpoint() { wid=ht=-1; }
	
	/// Return the center of projection (eye point)
	inline const CkVector3d &getEye(void) const {return E;}
	/// Return the projection plane origin (View Reference Point)
	inline const CkVector3d &getOrigin(void) const {return R;}
	/// Return the projection plane pixel-length X axis
	inline const CkVector3d &getX(void) const {return X;}
	/// Return the projection plane pixel-length Y axis (View Up Vector)
	inline const CkVector3d &getY(void) const {return Y;}
	
	/// Return the number of pixels in the X direction
	inline int getXsize(void) const { return wid; }
	inline int getWidth(void) const { return wid; }
	/// Return the number of pixels in the Y direction
	inline int getYsize(void) const { return ht; }
	inline int getHeight(void) const { return ht; }
	
	/// Return the 4x4 projection matrix.  This is a column-wise
	/// matrix, meaning the translation portion is the rightmost column.
	inline const CkMatrix3d &getMatrix(void) const {return m;}
	
	/// Project this point into the camera volume.
	///  The projected point is (return.x,return.y);
	///  return.z is 1.0/depth: +inf at the eye to 1 at the projection plane
	inline CkVector3d project(const CkVector3d &in) const {
		float w=1.0f/(float)(
		  m(3,0)*in.x+m(3,1)*in.y+m(3,2)*in.z+m(3,3)
		);
		return CkVector3d(
		  w*(m(0,0)*in.x+m(0,1)*in.y+m(0,2)*in.z+m(0,3)),
		  w*(m(1,0)*in.x+m(1,1)*in.y+m(1,2)*in.z+m(1,3)),
		  w*(m(2,0)*in.x+m(2,1)*in.y+m(2,2)*in.z+m(2,3))
		);
	}
	
	/// Project this point onto the screen, returning zero for the z axis.
	CkVector2d project_noz(const CkVector3d &in) const {
		float w=1.0f/(float)(
		  m(3,0)*in.x+m(3,1)*in.y+m(3,2)*in.z+m(3,3)
		);
		return CkVector2d(
		  w*(m(0,0)*in.x+m(0,1)*in.y+m(0,2)*in.z+m(0,3)),
		  w*(m(1,0)*in.x+m(1,1)*in.y+m(1,2)*in.z+m(1,3)),
		  0.0
		);
	}
	
	/// Backproject this view plane point into world coordinates
	inline CkVector3d viewplane(const CkVector2d &v) const {
		return R+v.x*X+v.y*Y;
	}
	/// Project, the back-project this universe point:
	inline CkVector3d projectViewplane(const CkVector3d &u) const {
		return viewplane(project_noz(u));
	}
	
	//Get the universe-coords view ray passing through this universe point
	CkRay getRay(const CkVector3d &univ) const {
		return CkRay(getEye(),univ-getEye());
	}
	//Get the universe-coords view ray passing through this screen point
	CkRay getPixelRay(const CkVector2d &screen) const {
		return CkRay(getEye(),viewplane(screen)-getEye());
	}
	
	//Return true if this screen location is (entirely) in-bounds:
	bool isInbounds(const CkVector2d &screen) const {
		return (screen.x>=0) && (screen.y>=0) &&
			(screen.x<wid) && (screen.y<ht);
	}
	//Clip this vector to lie totally onscreen
	void clip(CkVector2d &screen) const {
		if (screen.x<0) screen.x=0;
		if (screen.y<0) screen.y=0;
		if (screen.x>wid) screen.x=wid;
		if (screen.y>ht) screen.y=ht;
	}
	//Return a 16-entry OpenGL-compatible projection matrix for this viewpoint
	void makeOpenGL(double *dest,double near,double far) const;
	
	void pup(PUP::er &p);
};

/// X, Y, and Z axes: a right-handed frame, used for navigation
class CkAxes3d {
	CkVector3d axes[3]; //X, Y, and Z axes (orthonormal set)
	
	CkVector3d &x(void) {return axes[0];}
	CkVector3d &y(void) {return axes[1];}
	CkVector3d &z(void) {return axes[2];}

	//Make our axes orthogonal again by forcing z from x and y
	void ortho(void) {
		z()=x().cross(y());
		y()=z().cross(x());
	}
	//Make our axes all unit length
	void normalize(void) {
		for (int i=0;i<3;i++)
			axes[i]=axes[i].dir();
	}
public:	
	CkAxes3d() {
		axes[0]=CkVector3d(1,0,0);
		axes[1]=CkVector3d(0,1,0);
		axes[2]=CkVector3d(0,0,1);
	}
	
	//Get our x, y, and z axes (unit vectors):
	const CkVector3d &getX(void) const {return axes[0];}
	const CkVector3d &getY(void) const {return axes[1];}
	const CkVector3d &getZ(void) const {return axes[2];}
	
	//Push our x and y axes in the +z direction by these amounts.
	//  If dx and dy are small, they correspond to a right-handed
	//  rotation about the -Y and -X directions.
	void nudge(double dx,double dy) {
		x()+=dx*z();
		y()+=dy*z();
		ortho(); normalize();
	}
};




#endif /*def(thisHeader)*/
