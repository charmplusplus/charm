/*
Describes a viewpoint in 3d space: a projection matrix.
  (Copied out of Orion's Standard Library)

Orion Sky Lawlor, olawlor@acm.org, 2003/3/28
*/

#ifndef __UIUC_PPL_CHARM_VIEWPOINT_H
#define __UIUC_PPL_CHARM_VIEWPOINT_H

#ifndef CMK_USE_INTEL_SSE

#if defined (__SSE__)
#  define CMK_USE_INTEL_SSE 1
#endif

#endif

#if CMK_USE_INTEL_SSE
#include <xmmintrin.h>
#endif

#include "ckvector3d.h"
#ifdef __CHARMC__
#  include "pup.h"
#endif

/**
 * A start point and a direction
 */
class CkRay {
public:
	CkVector3d pos,dir;

	CkRay() {}
	CkRay(const CkVector3d &s,const CkVector3d &d)
		:pos(s), dir(d) {}

	CkVector3d at(double t) const {return pos+t*dir;}	

#ifdef __CK_PUP_H
	void pup(PUP::er &p) {p|pos; p|dir;}
#endif
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
	
#ifdef __CK_PUP_H
	void pup(PUP::er &p) {
		p(&data[0][0],16);
	}
#endif
};

/// Horrific hack: we don't have 2d vectors, so ignore z part of a 3d vector:
typedef CkVector3d CkVector2d;

#if CMK_USE_INTEL_SSE
/* sse utility routines */

/*
  Return the sum of all 4 floats in this SSE register.
  Computes the sum in all words, because it's just as cheap
  as computing it in just one.
*/
inline __m128 _ck_SSE_reduce(__m128 r) {
	/* r = a b c d
	   swapLo = b a d c
	   sumLo = a+b b+a c+d d+c
	   swapHi = c+d c+d a+b a+b
	   sum = 4 copies of a+b+d+c
	 */
		__m128 swapLo = _mm_shuffle_ps(r,r, _MM_SHUFFLE(2,3,0,1));
		__m128 sumLo = _mm_add_ps(r,swapLo);
		__m128 swapHi = _mm_shuffle_ps(sumLo,sumLo, _MM_SHUFFLE(1,1,3,3));
		__m128 sum = _mm_add_ps(sumLo,swapHi);
		return sum;
}

/* 
  Take the 4-float dot product of these two vectors, which
  must be 16-byte aligned. 
*/
inline __m128 _ck_SSE_dot(__m128 a,__m128 b) {
	return _ck_SSE_reduce(_mm_mul_ps(a,b));
}
#endif

/**
 * Describes a method for converting universe locations to screen pixels
 * and back again: a projection matrix.
 */
class CkViewpoint {
#if CMK_USE_INTEL_SSE
	float projX[4], projY[4], projZ[4], projW[4];
#endif
	CkVector3d E; //< Eye point (PHIGS Projection Reference Point)
	CkVector3d R; //< Projection plane origin (PHIGS View Reference Point)
	CkVector3d X,Y; //< Pixel-length axes of projection plane (Y is PHIGS View Up Vector)
	CkVector3d Z; //< Arbitrary-length normal to projection plane (PHIGS View Plane Normal)
	CkMatrix3d m; //< 4x4 projection matrix: universe to screen pixels
	int wid,ht; //< Width and height of view plane, in pixels
	
	bool isPerspective; //< If true, perspective projection is enabled.
	
	/// Fill our projection matrix m with values from E, R, X, Y, Z
	void buildM(void);
public:
	/// Build a camera at eye point E pointing toward R, with up vector Y.
	///   The up vector is not allowed to be parallel to E-R.
	/// This is an easy-to-use, but restricted (no off-axis) routine.
	/// It is normally followed by discretize or discretizeFlip.
	CkViewpoint(const CkVector3d &E_,const CkVector3d &R_,CkVector3d Y_=CkVector3d(0,1,1.0e-8));
	
	/// Build a camera at eye point E for view plane with origin R
	///  and X and Y as pixel sizes.  
	CkViewpoint(const CkVector3d &E_,const CkVector3d &R_,
		const CkVector3d &X_,const CkVector3d &Y_,int w,int h);
	
	/**
	  Build an orthogonal camera for a view plane with origin R
	   and X and Y as pixel sizes, and the given Z axis.
	  For a parallel-projection camera (yesThisIsPerspective==false) and
	    project(R_+x*X_+y*Y_+z*Z_) = (x,y,z)
	*/
	CkViewpoint(const CkVector3d &R_,
		const CkVector3d &X_,const CkVector3d &Y_,const CkVector3d &Z_,
		int w,int h,bool yesThisIsPerspective);
	
	/// Make this perspective camera orthogonal-- turn off perspective.
	void disablePerspective(void);
	
	/// Make this camera, fresh-built with the above constructor, have
	///  this X and Y resolution and horizontal full field-of-view (degrees).
	/// This routine rescales X and Y to have the appropriate length for
	///  the field of view, and shifts the projection origin by (-w/2,-h/2). 
	void discretize(int w,int h,double hFOV);
	
	/// Like discretize, but flips the Y axis (for typical raster viewing)
	void discretizeFlip(int w,int h,double hFOV);
	
	/// Flip the image's Y axis (for typical raster viewing)
	void flip(void);
	
	/// Extract a window with this width and height, with origin at this pixel
	void window(int w,int h, int x,int y);
	
	/// For use by pup:
	CkViewpoint() { wid=ht=-1; }
	
	/// Return the center of projection (eye point)
	inline const CkVector3d &getEye(void) const {return E;}
	/// Return true if this is an orthographic (perspective-free) camera.
	inline bool isOrthographic(void) const {return !isPerspective;}
	
	/// Return the projection plane origin (View Reference Point)
	inline const CkVector3d &getOrigin(void) const {return R;}
	/// Return the projection plane pixel-length X axis
	inline const CkVector3d &getX(void) const {return X;}
	/// Return the projection plane pixel-length Y axis (View Up Vector)
	inline const CkVector3d &getY(void) const {return Y;}
	/// Return the z-unit-length Z axis (from reference towards camera)
	inline const CkVector3d &getZ(void) const {return Z;}
	
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
	///  The projected screen point is (return.x,return.y);
	///
	///  return.z is 1.0/depth: +inf at the eye to 1 at the projection plane.
	///  This is the "perspective scale value"; the screen size 
	///  multiplier needed because of perspective.
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
	
	enum {nClip=4};
	
	/// Get our i'th clipping plane.
	///  0 and 1 are the left and right horizontal clip planes.
	///  2 and 3 are the top and bottom vertical clip planes.
	CkHalfspace3d getClip(int i) const; 
	
	/// Return true if any convex combination of these points is still offscreen.
	bool allOffscreen(int n,const CkVector3d *p) const {
		for (int c=0;c<nClip;c++) {
			CkHalfspace3d h=getClip(c);
			int i;
			for (i=0;i<n;i++)
				if (h.side(p[i])>=0)
					break; /* this point in-bounds */
			if (i==n) /* every point was outside halfspace h */
				return true;
		}
		return false; /* not all points outside the same halfspace */
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
#if CMK_USE_INTEL_SSE
	inline void project_noz(const float *in,int *x,int *y) const {
	/*
	   Here's what we're trying to do:
		float w=1.0f/(float)(
		  in[0]*projW[0]+in[1]*projW[1]+in[2]*projW[2]+in[3]*projW[3]
		);
		return CkVector2d(
		  w*(in[0]*projX[0]+in[1]*projX[1]+in[2]*projX[2]+in[3]*projX[3]),
		  w*(in[0]*projY[0]+in[1]*projY[1]+in[2]*projY[2]+in[3]*projY[3]),
		  0.0
		);
	*/
		__m128 inR=_mm_load_ps((float *)in);
		
		__m128 w=_mm_rcp_ss(_ck_SSE_dot(inR,_mm_load_ps((float *)projW)));
		
		*x=_mm_cvttss_si32(_mm_mul_ss(w,
			_ck_SSE_dot(inR,_mm_load_ps((float *)projX))
		));
		*y=_mm_cvttss_si32(_mm_mul_ss(w,
			_ck_SSE_dot(inR,_mm_load_ps((float *)projY))
		));
	}
#endif
	
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
		if (isPerspective)
			return CkRay(getEye(),viewplane(screen)-getEye());
		else /* orthogonal camera */
			return CkRay(viewplane(screen)-1000.0*Z, Z);
	}
	/// Get a universe-coords vector pointing to the camera from this point
	CkVector3d toCamera(const CkVector3d &pt) const {
		if (isPerspective)
			return getEye()-pt;
		else /* parallel projection */
			return Z;
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
	void makeOpenGL(double *dest,double z_near,double z_far) const;
	
#ifdef __CK_PUP_H
	void pup(PUP::er &p);
#endif
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
	//Rotate around our +Z axis by this differential amount.
	void rotate(double dz) {
		x()+=dz*y();
		y()-=dz*x();
		ortho(); normalize();
	}
};




#endif /*def(thisHeader)*/
