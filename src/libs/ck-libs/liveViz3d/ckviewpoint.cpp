/*
Describes a viewpoint in 3d space: a projection matrix.
  (Copied out of Orion's Standard Library)

Orion Sky Lawlor, olawlor@acm.org, 2003/3/28
*/
#include <math.h> /* for M_PI*/
#include "converse.h"
#include "ckviewpoint.h"
#ifndef M_PI
#  define M_PI 3.14159265358979323
#endif


inline bool orthogonal(const CkVector3d &a,const CkVector3d &b) {
	return fabs(a.dot(b))<1.0e-4;
}

/// Fill our projection matrix m with values from E, R, X, Y, Z
///   Assert: X, Y, and Z are orthogonal
void CkViewpoint::buildM(void) {
	// if (!(orthogonal(X,Y) && orthogonal(Y,Z) && orthogonal(X,Z)))
	//	osl::bad("Camera axes are non-orthogonal");
	
	/*
	  The projection matrix derivation begins by postulating
	  a universe point P, which we need to project into the view plane.
	  The view plane projection, S, satisfies S=E+t*(P-E) for 
	  some parameter value t, and also Z.dot(S-R)=0.
	  Solving this and taking screen_x=sX.dot(S-R), screen_y=sY.dot(S-R),
	  and screen_z=Z.dot(R-E)/Z.dot(P-E) leads to our matrix.
	 */
	// Scale X and Y so screen pixels==sX.dot(S)
	CkVector3d sX=X/X.magSqr();
	CkVector3d sY=Y/Y.magSqr();
	
	// Compute skew factors and skewed axes
	double skew_x=sX.dot(R-E), skew_y=sY.dot(R-E), skew_z=Z.dot(R-E);
	CkVector3d gX=skew_x*Z-skew_z*sX;
	CkVector3d gY=skew_y*Z-skew_z*sY;
	
	// Assign values to the matrix
	m(0,0)=gX.x; m(0,1)=gX.y; m(0,2)=gX.z; m(0,3)=-gX.dot(E); 
	m(1,0)=gY.x; m(1,1)=gY.y; m(1,2)=gY.z; m(1,3)=-gY.dot(E); 
	m(2,0)=0;    m(2,1)=0;    m(2,2)=0;    m(2,3)=-skew_z;
	m(3,0)=-Z.x; m(3,1)=-Z.y; m(3,2)=-Z.z; m(3,3)=Z.dot(E); 
}

/// Build a camera at eye point E pointing toward R, with up vector Y.
///   The up vector is not allowed to be parallel to E-R.
CkViewpoint::CkViewpoint(const CkVector3d &E_,const CkVector3d &R_,CkVector3d Y_)
	:E(E_), R(R_), Y(Y_), wid(-1), ht(-1)
{
	Z=(E-R).dir(); //Make view plane orthogonal to eye-origin line
	X=Y.cross(Z).dir();
	Y=Z.cross(X).dir();
	// assert: X, Y, and Z are orthogonal
	buildM();
}

/// Make this camera, fresh-built with the above constructor, have
///  this X and Y resolution and horizontal full field-of-view (degrees).
/// This routine rescales X and Y to have the appropriate length for
///  the field of view, and shifts the projection origin by (-wid/2,-ht/2). 
void CkViewpoint::discretize(int w,int h,double hFOV) {
	wid=w; ht=h;
	double pixSize=E.dist(R)*tan(0.5*(M_PI/180.0)*hFOV)*2.0/w;
	X*=pixSize;
	Y*=pixSize;
	R-=X*(0.5*w)+Y*(0.5*h);
	buildM();
}

/// Like discretize, but flips the Y axis (for typical raster viewing)
void CkViewpoint::discretizeFlip(int w,int h,double hFOV) {
	discretize(w,h,hFOV);
	R+=Y*h;
	Y*=-1;
	buildM();
}


/// Build a camera at eye point E for view plane with origin R
///  and X and Y as pixel sizes.  Note X and Y must be orthogonal.
CkViewpoint::CkViewpoint(const CkVector3d &E_,const CkVector3d &R_,
	const CkVector3d &X_,const CkVector3d &Y_,int w,int h)
	:E(E_), R(R_), X(X_), Y(Y_), wid(w), ht(h)
{
	if (!orthogonal(X,Y))
		CmiAbort("Non-orthogonal X and Y passed to Camera::Camera");
	Z=X.cross(Y).dir();
	buildM();
}


void CkViewpoint::pup(PUP::er &p) {
	p.comment("CkViewpoint {");
	p.comment("axes");
	p|X; p|Y; p|Z;
	p.comment("origin");
	p|R;
	p.comment("eye");
	p|E;
	p.comment("width and height");
	p|wid; p|ht;
	p.comment("} CkViewpoint");
	if (p.isUnpacking()) buildM();
}


//Return an OpenGL-compatible projection matrix for this viewpoint:
void CkViewpoint::makeOpenGL(double *dest,double z_near,double z_far) const {
	CkMatrix3d g=m;
	/// Step 1: convert X and Y from outputting pixels to outputting [0,2]:
	g.scaleRow(0,2.0/wid);
	g.scaleRow(1,2.0/ht);
	/// Step 2: center X and Y on [-1,1], by shifting post-divide output:
	g.addRow(3,-1.0,0);
	g.addRow(3,-1.0,1);
	
	/// Step 3: map output Z from [-z_far,-z_near] to [-1,1]
	double a=-2.0*z_near*z_far/(z_far-z_near); // 1/z term
	double b=(z_near+z_far)/(z_far-z_near); // constant term
	g(2,2)=0; g(2,3)=a;
	g.addRow(3,b,2);
	
	/// Finally, copy 
	for (int r=0;r<4;r++)
	for (int c=0;c<4;c++) 
		dest[c*4+r]=g(r,c);
}


