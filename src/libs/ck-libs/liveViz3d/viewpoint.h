/*
Describes a viewpoint in 3d space: a projection matrix.

Orion Sky Lawlor, olawlor@acm.org, 8/27/2002
*/

#ifndef __UIUC_PPL_CHARM_VIEWPOINT_H
#define __UIUC_PPL_CHARM_VIEWPOINT_H

#include "ckvector3d.h"

//A start point and a direction
class CkRay {
public:
	CkVector3d pos,dir;

	CkRay() {}
	CkRay(const CkVector3d &s,const CkVector3d &d)
		:pos(s), dir(d) {}
	
	void pup(PUP::er &p) {p|pos; p|dir;}
};
PUPmarshall(CkRay);

//Describes a method for converting universe locations to screen points:
// a projection matrix.
class CkViewpoint {
	/*
	FIXME: add perspective viewpoints, too.
	A perspective projection matrix converts universe coordinates to
	  screen pixels.  E.g., universe location (x,y,z) 
	  maps to the screen location (sx,sy) as:
	  
	  double m[4][4];
	  
	  double sw=x*m[3][0]+y*m[3][1]+z*m[3][2]+m[3][3];
	  double sx=(x*m[0][0]+y*m[0][1]+z*m[0][2]+m[0][3])/sw;
	  double sy=(x*m[1][0]+y*m[1][1]+z*m[1][2]+m[1][3])/sw;

	All the other methods (project, getRay, etc.) still make sense
	under a perpsective projection; it's just that the math is a lot
	hairy-er.
	*/
	
	//Orthographic projection matrix:
	CkVector3d axes[3]; //X, Y, and Z axis (pixel-length)
	CkVector3d origin; //Screen bottom-left
	int wid,ht; //Size (pixels) of corresponding display
	
	//Project v along this axis:
	inline double project(int axis,const CkVector3d &v) const {
		return v.dot(axes[axis]);
	}
	//Back-Project s along this axis
	// FIXME: only works if x, y, and z are orthogonal!
	inline CkVector3d backProject(int axis,const double s) const {
		return axes[axis]*(s/axes[axis].magSqr());
	}
	
public:
	//Simple orthogonal projection (ignore-Z):
	CkViewpoint(int wid_,int ht_,
		const CkVector3d &xAxis, //Screen's +x axis, world coords.
		const CkVector3d &yAxis, //Screen's +y axis, world coords.
		const CkVector3d &zAxis, //Screen's +z axis, world coords.
		const CkVector3d &origin_ //Screen origin (bottom-left)
	)
		:origin(origin_), wid(wid_), ht(ht_)
	{
		axes[0]=xAxis; axes[1]=yAxis; axes[2]=zAxis;
	}
	//Default copy constructor, assignment operator
	
	//Project this point onto the screen.
	CkVector3d project(const CkVector3d &univ) const {
		CkVector3d rel(univ-origin);
		double sx=project(0,rel);
		double sy=project(1,rel);
		double sz=project(2,rel);
		return CkVector3d(sx,sy,sz);
	}
	
	//Project this point onto the screen, ignoring the z axis.
	CkVector3d project_noz(const CkVector3d &univ) const {
		CkVector3d rel(univ-origin);
		double sx=project(0,rel);
		double sy=project(1,rel);
		return CkVector3d(sx,sy,0.0);
	}
	
	//Get the universe-coords view ray passing through this universe point
	CkRay getRay(const CkVector3d &univ) const {
		return CkRay(univ,axes[2]);
	}
	//Get the universe-coords view ray passing through this screen point
	CkRay getPixelRay(const CkVector3d &screen) const {
		CkVector3d univ(origin+screen.x*axes[0]+screen.y*axes[1]);
		return CkRay(univ,axes[2]);
	}
	
	//Return the i'th axis
	// FIXME: not sensible for perspective projection
	const CkVector3d &getAxis(int i) const {return axes[i];}
	const CkVector3d &getOrigin(void) const {return origin;}
	//Back-project this screen point to universe coordinates:
	CkVector3d backProject(const CkVector3d &screen) const {
		return origin+
			backProject(0,screen.x)+
			backProject(1,screen.y)+
			backProject(2,screen.z);
	}
	
	
	//Return true if this screen location is (entirely) in-bounds:
	bool isInbounds(const CkVector3d &screen) const {
		return (screen.x>=0) && (screen.y>=0) &&
			(screen.x<wid) && (screen.y<ht);
	}
	
	CkViewpoint() { wid=ht=-1; }
	void pup(PUP::er &p);
};
PUPmarshall(CkViewpoint);


//X, Y, and Z axes: a right-handed frame
class CkAxes3d {
	CkVector3d axes[3]; //X, Y, and Z axes (orthonormal set)
	CkVector3d pixSize; //Pixels sizes in X,Y, and Z
	
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
	
	//Project the pixel s back along axis i
	CkVector3d backProject(int i,double s) const {
		return axes[i]*(s*pixSize[i]);
	}
public:	
	CkAxes3d() {
		axes[0]=CkVector3d(1,0,0);
		axes[1]=CkVector3d(0,1,0);
		axes[2]=CkVector3d(0,0,1);
		pixSize=CkVector3d(1,1,1);
	}
	
	const CkVector3d &getPixelSize(void) const {return pixSize;}
	void setPixelSize(const CkVector3d &v) {pixSize=v;}
	
	//Get our x, y, and z axes (unit vectors):
	const CkVector3d &x(void) const {return axes[0];}
	const CkVector3d &y(void) const {return axes[1];}
	const CkVector3d &z(void) const {return axes[2];}
	
	//Push our x and y axes in the +z direction by these amounts.
	//  If dx and dy are small, they correspond to a right-handed
	//  rotation about the -Y and -X directions.
	void nudge(double dx,double dy) {
		x()+=dx*z();
		y()+=dy*z();
		ortho(); normalize();
	}
	
	//Return the vector that would produce screen pixel a:
	CkVector3d backProject(const CkVector3d &a) const {
		return backProject(0,a.x)+backProject(1,a.y)+backProject(2,a.z);
	}
	
	inline CkViewpoint makeView(int wid,int ht,const CkVector3d &origin) const
		{return CkViewpoint(wid,ht, x()/pixSize.x,y()/pixSize.y,z()/pixSize.z, origin);}
};




#endif /*def(thisHeader)*/
