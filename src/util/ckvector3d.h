/*
This file provides various utility routines for easily
manipulating 3-D vectors-- included are arithmetic,
dot/cross product, magnitude and normalization terms. 
Most routines are provided right in the header file (for inlining).

Orion Sky Lawlor, olawlor@acm.org, 11/3/1999
*/

#ifndef __UIUC_PPL_CHARM_VECTOR_3D_H
#define __UIUC_PPL_CHARM_VECTOR_3D_H

#include "pup.h"

#include <math.h>

//MS Visual C++ defines max/min as a (cursed) macro
#ifdef max
#  undef max
#  undef min
#endif


//CkVector3d is a cartesian vector in 3-space-- an x, y, and z.
// For cross products, the space is assumed to be right-handed (x cross y = +z)
template <class real>
class CkVector3dT {
	typedef CkVector3dT<real> vec;
public:
	real x,y,z;
	CkVector3dT(void) {}//Default consructor
	//Simple 1-value constructors
	explicit CkVector3dT(int init) {x=y=z=(real)init;}
	explicit CkVector3dT(float init) {x=y=z=(real)init;}
	explicit CkVector3dT(double init) {x=y=z=(real)init;}
	//3-value constructor
	CkVector3dT(const real Nx,const real Ny,const real Nz) {x=Nx;y=Ny;z=Nz;}
	//real array constructor
	CkVector3dT(const real *arr) {x=arr[0];y=arr[1];z=arr[2];}

	//Constructors from other types of CkVector:
	CkVector3dT(const CkVector3dT<float> &src) 
	  {x=(real)src.x; y=(real)src.y; z=(real)src.z;}
	CkVector3dT(const CkVector3dT<double> &src) 
	  {x=(real)src.x; y=(real)src.y; z=(real)src.z;}
	CkVector3dT(const CkVector3dT<int> &src) 
	  {x=(real)src.x; y=(real)src.y; z=(real)src.z;}

	//Copy constructor & assignment operator by default
	
	//This lets you typecast a vector to a real array
	operator real *() {return (real *)&x;}
	operator const real *() const {return (const real *)&x;}

//Basic mathematical operators	
	int operator==(const vec &b) const {return (x==b.x)&&(y==b.y)&&(z==b.z);}
	int operator!=(const vec &b) const {return (x!=b.x)||(y!=b.y)||(z!=b.z);}
	vec operator+(const vec &b) const {return vec(x+b.x,y+b.y,z+b.z);}
	vec operator-(const vec &b) const {return vec(x-b.x,y-b.y,z-b.z);}
	vec operator*(const real scale) const 
		{return vec(x*scale,y*scale,z*scale);}
	friend vec operator*(const real scale,const vec &v)
		{return vec(v.x*scale,v.y*scale,v.z*scale);}
	vec operator/(const real &div) const
		{real scale=1.0/div;return vec(x*scale,y*scale,z*scale);}
	vec operator-(void) const {return vec(-x,-y,-z);}
	void operator+=(const vec &b) {x+=b.x;y+=b.y;z+=b.z;}
	void operator-=(const vec &b) {x-=b.x;y-=b.y;z-=b.z;}
	void operator*=(const real scale) {x*=scale;y*=scale;z*=scale;}
	void operator/=(const real div) {real scale=1.0/div;x*=scale;y*=scale;z*=scale;}

//Vector-specific operations
	//Return the square of the magnitude of this vector
	real magSqr(void) const {return x*x+y*y+z*z;}
	//Return the magnitude (length) of this vector
	real mag(void) const {return sqrt(magSqr());}
	
	//Return the square of the distance to the vector b
	real distSqr(const vec &b) const 
		{return (x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z);}
	//Return the distance to the vector b
	real dist(const vec &b) const {return sqrt(distSqr(b));}
	
	//Return the dot product of this vector and b
	real dot(const vec &b) const {return x*b.x+y*b.y+z*b.z;}
	//Return the cosine of the angle between this vector and b
	real cosAng(const vec &b) const {return dot(b)/(mag()*b.mag());}
	
	//Return the "direction" (unit vector) of this vector
	vec dir(void) const {return (*this)/mag();}
	//Return the right-handed cross product of this vector and b
	vec cross(const vec &b) const {
		return vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
	
	//Return the largest coordinate in this vector
	real max(void) {
		real big=x;
		if (big<y) big=y;
		if (big<z) big=z;
		return big;
	}
	//Make each of this vector's coordinates at least as big
	// as the given vector's coordinates.
	void enlarge(const vec &by) {
		if (x<by.x) x=by.x;
		if (y<by.y) y=by.y;
		if (z<by.z) z=by.z;     
	}
	
#ifdef __CK_PUP_H
	void pup(PUP::er &p) {p|x;p|y;p|z;}
#endif
};

typedef CkVector3dT<double> CkVector3d;
typedef CkVector3dT<float> CkVector3f;
typedef CkVector3dT<int> CkVector3i;


//An axis-aligned 3D bounding box
class CkBbox3d {
public:
	CkVector3d min,max;
	CkBbox3d() {empty();}
	void empty(void) {min.x=min.y=min.z=1e30;max.x=max.y=max.z=-1e30;}
	bool isEmpty(void) const {return min.x>max.x;}
	void add(const CkVector3d &p) {
		if (min.x>p.x) min.x=p.x;
		if (min.y>p.y) min.y=p.y;
		if (min.z>p.z) min.z=p.z;
		if (max.x<p.x) max.x=p.x;
		if (max.y<p.y) max.y=p.y;
		if (max.z<p.z) max.z=p.z;
	}
	void add(const CkBbox3d &b) {
		add(b.min); add(b.max);
	}
#ifdef __CK_PUP_H
	void pup(PUP::er &p) {p|min;p|max;}
#endif
};

//A CkHalfspace3d is the portion of a 3d plane lying on
// one side of the plane (p1,p2,p3).
class CkHalfspace3d {
public:
	// n dot p+d==0 on plane point p
	CkVector3d n;//Plane normal
	double d;
	
	typedef const CkVector3d cv;
	CkHalfspace3d() {}
	CkHalfspace3d(cv &p1,cv &p2,cv &p3) {init(p1,p2,p3);}
	CkHalfspace3d(cv &p1,cv &p2,cv &p3,cv &in) {initCheck(p1,p2,p3,in);}
	//Norm points into the halfspace; p0 is on the line
	CkHalfspace3d(cv &norm,cv &p0) {n=norm;d=-n.dot(p0);}

	//Set this halfspace to (p1,p2,p3).
	// inside points are on the right-handed thumb side of p1,p2,p3
	void init(cv &p1,cv &p2,cv &p3) {
		n=(p2-p1).cross(p3-p1);
		d=-n.dot(p1);
	}
	
	//Set this halfspace to (p1,p2,p3) with in inside.
	void initCheck(cv &p1,cv &p2,cv &p3,cv &in)
	{ init(p1,p2,p3); if (side(in)<0) {n=-n;d=-d;} }
	
	
	//Returns + if inside halfspace, - if outside (and 0 on line).
	double side(cv &pt) const
	{return n.dot(pt)+d;}
	
	//Return a value t such that pos+t*dir lies on our plane.
	double intersect(cv &pos,cv &dir) const
		{return -(d+n.dot(pos))/n.dot(dir);}
	
	/*Returns the point that lies on our plane and 
	  the line starting at start and going in dir.*/
	CkVector3d intersectPt(cv &start,cv &dir) const
	{
		return start+dir*intersect(start,dir);
	}
};

#endif /*def(thisHeader)*/
