/*
Orion's Standard Library
Orion Sky Lawlor, 11/3/1999
NAME:		vector3d.h

DESCRIPTION:	C++ 3-Dimentional vector library (no templates)

This file provides various utility routines for easily
manipulating 3-D vectors-- included are arithmetic,
dot/cross product, magnitude and normalization terms. 
Most routines are provided right in the header file (for inlining).
*/

#ifndef __OSL_VECTOR_3D_H
#define __OSL_VECTOR_3D_H

typedef double real;

class vector3d;
//Polar3d is a point expressed in a 3D spherical coordinate system--
// theta is the angle (right-handed about +z) in the x-y axis (longitude);
// phi is the angle up (toward +z) from the x-y axis (latitude);
// r is the distance of the point from the origin.
class polar3d {
public:
	real theta, phi;//Angles in radians
	real r;//Distance from origin
//Nothing too useful can be done here, except convert to/from a vector3d (see below)
	polar3d() {} //Default constructor
	polar3d(real Nt,real Np,real Nr) {theta=Nt;phi=Np;r=Nr;}
	polar3d(const vector3d &v);
};

//Vector3d is a cartesian vector in 3-space-- an x, y, and z.
// For cross products, the space is assumed to be right-handed (x cross y = +z)
class vector3d {
public:
	real x,y,z;
	vector3d(void) {}//Default consructor
	//Simple 1-value constructor
	explicit vector3d(const real init) {x=y=z=init;}
	//Simple 1-value constructor
	explicit vector3d(int init) {x=y=z=init;}
	//3-value constructor
	vector3d(const real Nx,const real Ny,const real Nz) {x=Nx;y=Ny;z=Nz;}
	//Real array constructor
	vector3d(const real *arr) {x=arr[0];y=arr[1];z=arr[2];}
	//Copy constructor
	vector3d(const vector3d &copy) {x=copy.x;y=copy.y;z=copy.z;}
	//Polar coordinate constructor
	vector3d(const polar3d &p);
	vector3d &operator=(const vector3d &b) {x=b.x;y=b.y;z=b.z;return *this;}

	
	//This lets you typecast a vector to a real array
	operator real *() {return (real *)&x;}
	operator const real *() const {return (const real *)&x;}

//Basic mathematical operators	
	int operator==(const vector3d &b) const {return (x==b.x)&&(y==b.y)&&(z==b.z);}
	int operator!=(const vector3d &b) const {return (x!=b.x)||(y!=b.y)||(z!=b.z);}
	vector3d operator+(const vector3d &b) const {return vector3d(x+b.x,y+b.y,z+b.z);}
	vector3d operator-(const vector3d &b) const {return vector3d(x-b.x,y-b.y,z-b.z);}
	vector3d operator*(const real scale) const 
		{return vector3d(x*scale,y*scale,z*scale);}
	friend vector3d operator*(const real scale,const vector3d &v)
		{return vector3d(v.x*scale,v.y*scale,v.z*scale);}
	vector3d operator/(const real &div) const
		{real scale=1.0/div;return vector3d(x*scale,y*scale,z*scale);}
	vector3d operator-(void) const {return vector3d(-x,-y,-z);}
	void operator+=(const vector3d &b) {x+=b.x;y+=b.y;z+=b.z;}
	void operator-=(const vector3d &b) {x-=b.x;y-=b.y;z-=b.z;}
	void operator*=(const real scale) {x*=scale;y*=scale;z*=scale;}
	void operator/=(const real div) {real scale=1.0/div;x*=scale;y*=scale;z*=scale;}

//Vector-specific operations
	//Return the square of the magnitude of this vector
	real magSqr(void) const {return x*x+y*y+z*z;}
	//Return the magnitude (length) of this vector
	real mag(void) const {return sqrt(magSqr());}
	
	//Return the square of the distance to the vector b
	real distSqr(const vector3d &b) const 
		{return (x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z);}
	//Return the distance to the vector b
	real dist(const vector3d &b) const {return sqrt(distSqr(b));}
	
	//Return the dot product of this vector and b
	real dot(const vector3d &b) const {return x*b.x+y*b.y+z*b.z;}
	//Return the cosine of the angle between this vector and b
	real cosAng(const vector3d &b) const {return dot(b)/(mag()*b.mag());}
	
	//Return the "direction" (unit vector) of this vector
	vector3d dir(void) const {return (*this)/mag();}
	//Return the right-handed cross product of this vector and b
	vector3d cross(const vector3d &b) const;
	
	//Return the largest coordinate in this vector
	real max(void);
	//Make each of this vector's coordinates at least as big
	// as the given vector's coordinates.
	void enlarge(const vector3d &by);
};

//An axis-aligned 3D bounding box
class bbox3d {
public:
	vector3d min,max;
	bbox3d() {empty();}
	void empty(void) {min.x=min.y=min.z=1e30;max.x=max.y=max.z=-1e30;}
	bool isEmpty(void) const {return min.x>max.x;}
	void add(const vector3d &p) {
		if (min.x>p.x) min.x=p.x;
		if (min.y>p.y) min.y=p.y;
		if (min.z>p.z) min.z=p.z;
		if (max.x<p.x) max.x=p.x;
		if (max.y<p.y) max.y=p.y;
		if (max.z<p.z) max.z=p.z;
	}
};

//A halfspace3d is the portion of a 3d plane lying on
// one side of the plane (p1,p2,p3).
class halfspace3d {
public:
	// n dot p+d==0 on plane point p
	vector3d n;//Plane normal
	real d;
	
	typedef const vector3d cv;
	halfspace3d() {}
	halfspace3d(cv &p1,cv &p2,cv &p3) {init(p1,p2,p3);}
	halfspace3d(cv &p1,cv &p2,cv &p3,cv &in) {initCheck(p1,p2,p3,in);}
	//Norm points into the halfspace; p0 is on the line
	halfspace3d(cv &norm,cv &p0) {n=norm;d=-n.dot(p0);}

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
	real side(cv &pt) const
	{return n.dot(pt)+d;}
	
	//Return a value t such that pos+t*dir lies on our plane.
	real intersect(cv &pos,cv &dir) const
		{return -(d+n.dot(pos))/n.dot(dir);}
	
	/*Returns the point that lies on our plane and 
	  the line starting at start and going in dir.*/
	vector3d intersectPt(cv &start,cv &dir) const
	{
		return start+dir*intersect(start,dir);
	}
};
#endif //__OSL_VECTOR3D_H


