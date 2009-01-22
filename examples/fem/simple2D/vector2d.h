/*
Orion's Standard Library
Orion Sky Lawlor, 2/22/2000
NAME:		vector2d.h

DESCRIPTION:	C++ 2-Dimentional vector library (no templates)

This file provides various utility routines for easily
manipulating 2-D vectors-- included are arithmetic,
dot product, magnitude and normalization terms. 
All routines are provided right in the header file (for inlining).

Converted from vector3d.h.

*/

#ifndef __OSL_VECTOR_2D_H
#define __OSL_VECTOR_2D_H

#include <math.h>

typedef double real;

//vector2d is a cartesian vector in 2-space-- an x and y.
class vector2d {
public:
	real x,y;
	vector2d(void) {}//Default consructor
	//Simple 1-value constructor
	explicit vector2d(const real init) {x=y=init;}
	//Simple 1-value constructor
	explicit vector2d(int init) {x=y=init;}
	//2-value constructor
	vector2d(const real Nx,const real Ny) {x=Nx;y=Ny;}
	//Copy constructor
	vector2d(const vector2d &copy) {x=copy.x;y=copy.y;}
	
	//Cast-to-real * operators (treat vector as array)
	operator real *() {return &x;}
	operator const real *() const {return &x;}
	
/*Arithmetic operations: these are carefully restricted to just those
 that make unambiguous sense (to me... now...  ;-)
Counterexamples: vector*vector makes no sense (use .dot()) because
real/vector is meaningless (and we'd want a*b/b==a for b!=0), 
ditto for vector&vector (dot?), vector|vector (projection?), 
vector^vector (cross?),real+vector, vector+=real, etc.
*/
	vector2d &operator=(const vector2d &b) {x=b.x;y=b.y;return *this;}
	int operator==(const vector2d &b) const {return (x==b.x)&&(y==b.y);}
	int operator!=(const vector2d &b) const {return (x!=b.x)||(y!=b.y);}
	vector2d operator+(const vector2d &b) const {return vector2d(x+b.x,y+b.y);}
	vector2d operator-(const vector2d &b) const {return vector2d(x-b.x,y-b.y);}
	vector2d operator*(const real scale) const 
		{return vector2d(x*scale,y*scale);}
	friend vector2d operator*(const real scale,const vector2d &v)
		{return vector2d(v.x*scale,v.y*scale);}
	vector2d operator/(const real &div) const
		{real scale=1.0/div;return vector2d(x*scale,y*scale);}
	vector2d operator-(void) const {return vector2d(-x,-y);}
	void operator+=(const vector2d &b) {x+=b.x;y+=b.y;}
	void operator-=(const vector2d &b) {x-=b.x;y-=b.y;}
	void operator*=(const real scale) {x*=scale;y*=scale;}
	void operator/=(const real div) {real scale=1.0/div;x*=scale;y*=scale;}

//Vector-specific operations
	//Return the square of the magnitude of this vector
	real magSqr(void) const {return x*x+y*y;}
	//Return the magnitude (length) of this vector
	real mag(void) const {return sqrt(magSqr());}
	
	//Return the square of the distance to the vector b
	real distSqr(const vector2d &b) const 
		{return (x-b.x)*(x-b.x)+(y-b.y)*(y-b.y);}
	//Return the distance to the vector b
	real dist(const vector2d &b) const {return sqrt(distSqr(b));}
	
	//Return the dot product of this vector and b
	real dot(const vector2d &b) const {return x*b.x+y*b.y;}
	//Return the cosine of the angle between this vector and b
	real cosAng(const vector2d &b) const {return dot(b)/(mag()*b.mag());}
	
	//Return the "direction" (unit vector) of this vector
	vector2d dir(void) const {return (*this)/mag();}

	//Return the CCW perpendicular vector
	vector2d perp(void) const {return vector2d(-y,x);}

	//Return this vector scaled by that
	vector2d &scale(const vector2d &b) {x*=b.x;y*=b.y;return *this;}
	
	//Return the largest coordinate in this vector
	real max_(void) {return (x>y)?x:y;}
	//Make each of this vector's coordinates at least as big
	// as the given vector's coordinates.
	void enlarge(const vector2d &by)
	{if (by.x>x) x=by.x; if (by.y>y) y=by.y;}
};

#endif //__OSL_VECTOR2D_H


