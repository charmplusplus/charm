/*
Simple grid manipulation routines.

Orion Sky Lawlor, olawlor@acm.org, 5/30/2001
*/
#ifndef __CSAR_GRIDUTIL_H
#define __CSAR_GRIDUTIL_H

#include <math.h>

//This typedef allows us to easily change the floating-point
// type used in all geometry calculations.
typedef double real;

#include <charm++.h>
#include "vector3d.h"

//An i,j,k location in one 3D grid block
class blockDim;
class blockLoc {
protected:
	int i,j,k;
public:
	blockLoc() { }
	blockLoc(int i_,int j_,int k_) 
		:i(i_), j(j_), k(k_) { }
	blockLoc operator+ (const blockLoc &b) const
	  { return blockLoc(i+b.i,j+b.j,k+b.k); }
	blockLoc &operator+= (const blockLoc &b)
	  { i+=b.i; j+=b.j; k+=b.k; return *this; }
	blockDim operator- (const blockLoc &b) const;
	friend blockLoc operator*(const blockLoc &a,int k) 
	  { return blockLoc(a.i*k,a.j*k,a.k*k); }
	friend blockLoc operator*(int k,const blockLoc &a) 
	  { return a*k; }

	bool operator==(const blockLoc &o) const 
		{return i==o.i && j==o.j && k==o.k; }
	bool operator!=(const blockLoc &o) const 
		{return i!=o.i || j!=o.j || k!=o.k; }

	//Dimension indexing
	int &operator[](int d) {return (&i)[d];}
	int operator[](int d) const {return (&i)[d];}
  void pup(PUP::er &p)
  {
    p(i);p(j);p(k);
  }
};

//The i, j, and k dimentions of one block
class blockDim : public blockLoc {
public:
	blockDim() { }
	blockDim(int i_,int j_,int k_) 
		:blockLoc(i_,j_,k_)
		{ }
	int getSize(void) const 
		{ return i*j*k; }
	//Return the (0-based) array index of the (0-based) point (xi,xj,xk)
	int c_index(int xi,int xj,int xk) const 
		{ return xi+i*(xj+j*xk);  }

	//Shorthand for above
	int operator[](const blockLoc &l) const
	  { return c_index(l[0],l[1],l[2]); }

	//Dimension indexing
	int &operator[](int d) {return (&i)[d];}
	int operator[](int d) const {return (&i)[d];}
};

inline blockDim blockLoc::operator- (const blockLoc &b) const
       { return blockDim(i-b.i,j-b.j,k-b.k); }

#endif








