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
#include "ckvector3d.h"
typedef CkVector3d vector3d;

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

	void getInt3(int *dest,int del=0) const {
		dest[0]=i-del; dest[1]=j-del; dest[2]=k-del;
	}
  void pup(PUP::er &p)
  {
    p(i);p(j);p(k);
  }
	void print(void) {
		CkPrintf("%d,%d,%d",i,j,k);
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
	inline int operator[](const blockLoc &l) const
	  { return c_index(l[0],l[1],l[2]); }

	//Dimension indexing
	int &operator[](int d) {return (&i)[d];}
	int operator[](int d) const {return (&i)[d];}
};

inline blockDim blockLoc::operator- (const blockLoc &b) const
       { return blockDim(i-b.i,j-b.j,k-b.k); }

//Some subset of a block
class blockSpan {
public:
	blockLoc start; //First included grid location
	blockLoc end; //Last included grid location PLUS 1 ON EACH AXIS

	blockSpan() { }
	blockSpan(const blockLoc &s,const blockLoc &e) 
		:start(s), end(e) { }

	void pup(PUP::er &p) {
		start.pup(p);
		end.pup(p);
	}

	blockDim getDim(void) const { return end-start; }
	void getInt3(int *start3,int *end3) const {
		start.getInt3(start3);
		end.getInt3(end3,1);
	}

	//Swap so start and end are sensible
	void orient(void) {
		for (int axis=0;axis<3;axis++) {
			if (start[axis]>=end[axis]) {
				end[axis]--;
				int tmp=start[axis];
				start[axis]=end[axis];
				end[axis]=tmp;
				end[axis]++;
			}
		}
	}

	//Return the axis we have no thickness along, or -1 if none
	int getFlatAxis(void) const {
		for (int axis=0;axis<3;axis++)
			if (start[axis]+1==end[axis])
				return axis;
		return -1;
	}
	//Return the block::face number we apply to, or -1 if none
	int getFace(void) const {
		int axis=getFlatAxis();
		if (axis==-1) return -1;
		if (start[axis]==0) return axis;
		else return axis+3;
	}

	//Return true if we contain the given location
	bool contains(const blockLoc &l) const
	{
		for (int axis=0;axis<3;axis++)
			if (!(start[axis]<=l[axis] && l[axis]<end[axis]))
				return false;
		return true;
	}

	//Return true if we have nonzero area
	bool hasVolume(void) const {
		for (int axis=0;axis<3;axis++)
			if (start[axis]==end[axis])
				return false;
		return true;
	}

	blockSpan operator+(const blockLoc &l) const 
		{return blockSpan(start+l,end+l);}
	blockSpan operator-(const blockLoc &l) const 
		{return blockSpan(start-l,end-l);}

	bool operator==(const blockSpan &o) const
		{return start==o.start && end==o.end;}
	bool operator!=(const blockSpan &o) const
		{return start!=o.start || end!=o.end;}

	void print(void) {
		CkPrintf(" start=");start.print();
		CkPrintf(" end=");end.print();
	}
};

#define BLOCKSPAN_FOR(i,span) \
	blockSpan loop_iter=span; \
	for (i[2]=loop_iter.start[2];i[2]<loop_iter.end[2];i[2]++) \
	for (i[1]=loop_iter.start[1];i[1]<loop_iter.end[1];i[1]++) \
	for (i[0]=loop_iter.start[0];i[0]<loop_iter.end[0];i[0]++)

#endif








