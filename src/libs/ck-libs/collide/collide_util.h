/*
Simple, basic data types for Collision detection.

Orion Sky Lawlor, olawlor@acm.org, 2003/3/19
*/
#ifndef __UIUC_CHARM_COLLIDE_UTIL_H
#define __UIUC_CHARM_COLLIDE_UTIL_H

#include "pup.h"
#include <math.h>
#include "ckvector3d.h"
typedef CkVector3d vector3d;
typedef double real;
#include "bbox.h"
#include "collide_cfg.h" // For COLLIDE_FLOAT_HACK
#include "collide_buffers.h" // For CollisionList

//Identifies an object across the machine
struct CollideObjID {
	int chunk;//Source chunk
	int number;//Chunk-local number
	int prio; //Collision "priority"--objects with same priority won't collide
	int pe; //Processor chunk was last living on
	CollideObjID(int chunk_,int number_,int prio_,int pe_) 
		:chunk(chunk_),number(number_),prio(prio_),pe(pe_) {}
	
	/**
 	 * Return true if we should be collided with b.
	 * This ends up comparing Collision priorities.
	 */
	inline int shouldCollide(const CollideObjID &b) const {
		if (prio<b.prio) return 1; //First check priority
		return 0;
	}
};

//Records a single pair of intersecting objects
class Collision {
public:
	CollideObjID A,B;
	Collision(const CollideObjID &A_, const CollideObjID &B_) :A(A_),B(B_) { }
};

class CollisionList : public growableBufferT<Collision> {
public:
	CollisionList() {}
	void add(const CollideObjID &A, const CollideObjID &B) {
		push_back(Collision(A,B));
	}
};


//An object sent to another processor
struct CollideObjRec {
	CollideObjID id;
	bbox3d box;
	CollideObjRec(const CollideObjID &id_,const bbox3d &box_)
		:id(id_),box(box_) {}

	const bbox3d &getBbox(void) const {return box;}
};

//Identifies a Collision voxel in the problem domain
class CollideLoc3d {
	//Circular-left-shift x by n bits
	static inline int cls(int x,unsigned int n) {
		const unsigned int intBits=8*sizeof(int);
		n&=(intBits-1);//Modulo intBits
		return (x<<n)|(x>>(intBits-n));
	}
public:
	int x,y,z;
	CollideLoc3d(int Nx,int Ny,int Nz) {x=Nx;y=Ny;z=Nz;}
	CollideLoc3d() {}
	int getX() const {return x;}
	int getY() const {return y;}
	int getZ() const {return z;}
	inline unsigned int  hash(void) const {
		return cls(x,6)+cls(y,17)+cls(z,28);
		//return (x<<8)+(y<<17)+cls(z,28);
	}
	static unsigned int staticHash(const void *key,size_t ignored) {
		return ((const CollideLoc3d *)key)->hash();
	}
	inline int compare(const CollideLoc3d &b) const {
		return x==b.x && y==b.y && z==b.z;
	}
	static int staticCompare(const void *k1,const void *k2,size_t ignored) {
		return ((const CollideLoc3d *)k1)->compare(*(const CollideLoc3d *)k2);
	}
};


/// Map real world (x,y,z) coordinates to integer (i,j,k) grid indices
class CollideGrid3d {
	vector3d origin,scales,sizes;//Convert world->grid coordinates
	#if COLLIDE_USE_FLOAT_HACK
	double hakShift[3]; //For bitwise float rounding hack
	int hakStart[3];
	#endif
public:
	CollideGrid3d() {}
	CollideGrid3d(const vector3d &Norigin,const vector3d &desiredSize) {
		init(Norigin,desiredSize);
	}
	void pup(PUP::er &p);
	
	void init(const vector3d &Norigin,//Grid cell corner 0,0,0
		const vector3d &desiredSize);//Size of each cell
	
	real world2grid(int axis,real x) const {
		return (x-origin[axis])*scales[axis];
	}
	iSeg1d world2grid(int axis,const rSeg1d &s) const {
	#if COLLIDE_USE_FLOAT_HACK
	 //Bizarre bitwise hack:
		float fl=(float)(hakShift[axis]+s.getMin());
		int lo=*(int *)&fl;
		float fh=(float)(hakShift[axis]+s.getMax());
		int hi=1+*(int *)&fh;
	#else
		int lo=(int)floor(world2grid(axis,s.getMin()));
		int hi=(int)ceil(world2grid(axis,s.getMax()));
	#endif
		return iSeg1d(lo,hi);
	}
	//Map grid coordinates->world coordinates
	real grid2world(int axis,real val) const {
	#if COLLIDE_USE_FLOAT_HACK
		return (val-hakStart[axis])
	#else
		return val
	#endif
		/*continued*/  *sizes[axis]+origin[axis];
	}
	rSeg1d grid2world(int axis,rSeg1d src) const {
		return rSeg1d(grid2world(axis,src.getMin()),
		              grid2world(axis,src.getMax()));
	}
	//Print this location nicely
	void print(const CollideLoc3d &g);
};


#endif
