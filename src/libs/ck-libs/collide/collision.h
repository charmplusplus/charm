/*
Orion's Standard Library
written by 
Orion Sky Lawlor, olawlor@acm.org, 2/5/2001

Utilities for efficiently determining object intersections,
given a giant list of objects.
*/
#ifndef __OSL_COLLISION_H
#define __OSL_COLLISION_H
#include "col_aggregate.h"

#define STATS(x) //stats::x;

#ifndef COLLISION_USE_FLOAT_HACK
#define COLLISION_USE_FLOAT_HACK 1
#endif
#ifndef COLLISION_IS_RECURSIVE
#define COLLISION_IS_RECURSIVE 1
#endif
#ifndef COLLISION_RECURSIVE_THRESH
#define COLLISION_RECURSIVE_THRESH 10 //More than this many polys -> recurse
#endif

#include <stdlib.h>

//Map real world coordinates to integer grid indices
class gridMapping {
	vector3d origin,scales,sizes;//Convert world->grid coordinates
	#if COLLISION_USE_FLOAT_HACK
	double hakShift[3]; //For bitwise float rounding hack
	int hakStart[3];
	#endif
public:
	gridMapping() {}
	void init(const vector3d &Norigin,//Grid cell corner 0,0,0
		const vector3d &desiredSize);//Size of each cell
	
	real world2grid(int axis,real x) const {
		return (x-origin[axis])*scales[axis];
	}
	iSeg1d world2grid(int axis,const rSeg1d &s) const {
	#if COLLISION_USE_FLOAT_HACK
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
	#if COLLISION_USE_FLOAT_HACK
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
	void print(const gridLoc3d &g) {
	#if !COLLISION_USE_FLOAT_HACK
		const static int hakStart[3]={0,0,0};
	#endif
		printf("%d,%d,%d",g.x-hakStart[0],g.y-hakStart[1],g.z-hakStart[2]);
	}
};
extern gridMapping gridMap;//Global mapping

class collisionList;

class objConsumer {
public:
	virtual void add(const crossObjRec *obj)=0;
};

//A set of objects, organized by the location of the small
// corner of their bbox.
// Objects with their smallest bbox corner here are called "home" polys.
class octant : public growableBufferT<const crossObjRec *>, public objConsumer 
{
	typedef growableBufferT<const crossObjRec *> parent;
	int nHome;//Number of non-boundary elements
	bbox3d box;//We need every object that touches this region
	bbox3d territory;//We are responsible for intersections here

	//Figure out what index to divide our polys at
	int splitAt(int alongAxis);
	
public:
	octant(int size,bbox3d myTerritory) 
		: parent(size),territory(myTerritory)
	{nHome=0;box.empty();}
	virtual ~octant();
	
	void setBbox(const bbox3d &b) {box=b;}
	const bbox3d &getBbox(void) const {return box;}
	const bbox3d &getTerritory(void) const {return territory;}
	bool x_inTerritory(real v) const {return territory.axis(0).containsHalf(v);}
	bool y_inTerritory(real v) const {return territory.axis(1).containsHalf(v);}
	bool z_inTerritory(real v) const {return territory.axis(2).containsHalf(v);}
	
	//Get total count
	int getTotal(void) const {return length();}
	//Get home count
	int getHome(void) const {return nHome;}
	//Mark 0..h as at home
	void markHome(int h) {nHome=h;}
	
	//Grow to contain this added object (adjusts length if needed)
	void growTo(const crossObjRec *b) {
		box.expand(b->getBbox());
		push_fast(b);
	}
	//If needed, add this boundary poly (length must be preallocated)
	void addIfBoundary(const crossObjRec *b) {
		if (box.intersects(b->getBbox())) push_fast(b);
	}
	
	// Divide this octant along the given axis.
	// This octant shrinks, the new one grows.
	// Respects non-home polys.
	octant *divide(int alongAxis);
	
	void check(void) const;//Ensure our constraints hold
	void print(const char *desc=NULL) const;
	virtual void add(const crossObjRec *);
	
	void findCollisions(int splitAxis,collisionList &dest);
	void findCollisions(collisionList &dest)
		{findCollisions(0,dest);}
};

//A sparse but regular 3D grid of octants
class grid3d : public objConsumer {
	CkHashtableT<gridLoc3d,octant *> table;
	inline void addAt(crossObjRec *p,const gridLoc3d &g);
public:
	grid3d();
	virtual ~grid3d();
	virtual void add(crossObjRec *);
	void findCollisions(collisionList &dest);
};


//Records a single pair of intersecting polygons
class collision {
public:
	globalObjID A,B;
	collision(const globalObjID &A_, const globalObjID &B_) :A(A_),B(B_) { }
};

class collisionList : public growableBufferT<collision> {
public:
	collisionList() {}
	void add(const globalObjID &A, const globalObjID &B) {
		push_back(collision(A,B));
	}
};

//Collision statistics (for optimization)
class stats {public:
	static int objects;//Total number of objects
	static int gridCells;//Total number of grid cells
	static int gridAdds;//Number of additions to grid cells
	static int gridSizes[3];//Total sizes of grid cells in each dimension
	static int recursiveCalls;//Number of recursive calls
	static int simpleCalls;//Number of simple calls
	static int simpleFallbackCalls;//Number of unsplittable octants
	static int splits[3];//Number of divisions along each axis
	static int splitFailures[3];//Number of failed divisions along each axis
	static int pivots;//Number of pivot operations (octant::splitAt)
	static int rejHomo;//Call rejected for being from one object
	static int rejID;//Pair rejected for being out-of-order
	static int rejBbox;//Pair rejected for BBox mismatch
	static int rejTerritory[3];//Pair rejected for being out of territory
	static int rejCollide;//Pair rejected by slow intersection algorithm
	static int collisions;//Number of actual intersections
	static void print(void);
};

#endif //def(thisHeader)
