/*
Serial Collision detection classes.

Orion Sky Lawlor, olawlor@acm.org, 2003/3/19
*/
#ifndef __UIUC_CHARM_COLLIDE_SERIAL_H
#define __UIUC_CHARM_COLLIDE_SERIAL_H

#include "collide_util.h"
#include "collide_buffers.h"
#include "collide_cfg.h"

#if !COLLIDE_STATS 
# define STATS(x) /* empty */
#else
# define STATS(x) stats::x;
//Collision statistics (for measurement-based optimization)
class stats {public:
	static int objects;//Total number of objects
	static int gridCells;//Total number of grid cells
	static int gridAdds;//Number of additions to grid cells
	static int gridSizes[3];//Total sizes of grid cells in each dimension
	static int recursiveCalls;//Number of recursive calls
	static int simpleCalls;//Number of simple calls
	static int simpleFallbackCalls;//Number of unsplittable CollideOctants
	static int splits[3];//Number of divisions along each axis
	static int splitFailures[3];//Number of failed divisions along each axis
	static int pivots;//Number of pivot operations (CollideOctant::splitAt)
	static int rejHomo;//Call rejected for being from one object
	static int rejID;//Pair rejected for being out-of-order
	static int rejBbox;//Pair rejected for BBox mismatch
	static int rejTerritory[3];//Pair rejected for being out of territory
	static int rejCollide;//Pair rejected by slow intersection algorithm
	static int Collisions;//Number of actual intersections
	static void print(void);
};
#endif

class CollideObjConsumer {
public:
	virtual void add(const CollideObjRec *obj)=0;
};

//A set of objects, organized by the location of the small
// corner of their bbox.
// Objects with their smallest bbox corner here are called "home" polys.
class CollideOctant : public growableBufferT<const CollideObjRec *>, public CollideObjConsumer 
{
	typedef growableBufferT<const CollideObjRec *> parent;
	int nHome;//Number of non-boundary elements
	bbox3d box;//We need every object that touches this region
	bbox3d territory;//We are responsible for intersections here

	//Figure out what index to divide our polys at
	int splitAt(int alongAxis);
	
public:
	CollideOctant(int size,bbox3d myTerritory) 
		: parent(size),territory(myTerritory)
	{nHome=0;box.empty();}
	virtual ~CollideOctant();
	
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
	void growTo(const CollideObjRec *b) {
		box.add(b->getBbox());
		push_fast(b);
	}
	//If needed, add this boundary poly (length must be preallocated)
	void addIfBoundary(const CollideObjRec *b) {
		if (box.intersects(b->getBbox())) push_fast(b);
	}
	
	// Divide this CollideOctant along the given axis.
	// This CollideOctant shrinks, the new one grows.
	// Respects non-home polys.
	CollideOctant *divide(int alongAxis);
	
	void check(void) const;//Ensure our constraints hold
	void print(const char *desc=NULL) const;
	virtual void add(const CollideObjRec *);
	
	void findCollisions(int splitAxis,CollisionList &dest);
	void findCollisions(CollisionList &dest)
		{findCollisions(0,dest);}
};

#endif //def(thisHeader)
