/*
Orion's Standard Library
written by 
Orion Sky Lawlor, olawlor@acm.org, 2/5/2001

Utilities for efficiently determining polygon intersections,
given a giant list of polygons.
*/
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "collision.h"

#define DEBUG_CHECKS 0 //Check invariants
namespace impl_aggregate_fem {
	void bad(const char *why) {
		fprintf(stderr,"Fatal error in collision system: %s\n",why);
		abort();
	}
};
using impl_aggregate_fem::bad;

/************** MemoryBuffer ****************
Manages an expandable buffer of bytes.  Like std::vector,
but typeless.
*/
memoryBuffer::memoryBuffer()//Empty initial buffer
{
	data=NULL;len=0;
}
memoryBuffer::memoryBuffer(size_t initLen)//Initial capacity specified
{
	data=NULL;len=0;reallocate(initLen);
}
memoryBuffer::~memoryBuffer()//Deletes array
{
	free(data);
}
void memoryBuffer::setData(const void *toData,size_t toLen)//Reallocate and copy
{
	reallocate(toLen);
	memcpy(data,toData,toLen);
}
void memoryBuffer::resize(size_t newlen)//Reallocate, preserving old data
{
	if (len==0) {reallocate(newlen); return;}
	if (len==newlen) return;
	void *oldData=data; size_t oldlen=len;
	data=malloc(len=newlen);
	memcpy(data,oldData,oldlen<newlen?oldlen:newlen);
	free(oldData);
}
void memoryBuffer::reallocate(size_t newlen)//Free old data, allocate new
{
	free(data);
	data=malloc(len=newlen);
}


#if 0 //A 2-element cache is actually slower than the 1-element case below
//Hashtable cache
template <int n>
class hashCache {
	typedef gridLoc3d KEY;
	typedef cellAggregator *OBJ;
	KEY keys[n];
	OBJ objs[n];
	int lastFound;
public:
	hashCache(const KEY &invalidKey) {
		for (int i=0;i<n;i++) keys[i]=invalidKey;
		lastFound=0;
	}
	inline OBJ lookup(const KEY &k) {
		if (k.compare(keys[lastFound]))
			return objs[lastFound];
		for (int i=0;i<n;i++)
			if (i!=lastFound)
				if (k.compare(keys[i]))
				{
					lastFound=i;
					return objs[i];
				}
		return OBJ(0);
	}
	void add(const KEY &k,const OBJ &o) {
		int doomed=lastFound+1;
		if (doomed>=n) doomed-=n;
		keys[doomed]=k;
		objs[doomed]=o;
	}
};
#endif

//Specialization of above for n==1
template <class KEY,class OBJ>
class hashCache1 {
	KEY key;
	OBJ obj;
public:
	hashCache1(const KEY &invalidKey) {
		key=invalidKey;
	}
	inline OBJ lookup(const KEY &k) {
		if (k.compare(key)) return obj;
		else return OBJ(0);
	}
	inline void add(const KEY &k,const OBJ &o) {
		key=k;
		obj=o;
	}
};

/************* voxelAggregator ***********
Accumulates lists of objects until there are enough to send off to a voxel.  
*/
voxelAggregator::voxelAggregator(const gridLoc3d &dest,collideMgr *mgr_)
	:destination(dest),mgr(mgr_)
{

}

//Send off any accumulated triangles.
void voxelAggregator::send(void) {
	if (obj.length()>0) sendMessage();
}

/* voxelAggregator::sendMessage is in parCollide.C */

/************* collisionAggregator ***************
Receives lists of points and triangles from the sources
on a particular machine.  Determines which voxels each
triangle spans, and adds the triangle to each voxelAggregator.
Maintains a sparse hashtable voxels.
*/
collisionAggregator::collisionAggregator(collideMgr *Nmgr)
	 :voxels(17,0.25),mgr(Nmgr)
{}
collisionAggregator::~collisionAggregator()
{
	compact();
}

//Add a new accumulator to the hashtable
voxelAggregator *collisionAggregator::addAccum(const gridLoc3d &dest)
{
	voxelAggregator *ret=new voxelAggregator(dest,mgr);
	voxels.put(dest)=ret;
	return ret;
}

//Add this chunk's triangles
void collisionAggregator::aggregate(int pe,int chunk,
	int n,const bbox3d *boxes)
{
	hashCache1<gridLoc3d,voxelAggregator *>
		cache(gridLoc3d(-1000000000,-1000000000,-1000000000));
	
	//Add each object to its corresponding voxelAggregators
	for (int i=0;i<n;i++) {
#if 1 //Compute bbox. and location
		const bbox3d &bbox=boxes[i];
		crossObjRec obj(globalObjID(pe,chunk,i),bbox);
		iSeg1d sx(gridMap.world2grid(0,bbox.axis(0))),
		       sy(gridMap.world2grid(1,bbox.axis(1))),
		       sz(gridMap.world2grid(2,bbox.axis(2)));
		
		STATS(objects++)
		STATS(gridSizes[0]+=sx.getMax()-sx.getMin())
		STATS(gridSizes[1]+=sy.getMax()-sy.getMin())
		STATS(gridSizes[2]+=sz.getMax()-sz.getMin())
#endif
		//Loop over grid voxels
		gridLoc3d g(sx.getMin(),sy.getMin(),sz.getMin());
		do {
		  do {
		    do {
			voxelAggregator *c=cache.lookup(g);
			if (c==NULL) {
				c=voxels.get(g);
				if (c==NULL) c=addAccum(g);
				cache.add(g,c);
			}
			c->add(obj);
		    } while (++g.x<sx.getMax());
		  } while (++g.y<sy.getMax());
		} while (++g.z<sz.getMax());
	}
}

//Send off all accumulated messages
void collisionAggregator::send(void)
{
	CkHashtableIterator *it=voxels.iterator();
	void *c;
	while (NULL!=(c=it->next())) (*(voxelAggregator **)c)->send();
	delete it;
}

//Delete all cached accumulators
void collisionAggregator::compact(void)
{
	CkHashtableIterator *it=voxels.iterator();
	void *c;
	while (NULL!=(c=it->next())) delete *(voxelAggregator **)c;
	delete it;
	voxels.empty();
}

