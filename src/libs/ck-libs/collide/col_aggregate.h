/*
Orion's Standard Library
written by 
Orion Sky Lawlor, olawlor@acm.org, 2/5/2001

Utilities for efficiently determining polygon intersections,
given a giant list of polygons.
*/
#ifndef __OSL_AGGREGATE_H
#define __OSL_AGGREGATE_H

#include <math.h>
#include "ckhashtable.h"
typedef double real;
#include "vector3d.h"
#include "bbox.h"

//Java programmer compatability
#define null NULL
typedef bool boolean;

/********************* Generic utilities ***************/
//A simple extensible untyped chunk of memory.
//  Lengths are in bytes
class memoryBuffer {
public:
	typedef unsigned int size_t;
private:
	void *data;
	size_t len;//Length of data array above
	void setData(const void *toData,size_t toLen);//Reallocate and copy
public:
	memoryBuffer();//Empty initial buffer
	memoryBuffer(size_t initLen);//Initial capacity specified
	~memoryBuffer();//Deletes heap-allocated buffer
	memoryBuffer(const memoryBuffer &in) {data=NULL;setData(in.data,in.len);}
	memoryBuffer &operator=(const memoryBuffer &in) {setData(in.data,in.len); return *this;}
	
	size_t length(void) const {return len;}
	void *getData(void) {return data;}
	const void *getData(void) const {return data;}
	void detachBuffer(void) {data=NULL;len=0;}
	
	void resize(size_t newlen);//Reallocate, preserving old data
	void reallocate(size_t newlen);//Free old data, allocate new
};

//Superclass for simple flat memory containers
template <class T> class bufferT {
	T *data; //Data items in container
	int len; //Index of last valid member + 1
protected:
	bufferT() :data(NULL),len(0) {}
	bufferT(T *data_,int len_) :data(data_),len(len_) {}
	void set(T *data_,int len_) {data=data_;len=len_;}
	void setData(T *data_) {data=data_;}
	void setLength(int len_) {len=len_;}
	int &getLength(void) {return len;}
public:
	int length(void) const {return len;}
	
	T &operator[](int t) {return data[t];}
	const T &operator[](int t) const {return data[t];}
	T &at(int t) {return data[t];}
	const T &at(int t) const {return data[t];}
	
	T *getData(void) {return (T *)data;}
	const T *getData(void) const {return (T *)data;}
};

//For preallocated data
template <class T> class fixedBufferT : public bufferT<T> {
public:
	fixedBufferT(T *data_,int len_) :bufferT<T>(data_,len_) {}
};

//Variable size buffer
//T's constructors/destructors are not called by this (use std::vector)
//Copying the memory of a T must be equivalent to copying a T.
template <class T> class growableBufferT : public bufferT<T> {
	typedef bufferT<T> super;
	enum { sT=sizeof(T) };
	memoryBuffer buf;//Data storage
	int max;//Length of storage buffer
	//Don't use these:
	growableBufferT<T>(const growableBufferT<T> &in);
	growableBufferT<T> &operator=(const growableBufferT<T> &in);
public:
	growableBufferT<T>() :buf() {max=0;}
	growableBufferT<T>(size_t Len) :buf(Len*sT)
		{set((T*)buf.getData(),Len);max=Len;}
	
	int length(void) const {return super::length();}
	int &length(void) {return getLength();}
	int capacity(void) const {return max;}
	
	T *detachBuffer(void) {
		T *ret=(T *)buf.getData();
		buf.detachBuffer();
		set(NULL,0);
		max=0;
		return ret;
	}
	void empty(void) {reallocate(0);}
	void push_back(const T& v) {
		grow(length()+1);
		at(getLength()++)=v;
	}
	//Push without checking bounds
	void push_fast(const T& v) {
		at(getLength()++)=v;
	}
	void grow(int min) {
		if (min>max) resize(min+max+8);
	}
	void atLeast(int min) {//More conservative version of grow
		if (min>max) resize(min);
	}
	void resize(int Len);
	void reallocate(int Len);
};

template <class T> void growableBufferT<T>::resize(int Len) 
{
	buf.resize(Len*sT);
	setData((T*)buf.getData());
	max=Len;
}
template <class T> void growableBufferT<T>::reallocate(int Len) 
{
	buf.reallocate(Len*sT);
	setData((T*)buf.getData());
	setLength(0);
	max=Len;
}

/************************ parallel c.d. utilities **************/

//Identifies an object across the machine
struct globalObjID {
	int pe,chunk;//Source group and chunk
	int number;//Chunk-local number
	globalObjID(int p,int c,int n) :pe(p),chunk(c),number(n) {}

	int operator<(const globalObjID &b) const {
		if (pe<b.pe) return 1;
		if (pe>b.pe) return 0;
		if (chunk<b.chunk) return 1;
		if (chunk>b.chunk) return 0;
		return number<b.number;
	}
};

//An object sent to another processor
struct crossObjRec {
	globalObjID id;
	bbox3d box;
	crossObjRec(const globalObjID &id_,const bbox3d &box_)
		:id(id_),box(box_) {}

	const bbox3d &getBbox(void) const {return box;}
};

//Identifies a collision voxel in the problem domain
class gridLoc3d {
	//Circular-left-shift x by n bits
	static inline int cls(int x,unsigned int n) {
		const unsigned int intBits=8*sizeof(int);
		n&=(intBits-1);//Modulo intBits
		return (x<<n)|(x>>(intBits-n));
	}
public:
	int x,y,z;
	gridLoc3d(int Nx,int Ny,int Nz) {x=Nx;y=Ny;z=Nz;}
	gridLoc3d() {}
	int getX() const {return x;}
	int getY() const {return y;}
	int getZ() const {return z;}
	inline CkHashCode hash(void) const {
		return cls(x,6)+cls(y,17)+cls(z,28);
		//return (x<<8)+(y<<17)+cls(z,28);
	}
	static CkHashCode staticHash(const void *key,size_t ignored) {
		return ((const gridLoc3d *)key)->hash();
	}
	inline int compare(const gridLoc3d &b) const {
		return x==b.x && y==b.y && z==b.z;
	}
	static int staticCompare(const void *k1,const void *k2,size_t ignored) {
		return ((const gridLoc3d *)k1)->compare(*(const gridLoc3d *)k2);
	}
};

/****************** Parallel Implmentation: Aggregators *************/
/*This class aggregates triangles destined for one remote voxel
 */
class collideMgr;

class voxelAggregator {
private:
	//Accumulates objects for the current message
	growableBufferT<crossObjRec> obj;
	gridLoc3d destination;
	collideMgr *mgr;
	
	void sendMessage(void);//Send off accumulated points
	
public:
	voxelAggregator(const gridLoc3d &dest,collideMgr *mgr);
	
	//Add this object to the packList
	inline void add(const crossObjRec &o) {
		obj.push_back(o);
	}
	
	//Send off all accumulated objects.
	void send(void);
};

/*This class splits each chunk's triangles into messages
 * headed out to each voxel.  It is implemented as a group.
 */
class collisionAggregator {
	CkHashtableT<gridLoc3d,voxelAggregator *> voxels;
	collideMgr *mgr;
	
	//Add a new accumulator to the hashtable
	voxelAggregator *addAccum(const gridLoc3d &dest);
public:
	collisionAggregator(collideMgr *mgr);
	~collisionAggregator();
	
	//Add this chunk's triangles
	void aggregate(int pe,int chunk,
		int n,const bbox3d *boxes);
	
	//Send off all accumulated voxel messages
	void send(void);
	
	//Delete all cached accumulators
	void compact(void);
};

#endif //def(thisHeader)
