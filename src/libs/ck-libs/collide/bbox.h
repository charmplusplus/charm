/*
Orion's Standard Library
written by 
Orion Sky Lawlor, olawlor@acm.org, 7/18/2001

Rather complete interval and box classes.
*/
#ifndef __OSL_BBOX_H
#define __OSL_BBOX_H

#include "pup.h"
#include "ckvector3d.h"

//A closed segment of 1d space-- [min,max]
template <class T>
class seg1dT {
	typedef seg1dT<T> seg1d;
	T min,max;
public:
	seg1dT(void) {}
	seg1dT(T Nmin,T Nmax) :min(Nmin),max(Nmax) {}
	
	void init(T a,T b,T c) {
		if (a<b) min=a,max=b; else min=b,max=a;
		if (c<min) min=c;
		else if (c>max) max=c;
	}
	T getMin(void) const {return min;}
	T getMax(void) const {return max;}
	void setMin(T m) {min=m;}
	void setMax(T m) {max=m;}
	
	//Set this span to contain no points
	seg1d &empty(void) {
		min=2000000000; max=-2000000000; return *this;
	}
	//Set this span to contain all points
	seg1d &infinity(void) {
		min=-2000000000; max=2000000000; return *this;
	}
	
	bool isEmpty(void) const { return max<min;}
	//Set this span to contain only this point
	seg1d &set(T b) {min=max=b; return *this;}
	//Set this span to contain these two points and everything in between
	seg1d &set(T a,T b) {
		if (a<b) {min=a;max=b;}
		else     {min=b;max=a;}
		return *this;
	}
	//Expand this span to contain this point
	void expandMin(T b) {if (min>b) min=b;}
	void expandMax(T b) {if (max<b) max=b;}
	seg1d &add(T b) {
		expandMin(b);expandMax(b);
		return *this;
	}
	//Expand this span to contain this span
	seg1d &add(const seg1d &b) {
		expandMin(b.min);expandMax(b.max);
		return *this;
	}
	
	//Return the intersection of this and that seg
	seg1d getIntersection(const seg1d &b) const {
		return seg1d(min>b.min?min:b.min, max<b.max?max:b.max);
	}
	//Return the union of this and that seg
	seg1d getUnion(const seg1d &b) const {
		return seg1d(min<b.min?min:b.min, max>b.max?max:b.max);
	}
	
	//Return true if this seg contains this point
	// in its interior or boundary (closed interval)
	bool contains(T b) const {
		return (min<=b)&&(b<=max);
	}
	//Return true if this seg contains this point 
	// in its interior (open interval)
	bool containsOpen(T b) const {
		return (min<b)&&(b<max);
	}
	//Return true if this seg contains this point 
	// in its interior or left endpoint (half-open interval)
	bool containsHalf(T b) const {
		return (min<=b)&&(b<max);
	}
	//Return true if this seg and that share any points
	bool intersects(const seg1d &b) const {
		return contains(b.min)||b.contains(min);
	}
	//Return true if this seg and that share any interior points
	bool intersectsOpen(const seg1d &b) const {
		return containsHalf(b.min)||b.containsOpen(min);
	}
	//Return true if this seg and that share any half-open points
	bool intersectsHalf(const seg1d &b) const {
		return containsHalf(b.min)||b.containsHalf(min);
	}
	
	inline void pup(PUP::er &p) {
		p|min;
		p|max;
	}
};
typedef seg1dT<double> rSeg1d;
typedef seg1dT<int> iSeg1d;

class bbox3d {
	rSeg1d segs[3];//Spans for x (0), y (1), and z (2)
public:   
	bbox3d() {}
	bbox3d(const rSeg1d &x,const rSeg1d &y,const rSeg1d &z)
		{segs[0]=x; segs[1]=y; segs[2]=z;}
	bbox3d(const CkVector3d &a,const CkVector3d &b,const CkVector3d &c)
	{
		segs[0].init(a[0],b[0],c[0]);
		segs[1].init(a[1],b[1],c[1]);
		segs[2].init(a[2],b[2],c[2]);
	}
	
	void print(const char *desc=NULL) const;

	rSeg1d &axis(int i) {return segs[i];}
	const rSeg1d &axis(int i) const {return segs[i];}
	
	void add(const CkVector3d &b) {
		for (int i=0;i<3;i++) segs[i].add(b[i]);
	}
	void add(const bbox3d &b) {
		for (int i=0;i<3;i++) segs[i].add(b.segs[i]);
	}
	bbox3d getUnion(const bbox3d &b) {
		return bbox3d(segs[0].getUnion(b.segs[0]),
			segs[1].getUnion(b.segs[1]),
			segs[2].getUnion(b.segs[2]));
	}
	bbox3d getIntersection(const bbox3d &b) {
		return bbox3d(segs[0].getIntersection(b.segs[0]),
			segs[1].getIntersection(b.segs[1]),
			segs[2].getIntersection(b.segs[2]));
	}
	//Interior or boundary (closed interval)
	bool intersects(const bbox3d &b) const {
		for (int i=0;i<3;i++)
			if (!segs[i].intersects(b.segs[i])) return false;
		return true;
	}
	//Interior only (open interval)
	bool intersectsOpen(const bbox3d &b) const {
		for (int i=0;i<3;i++)
			if (!segs[i].intersectsOpen(b.segs[i])) return false;
		return true;
	}
	//Interior or boundary (closed interval)
	bool contains(const CkVector3d &b) const {
		for (int i=0;i<3;i++)
			if (!segs[i].contains(b[i])) return false;
		return true;
	}
	//Interior only (open interval)
	bool containsOpen(const CkVector3d &b) const {
		for (int i=0;i<3;i++)
			if (!segs[i].containsOpen(b[i])) return false;
		return true;
	}
	//Interior or left endpoint (half-open interval)
	bool containsHalf(const CkVector3d &b) const {
		for (int i=0;i<3;i++)
			if (!segs[i].containsHalf(b[i])) return false;
		return true;
	}
	void empty(void) {
		for (int i=0;i<3;i++) segs[i].empty();
	}
	void infinity(void) {
		for (int i=0;i<3;i++) segs[i].infinity();
	}
	bool isEmpty(void) const {
		for (int i=0;i<3;i++) if (segs[i].isEmpty()) return true;
		return false;
	}
	CkVector3d getSmallest(void) const 
	{
		return CkVector3d(segs[0].getMin(),
			segs[1].getMin(),
			segs[2].getMin());
	}
	inline void pup(PUP::er &p) {
		p|segs[0]; p|segs[1]; p|segs[2];
	}
};

#endif //def(thisHeader)
