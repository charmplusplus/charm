/*
Orion's Standard Library
written by 
Orion Sky Lawlor, olawlor@acm.org, 2/5/2001

Utilities for efficiently determining object intersections,
given a giant list of objects.
*/
#include <stdio.h>
#include <iostream.h>
#include "collision.h"

#define DEBUG_CHECKS 0 //Check invariants

namespace impl_collision_fem {
	void bad(const char *why) {
		fprintf(stderr,"Fatal error in collision system: %s\n",why);
		abort();
	}
};
using impl_collision_fem::bad;

static void print(const rSeg1d &s) {
	printf("[%.3g,%.3g]",s.getMin(),s.getMax());
}
void bbox3d::print(const char *desc) const
{
	if (desc) printf("%s: ",desc);
	for (int i=0;i<3;i++)
		{::print(segs[i]);if (i<2) printf("x");}
}

static void print(const vector3d &v) {
	printf("(%.3g,%.3g,%.3g) ",v.x,v.y,v.z);
}

void octant::print(const char *desc) const
{
	if (desc) printf("%s: ",desc);
	printf("%d home, %d boundary; ",nHome,length()-nHome);
	box.print("\nbbox");
	printf("\n");
}
void stats::print(void) 
{
	const char *names[3]={"x","y","z"};//Axis names
	int i;
	cout<<"---- Run-time statistics: ---"<<endl;
	cout<<"objects= "<<objects<<endl;
	cout<<"gridCells= "<<gridCells<<endl;
	cout<<"gridAdds= "<<gridAdds<<endl;
	
	cout<<"gridSizes (ave)= ";
	for (i=0;i<3;i++) cout<<(double)gridSizes[i]/objects<<names[i]<<" ";
	cout<<endl;
	
	cout<<"recursiveCalls= "<<recursiveCalls<<endl;
	cout<<"rejHomo = "<< rejHomo <<endl;
	cout<<"simpleCalls = "<< simpleCalls <<endl;
	cout<<"simpleFallbackCalls = "<< simpleFallbackCalls <<endl;
	
	cout<<"splits = ";
	for (i=0;i<3;i++) cout<<splits[i]<<names[i]<<" ";
	cout<<endl;
	cout<<"splitFailures = ";
	for (i=0;i<3;i++) cout<<splitFailures[i]<<names[i]<<" ";
	cout<<endl;
	
	cout<<"pivots = "<< pivots <<endl;
	cout<<"rejID = "<< rejID <<endl;
	cout<<"rejBbox = "<< rejBbox <<endl;
	cout<<"rejTerritory = ";
	for (i=0;i<3;i++) cout<<rejTerritory[i]<<names[i]<<" ";
	cout<<endl;
	cout<<"rejCollide = "<< rejCollide <<endl;
	cout<<"collisions = "<< collisions <<endl;
}
int stats::objects=0;//Total number of objects
int stats::gridCells=0;//Total number of grid cells
int stats::gridAdds=0;//Number of additions to grid cells
int stats::gridSizes[3]={0,0,0};//Total sizes of grid cells in each dimension
int stats::recursiveCalls=0;//Number of recursive calls
int stats::simpleCalls=0;//Number of simple calls
int stats::simpleFallbackCalls=0;//Number of simple calls because couldn't split
int stats::splits[3]={0,0,0};//Number of divisions along each axis
int stats::splitFailures[3]={0,0,0};//Number of failed divisions along each axis
int stats::pivots=0;//Number of pivot operations (octant::splitAt)
int stats::rejHomo=0;//Call rejected for being from one object
int stats::rejID=0;//Pair rejected for being out-of-order
int stats::rejBbox=0;//Pair rejected for BBox mismatch
int stats::rejTerritory[3]={0,0,0};//Pair rejected for being out of territory
int stats::rejCollide=0;//Pair rejected by slow intersection algorithm
int stats::collisions=0;//Number of actual intersections

//Ensure our constraints hold
void octant::check(void) const
{
	if (nHome>length()) bad("nHome cannot exceed n");
	if (nHome<=0) bad("nHome cannot be negative or zero");
	if (length()<0) bad("n cannot be negative");
	if (box.isEmpty()) bad("Bbox is empty");
	int i;
	for (i=0;i<nHome;i++)
		if (!box.contains(at(i)->getBbox().getSmallest()))
			bad("'Home' element not in bbox");
	for (i=nHome;i<length();i++) {
		if (!box.intersects(at(i)->getBbox()))
			bad("contained element does not touch us");
		vector3d s=at(i)->getBbox().getSmallest();
		if (box.containsOpen(s))
			bad("non-home element should be home");
	}
}

//////////////////////// Octant ///////////////////////////

octant::~octant() {}
void octant::add(const crossObjRec *p) {
	push_back(p);
}

//Figure out where to divide (about) half this octant's objs.
/*Algorithm:
 We do a find-median in place in our array, by repeatedly
 choosing a pivot, partitioning, and then recurse on the proper
 half of the array.
 Expected run time is O(n)-- n*(1 + 1/2 + 1/4 + 1/8 +...); 
 pathological pivots may give O(n^2)
*/
template <class T> 
inline void swap(T &a,T &b) {T tmp(a);a=b;b=tmp;}

int octant::splitAt(int alongAxis)
{
	octant &us=*this;
	
	//Find our median element in-place by recursive partitioning
	int target=nHome/2;//Rank of median element
	int lo=0, hi=nHome-1;//Unsearched values in our array
	int attempts=0;
	int l=-1,r=-1;//Elements [0..l] are <pivot; [r..hi] are >pivot
	STATS(splits[alongAxis]++)
	while (lo<hi) {
		attempts++;  STATS(pivots++)
		//Choose a pivot element and value
		int pivot=lo+(rand()&0x7fff)*(hi-lo+1)/0x8000;
#define val(x) us[x]->getBbox().axis(alongAxis).getMin()
		real pval=val(pivot);
		
		//Partition elements into those less and greater than pivot
		l=lo-1, r=hi+1;
		while (1) {
			real lval,rval;
			while ((lval=val(l+1))<pval) l++;
			while ((rval=val(r-1))>pval) r--;
			if (!(lval==pval && rval==pval))
				swap(us[l+1],us[r-1]);
			else 
			{//Both elements are equal to the pivot-- check if done
				bool finished=true;
				int i;
				for (i=l+2;i<=r-2;i++)
					if (val(i)!=pval) {finished=false;break;}
				if (finished) break;
				else swap(us[l+1],us[i]);
			}
		}
#if DEBUG_CHECKS //Check this step of the partitioning
	//	printf("Partitioned (%d-%d) at %d-- result %d,%d\n",lo,hi,pivot,l,r);
		if (l+1<0) bad("L negative!\n");
		if (l>=nHome) bad("L too big!\n");
		if (r<0) bad("R negative!\n");
		if (r-1>=nHome) bad("R too big!\n");
		for (int i=l+1;i<r;i++) if (val(i)!=pval) bad("equals aren't!");
		for (int i=0;i<=l;i++) if (val(i)>=pval) bad("lesses aren't!");
		for (int i=r;i<nHome;i++) if (val(i)<=pval) bad("greaters aren't!");
#endif
		//Do we need to partition again on a smaller set?
		if (l<target && target<r) break;
		else if (target<=l) hi=l;
		else /*target>=r*/ lo=r;
	}
#if DEBUG_CHECKS //Check this step of the partitioning
//	printf("Took %d tries to split %d (+%d)-element box (%d<%d<%d)\n",
//		attempts,nHome,(n-nHome),l,target,r);
#endif

#if 1 //Try to split homes into separated (non-overlapping) pieces
	if (r<nHome) return r;
	else if (l>0) return l;
	else {
		STATS(splitFailures[alongAxis]++);
		return -1;
	}
#else
	//Try to split into equal-sized pieces
	return target;
#endif
}

// Divide this octant along the given axis.
// This octant shrinks, the new one grows.
octant *octant::divide(int alongAxis)
{
	int oStart=nHome,oEnd=length();
	int i;
	
	//Figure out where to divide our elements
	int div=splitAt(alongAxis);
	if (div==-1) return null;
	
	//Make a new element to hold our other half
	octant &ret=*(new octant(length(),territory));
	octant &us=*this;
	
	//Divide home elements: us<-[0..div-1] and dest<-[div..nHome-1]
	ret.nHome=ret.length()=nHome-div;
	for (i=div;i<nHome;i++) ret[i-div]=us[i];
	us.nHome=us.length()=div;
	int ret_length=ret.length();//Prevents re-adding our own elements
	
	//Update bounding boxes
	us.box=us.at(0)->getBbox();
	ret.box=ret.at(0)->getBbox();
	for (i=length()-1;i>0;i--) us.box.expand(us.at(i)->getBbox());
	for (i=ret_length-1;i>0;i--) ret.box.expand(ret.at(i)->getBbox()); 
	
	//Find the boundary elements
	for (i=length()-1;i>=0;i--) 
		ret.addIfBoundary(at(i)); //We form his boundary
	for (i=ret_length-1;i>=0;i--) 
		addIfBoundary(ret.at(i)); //He forms our boundary
	for (i=oStart;i<oEnd;i++) {//Our old boundary elements are now shared
		addIfBoundary(at(i));
		ret.addIfBoundary(at(i));
	}
	
	//The lists are shorter now-- take up the slack
	//ret.trim();us.trim(); //<- old version could do this
	
#if DEBUG_CHECKS //Debugging-- check invariants
	ret.check();
	us.check();
#endif
	return &ret;
}

/////////////////////////////////////////////////////
//Enumerate all collisions in this octant:

template <class T>
inline T max(T a,T b) {if (a>b) return a; return b;}
template <class T>
inline T min(T a,T b) {if (a<b) return a; return b;}

//The basic O(n^2) collision algorithm:
static void simpleFindCollisions(const octant &o,collisionList &dest)
{
	STATS(simpleCalls++)
	int nH=o.getHome(),nI=o.getTotal();
	int h,i;
	for (h=0;h<nH;h++)
		for (i=0;i<nI;i++) {
			const crossObjRec &a=*o[h];
			const crossObjRec &b=*o[i];
			if (!(a.id<b.id)) {STATS(rejID++) continue;}
			const bbox3d &abox=a.getBbox();
			const bbox3d &bbox=b.getBbox();
			if (!abox.intersectsOpen(bbox)) {STATS(rejBbox++) continue;}
			
			//Territory is used to avoid duplicate intersections 
			// across different grid cells
			if (!o.x_inTerritory(max(abox.axis(0).getMin(),bbox.axis(0).getMin())))
				{STATS(rejTerritory[0]++) continue;}
			if (!o.y_inTerritory(max(abox.axis(1).getMin(),bbox.axis(1).getMin())))
				{STATS(rejTerritory[1]++) continue;}
			if (!o.z_inTerritory(max(abox.axis(2).getMin(),bbox.axis(2).getMin())))
				{STATS(rejTerritory[2]++) continue;}
			
			//	if (!poly3d::collide(a,b)) {STATS(rejCollide++) continue;}
			STATS(collisions++)
			dest.add(a.id,b.id);
		}
}

void octant::findCollisions(int splitAxis,collisionList &dest)
{
#if COLLISION_IS_RECURSIVE
	STATS(recursiveCalls++)
	
#if 0
	int iprio=at(0)->id.prio, i;//See if it's all the same ID
	for (i=length()-1;i>0;i--)
		if (at(i)->id.prio!=iprio)
			break;
	if (i==0) {STATS(rejHomo++) return;}//All the same ID
#endif
	
	if (getHome()>COLLISION_RECURSIVE_THRESH) 
	{//Split the problem once and then recurse
		octant *n;
		int divideCount=0;
		do {
			n=divide(splitAxis);
			splitAxis=(splitAxis+1)%3;
			divideCount++;
		} while (n==null && divideCount<3);
		if (n==null) 
		{//Couldn't divide along any axis-- use simple version instead
			STATS(simpleFallbackCalls++)
			simpleFindCollisions(*this,dest);
		} else {//Regular recursive division
			n->findCollisions(splitAxis,dest);
			delete n;
			findCollisions(splitAxis,dest);
		}
	} else
#endif
		simpleFindCollisions(*this,dest);
}

#if 0
////////////////////// Slow Polygon Collision Routine //////////////
inline bool sameSign(real a,real b,real c)
{
	if (a<=0 && b<=0 && c<=0) return true;
	if (a>=0 && b>=0 && c>=0) return true;
	return false;
}

static const real epsilon=1e-14; //For inside-ness test, below
bool poly3d::collide(const poly3d &a, const poly3d &b)
{
	class hPoly {
	public:
		halfspace3d plane;
		halfspace3d edges[3];
		
		hPoly(const poly3d &p) 
			:plane(p[0],p[1],p[2])
		{
			for (int i=0;i<3;i++) {
				int ip=(i+1)%3,im=(i+2)%3;
				edges[i].initCheck(p[i],
					p[i]+(p[ip]-p[i]).cross(p[im]-p[i]),
					p[ip],
					p[im]);
			}
		}
		//Return true if this plane point is also within the edges
		bool inside(const vector3d &x) {
			return (edges[0].side(x)>epsilon)&&
			       (edges[1].side(x)>epsilon)&&
			       (edges[2].side(x)>epsilon);
		}
		//Return true if this line segment intersects us
		bool intersects(const vector3d &start,const vector3d &end) {
			vector3d dir=end-start;
			real t=plane.intersect(start,dir);
		//	if (0!=isinf(t)) return false; //Line & planeparallel
			if (t<=0||t>=1) return false; //Plane outside segment
			return inside(start+t*dir);
		}
	};
	hPoly ha(a),hb(b);
	int i;
	for (i=0;i<3;i++)
		if (ha.intersects(b[i],b[(i+1)%3]))
			return true;
	for (i=0;i<3;i++)
		if (hb.intersects(a[i],a[(i+1)%3]))
			return true;
	return false;//No collisions
}

#endif



