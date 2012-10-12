/*
Orion's Standard Library
written by 
Orion Sky Lawlor, olawlor@acm.org, 2/5/2001

Utilities for efficiently determining object intersections,
given a giant list of objects.
*/
#include <stdio.h>
#include <iostream>
using namespace std;
#include "collide_serial.h"
#include "charm.h" /* for CkAbort */

#define DEBUG_CHECKS 0 //Check invariants

static void print(const rSeg1d &s) {
	printf("[%.3g,%.3g]",s.getMin(),s.getMax());
}

#if 0
static void print(const vector3d &v) {
	printf("(%.3g,%.3g,%.3g) ",v.x,v.y,v.z);
}
#endif
void bbox3d::print(const char *desc) const
{
	if (desc) printf("%s: ",desc);
	// for (int i=0;i<3;i++)
	//	{::print(segs[i]);if (i<2) printf("x");}
	printf("(%.3g,%.3g,%.3g) - (%.3g,%.3g,%.3g) ",
		axis(0).getMin(),axis(1).getMin(),axis(2).getMin(),
		axis(0).getMax(),axis(1).getMax(),axis(2).getMax());
}

void CollideOctant::print(const char *desc) const
{
	if (desc) printf("%s: ",desc);
	printf("%d home, %d boundary; ",nHome,length()-nHome);
	box.print("\nbbox");
	printf("\n");
}

#if COLLIDE_STATS
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
	cout<<"Collisions = "<< Collisions <<endl;
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
int stats::pivots=0;//Number of pivot operations (CollideOctant::splitAt)
int stats::rejHomo=0;//Call rejected for being from one object
int stats::rejID=0;//Pair rejected for being out-of-order
int stats::rejBbox=0;//Pair rejected for BBox mismatch
int stats::rejTerritory[3]={0,0,0};//Pair rejected for being out of territory
int stats::rejCollide=0;//Pair rejected by slow intersection algorithm
int stats::Collisions=0;//Number of actual intersections
#endif

//Ensure our constraints hold
void CollideOctant::check(void) const
{
	if (nHome>length()) CkAbort("nHome cannot exceed n");
	if (nHome<=0) CkAbort("nHome cannot be negative or zero");
	if (length()<0) CkAbort("n cannot be negative");
	if (box.isEmpty()) CkAbort("Bbox is empty");
	int i;
	for (i=0;i<nHome;i++)
		if (!box.contains(at(i)->getBbox().getSmallest()))
			CkAbort("'Home' element not in bbox");
	for (i=nHome;i<length();i++) {
		if (!box.intersects(at(i)->getBbox()))
			CkAbort("contained element does not touch us");
		vector3d s=at(i)->getBbox().getSmallest();
		if (box.containsOpen(s))
			CkAbort("non-home element should be home");
	}
}

//////////////////////// Octant ///////////////////////////

CollideOctant::~CollideOctant() {}
void CollideOctant::add(const CollideObjRec *p) {
	push_back(p);
}

//Figure out where to divide (about) half this CollideOctant's objs.
/*Algorithm:
 We do a find-median in place in our array, by repeatedly
 choosing a pivot, partitioning, and then recurse on the proper
 half of the array.
 Expected run time is O(n)-- n*(1 + 1/2 + 1/4 + 1/8 +...); 
 pathological pivots may give O(n^2)
*/
template <class T> 
inline void myswap(T &a,T &b) {T tmp(a);a=b;b=tmp;}

int CollideOctant::splitAt(int alongAxis)
{
	CollideOctant &us=*this;
	
	//Find our median element in-place by recursive partitioning
	int target=nHome/2;//Rank of median element
	int lo=0, hi=nHome-1;//Unsearched values in our array
	int attempts=0;
	int l=-1,r=-1;//Elements [0..l] are <pivot; [r..hi] are >pivot
	STATS(splits[alongAxis]++)
	while (lo<hi) {
		attempts++;  STATS(pivots++)
		//Choose a pivot element and value
		//int pivot=lo+(rand()&0x7fff)*(hi-lo+1)/0x8000;
	       int pivot = (lo/2)+(hi/2);	       
#define val(x) us[x]->getBbox().axis(alongAxis).getMin()
		real pval=val(pivot);
		
		//Partition elements into those less and greater than pivot
		l=lo-1, r=hi+1;
		while (1) {
			real lval,rval;
			while ((lval=val(l+1))<pval) l++;
			while ((rval=val(r-1))>pval) r--;
			if (!(lval==pval && rval==pval))
				myswap(us[l+1],us[r-1]);
			else 
			{//Both elements are equal to the pivot-- check if done
				bool finished=true;
				int i;
				for (i=l+2;i<=r-2;i++)
					if (val(i)!=pval) {finished=false;break;}
				if (finished) break;
				else myswap(us[l+1],us[i]);
			}
		}
#if DEBUG_CHECKS //Check this step of the partitioning
	//	printf("Partitioned (%d-%d) at %d-- result %d,%d\n",lo,hi,pivot,l,r);
		if (l+1<0) CkAbort("L negative!\n");
		if (l>=nHome) CkAbort("L too big!\n");
		if (r<0) CkAbort("R negative!\n");
		if (r-1>=nHome) CkAbort("R too big!\n");
		for (int i=l+1;i<r;i++) if (val(i)!=pval) CkAbort("equals aren't!");
		for (int i=0;i<=l;i++) if (val(i)>=pval) CkAbort("lesses aren't!");
		for (int i=r;i<nHome;i++) if (val(i)<=pval) CkAbort("greaters aren't!");
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

// Divide this CollideOctant along the given axis.
// This CollideOctant shrinks, the new one grows.
CollideOctant *CollideOctant::divide(int alongAxis)
{
	int oStart=nHome,oEnd=length();
	int i;
	
	//Figure out where to divide our elements
	int div=splitAt(alongAxis);
	if (div==-1) return NULL;
	
	//Make a new element to hold our other half
	CollideOctant &ret=*(new CollideOctant(length(),territory));
	CollideOctant &us=*this;
	
	//Divide home elements: us<-[0..div-1] and dest<-[div..nHome-1]
	ret.nHome=ret.length()=nHome-div;
	for (i=div;i<nHome;i++) ret[i-div]=us[i];
	us.nHome=us.length()=div;
	int ret_length=ret.length();//Prevents re-adding our own elements
	
	//Update bounding boxes
	us.box=us.at(0)->getBbox();
	ret.box=ret.at(0)->getBbox();
	for (i=length()-1;i>0;i--) us.box.add(us.at(i)->getBbox());
	for (i=ret_length-1;i>0;i--) ret.box.add(ret.at(i)->getBbox()); 
	
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
//Enumerate all Collisions in this CollideOctant:

template <class T>
inline T mymax(T a,T b) {if (a>b) return a; return b;}
template <class T>
inline T mymin(T a,T b) {if (a<b) return a; return b;}

//The basic O(n^2) Collision algorithm:
static void simpleFindCollisions(const CollideOctant &o,CollisionList &dest)
{
	STATS(simpleCalls++)
	int nH=o.getHome(),nI=o.getTotal();
	int h,i;
	for (h=0;h<nH;h++)
		for (i=0;i<nI;i++) {
			const CollideObjRec &a=*o[h];
			const CollideObjRec &b=*o[i];
			if (!a.id.shouldCollide(b.id)) {STATS(rejID++) continue;}
			const bbox3d &abox=a.getBbox();
			const bbox3d &bbox=b.getBbox();
			if (!abox.intersectsOpen(bbox)) {STATS(rejBbox++) continue;}
			
			//Territory is used to avoid duplicate intersections 
			// across different grid cells
			if (!o.x_inTerritory(mymax(abox.axis(0).getMin(),bbox.axis(0).getMin())))
				{STATS(rejTerritory[0]++) continue;}
			if (!o.y_inTerritory(mymax(abox.axis(1).getMin(),bbox.axis(1).getMin())))
				{STATS(rejTerritory[1]++) continue;}
			if (!o.z_inTerritory(mymax(abox.axis(2).getMin(),bbox.axis(2).getMin())))
				{STATS(rejTerritory[2]++) continue;}
			
			//	if (!poly3d::collide(a,b)) {STATS(rejCollide++) continue;}
			STATS(Collisions++)
			dest.add(a.id,b.id);
		}
}

void CollideOctant::findCollisions(int splitAxis,CollisionList &dest)
{
	if (getHome()==0) return; /* nothing to do */
#if COLLIDE_IS_RECURSIVE
	STATS(recursiveCalls++)
	
#if 1
	int iprio=at(0)->id.prio, i;//See if it's all the same ID
	for (i=length()-1;i>0;i--)
		if (at(i)->id.prio!=iprio)
			break;
	if (i==0) {STATS(rejHomo++) return;}//All the same ID
#endif
	
	if (getHome()>COLLIDE_RECURSIVE_THRESH) 
	{//Split the problem once and then recurse
		CollideOctant *n;
		int divideCount=0;
		do {
			n=divide(splitAxis);
			splitAxis=(splitAxis+1)%3;
			divideCount++;
		} while (n==NULL && divideCount<3);
		if (n==NULL) 
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



