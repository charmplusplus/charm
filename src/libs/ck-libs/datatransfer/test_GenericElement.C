/**
Testing stub for GenericElement.

Orion Sky Lawlor, olawlor@acm.org, 2004/3/27
*/
#include <stdio.h>
#include <stdlib.h>
#include "GenericElement.h"

class fixedConcreteElement : public ConcreteElementNodeData {
public:
	enum{nPts=4};
	CPoint pts[nPts], data[nPts];
	fixedConcreteElement(
		CPoint a,CPoint b,CPoint c,CPoint d)
	{
		pts[0]=a; pts[1]=b; pts[2]=c; pts[3]=d;
		
		/** I'll transfer the coordinate data, for testing purposes */
		for (int i=0;i<nPts;i++) data[i]=pts[i];
	}
	
	void shift(const CVector &v) {
		for (int i=0;i<nPts;i++) pts[i]+=v;
	}
	
	CPoint getNodeLocation(int i) const {return pts[i];}
	const double *getNodeData(int i) const {return data[i];}
	
	CPoint getCenter(void) const {
		CPoint r(0);
		for (int i=0;i<nPts;i++) r+=pts[i];
		return r*(1.0/nPts);
	}
};

void die(const char *fmt,const char *file,int line,const char *why) {
	fprintf(stderr,fmt,file,line,why);
	abort();
}

#define check(expr) { bool b=(expr); \
	if (!b) die("Check failed: %s:%d> (%s) is false\n",__FILE__,__LINE__,#expr); \
  }

bool equalVectors(const CVector &l,const CVector &r) {
	for (int i=0;i<3;i++)
		if (fabs(l[i]-r[i])>1.0e-10)
			return false;
	return true;
}

int main() {
	CVector geoShift(1.2,2.3,3.4);
	
	/**
	              .
	       .  l   .   r  .
	   y
	   o  x
	*/
	fixedConcreteElement r(CPoint(0,0,0),CPoint(1,0,0),CPoint(0,1,0),CPoint(0,0,1));
	fixedConcreteElement l(CPoint(0,0,0),CPoint(-1,0,0),CPoint(0,1,0),CPoint(0,0,1));
	l.shift(geoShift);
	r.shift(geoShift);
	
	GenericElement tet(4);
	CVector natc;
	
	// Check element centers:
	check(tet.element_contains_point(r.getCenter(),r,natc));
	check(tet.element_contains_point(l.getCenter(),l,natc));
	check(!tet.element_contains_point(r.getCenter(),l,natc));
	check(!tet.element_contains_point(l.getCenter(),r,natc));
	
	// Check interpolation on centers:
	CVector interp;
	check(tet.interpolate(3,r,r.getCenter(),interp));
	check(equalVectors(interp,r.getCenter()-geoShift));
	check(tet.interpolate(3,l,l.getCenter(),interp));
	check(equalVectors(interp,l.getCenter()-geoShift));
	
	// Check boundaries:
	for (int i=0;i<4;i++) {
		check(tet.interpolate(3,r,r.getNodeLocation(i),interp));
		check(equalVectors(interp,CVector(r.getNodeData(i))));
		
		double frac=0.1+0.2*i;
		CPoint middle(0,frac,0); middle+=geoShift;
		CVector interpL,interpR;
		check(tet.interpolate(3,l,middle,interpL));
		check(tet.interpolate(3,r,middle,interpR));
		check(equalVectors(interpL,interpR));
		CVector theoryValue=(1-frac)*CVector(r.getNodeData(0))+
		                       frac*CVector(r.getNodeData(2));
		check(equalVectors(interpL,theoryValue));
	}
	
	printf("All tests passed.\n");
	return 0;
}
