/*
Simple, basic data types for Collision detection.

Orion Sky Lawlor, olawlor@acm.org, 2003/3/19
*/
#include <stdio.h>
#include "collide_util.h"
#include "charm.h" /* for CkAbort */


/************** CollideGrid3d ***********/
static void testMapping(CollideGrid3d &map,int axis,
	double origin,double size)
{
	int m1=map.world2grid(axis,rSeg1d(-1.0e20,origin-0.6*size)).getMax();
	int m2=map.world2grid(axis,rSeg1d(-1.0e20,origin-0.4*size)).getMax();
	int m3=map.world2grid(axis,rSeg1d(-1.0e20,origin-0.1*size)).getMax();
	int e1=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.1*size)).getMax();
	int e2=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.4*size)).getMax();
	int e3=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.6*size)).getMax();
	int e4=map.world2grid(axis,rSeg1d(-1.0e20,origin+0.9*size)).getMax();
	int p1=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.1*size)).getMax();
	int p2=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.4*size)).getMax();
	int p3=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.6*size)).getMax();
	int p4=map.world2grid(axis,rSeg1d(-1.0e20,origin+1.9*size)).getMax();
	if (m1!=m2 || m1!=m3) 
		CkAbort("CollideGrid3d::Grid initialization error (m)!\n");
	if (e1!=e2 || e1!=e3 || e1!=e4) 
		CkAbort("CollideGrid3d::Grid initialization error (e)!\n");
	if (p1!=p2 || p1!=p3 || p1!=p4) 
		CkAbort("CollideGrid3d::Grid initialization error (p)!\n");
}

void CollideGrid3d::init(const vector3d &Norigin,//Grid voxel corner 0,0,0
                const vector3d &desiredSize)//Size of each voxel
{
	origin=Norigin;
        for (int i=0;i<3;i++) {
#if COLLIDE_USE_FLOAT_HACK
                //Compute gridhack shift-- round grid size down 
                //  to nearest smaller power of two
                double s=(1<<20);
                while (s>(1.25*desiredSize[i])) s*=0.5;
                ((double*)sizes)[i]=s;
                hakShift[i]=(1.5*(1<<23)-0.5)*s-((double*)origin)[i];
                float o=(float)(hakShift[i]);
                hakStart[i]=*(int *)&o;
#else
                sizes[i]=desiredSize[i];
#endif
                ((double *)scales)[i]=1.0/((double *)sizes)[i];
                testMapping(*this,i,((double *)origin)[i],((double *)sizes)[i]);
        }
        
}

void CollideGrid3d::pup(PUP::er &p) {
	p|origin; p|scales; p|sizes;
#if COLLIDE_USE_FLOAT_HACK
	p(hakShift,3);
	p(hakStart,3);
#endif
}

void CollideGrid3d::print(const CollideLoc3d &g) {
#if !COLLIDE_USE_FLOAT_HACK
	const static int hakStart[3]={0,0,0};
#endif
	printf("%d,%d,%d",g.x-hakStart[0],g.y-hakStart[1],g.z-hakStart[2]);	
}

