/*  
Test out plane-based volume computation.
Orion Sky Lawlor, olawlor@acm.org, 2004/7/23
*/
#include <stdio.h>
#include <stdlib.h>
#include "cg3d.h"
#include "volume_planes.h"
using namespace cg3d;

/*************************** TEST DRIVER *****************/

double test_vol_planes(const PointSet3d *ps,const Tet3d &A,const Tet3d &B) 
{
	const int nDir=4;
	const int nDim=3;
	const int rowSize=nDim+1;
	double planes[2*nDir*rowSize];
	
	for (int dir=0;dir<nDir;dir++) {
		CkHalfspace3d Ah=A.getHalfspace(dir);
		CkHalfspace3d Bh=B.getHalfspace(dir);
		for (int i=0;i<nDim;i++) {
			planes[ dir      *rowSize+i]=-Ah.n[i];
			planes[(dir+nDir)*rowSize+i]=-Bh.n[i];
		}
		planes[ dir      *rowSize+nDim]=Ah.d;
		planes[(dir+nDir)*rowSize+nDim]=Bh.d;
	}
	
	double vol=computeVolumePlanes(planes,2*nDir);
	
	return vol;
}

