/*  
Test out Mgc volume computation routines.
Orion Sky Lawlor, olawlor@acm.org, 2004/7/23
*/
#include <stdio.h>
#include <stdlib.h>
#include "cg3d.h"
#include "MgcIntr3DTetrTetr.h"
using namespace cg3d;

double test_vol_mgc(const PointSet3d *ps,const Tet3d &A,const Tet3d &B) 
{
        Mgc::Tetrahedron kT0,kT1;
        for(int i=0;i<4;i++){
                CkVector3d pts0 = ps->getPoint(A.getPointIndex(i));
                kT0[i] = Mgc::Vector3((double)pts0.x,(double)pts0.y,(double)pts0.z);
                CkVector3d pts1 = ps->getPoint(B.getPointIndex(i));
                kT1[i] = Mgc::Vector3((double)pts1.x,(double)pts1.y,(double)pts1.z);
        }
        Mgc::TetrahedronVolumeConsumer vol; 
        Mgc::FindIntersection(kT0,kT1,vol);
        return vol;
}

