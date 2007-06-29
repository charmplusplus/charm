/**
 * Conservative, accurate serial cell-centered data transfer.
 * Orion Sky Lawlor, olawlor@acm.org, 2003/2/26
 */
#include <stdio.h>
#include <string.h> // For memmove
#include <stdlib.h> // for abort
#include <vector> //for std::vector 
#include "transfer.h" 
//#include "geom_util.h" 
#include "charm++.h" 
#include "MgcIntr3DTetrTetr.h"
using namespace Mgc;

/// Compute the volume shared by elements A and B, which must be tets.
double getSharedVolumeTets(const ConcreteElement &A,const ConcreteElement &B)
{
	Mgc::Tetrahedron kT0,kT1;
	for(int i=0;i<4;i++){
		CkVector3d pts0 = A.getNodeLocation(i);
		kT0[i] = Mgc::Vector3((double)pts0.x,(double)pts0.y,(double)pts0.z);
		CkVector3d pts1 = B.getNodeLocation(i);
		kT1[i] = Mgc::Vector3((double)pts1.x,(double)pts1.y,(double)pts1.z);
	}
	Mgc::TetrahedronVolumeConsumer vol;
	Mgc::FindIntersection(kT0,kT1,vol);
	double sumVol = vol;
	if(sumVol < 0){
		printf("volume less than zero!\n");
		abort();
	}
	return sumVol;
}


double getSharedArea(const ConcreteElement &A,const ConcreteElement &B) {
  double area=0.0;
  /*  vector<Vec3D> xpoly2;
  double tri[3][3];
  double pri[6][3];

  for (int i=0; i<6; i++) {
    CkVector3d pt = A.getNodeLocation(i);
    pri[i][0] = pt[0];
    pri[i][1] = pt[1];
    pri[i][2] = pt[2];
  }

  for (int i=0; i<3; i++) {
    CkVector3d pt = B.getNodeLocation(i);
    tri[i][0] = pt[0];
    tri[i][1] = pt[1];
    tri[i][2] = pt[2];
  }
  */
  // need to uncomment the line below when geom_util is functional
  //area = tri_prism_X(tri,pri,xpoly2);
  CkPrintf("datatransfer> transfer.C:getSharedArea: not yet functional\n");
  return area;
}

// Compute the volume shared by cell s of srcMesh
//   and cell d of destMesh.
double getSharedVolume(int s,const TetMesh &srcMesh,
	int d,const TetMesh &destMesh) 
{
	TetMeshElement se(s,srcMesh);
	TetMeshElement de(d,destMesh);
	return getSharedVolumeTets(se,de);
}


/**
 * Conservatively, accurately transfer 
 *   srcVals, tet-centered values on srcMesh
 * to
 *   destVals, tet-centered values on destMesh
 */
void transferCells(int valsPerTet,
	double *srcVals,const TetMesh &srcMesh,
	double *destVals,const TetMesh &destMesh)
{
	int d,nd=destMesh.getTets(); //Destination cells
	int s,ns=srcMesh.getTets(); //Source cells
	int v,nv=valsPerTet; //Values (for one cell)
	const int maxV=30;
	
	/* For each dest cell: */
	for (d=0;d<nd;d++) {
		
		//Accumulate volume-weighted-average destination values
		double destAccum[maxV]; 
		for (v=0;v<nv;v++) destAccum[v]=0.0;
		double destVolume=0; // Volume accumulator
		
		/* For each source cell: */
		for (s=0;s<ns;s++) {
			// Compute the volume shared by s and d:
			double shared=getSharedVolume(s,srcMesh,d,destMesh);
			if (shared<-1.0e-10) CkAbort("Negative volume shared region!");
			if (shared>0) {
				for (int v=0;v<nv;v++) 
					destAccum[v]+=shared*srcVals[s*nv+v];
				destVolume+=shared;
				
			}
		}
		
		/* Check the relative volume error, to make sure we've 
		   totally covered each destination cell. Checking precision
		   is low, since meshing tools often use single precision. */
		double trueVolume=destMesh.getTetVolume(d);
		double volErr=destVolume-trueVolume;
		double accumScale=1.0/destVolume; //Reverse volume weighting
		if (fabs(volErr*accumScale)>1.0e-10) {
			printf("WARNING: ------------- volume mismatch: dest tet %d -------------\n"
				" True volume %g, but total is only %g (err %g)\n",
				d,trueVolume,destVolume,volErr);
		}
		
		/* Copy the accumulated values into dest */
		for (v=0;v<nv;v++) 
			destVals[d*nv+v]=destAccum[v]*accumScale;
	}
}

